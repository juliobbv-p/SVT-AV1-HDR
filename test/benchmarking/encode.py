#!/usr/bin/env python3
# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import utils
from config_manager import ConfigManager
from tqdm import tqdm

# First non-flag argument is the config path; flags may follow it in any order.
_SUPPORTED_FLAGS = {"--svt-psnr-fast", "--time-l-counters", "--resume"}
_unknown_flags = [arg for arg in sys.argv[1:] if arg.startswith("--") and arg not in _SUPPORTED_FLAGS]
if _unknown_flags:
    raise ValueError(f"unsupported arguments: {', '.join(_unknown_flags)}")
_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
if len(_args) > 1:
    raise ValueError("expected at most one benchmark config path")
config_path = _args[0] if _args else None
# SVT-only PSNR fast mode: append --enable-stat-report 1 and read PSNR straight
# from the encoder stderr instead of a separate dav1d-decode + VMAF pass.
SVT_PSNR_FAST = "--svt-psnr-fast" in sys.argv
# Explicit macOS per-child counter mode for `/usr/bin/time -l` collection.
TIME_L_COUNTERS = "--time-l-counters" in sys.argv
# Preserve validated completed rows and queue only missing task keys after an
# interrupted long-running matrix.
RESUME = "--resume" in sys.argv
if SVT_PSNR_FAST and TIME_L_COUNTERS:
    raise ValueError("--svt-psnr-fast and --time-l-counters must be run as separate passes")
config_manager = ConfigManager(config_path=config_path)
from format_conversion import detect_source_format, generate_input_data_reformats

PATHS = config_manager.get_paths()
BINARIES = config_manager.get_binaries()
ENCODER_SETTINGS = config_manager.get_encoder_settings()
COMMON_SETTINGS = config_manager.get_common_settings()
SETTINGS = config_manager.get_settings()
PROFILER = config_manager.get_profiler()
PROFILE_DIR: str = PATHS.get("profile_dir", "")
if TIME_L_COUNTERS and PROFILER.get("enabled"):
    raise ValueError("--time-l-counters requires profiling to be disabled so /usr/bin/time wraps the encoder directly")

DRY_RUN_MODE = SETTINGS.get("dry_run", False)
if DRY_RUN_MODE and RESUME:
    raise ValueError("--resume cannot be combined with dry_run because dry-run rows are not resumable results")
MAX_PROC = SETTINGS.get("max_processes", 1)

SOURCE_DATA_DIR: str = PATHS["source_data_dir"]
INPUT_DIRS: Dict[str, str] = {
    "y4m": PATHS["source_y4m_dir"],
    "yuv": PATHS["source_yuv_dir"],
    "png": PATHS["source_png_dir"],
}

ENCODED_DIR: str = PATHS["encoded_dir"]
ENC_LOG_PATH: str = PATHS["enc_log_path"]
ENC_CSV_PATH: str = PATHS["enc_csv_path"]

enc_logger = utils.create_logger("enc_logger", ENC_LOG_PATH)


@dataclass
class EncodeTask:
    encoder_type: str
    encoder_name: str
    speed: str
    quality: int
    threads: int
    input_file: str


@dataclass
class EncodeResult:
    encoded_path: str
    encode_time: float
    input_size: int
    output_size: int
    # Optional Nsight Systems columns; default to empty / 0 when profiler disabled.
    nsys_report_path: str = ""
    cpu_sampling_top1_func: str = ""
    osrt_total_ms: float = 0.0
    # Optional macOS /usr/bin/time -l counter columns.
    real_time: Optional[float] = None
    user_time: Optional[float] = None
    system_time: Optional[float] = None
    instructions_retired: Optional[int] = None
    cycles: Optional[int] = None
    max_rss_bytes: Optional[int] = None
    # Optional PSNR columns, populated only in SVT-only PSNR fast mode (parsed
    # from the encoder's --enable-stat-report summary). None otherwise.
    psnr_y: Optional[float] = None
    psnr_cb: Optional[float] = None
    psnr_cr: Optional[float] = None


# Matches a PSNR value with its trailing dB unit, e.g. "40.69 dB".
_PSNR_DB_RE = re.compile(r"(-?\d+\.\d+)\s*dB")


def parse_svt_avg_psnr(
    stderr_text: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract the encoder's "Average PSNR (using per-frame PSNR)" Y/U/V values.

    The --enable-stat-report summary data line packs two PSNR groups split by
    '|':  <frames> <qp> Y dB U dB V dB | Yo dB Uo dB Vo dB | ...SSIM... | kbps
    The first segment holds the per-frame-averaged PSNR (the same quantity VMAF
    reports as its pooled-mean PSNR); the second is the MSE-pooled "Overall
    PSNR". We take the first for parity with the decode+VMAF pipeline.
    """
    for line in stderr_text.splitlines():
        if "dB" not in line or "|" not in line or "kbps" not in line:
            continue
        first_group = line.split("|", 1)[0]
        vals = _PSNR_DB_RE.findall(first_group)
        if len(vals) >= 3:
            return float(vals[0]), float(vals[1]), float(vals[2])
    return None, None, None


_profiler_warned = False


def build_profiler_prefix(task: "EncodeTask") -> Tuple[str, str]:
    """Build the nsys-profile shell prefix to inject in front of the encoder
    binary, plus the resulting .nsys-rep path. Returns ("", "") when profiling
    does not apply (disabled, codec not in apply_to, nsys missing, etc.) so the
    caller can format the command line uniformly."""
    global _profiler_warned

    if not PROFILER.get("enabled"):
        return "", ""
    if task.encoder_name not in PROFILER.get("apply_to", []):
        return "", ""
    if not PROFILE_DIR:
        if not _profiler_warned:
            enc_logger.warning(
                "profiler.enabled=true but `paths.profile_dir` is unset; skipping profile capture"
            )
            _profiler_warned = True
        return "", ""
    if not shutil.which("nsys"):
        if not _profiler_warned:
            enc_logger.warning(
                "profiler.enabled=true but `nsys` not on PATH; skipping profile capture for this run"
            )
            _profiler_warned = True
        return "", ""

    os.makedirs(PROFILE_DIR, exist_ok=True)
    stem = os.path.splitext(task.input_file)[0]
    profile_path = os.path.join(
        PROFILE_DIR,
        f"{task.encoder_name}_s{task.speed}_q{task.quality}_t{task.threads}_{stem}",
    )
    prefix = PROFILER["command"].format(profile_path=profile_path)
    return prefix, profile_path + ".nsys-rep"


def encode_file(task: EncodeTask, output_dir: str) -> EncodeResult:
    """Encode a single file with the specified parameters"""

    codec_settings = COMMON_SETTINGS[task.encoder_type]

    # re-implement monochrome if needed
    assert not codec_settings[task.encoder_name].get("monochrome", False)

    dir = INPUT_DIRS[codec_settings[task.encoder_name]["input_extension"]]
    input_path: str = os.path.join(dir, task.input_file)
    # passes enables more accurate runtime measurements. Counter mode must keep
    # one encoder child per job so the /usr/bin/time -l counters belong to that
    # exact encode, not an aggregate shell loop.
    passes = 1 if TIME_L_COUNTERS else codec_settings[task.encoder_name].get("passes", 1)

    filename_without_extension: str = os.path.splitext(task.input_file)[0]
    extension: str = codec_settings["encode_extension"]

    width, height, fps = utils.get_file_desc(filename_without_extension)
    kbps = utils.quality_to_kbps(width, height, fps, task.quality)

    encoded_file: str = f"{filename_without_extension}.{extension}"
    encoded_path: str = os.path.join(output_dir, encoded_file)

    enc_settings = ENCODER_SETTINGS[task.encoder_type][task.encoder_name]
    encoder_bin_name: str = enc_settings["encoder"]
    cfg_path: str = enc_settings.get("cfg_path", "")

    profiler_prefix, profile_report_path = build_profiler_prefix(task)

    orig_settings = enc_settings["command"]
    command = eval(f"f'{orig_settings}'", dict(
            binary_dir=BINARIES[encoder_bin_name],
            q=task.quality,
            kbps=kbps,
            speed=task.speed,
            nthreads=task.threads,
            input_path=input_path,
            width=width,
            height=height,
            fps=fps,
            cfg_path=cfg_path,
            output_path=encoded_path,
        ))
    if profiler_prefix:
        command = profiler_prefix + command

    if SVT_PSNR_FAST:
        # Make the encoder compute & print PSNR so we can skip decode + VMAF.
        command = f"{command} --enable-stat-report 1"

    # Get input file size
    input_size = os.path.getsize(input_path)
    enc_logger.info(command)

    if DRY_RUN_MODE:
        return EncodeResult(
            encoded_path=encoded_path,
            encode_time=0.0,
            output_size=0,
            input_size=input_size,
        )

    try:
        counter_metrics = None
        if TIME_L_COUNTERS:
            counter_metrics, stderr_text = utils.get_cmd_times(
                command, passes, return_stderr=True, time_mode="macos_time_l"
            )
            encode_time = counter_metrics["cpu_time"]
        elif SVT_PSNR_FAST:
            encode_time, stderr_text = utils.get_cmd_times(
                command, passes, return_stderr=True
            )
        else:
            encode_time = utils.get_cmd_times(command, passes)

        # Get output file size
        output_size = os.path.getsize(encoded_path)

        result = EncodeResult(
            encoded_path=encoded_path,
            encode_time=encode_time,
            output_size=output_size,
            input_size=input_size,
            nsys_report_path=profile_report_path,
        )
        if counter_metrics is not None:
            result.real_time = counter_metrics["real_time"]
            result.user_time = counter_metrics["user_time"]
            result.system_time = counter_metrics["system_time"]
            result.instructions_retired = counter_metrics["instructions_retired"]
            result.cycles = counter_metrics["cycles"]
            result.max_rss_bytes = counter_metrics["max_rss_bytes"]
        if SVT_PSNR_FAST:
            result.psnr_y, result.psnr_cb, result.psnr_cr = parse_svt_avg_psnr(
                stderr_text
            )
            if result.psnr_y is None:
                enc_logger.warning(
                    f"No PSNR parsed for {task.encoder_name} s{task.speed} "
                    f"q{task.quality} {task.input_file}; encoder summary missing?"
                )
        if profile_report_path and os.path.exists(profile_report_path):
            stats = utils.collect_nsys_stats(profile_report_path)
            result.cpu_sampling_top1_func = stats.get("cpu_sampling_top1_func", "")
            result.osrt_total_ms = stats.get("osrt_total_ms", 0.0)
        return result
    except subprocess.CalledProcessError as e:
        if "Signals.SIGINT" in str(e):
            raise KeyboardInterrupt
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr
        enc_logger.exception(f"Error: {stderr}")
        raise RuntimeError("Encoding failed") from e
    except KeyError as e:
        enc_logger.exception(f"Configuration error: {e}")
        raise ValueError("Invalid configuration") from e


def get_input_files(encoder_type: str, encoder_name: str) -> List[str]:
    codec_settings = COMMON_SETTINGS[encoder_type]

    # re-implement monochrome if needed
    assert not codec_settings[encoder_name].get("monochrome", False)

    ext: str = codec_settings[encoder_name]["input_extension"]
    return [x for x in os.listdir(INPUT_DIRS[ext]) if x.endswith(ext)]


def get_quality_values(encoder_type: str, encoder_name: str) -> List[int]:
    dir_format: Dict[str, Any] = COMMON_SETTINGS[encoder_type].get("dir_format", {})
    quality_param: str = dir_format.get("quality_param", "q")
    quality_values_key: str = f"{quality_param}_values"
    quality_values: List[int] = COMMON_SETTINGS[encoder_type][encoder_name].get(
        quality_values_key
    )
    return quality_values


def get_output_dir(task: EncodeTask) -> str:
    base_dir: str = ENCODED_DIR

    dir_format = COMMON_SETTINGS[task.encoder_type].get("dir_format", {})

    speed_suffix = dir_format.get("speed_suffix", "")
    if speed_suffix:
        name_suffix = speed_suffix.format(speed=task.speed)
    else:
        name_suffix = ""

    quality_param = dir_format.get("quality_param", "q")
    quality_str = f"{quality_param}{task.quality}"

    base_output_dir: str = os.path.join(base_dir, f"{task.encoder_name}{name_suffix}")
    output_dir: str = os.path.join(base_output_dir, f"{quality_str}_t{task.threads}")
    return output_dir


def execute_encode_job(task: EncodeTask) -> Tuple[EncodeTask, EncodeResult]:
    """Execute a single encode job and return the result"""
    output_dir = get_output_dir(task)

    if not DRY_RUN_MODE:
        os.makedirs(output_dir, exist_ok=True)

    return task, encode_file(task, output_dir)


def create_encode_jobs() -> List[EncodeTask]:
    """Create all encode jobs for all combinations of parameters"""
    jobs = []

    for encoder_type, encoder_settings in ENCODER_SETTINGS.items():
        for encoder_name, encoder_config in encoder_settings.items():
            if encoder_name not in config_manager.get_codecs()["allowed_codecs"]:
                continue

            settings = COMMON_SETTINGS[encoder_type][encoder_name]
            speed_values = settings["speed_values"]
            quality_values = get_quality_values(encoder_type, encoder_name)
            nthreads = encoder_config["nthreads"]
            input_files = get_input_files(encoder_type, encoder_name)

            for speed in speed_values:
                for quality in quality_values:
                    for nthread in nthreads:
                        for input_file in input_files:
                            jobs.append(
                                EncodeTask(
                                    encoder_type=encoder_type,
                                    encoder_name=encoder_name,
                                    speed=speed,
                                    quality=quality,
                                    input_file=input_file,
                                    threads=nthread,
                                )
                            )

    return jobs


def task_key(task: EncodeTask) -> Tuple[str, str, str, int, int, str]:
    return (
        task.encoder_type,
        task.encoder_name,
        str(task.speed),
        int(task.quality),
        int(task.threads),
        task.input_file,
    )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_signature(jobs: List[EncodeTask]) -> Dict[str, Any]:
    binary_names = {
        ENCODER_SETTINGS[job.encoder_type][job.encoder_name]["encoder"]
        for job in jobs
    }
    binaries = {}
    for name in sorted(binary_names):
        configured_path = BINARIES[name]
        resolved_path = shutil.which(configured_path) or os.path.realpath(configured_path)
        binaries[name] = {
            "path": resolved_path,
            "sha256": file_sha256(resolved_path),
        }

    input_paths = set()
    auxiliary_paths = set()
    for job in jobs:
        codec_settings = COMMON_SETTINGS[job.encoder_type]
        extension = codec_settings[job.encoder_name]["input_extension"]
        input_paths.add(os.path.realpath(os.path.join(INPUT_DIRS[extension], job.input_file)))
        cfg_path = ENCODER_SETTINGS[job.encoder_type][job.encoder_name].get("cfg_path", "")
        if cfg_path:
            auxiliary_paths.add(os.path.realpath(cfg_path))

    return {
        "schema": 1,
        "measurement_mode": (
            "svt_psnr_fast" if SVT_PSNR_FAST else "time_l_counters" if TIME_L_COUNTERS else "default"
        ),
        "config": config_manager.config,
        "binaries": binaries,
        "inputs": {path: file_sha256(path) for path in sorted(input_paths)},
        "auxiliary_files": {path: file_sha256(path) for path in sorted(auxiliary_paths)},
    }


def validate_or_write_run_signature(signature_path: str, resume_existing: bool, signature: Dict[str, Any]) -> None:
    if resume_existing:
        if not os.path.isfile(signature_path):
            raise ValueError(f"resume signature is missing: {signature_path}")
        try:
            with open(signature_path, "r", encoding="utf-8") as file:
                existing = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"resume signature is invalid: {signature_path}") from error
        if existing != signature:
            raise ValueError("resume signature does not match the current measurement mode, config, or binaries")
        return

    temporary_path = f"{signature_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(signature, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, signature_path)


def load_completed_task_keys(csv_path: str) -> set[Tuple[str, str, str, int, int, str]]:
    completed = pd.read_csv(csv_path)
    key_columns = [
        "encoder_type",
        "encoder_name",
        "speed",
        "quality",
        "threads",
        "input_file",
    ]
    missing = [column for column in key_columns if column not in completed.columns]
    if missing:
        raise ValueError(f"resume CSV is missing task key columns: {', '.join(missing)}")
    if completed[key_columns].isna().any().any():
        raise ValueError("resume CSV contains blank task key values")
    if "encoded_path" not in completed.columns:
        raise ValueError("resume CSV is missing required result column: encoded_path")

    required_positive = ["output_size"]
    if TIME_L_COUNTERS:
        required_positive.extend(["instructions_retired", "cycles", "max_rss_bytes"])
    if SVT_PSNR_FAST:
        required_positive.extend(["psnr_y", "psnr_cb", "psnr_cr"])
    for column in required_positive:
        if column not in completed.columns:
            raise ValueError(f"resume CSV is missing required result column: {column}")
        values = pd.to_numeric(completed[column], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise ValueError(f"resume CSV contains invalid {column} values")
    for column in ("quality", "threads"):
        values = pd.to_numeric(completed[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"resume CSV contains non-integral {column} values")

    for row in completed[["encoded_path", "output_size"]].itertuples(index=False):
        if not os.path.isfile(row.encoded_path):
            raise ValueError(f"resume artifact is missing: {row.encoded_path}")
        if os.path.getsize(row.encoded_path) != int(row.output_size):
            raise ValueError(f"resume artifact size does not match CSV: {row.encoded_path}")

    duplicate = completed.duplicated(key_columns, keep=False)
    if duplicate.any():
        raise ValueError(f"resume CSV contains duplicate task keys:\n{completed.loc[duplicate, key_columns].head(10)}")
    return {
        (
            str(row.encoder_type),
            str(row.encoder_name),
            str(row.speed),
            int(row.quality),
            int(row.threads),
            str(row.input_file),
        )
        for row in completed[key_columns].itertuples(index=False)
    }


def main() -> None:
    """Main function to execute all encode jobs and log results"""

    os.makedirs(os.path.dirname(ENC_CSV_PATH), exist_ok=True)

    input_format = detect_source_format(SOURCE_DATA_DIR)
    generate_input_data_reformats(input_format, SOURCE_DATA_DIR, INPUT_DIRS, enc_logger)

    # Create all encode jobs before validating resume identity so only binaries
    # used by this matrix are fingerprinted.
    jobs = create_encode_jobs()
    total_jobs = len(jobs)

    resume_existing = RESUME and os.path.isfile(ENC_CSV_PATH) and os.path.getsize(ENC_CSV_PATH) > 0
    if DRY_RUN_MODE:
        enc_logger.info("#" + "=" * 59)
        enc_logger.info("# DRY-RUN MODE: Commands will be logged but not executed")
        enc_logger.info("#" + "=" * 59)
    else:
        signature_path = f"{ENC_CSV_PATH}.resume.json"
        signature = run_signature(jobs)
        if resume_existing:
            validate_or_write_run_signature(signature_path, True, signature)
        else:
            utils.clean_directory(ENCODED_DIR)
            if os.path.exists(ENC_CSV_PATH):
                os.remove(ENC_CSV_PATH)
            validate_or_write_run_signature(signature_path, False, signature)

    if resume_existing:
        completed_keys = load_completed_task_keys(ENC_CSV_PATH)
        configured_keys = {task_key(job) for job in jobs}
        unknown_keys = completed_keys - configured_keys
        if unknown_keys:
            raise ValueError(f"resume CSV contains {len(unknown_keys)} task keys outside the current configuration")
        jobs = [job for job in jobs if task_key(job) not in completed_keys]
        enc_logger.info(f"Resuming {len(jobs)} missing jobs; preserving {len(completed_keys)} completed rows")
    enc_logger.info(f"Created {total_jobs} encode jobs; {len(jobs)} queued")

    if not jobs:
        enc_logger.info("No encode jobs to run")
        return

    print(f"Output CSV: {ENC_CSV_PATH}")
    need_csv_header = not resume_existing

    # be nice when using multiprocessing
    os.nice(10)
    max_workers = utils.get_max_workers(MAX_PROC)

    failed_jobs = 0

    # Execute jobs in threadpool with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        def signal_handler(sig, frame):
            print("Received Ctrl-C, shutting down...")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        futures = [executor.submit(execute_encode_job, job) for job in jobs]

        for future in tqdm(
            as_completed(futures), total=len(jobs), desc="Encoding files"
        ):
            try:
                job, result = future.result()
                pd.DataFrame([{**asdict(job), **asdict(result)}]).to_csv(
                    ENC_CSV_PATH,
                    index=False,
                    header=need_csv_header,
                    mode="w" if need_csv_header else "a",
                )
                need_csv_header = False
                if result.output_size > 0:
                    compression_ratio = result.input_size / result.output_size
                else:
                    compression_ratio = 0.0
                log_msg = (
                    f"Completed: {job.encoder_name} speed={job.speed} "
                    f"quality={job.quality} threads={job.threads} "
                    f"file={job.input_file} "
                    f"input_size={result.input_size} bytes "
                    f"output_size={result.output_size} bytes "
                    f"compression_ratio={compression_ratio:.3f} -> ok"
                )
            except Exception as e:
                failed_jobs += 1
                log_msg = f"Failed job -> {e}"

            enc_logger.info(log_msg)

    # Log summary statistics
    enc_logger.info("")
    enc_logger.info("Encoding Summary:")
    enc_logger.info(f"Total jobs: {len(jobs)}")
    enc_logger.info(f"Failed jobs: {failed_jobs}")
    enc_logger.info(f"Results saved to: {ENC_CSV_PATH}")
    if TIME_L_COUNTERS and failed_jobs:
        raise RuntimeError(f"Counter pass failed for {failed_jobs} encode job(s)")


if __name__ == "__main__":
    main()
