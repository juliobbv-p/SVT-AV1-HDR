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

"""Decide whether the optimized SVT-only PSNR fast path applies to a config.

Exit code 0  -> fast path applies (all codecs are SVT-AV1, metrics == {psnr}).
Exit code 1  -> fast path does NOT apply; use the standard pipeline.

run_comparison.sh uses the exit code to dispatch.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.dirname(_THIS_DIR)
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from config_manager import ConfigManager  # noqa: E402


def fast_path_applies(config_path: str) -> bool:
    cm = ConfigManager(config_path=config_path)
    allowed_codecs = cm.get_codecs().get("allowed_codecs", [])
    allowed_metrics = cm.get_metrics().get("allowed_metrics", [])
    encoder_settings = cm.get_encoder_settings()

    if not allowed_codecs:
        return False
    # Only PSNR requested (the encoder can self-report this one).
    if set(allowed_metrics) != {"psnr"}:
        return False

    # Every requested codec must be an SVT-AV1 encoder.
    def is_svt(name: str) -> bool:
        for es in encoder_settings.values():
            if name in es:
                return "svtav1" in str(es[name].get("encoder", "")).lower()
        return False

    return all(is_svt(c) for c in allowed_codecs)


def main() -> int:
    if len(sys.argv) < 2:
        return 1
    return 0 if fast_path_applies(sys.argv[1]) else 1


if __name__ == "__main__":
    sys.exit(main())
