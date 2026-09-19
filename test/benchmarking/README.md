# Test scripts for image and video codec comparison

# QUICK START
- `./setup.sh`
- `./run_comparison.sh configs/test_image_config.yaml`
- `./run_comparison.sh configs/test_video_config.yaml`

Outputs are in `~/benchmark/out`


# MORE INFO

### SETUP
Setup:
 `./setup.sh <benchmark_dir>`
This will install all the required libraries, setup a conda environment and activate it.


### CONFIG
Experiment setup should be provided through a config file. Example config files are provided under the configs directory.
In the config file generally the following fields only need to be modified:
`"root_dir", "out_dir", "tag", "dataset", "allowed_codecs", "allowed_metrics"`
For completeness, the following snippet shows a more comprehensive set of fields that might need modification, along with their explanation.

```
    placeholders:
        root_dir: "~/benchmark"                             # Points to the root directory of where the binary files and datasets are located
        out_dir: "~/benchmark/out"                          # Output directory for the results (encoded/decoded/summary results)
        tag: "AV2-CTC"                                      # Name used for output folders and log files

    dataset:
        source_dir: "{root_dir}/DataSet/AV2-CTC"            # Which dataset to use as input (will contain any of png/yuv/y4m files)
        tmp_dir: "{root_dir}/tmp"                           # Directory where different formats of the input dataset will be stored during (if input is png, then yuv and y4m will be generated, etc)

    codecs:
        allowed_codecs: ["jpegli", "svtav1"]                # List of codecs that should be used

    metrics:
        allowed_metrics: ["vmaf", "ssimulacra2", "ms_ssim"] # List of metrics to compute. PSNR is calculated via VMAF library
        allow_bdrate: true                                  # Set to true to enable BDRate metric computation by default
        anchor_encoder: "jpegli"                            # Encoder used to generate the reference for BDRate
        anchor_speed: 0                                     # Encoder speed for the reference BDRate computation
        aom_ctc_model: "v6.0"                               # AOM CTC model version

    settings:
        max_processes: 0                                    # 0 for auto scaling based on host configuration
        remove_decoded_files: true                          # Removes decoded files to reduce storage required to run benchmark
```


### RUN
Once you have the config file, you can run the experiment with:
 `./run_comparison.sh <config_file>`

This will generate all encodings using specified quality and speed values.
Then it will generate the decoded files and run the quality metric computation, storing the xml, ssimulacra2 results to file.
Lastly it'll generate the final report.

Encoding, decoding and summary maybe performed on separate machines, see summary.py for details.


### OPTIMIZED SVT-AV1-ONLY PSNR PATH

When every codec under test is an SVT-AV1 encoder **and** the only requested
metric is PSNR (`allowed_metrics: ["psnr"]`), `run_comparison.sh` automatically
switches to a faster path:

- The encoder is run with `--enable-stat-report 1` and PSNR is read straight
  from its summary output, so the dav1d **decode and VMAF steps are skipped**.
- The encoder's "Average PSNR (using per-frame PSNR)" is the same quantity VMAF
  reports as its pooled-mean PSNR, so BD-rate results match the full pipeline
  (verified to within ~0.005 dB, i.e. the encoder's 2-decimal print rounding).
- BD-rate computation and CSV/PDF reporting are unchanged (reused as-is).

Detection and dispatch are automatic; a NOTE banner is printed when the fast
path is used. To run it manually:

```
python3 encode.py  <config> --svt-psnr-fast
python3 summary.py <config> --svt_psnr_fast
```

Notes:
- Set `anchor_encoder` in the config to one of the SVT codecs present in the run
  (the standard config anchor is used for BD-rate).
- The reported `encode_time` **includes** the in-encoder PSNR cost
  (~10-15% over a metric-free encode). Use the standard pipeline if you need
  clean encoder-speed numbers.
- Detection logic lives in `scripts/detect_svt_psnr_fast.py`.
