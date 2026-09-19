#!/bin/bash
# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

DEF_CONFIG=configs/test_image_config.yaml
CONFIG=${1:-${DEF_CONFIG}}

conda activate codec_eval

# Optimized path: when every codec is SVT-AV1 and PSNR is the only metric, the
# encoder can report PSNR itself (--enable-stat-report 1), so we skip the
# expensive dav1d decode + VMAF measurement entirely.
if python3 scripts/detect_svt_psnr_fast.py "${CONFIG}"; then
    echo "############################################################"
    echo "# NOTE: SVT-AV1-only + PSNR config detected."
    echo "#       Using the optimized fast path: PSNR is read straight"
    echo "#       from the encoder (--enable-stat-report 1); the dav1d"
    echo "#       decode and VMAF quality steps are SKIPPED."
    echo "#       (encoder per-frame-averaged PSNR == VMAF pooled PSNR)"
    echo "#       Caveat: reported encode_time INCLUDES the in-encoder PSNR"
    echo "#       cost (~10-15% over a metric-free encode); use the full"
    echo "#       pipeline if you need clean encoder-speed numbers."
    echo "#       Run the standard pipeline manually if you need a"
    echo "#       decode-based cross-check."
    echo "############################################################"
    python3 encode.py "${CONFIG}" --svt-psnr-fast
    python3 summary.py "${CONFIG}" --svt_psnr_fast
else
    python3 encode.py "${CONFIG}"
    python3 decode_and_qm.py "${CONFIG}"
    python3 summary.py "${CONFIG}" --use_enc_qm_logs
fi
