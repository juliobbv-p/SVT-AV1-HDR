/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <assert.h>
#include <stdbool.h>

#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "highbd_warp_plane_neon.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "transpose_neon.h"
#include "utility.h"
#include "warped_motion.h"

#include "highbd_warp_plane_neon_kernels.h"

void svt_av1_highbd_warp_affine_neon(const int32_t* mat, const uint8_t* ref8b, const uint8_t* ref2b, int width,
                                     int height, int stride8b, int stride2b, uint16_t* pred, int p_col, int p_row,
                                     int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y,
                                     int bd, ConvolveParams* conv_params, int16_t alpha, int16_t beta, int16_t gamma,
                                     int16_t delta) {
    highbd_warp_affine_common(mat,
                              ref8b,
                              ref2b,
                              width,
                              height,
                              stride8b,
                              stride2b,
                              pred,
                              p_col,
                              p_row,
                              p_width,
                              p_height,
                              p_stride,
                              subsampling_x,
                              subsampling_y,
                              bd,
                              conv_params,
                              alpha,
                              beta,
                              gamma,
                              delta,
                              warp_affine_horizontal,
                              warp_affine_vertical_reuse);
}
