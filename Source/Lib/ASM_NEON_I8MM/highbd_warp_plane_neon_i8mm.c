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

#include "definitions.h"
#if CONFIG_ENABLE_HIGH_BIT_DEPTH && HAVE_NEON_I8MM
#include <arm_neon.h>
#include "common_dsp_rtcd.h"
#include "highbd_warp_plane_neon.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "transpose_neon.h"
#include "highbd_warp_plane_neon_kernels.h"

static AOM_FORCE_INLINE int16x8_t horizontal_filter_8x1_i8mm(uint8x16_t hi, uint8x16_t lo, int sx, int alpha) {
    // The lower two bits live at the top of their stored byte.
    lo = vshrq_n_u8(lo, 6);
    // Bias once per output, before scaling the upper-eight-bit contribution.
    const int32x4_t  bias    = vreinterpretq_s32_u64(vdupq_n_u64(1 << (10 + FILTER_BITS - 3)));
    const int        idx0    = ROUND_POWER_OF_TWO(sx + alpha * 0, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f0      = vld1q_s16(svt_aom_warped_filter[idx0]);
    const int        idx1    = ROUND_POWER_OF_TWO(sx + alpha * 1, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f1      = vld1q_s16(svt_aom_warped_filter[idx1]);
    const int        idx2    = ROUND_POWER_OF_TWO(sx + alpha * 2, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f2      = vld1q_s16(svt_aom_warped_filter[idx2]);
    const int        idx3    = ROUND_POWER_OF_TWO(sx + alpha * 3, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f3      = vld1q_s16(svt_aom_warped_filter[idx3]);
    const int        idx4    = ROUND_POWER_OF_TWO(sx + alpha * 4, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f4      = vld1q_s16(svt_aom_warped_filter[idx4]);
    const int        idx5    = ROUND_POWER_OF_TWO(sx + alpha * 5, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f5      = vld1q_s16(svt_aom_warped_filter[idx5]);
    const int        idx6    = ROUND_POWER_OF_TWO(sx + alpha * 6, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f6      = vld1q_s16(svt_aom_warped_filter[idx6]);
    const int        idx7    = ROUND_POWER_OF_TWO(sx + alpha * 7, WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS;
    const int16x8_t  f7      = vld1q_s16(svt_aom_warped_filter[idx7]);
    const int8x16_t  f01     = vcombine_s8(vmovn_s16(f0), vmovn_s16(f1));
    const uint8_t    ix0[16] = {0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8};
    const uint8x16_t index0  = vld1q_u8(ix0);
    const uint8x16_t hi0     = vqtbl1q_u8(hi, index0);
    const uint8x16_t lo0     = vqtbl1q_u8(lo, index0);
    int32x4_t        sum0    = vusdotq_s32(bias, hi0, f01);
    sum0                     = vusdotq_s32(vshlq_n_s32(sum0, 2), lo0, f01);
    const int8x16_t  f23     = vcombine_s8(vmovn_s16(f2), vmovn_s16(f3));
    const uint8_t    ix1[16] = {2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10};
    const uint8x16_t index1  = vld1q_u8(ix1);
    const uint8x16_t hi1     = vqtbl1q_u8(hi, index1);
    const uint8x16_t lo1     = vqtbl1q_u8(lo, index1);
    int32x4_t        sum1    = vusdotq_s32(bias, hi1, f23);
    sum1                     = vusdotq_s32(vshlq_n_s32(sum1, 2), lo1, f23);
    const int8x16_t  f45     = vcombine_s8(vmovn_s16(f4), vmovn_s16(f5));
    const uint8_t    ix2[16] = {4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12};
    const uint8x16_t index2  = vld1q_u8(ix2);
    const uint8x16_t hi2     = vqtbl1q_u8(hi, index2);
    const uint8x16_t lo2     = vqtbl1q_u8(lo, index2);
    int32x4_t        sum2    = vusdotq_s32(bias, hi2, f45);
    sum2                     = vusdotq_s32(vshlq_n_s32(sum2, 2), lo2, f45);
    const int8x16_t  f67     = vcombine_s8(vmovn_s16(f6), vmovn_s16(f7));
    const uint8_t    ix3[16] = {6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14};
    const uint8x16_t index3  = vld1q_u8(ix3);
    const uint8x16_t hi3     = vqtbl1q_u8(hi, index3);
    const uint8x16_t lo3     = vqtbl1q_u8(lo, index3);
    int32x4_t        sum3    = vusdotq_s32(bias, hi3, f67);
    sum3                     = vusdotq_s32(vshlq_n_s32(sum3, 2), lo3, f67);
    const int32x4_t first    = vpaddq_s32(sum0, sum1);
    const int32x4_t last     = vpaddq_s32(sum2, sum3);
    return vcombine_s16(vrshrn_n_s32(first, ROUND0_BITS), vrshrn_n_s32(last, ROUND0_BITS));
}

static AOM_FORCE_INLINE void warp_affine_horizontal_i8mm(const uint8_t* ref8, const uint8_t* ref2, int width,
                                                         int height, int stride8, int stride2, int p_width,
                                                         int16_t alpha, int16_t beta, int iy4, int sx4, int ix4,
                                                         int16x8_t tmp[], int bd) {
    (void)p_width;
    assert(bd == 10);
    (void)bd;
    if (ix4 <= -7 || ix4 >= width + 6) {
        const int ix = ix4 <= -7 ? 0 : width - 1;
        for (int k = 0; k < 15; ++k) {
            const int iy    = clamp(iy4 + k - 7, 0, height - 1);
            const int value = (ref8[iy * stride8 + ix] << 2) | (ref2[iy * stride2 + ix] >> 6);
            tmp[k]          = vdupq_n_s16((1 << (10 + FILTER_BITS - ROUND0_BITS - 1)) +
                                 value * (1 << (FILTER_BITS - ROUND0_BITS)));
        }
        return;
    }
    const int            left        = 6 - ix4;
    const int            right       = ix4 + 8 - width;
    static const uint8_t indices[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const uint8x16_t     index       = vld1q_u8(indices);
#define WARP_I8MM_ROWS(boundary, phase, shear)                                                              \
    do {                                                                                                    \
        for (int k = 0; k < 15; ++k) {                                                                      \
            const int  iy = clamp(iy4 + k - 7, 0, height - 1);                                              \
            uint8x16_t hi = vld1q_u8(ref8 + iy * stride8 + ix4 - 7);                                        \
            uint8x16_t lo = vld1q_u8(ref2 + iy * stride2 + ix4 - 7);                                        \
            if (boundary) {                                                                                 \
                if (left >= 0) {                                                                            \
                    const uint8x16_t mask = vcleq_u8(index, vdupq_n_u8(left));                              \
                    hi                    = vbslq_u8(mask, vdupq_n_u8(ref8[iy * stride8]), hi);             \
                    lo                    = vbslq_u8(mask, vdupq_n_u8(ref2[iy * stride2]), lo);             \
                }                                                                                           \
                if (right >= 0) {                                                                           \
                    const uint8x16_t mask = vcgeq_u8(index, vdupq_n_u8(15 - right));                        \
                    hi                    = vbslq_u8(mask, vdupq_n_u8(ref8[iy * stride8 + width - 1]), hi); \
                    lo                    = vbslq_u8(mask, vdupq_n_u8(ref2[iy * stride2 + width - 1]), lo); \
                }                                                                                           \
            }                                                                                               \
            tmp[k] = horizontal_filter_8x1_i8mm(hi, lo, phase, shear);                                      \
        }                                                                                                   \
    } while (0)
#define WARP_I8MM_SELECT(boundary)                                     \
    do {                                                               \
        if (!beta) {                                                   \
            if (!alpha)                                                \
                WARP_I8MM_ROWS(boundary, sx4, 0);                      \
            else                                                       \
                WARP_I8MM_ROWS(boundary, sx4, alpha);                  \
        } else {                                                       \
            if (!alpha)                                                \
                WARP_I8MM_ROWS(boundary, sx4 + beta * (k - 3), 0);     \
            else                                                       \
                WARP_I8MM_ROWS(boundary, sx4 + beta * (k - 3), alpha); \
        }                                                              \
    } while (0)
    if (left >= 0 || right >= 0) {
        WARP_I8MM_SELECT(true);
    } else {
        WARP_I8MM_SELECT(false);
    }
#undef WARP_I8MM_SELECT
#undef WARP_I8MM_ROWS
}

void svt_av1_highbd_warp_affine_neon_i8mm(const int32_t* mat, const uint8_t* ref8b, const uint8_t* ref2b, int width,
                                          int height, int stride8b, int stride2b, uint16_t* pred, int p_col, int p_row,
                                          int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y,
                                          int bd, ConvolveParams* conv_params, int16_t alpha, int16_t beta,
                                          int16_t gamma, int16_t delta) {
    // The encoder uses eight-wide tiles. Preserve the general NEON API for other inputs.
    if (bd != 10 || p_width == 4) {
        svt_av1_highbd_warp_affine_neon(mat,
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
                                        delta);
        return;
    }

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
                              warp_affine_horizontal_i8mm,
                              warp_affine_vertical_reuse);
}

#endif // CONFIG_ENABLE_HIGH_BIT_DEPTH && HAVE_NEON_I8MM
