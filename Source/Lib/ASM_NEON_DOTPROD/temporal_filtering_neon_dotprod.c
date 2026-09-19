/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include <assert.h>
#include <math.h>
#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "temporal_filtering_neon.h"
#include "temporal_filtering_neon_dotprod.h"

DECLARE_ALIGNED(16, static const uint8_t, mean_broadcast_tbl[16]) = {1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9};

static inline uint64x2_t avg8x8x2_neon_dotprod(const uint8x16_t s[8]) {
    const uint8x16_t scale = vdupq_n_u8(4);

    uint32x4_t sum0 = vdotq_u32(vdupq_n_u32(0), s[0], scale);
    uint32x4_t sum1 = vdotq_u32(vdupq_n_u32(0), s[1], scale);
    sum0            = vdotq_u32(sum0, s[2], scale);
    sum1            = vdotq_u32(sum1, s[3], scale);
    sum0            = vdotq_u32(sum0, s[4], scale);
    sum1            = vdotq_u32(sum1, s[5], scale);
    sum0            = vdotq_u32(sum0, s[6], scale);
    sum1            = vdotq_u32(sum1, s[7], scale);

    sum0 = vaddq_u32(sum0, sum1);

    return vpaddlq_u32(sum0);
}

uint32_t svt_vmaf_compute_avg_mad_neon_dotprod(const uint8_t* src, int width, int height, int stride) {
    assert(width >= 8 && width % 8 == 0 && "width must be at least 8 and multiple of 8");
    assert(height >= 8 && height % 8 == 0 && "height must be at least 8 and multiple of 8");

    const uint64_t block_count = (height * width) >> 6;

    const uint8x16_t broadcast_tbl = vld1q_u8(mean_broadcast_tbl);

    uint64_t total_activity = 0;
    int      by             = 0;
    do {
        uint32x4_t activity_vec0 = vdupq_n_u32(0);
        uint32x4_t activity_vec1 = vdupq_n_u32(0);
        int        bx            = 0;
        for (; bx + 16 <= width; bx += 16) {
            uint8x16_t s[8];
            load_u8_16x8(src + by * stride + bx, stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);

            const uint8x16_t mean     = vreinterpretq_u8_u64(avg8x8x2_neon_dotprod(s));
            const uint8x16_t mean_vec = vqtbl1q_u8(mean, broadcast_tbl);

            mad8x8x2_neon_dotprod(s, mean_vec, &activity_vec0, &activity_vec1);
        }

        if (bx + 8 <= width) {
            uint8x8_t s[8];
            load_u8_8x8(src + by * stride + bx, stride, &s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6], &s[7]);

            uint8x16_t mean = vdupq_n_u8(avg8x8_neon(s));

            mad8x8_neon_dotprod(s, mean, &activity_vec0, &activity_vec1);
        }

        total_activity += vaddvq_u32(vaddq_u32(activity_vec0, activity_vec1));
        by += 8;
    } while (by + 8 <= height);

    return (uint32_t)(total_activity / (block_count * 64));
}

static inline void gradient_coherence_accumulate_row_8_neon_dotprod(uint8x8_t row_r, uint8x8_t row_l, uint8x8_t row_d,
                                                                    uint8x8_t row_u, uint32x2_t* acc_xx,
                                                                    uint32x2_t* acc_yy, int32x4_t* acc_xy) {
    const uint8x8_t gx_u8 = vabd_u8(row_r, row_l);
    const uint8x8_t gy_u8 = vabd_u8(row_d, row_u);
    *acc_xx               = vdot_u32(*acc_xx, gx_u8, gx_u8);
    *acc_yy               = vdot_u32(*acc_yy, gy_u8, gy_u8);

    const int16x8_t gx_s16 = vreinterpretq_s16_u16(vsubl_u8(row_r, row_l));
    const int16x8_t gy_s16 = vreinterpretq_s16_u16(vsubl_u8(row_d, row_u));
    *acc_xy                = vmlal_s16(*acc_xy, vget_low_s16(gx_s16), vget_low_s16(gy_s16));
    *acc_xy                = vmlal_s16(*acc_xy, vget_high_s16(gx_s16), vget_high_s16(gy_s16));
}

float svt_vmaf_compute_gradient_coherence_neon_dotprod(const uint8_t* src, int width, int height, int stride) {
    assert(width % 8 == 0 && "width must be multiple of 8");

    double weighted_coh = 0.0;
    double weight_sum   = 0.0;

    for (int by = 1; by < height - 1; by += 16) {
        const int y_end = (by + 16 < height - 1) ? by + 16 : height - 1;

        int bx = 1;
        for (; bx + 16 <= width - 1; bx += 16) {
            uint32x4_t acc_xx  = vdupq_n_u32(0);
            uint32x4_t acc_yy  = vdupq_n_u32(0);
            int32x4_t  acc_xy0 = vdupq_n_s32(0);
            int32x4_t  acc_xy1 = vdupq_n_s32(0);

            const uint8_t* row  = src + (size_t)by * stride;
            const uint8_t* up   = src + (size_t)(by - 1) * stride;
            const uint8_t* down = src + (size_t)(by + 1) * stride;

            int y = by;
            do {
                const uint8x16_t row_r = vld1q_u8(row + bx + 1);
                const uint8x16_t row_l = vld1q_u8(row + bx - 1);
                const uint8x16_t row_d = vld1q_u8(down + bx);
                const uint8x16_t row_u = vld1q_u8(up + bx);

                const uint8x16_t gx_u8 = vabdq_u8(row_r, row_l);
                const uint8x16_t gy_u8 = vabdq_u8(row_d, row_u);
                acc_xx                 = vdotq_u32(acc_xx, gx_u8, gx_u8);
                acc_yy                 = vdotq_u32(acc_yy, gy_u8, gy_u8);

                const int16x8_t gx_s16_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(row_r), vget_low_u8(row_l)));
                const int16x8_t gx_s16_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(row_r), vget_high_u8(row_l)));
                const int16x8_t gy_s16_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(row_d), vget_low_u8(row_u)));
                const int16x8_t gy_s16_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(row_d), vget_high_u8(row_u)));
                acc_xy0                   = vmlal_s16(acc_xy0, vget_low_s16(gx_s16_lo), vget_low_s16(gy_s16_lo));
                acc_xy1                   = vmlal_s16(acc_xy1, vget_high_s16(gx_s16_lo), vget_high_s16(gy_s16_lo));
                acc_xy0                   = vmlal_s16(acc_xy0, vget_low_s16(gx_s16_hi), vget_low_s16(gy_s16_hi));
                acc_xy1                   = vmlal_s16(acc_xy1, vget_high_s16(gx_s16_hi), vget_high_s16(gy_s16_hi));

                row += stride;
                up += stride;
                down += stride;
            } while (++y != y_end);

            const double xx = (double)vaddvq_u32(acc_xx);
            const double yy = (double)vaddvq_u32(acc_yy);
            const double xy = (double)(int64_t)vaddvq_s32(vaddq_s32(acc_xy0, acc_xy1));
            weighted_coh += sqrtf((float)((xx - yy) * (xx - yy) + 4.0 * xy * xy));
            weight_sum += xx + yy;
        }

        // Tail can be either 6 pixels or 8+6 pixels.
        uint32x2_t acc_xx = vdup_n_u32(0);
        uint32x2_t acc_yy = vdup_n_u32(0);
        int32x4_t  acc_xy = vdupq_n_s32(0);
        if (bx + 8 < width - 1) {
            const uint8_t* row  = src + (size_t)by * stride;
            const uint8_t* up   = src + (size_t)(by - 1) * stride;
            const uint8_t* down = src + (size_t)(by + 1) * stride;

            int y = by;
            do {
                const uint8x8_t row_r = vld1_u8(row + bx + 1);
                const uint8x8_t row_l = vld1_u8(row + bx - 1);
                const uint8x8_t row_d = vld1_u8(down + bx);
                const uint8x8_t row_u = vld1_u8(up + bx);

                gradient_coherence_accumulate_row_8_neon_dotprod(row_r, row_l, row_d, row_u, &acc_xx, &acc_yy, &acc_xy);

                row += stride;
                up += stride;
                down += stride;
            } while (++y != y_end);

            bx += 8;
        }
        if (bx < width - 1) {
            const uint8x8_t mask = vcreate_u8(0x0000FFFFFFFFFFFF);
            const uint8_t*  row  = src + (size_t)by * stride;
            const uint8_t*  up   = src + (size_t)(by - 1) * stride;
            const uint8_t*  down = src + (size_t)(by + 1) * stride;

            int y = by;
            do {
                const uint8x8_t row_r = vand_u8(vld1_u8(row + bx + 1), mask);
                const uint8x8_t row_l = vand_u8(vld1_u8(row + bx - 1), mask);
                const uint8x8_t row_d = vand_u8(vld1_u8(down + bx), mask);
                const uint8x8_t row_u = vand_u8(vld1_u8(up + bx), mask);

                gradient_coherence_accumulate_row_8_neon_dotprod(row_r, row_l, row_d, row_u, &acc_xx, &acc_yy, &acc_xy);

                row += stride;
                up += stride;
                down += stride;
            } while (++y != y_end);

            const double xx = (double)vaddv_u32(acc_xx);
            const double yy = (double)vaddv_u32(acc_yy);
            const double xy = (double)(int64_t)vaddvq_s32(acc_xy);
            weighted_coh += sqrtf((float)((xx - yy) * (xx - yy) + 4.0 * xy * xy));
            weight_sum += xx + yy;
        }
    }
    if (weight_sum <= 0.0) {
        return 1.0f;
    }
    return (float)(weighted_coh / weight_sum);
}
