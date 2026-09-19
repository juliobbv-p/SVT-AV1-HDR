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
#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "temporal_filtering_neon.h"
#include "temporal_filtering_neon_dotprod.h"

DECLARE_ALIGNED(16, static const uint8_t, mean_broadcast_tbl[16]) = {1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9};

static inline uint8x16_t avg8x8x2_neon_i8mm(const uint8x16_t s[8]) {
    const uint8x16_t scale = vdupq_n_u8(4);

    uint32x4_t sum = vmmlaq_u32(vdupq_n_u32(0), s[0], scale);
    sum            = vmmlaq_u32(sum, s[1], scale);
    sum            = vmmlaq_u32(sum, s[2], scale);
    sum            = vmmlaq_u32(sum, s[3], scale);
    sum            = vmmlaq_u32(sum, s[4], scale);
    sum            = vmmlaq_u32(sum, s[5], scale);
    sum            = vmmlaq_u32(sum, s[6], scale);
    sum            = vmmlaq_u32(sum, s[7], scale);

    return vreinterpretq_u8_u32(sum);
}

uint32_t svt_vmaf_compute_avg_mad_neon_i8mm(const uint8_t* src, int width, int height, int stride) {
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

            const uint8x16_t mean     = avg8x8x2_neon_i8mm(s);
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
