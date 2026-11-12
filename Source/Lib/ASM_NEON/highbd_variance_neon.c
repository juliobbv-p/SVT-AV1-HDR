/*
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"

uint32_t svt_aom_highbd_mse16x16_neon(const uint8_t* src_ptr, int src_stride, const uint8_t* ref_ptr, int ref_stride) {
    uint16_t* src = CONVERT_TO_SHORTPTR(src_ptr);
    uint16_t* ref = CONVERT_TO_SHORTPTR(ref_ptr);

    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    int i = 16;
    do {
        int j = 0;
        do {
            uint16x8_t s = vld1q_u16(src + j);
            uint16x8_t r = vld1q_u16(ref + j);

            uint16x8_t diff = vabdq_u16(s, r);

            sse_u32[0] = vmlal_u16(sse_u32[0], vget_low_u16(diff), vget_low_u16(diff));
            sse_u32[1] = vmlal_u16(sse_u32[1], vget_high_u16(diff), vget_high_u16(diff));

            j += 8;
        } while (j < 16);

        src += src_stride;
        ref += ref_stride;
    } while (--i != 0);

    return vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}

static inline void highbd_variance_4xh_neon(const uint16_t* src_ptr, int src_stride, const uint16_t* ref_ptr,
                                            int ref_stride, int h, uint64_t* sse, int64_t* sum) {
    int16x8_t sum_s16 = vdupq_n_s16(0);
    int32x4_t sse_s32 = vdupq_n_s32(0);

    int i = h;
    do {
        const uint16x8_t s = load_u16_4x2(src_ptr, src_stride);
        const uint16x8_t r = load_u16_4x2(ref_ptr, ref_stride);

        int16x8_t diff = vreinterpretq_s16_u16(vsubq_u16(s, r));
        sum_s16        = vaddq_s16(sum_s16, diff);

        sse_s32 = vmlal_s16(sse_s32, vget_low_s16(diff), vget_low_s16(diff));
        sse_s32 = vmlal_s16(sse_s32, vget_high_s16(diff), vget_high_s16(diff));

        src_ptr += 2 * src_stride;
        ref_ptr += 2 * ref_stride;
        i -= 2;
    } while (i != 0);

    *sum = vaddlvq_s16(sum_s16);
    *sse = vaddvq_s32(sse_s32);
}

// One 8-sample vector: its differences go to one 16-bit sum and two squared-error chains.
#define HBD_VAR_STEP8(sp, rp, sum, e0, e1)                                                   \
    do {                                                                                     \
        const int16x8_t d_ = vreinterpretq_s16_u16(vsubq_u16(vld1q_u16(sp), vld1q_u16(rp))); \
        sum                = vaddq_s16(sum, d_);                                             \
        e0                 = vmlal_s16(e0, vget_low_s16(d_), vget_low_s16(d_));              \
        e1                 = vmlal_high_s16(e1, d_, d_);                                     \
    } while (0)

// Widths 8 and up (heights a multiple of 4 for width 8 and of 2 for width 16). Every step takes four 8-sample
// vectors - 32 samples of a row, two rows of a 16-wide block or four rows of an 8-wide one - each with its own
// sum and its own pair of squared-error chains, so that the multiply-accumulate latency of a chain is spread over
// four vectors instead of two.
static inline void highbd_variance_large_neon(const uint16_t* src_ptr, int src_stride, const uint16_t* ref_ptr,
                                              int ref_stride, int w, int h, uint64_t* sse, int64_t* sum) {
    int32x4_t sum_s32 = vdupq_n_s32(0);
    int32x4_t e0 = vdupq_n_s32(0), e1 = e0, e2 = e0, e3 = e0, e4 = e0, e5 = e0, e6 = e0, e7 = e0;
    // offsets of the four vectors of a step and the rows a step covers
    const int rows_per_step = w == 8 ? 4 : w == 16 ? 2 : 1;
    const int so1 = w == 8 ? src_stride : 8, so2 = w == 8 ? 2 * src_stride : w == 16 ? src_stride : 16;
    const int so3 = w == 8 ? 3 * src_stride : w == 16 ? src_stride + 8 : 24;
    const int ro1 = w == 8 ? ref_stride : 8, ro2 = w == 8 ? 2 * ref_stride : w == 16 ? ref_stride : 16;
    const int ro3   = w == 8 ? 3 * ref_stride : w == 16 ? ref_stride + 8 : 24;
    const int row_w = w < 32 ? 32 : w; // samples of a row handled by the inner loop, in steps of 32
    assert(h % rows_per_step == 0);

    // Each signed 16-bit sum lane receives at most 32 differences between widenings: 32 * 1023 = 32736.
    const int steps_per_row  = row_w / 32;
    const int rows_per_chunk = (32 / steps_per_row) * rows_per_step;
    for (int row = 0; row < h; row += rows_per_chunk) {
        int16x8_t sum0 = vdupq_n_s16(0), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        const int rows = AOMMIN(h - row, rows_per_chunk);
        for (int i = 0; i < rows; i += rows_per_step) {
            for (int j = 0; j < row_w; j += 32) {
                HBD_VAR_STEP8(src_ptr + j, ref_ptr + j, sum0, e0, e1);
                HBD_VAR_STEP8(src_ptr + j + so1, ref_ptr + j + ro1, sum1, e2, e3);
                HBD_VAR_STEP8(src_ptr + j + so2, ref_ptr + j + ro2, sum2, e4, e5);
                HBD_VAR_STEP8(src_ptr + j + so3, ref_ptr + j + ro3, sum3, e6, e7);
            }
            src_ptr += rows_per_step * src_stride;
            ref_ptr += rows_per_step * ref_stride;
        }
        sum_s32 = vpadalq_s16(sum_s32, sum0);
        sum_s32 = vpadalq_s16(sum_s32, sum1);
        sum_s32 = vpadalq_s16(sum_s32, sum2);
        sum_s32 = vpadalq_s16(sum_s32, sum3);
    }

    *sum = vaddvq_s32(sum_s32);
    // Even at 128x128 each unsigned lane holds at most 2048 * 1023^2, below UINT32_MAX; the eight chains are
    // widened before they are added together.
    const uint64x2_t t0 = vpaddlq_u32(vreinterpretq_u32_s32(vaddq_s32(e0, e1)));
    const uint64x2_t t1 = vpaddlq_u32(vreinterpretq_u32_s32(vaddq_s32(e2, e3)));
    const uint64x2_t t2 = vpaddlq_u32(vreinterpretq_u32_s32(vaddq_s32(e4, e5)));
    const uint64x2_t t3 = vpaddlq_u32(vreinterpretq_u32_s32(vaddq_s32(e6, e7)));
    *sse                = vaddvq_u64(vaddq_u64(vaddq_u64(t0, t1), vaddq_u64(t2, t3)));
}

static inline void highbd_variance_small_neon(const uint16_t* src_ptr, int src_stride, const uint16_t* ref_ptr,
                                              int ref_stride, int w, int h, uint64_t* sse, int64_t* sum) {
    int32x4_t sum_s32    = vdupq_n_s32(0);
    int32x4_t sse_s32[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};

    int i = h;
    do {
        int j = 0;
        do {
            const uint16x8_t s = vld1q_u16(src_ptr + j);
            const uint16x8_t r = vld1q_u16(ref_ptr + j);

            const int16x8_t diff = vreinterpretq_s16_u16(vsubq_u16(s, r));
            sum_s32              = vpadalq_s16(sum_s32, diff);

            sse_s32[0] = vmlal_s16(sse_s32[0], vget_low_s16(diff), vget_low_s16(diff));
            sse_s32[1] = vmlal_s16(sse_s32[1], vget_high_s16(diff), vget_high_s16(diff));

            j += 8;
        } while (j < w);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--i != 0);

    *sum = vaddvq_s32(sum_s32);
    *sse = vaddlvq_u32(vreinterpretq_u32_s32(vaddq_s32(sse_s32[0], sse_s32[1])));
}

static inline void highbd_variance_8xh_neon(const uint16_t* src, int src_stride, const uint16_t* ref, int ref_stride,
                                            int h, uint64_t* sse, int64_t* sum) {
    if (h <= 8) {
        highbd_variance_small_neon(src, src_stride, ref, ref_stride, 8, h, sse, sum);
    } else {
        highbd_variance_large_neon(src, src_stride, ref, ref_stride, 8, h, sse, sum);
    }
}

static inline void highbd_variance_16xh_neon(const uint16_t* src, int src_stride, const uint16_t* ref, int ref_stride,
                                             int h, uint64_t* sse, int64_t* sum) {
    highbd_variance_large_neon(src, src_stride, ref, ref_stride, 16, h, sse, sum);
}

static inline void highbd_variance_32xh_neon(const uint16_t* src, int src_stride, const uint16_t* ref, int ref_stride,
                                             int h, uint64_t* sse, int64_t* sum) {
    highbd_variance_large_neon(src, src_stride, ref, ref_stride, 32, h, sse, sum);
}

static inline void highbd_variance_64xh_neon(const uint16_t* src, int src_stride, const uint16_t* ref, int ref_stride,
                                             int h, uint64_t* sse, int64_t* sum) {
    highbd_variance_large_neon(src, src_stride, ref, ref_stride, 64, h, sse, sum);
}

static inline void highbd_variance_128xh_neon(const uint16_t* src, int src_stride, const uint16_t* ref, int ref_stride,
                                              int h, uint64_t* sse, int64_t* sum) {
    highbd_variance_large_neon(src, src_stride, ref, ref_stride, 128, h, sse, sum);
}

#define HBD_VARIANCE_WXH_10_NEON(w, h)                                                                   \
    uint32_t svt_aom_highbd_10_variance##w##x##h##_neon(                                                 \
        const uint8_t* src_ptr, int src_stride, const uint8_t* ref_ptr, int ref_stride, uint32_t* sse) { \
        uint64_t  sse_long = 0;                                                                          \
        int64_t   sum_long = 0;                                                                          \
        uint16_t* src      = CONVERT_TO_SHORTPTR(src_ptr);                                               \
        uint16_t* ref      = CONVERT_TO_SHORTPTR(ref_ptr);                                               \
        highbd_variance_##w##xh_neon(src, src_stride, ref, ref_stride, h, &sse_long, &sum_long);         \
        *sse        = (uint32_t)ROUND_POWER_OF_TWO(sse_long, 4);                                         \
        int     sum = (int)ROUND_POWER_OF_TWO(sum_long, 2);                                              \
        int64_t var = (int64_t)*sse - (((int64_t)sum * sum) / (w * h));                                  \
        return var >= 0 ? (uint32_t)var : 0;                                                             \
    }

HBD_VARIANCE_WXH_10_NEON(4, 4)
HBD_VARIANCE_WXH_10_NEON(4, 8)
HBD_VARIANCE_WXH_10_NEON(4, 16)

HBD_VARIANCE_WXH_10_NEON(8, 4)
HBD_VARIANCE_WXH_10_NEON(8, 8)
HBD_VARIANCE_WXH_10_NEON(8, 16)
HBD_VARIANCE_WXH_10_NEON(8, 32)

HBD_VARIANCE_WXH_10_NEON(16, 4)
HBD_VARIANCE_WXH_10_NEON(16, 8)
HBD_VARIANCE_WXH_10_NEON(16, 16)
HBD_VARIANCE_WXH_10_NEON(16, 32)
HBD_VARIANCE_WXH_10_NEON(16, 64)

HBD_VARIANCE_WXH_10_NEON(32, 8)
HBD_VARIANCE_WXH_10_NEON(32, 16)
HBD_VARIANCE_WXH_10_NEON(32, 32)
HBD_VARIANCE_WXH_10_NEON(32, 64)

HBD_VARIANCE_WXH_10_NEON(64, 16)
HBD_VARIANCE_WXH_10_NEON(64, 32)
HBD_VARIANCE_WXH_10_NEON(64, 64)
HBD_VARIANCE_WXH_10_NEON(64, 128)

HBD_VARIANCE_WXH_10_NEON(128, 64)
HBD_VARIANCE_WXH_10_NEON(128, 128)

// bit-exact with svt_aom_variance_highbd_c (uint32 sse wraps mod 2^32).
// Assumes w is a multiple of 4 and |a - b| fits in int16 (high bit depth content is <= 12-bit).
uint32_t svt_aom_variance_highbd_neon(const uint16_t* a, int a_stride, const uint16_t* b, int b_stride, int w, int h,
                                      uint32_t* sse) {
    int32x4_t sum_s32    = vdupq_n_s32(0);
    int32x4_t sse_s32[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};

    for (int i = 0; i < h; ++i) {
        int j = 0;
        for (; j + 8 <= w; j += 8) {
            const uint16x8_t s    = vld1q_u16(a + j);
            const uint16x8_t r    = vld1q_u16(b + j);
            const int16x8_t  diff = vreinterpretq_s16_u16(vsubq_u16(s, r));
            sum_s32               = vpadalq_s16(sum_s32, diff);
            sse_s32[0]            = vmlal_s16(sse_s32[0], vget_low_s16(diff), vget_low_s16(diff));
            sse_s32[1]            = vmlal_s16(sse_s32[1], vget_high_s16(diff), vget_high_s16(diff));
        }
        if (j + 4 <= w) {
            const uint16x4_t s    = vld1_u16(a + j);
            const uint16x4_t r    = vld1_u16(b + j);
            const int16x4_t  diff = vreinterpret_s16_u16(vsub_u16(s, r));
            sum_s32               = vaddw_s16(sum_s32, diff);
            sse_s32[0]            = vmlal_s16(sse_s32[0], diff, diff);
        }
        a += a_stride;
        b += b_stride;
    }

    const int sad = vaddvq_s32(sum_s32);
    *sse          = vaddvq_u32(vaddq_u32(vreinterpretq_u32_s32(sse_s32[0]), vreinterpretq_u32_s32(sse_s32[1])));
    return (uint32_t)(*sse - ((int64_t)sad * sad) / (w * h));
}
