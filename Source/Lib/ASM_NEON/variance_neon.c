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
#include "sum_neon.h"
#include "var_filter_neon.h"

#ifdef __clang__
#define DISABLE_LOOP_UNROLL 1
#else
#define DISABLE_LOOP_UNROLL 0
#endif

static inline void variance_4xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                     uint32_t* sse, int* sum) {
    int16x8_t  sum_s16 = vdupq_n_s16(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

    // Number of rows we can process before 'sum_s16' overflows:
    // 32767 / 255 ~= 128, but we use an 8-wide accumulator; so 256 4-wide rows.
    assert(h <= 256);

    int i = h;
    do {
        uint8x8_t s    = load_u8_4x2(src, src_stride);
        uint8x8_t r    = load_u8_4x2(ref, ref_stride);
        int16x8_t diff = vreinterpretq_s16_u16(vsubl_u8(s, r));

        sum_s16 = vaddq_s16(sum_s16, diff);

        uint16x8_t sq = vreinterpretq_u16_s16(vmulq_s16(diff, diff));
        sse_u32       = vpadalq_u16(sse_u32, sq);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        i -= 2;
    } while (i != 0);

    *sum = vaddlvq_s16(sum_s16);
    *sse = vaddvq_u32(sse_u32);
}

static inline void variance_8xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                     uint32_t* sse, int* sum) {
    int16x8_t  sum_s16[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    assert(h <= 128);
    assert((h & 1) == 0);

    int i = h;
#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        uint8x8_t s0 = vld1_u8(src);
        uint8x8_t r0 = vld1_u8(ref);
        uint8x8_t s1 = vld1_u8(src + src_stride);
        uint8x8_t r1 = vld1_u8(ref + ref_stride);

        int16x8_t diff0 = vreinterpretq_s16_u16(vsubl_u8(s0, r0));
        int16x8_t diff1 = vreinterpretq_s16_u16(vsubl_u8(s1, r1));

        sum_s16[0] = vaddq_s16(sum_s16[0], diff0);
        sum_s16[1] = vaddq_s16(sum_s16[1], diff1);

        uint16x8_t sq0 = vreinterpretq_u16_s16(vmulq_s16(diff0, diff0));
        uint16x8_t sq1 = vreinterpretq_u16_s16(vmulq_s16(diff1, diff1));

        sse_u32[0] = vpadalq_u16(sse_u32[0], sq0);
        sse_u32[1] = vpadalq_u16(sse_u32[1], sq1);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        i -= 2;
    } while (i != 0);

    *sum = vaddlvq_s16(vaddq_s16(sum_s16[0], sum_s16[1]));
    *sse = vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}

static inline void variance_16xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                      uint32_t* sse, int* sum) {
    int16x8_t sum_s16[2] = {vdupq_n_s16(0), vdupq_n_s16(0)};
    int32x4_t sse_s32[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};

    // Number of rows we can process before 'sum_s16' accumulators overflow:
    // 32767 / 255 ~= 128, so 128 16-wide rows.
    assert(h <= 128);

    int i = h;
    do {
        uint8x16_t s = vld1q_u8(src);
        uint8x16_t r = vld1q_u8(ref);

        int16x8_t diff_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
        int16x8_t diff_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

        sum_s16[0] = vaddq_s16(sum_s16[0], diff_l);
        sum_s16[1] = vaddq_s16(sum_s16[1], diff_h);

        sse_s32[0] = vmlal_s16(sse_s32[0], vget_low_s16(diff_l), vget_low_s16(diff_l));
        sse_s32[1] = vmlal_s16(sse_s32[1], vget_high_s16(diff_l), vget_high_s16(diff_l));
        sse_s32[0] = vmlal_s16(sse_s32[0], vget_low_s16(diff_h), vget_low_s16(diff_h));
        sse_s32[1] = vmlal_s16(sse_s32[1], vget_high_s16(diff_h), vget_high_s16(diff_h));

        src += src_stride;
        ref += ref_stride;
    } while (--i != 0);

    *sum = vaddlvq_s16(vaddq_s16(sum_s16[0], sum_s16[1]));
    *sse = (uint32_t)vaddvq_s32(vaddq_s32(sse_s32[0], sse_s32[1]));
}

static inline void variance_large_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int w,
                                       int h, int h_limit, uint32_t* sse, int* sum) {
    int32x4_t sum_s32 = vdupq_n_s32(0);

    uint32x4_t sse_u32[4] = {
        vdupq_n_u32(0),
        vdupq_n_u32(0),
        vdupq_n_u32(0),
        vdupq_n_u32(0),
    };

    // 'h_limit' is the number of 'w'-width rows we can process before our 16-bit
    // accumulator overflows. After hitting this limit we accumulate into 32-bit
    // elements.
    int h_tmp = h > h_limit ? h_limit : h;

    int i = 0;
#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        int16x8_t sum_s16_0 = vdupq_n_s16(0);
        int16x8_t sum_s16_1 = vdupq_n_s16(0);

        do {
            int j = 0;
            int t = 0;
            do {
                uint8x16_t s = vld1q_u8(src + j);
                uint8x16_t r = vld1q_u8(ref + j);

                const int16x8_t diff_l = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s), vget_low_u8(r)));
                const int16x8_t diff_h = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s), vget_high_u8(r)));

                sum_s16_0 = vaddq_s16(sum_s16_0, diff_l);
                sum_s16_1 = vaddq_s16(sum_s16_1, diff_h);

                uint16x8_t sq_l = vreinterpretq_u16_s16(vmulq_s16(diff_l, diff_l));
                uint16x8_t sq_h = vreinterpretq_u16_s16(vmulq_s16(diff_h, diff_h));
                sse_u32[t]      = vpadalq_u16(sse_u32[t], sq_l);
                sse_u32[t]      = vpadalq_u16(sse_u32[t], sq_h);

                j += 16;
                t = (t + 1) & 3;
            } while (j < w);

            src += src_stride;
            ref += ref_stride;
            i++;
        } while (i < h_tmp);

        sum_s32 = vpadalq_s16(sum_s32, sum_s16_0);
        sum_s32 = vpadalq_s16(sum_s32, sum_s16_1);

        h_tmp += h_limit;
    } while (i < h);

    *sum = vaddvq_s32(sum_s32);

    uint32x4_t sse_sum = vaddq_u32(vaddq_u32(sse_u32[0], sse_u32[1]), vaddq_u32(sse_u32[2], sse_u32[3]));
    *sse               = vaddvq_u32(sse_sum);
}

static inline void variance_32xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                      uint32_t* sse, int* sum) {
    variance_large_neon(src, src_stride, ref, ref_stride, 32, h, 64, sse, sum);
}

static inline void variance_64xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                      uint32_t* sse, int* sum) {
    variance_large_neon(src, src_stride, ref, ref_stride, 64, h, 32, sse, sum);
}

static inline void variance_128xh_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int h,
                                       uint32_t* sse, int* sum) {
    variance_large_neon(src, src_stride, ref, ref_stride, 128, h, 16, sse, sum);
}

#define VARIANCE_WXH_NEON(w, h, shift)                                                               \
    unsigned int svt_aom_variance##w##x##h##_neon(                                                   \
        const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, unsigned int* sse) { \
        int sum;                                                                                     \
        variance_##w##xh_neon(src, src_stride, ref, ref_stride, h, sse, &sum);                       \
        return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                                     \
    }

VARIANCE_WXH_NEON(4, 4, 4)
VARIANCE_WXH_NEON(4, 8, 5)
VARIANCE_WXH_NEON(4, 16, 6)

VARIANCE_WXH_NEON(8, 4, 5)
VARIANCE_WXH_NEON(8, 8, 6)
VARIANCE_WXH_NEON(8, 16, 7)
VARIANCE_WXH_NEON(8, 32, 8)

VARIANCE_WXH_NEON(16, 4, 6)
VARIANCE_WXH_NEON(16, 8, 7)
VARIANCE_WXH_NEON(16, 16, 8)
VARIANCE_WXH_NEON(16, 32, 9)
VARIANCE_WXH_NEON(16, 64, 10)

VARIANCE_WXH_NEON(32, 8, 8)
VARIANCE_WXH_NEON(32, 16, 9)
VARIANCE_WXH_NEON(32, 32, 10)
VARIANCE_WXH_NEON(32, 64, 11)

VARIANCE_WXH_NEON(64, 16, 10)
VARIANCE_WXH_NEON(64, 32, 11)
VARIANCE_WXH_NEON(64, 64, 12)
VARIANCE_WXH_NEON(64, 128, 13)

VARIANCE_WXH_NEON(128, 64, 13)
VARIANCE_WXH_NEON(128, 128, 14)

#undef VARIANCE_WXH_NEON

// =============================================================================
// Small-block sub_pixel_variance helpers (4xH)
// =============================================================================
#define SUBPEL_VARIANCE_4XH_NEON(h, padding)                                         \
    unsigned int svt_aom_sub_pixel_variance4##x##h##_neon(const uint8_t* src,        \
                                                          int            src_stride, \
                                                          int            xoffset,    \
                                                          int            yoffset,    \
                                                          const uint8_t* ref,        \
                                                          int            ref_stride, \
                                                          uint32_t*      sse) {           \
        uint8_t tmp0[4 * (h + padding)];                                             \
        uint8_t tmp1[4 * h];                                                         \
        var_filter_block2d_bil_w4(src, tmp0, src_stride, 1, (h + padding), xoffset); \
        var_filter_block2d_bil_w4(tmp0, tmp1, 4, 4, h, yoffset);                     \
        return svt_aom_variance4##x##h##_neon(tmp1, 4, ref, ref_stride, sse);        \
    }

SUBPEL_VARIANCE_4XH_NEON(4, 2)
SUBPEL_VARIANCE_4XH_NEON(8, 2)
SUBPEL_VARIANCE_4XH_NEON(16, 2)

#undef SUBPEL_VARIANCE_4XH_NEON

static inline void var_w8x2_accum(uint16x8_t* src_sum, uint16x8_t* ref_sum, uint32x4_t* sse0, uint32x4_t* sse1,
                                  uint8x8_t pred0, uint8x8_t pred1, const uint8_t* ref, int64_t ref_stride) {
    const uint8x8_t r0 = vld1_u8(ref);
    const uint8x8_t r1 = vld1_u8(ref + ref_stride);

    *src_sum = vaddw_u8(*src_sum, pred0);
    *src_sum = vaddw_u8(*src_sum, pred1);
    *ref_sum = vaddw_u8(*ref_sum, r0);
    *ref_sum = vaddw_u8(*ref_sum, r1);

    const uint8x8_t abs_diff0 = vabd_u8(pred0, r0);
    const uint8x8_t abs_diff1 = vabd_u8(pred1, r1);
    *sse0                     = vpadalq_u16(*sse0, vmull_u8(abs_diff0, abs_diff0));
    *sse1                     = vpadalq_u16(*sse1, vmull_u8(abs_diff1, abs_diff1));
}

// Calculate variance from src_sum, ref_sum and sse accumulator vectors.
static inline unsigned int var_w8_reduce(uint16x8_t src_sum, uint16x8_t ref_sum, uint32x4_t sse0, uint32x4_t sse1,
                                         int shift, unsigned int* sse_out) {
    const int16x8_t sum_diff = vreinterpretq_s16_u16(vsubq_u16(src_sum, ref_sum));
    const int32_t   sum      = vaddlvq_s16(sum_diff);
    *sse_out                 = vaddvq_u32(vaddq_u32(sse0, sse1));
    return *sse_out - (uint32_t)(((int64_t)sum * sum) >> shift);
}

// Emit a fused sub pixel variance function for blocks of width 8 that uses the
// specified horizontal 2-tap filter.
#define FUSED_SUBPEL_VAR_8XH_X(X_FILTER_OFFSET, X_FILTER)                                      \
    static unsigned int fused_subpel_variance_8xh_##X_FILTER_OFFSET(const uint8_t* src,        \
                                                                    int64_t        src_stride, \
                                                                    int            h,          \
                                                                    int            shift,      \
                                                                    const uint8_t* ref,        \
                                                                    int64_t        ref_stride, \
                                                                    unsigned int*  sse_out) {   \
        uint16x8_t src_sum = vdupq_n_u16(0);                                                   \
        uint16x8_t ref_sum = vdupq_n_u16(0);                                                   \
        uint32x4_t sse0    = vdupq_n_u32(0);                                                   \
        uint32x4_t sse1    = vdupq_n_u32(0);                                                   \
        do {                                                                                   \
            const uint8x8_t pred0 = X_FILTER(src);                                             \
            const uint8x8_t pred1 = X_FILTER(src + src_stride);                                \
            var_w8x2_accum(&src_sum, &ref_sum, &sse0, &sse1, pred0, pred1, ref, ref_stride);   \
            src += 2 * src_stride;                                                             \
            ref += 2 * ref_stride;                                                             \
            h -= 2;                                                                            \
        } while (h != 0);                                                                      \
        return var_w8_reduce(src_sum, ref_sum, sse0, sse1, shift, sse_out);                    \
    }

FUSED_SUBPEL_VAR_8XH_X(2, load_interp_3_1_w8)
FUSED_SUBPEL_VAR_8XH_X(4, load_interp_1_1_w8)
FUSED_SUBPEL_VAR_8XH_X(6, load_interp_1_3_w8)

// Emit a fused sub pixel variance function for blocks of width 8 that uses the
// specified 2-tap bilinear filter.
#define FUSED_SUBPEL_VAR_8XH_XY(X_FILTER_OFFSET, Y_FILTER_OFFSET, X_FILTER, Y_FILTER)                              \
    static unsigned int fused_subpel_variance_8xh_##X_FILTER_OFFSET##_##Y_FILTER_OFFSET(const uint8_t* src,        \
                                                                                        int64_t        src_stride, \
                                                                                        int            h,          \
                                                                                        int            shift,      \
                                                                                        const uint8_t* ref,        \
                                                                                        int64_t        ref_stride, \
                                                                                        unsigned int*  sse_out) {   \
        uint16x8_t src_sum = vdupq_n_u16(0);                                                                       \
        uint16x8_t ref_sum = vdupq_n_u16(0);                                                                       \
        uint32x4_t sse0    = vdupq_n_u32(0);                                                                       \
        uint32x4_t sse1    = vdupq_n_u32(0);                                                                       \
        uint8x8_t  prev    = X_FILTER(src);                                                                        \
        src += src_stride;                                                                                         \
        do {                                                                                                       \
            const uint8x8_t cur0  = X_FILTER(src);                                                                 \
            const uint8x8_t pred0 = Y_FILTER(prev, cur0);                                                          \
            const uint8x8_t cur1  = X_FILTER(src + src_stride);                                                    \
            const uint8x8_t pred1 = Y_FILTER(cur0, cur1);                                                          \
            var_w8x2_accum(&src_sum, &ref_sum, &sse0, &sse1, pred0, pred1, ref, ref_stride);                       \
            prev = cur1;                                                                                           \
            src += 2 * src_stride;                                                                                 \
            ref += 2 * ref_stride;                                                                                 \
            h -= 2;                                                                                                \
        } while (h != 0);                                                                                          \
        return var_w8_reduce(src_sum, ref_sum, sse0, sse1, shift, sse_out);                                        \
    }

#define FUSED_SUBPEL_VAR_8XH(X_FILTER_OFFSET, X_FILTER)                  \
    FUSED_SUBPEL_VAR_8XH_XY(X_FILTER_OFFSET, 2, X_FILTER, interp_3_1_w8) \
    FUSED_SUBPEL_VAR_8XH_XY(X_FILTER_OFFSET, 4, X_FILTER, interp_1_1_w8) \
    FUSED_SUBPEL_VAR_8XH_XY(X_FILTER_OFFSET, 6, X_FILTER, interp_1_3_w8)

FUSED_SUBPEL_VAR_8XH(0, vld1_u8)
FUSED_SUBPEL_VAR_8XH(2, load_interp_3_1_w8)
FUSED_SUBPEL_VAR_8XH(4, load_interp_1_1_w8)
FUSED_SUBPEL_VAR_8XH(6, load_interp_1_3_w8)

#undef FUSED_SUBPEL_VAR_8XH_X
#undef FUSED_SUBPEL_VAR_8XH_XY
#undef FUSED_SUBPEL_VAR_8XH

static unsigned int fused_subpel_variance_8xh_generic(const uint8_t* src, int64_t src_stride, int h, int shift,
                                                      const uint8_t* ref, int64_t ref_stride, unsigned int* sse_out,
                                                      unsigned int offsets) {
    const int  xoffset = offsets & 7;
    const int  yoffset = (offsets >> 3) & 7;
    uint16x8_t src_sum = vdupq_n_u16(0);
    uint16x8_t ref_sum = vdupq_n_u16(0);
    uint32x4_t sse0    = vdupq_n_u32(0);
    uint32x4_t sse1    = vdupq_n_u32(0);

    if (yoffset == 0) {
        do {
            const uint8x8_t pred0 = load_interp_w8(src, xoffset);
            const uint8x8_t pred1 = load_interp_w8(src + src_stride, xoffset);
            var_w8x2_accum(&src_sum, &ref_sum, &sse0, &sse1, pred0, pred1, ref, ref_stride);
            src += 2 * src_stride;
            ref += 2 * ref_stride;
            h -= 2;
        } while (h != 0);
    } else {
        const uint8x8_t prev_row  = load_interp_w8(src, xoffset);
        uint8x16_t      prev_pair = vcombine_u8(prev_row, prev_row);
        src += src_stride;
        do {
            const uint8x16_t cur_pair = load_interp_w8x2(src, src_stride, xoffset);
            const uint8x16_t adj_pair = vcombine_u8(vget_high_u8(prev_pair), vget_low_u8(cur_pair));
            const uint8x16_t pred     = interp_w16(adj_pair, cur_pair, yoffset);
            var_w8x2_accum(&src_sum, &ref_sum, &sse0, &sse1, vget_low_u8(pred), vget_high_u8(pred), ref, ref_stride);
            prev_pair = cur_pair;
            src += 2 * src_stride;
            ref += 2 * ref_stride;
            h -= 2;
        } while (h != 0);
    }

    return var_w8_reduce(src_sum, ref_sum, sse0, sse1, shift, sse_out);
}

typedef unsigned int (*SubpelVarFn)(const uint8_t* src, int64_t src_stride, int h, int shift, const uint8_t* ref,
                                    int64_t ref_stride, unsigned int* sse);

#define FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE(W)                                 \
    static const SubpelVarFn fused_subpel_variance_##W##xh_fn_table[4][4] = { \
        {NULL,                                                                \
         fused_subpel_variance_##W##xh_0_2,                                   \
         fused_subpel_variance_##W##xh_0_4,                                   \
         fused_subpel_variance_##W##xh_0_6},                                  \
        {fused_subpel_variance_##W##xh_2,                                     \
         fused_subpel_variance_##W##xh_2_2,                                   \
         fused_subpel_variance_##W##xh_2_4,                                   \
         fused_subpel_variance_##W##xh_2_6},                                  \
        {fused_subpel_variance_##W##xh_4,                                     \
         fused_subpel_variance_##W##xh_4_2,                                   \
         fused_subpel_variance_##W##xh_4_4,                                   \
         fused_subpel_variance_##W##xh_4_6},                                  \
        {fused_subpel_variance_##W##xh_6,                                     \
         fused_subpel_variance_##W##xh_6_2,                                   \
         fused_subpel_variance_##W##xh_6_4,                                   \
         fused_subpel_variance_##W##xh_6_6},                                  \
    }

FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE(8);

static unsigned int sub_pixel_variance_8xh_neon(const uint8_t* src, int src_stride, int xoffset, int yoffset,
                                                const uint8_t* ref, int ref_stride, int h, unsigned int* sse) {
    if (xoffset == 0 && yoffset == 0) {
        int sum;
        variance_8xh_neon(src, src_stride, ref, ref_stride, h, sse, &sum);
        return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz(8 * h));
    }
    if ((xoffset | yoffset) & 1) {
        return fused_subpel_variance_8xh_generic(
            src, src_stride, h, svt_ctz(8 * h), ref, ref_stride, sse, (unsigned int)(xoffset | (yoffset << 3)));
    }
    const SubpelVarFn f = fused_subpel_variance_8xh_fn_table[xoffset >> 1][yoffset >> 1];
    return f(src, src_stride, h, svt_ctz(8 * h), ref, ref_stride, sse);
}

#define FUSED_SUBPEL_VARIANCE_8XH_NEON(H)                                                                 \
    unsigned int svt_aom_sub_pixel_variance8x##H##_neon(const uint8_t* src,                               \
                                                        int            src_stride,                        \
                                                        int            xoffset,                           \
                                                        int            yoffset,                           \
                                                        const uint8_t* ref,                               \
                                                        int            ref_stride,                        \
                                                        unsigned int*  sse) {                              \
        return sub_pixel_variance_8xh_neon(src, src_stride, xoffset, yoffset, ref, ref_stride, (H), sse); \
    }

FUSED_SUBPEL_VARIANCE_8XH_NEON(4)
FUSED_SUBPEL_VARIANCE_8XH_NEON(8)
FUSED_SUBPEL_VARIANCE_8XH_NEON(16)
FUSED_SUBPEL_VARIANCE_8XH_NEON(32)

#undef FUSED_SUBPEL_VARIANCE_8XH_NEON

// Number of accumulators for block width W.
#define ACCUM(W) (((W) / 16 >= 4) ? 4 : (W) / 16)

// Only 128x128 needs two 64-row chunks to prevent the 16-bit sums overflowing.
#define HEIGHT_LIMIT(W, H) (((W) == 128 && (H) == 128) ? 64 : (H))

static inline void var_wide_accum(uint16x8_t* src_sum, uint16x8_t* ref_sum, uint32x4_t* sse_lo, uint32x4_t* sse_hi,
                                  uint8x16_t pred, uint8x16_t r) {
    *src_sum                  = vpadalq_u8(*src_sum, pred);
    *ref_sum                  = vpadalq_u8(*ref_sum, r);
    const uint8x16_t abs_diff = vabdq_u8(pred, r);
    const uint16x8_t sq_lo    = vmull_u8(vget_low_u8(abs_diff), vget_low_u8(abs_diff));
    const uint16x8_t sq_hi    = vmull_u8(vget_high_u8(abs_diff), vget_high_u8(abs_diff));
    *sse_lo                   = vpadalq_u16(*sse_lo, sq_lo);
    *sse_hi                   = vpadalq_u16(*sse_hi, sq_hi);
}

static inline int32_t var_wide_sum_reduce(const uint16x8_t* src_sum, const uint16x8_t* ref_sum, int n) {
    uint32_t src_sum_u32 = 0;
    uint32_t ref_sum_u32 = 0;
    for (int i = 0; i < n; ++i) {
        src_sum_u32 += vaddlvq_u16(src_sum[i]);
        ref_sum_u32 += vaddlvq_u16(ref_sum[i]);
    }
    return (int32_t)src_sum_u32 - (int32_t)ref_sum_u32;
}

static inline int32_t var_wide_sum_flush(uint16x8_t* src_sum, uint16x8_t* ref_sum, int n) {
    const int32_t sum = var_wide_sum_reduce(src_sum, ref_sum, n);
    for (int i = 0; i < n; ++i) {
        src_sum[i] = vdupq_n_u16(0);
        ref_sum[i] = vdupq_n_u16(0);
    }
    return sum;
}

// Calculate variance from 16-bit sum and split SSE accumulator vectors.
static inline int32_t var_wide_reduce(const uint16x8_t* src_sum, const uint16x8_t* ref_sum, const uint32x4_t* sse_lo,
                                      const uint32x4_t* sse_hi, int n, unsigned int* sse_out) {
    uint32x4_t sse_sum = vaddq_u32(sse_lo[0], sse_hi[0]);
    for (int i = 1; i < n; ++i) {
        sse_sum = vaddq_u32(sse_sum, vaddq_u32(sse_lo[i], sse_hi[i]));
    }
    *sse_out = vaddvq_u32(sse_sum);
    return var_wide_sum_reduce(src_sum, ref_sum, n);
}

// Emit a fused sub pixel variance function for blocks of width W that uses the
// specified horizontal 2-tap filter.
#define FUSED_SUBPEL_VAR_WXH_X(W, X_FILTER_OFFSET, X_FILTER)                                       \
    static unsigned int fused_subpel_variance_##W##xh_##X_FILTER_OFFSET(const uint8_t* src,        \
                                                                        int64_t        src_stride, \
                                                                        int            h,          \
                                                                        int            shift,      \
                                                                        const uint8_t* ref,        \
                                                                        int64_t        ref_stride, \
                                                                        unsigned int*  sse_out) {   \
        uint16x8_t src_sum[ACCUM(W)];                                                              \
        uint16x8_t ref_sum[ACCUM(W)];                                                              \
        uint32x4_t sse_lo[ACCUM(W)];                                                               \
        uint32x4_t sse_hi[ACCUM(W)];                                                               \
        for (int i = 0; i < ACCUM(W); ++i) {                                                       \
            src_sum[i] = vdupq_n_u16(0);                                                           \
            ref_sum[i] = vdupq_n_u16(0);                                                           \
            sse_lo[i]  = vdupq_n_u32(0);                                                           \
            sse_hi[i]  = vdupq_n_u32(0);                                                           \
        }                                                                                          \
        int32_t sum = 0;                                                                           \
        do {                                                                                       \
            int h_chunk = HEIGHT_LIMIT(W, h);                                                      \
            h -= h_chunk;                                                                          \
            do {                                                                                   \
                const uint8_t* sp_row = src;                                                       \
                const uint8_t* rp_row = ref;                                                       \
                for (int t = 0; t < W / 16; ++t) {                                                 \
                    const uint8x16_t pred = X_FILTER(sp_row + t * 16);                             \
                    const uint8x16_t r    = vld1q_u8(rp_row + t * 16);                             \
                    const int        i    = t & (ACCUM(W) - 1);                                    \
                    var_wide_accum(&src_sum[i], &ref_sum[i], &sse_lo[i], &sse_hi[i], pred, r);     \
                }                                                                                  \
                src += src_stride;                                                                 \
                ref += ref_stride;                                                                 \
            } while (--h_chunk != 0);                                                              \
            if (h != 0) {                                                                          \
                sum += var_wide_sum_flush(src_sum, ref_sum, ACCUM(W));                             \
            }                                                                                      \
        } while (h != 0);                                                                          \
        sum += var_wide_reduce(src_sum, ref_sum, sse_lo, sse_hi, ACCUM(W), sse_out);               \
        return *sse_out - (uint32_t)(((int64_t)sum * sum) >> shift);                               \
    }

#define FUSED_SUBPEL_VAR_WXH_X_COMBOS(W)              \
    FUSED_SUBPEL_VAR_WXH_X(W, 2, load_interp_3_1_w16) \
    FUSED_SUBPEL_VAR_WXH_X(W, 4, load_interp_1_1_w16) \
    FUSED_SUBPEL_VAR_WXH_X(W, 6, load_interp_1_3_w16)

// Emit a fused sub pixel variance function for blocks of width W that uses the
// specified 2-tap bilinear filter.
#define FUSED_SUBPEL_VAR_WXH_XY(W, X_FILTER_OFFSET, Y_FILTER_OFFSET, X_FILTER, Y_FILTER)                               \
    static unsigned int fused_subpel_variance_##W##xh_##X_FILTER_OFFSET##_##Y_FILTER_OFFSET(const uint8_t* src,        \
                                                                                            int64_t        src_stride, \
                                                                                            int            h,          \
                                                                                            int            shift,      \
                                                                                            const uint8_t* ref,        \
                                                                                            int64_t        ref_stride, \
                                                                                            unsigned int*  sse_out) {   \
        uint16x8_t src_sum[ACCUM(W)];                                                                                  \
        uint16x8_t ref_sum[ACCUM(W)];                                                                                  \
        uint32x4_t sse_lo[ACCUM(W)];                                                                                   \
        uint32x4_t sse_hi[ACCUM(W)];                                                                                   \
        uint8x16_t prev[W / 16];                                                                                       \
        uint8x16_t cur[W / 16];                                                                                        \
        for (int i = 0; i < ACCUM(W); ++i) {                                                                           \
            src_sum[i] = vdupq_n_u16(0);                                                                               \
            ref_sum[i] = vdupq_n_u16(0);                                                                               \
            sse_lo[i]  = vdupq_n_u32(0);                                                                               \
            sse_hi[i]  = vdupq_n_u32(0);                                                                               \
        }                                                                                                              \
        int32_t sum = 0;                                                                                               \
        for (int t = 0; t < W / 16; ++t) {                                                                             \
            prev[t] = X_FILTER(src + t * 16);                                                                          \
        }                                                                                                              \
        src += src_stride;                                                                                             \
        do {                                                                                                           \
            int h_chunk = HEIGHT_LIMIT(W, h);                                                                          \
            h -= h_chunk;                                                                                              \
            do {                                                                                                       \
                for (int t = 0; t < W / 16; ++t) {                                                                     \
                    cur[t]                = X_FILTER(src + t * 16);                                                    \
                    const uint8x16_t pred = Y_FILTER(prev[t], cur[t]);                                                 \
                    const uint8x16_t r    = vld1q_u8(ref + t * 16);                                                    \
                    const int        i    = t & (ACCUM(W) - 1);                                                        \
                    var_wide_accum(&src_sum[i], &ref_sum[i], &sse_lo[i], &sse_hi[i], pred, r);                         \
                }                                                                                                      \
                src += src_stride;                                                                                     \
                ref += ref_stride;                                                                                     \
                for (int t = 0; t < W / 16; ++t) {                                                                     \
                    prev[t]               = X_FILTER(src + t * 16);                                                    \
                    const uint8x16_t pred = Y_FILTER(cur[t], prev[t]);                                                 \
                    const uint8x16_t r    = vld1q_u8(ref + t * 16);                                                    \
                    const int        i    = t & (ACCUM(W) - 1);                                                        \
                    var_wide_accum(&src_sum[i], &ref_sum[i], &sse_lo[i], &sse_hi[i], pred, r);                         \
                }                                                                                                      \
                src += src_stride;                                                                                     \
                ref += ref_stride;                                                                                     \
                h_chunk -= 2;                                                                                          \
            } while (h_chunk != 0);                                                                                    \
            if (h != 0) {                                                                                              \
                sum += var_wide_sum_flush(src_sum, ref_sum, ACCUM(W));                                                 \
            }                                                                                                          \
        } while (h != 0);                                                                                              \
        sum += var_wide_reduce(src_sum, ref_sum, sse_lo, sse_hi, ACCUM(W), sse_out);                                   \
        return *sse_out - (uint32_t)(((int64_t)sum * sum) >> shift);                                                   \
    }

// Emit a fused sub pixel variance function for the specified block width W. The
// function is generic in the sense that it can use any 2-tap bilinear filter.
#define FUSED_SUBPEL_VAR_WXH_GENERIC(W)                                                            \
    static unsigned int fused_subpel_variance_##W##xh_generic(const uint8_t* src,                  \
                                                              int64_t        src_stride,           \
                                                              int            h,                    \
                                                              int            shift,                \
                                                              const uint8_t* ref,                  \
                                                              int64_t        ref_stride,           \
                                                              unsigned int*  sse_out,              \
                                                              unsigned int   offsets) {              \
        const int  xoffset = offsets & 7;                                                          \
        const int  yoffset = (offsets >> 3) & 7;                                                   \
        uint16x8_t src_sum[ACCUM(W)];                                                              \
        uint16x8_t ref_sum[ACCUM(W)];                                                              \
        uint32x4_t sse_lo[ACCUM(W)];                                                               \
        uint32x4_t sse_hi[ACCUM(W)];                                                               \
        for (int i = 0; i < ACCUM(W); ++i) {                                                       \
            src_sum[i] = vdupq_n_u16(0);                                                           \
            ref_sum[i] = vdupq_n_u16(0);                                                           \
            sse_lo[i]  = vdupq_n_u32(0);                                                           \
            sse_hi[i]  = vdupq_n_u32(0);                                                           \
        }                                                                                          \
        int32_t sum = 0;                                                                           \
        if (yoffset == 0) {                                                                        \
            do {                                                                                   \
                int h_chunk = HEIGHT_LIMIT(W, h);                                                  \
                h -= h_chunk;                                                                      \
                do {                                                                               \
                    for (int t = 0; t < W / 16; ++t) {                                             \
                        const uint8x16_t pred = load_interp_w16(src + t * 16, xoffset);            \
                        const uint8x16_t r    = vld1q_u8(ref + t * 16);                            \
                        const int        i    = t & (ACCUM(W) - 1);                                \
                        var_wide_accum(&src_sum[i], &ref_sum[i], &sse_lo[i], &sse_hi[i], pred, r); \
                    }                                                                              \
                    src += src_stride;                                                             \
                    ref += ref_stride;                                                             \
                } while (--h_chunk != 0);                                                          \
                if (h != 0) {                                                                      \
                    sum += var_wide_sum_flush(src_sum, ref_sum, ACCUM(W));                         \
                }                                                                                  \
            } while (h != 0);                                                                      \
        } else {                                                                                   \
            uint8x16_t prev[W / 16];                                                               \
            for (int t = 0; t < W / 16; ++t) {                                                     \
                prev[t] = load_interp_w16(src + t * 16, xoffset);                                  \
            }                                                                                      \
            src += src_stride;                                                                     \
            do {                                                                                   \
                int h_chunk = HEIGHT_LIMIT(W, h);                                                  \
                h -= h_chunk;                                                                      \
                do {                                                                               \
                    for (int t = 0; t < W / 16; ++t) {                                             \
                        const uint8x16_t cur  = load_interp_w16(src + t * 16, xoffset);            \
                        const uint8x16_t pred = interp_w16(prev[t], cur, yoffset);                 \
                        const uint8x16_t r    = vld1q_u8(ref + t * 16);                            \
                        const int        i    = t & (ACCUM(W) - 1);                                \
                        var_wide_accum(&src_sum[i], &ref_sum[i], &sse_lo[i], &sse_hi[i], pred, r); \
                        prev[t] = cur;                                                             \
                    }                                                                              \
                    src += src_stride;                                                             \
                    ref += ref_stride;                                                             \
                } while (--h_chunk != 0);                                                          \
                if (h != 0) {                                                                      \
                    sum += var_wide_sum_flush(src_sum, ref_sum, ACCUM(W));                         \
                }                                                                                  \
            } while (h != 0);                                                                      \
        }                                                                                          \
        sum += var_wide_reduce(src_sum, ref_sum, sse_lo, sse_hi, ACCUM(W), sse_out);               \
        return *sse_out - (uint32_t)(((int64_t)sum * sum) >> shift);                               \
    }

#define FUSED_SUBPEL_VAR_WXH_XY_COMBOS(W)                                 \
    FUSED_SUBPEL_VAR_WXH_XY(W, 0, 2, vld1q_u8, interp_3_1_w16)            \
    FUSED_SUBPEL_VAR_WXH_XY(W, 0, 4, vld1q_u8, interp_1_1_w16)            \
    FUSED_SUBPEL_VAR_WXH_XY(W, 0, 6, vld1q_u8, interp_1_3_w16)            \
    FUSED_SUBPEL_VAR_WXH_XY(W, 2, 2, load_interp_3_1_w16, interp_3_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 2, 4, load_interp_3_1_w16, interp_1_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 2, 6, load_interp_3_1_w16, interp_1_3_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 4, 2, load_interp_1_1_w16, interp_3_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 4, 4, load_interp_1_1_w16, interp_1_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 4, 6, load_interp_1_1_w16, interp_1_3_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 6, 2, load_interp_1_3_w16, interp_3_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 6, 4, load_interp_1_3_w16, interp_1_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(W, 6, 6, load_interp_1_3_w16, interp_1_3_w16)

#define FUSED_SUBPEL_VAR_WXH(W)      \
    FUSED_SUBPEL_VAR_WXH_GENERIC(W)  \
    FUSED_SUBPEL_VAR_WXH_X_COMBOS(W) \
    FUSED_SUBPEL_VAR_WXH_XY_COMBOS(W)

FUSED_SUBPEL_VAR_WXH(16)
FUSED_SUBPEL_VAR_WXH(32)
FUSED_SUBPEL_VAR_WXH(64)
FUSED_SUBPEL_VAR_WXH(128)

#undef FUSED_SUBPEL_VAR_WXH_X
#undef FUSED_SUBPEL_VAR_WXH_X_COMBOS
#undef FUSED_SUBPEL_VAR_WXH_XY
#undef FUSED_SUBPEL_VAR_WXH_XY_COMBOS
#undef FUSED_SUBPEL_VAR_WXH_GENERIC
#undef FUSED_SUBPEL_VAR_WXH

FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE(16);
FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE(32);
FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE(64);
FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE(128);

#undef FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE

// Emit svt_aom_sub_pixel_variance{W}x{H}_neon for supported (W, H)
// pairs. Filter offset (0,0) uses the variance fast path. Even/even filter
// offsets dispatch through width-specific tables; odd offsets use a generic
// loop. Shift is computed at compile time via svt_ctz((W) * (H)).
#define SUBPEL_VARIANCE_WXH_NEON(W, H)                                                              \
    unsigned int svt_aom_sub_pixel_variance##W##x##H##_neon(const uint8_t* src,                     \
                                                            int            src_stride,              \
                                                            int            xoffset,                 \
                                                            int            yoffset,                 \
                                                            const uint8_t* ref,                     \
                                                            int            ref_stride,              \
                                                            unsigned int*  sse) {                    \
        if (xoffset == 0 && yoffset == 0) {                                                         \
            int sum;                                                                                \
            variance_##W##xh_neon(src, src_stride, ref, ref_stride, (H), sse, &sum);                \
            return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz((W) * (H)));                   \
        }                                                                                           \
        if ((xoffset | yoffset) & 1) {                                                              \
            return fused_subpel_variance_##W##xh_generic(src,                                       \
                                                         src_stride,                                \
                                                         (H),                                       \
                                                         svt_ctz((W) * (H)),                        \
                                                         ref,                                       \
                                                         ref_stride,                                \
                                                         sse,                                       \
                                                         (unsigned int)(xoffset | (yoffset << 3))); \
        }                                                                                           \
        const SubpelVarFn f = fused_subpel_variance_##W##xh_fn_table[xoffset >> 1][yoffset >> 1];   \
        return f(src, src_stride, (H), svt_ctz((W) * (H)), ref, ref_stride, sse);                   \
    }

SUBPEL_VARIANCE_WXH_NEON(16, 4)
SUBPEL_VARIANCE_WXH_NEON(16, 8)
SUBPEL_VARIANCE_WXH_NEON(16, 16)
SUBPEL_VARIANCE_WXH_NEON(16, 32)
SUBPEL_VARIANCE_WXH_NEON(16, 64)
SUBPEL_VARIANCE_WXH_NEON(32, 8)
SUBPEL_VARIANCE_WXH_NEON(32, 16)
SUBPEL_VARIANCE_WXH_NEON(32, 32)
SUBPEL_VARIANCE_WXH_NEON(32, 64)
SUBPEL_VARIANCE_WXH_NEON(64, 16)
SUBPEL_VARIANCE_WXH_NEON(64, 32)
SUBPEL_VARIANCE_WXH_NEON(64, 64)
SUBPEL_VARIANCE_WXH_NEON(64, 128)
SUBPEL_VARIANCE_WXH_NEON(128, 64)
SUBPEL_VARIANCE_WXH_NEON(128, 128)

#undef SUBPEL_VARIANCE_WXH_NEON
#undef HEIGHT_LIMIT
#undef ACCUM

unsigned int svt_aom_mse16x16_neon(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride) {
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    int i = 16;
    do {
        uint8x16_t s0 = vld1q_u8(src);
        uint8x16_t r0 = vld1q_u8(ref);

        int16x8_t diff0 = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(s0), vget_low_u8(r0)));
        int16x8_t diff1 = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(s0), vget_high_u8(r0)));

        uint16x8_t sq0 = vreinterpretq_u16_s16(vmulq_s16(diff0, diff0));
        uint16x8_t sq1 = vreinterpretq_u16_s16(vmulq_s16(diff1, diff1));
        sse_u32[0]     = vpadalq_u16(sse_u32[0], sq0);
        sse_u32[1]     = vpadalq_u16(sse_u32[1], sq1);

        src += src_stride;
        ref += ref_stride;
    } while (--i != 0);

    return vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
}
