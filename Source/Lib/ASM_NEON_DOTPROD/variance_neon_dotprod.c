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

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "var_filter_neon.h"

#ifdef __clang__
#define DISABLE_LOOP_UNROLL 1
#else
#define DISABLE_LOOP_UNROLL 0
#endif

static inline void variance_4xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                             int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum = vdupq_n_u32(0);
    uint32x4_t ref_sum = vdupq_n_u32(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

    do {
        uint8x16_t s = load_u8_4x4(src, src_stride);
        uint8x16_t r = load_u8_4x4(ref, ref_stride);

        src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
        ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

        uint8x16_t abs_diff = vabdq_u8(s, r);
        sse_u32             = vdotq_u32(sse_u32, abs_diff, abs_diff);

        src += 4 * src_stride;
        ref += 4 * ref_stride;
        h -= 4;
    } while (h != 0);

    int32x4_t sum_diff = vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum));
    *sum               = vaddvq_s32(sum_diff);
    *sse               = vaddvq_u32(sse_u32);
}

static inline void variance_8xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                             int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum = vdupq_n_u32(0);
    uint32x4_t ref_sum = vdupq_n_u32(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

    do {
        uint8x16_t s = load_u8_8x2(src, src_stride);
        uint8x16_t r = load_u8_8x2(ref, ref_stride);

        src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
        ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

        uint8x16_t abs_diff = vabdq_u8(s, r);
        sse_u32             = vdotq_u32(sse_u32, abs_diff, abs_diff);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        h -= 2;
    } while (h != 0);

    int32x4_t sum_diff = vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum));
    *sum               = vaddvq_s32(sum_diff);
    *sse               = vaddvq_u32(sse_u32);
}

static inline void variance_16xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                              int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum = vdupq_n_u32(0);
    uint32x4_t ref_sum = vdupq_n_u32(0);
    uint32x4_t sse_u32 = vdupq_n_u32(0);

    do {
        uint8x16_t s = vld1q_u8(src);
        uint8x16_t r = vld1q_u8(ref);

        src_sum = vdotq_u32(src_sum, s, vdupq_n_u8(1));
        ref_sum = vdotq_u32(ref_sum, r, vdupq_n_u8(1));

        uint8x16_t abs_diff = vabdq_u8(s, r);
        sse_u32             = vdotq_u32(sse_u32, abs_diff, abs_diff);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    int32x4_t sum_diff = vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum));
    *sum               = vaddvq_s32(sum_diff);
    *sse               = vaddvq_u32(sse_u32);
}

static inline void variance_large_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                               int w, int h, uint32_t* sse, int* sum) {
    uint32x4_t src_sum[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t ref_sum[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t sse_u32[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};

#if DISABLE_LOOP_UNROLL
#pragma clang loop unroll(disable)
#endif
    do {
        int i = 0;
        int t = 0;
        do {
            uint8x16_t s = vld1q_u8(src + i);
            uint8x16_t r = vld1q_u8(ref + i);

            src_sum[t] = vdotq_u32(src_sum[t], s, vdupq_n_u8(1));
            ref_sum[t] = vdotq_u32(ref_sum[t], r, vdupq_n_u8(1));

            uint8x16_t abs_diff = vabdq_u8(s, r);
            sse_u32[t]          = vdotq_u32(sse_u32[t], abs_diff, abs_diff);

            i += 16;
            t = (t + 1) & 3;
        } while (i < w);

        src += src_stride;
        ref += ref_stride;
    } while (--h != 0);

    uint32x4_t src_sum_v = vaddq_u32(vaddq_u32(src_sum[0], src_sum[1]), vaddq_u32(src_sum[2], src_sum[3]));
    uint32x4_t ref_sum_v = vaddq_u32(vaddq_u32(ref_sum[0], ref_sum[1]), vaddq_u32(ref_sum[2], ref_sum[3]));
    int32x4_t  sum_diff  = vsubq_s32(vreinterpretq_s32_u32(src_sum_v), vreinterpretq_s32_u32(ref_sum_v));
    *sum                 = vaddvq_s32(sum_diff);
    uint32x4_t sse_sum   = vaddq_u32(vaddq_u32(sse_u32[0], sse_u32[1]), vaddq_u32(sse_u32[2], sse_u32[3]));
    *sse                 = vaddvq_u32(sse_sum);
}

static inline void variance_32xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                              int h, uint32_t* sse, int* sum) {
    variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 32, h, sse, sum);
}

static inline void variance_64xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                              int h, uint32_t* sse, int* sum) {
    variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 64, h, sse, sum);
}

static inline void variance_128xh_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride,
                                               int h, uint32_t* sse, int* sum) {
    variance_large_neon_dotprod(src, src_stride, ref, ref_stride, 128, h, sse, sum);
}

#define VARIANCE_WXH_NEON_DOTPROD(w, h, shift)                                                       \
    unsigned int svt_aom_variance##w##x##h##_neon_dotprod(                                           \
        const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, unsigned int* sse) { \
        int sum;                                                                                     \
        variance_##w##xh_neon_dotprod(src, src_stride, ref, ref_stride, h, sse, &sum);               \
        return *sse - (uint32_t)(((int64_t)sum * sum) >> shift);                                     \
    }

// The Armv8.0 Neon implementation is faster than Neon DotProd for 4x4.
VARIANCE_WXH_NEON_DOTPROD(4, 8, 5)
VARIANCE_WXH_NEON_DOTPROD(4, 16, 6)

VARIANCE_WXH_NEON_DOTPROD(8, 4, 5)
VARIANCE_WXH_NEON_DOTPROD(8, 8, 6)
VARIANCE_WXH_NEON_DOTPROD(8, 16, 7)
VARIANCE_WXH_NEON_DOTPROD(8, 32, 8)

VARIANCE_WXH_NEON_DOTPROD(16, 8, 7)
VARIANCE_WXH_NEON_DOTPROD(16, 16, 8)
VARIANCE_WXH_NEON_DOTPROD(16, 32, 9)
VARIANCE_WXH_NEON_DOTPROD(16, 4, 6)
VARIANCE_WXH_NEON_DOTPROD(16, 64, 10)

VARIANCE_WXH_NEON_DOTPROD(32, 16, 9)
VARIANCE_WXH_NEON_DOTPROD(32, 32, 10)
VARIANCE_WXH_NEON_DOTPROD(32, 64, 11)
VARIANCE_WXH_NEON_DOTPROD(32, 8, 8)

VARIANCE_WXH_NEON_DOTPROD(64, 32, 11)
VARIANCE_WXH_NEON_DOTPROD(64, 64, 12)
VARIANCE_WXH_NEON_DOTPROD(64, 128, 13)
VARIANCE_WXH_NEON_DOTPROD(64, 16, 10)

VARIANCE_WXH_NEON_DOTPROD(128, 64, 13)
VARIANCE_WXH_NEON_DOTPROD(128, 128, 14)

#undef VARIANCE_WXH_NEON_DOTPROD

// =============================================================================
// Small-block sub_pixel_variance helpers (4xH) dotprod
// =============================================================================
#define SUBPEL_VARIANCE_4XH_NEON_DOTPROD(h, padding)                                       \
    unsigned int svt_aom_sub_pixel_variance4x##h##_neon_dotprod(const uint8_t* src,        \
                                                                int            src_stride, \
                                                                int            xoffset,    \
                                                                int            yoffset,    \
                                                                const uint8_t* ref,        \
                                                                int            ref_stride, \
                                                                uint32_t*      sse) {      \
        uint8_t tmp0[4 * (h + padding)];                                                   \
        uint8_t tmp1[4 * h];                                                               \
        var_filter_block2d_bil_w4(src, tmp0, src_stride, 1, (h + padding), xoffset);       \
        var_filter_block2d_bil_w4(tmp0, tmp1, 4, 4, h, yoffset);                           \
        return svt_aom_variance4x##h##_neon_dotprod(tmp1, 4, ref, ref_stride, sse);        \
    }

SUBPEL_VARIANCE_4XH_NEON_DOTPROD(8, 2)
SUBPEL_VARIANCE_4XH_NEON_DOTPROD(16, 2)

#undef SUBPEL_VARIANCE_4XH_NEON_DOTPROD

// Accumulate src and ref sums as well as the sum of squared error required to
// calculate variance.
static inline void var_w16_accum(uint32x4_t* src_sum, uint32x4_t* ref_sum, uint32x4_t* sse, uint8x16_t pred,
                                 uint8x16_t r) {
    *src_sum                  = vdotq_u32(*src_sum, pred, vdupq_n_u8(1));
    *ref_sum                  = vdotq_u32(*ref_sum, r, vdupq_n_u8(1));
    const uint8x16_t abs_diff = vabdq_u8(pred, r);
    *sse                      = vdotq_u32(*sse, abs_diff, abs_diff);
}

static inline void var_w8x2_accum(uint32x4_t* src_sum, uint32x4_t* ref_sum, uint32x4_t* sse, uint8x8_t pred0,
                                  uint8x8_t pred1, const uint8_t* ref, int64_t ref_stride) {
    const uint8x16_t pred = vcombine_u8(pred0, pred1);
    const uint8x16_t r    = vcombine_u8(vld1_u8(ref), vld1_u8(ref + ref_stride));
    var_w16_accum(src_sum, ref_sum, sse, pred, r);
}

// Calculate variance from src_sum, ref_sum and sse accumulator vectors.
static inline unsigned int var_w8_reduce(uint32x4_t src_sum, uint32x4_t ref_sum, uint32x4_t sse, int shift,
                                         unsigned int* sse_out) {
    const int32x4_t sum_diff = vreinterpretq_s32_u32(vsubq_u32(src_sum, ref_sum));
    const int32_t   sum      = vaddvq_s32(sum_diff);
    *sse_out                 = vaddvq_u32(sse);
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
                                                                    unsigned int*  sse_out) {  \
        uint32x4_t src_sum = vdupq_n_u32(0);                                                   \
        uint32x4_t ref_sum = vdupq_n_u32(0);                                                   \
        uint32x4_t sse     = vdupq_n_u32(0);                                                   \
        do {                                                                                   \
            const uint8x8_t pred0 = X_FILTER(src);                                             \
            const uint8x8_t pred1 = X_FILTER(src + src_stride);                                \
            var_w8x2_accum(&src_sum, &ref_sum, &sse, pred0, pred1, ref, ref_stride);           \
            src += 2 * src_stride;                                                             \
            ref += 2 * ref_stride;                                                             \
            h -= 2;                                                                            \
        } while (h != 0);                                                                      \
        return var_w8_reduce(src_sum, ref_sum, sse, shift, sse_out);                           \
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
                                                                                        unsigned int*  sse_out) {  \
        uint32x4_t src_sum = vdupq_n_u32(0);                                                                       \
        uint32x4_t ref_sum = vdupq_n_u32(0);                                                                       \
        uint32x4_t sse     = vdupq_n_u32(0);                                                                       \
        uint8x8_t  prev    = X_FILTER(src);                                                                        \
        src += src_stride;                                                                                         \
        do {                                                                                                       \
            const uint8x8_t cur0  = X_FILTER(src);                                                                 \
            const uint8x8_t pred0 = Y_FILTER(prev, cur0);                                                          \
            const uint8x8_t cur1  = X_FILTER(src + src_stride);                                                    \
            const uint8x8_t pred1 = Y_FILTER(cur0, cur1);                                                          \
            var_w8x2_accum(&src_sum, &ref_sum, &sse, pred0, pred1, ref, ref_stride);                               \
            prev = cur1;                                                                                           \
            src += 2 * src_stride;                                                                                 \
            ref += 2 * ref_stride;                                                                                 \
            h -= 2;                                                                                                \
        } while (h != 0);                                                                                          \
        return var_w8_reduce(src_sum, ref_sum, sse, shift, sse_out);                                               \
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

// Odd (1/8-pel) offsets are rare: compact builds keep one out-of-line copy instead of one per 8xN function.
#if CONFIG_COMPACT_KERNELS
#define SUBPEL_8XH_GENERIC_ATTR NOINLINE
#else
#define SUBPEL_8XH_GENERIC_ATTR
#endif
static SUBPEL_8XH_GENERIC_ATTR unsigned int fused_subpel_variance_8xh_generic(const uint8_t* src, int64_t src_stride,
                                                                              int h, int shift, const uint8_t* ref,
                                                                              int64_t ref_stride, unsigned int* sse_out,
                                                                              unsigned int offsets) {
    const int  xoffset = offsets & 7;
    const int  yoffset = (offsets >> 3) & 7;
    uint32x4_t src_sum = vdupq_n_u32(0);
    uint32x4_t ref_sum = vdupq_n_u32(0);
    uint32x4_t sse     = vdupq_n_u32(0);

    if (yoffset == 0) {
        do {
            const uint8x8_t pred0 = load_interp_w8(src, xoffset);
            const uint8x8_t pred1 = load_interp_w8(src + src_stride, xoffset);
            var_w8x2_accum(&src_sum, &ref_sum, &sse, pred0, pred1, ref, ref_stride);
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
            var_w16_accum(&src_sum, &ref_sum, &sse, pred, load_u8_8x2(ref, ref_stride));
            prev_pair = cur_pair;
            src += 2 * src_stride;
            ref += 2 * ref_stride;
            h -= 2;
        } while (h != 0);
    }

    return var_w8_reduce(src_sum, ref_sum, sse, shift, sse_out);
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

#if CONFIG_COMPACT_KERNELS
#undef FUSED_SUBPEL_VARIANCE_WXH_FN_TABLE
#endif // CONFIG_COMPACT_KERNELS

static unsigned int sub_pixel_variance_8xh_neon_dotprod(const uint8_t* src, int src_stride, int xoffset, int yoffset,
                                                        const uint8_t* ref, int ref_stride, int h, unsigned int* sse) {
    if (xoffset == 0 && yoffset == 0) {
        int sum;
        variance_8xh_neon_dotprod(src, src_stride, ref, ref_stride, h, sse, &sum);
        return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz(8 * h));
    }
    if ((xoffset | yoffset) & 1) {
        return fused_subpel_variance_8xh_generic(
            src, src_stride, h, svt_ctz(8 * h), ref, ref_stride, sse, (unsigned int)(xoffset | (yoffset << 3)));
    }
    const SubpelVarFn f = fused_subpel_variance_8xh_fn_table[xoffset >> 1][yoffset >> 1];
    return f(src, src_stride, h, svt_ctz(8 * h), ref, ref_stride, sse);
}

#define FUSED_SUBPEL_VARIANCE_8XH_NEON_DOTPROD(H)                                                                 \
    unsigned int svt_aom_sub_pixel_variance8x##H##_neon_dotprod(const uint8_t* src,                               \
                                                                int            src_stride,                        \
                                                                int            xoffset,                           \
                                                                int            yoffset,                           \
                                                                const uint8_t* ref,                               \
                                                                int            ref_stride,                        \
                                                                unsigned int*  sse) {                             \
        return sub_pixel_variance_8xh_neon_dotprod(src, src_stride, xoffset, yoffset, ref, ref_stride, (H), sse); \
    }

FUSED_SUBPEL_VARIANCE_8XH_NEON_DOTPROD(4)
FUSED_SUBPEL_VARIANCE_8XH_NEON_DOTPROD(8)
FUSED_SUBPEL_VARIANCE_8XH_NEON_DOTPROD(16)
FUSED_SUBPEL_VARIANCE_8XH_NEON_DOTPROD(32)

#undef FUSED_SUBPEL_VARIANCE_8XH_NEON_DOTPROD

#if CONFIG_COMPACT_KERNELS
// Calculate variance from two sets of src_sum, ref_sum and sse accumulator vectors.
static inline unsigned int var_w16x2_reduce(const uint32x4_t src_sum[2], const uint32x4_t ref_sum[2],
                                            const uint32x4_t sse[2], int shift, unsigned int* sse_out) {
    return var_w8_reduce(vaddq_u32(src_sum[0], src_sum[1]),
                         vaddq_u32(ref_sum[0], ref_sum[1]),
                         vaddq_u32(sse[0], sse[1]),
                         shift,
                         sse_out);
}

static inline void var_w16_accum_pair(uint32x4_t src_sum[2], uint32x4_t ref_sum[2], uint32x4_t sse[2], uint8x16_t pred0,
                                      uint8x16_t pred1, const uint8_t* ref, int64_t ref_stride) {
    var_w16_accum(&src_sum[0], &ref_sum[0], &sse[0], pred0, vld1q_u8(ref));
    var_w16_accum(&src_sum[1], &ref_sum[1], &sse[1], pred1, vld1q_u8(ref + ref_stride));
}

typedef unsigned int (*SubpelVarWxhFn)(const uint8_t* src, int64_t src_stride, int w, int h, int shift,
                                       const uint8_t* ref, int64_t ref_stride, unsigned int* sse);

// Emit a fused sub pixel variance function for blocks of width w (a multiple of 16) that uses the
// specified horizontal 2-tap filter. The block is processed one 16-wide column at a time, four rows
// per iteration.
#define FUSED_SUBPEL_VAR_WXH_X(X_FILTER_OFFSET, X_FILTER)                                                        \
    static unsigned int fused_subpel_variance_wxh_##X_FILTER_OFFSET(const uint8_t* src,                          \
                                                                    int64_t        src_stride,                   \
                                                                    int            w,                            \
                                                                    int            h,                            \
                                                                    int            shift,                        \
                                                                    const uint8_t* ref,                          \
                                                                    int64_t        ref_stride,                   \
                                                                    unsigned int*  sse_out) {                    \
        uint32x4_t src_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};                                                \
        uint32x4_t ref_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};                                                \
        uint32x4_t sse[2]     = {vdupq_n_u32(0), vdupq_n_u32(0)};                                                \
        for (int x = 0; x < w; x += 16) {                                                                        \
            const uint8_t* s = src + x;                                                                          \
            const uint8_t* r = ref + x;                                                                          \
            int            i = h;                                                                                \
            do {                                                                                                 \
                var_w16_accum_pair(src_sum, ref_sum, sse, X_FILTER(s), X_FILTER(s + src_stride), r, ref_stride); \
                var_w16_accum_pair(src_sum,                                                                      \
                                   ref_sum,                                                                      \
                                   sse,                                                                          \
                                   X_FILTER(s + 2 * src_stride),                                                 \
                                   X_FILTER(s + 3 * src_stride),                                                 \
                                   r + 2 * ref_stride,                                                           \
                                   ref_stride);                                                                  \
                s += 4 * src_stride;                                                                             \
                r += 4 * ref_stride;                                                                             \
                i -= 4;                                                                                          \
            } while (i != 0);                                                                                    \
        }                                                                                                        \
        return var_w16x2_reduce(src_sum, ref_sum, sse, shift, sse_out);                                          \
    }

FUSED_SUBPEL_VAR_WXH_X(2, load_interp_3_1_w16)
FUSED_SUBPEL_VAR_WXH_X(4, load_interp_1_1_w16)
FUSED_SUBPEL_VAR_WXH_X(6, load_interp_1_3_w16)

// Emit a fused sub pixel variance function for blocks of width w (a multiple of 16) that uses the
// specified 2-tap bilinear filter.
#define FUSED_SUBPEL_VAR_WXH_XY(X_FILTER_OFFSET, Y_FILTER_OFFSET, X_FILTER, Y_FILTER)                                 \
    static unsigned int fused_subpel_variance_wxh_##X_FILTER_OFFSET##_##Y_FILTER_OFFSET(const uint8_t* src,           \
                                                                                        int64_t        src_stride,    \
                                                                                        int            w,             \
                                                                                        int            h,             \
                                                                                        int            shift,         \
                                                                                        const uint8_t* ref,           \
                                                                                        int64_t        ref_stride,    \
                                                                                        unsigned int*  sse_out) {     \
        uint32x4_t src_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};                                                     \
        uint32x4_t ref_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};                                                     \
        uint32x4_t sse[2]     = {vdupq_n_u32(0), vdupq_n_u32(0)};                                                     \
        for (int x = 0; x < w; x += 16) {                                                                             \
            const uint8_t* s    = src + x;                                                                            \
            const uint8_t* r    = ref + x;                                                                            \
            int            i    = h;                                                                                  \
            uint8x16_t     prev = X_FILTER(s);                                                                        \
            s += src_stride;                                                                                          \
            do {                                                                                                      \
                const uint8x16_t cur0 = X_FILTER(s);                                                                  \
                const uint8x16_t cur1 = X_FILTER(s + src_stride);                                                     \
                const uint8x16_t cur2 = X_FILTER(s + 2 * src_stride);                                                 \
                const uint8x16_t cur3 = X_FILTER(s + 3 * src_stride);                                                 \
                var_w16_accum_pair(src_sum, ref_sum, sse, Y_FILTER(prev, cur0), Y_FILTER(cur0, cur1), r, ref_stride); \
                var_w16_accum_pair(src_sum,                                                                           \
                                   ref_sum,                                                                           \
                                   sse,                                                                               \
                                   Y_FILTER(cur1, cur2),                                                              \
                                   Y_FILTER(cur2, cur3),                                                              \
                                   r + 2 * ref_stride,                                                                \
                                   ref_stride);                                                                       \
                prev = cur3;                                                                                          \
                s += 4 * src_stride;                                                                                  \
                r += 4 * ref_stride;                                                                                  \
                i -= 4;                                                                                               \
            } while (i != 0);                                                                                         \
        }                                                                                                             \
        return var_w16x2_reduce(src_sum, ref_sum, sse, shift, sse_out);                                               \
    }

#define FUSED_SUBPEL_VAR_WXH(X_FILTER_OFFSET, X_FILTER)                   \
    FUSED_SUBPEL_VAR_WXH_XY(X_FILTER_OFFSET, 2, X_FILTER, interp_3_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(X_FILTER_OFFSET, 4, X_FILTER, interp_1_1_w16) \
    FUSED_SUBPEL_VAR_WXH_XY(X_FILTER_OFFSET, 6, X_FILTER, interp_1_3_w16)

FUSED_SUBPEL_VAR_WXH(0, vld1q_u8)
FUSED_SUBPEL_VAR_WXH(2, load_interp_3_1_w16)
FUSED_SUBPEL_VAR_WXH(4, load_interp_1_1_w16)
FUSED_SUBPEL_VAR_WXH(6, load_interp_1_3_w16)

#undef FUSED_SUBPEL_VAR_WXH_X
#undef FUSED_SUBPEL_VAR_WXH_XY
#undef FUSED_SUBPEL_VAR_WXH

// Fused sub pixel variance for blocks of width w (a multiple of 16) and any 2-tap bilinear filter.
static unsigned int fused_subpel_variance_wxh_generic(const uint8_t* src, int64_t src_stride, int w, int h, int shift,
                                                      const uint8_t* ref, int64_t ref_stride, unsigned int* sse_out,
                                                      int xoffset, int yoffset) {
    uint32x4_t src_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t ref_sum[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};
    uint32x4_t sse[2]     = {vdupq_n_u32(0), vdupq_n_u32(0)};

    for (int x = 0; x < w; x += 16) {
        const uint8_t* s = src + x;
        const uint8_t* r = ref + x;
        int            i = h;
        if (yoffset == 0) {
            do {
                var_w16_accum_pair(src_sum,
                                   ref_sum,
                                   sse,
                                   load_interp_w16(s, xoffset),
                                   load_interp_w16(s + src_stride, xoffset),
                                   r,
                                   ref_stride);
                s += 2 * src_stride;
                r += 2 * ref_stride;
                i -= 2;
            } while (i != 0);
        } else {
            uint8x16_t prev = load_interp_w16(s, xoffset);
            s += src_stride;
            do {
                const uint8x16_t cur0 = load_interp_w16(s, xoffset);
                const uint8x16_t cur1 = load_interp_w16(s + src_stride, xoffset);
                var_w16_accum_pair(src_sum,
                                   ref_sum,
                                   sse,
                                   interp_w16(prev, cur0, yoffset),
                                   interp_w16(cur0, cur1, yoffset),
                                   r,
                                   ref_stride);
                prev = cur1;
                s += 2 * src_stride;
                r += 2 * ref_stride;
                i -= 2;
            } while (i != 0);
        }
    }

    return var_w16x2_reduce(src_sum, ref_sum, sse, shift, sse_out);
}

static const SubpelVarWxhFn fused_subpel_variance_wxh_fn_table[4][4] = {
    {NULL, fused_subpel_variance_wxh_0_2, fused_subpel_variance_wxh_0_4, fused_subpel_variance_wxh_0_6},
    {fused_subpel_variance_wxh_2,
     fused_subpel_variance_wxh_2_2,
     fused_subpel_variance_wxh_2_4,
     fused_subpel_variance_wxh_2_6},
    {fused_subpel_variance_wxh_4,
     fused_subpel_variance_wxh_4_2,
     fused_subpel_variance_wxh_4_4,
     fused_subpel_variance_wxh_4_6},
    {fused_subpel_variance_wxh_6,
     fused_subpel_variance_wxh_6_2,
     fused_subpel_variance_wxh_6_4,
     fused_subpel_variance_wxh_6_6},
};

// Sub pixel variance for blocks of width w (a multiple of 16) at a non-zero filter offset. Even/even
// filter offsets dispatch through the table; odd offsets use the generic loop.
static AOM_FORCE_INLINE unsigned int fused_subpel_variance_wxh(const uint8_t* src, int src_stride, int xoffset,
                                                               int yoffset, const uint8_t* ref, int ref_stride, int w,
                                                               int h, int shift, unsigned int* sse) {
    if ((xoffset | yoffset) & 1) {
        return fused_subpel_variance_wxh_generic(src, src_stride, w, h, shift, ref, ref_stride, sse, xoffset, yoffset);
    }
    return fused_subpel_variance_wxh_fn_table[xoffset >> 1][yoffset >> 1](
        src, src_stride, w, h, shift, ref, ref_stride, sse);
}

// Emit svt_aom_sub_pixel_variance{W}x{H}_neon_dotprod for supported (W, H)
// pairs. Filter offset (0,0) uses the variance fast path. Shift is computed at
// compile time via svt_ctz((W) * (H)).
#define SUBPEL_VARIANCE_WXH_NEON_DOTPROD(W, H)                                                      \
    unsigned int svt_aom_sub_pixel_variance##W##x##H##_neon_dotprod(const uint8_t* src,             \
                                                                    int            src_stride,      \
                                                                    int            xoffset,         \
                                                                    int            yoffset,         \
                                                                    const uint8_t* ref,             \
                                                                    int            ref_stride,      \
                                                                    unsigned int*  sse) {           \
        if (xoffset == 0 && yoffset == 0) {                                                         \
            int sum;                                                                                \
            variance_##W##xh_neon_dotprod(src, src_stride, ref, ref_stride, (H), sse, &sum);        \
            return *sse - (uint32_t)(((int64_t)sum * sum) >> svt_ctz((W) * (H)));                   \
        }                                                                                           \
        return fused_subpel_variance_wxh(                                                           \
            src, src_stride, xoffset, yoffset, ref, ref_stride, (W), (H), svt_ctz((W) * (H)), sse); \
    }

SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 4)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 8)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 16)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 32)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 8)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 16)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 32)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 16)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 32)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 128)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(128, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(128, 128)

#undef SUBPEL_VARIANCE_WXH_NEON_DOTPROD
#else
// Number of accumulators for block width W.
#define ACCUM(W) (((W) / 16 >= 4) ? 4 : (W) / 16)

// Calculate variance from src_sum, ref_sum and sse accumulator vectors.
static inline int32_t var_reduce(const uint32x4_t* src_sum, const uint32x4_t* ref_sum, const uint32x4_t* sse, int n,
                                 unsigned int* sse_out) {
    uint32x4_t src_sum_v = src_sum[0];
    uint32x4_t ref_sum_v = ref_sum[0];
    uint32x4_t sse_sum   = sse[0];

    for (int i = 1; i < n; ++i) {
        src_sum_v = vaddq_u32(src_sum_v, src_sum[i]);
        ref_sum_v = vaddq_u32(ref_sum_v, ref_sum[i]);
        sse_sum   = vaddq_u32(sse_sum, sse[i]);
    }

    const int32x4_t sum_diff = vreinterpretq_s32_u32(vsubq_u32(src_sum_v, ref_sum_v));
    *sse_out                 = vaddvq_u32(sse_sum);
    return vaddvq_s32(sum_diff);
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
                                                                        unsigned int*  sse_out) {  \
        uint32x4_t src_sum[ACCUM(W)];                                                              \
        uint32x4_t ref_sum[ACCUM(W)];                                                              \
        uint32x4_t sse[ACCUM(W)];                                                                  \
        for (int i = 0; i < ACCUM(W); ++i) {                                                       \
            src_sum[i] = vdupq_n_u32(0);                                                           \
            ref_sum[i] = vdupq_n_u32(0);                                                           \
            sse[i]     = vdupq_n_u32(0);                                                           \
        }                                                                                          \
        do {                                                                                       \
            const uint8_t* sp_row = src;                                                           \
            const uint8_t* rp_row = ref;                                                           \
            for (int t = 0; t < W / 16; ++t) {                                                     \
                const uint8x16_t pred = X_FILTER(sp_row + t * 16);                                 \
                const uint8x16_t r    = vld1q_u8(rp_row + t * 16);                                 \
                const int        i    = t & (ACCUM(W) - 1);                                        \
                var_w16_accum(&src_sum[i], &ref_sum[i], &sse[i], pred, r);                         \
            }                                                                                      \
            src += src_stride;                                                                     \
            ref += ref_stride;                                                                     \
        } while (--h != 0);                                                                        \
        const int32_t sum = var_reduce(src_sum, ref_sum, sse, ACCUM(W), sse_out);                  \
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
                                                                                            unsigned int*  sse_out) {  \
        uint32x4_t src_sum[ACCUM(W)];                                                                                  \
        uint32x4_t ref_sum[ACCUM(W)];                                                                                  \
        uint32x4_t sse[ACCUM(W)];                                                                                      \
        uint8x16_t prev[W / 16];                                                                                       \
        uint8x16_t cur[W / 16];                                                                                        \
        for (int i = 0; i < ACCUM(W); ++i) {                                                                           \
            src_sum[i] = vdupq_n_u32(0);                                                                               \
            ref_sum[i] = vdupq_n_u32(0);                                                                               \
            sse[i]     = vdupq_n_u32(0);                                                                               \
        }                                                                                                              \
        for (int t = 0; t < W / 16; ++t) {                                                                             \
            prev[t] = X_FILTER(src + t * 16);                                                                          \
        }                                                                                                              \
        src += src_stride;                                                                                             \
        do {                                                                                                           \
            for (int t = 0; t < W / 16; ++t) {                                                                         \
                cur[t]                = X_FILTER(src + t * 16);                                                        \
                const uint8x16_t pred = Y_FILTER(prev[t], cur[t]);                                                     \
                const uint8x16_t r    = vld1q_u8(ref + t * 16);                                                        \
                const int        i    = t & (ACCUM(W) - 1);                                                            \
                var_w16_accum(&src_sum[i], &ref_sum[i], &sse[i], pred, r);                                             \
            }                                                                                                          \
            src += src_stride;                                                                                         \
            ref += ref_stride;                                                                                         \
            for (int t = 0; t < W / 16; ++t) {                                                                         \
                prev[t]               = X_FILTER(src + t * 16);                                                        \
                const uint8x16_t pred = Y_FILTER(cur[t], prev[t]);                                                     \
                const uint8x16_t r    = vld1q_u8(ref + t * 16);                                                        \
                const int        i    = t & (ACCUM(W) - 1);                                                            \
                var_w16_accum(&src_sum[i], &ref_sum[i], &sse[i], pred, r);                                             \
            }                                                                                                          \
            src += src_stride;                                                                                         \
            ref += ref_stride;                                                                                         \
            h -= 2;                                                                                                    \
        } while (h != 0);                                                                                              \
        const int32_t sum = var_reduce(src_sum, ref_sum, sse, ACCUM(W), sse_out);                                      \
        return *sse_out - (uint32_t)(((int64_t)sum * sum) >> shift);                                                   \
    }

// Emit a fused sub pixel variance function for the specified block width W. The
// function is generic in the sense that it can use any 2-tap bilinear filter.
#define FUSED_SUBPEL_VAR_WXH_GENERIC(W)                                                  \
    static unsigned int fused_subpel_variance_##W##xh_generic(const uint8_t* src,        \
                                                              int64_t        src_stride, \
                                                              int            h,          \
                                                              int            shift,      \
                                                              const uint8_t* ref,        \
                                                              int64_t        ref_stride, \
                                                              unsigned int*  sse_out,    \
                                                              unsigned int   offsets) {  \
        const int  xoffset = offsets & 7;                                                \
        const int  yoffset = (offsets >> 3) & 7;                                         \
        uint32x4_t src_sum[ACCUM(W)];                                                    \
        uint32x4_t ref_sum[ACCUM(W)];                                                    \
        uint32x4_t sse[ACCUM(W)];                                                        \
        for (int i = 0; i < ACCUM(W); ++i) {                                             \
            src_sum[i] = vdupq_n_u32(0);                                                 \
            ref_sum[i] = vdupq_n_u32(0);                                                 \
            sse[i]     = vdupq_n_u32(0);                                                 \
        }                                                                                \
        if (yoffset == 0) {                                                              \
            do {                                                                         \
                for (int t = 0; t < W / 16; ++t) {                                       \
                    const uint8x16_t pred = load_interp_w16(src + t * 16, xoffset);      \
                    const uint8x16_t r    = vld1q_u8(ref + t * 16);                      \
                    const int        i    = t & (ACCUM(W) - 1);                          \
                    var_w16_accum(&src_sum[i], &ref_sum[i], &sse[i], pred, r);           \
                }                                                                        \
                src += src_stride;                                                       \
                ref += ref_stride;                                                       \
            } while (--h != 0);                                                          \
        } else {                                                                         \
            uint8x16_t prev[W / 16];                                                     \
            for (int t = 0; t < W / 16; ++t) {                                           \
                prev[t] = load_interp_w16(src + t * 16, xoffset);                        \
            }                                                                            \
            src += src_stride;                                                           \
            do {                                                                         \
                for (int t = 0; t < W / 16; ++t) {                                       \
                    const uint8x16_t cur  = load_interp_w16(src + t * 16, xoffset);      \
                    const uint8x16_t pred = interp_w16(prev[t], cur, yoffset);           \
                    const uint8x16_t r    = vld1q_u8(ref + t * 16);                      \
                    const int        i    = t & (ACCUM(W) - 1);                          \
                    var_w16_accum(&src_sum[i], &ref_sum[i], &sse[i], pred, r);           \
                    prev[t] = cur;                                                       \
                }                                                                        \
                src += src_stride;                                                       \
                ref += ref_stride;                                                       \
            } while (--h != 0);                                                          \
        }                                                                                \
        const int32_t sum = var_reduce(src_sum, ref_sum, sse, ACCUM(W), sse_out);        \
        return *sse_out - (uint32_t)(((int64_t)sum * sum) >> shift);                     \
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

// Emit svt_aom_sub_pixel_variance{W}x{H}_neon_dotprod for supported (W, H)
// pairs. Filter offset (0,0) uses the variance fast path. Even/even filter
// offsets dispatch through width-specific tables; odd offsets use a generic
// loop. Shift is computed at compile time via svt_ctz((W) * (H)).
#define SUBPEL_VARIANCE_WXH_NEON_DOTPROD(W, H)                                                      \
    unsigned int svt_aom_sub_pixel_variance##W##x##H##_neon_dotprod(const uint8_t* src,             \
                                                                    int            src_stride,      \
                                                                    int            xoffset,         \
                                                                    int            yoffset,         \
                                                                    const uint8_t* ref,             \
                                                                    int            ref_stride,      \
                                                                    unsigned int*  sse) {           \
        if (xoffset == 0 && yoffset == 0) {                                                         \
            int sum;                                                                                \
            variance_##W##xh_neon_dotprod(src, src_stride, ref, ref_stride, (H), sse, &sum);        \
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

SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 4)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 8)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 16)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 32)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(16, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 8)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 16)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 32)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(32, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 16)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 32)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(64, 128)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(128, 64)
SUBPEL_VARIANCE_WXH_NEON_DOTPROD(128, 128)

#undef FUSED_SUBPEL_VARIANCE_WXH_NEON_DOTPROD
#undef ACCUM
#endif // CONFIG_COMPACT_KERNELS

unsigned int svt_aom_mse16x16_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride) {
    uint32x4_t sse_u32[2] = {vdupq_n_u32(0), vdupq_n_u32(0)};

    int h = 16;
    do {
        uint8x16_t s0 = vld1q_u8(src);
        uint8x16_t s1 = vld1q_u8(src + src_stride);
        uint8x16_t r0 = vld1q_u8(ref);
        uint8x16_t r1 = vld1q_u8(ref + ref_stride);

        uint8x16_t abs_diff0 = vabdq_u8(s0, r0);
        uint8x16_t abs_diff1 = vabdq_u8(s1, r1);

        sse_u32[0] = vdotq_u32(sse_u32[0], abs_diff0, abs_diff0);
        sse_u32[1] = vdotq_u32(sse_u32[1], abs_diff1, abs_diff1);

        src += 2 * src_stride;
        ref += 2 * ref_stride;
        h -= 2;
    } while (h != 0);

    unsigned int sse = vaddvq_u32(vaddq_u32(sse_u32[0], sse_u32[1]));
    return sse;
}

// Variances of one bw x bh source block against K horizontally adjacent reference blocks (ref + k * step),
// in the manner of the SAD x4 kernels: each source vector is loaded once and every reference position keeps its
// own accumulators, so the positions' dependency chains stay independent. src_total is the sum of the source
// block. 8-wide blocks (even heights) take two rows per vector.
static AOM_FORCE_INLINE void variance_xk_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref,
                                                      int ref_stride, const int bw, int bh, const int K, int step,
                                                      int shift, int32_t src_total, uint32_t* var, int var_stride) {
    const uint8x16_t ones = vdupq_n_u8(1);

    uint32x4_t ref_sum[5], sse[5];
    for (int k = 0; k < K; k++) {
        ref_sum[k] = vdupq_n_u32(0);
        sse[k]     = vdupq_n_u32(0);
    }
    if (bw == 8) {
        assert(!(bh & 1));
        for (int r = 0; r < bh; r += 2) {
            const uint8x16_t s = vcombine_u8(vld1_u8(src), vld1_u8(src + src_stride));
            for (int k = 0; k < K; k++) {
                const uint8x16_t p = vcombine_u8(vld1_u8(ref + k * step), vld1_u8(ref + ref_stride + k * step));
                const uint8x16_t d = vabdq_u8(s, p);

                ref_sum[k] = vdotq_u32(ref_sum[k], p, ones);

                sse[k] = vdotq_u32(sse[k], d, d);
            }
            src += 2 * src_stride;
            ref += 2 * ref_stride;
        }
    } else {
        for (int r = 0; r < bh; r++) {
            for (int c = 0; c < bw; c += 16) {
                const uint8x16_t s = vld1q_u8(src + c);
                for (int k = 0; k < K; k++) {
                    const uint8x16_t p = vld1q_u8(ref + k * step + c);
                    const uint8x16_t d = vabdq_u8(s, p);

                    ref_sum[k] = vdotq_u32(ref_sum[k], p, ones);

                    sse[k] = vdotq_u32(sse[k], d, d);
                }
            }
            src += src_stride;
            ref += ref_stride;
        }
    }
    // Positions 0..3 (0..2 for K == 3) are finished together, one lane per position.
    const uint32x4_t zero = vdupq_n_u32(0);
    const uint32x4_t e4   = vpaddq_u32(vpaddq_u32(sse[0], sse[1]), vpaddq_u32(sse[2], K > 3 ? sse[3] : zero));
    const uint32x4_t r4   = vpaddq_u32(vpaddq_u32(ref_sum[0], ref_sum[1]),
                                     vpaddq_u32(ref_sum[2], K > 3 ? ref_sum[3] : zero));
    const int32x4_t  sum4 = vsubq_s32(vdupq_n_s32(src_total), vreinterpretq_s32_u32(r4));
    const int64x2_t  nsh  = vdupq_n_s64(-shift);
    const uint64x2_t q_lo = vshlq_u64(vreinterpretq_u64_s64(vmull_s32(vget_low_s32(sum4), vget_low_s32(sum4))), nsh);
    const uint64x2_t q_hi = vshlq_u64(vreinterpretq_u64_s64(vmull_high_s32(sum4, sum4)), nsh);
    const uint32x4_t var4 = vsubq_u32(e4, vcombine_u32(vmovn_u64(q_lo), vmovn_u64(q_hi)));
    vst1q_lane_u32(var + 0 * var_stride, var4, 0);
    vst1q_lane_u32(var + 1 * var_stride, var4, 1);
    vst1q_lane_u32(var + 2 * var_stride, var4, 2);
    if (K > 3) {
        vst1q_lane_u32(var + 3 * var_stride, var4, 3);
    }
    for (int k = 4; k < K; k++) {
        const uint32_t e    = vaddvq_u32(sse[k]);
        const int64_t  sum  = (int64_t)src_total - (int64_t)vaddvq_u32(ref_sum[k]);
        var[k * var_stride] = e - (uint32_t)((sum * sum) >> shift);
    }
}

#define VARIANCE_XK(w, k)                                                                                              \
    static NOINLINE void variance_##w##_x##k##_neon_dotprod(const uint8_t* src,                                        \
                                                            int            src_stride,                                 \
                                                            const uint8_t* ref,                                        \
                                                            int            ref_stride,                                 \
                                                            int            bh,                                         \
                                                            int            step,                                       \
                                                            int            shift,                                      \
                                                            int32_t        src_total,                                  \
                                                            uint32_t*      var,                                        \
                                                            int            var_stride) {                                          \
        variance_xk_neon_dotprod(src, src_stride, ref, ref_stride, w, bh, k, step, shift, src_total, var, var_stride); \
    }
VARIANCE_XK(8, 3)
VARIANCE_XK(8, 5)
VARIANCE_XK(16, 3)
VARIANCE_XK(16, 5)
VARIANCE_XK(32, 3)
VARIANCE_XK(32, 5)
VARIANCE_XK(64, 3)
VARIANCE_XK(64, 5)

// Sum of a bw x bh block (bw 8 or a multiple of 16).
static inline int32_t block_sum_neon_dotprod(const uint8_t* src, int src_stride, int bw, int bh) {
    const uint8x16_t ones = vdupq_n_u8(1);

    uint32x4_t sum = vdupq_n_u32(0);
    for (int r = 0; r < bh; r++) {
        if (bw == 8) {
            sum = vdotq_u32(sum, vcombine_u8(vld1_u8(src), vdup_n_u8(0)), ones);
        } else {
            for (int c = 0; c < bw; c += 16) {
                sum = vdotq_u32(sum, vld1q_u8(src + c), ones);
            }
        }
        src += src_stride;
    }
    return (int32_t)vaddvq_u32(sum);
}

// Variances of the bw x bh source block against the reference blocks of an nx x ny grid of full-pel positions
// (nx is 3 or 5, bw is 8, 16, 32 or 64): var[i * ny + j] belongs to ref + (j * step) * ref_stride + i * step
// and equals svt_aom_varianceWxH(that block, src).
// The same variances from sums that are cheaper to accumulate. With s the source block and r a reference block,
//     sum((s - r)^2) = sum(s^2) + sum(r^2) - 2 * sum(s * r),
// so a position only needs its correlation sum(s * r) over the block: one dot product per vector instead of an
// absolute difference and two dot products. sum(r) and sum(r^2) of the K positions of a grid row follow from the
// previous grid row's by adding the rows that entered the window and removing those that left it (kept as two
// running accumulators each, entered and left, so nothing is ever subtracted lane by lane).
// 8-wide blocks take two rows per vector in the correlation; single rows are zero-extended.
static AOM_FORCE_INLINE uint8x16_t grid_row8(const uint8_t* p) {
    return vcombine_u8(vld1_u8(p), vdup_n_u8(0));
}

static AOM_FORCE_INLINE void variance_grid_corr_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref,
                                                             int ref_stride, const int bw, int bh, const int K, int ny,
                                                             int step, uint32_t* var) {
    const uint8x16_t ones = vdupq_n_u8(1);

    const int shift = 31 - __builtin_clz((unsigned)(bw * bh));
    // source sum and sum of squares
    uint32x4_t s_sum_v = vdupq_n_u32(0), s_sq_v = vdupq_n_u32(0);
    for (int r = 0; r < bh; r++) {
        if (bw == 8) {
            const uint8x16_t sv = grid_row8(src + r * src_stride);

            s_sum_v = vdotq_u32(s_sum_v, sv, ones);

            s_sq_v = vdotq_u32(s_sq_v, sv, sv);
        } else {
            for (int c = 0; c < bw; c += 16) {
                const uint8x16_t sv = vld1q_u8(src + r * src_stride + c);

                s_sum_v = vdotq_u32(s_sum_v, sv, ones);

                s_sq_v = vdotq_u32(s_sq_v, sv, sv);
            }
        }
    }
    const uint32_t s_sum = vaddvq_u32(s_sum_v);
    const uint32_t s_sq  = vaddvq_u32(s_sq_v);
    // reference rows that entered (in) and left (out) the window so far, per horizontal position
    uint32x4_t in_sum[5], in_sq[5], out_sum[5], out_sq[5];
    for (int k = 0; k < K; k++) {
        in_sum[k] = in_sq[k] = out_sum[k] = out_sq[k] = vdupq_n_u32(0);
    }
    for (int j = 0; j < ny; j++) {
        const uint8_t* p = ref + j * step * ref_stride;
        uint32x4_t     corr[5];
        for (int k = 0; k < K; k++) {
            corr[k] = vdupq_n_u32(0);
        }
        // rows above the window that the previous grid row still covered
        for (int r = j ? -step : 0; r < 0; r++) {
            if (bw == 8) {
                for (int k = 0; k < K; k++) {
                    const uint8x16_t v = grid_row8(p + r * ref_stride + k * step);

                    out_sum[k] = vdotq_u32(out_sum[k], v, ones);

                    out_sq[k] = vdotq_u32(out_sq[k], v, v);
                }
            } else {
                for (int c = 0; c < bw; c += 16) {
                    for (int k = 0; k < K; k++) {
                        const uint8x16_t v = vld1q_u8(p + r * ref_stride + k * step + c);

                        out_sum[k] = vdotq_u32(out_sum[k], v, ones);

                        out_sq[k] = vdotq_u32(out_sq[k], v, v);
                    }
                }
            }
        }
        // rows of this window: the first `fresh` of them were already counted by the previous grid row
        const int fresh = j ? bh - step : 0;
        if (bw == 8) {
            assert(!(bh & 1));
            int r = 0;
            for (; r + 1 < fresh; r += 2) { // both rows already counted
                const uint8x16_t sv = vcombine_u8(vld1_u8(src + r * src_stride), vld1_u8(src + (r + 1) * src_stride));
                for (int k = 0; k < K; k++) {
                    const uint8x16_t v = vcombine_u8(vld1_u8(p + r * ref_stride + k * step),
                                                     vld1_u8(p + (r + 1) * ref_stride + k * step));

                    corr[k] = vdotq_u32(corr[k], sv, v);
                }
            }
            if (r < fresh) { // the second row of the pair is new
                const uint8x16_t sv = vcombine_u8(vld1_u8(src + r * src_stride), vld1_u8(src + (r + 1) * src_stride));
                for (int k = 0; k < K; k++) {
                    const uint8x8_t  v1 = vld1_u8(p + (r + 1) * ref_stride + k * step);
                    const uint8x16_t v  = vcombine_u8(vld1_u8(p + r * ref_stride + k * step), v1);
                    const uint8x16_t n  = vcombine_u8(vdup_n_u8(0), v1);

                    corr[k] = vdotq_u32(corr[k], sv, v);

                    in_sum[k] = vdotq_u32(in_sum[k], n, ones);

                    in_sq[k] = vdotq_u32(in_sq[k], n, n);
                }
                r += 2;
            }
            for (; r < bh; r += 2) { // both rows new
                const uint8x16_t sv = vcombine_u8(vld1_u8(src + r * src_stride), vld1_u8(src + (r + 1) * src_stride));
                for (int k = 0; k < K; k++) {
                    const uint8x16_t v = vcombine_u8(vld1_u8(p + r * ref_stride + k * step),
                                                     vld1_u8(p + (r + 1) * ref_stride + k * step));

                    corr[k] = vdotq_u32(corr[k], sv, v);

                    in_sum[k] = vdotq_u32(in_sum[k], v, ones);

                    in_sq[k] = vdotq_u32(in_sq[k], v, v);
                }
            }
        } else {
            for (int r = 0; r < fresh; r++) {
                for (int c = 0; c < bw; c += 16) {
                    const uint8x16_t sv = vld1q_u8(src + r * src_stride + c);
                    for (int k = 0; k < K; k++) {
                        corr[k] = vdotq_u32(corr[k], sv, vld1q_u8(p + r * ref_stride + k * step + c));
                    }
                }
            }
            for (int r = fresh; r < bh; r++) {
                for (int c = 0; c < bw; c += 16) {
                    const uint8x16_t sv = vld1q_u8(src + r * src_stride + c);
                    for (int k = 0; k < K; k++) {
                        const uint8x16_t v = vld1q_u8(p + r * ref_stride + k * step + c);

                        corr[k] = vdotq_u32(corr[k], sv, v);

                        in_sum[k] = vdotq_u32(in_sum[k], v, ones);

                        in_sq[k] = vdotq_u32(in_sq[k], v, v);
                    }
                }
            }
        }
        // Positions 0..3 (0..2 for K == 3) are finished together: one lane per position.
        const uint32x4_t zero = vdupq_n_u32(0);
        const uint32x4_t c3   = K > 3 ? corr[3] : zero;
        const uint32x4_t rs3  = K > 3 ? vsubq_u32(in_sum[3], out_sum[3]) : zero;
        const uint32x4_t rq3  = K > 3 ? vsubq_u32(in_sq[3], out_sq[3]) : zero;
        const uint32x4_t c4   = vpaddq_u32(vpaddq_u32(corr[0], corr[1]), vpaddq_u32(corr[2], c3));
        const uint32x4_t rs4  = vpaddq_u32(
            vpaddq_u32(vsubq_u32(in_sum[0], out_sum[0]), vsubq_u32(in_sum[1], out_sum[1])),
            vpaddq_u32(vsubq_u32(in_sum[2], out_sum[2]), rs3));
        const uint32x4_t rq4  = vpaddq_u32(vpaddq_u32(vsubq_u32(in_sq[0], out_sq[0]), vsubq_u32(in_sq[1], out_sq[1])),
                                          vpaddq_u32(vsubq_u32(in_sq[2], out_sq[2]), rq3));
        const uint32x4_t sse4 = vsubq_u32(vaddq_u32(vdupq_n_u32(s_sq), rq4), vshlq_n_u32(c4, 1));
        const int32x4_t  sum4 = vsubq_s32(vdupq_n_s32((int32_t)s_sum), vreinterpretq_s32_u32(rs4));
        const int64x2_t  nsh  = vdupq_n_s64(-shift);
        const uint64x2_t q_lo = vshlq_u64(vreinterpretq_u64_s64(vmull_s32(vget_low_s32(sum4), vget_low_s32(sum4))),
                                          nsh);
        const uint64x2_t q_hi = vshlq_u64(vreinterpretq_u64_s64(vmull_high_s32(sum4, sum4)), nsh);
        const uint32x4_t var4 = vsubq_u32(sse4, vcombine_u32(vmovn_u64(q_lo), vmovn_u64(q_hi)));
        vst1q_lane_u32(var + 0 * ny + j, var4, 0);
        vst1q_lane_u32(var + 1 * ny + j, var4, 1);
        vst1q_lane_u32(var + 2 * ny + j, var4, 2);
        if (K > 3) {
            vst1q_lane_u32(var + 3 * ny + j, var4, 3);
        }
        for (int k = 4; k < K; k++) {
            const uint32_t r_sum = vaddvq_u32(vsubq_u32(in_sum[k], out_sum[k]));
            const uint32_t r_sq  = vaddvq_u32(vsubq_u32(in_sq[k], out_sq[k]));
            const uint32_t sse   = s_sq + r_sq - 2 * vaddvq_u32(corr[k]);
            const int64_t  sum   = (int64_t)s_sum - (int64_t)r_sum;
            var[k * ny + j]      = sse - (uint32_t)((sum * sum) >> shift);
        }
    }
}

#define VARIANCE_GRID_CORR(w, k)                                                                    \
    static NOINLINE void variance_grid_corr_##w##_x##k##_neon_dotprod(const uint8_t* src,           \
                                                                      int            src_stride,    \
                                                                      const uint8_t* ref,           \
                                                                      int            ref_stride,    \
                                                                      int            bh,            \
                                                                      int            ny,            \
                                                                      int            step,          \
                                                                      uint32_t*      var) {              \
        variance_grid_corr_neon_dotprod(src, src_stride, ref, ref_stride, w, bh, k, ny, step, var); \
    }
VARIANCE_GRID_CORR(16, 3)
VARIANCE_GRID_CORR(16, 5)
VARIANCE_GRID_CORR(32, 3)
VARIANCE_GRID_CORR(32, 5)
VARIANCE_GRID_CORR(64, 3)
VARIANCE_GRID_CORR(64, 5)

void svt_aom_variance_grid_neon_dotprod(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int bw,
                                        int bh, int nx, int ny, int step, uint32_t* var) {
    assert((nx == 3 || nx == 5) && (bw == 8 || bw == 16 || bw == 32 || bw == 64));
    // (The sliding sums assume consecutive grid rows overlap.)
    // 8-wide blocks are faster through the per-row kernels below: their cost is in forming the vectors.
    if (step <= bh && bw >= 16) {
        if (nx == 3) {
            switch (bw) {
            case 16:
                variance_grid_corr_16_x3_neon_dotprod(src, src_stride, ref, ref_stride, bh, ny, step, var);
                break;
            case 32:
                variance_grid_corr_32_x3_neon_dotprod(src, src_stride, ref, ref_stride, bh, ny, step, var);
                break;
            default:
                variance_grid_corr_64_x3_neon_dotprod(src, src_stride, ref, ref_stride, bh, ny, step, var);
                break;
            }
        } else {
            switch (bw) {
            case 16:
                variance_grid_corr_16_x5_neon_dotprod(src, src_stride, ref, ref_stride, bh, ny, step, var);
                break;
            case 32:
                variance_grid_corr_32_x5_neon_dotprod(src, src_stride, ref, ref_stride, bh, ny, step, var);
                break;
            default:
                variance_grid_corr_64_x5_neon_dotprod(src, src_stride, ref, ref_stride, bh, ny, step, var);
                break;
            }
        }
        return;
    }
    const int     shift = 31 - __builtin_clz((unsigned)(bw * bh));
    const int32_t st    = block_sum_neon_dotprod(src, src_stride, bw, bh);
    for (int j = 0; j < ny; j++) {
        const uint8_t* p = ref + j * step * ref_stride;
        uint32_t*      v = var + j;
        if (nx == 3) {
            switch (bw) {
            case 8:
                variance_8_x3_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            case 16:
                variance_16_x3_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            case 32:
                variance_32_x3_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            default:
                variance_64_x3_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            }
        } else {
            switch (bw) {
            case 8:
                variance_8_x5_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            case 16:
                variance_16_x5_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            case 32:
                variance_32_x5_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            default:
                variance_64_x5_neon_dotprod(src, src_stride, p, ref_stride, bh, step, shift, st, v, ny);
                break;
            }
        }
    }
}

// Variances of the w x h source block against the four half-pel bilinear predictions around the full-pel
// position c: left (c - 1/2, .), right (c + 1/2, .), up (., c - 1/2), down (., c + 1/2), in that order. Each
// equals svt_aom_sub_pixel_variance{w}x{h}() for that position (a half-pel bilinear sample is the rounded mean of
// its two neighbours); the source row, its sum and the centre row are loaded once for the four of them.
// w is 8, 16 or 32 (wider blocks are faster through the per-position kernels) and h at most 64.
void svt_aom_sub_pixel_variance_cross_half_neon_dotprod(const uint8_t* c, int c_stride, const uint8_t* src,
                                                        int src_stride, int w, int h, uint32_t var[4]) {
    const uint8x16_t ones = vdupq_n_u8(1);

    uint32x4_t sse[4], psum[4];

    uint32x4_t ssum = vdupq_n_u32(0);
    for (int k = 0; k < 4; k++) {
        sse[k]  = vdupq_n_u32(0);
        psum[k] = vdupq_n_u32(0);
    }
#define CROSS_HALF_ACCUM(k, a, b)                               \
    do {                                                        \
        const uint8x16_t pred = vrhaddq_u8(a, b);               \
        const uint8x16_t d    = vabdq_u8(pred, s);              \
        psum[k]               = vdotq_u32(psum[k], pred, ones); \
        sse[k]                = vdotq_u32(sse[k], d, d);        \
    } while (0)
    if (w == 8) {
        const uint8x8_t zero = vdup_n_u8(0);
        for (int r = 0; r < h; r++) {
            const uint8x16_t s  = vcombine_u8(vld1_u8(src), zero);
            const uint8x16_t cc = vcombine_u8(vld1_u8(c), zero);
            const uint8x16_t cl = vcombine_u8(vld1_u8(c - 1), zero);
            const uint8x16_t cr = vcombine_u8(vld1_u8(c + 1), zero);
            const uint8x16_t cu = vcombine_u8(vld1_u8(c - c_stride), zero);
            const uint8x16_t cd = vcombine_u8(vld1_u8(c + c_stride), zero);

            ssum = vdotq_u32(ssum, s, ones);
            CROSS_HALF_ACCUM(0, cl, cc);
            CROSS_HALF_ACCUM(1, cc, cr);
            CROSS_HALF_ACCUM(2, cu, cc);
            CROSS_HALF_ACCUM(3, cc, cd);
            c += c_stride;
            src += src_stride;
        }
    } else {
        for (int r = 0; r < h; r++) {
            for (int x = 0; x < w; x += 16) {
                const uint8x16_t s  = vld1q_u8(src + x);
                const uint8x16_t cc = vld1q_u8(c + x);
                const uint8x16_t cl = vld1q_u8(c + x - 1);
                const uint8x16_t cr = vld1q_u8(c + x + 1);
                const uint8x16_t cu = vld1q_u8(c + x - c_stride);
                const uint8x16_t cd = vld1q_u8(c + x + c_stride);

                ssum = vdotq_u32(ssum, s, ones);
                CROSS_HALF_ACCUM(0, cl, cc);
                CROSS_HALF_ACCUM(1, cc, cr);
                CROSS_HALF_ACCUM(2, cu, cc);
                CROSS_HALF_ACCUM(3, cc, cd);
            }
            c += c_stride;
            src += src_stride;
        }
    }
#undef CROSS_HALF_ACCUM
    const int     shift   = 31 - __builtin_clz((unsigned)(w * h));
    const int32_t s_total = (int32_t)vaddvq_u32(ssum);
    for (int k = 0; k < 4; k++) {
        const uint32_t e   = vaddvq_u32(sse[k]);
        const int64_t  sum = (int64_t)vaddvq_u32(psum[k]) - s_total;

        var[k] = e - (uint32_t)((sum * sum) >> shift);
    }
}

// Sixteen samples at p (eight, zero-extended, for 8-wide blocks).
static AOM_FORCE_INLINE uint8x16_t cross_load(const uint8_t* p, const bool w8) {
    return w8 ? vcombine_u8(vld1_u8(p), vdup_n_u8(0)) : vld1q_u8(p);
}

// Bilinear blend of a and b at offset o eighths (o is 2, 4 or 6).
static AOM_FORCE_INLINE uint8x16_t cross_blend(uint8x16_t a, uint8x16_t b, const int o) {
    return o == 2 ? interp_3_1_w16(a, b) : o == 4 ? interp_1_1_w16(a, b) : interp_1_3_w16(a, b);
}

// Horizontal pass at offset xo eighths (0, 2, 4 or 6) of the samples at p.
static AOM_FORCE_INLINE uint8x16_t cross_hpass(const uint8_t* p, const int xo, const bool w8) {
    const uint8x16_t a = cross_load(p, w8);
    return xo == 0 ? a : cross_blend(a, cross_load(p + 1, w8), xo);
}

// The four quarter-pel positions around a centre whose sub-pel part (sx, sy) is 0 or 4 eighths on each axis; c
// points at the centre's integer position. Left / right move the horizontal offset by two eighths and keep the
// vertical one, up / down the reverse, so up and down share their horizontal pass and, for a half-pel row centre,
// every position blends the same two filtered rows.
static AOM_FORCE_INLINE void cross_quarter_impl(const uint8_t* c, int c_stride, const uint8_t* src, int src_stride,
                                                int w, int h, uint32_t var[4], const int sx, const int sy) {
    const bool w8   = w == 8;
    const int  cols = w8 ? 1 : w / 16;
    const int  lx = sx == 0 ? -1 : 0, lxo = sx == 0 ? 6 : 2, rxo = sx == 0 ? 2 : 6;

    const uint8x16_t ones = vdupq_n_u8(1);

    uint32x4_t sse[4], psum[4];

    uint32x4_t ssum = vdupq_n_u32(0);
    for (int k = 0; k < 4; k++) {
        sse[k]  = vdupq_n_u32(0);
        psum[k] = vdupq_n_u32(0);
    }
    uint8x16_t hc_prev[2], hc[2], hl[2], hr[2];
    for (int t = 0; t < cols; t++) {
        hc[t] = cross_hpass(c + t * 16, sx, w8);
        if (sy == 0) {
            hc_prev[t] = cross_hpass(c - c_stride + t * 16, sx, w8);
        } else {
            hl[t] = cross_hpass(c + lx + t * 16, lxo, w8);
            hr[t] = cross_hpass(c + t * 16, rxo, w8);
        }
    }
#define CROSS_Q_ACCUM(k, p)                                     \
    do {                                                        \
        const uint8x16_t pred = p;                              \
        const uint8x16_t d    = vabdq_u8(pred, s);              \
        psum[k]               = vdotq_u32(psum[k], pred, ones); \
        sse[k]                = vdotq_u32(sse[k], d, d);        \
    } while (0)
    for (int r = 0; r < h; r++) {
        for (int t = 0; t < cols; t++) {
            const uint8x16_t s = cross_load(src + t * 16, w8);

            const uint8x16_t hc_next = cross_hpass(c + c_stride + t * 16, sx, w8);

            ssum = vdotq_u32(ssum, s, ones);
            if (sy == 0) {
                CROSS_Q_ACCUM(0, cross_hpass(c + lx + t * 16, lxo, w8));
                CROSS_Q_ACCUM(1, cross_hpass(c + t * 16, rxo, w8));
                CROSS_Q_ACCUM(2, cross_blend(hc_prev[t], hc[t], 6));
                CROSS_Q_ACCUM(3, cross_blend(hc[t], hc_next, 2));
                hc_prev[t] = hc[t];
            } else {
                const uint8x16_t hl_next = cross_hpass(c + c_stride + lx + t * 16, lxo, w8);
                const uint8x16_t hr_next = cross_hpass(c + c_stride + t * 16, rxo, w8);
                CROSS_Q_ACCUM(0, cross_blend(hl[t], hl_next, 4));
                CROSS_Q_ACCUM(1, cross_blend(hr[t], hr_next, 4));
                CROSS_Q_ACCUM(2, cross_blend(hc[t], hc_next, 2));
                CROSS_Q_ACCUM(3, cross_blend(hc[t], hc_next, 6));
                hl[t] = hl_next;
                hr[t] = hr_next;
            }
            hc[t] = hc_next;
        }
        c += c_stride;
        src += src_stride;
    }
#undef CROSS_Q_ACCUM
    const int     shift   = 31 - __builtin_clz((unsigned)(w * h));
    const int32_t s_total = (int32_t)vaddvq_u32(ssum);
    for (int k = 0; k < 4; k++) {
        const uint32_t e   = vaddvq_u32(sse[k]);
        const int64_t  sum = (int64_t)vaddvq_u32(psum[k]) - s_total;

        var[k] = e - (uint32_t)((sum * sum) >> shift);
    }
}

// Variances of the w x h source block against the bilinear predictions at (sx - 2, sy), (sx + 2, sy), (sx, sy - 2)
// and (sx, sy + 2) eighths from the integer position c, in that order, where sx and sy are each 0 or 4. Each
// equals svt_aom_sub_pixel_variance{w}x{h}() for that position. w is 8, 16 or 32 and h at most 64.
void svt_aom_sub_pixel_variance_cross_quarter_neon_dotprod(const uint8_t* c, int c_stride, int sx, int sy,
                                                           const uint8_t* src, int src_stride, int w, int h,
                                                           uint32_t var[4]) {
    if (sy == 0) {
        if (sx == 0) {
            cross_quarter_impl(c, c_stride, src, src_stride, w, h, var, 0, 0);
        } else {
            cross_quarter_impl(c, c_stride, src, src_stride, w, h, var, 4, 0);
        }
    } else if (sx == 0) {
        cross_quarter_impl(c, c_stride, src, src_stride, w, h, var, 0, 4);
    } else {
        cross_quarter_impl(c, c_stride, src, src_stride, w, h, var, 4, 4);
    }
}
