/*
 * Copyright (c) 2022, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

/**
 * See svt_av1_wedge_sse_from_residuals_c for details of the parameters and
 * computation.
 */

#include <arm_neon.h>
#include <assert.h>
#include <stdbool.h>

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "sum_neon.h"

uint64_t svt_av1_wedge_sse_from_residuals_neon(const int16_t* r1, const int16_t* d, const uint8_t* m, int N) {
    assert(N % 64 == 0);

    uint64x2_t v_csse[2] = {vdupq_n_u64(0), vdupq_n_u64(0)};

    int i = 0;
    do {
        int32x4_t sum[4];
        int32x4_t sse[2];
        int16x4_t sum_s16[4];

        const int16x8_t r1_l = vld1q_s16(r1 + i);
        const int16x8_t r1_h = vld1q_s16(r1 + i + 8);
        const int16x8_t d_l  = vld1q_s16(d + i);
        const int16x8_t d_h  = vld1q_s16(d + i + 8);
        // The following three lines are a bit inelegant compared to using a pair
        // of vmovl_u8()... but it forces the compiler to emit a ZIP1, ZIP2 pair -
        // which can be executed in parallel with the subsequent SSHL instructions.
        // (SSHL can only be executed on half of the Neon pipes in modern Arm
        // cores, whereas ZIP1/2 can be executed on all of them.)
        const uint8x16x2_t m_u16 = vzipq_u8(vld1q_u8(m + i), vdupq_n_u8(0));
        const int16x8_t    m_l   = vreinterpretq_s16_u8(m_u16.val[0]);
        const int16x8_t    m_h   = vreinterpretq_s16_u8(m_u16.val[1]);

        sum[0] = vshll_n_s16(vget_low_s16(r1_l), WEDGE_WEIGHT_BITS);
        sum[1] = vshll_n_s16(vget_high_s16(r1_l), WEDGE_WEIGHT_BITS);
        sum[2] = vshll_n_s16(vget_low_s16(r1_h), WEDGE_WEIGHT_BITS);
        sum[3] = vshll_n_s16(vget_high_s16(r1_h), WEDGE_WEIGHT_BITS);

        sum[0] = vmlal_s16(sum[0], vget_low_s16(m_l), vget_low_s16(d_l));
        sum[1] = vmlal_s16(sum[1], vget_high_s16(m_l), vget_high_s16(d_l));
        sum[2] = vmlal_s16(sum[2], vget_low_s16(m_h), vget_low_s16(d_h));
        sum[3] = vmlal_s16(sum[3], vget_high_s16(m_h), vget_high_s16(d_h));

        sum_s16[0] = vqmovn_s32(sum[0]);
        sum_s16[1] = vqmovn_s32(sum[1]);
        sum_s16[2] = vqmovn_s32(sum[2]);
        sum_s16[3] = vqmovn_s32(sum[3]);

        sse[0] = vmull_s16(sum_s16[0], sum_s16[0]);
        sse[1] = vmull_s16(sum_s16[2], sum_s16[2]);
        sse[0] = vmlal_s16(sse[0], sum_s16[1], sum_s16[1]);
        sse[1] = vmlal_s16(sse[1], sum_s16[3], sum_s16[3]);

        v_csse[0] = vpadalq_u32(v_csse[0], vreinterpretq_u32_s32(sse[0]));
        v_csse[1] = vpadalq_u32(v_csse[1], vreinterpretq_u32_s32(sse[1]));

        i += 16;
    } while (i < N);

    uint64_t csse = vaddvq_u64(vaddq_u64(v_csse[0], v_csse[1]));
    return ROUND_POWER_OF_TWO(csse, 2 * WEDGE_WEIGHT_BITS);
}

// True when every r1[i] and d[i] is at most 255 in magnitude: then |64 * r1 + m * d| <= 32640 for masks up to 64,
// which is what svt_av1_wedge_sse_from_residuals_small_neon() needs.
bool svt_av1_wedge_residuals_small_neon(const int16_t* r1, const int16_t* d, int N) {
    assert(N % 16 == 0);
    uint16x8_t m0 = vdupq_n_u16(0), m1 = m0, m2 = m0, m3 = m0;
    for (int i = 0; i < N; i += 16) {
        m0 = vmaxq_u16(m0, vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(r1 + i))));
        m1 = vmaxq_u16(m1, vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(r1 + i + 8))));
        m2 = vmaxq_u16(m2, vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(d + i))));
        m3 = vmaxq_u16(m3, vreinterpretq_u16_s16(vabsq_s16(vld1q_s16(d + i + 8))));
    }
    return vmaxvq_u16(vmaxq_u16(vmaxq_u16(m0, m1), vmaxq_u16(m2, m3))) <= 255;
}

// svt_av1_wedge_sse_from_residuals_neon() for residuals accepted by svt_av1_wedge_residuals_small_neon(): the
// weighted residual 64 * r1 + m * d then fits 16 bits, so it is formed in 16-bit arithmetic and the saturating
// narrowing of the general kernel (a no-op in that range) is not needed.
uint64_t svt_av1_wedge_sse_from_residuals_small_neon(const int16_t* r1, const int16_t* d, const uint8_t* m, int N) {
    assert(N % 64 == 0);

    uint64x2_t v_csse[2] = {vdupq_n_u64(0), vdupq_n_u64(0)};

    int i = 0;
    do {
        const int16x8_t    r1_l  = vld1q_s16(r1 + i);
        const int16x8_t    r1_h  = vld1q_s16(r1 + i + 8);
        const int16x8_t    d_l   = vld1q_s16(d + i);
        const int16x8_t    d_h   = vld1q_s16(d + i + 8);
        const uint8x16x2_t m_u16 = vzipq_u8(vld1q_u8(m + i), vdupq_n_u8(0));
        const int16x8_t    m_l   = vreinterpretq_s16_u8(m_u16.val[0]);
        const int16x8_t    m_h   = vreinterpretq_s16_u8(m_u16.val[1]);

        const int16x8_t t_l = vmlaq_s16(vshlq_n_s16(r1_l, WEDGE_WEIGHT_BITS), m_l, d_l);
        const int16x8_t t_h = vmlaq_s16(vshlq_n_s16(r1_h, WEDGE_WEIGHT_BITS), m_h, d_h);

        // two squares of at most 32640^2 per lane: below 2^32
        int32x4_t sse0 = vmull_s16(vget_low_s16(t_l), vget_low_s16(t_l));
        int32x4_t sse1 = vmull_s16(vget_low_s16(t_h), vget_low_s16(t_h));

        sse0 = vmlal_high_s16(sse0, t_l, t_l);

        sse1 = vmlal_high_s16(sse1, t_h, t_h);

        v_csse[0] = vpadalq_u32(v_csse[0], vreinterpretq_u32_s32(sse0));
        v_csse[1] = vpadalq_u32(v_csse[1], vreinterpretq_u32_s32(sse1));

        i += 16;
    } while (i < N);

    uint64_t csse = vaddvq_u64(vaddq_u64(v_csse[0], v_csse[1]));
    return ROUND_POWER_OF_TWO(csse, 2 * WEDGE_WEIGHT_BITS);
}

int8_t svt_av1_wedge_sign_from_residuals_neon(const int16_t* ds, const uint8_t* m, int N, int64_t limit) {
    int64x2_t sum;
    int32x4_t acc[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};

    do {
        int16x8_t ds_l = vld1q_s16(ds);
        int16x8_t ds_h = vld1q_s16(ds + 8);

        int8x16_t m_s8 = vreinterpretq_s8_u8(vld1q_u8(m));
        int16x8_t m_l  = vmovl_s8(vget_low_s8(m_s8));
        int16x8_t m_h  = vmovl_s8(vget_high_s8(m_s8));

        acc[0] = vmlal_s16(acc[0], vget_low_s16(ds_l), vget_low_s16(m_l));
        acc[1] = vmlal_s16(acc[1], vget_high_s16(ds_l), vget_high_s16(m_l));
        acc[2] = vmlal_s16(acc[2], vget_low_s16(ds_h), vget_low_s16(m_h));
        acc[3] = vmlal_s16(acc[3], vget_high_s16(ds_h), vget_high_s16(m_h));

        ds += 16;
        m += 16;
        N -= 16;
    } while (N != 0);

    sum = vpaddlq_s32(acc[0]);
    sum = vpadalq_s32(sum, acc[1]);
    sum = vpadalq_s32(sum, acc[2]);
    sum = vpadalq_s32(sum, acc[3]);

    return (vaddvq_s64(sum) > limit);
}

void svt_av1_wedge_compute_delta_squares_neon(int16_t* d_ptr, const int16_t* a_ptr, const int16_t* b_ptr, int N) {
    do {
        int16x8_t a = vld1q_s16(a_ptr);
        int16x8_t b = vld1q_s16(b_ptr);

        int32x4_t sq_lo = vmull_s16(vget_low_s16(a), vget_low_s16(a));
        int32x4_t sq_hi = vmull_s16(vget_high_s16(a), vget_high_s16(a));

        sq_lo = vmlsl_s16(sq_lo, vget_low_s16(b), vget_low_s16(b));
        sq_hi = vmlsl_s16(sq_hi, vget_high_s16(b), vget_high_s16(b));

        int16x8_t res = vcombine_s16(vqmovn_s32(sq_lo), vqmovn_s32(sq_hi));

        vst1q_s16(d_ptr, res);

        d_ptr += 8;
        a_ptr += 8;
        b_ptr += 8;
        N -= 8;
    } while (N != 0);
}
