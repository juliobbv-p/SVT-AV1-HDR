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

#include "common_dsp_rtcd.h"
#include "convolve.h"
#include "definitions.h"
#include "mem_neon.h"

#define WIENER_WIN 7
#define WIENER_WIN_REDUCED 5

static inline uint16x8_t highbd_wiener_convolve5_8_2d_h(const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
                                                        const int16x8_t s3, const int16x8_t s4,
                                                        const int16x4_t x_filter, const int32x4_t round_vec,
                                                        const uint16x8_t im_max_val) {
    /* Wiener filter is symmetric so add mirrored source elements. */
    int16x8_t s04 = vaddq_s16(s0, s4);
    int16x8_t s13 = vaddq_s16(s1, s3);

    /* x_filter[0] = 0. (5-tap filters are 0-padded to 7 taps.) */
    int32x4_t sum_lo = vmlal_lane_s16(round_vec, vget_low_s16(s04), x_filter, 1);
    sum_lo           = vmlal_lane_s16(sum_lo, vget_low_s16(s13), x_filter, 2);
    sum_lo           = vmlal_lane_s16(sum_lo, vget_low_s16(s2), x_filter, 3);

    int32x4_t sum_hi = vmlal_lane_s16(round_vec, vget_high_s16(s04), x_filter, 1);
    sum_hi           = vmlal_lane_s16(sum_hi, vget_high_s16(s13), x_filter, 2);
    sum_hi           = vmlal_lane_s16(sum_hi, vget_high_s16(s2), x_filter, 3);

    uint16x4_t res_lo = vqrshrun_n_s32(sum_lo, WIENER_ROUND0_BITS);
    uint16x4_t res_hi = vqrshrun_n_s32(sum_hi, WIENER_ROUND0_BITS);

    return vminq_u16(vcombine_u16(res_lo, res_hi), im_max_val);
}

static inline void highbd_convolve_add_src_5tap_horiz(const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
                                                      ptrdiff_t dst_stride, int w, int h, const int16x4_t x_filter,
                                                      const int32x4_t round_vec, const uint16x8_t im_max_val) {
    do {
        const int16_t* s     = (int16_t*)src_ptr;
        uint16_t*      d     = dst_ptr;
        int            width = w;

        do {
            int16x8_t s0, s1, s2, s3, s4;
            load_s16_8x5(s, 1, &s0, &s1, &s2, &s3, &s4);

            uint16x8_t d0 = highbd_wiener_convolve5_8_2d_h(s0, s1, s2, s3, s4, x_filter, round_vec, im_max_val);

            vst1q_u16(d, d0);

            s += 8;
            d += 8;
            width -= 8;
        } while (width != 0);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    } while (--h != 0);
}

static inline uint16x8_t highbd_wiener_convolve7_8_2d_h(const int16x8_t s0, const int16x8_t s1, const int16x8_t s2,
                                                        const int16x8_t s3, const int16x8_t s4, const int16x8_t s5,
                                                        const int16x8_t s6, const int16x4_t x_filter,
                                                        const int32x4_t round_vec, const uint16x8_t im_max_val) {
    /* Wiener filter is symmetric so add mirrored source elements. */
    int16x8_t s06 = vaddq_s16(s0, s6);
    int16x8_t s15 = vaddq_s16(s1, s5);
    int16x8_t s24 = vaddq_s16(s2, s4);

    int32x4_t sum_lo = vmlal_lane_s16(round_vec, vget_low_s16(s06), x_filter, 0);
    sum_lo           = vmlal_lane_s16(sum_lo, vget_low_s16(s15), x_filter, 1);
    sum_lo           = vmlal_lane_s16(sum_lo, vget_low_s16(s24), x_filter, 2);
    sum_lo           = vmlal_lane_s16(sum_lo, vget_low_s16(s3), x_filter, 3);

    int32x4_t sum_hi = vmlal_lane_s16(round_vec, vget_high_s16(s06), x_filter, 0);
    sum_hi           = vmlal_lane_s16(sum_hi, vget_high_s16(s15), x_filter, 1);
    sum_hi           = vmlal_lane_s16(sum_hi, vget_high_s16(s24), x_filter, 2);
    sum_hi           = vmlal_lane_s16(sum_hi, vget_high_s16(s3), x_filter, 3);

    uint16x4_t res_lo = vqrshrun_n_s32(sum_lo, WIENER_ROUND0_BITS);
    uint16x4_t res_hi = vqrshrun_n_s32(sum_hi, WIENER_ROUND0_BITS);

    return vminq_u16(vcombine_u16(res_lo, res_hi), im_max_val);
}

static inline void highbd_convolve_add_src_7tap_horiz(const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
                                                      ptrdiff_t dst_stride, int w, int h, const int16x4_t x_filter,
                                                      const int32x4_t round_vec, const uint16x8_t im_max_val) {
    do {
        const int16_t* s     = (int16_t*)src_ptr;
        uint16_t*      d     = dst_ptr;
        int            width = w;

        do {
            int16x8_t s0, s1, s2, s3, s4, s5, s6;
            load_s16_8x7(s, 1, &s0, &s1, &s2, &s3, &s4, &s5, &s6);

            uint16x8_t d0 = highbd_wiener_convolve7_8_2d_h(s0, s1, s2, s3, s4, s5, s6, x_filter, round_vec, im_max_val);

            vst1q_u16(d, d0);

            s += 8;
            d += 8;
            width -= 8;
        } while (width != 0);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    } while (--h != 0);
}

// Vertical pass. The intermediate is clamped to [0, 2^(bd + 1 + FILTER_BITS - WIENER_ROUND0_BITS) - 1], at most
// 32767 for the bit depths this file supports (8 and 10). A row minus 16384 therefore lies within [-16384, 16383]
// and the sum of two mirrored rows within [-32768, 32766]: the pair sums fit 16 signed bits, and one widening
// multiply-accumulate per half vector takes each of them. The rows arrive with the 16384 already removed; it
// comes back through the rounding vector as 16384 * (sum of the seven taps), see the caller.
// r points at the first of the `taps` rows.
static AOM_FORCE_INLINE uint16x8_t highbd_wiener_convolve_8_2d_v(const int16x8_t* r, const int16x4_t y_filter,
                                                                 const int32x4_t  round_vec,
                                                                 const uint16x8_t res_max_val, const int taps) {
    int32x4_t sum_lo = round_vec, sum_hi = round_vec;
    if (taps == WIENER_WIN) {
        const int16x8_t s06 = vaddq_s16(r[0], r[6]);

        sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s06), y_filter, 0);

        sum_hi = vmlal_high_lane_s16(sum_hi, s06, y_filter, 0);
        r += 1;
    }
    /* Wiener filter is symmetric so add mirrored source elements. (5-tap filters are 0-padded to 7 taps.) */
    const int16x8_t s04 = vaddq_s16(r[0], r[4]);
    const int16x8_t s13 = vaddq_s16(r[1], r[3]);

    sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s04), y_filter, 1);

    sum_hi = vmlal_high_lane_s16(sum_hi, s04, y_filter, 1);

    sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(s13), y_filter, 2);

    sum_hi = vmlal_high_lane_s16(sum_hi, s13, y_filter, 2);

    sum_lo = vmlal_lane_s16(sum_lo, vget_low_s16(r[2]), y_filter, 3);

    sum_hi = vmlal_high_lane_s16(sum_hi, r[2], y_filter, 3);

    uint16x4_t res_lo = vqrshrun_n_s32(sum_lo, 2 * FILTER_BITS - WIENER_ROUND0_BITS);
    uint16x4_t res_hi = vqrshrun_n_s32(sum_hi, 2 * FILTER_BITS - WIENER_ROUND0_BITS);

    return vminq_u16(vcombine_u16(res_lo, res_hi), res_max_val);
}

#define WIENER_V_BIAS 16384

// round_vec must include WIENER_V_BIAS * (sum of the seven taps).
static AOM_FORCE_INLINE void highbd_convolve_add_src_vert(const uint16_t* src_ptr, ptrdiff_t src_stride,
                                                          uint16_t* dst_ptr, ptrdiff_t dst_stride, int w, int h,
                                                          const int16x4_t y_filter, const int32x4_t round_vec,
                                                          const uint16x8_t res_max_val, const int taps) {
    const int16x8_t bias = vdupq_n_s16(WIENER_V_BIAS);
    do {
        const int16_t* s      = (const int16_t*)src_ptr;
        uint16_t*      d      = dst_ptr;
        int            height = h;
        int16x8_t      r[WIENER_WIN + 3];
        for (int i = 0; i < taps - 1; i++) {
            r[i] = vsubq_s16(vld1q_s16(s + i * src_stride), bias);
        }
        s += (taps - 1) * src_stride;

        while (height > 3) {
            for (int i = 0; i < 4; i++) {
                r[taps - 1 + i] = vsubq_s16(vld1q_s16(s + i * src_stride), bias);
            }
            uint16x8_t d0 = highbd_wiener_convolve_8_2d_v(r + 0, y_filter, round_vec, res_max_val, taps);
            uint16x8_t d1 = highbd_wiener_convolve_8_2d_v(r + 1, y_filter, round_vec, res_max_val, taps);
            uint16x8_t d2 = highbd_wiener_convolve_8_2d_v(r + 2, y_filter, round_vec, res_max_val, taps);
            uint16x8_t d3 = highbd_wiener_convolve_8_2d_v(r + 3, y_filter, round_vec, res_max_val, taps);

            store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

            for (int i = 0; i < taps - 1; i++) {
                r[i] = r[i + 4];
            }
            s += 4 * src_stride;
            d += 4 * dst_stride;
            height -= 4;
        }

        while (height-- != 0) {
            r[taps - 1] = vsubq_s16(vld1q_s16(s), bias);
            vst1q_u16(d, highbd_wiener_convolve_8_2d_v(r, y_filter, round_vec, res_max_val, taps));
            for (int i = 0; i < taps - 1; i++) {
                r[i] = r[i + 1];
            }
            s += src_stride;
            d += dst_stride;
        }

        src_ptr += 8;
        dst_ptr += 8;
        w -= 8;
    } while (w != 0);
}

static inline void highbd_convolve_add_src_5tap_vert(const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
                                                     ptrdiff_t dst_stride, int w, int h, const int16x4_t y_filter,
                                                     const int32x4_t round_vec, const uint16x8_t res_max_val) {
    highbd_convolve_add_src_vert(
        src_ptr, src_stride, dst_ptr, dst_stride, w, h, y_filter, round_vec, res_max_val, WIENER_WIN_REDUCED);
}

static inline void highbd_convolve_add_src_7tap_vert(const uint16_t* src_ptr, ptrdiff_t src_stride, uint16_t* dst_ptr,
                                                     ptrdiff_t dst_stride, int w, int h, const int16x4_t y_filter,
                                                     const int32x4_t round_vec, const uint16x8_t res_max_val) {
    highbd_convolve_add_src_vert(
        src_ptr, src_stride, dst_ptr, dst_stride, w, h, y_filter, round_vec, res_max_val, WIENER_WIN);
}

static inline int get_wiener_filter_taps(const int16_t* filter) {
    assert(filter[7] == 0);
    if (filter[0] == 0 && filter[6] == 0) {
        return WIENER_WIN_REDUCED;
    }
    return WIENER_WIN;
}

void svt_av1_highbd_wiener_convolve_add_src_neon(const uint8_t* const src8, ptrdiff_t src_stride, uint8_t* dst8,
                                                 ptrdiff_t dst_stride, const int16_t* const x_filter,
                                                 const int16_t* const y_filter, int w, int h,
                                                 const ConvolveParams* conv_params, int bd) {
    (void)conv_params;
    assert(w % 8 == 0);
    assert(w <= MAX_SB_SIZE && h <= MAX_SB_SIZE);
    assert(x_filter[7] == 0 && y_filter[7] == 0);

    DECLARE_ALIGNED(16, uint16_t, im_block[(MAX_SB_SIZE + WIENER_WIN - 1) * MAX_SB_SIZE]);

    const int x_filter_taps = get_wiener_filter_taps(x_filter);
    const int y_filter_taps = get_wiener_filter_taps(y_filter);
    int16x4_t x_filter_s16  = vld1_s16(x_filter);
    int16x4_t y_filter_s16  = vld1_s16(y_filter);
    // Add 128 to tap 3. (Needed for rounding.)
    x_filter_s16 = vadd_s16(x_filter_s16, vcreate_s16(128ULL << 48));
    y_filter_s16 = vadd_s16(y_filter_s16, vcreate_s16(128ULL << 48));

    const int im_stride    = MAX_SB_SIZE;
    const int im_h         = h + y_filter_taps - 1;
    const int horiz_offset = x_filter_taps / 2;
    const int vert_offset  = (y_filter_taps / 2) * (int)src_stride;

    const int        extraprec_clamp_limit = WIENER_CLAMP_LIMIT(WIENER_ROUND0_BITS, bd);
    const uint16x8_t im_max_val            = vdupq_n_u16(extraprec_clamp_limit - 1);
    const int32x4_t  horiz_round_vec       = vdupq_n_s32(1 << (bd + FILTER_BITS - 1));

    const uint16x8_t res_max_val = vdupq_n_u16((1 << bd) - 1);
    // The vertical pass takes its rows minus WIENER_V_BIAS (their clamp limit must not exceed twice that); the
    // seven taps, tap 3 with its added 128, weigh the bias back in.
    assert(extraprec_clamp_limit <= 2 * WIENER_V_BIAS);
    const int32_t   y_tap_sum      = 2 * (y_filter[0] + y_filter[1] + y_filter[2]) + y_filter[3] + 128;
    const int32x4_t vert_round_vec = vdupq_n_s32(WIENER_V_BIAS * y_tap_sum -
                                                 (1 << (bd + 2 * FILTER_BITS - WIENER_ROUND0_BITS - 1)));

    uint16_t* src = CONVERT_TO_SHORTPTR(src8);
    uint16_t* dst = CONVERT_TO_SHORTPTR(dst8);

    if (x_filter_taps == WIENER_WIN_REDUCED) {
        highbd_convolve_add_src_5tap_horiz(src - horiz_offset - vert_offset,
                                           src_stride,
                                           im_block,
                                           im_stride,
                                           w,
                                           im_h,
                                           x_filter_s16,
                                           horiz_round_vec,
                                           im_max_val);
    } else {
        highbd_convolve_add_src_7tap_horiz(src - horiz_offset - vert_offset,
                                           src_stride,
                                           im_block,
                                           im_stride,
                                           w,
                                           im_h,
                                           x_filter_s16,
                                           horiz_round_vec,
                                           im_max_val);
    }

    if (y_filter_taps == WIENER_WIN_REDUCED) {
        highbd_convolve_add_src_5tap_vert(
            im_block, im_stride, dst, dst_stride, w, h, y_filter_s16, vert_round_vec, res_max_val);
    } else {
        highbd_convolve_add_src_7tap_vert(
            im_block, im_stride, dst, dst_stride, w, h, y_filter_s16, vert_round_vec, res_max_val);
    }
}
