/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef HIGHBD_10_CONVOLVE_NEON_H_
#define HIGHBD_10_CONVOLVE_NEON_H_

#include "definitions.h"

#if defined(ARCH_AARCH64) && CONFIG_ENABLE_HIGH_BIT_DEPTH
#include <arm_neon.h>
#include <assert.h>

#include "mem_neon.h"

// All AV1 interpolation coefficients are even and sum to 128. Split the
// half-filter into a simple center predictor B and a zero-sum residual G.
// Choose B=(64,0), (0,64), or (32,32) so the positive and negative weight sums of G have magnitude <=31.
// For ten-bit pixels every partial residual MAC is bounded by 31*1023=31713.
// Since B's weights are divisible by four, the original horizontal pass is
// exactly 8192 + B*pixels/4 + round(G*pixels/4). The result and all intermediate
// sums fit signed 16 bits. Keep the vertical pass and its rounding unchanged.

static inline uint16x8_t highbd_10_convolve6_8_h(const int16x8_t s[6], int16x8_t filter, int16x4_t base,
                                                 int16x8_t offset) {
    const int16x4_t f0    = vget_low_s16(filter);
    const int16x4_t f1    = vget_high_s16(filter);
    int16x8_t       delta = vmulq_lane_s16(s[0], f0, 1);
    delta                 = vmlaq_lane_s16(delta, s[1], f0, 2);
    delta                 = vmlaq_lane_s16(delta, s[2], f0, 3);
    delta                 = vmlaq_lane_s16(delta, s[3], f1, 0);
    delta                 = vmlaq_lane_s16(delta, s[4], f1, 1);
    delta                 = vmlaq_lane_s16(delta, s[5], f1, 2);
    int16x8_t predictor   = vmulq_lane_s16(s[2], base, 0);
    predictor             = vmlaq_lane_s16(predictor, s[3], base, 1);
    return vreinterpretq_u16_s16(vaddq_s16(offset, vaddq_s16(predictor, vrshrq_n_s16(delta, 2))));
}

static inline void highbd_10_convolve_2d_horiz_6tap_neon(const uint16_t* src_ptr, int src_stride, uint16_t* dst_ptr,
                                                         int dst_stride, int w, int h, int16x8_t x_filter,
                                                         int16x4_t base) {
    // The smallest block height is 4, and the horizontal convolution needs to
    // process an extra (filter_taps/2 - 1) lines for the vertical convolution.
    assert(h >= 5);

    int height = h;

    do {
        int            width = w;
        const int16_t* s     = (const int16_t*)src_ptr;
        uint16_t*      d     = dst_ptr;

        do {
            int16x8_t s0[6], s1[6], s2[6], s3[6];
            load_s16_8x6(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3], &s0[4], &s0[5]);
            load_s16_8x6(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3], &s1[4], &s1[5]);
            load_s16_8x6(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3], &s2[4], &s2[5]);
            load_s16_8x6(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3], &s3[4], &s3[5]);

            uint16x8_t d0 = highbd_10_convolve6_8_h(s0, x_filter, base, vdupq_n_s16(8192));
            uint16x8_t d1 = highbd_10_convolve6_8_h(s1, x_filter, base, vdupq_n_s16(8192));
            uint16x8_t d2 = highbd_10_convolve6_8_h(s2, x_filter, base, vdupq_n_s16(8192));
            uint16x8_t d3 = highbd_10_convolve6_8_h(s3, x_filter, base, vdupq_n_s16(8192));

            store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

            s += 8;
            d += 8;
            width -= 8;
        } while (width != 0);
        src_ptr += 4 * src_stride;
        dst_ptr += 4 * dst_stride;
        height -= 4;
    } while (height > 4);

    do {
        int            width = w;
        const int16_t* s     = (const int16_t*)src_ptr;
        uint16_t*      d     = dst_ptr;

        do {
            int16x8_t s0[6];
            load_s16_8x6(s, 1, &s0[0], &s0[1], &s0[2], &s0[3], &s0[4], &s0[5]);

            uint16x8_t d0 = highbd_10_convolve6_8_h(s0, x_filter, base, vdupq_n_s16(8192));
            vst1q_u16(d, d0);

            s += 8;
            d += 8;
            width -= 8;
        } while (width != 0);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    } while (--height != 0);
}

static inline uint16x8_t highbd_10_convolve8_8_h(const int16x8_t s[8], int16x8_t filter, int16x4_t base,
                                                 int16x8_t offset) {
    const int16x4_t f0    = vget_low_s16(filter);
    const int16x4_t f1    = vget_high_s16(filter);
    int16x8_t       delta = vmulq_lane_s16(s[0], f0, 0);
    delta                 = vmlaq_lane_s16(delta, s[1], f0, 1);
    delta                 = vmlaq_lane_s16(delta, s[2], f0, 2);
    delta                 = vmlaq_lane_s16(delta, s[3], f0, 3);
    delta                 = vmlaq_lane_s16(delta, s[4], f1, 0);
    delta                 = vmlaq_lane_s16(delta, s[5], f1, 1);
    delta                 = vmlaq_lane_s16(delta, s[6], f1, 2);
    delta                 = vmlaq_lane_s16(delta, s[7], f1, 3);
    int16x8_t predictor   = vmulq_lane_s16(s[3], base, 0);
    predictor             = vmlaq_lane_s16(predictor, s[4], base, 1);
    return vreinterpretq_u16_s16(vaddq_s16(offset, vaddq_s16(predictor, vrshrq_n_s16(delta, 2))));
}

static inline void highbd_10_convolve_2d_horiz_8tap_neon(const uint16_t* src_ptr, int src_stride, uint16_t* dst_ptr,
                                                         int dst_stride, int w, int h, int16x8_t x_filter,
                                                         int16x4_t base) {
    // The smallest block height is 4, and the horizontal convolution needs to
    // process an extra (filter_taps/2 - 1) lines for the vertical convolution.
    assert(h >= 5);

    int height = h;

    do {
        int            width = w;
        const int16_t* s     = (const int16_t*)src_ptr;
        uint16_t*      d     = dst_ptr;

        do {
            int16x8_t s0[8], s1[8], s2[8], s3[8];
            load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3], &s0[4], &s0[5], &s0[6], &s0[7]);
            load_s16_8x8(s + 1 * src_stride, 1, &s1[0], &s1[1], &s1[2], &s1[3], &s1[4], &s1[5], &s1[6], &s1[7]);
            load_s16_8x8(s + 2 * src_stride, 1, &s2[0], &s2[1], &s2[2], &s2[3], &s2[4], &s2[5], &s2[6], &s2[7]);
            load_s16_8x8(s + 3 * src_stride, 1, &s3[0], &s3[1], &s3[2], &s3[3], &s3[4], &s3[5], &s3[6], &s3[7]);

            uint16x8_t d0 = highbd_10_convolve8_8_h(s0, x_filter, base, vdupq_n_s16(8192));
            uint16x8_t d1 = highbd_10_convolve8_8_h(s1, x_filter, base, vdupq_n_s16(8192));
            uint16x8_t d2 = highbd_10_convolve8_8_h(s2, x_filter, base, vdupq_n_s16(8192));
            uint16x8_t d3 = highbd_10_convolve8_8_h(s3, x_filter, base, vdupq_n_s16(8192));

            store_u16_8x4(d, dst_stride, d0, d1, d2, d3);

            s += 8;
            d += 8;
            width -= 8;
        } while (width != 0);
        src_ptr += 4 * src_stride;
        dst_ptr += 4 * dst_stride;
        height -= 4;
    } while (height > 4);

    do {
        int            width = w;
        const int16_t* s     = (const int16_t*)src_ptr;
        uint16_t*      d     = dst_ptr;

        do {
            int16x8_t s0[8];
            load_s16_8x8(s + 0 * src_stride, 1, &s0[0], &s0[1], &s0[2], &s0[3], &s0[4], &s0[5], &s0[6], &s0[7]);

            uint16x8_t d0 = highbd_10_convolve8_8_h(s0, x_filter, base, vdupq_n_s16(8192));
            vst1q_u16(d, d0);

            s += 8;
            d += 8;
            width -= 8;
        } while (width != 0);
        src_ptr += src_stride;
        dst_ptr += dst_stride;
    } while (--height != 0);
}

static inline void highbd_10_convolve_2d_horiz_neon(const uint16_t* src, int src_stride, uint16_t* dst, int dst_stride,
                                                    int w, int h, const int16_t* filter, int taps) {
    assert(w >= 8 && !(w & 7) && h >= 5);
    const int       mode       = filter[3] >= 110 ? 0 : filter[4] >= 110 ? 2 : 1;
    const int16x8_t correction = mode == 0 ? (int16x8_t){0, 0, 0, 64, 0, 0, 0, 0}
        : mode == 2                        ? (int16x8_t){0, 0, 0, 0, 64, 0, 0, 0}
                                           : (int16x8_t){0, 0, 0, 32, 32, 0, 0, 0};
    const int16x4_t base       = mode == 0 ? (int16x4_t){16, 0, 0, 0}
              : mode == 2                  ? (int16x4_t){0, 16, 0, 0}
                                           : (int16x4_t){8, 8, 0, 0};
    const int16x8_t residual   = vsubq_s16(vshrq_n_s16(vld1q_s16(filter), 1), correction);
    if (taps <= 6) {
        highbd_10_convolve_2d_horiz_6tap_neon(src, src_stride, dst, dst_stride, w, h, residual, base);
    } else {
        highbd_10_convolve_2d_horiz_8tap_neon(src, src_stride, dst, dst_stride, w, h, residual, base);
    }
}
#endif // ARCH_AARCH64 && CONFIG_ENABLE_HIGH_BIT_DEPTH
#endif // HIGHBD_10_CONVOLVE_NEON_H_
