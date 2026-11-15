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

#include <arm_neon.h>
#include <assert.h>

#include "common_dsp_rtcd.h"
#include "restoration.h"
#include "definitions.h"
#include "mem_neon.h"
#include "transpose_neon.h"

// Constants used for right shift in final_filter calculation.
#define NB_EVEN 5
#define NB_ODD 4

// 256 - svt_aom_eb_x_by_xplus1[z] for z < 48. For larger z the value is 5 minus the number of the thresholds
// 55, 72, 101, 169 and 254 that z exceeds (checked against the full table for all 256 indices).
static const uint8_t sgr_256_minus_x_by_xplus1_lo[48] = {
    255, 128, 85, 64, 51, 43, 37, 32, 28, 26, 23, 21, 20, 18, 17, 16, 15, 14, 13, 13, 12, 12, 11, 11,
    10,  10,  9,  9,  9,  9,  8,  8,  8,  8,  7,  7,  7,  7,  7,  6,  6,  6,  6,  6,  6,  6,  5,  5,
};

static inline uint8x16_t sgr_256_minus_x_by_xplus1(uint8x16_t z) {
    const uint8x16x3_t tbl = {{vld1q_u8(sgr_256_minus_x_by_xplus1_lo),
                               vld1q_u8(sgr_256_minus_x_by_xplus1_lo + 16),
                               vld1q_u8(sgr_256_minus_x_by_xplus1_lo + 32)}};
    const uint8x16_t   lo  = vqtbl3q_u8(tbl, z); // 0 for z >= 48
    // Each compare yields 0xff (-1) where z exceeds the threshold.
    uint8x16_t hi = vaddq_u8(vdupq_n_u8(5), vcgtq_u8(z, vdupq_n_u8(55)));

    hi = vaddq_u8(hi, vcgtq_u8(z, vdupq_n_u8(72)));

    hi = vaddq_u8(hi, vcgtq_u8(z, vdupq_n_u8(101)));

    hi = vaddq_u8(hi, vcgtq_u8(z, vdupq_n_u8(169)));

    hi = vaddq_u8(hi, vcgtq_u8(z, vdupq_n_u8(254)));
    return vbslq_u8(vcgtq_u8(z, vdupq_n_u8(47)), hi, lo);
}

static inline void calc_ab_fast_internal_common(uint32x4_t s0, uint32x4_t s1, uint32x4_t s2, uint32x4_t s3,
                                                uint32x4_t s4, uint32x4_t s5, uint32x4_t s6, uint32x4_t s7,
                                                int32x4_t sr4, int32x4_t sr5, int32x4_t sr6, int32x4_t sr7,
                                                uint32x4_t const_n_val, uint32x4_t s_vec, uint32x4_t const_val,
                                                uint32x4_t one_by_n_minus_1_vec, uint16x4_t sgrproj_sgr,
                                                uint16_t* dst_A16, int32_t* dst2, const int buf_stride) {
    uint32x4_t q0, q1, q2, q3;
    uint32x4_t p0, p1, p2, p3;

    s0 = vmulq_u32(s0, const_n_val);
    s1 = vmulq_u32(s1, const_n_val);
    s2 = vmulq_u32(s2, const_n_val);
    s3 = vmulq_u32(s3, const_n_val);

    q0 = vmulq_u32(s4, s4);
    q1 = vmulq_u32(s5, s5);
    q2 = vmulq_u32(s6, s6);
    q3 = vmulq_u32(s7, s7);

    // max(n * sum_sq - sum^2, 0): the unsigned saturating subtraction is exactly that.
    p0 = vqsubq_u32(s0, q0);
    p1 = vqsubq_u32(s1, q1);
    p2 = vqsubq_u32(s2, q2);
    p3 = vqsubq_u32(s3, q3);

    p0 = vmulq_u32(p0, s_vec);
    p1 = vmulq_u32(p1, s_vec);
    p2 = vmulq_u32(p2, s_vec);
    p3 = vmulq_u32(p3, s_vec);

    p0 = vrshrq_n_u32(p0, SGRPROJ_MTABLE_BITS);
    p1 = vrshrq_n_u32(p1, SGRPROJ_MTABLE_BITS);
    p2 = vrshrq_n_u32(p2, SGRPROJ_MTABLE_BITS);
    p3 = vrshrq_n_u32(p3, SGRPROJ_MTABLE_BITS);

    p0 = vminq_u32(p0, const_val);
    p1 = vminq_u32(p1, const_val);
    p2 = vminq_u32(p2, const_val);
    p3 = vminq_u32(p3, const_val);

    {
        // p holds z = min(.., 255) per sample; the a plane stores x_by_xplus1[z] and the b computation needs
        // SGRPROJ_SGR - x_by_xplus1[z].
        assert(SGRPROJ_SGR == 256);
        (void)sgrproj_sgr;
        const uint8x16_t z   = vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(p0), vmovn_u32(p1))),
                                         vmovn_u16(vcombine_u16(vmovn_u32(p2), vmovn_u32(p3))));
        const uint8x16_t r   = sgr_256_minus_x_by_xplus1(z);
        const uint16x8_t r01 = vmovl_u8(vget_low_u8(r));
        const uint16x8_t r23 = vmovl_u8(vget_high_u8(r));
        const uint16x8_t a01 = vsubq_u16(vdupq_n_u16(256), r01);
        const uint16x8_t a23 = vsubq_u16(vdupq_n_u16(256), r23);
        vst1_u16(dst_A16 + 0 * buf_stride, vget_low_u16(a01));
        vst1_u16(dst_A16 + 1 * buf_stride, vget_high_u16(a01));
        vst1_u16(dst_A16 + 2 * buf_stride, vget_low_u16(a23));
        vst1_u16(dst_A16 + 3 * buf_stride, vget_high_u16(a23));
        p0 = vmovl_u16(vget_low_u16(r01));
        p1 = vmovl_u16(vget_high_u16(r01));
        p2 = vmovl_u16(vget_low_u16(r23));
        p3 = vmovl_u16(vget_high_u16(r23));
    }

    s4 = vmulq_u32(vreinterpretq_u32_s32(sr4), one_by_n_minus_1_vec);
    s5 = vmulq_u32(vreinterpretq_u32_s32(sr5), one_by_n_minus_1_vec);
    s6 = vmulq_u32(vreinterpretq_u32_s32(sr6), one_by_n_minus_1_vec);
    s7 = vmulq_u32(vreinterpretq_u32_s32(sr7), one_by_n_minus_1_vec);

    s4 = vmulq_u32(s4, p0);
    s5 = vmulq_u32(s5, p1);
    s6 = vmulq_u32(s6, p2);
    s7 = vmulq_u32(s7, p3);

    p0 = vrshrq_n_u32(s4, SGRPROJ_RECIP_BITS);
    p1 = vrshrq_n_u32(s5, SGRPROJ_RECIP_BITS);
    p2 = vrshrq_n_u32(s6, SGRPROJ_RECIP_BITS);
    p3 = vrshrq_n_u32(s7, SGRPROJ_RECIP_BITS);

    store_s32_4x4(dst2,
                  buf_stride,
                  vreinterpretq_s32_u32(p0),
                  vreinterpretq_s32_u32(p1),
                  vreinterpretq_s32_u32(p2),
                  vreinterpretq_s32_u32(p3));
}

static inline void calc_ab_internal_common(uint32x4_t s0, uint32x4_t s1, uint32x4_t s2, uint32x4_t s3, uint32x4_t s4,
                                           uint32x4_t s5, uint32x4_t s6, uint32x4_t s7, uint16x8_t s16_0,
                                           uint16x8_t s16_1, uint16x8_t s16_2, uint16x8_t s16_3, uint16x8_t s16_4,
                                           uint16x8_t s16_5, uint16x8_t s16_6, uint16x8_t s16_7, uint32x4_t const_n_val,
                                           uint32x4_t s_vec, uint32x4_t const_val, uint16x4_t one_by_n_minus_1_vec,
                                           uint16x8_t sgrproj_sgr, uint16_t* dst_A16, int32_t* dst2,
                                           const int buf_stride) {
    uint16x4_t d0, d1, d2, d3, d4, d5, d6, d7;
    uint32x4_t q0, q1, q2, q3, q4, q5, q6, q7;
    uint32x4_t p0, p1, p2, p3, p4, p5, p6, p7;

    s0 = vmulq_u32(s0, const_n_val);
    s1 = vmulq_u32(s1, const_n_val);
    s2 = vmulq_u32(s2, const_n_val);
    s3 = vmulq_u32(s3, const_n_val);
    s4 = vmulq_u32(s4, const_n_val);
    s5 = vmulq_u32(s5, const_n_val);
    s6 = vmulq_u32(s6, const_n_val);
    s7 = vmulq_u32(s7, const_n_val);

    d0 = vget_low_u16(s16_4);
    d1 = vget_low_u16(s16_5);
    d2 = vget_low_u16(s16_6);
    d3 = vget_low_u16(s16_7);
    d4 = vget_high_u16(s16_4);
    d5 = vget_high_u16(s16_5);
    d6 = vget_high_u16(s16_6);
    d7 = vget_high_u16(s16_7);

    q0 = vmull_u16(d0, d0);
    q1 = vmull_u16(d1, d1);
    q2 = vmull_u16(d2, d2);
    q3 = vmull_u16(d3, d3);
    q4 = vmull_u16(d4, d4);
    q5 = vmull_u16(d5, d5);
    q6 = vmull_u16(d6, d6);
    q7 = vmull_u16(d7, d7);

    // max(n * sum_sq - sum^2, 0): the unsigned saturating subtraction is exactly that.
    p0 = vqsubq_u32(s0, q0);
    p1 = vqsubq_u32(s1, q1);
    p2 = vqsubq_u32(s2, q2);
    p3 = vqsubq_u32(s3, q3);
    p4 = vqsubq_u32(s4, q4);
    p5 = vqsubq_u32(s5, q5);
    p6 = vqsubq_u32(s6, q6);
    p7 = vqsubq_u32(s7, q7);

    p0 = vmulq_u32(p0, s_vec);
    p1 = vmulq_u32(p1, s_vec);
    p2 = vmulq_u32(p2, s_vec);
    p3 = vmulq_u32(p3, s_vec);
    p4 = vmulq_u32(p4, s_vec);
    p5 = vmulq_u32(p5, s_vec);
    p6 = vmulq_u32(p6, s_vec);
    p7 = vmulq_u32(p7, s_vec);

    p0 = vrshrq_n_u32(p0, SGRPROJ_MTABLE_BITS);
    p1 = vrshrq_n_u32(p1, SGRPROJ_MTABLE_BITS);
    p2 = vrshrq_n_u32(p2, SGRPROJ_MTABLE_BITS);
    p3 = vrshrq_n_u32(p3, SGRPROJ_MTABLE_BITS);
    p4 = vrshrq_n_u32(p4, SGRPROJ_MTABLE_BITS);
    p5 = vrshrq_n_u32(p5, SGRPROJ_MTABLE_BITS);
    p6 = vrshrq_n_u32(p6, SGRPROJ_MTABLE_BITS);
    p7 = vrshrq_n_u32(p7, SGRPROJ_MTABLE_BITS);

    p0 = vminq_u32(p0, const_val);
    p1 = vminq_u32(p1, const_val);
    p2 = vminq_u32(p2, const_val);
    p3 = vminq_u32(p3, const_val);
    p4 = vminq_u32(p4, const_val);
    p5 = vminq_u32(p5, const_val);
    p6 = vminq_u32(p6, const_val);
    p7 = vminq_u32(p7, const_val);

    {
        // Row x of the group is (p[x], p[x + 4]); see calc_ab_fast_internal_common for the lookup.
        assert(SGRPROJ_SGR == 256);
        (void)sgrproj_sgr;
        const uint8x16_t z01 = vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(p0), vmovn_u32(p4))),
                                           vmovn_u16(vcombine_u16(vmovn_u32(p1), vmovn_u32(p5))));
        const uint8x16_t z23 = vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(p2), vmovn_u32(p6))),
                                           vmovn_u16(vcombine_u16(vmovn_u32(p3), vmovn_u32(p7))));
        const uint8x16_t r01 = sgr_256_minus_x_by_xplus1(z01);
        const uint8x16_t r23 = sgr_256_minus_x_by_xplus1(z23);

        s16_4 = vmovl_u8(vget_low_u8(r01));

        s16_5 = vmovl_u8(vget_high_u8(r01));

        s16_6 = vmovl_u8(vget_low_u8(r23));

        s16_7 = vmovl_u8(vget_high_u8(r23));
        vst1q_u16(dst_A16 + 0 * buf_stride, vsubq_u16(vdupq_n_u16(256), s16_4));
        vst1q_u16(dst_A16 + 1 * buf_stride, vsubq_u16(vdupq_n_u16(256), s16_5));
        vst1q_u16(dst_A16 + 2 * buf_stride, vsubq_u16(vdupq_n_u16(256), s16_6));
        vst1q_u16(dst_A16 + 3 * buf_stride, vsubq_u16(vdupq_n_u16(256), s16_7));
    }

    // (256 - a) * sum * one_by_n: the box sum (16 bits) and 256 - a (at most 255) go through one widening
    // multiply, and the product takes the reciprocal in 32-bit lanes. The reference forms the same product modulo
    // 2^32, where the order of the factors does not matter.
    const uint32x4_t one_by_n = vmovl_u16(one_by_n_minus_1_vec);

    s0 = vmull_u16(vget_low_u16(s16_0), vget_low_u16(s16_4));

    s1 = vmull_u16(vget_low_u16(s16_1), vget_low_u16(s16_5));

    s2 = vmull_u16(vget_low_u16(s16_2), vget_low_u16(s16_6));

    s3 = vmull_u16(vget_low_u16(s16_3), vget_low_u16(s16_7));

    s4 = vmull_high_u16(s16_0, s16_4);

    s5 = vmull_high_u16(s16_1, s16_5);

    s6 = vmull_high_u16(s16_2, s16_6);

    s7 = vmull_high_u16(s16_3, s16_7);

    s0 = vmulq_u32(s0, one_by_n);
    s1 = vmulq_u32(s1, one_by_n);
    s2 = vmulq_u32(s2, one_by_n);
    s3 = vmulq_u32(s3, one_by_n);
    s4 = vmulq_u32(s4, one_by_n);
    s5 = vmulq_u32(s5, one_by_n);
    s6 = vmulq_u32(s6, one_by_n);
    s7 = vmulq_u32(s7, one_by_n);

    p0 = vrshrq_n_u32(s0, SGRPROJ_RECIP_BITS);
    p1 = vrshrq_n_u32(s1, SGRPROJ_RECIP_BITS);
    p2 = vrshrq_n_u32(s2, SGRPROJ_RECIP_BITS);
    p3 = vrshrq_n_u32(s3, SGRPROJ_RECIP_BITS);
    p4 = vrshrq_n_u32(s4, SGRPROJ_RECIP_BITS);
    p5 = vrshrq_n_u32(s5, SGRPROJ_RECIP_BITS);
    p6 = vrshrq_n_u32(s6, SGRPROJ_RECIP_BITS);
    p7 = vrshrq_n_u32(s7, SGRPROJ_RECIP_BITS);

    store_s32_4x4(dst2,
                  buf_stride,
                  vreinterpretq_s32_u32(p0),
                  vreinterpretq_s32_u32(p1),
                  vreinterpretq_s32_u32(p2),
                  vreinterpretq_s32_u32(p3));
    store_s32_4x4(dst2 + 4,
                  buf_stride,
                  vreinterpretq_s32_u32(p4),
                  vreinterpretq_s32_u32(p5),
                  vreinterpretq_s32_u32(p6),
                  vreinterpretq_s32_u32(p7));
}

static inline void boxsum2_square_sum_calc(int16x4_t t1, int16x4_t t2, int16x4_t t3, int16x4_t t4, int16x4_t t5,
                                           int16x4_t t6, int16x4_t t7, int16x4_t t8, int16x4_t t9, int16x4_t t10,
                                           int16x4_t t11, int32x4_t* r0, int32x4_t* r1, int32x4_t* r2, int32x4_t* r3) {
    int32x4_t d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11;
    int32x4_t r12, r34, r67, r89, r1011;
    int32x4_t r345, r6789, r789;

    d1  = vmull_s16(t1, t1);
    d2  = vmull_s16(t2, t2);
    d3  = vmull_s16(t3, t3);
    d4  = vmull_s16(t4, t4);
    d5  = vmull_s16(t5, t5);
    d6  = vmull_s16(t6, t6);
    d7  = vmull_s16(t7, t7);
    d8  = vmull_s16(t8, t8);
    d9  = vmull_s16(t9, t9);
    d10 = vmull_s16(t10, t10);
    d11 = vmull_s16(t11, t11);

    r12   = vaddq_s32(d1, d2);
    r34   = vaddq_s32(d3, d4);
    r67   = vaddq_s32(d6, d7);
    r89   = vaddq_s32(d8, d9);
    r1011 = vaddq_s32(d10, d11);
    r345  = vaddq_s32(r34, d5);
    r6789 = vaddq_s32(r67, r89);
    r789  = vsubq_s32(r6789, d6);
    *r0   = vaddq_s32(r12, r345);
    *r1   = vaddq_s32(r67, r345);
    *r2   = vaddq_s32(d5, r6789);
    *r3   = vaddq_s32(r789, r1011);
}

static AOM_FORCE_INLINE void boxsum2(int16_t* src, const int src_stride, int16_t* dst16, int32_t* dst32, int32_t* dst2,
                                     const int dst_stride, const int width, const int height) {
    assert(width > 2 * SGRPROJ_BORDER_HORZ);
    assert(height > 2 * SGRPROJ_BORDER_VERT);

    int16_t * dst1_16_ptr, *src_ptr;
    int32_t*  dst2_ptr;
    int       h, w, count = 0;
    const int dst_stride_2 = (dst_stride << 1);
    const int dst_stride_8 = (dst_stride << 3);
    int       x;

    dst1_16_ptr = dst16;
    dst2_ptr    = dst2;
    src_ptr     = src;
    w           = width;
    {
        int16x8_t t1, t2, t3, t4, t5, t6, t7;
        int16x8_t t8, t9, t10, t11, t12;

        int16x8_t q12345, q56789, q34567, q7891011;
        int16x8_t q12, q34, q67, q89, q1011;
        int16x8_t q345, q6789, q789;

        int32x4_t r12345, r56789, r34567, r7891011;

        do {
            h           = height;
            dst1_16_ptr = dst16 + (count << 3);
            dst2_ptr    = dst2 + (count << 3);
            src_ptr     = src + (count << 3);

            dst1_16_ptr += dst_stride_2;
            dst2_ptr += dst_stride_2;
            do {
                load_s16_8x4(src_ptr, src_stride, &t1, &t2, &t3, &t4);
                src_ptr += 4 * src_stride;
                load_s16_8x4(src_ptr, src_stride, &t5, &t6, &t7, &t8);
                src_ptr += 4 * src_stride;
                load_s16_8x4(src_ptr, src_stride, &t9, &t10, &t11, &t12);

                q12      = vaddq_s16(t1, t2);
                q34      = vaddq_s16(t3, t4);
                q67      = vaddq_s16(t6, t7);
                q89      = vaddq_s16(t8, t9);
                q1011    = vaddq_s16(t10, t11);
                q345     = vaddq_s16(q34, t5);
                q6789    = vaddq_s16(q67, q89);
                q789     = vaddq_s16(q89, t7);
                q12345   = vaddq_s16(q12, q345);
                q34567   = vaddq_s16(q67, q345);
                q56789   = vaddq_s16(t5, q6789);
                q7891011 = vaddq_s16(q789, q1011);

                store_s16_8x4(dst1_16_ptr, dst_stride_2, q12345, q34567, q56789, q7891011);
                dst1_16_ptr += dst_stride_8;

                boxsum2_square_sum_calc(vget_low_s16(t1),
                                        vget_low_s16(t2),
                                        vget_low_s16(t3),
                                        vget_low_s16(t4),
                                        vget_low_s16(t5),
                                        vget_low_s16(t6),
                                        vget_low_s16(t7),
                                        vget_low_s16(t8),
                                        vget_low_s16(t9),
                                        vget_low_s16(t10),
                                        vget_low_s16(t11),
                                        &r12345,
                                        &r34567,
                                        &r56789,
                                        &r7891011);

                store_s32_4x4(dst2_ptr, dst_stride_2, r12345, r34567, r56789, r7891011);

                boxsum2_square_sum_calc(vget_high_s16(t1),
                                        vget_high_s16(t2),
                                        vget_high_s16(t3),
                                        vget_high_s16(t4),
                                        vget_high_s16(t5),
                                        vget_high_s16(t6),
                                        vget_high_s16(t7),
                                        vget_high_s16(t8),
                                        vget_high_s16(t9),
                                        vget_high_s16(t10),
                                        vget_high_s16(t11),
                                        &r12345,
                                        &r34567,
                                        &r56789,
                                        &r7891011);

                store_s32_4x4(dst2_ptr + 4, dst_stride_2, r12345, r34567, r56789, r7891011);
                dst2_ptr += (dst_stride_8);
                h -= 8;
            } while (h > 0);
            w -= 8;
            count++;
        } while (w > 0);

        // memset needed for row pixels as 2nd stage of boxsum filter uses
        // first 2 rows of dst16, dst2 buffer which is not filled in first stage.
        for (x = 0; x < 2; x++) {
            memset(dst16 + x * dst_stride, 0, (width + 4) * sizeof(*dst16));
            memset(dst2 + x * dst_stride, 0, (width + 4) * sizeof(*dst2));
        }

        // memset needed for extra columns as 2nd stage of boxsum filter uses
        // last 2 columns of dst16, dst2 buffer which is not filled in first stage.
        for (x = 2; x < height + 2; x++) {
            int dst_offset = x * dst_stride + width + 2;
            memset(dst16 + dst_offset, 0, 3 * sizeof(*dst16));
            memset(dst2 + dst_offset, 0, 3 * sizeof(*dst2));
        }
    }

    {
        int16x4_t s1, s2, s3, s4, s5, s6, s7, s8;
        int32x4_t d1, d2, d3, d4, d5, d6, d7, d8;
        int32x4_t q12345, q34567, q23456, q45678;
        int32x4_t q23, q45, q67;
        int32x4_t q2345, q4567;

        int32x4_t r12345, r34567, r23456, r45678;
        int32x4_t r23, r45, r67;
        int32x4_t r2345, r4567;

        int32_t *src2_ptr, *dst1_32_ptr;
        int16_t* src1_ptr;
        count = 0;
        h     = height;
        do {
            dst1_32_ptr = dst32 + count * dst_stride_8 + (dst_stride_2);
            dst2_ptr    = dst2 + count * dst_stride_8 + (dst_stride_2);
            src1_ptr    = dst16 + count * dst_stride_8 + (dst_stride_2);
            src2_ptr    = dst2 + count * dst_stride_8 + (dst_stride_2);
            w           = width;

            dst1_32_ptr += 2;
            dst2_ptr += 2;
            load_s16_4x4(src1_ptr, dst_stride_2, &s1, &s2, &s3, &s4);
            transpose_s16_4x4d(&s1, &s2, &s3, &s4);
            load_s32_4x4(src2_ptr, dst_stride_2, &d1, &d2, &d3, &d4);
            transpose_s32_4x4(&d1, &d2, &d3, &d4);
            do {
                src1_ptr += 4;
                src2_ptr += 4;
                load_s16_4x4(src1_ptr, dst_stride_2, &s5, &s6, &s7, &s8);
                transpose_s16_4x4d(&s5, &s6, &s7, &s8);
                load_s32_4x4(src2_ptr, dst_stride_2, &d5, &d6, &d7, &d8);
                transpose_s32_4x4(&d5, &d6, &d7, &d8);
                q23    = vaddl_s16(s2, s3);
                q45    = vaddl_s16(s4, s5);
                q67    = vaddl_s16(s6, s7);
                q2345  = vaddq_s32(q23, q45);
                q4567  = vaddq_s32(q45, q67);
                q12345 = vaddq_s32(vmovl_s16(s1), q2345);
                q23456 = vaddq_s32(q2345, vmovl_s16(s6));
                q34567 = vaddq_s32(q4567, vmovl_s16(s3));
                q45678 = vaddq_s32(q4567, vmovl_s16(s8));

                transpose_s32_4x4(&q12345, &q23456, &q34567, &q45678);
                store_s32_4x4(dst1_32_ptr, dst_stride_2, q12345, q23456, q34567, q45678);
                dst1_32_ptr += 4;
                s1 = s5;
                s2 = s6;
                s3 = s7;
                s4 = s8;

                r23    = vaddq_s32(d2, d3);
                r45    = vaddq_s32(d4, d5);
                r67    = vaddq_s32(d6, d7);
                r2345  = vaddq_s32(r23, r45);
                r4567  = vaddq_s32(r45, r67);
                r12345 = vaddq_s32(d1, r2345);
                r23456 = vaddq_s32(r2345, d6);
                r34567 = vaddq_s32(r4567, d3);
                r45678 = vaddq_s32(r4567, d8);

                transpose_s32_4x4(&r12345, &r23456, &r34567, &r45678);
                store_s32_4x4(dst2_ptr, dst_stride_2, r12345, r23456, r34567, r45678);
                dst2_ptr += 4;
                d1 = d5;
                d2 = d6;
                d3 = d7;
                d4 = d8;
                w -= 4;
            } while (w > 0);
            h -= 8;
            count++;
        } while (h > 0);
    }
}

static AOM_FORCE_INLINE void calc_ab_internal_lbd(int32_t* A, uint16_t* A16, uint16_t* B16, int32_t* B,
                                                  const int buf_stride, const int width, const int height, const int r,
                                                  const int s, const int ht_inc) {
    int32_t *        src1, *dst2, count = 0;
    uint16_t *       dst_A16, *src2;
    const uint32_t   n                    = (2 * r + 1) * (2 * r + 1);
    const uint32x4_t const_n_val          = vdupq_n_u32(n);
    const uint16x8_t sgrproj_sgr          = vdupq_n_u16(SGRPROJ_SGR);
    const uint16x4_t one_by_n_minus_1_vec = vdup_n_u16(svt_aom_eb_one_by_x[n - 1]);
    const uint32x4_t const_val            = vdupq_n_u32(255);

    uint16x8_t s16_0, s16_1, s16_2, s16_3, s16_4, s16_5, s16_6, s16_7;

    uint32x4_t s0, s1, s2, s3, s4, s5, s6, s7;

    const uint32x4_t s_vec = vdupq_n_u32(s);
    int              w, h = height;

    do {
        dst_A16 = A16 + (count << 2) * buf_stride;
        src1    = A + (count << 2) * buf_stride;
        src2    = B16 + (count << 2) * buf_stride;
        dst2    = B + (count << 2) * buf_stride;
        w       = width;
        do {
            load_u32_4x4((uint32_t*)src1, buf_stride, &s0, &s1, &s2, &s3);
            load_u32_4x4((uint32_t*)src1 + 4, buf_stride, &s4, &s5, &s6, &s7);
            load_u16_8x4(src2, buf_stride, &s16_0, &s16_1, &s16_2, &s16_3);

            s16_4 = s16_0;
            s16_5 = s16_1;
            s16_6 = s16_2;
            s16_7 = s16_3;

            calc_ab_internal_common(s0,
                                    s1,
                                    s2,
                                    s3,
                                    s4,
                                    s5,
                                    s6,
                                    s7,
                                    s16_0,
                                    s16_1,
                                    s16_2,
                                    s16_3,
                                    s16_4,
                                    s16_5,
                                    s16_6,
                                    s16_7,
                                    const_n_val,
                                    s_vec,
                                    const_val,
                                    one_by_n_minus_1_vec,
                                    sgrproj_sgr,
                                    dst_A16,
                                    dst2,
                                    buf_stride);

            w -= 8;
            dst2 += 8;
            src1 += 8;
            src2 += 8;
            dst_A16 += 8;
        } while (w > 0);
        count++;
        h -= (ht_inc * 4);
    } while (h > 0);
}

static AOM_FORCE_INLINE void calc_ab_internal_hbd(int32_t* A, uint16_t* A16, uint16_t* B16, int32_t* B,
                                                  const int buf_stride, const int width, const int height,
                                                  const int bit_depth, const int r, const int s, const int ht_inc) {
    int32_t *        src1, *dst2, count = 0;
    uint16_t *       dst_A16, *src2;
    const uint32_t   n                    = (2 * r + 1) * (2 * r + 1);
    const int16x8_t  bd_min_2_vec         = vdupq_n_s16(-(bit_depth - 8));
    const int32x4_t  bd_min_1_vec         = vdupq_n_s32(-((bit_depth - 8) << 1));
    const uint32x4_t const_n_val          = vdupq_n_u32(n);
    const uint16x8_t sgrproj_sgr          = vdupq_n_u16(SGRPROJ_SGR);
    const uint16x4_t one_by_n_minus_1_vec = vdup_n_u16(svt_aom_eb_one_by_x[n - 1]);
    const uint32x4_t const_val            = vdupq_n_u32(255);

    int32x4_t  sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7;
    uint16x8_t s16_0, s16_1, s16_2, s16_3;
    uint16x8_t s16_4, s16_5, s16_6, s16_7;
    uint32x4_t s0, s1, s2, s3, s4, s5, s6, s7;

    const uint32x4_t s_vec = vdupq_n_u32(s);
    int              w, h = height;

    do {
        src1    = A + (count << 2) * buf_stride;
        src2    = B16 + (count << 2) * buf_stride;
        dst2    = B + (count << 2) * buf_stride;
        dst_A16 = A16 + (count << 2) * buf_stride;
        w       = width;
        do {
            load_s32_4x4(src1, buf_stride, &sr0, &sr1, &sr2, &sr3);
            load_s32_4x4(src1 + 4, buf_stride, &sr4, &sr5, &sr6, &sr7);
            load_u16_8x4(src2, buf_stride, &s16_0, &s16_1, &s16_2, &s16_3);

            s0 = vrshlq_u32(vreinterpretq_u32_s32(sr0), bd_min_1_vec);
            s1 = vrshlq_u32(vreinterpretq_u32_s32(sr1), bd_min_1_vec);
            s2 = vrshlq_u32(vreinterpretq_u32_s32(sr2), bd_min_1_vec);
            s3 = vrshlq_u32(vreinterpretq_u32_s32(sr3), bd_min_1_vec);
            s4 = vrshlq_u32(vreinterpretq_u32_s32(sr4), bd_min_1_vec);
            s5 = vrshlq_u32(vreinterpretq_u32_s32(sr5), bd_min_1_vec);
            s6 = vrshlq_u32(vreinterpretq_u32_s32(sr6), bd_min_1_vec);
            s7 = vrshlq_u32(vreinterpretq_u32_s32(sr7), bd_min_1_vec);

            s16_4 = vrshlq_u16(s16_0, bd_min_2_vec);
            s16_5 = vrshlq_u16(s16_1, bd_min_2_vec);
            s16_6 = vrshlq_u16(s16_2, bd_min_2_vec);
            s16_7 = vrshlq_u16(s16_3, bd_min_2_vec);

            calc_ab_internal_common(s0,
                                    s1,
                                    s2,
                                    s3,
                                    s4,
                                    s5,
                                    s6,
                                    s7,
                                    s16_0,
                                    s16_1,
                                    s16_2,
                                    s16_3,
                                    s16_4,
                                    s16_5,
                                    s16_6,
                                    s16_7,
                                    const_n_val,
                                    s_vec,
                                    const_val,
                                    one_by_n_minus_1_vec,
                                    sgrproj_sgr,
                                    dst_A16,
                                    dst2,
                                    buf_stride);

            w -= 8;
            dst2 += 8;
            src1 += 8;
            src2 += 8;
            dst_A16 += 8;
        } while (w > 0);
        count++;
        h -= (ht_inc * 4);
    } while (h > 0);
}

// B holds the box sums and dst receives the b plane; the two may be the same buffer.
static AOM_FORCE_INLINE void calc_ab_fast_internal_lbd(int32_t* A, uint16_t* A16, int32_t* B, int32_t* dst,
                                                       const int buf_stride, const int width, const int height,
                                                       const int r, const int s, const int ht_inc) {
    int32_t *        src1, *src2, *dst2, count = 0;
    uint16_t*        dst_A16;
    const uint32_t   n                    = (2 * r + 1) * (2 * r + 1);
    const uint32x4_t const_n_val          = vdupq_n_u32(n);
    const uint16x4_t sgrproj_sgr          = vdup_n_u16(SGRPROJ_SGR);
    const uint32x4_t one_by_n_minus_1_vec = vdupq_n_u32(svt_aom_eb_one_by_x[n - 1]);
    const uint32x4_t const_val            = vdupq_n_u32(255);

    int32x4_t  sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7;
    uint32x4_t s0, s1, s2, s3, s4, s5, s6, s7;

    const uint32x4_t s_vec = vdupq_n_u32(s);
    int              w, h = height;

    do {
        src1    = A + (count << 2) * buf_stride;
        src2    = B + (count << 2) * buf_stride;
        dst2    = dst + (count << 2) * buf_stride;
        dst_A16 = A16 + (count << 2) * buf_stride;
        w       = width;
        do {
            load_s32_4x4(src1, buf_stride, &sr0, &sr1, &sr2, &sr3);
            load_s32_4x4(src2, buf_stride, &sr4, &sr5, &sr6, &sr7);

            s0 = vreinterpretq_u32_s32(sr0);
            s1 = vreinterpretq_u32_s32(sr1);
            s2 = vreinterpretq_u32_s32(sr2);
            s3 = vreinterpretq_u32_s32(sr3);
            s4 = vreinterpretq_u32_s32(sr4);
            s5 = vreinterpretq_u32_s32(sr5);
            s6 = vreinterpretq_u32_s32(sr6);
            s7 = vreinterpretq_u32_s32(sr7);

            calc_ab_fast_internal_common(s0,
                                         s1,
                                         s2,
                                         s3,
                                         s4,
                                         s5,
                                         s6,
                                         s7,
                                         sr4,
                                         sr5,
                                         sr6,
                                         sr7,
                                         const_n_val,
                                         s_vec,
                                         const_val,
                                         one_by_n_minus_1_vec,
                                         sgrproj_sgr,
                                         dst_A16,
                                         dst2,
                                         buf_stride);

            w -= 4;
            src1 += 4;
            src2 += 4;
            dst2 += 4;
            dst_A16 += 4;
        } while (w > 0);
        count++;
        h -= (ht_inc * 4);
    } while (h > 0);
}

static AOM_FORCE_INLINE void calc_ab_fast_internal_hbd(int32_t* A, uint16_t* A16, int32_t* B, int32_t* dst,
                                                       const int buf_stride, const int width, const int height,
                                                       const int bit_depth, const int r, const int s,
                                                       const int ht_inc) {
    int32_t *        src1, *src2, *dst2, count = 0;
    uint16_t*        dst_A16;
    const uint32_t   n                    = (2 * r + 1) * (2 * r + 1);
    const int32x4_t  bd_min_2_vec         = vdupq_n_s32(-(bit_depth - 8));
    const int32x4_t  bd_min_1_vec         = vdupq_n_s32(-((bit_depth - 8) << 1));
    const uint32x4_t const_n_val          = vdupq_n_u32(n);
    const uint16x4_t sgrproj_sgr          = vdup_n_u16(SGRPROJ_SGR);
    const uint32x4_t one_by_n_minus_1_vec = vdupq_n_u32(svt_aom_eb_one_by_x[n - 1]);
    const uint32x4_t const_val            = vdupq_n_u32(255);

    int32x4_t  sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7;
    uint32x4_t s0, s1, s2, s3, s4, s5, s6, s7;

    const uint32x4_t s_vec = vdupq_n_u32(s);
    int              w, h = height;

    do {
        src1    = A + (count << 2) * buf_stride;
        src2    = B + (count << 2) * buf_stride;
        dst2    = dst + (count << 2) * buf_stride;
        dst_A16 = A16 + (count << 2) * buf_stride;
        w       = width;
        do {
            load_s32_4x4(src1, buf_stride, &sr0, &sr1, &sr2, &sr3);
            load_s32_4x4(src2, buf_stride, &sr4, &sr5, &sr6, &sr7);

            s0 = vrshlq_u32(vreinterpretq_u32_s32(sr0), bd_min_1_vec);
            s1 = vrshlq_u32(vreinterpretq_u32_s32(sr1), bd_min_1_vec);
            s2 = vrshlq_u32(vreinterpretq_u32_s32(sr2), bd_min_1_vec);
            s3 = vrshlq_u32(vreinterpretq_u32_s32(sr3), bd_min_1_vec);
            s4 = vrshlq_u32(vreinterpretq_u32_s32(sr4), bd_min_2_vec);
            s5 = vrshlq_u32(vreinterpretq_u32_s32(sr5), bd_min_2_vec);
            s6 = vrshlq_u32(vreinterpretq_u32_s32(sr6), bd_min_2_vec);
            s7 = vrshlq_u32(vreinterpretq_u32_s32(sr7), bd_min_2_vec);

            calc_ab_fast_internal_common(s0,
                                         s1,
                                         s2,
                                         s3,
                                         s4,
                                         s5,
                                         s6,
                                         s7,
                                         sr4,
                                         sr5,
                                         sr6,
                                         sr7,
                                         const_n_val,
                                         s_vec,
                                         const_val,
                                         one_by_n_minus_1_vec,
                                         sgrproj_sgr,
                                         dst_A16,
                                         dst2,
                                         buf_stride);

            w -= 4;
            src1 += 4;
            src2 += 4;
            dst2 += 4;
            dst_A16 += 4;
        } while (w > 0);
        count++;
        h -= (ht_inc * 4);
    } while (h > 0);
}

static AOM_FORCE_INLINE void boxsum1(int16_t* src, const int src_stride, uint16_t* dst1, int32_t* dst2,
                                     const int dst_stride, const int width, const int height) {
    assert(width > 2 * SGRPROJ_BORDER_HORZ);
    assert(height > 2 * SGRPROJ_BORDER_VERT);

    int16_t*  src_ptr;
    int32_t*  dst2_ptr;
    uint16_t* dst1_ptr;
    int       h, w, count = 0;
    int       x;

    w = width;
    {
        int16x8_t s1, s2, s3, s4, s5, s6, s7, s8;
        int16x8_t q23, q34, q56, q234, q345, q456, q567;
        int32x4_t r23, r56, r345, r456, r567, r78, r678;
        int32x4_t r4_low, r4_high, r34_low, r34_high, r234_low, r234_high;
        int32x4_t r2, r3, r5, r6, r7, r8;
        int16x8_t q678, q78;

        do {
            dst1_ptr = dst1 + (count << 3);
            dst2_ptr = dst2 + (count << 3);
            src_ptr  = src + (count << 3);
            h        = height;

            load_s16_8x4(src_ptr, src_stride, &s1, &s2, &s3, &s4);
            src_ptr += 4 * src_stride;

            q23  = vaddq_s16(s2, s3);
            q234 = vaddq_s16(q23, s4);
            q34  = vaddq_s16(s3, s4);
            dst1_ptr += (dst_stride << 1);

            r2       = vmull_s16(vget_low_s16(s2), vget_low_s16(s2));
            r3       = vmull_s16(vget_low_s16(s3), vget_low_s16(s3));
            r4_low   = vmull_s16(vget_low_s16(s4), vget_low_s16(s4));
            r23      = vaddq_s32(r2, r3);
            r234_low = vaddq_s32(r23, r4_low);
            r34_low  = vaddq_s32(r3, r4_low);

            r2        = vmull_s16(vget_high_s16(s2), vget_high_s16(s2));
            r3        = vmull_s16(vget_high_s16(s3), vget_high_s16(s3));
            r4_high   = vmull_s16(vget_high_s16(s4), vget_high_s16(s4));
            r23       = vaddq_s32(r2, r3);
            r234_high = vaddq_s32(r23, r4_high);
            r34_high  = vaddq_s32(r3, r4_high);

            dst2_ptr += (dst_stride << 1);

            do {
                load_s16_8x4(src_ptr, src_stride, &s5, &s6, &s7, &s8);
                src_ptr += 4 * src_stride;

                q345 = vaddq_s16(s5, q34);
                q56  = vaddq_s16(s5, s6);
                q456 = vaddq_s16(s4, q56);
                q567 = vaddq_s16(s7, q56);
                q78  = vaddq_s16(s7, s8);
                q678 = vaddq_s16(s6, q78);

                store_s16_8x4((int16_t*)dst1_ptr, dst_stride, q234, q345, q456, q567);
                dst1_ptr += (dst_stride << 2);

                s4   = s8;
                q34  = q78;
                q234 = q678;

                r5 = vmull_s16(vget_low_s16(s5), vget_low_s16(s5));
                r6 = vmull_s16(vget_low_s16(s6), vget_low_s16(s6));
                r7 = vmull_s16(vget_low_s16(s7), vget_low_s16(s7));
                r8 = vmull_s16(vget_low_s16(s8), vget_low_s16(s8));

                r345 = vaddq_s32(r5, r34_low);
                r56  = vaddq_s32(r5, r6);
                r456 = vaddq_s32(r4_low, r56);
                r567 = vaddq_s32(r7, r56);
                r78  = vaddq_s32(r7, r8);
                r678 = vaddq_s32(r6, r78);
                store_s32_4x4(dst2_ptr, dst_stride, r234_low, r345, r456, r567);

                r4_low   = r8;
                r34_low  = r78;
                r234_low = r678;

                r5 = vmull_s16(vget_high_s16(s5), vget_high_s16(s5));
                r6 = vmull_s16(vget_high_s16(s6), vget_high_s16(s6));
                r7 = vmull_s16(vget_high_s16(s7), vget_high_s16(s7));
                r8 = vmull_s16(vget_high_s16(s8), vget_high_s16(s8));

                r345 = vaddq_s32(r5, r34_high);
                r56  = vaddq_s32(r5, r6);
                r456 = vaddq_s32(r4_high, r56);
                r567 = vaddq_s32(r7, r56);
                r78  = vaddq_s32(r7, r8);
                r678 = vaddq_s32(r6, r78);
                store_s32_4x4((dst2_ptr + 4), dst_stride, r234_high, r345, r456, r567);
                dst2_ptr += (dst_stride << 2);

                r4_high   = r8;
                r34_high  = r78;
                r234_high = r678;

                h -= 4;
            } while (h > 0);
            w -= 8;
            count++;
        } while (w > 0);

        // memset needed for row pixels as 2nd stage of boxsum filter uses
        // first 2 rows of dst1, dst2 buffer which is not filled in first stage.
        for (x = 0; x < 2; x++) {
            memset(dst1 + x * dst_stride, 0, (width + 4) * sizeof(*dst1));
            memset(dst2 + x * dst_stride, 0, (width + 4) * sizeof(*dst2));
        }

        // memset needed for extra columns as 2nd stage of boxsum filter uses
        // last 2 columns of dst1, dst2 buffer which is not filled in first stage.
        for (x = 2; x < height + 2; x++) {
            int dst_offset = x * dst_stride + width + 2;
            memset(dst1 + dst_offset, 0, 3 * sizeof(*dst1));
            memset(dst2 + dst_offset, 0, 3 * sizeof(*dst2));
        }
    }

    // Second stage: the sum of each column with its two neighbours, row by row and in place. The sums go to the
    // columns 2 to cols + 1 of the rows 0 to rows - 1, cols and rows being the width and the height rounded up to
    // multiples of 4. Every sum is formed from values of the first stage: all the loads of a pass precede its
    // stores, and prev keeps the columns that the pass before it has overwritten.
    const int cols = (width + 3) & ~3;
    const int rows = (height + 3) & ~3;
    for (int i = 0; i < rows; i++) {
        uint16_t* const row1  = dst1 + i * dst_stride + 2;
        int32_t* const  row2  = dst2 + i * dst_stride + 2;
        uint16x8_t      prev1 = vld1q_dup_u16(row1 - 1);
        int32x4_t       prev2 = vld1q_dup_s32(row2 - 1);
        int             j     = 0;
        for (; j + 8 <= cols; j += 8) {
            const uint16x8_t cur1     = vld1q_u16(row1 + j);
            const uint16x8_t next1    = vld1q_u16(row1 + j + 1);
            const int32x4_t  cur2_lo  = vld1q_s32(row2 + j);
            const int32x4_t  cur2_hi  = vld1q_s32(row2 + j + 4);
            const int32x4_t  next2_lo = vld1q_s32(row2 + j + 1);
            const int32x4_t  next2_hi = vld1q_s32(row2 + j + 5);

            vst1q_u16(row1 + j, vaddq_u16(vaddq_u16(vextq_u16(prev1, cur1, 7), cur1), next1));
            vst1q_s32(row2 + j, vaddq_s32(vaddq_s32(vextq_s32(prev2, cur2_lo, 3), cur2_lo), next2_lo));
            vst1q_s32(row2 + j + 4, vaddq_s32(vaddq_s32(vextq_s32(cur2_lo, cur2_hi, 3), cur2_hi), next2_hi));
            prev1 = cur1;
            prev2 = cur2_hi;
        }
        if (j < cols) {
            // four columns are left
            const uint16x4_t cur1  = vld1_u16(row1 + j);
            const uint16x4_t next1 = vld1_u16(row1 + j + 1);
            const int32x4_t  cur2  = vld1q_s32(row2 + j);
            const int32x4_t  next2 = vld1q_s32(row2 + j + 1);

            vst1_u16(row1 + j, vadd_u16(vadd_u16(vext_u16(vget_high_u16(prev1), cur1, 3), cur1), next1));
            vst1q_s32(row2 + j, vaddq_s32(vaddq_s32(vextq_s32(prev2, cur2, 3), cur2), next2));
        }
    }
}

static inline int32x4_t cross_sum_inp_s32(int32_t* buf, int buf_stride) {
    int32x4_t xtr, xt, xtl, xl, x, xr, xbr, xb, xbl;
    int32x4_t fours, threes, res;

    xtl = vld1q_s32(buf - buf_stride - 1);
    xt  = vld1q_s32(buf - buf_stride);
    xtr = vld1q_s32(buf - buf_stride + 1);
    xl  = vld1q_s32(buf - 1);
    x   = vld1q_s32(buf);
    xr  = vld1q_s32(buf + 1);
    xbl = vld1q_s32(buf + buf_stride - 1);
    xb  = vld1q_s32(buf + buf_stride);
    xbr = vld1q_s32(buf + buf_stride + 1);

    fours  = vaddq_s32(xl, vaddq_s32(xt, vaddq_s32(xr, vaddq_s32(xb, x))));
    threes = vaddq_s32(xtl, vaddq_s32(xtr, vaddq_s32(xbr, xbl)));
    res    = vsubq_s32(vshlq_n_s32(vaddq_s32(fours, threes), 2), threes);
    return res;
}

// Weighted 3x3 sum of the a plane (weights 4 and 3, 32 in total). a is at most 256, so the sum is at most 8192 and
// stays in 16-bit lanes.
static inline uint16x8_t cross_sum_inp_u16(uint16_t* buf, int buf_stride) {
    uint16x8_t xtr, xt, xtl, xl, x, xr, xbr, xb, xbl;
    uint16x8_t r0, r1;

    xtl = vld1q_u16(buf - buf_stride - 1);
    xt  = vld1q_u16(buf - buf_stride);
    xtr = vld1q_u16(buf - buf_stride + 1);
    xl  = vld1q_u16(buf - 1);
    x   = vld1q_u16(buf);
    xr  = vld1q_u16(buf + 1);
    xbl = vld1q_u16(buf + buf_stride - 1);
    xb  = vld1q_u16(buf + buf_stride);
    xbr = vld1q_u16(buf + buf_stride + 1);

    xb = vaddq_u16(xb, x);
    xt = vaddq_u16(xt, xr);
    xl = vaddq_u16(xl, xb);
    xl = vaddq_u16(xl, xt);

    r0 = vshlq_n_u16(xl, 2);

    xbl = vaddq_u16(xbl, xbr);
    xtl = vaddq_u16(xtl, xtr);
    xtl = vaddq_u16(xtl, xbl);

    r1 = vshlq_n_u16(xtl, 2);
    r1 = vsubq_u16(r1, xtl);

    return vaddq_u16(r0, r1);
}

static inline int32x4_t cross_sum_fast_even_row(int32_t* buf, int buf_stride) {
    int32x4_t xtr, xt, xtl, xbr, xb, xbl;
    int32x4_t fives, sixes, fives_plus_sixes;

    xtl = vld1q_s32(buf - buf_stride - 1);
    xt  = vld1q_s32(buf - buf_stride);
    xtr = vld1q_s32(buf - buf_stride + 1);
    xbl = vld1q_s32(buf + buf_stride - 1);
    xb  = vld1q_s32(buf + buf_stride);
    xbr = vld1q_s32(buf + buf_stride + 1);

    fives            = vaddq_s32(xtl, vaddq_s32(xtr, vaddq_s32(xbr, xbl)));
    sixes            = vaddq_s32(xt, xb);
    fives_plus_sixes = vaddq_s32(fives, sixes);

    return vaddq_s32(vaddq_s32(vshlq_n_s32(fives_plus_sixes, 2), fives_plus_sixes), sixes);
}

// Weights 5 and 6, 32 in total: at most 8192 for a of at most 256.
static inline uint16x8_t cross_sum_fast_even_row_inp16(uint16_t* buf, int buf_stride) {
    uint16x8_t xtr, xt, xtl, xbr, xb, xbl, xb0;

    xtl = vld1q_u16(buf - buf_stride - 1);
    xt  = vld1q_u16(buf - buf_stride);
    xtr = vld1q_u16(buf - buf_stride + 1);
    xbl = vld1q_u16(buf + buf_stride - 1);
    xb  = vld1q_u16(buf + buf_stride);
    xbr = vld1q_u16(buf + buf_stride + 1);

    xbr = vaddq_u16(xbr, xbl);
    xtr = vaddq_u16(xtr, xtl);
    xbr = vaddq_u16(xbr, xtr);
    xtl = vshlq_n_u16(xbr, 2);
    xbr = vaddq_u16(xtl, xbr);

    xb  = vaddq_u16(xb, xt);
    xb0 = vshlq_n_u16(xb, 1);
    xb  = vshlq_n_u16(xb, 2);
    xb  = vaddq_u16(xb, xb0);

    return vaddq_u16(xbr, xb);
}

static inline int32x4_t cross_sum_fast_odd_row(int32_t* buf) {
    int32x4_t xl, x, xr;
    int32x4_t fives, sixes, fives_plus_sixes;

    xl               = vld1q_s32(buf - 1);
    x                = vld1q_s32(buf);
    xr               = vld1q_s32(buf + 1);
    fives            = vaddq_s32(xl, xr);
    sixes            = x;
    fives_plus_sixes = vaddq_s32(fives, sixes);

    return vaddq_s32(vaddq_s32(vshlq_n_s32(fives_plus_sixes, 2), fives_plus_sixes), sixes);
}

// Weights 5, 6, 5, 16 in total: at most 4096 for a of at most 256.
static inline uint16x8_t cross_sum_fast_odd_row_inp16(uint16_t* buf) {
    uint16x8_t xl, x, xr;
    uint16x8_t x0;

    xl = vld1q_u16(buf - 1);
    x  = vld1q_u16(buf);
    xr = vld1q_u16(buf + 1);
    xl = vaddq_u16(xl, xr);
    x0 = vshlq_n_u16(xl, 2);
    xl = vaddq_u16(xl, x0);

    x0 = vshlq_n_u16(x, 1);
    x  = vshlq_n_u16(x, 2);
    x  = vaddq_u16(x, x0);

    return vaddq_u16(xl, x);
}

static void final_filter_fast_internal(uint16_t* A, int32_t* B, const int buf_stride, int16_t* src,
                                       const int src_stride, int32_t* dst, const int dst_stride, const int width,
                                       const int height) {
    int16x8_t s0;
    int32_t * B_tmp, *dst_ptr;
    uint16_t* A_tmp;
    int16_t*  src_ptr;
    int32x4_t a_res0, a_res1, b_res0, b_res1;
    int       w, h, count = 0;
    assert(SGRPROJ_SGR_BITS == 8);
    assert(SGRPROJ_RST_BITS == 4);

    A_tmp   = A;
    B_tmp   = B;
    src_ptr = src;
    dst_ptr = dst;
    h       = height;
    do {
        A_tmp   = (A + count * buf_stride);
        B_tmp   = (B + count * buf_stride);
        src_ptr = (src + count * src_stride);
        dst_ptr = (dst + count * dst_stride);
        w       = width;
        if (!(count & 1)) {
            do {
                s0 = vld1q_s16(src_ptr);
                // sample (at most 12 bits) times neighbour sum (at most 8192): one widening multiply
                const int16x8_t a_sum = vreinterpretq_s16_u16(cross_sum_fast_even_row_inp16(A_tmp, buf_stride));

                a_res0 = vmull_s16(vget_low_s16(s0), vget_low_s16(a_sum));

                a_res1 = vmull_high_s16(s0, a_sum);

                b_res0 = cross_sum_fast_even_row(B_tmp, buf_stride);
                b_res1 = cross_sum_fast_even_row(B_tmp + 4, buf_stride);
                a_res0 = vaddq_s32(a_res0, b_res0);
                a_res1 = vaddq_s32(a_res1, b_res1);

                a_res0 = vrshrq_n_s32(a_res0, SGRPROJ_SGR_BITS + NB_EVEN - SGRPROJ_RST_BITS);
                a_res1 = vrshrq_n_s32(a_res1, SGRPROJ_SGR_BITS + NB_EVEN - SGRPROJ_RST_BITS);

                vst1q_s32(dst_ptr, a_res0);
                vst1q_s32(dst_ptr + 4, a_res1);

                A_tmp += 8;
                B_tmp += 8;
                src_ptr += 8;
                dst_ptr += 8;
                w -= 8;
            } while (w > 0);
        } else {
            do {
                s0 = vld1q_s16(src_ptr);

                const int16x8_t a_sum = vreinterpretq_s16_u16(cross_sum_fast_odd_row_inp16(A_tmp));

                a_res0 = vmull_s16(vget_low_s16(s0), vget_low_s16(a_sum));

                a_res1 = vmull_high_s16(s0, a_sum);

                b_res0 = cross_sum_fast_odd_row(B_tmp);
                b_res1 = cross_sum_fast_odd_row(B_tmp + 4);
                a_res0 = vaddq_s32(a_res0, b_res0);
                a_res1 = vaddq_s32(a_res1, b_res1);

                a_res0 = vrshrq_n_s32(a_res0, SGRPROJ_SGR_BITS + NB_ODD - SGRPROJ_RST_BITS);
                a_res1 = vrshrq_n_s32(a_res1, SGRPROJ_SGR_BITS + NB_ODD - SGRPROJ_RST_BITS);

                vst1q_s32(dst_ptr, a_res0);
                vst1q_s32(dst_ptr + 4, a_res1);

                A_tmp += 8;
                B_tmp += 8;
                src_ptr += 8;
                dst_ptr += 8;
                w -= 8;
            } while (w > 0);
        }
        count++;
        h -= 1;
    } while (h > 0);
}

static void final_filter_internal(uint16_t* A, int32_t* B, const int buf_stride, int16_t* src, const int src_stride,
                                  int32_t* dst, const int dst_stride, const int width, const int height) {
    int16x8_t s0;
    int32_t * B_tmp, *dst_ptr;
    uint16_t* A_tmp;
    int16_t*  src_ptr;
    int32x4_t a_res0, a_res1, b_res0, b_res1;
    int       w, h, count = 0;

    assert(SGRPROJ_SGR_BITS == 8);
    assert(SGRPROJ_RST_BITS == 4);
    h = height;

    do {
        A_tmp   = (A + count * buf_stride);
        B_tmp   = (B + count * buf_stride);
        src_ptr = (src + count * src_stride);
        dst_ptr = (dst + count * dst_stride);
        w       = width;
        do {
            s0 = vld1q_s16(src_ptr);
            // sample (at most 12 bits) times neighbour sum (at most 8192): one widening multiply
            const int16x8_t a_sum = vreinterpretq_s16_u16(cross_sum_inp_u16(A_tmp, buf_stride));

            a_res0 = vmull_s16(vget_low_s16(s0), vget_low_s16(a_sum));

            a_res1 = vmull_high_s16(s0, a_sum);

            b_res0 = cross_sum_inp_s32(B_tmp, buf_stride);
            b_res1 = cross_sum_inp_s32(B_tmp + 4, buf_stride);
            a_res0 = vaddq_s32(a_res0, b_res0);
            a_res1 = vaddq_s32(a_res1, b_res1);

            a_res0 = vrshrq_n_s32(a_res0, SGRPROJ_SGR_BITS + NB_EVEN - SGRPROJ_RST_BITS);
            a_res1 = vrshrq_n_s32(a_res1, SGRPROJ_SGR_BITS + NB_EVEN - SGRPROJ_RST_BITS);
            vst1q_s32(dst_ptr, a_res0);
            vst1q_s32(dst_ptr + 4, a_res1);

            A_tmp += 8;
            B_tmp += 8;
            src_ptr += 8;
            dst_ptr += 8;
            w -= 8;
        } while (w > 0);
        count++;
        h -= 1;
    } while (h > 0);
}

static inline void restoration_fast_internal(uint16_t* dgd16, int width, int height, int dgd_stride, int32_t* dst,
                                             int dst_stride, int bit_depth, int sgr_params_idx, int radius_idx) {
    const SgrParamsType* const params     = &svt_aom_eb_sgr_params[sgr_params_idx];
    const int                  r          = params->r[radius_idx];
    const int                  width_ext  = width + 2 * SGRPROJ_BORDER_HORZ;
    const int                  height_ext = height + 2 * SGRPROJ_BORDER_VERT;

    const int buf_stride = ((width_ext + 3) & ~3) + 16;
    int32_t   A_[RESTORATION_PROC_UNIT_PELS];
    uint16_t  A16_[RESTORATION_PROC_UNIT_PELS];
    int32_t   B_[RESTORATION_PROC_UNIT_PELS];
    int32_t*  square_sum_buf = A_;
    int32_t*  sum_buf        = B_;
    uint16_t* tmp16_buf      = A16_;

    assert(r <= MAX_RADIUS && "Need MAX_RADIUS >= r");
    assert(r <= SGRPROJ_BORDER_VERT - 1 && r <= SGRPROJ_BORDER_HORZ - 1 && "Need SGRPROJ_BORDER_* >= r+1");

    assert(radius_idx == 0);
    assert(r == 2);

    // input(dgd16) is 16bit.
    // sum of pixels 1st stage output will be in 16bit(tmp16_buf). End output is
    // kept in 32bit [sum_buf]. sum of squares output is kept in 32bit
    // buffer(square_sum_buf).
    boxsum2((int16_t*)(dgd16 - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ),
            dgd_stride,
            (int16_t*)tmp16_buf,
            sum_buf,
            square_sum_buf,
            buf_stride,
            width_ext,
            height_ext);

    square_sum_buf += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    sum_buf += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    tmp16_buf += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;

    // Calculation of a, b. a output is in 16bit tmp_buf which is in range of
    // [1, 256] for all bit depths. b output is kept in 32bit buffer.
    if (bit_depth > 8) {
        calc_ab_fast_internal_hbd((square_sum_buf - buf_stride - 1),
                                  (tmp16_buf - buf_stride - 1),
                                  (sum_buf - buf_stride - 1),
                                  (sum_buf - buf_stride - 1),
                                  buf_stride * 2,
                                  width + 2,
                                  height + 2,
                                  bit_depth,
                                  r,
                                  params->s[radius_idx],
                                  2);
    } else {
        calc_ab_fast_internal_lbd((square_sum_buf - buf_stride - 1),
                                  (tmp16_buf - buf_stride - 1),
                                  (sum_buf - buf_stride - 1),
                                  (sum_buf - buf_stride - 1),
                                  buf_stride * 2,
                                  width + 2,
                                  height + 2,
                                  r,
                                  params->s[radius_idx],
                                  2);
    }

    final_filter_fast_internal(
        tmp16_buf, sum_buf, buf_stride, (int16_t*)dgd16, dgd_stride, dst, dst_stride, width, height);
}

static inline void restoration_internal(uint16_t* dgd16, int width, int height, int dgd_stride, int32_t* dst,
                                        int dst_stride, int bit_depth, int sgr_params_idx, int radius_idx) {
    const SgrParamsType* const params     = &svt_aom_eb_sgr_params[sgr_params_idx];
    const int                  r          = params->r[radius_idx];
    const int                  width_ext  = width + 2 * SGRPROJ_BORDER_HORZ;
    const int                  height_ext = height + 2 * SGRPROJ_BORDER_VERT;

    int       buf_stride = ((width_ext + 3) & ~3) + 16;
    int32_t   A_[RESTORATION_PROC_UNIT_PELS];
    uint16_t  A16_[RESTORATION_PROC_UNIT_PELS];
    uint16_t  B16_[RESTORATION_PROC_UNIT_PELS];
    int32_t   B_[RESTORATION_PROC_UNIT_PELS];
    int32_t*  square_sum_buf = A_;
    uint16_t* sum_buf        = B16_;
    uint16_t* A16            = A16_;
    int32_t*  B              = B_;

    assert(r <= MAX_RADIUS && "Need MAX_RADIUS >= r");
    assert(r <= SGRPROJ_BORDER_VERT - 1 && r <= SGRPROJ_BORDER_HORZ - 1 && "Need SGRPROJ_BORDER_* >= r+1");

    assert(radius_idx == 1);
    assert(r == 1);

    // input(dgd16) is 16bit.
    // sum of pixels output will be in 16bit(sum_buf).
    // sum of squares output is kept in 32bit buffer(square_sum_buf).
    boxsum1((int16_t*)(dgd16 - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ),
            dgd_stride,
            sum_buf,
            square_sum_buf,
            buf_stride,
            width_ext,
            height_ext);

    square_sum_buf += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    B += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    A16 += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    sum_buf += SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;

    // Calculation of a, b. a output is in 16bit tmp_buf which is in range of
    // [1, 256] for all bit depths. b output is kept in 32bit buffer.
    if (bit_depth > 8) {
        calc_ab_internal_hbd((square_sum_buf - buf_stride - 1),
                             (A16 - buf_stride - 1),
                             (sum_buf - buf_stride - 1),
                             (B - buf_stride - 1),
                             buf_stride,
                             width + 2,
                             height + 2,
                             bit_depth,
                             r,
                             params->s[radius_idx],
                             1);
    } else {
        calc_ab_internal_lbd((square_sum_buf - buf_stride - 1),
                             (A16 - buf_stride - 1),
                             (sum_buf - buf_stride - 1),
                             (B - buf_stride - 1),
                             buf_stride,
                             width + 2,
                             height + 2,
                             r,
                             params->s[radius_idx],
                             1);
    }

    final_filter_internal(A16, B, buf_stride, (int16_t*)dgd16, dgd_stride, dst, dst_stride, width, height);
}

static inline void src_convert_u8_to_u16(const uint8_t* src, const int src_stride, uint16_t* dst, const int dst_stride,
                                         const int width, const int height) {
    const uint8_t* src_ptr;
    uint16_t*      dst_ptr;
    int            h, w, count = 0;
    int            x, y;

    uint8x8_t  t1, t2, t3, t4;
    uint16x8_t s1, s2, s3, s4;
    h = height;
    do {
        src_ptr = src + (count << 2) * src_stride;
        dst_ptr = dst + (count << 2) * dst_stride;
        w       = width;
        if (w >= 7) {
            do {
                load_u8_8x4(src_ptr, src_stride, &t1, &t2, &t3, &t4);
                s1 = vmovl_u8(t1);
                s2 = vmovl_u8(t2);
                s3 = vmovl_u8(t3);
                s4 = vmovl_u8(t4);
                store_u16_8x4(dst_ptr, dst_stride, s1, s2, s3, s4);

                src_ptr += 8;
                dst_ptr += 8;
                w -= 8;
            } while (w > 7);
        }

        for (y = 0; y < w; y++) {
            dst_ptr[y]                  = src_ptr[y];
            dst_ptr[y + 1 * dst_stride] = src_ptr[y + 1 * src_stride];
            dst_ptr[y + 2 * dst_stride] = src_ptr[y + 2 * src_stride];
            dst_ptr[y + 3 * dst_stride] = src_ptr[y + 3 * src_stride];
        }
        count++;
        h -= 4;
    } while (h > 3);

    src_ptr = src + (count << 2) * src_stride;
    dst_ptr = dst + (count << 2) * dst_stride;
    for (x = 0; x < h; x++) {
        for (y = 0; y < width; y++) {
            dst_ptr[y + x * dst_stride] = src_ptr[y + x * src_stride];
        }
    }

    // memset uninitialized rows of src buffer as they are needed for the
    // boxsum filter calculation.
    for (x = height; x < height + 5; x++) {
        memset(dst + x * dst_stride, 0, (width + 2) * sizeof(*dst));
    }
}

static inline void src_convert_hbd_copy(const uint16_t* src, int src_stride, uint16_t* dst, const int dst_stride,
                                        int width, int height) {
    const uint16_t* src_ptr;
    uint16_t*       dst_ptr;
    int             h, w, count = 0;
    uint16x8_t      s1, s2, s3, s4;
    int             x, y;

    h = height;
    do {
        src_ptr = src + (count << 2) * src_stride;
        dst_ptr = dst + (count << 2) * dst_stride;
        w       = width;
        do {
            load_u16_8x4(src_ptr, src_stride, &s1, &s2, &s3, &s4);
            store_u16_8x4(dst_ptr, dst_stride, s1, s2, s3, s4);
            src_ptr += 8;
            dst_ptr += 8;
            w -= 8;
        } while (w > 7);

        for (y = 0; y < w; y++) {
            dst_ptr[y]                  = src_ptr[y];
            dst_ptr[y + 1 * dst_stride] = src_ptr[y + 1 * src_stride];
            dst_ptr[y + 2 * dst_stride] = src_ptr[y + 2 * src_stride];
            dst_ptr[y + 3 * dst_stride] = src_ptr[y + 3 * src_stride];
        }
        count++;
        h -= 4;
    } while (h > 3);

    src_ptr = src + (count << 2) * src_stride;
    dst_ptr = dst + (count << 2) * dst_stride;

    for (x = 0; x < h; x++) {
        memcpy((dst_ptr + x * dst_stride), (src_ptr + x * src_stride), sizeof(uint16_t) * width);
    }
    // memset uninitialized rows of src buffer as they are needed for the
    // boxsum filter calculation.
    for (x = height; x < height + 5; x++) {
        memset(dst + x * dst_stride, 0, (width + 2) * sizeof(*dst));
    }
}

void svt_av1_selfguided_restoration_neon(const uint8_t* dat8, int32_t width, int32_t height, int32_t stride,
                                         int32_t* flt0, int32_t* flt1, int32_t flt_stride, int32_t sgr_params_idx,
                                         int32_t bit_depth, int32_t highbd) {
    const SgrParamsType* const params = &svt_aom_eb_sgr_params[sgr_params_idx];
    assert(!(params->r[0] == 0 && params->r[1] == 0));

    uint16_t  dgd16_[RESTORATION_PROC_UNIT_PELS];
    const int dgd16_stride = width + 2 * SGRPROJ_BORDER_HORZ;
    uint16_t* dgd16        = dgd16_ + dgd16_stride * SGRPROJ_BORDER_VERT + SGRPROJ_BORDER_HORZ;
    const int width_ext    = width + 2 * SGRPROJ_BORDER_HORZ;
    const int height_ext   = height + 2 * SGRPROJ_BORDER_VERT;
    const int dgd_stride   = stride;

    if (highbd) {
        const uint16_t* dgd16_tmp = CONVERT_TO_SHORTPTR(dat8);
        src_convert_hbd_copy(dgd16_tmp - SGRPROJ_BORDER_VERT * dgd_stride - SGRPROJ_BORDER_HORZ,
                             dgd_stride,
                             dgd16 - SGRPROJ_BORDER_VERT * dgd16_stride - SGRPROJ_BORDER_HORZ,
                             dgd16_stride,
                             width_ext,
                             height_ext);
    } else {
        src_convert_u8_to_u16(dat8 - SGRPROJ_BORDER_VERT * dgd_stride - SGRPROJ_BORDER_HORZ,
                              dgd_stride,
                              dgd16 - SGRPROJ_BORDER_VERT * dgd16_stride - SGRPROJ_BORDER_HORZ,
                              dgd16_stride,
                              width_ext,
                              height_ext);
    }

    if (params->r[0] > 0) {
        restoration_fast_internal(dgd16, width, height, dgd16_stride, flt0, flt_stride, bit_depth, sgr_params_idx, 0);
    }
    if (params->r[1] > 0) {
        restoration_internal(dgd16, width, height, dgd16_stride, flt1, flt_stride, bit_depth, sgr_params_idx, 1);
    }
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// One block of 8 residuals a = flt - (dgd << SGRPROJ_RST_BITS), saturated to int16. As in
// svt_av1_highbd_sgrproj_residual_neon(), the subtraction is done modulo 2^32 and a difference outside the int16
// range leaves INT16_MIN or INT16_MAX in its lane.
static inline int16x8_t sgr_residual_block(const int32_t* flt, const uint16_t* dgd) {
    const uint16x8_t u  = vshlq_n_u16(vld1q_u16(dgd), SGRPROJ_RST_BITS);
    const uint32x4_t lo = vsubw_u16(vreinterpretq_u32_s32(vld1q_s32(flt)), vget_low_u16(u));
    const uint32x4_t hi = vsubw_high_u16(vreinterpretq_u32_s32(vld1q_s32(flt + 4)), u);
    return vqmovn_high_s32(vqmovn_s32(vreinterpretq_s32_u32(lo)), vreinterpretq_s32_u32(hi));
}

// Stores the residual plane of a filter output and folds the stored values into the extremes min and max. Two
// blocks per pass keep separate extremes, so that the passes do not wait for each other.
static inline void sgr_residual(const int32_t* flt, const int flt_stride, const uint16_t* dgd, const int dgd_stride,
                                int16_t* a, const int a_stride, const int width, const int height, int16x8_t* min,
                                int16x8_t* max) {
    const int width8 = width & ~7;
    int16x8_t lo0    = *min;
    int16x8_t hi0    = *max;
    int16x8_t lo1    = lo0;
    int16x8_t hi1    = hi0;

    for (int i = 0; i < height; i++) {
        const int32_t* const  flt_row = flt + i * flt_stride;
        const uint16_t* const dgd_row = dgd + i * dgd_stride;
        int16_t* const        a_row   = a + i * a_stride;
        int                   j       = 0;
        for (; j + 16 <= width8; j += 16) {
            const int16x8_t r0 = sgr_residual_block(flt_row + j, dgd_row + j);
            const int16x8_t r1 = sgr_residual_block(flt_row + j + 8, dgd_row + j + 8);
            hi0                = vmaxq_s16(hi0, r0);
            lo0                = vminq_s16(lo0, r0);
            hi1                = vmaxq_s16(hi1, r1);
            lo1                = vminq_s16(lo1, r1);
            vst1q_s16(a_row + j, r0);
            vst1q_s16(a_row + j + 8, r1);
        }
        if (j < width8) {
            const int16x8_t r0 = sgr_residual_block(flt_row + j, dgd_row + j);
            hi0                = vmaxq_s16(hi0, r0);
            lo0                = vminq_s16(lo0, r0);
            vst1q_s16(a_row + j, r0);
        }
    }
    if (width8 < width) {
        // The samples left over in each row, padded with zeros: a zero residual does not move the extremes.
        const int left = width - width8;
        for (int i = 0; i < height; i++) {
            int32_t  f[8] = {0};
            uint16_t d[8] = {0};
            int16_t  r[8];
            memcpy(f, flt + i * flt_stride + width8, left * sizeof(*f));
            memcpy(d, dgd + i * dgd_stride + width8, left * sizeof(*d));
            const int16x8_t r0 = sgr_residual_block(f, d);
            hi0                = vmaxq_s16(hi0, r0);
            lo0                = vminq_s16(lo0, r0);
            vst1q_s16(r, r0);
            memcpy(a + i * a_stride + width8, r, left * sizeof(*r));
        }
    }
    *min = vminq_s16(lo0, lo1);
    *max = vmaxq_s16(hi0, hi1);
}

// restoration_fast_internal() for n parameter sets, with the box sums computed once and each filter output reduced
// to its residual plane a[2 * k]. The buffers and their layout are those of that function, except that the b plane
// does not replace the box sums, which the next set needs.
static inline void sgr_residuals_fast_internal(uint16_t* dgd16, int width, int height, int dgd_stride, int bit_depth,
                                               const int32_t* sgr_params_idx, int n, int16_t* const* a, int a_stride,
                                               int16x8_t* min, int16x8_t* max) {
    const int width_ext  = width + 2 * SGRPROJ_BORDER_HORZ;
    const int height_ext = height + 2 * SGRPROJ_BORDER_VERT;
    const int buf_stride = ((width_ext + 3) & ~3) + 16;
    const int buf_offset = SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    int32_t   A_[RESTORATION_PROC_UNIT_PELS];
    uint16_t  A16_[RESTORATION_PROC_UNIT_PELS];
    int32_t   sum_[RESTORATION_PROC_UNIT_PELS];
    int32_t   B_[RESTORATION_PROC_UNIT_PELS];
    // The final filter writes whole blocks of 8 samples.
    int32_t   flt[RESTORATION_PROC_UNIT_SIZE * RESTORATION_PROC_UNIT_SIZE];
    const int flt_stride = (width + 7) & ~7;

    boxsum2((int16_t*)(dgd16 - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ),
            dgd_stride,
            (int16_t*)A16_,
            sum_,
            A_,
            buf_stride,
            width_ext,
            height_ext);

    int32_t* const  square_sum_buf = A_ + buf_offset;
    int32_t* const  sum_buf        = sum_ + buf_offset;
    uint16_t* const tmp16_buf      = A16_ + buf_offset;
    int32_t* const  B              = B_ + buf_offset;

    for (int k = 0; k < n; k++) {
        const SgrParamsType* const params = &svt_aom_eb_sgr_params[sgr_params_idx[k]];
        assert(params->r[0] == 2);

        if (bit_depth > 8) {
            calc_ab_fast_internal_hbd((square_sum_buf - buf_stride - 1),
                                      (tmp16_buf - buf_stride - 1),
                                      (sum_buf - buf_stride - 1),
                                      (B - buf_stride - 1),
                                      buf_stride * 2,
                                      width + 2,
                                      height + 2,
                                      bit_depth,
                                      2,
                                      params->s[0],
                                      2);
        } else {
            calc_ab_fast_internal_lbd((square_sum_buf - buf_stride - 1),
                                      (tmp16_buf - buf_stride - 1),
                                      (sum_buf - buf_stride - 1),
                                      (B - buf_stride - 1),
                                      buf_stride * 2,
                                      width + 2,
                                      height + 2,
                                      2,
                                      params->s[0],
                                      2);
        }
        final_filter_fast_internal(
            tmp16_buf, B, buf_stride, (int16_t*)dgd16, dgd_stride, flt, flt_stride, width, height);
        sgr_residual(flt, flt_stride, dgd16, dgd_stride, a[2 * k], a_stride, width, height, min, max);
    }
}

// restoration_internal() for n parameter sets, in the same way; the residual plane of set k is a[2 * k + 1].
static inline void sgr_residuals_internal(uint16_t* dgd16, int width, int height, int dgd_stride, int bit_depth,
                                          const int32_t* sgr_params_idx, int n, int16_t* const* a, int a_stride,
                                          int16x8_t* min, int16x8_t* max) {
    const int width_ext  = width + 2 * SGRPROJ_BORDER_HORZ;
    const int height_ext = height + 2 * SGRPROJ_BORDER_VERT;
    const int buf_stride = ((width_ext + 3) & ~3) + 16;
    const int buf_offset = SGRPROJ_BORDER_VERT * buf_stride + SGRPROJ_BORDER_HORZ;
    int32_t   A_[RESTORATION_PROC_UNIT_PELS];
    uint16_t  A16_[RESTORATION_PROC_UNIT_PELS];
    uint16_t  B16_[RESTORATION_PROC_UNIT_PELS];
    int32_t   B_[RESTORATION_PROC_UNIT_PELS];
    int32_t   flt[RESTORATION_PROC_UNIT_SIZE * RESTORATION_PROC_UNIT_SIZE];
    const int flt_stride = (width + 7) & ~7;

    boxsum1((int16_t*)(dgd16 - dgd_stride * SGRPROJ_BORDER_VERT - SGRPROJ_BORDER_HORZ),
            dgd_stride,
            B16_,
            A_,
            buf_stride,
            width_ext,
            height_ext);

    int32_t* const  square_sum_buf = A_ + buf_offset;
    uint16_t* const sum_buf        = B16_ + buf_offset;
    uint16_t* const A16            = A16_ + buf_offset;
    int32_t* const  B              = B_ + buf_offset;

    for (int k = 0; k < n; k++) {
        const SgrParamsType* const params = &svt_aom_eb_sgr_params[sgr_params_idx[k]];
        assert(params->r[1] == 1);

        if (bit_depth > 8) {
            calc_ab_internal_hbd((square_sum_buf - buf_stride - 1),
                                 (A16 - buf_stride - 1),
                                 (sum_buf - buf_stride - 1),
                                 (B - buf_stride - 1),
                                 buf_stride,
                                 width + 2,
                                 height + 2,
                                 bit_depth,
                                 1,
                                 params->s[1],
                                 1);
        } else {
            calc_ab_internal_lbd((square_sum_buf - buf_stride - 1),
                                 (A16 - buf_stride - 1),
                                 (sum_buf - buf_stride - 1),
                                 (B - buf_stride - 1),
                                 buf_stride,
                                 width + 2,
                                 height + 2,
                                 1,
                                 params->s[1],
                                 1);
        }
        final_filter_internal(A16, B, buf_stride, (int16_t*)dgd16, dgd_stride, flt, flt_stride, width, height);
        sgr_residual(flt, flt_stride, dgd16, dgd_stride, a[2 * k + 1], a_stride, width, height, min, max);
    }
}

// The filter outputs svt_av1_selfguided_restoration() computes for one processing unit of high bit depth samples,
// for n parameter sets that all filter the radii 2 and 1, reduced to the residual planes
// a = flt - (dat << SGRPROJ_RST_BITS) the restoration search works on: a[2 * k] receives the plane of flt0 and
// a[2 * k + 1] the plane of flt1 of the set sgr_params_idx[k]. The box sums depend on the samples and on the radius
// only, so each radius computes them once and every set then derives its a and b planes and its filter output from
// them. Returns whether every residual lies strictly inside the int16 range, in which case the stored planes are
// exact.
bool svt_av1_highbd_selfguided_residuals_neon(const uint8_t* dat8, int32_t width, int32_t height, int32_t stride,
                                              int32_t bit_depth, const int32_t* sgr_params_idx, int32_t n,
                                              int16_t* const* a, int32_t a_stride) {
    assert(width <= RESTORATION_PROC_UNIT_SIZE && height <= RESTORATION_PROC_UNIT_SIZE);

    uint16_t  dgd16_[RESTORATION_PROC_UNIT_PELS];
    const int dgd16_stride = width + 2 * SGRPROJ_BORDER_HORZ;
    uint16_t* dgd16        = dgd16_ + dgd16_stride * SGRPROJ_BORDER_VERT + SGRPROJ_BORDER_HORZ;
    const int width_ext    = width + 2 * SGRPROJ_BORDER_HORZ;
    const int height_ext   = height + 2 * SGRPROJ_BORDER_VERT;

    const uint16_t* dat = CONVERT_TO_SHORTPTR(dat8);
    src_convert_hbd_copy(
        dat - SGRPROJ_BORDER_VERT * stride - SGRPROJ_BORDER_HORZ, stride, dgd16_, dgd16_stride, width_ext, height_ext);

    int16x8_t max = vdupq_n_s16(0);
    int16x8_t min = vdupq_n_s16(0);
    sgr_residuals_fast_internal(
        dgd16, width, height, dgd16_stride, bit_depth, sgr_params_idx, n, a, a_stride, &min, &max);
    sgr_residuals_internal(dgd16, width, height, dgd16_stride, bit_depth, sgr_params_idx, n, a, a_stride, &min, &max);

    return vmaxvq_s16(max) < INT16_MAX && vminvq_s16(min) > INT16_MIN;
}
#endif

typedef struct {
    int r[2]; // radii
} sgr_params_type;

// The 's' values are calculated based on original 'r' and 'e' values in the
// spec using GenSgrprojVtable().
// Note: Setting r = 0 skips the filter; with corresponding s = -1 (invalid).
static const sgr_params_type av1_sgr_params[SGRPROJ_PARAMS] = {
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{2, 1}},
    {{0, 1}},
    {{0, 1}},
    {{0, 1}},
    {{0, 1}},
    {{2, 0}},
    {{2, 0}},
};

static inline void av1_decode_xq(const int* xqd, int* xq, const sgr_params_type* params) {
    if (params->r[0] == 0) {
        xq[0] = 0;
        xq[1] = (1 << SGRPROJ_PRJ_BITS) - xqd[1];
    } else if (params->r[1] == 0) {
        xq[0] = xqd[0];
        xq[1] = 0;
    } else {
        xq[0] = xqd[0];
        xq[1] = (1 << SGRPROJ_PRJ_BITS) - xq[0] - xqd[1];
    }
}

void svt_aom_apply_selfguided_restoration_neon(const uint8_t* dat, int32_t width, int32_t height, int32_t stride,
                                               int32_t eps, const int32_t* xqd, uint8_t* dst, int32_t dst_stride,
                                               int32_t* tmpbuf, int32_t bit_depth, int32_t highbd) {
    int32_t* flt0 = tmpbuf;
    int32_t* flt1 = flt0 + RESTORATION_UNITPELS_MAX;
    assert(width * height <= RESTORATION_UNITPELS_MAX);
    uint16_t                     dgd16_[RESTORATION_PROC_UNIT_PELS];
    const int                    dgd16_stride = width + 2 * SGRPROJ_BORDER_HORZ;
    uint16_t*                    dgd16        = dgd16_ + dgd16_stride * SGRPROJ_BORDER_VERT + SGRPROJ_BORDER_HORZ;
    const int                    width_ext    = width + 2 * SGRPROJ_BORDER_HORZ;
    const int                    height_ext   = height + 2 * SGRPROJ_BORDER_VERT;
    const int                    dgd_stride   = stride;
    const sgr_params_type* const params       = &av1_sgr_params[eps];
    int                          xq[2];

    // these can't be both false at the same time
    assert(!(params->r[0] == 0 && params->r[1] == 0));

    if (highbd) {
        const uint16_t* dgd16_tmp = CONVERT_TO_SHORTPTR(dat);
        src_convert_hbd_copy(dgd16_tmp - SGRPROJ_BORDER_VERT * dgd_stride - SGRPROJ_BORDER_HORZ,
                             dgd_stride,
                             dgd16 - SGRPROJ_BORDER_VERT * dgd16_stride - SGRPROJ_BORDER_HORZ,
                             dgd16_stride,
                             width_ext,
                             height_ext);
    } else {
        src_convert_u8_to_u16(dat - SGRPROJ_BORDER_VERT * dgd_stride - SGRPROJ_BORDER_HORZ,
                              dgd_stride,
                              dgd16 - SGRPROJ_BORDER_VERT * dgd16_stride - SGRPROJ_BORDER_HORZ,
                              dgd16_stride,
                              width_ext,
                              height_ext);
    }

    if (params->r[0] > 0) {
        restoration_fast_internal(dgd16, width, height, dgd16_stride, flt0, width, bit_depth, eps, 0);
    }
    if (params->r[1] > 0) {
        restoration_internal(dgd16, width, height, dgd16_stride, flt1, width, bit_depth, eps, 1);
    }

    av1_decode_xq(xqd, xq, params);

    {
        int16_t*  src_ptr;
        uint8_t*  dst_ptr;
        uint16_t* dst16 = CONVERT_TO_SHORTPTR(dst);
        uint16_t* dst16_ptr;
        int       count = 0, rc = 0;

        const int32x4_t  xq0_vec = vdupq_n_s32(xq[0]);
        const int32x4_t  xq1_vec = vdupq_n_s32(xq[1]);
        const int16x8_t  zero    = vdupq_n_s16(0);
        const uint16x8_t max     = vdupq_n_u16((1 << bit_depth) - 1);
        src_ptr                  = (int16_t*)dgd16;

        do {
            count                = 0;
            const int32_t offset = rc * dst_stride;
            dst_ptr              = dst + offset;
            dst16_ptr            = dst16 + offset;

            do {
                const int16x4x2_t s0 = vld1_s16_x2(src_ptr + count);

                const int32x4_t u0 = vshll_n_s16(s0.val[0], SGRPROJ_RST_BITS);
                const int32x4_t u4 = vshll_n_s16(s0.val[1], SGRPROJ_RST_BITS);

                int32x4_t v0 = vshlq_n_s32(u0, SGRPROJ_PRJ_BITS);
                int32x4_t v4 = vshlq_n_s32(u4, SGRPROJ_PRJ_BITS);

                if (params->r[0] > 0) {
                    const int32x4x2_t fx0 = vld1q_s32_x2(flt0 + count);

                    const int32x4_t f00 = vsubq_s32(fx0.val[0], u0);
                    const int32x4_t f10 = vsubq_s32(fx0.val[1], u4);

                    v0 = vmlaq_s32(v0, xq0_vec, f00);
                    v4 = vmlaq_s32(v4, xq0_vec, f10);
                }

                if (params->r[1] > 0) {
                    const int32x4x2_t fx0 = vld1q_s32_x2(flt1 + count);

                    const int32x4_t f00 = vsubq_s32(fx0.val[0], u0);
                    const int32x4_t f10 = vsubq_s32(fx0.val[1], u4);

                    v0 = vmlaq_s32(v0, xq1_vec, f00);
                    v4 = vmlaq_s32(v4, xq1_vec, f10);
                }

                const int16x4_t d0 = vqrshrn_n_s32(v0, SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
                const int16x4_t d4 = vqrshrn_n_s32(v4, SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);

                const int16x8_t r0 = vcombine_s16(d0, d4);

                uint16x8_t r4 = vreinterpretq_u16_s16(vmaxq_s16(r0, zero));

                if (highbd) {
                    r4 = vminq_u16(r4, max);
                    vst1q_u16(dst16_ptr, r4);
                    dst16_ptr += 8;
                } else {
                    const uint8x8_t t0 = vqmovn_u16(r4);
                    vst1_u8(dst_ptr, t0);
                    dst_ptr += 8;
                }

                count += 8;
            } while (count < width);

            src_ptr += dgd16_stride;
            flt1 += width;
            flt0 += width;
            rc++;
        } while (rc < height);
    }
}
