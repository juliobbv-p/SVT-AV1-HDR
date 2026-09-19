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
#include "cdef.h"
#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "bitstream_unit.h"

static inline int16x8_t constrain_neon(uint16x8_t a, uint16x8_t b, unsigned int threshold, int adjdamp) {
    uint16x8_t       diff   = vabdq_u16(a, b);
    const uint16x8_t a_gt_b = vcgtq_u16(a, b);
    const uint16x8_t s      = vqsubq_u16(vdupq_n_u16(threshold), vshlq_u16(diff, vdupq_n_s16(-adjdamp)));
    const int16x8_t  clip   = vreinterpretq_s16_u16(vminq_u16(diff, s));
    return vbslq_s16(a_gt_b, clip, vnegq_s16(clip));
}

void svt_av1_cdef_filter_block_8xn_8_neon(uint8_t* dst, int32_t dstride, const uint16_t* in, int32_t pri_strength,
                                          int32_t sec_strength, int32_t dir, int32_t pri_damping, int32_t sec_damping,
                                          int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor) {
    int16x8_t  sum, res;
    uint16x8_t max, min, tap, row;
    uint16x8_t large = vdupq_n_u16(CDEF_VERY_LARGE);
    int16x8_t  p0, p1, p2, p3;
    uint8x8_t  ans;
    uint8_t    i;

    const int* pri_taps = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int* sec_taps = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    if (pri_strength) {
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    }
    if (sec_strength) {
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
    }

    for (i = 0; i < height; i += subsampling_factor) {
        sum = vdupq_n_s16(0);
        row = vld1q_u16(in + i * CDEF_BSTRIDE);

        max = min = row;
        if (pri_strength) {
            /*primary near taps*/
            tap = vld1q_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir][0]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, pri_strength, pri_damping);

            tap = vld1q_u16(in + (i * CDEF_BSTRIDE) - svt_aom_eb_cdef_directions[dir][0]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, pri_strength, pri_damping);

            /*sum += (pri_taps[0]*(p0+p1))*/
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(pri_taps[0]), vaddq_s16(p0, p1)));

            /*primary far taps*/
            tap = vld1q_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir][1]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, pri_strength, pri_damping);

            tap = vld1q_u16(in + (i * CDEF_BSTRIDE) - svt_aom_eb_cdef_directions[dir][1]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, pri_strength, pri_damping);

            /*sum += (pri_taps[1]*(p0+p1))*/
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(pri_taps[1]), vaddq_s16(p0, p1)));
        }
        if (sec_strength) {
            /*secondary near taps*/
            tap = vld1q_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir + 2][0]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vld1q_u16(in + (i * CDEF_BSTRIDE) - svt_aom_eb_cdef_directions[dir + 2][0]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vld1q_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir - 2][0]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p2  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vld1q_u16(in + (i * CDEF_BSTRIDE) - svt_aom_eb_cdef_directions[dir - 2][0]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p3  = constrain_neon(tap, row, sec_strength, sec_damping);

            /*sum += (sec_taps[0]*(p0+p1+p2+p3))*/
            p0  = vaddq_s16(p0, p1);
            p2  = vaddq_s16(p2, p3);
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(sec_taps[0]), vaddq_s16(p0, p2)));

            /*secondary far taps*/
            tap = vld1q_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir + 2][1]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vld1q_u16(in + (i * CDEF_BSTRIDE) - svt_aom_eb_cdef_directions[dir + 2][1]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vld1q_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir - 2][1]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p2  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vld1q_u16(in + (i * CDEF_BSTRIDE) - svt_aom_eb_cdef_directions[dir - 2][1]);
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p3  = constrain_neon(tap, row, sec_strength, sec_damping);

            /*sum += (sec_taps[1]*(p0+p1+p2+p3))*/
            p0  = vaddq_s16(p0, p1);
            p2  = vaddq_s16(p2, p3);
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(sec_taps[1]), vaddq_s16(p0, p2)));
        }

        /*res =min(max(row+(sum+8+(sum<0))>>4 ,min),max)*/
        sum = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));
        res = vaddq_s16(sum, vdupq_n_s16(8));
        res = vshrq_n_s16(res, 4);
        res = vaddq_s16(vreinterpretq_s16_u16(row), res);
        res = vminq_s16(vmaxq_s16(res, vreinterpretq_s16_u16(min)), vreinterpretq_s16_u16(max));

        ans = vqmovun_s16(res);
        vst1_u8(dst + (i * dstride), ans);
    }
}

void svt_av1_cdef_filter_block_4xn_8_neon(uint8_t* dst, int32_t dstride, const uint16_t* in, int32_t pri_strength,
                                          int32_t sec_strength, int32_t dir, int32_t pri_damping, int32_t sec_damping,
                                          int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor) {
    int16x8_t  sum, res;
    uint16x8_t max, min, tap, row, large = vdupq_n_u16(CDEF_VERY_LARGE);
    int16x8_t  p0, p1, p2, p3;
    uint8x8_t  ans;
    uint8_t    i;
    uint32_t*  dst_r1_u32;
    uint32_t*  dst_r2_u32;

    const int* pri_taps = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int* sec_taps = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    if (pri_strength) {
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    }
    if (sec_strength) {
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
    }

    for (i = 0; i < height; i += (2 * subsampling_factor)) {
        sum = vdupq_n_s16(0);
        row = vcombine_u16(vld1_u16(in + i * CDEF_BSTRIDE), vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE));
        max = min = row;
        if (pri_strength) {
            /*primary near taps*/
            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir][0]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir][0]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, pri_strength, pri_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir][0]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir][0]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, pri_strength, pri_damping);

            /*sum += (pri_taps[0]*(p0+p1))*/
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(pri_taps[0]), vaddq_s16(p0, p1)));

            /*primary far taps*/
            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir][1]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir][1]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, pri_strength, pri_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir][1]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir][1]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, pri_strength, pri_damping);

            /*sum += (pri_taps[1]*(p0+p1))*/
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(pri_taps[1]), vaddq_s16(p0, p1)));
        }

        if (sec_strength) {
            /*secondary near taps*/
            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir + 2][0]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir + 2][0]));

            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir + 2][0]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir + 2][0]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir - 2][0]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir - 2][0]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p2  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir - 2][0]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir - 2][0]));

            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p3  = constrain_neon(tap, row, sec_strength, sec_damping);

            /*sum += (sec_taps[0]*(p0+p1+p2+p3))*/
            p0  = vaddq_s16(p0, p1);
            p2  = vaddq_s16(p2, p3);
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(sec_taps[0]), vaddq_s16(p0, p2)));

            /*secondary far taps*/
            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir + 2][1]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir + 2][1]));

            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p0  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir + 2][1]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir + 2][1]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p1  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir - 2][1]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE + svt_aom_eb_cdef_directions[dir - 2][1]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p2  = constrain_neon(tap, row, sec_strength, sec_damping);

            tap = vcombine_u16(
                vld1_u16(in + i * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir - 2][1]),
                vld1_u16(in + (i + subsampling_factor) * CDEF_BSTRIDE - svt_aom_eb_cdef_directions[dir - 2][1]));
            max = vmaxq_u16(max, vandq_u16(tap, vmvnq_u16(vceqq_u16(tap, large))));
            min = vminq_u16(min, tap);
            p3  = constrain_neon(tap, row, sec_strength, sec_damping);

            /*sum += (sec_taps[1]*(p0+p1+p2+p3))*/
            p0  = vaddq_s16(p0, p1);
            p2  = vaddq_s16(p2, p3);
            sum = vaddq_s16(sum, vmulq_s16(vdupq_n_s16(sec_taps[1]), vaddq_s16(p0, p2)));
        }
        /*res =min(max(row+(sum+8+(sum<0))>>4 ,min),max)*/
        sum = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));
        res = vaddq_s16(sum, vdupq_n_s16(8));
        res = vshrq_n_s16(res, 4);
        res = vaddq_s16(vreinterpretq_s16_u16(row), res);
        res = vminq_s16(vmaxq_s16(res, vreinterpretq_s16_u16(min)), vreinterpretq_s16_u16(max));

        ans = vqmovun_s16(res);
        /*storing 32 bits in the destination buffer of type uint8_t*/
        dst_r1_u32  = (uint32_t*)(dst + (i * dstride));
        dst_r2_u32  = (uint32_t*)(dst + ((i + subsampling_factor) * dstride));
        *dst_r1_u32 = vget_lane_u32(vreinterpret_u32_u8(ans), 0);
        *dst_r2_u32 = vget_lane_u32(vreinterpret_u32_u8(ans), 1);
    }
}

/* 8-bit-lane constrain (16 px/vec): mirrors constrain_neon in uint8 lanes; used by
 * the native interior kernels. Valid for 8-bit content (real |a-b|<=255, |constrain|<=15). */
static inline int8x16_t constrain8x16(uint8x16_t a, uint8x16_t b, uint8x16_t thr, int8x16_t negdamp) {
    const uint8x16_t diff = vabdq_u8(a, b);
    const uint8x16_t agtb = vcgtq_u8(a, b);
    const uint8x16_t s    = vqsubq_u8(thr, vshlq_u8(diff, negdamp)); // negdamp<0 => right shift
    const int8x16_t  clip = vreinterpretq_s8_u8(vminq_u8(diff, s));
    return vbslq_s8(agtb, clip, vnegq_s8(clip));
}

/* Bit-exact vs svt_cdef_filter_block_c only for fully-interior blocks (edges all present). */
static inline uint8x8_t cdef_finalize8(int16x8_t sum, uint8x8_t row_u8, uint8x8_t min_u8, uint8x8_t max_u8) {
    const int16x8_t row16 = vreinterpretq_s16_u16(vmovl_u8(row_u8));
    const int16x8_t min16 = vreinterpretq_s16_u16(vmovl_u8(min_u8));
    const int16x8_t max16 = vreinterpretq_s16_u16(vmovl_u8(max_u8));
    int16x8_t       s     = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));
    int16x8_t       res   = vaddq_s16(vshrq_n_s16(vaddq_s16(s, vdupq_n_s16(8)), 4), row16);
    res                   = vminq_s16(vmaxq_s16(res, min16), max16);
    return vqmovun_s16(res);
}

static void svt_av1_cdef_filter_block_8xn_8_native_neon(uint8_t* dst, int32_t dstride, const uint8_t* in,
                                                        int32_t pri_strength, int32_t sec_strength, int32_t dir,
                                                        int32_t damping, int32_t coeff_shift, uint8_t height,
                                                        uint8_t subsampling_factor) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;

    const int po1  = svt_aom_eb_cdef_directions[dir][0];
    const int po2  = svt_aom_eb_cdef_directions[dir][1];
    const int s1o1 = svt_aom_eb_cdef_directions[dir + 2][0];
    const int s1o2 = svt_aom_eb_cdef_directions[dir + 2][1];
    const int s2o1 = svt_aom_eb_cdef_directions[dir - 2][0];
    const int s2o2 = svt_aom_eb_cdef_directions[dir - 2][1];

    const uint8x16_t prithr8  = vdupq_n_u8((uint8_t)pri_strength);
    const uint8x16_t secthr8  = vdupq_n_u8((uint8_t)sec_strength);
    const int8x16_t  pridamp8 = vdupq_n_s8((int8_t)-pri_damping);
    const int8x16_t  secdamp8 = vdupq_n_s8((int8_t)-sec_damping);
    const int8x8_t   pri_t0   = vdup_n_s8((int8_t)pri_taps[0]);
    const int8x8_t   pri_t1   = vdup_n_s8((int8_t)pri_taps[1]);
    const int8x8_t   sec_t0   = vdup_n_s8((int8_t)sec_taps[0]);
    const int8x8_t   sec_t1   = vdup_n_s8((int8_t)sec_taps[1]);

    for (uint8_t i = 0; i < height; i += (2 * subsampling_factor)) {
        const uint8_t*   ra   = in + i * CDEF_BSTRIDE;
        const uint8_t*   rb   = in + (i + subsampling_factor) * CDEF_BSTRIDE;
        const uint8x16_t row8 = vcombine_u8(vld1_u8(ra), vld1_u8(rb));
        uint8x16_t       min8 = row8, max8 = row8, tap8;
        int16x8_t        suma = vdupq_n_s16(0), sumb = vdupq_n_s16(0);
        int8x16_t        c0, c1, c2, c3, csum;

#define NAT_TAP(OFF) vcombine_u8(vld1_u8(ra + (OFF)), vld1_u8(rb + (OFF)))

        if (pri_strength) {
            tap8 = NAT_TAP(po1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, prithr8, pridamp8);
            tap8 = NAT_TAP(-po1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, prithr8, pridamp8);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t0, vget_high_s8(csum));
            tap8 = NAT_TAP(po2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, prithr8, pridamp8);
            tap8 = NAT_TAP(-po2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, prithr8, pridamp8);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t1, vget_high_s8(csum));
        }
        if (sec_strength) {
            tap8 = NAT_TAP(s1o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT_TAP(-s1o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT_TAP(s2o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c2   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT_TAP(-s2o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c3   = constrain8x16(tap8, row8, secthr8, secdamp8);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t0, vget_high_s8(csum));
            tap8 = NAT_TAP(s1o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT_TAP(-s1o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT_TAP(s2o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c2   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT_TAP(-s2o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c3   = constrain8x16(tap8, row8, secthr8, secdamp8);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t1, vget_high_s8(csum));
        }
#undef NAT_TAP

        vst1_u8(dst + i * dstride, cdef_finalize8(suma, vget_low_u8(row8), vget_low_u8(min8), vget_low_u8(max8)));
        vst1_u8(dst + (i + subsampling_factor) * dstride,
                cdef_finalize8(sumb, vget_high_u8(row8), vget_high_u8(min8), vget_high_u8(max8)));
    }
}

static void svt_av1_cdef_filter_block_4xn_8_native_neon(uint8_t* dst, int32_t dstride, const uint8_t* in,
                                                        int32_t pri_strength, int32_t sec_strength, int32_t dir,
                                                        int32_t damping, int32_t coeff_shift, uint8_t height,
                                                        uint8_t subsampling_factor) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;

    const int po1  = svt_aom_eb_cdef_directions[dir][0];
    const int po2  = svt_aom_eb_cdef_directions[dir][1];
    const int s1o1 = svt_aom_eb_cdef_directions[dir + 2][0];
    const int s1o2 = svt_aom_eb_cdef_directions[dir + 2][1];
    const int s2o1 = svt_aom_eb_cdef_directions[dir - 2][0];
    const int s2o2 = svt_aom_eb_cdef_directions[dir - 2][1];

    const uint8x16_t prithr8  = vdupq_n_u8((uint8_t)pri_strength);
    const uint8x16_t secthr8  = vdupq_n_u8((uint8_t)sec_strength);
    const int8x16_t  pridamp8 = vdupq_n_s8((int8_t)-pri_damping);
    const int8x16_t  secdamp8 = vdupq_n_s8((int8_t)-sec_damping);
    const int8x8_t   pri_t0   = vdup_n_s8((int8_t)pri_taps[0]);
    const int8x8_t   pri_t1   = vdup_n_s8((int8_t)pri_taps[1]);
    const int8x8_t   sec_t0   = vdup_n_s8((int8_t)sec_taps[0]);
    const int8x8_t   sec_t1   = vdup_n_s8((int8_t)sec_taps[1]);

    const int sub = subsampling_factor;
    // Process 4 rows/iter when they tile the block; else 2-row tail.
    uint8_t i = 0;
    for (; i + 4 * sub <= height; i += (4 * sub)) {
        const uint8_t* r0 = in + (i + 0 * sub) * CDEF_BSTRIDE;
        const uint8_t* r1 = in + (i + 1 * sub) * CDEF_BSTRIDE;
        const uint8_t* r2 = in + (i + 2 * sub) * CDEF_BSTRIDE;
        const uint8_t* r3 = in + (i + 3 * sub) * CDEF_BSTRIDE;
#define NAT4_ROW(p) vreinterpret_u8_u32(vld1q_lane_u32((const uint32_t*)(p) + 0, vdup_n_u32(0), 0))
        // Build row8 = [r0:4 r1:4 | r2:4 r3:4] via 32-bit lane loads.
        uint32x4_t rv         = vdupq_n_u32(0);
        rv                    = vld1q_lane_u32((const uint32_t*)r0, rv, 0);
        rv                    = vld1q_lane_u32((const uint32_t*)r1, rv, 1);
        rv                    = vld1q_lane_u32((const uint32_t*)r2, rv, 2);
        rv                    = vld1q_lane_u32((const uint32_t*)r3, rv, 3);
        const uint8x16_t row8 = vreinterpretq_u8_u32(rv);
        uint8x16_t       min8 = row8, max8 = row8, tap8;
        int16x8_t        suma = vdupq_n_s16(0), sumb = vdupq_n_s16(0);
        int8x16_t        c0, c1, c2, c3, csum;

#define NAT4_TAP(OFF)                                                         \
    ({                                                                        \
        uint32x4_t _t = vdupq_n_u32(0);                                       \
        _t            = vld1q_lane_u32((const uint32_t*)(r0 + (OFF)), _t, 0); \
        _t            = vld1q_lane_u32((const uint32_t*)(r1 + (OFF)), _t, 1); \
        _t            = vld1q_lane_u32((const uint32_t*)(r2 + (OFF)), _t, 2); \
        _t            = vld1q_lane_u32((const uint32_t*)(r3 + (OFF)), _t, 3); \
        vreinterpretq_u8_u32(_t);                                             \
    })

        if (pri_strength) {
            tap8 = NAT4_TAP(po1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, prithr8, pridamp8);
            tap8 = NAT4_TAP(-po1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, prithr8, pridamp8);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t0, vget_high_s8(csum));
            tap8 = NAT4_TAP(po2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, prithr8, pridamp8);
            tap8 = NAT4_TAP(-po2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, prithr8, pridamp8);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t1, vget_high_s8(csum));
        }
        if (sec_strength) {
            tap8 = NAT4_TAP(s1o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT4_TAP(-s1o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT4_TAP(s2o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c2   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT4_TAP(-s2o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c3   = constrain8x16(tap8, row8, secthr8, secdamp8);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t0, vget_high_s8(csum));
            tap8 = NAT4_TAP(s1o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT4_TAP(-s1o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT4_TAP(s2o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c2   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT4_TAP(-s2o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c3   = constrain8x16(tap8, row8, secthr8, secdamp8);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t1, vget_high_s8(csum));
        }
#undef NAT4_TAP
#undef NAT4_ROW

        const uint8x8_t ansAB = cdef_finalize8(suma, vget_low_u8(row8), vget_low_u8(min8), vget_low_u8(max8));
        const uint8x8_t ansCD = cdef_finalize8(sumb, vget_high_u8(row8), vget_high_u8(min8), vget_high_u8(max8));
        *(uint32_t*)(dst + (i + 0 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansAB), 0);
        *(uint32_t*)(dst + (i + 1 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansAB), 1);
        *(uint32_t*)(dst + (i + 2 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansCD), 0);
        *(uint32_t*)(dst + (i + 3 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansCD), 1);
    }
    // 2-row tail (covers height/sub not a multiple of 4, e.g. sub=4 h=8 -> rows 0,4).
    for (; i < height; i += (2 * sub)) {
        const uint8_t* ra     = in + i * CDEF_BSTRIDE;
        const uint8_t* rb     = in + (i + sub) * CDEF_BSTRIDE;
        uint32x2_t     rv     = vdup_n_u32(0);
        rv                    = vld1_lane_u32((const uint32_t*)ra, rv, 0);
        rv                    = vld1_lane_u32((const uint32_t*)rb, rv, 1);
        const uint8x8_t  rlo  = vreinterpret_u8_u32(rv);
        const uint8x16_t row8 = vcombine_u8(rlo, rlo);
        uint8x16_t       min8 = row8, max8 = row8, tap8;
        int16x8_t        suma = vdupq_n_s16(0);
        int8x16_t        c0, c1, c2, c3, csum;
#define NAT2_TAP(OFF)                                                             \
    ({                                                                            \
        uint32x2_t _t      = vdup_n_u32(0);                                       \
        _t                 = vld1_lane_u32((const uint32_t*)(ra + (OFF)), _t, 0); \
        _t                 = vld1_lane_u32((const uint32_t*)(rb + (OFF)), _t, 1); \
        const uint8x8_t _l = vreinterpret_u8_u32(_t);                             \
        vcombine_u8(_l, _l);                                                      \
    })
        if (pri_strength) {
            tap8 = NAT2_TAP(po1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, prithr8, pridamp8);
            tap8 = NAT2_TAP(-po1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, prithr8, pridamp8);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t0, vget_low_s8(csum));
            tap8 = NAT2_TAP(po2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, prithr8, pridamp8);
            tap8 = NAT2_TAP(-po2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, prithr8, pridamp8);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t1, vget_low_s8(csum));
        }
        if (sec_strength) {
            tap8 = NAT2_TAP(s1o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT2_TAP(-s1o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT2_TAP(s2o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c2   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT2_TAP(-s2o1);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c3   = constrain8x16(tap8, row8, secthr8, secdamp8);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t0, vget_low_s8(csum));
            tap8 = NAT2_TAP(s1o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c0   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT2_TAP(-s1o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c1   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT2_TAP(s2o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c2   = constrain8x16(tap8, row8, secthr8, secdamp8);
            tap8 = NAT2_TAP(-s2o2);
            min8 = vminq_u8(min8, tap8);
            max8 = vmaxq_u8(max8, tap8);
            c3   = constrain8x16(tap8, row8, secthr8, secdamp8);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t1, vget_low_s8(csum));
        }
#undef NAT2_TAP
        const uint8x8_t ans             = cdef_finalize8(suma, vget_low_u8(row8), vget_low_u8(min8), vget_low_u8(max8));
        *(uint32_t*)(dst + i * dstride) = vget_lane_u32(vreinterpret_u32_u8(ans), 0);
        *(uint32_t*)(dst + (i + sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ans), 1);
    }
}

AOM_FORCE_INLINE int16x8_t constrain16(int16x8_t a, int16x8_t b, int16x8_t threshold, int16x8_t adjdamp) {
    const int16x8_t sign = vreinterpretq_s16_u16(vcltq_s16(a, b));

    const int16x8_t abs_diff = vabdq_s16(a, b);
    const int16x8_t s        = vreinterpretq_s16_u16(
        vqsubq_u16(vreinterpretq_u16_s16(threshold), vreinterpretq_u16_s16(vshlq_s16(abs_diff, adjdamp))));

    // invert result if sign was negative
    return veorq_s16(vaddq_s16(vminq_s16(abs_diff, s), sign), sign);
}

void svt_av1_cdef_filter_block_8xn_16_neon(uint16_t* dst, int dstride, const uint16_t* in, int pri_strength,
                                           int sec_strength, int dir, int pri_damping, int sec_damping, int coeff_shift,
                                           uint8_t height, uint8_t subsampling_factor) {
    int             i;
    const int16x8_t large = vdupq_n_s16(CDEF_VERY_LARGE);
    const int32_t   po1   = svt_aom_eb_cdef_directions[dir][0];
    const int32_t   po2   = svt_aom_eb_cdef_directions[dir][1];
    const int32_t   s1o1  = svt_aom_eb_cdef_directions[(dir + 2)][0];
    const int32_t   s1o2  = svt_aom_eb_cdef_directions[(dir + 2)][1];
    const int32_t   s2o1  = svt_aom_eb_cdef_directions[(dir - 2)][0];
    const int32_t   s2o2  = svt_aom_eb_cdef_directions[(dir - 2)][1];

    const int* pri_taps = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int* sec_taps = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    if (pri_strength) {
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    }
    if (sec_strength) {
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
    }

    const int16x8_t v_pri_strength = vdupq_n_s16(pri_strength);
    const int16x8_t v_sec_strength = vdupq_n_s16(sec_strength);
    const int16x8_t v_pri_damping  = vdupq_n_s16(-pri_damping);
    const int16x8_t v_sec_damping  = vdupq_n_s16(-sec_damping);

    for (i = 0; i < height; i += subsampling_factor) {
        const int16_t* ina = (const int16_t*)(in + i * CDEF_BSTRIDE);

        const int16x8_t row = vld1q_s16(ina);

        int16x8_t sum1, sum2, sum3, sum4;
        int16x8_t max1, max2, max3, max4;
        int16x8_t min1, min2, min3, min4;

        // Primary near taps
        {
            const int16x8_t p0 = vld1q_s16(&ina[po1]);
            const int16x8_t p1 = vld1q_s16(&ina[-po1]);

            max1 = vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                             vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large))));
            min1 = vminq_s16(p0, p1);

            const int16x8_t constrained_p0 = constrain16(p0, row, v_pri_strength, v_pri_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_pri_strength, v_pri_damping);

            // sum += pri_taps[0] * (p0 + p1)
            sum1 = vmulq_s16(vdupq_n_s16(pri_taps[0]), vaddq_s16(constrained_p0, constrained_p1));
        }

        // Primary far taps
        {
            const int16x8_t p0 = vld1q_s16(&ina[po2]);
            const int16x8_t p1 = vld1q_s16(&ina[-po2]);

            max2 = vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                             vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large))));
            min2 = vminq_s16(p0, p1);

            const int16x8_t constrained_p0 = constrain16(p0, row, v_pri_strength, v_pri_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_pri_strength, v_pri_damping);

            // sum += pri_taps[1] * (p0 + p1)
            sum2 = vmulq_s16(vdupq_n_s16(pri_taps[1]), vaddq_s16(constrained_p0, constrained_p1));
        }

        // Secondary near taps
        {
            const int16x8_t p0 = vld1q_s16(&ina[s1o1]);
            const int16x8_t p1 = vld1q_s16(&ina[-s1o1]);
            const int16x8_t p2 = vld1q_s16(&ina[s2o1]);
            const int16x8_t p3 = vld1q_s16(&ina[-s2o1]);

            max3 = vmaxq_s16(vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                                       vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large)))),
                             vmaxq_s16(vbicq_s16(p2, vreinterpretq_s16_u16(vceqq_s16(p2, large))),
                                       vbicq_s16(p3, vreinterpretq_s16_u16(vceqq_s16(p3, large)))));
            min3 = vminq_s16(vminq_s16(p0, p1), vminq_s16(p2, p3));

            const int16x8_t constrained_p0 = constrain16(p0, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p2 = constrain16(p2, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p3 = constrain16(p3, row, v_sec_strength, v_sec_damping);

            // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
            sum3 = vmulq_s16(
                vdupq_n_s16(sec_taps[0]),
                vaddq_s16(vaddq_s16(constrained_p0, constrained_p1), vaddq_s16(constrained_p2, constrained_p3)));
        }

        // Secondary far taps
        {
            const int16x8_t p0 = vld1q_s16(&ina[s1o2]);
            const int16x8_t p1 = vld1q_s16(&ina[-s1o2]);
            const int16x8_t p2 = vld1q_s16(&ina[s2o2]);
            const int16x8_t p3 = vld1q_s16(&ina[-s2o2]);

            max4 = vmaxq_s16(vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                                       vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large)))),
                             vmaxq_s16(vbicq_s16(p2, vreinterpretq_s16_u16(vceqq_s16(p2, large))),
                                       vbicq_s16(p3, vreinterpretq_s16_u16(vceqq_s16(p3, large)))));
            min4 = vminq_s16(vminq_s16(p0, p1), vminq_s16(p2, p3));

            const int16x8_t constrained_p0 = constrain16(p0, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p2 = constrain16(p2, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p3 = constrain16(p3, row, v_sec_strength, v_sec_damping);

            // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
            sum4 = vmulq_s16(
                vdupq_n_s16(sec_taps[1]),
                vaddq_s16(vaddq_s16(constrained_p0, constrained_p1), vaddq_s16(constrained_p2, constrained_p3)));
        }

        const int16x8_t max = vmaxq_s16(row, vmaxq_s16(vmaxq_s16(max1, max2), vmaxq_s16(max3, max4)));
        const int16x8_t min = vminq_s16(row, vminq_s16(vminq_s16(min1, min2), vminq_s16(min3, min4)));

        int16x8_t sum = vaddq_s16(vaddq_s16(sum1, sum2), vaddq_s16(sum3, sum4));

        // res = row + ((sum - (sum < 0) + 8) >> 4)
        sum = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));

        int16x8_t res = vrshrq_n_s16(sum, 4);
        res           = vaddq_s16(row, res);
        res           = vminq_s16(vmaxq_s16(res, min), max);

        vst1q_s16((int16_t*)(dst + i * dstride), res);
    }
}

void svt_av1_cdef_filter_block_4xn_16_neon(uint16_t* dst, int dstride, const uint16_t* in, int pri_strength,
                                           int sec_strength, int dir, int pri_damping, int sec_damping, int coeff_shift,
                                           uint8_t height, uint8_t subsampling_factor) {
    int             i;
    const int16x8_t large = vdupq_n_s16(CDEF_VERY_LARGE);
    const int32_t   po1   = svt_aom_eb_cdef_directions[dir][0];
    const int32_t   po2   = svt_aom_eb_cdef_directions[dir][1];
    const int32_t   s1o1  = svt_aom_eb_cdef_directions[(dir + 2)][0];
    const int32_t   s1o2  = svt_aom_eb_cdef_directions[(dir + 2)][1];
    const int32_t   s2o1  = svt_aom_eb_cdef_directions[(dir - 2)][0];
    const int32_t   s2o2  = svt_aom_eb_cdef_directions[(dir - 2)][1];

    const int* pri_taps = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int* sec_taps = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];

    if (pri_strength) {
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    }
    if (sec_strength) {
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
    }

    const int16x8_t v_pri_strength = vdupq_n_s16(pri_strength);
    const int16x8_t v_sec_strength = vdupq_n_s16(sec_strength);
    const int16x8_t v_pri_damping  = vdupq_n_s16(-pri_damping);
    const int16x8_t v_sec_damping  = vdupq_n_s16(-sec_damping);

    for (i = 0; i < height; i += (2 * subsampling_factor)) {
        const int16_t* ina = (const int16_t*)(in + i * CDEF_BSTRIDE);
        const int16_t* inb = (const int16_t*)(in + (i + 1 * subsampling_factor) * CDEF_BSTRIDE);

        const int16x8_t row = vcombine_s16(vld1_s16(ina), vld1_s16(inb));

        int16x8_t sum1, sum2, sum3, sum4;
        int16x8_t max1, max2, max3, max4;
        int16x8_t min1, min2, min3, min4;

        // Primary near taps
        {
            const int16x8_t p0 = vcombine_s16(vld1_s16(&ina[po1]), vld1_s16(&inb[po1]));
            const int16x8_t p1 = vcombine_s16(vld1_s16(&ina[-po1]), vld1_s16(&inb[-po1]));

            max1 = vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                             vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large))));
            min1 = vminq_s16(p0, p1);

            const int16x8_t constrained_p0 = constrain16(p0, row, v_pri_strength, v_pri_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_pri_strength, v_pri_damping);

            // sum += pri_taps[0] * (p0 + p1)
            sum1 = vmulq_s16(vdupq_n_s16(pri_taps[0]), vaddq_s16(constrained_p0, constrained_p1));
        }

        // Primary far taps
        {
            const int16x8_t p0 = vcombine_s16(vld1_s16(&ina[po2]), vld1_s16(&inb[po2]));
            const int16x8_t p1 = vcombine_s16(vld1_s16(&ina[-po2]), vld1_s16(&inb[-po2]));

            max2 = vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                             vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large))));
            min2 = vminq_s16(p0, p1);

            const int16x8_t constrained_p0 = constrain16(p0, row, v_pri_strength, v_pri_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_pri_strength, v_pri_damping);

            // sum += pri_taps[1] * (p0 + p1)
            sum2 = vmulq_s16(vdupq_n_s16(pri_taps[1]), vaddq_s16(constrained_p0, constrained_p1));
        }

        // Secondary near taps
        {
            const int16x8_t p0 = vcombine_s16(vld1_s16(&ina[s1o1]), vld1_s16(&inb[s1o1]));
            const int16x8_t p1 = vcombine_s16(vld1_s16(&ina[-s1o1]), vld1_s16(&inb[-s1o1]));
            const int16x8_t p2 = vcombine_s16(vld1_s16(&ina[s2o1]), vld1_s16(&inb[s2o1]));
            const int16x8_t p3 = vcombine_s16(vld1_s16(&ina[-s2o1]), vld1_s16(&inb[-s2o1]));

            max3 = vmaxq_s16(vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                                       vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large)))),
                             vmaxq_s16(vbicq_s16(p2, vreinterpretq_s16_u16(vceqq_s16(p2, large))),
                                       vbicq_s16(p3, vreinterpretq_s16_u16(vceqq_s16(p3, large)))));
            min3 = vminq_s16(vminq_s16(p0, p1), vminq_s16(p2, p3));

            const int16x8_t constrained_p0 = constrain16(p0, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p2 = constrain16(p2, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p3 = constrain16(p3, row, v_sec_strength, v_sec_damping);

            // sum += sec_taps[0] * (p0 + p1 + p2 + p3)
            sum3 = vmulq_s16(
                vdupq_n_s16(sec_taps[0]),
                vaddq_s16(vaddq_s16(constrained_p0, constrained_p1), vaddq_s16(constrained_p2, constrained_p3)));
        }

        // Secondary far taps
        {
            const int16x8_t p0 = vcombine_s16(vld1_s16(&ina[s1o2]), vld1_s16(&inb[s1o2]));
            const int16x8_t p1 = vcombine_s16(vld1_s16(&ina[-s1o2]), vld1_s16(&inb[-s1o2]));
            const int16x8_t p2 = vcombine_s16(vld1_s16(&ina[s2o2]), vld1_s16(&inb[s2o2]));
            const int16x8_t p3 = vcombine_s16(vld1_s16(&ina[-s2o2]), vld1_s16(&inb[-s2o2]));

            max4 = vmaxq_s16(vmaxq_s16(vbicq_s16(p0, vreinterpretq_s16_u16(vceqq_s16(p0, large))),
                                       vbicq_s16(p1, vreinterpretq_s16_u16(vceqq_s16(p1, large)))),
                             vmaxq_s16(vbicq_s16(p2, vreinterpretq_s16_u16(vceqq_s16(p2, large))),
                                       vbicq_s16(p3, vreinterpretq_s16_u16(vceqq_s16(p3, large)))));
            min4 = vminq_s16(vminq_s16(p0, p1), vminq_s16(p2, p3));

            const int16x8_t constrained_p0 = constrain16(p0, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p1 = constrain16(p1, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p2 = constrain16(p2, row, v_sec_strength, v_sec_damping);
            const int16x8_t constrained_p3 = constrain16(p3, row, v_sec_strength, v_sec_damping);

            // sum += sec_taps[1] * (p0 + p1 + p2 + p3)
            sum4 = vmulq_s16(
                vdupq_n_s16(sec_taps[1]),
                vaddq_s16(vaddq_s16(constrained_p0, constrained_p1), vaddq_s16(constrained_p2, constrained_p3)));
        }

        const int16x8_t max = vmaxq_s16(row, vmaxq_s16(vmaxq_s16(max1, max2), vmaxq_s16(max3, max4)));
        const int16x8_t min = vminq_s16(row, vminq_s16(vminq_s16(min1, min2), vminq_s16(min3, min4)));

        int16x8_t sum = vaddq_s16(vaddq_s16(sum1, sum2), vaddq_s16(sum3, sum4));

        // res = row + ((sum - (sum < 0) + 8) >> 4)
        sum = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));

        int16x8_t res = vrshrq_n_s16(sum, 4);
        res           = vaddq_s16(row, res);
        res           = vminq_s16(vmaxq_s16(res, min), max);

        vst1_s16((int16_t*)(dst + i * dstride), vget_low_s16(res));
        vst1_s16((int16_t*)(dst + (i + 1 * subsampling_factor) * dstride), vget_high_s16(res));
    }
}

void svt_cdef_filter_block_neon(uint8_t* dst8, uint16_t* dst16, int32_t dstride, const uint16_t* in,
                                int32_t pri_strength, int32_t sec_strength, int32_t dir, int32_t pri_damping,
                                int32_t sec_damping, int32_t bsize, int32_t coeff_shift, uint8_t subsampling_factor) {
    if (dst8) {
        if (bsize == BLOCK_8X8) {
            svt_av1_cdef_filter_block_8xn_8_neon(dst8,
                                                 dstride,
                                                 in,
                                                 pri_strength,
                                                 sec_strength,
                                                 dir,
                                                 pri_damping,
                                                 sec_damping,
                                                 coeff_shift,
                                                 8,
                                                 subsampling_factor);
        } else if (bsize == BLOCK_4X8) {
            svt_av1_cdef_filter_block_4xn_8_neon(dst8,
                                                 dstride,
                                                 in,
                                                 pri_strength,
                                                 sec_strength,
                                                 dir,
                                                 pri_damping,
                                                 sec_damping,
                                                 coeff_shift,
                                                 8,
                                                 subsampling_factor);
        } else if (bsize == BLOCK_8X4) {
            svt_av1_cdef_filter_block_8xn_8_neon(dst8,
                                                 dstride,
                                                 in,
                                                 pri_strength,
                                                 sec_strength,
                                                 dir,
                                                 pri_damping,
                                                 sec_damping,
                                                 coeff_shift,
                                                 4,
                                                 subsampling_factor);
        } else {
            svt_av1_cdef_filter_block_4xn_8_neon(
                dst8, dstride, in, pri_strength, sec_strength, dir, pri_damping, sec_damping, coeff_shift, 4, 1);
        }
    } else {
        if (bsize == BLOCK_8X8) {
            svt_av1_cdef_filter_block_8xn_16_neon(dst16,
                                                  dstride,
                                                  in,
                                                  pri_strength,
                                                  sec_strength,
                                                  dir,
                                                  pri_damping,
                                                  sec_damping,
                                                  coeff_shift,
                                                  8,
                                                  subsampling_factor);
        } else if (bsize == BLOCK_4X8) {
            svt_av1_cdef_filter_block_4xn_16_neon(dst16,
                                                  dstride,
                                                  in,
                                                  pri_strength,
                                                  sec_strength,
                                                  dir,
                                                  pri_damping,
                                                  sec_damping,
                                                  coeff_shift,
                                                  8,
                                                  subsampling_factor);
        } else if (bsize == BLOCK_8X4) {
            svt_av1_cdef_filter_block_8xn_16_neon(dst16,
                                                  dstride,
                                                  in,
                                                  pri_strength,
                                                  sec_strength,
                                                  dir,
                                                  pri_damping,
                                                  sec_damping,
                                                  coeff_shift,
                                                  4,
                                                  subsampling_factor);
        } else {
            assert(bsize == BLOCK_4X4);
            svt_av1_cdef_filter_block_4xn_16_neon(
                dst16, dstride, in, pri_strength, sec_strength, dir, pri_damping, sec_damping, coeff_shift, 4, 1);
        }
    }
}

// Native 8-bit interior filter, dispatched by bsize. C ref: svt_cdef_filter_block_8bit_c.
// Height passed as a literal so LTO can specialize/unroll the kernel row loop (~1% at M11).
void svt_cdef_filter_block_8bit_neon(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                     int32_t sec_strength, int32_t dir, int32_t damping, int32_t bsize,
                                     int32_t coeff_shift, uint8_t subsampling_factor) {
    if (bsize == BLOCK_8X8) {
        svt_av1_cdef_filter_block_8xn_8_native_neon(
            dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8, subsampling_factor);
    } else if (bsize == BLOCK_4X8) {
        svt_av1_cdef_filter_block_4xn_8_native_neon(
            dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8, subsampling_factor);
    } else if (bsize == BLOCK_8X4) {
        svt_av1_cdef_filter_block_8xn_8_native_neon(
            dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4, subsampling_factor);
    } else {
        svt_av1_cdef_filter_block_4xn_8_native_neon(
            dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4, 1);
    }
}

// ---------------------------------------------------------------------------
// Boundary-aware native 8-bit kernels (frame-perimeter "ring" blocks).
// Identical math to the native kernels above, but each tap is masked per-lane by geometry so
// off-frame taps are excluded from sum (constrained contribution zeroed), max (off lanes -> 0) and
// min (off lanes -> 255). Bit-exact with svt_cdef_filter_block_8bit_bounded_c.
// ---------------------------------------------------------------------------

// Per-column availability (0xFF in-frame / 0x00 off-frame) for a tap with column delta dc.
static inline uint8x8_t bnd_colmask8(int dc, int edge_left, int edge_right, int cols) {
    if (!edge_left && !edge_right) {
        return vdup_n_u8(0xFF); // no left/right frame edge -> all columns in-frame (common case)
    }
    uint8_t m[8];
    for (int c = 0; c < 8; c++) {
        const int off = (edge_left && (c + dc) < 0) || (edge_right && (c + dc) >= cols);
        m[c]          = (!off) ? 0xFF : 0x00;
    }
    return vld1_u8(m);
}

// 16-lane availability for a 2-row pack (rows i and i+sub) given a tap row delta dr and the tap's
// precomputed per-column mask cav. avail(lane) = row_in_frame(row) & col_in_frame(col).
static inline uint8x16_t bnd_avail16(int i, int sub, int dr, uint8x8_t cav, int edge_top, int edge_bottom, int rows) {
    const int       lo = !((edge_top && (i + dr) < 0) || (edge_bottom && (i + dr) >= rows));
    const int       hi = !((edge_top && (i + sub + dr) < 0) || (edge_bottom && (i + sub + dr) >= rows));
    const uint8x8_t z  = vdup_n_u8(0);
    return vcombine_u8(lo ? cav : z, hi ? cav : z);
}

static void cdef_filter_block_8xn_8_bounded_neon(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                                 int32_t sec_strength, int32_t dir, int32_t damping,
                                                 int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor,
                                                 int edge_top, int edge_left, int edge_bottom, int edge_right) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const int     rows        = height;
    const int     cols        = 8;

    // Flat tap offsets (memory) and decoded (drow,dcol) (geometry), indexed [0..5]:
    // 0=po1 1=po2 2=s1o1 3=s1o2 4=s2o1 5=s2o2.
    const int off_b[6] = {svt_aom_eb_cdef_directions[dir][0],
                          svt_aom_eb_cdef_directions[dir][1],
                          svt_aom_eb_cdef_directions[dir + 2][0],
                          svt_aom_eb_cdef_directions[dir + 2][1],
                          svt_aom_eb_cdef_directions[dir - 2][0],
                          svt_aom_eb_cdef_directions[dir - 2][1]};
    const int dr_b[6]  = {svt_aom_eb_cdef_directions_rc[dir][0][0],
                          svt_aom_eb_cdef_directions_rc[dir][1][0],
                          svt_aom_eb_cdef_directions_rc[dir + 2][0][0],
                          svt_aom_eb_cdef_directions_rc[dir + 2][1][0],
                          svt_aom_eb_cdef_directions_rc[dir - 2][0][0],
                          svt_aom_eb_cdef_directions_rc[dir - 2][1][0]};
    const int dc_b[6]  = {svt_aom_eb_cdef_directions_rc[dir][0][1],
                          svt_aom_eb_cdef_directions_rc[dir][1][1],
                          svt_aom_eb_cdef_directions_rc[dir + 2][0][1],
                          svt_aom_eb_cdef_directions_rc[dir + 2][1][1],
                          svt_aom_eb_cdef_directions_rc[dir - 2][0][1],
                          svt_aom_eb_cdef_directions_rc[dir - 2][1][1]};
    // Per-tap column masks for the +offset and -offset variants (loop-invariant). Also precompute,
    // per tap, whether masking is needed at all: a tap is off-frame only if it crosses an active
    // frame edge (col-sensitive: edge_left&&dc<0 or edge_right&&dc>0; row-sensitive: edge_top&&dr<0
    // or edge_bottom&&dr>0). Fully in-frame taps take the native (unmasked) path; col-only taps use
    // the loop-invariant col mask; only row-sensitive taps build a per-row mask.
    uint8x8_t cavp[6], cavn[6];
    int       rsensp[6], rsensn[6], needp[6], needn[6];
    for (int b = 0; b < 6; b++) {
        cavp[b]       = bnd_colmask8(dc_b[b], edge_left, edge_right, cols);
        cavn[b]       = bnd_colmask8(-dc_b[b], edge_left, edge_right, cols);
        const int csp = (edge_left && dc_b[b] < 0) || (edge_right && dc_b[b] > 0);
        const int csn = (edge_left && -dc_b[b] < 0) || (edge_right && -dc_b[b] > 0);
        rsensp[b]     = (edge_top && dr_b[b] < 0) || (edge_bottom && dr_b[b] > 0);
        rsensn[b]     = (edge_top && -dr_b[b] < 0) || (edge_bottom && -dr_b[b] > 0);
        needp[b]      = csp || rsensp[b];
        needn[b]      = csn || rsensn[b];
    }

    const uint8x16_t prithr8  = vdupq_n_u8((uint8_t)pri_strength);
    const uint8x16_t secthr8  = vdupq_n_u8((uint8_t)sec_strength);
    const int8x16_t  pridamp8 = vdupq_n_s8((int8_t)-pri_damping);
    const int8x16_t  secdamp8 = vdupq_n_s8((int8_t)-sec_damping);
    const int8x8_t   pri_t0   = vdup_n_s8((int8_t)pri_taps[0]);
    const int8x8_t   pri_t1   = vdup_n_s8((int8_t)pri_taps[1]);
    const int8x8_t   sec_t0   = vdup_n_s8((int8_t)sec_taps[0]);
    const int8x8_t   sec_t1   = vdup_n_s8((int8_t)sec_taps[1]);

    for (uint8_t i = 0; i < height; i += (2 * subsampling_factor)) {
        const uint8_t*   ra   = in + i * CDEF_BSTRIDE;
        const uint8_t*   rb   = in + (i + subsampling_factor) * CDEF_BSTRIDE;
        const uint8x16_t row8 = vcombine_u8(vld1_u8(ra), vld1_u8(rb));
        uint8x16_t       min8 = row8, max8 = row8, tap8, av;
        int16x8_t        suma = vdupq_n_s16(0), sumb = vdupq_n_s16(0);
        int8x16_t        c0, c1, c2, c3, csum;

#define NAT_TAP(OFF) vcombine_u8(vld1_u8(ra + (OFF)), vld1_u8(rb + (OFF)))
// Masked tap: B = base index, SGN = +1 (use +off,+dr,cavp) or -1 (use -off,-dr,cavn). Fully in-frame
// taps skip masking entirely; col-only taps reuse the loop-invariant col mask; row-sensitive taps
// build the per-row mask via bnd_avail16.
#define BND_TAP(B, SGN, THR, DAMP, CACC)                                                                          \
    do {                                                                                                          \
        const uint8x8_t _cav  = (SGN) > 0 ? cavp[B] : cavn[B];                                                    \
        const int       _need = (SGN) > 0 ? needp[B] : needn[B];                                                  \
        const int       _rs   = (SGN) > 0 ? rsensp[B] : rsensn[B];                                                \
        tap8                  = NAT_TAP((SGN) * off_b[B]);                                                        \
        if (!_need) {                                                                                             \
            max8   = vmaxq_u8(max8, tap8);                                                                        \
            min8   = vminq_u8(min8, tap8);                                                                        \
            (CACC) = constrain8x16(tap8, row8, (THR), (DAMP));                                                    \
        } else {                                                                                                  \
            av     = _rs ? bnd_avail16(i, subsampling_factor, (SGN) * dr_b[B], _cav, edge_top, edge_bottom, rows) \
                         : vcombine_u8(_cav, _cav);                                                               \
            max8   = vmaxq_u8(max8, vandq_u8(tap8, av));                                                          \
            min8   = vminq_u8(min8, vorrq_u8(tap8, vmvnq_u8(av)));                                                \
            (CACC) = vandq_s8(constrain8x16(tap8, row8, (THR), (DAMP)), vreinterpretq_s8_u8(av));                 \
        }                                                                                                         \
    } while (0)

        if (pri_strength) {
            BND_TAP(0, 1, prithr8, pridamp8, c0);
            BND_TAP(0, -1, prithr8, pridamp8, c1);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t0, vget_high_s8(csum));
            BND_TAP(1, 1, prithr8, pridamp8, c0);
            BND_TAP(1, -1, prithr8, pridamp8, c1);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t1, vget_high_s8(csum));
        }
        if (sec_strength) {
            BND_TAP(2, 1, secthr8, secdamp8, c0);
            BND_TAP(2, -1, secthr8, secdamp8, c1);
            BND_TAP(4, 1, secthr8, secdamp8, c2);
            BND_TAP(4, -1, secthr8, secdamp8, c3);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t0, vget_high_s8(csum));
            BND_TAP(3, 1, secthr8, secdamp8, c0);
            BND_TAP(3, -1, secthr8, secdamp8, c1);
            BND_TAP(5, 1, secthr8, secdamp8, c2);
            BND_TAP(5, -1, secthr8, secdamp8, c3);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t1, vget_high_s8(csum));
        }
#undef BND_TAP
#undef NAT_TAP

        vst1_u8(dst + i * dstride, cdef_finalize8(suma, vget_low_u8(row8), vget_low_u8(min8), vget_low_u8(max8)));
        vst1_u8(dst + (i + subsampling_factor) * dstride,
                cdef_finalize8(sumb, vget_high_u8(row8), vget_high_u8(min8), vget_high_u8(max8)));
    }
}

// 4-wide column availability replicated across a 16-lane pack: [m0m1m2m3] x4.
static inline uint8x16_t bnd_col16_4(int dc, int edge_left, int edge_right) {
    if (!edge_left && !edge_right) {
        return vdupq_n_u8(0xFF); // no left/right frame edge -> all columns in-frame (common case)
    }
    uint8_t m[4];
    for (int c = 0; c < 4; c++) {
        const int off = (edge_left && (c + dc) < 0) || (edge_right && (c + dc) >= 4);
        m[c]          = (!off) ? 0xFF : 0x00;
    }
    uint32_t w;
    memcpy(&w, m, 4);
    const uint8x8_t lo = vreinterpret_u8_u32(vdup_n_u32(w));
    return vcombine_u8(lo, lo);
}

// Row availability for the 4-row pack [r0 r1 r2 r3] (rows i+q*sub), 4 lanes per row.
static inline uint8x16_t bnd_row16_4(int i, int sub, int dr, int edge_top, int edge_bottom, int rows) {
    uint8_t b[16];
    for (int k = 0; k < 4; k++) {
        const int     r  = i + k * sub;
        const int     ok = !((edge_top && (r + dr) < 0) || (edge_bottom && (r + dr) >= rows));
        const uint8_t v  = ok ? 0xFF : 0x00;
        b[k * 4 + 0] = b[k * 4 + 1] = b[k * 4 + 2] = b[k * 4 + 3] = v;
    }
    return vld1q_u8(b);
}

// Row availability for the 2-row tail pack [r0 r1 r0 r1] (only the low 8 lanes are consumed).
static inline uint8x16_t bnd_row16_2(int i, int sub, int dr, int edge_top, int edge_bottom, int rows) {
    const int     ok0 = !((edge_top && (i + dr) < 0) || (edge_bottom && (i + dr) >= rows));
    const int     ok1 = !((edge_top && (i + sub + dr) < 0) || (edge_bottom && (i + sub + dr) >= rows));
    const uint8_t v0  = ok0 ? 0xFF : 0x00;
    const uint8_t v1  = ok1 ? 0xFF : 0x00;
    uint8_t       b[16];
    b[0] = b[1] = b[2] = b[3] = v0;
    b[4] = b[5] = b[6] = b[7] = v1;
    b[8] = b[9] = b[10] = b[11] = v0;
    b[12] = b[13] = b[14] = b[15] = v1;
    return vld1q_u8(b);
}

static void cdef_filter_block_4xn_8_bounded_neon(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                                 int32_t sec_strength, int32_t dir, int32_t damping,
                                                 int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor,
                                                 int edge_top, int edge_left, int edge_bottom, int edge_right) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const int     rows        = height;

    const int  off_b[6] = {svt_aom_eb_cdef_directions[dir][0],
                           svt_aom_eb_cdef_directions[dir][1],
                           svt_aom_eb_cdef_directions[dir + 2][0],
                           svt_aom_eb_cdef_directions[dir + 2][1],
                           svt_aom_eb_cdef_directions[dir - 2][0],
                           svt_aom_eb_cdef_directions[dir - 2][1]};
    const int  dr_b[6]  = {svt_aom_eb_cdef_directions_rc[dir][0][0],
                           svt_aom_eb_cdef_directions_rc[dir][1][0],
                           svt_aom_eb_cdef_directions_rc[dir + 2][0][0],
                           svt_aom_eb_cdef_directions_rc[dir + 2][1][0],
                           svt_aom_eb_cdef_directions_rc[dir - 2][0][0],
                           svt_aom_eb_cdef_directions_rc[dir - 2][1][0]};
    const int  dc_b[6]  = {svt_aom_eb_cdef_directions_rc[dir][0][1],
                           svt_aom_eb_cdef_directions_rc[dir][1][1],
                           svt_aom_eb_cdef_directions_rc[dir + 2][0][1],
                           svt_aom_eb_cdef_directions_rc[dir + 2][1][1],
                           svt_aom_eb_cdef_directions_rc[dir - 2][0][1],
                           svt_aom_eb_cdef_directions_rc[dir - 2][1][1]};
    uint8x16_t col16p[6], col16n[6];
    int        rsensp[6], rsensn[6], needp[6], needn[6];
    for (int b = 0; b < 6; b++) {
        col16p[b]     = bnd_col16_4(dc_b[b], edge_left, edge_right);
        col16n[b]     = bnd_col16_4(-dc_b[b], edge_left, edge_right);
        const int csp = (edge_left && dc_b[b] < 0) || (edge_right && dc_b[b] > 0);
        const int csn = (edge_left && -dc_b[b] < 0) || (edge_right && -dc_b[b] > 0);
        rsensp[b]     = (edge_top && dr_b[b] < 0) || (edge_bottom && dr_b[b] > 0);
        rsensn[b]     = (edge_top && -dr_b[b] < 0) || (edge_bottom && -dr_b[b] > 0);
        needp[b]      = csp || rsensp[b];
        needn[b]      = csn || rsensn[b];
    }

    const uint8x16_t prithr8  = vdupq_n_u8((uint8_t)pri_strength);
    const uint8x16_t secthr8  = vdupq_n_u8((uint8_t)sec_strength);
    const int8x16_t  pridamp8 = vdupq_n_s8((int8_t)-pri_damping);
    const int8x16_t  secdamp8 = vdupq_n_s8((int8_t)-sec_damping);
    const int8x8_t   pri_t0   = vdup_n_s8((int8_t)pri_taps[0]);
    const int8x8_t   pri_t1   = vdup_n_s8((int8_t)pri_taps[1]);
    const int8x8_t   sec_t0   = vdup_n_s8((int8_t)sec_taps[0]);
    const int8x8_t   sec_t1   = vdup_n_s8((int8_t)sec_taps[1]);

    const int sub = subsampling_factor;
    uint8_t   i   = 0;
    for (; i + 4 * sub <= height; i += (4 * sub)) {
        const uint8_t* r0     = in + (i + 0 * sub) * CDEF_BSTRIDE;
        const uint8_t* r1     = in + (i + 1 * sub) * CDEF_BSTRIDE;
        const uint8_t* r2     = in + (i + 2 * sub) * CDEF_BSTRIDE;
        const uint8_t* r3     = in + (i + 3 * sub) * CDEF_BSTRIDE;
        uint32x4_t     rv     = vdupq_n_u32(0);
        rv                    = vld1q_lane_u32((const uint32_t*)r0, rv, 0);
        rv                    = vld1q_lane_u32((const uint32_t*)r1, rv, 1);
        rv                    = vld1q_lane_u32((const uint32_t*)r2, rv, 2);
        rv                    = vld1q_lane_u32((const uint32_t*)r3, rv, 3);
        const uint8x16_t row8 = vreinterpretq_u8_u32(rv);
        uint8x16_t       min8 = row8, max8 = row8, tap8, av;
        int16x8_t        suma = vdupq_n_s16(0), sumb = vdupq_n_s16(0);
        int8x16_t        c0, c1, c2, c3, csum;
#define NAT4_TAP(OFF)                                                         \
    ({                                                                        \
        uint32x4_t _t = vdupq_n_u32(0);                                       \
        _t            = vld1q_lane_u32((const uint32_t*)(r0 + (OFF)), _t, 0); \
        _t            = vld1q_lane_u32((const uint32_t*)(r1 + (OFF)), _t, 1); \
        _t            = vld1q_lane_u32((const uint32_t*)(r2 + (OFF)), _t, 2); \
        _t            = vld1q_lane_u32((const uint32_t*)(r3 + (OFF)), _t, 3); \
        vreinterpretq_u8_u32(_t);                                             \
    })
#define B4(B, SGN, THR, DAMP, CACC)                                                                    \
    do {                                                                                               \
        const int _need = (SGN) > 0 ? needp[B] : needn[B];                                             \
        const int _rs   = (SGN) > 0 ? rsensp[B] : rsensn[B];                                           \
        tap8            = NAT4_TAP((SGN) * off_b[B]);                                                  \
        if (!_need) {                                                                                  \
            max8   = vmaxq_u8(max8, tap8);                                                             \
            min8   = vminq_u8(min8, tap8);                                                             \
            (CACC) = constrain8x16(tap8, row8, (THR), (DAMP));                                         \
        } else {                                                                                       \
            av     = _rs ? vandq_u8((SGN) > 0 ? col16p[B] : col16n[B],                                 \
                                bnd_row16_4(i, sub, (SGN) * dr_b[B], edge_top, edge_bottom, rows)) \
                         : ((SGN) > 0 ? col16p[B] : col16n[B]);                                        \
            max8   = vmaxq_u8(max8, vandq_u8(tap8, av));                                               \
            min8   = vminq_u8(min8, vorrq_u8(tap8, vmvnq_u8(av)));                                     \
            (CACC) = vandq_s8(constrain8x16(tap8, row8, (THR), (DAMP)), vreinterpretq_s8_u8(av));      \
        }                                                                                              \
    } while (0)
        if (pri_strength) {
            B4(0, 1, prithr8, pridamp8, c0);
            B4(0, -1, prithr8, pridamp8, c1);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t0, vget_high_s8(csum));
            B4(1, 1, prithr8, pridamp8, c0);
            B4(1, -1, prithr8, pridamp8, c1);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, pri_t1, vget_high_s8(csum));
        }
        if (sec_strength) {
            B4(2, 1, secthr8, secdamp8, c0);
            B4(2, -1, secthr8, secdamp8, c1);
            B4(4, 1, secthr8, secdamp8, c2);
            B4(4, -1, secthr8, secdamp8, c3);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t0, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t0, vget_high_s8(csum));
            B4(3, 1, secthr8, secdamp8, c0);
            B4(3, -1, secthr8, secdamp8, c1);
            B4(5, 1, secthr8, secdamp8, c2);
            B4(5, -1, secthr8, secdamp8, c3);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t1, vget_low_s8(csum));
            sumb = vmlal_s8(sumb, sec_t1, vget_high_s8(csum));
        }
#undef B4
#undef NAT4_TAP
        const uint8x8_t ansAB = cdef_finalize8(suma, vget_low_u8(row8), vget_low_u8(min8), vget_low_u8(max8));
        const uint8x8_t ansCD = cdef_finalize8(sumb, vget_high_u8(row8), vget_high_u8(min8), vget_high_u8(max8));
        *(uint32_t*)(dst + (i + 0 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansAB), 0);
        *(uint32_t*)(dst + (i + 1 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansAB), 1);
        *(uint32_t*)(dst + (i + 2 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansCD), 0);
        *(uint32_t*)(dst + (i + 3 * sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ansCD), 1);
    }
    // 2-row tail.
    for (; i < height; i += (2 * sub)) {
        const uint8_t* ra     = in + i * CDEF_BSTRIDE;
        const uint8_t* rb     = in + (i + sub) * CDEF_BSTRIDE;
        uint32x2_t     rv     = vdup_n_u32(0);
        rv                    = vld1_lane_u32((const uint32_t*)ra, rv, 0);
        rv                    = vld1_lane_u32((const uint32_t*)rb, rv, 1);
        const uint8x8_t  rlo  = vreinterpret_u8_u32(rv);
        const uint8x16_t row8 = vcombine_u8(rlo, rlo);
        uint8x16_t       min8 = row8, max8 = row8, tap8, av;
        int16x8_t        suma = vdupq_n_s16(0);
        int8x16_t        c0, c1, c2, c3, csum;
#define NAT2_TAP(OFF)                                                             \
    ({                                                                            \
        uint32x2_t _t      = vdup_n_u32(0);                                       \
        _t                 = vld1_lane_u32((const uint32_t*)(ra + (OFF)), _t, 0); \
        _t                 = vld1_lane_u32((const uint32_t*)(rb + (OFF)), _t, 1); \
        const uint8x8_t _l = vreinterpret_u8_u32(_t);                             \
        vcombine_u8(_l, _l);                                                      \
    })
#define B2(B, SGN, THR, DAMP, CACC)                                                                    \
    do {                                                                                               \
        const int _need = (SGN) > 0 ? needp[B] : needn[B];                                             \
        const int _rs   = (SGN) > 0 ? rsensp[B] : rsensn[B];                                           \
        tap8            = NAT2_TAP((SGN) * off_b[B]);                                                  \
        if (!_need) {                                                                                  \
            max8   = vmaxq_u8(max8, tap8);                                                             \
            min8   = vminq_u8(min8, tap8);                                                             \
            (CACC) = constrain8x16(tap8, row8, (THR), (DAMP));                                         \
        } else {                                                                                       \
            av     = _rs ? vandq_u8((SGN) > 0 ? col16p[B] : col16n[B],                                 \
                                bnd_row16_2(i, sub, (SGN) * dr_b[B], edge_top, edge_bottom, rows)) \
                         : ((SGN) > 0 ? col16p[B] : col16n[B]);                                        \
            max8   = vmaxq_u8(max8, vandq_u8(tap8, av));                                               \
            min8   = vminq_u8(min8, vorrq_u8(tap8, vmvnq_u8(av)));                                     \
            (CACC) = vandq_s8(constrain8x16(tap8, row8, (THR), (DAMP)), vreinterpretq_s8_u8(av));      \
        }                                                                                              \
    } while (0)
        if (pri_strength) {
            B2(0, 1, prithr8, pridamp8, c0);
            B2(0, -1, prithr8, pridamp8, c1);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t0, vget_low_s8(csum));
            B2(1, 1, prithr8, pridamp8, c0);
            B2(1, -1, prithr8, pridamp8, c1);
            csum = vaddq_s8(c0, c1);
            suma = vmlal_s8(suma, pri_t1, vget_low_s8(csum));
        }
        if (sec_strength) {
            B2(2, 1, secthr8, secdamp8, c0);
            B2(2, -1, secthr8, secdamp8, c1);
            B2(4, 1, secthr8, secdamp8, c2);
            B2(4, -1, secthr8, secdamp8, c3);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t0, vget_low_s8(csum));
            B2(3, 1, secthr8, secdamp8, c0);
            B2(3, -1, secthr8, secdamp8, c1);
            B2(5, 1, secthr8, secdamp8, c2);
            B2(5, -1, secthr8, secdamp8, c3);
            csum = vaddq_s8(vaddq_s8(c0, c1), vaddq_s8(c2, c3));
            suma = vmlal_s8(suma, sec_t1, vget_low_s8(csum));
        }
#undef B2
#undef NAT2_TAP
        const uint8x8_t ans             = cdef_finalize8(suma, vget_low_u8(row8), vget_low_u8(min8), vget_low_u8(max8));
        *(uint32_t*)(dst + i * dstride) = vget_lane_u32(vreinterpret_u32_u8(ans), 0);
        *(uint32_t*)(dst + (i + sub) * dstride) = vget_lane_u32(vreinterpret_u32_u8(ans), 1);
    }
}

void svt_cdef_filter_block_8bit_bounded_neon(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                             int32_t sec_strength, int32_t dir, int32_t damping, int32_t bsize,
                                             int32_t coeff_shift, uint8_t subsampling_factor, int edge_top,
                                             int edge_left, int edge_bottom, int edge_right) {
    if (bsize == BLOCK_8X8) {
        cdef_filter_block_8xn_8_bounded_neon(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             8,
                                             subsampling_factor,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    } else if (bsize == BLOCK_8X4) {
        cdef_filter_block_8xn_8_bounded_neon(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             4,
                                             subsampling_factor,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    } else if (bsize == BLOCK_4X8) {
        cdef_filter_block_4xn_8_bounded_neon(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             8,
                                             subsampling_factor,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    } else {
        // BLOCK_4X4
        cdef_filter_block_4xn_8_bounded_neon(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             4,
                                             1,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    }
}
