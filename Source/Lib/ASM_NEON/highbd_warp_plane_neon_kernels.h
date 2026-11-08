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

#ifndef HIGHBD_WARP_PLANE_NEON_KERNELS_H_
#define HIGHBD_WARP_PLANE_NEON_KERNELS_H_

static AOM_FORCE_INLINE int16x8_t highbd_horizontal_filter_4x1_f4(int16x8_t rv0, int16x8_t rv1, int16x8_t rv2,
                                                                  int16x8_t rv3, int bd, int sx, int alpha) {
    int16x8_t f[4];
    load_filters_4(f, sx, alpha);

    int32x4_t m0 = vmull_s16(vget_low_s16(f[0]), vget_low_s16(rv0));
    m0           = vmlal_s16(m0, vget_high_s16(f[0]), vget_high_s16(rv0));
    int32x4_t m1 = vmull_s16(vget_low_s16(f[1]), vget_low_s16(rv1));
    m1           = vmlal_s16(m1, vget_high_s16(f[1]), vget_high_s16(rv1));
    int32x4_t m2 = vmull_s16(vget_low_s16(f[2]), vget_low_s16(rv2));
    m2           = vmlal_s16(m2, vget_high_s16(f[2]), vget_high_s16(rv2));
    int32x4_t m3 = vmull_s16(vget_low_s16(f[3]), vget_low_s16(rv3));
    m3           = vmlal_s16(m3, vget_high_s16(f[3]), vget_high_s16(rv3));

    int32x4_t m0123[] = {m0, m1, m2, m3};

    const int offset_const = (1 << (bd + FILTER_BITS - 1)) + (1 << (ROUND0_BITS - 1));

    int32x4_t res = horizontal_add_4d_s32x4(m0123);
    res           = vaddq_s32(res, vdupq_n_s32(offset_const));
    return vcombine_s16(vshrn_n_s32(res, ROUND0_BITS), vdup_n_s16(0));
}

static AOM_FORCE_INLINE int16x8_t highbd_horizontal_filter_8x1_f8(int16x8_t rv0, int16x8_t rv1, int16x8_t rv2,
                                                                  int16x8_t rv3, int16x8_t rv4, int16x8_t rv5,
                                                                  int16x8_t rv6, int16x8_t rv7, int bd, int sx,
                                                                  int alpha) {
    int16x8_t f[8];
    load_filters_8(f, sx, alpha);

    int32x4_t m0 = vmull_s16(vget_low_s16(f[0]), vget_low_s16(rv0));
    m0           = vmlal_s16(m0, vget_high_s16(f[0]), vget_high_s16(rv0));
    int32x4_t m1 = vmull_s16(vget_low_s16(f[1]), vget_low_s16(rv1));
    m1           = vmlal_s16(m1, vget_high_s16(f[1]), vget_high_s16(rv1));
    int32x4_t m2 = vmull_s16(vget_low_s16(f[2]), vget_low_s16(rv2));
    m2           = vmlal_s16(m2, vget_high_s16(f[2]), vget_high_s16(rv2));
    int32x4_t m3 = vmull_s16(vget_low_s16(f[3]), vget_low_s16(rv3));
    m3           = vmlal_s16(m3, vget_high_s16(f[3]), vget_high_s16(rv3));
    int32x4_t m4 = vmull_s16(vget_low_s16(f[4]), vget_low_s16(rv4));
    m4           = vmlal_s16(m4, vget_high_s16(f[4]), vget_high_s16(rv4));
    int32x4_t m5 = vmull_s16(vget_low_s16(f[5]), vget_low_s16(rv5));
    m5           = vmlal_s16(m5, vget_high_s16(f[5]), vget_high_s16(rv5));
    int32x4_t m6 = vmull_s16(vget_low_s16(f[6]), vget_low_s16(rv6));
    m6           = vmlal_s16(m6, vget_high_s16(f[6]), vget_high_s16(rv6));
    int32x4_t m7 = vmull_s16(vget_low_s16(f[7]), vget_low_s16(rv7));
    m7           = vmlal_s16(m7, vget_high_s16(f[7]), vget_high_s16(rv7));

    int32x4_t m0123[] = {m0, m1, m2, m3};
    int32x4_t m4567[] = {m4, m5, m6, m7};

    const int offset_const = (1 << (bd + FILTER_BITS - 1)) + (1 << (ROUND0_BITS - 1));

    int32x4_t res0 = horizontal_add_4d_s32x4(m0123);
    int32x4_t res1 = horizontal_add_4d_s32x4(m4567);
    res0           = vaddq_s32(res0, vdupq_n_s32(offset_const));
    res1           = vaddq_s32(res1, vdupq_n_s32(offset_const));
    return vcombine_s16(vshrn_n_s32(res0, ROUND0_BITS), vshrn_n_s32(res1, ROUND0_BITS));
}

static AOM_FORCE_INLINE int16x8_t highbd_horizontal_filter_4x1_f1(int16x8_t rv0, int16x8_t rv1, int16x8_t rv2,
                                                                  int16x8_t rv3, int bd, int sx) {
    int16x8_t f = load_filters_1(sx);

    int32x4_t m0 = vmull_s16(vget_low_s16(f), vget_low_s16(rv0));
    m0           = vmlal_s16(m0, vget_high_s16(f), vget_high_s16(rv0));
    int32x4_t m1 = vmull_s16(vget_low_s16(f), vget_low_s16(rv1));
    m1           = vmlal_s16(m1, vget_high_s16(f), vget_high_s16(rv1));
    int32x4_t m2 = vmull_s16(vget_low_s16(f), vget_low_s16(rv2));
    m2           = vmlal_s16(m2, vget_high_s16(f), vget_high_s16(rv2));
    int32x4_t m3 = vmull_s16(vget_low_s16(f), vget_low_s16(rv3));
    m3           = vmlal_s16(m3, vget_high_s16(f), vget_high_s16(rv3));

    int32x4_t m0123[] = {m0, m1, m2, m3};

    const int offset_const = (1 << (bd + FILTER_BITS - 1)) + (1 << (ROUND0_BITS - 1));

    int32x4_t res = horizontal_add_4d_s32x4(m0123);
    res           = vaddq_s32(res, vdupq_n_s32(offset_const));
    return vcombine_s16(vshrn_n_s32(res, ROUND0_BITS), vdup_n_s16(0));
}

static AOM_FORCE_INLINE int16x8_t highbd_horizontal_filter_8x1_f1(int16x8_t rv0, int16x8_t rv1, int16x8_t rv2,
                                                                  int16x8_t rv3, int16x8_t rv4, int16x8_t rv5,
                                                                  int16x8_t rv6, int16x8_t rv7, int bd, int sx) {
    int16x8_t f = load_filters_1(sx);

    int32x4_t m0 = vmull_s16(vget_low_s16(f), vget_low_s16(rv0));
    m0           = vmlal_s16(m0, vget_high_s16(f), vget_high_s16(rv0));
    int32x4_t m1 = vmull_s16(vget_low_s16(f), vget_low_s16(rv1));
    m1           = vmlal_s16(m1, vget_high_s16(f), vget_high_s16(rv1));
    int32x4_t m2 = vmull_s16(vget_low_s16(f), vget_low_s16(rv2));
    m2           = vmlal_s16(m2, vget_high_s16(f), vget_high_s16(rv2));
    int32x4_t m3 = vmull_s16(vget_low_s16(f), vget_low_s16(rv3));
    m3           = vmlal_s16(m3, vget_high_s16(f), vget_high_s16(rv3));
    int32x4_t m4 = vmull_s16(vget_low_s16(f), vget_low_s16(rv4));
    m4           = vmlal_s16(m4, vget_high_s16(f), vget_high_s16(rv4));
    int32x4_t m5 = vmull_s16(vget_low_s16(f), vget_low_s16(rv5));
    m5           = vmlal_s16(m5, vget_high_s16(f), vget_high_s16(rv5));
    int32x4_t m6 = vmull_s16(vget_low_s16(f), vget_low_s16(rv6));
    m6           = vmlal_s16(m6, vget_high_s16(f), vget_high_s16(rv6));
    int32x4_t m7 = vmull_s16(vget_low_s16(f), vget_low_s16(rv7));
    m7           = vmlal_s16(m7, vget_high_s16(f), vget_high_s16(rv7));

    int32x4_t m0123[] = {m0, m1, m2, m3};
    int32x4_t m4567[] = {m4, m5, m6, m7};

    const int offset_const = (1 << (bd + FILTER_BITS - 1)) + (1 << (ROUND0_BITS - 1));

    int32x4_t res0 = horizontal_add_4d_s32x4(m0123);
    int32x4_t res1 = horizontal_add_4d_s32x4(m4567);
    res0           = vaddq_s32(res0, vdupq_n_s32(offset_const));
    res1           = vaddq_s32(res1, vdupq_n_s32(offset_const));
    return vcombine_s16(vshrn_n_s32(res0, ROUND0_BITS), vshrn_n_s32(res1, ROUND0_BITS));
}

static AOM_FORCE_INLINE int32x4_t vertical_filter_4x1_f1(const int16x8_t* tmp, int sy) {
    const int16x8_t f     = load_filters_1(sy);
    const int16x4_t f0123 = vget_low_s16(f);
    const int16x4_t f4567 = vget_high_s16(f);

    int32x4_t m0123 = vmull_lane_s16(vget_low_s16(tmp[0]), f0123, 0);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[1]), f0123, 1);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[2]), f0123, 2);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[3]), f0123, 3);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[4]), f4567, 0);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[5]), f4567, 1);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[6]), f4567, 2);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[7]), f4567, 3);
    return m0123;
}

static AOM_FORCE_INLINE int32x4x2_t vertical_filter_8x1_f1(const int16x8_t* tmp, int sy) {
    const int16x8_t f     = load_filters_1(sy);
    const int16x4_t f0123 = vget_low_s16(f);
    const int16x4_t f4567 = vget_high_s16(f);

    int32x4_t m0123 = vmull_lane_s16(vget_low_s16(tmp[0]), f0123, 0);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[1]), f0123, 1);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[2]), f0123, 2);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[3]), f0123, 3);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[4]), f4567, 0);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[5]), f4567, 1);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[6]), f4567, 2);
    m0123           = vmlal_lane_s16(m0123, vget_low_s16(tmp[7]), f4567, 3);

    int32x4_t m4567 = vmull_lane_s16(vget_high_s16(tmp[0]), f0123, 0);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[1]), f0123, 1);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[2]), f0123, 2);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[3]), f0123, 3);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[4]), f4567, 0);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[5]), f4567, 1);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[6]), f4567, 2);
    m4567           = vmlal_lane_s16(m4567, vget_high_s16(tmp[7]), f4567, 3);
    return (int32x4x2_t){{m0123, m4567}};
}

static AOM_FORCE_INLINE int32x4_t vertical_filter_4x1_f4(const int16x8_t* tmp, int sy, int gamma) {
    int16x8_t s0, s1, s2, s3;
    transpose_elems_s16_4x8(vget_low_s16(tmp[0]),
                            vget_low_s16(tmp[1]),
                            vget_low_s16(tmp[2]),
                            vget_low_s16(tmp[3]),
                            vget_low_s16(tmp[4]),
                            vget_low_s16(tmp[5]),
                            vget_low_s16(tmp[6]),
                            vget_low_s16(tmp[7]),
                            &s0,
                            &s1,
                            &s2,
                            &s3);

    int16x8_t f[4];
    load_filters_4(f, sy, gamma);

    int32x4_t m0 = vmull_s16(vget_low_s16(s0), vget_low_s16(f[0]));
    m0           = vmlal_s16(m0, vget_high_s16(s0), vget_high_s16(f[0]));
    int32x4_t m1 = vmull_s16(vget_low_s16(s1), vget_low_s16(f[1]));
    m1           = vmlal_s16(m1, vget_high_s16(s1), vget_high_s16(f[1]));
    int32x4_t m2 = vmull_s16(vget_low_s16(s2), vget_low_s16(f[2]));
    m2           = vmlal_s16(m2, vget_high_s16(s2), vget_high_s16(f[2]));
    int32x4_t m3 = vmull_s16(vget_low_s16(s3), vget_low_s16(f[3]));
    m3           = vmlal_s16(m3, vget_high_s16(s3), vget_high_s16(f[3]));

    int32x4_t m0123[] = {m0, m1, m2, m3};
    return horizontal_add_4d_s32x4(m0123);
}

static AOM_FORCE_INLINE int32x4x2_t vertical_filter_8x1_columns(const int16x8_t* tmp, int sy, int gamma) {
    int16x8_t s0 = tmp[0];
    int16x8_t s1 = tmp[1];
    int16x8_t s2 = tmp[2];
    int16x8_t s3 = tmp[3];
    int16x8_t s4 = tmp[4];
    int16x8_t s5 = tmp[5];
    int16x8_t s6 = tmp[6];
    int16x8_t s7 = tmp[7];

    int16x8_t f[8];
    load_filters_8(f, sy, gamma);

    int32x4_t m0 = vmull_s16(vget_low_s16(s0), vget_low_s16(f[0]));
    m0           = vmlal_s16(m0, vget_high_s16(s0), vget_high_s16(f[0]));
    int32x4_t m1 = vmull_s16(vget_low_s16(s1), vget_low_s16(f[1]));
    m1           = vmlal_s16(m1, vget_high_s16(s1), vget_high_s16(f[1]));
    int32x4_t m2 = vmull_s16(vget_low_s16(s2), vget_low_s16(f[2]));
    m2           = vmlal_s16(m2, vget_high_s16(s2), vget_high_s16(f[2]));
    int32x4_t m3 = vmull_s16(vget_low_s16(s3), vget_low_s16(f[3]));
    m3           = vmlal_s16(m3, vget_high_s16(s3), vget_high_s16(f[3]));
    int32x4_t m4 = vmull_s16(vget_low_s16(s4), vget_low_s16(f[4]));
    m4           = vmlal_s16(m4, vget_high_s16(s4), vget_high_s16(f[4]));
    int32x4_t m5 = vmull_s16(vget_low_s16(s5), vget_low_s16(f[5]));
    m5           = vmlal_s16(m5, vget_high_s16(s5), vget_high_s16(f[5]));
    int32x4_t m6 = vmull_s16(vget_low_s16(s6), vget_low_s16(f[6]));
    m6           = vmlal_s16(m6, vget_high_s16(s6), vget_high_s16(f[6]));
    int32x4_t m7 = vmull_s16(vget_low_s16(s7), vget_low_s16(f[7]));
    m7           = vmlal_s16(m7, vget_high_s16(s7), vget_high_s16(f[7]));

    int32x4_t m0123[] = {m0, m1, m2, m3};
    int32x4_t m4567[] = {m4, m5, m6, m7};

    int32x4x2_t ret;
    ret.val[0] = horizontal_add_4d_s32x4(m0123);
    ret.val[1] = horizontal_add_4d_s32x4(m4567);
    return ret;
}

static AOM_FORCE_INLINE int32x4x2_t vertical_filter_8x1_f8(const int16x8_t* tmp, int sy, int gamma) {
    int16x8_t s0 = tmp[0];
    int16x8_t s1 = tmp[1];
    int16x8_t s2 = tmp[2];
    int16x8_t s3 = tmp[3];
    int16x8_t s4 = tmp[4];
    int16x8_t s5 = tmp[5];
    int16x8_t s6 = tmp[6];
    int16x8_t s7 = tmp[7];
    transpose_elems_inplace_s16_8x8(&s0, &s1, &s2, &s3, &s4, &s5, &s6, &s7);

    const int16x8_t columns[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    return vertical_filter_8x1_columns(columns, sy, gamma);
}

static AOM_FORCE_INLINE void warp_affine_vertical_reuse(uint16_t* pred, int p_width, int p_height, int p_stride, int bd,
                                                        uint16_t* dst, int dst_stride, bool is_compound,
                                                        bool do_average, bool use_dist_wtd_comp_avg, int fwd, int bwd,
                                                        int16_t gamma, int16_t delta, const int16x8_t* tmp, int i,
                                                        int sy4, int j) {
    if (bd != 10 || !gamma || p_width <= 4 || p_height <= 4) {
        warp_affine_vertical(pred,
                             p_width,
                             p_height,
                             p_stride,
                             bd,
                             dst,
                             dst_stride,
                             is_compound,
                             do_average,
                             use_dist_wtd_comp_avg,
                             fwd,
                             bwd,
                             gamma,
                             delta,
                             tmp,
                             i,
                             sy4,
                             j);
        return;
    }
    // The eight output rows share a 15-row source tile. Transpose it once,
    // then select each column's sliding eight-row window with TBL2.
    // The zero padding is outside every selected window (last index is 29).
    int16x8_t lo[8], hi[8];
    for (int c = 0; c < 8; ++c) {
        lo[c] = tmp[c];
    }
    for (int c = 0; c < 7; ++c) {
        hi[c] = tmp[c + 8];
    }
    hi[7] = vdupq_n_s16(0);
    transpose_elems_inplace_s16_8x8(&lo[0], &lo[1], &lo[2], &lo[3], &lo[4], &lo[5], &lo[6], &lo[7]);
    transpose_elems_inplace_s16_8x8(&hi[0], &hi[1], &hi[2], &hi[3], &hi[4], &hi[5], &hi[6], &hi[7]);
    const uint8_t    base_indices[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const uint8x16_t base             = vld1q_u8(base_indices);
    for (int k = 0; k < 8; ++k) {
        const uint8x16_t indices = vaddq_u8(base, vdupq_n_u8(2 * k));
        int16x8_t        window[8];
        window[0] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[0]), vreinterpretq_u8_s16(hi[0])}}, indices));
        window[1] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[1]), vreinterpretq_u8_s16(hi[1])}}, indices));
        window[2] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[2]), vreinterpretq_u8_s16(hi[2])}}, indices));
        window[3] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[3]), vreinterpretq_u8_s16(hi[3])}}, indices));
        window[4] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[4]), vreinterpretq_u8_s16(hi[4])}}, indices));
        window[5] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[5]), vreinterpretq_u8_s16(hi[5])}}, indices));
        window[6] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[6]), vreinterpretq_u8_s16(hi[6])}}, indices));
        window[7] = vreinterpretq_s16_u8(
            vqtbl2q_u8((uint8x16x2_t){{vreinterpretq_u8_s16(lo[7]), vreinterpretq_u8_s16(hi[7])}}, indices));
        const int32x4x2_t sums = vertical_filter_8x1_columns(window, sy4 + delta * k, gamma);
        highbd_vertical_output_8x1(pred,
                                   p_stride,
                                   bd,
                                   dst,
                                   dst_stride,
                                   is_compound,
                                   do_average,
                                   use_dist_wtd_comp_avg,
                                   fwd,
                                   bwd,
                                   sums,
                                   i + k,
                                   j);
    }
}

#endif // HIGHBD_WARP_PLANE_NEON_KERNELS_H_
