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

#include "warp_plane_neon_i8mm.h"

// clang-format off
DECLARE_ALIGNED(16, const uint8_t, svt_kTblIdx0_3[16]) = {
    0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1,
};
// clang-format on

static AOM_FORCE_INLINE void vertical_filter_4x1_f4(const int16x8_t* src, int32x4_t* res, int sy, int gamma) {
    int16x8_t s0, s1, s2, s3;
    transpose_elems_s16_4x8(vget_low_s16(src[0]),
                            vget_low_s16(src[1]),
                            vget_low_s16(src[2]),
                            vget_low_s16(src[3]),
                            vget_low_s16(src[4]),
                            vget_low_s16(src[5]),
                            vget_low_s16(src[6]),
                            vget_low_s16(src[7]),
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

    int32x4_t m0123_pairs[] = {m0, m1, m2, m3};

    *res = horizontal_add_4d_s32x4(m0123_pairs);
}

// Filter taps of the eight output pixels of a row, in tap order: out[t / 2] holds tap t of every pixel in its
// low (even t) or high (odd t) half.
static AOM_FORCE_INLINE void load_filters_8_by_tap(int8x16_t out[4], int offset, int stride) {
    int8x16_t f[8];
    for (int n = 0; n < 8; ++n) {
        f[n] = vcombine_s8(vld1_s8(svt_aom_warped_filter8[(offset + n * stride) >> WARPEDDIFF_PREC_BITS]),
                           vdup_n_s8(0));
    }
    const int16x8_t p01 = vreinterpretq_s16_s8(vzip1q_s8(f[0], f[1]));
    const int16x8_t p23 = vreinterpretq_s16_s8(vzip1q_s8(f[2], f[3]));
    const int16x8_t p45 = vreinterpretq_s16_s8(vzip1q_s8(f[4], f[5]));
    const int16x8_t p67 = vreinterpretq_s16_s8(vzip1q_s8(f[6], f[7]));

    const int32x4_t q0 = vreinterpretq_s32_s16(vzip1q_s16(p01, p23));
    const int32x4_t q1 = vreinterpretq_s32_s16(vzip2q_s16(p01, p23));
    const int32x4_t q2 = vreinterpretq_s32_s16(vzip1q_s16(p45, p67));
    const int32x4_t q3 = vreinterpretq_s32_s16(vzip2q_s16(p45, p67));

    out[0] = vreinterpretq_s8_s32(vzip1q_s32(q0, q2));
    out[1] = vreinterpretq_s8_s32(vzip2q_s32(q0, q2));
    out[2] = vreinterpretq_s8_s32(vzip1q_s32(q1, q3));
    out[3] = vreinterpretq_s8_s32(vzip2q_s32(q1, q3));
}

static AOM_FORCE_INLINE void vertical_filter_8x1_f8(const int16x8_t* src, int32x4_t* res_low, int32x4_t* res_high,
                                                    int sy, int gamma) {
    int8x16_t f[4];
    load_filters_8_by_tap(f, sy, gamma);

    // Two accumulators per half keep the multiply-accumulate chains at four links.
    int16x8_t tap = vmovl_s8(vget_low_s8(f[0]));
    int32x4_t lo0 = vmull_s16(vget_low_s16(src[0]), vget_low_s16(tap));
    int32x4_t hi0 = vmull_high_s16(src[0], tap);
    tap           = vmovl_high_s8(f[0]);
    int32x4_t lo1 = vmull_s16(vget_low_s16(src[1]), vget_low_s16(tap));
    int32x4_t hi1 = vmull_high_s16(src[1], tap);
    for (int t = 1; t < 4; ++t) {
        tap = vmovl_s8(vget_low_s8(f[t]));
        lo0 = vmlal_s16(lo0, vget_low_s16(src[2 * t]), vget_low_s16(tap));
        hi0 = vmlal_high_s16(hi0, src[2 * t], tap);
        tap = vmovl_high_s8(f[t]);
        lo1 = vmlal_s16(lo1, vget_low_s16(src[2 * t + 1]), vget_low_s16(tap));
        hi1 = vmlal_high_s16(hi1, src[2 * t + 1], tap);
    }
    *res_low  = vaddq_s32(lo0, lo1);
    *res_high = vaddq_s32(hi0, hi1);
}

void svt_av1_warp_affine_neon_i8mm(const int32_t* mat, const uint8_t* ref, int width, int height, int stride,
                                   uint8_t* pred, int p_col, int p_row, int p_width, int p_height, int p_stride,
                                   int subsampling_x, int subsampling_y, ConvolveParams* conv_params, int16_t alpha,
                                   int16_t beta, int16_t gamma, int16_t delta) {
    const int       w0                    = conv_params->fwd_offset;
    const int       w1                    = conv_params->bck_offset;
    const int       is_compound           = conv_params->is_compound;
    uint16_t* const dst                   = conv_params->dst;
    const int       dst_stride            = conv_params->dst_stride;
    const int       do_average            = conv_params->do_average;
    const int       use_dist_wtd_comp_avg = conv_params->use_dist_wtd_comp_avg;

    assert(IMPLIES(is_compound, dst != NULL));
    assert(IMPLIES(do_average, is_compound));

    for (int i = 0; i < p_height; i += 8) {
        for (int j = 0; j < p_width; j += 8) {
            const int32_t src_x = (p_col + j + 4) << subsampling_x;
            const int32_t src_y = (p_row + i + 4) << subsampling_y;
            const int64_t dst_x = (int64_t)mat[2] * src_x + (int64_t)mat[3] * src_y + (int64_t)mat[0];
            const int64_t dst_y = (int64_t)mat[4] * src_x + (int64_t)mat[5] * src_y + (int64_t)mat[1];

            const int64_t x4 = dst_x >> subsampling_x;
            const int64_t y4 = dst_y >> subsampling_y;

            int16x8_t tmp[15];
            warp_affine_horizontal_neon_i8mm(
                ref, width, height, stride, p_width, p_height, alpha, beta, x4, y4, i, tmp);
            warp_affine_vertical(pred,
                                 p_width,
                                 p_height,
                                 p_stride,
                                 is_compound,
                                 dst,
                                 dst_stride,
                                 do_average,
                                 use_dist_wtd_comp_avg,
                                 gamma,
                                 delta,
                                 y4,
                                 i,
                                 j,
                                 tmp,
                                 w0,
                                 w1);
        }
    }
}
