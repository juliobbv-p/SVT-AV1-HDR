/*
 * Copyright (c) 2020, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */
#include <immintrin.h>
#include "common_dsp_rtcd.h"
#include "warped_motion.h"

// Four tap pairs in even-output / odd-output order. One vpmaddwd
// computes a pair for every output pixel of an eight-pixel row.
static INLINE void warp_weights(int phase, int step, __m256i w[4]) {
    const int16_t* t = svt_aom_warped_filter[phase >> WARPEDDIFF_PREC_BITS];
    if (!step) {
        for (int i = 0; i < 4; ++i) {
            w[i] = _mm256_broadcastd_epi32(_mm_loadu_si32(t + 2 * i));
        }
        return;
    }
    __m256i r[4];
    for (int i = 0; i < 4; ++i) {
        const int     p = phase + 2 * i * step;
        const __m128i a = _mm_loadu_si128((const __m128i*)svt_aom_warped_filter[p >> WARPEDDIFF_PREC_BITS]);
        const __m128i b = _mm_loadu_si128((const __m128i*)svt_aom_warped_filter[(p + step) >> WARPEDDIFF_PREC_BITS]);
        r[i]            = _mm256_set_m128i(b, a);
    }
    const __m256i a = _mm256_unpacklo_epi32(r[0], r[1]), b = _mm256_unpacklo_epi32(r[2], r[3]);
    const __m256i c = _mm256_unpackhi_epi32(r[0], r[1]), d = _mm256_unpackhi_epi32(r[2], r[3]);
    w[0] = _mm256_unpacklo_epi64(a, b);
    w[1] = _mm256_unpackhi_epi64(a, b);
    w[2] = _mm256_unpacklo_epi64(c, d);
    w[3] = _mm256_unpackhi_epi64(c, d);
}

static INLINE __m256i warp_horizontal(__m128i msb, __m128i lsb, const __m256i w[4], __m256i bias, __m128i shift) {
    const __m256i hi = _mm256_set_m128i(_mm_srli_si128(msb, 1), msb);
    const __m256i lo = _mm256_set_m128i(_mm_srli_si128(lsb, 1), lsb);
    const __m256i a  = _mm256_srli_epi16(_mm256_unpacklo_epi8(lo, hi), 6);
    const __m256i b  = _mm256_srli_epi16(_mm256_unpackhi_epi8(lo, hi), 6);
    const __m256i ab = _mm256_add_epi32(_mm256_madd_epi16(a, w[0]),
                                        _mm256_madd_epi16(_mm256_alignr_epi8(b, a, 4), w[1]));
    const __m256i cd = _mm256_add_epi32(_mm256_madd_epi16(_mm256_alignr_epi8(b, a, 8), w[2]),
                                        _mm256_madd_epi16(_mm256_alignr_epi8(b, a, 12), w[3]));
    return _mm256_sra_epi32(_mm256_add_epi32(_mm256_add_epi32(ab, cd), bias), shift);
}

// The low and high 128-bit halves hold even and odd columns, respectively.
// Interleave after narrowing for non-averaged outputs.
static INLINE __m128i warp_pack_pixels(__m256i v) {
    const __m256i packed = _mm256_packus_epi32(v, v);
    return _mm_unpacklo_epi16(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1));
}

void svt_av1_highbd_warp_affine_avx2(const int32_t* mat, const uint8_t* ref8b, const uint8_t* ref2b, int width,
                                     int height, int stride8b, int stride2b, uint16_t* pred, int p_col, int p_row,
                                     int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y,
                                     int bd, ConvolveParams* conv_params, int16_t alpha, int16_t beta, int16_t gamma,
                                     int16_t delta) {
    if (width < 16 || (p_width & 3)) {
        svt_av1_highbd_warp_affine_c(mat,
                                     ref8b,
                                     ref2b,
                                     width,
                                     height,
                                     stride8b,
                                     stride2b,
                                     pred,
                                     p_col,
                                     p_row,
                                     p_width,
                                     p_height,
                                     p_stride,
                                     subsampling_x,
                                     subsampling_y,
                                     bd,
                                     conv_params,
                                     alpha,
                                     beta,
                                     gamma,
                                     delta);
        return;
    }
    const int reduce_bits_horiz = conv_params->round_0 + AOMMAX(bd + FILTER_BITS - conv_params->round_0 - 14, 0);
    const int reduce_bits_vert  = conv_params->is_compound ? conv_params->round_1 : 2 * FILTER_BITS - reduce_bits_horiz;
    const int offset_bits_horiz = bd + FILTER_BITS - 1;
    const int offset_bits_vert  = bd + 2 * FILTER_BITS - reduce_bits_horiz;
    const int round_bits        = 2 * FILTER_BITS - conv_params->round_0 - conv_params->round_1;
    const int offset_bits       = bd + 2 * FILTER_BITS - conv_params->round_0;
    assert(IMPLIES(conv_params->is_compound, conv_params->dst != NULL));

    const __m256i clp_pxl                = _mm256_set1_epi16(bd == 10 ? 1023 : (bd == 12 ? 4095 : 255));
    const __m128i reduce_bits_vert_shift = _mm_cvtsi32_si128(reduce_bits_vert);
    const __m256i reduce_bits_vert_const = _mm256_set1_epi32(((1 << reduce_bits_vert) >> 1));
    const __m256i res_add_const          = _mm256_set1_epi32(1 << offset_bits_vert);
    const __m256i res_sub_const          = _mm256_set1_epi32(-(1 << (offset_bits - conv_params->round_1)) -
                                                             (1 << (offset_bits - conv_params->round_1 - 1)));
    __m128i       round_bits_shift       = _mm_cvtsi32_si128(round_bits);
    __m256i       round_bits_const       = _mm256_set1_epi32(((1 << round_bits) >> 1));

    const int     w0  = conv_params->fwd_offset;
    const int     w1  = conv_params->bck_offset;
    const __m256i wt0 = _mm256_set1_epi32(w0);
    const __m256i wt1 = _mm256_set1_epi32(w1);

    const __m256i horiz_bias  = _mm256_set1_epi32((1 << offset_bits_horiz) + (1 << (reduce_bits_horiz - 1)));
    const __m128i horiz_shift = _mm_cvtsi32_si128(reduce_bits_horiz);

    for (int i = 0; i < p_height; i += 8) {
        for (int j = 0; j < p_width; j += 8) {
            // Calculate the center of this 8x8 block,
            // project to luma coordinates (if in a subsampled chroma plane),
            // apply the affine transformation,
            // then convert back to the original coordinates (if necessary)
            const int32_t src_x = (p_col + j + 4) << subsampling_x;
            const int32_t src_y = (p_row + i + 4) << subsampling_y;
            const int32_t dst_x = mat[2] * src_x + mat[3] * src_y + mat[0];
            const int32_t dst_y = mat[4] * src_x + mat[5] * src_y + mat[1];
            const int32_t x4    = dst_x >> subsampling_x;
            const int32_t y4    = dst_y >> subsampling_y;

            const int32_t ix4 = x4 >> WARPEDMODEL_PREC_BITS;
            int32_t       sx4 = x4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);
            const int32_t iy4 = y4 >> WARPEDMODEL_PREC_BITS;
            int32_t       sy4 = y4 & ((1 << WARPEDMODEL_PREC_BITS) - 1);

            sx4 += alpha * (-4) + beta * (-4) + (1 << (WARPEDDIFF_PREC_BITS - 1)) +
                (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);
            sy4 += gamma * (-4) + delta * (-4) + (1 << (WARPEDDIFF_PREC_BITS - 1)) +
                (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);

            sx4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);
            sy4 &= ~((1 << WARP_PARAM_REDUCE_BITS) - 1);

            const bool    half         = p_width - j == 4;
            const int     nrows        = AOMMIN(8, p_height - i);
            const int     x            = clamp(ix4 - 7, 0, width - 16);
            const int     displacement = clamp(ix4 - 7 - x, -15, 15);
            const __m128i indices      = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            const __m128i edge         = _mm_min_epi8(
                _mm_set1_epi8(15),
                _mm_max_epi8(_mm_setzero_si128(), _mm_add_epi8(indices, _mm_set1_epi8(displacement))));
            __m256i    weights[4];
            const bool flat = ix4 <= -7 || ix4 - 7 >= width - 1;
            if (!beta && !flat) {
                warp_weights(sx4, alpha, weights);
            }
            __m256i pairs[14], previous = _mm256_setzero_si256();
            for (int k = -7; k < nrows; ++k) {
                const int y = clamp(iy4 + k, 0, height - 1);
                __m256i   row;
                if (flat) {
                    // All eight taps and all eight output columns see the same edge pixel.
                    const int edge_x = ix4 <= -7 ? 0 : width - 1;
                    const int sample = (ref8b[y * stride8b + edge_x] << 2) | (ref2b[y * stride2b + edge_x] >> 6);
                    row              = _mm256_set1_epi32(
                        ((sample << FILTER_BITS) + (1 << offset_bits_horiz) + (1 << (reduce_bits_horiz - 1))) >>
                        reduce_bits_horiz);
                } else {
                    __m128i msb = _mm_loadu_si128((const __m128i*)(ref8b + y * stride8b + x));
                    __m128i lsb = _mm_loadu_si128((const __m128i*)(ref2b + y * stride2b + x));
                    if (displacement) {
                        msb = _mm_shuffle_epi8(msb, edge);
                        lsb = _mm_shuffle_epi8(lsb, edge);
                    }
                    if (beta) {
                        warp_weights(sx4 + beta * (k + 4), alpha, weights);
                    }
                    row = warp_horizontal(msb, lsb, weights, horiz_bias, horiz_shift);
                }
                if (k > -7) {
                    pairs[k + 6] = _mm256_blend_epi16(previous, _mm256_slli_epi32(row, 16), 0xaa);
                }
                previous = row;
            }
            if (!delta) {
                warp_weights(sy4, gamma, weights);
            }
            for (int k = -4; k < nrows - 4; ++k) {
                if (delta) {
                    warp_weights(sy4 + delta * (k + 4), gamma, weights);
                }
                const __m256i* p     = pairs + k + 4;
                const __m256i  a     = _mm256_add_epi32(_mm256_madd_epi16(p[0], weights[0]),
                                                   _mm256_madd_epi16(p[2], weights[1]));
                const __m256i  b     = _mm256_add_epi32(_mm256_madd_epi16(p[4], weights[2]),
                                                   _mm256_madd_epi16(p[6], weights[3]));
                __m256i        v_sum = _mm256_add_epi32(a, b);
                if (conv_params->is_compound) {
                    __m128i* const p = (__m128i*)&conv_params->dst[(i + k + 4) * conv_params->dst_stride + j];

                    v_sum = _mm256_add_epi32(v_sum, res_add_const);
                    v_sum = _mm256_sra_epi32(_mm256_add_epi32(v_sum, reduce_bits_vert_const), reduce_bits_vert_shift);
                    if (conv_params->do_average) {
                        v_sum = _mm256_permutevar8x32_epi32(v_sum, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                        __m128i* const dst16 = (__m128i*)&pred[(i + k + 4) * p_stride + j];
                        __m256i        p_32  = _mm256_cvtepu16_epi32(half ? _mm_loadl_epi64(p) : _mm_loadu_si128(p));

                        if (conv_params->use_jnt_comp_avg) {
                            v_sum = _mm256_add_epi32(_mm256_mullo_epi32(p_32, wt0), _mm256_mullo_epi32(v_sum, wt1));
                            v_sum = _mm256_srai_epi32(v_sum, DIST_PRECISION_BITS);
                        } else {
                            v_sum = _mm256_srai_epi32(_mm256_add_epi32(p_32, v_sum), 1);
                        }

                        __m256i v_sum1 = _mm256_add_epi32(v_sum, res_sub_const);
                        v_sum1         = _mm256_sra_epi32(_mm256_add_epi32(v_sum1, round_bits_const), round_bits_shift);

                        __m256i v_sum16 = _mm256_packus_epi32(v_sum1, v_sum1);
                        v_sum16         = _mm256_permute4x64_epi64(v_sum16, 0xD8);
                        v_sum16         = _mm256_min_epi16(v_sum16, clp_pxl);
                        if (half) {
                            _mm_storel_epi64(dst16, _mm256_castsi256_si128(v_sum16));
                        } else {
                            _mm_storeu_si128(dst16, _mm256_castsi256_si128(v_sum16));
                        }
                    } else {
                        const __m128i pixels = warp_pack_pixels(v_sum);
                        if (half) {
                            _mm_storel_epi64(p, pixels);
                        } else {
                            _mm_storeu_si128(p, pixels);
                        }
                    }
                } else {
                    const __m256i round_const = _mm256_set1_epi32(-(1 << (bd + reduce_bits_vert - 1)) +
                                                                  ((1 << reduce_bits_vert) >> 1));
                    const __m256i rounded     = _mm256_sra_epi32(_mm256_add_epi32(v_sum, round_const),
                                                             reduce_bits_vert_shift);
                    const __m128i pixels = _mm_min_epu16(warp_pack_pixels(rounded), _mm256_castsi256_si128(clp_pxl));
                    __m128i*      p      = (__m128i*)&pred[(i + k + 4) * p_stride + j];
                    if (half) {
                        _mm_storel_epi64(p, pixels);
                    } else {
                        _mm_storeu_si128(p, pixels);
                    }
                }
            }
        }
    }
}
