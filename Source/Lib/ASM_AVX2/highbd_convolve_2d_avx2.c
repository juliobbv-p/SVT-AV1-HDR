/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include <immintrin.h>
#include <assert.h>

#include "definitions.h"
#include "common_dsp_rtcd.h"

#include "convolve_avx2.h"
#include "synonyms.h"
#include "synonyms_avx2.h"
#include "convolve.h"

// Bilinear coefficients are multiples of eight. With round_0 = 3, the
// horizontal result is exactly (16 - phase) * a + phase * b, in 14 bits.
static INLINE __m256i bilinear_row_10bit(const uint16_t* src, int n, __m256i phase) {
    const __m256i a = n == 16 ? _mm256_loadu_si256((const __m256i*)src)
                              : _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)src));
    const __m256i b = n == 16 ? _mm256_loadu_si256((const __m256i*)(src + 1))
                              : _mm256_zextsi128_si256(_mm_loadu_si128((const __m128i*)(src + 1)));
    return _mm256_add_epi16(_mm256_slli_epi16(a, 4), _mm256_mullo_epi16(_mm256_sub_epi16(b, a), phase));
}

static INLINE void bilinear_columns_10bit(const uint16_t* src, int ss, uint16_t* dst, int ds, int n, int h, __m256i px,
                                          __m256i weights) {
    __m256i prev = bilinear_row_10bit(src, n, px);
    for (int y = 0; y < h; ++y) {
        src += ss;
        const __m256i next   = bilinear_row_10bit(src, n, px);
        const __m256i lo     = _mm256_madd_epi16(_mm256_unpacklo_epi16(prev, next), weights);
        const __m256i hi     = _mm256_madd_epi16(_mm256_unpackhi_epi16(prev, next), weights);
        const __m256i bias   = _mm256_set1_epi32(128);
        const __m256i result = _mm256_packus_epi32(_mm256_srli_epi32(_mm256_add_epi32(lo, bias), 8),
                                                   _mm256_srli_epi32(_mm256_add_epi32(hi, bias), 8));
        if (n == 16) {
            _mm256_storeu_si256((__m256i*)dst, result);
        } else {
            _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(result));
        }
        prev = next;
        dst += ds;
    }
}

void svt_av1_highbd_convolve_2d_sr_avx2(const uint16_t* src, int32_t src_stride, uint16_t* dst, int32_t dst_stride,
                                        int32_t w, int32_t h, const InterpFilterParams* filter_params_x,
                                        const InterpFilterParams* filter_params_y, const int32_t subpel_x_q4,
                                        const int32_t subpel_y_q4, ConvolveParams* conv_params, int32_t bd) {
    if (bd == 10 && !(w & 7) && conv_params->round_0 == 3 && conv_params->round_1 == 11 &&
        is_convolve_2tap(filter_params_x->filter_ptr) && is_convolve_2tap(filter_params_y->filter_ptr)) {
        const __m256i px      = _mm256_set1_epi16(subpel_x_q4 & SUBPEL_MASK);
        const int     py      = subpel_y_q4 & SUBPEL_MASK;
        const __m256i weights = _mm256_set1_epi32((16 - py) | (py << 16));
        int           x       = 0;
        for (; x + 16 <= w; x += 16) {
            bilinear_columns_10bit(src + x, src_stride, dst + x, dst_stride, 16, h, px, weights);
        }
        if (x < w) {
            bilinear_columns_10bit(src + x, src_stride, dst + x, dst_stride, 8, h, px, weights);
        }
        return;
    }
    if ((h & 1) || (w != 2 && w != 4 && (w & 7))) {
        svt_av1_highbd_convolve_2d_sr_c(src,
                                        src_stride,
                                        dst,
                                        dst_stride,
                                        w,
                                        h,
                                        filter_params_x,
                                        filter_params_y,
                                        subpel_x_q4,
                                        subpel_y_q4,
                                        conv_params,
                                        bd);
        return;
    }
    hbd_convolve_2d_block(src,
                          src_stride,
                          dst,
                          dst_stride,
                          w,
                          h,
                          filter_params_x,
                          filter_params_y,
                          subpel_x_q4,
                          subpel_y_q4,
                          conv_params,
                          bd,
                          0);
}

static INLINE void copy_64(const uint16_t* src, uint16_t* dst) {
    __m256i s[4];
    s[0] = _mm256_loadu_si256((__m256i*)(src + 0 * 16));
    s[1] = _mm256_loadu_si256((__m256i*)(src + 1 * 16));
    s[2] = _mm256_loadu_si256((__m256i*)(src + 2 * 16));
    s[3] = _mm256_loadu_si256((__m256i*)(src + 3 * 16));
    _mm256_storeu_si256((__m256i*)(dst + 0 * 16), s[0]);
    _mm256_storeu_si256((__m256i*)(dst + 1 * 16), s[1]);
    _mm256_storeu_si256((__m256i*)(dst + 2 * 16), s[2]);
    _mm256_storeu_si256((__m256i*)(dst + 3 * 16), s[3]);
}

static INLINE void copy_128(const uint16_t* src, uint16_t* dst) {
    __m256i s[8];
    s[0] = _mm256_loadu_si256((__m256i*)(src + 0 * 16));
    s[1] = _mm256_loadu_si256((__m256i*)(src + 1 * 16));
    s[2] = _mm256_loadu_si256((__m256i*)(src + 2 * 16));
    s[3] = _mm256_loadu_si256((__m256i*)(src + 3 * 16));
    s[4] = _mm256_loadu_si256((__m256i*)(src + 4 * 16));
    s[5] = _mm256_loadu_si256((__m256i*)(src + 5 * 16));
    s[6] = _mm256_loadu_si256((__m256i*)(src + 6 * 16));
    s[7] = _mm256_loadu_si256((__m256i*)(src + 7 * 16));

    _mm256_storeu_si256((__m256i*)(dst + 0 * 16), s[0]);
    _mm256_storeu_si256((__m256i*)(dst + 1 * 16), s[1]);
    _mm256_storeu_si256((__m256i*)(dst + 2 * 16), s[2]);
    _mm256_storeu_si256((__m256i*)(dst + 3 * 16), s[3]);
    _mm256_storeu_si256((__m256i*)(dst + 4 * 16), s[4]);
    _mm256_storeu_si256((__m256i*)(dst + 5 * 16), s[5]);
    _mm256_storeu_si256((__m256i*)(dst + 6 * 16), s[6]);
    _mm256_storeu_si256((__m256i*)(dst + 7 * 16), s[7]);
}

void svt_av1_highbd_convolve_2d_copy_sr_avx2(const uint16_t* src, int32_t src_stride, uint16_t* dst, int32_t dst_stride,
                                             int32_t w, int32_t h, const InterpFilterParams* filter_params_x,
                                             const InterpFilterParams* filter_params_y, const int32_t subpel_x_q4,
                                             const int32_t subpel_y_q4, ConvolveParams* conv_params, int32_t bd) {
    (void)filter_params_x;
    (void)filter_params_y;
    (void)subpel_x_q4;
    (void)subpel_y_q4;
    (void)conv_params;
    (void)bd;

    if (w == 2) {
        do {
            memcpy(dst, src, 2 * sizeof(*src));
            src += src_stride;
            dst += dst_stride;
            memcpy(dst, src, 2 * sizeof(*src));
            src += src_stride;
            dst += dst_stride;
            h -= 2;
        } while (h);
    } else if (w == 4) {
        do {
            __m128i s[2];
            s[0] = _mm_loadl_epi64((__m128i*)src);
            src += src_stride;
            s[1] = _mm_loadl_epi64((__m128i*)src);
            src += src_stride;
            _mm_storel_epi64((__m128i*)dst, s[0]);
            dst += dst_stride;
            _mm_storel_epi64((__m128i*)dst, s[1]);
            dst += dst_stride;
            h -= 2;
        } while (h);
    } else if (w == 8) {
        do {
            __m128i s[2];
            s[0] = _mm_loadu_si128((__m128i*)src);
            src += src_stride;
            s[1] = _mm_loadu_si128((__m128i*)src);
            src += src_stride;
            _mm_storeu_si128((__m128i*)dst, s[0]);
            dst += dst_stride;
            _mm_storeu_si128((__m128i*)dst, s[1]);
            dst += dst_stride;
            h -= 2;
        } while (h);
    } else if (w == 16) {
        do {
            __m256i s[2];
            s[0] = _mm256_loadu_si256((__m256i*)src);
            src += src_stride;
            s[1] = _mm256_loadu_si256((__m256i*)src);
            src += src_stride;
            _mm256_storeu_si256((__m256i*)dst, s[0]);
            dst += dst_stride;
            _mm256_storeu_si256((__m256i*)dst, s[1]);
            dst += dst_stride;
            h -= 2;
        } while (h);
    } else if (w == 32) {
        do {
            __m256i s[4];
            s[0] = _mm256_loadu_si256((__m256i*)(src + 0 * 16));
            s[1] = _mm256_loadu_si256((__m256i*)(src + 1 * 16));
            src += src_stride;
            s[2] = _mm256_loadu_si256((__m256i*)(src + 0 * 16));
            s[3] = _mm256_loadu_si256((__m256i*)(src + 1 * 16));
            src += src_stride;
            _mm256_storeu_si256((__m256i*)(dst + 0 * 16), s[0]);
            _mm256_storeu_si256((__m256i*)(dst + 1 * 16), s[1]);
            dst += dst_stride;
            _mm256_storeu_si256((__m256i*)(dst + 0 * 16), s[2]);
            _mm256_storeu_si256((__m256i*)(dst + 1 * 16), s[3]);
            dst += dst_stride;
            h -= 2;
        } while (h);
    } else if (w == 64) {
        do {
            copy_64(src, dst);
            src += src_stride;
            dst += dst_stride;
            copy_64(src, dst);
            src += src_stride;
            dst += dst_stride;
            h -= 2;
        } while (h);
    } else {
        do {
            copy_128(src, dst);
            src += src_stride;
            dst += dst_stride;
            copy_128(src, dst);
            src += src_stride;
            dst += dst_stride;
            h -= 2;
        } while (h);
    }
}
