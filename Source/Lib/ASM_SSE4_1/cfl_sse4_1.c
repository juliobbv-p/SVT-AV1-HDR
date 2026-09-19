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

#include <string.h>
#include <smmintrin.h>
#include "definitions.h"

static INLINE __m128i predict_unclipped_sse4_1(const __m128i* input, __m128i alpha_q12, __m128i alpha_sign,
                                               __m128i dc_q0) {
    __m128i ac_q3          = _mm_loadu_si128(input);
    __m128i ac_sign        = _mm_sign_epi16(alpha_sign, ac_q3);
    __m128i scaled_luma_q0 = _mm_mulhrs_epi16(_mm_abs_epi16(ac_q3), alpha_q12);
    scaled_luma_q0         = _mm_sign_epi16(scaled_luma_q0, ac_sign);
    return _mm_add_epi16(scaled_luma_q0, dc_q0);
}

void svt_cfl_predict_lbd_sse4_1(const int16_t* pred_buf_q3, uint8_t* pred, int32_t pred_stride, uint8_t* dst,
                                int32_t dst_stride, int32_t alpha_q3, int32_t bit_depth, int32_t width,
                                int32_t height) {
    (void)bit_depth;
    (void)pred_stride;
    const __m128i        alpha_sign = _mm_set1_epi16((int16_t)alpha_q3);
    const __m128i        alpha_q12  = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
    const __m128i        dc_q0      = _mm_set1_epi16(*pred);
    const __m128i*       row        = (const __m128i*)pred_buf_q3;
    const __m128i* const row_end    = row + height * CFL_BUF_LINE_I128;
    do {
        if (width < 16) {
            __m128i res = predict_unclipped_sse4_1(row, alpha_q12, alpha_sign, dc_q0);
            res         = _mm_packus_epi16(res, res);
            if (width == 4) {
                const int v = _mm_cvtsi128_si32(res);
                memcpy(dst, &v, 4);
            } else {
                _mm_storel_epi64((__m128i*)dst, res);
            }
        } else {
            for (int32_t x = 0; x < width; x += 16) {
                const __m128i a = predict_unclipped_sse4_1(row + (x >> 3), alpha_q12, alpha_sign, dc_q0);
                const __m128i b = predict_unclipped_sse4_1(row + (x >> 3) + 1, alpha_q12, alpha_sign, dc_q0);
                _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(a, b));
            }
        }
        dst += dst_stride;
        row += CFL_BUF_LINE_I128;
    } while (row < row_end);
}

// (1 << bd) - 1, computed as -1 ^ (-1 << bd) to avoid a variable-shift of 1.
static INLINE __m128i highbd_max_epi16_sse4_1(int32_t bd) {
    const __m128i neg_one = _mm_set1_epi16(-1);
    return _mm_xor_si128(_mm_slli_epi16(neg_one, bd), neg_one);
}

void svt_cfl_predict_hbd_sse4_1(const int16_t* pred_buf_q3, uint16_t* pred, int32_t pred_stride, uint16_t* dst,
                                int32_t dst_stride, int32_t alpha_q3, int32_t bit_depth, int32_t width,
                                int32_t height) {
    (void)pred_stride;
    const __m128i        alpha_sign = _mm_set1_epi16((int16_t)alpha_q3);
    const __m128i        alpha_q12  = _mm_slli_epi16(_mm_abs_epi16(alpha_sign), 9);
    const __m128i        dc_q0      = _mm_set1_epi16(*pred);
    const __m128i        max        = highbd_max_epi16_sse4_1(bit_depth);
    const __m128i        zeros      = _mm_setzero_si128();
    const __m128i*       row        = (const __m128i*)pred_buf_q3;
    const __m128i* const row_end    = row + height * CFL_BUF_LINE_I128;
    do {
        for (int32_t x = 0; x < width; x += 8) {
            __m128i res = predict_unclipped_sse4_1(row + (x >> 3), alpha_q12, alpha_sign, dc_q0);
            res         = _mm_max_epi16(_mm_min_epi16(res, max), zeros);
            if (width == 4) {
                _mm_storel_epi64((__m128i*)dst, res);
            } else {
                _mm_storeu_si128((__m128i*)(dst + x), res);
            }
        }
        dst += dst_stride;
        row += CFL_BUF_LINE_I128;
    } while (row < row_end);
}

// 420 subsampling: out = (top[2i] + top[2i+1] + bot[2i] + bot[2i+1]) << 1.
// For lbd, _mm_maddubs_epi16 against a vector of 2s yields 2*(a[2i]+a[2i+1])
// directly; summing the top/bot maddubs results gives the << 1 of the 2x2 sum.
void svt_cfl_luma_subsampling_420_lbd_sse4_1(const uint8_t* input, int32_t input_stride, int16_t* output_q3,
                                             int32_t width, int32_t height) {
    const __m128i  twos        = _mm_set1_epi8(2);
    const int      luma_stride = input_stride << 1;
    __m128i*       row         = (__m128i*)output_q3;
    const __m128i* row_end     = row + (height >> 1) * CFL_BUF_LINE_I128;
    do {
        if (width == 4) {
            __m128i       top = _mm_maddubs_epi16(_mm_cvtsi32_si128(*(const int*)input), twos);
            __m128i       bot = _mm_maddubs_epi16(_mm_cvtsi32_si128(*(const int*)(input + input_stride)), twos);
            const __m128i sum = _mm_add_epi16(top, bot);
            const int     v   = _mm_cvtsi128_si32(sum);
            memcpy(row, &v, 4);
        } else if (width == 8) {
            __m128i       top = _mm_maddubs_epi16(_mm_loadl_epi64((const __m128i*)input), twos);
            __m128i       bot = _mm_maddubs_epi16(_mm_loadl_epi64((const __m128i*)(input + input_stride)), twos);
            const __m128i sum = _mm_add_epi16(top, bot);
            _mm_storel_epi64(row, sum);
        } else {
            for (int32_t x = 0; x < width; x += 16) {
                __m128i top = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*)(input + x)), twos);
                __m128i bot = _mm_maddubs_epi16(_mm_loadu_si128((const __m128i*)(input + x + input_stride)), twos);
                _mm_storeu_si128(row + (x >> 4), _mm_add_epi16(top, bot));
            }
        }
        input += luma_stride;
    } while ((row += CFL_BUF_LINE_I128) < row_end);
}

void svt_cfl_luma_subsampling_420_hbd_sse4_1(const uint16_t* input, int32_t input_stride, int16_t* output_q3,
                                             int32_t width, int32_t height) {
    const int      luma_stride = input_stride << 1;
    __m128i*       row         = (__m128i*)output_q3;
    const __m128i* row_end     = row + (height >> 1) * CFL_BUF_LINE_I128;
    do {
        if (width == 4) {
            const __m128i top = _mm_loadl_epi64((const __m128i*)input);
            const __m128i bot = _mm_loadl_epi64((const __m128i*)(input + input_stride));
            __m128i       sum = _mm_add_epi16(top, bot);
            sum               = _mm_hadd_epi16(sum, sum);
            const int v       = _mm_cvtsi128_si32(_mm_add_epi16(sum, sum));
            memcpy(row, &v, 4);
        } else if (width == 8) {
            const __m128i top = _mm_loadu_si128((const __m128i*)input);
            const __m128i bot = _mm_loadu_si128((const __m128i*)(input + input_stride));
            __m128i       sum = _mm_add_epi16(top, bot);
            sum               = _mm_hadd_epi16(sum, sum);
            _mm_storel_epi64(row, _mm_add_epi16(sum, sum));
        } else {
            for (int32_t x = 0; x < width; x += 16) {
                const __m128i top_a = _mm_loadu_si128((const __m128i*)(input + x));
                const __m128i bot_a = _mm_loadu_si128((const __m128i*)(input + x + input_stride));
                const __m128i top_b = _mm_loadu_si128((const __m128i*)(input + x + 8));
                const __m128i bot_b = _mm_loadu_si128((const __m128i*)(input + x + 8 + input_stride));
                __m128i       hsum  = _mm_hadd_epi16(_mm_add_epi16(top_a, bot_a), _mm_add_epi16(top_b, bot_b));
                _mm_storeu_si128(row + (x >> 4), _mm_add_epi16(hsum, hsum));
            }
        }
        input += luma_stride;
    } while ((row += CFL_BUF_LINE_I128) < row_end);
}
