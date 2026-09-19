/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

/* SSE4.1 tiers for three trivially-vectorizable helpers that had C -> AVX2
 * dispatch with no SSE tier. Bit-identical to their C references. */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include "definitions.h"

// dst[k] = src[k], widening uint8 -> uint16.
void svt_convert_8bit_to_16bit_sse4_1(uint8_t* src, uint32_t src_stride, uint16_t* dst, uint32_t dst_stride,
                                      uint32_t width, uint32_t height) {
    for (uint32_t j = 0; j < height; j++) {
        const uint8_t* s = src + j * src_stride;
        uint16_t*      d = dst + j * dst_stride;
        uint32_t       k = 0;
        for (; k + 16 <= width; k += 16) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(s + k));
            _mm_storeu_si128((__m128i*)(d + k), _mm_cvtepu8_epi16(v));
            _mm_storeu_si128((__m128i*)(d + k + 8), _mm_cvtepu8_epi16(_mm_srli_si128(v, 8)));
        }
        for (; k < width; k++) d[k] = s[k];
    }
}

// dst[k] = (uint8_t)src[k], truncating uint16 -> uint8 (low byte, not saturating).
void svt_convert_16bit_to_8bit_sse4_1(uint16_t* src, uint32_t src_stride, uint8_t* dst, uint32_t dst_stride,
                                      uint32_t width, uint32_t height) {
    const __m128i lo = _mm_set1_epi16(0x00ff);
    for (uint32_t j = 0; j < height; j++) {
        const uint16_t* s = src + j * src_stride;
        uint8_t*        d = dst + j * dst_stride;
        uint32_t        k = 0;
        for (; k + 16 <= width; k += 16) {
            const __m128i a = _mm_and_si128(_mm_loadu_si128((const __m128i*)(s + k)), lo);
            const __m128i b = _mm_and_si128(_mm_loadu_si128((const __m128i*)(s + k + 8)), lo);
            _mm_storeu_si128((__m128i*)(d + k), _mm_packus_epi16(a, b));
        }
        for (; k < width; k++) d[k] = (uint8_t)s[k];
    }
}

// diff[c] = src[c] - pred[c] as int16.
void svt_aom_subtract_block_sse4_1(int rows, int cols, int16_t* diff, ptrdiff_t diff_stride, const uint8_t* src,
                                   ptrdiff_t src_stride, const uint8_t* pred, ptrdiff_t pred_stride) {
    for (int r = 0; r < rows; r++, diff += diff_stride, src += src_stride, pred += pred_stride) {
        int c = 0;
        for (; c + 16 <= cols; c += 16) {
            const __m128i s = _mm_loadu_si128((const __m128i*)(src + c));
            const __m128i p = _mm_loadu_si128((const __m128i*)(pred + c));
            _mm_storeu_si128((__m128i*)(diff + c),
                             _mm_sub_epi16(_mm_cvtepu8_epi16(s), _mm_cvtepu8_epi16(p)));
            _mm_storeu_si128((__m128i*)(diff + c + 8),
                             _mm_sub_epi16(_mm_cvtepu8_epi16(_mm_srli_si128(s, 8)),
                                           _mm_cvtepu8_epi16(_mm_srli_si128(p, 8))));
        }
        for (; c + 8 <= cols; c += 8) {
            const __m128i s = _mm_loadl_epi64((const __m128i*)(src + c));
            const __m128i p = _mm_loadl_epi64((const __m128i*)(pred + c));
            _mm_storeu_si128((__m128i*)(diff + c), _mm_sub_epi16(_mm_cvtepu8_epi16(s), _mm_cvtepu8_epi16(p)));
        }
        for (; c < cols; c++) diff[c] = (int16_t)(src[c] - pred[c]);
    }
}
