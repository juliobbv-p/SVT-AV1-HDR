/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include "definitions.h"
#include <immintrin.h> // AVX2

#include "aom_dsp_rtcd.h"

#include "memory_avx2.h"

static INLINE __m256i variance_pixels(const uint16_t* p, int stride, int w) {
    if (w >= 16) {
        return _mm256_loadu_si256((const __m256i*)p);
    }
    if (w == 8) {
        return loadu_u16_8x2_avx2(p, stride);
    }
    return load_u8_8x4_avx2((const uint8_t*)p, 2 * stride);
}

static INLINE int horizontal_sum32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s         = _mm_add_epi32(s, _mm_srli_si128(s, 8));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_srli_si128(s, 4)));
}

static INLINE uint32_t variance_block(const uint16_t* a, int as, const uint16_t* b, int bs, int w, int h, uint32_t* sse,
                                      int normalize) {
    __m256i    squares = _mm256_setzero_si256(), sums = squares;
    const bool narrow = w * h <= 512;
    const int  rows   = w < 16 ? 16 / w : 1;
    for (int y = 0; y < h; y += rows) {
        for (int x = 0; x < w; x += 16) {
            const __m256i d = _mm256_sub_epi16(variance_pixels(a + x, as, w), variance_pixels(b + x, bs, w));
            sums            = narrow ? _mm256_add_epi16(sums, d)
                                     : _mm256_add_epi32(sums, _mm256_madd_epi16(d, _mm256_set1_epi16(1)));
            squares         = _mm256_add_epi32(squares, _mm256_madd_epi16(d, d));
        }
        a += rows * as;
        b += rows * bs;
    }
    // Up to 32 ten-bit differences fit per word. Widen just once on small blocks.
    if (narrow) {
        sums = _mm256_madd_epi16(sums, _mm256_set1_epi16(1));
    }
    uint64_t energy;
    // At 128x128 each dword sees at most 2048 ten-bit squares.
    if (w * h > 4096) {
        const __m256i words = _mm256_add_epi64(_mm256_srli_epi64(squares, 32),
                                               _mm256_and_si256(squares, _mm256_set1_epi64x(UINT32_MAX)));
        const __m128i v     = _mm_add_epi64(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
        energy              = _mm_cvtsi128_si64(_mm_add_epi64(v, _mm_srli_si128(v, 8)));
    } else {
        energy = (uint32_t)horizontal_sum32(squares);
    }
    int64_t sum = horizontal_sum32(sums);
    if (normalize) {
        sum    = ROUND_POWER_OF_TWO(sum, 2);
        energy = ROUND_POWER_OF_TWO(energy, 4);
    }
    *sse                   = (uint32_t)energy;
    const int64_t variance = (int64_t)*sse - (sum * sum) / (w * h);
    return normalize ? (uint32_t)AOMMAX(0, variance) : (uint32_t)variance;
}

#define VAR_FN(w, h)                                                                                 \
    uint32_t svt_aom_highbd_10_variance##w##x##h##_avx2(                                             \
        const uint8_t* a, int as, const uint8_t* b, int bs, uint32_t* sse) {                         \
        return variance_block(CONVERT_TO_SHORTPTR(a), as, CONVERT_TO_SHORTPTR(b), bs, w, h, sse, 1); \
    }

VAR_FN(128, 128)
VAR_FN(128, 64)
VAR_FN(64, 128)
VAR_FN(64, 64)
VAR_FN(64, 32)
VAR_FN(32, 64)
VAR_FN(32, 32)
VAR_FN(32, 16)
VAR_FN(16, 32)
VAR_FN(16, 16)
VAR_FN(16, 8)
VAR_FN(8, 16)
VAR_FN(8, 8)
VAR_FN(8, 32)
VAR_FN(32, 8)
VAR_FN(16, 64)
VAR_FN(64, 16)
VAR_FN(4, 4)
VAR_FN(4, 8)
VAR_FN(4, 16)
VAR_FN(8, 4)
VAR_FN(16, 4)
#undef VAR_FN

uint32_t svt_aom_variance_highbd_avx2(const uint16_t* a, int as, const uint16_t* b, int bs, int w, int h,
                                      uint32_t* sse) {
    assert(w == h && (w == 16 || w == 32));
    (void)h;
    return w == 16 ? variance_block(a, as, b, bs, 16, 16, sse, 0) : variance_block(a, as, b, bs, 32, 32, sse, 0);
}
