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

/* SSE4.1 tier for the low-bit-depth DC/V intra predictors at widths 32/64.
 * These dispatched C -> AVX2(/AVX512) with no SSE tier; the 4/8/16-wide sizes
 * already have SSE2. Bit-identical to the C reference in intra_prediction.c. */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include "definitions.h"

// Sum of n uint8 (n a multiple of 16) via SAD-against-zero.
static INLINE uint32_t dcv_sum_u8(const uint8_t* p, int32_t n) {
    const __m128i zero = _mm_setzero_si128();
    __m128i       acc  = _mm_setzero_si128();
    for (int32_t i = 0; i < n; i += 16)
        acc = _mm_add_epi32(acc, _mm_sad_epu8(_mm_loadu_si128((const __m128i*)(p + i)), zero));
    return (uint32_t)_mm_cvtsi128_si32(acc) + (uint32_t)_mm_extract_epi32(acc, 2);
}

// Broadcast-fill a w x h block (w a multiple of 16) with a constant byte.
static INLINE void dcv_fill(uint8_t* dst, ptrdiff_t stride, int32_t w, int32_t h, uint8_t val) {
    const __m128i v = _mm_set1_epi8((char)val);
    for (int32_t r = 0; r < h; r++, dst += stride)
        for (int32_t c = 0; c < w; c += 16) _mm_storeu_si128((__m128i*)(dst + c), v);
}

// Copy the top row to every row (w a multiple of 16).
static INLINE void dcv_v(uint8_t* dst, ptrdiff_t stride, int32_t w, int32_t h, const uint8_t* above) {
    for (int32_t r = 0; r < h; r++, dst += stride)
        for (int32_t c = 0; c < w; c += 16)
            _mm_storeu_si128((__m128i*)(dst + c), _mm_loadu_si128((const __m128i*)(above + c)));
}

#define DCV_V_FN(w, h)                                                                                            \
    void svt_aom_v_predictor_##w##x##h##_sse4_1(uint8_t* dst, ptrdiff_t stride, const uint8_t* above,             \
                                                const uint8_t* left) {                                            \
        (void)left;                                                                                              \
        dcv_v(dst, stride, w, h, above);                                                                         \
    }

#define DCV_DC128_FN(w, h)                                                                                       \
    void svt_aom_dc_128_predictor_##w##x##h##_sse4_1(uint8_t* dst, ptrdiff_t stride, const uint8_t* above,        \
                                                     const uint8_t* left) {                                       \
        (void)above;                                                                                             \
        (void)left;                                                                                              \
        dcv_fill(dst, stride, w, h, 128);                                                                        \
    }

#define DCV_DCL_FN(w, h)                                                                                         \
    void svt_aom_dc_left_predictor_##w##x##h##_sse4_1(uint8_t* dst, ptrdiff_t stride, const uint8_t* above,       \
                                                      const uint8_t* left) {                                      \
        (void)above;                                                                                             \
        const uint32_t s = dcv_sum_u8(left, h);                                                                  \
        dcv_fill(dst, stride, w, h, (uint8_t)((s + (h >> 1)) / h));                                              \
    }

#define DCV_DCT_FN(w, h)                                                                                         \
    void svt_aom_dc_top_predictor_##w##x##h##_sse4_1(uint8_t* dst, ptrdiff_t stride, const uint8_t* above,        \
                                                     const uint8_t* left) {                                       \
        (void)left;                                                                                              \
        const uint32_t s = dcv_sum_u8(above, w);                                                                 \
        dcv_fill(dst, stride, w, h, (uint8_t)((s + (w >> 1)) / w));                                              \
    }

#define DCV_DC_FN(w, h)                                                                                          \
    void svt_aom_dc_predictor_##w##x##h##_sse4_1(uint8_t* dst, ptrdiff_t stride, const uint8_t* above,            \
                                                 const uint8_t* left) {                                           \
        const uint32_t s = dcv_sum_u8(above, w) + dcv_sum_u8(left, h);                                           \
        dcv_fill(dst, stride, w, h, (uint8_t)((s + ((w + h) >> 1)) / (w + h)));                                  \
    }

#define DCV_SIZES(X) X(32, 16) X(32, 32) X(32, 64) X(64, 16) X(64, 32) X(64, 64)

DCV_SIZES(DCV_V_FN)
DCV_SIZES(DCV_DC128_FN)
DCV_SIZES(DCV_DCL_FN)
DCV_SIZES(DCV_DCT_FN)
DCV_SIZES(DCV_DC_FN)

// H predictor: each row is a broadcast of left[r]. Only 32x32 lacks an SSE tier.
void svt_aom_h_predictor_32x32_sse4_1(uint8_t* dst, ptrdiff_t stride, const uint8_t* above, const uint8_t* left) {
    (void)above;
    for (int32_t r = 0; r < 32; r++, dst += stride) {
        const __m128i v = _mm_set1_epi8((char)left[r]);
        _mm_storeu_si128((__m128i*)dst, v);
        _mm_storeu_si128((__m128i*)(dst + 16), v);
    }
}
