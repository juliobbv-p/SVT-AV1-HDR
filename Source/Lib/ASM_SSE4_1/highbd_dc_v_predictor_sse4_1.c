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

/* SSE4.1 tier for the high-bit-depth DC/V intra predictors at widths 16/32/64.
 * These had C -> AVX2(/AVX512) dispatch with no SSE tier, so on pre-Haswell
 * x86 they ran scalar C. The narrower 4/8-wide sizes already have SSE2.
 * Bit-identical to the C reference in intra_prediction.c. */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include "definitions.h"

// Widen-sum of n uint16 (n a multiple of 4) into a scalar.
static INLINE uint32_t hbd_sum_u16(const uint16_t* p, int32_t n) {
    __m128i       acc  = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();
    int32_t       i    = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(p + i));
        acc             = _mm_add_epi32(acc, _mm_cvtepu16_epi32(v));
        acc             = _mm_add_epi32(acc, _mm_cvtepu16_epi32(_mm_srli_si128(v, 8)));
    }
    if (i + 4 <= n) {
        const __m128i v = _mm_loadl_epi64((const __m128i*)(p + i));
        acc             = _mm_add_epi32(acc, _mm_cvtepu16_epi32(v));
    }
    (void)zero;
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return (uint32_t)_mm_cvtsi128_si32(acc);
}

// Broadcast-fill a w x h block (w a multiple of 8) with a constant.
static INLINE void hbd_fill(uint16_t* dst, ptrdiff_t stride, int32_t w, int32_t h, uint16_t val) {
    const __m128i v = _mm_set1_epi16((int16_t)val);
    for (int32_t r = 0; r < h; r++, dst += stride)
        for (int32_t c = 0; c < w; c += 8) _mm_storeu_si128((__m128i*)(dst + c), v);
}

// Copy the top row to every row (w a multiple of 8).
static INLINE void hbd_v(uint16_t* dst, ptrdiff_t stride, int32_t w, int32_t h, const uint16_t* above) {
    for (int32_t r = 0; r < h; r++, dst += stride)
        for (int32_t c = 0; c < w; c += 8)
            _mm_storeu_si128((__m128i*)(dst + c), _mm_loadu_si128((const __m128i*)(above + c)));
}

#define HBD_V_FN(w, h)                                                                                       \
    void svt_aom_highbd_v_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride, const uint16_t* above,\
                                                       const uint16_t* left, int32_t bd) {                    \
        (void)left;                                                                                          \
        (void)bd;                                                                                            \
        hbd_v(dst, stride, w, h, above);                                                                     \
    }

#define HBD_DC128_FN(w, h)                                                                                    \
    void svt_aom_highbd_dc_128_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                  \
                                                            const uint16_t* above, const uint16_t* left,      \
                                                            int32_t bd) {                                     \
        (void)above;                                                                                          \
        (void)left;                                                                                           \
        hbd_fill(dst, stride, w, h, (uint16_t)(128 << (bd - 8)));                                             \
    }

#define HBD_DCL_FN(w, h)                                                                                      \
    void svt_aom_highbd_dc_left_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                 \
                                                             const uint16_t* above, const uint16_t* left,     \
                                                             int32_t bd) {                                    \
        (void)above;                                                                                          \
        (void)bd;                                                                                             \
        const uint32_t s = hbd_sum_u16(left, h);                                                              \
        hbd_fill(dst, stride, w, h, (uint16_t)((s + (h >> 1)) / h));                                          \
    }

#define HBD_DCT_FN(w, h)                                                                                      \
    void svt_aom_highbd_dc_top_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                  \
                                                            const uint16_t* above, const uint16_t* left,      \
                                                            int32_t bd) {                                     \
        (void)left;                                                                                           \
        (void)bd;                                                                                             \
        const uint32_t s = hbd_sum_u16(above, w);                                                             \
        hbd_fill(dst, stride, w, h, (uint16_t)((s + (w >> 1)) / w));                                          \
    }

#define HBD_DC_FN(w, h)                                                                                       \
    void svt_aom_highbd_dc_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride, const uint16_t* above,\
                                                        const uint16_t* left, int32_t bd) {                   \
        (void)bd;                                                                                             \
        const uint32_t s = hbd_sum_u16(above, w) + hbd_sum_u16(left, h);                                      \
        hbd_fill(dst, stride, w, h, (uint16_t)((s + ((w + h) >> 1)) / (w + h)));                              \
    }

#define HBD_SIZES(X) \
    X(16, 4) X(16, 8) X(16, 16) X(16, 32) X(16, 64) X(32, 8) X(32, 16) X(32, 32) X(32, 64) X(64, 16) X(64, 32) X(64, 64)

HBD_SIZES(HBD_V_FN)
HBD_SIZES(HBD_DC128_FN)
HBD_SIZES(HBD_DCL_FN)
HBD_SIZES(HBD_DCT_FN)
HBD_SIZES(HBD_DC_FN)

// H predictor: each row is a broadcast of left[r] (w a multiple of 8).
static INLINE void hbd_h(uint16_t* dst, ptrdiff_t stride, int32_t w, int32_t h, const uint16_t* left) {
    for (int32_t r = 0; r < h; r++, dst += stride) {
        const __m128i v = _mm_set1_epi16((int16_t)left[r]);
        for (int32_t c = 0; c < w; c += 8) _mm_storeu_si128((__m128i*)(dst + c), v);
    }
}

#define HBD_H_FN(w, h)                                                                                        \
    void svt_aom_highbd_h_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride, const uint16_t* above,\
                                                       const uint16_t* left, int32_t bd) {                    \
        (void)above;                                                                                         \
        (void)bd;                                                                                            \
        hbd_h(dst, stride, w, h, left);                                                                       \
    }

HBD_H_FN(16, 4) HBD_H_FN(16, 64) HBD_H_FN(32, 8) HBD_H_FN(32, 64) HBD_H_FN(64, 16) HBD_H_FN(64, 32) HBD_H_FN(64, 64)
