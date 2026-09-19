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

/* SSE4.1 tier for the high-bit-depth SMOOTH / SMOOTH_V / SMOOTH_H intra
 * predictors. These sizes dispatched C -> AVX2 with no SSE tier. Bit-identical
 * to the C reference (scale = 256, sm_weight_log2_scale = 8): a weighted
 * average of edge pixels, rounded by divide_round. Row-scalar terms are folded
 * into a single 32-bit constant so each output is (weight*pixel + K) >> shift.
 * Referenced against dav1d ipred16_sse.asm SSSE3 SMOOTH. */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include "definitions.h"

extern const uint8_t sm_weight_arrays[];

// Pack two 4x int32 results (each already >> shift, in [0,4095]) to 8x uint16.
static INLINE void store8(uint16_t* dst, __m128i lo, __m128i hi) {
    _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(lo, hi));
}

// SMOOTH_V: dst[c] = (w_r*above[c] + K) >> 8,  K = (256-w_r)*below + 128.
#define HBD_SMV_FN(w, h)                                                                                       \
    void svt_aom_highbd_smooth_v_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                 \
                                                              const uint16_t* above, const uint16_t* left,     \
                                                              int32_t bd) {                                    \
        (void)bd;                                                                                              \
        const uint16_t       below = left[h - 1];                                                             \
        const uint8_t* const smh   = sm_weight_arrays + h;                                                    \
        for (int32_t r = 0; r < h; r++, dst += stride) {                                                       \
            const int32_t wr  = smh[r];                                                                        \
            const __m128i wrv = _mm_set1_epi32(wr);                                                            \
            const __m128i kv  = _mm_set1_epi32((256 - wr) * below + 128);                                      \
            for (int32_t c = 0; c < w; c += 8) {                                                               \
                const __m128i a  = _mm_loadu_si128((const __m128i*)(above + c));                               \
                const __m128i lo = _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(a), wrv), kv), 8); \
                const __m128i hi = _mm_srli_epi32(                                                             \
                    _mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a, 8)), wrv), kv), 8);     \
                store8(dst + c, lo, hi);                                                                       \
            }                                                                                                  \
        }                                                                                                      \
    }

// SMOOTH_H: dst[c] = (wc[c]*A + B) >> 8,  A = left[r]-right, B = right*256+128.
#define HBD_SMH_FN(w, h)                                                                                       \
    void svt_aom_highbd_smooth_h_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                 \
                                                              const uint16_t* above, const uint16_t* left,     \
                                                              int32_t bd) {                                    \
        (void)bd;                                                                                              \
        const uint16_t       right = above[w - 1];                                                            \
        const uint8_t* const smw   = sm_weight_arrays + w;                                                    \
        for (int32_t r = 0; r < h; r++, dst += stride) {                                                       \
            const __m128i av = _mm_set1_epi32((int32_t)left[r] - right);                                       \
            const __m128i bv = _mm_set1_epi32(right * 256 + 128);                                              \
            for (int32_t c = 0; c < w; c += 8) {                                                               \
                const __m128i wc = _mm_loadl_epi64((const __m128i*)(smw + c));                                 \
                const __m128i lo =                                                                             \
                    _mm_srli_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu8_epi32(wc), av), bv), 8);          \
                const __m128i hi = _mm_srli_epi32(                                                             \
                    _mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(wc, 4)), av), bv), 8);      \
                store8(dst + c, lo, hi);                                                                       \
            }                                                                                                  \
        }                                                                                                      \
    }

// SMOOTH: dst[c] = (w_r*above[c] + wc[c]*M + C) >> 9,
//   M = left[r]-right, C = (256-w_r)*below + right*256 + 256.
#define HBD_SM_FN(w, h)                                                                                        \
    void svt_aom_highbd_smooth_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                   \
                                                            const uint16_t* above, const uint16_t* left,       \
                                                            int32_t bd) {                                      \
        (void)bd;                                                                                              \
        const uint16_t       below = left[h - 1];                                                             \
        const uint16_t       right = above[w - 1];                                                            \
        const uint8_t* const smh   = sm_weight_arrays + h;                                                    \
        const uint8_t* const smw   = sm_weight_arrays + w;                                                    \
        for (int32_t r = 0; r < h; r++, dst += stride) {                                                       \
            const int32_t wr  = smh[r];                                                                        \
            const __m128i wrv = _mm_set1_epi32(wr);                                                            \
            const __m128i mv  = _mm_set1_epi32((int32_t)left[r] - right);                                      \
            const __m128i cv  = _mm_set1_epi32((256 - wr) * below + right * 256 + 256);                        \
            for (int32_t c = 0; c < w; c += 8) {                                                               \
                const __m128i a  = _mm_loadu_si128((const __m128i*)(above + c));                               \
                const __m128i wc = _mm_loadl_epi64((const __m128i*)(smw + c));                                 \
                const __m128i lo = _mm_srli_epi32(                                                             \
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(a), wrv),                   \
                                                _mm_mullo_epi32(_mm_cvtepu8_epi32(wc), mv)),                   \
                                  cv),                                                                         \
                    9);                                                                                        \
                const __m128i wc_hi = _mm_srli_si128(wc, 4);                                                   \
                const __m128i hi    = _mm_srli_epi32(                                                          \
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a, 8)), wrv),\
                                                _mm_mullo_epi32(_mm_cvtepu8_epi32(wc_hi), mv)),                \
                                  cv),                                                                         \
                    9);                                                                                        \
                store8(dst + c, lo, hi);                                                                       \
            }                                                                                                  \
        }                                                                                                      \
    }

#define HBD_SMOOTH_SIZES(X)                                                                                    \
    X(8, 4) X(8, 8) X(8, 16) X(8, 32) X(16, 4) X(16, 8) X(16, 16) X(16, 32) X(16, 64) X(32, 8) X(32, 16)       \
        X(32, 32) X(32, 64) X(64, 16) X(64, 32) X(64, 64)

HBD_SMOOTH_SIZES(HBD_SMV_FN)
HBD_SMOOTH_SIZES(HBD_SMH_FN)
HBD_SMOOTH_SIZES(HBD_SM_FN)
