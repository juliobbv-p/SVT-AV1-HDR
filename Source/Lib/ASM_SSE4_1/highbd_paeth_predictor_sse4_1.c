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

/* SSE4.1 tier for the high-bit-depth Paeth intra predictor. These sizes
 * dispatched C -> AVX2 with no SSE tier. Bit-identical to the C reference:
 * per pixel, pick whichever of {left, top, top_left} is nearest to
 * base = top + left - top_left, with ties preferring left, then top.
 * For bd <= 12 the base and the abs-differences all fit in signed 16-bit
 * (|.| <= ~12k < 32767), so the whole select runs at int16.
 * Referenced against dav1d's ipred16_sse.asm SSSE3 Paeth. */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include "definitions.h"

static INLINE __m128i paeth_row_sse4_1(__m128i t, __m128i l, __m128i tl) {
    const __m128i ones = _mm_set1_epi16(-1);
    const __m128i base = _mm_sub_epi16(_mm_add_epi16(t, l), tl);
    const __m128i pl   = _mm_abs_epi16(_mm_sub_epi16(base, l));
    const __m128i pt   = _mm_abs_epi16(_mm_sub_epi16(base, t));
    const __m128i ptl  = _mm_abs_epi16(_mm_sub_epi16(base, tl));
    // a <= b  ==  ~(a > b)
    const __m128i le_pl_pt  = _mm_xor_si128(_mm_cmpgt_epi16(pl, pt), ones);
    const __m128i le_pl_ptl = _mm_xor_si128(_mm_cmpgt_epi16(pl, ptl), ones);
    const __m128i mask_left = _mm_and_si128(le_pl_pt, le_pl_ptl);
    const __m128i mask_top  = _mm_xor_si128(_mm_cmpgt_epi16(pt, ptl), ones);
    const __m128i top_or_tl = _mm_blendv_epi8(tl, t, mask_top);
    return _mm_blendv_epi8(top_or_tl, l, mask_left);
}

#define HBD_PAETH_FN(w, h)                                                                                     \
    void svt_aom_highbd_paeth_predictor_##w##x##h##_sse4_1(uint16_t* dst, ptrdiff_t stride,                    \
                                                           const uint16_t* above, const uint16_t* left,        \
                                                           int32_t bd) {                                       \
        (void)bd;                                                                                              \
        const __m128i tl = _mm_set1_epi16((int16_t)above[-1]);                                                 \
        for (int32_t r = 0; r < h; r++, dst += stride) {                                                       \
            const __m128i l = _mm_set1_epi16((int16_t)left[r]);                                                \
            for (int32_t c = 0; c < w; c += 8) {                                                               \
                const __m128i t   = (w == 4) ? _mm_loadl_epi64((const __m128i*)(above + c))                    \
                                             : _mm_loadu_si128((const __m128i*)(above + c));                   \
                const __m128i res = paeth_row_sse4_1(t, l, tl);                                                \
                if (w == 4)                                                                                    \
                    _mm_storel_epi64((__m128i*)(dst + c), res);                                                \
                else                                                                                           \
                    _mm_storeu_si128((__m128i*)(dst + c), res);                                                \
            }                                                                                                  \
        }                                                                                                      \
    }

#define HBD_PAETH_SIZES(X)                                                                                     \
    X(4, 4) X(4, 8) X(4, 16) X(8, 4) X(8, 8) X(8, 16) X(8, 32) X(16, 4) X(16, 8) X(16, 16) X(16, 32) X(16, 64) \
        X(32, 8) X(32, 16) X(32, 32) X(32, 64) X(64, 16) X(64, 32) X(64, 64)

HBD_PAETH_SIZES(HBD_PAETH_FN)
