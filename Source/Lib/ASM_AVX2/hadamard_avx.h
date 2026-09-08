/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#ifndef SVT_HADAMARD_AVX_H
#define SVT_HADAMARD_AVX_H

#include <immintrin.h>
#include "common_dsp_rtcd.h"
#include "synonyms.h"

#ifdef HADAMARD_AVX512
typedef __m512i HadamardVec;
#define V(op) _mm512_##op
#else
typedef __m256i HadamardVec;
#define V(op) _mm256_##op
#endif

SIMD_INLINE __m128i load8(const uint8_t* p, int hbd) {
    return hbd ? _mm_loadu_si128((const __m128i*)p) : _mm_cvtepu8_epi16(xx_loadl_64(p));
}

SIMD_INLINE __m256i join128(__m128i a, __m128i b) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
}

// Signed saturation is exact here. It keeps the first pair differences
// independent when the compiler reassociates wrapping word arithmetic.
SIMD_INLINE void butterfly4(HadamardVec r[4]) {
    const HadamardVec a = V(add_epi16)(r[0], r[1]), b = V(subs_epi16)(r[0], r[1]);
    const HadamardVec c = V(add_epi16)(r[2], r[3]), d = V(subs_epi16)(r[2], r[3]);
    r[0] = V(add_epi16)(a, c);
    r[1] = V(sub_epi16)(a, c);
    r[2] = V(add_epi16)(b, d);
    r[3] = V(sub_epi16)(b, d);
}

// Place four vertical frequencies in each qword. Each register then holds
// a pair of adjacent columns per 128-bit lane.
SIMD_INLINE void transpose4x8(HadamardVec r[4]) {
    const HadamardVec a = V(unpacklo_epi16)(r[0], r[1]), b = V(unpacklo_epi16)(r[2], r[3]);
    const HadamardVec c = V(unpackhi_epi16)(r[0], r[1]), d = V(unpackhi_epi16)(r[2], r[3]);
    r[0] = V(unpacklo_epi32)(a, b);
    r[1] = V(unpackhi_epi32)(a, b);
    r[2] = V(unpacklo_epi32)(c, d);
    r[3] = V(unpackhi_epi32)(c, d);
}

SIMD_INLINE void last_columns(HadamardVec r[4]) {
    for (int i = 0; i < 4; i += 2) {
        const HadamardVec even = V(unpacklo_epi64)(r[i], r[i + 1]), odd = V(unpackhi_epi64)(r[i], r[i + 1]);
        r[i]     = V(add_epi16)(even, odd);
        r[i + 1] = V(sub_epi16)(even, odd);
    }
}

// Five butterfly stages cover a 4x8 half-tile. Even a 10-bit residual fits
// signed words through this stage (32 * 1023 = 32736).
SIMD_INLINE void transform4x8(HadamardVec r[4]) {
    butterfly4(r);
    transpose4x8(r);
    butterfly4(r);
    last_columns(r);
}

// |a+b| + |a-b| = 2*max(|a|,|b|). This gives half the SATD of the
// complete 8x8. Each 128-bit lane contains eight unsigned partial sums.
SIMD_INLINE HadamardVec magnitude_sums(const HadamardVec top[4], const HadamardVec bottom[4]) {
    HadamardVec m[4];
    for (int i = 0; i < 4; i++) {
        m[i] = V(max_epi16)(V(abs_epi16)(top[i]), V(abs_epi16)(bottom[i]));
    }
    // The three non-DC rows each have magnitude <= 16*1023. Two selected
    // from the same half have joint magnitude <= 16*1023 by orthogonality.
    // Including the DC row, the four maxima therefore fit an unsigned word.
    return V(add_epi16)(V(add_epi16)(m[0], m[1]), V(add_epi16)(m[2], m[3]));
}

SIMD_INLINE int reduce128(__m128i x) {
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0x4e));
    return _mm_cvtsi128_si32(_mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1)));
}

SIMD_INLINE __m256i load16(const uint8_t* p, int hbd) {
    return hbd ? _mm256_loadu_si256((const __m256i*)p) : _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)p));
}

SIMD_INLINE HadamardVec load_row(const uint8_t* p, int hbd) {
#ifdef HADAMARD_AVX512
    return hbd ? _mm512_loadu_si512(p) : _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)p));
#else
    return load16(p, hbd);
#endif
}

SIMD_INLINE HadamardVec load_quadrants(const uint8_t* p, ptrdiff_t stride, int hbd, int wide) {
#ifdef HADAMARD_AVX512
    if (wide) {
        return load_row(p, hbd);
    }
    return _mm512_inserti64x4(_mm512_castsi256_si512(load16(p, hbd)), load16(p + 8 * stride, hbd), 1);
#else
    (void)stride;
    (void)wide;
    return load16(p, hbd);
#endif
}

SIMD_INLINE HadamardVec tile_columns(HadamardVec a, HadamardVec b, int right) {
#ifdef HADAMARD_AVX512
    return right ? _mm512_shuffle_i64x2(a, b, 0xdd) : _mm512_shuffle_i64x2(a, b, 0x88);
#else
    return right ? _mm256_permute2x128_si256(a, b, 0x31) : _mm256_permute2x128_si256(a, b, 0x20);
#endif
}

// Transform two neighboring tiles per band. A last-stage 10-bit coefficient
// can reach 65472. PMADDWD completes that butterfly directly in dwords.
static INLINE void horizontal16(const uint8_t* s, ptrdiff_t ss, const uint8_t* p, ptrdiff_t ps, int hbd, int wide,
                                HadamardVec out[16]) {
    HadamardVec top[4], bot[4];
    for (int i = 0; i < 4; i++) {
        top[i] = V(sub_epi16)(load_quadrants(s + i * ss, ss, hbd, wide), load_quadrants(p + i * ps, ps, hbd, wide));
        bot[i] = V(sub_epi16)(load_quadrants(s + (i + 4) * ss, ss, hbd, wide),
                              load_quadrants(p + (i + 4) * ps, ps, hbd, wide));
    }
    transform4x8(top);
    transform4x8(bot);
    for (int i = 0; i < 4; i++) {
        if (!hbd) {
            const HadamardVec a = V(add_epi16)(top[i], bot[i]), b = V(sub_epi16)(top[i], bot[i]);
            const HadamardVec l = tile_columns(a, b, 0), r = tile_columns(a, b, 1);
            const HadamardVec plus = V(srai_epi16)(V(add_epi16)(l, r), 1), minus = V(srai_epi16)(V(sub_epi16)(l, r), 1);
            out[4 * i]     = V(srai_epi32)(V(slli_epi32)(plus, 16), 16);
            out[4 * i + 1] = V(srai_epi32)(plus, 16);
            out[4 * i + 2] = V(srai_epi32)(V(slli_epi32)(minus, 16), 16);
            out[4 * i + 3] = V(srai_epi32)(minus, 16);
            continue;
        }
        const HadamardVec lo = V(unpacklo_epi16)(top[i], bot[i]), hi = V(unpackhi_epi16)(top[i], bot[i]);
        for (int j = 0; j < 2; j++) {
            // Alternating +1/-1 words produce the difference in PMADDWD.
            const HadamardVec sign = j ? V(set1_epi32)(-65535) : V(set1_epi16)(1);
            const HadamardVec a = V(madd_epi16)(lo, sign), b = V(madd_epi16)(hi, sign);
            const HadamardVec left = tile_columns(a, b, 0), right = tile_columns(a, b, 1);
            out[4 * i + 2 * j]     = V(srai_epi32)(V(add_epi32)(left, right), 1);
            out[4 * i + 2 * j + 1] = V(srai_epi32)(V(sub_epi32)(left, right), 1);
        }
    }
}

SIMD_INLINE int finish_satd(HadamardVec sum) {
#ifdef HADAMARD_AVX512
    return 2 * _mm512_reduce_add_epi32(sum);
#else
    return 2 * reduce128(_mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1)));
#endif
}

// A standalone 16x16 only needs the magnitudes of the vertical merge.
static INLINE int satd16(const uint8_t* s, ptrdiff_t ss, const uint8_t* p, ptrdiff_t ps, int hbd) {
    HadamardVec top[16];
    horizontal16(s, ss, p, ps, hbd, 0, top);
    HadamardVec sum = V(set1_epi32)(0);
#ifdef HADAMARD_AVX512
    // Adjacent 128-bit lanes hold matching top and bottom coefficients.
    // Pack both signs of the horizontal merge before taking their maxima.
    for (int i = 0; i < 16; i += 2) {
        const __m512i a = _mm512_abs_epi32(top[i]), b = _mm512_abs_epi32(top[i + 1]);
        sum = _mm512_add_epi32(sum, _mm512_max_epi32(tile_columns(a, b, 0), tile_columns(a, b, 1)));
    }
#else
    HadamardVec bot[16];
    horizontal16(s + 8 * ss, ss, p + 8 * ps, ps, hbd, 0, bot);
    for (int i = 0; i < 16; i++) {
        sum = _mm256_add_epi32(sum, _mm256_max_epi32(_mm256_abs_epi32(top[i]), _mm256_abs_epi32(bot[i])));
    }
#endif
    return finish_satd(sum);
}

SIMD_INLINE int satd32(const uint8_t* s, ptrdiff_t ss, const uint8_t* p, ptrdiff_t ps, int hbd) {
    // AVX-512 holds two adjacent 16x8 bands per register. AVX2 holds one
    // and keeps the two vertical bands of each quadrant together. Both
    // merges precede the prescribed >> 2, so they may run in either order.
#ifdef HADAMARD_AVX512
    HadamardVec coeff[4][16];
#else
    HadamardVec coeff[4][2][16];
#endif
    for (int i = 0; i < 4; i++) {
#ifdef HADAMARD_AVX512
        horizontal16(s + 8 * i * ss, ss, p + 8 * i * ps, ps, hbd, 1, coeff[i]);
#else
        const int x = (i & 1) * (16 << hbd), y = (i >> 1) * 16;
        for (int j = 0; j < 2; j++) {
            horizontal16(s + (y + 8 * j) * ss + x, ss, p + (y + 8 * j) * ps + x, ps, hbd, 0, coeff[i][j]);
        }
#endif
    }
    HadamardVec sum = V(set1_epi32)(0);
    for (int i = 0; i < 16; i++) {
        for (int sign = 0; sign < 2; sign++) {
#ifdef HADAMARD_AVX512
            const __m512i top = sign ? _mm512_sub_epi32(coeff[0][i], coeff[1][i])
                                     : _mm512_add_epi32(coeff[0][i], coeff[1][i]);
            const __m512i bot = sign ? _mm512_sub_epi32(coeff[2][i], coeff[3][i])
                                     : _mm512_add_epi32(coeff[2][i], coeff[3][i]);
            const __m512i l = tile_columns(top, bot, 0), r = tile_columns(top, bot, 1);
            const __m512i a = _mm512_abs_epi32(_mm512_srai_epi32(_mm512_add_epi32(l, r), 2));
            const __m512i b = _mm512_abs_epi32(_mm512_srai_epi32(_mm512_sub_epi32(l, r), 2));
            // The top and bottom quadrants now occupy separate 256-bit halves.
            sum = _mm512_add_epi32(
                sum, _mm512_max_epi32(_mm512_shuffle_i64x2(a, b, 0x44), _mm512_shuffle_i64x2(a, b, 0xee)));
#else
            __m256i hi[2], lo[2];
            for (int row = 0; row < 2; row++) {
                const __m256i a = coeff[2 * row][0][i], b = coeff[2 * row + 1][0][i];
                const __m256i c = coeff[2 * row][1][i], d = coeff[2 * row + 1][1][i];
                lo[row] = sign ? V(sub_epi32)(a, b) : V(add_epi32)(a, b);
                hi[row] = sign ? V(sub_epi32)(c, d) : V(add_epi32)(c, d);
            }
            const __m256i a = _mm256_srai_epi32(_mm256_add_epi32(lo[0], hi[0]), 2);
            const __m256i b = _mm256_srai_epi32(_mm256_add_epi32(lo[1], hi[1]), 2);
            const __m256i c = _mm256_srai_epi32(_mm256_sub_epi32(lo[0], hi[0]), 2);
            const __m256i d = _mm256_srai_epi32(_mm256_sub_epi32(lo[1], hi[1]), 2);
            const __m256i m = _mm256_max_epi32(_mm256_abs_epi32(a), _mm256_abs_epi32(b));
            const __m256i k = _mm256_max_epi32(_mm256_abs_epi32(c), _mm256_abs_epi32(d));
            sum             = _mm256_add_epi32(sum, _mm256_add_epi32(m, k));
#endif
        }
    }
    return finish_satd(sum);
}
#endif
