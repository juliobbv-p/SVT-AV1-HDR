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

#define HADAMARD_AVX512
#include "hadamard_avx.h"

SIMD_INLINE __m512i join256(__m256i a, __m256i b) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
}

// VPSADBW sums the eight low bytes and eight high bytes in separate qwords
// per tile. If their sums are L and H, the rounded half is ceil(L/2) + 128*H.
SIMD_INLINE __m512i rounded_byte_sums(__m512i words) {
    const __m512i order = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
    const __m512i bytes = _mm512_sad_epu8(_mm512_shuffle_epi8(words, order), _mm512_setzero_si512());
    return _mm512_mask_avg_epu16(bytes, 0x01010101, bytes, _mm512_setzero_si512());
}

SIMD_INLINE __m512i combine_bytes(__m512i sums) {
    return _mm512_add_epi32(sums, _mm512_slli_epi32(_mm512_shuffle_epi32(sums, 0x4e), 7));
}

SIMD_INLINE __m512i tile_energy(__m512i words, __m512i dc) {
    const __m512i rounded = _mm512_srli_epi32(_mm512_slli_epi32(_mm512_add_epi16(dc, _mm512_set1_epi16(2)), 16), 18);
    return _mm512_sub_epi32(combine_bytes(rounded_byte_sums(words)), rounded);
}

// Pack the two inputs and both four-row halves of a single tile into a ZMM.
SIMD_INLINE unsigned single8(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int hbd) {
    __m512i v[4];
    for (int i = 0; i < 4; i++) {
        const __m256i a = join128(load8(s + i * ss, hbd), load8(r + i * rs, hbd));
        const __m256i b = join128(load8(s + (i + 4) * ss, hbd), load8(r + (i + 4) * rs, hbd));
        v[i]            = join256(a, b);
    }
    transform4x8(v);
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < 4; i++) {
        const __m512i a = _mm512_abs_epi16(v[i]);
        const __m256i m = _mm256_max_epi16(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
        sum             = _mm256_add_epi16(sum, m);
    }
    const __m256i dc = _mm256_add_epi16(_mm512_castsi512_si256(v[0]), _mm512_extracti64x4_epi64(v[0], 1));
    const __m256i e  = _mm512_castsi512_si256(tile_energy(_mm512_castsi256_si512(sum), _mm512_castsi256_si512(dc)));
    return abs(_mm256_extract_epi32(e, 0) - _mm256_extract_epi32(e, 4));
}

SIMD_INLINE unsigned sum2(__m256i g) {
    return _mm256_extract_epi32(g, 0) + _mm256_extract_epi32(g, 4);
}

// Fill a YMM with two horizontal tiles when both fit the row. For a
// single tile column, take the second tile from the next eight-row band.
SIMD_INLINE __m256i load_pair(const uint8_t* p, ptrdiff_t stride, unsigned cols, int hbd) {
    return cols == 1 ? join128(load8(p, hbd), load8(p + 8 * stride, hbd)) : load16(p, hbd);
}

SIMD_INLINE __m512i load_quad(const uint8_t* p, ptrdiff_t stride, unsigned cols, int hbd) {
    if (cols == 4) {
        return load_row(p, hbd);
    }
    const ptrdiff_t second = (cols == 2 ? 8 : 16) * stride;
    return join256(load_pair(p, stride, cols, hbd), load_pair(p + second, stride, cols, hbd));
}

SIMD_INLINE unsigned two_gaps(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, unsigned cols, int hbd) {
    __m512i a[8];
    for (int i = 0; i < 8; i++) {
        a[i] = join256(load_pair(s + i * ss, ss, cols, hbd), load_pair(r + i * rs, rs, cols, hbd));
    }
    transform4x8(a);
    transform4x8(a + 4);
    const __m512i e = tile_energy(magnitude_sums(a, a + 4), _mm512_add_epi16(a[0], a[4]));
    return sum2(_mm256_abs_epi32(_mm256_sub_epi32(_mm512_castsi512_si256(e), _mm512_extracti64x4_epi64(e, 1))));
}

SIMD_INLINE __m512i four_gaps(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, unsigned cols, int hbd) {
    __m512i v[4][4];
    // Interleave source and reconstruction at every stage. The four sets
    // of butterflies are independent, including both halves of each tile.
    for (int i = 0; i < 4; i++) {
        v[0][i] = load_quad(s + i * ss, ss, cols, hbd);
        v[1][i] = load_quad(r + i * rs, rs, cols, hbd);
        v[2][i] = load_quad(s + (4 + i) * ss, ss, cols, hbd);
        v[3][i] = load_quad(r + (4 + i) * rs, rs, cols, hbd);
    }
    for (int i = 0; i < 4; i++) {
        butterfly4(v[i]);
    }
    for (int i = 0; i < 4; i++) {
        transpose4x8(v[i]);
    }
    for (int i = 0; i < 4; i++) {
        butterfly4(v[i]);
    }
    for (int i = 0; i < 4; i++) {
        last_columns(v[i]);
    }
    const __m512i da = _mm512_add_epi16(v[0][0], v[2][0]), db = _mm512_add_epi16(v[1][0], v[3][0]);
    const __m512i a = rounded_byte_sums(magnitude_sums(v[0], v[2]));
    const __m512i b = rounded_byte_sums(magnitude_sums(v[1], v[3]));
    // Each rounded DC fits a signed word. PMADDWD with (+1,-1) subtracts the two.
    const __m512i packed  = _mm512_unpacklo_epi16(da, db);
    const __m512i rounded = _mm512_srli_epi16(_mm512_add_epi16(packed, _mm512_set1_epi16(2)), 2);
    const __m512i dc      = _mm512_madd_epi16(rounded, _mm512_set1_epi32(-65535));
    return _mm512_abs_epi32(_mm512_sub_epi32(combine_bytes(_mm512_sub_epi32(a, b)), dc));
}

// Only the first dword in each 128-bit lane holds a tile gap.
SIMD_INLINE unsigned sum_four_tiles(__m512i gaps) {
    const __m256i halves = _mm256_add_epi32(_mm512_castsi512_si256(gaps), _mm512_extracti64x4_epi64(gaps, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));
}

// The 128-bit lanes hold independent tiles. The two tables select vertical
// or horizontal packing for a pair, and vertical or square packing for four.
typedef unsigned (*BatchCost)(const uint8_t*, ptrdiff_t, const uint8_t*, ptrdiff_t);

typedef struct {
    BatchCost single;
    BatchCost pair[2];
    BatchCost quad[2];
} BatchCosts;

// Each complete column of 32 pixels uses full-register loads. Keep the
// four tile sums in separate 64-bit lanes until the end of the rectangle.
SIMD_INLINE uint64_t wide_band(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, size_t across,
                               size_t down, int hbd) {
    __m512i sum = _mm512_setzero_si512();
    for (size_t y = 0; y < down; y++) {
        for (size_t x = 0; x < across; x++) {
            const size_t  dx   = x * (32 << hbd);
            const __m512i gaps = four_gaps(s + 8 * y * ss + dx, ss, r + 8 * y * rs + dx, rs, 4, hbd);
            sum                = _mm512_add_epi64(sum, _mm512_maskz_mov_epi32(0x1111, gaps));
        }
    }
    return _mm512_reduce_add_epi64(sum);
}

#define AC_TILE(name, expression)                                                                   \
    static NOINLINE unsigned name(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) { \
        return (expression);                                                                        \
    }

AC_TILE(single_lbd, single8(s, ss, r, rs, 0))
AC_TILE(pair_v_lbd, two_gaps(s, ss, r, rs, 1, 0))
AC_TILE(pair_h_lbd, two_gaps(s, ss, r, rs, 2, 0))
AC_TILE(quad_v_lbd, sum_four_tiles(four_gaps(s, ss, r, rs, 1, 0)))
AC_TILE(quad_sq_lbd, sum_four_tiles(four_gaps(s, ss, r, rs, 2, 0)))
static const BatchCosts costs_lbd = {single_lbd, {pair_v_lbd, pair_h_lbd}, {quad_v_lbd, quad_sq_lbd}};
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
AC_TILE(single_hbd, single8(s, ss, r, rs, 1))
AC_TILE(pair_v_hbd, two_gaps(s, ss, r, rs, 1, 1))
AC_TILE(pair_h_hbd, two_gaps(s, ss, r, rs, 2, 1))
AC_TILE(quad_v_hbd, sum_four_tiles(four_gaps(s, ss, r, rs, 1, 1)))
AC_TILE(quad_sq_hbd, sum_four_tiles(four_gaps(s, ss, r, rs, 2, 1)))
static const BatchCosts costs_hbd = {single_hbd, {pair_v_hbd, pair_h_hbd}, {quad_v_hbd, quad_sq_hbd}};
#endif

static NOINLINE uint64_t wide(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, size_t across,
                              size_t down, int hbd) {
    if (hbd) {
        return wide_band(s, ss, r, rs, across, down, 1);
    }
    return wide_band(s, ss, r, rs, across, down, 0);
}

static NOINLINE uint64_t columns(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, size_t ny,
                                 unsigned cols, const BatchCosts* cost) {
    const unsigned  rows  = 4 / cols;
    const BatchCost quad  = cost->quad[cols - 1];
    size_t          y     = 0;
    uint64_t        total = 0;
    for (; y + rows <= ny; y += rows) {
        total += quad(s + 8 * y * ss, ss, r + 8 * y * rs, rs);
    }
    if ((ny - y) * cols >= 2) {
        total += cost->pair[cols - 1](s + 8 * y * ss, ss, r + 8 * y * rs, rs);
        y += 2 / cols;
    }
    if (y < ny) {
        total += cost->single(s + 8 * y * ss, ss, r + 8 * y * rs, rs);
    }
    return total;
}

SIMD_INLINE uint64_t distortion(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, unsigned w, unsigned h,
                                int hbd, const BatchCosts* cost) {
    const size_t nx = (w + 7ULL) / 8, ny = (h + 7ULL) / 8;
    if (w == 8 && h == 8) {
        return (uint64_t)cost->single(s, ss, r, rs) << (2 * hbd);
    }
    if (nx <= 2) {
        const size_t count = nx * ny;
        if (count == 2) {
            return (uint64_t)cost->pair[nx - 1](s, ss, r, rs) << (2 * hbd);
        }
        if (count == 4) {
            return (uint64_t)cost->quad[nx - 1](s, ss, r, rs) << (2 * hbd);
        }
    }
    uint64_t total = nx >= 4 ? wide(s, ss, r, rs, nx / 4, ny, hbd) : 0;
    if (nx & 2) {
        const size_t dx = (nx & ~(size_t)3) * (8 << hbd);
        total += columns(s + dx, ss, r + dx, rs, ny, 2, cost);
    }
    if (nx & 1) {
        const size_t dx = (nx - 1) * (8 << hbd);
        total += columns(s + dx, ss, r + dx, rs, ny, 1, cost);
    }
    return total << (2 * hbd);
}

uint64_t svt_psy_distortion_avx512(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs, uint32_t w,
                                   uint32_t h) {
    if (w < 8 || h < 8) {
        return svt_psy_distortion_avx2(s, ss, r, rs, w, h);
    }
    return distortion(s, ss, r, rs, w, h, 0, &costs_lbd);
}

int svt_av1_hadamard_satd_4x4_avx512(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return svt_av1_hadamard_satd_4x4_avx2(s, ss, r, rs);
}

int svt_av1_hadamard_satd_8x8_avx512(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return svt_av1_hadamard_satd_8x8_avx2(s, ss, r, rs);
}

int svt_av1_hadamard_satd_16x16_avx512(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return satd16(s, ss, r, rs, 0);
}

int svt_av1_hadamard_satd_32x32_avx512(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return satd32(s, ss, r, rs, 0);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

uint64_t svt_psy_distortion_hbd_avx512(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs, uint32_t w,
                                       uint32_t h) {
    if (w < 8 || h < 8) {
        return svt_psy_distortion_hbd_avx2(s, ss, r, rs, w, h);
    }
    return distortion((const uint8_t*)s, 2 * (ptrdiff_t)ss, (const uint8_t*)r, 2 * (ptrdiff_t)rs, w, h, 1, &costs_hbd);
}

int svt_av1_highbd_hadamard_satd_4x4_avx512(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return svt_av1_highbd_hadamard_satd_4x4_avx2(s, ss, r, rs);
}

int svt_av1_highbd_hadamard_satd_8x8_avx512(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    // Each lane holds an eight-pixel row. Pair rows four apart to keep
    // both halves of the vertical transform in the same registers.
    __m256i row[4];
    for (int i = 0; i < 4; i++) {
        const __m256i src  = join128(_mm_loadu_si128((const __m128i*)(s + i * ss)),
                                    _mm_loadu_si128((const __m128i*)(s + (i + 4) * ss)));
        const __m256i pred = join128(_mm_loadu_si128((const __m128i*)(r + i * rs)),
                                     _mm_loadu_si128((const __m128i*)(r + (i + 4) * rs)));
        row[i]             = _mm256_sub_epi16(src, pred);
    }
    const __m512i even = join256(row[0], row[2]), odd = join256(row[1], row[3]);
    const __m512i p = _mm512_add_epi16(even, odd), q = _mm512_subs_epi16(even, odd);
    __m512i       lo = _mm512_shuffle_i64x2(p, q, 0x44), hi = _mm512_shuffle_i64x2(p, q, 0xee);
    __m512i       a = _mm512_add_epi16(lo, hi), b = _mm512_sub_epi16(lo, hi);

    // Merge the two four-column groups, then the two-column groups.
    lo = _mm512_unpacklo_epi64(a, b);
    hi = _mm512_unpackhi_epi64(a, b);
    a  = _mm512_add_epi16(lo, hi);
    b  = _mm512_subs_epi16(lo, hi);

    const __m512 x = _mm512_castsi512_ps(a), y = _mm512_castsi512_ps(b);
    lo = _mm512_castps_si512(_mm512_shuffle_ps(x, y, 0x88));
    hi = _mm512_castps_si512(_mm512_shuffle_ps(x, y, 0xdd));
    a  = _mm512_add_epi16(lo, hi);
    b  = _mm512_subs_epi16(lo, hi);

    // Join the vertical halves. The last horizontal butterfly contributes
    // twice the larger magnitude of each pair of adjacent words.
    lo = tile_columns(a, b, 0);
    hi = tile_columns(a, b, 1);
    a  = _mm512_abs_epi16(_mm512_add_epi16(lo, hi));
    b  = _mm512_abs_epi16(_mm512_sub_epi16(lo, hi));

    static const uint16_t even_words[32] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                            32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62};

    const __m512i indices = _mm512_loadu_si512(even_words);
    lo                    = _mm512_permutex2var_epi16(a, indices, b);
    hi                    = _mm512_permutex2var_epi16(a, _mm512_add_epi16(indices, _mm512_set1_epi16(1)), b);
    return _mm512_reduce_add_epi32(_mm512_madd_epi16(_mm512_max_epi16(lo, hi), _mm512_set1_epi16(2)));
}

int svt_av1_highbd_hadamard_satd_16x16_avx512(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return satd16((const uint8_t*)s, 2 * ss, (const uint8_t*)r, 2 * rs, 1);
}

int svt_av1_highbd_hadamard_satd_32x32_avx512(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return satd32((const uint8_t*)s, 2 * ss, (const uint8_t*)r, 2 * rs, 1);
}
#endif
