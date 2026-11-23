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

#include "hadamard_avx.h"

SIMD_INLINE __m128i load4x2(const uint8_t* p, ptrdiff_t stride, int hbd) {
    if (hbd) {
        return _mm_unpacklo_epi64(xx_loadl_64(p), xx_loadl_64(p + stride));
    }
    return _mm_cvtepu8_epi16(_mm_unpacklo_epi32(xx_loadl_32(p), xx_loadl_32(p + stride)));
}

// Even and odd rows occupy separate registers, one 4x4 tile per 128-bit
// lane. In each dimension only the adjacent-pair stage rounds down.
SIMD_INLINE __m256i cost4(__m256i even, __m256i odd, int ac) {
    const __m256i p  = _mm256_srai_epi16(_mm256_add_epi16(even, odd), 1),
                  q  = _mm256_srai_epi16(_mm256_sub_epi16(even, odd), 1);
    const __m256i lo = _mm256_unpacklo_epi64(p, q), hi = _mm256_unpackhi_epi64(p, q);
    const __m256i order = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7, 14, 15));
    const __m256i a = _mm256_shuffle_epi8(_mm256_add_epi16(lo, hi), order),
                  b = _mm256_shuffle_epi8(_mm256_sub_epi16(lo, hi), order);
    const __m256i x = _mm256_unpacklo_epi64(a, b), y = _mm256_unpackhi_epi64(a, b);
    const __m256i u = _mm256_srai_epi16(_mm256_add_epi16(x, y), 1), v = _mm256_srai_epi16(_mm256_sub_epi16(x, y), 1);
    const __m256i l   = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(u), _mm256_castsi256_ps(v), 0x88));
    const __m256i r   = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(u), _mm256_castsi256_ps(v), 0xdd));
    const __m256i m   = _mm256_max_epi16(_mm256_abs_epi16(l), _mm256_abs_epi16(r));
    const __m256i sum = _mm256_madd_epi16(m, _mm256_set1_epi16(ac ? 4 : 2));
    return ac ? _mm256_sub_epi32(sum, _mm256_blend_epi16(_mm256_setzero_si256(), _mm256_add_epi16(l, r), 1)) : sum;
}

SIMD_INLINE int tile4(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int hbd, int ac) {
    const __m128i se = load4x2(s, 2 * ss, hbd), so = load4x2(s + ss, 2 * ss, hbd);
    const __m128i pe = load4x2(r, 2 * rs, hbd), po = load4x2(r + rs, 2 * rs, hbd);
    if (ac) {
        const __m256i e = cost4(join128(se, pe), join128(so, po), 1);
        return abs(reduce128(_mm_sub_epi32(_mm256_castsi256_si128(e), _mm256_extracti128_si256(e, 1))));
    }
    const __m256i e = cost4(
        _mm256_castsi128_si256(_mm_sub_epi16(se, pe)), _mm256_castsi128_si256(_mm_sub_epi16(so, po)), 0);
    return reduce128(_mm256_castsi256_si128(e));
}

SIMD_INLINE int satd8(const uint8_t* src, ptrdiff_t ss, const uint8_t* pred, ptrdiff_t ps, int hbd) {
    __m256i r[4];
    for (int i = 0; i < 4; i++) {
        const __m256i s = join128(load8(src + i * ss, hbd), load8(src + (i + 4) * ss, hbd));
        const __m256i p = join128(load8(pred + i * ps, hbd), load8(pred + (i + 4) * ps, hbd));
        r[i]            = _mm256_sub_epi16(s, p);
    }
    transform4x8(r);
    __m128i sum = _mm_setzero_si128();
    for (int i = 0; i < 4; i++) {
        const __m256i a = _mm256_abs_epi16(r[i]);
        const __m128i m = _mm_max_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        sum             = _mm_add_epi32(sum, _mm_madd_epi16(m, _mm_set1_epi16(1)));
    }
    return 2 * reduce128(sum);
}

// Sum eight unsigned words with a signed multiply-add, then remove the bias.
SIMD_INLINE __m256i energy8(__m256i top[4], __m256i bottom[4]) {
    transform4x8(top);
    transform4x8(bottom);
    const __m256i dc      = _mm256_add_epi16(top[0], bottom[0]);
    const __m256i words   = magnitude_sums(top, bottom);
    __m256i       sum     = _mm256_madd_epi16(_mm256_xor_si256(words, _mm256_set1_epi16(-32768)), _mm256_set1_epi16(1));
    sum                   = _mm256_add_epi32(sum, _mm256_shuffle_epi32(sum, 0x4e));
    sum                   = _mm256_add_epi32(sum, _mm256_shuffle_epi32(sum, 0xb1));
    const __m256i rounded = _mm256_srli_epi16(_mm256_add_epi16(dc, _mm256_set1_epi16(2)), 2);
    return _mm256_sub_epi32(_mm256_srli_epi32(_mm256_add_epi32(sum, _mm256_set1_epi32(8 * 32768 + 1)), 1),
                            _mm256_srli_epi32(_mm256_slli_epi32(rounded, 16), 16));
}

SIMD_INLINE int gap8(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int hbd) {
    __m256i top[4], bottom[4];
    for (int i = 0; i < 4; i++) {
        top[i]    = join128(load8(s + i * ss, hbd), load8(r + i * rs, hbd));
        bottom[i] = join128(load8(s + (i + 4) * ss, hbd), load8(r + (i + 4) * rs, hbd));
    }
    const __m256i e = energy8(top, bottom);
    return abs(_mm256_extract_epi32(e, 0) - _mm256_extract_epi32(e, 4));
}

SIMD_INLINE __m256i strip_energy(const uint8_t* p, ptrdiff_t stride, int hbd) {
    __m256i top[4], bot[4];
    for (int i = 0; i < 4; i++) {
        top[i] = load_row(p + i * stride, hbd);
        bot[i] = load_row(p + (i + 4) * stride, hbd);
    }
    return energy8(top, bot);
}

SIMD_INLINE unsigned strip_gap(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, int hbd) {
    const __m256i gap = _mm256_abs_epi32(_mm256_sub_epi32(strip_energy(s, ss, hbd), strip_energy(r, rs, hbd)));

    return _mm256_extract_epi32(gap, 0) + _mm256_extract_epi32(gap, 4);
}

static NOINLINE uint64_t tiled_distortion(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, unsigned w,
                                          unsigned h, int hbd) {
    uint64_t gap = 0;
    if (w >= 8 && h >= 8) {
        for (size_t y = 0; y < h; y += 8) {
            size_t x = 0;
            for (; x + 8 < w; x += 16) {
                gap += strip_gap(s + y * ss + (x << hbd), ss, r + y * rs + (x << hbd), rs, hbd);
            }
            if (x < w) {
                gap += gap8(s + y * ss + (x << hbd), ss, r + y * rs + (x << hbd), rs, hbd);
            }
        }
        return gap << (2 * hbd);
    }
    if (!w || !h) {
        return 0;
    }
    for (size_t y = 0; y < h; y += 4) {
        for (size_t x = 0; x < w; x += 4) {
            const uint8_t *src = s + y * ss + (x << hbd), *pred = r + y * rs + (x << hbd);
            gap += tile4(src, ss, pred, rs, hbd, 1);
        }
    }
    return gap << (2 * hbd);
}

SIMD_INLINE uint64_t distortion(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs, unsigned w, unsigned h,
                                int hbd) {
    if (w == 4 && h == 4) {
        return (uint64_t)tile4(s, ss, r, rs, hbd, 1) << (2 * hbd);
    }
    if (w == 8 && h == 8) {
        return (uint64_t)gap8(s, ss, r, rs, hbd) << (2 * hbd);
    }
    return tiled_distortion(s, ss, r, rs, w, h, hbd);
}

uint64_t svt_psy_distortion_avx2(const uint8_t* s, uint32_t ss, const uint8_t* r, uint32_t rs, uint32_t w, uint32_t h) {
    return distortion(s, ss, r, rs, w, h, 0);
}

int svt_av1_hadamard_satd_4x4_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return tile4(s, ss, r, rs, 0, 0);
}

int svt_av1_hadamard_satd_8x8_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return satd8(s, ss, r, rs, 0);
}

int svt_av1_hadamard_satd_16x16_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return satd16(s, ss, r, rs, 0);
}

int svt_av1_hadamard_satd_32x32_avx2(const uint8_t* s, ptrdiff_t ss, const uint8_t* r, ptrdiff_t rs) {
    return satd32(s, ss, r, rs, 0);
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

uint64_t svt_psy_distortion_hbd_avx2(const uint16_t* s, uint32_t ss, const uint16_t* r, uint32_t rs, uint32_t w,
                                     uint32_t h) {
    return distortion((const uint8_t*)s, 2 * (ptrdiff_t)ss, (const uint8_t*)r, 2 * (ptrdiff_t)rs, w, h, 1);
}

int svt_av1_highbd_hadamard_satd_4x4_avx2(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return tile4((const uint8_t*)s, 2 * ss, (const uint8_t*)r, 2 * rs, 1, 0);
}

int svt_av1_highbd_hadamard_satd_8x8_avx2(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return satd8((const uint8_t*)s, 2 * ss, (const uint8_t*)r, 2 * rs, 1);
}

int svt_av1_highbd_hadamard_satd_16x16_avx2(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return satd16((const uint8_t*)s, 2 * ss, (const uint8_t*)r, 2 * rs, 1);
}

int svt_av1_highbd_hadamard_satd_32x32_avx2(const uint16_t* s, ptrdiff_t ss, const uint16_t* r, ptrdiff_t rs) {
    return satd32((const uint8_t*)s, 2 * ss, (const uint8_t*)r, 2 * rs, 1);
}
#endif
