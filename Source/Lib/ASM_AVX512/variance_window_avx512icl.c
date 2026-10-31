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

#include <immintrin.h>
#include "aom_dsp_rtcd.h"
#include "definitions.h"

// With x' = x ^ 0x80 read as int8 (x - 128): s * r = s' * r + 128 * r and r * r = r' * r + 128 * r, so vpdpbusd gives
// every product sum exactly. sum(s) and sum(s * s) are the same for all positions.
static INLINE uint32_t window_var(int64_t ss, int64_t ss2, int64_t sr, int64_t c, int64_t r2, int bits) {
    const int64_t  ssr = c + 128 * sr;
    const int64_t  srr = r2 + 128 * sr;
    const uint32_t sse = (uint32_t)(ss2 - 2 * ssr + srr);
    const int64_t  sum = ss - sr;
    return sse - (uint32_t)((sum * sum) >> bits);
}

// [sum(a), sum(b), sum(c), sum(c)]
static INLINE __m128i hsum3_epi32(__m512i a, __m512i b, __m512i c) {
    const __m256i a8 = _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
    const __m256i b8 = _mm256_add_epi32(_mm512_castsi512_si256(b), _mm512_extracti64x4_epi64(b, 1));
    const __m256i c8 = _mm256_add_epi32(_mm512_castsi512_si256(c), _mm512_extracti64x4_epi64(c, 1));
    const __m128i a4 = _mm_add_epi32(_mm256_castsi256_si128(a8), _mm256_extracti128_si256(a8, 1));
    const __m128i b4 = _mm_add_epi32(_mm256_castsi256_si128(b8), _mm256_extracti128_si256(b8, 1));
    const __m128i c4 = _mm_add_epi32(_mm256_castsi256_si128(c8), _mm256_extracti128_si256(c8, 1));
    return _mm_hadd_epi32(_mm_hadd_epi32(a4, b4), _mm_hadd_epi32(c4, c4));
}

// 64 bytes of a bw-wide block: 1 row (bw 64), 2 rows (bw 32) or 4 rows (bw 16)
static INLINE __m512i load_rows64(const uint8_t* p, int stride, int bw) {
    if (bw == 64) {
        return _mm512_loadu_si512((const __m512i*)p);
    }
    if (bw == 32) {
        return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)p)),
                                  _mm256_loadu_si256((const __m256i*)(p + stride)),
                                  1);
    }
    __m512i v = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*)p));
    v         = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i*)(p + stride)), 1);
    v         = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i*)(p + 2 * stride)), 2);
    return _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i*)(p + 3 * stride)), 3);
}

// sx = src ^ 0x80 in 64-byte steps; returns sum(s) and sum(s * s)
static INLINE void prep_src(const uint8_t* src, int src_stride, int bw, int bh, uint8_t* sx, int64_t* ss,
                            int64_t* ss2) {
    const __m512i k80   = _mm512_set1_epi8((char)0x80);
    const __m512i zero  = _mm512_setzero_si512();
    const int     rows  = 64 / bw;
    const int     steps = bh / rows;
    // sad partial sums stay below 2^32, so their 64-bit lanes can be summed as 32-bit lanes
    __m512i s1 = zero, s2 = zero;
    for (int i = 0; i < steps; i++) {
        const __m512i v  = load_rows64(src + i * rows * src_stride, src_stride, bw);
        const __m512i vx = _mm512_xor_si512(v, k80);
        _mm512_storeu_si512((__m512i*)(sx + i * 64), vx);
        s1 = _mm512_add_epi64(s1, _mm512_sad_epu8(v, zero));
        s2 = _mm512_dpbusd_epi32(s2, v, vx);
    }
    const __m128i st = hsum3_epi32(s1, s2, zero);
    *ss              = _mm_cvtsi128_si32(st);
    *ss2             = _mm_extract_epi32(st, 1) + 128 * *ss;
}

// Variance of the prepared source block against the bw x bh block at r
static INLINE uint32_t var_vs_prepped(const uint8_t* sx, int64_t ss, int64_t ss2, const uint8_t* r, int r_stride,
                                      int bw, int bh) {
    const __m512i k80   = _mm512_set1_epi8((char)0x80);
    const __m512i zero  = _mm512_setzero_si512();
    const int     rows  = 64 / bw;
    const int     steps = bh / rows;
    __m512i       c = zero, r2 = zero, sr = zero;
    for (int i = 0; i < steps; i++, r += rows * r_stride) {
        const __m512i rv = load_rows64(r, r_stride, bw);
        c                = _mm512_dpbusd_epi32(c, rv, _mm512_loadu_si512((const __m512i*)(sx + i * 64)));
        r2               = _mm512_dpbusd_epi32(r2, rv, _mm512_xor_si512(rv, k80));
        sr               = _mm512_add_epi64(sr, _mm512_sad_epu8(rv, zero));
    }
    const __m128i t = hsum3_epi32(c, r2, sr);
    return window_var(
        ss, ss2, _mm_extract_epi32(t, 2), _mm_cvtsi128_si32(t), _mm_extract_epi32(t, 1), svt_log2f(bw * bh));
}

// Variances of the bw x bh source block against the nx x ny full-pel positions whose top-left starts at ref;
// var[x * ny + y]. bw is 16, 32 or 64, bh at most 64 and bw * bh a multiple of 64.
void svt_aom_variance_window_avx512icl(const uint8_t* src, int src_stride, const uint8_t* ref, int ref_stride, int bw,
                                       int bh, int nx, int ny, uint32_t* var) {
    DECLARE_ALIGNED(64, uint8_t, sx[64 * 64]);
    int64_t ss, ss2;
    prep_src(src, src_stride, bw, bh, sx, &ss, &ss2);
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            var[x * ny + y] = var_vs_prepped(sx, ss, ss2, ref + y * ref_stride + x, ref_stride, bw, bh);
        }
    }
}

// bw is 16, 32 or 64, bh at most 64 and bw * bh a multiple of 64
void svt_aom_src_var_prep_avx512icl(SrcVarCtx* c, const uint8_t* src, int stride, int bw, int bh) {
    c->src    = src;
    c->stride = stride;
    c->bw     = bw;
    c->bh     = bh;
    prep_src(src, stride, bw, bh, c->sx, &c->ss, &c->ss2);
}

uint32_t svt_aom_src_var_avx512icl(const SrcVarCtx* c, const uint8_t* pred, int pred_stride) {
    return var_vs_prepped(c->sx, c->ss, c->ss2, pred, pred_stride, c->bw, c->bh);
}
