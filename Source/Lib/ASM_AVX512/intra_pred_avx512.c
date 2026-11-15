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
#include "definitions.h"
#include "common_dsp_rtcd.h"
#include <assert.h>
#include <string.h>

// Zone-2 rows: lanes with base_x + c < -1 predict from left[], the rest from above[].
// 32 16-bit lanes per step: one row of a 32-wide chunk, or two rows of a 16-wide block.
static INLINE __m512i z2_interp_avx512(const __m512i a0, const __m512i a1, const __m512i shift) {
    const __m512i a32 = _mm512_add_epi16(_mm512_slli_epi16(a0, 5), _mm512_set1_epi16(16));
    return _mm512_srli_epi16(_mm512_add_epi16(a32, _mm512_mullo_epi16(_mm512_sub_epi16(a1, a0), shift)), 5);
}

// Left-derived samples of 32 lanes. row6: (row << 6) per lane, colp1: (col + 1) per lane.
// For lanes that use left[], base_y is in [-1, H - 2], so idx = base_y + 1 addresses the 64-entry tables.
static INLINE __m512i z2_left_avx512(const __m512i row6, const __m512i colp1, const __m512i dy, const __m512i l0_lo,
                                     const __m512i l0_hi, const __m512i l1_lo, const __m512i l1_hi) {
    const __m512i y_c   = _mm512_sub_epi16(row6, _mm512_mullo_epi16(colp1, dy));
    const __m512i idx   = _mm512_add_epi16(_mm512_srai_epi16(y_c, 6), _mm512_set1_epi16(1));
    const __m512i shift = _mm512_srli_epi16(_mm512_and_si512(y_c, _mm512_set1_epi16(0x3f)), 1);
    const __m512i a0    = _mm512_permutex2var_epi16(l0_lo, idx, l0_hi);
    const __m512i a1    = _mm512_permutex2var_epi16(l1_lo, idx, l1_hi);
    return z2_interp_avx512(a0, a1, shift);
}

static INLINE uint32_t z2_left_lanes_mask(int32_t n) {
    return n <= 0 ? 0 : n >= 32 ? 0xffffffffu : (1u << n) - 1;
}

static void dr_prediction_z2_hxw_avx512(int32_t H, int32_t W, uint8_t* dst, ptrdiff_t stride, const uint8_t* above,
                                        const uint8_t* left, int32_t dx, int32_t dy) {
    // l0[i] = left[i - 1], l1[i] = left[i], i < H
    const __mmask64 hmask = H >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << H) - 1);
    const __m512i   l0b   = _mm512_maskz_loadu_epi8(hmask, left - 1);
    const __m512i   l1b   = _mm512_maskz_loadu_epi8(hmask, left);
    const __m512i   l0_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(l0b));
    const __m512i   l0_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(l0b, 1));
    const __m512i   l1_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(l1b));
    const __m512i   l1_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(l1b, 1));
    const __m512i   dy512 = _mm512_set1_epi16(dy);
    const __m512i   lane  = _mm512_cvtepu8_epi16(_mm256_setr_epi8(1,
                                                                  2,
                                                                  3,
                                                                  4,
                                                                  5,
                                                                  6,
                                                                  7,
                                                                  8,
                                                                  9,
                                                                  10,
                                                                  11,
                                                                  12,
                                                                  13,
                                                                  14,
                                                                  15,
                                                                  16,
                                                                  17,
                                                                  18,
                                                                  19,
                                                                  20,
                                                                  21,
                                                                  22,
                                                                  23,
                                                                  24,
                                                                  25,
                                                                  26,
                                                                  27,
                                                                  28,
                                                                  29,
                                                                  30,
                                                                  31,
                                                                  32));

    if (W == 16) {
        const __m512i colp1  = _mm512_and_si512(_mm512_sub_epi16(lane, _mm512_set1_epi16(1)), _mm512_set1_epi16(15));
        const __m512i colp1_ = _mm512_add_epi16(colp1, _mm512_set1_epi16(1));
        const __m512i row_hi = _mm512_mask_set1_epi16(_mm512_setzero_si512(), 0xffff0000, 64);
        for (int32_t r = 0; r < H; r += 2) {
            const int32_t x0 = -(r + 1) * dx, x1 = -(r + 2) * dx;
            const int32_t b0 = x0 >> 6, b1 = x1 >> 6;
            // lanes of each row with base_x + c < -1
            const uint32_t  nl0 = z2_left_lanes_mask(-1 - b0);
            const uint32_t  nl1 = z2_left_lanes_mask(-1 - b1);
            const __mmask32 lm  = (nl0 & 0xffff) | ((nl1 & 0xffff) << 16);
            __m512i         res;
            if (lm != 0xffffffffu) {
                const __m128i a0_0 = _mm_maskz_loadu_epi8((__mmask16)~nl0, above + b0);
                const __m128i a1_0 = _mm_maskz_loadu_epi8((__mmask16)~nl0, above + b0 + 1);
                const __m128i a0_1 = _mm_maskz_loadu_epi8((__mmask16)~nl1, above + b1);
                const __m128i a1_1 = _mm_maskz_loadu_epi8((__mmask16)~nl1, above + b1 + 1);
                const __m512i a0 = _mm512_cvtepu8_epi16(_mm256_inserti128_si256(_mm256_castsi128_si256(a0_0), a0_1, 1));
                const __m512i a1 = _mm512_cvtepu8_epi16(_mm256_inserti128_si256(_mm256_castsi128_si256(a1_0), a1_1, 1));
                const __m512i sh = _mm512_mask_set1_epi16(
                    _mm512_set1_epi16((x0 & 0x3f) >> 1), 0xffff0000, (x1 & 0x3f) >> 1);
                res = z2_interp_avx512(a0, a1, sh);
            } else {
                res = _mm512_setzero_si512();
            }
            if (lm) {
                const __m512i row6 = _mm512_add_epi16(_mm512_set1_epi16(r << 6), row_hi);
                res = _mm512_mask_blend_epi16(lm, res, z2_left_avx512(row6, colp1_, dy512, l0_lo, l0_hi, l1_lo, l1_hi));
            }
            const __m256i out = _mm512_cvtepi16_epi8(res);
            _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(out));
            _mm_storeu_si128((__m128i*)(dst + stride), _mm256_extracti128_si256(out, 1));
            dst += 2 * stride;
        }
        return;
    }

    for (int32_t r = 0; r < H; r++) {
        const int32_t x      = -(r + 1) * dx;
        const int32_t base_x = x >> 6;
        const __m512i sh     = _mm512_set1_epi16((x & 0x3f) >> 1);
        const __m512i row6   = _mm512_set1_epi16(r << 6);
        for (int32_t j = 0; j < W; j += 32) {
            const uint32_t nl = z2_left_lanes_mask(-1 - base_x - j);
            __m512i        res;
            if (nl != 0xffffffffu) {
                const __m512i a0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(~nl, above + base_x + j));
                const __m512i a1 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(~nl, above + base_x + j + 1));
                res              = z2_interp_avx512(a0, a1, sh);
            } else {
                res = _mm512_setzero_si512();
            }
            if (nl) {
                const __m512i colp1 = _mm512_add_epi16(lane, _mm512_set1_epi16(j));
                res = _mm512_mask_blend_epi16(nl, res, z2_left_avx512(row6, colp1, dy512, l0_lo, l0_hi, l1_lo, l1_hi));
            }
            _mm256_storeu_si256((__m256i*)(dst + j), _mm512_cvtepi16_epi8(res));
        }
        dst += stride;
    }
}

// W in {4, 8}: 32 / W rows per step; above[] and left[] both gathered from 64-entry tables.
static void dr_prediction_z2_small_avx512(int32_t H, int32_t W, uint8_t* dst, ptrdiff_t stride, const uint8_t* above,
                                          const uint8_t* left, int32_t dx, int32_t dy) {
    const __mmask64 hmask = H >= 64 ? ~(__mmask64)0 : (((__mmask64)1 << H) - 1);
    const __m512i   l0b   = _mm512_maskz_loadu_epi8(hmask, left - 1);
    const __m512i   l1b   = _mm512_maskz_loadu_epi8(hmask, left);
    const __m512i   l0_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(l0b));
    const __m512i   l0_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(l0b, 1));
    const __m512i   l1_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(l1b));
    const __m512i   l1_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(l1b, 1));
    // a0[i] = above[i - 1], a1[i] = above[i], i <= W
    const __m256i a0b   = _mm256_maskz_loadu_epi8((1u << (W + 1)) - 1, above - 1);
    const __m256i a1b   = _mm256_maskz_loadu_epi8((1u << W) - 1, above);
    const __m512i a0t   = _mm512_cvtepu8_epi16(a0b);
    const __m512i a1t   = _mm512_cvtepu8_epi16(a1b);
    const __m512i dy512 = _mm512_set1_epi16(dy);
    const int     log2w = W == 4 ? 2 : W == 8 ? 3 : 4;
    const int     rows  = 32 >> log2w;
    const __m512i lane  = _mm512_cvtepu8_epi16(_mm256_setr_epi8(0,
                                                                1,
                                                                2,
                                                                3,
                                                                4,
                                                                5,
                                                                6,
                                                                7,
                                                                8,
                                                                9,
                                                                10,
                                                                11,
                                                                12,
                                                                13,
                                                                14,
                                                                15,
                                                                16,
                                                                17,
                                                                18,
                                                                19,
                                                                20,
                                                                21,
                                                                22,
                                                                23,
                                                                24,
                                                                25,
                                                                26,
                                                                27,
                                                                28,
                                                                29,
                                                                30,
                                                                31));
    const __m512i col   = _mm512_and_si512(lane, _mm512_set1_epi16(W - 1));
    const __m512i colp1 = _mm512_add_epi16(col, _mm512_set1_epi16(1));
    const __m512i row6o = _mm512_slli_epi16(_mm512_srli_epi16(lane, log2w), 6);
    for (int32_t r = 0; r < H; r += rows) {
        __m512i bx = _mm512_setzero_si512(), sh = _mm512_setzero_si512();
        for (int i = 0; i < rows; i++) {
            const int32_t   x = -(r + i + 1) * dx;
            const __mmask32 m = (__mmask32)((((uint64_t)1 << W) - 1) << (i * W));
            bx                = _mm512_mask_set1_epi16(bx, m, (x >> 6) + 1);
            sh                = _mm512_mask_set1_epi16(sh, m, (x & 0x3f) >> 1);
        }
        // idx = base_x + c + 1 < 0: the lane predicts from left[]
        const __m512i   idx = _mm512_add_epi16(bx, col);
        const __mmask32 lm  = _mm512_cmplt_epi16_mask(idx, _mm512_setzero_si512());
        __m512i res = z2_interp_avx512(_mm512_permutexvar_epi16(idx, a0t), _mm512_permutexvar_epi16(idx, a1t), sh);
        if (lm) {
            const __m512i row6 = _mm512_add_epi16(_mm512_set1_epi16(r << 6), row6o);
            res = _mm512_mask_blend_epi16(lm, res, z2_left_avx512(row6, colp1, dy512, l0_lo, l0_hi, l1_lo, l1_hi));
        }
        DECLARE_ALIGNED(32, uint8_t, buf[32]);
        _mm256_store_si256((__m256i*)buf, _mm512_cvtepi16_epi8(res));
        const int n = H - r < rows ? H - r : rows;
        if (W == 4) {
            for (int i = 0; i < n; i++) {
                memcpy(dst + (r + i) * stride, buf + i * 4, 4);
            }
        } else if (W == 8) {
            for (int i = 0; i < n; i++) {
                memcpy(dst + (r + i) * stride, buf + i * 8, 8);
            }
        } else {
            for (int i = 0; i < n; i++) {
                memcpy(dst + (r + i) * stride, buf + i * 16, 16);
            }
        }
    }
}

// Directional prediction, zone 2: 90 < angle < 180
void svt_av1_dr_prediction_z2_avx512(uint8_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t* above,
                                     const uint8_t* left, int32_t upsample_above, int32_t upsample_left, int32_t dx,
                                     int32_t dy) {
    assert(dx > 0);
    assert(dy > 0);
    if (bw < 16) {
        if (upsample_above || upsample_left) {
            svt_av1_dr_prediction_z2_avx2(dst, stride, bw, bh, above, left, upsample_above, upsample_left, dx, dy);
        } else {
            dr_prediction_z2_small_avx512(bh, bw, dst, stride, above, left, dx, dy);
        }
        return;
    }
    // no edge upsampling for blocks this wide (av1_use_intra_edge_upsample)
    assert(!upsample_above && !upsample_left);
    dr_prediction_z2_hxw_avx512(bh, bw, dst, stride, above, left, dx, dy);
}
