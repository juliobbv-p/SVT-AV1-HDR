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

/* SSE4.1 tier for hadamard_16x16 / hadamard_32x32 (lbd), which dispatched
 * C -> AVX2 with no SSE tier. Each is 4 sub-block transforms followed by a
 * cross-block butterfly; the sub-blocks reuse the existing SSE 8x8/16x16 and
 * the butterfly (int32, arithmetic >> to match the C's signed shift) is
 * vectorized 4 coeffs at a time. Bit-identical to the C reference.
 * (highbd_hadamard_8x8 is a full transform and is handled separately.) */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>

void svt_aom_hadamard_8x8_sse2(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff);

/* 8-point Walsh-Hadamard butterfly with SVT's output permutation, int16 lanes. */
#define WHT8_16(i0, i1, i2, i3, i4, i5, i6, i7, o0, o1, o2, o3, o4, o5, o6, o7)                    \
    do {                                                                                          \
        __m128i b0 = _mm_add_epi16(i0, i1), b1 = _mm_sub_epi16(i0, i1);                            \
        __m128i b2 = _mm_add_epi16(i2, i3), b3 = _mm_sub_epi16(i2, i3);                            \
        __m128i b4 = _mm_add_epi16(i4, i5), b5 = _mm_sub_epi16(i4, i5);                            \
        __m128i b6 = _mm_add_epi16(i6, i7), b7 = _mm_sub_epi16(i6, i7);                            \
        __m128i c0 = _mm_add_epi16(b0, b2), c1 = _mm_add_epi16(b1, b3);                            \
        __m128i c2 = _mm_sub_epi16(b0, b2), c3 = _mm_sub_epi16(b1, b3);                            \
        __m128i c4 = _mm_add_epi16(b4, b6), c5 = _mm_add_epi16(b5, b7);                            \
        __m128i c6 = _mm_sub_epi16(b4, b6), c7 = _mm_sub_epi16(b5, b7);                            \
        o0 = _mm_add_epi16(c0, c4); o7 = _mm_add_epi16(c1, c5); o3 = _mm_add_epi16(c2, c6);        \
        o4 = _mm_add_epi16(c3, c7); o2 = _mm_sub_epi16(c0, c4); o6 = _mm_sub_epi16(c1, c5);        \
        o1 = _mm_sub_epi16(c2, c6); o5 = _mm_sub_epi16(c3, c7);                                    \
    } while (0)

void svt_aom_highbd_hadamard_8x8_sse4_1(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    // load 8 rows
    __m128i r0 = _mm_loadu_si128((const __m128i*)(src_diff + 0 * src_stride));
    __m128i r1 = _mm_loadu_si128((const __m128i*)(src_diff + 1 * src_stride));
    __m128i r2 = _mm_loadu_si128((const __m128i*)(src_diff + 2 * src_stride));
    __m128i r3 = _mm_loadu_si128((const __m128i*)(src_diff + 3 * src_stride));
    __m128i r4 = _mm_loadu_si128((const __m128i*)(src_diff + 4 * src_stride));
    __m128i r5 = _mm_loadu_si128((const __m128i*)(src_diff + 5 * src_stride));
    __m128i r6 = _mm_loadu_si128((const __m128i*)(src_diff + 6 * src_stride));
    __m128i r7 = _mm_loadu_si128((const __m128i*)(src_diff + 7 * src_stride));
    // pass 1: transform the 8 columns (vertical butterfly across the row-vectors)
    __m128i p0, p1, p2, p3, p4, p5, p6, p7;
    WHT8_16(r0, r1, r2, r3, r4, r5, r6, r7, p0, p1, p2, p3, p4, p5, p6, p7);
    // transpose 8x8 (int16) so pass 2 columns become vertical
    __m128i a0 = _mm_unpacklo_epi16(p0, p1), a1 = _mm_unpackhi_epi16(p0, p1);
    __m128i a2 = _mm_unpacklo_epi16(p2, p3), a3 = _mm_unpackhi_epi16(p2, p3);
    __m128i a4 = _mm_unpacklo_epi16(p4, p5), a5 = _mm_unpackhi_epi16(p4, p5);
    __m128i a6 = _mm_unpacklo_epi16(p6, p7), a7 = _mm_unpackhi_epi16(p6, p7);
    __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2);
    __m128i b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
    __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6);
    __m128i b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
    __m128i t0 = _mm_unpacklo_epi64(b0, b4), t1 = _mm_unpackhi_epi64(b0, b4);
    __m128i t2 = _mm_unpacklo_epi64(b1, b5), t3 = _mm_unpackhi_epi64(b1, b5);
    __m128i t4 = _mm_unpacklo_epi64(b2, b6), t5 = _mm_unpackhi_epi64(b2, b6);
    __m128i t6 = _mm_unpacklo_epi64(b3, b7), t7 = _mm_unpackhi_epi64(b3, b7);

    // t0..t7 are exactly the C first-pass buffer rows (buffer[c*8 .. c*8+7]).
    union {
        __m128i v[8];
        int16_t buffer[64];
    } u;

    u.v[0] = t0;
    u.v[1] = t1;
    u.v[2] = t2;
    u.v[3] = t3;
    u.v[4] = t4;
    u.v[5] = t5;
    u.v[6] = t6;
    u.v[7] = t7;
    // pass 2: identical to the C second pass (int32, stride 8), verbatim.
    for (int32_t idx = 0; idx < 8; ++idx) {
        const int16_t* s  = u.buffer + idx;
        int32_t        d0 = s[0 * 8] + s[1 * 8], d1 = s[0 * 8] - s[1 * 8];
        int32_t        d2 = s[2 * 8] + s[3 * 8], d3 = s[2 * 8] - s[3 * 8];
        int32_t        d4 = s[4 * 8] + s[5 * 8], d5 = s[4 * 8] - s[5 * 8];
        int32_t        d6 = s[6 * 8] + s[7 * 8], d7 = s[6 * 8] - s[7 * 8];
        int32_t        e0 = d0 + d2, e1 = d1 + d3, e2 = d0 - d2, e3 = d1 - d3;
        int32_t        e4 = d4 + d6, e5 = d5 + d7, e6 = d4 - d6, e7 = d5 - d7;
        int32_t*       o  = coeff + 8 * idx;
        o[0] = e0 + e4; o[7] = e1 + e5; o[3] = e2 + e6; o[4] = e3 + e7;
        o[2] = e0 - e4; o[6] = e1 - e5; o[1] = e2 - e6; o[5] = e3 - e7;
    }
}

void svt_aom_hadamard_16x16_sse4_1(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    for (int32_t idx = 0; idx < 4; ++idx) {
        const int16_t* src_ptr = src_diff + (idx >> 1) * 8 * src_stride + (idx & 1) * 8;
        svt_aom_hadamard_8x8_sse2(src_ptr, src_stride, coeff + idx * 64);
    }
    for (int32_t i = 0; i < 64; i += 4) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)(coeff + i));
        const __m128i a1 = _mm_loadu_si128((const __m128i*)(coeff + i + 64));
        const __m128i a2 = _mm_loadu_si128((const __m128i*)(coeff + i + 128));
        const __m128i a3 = _mm_loadu_si128((const __m128i*)(coeff + i + 192));
        const __m128i b0 = _mm_srai_epi32(_mm_add_epi32(a0, a1), 1);
        const __m128i b1 = _mm_srai_epi32(_mm_sub_epi32(a0, a1), 1);
        const __m128i b2 = _mm_srai_epi32(_mm_add_epi32(a2, a3), 1);
        const __m128i b3 = _mm_srai_epi32(_mm_sub_epi32(a2, a3), 1);
        _mm_storeu_si128((__m128i*)(coeff + i), _mm_add_epi32(b0, b2));
        _mm_storeu_si128((__m128i*)(coeff + i + 64), _mm_add_epi32(b1, b3));
        _mm_storeu_si128((__m128i*)(coeff + i + 128), _mm_sub_epi32(b0, b2));
        _mm_storeu_si128((__m128i*)(coeff + i + 192), _mm_sub_epi32(b1, b3));
    }
}

void svt_aom_hadamard_32x32_sse4_1(const int16_t* src_diff, ptrdiff_t src_stride, int32_t* coeff) {
    for (int32_t idx = 0; idx < 4; ++idx) {
        const int16_t* src_ptr = src_diff + (idx >> 1) * 16 * src_stride + (idx & 1) * 16;
        svt_aom_hadamard_16x16_sse4_1(src_ptr, src_stride, coeff + idx * 256);
    }
    for (int32_t i = 0; i < 256; i += 4) {
        const __m128i a0 = _mm_loadu_si128((const __m128i*)(coeff + i));
        const __m128i a1 = _mm_loadu_si128((const __m128i*)(coeff + i + 256));
        const __m128i a2 = _mm_loadu_si128((const __m128i*)(coeff + i + 512));
        const __m128i a3 = _mm_loadu_si128((const __m128i*)(coeff + i + 768));
        const __m128i b0 = _mm_srai_epi32(_mm_add_epi32(a0, a1), 2);
        const __m128i b1 = _mm_srai_epi32(_mm_sub_epi32(a0, a1), 2);
        const __m128i b2 = _mm_srai_epi32(_mm_add_epi32(a2, a3), 2);
        const __m128i b3 = _mm_srai_epi32(_mm_sub_epi32(a2, a3), 2);
        _mm_storeu_si128((__m128i*)(coeff + i), _mm_add_epi32(b0, b2));
        _mm_storeu_si128((__m128i*)(coeff + i + 256), _mm_add_epi32(b1, b3));
        _mm_storeu_si128((__m128i*)(coeff + i + 512), _mm_sub_epi32(b0, b2));
        _mm_storeu_si128((__m128i*)(coeff + i + 768), _mm_sub_epi32(b1, b3));
    }
}
