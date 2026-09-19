/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <smmintrin.h>
#include "definitions.h"

static INLINE void energy_kernel_sse4_1(const int32_t *in, __m128i *sum) {
    const __m128i zero  = _mm_setzero_si128();
    const __m128i input = _mm_loadu_si128((const __m128i *)in);
    const __m128i lo    = _mm_unpacklo_epi32(input, zero);
    const __m128i hi    = _mm_unpackhi_epi32(input, zero);
    *sum                = _mm_add_epi64(*sum, _mm_mul_epi32(lo, lo));
    *sum                = _mm_add_epi64(*sum, _mm_mul_epi32(hi, hi));
}

static INLINE uint64_t hadd64_sse4_1(const __m128i sum) {
    const __m128i hi = _mm_srli_si128(sum, 8);
    return (uint64_t)_mm_extract_epi64(_mm_add_epi64(sum, hi), 0);
}

static INLINE uint64_t energy_computation_sse4_1(const int32_t *in, const uint32_t size) {
    __m128i  sum = _mm_setzero_si128();
    uint32_t i   = 0;
    do {
        energy_kernel_sse4_1(in + i, &sum);
        i += 4;
    } while (i < size);
    return hadd64_sse4_1(sum);
}

static INLINE uint64_t energy_computation_64_sse4_1(const int32_t *in, const uint32_t height) {
    __m128i  sum = _mm_setzero_si128();
    uint32_t i   = height;
    do {
        energy_kernel_sse4_1(in + 0 * 4, &sum);
        energy_kernel_sse4_1(in + 1 * 4, &sum);
        energy_kernel_sse4_1(in + 2 * 4, &sum);
        energy_kernel_sse4_1(in + 3 * 4, &sum);
        energy_kernel_sse4_1(in + 4 * 4, &sum);
        energy_kernel_sse4_1(in + 5 * 4, &sum);
        energy_kernel_sse4_1(in + 6 * 4, &sum);
        energy_kernel_sse4_1(in + 7 * 4, &sum);
        in += 64;
    } while (--i);
    return hadd64_sse4_1(sum);
}

static INLINE void repack_32x_sse4_1(const int32_t *src, int32_t *dst, const uint32_t height) {
    uint32_t h = height;
    do {
        for (int x = 0; x < 32; x += 4)
            _mm_storeu_si128((__m128i *)(dst + x), _mm_loadu_si128((const __m128i *)(src + x)));
        src += 64;
        dst += 32;
    } while (--h);
}

uint64_t svt_handle_transform16x64_sse4_1(int32_t *output) {
    const uint64_t three_quad_energy = energy_computation_sse4_1(output + 16 * 32, 16 * 32);
    return three_quad_energy;
}

uint64_t svt_handle_transform32x64_sse4_1(int32_t *output) {
    const uint64_t three_quad_energy = energy_computation_sse4_1(output + 32 * 32, 32 * 32);
    return three_quad_energy;
}

uint64_t svt_handle_transform64x16_sse4_1(int32_t *output) {
    const uint64_t three_quad_energy = energy_computation_64_sse4_1(output + 32, 16);
    repack_32x_sse4_1(output + 64, output + 32, 15);
    return three_quad_energy;
}

uint64_t svt_handle_transform64x32_sse4_1(int32_t *output) {
    const uint64_t three_quad_energy = energy_computation_64_sse4_1(output + 32, 32);
    repack_32x_sse4_1(output + 64, output + 32, 31);
    return three_quad_energy;
}

uint64_t svt_handle_transform64x64_sse4_1(int32_t *output) {
    uint64_t three_quad_energy = energy_computation_64_sse4_1(output + 32, 32);
    three_quad_energy += energy_computation_sse4_1(output + 32 * 64, 64 * 32);
    repack_32x_sse4_1(output + 64, output + 32, 31);
    return three_quad_energy;
}

uint64_t svt_handle_transform16x64_N2_N4_sse4_1(int32_t *output) {
    (void)output;
    return 0;
}

uint64_t svt_handle_transform32x64_N2_N4_sse4_1(int32_t *output) {
    (void)output;
    return 0;
}

uint64_t svt_handle_transform64x16_N2_N4_sse4_1(int32_t *output) {
    repack_32x_sse4_1(output + 64, output + 32, 15);
    return 0;
}

uint64_t svt_handle_transform64x32_N2_N4_sse4_1(int32_t *output) {
    repack_32x_sse4_1(output + 64, output + 32, 31);
    return 0;
}

uint64_t svt_handle_transform64x64_N2_N4_sse4_1(int32_t *output) {
    repack_32x_sse4_1(output + 64, output + 32, 31);
    return 0;
}
