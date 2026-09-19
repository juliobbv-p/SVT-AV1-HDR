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

#include <string.h>
#include <smmintrin.h>
#include "definitions.h"
#include "common_dsp_rtcd.h"
#include "random.h"
#include "utility.h"

void svt_av1_calc_indices_dim1_sse4_1(const int* data, const int* centroids, uint8_t* indices, int n, int k) {
    __m128i results[MAX_SB_SQUARE / 4];
    memset(indices, 0, n * sizeof(uint8_t));

    __m128i c0 = _mm_set1_epi32(centroids[0]);
    for (int i = 0; i < n; i += 4) {
        __m128i sub    = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(data + i)), c0);
        results[i / 4] = _mm_mullo_epi32(sub, sub);
    }

    for (int c = 1; c < k; c++) {
        const __m128i cent  = _mm_set1_epi32(centroids[c]);
        const __m128i idx_v = _mm_set1_epi32(c);
        for (int i = 0; i < n; i += 8) {
            const __m128i d1   = _mm_loadu_si128((const __m128i*)(data + i));
            const __m128i d2   = _mm_loadu_si128((const __m128i*)(data + i + 4));
            const __m128i s1   = _mm_sub_epi32(d1, cent);
            const __m128i s2   = _mm_sub_epi32(d2, cent);
            const __m128i dst1 = _mm_mullo_epi32(s1, s1);
            const __m128i dst2 = _mm_mullo_epi32(s2, s2);

            const __m128i prev1 = results[i / 4];
            const __m128i prev2 = results[i / 4 + 1];
            const __m128i cmp1  = _mm_cmpgt_epi32(prev1, dst1);
            const __m128i cmp2  = _mm_cmpgt_epi32(prev2, dst2);

            results[i / 4]     = _mm_blendv_epi8(prev1, dst1, cmp1);
            results[i / 4 + 1] = _mm_blendv_epi8(prev2, dst2, cmp2);

            const __m128i iv1  = _mm_and_si128(idx_v, cmp1);
            const __m128i iv2  = _mm_and_si128(idx_v, cmp2);
            const __m128i i16  = _mm_packus_epi32(iv1, iv2);
            const __m128i i8   = _mm_packs_epi16(i16, _mm_setzero_si128());
            const __m128i load = _mm_loadl_epi64((const __m128i*)(indices + i));
            _mm_storel_epi64((__m128i*)(indices + i), _mm_max_epi8(load, i8));
        }
    }
}

static INLINE int64_t calc_indices_dist_dim1_sse4_1(const int* data, const int* centroids, uint8_t* indices, unsigned n,
                                                    int k) {
    __m128i results[MAX_SB_SQUARE / 4];
    memset(indices, 0, n * sizeof(uint8_t));

    __m128i c0 = _mm_set1_epi32(centroids[0]);
    for (unsigned i = 0; i < n; i += 4) {
        __m128i sub    = _mm_sub_epi32(_mm_loadu_si128((const __m128i*)(data + i)), c0);
        results[i / 4] = _mm_mullo_epi32(sub, sub);
    }

    for (int c = 1; c < k; c++) {
        const __m128i cent  = _mm_set1_epi32(centroids[c]);
        const __m128i idx_v = _mm_set1_epi32(c);
        for (unsigned i = 0; i < n; i += 8) {
            const __m128i d1   = _mm_loadu_si128((const __m128i*)(data + i));
            const __m128i d2   = _mm_loadu_si128((const __m128i*)(data + i + 4));
            const __m128i s1   = _mm_sub_epi32(d1, cent);
            const __m128i s2   = _mm_sub_epi32(d2, cent);
            const __m128i dst1 = _mm_mullo_epi32(s1, s1);
            const __m128i dst2 = _mm_mullo_epi32(s2, s2);

            const __m128i prev1 = results[i / 4];
            const __m128i prev2 = results[i / 4 + 1];
            const __m128i cmp1  = _mm_cmpgt_epi32(prev1, dst1);
            const __m128i cmp2  = _mm_cmpgt_epi32(prev2, dst2);

            results[i / 4]     = _mm_blendv_epi8(prev1, dst1, cmp1);
            results[i / 4 + 1] = _mm_blendv_epi8(prev2, dst2, cmp2);

            const __m128i iv1  = _mm_and_si128(idx_v, cmp1);
            const __m128i iv2  = _mm_and_si128(idx_v, cmp2);
            const __m128i i16  = _mm_packus_epi32(iv1, iv2);
            const __m128i i8   = _mm_packs_epi16(i16, _mm_setzero_si128());
            const __m128i load = _mm_loadl_epi64((const __m128i*)(indices + i));
            _mm_storel_epi64((__m128i*)(indices + i), _mm_max_epi8(load, i8));
        }
    }

    __m128i sum64 = _mm_setzero_si128();
    for (unsigned i = 0; i < n; i += 4) {
        const __m128i prev = results[i / 4];
        sum64              = _mm_add_epi64(sum64, _mm_unpacklo_epi32(prev, _mm_setzero_si128()));
        sum64              = _mm_add_epi64(sum64, _mm_unpackhi_epi32(prev, _mm_setzero_si128()));
    }
    return _mm_extract_epi64(sum64, 0) + _mm_extract_epi64(sum64, 1);
}

static INLINE void calc_centroids_1_sse4_1(const int* data, int* centroids, const uint8_t* indices, int n, int k) {
    int          count[PALETTE_MAX_SIZE] = {0};
    unsigned int rand_state              = (unsigned int)data[0];
    assert(n <= 32768);
    memset(centroids, 0, sizeof(centroids[0]) * k);

    for (int i = 0; i < n; ++i) {
        const int index = indices[i];
        assert(index < k);
        ++count[index];
        centroids[index] += data[i];
    }

    for (int i = 0; i < k; ++i) {
        if (count[i] == 0) {
            centroids[i] = *(data + (lcg_rand16(&rand_state) % n));
        } else {
            centroids[i] = DIVIDE_AND_ROUND(centroids[i], count[i]);
        }
    }
}

void svt_av1_k_means_dim1_sse4_1(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr) {
    int     pre_centroids[2 * PALETTE_MAX_SIZE];
    uint8_t pre_indices[MAX_SB_SQUARE];
    assert((n & 15) == 0);

    int64_t this_dist = calc_indices_dist_dim1_sse4_1(data, centroids, indices, n, k);

    for (int i = 0; i < max_itr; ++i) {
        const int64_t pre_dist = this_dist;
        svt_memcpy_intrin_sse(pre_centroids, centroids, sizeof(pre_centroids[0]) * k);
        svt_memcpy_intrin_sse(pre_indices, indices, sizeof(pre_indices[0]) * n);

        calc_centroids_1_sse4_1(data, centroids, indices, n, k);
        this_dist = calc_indices_dist_dim1_sse4_1(data, centroids, indices, n, k);

        if (this_dist > pre_dist) {
            svt_memcpy_intrin_sse(centroids, pre_centroids, sizeof(pre_centroids[0]) * k);
            svt_memcpy_intrin_sse(indices, pre_indices, sizeof(pre_indices[0]) * n);
            break;
        }
        if (!memcmp(centroids, pre_centroids, sizeof(pre_centroids[0]) * k)) {
            break;
        }
    }
}

static INLINE void dist2_4pts_sse4_1(const int* data, __m128i cent01, __m128i* out) {
    const __m128i a  = _mm_loadu_si128((const __m128i*)(data + 0));
    const __m128i b  = _mm_loadu_si128((const __m128i*)(data + 4));
    const __m128i sa = _mm_sub_epi32(a, cent01);
    const __m128i sb = _mm_sub_epi32(b, cent01);
    *out             = _mm_hadd_epi32(_mm_mullo_epi32(sa, sa), _mm_mullo_epi32(sb, sb));
}

void svt_av1_calc_indices_dim2_sse4_1(const int* data, const int* centroids, uint8_t* indices, int n, int k) {
    __m128i results[MAX_SB_SQUARE / 4];
    memset(indices, 0, n * sizeof(uint8_t));

    __m128i cent01 = _mm_set1_epi64x(*((const uint64_t*)&centroids[0]));
    for (int i = 0; i < n; i += 4) {
        __m128i dist;
        dist2_4pts_sse4_1(data + 2 * i, cent01, &dist);
        results[i / 4] = dist;
    }

    for (int j = 1; j < k; ++j) {
        cent01              = _mm_set1_epi64x(*((const uint64_t*)&centroids[2 * j]));
        const __m128i idx_v = _mm_set1_epi32(j);
        for (int i = 0; i < n; i += 8) {
            __m128i dlo, dhi;
            dist2_4pts_sse4_1(data + 2 * i, cent01, &dlo);
            dist2_4pts_sse4_1(data + 2 * (i + 4), cent01, &dhi);

            const __m128i prev_lo = results[i / 4];
            const __m128i prev_hi = results[i / 4 + 1];
            const __m128i cmp_lo  = _mm_cmpgt_epi32(prev_lo, dlo);
            const __m128i cmp_hi  = _mm_cmpgt_epi32(prev_hi, dhi);

            results[i / 4]     = _mm_blendv_epi8(prev_lo, dlo, cmp_lo);
            results[i / 4 + 1] = _mm_blendv_epi8(prev_hi, dhi, cmp_hi);

            const __m128i iv_lo = _mm_and_si128(idx_v, cmp_lo);
            const __m128i iv_hi = _mm_and_si128(idx_v, cmp_hi);
            const __m128i i16   = _mm_packus_epi32(iv_lo, iv_hi);
            const __m128i i8    = _mm_packs_epi16(i16, _mm_setzero_si128());
            const __m128i load  = _mm_loadl_epi64((const __m128i*)(indices + i));
            _mm_storel_epi64((__m128i*)(indices + i), _mm_max_epi8(load, i8));
        }
    }
}

static INLINE int64_t calc_indices_dist_dim2_sse4_1(const int* data, const int* centroids, uint8_t* indices, unsigned n,
                                                    int k) {
    __m128i results[MAX_SB_SQUARE / 4];
    memset(indices, 0, n * sizeof(uint8_t));

    __m128i cent01 = _mm_set1_epi64x(*((const uint64_t*)&centroids[0]));
    for (unsigned i = 0; i < n; i += 4) {
        __m128i dist;
        dist2_4pts_sse4_1(data + 2 * i, cent01, &dist);
        results[i / 4] = dist;
    }

    for (int j = 1; j < k; ++j) {
        cent01              = _mm_set1_epi64x(*((const uint64_t*)&centroids[2 * j]));
        const __m128i idx_v = _mm_set1_epi32(j);
        for (unsigned i = 0; i < n; i += 8) {
            __m128i dlo, dhi;
            dist2_4pts_sse4_1(data + 2 * i, cent01, &dlo);
            dist2_4pts_sse4_1(data + 2 * (i + 4), cent01, &dhi);

            const __m128i prev_lo = results[i / 4];
            const __m128i prev_hi = results[i / 4 + 1];
            const __m128i cmp_lo  = _mm_cmpgt_epi32(prev_lo, dlo);
            const __m128i cmp_hi  = _mm_cmpgt_epi32(prev_hi, dhi);

            results[i / 4]     = _mm_blendv_epi8(prev_lo, dlo, cmp_lo);
            results[i / 4 + 1] = _mm_blendv_epi8(prev_hi, dhi, cmp_hi);

            const __m128i iv_lo = _mm_and_si128(idx_v, cmp_lo);
            const __m128i iv_hi = _mm_and_si128(idx_v, cmp_hi);
            const __m128i i16   = _mm_packus_epi32(iv_lo, iv_hi);
            const __m128i i8    = _mm_packs_epi16(i16, _mm_setzero_si128());
            const __m128i load  = _mm_loadl_epi64((const __m128i*)(indices + i));
            _mm_storel_epi64((__m128i*)(indices + i), _mm_max_epi8(load, i8));
        }
    }

    __m128i sum64 = _mm_setzero_si128();
    for (unsigned i = 0; i < n; i += 4) {
        const __m128i prev = results[i / 4];
        sum64              = _mm_add_epi64(sum64, _mm_unpacklo_epi32(prev, _mm_setzero_si128()));
        sum64              = _mm_add_epi64(sum64, _mm_unpackhi_epi32(prev, _mm_setzero_si128()));
    }
    return _mm_extract_epi64(sum64, 0) + _mm_extract_epi64(sum64, 1);
}

static INLINE void calc_centroids_2_sse4_1(const int* data, int* centroids, const uint8_t* indices, int n, int k) {
    int          count[PALETTE_MAX_SIZE] = {0};
    unsigned int rand_state              = (unsigned int)data[0];
    assert(n <= 32768);
    memset(centroids, 0, sizeof(centroids[0]) * k * 2);

    for (int i = 0; i < n; ++i) {
        const int index = indices[i];
        assert(index < k);
        ++count[index];
        centroids[index * 2] += data[i * 2];
        centroids[index * 2 + 1] += data[i * 2 + 1];
    }

    for (int i = 0; i < k; ++i) {
        if (count[i] == 0) {
            svt_memcpy_intrin_sse(
                centroids + i * 2, (void*)(data + (lcg_rand16(&rand_state) % n) * 2), sizeof(centroids[0]) * 2);
        } else {
            centroids[i * 2]     = DIVIDE_AND_ROUND(centroids[i * 2], count[i]);
            centroids[i * 2 + 1] = DIVIDE_AND_ROUND(centroids[i * 2 + 1], count[i]);
        }
    }
}

void svt_av1_k_means_dim2_sse4_1(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr) {
    int     pre_centroids[2 * PALETTE_MAX_SIZE];
    uint8_t pre_indices[MAX_SB_SQUARE];
    assert((n & 15) == 0);

    int64_t this_dist = calc_indices_dist_dim2_sse4_1(data, centroids, indices, n, k);

    for (int i = 0; i < max_itr; ++i) {
        const int64_t pre_dist = this_dist;
        svt_memcpy_intrin_sse(pre_centroids, centroids, sizeof(pre_centroids[0]) * k * 2);
        svt_memcpy_intrin_sse(pre_indices, indices, sizeof(pre_indices[0]) * n);

        calc_centroids_2_sse4_1(data, centroids, indices, n, k);
        this_dist = calc_indices_dist_dim2_sse4_1(data, centroids, indices, n, k);

        if (this_dist > pre_dist) {
            svt_memcpy_intrin_sse(centroids, pre_centroids, sizeof(pre_centroids[0]) * k * 2);
            svt_memcpy_intrin_sse(indices, pre_indices, sizeof(pre_indices[0]) * n);
            break;
        }
        if (!memcmp(centroids, pre_centroids, sizeof(pre_centroids[0]) * k * 2)) {
            break;
        }
    }
}
