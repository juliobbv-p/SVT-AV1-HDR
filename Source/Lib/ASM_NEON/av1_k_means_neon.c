/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include <assert.h>
#include <string.h>

#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "random.h"

#define DIVIDE_AND_ROUND(x, y) (((x) + ((y) >> 1)) / (y))

static inline int32x4_t k_means_multiply_add_neon(const int16x8_t a) {
    const int32x4_t l = vmull_s16(vget_low_s16(a), vget_low_s16(a));
    const int32x4_t h = vmull_s16(vget_high_s16(a), vget_high_s16(a));
    return vpaddq_s32(l, h);
}

void svt_av1_calc_indices_dim1_neon(const int* data, const int* centroids, uint8_t* indices, int n, int k) {
    int16x8_t cents[PALETTE_MAX_SIZE];

    int j = 0;
    do {
        cents[j] = vdupq_n_s16((int16_t)centroids[j]);
    } while (++j != k);

    do {
        const int16x8_t input0 = vcombine_s16(vmovn_s32(vld1q_s32(data)), vmovn_s32(vld1q_s32(data + 4)));
        const int16x8_t input1 = vcombine_s16(vmovn_s32(vld1q_s32(data + 8)), vmovn_s32(vld1q_s32(data + 12)));
        uint16x8_t      ind0   = vdupq_n_u16(0);
        uint16x8_t      ind1   = vdupq_n_u16(0);
        // Compute the distance to the first centroid.
        int16x8_t dist_min0 = vabdq_s16(input0, cents[0]);
        int16x8_t dist_min1 = vabdq_s16(input1, cents[0]);

        j = 1;
        do {
            // Compute the distance to the centroid.
            const int16x8_t dist0 = vabdq_s16(input0, cents[j]);
            const int16x8_t dist1 = vabdq_s16(input1, cents[j]);

            // Compare to the minimal one.
            const uint16x8_t cmp0 = vcgtq_s16(dist_min0, dist0);
            const uint16x8_t cmp1 = vcgtq_s16(dist_min1, dist1);

            dist_min0 = vminq_s16(dist_min0, dist0);
            dist_min1 = vminq_s16(dist_min1, dist1);

            const uint16x8_t index = vdupq_n_u16(j);
            ind0                   = vbslq_u16(cmp0, index, ind0);
            ind1                   = vbslq_u16(cmp1, index, ind1);
        } while (++j != k);

        vst1_u8(indices + 0, vmovn_u16(ind0));
        vst1_u8(indices + 8, vmovn_u16(ind1));

        indices += 16;
        data += 16;
        n -= 16;
    } while (n != 0);
}

void svt_av1_calc_indices_dim2_neon(const int* data, const int* centroids, uint8_t* indices, int n, int k) {
    int16x8_t cents[PALETTE_MAX_SIZE];

    int j = 0;
    do {
        int32x4_t cxcy_s32 = vreinterpretq_s32_u64(vld1q_dup_u64((const uint64_t*)&centroids[2 * j]));
        cents[j]           = vcombine_s16(vmovn_s32(cxcy_s32), vmovn_s32(cxcy_s32));
    } while (++j != k);

    do {
        const int16x8_t input0 = vcombine_s16(vmovn_s32(vld1q_s32(data)), vmovn_s32(vld1q_s32(data + 4)));
        const int16x8_t input1 = vcombine_s16(vmovn_s32(vld1q_s32(data + 8)), vmovn_s32(vld1q_s32(data + 12)));
        uint32x4_t      ind0   = vdupq_n_u32(0);
        uint32x4_t      ind1   = vdupq_n_u32(0);

        // Compute the distance to the first centroid.
        int16x8_t d0        = vsubq_s16(input0, cents[0]);
        int16x8_t d1        = vsubq_s16(input1, cents[0]);
        int32x4_t dist_min0 = k_means_multiply_add_neon(d0);
        int32x4_t dist_min1 = k_means_multiply_add_neon(d1);

        j = 1;
        do {
            // Compute the distance to the centroid.
            d0 = vsubq_s16(input0, cents[j]);
            d1 = vsubq_s16(input1, cents[j]);

            const int32x4_t dist0 = k_means_multiply_add_neon(d0);
            const int32x4_t dist1 = k_means_multiply_add_neon(d1);

            // Compare to the minimal one.
            const uint32x4_t cmp0 = vcgtq_s32(dist_min0, dist0);
            const uint32x4_t cmp1 = vcgtq_s32(dist_min1, dist1);

            dist_min0 = vminq_s32(dist_min0, dist0);
            dist_min1 = vminq_s32(dist_min1, dist1);

            const uint32x4_t index = vdupq_n_u32(j);
            ind0                   = vbslq_u32(cmp0, index, ind0);
            ind1                   = vbslq_u32(cmp1, index, ind1);
        } while (++j != k);

        // Cast to 8 bit and store.
        vst1_u8(indices, vmovn_u16(vcombine_u16(vmovn_u32(ind0), vmovn_u32(ind1))));

        data += 16;
        indices += 8;
        n -= 8;
    } while (n != 0);
}

static inline int64_t calc_indices_dist_dim1_neon(const int* data, const int* centroids, uint8_t* indices, int n,
                                                  int k) {
    int16x8_t cents[PALETTE_MAX_SIZE];

    int j = 0;
    do {
        cents[j] = vdupq_n_s16((int16_t)centroids[j]);
    } while (++j != k);

    int64x2_t sum = vdupq_n_s64(0);

    do {
        const int16x8_t input0 = vcombine_s16(vmovn_s32(vld1q_s32(data)), vmovn_s32(vld1q_s32(data + 4)));
        const int16x8_t input1 = vcombine_s16(vmovn_s32(vld1q_s32(data + 8)), vmovn_s32(vld1q_s32(data + 12)));
        uint16x8_t      ind0   = vdupq_n_u16(0);
        uint16x8_t      ind1   = vdupq_n_u16(0);
        // Compute the distance to the first centroid.
        int16x8_t dist_min0 = vabdq_s16(input0, cents[0]);
        int16x8_t dist_min1 = vabdq_s16(input1, cents[0]);

        j = 1;
        do {
            // Compute the distance to the centroid.
            const int16x8_t dist0 = vabdq_s16(input0, cents[j]);
            const int16x8_t dist1 = vabdq_s16(input1, cents[j]);

            // Compare to the minimal one.
            const uint16x8_t cmp0 = vcgtq_s16(dist_min0, dist0);
            const uint16x8_t cmp1 = vcgtq_s16(dist_min1, dist1);

            dist_min0 = vminq_s16(dist_min0, dist0);
            dist_min1 = vminq_s16(dist_min1, dist1);

            const uint16x8_t index = vdupq_n_u16((uint16_t)j);
            ind0                   = vbslq_u16(cmp0, index, ind0);
            ind1                   = vbslq_u16(cmp1, index, ind1);
        } while (++j != k);

        vst1_u8(indices + 0, vmovn_u16(ind0));
        vst1_u8(indices + 8, vmovn_u16(ind1));

        const int32x4_t sq0 = vmull_s16(vget_low_s16(dist_min0), vget_low_s16(dist_min0));
        const int32x4_t sq1 = vmull_s16(vget_high_s16(dist_min0), vget_high_s16(dist_min0));
        const int32x4_t sq2 = vmull_s16(vget_low_s16(dist_min1), vget_low_s16(dist_min1));
        const int32x4_t sq3 = vmull_s16(vget_high_s16(dist_min1), vget_high_s16(dist_min1));
        sum                 = vpadalq_s32(sum, sq0);
        sum                 = vpadalq_s32(sum, sq1);
        sum                 = vpadalq_s32(sum, sq2);
        sum                 = vpadalq_s32(sum, sq3);

        indices += 16;
        data += 16;
        n -= 16;
    } while (n != 0);

    return vaddvq_s64(sum);
}

static inline void calc_centroids_1_neon(const int* data, int* centroids, const uint8_t* indices, int n, int k) {
    int          i;
    int          count[PALETTE_MAX_SIZE] = {0};
    unsigned int rand_state              = (unsigned int)data[0];
    assert(n <= MAX_SB_SQUARE); // n is at most a 128x128 superblock
    memset(centroids, 0, sizeof(centroids[0]) * k);

    for (i = 0; i < n; ++i) {
        const int index = indices[i];
        assert(index < k);
        ++count[index];
        centroids[index] += data[i];
    }

    for (i = 0; i < k; ++i) {
        if (count[i] == 0) {
            centroids[i] = data[lcg_rand16(&rand_state) % n];
        } else {
            centroids[i] = DIVIDE_AND_ROUND(centroids[i], count[i]);
        }
    }
}

void svt_av1_k_means_dim1_neon(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr) {
    int     pre_centroids[2 * PALETTE_MAX_SIZE];
    uint8_t idx_buf[MAX_SB_SQUARE];
    assert((n & 15) == 0);

    // Double-buffer the per-pixel assignment: `cur` is the best assignment so far, `nxt` is scratch for
    // the next one. Each accepted iteration just swaps the pointers instead of copying up to MAX_SB_SQUARE
    // indices; the result is copied back into the caller's `indices` once, at the end.
    uint8_t* cur = indices;
    uint8_t* nxt = idx_buf;

    int64_t this_dist = calc_indices_dist_dim1_neon(data, centroids, cur, n, k);

    for (int i = 0; i < max_itr; ++i) {
        const int64_t pre_dist = this_dist;
        memcpy(pre_centroids, centroids, sizeof(pre_centroids[0]) * k);

        calc_centroids_1_neon(data, centroids, cur, n, k);
        const int64_t new_dist = calc_indices_dist_dim1_neon(data, centroids, nxt, n, k);

        if (new_dist > pre_dist) {
            // The previous assignment (cur) was better: roll back the centroids and keep cur.
            memcpy(centroids, pre_centroids, sizeof(pre_centroids[0]) * k);
            break;
        }
        this_dist          = new_dist;
        uint8_t* const tmp = cur;
        cur                = nxt;
        nxt                = tmp;
        if (!memcmp(centroids, pre_centroids, sizeof(pre_centroids[0]) * k)) {
            break;
        }
    }

    if (cur != indices) {
        memcpy(indices, cur, sizeof(indices[0]) * n);
    }
}

static inline int64_t calc_indices_dist_dim2_neon(const int* data, const int* centroids, uint8_t* indices, int n,
                                                  int k) {
    int16x8_t cents[PALETTE_MAX_SIZE];

    int j = 0;
    do {
        int32x4_t cxcy_s32 = vreinterpretq_s32_u64(vld1q_dup_u64((const uint64_t*)&centroids[2 * j]));
        cents[j]           = vcombine_s16(vmovn_s32(cxcy_s32), vmovn_s32(cxcy_s32));
    } while (++j != k);

    int64x2_t sum = vdupq_n_s64(0);

    do {
        const int16x8_t input0 = vcombine_s16(vmovn_s32(vld1q_s32(data)), vmovn_s32(vld1q_s32(data + 4)));
        const int16x8_t input1 = vcombine_s16(vmovn_s32(vld1q_s32(data + 8)), vmovn_s32(vld1q_s32(data + 12)));
        uint32x4_t      ind0   = vdupq_n_u32(0);
        uint32x4_t      ind1   = vdupq_n_u32(0);

        // Compute the distance to the first centroid.
        int16x8_t d0        = vsubq_s16(input0, cents[0]);
        int16x8_t d1        = vsubq_s16(input1, cents[0]);
        int32x4_t dist_min0 = k_means_multiply_add_neon(d0);
        int32x4_t dist_min1 = k_means_multiply_add_neon(d1);

        j = 1;
        do {
            // Compute the distance to the centroid.
            d0 = vsubq_s16(input0, cents[j]);
            d1 = vsubq_s16(input1, cents[j]);

            const int32x4_t dist0 = k_means_multiply_add_neon(d0);
            const int32x4_t dist1 = k_means_multiply_add_neon(d1);

            // Compare to the minimal one.
            const uint32x4_t cmp0 = vcgtq_s32(dist_min0, dist0);
            const uint32x4_t cmp1 = vcgtq_s32(dist_min1, dist1);

            dist_min0 = vminq_s32(dist_min0, dist0);
            dist_min1 = vminq_s32(dist_min1, dist1);

            const uint32x4_t index = vdupq_n_u32((uint32_t)j);
            ind0                   = vbslq_u32(cmp0, index, ind0);
            ind1                   = vbslq_u32(cmp1, index, ind1);
        } while (++j != k);

        // Cast to 8 bit and store.
        vst1_u8(indices, vmovn_u16(vcombine_u16(vmovn_u32(ind0), vmovn_u32(ind1))));

        sum = vpadalq_s32(sum, dist_min0);
        sum = vpadalq_s32(sum, dist_min1);

        data += 16;
        indices += 8;
        n -= 8;
    } while (n != 0);

    return vaddvq_s64(sum);
}

static inline void calc_centroids_2_neon(const int* data, int* centroids, const uint8_t* indices, int n, int k) {
    int          i;
    int          count[PALETTE_MAX_SIZE] = {0};
    unsigned int rand_state              = (unsigned int)data[0];
    assert(n <= MAX_SB_SQUARE); // n is at most a 128x128 superblock
    memset(centroids, 0, sizeof(centroids[0]) * k * 2);

    for (i = 0; i < n; ++i) {
        const int index = indices[i];
        assert(index < k);
        ++count[index];
        centroids[index * 2] += data[i * 2];
        centroids[index * 2 + 1] += data[i * 2 + 1];
    }

    for (i = 0; i < k; ++i) {
        if (count[i] == 0) {
            const int src        = (lcg_rand16(&rand_state) % n) * 2;
            centroids[i * 2]     = data[src];
            centroids[i * 2 + 1] = data[src + 1];
        } else {
            centroids[i * 2]     = DIVIDE_AND_ROUND(centroids[i * 2], count[i]);
            centroids[i * 2 + 1] = DIVIDE_AND_ROUND(centroids[i * 2 + 1], count[i]);
        }
    }
}

void svt_av1_k_means_dim2_neon(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr) {
    int     pre_centroids[2 * PALETTE_MAX_SIZE];
    uint8_t idx_buf[MAX_SB_SQUARE];
    assert((n & 15) == 0);

    // Double-buffer the assignment (see svt_av1_k_means_dim1_neon): swap `cur`/`nxt` instead of copying
    // the indices each iteration, and copy back into the caller's `indices` once at the end.
    uint8_t* cur = indices;
    uint8_t* nxt = idx_buf;

    int64_t this_dist = calc_indices_dist_dim2_neon(data, centroids, cur, n, k);

    for (int i = 0; i < max_itr; ++i) {
        const int64_t pre_dist = this_dist;
        memcpy(pre_centroids, centroids, sizeof(pre_centroids[0]) * k * 2);

        calc_centroids_2_neon(data, centroids, cur, n, k);
        const int64_t new_dist = calc_indices_dist_dim2_neon(data, centroids, nxt, n, k);

        if (new_dist > pre_dist) {
            memcpy(centroids, pre_centroids, sizeof(pre_centroids[0]) * k * 2);
            break;
        }
        this_dist          = new_dist;
        uint8_t* const tmp = cur;
        cur                = nxt;
        nxt                = tmp;
        if (!memcmp(centroids, pre_centroids, sizeof(pre_centroids[0]) * k * 2)) {
            break;
        }
    }

    if (cur != indices) {
        memcpy(indices, cur, sizeof(indices[0]) * n);
    }
}
