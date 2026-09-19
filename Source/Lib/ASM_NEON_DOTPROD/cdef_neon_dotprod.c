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
#include <math.h>

#include "aom_dsp_rtcd.h"
#include "cdef.h"
#include "definitions.h"
#include "mem_neon.h"

/* 8-bit MSE via vdotq self-square. Two independent accumulators (2-way block unroll)
 * break the per-block vdotq dependency chain so it runs throughput-bound. Bit-exact. */
static inline uint32x4_t mse_block_8w_dotprod(uint32x4_t acc, const uint8_t* s, const uint8_t* d, int32_t dstride,
                                              int bh, int sub) {
    for (int h = 0; h < bh; h += 2 * sub) {
        const uint8x16_t sv = vcombine_u8(vld1_u8(s), vld1_u8(s + sub * 8));
        const uint8x16_t dv = vcombine_u8(vld1_u8(d), vld1_u8(d + sub * dstride));
        const uint8x16_t ab = vabdq_u8(sv, dv);
        acc                 = vdotq_u32(acc, ab, ab);
        s += 8 * 2 * sub;
        d += 2 * sub * dstride;
    }
    return acc;
}

static inline uint32x4_t mse_block_4w_dotprod(uint32x4_t acc, const uint8_t* s, const uint8_t* d, int32_t dstride,
                                              int bh, int sub) {
    for (int h = 0; h < bh; h += 4 * sub) {
        const uint8x16_t sv = load_u8_4x4(s, 4 * sub);
        const uint8x16_t dv = load_u8_4x4(d, dstride * sub);
        const uint8x16_t ab = vabdq_u8(sv, dv);
        acc                 = vdotq_u32(acc, ab, ab);
        s += 4 * 4 * sub;
        d += 4 * sub * dstride;
    }
    return acc;
}

uint64_t svt_aom_compute_cdef_dist_8bit_neon_dotprod(const uint8_t* dst8, int32_t dstride, const uint8_t* src8,
                                                     const CdefList* dlist, int32_t cdef_count, BlockSize bsize,
                                                     int32_t coeff_shift, uint8_t subsampling_factor) {
    uint32x4_t a0 = vdupq_n_u32(0), a1 = vdupq_n_u32(0);

    const int bw  = (bsize == BLOCK_8X8 || bsize == BLOCK_8X4) ? 8 : 4;
    const int bh  = (bsize == BLOCK_8X8 || bsize == BLOCK_4X8) ? 8 : 4;
    const int sub = subsampling_factor;
    const int blk = bw * bh;

    int bi = 0;
    if (bw == 8) {
        for (; bi + 2 <= cdef_count; bi += 2) {
            const uint8_t* s = src8 + bi * blk;
            a0               = mse_block_8w_dotprod(
                a0, s + 0 * blk, dst8 + bh * dlist[bi + 0].by * dstride + 8 * dlist[bi + 0].bx, dstride, bh, sub);
            a1 = mse_block_8w_dotprod(
                a1, s + 1 * blk, dst8 + bh * dlist[bi + 1].by * dstride + 8 * dlist[bi + 1].bx, dstride, bh, sub);
        }
        for (; bi < cdef_count; bi++) {
            a0 = mse_block_8w_dotprod(
                a0, src8 + bi * blk, dst8 + bh * dlist[bi].by * dstride + 8 * dlist[bi].bx, dstride, bh, sub);
        }
    } else {
        for (; bi + 2 <= cdef_count; bi += 2) {
            const uint8_t* s = src8 + bi * blk;
            a0               = mse_block_4w_dotprod(
                a0, s + 0 * blk, dst8 + bh * dlist[bi + 0].by * dstride + 4 * dlist[bi + 0].bx, dstride, bh, sub);
            a1 = mse_block_4w_dotprod(
                a1, s + 1 * blk, dst8 + bh * dlist[bi + 1].by * dstride + 4 * dlist[bi + 1].bx, dstride, bh, sub);
        }
        for (; bi < cdef_count; bi++) {
            a0 = mse_block_4w_dotprod(
                a0, src8 + bi * blk, dst8 + bh * dlist[bi].by * dstride + 4 * dlist[bi].bx, dstride, bh, sub);
        }
    }

    const uint64_t sum = vaddlvq_u32(vaddq_u32(a0, a1));
    return sum >> 2 * coeff_shift;
}
