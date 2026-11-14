/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance
 * for Open Media Patent License 1.0 was not distributed with this source code
 * in the PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include <arm_neon.h>

#include "cabac_context_model.h"
#include "definitions.h"
#include "mem_neon.h"

// Base-level rate of sixteen consecutive raster positions. lv holds their levels (clamped as in the levels
// buffer), acc[0..1] sum the low and high bytes of the costs, acc[2] counts the nonzero levels (as -1 each).
// Returns the positions whose level exceeds the base levels, four mask bits per position.
static AOM_FORCE_INLINE uint64_t cost_base_16(const int16_t* iscan, const int8_t* coeff_contexts, uint8x16_t lv,
                                              const uint8x16x4_t tbl[4], uint16x8_t last, uint16x8_t acc[2],
                                              uint8x16_t* nz) {
    // In the scan range [1, eob - 2]?
    const uint16x8_t in0 = vcltq_u16(vsubq_u16(vld1q_u16((const uint16_t*)iscan), vdupq_n_u16(1)), last);
    const uint16x8_t in1 = vcltq_u16(vsubq_u16(vld1q_u16((const uint16_t*)iscan + 8), vdupq_n_u16(1)), last);
    const uint8x16_t in  = vuzp1q_u8(vreinterpretq_u8_u16(in0), vreinterpretq_u8_u16(in1));

    // Table index: the context (below 32 for the 2D class) in the low five bits, min(level, 3) above them.
    const uint8x16_t ctx  = vreinterpretq_u8_s8(vld1q_s8(coeff_contexts));
    const uint8x16_t idx  = vsliq_n_u8(ctx, vminq_u8(lv, vdupq_n_u8(3)), 5);
    const uint8x16_t idx2 = vsubq_u8(idx, vdupq_n_u8(64));
    const uint8x16_t lo   = vqtbx4q_u8(vqtbl4q_u8(tbl[0], idx), tbl[1], idx2);
    const uint8x16_t hi   = vqtbx4q_u8(vqtbl4q_u8(tbl[2], idx), tbl[3], idx2);

    acc[0] = vpadalq_u8(acc[0], vandq_u8(lo, in));
    acc[1] = vpadalq_u8(acc[1], vandq_u8(hi, in));
    *nz    = vaddq_u8(*nz, vandq_u8(vtstq_u8(lv, lv), in));

    const uint8x16_t big = vandq_u8(vcgtq_u8(lv, vdupq_n_u8(NUM_BASE_LEVELS)), in);
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(big), 4)), 0);
}

// Rate of the base levels of the scan positions [1, eob - 2] of a 2D-class transform block, visited in raster
// order: a position is in that range iff its iscan value is. Positions on anti-diagonals beyond `diag` (that of
// the last coefficient) are never in range, which bounds the rows and the columns visited. Each group of sixteen
// positions with levels above the base levels is reported through big_start (its first raster position) and
// big_mask (four bits per position); the caller adds their range rates. The return value is the number of such
// groups; *cost receives the base-level rate without the sign bits and *nz the number of nonzero levels.
//
// A group adds at most 2 * 255 to a 16-bit lane of the byte sums and 1 to a lane of the nonzero count, and a
// block of at most 32x32 coded positions has at most 64 groups.
int svt_av1_cost_coeffs_base_2d_neon(const int16_t* iscan, const int8_t* coeff_contexts, const uint8_t* levels,
                                     const uint8_t* base_cost8, int bwl, int height, int diag, int eob, int32_t* cost,
                                     int32_t* nz_count, uint16_t* big_start, uint64_t* big_mask) {
    const uint8x16x4_t tbl[4] = {
        {{vld1q_u8(base_cost8), vld1q_u8(base_cost8 + 16), vld1q_u8(base_cost8 + 32), vld1q_u8(base_cost8 + 48)}},
        {{vld1q_u8(base_cost8 + 64), vld1q_u8(base_cost8 + 80), vld1q_u8(base_cost8 + 96), vld1q_u8(base_cost8 + 112)}},
        {{vld1q_u8(base_cost8 + 128),
          vld1q_u8(base_cost8 + 144),
          vld1q_u8(base_cost8 + 160),
          vld1q_u8(base_cost8 + 176)}},
        {{vld1q_u8(base_cost8 + 192),
          vld1q_u8(base_cost8 + 208),
          vld1q_u8(base_cost8 + 224),
          vld1q_u8(base_cost8 + 240)}},
    };
    const uint16x8_t last   = vdupq_n_u16((uint16_t)(eob - 2));
    uint16x8_t       acc[2] = {vdupq_n_u16(0), vdupq_n_u16(0)};
    uint8x16_t       nz     = vdupq_n_u8(0);
    const int        width  = 1 << bwl;
    const int        stride = width + TX_PAD_HOR;
    const int        rows   = AOMMIN(height, diag + 1);
    int              n      = 0;

#define COST_BASE_GROUP(p, lv)                                                                         \
    do {                                                                                               \
        const uint64_t big = cost_base_16(iscan + (p), coeff_contexts + (p), lv, tbl, last, acc, &nz); \
        big_start[n]       = (uint16_t)(p);                                                            \
        big_mask[n]        = big;                                                                      \
        n += big != 0;                                                                                 \
    } while (0)

    if (bwl >= 4) {
        for (int r = 0; r < rows; ++r) {
            // Columns 0 .. diag - r of this row can be in range.
            for (int c = 0; c < width && c <= diag - r; c += 16) {
                COST_BASE_GROUP(r * width + c, vld1q_u8(levels + r * stride + c));
            }
        }
    } else if (bwl == 3) {
        for (int r = 0; r < rows; r += 2) {
            const uint8x16_t lv = vcombine_u8(vld1_u8(levels + r * stride), vld1_u8(levels + (r + 1) * stride));
            COST_BASE_GROUP(r * 8, lv);
        }
    } else {
        for (int r = 0; r < rows; r += 4) {
            COST_BASE_GROUP(r * 4, load_u8_4x4(levels + r * stride, stride));
        }
    }
#undef COST_BASE_GROUP

    *cost = (int32_t)(vaddlvq_u16(acc[0]) + ((uint32_t)vaddlvq_u16(acc[1]) << 8));
    // each nonzero level added 0xff, i.e. -1
    *nz_count = (int32_t)vaddlvq_u8(vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(nz))));
    return n;
}
