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

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "sad_neon_dotprod.h"
#include "sum_neon.h"

void svt_ext_sad_calculation_8x8_16x16_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                    uint32_t ref_stride, uint32_t* p_best_sad_8x8,
                                                    uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8,
                                                    uint32_t* p_best_mv16x16, uint32_t mv, uint32_t* p_sad16x16,
                                                    uint32_t* p_sad8x8, bool sub_sad) {
    uint32_t   sad16x16;
    uint32x4_t sad;
    uint32x4_t best_sad_vec = vld1q_u32(p_best_sad_8x8);
    uint32x4_t best_mv_vec  = vld1q_u32(p_best_mv8x8);
    uint32x4_t mv_vec       = vdupq_n_u32(mv);

    if (sub_sad) {
        uint32x4_t sad01 = sad_8x4_dual_neon(
            src + 0 * src_stride, 2 * src_stride, ref + 0 * ref_stride, 2 * ref_stride, 2);
        uint32x4_t sad23 = sad_8x4_dual_neon(
            src + 8 * src_stride, 2 * src_stride, ref + 8 * ref_stride, 2 * ref_stride, 2);

        sad = vpaddq_u32(sad01, sad23);
    } else {
        uint32x4_t sad01 = sad_8x8_dual_neon(src + 0 * src_stride, src_stride, ref + 0 * ref_stride, ref_stride, 1);
        uint32x4_t sad23 = sad_8x8_dual_neon(src + 8 * src_stride, src_stride, ref + 8 * ref_stride, ref_stride, 1);

        sad = vpaddq_u32(sad01, sad23);
    }

    uint32x4_t cmp = vcltq_u32(sad, best_sad_vec);
    best_sad_vec   = vbslq_u32(cmp, sad, best_sad_vec);
    best_mv_vec    = vbslq_u32(cmp, mv_vec, best_mv_vec);

    sad16x16 = vaddvq_u32(sad);
    if (sad16x16 < p_best_sad_16x16[0]) {
        p_best_sad_16x16[0] = sad16x16;
        p_best_mv16x16[0]   = mv;
    }

    *p_sad16x16 = sad16x16;
    vst1q_u32(p_sad8x8, sad);
    vst1q_u32(p_best_sad_8x8, best_sad_vec);
    vst1q_u32(p_best_mv8x8, best_mv_vec);
}

// Keeps the (sad, mv index) of b where its sad is strictly lower, so ties keep a (the lower search index).
static inline void min_sad_idx(uint32x4_t* sad_a, uint32x4_t* idx_a, uint32x4_t sad_b, uint32x4_t idx_b) {
    const uint32x4_t lt = vcltq_u32(sad_b, *sad_a);
    *sad_a              = vbslq_u32(lt, sad_b, *sad_a);
    *idx_a              = vbslq_u32(lt, idx_b, *idx_a);
}

// sad[k] holds the four 8x8 SADs of one 16x16 block at search position x_mv + k. The bests are reduced with a tree
// instead of position by position, which keeps the first (lowest-index) minimum like a sequential strict-less scan.
static inline void update_eight_sad_8x8_16x16(const uint32x4_t sad[8], uint32_t mv, uint32_t start_16x16_pos,
                                              uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16,
                                              uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
                                              uint32_t p_eight_sad16x16[16][8]) {
    const uint32_t start_8x8_pos = 4 * start_16x16_pos;
    const uint32_t x_mv          = _MVXT(mv);
    const uint32_t y_mv          = _MVYT(mv);

    // 8x8: per lane minimum over the 8 positions
    uint32x4_t s[4], i[4];
    for (int k = 0; k < 4; k++) {
        s[k] = sad[2 * k];
        i[k] = vdupq_n_u32(2 * k);
        min_sad_idx(&s[k], &i[k], sad[2 * k + 1], vdupq_n_u32(2 * k + 1));
    }
    min_sad_idx(&s[0], &i[0], s[1], i[1]);
    min_sad_idx(&s[2], &i[2], s[3], i[3]);
    min_sad_idx(&s[0], &i[0], s[2], i[2]);
    const uint32x4_t mv_vec       = vorrq_u32(vdupq_n_u32(y_mv << 16),
                                              vandq_u32(vaddq_u32(vdupq_n_u32(x_mv), i[0]), vdupq_n_u32(0xFFFF)));
    uint32x4_t       best_sad_vec = vld1q_u32(p_best_sad_8x8 + start_8x8_pos);
    uint32x4_t       best_mv_vec  = vld1q_u32(p_best_mv8x8 + start_8x8_pos);
    min_sad_idx(&best_sad_vec, &best_mv_vec, s[0], mv_vec);
    vst1q_u32(p_best_sad_8x8 + start_8x8_pos, best_sad_vec);
    vst1q_u32(p_best_mv8x8 + start_8x8_pos, best_mv_vec);

    // 16x16: the sum of the four 8x8 SADs at each position
    const uint32x4_t sad16_lo = vpaddq_u32(vpaddq_u32(sad[0], sad[1]), vpaddq_u32(sad[2], sad[3]));
    const uint32x4_t sad16_hi = vpaddq_u32(vpaddq_u32(sad[4], sad[5]), vpaddq_u32(sad[6], sad[7]));
    vst1q_u32(p_eight_sad16x16[start_16x16_pos], sad16_lo);
    vst1q_u32(p_eight_sad16x16[start_16x16_pos] + 4, sad16_hi);
    const uint32_t min16 = vminvq_u32(vminq_u32(sad16_lo, sad16_hi));
    if (min16 < p_best_sad_16x16[start_16x16_pos]) {
        static const uint32_t idx[8]      = {0, 1, 2, 3, 4, 5, 6, 7};
        const uint32x4_t      eight       = vdupq_n_u32(8);
        const uint32x4_t      m           = vdupq_n_u32(min16);
        const uint32x4_t      k_lo        = vbslq_u32(vceqq_u32(sad16_lo, m), vld1q_u32(idx), eight);
        const uint32x4_t      k_hi        = vbslq_u32(vceqq_u32(sad16_hi, m), vld1q_u32(idx + 4), eight);
        const uint32_t        k           = vminvq_u32(vminq_u32(k_lo, k_hi));
        p_best_sad_16x16[start_16x16_pos] = min16;
        p_best_mv16x16[start_16x16_pos]   = (y_mv << 16) | ((x_mv + k) & 0xFFFF);
    }
}

static inline void svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_1_neon_dotprod(
    uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t mv, uint32_t start_16x16_pos,
    uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
    uint32_t p_eight_sad16x16[16][8]) {
    uint32x4_t sad[8];
    for (int search_index = 0; search_index < 8; search_index++) {
        const uint32x4_t sad01 = sad_8x4_dual_neon(src, 2 * src_stride, ref + search_index, 2 * ref_stride, 2);
        const uint32x4_t sad23 = sad_8x4_dual_neon(
            src + 8 * src_stride, 2 * src_stride, ref + 8 * ref_stride + search_index, 2 * ref_stride, 2);
        sad[search_index] = vpaddq_u32(sad01, sad23);
    }
    update_eight_sad_8x8_16x16(
        sad, mv, start_16x16_pos, p_best_sad_8x8, p_best_sad_16x16, p_best_mv8x8, p_best_mv16x16, p_eight_sad16x16);
}

static inline void svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_0_neon_dotprod(
    uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t mv, uint32_t start_16x16_pos,
    uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
    uint32_t p_eight_sad16x16[16][8]) {
    uint32x4_t sad[8];
    for (int search_index = 0; search_index < 8; search_index++) {
        const uint32x4_t sad01 = sad_8x8_dual_neon(src, src_stride, ref + search_index, ref_stride, 1);
        const uint32x4_t sad23 = sad_8x8_dual_neon(
            src + 8 * src_stride, src_stride, ref + 8 * ref_stride + search_index, ref_stride, 1);
        sad[search_index] = vpaddq_u32(sad01, sad23);
    }
    update_eight_sad_8x8_16x16(
        sad, mv, start_16x16_pos, p_best_sad_8x8, p_best_sad_16x16, p_best_mv8x8, p_best_mv16x16, p_eight_sad16x16);
}

void svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                        uint32_t ref_stride, uint32_t mv, uint32_t* p_best_sad_8x8,
                                                        uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8,
                                                        uint32_t* p_best_mv16x16, uint32_t p_eight_sad16x16[16][8],
                                                        uint32_t p_eight_sad8x8[64][8], bool sub_sad) {
    (void)p_eight_sad8x8;
    static const char offsets[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};

    // Successive calls slide ref by +8 columns over the same rows, so only the second cache line of each row
    // (cols 64..) is new; sub_sad reads only even rows.
    const uint32_t pf_step = sub_sad ? 2 : 1;
    for (uint32_t r = 0; r < 64; r += pf_step) {
        __builtin_prefetch(ref + r * ref_stride + 64, 0, 3);
    }

    //---- 16x16 : 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15
    if (sub_sad) {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                const uint32_t block_index           = 16 * y * src_stride + 16 * x;
                const uint32_t search_position_index = 16 * y * ref_stride + 16 * x;
                svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_1_neon_dotprod(src + block_index,
                                                                               src_stride,
                                                                               ref + search_position_index,
                                                                               ref_stride,
                                                                               mv,
                                                                               offsets[4 * y + x],
                                                                               p_best_sad_8x8,
                                                                               p_best_sad_16x16,
                                                                               p_best_mv8x8,
                                                                               p_best_mv16x16,
                                                                               p_eight_sad16x16);
            }
        }
    } else {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                const uint32_t block_index           = 16 * y * src_stride + 16 * x;
                const uint32_t search_position_index = 16 * y * ref_stride + 16 * x;
                svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_0_neon_dotprod(src + block_index,
                                                                               src_stride,
                                                                               ref + search_position_index,
                                                                               ref_stride,
                                                                               mv,
                                                                               offsets[4 * y + x],
                                                                               p_best_sad_8x8,
                                                                               p_best_sad_16x16,
                                                                               p_best_mv8x8,
                                                                               p_best_mv16x16,
                                                                               p_eight_sad16x16);
            }
        }
    }
}
