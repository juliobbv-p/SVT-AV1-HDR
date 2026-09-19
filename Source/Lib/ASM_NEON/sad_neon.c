/*
* Copyright (c) 2023, Alliance for Open Media. All rights reserved
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
#include "sum_neon.h"

int svt_aom_satd_neon(const TranLow* coeff, int length) {
    const int32x4_t zero  = vdupq_n_s32(0);
    int32x4_t       accum = zero;
    do {
        const int32x4_t src0  = vld1q_s32(&coeff[0]);
        const int32x4_t src8  = vld1q_s32(&coeff[4]);
        const int32x4_t src16 = vld1q_s32(&coeff[8]);
        const int32x4_t src24 = vld1q_s32(&coeff[12]);
        accum                 = vabaq_s32(accum, src0, zero);
        accum                 = vabaq_s32(accum, src8, zero);
        accum                 = vabaq_s32(accum, src16, zero);
        accum                 = vabaq_s32(accum, src24, zero);
        length -= 16;
        coeff += 16;
    } while (length != 0);

    // satd: 26 bits, dynamic range [-32640 * 1024, 32640 * 1024]
    return vaddvq_s32(accum);
}

static AOM_FORCE_INLINE uint32x4_t compute8xh_sad_kernel_dual_neon(uint8_t* restrict src_ptr, uint32_t src_stride,
                                                                   uint8_t* restrict ref_ptr, uint32_t ref_stride,
                                                                   int h) {
    uint16x8_t sum = vdupq_n_u16(0);

    do {
        uint8x16_t s = vld1q_u8(src_ptr);
        uint8x16_t r = vld1q_u8(ref_ptr);

        uint8x16_t abs_diff = vabdq_u8(s, r);

        sum = vpadalq_u8(sum, abs_diff);

        src_ptr += src_stride;
        ref_ptr += ref_stride;
    } while (--h != 0);

    return vpaddlq_u16(sum);
}

/*******************************************
Calculate SAD for 16x16 and its 8x8 sublocks and check if there is an
improvement, if yes keep the best SAD+MV.
*******************************************/
void svt_ext_sad_calculation_8x8_16x16_neon(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                            uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16,
                                            uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16, uint32_t mv,
                                            uint32_t* p_sad16x16, uint32_t* p_sad8x8, bool sub_sad) {
    uint32_t   sad16x16;
    uint32x4_t sad;
    uint32x4_t best_sad_vec = vld1q_u32(p_best_sad_8x8);
    uint32x4_t best_mv_vec  = vld1q_u32(p_best_mv8x8);
    uint32x4_t mv_vec       = vdupq_n_u32(mv);

    if (sub_sad) {
        uint32x4_t sad01 = compute8xh_sad_kernel_dual_neon(
            src + 0 * src_stride + 0, 2 * src_stride, ref + 0 * ref_stride + 0, 2 * ref_stride, 4);
        uint32x4_t sad23 = compute8xh_sad_kernel_dual_neon(
            src + 8 * src_stride + 0, 2 * src_stride, ref + 8 * ref_stride + 0, 2 * ref_stride, 4);

        sad = vpaddq_u32(sad01, sad23);
        sad = vaddq_u32(sad, sad);
    } else {
        uint32x4_t sad01 = compute8xh_sad_kernel_dual_neon(
            src + 0 * src_stride + 0, src_stride, ref + 0 * ref_stride + 0, ref_stride, 8);
        uint32x4_t sad23 = compute8xh_sad_kernel_dual_neon(
            src + 8 * src_stride + 0, src_stride, ref + 8 * ref_stride + 0, ref_stride, 8);

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

/*******************************************
 * svt_ext_eight_sad_calculation_8x8_16x16_neon
 *
 * Computes, for 8 horizontal search positions, the four 8x8 SADs (and their
 * 16x16 sum) of one 16x16 block, updating the running best 8x8/16x16 SAD+MV.
 *
 * The 16 (or 8, when sub-sampling) src rows are loaded once and reused across
 * all 8 search positions; ref is loaded at runtime offset +search_index so the
 * position loop stays rolled. `sub_sad` is passed as a compile-time-constant via
 * AOM_FORCE_INLINE so n/step fold away and the row loops fully unroll.
 *******************************************/
static AOM_FORCE_INLINE void svt_ext_eight_sad_calculation_8x8_16x16_neon(
    uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t mv, uint32_t start_16x16_pos,
    uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
    uint32_t p_eight_sad16x16[16][8], const bool sub_sad) {
    const uint32_t start_8x8_pos = 4 * start_16x16_pos;

    p_best_sad_8x8 += start_8x8_pos;
    p_best_mv8x8 += start_8x8_pos;
    p_best_sad_16x16 += start_16x16_pos;
    p_best_mv16x16 += start_16x16_pos;
    const uint32_t x_mv         = _MVXT(mv);
    const uint32_t y_mv         = _MVYT(mv);
    uint32x4_t     best_sad_vec = vld1q_u32(p_best_sad_8x8);
    uint32x4_t     best_mv_vec  = vld1q_u32(p_best_mv8x8);

    // Sub-sampling sums every other row (4 per 8x8 half, x2 at the end); the full
    // SAD sums all 8 rows. Both reuse each src row across the 8 search positions.
    const int      n    = sub_sad ? 4 : 8;
    const uint32_t step = sub_sad ? 2 : 1;

    uint8x16_t st[8], sb[8];
    for (int r = 0; r < n; r++) {
        st[r] = vld1q_u8(src + (r * step) * src_stride);
        sb[r] = vld1q_u8(src + (8 + r * step) * src_stride);
    }

    for (int search_index = 0; search_index < 8; search_index++) {
        const uint8_t* rt = ref + search_index;

        uint16x8_t sum_t = vdupq_n_u16(0);
        uint16x8_t sum_b = vdupq_n_u16(0);
        for (int r = 0; r < n; r++) {
            sum_t = vpadalq_u8(sum_t, vabdq_u8(st[r], vld1q_u8(rt + (r * step) * ref_stride)));
            sum_b = vpadalq_u8(sum_b, vabdq_u8(sb[r], vld1q_u8(rt + (8 + r * step) * ref_stride)));
        }

        uint32x4_t sad = vpaddq_u32(vpaddlq_u16(sum_t), vpaddlq_u16(sum_b));
        if (sub_sad) {
            sad = vaddq_u32(sad, sad);
        }

        const uint32_t tmp_mv = (y_mv << 16) | ((x_mv + search_index) & 0xFFFF);
        uint32x4_t     mv_vec = vdupq_n_u32(tmp_mv);
        uint32x4_t     cmp    = vcltq_u32(sad, best_sad_vec);
        best_sad_vec          = vbslq_u32(cmp, sad, best_sad_vec);
        best_mv_vec           = vbslq_u32(cmp, mv_vec, best_mv_vec);

        uint32_t sad16x16 = p_eight_sad16x16[start_16x16_pos][search_index] = vaddvq_u32(sad);
        if (sad16x16 < p_best_sad_16x16[0]) {
            p_best_sad_16x16[0] = sad16x16;
            p_best_mv16x16[0]   = tmp_mv;
        }
    }
    vst1q_u32(p_best_sad_8x8, best_sad_vec);
    vst1q_u32(p_best_mv8x8, best_mv_vec);
}

void svt_ext_all_sad_calculation_8x8_16x16_neon(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                                uint32_t mv, uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16,
                                                uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
                                                uint32_t p_eight_sad16x16[16][8], uint32_t p_eight_sad8x8[64][8],
                                                bool sub_sad) {
    (void)p_eight_sad8x8;
    static const char offsets[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};

    // Prefetch the cold part of the ref window. Successive ME calls slide the ref
    // base by +8 columns over the same rows (see open_loop_me_get_eight_search_
    // point_results_block), so the first cache line of each row (cols 0..63) is
    // almost always still warm from the previous call; the newly-exposed columns
    // land in the second line (cols 64..). So prefetch only that +64 line. src is
    // the current SB (hot, reused across searches) and is not prefetched.
    // sub_sad reads only even rows, so step by 2 to skip unused odd-row lines.
    const uint32_t pf_step = sub_sad ? 2 : 1;
    for (uint32_t r = 0; r < 64; r += pf_step) {
        __builtin_prefetch(ref + r * ref_stride + 64, 0, 3);
    }

    //---- 16x16 : 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            const uint32_t block_index           = 16 * y * src_stride + 16 * x;
            const uint32_t search_position_index = 16 * y * ref_stride + 16 * x;
            // Pass sub_sad as a literal so the impl specializes for each case.
            if (sub_sad) {
                svt_ext_eight_sad_calculation_8x8_16x16_neon(src + block_index,
                                                             src_stride,
                                                             ref + search_position_index,
                                                             ref_stride,
                                                             mv,
                                                             offsets[4 * y + x],
                                                             p_best_sad_8x8,
                                                             p_best_sad_16x16,
                                                             p_best_mv8x8,
                                                             p_best_mv16x16,
                                                             p_eight_sad16x16,
                                                             true);
            } else {
                svt_ext_eight_sad_calculation_8x8_16x16_neon(src + block_index,
                                                             src_stride,
                                                             ref + search_position_index,
                                                             ref_stride,
                                                             mv,
                                                             offsets[4 * y + x],
                                                             p_best_sad_8x8,
                                                             p_best_sad_16x16,
                                                             p_best_mv8x8,
                                                             p_best_mv16x16,
                                                             p_eight_sad16x16,
                                                             false);
            }
        }
    }
}
