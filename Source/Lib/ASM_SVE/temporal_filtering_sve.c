/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <assert.h>
#include <math.h>
#include <arm_neon.h>
#include <arm_sve.h>
#include "neon_sve_bridge.h"

#include "definitions.h"
#include "mem_neon.h"
#include "temporal_filtering.h"
#include "utility.h"

static void process_block_hbd_sve(int h, int w, uint16_t* buff_hbd_start, uint32_t* accum, uint16_t* count,
                                  uint32_t stride) {
    do {
        int width = w;
        do {
            // buff_lbd_start[pos] = (uint8_t)OD_DIVU(accum[k] + (count[k] >> 1), count[k]);
            // buff_lbd_start[pos] = (uint8_t)((accum[k] + (count[k] >> 1))/ count[k]);
            uint32x4_t accum0     = vld1q_u32(accum);
            uint32x4_t accum1     = vld1q_u32(accum + 4);
            uint16x8_t count_u16  = vld1q_u16(count);
            uint32x4_t count0_u32 = vmovl_u16(vget_low_u16(count_u16));
            uint32x4_t count1_u32 = vmovl_u16(vget_high_u16(count_u16));

            // accum[k] + (count[k] >> 1)
            accum0 = vsraq_n_u32(accum0, count0_u32, 1);
            accum1 = vsraq_n_u32(accum1, count1_u32, 1);

            uint32x4_t d0  = svt_div_u32(accum0, count0_u32);
            uint32x4_t d1  = svt_div_u32(accum1, count1_u32);
            uint16x8_t d01 = vcombine_u16(vmovn_u32(d0), vmovn_u32(d1));

            vst1q_u16(buff_hbd_start, d01);

            accum += 8;
            count += 8;
            buff_hbd_start += 8;
            width -= 8;
        } while (width != 0);
        buff_hbd_start += stride;
    } while (--h != 0);
}

static void process_block_lbd_sve(int h, int w, uint8_t* buff_lbd_start, uint32_t* accum, uint16_t* count,
                                  uint32_t stride) {
    do {
        int width = w;
        do {
            // buff_lbd_start[pos] = (uint8_t)OD_DIVU(accum[k] + (count[k] >> 1), count[k]);
            // buff_lbd_start[pos] = (uint8_t)((accum[k] + (count[k] >> 1))/ count[k]);
            uint32x4_t accum0     = vld1q_u32(accum);
            uint32x4_t accum1     = vld1q_u32(accum + 4);
            uint16x8_t count_u16  = vld1q_u16(count);
            uint32x4_t count0_u32 = vmovl_u16(vget_low_u16(count_u16));
            uint32x4_t count1_u32 = vmovl_u16(vget_high_u16(count_u16));

            // accum[k] + (count[k] >> 1)
            accum0 = vsraq_n_u32(accum0, count0_u32, 1);
            accum1 = vsraq_n_u32(accum1, count1_u32, 1);

            uint32x4_t d0  = svt_div_u32(accum0, count0_u32);
            uint32x4_t d1  = svt_div_u32(accum1, count1_u32);
            uint16x8_t d01 = vcombine_u16(vmovn_u32(d0), vmovn_u32(d1));

            svuint16_t d01_sve = svset_neonq_u16(svundef_u16(), d01);
            svst1b(svptrue_pat_b8(SV_VL16), buff_lbd_start, d01_sve);

            accum += 8;
            count += 8;
            buff_lbd_start += 8;
            width -= 8;
        } while (width != 0);
        buff_lbd_start += stride;
    } while (--h != 0);
}

void svt_aom_get_final_filtered_pixels_sve(MeContext* me_ctx, EbByte* src_center_ptr_start,
                                           uint16_t** altref_buffer_highbd_start, uint32_t** accum, uint16_t** count,
                                           const uint32_t* stride, int blk_y_src_offset, int blk_ch_src_offset,
                                           uint16_t blk_width_ch, uint16_t blk_height_ch, bool is_highbd) {
    assert(blk_width_ch % 16 == 0);
    assert(TF_BW % 16 == 0);

    if (!is_highbd) {
        //Process luma
        process_block_lbd_sve(TF_BH,
                              TF_BW,
                              &src_center_ptr_start[PLANE_Y][blk_y_src_offset],
                              accum[PLANE_Y],
                              count[PLANE_Y],
                              stride[PLANE_Y] - TF_BW);
        // Process chroma
        if (me_ctx->tf_chroma) {
            process_block_lbd_sve(blk_height_ch,
                                  blk_width_ch,
                                  &src_center_ptr_start[PLANE_U][blk_ch_src_offset],
                                  accum[PLANE_U],
                                  count[PLANE_U],
                                  stride[PLANE_U] - blk_width_ch);
            process_block_lbd_sve(blk_height_ch,
                                  blk_width_ch,
                                  &src_center_ptr_start[PLANE_V][blk_ch_src_offset],
                                  accum[PLANE_V],
                                  count[PLANE_V],
                                  stride[PLANE_V] - blk_width_ch);
        }
    } else {
        // Process luma
        process_block_hbd_sve(TF_BH,
                              TF_BW,
                              &altref_buffer_highbd_start[PLANE_Y][blk_y_src_offset],
                              accum[PLANE_Y],
                              count[PLANE_Y],
                              stride[PLANE_Y] - TF_BW);
        // Process chroma
        if (me_ctx->tf_chroma) {
            process_block_hbd_sve(blk_height_ch,
                                  blk_width_ch,
                                  &altref_buffer_highbd_start[PLANE_U][blk_ch_src_offset],
                                  accum[PLANE_U],
                                  count[PLANE_U],
                                  stride[PLANE_U] - blk_width_ch);
            process_block_hbd_sve(blk_height_ch,
                                  blk_width_ch,
                                  &altref_buffer_highbd_start[PLANE_V][blk_ch_src_offset],
                                  accum[PLANE_V],
                                  count[PLANE_V],
                                  stride[PLANE_V] - blk_width_ch);
        }
    }
}

static inline void gradient_coherence_accumulate_row_8_sve(uint8x8_t row_r, uint8x8_t row_l, uint8x8_t row_d,
                                                           uint8x8_t row_u, uint32x2_t* acc_xx, uint32x2_t* acc_yy,
                                                           int64x2_t* acc_xy) {
    const uint8x8_t gx_u8 = vabd_u8(row_r, row_l);
    const uint8x8_t gy_u8 = vabd_u8(row_d, row_u);
    *acc_xx               = vdot_u32(*acc_xx, gx_u8, gx_u8);
    *acc_yy               = vdot_u32(*acc_yy, gy_u8, gy_u8);

    const int16x8_t gx_s16 = vreinterpretq_s16_u16(vsubl_u8(row_r, row_l));
    const int16x8_t gy_s16 = vreinterpretq_s16_u16(vsubl_u8(row_d, row_u));
    *acc_xy                = svt_sdotq_s16(*acc_xy, gx_s16, gy_s16);
}

float svt_vmaf_compute_gradient_coherence_sve(const uint8_t* src, int width, int height, int stride) {
    assert(width % 8 == 0 && "width must be multiple of 8");

    double weighted_coh = 0.0;
    double weight_sum   = 0.0;

    for (int by = 1; by < height - 1; by += 16) {
        const int y_end = (by + 16 < height - 1) ? by + 16 : height - 1;

        int bx = 1;
        for (; bx + 16 <= width - 1; bx += 16) {
            uint32x4_t acc_xx  = vdupq_n_u32(0);
            uint32x4_t acc_yy  = vdupq_n_u32(0);
            int64x2_t  acc_xy0 = vdupq_n_s64(0);
            int64x2_t  acc_xy1 = vdupq_n_s64(0);

            const uint8_t* row  = src + (size_t)by * stride;
            const uint8_t* up   = src + (size_t)(by - 1) * stride;
            const uint8_t* down = src + (size_t)(by + 1) * stride;

            int y = by;
            do {
                const uint8x16_t row_r = vld1q_u8(row + bx + 1);
                const uint8x16_t row_l = vld1q_u8(row + bx - 1);
                const uint8x16_t row_d = vld1q_u8(down + bx);
                const uint8x16_t row_u = vld1q_u8(up + bx);

                const uint8x16_t gx_u8 = vabdq_u8(row_r, row_l);
                const uint8x16_t gy_u8 = vabdq_u8(row_d, row_u);
                acc_xx                 = vdotq_u32(acc_xx, gx_u8, gx_u8);
                acc_yy                 = vdotq_u32(acc_yy, gy_u8, gy_u8);

                const int16x8_t gx_s16_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(row_r), vget_low_u8(row_l)));
                const int16x8_t gx_s16_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(row_r), vget_high_u8(row_l)));
                const int16x8_t gy_s16_lo = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(row_d), vget_low_u8(row_u)));
                const int16x8_t gy_s16_hi = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(row_d), vget_high_u8(row_u)));
                acc_xy0                   = svt_sdotq_s16(acc_xy0, gx_s16_lo, gy_s16_lo);
                acc_xy1                   = svt_sdotq_s16(acc_xy1, gx_s16_hi, gy_s16_hi);

                row += stride;
                up += stride;
                down += stride;
            } while (++y != y_end);

            const double xx = (double)vaddvq_u32(acc_xx);
            const double yy = (double)vaddvq_u32(acc_yy);
            const double xy = (double)vaddvq_s64(vaddq_s64(acc_xy0, acc_xy1));
            weighted_coh += sqrtf((float)((xx - yy) * (xx - yy) + 4.0 * xy * xy));
            weight_sum += xx + yy;
        }

        // Tail can be either 6 pixels or 8+6 pixels.
        uint32x2_t acc_xx = vdup_n_u32(0);
        uint32x2_t acc_yy = vdup_n_u32(0);
        int64x2_t  acc_xy = vdupq_n_s64(0);
        if (bx + 8 < width - 1) {
            const uint8_t* row  = src + (size_t)by * stride;
            const uint8_t* up   = src + (size_t)(by - 1) * stride;
            const uint8_t* down = src + (size_t)(by + 1) * stride;

            int y = by;
            do {
                const uint8x8_t row_r = vld1_u8(row + bx + 1);
                const uint8x8_t row_l = vld1_u8(row + bx - 1);
                const uint8x8_t row_d = vld1_u8(down + bx);
                const uint8x8_t row_u = vld1_u8(up + bx);

                gradient_coherence_accumulate_row_8_sve(row_r, row_l, row_d, row_u, &acc_xx, &acc_yy, &acc_xy);

                row += stride;
                up += stride;
                down += stride;
            } while (++y != y_end);

            bx += 8;
        }
        if (bx < width - 1) {
            const svbool_t pred_tail = svwhilelt_b8_s32(0, width - 1 - bx);
            const uint8_t* row       = src + (size_t)by * stride;
            const uint8_t* up        = src + (size_t)(by - 1) * stride;
            const uint8_t* down      = src + (size_t)(by + 1) * stride;

            int y = by;
            do {
                const uint8x8_t row_r = vget_low_u8(svget_neonq_u8(svld1_u8(pred_tail, row + bx + 1)));
                const uint8x8_t row_l = vget_low_u8(svget_neonq_u8(svld1_u8(pred_tail, row + bx - 1)));
                const uint8x8_t row_d = vget_low_u8(svget_neonq_u8(svld1_u8(pred_tail, down + bx)));
                const uint8x8_t row_u = vget_low_u8(svget_neonq_u8(svld1_u8(pred_tail, up + bx)));

                gradient_coherence_accumulate_row_8_sve(row_r, row_l, row_d, row_u, &acc_xx, &acc_yy, &acc_xy);

                row += stride;
                up += stride;
                down += stride;
            } while (++y != y_end);

            const double xx = (double)vaddv_u32(acc_xx);
            const double yy = (double)vaddv_u32(acc_yy);
            const double xy = (double)vaddvq_s64(acc_xy);
            weighted_coh += sqrtf((float)((xx - yy) * (xx - yy) + 4.0 * xy * xy));
            weight_sum += xx + yy;
        }
    }
    if (weight_sum <= 0.0) {
        return 1.0f;
    }
    return (float)(weighted_coh / weight_sum);
}
