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

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "pickrst_neon.h"
#include "utility.h"

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// Keeps the low byte of each sample of `rows` rows of `groups` groups of 16 samples. wide[] tells per row and group
// whether one of its samples does not fit a signed byte; with `margin`, the 16 samples after the last group are
// narrowed as well, and a group is also wide if one of the 4 samples that follow it does not fit. next[] is scratch
// for one row of groups.
static inline void narrow_rows_neon_dotprod(const int16_t* src, const int32_t stride, int32_t rows, int32_t groups,
                                            const int margin, int8_t* dst, uint8_t* wide, const int32_t wide_stride,
                                            uint8_t* const next) {
    const int16x8_t bias = vdupq_n_s16(128);
    groups += margin;
    do {
        for (int32_t c = 0; c < groups; c++) {
            const int16x8_t v0 = vld1q_s16(src + 16 * c);
            const int16x8_t v1 = vld1q_s16(src + 16 * c + 8);
            vst1q_s8(dst + 16 * c, vuzp1q_s8(vreinterpretq_s8_s16(v0), vreinterpretq_s8_s16(v1)));
            // A sample fits a signed byte if the high byte of the sample plus 128 is zero.
            const uint8x16_t high = vuzp2q_u8(vreinterpretq_u8_s16(vaddq_s16(v0, bias)),
                                              vreinterpretq_u8_s16(vaddq_s16(v1, bias)));
            wide[c]               = vmaxvq_u8(high);
            next[c]               = vgetq_lane_u32(vreinterpretq_u32_u8(high), 0) != 0;
        }
        if (margin) {
            for (int32_t c = 0; c < groups - 1; c++) {
                wide[c] |= next[c + 1];
            }
        }
        src += stride;
        dst += stride;
        wide += wide_stride;
    } while (--rows);
}

// Steps 1 and 2 of the 5-tap window, as in highbd_pickrst_neon.c, except that a group of 16 columns is summed from
// byte copies of the two planes if all the samples it is multiplied with fit signed bytes: its source row, and its
// five rows of the reconstruction with the four columns that follow. A product then takes a sixteenth of a dot
// product instruction instead of a quarter of a widening multiply-accumulate. The samples have the average of the
// reconstruction removed, which is what makes most groups of a typical picture fit.
static void stats_edges_win5_highbd_neon_dotprod(const int16_t* const d, const int32_t d_stride, const int16_t* const s,
                                                 const int32_t s_stride, const int32_t width, const int32_t height,
                                                 int64_t* const M, int64_t* const H, EbBitDepth bit_depth) {
    const int32_t     wiener_win  = WIENER_WIN_CHROMA;
    const int32_t     wiener_win2 = wiener_win * wiener_win;
    const int32_t     w16         = width & ~15;
    const int16x8x2_t mask        = vld1q_s16_x2(&(mask_16bit[16]) - width % 16);
    int32_t           i, j, x, y;

    // The byte planes, and per row one byte per group (and one more, in whole vectors).
    const int32_t groups   = w16 >> 4;
    const int32_t g_stride = (groups + 16) & ~15;
    const int32_t d_rows   = height + wiener_win - 1;
    int8_t* const d8       = groups
              ? svt_aom_memalign(
              16, (size_t)d_stride * d_rows + (size_t)s_stride * height + (size_t)g_stride * (d_rows + 2 * height + 1))
              : NULL;
    int8_t* const s8       = d8 ? d8 + d_stride * d_rows : NULL;
    // Tells per row and group whether the group is summed from the bytes.
    const uint8_t* bytes = NULL;
    if (d8) {
        uint8_t* const wide_d = (uint8_t*)s8 + s_stride * height;
        uint8_t* const wide_s = wide_d + g_stride * d_rows;
        uint8_t* const fit    = wide_s + g_stride * height;
        uint8_t* const next   = fit + g_stride * height;

        narrow_rows_neon_dotprod(d, d_stride, d_rows, groups, 1, d8, wide_d, g_stride, next);
        narrow_rows_neon_dotprod(s, s_stride, height, groups, 0, s8, wide_s, g_stride, next);
        for (y = 0; y < height; y++) {
            for (x = 0; x < groups; x += 16) {
                uint8x16_t wide = vld1q_u8(wide_s + y * g_stride + x);
                for (i = 0; i < wiener_win; i++) {
                    wide = vorrq_u8(wide, vld1q_u8(wide_d + (y + i) * g_stride + x));
                }
                vst1q_u8(fit + y * g_stride + x, vceqzq_u8(wide));
            }
        }
        bytes = fit;
    }
    // Samples minus their average are smaller than 2^bit_depth in magnitude, so a product is below
    // 2^(2 * bit_depth). A 32-bit lane of a row accumulator takes at most 4 products per 16 columns, i.e. at most
    // h_allowed * (w16 + 16) / 4 <= 2^(31 - 2 * bit_depth) products before it is widened to 64 bits: its magnitude
    // stays below 2^31 for 8-bit and for 10-bit samples. A dot product adds 4 products per 16 columns to a lane as
    // well, and those are at most 2^14 in magnitude.
    const int32_t num_bit_left = 32 - 1 /* sign */ - 2 * bit_depth /* energy */ + 2 /* SIMD */;
    const int32_t h_allowed    = (1 << num_bit_left) / (w16 + ((w16 != width) ? 16 : 0));

    // Step 1: Calculate the top edge of the whole matrix, i.e., the top
    // edge of each triangle and square on the top row.
    j = 0;
    do {
        const int16_t* s_t                      = s;
        const int16_t* d_t                      = d;
        int32_t        row                      = 0;
        int32_t        height_t                 = 0;
        int64x2_t      sum_m[WIENER_WIN_CHROMA] = {vdupq_n_s64(0)};
        int64x2_t      sum_h[WIENER_WIN_CHROMA] = {vdupq_n_s64(0)};
        int16x8_t      src[2], dgd[2];

        do {
            const int32_t h_t = ((height - height_t) < h_allowed) ? (height - height_t) : h_allowed;
            int32x4_t     row_m[2 * WIENER_WIN_CHROMA] = {vdupq_n_s32(0)};
            int32x4_t     row_h[2 * WIENER_WIN_CHROMA] = {vdupq_n_s32(0)};

            y = h_t;
            do {
                x = 0;
                while (x < w16) {
                    if (bytes && bytes[row * g_stride + (x >> 4)]) {
                        const int8_t* const d8_t = d8 + row * d_stride + x;
                        const int8x16_t     src8 = vld1q_s8(s8 + row * s_stride + x);
                        const int8x16_t     dgd8 = vld1q_s8(d8_t);

                        for (i = 0; i < wiener_win; i++) {
                            const int8x16_t below = vld1q_s8(d8_t + i * d_stride + j);

                            row_m[2 * i] = vdotq_s32(row_m[2 * i], src8, below);
                            row_h[2 * i] = vdotq_s32(row_h[2 * i], dgd8, below);
                        }
                    } else {
                        src[0] = vld1q_s16(s_t + x + 0);
                        src[1] = vld1q_s16(s_t + x + 8);
                        dgd[0] = vld1q_s16(d_t + x + 0);
                        dgd[1] = vld1q_s16(d_t + x + 8);
                        stats_top_win5_highbd_neon(src, dgd, d_t + j + x, d_stride, row_m, row_h);
                    }
                    x += 16;
                }

                if (w16 != width) {
                    src[0] = vld1q_s16(s_t + w16 + 0);
                    src[1] = vld1q_s16(s_t + w16 + 8);
                    dgd[0] = vld1q_s16(d_t + w16 + 0);
                    dgd[1] = vld1q_s16(d_t + w16 + 8);
                    src[0] = vandq_s16(src[0], mask.val[0]);
                    src[1] = vandq_s16(src[1], mask.val[1]);
                    dgd[0] = vandq_s16(dgd[0], mask.val[0]);
                    dgd[1] = vandq_s16(dgd[1], mask.val[1]);
                    stats_top_win5_highbd_neon(src, dgd, d_t + j + w16, d_stride, row_m, row_h);
                }

                s_t += s_stride;
                d_t += d_stride;
                row++;
            } while (--y);

            for (i = 0; i < wiener_win; i++) {
                sum_m[i] = vpadalq_s32(sum_m[i], row_m[2 * i + 0]);
                sum_m[i] = vpadalq_s32(sum_m[i], row_m[2 * i + 1]);
                sum_h[i] = vpadalq_s32(sum_h[i], row_h[2 * i + 0]);
                sum_h[i] = vpadalq_s32(sum_h[i], row_h[2 * i + 1]);
            }

            height_t += h_t;
        } while (height_t < height);

        for (i = 0; i < wiener_win; i++) {
            M[wiener_win * j + i] = vaddvq_s64(sum_m[i]);
            H[wiener_win * j + i] = vaddvq_s64(sum_h[i]);
        }
    } while (++j < wiener_win);

    // Step 2: Calculate the left edge of each square on the top row, for all four squares in one pass over the
    // unit.
    {
        const int16_t* d_t                                                 = d;
        int32_t        row                                                 = 0;
        int32_t        height_t                                            = 0;
        int64x2_t      sum_h[WIENER_WIN_CHROMA - 1][WIENER_WIN_CHROMA - 1] = {{vdupq_n_s64(0)}};
        int16x8_t      dgd[WIENER_WIN_CHROMA - 1][2];

        do {
            const int32_t h_t = ((height - height_t) < h_allowed) ? (height - height_t) : h_allowed;
            int32x4_t     row_h[WIENER_WIN_CHROMA - 1][WIENER_WIN_CHROMA - 1] = {{vdupq_n_s32(0)}};

            y = h_t;
            do {
                x = 0;
                while (x < w16) {
                    if (bytes && bytes[row * g_stride + (x >> 4)]) {
                        const int8_t* const d8_t = d8 + row * d_stride + x;
                        int8x16_t           dgd8[WIENER_WIN_CHROMA - 1];

                        for (j = 1; j < wiener_win; j++) {
                            dgd8[j - 1] = vld1q_s8(d8_t + j);
                        }
                        for (i = 1; i < wiener_win; i++) {
                            const int8x16_t below = vld1q_s8(d8_t + i * d_stride);

                            for (j = 1; j < wiener_win; j++) {
                                row_h[j - 1][i - 1] = vdotq_s32(row_h[j - 1][i - 1], dgd8[j - 1], below);
                            }
                        }
                    } else {
                        for (j = 1; j < wiener_win; j++) {
                            dgd[j - 1][0] = vld1q_s16(d_t + j + x + 0);
                            dgd[j - 1][1] = vld1q_s16(d_t + j + x + 8);
                        }
                        stats_left_win5_highbd_neon(dgd, d_t + x, d_stride, row_h);
                    }
                    x += 16;
                }

                if (w16 != width) {
                    for (j = 1; j < wiener_win; j++) {
                        dgd[j - 1][0] = vandq_s16(vld1q_s16(d_t + j + x + 0), mask.val[0]);
                        dgd[j - 1][1] = vandq_s16(vld1q_s16(d_t + j + x + 8), mask.val[1]);
                    }
                    stats_left_win5_highbd_neon(dgd, d_t + x, d_stride, row_h);
                }

                d_t += d_stride;
                row++;
            } while (--y);

            for (j = 0; j < wiener_win - 1; j++) {
                for (i = 0; i < wiener_win - 1; i++) {
                    sum_h[j][i] = vpadalq_s32(sum_h[j][i], row_h[j][i]);
                }
            }

            height_t += h_t;
        } while (height_t < height);

        for (j = 1; j < wiener_win; j++) {
            for (i = 1; i < wiener_win; i++) {
                H[i * wiener_win2 + j * wiener_win] = vaddvq_s64(sum_h[j - 1][i - 1]);
            }
        }
    }

    svt_aom_free(d8);
}

void svt_av1_compute_stats_highbd_neon_dotprod(int32_t wiener_win, const uint8_t* dgd8, const uint8_t* src8,
                                               int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end,
                                               int32_t dgd_stride, int32_t src_stride, int64_t* M, int64_t* H,
                                               EbBitDepth bit_depth) {
    svt_av1_compute_stats_highbd_edges_neon(wiener_win,
                                            dgd8,
                                            src8,
                                            h_start,
                                            h_end,
                                            v_start,
                                            v_end,
                                            dgd_stride,
                                            src_stride,
                                            M,
                                            H,
                                            bit_depth,
                                            stats_edges_win5_highbd_neon_dotprod);
}
#endif // CONFIG_ENABLE_HIGH_BIT_DEPTH
