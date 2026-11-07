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
#include <stdlib.h>

#include "common_dsp_rtcd.h"
#include "hadamard_path_neon.h"

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// Energy-only transform of nonnegative 10-bit pixels. The horizontal stages
// commute without rounding. Processing distances 4, 2, 1 lets UMAXP replace
// the final transposes; SATD is independent of coefficient order.
static inline uint32x4_t highbd_psy_hadamard_h_pass_neon(int16x8_t s[8]) {
    int16x8_t a0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(s[0]), vreinterpretq_s64_s16(s[1])));
    int16x8_t a1 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(s[0]), vreinterpretq_s64_s16(s[1])));
    int16x8_t a2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(s[2]), vreinterpretq_s64_s16(s[3])));
    int16x8_t a3 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(s[2]), vreinterpretq_s64_s16(s[3])));
    int16x8_t a4 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(s[4]), vreinterpretq_s64_s16(s[5])));
    int16x8_t a5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(s[4]), vreinterpretq_s64_s16(s[5])));
    int16x8_t a6 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s16(s[6]), vreinterpretq_s64_s16(s[7])));
    int16x8_t a7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s16(s[6]), vreinterpretq_s64_s16(s[7])));

    int16x8_t b0 = vaddq_s16(a0, a1);
    int16x8_t b1 = vsubq_s16(a0, a1);
    int16x8_t b2 = vaddq_s16(a2, a3);
    int16x8_t b3 = vsubq_s16(a2, a3);
    int16x8_t b4 = vaddq_s16(a4, a5);
    int16x8_t b5 = vsubq_s16(a4, a5);
    int16x8_t b6 = vaddq_s16(a6, a7);
    int16x8_t b7 = vsubq_s16(a6, a7);

    a0 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a1 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b0), vreinterpretq_s32_s16(b1)));
    a2 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
    a3 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b2), vreinterpretq_s32_s16(b3)));
    a4 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
    a5 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b4), vreinterpretq_s32_s16(b5)));
    a6 = vreinterpretq_s16_s32(vtrn1q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));
    a7 = vreinterpretq_s16_s32(vtrn2q_s32(vreinterpretq_s32_s16(b6), vreinterpretq_s32_s16(b7)));

    b0 = vabsq_s16(vaddq_s16(a0, a1));
    b1 = vabdq_s16(a0, a1);
    b2 = vabsq_s16(vaddq_s16(a2, a3));
    b3 = vabdq_s16(a2, a3);
    b4 = vabsq_s16(vaddq_s16(a4, a5));
    b5 = vabdq_s16(a4, a5);
    b6 = vabsq_s16(vaddq_s16(a6, a7));
    b7 = vabdq_s16(a6, a7);

    // Partial butterflies are bounded by 32 * 1023. Each four-term narrow
    // reduction below is bounded by 64 * 1023; the next addition widens.
    const uint16x8_t max[4] = {
        vpmaxq_u16(vreinterpretq_u16_s16(b0), vreinterpretq_u16_s16(b1)),
        vpmaxq_u16(vreinterpretq_u16_s16(b2), vreinterpretq_u16_s16(b3)),
        vpmaxq_u16(vreinterpretq_u16_s16(b4), vreinterpretq_u16_s16(b5)),
        vpmaxq_u16(vreinterpretq_u16_s16(b6), vreinterpretq_u16_s16(b7)),
    };
    return horizontal_add_4d_u16x8(max);
}

static inline void highbd_psy_hadamard_8x8_neon(const uint16_t* src, ptrdiff_t stride, uint32x4_t* satd,
                                                int32x4_t* dc) {
    int16x8_t a[8];
    load_s16_8x8((const int16_t*)src, stride, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
    hadamard_8x8_v_pass_neon(a);
    *dc   = vpaddlq_s16(a[0]);
    *satd = highbd_psy_hadamard_h_pass_neon(a);
}

static int highbd_psy_energy_8x8_neon(const uint16_t* src, ptrdiff_t src_stride) {
    int32x4_t  dc;
    uint32x4_t sum;
    highbd_psy_hadamard_8x8_neon(src, src_stride, &sum, &dc);

    const int satd   = vaddvq_u32(sum) << 1;
    const int dc_sum = vaddvq_s32(dc);
    return ((satd + 2) >> 2) - ((dc_sum + 2) >> 2);
}

static int32x4_t highbd_psy_energy_16x16_neon(const uint16_t* src, ptrdiff_t src_stride) {
    int32x4_t  dc[4];
    uint32x4_t sum[4];

    highbd_psy_hadamard_8x8_neon(src + 0 * src_stride + 0, src_stride, &sum[0], &dc[0]);
    highbd_psy_hadamard_8x8_neon(src + 0 * src_stride + 8, src_stride, &sum[1], &dc[1]);
    highbd_psy_hadamard_8x8_neon(src + 8 * src_stride + 0, src_stride, &sum[2], &dc[2]);
    highbd_psy_hadamard_8x8_neon(src + 8 * src_stride + 8, src_stride, &sum[3], &dc[3]);

    dc[0] = vpaddq_s32(dc[0], dc[1]);
    dc[2] = vpaddq_s32(dc[2], dc[3]);
    dc[0] = vpaddq_s32(dc[0], dc[2]);

    sum[0] = vpaddq_u32(sum[0], sum[1]);
    sum[2] = vpaddq_u32(sum[2], sum[3]);
    sum[0] = vpaddq_u32(sum[0], sum[2]);

    const int32x4_t dc_energy   = vrshrq_n_s32(dc[0], 2);
    const int32x4_t satd_energy = vreinterpretq_s32_u32(vrshrq_n_u32(sum[0], 1));

    return vsubq_s32(satd_energy, dc_energy);
}

uint64_t svt_psy_distortion_hbd_neon(const uint16_t* input, const uint32_t input_stride, const uint16_t* recon,
                                     const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width % 16 == 0 && height % 16 == 0) {
        int32x4_t energy_gap_16x16 = vdupq_n_s32(0);

        for (uint32_t j = 0; j < height; j += 16) {
            for (uint32_t i = 0; i < width; i += 16) {
                const int32x4_t input_energy = highbd_psy_energy_16x16_neon(input + j * input_stride + i, input_stride);
                const int32x4_t recon_energy = highbd_psy_energy_16x16_neon(recon + j * recon_stride + i, recon_stride);

                energy_gap_16x16 = vabaq_s32(energy_gap_16x16, input_energy, recon_energy);
            }
        }

        energy_gap += (uint64_t)vaddlvq_s32(energy_gap_16x16);

    } else if (width >= 8 && height >= 8) {
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const int input_energy = highbd_psy_energy_8x8_neon(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_8x8_neon(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) {
            for (uint32_t i = 0; i < width; i += 4) {
                const int input_energy = highbd_psy_energy_4x4_neon(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_4x4_neon(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    return energy_gap << 2;
}
#endif
