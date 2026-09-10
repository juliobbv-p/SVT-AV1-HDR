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

#include "common_dsp_rtcd.h"
#include "qm_psnr.h"

static INLINE uint64x2_t qm_square_sum(int32x4_t value, int32x4_t weight) {
    const int32x4_t  weighted = vmulq_s32(value, weight);
    const uint64x2_t lo       = vreinterpretq_u64_s64(vmull_s32(vget_low_s32(weighted), vget_low_s32(weighted)));
    const uint64x2_t hi       = vreinterpretq_u64_s64(vmull_high_s32(weighted, weighted));
    // Round each square before summing, exactly as svt_qm_coeff_dist does.
    return vaddq_u64(vrshrq_n_u64(lo, 2 * AOM_QM_BITS), vrshrq_n_u64(hi, 2 * AOM_QM_BITS));
}

// Weighted magnitudes fit in unsigned 16 bits, so each widening multiply
// produces four 32-bit squares instead of two 64-bit squares. Round each square
// before summing: the maximum rounded square is 4194176, so adding two in a
// 32-bit lane is safe. Widen into the persistent 64-bit accumulator.
static INLINE uint32x4_t qm_square_sum16(uint16x8_t weighted) {
    const uint32x4_t lo = vmull_u16(vget_low_u16(weighted), vget_low_u16(weighted));
    const uint32x4_t hi = vmull_high_u16(weighted, weighted);
    return vrsraq_n_u32(vrshrq_n_u32(lo, 2 * AOM_QM_BITS), hi, 2 * AOM_QM_BITS);
}

static INLINE void qm_distortion_neon(const int32_t* coeff, const int32_t* recon, uint32_t stride, uint32_t width,
                                      uint32_t height, uint64_t* distortion, bool has_coeff, const QmVal* weights,
                                      uint32_t weight_stride) {
    uint64x2_t energy0 = vdupq_n_u64(0), energy1 = vdupq_n_u64(0);
    uint64x2_t error0 = vdupq_n_u64(0), error1 = vdupq_n_u64(0);
    uint64_t   energy_tail = 0, error_tail = 0;
    for (uint32_t y = 0; y < height; ++y) {
        const int32_t* c = coeff + y * stride;
        const int32_t* r = has_coeff ? recon + y * stride : NULL;
        const QmVal*   w = weights + y * weight_stride;
        uint32_t       x = 0;
        for (; x + 16 <= width; x += 16) {
            const int32x4_t c0 = vld1q_s32(c + x);
            const int32x4_t c1 = vld1q_s32(c + x + 4);
            const int32x4_t c2 = vld1q_s32(c + x + 8);
            const int32x4_t c3 = vld1q_s32(c + x + 12);
            // Saturation prevents an overflowing difference from wrapping into
            // the accepted range. Saturated nonzero-weight groups fall back to C.
            const int32x4_t d0        = has_coeff ? vqsubq_s32(c0, vld1q_s32(r + x)) : vdupq_n_s32(0);
            const int32x4_t d1        = has_coeff ? vqsubq_s32(c1, vld1q_s32(r + x + 4)) : vdupq_n_s32(0);
            const int32x4_t d2        = has_coeff ? vqsubq_s32(c2, vld1q_s32(r + x + 8)) : vdupq_n_s32(0);
            const int32x4_t d3        = has_coeff ? vqsubq_s32(c3, vld1q_s32(r + x + 12)) : vdupq_n_s32(0);
            uint32x4_t      magnitude = vreinterpretq_u32_s32(vabsq_s32(c0));
            magnitude                 = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(c1)));
            magnitude                 = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(c2)));
            magnitude                 = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(c3)));
            if (has_coeff) {
                magnitude = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(d0)));
                magnitude = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(d1)));
                magnitude = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(d2)));
                magnitude = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(d3)));
            }
            const uint32_t   max_magnitude = vmaxvq_u32(magnitude);
            const uint8x16_t weights16     = vld1q_u8(w + x);
            // Amortize the range check over sixteen coefficients. This bound
            // keeps both weighted source and error magnitudes in unsigned 16 bits.
            // All-zero weights are exact for any input magnitude.
            if ((uint64_t)max_magnitude * vmaxvq_u8(weights16) <= UINT16_MAX) {
                {
                    const uint16x8_t w16    = vmovl_u8(vget_low_u8(weights16));
                    const uint16x8_t c16    = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(c0))),
                                                        vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(c1))));
                    const uint16x8_t energy = vmulq_u16(c16, w16);
                    energy0                 = vpadalq_u32(energy0, qm_square_sum16(energy));
                    if (has_coeff) {
                        const uint16x8_t d16   = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(d0))),
                                                            vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(d1))));
                        const uint16x8_t error = vmulq_u16(d16, w16);
                        error0                 = vpadalq_u32(error0, qm_square_sum16(error));
                    }
                }
                {
                    const uint16x8_t w16    = vmovl_high_u8(weights16);
                    const uint16x8_t c16    = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(c2))),
                                                        vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(c3))));
                    const uint16x8_t energy = vmulq_u16(c16, w16);
                    energy1                 = vpadalq_u32(energy1, qm_square_sum16(energy));
                    if (has_coeff) {
                        const uint16x8_t d16   = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(d2))),
                                                            vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(d3))));
                        const uint16x8_t error = vmulq_u16(d16, w16);
                        error1                 = vpadalq_u32(error1, qm_square_sum16(error));
                    }
                }
                continue;
            }
            // Saturated differences and INT32_MIN fail this bound. Recompute
            // them in scalar 64-bit arithmetic; otherwise |c*w| and |d*w| fit
            // signed 32 bits for every uint8 weight.
            if (max_magnitude >= (1u << 22)) {
                for (unsigned j = 0; j < 16; ++j) {
                    energy_tail += svt_qm_coeff_dist(c[x + j], 0, 0, w, x + j);
                    if (has_coeff) {
                        error_tail += svt_qm_coeff_dist(c[x + j], r[x + j], 0, w, x + j);
                    }
                }
                continue;
            }
            {
                const uint16x8_t w16 = vmovl_u8(vget_low_u8(weights16));
                const int32x4_t  w0  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w16)));
                const int32x4_t  w1  = vreinterpretq_s32_u32(vmovl_high_u16(w16));
                energy0              = vaddq_u64(energy0, qm_square_sum(c0, w0));
                energy1              = vaddq_u64(energy1, qm_square_sum(c1, w1));
                if (has_coeff) {
                    error0 = vaddq_u64(error0, qm_square_sum(d0, w0));
                    error1 = vaddq_u64(error1, qm_square_sum(d1, w1));
                }
            }
            {
                const uint16x8_t w16 = vmovl_high_u8(weights16);
                const int32x4_t  w0  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w16)));
                const int32x4_t  w1  = vreinterpretq_s32_u32(vmovl_high_u16(w16));
                energy0              = vaddq_u64(energy0, qm_square_sum(c2, w0));
                energy1              = vaddq_u64(energy1, qm_square_sum(c3, w1));
                if (has_coeff) {
                    error0 = vaddq_u64(error0, qm_square_sum(d2, w0));
                    error1 = vaddq_u64(error1, qm_square_sum(d3, w1));
                }
            }
        }
        for (; x + 8 <= width; x += 8) {
            const int32x4_t c0 = vld1q_s32(c + x), c1 = vld1q_s32(c + x + 4);
            const int32x4_t d0        = has_coeff ? vqsubq_s32(c0, vld1q_s32(r + x)) : vdupq_n_s32(0);
            const int32x4_t d1        = has_coeff ? vqsubq_s32(c1, vld1q_s32(r + x + 4)) : vdupq_n_s32(0);
            uint32x4_t      magnitude = vmaxq_u32(vreinterpretq_u32_s32(vabsq_s32(c0)),
                                             vreinterpretq_u32_s32(vabsq_s32(c1)));
            if (has_coeff) {
                magnitude = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(d0)));
                magnitude = vmaxq_u32(magnitude, vreinterpretq_u32_s32(vabsq_s32(d1)));
            }
            const uint32_t   max_magnitude = vmaxvq_u32(magnitude);
            const uint8x8_t  weights8      = vld1_u8(w + x);
            const uint16x8_t w16           = vmovl_u8(weights8);
            // Apply the same unsigned-halfword bound to the eight-element tail.
            if ((uint64_t)max_magnitude * vmaxv_u8(weights8) <= UINT16_MAX) {
                const uint16x8_t c16    = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(c0))),
                                                    vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(c1))));
                const uint16x8_t energy = vmulq_u16(c16, w16);
                energy0                 = vpadalq_u32(energy0, qm_square_sum16(energy));
                if (has_coeff) {
                    const uint16x8_t d16   = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(d0))),
                                                        vmovn_u32(vreinterpretq_u32_s32(vabsq_s32(d1))));
                    const uint16x8_t error = vmulq_u16(d16, w16);
                    error0                 = vpadalq_u32(error0, qm_square_sum16(error));
                }
                continue;
            }
            // Preserve the scalar contract for larger magnitudes and saturated
            // differences. Smaller source/error products fit in signed 32 bits.
            if (max_magnitude >= (1u << 22)) {
                for (unsigned j = 0; j < 8; ++j) {
                    energy_tail += svt_qm_coeff_dist(c[x + j], 0, 0, w, x + j);
                    if (has_coeff) {
                        error_tail += svt_qm_coeff_dist(c[x + j], r[x + j], 0, w, x + j);
                    }
                }
                continue;
            }
            const int32x4_t w0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(w16)));
            const int32x4_t w1 = vreinterpretq_s32_u32(vmovl_high_u16(w16));
            energy0            = vaddq_u64(energy0, qm_square_sum(c0, w0));
            energy1            = vaddq_u64(energy1, qm_square_sum(c1, w1));
            if (has_coeff) {
                error0 = vaddq_u64(error0, qm_square_sum(d0, w0));
                error1 = vaddq_u64(error1, qm_square_sum(d1, w1));
            }
        }
        // Width four and partial rows remain exact without loading beyond the
        // weight row; the preceding loops handle sixteen or eight at a time.
        for (; x < width; ++x) {
            energy_tail += svt_qm_coeff_dist(c[x], 0, 0, w, x);
            if (has_coeff) {
                error_tail += svt_qm_coeff_dist(c[x], r[x], 0, w, x);
            }
        }
    }
    const uint64_t energy            = vaddvq_u64(vaddq_u64(energy0, energy1)) + energy_tail;
    distortion[DIST_CALC_PREDICTION] = energy;
    distortion[DIST_CALC_RESIDUAL]   = has_coeff ? vaddvq_u64(vaddq_u64(error0, error1)) + error_tail : energy;
}

void svt_qm_full_distortion_neon(const int32_t* coeff, const int32_t* recon, uint32_t stride, uint32_t width,
                                 uint32_t height, uint64_t* distortion, uint32_t eob, const QmVal* weights,
                                 uint32_t weight_stride) {
    // Normal transform buffers are packed. Flatten them so 4-wide transforms
    // also use the sixteen-coefficient loop, without any overread across strides.
    if (stride == width && weight_stride == width) {
        width *= height;
        height = 1;
    }
    if (!weights) {
        svt_qm_full_distortion_c(coeff, recon, stride, width, height, distortion, eob, weights, weight_stride);
    } else if (eob) {
        qm_distortion_neon(coeff, recon, stride, width, height, distortion, true, weights, weight_stride);
    } else {
        qm_distortion_neon(coeff, recon, stride, width, height, distortion, false, weights, weight_stride);
    }
}
