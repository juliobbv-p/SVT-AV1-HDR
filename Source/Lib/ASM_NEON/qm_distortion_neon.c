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

static AOM_FORCE_INLINE void qm_distortion_neon(const int32_t* coeff, const int32_t* recon, uint32_t stride,
                                                uint32_t width, uint32_t height, uint64_t* distortion, bool has_coeff,
                                                const QmVal* weights, uint32_t weight_stride) {
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

// Rounded squares of eight weighted magnitudes (unsigned 16 bits each), added to two accumulators of 32-bit lanes
// (one accumulate per accumulator and call, so that successive calls do not wait for two in a row).
static AOM_FORCE_INLINE void qm_square_acc16(uint16x8_t weighted, uint32x4_t* acc_lo, uint32x4_t* acc_hi) {
    *acc_lo = vrsraq_n_u32(*acc_lo, vmull_u16(vget_low_u16(weighted), vget_low_u16(weighted)), 2 * AOM_QM_BITS);
    *acc_hi = vrsraq_n_u32(*acc_hi, vmull_high_u16(weighted, weighted), 2 * AOM_QM_BITS);
}

// Sum of the rounded squares of eight magnitudes times their weights, the products (below 2^24 for any 16-bit
// magnitude) taken in 32 bits and their squares in 64.
static AOM_FORCE_INLINE uint64x2_t qm_square_sum_wide16(uint16x8_t magnitude, uint16x8_t weight) {
    const uint32x4_t lo     = vmull_u16(vget_low_u16(magnitude), vget_low_u16(weight));
    const uint32x4_t hi     = vmull_high_u16(magnitude, weight);
    uint64x2_t       sum_lo = vrshrq_n_u64(vmull_u32(vget_low_u32(lo), vget_low_u32(lo)), 2 * AOM_QM_BITS);
    uint64x2_t       sum_hi = vrshrq_n_u64(vmull_u32(vget_low_u32(hi), vget_low_u32(hi)), 2 * AOM_QM_BITS);
    sum_lo                  = vrsraq_n_u64(sum_lo, vmull_high_u32(lo, lo), 2 * AOM_QM_BITS);
    sum_hi                  = vrsraq_n_u64(sum_hi, vmull_high_u32(hi, hi), 2 * AOM_QM_BITS);
    return vaddq_u64(sum_lo, sum_hi);
}

// qm_distortion_neon() out of line for one row, for the groups and row ends the 16-bit-lane kernel leaves to it.
static NOINLINE void qm_distortion_general_neon(const int32_t* coeff, const int32_t* recon, uint32_t width,
                                                uint64_t* distortion, bool has_coeff, const QmVal* weights) {
    if (has_coeff) {
        qm_distortion_neon(coeff, recon, width, width, 1, distortion, true, weights, width);
    } else {
        qm_distortion_neon(coeff, recon, width, width, 1, distortion, false, weights, width);
    }
}

// A rounded square of 16-bit weighted magnitudes is at most (65535^2 + 512) >> 10 = 4194176, and a lane of an
// accumulator takes one per group of sixteen coefficients: after 255 groups a lane is below 2^30, and the lanes of
// four accumulators add up in 32 bits.
#define QM_ACC_GROUPS 255

static AOM_FORCE_INLINE uint64_t qm_acc_sum(uint32x4_t a0, uint32x4_t a1, uint32x4_t a2, uint32x4_t a3) {
    return vaddlvq_u32(vaddq_u32(vaddq_u32(a0, a1), vaddq_u32(a2, a3)));
}

// One row of at least 32 coefficients in 16-bit lanes from the start; the sums of its source and error terms are
// added to *energy_sum and *error_sum. A group of sixteen that does not fit, and what remains of the row after the
// last full group, go through qm_distortion_general_neon().
static AOM_FORCE_INLINE void qm_distortion_row_neon(const int32_t* c, const int32_t* r, const QmVal* w, uint32_t width,
                                                    bool has_coeff, uint64_t* energy_sum, uint64_t* error_sum) {
    const uint32x4_t zero    = vdupq_n_u32(0);
    uint32x4_t       energy0 = zero, energy1 = zero, energy2 = zero, energy3 = zero;
    uint32x4_t       error0 = zero, error1 = zero, error2 = zero, error3 = zero;
    uint64x2_t       energy_wide = vdupq_n_u64(0), error_wide = vdupq_n_u64(0);
    uint32_t         acc_groups = 0;
    uint64_t         energy = 0, error = 0;
    uint64_t         part[DIST_CALC_TOTAL];
    uint32_t         x = 0;
    for (; x + 16 <= width; x += 16) {
        const uint8x16_t weights16 = vld1q_u8(w + x);
        // Saturating narrowing keeps a value that does not fit 16 bits at a magnitude of at least 32767, and the
        // difference of two narrowed values is exact in unsigned 16 bits. When every magnitude and every difference
        // is at most 16383, neither a coefficient nor a reconstructed coefficient was saturated (a saturated one is
        // at least 32767 - 16383 away from any such coefficient), so all of them are exact. The first test below
        // implies that bound, by counting a weight as at least 4, and keeps each weighted value in unsigned 16 bits.
        const int16x8_t  cn0 = vqmovn_high_s32(vqmovn_s32(vld1q_s32(c + x)), vld1q_s32(c + x + 4));
        const int16x8_t  cn1 = vqmovn_high_s32(vqmovn_s32(vld1q_s32(c + x + 8)), vld1q_s32(c + x + 12));
        const uint16x8_t ca0 = vreinterpretq_u16_s16(vabsq_s16(cn0));
        const uint16x8_t ca1 = vreinterpretq_u16_s16(vabsq_s16(cn1));
        const uint16x8_t w0  = vmovl_u8(vget_low_u8(weights16));
        const uint16x8_t w1  = vmovl_high_u8(weights16);
        uint16x8_t       mag = vmaxq_u16(ca0, ca1);
        uint16x8_t       da0 = ca0, da1 = ca1;
        if (has_coeff) {
            const int16x8_t rn0 = vqmovn_high_s32(vqmovn_s32(vld1q_s32(r + x)), vld1q_s32(r + x + 4));
            const int16x8_t rn1 = vqmovn_high_s32(vqmovn_s32(vld1q_s32(r + x + 8)), vld1q_s32(r + x + 12));
            da0                 = vreinterpretq_u16_s16(vabdq_s16(cn0, rn0));
            da1                 = vreinterpretq_u16_s16(vabdq_s16(cn1, rn1));
            mag                 = vmaxq_u16(mag, vmaxq_u16(da0, da1));
        }
        const uint32_t max_weight = vmaxvq_u8(weights16);
        const uint32_t max_mag    = vmaxvq_u16(mag);
        if (max_mag * AOMMAX(max_weight, 4) > UINT16_MAX) {
            if (max_mag <= 16383) {
                // exact magnitudes and differences as above; only their products with the weights need more than
                // 16 bits (at most 16383 * 255 < 2^22, squares below 2^44)
                energy_wide = vaddq_u64(energy_wide,
                                        vaddq_u64(qm_square_sum_wide16(ca0, w0), qm_square_sum_wide16(ca1, w1)));
                if (has_coeff) {
                    error_wide = vaddq_u64(error_wide,
                                           vaddq_u64(qm_square_sum_wide16(da0, w0), qm_square_sum_wide16(da1, w1)));
                }
            } else {
                qm_distortion_general_neon(c + x, has_coeff ? r + x : NULL, 16, part, has_coeff, w + x);
                energy += part[DIST_CALC_PREDICTION];
                error += part[DIST_CALC_RESIDUAL];
            }
            continue;
        }
        if (acc_groups == QM_ACC_GROUPS) {
            energy += qm_acc_sum(energy0, energy1, energy2, energy3);
            error += qm_acc_sum(error0, error1, error2, error3);
            energy0 = energy1 = energy2 = energy3 = zero;
            error0 = error1 = error2 = error3 = zero;
            acc_groups                        = 0;
        }
        acc_groups++;
        qm_square_acc16(vmulq_u16(ca0, w0), &energy0, &energy1);
        qm_square_acc16(vmulq_u16(ca1, w1), &energy2, &energy3);
        if (has_coeff) {
            qm_square_acc16(vmulq_u16(da0, w0), &error0, &error1);
            qm_square_acc16(vmulq_u16(da1, w1), &error2, &error3);
        }
    }
    if (x < width) {
        qm_distortion_general_neon(c + x, has_coeff ? r + x : NULL, width - x, part, has_coeff, w + x);
        energy += part[DIST_CALC_PREDICTION];
        error += part[DIST_CALC_RESIDUAL];
    }
    *energy_sum += energy + qm_acc_sum(energy0, energy1, energy2, energy3) + vaddvq_u64(energy_wide);
    *error_sum += error + qm_acc_sum(error0, error1, error2, error3) + vaddvq_u64(error_wide);
}

// Blocks with rows of at least 32 coefficients.
static NOINLINE void qm_distortion_rows_neon(const int32_t* coeff, const int32_t* recon, uint32_t stride,
                                             uint32_t width, uint32_t height, uint64_t* distortion, uint32_t eob,
                                             const QmVal* weights, uint32_t weight_stride) {
    uint64_t energy = 0, error = 0;
    if (eob) {
        for (uint32_t y = 0; y < height; ++y) {
            qm_distortion_row_neon(
                coeff + y * stride, recon + y * stride, weights + y * weight_stride, width, true, &energy, &error);
        }
    } else {
        for (uint32_t y = 0; y < height; ++y) {
            qm_distortion_row_neon(
                coeff + y * stride, NULL, weights + y * weight_stride, width, false, &energy, &error);
        }
    }
    distortion[DIST_CALC_PREDICTION] = energy;
    distortion[DIST_CALC_RESIDUAL]   = eob ? error : energy;
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
    } else if (width >= 32) {
        qm_distortion_rows_neon(coeff, recon, stride, width, height, distortion, eob, weights, weight_stride);
    } else if (eob) {
        qm_distortion_neon(coeff, recon, stride, width, height, distortion, true, weights, weight_stride);
    } else {
        qm_distortion_neon(coeff, recon, stride, width, height, distortion, false, weights, weight_stride);
    }
}
