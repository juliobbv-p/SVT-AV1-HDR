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
#include <assert.h>
#include <math.h>

#include "aom_dsp_rtcd.h"
#include "inv_transforms.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "utility.h"

static inline uint16_t get_max_eob(int16x8_t v_eobmax) {
    int16_t max_val = vmaxvq_s16(v_eobmax);
    return (uint16_t)max_val + 1;
}

static inline int16x8_t get_max_lane_eob(const int16_t* iscan, int16x8_t v_eobmax, uint16x8_t v_mask) {
    const int16x8_t v_iscan    = vld1q_s16(iscan);
    const int16x8_t v_nz_iscan = vbslq_s16(v_mask, v_iscan, vdupq_n_s16(-1));
    return vmaxq_s16(v_eobmax, v_nz_iscan);
}

static inline uint16x8_t quantize_fp_8(const TranLow* coeff_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                       int16x8_t v_quant, int16x8_t v_dequant, int16x8_t v_round, int16x8_t v_zero) {
    const int16x8_t  v_coeff      = load_tran_low_to_s16q(coeff_ptr);
    const int16x8_t  v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    const int16x8_t  v_abs        = vabsq_s16(v_coeff);
    const int16x8_t  v_tmp        = vqaddq_s16(v_abs, v_round);
    const int16x8_t  v_tmp2       = vshrq_n_s16(vqdmulhq_s16(v_tmp, v_quant), 1);
    const uint16x8_t v_nz_mask    = vcgtq_s16(v_tmp2, v_zero);
    const int16x8_t  v_qcoeff_a   = veorq_s16(v_tmp2, v_coeff_sign);
    const int16x8_t  v_qcoeff     = vsubq_s16(v_qcoeff_a, v_coeff_sign);
    const int16x8_t  v_dqcoeff    = vmulq_s16(v_qcoeff, v_dequant);
    store_s16q_to_tran_low(qcoeff_ptr, v_qcoeff);
    store_s16q_to_tran_low(dqcoeff_ptr, v_dqcoeff);
    return v_nz_mask;
}

void svt_av1_quantize_fp_neon(const TranLow* coeff_ptr, intptr_t count, const int16_t* zbin_ptr,
                              const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                              TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr, uint16_t* eob_ptr,
                              const int16_t* scan, const int16_t* iscan) {
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    (void)scan;

    // Quantization pass: All coefficients with index >= zero_flag are
    // skippable. Note: zero_flag can be zero.
    const int16x8_t v_zero            = vdupq_n_s16(0);
    int16x8_t       v_quant           = vld1q_s16(quant_ptr);
    int16x8_t       v_dequant         = vld1q_s16(dequant_ptr);
    int16x8_t       v_round           = vld1q_s16(round_ptr);
    int16x8_t       v_eobmax_76543210 = vdupq_n_s16(-1);
    uint16x8_t      v_nz_mask;
    // process dc and the first seven ac coeffs
    v_nz_mask         = quantize_fp_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant, v_dequant, v_round, v_zero);
    v_eobmax_76543210 = get_max_lane_eob(iscan, v_eobmax_76543210, v_nz_mask);
    // overwrite the dc constants with ac constants
    v_quant   = vdupq_lane_s16(vget_low_s16(v_quant), 1);
    v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
    v_round   = vdupq_lane_s16(vget_low_s16(v_round), 1);

    count -= 8;
    // now process the rest of the ac coeffs
    do {
        coeff_ptr += 8;
        qcoeff_ptr += 8;
        dqcoeff_ptr += 8;
        iscan += 8;
        v_nz_mask         = quantize_fp_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant, v_dequant, v_round, v_zero);
        v_eobmax_76543210 = get_max_lane_eob(iscan, v_eobmax_76543210, v_nz_mask);
        count -= 8;
    } while (count > 0);
    *eob_ptr = get_max_eob(v_eobmax_76543210);
}

static AOM_FORCE_INLINE uint16x8_t quantize_fp_logscale_8(const TranLow* coeff_ptr, TranLow* qcoeff_ptr,
                                                          TranLow* dqcoeff_ptr, int16x8_t v_quant, int16x8_t v_dequant,
                                                          int16x8_t v_round, int16x8_t v_zero, int log_scale) {
    const int16x8_t  v_log_scale_minus_1    = vdupq_n_s16(log_scale - 1);
    const int16x8_t  v_neg_log_scale_plus_1 = vdupq_n_s16(-(1 + log_scale));
    const int16x8_t  v_coeff                = load_tran_low_to_s16q(coeff_ptr);
    const int16x8_t  v_coeff_sign           = vshrq_n_s16(v_coeff, 15);
    const int16x8_t  v_abs_coeff            = vabsq_s16(v_coeff);
    const uint16x8_t v_mask                 = vcgeq_s16(v_abs_coeff, vshlq_s16(v_dequant, v_neg_log_scale_plus_1));
    // const int64_t tmp = vmask ? (int64_t)abs_coeff + log_scaled_round : 0
    const int16x8_t  v_tmp     = vandq_s16(vqaddq_s16(v_abs_coeff, v_round), vreinterpretq_s16_u16(v_mask));
    const int16x8_t  v_tmp2    = vqdmulhq_s16(vshlq_s16(v_tmp, v_log_scale_minus_1), v_quant);
    const uint16x8_t v_nz_mask = vcgtq_s16(v_tmp2, v_zero);
    const int16x8_t  v_qcoeff  = vsubq_s16(veorq_s16(v_tmp2, v_coeff_sign), v_coeff_sign);
    // Multiplying by dequant here will use all 16 bits. Cast to unsigned before
    // shifting right. (vshlq_s16 will shift right if shift value is negative)
    const uint16x8_t v_abs_dqcoeff = vshlq_u16(vreinterpretq_u16_s16(vmulq_s16(v_tmp2, v_dequant)),
                                               vdupq_n_s16(-log_scale));
    const int16x8_t  v_dqcoeff = vsubq_s16(veorq_s16(vreinterpretq_s16_u16(v_abs_dqcoeff), v_coeff_sign), v_coeff_sign);
    store_s16q_to_tran_low(qcoeff_ptr, v_qcoeff);
    store_s16q_to_tran_low(dqcoeff_ptr, v_dqcoeff);
    return v_nz_mask;
}

static AOM_FORCE_INLINE uint16x8_t quantize_fp_logscale2_8(const TranLow* coeff_ptr, TranLow* qcoeff_ptr,
                                                           TranLow* dqcoeff_ptr, int16x8_t v_quant, int16x8_t v_dequant,
                                                           int16x8_t v_round, int16x8_t v_zero) {
    const int16x8_t  v_coeff      = load_tran_low_to_s16q(coeff_ptr);
    const int16x8_t  v_coeff_sign = vshrq_n_s16(v_coeff, 15);
    const int16x8_t  v_abs_coeff  = vabsq_s16(v_coeff);
    const uint16x8_t v_mask       = vcgeq_u16(vshlq_n_u16(vreinterpretq_u16_s16(v_abs_coeff), 1),
                                        vshrq_n_u16(vreinterpretq_u16_s16(v_dequant), 2));
    // abs_coeff = vmask ? (int64_t)abs_coeff + log_scaled_round : 0
    const int16x8_t v_tmp = vandq_s16(vqaddq_s16(v_abs_coeff, v_round), vreinterpretq_s16_u16(v_mask));
    // tmp32 = (int)((abs_coeff * quant_ptr[rc != 0]) >> (16 - log_scale));
    const int16x8_t v_tmp2 = vorrq_s16(
        vshlq_n_s16(vqdmulhq_s16(v_tmp, v_quant), 1),
        vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(vmulq_s16(v_tmp, v_quant)), 14)));
    const uint16x8_t v_nz_mask = vcgtq_s16(v_tmp2, v_zero);
    const int16x8_t  v_qcoeff  = vsubq_s16(veorq_s16(v_tmp2, v_coeff_sign), v_coeff_sign);
    // const TranLow abs_dqcoeff = (tmp32 * dequant_ptr[rc != 0]) >> log_scale;
    const int16x8_t v_abs_dqcoeff = vorrq_s16(
        vshlq_n_s16(vqdmulhq_s16(v_tmp2, v_dequant), 13),
        vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(vmulq_s16(v_tmp2, v_dequant)), 2)));
    const int16x8_t v_dqcoeff = vsubq_s16(veorq_s16(v_abs_dqcoeff, v_coeff_sign), v_coeff_sign);
    store_s16q_to_tran_low(qcoeff_ptr, v_qcoeff);
    store_s16q_to_tran_low(dqcoeff_ptr, v_dqcoeff);
    return v_nz_mask;
}

static AOM_FORCE_INLINE void quantize_fp_no_qmatrix_neon(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                         const int16_t* round_ptr, const int16_t* quant_ptr,
                                                         TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                         const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                         const int16_t* iscan, int log_scale) {
    const int16x8_t v_zero            = vdupq_n_s16(0);
    int16x8_t       v_quant           = vld1q_s16(quant_ptr);
    int16x8_t       v_dequant         = vld1q_s16(dequant_ptr);
    const int16x8_t v_round_no_scale  = vld1q_s16(round_ptr);
    int16x8_t       v_round           = vqrdmulhq_n_s16(v_round_no_scale, (int16_t)(1 << (15 - log_scale)));
    int16x8_t       v_eobmax_76543210 = vdupq_n_s16(-1);
    intptr_t        non_zero_count    = n_coeffs;

    assert(n_coeffs > 16);
    // Pre-scan pass. Zero the skipped chunks in place as we go: fusing the store into the scan is
    // markedly faster on sparse blocks than memset-ing the whole tail afterwards, and free when nothing
    // is skipped.
    const int16x8_t v_dequant_scaled = vshlq_s16(v_dequant, vdupq_n_s16(-(1 + log_scale)));
    const int16x8_t v_zbin_s16       = vdupq_lane_s16(vget_low_s16(v_dequant_scaled), 1);
    const int32x4_t v_zero32         = vdupq_n_s32(0);
    intptr_t        i                = n_coeffs;
    do {
        const int16x8_t  v_coeff_a     = load_tran_low_to_s16q(coeff_ptr + i - 8);
        const int16x8_t  v_coeff_b     = load_tran_low_to_s16q(coeff_ptr + i - 16);
        const int16x8_t  v_abs_coeff_a = vabsq_s16(v_coeff_a);
        const int16x8_t  v_abs_coeff_b = vabsq_s16(v_coeff_b);
        const uint16x8_t v_mask_a      = vcgeq_s16(v_abs_coeff_a, v_zbin_s16);
        const uint16x8_t v_mask_b      = vcgeq_s16(v_abs_coeff_b, v_zbin_s16);
        // If the coefficient is in the base ZBIN range, then discard.
        if (horizontal_long_add_u16x8(v_mask_a, v_mask_b) != 0) {
            break;
        }
        vst1q_s32(qcoeff_ptr + i - 4, v_zero32);
        vst1q_s32(qcoeff_ptr + i - 8, v_zero32);
        vst1q_s32(qcoeff_ptr + i - 12, v_zero32);
        vst1q_s32(qcoeff_ptr + i - 16, v_zero32);
        vst1q_s32(dqcoeff_ptr + i - 4, v_zero32);
        vst1q_s32(dqcoeff_ptr + i - 8, v_zero32);
        vst1q_s32(dqcoeff_ptr + i - 12, v_zero32);
        vst1q_s32(dqcoeff_ptr + i - 16, v_zero32);
        non_zero_count -= 16;
        i -= 16;
    } while (i > 0);

    // process dc and the first seven ac coeffs
    uint16x8_t v_nz_mask;
    if (log_scale == 2) {
        v_nz_mask = quantize_fp_logscale2_8(coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant, v_dequant, v_round, v_zero);
    } else {
        v_nz_mask = quantize_fp_logscale_8(
            coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant, v_dequant, v_round, v_zero, log_scale);
    }
    v_eobmax_76543210 = get_max_lane_eob(iscan, v_eobmax_76543210, v_nz_mask);
    // overwrite the dc constants with ac constants
    v_quant   = vdupq_lane_s16(vget_low_s16(v_quant), 1);
    v_dequant = vdupq_lane_s16(vget_low_s16(v_dequant), 1);
    v_round   = vdupq_lane_s16(vget_low_s16(v_round), 1);

    for (intptr_t count = non_zero_count - 8; count > 0; count -= 8) {
        coeff_ptr += 8;
        qcoeff_ptr += 8;
        dqcoeff_ptr += 8;
        iscan += 8;
        if (log_scale == 2) {
            v_nz_mask = quantize_fp_logscale2_8(
                coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant, v_dequant, v_round, v_zero);
        } else {
            v_nz_mask = quantize_fp_logscale_8(
                coeff_ptr, qcoeff_ptr, dqcoeff_ptr, v_quant, v_dequant, v_round, v_zero, log_scale);
        }
        v_eobmax_76543210 = get_max_lane_eob(iscan, v_eobmax_76543210, v_nz_mask);
    }
    *eob_ptr = get_max_eob(v_eobmax_76543210);
}

void svt_av1_quantize_fp_32x32_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                    const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                    TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                    uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan) {
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    (void)scan;
    quantize_fp_no_qmatrix_neon(
        coeff_ptr, n_coeffs, round_ptr, quant_ptr, qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, iscan, 1);
}

void svt_av1_quantize_fp_64x64_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                    const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                    TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                    uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan) {
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    (void)scan;
    quantize_fp_no_qmatrix_neon(
        coeff_ptr, n_coeffs, round_ptr, quant_ptr, qcoeff_ptr, dqcoeff_ptr, dequant_ptr, eob_ptr, iscan, 2);
}

// Quantize 4 coefficients (one int32x4 lane group) using a quantization matrix (fast-quantize path).
// Mirrors the scalar reference quantize_fp_helper_c() qm path and the AVX2 quantize_qm() kernel.
// `highbd` is a compile-time constant selecting only the int16 clamp: the 8-bit forward path clamps
// (abs_coeff + round) to int16 to bit-match the int16-lane SIMD kernels, while the high-bit-depth
// helpers (highbd_quantize_fp_helper_c) leave abs_coeff unclamped. The quant multiply itself is the
// same 32-bit doubling multiply-high at every bit depth -- vqdmulhq_s32 holds the wide product in its
// internal 64-bit accumulator, so no widening to 64-bit lanes is needed (see the range note below).
// Returns a per-lane non-zero mask (abs_qcoeff != 0, matching the C reference's EOB rule).
static AOM_FORCE_INLINE uint32x4_t quantize_fp_qm_4(int32x4_t v_coeff, int32x4_t v_abs, uint32x4_t v_below,
                                                    int32x4_t v_qm, int32x4_t v_iqm, int32x4_t v_round,
                                                    int32x4_t v_q_scale, int32x4_t v_dequant, TranLow* qcoeff_ptr,
                                                    TranLow* dqcoeff_ptr, int log_scale, const int highbd) {
    const int32x4_t v_sign = vshrq_n_s32(v_coeff, 31);
    // abs_qcoeff = ((abs_coeff + round) * wt * quant) >> (16 - log_scale + AOM_QM_BITS). The "* quant >> S"
    // step folds into a doubling multiply-high: vqdmulhq_s32(p, v_q_scale) == (p * quant) >> S (truncating,
    // matching the C >>), with the wide product held in the SQDMULH accumulator. v_q_scale = quant <<
    // (31 - S) is loop-invariant, so the caller precomputes it once. The 32-bit-lane operands p = (abs +
    // round) * wt and v_q_scale both stay inside int32 at every bit depth (|coeff| < 1 << (7 + bd), wt <=
    // 255 so p <= 2^27; quant <= 16384, S = 21 - log_scale so v_q_scale <= 2^27). One path serves
    // 8/10/12-bit; only the 8-bit int16 clamp differs (highbd leaves abs_coeff unclamped, matching it).
    int32x4_t v_q = vaddq_s32(v_abs, v_round);
    if (!highbd) {
        v_q = vminq_s32(v_q, vdupq_n_s32(32767));
    }
    // Zero the below-threshold lanes once, here; the zeros propagate through the quant to qcoeff and
    // dqcoeff, so no separate output masking is needed.
    v_q                      = vbicq_s32(v_q, vreinterpretq_s32_u32(v_below));
    const int32x4_t v_p      = vmulq_s32(v_q, v_qm);
    int32x4_t       v_qcoeff = vqdmulhq_s32(v_p, v_q_scale);

    // dequant_qm = (dequant * iqm + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS; abs_dqcoeff = (abs_qcoeff * dequant_qm) >> log_scale.
    // The C reference computes this product in int32 at every bit depth, so keep it in 32-bit lanes; a
    // wider synthetic abs_qcoeff wraps mod 2^32 identically to the reference (real coefficients never overflow).
    int32x4_t v_dqf     = vmulq_s32(v_dequant, v_iqm);
    v_dqf               = vrshrq_n_s32(v_dqf, AOM_QM_BITS);
    int32x4_t v_dqcoeff = vshlq_s32(vmulq_s32(v_qcoeff, v_dqf), vdupq_n_s32(-log_scale));

    // Per-lane nonzero mask (abs_qcoeff != 0; below-threshold lanes are already zero), scanned for the EOB.
    const uint32x4_t v_nz = vbicq_u32(vmvnq_u32(vceqzq_s32(v_qcoeff)), v_below);

    // Restore sign (below-threshold lanes were zeroed before the quant, so they stay zero).
    v_qcoeff  = vsubq_s32(veorq_s32(v_qcoeff, v_sign), v_sign);
    v_dqcoeff = vsubq_s32(veorq_s32(v_dqcoeff, v_sign), v_sign);

    vst1q_s32(qcoeff_ptr, v_qcoeff);
    vst1q_s32(dqcoeff_ptr, v_dqcoeff);
    return v_nz;
}

// Build a per-lane parameter vector holding the DC value in lane 0 and the AC value in lanes 1..3.
static AOM_FORCE_INLINE int32x4_t dc_ac_s32(int32_t dc, int32_t ac) {
    return vsetq_lane_s32(dc, vdupq_n_s32(ac), 0);
}

// Largest weight of a quantization matrix (n is a multiple of 8, at least 16).
static AOM_FORCE_INLINE int32_t qm_max_weight_neon(const QmVal* qm_ptr, intptr_t n) {
    uint8x16_t m0 = vdupq_n_u8(0), m1 = m0, m2 = m0, m3 = m0;
    intptr_t   j = 0;
    for (; j + 64 <= n; j += 64) {
        m0 = vmaxq_u8(m0, vld1q_u8(qm_ptr + j));
        m1 = vmaxq_u8(m1, vld1q_u8(qm_ptr + j + 16));
        m2 = vmaxq_u8(m2, vld1q_u8(qm_ptr + j + 32));
        m3 = vmaxq_u8(m3, vld1q_u8(qm_ptr + j + 48));
    }
    for (; j + 16 <= n; j += 16) {
        m0 = vmaxq_u8(m0, vld1q_u8(qm_ptr + j));
    }
    if (j < n) {
        m1 = vmaxq_u8(m1, vcombine_u8(vld1_u8(qm_ptr + j), vdup_n_u8(0)));
    }
    return vmaxvq_u8(vmaxq_u8(vmaxq_u8(m0, m1), vmaxq_u8(m2, m3)));
}

// The same value, remembered per table. The encoder's quantization matrices are constant data
// (svt_av1_qm_table_is_immutable() recognises them by address), so the largest weight of the n entries at such an
// address never changes, and every call after the first for a table skips the scan above. An entry packs the
// address (below 2^47), n / 8 (n is a multiple of 8 of at most 1024) and the weight into one word, so that
// entries read and written concurrently by several threads are always whole. An entry is only ever stored for a
// table that passed the check, so a hit needs no check of its own; any other table never gets an entry and is
// scanned on every call, as is anything that does not fit an entry. Matrices of fewer than 64 weights are
// scanned too: one or two vectors cost less than the lookup.
bool svt_av1_qm_table_is_immutable(const QmVal* p);

static AOM_FORCE_INLINE int32_t qm_max_weight_cached_neon(const QmVal* qm_ptr, intptr_t n) {
#if defined(__GNUC__) || defined(__clang__)
    static uint64_t cache[64];
    const uint64_t  addr = (uint64_t)(uintptr_t)qm_ptr;
    if (n >= 64 && n <= 1024 && !(addr >> 47)) {
        const uint64_t tag   = (addr << 16) | ((uint64_t)(n >> 3) << 8);
        uint64_t*      slot  = &cache[(addr ^ (addr >> 6) ^ (addr >> 12) ^ (uint64_t)n) & 63];
        const uint64_t entry = __atomic_load_n(slot, __ATOMIC_RELAXED);
        if ((entry & ~(uint64_t)0xff) == tag && (entry & 0xff)) {
            return (int32_t)(entry & 0xff);
        }
        const int32_t max_weight = qm_max_weight_neon(qm_ptr, n);
        if (svt_av1_qm_table_is_immutable(qm_ptr)) {
            __atomic_store_n(slot, tag | (uint64_t)max_weight, __ATOMIC_RELAXED);
        }
        return max_weight;
    }
#endif
    return qm_max_weight_neon(qm_ptr, n);
}

// True when the 16 coefficients at coeff are all at most `bound` in magnitude; they are then below the
// quantization threshold for every weight of the matrix, and their outputs are zeroed here.
static AOM_FORCE_INLINE int qm_skip16_neon(const TranLow* coeff, TranLow* qcoeff, TranLow* dqcoeff, int32x4_t bound) {
    const int32x4_t a0 = vabsq_s32(vld1q_s32(coeff));
    const int32x4_t a1 = vabsq_s32(vld1q_s32(coeff + 4));
    const int32x4_t a2 = vabsq_s32(vld1q_s32(coeff + 8));
    const int32x4_t a3 = vabsq_s32(vld1q_s32(coeff + 12));
    const int32x4_t m  = vmaxq_s32(vmaxq_s32(a0, a1), vmaxq_s32(a2, a3));
    if (vmaxvq_u32(vcgtq_s32(m, bound))) {
        return 0;
    }
    const int32x4_t zero = vdupq_n_s32(0);
    vst1q_s32(qcoeff, zero);
    vst1q_s32(qcoeff + 4, zero);
    vst1q_s32(qcoeff + 8, zero);
    vst1q_s32(qcoeff + 12, zero);
    vst1q_s32(dqcoeff, zero);
    vst1q_s32(dqcoeff + 4, zero);
    vst1q_s32(dqcoeff + 8, zero);
    vst1q_s32(dqcoeff + 12, zero);
    return 1;
}

static AOM_FORCE_INLINE void quantize_fp_qm_neon_impl(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                      const int16_t* round_ptr, const int16_t* quant_ptr,
                                                      TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                      const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                      const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                                      int log_scale, const int highbd) {
    assert(qm_ptr != NULL && iqm_ptr != NULL);
    assert(n_coeffs >= 16 && (n_coeffs & 7) == 0);

    const int       shift_q      = 16 - log_scale + AOM_QM_BITS;
    const int       thr_sh       = AOM_QM_BITS - (1 + log_scale);
    const int32_t   rnd_dc       = log_scale > 0 ? (round_ptr[0] + (1 << (log_scale - 1))) >> log_scale : round_ptr[0];
    const int32_t   rnd_ac       = log_scale > 0 ? (round_ptr[1] + (1 << (log_scale - 1))) >> log_scale : round_ptr[1];
    const int32x4_t v_round_ac   = vdupq_n_s32(rnd_ac);
    const int32x4_t v_quant_ac   = vdupq_n_s32(quant_ptr[1]);
    const int32x4_t v_dequant_ac = vdupq_n_s32(dequant_ptr[1]);
    const int32x4_t v_thr_ac     = vshlq_s32(v_dequant_ac, vdupq_n_s32(thr_sh));
    // quant << (31 - shift_q) is loop-invariant; precompute it once instead of per coefficient group.
    const int       qsh         = 31 - shift_q;
    const int32x4_t v_qscale_ac = vshlq_s32(v_quant_ac, vdupq_n_s32(qsh));

    int16x8_t v_eobmax = vdupq_n_s16(-1);

    // First group of 8: lane 0 of the low half carries the DC quant constants, the rest use AC.
    int32x4_t v_round_lo   = dc_ac_s32(rnd_dc, rnd_ac);
    int32x4_t v_quant_lo   = dc_ac_s32(quant_ptr[0], quant_ptr[1]);
    int32x4_t v_dequant_lo = dc_ac_s32(dequant_ptr[0], dequant_ptr[1]);
    int32x4_t v_thr_lo     = vshlq_s32(v_dequant_lo, vdupq_n_s32(thr_sh));
    int32x4_t v_qscale_lo  = vshlq_s32(v_quant_lo, vdupq_n_s32(qsh));

    // abs * wt < thr holds for every weight when abs <= (thr - 1) / max(wt).
    const int32_t   thr_ac_s   = (int32_t)dequant_ptr[1] << thr_sh;
    const int32x4_t v_skip_max = vdupq_n_s32((thr_ac_s - 1) / AOMMAX(qm_max_weight_cached_neon(qm_ptr, n_coeffs), 1));

    intptr_t i = 0;
    for (;;) {
        // AC-only groups: most high-frequency coefficients are tiny, so test sixteen at a time first.
        while (i >= 8 && i + 16 <= n_coeffs &&
               qm_skip16_neon(coeff_ptr + i, qcoeff_ptr + i, dqcoeff_ptr + i, v_skip_max)) {
            i += 16;
        }
        if (i >= n_coeffs) {
            break;
        }
        const int32x4_t  coeff_lo = vld1q_s32(coeff_ptr + i);
        const int32x4_t  coeff_hi = vld1q_s32(coeff_ptr + i + 4);
        const int32x4_t  abs_lo = vabsq_s32(coeff_lo), abs_hi = vabsq_s32(coeff_hi);
        const uint16x8_t v_qm     = vmovl_u8(vld1_u8(qm_ptr + i));
        const int32x4_t  v_qm_lo  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_qm)));
        const int32x4_t  v_qm_hi  = vreinterpretq_s32_u32(vmovl_high_u16(v_qm));
        const uint32x4_t below_lo = vcgtq_s32(v_thr_lo, vmulq_s32(abs_lo, v_qm_lo));
        const uint32x4_t below_hi = vcgtq_s32(v_thr_ac, vmulq_s32(abs_hi, v_qm_hi));
        // Eight below-threshold coefficients cannot affect EOB. Skip inverse-matrix
        // loads and scan tracking as well as the quantization and dequantization math.
        if (vminvq_u32(vandq_u32(below_lo, below_hi)) == UINT32_MAX) {
            vst1q_s32(qcoeff_ptr + i, vdupq_n_s32(0));
            vst1q_s32(qcoeff_ptr + i + 4, vdupq_n_s32(0));
            vst1q_s32(dqcoeff_ptr + i, vdupq_n_s32(0));
            vst1q_s32(dqcoeff_ptr + i + 4, vdupq_n_s32(0));
        } else {
            const uint16x8_t v_iqm     = vmovl_u8(vld1_u8(iqm_ptr + i));
            const int32x4_t  v_iqm_lo  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_iqm)));
            const int32x4_t  v_iqm_hi  = vreinterpretq_s32_u32(vmovl_high_u16(v_iqm));
            const uint32x4_t nz_lo     = quantize_fp_qm_4(coeff_lo,
                                                      abs_lo,
                                                      below_lo,
                                                      v_qm_lo,
                                                      v_iqm_lo,
                                                      v_round_lo,
                                                      v_qscale_lo,
                                                      v_dequant_lo,
                                                      qcoeff_ptr + i,
                                                      dqcoeff_ptr + i,
                                                      log_scale,
                                                      highbd);
            const uint32x4_t nz_hi     = quantize_fp_qm_4(coeff_hi,
                                                      abs_hi,
                                                      below_hi,
                                                      v_qm_hi,
                                                      v_iqm_hi,
                                                      v_round_ac,
                                                      v_qscale_ac,
                                                      v_dequant_ac,
                                                      qcoeff_ptr + i + 4,
                                                      dqcoeff_ptr + i + 4,
                                                      log_scale,
                                                      highbd);
            const uint16x8_t v_nz_mask = vcombine_u16(vmovn_u32(nz_lo), vmovn_u32(nz_hi));
            v_eobmax                   = get_max_lane_eob(iscan + i, v_eobmax, v_nz_mask);
        }
        i += 8;
        if (i >= n_coeffs) {
            break;
        }
        // All subsequent lanes use AC constants.
        v_round_lo   = v_round_ac;
        v_qscale_lo  = v_qscale_ac;
        v_dequant_lo = v_dequant_ac;
        v_thr_lo     = v_thr_ac;
    }
    *eob_ptr = get_max_eob(v_eobmax);
}

// Largest magnitudes of the first and of the last eight of sixteen coefficients (given as four vectors of
// magnitudes), in the low and the high half of the result.
static AOM_FORCE_INLINE uint64_t qm_max_abs_halves(int32x4_t a0, int32x4_t a1, int32x4_t a2, int32x4_t a3) {
    const uint32x4_t m01 = vmaxq_u32(vreinterpretq_u32_s32(a0), vreinterpretq_u32_s32(a1));
    const uint32x4_t m23 = vmaxq_u32(vreinterpretq_u32_s32(a2), vreinterpretq_u32_s32(a3));
    const uint32x4_t m   = vpmaxq_u32(m01, m23);
    return vgetq_lane_u64(vreinterpretq_u64_u32(vpmaxq_u32(m, m)), 0);
}

static AOM_FORCE_INLINE void qm_zero_8(TranLow* qcoeff, TranLow* dqcoeff) {
    const int32x4_t zero = vdupq_n_s32(0);
    vst1q_s32(qcoeff, zero);
    vst1q_s32(qcoeff + 4, zero);
    vst1q_s32(dqcoeff, zero);
    vst1q_s32(dqcoeff + 4, zero);
}

// dequant_qm = (dequant * iqm + 16) >> 5 of eight positions in 16-bit lanes, for dequant = 32 * deq_hi + deq_lo with
// deq_hi <= 255 and deq_lo <= 31 (dequant <= 8191): the product splits into 32 * deq_hi * iqm, a multiple of 32, and
// deq_lo * iqm <= 31 * 255, so the result is deq_hi * iqm + ((deq_lo * iqm + 16) >> 5) <= 255 * 255 + 247 = 65272.
// For dequant <= 4095 (deq_hi <= 127) it is at most 127 * 255 + 247 = 32632.
static AOM_FORCE_INLINE uint16x8_t qm_dequant_8(const QmVal* iqm, uint8x8_t deq_hi, uint8x8_t deq_lo) {
    const uint8x8_t v_iqm = vld1_u8(iqm);
    return vmlal_u8(vrshrq_n_u16(vmull_u8(v_iqm, deq_lo), AOM_QM_BITS), v_iqm, deq_hi);
}

// Signs, stores and scan tracking shared by the 16-bit-lane quantizers: q_lo / q_hi hold eight levels 0..32767 (zero
// where the coefficient is below its threshold), c_lo / c_hi the coefficients, v_dqf the output of qm_dequant_8().
// level * dequant_qm <= 32767 * 65272 < 2^31 is the 32-bit product of the reference. `direct` (log_scale 0 and
// dequant <= 4095, so dequant_qm <= 32632 is a positive 16-bit value) multiplies the signed level instead of giving
// the product its sign afterwards.
static AOM_FORCE_INLINE int16x8_t qm_finish_8(int32x4_t q_lo, int32x4_t q_hi, int32x4_t c_lo, int32x4_t c_hi,
                                              uint16x8_t v_dqf, TranLow* qcoeff, TranLow* dqcoeff, const int16_t* iscan,
                                              int16x8_t v_eobmax, int log_scale, const int direct) {
    const int16x8_t v_q = vuzp1q_s16(vreinterpretq_s16_s32(q_lo), vreinterpretq_s16_s32(q_hi));
    // the upper half of a 32-bit coefficient carries its sign
    const int16x8_t v_sign = vshrq_n_s16(vuzp2q_s16(vreinterpretq_s16_s32(c_lo), vreinterpretq_s16_s32(c_hi)), 15);
    const int16x8_t v_qs   = vsubq_s16(veorq_s16(v_q, v_sign), v_sign);
    vst1q_s32(qcoeff, vmovl_s16(vget_low_s16(v_qs)));
    vst1q_s32(qcoeff + 4, vmovl_high_s16(v_qs));
    if (direct) {
        const int16x8_t v_dqf_s = vreinterpretq_s16_u16(v_dqf);
        vst1q_s32(dqcoeff, vmull_s16(vget_low_s16(v_qs), vget_low_s16(v_dqf_s)));
        vst1q_s32(dqcoeff + 4, vmull_high_s16(v_qs, v_dqf_s));
    } else {
        // the shift applies to the magnitude
        const uint16x8_t v_qu    = vreinterpretq_u16_s16(v_q);
        const int32x4_t  v_shift = vdupq_n_s32(-log_scale);
        const int32x4_t  s_lo    = vmovl_s16(vget_low_s16(v_sign));
        const int32x4_t  s_hi    = vmovl_high_s16(v_sign);
        const uint32x4_t d_lo    = vshlq_u32(vmull_u16(vget_low_u16(v_qu), vget_low_u16(v_dqf)), v_shift);
        const uint32x4_t d_hi    = vshlq_u32(vmull_high_u16(v_qu, v_dqf), v_shift);
        vst1q_s32(dqcoeff, vsubq_s32(veorq_s32(vreinterpretq_s32_u32(d_lo), s_lo), s_lo));
        vst1q_s32(dqcoeff + 4, vsubq_s32(veorq_s32(vreinterpretq_s32_u32(d_hi), s_hi), s_hi));
    }
    return get_max_lane_eob(iscan, v_eobmax, vtstq_s16(v_q, v_q));
}

// Constants of the 16-bit-lane fast-quantize path for eight positions: rounding offset, threshold on magnitude *
// weight, quant << (31 - shift), and dequant split as 32 * deq_hi + deq_lo.
typedef struct {
    uint16x8_t round;
    uint32x4_t thr;
    int32x4_t  qscale;
    uint8x8_t  deq_hi;
    uint8x8_t  deq_lo;
} QmFpVectors;

// Eight coefficients with magnitude + round <= 65535 (c_lo / c_hi, magnitudes a_lo / a_hi): the magnitudes and the
// weights are narrowed to 16-bit lanes, so that each product with a weight comes from one widening multiply of
// 16-bit halves instead of widened operands. The caller guarantees levels of at most 32767 (see
// quantize_fp_qm_fast_neon()); `lo` holds the constants of the first four positions (DC or AC) and of the 16-bit and
// 8-bit lanes, `ac` those of the last four.
//   magnitude * weight <= 65535 * 255 < 2^24 and (magnitude + round) * weight < 2^24: exact in unsigned 32 bits, and
//   positive as the signed operand of the multiply-high, which is the one of quantize_fp_qm_4().
static AOM_FORCE_INLINE int16x8_t quantize_fp_qm_narrow_8(int32x4_t c_lo, int32x4_t c_hi, int32x4_t a_lo,
                                                          int32x4_t a_hi, const QmVal* qm, const QmVal* iqm,
                                                          const int16_t* iscan, TranLow* qcoeff, TranLow* dqcoeff,
                                                          const QmFpVectors* lo, const QmFpVectors* ac,
                                                          int16x8_t v_eobmax, int log_scale, const int direct) {
    const uint16x8_t v_abs    = vuzp1q_u16(vreinterpretq_u16_s32(a_lo), vreinterpretq_u16_s32(a_hi));
    const uint16x8_t v_qm     = vmovl_u8(vld1_u8(qm));
    const uint32x4_t below_lo = vcgtq_u32(lo->thr, vmull_u16(vget_low_u16(v_abs), vget_low_u16(v_qm)));
    const uint32x4_t below_hi = vcgtq_u32(ac->thr, vmull_high_u16(v_abs, v_qm));
    const uint16x8_t v_x      = vaddq_u16(v_abs, lo->round);
    const int32x4_t  p_lo     = vreinterpretq_s32_u32(vmull_u16(vget_low_u16(v_x), vget_low_u16(v_qm)));
    const int32x4_t  p_hi     = vreinterpretq_s32_u32(vmull_high_u16(v_x, v_qm));
    const int32x4_t  q_lo     = vbicq_s32(vqdmulhq_s32(p_lo, lo->qscale), vreinterpretq_s32_u32(below_lo));
    const int32x4_t  q_hi     = vbicq_s32(vqdmulhq_s32(p_hi, ac->qscale), vreinterpretq_s32_u32(below_hi));
    return qm_finish_8(q_lo,
                       q_hi,
                       c_lo,
                       c_hi,
                       qm_dequant_8(iqm, lo->deq_hi, lo->deq_lo),
                       qcoeff,
                       dqcoeff,
                       iscan,
                       v_eobmax,
                       log_scale,
                       direct);
}

// Sixteen coefficients, each half by its largest magnitude: all below the threshold for every weight (zeroed) or the
// 16-bit-lane path. A group with a magnitude above narrow_max is left to quantize_fp_qm_pending() by setting
// `group_bit` in *pending. skip_add holds 2^31 - 1 - (largest magnitude that is below the threshold for every
// weight) for the first half in its low and for the second half in its high 32 bits, so that bit 31 of a half of the
// sum with the two largest magnitudes (at most 2^31 each) is set exactly when that half has a larger magnitude.
static AOM_FORCE_INLINE int16x8_t quantize_fp_qm_16(const TranLow* coeff, TranLow* qcoeff, TranLow* dqcoeff,
                                                    const int16_t* iscan, const QmVal* qm, const QmVal* iqm,
                                                    const QmFpVectors* lo, const QmFpVectors* ac, uint64_t skip_add,
                                                    int64_t narrow_max, int16x8_t v_eobmax, uint64_t* pending,
                                                    uint64_t group_bit, int log_scale, const int direct) {
    const int32x4_t c0 = vld1q_s32(coeff), c1 = vld1q_s32(coeff + 4);
    const int32x4_t c2 = vld1q_s32(coeff + 8), c3 = vld1q_s32(coeff + 12);
    const int32x4_t a0 = vabsq_s32(c0), a1 = vabsq_s32(c1), a2 = vabsq_s32(c2), a3 = vabsq_s32(c3);
    const uint64_t  max_abs = qm_max_abs_halves(a0, a1, a2, a3);
    const uint64_t  over    = (max_abs + skip_add) & 0x8000000080000000ull;
    if (!over) {
        qm_zero_8(qcoeff, dqcoeff);
        qm_zero_8(qcoeff + 8, dqcoeff + 8);
        return v_eobmax;
    }
    if ((int64_t)(uint32_t)max_abs > narrow_max || (int64_t)(max_abs >> 32) > narrow_max) {
        *pending |= group_bit;
        return v_eobmax;
    }
    if (!(uint32_t)over) {
        qm_zero_8(qcoeff, dqcoeff);
    } else {
        v_eobmax = quantize_fp_qm_narrow_8(
            c0, c1, a0, a1, qm, iqm, iscan, qcoeff, dqcoeff, lo, ac, v_eobmax, log_scale, direct);
    }
    if (!(over >> 32)) {
        qm_zero_8(qcoeff + 8, dqcoeff + 8);
    } else {
        v_eobmax = quantize_fp_qm_narrow_8(
            c2, c3, a2, a3, qm + 8, iqm + 8, iscan + 8, qcoeff + 8, dqcoeff + 8, ac, ac, v_eobmax, log_scale, direct);
    }
    return v_eobmax;
}

// The groups of sixteen coefficients whose bit is set in `pending` (bit g: coefficients 16 * g onwards, as far as
// the block goes), in 32-bit lanes for any magnitude, eight at a time as quantize_fp_qm_neon_impl() handles them.
// Returns the largest scan index of a nonzero level among them, -1 without one.
static AOM_FORCE_INLINE int quantize_fp_qm_pending(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                   const int16_t* round_ptr, const int16_t* quant_ptr,
                                                   TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                   const int16_t* dequant_ptr, const int16_t* iscan,
                                                   const QmVal* qm_ptr, const QmVal* iqm_ptr, int log_scale,
                                                   uint64_t pending, const int highbd) {
    const int       qsh          = 31 - (16 - log_scale + AOM_QM_BITS);
    const int32x4_t v_thr_sh     = vdupq_n_s32(AOM_QM_BITS - (1 + log_scale));
    const int32_t   rnd_dc       = ROUND_POWER_OF_TWO(round_ptr[0], log_scale);
    const int32_t   rnd_ac       = ROUND_POWER_OF_TWO(round_ptr[1], log_scale);
    const int32x4_t v_round_ac   = vdupq_n_s32(rnd_ac);
    const int32x4_t v_dequant_ac = vdupq_n_s32(dequant_ptr[1]);
    const int32x4_t v_thr_ac     = vshlq_s32(v_dequant_ac, v_thr_sh);
    const int32x4_t v_qscale_ac  = vshlq_s32(vdupq_n_s32(quant_ptr[1]), vdupq_n_s32(qsh));
    int16x8_t       v_eobmax     = vdupq_n_s16(-1);
    for (; pending; pending &= pending - 1) {
        const intptr_t first = 16 * (intptr_t)svt_ctzll(pending);
        const intptr_t last  = AOMMIN(first + 16, n_coeffs);
        for (intptr_t i = first; i < last; i += 8) {
            // the first group carries the DC constants in lane 0
            const int32x4_t v_round_lo   = i ? v_round_ac : dc_ac_s32(rnd_dc, rnd_ac);
            const int32x4_t v_dequant_lo = i ? v_dequant_ac : dc_ac_s32(dequant_ptr[0], dequant_ptr[1]);
            const int32x4_t v_thr_lo     = vshlq_s32(v_dequant_lo, v_thr_sh);
            const int32x4_t v_qscale_lo  = i ? v_qscale_ac
                                             : vshlq_s32(dc_ac_s32(quant_ptr[0], quant_ptr[1]), vdupq_n_s32(qsh));

            const int32x4_t  coeff_lo = vld1q_s32(coeff_ptr + i);
            const int32x4_t  coeff_hi = vld1q_s32(coeff_ptr + i + 4);
            const int32x4_t  abs_lo = vabsq_s32(coeff_lo), abs_hi = vabsq_s32(coeff_hi);
            const uint16x8_t v_qm     = vmovl_u8(vld1_u8(qm_ptr + i));
            const int32x4_t  v_qm_lo  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_qm)));
            const int32x4_t  v_qm_hi  = vreinterpretq_s32_u32(vmovl_high_u16(v_qm));
            const uint32x4_t below_lo = vcgtq_s32(v_thr_lo, vmulq_s32(abs_lo, v_qm_lo));
            const uint32x4_t below_hi = vcgtq_s32(v_thr_ac, vmulq_s32(abs_hi, v_qm_hi));
            const uint16x8_t v_iqm    = vmovl_u8(vld1_u8(iqm_ptr + i));
            const int32x4_t  v_iqm_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_iqm)));
            const int32x4_t  v_iqm_hi = vreinterpretq_s32_u32(vmovl_high_u16(v_iqm));
            const uint32x4_t nz_lo    = quantize_fp_qm_4(coeff_lo,
                                                      abs_lo,
                                                      below_lo,
                                                      v_qm_lo,
                                                      v_iqm_lo,
                                                      v_round_lo,
                                                      v_qscale_lo,
                                                      v_dequant_lo,
                                                      qcoeff_ptr + i,
                                                      dqcoeff_ptr + i,
                                                      log_scale,
                                                      highbd);
            const uint32x4_t nz_hi    = quantize_fp_qm_4(coeff_hi,
                                                      abs_hi,
                                                      below_hi,
                                                      v_qm_hi,
                                                      v_iqm_hi,
                                                      v_round_ac,
                                                      v_qscale_ac,
                                                      v_dequant_ac,
                                                      qcoeff_ptr + i + 4,
                                                      dqcoeff_ptr + i + 4,
                                                      log_scale,
                                                      highbd);
            v_eobmax = get_max_lane_eob(iscan + i, v_eobmax, vcombine_u16(vmovn_u32(nz_lo), vmovn_u32(nz_hi)));
        }
    }
    return vmaxvq_s16(v_eobmax);
}

static NOINLINE int quantize_fp_qm_pending_lbd(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* round_ptr,
                                               const int16_t* quant_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                               const int16_t* dequant_ptr, const int16_t* iscan, const QmVal* qm_ptr,
                                               const QmVal* iqm_ptr, int log_scale, uint64_t pending) {
    return quantize_fp_qm_pending(coeff_ptr,
                                  n_coeffs,
                                  round_ptr,
                                  quant_ptr,
                                  qcoeff_ptr,
                                  dqcoeff_ptr,
                                  dequant_ptr,
                                  iscan,
                                  qm_ptr,
                                  iqm_ptr,
                                  log_scale,
                                  pending,
                                  /*highbd=*/0);
}

static NOINLINE int quantize_fp_qm_pending_hbd(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* round_ptr,
                                               const int16_t* quant_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                               const int16_t* dequant_ptr, const int16_t* iscan, const QmVal* qm_ptr,
                                               const QmVal* iqm_ptr, int log_scale, uint64_t pending) {
    return quantize_fp_qm_pending(coeff_ptr,
                                  n_coeffs,
                                  round_ptr,
                                  quant_ptr,
                                  qcoeff_ptr,
                                  dqcoeff_ptr,
                                  dequant_ptr,
                                  iscan,
                                  qm_ptr,
                                  iqm_ptr,
                                  log_scale,
                                  pending,
                                  /*highbd=*/1);
}

// Groups of sixteen aligned to the block; the result holds the scan indices of the nonzero levels found.
static AOM_FORCE_INLINE int16x8_t quantize_fp_qm_groups_neon(
    const TranLow* coeff_ptr, intptr_t n_coeffs, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* iscan,
    const QmVal* qm_ptr, const QmVal* iqm_ptr, const QmFpVectors* dc, const QmFpVectors* ac, uint64_t skip_add_dc,
    uint64_t skip_add_ac, int64_t narrow_max, uint64_t* pending, int log_scale, const int direct) {
    int16x8_t v_eobmax = quantize_fp_qm_16(coeff_ptr,
                                           qcoeff_ptr,
                                           dqcoeff_ptr,
                                           iscan,
                                           qm_ptr,
                                           iqm_ptr,
                                           dc,
                                           ac,
                                           skip_add_dc,
                                           narrow_max,
                                           vdupq_n_s16(-1),
                                           pending,
                                           1,
                                           log_scale,
                                           direct);
    // the zero test of a group needs the coefficients and the outputs only, so these pointers carry the loop
    const TranLow* const end = coeff_ptr + (n_coeffs & ~(intptr_t)15);
    const TranLow*       c   = coeff_ptr + 16;
    TranLow*             q   = qcoeff_ptr + 16;
    TranLow*             dq  = dqcoeff_ptr + 16;
    for (; c != end; c += 16, q += 16, dq += 16) {
        const intptr_t i = c - coeff_ptr;
        v_eobmax         = quantize_fp_qm_16(c,
                                     q,
                                     dq,
                                     iscan + i,
                                     qm_ptr + i,
                                     iqm_ptr + i,
                                     ac,
                                     ac,
                                     skip_add_ac,
                                     narrow_max,
                                     v_eobmax,
                                     pending,
                                     (uint64_t)1 << (i >> 4),
                                     log_scale,
                                     direct);
    }
    return v_eobmax;
}

// Fast-quantize path with a quantization matrix for the parameters of 8-bit and 10-bit video: coefficients are
// handled in groups of sixteen aligned to the block (see quantize_fp_qm_16()). Returns false, without writing
// anything, for parameters outside the conditions below; the caller then runs quantize_fp_qm_neon_impl().
//   n_coeffs <= 1024, log_scale in 0..2, round >= 0, quant >= 0,
//   dequant in 1..8191 (every quantizer of 8-bit and 10-bit video, whose largest step is 7312).
// With these every threshold is at least 4, so a coefficient of magnitude 0 is below it, and each path computes the
// same function of a coefficient and its position as quantize_fp_qm_neon_impl(), whose result then does not depend
// on how it groups the coefficients either; the paths differ in the lane width only.
//   All below the threshold: magnitude * weight <= (thr - 1) / max_weight * max_weight < thr.
//   16-bit lanes, level <= 32767: level = ((magnitude + round) * weight * quant) >> shift with magnitude + round <=
//   65535 and weight <= max_weight, so it is enough that 65535 * max_weight * quant < 32768 << shift. That is checked
//   here; at log_scale 0 it holds for every quantizer of 8-bit and 10-bit video (quant <= 16384 and max_weight <= 34
//   give 2^35.1 < 2^36), at log_scale 1 and 2 for all but the smallest steps (4, and 4 to 8). Otherwise no group
//   takes that path (narrow_max = -1). The 8-bit kernel also clamps magnitude + round to 32767, which a limit of
//   32767 - round on the magnitude makes a no-op.
static AOM_FORCE_INLINE bool quantize_fp_qm_fast_neon(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                      const int16_t* round_ptr, const int16_t* quant_ptr,
                                                      TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                      const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                      const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                                      int log_scale, const int highbd) {
    assert(qm_ptr != NULL && iqm_ptr != NULL);
    assert(n_coeffs >= 16 && (n_coeffs & 7) == 0);

    const int32_t dequant_dc = dequant_ptr[0], dequant_ac = dequant_ptr[1];
    const int32_t quant_dc = quant_ptr[0], quant_ac = quant_ptr[1];
    if (n_coeffs > 1024 || (uint32_t)log_scale > 2 || (uint32_t)(dequant_dc - 1) >= 8191 ||
        (uint32_t)(dequant_ac - 1) >= 8191 || (round_ptr[0] | round_ptr[1] | quant_dc | quant_ac) < 0) {
        return false;
    }
    const int      shift_q    = 16 - log_scale + AOM_QM_BITS;
    const int      thr_sh     = AOM_QM_BITS - (1 + log_scale);
    const int32_t  round_dc   = ROUND_POWER_OF_TWO(round_ptr[0], log_scale);
    const int32_t  round_ac   = ROUND_POWER_OF_TWO(round_ptr[1], log_scale);
    const uint32_t thr_dc     = (uint32_t)dequant_dc << thr_sh;
    const uint32_t thr_ac     = (uint32_t)dequant_ac << thr_sh;
    const uint32_t max_weight = (uint32_t)qm_max_weight_cached_neon(qm_ptr, n_coeffs);
    const uint32_t weight_div = AOMMAX(max_weight, 1);
    // largest magnitudes that are below the threshold for every weight, as the addends of quantize_fp_qm_16(); the
    // first half of the first group holds the DC coefficient and takes the smaller of the two thresholds
    const uint64_t skip_ac     = 0x7fffffff - (thr_ac - 1) / weight_div;
    const uint64_t skip_dc     = 0x7fffffff - (AOMMIN(thr_dc, thr_ac) - 1) / weight_div;
    const uint64_t skip_add_ac = skip_ac << 32 | skip_ac;
    const uint64_t skip_add_dc = skip_ac << 32 | skip_dc;
    int64_t        narrow_max  = -1;
    if ((uint64_t)65535 * max_weight * (uint32_t)AOMMAX(quant_dc, quant_ac) < ((uint64_t)32768 << shift_q)) {
        narrow_max = (highbd ? 65535 : 32767) - AOMMAX(round_dc, round_ac);
    }

    QmFpVectors ac, dc;
    ac.round  = vdupq_n_u16((uint16_t)round_ac);
    ac.thr    = vdupq_n_u32(thr_ac);
    ac.qscale = vdupq_n_s32(quant_ac << (31 - shift_q));
    ac.deq_hi = vdup_n_u8((uint8_t)(dequant_ac >> AOM_QM_BITS));
    ac.deq_lo = vdup_n_u8((uint8_t)(dequant_ac & ((1 << AOM_QM_BITS) - 1)));
    dc.round  = vsetq_lane_u16((uint16_t)round_dc, ac.round, 0);
    dc.thr    = vsetq_lane_u32(thr_dc, ac.thr, 0);
    dc.qscale = vsetq_lane_s32(quant_dc << (31 - shift_q), ac.qscale, 0);
    dc.deq_hi = vset_lane_u8((uint8_t)(dequant_dc >> AOM_QM_BITS), ac.deq_hi, 0);
    dc.deq_lo = vset_lane_u8((uint8_t)(dequant_dc & ((1 << AOM_QM_BITS) - 1)), ac.deq_lo, 0);

    // a last group of eight (n_coeffs is not a multiple of 16) goes to the 32-bit lanes
    uint64_t  pending = (n_coeffs & 8) ? (uint64_t)1 << (n_coeffs >> 4) : 0;
    int16x8_t v_eobmax;
    if (log_scale == 0 && AOMMAX(dequant_dc, dequant_ac) <= 4095) {
        v_eobmax = quantize_fp_qm_groups_neon(coeff_ptr,
                                              n_coeffs,
                                              qcoeff_ptr,
                                              dqcoeff_ptr,
                                              iscan,
                                              qm_ptr,
                                              iqm_ptr,
                                              &dc,
                                              &ac,
                                              skip_add_dc,
                                              skip_add_ac,
                                              narrow_max,
                                              &pending,
                                              0,
                                              1);
    } else {
        v_eobmax = quantize_fp_qm_groups_neon(coeff_ptr,
                                              n_coeffs,
                                              qcoeff_ptr,
                                              dqcoeff_ptr,
                                              iscan,
                                              qm_ptr,
                                              iqm_ptr,
                                              &dc,
                                              &ac,
                                              skip_add_dc,
                                              skip_add_ac,
                                              narrow_max,
                                              &pending,
                                              log_scale,
                                              0);
    }
    int eob = vmaxvq_s16(v_eobmax);
    if (pending) {
        const int eob_32 = highbd ? quantize_fp_qm_pending_hbd(coeff_ptr,
                                                               n_coeffs,
                                                               round_ptr,
                                                               quant_ptr,
                                                               qcoeff_ptr,
                                                               dqcoeff_ptr,
                                                               dequant_ptr,
                                                               iscan,
                                                               qm_ptr,
                                                               iqm_ptr,
                                                               log_scale,
                                                               pending)
                                  : quantize_fp_qm_pending_lbd(coeff_ptr,
                                                               n_coeffs,
                                                               round_ptr,
                                                               quant_ptr,
                                                               qcoeff_ptr,
                                                               dqcoeff_ptr,
                                                               dequant_ptr,
                                                               iscan,
                                                               qm_ptr,
                                                               iqm_ptr,
                                                               log_scale,
                                                               pending);
        eob              = AOMMAX(eob, eob_32);
    }
    *eob_ptr = (uint16_t)(eob + 1);
    return true;
}

// quantize_fp_qm_neon_impl() out of line, for the parameters quantize_fp_qm_fast_neon() leaves to it.
static NOINLINE void quantize_fp_qm_general_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* round_ptr,
                                                 const int16_t* quant_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                 const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* iscan,
                                                 const QmVal* qm_ptr, const QmVal* iqm_ptr, int log_scale, int highbd) {
    if (highbd) {
        quantize_fp_qm_neon_impl(coeff_ptr,
                                 n_coeffs,
                                 round_ptr,
                                 quant_ptr,
                                 qcoeff_ptr,
                                 dqcoeff_ptr,
                                 dequant_ptr,
                                 eob_ptr,
                                 iscan,
                                 qm_ptr,
                                 iqm_ptr,
                                 log_scale,
                                 /*highbd=*/1);
    } else {
        quantize_fp_qm_neon_impl(coeff_ptr,
                                 n_coeffs,
                                 round_ptr,
                                 quant_ptr,
                                 qcoeff_ptr,
                                 dqcoeff_ptr,
                                 dequant_ptr,
                                 eob_ptr,
                                 iscan,
                                 qm_ptr,
                                 iqm_ptr,
                                 log_scale,
                                 /*highbd=*/0);
    }
}

void svt_av1_quantize_fp_qm_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                 const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                 TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                 uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan, const QmVal* qm_ptr,
                                 const QmVal* iqm_ptr, int16_t log_scale) {
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    (void)scan;
    if (!quantize_fp_qm_fast_neon(coeff_ptr,
                                  n_coeffs,
                                  round_ptr,
                                  quant_ptr,
                                  qcoeff_ptr,
                                  dqcoeff_ptr,
                                  dequant_ptr,
                                  eob_ptr,
                                  iscan,
                                  qm_ptr,
                                  iqm_ptr,
                                  log_scale,
                                  /*highbd=*/0)) {
        quantize_fp_qm_general_neon(coeff_ptr,
                                    n_coeffs,
                                    round_ptr,
                                    quant_ptr,
                                    qcoeff_ptr,
                                    dqcoeff_ptr,
                                    dequant_ptr,
                                    eob_ptr,
                                    iscan,
                                    qm_ptr,
                                    iqm_ptr,
                                    log_scale,
                                    /*highbd=*/0);
    }
}

void svt_av1_highbd_quantize_fp_qm_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                        const int16_t* round_ptr, const int16_t* quant_ptr,
                                        const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                        const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                        const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                        int16_t log_scale) {
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    (void)scan;
    if (!quantize_fp_qm_fast_neon(coeff_ptr,
                                  n_coeffs,
                                  round_ptr,
                                  quant_ptr,
                                  qcoeff_ptr,
                                  dqcoeff_ptr,
                                  dequant_ptr,
                                  eob_ptr,
                                  iscan,
                                  qm_ptr,
                                  iqm_ptr,
                                  log_scale,
                                  /*highbd=*/1)) {
        quantize_fp_qm_general_neon(coeff_ptr,
                                    n_coeffs,
                                    round_ptr,
                                    quant_ptr,
                                    qcoeff_ptr,
                                    dqcoeff_ptr,
                                    dequant_ptr,
                                    eob_ptr,
                                    iscan,
                                    qm_ptr,
                                    iqm_ptr,
                                    log_scale,
                                    /*highbd=*/1);
    }
}

// Quantize 4 coefficients (one int32x4 lane group) for the quantize_b qm path.
// Mirrors svt_aom_quantize_b_c()'s qm branch and the AVX2 quantize_qm() kernel: the two-step quantizer
// ((q*wt*quant) >> 16 + q*wt) * quant_shift, kept in 32-bit lanes via doubling multiply-highs. Returns a
// per-lane non-zero mask derived from abs_qcoeff (matching the C reference's EOB rule).
static AOM_FORCE_INLINE uint32x4_t quantize_b_qm_4(int32x4_t v_coeff, int32x4_t v_abs, uint32x4_t v_below,
                                                   int32x4_t v_qm, int32x4_t v_iqm, int32x4_t v_round,
                                                   int32x4_t v_quant_scl, int32x4_t v_qshift_scl, int32x4_t v_dequant,
                                                   TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, int log_scale,
                                                   const int highbd) {
    const int32x4_t v_sign = vshrq_n_s32(v_coeff, 31);
    // Two-step quantizer, both "(x * y) >> n" steps folded into doubling multiply-highs so the wide
    // products stay inside the SQDMULH accumulators (no widening to 64-bit lanes):
    //   step1: (q_wt * quant) >> 16      == vqdmulhq_s32(q_wt, v_quant_scl),  v_quant_scl  = quant << 15
    //   abs_q: (q2 * quant_shift) >> S    == vqdmulhq_s32(q2, v_qshift_scl),  v_qshift_scl = quant_shift << (31 - S)
    // Both scales are loop-invariant, so the caller precomputes them once. The 32-bit-lane operands all
    // stay inside int32 at every bit depth: with |coeff| < 1 << (7 + bd) and wt a uint8 (<= 255), q_wt =
    // (abs + round) * wt <= 2^27; quant is int16 so quant << 15 <= 2^30; q2 = step1 + q_wt <= 2^28;
    // quant_shift << (31 - S) <= 2^27. quant/quant_shift can be negative, but SQDMULH's high-half is an
    // arithmetic shift matching the C >>, and saturation is unreachable (no operand reaches INT32_MIN).
    // Only the 8-bit int16 clamp differs from the (unclamped) highbd path.
    int32x4_t v_q = vaddq_s32(v_abs, v_round);
    if (!highbd) {
        v_q = vminq_s32(v_q, vdupq_n_s32(32767));
    }
    // Zero the below-ZBIN lanes once, here; the zeros propagate through the two-step quant to qcoeff and
    // dqcoeff, so no separate output masking is needed.
    v_q                      = vbicq_s32(v_q, vreinterpretq_s32_u32(v_below));
    const int32x4_t v_q_wt   = vmulq_s32(v_q, v_qm);
    const int32x4_t v_step1  = vqdmulhq_s32(v_q_wt, v_quant_scl);
    const int32x4_t v_q2     = vaddq_s32(v_step1, v_q_wt);
    int32x4_t       v_qcoeff = vqdmulhq_s32(v_q2, v_qshift_scl);

    // dequant_qm = (dequant * iqm + (1 << (AOM_QM_BITS - 1))) >> AOM_QM_BITS; abs_dqcoeff = (abs_qcoeff * dequant_qm) >> log_scale.
    // The C reference computes this product in int32 at every bit depth, so keep it in 32-bit lanes; a
    // wider synthetic abs_qcoeff wraps mod 2^32 identically to the reference (real coefficients never overflow).
    int32x4_t v_dqf     = vmulq_s32(v_dequant, v_iqm);
    v_dqf               = vrshrq_n_s32(v_dqf, AOM_QM_BITS);
    int32x4_t v_dqcoeff = vshlq_s32(vmulq_s32(v_qcoeff, v_dqf), vdupq_n_s32(-log_scale));

    // Per-lane nonzero mask (abs_qcoeff != 0; below-ZBIN lanes excluded), scanned by the caller for the EOB.
    const uint32x4_t v_nz = vbicq_u32(vmvnq_u32(vceqzq_s32(v_qcoeff)), v_below);

    v_qcoeff  = vsubq_s32(veorq_s32(v_qcoeff, v_sign), v_sign);
    v_dqcoeff = vsubq_s32(veorq_s32(v_dqcoeff, v_sign), v_sign);
    vst1q_s32(qcoeff_ptr, v_qcoeff);
    vst1q_s32(dqcoeff_ptr, v_dqcoeff);
    return v_nz;
}

static AOM_FORCE_INLINE void quantize_b_qm_neon_impl(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                     const int16_t* zbin_ptr, const int16_t* round_ptr,
                                                     const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                                     TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                     const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                     const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                                     int log_scale, const int highbd) {
    assert(qm_ptr != NULL && iqm_ptr != NULL);
    assert(n_coeffs >= 16 && (n_coeffs & 7) == 0);

    const int     shift_q = 16 - log_scale + AOM_QM_BITS;
    const int32_t rnd_dc  = log_scale > 0 ? (round_ptr[0] + (1 << (log_scale - 1))) >> log_scale : round_ptr[0];
    const int32_t rnd_ac  = log_scale > 0 ? (round_ptr[1] + (1 << (log_scale - 1))) >> log_scale : round_ptr[1];
    const int32_t zbn_dc  = (log_scale > 0 ? (zbin_ptr[0] + (1 << (log_scale - 1))) >> log_scale : zbin_ptr[0])
        << AOM_QM_BITS;
    const int32_t zbn_ac = (log_scale > 0 ? (zbin_ptr[1] + (1 << (log_scale - 1))) >> log_scale : zbin_ptr[1])
        << AOM_QM_BITS;

    const int32x4_t v_round_ac    = vdupq_n_s32(rnd_ac);
    const int32x4_t v_quant_ac    = vdupq_n_s32(quant_ptr[1]);
    const int32x4_t v_qshift_ac   = vdupq_n_s32(quant_shift_ptr[1]);
    const int32x4_t v_dequant_ac  = vdupq_n_s32(dequant_ptr[1]);
    const int32x4_t v_zbin_thr_ac = vdupq_n_s32(zbn_ac);
    // quant << 15 and quant_shift << (31 - shift_q) are loop-invariant; precompute them once.
    const int       qsh             = 31 - shift_q;
    const int32x4_t v_quant_scl_ac  = vshlq_s32(v_quant_ac, vdupq_n_s32(15));
    const int32x4_t v_qshift_scl_ac = vshlq_s32(v_qshift_ac, vdupq_n_s32(qsh));

    int32x4_t v_round_lo      = dc_ac_s32(rnd_dc, rnd_ac);
    int32x4_t v_quant_lo      = dc_ac_s32(quant_ptr[0], quant_ptr[1]);
    int32x4_t v_qshift_lo     = dc_ac_s32(quant_shift_ptr[0], quant_shift_ptr[1]);
    int32x4_t v_dequant_lo    = dc_ac_s32(dequant_ptr[0], dequant_ptr[1]);
    int32x4_t v_zbin_thr_lo   = dc_ac_s32(zbn_dc, zbn_ac);
    int32x4_t v_quant_scl_lo  = vshlq_s32(v_quant_lo, vdupq_n_s32(15));
    int32x4_t v_qshift_scl_lo = vshlq_s32(v_qshift_lo, vdupq_n_s32(qsh));

    // abs * wt < zbin holds for every weight when abs <= (zbin - 1) / max(wt).
    const int32x4_t v_skip_max = vdupq_n_s32(
        zbn_ac > 0 ? (zbn_ac - 1) / AOMMAX(qm_max_weight_cached_neon(qm_ptr, n_coeffs), 1) : -1);

    int16x8_t v_eobmax = vdupq_n_s16(-1);
    for (intptr_t i = 0;;) {
        // AC-only groups: most high-frequency coefficients are tiny, so test sixteen at a time first.
        while (i >= 8 && i + 16 <= n_coeffs &&
               qm_skip16_neon(coeff_ptr + i, qcoeff_ptr + i, dqcoeff_ptr + i, v_skip_max)) {
            i += 16;
        }
        if (i >= n_coeffs) {
            break;
        }
        const int32x4_t  coeff_lo = vld1q_s32(coeff_ptr + i);
        const int32x4_t  coeff_hi = vld1q_s32(coeff_ptr + i + 4);
        const int32x4_t  abs_lo = vabsq_s32(coeff_lo), abs_hi = vabsq_s32(coeff_hi);
        const uint16x8_t v_qm     = vmovl_u8(vld1_u8(qm_ptr + i));
        const int32x4_t  v_qm_lo  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_qm)));
        const int32x4_t  v_qm_hi  = vreinterpretq_s32_u32(vmovl_high_u16(v_qm));
        const uint32x4_t below_lo = vcgtq_s32(v_zbin_thr_lo, vmulq_s32(abs_lo, v_qm_lo));
        const uint32x4_t below_hi = vcgtq_s32(v_zbin_thr_ac, vmulq_s32(abs_hi, v_qm_hi));
        // Eight below-threshold coefficients cannot affect EOB. Skip inverse-matrix
        // loads and scan tracking as well as the quantization and dequantization math.
        if (vminvq_u32(vandq_u32(below_lo, below_hi)) == UINT32_MAX) {
            vst1q_s32(qcoeff_ptr + i, vdupq_n_s32(0));
            vst1q_s32(qcoeff_ptr + i + 4, vdupq_n_s32(0));
            vst1q_s32(dqcoeff_ptr + i, vdupq_n_s32(0));
            vst1q_s32(dqcoeff_ptr + i + 4, vdupq_n_s32(0));
        } else {
            const uint16x8_t v_iqm     = vmovl_u8(vld1_u8(iqm_ptr + i));
            const int32x4_t  v_iqm_lo  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_iqm)));
            const int32x4_t  v_iqm_hi  = vreinterpretq_s32_u32(vmovl_high_u16(v_iqm));
            const uint32x4_t nz_lo     = quantize_b_qm_4(coeff_lo,
                                                     abs_lo,
                                                     below_lo,
                                                     v_qm_lo,
                                                     v_iqm_lo,
                                                     v_round_lo,
                                                     v_quant_scl_lo,
                                                     v_qshift_scl_lo,
                                                     v_dequant_lo,
                                                     qcoeff_ptr + i,
                                                     dqcoeff_ptr + i,
                                                     log_scale,
                                                     highbd);
            const uint32x4_t nz_hi     = quantize_b_qm_4(coeff_hi,
                                                     abs_hi,
                                                     below_hi,
                                                     v_qm_hi,
                                                     v_iqm_hi,
                                                     v_round_ac,
                                                     v_quant_scl_ac,
                                                     v_qshift_scl_ac,
                                                     v_dequant_ac,
                                                     qcoeff_ptr + i + 4,
                                                     dqcoeff_ptr + i + 4,
                                                     log_scale,
                                                     highbd);
            const uint16x8_t v_nz_mask = vcombine_u16(vmovn_u32(nz_lo), vmovn_u32(nz_hi));
            v_eobmax                   = get_max_lane_eob(iscan + i, v_eobmax, v_nz_mask);
        }
        i += 8;
        if (i >= n_coeffs) {
            break;
        }
        v_round_lo      = v_round_ac;
        v_quant_scl_lo  = v_quant_scl_ac;
        v_qshift_scl_lo = v_qshift_scl_ac;
        v_dequant_lo    = v_dequant_ac;
        v_zbin_thr_lo   = v_zbin_thr_ac;
    }
    *eob_ptr = get_max_eob(v_eobmax);
}

// Constants of the 16-bit-lane quantize_b path for eight positions: rounding offset, zbin threshold on magnitude *
// weight, quant << 15, quant_shift << (31 - shift), and dequant split as 32 * deq_hi + deq_lo.
typedef struct {
    uint16x8_t round;
    uint32x4_t zbin;
    int32x4_t  quant_scl;
    int32x4_t  qshift_scl;
    uint8x8_t  deq_hi;
    uint8x8_t  deq_lo;
} QmBVectors;

// Eight coefficients with magnitude + round <= 65535, as quantize_fp_qm_narrow_8() but with the two-step quantizer of
// quantize_b_qm_4(): (magnitude + round) * weight < 2^24 is its 32-bit operand, the multiply-highs are the same.
static AOM_FORCE_INLINE int16x8_t quantize_b_qm_narrow_8(int32x4_t c_lo, int32x4_t c_hi, int32x4_t a_lo, int32x4_t a_hi,
                                                         const QmVal* qm, const QmVal* iqm, const int16_t* iscan,
                                                         TranLow* qcoeff, TranLow* dqcoeff, const QmBVectors* lo,
                                                         const QmBVectors* ac, int16x8_t v_eobmax, int log_scale,
                                                         const int direct) {
    const uint16x8_t v_abs    = vuzp1q_u16(vreinterpretq_u16_s32(a_lo), vreinterpretq_u16_s32(a_hi));
    const uint16x8_t v_qm     = vmovl_u8(vld1_u8(qm));
    const uint32x4_t below_lo = vcgtq_u32(lo->zbin, vmull_u16(vget_low_u16(v_abs), vget_low_u16(v_qm)));
    const uint32x4_t below_hi = vcgtq_u32(ac->zbin, vmull_high_u16(v_abs, v_qm));
    const uint16x8_t v_x      = vaddq_u16(v_abs, lo->round);
    const int32x4_t  p_lo     = vreinterpretq_s32_u32(vmull_u16(vget_low_u16(v_x), vget_low_u16(v_qm)));
    const int32x4_t  p_hi     = vreinterpretq_s32_u32(vmull_high_u16(v_x, v_qm));
    const int32x4_t  q2_lo    = vaddq_s32(vqdmulhq_s32(p_lo, lo->quant_scl), p_lo);
    const int32x4_t  q2_hi    = vaddq_s32(vqdmulhq_s32(p_hi, ac->quant_scl), p_hi);
    const int32x4_t  q_lo     = vbicq_s32(vqdmulhq_s32(q2_lo, lo->qshift_scl), vreinterpretq_s32_u32(below_lo));
    const int32x4_t  q_hi     = vbicq_s32(vqdmulhq_s32(q2_hi, ac->qshift_scl), vreinterpretq_s32_u32(below_hi));
    return qm_finish_8(q_lo,
                       q_hi,
                       c_lo,
                       c_hi,
                       qm_dequant_8(iqm, lo->deq_hi, lo->deq_lo),
                       qcoeff,
                       dqcoeff,
                       iscan,
                       v_eobmax,
                       log_scale,
                       direct);
}

// Sixteen coefficients, as quantize_fp_qm_16().
static AOM_FORCE_INLINE int16x8_t quantize_b_qm_16(const TranLow* coeff, TranLow* qcoeff, TranLow* dqcoeff,
                                                   const int16_t* iscan, const QmVal* qm, const QmVal* iqm,
                                                   const QmBVectors* lo, const QmBVectors* ac, uint64_t skip_add,
                                                   int64_t narrow_max, int16x8_t v_eobmax, uint64_t* pending,
                                                   uint64_t group_bit, int log_scale, const int direct) {
    const int32x4_t c0 = vld1q_s32(coeff), c1 = vld1q_s32(coeff + 4);
    const int32x4_t c2 = vld1q_s32(coeff + 8), c3 = vld1q_s32(coeff + 12);
    const int32x4_t a0 = vabsq_s32(c0), a1 = vabsq_s32(c1), a2 = vabsq_s32(c2), a3 = vabsq_s32(c3);
    const uint64_t  max_abs = qm_max_abs_halves(a0, a1, a2, a3);
    const uint64_t  over    = (max_abs + skip_add) & 0x8000000080000000ull;
    if (!over) {
        qm_zero_8(qcoeff, dqcoeff);
        qm_zero_8(qcoeff + 8, dqcoeff + 8);
        return v_eobmax;
    }
    if ((int64_t)(uint32_t)max_abs > narrow_max || (int64_t)(max_abs >> 32) > narrow_max) {
        *pending |= group_bit;
        return v_eobmax;
    }
    if (!(uint32_t)over) {
        qm_zero_8(qcoeff, dqcoeff);
    } else {
        v_eobmax = quantize_b_qm_narrow_8(
            c0, c1, a0, a1, qm, iqm, iscan, qcoeff, dqcoeff, lo, ac, v_eobmax, log_scale, direct);
    }
    if (!(over >> 32)) {
        qm_zero_8(qcoeff + 8, dqcoeff + 8);
    } else {
        v_eobmax = quantize_b_qm_narrow_8(
            c2, c3, a2, a3, qm + 8, iqm + 8, iscan + 8, qcoeff + 8, dqcoeff + 8, ac, ac, v_eobmax, log_scale, direct);
    }
    return v_eobmax;
}

// The groups of sixteen coefficients whose bit is set in `pending`, as quantize_fp_qm_pending(): 32-bit lanes for
// any magnitude, eight at a time as quantize_b_qm_neon_impl() handles them.
static AOM_FORCE_INLINE int quantize_b_qm_pending(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                                  const int16_t* round_ptr, const int16_t* quant_ptr,
                                                  const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr,
                                                  TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                                  const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                                  int log_scale, uint64_t pending, const int highbd) {
    const int32x4_t v_qsh           = vdupq_n_s32(31 - (16 - log_scale + AOM_QM_BITS));
    const int32_t   rnd_dc          = ROUND_POWER_OF_TWO(round_ptr[0], log_scale);
    const int32_t   rnd_ac          = ROUND_POWER_OF_TWO(round_ptr[1], log_scale);
    const int32_t   zbn_dc          = ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale) << AOM_QM_BITS;
    const int32_t   zbn_ac          = ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale) << AOM_QM_BITS;
    const int32x4_t v_round_ac      = vdupq_n_s32(rnd_ac);
    const int32x4_t v_dequant_ac    = vdupq_n_s32(dequant_ptr[1]);
    const int32x4_t v_zbin_ac       = vdupq_n_s32(zbn_ac);
    const int32x4_t v_quant_scl_ac  = vshlq_n_s32(vdupq_n_s32(quant_ptr[1]), 15);
    const int32x4_t v_qshift_scl_ac = vshlq_s32(vdupq_n_s32(quant_shift_ptr[1]), v_qsh);
    int16x8_t       v_eobmax        = vdupq_n_s16(-1);
    for (; pending; pending &= pending - 1) {
        const intptr_t first = 16 * (intptr_t)svt_ctzll(pending);
        const intptr_t last  = AOMMIN(first + 16, n_coeffs);
        for (intptr_t i = first; i < last; i += 8) {
            // the first group carries the DC constants in lane 0
            const int32x4_t v_round_lo      = i ? v_round_ac : dc_ac_s32(rnd_dc, rnd_ac);
            const int32x4_t v_dequant_lo    = i ? v_dequant_ac : dc_ac_s32(dequant_ptr[0], dequant_ptr[1]);
            const int32x4_t v_zbin_lo       = i ? v_zbin_ac : dc_ac_s32(zbn_dc, zbn_ac);
            const int32x4_t v_quant_scl_lo  = i ? v_quant_scl_ac
                                                : vshlq_n_s32(dc_ac_s32(quant_ptr[0], quant_ptr[1]), 15);
            const int32x4_t v_qshift_scl_lo = i ? v_qshift_scl_ac
                                                : vshlq_s32(dc_ac_s32(quant_shift_ptr[0], quant_shift_ptr[1]), v_qsh);

            const int32x4_t  coeff_lo = vld1q_s32(coeff_ptr + i);
            const int32x4_t  coeff_hi = vld1q_s32(coeff_ptr + i + 4);
            const int32x4_t  abs_lo = vabsq_s32(coeff_lo), abs_hi = vabsq_s32(coeff_hi);
            const uint16x8_t v_qm     = vmovl_u8(vld1_u8(qm_ptr + i));
            const int32x4_t  v_qm_lo  = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_qm)));
            const int32x4_t  v_qm_hi  = vreinterpretq_s32_u32(vmovl_high_u16(v_qm));
            const uint32x4_t below_lo = vcgtq_s32(v_zbin_lo, vmulq_s32(abs_lo, v_qm_lo));
            const uint32x4_t below_hi = vcgtq_s32(v_zbin_ac, vmulq_s32(abs_hi, v_qm_hi));
            const uint16x8_t v_iqm    = vmovl_u8(vld1_u8(iqm_ptr + i));
            const int32x4_t  v_iqm_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_iqm)));
            const int32x4_t  v_iqm_hi = vreinterpretq_s32_u32(vmovl_high_u16(v_iqm));
            const uint32x4_t nz_lo    = quantize_b_qm_4(coeff_lo,
                                                     abs_lo,
                                                     below_lo,
                                                     v_qm_lo,
                                                     v_iqm_lo,
                                                     v_round_lo,
                                                     v_quant_scl_lo,
                                                     v_qshift_scl_lo,
                                                     v_dequant_lo,
                                                     qcoeff_ptr + i,
                                                     dqcoeff_ptr + i,
                                                     log_scale,
                                                     highbd);
            const uint32x4_t nz_hi    = quantize_b_qm_4(coeff_hi,
                                                     abs_hi,
                                                     below_hi,
                                                     v_qm_hi,
                                                     v_iqm_hi,
                                                     v_round_ac,
                                                     v_quant_scl_ac,
                                                     v_qshift_scl_ac,
                                                     v_dequant_ac,
                                                     qcoeff_ptr + i + 4,
                                                     dqcoeff_ptr + i + 4,
                                                     log_scale,
                                                     highbd);
            v_eobmax = get_max_lane_eob(iscan + i, v_eobmax, vcombine_u16(vmovn_u32(nz_lo), vmovn_u32(nz_hi)));
        }
    }
    return vmaxvq_s16(v_eobmax);
}

static NOINLINE int quantize_b_qm_pending_lbd(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                              const int16_t* round_ptr, const int16_t* quant_ptr,
                                              const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                              const int16_t* dequant_ptr, const int16_t* iscan, const QmVal* qm_ptr,
                                              const QmVal* iqm_ptr, int log_scale, uint64_t pending) {
    return quantize_b_qm_pending(coeff_ptr,
                                 n_coeffs,
                                 zbin_ptr,
                                 round_ptr,
                                 quant_ptr,
                                 quant_shift_ptr,
                                 qcoeff_ptr,
                                 dqcoeff_ptr,
                                 dequant_ptr,
                                 iscan,
                                 qm_ptr,
                                 iqm_ptr,
                                 log_scale,
                                 pending,
                                 /*highbd=*/0);
}

static NOINLINE int quantize_b_qm_pending_hbd(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                              const int16_t* round_ptr, const int16_t* quant_ptr,
                                              const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                              const int16_t* dequant_ptr, const int16_t* iscan, const QmVal* qm_ptr,
                                              const QmVal* iqm_ptr, int log_scale, uint64_t pending) {
    return quantize_b_qm_pending(coeff_ptr,
                                 n_coeffs,
                                 zbin_ptr,
                                 round_ptr,
                                 quant_ptr,
                                 quant_shift_ptr,
                                 qcoeff_ptr,
                                 dqcoeff_ptr,
                                 dequant_ptr,
                                 iscan,
                                 qm_ptr,
                                 iqm_ptr,
                                 log_scale,
                                 pending,
                                 /*highbd=*/1);
}

// Groups of sixteen aligned to the block; the result holds the scan indices of the nonzero levels found.
static AOM_FORCE_INLINE int16x8_t quantize_b_qm_groups_neon(
    const TranLow* coeff_ptr, intptr_t n_coeffs, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* iscan,
    const QmVal* qm_ptr, const QmVal* iqm_ptr, const QmBVectors* dc, const QmBVectors* ac, uint64_t skip_add_dc,
    uint64_t skip_add_ac, int64_t narrow_max, uint64_t* pending, int log_scale, const int direct) {
    int16x8_t v_eobmax = quantize_b_qm_16(coeff_ptr,
                                          qcoeff_ptr,
                                          dqcoeff_ptr,
                                          iscan,
                                          qm_ptr,
                                          iqm_ptr,
                                          dc,
                                          ac,
                                          skip_add_dc,
                                          narrow_max,
                                          vdupq_n_s16(-1),
                                          pending,
                                          1,
                                          log_scale,
                                          direct);
    // the zero test of a group needs the coefficients and the outputs only, so these pointers carry the loop
    const TranLow* const end = coeff_ptr + (n_coeffs & ~(intptr_t)15);
    const TranLow*       c   = coeff_ptr + 16;
    TranLow*             q   = qcoeff_ptr + 16;
    TranLow*             dq  = dqcoeff_ptr + 16;
    for (; c != end; c += 16, q += 16, dq += 16) {
        const intptr_t i = c - coeff_ptr;
        v_eobmax         = quantize_b_qm_16(c,
                                    q,
                                    dq,
                                    iscan + i,
                                    qm_ptr + i,
                                    iqm_ptr + i,
                                    ac,
                                    ac,
                                    skip_add_ac,
                                    narrow_max,
                                    v_eobmax,
                                    pending,
                                    (uint64_t)1 << (i >> 4),
                                    log_scale,
                                    direct);
    }
    return v_eobmax;
}

// quantize_b path with a quantization matrix for the parameters of 8-bit and 10-bit video: coefficients are handled
// in groups of sixteen aligned to the block (see quantize_fp_qm_16()). Returns false, without writing anything, for
// parameters outside the conditions below; the caller then runs quantize_b_qm_neon_impl().
//   n_coeffs <= 1024, log_scale in 0..2, round >= 0, quant_shift >= 0, zbin (after the log_scale rounding) >= 1,
//   dequant in 1..8191 (every quantizer of 8-bit and 10-bit video, whose largest step is 7312).
// With a positive zbin a coefficient of magnitude 0 is below its threshold, and each path computes the same function
// of a coefficient and its position as quantize_b_qm_neon_impl(), whose result then does not depend on how it groups
// the coefficients either; the paths differ in the lane width only.
//   All below the threshold: magnitude * weight <= (zbin - 1) / max_weight * max_weight < zbin.
//   16-bit lanes, level <= 32767: with p = (magnitude + round) * weight <= 65535 * max_weight, the first step gives
//   0 <= p + ((p * quant) >> 16) < 3 * p / 2 for any 16-bit quant, and the level is that times quant_shift >> shift,
//   so it is enough that 3 * 65535 * max_weight * quant_shift <= 65536 << shift. That is checked here; at log_scale 0
//   it holds for every quantizer of 8-bit and 10-bit video (quant_shift <= 16384 and max_weight <= 34 give 2^36.67 <
//   2^37), at log_scale 1 and 2 for all but the smallest steps (below 8, and below 16). Otherwise no group takes that
//   path (narrow_max = -1). The 8-bit kernel also clamps magnitude + round to 32767, which a limit of 32767 - round
//   on the magnitude makes a no-op.
static AOM_FORCE_INLINE bool quantize_b_qm_fast_neon(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                     const int16_t* zbin_ptr, const int16_t* round_ptr,
                                                     const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                                     TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                     const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                     const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                                     int log_scale, const int highbd) {
    assert(qm_ptr != NULL && iqm_ptr != NULL);
    assert(n_coeffs >= 16 && (n_coeffs & 7) == 0);

    const int32_t dequant_dc = dequant_ptr[0], dequant_ac = dequant_ptr[1];
    const int32_t qshift_dc = quant_shift_ptr[0], qshift_ac = quant_shift_ptr[1];
    if (n_coeffs > 1024 || (uint32_t)log_scale > 2 || (uint32_t)(dequant_dc - 1) >= 8191 ||
        (uint32_t)(dequant_ac - 1) >= 8191 || (round_ptr[0] | round_ptr[1] | qshift_dc | qshift_ac) < 0) {
        return false;
    }
    const int32_t zbin_dc = ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale);
    const int32_t zbin_ac = ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale);
    if (zbin_dc < 1 || zbin_ac < 1) {
        return false;
    }
    const int      shift_q    = 16 - log_scale + AOM_QM_BITS;
    const int32_t  round_dc   = ROUND_POWER_OF_TWO(round_ptr[0], log_scale);
    const int32_t  round_ac   = ROUND_POWER_OF_TWO(round_ptr[1], log_scale);
    const uint32_t thr_dc     = (uint32_t)zbin_dc << AOM_QM_BITS;
    const uint32_t thr_ac     = (uint32_t)zbin_ac << AOM_QM_BITS;
    const uint32_t max_weight = (uint32_t)qm_max_weight_cached_neon(qm_ptr, n_coeffs);
    const uint32_t weight_div = AOMMAX(max_weight, 1);
    // largest magnitudes that are below the threshold for every weight, as the addends of quantize_b_qm_16(); the
    // first half of the first group holds the DC coefficient and takes the smaller of the two thresholds
    const uint64_t skip_ac     = 0x7fffffff - (thr_ac - 1) / weight_div;
    const uint64_t skip_dc     = 0x7fffffff - (AOMMIN(thr_dc, thr_ac) - 1) / weight_div;
    const uint64_t skip_add_ac = skip_ac << 32 | skip_ac;
    const uint64_t skip_add_dc = skip_ac << 32 | skip_dc;
    int64_t        narrow_max  = -1;
    if ((uint64_t)(3 * 65535) * max_weight * (uint32_t)AOMMAX(qshift_dc, qshift_ac) <= ((uint64_t)65536 << shift_q)) {
        narrow_max = (highbd ? 65535 : 32767) - AOMMAX(round_dc, round_ac);
    }

    QmBVectors ac, dc;
    ac.round      = vdupq_n_u16((uint16_t)round_ac);
    ac.zbin       = vdupq_n_u32(thr_ac);
    ac.quant_scl  = vdupq_n_s32(quant_ptr[1] * (1 << 15));
    ac.qshift_scl = vdupq_n_s32(qshift_ac << (31 - shift_q));
    ac.deq_hi     = vdup_n_u8((uint8_t)(dequant_ac >> AOM_QM_BITS));
    ac.deq_lo     = vdup_n_u8((uint8_t)(dequant_ac & ((1 << AOM_QM_BITS) - 1)));
    dc.round      = vsetq_lane_u16((uint16_t)round_dc, ac.round, 0);
    dc.zbin       = vsetq_lane_u32(thr_dc, ac.zbin, 0);
    dc.quant_scl  = vsetq_lane_s32(quant_ptr[0] * (1 << 15), ac.quant_scl, 0);
    dc.qshift_scl = vsetq_lane_s32(qshift_dc << (31 - shift_q), ac.qshift_scl, 0);
    dc.deq_hi     = vset_lane_u8((uint8_t)(dequant_dc >> AOM_QM_BITS), ac.deq_hi, 0);
    dc.deq_lo     = vset_lane_u8((uint8_t)(dequant_dc & ((1 << AOM_QM_BITS) - 1)), ac.deq_lo, 0);

    // a last group of eight (n_coeffs is not a multiple of 16) goes to the 32-bit lanes
    uint64_t  pending = (n_coeffs & 8) ? (uint64_t)1 << (n_coeffs >> 4) : 0;
    int16x8_t v_eobmax;
    if (log_scale == 0 && AOMMAX(dequant_dc, dequant_ac) <= 4095) {
        v_eobmax = quantize_b_qm_groups_neon(coeff_ptr,
                                             n_coeffs,
                                             qcoeff_ptr,
                                             dqcoeff_ptr,
                                             iscan,
                                             qm_ptr,
                                             iqm_ptr,
                                             &dc,
                                             &ac,
                                             skip_add_dc,
                                             skip_add_ac,
                                             narrow_max,
                                             &pending,
                                             0,
                                             1);
    } else {
        v_eobmax = quantize_b_qm_groups_neon(coeff_ptr,
                                             n_coeffs,
                                             qcoeff_ptr,
                                             dqcoeff_ptr,
                                             iscan,
                                             qm_ptr,
                                             iqm_ptr,
                                             &dc,
                                             &ac,
                                             skip_add_dc,
                                             skip_add_ac,
                                             narrow_max,
                                             &pending,
                                             log_scale,
                                             0);
    }
    int eob = vmaxvq_s16(v_eobmax);
    if (pending) {
        const int eob_32 = highbd ? quantize_b_qm_pending_hbd(coeff_ptr,
                                                              n_coeffs,
                                                              zbin_ptr,
                                                              round_ptr,
                                                              quant_ptr,
                                                              quant_shift_ptr,
                                                              qcoeff_ptr,
                                                              dqcoeff_ptr,
                                                              dequant_ptr,
                                                              iscan,
                                                              qm_ptr,
                                                              iqm_ptr,
                                                              log_scale,
                                                              pending)
                                  : quantize_b_qm_pending_lbd(coeff_ptr,
                                                              n_coeffs,
                                                              zbin_ptr,
                                                              round_ptr,
                                                              quant_ptr,
                                                              quant_shift_ptr,
                                                              qcoeff_ptr,
                                                              dqcoeff_ptr,
                                                              dequant_ptr,
                                                              iscan,
                                                              qm_ptr,
                                                              iqm_ptr,
                                                              log_scale,
                                                              pending);
        eob              = AOMMAX(eob, eob_32);
    }
    *eob_ptr = (uint16_t)(eob + 1);
    return true;
}

// quantize_b_qm_neon_impl() out of line, for the parameters quantize_b_qm_fast_neon() leaves to it.
static NOINLINE void quantize_b_qm_general_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                                const int16_t* round_ptr, const int16_t* quant_ptr,
                                                const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr,
                                                TranLow* dqcoeff_ptr, const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                                int log_scale, int highbd) {
    if (highbd) {
        quantize_b_qm_neon_impl(coeff_ptr,
                                n_coeffs,
                                zbin_ptr,
                                round_ptr,
                                quant_ptr,
                                quant_shift_ptr,
                                qcoeff_ptr,
                                dqcoeff_ptr,
                                dequant_ptr,
                                eob_ptr,
                                iscan,
                                qm_ptr,
                                iqm_ptr,
                                log_scale,
                                /*highbd=*/1);
    } else {
        quantize_b_qm_neon_impl(coeff_ptr,
                                n_coeffs,
                                zbin_ptr,
                                round_ptr,
                                quant_ptr,
                                quant_shift_ptr,
                                qcoeff_ptr,
                                dqcoeff_ptr,
                                dequant_ptr,
                                eob_ptr,
                                iscan,
                                qm_ptr,
                                iqm_ptr,
                                log_scale,
                                /*highbd=*/0);
    }
}

void svt_av1_quantize_b_qm_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan, const QmVal* qm_ptr,
                                const QmVal* iqm_ptr, const int32_t log_scale) {
    (void)scan;
    if (!quantize_b_qm_fast_neon(coeff_ptr,
                                 n_coeffs,
                                 zbin_ptr,
                                 round_ptr,
                                 quant_ptr,
                                 quant_shift_ptr,
                                 qcoeff_ptr,
                                 dqcoeff_ptr,
                                 dequant_ptr,
                                 eob_ptr,
                                 iscan,
                                 qm_ptr,
                                 iqm_ptr,
                                 log_scale,
                                 /*highbd=*/0)) {
        quantize_b_qm_general_neon(coeff_ptr,
                                   n_coeffs,
                                   zbin_ptr,
                                   round_ptr,
                                   quant_ptr,
                                   quant_shift_ptr,
                                   qcoeff_ptr,
                                   dqcoeff_ptr,
                                   dequant_ptr,
                                   eob_ptr,
                                   iscan,
                                   qm_ptr,
                                   iqm_ptr,
                                   log_scale,
                                   /*highbd=*/0);
    }
}

void svt_av1_highbd_quantize_b_qm_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                       const int16_t* round_ptr, const int16_t* quant_ptr,
                                       const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                       const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                       const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                       const int32_t log_scale) {
    (void)scan;
    if (!quantize_b_qm_fast_neon(coeff_ptr,
                                 n_coeffs,
                                 zbin_ptr,
                                 round_ptr,
                                 quant_ptr,
                                 quant_shift_ptr,
                                 qcoeff_ptr,
                                 dqcoeff_ptr,
                                 dequant_ptr,
                                 eob_ptr,
                                 iscan,
                                 qm_ptr,
                                 iqm_ptr,
                                 log_scale,
                                 /*highbd=*/1)) {
        quantize_b_qm_general_neon(coeff_ptr,
                                   n_coeffs,
                                   zbin_ptr,
                                   round_ptr,
                                   quant_ptr,
                                   quant_shift_ptr,
                                   qcoeff_ptr,
                                   dqcoeff_ptr,
                                   dequant_ptr,
                                   eob_ptr,
                                   iscan,
                                   qm_ptr,
                                   iqm_ptr,
                                   log_scale,
                                   /*highbd=*/1);
    }
}

// Quantizes 8 coefficients whose zbin test is in cond (a lane failing it quantizes to 0); returns the nonzero mask.
static AOM_FORCE_INLINE uint16x8_t quantize_b_8(int16x8_t coeff, int16x8_t abs, uint16x8_t cond, int16x8_t round,
                                                int16x8_t dequant, int16x8_t quant, int16x8_t neg_shift,
                                                TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int log_scale) {
    int16x8_t coeff_sign = vreinterpretq_s16_u16(vcltzq_s16(coeff));

    int16x8_t tmp = vqaddq_s16(abs, round);
    tmp           = vsraq_n_s16(tmp, vqdmulhq_s16(tmp, quant), 1);
    tmp           = vshlq_s16(tmp, neg_shift);
    tmp           = vandq_s16(tmp, vreinterpretq_s16_u16(cond));

    int16x8_t qcoeff = vsubq_s16(veorq_s16(tmp, coeff_sign), coeff_sign);
    store_s16q_to_tran_low(qcoeff_ptr, qcoeff);

    int16x8_t dqcoeff;
    if (log_scale == 0) {
        dqcoeff = vmulq_s16(qcoeff, dequant);
    } else {
        // Shift the product right by log_scale; for log_scale 2 the low bits of its high half fill the top bits.
        const uint16x8_t prod = vreinterpretq_u16_s16(vmulq_s16(tmp, dequant));
        if (log_scale == 1) {
            dqcoeff = vreinterpretq_s16_u16(vshrq_n_u16(prod, 1));
        } else {
            dqcoeff = vorrq_s16(vshlq_n_s16(vqdmulhq_s16(tmp, dequant), 13),
                                vreinterpretq_s16_u16(vshrq_n_u16(prod, 2)));
        }
        dqcoeff = vsubq_s16(veorq_s16(dqcoeff, coeff_sign), coeff_sign);
    }
    store_s16q_to_tran_low(dqcoeff_ptr, dqcoeff);

    return vtstq_s16(qcoeff, qcoeff);
}

static AOM_FORCE_INLINE int16x8_t update_eobmax(int16x8_t eobmax, uint16x8_t nz_mask, const int16_t* iscan) {
    return vbslq_s16(nz_mask, vmaxq_s16(vld1q_s16(iscan), eobmax), eobmax);
}

// Coefficients are in raster order, so the nonzero ones cluster in the first columns of the first rows: each
// 16-coefficient group is tested at once and an all-zero group (most of them) is only zero-filled.
static AOM_FORCE_INLINE void aom_quantize_b_helper_group_neon(const TranLow* coeff_ptr, intptr_t n_coeffs,
                                                              const int16_t* zbin_ptr, const int16_t* round_ptr,
                                                              const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                                              TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                                              const int16_t* dequant_ptr, uint16_t* eob_ptr,
                                                              const int16_t* iscan, const int log_scale) {
    const int zbins[2]  = {ROUND_POWER_OF_TWO(zbin_ptr[0], log_scale), ROUND_POWER_OF_TWO(zbin_ptr[1], log_scale)};
    const int rounds[2] = {ROUND_POWER_OF_TWO(round_ptr[0], log_scale), ROUND_POWER_OF_TWO(round_ptr[1], log_scale)};

    // The shift path is valid only because quant_shift is a power of two.
    assert(quant_shift_ptr[0] == (1 << svt_ctz((unsigned)quant_shift_ptr[0])));
    assert(quant_shift_ptr[1] == (1 << svt_ctz((unsigned)quant_shift_ptr[1])));
    const int16x8_t v_zbins     = vdupq_n_s16(zbins[1]);
    const int16x8_t v_round     = vdupq_n_s16(rounds[1]);
    const int16x8_t v_dequant   = vdupq_n_s16(dequant_ptr[1]);
    const int16x8_t v_quant     = vdupq_n_s16(quant_ptr[1]);
    const int16x8_t v_neg_shift = vdupq_n_s16((int16_t)(svt_ctz((unsigned)quant_shift_ptr[1]) - (16 - log_scale)));
    // DC in lane 0 of the first group
    const int16x8_t v_zbins0     = vsetq_lane_s16(zbins[0], v_zbins, 0);
    const int16x8_t v_round0     = vsetq_lane_s16(rounds[0], v_round, 0);
    const int16x8_t v_dequant0   = vsetq_lane_s16(dequant_ptr[0], v_dequant, 0);
    const int16x8_t v_quant0     = vsetq_lane_s16(quant_ptr[0], v_quant, 0);
    const int16x8_t v_neg_shift0 = vsetq_lane_s16(
        (int16_t)(svt_ctz((unsigned)quant_shift_ptr[0]) - (16 - log_scale)), v_neg_shift, 0);

    int16x8_t v_eobmax = vdupq_n_s16(-1);
    for (intptr_t i = 0; i < n_coeffs; i += 16) {
        const int16x8_t  v_coeff0 = load_tran_low_to_s16q(coeff_ptr + i);
        const int16x8_t  v_coeff1 = load_tran_low_to_s16q(coeff_ptr + i + 8);
        const int16x8_t  v_abs0   = vabsq_s16(v_coeff0);
        const int16x8_t  v_abs1   = vabsq_s16(v_coeff1);
        const uint16x8_t v_cond0  = vcgeq_s16(v_abs0, i ? v_zbins : v_zbins0);
        const uint16x8_t v_cond1  = vcgeq_s16(v_abs1, v_zbins);

        if (!vmaxvq_u16(vorrq_u16(v_cond0, v_cond1))) {
            store_s16q_to_tran_low(qcoeff_ptr + i, vdupq_n_s16(0));
            store_s16q_to_tran_low(qcoeff_ptr + i + 8, vdupq_n_s16(0));
            store_s16q_to_tran_low(dqcoeff_ptr + i, vdupq_n_s16(0));
            store_s16q_to_tran_low(dqcoeff_ptr + i + 8, vdupq_n_s16(0));
            continue;
        }
        if (vmaxvq_u16(v_cond0)) {
            const uint16x8_t v_nz0 = i ? quantize_b_8(v_coeff0,
                                                      v_abs0,
                                                      v_cond0,
                                                      v_round,
                                                      v_dequant,
                                                      v_quant,
                                                      v_neg_shift,
                                                      qcoeff_ptr + i,
                                                      dqcoeff_ptr + i,
                                                      log_scale)
                                       : quantize_b_8(v_coeff0,
                                                      v_abs0,
                                                      v_cond0,
                                                      v_round0,
                                                      v_dequant0,
                                                      v_quant0,
                                                      v_neg_shift0,
                                                      qcoeff_ptr,
                                                      dqcoeff_ptr,
                                                      log_scale);
            v_eobmax               = update_eobmax(v_eobmax, v_nz0, iscan + i);
        } else {
            store_s16q_to_tran_low(qcoeff_ptr + i, vdupq_n_s16(0));
            store_s16q_to_tran_low(dqcoeff_ptr + i, vdupq_n_s16(0));
        }
        if (vmaxvq_u16(v_cond1)) {
            const uint16x8_t v_nz1 = quantize_b_8(v_coeff1,
                                                  v_abs1,
                                                  v_cond1,
                                                  v_round,
                                                  v_dequant,
                                                  v_quant,
                                                  v_neg_shift,
                                                  qcoeff_ptr + i + 8,
                                                  dqcoeff_ptr + i + 8,
                                                  log_scale);
            v_eobmax               = update_eobmax(v_eobmax, v_nz1, iscan + i + 8);
        } else {
            store_s16q_to_tran_low(qcoeff_ptr + i + 8, vdupq_n_s16(0));
            store_s16q_to_tran_low(dqcoeff_ptr + i + 8, vdupq_n_s16(0));
        }
    }
    *eob_ptr = vmaxvq_s16(v_eobmax) + 1;
}

void svt_aom_quantize_b_neon(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                             const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                             TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr, uint16_t* eob_ptr,
                             const int16_t* scan, const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                             const int log_scale) {
    assert(qm_ptr == NULL);
    assert(iqm_ptr == NULL);
    (void)scan;
    (void)qm_ptr;
    (void)iqm_ptr;

    // log_scale for AV1 encoder can be only 0, 1, 2
    switch (log_scale) {
    case 0: {
        aom_quantize_b_helper_group_neon(coeff_ptr,
                                         n_coeffs,
                                         zbin_ptr,
                                         round_ptr,
                                         quant_ptr,
                                         quant_shift_ptr,
                                         qcoeff_ptr,
                                         dqcoeff_ptr,
                                         dequant_ptr,
                                         eob_ptr,
                                         iscan,
                                         0);
        break;
    }
    case 1: {
        aom_quantize_b_helper_group_neon(coeff_ptr,
                                         n_coeffs,
                                         zbin_ptr,
                                         round_ptr,
                                         quant_ptr,
                                         quant_shift_ptr,
                                         qcoeff_ptr,
                                         dqcoeff_ptr,
                                         dequant_ptr,
                                         eob_ptr,
                                         iscan,
                                         1);
        break;
    }
    default: {
        aom_quantize_b_helper_group_neon(coeff_ptr,
                                         n_coeffs,
                                         zbin_ptr,
                                         round_ptr,
                                         quant_ptr,
                                         quant_shift_ptr,
                                         qcoeff_ptr,
                                         dqcoeff_ptr,
                                         dequant_ptr,
                                         eob_ptr,
                                         iscan,
                                         2);
        break;
    }
    }
}

int32_t svt_av1_compute_cul_level_neon(const int16_t* const scan, const int32_t* const quant_coeff, int32_t eob,
                                       int32_t n_coeffs) {
    int32x4_t sum_s32[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};

    if (eob * 12 < n_coeffs) {
        // Sparse: scan-based gather, bounded by eob.
        int32x4_t      zeros    = vdupq_n_s32(0);
        int32_t        sz       = eob;
        const int16_t* scan_ptr = scan;
        while (sz >= 8) {
            const int32_t quant_coeff0 = quant_coeff[scan_ptr[0]];
            const int32_t quant_coeff1 = quant_coeff[scan_ptr[1]];
            const int32_t quant_coeff2 = quant_coeff[scan_ptr[2]];
            const int32_t quant_coeff3 = quant_coeff[scan_ptr[3]];
            const int32_t quant_coeff4 = quant_coeff[scan_ptr[4]];
            const int32_t quant_coeff5 = quant_coeff[scan_ptr[5]];
            const int32_t quant_coeff6 = quant_coeff[scan_ptr[6]];
            const int32_t quant_coeff7 = quant_coeff[scan_ptr[7]];

            int32x4_t quant_coeff_0123 = vcombine_s32(
                vcreate_s32((((uint64_t)quant_coeff1) << 32) | (uint32_t)quant_coeff0),
                vcreate_s32((((uint64_t)quant_coeff3) << 32) | (uint32_t)quant_coeff2));

            int32x4_t quant_coeff_4567 = vcombine_s32(
                vcreate_s32((((uint64_t)quant_coeff5) << 32) | (uint32_t)quant_coeff4),
                vcreate_s32((((uint64_t)quant_coeff7) << 32) | (uint32_t)quant_coeff6));

            sum_s32[0] = vabaq_s32(sum_s32[0], quant_coeff_0123, zeros);
            sum_s32[1] = vabaq_s32(sum_s32[1], quant_coeff_4567, zeros);

            scan_ptr += 8;
            sz -= 8;
        }

        if (sz >= 4) {
            const int32_t quant_coeff0 = quant_coeff[scan_ptr[0]];
            const int32_t quant_coeff1 = quant_coeff[scan_ptr[1]];
            const int32_t quant_coeff2 = quant_coeff[scan_ptr[2]];
            const int32_t quant_coeff3 = quant_coeff[scan_ptr[3]];

            int32x4_t quant_coeff_0123 = vcombine_s32(
                vcreate_s32((((uint64_t)quant_coeff1) << 32) | (uint32_t)quant_coeff0),
                vcreate_s32((((uint64_t)quant_coeff3) << 32) | (uint32_t)quant_coeff2));

            sum_s32[0] = vabaq_s32(sum_s32[0], quant_coeff_0123, zeros);

            scan_ptr += 4;
            sz -= 4;
        }

        int sum = 0;
        while (sz) {
            sum += abs(quant_coeff[*scan_ptr]);
            scan_ptr++;
            sz--;
        }
        return sum + vaddvq_s32(vaddq_s32(sum_s32[0], sum_s32[1]));
    }

    // Dense: gather-free linear over the whole block. n_coeffs is always a multiple of 16. The caller clamps the
    // result to COEFF_CONTEXT_MASK (as the C version does by stopping there), so the scan stops once the sum of
    // the chunks seen so far reaches it; a dense block usually gets there within the first chunk.
    assert(n_coeffs % 16 == 0);
    for (int32_t i = 0; i < n_coeffs; i += 16) {
        sum_s32[0] = vaddq_s32(sum_s32[0], vabsq_s32(vld1q_s32(quant_coeff + i)));
        sum_s32[1] = vaddq_s32(sum_s32[1], vabsq_s32(vld1q_s32(quant_coeff + i + 4)));
        sum_s32[0] = vaddq_s32(sum_s32[0], vabsq_s32(vld1q_s32(quant_coeff + i + 8)));
        sum_s32[1] = vaddq_s32(sum_s32[1], vabsq_s32(vld1q_s32(quant_coeff + i + 12)));

        const int32_t sum = vaddvq_s32(vaddq_s32(sum_s32[0], sum_s32[1]));
        if (sum >= COEFF_CONTEXT_MASK) {
            return sum;
        }
    }
    return vaddvq_s32(vaddq_s32(sum_s32[0], sum_s32[1]));
}
