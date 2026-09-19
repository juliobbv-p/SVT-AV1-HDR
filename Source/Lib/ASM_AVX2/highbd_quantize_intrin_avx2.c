/*
 * Copyright (c) 2017, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include "definitions.h"
#include <immintrin.h>

#include "aom_dsp_rtcd.h"

// Note: TranHigh is the datatype used for intermediate transform stages.
typedef int64_t TranHigh;

#define AOM_QM_BITS 5

// quant_shift is always a power of two, so the second multiply-high in
// quantize() collapses to a right shift:
//   (q * quant_shift) >> (16 - log_scale) == q >> (16 - log_scale - ctz(qs))
// q is non-negative here (abs + round, then clamped), so a logical shift is
// correct.
//
// svt_aom_invert_quant writes shift = 1 << (16 - floor(log2(dequant))), and both
// qlookup tables start at dequant == 4, so shift is at most 1 << 14 and never
// truncates to 0 in int16. svt_ctz therefore never sees 0. Lanes 2..7 are
// replicated from lane 1 by the quantizer setup ("8: SIMD width").
static INLINE __m256i init_quant_shift_amt(const int16_t* quant_shift_ptr, const int add_shift) {
    int32_t sh[8];
    for (int i = 0; i < 8; i++) {
        sh[i] = 16 - add_shift - (int32_t)svt_ctz((unsigned)(uint16_t)quant_shift_ptr[i]);
    }
    return _mm256_loadu_si256((const __m256i*)sh);
}

static INLINE void update_qp(__m256i* qp) {
    int32_t i;
    for (i = 0; i < 5; ++i) {
        qp[i] = _mm256_permute2x128_si256(qp[i], qp[i], 0x11);
    }
}

static INLINE void init_qp_add_shift(const int16_t* zbin_ptr, const int16_t* round_ptr, const int16_t* quant_ptr,
                                     const int16_t* dequant_ptr, const int16_t* quant_shift_ptr, __m256i* qp,
                                     const int add_shift) {
    __m128i       zbin        = _mm_loadu_si128((const __m128i*)zbin_ptr);
    __m128i       round       = _mm_loadu_si128((const __m128i*)round_ptr);
    const __m128i quant       = _mm_loadu_si128((const __m128i*)quant_ptr);
    const __m128i dequant     = _mm_loadu_si128((const __m128i*)dequant_ptr);
    const __m128i quant_shift = _mm_loadu_si128((const __m128i*)quant_shift_ptr);
    if (add_shift) {
        const __m128i add = _mm_set1_epi16((int16_t)add_shift);
        zbin              = _mm_add_epi16(zbin, add);
        round             = _mm_add_epi16(round, add);
        zbin              = _mm_srli_epi16(zbin, add_shift);
        round             = _mm_srli_epi16(round, add_shift);
    }
    qp[0] = _mm256_cvtepi16_epi32(zbin);
    qp[1] = _mm256_cvtepi16_epi32(round);
    qp[2] = _mm256_cvtepi16_epi32(quant);
    qp[3] = _mm256_cvtepi16_epi32(dequant);
    qp[4] = _mm256_cvtepi16_epi32(quant_shift);
}

// Note:
// *x is vector multiplied by *y which is 8 int32_t parallel multiplication
// and right shift 16.  The output, 8 int32_t is save in *p.
static INLINE void mm256_mul_shift_epi32(const __m256i* x, const __m256i* y, __m256i* p, int shift) {
    __m256i prod_lo       = _mm256_mul_epi32(*x, *y);
    prod_lo               = _mm256_srli_epi64(prod_lo, shift);
    __m256i       prod_hi = _mm256_srli_epi64(*x, 32);
    const __m256i mult_hi = _mm256_srli_epi64(*y, 32);
    prod_hi               = _mm256_mul_epi32(prod_hi, mult_hi);

    prod_hi = _mm256_srli_epi64(prod_hi, shift);

    prod_hi = _mm256_slli_epi64(prod_hi, 32);
    *p      = _mm256_blend_epi32(prod_lo, prod_hi, 0xAA); //interleave prod_lo, prod_hi
}

static INLINE void clamp_epi32(__m256i* x, __m256i min, __m256i max) {
    *x = _mm256_min_epi32(*x, max);
    *x = _mm256_max_epi32(*x, min);
}

// int16 quantize: 16 coefficients per __m256i. Ported from the Arm int16 path
// (av1_quantize_neon.c). Valid for 8-bit content, where the forward transform
// produces coefficients that fit int16 (they are widened to int32 only on
// store), and where dqcoeff = qcoeff * dequant also fits.
//
// qs16 lanes hold zbin, round, quant, dequant and the power-of-two shift amount.
static INLINE void quantize16(const __m256i* qs16, __m256i c0, __m256i c1, const int16_t* iscan_ptr, TranLow* qcoeff,
                              TranLow* dqcoeff, __m256i* eob16) {
    const __m256i zero = _mm256_setzero_si256();
    // packs_epi32 emits per-128-bit-lane, so the permute restores coefficient order
    const __m256i cpk = _mm256_permute4x64_epi64(_mm256_packs_epi32(c0, c1), 0xD8);

    const __m256i abs  = _mm256_abs_epi16(cpk);
    const __m256i skip = _mm256_cmpgt_epi16(qs16[0], abs);
    if (EB_LIKELY(_mm256_movemask_epi8(skip) != (int)0xFFFFFFFF)) {
        __m256i q = _mm256_adds_epi16(abs, qs16[1]);
        q         = _mm256_add_epi16(q, _mm256_mulhi_epi16(q, qs16[2]));
        q         = _mm256_sra_epi16(q, _mm256_castsi256_si128(qs16[4]));
        q         = _mm256_andnot_si256(skip, q);

        // log_scale is 0 on this path (see the caller's gate), so there is no
        // post-shift; at log_scale > 0 the pre-shift product would overflow
        // int16 anyway, which is why the caller restricts it.
        const __m256i dq = _mm256_mullo_epi16(q, qs16[3]);

        q                 = _mm256_sign_epi16(q, cpk);
        const __m256i dqs = _mm256_sign_epi16(dq, cpk);

        const __m256i qlo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q));
        const __m256i qhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q, 1));
        const __m256i dlo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(dqs));
        const __m256i dhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(dqs, 1));
        _mm256_store_si256((__m256i*)(qcoeff + 0), qlo);
        _mm256_store_si256((__m256i*)(qcoeff + 8), qhi);
        _mm256_store_si256((__m256i*)(dqcoeff + 0), dlo);
        _mm256_store_si256((__m256i*)(dqcoeff + 8), dhi);

        const __m256i iscan   = _mm256_loadu_si256((const __m256i*)iscan_ptr);
        const __m256i nz      = _mm256_cmpeq_epi16(_mm256_cmpeq_epi16(dqs, zero), zero);
        __m256i       cur_eob = _mm256_sub_epi16(iscan, nz);
        cur_eob               = _mm256_and_si256(cur_eob, nz);
        *eob16                = _mm256_max_epi16(cur_eob, *eob16);
    } else {
        _mm256_store_si256((__m256i*)(qcoeff + 0), zero);
        _mm256_store_si256((__m256i*)(qcoeff + 8), zero);
        _mm256_store_si256((__m256i*)(dqcoeff + 0), zero);
        _mm256_store_si256((__m256i*)(dqcoeff + 8), zero);
    }
}

// int16 parameter vectors. qs16[4] holds the scalar shift amount in its low
// 128 bits for _mm256_sra_epi16.
// AC lanes only, and log_scale 0 only: the int16 path is not taken otherwise,
// so neither the DC lane nor the log_scale pre-shift of zbin/round applies here.
static INLINE void init_qs16(const int16_t* zbin_ptr, const int16_t* round_ptr, const int16_t* quant_ptr,
                             const int16_t* dequant_ptr, const int16_t* quant_shift_ptr, __m256i* qs16) {
    const int     i  = 1;
    const int16_t zb = zbin_ptr[i], rd = round_ptr[i];
    qs16[0] = _mm256_set1_epi16(zb);
    qs16[1] = _mm256_set1_epi16(rd);
    qs16[2] = _mm256_set1_epi16(quant_ptr[i]);
    qs16[3] = _mm256_set1_epi16(dequant_ptr[i]);
    qs16[4] = _mm256_castsi128_si256(_mm_cvtsi32_si128(16 - (int)svt_ctz((unsigned)(uint16_t)quant_shift_ptr[i])));
}

static INLINE void quantize(const __m256i* qp, __m256i c, const int16_t* iscan_ptr, TranLow* qcoeff, TranLow* dqcoeff,
                            __m256i* eob, __m256i min, __m256i max, int shift_dq) {
    const __m256i zero   = _mm256_setzero_si256();
    const __m256i abs    = _mm256_abs_epi32(c);
    const __m256i flag1  = _mm256_cmpgt_epi32(qp[0], abs);
    const int32_t nzflag = _mm256_movemask_epi8(flag1);

    if (EB_LIKELY(~nzflag)) {
        __m256i q = _mm256_add_epi32(abs, qp[1]);
        clamp_epi32(&q, min, max);
        // q is clamped to [0, 32767] and quant is int16, so q * quant is at most
        // 32767^2 ~= 1.07e9 and stays inside int32.
        const __m256i tmp = _mm256_srai_epi32(_mm256_mullo_epi32(q, qp[2]), 16);
        q                 = _mm256_add_epi32(tmp, q);

        q          = _mm256_srlv_epi32(q, qp[4]);
        __m256i dq = _mm256_mullo_epi32(q, qp[3]);
        dq         = _mm256_srli_epi32(dq, shift_dq);

        q  = _mm256_sign_epi32(q, c);
        dq = _mm256_sign_epi32(dq, c);
        q  = _mm256_andnot_si256(flag1, q);
        dq = _mm256_andnot_si256(flag1, dq);

        _mm256_store_si256((__m256i*)qcoeff, q);
        _mm256_store_si256((__m256i*)dqcoeff, dq);

        const __m128i isc   = _mm_loadu_si128((const __m128i*)iscan_ptr);
        const __m256i iscan = _mm256_cvtepi16_epi32(isc);

        const __m256i zc      = _mm256_cmpeq_epi32(dq, zero);
        const __m256i nz      = _mm256_cmpeq_epi32(zc, zero);
        __m256i       cur_eob = _mm256_sub_epi32(iscan, nz);
        cur_eob               = _mm256_and_si256(cur_eob, nz);
        *eob                  = _mm256_max_epi32(cur_eob, *eob);
    } else {
        _mm256_store_si256((__m256i*)qcoeff, zero);
        _mm256_store_si256((__m256i*)dqcoeff, zero);
    }
}

static INLINE void quantize_qm(const __m256i* qp, __m256i c, const int16_t* iscan_ptr, TranLow* qcoeff,
                               TranLow* dqcoeff, __m256i* eob, __m256i min, __m256i max, __m256i qm, __m256i iqm,
                               int shift_dq, const __m256i shift16) {
    const __m256i zero   = _mm256_setzero_si256();
    const __m256i abs    = _mm256_abs_epi32(c);
    const __m256i abs_qm = _mm256_mullo_epi32(abs, qm); // abs * wt
    const __m256i flag1  = _mm256_cmpgt_epi32(qp[0], abs_qm);
    const int32_t nzflag = _mm256_movemask_epi8(flag1);

    if (EB_LIKELY(~nzflag)) {
        __m256i q = _mm256_add_epi32(abs, qp[1]);
        clamp_epi32(&q, min, max);

        __m256i q_wt = _mm256_mullo_epi32(q, qm); // q_wt = q * wt
        __m256i tmp;
        mm256_mul_shift_epi32(&q_wt, &qp[2], &tmp, 16); // tmp = (tmp * qp[2]) >> 16
        q = _mm256_add_epi32(tmp, q_wt); // q = tmp + q_wt
        mm256_mul_shift_epi32(&q, &qp[4], &q, 16 - shift_dq + AOM_QM_BITS); // q = (q * qp[4]) >> (16 - shift_dq + 5)

        __m256i dq = _mm256_mullo_epi32(qp[3], iqm); // dq = qp[3] * iqm
        dq         = _mm256_add_epi32(dq, shift16); // dq = dq + 16
        dq         = _mm256_srli_epi32(dq, AOM_QM_BITS); // dq = dq >> 5
        dq         = _mm256_mullo_epi32(dq, q);
        dq         = _mm256_srli_epi32(dq, shift_dq);

        q  = _mm256_sign_epi32(q, c);
        dq = _mm256_sign_epi32(dq, c);
        q  = _mm256_andnot_si256(flag1, q);
        dq = _mm256_andnot_si256(flag1, dq);

        _mm256_store_si256((__m256i*)qcoeff, q);
        _mm256_store_si256((__m256i*)dqcoeff, dq);

        const __m128i isc   = _mm_loadu_si128((const __m128i*)iscan_ptr);
        const __m256i iscan = _mm256_cvtepi16_epi32(isc);

        const __m256i zc      = _mm256_cmpeq_epi32(dq, zero);
        const __m256i nz      = _mm256_cmpeq_epi32(zc, zero);
        __m256i       cur_eob = _mm256_sub_epi32(iscan, nz);
        cur_eob               = _mm256_and_si256(cur_eob, nz);
        *eob                  = _mm256_max_epi32(cur_eob, *eob);
    } else {
        _mm256_store_si256((__m256i*)qcoeff, zero);
        _mm256_store_si256((__m256i*)dqcoeff, zero);
    }
}

static INLINE __m256i load_bytes_to_m256_avx2(const QmVal* p) {
    __m128i small_load = _mm_loadl_epi64((const __m128i*)p);
    return _mm256_cvtepu8_epi32(small_load);
}

void svt_av1_quantize_b_qm_avx2(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan, const QmVal* qm_ptr,
                                const QmVal* iqm_ptr, const int32_t log_scale) {
    (void)scan;
    const uint32_t step = 8;

    __m256i qp[5], coeff;
    init_qp_add_shift(zbin_ptr, round_ptr, quant_ptr, dequant_ptr, quant_shift_ptr, qp, log_scale);
    qp[0] = _mm256_slli_epi32(qp[0], AOM_QM_BITS);
    coeff = _mm256_load_si256((const __m256i*)coeff_ptr);

    __m256i qm, iqm;
    qm  = load_bytes_to_m256_avx2(qm_ptr);
    iqm = load_bytes_to_m256_avx2(iqm_ptr);

    __m256i       eob     = _mm256_setzero_si256();
    __m256i       min     = _mm256_set1_epi32(INT16_MIN);
    __m256i       max     = _mm256_set1_epi32(INT16_MAX);
    const __m256i shift16 = _mm256_set1_epi32(16);
    quantize_qm(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, min, max, qm, iqm, log_scale, shift16);
    update_qp(qp);

    while (n_coeffs > step) {
        coeff_ptr += step;
        qcoeff_ptr += step;
        dqcoeff_ptr += step;
        iscan += step;
        qm_ptr += step;
        iqm_ptr += step;
        n_coeffs -= step;

        coeff = _mm256_load_si256((const __m256i*)coeff_ptr);
        qm    = load_bytes_to_m256_avx2(qm_ptr);
        iqm   = load_bytes_to_m256_avx2(iqm_ptr);
        quantize_qm(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, min, max, qm, iqm, log_scale, shift16);
    }
    {
        __m256i eob_s;
        eob_s                   = _mm256_shuffle_epi32(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 1);
        eob                     = _mm256_max_epi16(eob, eob_s);
        const __m128i final_eob = _mm_max_epi16(_mm256_castsi256_si128(eob), _mm256_extractf128_si256(eob, 1));
        *eob_ptr                = _mm_extract_epi16(final_eob, 0);
    }
}

void svt_aom_quantize_b_avx2(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                             const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                             TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr, uint16_t* eob_ptr,
                             const int16_t* scan, const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                             const int32_t log_scale) {
    (void)qm_ptr;
    (void)iqm_ptr;
    (void)scan;
    const uint32_t step = 8;

    // The int16 loop below advances 8, consumes 16, then advances 8 again, so
    // the coefficient count must be a multiple of 16. Every AV1 transform area
    // is. Checked here, before the two leading groups decrement n_coeffs.
    assert(n_coeffs % 16 == 0);

    __m256i qp[5], coeff;
    init_qp_add_shift(zbin_ptr, round_ptr, quant_ptr, dequant_ptr, quant_shift_ptr, qp, log_scale);
    // svt_aom_invert_quant is the only producer of quant_shift and always writes
    // 1 << (16 - l), so the shift path in quantize() is always valid. Asserted
    // rather than branched on, matching av1_quantize_neon.c.
    assert(quant_shift_ptr[0] == (1 << svt_ctz((unsigned)(uint16_t)quant_shift_ptr[0])));
    assert(quant_shift_ptr[1] == (1 << svt_ctz((unsigned)(uint16_t)quant_shift_ptr[1])));
    qp[4] = init_quant_shift_amt(quant_shift_ptr, log_scale);
    coeff = _mm256_load_si256((const __m256i*)coeff_ptr);

    __m256i eob = _mm256_setzero_si256();
    __m256i min = _mm256_set1_epi32(INT16_MIN);
    __m256i max = _mm256_set1_epi32(INT16_MAX);
    quantize(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, min, max, log_scale);
    update_qp(qp);

    // Second group of 8 stays on the int32 path, then the bulk switches to the
    // int16 kernel (16 coefficients per register). Only the DC coefficient needs
    // its own shift amount, and AVX2 has no per-lane 16-bit variable shift, so
    // keeping the first 16 on the int32 path avoids that entirely.
    if (n_coeffs > step) {
        coeff_ptr += step;
        qcoeff_ptr += step;
        dqcoeff_ptr += step;
        iscan += step;
        n_coeffs -= step;
        coeff = _mm256_load_si256((const __m256i*)coeff_ptr);
        quantize(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, min, max, log_scale);
    }

    // log_scale 0 only: dqcoeff is |q| * dequant BEFORE the log_scale shift, so
    // at log_scale > 0 that intermediate is up to 4x the final value and
    // overflows int16. The int32 path keeps it in 32 bits. NEON restricts its
    // int16 kernel the same way (quantize_b_logscale0_8).
    if (log_scale == 0 && n_coeffs > (intptr_t)step) {
        __m256i qs16[5];
        init_qs16(zbin_ptr, round_ptr, quant_ptr, dequant_ptr, quant_shift_ptr, qs16);
        __m256i eob16 = _mm256_setzero_si256();
        while (n_coeffs > (intptr_t)step) {
            coeff_ptr += step;
            qcoeff_ptr += step;
            dqcoeff_ptr += step;
            iscan += step;
            n_coeffs -= step;
            const __m256i c0 = _mm256_load_si256((const __m256i*)coeff_ptr);
            const __m256i c1 = _mm256_load_si256((const __m256i*)(coeff_ptr + 8));
            quantize16(qs16, c0, c1, iscan, qcoeff_ptr, dqcoeff_ptr, &eob16);
            coeff_ptr += step;
            qcoeff_ptr += step;
            dqcoeff_ptr += step;
            iscan += step;
            n_coeffs -= step;
        }
        const __m256i e32lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(eob16));
        const __m256i e32hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(eob16, 1));
        eob                 = _mm256_max_epi32(eob, _mm256_max_epi32(e32lo, e32hi));
    } else {
        while (n_coeffs > step) {
            coeff_ptr += step;
            qcoeff_ptr += step;
            dqcoeff_ptr += step;
            iscan += step;
            n_coeffs -= step;
            coeff = _mm256_load_si256((const __m256i*)coeff_ptr);
            quantize(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, min, max, log_scale);
        }
    }
    {
        __m256i eob_s;
        eob_s                   = _mm256_shuffle_epi32(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 1);
        eob                     = _mm256_max_epi16(eob, eob_s);
        const __m128i final_eob = _mm_max_epi16(_mm256_castsi256_si128(eob), _mm256_extractf128_si256(eob, 1));
        *eob_ptr                = _mm_extract_epi16(final_eob, 0);
    }
}

static INLINE void quantize_highbd_qm(const __m256i* qp, __m256i c, const int16_t* iscan_ptr, TranLow* qcoeff,
                                      TranLow* dqcoeff, __m256i* eob, int shift_dq, const __m256i wt, const __m256i iwt,
                                      const __m256i shift16) {
    const __m256i zero   = _mm256_setzero_si256();
    const __m256i abs    = _mm256_abs_epi32(c);
    const __m256i abs_wt = _mm256_mullo_epi32(abs, wt);
    const __m256i flag1  = _mm256_cmpgt_epi32(qp[0], abs_wt);
    const int32_t nzflag = _mm256_movemask_epi8(flag1);

    if (EB_LIKELY(~nzflag)) {
        __m256i q = _mm256_add_epi32(abs, qp[1]);
        q         = _mm256_mullo_epi32(q, wt);
        __m256i tmp;
        mm256_mul_shift_epi32(&q, &qp[2], &tmp, 16);
        q = _mm256_add_epi32(tmp, q);
        mm256_mul_shift_epi32(&q,
                              &qp[4],
                              &q,
                              16 - shift_dq + AOM_QM_BITS); // q = (q * quant_shift) >> (16 - shift_dq + 5)

        __m256i dq = _mm256_mullo_epi32(qp[3], iwt);
        dq         = _mm256_add_epi32(dq, shift16); // dq = dq + 16
        dq         = _mm256_srli_epi32(dq, AOM_QM_BITS);
        dq         = _mm256_mullo_epi32(dq, q);
        dq         = _mm256_srli_epi32(dq, shift_dq);

        q  = _mm256_sign_epi32(q, c);
        dq = _mm256_sign_epi32(dq, c);
        q  = _mm256_andnot_si256(flag1, q);
        dq = _mm256_andnot_si256(flag1, dq);

        _mm256_store_si256((__m256i*)qcoeff, q);
        _mm256_store_si256((__m256i*)dqcoeff, dq);

        const __m128i isc   = _mm_loadu_si128((const __m128i*)iscan_ptr);
        const __m256i iscan = _mm256_cvtepi16_epi32(isc);

        const __m256i zc      = _mm256_cmpeq_epi32(dq, zero);
        const __m256i nz      = _mm256_cmpeq_epi32(zc, zero);
        __m256i       cur_eob = _mm256_sub_epi32(iscan, nz);
        cur_eob               = _mm256_and_si256(cur_eob, nz);
        *eob                  = _mm256_max_epi32(cur_eob, *eob);
    } else {
        _mm256_store_si256((__m256i*)qcoeff, zero);
        _mm256_store_si256((__m256i*)dqcoeff, zero);
    }
}

void svt_av1_highbd_quantize_b_qm_avx2(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                       const int16_t* round_ptr, const int16_t* quant_ptr,
                                       const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                       const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                       const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                       const int32_t log_scale) {
    (void)scan;
    const uint32_t step = 8;

    __m256i qp[5], coeff, qm, iqm;
    init_qp_add_shift(zbin_ptr, round_ptr, quant_ptr, dequant_ptr, quant_shift_ptr, qp, log_scale);
    qp[0] = _mm256_slli_epi32(qp[0], AOM_QM_BITS); // zbin = zbin << 5
    coeff = _mm256_load_si256((const __m256i*)coeff_ptr);
    qm    = load_bytes_to_m256_avx2(qm_ptr);
    iqm   = load_bytes_to_m256_avx2(iqm_ptr);

    __m256i       eob     = _mm256_setzero_si256();
    const __m256i shift16 = _mm256_set1_epi32(16);
    quantize_highbd_qm(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, log_scale, qm, iqm, shift16);
    update_qp(qp);

    while (n_coeffs > step) {
        coeff_ptr += step;
        qcoeff_ptr += step;
        dqcoeff_ptr += step;
        qm_ptr += step;
        iqm_ptr += step;
        iscan += step;
        n_coeffs -= step;
        coeff = _mm256_load_si256((const __m256i*)coeff_ptr);
        qm    = load_bytes_to_m256_avx2(qm_ptr);
        iqm   = load_bytes_to_m256_avx2(iqm_ptr);
        quantize_highbd_qm(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, log_scale, qm, iqm, shift16);
    }
    {
        __m256i eob_s;
        eob_s                   = _mm256_shuffle_epi32(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 1);
        eob                     = _mm256_max_epi16(eob, eob_s);
        const __m128i final_eob = _mm_max_epi16(_mm256_castsi256_si128(eob), _mm256_extractf128_si256(eob, 1));
        *eob_ptr                = _mm_extract_epi16(final_eob, 0);
    }
}

static INLINE void quantize_highbd(const __m256i* qp, __m256i c, const int16_t* iscan_ptr, TranLow* qcoeff,
                                   TranLow* dqcoeff, __m256i* eob, int shift_dq) {
    const __m256i zero   = _mm256_setzero_si256();
    const __m256i abs    = _mm256_abs_epi32(c);
    const __m256i flag1  = _mm256_cmpgt_epi32(qp[0], abs);
    const int32_t nzflag = _mm256_movemask_epi8(flag1);

    if (EB_LIKELY(~nzflag)) {
        __m256i q = _mm256_add_epi32(abs, qp[1]);
        __m256i tmp;
        mm256_mul_shift_epi32(&q, &qp[2], &tmp, 16);
        q = _mm256_add_epi32(tmp, q);

        mm256_mul_shift_epi32(&q, &qp[4], &q, 16 - shift_dq);
        __m256i dq = _mm256_mullo_epi32(q, qp[3]);
        dq         = _mm256_srli_epi32(dq, shift_dq);

        q  = _mm256_sign_epi32(q, c);
        dq = _mm256_sign_epi32(dq, c);
        q  = _mm256_andnot_si256(flag1, q);
        dq = _mm256_andnot_si256(flag1, dq);

        _mm256_store_si256((__m256i*)qcoeff, q);
        _mm256_store_si256((__m256i*)dqcoeff, dq);

        const __m128i isc   = _mm_loadu_si128((const __m128i*)iscan_ptr);
        const __m256i iscan = _mm256_cvtepi16_epi32(isc);

        const __m256i zc      = _mm256_cmpeq_epi32(dq, zero);
        const __m256i nz      = _mm256_cmpeq_epi32(zc, zero);
        __m256i       cur_eob = _mm256_sub_epi32(iscan, nz);
        cur_eob               = _mm256_and_si256(cur_eob, nz);
        *eob                  = _mm256_max_epi32(cur_eob, *eob);
    } else {
        _mm256_store_si256((__m256i*)qcoeff, zero);
        _mm256_store_si256((__m256i*)dqcoeff, zero);
    }
}

void svt_aom_highbd_quantize_b_avx2(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                    const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                    TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                    uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan, const QmVal* qm_ptr,
                                    const QmVal* iqm_ptr, const int32_t log_scale) {
    (void)qm_ptr;
    (void)iqm_ptr;
    (void)scan;
    const uint32_t step = 8;

    __m256i qp[5], coeff;
    init_qp_add_shift(zbin_ptr, round_ptr, quant_ptr, dequant_ptr, quant_shift_ptr, qp, log_scale);
    coeff = _mm256_load_si256((const __m256i*)coeff_ptr);

    __m256i eob = _mm256_setzero_si256();
    quantize_highbd(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, log_scale);
    update_qp(qp);

    while (n_coeffs > step) {
        coeff_ptr += step;
        qcoeff_ptr += step;
        dqcoeff_ptr += step;
        iscan += step;
        n_coeffs -= step;
        coeff = _mm256_load_si256((const __m256i*)coeff_ptr);
        quantize_highbd(qp, coeff, iscan, qcoeff_ptr, dqcoeff_ptr, &eob, log_scale);
    }
    {
        __m256i eob_s;
        eob_s                   = _mm256_shuffle_epi32(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 1);
        eob                     = _mm256_max_epi16(eob, eob_s);
        const __m128i final_eob = _mm_max_epi16(_mm256_castsi256_si128(eob), _mm256_extractf128_si256(eob, 1));
        *eob_ptr                = _mm_extract_epi16(final_eob, 0);
    }
}

static INLINE void init_one_qp_fp(const __m128i* p, __m256i* qp) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i dc   = _mm_unpacklo_epi16(*p, zero);
    const __m128i ac   = _mm_unpackhi_epi16(*p, zero);
    *qp                = _mm256_insertf128_si256(_mm256_castsi128_si256(dc), ac, 1);
}

static INLINE void init_qp_fp(const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* dequant_ptr,
                              int log_scale, __m256i* qp) {
    __m128i round = _mm_loadu_si128((const __m128i*)round_ptr);
    if (log_scale) {
        const __m128i round_scale = _mm_set1_epi16(1 << (15 - log_scale));
        round                     = _mm_mulhrs_epi16(round, round_scale);
    }
    const __m128i quant   = _mm_loadu_si128((const __m128i*)quant_ptr);
    const __m128i dequant = _mm_loadu_si128((const __m128i*)dequant_ptr);

    init_one_qp_fp(&round, &qp[0]);
    init_one_qp_fp(&quant, &qp[1]);
    init_one_qp_fp(&dequant, &qp[2]);
}

static INLINE void quantize_highbd_fp(const __m256i* qp, __m256i* c, const int16_t* iscan_ptr, int log_scale,
                                      TranLow* qcoeff, TranLow* dqcoeff, __m256i* eob) {
    const __m256i abs_coeff = _mm256_abs_epi32(*c);
    __m256i       q         = _mm256_add_epi32(abs_coeff, qp[0]);

    __m256i       q_lo  = _mm256_mul_epi32(q, qp[1]);
    __m256i       q_hi  = _mm256_srli_epi64(q, 32);
    const __m256i qp_hi = _mm256_srli_epi64(qp[1], 32);
    q_hi                = _mm256_mul_epi32(q_hi, qp_hi);
    q_lo                = _mm256_srli_epi64(q_lo, 16 - log_scale);
    q_hi                = _mm256_srli_epi64(q_hi, 16 - log_scale);
    q_hi                = _mm256_slli_epi64(q_hi, 32);
    q                   = _mm256_or_si256(q_lo, q_hi);
    const __m256i abs_s = _mm256_slli_epi32(abs_coeff, 1 + log_scale);
    const __m256i mask  = _mm256_cmpgt_epi32(qp[2], abs_s);
    q                   = _mm256_andnot_si256(mask, q);

    __m256i dq = _mm256_mullo_epi32(q, qp[2]);
    dq         = _mm256_srai_epi32(dq, log_scale);
    q          = _mm256_sign_epi32(q, *c);
    dq         = _mm256_sign_epi32(dq, *c);

    _mm256_storeu_si256((__m256i*)qcoeff, q);
    _mm256_storeu_si256((__m256i*)dqcoeff, dq);

    const __m128i isc   = _mm_loadu_si128((const __m128i*)iscan_ptr);
    const __m128i zr    = _mm_setzero_si128();
    const __m128i lo    = _mm_unpacklo_epi16(isc, zr);
    const __m128i hi    = _mm_unpackhi_epi16(isc, zr);
    const __m256i iscan = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);

    const __m256i zero    = _mm256_setzero_si256();
    const __m256i zc      = _mm256_cmpeq_epi32(dq, zero);
    const __m256i nz      = _mm256_cmpeq_epi32(zc, zero);
    __m256i       cur_eob = _mm256_sub_epi32(iscan, nz);
    cur_eob               = _mm256_and_si256(cur_eob, nz);
    *eob                  = _mm256_max_epi32(cur_eob, *eob);
}

static INLINE void update_qp_fp(__m256i* qp) {
    qp[0] = _mm256_permute2x128_si256(qp[0], qp[0], 0x11);
    qp[1] = _mm256_permute2x128_si256(qp[1], qp[1], 0x11);
    qp[2] = _mm256_permute2x128_si256(qp[2], qp[2], 0x11);
}

void svt_av1_highbd_quantize_fp_avx2(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                     const int16_t* round_ptr, const int16_t* quant_ptr, const int16_t* quant_shift_ptr,
                                     TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr, const int16_t* dequant_ptr,
                                     uint16_t* eob_ptr, const int16_t* scan, const int16_t* iscan, int16_t log_scale) {
    (void)scan;
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    const unsigned int step = 8;
    __m256i            qp[3], coeff;

    init_qp_fp(round_ptr, quant_ptr, dequant_ptr, log_scale, qp);
    coeff = _mm256_loadu_si256((const __m256i*)coeff_ptr);

    __m256i eob = _mm256_setzero_si256();
    quantize_highbd_fp(qp, &coeff, iscan, log_scale, qcoeff_ptr, dqcoeff_ptr, &eob);

    coeff_ptr += step;
    qcoeff_ptr += step;
    dqcoeff_ptr += step;
    iscan += step;
    n_coeffs -= step;

    update_qp_fp(qp);
    while (n_coeffs > 0) {
        coeff = _mm256_loadu_si256((const __m256i*)coeff_ptr);
        quantize_highbd_fp(qp, &coeff, iscan, log_scale, qcoeff_ptr, dqcoeff_ptr, &eob);

        coeff_ptr += step;
        qcoeff_ptr += step;
        dqcoeff_ptr += step;
        iscan += step;
        n_coeffs -= step;
    }

    {
        __m256i eob_s;
        eob_s                   = _mm256_shuffle_epi32(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 1);
        eob                     = _mm256_max_epi16(eob, eob_s);
        const __m128i final_eob = _mm_max_epi16(_mm256_castsi256_si128(eob), _mm256_extractf128_si256(eob, 1));
        *eob_ptr                = _mm_extract_epi16(final_eob, 0);
    }
}

// 64 bit multiply. return the low 64 bits of the intermediate integers
static inline __m256i mm256_mullo_epi64(const __m256i a, const __m256i b) {
    // if a 64bit integer 'a' can be represented by its low 32bit part a0 and high 32bit part a1 as: a1<<32+a0,
    // 64bit integer multiply a*b can expand to: (a1*b1)<<64 + (a1*b0 + a0*b1)<<32 + a0*b0.
    // since only the low 64bit part of the result 128bit integer is needed, the above expression can be simplified as: (a1*b0 + a0*b1)<<32 + a0*b0
    const __m256i bswap   = _mm256_shuffle_epi32(b, 0xB1); // b6 b7 b4 b5 b2 b3 b0 b1
    __m256i       prod_hi = _mm256_mullo_epi32(a,
                                         bswap); // a7*b6 a6*b7 a5*b4 a4*b5 a3*b2 a2*b3 a1*b0 a0*b1
    const __m256i zero    = _mm256_setzero_si256();
    prod_hi               = _mm256_hadd_epi32(prod_hi,
                                zero); // 0 0 a7*b6+a6*b7 a5*b4+a4*b5 0 0 a3*b2+a2*b3 a1*b0+a0*b1
    prod_hi               = _mm256_shuffle_epi32(prod_hi,
                                   0x73); // a7*b6+a6*b7 0 a5*b4+a4*b5 0 a3*b2+a2*b3 0 a1*b0+a0*b1 0
    const __m256i prod_lo = _mm256_mul_epu32(a, b); // 0 a6*b6 0 a4*b4 0 a2*b2 0 a0*b0
    const __m256i prod    = _mm256_add_epi64(prod_lo, prod_hi);
    return prod;
}

static INLINE void quantize_highbd_fp_qm(const __m256i* qp, __m256i* c, const int16_t* iscan_ptr, int log_scale,
                                         TranLow* qcoeff, TranLow* dqcoeff, __m256i* eob, const __m256i qm,
                                         const __m256i iqm) {
    const __m256i abs_coeff = _mm256_abs_epi32(*c);
    __m256i       q         = _mm256_add_epi32(abs_coeff, qp[0]);

    const __m256i wt_hi = _mm256_srli_epi64(qm, 32);
    const __m256i wt_lo = _mm256_srli_epi64(_mm256_slli_epi64(qm, 32), 32);

    __m256i q_lo = _mm256_mul_epi32(q, qp[1]);
    q_lo         = mm256_mullo_epi64(q_lo, wt_lo);

    __m256i       q_hi  = _mm256_srli_epi64(q, 32);
    const __m256i qp_hi = _mm256_srli_epi64(qp[1], 32);
    q_hi                = _mm256_mul_epi32(q_hi, qp_hi);
    q_hi                = mm256_mullo_epi64(q_hi, wt_hi);

    q_lo = _mm256_srli_epi64(q_lo, 16 - log_scale + AOM_QM_BITS);
    q_hi = _mm256_srli_epi64(q_hi, 16 - log_scale + AOM_QM_BITS);
    q_hi = _mm256_slli_epi64(q_hi, 32);
    q    = _mm256_or_si256(q_lo, q_hi);

    const __m256i abs_s = _mm256_mullo_epi32(abs_coeff, qm);
    __m256i       mask  = _mm256_slli_epi32(qp[2], AOM_QM_BITS - (1 + log_scale));
    mask                = _mm256_cmpgt_epi32(mask, abs_s);
    q                   = _mm256_andnot_si256(mask, q);

    __m256i       dq  = _mm256_mullo_epi32(qp[2], iqm);
    const __m256i a16 = _mm256_set1_epi32(16);
    dq                = _mm256_add_epi32(dq, a16);
    dq                = _mm256_srli_epi32(dq, AOM_QM_BITS);
    dq                = _mm256_mullo_epi32(q, dq);
    dq                = _mm256_srai_epi32(dq, log_scale);
    q                 = _mm256_sign_epi32(q, *c);
    dq                = _mm256_sign_epi32(dq, *c);

    _mm256_storeu_si256((__m256i*)qcoeff, q);
    _mm256_storeu_si256((__m256i*)dqcoeff, dq);

    const __m128i isc   = _mm_loadu_si128((const __m128i*)iscan_ptr);
    const __m128i zr    = _mm_setzero_si128();
    const __m128i lo    = _mm_unpacklo_epi16(isc, zr);
    const __m128i hi    = _mm_unpackhi_epi16(isc, zr);
    const __m256i iscan = _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);

    const __m256i zero    = _mm256_setzero_si256();
    const __m256i zc      = _mm256_cmpeq_epi32(dq, zero);
    const __m256i nz      = _mm256_cmpeq_epi32(zc, zero);
    __m256i       cur_eob = _mm256_sub_epi32(iscan, nz);
    cur_eob               = _mm256_and_si256(cur_eob, nz);
    *eob                  = _mm256_max_epi32(cur_eob, *eob);
}

void svt_av1_highbd_quantize_fp_qm_avx2(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                        const int16_t* round_ptr, const int16_t* quant_ptr,
                                        const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                        const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                        const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                        int16_t log_scale) {
    (void)scan;
    (void)zbin_ptr;
    (void)quant_shift_ptr;
    const unsigned int step = 8;
    __m256i            qp[3];
    init_qp_fp(round_ptr, quant_ptr, dequant_ptr, log_scale, qp);
    __m256i eob = _mm256_setzero_si256();

    __m256i coeff = _mm256_loadu_si256((const __m256i*)coeff_ptr);
    __m256i qm    = load_bytes_to_m256_avx2(qm_ptr);
    __m256i iqm   = load_bytes_to_m256_avx2(iqm_ptr);
    quantize_highbd_fp_qm(qp, &coeff, iscan, log_scale, qcoeff_ptr, dqcoeff_ptr, &eob, qm, iqm);

    update_qp_fp(qp);
    while (n_coeffs > step) {
        coeff_ptr += step;
        qcoeff_ptr += step;
        dqcoeff_ptr += step;
        iscan += step;
        qm_ptr += step;
        iqm_ptr += step;
        n_coeffs -= step;

        coeff = _mm256_loadu_si256((const __m256i*)coeff_ptr);
        qm    = load_bytes_to_m256_avx2(qm_ptr);
        iqm   = load_bytes_to_m256_avx2(iqm_ptr);
        quantize_highbd_fp_qm(qp, &coeff, iscan, log_scale, qcoeff_ptr, dqcoeff_ptr, &eob, qm, iqm);
    }

    {
        __m256i eob_s;
        eob_s                   = _mm256_shuffle_epi32(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 0xe);
        eob                     = _mm256_max_epi16(eob, eob_s);
        eob_s                   = _mm256_shufflelo_epi16(eob, 1);
        eob                     = _mm256_max_epi16(eob, eob_s);
        const __m128i final_eob = _mm_max_epi16(_mm256_castsi256_si128(eob), _mm256_extractf128_si256(eob, 1));
        *eob_ptr                = _mm_extract_epi16(final_eob, 0);
    }
}
