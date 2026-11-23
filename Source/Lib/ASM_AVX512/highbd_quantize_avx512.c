/*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "aom_dsp_rtcd.h"
#include "qm_psnr.h"
#if EN_AVX512_SUPPORT
#include <immintrin.h>

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
static INLINE __m512i dc_ac(int dc, int ac) {
    return _mm512_mask_set1_epi32(_mm512_set1_epi32(ac), 1, dc);
}

// Two widening multiplies cover all sixteen raster positions. Both scale
// factors are powers of two, so the regular quantizer needs only one shift.
static INLINE __m512i quant_product(__m512i a, __m512i b, __m512i shift) {
    const __m512i lo = _mm512_srlv_epi64(_mm512_mul_epu32(a, b), _mm512_maskz_mov_epi32(0x5555, shift));
    const __m512i hi = _mm512_srlv_epi64(_mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32)),
                                         _mm512_srli_epi64(shift, 32));
    return _mm512_mask_blend_epi32(0xaaaa, lo, _mm512_slli_epi64(hi, 32));
}

static AOM_FORCE_INLINE void quantize(const TranLow* coeff, intptr_t n, const int16_t* zbin, const int16_t* round,
                                      const int16_t* quant, const int16_t* qs, TranLow* qcoeff, TranLow* dqcoeff,
                                      const int16_t* dq, uint16_t* eob, const int16_t* iscan, const QmVal* qm,
                                      const QmVal* iqm, int scale, bool fast, bool weighted) {
    const int extra      = weighted ? AOM_QM_BITS : 0;
    const int base_shift = (fast ? 16 : 32) - scale + extra;
    int       threshold[2], factor[2], shift[2], rounding[2];
    for (int j = 0; j < 2; ++j) {
        const int half = 1 << (1 + scale);
        threshold[j]   = fast ? ((dq[j] << extra) + half - 1) / half : ROUND_POWER_OF_TWO(zbin[j], scale) << extra;
        rounding[j]    = ROUND_POWER_OF_TWO(round[j], scale);
        factor[j]      = fast ? quant[j] : 65536 + quant[j];
        shift[j]       = fast ? base_shift : base_shift - svt_ctz((uint16_t)qs[j]);
    }
    __m512i thresh = dc_ac(threshold[0], threshold[1]), mul = dc_ac(factor[0], factor[1]);
    __m512i count = dc_ac(shift[0], shift[1]), rnd = dc_ac(rounding[0], rounding[1]), deq = dc_ac(dq[0], dq[1]);
    __m256i last = _mm256_set1_epi16(-1);
    for (intptr_t i = 0; i < n; i += 16) {
        const __mmask16 mask = n - i >= 16 ? 0xffff : (1u << (n - i)) - 1;
        const __m512i   c = _mm512_maskz_loadu_epi32(mask, coeff + i), mag = _mm512_abs_epi32(c);
        const __m512i   wt = weighted && qm ? _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, qm + i))
                                            : _mm512_set1_epi32(1 << extra);
        // Capping at the threshold makes the comparison exact even for a
        // weighted magnitude that would overflow a dword.
        const __m512i   candidate = weighted ? _mm512_mullo_epi32(_mm512_min_epu32(mag, thresh), wt) : mag;
        const __mmask16 active    = _mm512_mask_cmpge_epu32_mask(mask, candidate, thresh);
        __m512i         q = _mm512_setzero_si512(), d = _mm512_setzero_si512();
        if (active) {
            const __m512i a = _mm512_add_epi32(mag, rnd), b = weighted ? _mm512_mullo_epi32(mul, wt) : mul;
            if (fast) {
                const __m512i even = _mm512_mul_epu32(a, b);
                const __m512i odd  = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), _mm512_srli_epi64(b, 32));
                // Shift each widened product directly into its final dword.
                q = _mm512_mask_blend_epi32(0xaaaa,
                                            _mm512_srl_epi64(even, _mm_cvtsi32_si128(base_shift)),
                                            _mm512_sll_epi64(odd, _mm_cvtsi32_si128(32 - base_shift)));
            } else {
                q = quant_product(a, b, count);
            }
            q            = _mm512_maskz_mov_epi32(active, q);
            __m512i step = deq;
            if (weighted && iqm) {
                const __m512i iw = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(mask, iqm + i));
                step             = _mm512_srli_epi32(
                    _mm512_add_epi32(_mm512_mullo_epi32(deq, iw), _mm512_set1_epi32(1 << (AOM_QM_BITS - 1))),
                    AOM_QM_BITS);
            }
            d                   = _mm512_sra_epi32(_mm512_mullo_epi32(q, step), _mm_cvtsi32_si128(scale));
            const __mmask16 neg = _mm512_movepi32_mask(c);
            d                   = _mm512_mask_sub_epi32(d, neg, _mm512_setzero_si512(), d);
            q                   = _mm512_mask_sub_epi32(q, neg, _mm512_setzero_si512(), q);
            const __m256i pos   = _mm256_maskz_loadu_epi16(mask, iscan + i);
            last                = _mm256_mask_max_epi16(last, _mm512_test_epi32_mask(q, q), last, pos);
        }
        _mm512_mask_storeu_epi32(qcoeff + i, mask, q);
        _mm512_mask_storeu_epi32(dqcoeff + i, mask, d);
        thresh = _mm512_set1_epi32(threshold[1]);
        mul    = _mm512_set1_epi32(factor[1]);
        count  = _mm512_set1_epi32(shift[1]);
        rnd    = _mm512_set1_epi32(rounding[1]);
        deq    = _mm512_set1_epi32(dq[1]);
    }
    const __m128i maximum = _mm_max_epi16(_mm256_castsi256_si128(last), _mm256_extracti128_si256(last, 1));
    // Min-position on complemented nonnegative scan positions reduces eight
    // word lanes in one instruction. Position -1 denotes an empty block.
    const __m128i inverted = _mm_sub_epi16(_mm_set1_epi16(INT16_MAX), maximum);
    *eob                   = (uint16_t)(32768 - _mm_extract_epi16(_mm_minpos_epu16(inverted), 0));
}

void svt_av1_highbd_quantize_fp_avx512(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                       const int16_t* round_ptr, const int16_t* quant_ptr,
                                       const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                       const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                       const int16_t* iscan, int16_t log_scale) {
    (void)scan;
    quantize(coeff_ptr,
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
             NULL,
             NULL,
             log_scale,
             true,
             false);
}

void svt_av1_highbd_quantize_fp_qm_avx512(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                          const int16_t* round_ptr, const int16_t* quant_ptr,
                                          const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                          const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                          const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                          int16_t log_scale) {
    (void)scan;
    quantize(coeff_ptr,
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
             true,
             true);
}

void svt_aom_highbd_quantize_b_avx512(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                      const int16_t* round_ptr, const int16_t* quant_ptr,
                                      const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                      const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                      const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                      const int32_t log_scale) {
    // Encoder shift multipliers are positive powers of two. Preserve the two
    // rounding steps for callers supplying a general fixed-point multiplier.
    if (quant_shift_ptr[0] <= 0 || quant_shift_ptr[1] <= 0 || (quant_shift_ptr[0] & (quant_shift_ptr[0] - 1)) ||
        (quant_shift_ptr[1] & (quant_shift_ptr[1] - 1))) {
        svt_aom_highbd_quantize_b_c(coeff_ptr,
                                    n_coeffs,
                                    zbin_ptr,
                                    round_ptr,
                                    quant_ptr,
                                    quant_shift_ptr,
                                    qcoeff_ptr,
                                    dqcoeff_ptr,
                                    dequant_ptr,
                                    eob_ptr,
                                    scan,
                                    iscan,
                                    qm_ptr,
                                    iqm_ptr,
                                    log_scale);
        return;
    }
    if (qm_ptr || iqm_ptr) {
        svt_av1_highbd_quantize_b_qm_avx512(coeff_ptr,
                                            n_coeffs,
                                            zbin_ptr,
                                            round_ptr,
                                            quant_ptr,
                                            quant_shift_ptr,
                                            qcoeff_ptr,
                                            dqcoeff_ptr,
                                            dequant_ptr,
                                            eob_ptr,
                                            scan,
                                            iscan,
                                            qm_ptr,
                                            iqm_ptr,
                                            log_scale);
        return;
    }
    (void)scan;
    quantize(coeff_ptr,
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
             NULL,
             NULL,
             log_scale,
             false,
             false);
}

void svt_av1_highbd_quantize_b_qm_avx512(const TranLow* coeff_ptr, intptr_t n_coeffs, const int16_t* zbin_ptr,
                                         const int16_t* round_ptr, const int16_t* quant_ptr,
                                         const int16_t* quant_shift_ptr, TranLow* qcoeff_ptr, TranLow* dqcoeff_ptr,
                                         const int16_t* dequant_ptr, uint16_t* eob_ptr, const int16_t* scan,
                                         const int16_t* iscan, const QmVal* qm_ptr, const QmVal* iqm_ptr,
                                         const int32_t log_scale) {
    // Encoder shift multipliers are positive powers of two. Preserve the two
    // rounding steps for callers supplying a general fixed-point multiplier.
    if (quant_shift_ptr[0] <= 0 || quant_shift_ptr[1] <= 0 || (quant_shift_ptr[0] & (quant_shift_ptr[0] - 1)) ||
        (quant_shift_ptr[1] & (quant_shift_ptr[1] - 1))) {
        svt_aom_highbd_quantize_b_c(coeff_ptr,
                                    n_coeffs,
                                    zbin_ptr,
                                    round_ptr,
                                    quant_ptr,
                                    quant_shift_ptr,
                                    qcoeff_ptr,
                                    dqcoeff_ptr,
                                    dequant_ptr,
                                    eob_ptr,
                                    scan,
                                    iscan,
                                    qm_ptr,
                                    iqm_ptr,
                                    log_scale);
        return;
    }
    (void)scan;
    quantize(coeff_ptr,
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
             false,
             true);
}
#endif // CONFIG_ENABLE_HIGH_BIT_DEPTH

// Keep the eight errors in output order from the widening load to the store.
// Without matrix rounding the difference of squares takes a single multiply.
static AOM_FORCE_INLINE void rdoq_deltas(const TranLow* t, const TranLow* q, const TranLow* dq, const QmVal* qm,
                                         const QmVal* iqm, int dequant, int shift, int n, int64_t* out) {
    const __m128i scale = _mm_cvtsi32_si128(shift);
    for (int i = 0; i < n; i += 8) {
        const __m256i orig  = _mm256_abs_epi32(_mm256_loadu_si256((const __m256i*)(t + i)));
        const __m256i recon = _mm256_abs_epi32(_mm256_loadu_si256((const __m256i*)(dq + i)));
        const __m256i level = _mm256_abs_epi32(_mm256_loadu_si256((const __m256i*)(q + i)));
        __m256i       step  = _mm256_set1_epi32(dequant);
        if (iqm) {
            const __m256i iw = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(iqm + i)));
            step             = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(step, iw), _mm256_set1_epi32(1 << (AOM_QM_BITS - 1))), AOM_QM_BITS);
        }
        const __m256i low = _mm256_srl_epi32(
            _mm256_mullo_epi32(_mm256_max_epi32(_mm256_sub_epi32(level, _mm256_set1_epi32(1)), _mm256_setzero_si256()),
                               step),
            scale);
        const __m256i a = _mm256_sll_epi32(_mm256_sub_epi32(recon, orig), scale);
        const __m256i b = _mm256_sll_epi32(_mm256_sub_epi32(low, orig), scale);
        __m512i       delta;
        if (qm) {
            const __m512i w     = _mm512_cvtepu8_epi64(_mm_loadl_epi64((const __m128i*)(qm + i)));
            const __m512i u     = _mm512_mul_epi32(_mm512_cvtepi32_epi64(a), w),
                          v     = _mm512_mul_epi32(_mm512_cvtepi32_epi64(b), w);
            const __m512i round = _mm512_set1_epi64(1 << (2 * AOM_QM_BITS - 1));
            const __m512i aa    = _mm512_srli_epi64(_mm512_add_epi64(_mm512_mullo_epi64(u, u), round), 2 * AOM_QM_BITS);
            const __m512i bb    = _mm512_srli_epi64(_mm512_add_epi64(_mm512_mullo_epi64(v, v), round), 2 * AOM_QM_BITS);
            delta               = _mm512_sub_epi64(aa, bb);
        } else {
            const __m512i u = _mm512_cvtepi32_epi64(a), v = _mm512_cvtepi32_epi64(b);
            delta = _mm512_mullo_epi64(_mm512_sub_epi64(u, v), _mm512_add_epi64(u, v));
        }
        _mm512_storeu_si512(
            out + i, _mm512_mask_mov_epi64(delta, _mm256_cmpgt_epi32_mask(orig, recon), _mm512_set1_epi64(INT64_MIN)));
    }
}

void svt_av1_rdoq_deltas_avx512(const TranLow* t, const TranLow* q, const TranLow* dq, const QmVal* qm,
                                const QmVal* iqm, int dequant, int shift, int n, int64_t* out) {
    if (qm) {
        if (iqm) {
            rdoq_deltas(t, q, dq, qm, iqm, dequant, shift, n, out);
        } else {
            rdoq_deltas(t, q, dq, qm, NULL, dequant, shift, n, out);
        }
    } else if (iqm) {
        rdoq_deltas(t, q, dq, NULL, iqm, dequant, shift, n, out);
    } else {
        rdoq_deltas(t, q, dq, NULL, NULL, dequant, shift, n, out);
    }
}
#endif
