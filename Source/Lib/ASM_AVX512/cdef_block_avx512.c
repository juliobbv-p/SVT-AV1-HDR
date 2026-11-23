/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "definitions.h"

#if EN_AVX512_SUPPORT
#include <immintrin.h>
#include "common_dsp_rtcd.h"
#include "bitstream_unit.h"
#include "cdef.h"
#include "memory_avx2.h"

static INLINE __m512i loadu_u16_8x4_avx512(const uint16_t* const src, const uint32_t stride) {
    const __m256i s0 = loadu_u16_8x2_avx2(src + 0 * stride, stride);
    const __m256i s1 = loadu_u16_8x2_avx2(src + 2 * stride, stride);
    return _mm512_inserti64x4(_mm512_castsi256_si512(s0), s1, 1);
}

// sign(a-b) * min(abs(a-b), max(0, threshold - (abs(a-b) >> adjdamp)))
static INLINE __m512i constrain16_avx512(const __m512i in0, const __m512i in1, const __m512i threshold,
                                         const __m128i damping) {
    const __m512i   diff = _mm512_sub_epi16(in0, in1);
    const __mmask32 sign = _mm512_movepi16_mask(diff);
    const __m512i   a    = _mm512_abs_epi16(diff);
    const __m512i   l    = _mm512_srl_epi16(a, damping);
    const __m512i   s    = _mm512_subs_epu16(threshold, l);
    const __m512i   m    = _mm512_min_epi16(a, s);
    return _mm512_mask_sub_epi16(m, sign, _mm512_setzero_si512(), m);
}

static INLINE void cdef_filter_block_8xn_16_pri_avx512(const uint16_t* const in, const __m128i damping,
                                                       const int32_t po, const __m512i row, const __m512i strength,
                                                       const __m512i pri_taps, __m512i* const max, __m512i* const min,
                                                       __m512i* const sum, uint8_t subsampling_factor) {
    const __m512i large = _mm512_set1_epi16(CDEF_VERY_LARGE);
    const __m512i p0    = loadu_u16_8x4_avx512(in + po, subsampling_factor * CDEF_BSTRIDE);
    const __m512i p1    = loadu_u16_8x4_avx512(in - po, subsampling_factor * CDEF_BSTRIDE);

    *max = _mm512_mask_max_epi16(*max, _mm512_cmpneq_epi16_mask(p0, large), p0, *max);
    *max = _mm512_mask_max_epi16(*max, _mm512_cmpneq_epi16_mask(p1, large), p1, *max);
    *min = _mm512_min_epi16(*min, p0);
    *min = _mm512_min_epi16(*min, p1);

    const __m512i q0 = constrain16_avx512(p0, row, strength, damping);
    const __m512i q1 = constrain16_avx512(p1, row, strength, damping);

    // sum += pri_taps * (p0 + p1)
    *sum = _mm512_add_epi16(*sum, _mm512_mullo_epi16(pri_taps, _mm512_add_epi16(q0, q1)));
}

static INLINE void cdef_filter_block_8xn_16_sec_avx512(const uint16_t* const in, const __m128i damping,
                                                       const int32_t so1, const int32_t so2, const __m512i row,
                                                       const __m512i strength, const __m512i sec_taps,
                                                       __m512i* const max, __m512i* const min, __m512i* const sum,
                                                       uint8_t subsampling_factor) {
    const __m512i large = _mm512_set1_epi16(CDEF_VERY_LARGE);
    const __m512i p0    = loadu_u16_8x4_avx512(in + so1, subsampling_factor * CDEF_BSTRIDE);
    const __m512i p1    = loadu_u16_8x4_avx512(in - so1, subsampling_factor * CDEF_BSTRIDE);
    const __m512i p2    = loadu_u16_8x4_avx512(in + so2, subsampling_factor * CDEF_BSTRIDE);
    const __m512i p3    = loadu_u16_8x4_avx512(in - so2, subsampling_factor * CDEF_BSTRIDE);

    *max = _mm512_mask_max_epi16(*max, _mm512_cmpneq_epi16_mask(p0, large), p0, *max);
    *max = _mm512_mask_max_epi16(*max, _mm512_cmpneq_epi16_mask(p1, large), p1, *max);
    *max = _mm512_mask_max_epi16(*max, _mm512_cmpneq_epi16_mask(p2, large), p2, *max);
    *max = _mm512_mask_max_epi16(*max, _mm512_cmpneq_epi16_mask(p3, large), p3, *max);
    *min = _mm512_min_epi16(*min, p0);
    *min = _mm512_min_epi16(*min, p1);
    *min = _mm512_min_epi16(*min, p2);
    *min = _mm512_min_epi16(*min, p3);

    const __m512i q0 = constrain16_avx512(p0, row, strength, damping);
    const __m512i q1 = constrain16_avx512(p1, row, strength, damping);
    const __m512i q2 = constrain16_avx512(p2, row, strength, damping);
    const __m512i q3 = constrain16_avx512(p3, row, strength, damping);

    // sum += sec_taps * (p0 + p1 + p2 + p3)
    *sum = _mm512_add_epi16(
        *sum, _mm512_mullo_epi16(sec_taps, _mm512_add_epi16(_mm512_add_epi16(q0, q1), _mm512_add_epi16(q2, q3))));
}

// subsampling_factor of 1 means no subsampling
// requires height/subsampling_factor >= 4
void svt_cdef_filter_block_8xn_16_avx512(const uint16_t* const in, const int32_t pri_strength,
                                         const int32_t sec_strength, const int32_t dir, int32_t pri_damping,
                                         int32_t sec_damping, const int32_t coeff_shift, uint16_t* const dst,
                                         const int32_t dstride, uint8_t height, uint8_t subsampling_factor) {
    if (!pri_strength || !sec_strength) {
        const bool    primary  = pri_strength != 0;
        const int     strength = primary ? pri_strength : sec_strength;
        const __m512i limit    = _mm512_set1_epi16(strength);
        const __m128i damp     = _mm_cvtsi32_si128(
            strength ? AOMMAX(0, (primary ? pri_damping : sec_damping) - get_msb(strength)) : 0);
        const int* taps = primary ? svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1]
                                  : svt_aom_eb_cdef_sec_taps[0];
        for (int y = 0; y < height; y += 4 * subsampling_factor) {
            const uint16_t* src = in + y * CDEF_BSTRIDE;
            const int       ss  = subsampling_factor * CDEF_BSTRIDE;
            __m512i         c   = loadu_u16_8x4_avx512(src, ss);
            if (strength) {
                __m512i sum = _mm512_setzero_si512();
                for (int k = 0; k < 2; ++k) {
                    const int off  = svt_aom_eb_cdef_directions[primary ? dir : dir + 2][k];
                    __m512i   pair = _mm512_add_epi16(
                        constrain16_avx512(loadu_u16_8x4_avx512(src + off, ss), c, limit, damp),
                        constrain16_avx512(loadu_u16_8x4_avx512(src - off, ss), c, limit, damp));
                    if (!primary) {
                        const int off2 = svt_aom_eb_cdef_directions[dir - 2][k];
                        pair           = _mm512_add_epi16(
                            pair,
                            _mm512_add_epi16(constrain16_avx512(loadu_u16_8x4_avx512(src + off2, ss), c, limit, damp),
                                             constrain16_avx512(loadu_u16_8x4_avx512(src - off2, ss), c, limit, damp)));
                    }
                    sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(pair, _mm512_set1_epi16(taps[k])));
                }
                // Either family alone has total tap weight <= 16.
                c = _mm512_add_epi16(
                    c, _mm512_mulhrs_epi16(_mm512_add_epi16(sum, _mm512_srai_epi16(sum, 15)), _mm512_set1_epi16(2048)));
            }
            uint16_t* d = dst + y * dstride;
            _mm_storeu_si128((__m128i*)d, _mm512_castsi512_si128(c));
            _mm_storeu_si128((__m128i*)(d + subsampling_factor * dstride), _mm512_extracti32x4_epi32(c, 1));
            _mm_storeu_si128((__m128i*)(d + 2 * subsampling_factor * dstride), _mm512_extracti32x4_epi32(c, 2));
            _mm_storeu_si128((__m128i*)(d + 3 * subsampling_factor * dstride), _mm512_extracti32x4_epi32(c, 3));
        }
        return;
    }
    const int32_t  po1              = svt_aom_eb_cdef_directions[dir][0];
    const int32_t  po2              = svt_aom_eb_cdef_directions[dir][1];
    const int32_t  s1o1             = svt_aom_eb_cdef_directions[(dir + 2)][0];
    const int32_t  s1o2             = svt_aom_eb_cdef_directions[(dir + 2)][1];
    const int32_t  s2o1             = svt_aom_eb_cdef_directions[(dir - 2)][0];
    const int32_t  s2o2             = svt_aom_eb_cdef_directions[(dir - 2)][1];
    const int32_t* pri_taps         = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t* sec_taps         = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const __m512i  pri_taps_0       = _mm512_set1_epi16(pri_taps[0]);
    const __m512i  pri_taps_1       = _mm512_set1_epi16(pri_taps[1]);
    const __m512i  sec_taps_0       = _mm512_set1_epi16(sec_taps[0]);
    const __m512i  sec_taps_1       = _mm512_set1_epi16(sec_taps[1]);
    const __m512i  duplicate_8      = _mm512_set1_epi16(8);
    const __m512i  pri_strength_256 = _mm512_set1_epi16(pri_strength);
    const __m512i  sec_strength_256 = _mm512_set1_epi16(sec_strength);
    const __m512i  zero             = _mm512_setzero_si512();

    if (pri_strength) {
        pri_damping = AOMMAX(0, pri_damping - get_msb(pri_strength));
    }
    if (sec_strength) {
        sec_damping = AOMMAX(0, sec_damping - get_msb(sec_strength));
    }

    const __m128i pri_d = _mm_cvtsi32_si128(pri_damping);
    const __m128i sec_d = _mm_cvtsi32_si128(sec_damping);

    for (uint32_t i = 0; i < height; i += (4 * subsampling_factor)) {
        const __m512i row = loadu_u16_8x4_avx512(in + i * CDEF_BSTRIDE, subsampling_factor * CDEF_BSTRIDE);
        __m512i       sum, res, max, min;

        min = max = row;
        sum       = zero;

        // Primary near taps
        cdef_filter_block_8xn_16_pri_avx512(
            in + i * CDEF_BSTRIDE, pri_d, po1, row, pri_strength_256, pri_taps_0, &max, &min, &sum, subsampling_factor);

        // Primary far taps
        cdef_filter_block_8xn_16_pri_avx512(
            in + i * CDEF_BSTRIDE, pri_d, po2, row, pri_strength_256, pri_taps_1, &max, &min, &sum, subsampling_factor);

        // Secondary near taps
        cdef_filter_block_8xn_16_sec_avx512(in + i * CDEF_BSTRIDE,
                                            sec_d,
                                            s1o1,
                                            s2o1,
                                            row,
                                            sec_strength_256,
                                            sec_taps_0,
                                            &max,
                                            &min,
                                            &sum,
                                            subsampling_factor);

        // Secondary far taps
        cdef_filter_block_8xn_16_sec_avx512(in + i * CDEF_BSTRIDE,
                                            sec_d,
                                            s1o2,
                                            s2o2,
                                            row,
                                            sec_strength_256,
                                            sec_taps_1,
                                            &max,
                                            &min,
                                            &sum,
                                            subsampling_factor);

        // res = row + ((sum - (sum < 0) + 8) >> 4)
        const __mmask32 mask = _mm512_cmpgt_epi16_mask(zero, sum);
        sum                  = _mm512_mask_add_epi16(sum, mask, sum, _mm512_set1_epi16(-1));
        res                  = _mm512_add_epi16(sum, duplicate_8);
        res                  = _mm512_srai_epi16(res, 4);
        res                  = _mm512_add_epi16(row, res);
        res                  = _mm512_max_epi16(res, min);
        res                  = _mm512_min_epi16(res, max);

        _mm_storeu_si128((__m128i*)&dst[i * dstride], _mm512_castsi512_si128(res));
        _mm_storeu_si128((__m128i*)&dst[(i + 1 * subsampling_factor) * dstride], _mm512_extracti32x4_epi32(res, 1));
        _mm_storeu_si128((__m128i*)&dst[(i + 2 * subsampling_factor) * dstride], _mm512_extracti32x4_epi32(res, 2));
        _mm_storeu_si128((__m128i*)&dst[(i + 3 * subsampling_factor) * dstride], _mm512_extracti32x4_epi32(res, 3));
    }
}
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
typedef struct SearchStripe {
    __m512i   index;
    __mmask16 mask;
    int       stride;
    bool      full;
} SearchStripe;

static INLINE SearchStripe stripe_layout(int stride, int w, int rows) {
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    SearchStripe  s;
    const __m512i y = _mm512_srli_epi32(lane, w == 8 ? 2 : 1);
    s.index         = _mm512_add_epi32(_mm512_mullo_epi32(y, _mm512_set1_epi32(stride)),
                               _mm512_slli_epi32(_mm512_and_si512(lane, _mm512_set1_epi32(w / 2 - 1)), 1));
    s.mask          = (__mmask16)((1u << (w * rows / 2)) - 1);
    s.stride        = stride;
    s.full          = w == 8 && rows == 4;
    return s;
}

static INLINE __m512i stripe_load(const uint16_t* p, const SearchStripe* s) {
    if (s->full) {
        return loadu_u16_8x4_avx512(p, s->stride);
    }
    return _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), s->mask, s->index, p, 2);
}

typedef struct CdefDifference {
    __m512i   mag;
    __mmask32 neg;
} CdefDifference;

static INLINE __m512i constrained(CdefDifference diff, __m512i strength, __m128i damp) {
    const __m512i cap = _mm512_subs_epu16(strength, _mm512_srl_epi16(diff.mag, damp));
    const __m512i m   = _mm512_min_epu16(diff.mag, cap);
    return _mm512_mask_sub_epi16(m, diff.neg, _mm512_setzero_si512(), m);
}

static INLINE __m512i family_sum(const CdefDifference* diff, int axes, int strength, int damping, int tap0, int tap1) {
    const __m512i s = _mm512_set1_epi16(strength);
    const __m128i d = _mm_cvtsi32_si128(AOMMAX(0, damping - get_msb(strength)));
    __m512i       a = _mm512_setzero_si512(), b = _mm512_setzero_si512();
    for (int i = 0; i < axes * 4; i += 4) {
        a = _mm512_add_epi16(a, _mm512_add_epi16(constrained(diff[i], s, d), constrained(diff[i + 1], s, d)));
        b = _mm512_add_epi16(b, _mm512_add_epi16(constrained(diff[i + 2], s, d), constrained(diff[i + 3], s, d)));
    }
    return _mm512_add_epi16(_mm512_mullo_epi16(a, _mm512_set1_epi16(tap0)),
                            _mm512_mullo_epi16(b, _mm512_set1_epi16(tap1)));
}

static void search_stripe(const uint16_t* src, const SearchStripe* layout, __m512i c, __m512i target, int dir,
                          int damping, const CdefSearchPlan* plan, const uint8_t* primary, int first, int end,
                          int sec_mask, bool add, uint64_t* ssd) {
    CdefDifference diff[12];
    __m512i        min = c, max = c;
    const bool     use_primary = end > first && plan->primary[end - 1];
    for (int a = use_primary ? 0 : 1; a < (sec_mask ? 3 : 1); ++a) {
        const int* offsets = svt_aom_eb_cdef_directions[dir + (a == 0 ? 0 : a == 1 ? 2 : -2)];
        for (int k = 0; k < 4; ++k) {
            const int     off   = offsets[k / 2];
            const __m512i pix   = stripe_load(src + ((k & 1) ? -off : off), layout);
            const __m512i d     = _mm512_sub_epi16(pix, c);
            diff[4 * a + k].mag = _mm512_abs_epi16(d);
            diff[4 * a + k].neg = _mm512_movepi16_mask(d);
            min                 = _mm512_min_epi16(min, pix);
            max                 = _mm512_mask_max_epi16(
                max, _mm512_cmpneq_epi16_mask(pix, _mm512_set1_epi16(CDEF_VERY_LARGE)), max, pix);
        }
    }
    __m512i sec[4];
    sec[0] = _mm512_setzero_si512();
    for (int s = 1; s < 4; ++s) {
        if (sec_mask & (1 << s)) {
            sec[s] = family_sum(diff + 4, 2, (s + (s == 3)) * 4, damping, 2, 1);
        }
    }
    for (int p = first; p < end; ++p) {
        __m512i pri = _mm512_setzero_si512();
        if (primary[p]) {
            const int* taps = svt_aom_eb_cdef_pri_taps[(primary[p] >> 2) & 1];
            pri             = family_sum(diff, 1, primary[p], damping, taps[0], taps[1]);
        }
        for (int i = plan->group_start[p]; i < plan->group_start[p + 1]; ++i) {
            const int     s   = plan->secondary[i];
            const __m512i sum = _mm512_add_epi16(pri, sec[s]);
            __m512i       pix = _mm512_add_epi16(
                c, _mm512_mulhrs_epi16(_mm512_add_epi16(sum, _mm512_srai_epi16(sum, 15)), _mm512_set1_epi16(2048)));
            if (primary[p] && s) {
                pix = _mm512_min_epi16(max, _mm512_max_epi16(min, pix));
            }
            const __m512i  err  = _mm512_sub_epi16(pix, target);
            const unsigned cost = _mm512_reduce_add_epi32(_mm512_madd_epi16(err, err));
            ssd[i]              = add ? ssd[i] + cost : cost;
        }
    }
}

void svt_cdef_search_block_10bit_avx512(const uint16_t* in, const uint16_t* target, int ts, const CdefSearchPlan* plan,
                                        const uint8_t primary[16], int dir, int damping, int bsize, int sub,
                                        uint64_t ssd[64]) {
    const int w = bsize == BLOCK_8X8 || bsize == BLOCK_8X4 ? 8 : 4;
    const int h = bsize == BLOCK_8X8 || bsize == BLOCK_4X8 ? 8 : 4;
    // The narrow implementation holds a complete stripe in one YMM.
    if (w * ((h + sub - 1) / sub) <= 16) {
        svt_cdef_search_block_10bit_avx2(in, target, ts, plan, primary, dir, damping, bsize, sub, ssd);
        return;
    }
    const int          rows = AOMMIN(32 / w, (h + sub - 1) / sub);
    const SearchStripe src = stripe_layout(sub * CDEF_BSTRIDE, w, rows), dst = stripe_layout(sub * ts, w, rows);
    for (int y = 0; y < h; y += rows * sub) {
        const uint16_t* p = in + y * CDEF_BSTRIDE;
        const __m512i   c = stripe_load(p, &src), t = stripe_load(target + y * ts, &dst);
        const int       separate = plan->primary_count && !plan->primary[0] && (dir & 3);
        if (separate) {
            search_stripe(p, &src, c, t, 0, damping, plan, primary, 0, 1, plan->zero_secondary_mask, y != 0, ssd);
        }
        if (separate < plan->primary_count) {
            search_stripe(p,
                          &src,
                          c,
                          t,
                          dir,
                          damping,
                          plan,
                          primary,
                          separate,
                          plan->primary_count,
                          plan->secondary_mask | (separate ? 0 : plan->zero_secondary_mask),
                          y != 0,
                          ssd);
        }
    }
}
#endif

#endif // EN_AVX512_SUPPORT
