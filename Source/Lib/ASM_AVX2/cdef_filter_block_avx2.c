/*
* Copyright(c) 2026 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

// Native 8-bit CDEF filter kernels for AVX2: taps stay in uint8 lanes, so no
// 16-bit sentinel buffer is needed. Bit-exact with svt_cdef_filter_block_8bit_c.
// Four 8-pixel rows (i, i+sub, i+2*sub, i+3*sub) per 256-bit vector, with a
// 2-row path for the tail.

#include <immintrin.h>
#include <stddef.h>

#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "cdef.h"

// sign(a-b) * min(|a-b|, max(0, thr - (|a-b| >> damping))) in int8 lanes.
// |a-b| <= 255 and |result| <= 15, so the int8 lanes never overflow.
static INLINE __m256i constrain8x32_avx2(const __m256i a, const __m256i b, const __m256i thr, const __m128i damping,
                                         const __m256i shift_mask) {
    const __m256i bma     = _mm256_subs_epu8(b, a); // max(0, b - a)
    const __m256i amb     = _mm256_subs_epu8(a, b); // max(0, a - b)
    const __m256i diff    = _mm256_or_si256(amb, bma);
    const __m256i shifted = _mm256_and_si256(_mm256_srl_epi16(diff, damping), shift_mask);
    const __m256i s       = _mm256_subs_epu8(thr, shifted);
    // Exactly one of amb/bma is non-zero, so min(amb,s) - min(bma,s) is already
    // the signed clip: +clip when a > b, -clip when a < b, 0 when equal.
    return _mm256_sub_epi8(_mm256_min_epu8(amb, s), _mm256_min_epu8(bma, s));
}

// Byte-lane mask that clears the bits a 16-bit shift drags across byte boundaries.
static INLINE __m256i byte_shift_mask(const int damping) {
    // damping is an int and never reaches 32, so 0xFF >> damping already yields 0
    // for damping >= 8; no branch needed.
    return _mm256_set1_epi8((char)(0xFF >> damping));
}

// res = clamp(row + ((sum + 8 + (sum < 0)) >> 4), min, max), for 16 int16 lanes
// holding two 8-pixel rows. Returns the two rows packed as bytes in lanes 0 and 1.
static INLINE __m256i cdef_finalize8_avx2(const __m256i sum, const __m128i row_u8, const __m128i min_u8,
                                          const __m128i max_u8) {
    const __m256i row16 = _mm256_cvtepu8_epi16(row_u8);
    const __m256i min16 = _mm256_cvtepu8_epi16(min_u8);
    const __m256i max16 = _mm256_cvtepu8_epi16(max_u8);
    // sum < 0 ? sum - 1 : sum
    const __m256i s   = _mm256_add_epi16(sum, _mm256_cmpgt_epi16(_mm256_setzero_si256(), sum));
    __m256i       res = _mm256_add_epi16(_mm256_srai_epi16(_mm256_add_epi16(s, _mm256_set1_epi16(8)), 4), row16);
    res               = _mm256_min_epi16(_mm256_max_epi16(res, min16), max16);
    return _mm256_packus_epi16(res, res);
}

// Widening int8 -> int16 multiply-accumulate of a tap against 32 constrained lanes.
static INLINE void accum_taps(__m256i* acc_lo, __m256i* acc_hi, const __m256i tap16, const __m256i csum) {
    const __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(csum));
    const __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(csum, 1));
    *acc_lo          = _mm256_add_epi16(*acc_lo, _mm256_mullo_epi16(tap16, lo));
    *acc_hi          = _mm256_add_epi16(*acc_hi, _mm256_mullo_epi16(tap16, hi));
}

// Taps are small positive values so they are safe as the unsigned maddubs
// operand; the constrained sums are the signed one.
//
// Row order in the vectors is (r0, r2, r1, r3), which makes the per-128-bit-lane
// unpack land rows {0,1} in the low result and rows {2,3} in the high result.
static INLINE void accum_taps2(__m256i* acc_lo, __m256i* acc_hi, const __m256i taps_u8, const __m256i csum_a,
                               const __m256i csum_b) {
    const __m256i il = _mm256_unpacklo_epi8(csum_a, csum_b);
    const __m256i ih = _mm256_unpackhi_epi8(csum_a, csum_b);
    *acc_lo          = _mm256_add_epi16(*acc_lo, _mm256_maddubs_epi16(taps_u8, il));
    *acc_hi          = _mm256_add_epi16(*acc_hi, _mm256_maddubs_epi16(taps_u8, ih));
}

static INLINE __m256i load_4rows_x8(const uint8_t* r0, const uint8_t* r1, const uint8_t* r2, const uint8_t* r3) {
    const __m128i a = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)r0), _mm_loadl_epi64((const __m128i*)r1));
    const __m128i b = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)r2), _mm_loadl_epi64((const __m128i*)r3));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
}

// Compile-time row stride, so the four displacements fold into the addressing
// mode. Row order is (0, 2S, 1S, 3S) so the per-128-bit-lane unpack in
// accum_taps2 lands rows {0,1} low and rows {2,3} high.
// Natural row order (0,S,2S,3S), used by the bounded kernels.
#define LOAD4_NAT(BASE, OFF, S)                                                                              \
    _mm256_inserti128_si256(                                                                                 \
        _mm256_castsi128_si256(                                                                              \
            _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)((BASE) + (OFF))),                            \
                               _mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + 1 * (S) * CDEF_BSTRIDE)))), \
        _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + 2 * (S) * CDEF_BSTRIDE)),       \
                           _mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + 3 * (S) * CDEF_BSTRIDE))),      \
        1)

#define LOAD4_ORD(BASE, OFF, S)                                                                              \
    _mm256_inserti128_si256(                                                                                 \
        _mm256_castsi128_si256(                                                                              \
            _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)((BASE) + (OFF))),                            \
                               _mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + 2 * (S) * CDEF_BSTRIDE)))), \
        _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + 1 * (S) * CDEF_BSTRIDE)),       \
                           _mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + 3 * (S) * CDEF_BSTRIDE))),      \
        1)

// Full groups keep the compile-time stride; a partial tail uses row pointers
// clamped to the last valid row so the gather never reads past the block.
// Argument order is (r0, r2, r1, r3) to match LOAD4_ORD's (0, 2S, 1S, 3S).
#define LOAD4_GUARD(NROWS, BASE, R0, R1, R2, R3, OFF, S) \
    (((NROWS) == 4) ? LOAD4_ORD(BASE, (OFF), S) : load_4rows_x8((R0) + (OFF), (R2) + (OFF), (R1) + (OFF), (R3) + (OFF)))

static INLINE void store_4rows_x8(uint8_t* dst, const int32_t dstride, const int i, const int sub,
                                  const __m256i packed_lo, const __m256i packed_hi) {
    _mm_storel_epi64((__m128i*)(dst + (i + 0 * sub) * dstride), _mm256_castsi256_si128(packed_lo));
    _mm_storel_epi64((__m128i*)(dst + (i + 1 * sub) * dstride), _mm256_extracti128_si256(packed_lo, 1));
    _mm_storel_epi64((__m128i*)(dst + (i + 2 * sub) * dstride), _mm256_castsi256_si128(packed_hi));
    _mm_storel_epi64((__m128i*)(dst + (i + 3 * sub) * dstride), _mm256_extracti128_si256(packed_hi, 1));
}

// Stores only the rows that exist: a partial tail computes duplicated rows that
// must not be written past the block bottom.
static INLINE void store_nrows_x8(uint8_t* dst, const int32_t dstride, const int i, const int sub,
                                  const __m256i packed_lo, const __m256i packed_hi, const int nrows) {
    const __m128i r[4] = {_mm256_castsi256_si128(packed_lo),
                          _mm256_extracti128_si256(packed_lo, 1),
                          _mm256_castsi256_si128(packed_hi),
                          _mm256_extracti128_si256(packed_hi, 1)};
    for (int k = 0; k < nrows; k++) {
        _mm_storel_epi64((__m128i*)(dst + (i + k * sub) * dstride), r[k]);
    }
}

// One instantiation per subsampling factor: two macro copies in a single
// function made GCC allocate registers across both bodies and spill broadcasts
// to a 520-byte frame.
#define DEFINE_8XN_IMPL(NAME, S)                                                                                      \
    static void NAME(uint8_t*       dst,                                                                              \
                     int32_t        dstride,                                                                          \
                     const uint8_t* in,                                                                               \
                     int32_t        pri_strength,                                                                     \
                     int32_t        sec_strength,                                                                     \
                     int32_t        dir,                                                                              \
                     int32_t        damping,                                                                          \
                     int32_t        coeff_shift,                                                                      \
                     uint8_t        height) {                                                                                \
        const int*      pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];                    \
        const int*      sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];                    \
        const int32_t   pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;                  \
        const int32_t   sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;                  \
        const ptrdiff_t po1         = svt_aom_eb_cdef_directions[dir][0];                                             \
        const ptrdiff_t po2         = svt_aom_eb_cdef_directions[dir][1];                                             \
        const ptrdiff_t s1o1        = svt_aom_eb_cdef_directions[dir + 2][0];                                         \
        const ptrdiff_t s1o2        = svt_aom_eb_cdef_directions[dir + 2][1];                                         \
        const ptrdiff_t s2o1        = svt_aom_eb_cdef_directions[dir - 2][0];                                         \
        const ptrdiff_t s2o2        = svt_aom_eb_cdef_directions[dir - 2][1];                                         \
        const __m256i   prithr      = _mm256_set1_epi8((char)pri_strength);                                           \
        const __m256i   secthr      = _mm256_set1_epi8((char)sec_strength);                                           \
        const __m128i   pridamp     = _mm_cvtsi32_si128(pri_damping);                                                 \
        const __m128i   secdamp     = _mm_cvtsi32_si128(sec_damping);                                                 \
        const __m256i   primask     = byte_shift_mask(pri_damping);                                                   \
        const __m256i   secmask     = byte_shift_mask(sec_damping);                                                   \
        const __m256i   pri_tt      = _mm256_set1_epi16((short)(((pri_taps[1] & 0xFF) << 8) | (pri_taps[0] & 0xFF))); \
        const __m256i   sec_tt      = _mm256_set1_epi16((short)(((sec_taps[1] & 0xFF) << 8) | (sec_taps[0] & 0xFF))); \
        int             i           = 0;                                                                              \
        for (; i < height; i += 4 * (S)) {                                                                            \
            const int      nrows = ((height - i) + (S) - 1) / (S) < 4 ? ((height - i) + (S) - 1) / (S) : 4;           \
            const uint8_t* base  = in + i * CDEF_BSTRIDE;                                                             \
            /* A partial tail must not gather rows past the block: clamp to the  */                                   \
            /* last valid row, as the bounded kernels do. Full groups keep the   */                                   \
            /* compile-time stride so the offsets stay folded into the address.  */                                   \
            const uint8_t* rp0 = base;                                                                                \
            const uint8_t* rp1 = base + (nrows > 1 ? 1 : 0) * (S) * CDEF_BSTRIDE;                                     \
            const uint8_t* rp2 = base + (nrows > 2 ? 2 : nrows - 1) * (S) * CDEF_BSTRIDE;                             \
            const uint8_t* rp3 = base + (nrows > 3 ? 3 : nrows - 1) * (S) * CDEF_BSTRIDE;                             \
            const __m256i  row = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, 0, S);                                  \
            __m256i        mn = row, mx = row, tap, ca, cb;                                                           \
            __m256i        suma = _mm256_setzero_si256();                                                             \
            __m256i        sumb = _mm256_setzero_si256();                                                             \
            if (pri_strength) {                                                                                       \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, po1, S);                                           \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                ca  = constrain8x32_avx2(tap, row, prithr, pridamp, primask);                                         \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, -po1, S);                                          \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                ca  = _mm256_add_epi8(ca, constrain8x32_avx2(tap, row, prithr, pridamp, primask));                    \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, po2, S);                                           \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                cb  = constrain8x32_avx2(tap, row, prithr, pridamp, primask);                                         \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, -po2, S);                                          \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                cb  = _mm256_add_epi8(cb, constrain8x32_avx2(tap, row, prithr, pridamp, primask));                    \
                accum_taps2(&suma, &sumb, pri_tt, ca, cb);                                                            \
            }                                                                                                         \
            if (sec_strength) {                                                                                       \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, s1o1, S);                                          \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                ca  = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);                                         \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, -s1o1, S);                                         \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                ca  = _mm256_add_epi8(ca, constrain8x32_avx2(tap, row, secthr, secdamp, secmask));                    \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, s2o1, S);                                          \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                ca  = _mm256_add_epi8(ca, constrain8x32_avx2(tap, row, secthr, secdamp, secmask));                    \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, -s2o1, S);                                         \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                ca  = _mm256_add_epi8(ca, constrain8x32_avx2(tap, row, secthr, secdamp, secmask));                    \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, s1o2, S);                                          \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                cb  = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);                                         \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, -s1o2, S);                                         \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                cb  = _mm256_add_epi8(cb, constrain8x32_avx2(tap, row, secthr, secdamp, secmask));                    \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, s2o2, S);                                          \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                cb  = _mm256_add_epi8(cb, constrain8x32_avx2(tap, row, secthr, secdamp, secmask));                    \
                tap = LOAD4_GUARD(nrows, base, rp0, rp1, rp2, rp3, -s2o2, S);                                         \
                mn  = _mm256_min_epu8(mn, tap);                                                                       \
                mx  = _mm256_max_epu8(mx, tap);                                                                       \
                cb  = _mm256_add_epi8(cb, constrain8x32_avx2(tap, row, secthr, secdamp, secmask));                    \
                accum_taps2(&suma, &sumb, sec_tt, ca, cb);                                                            \
            }                                                                                                         \
            const __m256i rowo = _mm256_permute4x64_epi64(row, _MM_SHUFFLE(3, 1, 2, 0));                              \
            const __m256i mno  = _mm256_permute4x64_epi64(mn, _MM_SHUFFLE(3, 1, 2, 0));                               \
            const __m256i mxo  = _mm256_permute4x64_epi64(mx, _MM_SHUFFLE(3, 1, 2, 0));                               \
            const __m256i plo  = cdef_finalize8_avx2(                                                                 \
                suma, _mm256_castsi256_si128(rowo), _mm256_castsi256_si128(mno), _mm256_castsi256_si128(mxo));       \
            const __m256i phi = cdef_finalize8_avx2(sumb,                                                             \
                                                    _mm256_extracti128_si256(rowo, 1),                                \
                                                    _mm256_extracti128_si256(mno, 1),                                 \
                                                    _mm256_extracti128_si256(mxo, 1));                                \
            if (nrows == 4)                                                                                           \
                store_4rows_x8(dst, dstride, i, S, plo, phi);                                                         \
            else                                                                                                      \
                store_nrows_x8(dst, dstride, i, S, plo, phi, nrows);                                                  \
        }                                                                                                             \
    }

DEFINE_8XN_IMPL(cdef_8xn_native_s1, 1)
DEFINE_8XN_IMPL(cdef_8xn_native_s2, 2)
DEFINE_8XN_IMPL(cdef_8xn_native_s4, 4)
#undef DEFINE_8XN_IMPL

#define LOAD4_S(BASE, OFF, S)                                                                       \
    _mm256_castsi128_si256(_mm_setr_epi32(*(const int*)((BASE) + (0 * (S)) * CDEF_BSTRIDE + (OFF)), \
                                          *(const int*)((BASE) + (1 * (S)) * CDEF_BSTRIDE + (OFF)), \
                                          *(const int*)((BASE) + (2 * (S)) * CDEF_BSTRIDE + (OFF)), \
                                          *(const int*)((BASE) + (3 * (S)) * CDEF_BSTRIDE + (OFF))))
#define LOAD8_S(BASE, OFF, S)                                                   \
    _mm256_setr_epi32(*(const int*)((BASE) + (0 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (1 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (2 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (3 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (4 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (5 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (6 * (S)) * CDEF_BSTRIDE + (OFF)), \
                      *(const int*)((BASE) + (7 * (S)) * CDEF_BSTRIDE + (OFF)))

static void svt_av1_cdef_filter_block_4xn_8_native_avx2(uint8_t* dst, int32_t dstride, const uint8_t* in,
                                                        int32_t pri_strength, int32_t sec_strength, int32_t dir,
                                                        int32_t damping, int32_t coeff_shift, uint8_t height,
                                                        uint8_t subsampling_factor) {
    const int       sub         = subsampling_factor;
    const int*      pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*      sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t   pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t   sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const ptrdiff_t po1         = svt_aom_eb_cdef_directions[dir][0];
    const ptrdiff_t po2         = svt_aom_eb_cdef_directions[dir][1];
    const ptrdiff_t s1o1        = svt_aom_eb_cdef_directions[dir + 2][0];
    const ptrdiff_t s1o2        = svt_aom_eb_cdef_directions[dir + 2][1];
    const ptrdiff_t s2o1        = svt_aom_eb_cdef_directions[dir - 2][0];
    const ptrdiff_t s2o2        = svt_aom_eb_cdef_directions[dir - 2][1];
    const __m256i   prithr      = _mm256_set1_epi8((char)pri_strength);
    const __m256i   secthr      = _mm256_set1_epi8((char)sec_strength);
    const __m128i   pridamp     = _mm_cvtsi32_si128(pri_damping);
    const __m128i   secdamp     = _mm_cvtsi32_si128(sec_damping);
    const __m256i   primask     = byte_shift_mask(pri_damping);
    const __m256i   secmask     = byte_shift_mask(sec_damping);
    const __m256i   pri_t0      = _mm256_set1_epi16((short)pri_taps[0]);
    const __m256i   pri_t1      = _mm256_set1_epi16((short)pri_taps[1]);
    const __m256i   sec_t0      = _mm256_set1_epi16((short)sec_taps[0]);
    const __m256i   sec_t1      = _mm256_set1_epi16((short)sec_taps[1]);
    int             i           = 0;
    for (; i + 7 * (sub) < height; i += 8 * (sub)) {
        const uint8_t* base = in + i * CDEF_BSTRIDE;
        const __m256i  row8 = ((sub == 1)       ? LOAD8_S(base, 0, 1)
                                   : (sub == 2) ? LOAD8_S(base, 0, 2)
                                                : LOAD8_S(base, 0, 4));
        __m256i        mn8 = row8, mx8 = row8, t8, d0, d1, d2, d3, dsum;
        __m256i        sa = _mm256_setzero_si256();
        __m256i        sb = _mm256_setzero_si256();
        if (pri_strength) {
            t8   = ((sub == 1) ? LOAD8_S(base, po1, 1) : (sub == 2) ? LOAD8_S(base, po1, 2) : LOAD8_S(base, po1, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d0   = constrain8x32_avx2(t8, row8, prithr, pridamp, primask);
            t8   = ((sub == 1) ? LOAD8_S(base, -po1, 1) : (sub == 2) ? LOAD8_S(base, -po1, 2) : LOAD8_S(base, -po1, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d1   = constrain8x32_avx2(t8, row8, prithr, pridamp, primask);
            dsum = _mm256_add_epi8(d0, d1);
            accum_taps(&sa, &sb, pri_t0, dsum);
            t8   = ((sub == 1) ? LOAD8_S(base, po2, 1) : (sub == 2) ? LOAD8_S(base, po2, 2) : LOAD8_S(base, po2, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d0   = constrain8x32_avx2(t8, row8, prithr, pridamp, primask);
            t8   = ((sub == 1) ? LOAD8_S(base, -po2, 1) : (sub == 2) ? LOAD8_S(base, -po2, 2) : LOAD8_S(base, -po2, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d1   = constrain8x32_avx2(t8, row8, prithr, pridamp, primask);
            dsum = _mm256_add_epi8(d0, d1);
            accum_taps(&sa, &sb, pri_t1, dsum);
        }
        if (sec_strength) {
            t8   = ((sub == 1) ? LOAD8_S(base, s1o1, 1) : (sub == 2) ? LOAD8_S(base, s1o1, 2) : LOAD8_S(base, s1o1, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d0   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            t8   = ((sub == 1)       ? LOAD8_S(base, -s1o1, 1)
                        : (sub == 2) ? LOAD8_S(base, -s1o1, 2)
                                     : LOAD8_S(base, -s1o1, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d1   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            t8   = ((sub == 1) ? LOAD8_S(base, s2o1, 1) : (sub == 2) ? LOAD8_S(base, s2o1, 2) : LOAD8_S(base, s2o1, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d2   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            t8   = ((sub == 1)       ? LOAD8_S(base, -s2o1, 1)
                        : (sub == 2) ? LOAD8_S(base, -s2o1, 2)
                                     : LOAD8_S(base, -s2o1, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d3   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            dsum = _mm256_add_epi8(_mm256_add_epi8(d0, d1), _mm256_add_epi8(d2, d3));
            accum_taps(&sa, &sb, sec_t0, dsum);
            t8   = ((sub == 1) ? LOAD8_S(base, s1o2, 1) : (sub == 2) ? LOAD8_S(base, s1o2, 2) : LOAD8_S(base, s1o2, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d0   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            t8   = ((sub == 1)       ? LOAD8_S(base, -s1o2, 1)
                        : (sub == 2) ? LOAD8_S(base, -s1o2, 2)
                                     : LOAD8_S(base, -s1o2, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d1   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            t8   = ((sub == 1) ? LOAD8_S(base, s2o2, 1) : (sub == 2) ? LOAD8_S(base, s2o2, 2) : LOAD8_S(base, s2o2, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d2   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            t8   = ((sub == 1)       ? LOAD8_S(base, -s2o2, 1)
                        : (sub == 2) ? LOAD8_S(base, -s2o2, 2)
                                     : LOAD8_S(base, -s2o2, 4));
            mn8  = _mm256_min_epu8(mn8, t8);
            mx8  = _mm256_max_epu8(mx8, t8);
            d3   = constrain8x32_avx2(t8, row8, secthr, secdamp, secmask);
            dsum = _mm256_add_epi8(_mm256_add_epi8(d0, d1), _mm256_add_epi8(d2, d3));
            accum_taps(&sa, &sb, sec_t1, dsum);
        }
        const __m256i pa = cdef_finalize8_avx2(
            sa, _mm256_castsi256_si128(row8), _mm256_castsi256_si128(mn8), _mm256_castsi256_si128(mx8));
        const __m256i pb = cdef_finalize8_avx2(
            sb, _mm256_extracti128_si256(row8, 1), _mm256_extracti128_si256(mn8, 1), _mm256_extracti128_si256(mx8, 1));
        const __m128i q0                         = _mm256_castsi256_si128(pa);
        const __m128i q1                         = _mm256_extracti128_si256(pa, 1);
        const __m128i q2                         = _mm256_castsi256_si128(pb);
        const __m128i q3                         = _mm256_extracti128_si256(pb, 1);
        *(int*)(dst + (i + 0 * (sub)) * dstride) = _mm_cvtsi128_si32(q0);
        *(int*)(dst + (i + 1 * (sub)) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(q0, 4));
        *(int*)(dst + (i + 2 * (sub)) * dstride) = _mm_cvtsi128_si32(q1);
        *(int*)(dst + (i + 3 * (sub)) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(q1, 4));
        *(int*)(dst + (i + 4 * (sub)) * dstride) = _mm_cvtsi128_si32(q2);
        *(int*)(dst + (i + 5 * (sub)) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(q2, 4));
        *(int*)(dst + (i + 6 * (sub)) * dstride) = _mm_cvtsi128_si32(q3);
        *(int*)(dst + (i + 7 * (sub)) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(q3, 4));
    }
    for (; i + 3 * (sub) < height; i += 4 * (sub)) {
        const uint8_t* base = in + i * CDEF_BSTRIDE;
        const __m256i row = ((sub == 1) ? LOAD4_S(base, 0, 1) : (sub == 2) ? LOAD4_S(base, 0, 2) : LOAD4_S(base, 0, 4));
        __m256i       mn = row, mx = row, tap, c0, c1, c2, c3, csum;
        __m256i       suma = _mm256_setzero_si256();
        __m256i       sumb = _mm256_setzero_si256();
        if (pri_strength) {
            tap  = ((sub == 1) ? LOAD4_S(base, po1, 1) : (sub == 2) ? LOAD4_S(base, po1, 2) : LOAD4_S(base, po1, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            tap  = ((sub == 1) ? LOAD4_S(base, -po1, 1) : (sub == 2) ? LOAD4_S(base, -po1, 2) : LOAD4_S(base, -po1, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t0, csum);
            tap  = ((sub == 1) ? LOAD4_S(base, po2, 1) : (sub == 2) ? LOAD4_S(base, po2, 2) : LOAD4_S(base, po2, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            tap  = ((sub == 1) ? LOAD4_S(base, -po2, 1) : (sub == 2) ? LOAD4_S(base, -po2, 2) : LOAD4_S(base, -po2, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t1, csum);
        }
        if (sec_strength) {
            tap  = ((sub == 1) ? LOAD4_S(base, s1o1, 1) : (sub == 2) ? LOAD4_S(base, s1o1, 2) : LOAD4_S(base, s1o1, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = ((sub == 1)       ? LOAD4_S(base, -s1o1, 1)
                        : (sub == 2) ? LOAD4_S(base, -s1o1, 2)
                                     : LOAD4_S(base, -s1o1, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = ((sub == 1) ? LOAD4_S(base, s2o1, 1) : (sub == 2) ? LOAD4_S(base, s2o1, 2) : LOAD4_S(base, s2o1, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c2   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = ((sub == 1)       ? LOAD4_S(base, -s2o1, 1)
                        : (sub == 2) ? LOAD4_S(base, -s2o1, 2)
                                     : LOAD4_S(base, -s2o1, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c3   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t0, csum);
            tap  = ((sub == 1) ? LOAD4_S(base, s1o2, 1) : (sub == 2) ? LOAD4_S(base, s1o2, 2) : LOAD4_S(base, s1o2, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = ((sub == 1)       ? LOAD4_S(base, -s1o2, 1)
                        : (sub == 2) ? LOAD4_S(base, -s1o2, 2)
                                     : LOAD4_S(base, -s1o2, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = ((sub == 1) ? LOAD4_S(base, s2o2, 1) : (sub == 2) ? LOAD4_S(base, s2o2, 2) : LOAD4_S(base, s2o2, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c2   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = ((sub == 1)       ? LOAD4_S(base, -s2o2, 1)
                        : (sub == 2) ? LOAD4_S(base, -s2o2, 2)
                                     : LOAD4_S(base, -s2o2, 4));
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c3   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t1, csum);
        }
        const __m256i plo = cdef_finalize8_avx2(
            suma, _mm256_castsi256_si128(row), _mm256_castsi256_si128(mn), _mm256_castsi256_si128(mx));
        const __m128i g01                        = _mm256_castsi256_si128(plo);
        const __m128i g23                        = _mm256_extracti128_si256(plo, 1);
        *(int*)(dst + (i + 0 * (sub)) * dstride) = _mm_cvtsi128_si32(g01);
        *(int*)(dst + (i + 1 * (sub)) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(g01, 4));
        *(int*)(dst + (i + 2 * (sub)) * dstride) = _mm_cvtsi128_si32(g23);
        *(int*)(dst + (i + 3 * (sub)) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(g23, 4));
        (void)sumb;
    }
    for (; i < height; i += 2 * sub) {
        const uint8_t* base = in + i * CDEF_BSTRIDE;
#define LOAD2(BASE, OFF)                                                                            \
    _mm256_castsi128_si256(_mm_setr_epi32(*(const int*)((BASE) + (0 * sub) * CDEF_BSTRIDE + (OFF)), \
                                          *(const int*)((BASE) + (1 * sub) * CDEF_BSTRIDE + (OFF)), \
                                          0,                                                        \
                                          0))
        const __m256i row = LOAD2(base, 0);
        __m256i       mn = row, mx = row, tap, c0, c1, c2, c3, csum;
        __m256i       suma = _mm256_setzero_si256();
        __m256i       sumb = _mm256_setzero_si256();
        if (pri_strength) {
            tap  = LOAD2(base, po1);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            tap  = LOAD2(base, -po1);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t0, csum);
            tap  = LOAD2(base, po2);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            tap  = LOAD2(base, -po2);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, prithr, pridamp, primask);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t1, csum);
        }
        if (sec_strength) {
            tap  = LOAD2(base, s1o1);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = LOAD2(base, -s1o1);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = LOAD2(base, s2o1);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c2   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = LOAD2(base, -s2o1);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c3   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t0, csum);
            tap  = LOAD2(base, s1o2);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c0   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = LOAD2(base, -s1o2);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c1   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = LOAD2(base, s2o2);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c2   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            tap  = LOAD2(base, -s2o2);
            mn   = _mm256_min_epu8(mn, tap);
            mx   = _mm256_max_epu8(mx, tap);
            c3   = constrain8x32_avx2(tap, row, secthr, secdamp, secmask);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t1, csum);
        }
#undef LOAD2
        const __m256i plo = cdef_finalize8_avx2(
            suma, _mm256_castsi256_si128(row), _mm256_castsi256_si128(mn), _mm256_castsi256_si128(mx));
        const __m128i g01                      = _mm256_castsi256_si128(plo);
        *(int*)(dst + (i + 0 * sub) * dstride) = _mm_cvtsi128_si32(g01);
        *(int*)(dst + (i + 1 * sub) * dstride) = _mm_cvtsi128_si32(_mm_srli_si128(g01, 4));
        (void)sumb;
    }
}

void svt_cdef_filter_block_8bit_avx2(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                     int32_t sec_strength, int32_t dir, int32_t damping, int32_t bsize,
                                     int32_t coeff_shift, uint8_t subsampling_factor) {
    // Dispatch straight to the per-stride implementation rather than through a
    // generic wrapper, which would add a call layer per block.
    if (bsize == BLOCK_8X8) {
        if (subsampling_factor == 1) {
            cdef_8xn_native_s1(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        } else if (subsampling_factor == 2) {
            cdef_8xn_native_s2(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        } else {
            cdef_8xn_native_s4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        }
    } else if (bsize == BLOCK_4X8) {
        svt_av1_cdef_filter_block_4xn_8_native_avx2(
            dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8, subsampling_factor);
    } else if (bsize == BLOCK_8X4) {
        if (subsampling_factor == 1) {
            cdef_8xn_native_s1(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
        } else if (subsampling_factor == 2) {
            cdef_8xn_native_s2(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
        } else {
            cdef_8xn_native_s4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
        }
    } else {
        svt_av1_cdef_filter_block_4xn_8_native_avx2(
            dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4, 1);
    }
}

// Boundary-aware kernels. Same math as the interior kernels, but every tap is
// masked per lane by geometry so off-frame taps drop out of the sum, the max
// (off lanes -> 0) and the min (off lanes -> 255). Bit-exact with
// svt_cdef_filter_block_8bit_bounded_c.

// Column availability for an 8-wide row, replicated across all four row slots.
static INLINE __m256i bnd_col32_w8(const int dc, const int edge_left, const int edge_right) {
    if (!edge_left && !edge_right) {
        return _mm256_set1_epi8((char)0xFF);
    }
    uint8_t m[8];
    for (int c = 0; c < 8; c++) {
        const int off = (edge_left && (c + dc) < 0) || (edge_right && (c + dc) >= 8);
        m[c]          = off ? 0x00 : 0xFF;
    }
    int64_t w;
    memcpy(&w, m, 8);
    return _mm256_set1_epi64x(w);
}

// Column availability for a 4-wide row, replicated across all four row slots.
static INLINE __m256i bnd_col32_w4(const int dc, const int edge_left, const int edge_right) {
    if (!edge_left && !edge_right) {
        return _mm256_set1_epi8((char)0xFF);
    }
    uint8_t m[4];
    for (int c = 0; c < 4; c++) {
        const int off = (edge_left && (c + dc) < 0) || (edge_right && (c + dc) >= 4);
        m[c]          = off ? 0x00 : 0xFF;
    }
    int32_t w;
    memcpy(&w, m, 4);
    return _mm256_set1_epi32(w);
}

// Replicated across all four row slots.
// Row availability for the four rows i + k*sub, as whole 8-byte (w8) row slots.
static INLINE __m256i bnd_row32_w8(const int i, const int sub, const int dr, const int edge_top, const int edge_bottom,
                                   const int rows, const int nrows) {
    int64_t v[4];
    for (int k = 0; k < 4; k++) {
        const int r  = i + (k < nrows ? k : nrows - 1) * sub;
        const int ok = !((edge_top && (r + dr) < 0) || (edge_bottom && (r + dr) >= rows));
        v[k]         = ok ? (int64_t)0xFFFFFFFFFFFFFFFFULL : 0;
    }
    return _mm256_set_epi64x(v[3], v[2], v[1], v[0]);
}

// Four 4-wide rows packed into the low 16 bytes.
// Row availability for four 4-wide rows packed into the low 16 bytes.
static INLINE __m256i bnd_row32_w4(const int i, const int sub, const int dr, const int edge_top, const int edge_bottom,
                                   const int rows, const int nrows) {
    int32_t v[8];
    for (int k = 0; k < 4; k++) {
        const int r  = i + (k < nrows ? k : nrows - 1) * sub;
        const int ok = !((edge_top && (r + dr) < 0) || (edge_bottom && (r + dr) >= rows));
        v[k]         = ok ? (int32_t)0xFFFFFFFFu : 0;
        v[k + 4]     = 0;
    }
    return _mm256_setr_epi32(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

// Per-tap geometry, hoisted out of the row loop.
typedef struct {
    int     off[6];
    int     dr[6];
    __m256i colp[6], coln[6];
    int     rsensp[6], rsensn[6], needp[6], needn[6];
} BndTapCfg;

static INLINE void bnd_setup(BndTapCfg* cfg, const int dir, const int edge_top, const int edge_left,
                             const int edge_bottom, const int edge_right, const int width) {
    const int idx[6][1] = {{dir}, {dir}, {dir + 2}, {dir + 2}, {dir - 2}, {dir - 2}};
    const int sel[6]    = {0, 1, 0, 1, 0, 1};
    for (int b = 0; b < 6; b++) {
        cfg->off[b]  = svt_aom_eb_cdef_directions[idx[b][0]][sel[b]];
        cfg->dr[b]   = svt_aom_eb_cdef_directions_rc[idx[b][0]][sel[b]][0];
        const int dc = svt_aom_eb_cdef_directions_rc[idx[b][0]][sel[b]][1];
        if (width == 8) {
            cfg->colp[b] = bnd_col32_w8(dc, edge_left, edge_right);
            cfg->coln[b] = bnd_col32_w8(-dc, edge_left, edge_right);
        } else {
            cfg->colp[b] = bnd_col32_w4(dc, edge_left, edge_right);
            cfg->coln[b] = bnd_col32_w4(-dc, edge_left, edge_right);
        }
        const int csp  = (edge_left && dc < 0) || (edge_right && dc > 0);
        const int csn  = (edge_left && -dc < 0) || (edge_right && -dc > 0);
        cfg->rsensp[b] = (edge_top && cfg->dr[b] < 0) || (edge_bottom && cfg->dr[b] > 0);
        cfg->rsensn[b] = (edge_top && -cfg->dr[b] < 0) || (edge_bottom && -cfg->dr[b] > 0);
        cfg->needp[b]  = csp || cfg->rsensp[b];
        cfg->needn[b]  = csn || cfg->rsensn[b];
    }
}

static void cdef_filter_block_8xn_8_bounded_avx2(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                                 int32_t sec_strength, int32_t dir, int32_t damping,
                                                 int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor,
                                                 int edge_top, int edge_left, int edge_bottom, int edge_right) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const int     rows        = height;

    BndTapCfg cfg;
    bnd_setup(&cfg, dir, edge_top, edge_left, edge_bottom, edge_right, 8);

    const __m256i prithr  = _mm256_set1_epi8((char)pri_strength);
    const __m256i secthr  = _mm256_set1_epi8((char)sec_strength);
    const __m128i pridamp = _mm_cvtsi32_si128(pri_damping);
    const __m128i secdamp = _mm_cvtsi32_si128(sec_damping);
    const __m256i primask = byte_shift_mask(pri_damping);
    const __m256i secmask = byte_shift_mask(sec_damping);
    const __m256i pri_t0  = _mm256_set1_epi16((short)pri_taps[0]);
    const __m256i pri_t1  = _mm256_set1_epi16((short)pri_taps[1]);
    const __m256i sec_t0  = _mm256_set1_epi16((short)sec_taps[0]);
    const __m256i sec_t1  = _mm256_set1_epi16((short)sec_taps[1]);
    const __m256i ones    = _mm256_set1_epi8((char)0xFF);

    const int sub = subsampling_factor;

    for (int i = 0; i < height; i += 4 * sub) {
        const int nrows = ((height - i) + sub - 1) / sub;
        const int nr    = nrows > 4 ? 4 : nrows;

        const uint8_t* rp[4];
        for (int k = 0; k < 4; k++) {
            rp[k] = in + (i + (k < nr ? k : nr - 1) * sub) * CDEF_BSTRIDE;
        }

        const uint8_t* base = in + i * CDEF_BSTRIDE;
// Full 4-row groups use the strided gather; only a partial tail needs rp[].
#define BND_LOAD(OFF)                                                                 \
    ((nr == 4) ? ((sub == 1) ? LOAD4_NAT(base, (OFF), 1) : LOAD4_NAT(base, (OFF), 2)) \
               : load_4rows_x8(rp[0] + (OFF), rp[1] + (OFF), rp[2] + (OFF), rp[3] + (OFF)))

        const __m256i row = BND_LOAD(0);
        __m256i       mn = row, mx = row, tap, av, c0, c1, c2, c3, csum;
        __m256i       suma = _mm256_setzero_si256();
        __m256i       sumb = _mm256_setzero_si256();

#define BND_TAP(B, SGN, THR, DAMP, MASK, CACC)                                                                                     \
    do {                                                                                                                           \
        const int _o    = (SGN) * cfg.off[B];                                                                                      \
        const int _need = (SGN) > 0 ? cfg.needp[B] : cfg.needn[B];                                                                 \
        const int _rs   = (SGN) > 0 ? cfg.rsensp[B] : cfg.rsensn[B];                                                               \
        tap             = BND_LOAD(_o);                                                                                            \
        if (!_need) {                                                                                                              \
            mn     = _mm256_min_epu8(mn, tap);                                                                                     \
            mx     = _mm256_max_epu8(mx, tap);                                                                                     \
            (CACC) = constrain8x32_avx2(tap, row, (THR), (DAMP), (MASK));                                                          \
        } else {                                                                                                                   \
            const __m256i _col = (SGN) > 0 ? cfg.colp[B] : cfg.coln[B];                                                            \
            av                 = _rs                                                                                               \
                                ? _mm256_and_si256(_col, bnd_row32_w8(i, sub, (SGN) * cfg.dr[B], edge_top, edge_bottom, rows, nr)) \
                                : _col;                                                                                            \
            mn                 = _mm256_min_epu8(mn, _mm256_or_si256(tap, _mm256_xor_si256(av, ones)));                            \
            mx                 = _mm256_max_epu8(mx, _mm256_and_si256(tap, av));                                                   \
            (CACC)             = _mm256_and_si256(constrain8x32_avx2(tap, row, (THR), (DAMP), (MASK)), av);                        \
        }                                                                                                                          \
    } while (0)

        if (pri_strength) {
            BND_TAP(0, 1, prithr, pridamp, primask, c0);
            BND_TAP(0, -1, prithr, pridamp, primask, c1);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t0, csum);
            BND_TAP(1, 1, prithr, pridamp, primask, c0);
            BND_TAP(1, -1, prithr, pridamp, primask, c1);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t1, csum);
        }
        if (sec_strength) {
            BND_TAP(2, 1, secthr, secdamp, secmask, c0);
            BND_TAP(2, -1, secthr, secdamp, secmask, c1);
            BND_TAP(4, 1, secthr, secdamp, secmask, c2);
            BND_TAP(4, -1, secthr, secdamp, secmask, c3);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t0, csum);
            BND_TAP(3, 1, secthr, secdamp, secmask, c0);
            BND_TAP(3, -1, secthr, secdamp, secmask, c1);
            BND_TAP(5, 1, secthr, secdamp, secmask, c2);
            BND_TAP(5, -1, secthr, secdamp, secmask, c3);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t1, csum);
        }
#undef BND_TAP
#undef BND_LOAD

        const __m256i plo = cdef_finalize8_avx2(
            suma, _mm256_castsi256_si128(row), _mm256_castsi256_si128(mn), _mm256_castsi256_si128(mx));
        const __m256i phi = cdef_finalize8_avx2(
            sumb, _mm256_extracti128_si256(row, 1), _mm256_extracti128_si256(mn, 1), _mm256_extracti128_si256(mx, 1));
        const __m128i out[4] = {_mm256_castsi256_si128(plo),
                                _mm256_extracti128_si256(plo, 1),
                                _mm256_castsi256_si128(phi),
                                _mm256_extracti128_si256(phi, 1)};
        for (int k = 0; k < nr; k++) {
            _mm_storel_epi64((__m128i*)(dst + (i + k * sub) * dstride), out[k]);
        }
    }
}

static void cdef_filter_block_4xn_8_bounded_avx2(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                                 int32_t sec_strength, int32_t dir, int32_t damping,
                                                 int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor,
                                                 int edge_top, int edge_left, int edge_bottom, int edge_right) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const int     rows        = height;

    BndTapCfg cfg;
    bnd_setup(&cfg, dir, edge_top, edge_left, edge_bottom, edge_right, 4);

    const __m256i prithr  = _mm256_set1_epi8((char)pri_strength);
    const __m256i secthr  = _mm256_set1_epi8((char)sec_strength);
    const __m128i pridamp = _mm_cvtsi32_si128(pri_damping);
    const __m128i secdamp = _mm_cvtsi32_si128(sec_damping);
    const __m256i primask = byte_shift_mask(pri_damping);
    const __m256i secmask = byte_shift_mask(sec_damping);
    const __m256i pri_t0  = _mm256_set1_epi16((short)pri_taps[0]);
    const __m256i pri_t1  = _mm256_set1_epi16((short)pri_taps[1]);
    const __m256i sec_t0  = _mm256_set1_epi16((short)sec_taps[0]);
    const __m256i sec_t1  = _mm256_set1_epi16((short)sec_taps[1]);
    const __m256i ones    = _mm256_set1_epi8((char)0xFF);

    const int sub = subsampling_factor;

    for (int i = 0; i < height; i += 4 * sub) {
        const int nrows = ((height - i) + sub - 1) / sub;
        const int nr    = nrows > 4 ? 4 : nrows;

        const uint8_t* rp[4];
        for (int k = 0; k < 4; k++) {
            rp[k] = in + (i + (k < nr ? k : nr - 1) * sub) * CDEF_BSTRIDE;
        }

#define LOAD4P(OFF)                                                     \
    _mm256_castsi128_si256(_mm_setr_epi32(*(const int*)(rp[0] + (OFF)), \
                                          *(const int*)(rp[1] + (OFF)), \
                                          *(const int*)(rp[2] + (OFF)), \
                                          *(const int*)(rp[3] + (OFF))))

        const __m256i row = LOAD4P(0);
        __m256i       mn = row, mx = row, tap, av, c0, c1, c2, c3, csum;
        __m256i       suma = _mm256_setzero_si256();
        __m256i       sumb = _mm256_setzero_si256();

#define BND_TAP4(B, SGN, THR, DAMP, MASK, CACC)                                                                                    \
    do {                                                                                                                           \
        const int _o    = (SGN) * cfg.off[B];                                                                                      \
        const int _need = (SGN) > 0 ? cfg.needp[B] : cfg.needn[B];                                                                 \
        const int _rs   = (SGN) > 0 ? cfg.rsensp[B] : cfg.rsensn[B];                                                               \
        tap             = LOAD4P(_o);                                                                                              \
        if (!_need) {                                                                                                              \
            mn     = _mm256_min_epu8(mn, tap);                                                                                     \
            mx     = _mm256_max_epu8(mx, tap);                                                                                     \
            (CACC) = constrain8x32_avx2(tap, row, (THR), (DAMP), (MASK));                                                          \
        } else {                                                                                                                   \
            const __m256i _col = (SGN) > 0 ? cfg.colp[B] : cfg.coln[B];                                                            \
            av                 = _rs                                                                                               \
                                ? _mm256_and_si256(_col, bnd_row32_w4(i, sub, (SGN) * cfg.dr[B], edge_top, edge_bottom, rows, nr)) \
                                : _col;                                                                                            \
            mn                 = _mm256_min_epu8(mn, _mm256_or_si256(tap, _mm256_xor_si256(av, ones)));                            \
            mx                 = _mm256_max_epu8(mx, _mm256_and_si256(tap, av));                                                   \
            (CACC)             = _mm256_and_si256(constrain8x32_avx2(tap, row, (THR), (DAMP), (MASK)), av);                        \
        }                                                                                                                          \
    } while (0)

        if (pri_strength) {
            BND_TAP4(0, 1, prithr, pridamp, primask, c0);
            BND_TAP4(0, -1, prithr, pridamp, primask, c1);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t0, csum);
            BND_TAP4(1, 1, prithr, pridamp, primask, c0);
            BND_TAP4(1, -1, prithr, pridamp, primask, c1);
            csum = _mm256_add_epi8(c0, c1);
            accum_taps(&suma, &sumb, pri_t1, csum);
        }
        if (sec_strength) {
            BND_TAP4(2, 1, secthr, secdamp, secmask, c0);
            BND_TAP4(2, -1, secthr, secdamp, secmask, c1);
            BND_TAP4(4, 1, secthr, secdamp, secmask, c2);
            BND_TAP4(4, -1, secthr, secdamp, secmask, c3);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t0, csum);
            BND_TAP4(3, 1, secthr, secdamp, secmask, c0);
            BND_TAP4(3, -1, secthr, secdamp, secmask, c1);
            BND_TAP4(5, 1, secthr, secdamp, secmask, c2);
            BND_TAP4(5, -1, secthr, secdamp, secmask, c3);
            csum = _mm256_add_epi8(_mm256_add_epi8(c0, c1), _mm256_add_epi8(c2, c3));
            accum_taps(&suma, &sumb, sec_t1, csum);
        }
#undef BND_TAP4
#undef LOAD4P

        const __m256i plo = cdef_finalize8_avx2(
            suma, _mm256_castsi256_si128(row), _mm256_castsi256_si128(mn), _mm256_castsi256_si128(mx));
        const __m128i g01  = _mm256_castsi256_si128(plo);
        const __m128i g23  = _mm256_extracti128_si256(plo, 1);
        const int     o[4] = {_mm_cvtsi128_si32(g01),
                              _mm_cvtsi128_si32(_mm_srli_si128(g01, 4)),
                              _mm_cvtsi128_si32(g23),
                              _mm_cvtsi128_si32(_mm_srli_si128(g23, 4))};
        for (int k = 0; k < nr; k++) {
            *(int*)(dst + (i + k * sub) * dstride) = o[k];
        }
        (void)sumb;
    }
}

void svt_cdef_filter_block_8bit_bounded_avx2(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                             int32_t sec_strength, int32_t dir, int32_t damping, int32_t bsize,
                                             int32_t coeff_shift, uint8_t subsampling_factor, int edge_top,
                                             int edge_left, int edge_bottom, int edge_right) {
    if (bsize == BLOCK_8X8) {
        cdef_filter_block_8xn_8_bounded_avx2(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             8,
                                             subsampling_factor,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    } else if (bsize == BLOCK_4X8) {
        cdef_filter_block_4xn_8_bounded_avx2(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             8,
                                             subsampling_factor,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    } else if (bsize == BLOCK_8X4) {
        cdef_filter_block_8xn_8_bounded_avx2(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             4,
                                             subsampling_factor,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    } else {
        cdef_filter_block_4xn_8_bounded_avx2(dst,
                                             dstride,
                                             in,
                                             pri_strength,
                                             sec_strength,
                                             dir,
                                             damping,
                                             coeff_shift,
                                             4,
                                             1,
                                             edge_top,
                                             edge_left,
                                             edge_bottom,
                                             edge_right);
    }
}
