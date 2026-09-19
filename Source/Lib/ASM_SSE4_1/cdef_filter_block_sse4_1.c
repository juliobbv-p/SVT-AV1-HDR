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

// Native 8-bit CDEF filter kernels for SSE4.1: taps stay in uint8 lanes, so no
// 16-bit sentinel buffer is needed. Bit-exact with svt_cdef_filter_block_8bit_c.
// An 8-pixel row is 8 bytes, so one register holds exactly two rows in natural
// order and no cross-lane permute fixups are needed.

#include <smmintrin.h>
#include <stddef.h>

#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "cdef.h"

// sign(a-b) * min(|a-b|, max(0, thr - (|a-b| >> damping))) in int8 lanes.
// |a-b| <= 255 and |result| <= 15, so the int8 lanes never overflow.
static INLINE __m128i constrain8x16_sse4(const __m128i a, const __m128i b, const __m128i thr, const __m128i damping,
                                         const __m128i shift_mask) {
    const __m128i bma     = _mm_subs_epu8(b, a); // max(0, b - a)
    const __m128i amb     = _mm_subs_epu8(a, b); // max(0, a - b)
    const __m128i diff    = _mm_or_si128(amb, bma);
    const __m128i shifted = _mm_and_si128(_mm_srl_epi16(diff, damping), shift_mask);
    const __m128i s       = _mm_subs_epu8(thr, shifted);
    // Exactly one of amb/bma is non-zero, so min(amb,s) - min(bma,s) is already
    // the signed clip and no sign mask or blend is needed.
    return _mm_sub_epi8(_mm_min_epu8(amb, s), _mm_min_epu8(bma, s));
}

// Byte-lane mask clearing the bits a 16-bit shift drags across byte boundaries.
static INLINE __m128i byte_shift_mask_128(const int damping) {
    return _mm_set1_epi8((char)(0xFF >> damping));
}

// res = clamp(row + ((sum + 8 + (sum < 0)) >> 4), min, max) for one 8-pixel row
// held as 8 int16 lanes. Returns the row packed into the low 8 bytes.
static INLINE __m128i cdef_finalize_row_sse4(const __m128i sum, const __m128i row_u8, const __m128i min_u8,
                                             const __m128i max_u8) {
    const __m128i row16 = _mm_cvtepu8_epi16(row_u8);
    const __m128i min16 = _mm_cvtepu8_epi16(min_u8);
    const __m128i max16 = _mm_cvtepu8_epi16(max_u8);
    const __m128i s     = _mm_add_epi16(sum, _mm_cmpgt_epi16(_mm_setzero_si128(), sum)); // sum<0 ? sum-1 : sum
    __m128i       res   = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(s, _mm_set1_epi16(8)), 4), row16);
    res                 = _mm_min_epi16(_mm_max_epi16(res, min16), max16);
    return _mm_packus_epi16(res, res);
}

// Taps are small positive values so they are safe as the unsigned maddubs
// operand; the constrained sums are the signed one. With two rows per register,
// unpacklo covers row 0 and unpackhi row 1, so the results need no lane fixup.
static INLINE void accum_taps2_sse4(__m128i* acc0, __m128i* acc1, const __m128i taps_u8, const __m128i csum_a,
                                    const __m128i csum_b) {
    *acc0 = _mm_add_epi16(*acc0, _mm_maddubs_epi16(taps_u8, _mm_unpacklo_epi8(csum_a, csum_b)));
    *acc1 = _mm_add_epi16(*acc1, _mm_maddubs_epi16(taps_u8, _mm_unpackhi_epi8(csum_a, csum_b)));
}

// Two 8-pixel rows, compile-time stride so the displacement folds into the
// addressing mode.
#define LOAD2_S(BASE, OFF, S)                                             \
    _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)((BASE) + (OFF))), \
                       _mm_loadl_epi64((const __m128i*)((BASE) + (OFF) + (S) * CDEF_BSTRIDE)))

static INLINE __m128i load_2rows_x8(const uint8_t* r0, const uint8_t* r1) {
    return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)r0), _mm_loadl_epi64((const __m128i*)r1));
}

// Full groups keep the compile-time stride; a partial tail uses a row pointer
// clamped to the last valid row so the gather never reads past the block.
#define LOAD2_GUARD(NROWS, BASE, R1, OFF, S) \
    (((NROWS) == 2) ? LOAD2_S(BASE, (OFF), S) : load_2rows_x8((BASE) + (OFF), (R1) + (OFF)))

// Four 4-pixel rows.
#define LOAD4W_S(BASE, OFF, S)                                               \
    _mm_setr_epi32(*(const int*)((BASE) + (0 * (S)) * CDEF_BSTRIDE + (OFF)), \
                   *(const int*)((BASE) + (1 * (S)) * CDEF_BSTRIDE + (OFF)), \
                   *(const int*)((BASE) + (2 * (S)) * CDEF_BSTRIDE + (OFF)), \
                   *(const int*)((BASE) + (3 * (S)) * CDEF_BSTRIDE + (OFF)))

// One instantiation per subsampling factor: a runtime `sub` in the tap loads
// costs a predictable branch plus a lost addressing-mode fold on every tap.
#define DEFINE_8XN_SSE4(NAME, S)                                                                                   \
    static void NAME(uint8_t*       dst,                                                                           \
                     int32_t        dstride,                                                                       \
                     const uint8_t* in,                                                                            \
                     int32_t        pri_strength,                                                                  \
                     int32_t        sec_strength,                                                                  \
                     int32_t        dir,                                                                           \
                     int32_t        damping,                                                                       \
                     int32_t        coeff_shift,                                                                   \
                     uint8_t        height) {                                                                             \
        const int*      pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];                 \
        const int*      sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];                 \
        const int32_t   pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;               \
        const int32_t   sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;               \
        const ptrdiff_t po1         = svt_aom_eb_cdef_directions[dir][0];                                          \
        const ptrdiff_t po2         = svt_aom_eb_cdef_directions[dir][1];                                          \
        const ptrdiff_t s1o1        = svt_aom_eb_cdef_directions[dir + 2][0];                                      \
        const ptrdiff_t s1o2        = svt_aom_eb_cdef_directions[dir + 2][1];                                      \
        const ptrdiff_t s2o1        = svt_aom_eb_cdef_directions[dir - 2][0];                                      \
        const ptrdiff_t s2o2        = svt_aom_eb_cdef_directions[dir - 2][1];                                      \
        int             i           = 0;                                                                           \
        for (; i < height; i += 2 * (S)) {                                                                         \
            const int      nrows = ((height - i) + (S) - 1) / (S) < 2 ? ((height - i) + (S) - 1) / (S) : 2;        \
            const uint8_t* base  = in + i * CDEF_BSTRIDE;                                                          \
            const uint8_t* rp1   = base + (nrows > 1 ? 1 : 0) * (S) * CDEF_BSTRIDE;                                \
            const __m128i  row   = LOAD2_GUARD(nrows, base, rp1, 0, S);                                            \
            __m128i        mn = row, mx = row, tap, ca, cb;                                                        \
            __m128i        sum0 = _mm_setzero_si128();                                                             \
            __m128i        sum1 = _mm_setzero_si128();                                                             \
            if (pri_strength) {                                                                                    \
                const __m128i prithr  = _mm_set1_epi8((char)pri_strength);                                         \
                const __m128i pridamp = _mm_cvtsi32_si128(pri_damping);                                            \
                const __m128i primask = byte_shift_mask_128(pri_damping);                                          \
                tap                   = LOAD2_GUARD(nrows, base, rp1, po1, S);                                     \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                ca                    = constrain8x16_sse4(tap, row, prithr, pridamp, primask);                    \
                tap                   = LOAD2_GUARD(nrows, base, rp1, -po1, S);                                    \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, prithr, pridamp, primask));  \
                tap                   = LOAD2_GUARD(nrows, base, rp1, po2, S);                                     \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                cb                    = constrain8x16_sse4(tap, row, prithr, pridamp, primask);                    \
                tap                   = LOAD2_GUARD(nrows, base, rp1, -po2, S);                                    \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, prithr, pridamp, primask));  \
                accum_taps2_sse4(&sum0,                                                                            \
                                 &sum1,                                                                            \
                                 _mm_set1_epi16((short)(((pri_taps[1] & 0xFF) << 8) | (pri_taps[0] & 0xFF))),      \
                                 ca,                                                                               \
                                 cb);                                                                              \
            }                                                                                                      \
            if (sec_strength) {                                                                                    \
                const __m128i secthr  = _mm_set1_epi8((char)sec_strength);                                         \
                const __m128i secdamp = _mm_cvtsi32_si128(sec_damping);                                            \
                const __m128i secmask = byte_shift_mask_128(sec_damping);                                          \
                tap                   = LOAD2_GUARD(nrows, base, rp1, s1o1, S);                                    \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                ca                    = constrain8x16_sse4(tap, row, secthr, secdamp, secmask);                    \
                tap                   = LOAD2_GUARD(nrows, base, rp1, -s1o1, S);                                   \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, secthr, secdamp, secmask));  \
                tap                   = LOAD2_GUARD(nrows, base, rp1, s2o1, S);                                    \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, secthr, secdamp, secmask));  \
                tap                   = LOAD2_GUARD(nrows, base, rp1, -s2o1, S);                                   \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, secthr, secdamp, secmask));  \
                tap                   = LOAD2_GUARD(nrows, base, rp1, s1o2, S);                                    \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                cb                    = constrain8x16_sse4(tap, row, secthr, secdamp, secmask);                    \
                tap                   = LOAD2_GUARD(nrows, base, rp1, -s1o2, S);                                   \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, secthr, secdamp, secmask));  \
                tap                   = LOAD2_GUARD(nrows, base, rp1, s2o2, S);                                    \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, secthr, secdamp, secmask));  \
                tap                   = LOAD2_GUARD(nrows, base, rp1, -s2o2, S);                                   \
                mn                    = _mm_min_epu8(mn, tap);                                                     \
                mx                    = _mm_max_epu8(mx, tap);                                                     \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, secthr, secdamp, secmask));  \
                accum_taps2_sse4(&sum0,                                                                            \
                                 &sum1,                                                                            \
                                 _mm_set1_epi16((short)(((sec_taps[1] & 0xFF) << 8) | (sec_taps[0] & 0xFF))),      \
                                 ca,                                                                               \
                                 cb);                                                                              \
            }                                                                                                      \
            const __m128i p0 = cdef_finalize_row_sse4(sum0, row, mn, mx);                                          \
            _mm_storel_epi64((__m128i*)(dst + (i + 0 * (S)) * dstride), p0);                                       \
            if (nrows == 2) {                                                                                      \
                const __m128i hi = _mm_srli_si128(row, 8);                                                         \
                const __m128i p1 = cdef_finalize_row_sse4(sum1, hi, _mm_srli_si128(mn, 8), _mm_srli_si128(mx, 8)); \
                _mm_storel_epi64((__m128i*)(dst + (i + 1 * (S)) * dstride), p1);                                   \
            }                                                                                                      \
        }                                                                                                          \
    }

DEFINE_8XN_SSE4(cdef_8xn_native_s1_sse4, 1)
DEFINE_8XN_SSE4(cdef_8xn_native_s2_sse4, 2)
DEFINE_8XN_SSE4(cdef_8xn_native_s4_sse4, 4)
#undef DEFINE_8XN_SSE4

// 4-wide blocks: a row is 4 bytes, so one register holds four rows. unpacklo
// covers rows {0,1} and unpackhi rows {2,3}.
#define DEFINE_4XN_SSE4(NAME, S)                                                                                  \
    static void NAME(uint8_t*       dst,                                                                          \
                     int32_t        dstride,                                                                      \
                     const uint8_t* in,                                                                           \
                     int32_t        pri_strength,                                                                 \
                     int32_t        sec_strength,                                                                 \
                     int32_t        dir,                                                                          \
                     int32_t        damping,                                                                      \
                     int32_t        coeff_shift,                                                                  \
                     uint8_t        height) {                                                                            \
        const int*      pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];                \
        const int*      sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];                \
        const int32_t   pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;              \
        const int32_t   sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;              \
        const ptrdiff_t po1         = svt_aom_eb_cdef_directions[dir][0];                                         \
        const ptrdiff_t po2         = svt_aom_eb_cdef_directions[dir][1];                                         \
        const ptrdiff_t s1o1        = svt_aom_eb_cdef_directions[dir + 2][0];                                     \
        const ptrdiff_t s1o2        = svt_aom_eb_cdef_directions[dir + 2][1];                                     \
        const ptrdiff_t s2o1        = svt_aom_eb_cdef_directions[dir - 2][0];                                     \
        const ptrdiff_t s2o2        = svt_aom_eb_cdef_directions[dir - 2][1];                                     \
        int             i           = 0;                                                                          \
        for (; i < height; i += 4 * (S)) {                                                                        \
            const int      nrows = ((height - i) + (S) - 1) / (S) < 4 ? ((height - i) + (S) - 1) / (S) : 4;       \
            const uint8_t* base  = in + i * CDEF_BSTRIDE;                                                         \
            const __m128i  row   = LOAD4W_S(base, 0, S);                                                          \
            __m128i        mn = row, mx = row, tap, ca, cb;                                                       \
            __m128i        sum0 = _mm_setzero_si128();                                                            \
            __m128i        sum1 = _mm_setzero_si128();                                                            \
            if (pri_strength) {                                                                                   \
                const __m128i prithr  = _mm_set1_epi8((char)pri_strength);                                        \
                const __m128i pridamp = _mm_cvtsi32_si128(pri_damping);                                           \
                const __m128i primask = byte_shift_mask_128(pri_damping);                                         \
                tap                   = LOAD4W_S(base, po1, S);                                                   \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                ca                    = constrain8x16_sse4(tap, row, prithr, pridamp, primask);                   \
                tap                   = LOAD4W_S(base, -po1, S);                                                  \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, prithr, pridamp, primask)); \
                tap                   = LOAD4W_S(base, po2, S);                                                   \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                cb                    = constrain8x16_sse4(tap, row, prithr, pridamp, primask);                   \
                tap                   = LOAD4W_S(base, -po2, S);                                                  \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, prithr, pridamp, primask)); \
                accum_taps2_sse4(&sum0,                                                                           \
                                 &sum1,                                                                           \
                                 _mm_set1_epi16((short)(((pri_taps[1] & 0xFF) << 8) | (pri_taps[0] & 0xFF))),     \
                                 ca,                                                                              \
                                 cb);                                                                             \
            }                                                                                                     \
            if (sec_strength) {                                                                                   \
                const __m128i secthr  = _mm_set1_epi8((char)sec_strength);                                        \
                const __m128i secdamp = _mm_cvtsi32_si128(sec_damping);                                           \
                const __m128i secmask = byte_shift_mask_128(sec_damping);                                         \
                tap                   = LOAD4W_S(base, s1o1, S);                                                  \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                ca                    = constrain8x16_sse4(tap, row, secthr, secdamp, secmask);                   \
                tap                   = LOAD4W_S(base, -s1o1, S);                                                 \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, secthr, secdamp, secmask)); \
                tap                   = LOAD4W_S(base, s2o1, S);                                                  \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, secthr, secdamp, secmask)); \
                tap                   = LOAD4W_S(base, -s2o1, S);                                                 \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                ca                    = _mm_add_epi8(ca, constrain8x16_sse4(tap, row, secthr, secdamp, secmask)); \
                tap                   = LOAD4W_S(base, s1o2, S);                                                  \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                cb                    = constrain8x16_sse4(tap, row, secthr, secdamp, secmask);                   \
                tap                   = LOAD4W_S(base, -s1o2, S);                                                 \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, secthr, secdamp, secmask)); \
                tap                   = LOAD4W_S(base, s2o2, S);                                                  \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, secthr, secdamp, secmask)); \
                tap                   = LOAD4W_S(base, -s2o2, S);                                                 \
                mn                    = _mm_min_epu8(mn, tap);                                                    \
                mx                    = _mm_max_epu8(mx, tap);                                                    \
                cb                    = _mm_add_epi8(cb, constrain8x16_sse4(tap, row, secthr, secdamp, secmask)); \
                accum_taps2_sse4(&sum0,                                                                           \
                                 &sum1,                                                                           \
                                 _mm_set1_epi16((short)(((sec_taps[1] & 0xFF) << 8) | (sec_taps[0] & 0xFF))),     \
                                 ca,                                                                              \
                                 cb);                                                                             \
            }                                                                                                     \
            /* p01 holds rows 0,1 as 8 bytes; p23 holds rows 2,3 */                                               \
            const __m128i p01 = cdef_finalize_row_sse4(sum0, row, mn, mx);                                        \
            const __m128i p23 = cdef_finalize_row_sse4(                                                           \
                sum1, _mm_srli_si128(row, 8), _mm_srli_si128(mn, 8), _mm_srli_si128(mx, 8));                      \
            const __m128i pk[4] = {p01, _mm_srli_si128(p01, 4), p23, _mm_srli_si128(p23, 4)};                     \
            for (int k = 0; k < nrows; k++) {                                                                     \
                *(int*)(dst + (i + k * (S)) * dstride) = _mm_cvtsi128_si32(pk[k]);                                \
            }                                                                                                     \
        }                                                                                                         \
    }

DEFINE_4XN_SSE4(cdef_4xn_native_s1_sse4, 1)
DEFINE_4XN_SSE4(cdef_4xn_native_s2_sse4, 2)
DEFINE_4XN_SSE4(cdef_4xn_native_s4_sse4, 4)
#undef DEFINE_4XN_SSE4

void svt_cdef_filter_block_8bit_sse4_1(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                       int32_t sec_strength, int32_t dir, int32_t damping, int32_t bsize,
                                       int32_t coeff_shift, uint8_t subsampling_factor) {
    if (bsize == BLOCK_8X8) {
        if (subsampling_factor == 1) {
            cdef_8xn_native_s1_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        } else if (subsampling_factor == 2) {
            cdef_8xn_native_s2_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        } else {
            cdef_8xn_native_s4_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        }
    } else if (bsize == BLOCK_4X8) {
        if (subsampling_factor == 1) {
            cdef_4xn_native_s1_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        } else if (subsampling_factor == 2) {
            cdef_4xn_native_s2_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        } else {
            cdef_4xn_native_s4_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 8);
        }
    } else if (bsize == BLOCK_8X4) {
        if (subsampling_factor == 1) {
            cdef_8xn_native_s1_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
        } else if (subsampling_factor == 2) {
            cdef_8xn_native_s2_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
        } else {
            cdef_8xn_native_s4_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
        }
    } else {
        cdef_4xn_native_s1_sse4(dst, dstride, in, pri_strength, sec_strength, dir, damping, coeff_shift, 4);
    }
}

// Boundary-aware kernels. Same math as the interior kernels, but every tap is
// masked per lane by geometry so off-frame taps drop out of the sum, the max
// (off lanes -> 0) and the min (off lanes -> 255). Bit-exact with
// svt_cdef_filter_block_8bit_bounded_c.

// Column availability for an 8-wide row, replicated into both row slots.
static INLINE __m128i bnd_col16_w8(const int dc, const int edge_left, const int edge_right) {
    if (!edge_left && !edge_right) {
        return _mm_set1_epi8((char)0xFF);
    }
    uint8_t m[8];
    for (int c = 0; c < 8; c++) {
        const int off = (edge_left && (c + dc) < 0) || (edge_right && (c + dc) >= 8);
        m[c]          = off ? 0x00 : 0xFF;
    }
    int64_t w;
    memcpy(&w, m, 8);
    return _mm_set1_epi64x(w);
}

// Column availability for a 4-wide row, replicated into all four row slots.
static INLINE __m128i bnd_col16_w4(const int dc, const int edge_left, const int edge_right) {
    if (!edge_left && !edge_right) {
        return _mm_set1_epi8((char)0xFF);
    }
    uint8_t m[4];
    for (int c = 0; c < 4; c++) {
        const int off = (edge_left && (c + dc) < 0) || (edge_right && (c + dc) >= 4);
        m[c]          = off ? 0x00 : 0xFF;
    }
    int32_t w;
    memcpy(&w, m, 4);
    return _mm_set1_epi32(w);
}

// Row availability for the two 8-byte row slots.
static INLINE __m128i bnd_row16_w8(const int i, const int sub, const int dr, const int edge_top, const int edge_bottom,
                                   const int rows, const int nrows) {
    int64_t v[2];
    for (int k = 0; k < 2; k++) {
        const int r  = i + (k < nrows ? k : nrows - 1) * sub;
        const int ok = !((edge_top && (r + dr) < 0) || (edge_bottom && (r + dr) >= rows));
        v[k]         = ok ? (int64_t)0xFFFFFFFFFFFFFFFFULL : 0;
    }
    return _mm_set_epi64x(v[1], v[0]);
}

// Row availability for the four 4-byte row slots.
static INLINE __m128i bnd_row16_w4(const int i, const int sub, const int dr, const int edge_top, const int edge_bottom,
                                   const int rows, const int nrows) {
    int32_t v[4];
    for (int k = 0; k < 4; k++) {
        const int r  = i + (k < nrows ? k : nrows - 1) * sub;
        const int ok = !((edge_top && (r + dr) < 0) || (edge_bottom && (r + dr) >= rows));
        v[k]         = ok ? (int32_t)0xFFFFFFFF : 0;
    }
    return _mm_setr_epi32(v[0], v[1], v[2], v[3]);
}

typedef struct {
    ptrdiff_t off[6];
    int       dr[6];
    __m128i   colp[6], coln[6];
    int       needp[6], needn[6], rsensp[6], rsensn[6];
} BndTapCfg128;

static INLINE void bnd_setup128(BndTapCfg128* cfg, const int dir, const int edge_top, const int edge_left,
                                const int edge_bottom, const int edge_right, const int width) {
    const int idx[6] = {dir, dir, dir + 2, dir + 2, dir - 2, dir - 2};
    const int sel[6] = {0, 1, 0, 1, 0, 1};
    for (int b = 0; b < 6; b++) {
        cfg->off[b]  = svt_aom_eb_cdef_directions[idx[b]][sel[b]];
        cfg->dr[b]   = svt_aom_eb_cdef_directions_rc[idx[b]][sel[b]][0];
        const int dc = svt_aom_eb_cdef_directions_rc[idx[b]][sel[b]][1];
        if (width == 8) {
            cfg->colp[b] = bnd_col16_w8(dc, edge_left, edge_right);
            cfg->coln[b] = bnd_col16_w8(-dc, edge_left, edge_right);
        } else {
            cfg->colp[b] = bnd_col16_w4(dc, edge_left, edge_right);
            cfg->coln[b] = bnd_col16_w4(-dc, edge_left, edge_right);
        }
        const int csp  = (edge_left && dc < 0) || (edge_right && dc > 0);
        const int csn  = (edge_left && -dc < 0) || (edge_right && -dc > 0);
        cfg->rsensp[b] = (edge_top && cfg->dr[b] < 0) || (edge_bottom && cfg->dr[b] > 0);
        cfg->rsensn[b] = (edge_top && -cfg->dr[b] < 0) || (edge_bottom && -cfg->dr[b] > 0);
        cfg->needp[b]  = csp || cfg->rsensp[b];
        cfg->needn[b]  = csn || cfg->rsensn[b];
    }
}

// Widening int8 -> int16 multiply-accumulate of one tap. The bounded path sums
// 2 or 4 masked taps before scaling, so the taps are not paired into a maddubs.
static INLINE void accum_tap_sse4(__m128i* acc0, __m128i* acc1, const __m128i tap16, const __m128i csum) {
    *acc0 = _mm_add_epi16(*acc0, _mm_mullo_epi16(tap16, _mm_cvtepi8_epi16(csum)));
    *acc1 = _mm_add_epi16(*acc1, _mm_mullo_epi16(tap16, _mm_cvtepi8_epi16(_mm_srli_si128(csum, 8))));
}

static void cdef_filter_block_8xn_8_bounded_sse4(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                                 int32_t sec_strength, int32_t dir, int32_t damping,
                                                 int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor,
                                                 int edge_top, int edge_left, int edge_bottom, int edge_right) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const int     rows        = height;
    const int     sub         = subsampling_factor;

    BndTapCfg128 cfg;
    bnd_setup128(&cfg, dir, edge_top, edge_left, edge_bottom, edge_right, 8);

    const __m128i prithr  = _mm_set1_epi8((char)pri_strength);
    const __m128i secthr  = _mm_set1_epi8((char)sec_strength);
    const __m128i pridamp = _mm_cvtsi32_si128(pri_damping);
    const __m128i secdamp = _mm_cvtsi32_si128(sec_damping);
    const __m128i primask = byte_shift_mask_128(pri_damping);
    const __m128i secmask = byte_shift_mask_128(sec_damping);
    const __m128i ones    = _mm_set1_epi8((char)0xFF);

    for (int i = 0; i < height; i += 2 * sub) {
        const int nrows = ((height - i) + sub - 1) / sub;
        const int nr    = nrows > 2 ? 2 : nrows;

        const uint8_t* rp[2];
        for (int k = 0; k < 2; k++) {
            rp[k] = in + (i + (k < nr ? k : nr - 1) * sub) * CDEF_BSTRIDE;
        }

#define BND_LOAD8(OFF)                                                   \
    _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)(rp[0] + (OFF))), \
                       _mm_loadl_epi64((const __m128i*)(rp[1] + (OFF))))

        const __m128i row = BND_LOAD8(0);
        __m128i       mn = row, mx = row, tap, av, c0, c1, c2, c3, csum;
        __m128i       acc0 = _mm_setzero_si128();
        __m128i       acc1 = _mm_setzero_si128();

#define BND_TAP8(B, SGN, THR, DAMP, MASK, CACC)                                                                          \
    do {                                                                                                                 \
        const int _need = (SGN) > 0 ? cfg.needp[B] : cfg.needn[B];                                                       \
        const int _rs   = (SGN) > 0 ? cfg.rsensp[B] : cfg.rsensn[B];                                                     \
        tap             = BND_LOAD8((SGN) * cfg.off[B]);                                                                 \
        if (!_need) {                                                                                                    \
            mn     = _mm_min_epu8(mn, tap);                                                                              \
            mx     = _mm_max_epu8(mx, tap);                                                                              \
            (CACC) = constrain8x16_sse4(tap, row, (THR), (DAMP), (MASK));                                                \
        } else {                                                                                                         \
            const __m128i _col = (SGN) > 0 ? cfg.colp[B] : cfg.coln[B];                                                  \
            av     = _rs ? _mm_and_si128(_col, bnd_row16_w8(i, sub, (SGN) * cfg.dr[B], edge_top, edge_bottom, rows, nr)) \
                         : _col;                                                                                         \
            mn     = _mm_min_epu8(mn, _mm_or_si128(tap, _mm_xor_si128(av, ones)));                                       \
            mx     = _mm_max_epu8(mx, _mm_and_si128(tap, av));                                                           \
            (CACC) = _mm_and_si128(constrain8x16_sse4(tap, row, (THR), (DAMP), (MASK)), av);                             \
        }                                                                                                                \
    } while (0)

        if (pri_strength) {
            BND_TAP8(0, 1, prithr, pridamp, primask, c0);
            BND_TAP8(0, -1, prithr, pridamp, primask, c1);
            csum = _mm_add_epi8(c0, c1);
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)pri_taps[0]), csum);
            BND_TAP8(1, 1, prithr, pridamp, primask, c0);
            BND_TAP8(1, -1, prithr, pridamp, primask, c1);
            csum = _mm_add_epi8(c0, c1);
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)pri_taps[1]), csum);
        }
        if (sec_strength) {
            BND_TAP8(2, 1, secthr, secdamp, secmask, c0);
            BND_TAP8(2, -1, secthr, secdamp, secmask, c1);
            BND_TAP8(4, 1, secthr, secdamp, secmask, c2);
            BND_TAP8(4, -1, secthr, secdamp, secmask, c3);
            csum = _mm_add_epi8(_mm_add_epi8(c0, c1), _mm_add_epi8(c2, c3));
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)sec_taps[0]), csum);
            BND_TAP8(3, 1, secthr, secdamp, secmask, c0);
            BND_TAP8(3, -1, secthr, secdamp, secmask, c1);
            BND_TAP8(5, 1, secthr, secdamp, secmask, c2);
            BND_TAP8(5, -1, secthr, secdamp, secmask, c3);
            csum = _mm_add_epi8(_mm_add_epi8(c0, c1), _mm_add_epi8(c2, c3));
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)sec_taps[1]), csum);
        }
#undef BND_TAP8
#undef BND_LOAD8

        const __m128i p0 = cdef_finalize_row_sse4(acc0, row, mn, mx);
        const __m128i p1 = cdef_finalize_row_sse4(
            acc1, _mm_srli_si128(row, 8), _mm_srli_si128(mn, 8), _mm_srli_si128(mx, 8));
        const __m128i out[2] = {p0, p1};
        for (int k = 0; k < nr; k++) {
            _mm_storel_epi64((__m128i*)(dst + (i + k * sub) * dstride), out[k]);
        }
    }
}

// 4-wide bounded: one register holds all four rows.
static void cdef_filter_block_4xn_8_bounded_sse4(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                                 int32_t sec_strength, int32_t dir, int32_t damping,
                                                 int32_t coeff_shift, uint8_t height, uint8_t subsampling_factor,
                                                 int edge_top, int edge_left, int edge_bottom, int edge_right) {
    const int*    pri_taps    = svt_aom_eb_cdef_pri_taps[(pri_strength >> coeff_shift) & 1];
    const int*    sec_taps    = svt_aom_eb_cdef_sec_taps[(pri_strength >> coeff_shift) & 1];
    const int32_t pri_damping = pri_strength ? AOMMAX(0, damping - get_msb(pri_strength)) : 0;
    const int32_t sec_damping = sec_strength ? AOMMAX(0, damping - get_msb(sec_strength)) : 0;
    const int     rows        = height;
    const int     sub         = subsampling_factor;

    BndTapCfg128 cfg;
    bnd_setup128(&cfg, dir, edge_top, edge_left, edge_bottom, edge_right, 4);

    const __m128i prithr  = _mm_set1_epi8((char)pri_strength);
    const __m128i secthr  = _mm_set1_epi8((char)sec_strength);
    const __m128i pridamp = _mm_cvtsi32_si128(pri_damping);
    const __m128i secdamp = _mm_cvtsi32_si128(sec_damping);
    const __m128i primask = byte_shift_mask_128(pri_damping);
    const __m128i secmask = byte_shift_mask_128(sec_damping);
    const __m128i ones    = _mm_set1_epi8((char)0xFF);

    for (int i = 0; i < height; i += 4 * sub) {
        const int nrows = ((height - i) + sub - 1) / sub;
        const int nr    = nrows > 4 ? 4 : nrows;

        const uint8_t* rp[4];
        for (int k = 0; k < 4; k++) {
            rp[k] = in + (i + (k < nr ? k : nr - 1) * sub) * CDEF_BSTRIDE;
        }

#define BND_LOAD4(OFF)                           \
    _mm_setr_epi32(*(const int*)(rp[0] + (OFF)), \
                   *(const int*)(rp[1] + (OFF)), \
                   *(const int*)(rp[2] + (OFF)), \
                   *(const int*)(rp[3] + (OFF)))

        const __m128i row = BND_LOAD4(0);
        __m128i       mn = row, mx = row, tap, av, c0, c1, c2, c3, csum;
        __m128i       acc0 = _mm_setzero_si128();
        __m128i       acc1 = _mm_setzero_si128();

#define BND_TAP4(B, SGN, THR, DAMP, MASK, CACC)                                                                          \
    do {                                                                                                                 \
        const int _need = (SGN) > 0 ? cfg.needp[B] : cfg.needn[B];                                                       \
        const int _rs   = (SGN) > 0 ? cfg.rsensp[B] : cfg.rsensn[B];                                                     \
        tap             = BND_LOAD4((SGN) * cfg.off[B]);                                                                 \
        if (!_need) {                                                                                                    \
            mn     = _mm_min_epu8(mn, tap);                                                                              \
            mx     = _mm_max_epu8(mx, tap);                                                                              \
            (CACC) = constrain8x16_sse4(tap, row, (THR), (DAMP), (MASK));                                                \
        } else {                                                                                                         \
            const __m128i _col = (SGN) > 0 ? cfg.colp[B] : cfg.coln[B];                                                  \
            av     = _rs ? _mm_and_si128(_col, bnd_row16_w4(i, sub, (SGN) * cfg.dr[B], edge_top, edge_bottom, rows, nr)) \
                         : _col;                                                                                         \
            mn     = _mm_min_epu8(mn, _mm_or_si128(tap, _mm_xor_si128(av, ones)));                                       \
            mx     = _mm_max_epu8(mx, _mm_and_si128(tap, av));                                                           \
            (CACC) = _mm_and_si128(constrain8x16_sse4(tap, row, (THR), (DAMP), (MASK)), av);                             \
        }                                                                                                                \
    } while (0)

        if (pri_strength) {
            BND_TAP4(0, 1, prithr, pridamp, primask, c0);
            BND_TAP4(0, -1, prithr, pridamp, primask, c1);
            csum = _mm_add_epi8(c0, c1);
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)pri_taps[0]), csum);
            BND_TAP4(1, 1, prithr, pridamp, primask, c0);
            BND_TAP4(1, -1, prithr, pridamp, primask, c1);
            csum = _mm_add_epi8(c0, c1);
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)pri_taps[1]), csum);
        }
        if (sec_strength) {
            BND_TAP4(2, 1, secthr, secdamp, secmask, c0);
            BND_TAP4(2, -1, secthr, secdamp, secmask, c1);
            BND_TAP4(4, 1, secthr, secdamp, secmask, c2);
            BND_TAP4(4, -1, secthr, secdamp, secmask, c3);
            csum = _mm_add_epi8(_mm_add_epi8(c0, c1), _mm_add_epi8(c2, c3));
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)sec_taps[0]), csum);
            BND_TAP4(3, 1, secthr, secdamp, secmask, c0);
            BND_TAP4(3, -1, secthr, secdamp, secmask, c1);
            BND_TAP4(5, 1, secthr, secdamp, secmask, c2);
            BND_TAP4(5, -1, secthr, secdamp, secmask, c3);
            csum = _mm_add_epi8(_mm_add_epi8(c0, c1), _mm_add_epi8(c2, c3));
            accum_tap_sse4(&acc0, &acc1, _mm_set1_epi16((short)sec_taps[1]), csum);
        }
#undef BND_TAP4
#undef BND_LOAD4

        const __m128i g01 = cdef_finalize_row_sse4(acc0, row, mn, mx);
        const __m128i g23 = cdef_finalize_row_sse4(
            acc1, _mm_srli_si128(row, 8), _mm_srli_si128(mn, 8), _mm_srli_si128(mx, 8));
        const int o[4] = {_mm_cvtsi128_si32(g01),
                          _mm_cvtsi128_si32(_mm_srli_si128(g01, 4)),
                          _mm_cvtsi128_si32(g23),
                          _mm_cvtsi128_si32(_mm_srli_si128(g23, 4))};
        for (int k = 0; k < nr; k++) {
            *(int*)(dst + (i + k * sub) * dstride) = o[k];
        }
    }
}

void svt_cdef_filter_block_8bit_bounded_sse4_1(uint8_t* dst, int32_t dstride, const uint8_t* in, int32_t pri_strength,
                                               int32_t sec_strength, int32_t dir, int32_t damping, int32_t bsize,
                                               int32_t coeff_shift, uint8_t subsampling_factor, int edge_top,
                                               int edge_left, int edge_bottom, int edge_right) {
    if (bsize == BLOCK_8X8) {
        cdef_filter_block_8xn_8_bounded_sse4(dst,
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
        cdef_filter_block_4xn_8_bounded_sse4(dst,
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
        cdef_filter_block_8xn_8_bounded_sse4(dst,
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
        cdef_filter_block_4xn_8_bounded_sse4(dst,
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
