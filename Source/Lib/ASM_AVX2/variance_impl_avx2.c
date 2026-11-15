/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <immintrin.h> // AVX2
#include "aom_dsp_rtcd.h"

//#include "aom_ports/mem.h"

/* clang-format off */
DECLARE_ALIGNED(32, static const uint8_t, bilinear_filters_avx2[512]) = {
  16,  0, 16,  0, 16,  0, 16,  0, 16,  0, 16,  0, 16,  0, 16,  0,
  16,  0, 16,  0, 16,  0, 16,  0, 16,  0, 16,  0, 16,  0, 16,  0,
  14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2,
  14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2,
  12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4,
  12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4,
  10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6,
  10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6,
   8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
   8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
   6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,
   6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,  6, 10,
   4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,
   4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,  4, 12,
   2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,
   2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,  2, 14,
};
/* clang-format on */

#define FILTER_SRC(filter)                                 \
    /* filter the source */                                \
    exp_src_lo = _mm256_maddubs_epi16(exp_src_lo, filter); \
    exp_src_hi = _mm256_maddubs_epi16(exp_src_hi, filter); \
                                                           \
    /* add 8 to source */                                  \
    exp_src_lo = _mm256_add_epi16(exp_src_lo, pw8);        \
    exp_src_hi = _mm256_add_epi16(exp_src_hi, pw8);        \
                                                           \
    /* divide source by 16 */                              \
    exp_src_lo = _mm256_srai_epi16(exp_src_lo, 4);         \
    exp_src_hi = _mm256_srai_epi16(exp_src_hi, 4);

#define MERGE_WITH_SRC(src_reg, reg)                 \
    exp_src_lo = _mm256_unpacklo_epi8(src_reg, reg); \
    exp_src_hi = _mm256_unpackhi_epi8(src_reg, reg);

#define LOAD_SRC_DST                                     \
    /* load source and destination */                    \
    src_reg = _mm256_loadu_si256((__m256i const*)(src)); \
    dst_reg = _mm256_loadu_si256((__m256i const*)(dst));

#define AVG_NEXT_SRC(src_reg, size_stride)                                  \
    src_next_reg = _mm256_loadu_si256((__m256i const*)(src + size_stride)); \
    /* average between current and next stride source */                    \
    src_reg = _mm256_avg_epu8(src_reg, src_next_reg);

#define MERGE_NEXT_SRC(src_reg, size_stride)                                \
    src_next_reg = _mm256_loadu_si256((__m256i const*)(src + size_stride)); \
    MERGE_WITH_SRC(src_reg, src_next_reg)

#define CALC_SUM_SSE_INSIDE_LOOP                            \
    /* expand each byte to 2 bytes */                       \
    exp_dst_lo = _mm256_unpacklo_epi8(dst_reg, zero_reg);   \
    exp_dst_hi = _mm256_unpackhi_epi8(dst_reg, zero_reg);   \
    /* source - dest */                                     \
    exp_src_lo = _mm256_sub_epi16(exp_src_lo, exp_dst_lo);  \
    exp_src_hi = _mm256_sub_epi16(exp_src_hi, exp_dst_hi);  \
    /* caculate sum */                                      \
    sum_reg    = _mm256_add_epi16(sum_reg, exp_src_lo);     \
    exp_src_lo = _mm256_madd_epi16(exp_src_lo, exp_src_lo); \
    sum_reg    = _mm256_add_epi16(sum_reg, exp_src_hi);     \
    exp_src_hi = _mm256_madd_epi16(exp_src_hi, exp_src_hi); \
    /* calculate sse */                                     \
    sse_reg = _mm256_add_epi32(sse_reg, exp_src_lo);        \
    sse_reg = _mm256_add_epi32(sse_reg, exp_src_hi);

// final calculation to sum and sse
#define CALC_SUM_AND_SSE                                                \
    res_cmp    = _mm256_cmpgt_epi16(zero_reg, sum_reg);                 \
    sse_reg_hi = _mm256_srli_si256(sse_reg, 8);                         \
    sum_reg_lo = _mm256_unpacklo_epi16(sum_reg, res_cmp);               \
    sum_reg_hi = _mm256_unpackhi_epi16(sum_reg, res_cmp);               \
    sse_reg    = _mm256_add_epi32(sse_reg, sse_reg_hi);                 \
    sum_reg    = _mm256_add_epi32(sum_reg_lo, sum_reg_hi);              \
                                                                        \
    sse_reg_hi = _mm256_srli_si256(sse_reg, 4);                         \
    sum_reg_hi = _mm256_srli_si256(sum_reg, 8);                         \
                                                                        \
    sse_reg      = _mm256_add_epi32(sse_reg, sse_reg_hi);               \
    sum_reg      = _mm256_add_epi32(sum_reg, sum_reg_hi);               \
    *((int*)sse) = _mm_cvtsi128_si32(_mm256_castsi256_si128(sse_reg)) + \
        _mm_cvtsi128_si32(_mm256_extractf128_si256(sse_reg, 1));        \
    sum_reg_hi = _mm256_srli_si256(sum_reg, 4);                         \
    sum_reg    = _mm256_add_epi32(sum_reg, sum_reg_hi);                 \
    sum = _mm_cvtsi128_si32(_mm256_castsi256_si128(sum_reg)) + _mm_cvtsi128_si32(_mm256_extractf128_si256(sum_reg, 1));

// Functions related to sub pixel variance width 16
#define LOAD_SRC_DST_INSERT(src_stride, dst_stride)                                                                  \
    /* load source and destination of 2 rows and insert*/                                                            \
    src_reg = _mm256_inserti128_si256(                                                                               \
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src))), _mm_loadu_si128((__m128i*)(src + src_stride)), 1); \
    dst_reg = _mm256_inserti128_si256(                                                                               \
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(dst))), _mm_loadu_si128((__m128i*)(dst + dst_stride)), 1);

#define AVG_NEXT_SRC_INSERT(src_reg, size_stride)                                                                  \
    src_next_reg = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src + size_stride))), \
                                           _mm_loadu_si128((__m128i*)(src + (size_stride << 1))),                  \
                                           1);                                                                     \
    /* average between current and next stride source */                                                           \
    src_reg = _mm256_avg_epu8(src_reg, src_next_reg);

#define MERGE_NEXT_SRC_INSERT(src_reg, size_stride)                                                                \
    src_next_reg = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src + size_stride))), \
                                           _mm_loadu_si128((__m128i*)(src + (src_stride + size_stride))),          \
                                           1);                                                                     \
    MERGE_WITH_SRC(src_reg, src_next_reg)

#define LOAD_SRC_NEXT_BYTE_INSERT                                                                                    \
    /* load source and another source from next row   */                                                             \
    src_reg = _mm256_inserti128_si256(                                                                               \
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src))), _mm_loadu_si128((__m128i*)(src + src_stride)), 1); \
    /* load source and next row source from 1 byte onwards   */                                                      \
    src_next_reg = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(src + 1))),             \
                                           _mm_loadu_si128((__m128i*)(src + src_stride + 1)),                        \
                                           1);

#define LOAD_DST_INSERT                \
    dst_reg = _mm256_inserti128_si256( \
        _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(dst))), _mm_loadu_si128((__m128i*)(dst + dst_stride)), 1);

#define LOAD_SRC_MERGE_128BIT(filter)                                \
    __m128i src_reg_0     = _mm_loadu_si128((__m128i*)(src));        \
    __m128i src_reg_1     = _mm_loadu_si128((__m128i*)(src + 1));    \
    __m128i src_lo        = _mm_unpacklo_epi8(src_reg_0, src_reg_1); \
    __m128i src_hi        = _mm_unpackhi_epi8(src_reg_0, src_reg_1); \
    __m128i filter_128bit = _mm256_castsi256_si128(filter);          \
    __m128i pw8_128bit    = _mm256_castsi256_si128(pw8);

#define FILTER_SRC_128BIT(filter)               \
    /* filter the source */                     \
    src_lo = _mm_maddubs_epi16(src_lo, filter); \
    src_hi = _mm_maddubs_epi16(src_hi, filter); \
                                                \
    /* add 8 to source */                       \
    src_lo = _mm_add_epi16(src_lo, pw8_128bit); \
    src_hi = _mm_add_epi16(src_hi, pw8_128bit); \
                                                \
    /* divide source by 16 */                   \
    src_lo = _mm_srai_epi16(src_lo, 4);         \
    src_hi = _mm_srai_epi16(src_hi, 4);

// Horizontal bilinear filter of 32 pixels. Output bytes are de-interleaved per
// 128-bit lane: the 8 even-position pixels followed by the 8 odd-position ones.
static inline __m256i bilinear_h_32_avx2(const uint8_t* src, const __m256i xfilter, const __m256i pw8) {
    __m256i even = _mm256_maddubs_epi16(_mm256_loadu_si256((__m256i const*)src), xfilter);
    __m256i odd  = _mm256_maddubs_epi16(_mm256_loadu_si256((__m256i const*)(src + 1)), xfilter);
    even         = _mm256_srai_epi16(_mm256_add_epi16(even, pw8), 4);
    odd          = _mm256_srai_epi16(_mm256_add_epi16(odd, pw8), 4);
    return _mm256_packus_epi16(even, odd);
}

// Same as bilinear_h_32_avx2() for one row of 16 pixels.
static inline __m128i bilinear_h_16_sse(const uint8_t* src, const __m256i xfilter, const __m256i pw8) {
    const __m128i xfilter_128 = _mm256_castsi256_si128(xfilter);
    const __m128i pw8_128     = _mm256_castsi256_si128(pw8);
    __m128i       even        = _mm_maddubs_epi16(_mm_loadu_si128((__m128i const*)src), xfilter_128);
    __m128i       odd         = _mm_maddubs_epi16(_mm_loadu_si128((__m128i const*)(src + 1)), xfilter_128);
    even                      = _mm_srai_epi16(_mm_add_epi16(even, pw8_128), 4);
    odd                       = _mm_srai_epi16(_mm_add_epi16(odd, pw8_128), 4);
    return _mm_packus_epi16(even, odd);
}

static inline __m256i load_16x2_avx2(const uint8_t* src, int stride) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i const*)src)),
                                   _mm_loadu_si128((__m128i const*)(src + stride)),
                                   1);
}

// Two rows of 16 pixels, one per 128-bit lane, laid out as bilinear_h_32_avx2().
static inline __m256i bilinear_h_16x2_avx2(const uint8_t* src, int stride, const __m256i xfilter, const __m256i pw8) {
    __m256i even = _mm256_maddubs_epi16(load_16x2_avx2(src, stride), xfilter);
    __m256i odd  = _mm256_maddubs_epi16(load_16x2_avx2(src + 1, stride), xfilter);
    even         = _mm256_srai_epi16(_mm256_add_epi16(even, pw8), 4);
    odd          = _mm256_srai_epi16(_mm256_add_epi16(odd, pw8), 4);
    return _mm256_packus_epi16(even, odd);
}

// Vertical bilinear filter between horizontally filtered rows 'top' and 'bot'
// (de-interleaved layout), then accumulate the difference against 'dst', whose
// even and odd bytes are split to match.
static inline void bilinear_v_accum_avx2(const __m256i top, const __m256i bot, const __m256i dst, const __m256i yfilter,
                                         const __m256i pw8, const __m256i mask_00ff, __m256i* sum_reg,
                                         __m256i* sse_reg) {
    __m256i v_even          = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(top, bot), yfilter);
    __m256i v_odd           = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(top, bot), yfilter);
    v_even                  = _mm256_srai_epi16(_mm256_add_epi16(v_even, pw8), 4);
    v_odd                   = _mm256_srai_epi16(_mm256_add_epi16(v_odd, pw8), 4);
    const __m256i diff_even = _mm256_sub_epi16(v_even, _mm256_and_si256(dst, mask_00ff));
    const __m256i diff_odd  = _mm256_sub_epi16(v_odd, _mm256_srli_epi16(dst, 8));
    *sum_reg                = _mm256_add_epi16(*sum_reg, _mm256_add_epi16(diff_even, diff_odd));
    *sse_reg                = _mm256_add_epi32(
        *sse_reg, _mm256_add_epi32(_mm256_madd_epi16(diff_even, diff_even), _mm256_madd_epi16(diff_odd, diff_odd)));
}

static inline int bilinear_sum_sse_avx2(__m256i sum_reg, __m256i sse_reg, unsigned int* sse) {
    sum_reg = _mm256_madd_epi16(sum_reg, _mm256_set1_epi16(1));
    sse_reg = _mm256_add_epi32(sse_reg, _mm256_srli_si256(sse_reg, 8));
    sum_reg = _mm256_add_epi32(sum_reg, _mm256_srli_si256(sum_reg, 8));
    sse_reg = _mm256_add_epi32(sse_reg, _mm256_srli_si256(sse_reg, 4));
    sum_reg = _mm256_add_epi32(sum_reg, _mm256_srli_si256(sum_reg, 4));
    *sse = _mm_cvtsi128_si32(_mm256_castsi256_si128(sse_reg)) + _mm_cvtsi128_si32(_mm256_extracti128_si256(sse_reg, 1));
    const int sum = _mm_cvtsi128_si32(_mm256_castsi256_si128(sum_reg)) +
        _mm_cvtsi128_si32(_mm256_extracti128_si256(sum_reg, 1));
    _mm256_zeroupper();
    return sum;
}

unsigned int svt_aom_sub_pixel_variance32xh_avx2(const uint8_t* src, int src_stride, int x_offset, int y_offset,
                                                 const uint8_t* dst, int dst_stride, int height, unsigned int* sse) {
    __m256i src_reg, dst_reg, exp_src_lo, exp_src_hi, exp_dst_lo, exp_dst_hi;
    __m256i sse_reg, sum_reg, sse_reg_hi, res_cmp, sum_reg_lo, sum_reg_hi;
    __m256i zero_reg;
    int     i, sum;
    sum_reg  = _mm256_set1_epi16(0);
    sse_reg  = _mm256_set1_epi16(0);
    zero_reg = _mm256_set1_epi16(0);

    // x_offset = 0 and y_offset = 0
    if (x_offset == 0) {
        if (y_offset == 0) {
            for (i = 0; i < height; i++) {
                LOAD_SRC_DST
                // expend each byte to 2 bytes
                MERGE_WITH_SRC(src_reg, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src += src_stride;
                dst += dst_stride;
            }
            // x_offset = 0 and y_offset = 4
        } else if (y_offset == 4) {
            __m256i src_next_reg;
            for (i = 0; i < height; i++) {
                LOAD_SRC_DST
                AVG_NEXT_SRC(src_reg, src_stride)
                // expend each byte to 2 bytes
                MERGE_WITH_SRC(src_reg, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src += src_stride;
                dst += dst_stride;
            }
            // x_offset = 0 and y_offset = bilin interpolation
        } else {
            __m256i filter, pw8, src_next_reg;

            y_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + y_offset));
            pw8    = _mm256_set1_epi16(8);
            for (i = 0; i < height; i++) {
                LOAD_SRC_DST
                MERGE_NEXT_SRC(src_reg, src_stride)
                FILTER_SRC(filter)
                CALC_SUM_SSE_INSIDE_LOOP
                src += src_stride;
                dst += dst_stride;
            }
        }
        // x_offset = 4  and y_offset = 0
    } else if (x_offset == 4) {
        if (y_offset == 0) {
            __m256i src_next_reg;
            for (i = 0; i < height; i++) {
                LOAD_SRC_DST
                AVG_NEXT_SRC(src_reg, 1)
                // expand each byte to 2 bytes
                MERGE_WITH_SRC(src_reg, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src += src_stride;
                dst += dst_stride;
            }
            // x_offset = 4  and y_offset = 4
        } else if (y_offset == 4) {
            __m256i src_next_reg, src_avg;
            // load source and another source starting from the next
            // following byte
            src_reg = _mm256_loadu_si256((__m256i const*)(src));
            AVG_NEXT_SRC(src_reg, 1)
            for (i = 0; i < height; i++) {
                src_avg = src_reg;
                src += src_stride;
                LOAD_SRC_DST
                AVG_NEXT_SRC(src_reg, 1)
                // average between previous average to current average
                src_avg = _mm256_avg_epu8(src_avg, src_reg);
                // expand each byte to 2 bytes
                MERGE_WITH_SRC(src_avg, zero_reg)
                // save current source average
                CALC_SUM_SSE_INSIDE_LOOP
                dst += dst_stride;
            }
            // x_offset = 4  and y_offset = bilin interpolation
        } else {
            __m256i filter, pw8, src_next_reg, src_avg;
            y_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + y_offset));
            pw8    = _mm256_set1_epi16(8);
            // load source and another source starting from the next
            // following byte
            src_reg = _mm256_loadu_si256((__m256i const*)(src));
            AVG_NEXT_SRC(src_reg, 1)
            for (i = 0; i < height; i++) {
                // save current source average
                src_avg = src_reg;
                src += src_stride;
                LOAD_SRC_DST
                AVG_NEXT_SRC(src_reg, 1)
                MERGE_WITH_SRC(src_avg, src_reg)
                FILTER_SRC(filter)
                CALC_SUM_SSE_INSIDE_LOOP
                dst += dst_stride;
            }
        }
        // x_offset = bilin interpolation and y_offset = 0
    } else {
        if (y_offset == 0) {
            __m256i filter, pw8, src_next_reg;
            x_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + x_offset));
            pw8    = _mm256_set1_epi16(8);
            for (i = 0; i < height; i++) {
                LOAD_SRC_DST
                MERGE_NEXT_SRC(src_reg, 1)
                FILTER_SRC(filter)
                CALC_SUM_SSE_INSIDE_LOOP
                src += src_stride;
                dst += dst_stride;
            }
            // x_offset = bilin interpolation and y_offset = 4
        } else if (y_offset == 4) {
            __m256i filter, pw8, src_next_reg, src_pack;
            x_offset <<= 5;
            filter  = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + x_offset));
            pw8     = _mm256_set1_epi16(8);
            src_reg = _mm256_loadu_si256((__m256i const*)(src));
            MERGE_NEXT_SRC(src_reg, 1)
            FILTER_SRC(filter)
            // convert each 16 bit to 8 bit to each low and high lane source
            src_pack = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
            for (i = 0; i < height; i++) {
                src += src_stride;
                LOAD_SRC_DST
                MERGE_NEXT_SRC(src_reg, 1)
                FILTER_SRC(filter)
                src_reg = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
                // average between previous pack to the current
                src_pack = _mm256_avg_epu8(src_pack, src_reg);
                MERGE_WITH_SRC(src_pack, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src_pack = src_reg;
                dst += dst_stride;
            }
            // x_offset = bilin interpolation and y_offset = bilin interpolation
        } else {
            // Two output rows per iteration, with the horizontal pass of the
            // next rows interleaved with the vertical pass of the current ones.
            const __m256i xfilter   = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + (x_offset << 5)));
            const __m256i yfilter   = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + (y_offset << 5)));
            const __m256i pw8       = _mm256_set1_epi16(8);
            const __m256i mask_00ff = _mm256_set1_epi16(0x00ff);
            __m256i       p0        = bilinear_h_32_avx2(src, xfilter, pw8);
            __m256i       p1        = bilinear_h_32_avx2(src + src_stride, xfilter, pw8);
            __m256i       p2        = bilinear_h_32_avx2(src + 2 * src_stride, xfilter, pw8);
            src += 3 * src_stride;
            for (i = 0; i < height - 2; i += 2) {
                const __m256i p3 = bilinear_h_32_avx2(src, xfilter, pw8);
                const __m256i p4 = bilinear_h_32_avx2(src + src_stride, xfilter, pw8);
                src += src_stride << 1;
                bilinear_v_accum_avx2(
                    p0, p1, _mm256_loadu_si256((__m256i const*)dst), yfilter, pw8, mask_00ff, &sum_reg, &sse_reg);
                bilinear_v_accum_avx2(p1,
                                      p2,
                                      _mm256_loadu_si256((__m256i const*)(dst + dst_stride)),
                                      yfilter,
                                      pw8,
                                      mask_00ff,
                                      &sum_reg,
                                      &sse_reg);
                dst += dst_stride << 1;
                p0 = p2;
                p1 = p3;
                p2 = p4;
            }
            bilinear_v_accum_avx2(
                p0, p1, _mm256_loadu_si256((__m256i const*)dst), yfilter, pw8, mask_00ff, &sum_reg, &sse_reg);
            bilinear_v_accum_avx2(p1,
                                  p2,
                                  _mm256_loadu_si256((__m256i const*)(dst + dst_stride)),
                                  yfilter,
                                  pw8,
                                  mask_00ff,
                                  &sum_reg,
                                  &sse_reg);
            return bilinear_sum_sse_avx2(sum_reg, sse_reg, sse);
        }
    }
    CALC_SUM_AND_SSE
    _mm256_zeroupper();
    return sum;
}

unsigned int svt_aom_sub_pixel_variance16xh_avx2(const uint8_t* src, int src_stride, int x_offset, int y_offset,
                                                 const uint8_t* dst, int dst_stride, int height, unsigned int* sse) {
    __m256i src_reg, dst_reg, exp_src_lo, exp_src_hi, exp_dst_lo, exp_dst_hi;
    __m256i sse_reg, sum_reg, sse_reg_hi, res_cmp, sum_reg_lo, sum_reg_hi;
    __m256i zero_reg;
    int     i, sum;
    sum_reg  = _mm256_set1_epi16(0);
    sse_reg  = _mm256_set1_epi16(0);
    zero_reg = _mm256_set1_epi16(0);

    // x_offset = 0 and y_offset = 0
    if (x_offset == 0) {
        if (y_offset == 0) {
            for (i = 0; i < height; i += 2) {
                LOAD_SRC_DST_INSERT(src_stride, dst_stride)
                // expend each byte to 2 bytes
                MERGE_WITH_SRC(src_reg, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src += (src_stride << 1);
                dst += (dst_stride << 1);
            }
            // x_offset = 0 and y_offset = 4
        } else if (y_offset == 4) {
            __m256i src_next_reg;
            for (i = 0; i < height; i += 2) {
                LOAD_SRC_DST_INSERT(src_stride, dst_stride)
                AVG_NEXT_SRC_INSERT(src_reg, src_stride)
                // expend each byte to 2 bytes
                MERGE_WITH_SRC(src_reg, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src += (src_stride << 1);
                dst += (dst_stride << 1);
            }
            // x_offset = 0 and y_offset = bilin interpolation
        } else {
            __m256i filter, pw8, src_next_reg;
            y_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + y_offset));
            pw8    = _mm256_set1_epi16(8);
            for (i = 0; i < height; i += 2) {
                LOAD_SRC_DST_INSERT(src_stride, dst_stride)
                MERGE_NEXT_SRC_INSERT(src_reg, src_stride)
                FILTER_SRC(filter)
                CALC_SUM_SSE_INSIDE_LOOP
                src += (src_stride << 1);
                dst += (dst_stride << 1);
            }
        }
        // x_offset = 4  and y_offset = 0
    } else if (x_offset == 4) {
        if (y_offset == 0) {
            __m256i src_next_reg;
            for (i = 0; i < height; i += 2) {
                LOAD_SRC_NEXT_BYTE_INSERT
                LOAD_DST_INSERT
                /* average between current and next stride source */
                src_reg = _mm256_avg_epu8(src_reg, src_next_reg);
                // expand each byte to 2 bytes
                MERGE_WITH_SRC(src_reg, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src += (src_stride << 1);
                dst += (dst_stride << 1);
            }
            // x_offset = 4  and y_offset = 4
        } else if (y_offset == 4) {
            __m256i src_next_reg, src_avg, src_temp;
            // load and insert source and next row source
            LOAD_SRC_NEXT_BYTE_INSERT
            src_avg = _mm256_avg_epu8(src_reg, src_next_reg);
            src += src_stride << 1;
            for (i = 0; i < height - 2; i += 2) {
                LOAD_SRC_NEXT_BYTE_INSERT
                src_next_reg = _mm256_avg_epu8(src_reg, src_next_reg);
                src_temp     = _mm256_permute2x128_si256(src_avg, src_next_reg, 0x21);
                src_temp     = _mm256_avg_epu8(src_avg, src_temp);
                LOAD_DST_INSERT
                // expand each byte to 2 bytes
                MERGE_WITH_SRC(src_temp, zero_reg)
                // save current source average
                src_avg = src_next_reg;
                CALC_SUM_SSE_INSIDE_LOOP
                dst += dst_stride << 1;
                src += src_stride << 1;
            }
            // last 2 rows processing happens here
            __m128i src_reg_0 = _mm_loadu_si128((__m128i*)(src));
            __m128i src_reg_1 = _mm_loadu_si128((__m128i*)(src + 1));
            src_reg_0         = _mm_avg_epu8(src_reg_0, src_reg_1);
            src_next_reg      = _mm256_permute2x128_si256(src_avg, _mm256_castsi128_si256(src_reg_0), 0x21);
            LOAD_DST_INSERT
            src_avg = _mm256_avg_epu8(src_avg, src_next_reg);
            MERGE_WITH_SRC(src_avg, zero_reg)
            CALC_SUM_SSE_INSIDE_LOOP
        } else {
            // x_offset = 4  and y_offset = bilin interpolation
            __m256i filter, pw8, src_next_reg, src_avg, src_temp;
            y_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + y_offset));
            pw8    = _mm256_set1_epi16(8);
            // load and insert source and next row source
            LOAD_SRC_NEXT_BYTE_INSERT
            src_avg = _mm256_avg_epu8(src_reg, src_next_reg);
            src += src_stride << 1;
            for (i = 0; i < height - 2; i += 2) {
                LOAD_SRC_NEXT_BYTE_INSERT
                src_next_reg = _mm256_avg_epu8(src_reg, src_next_reg);
                src_temp     = _mm256_permute2x128_si256(src_avg, src_next_reg, 0x21);
                LOAD_DST_INSERT
                MERGE_WITH_SRC(src_avg, src_temp)
                // save current source average
                src_avg = src_next_reg;
                FILTER_SRC(filter)
                CALC_SUM_SSE_INSIDE_LOOP
                dst += dst_stride << 1;
                src += src_stride << 1;
            }
            // last 2 rows processing happens here
            __m128i src_reg_0 = _mm_loadu_si128((__m128i*)(src));
            __m128i src_reg_1 = _mm_loadu_si128((__m128i*)(src + 1));
            src_reg_0         = _mm_avg_epu8(src_reg_0, src_reg_1);
            src_next_reg      = _mm256_permute2x128_si256(src_avg, _mm256_castsi128_si256(src_reg_0), 0x21);
            LOAD_DST_INSERT
            MERGE_WITH_SRC(src_avg, src_next_reg)
            FILTER_SRC(filter)
            CALC_SUM_SSE_INSIDE_LOOP
        }
        // x_offset = bilin interpolation and y_offset = 0
    } else {
        if (y_offset == 0) {
            __m256i filter, pw8, src_next_reg;
            x_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + x_offset));
            pw8    = _mm256_set1_epi16(8);
            for (i = 0; i < height; i += 2) {
                LOAD_SRC_DST_INSERT(src_stride, dst_stride)
                MERGE_NEXT_SRC_INSERT(src_reg, 1)
                FILTER_SRC(filter)
                CALC_SUM_SSE_INSIDE_LOOP
                src += (src_stride << 1);
                dst += (dst_stride << 1);
            }
            // x_offset = bilin interpolation and y_offset = 4
        } else if (y_offset == 4) {
            __m256i filter, pw8, src_next_reg, src_pack;
            x_offset <<= 5;
            filter = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + x_offset));
            pw8    = _mm256_set1_epi16(8);
            // load and insert source and next row source
            LOAD_SRC_NEXT_BYTE_INSERT
            MERGE_WITH_SRC(src_reg, src_next_reg)
            FILTER_SRC(filter)
            // convert each 16 bit to 8 bit to each low and high lane source
            src_pack = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
            src += src_stride << 1;
            for (i = 0; i < height - 2; i += 2) {
                LOAD_SRC_NEXT_BYTE_INSERT
                LOAD_DST_INSERT
                MERGE_WITH_SRC(src_reg, src_next_reg)
                FILTER_SRC(filter)
                src_reg      = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
                src_next_reg = _mm256_permute2x128_si256(src_pack, src_reg, 0x21);
                // average between previous pack to the current
                src_pack = _mm256_avg_epu8(src_pack, src_next_reg);
                MERGE_WITH_SRC(src_pack, zero_reg)
                CALC_SUM_SSE_INSIDE_LOOP
                src_pack = src_reg;
                src += src_stride << 1;
                dst += dst_stride << 1;
            }
            // last 2 rows processing happens here
            LOAD_SRC_MERGE_128BIT(filter)
            LOAD_DST_INSERT
            FILTER_SRC_128BIT(filter_128bit)
            src_reg_0    = _mm_packus_epi16(src_lo, src_hi);
            src_next_reg = _mm256_permute2x128_si256(src_pack, _mm256_castsi128_si256(src_reg_0), 0x21);
            // average between previous pack to the current
            src_pack = _mm256_avg_epu8(src_pack, src_next_reg);
            MERGE_WITH_SRC(src_pack, zero_reg)
            CALC_SUM_SSE_INSIDE_LOOP
        } else {
            // x_offset = bilin interpolation and y_offset = bilin interpolation
            // Each register holds two rows (one per 128-bit lane). The next row
            // pair's horizontal pass is interleaved with the current vertical pass.
            const __m256i xfilter   = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + (x_offset << 5)));
            const __m256i yfilter   = _mm256_load_si256((__m256i const*)(bilinear_filters_avx2 + (y_offset << 5)));
            const __m256i pw8       = _mm256_set1_epi16(8);
            const __m256i mask_00ff = _mm256_set1_epi16(0x00ff);
            __m256i       p0        = bilinear_h_16x2_avx2(src, src_stride, xfilter, pw8);
            __m256i       p1        = bilinear_h_16x2_avx2(src + 2 * src_stride, src_stride, xfilter, pw8);
            src += 4 * src_stride;
            for (i = 0; i < height - 4; i += 2) {
                const __m256i p2 = bilinear_h_16x2_avx2(src, src_stride, xfilter, pw8);
                src += src_stride << 1;
                bilinear_v_accum_avx2(p0,
                                      _mm256_permute2x128_si256(p0, p1, 0x21),
                                      load_16x2_avx2(dst, dst_stride),
                                      yfilter,
                                      pw8,
                                      mask_00ff,
                                      &sum_reg,
                                      &sse_reg);
                dst += dst_stride << 1;
                p0 = p1;
                p1 = p2;
            }
            bilinear_v_accum_avx2(p0,
                                  _mm256_permute2x128_si256(p0, p1, 0x21),
                                  load_16x2_avx2(dst, dst_stride),
                                  yfilter,
                                  pw8,
                                  mask_00ff,
                                  &sum_reg,
                                  &sse_reg);
            dst += dst_stride << 1;
            // The last output row needs one more source row (height + 1 in total).
            const __m256i p_last = _mm256_castsi128_si256(bilinear_h_16_sse(src, xfilter, pw8));
            bilinear_v_accum_avx2(p1,
                                  _mm256_permute2x128_si256(p1, p_last, 0x21),
                                  load_16x2_avx2(dst, dst_stride),
                                  yfilter,
                                  pw8,
                                  mask_00ff,
                                  &sum_reg,
                                  &sse_reg);
            return bilinear_sum_sse_avx2(sum_reg, sse_reg, sse);
        }
    }
    CALC_SUM_AND_SSE
    _mm256_zeroupper();
    return sum;
}

/*
// Unused and conflicting function
unsigned int aom_sub_pixel_avg_variance32xh_avx2(
    const uint8_t *src, int src_stride, int x_offset, int y_offset,
    const uint8_t *dst, int dst_stride, const uint8_t *sec, int sec_stride,
    int height, unsigned int *sse) {
  __m256i sec_reg;
  __m256i src_reg, dst_reg, exp_src_lo, exp_src_hi, exp_dst_lo, exp_dst_hi;
  __m256i sse_reg, sum_reg, sse_reg_hi, res_cmp, sum_reg_lo, sum_reg_hi;
  __m256i zero_reg;
  int i, sum;
  sum_reg = _mm256_set1_epi16(0);
  sse_reg = _mm256_set1_epi16(0);
  zero_reg = _mm256_set1_epi16(0);

  // x_offset = 0 and y_offset = 0
  if (x_offset == 0) {
    if (y_offset == 0) {
      for (i = 0; i < height; i++) {
        LOAD_SRC_DST
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_reg = _mm256_avg_epu8(src_reg, sec_reg);
        sec += sec_stride;
        // expend each byte to 2 bytes
        MERGE_WITH_SRC(src_reg, zero_reg)
        CALC_SUM_SSE_INSIDE_LOOP
        src += src_stride;
        dst += dst_stride;
      }
    } else if (y_offset == 8) {
      __m256i src_next_reg;
      for (i = 0; i < height; i++) {
        LOAD_SRC_DST
        AVG_NEXT_SRC(src_reg, src_stride)
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_reg = _mm256_avg_epu8(src_reg, sec_reg);
        sec += sec_stride;
        // expend each byte to 2 bytes
        MERGE_WITH_SRC(src_reg, zero_reg)
        CALC_SUM_SSE_INSIDE_LOOP
        src += src_stride;
        dst += dst_stride;
      }
      // x_offset = 0 and y_offset = bilin interpolation
    } else {
      __m256i filter, pw8, src_next_reg;

      y_offset <<= 5;
      filter = _mm256_load_si256(
          (__m256i const *)(bilinear_filters_avx2 + y_offset));
      pw8 = _mm256_set1_epi16(8);
      for (i = 0; i < height; i++) {
        LOAD_SRC_DST
        MERGE_NEXT_SRC(src_reg, src_stride)
        FILTER_SRC(filter)
        src_reg = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_reg = _mm256_avg_epu8(src_reg, sec_reg);
        sec += sec_stride;
        MERGE_WITH_SRC(src_reg, zero_reg)
        CALC_SUM_SSE_INSIDE_LOOP
        src += src_stride;
        dst += dst_stride;
      }
    }
    // x_offset = 8  and y_offset = 0
  } else if (x_offset == 8) {
    if (y_offset == 0) {
      __m256i src_next_reg;
      for (i = 0; i < height; i++) {
        LOAD_SRC_DST
        AVG_NEXT_SRC(src_reg, 1)
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_reg = _mm256_avg_epu8(src_reg, sec_reg);
        sec += sec_stride;
        // expand each byte to 2 bytes
        MERGE_WITH_SRC(src_reg, zero_reg)
        CALC_SUM_SSE_INSIDE_LOOP
        src += src_stride;
        dst += dst_stride;
      }
      // x_offset = 8  and y_offset = 8
    } else if (y_offset == 8) {
      __m256i src_next_reg, src_avg;
      // load source and another source starting from the next
      // following byte
      src_reg = _mm256_loadu_si256((__m256i const *)(src));
      AVG_NEXT_SRC(src_reg, 1)
      for (i = 0; i < height; i++) {
        // save current source average
        src_avg = src_reg;
        src += src_stride;
        LOAD_SRC_DST
        AVG_NEXT_SRC(src_reg, 1)
        // average between previous average to current average
        src_avg = _mm256_avg_epu8(src_avg, src_reg);
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_avg = _mm256_avg_epu8(src_avg, sec_reg);
        sec += sec_stride;
        // expand each byte to 2 bytes
        MERGE_WITH_SRC(src_avg, zero_reg)
        CALC_SUM_SSE_INSIDE_LOOP
        dst += dst_stride;
      }
      // x_offset = 8  and y_offset = bilin interpolation
    } else {
      __m256i filter, pw8, src_next_reg, src_avg;
      y_offset <<= 5;
      filter = _mm256_load_si256(
          (__m256i const *)(bilinear_filters_avx2 + y_offset));
      pw8 = _mm256_set1_epi16(8);
      // load source and another source starting from the next
      // following byte
      src_reg = _mm256_loadu_si256((__m256i const *)(src));
      AVG_NEXT_SRC(src_reg, 1)
      for (i = 0; i < height; i++) {
        // save current source average
        src_avg = src_reg;
        src += src_stride;
        LOAD_SRC_DST
        AVG_NEXT_SRC(src_reg, 1)
        MERGE_WITH_SRC(src_avg, src_reg)
        FILTER_SRC(filter)
        src_avg = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_avg = _mm256_avg_epu8(src_avg, sec_reg);
        // expand each byte to 2 bytes
        MERGE_WITH_SRC(src_avg, zero_reg)
        sec += sec_stride;
        CALC_SUM_SSE_INSIDE_LOOP
        dst += dst_stride;
      }
    }
    // x_offset = bilin interpolation and y_offset = 0
  } else {
    if (y_offset == 0) {
      __m256i filter, pw8, src_next_reg;
      x_offset <<= 5;
      filter = _mm256_load_si256(
          (__m256i const *)(bilinear_filters_avx2 + x_offset));
      pw8 = _mm256_set1_epi16(8);
      for (i = 0; i < height; i++) {
        LOAD_SRC_DST
        MERGE_NEXT_SRC(src_reg, 1)
        FILTER_SRC(filter)
        src_reg = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_reg = _mm256_avg_epu8(src_reg, sec_reg);
        MERGE_WITH_SRC(src_reg, zero_reg)
        sec += sec_stride;
        CALC_SUM_SSE_INSIDE_LOOP
        src += src_stride;
        dst += dst_stride;
      }
      // x_offset = bilin interpolation and y_offset = 8
    } else if (y_offset == 8) {
      __m256i filter, pw8, src_next_reg, src_pack;
      x_offset <<= 5;
      filter = _mm256_load_si256(
          (__m256i const *)(bilinear_filters_avx2 + x_offset));
      pw8 = _mm256_set1_epi16(8);
      src_reg = _mm256_loadu_si256((__m256i const *)(src));
      MERGE_NEXT_SRC(src_reg, 1)
      FILTER_SRC(filter)
      // convert each 16 bit to 8 bit to each low and high lane source
      src_pack = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
      for (i = 0; i < height; i++) {
        src += src_stride;
        LOAD_SRC_DST
        MERGE_NEXT_SRC(src_reg, 1)
        FILTER_SRC(filter)
        src_reg = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
        // average between previous pack to the current
        src_pack = _mm256_avg_epu8(src_pack, src_reg);
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_pack = _mm256_avg_epu8(src_pack, sec_reg);
        sec += sec_stride;
        MERGE_WITH_SRC(src_pack, zero_reg)
        src_pack = src_reg;
        CALC_SUM_SSE_INSIDE_LOOP
        dst += dst_stride;
      }
      // x_offset = bilin interpolation and y_offset = bilin interpolation
    } else {
      __m256i xfilter, yfilter, pw8, src_next_reg, src_pack;
      x_offset <<= 5;
      xfilter = _mm256_load_si256(
          (__m256i const *)(bilinear_filters_avx2 + x_offset));
      y_offset <<= 5;
      yfilter = _mm256_load_si256(
          (__m256i const *)(bilinear_filters_avx2 + y_offset));
      pw8 = _mm256_set1_epi16(8);
      // load source and another source starting from the next
      // following byte
      src_reg = _mm256_loadu_si256((__m256i const *)(src));
      MERGE_NEXT_SRC(src_reg, 1)

      FILTER_SRC(xfilter)
      // convert each 16 bit to 8 bit to each low and high lane source
      src_pack = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
      for (i = 0; i < height; i++) {
        src += src_stride;
        LOAD_SRC_DST
        MERGE_NEXT_SRC(src_reg, 1)
        FILTER_SRC(xfilter)
        src_reg = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
        // merge previous pack to current pack source
        MERGE_WITH_SRC(src_pack, src_reg)
        // filter the source
        FILTER_SRC(yfilter)
        src_pack = _mm256_packus_epi16(exp_src_lo, exp_src_hi);
        sec_reg = _mm256_loadu_si256((__m256i const *)(sec));
        src_pack = _mm256_avg_epu8(src_pack, sec_reg);
        MERGE_WITH_SRC(src_pack, zero_reg)
        src_pack = src_reg;
        sec += sec_stride;
        CALC_SUM_SSE_INSIDE_LOOP
        dst += dst_stride;
      }
    }
  }
  CALC_SUM_AND_SSE
  _mm256_zeroupper();
  return sum;
}

*/
