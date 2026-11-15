/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include <smmintrin.h>
#include <string.h>
#include "common_dsp_rtcd.h"
#include "synonyms.h"
#include "common_utils.h"

extern const int8_t eb_av1_filter_intra_taps[FILTER_INTRA_MODES][8][8];
#define FILTER_INTRA_SCALE_BITS 4

static INLINE __m128i xx_load_128(const void* a) {
    return _mm_loadu_si128((const __m128i*)a);
}

void svt_av1_filter_intra_predictor_sse4_1(uint8_t* dst, ptrdiff_t stride, TxSize tx_size, const uint8_t* above,
                                           const uint8_t* left, int mode) {
    int       r, c;
    uint8_t   buffer[33][33];
    const int bw = tx_size_wide[tx_size];
    const int bh = tx_size_high[tx_size];

    assert(bw <= 32 && bh <= 32);

    for (r = 0; r < bh; ++r) {
        buffer[r + 1][0] = left[r];
    }
    memcpy(buffer[0], &above[-1], (bw + 1) * sizeof(uint8_t));

    const __m128i f1f0                    = xx_load_128(eb_av1_filter_intra_taps[mode][0]);
    const __m128i f3f2                    = xx_load_128(eb_av1_filter_intra_taps[mode][2]);
    const __m128i f5f4                    = xx_load_128(eb_av1_filter_intra_taps[mode][4]);
    const __m128i f7f6                    = xx_load_128(eb_av1_filter_intra_taps[mode][6]);
    const __m128i filter_intra_scale_bits = _mm_set1_epi16(1 << (15 - FILTER_INTRA_SCALE_BITS));
    // p0..p4 come from the row above; p5, p6 (left column) are carried in a register from the previous 4x2 block
    // so the serial chain avoids a store/reload. p7 has a zero tap and is kept zero.
    const __m128i above_mask = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i left_shuf  = _mm_setr_epi8(-1, -1, -1, -1, -1, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    for (r = 1; r < bh + 1; r += 2) {
        __m128i p_left = _mm_insert_epi8(_mm_insert_epi8(_mm_setzero_si128(), buffer[r][0], 5), buffer[r + 1][0], 6);
        for (c = 1; c < bw + 1; c += 4) {
            const __m128i p_b    = _mm_or_si128(_mm_and_si128(xx_loadl_64(&buffer[r - 1][c - 1]), above_mask), p_left);
            const __m128i in     = _mm_unpacklo_epi64(p_b, p_b);
            const __m128i out_01 = _mm_maddubs_epi16(in, f1f0);
            const __m128i out_23 = _mm_maddubs_epi16(in, f3f2);
            const __m128i out_45 = _mm_maddubs_epi16(in, f5f4);
            const __m128i out_67 = _mm_maddubs_epi16(in, f7f6);
            const __m128i out_0123     = _mm_hadd_epi16(out_01, out_23);
            const __m128i out_4567     = _mm_hadd_epi16(out_45, out_67);
            const __m128i out_01234567 = _mm_hadd_epi16(out_0123, out_4567);
            // Rounding
            const __m128i round_w = _mm_mulhrs_epi16(out_01234567, filter_intra_scale_bits);
            const __m128i out_r   = _mm_packus_epi16(round_w, round_w);
            const __m128i out_r1  = _mm_srli_si128(out_r, 4);
            // Storing
            xx_storel_32(&buffer[r][c], out_r);
            xx_storel_32(&buffer[r + 1][c], out_r1);
            p_left = _mm_shuffle_epi8(out_r, left_shuf);
        }
    }

    switch (bw) {
    case 4:
        for (r = 0; r < bh; ++r, dst += stride) {
            xx_storel_32(dst, xx_loadl_32(&buffer[r + 1][1]));
        }
        break;
    case 8:
        for (r = 0; r < bh; ++r, dst += stride) {
            xx_storel_64(dst, xx_loadl_64(&buffer[r + 1][1]));
        }
        break;
    case 16:
        for (r = 0; r < bh; ++r, dst += stride) {
            xx_storeu_128(dst, xx_loadu_128(&buffer[r + 1][1]));
        }
        break;
    default:
        for (r = 0; r < bh; ++r, dst += stride) {
            xx_storeu_128(dst, xx_loadu_128(&buffer[r + 1][1]));
            xx_storeu_128(dst + 16, xx_loadu_128(&buffer[r + 1][17]));
        }
        break;
    }
}
