/*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "common_dsp_rtcd.h"
#if EN_AVX512_SUPPORT
#include <immintrin.h>
#include "convolve.h"
#include "inter_prediction.h"

static INLINE __mmask32 pixel_mask(unsigned n) {
    return n >= 32 ? UINT32_MAX : (1u << n) - 1;
}

static INLINE __m512i pack_split(const uint8_t* msb, const uint8_t* lsb, __mmask32 k) {
    const __m512i hi = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(k, msb));
    const __m512i lo = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(k, lsb));
    return _mm512_or_si512(_mm512_slli_epi16(hi, 2), _mm512_srli_epi16(lo, 6));
}

static AOM_FORCE_INLINE void pack_rows(uint8_t* msb, uint32_t ms, uint8_t* lsb, uint16_t* dst, uint32_t ls, uint32_t ds,
                                       uint32_t w, uint32_t h) {
    for (uint32_t y = 0; y < h; ++y, msb += ms, lsb += ls, dst += ds) {
        uint32_t x = 0;
        for (; x + 32 <= w; x += 32) {
            _mm512_storeu_si512(dst + x, pack_split(msb + x, lsb + x, UINT32_MAX));
        }
        if (x < w) {
            if (w - x == 8) {
                const __m128i h8 = _mm_loadl_epi64((const __m128i*)(msb + x)),
                              l8 = _mm_loadl_epi64((const __m128i*)(lsb + x));
                _mm_storeu_si128((__m128i*)(dst + x), _mm_srli_epi16(_mm_unpacklo_epi8(l8, h8), 6));
                continue;
            }
            const __mmask32 k = pixel_mask(w - x);
            _mm512_mask_storeu_epi16(dst + x, k, pack_split(msb + x, lsb + x, k));
        }
    }
}

void svt_enc_msb_pack2d_avx512(uint8_t* msb, uint32_t ms, uint8_t* lsb, uint16_t* dst, uint32_t ls, uint32_t ds,
                               uint32_t w, uint32_t h) {
    if (w < 32 && !(h & 1) && !(w & 3)) {
        svt_enc_msb_pack2d_avx2_intrin_al(msb, ms, lsb, dst, ls, ds, w, h);
        return;
    }
    // Motion compensation also requests these widths with an eight-pixel border.
    switch (w) {
    case 32:
        pack_rows(msb, ms, lsb, dst, ls, ds, 32, h);
        break;
    case 40:
        pack_rows(msb, ms, lsb, dst, ls, ds, 40, h);
        break;
    case 64:
        pack_rows(msb, ms, lsb, dst, ls, ds, 64, h);
        break;
    case 72:
        pack_rows(msb, ms, lsb, dst, ls, ds, 72, h);
        break;
    case 128:
        pack_rows(msb, ms, lsb, dst, ls, ds, 128, h);
        break;
    case 136:
        pack_rows(msb, ms, lsb, dst, ls, ds, 136, h);
        break;
    default:
        pack_rows(msb, ms, lsb, dst, ls, ds, w, h);
        break;
    }
}

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// The 10-bit compound buffer carries four fractional bits and a bias of 24576.
static INLINE __m512i blend_predictors(__m512i cur, __m512i prev, const ConvolveParams* p, bool weighted) {
    __m512i v;
    if (weighted) {
        const __m512i bias    = _mm512_set1_epi16(INT16_MIN);
        cur                   = _mm512_xor_si512(cur, bias);
        prev                  = _mm512_xor_si512(prev, bias);
        const __m512i weights = _mm512_set1_epi32(p->fwd_offset | (p->bck_offset << 16));
        const __m512i rnd     = _mm512_set1_epi32((32768 - 24576 + 8) * 16);
        const __m512i lo      = _mm512_madd_epi16(_mm512_unpacklo_epi16(prev, cur), weights);
        const __m512i hi      = _mm512_madd_epi16(_mm512_unpackhi_epi16(prev, cur), weights);
        v                     = _mm512_packus_epi32(_mm512_srai_epi32(_mm512_add_epi32(lo, rnd), 8),
                                _mm512_srai_epi32(_mm512_add_epi32(hi, rnd), 8));
    } else {
        // The biased predictor is positive, so subtracting one before
        // the rounded unsigned average gives the required truncation.
        v = _mm512_avg_epu16(prev, _mm512_sub_epi16(cur, _mm512_set1_epi16(1)));
        v = _mm512_srli_epi16(_mm512_subs_epu16(v, _mm512_set1_epi16(24576 - 8)), 4);
    }
    return _mm512_min_epu16(v, _mm512_set1_epi16(1023));
}

static AOM_FORCE_INLINE void fullpel_rows(const uint8_t* msb, int ms, const uint8_t* lsb, int ls, uint16_t* dst, int ds,
                                          int w, int h, const ConvolveParams* p, bool compound, bool average,
                                          bool weighted) {
    if (compound && !average) {
        dst = p->dst;
        ds  = p->dst_stride;
    }
    for (int y = 0; y < h; ++y, msb += ms, lsb += ls, dst += ds) {
        const __m512i order = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
        int           x     = 0;
        for (; x + 64 <= w; x += 64) {
            // Arrange eight-byte groups once so the interleaves directly
            // produce two contiguous ZMM rows in display order.
            const __m512i hi = _mm512_permutexvar_epi64(order, _mm512_loadu_si512(msb + x));
            const __m512i lo = _mm512_permutexvar_epi64(order, _mm512_loadu_si512(lsb + x));
            __m512i       a  = _mm512_srli_epi16(_mm512_unpacklo_epi8(lo, hi), 6);
            __m512i       b  = _mm512_srli_epi16(_mm512_unpackhi_epi8(lo, hi), 6);
            if (compound) {
                a = _mm512_add_epi16(_mm512_slli_epi16(a, 4), _mm512_set1_epi16(24576));
                b = _mm512_add_epi16(_mm512_slli_epi16(b, 4), _mm512_set1_epi16(24576));
            }
            if (average) {
                a = blend_predictors(a, _mm512_loadu_si512(p->dst + y * p->dst_stride + x), p, weighted);
                b = blend_predictors(b, _mm512_loadu_si512(p->dst + y * p->dst_stride + x + 32), p, weighted);
            }
            _mm512_storeu_si512(dst + x, a);
            _mm512_storeu_si512(dst + x + 32, b);
        }
        for (; x < w; x += 32) {
            const __mmask32 k = pixel_mask(w - x);
            __m512i         v = pack_split(msb + x, lsb + x, k);
            if (compound) {
                v = _mm512_add_epi16(_mm512_slli_epi16(v, 4), _mm512_set1_epi16(24576));
            }
            if (average) {
                v = blend_predictors(v, _mm512_maskz_loadu_epi16(k, p->dst + y * p->dst_stride + x), p, weighted);
            }
            _mm512_mask_storeu_epi16(dst + x, k, v);
        }
    }
}

static AOM_FORCE_INLINE void fullpel_split(const uint8_t* msb, int ms, const uint8_t* lsb, int ls, uint16_t* dst,
                                           int ds, int w, int h, ConvolveParams* p) {
    if (!p->is_compound) {
        fullpel_rows(msb, ms, lsb, ls, dst, ds, w, h, p, false, false, false);
    } else if (!p->do_average) {
        fullpel_rows(msb, ms, lsb, ls, dst, ds, w, h, p, true, false, false);
    } else if (p->use_jnt_comp_avg) {
        fullpel_rows(msb, ms, lsb, ls, dst, ds, w, h, p, true, true, true);
    } else {
        fullpel_rows(msb, ms, lsb, ls, dst, ds, w, h, p, true, true, false);
    }
}

void svt_aom_highbd_10_fullpel_split_avx512(const uint8_t* msb, int ms, const uint8_t* lsb, int ls, uint16_t* dst,
                                            int ds, int w, int h, ConvolveParams* p) {
    if (w < 32) {
        svt_aom_highbd_10_fullpel_split_avx2(msb, ms, lsb, ls, dst, ds, w, h, p);
    } else if (w == 32) {
        fullpel_split(msb, ms, lsb, ls, dst, ds, 32, h, p);
    } else {
        fullpel_split(msb, ms, lsb, ls, dst, ds, w, h, p);
    }
}

typedef struct FilterPairs {
    __m512i coeff[4];
    int     start, count;
} FilterPairs;

static INLINE FilterPairs active_pairs(const int16_t* taps) {
    FilterPairs f     = {0};
    int         start = 0, end = 8;
    while (start < 6 && !taps[start] && !taps[start + 1]) {
        start += 2;
    }
    while (end > start + 2 && !taps[end - 2] && !taps[end - 1]) {
        end -= 2;
    }
    f.start = start;
    f.count = (end - start) / 2;
    for (int i = 0; i < f.count; ++i) {
        f.coeff[i] = _mm512_broadcastd_epi32(_mm_loadu_si32(taps + start + 2 * i));
    }
    return f;
}

static INLINE __m512i horizontal_16(const uint16_t* src, const FilterPairs* f, int n, int count) {
    // Only n + taps - 1 source words belong to this stripe, including the border.
    const __m512i samples = _mm512_maskz_loadu_epi16(pixel_mask(n + 2 * count - 1), src);
    const __m512i pairs   = _mm512_setr_epi32(0x10000,
                                            0x20001,
                                            0x30002,
                                            0x40003,
                                            0x50004,
                                            0x60005,
                                            0x70006,
                                            0x80007,
                                            0x90008,
                                            0xa0009,
                                            0xb000a,
                                            0xc000b,
                                            0xd000c,
                                            0xe000d,
                                            0xf000e,
                                            0x10000f);
    __m512i       sum     = _mm512_madd_epi16(_mm512_permutexvar_epi16(pairs, samples), f->coeff[0]);
    for (int i = 1; i < count; ++i) {
        const __m512i idx = _mm512_add_epi16(pairs, _mm512_set1_epi16(2 * i));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(_mm512_permutexvar_epi16(idx, samples), f->coeff[i]));
    }
    return _mm512_srai_epi32(_mm512_add_epi32(sum, _mm512_set1_epi32((1 << 16) + 4)), 3);
}

static AOM_FORCE_INLINE void convolve_pairs(const uint16_t* src, int ss, uint16_t* dst, int ds, int w, int h,
                                            const FilterPairs* fx, const FilterPairs* fy, ConvolveParams* p,
                                            bool compound, int nx, int ny) {
    const uint16_t* first = src + (fy->start - 3) * ss + fx->start - 3;
    for (int x = 0; x < w; x += 16) {
        const int       n    = AOMMIN(16, w - x);
        const __mmask16 mask = (__mmask16)pixel_mask(n);
        __m512i         ring[8], prev = _mm512_setzero_si512();
        // Each ring entry contains one pair of neighboring rows per dword.
        // An output row uses up to four entries and creates just one new pair.
        for (int r = 0; r < h + 2 * ny - 1; ++r) {
            const __m512i row = horizontal_16(first + r * ss + x, fx, n, nx);
            ring[r & 7]       = _mm512_or_si512(prev, _mm512_slli_epi32(row, 16));
            prev              = row;
            const int y       = r - (2 * ny - 1);
            if (y < 0) {
                continue;
            }
            __m512i sum = _mm512_madd_epi16(ring[(y + 1) & 7], fy->coeff[0]);
            for (int j = 1; j < ny; ++j) {
                sum = _mm512_add_epi32(sum, _mm512_madd_epi16(ring[(y + 1 + 2 * j) & 7], fy->coeff[j]));
            }
            if (compound) {
                const __m256i v = _mm512_cvtepi32_epi16(
                    _mm512_srai_epi32(_mm512_add_epi32(sum, _mm512_set1_epi32((1 << 21) + 64)), 7));
                uint16_t* buf = p->dst + y * p->dst_stride + x;
                if (p->do_average) {
                    const __m512i mix = blend_predictors(_mm512_castsi256_si512(v),
                                                         _mm512_castsi256_si512(_mm256_maskz_loadu_epi16(mask, buf)),
                                                         p,
                                                         p->use_jnt_comp_avg);
                    _mm256_mask_storeu_epi16(dst + y * ds + x, mask, _mm512_castsi512_si256(mix));
                } else {
                    _mm256_mask_storeu_epi16(buf, mask, v);
                }
            } else {
                const __m512i value = _mm512_srai_epi32(_mm512_add_epi32(sum, _mm512_set1_epi32(1024 - (1 << 20))), 11);
                const __m256i pix   = _mm512_cvtusepi32_epi16(_mm512_max_epi32(value, _mm512_setzero_si512()));
                _mm256_mask_storeu_epi16(dst + y * ds + x, mask, _mm256_min_epu16(pix, _mm256_set1_epi16(1023)));
            }
        }
    }
}

// Regular and smooth phases usually need three tap pairs. Keeping these
// counts constant lets the compiler schedule the independent products together.
static AOM_FORCE_INLINE void convolve_10(const uint16_t* src, int ss, uint16_t* dst, int ds, int w, int h,
                                         const int16_t* xf, const int16_t* yf, ConvolveParams* p, bool compound) {
    const FilterPairs fx = active_pairs(xf), fy = active_pairs(yf);
    if (fx.count == 3 && fy.count == 3) {
        convolve_pairs(src, ss, dst, ds, w, h, &fx, &fy, p, compound, 3, 3);
    } else if (fx.count == 4 && fy.count == 4) {
        convolve_pairs(src, ss, dst, ds, w, h, &fx, &fy, p, compound, 4, 4);
    } else {
        convolve_pairs(src, ss, dst, ds, w, h, &fx, &fy, p, compound, fx.count, fy.count);
    }
}

// At 10 bits, the first bilinear pass fits in sixteen bits including
// the four fractional bits. Keep it in this form until the vertical dot.
static INLINE void bilinear_sr(const uint16_t* src, int ss, uint16_t* dst, int ds, int w, int h, int px, int py) {
    const __m512i fx = _mm512_set1_epi16(px), fy = _mm512_set1_epi16(py << 11);
    for (int x = 0; x < w; x += 32) {
        const __mmask32 k    = pixel_mask(w - x);
        __m512i         prev = _mm512_setzero_si512();
        for (int y = 0; y <= h; ++y) {
            const uint16_t* s = src + y * ss + x;
            const __m512i   a = _mm512_maskz_loadu_epi16(k, s), b = _mm512_maskz_loadu_epi16(k, s + 1);
            const __m512i   row = _mm512_add_epi16(_mm512_slli_epi16(a, 4),
                                                 _mm512_mullo_epi16(_mm512_sub_epi16(b, a), fx));
            if (y) {
                // Horizontal values are at most 16368. Doubling their signed
                // difference fits a word, so the high product gives floor(d*py/16).
                // This truncation is exact before the final rounded divide by 16.
                const __m512i diff  = _mm512_sub_epi16(row, prev);
                const __m512i delta = _mm512_mulhi_epi16(_mm512_add_epi16(diff, diff), fy);
                const __m512i pix   = _mm512_mulhrs_epi16(_mm512_add_epi16(prev, delta), _mm512_set1_epi16(2048));
                _mm512_mask_storeu_epi16(dst + (y - 1) * ds + x, k, pix);
            }
            prev = row;
        }
    }
}

void svt_av1_highbd_jnt_convolve_2d_avx512(const uint16_t* src, int ss, uint16_t* dst, int ds, int w, int h,
                                           const InterpFilterParams* fx, const InterpFilterParams* fy, int px, int py,
                                           ConvolveParams* p, int bd) {
    if (bd != 10 || fx->taps != 8 || fy->taps != 8 || p->round_0 != 3 || p->round_1 != 7 || w < 16) {
        svt_av1_highbd_jnt_convolve_2d_avx2(src, ss, dst, ds, w, h, fx, fy, px, py, p, bd);
        return;
    }
    convolve_10(src,
                ss,
                dst,
                ds,
                w,
                h,
                av1_get_interp_filter_subpel_kernel(*fx, px & SUBPEL_MASK),
                av1_get_interp_filter_subpel_kernel(*fy, py & SUBPEL_MASK),
                p,
                true);
}

void svt_av1_highbd_convolve_2d_sr_avx512(const uint16_t* src, int ss, uint16_t* dst, int ds, int w, int h,
                                          const InterpFilterParams* fx, const InterpFilterParams* fy, int px, int py,
                                          ConvolveParams* p, int bd) {
    if (bd != 10 || fx->taps != 8 || fy->taps != 8 || p->round_0 != 3 || p->round_1 != 11 || w < 32) {
        svt_av1_highbd_convolve_2d_sr_avx2(src, ss, dst, ds, w, h, fx, fy, px, py, p, bd);
        return;
    }
    if (fx->filter_ptr == &bilinear_filters[0][0] && fy->filter_ptr == &bilinear_filters[0][0]) {
        bilinear_sr(src, ss, dst, ds, w, h, px & SUBPEL_MASK, py & SUBPEL_MASK);
        return;
    }
    convolve_10(src,
                ss,
                dst,
                ds,
                w,
                h,
                av1_get_interp_filter_subpel_kernel(*fx, px & SUBPEL_MASK),
                av1_get_interp_filter_subpel_kernel(*fy, py & SUBPEL_MASK),
                p,
                false);
}
#endif
#endif
