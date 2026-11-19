/*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include "common_dsp_rtcd.h"
#include "warped_motion.h"
#if EN_AVX512_SUPPORT && CONFIG_ENABLE_HIGH_BIT_DEPTH
#include <immintrin.h>
#include "svt_threads.h"

// Each word holds two signed byte coefficients for one phase. A pair of
// registers covers 64 phases and feeds a word multiply-add after sign extension.
enum { WARP_PHASES = 3 * WARPEDPIXEL_PREC_SHIFTS + 1 };

static uint16_t filter_pairs[4][WARP_PHASES];
DEFINE_ONCE(filter_once);

static ONCE_ROUTINE(init_filter_pairs) {
    for (int p = 0; p < WARP_PHASES; ++p) {
        for (int k = 0; k < 4; ++k) {
            const int16_t* t   = svt_aom_warped_filter[p] + 2 * k;
            filter_pairs[k][p] = (uint8_t)t[0] | ((uint16_t)(uint8_t)t[1] << 8);
        }
    }
    ONCE_ROUTINE_EPILOG;
}

typedef struct WarpPhases {
    __m256i offsets;
    int     minimum, size, rows;
} WarpPhases;

static INLINE WarpPhases warp_phases(int dx, int dy, int rows) {
    const __m256i x       = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i columns = _mm256_mullo_epi16(x, _mm256_set1_epi16(dx >> WARP_PARAM_REDUCE_BITS));
    WarpPhases    p;
    p.offsets        = _mm256_mask_add_epi16(columns, 0xff00, columns, _mm256_set1_epi16(dy >> WARP_PARAM_REDUCE_BITS));
    const int span_x = 7 * abs(dx), step_y = abs(dy);
    // The caller guarantees that at least two rows fit in the window.
    while (span_x + (rows - 1) * step_y > (63 << WARPEDDIFF_PREC_BITS)) {
        rows /= 2;
    }
    p.rows    = rows;
    p.minimum = AOMMIN(0, 7 * dx) + AOMMIN(0, (rows - 1) * dy);
    p.size    = span_x + (rows - 1) * step_y <= (31 << WARPEDDIFF_PREC_BITS) ? 32 : 64;
    return p;
}

typedef struct WarpFilterWindow {
    __m512i first[4], second[4];
    __m256i position;
} WarpFilterWindow;

// The same 32 or 64 phases cover each group of consecutive row pairs.
// Load that window once, then advance all sixteen Q4 phase coordinates together.
static INLINE WarpFilterWindow warp_window(int base, WarpPhases phases) {
    const int bias     = (1 << (WARPEDDIFF_PREC_BITS - 1)) + (WARPEDPIXEL_PREC_SHIFTS << WARPEDDIFF_PREC_BITS);
    const int start    = clamp((base + bias + phases.minimum) >> WARPEDDIFF_PREC_BITS, 0, WARP_PHASES - phases.size);
    const int relative = ((base + bias) >> WARP_PARAM_REDUCE_BITS) -
        (start << (WARPEDDIFF_PREC_BITS - WARP_PARAM_REDUCE_BITS));
    WarpFilterWindow window;
    window.position = _mm256_add_epi16(phases.offsets, _mm256_set1_epi16(relative));
    for (int k = 0; k < 4; ++k) {
        window.first[k] = _mm512_loadu_si512(filter_pairs[k] + start);
        if (phases.size == 64) {
            window.second[k] = _mm512_loadu_si512(filter_pairs[k] + start + 32);
        }
    }
    return window;
}

static INLINE void warp_weights(const WarpFilterWindow* window, __m256i position, int size, __m512i weight[4]) {
    const __m512i index = _mm512_castsi256_si512(
        _mm256_srai_epi16(position, WARPEDDIFF_PREC_BITS - WARP_PARAM_REDUCE_BITS));
    for (int k = 0; k < 4; ++k) {
        const __m512i pairs = size == 32 ? _mm512_permutexvar_epi16(index, window->first[k])
                                         : _mm512_permutex2var_epi16(window->first[k], index, window->second[k]);
        weight[k]           = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(pairs));
    }
}

// Four independent tap products feed a balanced reduction tree.
static INLINE __m512i warp_sum(__m512i p0, __m512i p1, __m512i p2, __m512i p3, const __m512i weight[4], int bias) {
    const __m512i lo = _mm512_add_epi32(_mm512_madd_epi16(p0, weight[0]), _mm512_madd_epi16(p1, weight[1]));
    const __m512i hi = _mm512_add_epi32(_mm512_madd_epi16(p2, weight[2]), _mm512_madd_epi16(p3, weight[3]));
    return _mm512_add_epi32(_mm512_add_epi32(lo, hi), _mm512_set1_epi32(bias));
}

static INLINE __m256i warp_samples(const uint8_t* hi, const uint8_t* lo, __mmask16 mask, __m128i edge, bool interior) {
    __m128i a = _mm_maskz_loadu_epi8(mask, hi), b = _mm_maskz_loadu_epi8(mask, lo);
    if (!interior) {
        a = _mm_shuffle_epi8(a, edge);
        b = _mm_shuffle_epi8(b, edge);
    }
    return _mm256_srli_epi16(_mm256_set_m128i(_mm_unpackhi_epi8(b, a), _mm_unpacklo_epi8(b, a)), 6);
}

// The two halves contain (row 0,row 1) and (row 1,row 2) for eight pixels.
// Build each vertical pair once as consecutive horizontal rows become available.
static INLINE __m512i vertical_pair(__m256i previous, __m256i next) {
    const __m512i         rows        = _mm512_inserti64x4(_mm512_castsi256_si512(previous), next, 1);
    static const uint16_t indices[32] = {0, 8,  1, 9,  2,  10, 3,  11, 4,  12, 5,  13, 6,  14, 7,  15,
                                         8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23};
    return _mm512_permutexvar_epi16(_mm512_loadu_si512(indices), rows);
}

static void warp_affine_paired_rows(const int32_t* mat, const uint8_t* hi, const uint8_t* lo, int width, int height,
                                    int hs, int ls, uint16_t* pred, int pc, int pr, int pw, int ph, int ps, int sxsub,
                                    int sysub, int bd, ConvolveParams* p, int16_t alpha, int16_t beta, int16_t gamma,
                                    int16_t delta) {
    svt_run_once(&filter_once, init_filter_pairs);
    const WarpPhases horizontal = warp_phases(alpha, beta, 16), vertical = warp_phases(gamma, delta, 8);
    const __m256i    step_x = _mm256_set1_epi16(2 * (beta >> WARP_PARAM_REDUCE_BITS));
    const __m256i    step_y = _mm256_set1_epi16(2 * (delta >> WARP_PARAM_REDUCE_BITS));
    const int        hr     = p->round_0 + AOMMAX(bd + FILTER_BITS - p->round_0 - 14, 0);
    const int        vr     = p->is_compound ? p->round_1 : 2 * FILTER_BITS - hr;
    const int        bits   = 2 * FILTER_BITS - p->round_0 - p->round_1;
    const int        offset = bd + 2 * FILTER_BITS - hr;
    const int        bias   = (1 << (bd + 2 * FILTER_BITS - p->round_0 - p->round_1)) +
        (1 << (bd + 2 * FILTER_BITS - p->round_0 - p->round_1 - 1));
    static const uint16_t indices[32] = {0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
                                         16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24};
    const __m512i         windows     = _mm512_loadu_si512(indices);
    for (int y = 0; y < ph; y += 8) {
        for (int x = 0; x < pw; x += 8) {
            const int xx = (pc + x + 4) << sxsub, yy = (pr + y + 4) << sysub;
            const int cx = (mat[2] * xx + mat[3] * yy + mat[0]) >> sxsub,
                      cy = (mat[4] * xx + mat[5] * yy + mat[1]) >> sysub;
            const int ix = (cx >> WARPEDMODEL_PREC_BITS) - 7, iy = (cy >> WARPEDMODEL_PREC_BITS) - 7;
            const int frac = (1 << WARPEDMODEL_PREC_BITS) - 1, keep = ~((1 << WARP_PARAM_REDUCE_BITS) - 1);
            const int sx = ((cx & frac) - 4 * (alpha + beta)) & keep, sy = ((cy & frac) - 4 * (gamma + delta)) & keep;
            const int h = AOMMIN(8, ph - y), w = AOMMIN(8, pw - x);
            const __mmask8  mask = (1u << w) - 1;
            __m512i         pairs[8];
            __m256i         previous = _mm256_setzero_si256();
            const int       column   = clamp(ix, 0, AOMMAX(0, width - 16));
            const __mmask16 loadmask = width >= 16 ? UINT16_MAX : (1u << width) - 1;
            const bool      interior = ix == column && width >= 16;
            const __m128i   edge     = _mm_min_epi8(
                _mm_set1_epi8(AOMMIN(width, 16) - 1),
                _mm_max_epi8(_mm_setzero_si128(),
                             _mm_add_epi8(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                                          _mm_set1_epi8(clamp(ix - column, -15, 15)))));

            __m512i wx[4], wy[4];
            for (int group = 0; group < h + 7; group += horizontal.rows) {
                const WarpFilterWindow top   = warp_window(sx + (group - 3) * beta, horizontal);
                __m256i                pos_x = top.position;
                warp_weights(&top, pos_x, horizontal.size, wx);
                for (int r = group; r < AOMMIN(group + horizontal.rows, h + 7); r += 2) {
                    const int y0 = clamp(iy + r, 0, height - 1), y1 = clamp(iy + r + 1, 0, height - 1);
                    __m256i   v;
                    if (ix + 14 <= 0 || ix >= width - 1) {
                        const int edge = ix < 0 ? 0 : width - 1;
                        const int a    = (hi[y0 * hs + edge] << 2) | (lo[y0 * ls + edge] >> 6);
                        const int b    = (hi[y1 * hs + edge] << 2) | (lo[y1 * ls + edge] >> 6);
                        v              = _mm256_set_m128i(
                            _mm_set1_epi16(ROUND_POWER_OF_TWO((1 << (bd + FILTER_BITS - 1)) + (b << FILTER_BITS), hr)),
                            _mm_set1_epi16(ROUND_POWER_OF_TWO((1 << (bd + FILTER_BITS - 1)) + (a << FILTER_BITS), hr)));
                    } else {
                        const __m256i a = warp_samples(
                            hi + y0 * hs + column, lo + y0 * ls + column, loadmask, edge, interior);
                        const __m256i b = warp_samples(
                            hi + y1 * hs + column, lo + y1 * ls + column, loadmask, edge, interior);
                        const __m512i pix = _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);

                        if (beta && r > group) {
                            pos_x = _mm256_add_epi16(pos_x, step_x);
                            warp_weights(&top, pos_x, horizontal.size, wx);
                        }
                        __m512i samples[4];
                        for (int k = 0; k < 4; ++k) {
                            samples[k] = _mm512_permutexvar_epi16(_mm512_add_epi16(windows, _mm512_set1_epi16(2 * k)),
                                                                  pix);
                        }
                        const __m512i sum = warp_sum(samples[0],
                                                     samples[1],
                                                     samples[2],
                                                     samples[3],
                                                     wx,
                                                     (1 << (bd + FILTER_BITS - 1)) + (1 << (hr - 1)));
                        v                 = _mm512_cvtepi32_epi16(_mm512_sra_epi32(sum, _mm_cvtsi32_si128(hr)));
                    }
                    if (r) {
                        pairs[r / 2 - 1] = vertical_pair(previous, v);
                    }
                    previous = v;
                }
            }
            // An odd tile height needs only the low half of the last pair.
            if (h & 1) {
                pairs[(h + 5) / 2] = vertical_pair(previous, previous);
            }
            for (int group = 0; group < h; group += vertical.rows) {
                const WarpFilterWindow side  = warp_window(sy + group * delta, vertical);
                __m256i                pos_y = side.position;
                warp_weights(&side, pos_y, vertical.size, wy);
                for (int r = group; r < AOMMIN(group + vertical.rows, h); r += 2) {
                    const int rr = AOMMIN(r + 1, h - 1);
                    if (delta && r > group) {
                        pos_y = _mm256_add_epi16(pos_y, step_y);
                        warp_weights(&side, pos_y, vertical.size, wy);
                    }
                    const __m512i* row = pairs + r / 2;
                    const __m512i  sum = warp_sum(row[0], row[1], row[2], row[3], wy, (1 << offset) + (1 << (vr - 1)));
                    __m512i        value = _mm512_sra_epi32(sum, _mm_cvtsi32_si128(vr));
                    if (p->is_compound && p->do_average) {
                        const uint16_t* buf = p->dst + (y + r) * p->dst_stride + x;
                        const __m128i   a   = _mm_maskz_loadu_epi16(mask, buf),
                                      b     = _mm_maskz_loadu_epi16(mask, buf + (rr - r) * p->dst_stride);
                        const __m512i prev  = _mm512_cvtepu16_epi32(_mm256_set_m128i(b, a));
                        if (p->use_jnt_comp_avg) {
                            value = _mm512_srai_epi32(
                                _mm512_add_epi32(_mm512_mullo_epi32(prev, _mm512_set1_epi32(p->fwd_offset)),
                                                 _mm512_mullo_epi32(value, _mm512_set1_epi32(p->bck_offset))),
                                DIST_PRECISION_BITS);
                        } else {
                            value = _mm512_srli_epi32(_mm512_add_epi32(prev, value), 1);
                        }
                        value = _mm512_sra_epi32(
                            _mm512_add_epi32(value, _mm512_set1_epi32((bits ? 1 << (bits - 1) : 0) - bias)),
                            _mm_cvtsi32_si128(bits));
                    } else if (!p->is_compound) {
                        value = _mm512_sub_epi32(value, _mm512_set1_epi32(3 << (bd - 1)));
                    }
                    uint16_t* out;
                    int       stride;
                    if (p->is_compound && !p->do_average) {
                        out    = p->dst;
                        stride = p->dst_stride;
                    } else {
                        value  = _mm512_min_epi32(_mm512_max_epi32(value, _mm512_setzero_si512()),
                                                 _mm512_set1_epi32((1 << bd) - 1));
                        out    = pred;
                        stride = ps;
                    }
                    const __m256i v = _mm512_cvtepi32_epi16(value);
                    out += (y + r) * stride + x;
                    _mm_mask_storeu_epi16(out, mask, _mm256_castsi256_si128(v));
                    if (r + 1 < h) {
                        _mm_mask_storeu_epi16(out + stride, mask, _mm256_extracti128_si256(v, 1));
                    }
                }
            }
        }
    }
}

void svt_av1_highbd_warp_affine_avx512(const int32_t* mat, const uint8_t* hi, const uint8_t* lo, int width, int height,
                                       int hs, int ls, uint16_t* pred, int pc, int pr, int pw, int ph, int ps,
                                       int sxsub, int sysub, int bd, ConvolveParams* p, int16_t alpha, int16_t beta,
                                       int16_t gamma, int16_t delta) {
    // AV1 shear parameters have six low zero bits. Wider phase spans use
    // the eight-column horizontal schedule, which loads whole filter rows.
    const int shear = alpha | beta | gamma | delta;
    if (!shear || (shear & ((1 << WARP_PARAM_REDUCE_BITS) - 1)) ||
        7 * abs(alpha) + abs(beta) > (63 << WARPEDDIFF_PREC_BITS) ||
        7 * abs(gamma) + abs(delta) > (63 << WARPEDDIFF_PREC_BITS)) {
        svt_av1_highbd_warp_affine_avx2(mat,
                                        hi,
                                        lo,
                                        width,
                                        height,
                                        hs,
                                        ls,
                                        pred,
                                        pc,
                                        pr,
                                        pw,
                                        ph,
                                        ps,
                                        sxsub,
                                        sysub,
                                        bd,
                                        p,
                                        alpha,
                                        beta,
                                        gamma,
                                        delta);
        return;
    }
    warp_affine_paired_rows(
        mat, hi, lo, width, height, hs, ls, pred, pc, pr, pw, ph, ps, sxsub, sysub, bd, p, alpha, beta, gamma, delta);
}
#endif
