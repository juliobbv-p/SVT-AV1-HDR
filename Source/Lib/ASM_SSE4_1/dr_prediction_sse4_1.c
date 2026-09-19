/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

/* SSE4.1 tier for directional intra prediction zones 1 and 3 (lbd + highbd).
 * These dispatched C -> AVX2 with no SSE tier. The fractional interpolation
 * above[b]*(32-shift) + above[b+1]*shift is vectorized (128-bit) for the
 * common upsample==0 case with contiguous neighbours; the max-base clamp,
 * the upsample==1 (strided) case, and the tail are handled by the scalar
 * path, which is a verbatim copy of the C reference so the result is
 * bit-identical by construction. Zone 2 mixes above/left per pixel; its
 * above-region (upsample_above==0) has contiguous base and row-constant
 * shift, so that run is vectorized while the left-region, upsample==1, and
 * the tail stay on the verbatim-C scalar path. */

#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define RPO5(v) (((v) + 16) >> 5)

/* ---- Zone 1 (row-major: shift per row, base advances per column) ---- */
void svt_av1_dr_prediction_z1_sse4_1(uint8_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t* above,
                                     const uint8_t* left, int32_t upsample_above, int32_t dx, int32_t dy) {
    (void)left;
    (void)dy;
    const int32_t max_base_x = ((bw + bh) - 1) << upsample_above;
    const int32_t frac_bits  = 6 - upsample_above;
    const int32_t base_inc    = 1 << upsample_above;
    for (int32_t r = 0, x = dx; r < bh; ++r, dst += stride, x += dx) {
        int32_t       base  = x >> frac_bits;
        const int32_t shift = ((x << upsample_above) & 0x3F) >> 1;
        if (base >= max_base_x) {
            for (int32_t i = r; i < bh; ++i) {
                memset(dst, above[max_base_x], bw);
                dst += stride;
            }
            return;
        }
        int32_t c = 0;
        if (base_inc == 1) {
            int32_t       clim = max_base_x - base; // cols with base+c < max_base_x
            if (clim > bw)
                clim = bw;
            const __m128i s1  = _mm_set1_epi16((int16_t)shift);
            const __m128i s2  = _mm_set1_epi16((int16_t)(32 - shift));
            const __m128i r16 = _mm_set1_epi16(16);
            for (; c + 8 <= clim; c += 8) {
                const __m128i a0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(above + base + c)));
                const __m128i a1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(above + base + c + 1)));
                __m128i       v  = _mm_add_epi16(_mm_mullo_epi16(a0, s2), _mm_mullo_epi16(a1, s1));
                v                = _mm_srli_epi16(_mm_add_epi16(v, r16), 5);
                _mm_storel_epi64((__m128i*)(dst + c), _mm_packus_epi16(v, v));
            }
        }
        int32_t basec = base + c * base_inc;
        for (; c < bw; ++c, basec += base_inc) {
            if (basec < max_base_x)
                dst[c] = (uint8_t)RPO5(above[basec] * (32 - shift) + above[basec + 1] * shift);
            else
                dst[c] = above[max_base_x];
        }
    }
}

void svt_av1_highbd_dr_prediction_z1_sse4_1(uint16_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh,
                                            const uint16_t* above, const uint16_t* left, int32_t upsample_above,
                                            int32_t dx, int32_t dy, int32_t bd) {
    (void)left;
    (void)dy;
    (void)bd;
    const int32_t max_base_x = ((bw + bh) - 1) << upsample_above;
    const int32_t frac_bits  = 6 - upsample_above;
    const int32_t base_inc    = 1 << upsample_above;
    for (int32_t r = 0, x = dx; r < bh; ++r, dst += stride, x += dx) {
        int32_t       base  = x >> frac_bits;
        const int32_t shift = ((x << upsample_above) & 0x3F) >> 1;
        if (base >= max_base_x) {
            for (int32_t i = r; i < bh; ++i) {
                for (int32_t c = 0; c < bw; ++c) dst[c] = above[max_base_x];
                dst += stride;
            }
            return;
        }
        int32_t c = 0;
        if (base_inc == 1) {
            int32_t clim = max_base_x - base;
            if (clim > bw)
                clim = bw;
            const __m128i s1  = _mm_set1_epi32(shift);
            const __m128i s2  = _mm_set1_epi32(32 - shift);
            const __m128i r16 = _mm_set1_epi32(16);
            for (; c + 8 <= clim; c += 8) {
                const __m128i a0 = _mm_loadu_si128((const __m128i*)(above + base + c));
                const __m128i a1 = _mm_loadu_si128((const __m128i*)(above + base + c + 1));
                const __m128i lo = _mm_srli_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(a0), s2),
                                                _mm_mullo_epi32(_mm_cvtepu16_epi32(a1), s1)),
                                  r16),
                    5);
                const __m128i hi = _mm_srli_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a0, 8)), s2),
                                                _mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a1, 8)), s1)),
                                  r16),
                    5);
                _mm_storeu_si128((__m128i*)(dst + c), _mm_packus_epi32(lo, hi));
            }
        }
        int32_t basec = base + c * base_inc;
        for (; c < bw; ++c, basec += base_inc) {
            if (basec < max_base_x)
                dst[c] = (uint16_t)RPO5(above[basec] * (32 - shift) + above[basec + 1] * shift);
            else
                dst[c] = above[max_base_x];
        }
    }
}

/* ---- Zone 3 (column-major: shift per column, base advances per row) ---- */
void svt_av1_dr_prediction_z3_sse4_1(uint8_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t* above,
                                     const uint8_t* left, int32_t upsample_left, int32_t dx, int32_t dy) {
    (void)above;
    (void)dx;
    const int32_t max_base_y = (bw + bh - 1) << upsample_left;
    const int32_t frac_bits  = 6 - upsample_left;
    const int32_t base_inc    = 1 << upsample_left;
    for (int32_t c = 0, y = dy; c < bw; ++c, y += dy) {
        const int32_t base0 = y >> frac_bits;
        const int32_t shift = ((y << upsample_left) & 0x3F) >> 1;
        int32_t       r     = 0;
        if (base_inc == 1) {
            int32_t rlim = max_base_y - base0; // rows with base0+r < max_base_y
            if (rlim > bh)
                rlim = bh;
            const __m128i s1  = _mm_set1_epi16((int16_t)shift);
            const __m128i s2  = _mm_set1_epi16((int16_t)(32 - shift));
            const __m128i r16 = _mm_set1_epi16(16);
            uint8_t       tmp[16];
            for (; r + 8 <= rlim; r += 8) {
                const __m128i a0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(left + base0 + r)));
                const __m128i a1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(left + base0 + r + 1)));
                __m128i       v  = _mm_add_epi16(_mm_mullo_epi16(a0, s2), _mm_mullo_epi16(a1, s1));
                v                = _mm_srli_epi16(_mm_add_epi16(v, r16), 5);
                _mm_storel_epi64((__m128i*)tmp, _mm_packus_epi16(v, v));
                for (int32_t k = 0; k < 8; ++k) dst[(r + k) * stride + c] = tmp[k];
            }
        }
        int32_t base = base0 + r * base_inc;
        for (; r < bh; ++r, base += base_inc) {
            if (base < max_base_y)
                dst[r * stride + c] = (uint8_t)RPO5(left[base] * (32 - shift) + left[base + 1] * shift);
            else {
                for (; r < bh; ++r) dst[r * stride + c] = left[max_base_y];
                break;
            }
        }
    }
}

void svt_av1_highbd_dr_prediction_z3_sse4_1(uint16_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh,
                                            const uint16_t* above, const uint16_t* left, int32_t upsample_left,
                                            int32_t dx, int32_t dy, int32_t bd) {
    (void)above;
    (void)dx;
    (void)bd;
    const int32_t max_base_y = (bw + bh - 1) << upsample_left;
    const int32_t frac_bits  = 6 - upsample_left;
    const int32_t base_inc    = 1 << upsample_left;
    for (int32_t c = 0, y = dy; c < bw; ++c, y += dy) {
        const int32_t base0 = y >> frac_bits;
        const int32_t shift = ((y << upsample_left) & 0x3F) >> 1;
        int32_t       r     = 0;
        if (base_inc == 1) {
            int32_t rlim = max_base_y - base0;
            if (rlim > bh)
                rlim = bh;
            const __m128i s1  = _mm_set1_epi32(shift);
            const __m128i s2  = _mm_set1_epi32(32 - shift);
            const __m128i r16 = _mm_set1_epi32(16);
            uint16_t      tmp[8];
            for (; r + 8 <= rlim; r += 8) {
                const __m128i a0 = _mm_loadu_si128((const __m128i*)(left + base0 + r));
                const __m128i a1 = _mm_loadu_si128((const __m128i*)(left + base0 + r + 1));
                const __m128i lo = _mm_srli_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(a0), s2),
                                                _mm_mullo_epi32(_mm_cvtepu16_epi32(a1), s1)),
                                  r16),
                    5);
                const __m128i hi = _mm_srli_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a0, 8)), s2),
                                                _mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a1, 8)), s1)),
                                  r16),
                    5);
                _mm_storeu_si128((__m128i*)tmp, _mm_packus_epi32(lo, hi));
                for (int32_t k = 0; k < 8; ++k) dst[(r + k) * stride + c] = tmp[k];
            }
        }
        int32_t base = base0 + r * base_inc;
        for (; r < bh; ++r, base += base_inc) {
            if (base < max_base_y)
                dst[r * stride + c] = (uint16_t)RPO5(left[base] * (32 - shift) + left[base + 1] * shift);
            else {
                for (; r < bh; ++r) dst[r * stride + c] = left[max_base_y];
                break;
            }
        }
    }
}

/* ---- Zone 2 (90 < angle < 180: per-pixel above/left mix) ---- */
void svt_av1_dr_prediction_z2_sse4_1(uint8_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t* above,
                                     const uint8_t* left, int32_t upsample_above, int32_t upsample_left, int32_t dx,
                                     int32_t dy) {
    const int32_t min_base_x = -(1 << upsample_above);
    const int32_t fbx        = 6 - upsample_above;
    const int32_t fby        = 6 - upsample_left;
    for (int32_t r = 0; r < bh; ++r, dst += stride) {
        int32_t c = 0;
        // left region: base < min_base_x (scalar, verbatim C)
        for (; c < bw; ++c) {
            const int32_t x = (c << 6) - (r + 1) * dx;
            if ((x >> fbx) >= min_base_x)
                break;
            const int32_t yy = (r << 6) - (c + 1) * dy;
            const int32_t b2 = yy >> fby, sh2 = ((yy * (1 << upsample_left)) & 0x3F) >> 1;
            dst[c] = (uint8_t)RPO5(left[b2] * (32 - sh2) + left[b2 + 1] * sh2);
        }
        // above region, upsample_above==0: base = c + bx, shift row-constant
        if (upsample_above == 0) {
            const int32_t x0  = -(r + 1) * dx;
            const int32_t bx  = x0 >> 6;
            const int32_t sh1 = (x0 & 0x3F) >> 1;
            const __m128i s1 = _mm_set1_epi16((int16_t)sh1), s2 = _mm_set1_epi16((int16_t)(32 - sh1)),
                          r16 = _mm_set1_epi16(16);
            for (; c + 8 <= bw; c += 8) {
                const int32_t base = c + bx;
                const __m128i a0   = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(above + base)));
                const __m128i a1   = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(above + base + 1)));
                __m128i       v    = _mm_add_epi16(_mm_mullo_epi16(a0, s2), _mm_mullo_epi16(a1, s1));
                v                  = _mm_srli_epi16(_mm_add_epi16(v, r16), 5);
                _mm_storel_epi64((__m128i*)(dst + c), _mm_packus_epi16(v, v));
            }
        }
        // tail + upsample_above==1 above region (scalar, verbatim C)
        for (; c < bw; ++c) {
            const int32_t x    = (c << 6) - (r + 1) * dx;
            const int32_t base = x >> fbx;
            if (base >= min_base_x) {
                const int32_t sh = ((x * (1 << upsample_above)) & 0x3F) >> 1;
                dst[c]           = (uint8_t)RPO5(above[base] * (32 - sh) + above[base + 1] * sh);
            } else {
                const int32_t yy = (r << 6) - (c + 1) * dy;
                const int32_t b2 = yy >> fby, sh2 = ((yy * (1 << upsample_left)) & 0x3F) >> 1;
                dst[c]           = (uint8_t)RPO5(left[b2] * (32 - sh2) + left[b2 + 1] * sh2);
            }
        }
    }
}

void svt_av1_highbd_dr_prediction_z2_sse4_1(uint16_t* dst, ptrdiff_t stride, int32_t bw, int32_t bh,
                                            const uint16_t* above, const uint16_t* left, int32_t upsample_above,
                                            int32_t upsample_left, int32_t dx, int32_t dy, int32_t bd) {
    (void)bd;
    const int32_t min_base_x = -(1 << upsample_above);
    const int32_t fbx        = 6 - upsample_above;
    const int32_t fby        = 6 - upsample_left;
    for (int32_t r = 0; r < bh; ++r, dst += stride) {
        int32_t c = 0;
        for (; c < bw; ++c) {
            const int32_t x = (c << 6) - (r + 1) * dx;
            if ((x >> fbx) >= min_base_x)
                break;
            const int32_t yy = (r << 6) - (c + 1) * dy;
            const int32_t b2 = yy >> fby, sh2 = ((yy * (1 << upsample_left)) & 0x3F) >> 1;
            dst[c] = (uint16_t)RPO5(left[b2] * (32 - sh2) + left[b2 + 1] * sh2);
        }
        if (upsample_above == 0) {
            const int32_t x0  = -(r + 1) * dx;
            const int32_t bx  = x0 >> 6;
            const int32_t sh1 = (x0 & 0x3F) >> 1;
            const __m128i s1 = _mm_set1_epi32(sh1), s2 = _mm_set1_epi32(32 - sh1), r16 = _mm_set1_epi32(16);
            for (; c + 8 <= bw; c += 8) {
                const int32_t base = c + bx;
                const __m128i a0   = _mm_loadu_si128((const __m128i*)(above + base));
                const __m128i a1   = _mm_loadu_si128((const __m128i*)(above + base + 1));
                const __m128i lo   = _mm_srli_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(a0), s2),
                                                  _mm_mullo_epi32(_mm_cvtepu16_epi32(a1), s1)),
                                  r16),
                    5);
                const __m128i hi = _mm_srli_epi32(
                    _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a0, 8)), s2),
                                                _mm_mullo_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(a1, 8)), s1)),
                                  r16),
                    5);
                _mm_storeu_si128((__m128i*)(dst + c), _mm_packus_epi32(lo, hi));
            }
        }
        for (; c < bw; ++c) {
            const int32_t x    = (c << 6) - (r + 1) * dx;
            const int32_t base = x >> fbx;
            if (base >= min_base_x) {
                const int32_t sh = ((x * (1 << upsample_above)) & 0x3F) >> 1;
                dst[c]           = (uint16_t)RPO5(above[base] * (32 - sh) + above[base + 1] * sh);
            } else {
                const int32_t yy = (r << 6) - (c + 1) * dy;
                const int32_t b2 = yy >> fby, sh2 = ((yy * (1 << upsample_left)) & 0x3F) >> 1;
                dst[c]           = (uint16_t)RPO5(left[b2] * (32 - sh2) + left[b2 + 1] * sh2);
            }
        }
    }
}
