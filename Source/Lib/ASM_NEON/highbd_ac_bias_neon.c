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

#include <arm_neon.h>
#include <stdlib.h>

#include "common_dsp_rtcd.h"
#include "hadamard_path_neon.h"
#include "psy_energy_cache.h"

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
// A vector of twos that the optimiser cannot see through: it rewrites a
// multiply-subtract by a visible 2 into an addition and a subtraction.
// Create it once, outside the loops.
static AOM_FORCE_INLINE int16x8_t highbd_psy_two_neon(void) {
    int16x8_t v = vdupq_n_s16(2);
#if defined(__GNUC__) || defined(__clang__)
    __asm__("" : "+w"(v));
#endif
    return v;
}

// Energy-only transform of nonnegative 10-bit pixels. The horizontal stages
// commute without rounding, and SATD is independent of coefficient order, so
// all three stages combine adjacent lanes: a pairwise addition gives the sums
// of a stage for two rows at once, the differences follow from the sums and
// the odd lanes (a - b = a + b - 2b), and UMAXP folds the last stage
// (|a + b| + |a - b| = 2 * max(|a|, |b|)).
// Returns half the absolute coefficient sum of the tile as eight partial sums.
//
// Ranges for pixels 0..1023: every value before the last stage is a signed sum
// of at most 32 pixels, so its magnitude is at most 32 * 1023 = 32736 and the
// 16-bit lanes are exact. A lane of max0..max3 is half the absolute sum of two
// coefficients; the two narrow pairwise additions that follow leave half the
// absolute sum of eight coefficients c_k = sum_p h_k(p) * x(p) per lane, the
// h_k being orthogonal +-1 patterns over the 64 pixels p. With
// g = sum_k sign(c_k) * h_k, sum_k |c_k| = sum_p g(p) * x(p) is at most
// 1023 * (sum_p |g(p)| + sum_p g(p)) / 2: here sum_p g(p)^2 = 8 * 64, hence
// sum_p |g(p)| <= sqrt(64 * 512) < 181.1, and sum_p g(p) is +-64 when the DC
// pattern is among the eight and 0 otherwise. So sum_k |c_k| < 1023 * 122.6
// and each returned lane is below 62710: it fits 16 bits, but the next
// addition must widen.
static AOM_FORCE_INLINE uint16x8_t highbd_psy_hadamard_h_pass_neon(const int16x8_t s[8], const int16x8_t two) {
    const int16x8_t a0 = vpaddq_s16(s[0], s[1]);
    const int16x8_t a2 = vpaddq_s16(s[2], s[3]);
    const int16x8_t a4 = vpaddq_s16(s[4], s[5]);
    const int16x8_t a6 = vpaddq_s16(s[6], s[7]);
    const int16x8_t a1 = vmlsq_s16(a0, vuzp2q_s16(s[0], s[1]), two);
    const int16x8_t a3 = vmlsq_s16(a2, vuzp2q_s16(s[2], s[3]), two);
    const int16x8_t a5 = vmlsq_s16(a4, vuzp2q_s16(s[4], s[5]), two);
    const int16x8_t a7 = vmlsq_s16(a6, vuzp2q_s16(s[6], s[7]), two);

    const int16x8_t b0 = vabsq_s16(vpaddq_s16(a0, a1));
    const int16x8_t b1 = vabdq_s16(vuzp1q_s16(a0, a1), vuzp2q_s16(a0, a1));
    const int16x8_t b2 = vabsq_s16(vpaddq_s16(a2, a3));
    const int16x8_t b3 = vabdq_s16(vuzp1q_s16(a2, a3), vuzp2q_s16(a2, a3));
    const int16x8_t b4 = vabsq_s16(vpaddq_s16(a4, a5));
    const int16x8_t b5 = vabdq_s16(vuzp1q_s16(a4, a5), vuzp2q_s16(a4, a5));
    const int16x8_t b6 = vabsq_s16(vpaddq_s16(a6, a7));
    const int16x8_t b7 = vabdq_s16(vuzp1q_s16(a6, a7), vuzp2q_s16(a6, a7));

    const uint16x8_t max0 = vpmaxq_u16(vreinterpretq_u16_s16(b0), vreinterpretq_u16_s16(b1));
    const uint16x8_t max1 = vpmaxq_u16(vreinterpretq_u16_s16(b2), vreinterpretq_u16_s16(b3));
    const uint16x8_t max2 = vpmaxq_u16(vreinterpretq_u16_s16(b4), vreinterpretq_u16_s16(b5));
    const uint16x8_t max3 = vpmaxq_u16(vreinterpretq_u16_s16(b6), vreinterpretq_u16_s16(b7));
    return vpaddq_u16(vpaddq_u16(max0, max1), vpaddq_u16(max2, max3));
}

// Partial sums of half the absolute coefficient sum of an 8x8 tile, and the
// column sums of the tile (each at most 8 * 1023), whose total is its DC.
static AOM_FORCE_INLINE uint16x8_t highbd_psy_hadamard_8x8_neon(const uint16_t* src, ptrdiff_t stride,
                                                                const int16x8_t two, uint16x8_t* col_sums) {
    int16x8_t a[8];
    load_s16_8x8((const int16_t*)src, stride, &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
    hadamard_8x8_v_pass_neon(a);
    *col_sums = vreinterpretq_u16_s16(a[0]);
    return highbd_psy_hadamard_h_pass_neon(a, two);
}

static AOM_FORCE_INLINE int highbd_psy_energy_8x8_neon(const uint16_t* src, ptrdiff_t src_stride, const int16x8_t two) {
    uint16x8_t       col_sums;
    const uint16x8_t sums = highbd_psy_hadamard_8x8_neon(src, src_stride, two, &col_sums);

    const int satd   = vaddlvq_u16(sums) << 1;
    const int dc_sum = vaddlvq_u16(col_sums);
    return ((satd + 2) >> 2) - ((dc_sum + 2) >> 2);
}

// Two side-by-side 8x8 tiles, loaded row by row. Returns half the absolute
// coefficient sum of each tile in two lanes (left, left, right, right) and
// sets dc to the pairwise sums of their column sums (four lanes per tile, at
// most 16 * 1023 each).
static AOM_FORCE_INLINE uint32x4_t highbd_psy_hadamard_16x8_neon(const uint16_t* src, ptrdiff_t stride,
                                                                 const int16x8_t two, uint16x8_t* dc) {
    int16x8_t a[8], b[8];
    for (int i = 0; i < 8; i++) {
        a[i] = vld1q_s16((const int16_t*)src + i * stride);
        b[i] = vld1q_s16((const int16_t*)src + i * stride + 8);
    }
    hadamard_8x8_v_pass_neon(a);
    hadamard_8x8_v_pass_neon(b);
    *dc = vpaddq_u16(vreinterpretq_u16_s16(a[0]), vreinterpretq_u16_s16(b[0]));

    const uint16x8_t sums_a = highbd_psy_hadamard_h_pass_neon(a, two);
    const uint16x8_t sums_b = highbd_psy_hadamard_h_pass_neon(b, two);
    return vpaddq_u32(vpaddlq_u16(sums_a), vpaddlq_u16(sums_b));
}

// Energies of the four 8x8 tiles of a 16x16 group, in raster order.
static AOM_FORCE_INLINE int32x4_t highbd_psy_energy_16x16_neon(const uint16_t* src, ptrdiff_t src_stride,
                                                               const int16x8_t two) {
    uint16x8_t dc01, dc23;

    const uint32x4_t sum01 = highbd_psy_hadamard_16x8_neon(src, src_stride, two, &dc01);
    const uint32x4_t sum23 = highbd_psy_hadamard_16x8_neon(src + 8 * src_stride, src_stride, two, &dc23);

    // One more narrow pairwise addition leaves two lanes per tile, at most
    // 32 * 1023 each; the DC of a tile is at most 64 * 1023.
    const uint32x4_t dc  = vpaddlq_u16(vpaddq_u16(dc01, dc23));
    const uint32x4_t sum = vpaddq_u32(sum01, sum23);

    const uint32x4_t dc_energy   = vrshrq_n_u32(dc, 2);
    const uint32x4_t satd_energy = vrshrq_n_u32(sum, 1);

    return vreinterpretq_s32_u32(vsubq_u32(satd_energy, dc_energy));
}

// One transform of a 16x16 group already needs most vector registers: a
// function that inlines two of them keeps values on the stack.
static NOINLINE int32x4_t highbd_psy_energy_16x16_call_neon(const uint16_t* src, ptrdiff_t src_stride) {
    return highbd_psy_energy_16x16_neon(src, src_stride, highbd_psy_two_neon());
}

uint64_t svt_psy_distortion_hbd_neon(const uint16_t* input, const uint32_t input_stride, const uint16_t* recon,
                                     const uint32_t recon_stride, const uint32_t width, const uint32_t height) {
    uint64_t energy_gap = 0;

    if (width % 16 == 0 && height % 16 == 0) {
        int32x4_t energy_gap_16x16 = vdupq_n_s32(0);

        for (uint32_t j = 0; j < height; j += 16) {
            for (uint32_t i = 0; i < width; i += 16) {
                const int32x4_t input_energy = highbd_psy_energy_16x16_call_neon(input + j * input_stride + i,
                                                                                 input_stride);
                const int32x4_t recon_energy = highbd_psy_energy_16x16_call_neon(recon + j * recon_stride + i,
                                                                                 recon_stride);

                energy_gap_16x16 = vabaq_s32(energy_gap_16x16, input_energy, recon_energy);
            }
        }

        energy_gap += (uint64_t)vaddlvq_s32(energy_gap_16x16);

    } else if (width >= 8 && height >= 8) {
        const int16x8_t two = highbd_psy_two_neon();
        for (uint32_t j = 0; j < height; j += 8) {
            for (uint32_t i = 0; i < width; i += 8) {
                const int input_energy = highbd_psy_energy_8x8_neon(input + j * input_stride + i, input_stride, two);
                const int recon_energy = highbd_psy_energy_8x8_neon(recon + j * recon_stride + i, recon_stride, two);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    } else {
        for (uint32_t j = 0; j < height; j += 4) {
            for (uint32_t i = 0; i < width; i += 4) {
                const int input_energy = highbd_psy_energy_4x4_neon(input + j * input_stride + i, input_stride);
                const int recon_energy = highbd_psy_energy_4x4_neon(recon + j * recon_stride + i, recon_stride);

                energy_gap += abs(input_energy - recon_energy);
            }
        }
    }

    return energy_gap << 2;
}

// The hit paths read only the changing prediction/reconstruction and compact
// source energies. They perform no key checks or cache writes. Each block
// class has its own function, so that the frequent small blocks do not pay
// for the register saves of the 16x16 groups.
//
// A tile energy is at most 2^17, and a block of at most 128x128 pixels adds
// 64 differences per lane: the 32-bit accumulator cannot overflow.
static NOINLINE uint64_t highbd_psy_cached_gap_16x16_neon(const uint16_t* recon, uint32_t stride, uint32_t width,
                                                          uint32_t height, const int32_t* energy) {
    const int16x8_t two = highbd_psy_two_neon();
    int32x4_t       sum = vdupq_n_s32(0);
    for (uint32_t y = 0; y < height; y += 16) {
        for (uint32_t x = 0; x < width; x += 16) {
            const int32x4_t prediction = highbd_psy_energy_16x16_neon(recon + y * stride + x, stride, two);

            sum = vabaq_s32(sum, vld1q_s32(energy), prediction);
            energy += 4;
        }
    }
    return (uint64_t)vaddlvq_s32(sum) << 2;
}

static NOINLINE uint64_t highbd_psy_cached_gap_8x8_neon(const uint16_t* recon, uint32_t stride, uint32_t width,
                                                        uint32_t height, const int32_t* energy) {
    const int16x8_t two = highbd_psy_two_neon();
    uint64_t        gap = 0;
    for (uint32_t y = 0; y < height; y += 8) {
        for (uint32_t x = 0; x < width; x += 8) {
            gap += abs(*energy++ - highbd_psy_energy_8x8_neon(recon + y * stride + x, stride, two));
        }
    }
    return gap << 2;
}

static NOINLINE uint64_t highbd_psy_cached_gap_4x4_neon(const uint16_t* recon, uint32_t stride, uint32_t width,
                                                        uint32_t height, const int32_t* energy) {
    uint64_t gap = 0;
    for (uint32_t y = 0; y < height; y += 4) {
        for (uint32_t x = 0; x < width; x += 4) {
            gap += abs(*energy++ - highbd_psy_energy_4x4_neon(recon + y * stride + x, stride));
        }
    }
    return gap << 2;
}

// The miss paths fill the entry. For 16x16 groups the source energies go to
// the entry first and the hit path follows, one group transform per loop; the
// smaller tiles are filled while evaluating the first prediction.
static NOINLINE uint64_t highbd_psy_fill_gap_16x16_neon(const uint16_t* input, uint32_t input_stride,
                                                        const uint16_t* recon, uint32_t recon_stride, uint32_t width,
                                                        uint32_t height, int32_t* energy) {
    const int16x8_t two    = highbd_psy_two_neon();
    int32_t*        source = energy;
    for (uint32_t y = 0; y < height; y += 16) {
        for (uint32_t x = 0; x < width; x += 16) {
            vst1q_s32(source, highbd_psy_energy_16x16_neon(input + y * input_stride + x, input_stride, two));
            source += 4;
        }
    }
    return highbd_psy_cached_gap_16x16_neon(recon, recon_stride, width, height, energy);
}

static NOINLINE uint64_t highbd_psy_fill_gap_8x8_neon(const uint16_t* input, uint32_t input_stride,
                                                      const uint16_t* recon, uint32_t recon_stride, uint32_t width,
                                                      uint32_t height, int32_t* energy) {
    const int16x8_t two = highbd_psy_two_neon();
    uint64_t        gap = 0;
    for (uint32_t y = 0; y < height; y += 8) {
        for (uint32_t x = 0; x < width; x += 8) {
            const int source = highbd_psy_energy_8x8_neon(input + y * input_stride + x, input_stride, two);
            *energy++        = source;
            gap += abs(source - highbd_psy_energy_8x8_neon(recon + y * recon_stride + x, recon_stride, two));
        }
    }
    return gap << 2;
}

static NOINLINE uint64_t highbd_psy_fill_gap_4x4_neon(const uint16_t* input, uint32_t input_stride,
                                                      const uint16_t* recon, uint32_t recon_stride, uint32_t width,
                                                      uint32_t height, int32_t* energy) {
    uint64_t gap = 0;
    for (uint32_t y = 0; y < height; y += 4) {
        for (uint32_t x = 0; x < width; x += 4) {
            const int source = highbd_psy_energy_4x4_neon(input + y * input_stride + x, input_stride);
            *energy++        = source;
            gap += abs(source - highbd_psy_energy_4x4_neon(recon + y * recon_stride + x, recon_stride));
        }
    }
    return gap << 2;
}

uint64_t svt_psy_distortion_hbd_cached_neon(const uint16_t* input, uint32_t input_stride, const uint16_t* recon,
                                            uint32_t recon_stride, uint32_t width, uint32_t height,
                                            SvtPsyEnergyCache* cache) {
    if (!cache || width > 128 || height > 128) {
        if (cache) {
            cache->input = NULL;
        }
        return svt_psy_distortion_hbd_neon(input, input_stride, recon, recon_stride, width, height);
    }
    if (!width || !height) {
        return 0;
    }
    const uint64_t layout = ((uint64_t)input_stride << 32) | (width << 16) | height;
    const bool     hit    = cache->input == input && cache->layout == layout;

    // Blocks of one tile are the most frequent calls: no loops, and no call.
    if (width <= 4 && height <= 4) {
        int source = cache->energy[0];
        if (!hit) {
            source           = highbd_psy_energy_4x4_neon(input, input_stride);
            cache->input     = input;
            cache->layout    = layout;
            cache->energy[0] = source;
        }
        return (uint64_t)abs(source - highbd_psy_energy_4x4_neon(recon, recon_stride)) << 2;
    }
    if (width == 8 && height == 8) {
        const int16x8_t two    = highbd_psy_two_neon();
        int             source = cache->energy[0];
        if (!hit) {
            source           = highbd_psy_energy_8x8_neon(input, input_stride, two);
            cache->input     = input;
            cache->layout    = layout;
            cache->energy[0] = source;
        }
        return (uint64_t)abs(source - highbd_psy_energy_8x8_neon(recon, recon_stride, two)) << 2;
    }

    const bool groups_16x16 = width % 16 == 0 && height % 16 == 0;
    const bool tiles_8x8    = width >= 8 && height >= 8;
    if (hit) {
        if (groups_16x16) {
            return highbd_psy_cached_gap_16x16_neon(recon, recon_stride, width, height, cache->energy);
        }
        if (tiles_8x8) {
            return highbd_psy_cached_gap_8x8_neon(recon, recon_stride, width, height, cache->energy);
        }
        return highbd_psy_cached_gap_4x4_neon(recon, recon_stride, width, height, cache->energy);
    }

    cache->input  = input;
    cache->layout = layout;
    if (groups_16x16) {
        return highbd_psy_fill_gap_16x16_neon(input, input_stride, recon, recon_stride, width, height, cache->energy);
    }
    if (tiles_8x8) {
        return highbd_psy_fill_gap_8x8_neon(input, input_stride, recon, recon_stride, width, height, cache->energy);
    }
    return highbd_psy_fill_gap_4x4_neon(input, input_stride, recon, recon_stride, width, height, cache->energy);
}
#endif
