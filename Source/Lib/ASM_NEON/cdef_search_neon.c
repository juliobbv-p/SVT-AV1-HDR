/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "definitions.h"
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
#include <arm_neon.h>
#include "cdef.h"
#include "utility.h"

// Reuse unrounded primary/secondary contributions across the requested strengths.
// Round only their sum, and accumulate distortion without an intermediate image.
typedef struct {
    int16x8_t magnitude, difference;
} CdefDifference;

static INLINE CdefDifference cdef_difference(int16x8_t tap, int16x8_t center) {
    // Valid 10-bit pixels and the 0x7f7f sentinel have representable signed
    // differences. Form the magnitude independently of the signed subtraction.
    const int16x8_t difference = vsubq_s16(tap, center);
    return (CdefDifference){vabdq_s16(tap, center), difference};
}

static INLINE int16x8_t cdef_constrained(CdefDifference difference, int16x8_t threshold, int16x8_t damping) {
    const uint16x8_t bound = vqsubq_u16(vreinterpretq_u16_s16(threshold),
                                        vreinterpretq_u16_s16(vshlq_s16(difference.magnitude, damping)));
    const int16x8_t  limit = vreinterpretq_s16_u16(bound);
    return vmaxq_s16(vminq_s16(difference.difference, limit), vnegq_s16(limit));
}

static INLINE int16x8_t cdef_secondary(const CdefDifference taps[8], int strength, int damping) {
    const int16x8_t threshold = vdupq_n_s16(strength);
    const int16x8_t shift     = vdupq_n_s16(-AOMMAX(0, damping - get_msb(strength)));
    int16x8_t       near      = vaddq_s16(cdef_constrained(taps[0], threshold, shift),
                               cdef_constrained(taps[1], threshold, shift));
    near                      = vaddq_s16(
        near, vaddq_s16(cdef_constrained(taps[2], threshold, shift), cdef_constrained(taps[3], threshold, shift)));
    int16x8_t far = vaddq_s16(cdef_constrained(taps[4], threshold, shift), cdef_constrained(taps[5], threshold, shift));
    far           = vaddq_s16(
        far, vaddq_s16(cdef_constrained(taps[6], threshold, shift), cdef_constrained(taps[7], threshold, shift)));
    return vaddq_s16(vaddq_s16(near, near), far);
}

static INLINE int16x8_t cdef_primary(const CdefDifference taps[4], int strength, int damping) {
    if (!strength) {
        return vdupq_n_s16(0);
    }
    const int16x8_t threshold = vdupq_n_s16(strength);
    const int16x8_t shift     = vdupq_n_s16(-AOMMAX(0, damping - get_msb(strength)));
    const int16x8_t near      = vaddq_s16(cdef_constrained(taps[0], threshold, shift),
                                     cdef_constrained(taps[1], threshold, shift));
    const int16x8_t far       = vaddq_s16(cdef_constrained(taps[2], threshold, shift),
                                    cdef_constrained(taps[3], threshold, shift));
    // Primary weights are either (3,3) or (4,2). Adds/shifts avoid the
    // per-strength broadcasts and vector multipliers for these small constants.
    if (strength & 4) {
        const int16x8_t total = vaddq_s16(near, far);
        return vaddq_s16(total, vshlq_n_s16(total, 1));
    }
    return vshlq_n_s16(vaddq_s16(vaddq_s16(near, near), far), 1);
}

static INLINE void cdef_accumulate(int32x4_t* ssd, int16x8_t difference, int16x8_t sum, int16x8_t minimum,
                                   int16x8_t maximum, int clamp_output) {
    sum             = vaddq_s16(sum, vreinterpretq_s16_u16(vcltq_s16(sum, vdupq_n_s16(0))));
    int16x8_t value = vaddq_s16(difference, vrshrq_n_s16(sum, 4));
    if (clamp_output) {
        value = vminq_s16(vmaxq_s16(value, minimum), maximum);
    }
    *ssd = vmlal_s16(*ssd, vget_low_s16(value), vget_low_s16(value));
    *ssd = vmlal_high_s16(*ssd, value, value);
}

static INLINE int16x8_t cdef_load(const int16_t* src, int width, int row_stride) {
    return width == 8 ? vld1q_s16(src) : vcombine_s16(vld1_s16(src), vld1_s16(src + row_stride));
}

// The narrow path keeps the tap differences of 16 pixels (two vectors of the wide path) in 8-bit lanes.
// With m = |tap - center| and shift = max(0, damping - msb(threshold)), a constrained tap is
// min(m, max(0, threshold - (m >> shift))). It never exceeds its threshold, which the path requires to be at
// most 63 (60 is the largest 10-bit primary, 16 the largest secondary strength), so min(m, 255) can replace
// the first m. The shifted m is taken from s = min(m >> pre, 255) as s >> (shift - pre), with pre = 0 for
// damping <= 7 and pre = 1 for damping 8 (where shift >= 8 - msb(63) = 3). In both cases shift - pre is 0 or
// at most 7 - msb(threshold), so threshold <= 2^(msb + 1) - 1 = 255 >> (7 - msb) <= 255 >> (shift - pre):
// when s saturates, the saturated and the exact shifted magnitude both reach the threshold and the tap
// constrains to zero either way. That covers the 0x7f7f sentinel, whose difference to a pixel is over 31000.
typedef struct {
    uint8x16_t magnitude; // min(|tap - center|, 255)
    uint8x16_t scaled; // min(|tap - center| >> pre, 255)
    int8x16_t  sign; // -1 where tap < center, otherwise 1
} CdefTap8;

typedef struct {
    int16x8_t a, b;
} CdefSum16;

typedef struct {
    uint8x16_t threshold;
    int8x16_t  shift, near_weight, far_weight;
} CdefPrimary8;

static INLINE CdefTap8 cdef_tap8(const int16_t* src, int width, int row_stride, int vector_stride, int16x8_t center_a,
                                 int16x8_t center_b, bool halved, uint8x16_t* below) {
    const int16x8_t  tap_a     = cdef_load(src, width, row_stride);
    const int16x8_t  tap_b     = cdef_load(src + vector_stride, width, row_stride);
    const uint16x8_t wide_a    = vreinterpretq_u16_s16(vabdq_s16(tap_a, center_a));
    const uint16x8_t wide_b    = vreinterpretq_u16_s16(vabdq_s16(tap_b, center_b));
    const uint8x16_t magnitude = vqmovn_high_u16(vqmovn_u16(wide_a), wide_b);
    const uint8x16_t scaled    = halved ? vqshrn_high_n_u16(vqshrn_n_u16(wide_a, 1), wide_b, 1) : magnitude;
    const uint8x16_t below_a   = vreinterpretq_u8_u16(vcgtq_s16(center_a, tap_a));
    const uint8x16_t below_b   = vreinterpretq_u8_u16(vcgtq_s16(center_b, tap_b));
    *below                     = vuzp1q_u8(below_a, below_b);
    return (CdefTap8){magnitude, scaled, vreinterpretq_s8_u8(vorrq_u8(*below, vdupq_n_u8(1)))};
}

static INLINE int8x16_t cdef_constrained8(CdefTap8 tap, uint8x16_t threshold, int8x16_t shift, bool halved) {
    return vreinterpretq_s8_u8(
        vminq_u8(tap.magnitude, vqsubq_u8(threshold, vshlq_u8(halved ? tap.scaled : tap.magnitude, shift))));
}

static INLINE CdefSum16 cdef_secondary8(const CdefTap8 taps[8], int strength, int8x16_t shift, bool halved) {
    const uint8x16_t threshold = vdupq_n_u8(strength);
    // Four taps of at most 16 each per sum.
    int8x16_t near = vmulq_s8(cdef_constrained8(taps[0], threshold, shift, halved), taps[0].sign);
    int8x16_t far  = vmulq_s8(cdef_constrained8(taps[4], threshold, shift, halved), taps[4].sign);
    for (int k = 1; k < 4; ++k) {
        near = vmlaq_s8(near, cdef_constrained8(taps[k], threshold, shift, halved), taps[k].sign);
        far  = vmlaq_s8(far, cdef_constrained8(taps[k + 4], threshold, shift, halved), taps[k + 4].sign);
    }
    return (CdefSum16){vaddw_s8(vshll_n_s8(vget_low_s8(near), 1), vget_low_s8(far)),
                       vaddw_high_s8(vshll_high_n_s8(near, 1), far)};
}

static INLINE CdefSum16 cdef_primary8(const CdefTap8 taps[4], const CdefPrimary8* primary, bool halved) {
    // Two taps of at most 63 each per sum.
    int8x16_t near = vmulq_s8(cdef_constrained8(taps[0], primary->threshold, primary->shift, halved), taps[0].sign);
    int8x16_t far  = vmulq_s8(cdef_constrained8(taps[2], primary->threshold, primary->shift, halved), taps[2].sign);
    near = vmlaq_s8(near, cdef_constrained8(taps[1], primary->threshold, primary->shift, halved), taps[1].sign);
    far  = vmlaq_s8(far, cdef_constrained8(taps[3], primary->threshold, primary->shift, halved), taps[3].sign);
    return (CdefSum16){vmlal_s8(vmull_s8(vget_low_s8(near), vget_low_s8(primary->near_weight)),
                                vget_low_s8(far),
                                vget_low_s8(primary->far_weight)),
                       vmlal_high_s8(vmull_high_s8(near, primary->near_weight), far, primary->far_weight)};
}

static INLINE int32x4_t cdef_ssd16(int32x4_t ssd, int16x8_t value) {
    return vmlal_high_s16(vmlal_s16(ssd, vget_low_s16(value), vget_low_s16(value)), value, value);
}

static INLINE int16x8_t cdef_round16(int16x8_t sum) {
    return vsraq_n_s16(sum, sum, 15);
}

// True when a pixel of the block or of its two-pixel border is not a 10-bit sample.
static INLINE bool cdef_has_unavailable(const uint16_t* input, int width, int last_row) {
    uint16x8_t seen = vdupq_n_u16(0);
    for (int row = -2; row <= last_row + 2; ++row) {
        const uint16_t* src = input + row * CDEF_BSTRIDE - 2;
        seen                = vorrq_u16(seen, vld1q_u16(src));
        if (width == 8) {
            seen = vorrq_u16(seen, vld1q_u16(src + 4));
        }
    }
    return vmaxvq_u16(seen) > 1023;
}

static AOM_FORCE_INLINE bool cdef_search_block_narrow(const uint16_t* input, const uint16_t* target, int target_stride,
                                                      const CdefSearchPlan* plan, const uint8_t primary[16], int dir,
                                                      int damping, int width, int height, int subsampling_factor,
                                                      bool halved, uint64_t ssd[64]) {
    const int  row_step    = subsampling_factor * (8 / width);
    const bool has_primary = plan->primary_count && plan->primary[plan->primary_count - 1] != 0;
    const bool mixed       = plan->secondary_mask != 0;
    // Halved magnitudes serve damping 8 only. Vectors of the wide path are processed in pairs.
    if ((halved ? damping != 8 : damping > 7) ||
        (2 * row_step != height && 4 * row_step != height && 8 * row_step != height)) {
        return false;
    }
    CdefPrimary8 strengths[CDEF_PRI_STRENGTHS];
    for (int p = 0; p < plan->primary_count; ++p) {
        const int strength = primary[p];
        if (strength > 63) {
            return false;
        }
        if (strength) {
            strengths[p].threshold   = vdupq_n_u8(strength);
            strengths[p].shift       = vdupq_n_s8(halved - AOMMAX(0, damping - get_msb(strength)));
            strengths[p].near_weight = vdupq_n_s8(strength & 4 ? 3 : 4);
            strengths[p].far_weight  = vdupq_n_s8(strength & 4 ? 3 : 2);
        }
    }
    // The clamp of the mixed trials needs the largest available neighbor. The saturated magnitudes
    // cannot tell an unavailable tap from a large difference, so those blocks take the wide path.
    if (mixed && cdef_has_unavailable(input, width, height - subsampling_factor)) {
        return false;
    }
    // Directions 0 and 4 have the same secondary neighbor sets. Share their
    // loads and contributions with the configured-primary-zero trials.
    const bool own_zero_taps      = (dir & 3) && plan->zero_secondary_mask;
    const int  sec_mask           = plan->secondary_mask | ((dir & 3) ? 0 : plan->zero_secondary_mask);
    const int  primary_offsets[4] = {svt_aom_eb_cdef_directions[dir][0],
                                     -svt_aom_eb_cdef_directions[dir][0],
                                     svt_aom_eb_cdef_directions[dir][1],
                                     -svt_aom_eb_cdef_directions[dir][1]};
    int        offsets[8], zero_offsets[8];
    int8x16_t  secondary_shift[4];
    if (sec_mask | plan->zero_secondary_mask) {
        for (int k = 0; k < 2; ++k) {
            offsets[k * 4]          = svt_aom_eb_cdef_directions[dir + 2][k];
            offsets[k * 4 + 1]      = -offsets[k * 4];
            offsets[k * 4 + 2]      = svt_aom_eb_cdef_directions[dir - 2][k];
            offsets[k * 4 + 3]      = -offsets[k * 4 + 2];
            zero_offsets[k * 4]     = svt_aom_eb_cdef_directions[2][k];
            zero_offsets[k * 4 + 1] = -zero_offsets[k * 4];
            zero_offsets[k * 4 + 2] = svt_aom_eb_cdef_directions[-2][k];
            zero_offsets[k * 4 + 3] = -zero_offsets[k * 4 + 2];
        }
        for (int sec = 1; sec < 4; ++sec) {
            secondary_shift[sec] = vdupq_n_s8(halved - AOMMAX(0, damping - sec - 1));
        }
    }
    // Each lane accumulates at most 16 * 1023^2, safely within signed 32 bits.
    // Plans of up to eight trials are cleared inline.
    int32x4_t accum[64];
    for (int i = 0; i < 8; ++i) {
        accum[i] = vdupq_n_s32(0);
    }
    for (int i = 8; i < plan->strength_count; ++i) {
        accum[i] = vdupq_n_s32(0);
    }
    const int row_stride    = subsampling_factor * CDEF_BSTRIDE;
    const int vector_stride = row_step * CDEF_BSTRIDE;
    for (int row = 0; row < height; row += 2 * row_step) {
        const int16_t*  src      = (const int16_t*)(input + row * CDEF_BSTRIDE);
        const int16_t*  ref      = (const int16_t*)target + row * target_stride;
        const int16x8_t center_a = cdef_load(src, width, row_stride);
        const int16x8_t center_b = cdef_load(src + vector_stride, width, row_stride);
        // Distortion of the unfiltered pixels; the filter adds its rounded correction to it.
        const int16x8_t difference_a = vsubq_s16(center_a, cdef_load(ref, width, subsampling_factor * target_stride));
        const int16x8_t difference_b = vsubq_s16(
            center_b, cdef_load(ref + row_step * target_stride, width, subsampling_factor * target_stride));
        CdefTap8   pri_taps[4], sec_taps[8], zero_taps[8];
        uint8x16_t below, rise = vdupq_n_u8(0), fall = vdupq_n_u8(0);
        if (has_primary) {
            for (int k = 0; k < 4; ++k) {
                pri_taps[k] = cdef_tap8(
                    src + primary_offsets[k], width, row_stride, vector_stride, center_a, center_b, halved, &below);
                if (mixed) {
                    rise = vmaxq_u8(rise, vbicq_u8(pri_taps[k].magnitude, below));
                    fall = vmaxq_u8(fall, vandq_u8(pri_taps[k].magnitude, below));
                }
            }
        }
        if (sec_mask) {
            for (int k = 0; k < 8; ++k) {
                sec_taps[k] = cdef_tap8(
                    src + offsets[k], width, row_stride, vector_stride, center_a, center_b, halved, &below);
                if (mixed) {
                    rise = vmaxq_u8(rise, vbicq_u8(sec_taps[k].magnitude, below));
                    fall = vmaxq_u8(fall, vandq_u8(sec_taps[k].magnitude, below));
                }
            }
        }
        if (own_zero_taps) {
            for (int k = 0; k < 8; ++k) {
                zero_taps[k] = cdef_tap8(
                    src + zero_offsets[k], width, row_stride, vector_stride, center_a, center_b, halved, &below);
            }
        }
        // The rounded correction is at most (12 * 63 + 12 * 16 + 8) >> 4 = 59 in magnitude, so bounds
        // saturated at 127 clamp it like the neighbor range itself.
        const int8x16_t upper = vreinterpretq_s8_u8(vminq_u8(rise, vdupq_n_u8(127)));
        const int8x16_t lower = vnegq_s8(vreinterpretq_s8_u8(vminq_u8(fall, vdupq_n_u8(127))));
        CdefSum16       secondary[4], secondary_zero[4];
        for (int sec = 1; sec < 4; ++sec) {
            if (sec_mask & (1 << sec)) {
                secondary[sec] = cdef_secondary8(sec_taps, 2 << sec, secondary_shift[sec], halved);
            }
            if (plan->zero_secondary_mask & (1 << sec)) {
                secondary_zero[sec] = own_zero_taps ? cdef_secondary8(zero_taps, 2 << sec, secondary_shift[sec], halved)
                                                    : secondary[sec];
            }
        }
        for (int p = 0; p < plan->primary_count; ++p) {
            const bool configured_primary = plan->primary[p] != 0;
            CdefSum16  contribution;
            if (primary[p]) {
                contribution = cdef_primary8(pri_taps, &strengths[p], halved);
            }
            for (int slot = plan->group_start[p]; slot < plan->group_start[p + 1]; ++slot) {
                const int sec = plan->secondary[slot];
                int16x8_t value_a, value_b;
                if (!sec) {
                    value_a = primary[p] ? vrsraq_n_s16(difference_a, cdef_round16(contribution.a), 4) : difference_a;
                    value_b = primary[p] ? vrsraq_n_s16(difference_b, cdef_round16(contribution.b), 4) : difference_b;
                } else if (!primary[p]) {
                    const CdefSum16* sum = configured_primary ? &secondary[sec] : &secondary_zero[sec];
                    value_a              = vrsraq_n_s16(difference_a, cdef_round16(sum->a), 4);
                    value_b              = vrsraq_n_s16(difference_b, cdef_round16(sum->b), 4);
                } else {
                    // Single-family filtering is a convex constrained average, so
                    // only mixed nonzero contributions require the neighbor clamp.
                    const int16x8_t sum_a = cdef_round16(vaddq_s16(contribution.a, secondary[sec].a));
                    const int16x8_t sum_b = cdef_round16(vaddq_s16(contribution.b, secondary[sec].b));
                    int8x16_t       step  = vrshrn_high_n_s16(vrshrn_n_s16(sum_a, 4), sum_b, 4);
                    step                  = vminq_s8(vmaxq_s8(step, lower), upper);
                    value_a               = vaddw_s8(difference_a, vget_low_s8(step));
                    value_b               = vaddw_high_s8(difference_b, step);
                }
                accum[slot] = cdef_ssd16(cdef_ssd16(accum[slot], value_a), value_b);
            }
        }
    }
    for (int i = 0; i < plan->strength_count; ++i) {
        ssd[i] = (uint32_t)vaddvq_s32(accum[i]);
    }
    return true;
}

void svt_cdef_search_block_10bit_neon(const uint16_t* input, const uint16_t* target, int target_stride,
                                      const CdefSearchPlan* plan, const uint8_t primary[16], int dir, int damping,
                                      int bsize, int subsampling_factor, uint64_t ssd[64]) {
    const int width  = bsize == BLOCK_8X8 || bsize == BLOCK_8X4 ? 8 : 4;
    const int height = bsize == BLOCK_8X8 || bsize == BLOCK_4X8 ? 8 : 4;
    // Damping 8 is a luma value (chroma damping is one lower), so only the 8-wide path has that variant.
    if (width == 4) {
        if (cdef_search_block_narrow(
                input, target, target_stride, plan, primary, dir, damping, 4, height, subsampling_factor, false, ssd)) {
            return;
        }
    } else if (damping == 8) {
        if (cdef_search_block_narrow(
                input, target, target_stride, plan, primary, dir, damping, 8, height, subsampling_factor, true, ssd)) {
            return;
        }
    } else if (cdef_search_block_narrow(input,
                                        target,
                                        target_stride,
                                        plan,
                                        primary,
                                        dir,
                                        damping,
                                        8,
                                        height,
                                        subsampling_factor,
                                        false,
                                        ssd)) {
        return;
    }
    // Each lane accumulates at most 16 * 1023^2, safely within signed 32 bits.
    int32x4_t accum[64];
    for (int i = 0; i < plan->strength_count; ++i) {
        accum[i] = vdupq_n_s32(0);
    }
    const int  row_step           = subsampling_factor * (8 / width);
    const bool has_primary        = plan->primary_count && plan->primary[plan->primary_count - 1] != 0;
    const bool mixed              = plan->secondary_mask != 0;
    const int  primary_offsets[4] = {svt_aom_eb_cdef_directions[dir][0],
                                     -svt_aom_eb_cdef_directions[dir][0],
                                     svt_aom_eb_cdef_directions[dir][1],
                                     -svt_aom_eb_cdef_directions[dir][1]};
    int        offsets[8], zero_offsets[8];
    for (int k = 0; k < 2; ++k) {
        offsets[k * 4]          = svt_aom_eb_cdef_directions[dir + 2][k];
        offsets[k * 4 + 1]      = -offsets[k * 4];
        offsets[k * 4 + 2]      = svt_aom_eb_cdef_directions[dir - 2][k];
        offsets[k * 4 + 3]      = -offsets[k * 4 + 2];
        zero_offsets[k * 4]     = svt_aom_eb_cdef_directions[2][k];
        zero_offsets[k * 4 + 1] = -zero_offsets[k * 4];
        zero_offsets[k * 4 + 2] = svt_aom_eb_cdef_directions[-2][k];
        zero_offsets[k * 4 + 3] = -zero_offsets[k * 4 + 2];
    }
    for (int row = 0; row < height; row += row_step) {
        const int16_t*  src       = (const int16_t*)(input + row * CDEF_BSTRIDE);
        const int16x8_t center    = cdef_load(src, width, subsampling_factor * CDEF_BSTRIDE);
        const int16x8_t reference = cdef_load(
            (const int16_t*)target + row * target_stride, width, subsampling_factor * target_stride);
        // Doubling keeps 10-bit pixels ordered but makes the 0x7f7f sentinel negative.
        // Include the valid center so an off-frame tap can never win the maximum.
        int16x8_t      minimum = center, maximum = vaddq_s16(center, center);
        CdefDifference pri_taps[4], sec_taps[8], zero_taps[8];
        if (has_primary) {
            for (int k = 0; k < 4; ++k) {
                const int16x8_t tap = cdef_load(src + primary_offsets[k], width, subsampling_factor * CDEF_BSTRIDE);
                if (mixed) {
                    minimum = vminq_s16(minimum, tap);
                    maximum = vmaxq_s16(maximum, vaddq_s16(tap, tap));
                }
                pri_taps[k] = cdef_difference(tap, center);
            }
        }
        // Directions 0 and 4 have the same secondary neighbor sets. Share their
        // loads and contributions with the configured-primary-zero trials.
        const int sec_mask = plan->secondary_mask | ((dir & 3) ? 0 : plan->zero_secondary_mask);
        if (sec_mask) {
            for (int k = 0; k < 8; ++k) {
                const int16x8_t tap = cdef_load(src + offsets[k], width, subsampling_factor * CDEF_BSTRIDE);
                if (mixed) {
                    minimum = vminq_s16(minimum, tap);
                    maximum = vmaxq_s16(maximum, vaddq_s16(tap, tap));
                }
                sec_taps[k] = cdef_difference(tap, center);
            }
        }
        if ((dir & 3) && plan->zero_secondary_mask) {
            for (int k = 0; k < 8; ++k) {
                zero_taps[k] = cdef_difference(
                    cdef_load(src + zero_offsets[k], width, subsampling_factor * CDEF_BSTRIDE), center);
            }
        }
        maximum = vshrq_n_s16(maximum, 1);
        int16x8_t secondary[4], secondary_zero[4];
        secondary[0] = secondary_zero[0] = vdupq_n_s16(0);
        for (int sec = 1; sec < 4; ++sec) {
            if (sec_mask & (1 << sec)) {
                secondary[sec] = cdef_secondary(sec_taps, 2 << sec, damping);
            }
            if (plan->zero_secondary_mask & (1 << sec)) {
                secondary_zero[sec] = (dir & 3) ? cdef_secondary(zero_taps, 2 << sec, damping) : secondary[sec];
            }
        }
        // Translate into distortion space once per vector, preserving the clamp
        // while avoiding a target subtraction for every filtered candidate.
        const int16x8_t difference = vsubq_s16(center, reference);
        if (mixed) {
            minimum = vsubq_s16(minimum, reference);
            maximum = vsubq_s16(maximum, reference);
        }
        for (int p = 0; p < plan->primary_count; ++p) {
            const bool      configured_primary = plan->primary[p] != 0;
            const int16x8_t contribution       = configured_primary ? cdef_primary(pri_taps, primary[p], damping)
                                                                    : vdupq_n_s16(0);
            const int       first              = plan->group_start[p];
            if (plan->group_start[p + 1] - first == 4) {
                // A complete secondary group is always ordered 0,1,2,3. Keep
                // constant offsets here so dense grids retain the unrolled SSD loop.
                const int16x8_t* sec = configured_primary ? secondary : secondary_zero;
                cdef_accumulate(&accum[first], difference, contribution, minimum, maximum, 0);
                cdef_accumulate(
                    &accum[first + 1], difference, vaddq_s16(contribution, sec[1]), minimum, maximum, primary[p] != 0);
                cdef_accumulate(
                    &accum[first + 2], difference, vaddq_s16(contribution, sec[2]), minimum, maximum, primary[p] != 0);
                cdef_accumulate(
                    &accum[first + 3], difference, vaddq_s16(contribution, sec[3]), minimum, maximum, primary[p] != 0);
                continue;
            }
            for (int slot = plan->group_start[p]; slot < plan->group_start[p + 1]; ++slot) {
                const int       sec = plan->secondary[slot];
                const int16x8_t sum = vaddq_s16(contribution,
                                                configured_primary ? secondary[sec] : secondary_zero[sec]);
                // Single-family filtering is a convex constrained average, so
                // only mixed nonzero contributions require the neighbor clamp.
                cdef_accumulate(&accum[slot], difference, sum, minimum, maximum, primary[p] && sec);
            }
        }
    }
    for (int i = 0; i < plan->strength_count; ++i) {
        ssd[i] = (uint32_t)vaddvq_s32(accum[i]);
    }
}

#endif // CONFIG_ENABLE_HIGH_BIT_DEPTH
