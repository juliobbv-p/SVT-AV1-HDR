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

// The wide path keeps every quantity in 16-bit lanes and takes any block.
static void cdef_search_block_wide(const uint16_t* input, const uint16_t* target, int target_stride,
                                   const CdefSearchPlan* plan, const uint8_t primary[16], int dir, int damping,
                                   int width, int height, int subsampling_factor, uint64_t ssd[64]) {
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

// A primary strength in 8-bit lanes: threshold, shift of the (for damping 8 halved) magnitude and tap weights.
typedef struct {
    uint8_t threshold;
    int8_t  shift, near_weight, far_weight;
} CdefStrength8;

#define CDEF_MSB6(t) ((t) >= 32 ? 5 : (t) >= 16 ? 4 : (t) >= 8 ? 3 : (t) >= 4 ? 2 : (t) >= 2 ? 1 : 0)
#define CDEF_S1(d, t) \
    {(t), ((d) == 8) - ((d) > CDEF_MSB6(t) ? (d) - CDEF_MSB6(t) : 0), ((t) & 4) ? 3 : 4, ((t) & 4) ? 3 : 2}
#define CDEF_S8(d, t)                                                                                  \
    CDEF_S1(d, t), CDEF_S1(d, (t) + 1), CDEF_S1(d, (t) + 2), CDEF_S1(d, (t) + 3), CDEF_S1(d, (t) + 4), \
        CDEF_S1(d, (t) + 5), CDEF_S1(d, (t) + 6), CDEF_S1(d, (t) + 7)
#define CDEF_S64(d)  \
    {CDEF_S8(d, 0),  \
     CDEF_S8(d, 8),  \
     CDEF_S8(d, 16), \
     CDEF_S8(d, 24), \
     CDEF_S8(d, 32), \
     CDEF_S8(d, 40), \
     CDEF_S8(d, 48), \
     CDEF_S8(d, 56)}
// By damping (0 to 8) and primary strength (1 to 63; the threshold 0 of entry 0 stands for no primary filter).
static const CdefStrength8 cdef_strength8[9][64] = {CDEF_S64(0),
                                                    CDEF_S64(1),
                                                    CDEF_S64(2),
                                                    CDEF_S64(3),
                                                    CDEF_S64(4),
                                                    CDEF_S64(5),
                                                    CDEF_S64(6),
                                                    CDEF_S64(7),
                                                    CDEF_S64(8)};
#undef CDEF_S64
#undef CDEF_S8
#undef CDEF_S1
#undef CDEF_MSB6

// What the narrow path derives from a plan.
typedef struct {
    int8x16_t secondary_shift[4];
    uint8_t   group_mask[CDEF_PRI_STRENGTHS]; // secondary strengths of each group, one bit each
    uint8_t   first; // first group with a nonzero configured primary strength
    bool      has_primary, mixed;
} CdefNarrow;

// False when the blocks have to take the wide path.
static bool cdef_narrow_setup(CdefNarrow* narrow, const CdefSearchPlan* plan, int damping, int width, int height,
                              int subsampling_factor, bool halved) {
    const int row_step = subsampling_factor * (8 / width);
    // Halved magnitudes serve damping 8 only. Vectors of the wide path are processed in pairs.
    if ((halved ? damping != 8 : damping > 7) || damping < 0 ||
        (2 * row_step != height && 4 * row_step != height && 8 * row_step != height)) {
        return false;
    }
    for (int sec = 1; sec < 4; ++sec) {
        narrow->secondary_shift[sec] = vdupq_n_s8(halved - AOMMAX(0, damping - sec - 1));
    }
    narrow->first       = plan->primary_count && !plan->primary[0];
    narrow->has_primary = plan->primary_count && plan->primary[plan->primary_count - 1] != 0;
    narrow->mixed       = plan->secondary_mask != 0;
    for (int p = 0; p < plan->primary_count; ++p) {
        int mask = 0;
        for (int slot = plan->group_start[p]; slot < plan->group_start[p + 1]; ++slot) {
            mask |= 1 << plan->secondary[slot];
        }
        narrow->group_mask[p] = mask;
    }
    return true;
}

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

// Adds four constrained taps to the sums of the secondary strengths in the mask. Four taps of at most 16 each
// per sum.
static AOM_FORCE_INLINE void cdef_secondary_taps8(int8x16_t sums[4], uint8x16_t* rise, uint8x16_t* fall,
                                                  const int16_t* src, const int offsets[4], int mask,
                                                  const int8x16_t shifts[4], int width, int row_stride,
                                                  int vector_stride, int16x8_t center_a, int16x8_t center_b,
                                                  bool halved, bool bounds) {
    for (int k = 0; k < 4; ++k) {
        uint8x16_t     below;
        const CdefTap8 tap = cdef_tap8(
            src + offsets[k], width, row_stride, vector_stride, center_a, center_b, halved, &below);
        if (bounds) {
            *rise = vmaxq_u8(*rise, vbicq_u8(tap.magnitude, below));
            *fall = vmaxq_u8(*fall, vandq_u8(tap.magnitude, below));
        }
        if (mask & 2) {
            sums[1] = vmlaq_s8(sums[1], cdef_constrained8(tap, vdupq_n_u8(4), shifts[1], halved), tap.sign);
        }
        if (mask & 4) {
            sums[2] = vmlaq_s8(sums[2], cdef_constrained8(tap, vdupq_n_u8(8), shifts[2], halved), tap.sign);
        }
        if (mask & 8) {
            sums[3] = vmlaq_s8(sums[3], cdef_constrained8(tap, vdupq_n_u8(16), shifts[3], halved), tap.sign);
        }
    }
}

// The weighted sums of the secondary strengths in the mask over the eight taps at the offsets (near taps first).
static AOM_FORCE_INLINE void cdef_secondary8(CdefSum16 sums[4], uint8x16_t* rise, uint8x16_t* fall, const int16_t* src,
                                             const int offsets[8], int mask, const int8x16_t shifts[4], int width,
                                             int row_stride, int vector_stride, int16x8_t center_a, int16x8_t center_b,
                                             bool halved, bool bounds) {
    int8x16_t near[4] = {vdupq_n_s8(0), vdupq_n_s8(0), vdupq_n_s8(0), vdupq_n_s8(0)};
    int8x16_t far[4]  = {vdupq_n_s8(0), vdupq_n_s8(0), vdupq_n_s8(0), vdupq_n_s8(0)};
    cdef_secondary_taps8(near,
                         rise,
                         fall,
                         src,
                         offsets,
                         mask,
                         shifts,
                         width,
                         row_stride,
                         vector_stride,
                         center_a,
                         center_b,
                         halved,
                         bounds);
    cdef_secondary_taps8(far,
                         rise,
                         fall,
                         src,
                         offsets + 4,
                         mask,
                         shifts,
                         width,
                         row_stride,
                         vector_stride,
                         center_a,
                         center_b,
                         halved,
                         bounds);
    for (int sec = 1; sec < 4; ++sec) {
        if (mask & (1 << sec)) {
            sums[sec] = (CdefSum16){vaddw_s8(vshll_n_s8(vget_low_s8(near[sec]), 1), vget_low_s8(far[sec])),
                                    vaddw_high_s8(vshll_high_n_s8(near[sec], 1), far[sec])};
        }
    }
}

static AOM_FORCE_INLINE CdefSum16 cdef_primary8(const CdefTap8 taps[4], const CdefStrength8* strength, bool halved) {
    const uint8x16_t threshold   = vld1q_dup_u8(&strength->threshold);
    const int8x16_t  shift       = vld1q_dup_s8(&strength->shift);
    const int8x16_t  near_weight = vld1q_dup_s8(&strength->near_weight);
    const int8x16_t  far_weight  = vld1q_dup_s8(&strength->far_weight);
    // Two taps of at most 63 each per sum.
    int8x16_t near = vmulq_s8(cdef_constrained8(taps[0], threshold, shift, halved), taps[0].sign);
    int8x16_t far  = vmulq_s8(cdef_constrained8(taps[2], threshold, shift, halved), taps[2].sign);
    near           = vmlaq_s8(near, cdef_constrained8(taps[1], threshold, shift, halved), taps[1].sign);
    far            = vmlaq_s8(far, cdef_constrained8(taps[3], threshold, shift, halved), taps[3].sign);
    return (CdefSum16){
        vmlal_s8(vmull_s8(vget_low_s8(near), vget_low_s8(near_weight)), vget_low_s8(far), vget_low_s8(far_weight)),
        vmlal_high_s8(vmull_high_s8(near, near_weight), far, far_weight)};
}

static INLINE int32x4_t cdef_ssd16(int32x4_t ssd, int16x8_t value) {
    return vmlal_high_s16(vmlal_s16(ssd, vget_low_s16(value), vget_low_s16(value)), value, value);
}

static INLINE int16x8_t cdef_round16(int16x8_t sum) {
    return vsraq_n_s16(sum, sum, 15);
}

// Adds the squared error of one trial over the 16 pixels.
static AOM_FORCE_INLINE void cdef_trial(int32x4_t* accum, int16x8_t value_a, int16x8_t value_b) {
    *accum = cdef_ssd16(cdef_ssd16(*accum, value_a), value_b);
}

// The trials of a group without primary contribution: the unfiltered pixels and the secondary filters.
// Single-family filtering is a convex constrained average, so it needs no neighbor clamp.
static AOM_FORCE_INLINE void cdef_group_secondary(int32x4_t* accum, int mask, int16x8_t difference_a,
                                                  int16x8_t difference_b, const CdefSum16 sums[4],
                                                  bool with_secondary) {
    if (mask & 1) {
        cdef_trial(accum++, difference_a, difference_b);
    }
    if (with_secondary) {
        for (int sec = 1; sec < 4; ++sec) {
            if (mask & (1 << sec)) {
                cdef_trial(accum++,
                           vrsraq_n_s16(difference_a, cdef_round16(sums[sec].a), 4),
                           vrsraq_n_s16(difference_b, cdef_round16(sums[sec].b), 4));
            }
        }
    }
}

// The trials of a group with a primary contribution. Only mixed nonzero contributions require the neighbor
// clamp. The rounded correction is at most (12 * 63 + 12 * 16 + 8) >> 4 = 59 in magnitude, so bounds saturated
// at 127 clamp it like the neighbor range itself.
static AOM_FORCE_INLINE void cdef_group_primary(int32x4_t* accum, int mask, int16x8_t difference_a,
                                                int16x8_t difference_b, CdefSum16 primary, const CdefSum16 sums[4],
                                                int8x16_t lower, int8x16_t upper, bool with_secondary) {
    if (mask & 1) {
        cdef_trial(accum++,
                   vrsraq_n_s16(difference_a, cdef_round16(primary.a), 4),
                   vrsraq_n_s16(difference_b, cdef_round16(primary.b), 4));
    }
    if (with_secondary) {
        for (int sec = 1; sec < 4; ++sec) {
            if (mask & (1 << sec)) {
                const int16x8_t sum_a = cdef_round16(vaddq_s16(primary.a, sums[sec].a));
                const int16x8_t sum_b = cdef_round16(vaddq_s16(primary.b, sums[sec].b));
                int8x16_t       step  = vrshrn_high_n_s16(vrshrn_n_s16(sum_a, 4), sum_b, 4);
                step                  = vminq_s8(vmaxq_s8(step, lower), upper);
                cdef_trial(accum++, vaddw_s8(difference_a, vget_low_s8(step)), vaddw_high_s8(difference_b, step));
            }
        }
    }
}

// The trials of one group of a nonzero configured primary strength (zero for some blocks).
static AOM_FORCE_INLINE void cdef_group(int32x4_t* accum, int mask, int16x8_t difference_a, int16x8_t difference_b,
                                        const CdefTap8 taps[4], const CdefStrength8* strength, const CdefSum16 sums[4],
                                        int8x16_t lower, int8x16_t upper, bool halved, bool with_secondary) {
    if (!strength->threshold) {
        cdef_group_secondary(accum, mask, difference_a, difference_b, sums, with_secondary);
    } else {
        cdef_group_primary(accum,
                           mask,
                           difference_a,
                           difference_b,
                           cdef_primary8(taps, strength, halved),
                           sums,
                           lower,
                           upper,
                           with_secondary);
    }
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

// The tap offsets of a block: primary taps along its direction, secondary taps (near taps first) across it and,
// for the group of the configured primary strength zero, across direction 0.
typedef struct {
    int  primary[4], secondary[8], zero[8];
    int  secondary_mask; // secondary strengths to filter with the block's secondary taps
    bool own_zero_taps;
} CdefBlockTaps;

static AOM_FORCE_INLINE void cdef_block_taps(CdefBlockTaps* taps, const CdefSearchPlan* plan, int dir,
                                             bool with_secondary) {
    taps->primary[0] = svt_aom_eb_cdef_directions[dir][0];
    taps->primary[1] = -taps->primary[0];
    taps->primary[2] = svt_aom_eb_cdef_directions[dir][1];
    taps->primary[3] = -taps->primary[2];
    // Directions 0 and 4 have the same secondary neighbor sets. Share their
    // loads and contributions with the configured-primary-zero trials.
    taps->own_zero_taps  = with_secondary && (dir & 3) && plan->zero_secondary_mask;
    taps->secondary_mask = with_secondary ? plan->secondary_mask | ((dir & 3) ? 0 : plan->zero_secondary_mask) : 0;
    if (with_secondary) {
        for (int k = 0; k < 2; ++k) {
            taps->secondary[k * 4]     = svt_aom_eb_cdef_directions[dir + 2][k];
            taps->secondary[k * 4 + 1] = -taps->secondary[k * 4];
            taps->secondary[k * 4 + 2] = svt_aom_eb_cdef_directions[dir - 2][k];
            taps->secondary[k * 4 + 3] = -taps->secondary[k * 4 + 2];
            taps->zero[k * 4]          = svt_aom_eb_cdef_directions[2][k];
            taps->zero[k * 4 + 1]      = -taps->zero[k * 4];
            taps->zero[k * 4 + 2]      = svt_aom_eb_cdef_directions[-2][k];
            taps->zero[k * 4 + 3]      = -taps->zero[k * 4 + 2];
        }
    }
}

// Adds the squared errors of the trials of 16 pixels to the sums of the plan's slots, four 32-bit partial sums
// per slot. The strengths are those of the block's damping; primary holds the strength of each group.
static AOM_FORCE_INLINE void cdef_search_unit(const int16_t* src, const int16_t* ref, int target_stride,
                                              const CdefSearchPlan* plan, const CdefNarrow* narrow,
                                              const CdefStrength8* strengths, const uint8_t primary[16],
                                              const CdefBlockTaps* offsets, int width, int subsampling_factor,
                                              bool halved, bool with_secondary, int32x4_t* accum) {
    const bool      mixed         = with_secondary && narrow->mixed;
    const int       row_step      = subsampling_factor * (8 / width);
    const int       row_stride    = subsampling_factor * CDEF_BSTRIDE;
    const int       vector_stride = row_step * CDEF_BSTRIDE;
    const int16x8_t center_a      = cdef_load(src, width, row_stride);
    const int16x8_t center_b      = cdef_load(src + vector_stride, width, row_stride);
    // Distortion of the unfiltered pixels; the filter adds its rounded correction to it.
    const int16x8_t difference_a = vsubq_s16(center_a, cdef_load(ref, width, subsampling_factor * target_stride));
    const int16x8_t difference_b = vsubq_s16(
        center_b, cdef_load(ref + row_step * target_stride, width, subsampling_factor * target_stride));
    const CdefSum16 none    = {vdupq_n_s16(0), vdupq_n_s16(0)};
    CdefSum16       sums[4] = {none, none, none, none};
    uint8x16_t      rise = vdupq_n_u8(0), fall = vdupq_n_u8(0);
    if (offsets->secondary_mask) {
        cdef_secondary8(sums,
                        &rise,
                        &fall,
                        src,
                        offsets->secondary,
                        offsets->secondary_mask,
                        narrow->secondary_shift,
                        width,
                        row_stride,
                        vector_stride,
                        center_a,
                        center_b,
                        halved,
                        mixed);
    }
    if (narrow->first) {
        // The group of the configured primary strength zero filters along direction 0.
        if (offsets->own_zero_taps) {
            CdefSum16 zero_sums[4] = {none, none, none, none};
            cdef_secondary8(zero_sums,
                            &rise,
                            &fall,
                            src,
                            offsets->zero,
                            plan->zero_secondary_mask,
                            narrow->secondary_shift,
                            width,
                            row_stride,
                            vector_stride,
                            center_a,
                            center_b,
                            halved,
                            false);
            cdef_group_secondary(accum, narrow->group_mask[0], difference_a, difference_b, zero_sums, with_secondary);
        } else {
            cdef_group_secondary(accum, narrow->group_mask[0], difference_a, difference_b, sums, with_secondary);
        }
    }
    if (!narrow->has_primary) {
        return;
    }
    CdefTap8 taps[4];
    for (int k = 0; k < 4; ++k) {
        uint8x16_t below;
        taps[k] = cdef_tap8(
            src + offsets->primary[k], width, row_stride, vector_stride, center_a, center_b, halved, &below);
        if (mixed) {
            rise = vmaxq_u8(rise, vbicq_u8(taps[k].magnitude, below));
            fall = vmaxq_u8(fall, vandq_u8(taps[k].magnitude, below));
        }
    }
    // Bounds of the correction of the mixed trials, see cdef_group_primary().
    const int8x16_t upper = vreinterpretq_s8_u8(vminq_u8(rise, vdupq_n_u8(127)));
    const int8x16_t lower = vnegq_s8(vreinterpretq_s8_u8(vminq_u8(fall, vdupq_n_u8(127))));
    for (int p = narrow->first; p < plan->primary_count; ++p) {
        cdef_group(accum + plan->group_start[p],
                   narrow->group_mask[p],
                   difference_a,
                   difference_b,
                   taps,
                   &strengths[primary[p]],
                   sums,
                   lower,
                   upper,
                   halved,
                   with_secondary);
    }
}

// A block of the list (or, without a list, the block at the input) and what its trials depend on.
typedef struct {
    const uint16_t* in;
    const uint16_t* target;
    const uint8_t*  strength; // of each group
    int             dir;
} CdefBlock;

static AOM_FORCE_INLINE CdefBlock cdef_block(const uint16_t* in, const uint16_t* target, int target_stride,
                                             uint8_t (*dir)[CDEF_NBLOCKS], int32_t (*var)[CDEF_NBLOCKS],
                                             const CdefList* dlist, int index, const CdefSearchPlan* plan,
                                             const uint8_t* primary, int fixed_dir, int width, int height) {
    const int  by = dlist ? dlist[index].by : 0, bx = dlist ? dlist[index].bx : 0;
    const bool has_primary = plan->primary_count && plan->primary[plan->primary_count - 1] != 0;
    return (CdefBlock){in + height * by * CDEF_BSTRIDE + width * bx,
                       target + height * by * target_stride + width * bx,
                       primary ? primary : plan->luma_primary[svt_cdef_variance_class(var[by][bx])],
                       !dlist            ? fixed_dir
                           : has_primary ? dir[by][bx]
                                         : 0};
}

// Searches the blocks of a list (or, without a list, the block at the input) with sums that live across blocks.
// primary holds the strengths of the plan's groups when they are the same for every block; otherwise they
// follow each block's directional variance.
static AOM_FORCE_INLINE void cdef_search_blocks(const uint16_t* in, const uint16_t* target, int target_stride,
                                                uint8_t (*dir)[CDEF_NBLOCKS], int32_t (*var)[CDEF_NBLOCKS],
                                                const CdefList* dlist, int count, const CdefSearchPlan* plan,
                                                const uint8_t* primary, int fixed_dir, int damping, int width,
                                                int height, int subsampling_factor, bool halved, bool with_secondary,
                                                uint64_t ssd[TOTAL_STRENGTHS]) {
    CdefNarrow narrow;
    bool       usable = cdef_narrow_setup(&narrow, plan, damping, width, height, subsampling_factor, halved);
    if (primary) {
        for (int p = 0; p < plan->primary_count; ++p) {
            usable &= primary[p] <= 63;
        }
    }
    const CdefStrength8* strengths = cdef_strength8[usable ? damping : 0];
    const bool           mixed     = usable && with_secondary && narrow.mixed;
    int32x4_t            accum[TOTAL_STRENGTHS];
    for (int i = 0; i < plan->strength_count; ++i) {
        accum[i] = vdupq_n_s32(0);
        ssd[i]   = 0;
    }
    // 16 pixels add four squared errors to each 32-bit partial sum, so 64 such units of errors below 4096 fit
    // an unsigned sum (10-bit pixels differ by less than 1024).
    const int row_step = subsampling_factor * (8 / width);
    const int chunk    = usable ? 128 * row_step / height : 64;
    for (int start = 0; start < count; start += chunk) {
        const int end = AOMMIN(count, start + chunk);
        for (int bi = start; bi < end; ++bi) {
            const CdefBlock block = cdef_block(
                in, target, target_stride, dir, var, dlist, bi, plan, primary, fixed_dir, width, height);
            // The clamp of the mixed trials needs the largest available neighbor. The saturated magnitudes
            // cannot tell an unavailable tap from a large difference, so those blocks take the wide path.
            if (!usable || (mixed && cdef_has_unavailable(block.in, width, height - subsampling_factor))) {
                uint64_t one[TOTAL_STRENGTHS];
                cdef_search_block_wide(block.in,
                                       block.target,
                                       target_stride,
                                       plan,
                                       block.strength,
                                       block.dir,
                                       damping,
                                       width,
                                       height,
                                       subsampling_factor,
                                       one);
                for (int i = 0; i < plan->strength_count; ++i) {
                    ssd[i] += one[i];
                }
                continue;
            }
            CdefBlockTaps offsets;
            cdef_block_taps(&offsets, plan, block.dir, with_secondary);
            for (int row = 0; row < height; row += 2 * row_step) {
                const int16_t* src = (const int16_t*)(block.in + row * CDEF_BSTRIDE);
                const int16_t* ref = (const int16_t*)block.target + row * target_stride;
                cdef_search_unit(src,
                                 ref,
                                 target_stride,
                                 plan,
                                 &narrow,
                                 strengths,
                                 block.strength,
                                 &offsets,
                                 width,
                                 subsampling_factor,
                                 halved,
                                 with_secondary,
                                 accum);
            }
        }
        for (int i = 0; i < plan->strength_count; ++i) {
            ssd[i] += vaddlvq_u32(vreinterpretq_u32_s32(accum[i]));
            accum[i] = vdupq_n_s32(0);
        }
    }
}

#define CDEF_SEARCH_BLOCKS_ARGS                                                                     \
    const uint16_t *in, const uint16_t *target, int target_stride, uint8_t (*dir)[CDEF_NBLOCKS],    \
        int32_t (*var)[CDEF_NBLOCKS], const CdefList *dlist, int count, const CdefSearchPlan *plan, \
        const uint8_t *primary, int fixed_dir, int damping, int height, int subsampling_factor, uint64_t *ssd

// The variants of the search: 4 pixels wide with and without secondary strengths, 8 pixels wide with full and
// (for damping 8) halved magnitudes.
static void cdef_search_blocks_4_primary(CDEF_SEARCH_BLOCKS_ARGS) {
    cdef_search_blocks(in,
                       target,
                       target_stride,
                       dir,
                       var,
                       dlist,
                       count,
                       plan,
                       primary,
                       fixed_dir,
                       damping,
                       4,
                       height,
                       subsampling_factor,
                       false,
                       false,
                       ssd);
}

static void cdef_search_blocks_4(CDEF_SEARCH_BLOCKS_ARGS) {
    cdef_search_blocks(in,
                       target,
                       target_stride,
                       dir,
                       var,
                       dlist,
                       count,
                       plan,
                       primary,
                       fixed_dir,
                       damping,
                       4,
                       height,
                       subsampling_factor,
                       false,
                       true,
                       ssd);
}

static void cdef_search_blocks_8(CDEF_SEARCH_BLOCKS_ARGS) {
    cdef_search_blocks(in,
                       target,
                       target_stride,
                       dir,
                       var,
                       dlist,
                       count,
                       plan,
                       primary,
                       fixed_dir,
                       damping,
                       8,
                       height,
                       subsampling_factor,
                       false,
                       true,
                       ssd);
}

static void cdef_search_blocks_8_halved(CDEF_SEARCH_BLOCKS_ARGS) {
    cdef_search_blocks(in,
                       target,
                       target_stride,
                       dir,
                       var,
                       dlist,
                       count,
                       plan,
                       primary,
                       fixed_dir,
                       damping,
                       8,
                       height,
                       subsampling_factor,
                       true,
                       true,
                       ssd);
}

static void cdef_search_blocks_any(const uint16_t* in, const uint16_t* target, int target_stride,
                                   uint8_t (*dir)[CDEF_NBLOCKS], int32_t (*var)[CDEF_NBLOCKS], const CdefList* dlist,
                                   int count, const CdefSearchPlan* plan, const uint8_t* primary, int fixed_dir,
                                   int damping, int bsize, int subsampling_factor, uint64_t* ssd) {
    const int width  = bsize == BLOCK_8X8 || bsize == BLOCK_8X4 ? 8 : 4;
    const int height = bsize == BLOCK_8X8 || bsize == BLOCK_4X8 ? 8 : 4;
    // Damping 8 is a luma value (chroma damping is one lower), so only the 8-wide path has that variant.
    (width == 4
         ? (plan->secondary_mask | plan->zero_secondary_mask) ? cdef_search_blocks_4 : cdef_search_blocks_4_primary
         : damping == 8 ? cdef_search_blocks_8_halved
                        : cdef_search_blocks_8)(in,
                                                target,
                                                target_stride,
                                                dir,
                                                var,
                                                dlist,
                                                count,
                                                plan,
                                                primary,
                                                fixed_dir,
                                                damping,
                                                height,
                                                subsampling_factor,
                                                ssd);
}

void svt_cdef_search_block_10bit_neon(const uint16_t* input, const uint16_t* target, int target_stride,
                                      const CdefSearchPlan* plan, const uint8_t primary[16], int dir, int damping,
                                      int bsize, int subsampling_factor, uint64_t ssd[64]) {
    cdef_search_blocks_any(
        input, target, target_stride, NULL, NULL, NULL, 1, plan, primary, dir, damping, bsize, subsampling_factor, ssd);
}

// The sums of squared errors of the trials of a plan over the listed blocks of a filter block (before the
// normalization of svt_cdef_search_fb_10bit()). The directions are those of the blocks' luma.
void svt_cdef_search_blocks_10bit_neon(const uint16_t* in, const uint16_t* target, int target_stride,
                                       uint8_t dir[CDEF_NBLOCKS][CDEF_NBLOCKS], int32_t var[CDEF_NBLOCKS][CDEF_NBLOCKS],
                                       const CdefList* dlist, int count, const CdefSearchPlan* plan, int pli,
                                       int damping, int bsize, int subsampling_factor, uint64_t ssd[TOTAL_STRENGTHS]) {
    // Chroma strengths do not depend on the block.
    uint8_t chroma_primary[CDEF_PRI_STRENGTHS];
    for (int p = 0; p < plan->primary_count; ++p) {
        chroma_primary[p] = plan->primary[p] << 2;
    }
    cdef_search_blocks_any(in,
                           target,
                           target_stride,
                           dir,
                           var,
                           dlist,
                           count,
                           plan,
                           pli ? chroma_primary : NULL,
                           0,
                           damping,
                           bsize,
                           subsampling_factor,
                           ssd);
}

#endif // CONFIG_ENABLE_HIGH_BIT_DEPTH
