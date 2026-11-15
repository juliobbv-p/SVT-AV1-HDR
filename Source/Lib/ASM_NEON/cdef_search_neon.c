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

void svt_cdef_search_block_10bit_neon(const uint16_t* input, const uint16_t* target, int target_stride,
                                      const CdefSearchPlan* plan, const uint8_t primary[16], int dir, int damping,
                                      int bsize, int subsampling_factor, uint64_t ssd[64]) {
    // Each lane accumulates at most 16 * 1023^2, safely within signed 32 bits.
    int32x4_t accum[64];
    for (int i = 0; i < plan->strength_count; ++i) {
        accum[i] = vdupq_n_s32(0);
    }
    const int  width              = bsize == BLOCK_8X8 || bsize == BLOCK_8X4 ? 8 : 4;
    const int  height             = bsize == BLOCK_8X8 || bsize == BLOCK_4X8 ? 8 : 4;
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
