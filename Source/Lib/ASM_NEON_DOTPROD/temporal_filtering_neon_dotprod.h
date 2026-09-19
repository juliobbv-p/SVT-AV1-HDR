/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#ifndef TEMPORAL_FILTERING_NEON_DOTPROD_H_
#define TEMPORAL_FILTERING_NEON_DOTPROD_H_

#include <arm_neon.h>

static inline void mad8x8x2_neon_dotprod(const uint8x16_t s[8], const uint8x16_t mean, uint32x4_t* activity_vec0,
                                         uint32x4_t* activity_vec1) {
    const uint8x16_t ones = vdupq_n_u8(1);

    uint8x16_t abs0 = vabdq_u8(s[0], mean);
    uint8x16_t abs1 = vabdq_u8(s[1], mean);
    *activity_vec0  = vdotq_u32(*activity_vec0, abs0, ones);
    *activity_vec1  = vdotq_u32(*activity_vec1, abs1, ones);

    abs0           = vabdq_u8(s[2], mean);
    abs1           = vabdq_u8(s[3], mean);
    *activity_vec0 = vdotq_u32(*activity_vec0, abs0, ones);
    *activity_vec1 = vdotq_u32(*activity_vec1, abs1, ones);

    abs0           = vabdq_u8(s[4], mean);
    abs1           = vabdq_u8(s[5], mean);
    *activity_vec0 = vdotq_u32(*activity_vec0, abs0, ones);
    *activity_vec1 = vdotq_u32(*activity_vec1, abs1, ones);

    abs0           = vabdq_u8(s[6], mean);
    abs1           = vabdq_u8(s[7], mean);
    *activity_vec0 = vdotq_u32(*activity_vec0, abs0, ones);
    *activity_vec1 = vdotq_u32(*activity_vec1, abs1, ones);
}

static inline void mad8x8_neon_dotprod(const uint8x8_t s[8], const uint8x16_t mean, uint32x4_t* activity_vec0,
                                       uint32x4_t* activity_vec1) {
    const uint8x16_t ones = vdupq_n_u8(1);

    const uint8x16_t s01 = vcombine_u8(s[0], s[1]);
    const uint8x16_t s23 = vcombine_u8(s[2], s[3]);
    const uint8x16_t s45 = vcombine_u8(s[4], s[5]);
    const uint8x16_t s67 = vcombine_u8(s[6], s[7]);

    uint8x16_t abs0 = vabdq_u8(s01, mean);
    uint8x16_t abs1 = vabdq_u8(s23, mean);
    *activity_vec0  = vdotq_u32(*activity_vec0, abs0, ones);
    *activity_vec1  = vdotq_u32(*activity_vec1, abs1, ones);

    abs0           = vabdq_u8(s45, mean);
    abs1           = vabdq_u8(s67, mean);
    *activity_vec0 = vdotq_u32(*activity_vec0, abs0, ones);
    *activity_vec1 = vdotq_u32(*activity_vec1, abs1, ones);
}

#endif // TEMPORAL_FILTERING_NEON_DOTPROD_H_
