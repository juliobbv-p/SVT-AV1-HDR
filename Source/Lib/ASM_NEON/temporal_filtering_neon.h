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

#ifndef TEMPORAL_FILTERING_NEON_H_
#define TEMPORAL_FILTERING_NEON_H_

#include <arm_neon.h>
#include <stdint.h>

static inline uint8_t avg8x8_neon(uint8x8_t s[8]) {
    uint16x8_t sum = vaddl_u8(s[0], s[1]);
    sum            = vaddw_u8(sum, s[2]);
    sum            = vaddw_u8(sum, s[3]);
    sum            = vaddw_u8(sum, s[4]);
    sum            = vaddw_u8(sum, s[5]);
    sum            = vaddw_u8(sum, s[6]);
    sum            = vaddw_u8(sum, s[7]);

    return vaddvq_u16(sum) >> 6;
}

#endif // TEMPORAL_FILTERING_NEON_H_
