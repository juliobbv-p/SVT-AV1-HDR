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

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <arm_acle.h>
#endif
#include <string.h>

#include "aom_dsp_rtcd.h"

/* CRC-32C (Castagnoli) using the Armv8.0-A CRC32 extension. Bit-exact with
 * svt_av1_get_crc32c_value_c. */
uint32_t svt_av1_get_crc32c_value_arm_crc32(const uint8_t* buf, size_t len) {
    const uint8_t* next = buf;
    uint32_t       crc  = 0xFFFFFFFF;

    while (len >= 8) {
        uint64_t v;
        memcpy(&v, next, sizeof(v));
        crc = __crc32cd(crc, v);
        next += 8;
        len -= 8;
    }
    if (len >= 4) {
        uint32_t v;
        memcpy(&v, next, sizeof(v));
        crc = __crc32cw(crc, v);
        next += 4;
        len -= 4;
    }
    if (len >= 2) {
        uint16_t v;
        memcpy(&v, next, sizeof(v));
        crc = __crc32ch(crc, v);
        next += 2;
        len -= 2;
    }
    if (len) {
        crc = __crc32cb(crc, *next);
    }
    return crc ^ 0xFFFFFFFF;
}
