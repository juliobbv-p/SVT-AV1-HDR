/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
 */

#include "hash.h"

/* CRC-32C (iSCSI) polynomial in reversed bit order. */
#define POLY 0x82f63b78

/* Process-wide table for a quadword-at-a-time software crc. The only writer is
 * svt_av1_crc32c_table_init(), which runs from init_global_tables() under
 * svt_run_once() (pthread_once / InitOnce): the table is built exactly once per
 * process, before any encoder instance starts encoding, and is read-only after
 * that. The once-guard's release/acquire semantics make the completed table
 * visible to every thread and every concurrent encoder instance, so sharing one
 * copy across instances is safe (same lifetime as every other global lookup
 * table built in init_global_tables). */
static uint32_t crc32c_table[8][256];

/* Construct table for software CRC-32C calculation. */
void svt_av1_crc32c_table_init(void) {
    uint32_t crc;

    for (int n = 0; n < 256; n++) {
        crc                = n;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc                = (crc & 1) ? (crc >> 1) ^ POLY : crc >> 1;
        crc32c_table[0][n] = crc;
    }
    for (int n = 0; n < 256; n++) {
        crc = crc32c_table[0][n];
        for (int k = 1; k < 8; k++) {
            crc                = crc32c_table[0][crc & 0xff] ^ (crc >> 8);
            crc32c_table[k][n] = crc;
        }
    }
}

/* Table-driven software version as a fall-back.  This is about 15 times slower
 than using the hardware instructions.  This assumes little-endian integers,
 as is the case on Intel processors that the assembler code here is for. */
uint32_t svt_av1_get_crc32c_value_c(const uint8_t* buf, size_t len) {
    const uint8_t* next = buf;
    uint64_t       crc;
    crc = 0 ^ 0xffffffff;
    while (len && ((uintptr_t)next & 7) != 0) {
        crc = crc32c_table[0][(crc ^ *next++) & 0xff] ^ (crc >> 8);
        len--;
    }
    while (len >= 8) {
        crc ^= *(uint64_t*)next;
        crc = crc32c_table[7][crc & 0xff] ^ crc32c_table[6][(crc >> 8) & 0xff] ^ crc32c_table[5][(crc >> 16) & 0xff] ^
            crc32c_table[4][(crc >> 24) & 0xff] ^ crc32c_table[3][(crc >> 32) & 0xff] ^
            crc32c_table[2][(crc >> 40) & 0xff] ^ crc32c_table[1][(crc >> 48) & 0xff] ^ crc32c_table[0][crc >> 56];
        next += 8;
        len -= 8;
    }
    while (len) {
        crc = crc32c_table[0][(crc ^ *next++) & 0xff] ^ (crc >> 8);
        len--;
    }
    return (uint32_t)crc ^ 0xffffffff;
}
