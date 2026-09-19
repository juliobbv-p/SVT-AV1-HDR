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

#ifndef AOM_AV1_ENCODER_HASH_H_
#define AOM_AV1_ENCODER_HASH_H_

#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build the process-wide table used by the software fall-back CRC-32C kernel
 * (svt_av1_get_crc32c_value_c); called once at library init
 * (init_global_tables) and read-only afterwards. The hardware kernels need no
 * setup. */
void svt_av1_crc32c_table_init(void);

// Number of 2x2 pixel blocks per superblock
// The biggest superblock supported by AV1 is 128x128, therefore there can be
// a maximum of 64x64 blocks per superblock: 64 * 64 = 4096
#define AOM_BUFFER_SIZE_FOR_BLOCK_HASH (4096)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOM_AV1_ENCODER_HASH_H_
