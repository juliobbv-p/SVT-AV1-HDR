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

#ifndef SVT_AV1_HIGHBD_INV_TXFM_NEON_H
#define SVT_AV1_HIGHBD_INV_TXFM_NEON_H

#include "definitions.h"

#if defined(ARCH_AARCH64) && CONFIG_ENABLE_HIGH_BIT_DEPTH
// A 10-bit DCT_DCT square of size 16/32/64 with nonzero coefficients confined
// to a band x band square (band 4/8/16). Returns zero without modifying pixels
// if any coefficient exceeds the proven range for eight-lane arithmetic.
int svt_av1_highbd_inv_txfm_try_narrow_neon(const int32_t* input, const uint16_t* pred, int sp, uint16_t* dest, int dp,
                                            TxSize tx_size, unsigned band);
#endif

#endif // SVT_AV1_HIGHBD_INV_TXFM_NEON_H
