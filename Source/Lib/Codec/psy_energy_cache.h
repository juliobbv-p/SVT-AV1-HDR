/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#ifndef SVT_AV1_PSY_ENERGY_CACHE_H_
#define SVT_AV1_PSY_ENERGY_CACHE_H_

#include <stdint.h>

// Source pixels must remain immutable while an entry is active. A NULL input
// invalidates it. The layout key contains the source stride and exact dimensions.
typedef struct SvtPsyEnergyCache {
    const uint16_t* input;

    uint64_t layout;
    // At most 256 8x8 energies for 128x128. Narrow blocks use 4x4 energies.
    // A 16x16 group stores its four energies consecutively for one vector load.
    int32_t energy[256];
} SvtPsyEnergyCache;

#endif // SVT_AV1_PSY_ENERGY_CACHE_H_
