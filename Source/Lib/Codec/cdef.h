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
#ifndef EbCdef_h
#define EbCdef_h

#include <stdint.h>
#include "definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CDEF_STRENGTH_BITS 6
#define CDEF_PRI_STRENGTHS 16
#define CDEF_SEC_STRENGTHS 4

#define CDEF_BLOCKSIZE 64
#define CDEF_BLOCKSIZE_LOG2 6
#define CDEF_NBLOCKS ((1 << MAX_SB_SIZE_LOG2) / 8)

/* We need to buffer three vertical lines. */
#define CDEF_VBORDER (3)
/* We only need to buffer three horizontal pixels too, but let's align to
16 bytes (8 x 16 bits) to make vectorization easier. */
#define CDEF_HBORDER (8)
// CDEF taps reach at most +-CDEF_HALO pixels (see eb_cdef_directions), so the recon copies/narrows in
// svt_av1_cdef_frame only need a 2-px halo; the buffer stays HBORDER/VBORDER-padded for aligned loads.
#define CDEF_HALO 2
#define CDEF_BSTRIDE ALIGN_POWER_OF_TWO((1 << MAX_SB_SIZE_LOG2) + 2 * CDEF_HBORDER, 3)
// Value is chosen so that memset can be used in cdef_seg_search().  Must be a large
// int16_t value.
#define CDEF_VERY_LARGE ((uint8_t)~0 >> 1 | ((uint8_t)~0 >> 1) << 8)
#define CDEF_INBUF_SIZE (CDEF_BSTRIDE * ((1 << MAX_SB_SIZE_LOG2) + 2 * CDEF_VBORDER))

extern const int32_t svt_aom_eb_cdef_pri_taps[2][2];
extern const int32_t svt_aom_eb_cdef_sec_taps[2][2];
extern const int (*const svt_aom_eb_cdef_directions)[2];
// Per-tap (drow,dcol) decode of the tap table (same +2-offset indexing as svt_aom_eb_cdef_directions).
// Used by the boundary-aware kernels to test off-frame taps by geometry.
extern const int8_t (*const svt_aom_eb_cdef_directions_rc)[2][2];

#define TOTAL_STRENGTHS (CDEF_PRI_STRENGTHS * CDEF_SEC_STRENGTHS)

void svt_cdef_filter_fb(uint8_t* dst8, uint16_t* dst16, int32_t dstride, uint16_t* in, int32_t xdec, int32_t ydec,
                        uint8_t dir[CDEF_NBLOCKS][CDEF_NBLOCKS], int32_t* dirinit,
                        int32_t var[CDEF_NBLOCKS][CDEF_NBLOCKS], int32_t pli, CdefList* dlist, int32_t cdef_count,
                        int32_t cdef_strength, int32_t damping, int32_t coeff_shift, uint8_t subsampling_factor);

#if CDEF_8BITS_PATH
// Hybrid CDEF (LBD / 8-bit): native 8-bit for interior blocks, boundary-aware 8-bit for frame-edge
// blocks. Interior-vs-edge dispatch is decided from the frame_* geometry flags.
void svt_cdef_filter_fb_lbd(uint8_t* dst8, int32_t dstride, const uint8_t* in8, int frame_top, int frame_left,
                            int frame_bottom, int frame_right, int vsize, int hsize, int32_t xdec, int32_t ydec,
                            uint8_t dir[CDEF_NBLOCKS][CDEF_NBLOCKS], int32_t* dirinit,
                            int32_t var[CDEF_NBLOCKS][CDEF_NBLOCKS], int32_t pli, CdefList* dlist, int32_t cdef_count,
                            int32_t cdef_strength, int32_t damping, int32_t coeff_shift, uint8_t subsampling_factor);
#endif // CDEF_8BITS_PATH

#ifdef __cplusplus
}
#endif
#endif // EbCdef_h
