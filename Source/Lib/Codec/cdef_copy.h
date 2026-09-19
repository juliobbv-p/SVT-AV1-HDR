/*
 * Copyright (c) 2025, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

// CDEF pixel-copy helpers. Portable C (no target intrinsics): the 8-bit gather copy dispatches on
// its (small, fixed) set of widths to constant-size memcpy, which the compiler lowers to optimal
// inline load/stores on every target.

#ifndef EbCdefCopy_h
#define EbCdefCopy_h

#include <stdint.h>
#include <string.h>

#include "svt_malloc.h" // svt_aom_memset16

// 8-bit recon -> padded src8 rect copy. The CDEF gather (apply + native-8 search) only ever uses the
// widths enumerated below (all CDEF reads are +/-CDEF_HALO, so body is hsz+2*HALO and halo strips are
// HALO). Dispatch ONCE per call to a constant-size memcpy loop so each row folds to optimal inline
// load/stores on arm64 AND x86 -- no per-row libc memmove call, no target intrinsics.
static inline void svt_cdef_copy_rect8(uint8_t* dst, int dstride, const uint8_t* src, int src_voffset, int src_hoffset,
                                       int sstride, int v, int h) {
    const uint8_t* base = &src[src_voffset * sstride + src_hoffset];
#define CDEF_RECT_CW(W)                  \
    do {                                 \
        for (int _r = 0; _r < v; _r++) { \
            memcpy(dst, base, (W));      \
            dst += dstride;              \
            base += sstride;             \
        }                                \
    } while (0)
    switch (h) {
    case 2:
        CDEF_RECT_CW(2);
        return;
    case 32:
        CDEF_RECT_CW(32);
        return;
    case 34:
        CDEF_RECT_CW(34);
        return;
    case 36:
        CDEF_RECT_CW(36);
        return;
    case 64:
        CDEF_RECT_CW(64);
        return;
    case 66:
        CDEF_RECT_CW(66);
        return;
    case 68:
        CDEF_RECT_CW(68);
        return;
    default:
        break;
    }
#undef CDEF_RECT_CW
    // Other widths are legitimate when the coded frame width is not a multiple of the CDEF block
    // size (e.g. reference scaling / --resize-mode produces partial right-edge filter blocks).
    // The fast cases above cover the common geometries; any other width uses this variable-width
    // memcpy fallback.
    for (int r = 0; r < v; r++) {
        memcpy(dst, base, (size_t)h);
        dst += dstride;
        base += sstride;
    }
}

// HBD-only paths below (compiled out of RTC via the CDEF_8BITS_PATH guards at the call sites).

// uint16->uint16 rect copy (edge-path src assembly).
static inline void svt_aom_copy_rect(uint16_t* dst, int dstride, const uint16_t* src, int sstride, int v, int h) {
    for (int r = 0; r < v; r++) {
        memcpy(dst, src, sizeof(*dst) * (size_t)h);
        dst += dstride;
        src += sstride;
    }
}

// uint16 rect fill (edge-path off-frame sentinel).
static inline void svt_aom_fill_rect(uint16_t* dst, int dstride, int v, int h, uint16_t x) {
    for (int r = 0; r < v; r++) {
        svt_aom_memset16(dst + r * dstride, x, (size_t)h);
    }
}

// CDEF superblock copy into the padded src buffer: widen 8-bit recon (u8->u16), or copy 16-bit recon
// (u16->u16), from a (voffset,hoffset) origin. Composes the primitives above.
static inline void svt_aom_copy_sb8_16(uint16_t* dst, int dstride, const uint8_t* src, int src_voffset, int src_hoffset,
                                       int sstride, int vsize, int hsize, int is_16bit) {
    if (is_16bit) {
        svt_aom_copy_rect(
            dst, dstride, (const uint16_t*)src + (src_voffset * sstride + src_hoffset), sstride, vsize, hsize);
    } else {
        svt_aom_copy_rect8_8bit_to_16bit(
            dst, dstride, &src[src_voffset * sstride + src_hoffset], sstride, vsize, hsize);
    }
}

#endif // EbCdefCopy_h
