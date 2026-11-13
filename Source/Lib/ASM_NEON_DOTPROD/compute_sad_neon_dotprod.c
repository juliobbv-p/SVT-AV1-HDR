/*
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2023, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "compute_sad_c.h"
#include "compute_sad_neon.h"
#include "compute_sad_neon_dotprod.h"
#include "mem_neon.h"
#include "sum_neon.h"
#include "utility.h"

DECLARE_ALIGNED(16, static const uint8_t, kPermTable[32]) = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
                                                             4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10};

static inline uint32x4_t sadwxhx4d_neon_dotprod(const uint8_t* src, uint32_t src_stride, const uint8_t* ref,
                                                uint32_t ref_stride, uint32_t width, uint32_t height) {
    uint32x4_t sum_u32[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    uint16x8_t sum_u16[4] = {vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0)};
    uint32x4_t sum4       = vdupq_n_u32(0);
    uint16x8_t sum        = vdupq_n_u16(0);

    do {
        int w = width;

        const uint8_t* src_ptr = src;
        const uint8_t* ref_ptr = ref;

        while (w >= 16) {
            const uint8x16_t s = vld1q_u8(src_ptr);
            sad16_neon_dotprod(s, vld1q_u8(ref_ptr + 0), &sum_u32[0]);
            sad16_neon_dotprod(s, vld1q_u8(ref_ptr + 1), &sum_u32[1]);
            sad16_neon_dotprod(s, vld1q_u8(ref_ptr + 2), &sum_u32[2]);
            sad16_neon_dotprod(s, vld1q_u8(ref_ptr + 3), &sum_u32[3]);

            src_ptr += 16;
            ref_ptr += 16;
            w -= 16;
        }

        if (w >= 8) {
            const uint8x8_t s = vld1_u8(src_ptr);
            sum_u16[0]        = vabal_u8(sum_u16[0], s, vld1_u8(ref_ptr + 0));
            sum_u16[1]        = vabal_u8(sum_u16[1], s, vld1_u8(ref_ptr + 1));
            sum_u16[2]        = vabal_u8(sum_u16[2], s, vld1_u8(ref_ptr + 2));
            sum_u16[3]        = vabal_u8(sum_u16[3], s, vld1_u8(ref_ptr + 3));

            src_ptr += 8;
            ref_ptr += 8;
            w -= 8;
        }

        if (w >= 4) {
            uint8x16_t perm_tbl = vld1q_u8(kPermTable);
            uint8x16_t s        = vreinterpretq_u8_u32(vld1q_dup_u32((const uint32_t*)src_ptr));
            uint8x16_t r        = vqtbl1q_u8(vld1q_u8(ref_ptr), perm_tbl);
            uint8x16_t abs      = vabdq_u8(s, r);
            sum4                = vdotq_u32(sum4, abs, vdupq_n_u8(1));
            src_ptr += 4;
            ref_ptr += 4;
            w -= 4;
        }

        while (--w >= 0) {
            uint8x8_t s = vld1_dup_u8(src_ptr + w);
            uint8x8_t r = load_u8_4x1(ref_ptr + w);
            sum         = vabal_u8(sum, s, r);
        }

        src += src_stride;
        ref += ref_stride;
    } while (--height != 0);

    sum_u32[0] = vpadalq_u16(sum_u32[0], sum_u16[0]);
    sum_u32[1] = vpadalq_u16(sum_u32[1], sum_u16[1]);
    sum_u32[2] = vpadalq_u16(sum_u32[2], sum_u16[2]);
    sum_u32[3] = vpadalq_u16(sum_u32[3], sum_u16[3]);
    return vaddq_u32(vaddw_u16(sum4, vget_low_u16(sum)), horizontal_add_4d_u32x4(sum_u32));
}

static inline void svt_sad_loop_kernelwxh_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                       uint32_t ref_stride, uint32_t block_width, uint32_t block_height,
                                                       uint64_t* best_sad, int16_t* x_search_center,
                                                       int16_t* y_search_center, uint32_t src_stride_raw,
                                                       int16_t search_area_width, int16_t search_area_height) {
    for (int16_t y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        for (int16_t x_search_index = 0; x_search_index < search_area_width; x_search_index += 8) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4_0 = sadwxhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_width, block_height);
            uint32x4_t sad4_1 = sadwxhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index + 4, ref_stride, block_width, block_height);
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(sad4_1, best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }

        ref += src_stride_raw;
    }
}

static inline void svt_sad_loop_kernelwxh_small_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                             uint32_t ref_stride, uint32_t block_width,
                                                             uint32_t block_height, uint64_t* best_sad,
                                                             int16_t* x_search_center, int16_t* y_search_center,
                                                             uint32_t src_stride_raw, int16_t search_area_width,
                                                             int16_t search_area_height) {
    for (int16_t y_search_index = 0; y_search_index < search_area_height; y_search_index++) {
        int16_t x_search_index;
        for (x_search_index = 0; x_search_index <= search_area_width - 4; x_search_index += 4) {
            /* Get the SAD of 4 search spaces aligned along the width and store it in 'sad4'. */
            uint32x4_t sad4 = sadwxhx4d_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_width, block_height);
            update_best_sad_u32(sad4, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }

        for (; x_search_index < search_area_width; x_search_index++) {
            /* Get the SAD of 1 search spaces aligned along the width and store it in 'temp_sad'. */
            uint64_t temp_sad = sad_anywxh_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, block_width, block_height);
            update_best_sad(temp_sad, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
        }
        ref += src_stride_raw;
    }
}

// ---------------------------------------------------------------------------
// Searches whose value ranges allow several rows of absolute differences per 8-bit lane.
//
// With every source sample of the block within [smin, smax] and every reference sample the search reads within
// [rmin, rmax], no absolute difference exceeds D = max(smax - rmin, rmax - smin), and a lane that accumulates
// `chunk` rows holds at most chunk * D: two rows for D <= 127 (254), four for D <= 63 (252), eight for D <= 31
// (248). The plain kernels take one difference and one dot product per vector and position; eight rows per
// lane take eight differences and one dot product.
// ---------------------------------------------------------------------------

// 0, or the number of rows (2, 4, 8) that may share a lane. The block is bw x bh (bw a multiple of 16) with rows
// src_stride apart; the reference region is reg_w x reg_h (reg_w at least 16) with rows ref_stride apart.
static int sad_range_chunk_neon(const uint8_t* src, uint32_t src_stride, uint32_t bw, uint32_t bh, const uint8_t* ref,
                                uint32_t ref_stride, uint32_t reg_w, uint32_t reg_h) {
    uint8x16_t smin = vdupq_n_u8(255), smax = vdupq_n_u8(0);
    for (uint32_t y = 0; y < bh; y++) {
        for (uint32_t x = 0; x < bw; x += 16) {
            const uint8x16_t v = vld1q_u8(src + y * src_stride + x);

            smin = vminq_u8(smin, v);

            smax = vmaxq_u8(smax, v);
        }
    }
    uint8x16_t rmin0 = vdupq_n_u8(255), rmax0 = vdupq_n_u8(0), rmin1 = rmin0, rmax1 = rmax0;
    for (uint32_t y = 0; y < reg_h; y++) {
        const uint8_t* p = ref + y * ref_stride;
        for (uint32_t x = 0; x + 16 <= reg_w; x += 16) {
            const uint8x16_t v = vld1q_u8(p + x);

            rmin0 = vminq_u8(rmin0, v);

            rmax0 = vmaxq_u8(rmax0, v);
        }
        // the last, overlapping vector of the row
        const uint8x16_t v = vld1q_u8(p + reg_w - 16);

        rmin1 = vminq_u8(rmin1, v);

        rmax1 = vmaxq_u8(rmax1, v);
    }
    const int s_lo = vminvq_u8(smin), s_hi = vmaxvq_u8(smax);
    const int r_lo = vminvq_u8(vminq_u8(rmin0, rmin1)), r_hi = vmaxvq_u8(vmaxq_u8(rmax0, rmax1));
    const int d = AOMMAX(s_hi - r_lo, r_hi - s_lo);
    return d <= 31 ? 8 : d <= 63 ? 4 : d <= 127 ? 2 : 0;
}

// SADs of a w x h block (w a multiple of 16) at four horizontally adjacent positions, `chunk` rows per lane.
static AOM_FORCE_INLINE uint32x4_t sadwxhx4d_chunk_neon_dotprod(const uint8_t* src, uint32_t src_stride,
                                                                const uint8_t* ref, uint32_t ref_stride, const int w,
                                                                int h, const int chunk) {
    const uint8x16_t ones = vdupq_n_u8(1);

    uint32x4_t sum[4] = {vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)};
    while (h > 0) {
        const int n = h < chunk ? h : chunk;
        for (int x = 0; x < w; x += 16) {
            const uint8_t*   s  = src + x;
            const uint8_t*   r  = ref + x;
            const uint8x16_t s0 = vld1q_u8(s);

            uint8x16_t a0 = vabdq_u8(s0, vld1q_u8(r));

            uint8x16_t a1 = vabdq_u8(s0, vld1q_u8(r + 1));

            uint8x16_t a2 = vabdq_u8(s0, vld1q_u8(r + 2));

            uint8x16_t a3 = vabdq_u8(s0, vld1q_u8(r + 3));
            for (int i = 1; i < n; i++) {
                s += src_stride;
                r += ref_stride;
                const uint8x16_t sv = vld1q_u8(s);

                a0 = vabaq_u8(a0, sv, vld1q_u8(r));

                a1 = vabaq_u8(a1, sv, vld1q_u8(r + 1));

                a2 = vabaq_u8(a2, sv, vld1q_u8(r + 2));

                a3 = vabaq_u8(a3, sv, vld1q_u8(r + 3));
            }
            sum[0] = vdotq_u32(sum[0], a0, ones);
            sum[1] = vdotq_u32(sum[1], a1, ones);
            sum[2] = vdotq_u32(sum[2], a2, ones);
            sum[3] = vdotq_u32(sum[3], a3, ones);
        }
        src += n * src_stride;
        ref += n * ref_stride;
        h -= n;
    }
    return horizontal_add_4d_u32x4(sum);
}

// The search loop of svt_sad_loop_kernel{16,32,64}xh_neon_dotprod() over the chunked SADs: eight positions per
// step, top half of the block first; when none of the eight partial SADs is below the best so far no full SAD
// can be, and the bottom half is skipped.
static AOM_FORCE_INLINE void sad_loop_chunk_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                         uint32_t ref_stride, const int bw, uint32_t block_height,
                                                         uint64_t* best_sad, int16_t* x_search_center,
                                                         int16_t* y_search_center, uint32_t src_stride_raw,
                                                         int16_t y_search_start, int16_t y_search_step,
                                                         int16_t search_area_width, int16_t search_area_height,
                                                         const int chunk) {
    const uint32_t h0 = block_height >= 4 ? block_height >> 1 : block_height;
    for (int16_t y_search_index = y_search_start; y_search_index < search_area_height;
         y_search_index += y_search_step) {
        for (int16_t x_search_index = 0; x_search_index < search_area_width; x_search_index += 8) {
            uint32x4_t sad4_0 = sadwxhx4d_chunk_neon_dotprod(
                src, src_stride, ref + x_search_index, ref_stride, bw, (int)h0, chunk);
            uint32x4_t sad4_1 = sadwxhx4d_chunk_neon_dotprod(
                src, src_stride, ref + x_search_index + 4, ref_stride, bw, (int)h0, chunk);
            if (vminvq_u32(vminq_u32(sad4_0, sad4_1)) >= *best_sad) {
                continue;
            }
            if (h0 < block_height) {
                sad4_0 = vaddq_u32(sad4_0,
                                   sadwxhx4d_chunk_neon_dotprod(src + h0 * src_stride,
                                                                src_stride,
                                                                ref + x_search_index + h0 * ref_stride,
                                                                ref_stride,
                                                                bw,
                                                                (int)(block_height - h0),
                                                                chunk));
                sad4_1 = vaddq_u32(sad4_1,
                                   sadwxhx4d_chunk_neon_dotprod(src + h0 * src_stride,
                                                                src_stride,
                                                                ref + x_search_index + 4 + h0 * ref_stride,
                                                                ref_stride,
                                                                bw,
                                                                (int)(block_height - h0),
                                                                chunk));
            }
            update_best_sad_u32(sad4_0, best_sad, x_search_center, y_search_center, x_search_index, y_search_index);
            update_best_sad_u32(sad4_1, best_sad, x_search_center, y_search_center, x_search_index + 4, y_search_index);
        }
        ref += src_stride_raw;
    }
}

#define SAD_LOOP_CHUNK(bw, chunk)                                                                  \
    static NOINLINE void sad_loop_##bw##_chunk##chunk##_neon_dotprod(uint8_t*  src,                \
                                                                     uint32_t  src_stride,         \
                                                                     uint8_t*  ref,                \
                                                                     uint32_t  ref_stride,         \
                                                                     uint32_t  block_height,       \
                                                                     uint64_t* best_sad,           \
                                                                     int16_t*  x_search_center,    \
                                                                     int16_t*  y_search_center,    \
                                                                     uint32_t  src_stride_raw,     \
                                                                     int16_t   y_search_start,     \
                                                                     int16_t   y_search_step,      \
                                                                     int16_t   search_area_width,  \
                                                                     int16_t   search_area_height) { \
        sad_loop_chunk_neon_dotprod(src,                                                           \
                                    src_stride,                                                    \
                                    ref,                                                           \
                                    ref_stride,                                                    \
                                    bw,                                                            \
                                    block_height,                                                  \
                                    best_sad,                                                      \
                                    x_search_center,                                               \
                                    y_search_center,                                               \
                                    src_stride_raw,                                                \
                                    y_search_start,                                                \
                                    y_search_step,                                                 \
                                    search_area_width,                                             \
                                    search_area_height,                                            \
                                    chunk);                                                        \
    }
SAD_LOOP_CHUNK(16, 2)
SAD_LOOP_CHUNK(16, 4)
SAD_LOOP_CHUNK(16, 8)
SAD_LOOP_CHUNK(32, 2)
SAD_LOOP_CHUNK(32, 4)
SAD_LOOP_CHUNK(32, 8)
SAD_LOOP_CHUNK(64, 2)
SAD_LOOP_CHUNK(64, 4)
SAD_LOOP_CHUNK(64, 8)

// Runs the search through the chunked kernels when the block is 16, 32 or 64 wide, the search is long enough to
// pay for the pass over the value ranges, and the ranges allow it. Returns false when nothing was done.
static bool sad_loop_ranged_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                         uint32_t block_height, uint32_t block_width, uint64_t* best_sad,
                                         int16_t* x_search_center, int16_t* y_search_center, uint32_t src_stride_raw,
                                         uint8_t skip_search_line, int16_t search_area_width,
                                         int16_t search_area_height) {
    if ((block_width != 16 && block_width != 32 && block_width != 64) || search_area_width * search_area_height < 32 ||
        src_stride_raw == 0 || ref_stride % src_stride_raw) {
        return false;
    }
    // Rows of the reference are src_stride_raw apart; the block's rows are a whole number of them apart.
    const uint32_t row_ratio = ref_stride / src_stride_raw;
    const int      chunk     = sad_range_chunk_neon(src,
                                           src_stride,
                                           block_width,
                                           block_height,
                                           ref,
                                           src_stride_raw,
                                           block_width + search_area_width - 1,
                                           (uint32_t)(search_area_height - 1) + (block_height - 1) * row_ratio + 1);
    if (!chunk) {
        return false;
    }
    int16_t y_search_start = 0, y_search_step = 1;
    if (block_width == 16 && block_height <= 16 && skip_search_line) {
        ref += src_stride_raw;
        src_stride_raw *= 2;
        y_search_start = 1;
        y_search_step  = 2;
    }
#define SAD_LOOP_ARGS                                                                                           \
    src, src_stride, ref, ref_stride, block_height, best_sad, x_search_center, y_search_center, src_stride_raw, \
        y_search_start, y_search_step, search_area_width, search_area_height
    if (block_width == 16) {
        if (chunk == 8) {
            sad_loop_16_chunk8_neon_dotprod(SAD_LOOP_ARGS);
        } else if (chunk == 4) {
            sad_loop_16_chunk4_neon_dotprod(SAD_LOOP_ARGS);
        } else {
            sad_loop_16_chunk2_neon_dotprod(SAD_LOOP_ARGS);
        }
    } else if (block_width == 32) {
        if (chunk == 8) {
            sad_loop_32_chunk8_neon_dotprod(SAD_LOOP_ARGS);
        } else if (chunk == 4) {
            sad_loop_32_chunk4_neon_dotprod(SAD_LOOP_ARGS);
        } else {
            sad_loop_32_chunk2_neon_dotprod(SAD_LOOP_ARGS);
        }
    } else {
        if (chunk == 8) {
            sad_loop_64_chunk8_neon_dotprod(SAD_LOOP_ARGS);
        } else if (chunk == 4) {
            sad_loop_64_chunk4_neon_dotprod(SAD_LOOP_ARGS);
        } else {
            sad_loop_64_chunk2_neon_dotprod(SAD_LOOP_ARGS);
        }
    }
#undef SAD_LOOP_ARGS
    return true;
}

void svt_sad_loop_kernel_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride,
                                      uint32_t block_height, uint32_t block_width, uint64_t* best_sad,
                                      int16_t* x_search_center, int16_t* y_search_center, uint32_t src_stride_raw,
                                      uint8_t skip_search_line, int16_t search_area_width, int16_t search_area_height) {
    *best_sad = UINT64_MAX;
    /* Most of the time search_area_width is a multiple of 8, so specialize for this case so that we run only sad4d. */
    if (search_area_width % 8 == 0) {
        if (search_area_width > 0 &&
            sad_loop_ranged_neon_dotprod(src,
                                         src_stride,
                                         ref,
                                         ref_stride,
                                         block_height,
                                         block_width,
                                         best_sad,
                                         x_search_center,
                                         y_search_center,
                                         src_stride_raw,
                                         skip_search_line,
                                         search_area_width,
                                         search_area_height)) {
            return;
        }
        switch (block_width) {
        case 4: {
            svt_sad_loop_kernel4xh_neon_dotprod(src,
                                                src_stride,
                                                ref,
                                                ref_stride,
                                                block_height,
                                                best_sad,
                                                x_search_center,
                                                y_search_center,
                                                src_stride_raw,
                                                search_area_width,
                                                search_area_height);
            break;
        }
        case 6: {
            svt_sad_loop_kernel6xh_neon_dotprod(src,
                                                src_stride,
                                                ref,
                                                ref_stride,
                                                block_height,
                                                best_sad,
                                                x_search_center,
                                                y_search_center,
                                                src_stride_raw,
                                                search_area_width,
                                                search_area_height);
            break;
        }
        case 8: {
            svt_sad_loop_kernel8xh_neon(src,
                                        src_stride,
                                        ref,
                                        ref_stride,
                                        block_height,
                                        best_sad,
                                        x_search_center,
                                        y_search_center,
                                        src_stride_raw,
                                        search_area_width,
                                        search_area_height);
            break;
        }
        case 12: {
            svt_sad_loop_kernel12xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 16: {
            svt_sad_loop_kernel16xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 skip_search_line,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 24: {
            svt_sad_loop_kernel24xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 32: {
            svt_sad_loop_kernel32xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 48: {
            svt_sad_loop_kernel48xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 64: {
            svt_sad_loop_kernel64xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        default: {
            svt_sad_loop_kernelwxh_neon_dotprod(src,
                                                src_stride,
                                                ref,
                                                ref_stride,
                                                block_width,
                                                block_height,
                                                best_sad,
                                                x_search_center,
                                                y_search_center,
                                                src_stride_raw,
                                                search_area_width,
                                                search_area_height);
            break;
        }
        }

    } else {
        switch (block_width) {
        case 4: {
            svt_sad_loop_kernel4xh_neon_dotprod(src,
                                                src_stride,
                                                ref,
                                                ref_stride,
                                                block_height,
                                                best_sad,
                                                x_search_center,
                                                y_search_center,
                                                src_stride_raw,
                                                search_area_width,
                                                search_area_height);
            break;
        }
        case 6: {
            svt_sad_loop_kernel6xh_neon_dotprod(src,
                                                src_stride,
                                                ref,
                                                ref_stride,
                                                block_height,
                                                best_sad,
                                                x_search_center,
                                                y_search_center,
                                                src_stride_raw,
                                                search_area_width,
                                                search_area_height);
            break;
        }
        case 8: {
            svt_sad_loop_kernel8xh_neon(src,
                                        src_stride,
                                        ref,
                                        ref_stride,
                                        block_height,
                                        best_sad,
                                        x_search_center,
                                        y_search_center,
                                        src_stride_raw,
                                        search_area_width,
                                        search_area_height);
            break;
        }
        case 12: {
            svt_sad_loop_kernel12xh_neon_dotprod(src,
                                                 src_stride,
                                                 ref,
                                                 ref_stride,
                                                 block_height,
                                                 best_sad,
                                                 x_search_center,
                                                 y_search_center,
                                                 src_stride_raw,
                                                 search_area_width,
                                                 search_area_height);
            break;
        }
        case 16: {
            svt_sad_loop_kernel16xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       skip_search_line,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        case 24: {
            svt_sad_loop_kernel24xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        case 32: {
            svt_sad_loop_kernel32xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        case 48: {
            svt_sad_loop_kernel48xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        case 64: {
            svt_sad_loop_kernel64xh_small_neon_dotprod(src,
                                                       src_stride,
                                                       ref,
                                                       ref_stride,
                                                       block_height,
                                                       best_sad,
                                                       x_search_center,
                                                       y_search_center,
                                                       src_stride_raw,
                                                       search_area_width,
                                                       search_area_height);
            break;
        }
        default: {
            svt_sad_loop_kernelwxh_small_neon_dotprod(src,
                                                      src_stride,
                                                      ref,
                                                      ref_stride,
                                                      block_width,
                                                      block_height,
                                                      best_sad,
                                                      x_search_center,
                                                      y_search_center,
                                                      src_stride_raw,
                                                      search_area_width,
                                                      search_area_height);
            break;
        }
        }
    }
}
