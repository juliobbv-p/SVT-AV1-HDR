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

#include <arm_neon.h>

#include "aom_dsp_rtcd.h"
#include "mem_neon.h"
#include "sad_neon_dotprod.h"
#include "sum_neon.h"

void svt_ext_sad_calculation_8x8_16x16_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                    uint32_t ref_stride, uint32_t* p_best_sad_8x8,
                                                    uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8,
                                                    uint32_t* p_best_mv16x16, uint32_t mv, uint32_t* p_sad16x16,
                                                    uint32_t* p_sad8x8, bool sub_sad) {
    uint32_t   sad16x16;
    uint32x4_t sad;
    uint32x4_t best_sad_vec = vld1q_u32(p_best_sad_8x8);
    uint32x4_t best_mv_vec  = vld1q_u32(p_best_mv8x8);
    uint32x4_t mv_vec       = vdupq_n_u32(mv);

    if (sub_sad) {
        uint32x4_t sad01 = sad_8x4_dual_neon(
            src + 0 * src_stride, 2 * src_stride, ref + 0 * ref_stride, 2 * ref_stride, 2);
        uint32x4_t sad23 = sad_8x4_dual_neon(
            src + 8 * src_stride, 2 * src_stride, ref + 8 * ref_stride, 2 * ref_stride, 2);

        sad = vpaddq_u32(sad01, sad23);
    } else {
        uint32x4_t sad01 = sad_8x8_dual_neon(src + 0 * src_stride, src_stride, ref + 0 * ref_stride, ref_stride, 1);
        uint32x4_t sad23 = sad_8x8_dual_neon(src + 8 * src_stride, src_stride, ref + 8 * ref_stride, ref_stride, 1);

        sad = vpaddq_u32(sad01, sad23);
    }

    uint32x4_t cmp = vcltq_u32(sad, best_sad_vec);
    best_sad_vec   = vbslq_u32(cmp, sad, best_sad_vec);
    best_mv_vec    = vbslq_u32(cmp, mv_vec, best_mv_vec);

    sad16x16 = vaddvq_u32(sad);
    if (sad16x16 < p_best_sad_16x16[0]) {
        p_best_sad_16x16[0] = sad16x16;
        p_best_mv16x16[0]   = mv;
    }

    *p_sad16x16 = sad16x16;
    vst1q_u32(p_sad8x8, sad);
    vst1q_u32(p_best_sad_8x8, best_sad_vec);
    vst1q_u32(p_best_mv8x8, best_mv_vec);
}

// Keeps the (sad, mv index) of b where its sad is strictly lower, so ties keep a (the lower search index).
static inline void min_sad_idx(uint32x4_t* sad_a, uint32x4_t* idx_a, uint32x4_t sad_b, uint32x4_t idx_b) {
    const uint32x4_t lt = vcltq_u32(sad_b, *sad_a);
    *sad_a              = vbslq_u32(lt, sad_b, *sad_a);
    *idx_a              = vbslq_u32(lt, idx_b, *idx_a);
}

// sad[k] holds the four 8x8 SADs of one 16x16 block at search position x_mv + k. The bests are reduced with a tree
// instead of position by position, which keeps the first (lowest-index) minimum like a sequential strict-less scan.
static inline void update_eight_sad_8x8_16x16(const uint32x4_t sad[8], uint32_t mv, uint32_t start_16x16_pos,
                                              uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16,
                                              uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
                                              uint32_t p_eight_sad16x16[16][8]) {
    const uint32_t start_8x8_pos = 4 * start_16x16_pos;
    const uint32_t x_mv          = _MVXT(mv);
    const uint32_t y_mv          = _MVYT(mv);

    // 8x8: per lane minimum over the 8 positions
    uint32x4_t s[4], i[4];
    for (int k = 0; k < 4; k++) {
        s[k] = sad[2 * k];
        i[k] = vdupq_n_u32(2 * k);
        min_sad_idx(&s[k], &i[k], sad[2 * k + 1], vdupq_n_u32(2 * k + 1));
    }
    min_sad_idx(&s[0], &i[0], s[1], i[1]);
    min_sad_idx(&s[2], &i[2], s[3], i[3]);
    min_sad_idx(&s[0], &i[0], s[2], i[2]);
    const uint32x4_t mv_vec       = vorrq_u32(vdupq_n_u32(y_mv << 16),
                                        vandq_u32(vaddq_u32(vdupq_n_u32(x_mv), i[0]), vdupq_n_u32(0xFFFF)));
    uint32x4_t       best_sad_vec = vld1q_u32(p_best_sad_8x8 + start_8x8_pos);
    uint32x4_t       best_mv_vec  = vld1q_u32(p_best_mv8x8 + start_8x8_pos);
    min_sad_idx(&best_sad_vec, &best_mv_vec, s[0], mv_vec);
    vst1q_u32(p_best_sad_8x8 + start_8x8_pos, best_sad_vec);
    vst1q_u32(p_best_mv8x8 + start_8x8_pos, best_mv_vec);

    // 16x16: the sum of the four 8x8 SADs at each position
    const uint32x4_t sad16_lo = vpaddq_u32(vpaddq_u32(sad[0], sad[1]), vpaddq_u32(sad[2], sad[3]));
    const uint32x4_t sad16_hi = vpaddq_u32(vpaddq_u32(sad[4], sad[5]), vpaddq_u32(sad[6], sad[7]));
    vst1q_u32(p_eight_sad16x16[start_16x16_pos], sad16_lo);
    vst1q_u32(p_eight_sad16x16[start_16x16_pos] + 4, sad16_hi);
    const uint32_t min16 = vminvq_u32(vminq_u32(sad16_lo, sad16_hi));
    if (min16 < p_best_sad_16x16[start_16x16_pos]) {
        static const uint32_t idx[8]      = {0, 1, 2, 3, 4, 5, 6, 7};
        const uint32x4_t      eight       = vdupq_n_u32(8);
        const uint32x4_t      m           = vdupq_n_u32(min16);
        const uint32x4_t      k_lo        = vbslq_u32(vceqq_u32(sad16_lo, m), vld1q_u32(idx), eight);
        const uint32x4_t      k_hi        = vbslq_u32(vceqq_u32(sad16_hi, m), vld1q_u32(idx + 4), eight);
        const uint32_t        k           = vminvq_u32(vminq_u32(k_lo, k_hi));
        p_best_sad_16x16[start_16x16_pos] = min16;
        p_best_mv16x16[start_16x16_pos]   = (y_mv << 16) | ((x_mv + k) & 0xFFFF);
    }
}

static AOM_FORCE_INLINE void svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_1_neon_dotprod(
    uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t mv, uint32_t start_16x16_pos,
    uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
    uint32_t p_eight_sad16x16[16][8]) {
    uint32x4_t sad[8];
    for (int search_index = 0; search_index < 8; search_index++) {
        const uint32x4_t sad01 = sad_8x4_dual_neon(src, 2 * src_stride, ref + search_index, 2 * ref_stride, 2);
        const uint32x4_t sad23 = sad_8x4_dual_neon(
            src + 8 * src_stride, 2 * src_stride, ref + 8 * ref_stride + search_index, 2 * ref_stride, 2);
        sad[search_index] = vpaddq_u32(sad01, sad23);
    }
    update_eight_sad_8x8_16x16(
        sad, mv, start_16x16_pos, p_best_sad_8x8, p_best_sad_16x16, p_best_mv8x8, p_best_mv16x16, p_eight_sad16x16);
}

static inline void svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_0_neon_dotprod(
    uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t mv, uint32_t start_16x16_pos,
    uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
    uint32_t p_eight_sad16x16[16][8]) {
    uint32x4_t sad[8];
    for (int search_index = 0; search_index < 8; search_index++) {
        const uint32x4_t sad01 = sad_8x8_dual_neon(src, src_stride, ref + search_index, ref_stride, 1);
        const uint32x4_t sad23 = sad_8x8_dual_neon(
            src + 8 * src_stride, src_stride, ref + 8 * ref_stride + search_index, ref_stride, 1);
        sad[search_index] = vpaddq_u32(sad01, sad23);
    }
    update_eight_sad_8x8_16x16(
        sad, mv, start_16x16_pos, p_best_sad_8x8, p_best_sad_16x16, p_best_mv8x8, p_best_mv16x16, p_eight_sad16x16);
}

// ---------------------------------------------------------------------------
// The same eight-position search with the absolute differences of several rows accumulated in 8-bit lanes
// before each dot product. Whether that is safe is decided per 16x16 block, once per search, from value ranges:
// with every source sample of the block within [smin, smax] and every reference sample the block's searches
// read within [rmin, rmax], no absolute difference exceeds D = max(smax - rmin, rmax - smin), and a lane that
// accumulates `chunk` rows holds at most chunk * D. The tiers are
//     1: D <= 127, two rows per lane   (at most 254)
//     2: D <=  63, four rows per lane  (at most 252)
//     3: D <=  31, eight rows per lane (at most 248)
// and 0 otherwise (the kernel above). A full search takes 16 difference and 16 dot-product operations per
// position and 16x16 block; tier 3 takes 16 and 2.
// ---------------------------------------------------------------------------

// Tiers of the sixteen 16x16 blocks of a 64x64 block (raster order) for a search whose eight-position calls start
// at the reference positions (x, y), x in [0, area_w), y in [0, area_h), area_w a multiple of 8: a block's
// reference region is the (area_w + 15) x (area_h + 15) rectangle at its own offset.
void svt_ext_all_sad_tiers_neon_dotprod(const uint8_t* src, uint32_t src_stride, const uint8_t* ref,
                                        uint32_t ref_stride, uint32_t area_w, uint32_t area_h, uint8_t tiers[16]) {
    const uint32_t reg_w = area_w + 15, reg_h = area_h + 15;
    for (int by = 0; by < 4; by++) {
        for (int bx = 0; bx < 4; bx++) {
            const uint8_t* s    = src + 16 * by * src_stride + 16 * bx;
            const uint8_t* r    = ref + 16 * by * ref_stride + 16 * bx;
            uint8x16_t     smin = vdupq_n_u8(255), smax = vdupq_n_u8(0);
            for (int y = 0; y < 16; y++) {
                const uint8x16_t v = vld1q_u8(s + y * src_stride);

                smin = vminq_u8(smin, v);

                smax = vmaxq_u8(smax, v);
            }
            uint8x16_t rmin0 = vdupq_n_u8(255), rmax0 = vdupq_n_u8(0), rmin1 = rmin0, rmax1 = rmax0;
            for (uint32_t y = 0; y < reg_h; y++) {
                const uint8_t* p = r + y * ref_stride;

                uint32_t x = 0;
                for (; x + 16 <= reg_w; x += 16) {
                    const uint8x16_t v = vld1q_u8(p + x);

                    rmin0 = vminq_u8(rmin0, v);

                    rmax0 = vmaxq_u8(rmax0, v);
                }
                // the last, overlapping vector of the row (reg_w is at least 23)
                const uint8x16_t v = vld1q_u8(p + reg_w - 16);

                rmin1 = vminq_u8(rmin1, v);

                rmax1 = vmaxq_u8(rmax1, v);
            }
            const int s_lo = vminvq_u8(smin), s_hi = vmaxvq_u8(smax);
            const int r_lo = vminvq_u8(vminq_u8(rmin0, rmin1)), r_hi = vmaxvq_u8(vmaxq_u8(rmax0, rmax1));
            const int d = AOMMAX(s_hi - r_lo, r_hi - s_lo);

            tiers[4 * by + bx] = d <= 31 ? 3 : d <= 63 ? 2 : d <= 127 ? 1 : 0;
        }
    }
}

// The four 8x8 SADs of one 16x16 block at eight positions: `rows` rows per 8x8 (8, or 4 with every other row
// and the sums doubled), `chunk` of them accumulated per dot product. Position by position inside each row keeps
// the eight accumulation chains independent.
static AOM_FORCE_INLINE void eight_sad_chunked_neon_dotprod(const uint8_t* src, uint32_t src_stride, const uint8_t* ref,
                                                            uint32_t ref_stride, const int rows, const int chunk,
                                                            const int mult, uint32x4_t sad[8]) {
    const uint8x16_t weight = vdupq_n_u8((uint8_t)mult);

    uint32x4_t half[2][8];
    for (int h = 0; h < 2; h++) {
        for (int k = 0; k < 8; k++) {
            half[h][k] = vdupq_n_u32(0);
        }
        for (int r = 0; r < rows; r += chunk) {
            uint8x16_t acc[8];

            const uint8x16_t s0 = vld1q_u8(src);
            for (int k = 0; k < 8; k++) {
                acc[k] = vabdq_u8(s0, vld1q_u8(ref + k));
            }
            src += src_stride;
            ref += ref_stride;
            for (int c = 1; c < chunk; c++) {
                const uint8x16_t s = vld1q_u8(src);
                for (int k = 0; k < 8; k++) {
                    acc[k] = vabaq_u8(acc[k], s, vld1q_u8(ref + k));
                }
                src += src_stride;
                ref += ref_stride;
            }
            for (int k = 0; k < 8; k++) {
                half[h][k] = vdotq_u32(half[h][k], acc[k], weight);
            }
        }
    }
    for (int k = 0; k < 8; k++) {
        sad[k] = vpaddq_u32(half[0][k], half[1][k]);
    }
}

#define EIGHT_SAD_CHUNKED(name, rows, chunk, mult, stride_mult)                                        \
    static NOINLINE void name(uint8_t*  src,                                                           \
                              uint32_t  src_stride,                                                    \
                              uint8_t*  ref,                                                           \
                              uint32_t  ref_stride,                                                    \
                              uint32_t  mv,                                                            \
                              uint32_t  start_16x16_pos,                                               \
                              uint32_t* p_best_sad_8x8,                                                \
                              uint32_t* p_best_sad_16x16,                                              \
                              uint32_t* p_best_mv8x8,                                                  \
                              uint32_t* p_best_mv16x16,                                                \
                              uint32_t  p_eight_sad16x16[16][8]) {                                      \
        uint32x4_t sad[8];                                                                             \
        eight_sad_chunked_neon_dotprod(                                                                \
            src, (stride_mult) * src_stride, ref, (stride_mult) * ref_stride, rows, chunk, mult, sad); \
        update_eight_sad_8x8_16x16(sad,                                                                \
                                   mv,                                                                 \
                                   start_16x16_pos,                                                    \
                                   p_best_sad_8x8,                                                     \
                                   p_best_sad_16x16,                                                   \
                                   p_best_mv8x8,                                                       \
                                   p_best_mv16x16,                                                     \
                                   p_eight_sad16x16);                                                  \
    }
EIGHT_SAD_CHUNKED(eight_sad_full_chunk2_neon_dotprod, 8, 2, 1, 1)
EIGHT_SAD_CHUNKED(eight_sad_full_chunk4_neon_dotprod, 8, 4, 1, 1)
EIGHT_SAD_CHUNKED(eight_sad_full_chunk8_neon_dotprod, 8, 8, 1, 1)
// every other row: four rows per 8x8, so tiers 2 and 3 both take them in one chunk
EIGHT_SAD_CHUNKED(eight_sad_sub_chunk2_neon_dotprod, 4, 2, 2, 2)
EIGHT_SAD_CHUNKED(eight_sad_sub_chunk4_neon_dotprod, 4, 4, 2, 2)

// svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod() with the tiers of svt_ext_all_sad_tiers_neon_dotprod().
void svt_ext_all_sad_calculation_8x8_16x16_tiers_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                              uint32_t ref_stride, uint32_t mv,
                                                              uint32_t* p_best_sad_8x8, uint32_t* p_best_sad_16x16,
                                                              uint32_t* p_best_mv8x8, uint32_t* p_best_mv16x16,
                                                              uint32_t p_eight_sad16x16[16][8], bool sub_sad,
                                                              const uint8_t tiers[16]) {
    static const char offsets[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
    const uint32_t    pf_step     = sub_sad ? 2 : 1;
    for (uint32_t r = 0; r < 64; r += pf_step) {
        __builtin_prefetch(ref + r * ref_stride + 64, 0, 3);
    }
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            uint8_t* const s   = src + 16 * y * src_stride + 16 * x;
            uint8_t* const p   = ref + 16 * y * ref_stride + 16 * x;
            const uint32_t pos = (uint32_t)offsets[4 * y + x];
#define EIGHT_SAD_ARGS                                                                                     \
    s, src_stride, p, ref_stride, mv, pos, p_best_sad_8x8, p_best_sad_16x16, p_best_mv8x8, p_best_mv16x16, \
        p_eight_sad16x16
            switch (tiers[4 * y + x]) {
            case 3:
                if (sub_sad) {
                    eight_sad_sub_chunk4_neon_dotprod(EIGHT_SAD_ARGS);
                } else {
                    eight_sad_full_chunk8_neon_dotprod(EIGHT_SAD_ARGS);
                }
                break;
            case 2:
                if (sub_sad) {
                    eight_sad_sub_chunk4_neon_dotprod(EIGHT_SAD_ARGS);
                } else {
                    eight_sad_full_chunk4_neon_dotprod(EIGHT_SAD_ARGS);
                }
                break;
            case 1:
                if (sub_sad) {
                    eight_sad_sub_chunk2_neon_dotprod(EIGHT_SAD_ARGS);
                } else {
                    eight_sad_full_chunk2_neon_dotprod(EIGHT_SAD_ARGS);
                }
                break;
            default:
                if (sub_sad) {
                    svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_1_neon_dotprod(EIGHT_SAD_ARGS);
                } else {
                    svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_0_neon_dotprod(EIGHT_SAD_ARGS);
                }
                break;
            }
#undef EIGHT_SAD_ARGS
        }
    }
}

void svt_ext_all_sad_calculation_8x8_16x16_neon_dotprod(uint8_t* src, uint32_t src_stride, uint8_t* ref,
                                                        uint32_t ref_stride, uint32_t mv, uint32_t* p_best_sad_8x8,
                                                        uint32_t* p_best_sad_16x16, uint32_t* p_best_mv8x8,
                                                        uint32_t* p_best_mv16x16, uint32_t p_eight_sad16x16[16][8],
                                                        uint32_t p_eight_sad8x8[64][8], bool sub_sad) {
    (void)p_eight_sad8x8;
    static const char offsets[16] = {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};

    // Successive calls slide ref by +8 columns over the same rows, so only the second cache line of each row
    // (cols 64..) is new; sub_sad reads only even rows.
    const uint32_t pf_step = sub_sad ? 2 : 1;
    for (uint32_t r = 0; r < 64; r += pf_step) {
        __builtin_prefetch(ref + r * ref_stride + 64, 0, 3);
    }

    //---- 16x16 : 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15
    if (sub_sad) {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                const uint32_t block_index           = 16 * y * src_stride + 16 * x;
                const uint32_t search_position_index = 16 * y * ref_stride + 16 * x;
                svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_1_neon_dotprod(src + block_index,
                                                                               src_stride,
                                                                               ref + search_position_index,
                                                                               ref_stride,
                                                                               mv,
                                                                               offsets[4 * y + x],
                                                                               p_best_sad_8x8,
                                                                               p_best_sad_16x16,
                                                                               p_best_mv8x8,
                                                                               p_best_mv16x16,
                                                                               p_eight_sad16x16);
            }
        }
    } else {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                const uint32_t block_index           = 16 * y * src_stride + 16 * x;
                const uint32_t search_position_index = 16 * y * ref_stride + 16 * x;
                svt_ext_eight_sad_calculation_8x8_16x16_sub_sad_0_neon_dotprod(src + block_index,
                                                                               src_stride,
                                                                               ref + search_position_index,
                                                                               ref_stride,
                                                                               mv,
                                                                               offsets[4 * y + x],
                                                                               p_best_sad_8x8,
                                                                               p_best_sad_16x16,
                                                                               p_best_mv8x8,
                                                                               p_best_mv16x16,
                                                                               p_eight_sad16x16);
            }
        }
    }
}
