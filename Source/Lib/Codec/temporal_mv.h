/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#ifndef SVT_AV1_TEMPORAL_MV_H
#define SVT_AV1_TEMPORAL_MV_H

#include "inter_prediction.h"
#include "coding_unit.h"

// At most 4x4 interior samples and three extension samples per block.
#define TPL_MV_SAMPLE_CAP 19

typedef struct TplMvSample {
    Mv  mv;
    int denominator;
    int weight;
} TplMvSample;

typedef struct TplMvCache {
    TplMvSample samples[TPL_MV_SAMPLE_CAP];
    Mv          projected[REF_FRAMES][TPL_MV_SAMPLE_CAP];
    unsigned    ready;
    int         count;
    int         first_available;
} TplMvCache;

// Entries are never removed or reordered during the temporal stage. Repeated
// inputs therefore update the same first match, or are all rejected by a full
// stack. Keep first-occurrence order and combine their weight increments.
static INLINE int tpl_mv_gather_sample(TplMvCache* cache, const TPL_MV_REF* field, int stride, const TileInfo* tile,
                                       int mi_row, int mi_col, int row, int col) {
    row += mi_row + !(mi_row & 1);
    col += mi_col + !(mi_col & 1);
    if (row < tile->mi_row_start || row >= tile->mi_row_end || col < tile->mi_col_start || col >= tile->mi_col_end) {
        return 0;
    }
    const TPL_MV_REF* sample = &field[(row >> 1) * stride + (col >> 1)];
    if (sample->mfmv0.as_int == INVALID_MV) {
        return 0;
    }
    int i;
    for (i = 0; i < cache->count; ++i) {
        if (cache->samples[i].mv.as_int == sample->mfmv0.as_int &&
            cache->samples[i].denominator == sample->ref_frame_offset) {
            cache->samples[i].weight += 2;
            return 1;
        }
    }
    assert(i < TPL_MV_SAMPLE_CAP);
    cache->samples[i].mv          = sample->mfmv0;
    cache->samples[i].denominator = sample->ref_frame_offset;
    cache->samples[i].weight      = 2;
    ++cache->count;
    return 1;
}

static INLINE void tpl_mv_gather(TplMvCache* cache, const TPL_MV_REF* field, int stride, const MacroBlockD* xd,
                                 int mi_row, int mi_col, int square) {
    cache->count           = 0;
    cache->ready           = 0;
    cache->first_available = 0;
    int row_end, col_end, step_w, step_h, allow_extension;
    if (square) {
        row_end = col_end = xd->n4_w;
        step_w            = (xd->n4_w >= MI_SIZE_W_64X64) ? MI_SIZE_W_16X16 : MI_SIZE_W_8X8;
        step_h            = step_w;
        allow_extension   = (xd->n4_w >= MI_SIZE_W_8X8) && (xd->n4_w < MI_SIZE_W_64X64);
    } else {
        row_end         = AOMMIN(xd->n4_h, mi_size_high[BLOCK_64X64]);
        col_end         = AOMMIN(xd->n4_w, mi_size_wide[BLOCK_64X64]);
        allow_extension = (xd->n4_h >= mi_size_high[BLOCK_8X8]) && (xd->n4_h < mi_size_high[BLOCK_64X64]) &&
            (xd->n4_w >= mi_size_wide[BLOCK_8X8]) && (xd->n4_w < mi_size_wide[BLOCK_64X64]);
        step_h = (xd->n4_h >= mi_size_high[BLOCK_64X64]) ? mi_size_high[BLOCK_16X16] : mi_size_high[BLOCK_8X8];
        step_w = (xd->n4_w >= mi_size_wide[BLOCK_64X64]) ? mi_size_wide[BLOCK_16X16] : mi_size_high[BLOCK_8X8];
    }
    for (int row = 0; row < row_end; row += step_h) {
        for (int col = 0; col < col_end; col += step_w) {
            const int available = tpl_mv_gather_sample(cache, field, stride, &xd->tile, mi_row, mi_col, row, col);
            if (!row && !col) {
                cache->first_available = available;
            }
        }
    }
    if (allow_extension) {
        const int voffset   = square ? xd->n4_h : AOMMAX(mi_size_high[BLOCK_8X8], xd->n4_h);
        const int hoffset   = square ? xd->n4_h : AOMMAX(mi_size_wide[BLOCK_8X8], xd->n4_w);
        const int pos[3][2] = {{voffset, -2}, {voffset, hoffset}, {voffset - 2, hoffset}};
        for (int i = 0; i < 3; ++i) {
            if (check_sb_border(mi_row, mi_col, pos[i][0], pos[i][1])) {
                tpl_mv_gather_sample(cache, field, stride, &xd->tile, mi_row, mi_col, pos[i][0], pos[i][1]);
            }
        }
    }
}

static INLINE void tpl_mv_project(TplMvCache* cache, MvReferenceFrame frame, int offset, int allow_hp, int symmetric) {
    if (symmetric && frame == BWDREF_FRAME) {
        // Preserve negation after LAST's clamping and precision reduction.
        assert(cache->ready & (1u << LAST_FRAME));
        for (int i = 0; i < cache->count; ++i) {
            cache->projected[frame][i].x = -cache->projected[LAST_FRAME][i].x;
            cache->projected[frame][i].y = -cache->projected[LAST_FRAME][i].y;
        }
    } else {
        for (int i = 0; i < cache->count; ++i) {
            get_mv_projection(&cache->projected[frame][i], cache->samples[i].mv, offset, cache->samples[i].denominator);
            lower_mv_precision(&cache->projected[frame][i], allow_hp);
        }
    }
    cache->ready |= 1u << frame;
}

static INLINE void tpl_mv_add(const TplMvCache* cache, const MvReferenceFrame rf[2], const Mv gm[2],
                              CandidateMv stack[MAX_REF_MV_STACK_SIZE], uint8_t* count, int16_t* context) {
    if (!cache->first_available) {
        *context |= 1 << GLOBALMV_OFFSET;
    } else {
        const Mv a = cache->projected[rf[0]][0];
        if (abs(a.x - gm[0].x) >= 16 || abs(a.y - gm[0].y) >= 16) {
            *context |= 1 << GLOBALMV_OFFSET;
        }
        if (rf[1] != NONE_FRAME) {
            const Mv b = cache->projected[rf[1]][0];
            if (abs(b.x - gm[1].x) >= 16 || abs(b.y - gm[1].y) >= 16) {
                *context |= 1 << GLOBALMV_OFFSET;
            }
        }
    }
    if (rf[1] == NONE_FRAME) {
        for (int i = 0; i < cache->count; ++i) {
            const Mv a = cache->projected[rf[0]][i];
            int      idx;
            for (idx = 0; idx < *count; ++idx) {
                if (a.as_int == stack[idx].this_mv.as_int) {
                    stack[idx].weight += cache->samples[i].weight;
                    break;
                }
            }
            if (idx == *count && *count < MAX_REF_MV_STACK_SIZE) {
                stack[idx].this_mv = a;
                stack[idx].weight  = cache->samples[i].weight;
                ++*count;
            }
        }
    } else {
        for (int i = 0; i < cache->count; ++i) {
            const Mv a = cache->projected[rf[0]][i];
            const Mv b = cache->projected[rf[1]][i];
            int      idx;
            for (idx = 0; idx < *count; ++idx) {
                if (a.as_int == stack[idx].this_mv.as_int && b.as_int == stack[idx].comp_mv.as_int) {
                    stack[idx].weight += cache->samples[i].weight;
                    break;
                }
            }
            if (idx == *count && *count < MAX_REF_MV_STACK_SIZE) {
                stack[idx].this_mv = a;
                stack[idx].comp_mv = b;
                stack[idx].weight  = cache->samples[i].weight;
                ++*count;
            }
        }
    }
}

#endif // SVT_AV1_TEMPORAL_MV_H
