/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved.
 * Copyright (c) 2026 Halide Compression, LLC
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. See LICENSE.md and
 * PATENTS.md for details.
 */

#ifndef APP_FILM_GRAIN_H
#define APP_FILM_GRAIN_H

#include <stdio.h>
#include "EbSvtAv1Enc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FilmGrainEntry {
    AomFilmGrain           grain;
    uint64_t               start_time;
    uint64_t               end_time;
    struct FilmGrainEntry* next;
} FilmGrainEntry;

typedef struct FilmGrainTable {
    FilmGrainEntry* head;
    FilmGrainEntry* current;
    uint16_t        seed;
} FilmGrainTable;

EbErrorType read_film_grain_table(FILE* file, FilmGrainTable* table);
void        free_film_grain_table(FilmGrainTable* table);
void        lookup_film_grain_table(FilmGrainTable* table, uint64_t frame, const EbSvtAv1EncConfiguration* config,
                                    AomFilmGrain* grain);

#ifdef __cplusplus
}
#endif

#endif
