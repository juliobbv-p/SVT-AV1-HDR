/*
 * Copyright(c) 2019 Intel Corporation
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved.
 * Copyright (c) 2026 Halide Compression, LLC
 *
 * This source code is subject to the terms of the BSD 3-Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. See LICENSE.md and
 * PATENTS.md for details.
 */

#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include "app_film_grain.h"
#include "../Lib/Codec/grainSynthesis.h"

static bool read_token(FILE* file, char token[32]) {
    if (fscanf(file, " %31s", token) != 1) {
        return false;
    }
    const int next = fgetc(file);
    return (next == EOF || isspace((unsigned char)next)) && !ferror(file);
}

static bool read_label(FILE* file, const char* label) {
    char token[32];
    return read_token(file, token) && !strcmp(token, label);
}

static bool read_integer(FILE* file, int32_t* value) {
    char token[32], *end;
    if (!read_token(file, token)) {
        return false;
    }
    errno                  = 0;
    const long long parsed = strtoll(token, &end, 10);
    if (errno == ERANGE || end == token || *end || parsed < INT32_MIN || parsed > INT32_MAX) {
        return false;
    }
    *value = (int32_t)parsed;
    return true;
}

static bool read_time(FILE* file, uint64_t* time) {
    char token[32], *end;
    if (!read_token(file, token) || !isdigit((unsigned char)token[0])) {
        return false;
    }
    errno = 0;
    *time = strtoull(token, &end, 10);
    return errno != ERANGE && *end == '\0';
}

static bool read_points(FILE* file, const char* label, int32_t* count, int max, int32_t points[][2]) {
    if (!read_label(file, label) || !read_integer(file, count) || *count < 0 || *count > max) {
        return false;
    }
    for (int i = 0; i < *count; ++i) {
        if (!read_integer(file, &points[i][0]) || !read_integer(file, &points[i][1])) {
            return false;
        }
    }
    return true;
}

static bool read_coeffs(FILE* file, const char* label, int count, int32_t* coeffs) {
    if (!read_label(file, label)) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        if (!read_integer(file, &coeffs[i])) {
            return false;
        }
    }
    return true;
}

static bool read_model(FILE* file, AomFilmGrain* grain) {
    if (!read_label(file, "p")) {
        return false;
    }
    int32_t* fields[] = {&grain->ar_coeff_lag,
                         &grain->ar_coeff_shift,
                         &grain->grain_scale_shift,
                         &grain->scaling_shift,
                         &grain->chroma_scaling_from_luma,
                         &grain->overlap_flag,
                         &grain->cb_mult,
                         &grain->cb_luma_mult,
                         &grain->cb_offset,
                         &grain->cr_mult,
                         &grain->cr_luma_mult,
                         &grain->cr_offset};
    for (unsigned i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        if (!read_integer(file, fields[i])) {
            return false;
        }
    }
    if ((uint32_t)grain->ar_coeff_lag > 3) {
        return false;
    }
    const int n = 2 * grain->ar_coeff_lag * (grain->ar_coeff_lag + 1);
    return read_points(file, "sY", &grain->num_y_points, 14, grain->scaling_points_y) &&
        read_points(file, "sCb", &grain->num_cb_points, 10, grain->scaling_points_cb) &&
        read_points(file, "sCr", &grain->num_cr_points, 10, grain->scaling_points_cr) &&
        read_coeffs(file, "cY", n, grain->ar_coeffs_y) && read_coeffs(file, "cCb", n + 1, grain->ar_coeffs_cb) &&
        read_coeffs(file, "cCr", n + 1, grain->ar_coeffs_cr);
}

void free_film_grain_table(FilmGrainTable* table) {
    while (table->head) {
        FilmGrainEntry* next = table->head->next;
        free(table->head);
        table->head = next;
    }
    table->current = NULL;
    table->seed    = 0;
}

EbErrorType read_film_grain_table(FILE* file, FilmGrainTable* table) {
    EbErrorType     status     = EB_ErrorBadParameter;
    FilmGrainEntry* tail       = NULL;
    bool            have_model = false;
    char            magic[9];
    if (fread(magic, 1, sizeof(magic), file) != sizeof(magic) || memcmp(magic, "filmgrn1", 8) ||
        !isspace((unsigned char)magic[8])) {
        return status;
    }
    for (;;) {
        FilmGrainEntry entry = {0};
        if (tail) {
            entry.grain = tail->grain;
        }
        char    label[32];
        int32_t seed;
        if (!read_token(file, label)) {
            if (feof(file) && !ferror(file)) {
                break;
            }
            goto fail;
        }
        if (strcmp(label, "E") || !read_time(file, &entry.start_time) || !read_time(file, &entry.end_time) ||
            !read_integer(file, &entry.grain.apply_grain) || !read_integer(file, &seed) ||
            !read_integer(file, &entry.grain.update_parameters) || entry.start_time >= entry.end_time ||
            (tail && entry.start_time < tail->end_time) || seed < 0 || seed > UINT16_MAX ||
            (uint32_t)entry.grain.apply_grain > 1 || (uint32_t)entry.grain.update_parameters > 1 ||
            (entry.grain.apply_grain && !entry.grain.update_parameters && !have_model)) {
            goto fail;
        }
        if (entry.grain.update_parameters) {
            if (!read_model(file, &entry.grain)) {
                goto fail;
            }
            have_model = true;
        }
        entry.grain.random_seed = (uint16_t)seed;
        entry.grain.ignore_ref  = 1;
        entry.grain.bit_depth   = 8;
        // The input chroma format is not known here. The encoder validates the
        // 4:2:0-only chroma point-count constraint when accepting the grain event.
        if (!svt_aom_film_grain_params_valid(&entry.grain, false)) {
            goto fail;
        }
        FilmGrainEntry* next = malloc(sizeof(*next));
        if (!next) {
            status = EB_ErrorInsufficientResources;
            goto fail;
        }
        *next = entry;
        if (tail) {
            tail->next = next;
        } else {
            table->head = next;
        }
        tail = next;
    }
    if (!table->head) {
        goto fail;
    }
    table->current = table->head;
    table->seed    = table->head->grain.random_seed;
    return EB_ErrorNone;
fail:
    free_film_grain_table(table);
    return status;
}

void lookup_film_grain_table(FilmGrainTable* table, uint64_t frame, const EbSvtAv1EncConfiguration* config,
                             AomFilmGrain* grain) {
    const long double time = (long double)frame * 10000000 * config->frame_rate_denominator /
        config->frame_rate_numerator;
    while (table->current && time >= table->current->end_time) {
        table->current = table->current->next;
        if (table->current) {
            table->seed = table->current->grain.random_seed;
        }
    }
    memset(grain, 0, sizeof(*grain));
    if (table->current && time >= table->current->start_time) {
        *grain             = table->current->grain;
        grain->random_seed = table->seed;
        table->seed += 3381;
        if (!table->seed) {
            table->seed += 7391;
        }
    }
    grain->bit_depth = config->encoder_bit_depth;
}
