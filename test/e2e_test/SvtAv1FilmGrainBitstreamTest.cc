/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved.
 * Copyright (c) 2026 Halide Compression, LLC
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. See LICENSE-BSD2.md and
 * PATENTS.md for details.
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <vector>

#include "app_film_grain.h"
#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "ParseUtil.h"
#include "aom/aom_decoder.h"
#include "aom/aomdx.h"
#include "gtest/gtest.h"

namespace {
TEST(FilmGrainBitstreamTest, TableFlagMatchesBuild) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    AomFilmGrain table{};
    cfg.fgs_table = &table;
    cfg.source_width = cfg.source_height = 64;
    cfg.enc_mode = 10;
    cfg.level_of_parallelism = 1;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));
    EbBufferHeaderType* header = nullptr;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_stream_header(enc, &header));
    svt_av1_e2e_tools::SequenceHeaderParser parser;
    RefDecoder::StreamInfo info{};
    parser.input_obu_data(header->p_buffer, header->n_filled_len, &info);
    EXPECT_EQ(CONFIG_ENABLE_FILM_GRAIN, info.film_grain_params_present);
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_stream_header_release(header));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(enc));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
}

#if CONFIG_ENABLE_FILM_GRAIN
const std::string kTableModel =
    " p 0 6 0 8 0 1 0 0 0 0 0 0\n"
    " sY 2 0 16 255 16\n sCb 0\n sCr 0\n cY\n cCb 0\n cCr 0\n";

EbErrorType parse_table(const std::string& text, FilmGrainTable* table) {
    FILE* file = tmpfile();
    if (!file)
        return EB_ErrorInsufficientResources;
    fputs(text.c_str(), file);
    rewind(file);
    EbErrorType status = read_film_grain_table(file, table);
    fclose(file);
    return status;
}

TEST(FilmGrainTableTest, TimedEntriesGapsReuseAndFractionalRate) {
    FilmGrainTable table{};
    ASSERT_EQ(EB_ErrorNone,
              parse_table("filmgrn1\nE 0 10000000 0 0 0\n"
                          "E 10000000 20000000 1 62155 1\n" +
                              kTableModel +
                              "E 30000000 40000000 1 42 0\n"
                              "E 40000000 50000000 0 0 0\n"
                              "E 50000000 60000000 1 100 0\n",
                          &table));
    EbSvtAv1EncConfiguration config{};
    config.frame_rate_numerator = 24000;
    config.frame_rate_denominator = 1001;
    config.encoder_bit_depth = 10;
    AomFilmGrain grain{};
    for (unsigned frame = 0; frame <= 144; ++frame) {
        lookup_film_grain_table(&table, frame, &config, &grain);
        const bool on = (frame >= 24 && frame < 48) ||
                        (frame >= 72 && frame < 96) ||
                        (frame >= 120 && frame < 144);
        EXPECT_EQ(on, grain.apply_grain != 0) << frame;
        EXPECT_EQ(10, grain.bit_depth);
        if (on) {
            EXPECT_EQ(16, grain.scaling_points_y[0][1]);
        }
        if (frame == 24) {
            EXPECT_EQ(62155, grain.random_seed);
        }
        if (frame == 25) {
            EXPECT_EQ(7391, grain.random_seed);
        }
        if (frame == 26) {
            EXPECT_EQ(10772, grain.random_seed);
        }
        if (frame == 72) {
            EXPECT_EQ(42, grain.random_seed);
        }
        if (frame == 120) {
            EXPECT_EQ(100, grain.random_seed);
        }
    }
    free_film_grain_table(&table);
}

TEST(FilmGrainTableTest, RejectsInvalidEntriesAndAcceptsFullRange) {
    const std::string valid =
        "filmgrn1\nE 0 10000000 1 12345 1\n" + kTableModel;
    const auto changed = [&](const char* from, const char* to) {
        std::string text = valid;
        text.replace(text.find(from), strlen(from), to);
        return text;
    };
    const std::string invalid[] = {
        "filmgrn1\n",
        changed("1 12345 1", "4294967297 12345 1"),
        changed("1 12345 1", "1 4294967296 1"),
        changed("1 12345 1", "1 12345 4294967297"),
        changed("p 0 6", "p 0 4294967302"),
        changed("sY 2", "sY 4294967298"),
        changed("sY 2 0", "sY 2 4294967296"),
        changed("cCb 0", "cCb 4294967296"),
        changed("12345", "9999999999999999999999999999999999999999"),
        changed("sY 2 0 16 255 16", "sY 2 0 16 0 16"),
        "filmgrn1\nE 0 10000000 1 1 0\n",
        "filmgrn1\nE -1 10000000 0 0 0\n",
        "filmgrn1\nE 0 10000000 1 65536 1\n" + kTableModel,
        "filmgrn1\nE 0 10000000 1 1 1\n p 4 6 0 8 0 1 0 0 0 0 0 0\n",
        "filmgrn1\nE 0 10000000 1 1 1\n p 0 6 0 8 0 1 0 0 0 0 0 0\n sY 15\n",
        valid + "E 100 20000000 0 0 0\n",
        valid + "E 10000000 10000000 0 0 0\n",
        valid + "E 10000000 18446744073709551616 0 0 0\n"};
    for (const auto& text : invalid) {
        FilmGrainTable table{};
        EXPECT_EQ(EB_ErrorBadParameter, parse_table(text, &table)) << text;
        EXPECT_EQ(nullptr, table.head);
        free_film_grain_table(&table);
    }
    FilmGrainTable table{};
    EXPECT_EQ(EB_ErrorNone,
              parse_table("filmgrn1\nE 0 18446744073709551615 1 12345 1\n" +
                              kTableModel,
                          &table));
    ASSERT_NE(nullptr, table.head);
    EXPECT_EQ(UINT64_MAX, table.head->end_time);
    free_film_grain_table(&table);
}

constexpr uint32_t kFrames = 80;
constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;

enum GrainSource { Table, Events, Mixed };
struct GrainParam {
    uint8_t bit_depth;
    bool overlays;
    bool ignore_ref;
};
using Packets = std::vector<std::vector<uint8_t>>;

AomFilmGrain grain_model(uint32_t frame, const GrainParam& prm) {
    AomFilmGrain grain{};
    grain.apply_grain = 1;
    grain.update_parameters = 1;
    grain.bit_depth = prm.bit_depth;
    grain.num_y_points = 2;
    grain.scaling_points_y[0][0] = 0;
    grain.scaling_points_y[1][0] = 255;
    grain.scaling_points_y[0][1] = grain.scaling_points_y[1][1] =
        32 + (frame / 8) * 8;
    grain.chroma_scaling_from_luma = (frame / 8) % 2;
    if (!grain.chroma_scaling_from_luma) {
        grain.num_cb_points = grain.num_cr_points = 2;
        grain.scaling_points_cb[1][0] = grain.scaling_points_cr[1][0] = 255;
        grain.scaling_points_cb[0][1] = grain.scaling_points_cb[1][1] = 40;
        grain.scaling_points_cr[0][1] = grain.scaling_points_cr[1][1] = 48;
    }
    grain.scaling_shift = 8;
    grain.ar_coeff_lag = 1;
    grain.ar_coeffs_y[0] = 8;
    grain.ar_coeffs_cb[4] = grain.ar_coeffs_cr[4] = 16;
    grain.ar_coeff_shift = 7;
    grain.cb_mult = grain.cr_mult = 128;
    grain.cb_luma_mult = grain.cr_luma_mult = 128;
    grain.cb_offset = grain.cr_offset = 256;
    grain.overlap_flag = 1;
    grain.random_seed = static_cast<uint16_t>(1234 + frame * 37);
    grain.ignore_ref = prm.ignore_ref;
    return grain;
}

int grain_kind(uint32_t frame) {
    if (frame == 0)
        return 2;
    const uint32_t phase = frame % 8;
    return phase == 0 || phase == 5 ? 0 : phase == 3 || phase == 4 ? 2 : 1;
}

void encode(const GrainParam& prm, GrainSource source, Packets* packets) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    AomFilmGrain table = grain_model(0, prm);
    cfg.fgs_table = &table;
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.encoder_bit_depth = prm.bit_depth;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.enc_mode = 10;
    cfg.hierarchical_levels = 4;
    cfg.enable_overlays = prm.overlays;
    cfg.intra_period_length = -1;
    cfg.level_of_parallelism = 1;
    cfg.aq_mode = 0;
    cfg.enable_tf = prm.overlays;
    cfg.tune = 1;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    svt_av1_video_source::DummyVideoSource src(
        IMG_FMT_420, kWidth, kHeight, prm.bit_depth);
    ASSERT_EQ(EB_ErrorNone, src.open_source(0, kFrames));
    bool eos = false;
    auto drain = [&](bool done) {
        EbBufferHeaderType* out = nullptr;
        while (!eos &&
               svt_av1_enc_get_packet(enc, &out, done) == EB_ErrorNone) {
            eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
            if (out->n_filled_len)
                packets->emplace_back(out->p_buffer,
                                      out->p_buffer + out->n_filled_len);
            svt_av1_enc_release_out_buffer(&out);
        }
    };
    for (uint32_t i = 0; i < kFrames; ++i) {
        EbBufferHeaderType in{};
        in.size = sizeof(in);
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);
        uint8_t* planes[] = {frame->luma, frame->cb, frame->cr};
        const uint32_t strides[] = {
            frame->y_stride, frame->cb_stride, frame->cr_stride};
        for (int plane = 0; plane < 3; ++plane) {
            const uint32_t width = kWidth >> (plane ? 1 : 0);
            const uint32_t height = kHeight >> (plane ? 1 : 0);
            for (uint32_t y = 0; y < height; ++y) {
                if (prm.bit_depth == 10) {
                    uint16_t* row = reinterpret_cast<uint16_t*>(planes[plane]) +
                                    y * strides[plane];
                    std::fill(row, row + width, 512);
                } else {
                    memset(planes[plane] + y * strides[plane], 128, width);
                }
            }
        }
        in.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in.n_filled_len = src.get_frame_size();
        in.pts = i;
        in.pic_type = EB_AV1_INVALID_PICTURE;
        AomFilmGrain grain = grain_model(i, prm);
        EbPrivDataNode node{FILM_GRAIN_EVENT, &grain, sizeof(grain), nullptr};
        if (source == Events || (source == Mixed && grain_kind(i) != 0)) {
            if (source == Mixed && grain_kind(i) == 2)
                grain = AomFilmGrain{};
            in.p_app_private = &node;
        }
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in));
        grain = AomFilmGrain{};
        drain(false);
    }
    EbBufferHeaderType end{};
    end.size = sizeof(end);
    end.flags = EB_BUFFERFLAG_EOS;
    end.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &end));
    drain(true);
    EXPECT_TRUE(eos);
    src.close_source();
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(enc));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
}

void decode_noise(const Packets& packets, std::vector<uint64_t>* frames,
                  std::vector<uint64_t>* energy = nullptr,
                  unsigned frame_count = kFrames) {
    aom_codec_ctx_t decoders[2]{};
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(
            AOM_CODEC_OK,
            aom_codec_dec_init(&decoders[i], aom_codec_av1_dx(), nullptr, 0));
        ASSERT_EQ(AOM_CODEC_OK,
                  aom_codec_control(&decoders[i], AV1D_SET_SKIP_FILM_GRAIN, i));
    }
    for (const auto& packet : packets) {
        for (auto& decoder : decoders)
            ASSERT_EQ(AOM_CODEC_OK,
                      aom_codec_decode(
                          &decoder, packet.data(), packet.size(), nullptr))
                << aom_codec_error_detail(&decoder);
        aom_codec_iter_t iters[2]{};
        aom_image_t* image;
        while ((image = aom_codec_get_frame(&decoders[0], &iters[0]))) {
            aom_image_t* clean = aom_codec_get_frame(&decoders[1], &iters[1]);
            ASSERT_NE(nullptr, clean);
            uint64_t hash = 14695981039346656037ULL;
            uint64_t sum = 0;
            for (int plane = 0; plane < 3; ++plane) {
                const unsigned width = image->d_w >> (plane ? 1 : 0);
                const unsigned height = image->d_h >> (plane ? 1 : 0);
                for (unsigned y = 0; y < height; ++y) {
                    const uint8_t* row =
                        image->planes[plane] + y * image->stride[plane];
                    const uint8_t* clean_row =
                        clean->planes[plane] + y * clean->stride[plane];
                    for (unsigned x = 0; x < width; ++x) {
                        const int noise =
                            image->fmt & AOM_IMG_FMT_HIGHBITDEPTH
                                ? reinterpret_cast<const uint16_t*>(row)[x] -
                                      reinterpret_cast<const uint16_t*>(
                                          clean_row)[x]
                                : row[x] - clean_row[x];
                        sum += static_cast<uint64_t>(noise * noise);
                        hash = (hash ^ static_cast<uint16_t>(noise)) *
                               1099511628211ULL;
                    }
                }
            }
            frames->push_back(hash);
            if (energy)
                energy->push_back(sum);
        }
        EXPECT_EQ(nullptr, aom_codec_get_frame(&decoders[1], &iters[1]));
    }
    for (auto& decoder : decoders)
        EXPECT_EQ(AOM_CODEC_OK, aom_codec_destroy(&decoder));
    ASSERT_EQ(frame_count, frames->size());
}

TEST(FilmGrainCliTest, TimedTable) {
    const char* path = std::getenv("SVT_GRAIN_CLI_OUTPUT");
    if (!path)
        GTEST_SKIP();
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input);
    char header[32];
    ASSERT_TRUE(input.read(header, sizeof(header)));
    Packets packets;
    uint8_t frame_header[12];
    while (input.read(reinterpret_cast<char*>(frame_header),
                      sizeof(frame_header))) {
        const uint32_t size =
            uint32_t(frame_header[0]) | uint32_t(frame_header[1]) << 8 |
            uint32_t(frame_header[2]) << 16 | uint32_t(frame_header[3]) << 24;
        ASSERT_LT(size, 1u << 24);
        packets.emplace_back(size);
        ASSERT_TRUE(
            input.read(reinterpret_cast<char*>(packets.back().data()), size));
    }
    std::vector<uint64_t> hashes, energy;
    ASSERT_NO_FATAL_FAILURE(decode_noise(packets, &hashes, &energy, 210));
    uint64_t light = 0, strong = 0;
    for (unsigned frame = 0; frame < energy.size(); ++frame) {
        if (frame < 120)
            EXPECT_GT(energy[frame], 0u) << frame;
        else
            EXPECT_EQ(0u, energy[frame]) << frame;
        if (frame < 60)
            light += energy[frame];
        else if (frame < 120)
            strong += energy[frame];
    }
    EXPECT_GT(strong, 4 * light);
}

class FilmGrainBitstreamTest : public ::testing::TestWithParam<GrainParam> {};

TEST_P(FilmGrainBitstreamTest, EventsFollowPicturesAndFallBackToTable) {
    const GrainParam prm = GetParam();
    Packets table, events, mixed;
    ASSERT_NO_FATAL_FAILURE(encode(prm, Table, &table));
    ASSERT_NO_FATAL_FAILURE(encode(prm, Events, &events));
    ASSERT_NO_FATAL_FAILURE(encode(prm, Mixed, &mixed));
    std::vector<uint64_t> table_frames, event_frames, mixed_frames;
    ASSERT_NO_FATAL_FAILURE(decode_noise(table, &table_frames));
    ASSERT_NO_FATAL_FAILURE(decode_noise(events, &event_frames));
    ASSERT_NO_FATAL_FAILURE(decode_noise(mixed, &mixed_frames));
    uint64_t no_grain = 14695981039346656037ULL;
    for (unsigned i = 0; i < kWidth * kHeight * 3 / 2; ++i)
        no_grain *= 1099511628211ULL;
    for (uint32_t i = 0; i < kFrames; ++i) {
        SCOPED_TRACE(i);
        const int kind = grain_kind(i);
        EXPECT_EQ(kind == 0   ? table_frames[i]
                  : kind == 1 ? event_frames[i]
                              : no_grain,
                  mixed_frames[i]);
        if (kind != 2) {
            EXPECT_NE(no_grain, mixed_frames[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TemporalGrain, FilmGrainBitstreamTest,
                         ::testing::Values(GrainParam{8, false, false},
                                           GrainParam{8, true, false},
                                           GrainParam{10, false, false},
                                           GrainParam{10, true, false},
                                           GrainParam{10, true, true}));
#endif
}  // namespace
