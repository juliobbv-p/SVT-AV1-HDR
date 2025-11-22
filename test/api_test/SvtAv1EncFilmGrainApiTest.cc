/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved.
 * Copyright (c) 2026 Halide Compression, LLC
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. See LICENSE-BSD2.md and
 * PATENTS.md for details.
 */

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

#if CONFIG_ENABLE_FILM_GRAIN
namespace {
class FilmGrainApiTest : public ::testing::TestWithParam<bool> {};

TEST_P(FilmGrainApiTest, RejectsBadEventsAndCanContinueEncoding) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    AomFilmGrain table{};
    table.scaling_shift = 8;
    table.ar_coeff_shift = 6;
    table.bit_depth = 8;
    cfg.encoder_bit_depth = 8;
    cfg.source_width = 320;
    cfg.source_height = 240;
    cfg.enc_mode = 10;
    cfg.level_of_parallelism = 1;
    cfg.fgs_table = GetParam() ? &table : nullptr;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));
    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, 320, 240, 8);
    ASSERT_EQ(EB_ErrorNone, src.open_source(0, 1));
    EbBufferHeaderType in{};
    in.size = sizeof(in);
    in.p_buffer = reinterpret_cast<uint8_t*>(src.get_next_frame());
    in.n_filled_len = src.get_frame_size();
    in.pic_type = EB_AV1_INVALID_PICTURE;
    EbPrivDataNode node{FILM_GRAIN_EVENT, nullptr, sizeof(table), nullptr};
    in.p_app_private = &node;
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc, &in));
    EXPECT_EQ(0u, in.flags);
    node.data = &table;
    node.size = sizeof(table) - 1;
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc, &in));
    EXPECT_EQ(0u, in.flags);
    node.size = sizeof(table) + 1;
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc, &in));
    EXPECT_EQ(0u, in.flags);
    node.size = sizeof(table);
    if (GetParam()) {
        AomFilmGrain valid = table;
        valid.apply_grain = 1;
        valid.ar_coeff_lag = 1;
        valid.num_y_points = valid.num_cb_points = valid.num_cr_points = 2;
        valid.scaling_points_y[1][0] = valid.scaling_points_cb[1][0] =
            valid.scaling_points_cr[1][0] = 255;
        AomFilmGrain bad = valid;
        const auto reject = [&]() {
            node.data = &bad;
            EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc, &in));
            EXPECT_EQ(0u, in.flags);
            bad = valid;
        };
        bad.num_y_points = 15;
        reject();
        bad.num_y_points = -1;
        reject();
        bad.num_cb_points = 11;
        reject();
        bad.num_cr_points = 11;
        reject();
        bad.num_cb_points = 0;
        reject();
        bad.ar_coeff_lag = 4;
        reject();
        bad.ar_coeff_shift = 5;
        reject();
        bad.scaling_shift = 12;
        reject();
        bad.grain_scale_shift = -1;
        reject();
        bad.bit_depth = 10;
        reject();
        bad.apply_grain = 2;
        reject();
        bad.clip_to_restricted_range = 2;
        reject();
        bad.scaling_points_y[1][0] = 0;
        reject();
        bad.scaling_points_cb[0][1] = 256;
        reject();
        bad.ar_coeffs_y[0] = 128;
        reject();
        bad.ar_coeffs_cb[4] = -129;
        reject();
        bad.cb_mult = 256;
        reject();
        bad.cr_offset = 512;
        reject();
        node.data = &table;
    }
    if (!GetParam()) {
        EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc, &in));
        EXPECT_EQ(0u, in.flags);
        in.p_app_private = nullptr;
    }
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in));
    EbBufferHeaderType end{};
    end.size = sizeof(end);
    end.flags = EB_BUFFERFLAG_EOS;
    end.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &end));
    bool eos = false;
    unsigned frames = 0;
    EbBufferHeaderType* packet = nullptr;
    while (!eos && svt_av1_enc_get_packet(enc, &packet, true) == EB_ErrorNone) {
        eos = (packet->flags & EB_BUFFERFLAG_EOS) != 0;
        frames += packet->n_filled_len != 0;
        svt_av1_enc_release_out_buffer(&packet);
    }
    EXPECT_TRUE(eos);
    EXPECT_EQ(1u, frames);
    src.close_source();
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(enc));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
}

INSTANTIATE_TEST_SUITE_P(Signaling, FilmGrainApiTest, ::testing::Bool());
}  // namespace
#endif
