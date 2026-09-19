/*
 * Copyright(c) 2026 Meta Platforms, Inc.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file SvtAv1EncRefMgmtApiTest.cc
 *
 * @brief SVT-AV1 reference-frame management public API tests.
 *
 * Covers:
 *   - Enum / struct layout for REF_STORE_EVENT, REF_CLEAR_EVENT,
 *     REF_USE_EVENT, and SvtAv1RefFrameCmd.
 *   - EbSvtAv1EncConfiguration::max_managed_refs init-time validation.
 *   - FAIL-HARD per-event validation (synchronous return of
 *     EB_ErrorBadParameter from svt_av1_enc_send_picture for malformed
 *     events, disabled feature, pic_id collisions, duplicate event
 *     types).
 *   - Happy-path smoke: a sequence of STORE / USE / CLEAR events on a
 *     valid config encodes without error.
 *
 * Bitstream-level correctness (refresh_frame_flags / corruption checks
 * via libaom AOMD_GET_LAST_REF_UPDATES) is covered in the companion
 * e2e_test/SvtAv1RefMgmtBitstreamTest.cc.
 ******************************************************************************/

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "SvtAv1EncApiTest.h"
#include "gtest/gtest.h"

using namespace svt_av1_test;

namespace {

/* ------------------------------------------------------------------------- */
/* Header-only checks.                                                       */
/* ------------------------------------------------------------------------- */

TEST(RefMgmtApiHeaderTest, enum_values_present) {
    constexpr PrivDataType s = REF_STORE_EVENT;
    constexpr PrivDataType c = REF_CLEAR_EVENT;
    constexpr PrivDataType u = REF_USE_EVENT;
    EXPECT_NE(static_cast<int>(s), static_cast<int>(c));
    EXPECT_NE(static_cast<int>(s), static_cast<int>(u));
    EXPECT_NE(static_cast<int>(c), static_cast<int>(u));
    EXPECT_LT(static_cast<int>(s), static_cast<int>(PRIVATE_DATA_TYPES));
    EXPECT_LT(static_cast<int>(c), static_cast<int>(PRIVATE_DATA_TYPES));
    EXPECT_LT(static_cast<int>(u), static_cast<int>(PRIVATE_DATA_TYPES));
}

TEST(RefMgmtApiHeaderTest, payload_structs_compile) {
    SvtAv1RefFrameCmd s{};
    SvtAv1RefFrameCmd c{};
    SvtAv1RefFrameCmd u{};
    s.pic_id = 0x12345678u;
    c.pic_id = 0xaabbccddu;
    u.pic_id = 0x9abcdef0u;
    EXPECT_EQ(s.pic_id, 0x12345678u);
    EXPECT_EQ(c.pic_id, 0xaabbccddu);
    EXPECT_EQ(u.pic_id, 0x9abcdef0u);
    EXPECT_EQ(sizeof(s), sizeof(uint32_t));
}

/* ------------------------------------------------------------------------- */
/* Setting validation — svt_av1_enc_set_parameter end-to-end.                */
/* ------------------------------------------------------------------------- */

class RefMgmtSettingTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ASSERT_EQ(
            EB_ErrorNone,
            svt_av1_enc_init_handle(&ctxt_.enc_handle, &ctxt_.enc_params));
        ASSERT_NE(nullptr, ctxt_.enc_handle);
        ctxt_.enc_params.source_width = 640;
        ctxt_.enc_params.source_height = 480;
        // Default preset (ENC_M8) is below MIN_ENC_PRESET in RTC builds
        // (ENC_M9); set an explicitly valid preset so these tests exercise
        // ref-mgmt validation rather than tripping the preset range check.
        ctxt_.enc_params.enc_mode = 9;
    }
    void TearDown() override {
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit(ctxt_.enc_handle));
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(ctxt_.enc_handle));
    }

    SvtAv1Context ctxt_{};
};

TEST_F(RefMgmtSettingTest, default_max_managed_refs_is_zero) {
    EXPECT_EQ(ctxt_.enc_params.max_managed_refs, 0u);
}

TEST_F(RefMgmtSettingTest, valid_range_in_low_delay) {
    ctxt_.enc_params.pred_structure = LOW_DELAY;
    ctxt_.enc_params.rtc = 1;
    ctxt_.enc_params.rate_control_mode = SVT_AV1_RC_MODE_CBR;
    ctxt_.enc_params.target_bit_rate = 1000000;
    for (uint8_t v = 0; v <= 4; ++v) {
        ctxt_.enc_params.max_managed_refs = v;
        EXPECT_EQ(
            EB_ErrorNone,
            svt_av1_enc_set_parameter(ctxt_.enc_handle, &ctxt_.enc_params))
            << "max_managed_refs=" << static_cast<int>(v) << " rejected";
    }
}

TEST_F(RefMgmtSettingTest, out_of_range_rejected) {
    ctxt_.enc_params.pred_structure = LOW_DELAY;
    ctxt_.enc_params.rtc = 1;
    ctxt_.enc_params.rate_control_mode = SVT_AV1_RC_MODE_CBR;
    ctxt_.enc_params.target_bit_rate = 1000000;
    for (uint8_t v = 5; v < 16; ++v) {
        ctxt_.enc_params.max_managed_refs = v;
        EXPECT_EQ(
            EB_ErrorBadParameter,
            svt_av1_enc_set_parameter(ctxt_.enc_handle, &ctxt_.enc_params))
            << "out-of-range max_managed_refs=" << static_cast<int>(v)
            << " accepted";
    }
}

TEST_F(RefMgmtSettingTest, requires_low_delay) {
    ctxt_.enc_params.pred_structure = RANDOM_ACCESS;
    ctxt_.enc_params.hierarchical_levels = 4;
    ctxt_.enc_params.max_managed_refs = 3;
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_set_parameter(ctxt_.enc_handle, &ctxt_.enc_params));
}

TEST_F(RefMgmtSettingTest, zero_allowed_outside_low_delay) {
    ctxt_.enc_params.pred_structure = RANDOM_ACCESS;
    ctxt_.enc_params.hierarchical_levels = 4;
    ctxt_.enc_params.max_managed_refs = 0;
    EXPECT_EQ(EB_ErrorNone,
              svt_av1_enc_set_parameter(ctxt_.enc_handle, &ctxt_.enc_params));
}

/* ------------------------------------------------------------------------- */
/* FAIL-HARD per-event validation: svt_av1_enc_send_picture must return      */
/* EB_ErrorBadParameter for caller misuse the synchronous path can detect.   */
/* ------------------------------------------------------------------------- */

namespace failhard {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;

class RefMgmtFailHardTest : public ::testing::Test {
  protected:
    EbComponentType* enc_{nullptr};
    EbSvtAv1EncConfiguration cfg_{};

    void init_with(uint8_t max_managed_refs, PredStructure pred = LOW_DELAY) {
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc_, &cfg_));
        cfg_.source_width = kWidth;
        cfg_.source_height = kHeight;
        cfg_.frame_rate_numerator = 30000;
        cfg_.frame_rate_denominator = 1000;
        cfg_.encoder_bit_depth = 8;
        cfg_.encoder_color_format = EB_YUV420;
        cfg_.enc_mode = 9;
        cfg_.tune = 1;
        cfg_.rtc = (pred == LOW_DELAY);
        cfg_.intra_period_length = 100;
        cfg_.hierarchical_levels = (pred == LOW_DELAY) ? 0 : 4;
        cfg_.pred_structure = pred;
        cfg_.rate_control_mode =
            (pred == LOW_DELAY) ? SVT_AV1_RC_MODE_CBR : SVT_AV1_RC_MODE_VBR;
        cfg_.target_bit_rate = 500000;
        cfg_.max_qp_allowed = 63;
        cfg_.min_qp_allowed = 4;
        cfg_.look_ahead_distance = 0;
        cfg_.recode_loop = 0;
        cfg_.max_managed_refs = max_managed_refs;

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc_, &cfg_));
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc_));
    }

    void TearDown() override {
        if (enc_) {
            svt_av1_enc_deinit(enc_);
            svt_av1_enc_deinit_handle(enc_);
            enc_ = nullptr;
        }
    }

    /* Build a minimal one-frame send with a single ref-mgmt event node. */
    EbErrorType send_with_event(PrivDataType type, SvtAv1RefFrameCmd* payload) {
        svt_av1_video_source::DummyVideoSource src(
            IMG_FMT_420, kWidth, kHeight, 8);
        if (src.open_source(0, 0) != 0) {
            return EB_ErrorBadParameter;
        }
        EbSvtIOFormat* frame = src.get_next_frame();
        if (!frame) {
            return EB_ErrorBadParameter;
        }
        EbPrivDataNode node{};
        node.node_type = type;
        node.data = payload;
        node.size = payload ? sizeof(SvtAv1RefFrameCmd) : 0;
        node.next = nullptr;

        EbBufferHeaderType in_hdr{};
        in_hdr.size = sizeof(EbBufferHeaderType);
        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = 0;
        in_hdr.flags = 0;
        in_hdr.pic_type = EB_AV1_KEY_PICTURE;
        in_hdr.p_app_private = &node;
        return svt_av1_enc_send_picture(enc_, &in_hdr);
    }
};

TEST_F(RefMgmtFailHardTest, store_with_pic_id_zero_rejected) {
    init_with(/*max_managed_refs=*/4);
    SvtAv1RefFrameCmd payload{};
    payload.pic_id = 0u;  // reserved sentinel
    EXPECT_EQ(EB_ErrorBadParameter, send_with_event(REF_STORE_EVENT, &payload));
}

TEST_F(RefMgmtFailHardTest, clear_with_pic_id_zero_rejected) {
    init_with(/*max_managed_refs=*/4);
    SvtAv1RefFrameCmd payload{};
    payload.pic_id = 0u;
    EXPECT_EQ(EB_ErrorBadParameter, send_with_event(REF_CLEAR_EVENT, &payload));
}

TEST_F(RefMgmtFailHardTest, use_with_pic_id_zero_rejected) {
    init_with(/*max_managed_refs=*/4);
    SvtAv1RefFrameCmd payload{};
    payload.pic_id = 0u;
    EXPECT_EQ(EB_ErrorBadParameter, send_with_event(REF_USE_EVENT, &payload));
}

TEST_F(RefMgmtFailHardTest, store_with_feature_disabled_rejected) {
    init_with(/*max_managed_refs=*/0);  // feature OFF
    SvtAv1RefFrameCmd payload{};
    payload.pic_id = 42u;
    EXPECT_EQ(EB_ErrorBadParameter, send_with_event(REF_STORE_EVENT, &payload));
}

TEST_F(RefMgmtFailHardTest, malformed_payload_rejected) {
    init_with(/*max_managed_refs=*/4);
    EXPECT_EQ(EB_ErrorBadParameter, send_with_event(REF_STORE_EVENT, nullptr));
}

TEST_F(RefMgmtFailHardTest, duplicate_event_type_in_chain_rejected) {
    init_with(/*max_managed_refs=*/4);
    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(0, src.open_source(0, 0));
    EbSvtIOFormat* frame = src.get_next_frame();
    ASSERT_NE(nullptr, frame);

    SvtAv1RefFrameCmd a{42u};
    SvtAv1RefFrameCmd b{43u};
    EbPrivDataNode node_b{REF_STORE_EVENT, &b, sizeof(b), nullptr};
    EbPrivDataNode node_a{REF_STORE_EVENT, &a, sizeof(a), &node_b};

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);
    in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
    in_hdr.n_filled_len = src.get_frame_size();
    in_hdr.pts = 0;
    in_hdr.flags = 0;
    in_hdr.pic_type = EB_AV1_KEY_PICTURE;
    in_hdr.p_app_private = &node_a;
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc_, &in_hdr));
}

TEST_F(RefMgmtFailHardTest, pic_id_collision_across_events_rejected) {
    init_with(/*max_managed_refs=*/4);
    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(0, src.open_source(0, 0));
    EbSvtIOFormat* frame = src.get_next_frame();
    ASSERT_NE(nullptr, frame);

    /* STORE pic_id=42 chained with USE pic_id=42 — collision. */
    SvtAv1RefFrameCmd store{42u};
    SvtAv1RefFrameCmd use{42u};
    EbPrivDataNode node_use{REF_USE_EVENT, &use, sizeof(use), nullptr};
    EbPrivDataNode node_store{
        REF_STORE_EVENT, &store, sizeof(store), &node_use};

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);
    in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
    in_hdr.n_filled_len = src.get_frame_size();
    in_hdr.pts = 0;
    in_hdr.flags = 0;
    in_hdr.pic_type = EB_AV1_KEY_PICTURE;
    in_hdr.p_app_private = &node_store;
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(enc_, &in_hdr));
}

}  // namespace failhard

/* ------------------------------------------------------------------------- */
/* Happy-path smoke: STORE / USE / CLEAR sequence encodes without error.     */
/* Correctness of the encoded bitstream is verified in the companion         */
/* e2e_test SvtAv1RefMgmtBitstreamTest.                                      */
/* ------------------------------------------------------------------------- */

namespace happy {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
constexpr uint32_t kNumFrames = 24;
constexpr uint32_t kStoreFrame = 5;
constexpr uint32_t kUseFrame = 12;
constexpr uint32_t kClearFrame = 18;
constexpr uint32_t kStoreId = 42u;

class RefMgmtSmokeTest : public ::testing::Test {
  protected:
    EbComponentType* enc_{nullptr};
    EbSvtAv1EncConfiguration cfg_{};

    void SetUp() override {
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc_, &cfg_));
        cfg_.source_width = kWidth;
        cfg_.source_height = kHeight;
        cfg_.frame_rate_numerator = 30000;
        cfg_.frame_rate_denominator = 1000;
        cfg_.encoder_bit_depth = 8;
        cfg_.encoder_color_format = EB_YUV420;
        cfg_.enc_mode = 9;
        cfg_.tune = 1;
        cfg_.rtc = true;
        cfg_.intra_period_length = 100;
        cfg_.hierarchical_levels = 0;
        cfg_.pred_structure = LOW_DELAY;
        cfg_.rate_control_mode = SVT_AV1_RC_MODE_CBR;
        cfg_.target_bit_rate = 500000;
        cfg_.max_qp_allowed = 63;
        cfg_.min_qp_allowed = 4;
        cfg_.look_ahead_distance = 0;
        cfg_.recode_loop = 0;
        cfg_.max_managed_refs = 4;

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc_, &cfg_));
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc_));
    }

    void TearDown() override {
        if (enc_) {
            svt_av1_enc_deinit(enc_);
            svt_av1_enc_deinit_handle(enc_);
            enc_ = nullptr;
        }
    }
};

TEST_F(RefMgmtSmokeTest, store_use_clear_sequence_encodes_without_error) {
    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(0, src.open_source(0, 0));

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);

    size_t out_packets = 0;
    for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);

        SvtAv1RefFrameCmd store_payload{kStoreId};
        SvtAv1RefFrameCmd use_payload{kStoreId};
        SvtAv1RefFrameCmd clear_payload{kStoreId};
        EbPrivDataNode node{};

        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.flags = 0;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
        in_hdr.p_app_private = nullptr;

        if (fi == kStoreFrame) {
            node.node_type = REF_STORE_EVENT;
            node.data = &store_payload;
            node.size = sizeof(store_payload);
            in_hdr.p_app_private = &node;
        } else if (fi == kUseFrame) {
            node.node_type = REF_USE_EVENT;
            node.data = &use_payload;
            node.size = sizeof(use_payload);
            in_hdr.p_app_private = &node;
        } else if (fi == kClearFrame) {
            node.node_type = REF_CLEAR_EVENT;
            node.data = &clear_payload;
            node.size = sizeof(clear_payload);
            in_hdr.p_app_private = &node;
        }

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc_, &in_hdr))
            << "send_picture failed on frame " << fi;
        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc_, &out, 0);
        if (rc == EB_ErrorNone && out) {
            ++out_packets;
            svt_av1_enc_release_out_buffer(&out);
        }
    }
    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc_, &eos));
    for (;;) {
        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc_, &out, 1);
        if (rc != EB_ErrorNone || !out) {
            break;
        }
        const bool is_eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos) {
            ++out_packets;
        }
        svt_av1_enc_release_out_buffer(&out);
        if (is_eos) {
            break;
        }
    }

    EXPECT_GE(out_packets, static_cast<size_t>(kClearFrame + 1u))
        << "encoder only emitted " << out_packets << " non-EOS packets";
}

}  // namespace happy

}  // namespace
