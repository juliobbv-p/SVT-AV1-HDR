/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
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
 * @file SvtAv1EncFrameSkipApiTest.cc
 *
 * @brief Regression tests for pre-encode frame skipping in RTC low-delay CBR.
 ******************************************************************************/

#include <cstdint>

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

namespace {

constexpr uint16_t kWidth = 320;
constexpr uint16_t kHeight = 240;

template <typename T>
EbPrivDataNode make_event(PrivDataType type, T* data,
                          EbPrivDataNode* next = nullptr) {
    EbPrivDataNode node{};
    node.node_type = type;
    node.data = data;
    node.size = sizeof(T);
    node.next = next;
    return node;
}

class Encoder {
  public:
    Encoder() : source_(IMG_FMT_420, kWidth, kHeight, 8) {
        EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&handle_, &config_));
        EXPECT_EQ(EB_ErrorNone, source_.open_source(0, 0));
        default_target_bit_rate_ = config_.target_bit_rate;

        config_.enc_mode = 9;
        config_.tune = 1;
        config_.rtc = true;
        config_.level_of_parallelism = 1;
        config_.intra_period_length = -1;
        config_.hierarchical_levels = 0;
        config_.pred_structure = LOW_DELAY;
        config_.source_width = kWidth;
        config_.source_height = kHeight;
        config_.frame_rate_numerator = 30;
        config_.frame_rate_denominator = 1;
        config_.encoder_bit_depth = 8;
        config_.encoder_color_format = EB_YUV420;
        config_.rate_control_mode = SVT_AV1_RC_MODE_CBR;
        config_.target_bit_rate = 1000;
        config_.max_qp_allowed = 63;
        config_.min_qp_allowed = 4;
        config_.under_shoot_pct = 100;
        config_.over_shoot_pct = 100;
        config_.maximum_buffer_size_ms = 600;
        config_.starting_buffer_level_ms = 599;
        config_.optimal_buffer_level_ms = 599;
        config_.look_ahead_distance = 0;
        config_.enable_overlays = false;
        config_.scene_change_detection = 0;
        config_.recode_loop = 4;
    }

    ~Encoder() {
        if (handle_ != nullptr) {
            if (initialized_) {
                EbBufferHeaderType eos{};
                eos.size = sizeof(eos);
                eos.flags = EB_BUFFERFLAG_EOS;
                eos.pic_type = EB_AV1_INVALID_PICTURE;
                svt_av1_enc_send_picture(handle_, &eos);
                svt_av1_enc_deinit(handle_);
            }
            svt_av1_enc_deinit_handle(handle_);
        }
        source_.close_source();
    }

    EbSvtAv1EncConfiguration& config() {
        return config_;
    }

    uint32_t default_target_bit_rate() const {
        return default_target_bit_rate_;
    }

    EbErrorType set_parameter() {
        return svt_av1_enc_set_parameter(handle_, &config_);
    }

    EbErrorType init() {
        const EbErrorType ret = svt_av1_enc_init(handle_);
        initialized_ = ret == EB_ErrorNone;
        return ret;
    }

    EbErrorType send(EbAv1PictureType pic_type,
                     EbPrivDataNode* private_data = nullptr) {
        EbSvtIOFormat* frame = source_.get_next_frame();
        if (frame == nullptr) {
            return EB_ErrorInsufficientResources;
        }

        EbBufferHeaderType input{};
        input.size = sizeof(input);
        input.p_buffer = reinterpret_cast<uint8_t*>(frame);
        input.n_alloc_len = source_.get_frame_size();
        input.n_filled_len = source_.get_frame_size();
        input.pts = pts_++;
        input.pic_type = pic_type;
        input.p_app_private = private_data;

        const EbErrorType ret = svt_av1_enc_send_picture(handle_, &input);
        if (ret == EB_ErrorNone) {
            drain_one_picture(input.pts);
        }
        return ret;
    }

  private:
    void drain_one_picture(int64_t expected_pts) {
        bool stop = false;
        while (!stop) {
            EbBufferHeaderType* output = nullptr;
            ASSERT_EQ(EB_ErrorNone,
                      svt_av1_enc_get_packet(handle_, &output, 0));
            ASSERT_NE(nullptr, output);
            stop = (output->flags & EB_BUFFERFLAG_EOS) ||
                   !(output->flags & EB_BUFFERFLAG_IS_ALT_REF);
            if (stop && !(output->flags & EB_BUFFERFLAG_EOS)) {
                EXPECT_EQ(expected_pts, output->pts);
            }
            svt_av1_enc_release_out_buffer(&output);
        }
    }

    EbComponentType* handle_ = nullptr;
    EbSvtAv1EncConfiguration config_{};
    uint32_t default_target_bit_rate_ = 0;
    bool initialized_ = false;
    int64_t pts_ = 0;
    svt_av1_video_source::DummyVideoSource source_;
};

TEST(SvtAv1FrameSkipApiTest, DefaultIsDisabled) {
    EbComponentType* handle = nullptr;
    EbSvtAv1EncConfiguration config{};
    config.max_allowed_consecutive_frames_skips = 0xff;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&handle, &config));
    EXPECT_EQ(0, config.max_allowed_consecutive_frames_skips);
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(handle));
}

TEST(SvtAv1FrameSkipApiTest, AcceptedOnlyForRtcLowDelayCbr) {
    Encoder valid;
    valid.config().max_allowed_consecutive_frames_skips = 2;
    EXPECT_EQ(EB_ErrorNone, valid.set_parameter());

    Encoder non_rtc;
    non_rtc.config().rtc = false;
    non_rtc.config().max_allowed_consecutive_frames_skips = 2;
    EXPECT_EQ(EB_ErrorBadParameter, non_rtc.set_parameter());

    Encoder non_cbr;
    non_cbr.config().rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    non_cbr.config().target_bit_rate = non_cbr.default_target_bit_rate();
    non_cbr.config().max_allowed_consecutive_frames_skips = 2;
    EXPECT_EQ(EB_ErrorBadParameter, non_cbr.set_parameter());
}

TEST(SvtAv1FrameSkipApiTest, ReturnsSuccessStatusAndCapsConsecutiveSkips) {
    Encoder encoder;
    encoder.config().max_allowed_consecutive_frames_skips = 2;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_INVALID_PICTURE));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
}

TEST(SvtAv1FrameSkipApiTest, SupportsTwoTemporalLayers) {
    Encoder encoder;
    encoder.config().hierarchical_levels = 1;
    encoder.config().max_allowed_consecutive_frames_skips = 1;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_INVALID_PICTURE));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
}

TEST(SvtAv1FrameSkipApiTest, KeyFramePreservesCreditForNextInterFrame) {
    Encoder encoder;
    encoder.config().max_allowed_consecutive_frames_skips = 1;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
}

TEST(SvtAv1FrameSkipApiTest, MgSizeCommandPreservesCreditForNextInterFrame) {
    Encoder encoder;
    encoder.config().max_allowed_consecutive_frames_skips = 1;
    encoder.config().max_hierarchical_levels = 1;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    SvtAv1MgSizeInfo info{};
    info.hierarchical_levels = 1;
    EbPrivDataNode node = make_event(MG_SIZE_CHANGE_EVENT, &info);
    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_INVALID_PICTURE, &node));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
}

TEST(SvtAv1FrameSkipApiTest, RefMgmtCommandPreservesCreditForNextInterFrame) {
    Encoder encoder;
    encoder.config().max_allowed_consecutive_frames_skips = 1;
    encoder.config().max_managed_refs = 1;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    SvtAv1RefFrameCmd command{};
    command.pic_id = 1;
    EbPrivDataNode node = make_event(REF_STORE_EVENT, &command);
    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_INVALID_PICTURE, &node));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
}

TEST(SvtAv1FrameSkipApiTest, RateChangeDoesNotProtectFrameFromSkipping) {
    Encoder encoder;
    encoder.config().max_allowed_consecutive_frames_skips = 1;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));
    SvtAv1RateInfo rate{};
    rate.target_bit_rate = 20000;
    EbPrivDataNode node = make_event(RATE_CHANGE_EVENT, &rate);
    EXPECT_EQ(EB_NoErrorFrameSkipped,
              encoder.send(EB_AV1_INVALID_PICTURE, &node));
    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_INVALID_PICTURE));
}

TEST(SvtAv1FrameSkipApiTest,
     RefFrameScalingCommandPreservesCreditForNextInterFrame) {
    Encoder encoder;
    encoder.config().max_allowed_consecutive_frames_skips = 1;
    ASSERT_EQ(EB_ErrorNone, encoder.set_parameter());
    ASSERT_EQ(EB_ErrorNone, encoder.init());

    ASSERT_EQ(EB_ErrorNone, encoder.send(EB_AV1_KEY_PICTURE));

    EbRefFrameScale scale{};
    scale.scale_mode = RESIZE_FIXED;
    scale.scale_denom = 12;
    scale.scale_kf_denom = 12;
    EbPrivDataNode node = make_event(REF_FRAME_SCALING_EVENT, &scale);

    EXPECT_EQ(EB_ErrorNone, encoder.send(EB_AV1_INVALID_PICTURE, &node));
    EXPECT_EQ(EB_NoErrorFrameSkipped, encoder.send(EB_AV1_INVALID_PICTURE));
}

}  // namespace
