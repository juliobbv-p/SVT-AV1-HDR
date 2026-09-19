/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 3 Clause Clear License
 * and the Alliance for Open Media Patent License 1.0. If the BSD 3 Clause Clear
 * License was not distributed with this source code in the LICENSE file, you
 * can obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

/******************************************************************************
 * @file SvtAv1EncRtcResolutionApiTest.cc
 *
 * @brief Regression coverage for RTC low-delay encoding at production
 *        resolutions (>=720p/1080p).
 *
 * The static-64x64 ME bypass (MR !2676) only engages on larger frames that
 * contain static regions, and no API/E2E test exercised the RTC low-delay
 * path at >=1080p. As a result a dirty-MV out-of-bounds read shipped on master
 * undetected: the bypass returned without init_me_hme_data(), so ref/list
 * slots other than list0/ref0 kept a stale MV from a previously processed b64,
 * and inter prediction then read a reference block past the reference picture
 * buffer (heap-buffer-overflow in svt_av1_convolve_2d_copy_sr_neon, via
 * svt_inter_predictor_light_pd0).
 *
 * These tests run multi-frame RTC low-delay encodes across resolutions and
 * assert the encode completes and emits valid packets. The first inter frame
 * is where the regression faulted, so the loop deliberately runs well past it.
 * In a release build a triggering reference read may simply return garbage; the
 * out-of-bounds access is caught deterministically under a sanitizer build
 * (CI runs -DSANITIZER=Address), which is the intended guard for this class.
 ******************************************************************************/

#include <cstdint>
#include <tuple>

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

namespace {

// (width, height, enc_mode) — covers the repro resolution plus a smaller
// control, and an unaligned height that exercises partial edge superblocks.
using RtcResParam = std::tuple<uint32_t, uint32_t, uint8_t>;

class RtcResolutionEncodeTest : public ::testing::TestWithParam<RtcResParam> {
  protected:
    EbComponentType* enc_{nullptr};
    EbSvtAv1EncConfiguration cfg_{};

    void SetUp() override {
        const uint32_t width = std::get<0>(GetParam());
        const uint32_t height = std::get<1>(GetParam());
        const uint8_t enc_mode = std::get<2>(GetParam());

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc_, &cfg_));
        cfg_.source_width = width;
        cfg_.source_height = height;
        cfg_.frame_rate_numerator = 30;
        cfg_.frame_rate_denominator = 1;
        cfg_.encoder_bit_depth = 8;
        cfg_.encoder_color_format = EB_YUV420;
        cfg_.enc_mode = enc_mode;
        cfg_.rtc = true;
        cfg_.pred_structure = LOW_DELAY;
        cfg_.hierarchical_levels = 0;
        cfg_.intra_period_length = 30;
        cfg_.look_ahead_distance = 0;
        cfg_.recode_loop = 0;
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

TEST_P(RtcResolutionEncodeTest, EncodesWithoutOutOfBoundsReference) {
    const uint32_t width = std::get<0>(GetParam());
    const uint32_t height = std::get<1>(GetParam());
    // Enough inter frames to pass the first one (where the !2676 regression
    // faulted) and let the static-64x64 ME bypass engage across several SBs.
    constexpr uint32_t kNumFrames = 24;

    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, width, height, 8);
    ASSERT_EQ(EB_ErrorNone, src.open_source(0, kNumFrames));

    size_t packets = 0;
    for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(frame, nullptr);

        EbBufferHeaderType in_hdr{};
        in_hdr.size = sizeof(EbBufferHeaderType);
        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc_, &in_hdr));

        // Low-delay get_packet blocks until the frame is produced; drain once
        // per sent frame to keep send/produce balanced.
        EbBufferHeaderType* out = nullptr;
        if (svt_av1_enc_get_packet(enc_, &out, 0) == EB_ErrorNone && out) {
            if (!(out->flags & EB_BUFFERFLAG_EOS)) {
                ++packets;
            }
            svt_av1_enc_release_out_buffer(&out);
        }
    }

    // Flush and drain the remaining packets.
    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc_, &eos));
    for (;;) {
        EbBufferHeaderType* out = nullptr;
        if (svt_av1_enc_get_packet(enc_, &out, 1) != EB_ErrorNone || !out) {
            break;
        }
        const bool is_eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos) {
            ++packets;
        }
        svt_av1_enc_release_out_buffer(&out);
        if (is_eos) {
            break;
        }
    }

    EXPECT_GT(packets, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    SvtAv1Enc, RtcResolutionEncodeTest,
    ::testing::Values(RtcResParam{1920, 1080, 7},   // repro resolution/preset
                      RtcResParam{1920, 1080, 12},  // LPD0-heavy fast preset
                      RtcResParam{1920, 1088, 7},  // unaligned height: edge SBs
                      RtcResParam{1280, 720, 7}));  // smaller control

}  // namespace
