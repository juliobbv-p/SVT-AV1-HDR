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
 * @file SvtAv1EncOnTheFlyApiTest.cc
 *
 * @brief Regression tests for on-the-fly setting-change rejection (issue
 *        #2364).
 *
 * A rejected on-the-fly setting change (rate / frame rate / preset /
 * resolution) must be a CLEAN REJECT: svt_av1_enc_send_picture() returns
 * EB_ErrorBadParameter and leaves the stream intact, so an application that
 * keeps sending — exactly what an RTC sender does when congestion control asks
 * for a bitrate the encoder can't honor — keeps encoding.
 *
 * The bug these tests guard against: the rejection used to stamp
 * EB_BUFFERFLAG_EOS on the caller's frame and latch end-of-stream, so the
 * encoder began its drain and stopped consuming input. A continuously-sending
 * application then blocked forever inside svt_av1_enc_send_picture() on the
 * input buffer pool — a deadlock.
 *
 * Each test runs its send loop on a worker thread guarded by a watchdog, so a
 * regression surfaces as a deterministic test failure instead of hanging the
 * whole suite.
 ******************************************************************************/

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
// Enough frames after the rejection to exhaust the input buffer pool, so a
// regression (EOS-latched drain that stops consuming input) actually deadlocks.
constexpr uint32_t kNumFrames = 64;
constexpr uint32_t kRejectFrame = 16;
// Generous bound: the fixed library encodes these tiny frames in well under a
// second; only a genuine deadlock reaches this.
constexpr int kWatchdogSec = 15;

/* Outcome of one send/drain pass, filled by the worker thread and read by the
 * main thread only after the worker has finished (no cross-thread gtest use).
 */
struct LoopResult {
    EbErrorType reject_rc = EB_ErrorNone;  // rc of the frame carrying the event
    EbErrorType first_post_reject_bad = EB_ErrorNone;
    bool all_post_reject_ok = true;  // every frame after reject was accepted
    size_t frames_accepted_after_reject = 0;
    size_t packets = 0;
    bool completed = false;
};

class OnTheFlyRejectTest : public ::testing::Test {
  protected:
    EbComponentType* enc_{nullptr};
    EbSvtAv1EncConfiguration cfg_{};
    bool hung_{false};

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
        // on-the-fly TBR change is only valid in RTC Low-Delay CBR
        cfg_.rate_control_mode = SVT_AV1_RC_MODE_CBR;
        cfg_.target_bit_rate = 500000;
        cfg_.max_qp_allowed = 63;
        cfg_.min_qp_allowed = 4;
        cfg_.look_ahead_distance = 0;
        cfg_.recode_loop = 0;

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc_, &cfg_));
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc_));
    }

    void TearDown() override {
        // On a detected deadlock the worker is still stuck inside the encoder;
        // do not tear it down (that would touch a live, locked instance).
        if (hung_ || !enc_) {
            return;
        }
        svt_av1_enc_deinit(enc_);
        svt_av1_enc_deinit_handle(enc_);
        enc_ = nullptr;
    }

    /* Send kNumFrames; at kRejectFrame attach a RATE_CHANGE_EVENT carrying
     * *evt* (nullptr = no event). Keep sending afterward, draining output as we
     * go, then flush. Writes the outcome to *out. Runs on a worker thread. */
    void run_loop(const SvtAv1RateInfo* evt, LoopResult* out) {
        svt_av1_video_source::DummyVideoSource src(
            IMG_FMT_420, kWidth, kHeight, 8);
        if (src.open_source(0, 0) != 0) {
            return;
        }

        EbBufferHeaderType in_hdr{};
        in_hdr.size = sizeof(EbBufferHeaderType);

        for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
            EbSvtIOFormat* frame = src.get_next_frame();
            if (!frame) {
                break;
            }

            // Per-iteration locals: send_picture copies the private-data list,
            // so they only need to outlive the call below.
            SvtAv1RateInfo evt_copy{};
            EbPrivDataNode node{};

            in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
            in_hdr.n_filled_len = src.get_frame_size();
            in_hdr.pts = fi;
            in_hdr.flags = 0;
            in_hdr.pic_type =
                (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
            in_hdr.p_app_private = nullptr;

            const bool is_reject_frame = (fi == kRejectFrame && evt != nullptr);
            if (is_reject_frame) {
                evt_copy = *evt;
                node.node_type = RATE_CHANGE_EVENT;
                node.data = &evt_copy;
                node.size = sizeof(SvtAv1RateInfo);
                node.next = nullptr;
                in_hdr.p_app_private = &node;
            }

            const EbErrorType rc = svt_av1_enc_send_picture(enc_, &in_hdr);

            if (is_reject_frame) {
                out->reject_rc = rc;
            } else if (evt != nullptr && fi > kRejectFrame) {
                if (rc == EB_ErrorNone) {
                    ++out->frames_accepted_after_reject;
                } else if (out->first_post_reject_bad == EB_ErrorNone) {
                    out->first_post_reject_bad = rc;
                    out->all_post_reject_ok = false;
                }
            }

            // In low-delay svt_av1_enc_get_packet() blocks regardless of the
            // done flag, and only a successfully-enqueued frame ever produces a
            // packet. Drain exactly once per accepted frame so get/produce
            // stays balanced; a rejected frame produces nothing, so draining
            // for it would block forever.
            if (rc == EB_ErrorNone) {
                EbBufferHeaderType* o = nullptr;
                if (svt_av1_enc_get_packet(enc_, &o, 0) == EB_ErrorNone && o) {
                    if (!(o->flags & EB_BUFFERFLAG_EOS)) {
                        ++out->packets;
                    }
                    svt_av1_enc_release_out_buffer(&o);
                }
            }
        }

        // Normal end-of-stream and final drain.
        EbBufferHeaderType eos{};
        eos.size = sizeof(EbBufferHeaderType);
        eos.flags = EB_BUFFERFLAG_EOS;
        eos.pic_type = EB_AV1_INVALID_PICTURE;
        svt_av1_enc_send_picture(enc_, &eos);
        for (;;) {
            EbBufferHeaderType* o = nullptr;
            const EbErrorType rc = svt_av1_enc_get_packet(enc_, &o, 1);
            if (rc != EB_ErrorNone || !o) {
                break;
            }
            const bool is_eos = (o->flags & EB_BUFFERFLAG_EOS) != 0;
            if (!is_eos) {
                ++out->packets;
            }
            svt_av1_enc_release_out_buffer(&o);
            if (is_eos) {
                break;
            }
        }

        out->completed = true;
    }

    /* Run run_loop() under a watchdog. Returns true if it finished in time;
     * false (and marks hung_) if it looks deadlocked. */
    bool run_with_watchdog(const SvtAv1RateInfo* evt, LoopResult* out) {
        std::atomic<bool> done{false};
        std::thread worker([&] {
            run_loop(evt, out);
            done.store(true, std::memory_order_release);
        });

        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(kWatchdogSec);
        while (!done.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        if (done.load(std::memory_order_acquire)) {
            worker.join();
            return true;
        }
        // Stuck inside a blocking encoder call (the regression deadlocks in
        // svt_av1_enc_send_picture); the worker cannot be joined.
        hung_ = true;
        worker.detach();
        return false;
    }
};

// A bitrate above the 100 Mbps cap is out of range and must be rejected — even
// on a correctly-configured RTC CBR encoder — without tearing down the stream.
TEST_F(OnTheFlyRejectTest, rejected_out_of_range_bitrate_does_not_deadlock) {
    SvtAv1RateInfo evt{};
    evt.seq_qp = 0;                   // 0 = do not override QP
    evt.target_bit_rate = 200000000;  // 200 Mbps, over the [0, 100000] kbps cap

    LoopResult r{};
    ASSERT_TRUE(run_with_watchdog(&evt, &r))
        << "svt_av1_enc_send_picture deadlocked after a rejected on-the-fly "
           "bitrate change — regression of #2364";
    EXPECT_EQ(EB_ErrorBadParameter, r.reject_rc)
        << "rejected bitrate change should report EB_ErrorBadParameter";
    EXPECT_TRUE(r.all_post_reject_ok)
        << "stream torn down: a frame after the rejected change returned "
        << r.first_post_reject_bad;
    EXPECT_GT(r.frames_accepted_after_reject, 0u)
        << "no frames accepted after the rejection";
    EXPECT_GT(r.packets, 0u) << "encoder produced no packets";
}

// A QP above 63 is out of range and must be rejected without tearing down the
// stream.
TEST_F(OnTheFlyRejectTest, rejected_out_of_range_qp_does_not_deadlock) {
    SvtAv1RateInfo evt{};
    evt.seq_qp = 100;  // > 63, out of range
    evt.target_bit_rate =
        cfg_.target_bit_rate;  // unchanged: isolate the QP reject

    LoopResult r{};
    ASSERT_TRUE(run_with_watchdog(&evt, &r))
        << "svt_av1_enc_send_picture deadlocked after a rejected on-the-fly "
           "QP change — regression of #2364";
    EXPECT_EQ(EB_ErrorBadParameter, r.reject_rc)
        << "rejected QP change should report EB_ErrorBadParameter";
    EXPECT_TRUE(r.all_post_reject_ok)
        << "stream torn down: a frame after the rejected change returned "
        << r.first_post_reject_bad;
    EXPECT_GT(r.frames_accepted_after_reject, 0u)
        << "no frames accepted after the rejection";
    EXPECT_GT(r.packets, 0u) << "encoder produced no packets";
}

// Sanity counterpart: a valid in-range bitrate change is accepted (guards
// against over-rejecting the happy path) and of course does not hang.
TEST_F(OnTheFlyRejectTest, valid_bitrate_change_is_accepted) {
    SvtAv1RateInfo evt{};
    evt.seq_qp = 0;
    evt.target_bit_rate =
        800000;  // in range, different from the configured TBR

    LoopResult r{};
    ASSERT_TRUE(run_with_watchdog(&evt, &r))
        << "encoder did not return for a valid on-the-fly bitrate change";
    EXPECT_EQ(EB_ErrorNone, r.reject_rc)
        << "valid bitrate change should be accepted";
    EXPECT_TRUE(r.all_post_reject_ok);
    EXPECT_GT(r.packets, 0u) << "encoder produced no packets";
}

}  // namespace
