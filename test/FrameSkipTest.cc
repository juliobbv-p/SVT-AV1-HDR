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

extern "C" {
#include "rc_process.h"
#include "sequence_control_set.h"
#include "svt_threads.h"
}

#include "gtest/gtest.h"

namespace {

class FrameSkipRateControlTest : public ::testing::Test {
  protected:
    void SetUp() override {
        rc_.rc_mutex = svt_create_mutex();
        ASSERT_NE(nullptr, rc_.rc_mutex);
        scs_.static_config.max_allowed_consecutive_frames_skips = 2;
    }

    void TearDown() override {
        ASSERT_EQ(EB_ErrorNone, svt_destroy_mutex(rc_.rc_mutex));
    }

    SequenceControlSet scs_{};
    RATE_CONTROL rc_{};
};

TEST_F(FrameSkipRateControlTest,
       OutstandingReservationBlocksCreditPublication) {
    rc_.avg_frame_bandwidth = 100;
    rc_.maximum_buffer_size = 200;
    rc_.buffer_level = 400;
    rc_.frame_skip_credits = 0;
    rc_.frame_skip_reservations = 1;

    svt_av1_rc_publish_frame_skip_credits(&scs_, &rc_);
    EXPECT_EQ(static_cast<uint8_t>(0), rc_.frame_skip_credits);

    rc_.frame_skip_reservations = 0;
    svt_av1_rc_publish_frame_skip_credits(&scs_, &rc_);
    EXPECT_EQ(static_cast<uint8_t>(2), rc_.frame_skip_credits);
}

TEST_F(FrameSkipRateControlTest, DrainPayloadIsConsumedOnce) {
    rc_.buffer_level = 400;
    rc_.frame_skip_reservations = 2;
    int64_t drain_bits = 200;
    uint32_t drain_count = 2;

    svt_av1_rc_apply_frame_skip_drain(&rc_, &drain_bits, &drain_count);
    EXPECT_EQ(200, rc_.buffer_level);
    EXPECT_EQ(0U, rc_.frame_skip_reservations);
    EXPECT_EQ(0, drain_bits);
    EXPECT_EQ(0U, drain_count);

    svt_av1_rc_apply_frame_skip_drain(&rc_, &drain_bits, &drain_count);
    EXPECT_EQ(200, rc_.buffer_level);
    EXPECT_EQ(0U, rc_.frame_skip_reservations);
}

}  // namespace
