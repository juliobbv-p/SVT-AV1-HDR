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
 * @file FrameIdTest.cc
 *
 * @brief Unit test of the AV1 frame_id id/delta arithmetic used for
 * application-managed LTR:
 *   - svt_aom_frame_id_from_pic_num id domain + wrap behavior
 *   - the per-ref delta_frame_id_minus1 formula (entropy_coding.c)
 *   - the apply_ref_use over-age gate boundary (pd_process.c)
 *
 * A short encode can never reach the 2^15 id wrap nor the 2^14 delta boundary,
 * so those are covered here directly and cheaply.
 ******************************************************************************/

#include <cstdint>
#include "gtest/gtest.h"
#include "definitions.h"

namespace {

constexpr uint32_t kIdSpan =
    1u << FRAME_ID_LENGTH;               // id domain and delta modulus (2^15)
constexpr uint32_t kIdPeriod = kIdSpan;  // ids step by one over the full span
constexpr uint32_t kMaxDelta = 1u << DELTA_FRAME_ID_LENGTH;  // 2^14

// Thin wrappers over the shipped helpers, so a change to either is observed
// here rather than passing against a private copy.
static int32_t delta_frame_id_minus1(uint32_t current_id, uint32_t ref_id) {
    return svt_aom_frame_id_delta_minus1(current_id, ref_id, FRAME_ID_LENGTH);
}

static bool use_rejected(uint64_t age) {
    return svt_aom_frame_id_age_unusable(age, DELTA_FRAME_ID_LENGTH);
}

// the decoder's mark_ref_frames: the decoder marks a reference invalid when its
// id falls outside the delta window. Returns true if the decoder keeps it.
static bool decoder_keeps_ref(uint32_t current_id, uint32_t ref_id) {
    if (current_id > kMaxDelta) {
        return !(ref_id > current_id || ref_id < current_id - kMaxDelta);
    }
    return !(ref_id > current_id &&
             ref_id < (kIdSpan + current_id - kMaxDelta));
}

// The encoder's USE gate must accept exactly the anchors the decoder still
// holds. More permissive and it emits a reference the decoder has dropped;
// stricter and it refuses usable anchors. Swept over real picture ages, not id
// pairs -- an id pair cannot express age beyond one period, which is precisely
// the case the gate exists to catch.
TEST(FrameId, UseGateMatchesDecoderMarkRefFrames) {
    const uint64_t anchor_poc = 1234;
    // Strictly inside one id period the ids are distinct and ordered, so the
    // id-window model is an independent oracle for the gate.
    for (uint64_t age = 1; age < kIdSpan; ++age) {
        const uint32_t anchor = svt_aom_frame_id_from_pic_num(anchor_poc);
        const uint32_t current =
            svt_aom_frame_id_from_pic_num(anchor_poc + age);
        EXPECT_EQ(decoder_keeps_ref(current, anchor), !use_rejected(age))
            << "gate disagrees with mark_ref_frames at age=" << age;
    }
    // From one full period on, the ids coincide and then realign, so the model
    // reads these as recent. The decoder's RefValid stays cleared, so the gate
    // must keep rejecting exactly where the id-window model cannot see.
    for (uint64_t age = kIdSpan; age <= kIdSpan + kMaxDelta; age += 97) {
        const uint32_t anchor = svt_aom_frame_id_from_pic_num(anchor_poc);
        const uint32_t current =
            svt_aom_frame_id_from_pic_num(anchor_poc + age);
        EXPECT_TRUE(decoder_keeps_ref(current, anchor))
            << "expected the id model to alias at age=" << age;
        EXPECT_TRUE(use_rejected(age))
            << "over-age anchor accepted at age=" << age;
    }
}

// Call-duration framing of the two limits, at 30 fps.
//   2^14 frames = 16384 ~= 9.1 min  -- oldest expressible reference age
//   2^15        = 32768 ~= 18.2 min -- frame_id wrap period
// A short encode can never reach either, so they are checked as arithmetic.
TEST(FrameId, NineMinuteAnchorIsTheAcceptBoundary) {
    const uint64_t store_poc = 1000;  // anchor stored ~33 s in
    const uint32_t anchor = svt_aom_frame_id_from_pic_num(store_poc);
    // Exactly 9.1 min later: still accepted, and the writer emits the largest
    // legal delta -- the property the removed id-space check used to assert.
    const uint32_t at_limit =
        svt_aom_frame_id_from_pic_num(store_poc + kMaxDelta);
    EXPECT_FALSE(use_rejected(kMaxDelta));
    EXPECT_EQ((int32_t)kMaxDelta - 1, delta_frame_id_minus1(at_limit, anchor));
    EXPECT_TRUE(decoder_keeps_ref(at_limit, anchor));
    // One frame past: rejected, and the decoder would have dropped it too.
    const uint32_t past_limit =
        svt_aom_frame_id_from_pic_num(store_poc + kMaxDelta + 1);
    EXPECT_TRUE(use_rejected(kMaxDelta + 1));
    EXPECT_FALSE(decoder_keeps_ref(past_limit, anchor));
}

TEST(FrameId, EighteenMinuteWrapStaysConformant) {
    // Walk the ~18.2 min wrap frame by frame. Normal refs are 1-2 frames old
    // (slot 0 is refreshed every base frame), which is the case that must keep
    // working forever -- a call does not end at the wrap.
    for (uint64_t poc = kIdPeriod - 5; poc <= kIdPeriod + 5; ++poc) {
        const uint32_t cur = svt_aom_frame_id_from_pic_num(poc);
        const uint32_t prev = svt_aom_frame_id_from_pic_num(poc - 1);
        ASSERT_NE(cur, prev)
            << "consecutive ids must differ (poc=" << poc << ")";
        EXPECT_FALSE(use_rejected(/*age=*/1))
            << "a 1-frame-old ref must stay usable across the wrap";
        EXPECT_TRUE(decoder_keeps_ref(cur, prev));
        const int32_t d1 = delta_frame_id_minus1(cur, prev);
        EXPECT_GE(d1, 0);
        EXPECT_LT(d1, (int32_t)kMaxDelta);
    }
}

TEST(FrameId, IdDomainCoversFullSpanIncludingZero) {
    // Ids occupy the whole 15-bit domain; 0 is an ordinary value, not reserved.
    bool seen_zero = false, seen_top = false;
    for (uint64_t poc = 0; poc < 2ull * kIdPeriod + 5; ++poc) {
        const uint32_t id = svt_aom_frame_id_from_pic_num(poc);
        ASSERT_LT(id, kIdSpan)
            << "id does not fit FRAME_ID_LENGTH bits (poc=" << poc << ")";
        seen_zero |= (id == 0u);
        seen_top |= (id == kIdSpan - 1u);
    }
    EXPECT_TRUE(seen_zero) << "0 must be part of the id domain";
    EXPECT_TRUE(seen_top) << "the top id must be part of the domain";
}

TEST(FrameId, ConsecutiveIdsAlwaysStepByExactlyOne) {
    // The decoder rejects a current_frame_id equal to the previous frame's, and
    // requires the forward difference below 2^(FRAME_ID_LENGTH-1). Stepping by
    // one over the full span satisfies both everywhere, wrap included.
    uint32_t prev = svt_aom_frame_id_from_pic_num(0);
    for (uint64_t poc = 1; poc < 2ull * kIdPeriod + 5; ++poc) {
        const uint32_t id = svt_aom_frame_id_from_pic_num(poc);
        const uint32_t d = (id + kIdSpan - prev) % kIdSpan;
        ASSERT_EQ(1u, d) << "consecutive id step must be exactly 1 at poc="
                         << poc;
        prev = id;
    }
}

// The gate bounds age; the writer emits an id delta. This is what ties them
// together: every age the gate accepts produces a delta the writer can encode,
// and every age it rejects would overflow. apply_ref_use relies on this, which
// is why it needs no separate check on the delta itself.
TEST(FrameId, UseGateBoundsWriterExactly) {
    for (uint64_t anchor_poc :
         {(uint64_t)100, (uint64_t)kIdSpan - 3, (uint64_t)kIdSpan * 2 - 1}) {
        for (uint64_t age = 1; age <= kMaxDelta + 2; ++age) {
            const uint32_t anchor_id =
                svt_aom_frame_id_from_pic_num(anchor_poc);
            const uint32_t current_id =
                svt_aom_frame_id_from_pic_num(anchor_poc + age);
            const int32_t d1 = delta_frame_id_minus1(current_id, anchor_id);
            const bool writer_in_range = d1 >= 0 && d1 < (int32_t)kMaxDelta;
            EXPECT_EQ(!use_rejected(age), writer_in_range)
                << "gate/writer disagree at anchor_poc=" << anchor_poc
                << " age=" << age << " (d1=" << d1 << ")";
        }
    }
}

TEST(FrameId, DeltaEqualsAgeAcrossWrap) {
    // Anchor just before the wrap, current just after. Two properties: the
    // decoder recovers the anchor id as
    // (current_id - (delta_frame_id_minus1 + 1)) mod 2^15, and the emitted
    // delta equals the picture age exactly. The second only holds because ids
    // cover the full span -- omitting any value would make a difference
    // spanning it run one ahead of the age.
    const uint64_t anchor_poc = kIdPeriod - 3;  // id near the top of the domain
    for (uint64_t age = 1; age <= 8; ++age) {
        const uint32_t anchor_id = svt_aom_frame_id_from_pic_num(anchor_poc);
        const uint32_t current_id =
            svt_aom_frame_id_from_pic_num(anchor_poc + age);
        ASSERT_FALSE(use_rejected(age))
            << "small-age USE across the wrap wrongly rejected (age=" << age
            << ")";
        const int32_t d1 = delta_frame_id_minus1(current_id, anchor_id);
        ASSERT_GE(d1, 0);
        ASSERT_LT(d1, (int32_t)kMaxDelta);
        const uint32_t recovered =
            (uint32_t)(((uint64_t)current_id + kIdSpan - (uint64_t)(d1 + 1)) %
                       kIdSpan);
        EXPECT_EQ(anchor_id, recovered)
            << "decoder cannot recover the anchor id at age=" << age;
        EXPECT_EQ((int32_t)age - 1, d1)
            << "delta must equal the picture age at age=" << age;
    }
}

// Why the gate must not be written in id space. Ids wrap, so an anchor older
// than one full period produces a small id difference again -- but the decoder
// cleared that slot's RefValid long before (mark_ref_frames) and only
// a refresh restores it, which a held anchor never gets. Bounding picture age
// is what rejects these.
TEST(FrameId, GateRejectsAnchorsOlderThanOnePeriodDespiteSmallIdDelta) {
    const uint64_t anchor_poc = 0;
    const uint32_t anchor_id = svt_aom_frame_id_from_pic_num(anchor_poc);

    for (uint64_t age = kIdSpan; age <= kIdSpan + 4; ++age) {
        const uint32_t current_id =
            svt_aom_frame_id_from_pic_num(anchor_poc + age);
        const uint32_t id_delta = (current_id + kIdSpan - anchor_id) % kIdSpan;
        // The id difference looks recent...
        EXPECT_LE(id_delta, kMaxDelta)
            << "expected an aliased id delta at age=" << age;
        // ...but the gate rejects on age.
        EXPECT_TRUE(use_rejected(age))
            << "over-age anchor accepted at age=" << age;
    }

    EXPECT_FALSE(use_rejected(kMaxDelta))
        << "age exactly at the window must stay usable";
    EXPECT_TRUE(use_rejected(kMaxDelta + 1));
    EXPECT_TRUE(use_rejected(0)) << "an anchor cannot be the current picture";
}

}  // namespace
