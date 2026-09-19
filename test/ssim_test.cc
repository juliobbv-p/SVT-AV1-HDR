/*
 * Copyright(c) 2022 Cidana Co.,Ltd.
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
 * @file ResizeTest.cc
 *
 * @brief Unit test for resize of downsampling functions:
 * - svt_av1_resize_plane
 * - svt_av1_highbd_resize_plane
 *
 * @author Cidana-Edmond
 *
 ******************************************************************************/

#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include "aom_dsp_rtcd.h"
#include "random.h"

namespace {
using svt_av1_test_tool::SVTRandom;

constexpr int test_times = 20;

/** setup_test_env and reset_test_env are implemented in test/TestEnv.c */
extern "C" void setup_test_env();
extern "C" void reset_test_env();

template <typename Sample, int bd>
class SsimTest : public ::testing::Test {
  public:
    SsimTest() {
        setup_test_env();
    }

    void prepare_zero_src() {
        src_8x8_.fill(0);
        src_4x4_.fill(0);
    }

    void prepare_extreme_src() {
        constexpr auto max_sample = (1 << bd) - 1;
        src_8x8_.fill(max_sample);
        src_4x4_.fill(max_sample);
    }

    void prepare_zero_rec() {
        rec_8x8_.fill(0);
        rec_4x4_.fill(0);
    }

    void prepare_extreme_rec() {
        constexpr auto max_sample = (1 << bd) - 1;
        rec_8x8_.fill(max_sample);
        rec_4x4_.fill(max_sample);
    }

    void prepare_random_data() {
        std::generate(
            src_8x8_.begin(), src_8x8_.end(), [&]() { return rnd_.random(); });
        std::generate(
            rec_8x8_.begin(), rec_8x8_.end(), [&]() { return rnd_.random(); });

        std::generate(
            src_4x4_.begin(), src_4x4_.end(), [&]() { return rnd_.random(); });
        std::generate(
            rec_4x4_.begin(), rec_4x4_.end(), [&]() { return rnd_.random(); });
    }

    virtual void run_8x8_test(double &score_ref, double &score_simd) = 0;
    virtual void run_4x4_test(double &score_ref, double &score_simd) = 0;

    static void check_data(double score_ref, double score_simd, int32_t index) {
        ASSERT_EQ(score_ref, score_simd)
            << "SSIM score mismatch at test(" << index << ")";
    }

    virtual void run_random_test(const int run_times) {
        for (int iter = 0; iter < run_times;) {
            double score_ref;
            double score_simd;
            prepare_random_data();

            run_8x8_test(score_ref, score_simd);
            check_data(score_ref, score_simd, iter++);
            if (HasFatalFailure())
                return;
            run_4x4_test(score_ref, score_simd);
            check_data(score_ref, score_simd, iter++);
            if (HasFatalFailure())
                return;
        }
    }

    virtual void run_extreme_test() {
        int iter = 0;
        double score_ref;
        double score_simd;

        prepare_zero_src();
        prepare_zero_rec();
        run_8x8_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;
        run_4x4_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;

        prepare_extreme_src();
        prepare_extreme_rec();
        run_8x8_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;
        run_4x4_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;

        prepare_zero_src();
        prepare_extreme_rec();
        run_8x8_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;
        run_4x4_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;

        prepare_extreme_src();
        prepare_zero_rec();
        run_8x8_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;
        run_4x4_test(score_ref, score_simd);
        check_data(score_ref, score_simd, iter++);
        if (HasFatalFailure())
            return;
    }

  protected:
    alignas(16) std::array<Sample, 64> src_8x8_{};
    alignas(16) std::array<Sample, 64> rec_8x8_{};
    alignas(16) std::array<Sample, 16> src_4x4_{};
    alignas(16) std::array<Sample, 16> rec_4x4_{};
    SVTRandom rnd_{bd, false};
};

class SsimLbdTest : public SsimTest<uint8_t, 8> {
  public:
    void run_8x8_test(double &score_ref, double &score_simd) override {
        // setup using c code
        reset_test_env();
        score_ref = svt_ssim_8x8(src_8x8_.data(), 8, rec_8x8_.data(), 8);

        // setup using simd accelerating
        setup_test_env();
        score_simd = svt_ssim_8x8(src_8x8_.data(), 8, rec_8x8_.data(), 8);
    }

    void run_4x4_test(double &score_ref, double &score_simd) override {
        // setup using c code
        reset_test_env();
        score_ref = svt_ssim_4x4(src_4x4_.data(), 4, rec_4x4_.data(), 4);

        // setup using simd accelerating
        setup_test_env();
        score_simd = svt_ssim_4x4(src_4x4_.data(), 4, rec_4x4_.data(), 4);
    }
};

class SsimHbdTest : public SsimTest<uint16_t, 10> {
  public:
    void run_8x8_test(double &score_ref, double &score_simd) override {
        // setup using c code
        reset_test_env();
        score_ref = svt_ssim_8x8_hbd(src_8x8_.data(), 8, rec_8x8_.data(), 8);

        // setup using simd accelerating
        setup_test_env();
        score_simd = svt_ssim_8x8_hbd(src_8x8_.data(), 8, rec_8x8_.data(), 8);
    }

    void run_4x4_test(double &score_ref, double &score_simd) override {
        // setup using c code
        reset_test_env();
        score_ref = svt_ssim_4x4_hbd(src_4x4_.data(), 4, rec_4x4_.data(), 4);

        // setup using simd accelerating
        setup_test_env();
        score_simd = svt_ssim_4x4_hbd(src_4x4_.data(), 4, rec_4x4_.data(), 4);
    }
};

TEST_F(SsimLbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
}
TEST_F(SsimLbdTest, MatchTestWithRandomValue) {
    run_random_test(test_times);
}
TEST_F(SsimHbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
}
TEST_F(SsimHbdTest, MatchTestWithRandomValue) {
    run_random_test(test_times);
}

}  // namespace
