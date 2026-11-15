/*
 * Copyright(c) 2019 Netflix, Inc.
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
#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "resize.h"
#include "utility.h"
#include "unit_test_utility.h"
#include "random.h"
#include "util.h"

#ifdef ARCH_X86_64
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#endif  // ARCH_X86_64

extern "C" void calculate_scaled_size_helper(uint16_t *dim, uint8_t denom);

namespace {
using std::make_tuple;
using svt_av1_test_tool::SVTRandom;

static const int min_test_times = 10;
static const int default_speed_test_times = 1000;
static const uint8_t REF_STUFF = 0xAA;
static const uint8_t TST_STUFF = 0xBB;

/** setup_test_env and reset_test_env are implemented in test/TestEnv.c */
extern "C" void setup_test_env();
extern "C" void reset_test_env();

extern "C" EbErrorType svt_av1_resize_plane_horizontal(
    const uint8_t *const input, int height, int width, int in_stride,
    uint8_t *output, int height2, int width2, int out_stride);
extern "C" EbErrorType svt_av1_highbd_resize_plane_horizontal(
    const uint16_t *const input, int height, int width, int in_stride,
    uint16_t *output, int height2, int width2, int out_stride, int bd);

typedef std::tuple<int, /**< width of source/upscaled picture */
                   int, /**< height of source picture */
                   int> /**< stride of source picture */
    PicSizeParam;

typedef std::tuple<PicSizeParam, int, /**< denominator of scaling: 8~16 */
                   int>               /**< bit depth: 8, 10, 12 */
    ResizeTestParam;

/**
 * @brief Unit test for resize down sampling:
 * - svt_av1_resize_plane
 * - svt_av1_highbd_resize_plane
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same random data and check test output and reference output.
 * Define a template class to handle the common process, and
 * declare sub class to handle different bit depth.
 *
 * Expected result:
 * Output from assemble functions should be the same with output from c.
 *
 * Test coverage:
 * Test cases:
 * input value: Fill with zero, extreme values and random values
 * test mode: common video frame resolution, 2D data blocks with random
 * rectangle size
 *
 */
template <typename Sample>
class ResizePlaneTest : public ::testing::TestWithParam<ResizeTestParam> {
  public:
    ResizePlaneTest() {
        src_width_ = std::get<0>(TEST_GET_PARAM(0));
        src_height_ = std::get<1>(TEST_GET_PARAM(0));
        src_stride_ = std::get<2>(TEST_GET_PARAM(0));
        denom_ = TEST_GET_PARAM(1);
        bd_ = TEST_GET_PARAM(2);
        uint16_t scaled_width = (uint16_t)src_width_;
        calculate_scaled_size_helper(&scaled_width, denom_);
        scaled_width_ = (int)scaled_width;
        uint16_t scaled_height = (uint16_t)src_height_;
        calculate_scaled_size_helper(&scaled_height, denom_);
        scaled_height_ = (int)scaled_height;
        width_only_ = false;
        rnd_ = new SVTRandom(bd_, false);
        setup_test_env();
    }

    virtual ~ResizePlaneTest() {
        delete rnd_;
    }

    virtual void SetUp() {
        ASSERT_LE(src_width_, src_stride_)
            << "picture width must be less equal than stride";
        ASSERT_LE(scaled_width_, src_width_)
            << "width of scaled picture must be less equal than source";
        src_ = (Sample *)(svt_aom_malloc(src_stride_ * src_height_ *
                                         sizeof(*src_)));
        ASSERT_NE(src_, nullptr);

        scaled_ref_ = (Sample *)(svt_aom_malloc(src_stride_ * src_height_ *
                                                sizeof(*scaled_ref_)));
        ASSERT_NE(scaled_ref_, nullptr);
        memset(scaled_ref_,
               REF_STUFF,
               src_stride_ * src_height_ * sizeof(*scaled_ref_));

        scaled_tst_ = (Sample *)(svt_aom_malloc(src_stride_ * src_height_ *
                                                sizeof(*scaled_tst_)));
        ASSERT_NE(scaled_tst_, nullptr);
        memset(scaled_tst_,
               TST_STUFF,
               src_stride_ * src_height_ * sizeof(*scaled_tst_));
    }

    virtual void TearDown() {
        svt_aom_free(src_);
        svt_aom_free(scaled_ref_);
        svt_aom_free(scaled_tst_);
    }

    void prepare_zero_data() {
        memset(src_, 0, src_stride_ * src_height_ * sizeof(*src_));
    }
    void prepare_random_data() {
        for (int i = 0; i < src_stride_ * src_height_; ++i) {
            src_[i] = rnd_->random();
        }
    }

    void prepare_extreme_data() {
        for (int i = 0; i < src_stride_ * src_height_; ++i) {
            src_[i] = (1 << bd_) - 1;
        }
    }
    virtual void run_test(bool width_only = false) = 0;
    virtual void speed_test(int loop, bool width_only = false) = 0;

    void check_data(const int index) {
        const uint16_t value_limit = 1 << bd_;
        Sample ref_guard, tst_guard;
        memset(&ref_guard, REF_STUFF, sizeof(ref_guard));
        memset(&tst_guard, TST_STUFF, sizeof(tst_guard));

        for (int y = 0; y < (width_only_ ? src_height_ : scaled_height_); y++) {
            // check upscaled data
            for (int x = 0; x < scaled_width_; x++) {
                ASSERT_LT(scaled_ref_[y * src_stride_ + x], value_limit);
                ASSERT_LT(scaled_tst_[y * src_stride_ + x], value_limit);
                ASSERT_EQ(scaled_ref_[y * src_stride_ + x],
                          scaled_tst_[y * src_stride_ + x])
                    << "scaled pixel mismatch at test(" << index
                    << ") row: " << y << ", col: " << x << ", "
                    << scaled_ref_[y * src_stride_ + x] << "<-->"
                    << scaled_tst_[y * src_stride_ + x];
            }
            // check padding data
            for (int x = scaled_width_; x < src_stride_; x++) {
                EXPECT_EQ(scaled_ref_[y * src_stride_ + x], ref_guard);
                EXPECT_EQ(scaled_tst_[y * src_stride_ + x], tst_guard);
            }
        }
    }

    virtual void run_zero_test(bool width_only = false) {
        const int iters = min_test_times;
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_zero_data();
            run_test(width_only);
            check_data(iter);
        }
    }

    virtual void run_random_test(const int run_times, bool width_only = false) {
        const int iters = AOMMAX(run_times, min_test_times);
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_random_data();
            run_test(width_only);
            check_data(iter);
        }
    }

    virtual void run_extreme_test(bool width_only = false) {
        const int iters = min_test_times;
        for (int iter = 0; iter < iters && !HasFatalFailure(); ++iter) {
            prepare_extreme_data();
            run_test(width_only);
            check_data(iter);
        }
    }

    virtual void run_speed_test(bool width_only = false) {
        prepare_random_data();
        speed_test(default_speed_test_times, width_only);
    }

  protected:
    int src_width_;
    int src_height_;
    int src_stride_;
    uint8_t denom_;
    int bd_;
    int scaled_width_;
    int scaled_height_;
    bool width_only_;

    Sample *src_;
    Sample *scaled_ref_;
    Sample *scaled_tst_;
    SVTRandom *rnd_;
};

class ResizePlaneLbdTest : public ResizePlaneTest<uint8_t> {
  public:
    void run_test(bool width_only) override {
        // setup using c code
        reset_test_env();
        if (width_only) {
            svt_av1_resize_plane_horizontal(src_,
                                            src_height_,
                                            src_width_,
                                            src_stride_,
                                            scaled_ref_,
                                            src_height_,
                                            scaled_width_,
                                            src_stride_);
        } else {
            svt_av1_resize_plane(src_,
                                 src_height_,
                                 src_width_,
                                 src_stride_,
                                 scaled_ref_,
                                 scaled_height_,
                                 scaled_width_,
                                 src_stride_);
        }
        // setup using simd accelerating
        setup_test_env();
        if (width_only) {
            svt_av1_resize_plane_horizontal(src_,
                                            src_height_,
                                            src_width_,
                                            src_stride_,
                                            scaled_tst_,
                                            src_height_,
                                            scaled_width_,
                                            src_stride_);
        } else {
            svt_av1_resize_plane(src_,
                                 src_height_,
                                 src_width_,
                                 src_stride_,
                                 scaled_tst_,
                                 scaled_height_,
                                 scaled_width_,
                                 src_stride_);
        }
    }
    void speed_test(int loop, bool width_only) override {
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;

        // setup using c code
        reset_test_env();
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (int i = 0; i < loop; i++) {
            if (width_only) {
                svt_av1_resize_plane_horizontal(src_,
                                                src_height_,
                                                src_width_,
                                                src_stride_,
                                                scaled_ref_,
                                                src_height_,
                                                scaled_width_,
                                                src_stride_);
            } else {
                svt_av1_resize_plane(src_,
                                     src_height_,
                                     src_width_,
                                     src_stride_,
                                     scaled_ref_,
                                     scaled_height_,
                                     scaled_width_,
                                     src_stride_);
            }
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_c = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);

        // setup using simd accelerating
        setup_test_env();
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (int i = 0; i < loop; i++) {
            if (width_only) {
                svt_av1_resize_plane_horizontal(src_,
                                                src_height_,
                                                src_width_,
                                                src_stride_,
                                                scaled_tst_,
                                                src_height_,
                                                scaled_width_,
                                                src_stride_);
            } else {
                svt_av1_resize_plane(src_,
                                     src_height_,
                                     src_width_,
                                     src_stride_,
                                     scaled_tst_,
                                     scaled_height_,
                                     scaled_width_,
                                     src_stride_);
            }
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_o = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);
        printf("Resize %ux%u --> %ux%u\n",
               src_width_,
               src_height_,
               scaled_width_,
               width_only ? src_height_ : scaled_height_);
        printf("Average Nanoseconds per Function Call\n");
        printf("    svt_av1_resize_plane_c   : %6.2f\n",
               1000000 * time_c / loop);
        printf(
            "    svt_av1_resize_plane_avx2 : %6.2f   (Comparison: "
            "%5.2fx)\n",
            1000000 * time_o / loop,
            time_c / time_o);
    }
};

TEST_P(ResizePlaneLbdTest, MatchTestWithZeroValue) {
    run_zero_test();
    run_zero_test(true);
}
TEST_P(ResizePlaneLbdTest, MatchTestWithRandomValue) {
    run_random_test(10);
    run_random_test(10, true);
}
TEST_P(ResizePlaneLbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
    run_extreme_test(true);
}
TEST_P(ResizePlaneLbdTest, DISABLED_SpeedTestWithRandomValue) {
    run_speed_test();
    run_speed_test(true);
}

static PicSizeParam pic_size_vector[] = {
    make_tuple(1280, 720, 1520),
    make_tuple(1920, 1080, 2240),
    make_tuple(3840, 2160, 4480),
};

INSTANTIATE_TEST_SUITE_P(
    Resize, ResizePlaneLbdTest,
    ::testing::Combine(::testing::ValuesIn(pic_size_vector),
                       ::testing::Range(8, 17), ::testing::Values(8)));

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
class ResizePlaneHbdTest : public ResizePlaneTest<uint16_t> {
  protected:
    void run_test(bool width_only) override {
        // setup using c code
        reset_test_env();
        if (width_only) {
            svt_av1_highbd_resize_plane_horizontal(src_,
                                                   src_height_,
                                                   src_width_,
                                                   src_stride_,
                                                   scaled_ref_,
                                                   src_height_,
                                                   scaled_width_,
                                                   src_stride_,
                                                   bd_);
        } else {
            svt_av1_highbd_resize_plane(src_,
                                        src_height_,
                                        src_width_,
                                        src_stride_,
                                        scaled_ref_,
                                        scaled_height_,
                                        scaled_width_,
                                        src_stride_,
                                        bd_);
        }
        // setup using simd accelerating
        setup_test_env();
        if (width_only) {
            svt_av1_highbd_resize_plane_horizontal(src_,
                                                   src_height_,
                                                   src_width_,
                                                   src_stride_,
                                                   scaled_tst_,
                                                   src_height_,
                                                   scaled_width_,
                                                   src_stride_,
                                                   bd_);
        } else {
            svt_av1_highbd_resize_plane(src_,
                                        src_height_,
                                        src_width_,
                                        src_stride_,
                                        scaled_tst_,
                                        scaled_height_,
                                        scaled_width_,
                                        src_stride_,
                                        bd_);
        }
    }
    void speed_test(int loop, bool width_only) override {
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;

        // setup using c code
        reset_test_env();
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (int i = 0; i < loop; i++) {
            if (width_only) {
                svt_av1_highbd_resize_plane_horizontal(src_,
                                                       src_height_,
                                                       src_width_,
                                                       src_stride_,
                                                       scaled_ref_,
                                                       src_height_,
                                                       scaled_width_,
                                                       src_stride_,
                                                       bd_);
            } else {
                svt_av1_highbd_resize_plane(src_,
                                            src_height_,
                                            src_width_,
                                            src_stride_,
                                            scaled_ref_,
                                            scaled_height_,
                                            scaled_width_,
                                            src_stride_,
                                            bd_);
            }
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_c = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);

        // setup using simd accelerating
        setup_test_env();
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (int i = 0; i < loop; i++) {
            if (width_only) {
                svt_av1_highbd_resize_plane_horizontal(src_,
                                                       src_height_,
                                                       src_width_,
                                                       src_stride_,
                                                       scaled_tst_,
                                                       src_height_,
                                                       scaled_width_,
                                                       src_stride_,
                                                       bd_);
            } else {
                svt_av1_highbd_resize_plane(src_,
                                            src_height_,
                                            src_width_,
                                            src_stride_,
                                            scaled_tst_,
                                            scaled_height_,
                                            scaled_width_,
                                            src_stride_,
                                            bd_);
            }
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_o = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);
        printf("Average Nanoseconds per Function Call\n");
        printf("    svt_av1_highbd_resize_plane_c   : %6.2f\n",
               1000000 * time_c / loop);
        printf(
            "    svt_av1_highbd_resize_plane_avx2 : %6.2f   (Comparison: "
            "%5.2fx)\n",
            1000000 * time_o / loop,
            time_c / time_o);
    }
};

TEST_P(ResizePlaneHbdTest, MatchTestWithZeroValue) {
    run_zero_test();
    run_zero_test(true);
}
TEST_P(ResizePlaneHbdTest, MatchTestWithRandomValue) {
    run_random_test(10);
    run_random_test(10, true);
}
TEST_P(ResizePlaneHbdTest, MatchTestWithExtremeValue) {
    run_extreme_test();
    run_extreme_test(true);
}
TEST_P(ResizePlaneHbdTest, DISABLED_SpeedTestWithRandomValue) {
    run_speed_test();
    run_speed_test(true);
}

INSTANTIATE_TEST_SUITE_P(
    Resize, ResizePlaneHbdTest,
    ::testing::Combine(::testing::ValuesIn(pic_size_vector),
                       ::testing::Range(8, 16), ::testing::Values(10, 12)));
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

/**
 * @brief Unit test for svt_av1_down2_symeven: compares a SIMD variant
 * against the plain C reference for a range of lengths chosen to exercise
 * all four branches of the C reference: short-input fallback, initial
 * (left-clamped), middle (unclamped), and end (right-clamped) parts.
 */
typedef void (*Down2SymevenFunc)(const uint8_t *const input, int length,
                                 uint8_t *output);

// Compares the requested SIMD variant directly against the C reference,
// rather than through the reset_test_env()/setup_test_env() RTCD toggle:
// this kernel is only ever dispatched to a NEON or an AVX2 implementation,
// and the two variants support different minimum input lengths, so picking
// "the fastest available" per architecture would run the small lengths
// below against whichever variant happens to be built for the host.
class Down2SymevenTest
    : public ::testing::TestWithParam<std::tuple<int, Down2SymevenFunc>> {
  public:
    Down2SymevenTest()
        : length_(std::get<0>(GetParam())),
          test_func_(std::get<1>(GetParam())),
          rnd_(0, 255) {
    }

    void SetUp() override {
        input_ = (uint8_t *)svt_aom_memalign(32, length_);
        ASSERT_NE(input_, nullptr);
        // output has (length_+1)/2 valid samples; allocate length_ and verify
        // the untouched tail keeps its sentinel value in run_case() below, to
        // catch a kernel writing past the true output size.
        ref_output_ = (uint8_t *)svt_aom_memalign(32, length_);
        ASSERT_NE(ref_output_, nullptr);
        tst_output_ = (uint8_t *)svt_aom_memalign(32, length_);
        ASSERT_NE(tst_output_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(input_);
        svt_aom_free(ref_output_);
        svt_aom_free(tst_output_);
    }

  protected:
    void run_case() {
        const int out_len = (length_ + 1) / 2;
        memset(ref_output_, 0xAA, length_);
        memset(tst_output_, 0xBB, length_);

        svt_av1_down2_symeven_c(input_, length_, ref_output_);
        test_func_(input_, length_, tst_output_);

        for (int i = 0; i < out_len; i++) {
            ASSERT_EQ(ref_output_[i], tst_output_[i])
                << "mismatch at output index " << i << " for length "
                << length_;
        }
        for (int i = out_len; i < length_; i++) {
            ASSERT_EQ(ref_output_[i], 0xAA)
                << "overrun past output index " << out_len << " for length "
                << length_;
            ASSERT_EQ(tst_output_[i], 0xBB)
                << "overrun past output index " << out_len << " for length "
                << length_;
        }
    }

    int length_;
    Down2SymevenFunc test_func_;
    SVTRandom rnd_;
    uint8_t *input_;
    uint8_t *ref_output_;
    uint8_t *tst_output_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Down2SymevenTest);

TEST_P(Down2SymevenTest, MatchTestWithRandomValue) {
    for (int t = 0; t < min_test_times; t++) {
        for (int i = 0; i < length_; i++) {
            input_[i] = (uint8_t)rnd_.random();
        }
        run_case();
    }
}

TEST_P(Down2SymevenTest, MatchTestWithZeroValue) {
    memset(input_, 0, length_);
    run_case();
}

TEST_P(Down2SymevenTest, MatchTestWithMaxValue) {
    memset(input_, 255, length_);
    run_case();
}

TEST_P(Down2SymevenTest, MatchTestWithAlternatingValue) {
    for (int i = 0; i < length_; i++) {
        input_[i] = (i & 1) ? 255 : 0;
    }
    run_case();
}

#ifdef ARCH_AARCH64
// The 8-wide Neon batch computes each output as two int16 partial sums
// (filter[0]*p0 + filter[2]*p2, and filter[1]*p1 + filter[3]*p3) that are
// combined with a saturating int16 add. Each partial sum's true worst case,
// reachable because the 8 input bytes behind p0..p3 are independently
// controllable, is designed to fit int16, but grouping the taps
// differently would not: the MatchTestWithRandomValue/MaxValue/
// AlternatingValue cases above don't happen to hit that worst case (their
// per-tap pair values are correlated - e.g. all-max gives every pJ=510,
// not the independent extremes below), so this test targets it directly.
TEST(Down2SymevenAdversarialTest, MatchTestWithAdversarialTapPattern) {
    const int length = 128;
    uint8_t *input = (uint8_t *)svt_aom_memalign(32, length);
    ASSERT_NE(input, nullptr);
    uint8_t *ref_output = (uint8_t *)svt_aom_memalign(32, length);
    ASSERT_NE(ref_output, nullptr);
    uint8_t *tst_output = (uint8_t *)svt_aom_memalign(32, length);
    ASSERT_NE(tst_output, nullptr);

    // For center c, p0=input[c]+input[c+1], p1=input[c-1]+input[c+2],
    // p2=input[c-2]+input[c+3], p3=input[c-3]+input[c+4]. Setting
    // input[c-3..c+4] = {0,0,255,255,255,255,0,0} makes p0=p1=510, p2=p3=0
    // - filter[0]*p0+filter[1]*p1 = 56*510+12*510 = 34680, which overflows
    // int16 if taps 0 and 1 (rather than 0 and 2) were grouped together.
    // The inverse pattern similarly maximizes the negative-coefficient
    // taps (p2, p3) while zeroing p0, p1. Only even centers are ever
    // evaluated, so the window must start at an odd offset (c-3, with c
    // even) - starting at offset 21 lands on center 24, and 61 on 64.
    memset(input, 128, length);
    const uint8_t max_window[8] = {0, 0, 255, 255, 255, 255, 0, 0};
    const uint8_t min_window[8] = {255, 255, 0, 0, 0, 0, 255, 255};
    memcpy(input + 21, max_window, sizeof(max_window));
    memcpy(input + 61, min_window, sizeof(min_window));

    svt_av1_down2_symeven_c(input, length, ref_output);
    svt_av1_down2_symeven_neon(input, length, tst_output);

    const int out_len = (length + 1) / 2;
    for (int i = 0; i < out_len; i++) {
        ASSERT_EQ(ref_output[i], tst_output[i])
            << "mismatch at output index " << i;
    }

    svt_aom_free(input);
    svt_aom_free(ref_output);
    svt_aom_free(tst_output);
}

INSTANTIATE_TEST_SUITE_P(
    NEON, Down2SymevenTest,
    ::testing::Combine(::testing::Values(1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16,
                                         18, 20, 22, 23, 24, 25, 30, 40, 64,
                                         128, 255, 256, 257, 511, 512, 640,
                                         1024, 1920, 3840),
                       ::testing::Values(svt_av1_down2_symeven_neon)));
#endif  // ARCH_AARCH64

#ifdef ARCH_X86_64
// svt_av1_down2_symeven_avx2 requires an even length (it asserts this) of
// at least 38, below which down2_symeven_w16_init_part_avx2's unconditional
// input+36 read would be out of bounds; shorter inputs fall back to the C
// reference. Separately, down2_symeven_w16_mid_part_avx2's last processed
// chunk reads past its nominal consumption, which overreads when
// (length - 32 - mid) is 3 or 4 - recurring at every even length = 68 +
// 32*k for k = 0, 1, 2, .... The list below covers every even length under
// 38 (exercising the short-input fallback and the SIMD branches at their
// boundary), the 37/38 floor, the periodic failure point (68/100/132/164)
// with its even neighbors, and some realistic plane widths.
INSTANTIATE_TEST_SUITE_P(
    AVX2, Down2SymevenTest,
    ::testing::Combine(::testing::Values(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                                         24, 26, 28, 30, 32, 34, 36, 38, 40, 64,
                                         66, 68, 70, 98, 100, 102, 130, 132,
                                         134, 162, 164, 166, 256, 512, 640,
                                         1024, 1920, 3840),
                       ::testing::Values(svt_av1_down2_symeven_avx2)));

// The sentinel-based Down2SymevenTest above only checks output correctness,
// so it can't catch the periodic bug fixed above on its own: the last 2 of
// the bytes down2_symeven_w16_mid_part_avx2's last chunk reads past its
// nominal window are genuinely unused (the rest feed real filter taps), but
// whether the overread is caught depends on whether AddressSanitizer's
// redzone happens to sit there - which is heap-layout dependent for an
// overread this small. Make the detection deterministic instead: place the
// input right against a guard page with its access removed, so any byte
// read past it is a hard fault no matter how the allocator laid things out.
// Owns a buffer placed directly against a page whose access is removed, so
// any read or write past `usable_size` bytes past `data()` is a hard fault.
// Construction never fails loudly: if the platform calls themselves fail,
// `data()` returns nullptr rather than a usable-looking but unprotected
// pointer, so callers that ASSERT_NE(data(), nullptr) catch it instead of
// silently losing the guarantee.
class GuardedInputBuffer {
  public:
    explicit GuardedInputBuffer(size_t usable_size) {
#ifdef _WIN32
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        const size_t page_size = sys_info.dwPageSize;
#else
        const long page_size_result = sysconf(_SC_PAGESIZE);
        if (page_size_result <= 0)
            return;
        const size_t page_size = (size_t)page_size_result;
#endif
        const size_t total_pages =
            (usable_size + page_size - 1) / page_size + 1;
        total_size_ = total_pages * page_size;
#ifdef _WIN32
        base_ = (uint8_t *)VirtualAlloc(
            nullptr, total_size_, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void *mapped = mmap(nullptr,
                            total_size_,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS,
                            -1,
                            0);
        base_ = (mapped == MAP_FAILED) ? nullptr : (uint8_t *)mapped;
#endif
        if (base_ == nullptr) {
            total_size_ = 0;
            return;
        }
        uint8_t *guard_page = base_ + total_size_ - page_size;
#ifdef _WIN32
        DWORD old_protect;
        const bool protected_ok =
            VirtualProtect(
                guard_page, page_size, PAGE_NOACCESS, &old_protect) != 0;
#else
        const bool protected_ok =
            mprotect(guard_page, page_size, PROT_NONE) == 0;
#endif
        if (protected_ok)
            ptr_ = guard_page - usable_size;
    }

    GuardedInputBuffer(const GuardedInputBuffer &) = delete;
    GuardedInputBuffer &operator=(const GuardedInputBuffer &) = delete;

    ~GuardedInputBuffer() {
        if (base_ == nullptr)
            return;
#ifdef _WIN32
        VirtualFree(base_, 0, MEM_RELEASE);
#else
        munmap(base_, total_size_);
#endif
    }

    // nullptr if setup failed (allocation or the protection call itself);
    // never a pointer backed by an unprotected guard page.
    uint8_t *data() const {
        return ptr_;
    }

  private:
    size_t total_size_ = 0;
    uint8_t *base_ = nullptr;
    uint8_t *ptr_ = nullptr;
};

class Down2SymevenAvx2GuardPageTest : public ::testing::TestWithParam<int> {
  public:
    Down2SymevenAvx2GuardPageTest()
        : length_(GetParam()), out_len_((length_ + 1) / 2), rnd_(0, 255) {
    }

    void SetUp() override {
        ref_output_ = (uint8_t *)svt_aom_memalign(32, out_len_);
        ASSERT_NE(ref_output_, nullptr);
        tst_output_ = (uint8_t *)svt_aom_memalign(32, out_len_);
        ASSERT_NE(tst_output_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(ref_output_);
        svt_aom_free(tst_output_);
    }

  protected:
    int length_;
    int out_len_;
    SVTRandom rnd_;
    uint8_t *ref_output_ = nullptr;
    uint8_t *tst_output_ = nullptr;
};

TEST_P(Down2SymevenAvx2GuardPageTest, NoOverreadPastInputEnd) {
    GuardedInputBuffer guarded(length_);
    uint8_t *input = guarded.data();
    ASSERT_NE(input, nullptr);
    for (int i = 0; i < length_; i++) {
        input[i] = (uint8_t)rnd_.random();
    }

    svt_av1_down2_symeven_c(input, length_, ref_output_);
    // Faults here (not at ASSERT_EQ below) if the kernel reads past the
    // guard page.
    svt_av1_down2_symeven_avx2(input, length_, tst_output_);

    for (int i = 0; i < out_len_; i++) {
        ASSERT_EQ(ref_output_[i], tst_output_[i])
            << "mismatch at output index " << i << " for length " << length_;
    }
}

INSTANTIATE_TEST_SUITE_P(AVX2, Down2SymevenAvx2GuardPageTest,
                         ::testing::Values(2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                                           22, 24, 26, 28, 30, 32, 34, 36, 38,
                                           40, 64, 66, 68, 70, 98, 100, 102,
                                           130, 132, 134, 162, 164, 166, 256,
                                           1920));

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
class HighbdDown2SymevenAvx2GuardPageTest
    : public ::testing::TestWithParam<std::tuple<int, int>> {
  public:
    HighbdDown2SymevenAvx2GuardPageTest()
        : length_(std::get<0>(GetParam())),
          bd_(std::get<1>(GetParam())),
          out_len_((length_ + 1) / 2),
          rnd_(0, (1 << bd_) - 1) {
    }

    void SetUp() override {
        ref_output_ =
            (uint16_t *)svt_aom_memalign(32, out_len_ * sizeof(uint16_t));
        ASSERT_NE(ref_output_, nullptr);
        tst_output_ =
            (uint16_t *)svt_aom_memalign(32, out_len_ * sizeof(uint16_t));
        ASSERT_NE(tst_output_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(ref_output_);
        svt_aom_free(tst_output_);
    }

  protected:
    int length_;
    int bd_;
    int out_len_;
    SVTRandom rnd_;
    uint16_t *ref_output_ = nullptr;
    uint16_t *tst_output_ = nullptr;
};

TEST_P(HighbdDown2SymevenAvx2GuardPageTest, NoOverreadPastInputEnd) {
    GuardedInputBuffer guarded(length_ * sizeof(uint16_t));
    uint16_t *input = (uint16_t *)guarded.data();
    ASSERT_NE(input, nullptr);
    for (int i = 0; i < length_; i++) {
        input[i] = (uint16_t)rnd_.random();
    }

    svt_av1_highbd_down2_symeven_c(input, length_, ref_output_, bd_);
    // Faults here (not at ASSERT_EQ below) if the kernel reads past the
    // guard page.
    svt_av1_highbd_down2_symeven_avx2(input, length_, tst_output_, bd_);

    for (int i = 0; i < out_len_; i++) {
        ASSERT_EQ(ref_output_[i], tst_output_[i])
            << "mismatch at output index " << i << " for length " << length_
            << " bd " << bd_;
    }
}

// svt_av1_highbd_down2_symeven_avx2 asserts an even length and uses SIMD
// only for lengths >=22. highbd_down2_symeven_w8_init_part_avx2's
// unconditional input+20 read overreads shorter even inputs, which fall
// back to the C reference. Separately, highbd_down2_symeven_w8_mid_part_avx2's
// last processed chunk reads past its nominal consumption, overreading at
// every even length = 36 + 16*k for k = 0, 1, 2, .... The list below covers
// every even length under 22 (short-input fallback), the 20/22 floor, the
// periodic failure point (36/52/68/84/100/116/132) with its even neighbors,
// and some realistic plane widths, at both supported bit depths.
INSTANTIATE_TEST_SUITE_P(
    AVX2, HighbdDown2SymevenAvx2GuardPageTest,
    ::testing::Combine(::testing::Values(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                                         24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
                                         50, 52, 54, 64, 66, 68, 70, 82, 84, 86,
                                         98, 100, 102, 114, 116, 118, 130, 132,
                                         134, 256, 1920),
                       ::testing::Values(10, 12)));
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH
#endif  // ARCH_X86_64

typedef void (*InterpolateCoreFunc)(const uint8_t *const input, int in_length,
                                    uint8_t *output, int out_length,
                                    const int16_t *interp_filters);

// The 5 filter tables svt_av1_interpolate_core can be called with (chosen in
// production by choose_interp_filter() based on the in/out length ratio, but
// the kernel itself just uses whichever table the caller passes) - exercised
// directly here so coverage doesn't depend on which in/out length ratios in
// the parameter list happen to select which table.
static const int16_t *const kInterpFilterTables[5] = {
    &svt_aom_av1_filteredinterp_filters500[0][0],
    &svt_aom_av1_filteredinterp_filters625[0][0],
    &svt_aom_av1_filteredinterp_filters750[0][0],
    &svt_aom_av1_filteredinterp_filters875[0][0],
    &svt_av1_resize_filter_normative[0][0],
};

class InterpolateCoreTest
    : public ::testing::TestWithParam<
          std::tuple<int, int, int, InterpolateCoreFunc>> {
  public:
    InterpolateCoreTest()
        : in_length_(std::get<0>(GetParam())),
          out_length_(std::get<1>(GetParam())),
          filter_(kInterpFilterTables[std::get<2>(GetParam())]),
          test_func_(std::get<3>(GetParam())),
          rnd_(0, 255),
          input_(nullptr),
          ref_output_(nullptr),
          tst_output_(nullptr) {
    }

    // The output buffers are over-allocated by kOverrunPad bytes and the pad
    // is sentinel-filled in run_case() below, to catch the 8-wide NEON store
    // (or any bug in the middle-loop bound) writing past out_length_ even
    // when the visible [0,out_length_) bytes still happen to match.
    static const int kOverrunPad = 8;

    void SetUp() override {
        input_ = (uint8_t *)svt_aom_memalign(32, in_length_);
        ASSERT_NE(input_, nullptr);
        ref_output_ =
            (uint8_t *)svt_aom_memalign(32, out_length_ + kOverrunPad);
        ASSERT_NE(ref_output_, nullptr);
        tst_output_ =
            (uint8_t *)svt_aom_memalign(32, out_length_ + kOverrunPad);
        ASSERT_NE(tst_output_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(input_);
        svt_aom_free(ref_output_);
        svt_aom_free(tst_output_);
    }

  protected:
    void run_case() {
        memset(ref_output_ + out_length_, 0xAA, kOverrunPad);
        memset(tst_output_ + out_length_, 0xBB, kOverrunPad);

        svt_av1_interpolate_core_c(
            input_, in_length_, ref_output_, out_length_, filter_);
        test_func_(input_, in_length_, tst_output_, out_length_, filter_);
        for (int i = 0; i < out_length_; i++) {
            ASSERT_EQ(ref_output_[i], tst_output_[i])
                << "mismatch at output index " << i << " for in_length "
                << in_length_ << " out_length " << out_length_;
        }
        for (int i = 0; i < kOverrunPad; i++) {
            ASSERT_EQ(ref_output_[out_length_ + i], 0xAA)
                << "overrun past out_length " << out_length_;
            ASSERT_EQ(tst_output_[out_length_ + i], 0xBB)
                << "overrun past out_length " << out_length_;
        }
    }

    int in_length_;
    int out_length_;
    const int16_t *filter_;
    InterpolateCoreFunc test_func_;
    SVTRandom rnd_;
    uint8_t *input_;
    uint8_t *ref_output_;
    uint8_t *tst_output_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InterpolateCoreTest);

TEST_P(InterpolateCoreTest, MatchTestWithRandomValue) {
    for (int t = 0; t < min_test_times; t++) {
        for (int i = 0; i < in_length_; i++) {
            input_[i] = (uint8_t)rnd_.random();
        }
        run_case();
    }
}

TEST_P(InterpolateCoreTest, MatchTestWithZeroValue) {
    memset(input_, 0, in_length_);
    run_case();
}

TEST_P(InterpolateCoreTest, MatchTestWithMaxValue) {
    memset(input_, 255, in_length_);
    run_case();
}

TEST_P(InterpolateCoreTest, MatchTestWithAlternatingValue) {
    for (int i = 0; i < in_length_; i++) {
        input_[i] = (i & 1) ? 255 : 0;
    }
    run_case();
}

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, InterpolateCoreTest,
    ::testing::Combine(::testing::Values(1, 2, 5, 7, 9, 64, 640, 1920, 3840),
                       ::testing::Values(1, 2, 5, 7, 9, 64, 640, 1920, 3840),
                       ::testing::Values(0, 1, 2, 3, 4),
                       ::testing::Values(svt_av1_interpolate_core_neon)));
#endif  // ARCH_AARCH64
}  // namespace
