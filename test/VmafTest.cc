/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and the
 * Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
 * not distributed with this source code in the LICENSE file, you can obtain it
 * at https://www.aomedia.org/license/software-license. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include <stdint.h>
#include <initializer_list>
#include <tuple>
#include <utility>

#include "aom_dsp_rtcd.h"
#include "gtest/gtest.h"
#include "random.h"
#include "svt_malloc.h"
#include "util.h"

using svt_av1_test_tool::SVTRandom;

namespace {
#if OPT_TUNE_VMAF

static const int kSvtAv1MaxWidth = 16384;
static const int kSvtAv1MaxHeight = 8704;

// Cover minimal 8x8 tiling, a typical 1080p frame, and max legal width/height
// cases.
static const std::pair<int, int> kVmafSizes[] = {
    {8, 8},
    {1920, 1080},
    {kSvtAv1MaxWidth, 8},
    {8, kSvtAv1MaxHeight},
};
static const int kVmafWidths[] = {8, 1920, kSvtAv1MaxWidth};

using VmafAvgMadFunc = uint32_t (*)(const uint8_t *src, int width, int height,
                                    int stride);

using VmafAvgMadParam = std::tuple<std::pair<int, int>, VmafAvgMadFunc>;

class VmafAvgMadTest : public ::testing::TestWithParam<VmafAvgMadParam> {
  public:
    VmafAvgMadTest() : size_(TEST_GET_PARAM(0)), test_func_(TEST_GET_PARAM(1)) {
    }

    ~VmafAvgMadTest() override = default;

  protected:
    static const int kIterations = 10;

    void SetUp() override {
        const int width = size_.first;
        const int height = size_.second;
        src_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width * height));
        ASSERT_NE(src_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
        src_ = nullptr;
    }

    std::pair<int, int> size_;
    VmafAvgMadFunc test_func_;
    uint8_t *src_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int i = 0; i < kIterations; ++i) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < stride; ++x) {
                    src_[y * stride + x] = rnd.Rand8();
                }
            }

            const uint32_t res_ref =
                svt_vmaf_compute_avg_mad_c(src_, width, height, stride);
            const uint32_t res_tst = test_func_(src_, width, height, stride);
            ASSERT_EQ(res_ref, res_tst) << "iteration " << i;
        }
    }

    void run_max_input_test() {
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int i = 0; i < height * stride; ++i) {
            src_[i] = 255;
        }

        const uint32_t res_ref =
            svt_vmaf_compute_avg_mad_c(src_, width, height, stride);
        const uint32_t res_tst = test_func_(src_, width, height, stride);
        ASSERT_EQ(res_ref, res_tst);
    }

    void run_max_activity_test() {
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < stride; ++x) {
                src_[y * stride + x] = (x & 1) ? 255 : 0;
            }
        }

        const uint32_t res_ref =
            svt_vmaf_compute_avg_mad_c(src_, width, height, stride);
        const uint32_t res_tst = test_func_(src_, width, height, stride);
        ASSERT_EQ(res_ref, res_tst);
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafAvgMadTest);

TEST_P(VmafAvgMadTest, MatchTest) {
    run_match_test();
}

TEST_P(VmafAvgMadTest, MaxInputTest) {
    run_max_input_test();
}

TEST_P(VmafAvgMadTest, MaxActivityTest) {
    run_max_activity_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafAvgMadTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(&svt_vmaf_compute_avg_mad_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafAvgMadTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(&svt_vmaf_compute_avg_mad_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VmafAvgMadTest,
    ::testing::Combine(
        ::testing::ValuesIn(kVmafSizes),
        ::testing::Values(&svt_vmaf_compute_avg_mad_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_NEON_I8MM
INSTANTIATE_TEST_SUITE_P(
    NEON_I8MM, VmafAvgMadTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(&svt_vmaf_compute_avg_mad_neon_i8mm)));
#endif  // HAVE_NEON_I8MM
#endif  // ARCH_AARCH64

using VmafUnsharpRowFunc = void (*)(const uint8_t *src, const uint8_t *blur,
                                    uint8_t *dst, int width, int amount,
                                    int32_t max_delta);

using VmafUnsharpRowParam = std::tuple<int, int, int, VmafUnsharpRowFunc>;

class VmafUnsharpRowTest
    : public ::testing::TestWithParam<VmafUnsharpRowParam> {
  public:
    VmafUnsharpRowTest()
        : width_(TEST_GET_PARAM(0)),
          amount_(TEST_GET_PARAM(1)),
          max_delta_(TEST_GET_PARAM(2)),
          test_func_(TEST_GET_PARAM(3)) {
    }

    ~VmafUnsharpRowTest() override = default;

  protected:
    static const int kIterations = 100;

    void SetUp() override {
        src_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*src_)));
        blur_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*blur_)));
        dst_ref_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*dst_ref_)));
        dst_tst_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*dst_tst_)));

        ASSERT_NE(src_, nullptr);
        ASSERT_NE(blur_, nullptr);
        ASSERT_NE(dst_ref_, nullptr);
        ASSERT_NE(dst_tst_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
        svt_aom_free(blur_);
        svt_aom_free(dst_ref_);
        svt_aom_free(dst_tst_);
    }

    int width_;
    int amount_;
    int max_delta_;
    VmafUnsharpRowFunc test_func_;
    uint8_t *src_ = nullptr;
    uint8_t *blur_ = nullptr;
    uint8_t *dst_ref_ = nullptr;
    uint8_t *dst_tst_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);

        for (int i = 0; i < kIterations; ++i) {
            for (int x = 0; x < width_; ++x) {
                src_[x] = rnd.Rand8();
                blur_[x] = rnd.Rand8();
            }

            svt_vmaf_apply_unsharp_row_c(
                src_, blur_, dst_ref_, width_, amount_, max_delta_);
            test_func_(src_, blur_, dst_tst_, width_, amount_, max_delta_);
            for (int x = 0; x < width_; ++x) {
                ASSERT_EQ(dst_ref_[x], dst_tst_[x])
                    << "x " << x << ", iteration " << i;
            }
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafUnsharpRowTest);

TEST_P(VmafUnsharpRowTest, MatchTest) {
    run_match_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafUnsharpRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(19660, 32767),
                       ::testing::Values(8, 12),
                       ::testing::Values(&svt_vmaf_apply_unsharp_row_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafUnsharpRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(19660, 32767),
                       ::testing::Values(8, 12),
                       ::testing::Values(&svt_vmaf_apply_unsharp_row_neon)));
#endif  // ARCH_AARCH64

using VmafHpassRowFunc = void (*)(const uint8_t *src_row, int width,
                                  int16_t *h_row);

using VmafHpassRowParam = std::tuple<int, VmafHpassRowFunc>;

class VmafHpassRowTest : public ::testing::TestWithParam<VmafHpassRowParam> {
  public:
    VmafHpassRowTest()
        : width_(TEST_GET_PARAM(0)), test_func_(TEST_GET_PARAM(1)) {
    }

    ~VmafHpassRowTest() override = default;

  protected:
    static const int kIterations = 100;
    static const int kStepsX = 2;

    void SetUp() override {
        const size_t src_size = (size_t)width_ * sizeof(*src_);
        const size_t out_size =
            ((size_t)width_ + 2 * kStepsX) * sizeof(*h_ref_);

        src_ = static_cast<uint8_t *>(svt_aom_memalign(16, src_size));
        h_ref_ = static_cast<int16_t *>(svt_aom_memalign(16, out_size));
        h_tst_ = static_cast<int16_t *>(svt_aom_memalign(16, out_size));
        ASSERT_NE(src_, nullptr);
        ASSERT_NE(h_ref_, nullptr);
        ASSERT_NE(h_tst_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
        svt_aom_free(h_ref_);
        svt_aom_free(h_tst_);
    }

    int width_;
    VmafHpassRowFunc test_func_;
    uint8_t *src_ = nullptr;
    int16_t *h_ref_ = nullptr;
    int16_t *h_tst_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int out_count = width_ + 2 * kStepsX;

        for (int i = 0; i < kIterations; ++i) {
            for (int x = 0; x < width_; ++x) {
                src_[x] = rnd.Rand8();
            }

            svt_vmaf_hpass_row_c(src_, width_, h_ref_);
            test_func_(src_, width_, h_tst_);

            for (int x = 0; x < out_count; ++x) {
                ASSERT_EQ(h_ref_[x], h_tst_[x])
                    << "h_row mismatch, x " << x << ", width " << width_
                    << ", iteration " << i;
            }
        }
    }

    void run_max_input_test() {
        const int out_count = width_ + 2 * kStepsX;

        for (int x = 0; x < width_; ++x) {
            src_[x] = 255;
        }

        svt_vmaf_hpass_row_c(src_, width_, h_ref_);
        test_func_(src_, width_, h_tst_);

        for (int x = 0; x < out_count; ++x) {
            ASSERT_EQ(h_ref_[x], h_tst_[x])
                << "h_row mismatch, x " << x << ", width " << width_;
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafHpassRowTest);

TEST_P(VmafHpassRowTest, MatchTest) {
    run_match_test();
}

TEST_P(VmafHpassRowTest, MaxInputTest) {
    run_max_input_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafHpassRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(&svt_vmaf_hpass_row_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafHpassRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(&svt_vmaf_hpass_row_neon)));
#endif  // ARCH_AARCH64

using VmafVpassRowFunc = void (*)(const int16_t *r0, const int16_t *r1,
                                  const int16_t *r2, const int16_t *r3,
                                  const int16_t *r4, uint8_t *blur_row,
                                  int width, int steps_x);

using VmafVpassRowParam = std::tuple<int, VmafVpassRowFunc>;

class VmafVpassRowTest : public ::testing::TestWithParam<VmafVpassRowParam> {
  public:
    VmafVpassRowTest()
        : width_(TEST_GET_PARAM(0)), test_func_(TEST_GET_PARAM(1)) {
    }

    ~VmafVpassRowTest() override = default;

  protected:
    static const int kIterations = 100;

    void SetUp() override {
        const int steps_x = 2;
        const int alloc_width = width_ + 2 * steps_x;
        for (int k = 0; k < 5; ++k) {
            rows_[k] = static_cast<int16_t *>(
                svt_aom_memalign(16, (size_t)alloc_width * sizeof(int16_t)));
            ASSERT_NE(rows_[k], nullptr);
        }
        blur_ref_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*blur_ref_)));
        blur_tst_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width_ * sizeof(*blur_tst_)));
        ASSERT_NE(blur_ref_, nullptr);
        ASSERT_NE(blur_tst_, nullptr);
    }

    void TearDown() override {
        for (int k = 0; k < 5; ++k) {
            svt_aom_free(rows_[k]);
        }
        svt_aom_free(blur_ref_);
        svt_aom_free(blur_tst_);
    }

    int width_;
    VmafVpassRowFunc test_func_;
    int16_t *rows_[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    uint8_t *blur_ref_ = nullptr;
    uint8_t *blur_tst_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int steps_x = 2;
        const int alloc_width = width_ + 2 * steps_x;

        for (int i = 0; i < kIterations; ++i) {
            for (int k = 0; k < 5; ++k) {
                for (int x = 0; x < alloc_width; ++x) {
                    // H-pass outputs span [0, 16*255]; exercise the full range.
                    rows_[k][x] = (int16_t)(rnd.Rand8() << 4);
                }
            }
            for (int x = 0; x < width_; ++x) {
                blur_ref_[x] = rnd.Rand8();
                blur_tst_[x] = blur_ref_[x];
            }

            svt_vmaf_vpass_row_c(rows_[0],
                                 rows_[1],
                                 rows_[2],
                                 rows_[3],
                                 rows_[4],
                                 blur_ref_,
                                 width_,
                                 steps_x);
            test_func_(rows_[0],
                       rows_[1],
                       rows_[2],
                       rows_[3],
                       rows_[4],
                       blur_tst_,
                       width_,
                       steps_x);

            for (int x = 0; x < width_; ++x) {
                ASSERT_EQ(blur_ref_[x], blur_tst_[x])
                    << "blur_row mismatch, x " << x << ", width " << width_
                    << ", iteration " << i;
            }
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafVpassRowTest);

TEST_P(VmafVpassRowTest, MatchTest) {
    run_match_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafVpassRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(&svt_vmaf_vpass_row_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafVpassRowTest,
    ::testing::Combine(::testing::ValuesIn(kVmafWidths),
                       ::testing::Values(&svt_vmaf_vpass_row_neon)));
#endif  // ARCH_AARCH64

using VmafCountDetailFunc = uint32_t (*)(const uint8_t *src,
                                         const uint8_t *blur, int width,
                                         int height, int src_stride,
                                         int thresh);

using VmafCountDetailParam =
    std::tuple<std::pair<int, int>, int, VmafCountDetailFunc>;

class VmafCountDetailTest
    : public ::testing::TestWithParam<VmafCountDetailParam> {
  public:
    VmafCountDetailTest()
        : size_(TEST_GET_PARAM(0)),
          thresh_(TEST_GET_PARAM(1)),
          test_func_(TEST_GET_PARAM(2)) {
    }

    ~VmafCountDetailTest() override = default;

  protected:
    static const int kIterations = 10;

    void SetUp() override {
        const int width = size_.first;
        const int height = size_.second;
        src_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width * height));
        blur_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width * height));
        ASSERT_NE(src_, nullptr);
        ASSERT_NE(blur_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
        svt_aom_free(blur_);
    }

    std::pair<int, int> size_;
    int thresh_;
    VmafCountDetailFunc test_func_;
    uint8_t *src_ = nullptr;
    uint8_t *blur_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int i = 0; i < kIterations; ++i) {
            for (int p = 0; p < width * height; ++p) {
                src_[p] = rnd.Rand8();
                blur_[p] = rnd.Rand8();
            }
            const uint32_t res_ref = svt_vmaf_count_detail_le_c(
                src_, blur_, width, height, stride, thresh_);
            const uint32_t res_tst =
                test_func_(src_, blur_, width, height, stride, thresh_);
            ASSERT_EQ(res_ref, res_tst) << "iteration " << i;
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafCountDetailTest);

TEST_P(VmafCountDetailTest, MatchTest) {
    run_match_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafCountDetailTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(0, 3, 10, 32, 255),
                       ::testing::Values(&svt_vmaf_count_detail_le_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafCountDetailTest,
    ::testing::Combine(::testing::ValuesIn(kVmafSizes),
                       ::testing::Values(0, 3, 10, 32, 255),
                       ::testing::Values(&svt_vmaf_count_detail_le_neon)));
#endif  // ARCH_AARCH64

using VmafGradientCoherenceFunc = float (*)(const uint8_t *src, int width,
                                            int height, int stride);

using VmafGradientCoherenceParam =
    std::tuple<std::pair<int, int>, VmafGradientCoherenceFunc>;

class VmafGradientCoherenceTest
    : public ::testing::TestWithParam<VmafGradientCoherenceParam> {
  public:
    VmafGradientCoherenceTest()
        : size_(TEST_GET_PARAM(0)), test_func_(TEST_GET_PARAM(1)) {
    }

    ~VmafGradientCoherenceTest() override = default;

  protected:
    static const int kIterations = 10;

    void SetUp() override {
        const int width = size_.first;
        const int height = size_.second;
        // The SIMD kernels do masked 16-wide loads past the block edge; pad the
        // allocation to mirror the padded luma plane used in production.
        src_ = static_cast<uint8_t *>(
            svt_aom_memalign(16, (size_t)width * height + 128));
        ASSERT_NE(src_, nullptr);
    }

    void TearDown() override {
        svt_aom_free(src_);
    }

    std::pair<int, int> size_;
    VmafGradientCoherenceFunc test_func_;
    uint8_t *src_ = nullptr;

    void run_match_test() {
        SVTRandom rnd(8, false);
        const int width = size_.first;
        const int height = size_.second;
        const int stride = width;

        for (int i = 0; i < kIterations; ++i) {
            for (int p = 0; p < width * height; ++p) {
                src_[p] = rnd.Rand8();
            }
            const float res_ref = svt_vmaf_compute_gradient_coherence_c(
                src_, width, height, stride);
            const float res_tst = test_func_(src_, width, height, stride);
            ASSERT_EQ(res_ref, res_tst) << "iteration " << i;
        }
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(VmafGradientCoherenceTest);

TEST_P(VmafGradientCoherenceTest, MatchTest) {
    run_match_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, VmafGradientCoherenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(kVmafSizes),
        ::testing::Values(&svt_vmaf_compute_gradient_coherence_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, VmafGradientCoherenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(kVmafSizes),
        ::testing::Values(&svt_vmaf_compute_gradient_coherence_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, VmafGradientCoherenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(kVmafSizes),
        ::testing::Values(&svt_vmaf_compute_gradient_coherence_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, VmafGradientCoherenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(kVmafSizes),
        ::testing::Values(&svt_vmaf_compute_gradient_coherence_sve)));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64

#endif  // OPT_TUNE_VMAF

}  // namespace
