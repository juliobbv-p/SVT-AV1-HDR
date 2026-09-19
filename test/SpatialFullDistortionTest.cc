/*
 * Copyright(c) 2019 Intel Corporation
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>
#include "random.h"
#include "definitions.h"
#include "pic_operators.h"
#include "unit_test_utility.h"
#include "util.h"

namespace {
using svt_av1_test_tool::SVTRandom;

enum TestPattern { VAL_MIN, VAL_MAX, VAL_RANDOM };
using AreaSize = std::tuple<uint32_t, uint32_t>;

/**
 * @Brief Base class for SpatialFullDistortionFunc test.
 */
class SpatialFullDistortionFuncTestBase : public ::testing::Test {
  protected:
    virtual void init_data(TestPattern pattern) {
        const uint8_t mask = (1 << 8) - 1;
        switch (pattern) {
        case VAL_MIN: {
            std::fill(input_.begin(), input_.end(), 0);
            std::fill(recon_.begin(), recon_.end(), 0);
            break;
        }
        case VAL_MAX: {
            std::fill(input_.begin(), input_.end(), mask);
            std::fill(recon_.begin(), recon_.end(), mask);
            break;
        }
        case VAL_RANDOM: {
            svt_buf_random_u8(input_.data(), input_.size());
            svt_buf_random_u8(recon_.data(), recon_.size());
            break;
        }
        }
    }

    std::vector<uint8_t> input_{};
    std::vector<uint8_t> recon_{};
};

const AreaSize TEST_AREA_SIZES[] = {
    AreaSize(4, 4),    AreaSize(4, 8),    AreaSize(8, 4),   AreaSize(8, 8),
    AreaSize(16, 16),  AreaSize(12, 16),  AreaSize(4, 16),  AreaSize(16, 4),
    AreaSize(16, 8),   AreaSize(20, 16),  AreaSize(24, 16), AreaSize(28, 16),
    AreaSize(8, 16),   AreaSize(32, 32),  AreaSize(32, 8),  AreaSize(16, 32),
    AreaSize(8, 32),   AreaSize(32, 16),  AreaSize(16, 64), AreaSize(64, 16),
    AreaSize(64, 64),  AreaSize(64, 32),  AreaSize(32, 64), AreaSize(128, 128),
    AreaSize(96, 128), AreaSize(64, 128), AreaSize(128, 64)};

using SpatialKernelTestParam =
    std::tuple<AreaSize, decltype(&svt_spatial_full_distortion_kernel_c)>;

/**
 * @brief Unit test for spatial distortion calculation functions include:
 *  - svt_spatial_full_distortion_kernel_{avx2,avx512}
 *
 *
 * Test strategy:
 *  This test case combine different area width{4-128} x area
 * height{4-128} and different test pattern(VAL_MIN, VAL_MAX, VAL_RANDOM). Check
 * the result by compare result from reference function and avx2/sse2 function.
 *
 *
 * Expect result:
 *  Results from reference function and avx2/avx512 function are
 * equal.
 *
 *
 * Test cases:
 *  Width {4, 8, 12, 16, 20, 24, 28, 32, 64, 96, 128} x height{ 4, 8, 16, 32,
 * 64, 128} Test vector pattern {VAL_MIN, VAL_MIN, VAL_RANDOM}
 *
 */
class SpatialFullDistortionKernelFuncTest
    : public SpatialFullDistortionFuncTestBase,
      public ::testing::WithParamInterface<SpatialKernelTestParam> {
  protected:
    SpatialFullDistortionKernelFuncTest() {
        input_.resize(MAX_SB_SIZE * input_stride_);
        recon_.resize(MAX_SB_SIZE * recon_stride_);
    }
    void RunCheckOutput(TestPattern pattern);
    uint32_t input_stride_{svt_create_random_aligned_stride(MAX_SB_SIZE, 64)};
    uint32_t recon_stride_{svt_create_random_aligned_stride(MAX_SB_SIZE, 64)};
};

void SpatialFullDistortionKernelFuncTest::RunCheckOutput(TestPattern pattern) {
    const uint32_t area_width = std::get<0>(TEST_GET_PARAM(0));
    const uint32_t area_height = std::get<1>(TEST_GET_PARAM(0));
    const auto test_func{TEST_GET_PARAM(1)};
    for (int i = 0; i < 10; i++) {
        init_data(pattern);
        const uint64_t dist_test = test_func(input_.data(),
                                             0,
                                             input_stride_,
                                             recon_.data(),
                                             0,
                                             recon_stride_,
                                             area_width,
                                             area_height);
        const uint64_t dist_c =
            svt_spatial_full_distortion_kernel_c(input_.data(),
                                                 0,
                                                 input_stride_,
                                                 recon_.data(),
                                                 0,
                                                 recon_stride_,
                                                 area_width,
                                                 area_height);

        EXPECT_EQ(dist_test, dist_c)
            << "Compare Spatial distortion result error";
    }
}
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    SpatialFullDistortionKernelFuncTest);

TEST_P(SpatialFullDistortionKernelFuncTest, Random) {
    RunCheckOutput(VAL_RANDOM);
}

TEST_P(SpatialFullDistortionKernelFuncTest, ExtremeMin) {
    RunCheckOutput(VAL_MIN);
}

TEST_P(SpatialFullDistortionKernelFuncTest, ExtremeMax) {
    RunCheckOutput(VAL_MAX);
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, SpatialFullDistortionKernelFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_spatial_full_distortion_kernel_sse4_1)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, SpatialFullDistortionKernelFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_spatial_full_distortion_kernel_avx2)));

#if EN_AVX512_SUPPORT
INSTANTIATE_TEST_SUITE_P(
    AVX512, SpatialFullDistortionKernelFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_spatial_full_distortion_kernel_avx512)));
#endif
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, SpatialFullDistortionKernelFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_spatial_full_distortion_kernel_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, SpatialFullDistortionKernelFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_spatial_full_distortion_kernel_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD
#endif  // ARCH_AARCH64

class FullDistortionKernel16BitsFuncTest
    : public SpatialFullDistortionFuncTestBase,
      public ::testing::WithParamInterface<SpatialKernelTestParam> {
    virtual void init_data(TestPattern pattern) override {
        /// Support up to 10 bit depth
        constexpr uint16_t mask = (1 << 10) - 1;
        uint16_t *input_16bit = reinterpret_cast<uint16_t *>(input_.data());
        uint16_t *recon_16bit = reinterpret_cast<uint16_t *>(recon_.data());
        SVTRandom rnd{0, mask};

        switch (pattern) {
        case VAL_MIN: {
            std::fill_n(input_16bit, input_.size() / 2, 0);
            std::fill_n(recon_16bit, recon_.size() / 2, mask);
            break;
        }
        case VAL_MAX: {
            std::fill_n(input_16bit, input_.size() / 2, mask);
            std::fill_n(recon_16bit, recon_.size() / 2, 0);
            break;
        }
        case VAL_RANDOM: {
            std::generate_n(input_16bit, input_.size() / 2, [&rnd]() {
                return rnd.random();
            });
            std::generate_n(recon_16bit, recon_.size() / 2, [&rnd]() {
                return rnd.random();
            });
            break;
        }
        }
    }

  public:
    FullDistortionKernel16BitsFuncTest() {
        input_.resize(MAX_SB_SIZE * input_stride_ * 2);
        recon_.resize(MAX_SB_SIZE * recon_stride_ * 2);
    }

  protected:
    uint32_t input_stride_{svt_create_random_aligned_stride(MAX_SB_SIZE, 64)};
    uint32_t recon_stride_{svt_create_random_aligned_stride(MAX_SB_SIZE, 64)};
    void RunCheckOutput(TestPattern pattern);
};

void FullDistortionKernel16BitsFuncTest::RunCheckOutput(TestPattern pattern) {
    const auto test_func{TEST_GET_PARAM(1)};
    const uint32_t area_width = std::get<0>(TEST_GET_PARAM(0));
    const uint32_t area_height = std::get<1>(TEST_GET_PARAM(0));
    for (int i = 0; i < 10; i++) {
        init_data(pattern);
        const uint64_t dist_test = test_func(input_.data(),
                                             0,
                                             input_stride_,
                                             recon_.data(),
                                             0,
                                             recon_stride_,
                                             area_width,
                                             area_height);
        const uint64_t dist_c =
            svt_full_distortion_kernel16_bits_c(input_.data(),
                                                0,
                                                input_stride_,
                                                recon_.data(),
                                                0,
                                                recon_stride_,
                                                area_width,
                                                area_height);

        EXPECT_EQ(dist_test, dist_c)
            << "Compare Full distortion kernel 16 bits result error";
    }
}
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    FullDistortionKernel16BitsFuncTest);

TEST_P(FullDistortionKernel16BitsFuncTest, Random) {
    RunCheckOutput(VAL_RANDOM);
}

TEST_P(FullDistortionKernel16BitsFuncTest, ExtremeMin) {
    RunCheckOutput(VAL_MIN);
}

TEST_P(FullDistortionKernel16BitsFuncTest, ExtremeMax) {
    RunCheckOutput(VAL_MAX);
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, FullDistortionKernel16BitsFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_full_distortion_kernel16_bits_sse4_1)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, FullDistortionKernel16BitsFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_full_distortion_kernel16_bits_avx2)));

#endif

#ifdef ARCH_AARCH64

INSTANTIATE_TEST_SUITE_P(
    NEON, FullDistortionKernel16BitsFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_full_distortion_kernel16_bits_neon)));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, FullDistortionKernel16BitsFuncTest,
    ::testing::Combine(
        ::testing::ValuesIn(TEST_AREA_SIZES),
        ::testing::Values(svt_full_distortion_kernel16_bits_sve)));
#endif  // HAVE_SVE
#endif

using FullDistortionKernel32Bits =
    ::testing::TestWithParam<decltype(&svt_full_distortion_kernel32_bits_c)>;

TEST_P(FullDistortionKernel32Bits, CheckOutput) {
    std::array<uint64_t, DIST_CALC_TOTAL> result_ref;
    std::array<uint64_t, DIST_CALC_TOTAL> result_mod;
    const auto stride{svt_create_random_aligned_stride(MAX_SB_SIZE, 64)};
    std::vector<int32_t> coeff(MAX_SB_SIZE * stride);
    std::vector<int32_t> recon(MAX_SB_SIZE * stride);
    const auto func{GetParam()};
    // The kernel is only ever called with AV1 transform widths 4/8/16/32
    // (64-wide transforms pass 32) -- independent of bit depth. Heights are the
    // same set, but pf_shape (N2/N4) can additionally halve/quarter the height
    // (no clamp), so heights of 1 and 2 also occur. Only the coefficient
    // magnitude differs by bit depth: 8-bit fits int16, HBD spans "bd + 8".
#if CONFIG_ENABLE_HIGH_BIT_DEPTH
    const int32_t max_abs = 1 << 17;
#else
    // int16 range so (coeff - recon)^2 stays < 2^32 for the 8-bit kernel.
    const int32_t max_abs = (1 << 15) - 1;
#endif
    const std::array<uint32_t, 4> widths = {4, 8, 16, 32};
    const std::array<uint32_t, 6> heights = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < 10; i++) {
        svt_buf_random_s32_with_max(coeff.data(), coeff.size(), max_abs);
        svt_buf_random_s32_with_max(recon.data(), recon.size(), max_abs);
        for (uint32_t area_width : widths) {
            for (uint32_t area_height : heights) {
                // Height is always a power of two, so the only odd height is 1,
                // and it only occurs for width 4 (a 4-tall TX reduced by
                // pf_shape N4). Width 8/16/32 always get an even height >= 2.
                if (area_height == 1 && area_width != 4)
                    continue;
                svt_full_distortion_kernel32_bits_c(coeff.data(),
                                                    recon.data(),
                                                    stride,
                                                    area_width,
                                                    area_height,
                                                    result_ref.data());
                func(coeff.data(),
                     recon.data(),
                     stride,
                     area_width,
                     area_height,
                     result_mod.data());

                EXPECT_EQ(result_ref, result_mod);
            }
        }
    }
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, FullDistortionKernel32Bits,
    ::testing::Values(svt_full_distortion_kernel32_bits_sse4_1));

INSTANTIATE_TEST_SUITE_P(
    AVX2, FullDistortionKernel32Bits,
    ::testing::Values(svt_full_distortion_kernel32_bits_avx2));

#endif

#ifdef ARCH_AARCH64

INSTANTIATE_TEST_SUITE_P(
    NEON, FullDistortionKernel32Bits,
    ::testing::Values(svt_full_distortion_kernel32_bits_neon));

#endif

using FullDistortionKernelCbfZero32Bits = ::testing::TestWithParam<
    decltype(&svt_full_distortion_kernel_cbf_zero32_bits_c)>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(
    FullDistortionKernelCbfZero32Bits);

TEST_P(FullDistortionKernelCbfZero32Bits, CheckOutput) {
    const auto func{GetParam()};
    std::array<uint64_t, DIST_CALC_TOTAL> result_ref;
    std::array<uint64_t, DIST_CALC_TOTAL> result_mod;
    const auto coeff_stride{svt_create_random_aligned_stride(MAX_SB_SIZE, 64)};
    std::vector<int32_t> coeff(MAX_SB_SIZE * coeff_stride);
    for (int i = 0; i < 10; i++) {
        svt_buf_random_u32_with_max(
            reinterpret_cast<uint32_t *>(coeff.data()), coeff.size(), 1 << 15);
        for (uint32_t area_width = 4; area_width <= 128; area_width += 4) {
            for (uint32_t area_height = 4; area_height <= 128;
                 area_height += 4) {
                svt_full_distortion_kernel_cbf_zero32_bits_c(coeff.data(),
                                                             coeff_stride,
                                                             result_ref.data(),
                                                             area_width,
                                                             area_height);
                func(coeff.data(),
                     coeff_stride,
                     result_mod.data(),
                     area_width,
                     area_height);

                EXPECT_EQ(result_ref, result_mod);
            }
        }
    }
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(
    SSE4_1, FullDistortionKernelCbfZero32Bits,
    ::testing::Values(svt_full_distortion_kernel_cbf_zero32_bits_sse4_1));

INSTANTIATE_TEST_SUITE_P(
    AVX2, FullDistortionKernelCbfZero32Bits,
    ::testing::Values(svt_full_distortion_kernel_cbf_zero32_bits_avx2));

#endif

#ifdef ARCH_AARCH64

INSTANTIATE_TEST_SUITE_P(
    NEON, FullDistortionKernelCbfZero32Bits,
    ::testing::Values(svt_full_distortion_kernel_cbf_zero32_bits_neon));

#endif  //  ARCH_AARCH64

}  // namespace
