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
 * @file WedgeUtilTest.cc
 *
 * @brief Unit test for util functions in wedge prediction:
 * - svt_av1_wedge_sign_from_residuals_avx2
 * - svt_av1_wedge_compute_delta_squares_avx2
 * - svt_av1_wedge_sse_from_residuals_avx2
 * - svt_aom_sum_squares_i16_sse2
 *
 * @author Cidana-Wenyao
 *
 ******************************************************************************/
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include "common_dsp_rtcd.h"
#include "common_utils.h"
#include "aom_dsp_rtcd.h"
#include "random.h"
#include "svt_malloc.h"
#include "util.h"

using svt_av1_test_tool::SVTRandom;
namespace {

// Choose the mask sign for a compound predictor.
class WedgeUtilTest : public ::testing::Test {
  public:
    void *operator new(size_t size) {
        if (void *ptr = svt_aom_memalign(alignof(WedgeUtilTest), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

  protected:
    alignas(32) std::array<uint8_t, MAX_SB_SQUARE> m{}; /* mask */
    /* two predicted residual */
    alignas(32) std::array<int16_t, MAX_SB_SQUARE> r0{}, r1{};
};

// test svt_av1_wedge_sign_from_residuals
using WedgeSignFromResidualsFunc =
    decltype(&svt_av1_wedge_sign_from_residuals_c);

constexpr auto MAX_MASK_VALUE = 1 << WEDGE_WEIGHT_BITS;

class WedgeSignFromResidualsTest
    : public WedgeUtilTest,
      public ::testing::WithParamInterface<WedgeSignFromResidualsFunc> {
  protected:
    void wedge_sign_test(int N, int k) {
        alignas(32) std::array<int16_t, MAX_SB_SQUARE> ds{};
        // pre-compute limit
        // MAX_MASK_VALUE/2 * (sum(r0**2) - sum(r1**2))
        int64_t limit;
        limit = (int64_t)svt_aom_sum_squares_i16_c(r0.data(), N);
        limit -= (int64_t)svt_aom_sum_squares_i16_c(r1.data(), N);
        limit *= (1 << WEDGE_WEIGHT_BITS) / 2;

        // calculate ds: r0**2 - r1**2
        for (int i = 0; i < N; ++i)
            ds[i] = clamp(r0[i] * r0[i] - r1[i] * r1[i], INT16_MIN, INT16_MAX);
        const int8_t ref_sign =
            svt_av1_wedge_sign_from_residuals_c(ds.data(), m.data(), N, limit);
        const int8_t tst_sign = test_func_(ds.data(), m.data(), N, limit);
        ASSERT_EQ(ref_sign, tst_sign)
            << "unit test for svt_av1_wedge_sign_from_residuals fail at "
               "iteration "
            << k;
    }

    void MaskSignRandomTest() {
        constexpr int iterations = 10000;
        SVTRandom rnd{13, true};             // max residual is 13-bit
        SVTRandom m_rnd{0, MAX_MASK_VALUE};  // [0, MAX_MASK_VALUE]
        SVTRandom n_rnd{1, 8191 / 64};  // required by assembly implementation

        for (int k = 0; k < iterations; ++k) {
            // populate the residual buffer randomly
            for (int i = 0; i < MAX_SB_SQUARE; ++i) {
                r0[i] = rnd.random();
                r1[i] = rnd.random();
                m[i] = m_rnd.random();
            }

            // N should be multiple of 64, required by
            // svt_av1_wedge_sign_from_residuals_avx2
            const int N = 64 * n_rnd.random();
            wedge_sign_test(N, k);
        }
    }

    void MaskSignExtremeTest() {
        constexpr int int_13bit_max = (1 << 12) - 1;
        SVTRandom rnd{13, true};             // max residual is 13-bit
        SVTRandom m_rnd{0, MAX_MASK_VALUE};  // [0, MAX_MASK_VALUE]
        SVTRandom n_rnd{1, 8191 / 64};  // required by assembly implementation

        for (int k = 0; k < 4; ++k) {
            switch (k) {
            case 0:
                r0.fill(0);
                r1.fill(int_13bit_max);
                break;
            case 1:
                r0.fill(int_13bit_max);
                r1.fill(0);
                break;
            case 2:
                r0.fill(-int_13bit_max);
                r1.fill(0);
                break;
            case 3:
                r0.fill(0);
                r1.fill(-int_13bit_max);
                break;
            }

            m.fill(MAX_MASK_VALUE);

            // N should be multiple of 64, required by
            // svt_av1_wedge_sign_from_residuals_avx2
            const int N = 64 * n_rnd.random();

            wedge_sign_test(N, k);
        }
    }

    WedgeSignFromResidualsFunc test_func_{GetParam()};
};

TEST_P(WedgeSignFromResidualsTest, RandomTest) {
    MaskSignRandomTest();
}

TEST_P(WedgeSignFromResidualsTest, ExtremeTest) {
    MaskSignExtremeTest();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, WedgeSignFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sign_from_residuals_sse2));

INSTANTIATE_TEST_SUITE_P(
    AVX2, WedgeSignFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sign_from_residuals_avx2));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, WedgeSignFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sign_from_residuals_neon));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, WedgeSignFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sign_from_residuals_sve));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64

// test svt_av1_wedge_compute_delta_squares
using WedgeComputeDeltaSquaresFunc =
    decltype(&svt_av1_wedge_compute_delta_squares_c);

class WedgeComputeDeltaSquaresTest
    : public WedgeUtilTest,
      public ::testing::WithParamInterface<WedgeComputeDeltaSquaresFunc> {
  protected:
    void ComputeDeltaSquareTest() {
        constexpr int iterations = 10000;
        SVTRandom rnd{13, true};  // max residual is 13-bit
        SVTRandom n_rnd{1, MAX_SB_SQUARE / 64};
        alignas(32) std::array<int16_t, MAX_SB_SQUARE> ref_diff;
        alignas(32) std::array<int16_t, MAX_SB_SQUARE> tst_diff;

        for (int k = 0; k < iterations; ++k) {
            // populate the residual buffer randomly
            for (int i = 0; i < MAX_SB_SQUARE; ++i) {
                r0[i] = rnd.random();
                r1[i] = rnd.random();
            }
            ref_diff.fill(0);
            tst_diff.fill(0);

            // N should be multiple of 64, required by
            // svt_av1_wedge_compute_delta_squares
            const int N = 64 * n_rnd.random();

            svt_av1_wedge_compute_delta_squares_c(
                ref_diff.data(), r0.data(), r1.data(), N);
            test_func_(tst_diff.data(), r0.data(), r1.data(), N);

            // check the output
            for (int i = 0; i < N; ++i) {
                ASSERT_EQ(ref_diff[i], tst_diff[i])
                    << "unit test for svt_av1_wedge_compute_delta_squares "
                       "fail at "
                       "iteration "
                    << k;
            }
        }
    }

    WedgeComputeDeltaSquaresFunc test_func_{GetParam()};
};

// element-by-element calculate the difference of square
TEST_P(WedgeComputeDeltaSquaresTest, ComputeDeltaSquareTest) {
    ComputeDeltaSquareTest();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, WedgeComputeDeltaSquaresTest,
    ::testing::Values(svt_av1_wedge_compute_delta_squares_avx2));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, WedgeComputeDeltaSquaresTest,
    ::testing::Values(svt_av1_wedge_compute_delta_squares_neon));
#endif  // ARCH_AARCH64

// test svt_av1_wedge_sse_from_residuals
using WedgeSseFromResidualsFunc = decltype(&svt_av1_wedge_sse_from_residuals_c);

class WedgeSseFromResidualsTest
    : public WedgeUtilTest,
      public ::testing::WithParamInterface<WedgeSseFromResidualsFunc> {
  protected:
    void SseFromResidualRandomTest() {
        constexpr int iterations = 10000;
        SVTRandom rnd{13, true};             // max residual is 13-bit
        SVTRandom m_rnd{0, MAX_MASK_VALUE};  // [0, MAX_MASK_VALUE]
        SVTRandom n_rnd{1, MAX_SB_SQUARE / 64};

        for (int k = 0; k < iterations; ++k) {
            // populate the residual buffer randomly
            for (int i = 0; i < MAX_SB_SQUARE; ++i) {
                r0[i] = rnd.random();
                r1[i] = rnd.random();
                m[i] = m_rnd.random();
            }

            // N should be multiple of 64, required by
            // svt_av1_wedge_sse_from_residuals
            const int N = 64 * n_rnd.random();

            uint64_t ref_sse = svt_av1_wedge_sse_from_residuals_c(
                r0.data(), r1.data(), m.data(), N);
            uint64_t tst_sse = test_func_(r0.data(), r1.data(), m.data(), N);

            // check output
            ASSERT_EQ(ref_sse, tst_sse)
                << "unit test for svt_av1_wedge_sse_from_residuals fail at "
                   "iteration "
                << k;
        }
    }

    void SseFromResidualExtremeTest() {
        constexpr int int_13bit_max = (1 << 12) - 1;
        SVTRandom rnd{13, true};             // max residual is 13-bit
        SVTRandom m_rnd{0, MAX_MASK_VALUE};  // [0, MAX_MASK_VALUE]
        SVTRandom n_rnd{1, MAX_SB_SQUARE / 64};

        for (int k = 0; k < 4; ++k) {
            switch (k) {
            case 0:
                r0.fill(0);
                r1.fill(int_13bit_max);
                break;
            case 1:
                r0.fill(int_13bit_max);
                r1.fill(0);
                break;
            case 2:
                r0.fill(-int_13bit_max);
                r1.fill(0);
                break;
            case 3:
                r0.fill(0);
                r1.fill(-int_13bit_max);
                break;
            }

            m.fill(MAX_MASK_VALUE);

            // N should be multiple of 64, required by
            // svt_av1_wedge_sse_from_residuals
            const int N = 64 * n_rnd.random();

            uint64_t ref_sse = svt_av1_wedge_sse_from_residuals_c(
                r0.data(), r1.data(), m.data(), N);
            uint64_t tst_sse = test_func_(r0.data(), r1.data(), m.data(), N);

            // check output
            ASSERT_EQ(ref_sse, tst_sse)
                << "unit test for svt_av1_wedge_sse_from_residuals fail at "
                   "iteration "
                << k;
        }
    }

    WedgeSseFromResidualsFunc test_func_{GetParam()};
};

// calculate the sse of two prediction combined with mask m
TEST_P(WedgeSseFromResidualsTest, RandomTest) {
    SseFromResidualRandomTest();
}

TEST_P(WedgeSseFromResidualsTest, ExtremeTest) {
    SseFromResidualExtremeTest();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, WedgeSseFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sse_from_residuals_sse2));

INSTANTIATE_TEST_SUITE_P(
    AVX2, WedgeSseFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sse_from_residuals_avx2));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, WedgeSseFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sse_from_residuals_neon));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, WedgeSseFromResidualsTest,
    ::testing::Values(svt_av1_wedge_sse_from_residuals_sve));
#endif
#endif  // ARCH_AARCH64

using AomSumSquaresI16Func = decltype(&svt_aom_sum_squares_i16_c);
using AomHSumSquaresParam = ::testing::tuple<BlockSize, AomSumSquaresI16Func>;

class AomSumSquaresTest : public ::testing::TestWithParam<AomHSumSquaresParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const AomSumSquaresI16Func test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> src_{};
        constexpr int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            std::generate_n(src_.begin(), width * height, [this]() {
                return rnd_.random();
            });

            uint64_t res_ref_ =
                svt_aom_sum_squares_i16_c(src_.data(), width * height);

            uint64_t res_tst_ = test_impl(src_.data(), width * height);

            ASSERT_EQ(res_ref_, res_tst_);
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(AomSumSquaresTest, MatchTest) {
    run_test();
}

#if ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, AomSumSquaresTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_sum_squares_i16_sse2)));
#endif  // ARCH_X86_64

#if ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AomSumSquaresTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_sum_squares_i16_neon)));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, AomSumSquaresTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_sum_squares_i16_sve)));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64

}  // namespace
