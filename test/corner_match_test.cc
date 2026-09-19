/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#include "gtest/gtest.h"
#include <array>
#include <tuple>
#include "aom_dsp_rtcd.h"
#include "util.h"
#include "acm_random.h"

namespace {
using libaom_test::ACMRandom;

using ComputeCrossCorrFunc = decltype(&svt_av1_compute_cross_correlation_c);
using CornerMatchParam = std::tuple<int, int, ComputeCrossCorrFunc>;

class AV1CornerMatchTest : public ::testing::TestWithParam<CornerMatchParam> {
  protected:
    void RunCheckOutput();
    const int mode_{TEST_GET_PARAM(0)};
    const int match_sz_{TEST_GET_PARAM(1)};
    const ComputeCrossCorrFunc target_func_{TEST_GET_PARAM(2)};

    ACMRandom rnd_{};
};

void AV1CornerMatchTest::RunCheckOutput() {
    const int match_sz_by2 = ((match_sz_ - 1) / 2);
    constexpr int w = 128, h = 128;
    constexpr int num_iters = 10000;

    std::array<uint8_t, w * h> input1{};
    std::array<uint8_t, w * h> input2{};

    // Test the two extreme cases:
    // i) Random data, should have correlation close to 0
    // ii) Linearly related data + noise, should have correlation close to 1
    if (mode_ == 0) {
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) {
                input1[i * w + j] = rnd_.Rand8();
                input2[i * w + j] = rnd_.Rand8();
            }
    } else if (mode_ == 1) {
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j) {
                int v = rnd_.Rand8();
                input1[i * w + j] = v;
                input2[i * w + j] = (v / 2) + (rnd_.Rand8() & 15);
            }
    }

    for (int i = 0; i < num_iters; ++i) {
        int x1 = match_sz_by2 + rnd_.PseudoUniform(w - 2 * match_sz_by2);
        int y1 = match_sz_by2 + rnd_.PseudoUniform(h - 2 * match_sz_by2);
        int x2 = match_sz_by2 + rnd_.PseudoUniform(w - 2 * match_sz_by2);
        int y2 = match_sz_by2 + rnd_.PseudoUniform(h - 2 * match_sz_by2);

        double res_c = svt_av1_compute_cross_correlation_c(
            input1.data(), w, x1, y1, input2.data(), w, x2, y2, match_sz_);
        double res_simd = target_func_(
            input1.data(), w, x1, y1, input2.data(), w, x2, y2, match_sz_);

        ASSERT_EQ(res_simd, res_c);
    }
}

TEST_P(AV1CornerMatchTest, CheckOutput) {
    RunCheckOutput();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AV1CornerMatchTest,
    ::testing::Combine(
        testing::Values(0, 1), testing::Range(3, 16, 2),
        testing::Values(svt_av1_compute_cross_correlation_sse4_1)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, AV1CornerMatchTest,
    ::testing::Combine(
        testing::Values(0, 1), testing::Range(3, 16, 2),
        testing::Values(svt_av1_compute_cross_correlation_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AV1CornerMatchTest,
    ::testing::Combine(
        testing::Values(0, 1), testing::Range(3, 16, 2),
        testing::Values(svt_av1_compute_cross_correlation_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, AV1CornerMatchTest,
    ::testing::Combine(
        testing::Values(0, 1), testing::Range(3, 16, 2),
        testing::Values(svt_av1_compute_cross_correlation_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, AV1CornerMatchTest,
    ::testing::Combine(testing::Values(0, 1), testing::Range(3, 16, 2),
                       testing::Values(svt_av1_compute_cross_correlation_sve)));
#endif  // HAVE_SVE

#endif  // ARCH_AARCH64

}  // namespace
