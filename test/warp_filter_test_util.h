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

#ifndef AOM_TEST_WARP_FILTER_TEST_UTIL_H_
#define AOM_TEST_WARP_FILTER_TEST_UTIL_H_

#include "gtest/gtest.h"
#include "common_dsp_rtcd.h"
#include "random.h"

namespace libaom_test {

namespace AV1WarpFilter {

using warp_affine_func = decltype(&svt_av1_warp_affine_c);

using WarpTestParam = std::tuple<int, int, int, warp_affine_func>;
using WarpTestParams = std::tuple<WarpTestParam, int, int, int, int>;

::testing::internal::ParamGenerator<WarpTestParams> BuildParams(
    warp_affine_func filter);

class AV1WarpFilterTest : public ::testing::TestWithParam<WarpTestParams> {
  public:
    AV1WarpFilterTest() = default;

  protected:
    void RunCheckOutput(const warp_affine_func test_impl);

    svt_av1_test_tool::SVTRandom rnd_{0, (1 << 8) - 1};
};

}  // namespace AV1WarpFilter

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1WarpFilterTest);

namespace AV1HighbdWarpFilter {
using highbd_warp_affine_func = decltype(&svt_av1_highbd_warp_affine_c);

using HighbdWarpTestParam =
    std::tuple<int, int, int, int, highbd_warp_affine_func>;
using HighbdWarpTestParams =
    std::tuple<HighbdWarpTestParam, int, int, int, int>;

::testing::internal::ParamGenerator<HighbdWarpTestParams> BuildParams(
    highbd_warp_affine_func filter);

class AV1HighbdWarpFilterTest
    : public ::testing::TestWithParam<HighbdWarpTestParams> {
  public:
    AV1HighbdWarpFilterTest() = default;

  protected:
    void RunCheckOutput(const highbd_warp_affine_func test_impl);

    svt_av1_test_tool::SVTRandom rnd_{0, (1 << 10) - 1};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AV1HighbdWarpFilterTest);

}  // namespace AV1HighbdWarpFilter

}  // namespace libaom_test

#endif  // AOM_TEST_WARP_FILTER_TEST_UTIL_H_
