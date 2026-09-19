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
 * @file FwhtAsmTest.cc
 *
 * @brief Unit test for svt_av1_fwht4x4 SIMD variants against the C reference.
 *
 ******************************************************************************/

#include <stdint.h>
#include "gtest/gtest.h"
#include "aom_dsp_rtcd.h"
#include "random.h"

namespace {
using svt_av1_test_tool::SVTRandom;

typedef void (*Fwht4x4Func)(int16_t *input, int32_t *output, uint32_t stride);

class Fwht4x4AsmTest : public ::testing::TestWithParam<Fwht4x4Func> {
  protected:
    void run_match(int times) {
        SVTRandom rnd(16, true);
        Fwht4x4Func test_func = GetParam();
        int16_t in[16];
        int32_t out_ref[16], out_tst[16];
        for (int t = 0; t < times; ++t) {
            for (int i = 0; i < 16; ++i)
                in[i] = (int16_t)rnd.random();
            svt_av1_fwht4x4_c(in, out_ref, 4);
            test_func(in, out_tst, 4);
            for (int i = 0; i < 16; ++i)
                ASSERT_EQ(out_ref[i], out_tst[i])
                    << "mismatch at idx " << i << " iter " << t;
        }
    }

    void run_case(const int16_t *in) {
        Fwht4x4Func test_func = GetParam();
        int32_t out_ref[16], out_tst[16];
        svt_av1_fwht4x4_c((int16_t *)in, out_ref, 4);
        test_func((int16_t *)in, out_tst, 4);
        for (int i = 0; i < 16; ++i)
            ASSERT_EQ(out_ref[i], out_tst[i]) << "mismatch at idx " << i;
    }

    // The kernel reads the 4x4 block with a caller-supplied stride, so the
    // rows are not necessarily contiguous. Exercise strides other than 4.
    void run_stride(int times, uint32_t stride) {
        SVTRandom rnd(16, true);
        Fwht4x4Func test_func = GetParam();
        int16_t *in = new int16_t[4 * stride];
        int32_t out_ref[16], out_tst[16];
        for (int t = 0; t < times; ++t) {
            for (uint32_t k = 0; k < 4 * stride; ++k)
                in[k] = (int16_t)rnd.random();
            svt_av1_fwht4x4_c(in, out_ref, stride);
            test_func(in, out_tst, stride);
            for (int i = 0; i < 16; ++i)
                ASSERT_EQ(out_ref[i], out_tst[i])
                    << "mismatch at idx " << i << " stride " << stride;
        }
        delete[] in;
    }

    void run_extreme() {
        int16_t in[16];
        // all-min / all-max / all-zero
        const int16_t consts[] = {INT16_MIN, INT16_MAX, 0};
        for (size_t c = 0; c < sizeof(consts) / sizeof(consts[0]); ++c) {
            for (int i = 0; i < 16; ++i)
                in[i] = consts[c];
            run_case(in);
            if (HasFatalFailure())
                return;
        }
        // checkerboard
        for (int i = 0; i < 16; ++i)
            in[i] = (i & 1) ? INT16_MAX : INT16_MIN;
        run_case(in);
        if (HasFatalFailure())
            return;
        // single impulses
        for (int p = 0; p < 16; ++p) {
            for (int i = 0; i < 16; ++i)
                in[i] = 0;
            in[p] = INT16_MAX;
            run_case(in);
            if (HasFatalFailure())
                return;
        }
    }
};

TEST_P(Fwht4x4AsmTest, MatchTest) {
    run_match(10000);
}

TEST_P(Fwht4x4AsmTest, ExtremeTest) {
    run_extreme();
}

TEST_P(Fwht4x4AsmTest, StrideTest) {
    run_stride(1000, 8);
    run_stride(1000, 16);
    run_stride(1000, 20);
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(SSE4_1, Fwht4x4AsmTest,
                         ::testing::Values(svt_av1_fwht4x4_sse4_1));
#endif

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, Fwht4x4AsmTest,
                         ::testing::Values(svt_av1_fwht4x4_neon));
#endif

}  // namespace
