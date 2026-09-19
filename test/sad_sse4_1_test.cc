/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

// Verifies that the SSE4.1 single-reference (svt_aom_sad<W>x<H>) and
// 4-reference (svt_aom_sad<W>x<H>x4d) SAD kernels are bit-identical to their C
// references, and provides a disabled speed microbenchmark.

#include <cstdint>

#include "gtest/gtest.h"
#include "aom_dsp_rtcd.h"
#include "random.h"
#include "svt_time.h"

namespace {
using svt_av1_test_tool::SVTRandom;

typedef uint32_t (*SadFn)(const uint8_t *, int, const uint8_t *, int);
typedef void (*SadX4Fn)(const uint8_t *, int, const uint8_t *const[], int,
                        uint32_t *);

struct SadEntry {
    int w, h;
    SadFn c, sse4_1;
    SadX4Fn c_x4, sse4_1_x4;
};

#define SAD_ENTRY(w, h)              \
    {w,                              \
     h,                              \
     &svt_aom_sad##w##x##h##_c,      \
     &svt_aom_sad##w##x##h##_sse4_1, \
     &svt_aom_sad##w##x##h##x4d_c,   \
     &svt_aom_sad##w##x##h##x4d_sse4_1}

const SadEntry kSadSizes[] = {
    SAD_ENTRY(128, 128), SAD_ENTRY(128, 64), SAD_ENTRY(64, 128),
    SAD_ENTRY(64, 64),   SAD_ENTRY(64, 32),  SAD_ENTRY(64, 16),
    SAD_ENTRY(32, 64),   SAD_ENTRY(32, 32),  SAD_ENTRY(32, 16),
    SAD_ENTRY(32, 8),    SAD_ENTRY(16, 64),  SAD_ENTRY(16, 32),
    SAD_ENTRY(16, 16),   SAD_ENTRY(16, 8),   SAD_ENTRY(16, 4),
    SAD_ENTRY(8, 32),    SAD_ENTRY(8, 16),   SAD_ENTRY(8, 8),
    SAD_ENTRY(8, 4),     SAD_ENTRY(4, 16),   SAD_ENTRY(4, 8),
    SAD_ENTRY(4, 4)};

class SadSse41Test : public ::testing::TestWithParam<int> {};

TEST_P(SadSse41Test, MatchesC) {
    const SadEntry &e = kSadSizes[GetParam()];
    const int stride = 128;
    SVTRandom rnd(8, false);

    static uint8_t src[128 * 128];
    static uint8_t ref[(128 + 8) * 128];

    for (int trial = 0; trial < 32; ++trial) {
        for (int i = 0; i < (int)sizeof(src); ++i)
            src[i] = (uint8_t)rnd.random();
        for (int i = 0; i < (int)sizeof(ref); ++i)
            ref[i] = (uint8_t)rnd.random();

        const uint32_t sad_c = e.c(src, stride, ref, stride);
        const uint32_t sad_sse4_1 = e.sse4_1(src, stride, ref, stride);
        ASSERT_EQ(sad_c, sad_sse4_1)
            << e.w << "x" << e.h << " single, trial " << trial;

        const uint8_t *refs[4] = {
            ref, ref + 7, ref + stride + 3, ref + 2 * stride + 11};
        uint32_t res_c[4], res_sse4_1[4];
        e.c_x4(src, stride, refs, stride, res_c);
        e.sse4_1_x4(src, stride, refs, stride, res_sse4_1);
        for (int i = 0; i < 4; ++i) {
            ASSERT_EQ(res_c[i], res_sse4_1[i])
                << e.w << "x" << e.h << " x4d[" << i << "], trial " << trial;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SSE4_1, SadSse41Test,
                         ::testing::Range(0, (int)(sizeof(kSadSizes) /
                                                   sizeof(kSadSizes[0]))));

// Run with: --gtest_also_run_disabled_tests --gtest_filter=*SadSse41Speed*
TEST(SadSse41Speed, DISABLED_SadSse41Speed) {
    const int stride = 128;
    const int iters = 200000;
    static uint8_t src[128 * 128];
    static uint8_t ref[(128 + 8) * 128];
    SVTRandom rnd(8, false);
    for (int i = 0; i < (int)sizeof(src); ++i)
        src[i] = (uint8_t)rnd.random();
    for (int i = 0; i < (int)sizeof(ref); ++i)
        ref[i] = (uint8_t)rnd.random();

    printf("%-9s %8s %8s %6s\n", "size", "C(ms)", "sse(ms)", "x");
    for (const SadEntry &e : kSadSizes) {
        volatile uint32_t sink = 0;
        uint64_t s_s, s_u, f_s, f_u;

        svt_av1_get_time(&s_s, &s_u);
        for (int i = 0; i < iters; ++i)
            sink += e.c(src, stride, ref, stride);
        svt_av1_get_time(&f_s, &f_u);
        const double c_ms =
            svt_av1_compute_overall_elapsed_time_ms(s_s, s_u, f_s, f_u);

        svt_av1_get_time(&s_s, &s_u);
        for (int i = 0; i < iters; ++i)
            sink += e.sse4_1(src, stride, ref, stride);
        svt_av1_get_time(&f_s, &f_u);
        const double o_ms =
            svt_av1_compute_overall_elapsed_time_ms(s_s, s_u, f_s, f_u);

        char name[16];
        snprintf(name, sizeof(name), "%dx%d", e.w, e.h);
        printf("%-9s %8.2f %8.2f %5.2fx\n",
               name,
               c_ms,
               o_ms,
               o_ms > 0 ? c_ms / o_ms : 0);
        (void)sink;
    }
}
}  // namespace
