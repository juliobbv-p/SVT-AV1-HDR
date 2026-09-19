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
 * @file intrapred_test.cc
 *
 * @brief Unit test for intra {h, v}_pred, dc_pred, smooth_{h, v}_pred :
 * - av1_highbd_{dc, h, v, smooth_h, smooth_v}_predictor_wxh_{sse2, avx2, ssse3}
 * - av1_{dc, h, v, smooth_h, smooth_v}_predictor_wxh_{sse2, avx2, ssse3}
 *
 * @author Cidana-Wenyao
 *
 ******************************************************************************/

#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <tuple>
#include "common_dsp_rtcd.h"
#include "random.h"
#include "svt_malloc.h"

namespace {
using std::make_tuple;
using std::tuple;
using svt_av1_test_tool::SVTRandom;

constexpr int count_test_block = 1000;

using INTRAPRED_HBD = void (*)(uint16_t *dst, ptrdiff_t stride,
                               const uint16_t *above, const uint16_t *left,
                               int bd);
using INTRAPRED_LBD = void (*)(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left);

using LBD_PARAMS = tuple<INTRAPRED_LBD, INTRAPRED_LBD, int, int>;
using HBD_PARAMS = tuple<INTRAPRED_HBD, INTRAPRED_HBD, int, int>;

/**
 * @brief Unit test for intra prediction:
 * - av1_highbd_{dc, h, v, smooth_h, smooth_v}_predictor_wxh_{sse2, avx2, ssse3}
 * - av1_{dc, h, v, smooth_h, smooth_v}_predictor_wxh_{sse2, avx2, ssse3}
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same data and check test output and reference output.
 * Define a templete class to handle the common process, and
 * declare sub class to handle different bitdepth and function types.
 *
 * Expect result:
 * Output from assemble functions should be the same with output from c.
 *
 * Test coverage:
 * Test cases:
 * Neighbor pixel buffer: Fill with random values
 * TxSize: all the TxSize.
 * BitDepth: 8bit and 10bit
 *
 */
template <typename FuncType, typename Sample, typename TupleType, int bd>
class AV1IntraPredTest : public ::testing::TestWithParam<TupleType> {
  protected:
    void prepare_data(SVTRandom &rnd, int cnt) {
        if (cnt == 0) {
            constexpr auto mask = (1 << bd) - 1;
            std::fill_n(above_row_ - 1, bw_ * 2 + 2, mask);
            std::fill_n(left_col_.begin(), bh_, mask);
        } else {
            std::generate_n(above_row_ - 1, bw_ * 2 + 2, [&rnd]() {
                return (Sample)rnd.random();
            });
            std::generate_n(left_col_.begin(), bh_, [&rnd]() {
                return (Sample)rnd.random();
            });
        }
        dst_tst_.fill(0);
        dst_ref_.fill(0);
    }

  public:
    void *operator new(size_t size) {
        if (void *ptr = svt_aom_memalign(alignof(AV1IntraPredTest), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

    void RunTest() {
        SVTRandom rnd{0, (1 << bd) - 1};
        for (int i = 0; i < count_test_block; ++i) {
            // prepare the neighbor pixels
            prepare_data(rnd, i);

            Predict();

            for (int y = 0; y < bh_; y++) {
                for (int x = 0; x < bw_; x++) {
                    ASSERT_EQ(dst_ref_[x + y * stride_],
                              dst_tst_[x + y * stride_])
                        << " Failed on loop " << i << " location: x = " << x
                        << " y = " << y;
                }
            }
        }
    }

  protected:
    virtual void Predict() = 0;

    alignas(32) std::array<Sample, 2 * 64> left_col_{};
    alignas(32) std::array<Sample, 3 * 64 * 64> dst_tst_{};
    alignas(32) std::array<Sample, 3 * 64 * 64> dst_ref_{};
    alignas(32) std::array<Sample, 3 * 64> above_row_data_{};
    Sample *const above_row_{above_row_data_.data() + 16};

    static constexpr ptrdiff_t stride_{64 * 3};
    const FuncType tst_func_{std::get<0>(this->GetParam())};
    const FuncType ref_func_{std::get<1>(this->GetParam())};
    const int bw_{std::get<2>(this->GetParam())};  // block width
    const int bh_{std::get<3>(this->GetParam())};  // block height
};

/** setup_test_env is implemented in test/TestEnv.c */
extern "C" void setup_test_env();

class LowbdIntraPredTest
    : public AV1IntraPredTest<INTRAPRED_LBD, uint8_t, LBD_PARAMS, 8> {
  protected:
    void SetUp() override {
        setup_test_env();
    }
    void Predict() override {
        ref_func_(dst_ref_.data(), stride_, above_row_, left_col_.data());
        tst_func_(dst_tst_.data(), stride_, above_row_, left_col_.data());
    }
};

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
class HighbdIntraPredTest
    : public AV1IntraPredTest<INTRAPRED_HBD, uint16_t, HBD_PARAMS, 10> {
  protected:
    void SetUp() override {
        svt_aom_setup_common_rtcd_internal(svt_aom_get_cpu_flags_to_use());
    }

    void Predict() override {
        ref_func_(dst_ref_.data(), stride_, above_row_, left_col_.data(), 10);
        tst_func_(dst_tst_.data(), stride_, above_row_, left_col_.data(), 10);
    }
};

TEST_P(HighbdIntraPredTest, match_test) {
    RunTest();
}
// -----------------------------------------------------------------------------
// High Bit Depth Tests
#define hbd_entry(type, width, height, opt)                                   \
    make_tuple(&svt_aom_highbd_##type##_predictor_##width##x##height##_##opt, \
               &svt_aom_highbd_##type##_predictor_##width##x##height##_c,     \
               width,                                                         \
               height)

#ifdef ARCH_X86_64
const HBD_PARAMS HighbdIntraPredTestVectorAsmSSE2[] = {
    hbd_entry(dc_128, 4, 16, sse2),  hbd_entry(dc_128, 4, 4, sse2),
    hbd_entry(dc_128, 4, 8, sse2),   hbd_entry(dc_128, 8, 16, sse2),
    hbd_entry(dc_128, 8, 32, sse2),  hbd_entry(dc_128, 8, 4, sse2),
    hbd_entry(dc_128, 8, 8, sse2),   hbd_entry(dc_left, 4, 16, sse2),
    hbd_entry(dc_left, 4, 4, sse2),  hbd_entry(dc_left, 4, 8, sse2),
    hbd_entry(dc_left, 8, 16, sse2), hbd_entry(dc_left, 8, 32, sse2),
    hbd_entry(dc_left, 8, 4, sse2),  hbd_entry(dc_left, 8, 8, sse2),
    hbd_entry(dc, 4, 16, sse2),      hbd_entry(dc, 4, 4, sse2),
    hbd_entry(dc, 4, 8, sse2),       hbd_entry(dc, 8, 16, sse2),
    hbd_entry(dc, 8, 32, sse2),      hbd_entry(dc, 8, 4, sse2),
    hbd_entry(dc, 8, 8, sse2),       hbd_entry(dc_top, 4, 16, sse2),
    hbd_entry(dc_top, 4, 4, sse2),   hbd_entry(dc_top, 4, 8, sse2),
    hbd_entry(dc_top, 8, 16, sse2),  hbd_entry(dc_top, 8, 4, sse2),
    hbd_entry(dc_top, 8, 8, sse2),   hbd_entry(h, 16, 16, sse2),
    hbd_entry(h, 16, 32, sse2),      hbd_entry(h, 16, 8, sse2),
    hbd_entry(h, 32, 16, sse2),      hbd_entry(h, 32, 32, sse2),
    hbd_entry(h, 4, 16, sse2),       hbd_entry(h, 4, 4, sse2),
    hbd_entry(h, 4, 8, sse2),        hbd_entry(h, 8, 16, sse2),
    hbd_entry(h, 8, 32, sse2),       hbd_entry(h, 8, 4, sse2),
    hbd_entry(h, 8, 8, sse2),        hbd_entry(v, 4, 16, sse2),
    hbd_entry(v, 4, 4, sse2),        hbd_entry(v, 4, 8, sse2),
    hbd_entry(v, 8, 16, sse2),       hbd_entry(v, 8, 32, sse2),
    hbd_entry(v, 8, 4, sse2),        hbd_entry(v, 8, 8, sse2),

};

const HBD_PARAMS HighbdIntraPredTestVectorAsmAVX2[] = {
    hbd_entry(dc_128, 16, 16, avx2),   hbd_entry(dc_128, 16, 32, avx2),
    hbd_entry(dc_128, 16, 4, avx2),    hbd_entry(dc_128, 16, 64, avx2),
    hbd_entry(dc_128, 16, 8, avx2),    hbd_entry(dc_128, 32, 16, avx2),
    hbd_entry(dc_128, 32, 32, avx2),   hbd_entry(dc_128, 32, 64, avx2),
    hbd_entry(dc_128, 32, 8, avx2),    hbd_entry(dc_128, 64, 16, avx2),
    hbd_entry(dc_128, 64, 32, avx2),   hbd_entry(dc_128, 64, 64, avx2),
    hbd_entry(dc_left, 16, 16, avx2),  hbd_entry(dc_left, 16, 32, avx2),
    hbd_entry(dc_left, 16, 4, avx2),   hbd_entry(dc_left, 16, 64, avx2),
    hbd_entry(dc_left, 16, 8, avx2),   hbd_entry(dc_left, 32, 16, avx2),
    hbd_entry(dc_left, 32, 32, avx2),  hbd_entry(dc_left, 32, 64, avx2),
    hbd_entry(dc_left, 32, 8, avx2),   hbd_entry(dc_left, 64, 16, avx2),
    hbd_entry(dc_left, 64, 32, avx2),  hbd_entry(dc_left, 64, 64, avx2),
    hbd_entry(dc, 16, 16, avx2),       hbd_entry(dc, 16, 32, avx2),
    hbd_entry(dc, 16, 4, avx2),        hbd_entry(dc, 16, 64, avx2),
    hbd_entry(dc, 16, 8, avx2),        hbd_entry(dc, 32, 16, avx2),
    hbd_entry(dc, 32, 32, avx2),       hbd_entry(dc, 32, 64, avx2),
    hbd_entry(dc, 32, 8, avx2),        hbd_entry(dc, 64, 16, avx2),
    hbd_entry(dc, 64, 32, avx2),       hbd_entry(dc, 64, 64, avx2),
    hbd_entry(dc_top, 16, 16, avx2),   hbd_entry(dc_top, 16, 32, avx2),
    hbd_entry(dc_top, 16, 4, avx2),    hbd_entry(dc_top, 16, 64, avx2),
    hbd_entry(dc_top, 16, 8, avx2),    hbd_entry(dc_top, 32, 16, avx2),
    hbd_entry(dc_top, 32, 32, avx2),   hbd_entry(dc_top, 32, 64, avx2),
    hbd_entry(dc_top, 32, 8, avx2),    hbd_entry(dc_top, 64, 16, avx2),
    hbd_entry(dc_top, 64, 32, avx2),   hbd_entry(dc_top, 64, 64, avx2),
    hbd_entry(h, 16, 4, avx2),         hbd_entry(h, 16, 64, avx2),
    hbd_entry(h, 32, 64, avx2),        hbd_entry(h, 32, 8, avx2),
    hbd_entry(h, 64, 16, avx2),        hbd_entry(h, 64, 32, avx2),
    hbd_entry(h, 64, 64, avx2),        hbd_entry(smooth_h, 16, 16, avx2),
    hbd_entry(smooth_h, 16, 32, avx2), hbd_entry(smooth_h, 16, 4, avx2),
    hbd_entry(smooth_h, 16, 64, avx2), hbd_entry(smooth_h, 16, 8, avx2),
    hbd_entry(smooth_h, 32, 16, avx2), hbd_entry(smooth_h, 32, 32, avx2),
    hbd_entry(smooth_h, 32, 64, avx2), hbd_entry(smooth_h, 32, 8, avx2),
    hbd_entry(smooth_h, 64, 16, avx2), hbd_entry(smooth_h, 64, 32, avx2),
    hbd_entry(smooth_h, 64, 64, avx2), hbd_entry(smooth_h, 8, 16, avx2),
    hbd_entry(smooth_h, 8, 32, avx2),  hbd_entry(smooth_h, 8, 4, avx2),
    hbd_entry(smooth_h, 8, 8, avx2),   hbd_entry(smooth, 16, 16, avx2),
    hbd_entry(smooth, 16, 32, avx2),   hbd_entry(smooth, 16, 4, avx2),
    hbd_entry(smooth, 16, 64, avx2),   hbd_entry(smooth, 16, 8, avx2),
    hbd_entry(smooth, 32, 16, avx2),   hbd_entry(smooth, 32, 32, avx2),
    hbd_entry(smooth, 32, 64, avx2),   hbd_entry(smooth, 32, 8, avx2),
    hbd_entry(smooth, 64, 16, avx2),   hbd_entry(smooth, 64, 32, avx2),
    hbd_entry(smooth, 64, 64, avx2),   hbd_entry(smooth, 8, 16, avx2),
    hbd_entry(smooth, 8, 32, avx2),    hbd_entry(smooth, 8, 4, avx2),
    hbd_entry(smooth, 8, 8, avx2),     hbd_entry(smooth_v, 16, 16, avx2),
    hbd_entry(smooth_v, 16, 32, avx2), hbd_entry(smooth_v, 16, 4, avx2),
    hbd_entry(smooth_v, 16, 64, avx2), hbd_entry(smooth_v, 16, 8, avx2),
    hbd_entry(smooth_v, 32, 16, avx2), hbd_entry(smooth_v, 32, 32, avx2),
    hbd_entry(smooth_v, 32, 64, avx2), hbd_entry(smooth_v, 32, 8, avx2),
    hbd_entry(smooth_v, 64, 16, avx2), hbd_entry(smooth_v, 64, 32, avx2),
    hbd_entry(smooth_v, 64, 64, avx2), hbd_entry(smooth_v, 8, 16, avx2),
    hbd_entry(smooth_v, 8, 32, avx2),  hbd_entry(smooth_v, 8, 4, avx2),
    hbd_entry(smooth_v, 8, 8, avx2),   hbd_entry(v, 16, 16, avx2),
    hbd_entry(v, 16, 32, avx2),        hbd_entry(v, 16, 4, avx2),
    hbd_entry(v, 16, 64, avx2),        hbd_entry(v, 16, 8, avx2),
    hbd_entry(v, 32, 16, avx2),        hbd_entry(v, 32, 32, avx2),
    hbd_entry(v, 32, 64, avx2),        hbd_entry(v, 32, 8, avx2),
    hbd_entry(v, 64, 16, avx2),        hbd_entry(v, 64, 32, avx2),
    hbd_entry(v, 64, 64, avx2),        hbd_entry(paeth, 16, 4, avx2),
    hbd_entry(paeth, 16, 8, avx2),     hbd_entry(paeth, 16, 16, avx2),
    hbd_entry(paeth, 16, 32, avx2),    hbd_entry(paeth, 16, 64, avx2),
    hbd_entry(paeth, 32, 8, avx2),     hbd_entry(paeth, 32, 16, avx2),
    hbd_entry(paeth, 32, 32, avx2),    hbd_entry(paeth, 32, 64, avx2),
    hbd_entry(paeth, 64, 16, avx2),    hbd_entry(paeth, 64, 32, avx2),
    hbd_entry(paeth, 64, 64, avx2),    hbd_entry(paeth, 8, 4, avx2),
    hbd_entry(paeth, 8, 8, avx2),      hbd_entry(paeth, 8, 16, avx2),
    hbd_entry(paeth, 8, 32, avx2),     hbd_entry(paeth, 4, 4, avx2),
    hbd_entry(paeth, 4, 8, avx2),      hbd_entry(paeth, 4, 16, avx2),
};

const HBD_PARAMS HighbdIntraPredTestVectorAsmSSE4_1[] = {
    hbd_entry(dc_128, 16, 4, sse4_1),
    hbd_entry(dc_128, 16, 8, sse4_1),
    hbd_entry(dc_128, 16, 16, sse4_1),
    hbd_entry(dc_128, 16, 32, sse4_1),
    hbd_entry(dc_128, 16, 64, sse4_1),
    hbd_entry(dc_128, 32, 8, sse4_1),
    hbd_entry(dc_128, 32, 16, sse4_1),
    hbd_entry(dc_128, 32, 32, sse4_1),
    hbd_entry(dc_128, 32, 64, sse4_1),
    hbd_entry(dc_128, 64, 16, sse4_1),
    hbd_entry(dc_128, 64, 32, sse4_1),
    hbd_entry(dc_128, 64, 64, sse4_1),
    hbd_entry(dc_left, 16, 4, sse4_1),
    hbd_entry(dc_left, 16, 8, sse4_1),
    hbd_entry(dc_left, 16, 16, sse4_1),
    hbd_entry(dc_left, 16, 32, sse4_1),
    hbd_entry(dc_left, 16, 64, sse4_1),
    hbd_entry(dc_left, 32, 8, sse4_1),
    hbd_entry(dc_left, 32, 16, sse4_1),
    hbd_entry(dc_left, 32, 32, sse4_1),
    hbd_entry(dc_left, 32, 64, sse4_1),
    hbd_entry(dc_left, 64, 16, sse4_1),
    hbd_entry(dc_left, 64, 32, sse4_1),
    hbd_entry(dc_left, 64, 64, sse4_1),
    hbd_entry(dc_top, 16, 4, sse4_1),
    hbd_entry(dc_top, 16, 8, sse4_1),
    hbd_entry(dc_top, 16, 16, sse4_1),
    hbd_entry(dc_top, 16, 32, sse4_1),
    hbd_entry(dc_top, 16, 64, sse4_1),
    hbd_entry(dc_top, 32, 8, sse4_1),
    hbd_entry(dc_top, 32, 16, sse4_1),
    hbd_entry(dc_top, 32, 32, sse4_1),
    hbd_entry(dc_top, 32, 64, sse4_1),
    hbd_entry(dc_top, 64, 16, sse4_1),
    hbd_entry(dc_top, 64, 32, sse4_1),
    hbd_entry(dc_top, 64, 64, sse4_1),
    hbd_entry(dc, 16, 4, sse4_1),
    hbd_entry(dc, 16, 8, sse4_1),
    hbd_entry(dc, 16, 16, sse4_1),
    hbd_entry(dc, 16, 32, sse4_1),
    hbd_entry(dc, 16, 64, sse4_1),
    hbd_entry(dc, 32, 8, sse4_1),
    hbd_entry(dc, 32, 16, sse4_1),
    hbd_entry(dc, 32, 32, sse4_1),
    hbd_entry(dc, 32, 64, sse4_1),
    hbd_entry(dc, 64, 16, sse4_1),
    hbd_entry(dc, 64, 32, sse4_1),
    hbd_entry(dc, 64, 64, sse4_1),
    hbd_entry(v, 16, 4, sse4_1),
    hbd_entry(v, 16, 8, sse4_1),
    hbd_entry(v, 16, 16, sse4_1),
    hbd_entry(v, 16, 32, sse4_1),
    hbd_entry(v, 16, 64, sse4_1),
    hbd_entry(v, 32, 8, sse4_1),
    hbd_entry(v, 32, 16, sse4_1),
    hbd_entry(v, 32, 32, sse4_1),
    hbd_entry(v, 32, 64, sse4_1),
    hbd_entry(v, 64, 16, sse4_1),
    hbd_entry(v, 64, 32, sse4_1),
    hbd_entry(v, 64, 64, sse4_1),
    hbd_entry(h, 16, 4, sse4_1),
    hbd_entry(h, 16, 64, sse4_1),
    hbd_entry(h, 32, 64, sse4_1),
    hbd_entry(h, 32, 8, sse4_1),
    hbd_entry(h, 64, 16, sse4_1),
    hbd_entry(h, 64, 32, sse4_1),
    hbd_entry(h, 64, 64, sse4_1),
    hbd_entry(smooth_h, 16, 16, sse4_1),
    hbd_entry(smooth_h, 16, 32, sse4_1),
    hbd_entry(smooth_h, 16, 4, sse4_1),
    hbd_entry(smooth_h, 16, 64, sse4_1),
    hbd_entry(smooth_h, 16, 8, sse4_1),
    hbd_entry(smooth_h, 32, 16, sse4_1),
    hbd_entry(smooth_h, 32, 32, sse4_1),
    hbd_entry(smooth_h, 32, 64, sse4_1),
    hbd_entry(smooth_h, 32, 8, sse4_1),
    hbd_entry(smooth_h, 64, 16, sse4_1),
    hbd_entry(smooth_h, 64, 32, sse4_1),
    hbd_entry(smooth_h, 64, 64, sse4_1),
    hbd_entry(smooth_h, 8, 16, sse4_1),
    hbd_entry(smooth_h, 8, 32, sse4_1),
    hbd_entry(smooth_h, 8, 4, sse4_1),
    hbd_entry(smooth_h, 8, 8, sse4_1),
    hbd_entry(smooth, 16, 16, sse4_1),
    hbd_entry(smooth, 16, 32, sse4_1),
    hbd_entry(smooth, 16, 4, sse4_1),
    hbd_entry(smooth, 16, 64, sse4_1),
    hbd_entry(smooth, 16, 8, sse4_1),
    hbd_entry(smooth, 32, 16, sse4_1),
    hbd_entry(smooth, 32, 32, sse4_1),
    hbd_entry(smooth, 32, 64, sse4_1),
    hbd_entry(smooth, 32, 8, sse4_1),
    hbd_entry(smooth, 64, 16, sse4_1),
    hbd_entry(smooth, 64, 32, sse4_1),
    hbd_entry(smooth, 64, 64, sse4_1),
    hbd_entry(smooth, 8, 16, sse4_1),
    hbd_entry(smooth, 8, 32, sse4_1),
    hbd_entry(smooth, 8, 4, sse4_1),
    hbd_entry(smooth, 8, 8, sse4_1),
    hbd_entry(smooth_v, 16, 16, sse4_1),
    hbd_entry(smooth_v, 16, 32, sse4_1),
    hbd_entry(smooth_v, 16, 4, sse4_1),
    hbd_entry(smooth_v, 16, 64, sse4_1),
    hbd_entry(smooth_v, 16, 8, sse4_1),
    hbd_entry(smooth_v, 32, 16, sse4_1),
    hbd_entry(smooth_v, 32, 32, sse4_1),
    hbd_entry(smooth_v, 32, 64, sse4_1),
    hbd_entry(smooth_v, 32, 8, sse4_1),
    hbd_entry(smooth_v, 64, 16, sse4_1),
    hbd_entry(smooth_v, 64, 32, sse4_1),
    hbd_entry(smooth_v, 64, 64, sse4_1),
    hbd_entry(smooth_v, 8, 16, sse4_1),
    hbd_entry(smooth_v, 8, 32, sse4_1),
    hbd_entry(smooth_v, 8, 4, sse4_1),
    hbd_entry(smooth_v, 8, 8, sse4_1),
    hbd_entry(paeth, 16, 4, sse4_1),
    hbd_entry(paeth, 16, 8, sse4_1),
    hbd_entry(paeth, 16, 16, sse4_1),
    hbd_entry(paeth, 16, 32, sse4_1),
    hbd_entry(paeth, 16, 64, sse4_1),
    hbd_entry(paeth, 32, 8, sse4_1),
    hbd_entry(paeth, 32, 16, sse4_1),
    hbd_entry(paeth, 32, 32, sse4_1),
    hbd_entry(paeth, 32, 64, sse4_1),
    hbd_entry(paeth, 64, 16, sse4_1),
    hbd_entry(paeth, 64, 32, sse4_1),
    hbd_entry(paeth, 64, 64, sse4_1),
    hbd_entry(paeth, 8, 4, sse4_1),
    hbd_entry(paeth, 8, 8, sse4_1),
    hbd_entry(paeth, 8, 16, sse4_1),
    hbd_entry(paeth, 8, 32, sse4_1),
    hbd_entry(paeth, 4, 4, sse4_1),
    hbd_entry(paeth, 4, 8, sse4_1),
    hbd_entry(paeth, 4, 16, sse4_1),

};

const HBD_PARAMS HighbdIntraPredTestVectorAsmSSSE3[] = {
    hbd_entry(smooth_h, 4, 16, ssse3),
    hbd_entry(smooth_h, 4, 4, ssse3),
    hbd_entry(smooth_h, 4, 8, ssse3),
    hbd_entry(smooth, 4, 16, ssse3),
    hbd_entry(smooth, 4, 4, ssse3),
    hbd_entry(smooth, 4, 8, ssse3),
    hbd_entry(smooth_v, 4, 16, ssse3),
    hbd_entry(smooth_v, 4, 4, ssse3),
    hbd_entry(smooth_v, 4, 8, ssse3),
};

INSTANTIATE_TEST_SUITE_P(SSE2, HighbdIntraPredTest,
                         ::testing::ValuesIn(HighbdIntraPredTestVectorAsmSSE2));

INSTANTIATE_TEST_SUITE_P(AVX2, HighbdIntraPredTest,
                         ::testing::ValuesIn(HighbdIntraPredTestVectorAsmAVX2));
INSTANTIATE_TEST_SUITE_P(SSE4_1, HighbdIntraPredTest,
                         ::testing::ValuesIn(HighbdIntraPredTestVectorAsmSSE4_1));

INSTANTIATE_TEST_SUITE_P(
    SSSE3, HighbdIntraPredTest,
    ::testing::ValuesIn(HighbdIntraPredTestVectorAsmSSSE3));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
const HBD_PARAMS HighbdIntraPredTestVectorAsmNEON[] = {
    hbd_entry(smooth_v, 4, 4, neon),   hbd_entry(smooth_v, 4, 8, neon),
    hbd_entry(smooth_v, 4, 16, neon),  hbd_entry(smooth_v, 8, 4, neon),
    hbd_entry(smooth_v, 8, 8, neon),   hbd_entry(smooth_v, 8, 16, neon),
    hbd_entry(smooth_v, 8, 32, neon),  hbd_entry(smooth_v, 16, 4, neon),
    hbd_entry(smooth_v, 16, 8, neon),  hbd_entry(smooth_v, 16, 16, neon),
    hbd_entry(smooth_v, 16, 32, neon), hbd_entry(smooth_v, 16, 64, neon),
    hbd_entry(smooth_v, 32, 8, neon),  hbd_entry(smooth_v, 32, 16, neon),
    hbd_entry(smooth_v, 32, 32, neon), hbd_entry(smooth_v, 32, 64, neon),
    hbd_entry(smooth_v, 64, 16, neon), hbd_entry(smooth_v, 64, 32, neon),
    hbd_entry(smooth_v, 64, 64, neon), hbd_entry(smooth_h, 4, 4, neon),
    hbd_entry(smooth_h, 4, 8, neon),   hbd_entry(smooth_h, 4, 16, neon),
    hbd_entry(smooth_h, 8, 4, neon),   hbd_entry(smooth_h, 8, 8, neon),
    hbd_entry(smooth_h, 8, 16, neon),  hbd_entry(smooth_h, 8, 32, neon),
    hbd_entry(smooth_h, 16, 4, neon),  hbd_entry(smooth_h, 16, 8, neon),
    hbd_entry(smooth_h, 16, 16, neon), hbd_entry(smooth_h, 16, 32, neon),
    hbd_entry(smooth_h, 16, 64, neon), hbd_entry(smooth_h, 32, 8, neon),
    hbd_entry(smooth_h, 32, 16, neon), hbd_entry(smooth_h, 32, 32, neon),
    hbd_entry(smooth_h, 32, 64, neon), hbd_entry(smooth_h, 64, 16, neon),
    hbd_entry(smooth_h, 64, 32, neon), hbd_entry(smooth_h, 64, 64, neon),
    hbd_entry(smooth, 4, 8, neon),     hbd_entry(smooth, 4, 16, neon),
    hbd_entry(smooth, 8, 4, neon),     hbd_entry(smooth, 8, 8, neon),
    hbd_entry(smooth, 8, 16, neon),    hbd_entry(smooth, 8, 32, neon),
    hbd_entry(smooth, 16, 4, neon),    hbd_entry(smooth, 16, 8, neon),
    hbd_entry(smooth, 16, 16, neon),   hbd_entry(smooth, 16, 32, neon),
    hbd_entry(smooth, 16, 64, neon),   hbd_entry(smooth, 32, 8, neon),
    hbd_entry(smooth, 32, 16, neon),   hbd_entry(smooth, 32, 32, neon),
    hbd_entry(smooth, 32, 64, neon),   hbd_entry(smooth, 64, 16, neon),
    hbd_entry(smooth, 64, 32, neon),   hbd_entry(smooth, 64, 64, neon),
    hbd_entry(v, 4, 8, neon),          hbd_entry(v, 4, 16, neon),
    hbd_entry(v, 8, 4, neon),          hbd_entry(v, 8, 8, neon),
    hbd_entry(v, 8, 16, neon),         hbd_entry(v, 8, 32, neon),
    hbd_entry(v, 16, 4, neon),         hbd_entry(v, 16, 8, neon),
    hbd_entry(v, 16, 16, neon),        hbd_entry(v, 16, 32, neon),
    hbd_entry(v, 16, 64, neon),        hbd_entry(v, 32, 8, neon),
    hbd_entry(v, 32, 16, neon),        hbd_entry(v, 32, 32, neon),
    hbd_entry(v, 32, 64, neon),        hbd_entry(v, 64, 16, neon),
    hbd_entry(v, 64, 32, neon),        hbd_entry(v, 64, 64, neon),
    hbd_entry(h, 4, 8, neon),          hbd_entry(h, 4, 16, neon),
    hbd_entry(h, 8, 4, neon),          hbd_entry(h, 8, 8, neon),
    hbd_entry(h, 8, 16, neon),         hbd_entry(h, 8, 32, neon),
    hbd_entry(h, 16, 4, neon),         hbd_entry(h, 16, 8, neon),
    hbd_entry(h, 16, 16, neon),        hbd_entry(h, 16, 32, neon),
    hbd_entry(h, 16, 64, neon),        hbd_entry(h, 32, 8, neon),
    hbd_entry(h, 32, 16, neon),        hbd_entry(h, 32, 32, neon),
    hbd_entry(h, 32, 64, neon),        hbd_entry(h, 64, 16, neon),
    hbd_entry(h, 64, 32, neon),        hbd_entry(h, 64, 64, neon),
    hbd_entry(paeth, 4, 8, neon),      hbd_entry(paeth, 4, 16, neon),
    hbd_entry(paeth, 8, 4, neon),      hbd_entry(paeth, 8, 8, neon),
    hbd_entry(paeth, 8, 16, neon),     hbd_entry(paeth, 8, 32, neon),
    hbd_entry(paeth, 16, 4, neon),     hbd_entry(paeth, 16, 8, neon),
    hbd_entry(paeth, 16, 16, neon),    hbd_entry(paeth, 16, 32, neon),
    hbd_entry(paeth, 16, 64, neon),    hbd_entry(paeth, 32, 8, neon),
    hbd_entry(paeth, 32, 16, neon),    hbd_entry(paeth, 32, 32, neon),
    hbd_entry(paeth, 32, 64, neon),    hbd_entry(paeth, 64, 16, neon),
    hbd_entry(paeth, 64, 32, neon),    hbd_entry(paeth, 64, 64, neon),
    hbd_entry(dc, 4, 8, neon),         hbd_entry(dc, 4, 16, neon),
    hbd_entry(dc, 8, 4, neon),         hbd_entry(dc, 8, 8, neon),
    hbd_entry(dc, 8, 16, neon),        hbd_entry(dc, 8, 32, neon),
    hbd_entry(dc, 16, 4, neon),        hbd_entry(dc, 16, 8, neon),
    hbd_entry(dc, 16, 16, neon),       hbd_entry(dc, 16, 32, neon),
    hbd_entry(dc, 16, 64, neon),       hbd_entry(dc, 32, 8, neon),
    hbd_entry(dc, 32, 16, neon),       hbd_entry(dc, 32, 32, neon),
    hbd_entry(dc, 32, 64, neon),       hbd_entry(dc, 64, 16, neon),
    hbd_entry(dc, 64, 32, neon),       hbd_entry(dc, 64, 64, neon),
    hbd_entry(dc_left, 4, 8, neon),    hbd_entry(dc_left, 4, 16, neon),
    hbd_entry(dc_left, 8, 4, neon),    hbd_entry(dc_left, 8, 8, neon),
    hbd_entry(dc_left, 8, 16, neon),   hbd_entry(dc_left, 8, 32, neon),
    hbd_entry(dc_left, 16, 4, neon),   hbd_entry(dc_left, 16, 8, neon),
    hbd_entry(dc_left, 16, 16, neon),  hbd_entry(dc_left, 16, 32, neon),
    hbd_entry(dc_left, 16, 64, neon),  hbd_entry(dc_left, 32, 8, neon),
    hbd_entry(dc_left, 32, 16, neon),  hbd_entry(dc_left, 32, 32, neon),
    hbd_entry(dc_left, 32, 64, neon),  hbd_entry(dc_left, 64, 16, neon),
    hbd_entry(dc_left, 64, 32, neon),  hbd_entry(dc_left, 64, 64, neon),
    hbd_entry(dc_top, 4, 8, neon),     hbd_entry(dc_top, 4, 16, neon),
    hbd_entry(dc_top, 8, 4, neon),     hbd_entry(dc_top, 8, 8, neon),
    hbd_entry(dc_top, 8, 16, neon),    hbd_entry(dc_top, 8, 32, neon),
    hbd_entry(dc_top, 16, 4, neon),    hbd_entry(dc_top, 16, 8, neon),
    hbd_entry(dc_top, 16, 16, neon),   hbd_entry(dc_top, 16, 32, neon),
    hbd_entry(dc_top, 16, 64, neon),   hbd_entry(dc_top, 32, 8, neon),
    hbd_entry(dc_top, 32, 16, neon),   hbd_entry(dc_top, 32, 32, neon),
    hbd_entry(dc_top, 32, 64, neon),   hbd_entry(dc_top, 64, 16, neon),
    hbd_entry(dc_top, 64, 32, neon),   hbd_entry(dc_top, 64, 64, neon),
    hbd_entry(dc_128, 4, 8, neon),     hbd_entry(dc_128, 4, 16, neon),
    hbd_entry(dc_128, 8, 4, neon),     hbd_entry(dc_128, 8, 8, neon),
    hbd_entry(dc_128, 8, 16, neon),    hbd_entry(dc_128, 8, 32, neon),
    hbd_entry(dc_128, 16, 4, neon),    hbd_entry(dc_128, 16, 8, neon),
    hbd_entry(dc_128, 16, 16, neon),   hbd_entry(dc_128, 16, 32, neon),
    hbd_entry(dc_128, 16, 64, neon),   hbd_entry(dc_128, 32, 8, neon),
    hbd_entry(dc_128, 32, 16, neon),   hbd_entry(dc_128, 32, 32, neon),
    hbd_entry(dc_128, 32, 64, neon),   hbd_entry(dc_128, 64, 16, neon),
    hbd_entry(dc_128, 64, 32, neon),   hbd_entry(dc_128, 64, 64, neon)};

INSTANTIATE_TEST_SUITE_P(NEON, HighbdIntraPredTest,
                         ::testing::ValuesIn(HighbdIntraPredTestVectorAsmNEON));

#endif  // ARCH_AARCH64

#endif

TEST_P(LowbdIntraPredTest, match_test) {
    RunTest();
}

// ---------------------------------------------------------------------------
// Low Bit Depth Tests
#define lbd_entry(type, width, height, opt)                            \
    LBD_PARAMS(&svt_aom_##type##_predictor_##width##x##height##_##opt, \
               &svt_aom_##type##_predictor_##width##x##height##_c,     \
               width,                                                  \
               height)

#ifdef ARCH_X86_64
const LBD_PARAMS LowbdIntraPredTestVectorAsmSSE2[] = {
    lbd_entry(dc, 4, 4, sse2),        lbd_entry(dc, 8, 8, sse2),
    lbd_entry(dc, 16, 16, sse2),      lbd_entry(dc, 16, 32, sse2),
    lbd_entry(dc, 16, 4, sse2),       lbd_entry(dc, 16, 64, sse2),
    lbd_entry(dc, 16, 8, sse2),       lbd_entry(dc, 32, 8, sse2),
    lbd_entry(dc, 4, 16, sse2),       lbd_entry(dc, 4, 8, sse2),
    lbd_entry(dc, 8, 16, sse2),       lbd_entry(dc, 8, 32, sse2),
    lbd_entry(dc, 8, 4, sse2),        lbd_entry(dc_left, 4, 4, sse2),
    lbd_entry(dc_left, 8, 8, sse2),   lbd_entry(dc_left, 16, 16, sse2),
    lbd_entry(dc_left, 16, 32, sse2), lbd_entry(dc_left, 16, 4, sse2),
    lbd_entry(dc_left, 16, 64, sse2), lbd_entry(dc_left, 16, 8, sse2),
    lbd_entry(dc_left, 32, 8, sse2),  lbd_entry(dc_left, 4, 16, sse2),
    lbd_entry(dc_left, 4, 8, sse2),   lbd_entry(dc_left, 8, 16, sse2),
    lbd_entry(dc_left, 8, 32, sse2),  lbd_entry(dc_left, 8, 4, sse2),
    lbd_entry(dc_top, 4, 4, sse2),    lbd_entry(dc_top, 8, 8, sse2),
    lbd_entry(dc_top, 16, 16, sse2),  lbd_entry(dc_top, 16, 32, sse2),
    lbd_entry(dc_top, 16, 4, sse2),   lbd_entry(dc_top, 16, 64, sse2),
    lbd_entry(dc_top, 16, 8, sse2),   lbd_entry(dc_top, 32, 8, sse2),
    lbd_entry(dc_top, 4, 16, sse2),   lbd_entry(dc_top, 4, 8, sse2),
    lbd_entry(dc_top, 8, 16, sse2),   lbd_entry(dc_top, 8, 32, sse2),
    lbd_entry(dc_top, 8, 4, sse2),    lbd_entry(dc_128, 4, 4, sse2),
    lbd_entry(dc_128, 8, 8, sse2),    lbd_entry(dc_128, 16, 16, sse2),
    lbd_entry(dc_128, 16, 32, sse2),  lbd_entry(dc_128, 16, 4, sse2),
    lbd_entry(dc_128, 16, 64, sse2),  lbd_entry(dc_128, 16, 8, sse2),
    lbd_entry(dc_128, 32, 8, sse2),   lbd_entry(dc_128, 4, 16, sse2),
    lbd_entry(dc_128, 4, 8, sse2),    lbd_entry(dc_128, 8, 16, sse2),
    lbd_entry(dc_128, 8, 32, sse2),   lbd_entry(dc_128, 8, 4, sse2),
    lbd_entry(v, 4, 4, sse2),         lbd_entry(v, 8, 8, sse2),
    lbd_entry(v, 16, 16, sse2),       lbd_entry(v, 16, 32, sse2),
    lbd_entry(v, 16, 4, sse2),        lbd_entry(v, 16, 64, sse2),
    lbd_entry(v, 16, 8, sse2),        lbd_entry(v, 32, 8, sse2),
    lbd_entry(v, 4, 16, sse2),        lbd_entry(v, 4, 8, sse2),
    lbd_entry(v, 8, 16, sse2),        lbd_entry(v, 8, 32, sse2),
    lbd_entry(v, 8, 4, sse2),         lbd_entry(h, 4, 4, sse2),
    lbd_entry(h, 8, 8, sse2),         lbd_entry(h, 16, 16, sse2),
    lbd_entry(h, 64, 64, sse2),       lbd_entry(h, 16, 32, sse2),
    lbd_entry(h, 16, 4, sse2),        lbd_entry(h, 16, 64, sse2),
    lbd_entry(h, 16, 8, sse2),        lbd_entry(h, 32, 16, sse2),
    lbd_entry(h, 32, 64, sse2),       lbd_entry(h, 32, 8, sse2),
    lbd_entry(h, 4, 16, sse2),        lbd_entry(h, 4, 8, sse2),
    lbd_entry(h, 64, 16, sse2),       lbd_entry(h, 64, 32, sse2),
    lbd_entry(h, 8, 16, sse2),        lbd_entry(h, 8, 32, sse2),
    lbd_entry(h, 8, 4, sse2),
};

const LBD_PARAMS LowbdIntraPredTestVectorAsmAVX2[] = {
    lbd_entry(dc, 32, 32, avx2),      lbd_entry(dc, 64, 64, avx2),
    lbd_entry(dc, 32, 16, avx2),      lbd_entry(dc, 32, 64, avx2),
    lbd_entry(dc, 64, 16, avx2),      lbd_entry(dc, 64, 32, avx2),
    lbd_entry(dc_left, 32, 32, avx2), lbd_entry(dc_left, 64, 64, avx2),
    lbd_entry(dc_left, 32, 16, avx2), lbd_entry(dc_left, 32, 64, avx2),
    lbd_entry(dc_left, 64, 16, avx2), lbd_entry(dc_left, 64, 32, avx2),
    lbd_entry(dc_top, 32, 32, avx2),  lbd_entry(dc_top, 64, 64, avx2),
    lbd_entry(dc_top, 32, 16, avx2),  lbd_entry(dc_top, 32, 64, avx2),
    lbd_entry(dc_top, 64, 16, avx2),  lbd_entry(dc_top, 64, 32, avx2),
    lbd_entry(dc_128, 32, 32, avx2),  lbd_entry(dc_128, 64, 64, avx2),
    lbd_entry(dc_128, 32, 16, avx2),  lbd_entry(dc_128, 32, 64, avx2),
    lbd_entry(dc_128, 64, 16, avx2),  lbd_entry(dc_128, 64, 32, avx2),
    lbd_entry(v, 32, 32, avx2),       lbd_entry(v, 64, 64, avx2),
    lbd_entry(v, 32, 16, avx2),       lbd_entry(v, 32, 64, avx2),
    lbd_entry(v, 64, 16, avx2),       lbd_entry(v, 64, 32, avx2),
    lbd_entry(h, 32, 32, avx2),       lbd_entry(paeth, 16, 16, avx2),
    lbd_entry(paeth, 16, 32, avx2),   lbd_entry(paeth, 16, 64, avx2),
    lbd_entry(paeth, 16, 8, avx2),    lbd_entry(paeth, 32, 16, avx2),
    lbd_entry(paeth, 32, 32, avx2),   lbd_entry(paeth, 32, 64, avx2),
    lbd_entry(paeth, 64, 16, avx2),   lbd_entry(paeth, 64, 32, avx2),
    lbd_entry(paeth, 64, 64, avx2),
};

const LBD_PARAMS LowbdIntraPredTestVectorAsmSSE4_1[] = {
    lbd_entry(dc, 32, 16, sse4_1),
    lbd_entry(dc, 32, 32, sse4_1),
    lbd_entry(dc, 32, 64, sse4_1),
    lbd_entry(dc, 64, 16, sse4_1),
    lbd_entry(dc, 64, 32, sse4_1),
    lbd_entry(dc, 64, 64, sse4_1),
    lbd_entry(dc_128, 32, 16, sse4_1),
    lbd_entry(dc_128, 32, 32, sse4_1),
    lbd_entry(dc_128, 32, 64, sse4_1),
    lbd_entry(dc_128, 64, 16, sse4_1),
    lbd_entry(dc_128, 64, 32, sse4_1),
    lbd_entry(dc_128, 64, 64, sse4_1),
    lbd_entry(dc_left, 32, 16, sse4_1),
    lbd_entry(dc_left, 32, 32, sse4_1),
    lbd_entry(dc_left, 32, 64, sse4_1),
    lbd_entry(dc_left, 64, 16, sse4_1),
    lbd_entry(dc_left, 64, 32, sse4_1),
    lbd_entry(dc_left, 64, 64, sse4_1),
    lbd_entry(dc_top, 32, 16, sse4_1),
    lbd_entry(dc_top, 32, 32, sse4_1),
    lbd_entry(dc_top, 32, 64, sse4_1),
    lbd_entry(dc_top, 64, 16, sse4_1),
    lbd_entry(dc_top, 64, 32, sse4_1),
    lbd_entry(dc_top, 64, 64, sse4_1),
    lbd_entry(h, 32, 32, sse4_1),
    lbd_entry(v, 32, 16, sse4_1),
    lbd_entry(v, 32, 32, sse4_1),
    lbd_entry(v, 32, 64, sse4_1),
    lbd_entry(v, 64, 16, sse4_1),
    lbd_entry(v, 64, 32, sse4_1),
    lbd_entry(v, 64, 64, sse4_1)};

const LBD_PARAMS LowbdIntraPredTestVectorAsmSSSE3[] = {
    lbd_entry(smooth_h, 64, 64, ssse3), lbd_entry(smooth_h, 32, 32, ssse3),
    lbd_entry(smooth_h, 16, 16, ssse3), lbd_entry(smooth_h, 8, 8, ssse3),
    lbd_entry(smooth_h, 4, 4, ssse3),   lbd_entry(smooth_h, 16, 32, ssse3),
    lbd_entry(smooth_h, 16, 4, ssse3),  lbd_entry(smooth_h, 16, 64, ssse3),
    lbd_entry(smooth_h, 16, 8, ssse3),  lbd_entry(smooth_h, 32, 16, ssse3),
    lbd_entry(smooth_h, 32, 64, ssse3), lbd_entry(smooth_h, 32, 8, ssse3),
    lbd_entry(smooth_h, 4, 16, ssse3),  lbd_entry(smooth_h, 4, 8, ssse3),
    lbd_entry(smooth_h, 64, 16, ssse3), lbd_entry(smooth_h, 64, 32, ssse3),
    lbd_entry(smooth_h, 8, 16, ssse3),  lbd_entry(smooth_h, 8, 32, ssse3),
    lbd_entry(smooth_h, 8, 4, ssse3),   lbd_entry(smooth_v, 64, 64, ssse3),
    lbd_entry(smooth_v, 32, 32, ssse3), lbd_entry(smooth_v, 16, 16, ssse3),
    lbd_entry(smooth_v, 8, 8, ssse3),   lbd_entry(smooth_v, 4, 4, ssse3),
    lbd_entry(smooth_v, 16, 32, ssse3), lbd_entry(smooth_v, 16, 4, ssse3),
    lbd_entry(smooth_v, 16, 64, ssse3), lbd_entry(smooth_v, 16, 8, ssse3),
    lbd_entry(smooth_v, 32, 16, ssse3), lbd_entry(smooth_v, 32, 64, ssse3),
    lbd_entry(smooth_v, 32, 8, ssse3),  lbd_entry(smooth_v, 4, 16, ssse3),
    lbd_entry(smooth_v, 4, 8, ssse3),   lbd_entry(smooth_v, 64, 16, ssse3),
    lbd_entry(smooth_v, 64, 32, ssse3), lbd_entry(smooth_v, 8, 16, ssse3),
    lbd_entry(smooth_v, 8, 32, ssse3),  lbd_entry(smooth_v, 8, 4, ssse3),
    lbd_entry(smooth, 64, 64, ssse3),   lbd_entry(smooth, 32, 32, ssse3),
    lbd_entry(smooth, 16, 16, ssse3),   lbd_entry(smooth, 8, 8, ssse3),
    lbd_entry(smooth, 4, 4, ssse3),     lbd_entry(smooth, 16, 32, ssse3),
    lbd_entry(smooth, 16, 4, ssse3),    lbd_entry(smooth, 16, 64, ssse3),
    lbd_entry(smooth, 16, 8, ssse3),    lbd_entry(smooth, 32, 16, ssse3),
    lbd_entry(smooth, 32, 64, ssse3),   lbd_entry(smooth, 32, 8, ssse3),
    lbd_entry(smooth, 4, 16, ssse3),    lbd_entry(smooth, 4, 8, ssse3),
    lbd_entry(smooth, 64, 16, ssse3),   lbd_entry(smooth, 64, 32, ssse3),
    lbd_entry(smooth, 8, 16, ssse3),    lbd_entry(smooth, 8, 32, ssse3),
    lbd_entry(smooth, 8, 4, ssse3),     lbd_entry(paeth, 16, 16, ssse3),
    lbd_entry(paeth, 16, 32, ssse3),    lbd_entry(paeth, 16, 4, ssse3),
    lbd_entry(paeth, 16, 64, ssse3),    lbd_entry(paeth, 16, 8, ssse3),
    lbd_entry(paeth, 32, 16, ssse3),    lbd_entry(paeth, 32, 32, ssse3),
    lbd_entry(paeth, 32, 64, ssse3),    lbd_entry(paeth, 32, 8, ssse3),
    lbd_entry(paeth, 4, 16, ssse3),     lbd_entry(paeth, 4, 4, ssse3),
    lbd_entry(paeth, 4, 8, ssse3),      lbd_entry(paeth, 64, 16, ssse3),
    lbd_entry(paeth, 64, 32, ssse3),    lbd_entry(paeth, 64, 64, ssse3),
    lbd_entry(paeth, 8, 16, ssse3),     lbd_entry(paeth, 8, 32, ssse3),
    lbd_entry(paeth, 8, 4, ssse3),      lbd_entry(paeth, 8, 8, ssse3),
};

INSTANTIATE_TEST_SUITE_P(SSE2, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorAsmSSE2));

INSTANTIATE_TEST_SUITE_P(AVX2, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorAsmAVX2));
INSTANTIATE_TEST_SUITE_P(SSE4_1, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorAsmSSE4_1));

INSTANTIATE_TEST_SUITE_P(SSSE3, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorAsmSSSE3));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64

const LBD_PARAMS LowbdIntraPredTestVectorAsmNEON[] = {
    lbd_entry(dc, 4, 4, neon),         lbd_entry(dc, 4, 8, neon),
    lbd_entry(dc, 4, 16, neon),        lbd_entry(dc, 8, 4, neon),
    lbd_entry(dc, 8, 8, neon),         lbd_entry(dc, 8, 16, neon),
    lbd_entry(dc, 8, 32, neon),        lbd_entry(dc, 16, 4, neon),
    lbd_entry(dc, 16, 8, neon),        lbd_entry(dc, 16, 16, neon),
    lbd_entry(dc, 16, 32, neon),       lbd_entry(dc, 16, 64, neon),
    lbd_entry(dc, 32, 8, neon),        lbd_entry(dc, 32, 16, neon),
    lbd_entry(dc, 32, 32, neon),       lbd_entry(dc, 32, 64, neon),
    lbd_entry(dc, 64, 16, neon),       lbd_entry(dc, 64, 32, neon),
    lbd_entry(dc, 64, 64, neon),       lbd_entry(dc_top, 4, 4, neon),
    lbd_entry(dc_top, 4, 8, neon),     lbd_entry(dc_top, 4, 16, neon),
    lbd_entry(dc_top, 8, 4, neon),     lbd_entry(dc_top, 8, 8, neon),
    lbd_entry(dc_top, 8, 16, neon),    lbd_entry(dc_top, 8, 32, neon),
    lbd_entry(dc_top, 16, 4, neon),    lbd_entry(dc_top, 16, 8, neon),
    lbd_entry(dc_top, 16, 16, neon),   lbd_entry(dc_top, 16, 32, neon),
    lbd_entry(dc_top, 16, 64, neon),   lbd_entry(dc_top, 32, 8, neon),
    lbd_entry(dc_top, 32, 16, neon),   lbd_entry(dc_top, 32, 32, neon),
    lbd_entry(dc_top, 32, 64, neon),   lbd_entry(dc_top, 64, 16, neon),
    lbd_entry(dc_top, 64, 32, neon),   lbd_entry(dc_top, 64, 64, neon),
    lbd_entry(dc_left, 4, 4, neon),    lbd_entry(dc_left, 4, 8, neon),
    lbd_entry(dc_left, 4, 16, neon),   lbd_entry(dc_left, 8, 4, neon),
    lbd_entry(dc_left, 8, 8, neon),    lbd_entry(dc_left, 8, 16, neon),
    lbd_entry(dc_left, 8, 32, neon),   lbd_entry(dc_left, 16, 4, neon),
    lbd_entry(dc_left, 16, 8, neon),   lbd_entry(dc_left, 16, 16, neon),
    lbd_entry(dc_left, 16, 32, neon),  lbd_entry(dc_left, 16, 64, neon),
    lbd_entry(dc_left, 32, 8, neon),   lbd_entry(dc_left, 32, 16, neon),
    lbd_entry(dc_left, 32, 32, neon),  lbd_entry(dc_left, 32, 64, neon),
    lbd_entry(dc_left, 64, 16, neon),  lbd_entry(dc_left, 64, 32, neon),
    lbd_entry(dc_left, 64, 64, neon),  lbd_entry(dc_128, 4, 4, neon),
    lbd_entry(dc_128, 4, 8, neon),     lbd_entry(dc_128, 4, 16, neon),
    lbd_entry(dc_128, 8, 4, neon),     lbd_entry(dc_128, 8, 8, neon),
    lbd_entry(dc_128, 8, 16, neon),    lbd_entry(dc_128, 8, 32, neon),
    lbd_entry(dc_128, 16, 4, neon),    lbd_entry(dc_128, 16, 8, neon),
    lbd_entry(dc_128, 16, 16, neon),   lbd_entry(dc_128, 16, 32, neon),
    lbd_entry(dc_128, 16, 64, neon),   lbd_entry(dc_128, 32, 8, neon),
    lbd_entry(dc_128, 32, 16, neon),   lbd_entry(dc_128, 32, 32, neon),
    lbd_entry(dc_128, 32, 64, neon),   lbd_entry(dc_128, 64, 16, neon),
    lbd_entry(dc_128, 64, 32, neon),   lbd_entry(dc_128, 64, 64, neon),
    lbd_entry(smooth_h, 4, 4, neon),   lbd_entry(smooth_h, 4, 8, neon),
    lbd_entry(smooth_h, 4, 16, neon),  lbd_entry(smooth_h, 8, 4, neon),
    lbd_entry(smooth_h, 8, 8, neon),   lbd_entry(smooth_h, 8, 16, neon),
    lbd_entry(smooth_h, 8, 32, neon),  lbd_entry(smooth_h, 16, 4, neon),
    lbd_entry(smooth_h, 16, 8, neon),  lbd_entry(smooth_h, 16, 16, neon),
    lbd_entry(smooth_h, 16, 32, neon), lbd_entry(smooth_h, 16, 64, neon),
    lbd_entry(smooth_h, 32, 8, neon),  lbd_entry(smooth_h, 32, 16, neon),
    lbd_entry(smooth_h, 32, 32, neon), lbd_entry(smooth_h, 32, 64, neon),
    lbd_entry(smooth_h, 64, 16, neon), lbd_entry(smooth_h, 64, 32, neon),
    lbd_entry(smooth_h, 64, 64, neon), lbd_entry(smooth_v, 4, 4, neon),
    lbd_entry(smooth_v, 4, 8, neon),   lbd_entry(smooth_v, 4, 16, neon),
    lbd_entry(smooth_v, 8, 4, neon),   lbd_entry(smooth_v, 8, 8, neon),
    lbd_entry(smooth_v, 8, 16, neon),  lbd_entry(smooth_v, 8, 32, neon),
    lbd_entry(smooth_v, 16, 4, neon),  lbd_entry(smooth_v, 16, 8, neon),
    lbd_entry(smooth_v, 16, 16, neon), lbd_entry(smooth_v, 16, 32, neon),
    lbd_entry(smooth_v, 16, 64, neon), lbd_entry(smooth_v, 32, 8, neon),
    lbd_entry(smooth_v, 32, 16, neon), lbd_entry(smooth_v, 32, 32, neon),
    lbd_entry(smooth_v, 32, 64, neon), lbd_entry(smooth_v, 64, 16, neon),
    lbd_entry(smooth_v, 64, 32, neon), lbd_entry(smooth_v, 64, 64, neon),
    lbd_entry(smooth, 4, 4, neon),     lbd_entry(smooth, 4, 8, neon),
    lbd_entry(smooth, 4, 16, neon),    lbd_entry(smooth, 8, 4, neon),
    lbd_entry(smooth, 8, 8, neon),     lbd_entry(smooth, 8, 16, neon),
    lbd_entry(smooth, 8, 32, neon),    lbd_entry(smooth, 16, 4, neon),
    lbd_entry(smooth, 16, 8, neon),    lbd_entry(smooth, 16, 16, neon),
    lbd_entry(smooth, 16, 32, neon),   lbd_entry(smooth, 16, 64, neon),
    lbd_entry(smooth, 32, 8, neon),    lbd_entry(smooth, 32, 16, neon),
    lbd_entry(smooth, 32, 32, neon),   lbd_entry(smooth, 32, 64, neon),
    lbd_entry(smooth, 64, 16, neon),   lbd_entry(smooth, 64, 32, neon),
    lbd_entry(smooth, 64, 64, neon),   lbd_entry(v, 4, 4, neon),
    lbd_entry(v, 4, 8, neon),          lbd_entry(v, 4, 16, neon),
    lbd_entry(v, 8, 4, neon),          lbd_entry(v, 8, 8, neon),
    lbd_entry(v, 8, 16, neon),         lbd_entry(v, 8, 32, neon),
    lbd_entry(v, 16, 4, neon),         lbd_entry(v, 16, 8, neon),
    lbd_entry(v, 16, 16, neon),        lbd_entry(v, 16, 32, neon),
    lbd_entry(v, 16, 64, neon),        lbd_entry(v, 32, 8, neon),
    lbd_entry(v, 32, 16, neon),        lbd_entry(v, 32, 32, neon),
    lbd_entry(v, 32, 64, neon),        lbd_entry(v, 64, 16, neon),
    lbd_entry(v, 64, 32, neon),        lbd_entry(v, 64, 64, neon),
    lbd_entry(h, 4, 4, neon),          lbd_entry(h, 4, 8, neon),
    lbd_entry(h, 4, 16, neon),         lbd_entry(h, 8, 4, neon),
    lbd_entry(h, 8, 8, neon),          lbd_entry(h, 8, 16, neon),
    lbd_entry(h, 8, 32, neon),         lbd_entry(h, 16, 4, neon),
    lbd_entry(h, 16, 8, neon),         lbd_entry(h, 16, 16, neon),
    lbd_entry(h, 16, 32, neon),        lbd_entry(h, 16, 64, neon),
    lbd_entry(h, 32, 8, neon),         lbd_entry(h, 32, 16, neon),
    lbd_entry(h, 32, 32, neon),        lbd_entry(h, 32, 64, neon),
    lbd_entry(h, 64, 16, neon),        lbd_entry(h, 64, 32, neon),
    lbd_entry(h, 64, 64, neon),        lbd_entry(paeth, 4, 4, neon),
    lbd_entry(paeth, 4, 8, neon),      lbd_entry(paeth, 4, 16, neon),
    lbd_entry(paeth, 8, 4, neon),      lbd_entry(paeth, 8, 8, neon),
    lbd_entry(paeth, 8, 16, neon),     lbd_entry(paeth, 8, 32, neon),
    lbd_entry(paeth, 16, 4, neon),     lbd_entry(paeth, 16, 8, neon),
    lbd_entry(paeth, 16, 16, neon),    lbd_entry(paeth, 16, 32, neon),
    lbd_entry(paeth, 16, 64, neon),    lbd_entry(paeth, 32, 8, neon),
    lbd_entry(paeth, 32, 16, neon),    lbd_entry(paeth, 32, 32, neon),
    lbd_entry(paeth, 32, 64, neon),    lbd_entry(paeth, 64, 16, neon),
    lbd_entry(paeth, 64, 32, neon),    lbd_entry(paeth, 64, 64, neon),
};

INSTANTIATE_TEST_SUITE_P(NEON, LowbdIntraPredTest,
                         ::testing::ValuesIn(LowbdIntraPredTestVectorAsmNEON));
#endif  // ARCH_AARCH64

}  // namespace
