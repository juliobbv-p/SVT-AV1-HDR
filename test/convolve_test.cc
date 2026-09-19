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
 * @file Convolve2dTest.cc
 *
 * @brief Unit test for interpolation in inter prediction:
 * - svt_av1_highbd_convolve_2d_copy_sr_avx2
 * - svt_av1_highbd_jnt_convolve_2d_copy_avx2
 * - svt_av1_highbd_convolve_x_sr_avx2
 * - svt_av1_highbd_convolve_y_sr_avx2
 * - svt_av1_highbd_convolve_2d_sr_avx2
 * - svt_av1_highbd_jnt_convolve_x_avx2
 * - svt_av1_highbd_jnt_convolve_y_avx2
 * - svt_av1_highbd_jnt_convolve_2d_avx2
 * - svt_av1_convolve_2d_copy_sr_avx2
 * - svt_av1_jnt_convolve_2d_copy_avx2
 * - svt_av1_convolve_x_sr_avx2
 * - svt_av1_convolve_y_sr_avx2
 * - svt_av1_convolve_2d_sr_avx2
 * - svt_av1_jnt_convolve_x_avx2
 * - svt_av1_jnt_convolve_y_avx2
 * - svt_av1_jnt_convolve_2d_avx2
 *
 * @author Cidana-Wenyao
 *
 ******************************************************************************/
#include <array>
#include <cstdlib>
#include "gtest/gtest.h"
#include "definitions.h"
#include "random.h"
#include "util.h"
#include "convolve.h"

#ifdef ARCH_X86_64
#include "convolve_avx2.h"
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
#include "convolve_neon.h"
#endif  // ARCH_AARCH64

#if defined(_MSC_VER)
#pragma warning(suppress : 4324)
#endif

/**
 * @brief Unit test for interpolation in inter prediction:
 * - av1_{highbd, }_{jnt, }_convolve_{x, y, 2d}_{sr, copy}
 *
 * Test strategy:
 * Verify this assembly code by comparing with reference c implementation.
 * Feed the same data and check test output and reference output.
 * Define a template class to handle the common process, and
 * declare sub class to handle different bitdepth and function types.
 *
 * Expect result:
 * Output from assemble functions should be the same with output from c.
 *
 * Test coverage:
 * Test cases:
 * input value: Fill with random values
 * modes: jnt, sr, copy, x, y modes
 * TxSize: all the TxSize.
 * BitDepth: 8bit, 10bit, 12bit
 *
 */

namespace {
using svt_av1_test_tool::SVTRandom;
constexpr int kMaxSize = 128 + 32;  // padding

using highbd_convolve_func =
    void (*)(const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride,
             int w, int h, const InterpFilterParams *filter_params_x,
             const InterpFilterParams *filter_params_y, const int subpel_x_qn,
             const int subpel_y_qn, ConvolveParams *conv_params, int bd);

using lowbd_convolve_func = void (*)(const uint8_t *src, int src_stride,
                                     uint8_t *dst, int dst_stride, int w, int h,
                                     const InterpFilterParams *filter_params_x,
                                     const InterpFilterParams *filter_params_y,
                                     const int subpel_x_qn,
                                     const int subpel_y_qn,
                                     ConvolveParams *conv_params);

using LowbdConvolveParam =
    std::tuple<int, bool, bool, lowbd_convolve_func, BlockSize>;
using HighbdConvolveParam =
    std::tuple<int, bool, bool, highbd_convolve_func, BlockSize>;

template <typename Sample, typename FuncType, typename ConvolveParam,
          bool is_jnt = false>
class AV1ConvolveTest : public ::testing::TestWithParam<ConvolveParam> {
  public:
    void *operator new(size_t size) {
        if (void *ptr = svt_aom_memalign(alignof(AV1ConvolveTest), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

    virtual void run_convolve(int offset_r, int offset_c, int src_stride,
                              int dst_stride, int w, int h,
                              const InterpFilterParams *filter_params_x,
                              const InterpFilterParams *filter_params_y,
                              const int32_t subpel_x_q4,
                              const int32_t subpel_y_q4,
                              ConvolveParams *conv_params1,
                              ConvolveParams *conv_params2) = 0;

    void test_convolve(bool has_subx, bool has_suby, int src_stride,
                       int dst_stride, int output_w, int output_h,
                       const InterpFilterParams *filter_params_x,
                       const InterpFilterParams *filter_params_y,
                       ConvolveParams *conv_params_ref,
                       ConvolveParams *conv_params_tst) {
        const auto subx_range = has_subx ? 16 : 1;
        const auto suby_range = has_suby ? 16 : 1;
        for (int subx = 0; subx < subx_range; ++subx) {
            for (int suby = 0; suby < suby_range; ++suby) {
                constexpr auto offset_r = 3;
                constexpr auto offset_c = 3;

                reset_output();

                run_convolve(offset_r,
                             offset_c,
                             src_stride,
                             dst_stride,
                             output_w,
                             output_h,
                             filter_params_x,
                             filter_params_y,
                             subx,
                             suby,
                             conv_params_ref,
                             conv_params_tst);

                if (output_ref_ != output_tst_) {
                    for (int i = 0; i < MAX_SB_SIZE; ++i) {
                        for (int j = 0; j < MAX_SB_SIZE; ++j) {
                            int idx = i * MAX_SB_SIZE + j;
                            ASSERT_EQ(output_ref_[idx], output_tst_[idx])
                                << output_w << "x" << output_h
                                << " Pixel mismatch at index " << idx << " = ("
                                << j << ", " << i << "), sub pixel offset = ("
                                << suby << ", " << subx << ") tap = ("
                                << get_convolve_tap(filter_params_x->filter_ptr)
                                << "x"
                                << get_convolve_tap(filter_params_y->filter_ptr)
                                << ") do_average: "
                                << conv_params_tst->do_average
                                << " use_jnt_comp_avg: "
                                << conv_params_tst->use_jnt_comp_avg;
                        }
                    }
                }

                if (is_jnt && conv_buf_ref_ != conv_buf_tst_) {
                    for (int i = 0; i < MAX_SB_SIZE; ++i) {
                        for (int j = 0; j < MAX_SB_SIZE; ++j) {
                            int idx = i * MAX_SB_SIZE + j;
                            ASSERT_EQ(conv_buf_ref_[idx], conv_buf_tst_[idx])
                                << output_w << "x" << output_h
                                << " Pixel mismatch at index " << idx << " = ("
                                << j << ", " << i << "), sub pixel offset = ("
                                << suby << ", " << subx << ")"
                                << " tap = ("
                                << get_convolve_tap(filter_params_x->filter_ptr)
                                << "x"
                                << get_convolve_tap(filter_params_y->filter_ptr)
                                << ") do_average: "
                                << conv_params_tst->do_average
                                << " use_jnt_comp_avg: "
                                << conv_params_tst->use_jnt_comp_avg;
                        }
                    }
                }
            }
        }
    }

  protected:
    void prepare_data(int w, int h) {
        SVTRandom rnd_(bd_, false);  // bd_-bits, unsigned
        SVTRandom rnd12_(12, false);

        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                input[i * w + j] = (Sample)rnd_.random();

        for (int i = 0; i < MAX_SB_SQUARE; ++i) {
            conv_buf_init_[i] = rnd12_.random();
            output_init_[i] = (Sample)rnd_.random();
        }
    }

    void reset_output() {
        conv_buf_ref_ = conv_buf_init_;
        conv_buf_tst_ = conv_buf_init_;
        output_ref_ = output_init_;
        output_tst_ = output_init_;
    }

    void run_test() {
        constexpr int input_w = kMaxSize, input_h = kMaxSize;

        // fill the input data with random
        prepare_data(input_w, input_h);

        svt_aom_setup_common_rtcd_internal(svt_aom_get_cpu_flags_to_use());

        // loop the filter type and subpixel position
        const int output_w = block_size_wide[block_idx_];
        const int output_h = block_size_high[block_idx_];
        const InterpFilter max_hfilter =
            has_subx_ ? INTERP_FILTERS_ALL : EIGHTTAP_SMOOTH;
        const InterpFilter max_vfilter =
            has_suby_ ? INTERP_FILTERS_ALL : EIGHTTAP_SMOOTH;
        for (int compIdx = 0; compIdx < 2; ++compIdx) {
            for (int hfilter = EIGHTTAP_REGULAR; hfilter < max_hfilter;
                 ++hfilter) {
                for (int vfilter = EIGHTTAP_REGULAR; vfilter < max_vfilter;
                     ++vfilter) {
                    InterpFilterParams filter_params_x =
                        av1_get_interp_filter_params_with_block_size(
                            (InterpFilter)hfilter, output_w >> compIdx);
                    InterpFilterParams filter_params_y =
                        av1_get_interp_filter_params_with_block_size(
                            (InterpFilter)vfilter, output_h >> compIdx);
                    for (int do_average = 0; do_average < (1 + is_jnt);
                         ++do_average) {
                        // setup convolveParams according to jnt or sr
                        ConvolveParams conv_params_ref;
                        ConvolveParams conv_params_tst;
                        if (is_jnt) {
                            conv_params_ref =
                                get_conv_params_no_round(do_average,
                                                         conv_buf_ref_.data(),
                                                         MAX_SB_SIZE,
                                                         1,
                                                         bd_);
                            conv_params_tst =
                                get_conv_params_no_round(do_average,
                                                         conv_buf_tst_.data(),
                                                         MAX_SB_SIZE,
                                                         1,
                                                         bd_);
                            // Test special case where dist_wtd_comp_avg is not
                            // used
                            conv_params_ref.use_jnt_comp_avg = 0;
                            conv_params_tst.use_jnt_comp_avg = 0;

                        } else {
                            conv_params_ref = get_conv_params_no_round(
                                do_average, nullptr, 0, 0, bd_);
                            conv_params_tst = get_conv_params_no_round(
                                do_average, nullptr, 0, 0, bd_);
                        }

                        test_convolve(has_subx_,
                                      has_suby_,
                                      input_w,
                                      MAX_SB_SIZE,
                                      output_w >> compIdx,
                                      output_h >> compIdx,
                                      &filter_params_x,
                                      &filter_params_y,
                                      &conv_params_ref,
                                      &conv_params_tst);

                        // AV1 standard won't have 32x4 case.
                        // This only favors some optimization feature which
                        // subsamples 32x8 to 32x4 and triggers 4-tap filter.
                        if (!is_jnt && has_suby_ &&
                            (output_w >> compIdx) == 32 &&
                            (output_h >> compIdx) == 8) {
                            filter_params_y =
                                av1_get_interp_filter_params_with_block_size(
                                    (InterpFilter)vfilter, 4);
                            test_convolve(has_subx_,
                                          has_suby_,
                                          input_w,
                                          MAX_SB_SIZE,
                                          32,
                                          4,
                                          &filter_params_x,
                                          &filter_params_y,
                                          &conv_params_ref,
                                          &conv_params_tst);
                        }

                        if (!is_jnt)
                            continue;

                        constexpr int quant_dist_lookup_table[2][4][2] = {
                            {{9, 7}, {11, 5}, {12, 4}, {13, 3}},
                            {{7, 9}, {5, 11}, {4, 12}, {3, 13}},
                        };
                        // Test different combination of fwd and bck offset
                        // weights
                        for (int k = 0; k < 2; ++k) {
                            for (int l = 0; l < 4; ++l) {
                                conv_params_ref.use_jnt_comp_avg = 1;
                                conv_params_tst.use_jnt_comp_avg = 1;
                                conv_params_ref.fwd_offset =
                                    quant_dist_lookup_table[k][l][0];
                                conv_params_ref.bck_offset =
                                    quant_dist_lookup_table[k][l][1];
                                conv_params_tst.fwd_offset =
                                    quant_dist_lookup_table[k][l][0];
                                conv_params_tst.bck_offset =
                                    quant_dist_lookup_table[k][l][1];

                                test_convolve(has_subx_,
                                              has_suby_,
                                              input_w,
                                              MAX_SB_SIZE,
                                              output_w >> compIdx,
                                              output_h >> compIdx,
                                              &filter_params_x,
                                              &filter_params_y,
                                              &conv_params_ref,
                                              &conv_params_tst);
                            }
                        }
                    }
                }
            }
        }
    }

  protected:
    const FuncType func_tst_{TEST_GET_PARAM(3)};
    const int bd_{TEST_GET_PARAM(0)};
    const int block_idx_{TEST_GET_PARAM(4)};
    const bool has_subx_{TEST_GET_PARAM(1)};
    const bool has_suby_{TEST_GET_PARAM(2)};
    Sample input[kMaxSize * kMaxSize]{};
    alignas(32) std::array<ConvBufType, MAX_SB_SQUARE> conv_buf_init_{};
    alignas(32) std::array<ConvBufType, MAX_SB_SQUARE> conv_buf_ref_{};
    alignas(32) std::array<ConvBufType, MAX_SB_SQUARE> conv_buf_tst_{};
    alignas(32) std::array<Sample, MAX_SB_SQUARE> output_init_{};
    alignas(32) std::array<Sample, MAX_SB_SQUARE> output_ref_{};
    alignas(32) std::array<Sample, MAX_SB_SQUARE> output_tst_{};
};

::testing::internal::ParamGenerator<LowbdConvolveParam> BuildParamsLbd(
    bool has_subx, bool has_suby, lowbd_convolve_func func) {
    return ::testing::Combine(::testing::Values(8),
                              ::testing::Values(has_subx),
                              ::testing::Values(has_suby),
                              ::testing::Values(func),
                              ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL));
}

template <lowbd_convolve_func func_ref, bool is_jnt = false>
class AV1LbdConvolveTest : public AV1ConvolveTest<uint8_t, lowbd_convolve_func,
                                                  LowbdConvolveParam, is_jnt> {
  public:
    void run_convolve(int offset_r, int offset_c, int src_stride,
                      int dst_stride, int output_w, int output_h,
                      const InterpFilterParams *filter_params_x,
                      const InterpFilterParams *filter_params_y,
                      const int32_t subpel_x_q4, const int32_t subpel_y_q4,
                      ConvolveParams *conv_params_ref,
                      ConvolveParams *conv_params_tst) override {
        func_ref(this->input + offset_r * src_stride + offset_c,
                 src_stride,
                 this->output_ref_.data(),
                 dst_stride,
                 output_w,
                 output_h,
                 filter_params_x,
                 filter_params_y,
                 subpel_x_q4,
                 subpel_y_q4,
                 conv_params_ref);
        this->func_tst_(this->input + offset_r * src_stride + offset_c,
                        src_stride,
                        this->output_tst_.data(),
                        dst_stride,
                        output_w,
                        output_h,
                        filter_params_x,
                        filter_params_y,
                        subpel_x_q4,
                        subpel_y_q4,
                        conv_params_tst);
    }
};

using AV1LbdJntConvolveTest =
    AV1LbdConvolveTest<svt_av1_jnt_convolve_2d_c, true>;

TEST_P(AV1LbdJntConvolveTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_AVX2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_AVX2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_jnt_convolve_x_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_AVX2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_jnt_convolve_y_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_AVX2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_jnt_convolve_2d_copy_avx2));

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SSE2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_sse2));
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SSSE3, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_ssse3));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_SSE2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_jnt_convolve_x_sse2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_SSE2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_jnt_convolve_y_sse2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_SSE2, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_jnt_convolve_2d_copy_sse2));
#if EN_AVX512_SUPPORT
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_AVX512, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_avx512));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_AVX512, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_jnt_convolve_x_avx512));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_AVX512, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_jnt_convolve_y_avx512));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_AVX512, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_jnt_convolve_2d_copy_avx512));
#endif

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_jnt_convolve_x_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_jnt_convolve_y_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_NEON, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_jnt_convolve_2d_copy_neon));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON_DOTPROD, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_neon_dotprod));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON_DOTPROD, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_jnt_convolve_x_neon_dotprod));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON_DOTPROD, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_jnt_convolve_y_neon_dotprod));
#endif  // HAVE_DOTPROD

#if HAVE_NEON_I8MM
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON_I8MM, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_jnt_convolve_2d_neon_i8mm));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON_I8MM, AV1LbdJntConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_jnt_convolve_x_neon_i8mm));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON_I8MM, AV1LbdJntConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_jnt_convolve_y_neon_i8mm));
#endif  // HAVE_NEON_I8MM
#endif  // ARCH_AARCH64

using AV1LbdSrConvolveTest =
    AV1LbdConvolveTest<svt_av1_convolve_2d_sr_c, false>;

TEST_P(AV1LbdSrConvolveTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_AVX2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_convolve_2d_sr_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_AVX2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_convolve_x_sr_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_AVX2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_convolve_y_sr_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_AVX2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_convolve_2d_copy_sr_avx2));

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SSE2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_convolve_2d_sr_sse2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_SSE2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_convolve_x_sr_sse2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_SSE2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_convolve_y_sr_sse2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_SSE2, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_convolve_2d_copy_sr_sse2));
#if EN_AVX512_SUPPORT
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_AVX512, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_convolve_2d_sr_avx512));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_AVX512, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_convolve_x_sr_avx512));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_AVX512, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_convolve_y_sr_avx512));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_AVX512, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_convolve_2d_copy_sr_avx512));
#endif

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_convolve_2d_sr_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_convolve_x_sr_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_convolve_y_sr_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestCOPY_NEON, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, false,
                                        svt_av1_convolve_2d_copy_sr_neon));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON_DOTPROD, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_convolve_2d_sr_neon_dotprod));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON_DOTPROD, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_convolve_x_sr_neon_dotprod));
#endif  // HAVE_NEON_DOTPROD

#if HAVE_NEON_I8MM
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON_I8MM, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, true,
                                        svt_av1_convolve_2d_sr_neon_i8mm));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON_I8MM, AV1LbdSrConvolveTest,
                         BuildParamsLbd(true, false,
                                        svt_av1_convolve_x_sr_neon_i8mm));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON_I8MM, AV1LbdSrConvolveTest,
                         BuildParamsLbd(false, true,
                                        svt_av1_convolve_y_sr_neon_i8mm));
#endif  // HAVE_NEON_I8MM
#endif  // ARCH_AARCH64

::testing::internal::ParamGenerator<HighbdConvolveParam> BuildParamsHbd(
    bool has_subx, bool has_suby, highbd_convolve_func func) {
    return ::testing::Combine(::testing::Range(8, 11, 2),
                              ::testing::Values(has_subx),
                              ::testing::Values(has_suby),
                              ::testing::Values(func),
                              ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL));
}

template <highbd_convolve_func func_ref, bool is_jnt = false>
class AV1HbdConvolveTest
    : public AV1ConvolveTest<uint16_t, highbd_convolve_func,
                             HighbdConvolveParam, is_jnt> {
  public:
    void run_convolve(int offset_r, int offset_c, int src_stride,
                      int dst_stride, int blk_w, int blk_h,
                      const InterpFilterParams *filter_params_x,
                      const InterpFilterParams *filter_params_y,
                      const int32_t subpel_x_q4, const int32_t subpel_y_q4,
                      ConvolveParams *conv_params_ref,
                      ConvolveParams *conv_params_tst) override {
        func_ref(this->input + offset_r * src_stride + offset_c,
                 src_stride,
                 this->output_ref_.data(),
                 dst_stride,
                 blk_w,
                 blk_h,
                 filter_params_x,
                 filter_params_y,
                 subpel_x_q4,
                 subpel_y_q4,
                 conv_params_ref,
                 this->bd_);
        this->func_tst_(this->input + offset_r * src_stride + offset_c,
                        src_stride,
                        this->output_tst_.data(),
                        dst_stride,
                        blk_w,
                        blk_h,
                        filter_params_x,
                        filter_params_y,
                        subpel_x_q4,
                        subpel_y_q4,
                        conv_params_tst,
                        this->bd_);
    }
};

using AV1HbdJntConvolveTest =
    AV1HbdConvolveTest<svt_av1_highbd_jnt_convolve_2d_c, true>;

TEST_P(AV1HbdJntConvolveTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SSE4_1, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_jnt_convolve_2d_sse4_1));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_SSE4_1, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_jnt_convolve_x_sse4_1));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_SSE4_1, AV1HbdJntConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_jnt_convolve_y_sse4_1));
INSTANTIATE_TEST_SUITE_P(
    ConvolveTestCOPY_SSE4_1, AV1HbdJntConvolveTest,
    BuildParamsHbd(false, false, svt_av1_highbd_jnt_convolve_2d_copy_sse4_1));

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_AVX2, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_jnt_convolve_2d_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_AVX2, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_jnt_convolve_x_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_AVX2, AV1HbdJntConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_jnt_convolve_y_avx2));
INSTANTIATE_TEST_SUITE_P(
    ConvolveTestCOPY_AVX2, AV1HbdJntConvolveTest,
    BuildParamsHbd(false, false, svt_av1_highbd_jnt_convolve_2d_copy_avx2));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_jnt_convolve_2d_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_jnt_convolve_x_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON, AV1HbdJntConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_jnt_convolve_y_neon));
INSTANTIATE_TEST_SUITE_P(
    ConvolveTestCOPY_NEON, AV1HbdJntConvolveTest,
    BuildParamsHbd(false, false, svt_av1_highbd_jnt_convolve_2d_copy_neon));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_SVE, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_jnt_convolve_x_sve));
#endif  // HAVE_SVE

#if HAVE_SVE2
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SVE2, AV1HbdJntConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_jnt_convolve_2d_sve2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_SVE2, AV1HbdJntConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_jnt_convolve_y_sve2));
#endif  // HAVE_SVE2
#endif  // ARCH_AARCH64

using AV1HbdSrConvolveTest =
    AV1HbdConvolveTest<svt_av1_highbd_convolve_2d_sr_c, false>;

TEST_P(AV1HbdSrConvolveTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SSSE3, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_convolve_2d_sr_ssse3));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_SSSE3, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_convolve_x_sr_ssse3));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_SSSE3, AV1HbdSrConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_convolve_y_sr_ssse3));
INSTANTIATE_TEST_SUITE_P(
    ConvolveTestCOPY_SSSE3, AV1HbdSrConvolveTest,
    BuildParamsHbd(false, false, svt_av1_highbd_convolve_2d_copy_sr_ssse3));

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_AVX2, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_convolve_2d_sr_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_AVX2, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_convolve_x_sr_avx2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_AVX2, AV1HbdSrConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_convolve_y_sr_avx2));
INSTANTIATE_TEST_SUITE_P(
    ConvolveTestCOPY_AVX2, AV1HbdSrConvolveTest,
    BuildParamsHbd(false, false, svt_av1_highbd_convolve_2d_copy_sr_avx2));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64

INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_NEON, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_convolve_2d_sr_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_NEON, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_convolve_x_sr_neon));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_NEON, AV1HbdSrConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_convolve_y_sr_neon));
INSTANTIATE_TEST_SUITE_P(
    ConvolveTestCOPY_NEON, AV1HbdSrConvolveTest,
    BuildParamsHbd(false, false, svt_av1_highbd_convolve_2d_copy_sr_neon));

#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(ConvolveTestX_SVE, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, false,
                                        svt_av1_highbd_convolve_x_sr_sve));
#endif  // HAVE_SVE

#if HAVE_SVE2
INSTANTIATE_TEST_SUITE_P(ConvolveTest2D_SVE2, AV1HbdSrConvolveTest,
                         BuildParamsHbd(true, true,
                                        svt_av1_highbd_convolve_2d_sr_sve2));
INSTANTIATE_TEST_SUITE_P(ConvolveTestY_SVE2, AV1HbdSrConvolveTest,
                         BuildParamsHbd(false, true,
                                        svt_av1_highbd_convolve_y_sr_sve2));
#endif  // HAVE_SVE2
#endif  // ARCH_AARCH64

}  // namespace
