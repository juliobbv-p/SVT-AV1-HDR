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
#include "definitions.h"
#include "util.h"
#include "warped_motion.h"
#include "warp_filter_test_util.h"
#include "convolve.h"
#include <array>
#include <vector>

using std::make_tuple;
using std::tuple;

namespace {
constexpr int quant_dist_lookup_table[2][4][2] = {
    {{9, 7}, {11, 5}, {12, 4}, {13, 3}},
    {{7, 9}, {5, 11}, {4, 12}, {3, 13}},
};

int32_t random_warped_param(svt_av1_test_tool::SVTRandom &rnd, int bits) {
    // 1 in 8 chance of generating zero (arbitrarily chosen)
    if (((rnd.Rand8()) & 7) == 0)
        return 0;
    // Otherwise, enerate uniform values in the range
    // [-(1 << bits), 1] U [1, 1<<bits]
    int32_t v = 1 + (rnd.Rand16() & ((1 << bits) - 1));
    if ((rnd.Rand8()) & 1)
        return -v;
    return v;
}

void generate_warped_model(svt_av1_test_tool::SVTRandom &rnd,
                           std::array<int32_t, 8> &mat, int16_t &alpha,
                           int16_t &beta, int16_t &gamma, int16_t &delta,
                           const int is_alpha_zero, const int is_beta_zero,
                           const int is_gamma_zero, const int is_delta_zero) {
    while (1) {
        int rnd8 = rnd.Rand8() & 3;
        mat[0] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS + 6);
        mat[1] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS + 6);
        mat[2] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
                 (1 << WARPEDMODEL_PREC_BITS);
        mat[3] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);

        if (rnd8 <= 1) {
            // AFFINE
            mat[4] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);
            mat[5] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
                     (1 << WARPEDMODEL_PREC_BITS);
        } else if (rnd8 == 2) {
            mat[4] = -mat[3];
            mat[5] = mat[2];
        } else {
            mat[4] = random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3);
            mat[5] = (random_warped_param(rnd, WARPEDMODEL_PREC_BITS - 3)) +
                     (1 << WARPEDMODEL_PREC_BITS);
            if (is_alpha_zero == 1)
                mat[2] = 1 << WARPEDMODEL_PREC_BITS;
            if (is_beta_zero == 1)
                mat[3] = 0;
            if (is_gamma_zero == 1)
                mat[4] = 0;
            if (is_delta_zero == 1)
                mat[5] = (((int64_t)mat[3] * mat[4] + (mat[2] / 2)) / mat[2]) +
                         (1 << WARPEDMODEL_PREC_BITS);
        }

        // Calculate the derived parameters and check that they are suitable
        // for the warp filter.
        assert(mat[2] != 0);

        alpha = static_cast<int16_t>(
            clamp(mat[2] - (1 << WARPEDMODEL_PREC_BITS), INT16_MIN, INT16_MAX));
        beta = static_cast<int16_t>(clamp(mat[3], INT16_MIN, INT16_MAX));
        gamma = static_cast<int16_t>(
            clamp(((int64_t)mat[4] * (1 << WARPEDMODEL_PREC_BITS)) / mat[2],
                  INT16_MIN,
                  INT16_MAX));
        delta = static_cast<int16_t>(clamp(
            mat[5] - (((int64_t)mat[3] * mat[4] + (mat[2] / 2)) / mat[2]) -
                (1 << WARPEDMODEL_PREC_BITS),
            INT16_MIN,
            INT16_MAX));

        if ((4 * abs(alpha) + 7 * abs(beta) >= (1 << WARPEDMODEL_PREC_BITS)) ||
            (4 * abs(gamma) + 4 * abs(delta) >= (1 << WARPEDMODEL_PREC_BITS)))
            continue;

        alpha = ROUND_POWER_OF_TWO_SIGNED(alpha, WARP_PARAM_REDUCE_BITS) *
                (1 << WARP_PARAM_REDUCE_BITS);
        beta = ROUND_POWER_OF_TWO_SIGNED(beta, WARP_PARAM_REDUCE_BITS) *
               (1 << WARP_PARAM_REDUCE_BITS);
        gamma = ROUND_POWER_OF_TWO_SIGNED(gamma, WARP_PARAM_REDUCE_BITS) *
                (1 << WARP_PARAM_REDUCE_BITS);
        delta = ROUND_POWER_OF_TWO_SIGNED(delta, WARP_PARAM_REDUCE_BITS) *
                (1 << WARP_PARAM_REDUCE_BITS);

        // We have a valid model, so finish
        return;
    }
}
}  // namespace

namespace libaom_test {

namespace AV1WarpFilter {
::testing::internal::ParamGenerator<WarpTestParams> BuildParams(
    warp_affine_func filter) {
    const WarpTestParam params[] = {
        make_tuple(4, 4, 50000, filter),
        make_tuple(8, 8, 50000, filter),
        make_tuple(64, 64, 1000, filter),
        make_tuple(4, 16, 20000, filter),
        make_tuple(32, 8, 10000, filter),
    };
    return ::testing::Combine(::testing::ValuesIn(params),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1));
}

void AV1WarpFilterTest::RunCheckOutput(const warp_affine_func test_impl) {
    constexpr int w = 128, h = 128;
    constexpr int border = 16;
    constexpr int stride = w + 2 * border;
    const auto params = TEST_GET_PARAM(0);
    const int is_alpha_zero = TEST_GET_PARAM(1);
    const int is_beta_zero = TEST_GET_PARAM(2);
    const int is_gamma_zero = TEST_GET_PARAM(3);
    const int is_delta_zero = TEST_GET_PARAM(4);
    const int out_w = std::get<0>(params), out_h = std::get<1>(params);
    const int num_iters = std::get<2>(params);
    int j;
    constexpr int bd = 8;

    // The warp functions always write rows with widths that are multiples of 8.
    // So to avoid a buffer overflow, we may need to pad rows to a multiple
    // of 8.
    const int output_n = ((out_w + 7) & ~7) * out_h;
    std::vector<uint8_t> input_(h * stride);
    uint8_t *input = input_.data() + border;
    std::vector<uint8_t> output(output_n);
    std::vector<uint8_t> output2(output_n);
    std::array<int32_t, 8> mat;
    int16_t alpha, beta, gamma, delta;
    std::vector<ConvBufType> dsta(output_n, 0);
    std::vector<ConvBufType> dstb(output_n, 0);
    for (int i = 0; i < output_n; ++i) {
        output[i] = output2[i] = rnd_.Rand8();
    }

    for (int i = 0; i < num_iters; ++i) {
        // Generate an input block and extend its borders horizontally
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                input[r * stride + c] = rnd_.Rand8();
        for (int r = 0; r < h; ++r) {
            memset(input + r * stride - border, input[r * stride], border);
            memset(input + r * stride + w, input[r * stride + (w - 1)], border);
        }
        const int use_no_round = rnd_.Rand8() & 1;
        for (int sub_x = 0; sub_x < 2; ++sub_x)
            for (int sub_y = 0; sub_y < 2; ++sub_y) {
                generate_warped_model(rnd_,
                                      mat,
                                      alpha,
                                      beta,
                                      gamma,
                                      delta,
                                      is_alpha_zero,
                                      is_beta_zero,
                                      is_gamma_zero,
                                      is_delta_zero);

                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 5; ++jj) {
                        for (int do_average = 0; do_average <= 1;
                             ++do_average) {
                            ConvolveParams conv_params;
                            if (use_no_round) {
                                conv_params = get_conv_params_no_round(
                                    do_average, dsta.data(), out_w, 1, bd);
                            } else {
                                conv_params = get_conv_params(0, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }
                            svt_av1_warp_affine_c(mat.data(),
                                                  input,
                                                  w,
                                                  h,
                                                  stride,
                                                  output.data(),
                                                  32,
                                                  32,
                                                  out_w,
                                                  out_h,
                                                  out_w,
                                                  sub_x,
                                                  sub_y,
                                                  &conv_params,
                                                  alpha,
                                                  beta,
                                                  gamma,
                                                  delta);
                            if (use_no_round) {
                                conv_params = get_conv_params_no_round(
                                    do_average, dstb.data(), out_w, 1, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }
                            test_impl(mat.data(),
                                      input,
                                      w,
                                      h,
                                      stride,
                                      output2.data(),
                                      32,
                                      32,
                                      out_w,
                                      out_h,
                                      out_w,
                                      sub_x,
                                      sub_y,
                                      &conv_params,
                                      alpha,
                                      beta,
                                      gamma,
                                      delta);
                            if (use_no_round) {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(dsta[j], dstb[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            } else {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            }
                        }
                    }
                }
            }
    }
}
}  // namespace AV1WarpFilter

namespace AV1HighbdWarpFilter {
::testing::internal::ParamGenerator<HighbdWarpTestParams> BuildParams(
    highbd_warp_affine_func filter) {
    const HighbdWarpTestParam params[] = {
        make_tuple(4, 4, 100, 8, filter),
        make_tuple(8, 8, 100, 8, filter),
        make_tuple(64, 64, 100, 8, filter),
        make_tuple(4, 16, 100, 8, filter),
        make_tuple(32, 8, 100, 8, filter),
        make_tuple(4, 4, 100, 10, filter),
        make_tuple(8, 8, 100, 10, filter),
        make_tuple(64, 64, 100, 10, filter),
        make_tuple(4, 16, 100, 10, filter),
        make_tuple(32, 8, 100, 10, filter),
    };
    return ::testing::Combine(::testing::ValuesIn(params),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1),
                              ::testing::Values(0, 1));
}

void AV1HighbdWarpFilterTest::RunCheckOutput(
    const highbd_warp_affine_func test_impl) {
    constexpr int w = 128, h = 128;
    constexpr int border = 16;
    constexpr int stride8b = w + 2 * border;
    constexpr int stride2b = w + 2 * border;
    const auto param = TEST_GET_PARAM(0);
    const int is_alpha_zero = TEST_GET_PARAM(1);
    const int is_beta_zero = TEST_GET_PARAM(2);
    const int is_gamma_zero = TEST_GET_PARAM(3);
    const int is_delta_zero = TEST_GET_PARAM(4);
    const int out_w = std::get<0>(param), out_h = std::get<1>(param);
    const int bd = std::get<3>(param);
    const int num_iters = std::get<2>(param);
    const int mask = (1 << bd) - 1;
    int i, j, sub_x, sub_y;

    // The warp functions always write rows with widths that are multiples of 8.
    // So to avoid a buffer overflow, we may need to pad rows to a multiple
    // of 8.
    int output_n = ((out_w + 7) & ~7) * out_h;
    std::vector<uint8_t> input8b_(h * stride8b);
    uint8_t *input8b = input8b_.data() + border;
    std::vector<uint8_t> input2b_(h * stride2b);
    uint8_t *input2b = input2b_.data() + border;
    std::vector<uint16_t> output(output_n);
    std::vector<uint16_t> output2(output_n);
    std::array<int32_t, 8> mat;
    int16_t alpha, beta, gamma, delta;
    ConvolveParams conv_params;
    std::vector<ConvBufType> dsta(output_n, 0);
    std::vector<ConvBufType> dstb(output_n, 0);
    for (i = 0; i < output_n; ++i) {
        output[i] = output2[i] = rnd_.Rand16();
    }

    for (i = 0; i < num_iters; ++i) {
        // Generate an input block and extend its borders horizontally
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c) {
                uint16_t val = rnd_.Rand16() & mask;
                input8b[r * stride8b + c] = val >> 2;
                input2b[r * stride2b + c] = (val & 3) << 6;
            }
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < border; ++c) {
                input8b[r * stride8b - border + c] = input8b[r * stride8b];
                input8b[r * stride8b + w + c] = input8b[r * stride8b + (w - 1)];
                input2b[r * stride2b - border + c] = input2b[r * stride2b];
                input2b[r * stride2b + w + c] = input2b[r * stride2b + (w - 1)];
            }
        }
        const int use_no_round = rnd_.Rand8() & 1;
        for (sub_x = 0; sub_x < 2; ++sub_x)
            for (sub_y = 0; sub_y < 2; ++sub_y) {
                generate_warped_model(rnd_,
                                      mat,
                                      alpha,
                                      beta,
                                      gamma,
                                      delta,
                                      is_alpha_zero,
                                      is_beta_zero,
                                      is_gamma_zero,
                                      is_delta_zero);
                for (int ii = 0; ii < 2; ++ii) {
                    for (int jj = 0; jj < 5; ++jj) {
                        for (int do_average = 0; do_average <= 1;
                             ++do_average) {
                            if (use_no_round) {
                                conv_params = get_conv_params_no_round(
                                    do_average, dsta.data(), out_w, 1, bd);
                            } else {
                                conv_params = get_conv_params(0, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }

                            svt_av1_highbd_warp_affine_c(mat.data(),
                                                         input8b,
                                                         input2b,
                                                         w,
                                                         h,
                                                         stride8b,
                                                         stride2b,
                                                         output.data(),
                                                         32,
                                                         32,
                                                         out_w,
                                                         out_h,
                                                         out_w,
                                                         sub_x,
                                                         sub_y,
                                                         bd,
                                                         &conv_params,
                                                         alpha,
                                                         beta,
                                                         gamma,
                                                         delta);
                            if (use_no_round) {
                                // TODO(angiebird): Change this to test_impl
                                // once we have SIMD implementation
                                conv_params = get_conv_params_no_round(
                                    do_average, dstb.data(), out_w, 1, bd);
                            }
                            if (jj >= 4) {
                                conv_params.use_jnt_comp_avg = 0;
                            } else {
                                conv_params.use_jnt_comp_avg = 1;
                                conv_params.fwd_offset =
                                    quant_dist_lookup_table[ii][jj][0];
                                conv_params.bck_offset =
                                    quant_dist_lookup_table[ii][jj][1];
                            }
                            test_impl(mat.data(),
                                      input8b,
                                      input2b,
                                      w,
                                      h,
                                      stride8b,
                                      stride2b,
                                      output2.data(),
                                      32,
                                      32,
                                      out_w,
                                      out_h,
                                      out_w,
                                      sub_x,
                                      sub_y,
                                      bd,
                                      &conv_params,
                                      alpha,
                                      beta,
                                      gamma,
                                      delta);

                            if (use_no_round) {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(dsta[j], dstb[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            } else {
                                for (j = 0; j < out_w * out_h; ++j)
                                    ASSERT_EQ(output[j], output2[j])
                                        << "Pixel mismatch at index " << j
                                        << " = (" << (j % out_w) << ", "
                                        << (j / out_w) << ") on iteration "
                                        << i;
                            }
                        }
                    }
                }
            }
    }
}
}  // namespace AV1HighbdWarpFilter
}  // namespace libaom_test
