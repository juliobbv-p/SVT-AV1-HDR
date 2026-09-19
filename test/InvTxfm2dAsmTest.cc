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

#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "random.h"
#include "util.h"
#include "aom_dsp_rtcd.h"
#include "unit_test_utility.h"
#include "TxfmCommon.h"

namespace {
using svt_av1_test_tool::SVTRandom;  // to generate the random

constexpr bool is_tx_type_imp_32x32(const TxType tx_type) {
    return tx_type == DCT_DCT || tx_type == IDTX;
}

constexpr bool is_tx_type_imp_64x64(const TxType tx_type) {
    return tx_type == DCT_DCT;
}

template <typename Params>
class InvTxfm2dAsmTestBase : public ::testing::TestWithParam<Params> {
  public:
    void *operator new(size_t size) {
        if (void *ptr = svt_aom_memalign(alignof(InvTxfm2dAsmTestBase), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

  protected:
    // clear the coeffs according to eob position, note the coeffs are
    // linear.
    void clear_high_freq_coeffs(const TxSize tx_size, const TxType tx_type,
                                const int eob, const int max_eob) {
        const ScanOrder *scan_order = get_scan_order(tx_size, tx_type);
        const int16_t *scan = scan_order->scan;

        for (int i = eob; i < max_eob; ++i) {
            input_[scan[i]] = 0;
        }
    }

    // fill the pixel_input with random data and do forward transform,
    // Note that the forward transform do not re-pack the coefficients,
    // so we have to re-pack the coefficients after transform for
    // some tx_size;
    void populate_with_random(const int width, const int height,
                              const TxType tx_type, const TxSize tx_size) {
        constexpr decltype(&svt_av1_transform_two_d_4x4_c)
            fwd_txfm_func[TX_SIZES_ALL]{
                svt_av1_transform_two_d_4x4_c,
                svt_av1_transform_two_d_8x8_c,
                svt_av1_transform_two_d_16x16_c,
                svt_av1_transform_two_d_32x32_c,
                svt_av1_transform_two_d_64x64_c,
                svt_av1_fwd_txfm2d_4x8_c,
                svt_av1_fwd_txfm2d_8x4_c,
                svt_av1_fwd_txfm2d_8x16_c,
                svt_av1_fwd_txfm2d_16x8_c,
                svt_av1_fwd_txfm2d_16x32_c,
                svt_av1_fwd_txfm2d_32x16_c,
                svt_av1_fwd_txfm2d_32x64_c,
                svt_av1_fwd_txfm2d_64x32_c,
                svt_av1_fwd_txfm2d_4x16_c,
                svt_av1_fwd_txfm2d_16x4_c,
                svt_av1_fwd_txfm2d_8x32_c,
                svt_av1_fwd_txfm2d_32x8_c,
                svt_av1_fwd_txfm2d_16x64_c,
                svt_av1_fwd_txfm2d_64x16_c,
            };

        output_ref_.fill(0);
        output_test_.fill(0);
        input_.fill(0);
        pixel_input_.fill(0);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                pixel_input_[i * stride_ + j] =
                    static_cast<int16_t>(s_bd_rnd_.random());
                output_ref_[i * stride_ + j] = output_test_[i * stride_ + j] =
                    static_cast<uint16_t>(u_bd_rnd_.random());
            }
        }

        fwd_txfm_func[tx_size](pixel_input_.data(),
                               input_.data(),
                               stride_,
                               tx_type,
                               static_cast<uint8_t>(bd_));
        // post-process, re-pack the coeffcients
        switch (tx_size) {
        case TX_64X64: svt_handle_transform64x64_c(input_.data()); break;
        case TX_64X32: svt_handle_transform64x32_c(input_.data()); break;
        case TX_32X64: svt_handle_transform32x64_c(input_.data()); break;
        case TX_64X16: svt_handle_transform64x16_c(input_.data()); break;
        case TX_16X64: svt_handle_transform16x64_c(input_.data()); break;
        default: break;
        }
    }

    int bd_{TEST_GET_PARAM(0)}; /**< input param 8bit or 10bit */
    SVTRandom u_bd_rnd_{0, (1 << bd_) - 1};
    SVTRandom s_bd_rnd_{-(1 << bd_) + 1, (1 << bd_) - 1};

    static constexpr int stride_ = MAX_TX_SIZE;
    alignas(64) std::array<int16_t, MAX_TX_SQUARE> pixel_input_{};
    alignas(64) std::array<int32_t, MAX_TX_SQUARE> input_{};
    alignas(64) std::array<uint16_t, MAX_TX_SQUARE> output_test_{};
    alignas(64) std::array<uint16_t, MAX_TX_SQUARE> output_ref_{};
};

using InvSqrTxfm2dFunc = void (*)(const int32_t *input, uint16_t *output_r,
                                  int32_t stride_r, uint16_t *output_w,
                                  int32_t stride_w, TxType tx_type, int32_t bd);
using InvSqrTxfmTestParam = std::tuple<int, InvSqrTxfm2dFunc, InvSqrTxfm2dFunc,
                                       IsTxTypeImpFunc, TxSize>;

using InvTxfm2dAsmSqrTest = InvTxfm2dAsmTestBase<InvSqrTxfmTestParam>;

TEST_P(InvTxfm2dAsmSqrTest, sqr_txfm_match_test) {
    const auto ref_func = TEST_GET_PARAM(1);
    const auto test_func = TEST_GET_PARAM(2);
    const auto check_imp_func = TEST_GET_PARAM(3);
    const auto tx_size = TEST_GET_PARAM(4);
    const int width = tx_size_wide[tx_size];
    const int height = tx_size_high[tx_size];

    if (ref_func == nullptr || test_func == nullptr)
        return;
    for (int tx_type = DCT_DCT; tx_type < TX_TYPES; ++tx_type) {
        const TxType type = static_cast<TxType>(tx_type);

        if (!is_txfm_allowed(type, tx_size))
            continue;

        // Some tx_type is not implemented yet, so we will skip this;
        if (!check_imp_func(type))
            continue;

        constexpr int loops = 100;
        for (int k = 0; k < loops; k++) {
            populate_with_random(width, height, type, tx_size);

            ref_func(input_.data(),
                     output_ref_.data(),
                     stride_,
                     output_ref_.data(),
                     stride_,
                     type,
                     bd_);
            test_func(input_.data(),
                      output_test_.data(),
                      stride_,
                      output_test_.data(),
                      stride_,
                      type,
                      bd_);

            ASSERT_EQ(output_ref_, output_test_)
                << "width: " << width << " height: " << height << " loop: " << k
                << " tx_type: " << tx_type << " tx_size: " << tx_size;
        }
    }
}

// clang-format off
#define SQR_FUNC_PAIRS(name, type, tx_size, is_tx_type_imp)                 \
    {8,reinterpret_cast<decltype(&name##_c)>(name##_c),                          \
     reinterpret_cast<decltype(&name##_c)>(name##_##type), is_tx_type_imp,     \
     tx_size},                                                           \
    {10, reinterpret_cast<decltype(&name##_c)>(name##_c),                          \
     reinterpret_cast<decltype(&name##_c)>(name##_##type), is_tx_type_imp,     \
     tx_size}

#define SQR_FUNC_PAIRS_DAV1D(name, type, tx_size, is_tx_type_imp)          \
    {8, reinterpret_cast<decltype(&svt_av1_##name##_c)>(svt_av1_##name##_c),               \
     reinterpret_cast<decltype(&svt_av1_##name##_c)>(svt_dav1d_##name##_##type),        \
     is_tx_type_imp, tx_size},                                          \
    {10, reinterpret_cast<decltype(&svt_av1_##name##_c)>(svt_av1_##name##_c),               \
     reinterpret_cast<decltype(&svt_av1_##name##_c)>(svt_dav1d_##name##_##type),        \
     is_tx_type_imp, tx_size}
// clang-format on

#ifdef ARCH_X86_64

const InvSqrTxfmTestParam sqr_inv_txfm_c_sse4_1_func_pairs[10] = {
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x4, sse4_1, TX_4X4,
                   dct_adst_combine_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x8, sse4_1, TX_8X8, all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x16, sse4_1, TX_16X16,
                   all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x32, sse4_1, TX_32X32,
                   dct_adst_combine_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x64, sse4_1, TX_64X64,
                   is_tx_type_imp_64x64),
};

INSTANTIATE_TEST_SUITE_P(SSE4_1, InvTxfm2dAsmSqrTest,
                         ::testing::ValuesIn(sqr_inv_txfm_c_sse4_1_func_pairs));

const InvSqrTxfmTestParam sqr_inv_txfm_c_avx2_func_pairs[10] = {
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x4, avx2, TX_4X4, all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x8, avx2, TX_8X8, all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x16, avx2, TX_16X16,
                   all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x32, avx2, TX_32X32,
                   is_tx_type_imp_32x32),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x64, avx2, TX_64X64,
                   is_tx_type_imp_64x64),
};

INSTANTIATE_TEST_SUITE_P(AVX2, InvTxfm2dAsmSqrTest,
                         ::testing::ValuesIn(sqr_inv_txfm_c_avx2_func_pairs));

const InvSqrTxfmTestParam sqr_dav1d_inv_txfm_c_avx2_func_pairs[10] = {
    SQR_FUNC_PAIRS_DAV1D(inv_txfm2d_add_4x4, avx2, TX_4X4, all_txtype_imp),
    SQR_FUNC_PAIRS_DAV1D(inv_txfm2d_add_8x8, avx2, TX_8X8, all_txtype_imp),
    SQR_FUNC_PAIRS_DAV1D(inv_txfm2d_add_16x16, avx2, TX_16X16, all_txtype_imp),
    SQR_FUNC_PAIRS_DAV1D(inv_txfm2d_add_32x32, avx2, TX_32X32,
                         is_tx_type_imp_32x32),
    SQR_FUNC_PAIRS_DAV1D(inv_txfm2d_add_64x64, avx2, TX_64X64,
                         is_tx_type_imp_64x64),
};

INSTANTIATE_TEST_SUITE_P(
    dav1d_AVX2, InvTxfm2dAsmSqrTest,
    ::testing::ValuesIn(sqr_dav1d_inv_txfm_c_avx2_func_pairs));

#if EN_AVX512_SUPPORT
const InvSqrTxfmTestParam sqr_inv_txfm_c_avx512_func_pairs[6] = {
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x16, avx512, TX_16X16,
                   all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x32, avx512, TX_32X32,
                   is_tx_type_imp_32x32),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x64, avx512, TX_64X64,
                   is_tx_type_imp_64x64),
};

INSTANTIATE_TEST_SUITE_P(AVX512, InvTxfm2dAsmSqrTest,
                         ::testing::ValuesIn(sqr_inv_txfm_c_avx512_func_pairs));
#endif

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
const InvSqrTxfmTestParam inv_txfm_c_neon_func_pairs[10] = {
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x4, neon, TX_4X4, all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x8, neon, TX_8X8, all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x16, neon, TX_16X16,
                   all_txtype_imp),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x32, neon, TX_32X32,
                   is_tx_type_imp_32x32),
    SQR_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x64, neon, TX_64X64,
                   is_tx_type_imp_64x64),
};

INSTANTIATE_TEST_SUITE_P(NEON, InvTxfm2dAsmSqrTest,
                         ::testing::ValuesIn(inv_txfm_c_neon_func_pairs));
#endif  // ARCH_AARCH64

using InvRectTxfm2dType1Func = void (*)(const int32_t *input,
                                        uint16_t *output_r, int32_t stride_r,
                                        uint16_t *output_w, int32_t stride_w,
                                        TxType tx_type, TxSize tx_size,
                                        int32_t eob, int32_t bd);
using InvRectTxfmType1TestParam =
    std::tuple<int, InvRectTxfm2dType1Func, InvRectTxfm2dType1Func, TxSize>;

using InvTxfm2dAsmType1Test = InvTxfm2dAsmTestBase<InvRectTxfmType1TestParam>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InvTxfm2dAsmType1Test);

TEST_P(InvTxfm2dAsmType1Test, rect_type1_txfm_match_test) {
    const auto ref_func = TEST_GET_PARAM(1);
    const auto test_func = TEST_GET_PARAM(2);
    const auto tx_size = TEST_GET_PARAM(3);
    const int width = tx_size_wide[tx_size];
    const int height = tx_size_high[tx_size];
    const int max_eob = av1_get_max_eob(tx_size);

    for (int tx_type = DCT_DCT; tx_type < TX_TYPES; ++tx_type) {
        const TxType type = static_cast<TxType>(tx_type);

        if (!is_txfm_allowed(type, tx_size))
            continue;

        const int loops = 10 * max_eob;
        SVTRandom eob_rnd{1, max_eob - 1};
        for (int k = 0; k < loops; k++) {
            const int eob = k < max_eob - 1 ? k + 1 : eob_rnd.random();
            // prepare data by forward transform and then
            // clear the values between eob and max_eob
            populate_with_random(width, height, type, tx_size);
            clear_high_freq_coeffs(tx_size, type, eob, max_eob);

            ref_func(input_.data(),
                     output_ref_.data(),
                     stride_,
                     output_ref_.data(),
                     stride_,
                     type,
                     tx_size,
                     eob,
                     bd_);
            test_func(input_.data(),
                      output_test_.data(),
                      stride_,
                      output_test_.data(),
                      stride_,
                      type,
                      tx_size,
                      eob,
                      bd_);

            ASSERT_EQ(output_ref_, output_test_)
                << "loop: " << k << " tx_type: " << tx_type
                << " tx_size: " << (int32_t)tx_size << " eob: " << eob;
        }
    }
}

#define RECT_TYPE1_FUNC_PAIRS(c, asm, tx_size)            \
    {                                                     \
        8,                                                \
        reinterpret_cast<decltype(&c)>(c),                \
        reinterpret_cast<decltype(&c)>(asm),              \
        tx_size,                                          \
    },                                                    \
    {                                                     \
        10, reinterpret_cast<decltype(&c)>(c),            \
            reinterpret_cast<decltype(&c)>(asm), tx_size, \
    }

#ifdef ARCH_X86_64
const InvRectTxfmType1TestParam rect_type1_ref_funcs_sse4_1[20] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x16_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_8X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x32_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_8X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x8_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_16X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x32_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_16X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x64_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_16X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x8_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_32X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x16_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_32X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x64_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_32X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x16_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_64X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x32_c,
                          svt_av1_highbd_inv_txfm_add_sse4_1, TX_64X32),
};

INSTANTIATE_TEST_SUITE_P(SSE4_1, InvTxfm2dAsmType1Test,
                         ::testing::ValuesIn(rect_type1_ref_funcs_sse4_1));

const InvRectTxfmType1TestParam rect_type1_ref_funcs_avx2[20] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x16_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_8X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x32_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_8X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x8_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_16X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x32_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_16X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x64_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_16X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x8_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_32X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x16_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_32X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x64_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_32X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x16_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_64X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x32_c,
                          svt_dav1d_highbd_inv_txfm_add_avx2, TX_64X32),
};

INSTANTIATE_TEST_SUITE_P(AVX2, InvTxfm2dAsmType1Test,
                         ::testing::ValuesIn(rect_type1_ref_funcs_avx2));

#if EN_AVX512_SUPPORT
const InvRectTxfmType1TestParam rect_type1_ref_funcs_avx512[12] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x32_c,
                          svt_av1_inv_txfm2d_add_16x32_avx512, TX_16X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x64_c,
                          svt_av1_inv_txfm2d_add_16x64_avx512, TX_16X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x16_c,
                          svt_av1_inv_txfm2d_add_32x16_avx512, TX_32X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x64_c,
                          svt_av1_inv_txfm2d_add_32x64_avx512, TX_32X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x16_c,
                          svt_av1_inv_txfm2d_add_64x16_avx512, TX_64X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x32_c,
                          svt_av1_inv_txfm2d_add_64x32_avx512, TX_64X32),
};

INSTANTIATE_TEST_SUITE_P(AVX512, InvTxfm2dAsmType1Test,
                         ::testing::ValuesIn(rect_type1_ref_funcs_avx512));
#endif

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
const InvRectTxfmType1TestParam rect_type1_ref_funcs_neon[20] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x16_c,
                          svt_av1_inv_txfm2d_add_8x16_neon, TX_8X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x32_c,
                          svt_av1_inv_txfm2d_add_8x32_neon, TX_8X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x8_c,
                          svt_av1_inv_txfm2d_add_16x8_neon, TX_16X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x32_c,
                          svt_av1_inv_txfm2d_add_16x32_neon, TX_16X32),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x64_c,
                          svt_av1_inv_txfm2d_add_16x64_neon, TX_16X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x8_c,
                          svt_av1_inv_txfm2d_add_32x8_neon, TX_32X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x16_c,
                          svt_av1_inv_txfm2d_add_32x16_neon, TX_32X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_32x64_c,
                          svt_av1_inv_txfm2d_add_32x64_neon, TX_32X64),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x16_c,
                          svt_av1_inv_txfm2d_add_64x16_neon, TX_64X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_64x32_c,
                          svt_av1_inv_txfm2d_add_64x32_neon, TX_64X32),
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(NEON, InvTxfm2dAsmType1Test,
                         ::testing::ValuesIn(rect_type1_ref_funcs_neon));
#endif  // ARCH_AARCH64

using InvRectTxfm2dType2Func = void (*)(const int32_t *input,
                                        uint16_t *output_r, int32_t stride_r,
                                        uint16_t *output_w, int32_t stride_w,
                                        TxType tx_type, TxSize tx_size,
                                        int32_t bd);

using InvRectTxfmType2TestParam =
    std::tuple<int, InvRectTxfm2dType2Func, InvRectTxfm2dType2Func, TxSize>;

using InvTxfm2dAsmType2Test = InvTxfm2dAsmTestBase<InvRectTxfmType2TestParam>;
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InvTxfm2dAsmType2Test);

TEST_P(InvTxfm2dAsmType2Test, rect_type2_txfm_match_test) {
    const auto ref_func = TEST_GET_PARAM(1);
    const auto test_func = TEST_GET_PARAM(2);
    const auto tx_size = TEST_GET_PARAM(3);
    const int width = tx_size_wide[tx_size];
    const int height = tx_size_high[tx_size];

    for (int tx_type = DCT_DCT; tx_type < TX_TYPES; ++tx_type) {
        const TxType type = static_cast<TxType>(tx_type);

        if (!is_txfm_allowed(type, tx_size))
            continue;

        constexpr int loops = 100;
        for (int k = 0; k < loops; k++) {
            populate_with_random(width, height, type, tx_size);

            ref_func(input_.data(),
                     output_ref_.data(),
                     stride_,
                     output_ref_.data(),
                     stride_,
                     type,
                     tx_size,
                     bd_);
            test_func(input_.data(),
                      output_test_.data(),
                      stride_,
                      output_test_.data(),
                      stride_,
                      type,
                      tx_size,
                      bd_);

            ASSERT_EQ(output_ref_, output_test_)
                << "loop: " << k << " tx_type: " << tx_type
                << " tx_size: " << tx_size;
        }
    }
}

#ifdef ARCH_X86_64
const InvRectTxfmType2TestParam rect_type2_ref_funcs_sse4_1[8] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x8_c,
                          svt_av1_inv_txfm2d_add_4x8_sse4_1, TX_4X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x4_c,
                          svt_av1_inv_txfm2d_add_8x4_sse4_1, TX_8X4),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x16_c,
                          svt_av1_inv_txfm2d_add_4x16_sse4_1, TX_4X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x4_c,
                          svt_av1_inv_txfm2d_add_16x4_sse4_1, TX_16X4),
};

INSTANTIATE_TEST_SUITE_P(SSE4_1, InvTxfm2dAsmType2Test,
                         ::testing::ValuesIn(rect_type2_ref_funcs_sse4_1));

const InvRectTxfmType2TestParam rect_type2_ref_funcs_avx2[8] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x8_c,
                          svt_dav1d_inv_txfm2d_add_4x8_avx2, TX_4X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x4_c,
                          svt_dav1d_inv_txfm2d_add_8x4_avx2, TX_8X4),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x16_c,
                          svt_dav1d_inv_txfm2d_add_4x16_avx2, TX_4X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x4_c,
                          svt_dav1d_inv_txfm2d_add_16x4_avx2, TX_16X4),
};

INSTANTIATE_TEST_SUITE_P(AVX2, InvTxfm2dAsmType2Test,
                         ::testing::ValuesIn(rect_type2_ref_funcs_avx2));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
const InvRectTxfmType2TestParam rect_type2_ref_funcs_neon[8] = {
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x8_c,
                          svt_av1_inv_txfm2d_add_4x8_neon, TX_4X8),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_4x16_c,
                          svt_av1_inv_txfm2d_add_4x16_neon, TX_4X16),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_8x4_c,
                          svt_av1_inv_txfm2d_add_8x4_neon, TX_8X4),
    RECT_TYPE1_FUNC_PAIRS(svt_av1_inv_txfm2d_add_16x4_c,
                          svt_av1_inv_txfm2d_add_16x4_neon, TX_16X4),
};

INSTANTIATE_TEST_SUITE_P(NEON, InvTxfm2dAsmType2Test,
                         ::testing::ValuesIn(rect_type2_ref_funcs_neon));
#endif  // ARCH_AARCH64

class InvTxfm2dAddTest
    : public InvTxfm2dAsmTestBase<
          std::tuple<EbBitDepth, decltype(&svt_av1_inv_txfm_add_c)>> {
  public:
    void run_svt_av1_inv_txfm_add_test(const TxSize tx_size, bool lossless) {
        const auto test_func = TEST_GET_PARAM(1);
        TxfmParam txfm_param;
        txfm_param.bd = bd_;
        txfm_param.lossless = lossless;
        txfm_param.tx_size = tx_size;
        txfm_param.eob = av1_get_max_eob(tx_size);

        if (bd_ > 8 && !lossless) {
            // Not support 10 bit with not lossless
            return;
        }

        constexpr int txfm_support_matrix[19][16] = {
            //[Size][type]" // O - No; 1 - lossless; 2 - !lossless; 3 - any
            /*0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15*/
            {3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},  // 0  TX_4X4,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 1  TX_8X8,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0},  // 2  TX_16X16,
            {3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0},  // 3  TX_32X32,
            {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},  // 4  TX_64X64,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 5  TX_4X8,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 6  TX_8X4,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 7  TX_8X16,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 8  TX_16X8,
            {3, 1, 3, 1, 1, 3, 1, 1, 1, 3, 3, 3, 1, 3, 1, 3},  // 9  TX_16X32,
            {3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3, 1},  // 10 TX_32X16,
            {3, 0, 1, 0, 0, 1, 0, 0, 0, 3, 3, 3, 0, 1, 0, 1},  // 11 TX_32X64,
            {3, 1, 0, 0, 1, 0, 0, 0, 0, 3, 3, 3, 1, 0, 1, 0},  // 12 TX_64X32,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 13 TX_4X16,
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},  // 14 TX_16X4,
            {3, 1, 3, 1, 1, 3, 1, 1, 1, 3, 3, 3, 1, 3, 1, 3},  // 15 TX_8X32,
            {3, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 3, 1},  // 16 TX_32X8,
            {3, 0, 3, 0, 0, 3, 0, 0, 0, 3, 3, 3, 0, 3, 0, 3},  // 17 TX_16X64,
            {3, 3, 0, 0, 3, 0, 0, 0, 0, 3, 3, 3, 3, 0, 3, 0}   // 18 TX_64X16,
            /*0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15*/
        };

        const int width = tx_size_wide[tx_size];
        const int height = tx_size_high[tx_size];

        for (int tx_type = DCT_DCT; tx_type < TX_TYPES; ++tx_type) {
            const TxType type = static_cast<TxType>(tx_type);
            txfm_param.tx_type = type;

            if ((lossless && ((txfm_support_matrix[tx_size][type] & 1) == 0)) ||
                (!lossless && ((txfm_support_matrix[tx_size][type] & 2) == 0)))
                continue;

            constexpr int loops = 10;
            for (int k = 0; k < loops; k++) {
                populate_with_random(width, height, type, tx_size);

                svt_av1_inv_txfm_add_c(
                    input_.data(),
                    reinterpret_cast<uint8_t *>(output_ref_.data()),
                    stride_,
                    reinterpret_cast<uint8_t *>(output_ref_.data()),
                    stride_,
                    &txfm_param);
                test_func(input_.data(),
                          reinterpret_cast<uint8_t *>(output_test_.data()),
                          stride_,
                          reinterpret_cast<uint8_t *>(output_test_.data()),
                          stride_,
                          &txfm_param);

                ASSERT_EQ(output_ref_, output_test_)
                    << "loop: " << k << " tx_type: " << (int)tx_type
                    << " tx_size: " << (int)tx_size;
            }
        }
    }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InvTxfm2dAddTest);

TEST_P(InvTxfm2dAddTest, svt_av1_inv_txfm_add) {
    // Reset all pointers to C
    svt_aom_setup_common_rtcd_internal(0);

    for (int i = TX_4X4; i < TX_SIZES_ALL; i++) {
        const TxSize tx_size = static_cast<TxSize>(i);
        run_svt_av1_inv_txfm_add_test(tx_size, false);
        run_svt_av1_inv_txfm_add_test(tx_size, true);
    }
}

#ifdef ARCH_X86_64

INSTANTIATE_TEST_SUITE_P(
    SSSE3, InvTxfm2dAddTest,
    ::testing::Combine(::testing::Values(EB_EIGHT_BIT, EB_TEN_BIT),
                       ::testing::Values(svt_av1_inv_txfm_add_ssse3)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, InvTxfm2dAddTest,
    ::testing::Combine(::testing::Values(EB_EIGHT_BIT, EB_TEN_BIT),
                       ::testing::Values(svt_av1_inv_txfm_add_avx2)));

INSTANTIATE_TEST_SUITE_P(
    dav1d_AVX2, InvTxfm2dAddTest,
    ::testing::Combine(::testing::Values(EB_EIGHT_BIT, EB_TEN_BIT),
                       ::testing::Values(svt_dav1d_inv_txfm_add_avx2)));

#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    dav1d_NEON, InvTxfm2dAddTest,
    ::testing::Combine(::testing::Values(EB_EIGHT_BIT, EB_TEN_BIT),
                       ::testing::Values(svt_dav1d_inv_txfm_add_neon)));
#endif  // ARCH_AARCH64

using HandleTransformFunc = uint64_t (*)(int32_t *output);

using HandleTransformParam =
    std::tuple<HandleTransformFunc, HandleTransformFunc, TxSize>;

using HandleTransformTest = ::testing::TestWithParam<HandleTransformParam>;

TEST_P(HandleTransformTest, match_test) {
    const auto ref_func_{TEST_GET_PARAM(0)};
    const auto test_func_{TEST_GET_PARAM(1)};
    const auto tx_size_{TEST_GET_PARAM(2)};
    alignas(64) std::array<int32_t, MAX_TX_SQUARE> input_ref{};
    svt_buf_random_s32(input_ref.data(), MAX_TX_SQUARE);
    alignas(64) std::array<int32_t, MAX_TX_SQUARE> input_test = input_ref;

    const uint64_t energy_ref = ref_func_(input_ref.data());
    const uint64_t energy_asm = test_func_(input_test.data());

    ASSERT_EQ(energy_ref, energy_asm);

    for (int i = 0; i < MAX_TX_SIZE; i++) {
        for (int j = 0; j < MAX_TX_SIZE; j++) {
            ASSERT_EQ(input_ref[i * MAX_TX_SIZE + j],
                      input_test[i * MAX_TX_SIZE + j])
                << " tx_size: " << tx_size_ << " " << j << " x " << i;
        }
    }
}

#ifdef ARCH_X86_64
const HandleTransformParam HandleTransformArrAVX2[10] = {
    // clang-format off
    { svt_handle_transform16x64_c, svt_handle_transform16x64_avx2, TX_16X64 },
    { svt_handle_transform32x64_c, svt_handle_transform32x64_avx2, TX_32X64 },
    { svt_handle_transform64x16_c, svt_handle_transform64x16_avx2, TX_64X16 },
    { svt_handle_transform64x32_c, svt_handle_transform64x32_avx2, TX_64X32 },
    { svt_handle_transform64x64_c, svt_handle_transform64x64_avx2, TX_64X64 },
    { svt_handle_transform16x64_N2_N4_c, svt_handle_transform16x64_N2_N4_avx2,
      TX_16X64 },
    { svt_handle_transform32x64_N2_N4_c, svt_handle_transform32x64_N2_N4_avx2,
      TX_32X64 },
    { svt_handle_transform64x16_N2_N4_c, svt_handle_transform64x16_N2_N4_avx2,
      TX_64X16 },
    { svt_handle_transform64x32_N2_N4_c, svt_handle_transform64x32_N2_N4_avx2,
      TX_64X32 },
    { svt_handle_transform64x64_N2_N4_c, svt_handle_transform64x64_N2_N4_avx2,
      TX_64X64 }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(AVX2, HandleTransformTest,
                         ::testing::ValuesIn(HandleTransformArrAVX2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
const HandleTransformParam HandleTransformArrNEON[10] = {
    // clang-format off
    { svt_handle_transform16x64_c, svt_handle_transform16x64_neon, TX_16X64 },
    { svt_handle_transform32x64_c, svt_handle_transform32x64_neon, TX_32X64 },
    { svt_handle_transform64x16_c, svt_handle_transform64x16_neon, TX_64X16 },
    { svt_handle_transform64x32_c, svt_handle_transform64x32_neon, TX_64X32 },
    { svt_handle_transform64x64_c, svt_handle_transform64x64_neon, TX_64X64 },
    { svt_handle_transform16x64_N2_N4_c, svt_handle_transform16x64_N2_N4_neon,
      TX_16X64 },
    { svt_handle_transform32x64_N2_N4_c, svt_handle_transform32x64_N2_N4_neon,
      TX_32X64 },
    { svt_handle_transform64x16_N2_N4_c, svt_handle_transform64x16_N2_N4_neon,
      TX_64X16 },
    { svt_handle_transform64x32_N2_N4_c, svt_handle_transform64x32_N2_N4_neon,
      TX_64X32 },
    { svt_handle_transform64x64_N2_N4_c, svt_handle_transform64x64_N2_N4_neon,
      TX_64X64 }
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(NEON, HandleTransformTest,
                         ::testing::ValuesIn(HandleTransformArrNEON));
#endif  // ARCH_AARCH64

}  // namespace
