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
 * @file CompoundUtilTest.cc
 *
 * @brief Unit test for util functions in wedge prediction:
 * - av1_build_compound_diffwtd_mask_{, d16}avx2
 * - svt_aom_blend_a64_mask_avx2/svt_aom_lowbd_blend_a64_d16_mask_avx2
 * - svt_aom_blend_a64_mask_sse4_1
 * -
 *svt_aom_highbd_blend_a64_mask_8bit_sse4_1/svt_aom_highbd_blend_a64_d16_mask_avx2
 * - svt_aom_blend_a64_hmask_sse4_1/svt_aom_blend_a64_vmask_sse4_1
 * -
 *svt_aom_highbd_blend_a64_hmask_16bit_sse4_1/svt_aom_highbd_blend_a64_vmask_16bit_sse4_1
 * - svt_aom_sse_avx2/svt_aom_highbd_sse_avx2
 *
 * @author Cidana-Wenyao
 *
 ******************************************************************************/
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include "definitions.h"
#include "aom_dsp_rtcd.h"
#include "random.h"
#include "convolve.h"
#include "svt_malloc.h"
#include "util.h"
#include "enc_inter_prediction.h"

using svt_av1_test_tool::SVTRandom;

constexpr auto MAX_MASK_SQUARE = (4 * MAX_SB_SQUARE);
#define MAKE_PARAM(func_type) std::tuple<func_type, func_type>

namespace {
template <typename SrcSample, typename DstSample, typename BlendTestParam,
          int bd, bool is_d16 = false, bool no_sub = false>
class CompBlendTest : public ::testing::TestWithParam<BlendTestParam> {
  public:
    void *operator new(size_t size) {
        if (void *ptr = svt_aom_memalign(alignof(CompBlendTest), size))
            return ptr;
        throw std::bad_alloc();
    }

    void operator delete(void *ptr) {
        svt_aom_free(ptr);
    }

    void run_test() {
        constexpr auto iterations = 1000;
        /*
          // max number of bits used by the source for d16 blends is as follows
          (from libaom): static const int kSrcMaxBitsMask = (1 << 14) - 1;
          static const int kSrcMaxBitsMaskHBD = (1 << 16) - 1;
        */
        constexpr auto max = is_d16 ? (bd == 8 ? 14 : 16) : bd;
        SVTRandom rnd{0, (1 << max) - 1};
        SVTRandom mask_rnd{0, 64};

        // generate random mask
        std::generate(mask_.begin(), mask_.end(), [&mask_rnd]() {
            return mask_rnd.random();
        });

        for (int k = 0; k < iterations; ++k) {
            for (int block_size = BLOCK_4X4; block_size < BLOCK_SIZES_ALL;
                 block_size += 1) {
                w_ = block_size_wide[block_size];
                h_ = block_size_high[block_size];

                // initial the input data
                for (int i = 0; i < h_; ++i) {
                    for (int j = 0; j < w_; ++j) {
                        src0_[i * src_stride_ + j] = rnd.random();
                        src1_[i * src_stride_ + j] = rnd.random();
                    }
                }

                int submax = 2;
                if (no_sub)
                    submax = 1;
                for (int subh = 0; subh < submax; ++subh) {
                    for (int subw = 0; subw < submax; ++subw) {
                        ref_dst_.fill(0);
                        tst_dst_.fill(0);

                        run_blend(subw, subh);

                        // check output
                        for (int i = 0; i < h_; ++i) {
                            for (int j = 0; j < w_; ++j) {
                                ASSERT_EQ(tst_dst_[i * dst_stride_ + j],
                                          ref_dst_[i * dst_stride_ + j])
                                    << " Pixel mismatch at index "
                                    << "[" << j << "x" << i
                                    << "]\nblock size: " << block_size
                                    << "\nbit-depth: " << bd
                                    << "\niterator: " << k;
                            }
                        }
                    }
                }
            }
        }
    }

    virtual void run_blend(int subw, int subh) = 0;

  protected:
    static constexpr int src_stride_ = MAX_SB_SIZE;
    static constexpr int dst_stride_ = MAX_SB_SIZE;
    static constexpr int mask_stride_ = 2 * MAX_SB_SIZE;
    alignas(32) std::array<DstSample, MAX_SB_SQUARE> ref_dst_{};
    alignas(32) std::array<DstSample, MAX_SB_SQUARE> tst_dst_{};
    alignas(32) std::array<SrcSample, MAX_SB_SQUARE> src0_{};
    alignas(32) std::array<SrcSample, MAX_SB_SQUARE> src1_{};
    alignas(32) std::array<uint8_t, MAX_MASK_SQUARE> mask_{};
    int w_{}, h_{};
};

using LbdBlendA64MaskFunc = decltype(&svt_aom_blend_a64_mask_c);

class LbdCompBlendTest
    : public CompBlendTest<uint8_t, uint8_t, LbdBlendA64MaskFunc, 8> {
  public:
    void run_blend(int subw, int subh) override {
        svt_aom_blend_a64_mask_c(ref_dst_.data(),
                                 dst_stride_,
                                 src0_.data(),
                                 src_stride_,
                                 src1_.data(),
                                 src_stride_,
                                 mask_.data(),
                                 mask_stride_,
                                 w_,
                                 h_,
                                 subw,
                                 subh);
        const auto test_impl = GetParam();
        test_impl(tst_dst_.data(),
                  dst_stride_,
                  src0_.data(),
                  src_stride_,
                  src1_.data(),
                  src_stride_,
                  mask_.data(),
                  mask_stride_,
                  w_,
                  h_,
                  subw,
                  subh);
    }
};

TEST_P(LbdCompBlendTest, BlendA64Mask) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(SSE4_1, LbdCompBlendTest,
                         ::testing::Values(svt_aom_blend_a64_mask_sse4_1));

INSTANTIATE_TEST_SUITE_P(AVX2, LbdCompBlendTest,
                         ::testing::Values(svt_aom_blend_a64_mask_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, LbdCompBlendTest,
                         ::testing::Values(svt_aom_blend_a64_mask_neon));
#endif  // ARCH_AARCH64

using LbdBlendA64D16MaskFunc = decltype(&svt_aom_lowbd_blend_a64_d16_mask_c);

class LbdCompBlendD16Test
    : public CompBlendTest<uint16_t, uint8_t, LbdBlendA64D16MaskFunc, 8, true> {
  public:
    void run_blend(int subw, int subh) override {
        ConvolveParams conv_params;
        conv_params.round_0 = ROUND0_BITS;
        conv_params.round_1 = COMPOUND_ROUND1_BITS;
        const auto test_impl = GetParam();
        test_impl(tst_dst_.data(),
                  dst_stride_,
                  src0_.data(),
                  src_stride_,
                  src1_.data(),
                  src_stride_,
                  mask_.data(),
                  mask_stride_,
                  w_,
                  h_,
                  subw,
                  subh,
                  &conv_params);
        svt_aom_lowbd_blend_a64_d16_mask_c(ref_dst_.data(),
                                           dst_stride_,
                                           src0_.data(),
                                           src_stride_,
                                           src1_.data(),
                                           src_stride_,
                                           mask_.data(),
                                           mask_stride_,
                                           w_,
                                           h_,
                                           subw,
                                           subh,
                                           &conv_params);
    }
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(LbdCompBlendD16Test);

TEST_P(LbdCompBlendD16Test, BlendA64MaskD16) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, LbdCompBlendD16Test,
    ::testing::Values(svt_aom_lowbd_blend_a64_d16_mask_sse4_1));
INSTANTIATE_TEST_SUITE_P(
    AVX2, LbdCompBlendD16Test,
    ::testing::Values(svt_aom_lowbd_blend_a64_d16_mask_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, LbdCompBlendD16Test,
    ::testing::Values(svt_aom_lowbd_blend_a64_d16_mask_neon));
#endif  // ARCH_AARCH64

using LbdBlendA64HMaskFunc = decltype(&svt_aom_blend_a64_hmask_c);

class LbdCompBlendHMaskTest
    : public CompBlendTest<uint8_t, uint8_t, LbdBlendA64HMaskFunc, 8, false,
                           true> {
  public:
    void run_blend(int, int) override {
        if (w_ > 32)
            return;
        const uint8_t *const mask = svt_av1_get_obmc_mask(w_);

        svt_aom_blend_a64_hmask_c(ref_dst_.data(),
                                  dst_stride_,
                                  src0_.data(),
                                  src_stride_,
                                  src1_.data(),
                                  src_stride_,
                                  mask,
                                  w_,
                                  h_);
        const auto test_impl = GetParam();
        test_impl(tst_dst_.data(),
                  dst_stride_,
                  src0_.data(),
                  src_stride_,
                  src1_.data(),
                  src_stride_,
                  mask,
                  w_,
                  h_);
    }
};

TEST_P(LbdCompBlendHMaskTest, BlendA64Mask) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(SSE4_1, LbdCompBlendHMaskTest,
                         ::testing::Values(svt_aom_blend_a64_hmask_sse4_1));

INSTANTIATE_TEST_SUITE_P(AVX2, LbdCompBlendHMaskTest,
                         ::testing::Values(svt_av1_blend_a64_hmask_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, LbdCompBlendHMaskTest,
                         ::testing::Values(svt_aom_blend_a64_hmask_neon));
#endif  // ARCH_AARCH64

using LbdBlendA64VMaskFunc = decltype(&svt_aom_blend_a64_vmask_c);

class LbdCompBlendVMaskTest
    : public CompBlendTest<uint8_t, uint8_t, LbdBlendA64VMaskFunc, 8, false,
                           true> {
  public:
    void run_blend(int, int) override {
        if (h_ > 32)
            return;
        const uint8_t *const mask = svt_av1_get_obmc_mask(h_);

        svt_aom_blend_a64_vmask_c(ref_dst_.data(),
                                  dst_stride_,
                                  src0_.data(),
                                  src_stride_,
                                  src1_.data(),
                                  src_stride_,
                                  mask,
                                  w_,
                                  h_);
        const auto test_impl = GetParam();
        test_impl(tst_dst_.data(),
                  dst_stride_,
                  src0_.data(),
                  src_stride_,
                  src1_.data(),
                  src_stride_,
                  mask,
                  w_,
                  h_);
    }
};

TEST_P(LbdCompBlendVMaskTest, BlendA64Mask) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(SSE4_1, LbdCompBlendVMaskTest,
                         ::testing::Values(svt_aom_blend_a64_vmask_sse4_1));

INSTANTIATE_TEST_SUITE_P(AVX2, LbdCompBlendVMaskTest,
                         ::testing::Values(svt_av1_blend_a64_vmask_avx2));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(NEON, LbdCompBlendVMaskTest,
                         ::testing::Values(svt_aom_blend_a64_vmask_neon));
#endif  // ARCH_AARCH64

using HbdBlendA64MaskFunc = decltype(&svt_aom_highbd_blend_a64_mask_c);

template <int bd_to_test>
class HbdCompBlendTest : public CompBlendTest<uint16_t, uint16_t,
                                              HbdBlendA64MaskFunc, bd_to_test> {
  public:
    void run_blend(int subw, int subh) override {
        svt_aom_highbd_blend_a64_mask_c(
            reinterpret_cast<uint8_t *>(this->ref_dst_.data()),
            this->dst_stride_,
            reinterpret_cast<uint8_t *>(this->src0_.data()),
            this->src_stride_,
            reinterpret_cast<uint8_t *>(this->src1_.data()),
            this->src_stride_,
            this->mask_.data(),
            this->mask_stride_,
            this->w_,
            this->h_,
            subw,
            subh,
            bd_to_test);
        const auto test_impl = this->GetParam();
        test_impl(reinterpret_cast<uint8_t *>(this->tst_dst_.data()),
                  this->dst_stride_,
                  reinterpret_cast<uint8_t *>(this->src0_.data()),
                  this->src_stride_,
                  reinterpret_cast<uint8_t *>(this->src1_.data()),
                  this->src_stride_,
                  this->mask_.data(),
                  this->mask_stride_,
                  this->w_,
                  this->h_,
                  subw,
                  subh,
                  bd_to_test);
    }
};

using HbdCompBlendTest8 = HbdCompBlendTest<8>;
using HbdCompBlendTest10 = HbdCompBlendTest<10>;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendTest8);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendTest10);

TEST_P(HbdCompBlendTest8, BlendA64Mask) {
    run_test();
}
TEST_P(HbdCompBlendTest10, BlendA64Mask) {
    run_test();
}

#define INSTANTIATE_HBD_BLEND_TEST(prefix, fixture, func)                  \
    INSTANTIATE_TEST_SUITE_P(prefix, fixture##8, ::testing::Values(func)); \
    INSTANTIATE_TEST_SUITE_P(prefix, fixture##10, ::testing::Values(func));

#ifdef ARCH_X86_64
INSTANTIATE_HBD_BLEND_TEST(SSE4_1, HbdCompBlendTest,
                           svt_aom_highbd_blend_a64_mask_8bit_sse4_1);
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_HBD_BLEND_TEST(NEON, HbdCompBlendTest,
                           svt_aom_highbd_blend_a64_mask_neon);
#endif  // ARCH_AARCH64

using HbdBlendA64D16MaskFunc = decltype(&svt_aom_highbd_blend_a64_d16_mask_c);

template <int bd_to_test>
class HbdCompBlendD16Test
    : public CompBlendTest<uint16_t, uint16_t, HbdBlendA64D16MaskFunc,
                           bd_to_test, true> {
  public:
    void run_blend(int subw, int subh) override {
        ConvolveParams conv_params;
        conv_params.round_0 = ROUND0_BITS;
        conv_params.round_1 = COMPOUND_ROUND1_BITS;
        const auto test_impl = this->GetParam();
        test_impl(reinterpret_cast<uint8_t *>(this->tst_dst_.data()),
                  this->dst_stride_,
                  this->src0_.data(),
                  this->src_stride_,
                  this->src1_.data(),
                  this->src_stride_,
                  this->mask_.data(),
                  this->mask_stride_,
                  this->w_,
                  this->h_,
                  subw,
                  subh,
                  &conv_params,
                  bd_to_test);
        svt_aom_highbd_blend_a64_d16_mask_c(
            reinterpret_cast<uint8_t *>(this->ref_dst_.data()),
            this->dst_stride_,
            this->src0_.data(),
            this->src_stride_,
            this->src1_.data(),
            this->src_stride_,
            this->mask_.data(),
            this->mask_stride_,
            this->w_,
            this->h_,
            subw,
            subh,
            &conv_params,
            bd_to_test);
    }
};

using HbdCompBlendD16Test8 = HbdCompBlendD16Test<8>;
using HbdCompBlendD16Test10 = HbdCompBlendD16Test<10>;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendD16Test8);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendD16Test10);

TEST_P(HbdCompBlendD16Test8, BlendA64MaskD16) {
    run_test();
}
TEST_P(HbdCompBlendD16Test10, BlendA64MaskD16) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_HBD_BLEND_TEST(SSE4_1, HbdCompBlendD16Test,
                           svt_aom_highbd_blend_a64_d16_mask_sse4_1);

INSTANTIATE_HBD_BLEND_TEST(AVX2, HbdCompBlendD16Test,
                           svt_aom_highbd_blend_a64_d16_mask_avx2);
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_HBD_BLEND_TEST(NEON, HbdCompBlendD16Test,
                           svt_aom_highbd_blend_a64_d16_mask_neon);
#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH
using HbdBlendA64HMaskFunc = decltype(&svt_aom_highbd_blend_a64_hmask_16bit_c);

template <int bd_to_test>
class HbdCompBlendHMaskTest
    : public CompBlendTest<uint16_t, uint16_t, HbdBlendA64HMaskFunc, bd_to_test,
                           false, true> {
  public:
    void run_blend(int, int) override {
        if (this->w_ > 32)
            return;
        const uint8_t *const mask = svt_av1_get_obmc_mask(this->w_);

        svt_aom_highbd_blend_a64_hmask_16bit_c(
            reinterpret_cast<uint16_t *>(this->ref_dst_.data()),
            this->dst_stride_,
            reinterpret_cast<uint16_t *>(this->src0_.data()),
            this->src_stride_,
            reinterpret_cast<uint16_t *>(this->src1_.data()),
            this->src_stride_,
            mask,
            this->w_,
            this->h_,
            bd_to_test);
        const auto test_impl = this->GetParam();
        test_impl(reinterpret_cast<uint16_t *>(this->tst_dst_.data()),
                  this->dst_stride_,
                  reinterpret_cast<uint16_t *>(this->src0_.data()),
                  this->src_stride_,
                  reinterpret_cast<uint16_t *>(this->src1_.data()),
                  this->src_stride_,
                  mask,
                  this->w_,
                  this->h_,
                  bd_to_test);
    }
};

using HbdCompBlendHMaskTest8 = HbdCompBlendHMaskTest<8>;
using HbdCompBlendHMaskTest10 = HbdCompBlendHMaskTest<10>;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendHMaskTest8);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendHMaskTest10);

TEST_P(HbdCompBlendHMaskTest8, BlendA64Mask) {
    run_test();
};
TEST_P(HbdCompBlendHMaskTest10, BlendA64Mask) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_HBD_BLEND_TEST(SSE4_1, HbdCompBlendHMaskTest,
                           svt_aom_highbd_blend_a64_hmask_16bit_sse4_1);

INSTANTIATE_HBD_BLEND_TEST(AVX2, HbdCompBlendHMaskTest,
                           svt_av1_highbd_blend_a64_hmask_16bit_avx2);
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_HBD_BLEND_TEST(NEON, HbdCompBlendHMaskTest,
                           svt_aom_highbd_blend_a64_hmask_16bit_neon);
#endif  // ARCH_AARCH64

using HbdBlendA64VMaskFunc = decltype(&svt_aom_highbd_blend_a64_vmask_16bit_c);

template <int bd_to_test>
class HbdCompBlendVMaskTest
    : public CompBlendTest<uint16_t, uint16_t, HbdBlendA64VMaskFunc, bd_to_test,
                           false, true> {
  public:
    void run_blend(int, int) override {
        if (this->h_ > 32)
            return;
        const uint8_t *const mask = svt_av1_get_obmc_mask(this->h_);

        svt_aom_highbd_blend_a64_vmask_16bit_c(
            reinterpret_cast<uint16_t *>(this->ref_dst_.data()),
            this->dst_stride_,
            reinterpret_cast<uint16_t *>(this->src0_.data()),
            this->src_stride_,
            reinterpret_cast<uint16_t *>(this->src1_.data()),
            this->src_stride_,
            mask,
            this->w_,
            this->h_,
            bd_to_test);
        HbdBlendA64VMaskFunc test_impl = this->GetParam();
        test_impl(reinterpret_cast<uint16_t *>(this->tst_dst_.data()),
                  this->dst_stride_,
                  reinterpret_cast<uint16_t *>(this->src0_.data()),
                  this->src_stride_,
                  reinterpret_cast<uint16_t *>(this->src1_.data()),
                  this->src_stride_,
                  mask,
                  this->w_,
                  this->h_,
                  bd_to_test);
    }
};

using HbdCompBlendVMaskTest8 = HbdCompBlendVMaskTest<8>;
using HbdCompBlendVMaskTest10 = HbdCompBlendVMaskTest<10>;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendVMaskTest8);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HbdCompBlendVMaskTest10);

TEST_P(HbdCompBlendVMaskTest8, BlendA64Mask) {
    run_test();
}
TEST_P(HbdCompBlendVMaskTest10, BlendA64Mask) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_HBD_BLEND_TEST(SSE4_1, HbdCompBlendVMaskTest,
                           svt_aom_highbd_blend_a64_vmask_16bit_sse4_1);

INSTANTIATE_HBD_BLEND_TEST(AVX2, HbdCompBlendVMaskTest,
                           svt_av1_highbd_blend_a64_vmask_16bit_avx2);
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_HBD_BLEND_TEST(NEON, HbdCompBlendVMaskTest,
                           svt_aom_highbd_blend_a64_vmask_16bit_neon);
#endif  // ARCH_AARCH64

#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

using BuildCompDiffwtdMaskedFunc =
    decltype(&svt_av1_build_compound_diffwtd_mask_c);
using BuildCompDiffwtdMaskParam =
    std::tuple<BlockSize, BuildCompDiffwtdMaskedFunc>;

class BuildCompDiffwtdMaskTest
    : public ::testing::TestWithParam<BuildCompDiffwtdMaskParam> {
  public:
    void run_test(const DIFFWTD_MASK_TYPE type) {
        const int block_size = TEST_GET_PARAM(0);
        const auto test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> mask_ref;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> mask_test;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> src0;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> src1;
        const int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            for (int j = 0; j < width * height; j++) {
                src0[j] = rnd_.random();
                src1[j] = rnd_.random();
            }

            svt_av1_build_compound_diffwtd_mask_c(mask_ref.data(),
                                                  type,
                                                  src0.data(),
                                                  width,
                                                  src1.data(),
                                                  width,
                                                  height,
                                                  width);

            test_impl(mask_test.data(),
                      type,
                      src0.data(),
                      width,
                      src1.data(),
                      width,
                      height,
                      width);
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    ASSERT_EQ(mask_ref[c + r * width], mask_test[c + r * width])
                        << "[" << r << "," << c << "] " << i << " @ " << width
                        << "x" << height << " inv " << type;
                }
            }
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(BuildCompDiffwtdMaskTest, MatchTest) {
    run_test(DIFFWTD_38);
    run_test(DIFFWTD_38_INV);
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, BuildCompDiffwtdMaskTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_sse4_1)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, BuildCompDiffwtdMaskTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, BuildCompDiffwtdMaskTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_neon)));
#endif  // ARCH_AARCH64

using BuildCompDiffwtdMaskedHighbdFunc =
    decltype(&svt_av1_build_compound_diffwtd_mask_highbd_c);
using BuildCompDiffwtdMaskHighbdParam =
    std::tuple<BlockSize, BuildCompDiffwtdMaskedHighbdFunc>;

class BuildCompDiffwtdMaskHighbdTest
    : public ::testing::TestWithParam<BuildCompDiffwtdMaskHighbdParam> {
  public:
    void run_test(const DIFFWTD_MASK_TYPE type, int bd) {
        const int block_size = TEST_GET_PARAM(0);
        const auto test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> mask_ref;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> mask_test;
        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> src0;
        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> src1;
        const int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            for (int j = 0; j < width * height; j++) {
                src0[j] = rnd_.random();
                src1[j] = rnd_.random();
            }

            svt_av1_build_compound_diffwtd_mask_highbd_c(
                mask_ref.data(),
                type,
                reinterpret_cast<uint8_t *>(src0.data()),
                width,
                reinterpret_cast<uint8_t *>(src1.data()),
                width,
                height,
                width,
                bd);

            test_impl(mask_test.data(),
                      type,
                      reinterpret_cast<uint8_t *>(src0.data()),
                      width,
                      reinterpret_cast<uint8_t *>(src1.data()),
                      width,
                      height,
                      width,
                      bd);
            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    ASSERT_EQ(mask_ref[c + r * width], mask_test[c + r * width])
                        << "[" << r << "," << c << "] " << i << " @ " << width
                        << "x" << height << " inv " << type;
                }
            }
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(BuildCompDiffwtdMaskHighbdTest, MatchTest) {
    run_test(DIFFWTD_38, 8);
    run_test(DIFFWTD_38_INV, 8);
    run_test(DIFFWTD_38, 10);
    run_test(DIFFWTD_38_INV, 10);
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSSE3, BuildCompDiffwtdMaskHighbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_highbd_ssse3)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, BuildCompDiffwtdMaskHighbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_highbd_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, BuildCompDiffwtdMaskHighbdTest,
    ::testing::Combine(
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_highbd_neon)));
#endif  // ARCH_AARCH64

// test svt_av1_build_compound_diffwtd_mask_d16_avx2
using BuildCompDiffwtdMaskD16Func =
    decltype(&svt_av1_build_compound_diffwtd_mask_d16_c);

using BuildCompDiffwtdMaskD16Param =
    std::tuple<int, BuildCompDiffwtdMaskD16Func, BlockSize>;

class BuildCompDiffwtdMaskD16Test
    : public ::testing::TestWithParam<BuildCompDiffwtdMaskD16Param> {
  public:
  protected:
    void run_test() {
        const auto tst_func = TEST_GET_PARAM(1);
        const int block_size = TEST_GET_PARAM(2);
        const int bd = TEST_GET_PARAM(0);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<uint8_t, 2 * MAX_SB_SQUARE> mask_ref;
        alignas(16) std::array<uint8_t, 2 * MAX_SB_SQUARE> mask_test;
        alignas(32) std::array<uint16_t, MAX_SB_SQUARE> src0;
        alignas(32) std::array<uint16_t, MAX_SB_SQUARE> src1;

        ConvolveParams conv_params =
            get_conv_params_no_round(0, NULL, 0, 1, bd);

        const auto in_precision = bd + 2 * FILTER_BITS - conv_params.round_0 -
                                  conv_params.round_1 + 2;
        const auto mask = (1 << in_precision) - 1;

        for (int i = 0; i < MAX_SB_SQUARE; i++) {
            src0[i] = rnd_.random() & mask;
            src1[i] = rnd_.random() & mask;
        }

        for (int mask_type = 0; mask_type < DIFFWTD_MASK_TYPES; mask_type++) {
            svt_av1_build_compound_diffwtd_mask_d16_c(
                mask_ref.data(),
                (DIFFWTD_MASK_TYPE)mask_type,
                src0.data(),
                width,
                src1.data(),
                width,
                height,
                width,
                &conv_params,
                bd);

            tst_func(mask_test.data(),
                     (DIFFWTD_MASK_TYPE)mask_type,
                     src0.data(),
                     width,
                     src1.data(),
                     width,
                     height,
                     width,
                     &conv_params,
                     bd);

            for (int r = 0; r < height; ++r) {
                for (int c = 0; c < width; ++c) {
                    ASSERT_EQ(mask_ref[c + r * width], mask_test[c + r * width])
                        << "Mismatch at unit tests for "
                           "BuildCompDiffwtdMaskD16Test\n"
                        << " Pixel mismatch at index "
                        << "[" << r << "," << c << "] "
                        << " @ " << width << "x" << height << " inv "
                        << mask_type;
                }
            }
        }
    }
    SVTRandom rnd_{16, false};
};  // class BuildCompDiffwtdMaskD16Test

TEST_P(BuildCompDiffwtdMaskD16Test, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, BuildCompDiffwtdMaskD16Test,
    ::testing::Combine(
        ::testing::Range(8, 11, 2),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_d16_sse4_1),
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, BuildCompDiffwtdMaskD16Test,
    ::testing::Combine(
        ::testing::Range(8, 11, 2),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_d16_avx2),
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, BuildCompDiffwtdMaskD16Test,
    ::testing::Combine(
        ::testing::Range(8, 11, 2),
        ::testing::Values(svt_av1_build_compound_diffwtd_mask_d16_neon),
        ::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL)));
#endif  // ARCH_AARCH64

using AomSseFunc = decltype(&svt_aom_sse_c);
using AomSseParam = std::tuple<BlockSize, AomSseFunc>;

class AomSseTest : public ::testing::TestWithParam<AomSseParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const auto test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> a_;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> b_;
        const int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            a_.fill(0);
            b_.fill(0);
            for (int j = 0; j < width * height; j++) {
                a_[j] = rnd_.random();
                b_[j] = rnd_.random();
            }

            int64_t res_ref = svt_aom_sse_c(
                a_.data(), width, b_.data(), width, height, width);
            int64_t res_tst =
                test_impl(a_.data(), width, b_.data(), width, height, width);

            ASSERT_EQ(res_ref, res_tst);
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(AomSseTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, AomSseTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_sse_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AomSseTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_sse_neon)));

#if HAVE_NEON_DOTPROD
INSTANTIATE_TEST_SUITE_P(
    NEON_DOTPROD, AomSseTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_sse_neon_dotprod)));
#endif  // HAVE_NEON_DOTPROD
#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

class AomSseHighbdTest : public ::testing::TestWithParam<AomSseParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const auto test_impl = TEST_GET_PARAM(1);
        int run_times = 100;
        int width;
        int height;
        if (block_size < BLOCK_SIZES_ALL) {
            width = block_size_wide[block_size];
            height = block_size_high[block_size];
        } else {
            run_times = 10;
            // unusual sizes
            if (block_size > BLOCK_SIZES_ALL) {
                // block_size == BLOCK_SIZES_ALL +1
                width = 36;
                height = 36;
            } else {
                // block_size == BLOCK_SIZES_ALL
                width = 40;
                height = 40;
            }
        }

        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> a_;
        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> b_;
        for (int i = 0; i < run_times; ++i) {
            a_.fill(0);
            b_.fill(0);
            for (int j = 0; j < width * height; j++) {
                a_[j] = rnd_.random();
                b_[j] = rnd_.random();
            }

            int64_t res_ref =
                svt_aom_highbd_sse_c(reinterpret_cast<uint8_t *>(a_.data()),
                                     width,
                                     reinterpret_cast<uint8_t *>(b_.data()),
                                     width,
                                     height,
                                     width);
            int64_t res_tst = test_impl(reinterpret_cast<uint8_t *>(a_.data()),
                                        width,
                                        reinterpret_cast<uint8_t *>(b_.data()),
                                        width,
                                        height,
                                        width);

            ASSERT_EQ(res_ref, res_tst)
                << "Mismatch: ref = " << res_ref << ", test = " << res_tst
                << ", width = " << width << ", height = " << height << "\n";
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AomSseHighbdTest);

TEST_P(AomSseHighbdTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    AVX2, AomSseHighbdTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4,
                                        (BlockSize)(BLOCK_SIZES_ALL + 2)),
                       ::testing::Values(svt_aom_highbd_sse_avx2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AomSseHighbdTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4,
                                        (BlockSize)(BLOCK_SIZES_ALL + 2)),
                       ::testing::Values(svt_aom_highbd_sse_neon)));
#if HAVE_SVE
INSTANTIATE_TEST_SUITE_P(
    SVE, AomSseHighbdTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4,
                                        (BlockSize)(BLOCK_SIZES_ALL + 2)),
                       ::testing::Values(svt_aom_highbd_sse_sve)));
#endif  // HAVE_SVE
#endif  // ARCH_AARCH64
#endif  // CONFIG_ENABLE_HIGH_BIT_DEPTH

using AomSubtractBlockFunc = decltype(&svt_aom_subtract_block_c);
using AomSubtractBlockParam = std::tuple<BlockSize, AomSubtractBlockFunc>;

class AomSubtractBlockTest
    : public ::testing::TestWithParam<AomSubtractBlockParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const auto test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> diff_ref_;
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> diff_tst_;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> src_;
        alignas(16) std::array<uint8_t, MAX_SB_SQUARE> pred_;
        const int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            src_.fill(0);
            pred_.fill(0);
            for (int j = 0; j < width * height; j++) {
                src_[j] = rnd_.random();
                pred_[j] = rnd_.random();
            }
            diff_ref_.fill(0);
            diff_tst_.fill(0);

            svt_aom_subtract_block_c(width,
                                     height,
                                     diff_ref_.data(),
                                     width,
                                     src_.data(),
                                     width,
                                     pred_.data(),
                                     width);
            test_impl(width,
                      height,
                      diff_tst_.data(),
                      width,
                      src_.data(),
                      width,
                      pred_.data(),
                      width);

            ASSERT_EQ(diff_ref_, diff_tst_);
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(AomSubtractBlockTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, AomSubtractBlockTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_subtract_block_sse2)));

INSTANTIATE_TEST_SUITE_P(
    AVX2, AomSubtractBlockTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_subtract_block_avx2)));
INSTANTIATE_TEST_SUITE_P(
    SSE4_1, AomSubtractBlockTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_subtract_block_sse4_1)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AomSubtractBlockTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_subtract_block_neon)));
#endif  // ARCH_AARCH64

#if CONFIG_ENABLE_HIGH_BIT_DEPTH

using AomHighbdSubtractBlockFunc = decltype(&svt_aom_highbd_subtract_block_c);
using AomHighbdSubtractBlockParam =
    std::tuple<BlockSize, AomHighbdSubtractBlockFunc>;

class AomHighbdSubtractBlockTest
    : public ::testing::TestWithParam<AomHighbdSubtractBlockParam> {
  public:
    void run_test() {
        const int block_size = TEST_GET_PARAM(0);
        const auto test_impl = TEST_GET_PARAM(1);
        const int width = block_size_wide[block_size];
        const int height = block_size_high[block_size];
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> diff_ref_;
        alignas(16) std::array<int16_t, MAX_SB_SQUARE> diff_tst_;
        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> src_;
        alignas(16) std::array<uint16_t, MAX_SB_SQUARE> pred_;
        constexpr int run_times = 100;
        for (int i = 0; i < run_times; ++i) {
            src_.fill(0);
            pred_.fill(0);
            for (int j = 0; j < width * height; j++) {
                src_[j] = rnd_.random();
                pred_[j] = rnd_.random();
            }
            diff_ref_.fill(0);
            diff_tst_.fill(0);

            svt_aom_highbd_subtract_block_c(
                width,
                height,
                diff_ref_.data(),
                width,
                reinterpret_cast<uint8_t *>(src_.data()),
                width,
                reinterpret_cast<uint8_t *>(pred_.data()),
                width,
                8);  // last parameter is unused
            test_impl(width,
                      height,
                      diff_tst_.data(),
                      width,
                      reinterpret_cast<uint8_t *>(src_.data()),
                      width,
                      reinterpret_cast<uint8_t *>(pred_.data()),
                      width,
                      8);

            ASSERT_EQ(diff_ref_, diff_tst_);
        }
    }

  private:
    SVTRandom rnd_{0, 255};
};

TEST_P(AomHighbdSubtractBlockTest, MatchTest) {
    run_test();
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE2, AomHighbdSubtractBlockTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_highbd_subtract_block_sse2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
INSTANTIATE_TEST_SUITE_P(
    NEON, AomHighbdSubtractBlockTest,
    ::testing::Combine(::testing::Range(BLOCK_4X4, BLOCK_SIZES_ALL),
                       ::testing::Values(svt_aom_highbd_subtract_block_neon)));
#endif  // ARCH_AARCH64

#endif

}  // namespace
