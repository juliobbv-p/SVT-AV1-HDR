/*
 * Copyright (c) 2026, Alliance for Open Media. All rights reserved
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
 * @file HashTest.cc
 *
 * @brief Unit test for the hardware CRC-32C kernels used by hash-based
 * motion estimation:
 * - svt_av1_get_crc32c_value_arm_crc32
 * - svt_av1_get_crc32c_value_sse4_2
 *
 ******************************************************************************/

#include <cinttypes>
#include <cstring>
#include <tuple>

#include "gtest/gtest.h"
#include "aom_dsp_rtcd.h"
#include "common_dsp_rtcd.h"
#include "definitions.h"
#include "hash.h"
#include "random.h"
#include "svt_time.h"

using svt_av1_test_tool::SVTRandom;

namespace {

uint32_t crc32c_ref(const uint8_t *buf, size_t len) {
    return svt_av1_get_crc32c_value_c(buf, len);
}

// CRC-32C check values, e.g. from RFC 3720 (iSCSI) appendix B.4.
TEST(HashCrc32cTest, KnownAnswer) {
    svt_av1_crc32c_table_init();

    const uint8_t check[] = "123456789";
    EXPECT_EQ(0xE3069283u, crc32c_ref(check, 9));

    const uint8_t zeros[32] = {0};
    EXPECT_EQ(0x8A9136AAu, crc32c_ref(zeros, 32));

    uint8_t ones[32];
    memset(ones, 0xFF, sizeof(ones));
    EXPECT_EQ(0x62A8AB43u, crc32c_ref(ones, 32));

    EXPECT_EQ(0x00000000u, crc32c_ref(check, 0));
}

using CrcFunc = uint32_t (*)(const uint8_t *buf, size_t len);
// hardware kernel, CPU flag it requires
using CrcParam = std::tuple<CrcFunc, EbCpuFlags>;

class HashCrc32cHwTest : public ::testing::TestWithParam<CrcParam> {
  public:
    HashCrc32cHwTest() : test_func_(std::get<0>(GetParam())) {
    }

    ~HashCrc32cHwTest() override = default;

  protected:
    void SetUp() override {
        if (!(svt_aom_get_cpu_flags() & std::get<1>(GetParam())))
            GTEST_SKIP() << "Hardware CRC32C not supported on this CPU";
        svt_av1_crc32c_table_init();
    }

    CrcFunc test_func_;
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(HashCrc32cHwTest);

TEST_P(HashCrc32cHwTest, KnownAnswer) {
    const uint8_t check[] = "123456789";
    EXPECT_EQ(0xE3069283u, test_func_(check, 9));
    EXPECT_EQ(0x00000000u, test_func_(check, 0));
}

TEST_P(HashCrc32cHwTest, MatchC) {
    constexpr size_t max_len = 1024;
    constexpr size_t max_offset = 16;
    uint8_t buf[max_len + max_offset];
    SVTRandom rnd(8, false);
    for (size_t i = 0; i < sizeof(buf); i++)
        buf[i] = rnd.Rand8();

    // All lengths up to a couple of 8-byte blocks to cover the head/tail
    // handling, plus some longer ones; all at varying alignments.
    for (size_t offset = 0; offset < max_offset; offset++) {
        for (size_t len = 0; len <= max_len;
             len = len < 32 ? len + 1 : len * 2) {
            const uint32_t ref_crc = crc32c_ref(buf + offset, len);
            const uint32_t hw_crc = test_func_(buf + offset, len);
            ASSERT_EQ(ref_crc, hw_crc)
                << "CRC mismatch at offset " << offset << " length " << len;
        }
    }
}

TEST_P(HashCrc32cHwTest, DISABLED_Speed) {
    // 16 bytes is what svt_av1_generate_block_hash_value and
    // svt_av1_get_block_hash_value hash on every call.
    const size_t lens[] = {16, 64, 1024};
    SVTRandom rnd(8, false);

    for (size_t len : lens) {
        uint8_t buf[1024];
        for (size_t i = 0; i < len; i++)
            buf[i] = rnd.Rand8();

        const uint64_t num_iter = 1000000000 / (len * 10);
        double time_c, time_o;
        uint64_t start_time_seconds, start_time_useconds;
        uint64_t finish_time_seconds, finish_time_useconds;

        uint32_t sum_c = 0;
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (uint64_t i = 0; i < num_iter; i++) {
            buf[0] = (uint8_t)i;
            sum_c += crc32c_ref(buf, len);
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_c = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);

        uint32_t sum_o = 0;
        svt_av1_get_time(&start_time_seconds, &start_time_useconds);
        for (uint64_t i = 0; i < num_iter; i++) {
            buf[0] = (uint8_t)i;
            sum_o += test_func_(buf, len);
        }
        svt_av1_get_time(&finish_time_seconds, &finish_time_useconds);
        time_o = svt_av1_compute_overall_elapsed_time_ms(start_time_seconds,
                                                         start_time_useconds,
                                                         finish_time_seconds,
                                                         finish_time_useconds);

        EXPECT_EQ(sum_c, sum_o);
        printf("len = %4zu, iterations = %" PRIu64
               ": c_time = %f \t o_time = %f \t Gain = %4.2f\n",
               len,
               num_iter,
               time_c,
               time_o,
               time_c / time_o);
    }
}

#ifdef ARCH_X86_64
INSTANTIATE_TEST_SUITE_P(
    SSE4_2, HashCrc32cHwTest,
    ::testing::Values(std::make_tuple(svt_av1_get_crc32c_value_sse4_2,
                                      EB_CPU_FLAGS_SSE4_2)));
#endif  // ARCH_X86_64

#ifdef ARCH_AARCH64
#if HAVE_ARM_CRC32
INSTANTIATE_TEST_SUITE_P(
    ARM_CRC32, HashCrc32cHwTest,
    ::testing::Values(std::make_tuple(svt_av1_get_crc32c_value_arm_crc32,
                                      EB_CPU_FLAGS_ARM_CRC32)));
#endif  // HAVE_ARM_CRC32
#endif  // ARCH_AARCH64

}  // namespace
