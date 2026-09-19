/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved.
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "common_dsp_rtcd.h"

#include <cstdio>
#include <cstdlib>

#include "third_party/googletest/include/gtest/gtest.h"

// The test sources themselves must be compiled in test mode. If this fires, the
// CMake configure did not set SVT_AV1_UNIT_TEST_BUILD (BUILD_TESTING=ON).
#ifndef SVT_AV1_UNIT_TEST_BUILD
#error \
    "SvtAv1UnitTests must be built with SVT_AV1_UNIT_TEST_BUILD defined " \
    "(configure with -DBUILD_TESTING=ON)."
#endif

#if ARCH_AARCH64 || ARCH_X86_64
static void append_negative_gtest_filter(const char *str) {
    std::string flag_value = GTEST_FLAG_GET(filter);
    // Negative patterns begin with one '-' followed by a ':' separated list.
    if (flag_value.find('-') == std::string::npos)
        flag_value += '-';
    // OPT.* matches TEST() functions
    // OPT/* matches TEST_P() functions
    // OPT_* matches tests which have been manually sharded.
    // We do not match OPT* because of SSE/SSE2 collisions.
    const char *search_terminators = "./_";
    for (size_t pos = 0; pos < strlen(search_terminators); ++pos) {
        flag_value += ":";
        flag_value += "*";
        flag_value += str;
        flag_value += search_terminators[pos];
        flag_value += "*";
    }
    GTEST_FLAG_SET(filter, flag_value);
}
#endif  // ARCH_AARCH64 || ARCH_X86_64

// Scan-order tables are generated at runtime (svt_aom_init_iscan) rather
// than shipped as constants; the encoder init path does this, but the unit
// tests exercise kernels directly, so do it here.
extern "C" void svt_aom_init_iscan(void);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    svt_aom_init_iscan();

    // Defensive handshake: ensure the linked library was also built in test
    // mode. A deployment library has CONFIG_ARM_NEON_IS_GUARANTEED=1, which
    // strips the C reference functions; svt_aom_setup_*_rtcd_internal(0) would
    // then leave the RTCD table unassigned ("Pointer ... is not assigned" ->
    // SIGSEGV). Catch that here with a clear message instead.
    if (!svt_aom_library_built_for_unit_tests()) {
        fprintf(stderr,
                "FATAL: SvtAv1 library was NOT built with "
                "SVT_AV1_UNIT_TEST_BUILD.\n"
                "The test binary is linked against a deployment-mode library "
                "where C reference\nfunctions are stripped, which breaks "
                "svt_aom_setup_*_rtcd_internal(0).\n"
                "Rebuild the library in the same configure with "
                "-DBUILD_TESTING=ON.\n");
        return EXIT_FAILURE;
    }

#if ARCH_AARCH64
    const EbCpuFlags caps = svt_aom_get_cpu_flags_to_use();
    // Note: Disable testing for Neoverse V2-only implementations if the
    // required ISA extensions don't exist on the test platform. Currently the
    // required ISA extensions in the Neoverse V2-only paths are NEON_DOTPROD
    // and SVE.
    if (!(caps & EB_CPU_FLAGS_ARM_CRC32))
        append_negative_gtest_filter("ARM_CRC32");
    if (!(caps & EB_CPU_FLAGS_NEON_DOTPROD)) {
        append_negative_gtest_filter("NEON_DOTPROD");
        append_negative_gtest_filter("NEOVERSE_V2");
    }
    if (!(caps & EB_CPU_FLAGS_NEON_I8MM))
        append_negative_gtest_filter("NEON_I8MM");
    if (!(caps & EB_CPU_FLAGS_SVE)) {
        append_negative_gtest_filter("SVE");
        append_negative_gtest_filter("NEOVERSE_V2");
    }
    if (!(caps & EB_CPU_FLAGS_SVE2))
        append_negative_gtest_filter("SVE2");
#endif  // ARCH_AARCH64

#if ARCH_X86_64
    const EbCpuFlags simd_caps = svt_aom_get_cpu_flags_to_use();
    if (!(simd_caps & EB_CPU_FLAGS_MMX))
        append_negative_gtest_filter("MMX");
    if (!(simd_caps & EB_CPU_FLAGS_SSE))
        append_negative_gtest_filter("SSE");
    if (!(simd_caps & EB_CPU_FLAGS_SSE2))
        append_negative_gtest_filter("SSE2");
    if (!(simd_caps & EB_CPU_FLAGS_SSE3))
        append_negative_gtest_filter("SSE3");
    if (!(simd_caps & EB_CPU_FLAGS_SSSE3))
        append_negative_gtest_filter("SSSE3");
    if (!(simd_caps & EB_CPU_FLAGS_SSE4_1))
        append_negative_gtest_filter("SSE4_1");
    if (!(simd_caps & EB_CPU_FLAGS_SSE4_2))
        append_negative_gtest_filter("SSE4_2");
    if (!(simd_caps & EB_CPU_FLAGS_AVX))
        append_negative_gtest_filter("AVX");
    if (!(simd_caps & EB_CPU_FLAGS_AVX2))
        append_negative_gtest_filter("AVX2");
    if (!(simd_caps & EB_CPU_FLAGS_AVX512F))
        append_negative_gtest_filter("AVX512");
    if (!(simd_caps & EB_CPU_FLAGS_AVX512ICL))
        append_negative_gtest_filter("AVX512ICL");
#endif  // ARCH_X86_64

    return RUN_ALL_TESTS();
}
