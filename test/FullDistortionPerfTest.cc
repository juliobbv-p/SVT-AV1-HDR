/*
 * Copyright (c) 2026, Meta Platforms, Inc. and affiliates.
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
 * FullDistortionPerfTest.cc
 *
 * Per-(W, H) microbenchmark for the 32-bit full-distortion kernel:
 *   - svt_full_distortion_kernel32_bits_c      (scalar reference)
 *   - svt_full_distortion_kernel32_bits_neon   (SIMD; in an 8-bit/RTC build
 *this is the int16-range-optimized specialization, otherwise the generic
 *kernel)
 *
 * Both compute, in the transform-coefficient domain:
 *   result[DIST_CALC_RESIDUAL]   = sum (coeff - recon_coeff)^2
 *   result[DIST_CALC_PREDICTION] = sum  coeff^2
 *
 * The test feeds int16-range data (valid for both the generic and the 8-bit
 * specialization), checks the NEON output is bit-identical to the C reference,
 * and reports ns/call plus the NEON speedup over C.
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "definitions.h"
#include "gtest/gtest.h"
#include "svt_malloc.h"

#if ARCH_AARCH64

extern "C" {
void svt_full_distortion_kernel32_bits_c(
    int32_t *coeff, int32_t *recon_coeff, uint32_t stride, uint32_t area_width,
    uint32_t area_height, uint64_t distortion_result[DIST_CALC_TOTAL]);
void svt_full_distortion_kernel32_bits_neon(
    int32_t *coeff, int32_t *recon_coeff, uint32_t stride, uint32_t area_width,
    uint32_t area_height, uint64_t distortion_result[DIST_CALC_TOTAL]);
}

namespace {

using DistFn = void (*)(int32_t *coeff, int32_t *recon_coeff, uint32_t stride,
                        uint32_t area_width, uint32_t area_height,
                        uint64_t distortion_result[DIST_CALC_TOTAL]);

// Number of warm-up runs per cell before timed measurement begins.
constexpr int kWarmupRuns = 3;

// Reproducible seed for coeff/recon buffers.
constexpr uint64_t kSeed = 0xD15705710Full;

// Max |value| for an 8-bit-depth transform coefficient (int16 range).
// Kept at 32767 so (coeff - recon)^2 stays < 2^32 (fits the uint32 squares of
// the 8-bit NEON specialization), and valid for the generic kernel too.
constexpr int32_t kMaxAbs8bd = 32767;

// Volatile sink to prevent dead-code elimination of the kernel calls.
volatile uint64_t g_sink = 0;

struct KernelEntry {
    const char *name;
    DistFn fn;
};

double run_cell(const KernelEntry &k, int32_t *coeff, int32_t *recon,
                uint32_t stride, int w, int h) {
    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    const int iters = 1000000000 / (w * h);

    uint64_t local_sink = 0;

    for (int b = 0; b < kWarmupRuns; ++b) {
        uint64_t acc = 0;
        for (int i = 0; i < iters; ++i) {
            std::array<uint64_t, DIST_CALC_TOTAL> res{};
            k.fn(coeff, recon, stride, w, h, res.data());
            acc += res[DIST_CALC_RESIDUAL] + res[DIST_CALC_PREDICTION];
        }
        local_sink += acc;
    }

    uint64_t acc = 0;
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        std::array<uint64_t, DIST_CALC_TOTAL> res{};
        k.fn(coeff, recon, stride, w, h, res.data());
        acc += res[DIST_CALC_RESIDUAL] + res[DIST_CALC_PREDICTION];
    }
    auto t1 = clock::now();
    local_sink += acc;
    g_sink += local_sink;

    const double total_ns =
        static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count());
    const double avg_ns = total_ns / static_cast<double>(iters);

    std::cout << "  " << std::left << std::setw(46) << k.name
              << " W=" << std::right << std::setw(3) << w
              << " H=" << std::setw(3) << h << "  iters=" << std::setw(8)
              << iters << "  avg=" << std::fixed << std::setprecision(2)
              << std::setw(9) << avg_ns << " ns" << std::endl;
    return avg_ns;
}

class FullDistortionPerf : public ::testing::Test {
  protected:
    void SetUp() override {
        constexpr int max_dim = 128;
        // Mildly asymmetric strides to mimic real (padded) coeff buffers.
        coeff_stride_ = max_dim;
        recon_stride_ = max_dim;
        const size_t coeff_sz = static_cast<size_t>(coeff_stride_) * max_dim;
        const size_t recon_sz = static_cast<size_t>(recon_stride_) * max_dim;
        coeff_ = static_cast<int32_t *>(
            svt_aom_memalign(64, coeff_sz * sizeof(int32_t)));
        recon_ = static_cast<int32_t *>(
            svt_aom_memalign(64, recon_sz * sizeof(int32_t)));
        ASSERT_NE(coeff_, nullptr);
        ASSERT_NE(recon_, nullptr);

        std::mt19937 rng(static_cast<uint32_t>(kSeed));
        std::uniform_int_distribution<int32_t> dist(-kMaxAbs8bd, kMaxAbs8bd);
        for (size_t i = 0; i < coeff_sz; ++i) {
            coeff_[i] = dist(rng);
        }
        for (size_t i = 0; i < recon_sz; ++i) {
            recon_[i] = dist(rng);
        }
    }

    void TearDown() override {
        if (coeff_) {
            svt_aom_free(coeff_);
            coeff_ = nullptr;
        }
        if (recon_) {
            svt_aom_free(recon_);
            recon_ = nullptr;
        }
    }

    int32_t *coeff_ = nullptr;
    int32_t *recon_ = nullptr;
    uint32_t coeff_stride_ = 0;
    uint32_t recon_stride_ = 0;
};

// All transform-block shapes the kernel can actually be called with.
// SVT clamps BOTH dimensions to 32 (callers pass `dim < 64 ? dim : 32`), and
// AV1 transforms have aspect ratio at most 1:4, so width,height in {4,8,16,32}
// with ratio <= 4 -> 14 shapes (4x32 and 32x4 do not exist).
const std::array<std::pair<int, int>, 14> kShapes = {{{4, 4},
                                                      {4, 8},
                                                      {4, 16},
                                                      {8, 4},
                                                      {8, 8},
                                                      {8, 16},
                                                      {8, 32},
                                                      {16, 4},
                                                      {16, 8},
                                                      {16, 16},
                                                      {16, 32},
                                                      {32, 8},
                                                      {32, 16},
                                                      {32, 32}}};

TEST_F(FullDistortionPerf, DISABLED_Shapes) {
    const KernelEntry ref_c{"svt_full_distortion_kernel32_bits_c",
                            &svt_full_distortion_kernel32_bits_c};
    const KernelEntry simd{"svt_full_distortion_kernel32_bits_neon",
                           &svt_full_distortion_kernel32_bits_neon};

    // Production layout is dense: the wrapper passes one stride == transform
    // width (coeffs are packed at the TX width).
    // Correctness gate: NEON must be bit-identical to the C reference for
    // int16-range inputs before any timing number is meaningful.
    for (const auto &s : kShapes) {
        const int w = s.first, h = s.second;
        std::array<uint64_t, DIST_CALC_TOTAL> ref{}, mod{};
        ref_c.fn(coeff_, recon_, w, w, h, ref.data());
        simd.fn(coeff_, recon_, w, w, h, mod.data());
        ASSERT_EQ(ref, mod) << "NEON mismatch at W=" << w << " H=" << h;
    }

    double sum_c = 0.0, sum_neon = 0.0;

    std::cout << std::endl
              << "  --- per-shape timing (stride == width, dense) ---"
              << std::endl;
    for (const auto &s : kShapes) {
        const int w = s.first, h = s.second;
        const double c = run_cell(ref_c, coeff_, recon_, w, w, h);
        const double n = run_cell(simd, coeff_, recon_, w, w, h);
        sum_c += c;
        sum_neon += n;
        std::cout << "    -> W=" << std::right << std::setw(3) << w
                  << " H=" << std::setw(3) << h
                  << "  speedup(neon/c)=" << std::fixed << std::setprecision(2)
                  << (n > 0 ? c / n : 0.0) << "x" << std::endl;
    }

    const double overall = (sum_neon > 0) ? (sum_c / sum_neon) : 0.0;
    std::cout << std::endl
              << "  === overall speedup (sum c / sum neon) = " << std::fixed
              << std::setprecision(3) << overall << "x ===" << std::endl;
    std::cout << "  sink=0x" << std::hex << std::setw(16) << std::setfill('0')
              << static_cast<unsigned long long>(g_sink) << std::dec
              << std::setfill(' ') << std::endl;
}

}  // namespace

#endif  // ARCH_AARCH64
