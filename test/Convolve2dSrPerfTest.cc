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
 * Convolve2dSrPerfTest.cc
 *
 * Microbenchmark for convolve _sr functions under all available NEON variants:
 *   - svt_av1_convolve_2d_sr_{neon, neon_dotprod, neon_i8mm}
 *   - svt_av1_convolve_x_sr_{neon, neon_dotprod, neon_i8mm}
 *   - svt_av1_convolve_y_sr_{neon, neon_dotprod, neon_i8mm}
 *
 * Two DISABLED tests:
 *   - M11HotCells   — small curated grid that covers most M11 LPD1 weight
 *                     (block shapes × subpel offsets, all REGULAR filter).
 *   - AllTapsSweep  — synthetic sweep over 4/6/8-tap-effective filters
 *                     across canonical block shapes.
 *
 * Opt in via:
 *   --gtest_also_run_disabled_tests --gtest_filter='*Convolve2dSrPerf*'
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "aom_dsp_rtcd.h"
#include "convolve.h"
#include "definitions.h"
#include "filter.h"
#include "gtest/gtest.h"
#include "inter_prediction.h"
#include "utility.h"

#if ARCH_AARCH64

namespace {

using ConvolveSrFn = void (*)(
    const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride,
    int32_t w, int32_t h, const InterpFilterParams *filter_params_x,
    const InterpFilterParams *filter_params_y, const int32_t subpel_x_qn,
    const int32_t subpel_y_qn, ConvolveParams *conv_params);

enum FnId : int { FN_2D_SR = 0, FN_X_SR = 1, FN_Y_SR = 2, FN_COUNT = 3 };
enum FnVariant : int {
    VAR_NEON = 0,
    VAR_DOTPROD = 1,
    VAR_I8MM = 2,
    VAR_COUNT = 3,
};
enum TapKind : int {
    TAP_4 = 0,
    TAP_6 = 1,
    TAP_8 = 2,
    TAP_KIND_COUNT = 3,
};

constexpr int kMedianRuns = 5;
constexpr int kWarmupRuns = 1;
constexpr uint64_t kSeed = 0xC0FFEE12345ULL;

// Volatile sink to defeat dead-code elimination.
static volatile uint64_t g_sink = 0;

static const char *fn_short(int fn) {
    switch (fn) {
    case FN_2D_SR: return "2d_sr";
    case FN_X_SR: return "x_sr";
    case FN_Y_SR: return "y_sr";
    default: return "?";
    }
}

static const char *variant_short(int v) {
    switch (v) {
    case VAR_NEON: return "neon";
    case VAR_DOTPROD: return "dotprod";
    case VAR_I8MM: return "i8mm";
    default: return "?";
    }
}

static const char *tap_short(int t) {
    switch (t) {
    case TAP_4: return "4tap";
    case TAP_6: return "6tap";
    case TAP_8: return "8tap";
    default: return "?";
    }
}

// Pick the AV1 filter type that matches a tap-effective dispatch path.
// (Used as documentation; tap-specific shape choice is in the test below.)

static ConvolveSrFn fn_pointer(int fn, int variant) {
    switch (variant) {
    case VAR_NEON:
        switch (fn) {
        case FN_2D_SR: return &svt_av1_convolve_2d_sr_neon;
        case FN_X_SR: return &svt_av1_convolve_x_sr_neon;
        case FN_Y_SR: return &svt_av1_convolve_y_sr_neon;
        default: return nullptr;
        }
    case VAR_DOTPROD:
        switch (fn) {
        case FN_2D_SR: return &svt_av1_convolve_2d_sr_neon_dotprod;
        case FN_X_SR: return &svt_av1_convolve_x_sr_neon_dotprod;
        // y_sr_neon_dotprod is ~1.44x slower than NEON (transpose+sdot); the
        // encoder uses the NEON fn for the dotprod slot, so measure that.
        case FN_Y_SR: return &svt_av1_convolve_y_sr_neon;
        default: return nullptr;
        }
    case VAR_I8MM:
#if HAVE_NEON_I8MM
        switch (fn) {
        case FN_2D_SR: return &svt_av1_convolve_2d_sr_neon_i8mm;
        case FN_X_SR: return &svt_av1_convolve_x_sr_neon_i8mm;
        case FN_Y_SR: return &svt_av1_convolve_y_sr_neon_i8mm;
        default: return nullptr;
        }
#else
        return nullptr;
#endif
    default: return nullptr;
    }
}

struct Cell {
    int fn;
    int w, h;
    int sx, sy;  // subpel offsets (use any in-range value for unused axis)
    InterpFilter hfilter;  // x-axis filter (use REGULAR for unused axis)
    InterpFilter vfilter;  // y-axis filter
};

static double run_once(const Cell &c, int variant, const uint8_t *src,
                       int src_stride, uint8_t *dst, int dst_stride,
                       int iters) {
    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;

    const InterpFilterParams fx =
        av1_get_interp_filter_params_with_block_size(c.hfilter, c.w);
    const InterpFilterParams fy =
        av1_get_interp_filter_params_with_block_size(c.vfilter, c.h);

    ConvolveSrFn fn = fn_pointer(c.fn, variant);
    if (!fn)
        return 0.0;
    ConvolveParams conv_params = get_conv_params_no_round(0, nullptr, 0, 0, 8);

    uint64_t acc = 0;
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) {
        fn(src,
           src_stride,
           dst,
           dst_stride,
           c.w,
           c.h,
           &fx,
           &fy,
           c.sx,
           c.sy,
           &conv_params);
        acc += static_cast<uint64_t>(dst[0]);
    }
    auto t1 = clock::now();
    g_sink += acc;

    const double total_ns =
        static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count());
    return total_ns / static_cast<double>(iters);
}

static double median_ns(const Cell &c, int variant, const uint8_t *src,
                        int src_stride, uint8_t *dst, int dst_stride) {
    // Target ~5 ms per timed batch.
    int iters = 5000000 / (c.w * c.h);
    if (iters < 64)
        iters = 64;

    for (int b = 0; b < kWarmupRuns; ++b)
        (void)run_once(c, variant, src, src_stride, dst, dst_stride, iters);

    std::array<double, kMedianRuns> ns{};
    for (int r = 0; r < kMedianRuns; ++r)
        ns[r] = run_once(c, variant, src, src_stride, dst, dst_stride, iters);
    std::sort(ns.begin(), ns.end());
    return ns[kMedianRuns / 2];
}

// --- Shared block-shape sets ------------------------------------------------
// AV1 LPD1 hot shapes (covers >95% of M11 weight). Used for M11HotCells +
// 6-tap/8-tap sweeps. 4x4 is excluded here because with REGULAR/SMOOTH it
// dispatches to the fused 4-tap path (per-axis 4-tap table selected when
// w<=4), which would contaminate the 6-tap measurement.
static const std::pair<int, int> kHotShapes[] = {
    {8, 4},
    {4, 8},
    {8, 8},
    {16, 8},
    {8, 16},
    {16, 16},
    {32, 16},
    {16, 32},
    {32, 32},
    {64, 32},
    {32, 64},
    {64, 64},
};

// 4-tap path requires BOTH axes to have <=4 effective taps. With REGULAR /
// SMOOTH this only happens when w<=4 AND h<=4 (both axes select the 4-tap
// filter table per av1_get_interp_filter_params_with_block_size). Hence the
// only AV1 shape that exercises the fused 4-tap dispatch is 4x4.
static const std::pair<int, int> k4TapShapes[] = {{4, 4}};

// Representative subpel offsets across the [0, 16) range. 8 is the center
// (non-fractional row 8 of the filter table). Avoids subpel=0 because that
// would degenerate to copy / vertical-only / horizontal-only paths handled
// elsewhere by the dispatcher.
static const int kHotSubpels[] = {2, 4, 6, 8, 10, 12, 14};

// Quarter-pel subpel offsets {4, 8, 12} — the dominant M11 fractional
// positions. Used by M11QuarterPelHotCells.
static const int kQuarterSubpels[] = {4, 8, 12};

// Shapes used by M11QuarterPelHotCells (the majority of M11 cells).
static const std::pair<int, int> kQuarterHotShapes[] = {
    {8, 8},
    {16, 8},
    {8, 16},
    {16, 16},
    {32, 16},
    {16, 32},
    {32, 32},
};

// --- Test fixture -----------------------------------------------------------
class Convolve2dSrPerf : public ::testing::Test {
  protected:
    void SetUp() override {
        constexpr int padding = 16;
        constexpr int max_w = 128 + 2 * padding;
        constexpr int max_h = 128 + 2 * padding;
        src_stride_ = max_w;
        dst_stride_ = MAX_SB_SIZE;

        const size_t src_sz = static_cast<size_t>(src_stride_) * max_h;
        const size_t dst_sz = static_cast<size_t>(dst_stride_) * MAX_SB_SIZE;
        src_alloc_ = static_cast<uint8_t *>(svt_aom_memalign(64, src_sz));
        dst_ = static_cast<uint8_t *>(svt_aom_memalign(64, dst_sz));
        ASSERT_NE(src_alloc_, nullptr);
        ASSERT_NE(dst_, nullptr);

        std::mt19937 rng(static_cast<uint32_t>(kSeed));
        std::uniform_int_distribution<int> dist(0, 255);
        for (size_t i = 0; i < src_sz; ++i)
            src_alloc_[i] = static_cast<uint8_t>(dist(rng));
        for (size_t i = 0; i < dst_sz; ++i)
            dst_[i] = static_cast<uint8_t>(dist(rng));

        src_ = src_alloc_ + padding * src_stride_ + padding;
        svt_aom_setup_rtcd_internal(EB_CPU_FLAGS_ALL);
    }

    void TearDown() override {
        if (src_alloc_) {
            svt_aom_free(src_alloc_);
            src_alloc_ = nullptr;
            src_ = nullptr;
        }
        if (dst_) {
            svt_aom_free(dst_);
            dst_ = nullptr;
        }
    }

    // Returns the set of available variants on this CPU.
    std::vector<int> available_variants() const {
        std::vector<int> v;
        v.push_back(VAR_NEON);
        const EbCpuFlags cpu = svt_aom_get_cpu_flags();
        if (cpu & EB_CPU_FLAGS_NEON_DOTPROD)
            v.push_back(VAR_DOTPROD);
#if HAVE_NEON_I8MM
        if (cpu & EB_CPU_FLAGS_NEON_I8MM)
            v.push_back(VAR_I8MM);
#endif
        return v;
    }

    // Iterate all valid cells for a given fn over (shape, subpel) and
    // accumulate total median ns into out[].
    void sweep_fn(int fn, InterpFilter filter,
                  const std::pair<int, int> *shapes, int nshapes,
                  const int *subpels, int nsub,
                  const std::vector<int> &variants, double out[VAR_COUNT],
                  int &cells_out) {
        const bool has_sx = (fn == FN_2D_SR || fn == FN_X_SR);
        const bool has_sy = (fn == FN_2D_SR || fn == FN_Y_SR);

        const int sx_count = has_sx ? nsub : 1;
        const int sy_count = has_sy ? nsub : 1;

        for (int si = 0; si < nshapes; ++si) {
            const auto &sh = shapes[si];
            for (int sxi = 0; sxi < sx_count; ++sxi) {
                for (int syi = 0; syi < sy_count; ++syi) {
                    Cell c{};
                    c.fn = fn;
                    c.w = sh.first;
                    c.h = sh.second;
                    c.sx = has_sx ? subpels[sxi] : 8;
                    c.sy = has_sy ? subpels[syi] : 8;
                    c.hfilter = filter;
                    c.vfilter = filter;
                    for (int v : variants)
                        out[v] += median_ns(
                            c, v, src_, src_stride_, dst_, dst_stride_);
                    cells_out += 1;
                }
            }
        }
    }

    uint8_t *src_alloc_ = nullptr;
    uint8_t *src_ = nullptr;
    uint8_t *dst_ = nullptr;
    int src_stride_ = 0;
    int dst_stride_ = 0;
};

// --- Helpers for pretty output ---------------------------------------------
static void print_section(const std::string &title) {
    std::cout << std::endl
              << "================ " << title
              << " ================" << std::endl;
}

static void print_totals(const std::vector<int> &variants,
                         const double tot_ns[FN_COUNT][VAR_COUNT],
                         const int cells[FN_COUNT][VAR_COUNT]) {
    for (int f = 0; f < FN_COUNT; ++f) {
        for (int v : variants) {
            if (cells[f][v] == 0)
                continue;
            std::cout << "  " << std::left << std::setw(5) << fn_short(f) << "/"
                      << std::setw(8) << variant_short(v)
                      << "  cells=" << std::setw(4) << std::right << cells[f][v]
                      << "  total_med_ns=" << std::scientific
                      << std::setprecision(4) << tot_ns[f][v] << std::endl;
        }
    }
    // Ratios relative to NEON.
    for (int f = 0; f < FN_COUNT; ++f) {
        if (cells[f][VAR_NEON] == 0 || tot_ns[f][VAR_NEON] <= 0)
            continue;
        for (int v : variants) {
            if (v == VAR_NEON || cells[f][v] == 0)
                continue;
            const double ratio = tot_ns[f][v] / tot_ns[f][VAR_NEON];
            std::cout << "    " << fn_short(f) << "  " << variant_short(v)
                      << " / neon = " << std::fixed << std::setprecision(3)
                      << ratio << "x" << std::endl;
        }
    }
}

// === Test 1: M11 hot cells =================================================
// Curated set of (shape × subpel) that covers most M11 LPD1 weighted calls.
// REGULAR filter only (M11 is dominated by EIGHTTAP_REGULAR).
TEST_F(Convolve2dSrPerf, DISABLED_M11HotCells) {
    const auto variants = available_variants();
    print_section("M11 hot cells (REGULAR filter, shape × subpel grid)");
    std::cout << "  shapes=" << (sizeof(kHotShapes) / sizeof(kHotShapes[0]))
              << "  subpels=" << (sizeof(kHotSubpels) / sizeof(kHotSubpels[0]))
              << "  variants=" << variants.size() << std::endl;

    double tot_ns[FN_COUNT][VAR_COUNT] = {{0}};
    int cells[FN_COUNT][VAR_COUNT] = {{0}};

    for (int f = 0; f < FN_COUNT; ++f) {
        int n = 0;
        sweep_fn(f,
                 EIGHTTAP_REGULAR,
                 kHotShapes,
                 sizeof(kHotShapes) / sizeof(kHotShapes[0]),
                 kHotSubpels,
                 sizeof(kHotSubpels) / sizeof(kHotSubpels[0]),
                 variants,
                 tot_ns[f],
                 n);
        for (int v : variants)
            cells[f][v] = n;
    }

    print_totals(variants, tot_ns, cells);
    std::cout << "  sink=0x" << std::hex << std::setw(16) << std::setfill('0')
              << static_cast<unsigned long long>(g_sink) << std::dec
              << std::setfill(' ') << std::endl;
}

// === Test 1b: M11 quarter-pel hot cells ====================================
// Subset of M11 dominated by REGULAR filter at quarter-pixel offsets
// {4, 8, 12} on x and y axes. Shapes: 8x8, 16x8, 8x16, 16x16, 32x16, 16x32,
// 32x32 (7 shapes). Per shape: x_sr 3 cells (sx in {4,8,12}, sy=8 fixed by
// dispatcher convention), y_sr 3 cells (sx=8 fixed, sy in {4,8,12}), 2d_sr
// 9 cells (sx × sy in {4,8,12}²). Total: 15 cells × 7 shapes = 105 cells.
TEST_F(Convolve2dSrPerf, DISABLED_M11QuarterPelHotCells) {
    const auto variants = available_variants();
    print_section("M11 quarter-pel hot cells (REGULAR, subpel ∈ {4,8,12})");
    std::cout << "  shapes="
              << (sizeof(kQuarterHotShapes) / sizeof(kQuarterHotShapes[0]))
              << "  subpels="
              << (sizeof(kQuarterSubpels) / sizeof(kQuarterSubpels[0]))
              << "  variants=" << variants.size() << std::endl;

    double tot_ns[FN_COUNT][VAR_COUNT] = {{0}};
    int cells[FN_COUNT][VAR_COUNT] = {{0}};

    for (int f = 0; f < FN_COUNT; ++f) {
        int n = 0;
        sweep_fn(f,
                 EIGHTTAP_REGULAR,
                 kQuarterHotShapes,
                 sizeof(kQuarterHotShapes) / sizeof(kQuarterHotShapes[0]),
                 kQuarterSubpels,
                 sizeof(kQuarterSubpels) / sizeof(kQuarterSubpels[0]),
                 variants,
                 tot_ns[f],
                 n);
        for (int v : variants)
            cells[f][v] = n;
    }

    print_totals(variants, tot_ns, cells);
    std::cout << "  sink=0x" << std::hex << std::setw(16) << std::setfill('0')
              << static_cast<unsigned long long>(g_sink) << std::dec
              << std::setfill(' ') << std::endl;
}
// Targets each dispatch path with the shape + filter combination that
// reliably routes to it:
//   4tap: shape 4x4 with REGULAR (forces 4-tap table on both axes)
//   6tap: kHotShapes (excludes 4x4) with REGULAR (most rows are 6-tap)
//   8tap: kHotShapes with MULTITAP_SHARP (always 8-tap)
TEST_F(Convolve2dSrPerf, DISABLED_AllTapsSweep) {
    const auto variants = available_variants();
    print_section("All-taps sweep (4/6/8-tap × 2d_sr/x_sr/y_sr)");
    std::cout << "  subpels=" << (sizeof(kHotSubpels) / sizeof(kHotSubpels[0]))
              << "  variants=" << variants.size() << std::endl;

    struct TapCfg {
        const std::pair<int, int> *shapes;
        int nshapes;
        InterpFilter filter;
    };
    const TapCfg cfgs[TAP_KIND_COUNT] = {
        {k4TapShapes,
         (int)(sizeof(k4TapShapes) / sizeof(k4TapShapes[0])),
         EIGHTTAP_REGULAR},
        {kHotShapes,
         (int)(sizeof(kHotShapes) / sizeof(kHotShapes[0])),
         EIGHTTAP_REGULAR},
        {kHotShapes,
         (int)(sizeof(kHotShapes) / sizeof(kHotShapes[0])),
         MULTITAP_SHARP},
    };

    // tot_ns[tap][fn][variant], cells[tap][fn]
    double tot_ns[TAP_KIND_COUNT][FN_COUNT][VAR_COUNT] = {{{0}}};
    int cells[TAP_KIND_COUNT][FN_COUNT][VAR_COUNT] = {{{0}}};

    for (int t = 0; t < TAP_KIND_COUNT; ++t) {
        for (int fn = 0; fn < FN_COUNT; ++fn) {
            int n = 0;
            sweep_fn(fn,
                     cfgs[t].filter,
                     cfgs[t].shapes,
                     cfgs[t].nshapes,
                     kHotSubpels,
                     sizeof(kHotSubpels) / sizeof(kHotSubpels[0]),
                     variants,
                     tot_ns[t][fn],
                     n);
            for (int v : variants)
                cells[t][fn][v] = n;
        }
    }

    // Per-tap summaries.
    for (int t = 0; t < TAP_KIND_COUNT; ++t) {
        std::cout << std::endl
                  << "  --- " << tap_short(t) << " ---" << std::endl;
        print_totals(variants, tot_ns[t], cells[t]);
    }
    std::cout << "  sink=0x" << std::hex << std::setw(16) << std::setfill('0')
              << static_cast<unsigned long long>(g_sink) << std::dec
              << std::setfill(' ') << std::endl;
}

}  // namespace

#endif  // ARCH_AARCH64
