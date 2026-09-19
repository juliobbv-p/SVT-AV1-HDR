/*
* Copyright(c) 2026 Meta Platforms, Inc. and affiliates.
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

// Low-bit-depth (8-bit) forward transforms with int16 intermediates, AVX2.
// Port of Source/Lib/ASM_NEON/lowbd_fwd_txfm2d_neon.c.
//
// Why this is worth doing on x86: the existing AVX2 forward transform keeps
// int32 intermediates, so a 16-wide row needs TWO __m256i. At int16 a 16-wide
// row is ONE __m256i, halving the vector count. (At 8x8 there is no win: 8
// int32 and 8 int16 both fit one register, which is why only the >=16 sizes
// are ported here.)
//
// Bit-exactness: every butterfly below reproduces the NEON/C arithmetic
// exactly. The two-weight butterflies keep the widening int32 accumulate
// (_mm256_madd_epi16 is an exact fit for "in0*wa + in1*wb"), and the cospi32
// butterflies use _mm256_mulhrs_epi16, which computes (a*b + 0x4000) >> 15 --
// identical to NEON's vqrdmulhq_s16. CONFIG_ENABLE_FAST_LBD_TXFM (the
// per-product vqrdmulh approximation) is deliberately NOT ported.

#include <immintrin.h>

#include "aom_dsp_rtcd.h"
#include "definitions.h"
#include "transforms.h"

#define TXFM_COS_BIT_MIN 10

// cospi constants in Q2.13, indexed [cos_bit - 10]; same table as the NEON port.
static const int16_t fwd_cospi_arr_q13_avx2[4][128] = {
    {
        5792, 5792, -5792, -5792, 7568, 3136, -7568, -3136, 8032, 1600, -8032, -1600, 6808, 4552, -6808, -4552,
        8152, 800,  -8152, -800,  7840, 2376, -7840, -2376, 7224, 3864, -7224, -3864, 6336, 5200, -6336, -5200,
        8184, 400,  -8184, -400,  8104, 1200, -8104, -1200, 7944, 1992, -7944, -1992, 7712, 2760, -7712, -2760,
        7408, 3504, -7408, -3504, 7024, 4208, -7024, -4208, 6576, 4880, -6576, -4880, 6072, 5504, -6072, -5504,
        8192, 200,  -8192, -200,  8168, 600,  -8168, -600,  8128, 1000, -8128, -1000, 8072, 1400, -8072, -1400,
        7992, 1792, -7992, -1792, 7896, 2184, -7896, -2184, 7776, 2568, -7776, -2568, 7640, 2952, -7640, -2952,
        7488, 3320, -7488, -3320, 7320, 3680, -7320, -3680, 7128, 4040, -7128, -4040, 6920, 4384, -6920, -4384,
        6696, 4720, -6696, -4720, 6456, 5040, -6456, -5040, 6200, 5352, -6200, -5352, 5936, 5648, -5936, -5648,
    },
    {
        5792, 5792, -5792, -5792, 7568, 3136, -7568, -3136, 8036, 1600, -8036, -1600, 6812, 4552, -6812, -4552,
        8152, 804,  -8152, -804,  7840, 2380, -7840, -2380, 7224, 3860, -7224, -3860, 6332, 5196, -6332, -5196,
        8184, 400,  -8184, -400,  8104, 1204, -8104, -1204, 7948, 1992, -7948, -1992, 7712, 2760, -7712, -2760,
        7404, 3504, -7404, -3504, 7028, 4212, -7028, -4212, 6580, 4880, -6580, -4880, 6068, 5500, -6068, -5500,
        8188, 200,  -8188, -200,  8168, 604,  -8168, -604,  8132, 1004, -8132, -1004, 8072, 1400, -8072, -1400,
        7992, 1796, -7992, -1796, 7896, 2184, -7896, -2184, 7780, 2568, -7780, -2568, 7644, 2948, -7644, -2948,
        7488, 3320, -7488, -3320, 7316, 3684, -7316, -3684, 7128, 4036, -7128, -4036, 6920, 4384, -6920, -4384,
        6696, 4716, -6696, -4716, 6460, 5040, -6460, -5040, 6204, 5352, -6204, -5352, 5932, 5648, -5932, -5648,
    },
    {
        5792, 5792, -5792, -5792, 7568, 3134, -7568, -3134, 8034, 1598, -8034, -1598, 6812, 4552, -6812, -4552,
        8152, 802,  -8152, -802,  7840, 2378, -7840, -2378, 7224, 3862, -7224, -3862, 6332, 5196, -6332, -5196,
        8182, 402,  -8182, -402,  8104, 1202, -8104, -1202, 7946, 1990, -7946, -1990, 7714, 2760, -7714, -2760,
        7406, 3502, -7406, -3502, 7026, 4212, -7026, -4212, 6580, 4880, -6580, -4880, 6070, 5502, -6070, -5502,
        8190, 202,  -8190, -202,  8170, 602,  -8170, -602,  8130, 1002, -8130, -1002, 8072, 1400, -8072, -1400,
        7992, 1794, -7992, -1794, 7896, 2184, -7896, -2184, 7778, 2570, -7778, -2570, 7644, 2948, -7644, -2948,
        7490, 3320, -7490, -3320, 7318, 3684, -7318, -3684, 7128, 4038, -7128, -4038, 6922, 4382, -6922, -4382,
        6698, 4718, -6698, -4718, 6458, 5040, -6458, -5040, 6204, 5350, -6204, -5350, 5934, 5648, -5934, -5648,
    },
    {
        5793, 5793, -5793, -5793, 7568, 3135, -7568, -3135, 8035, 1598, -8035, -1598, 6811, 4551, -6811, -4551,
        8153, 803,  -8153, -803,  7839, 2378, -7839, -2378, 7225, 3862, -7225, -3862, 6333, 5197, -6333, -5197,
        8182, 402,  -8182, -402,  8103, 1202, -8103, -1202, 7946, 1990, -7946, -1990, 7713, 2760, -7713, -2760,
        7405, 3503, -7405, -3503, 7027, 4212, -7027, -4212, 6580, 4880, -6580, -4880, 6070, 5501, -6070, -5501,
        8190, 201,  -8190, -201,  8170, 603,  -8170, -603,  8130, 1003, -8130, -1003, 8071, 1401, -8071, -1401,
        7993, 1795, -7993, -1795, 7895, 2185, -7895, -2185, 7779, 2570, -7779, -2570, 7643, 2948, -7643, -2948,
        7489, 3320, -7489, -3320, 7317, 3683, -7317, -3683, 7128, 4038, -7128, -4038, 6921, 4383, -6921, -4383,
        6698, 4717, -6698, -4717, 6458, 5040, -6458, -5040, 6203, 5351, -6203, -5351, 5933, 5649, -5933, -5649,
    }};

static INLINE const int16_t* fwd_cospi_q13_avx2(int cos_bit) {
    return fwd_cospi_arr_q13_avx2[cos_bit - TXFM_COS_BIT_MIN];
}

// Packs two Q13 weights into the 32-bit lanes _mm256_madd_epi16 consumes.
static INLINE __m256i wpair(const int16_t* w, int a, int b) {
    // Build via unsigned: shifting a negative int32 left is UB before C23 and
    // the cospi tables contain negative weights.
    return _mm256_set1_epi32((int32_t)(((uint32_t)(uint16_t)w[b] << 16) | (uint16_t)w[a]));
}

// out0 = rs13(in0*w[l0] + in1*w[l1]); out1 = rs13(in0*w[l2] + in1*w[l3]).
// madd_epi16 does the widening multiply-add in one op per half, which is the
// exact shape of this butterfly.
static INLINE void butterfly_x16(const __m256i in0, const __m256i in1, const __m256i w01, const __m256i w23,
                                 __m256i* out0, __m256i* out1) {
    const __m256i rnd = _mm256_set1_epi32(1 << 12);
    const __m256i lo  = _mm256_unpacklo_epi16(in0, in1);
    const __m256i hi  = _mm256_unpackhi_epi16(in0, in1);
    const __m256i u0  = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(lo, w01), rnd), 13);
    const __m256i u1  = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(hi, w01), rnd), 13);
    const __m256i v0  = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(lo, w23), rnd), 13);
    const __m256i v1  = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(hi, w23), rnd), 13);
    *out0             = _mm256_packs_epi32(u0, u1);
    *out1             = _mm256_packs_epi32(v0, v1);
}

// cospi32 pair: out0 = rdmulh(in0+in1, w32), out1 = rdmulh(in0-in1, w32).
// mulhrs_epi16 == vqrdmulhq_s16.
static INLINE void btf_cospi32_0112_x16(const __m256i w32, const __m256i in0, const __m256i in1, __m256i* out0,
                                        __m256i* out1) {
    *out0 = _mm256_mulhrs_epi16(_mm256_adds_epi16(in0, in1), w32);
    *out1 = _mm256_mulhrs_epi16(_mm256_subs_epi16(in0, in1), w32);
}

static INLINE void butterfly_dct_pre_x16(const __m256i* in, __m256i* out, int n) {
    for (int i = 0; i < n / 2; ++i) {
        out[i] = _mm256_adds_epi16(in[i], in[n - i - 1]);
    }
    for (int i = 0; i < n / 2; ++i) {
        out[n / 2 + i] = _mm256_subs_epi16(in[n / 2 - i - 1], in[n / 2 + i]);
    }
}

static INLINE void butterfly_dct_post_x16(const __m256i* in0, const __m256i* in1, __m256i* out, int n) {
    for (int i = 0; i < n / 4; ++i) {
        out[i] = _mm256_adds_epi16(in0[i], in1[n / 2 - i - 1]);
    }
    for (int i = 0; i < n / 4; ++i) {
        out[n / 4 + i] = _mm256_subs_epi16(in0[n / 4 - i - 1], in1[n / 4 + i]);
    }
    for (int i = 0; i < n / 4; ++i) {
        out[n / 2 + i] = _mm256_subs_epi16(in0[n - i - 1], in1[n / 2 + i]);
    }
    for (int i = 0; i < n / 4; ++i) {
        out[(3 * n) / 4 + i] = _mm256_adds_epi16(in0[(3 * n) / 4 + i], in1[(3 * n) / 4 - i - 1]);
    }
}

// 16-point forward DCT, int16 lanes. Mirrors fdct16x16_neon exactly.
static INLINE void fdct16_x16_avx2(const __m256i* input, __m256i* output, int cos_bit) {
    const int16_t* cospi = fwd_cospi_q13_avx2(cos_bit);
    const __m256i  w32   = _mm256_set1_epi16((short)(cospi[0] * 4));

    // cospiN occupies lanes [4*k .. 4*k+3] as {a, b, -a, -b}.
    const int16_t *c32 = cospi + 4 * 0, *c16 = cospi + 4 * 1;
    const int16_t *c8 = cospi + 4 * 2, *c24 = cospi + 4 * 3;
    const int16_t *c4 = cospi + 4 * 4, *c12 = cospi + 4 * 5;
    const int16_t *c20 = cospi + 4 * 6, *c28 = cospi + 4 * 7;

    __m256i x1[16];
    butterfly_dct_pre_x16(input, x1, 16);
    __m256i x2[16];
    butterfly_dct_pre_x16(x1, x2, 8);
    btf_cospi32_0112_x16(w32, x1[13], x1[10], &x2[13], &x2[10]);
    btf_cospi32_0112_x16(w32, x1[12], x1[11], &x2[12], &x2[11]);

    __m256i x3[16];
    butterfly_dct_pre_x16(x2, x3, 4);
    btf_cospi32_0112_x16(w32, x2[6], x2[5], &x3[6], &x3[5]);
    butterfly_dct_post_x16(x1 + 8, x2 + 8, x3 + 8, 8);

    __m256i x4[16];
    // stage-4 cospi32 stays widening: x3[0]+x3[1] can exceed int16 in the
    // cos_bit-12 row pass.
    butterfly_x16(x3[0], x3[1], wpair(c32, 0, 1), wpair(c32, 1, 2), &output[0], &output[8]);
    butterfly_x16(x3[3], x3[2], wpair(c16, 0, 1), wpair(c16, 1, 2), &output[4], &output[12]);
    butterfly_dct_post_x16(x2 + 4, x3 + 4, x4 + 4, 4);
    butterfly_x16(x3[14], x3[9], wpair(c16, 0, 1), wpair(c16, 1, 2), &x4[14], &x4[9]);
    butterfly_x16(x3[13], x3[10], wpair(c16, 1, 2), wpair(c16, 2, 3), &x4[13], &x4[10]);

    __m256i x5[16];
    butterfly_x16(x4[7], x4[4], wpair(c8, 0, 1), wpair(c8, 1, 2), &output[2], &output[14]);
    butterfly_x16(x4[6], x4[5], wpair(c24, 1, 0), wpair(c24, 0, 3), &output[10], &output[6]);
    butterfly_dct_post_x16(x3 + 8, x4 + 8, x5 + 8, 4);
    butterfly_dct_post_x16(x3 + 12, x4 + 12, x5 + 12, 4);

    butterfly_x16(x5[15], x5[8], wpair(c4, 0, 1), wpair(c4, 1, 2), &output[1], &output[15]);
    butterfly_x16(x5[14], x5[9], wpair(c28, 1, 0), wpair(c28, 0, 3), &output[9], &output[7]);
    butterfly_x16(x5[13], x5[10], wpair(c20, 0, 1), wpair(c20, 1, 2), &output[5], &output[11]);
    butterfly_x16(x5[12], x5[11], wpair(c12, 1, 0), wpair(c12, 0, 3), &output[13], &output[3]);
}

// Lane-local 8x8 int16 transpose: AVX2 unpacks never cross the 128-bit
// boundary, so this transposes the 8x8 in lane 0 and the 8x8 in lane 1 at once.
static INLINE void transpose_8x8_s16_lanes(const __m256i* in, __m256i* out) {
    __m256i p[4], q[4], r[8];
    for (int i = 0; i < 4; i++) {
        p[i] = _mm256_unpacklo_epi16(in[2 * i], in[2 * i + 1]);
        q[i] = _mm256_unpackhi_epi16(in[2 * i], in[2 * i + 1]);
    }
    r[0]   = _mm256_unpacklo_epi32(p[0], p[1]);
    r[1]   = _mm256_unpackhi_epi32(p[0], p[1]);
    r[2]   = _mm256_unpacklo_epi32(q[0], q[1]);
    r[3]   = _mm256_unpackhi_epi32(q[0], q[1]);
    r[4]   = _mm256_unpacklo_epi32(p[2], p[3]);
    r[5]   = _mm256_unpackhi_epi32(p[2], p[3]);
    r[6]   = _mm256_unpacklo_epi32(q[2], q[3]);
    r[7]   = _mm256_unpackhi_epi32(q[2], q[3]);
    out[0] = _mm256_unpacklo_epi64(r[0], r[4]);
    out[1] = _mm256_unpackhi_epi64(r[0], r[4]);
    out[2] = _mm256_unpacklo_epi64(r[1], r[5]);
    out[3] = _mm256_unpackhi_epi64(r[1], r[5]);
    out[4] = _mm256_unpacklo_epi64(r[2], r[6]);
    out[5] = _mm256_unpackhi_epi64(r[2], r[6]);
    out[6] = _mm256_unpacklo_epi64(r[3], r[7]);
    out[7] = _mm256_unpackhi_epi64(r[3], r[7]);
}

// 16x16 = four 8x8 quadrants. Transposing rows 0-7 handles quadrants A (lane 0)
// and B (lane 1); rows 8-15 handles C and D. The final permute puts A^T|C^T in
// the top half and B^T|D^T in the bottom.
static INLINE void transpose_16x16_s16_avx2(__m256i* v) {
    __m256i t0[8], t1[8];
    transpose_8x8_s16_lanes(v + 0, t0);
    transpose_8x8_s16_lanes(v + 8, t1);
    for (int k = 0; k < 8; k++) {
        v[k]     = _mm256_permute2x128_si256(t0[k], t1[k], 0x20);
        v[k + 8] = _mm256_permute2x128_si256(t0[k], t1[k], 0x31);
    }
}

// 32-point forward DCT, int16 lanes. Mirrors fdct32x32_neon exactly.
static INLINE void fdct32_x16_avx2(const __m256i* input, __m256i* output, int cos_bit) {
    const int16_t* cospi = fwd_cospi_q13_avx2(cos_bit);
    const __m256i  w32   = _mm256_set1_epi16((short)(cospi[0] * 4));
    const int16_t *c32 = cospi + 4 * 0, *c16 = cospi + 4 * 1;
    const int16_t *c8 = cospi + 4 * 2, *c24 = cospi + 4 * 3;
    const int16_t *c4 = cospi + 4 * 4, *c12 = cospi + 4 * 5;
    const int16_t *c20 = cospi + 4 * 6, *c28 = cospi + 4 * 7;
    const int16_t *c2 = cospi + 4 * 8, *c6 = cospi + 4 * 9;
    const int16_t *c10 = cospi + 4 * 10, *c14 = cospi + 4 * 11;
    const int16_t *c18 = cospi + 4 * 12, *c22 = cospi + 4 * 13;
    const int16_t *c26 = cospi + 4 * 14, *c30 = cospi + 4 * 15;

    __m256i x1[32];
    butterfly_dct_pre_x16(input, x1, 32);
    __m256i x2[32];
    butterfly_dct_pre_x16(x1, x2, 16);
    btf_cospi32_0112_x16(w32, x1[27], x1[20], &x2[27], &x2[20]);
    btf_cospi32_0112_x16(w32, x1[26], x1[21], &x2[26], &x2[21]);
    btf_cospi32_0112_x16(w32, x1[25], x1[22], &x2[25], &x2[22]);
    btf_cospi32_0112_x16(w32, x1[24], x1[23], &x2[24], &x2[23]);

    __m256i x3[32];
    butterfly_dct_pre_x16(x2, x3, 8);
    btf_cospi32_0112_x16(w32, x2[13], x2[10], &x3[13], &x3[10]);
    btf_cospi32_0112_x16(w32, x2[12], x2[11], &x3[12], &x3[11]);
    butterfly_dct_post_x16(x1 + 16, x2 + 16, x3 + 16, 16);

    __m256i x4[32];
    butterfly_dct_pre_x16(x3, x4, 4);
    btf_cospi32_0112_x16(w32, x3[6], x3[5], &x4[6], &x4[5]);
    butterfly_dct_post_x16(x2 + 8, x3 + 8, x4 + 8, 8);
    butterfly_x16(x3[29], x3[18], wpair(c16, 0, 1), wpair(c16, 1, 2), &x4[29], &x4[18]);
    butterfly_x16(x3[28], x3[19], wpair(c16, 0, 1), wpair(c16, 1, 2), &x4[28], &x4[19]);
    butterfly_x16(x3[27], x3[20], wpair(c16, 1, 2), wpair(c16, 2, 3), &x4[27], &x4[20]);
    butterfly_x16(x3[26], x3[21], wpair(c16, 1, 2), wpair(c16, 2, 3), &x4[26], &x4[21]);

    __m256i x5[32];
    butterfly_x16(x4[0], x4[1], wpair(c32, 0, 1), wpair(c32, 1, 2), &output[0], &output[16]);
    butterfly_x16(x4[3], x4[2], wpair(c16, 0, 1), wpair(c16, 1, 2), &output[8], &output[24]);
    butterfly_dct_post_x16(x3 + 4, x4 + 4, x5 + 4, 4);
    butterfly_x16(x4[14], x4[9], wpair(c16, 0, 1), wpair(c16, 1, 2), &x5[14], &x5[9]);
    butterfly_x16(x4[13], x4[10], wpair(c16, 1, 2), wpair(c16, 2, 3), &x5[13], &x5[10]);
    butterfly_dct_post_x16(x3 + 16, x4 + 16, x5 + 16, 8);
    butterfly_dct_post_x16(x3 + 24, x4 + 24, x5 + 24, 8);

    __m256i x6[32];
    butterfly_x16(x5[7], x5[4], wpair(c8, 0, 1), wpair(c8, 1, 2), &output[4], &output[28]);
    butterfly_x16(x5[6], x5[5], wpair(c24, 1, 0), wpair(c24, 0, 3), &output[20], &output[12]);
    butterfly_dct_post_x16(x4 + 8, x5 + 8, x6 + 8, 4);
    butterfly_dct_post_x16(x4 + 12, x5 + 12, x6 + 12, 4);
    butterfly_x16(x5[30], x5[17], wpair(c8, 0, 1), wpair(c8, 1, 2), &x6[30], &x6[17]);
    butterfly_x16(x5[29], x5[18], wpair(c8, 1, 2), wpair(c8, 2, 3), &x6[29], &x6[18]);
    butterfly_x16(x5[26], x5[21], wpair(c24, 1, 0), wpair(c24, 0, 3), &x6[26], &x6[21]);
    butterfly_x16(x5[25], x5[22], wpair(c24, 0, 3), wpair(c24, 3, 2), &x6[25], &x6[22]);

    __m256i x7[32];
    butterfly_x16(x6[15], x6[8], wpair(c4, 0, 1), wpair(c4, 1, 2), &output[2], &output[30]);
    butterfly_x16(x6[14], x6[9], wpair(c28, 1, 0), wpair(c28, 0, 3), &output[18], &output[14]);
    butterfly_x16(x6[13], x6[10], wpair(c20, 0, 1), wpair(c20, 1, 2), &output[10], &output[22]);
    butterfly_x16(x6[12], x6[11], wpair(c12, 1, 0), wpair(c12, 0, 3), &output[26], &output[6]);
    butterfly_dct_post_x16(x5 + 16, x6 + 16, x7 + 16, 4);
    butterfly_dct_post_x16(x5 + 20, x6 + 20, x7 + 20, 4);
    butterfly_dct_post_x16(x5 + 24, x6 + 24, x7 + 24, 4);
    butterfly_dct_post_x16(x5 + 28, x6 + 28, x7 + 28, 4);

    butterfly_x16(x7[31], x7[16], wpair(c2, 0, 1), wpair(c2, 1, 2), &output[1], &output[31]);
    butterfly_x16(x7[30], x7[17], wpair(c30, 1, 0), wpair(c30, 0, 3), &output[17], &output[15]);
    butterfly_x16(x7[29], x7[18], wpair(c18, 0, 1), wpair(c18, 1, 2), &output[9], &output[23]);
    butterfly_x16(x7[28], x7[19], wpair(c14, 1, 0), wpair(c14, 0, 3), &output[25], &output[7]);
    butterfly_x16(x7[27], x7[20], wpair(c10, 0, 1), wpair(c10, 1, 2), &output[5], &output[27]);
    butterfly_x16(x7[26], x7[21], wpair(c22, 1, 0), wpair(c22, 0, 3), &output[21], &output[11]);
    butterfly_x16(x7[25], x7[22], wpair(c26, 0, 1), wpair(c26, 1, 2), &output[13], &output[19]);
    butterfly_x16(x7[24], x7[23], wpair(c6, 1, 0), wpair(c6, 0, 3), &output[29], &output[3]);
}

// 32x32 DCT_DCT, int16. 32 columns = two __m256i, so the column pass runs twice
// and the row pass twice, with 16x16 transposes joining them.
void svt_lbd_fwd_txfm2d_32x32_dct_avx2(int16_t* input, int32_t* output, uint32_t stride) {
    __m256i cb[2][32];

    for (int g = 0; g < 2; g++) {
        __m256i buf[32];
        for (int r = 0; r < 32; r++) {
            buf[r] = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i*)(input + r * stride + 16 * g)), 2);
        }
        fdct32_x16_avx2(buf, cb[g], 12);
        for (int k = 0; k < 32; k++) {
            cb[g][k] = _mm256_srai_epi16(_mm256_add_epi16(cb[g][k], _mm256_set1_epi16(8)), 4);
        }
    }

    for (int j = 0; j < 2; j++) {
        __m256i rin[32], rout[32];
        for (int g = 0; g < 2; g++) {
            __m256i blk[16];
            for (int t = 0; t < 16; t++) {
                blk[t] = cb[g][16 * j + t];
            }
            transpose_16x16_s16_avx2(blk);
            for (int t = 0; t < 16; t++) {
                rin[16 * g + t] = blk[t];
            }
        }
        fdct32_x16_avx2(rin, rout, 12);
        for (int fb = 0; fb < 2; fb++) {
            __m256i blk[16];
            for (int t = 0; t < 16; t++) {
                blk[t] = rout[16 * fb + t];
            }
            transpose_16x16_s16_avx2(blk);
            for (int t = 0; t < 16; t++) {
                int32_t*      o  = output + (16 * j + t) * 32 + 16 * fb;
                const __m128i lo = _mm256_castsi256_si128(blk[t]);
                const __m128i hi = _mm256_extracti128_si256(blk[t], 1);
                _mm256_storeu_si256((__m256i*)(o + 0), _mm256_cvtepi16_epi32(lo));
                _mm256_storeu_si256((__m256i*)(o + 8), _mm256_cvtepi16_epi32(hi));
            }
        }
    }
}

#define NEW_SQRT2_BITS 12
#define NEW_SQRT2 5793

// Rectangular 1/sqrt(2) scaling: round_shift(a * NEW_SQRT2, 12), saturating on
// the narrow, matching NEON's vqrshrn_n_s32. madd_epi16 against the pair
// (NEW_SQRT2, 0) does the widening multiply in one op per half.
static INLINE __m256i round_shift_sqrt2_x16(const __m256i a) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i w    = _mm256_set1_epi32(NEW_SQRT2);
    const __m256i rnd  = _mm256_set1_epi32(1 << (NEW_SQRT2_BITS - 1));
    const __m256i lo   = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, zero), w);
    const __m256i hi   = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, zero), w);
    return _mm256_packs_epi32(_mm256_srai_epi32(_mm256_add_epi32(lo, rnd), NEW_SQRT2_BITS),
                              _mm256_srai_epi32(_mm256_add_epi32(hi, rnd), NEW_SQRT2_BITS));
}

static INLINE void store_rows_s32(const __m256i v, int32_t* o) {
    _mm256_storeu_si256((__m256i*)(o + 0), _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)));
    _mm256_storeu_si256((__m256i*)(o + 8), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1)));
}

// 16x32 DCT_DCT: 32-point column transform, 16-point row transform, rect scaled.
void svt_lbd_fwd_txfm2d_16x32_dct_avx2(int16_t* input, int32_t* output, uint32_t stride) {
    __m256i cb[32], buf[32];
    for (int r = 0; r < 32; r++) {
        buf[r] = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i*)(input + r * stride)), 2);
    }
    fdct32_x16_avx2(buf, cb, 12);
    for (int k = 0; k < 32; k++) {
        cb[k] = _mm256_srai_epi16(_mm256_add_epi16(cb[k], _mm256_set1_epi16(8)), 4);
    }

    for (int j = 0; j < 2; j++) {
        __m256i blk[16], rout[16];
        for (int t = 0; t < 16; t++) {
            blk[t] = cb[16 * j + t];
        }
        transpose_16x16_s16_avx2(blk);
        fdct16_x16_avx2(blk, rout, 13);
        for (int t = 0; t < 16; t++) {
            rout[t] = round_shift_sqrt2_x16(rout[t]);
        }
        transpose_16x16_s16_avx2(rout);
        for (int t = 0; t < 16; t++) {
            store_rows_s32(rout[t], output + (16 * j + t) * 16);
        }
    }
}

// 32x16 DCT_DCT: 16-point column transform, 32-point row transform, rect scaled.
void svt_lbd_fwd_txfm2d_32x16_dct_avx2(int16_t* input, int32_t* output, uint32_t stride) {
    __m256i cb[2][16];
    for (int g = 0; g < 2; g++) {
        __m256i buf[16];
        for (int r = 0; r < 16; r++) {
            buf[r] = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i*)(input + r * stride + 16 * g)), 2);
        }
        fdct16_x16_avx2(buf, cb[g], 13);
        for (int k = 0; k < 16; k++) {
            cb[g][k] = _mm256_srai_epi16(_mm256_add_epi16(cb[g][k], _mm256_set1_epi16(8)), 4);
        }
    }

    __m256i rin[32], rout[32];
    for (int g = 0; g < 2; g++) {
        __m256i blk[16];
        for (int t = 0; t < 16; t++) {
            blk[t] = cb[g][t];
        }
        transpose_16x16_s16_avx2(blk);
        for (int t = 0; t < 16; t++) {
            rin[16 * g + t] = blk[t];
        }
    }
    fdct32_x16_avx2(rin, rout, 13);
    for (int f = 0; f < 32; f++) {
        rout[f] = round_shift_sqrt2_x16(rout[f]);
    }
    for (int fb = 0; fb < 2; fb++) {
        __m256i blk[16];
        for (int t = 0; t < 16; t++) {
            blk[t] = rout[16 * fb + t];
        }
        transpose_16x16_s16_avx2(blk);
        for (int t = 0; t < 16; t++) {
            store_rows_s32(blk[t], output + t * 32 + 16 * fb);
        }
    }
}

// DCT_DCT only. Other tx_types stay on the existing int32 AVX2 path.
void svt_lbd_fwd_txfm2d_16x16_dct_avx2(int16_t* input, int32_t* output, uint32_t stride) {
    __m256i buf[16];

    // Column pass: 8-bit residual scaled by 4, cos_bit 13.
    for (int r = 0; r < 16; r++) {
        buf[r] = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i*)(input + r * stride)), 2);
    }
    fdct16_x16_avx2(buf, buf, 13);

    // round_shift(x, 2)
    for (int r = 0; r < 16; r++) {
        buf[r] = _mm256_srai_epi16(_mm256_add_epi16(buf[r], _mm256_set1_epi16(2)), 2);
    }

    transpose_16x16_s16_avx2(buf);
    fdct16_x16_avx2(buf, buf, 12);
    transpose_16x16_s16_avx2(buf);

    for (int r = 0; r < 16; r++) {
        const __m128i lo = _mm256_castsi256_si128(buf[r]);
        const __m128i hi = _mm256_extracti128_si256(buf[r], 1);
        _mm256_storeu_si256((__m256i*)(output + r * 16 + 0), _mm256_cvtepi16_epi32(lo));
        _mm256_storeu_si256((__m256i*)(output + r * 16 + 8), _mm256_cvtepi16_epi32(hi));
    }
}
