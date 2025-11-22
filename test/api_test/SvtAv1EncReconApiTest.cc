/*
 * Copyright(c) 2026 Alliance for Open Media. All rights reserved
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
 * @file SvtAv1EncReconApiTest.cc
 *
 * @brief Checks that requesting the reconstructed frames does not change the
 *        bitstream.
 *
 * Several in-loop filter stages skip filtering the recon of non-reference
 * frames when nothing reads it (no CDEF search, no restoration, no recon
 * output). Those shortcuts must only skip work: the filter decisions, and so
 * the bitstream, have to be the same whether or not the recon is output. The
 * deblocking level search used to run through such a shortcut, measured every
 * level on an unfiltered frame and signalled level 0 on non-reference frames
 * unless the recon was output.
 ******************************************************************************/

#include <cstdint>
#include <vector>

#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "EbSvtAv1Metadata.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
// a full 32-frame mini-GOP (the default at the slower presets) plus a key frame
constexpr uint32_t kFrames = 33;

// Smooth, moving 8-bit 4:2:0 content. The bug only shows when the search would
// pick a non-zero level for a non-reference frame; flat color bars or busy
// textures never need deblocking, smooth gradients do.
void fill_frame(std::vector<uint8_t>& yuv, uint32_t fi) {
    uint32_t seed = 12345u + fi;
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            seed = seed * 1664525u + 1013904223u;
            // moving rings over a moving ramp, plus light noise
            const int dx = (int)x - 160 - 2 * (int)fi;
            const int dy = (int)y - 120 + (int)fi;
            const int v = 64 + (((dx * dx + dy * dy) >> 5) & 0x3f) +
                          (int)((x + 2 * fi) & 0x3f) + (int)(seed >> 29);
            yuv[y * kWidth + x] = (uint8_t)v;
        }
    }
    for (uint32_t i = kWidth * kHeight; i < yuv.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        yuv[i] = (uint8_t)(120 + ((i / 7 + fi) & 0xf) + (int)(seed >> 29));
    }
}

struct ReconParam {
    int8_t enc_mode;
    bool enable_cdef;
    bool enable_restoration;
};

std::vector<uint8_t> encode(const ReconParam& prm, bool recon_enabled) {
    std::vector<uint8_t> bitstream;
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    cfg.encoder_bit_depth = 8;
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.enc_mode = prm.enc_mode;
    cfg.cdef_level = prm.enable_cdef ? -1 : 0;
    cfg.enable_restoration_filtering = prm.enable_restoration ? -1 : 0;
    cfg.recon_enabled = recon_enabled;
    cfg.level_of_parallelism = 4;
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    std::vector<uint8_t> yuv(kWidth * kHeight * 3 / 2);
    EbSvtIOFormat frame = {yuv.data(),
                           yuv.data() + kWidth * kHeight,
                           yuv.data() + kWidth * kHeight * 5 / 4,
                           kWidth,
                           kWidth / 2,
                           kWidth / 2};

    std::vector<uint8_t> recon_buf(kWidth * kHeight * 3 / 2);
    bool eos = false;
    auto drain = [&](bool done_sending) {
        if (recon_enabled) {
            // the recon queue has to be drained or the encoder stalls
            for (;;) {
                EbBufferHeaderType recon{};
                recon.size = sizeof(recon);
                recon.p_buffer = recon_buf.data();
                recon.n_alloc_len = (uint32_t)recon_buf.size();
                const EbErrorType rc = svt_av1_get_recon(enc, &recon);
                svt_metadata_array_free(&recon.metadata);
                if (rc != EB_ErrorNone) {
                    break;
                }
            }
        }
        EbBufferHeaderType* pkt = nullptr;
        while (!eos && svt_av1_enc_get_packet(enc, &pkt, done_sending) ==
                           EB_ErrorNone) {
            eos = (pkt->flags & EB_BUFFERFLAG_EOS) != 0;
            bitstream.insert(bitstream.end(),
                             pkt->p_buffer,
                             pkt->p_buffer + pkt->n_filled_len);
            svt_av1_enc_release_out_buffer(&pkt);
        }
    };

    for (uint32_t fi = 0; fi < kFrames; ++fi) {
        fill_frame(yuv, fi);
        EbBufferHeaderType in{};
        in.size = sizeof(in);
        in.p_buffer = reinterpret_cast<uint8_t*>(&frame);
        in.n_filled_len = (uint32_t)yuv.size();
        in.pts = fi;
        in.pic_type = EB_AV1_INVALID_PICTURE;
        EXPECT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in));
        drain(false);
    }
    EbBufferHeaderType eos_hdr{};
    eos_hdr.size = sizeof(eos_hdr);
    eos_hdr.flags = EB_BUFFERFLAG_EOS;
    eos_hdr.pic_type = EB_AV1_INVALID_PICTURE;
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &eos_hdr));
    while (!eos) {
        drain(true);
    }

    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(enc));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
    return bitstream;
}

class SvtAv1ReconApiTest : public ::testing::TestWithParam<ReconParam> {};

TEST_P(SvtAv1ReconApiTest, ReconOutputDoesNotChangeBitstream) {
    const std::vector<uint8_t> without_recon = encode(GetParam(), false);
    const std::vector<uint8_t> with_recon = encode(GetParam(), true);
    ASSERT_FALSE(without_recon.empty());
    EXPECT_EQ(without_recon.size(), with_recon.size());
    EXPECT_TRUE(without_recon == with_recon)
        << "enabling recon output changed the bitstream";
}

INSTANTIATE_TEST_SUITE_P(Recon, SvtAv1ReconApiTest,
                         ::testing::Values(ReconParam{2, false, false},
                                           ReconParam{2, true, true},
                                           ReconParam{8, false, false}));

}  // namespace
