/*
 * Copyright(c) 2026 Meta Platforms, Inc.
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
 * @file SvtAv1RefSlotBitstreamTest.cc
 *
 * @brief L1T2 + LTR DPB slot split.
 *
 * In low-delay CBR with temporal layers an LTR anchor must not occupy a DPB
 *slot the encoder still rotates through: the Phase-3 refresh guard freezes any
 *slot holding an anchor, so the base layer would go on predicting from an
 * ever-staler picture. Regular references therefore own slots {0..3} and
 *anchors own {4..7}.
 *
 * Encodes an LD-CBR L1T2 clip with one held LTR anchor, decodes through libaom,
 * and asserts (from refresh_frame_flags) that the anchor lands in the top-4
 *pool and that the stream decodes without corruption.
 ******************************************************************************/

#include <cstdint>
#include <memory>
#include <vector>
#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "RefDecoder.h"
#include "VideoFrame.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
constexpr uint32_t kNumFrames = 40;
constexpr uint32_t kStoreFrame = 6;
constexpr uint32_t kStoreId = 42u;
// LTR anchors live in the top 4 slots; the bottom 4 belong to the regular refs.
constexpr uint8_t kLtrPoolMask = 0xF0u;

static int popcount8(uint8_t m) {
    int n = 0;
    for (int i = 0; i < 8; ++i)
        if (m & (1u << i))
            ++n;
    return n;
}

// Encodes kNumFrames of L1T2 LD-CBR with `anchors` held LTR anchors (first at
// kStoreFrame), decodes the whole stream, and returns the parsed StreamInfo via
// out-params. `refresh` is the per-frame refresh_frame_flags; `clean` is true
// if every frame decoded without corruption.
static void encode_decode(int anchors, std::vector<uint8_t>* refresh,
                          bool* clean) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30000;
    cfg.frame_rate_denominator = 1000;
    cfg.encoder_bit_depth = 8;
    cfg.encoder_color_format = EB_YUV420;
    cfg.enc_mode = 9;
    cfg.tune = 1;
    cfg.rtc = true;
    cfg.intra_period_length = 100;
    cfg.hierarchical_levels = 1;  // L1T2
    cfg.pred_structure = LOW_DELAY;
    cfg.rate_control_mode = SVT_AV1_RC_MODE_CBR;
    cfg.target_bit_rate = 500000;
    cfg.max_qp_allowed = 63;
    cfg.min_qp_allowed = 4;
    cfg.look_ahead_distance = 0;
    cfg.recode_loop = 0;
    cfg.max_managed_refs = 4;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(0, src.open_source(0, 0));

    std::vector<std::vector<uint8_t>> packets;
    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);
    for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);
        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
        in_hdr.p_app_private = nullptr;

        SvtAv1RefFrameCmd store_payload{};
        EbPrivDataNode node{};
        for (int a = 0; a < anchors; ++a) {
            if (fi == kStoreFrame + static_cast<uint32_t>(a) * 2u) {
                store_payload.pic_id = kStoreId + static_cast<uint32_t>(a);
                node.node_type = REF_STORE_EVENT;
                node.data = &store_payload;
                node.size = sizeof(store_payload);
                in_hdr.p_app_private = &node;
            }
        }
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in_hdr));
        EbBufferHeaderType* out = nullptr;
        if (svt_av1_enc_get_packet(enc, &out, 0) == EB_ErrorNone && out) {
            packets.emplace_back(out->p_buffer,
                                 out->p_buffer + out->n_filled_len);
            svt_av1_enc_release_out_buffer(&out);
        }
    }
    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &eos));
    for (;;) {
        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc, &out, 1);
        if (rc != EB_ErrorNone || !out)
            break;
        const bool is_eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos && out->n_filled_len > 0)
            packets.emplace_back(out->p_buffer,
                                 out->p_buffer + out->n_filled_len);
        svt_av1_enc_release_out_buffer(&out);
        if (is_eos)
            break;
    }
    svt_av1_enc_deinit(enc);
    svt_av1_enc_deinit_handle(enc);
    ASSERT_GE(packets.size(), static_cast<size_t>(kNumFrames / 2));

    std::unique_ptr<RefDecoder> decoder(
        create_reference_decoder(/*enable_analyzer=*/false));
    ASSERT_NE(nullptr, decoder.get());
    for (size_t i = 0; i < packets.size(); ++i) {
        ASSERT_EQ(
            RefDecoder::REF_CODEC_OK,
            decoder->decode(packets[i].data(), (uint32_t)packets[i].size()))
            << "decode failed for packet " << i;
        VideoFrame vf;
        while (decoder->get_frame(vf) == RefDecoder::REF_CODEC_OK) {
        }
    }
    const RefDecoder::StreamInfo* info = decoder->get_stream_info();
    ASSERT_NE(nullptr, info);
    *refresh = info->refresh_frame_flags_list;
    bool ok = true;
    for (size_t i = 0; i < info->frame_corrupted_list.size(); ++i)
        if (info->frame_corrupted_list[i])
            ok = false;
    *clean = ok;
}

TEST(RefSlotBitstreamTest, AnchorAvoidsBaseWindow) {
    std::vector<uint8_t> refresh;
    bool clean = false;
    encode_decode(/*anchors=*/1, &refresh, &clean);
    ASSERT_GT(refresh.size(), kStoreFrame + 4);

    // The anchor slot is refreshed at the STORE frame and then preserved (never
    // refreshed again) by the guard. Find the bit set at STORE that stays clear
    // afterwards.
    uint8_t preserved = refresh[kStoreFrame];
    for (size_t i = kStoreFrame + 1; i < refresh.size(); ++i)
        preserved &= (uint8_t)~refresh[i];
    ASSERT_EQ(1, popcount8(preserved))
        << "expected exactly one preserved (anchor) slot; got 0x" << std::hex
        << (int)preserved;
    EXPECT_EQ(preserved, (uint8_t)(preserved & kLtrPoolMask))
        << "anchor slot 0x" << std::hex << (int)preserved
        << " is outside the LTR pool 0x" << (int)kLtrPoolMask
        << " and collides with the regular refs (the reference-slot bug)";
    EXPECT_TRUE(clean) << "stream must decode without corruption";
}

TEST(RefSlotBitstreamTest, FourAnchorsDecodeClean) {
    std::vector<uint8_t> refresh;
    bool clean = false;
    encode_decode(/*anchors=*/4, &refresh, &clean);
    EXPECT_TRUE(clean) << "4-anchor L1T2 stream must decode without corruption";
}

}  // namespace
