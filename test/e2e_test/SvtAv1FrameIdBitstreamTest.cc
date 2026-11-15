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
 * @file SvtAv1FrameIdBitstreamTest.cc
 *
 * @brief End-to-end check that AV1 frame_id is enabled iff application-managed
 * LTR is on. The encoder sets seq_header.frame_id_numbers_present_flag when
 * max_managed_refs > 0 so a receiver's frame-id view and the decoder's DPB-slot
 * view cannot diverge under loss; VOD / non-LTR streams keep the bit-exact
 * legacy path with the flag off.
 *
 * Encodes an LD-CBR clip, decodes through libaom (RefDecoder), and asserts the
 * sequence-header frame_id_numbers_present_flag parsed out of the bitstream and
 * that the stream decodes without corruption.
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
constexpr uint32_t kNumFrames = 30;
constexpr uint32_t kStoreFrame = 6;
constexpr uint32_t kStoreId = 42u;

// Encodes kNumFrames of L1T2 LD-CBR with the given max_managed_refs (STOREs one
// anchor when LTR is on, and USEs it at `use_frame` if >= 0), decodes the whole
// stream through libaom, and returns the parsed StreamInfo. `all_clean` reports
// whether every frame decoded without corruption.
static void encode_decode(uint8_t max_managed_refs, int* frame_id_flag,
                          bool* all_clean, int use_frame = -1) {
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
    cfg.max_managed_refs = max_managed_refs;
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

        SvtAv1RefFrameCmd store_payload{kStoreId};
        EbPrivDataNode node{};
        if (max_managed_refs > 0 && fi == kStoreFrame) {
            node.node_type = REF_STORE_EVENT;
            node.data = &store_payload;
            node.size = sizeof(store_payload);
            in_hdr.p_app_private = &node;
        } else if (max_managed_refs > 0 && use_frame >= 0 &&
                   fi == static_cast<uint32_t>(use_frame)) {
            node.node_type = REF_USE_EVENT;
            node.data = &store_payload;
            node.size = sizeof(store_payload);
            in_hdr.p_app_private = &node;
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
    // LD-CBR with no reordering emits exactly one packet per input frame.
    ASSERT_EQ(static_cast<size_t>(kNumFrames), packets.size());

    std::unique_ptr<RefDecoder> decoder(
        // Sequence-header parsing only runs with the analyzer enabled, and
        // that is what populates frame_id_numbers_present_flag.
        create_reference_decoder(/*enable_analyzer=*/true));
    ASSERT_NE(nullptr, decoder.get());
    for (size_t i = 0; i < packets.size(); ++i) {
        RefDecoder::RefDecoderErr drc =
            decoder->decode(packets[i].data(), (uint32_t)packets[i].size());
        ASSERT_EQ(RefDecoder::REF_CODEC_OK, drc)
            << "decode failed for packet " << i;
        VideoFrame vf;
        while (decoder->get_frame(vf) == RefDecoder::REF_CODEC_OK) {
        }
    }
    const RefDecoder::StreamInfo* info = decoder->get_stream_info();
    ASSERT_NE(nullptr, info);
    *frame_id_flag = info->frame_id_numbers_present_flag;
    bool clean = true;
    for (size_t i = 0; i < info->frame_corrupted_list.size(); ++i)
        if (info->frame_corrupted_list[i])
            clean = false;
    *all_clean = clean;
}

TEST(FrameIdBitstreamTest, PresentWhenLtrEnabled) {
    int flag = -2;
    bool clean = false;
    encode_decode(/*max_managed_refs=*/4, &flag, &clean);
    EXPECT_EQ(1, flag)
        << "frame_id_numbers_present_flag must be set when LTR is enabled";
    EXPECT_TRUE(clean) << "frame_id stream must decode without corruption";
}

TEST(FrameIdBitstreamTest, AbsentWhenLtrDisabled) {
    int flag = -2;
    bool clean = false;
    encode_decode(/*max_managed_refs=*/0, &flag, &clean);
    EXPECT_EQ(0, flag)
        << "frame_id must stay off for non-LTR streams (bit-exact legacy path)";
    EXPECT_TRUE(clean);
}

TEST(FrameIdBitstreamTest, UseAnchorDecodesClean) {
    // A USE emits a real, non-zero delta_frame_id for the anchor slot, which
    // the decoder must round-trip against its own ref_frame_id[] to accept the
    // frame.
    //
    // This does not establish that the redirect took effect: a USE that was
    // dropped encodes ordinary references and also decodes cleanly. Proving
    // that needs the slots the decoder predicted from, and no libaom control
    // reports them -- AOMD_GET_LAST_REF_USED is declared in aomdx.h but absent
    // from the AV1 control table, and the inspection API exposes per-block
    // reference types rather than DPB indices.
    int flag = -2;
    bool clean = false;
    encode_decode(/*max_managed_refs=*/4, &flag, &clean, /*use_frame=*/12);
    EXPECT_EQ(1, flag);
    EXPECT_TRUE(clean)
        << "frame_id stream with an LTR USE (real delta_frame_id) must decode "
           "without corruption";
}

}  // namespace
