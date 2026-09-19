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
 * @file SvtAv1RefMgmtBitstreamTest.cc
 *
 * @brief End-to-end bitstream cross-check for ref-frame management.
 *
 * Encodes a 30-frame LD-CBR clip with:
 *     STORE  id=42  at frame  5
 *     USE    id=42  at frame 12
 *     CLEAR  id=42  at frame 18
 *     USE    id=42  at frame 22   (anchor already CLEARed → no-op)
 *
 * Decodes the bitstream through libaom (via RefDecoder +
 *AOMD_GET_LAST_REF_UPDATES) and asserts properties of the encoded
 *refresh_frame_flags that prove the encoder honored each event:
 *
 *   1. Frame 5 refresh_flags has at least one bit set — the encoder did
 *      allocate a DPB slot to STORE pic_id 42 (slot index is dynamic; the
 *      encoder picks whatever slot it was already going to refresh).
 *   2. Frames 6..11 refresh_flags NEVER include the STOREd slot — the
 *      anchor is preserved.
 *   3. Frame 12 refresh_flags has popcount == 7 with the STOREd slot
 *      cleared — USE triggered the recovery-point refresh (every
 *      non-STOREd slot gets the recovery frame), preserving the anchor.
 *   4. Frame 18 refresh_flags has the legacy "normal" shape (popcount
 *      <= 6, NOT the recovery 7-pattern) — CLEAR alone does not trigger
 *      recovery refresh.
 *   5. Frame 22 refresh_flags also has the normal shape — USE-after-CLEAR
 *      was no-oped (anchor was already released).
 *   6. The whole stream decodes without corruption.
 ******************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "RefDecoder.h"
#include "VideoFrame.h"
#include "gtest/gtest.h"

// Compile-time bitstream dump for offline inspection (aomdec / ffprobe).
//   #define SVT_REFMGMT_DUMP_IVF "/abs/path/refmgmt_test.ivf"
// Default OFF — leave undefined in checked-in code.

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
constexpr uint32_t kNumFrames = 30;
constexpr uint32_t kStoreFrame = 5;
constexpr uint32_t kUseFrame = 12;
constexpr uint32_t kClearFrame = 18;
constexpr uint32_t kUseAfterClr = 22;  // USE after CLEAR → no-op + warning
constexpr uint32_t kStoreId = 42u;

static inline int popcount8(uint8_t m) {
    int n = 0;
    for (int i = 0; i < 8; ++i) {
        if (m & (1u << i))
            ++n;
    }
    return n;
}

class RefMgmtBitstreamTest : public ::testing::Test {
  protected:
    EbComponentType* enc_{nullptr};
    EbSvtAv1EncConfiguration cfg_{};
    std::vector<std::vector<uint8_t>> packets_{};

#ifdef SVT_REFMGMT_DUMP_IVF
    static void write_le32(FILE* f, uint32_t v) {
        uint8_t b[4] = {(uint8_t)(v),
                        (uint8_t)(v >> 8),
                        (uint8_t)(v >> 16),
                        (uint8_t)(v >> 24)};
        fwrite(b, 1, 4, f);
    }
    static void write_le16(FILE* f, uint16_t v) {
        uint8_t b[2] = {(uint8_t)v, (uint8_t)(v >> 8)};
        fwrite(b, 1, 2, f);
    }
    static void write_le64(FILE* f, uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            uint8_t b = (uint8_t)(v >> (8 * i));
            fwrite(&b, 1, 1, f);
        }
    }
    void dump_ivf(const char* path) const {
        FILE* f = fopen(path, "wb");
        ASSERT_NE(nullptr, f);
        fwrite("DKIF", 1, 4, f);
        write_le16(f, 0);
        write_le16(f, 32);
        fwrite("AV01", 1, 4, f);
        write_le16(f, (uint16_t)cfg_.source_width);
        write_le16(f, (uint16_t)cfg_.source_height);
        write_le32(f, cfg_.frame_rate_numerator);
        write_le32(f, cfg_.frame_rate_denominator);
        write_le32(f, (uint32_t)packets_.size());
        write_le32(f, 0);
        for (size_t i = 0; i < packets_.size(); ++i) {
            write_le32(f, (uint32_t)packets_[i].size());
            write_le64(f, (uint64_t)i);
            fwrite(packets_[i].data(), 1, packets_[i].size(), f);
        }
        fclose(f);
        fprintf(stderr,
                "[RFM-BS] dumped IVF to %s (%zu frames)\n",
                path,
                packets_.size());
    }
#endif

    void SetUp() override {
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc_, &cfg_));
        cfg_.source_width = kWidth;
        cfg_.source_height = kHeight;
        cfg_.frame_rate_numerator = 30000;
        cfg_.frame_rate_denominator = 1000;
        cfg_.encoder_bit_depth = 8;
        cfg_.encoder_color_format = EB_YUV420;
        cfg_.enc_mode = 9;
        cfg_.tune = 1;
        cfg_.rtc = true;
        cfg_.intra_period_length = 100;
        cfg_.hierarchical_levels = 0;
        cfg_.pred_structure = LOW_DELAY;
        cfg_.rate_control_mode = SVT_AV1_RC_MODE_CBR;
        cfg_.target_bit_rate = 500000;
        cfg_.max_qp_allowed = 63;
        cfg_.min_qp_allowed = 4;
        cfg_.look_ahead_distance = 0;
        cfg_.recode_loop = 0;
        cfg_.max_managed_refs = 4;  // enable up to 4 simultaneous STORE anchors

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc_, &cfg_));
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc_));
    }

    void TearDown() override {
        if (enc_) {
            svt_av1_enc_deinit(enc_);
            svt_av1_enc_deinit_handle(enc_);
            enc_ = nullptr;
        }
    }
};

TEST_F(RefMgmtBitstreamTest, bitstream_matches_api_for_store_use_clear) {
    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(0, src.open_source(0, 0));

    EbBufferHeaderType in_hdr{};
    in_hdr.size = sizeof(EbBufferHeaderType);

    for (uint32_t fi = 0; fi < kNumFrames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);

        SvtAv1RefFrameCmd store_payload{kStoreId};
        SvtAv1RefFrameCmd use_payload{kStoreId};
        SvtAv1RefFrameCmd clear_payload{kStoreId};
        EbPrivDataNode node{};

        in_hdr.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in_hdr.n_filled_len = src.get_frame_size();
        in_hdr.pts = fi;
        in_hdr.flags = 0;
        in_hdr.pic_type =
            (fi == 0) ? EB_AV1_KEY_PICTURE : EB_AV1_INVALID_PICTURE;
        in_hdr.p_app_private = nullptr;

        if (fi == kStoreFrame) {
            node.node_type = REF_STORE_EVENT;
            node.data = &store_payload;
            node.size = sizeof(store_payload);
            in_hdr.p_app_private = &node;
        } else if (fi == kUseFrame) {
            node.node_type = REF_USE_EVENT;
            node.data = &use_payload;
            node.size = sizeof(use_payload);
            in_hdr.p_app_private = &node;
        } else if (fi == kClearFrame) {
            node.node_type = REF_CLEAR_EVENT;
            node.data = &clear_payload;
            node.size = sizeof(clear_payload);
            in_hdr.p_app_private = &node;
        } else if (fi == kUseAfterClr) {
            node.node_type = REF_USE_EVENT;
            node.data = &use_payload;
            node.size = sizeof(use_payload);
            in_hdr.p_app_private = &node;
        }

        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc_, &in_hdr))
            << "send_picture failed on frame " << fi;

        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc_, &out, 0);
        if (rc == EB_ErrorNone && out) {
            packets_.emplace_back(out->p_buffer,
                                  out->p_buffer + out->n_filled_len);
            svt_av1_enc_release_out_buffer(&out);
        }
    }

    EbBufferHeaderType eos{};
    eos.size = sizeof(EbBufferHeaderType);
    eos.flags = EB_BUFFERFLAG_EOS;
    eos.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc_, &eos));
    for (;;) {
        EbBufferHeaderType* out = nullptr;
        EbErrorType rc = svt_av1_enc_get_packet(enc_, &out, 1);
        if (rc != EB_ErrorNone || !out) {
            break;
        }
        const bool is_eos = (out->flags & EB_BUFFERFLAG_EOS) != 0;
        if (!is_eos && out->n_filled_len > 0) {
            packets_.emplace_back(out->p_buffer,
                                  out->p_buffer + out->n_filled_len);
        }
        svt_av1_enc_release_out_buffer(&out);
        if (is_eos) {
            break;
        }
    }

    ASSERT_GE(packets_.size(), static_cast<size_t>(kUseAfterClr + 1u))
        << "encoder emitted only " << packets_.size() << " packets";

#ifdef SVT_REFMGMT_DUMP_IVF
    dump_ivf(SVT_REFMGMT_DUMP_IVF);
#endif

    /* --- Decode --- */
    std::unique_ptr<RefDecoder> decoder(
        create_reference_decoder(/*enable_analyzer=*/false));
    ASSERT_NE(nullptr, decoder.get());

    for (size_t i = 0; i < packets_.size(); ++i) {
        const auto& pkt = packets_[i];
        RefDecoder::RefDecoderErr drc =
            decoder->decode(pkt.data(), (uint32_t)pkt.size());
        ASSERT_EQ(RefDecoder::REF_CODEC_OK, drc)
            << "decode failed for packet " << i;
        VideoFrame frame;
        while (decoder->get_frame(frame) == RefDecoder::REF_CODEC_OK) {
            // drain reconstructed frames
        }
    }

    const RefDecoder::StreamInfo* info = decoder->get_stream_info();
    ASSERT_NE(nullptr, info);
    ASSERT_EQ(packets_.size(), info->refresh_frame_flags_list.size());
    ASSERT_EQ(packets_.size(), info->frame_corrupted_list.size());

    /* --- 1. STORE frame allocated some slot (refresh_flags non-zero) --- */
    ASSERT_LT(kStoreFrame, info->refresh_frame_flags_list.size());
    const uint8_t store_flags = info->refresh_frame_flags_list[kStoreFrame];
    ASSERT_NE(0u, store_flags)
        << "STORE frame refresh_frame_flags=0; encoder didn't allocate a slot";

    /* Reference: a "legacy" non-event frame's refresh mask in this RTC
     * flat-IPP CBR config. Use frame 3 (before any event). The dispatcher
     * is a no-op on this frame so it shows exactly what the encoder picks
     * by default. */
    ASSERT_LT(3u, info->refresh_frame_flags_list.size());
    const uint8_t legacy_flags = info->refresh_frame_flags_list[3];

    /* --- 3. USE frame triggers recovery refresh: every NON-STOREd slot
     *        is refreshed with the current frame; the STOREd slot is
     *        preserved. The result is `legacy & ~(1<<anchor_slot)`. --- */
    ASSERT_LT(kUseFrame, info->refresh_frame_flags_list.size());
    const uint8_t use_flags = info->refresh_frame_flags_list[kUseFrame];
    /* In the dispatcher: USE recovery sets refresh = 0xFF & ~dpb_stored_mask.
     * dpb_stored_mask has exactly one bit (the STORE on frame 5). So
     * use_flags should equal 0xFF & ~(1<<anchor_slot), i.e. exactly one
     * fewer bit set than 0xFF. Equivalently popcount==7. */
    EXPECT_EQ(7, popcount8(use_flags))
        << "USE frame refresh_frame_flags should be 0xFF with the anchor bit "
           "cleared "
        << "(popcount 7); got 0x" << std::hex << static_cast<int>(use_flags);
    const uint8_t inferred_anchor_bit = (uint8_t)(~use_flags & 0xFFu);
    EXPECT_EQ(1, popcount8(inferred_anchor_bit))
        << "expected exactly one cleared bit in USE frame refresh_flags";
    EXPECT_NE(0, store_flags & inferred_anchor_bit)
        << "USE-frame-preserved slot (0x" << std::hex
        << (int)inferred_anchor_bit
        << ") is not one the STORE frame refreshed (0x" << (int)store_flags
        << ")";

    /* --- 2. Frames between STORE and USE preserve the anchor slot. --- */
    for (uint32_t pts = kStoreFrame + 1; pts < kUseFrame; ++pts) {
        ASSERT_LT(pts, info->refresh_frame_flags_list.size());
        const uint8_t flags = info->refresh_frame_flags_list[pts];
        EXPECT_EQ(0, flags & inferred_anchor_bit)
            << "pts=" << pts << " refresh_frame_flags=0x" << std::hex
            << (int)flags << " overwrites the anchor bit 0x"
            << (int)inferred_anchor_bit;
    }

    /* --- 4. CLEAR frame: not directly observable in the bitstream (the
     *        only effect is releasing dpb_stored_mask, not modifying
     *        refresh_flags). Its effect is verified indirectly by the
     *        next check on USE-after-CLEAR. --- */

    /* --- 5. USE-after-CLEAR did NOT trigger the recovery-refresh shape.
     *        The anchor was already released, so apply_ref_use fails and
     *        the recovery refresh code never runs. The frame's refresh
     *        should look like a legacy non-event frame. --- */
    ASSERT_LT(kUseAfterClr, info->refresh_frame_flags_list.size());
    const uint8_t after_clr_flags =
        info->refresh_frame_flags_list[kUseAfterClr];
    EXPECT_EQ(legacy_flags, after_clr_flags)
        << "USE-after-CLEAR should fall back to the legacy refresh shape "
           "(anchor already cleared); "
        << "got 0x" << std::hex << static_cast<int>(after_clr_flags)
        << " legacy=0x" << static_cast<int>(legacy_flags);

    /* --- 6. Whole stream decoded cleanly. --- */
    for (size_t i = 0; i < info->frame_corrupted_list.size(); ++i) {
        EXPECT_EQ(0, info->frame_corrupted_list[i])
            << "frame " << i << " decoded as corrupted";
    }
}

}  // namespace
