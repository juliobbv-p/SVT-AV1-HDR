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
 * @file SvtAv1EncMetadataApiTest.cc
 *
 * @brief Checks that per-frame ITU-T T.35 (HDR10+) metadata reaches the
 *        bitstream on the picture it was attached to.
 *
 * Every input frame carries one T.35 payload holding its own index. Each
 * output packet displays exactly one picture (its pts), so each packet must
 * carry exactly one T.35 payload, and that payload must hold the packet's pts.
 *
 * In random access, hidden (not shown) frames are displayed later by a
 * show_existing_frame, which has to pick up the hidden frame's metadata. With
 * overlays enabled, the alt-ref is hidden for good and its overlay carries the
 * metadata instead, while most overlay candidates are dropped without being
 * coded; their metadata must neither leak into a later overlay nor pile up.
 ******************************************************************************/

#include <cstdint>
#include <string>
#include <vector>

#include "DummyVideoSource.h"
#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "EbSvtAv1Metadata.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;
constexpr uint8_t kT35Prefix[] = {0xB5, 0x00, 0x3C, 'S', 'V', 'T'};

struct MetadataParam {
    uint32_t frames;
    int8_t enc_mode;
    uint32_t hierarchical_levels;
    bool enable_overlays;
    // bytes appended to the 8-byte tagged payload; > 119 needs a two-byte OBU
    // length field, as real HDR10+ payloads do
    uint32_t extra_payload;
};

size_t read_leb128(const uint8_t* p, const uint8_t* end, uint64_t* value) {
    *value = 0;
    for (size_t i = 0; i < 8 && p + i < end; ++i) {
        *value |= (uint64_t)(p[i] & 0x7f) << (7 * i);
        if (!(p[i] & 0x80)) {
            return i + 1;
        }
    }
    return 0;
}

// Returns the frame indices found in the tagged T.35 payloads of a packet.
std::vector<int64_t> tagged_t35_in_packet(const EbBufferHeaderType* pkt) {
    std::vector<int64_t> tags;
    const uint8_t* p = pkt->p_buffer;
    const uint8_t* end = p + pkt->n_filled_len;
    while (p < end) {
        const int obu_type = (p[0] >> 3) & 0xf;
        const int has_extension = (p[0] >> 2) & 1;
        uint64_t obu_size;
        const uint8_t* size_field = p + 1 + has_extension;
        const size_t leb_len = read_leb128(size_field, end, &obu_size);
        if (!leb_len) {
            ADD_FAILURE() << "malformed OBU at pts " << pkt->pts;
            break;
        }
        const uint8_t* payload = size_field + leb_len;
        if (payload + obu_size > end) {
            ADD_FAILURE() << "truncated OBU at pts " << pkt->pts;
            break;
        }
        if (obu_type == 5 /* OBU_METADATA */) {
            uint64_t metadata_type;
            const size_t n =
                read_leb128(payload, payload + obu_size, &metadata_type);
            const uint8_t* t35 = payload + n;
            if (metadata_type == EB_AV1_METADATA_TYPE_ITUT_T35 &&
                obu_size >= n + 8 &&
                !memcmp(t35, kT35Prefix, sizeof(kT35Prefix))) {
                tags.push_back((t35[6] << 8) | t35[7]);
            }
        }
        p = payload + obu_size;
    }
    return tags;
}

class SvtAv1MetadataApiTest : public ::testing::TestWithParam<MetadataParam> {};

TEST_P(SvtAv1MetadataApiTest, T35FollowsItsPicture) {
    const MetadataParam& prm = GetParam();

    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    cfg.encoder_bit_depth = 8;
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.enc_mode = prm.enc_mode;
    cfg.hierarchical_levels = prm.hierarchical_levels;
    cfg.enable_overlays = prm.enable_overlays;
    cfg.intra_period_length = -1;
    cfg.level_of_parallelism = 4;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    svt_av1_video_source::DummyVideoSource src(IMG_FMT_420, kWidth, kHeight, 8);
    ASSERT_EQ(EB_ErrorNone, src.open_source(0, prm.frames));

    uint32_t packets = 0;
    bool eos = false;
    auto drain = [&](bool done_sending) {
        EbBufferHeaderType* pkt = nullptr;
        while (!eos && svt_av1_enc_get_packet(enc, &pkt, done_sending) ==
                           EB_ErrorNone) {
            eos = (pkt->flags & EB_BUFFERFLAG_EOS) != 0;
            if (pkt->n_filled_len) {
                const std::vector<int64_t> tags = tagged_t35_in_packet(pkt);
                std::string got;
                for (int64_t t : tags) {
                    got += std::to_string(t) + " ";
                }
                EXPECT_TRUE(tags.size() == 1 && tags[0] == pkt->pts)
                    << "packet pts " << pkt->pts << " carries T.35 of frames [ "
                    << got << "]";
                ++packets;
            }
            svt_av1_enc_release_out_buffer(&pkt);
        }
    };

    std::vector<uint8_t> t35(8 + prm.extra_payload);
    for (uint32_t fi = 0; fi < prm.frames; ++fi) {
        EbSvtIOFormat* frame = src.get_next_frame();
        ASSERT_NE(nullptr, frame);
        EbBufferHeaderType in{};
        in.size = sizeof(in);
        in.p_buffer = reinterpret_cast<uint8_t*>(frame);
        in.n_filled_len = src.get_frame_size();
        in.pts = fi;
        in.pic_type = EB_AV1_INVALID_PICTURE;
        memcpy(t35.data(), kT35Prefix, sizeof(kT35Prefix));
        t35[6] = (uint8_t)(fi >> 8);
        t35[7] = (uint8_t)fi;
        for (size_t i = 8; i < t35.size(); ++i) {
            t35[i] = (uint8_t)(i + fi);
        }
        ASSERT_EQ(
            0,
            svt_add_metadata(
                &in, EB_AV1_METADATA_TYPE_ITUT_T35, t35.data(), t35.size()));
        // send_picture copies the metadata
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in));
        svt_metadata_array_free(&in.metadata);
        drain(false);
    }
    EbBufferHeaderType eos_hdr{};
    eos_hdr.size = sizeof(eos_hdr);
    eos_hdr.flags = EB_BUFFERFLAG_EOS;
    eos_hdr.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &eos_hdr));
    drain(true);

    EXPECT_TRUE(eos);
    EXPECT_EQ(prm.frames, packets);
    src.close_source();
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(enc));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
}

INSTANTIATE_TEST_SUITE_P(Metadata, SvtAv1MetadataApiTest,
                         ::testing::Values(MetadataParam{80, 10, 4, false, 0},
                                           MetadataParam{80, 10, 5, false, 300},
                                           MetadataParam{120, 10, 4, true, 0},
                                           MetadataParam{
                                               120, 8, 5, true, 300}));

}  // namespace
