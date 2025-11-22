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
 * @file SvtAv1EncFastDecodeApiTest.cc
 *
 * @brief Encodes still images with fast_decode 3 and checks the coding tools
 *        signalled in the sequence and frame headers.
 *
 * fast_decode 3 turns CDEF, loop restoration and filter intra off and
 * quantization matrices on for still images (avif), unless the option was set
 * explicitly. Other encodes must be identical to fast_decode 2. fast_decode 3
 * is experimental: without the ENABLE_FAST_DECODE_3 build option it must be
 * rejected.
 ******************************************************************************/

#include <cstdint>
#include <string>
#include <vector>

#include "EbSvtAv1.h"
#include "EbSvtAv1Enc.h"
#include "gtest/gtest.h"

namespace {

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 128;

// Smooth 8-bit 4:2:0 content with some texture.
void fill_frame(std::vector<uint8_t>& yuv) {
    uint32_t seed = 4321u;
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            seed = seed * 1664525u + 1013904223u;
            const int dx = (int)x - 64;
            const int dy = (int)y - 48;
            yuv[y * kWidth + x] =
                (uint8_t)(48 + (((dx * dx + dy * dy) >> 4) & 0x7f) +
                          (int)(seed >> 28));
        }
    }
    for (uint32_t i = kWidth * kHeight; i < yuv.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        yuv[i] = (uint8_t)(100 + ((i / 5) & 0x3f) + (int)(seed >> 29));
    }
}

struct Option {
    const char* name;
    const char* value;
};

struct EncodeParams {
    EncodeParams(uint8_t fd, bool still,
                 std::vector<Option> opts = std::vector<Option>())
        : fast_decode(fd), avif(still), parsed(opts) {
    }
    uint8_t fast_decode;
    bool avif;
    std::vector<Option> parsed;
    // direct assignments, applied after the parsed options
    int enable_qm = -1;
    uint8_t user_opts = 0;
};

std::vector<uint8_t> encode(const EncodeParams& prm) {
    std::vector<uint8_t> bitstream;
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.frame_rate_numerator = 30;
    cfg.frame_rate_denominator = 1;
    cfg.encoder_bit_depth = 8;
    cfg.enc_mode = 6;
    cfg.qp = 30;
    cfg.rate_control_mode = SVT_AV1_RC_MODE_CQP_OR_CRF;
    cfg.avif = prm.avif;
    cfg.fast_decode = prm.fast_decode;
    cfg.level_of_parallelism = 1;
    for (const Option& o : prm.parsed) {
        EXPECT_EQ(EB_ErrorNone,
                  svt_av1_enc_parse_parameter(&cfg, o.name, o.value))
            << o.name;
    }
    if (prm.enable_qm >= 0) {
        cfg.enable_qm = prm.enable_qm != 0;
    }
    cfg.fast_decode_user_opts |= prm.user_opts;
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_set_parameter(enc, &cfg));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init(enc));

    std::vector<uint8_t> yuv(kWidth * kHeight * 3 / 2);
    fill_frame(yuv);
    EbSvtIOFormat frame = {yuv.data(),
                           yuv.data() + kWidth * kHeight,
                           yuv.data() + kWidth * kHeight * 5 / 4,
                           kWidth,
                           kWidth / 2,
                           kWidth / 2};
    EbBufferHeaderType in{};
    in.size = sizeof(in);
    in.p_buffer = reinterpret_cast<uint8_t*>(&frame);
    in.n_filled_len = (uint32_t)yuv.size();
    in.pts = 0;
    in.pic_type = EB_AV1_INVALID_PICTURE;
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &in));
    EbBufferHeaderType eos_hdr{};
    eos_hdr.size = sizeof(eos_hdr);
    eos_hdr.flags = EB_BUFFERFLAG_EOS;
    eos_hdr.pic_type = EB_AV1_INVALID_PICTURE;
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(enc, &eos_hdr));

    bool eos = false;
    while (!eos) {
        EbBufferHeaderType* pkt = nullptr;
        if (svt_av1_enc_get_packet(enc, &pkt, 1) != EB_ErrorNone) {
            break;
        }
        eos = (pkt->flags & EB_BUFFERFLAG_EOS) != 0;
        bitstream.insert(
            bitstream.end(), pkt->p_buffer, pkt->p_buffer + pkt->n_filled_len);
        svt_av1_enc_release_out_buffer(&pkt);
    }

    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(enc));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
    return bitstream;
}

class BitReader {
  public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {
    }
    uint32_t f(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) {
            const size_t byte = pos_ >> 3;
            const uint32_t bit =
                byte < size_ ? (data_[byte] >> (7 - (pos_ & 7))) & 1 : 0;
            v = (v << 1) | bit;
            ++pos_;
        }
        return v;
    }
    uint32_t ns(uint32_t n) {
        int w = 0;
        for (uint32_t x = n; x; x >>= 1) {
            ++w;
        }
        const uint32_t m = (1u << w) - n;
        const uint32_t v = f(w - 1);
        if (v < m) {
            return v;
        }
        return (v << 1) - m + f(1);
    }

  private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
};

int tile_log2(uint32_t blk_size, uint32_t target) {
    int k = 0;
    for (; (blk_size << k) < target; ++k) {
    }
    return k;
}

// The coding tools fast_decode 3 controls, as signalled for a reduced still
// picture header stream.
struct Headers {
    bool found_seq = false;
    bool found_frame = false;
    uint32_t enable_filter_intra = 0;
    uint32_t enable_cdef = 0;
    uint32_t enable_restoration = 0;
    uint32_t using_qmatrix = 0;
};

Headers parse(const std::vector<uint8_t>& bs) {
    Headers h;
    uint32_t width = 0, height = 0, use_128 = 0, enable_superres = 0, mono = 0,
             separate_uv_delta_q = 0;
    size_t pos = 0;
    while (pos < bs.size()) {
        const uint8_t hdr = bs[pos++];
        const uint32_t type = (hdr >> 3) & 0xf;
        if (hdr & 4) {
            ++pos;  // extension
        }
        uint64_t size = 0;
        if (hdr & 2) {
            for (int i = 0; i < 8 && pos < bs.size(); ++i) {
                const uint8_t b = bs[pos++];
                size |= (uint64_t)(b & 0x7f) << (7 * i);
                if (!(b & 0x80)) {
                    break;
                }
            }
        } else {
            size = bs.size() - pos;
        }
        if (pos + size > bs.size()) {
            break;
        }
        BitReader r(&bs[pos], (size_t)size);
        if (type == 1) {  // OBU_SEQUENCE_HEADER
            const uint32_t profile = r.f(3);
            r.f(1);  // still_picture
            const uint32_t reduced = r.f(1);
            EXPECT_EQ(1u, reduced)
                << "avif encodes use a reduced still picture header";
            if (!reduced) {
                return h;
            }
            r.f(5);  // seq_level_idx
            const int wbits = (int)r.f(4) + 1;
            const int hbits = (int)r.f(4) + 1;
            width = r.f(wbits) + 1;
            height = r.f(hbits) + 1;
            use_128 = r.f(1);
            h.enable_filter_intra = r.f(1);
            r.f(1);  // enable_intra_edge_filter
            enable_superres = r.f(1);
            h.enable_cdef = r.f(1);
            h.enable_restoration = r.f(1);
            const uint32_t high_bitdepth = r.f(1);
            if (profile == 2 && high_bitdepth) {
                r.f(1);  // twelve_bit
            }
            mono = profile == 1 ? 0 : r.f(1);
            if (r.f(1)) {  // color_description_present_flag
                r.f(24);
            }
            if (!mono) {
                r.f(1);  // color_range
                if (profile == 0) {
                    r.f(2);  // chroma_sample_position
                }
                separate_uv_delta_q = r.f(1);
            }
            h.found_seq = true;
        } else if ((type == 3 || type == 6) && h.found_seq &&
                   !h.found_frame) {  // OBU_FRAME_HEADER / OBU_FRAME
            r.f(1);                   // disable_cdf_update
            const uint32_t allow_sct =
                r.f(1);  // seq_force_screen_content_tools is SELECT
            if (allow_sct) {
                r.f(1);  // force_integer_mv
            }
            if (enable_superres && r.f(1)) {
                ADD_FAILURE() << "superres is not expected";
                return h;
            }
            if (r.f(1)) {  // render_and_frame_size_different
                r.f(32);
            }
            if (allow_sct) {
                r.f(1);  // allow_intrabc
            }
            // tile_info()
            const uint32_t mi_cols = 2 * ((width + 7) >> 3);
            const uint32_t mi_rows = 2 * ((height + 7) >> 3);
            const uint32_t sb_cols =
                use_128 ? (mi_cols + 31) >> 5 : (mi_cols + 15) >> 4;
            const uint32_t sb_rows =
                use_128 ? (mi_rows + 31) >> 5 : (mi_rows + 15) >> 4;
            const int max_log2_cols = tile_log2(1, sb_cols < 64 ? sb_cols : 64);
            const int max_log2_rows = tile_log2(1, sb_rows < 64 ? sb_rows : 64);
            if (r.f(1)) {  // uniform_tile_spacing_flag
                int cols_log2 = 0;
                while (cols_log2 < max_log2_cols && r.f(1)) {
                    ++cols_log2;
                }
                int rows_log2 = 0;
                while (rows_log2 < max_log2_rows && r.f(1)) {
                    ++rows_log2;
                }
                if (cols_log2 || rows_log2) {
                    ADD_FAILURE() << "a single tile is expected";
                    return h;
                }
            } else {
                for (uint32_t start = 0, n = 0; start < sb_cols; ++n) {
                    start += r.ns(sb_cols - start) + 1;
                    if (n) {
                        ADD_FAILURE() << "a single tile is expected";
                        return h;
                    }
                }
                for (uint32_t start = 0, n = 0; start < sb_rows; ++n) {
                    start += r.ns(sb_rows - start) + 1;
                    if (n) {
                        ADD_FAILURE() << "a single tile is expected";
                        return h;
                    }
                }
            }
            // quantization_params()
            r.f(8);  // base_q_idx
            auto delta_q = [&r]() {
                if (r.f(1)) {
                    r.f(7);
                }
            };
            delta_q();  // DeltaQYDc
            if (!mono) {
                const uint32_t diff_uv_delta = separate_uv_delta_q ? r.f(1) : 0;
                delta_q();  // DeltaQUDc
                delta_q();  // DeltaQUAc
                if (diff_uv_delta) {
                    delta_q();  // DeltaQVDc
                    delta_q();  // DeltaQVAc
                }
            }
            h.using_qmatrix = r.f(1);
            h.found_frame = true;
        }
        pos += (size_t)size;
    }
    return h;
}

Headers encode_and_parse(const EncodeParams& prm) {
    const std::vector<uint8_t> bs = encode(prm);
    EXPECT_FALSE(bs.empty());
    const Headers h = parse(bs);
    EXPECT_TRUE(h.found_seq);
    EXPECT_TRUE(h.found_frame);
    return h;
}

TEST(SvtAv1FastDecodeApiTest, Level2StillImageKeepsTools) {
    const Headers h = encode_and_parse(EncodeParams(2, true));
    EXPECT_EQ(1u, h.enable_filter_intra);
    EXPECT_EQ(1u, h.enable_cdef);
    EXPECT_EQ(1u, h.using_qmatrix);
}

#if CONFIG_ENABLE_FAST_DECODE_3
TEST(SvtAv1FastDecodeApiTest, Level3StillImageTools) {
    const Headers h = encode_and_parse(EncodeParams(3, true));
    EXPECT_EQ(0u, h.enable_filter_intra);
    EXPECT_EQ(0u, h.enable_cdef);
    EXPECT_EQ(0u, h.enable_restoration);
    EXPECT_EQ(1u, h.using_qmatrix);
}

TEST(SvtAv1FastDecodeApiTest, Level3ParsedOptionsWin) {
    const Headers h = encode_and_parse(
        EncodeParams(3, true, {{"enable-cdef", "1"}, {"enable-qm", "0"}}));
    EXPECT_EQ(1u, h.enable_cdef);
    EXPECT_EQ(0u, h.using_qmatrix);
    EXPECT_EQ(0u, h.enable_filter_intra);
}

TEST(SvtAv1FastDecodeApiTest, Level3DirectOptionsWithUserBitWin) {
    EncodeParams direct(3, true);
    direct.enable_qm = 0;
    direct.user_opts = SVT_AV1_FD_USER_QM;
    const Headers h = encode_and_parse(direct);
    EXPECT_EQ(0u, h.using_qmatrix);
    // the same stream as setting the option through the parser
    EXPECT_TRUE(encode(direct) ==
                encode(EncodeParams(3, true, {{"enable-qm", "0"}})));
}

TEST(SvtAv1FastDecodeApiTest, Level3DirectDefaultWithoutUserBitIsOverridden) {
    EncodeParams direct(3, true);
    direct.enable_qm = 0;
    const Headers h = encode_and_parse(direct);
    EXPECT_EQ(1u, h.using_qmatrix);
}

TEST(SvtAv1FastDecodeApiTest,
     Level3ExplicitValueEqualToLevel3ChoiceKeepsStream) {
    // enable-qm 1 marks QM as explicit; the QM levels are still fast_decode 3's
    EXPECT_TRUE(encode(EncodeParams(3, true)) ==
                encode(EncodeParams(3, true, {{"enable-qm", "1"}})));
}

TEST(SvtAv1FastDecodeApiTest, Level3WithoutAvifMatchesLevel2) {
    const std::vector<uint8_t> level2 = encode(EncodeParams(2, false));
    const std::vector<uint8_t> level3 = encode(EncodeParams(3, false));
    ASSERT_FALSE(level2.empty());
    EXPECT_TRUE(level2 == level3)
        << "fast_decode 3 without avif must behave as fast_decode 2";
}
#else
TEST(SvtAv1FastDecodeApiTest, Level3RejectedWithoutBuildOption) {
    EbComponentType* enc = nullptr;
    EbSvtAv1EncConfiguration cfg{};
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init_handle(&enc, &cfg));
    cfg.source_width = kWidth;
    cfg.source_height = kHeight;
    cfg.avif = true;
    cfg.fast_decode = 3;
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_set_parameter(enc, &cfg));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(enc));
}
#endif  // CONFIG_ENABLE_FAST_DECODE_3

}  // namespace
