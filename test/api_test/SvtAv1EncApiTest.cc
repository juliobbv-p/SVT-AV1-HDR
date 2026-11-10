/*
 * Copyright(c) 2019 Netflix, Inc.
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
 * @file SvtAv1EncApiTest.cc
 *
 * @brief SVT-AV1 encoder api test, check invalid input
 *
 * @author Cidana-Edmond, Cidana-Ryan, Cidana-Wenyao
 *
 ******************************************************************************/
#include "EbSvtAv1Enc.h"
#include "EbSvtAv1Metadata.h"
#include "gtest/gtest.h"
#include "SvtAv1EncApiTest.h"

using namespace svt_av1_test;

namespace {

TEST(EncApiTest, set_parameter_null_configuration) {
    SvtAv1Context context{};

    ASSERT_EQ(
        EB_ErrorNone,
        svt_av1_enc_init_handle(&context.enc_handle, &context.enc_params));
    ASSERT_NE(nullptr, context.enc_handle);
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_set_parameter(context.enc_handle, nullptr));

    // Rejecting the null configuration must leave the handle usable.
    context.enc_params.source_width = 64;
    context.enc_params.source_height = 64;
    context.enc_params.enc_mode = 13;
    context.enc_params.level_of_parallelism = 1;
    EXPECT_EQ(
        EB_ErrorNone,
        svt_av1_enc_set_parameter(context.enc_handle, &context.enc_params));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(context.enc_handle));
}

/** @brief check_null_pointer is a api test case
 * EncApiTest.check_null_pointer is a api test case for checking null pointer
 * parameters setting into api functions and expect report for a
 * EB_ErrorBadParameter return
 *
 * Test strategy: <br>
 * Input nullptr to the encoder API and check the return value.
 *
 * Expected result: <br>
 * Encoder API should not crash and report EB_ErrorBadParameter.
 *
 * Test coverage:
 * All the encoder parameters.
 */
TEST(EncApiTest, check_null_pointer) {
    SvtAv1Context context{};
    EbBufferHeaderType buffer{};
    EbBufferHeaderType *packet = nullptr;
    SvtAv1FixedBuf info{};

    // initialize encoder and with all null pointer
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_init_handle(nullptr, nullptr));
    // initialize encoder and with all null pointer and get handle
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_init_handle(&context.enc_handle, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_init_handle(nullptr, &context.enc_params));
    // setup encoder parameters with null pointer
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_set_parameter(nullptr, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_set_parameter(nullptr, &context.enc_params));
    // open encoder with null pointer
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_init(nullptr));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_stream_header(nullptr, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_stream_header(nullptr, &packet));
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(nullptr, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_send_picture(nullptr, &buffer));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_get_packet(nullptr, nullptr, 0));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_get_packet(nullptr, &packet, 0));
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_get_recon(nullptr, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_get_recon(nullptr, &buffer));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_get_stream_info(
                  nullptr, SVT_AV1_STREAM_INFO_FIRST_PASS_STATS_OUT, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_get_stream_info(
                  nullptr, SVT_AV1_STREAM_INFO_FIRST_PASS_STATS_OUT, &info));
    svt_av1_enc_release_out_buffer(nullptr);
    // close encoder with null pointer
    EXPECT_EQ(EB_ErrorBadParameter, svt_av1_enc_deinit(nullptr));
    // destroy encoder handle with null pointer
    EXPECT_EQ(EB_ErrorInvalidComponent, svt_av1_enc_deinit_handle(nullptr));
    SUCCEED();
}

TEST(EncApiTest, release_null_output_buffer) {
    EbBufferHeaderType *packet = nullptr;
    svt_av1_enc_release_out_buffer(&packet);
    EXPECT_EQ(nullptr, packet);
}

TEST(EncApiTest, null_buffers_do_not_disrupt_encoding) {
    SvtAv1Context context{};
    ASSERT_EQ(
        EB_ErrorNone,
        svt_av1_enc_init_handle(&context.enc_handle, &context.enc_params));
    context.enc_params.source_width = 64;
    context.enc_params.source_height = 64;
    context.enc_params.enc_mode = 13;
    context.enc_params.level_of_parallelism = 1;
    context.enc_params.recon_enabled = true;
    ASSERT_EQ(
        EB_ErrorNone,
        svt_av1_enc_set_parameter(context.enc_handle, &context.enc_params));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(context.enc_handle));

    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_stream_header(context.enc_handle, nullptr));
    EbBufferHeaderType *header = nullptr;
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_stream_header(context.enc_handle, &header));
    ASSERT_NE(nullptr, header);
    EXPECT_GT(header->n_filled_len, 0u);
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_stream_header_release(header));

    EXPECT_EQ(
        EB_ErrorBadParameter,
        svt_av1_enc_get_stream_info(context.enc_handle,
                                    SVT_AV1_STREAM_INFO_FIRST_PASS_STATS_OUT,
                                    nullptr));
    SvtAv1FixedBuf info{};
    EXPECT_EQ(
        EB_ErrorNone,
        svt_av1_enc_get_stream_info(context.enc_handle,
                                    SVT_AV1_STREAM_INFO_FIRST_PASS_STATS_OUT,
                                    &info));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_send_picture(context.enc_handle, nullptr));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_get_packet(context.enc_handle, nullptr, 0));
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_get_recon(context.enc_handle, nullptr));

    std::vector<uint8_t> pixels(64 * 64 * 3 / 2, 128);
    EbSvtIOFormat frame = {pixels.data(),
                           pixels.data() + 64 * 64,
                           pixels.data() + 64 * 64 * 5 / 4,
                           64,
                           32,
                           32};
    EbBufferHeaderType input{};
    input.size = sizeof(input);
    input.p_buffer = reinterpret_cast<uint8_t *>(&frame);
    input.n_filled_len = static_cast<uint32_t>(pixels.size());
    input.pic_type = EB_AV1_INVALID_PICTURE;
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_send_picture(context.enc_handle, &input));
    EbBufferHeaderType eos{};
    eos.size = sizeof(eos);
    eos.flags = EB_BUFFERFLAG_EOS;
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_send_picture(context.enc_handle, &eos));

    // A rejected output pointer must not consume a queued packet or recon.
    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_enc_get_packet(context.enc_handle, nullptr, 1));
    bool got_eos = false;
    unsigned frames = 0;
    EbBufferHeaderType *packet = nullptr;
    while (!got_eos && svt_av1_enc_get_packet(context.enc_handle, &packet, 1) ==
                           EB_ErrorNone) {
        got_eos = (packet->flags & EB_BUFFERFLAG_EOS) != 0;
        frames += packet->n_filled_len != 0;
        svt_av1_enc_release_out_buffer(&packet);
    }
    EXPECT_TRUE(got_eos);
    EXPECT_EQ(1u, frames);

    EXPECT_EQ(EB_ErrorBadParameter,
              svt_av1_get_recon(context.enc_handle, nullptr));
    EbBufferHeaderType recon{};
    recon.size = sizeof(recon);
    recon.p_buffer = pixels.data();
    recon.n_alloc_len = static_cast<uint32_t>(pixels.size());
    EXPECT_EQ(EB_ErrorNone, svt_av1_get_recon(context.enc_handle, &recon));
    EXPECT_EQ(pixels.size(), recon.n_filled_len);
    svt_metadata_array_free(&recon.metadata);
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(context.enc_handle));
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(context.enc_handle));
}

/** @brief check_normal_setup is a api test case
 * EncApiTest.check_normal_setup is a api test case with a normal setup
 * parameters into api functions and expect report for return EB_ErrorNone
 *
 * Test strategy: <br>
 * Input normal parameters to the encoder API and check the return value.
 *
 * Expected result: <br>
 * Encoder API should not crash and report EB_ErrorNone.
 *
 * Test coverage:
 * All the encoder parameters.
 *
 * Comments:
 * Disabled for it will hang after test report, might be thread can not exit
 * only happens without in IDE debugging mode.
 */
TEST(EncApiTest, DISABLED_check_normal_setup) {
    SvtAv1Context context{};

    const int width = 1280;
    const int height = 720;

    // initialize encoder and get handle
    EXPECT_EQ(EB_ErrorNone,
              svt_av1_enc_init_handle(&context.enc_handle, &context.enc_params))
        << "svt_av1_enc_init_handle failed";
    // setup source width/height with default value
    context.enc_params.source_width = width;
    context.enc_params.source_height = height;
    // setup encoder parameters with all default
    EXPECT_EQ(
        EB_ErrorNone,
        svt_av1_enc_set_parameter(context.enc_handle, &context.enc_params))
        << "svt_av1_enc_set_parameter failed";
    // open encoder
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_init(context.enc_handle))
        << "svt_av1_enc_init failed";
    // close encoder
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit(context.enc_handle))
        << "svt_av1_enc_deinit failed";
    // destroy encoder
    EXPECT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(context.enc_handle))
        << "svt_av1_enc_deinit_handle failed";
}

/** @brief repeat_normal_setup is a api test case
 * EncApiTest.repeat_normal_setup is a api test case of repeating test with a
 * default normal setup to check for a resource or memory leak
 *
 * Test strategy: <br>
 * Input normal parameters to the encoder API and repeat the processing of
 * initialize and destroy encoder handle and check the return value.
 *
 * Expected result: <br>
 * Encoder can initialize normally without any error reported.
 *
 * Test coverage:
 * Initialize and destroy APIs.
 *
 * Comments:
 * Disabled for it causes memory leak, and lead to effect other tests
 */
TEST(EncApiTest, DISABLED_repeat_normal_setup) {
    SvtAv1Context context{};

    const int width = 1280;
    const int height = 720;

    for (size_t i = 0; i < 500; ++i) {
        // initialize encoder and get handle
        ASSERT_EQ(
            EB_ErrorNone,
            svt_av1_enc_init_handle(&context.enc_handle, &context.enc_params))
            << "svt_av1_enc_init_handle failed at " << i << " times";
        // setup source width/height with default value
        context.enc_params.source_width = width;
        context.enc_params.source_height = height;
        // setup encoder parameters with all default
        ASSERT_EQ(
            EB_ErrorNone,
            svt_av1_enc_set_parameter(context.enc_handle, &context.enc_params))
            << "svt_av1_enc_set_parameter failed at " << i << " times";
        // TODO:if not calls svt_av1_enc_deinit, there is huge memory leak, fix
        // me
        // ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit(context.enc_handle))
        //    << "svt_av1_enc_deinit failed";
        // destroy encoder
        ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(context.enc_handle))
            << "svt_av1_enc_deinit_handle failed at " << i << " times";
    }
}

}  // namespace

/**
 * @brief Small resolution encoding tests
 *
 * Test strategy:
 * Initialize encoder with small resolutions, send a dummy frame,
 * retrieve output, and verify no crashes or hangs occur.
 *
 * Covers: 1x1, 3x3, 4x4, 64x1, 65x1, 360x1, 1x360
 * These test the minimum dimension support and the segment count
 * fix for non-square small images.
 */

struct SmallResParam {
    uint32_t width;
    uint32_t height;
};

using SmallResEncodeTest = ::testing::TestWithParam<SmallResParam>;

TEST_P(SmallResEncodeTest, encode_one_frame) {
    const SmallResParam &p = GetParam();
    svt_av1_test::SvtAv1Context ctxt{};

    // init handle
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_init_handle(&ctxt.enc_handle, &ctxt.enc_params));

    // configure for small resolution AVIF-style encoding
    ctxt.enc_params.source_width = p.width;
    ctxt.enc_params.source_height = p.height;
    ctxt.enc_params.encoder_bit_depth = 8;
    ctxt.enc_params.encoder_color_format = EB_YUV420;
    ctxt.enc_params.pred_structure = ALL_INTRA;

    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_set_parameter(ctxt.enc_handle, &ctxt.enc_params));

    // open encoder
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_init(ctxt.enc_handle));

    // allocate and send a dummy input frame (all zeros)
    EbBufferHeaderType input_buf{};
    input_buf.size = sizeof(EbBufferHeaderType);

    EbSvtIOFormat input_pic{};

    const uint32_t luma_size = p.width * p.height;
    const uint32_t chroma_w = (p.width + 1) / 2;
    const uint32_t chroma_h = (p.height + 1) / 2;
    const uint32_t chroma_size = chroma_w * chroma_h;

    std::vector<uint8_t> luma(luma_size, 128);
    std::vector<uint8_t> cb(chroma_size, 128);
    std::vector<uint8_t> cr(chroma_size, 128);

    input_pic.luma = luma.data();
    input_pic.cb = cb.data();
    input_pic.cr = cr.data();
    input_pic.y_stride = p.width;
    input_pic.cb_stride = chroma_w;
    input_pic.cr_stride = chroma_w;

    input_buf.p_buffer = reinterpret_cast<uint8_t *>(&input_pic);
    input_buf.n_filled_len = luma_size + 2 * chroma_size;
    input_buf.pts = 0;
    input_buf.pic_type = EB_AV1_INVALID_PICTURE;

    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_send_picture(ctxt.enc_handle, &input_buf));

    // send EOS
    EbBufferHeaderType eos_buf{};
    eos_buf.size = sizeof(EbBufferHeaderType);
    eos_buf.flags = EB_BUFFERFLAG_EOS;
    ASSERT_EQ(EB_ErrorNone,
              svt_av1_enc_send_picture(ctxt.enc_handle, &eos_buf));

    // get output — drain until EOS
    bool got_eos = false;
    bool got_data = false;
    while (!got_eos) {
        EbBufferHeaderType *out = nullptr;
        EbErrorType ret =
            svt_av1_enc_get_packet(ctxt.enc_handle, &out, /*pic_send_done=*/1);
        if (ret == EB_ErrorNone && out != nullptr) {
            if (out->n_filled_len > 0)
                got_data = true;
            if (out->flags & EB_BUFFERFLAG_EOS)
                got_eos = true;
            svt_av1_enc_release_out_buffer(&out);
        } else {
            break;
        }
    }
    EXPECT_TRUE(got_eos) << "Did not receive EOS for " << p.width << "x"
                         << p.height;
    EXPECT_TRUE(got_data) << "No encoded data for " << p.width << "x"
                          << p.height;

    // cleanup
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit(ctxt.enc_handle));
    ASSERT_EQ(EB_ErrorNone, svt_av1_enc_deinit_handle(ctxt.enc_handle));
}

INSTANTIATE_TEST_SUITE_P(
    SmallResolutions, SmallResEncodeTest,
    ::testing::Values(SmallResParam{1, 1},    // minimum possible
                      SmallResParam{2, 2},    // sub-MI size
                      SmallResParam{3, 3},    // odd sub-MI size
                      SmallResParam{4, 4},    // 1 MI unit
                      SmallResParam{8, 8},    // 1 MIN_BLOCK_SIZE
                      SmallResParam{64, 1},   // single SB width, minimal height
                      SmallResParam{65, 1},   // multi-SB width, minimal height
                      SmallResParam{65, 4},   // multi-SB width, small height
                      SmallResParam{128, 1},  // 2 SB width, minimal height
                      SmallResParam{360, 1},  // wide strip, minimal height
                      SmallResParam{1, 64},   // minimal width, single SB height
                      SmallResParam{1, 65},   // minimal width, multi-SB height
                      SmallResParam{1, 360}   // tall strip, minimal width
                      ));
