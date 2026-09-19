/*
 * Copyright(c) 2019 Intel Corporation
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at https://www.aomedia.org/license/software-license. If the
 * Alliance for Open Media Patent License 1.0 was not distributed with this
 * source code in the PATENTS file, you can obtain it at
 * https://www.aomedia.org/license/patent-license.
 */

#ifndef EbUnitTestUtility_h
#define EbUnitTestUtility_h

#include <stddef.h>
#include <stdint.h>
#include "definitions.h"
#include "svt_time.h"

#ifdef __cplusplus
#include "gtest/gtest.h"
#include <type_traits>

/***************************************
 * Compare Data
 ***************************************/
namespace {
template <typename T>
bool svt_buf_compare(const T *const buf1, const T *const buf2,
                     const size_t bufSize) {
    static_assert(std::is_integral<T>::value, "Integral type is expected.");
    bool result = true;
    for (size_t i = 0; i < bufSize; i++) {
        if (buf1[i] != buf2[i]) {
            ADD_FAILURE() << "buf1[" << i << "] = 0x" << std::hex
                          << static_cast<uint64_t>(buf1[i]) << "\tbuf2[" << i
                          << "] = 0x" << std::hex
                          << static_cast<uint64_t>(buf2[i]);
            result = false;
        }
    }
    return result;
}
}  // namespace
extern "C" {
#endif

#define Eb_UNIT_TEST_BUF_SIZE 0x04000000
#define Eb_UNIT_TEST_BUF_ALIGN_BYTE 256
#define Eb_UNIT_TEST_BUF_UNALIGN_BYTE 3
#define NELEMENTS(x) (int)(sizeof(x) / sizeof(x[0]))

extern void svt_buf_random_void(void *const buf, const size_t sizeBuf);
extern void svt_buf_random_u8(uint8_t *const buf, const size_t sizeBuf);
extern void svt_buf_random_u8_to_0_or_255(uint8_t *const buf,
                                          const size_t sizeBuf);
extern void svt_buf_random_u8_to_255(uint8_t *const buf, const size_t sizeBuf);
extern void svt_buf_random_s16(int16_t *const buf, const size_t sizeBuf);
extern void svt_buf_random_u16(uint16_t *const buf, const size_t sizeBuf);
extern void svt_buf_random_u16_to_0_or_bd(uint16_t *const buf,
                                          const size_t sizeBuf,
                                          const uint32_t bd);
extern void svt_buf_random_u16_to_bd(uint16_t *const buf, const size_t sizeBuf,
                                     const uint32_t bd);
extern void svt_buf_random_u16_with_bd(uint16_t *const buf,
                                       const size_t sizeBuf, const uint32_t bd);
extern void svt_buf_random_s32(int32_t *const buf, const size_t sizeBuf);
extern void svt_buf_random_s32_with_max(int32_t *const buf,
                                        const size_t sizeBuf,
                                        const int32_t max_abs);
extern void svt_buf_random_u32(uint32_t *const buf, const size_t sizeBuf);
extern void svt_buf_random_u32_with_max(uint32_t *const buf,
                                        const size_t sizeBuf,
                                        const uint32_t max);
extern void svt_buf_random_s64(int64_t *const buf, const size_t sizeBuf);

extern uint32_t svt_create_random_aligned_stride(const uint32_t width,
                                                 const uint32_t align);

#ifdef __cplusplus
}
#endif

#endif  // EbUnitTestUtility_h
