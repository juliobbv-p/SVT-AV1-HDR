#
# Copyright (c) 2026, Alliance for Open Media. All rights reserved
#
# This source code is subject to the terms of the BSD 2 Clause License and the
# Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
# not distributed with this source code in the LICENSE file, you can obtain it
# at www.aomedia.org/license/software. If the Alliance for Open Media Patent
# License 1.0 was not distributed with this source code in the PATENTS file, you
# can obtain it at www.aomedia.org/license/patent.
#

# ASM_ARM_CRC32 Directory CMakeLists.txt

# Include Encoder Subdirectories

check_both_flags_add(-march=armv8-a+crc)

add_library(ASM_ARM_CRC32 OBJECT)
target_sources(
  ASM_ARM_CRC32
  PUBLIC hash_arm_crc32.c)

target_include_directories(
  ASM_ARM_CRC32
  PRIVATE ${PROJECT_SOURCE_DIR}/Source/API/
  PRIVATE ${PROJECT_SOURCE_DIR}/Source/Lib/Codec/
  PRIVATE ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT/)
