# Copyright(c) 2026 Alliance for Open Media. All rights reserved.
# Copyright (c) 2026 Halide Compression, LLC
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. See LICENSE-BSD2.md and
# PATENTS.md for details.

execute_process(COMMAND "${ENCODER}" --help OUTPUT_VARIABLE help RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot run encoder")
endif()
if(NOT help MATCHES "--fgs-table")
    return()
endif()
file(MAKE_DIRECTORY "${SCRATCH}")
string(ASCII 128 sample)
string(REPEAT "${sample}" 1290240 video)
file(WRITE "${SCRATCH}/input.yuv" "${video}")
file(WRITE "${SCRATCH}/grain.tbl" "filmgrn1
E 0 20000000 1 12345 1
 p 0 6 0 8 0 1 0 0 0 0 0 0
 sY 2 0 16 255 16
 sCb 0
 sCr 0
 cY
 cCb 0
 cCr 0
E 20000000 40000000 1 23456 1
 p 0 6 0 8 0 1 0 0 0 0 0 0
 sY 2 0 64 255 64
 sCb 0
 sCr 0
 cY
 cCb 0
 cCr 0
E 40000000 60000000 0 0 0
")
execute_process(COMMAND "${ENCODER}" -i "${SCRATCH}/input.yuv" -w 64 -h 64
    --fps-num 30 --fps-denom 1 -n 210 --preset 10 --lp 1 --progress 0
    --enable-overlays 1 --fgs-table "${SCRATCH}/grain.tbl" -b "${SCRATCH}/output.ivf"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Encoder failed: ${output}\n${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env "SVT_GRAIN_CLI_OUTPUT=${SCRATCH}/output.ivf"
    "${TESTS}" --gtest_filter=FilmGrainCliTest.TimedTable
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Grain output test failed: ${output}\n${error}")
endif()
file(REMOVE_RECURSE "${SCRATCH}")
