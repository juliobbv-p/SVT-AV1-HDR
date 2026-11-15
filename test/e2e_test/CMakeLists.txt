#
# Copyright(c) 2019 Netflix, Inc
#
# This source code is subject to the terms of the BSD 2 Clause License and the
# Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License was
# not distributed with this source code in the LICENSE file, you can obtain it
# at https://www.aomedia.org/license/software-license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.
#

# APP Directory CMakeLists.txt

set(SVT_AV1_E2E_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
include(test_data_util.cmake)

# testdata download targets.
set(TEST_DATA_LIST_FILE "${SVT_AV1_E2E_ROOT}/test_vector_list.txt")
make_test_data_lists("${TEST_DATA_LIST_FILE}" test_files test_file_checksums)
list(LENGTH test_files num_test_files)
list(LENGTH test_file_checksums num_test_file_checksums)

# Create a custom build target for running each test data download target.
add_custom_target(TestVectors)

math(EXPR max_file_index "${num_test_files} - 1")
foreach(test_index RANGE ${max_file_index})
    list(GET test_files ${test_index} test_file)
    list(GET test_file_checksums ${test_index} test_file_checksum)
    add_custom_target(
        testdata_${test_file}
        COMMAND
            ${CMAKE_COMMAND} -DSVT_AV1_E2E_ROOT="${SVT_AV1_E2E_ROOT}"
            -DSVT_AV1_TEST_FILE="${test_file}"
            -DSVT_AV1_TEST_CHECKSUM="${test_file_checksum}" -P
            "${SVT_AV1_E2E_ROOT}/test_data_download_worker.cmake")
    add_dependencies(TestVectors testdata_${test_file})
endforeach()

if(SVT_AV1_TEST_VECTORS_ONLY)
    return()
endif()

include(libaom_dependency.cmake)

set(all_files
    ../../Source/App/app_input_y4m.c
    ../../Source/App/app_film_grain.c
    ../../Source/App/app_context.c
    ../../Source/App/app_output_ivf.c
    CompareTools.h
    ConfigEncoder.cc
    ConfigEncoder.h
    DummyVideoSource.h
    E2eTestVectors.h
    FrameQueue.cc
    FrameQueue.h
    ParseUtil.cc
    ParseUtil.h
    PerformanceCollect.cc
    PerformanceCollect.h
    RefDecoder.cc
    RefDecoder.h
    SvtAv1E2EFramework.cc
    SvtAv1E2EFramework.h
    SvtAv1E2EParamsTest.cc
    SvtAv1E2ETest.cc
    SvtAv1FrameIdBitstreamTest.cc
    SvtAv1FilmGrainBitstreamTest.cc
    SvtAv1RefMgmtBitstreamTest.cc
    SvtAv1RefSlotBitstreamTest.cc
    VideoFrame.h
    VideoSource.cc
    VideoSource.h
    Y4mVideoSource.cc
    Y4mVideoSource.h
    YuvVideoSource.cc
    YuvVideoSource.h
    dec_bitstream.h
    dec_bitstream.c)

set(lib_list SvtAv1Enc gtest_all aom)

if(TARGET safestringlib)
    list(APPEND lib_list $<TARGET_OBJECTS:safestringlib>)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
endif()

if(MSVC) # Use custom function for adding msvc's flags like bigobj
    cxx_executable_with_flags(SvtAv1E2ETests "${cxx_default}" "${lib_list}"
                              ${all_files})
else()
    # App Source Files
    add_executable(SvtAv1E2ETests ${all_files})
    # Link the Encoder App
    target_link_libraries(SvtAv1E2ETests ${lib_list} pthread m)
endif()

target_include_directories(
    SvtAv1E2ETests
    PRIVATE ${PROJECT_SOURCE_DIR}/Source/API
            ${PROJECT_SOURCE_DIR}/test
            ${PROJECT_SOURCE_DIR}/third_party/aom/inc
            ${PROJECT_SOURCE_DIR}/third_party/googletest/include
            ${PROJECT_SOURCE_DIR}/third_party/googletest/src)

add_test(NAME SvtAv1E2ETests COMMAND SvtAv1E2ETests)

if(BUILD_APPS)
    add_dependencies(SvtAv1E2ETests SvtAv1EncApp)
    add_test(NAME FilmGrainCliTest COMMAND ${CMAKE_COMMAND}
        "-DENCODER=$<TARGET_FILE:SvtAv1EncApp>"
        "-DTESTS=$<TARGET_FILE:SvtAv1E2ETests>"
        "-DSCRATCH=${CMAKE_CURRENT_BINARY_DIR}/film-grain-cli"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/FilmGrainCliTest.cmake")
    set_tests_properties(FilmGrainCliTest PROPERTIES TIMEOUT 60)
endif()
