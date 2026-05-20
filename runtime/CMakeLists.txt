# Booki runtime — baseline build.
#
# Today this thin shim wraps ONNX Runtime so the CI workflow has something
# real to bench against. When #13 starts, the same booki_baseline.h surface
# gets reimplemented by a hand-rolled engine and the CI gates whether it
# matches ORT for correctness and beats it on per-token latency.
#
# Targets:
#   libbooki_runtime.{so,a}   -- the engine + ORT backend
#   booki_bench                -- micro-benchmark, emits JSON
#   booki_eval                 -- golden-output diff against a fixture

cmake_minimum_required(VERSION 3.20)
project(booki_runtime LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(BOOKI_ENABLE_NEON      "Enable ARM NEON kernels (no-op for the baseline)" OFF)
set(BOOKI_SANITIZERS "" CACHE STRING "Comma-separated list passed to -fsanitize= (e.g. address,undefined)")

if(BOOKI_SANITIZERS)
    add_compile_options(-fsanitize=${BOOKI_SANITIZERS} -fno-omit-frame-pointer -O1 -g)
    add_link_options(-fsanitize=${BOOKI_SANITIZERS})
endif()

# -----------------------------------------------------------------------------
# Fetch the appropriate prebuilt ONNX Runtime
# -----------------------------------------------------------------------------
set(ORT_VERSION "1.20.1")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_PKG "onnxruntime-linux-aarch64-${ORT_VERSION}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ORT_PKG "onnxruntime-linux-x64-${ORT_VERSION}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ORT_PKG "onnxruntime-osx-universal2-${ORT_VERSION}")
elseif(ANDROID)
    # On Android, the host app links sherpa-onnx's bundled ORT; this CMake
    # tree is host-side only. The android-arm64-build CI job stops at the
    # cross-compile-link step to verify __ANDROID__ defines work.
    message(STATUS "Android cross-compile: skipping ORT fetch (header-only target)")
    set(ORT_PKG "")
else()
    message(FATAL_ERROR "Unsupported host: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(ORT_PKG)
    include(FetchContent)
    FetchContent_Declare(onnxruntime
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_PKG}.tgz"
    )
    FetchContent_MakeAvailable(onnxruntime)

    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION    "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    )
endif()

# -----------------------------------------------------------------------------
# libbooki_runtime (ORT baseline; host builds only — Android uses sherpa-onnx)
# -----------------------------------------------------------------------------
if(ORT_PKG)
    add_library(booki_runtime SHARED booki_baseline.c)
    target_include_directories(booki_runtime PUBLIC .)
    target_link_libraries(booki_runtime PUBLIC onnxruntime)
    target_compile_options(booki_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter)
    add_custom_command(TARGET booki_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
            "$<TARGET_FILE_DIR:booki_runtime>/")
endif()

# Stub flag preserved from the original scaffolding — actual NEON dispatch
# now lives in booki_native (compile-time __ARM_NEON detection).
if(BOOKI_ENABLE_NEON AND TARGET booki_runtime)
    target_compile_definitions(booki_runtime PRIVATE BOOKI_HAS_NEON=1)
endif()

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
if(ORT_PKG)
    add_executable(booki_bench tools/booki_bench.c)
    target_link_libraries(booki_bench PRIVATE booki_runtime)

    add_executable(booki_eval tools/booki_eval.c)
    target_link_libraries(booki_eval PRIVATE booki_runtime m)
endif()

# -----------------------------------------------------------------------------
# Native (hand-rolled) runtime — see runtime/native/ and issue #13.
#
# Built as a separate library so the ORT baseline and the native engine can
# coexist while op coverage is incomplete. Both feed runtime-ci.yml's bench
# regression gate.
# -----------------------------------------------------------------------------
set(BOOKI_NATIVE_SOURCES
    native/booki_tensor.c
    native/booki_matmul.c
    native/booki_ops.c
    native/booki_conv1d.c
    native/booki_gguf.c)

# SME backend: enabled when cross-compiling for aarch64 with a recent clang
# (NDK r28+ ships clang 19). Detection here is "are we aarch64 AND did the
# user pass -DBOOKI_FORCE_SME=OFF to skip" — the actual feature gate is the
# compiler flag below; if it's not supported, the build fails loudly.
option(BOOKI_FORCE_SME "Force-enable SME backend on aarch64" ON)
set(BOOKI_HAS_SME OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND BOOKI_FORCE_SME)
    set(BOOKI_HAS_SME ON)
endif()

if(BOOKI_HAS_SME)
    list(APPEND BOOKI_NATIVE_SOURCES native/booki_sme.c)
    message(STATUS "Booki: SME backend ENABLED")
else()
    message(STATUS "Booki: SME backend disabled (compiler / arch lacks SME2)")
endif()

add_library(booki_native SHARED ${BOOKI_NATIVE_SOURCES})
target_include_directories(booki_native PUBLIC native)
target_link_libraries(booki_native PUBLIC m)
target_compile_options(booki_native PRIVATE -Wall -Wextra -Wno-unused-parameter -O3)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(BOOKI_HAS_SME)
        target_compile_options(booki_native PRIVATE -march=armv9.2-a+sme+sme2+fp16+dotprod)
        target_compile_definitions(booki_native PRIVATE BOOKI_HAS_SME_BUILD=1)
    else()
        target_compile_options(booki_native PRIVATE -march=armv8.2-a+fp16+dotprod)
    endif()
endif()

add_executable(booki_bench_native tools/booki_bench_native.c)
target_link_libraries(booki_bench_native PRIVATE booki_native)

add_executable(booki_bench_block tools/booki_bench_block.c)
target_link_libraries(booki_bench_block PRIVATE booki_native)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()
if(ORT_PKG)
    add_executable(test_smoke tests/test_smoke.c)
    target_link_libraries(test_smoke PRIVATE booki_runtime)
    add_test(NAME smoke COMMAND test_smoke)
endif()

add_executable(test_arena native/tests/test_arena.c)
target_link_libraries(test_arena PRIVATE booki_native)
add_test(NAME arena COMMAND test_arena)

add_executable(test_matmul native/tests/test_matmul.c)
target_link_libraries(test_matmul PRIVATE booki_native m)
add_test(NAME matmul COMMAND test_matmul)

add_executable(test_ops native/tests/test_ops.c)
target_link_libraries(test_ops PRIVATE booki_native m)
add_test(NAME ops COMMAND test_ops)

add_executable(test_conv1d native/tests/test_conv1d.c)
target_link_libraries(test_conv1d PRIVATE booki_native m)
add_test(NAME conv1d COMMAND test_conv1d)

add_executable(test_mha native/tests/test_mha.c)
target_link_libraries(test_mha PRIVATE booki_native m)
add_test(NAME mha COMMAND test_mha)

add_executable(test_gguf native/tests/test_gguf.c)
target_link_libraries(test_gguf PRIVATE booki_native m)
add_test(NAME gguf COMMAND test_gguf)
