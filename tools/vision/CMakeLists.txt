# tools/vision - strata-vision, the image encoder for Strata's multimodal path (optional; setup.py builds it when
# the vision encoder is chosen).  Uses llama.cpp's mtmd library at the same pinned commit as the engine's ggml.
#
#   NVIDIA: cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp
#                 -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
#   AMD:    the same with -DSTRATA_VISION_HIP=ON instead of -DSTRATA_VISION_CUDA=ON.  ROCm is taken from
#                 -DROCM_ROOT=<dir>, then $ROCM_PATH / $HIP_PATH, then /opt/rocm; the device code is built for the
#                 card in the PC unless -DGPU_TARGETS=gfx1100 (or a ';' list) names one.  On Windows the C++ compiler
#                 has to be ROCm's own clang (see below): call vcvars64.bat first and pass the three -DCMAKE_*_COMPILER
#                 lines from tools/hip/build_windows.bat.  Windows builds the ROCm libraries shared, so strata-vision.exe
#                 needs the .dll files this build leaves next to it (bin/) plus ROCm's own bin/ on the PATH - what
#                 setup.py already does through the config's lib_dirs (docs/AMD_HIP.md).
#   CPU:    neither option (the encoder runs on the CPU; --gpu is then refused by the program itself).
#
#   cmake --build build-vision --target strata-vision
#
# Portable (AVX2 baseline) is the default, so an encoder built without extra flags runs on any AVX2 PC: the 0.1.32
# release's encoder was built native on an AVX-512 machine and stopped with an illegal instruction elsewhere
# (#411 #412 #419).  setup.py, which builds for the PC it runs on, passes -DSTRATA_PORTABLE=OFF.  The option is
# kept in CMakeCache.txt, so a build folder configured before this default still needs the flag (or a fresh folder).
cmake_minimum_required(VERSION 3.21)
project(strata_vision C CXX)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp" CACHE PATH "llama.cpp checkout")
option(STRATA_VISION_CUDA "run the vision encoder on an NVIDIA GPU (builds ggml-cuda)" OFF)
option(STRATA_VISION_HIP "run the vision encoder on an AMD GPU (builds ggml-hip)" OFF)
option(STRATA_VISION_VULKAN "run the vision encoder on the GPU through Vulkan (builds ggml-vulkan; AMD, any vendor)" OFF)
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" ON)
set(STRATA_VISION_GPU_COUNT 0)
foreach(_b STRATA_VISION_CUDA STRATA_VISION_HIP STRATA_VISION_VULKAN)
  if(${_b})
    math(EXPR STRATA_VISION_GPU_COUNT "${STRATA_VISION_GPU_COUNT} + 1")
  endif()
endforeach()
if(STRATA_VISION_GPU_COUNT GREATER 1)
  message(FATAL_ERROR "more than one GPU backend is on (STRATA_VISION_CUDA / _HIP / _VULKAN): one GPU backend per "
                      "build (as in the engine's CMakeLists.txt, docs/AMD_HIP.md)")
endif()
if(STRATA_PORTABLE AND MSVC AND NOT STRATA_VISION_HIP)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# Ninja + MSVC leaves CMAKE_SYSTEM_PROCESSOR empty (the compiler ABI check is skipped for cl.exe), and ggml then
# prints "Unknown CPU architecture. Falling back to generic implementations." and builds the quant kernels without
# AVX2 (measured on an AMD Ryzen with MSVC 14.44 and CMake 4.4.3: the same picture at 1,024 image tokens took over
# 300 s with the generic kernels and 13 s with the AVX2 ones).  This PC's own architecture is what a build here is
# for, so name it when the generator did not.
if(WIN32 AND NOT CMAKE_SYSTEM_PROCESSOR)
  set(CMAKE_SYSTEM_PROCESSOR "AMD64")
  message(STATUS "strata-vision: CMAKE_SYSTEM_PROCESSOR was empty, set to AMD64 (Windows x64)")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
if(STRATA_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)        # no vcomp140.dll / libgomp needed on the user's PC
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
  # nothing above AVX2 either, even when a -D asks for it (a 9900K or a Zen 3 has neither AVX-512 nor AVX-VNNI)
  foreach(_f GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 GGML_AVX_VNNI
             GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: portable build (AVX2 baseline, runs on other PCs)")
else()
  # this PC's instruction set; a build folder that was portable before gets a fresh native build's settings back
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  set(GGML_OPENMP ON CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} OFF CACHE BOOL "" FORCE)
  endforeach()
  message(STATUS "strata-vision: native build for this CPU (STRATA_PORTABLE=OFF) - not for other PCs")
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${STRATA_VISION_CUDA} CACHE BOOL "" FORCE)
set(GGML_HIP ${STRATA_VISION_HIP} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${STRATA_VISION_VULKAN} CACHE BOOL "" FORCE)

# ---- the AMD path: ggml-cuda's sources compiled by ROCm's clang for the card in this PC -------------------------
#
# There is no second image encoder here: the encoder is ggml, and ggml-cuda is the same source tree compiled for
# HIP (llama.cpp's ggml-hip target).  What the port has to get right is the compiler and the architecture, which
# is what the block below pins.  It follows the engine's own Windows HIP build (tools/hip/build_windows.bat):
# CMake will not mix cl.exe with a HIP device compiler, and on Windows llama.cpp hands the .cu sources to the C++
# compiler with hip::device's -x hip, so cl.exe cannot read them - ROCm's clang compiles host and device.
if(STRATA_VISION_HIP)
  if(NOT ROCM_ROOT)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
      set(ROCM_ROOT "$ENV{ROCM_PATH}")
    elseif(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
      set(ROCM_ROOT "$ENV{HIP_PATH}")
    elseif(EXISTS /opt/rocm)
      set(ROCM_ROOT /opt/rocm)
    endif()
  endif()
  if(NOT ROCM_ROOT OR NOT IS_DIRECTORY "${ROCM_ROOT}")
    message(FATAL_ERROR
      "STRATA_VISION_HIP needs a ROCm install: pass -DROCM_ROOT=<dir> (or set ROCM_PATH / HIP_PATH)."
      " On Windows that is the tree the engine uses: <repo>\.rocm-win\Lib\site-packages\_rocm_sdk_devel, "
      "or a ROCm install of your own.")
  endif()
  file(TO_CMAKE_PATH "${ROCM_ROOT}" ROCM_ROOT)
  set(ENV{ROCM_PATH} "${ROCM_ROOT}")             # ggml-hip's own discovery reads this
  set(ENV{HIP_PATH} "${ROCM_ROOT}")
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib" "${ROCM_ROOT}/lib/cmake"
                                "${ROCM_ROOT}/lib64/cmake")

  # The card to build for.  Strata builds for the PC it runs on, so the card is read from that PC (the same
  # amdgpu-arch the engine's build uses); -DGPU_TARGETS / -DCMAKE_HIP_ARCHITECTURES overrides it.
  if(NOT GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
    find_program(STRATA_AMDGPU_ARCH NAMES amdgpu-arch amdgpu-arch.exe
                 PATHS "${ROCM_ROOT}/bin" "${ROCM_ROOT}/lib/llvm/bin" NO_DEFAULT_PATH)
    if(NOT STRATA_AMDGPU_ARCH)
      message(FATAL_ERROR
        "STRATA_VISION_HIP: no amdgpu-arch under ${ROCM_ROOT} to find this PC's GPU: pass "
        "-DGPU_TARGETS=gfx1100 (or the architecture of your card).")
    endif()
    execute_process(COMMAND "${STRATA_AMDGPU_ARCH}" OUTPUT_VARIABLE _strata_vision_archs
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "\n" ";" _strata_vision_archs "${_strata_vision_archs}")
    set(_strata_vision_keep "")                 # ROCm's enumerator lists gfx000, the generic target, first
    foreach(_a IN LISTS _strata_vision_archs)
      if(NOT _a STREQUAL "" AND NOT _a MATCHES "^gfx000")
        list(APPEND _strata_vision_keep "${_a}")
      endif()
    endforeach()
    set(_strata_vision_archs "${_strata_vision_keep}")
    list(REMOVE_DUPLICATES _strata_vision_archs)
    if(NOT _strata_vision_archs)
      message(FATAL_ERROR
        "STRATA_VISION_HIP: amdgpu-arch found no AMD GPU on this PC. The image encoder needs one; if this build "
        "is for another PC, pass -DGPU_TARGETS=gfx1100 (its card's architecture).")
    endif()
    set(GPU_TARGETS "${_strata_vision_archs}")
  endif()
  if(NOT GPU_TARGETS)
    set(GPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}")
  endif()
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")   # the Linux path: CMake's own HIP language
  endif()
  set(GPU_TARGETS "${GPU_TARGETS}" CACHE STRING "AMD GPU targets to build the encoder for" FORCE)
  string(REPLACE ";" "," _strata_vision_arch_str "${GPU_TARGETS}")
  set(ENV{AMDGPU_TARGETS} "${GPU_TARGETS}")
  message(STATUS "strata-vision: HIP for ${_strata_vision_arch_str} from ${ROCM_ROOT}")

  # The HIP runtime and hipBLAS are shared libraries, and llama.cpp's ggml-hip refuses a fully static link (its
  # own CMakeLists says so), so this build links llama/ggml shared.  What that means for the installed program: the
  # libraries are written next to strata-vision.exe (the POST_BUILD copy below), and both the build script's install
  # mode and setuphip.py copy the pair into engine-hip/.
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(GGML_STATIC OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)         # no libgomp / vcomp next to the ROCm DLLs to pick up

  if(WIN32)
    # llama.cpp on Windows has no CMake HIP language: ggml-hip compiles ggml-cuda's .cu as C++ with hip::device's
    # -x hip, so the C++ compiler has to be ROCm's clang (tools/hip/build_windows.bat does the same for the engine).
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(FATAL_ERROR
        "strata-vision with STRATA_VISION_HIP on Windows must be compiled by ROCm's clang, not by "
        "${CMAKE_CXX_COMPILER_ID}: call vcvars64.bat first, then configure with "
        "\"-DCMAKE_C_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang.exe\" and "
        "\"-DCMAKE_CXX_COMPILER=${ROCM_ROOT}/lib/llvm/bin/clang.exe\" "
        "(tools/hip/build_windows.bat does the same for the engine; "
        "tools/vision/build_windows_hip.bat does both in one go).")
    endif()
    # The ROCm clang for the MSVC ABI needs the MSVC headers and linker from vcvars; if it was configured without
    # them, clang fails on a missing <stdio.h>, which reads like a broken ROCm.  Check for the one header only the
    # Windows SDK gives it, and say what to do instead.
    include(CheckIncludeFileCXX)
    check_include_file_cxx("corecrt_io.h" _strata_vision_vcvars)
    if(NOT _strata_vision_vcvars)
      message(FATAL_ERROR
        "strata-vision with STRATA_VISION_HIP: the compiler cannot find the Windows SDK / MSVC headers. Run "
        "vcvars64.bat from Visual Studio's Build Tools before configuring (tools/vision/build_windows_hip.bat does "
        "that and configures the build).")
    endif()

    # A newer MSVC toolset's <cmath> and ROCm's HIP headers do not agree: it declares isgreater and friends as
    # host+device builtins, and the __device__ overloads in ROCm's __clang_hip_cmath.h then cannot be declared -
    # twenty errors per ggml-cuda file, none of them about ggml (measured with MSVC 14.51.36231 of Visual Studio 18
    # and ROCm 10.0.0rc4; MSVC 14.44.35207 of Build Tools 2022 compiles the same line).  A hundred and forty
    # ggml-cuda files deep into the build that is a very confusing failure, so it is probed here in one line.
    list(GET GPU_TARGETS 0 _strata_vision_arch0)
    file(WRITE "${CMAKE_BINARY_DIR}/strata-hip-cmath-probe.cpp"
         "#include <cmath>\nint main() { return (int) isgreater(1.0, 2.0); }\n")
    try_compile(_strata_vision_hip_cmath "${CMAKE_BINARY_DIR}/strata-hip-cmath-probe"
      "${CMAKE_BINARY_DIR}/strata-hip-cmath-probe.cpp"
      COMPILE_DEFINITIONS "-x;hip;--offload-arch=${_strata_vision_arch0};-isystem;${ROCM_ROOT}/include")
    if(NOT _strata_vision_hip_cmath)
      message(FATAL_ERROR
        "strata-vision with STRATA_VISION_HIP: this compiler cannot include <cmath> as HIP code, so ggml-cuda "
        "would not compile either (the paragraph above: MSVC 14.51's <cmath> against ROCm's HIP headers; 14.44 "
        "compiles that line). Configure inside the vcvars64.bat of an older Visual Studio - Build Tools 2022 work "
        "(MSVC 14.4x) - or pass -vcvars_ver=14.44 to it; tools/vision/build_windows_hip.bat tries each installed "
        "toolset that way and builds with the first one that passes.")
    endif()
  endif()
endif()

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(strata-vision strata_vision.cpp)
target_link_libraries(strata-vision PRIVATE mtmd llama)
target_compile_features(strata-vision PRIVATE cxx_std_17)
set_target_properties(strata-vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# The AMD build is the one that links llama/ggml shared, so the folder holding strata-vision.exe has to hold the
# libraries it loads too: they are copied where the executable is written.  The CPU and CUDA builds link everything
# static and need nothing beside the program.
if(STRATA_VISION_HIP)
  set(_strata_vision_dlls "")
  foreach(_t IN ITEMS llama mtmd ggml ggml-base ggml-cpu ggml-hip)
    if(TARGET ${_t})
      list(APPEND _strata_vision_dlls $<TARGET_FILE:${_t}>)
    endif()
  endforeach()
  # The HIP runtime next to the program has to be the one the code was compiled for.  Windows looks in the
  # executable's own folder, then in C:\Windows\System32, and only then on the PATH - so the AMD driver's copy of
  # amdhip64_<so>.dll wins over any lib_dirs the server adds, and a runtime that does not match the compiler
  # faults on the first host-to-device copy (measured on gfx1201: access violation, 0xC0000005, inside
  # ggml_backend_cuda_buffer_set_tensor while it loaded the vision weights; the same program encodes the same
  # picture with the runtime of the ROCm it was built with).  The engine ships its own HIP runtime for the same
  # reason (#468 #461, docs/AMD_HIP.md).
  if(WIN32)
    file(GLOB _strata_vision_hiprt CONFIGURE_DEPENDS "${ROCM_ROOT}/bin/amdhip64_*.dll")
    if(NOT _strata_vision_hiprt)
      message(WARNING "strata-vision: no amdhip64_*.dll in ${ROCM_ROOT}/bin - the program will use whatever HIP "
                      "runtime Windows finds first, which is usually the driver's and may not match this compiler.")
    endif()
    list(APPEND _strata_vision_dlls ${_strata_vision_hiprt})
  endif()
  add_custom_command(TARGET strata-vision POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_strata_vision_dlls} $<TARGET_FILE_DIR:strata-vision>
    COMMENT "copying the llama/ggml libraries and this ROCm's HIP runtime next to strata-vision")
endif()
