cmake_minimum_required(VERSION 3.20)
project(resident_phase2b LANGUAGES C)

find_package(CUDAToolkit REQUIRED)
set(RESIDENT_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/msxresident_core_cuda.obj)
set(RESIDENT_CUDA_INCLUDE_ARGS)
foreach(RESIDENT_CUDA_INCLUDE IN LISTS CUDAToolkit_INCLUDE_DIRS)
  list(APPEND RESIDENT_CUDA_INCLUDE_ARGS -I${RESIDENT_CUDA_INCLUDE})
endforeach()
# The installed VS generator lacks a CUDA platform toolset.  Keep NVCC under
# CMake/MSBuild custom-build ownership, matching the production solver build.
add_custom_command(OUTPUT ${RESIDENT_CUDA_OBJ}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/solver/msxresident_core_cuda.cu
    -allow-unsupported-compiler -o ${RESIDENT_CUDA_OBJ}
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../src/solver
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../src
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../EPANET2.2/SRC_engines/include
    ${RESIDENT_CUDA_INCLUDE_ARGS} -D_CRT_SECURE_NO_WARNINGS -Xcompiler=/MD
  DEPENDS ../../src/solver/msxresident_core_cuda.cu
          ../../src/solver/msxresident_core_cuda.h
          ../../src/solver/msxresident_core.h
  VERBATIM)
add_executable(resident_phase2b_harness resident_phase2b_harness.c ${RESIDENT_CUDA_OBJ})
target_include_directories(resident_phase2b_harness PRIVATE
  ../../src/solver ../../src ../../EPANET2.2/SRC_engines/include)
target_compile_definitions(resident_phase2b_harness PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(resident_phase2b_harness PRIVATE CUDA::cudart)
