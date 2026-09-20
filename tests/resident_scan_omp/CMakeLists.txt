cmake_minimum_required(VERSION 3.20)
project(resident_scan_omp C)

find_package(OpenMP REQUIRED)

add_executable(resident_scan_omp_harness
  resident_scan_omp_harness.c
  ../../src/solver/msxresident_core.c
  ../../src/solver/msxresident_hash.c
  ../../src/solver/msxsegment_storage.c)
target_include_directories(resident_scan_omp_harness PRIVATE
  ../../src/solver ../../src ../../EPANET2.2/SRC_engines/include)
target_compile_definitions(resident_scan_omp_harness PRIVATE
  _CRT_SECURE_NO_WARNINGS MSX_RESIDENT_SCAN_OMP_TEST)
target_link_libraries(resident_scan_omp_harness PRIVATE OpenMP::OpenMP_C)
