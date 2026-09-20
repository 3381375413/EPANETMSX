cmake_minimum_required(VERSION 3.20)
project(resident_core_span C)

add_executable(resident_core_span_harness
  resident_core_span_harness.c
  ../../src/solver/msxresident_core.c
  ../../src/solver/msxresident_hash.c
  ../../src/solver/msxsegment_storage.c)
target_include_directories(resident_core_span_harness PRIVATE
  ../../src/solver ../../src ../../EPANET2.2/SRC_engines/include)
target_compile_definitions(resident_core_span_harness PRIVATE
  _CRT_SECURE_NO_WARNINGS
  MSX_RESIDENT_CORE_SPAN_TEST)
