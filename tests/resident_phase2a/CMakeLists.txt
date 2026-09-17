cmake_minimum_required(VERSION 3.16)
project(resident_phase2a C)

add_executable(resident_phase2a_harness resident_phase2a_harness.c
  ../../src/solver/msxresident_core.c
  ../../src/solver/msxresident_hash.c
  ../../src/solver/msxsegment_storage.c)
target_include_directories(resident_phase2a_harness PRIVATE
  ../../src/solver ../../src ../../EPANET2.2/SRC_engines/include)
target_compile_definitions(resident_phase2a_harness PRIVATE _CRT_SECURE_NO_WARNINGS)
