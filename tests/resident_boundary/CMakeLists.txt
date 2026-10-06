cmake_minimum_required(VERSION 3.16)
project(resident_boundary C)
add_executable(resident_boundary_harness resident_boundary_harness.c)
target_include_directories(resident_boundary_harness PRIVATE ../../src/solver ../../src ../../EPANET2.2/SRC_engines/include)
