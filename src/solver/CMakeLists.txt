#
# CMakeLists.txt - CMake configuration file for s
# Created: 11-13-2023
# Updated: 11-13-2023
#
# Author: Caleb Buahin
#         US EPA ORD/CESER
#

find_package(OpenMP)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE)
    set(EPANETMSX_REACT_TRANSFER_STATIC_CUDA ON)
else()
    set(EPANETMSX_REACT_TRANSFER_STATIC_CUDA OFF)
endif()


set(EPANETMSX_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/epanetmsx.h
)

file(GLOB
    EPANETMSX_SOURCES
    CONFIGURE_DEPENDS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.c *.h *.dat
)
# The glob sees the non-CUDA implementation too.  It must never be linked
# with the CUDA object because both export the same ABI.
list(REMOVE_ITEM EPANETMSX_SOURCES
    msxresident_core_cuda_stub.c
    ${CMAKE_CURRENT_SOURCE_DIR}/msxresident_core_cuda_stub.c)

if(EPANETMSX_REACT_TRANSFER_STATIC_CUDA)
    set(REACT_TRANSFER_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/msxreact_transfer_cuda.obj)
    set(REACT_TRANSFER_CUDA_INCLUDE_FLAGS "")
    foreach(REACT_TRANSFER_CUDA_INCLUDE_DIR IN LISTS CUDAToolkit_INCLUDE_DIRS)
        list(APPEND REACT_TRANSFER_CUDA_INCLUDE_FLAGS
            "-I${REACT_TRANSFER_CUDA_INCLUDE_DIR}")
    endforeach()
    set(REACT_TRANSFER_CUDA_OPENMP_FLAG "")
    if(OpenMP_C_FOUND AND MSVC)
        set(REACT_TRANSFER_CUDA_OPENMP_FLAG "-Xcompiler=/openmp")
    elseif(OpenMP_C_FOUND)
        set(REACT_TRANSFER_CUDA_OPENMP_FLAG "-Xcompiler=-fopenmp")
    endif()
    add_custom_command(
        OUTPUT ${REACT_TRANSFER_CUDA_OBJ}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            -c ${CMAKE_CURRENT_SOURCE_DIR}/msxreact_transfer_cuda.cu
            -allow-unsupported-compiler
            -o ${REACT_TRANSFER_CUDA_OBJ}
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/..
            -I${PROJECT_SOURCE_DIR}/EPANET2.2/SRC_engines/include
            ${REACT_TRANSFER_CUDA_INCLUDE_FLAGS}
            -DEPANETMSX_CUDA_ENABLED
            -D_CRT_SECURE_NO_WARNINGS
            -Xcompiler=/MD
            ${REACT_TRANSFER_CUDA_OPENMP_FLAG}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/msxreact_transfer_cuda.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/msxreact_transfer_cuda.h
                ${CMAKE_CURRENT_SOURCE_DIR}/msxsegment_storage.h
                ${CMAKE_CURRENT_SOURCE_DIR}/msxtypes.h
        VERBATIM
    )
    list(APPEND EPANETMSX_SOURCES ${REACT_TRANSFER_CUDA_OBJ})
    set(RESIDENT_CORE_CUDA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/msxresident_core_cuda.obj)
    add_custom_command(OUTPUT ${RESIDENT_CORE_CUDA_OBJ}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR}/msxresident_core_cuda.cu -allow-unsupported-compiler -o ${RESIDENT_CORE_CUDA_OBJ} -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_CURRENT_SOURCE_DIR}/.. -I${PROJECT_SOURCE_DIR}/EPANET2.2/SRC_engines/include ${REACT_TRANSFER_CUDA_INCLUDE_FLAGS} -DEPANETMSX_CUDA_ENABLED -D_CRT_SECURE_NO_WARNINGS -Xcompiler=/MD
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/msxresident_core_cuda.cu ${CMAKE_CURRENT_SOURCE_DIR}/msxresident_core_cuda.h ${CMAKE_CURRENT_SOURCE_DIR}/msxresident_core.h ${CMAKE_CURRENT_SOURCE_DIR}/msxtypes.h
        VERBATIM)
    list(APPEND EPANETMSX_SOURCES ${RESIDENT_CORE_CUDA_OBJ})
else()
    list(APPEND EPANETMSX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/msxresident_core_cuda_stub.c)
endif()

if(BUILD_DEF)
    # Builds library with def file interface for backward compatibility
    set_source_files_properties(${PROJECT_SOURCE_DIR}/include/epanetmsx.def
        PROPERTIES_HEADER_FILE_ONLY TRUE
    )

    add_library(epanetmsx
            ${EPANETMSX_SOURCES}
            ${PROJECT_SOURCE_DIR}/include/epanetmsx.def
    )

else()
    # Performs standard library build
    add_library(epanetmsx
            ${EPANETMSX_SOURCES}
    )

endif()

if(MSVC)
    set_property(TARGET epanetmsx PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
    )
endif()

target_link_libraries(epanetmsx
    PUBLIC
        epanet2
)

# if openmp is availabe set USE_OPENMP define for target
if(OpenMP_C_FOUND)
    target_compile_definitions(epanetmsx
        PUBLIC
            USE_OPENMP
    )
endif()

if(CUDAToolkit_FOUND)
    target_compile_definitions(epanetmsx
        PRIVATE
            EPANETMSX_CUDA_ENABLED
    )
    target_include_directories(epanetmsx
        PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
    )
endif()

if(EPANETMSX_REACT_TRANSFER_STATIC_CUDA)
    target_compile_definitions(epanetmsx
        PRIVATE
            EPANETMSX_REACT_TRANSFER_STATIC_CUDA
    )
endif()

# Sets MSVC compiler flags
target_compile_options(epanetmsx
    PUBLIC
        "$<$<C_COMPILER_ID:MSVC>:"
            "$<$<CONFIG:Release>:/GL>"
            "$<$<CONFIG:Release>:/fp:fast>"
            "$<$<CONFIG:Release>:/Zi>"
        ">"
        $<$<C_COMPILER_ID:AppleClang>:
            $<$<STREQUAL:"${CMAKE_GENERATOR}","Ninja">:-O3>
        >
)

target_link_options(epanetmsx
    PUBLIC
        "$<$<C_COMPILER_ID:MSVC>:"
            "$<$<CONFIG:Release>:/LTCG:incremental>"
            "$<$<BOOL:${CUDAToolkit_FOUND}>:/NODEFAULTLIB:LIBCMT>"
        ">"
)

include(GenerateExportHeader)
generate_export_header(epanetmsx 
    BASE_NAME epanetmsx
    EXPORT_MACRO_NAME MSXDLLEXPORT
    EXPORT_FILE_NAME epanetmsx_export.h
    STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC
)

file(COPY ${CMAKE_CURRENT_BINARY_DIR}/epanetmsx_export.h
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include
)



target_link_libraries(epanetmsx
    PUBLIC
        $<$<NOT:$<BOOL:$<C_COMPILER_ID:MSVC>>>:m>
        $<$<BOOL:${OpenMP_C_FOUND}>:OpenMP::OpenMP_C>
        $<$<BOOL:${OpenMP_AVAILABLE}>:omp>
        $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::cuda_driver>
        $<$<BOOL:${CUDAToolkit_FOUND}>:CUDA::nvrtc>
        $<$<BOOL:${EPANETMSX_REACT_TRANSFER_STATIC_CUDA}>:CUDA::cudart>
)



target_include_directories(epanetmsx
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/EPANET2.2/SRC_engines/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${INCLUDE_DIST}>

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)


install(TARGETS epanetmsx EXPORT epanetmsxTargets
    RUNTIME    DESTINATION "${TOOL_DIST}"
    LIBRARY    DESTINATION "${LIBRARY_DIST}"
    ARCHIVE    DESTINATION "${LIBRARY_DIST}"
    FRAMEWORK  DESTINATION "${TOOL_DIST}"
)

# Create target import scripts so other cmake projects can use swmm libraries
install(
    EXPORT
        epanetmsxTargets
    DESTINATION
        "${CONFIG_DIST}"
    FILE
        epanetmsx-config.cmake
)

install(
    FILES 
        ${EPANETMSX_PUBLIC_HEADERS} 
    DESTINATION 
        "${INCLUDE_DIST}"
)
