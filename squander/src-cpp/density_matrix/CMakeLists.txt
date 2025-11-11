# ===================================================================
# SQUANDER Density Matrix Module - Modern CMake
# ===================================================================

message(STATUS "")
message(STATUS "=== Configuring Density Matrix Module ===")

# ===================================================================
# Find pybind11 (required)
# ===================================================================

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, trying Python import...")
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(pybind11_DIR)
        message(STATUS "Found pybind11 via Python at: ${pybind11_DIR}")
        find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
    endif()
endif()

if(NOT pybind11_FOUND)
    message(WARNING "")
    message(WARNING "pybind11 not found - density matrix module will be skipped")
    message(WARNING "Install with: pip install pybind11")
    message(WARNING "")
    return()
endif()

message(STATUS "pybind11 version: ${pybind11_VERSION}")

# ===================================================================
# Source Files
# ===================================================================

set(DENSITY_MATRIX_SOURCES
    density_matrix.cpp
    density_circuit.cpp
    noise_channel.cpp
)

set(DENSITY_MATRIX_HEADERS
    include/density_matrix.h
    include/density_circuit.h
    include/noise_channel.h
)

# ===================================================================
# C++ Library Target (STATIC for internal use)
# ===================================================================

add_library(density_matrix_core STATIC
    ${DENSITY_MATRIX_SOURCES}
    ${DENSITY_MATRIX_HEADERS}
)

# Modern target-based include directories
target_include_directories(density_matrix_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include/squander/density_matrix>
)

# Link against INTERFACE library and main SQUANDER library
target_link_libraries(density_matrix_core
    PUBLIC
        squander_common    # INTERFACE library with common deps
    PRIVATE
        qgd                # Main SQUANDER library (for matrix_base, Gate, etc.)
)

# Modern compiler requirements
target_compile_features(density_matrix_core PUBLIC cxx_std_17)

# Compiler warnings and PIC (platform-specific using generator expressions)
target_compile_options(density_matrix_core PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
        -Wall -Wextra -Wpedantic -fPIC
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-g -O0>
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Debug>:/Od /Zi>
    >
)

# Set position independent code property
set_target_properties(density_matrix_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ===================================================================
# Python Module (pybind11)
# ===================================================================

pybind11_add_module(_density_matrix_cpp MODULE
    ../../density_matrix/bindings.cpp
)

# Link against C++ library
target_link_libraries(_density_matrix_cpp PRIVATE
    density_matrix_core
    qgd
)

# Set output directory to squander/density_matrix/
set_target_properties(_density_matrix_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/squander/density_matrix
    OUTPUT_NAME "_density_matrix_cpp"
)

# ===================================================================
# Tests (C++)
# ===================================================================

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(BUILD_TESTING OR DEFINED ENV{QGD_CTEST})
        enable_testing()
        
        add_executable(test_density_matrix_cpp
            tests/test_basic.cpp
        )
        
        target_link_libraries(test_density_matrix_cpp PRIVATE
            density_matrix_core
        )
        
        target_include_directories(test_density_matrix_cpp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        
        add_test(NAME DensityMatrixCppTests COMMAND test_density_matrix_cpp)
    endif()
endif()

# ===================================================================
# Installation
# ===================================================================

# Install C++ library (no export - internal use only)
install(TARGETS density_matrix_core
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        INCLUDES DESTINATION include/squander/density_matrix
)

# Install headers
install(FILES ${DENSITY_MATRIX_HEADERS}
        DESTINATION include/squander/density_matrix
)

# Install Python module
install(TARGETS _density_matrix_cpp
        LIBRARY DESTINATION squander/density_matrix
)

message(STATUS "=== Density Matrix Module Configured ===")
message(STATUS "")

