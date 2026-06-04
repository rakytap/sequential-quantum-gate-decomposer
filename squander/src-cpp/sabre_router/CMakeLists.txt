# ===================================================================
# SQUANDER SABRE Router Module - C++ Routing Engine + pybind11 Bindings
# ===================================================================

message(STATUS "")
message(STATUS "=== Configuring SABRE Router Module ===")

# ===================================================================
# Find pybind11
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
    message(WARNING "pybind11 not found - SABRE router module will be skipped")
    message(WARNING "Install with: pip install pybind11")
    message(WARNING "")
    return()
endif()

message(STATUS "pybind11 version: ${pybind11_VERSION}")

# ===================================================================
# Source Files
# ===================================================================

set(SABRE_SOURCES
    sabre_router.cpp
)

set(SABRE_HEADERS
    include/sabre_router.hpp
)

# ===================================================================
# Static C++ library
# ===================================================================

add_library(sabre_router_core STATIC
    ${SABRE_SOURCES}
    ${SABRE_HEADERS}
)

target_include_directories(sabre_router_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# C++17 for this module only (does not affect global C++11)
target_compile_features(sabre_router_core PUBLIC cxx_std_17)

target_compile_options(sabre_router_core PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
        -Wall -Wextra -fPIC
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-g -O0>
    >
    $<$<CXX_COMPILER_ID:MSVC>:
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Debug>:/Od /Zi>
    >
)

set_target_properties(sabre_router_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ===================================================================
# pybind11 module
# ===================================================================

pybind11_add_module(_sabre_router MODULE
    ../../synthesis/bindings.cpp
)

target_link_libraries(_sabre_router PRIVATE
    sabre_router_core
)

set_target_properties(_sabre_router PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/squander/synthesis
    OUTPUT_NAME "_sabre_router"
)

# Set BUILD_RPATH to prioritize conda libraries
if(DEFINED ENV{CONDA_PREFIX})
    set_target_properties(_sabre_router PROPERTIES
        BUILD_RPATH "${CONDA_PREFIX}/lib"
        BUILD_RPATH_USE_ORIGIN TRUE
    )
endif()

# ===================================================================
# Installation
# ===================================================================

install(TARGETS _sabre_router
        LIBRARY DESTINATION squander/synthesis
        RUNTIME DESTINATION squander/synthesis
        COMPONENT python)

message(STATUS "=== SABRE Router Module Configured ===")
message(STATUS "")
