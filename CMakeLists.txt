cmake_minimum_required(VERSION 3.10.2)

# CMAKE to create the shared library of the quantum gate decomposition project

# set the project name and version
project(CQGD VERSION 1.9.5)

# reuse compilation time linking for use runtime linking 
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Configure RPATH to prioritize conda libraries and avoid conflicts
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    set(CONDA_LIB_DIR "${CONDA_PREFIX}/lib")
    # Add conda lib directory to BUILD_RPATH to prioritize conda libraries at runtime
    set(CMAKE_BUILD_RPATH "${CONDA_LIB_DIR}" ${CMAKE_BUILD_RPATH})
    set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
    # Suppress RPATH warnings when conda libraries are prioritized
    # These warnings occur because system libraries might conflict with conda libraries,
    # but BUILD_RPATH ensures conda libraries are found first
    if(POLICY CMP0144)
        cmake_policy(SET CMP0144 NEW)
    endif()
    message(STATUS "Configured BUILD_RPATH to prioritize conda libraries: ${CONDA_LIB_DIR}")
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) #for lib not just dll when no exports

# include CMAKE modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(cmake/check_AVX.cmake)

# variables for compile and link options
set(CXX_FLAGS)
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(EXTRA_INCLUDES)
set(BLAS_DIR)
set(PYTHON_PLAT_LIB)

# Windows-specific settings
if(WIN32)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Increase object file path limit
    set(CMAKE_OBJECT_PATH_MAX 4096)
    # Set /bigobj flag directly in CMAKE_CXX_FLAGS to ensure it's applied globally
    # This is needed for large object files on Windows (prevents C1083 errors)
    # The /bigobj flag allows object files with more than 65535 sections
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" CACHE STRING "Flags used by the compiler during all build types" FORCE)
    # Also set for Release and Debug configurations to ensure coverage
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj" CACHE STRING "Flags used by the compiler during release builds" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
    message(STATUS "MSVC: Added /bigobj flag to CMAKE_CXX_FLAGS for Windows compilation")
  endif()
endif()


#enable test target
enable_testing()


if(SKBUILD)
    message(STATUS "The project is built using scikit-build")
endif()


# find out python packages
find_package(PythonInterp)
find_package(PythonLibs)
find_package(NumPy REQUIRED)


# contruct library directories
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; tmp = [s + '/../..' for s in [get_python_lib()]]; ret=';'.join(tmp); print(ret)"
    OUTPUT_VARIABLE PYTHON_SYS_PATH
    RESULT_VARIABLE SYSPATH_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python syspath not found")
endif()

message("Python syspaths: " ${PYTHON_SYS_PATH})




# contruct include directories
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_lib; tmp = [s + '/../../../include' for s in [get_python_lib()]]; ret=';'.join(tmp); print(ret)"
    OUTPUT_VARIABLE CMAKE_REQUIRED_INCLUDES
    RESULT_VARIABLE SYSPATH_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python include path not found")
endif()

message("Include paths: " ${CMAKE_REQUIRED_INCLUDES})

set(NUMPY_INC_DIR ${NumPy_INCLUDE_DIR})


if(NOT NumPy_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()


# adding BLAS library dir if given by environment variable
if(DEFINED ENV{BLAS_LIB_DIR})
  set(BLAS_DIR "$ENV{BLAS_LIB_DIR}")
else()
  set(FIND_NUMPY_BLAS_SCRIPT "${PROJECT_SOURCE_DIR}/cmake/find_numpy_blas_dir.py")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${FIND_NUMPY_BLAS_SCRIPT}
    OUTPUT_VARIABLE BLAS_DIR
    RESULT_VARIABLE NUMPY_BLAS_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # If that failed, try a simpler approach
  if(NUMPY_BLAS_NOT_FOUND OR "${BLAS_DIR}" STREQUAL "")
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, os; print(os.path.abspath(os.path.join(os.path.dirname(numpy.__file__), '..', 'lib')))"
      OUTPUT_VARIABLE BLAS_DIR
      RESULT_VARIABLE NUMPY_BLAS_NOT_FOUND2
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
endif()

message(STATUS "BLAS_DIR: ${BLAS_DIR}")


find_package(PythonExtensions REQUIRED)


set(CMAKE_VERBOSE_MAKEFILE ON)

#################################################################
# find MPI libraries if MPI is enables by environment variable PIQUASSOBOOST_MPI

if(DEFINED ENV{QGD_MPI})
  find_package(MPI REQUIRED)

  # openmpi which has a different c++ bindings
  #add_definitions(-DOMPI_SKIP_MPICXX)

  # setting basic compile flags
  list(APPEND CXX_FLAGS_DEBUG "-D__MPI__")
  list(APPEND CXX_FLAGS_RELEASE "-D__MPI__")

  list(APPEND EXTRA_INCLUDES "${MPI_C_INCLUDE_PATH}") 

endif()

#################################################################
# find CBLAS libraries. Hints are given by numpy library directory via variable BLAS_DIR
# Provide hints to FindBLAS
set(ENV{LD_LIBRARY_PATH} "${BLAS_DIR}:$ENV{LD_LIBRARY_PATH}")

# Set up library search paths
if(BLAS_DIR AND NOT "${BLAS_DIR}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "${BLAS_DIR}")
  list(APPEND CMAKE_LIBRARY_PATH "${BLAS_DIR}")
endif()

# On Windows with conda, try to find BLAS in conda environment
if(WIN32)
  # Check if we're in a conda environment
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_LIBRARY_PATH "${CONDA_PREFIX}/Library/lib")
    list(APPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}/Library")
    message(STATUS "Using conda environment: ${CONDA_PREFIX}")
  endif()
endif()

# Set PATH for runtime library finding (Windows)
if(WIN32)
  if(BLAS_DIR AND NOT "${BLAS_DIR}" STREQUAL "")
    set(ENV{PATH} "${BLAS_DIR}/Library/bin;$ENV{PATH}")
  endif()
  if(DEFINED ENV{CONDA_PREFIX})
    set(ENV{PATH} "$ENV{CONDA_PREFIX}/Library/bin;$ENV{PATH}")
  endif()
endif()

# Use MODULE mode to avoid config file issues (like nvpl)
# This forces FindBLAS to use the traditional module instead of looking for config files
set(BLAS_FOUND FALSE)

# Try to find BLAS libraries directly first (Windows-friendly)
if(WIN32)
  # On Windows, look for openblas or blas libraries
  find_library(BLAS_LIBRARIES
    NAMES openblas blas
    PATHS
      ${CMAKE_LIBRARY_PATH}
      ENV{PATH}
    NO_DEFAULT_PATH
  )
  
  if(BLAS_LIBRARIES)
    set(BLAS_FOUND TRUE)
    message(STATUS "Found BLAS library: ${BLAS_LIBRARIES}")
  endif()
endif()

# If not found via direct search, use find_package with MODULE mode
if(NOT BLAS_FOUND)
  # Force module mode to avoid nvpl config file requirement
  find_package(BLAS MODULE REQUIRED)
endif()


find_package(LAPACK REQUIRED)

# macOS-specific OpenMP detection
# On macOS, OpenMP is typically installed via Homebrew as libomp
# CMake's FindOpenMP module needs help finding it
if(APPLE)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(HOMEBREW_LIBOMP_PREFIX)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include" CACHE STRING "C compiler flags for OpenMP" FORCE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include" CACHE STRING "C++ compiler flags for OpenMP" FORCE)
    set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "C library names for OpenMP" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "C++ library names for OpenMP" FORCE)
    find_library(OpenMP_omp_LIBRARY
      NAMES omp
      PATHS ${HOMEBREW_LIBOMP_PREFIX}/lib
      NO_DEFAULT_PATH
    )
    if(OpenMP_omp_LIBRARY)
      set(OpenMP_omp_LIBRARY ${OpenMP_omp_LIBRARY} CACHE PATH "Path to OpenMP omp library" FORCE)
    endif()
    message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
    message(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP_omp_LIBRARY: ${OpenMP_omp_LIBRARY}")
    # Add linker flags to ensure libomp is found during linking
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${HOMEBREW_LIBOMP_PREFIX}/lib -lomp")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${HOMEBREW_LIBOMP_PREFIX}/lib -lomp")
  endif()
endif()

find_package(OpenMP REQUIRED)

# Convert OpenMP flags from string to list for proper handling on macOS
# This is especially important on macOS where flags are set manually as a single string
if(APPLE AND OpenMP_CXX_FLAGS)
  separate_arguments(OpenMP_CXX_FLAGS_LIST UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
else()
  # On other platforms, use the flags as-is (they should already be a list)
  set(OpenMP_CXX_FLAGS_LIST ${OpenMP_CXX_FLAGS})
endif()

list(APPEND EXTRA_INCLUDES "${OpenMP_CXX_INCLUDE_DIRS}") 

# make difference between MKL and OPENBLAS by checking specific functions
set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}" "-lm")
check_function_exists(MKL_Set_Num_Threads BLAS_IS_MKL)
check_function_exists(openblas_set_num_threads BLAS_IS_OPENBLAS)

message(STATUS "BLAS_IS_MKL: ${BLAS_IS_MKL}")
message(STATUS "BLAS_IS_OPENBLAS: ${BLAS_IS_OPENBLAS}")

# check the presence of lapacke library
check_function_exists(LAPACKE_zgesvd LAPACKE_FOUND)
set(CMAKE_REQUIRED_LIBRARIES "")

# If MKL is enabled
if(${BLAS_IS_MKL})
  message(STATUS "MKL is enabled")
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=1")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=1")

  # If OpenBlas is enabled
elseif(${BLAS_IS_OPENBLAS})
  message(STATUS "OpenBLAS is enabled")
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=2")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=2")
else()
  message(STATUS "OMP is enabled")
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=0")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=0")
endif()


# if LAPACKE not found try another round
if(${LAPACKE_FOUND})
    # LAPACKE functions found in existing libraries, use LAPACK_LIBRARIES
    set(LAPACKE_LIBRARIES ${LAPACK_LIBRARIES})
else()
    # On macOS with OpenBLAS, LAPACKE is typically included in OpenBLAS
    if(APPLE AND BLAS_IS_OPENBLAS)
        message(STATUS "macOS with OpenBLAS: LAPACKE should be included in OpenBLAS")
        # Try to find OpenBLAS library which includes LAPACKE
        find_library(OPENBLAS_LIB
            NAMES openblas
            PATHS /opt/homebrew/lib /usr/local/lib
            NO_DEFAULT_PATH
        )
        if(OPENBLAS_LIB)
            set(LAPACKE_LIBRARIES ${OPENBLAS_LIB})
            message(STATUS "Found OpenBLAS library with LAPACKE: ${OPENBLAS_LIB}")
        else()
            # Fallback to LAPACK_LIBRARIES
            set(LAPACKE_LIBRARIES ${LAPACK_LIBRARIES})
        endif()
    elseif(APPLE)
        # On macOS, try to find separate LAPACKE library first
        find_library(LAPACKE_LIB
            NAMES lapacke
            PATHS /opt/homebrew/lib /usr/local/lib
            NO_DEFAULT_PATH
        )
        if(LAPACKE_LIB)
            set(LAPACKE_LIBRARIES ${LAPACKE_LIB})
            message(STATUS "Found separate LAPACKE library: ${LAPACKE_LIB}")
        else()
            # Try OpenBLAS which often includes LAPACKE on macOS
            find_library(OPENBLAS_LIB
                NAMES openblas
                PATHS /opt/homebrew/lib /usr/local/lib
                NO_DEFAULT_PATH
            )
            if(OPENBLAS_LIB)
                set(LAPACKE_LIBRARIES ${OPENBLAS_LIB})
                message(STATUS "Using OpenBLAS library for LAPACKE: ${OPENBLAS_LIB}")
            else()
                message(STATUS "LAPACKE not found as separate library on macOS, assuming it's in LAPACK/OpenBLAS")
                # Set to LAPACK_LIBRARIES since LAPACKE is part of it
                set(LAPACKE_LIBRARIES ${LAPACK_LIBRARIES})
            endif()
        endif()
    else()
        # Try to find separate LAPACKE library
        find_library(LAPACKE_LIBRARIES lapacke)
        if(NOT LAPACKE_LIBRARIES)
            message(FATAL_ERROR "Could not find LAPACKE library")
        endif()
    endif()
endif()



# checking for AVX/AVX2 support
CHECK_FOR_AVX()

# Identify the compiler type and set compiler specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message("-- Using Clang compiler")
  # setting basic compile flags for Clang
  list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")
  list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-ggdb" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message("-- Using GNU compiler")
  # setting basic compile flags for GCC
  list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive" "-g3" "-ggdb")
  list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-ggdb" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive" "-ftree-vectorize")

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx512f" "-mfma" "-DUSE_AVX")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx2" "-mfma" "-DUSE_AVX")
    list(APPEND EXTRA_INCLUDES "./squander/src-cpp/gates/kernels/include/")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx" "-mfma" "-DUSE_AVX")
  endif()


elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  message("-- Using Intel compiler")
  # setting basic compile flags for Intel (similar to GCC/Clang)
  list(APPEND CXX_FLAGS_DEBUG "-Wall" "-DDEBUG")
  list(APPEND CXX_FLAGS_RELEASE "-O3" "-DNDEBUG")
  
  if (BLAS_IS_MKL)
    list(APPEND CXX_FLAGS_DEBUG "-mkl" "-tbb")
    list(APPEND CXX_FLAGS_RELEASE "-mkl" "-tbb")
  endif()

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx512f" "-mfma" "-DUSE_AVX512F")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx2" "-mfma" "-DUSE_AVX")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx" "-mfma" "-DUSE_AVX")
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  message("-- Using Visual Studio C++ compiler")

  # Add /bigobj flag for large object files (needed for Windows compilation)
  # This flag is critical for files that generate large object files (like numpy_interface.cpp)
  list(APPEND CXX_FLAGS "/bigobj")
  # Also ensure it's in the global CMAKE_CXX_FLAGS (applied to all targets)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" CACHE STRING "" FORCE)

  # MSVC-specific flags
  # Note: /O2 (optimization) and /W3 (warnings) are typically set by CMake for Release builds
  # /DNDEBUG is automatically set by CMake for Release configuration
  # No need for -fno-builtin-* or -fpermissive (these are GCC-specific)
  list(APPEND CXX_FLAGS_DEBUG "/W3" "/DDEBUG")
  list(APPEND CXX_FLAGS_RELEASE "/W3" "/DNDEBUG")

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX512F" "-DUSE_AVX512F")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX2" "-DUSE_AVX")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX" "-DUSE_AVX")
  endif()

endif()





############################################################xx
# checking TBB libraries and headers

# On Windows, check for conda environment first
if(WIN32)
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    set(CONDA_TBB_LIB_DIR "${CONDA_PREFIX}/Library/lib")
    set(CONDA_TBB_INC_DIR "${CONDA_PREFIX}/Library/include")
    message(STATUS "Found conda environment: ${CONDA_PREFIX}")
  endif()
endif()

# adding TBB library dir if given by environment variable
if(DEFINED ENV{TBB_LIB_DIR})
  set(TBB_LIB_DIR "$ENV{TBB_LIB_DIR}")
  message(STATUS "Using TBB_LIB_DIR from environment: ${TBB_LIB_DIR}")
elseif(WIN32 AND DEFINED CONDA_PREFIX)
  # Windows with conda: use conda's TBB location
  set(TBB_LIB_DIR "${CONDA_TBB_LIB_DIR}")
  message(STATUS "Using conda TBB library directory: ${TBB_LIB_DIR}")
else()
  # Try to find TBB in Python syspath (fallback)
  set(TBB_LIB_DIR "${PYTHON_SYS_PATH}")
  message(STATUS "Using Python syspath for TBB: ${TBB_LIB_DIR}")
endif()

# Set up search paths
set(TBB_SEARCH_PATHS)
if(TBB_LIB_DIR)
  list(APPEND TBB_SEARCH_PATHS "${TBB_LIB_DIR}")
endif()
if(WIN32 AND DEFINED CONDA_PREFIX)
  list(APPEND TBB_SEARCH_PATHS "${CONDA_TBB_LIB_DIR}")
  # Also check bin directory for DLLs (Windows)
  list(APPEND TBB_SEARCH_PATHS "${CONDA_PREFIX}/Library/bin")
endif()
list(APPEND TBB_SEARCH_PATHS ${PYTHON_SYS_PATH})

# Find TBB library - try multiple names for Windows
set(TBB_LIB)
if(WIN32)
  # On Windows, try different library names
  # TBB from conda might be named tbb, tbb12, or version-specific
  find_library(TBB_LIB
    NAMES tbb12 tbb tbb12.lib tbb.lib
    PATHS ${TBB_SEARCH_PATHS}
    PATH_SUFFIXES lib Library/lib
    NO_DEFAULT_PATH
  )
  
  # If not found with NO_DEFAULT_PATH, try with default paths
  if(NOT TBB_LIB)
    find_library(TBB_LIB
      NAMES tbb12 tbb tbb12.lib tbb.lib
      PATHS ${TBB_SEARCH_PATHS}
      PATH_SUFFIXES lib Library/lib
    )
  endif()
else()
  # Unix-like systems
  find_library(TBB_LIB
    NAMES tbb
    PATHS ${TBB_SEARCH_PATHS}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
  )
  
  if(NOT TBB_LIB)
    find_library(TBB_LIB
      NAMES tbb
      PATHS ${TBB_SEARCH_PATHS}
      PATH_SUFFIXES lib
    )
  endif()
endif()

if(NOT TBB_LIB)
  message(FATAL_ERROR "Could not find TBB library. Searched in: ${TBB_SEARCH_PATHS}")
else()
  message(STATUS "Found TBB library: ${TBB_LIB}")
endif()

# Find TBBMALLOC library
find_library(TBBMALLOC_LIB
  NAMES tbbmalloc tbbmalloc.lib
  PATHS ${TBB_SEARCH_PATHS}
  PATH_SUFFIXES lib Library/lib
  NO_DEFAULT_PATH
)

if(NOT TBBMALLOC_LIB)
  find_library(TBBMALLOC_LIB
    NAMES tbbmalloc tbbmalloc.lib
    PATHS ${TBB_SEARCH_PATHS}
    PATH_SUFFIXES lib Library/lib
  )
endif()

if(NOT TBBMALLOC_LIB)
  message(WARNING "Could not find TBBMALLOC library. Continuing without it.")
else()
  message(STATUS "Found TBBMALLOC library: ${TBBMALLOC_LIB}")
endif()

# Find TBBMALLOC_PROXY library
find_library(TBBMALLOC_PROXY_LIB
  NAMES tbbmalloc_proxy tbbmalloc_proxy.lib
  PATHS ${TBB_SEARCH_PATHS}
  PATH_SUFFIXES lib Library/lib
  NO_DEFAULT_PATH
)

if(NOT TBBMALLOC_PROXY_LIB)
  find_library(TBBMALLOC_PROXY_LIB
    NAMES tbbmalloc_proxy tbbmalloc_proxy.lib
    PATHS ${TBB_SEARCH_PATHS}
    PATH_SUFFIXES lib Library/lib
  )
endif()

if(NOT TBBMALLOC_PROXY_LIB)
  message(WARNING "Could not find TBBMALLOC_PROXY library. Continuing without it.")
else()
  message(STATUS "Found TBBMALLOC_PROXY library: ${TBBMALLOC_PROXY_LIB}")
endif()

# adding TBB include dir
set(TBB_INC_DIR)
set(TBB_HEADER_FOUND FALSE)

# First, try environment variable
if(DEFINED ENV{TBB_INC_DIR})
  set(TBB_INC_DIR "$ENV{TBB_INC_DIR}")
  message(STATUS "TBB_INC_DIR from environment: ${TBB_INC_DIR}")
endif()

# Clean up the path if set
if(TBB_INC_DIR)
  string(STRIP "${TBB_INC_DIR}" TBB_INC_DIR)
  string(REGEX REPLACE "\\*" "" TBB_INC_DIR "${TBB_INC_DIR}")
  string(REGEX REPLACE "  +" " " TBB_INC_DIR "${TBB_INC_DIR}")
  string(STRIP "${TBB_INC_DIR}" TBB_INC_DIR)
  
  # Validate the path
  if("${TBB_INC_DIR}" MATCHES "\\*|  " OR "${TBB_INC_DIR}" STREQUAL "")
    message(WARNING "Invalid TBB_INC_DIR detected: '${TBB_INC_DIR}'. Clearing it.")
    set(TBB_INC_DIR)
  endif()
endif()

# On Windows, also check CONDA_PREFIX
if(WIN32)
  if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    set(CONDA_TBB_INC_DIR "${CONDA_PREFIX}/Library/include")
    message(STATUS "CONDA_PREFIX: ${CONDA_PREFIX}")
    message(STATUS "CONDA_TBB_INC_DIR: ${CONDA_TBB_INC_DIR}")
    
    # If TBB_INC_DIR not set or invalid, use conda path
    if(NOT TBB_INC_DIR OR NOT EXISTS "${TBB_INC_DIR}/tbb/tbb.h")
      set(TBB_INC_DIR "${CONDA_TBB_INC_DIR}")
      message(STATUS "Using conda TBB include directory: ${TBB_INC_DIR}")
    endif()
  endif()
endif()

# Try multiple search paths
set(TBB_SEARCH_PATHS)
if(TBB_INC_DIR)
  list(APPEND TBB_SEARCH_PATHS "${TBB_INC_DIR}")
endif()
if(WIN32 AND DEFINED CONDA_PREFIX)
  list(APPEND TBB_SEARCH_PATHS "${CONDA_PREFIX}/Library/include")
  list(APPEND TBB_SEARCH_PATHS "${CONDA_PREFIX}/include")
endif()
# Add common system paths
if(WIN32)
  list(APPEND TBB_SEARCH_PATHS "C:/Program Files/Intel/oneAPI/tbb/latest/include")
  list(APPEND TBB_SEARCH_PATHS "C:/tbb/include")
endif()

message(STATUS "Searching for TBB header in: ${TBB_SEARCH_PATHS}")

# Try to find TBB header in search paths
foreach(SEARCH_PATH ${TBB_SEARCH_PATHS})
  if(EXISTS "${SEARCH_PATH}/tbb/tbb.h")
    set(TBB_INC_DIR "${SEARCH_PATH}")
    set(TBB_HEADER_FOUND TRUE)
    message(STATUS "Found TBB header at: ${TBB_INC_DIR}/tbb/tbb.h")
    break()
  endif()
endforeach()

# If not found by file check, try check_include_file_cxx
if(NOT TBB_HEADER_FOUND)
  foreach(SEARCH_PATH ${TBB_SEARCH_PATHS})
    set(CMAKE_REQUIRED_FLAGS "-c")
    file(TO_CMAKE_PATH "${SEARCH_PATH}" SEARCH_PATH_NORMALIZED)
    check_include_file_cxx(tbb/tbb.h TBB_HEADER "-I${SEARCH_PATH_NORMALIZED}")
    
    if(TBB_HEADER)
      set(TBB_INC_DIR "${SEARCH_PATH_NORMALIZED}")
      set(TBB_HEADER_FOUND TRUE)
      message(STATUS "Found TBB header via check_include_file_cxx at: ${TBB_INC_DIR}")
      break()
    endif()
  endforeach()
endif()

# Last resort: try without explicit path
if(NOT TBB_HEADER_FOUND)
  set(CMAKE_REQUIRED_FLAGS "-c")
  check_include_file_cxx(tbb/tbb.h TBB_HEADER)
  
  if(TBB_HEADER)
    set(TBB_HEADER_FOUND TRUE)
    message(STATUS "Found TBB header in system paths")
    list(APPEND EXTRA_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
  endif()
endif()

# Add the include directory if found
if(TBB_HEADER_FOUND AND TBB_INC_DIR)
  file(TO_CMAKE_PATH "${TBB_INC_DIR}" TBB_INC_DIR_NORMALIZED)
  list(APPEND EXTRA_INCLUDES "${TBB_INC_DIR_NORMALIZED}")
  message(STATUS "Added TBB include directory: ${TBB_INC_DIR_NORMALIZED}")
elseif(NOT TBB_HEADER_FOUND)
  message(FATAL_ERROR "TBB header tbb.h not found. Searched in: ${TBB_SEARCH_PATHS}. Please set TBB_INC_DIR or install TBB.")
endif()


list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/squander/src-cpp//common/common.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/config_element.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/dot.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/matrix.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/matrix_real.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/matrix_sparse.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/logging.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/Adam.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/grad_descend.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/BFGS_Powell.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/Powells_method.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/Bayes_Opt.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/n_aryGrayCodeCounter.cpp    
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/RL_experience.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/ADAM_BATCHED.cpp    
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/ADAM.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/AGENTS.cpp    
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/BAYES_OPT.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/BAYES_AGENTS.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/BFGS.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/BFGS2.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/COSINE.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/GRAD_DESCEND.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/optimization_engines/GRAD_DESCEND_PARAMETER_SHIFT_RULE.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CU.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CNOT.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/SYC.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CZ.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CH.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Gate.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/UN.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/ON.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CROT.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Gates_block.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/H.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/X.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Y.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Z.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/S.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/SDG.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/T.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Tdg.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/SX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/U1.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/U2.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/U3.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/RY.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CRY.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/RX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/RZ.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CRX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CRZ.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CP.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CCX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/SWAP.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CSWAP.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CR.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Adaptive.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/R.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/RZ_P.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/CZ_NU.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/custom_kernel_1qubit_gate.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/Composite.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_input.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_state_vector_input.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_large_kernel_to_input.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_dedicated_gate_kernel_to_input.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/nn/NN.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/Decomposition_Base.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/Optimization_Interface.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/N_Qubit_Decomposition.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/N_Qubit_Decomposition_adaptive.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/N_Qubit_Decomposition_custom.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/N_Qubit_Decomposition_Cost_Function.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/Sub_Matrix_Decomposition_Cost_Function.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/Sub_Matrix_Decomposition.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/N_Qubit_Decomposition_Tree_Search.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/N_Qubit_Decomposition_Tabu_Search.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/variational_quantum_eigensolver/Variational_Quantum_Eigensolver_Base.cpp  
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/variational_quantum_eigensolver/Generative_Quantum_Machine_Learning_Base.cpp  
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/random_unitary/Random_Unitary.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/random_unitary/Random_Orthogonal.cpp
)


if (${HAVE_AVX512F_EXTENSIONS})
  list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_input_AVX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_state_vector_input_AVX.cpp    
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_large_kernel_to_input_AVX.cpp    
  )
elseif (${HAVE_AVX2_EXTENSIONS})
  list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_input_AVX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_state_vector_input_AVX.cpp        
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_large_kernel_to_input_AVX.cpp        
  )
elseif (${HAVE_AVX_EXTENSIONS})
  list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_input_AVX.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_kernel_to_state_vector_input_AVX.cpp        
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/apply_large_kernel_to_input_AVX.cpp        
  )
endif()



if(DEFINED ENV{QGD_MPI})

    list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/mpi_base.cpp
    )

endif()


if (DEFINED ENV{QGD_DFE})
  list(APPEND qgd_files 
      ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/common_DFE.cpp
  )

  list(APPEND CXX_FLAGS_DEBUG "-D__DFE__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__DFE__=1")


endif()



if (DEFINED ENV{QGD_GROQ})
  list(APPEND qgd_files 
      ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/common_GROQ.cpp
  )

  list(APPEND CXX_FLAGS_DEBUG "-D__GROQ__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__GROQ__=1")


endif()


add_library(qgd SHARED
    ${qgd_files})

# adding compile options
target_compile_options(qgd PUBLIC
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    ${OpenMP_CXX_FLAGS_LIST}
)

# adding linking options
target_link_libraries( qgd PRIVATE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${TBBMALLOC_LIB}
    ${TBBMALLOC_PROXY_LIB}
    ${TBB_LIB}
    ${MPI_C_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    )


target_include_directories(qgd PRIVATE
                            ${EXTRA_INCLUDES}
                            .
                            ./squander/src-cpp/common/include
                            ./squander/src-cpp/gates/include
                            ./squander/src-cpp/decomposition/include
                            ./squander/src-cpp/random_unitary/include
                            ./squander/src-cpp/optimization_engines/include
                            ./squander/src-cpp/nn/include
                            ./squander/src-cpp/gates/kernels/include
                            ./squander/src-cpp/variational_quantum_eigensolver/include
                            )






configure_file(${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include/Config.h.in
               ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include/Config.h)


# Set BUILD_RPATH for qgd to prioritize conda libraries
if(DEFINED ENV{CONDA_PREFIX})
    set_target_properties(qgd PROPERTIES
        BUILD_RPATH "${CONDA_PREFIX}/lib"
        BUILD_RPATH_USE_ORIGIN TRUE
    )
endif()

set_target_properties(
    qgd
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/squander
    )

if(WIN32)
    set_target_properties(
        qgd
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/squander/decomposition
    )
endif()


# adding CMAKE files for Python extensions
add_subdirectory (squander/decomposition)
add_subdirectory (squander/gates)
add_subdirectory (squander/nn)
add_subdirectory (squander/VQA)

if(DEFINED ENV{QGD_CTEST})
    # adding CMAKE files for executables
    add_subdirectory (test_standalone)
endif()


install(TARGETS qgd 
        LIBRARY DESTINATION squander
        RUNTIME DESTINATION squander  # Add this for Windows
        COMPONENT python)

