cmake_minimum_required(VERSION 3.10.2)

# CMAKE to create the shared library of the quantum gate decomposition project

# set the project name and version
project(CQGD VERSION 1.4)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_OPENBLAS "Use OpenBLAS for CBLAS implementation" ON)
option(USE_MKL "Use Intel MKL for CBLAS implementation" OFF)

# include CMAKE modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)

# variables for compile and link options
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(TBB_LIBS)
set(GSL_LIBS)
set(EXTRA_INCLUDES)
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(TBB_LIBS)
set(BLAS_LIBRARIES)
set(BLAS_LINKER_FLAGS)
set(NUMPY_BLAS_LIB_DIR)
set(NUMPY_INC_DIR)
set(PYTHON_LIB_DIR)


#enable test target
enable_testing()


#################################################################
# checking python developer packages


# find out python packages
find_package(PythonInterp)
find_package(PythonLibs)


# Determine Numpy input directory
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INC_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()


# Determine CBLAS library directory Numpy
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; blas_info=numpy.__config__.get_info('blas_opt_info'); libs=blas_info.get('library_dirs',' '); print(libs[0])\""
             OUTPUT_VARIABLE NUMPY_BLAS_LIB_DIR
             RETURN_VALUE NUMPY_BLAS_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()



#################################################################



# If MKL is enabled, ObenBLAS should be disabled
if(USE_MKL)
  set(USE_OPENBLAS OFF)
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=1")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=1")
  list(APPEND BLAS_LIBRARIES "-Wl,-rpath=${NUMPY_BLAS_LIB_DIR}" "-L${NUMPY_BLAS_LIB_DIR}" "-lmkl_rt" "-lpthread")
endif()

# If OpenBlas is enabled, MKL should be disabled
if(USE_OPENBLAS)
  set(USE_MKL OFF)
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=2")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=2")
  list(APPEND BLAS_LIBRARIES "-Wl,-rpath=${NUMPY_BLAS_LIB_DIR}" "-L${NUMPY_BLAS_LIB_DIR}" "-lopenblas")
endif()



# If no CLBAS library was given for CMAKE trying to find out the BLAS library
if((NOT USE_MKL) AND (NOT USE_OPENBLAS))
  find_package(BLAS)

  if(NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS library not found")
  endif()

  list(APPEND BLAS_LIBRARIES ${BLAS_LIBRARIES})
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=0")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=0")

endif()


# setting basic compile flags
list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")
list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-msse3" "-mssse3" "-msse4.1" "-mavx2" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")


# Identify the compiler type and set compiler specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message("-- Using Clang compiler")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message("-- Using GNU compiler")
  list(APPEND CXX_FLAGS_DEBUG "-g3" "-ggdb")
  list(APPEND CXX_FLAGS_RELEASE "-ftree-vectorize")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  message("-- Using Intel compiler")
  if (USE_MKL)
    list(APPEND CXX_FLAGS_DEBUG "-mkl" "-tbb")
    list(APPEND CXX_FLAGS_RELEASE "-mkl" "-tbb")
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  message("-- Using Visual Studio C++ compiler")
endif()



############################################################
# checking GNU Scientific libraries and headers

# adding GSL library dir if given
if(DEFINED ENV{GSL_LIB_DIR})
  find_library(GSL_LIB gsl PATH $ENV{GSL_LIB_DIR} )

  if(NOT GSL_LIB)
    message(FATAL_ERROR "GSL library not found")
  endif()


  message("-- Adding library directory $ENV{GSL_LIB_DIR}")
  list(APPEND GSL_LIBS "-Wl,-rpath=$ENV{GSL_LIB_DIR}" "-L$ENV{GSL_LIB_DIR}" "-lgsl")

else()

  find_library(GSL_LIB gsl)

  if(NOT GSL_LIB)
    message(FATAL_ERROR "GSL library not found")
  endif()

  list(APPEND GSL_LIBS "-lgsl")

endif()


# adding GSL include dir
if(DEFINED ENV{GSL_INC_DIR})

  check_include_file_cxx(gsl/gsl_multimin.h GSL_HEADER "-I$ENV{GSL_INC_DIR}")

  if(NOT GSL_HEADER)
    #message(FATAL_ERROR "GSL header gsl_multimin.h not found")
  endif()

  message("-- Adding include directory $ENV{GSL_INC_DIR}")
  list(APPEND EXTRA_INCLUDES "$ENV{GSL_INC_DIR}")

else()

  check_include_file_cxx(gsl/gsl_multimin.h GSL_HEADER)

  if(NOT GSL_HEADER)
    message(FATAL_ERROR "GSL header gsl_multimin.h not found")
  endif()


endif()


############################################################xx
# checking TBB libraries and headers

# adding TBB library dir if given
if(DEFINED ENV{TBB_LIB_DIR})
  find_library(TBB_LIB tbb PATH $ENV{TBB_LIB_DIR} )

  if(NOT TBB_LIB)
    message(FATAL_ERROR "TBB library not found")
  endif()

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy PATH $ENV{TBB_LIB_DIR} )

  if(NOT TBBMALLOC_PROXY_LIB)
    message(FATAL_ERROR "TBBMALLOC_PROXY library not found")
  endif()

  message("-- Adding library directory $ENV{TBB_LIB_DIR}")
  list(APPEND TBB_LIBS "-Wl,-rpath=$ENV{TBB_LIB_DIR}" "-L$ENV{TBB_LIB_DIR}" "-ltbb" "-ltbbmalloc_proxy" "-ltbbmalloc")

else()

  find_library(TBB_LIB tbb)

  if(NOT TBB_LIB)
    message(FATAL_ERROR "TBB library not found")
  endif()

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy )

  if(NOT TBBMALLOC_PROXY_LIB)
    message(FATAL_ERROR "TBBMALLOC_PROXY library not found")
  endif()

  list(APPEND TBB_LIBS "-ltbb" "-ltbbmalloc_proxy" "-ltbbmalloc")

endif()

#set(CMAKE_REQUIRED_LIBRARIES "tbb")


# adding TBB include dir
if(DEFINED ENV{TBB_INC_DIR})

  check_include_file_cxx(tbb/tbb.h TBB_HEADER "-I$ENV{TBB_INC_DIR}")

  if(NOT TBB_HEADER)
    #message(FATAL_ERROR "TBB header tbb.h not found")
  endif()

  message("-- Adding include directory $ENV{TBB_INC_DIR}")
  list(APPEND EXTRA_INCLUDES "$ENV{TBB_INC_DIR}")

else()

  check_include_file_cxx(tbb/tbb.h TBB_HEADER)

  if(NOT TBB_HEADER)
    message(FATAL_ERROR "TBB header tbb.h not found")
  endif()


endif()


add_library( qgd SHARED
    ${PROJECT_SOURCE_DIR}/common/common.cpp
    ${PROJECT_SOURCE_DIR}/common/dot.cpp
    ${PROJECT_SOURCE_DIR}/common/matrix.cpp
    ${PROJECT_SOURCE_DIR}/operations/CNOT.cpp
    ${PROJECT_SOURCE_DIR}/operations/Operation.cpp
    ${PROJECT_SOURCE_DIR}/operations/Operation_block.cpp
    ${PROJECT_SOURCE_DIR}/operations/U3.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Decomposition_Base.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Functor_Cost_Function_Gradient.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Functor_Get_Operation_Products.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/N_Qubit_Decomposition.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/N_Qubit_Decomposition_Cost_Function.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Sub_Matrix_Decomposition_Cost_Function.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Sub_Matrix_Decomposition.cpp  
    ${PROJECT_SOURCE_DIR}/random_unitary/Random_Unitary.cpp
)



set_target_properties(
    qgd
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "qgd${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
    )


# adding compile options
target_compile_options(qgd PUBLIC
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding linking options
target_link_libraries( qgd PRIVATE
    ${TBB_LIBS}
    ${BLAS_LIBRARIES}
    ${GSL_LIBS}
)

target_include_directories(qgd PRIVATE
                            .
                            ./common/include
                            ./operations/include
                            ./decomposition/include
                            ./random_unitary/include
                            ${EXTRA_INCLUDES})

set_target_properties(qgd PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/common.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/dot.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/Config.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/matrix_base.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/matrix.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/QGDTypes.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/operations/include/CNOT.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/operations/include/Operation.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/operations/include/Operation_block.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/operations/include/U3.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Decomposition_Base.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Functor_Cost_Function_Gradient.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Functor_Get_Operation_Products.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition_Cost_Function.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Sub_Matrix_Decomposition_Cost_Function.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Sub_Matrix_Decomposition.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/random_unitary/include/Random_Unitary.h
)

configure_file(${PROJECT_SOURCE_DIR}/common/include/Config.h.in
               ${PROJECT_SOURCE_DIR}/common/include/Config.h)


# adding CMAKE files for Python extensions
add_subdirectory (qgd_python)
add_subdirectory (qgd_python/gates)


# adding CMAKE files for executables
add_subdirectory (test_standalone)






