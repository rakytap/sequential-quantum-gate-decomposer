cmake_minimum_required(VERSION 3.10.2)

# CMAKE to create the shared library of the quantum gate decomposition project

# set the project name and version
project(CQGD VERSION 1.8.7)

# reuse compilation time linking for use runtime linking 
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(EXTRA_INCLUDES)
set(BLAS_DIR)
set(PYTHON_PLAT_LIB)


#enable test target
enable_testing()


# find out python packages
find_package(PythonInterp)
find_package(PythonLibs)
find_package(NumPy REQUIRED)


exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"from sysconfig import get_paths; info = get_paths(); platlib=info.get('platlib',' '); print(platlib)\""
             OUTPUT_VARIABLE PYTHON_PLAT_LIB
             RETURN_VALUE PLATLIB_NOT_FOUND
            )
if(PLATLIB_NOT_FOUND)
    message(FATAL_ERROR "Python platlib not found")
endif()

set(ENV{PYTHONPATH} "$ENV{PYTHONPATH};${PYTHON_PLAT_LIB}")
message("PYTHONPATH is set to $ENV{PYTHONPATH}")

# contruct library directories
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"from distutils.sysconfig import get_python_lib; tmp = [s + '/../..' for s in [get_python_lib()]]; ret=';'.join(tmp); print(ret)\""
             OUTPUT_VARIABLE PYTHON_SYS_PATH
             RETURN_VALUE SYSPATH_NOT_FOUND
            )
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python syspath not found")
endif()

message("Python syspaths: " ${PYTHON_SYS_PATH})




# contruct include directories
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"from distutils.sysconfig import get_python_lib; tmp = [s + '/../../../include' for s in [get_python_lib()]]; ret=';'.join(tmp); print(ret)\""
             OUTPUT_VARIABLE CMAKE_REQUIRED_INCLUDES
             RETURN_VALUE SYSPATH_NOT_FOUND
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

# Determine CBLAS library directory behind Numpy
exec_program(
  ${PYTHON_EXECUTABLE}
  ARGS "-c \"import numpy; blas_info=numpy.__config__.get_info('blas_opt_info'); libs = blas_info.get('library_dirs'); print(libs[0])\""
  OUTPUT_VARIABLE BLAS_DIR
  RETURN_VALUE NUMPY_BLAS_NOT_FOUND
)

endif()



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

set(ENV{LD_LIBRARY_PATH} "${BLAS_DIR}:$ENV{LD_LIBRARY_PATH}")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# make difference between MKL and OPENBLAS by checking specific functions
set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}" "-lm")
check_function_exists(MKL_Set_Num_Threads BLAS_IS_MKL)
check_function_exists(openblas_set_num_threads BLAS_IS_OPENBLAS)

# check the presence of lapacke library
check_function_exists(LAPACKE_zggev LAPACKE_FOUND)
set(CMAKE_REQUIRED_LIBRARIES "")

# If MKL is enabled
if(${BLAS_IS_MKL})
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=1")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=1")

  # If OpenBlas is enabled
elseif(${BLAS_IS_OPENBLAS})
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=2")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=2")
else()

  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=0")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=0")
endif()


# if LAPACKE not found try another round
if(${LAPACKE_FOUND} )
    set(LAPACKE_LIBRARIES)
else()
    find_library(LAPACKE_LIBRARIES lapacke REQUIRED)    
endif()

# setting basic compile flags
list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")
list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")


# checking for AVX/AVX2 support
CHECK_FOR_AVX()

# Identify the compiler type and set compiler specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message("-- Using Clang compiler")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message("-- Using GNU compiler")
  list(APPEND CXX_FLAGS_DEBUG "-g3" "-ggdb")
  list(APPEND CXX_FLAGS_RELEASE "-ftree-vectorize")

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx512f" "-mfma" "-DUSE_AVX")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx2" "-mfma" "-DUSE_AVX")
    list(APPEND EXTRA_INCLUDES "./gates/kernels/include/")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx" "-mfma" "-DUSE_AVX")
  endif()


elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  message("-- Using Intel compiler")
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

# adding TBB library dir if given by environment variable
if(DEFINED ENV{TBB_LIB_DIR})
  if (WIN32)
    find_library(TBB_LIB tbb12
                 PATHS $ENV{TBB_LIB_DIR}
                 NO_DEFAULT_PATH
                 REQUIRED)
  else()
    find_library(TBB_LIB tbb
                 PATHS $ENV{TBB_LIB_DIR}
                 NO_DEFAULT_PATH
                 REQUIRED)
  endif()

  find_library(TBBMALLOC_LIB tbbmalloc
               PATHS $ENV{TBB_LIB_DIR}
               NO_DEFAULT_PATH
               REQUIRED)

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy
               PATHS $ENV{TBB_LIB_DIR}
               NO_DEFAULT_PATH
               REQUIRED)

else()

  if (WIN32)
    find_library(TBB_LIB tbb12
                 PATHS ${PYTHON_SYS_PATH}
                 NO_DEFAULT_PATH
                 REQUIRED)
  else()
    find_library(TBB_LIB tbb
                 PATHS ${PYTHON_SYS_PATH}
                 NO_DEFAULT_PATH
                 REQUIRED)  
  endif()

  find_library(TBBMALLOC_LIB tbbmalloc
               PATHS ${PYTHON_SYS_PATH}
               NO_DEFAULT_PATH
               REQUIRED)

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy
               PATHS ${PYTHON_SYS_PATH}
               NO_DEFAULT_PATH
               REQUIRED)
endif()


# adding TBB include dir
if(DEFINED ENV{TBB_INC_DIR})

  set(CMAKE_REQUIRED_FLAGS "-c")
  check_include_file_cxx(tbb/tbb.h TBB_HEADER "-I$ENV{TBB_INC_DIR}")

  if(NOT TBB_HEADER)
    message(FATAL_ERROR "TBB header tbb.h not found")
  endif()

  message("-- Adding include directory $ENV{TBB_INC_DIR}")
  list(APPEND EXTRA_INCLUDES "$ENV{TBB_INC_DIR}")

else()
 
  set(CMAKE_REQUIRED_FLAGS "-c")
  check_include_file_cxx(tbb/tbb.h TBB_HEADER )
  list(APPEND EXTRA_INCLUDES "${CMAKE_REQUIRED_INCLUDES}") 

  if(NOT TBB_HEADER)
    message(FATAL_ERROR "TBB header tbb.h not found")
  endif()

  list(APPEND EXTRA_INCLUDES "${TBB_HEADER}")

endif()


list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/common/common.cpp
    ${PROJECT_SOURCE_DIR}/common/config_element.cpp
    ${PROJECT_SOURCE_DIR}/common/dot.cpp
    ${PROJECT_SOURCE_DIR}/common/matrix.cpp
    ${PROJECT_SOURCE_DIR}/common/matrix_real.cpp
    ${PROJECT_SOURCE_DIR}/common/matrix_sparse.cpp
    ${PROJECT_SOURCE_DIR}/common/logging.cpp
    ${PROJECT_SOURCE_DIR}/common/Adam.cpp
    ${PROJECT_SOURCE_DIR}/common/grad_descend.cpp
    ${PROJECT_SOURCE_DIR}/common/BFGS_Powell.cpp
    ${PROJECT_SOURCE_DIR}/common/Powells_method.cpp
    ${PROJECT_SOURCE_DIR}/common/Bayes_Opt.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/RL_experience.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/ADAM_BATCHED.cpp    
    ${PROJECT_SOURCE_DIR}/optimization_engines/ADAM.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/AGENTS.cpp    
    ${PROJECT_SOURCE_DIR}/optimization_engines/BAYES_OPT.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/BAYES_AGENTS.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/BFGS.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/BFGS2.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/COSINE.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/GRAD_DESCEND.cpp
    ${PROJECT_SOURCE_DIR}/optimization_engines/GRAD_DESCEND_PARAMETER_SHIFT_RULE.cpp
    ${PROJECT_SOURCE_DIR}/gates/CNOT.cpp
    ${PROJECT_SOURCE_DIR}/gates/SYC.cpp
    ${PROJECT_SOURCE_DIR}/gates/CZ.cpp
    ${PROJECT_SOURCE_DIR}/gates/CH.cpp
    ${PROJECT_SOURCE_DIR}/gates/Gate.cpp
    ${PROJECT_SOURCE_DIR}/gates/UN.cpp
    ${PROJECT_SOURCE_DIR}/gates/ON.cpp
    ${PROJECT_SOURCE_DIR}/gates/Gates_block.cpp
    ${PROJECT_SOURCE_DIR}/gates/X.cpp
    ${PROJECT_SOURCE_DIR}/gates/Y.cpp
    ${PROJECT_SOURCE_DIR}/gates/Z.cpp
    ${PROJECT_SOURCE_DIR}/gates/SX.cpp
    ${PROJECT_SOURCE_DIR}/gates/U3.cpp
    ${PROJECT_SOURCE_DIR}/gates/RY.cpp
    ${PROJECT_SOURCE_DIR}/gates/CRY.cpp
    ${PROJECT_SOURCE_DIR}/gates/Adaptive.cpp
    ${PROJECT_SOURCE_DIR}/gates/RX.cpp
    ${PROJECT_SOURCE_DIR}/gates/RZ.cpp
    ${PROJECT_SOURCE_DIR}/gates/RZ_P.cpp
    ${PROJECT_SOURCE_DIR}/gates/custom_kernel_1qubit_gate.cpp
    ${PROJECT_SOURCE_DIR}/gates/Composite.cpp
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_input.cpp
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_state_vector_input.cpp
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_large_kernel_to_input.cpp
    ${PROJECT_SOURCE_DIR}/nn/NN.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Decomposition_Base.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Optimization_Interface.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/N_Qubit_Decomposition.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/N_Qubit_Decomposition_adaptive.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/N_Qubit_Decomposition_custom.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/N_Qubit_Decomposition_Cost_Function.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Sub_Matrix_Decomposition_Cost_Function.cpp
    ${PROJECT_SOURCE_DIR}/decomposition/Sub_Matrix_Decomposition.cpp
    ${PROJECT_SOURCE_DIR}/variational_quantum_eigensolver/Variational_Quantum_Eigensolver_Base.cpp  
    ${PROJECT_SOURCE_DIR}/random_unitary/Random_Unitary.cpp
    ${PROJECT_SOURCE_DIR}/random_unitary/Random_Orthogonal.cpp
)


if (${HAVE_AVX512F_EXTENSIONS})
  list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_input_AVX.cpp
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_state_vector_input_AVX.cpp    
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_large_kernel_to_input_AVX.cpp    
  )
elseif (${HAVE_AVX2_EXTENSIONS})
  list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_input_AVX.cpp
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_state_vector_input_AVX.cpp        
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_large_kernel_to_input_AVX.cpp        
  )
elseif (${HAVE_AVX_EXTENSIONS})
  list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_input_AVX.cpp
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_kernel_to_state_vector_input_AVX.cpp        
    ${PROJECT_SOURCE_DIR}/gates/kernels/apply_large_kernel_to_input_AVX.cpp        
  )
endif()



if(DEFINED ENV{QGD_MPI})

    list(APPEND qgd_files 
    ${PROJECT_SOURCE_DIR}/common/mpi_base.cpp
    )

endif()


if (DEFINED ENV{QGD_DFE})
  list(APPEND qgd_files 
      ${PROJECT_SOURCE_DIR}/common/common_DFE.cpp
  )

  list(APPEND CXX_FLAGS_DEBUG "-D__DFE__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__DFE__=1")


endif()


add_library(qgd SHARED
    ${qgd_files})

# adding compile options
target_compile_options(qgd PUBLIC
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding linking options
target_link_libraries( qgd PRIVATE
    "${BLAS_LIBRARIES}"
    "${LAPACK_LIBRARIES}"
    "${LAPACKE_LIBRARIES}"
    ${TBBMALLOC_LIB}
    ${TBBMALLOC_PROXY_LIB}
    ${TBB_LIB}
    ${MPI_C_LIBRARIES}
    )


target_include_directories(qgd PRIVATE
                            .
                            ./common/include
                            ./gates/include
                            ./decomposition/include
                            ./random_unitary/include
                            ./optimization_engines/include
                            ./nn/include
                            ./gates/kernels/include
                            ./variational_quantum_eigensolver/include
                            ${EXTRA_INCLUDES})



set_target_properties(qgd PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/common.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/dot.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/Config.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/matrix_base.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/matrix.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/matrix_real.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/matrix_sparse.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/common/include/QGDTypes.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/CNOT.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/SYC.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/CZ.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/CH.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/Gate.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/UN.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/ON.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/Gates_block.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/X.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/SX.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/U3.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/RX.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/RY.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/CRY.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/Adaptive.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/RZ.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/gates/include/Composite.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/nn/include/NN.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Decomposition_Base.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Optimization_Interface.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition_adaptive.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition_adaptive_general.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition_custom.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition_Cost_Function.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/N_Qubit_Decomposition.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Sub_Matrix_Decomposition_Cost_Function.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/decomposition/include/Sub_Matrix_Decomposition.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/variational_quantum_eigensolver/include/Variational_Quantum_Eigensolver_Base.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/random_unitary/include/Random_Unitary.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/random_unitary/include/Random_Orthogonal.h
)





configure_file(${PROJECT_SOURCE_DIR}/common/include/Config.h.in
               ${PROJECT_SOURCE_DIR}/common/include/Config.h)


set_target_properties(
    qgd
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/qgd_python
    )



# adding CMAKE files for Python extensions
add_subdirectory (qgd_python/decomposition)
add_subdirectory (qgd_python/gates)
add_subdirectory (qgd_python/nn)
add_subdirectory (qgd_python/variational_quantum_eigensolver)

if(DEFINED ENV{QGD_CTEST})
    # adding CMAKE files for executables
    add_subdirectory (test_standalone)
endif()


install(TARGETS qgd LIBRARY DESTINATION qgd_python)




