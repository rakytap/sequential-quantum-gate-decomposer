# CMAKE to create standalone executables


# add tests to the build
add_test(decomposition_test decomposition_test ...)
add_test(custom_gate_structure_test custom_gate_structure_test ...)
add_test(apply_kernel_test apply_kernel_test ...)

if(ENABLE_FLOAT32_MATRIX)
  add_test(NAME MatrixFloatTests COMMAND test_matrix_float)
endif()


# Add executable called "decomposition_test" that is built from the source files
# "decomposition_test.cpp". The extensions are automatically found.
add_executable (decomposition_test decomposition_test.cpp)
add_executable (custom_gate_structure_test custom_gate_structure_test.cpp)
add_executable(apply_kernel_test apply_kernel_test.cpp)

if(ENABLE_FLOAT32_MATRIX)
  add_executable(test_matrix_float test_matrix_float.cpp)
endif()


target_include_directories(decomposition_test PRIVATE
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/random_unitary/include
                            ${EXTRA_INCLUDES})



target_include_directories(custom_gate_structure_test PRIVATE
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/random_unitary/include
                            ${EXTRA_INCLUDES})



target_include_directories(apply_kernel_test PRIVATE
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/kernels/include
                            ${EXTRA_INCLUDES})

# Link the executable to the qgd library. Since the qgd library has
# public include directories we will use those link directories when building
# decomposition_test
target_link_libraries (decomposition_test 
                        qgd
                        ${TBBMALLOC_LIB}
                        ${TBBMALLOC_PROXY_LIB}
                        ${TBB_LIB}
                        ${BLAS_LIBRARIES}
                        ${GSL_LIBS})
target_link_libraries (custom_gate_structure_test
                        qgd
                        ${TBBMALLOC_LIB}
                        ${TBBMALLOC_PROXY_LIB}
                        ${TBB_LIB}
                        ${BLAS_LIBRARIES}
                        ${GSL_LIBS})
target_link_libraries(apply_kernel_test
                        qgd
                        ${TBBMALLOC_LIB}
                        ${TBBMALLOC_PROXY_LIB}
                        ${TBB_LIB}
                        ${BLAS_LIBRARIES}
                        ${GSL_LIBS})

if(ENABLE_FLOAT32_MATRIX)
  target_include_directories(test_matrix_float PRIVATE
                              ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                              ${EXTRA_INCLUDES})

  target_link_libraries(test_matrix_float
                          qgd
                          ${TBBMALLOC_LIB}
                          ${TBBMALLOC_PROXY_LIB}
                          ${TBB_LIB})
endif()

