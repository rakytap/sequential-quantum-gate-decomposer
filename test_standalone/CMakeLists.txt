# CMAKE to create standalone executables


# add tests to the build
add_test(decomposition_test decomposition_test ...)
add_test(custom_gate_structure_test custom_gate_structure_test ...)


# Add executable called "decomposition_test" that is built from the source files
# "decomposition_test.cpp". The extensions are automatically found.
add_executable (decomposition_test decomposition_test.cpp)
add_executable (custom_gate_structure_test custom_gate_structure_test.cpp)


target_include_directories(decomposition_test PRIVATE
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${PROJECT_SOURCE_DIR}/random_unitary/include
                            ${EXTRA_INCLUDES})



target_include_directories(custom_gate_structure_test PRIVATE
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${PROJECT_SOURCE_DIR}/random_unitary/include
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


