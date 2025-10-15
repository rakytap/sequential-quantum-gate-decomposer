
set(EXT_DIR ${PROJECT_SOURCE_DIR}/squander/decomposition)


# Unified decomposition wrapper for all decomposition types
# (Decomposition, Adaptive, Custom, Tree Search, Tabu Search)
add_library( qgd_N_Qubit_Decomposition_Wrapper_New MODULE
    ${EXT_DIR}/qgd_N_Qubit_Decomposition_Wrapper_New.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_N_Qubit_Decomposition_Wrapper_New qgd)

target_link_libraries (qgd_N_Qubit_Decomposition_Wrapper_New qgd ${BLAS_LIBRARIES} ${LAPACKE_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_N_Qubit_Decomposition_Wrapper_New)


# adding compile options
target_compile_options(qgd_N_Qubit_Decomposition_Wrapper_New PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_N_Qubit_Decomposition_Wrapper_New PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_N_Qubit_Decomposition_Wrapper_New PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_N_Qubit_Decomposition_Wrapper_New LIBRARY 
         DESTINATION squander/decomposition)

###############################################################################
