
set(EXT_DIR ${PROJECT_SOURCE_DIR}/squander/variational_quantum_eigensolver)


add_library( qgd_Variational_Quantum_Eigensolver_Base_Wrapper MODULE
    ${EXT_DIR}/qgd_VQE_Base_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp        
)


ADD_DEPENDENCIES (qgd_Variational_Quantum_Eigensolver_Base_Wrapper qgd)

target_link_libraries (qgd_Variational_Quantum_Eigensolver_Base_Wrapper qgd  ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_Variational_Quantum_Eigensolver_Base_Wrapper)


# adding compile options
target_compile_options(qgd_Variational_Quantum_Eigensolver_Base_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_Variational_Quantum_Eigensolver_Base_Wrapper PRIVATE   
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/variational_quantum_eigensolver/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_Variational_Quantum_Eigensolver_Base_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)




install(TARGETS qgd_Variational_Quantum_Eigensolver_Base_Wrapper LIBRARY 
         DESTINATION squander/variational_quantum_eigensolver)
