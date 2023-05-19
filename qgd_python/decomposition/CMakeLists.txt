
set(EXT_DIR ${PROJECT_SOURCE_DIR}/qgd_python/decomposition)


add_library( qgd_N_Qubit_Decomposition_Wrapper MODULE
    ${EXT_DIR}/qgd_N_Qubit_Decomposition_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/common/numpy_interface.cpp        
)


ADD_DEPENDENCIES (qgd_N_Qubit_Decomposition_Wrapper qgd)

target_link_libraries (qgd_N_Qubit_Decomposition_Wrapper qgd  ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_N_Qubit_Decomposition_Wrapper)


# adding compile options
target_compile_options(qgd_N_Qubit_Decomposition_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_N_Qubit_Decomposition_Wrapper PRIVATE    ${PROJECT_SOURCE_DIR}/common/numpy_interface.cpp
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_N_Qubit_Decomposition_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)




install(TARGETS qgd_N_Qubit_Decomposition_Wrapper LIBRARY 
         DESTINATION qgd_python/decomposition)



###############################################################################


add_library( qgd_N_Qubit_Decomposition_adaptive_Wrapper MODULE
    ${EXT_DIR}/qgd_N_Qubit_Decomposition_adaptive_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/common/numpy_interface.cpp    
)


ADD_DEPENDENCIES (qgd_N_Qubit_Decomposition_adaptive_Wrapper qgd)

target_link_libraries (qgd_N_Qubit_Decomposition_adaptive_Wrapper qgd  ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_N_Qubit_Decomposition_adaptive_Wrapper)


# adding compile options
target_compile_options(qgd_N_Qubit_Decomposition_adaptive_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_N_Qubit_Decomposition_adaptive_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_N_Qubit_Decomposition_adaptive_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)




install(TARGETS qgd_N_Qubit_Decomposition_adaptive_Wrapper LIBRARY 
         DESTINATION qgd_python/decomposition)




###############################################################################


add_library( qgd_N_Qubit_Decomposition_custom_Wrapper MODULE
    ${EXT_DIR}/qgd_N_Qubit_Decomposition_custom_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/common/numpy_interface.cpp        
)


ADD_DEPENDENCIES (qgd_N_Qubit_Decomposition_custom_Wrapper qgd)

target_link_libraries (qgd_N_Qubit_Decomposition_custom_Wrapper qgd  ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES}   ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_N_Qubit_Decomposition_custom_Wrapper)


# adding compile options
target_compile_options(qgd_N_Qubit_Decomposition_custom_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_N_Qubit_Decomposition_custom_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_N_Qubit_Decomposition_custom_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)




install(TARGETS qgd_N_Qubit_Decomposition_custom_Wrapper LIBRARY 
         DESTINATION qgd_python/decomposition)



