
set(EXT_DIR ${PROJECT_SOURCE_DIR}/qgd_python/nn)


add_library( qgd_nn_Wrapper MODULE
    ${EXT_DIR}/qgd_nn_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/common/numpy_interface.cpp        
)


ADD_DEPENDENCIES (qgd_nn_Wrapper qgd)

target_link_libraries (qgd_nn_Wrapper qgd  ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_nn_Wrapper)


# adding compile options
target_compile_options(qgd_nn_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_nn_Wrapper PRIVATE    ${PROJECT_SOURCE_DIR}/common/numpy_interface.cpp
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/nn/include
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_nn_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)




install(TARGETS qgd_nn_Wrapper LIBRARY 
         DESTINATION qgd_python/nn)


