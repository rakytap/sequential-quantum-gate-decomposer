
set(EXT_DIR ${PROJECT_SOURCE_DIR}/qgd_python)


add_library( qgd_N_Qubit_Decomposition_Wrapper SHARED
    ${EXT_DIR}/qgd_N_Qubit_Decomposition_Wrapper.cpp
)


ADD_DEPENDENCIES (qgd_N_Qubit_Decomposition_Wrapper qgd)

set_target_properties(
    qgd_N_Qubit_Decomposition_Wrapper
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "qgd_N_Qubit_Decomposition_Wrapper${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(qgd_N_Qubit_Decomposition_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(qgd_N_Qubit_Decomposition_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/decomposition/include
                            ${PROJECT_SOURCE_DIR}/operations/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

target_link_libraries (qgd_N_Qubit_Decomposition_Wrapper PRIVATE
                        qgd                        
                        ${PYTHON_LIBRARIES})




