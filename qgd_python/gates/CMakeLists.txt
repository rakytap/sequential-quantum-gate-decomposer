
set(EXT_DIR ${PROJECT_SOURCE_DIR}/qgd_python/gates)


add_library( qgd_U3 SHARED
    ${EXT_DIR}/qgd_U3.cpp
)


ADD_DEPENDENCIES (qgd_U3 qgd)

set_target_properties(
    qgd_U3
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "qgd_U3${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(qgd_U3 PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(qgd_U3 PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/operations/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

target_link_libraries (qgd_U3 PRIVATE
                        qgd                        
                        ${PYTHON_LIBRARIES})



###################################################################



add_library( qgd_CNOT SHARED
    ${EXT_DIR}/qgd_CNOT.cpp
)


ADD_DEPENDENCIES (qgd_CNOT qgd)

set_target_properties(
    qgd_CNOT
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "qgd_CNOT${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(qgd_CNOT PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(qgd_CNOT PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/operations/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

target_link_libraries (qgd_CNOT PRIVATE
                        qgd                        
                        ${PYTHON_LIBRARIES})




###################################################################



add_library( qgd_Operation_Block SHARED
    ${EXT_DIR}/qgd_Operation_Block.cpp
)


ADD_DEPENDENCIES (qgd_Operation_Block qgd)

set_target_properties(
    qgd_Operation_Block
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "qgd_Operation_Block${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(qgd_Operation_Block PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(qgd_Operation_Block PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/operations/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

target_link_libraries (qgd_Operation_Block PRIVATE
                        qgd                        
                        ${PYTHON_LIBRARIES})
