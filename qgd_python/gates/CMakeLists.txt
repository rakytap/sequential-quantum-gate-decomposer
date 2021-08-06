
set(EXT_DIR ${PROJECT_SOURCE_DIR}/qgd_python/gates)


add_library( qgd_U3 MODULE
    ${EXT_DIR}/qgd_U3.cpp
)


ADD_DEPENDENCIES (qgd_U3 qgd)

target_link_libraries (qgd_U3 qgd ${PYTHON_LIBRARIES})

python_extension_module(qgd_U3)

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


set_target_properties( qgd_U3 PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_U3 LIBRARY 
         DESTINATION qgd_python/gates)

###################################################################



add_library( qgd_CNOT MODULE
    ${EXT_DIR}/qgd_CNOT.cpp
)


ADD_DEPENDENCIES (qgd_CNOT qgd)

target_link_libraries (qgd_CNOT qgd ${PYTHON_LIBRARIES})

python_extension_module(qgd_CNOT)


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

set_target_properties( qgd_CNOT PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CNOT LIBRARY 
         DESTINATION qgd_python/gates)

###################################################################



add_library( qgd_Operation_Block MODULE
    ${EXT_DIR}/qgd_Operation_Block.cpp
)


ADD_DEPENDENCIES (qgd_Operation_Block qgd)

target_link_libraries (qgd_Operation_Block qgd ${PYTHON_LIBRARIES})


python_extension_module(qgd_Operation_Block)

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


set_target_properties( qgd_Operation_Block PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_Operation_Block LIBRARY 
         DESTINATION qgd_python/gates)

