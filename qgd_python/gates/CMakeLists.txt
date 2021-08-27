
set(EXT_DIR ${PROJECT_SOURCE_DIR}/qgd_python/gates)


add_library( qgd_U3 MODULE
    ${EXT_DIR}/qgd_U3.cpp
)


ADD_DEPENDENCIES (qgd_U3 qgd)

target_link_libraries (qgd_U3 qgd ${BLAS_LIBRARIES})

python_extension_module(qgd_U3)

# adding compile options
target_compile_options(qgd_U3 PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_U3 PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/gates/include
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

target_link_libraries (qgd_CNOT qgd ${BLAS_LIBRARIES})

python_extension_module(qgd_CNOT)


# adding compile options
target_compile_options(qgd_CNOT PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CNOT PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CNOT PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CNOT LIBRARY 
         DESTINATION qgd_python/gates)

###################################################################




add_library( qgd_CZ MODULE
    ${EXT_DIR}/qgd_CZ.cpp
)


ADD_DEPENDENCIES (qgd_CZ qgd)

target_link_libraries (qgd_CZ qgd ${BLAS_LIBRARIES})

python_extension_module(qgd_CZ)


# adding compile options
target_compile_options(qgd_CZ PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CZ PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CZ PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CZ LIBRARY 
         DESTINATION qgd_python/gates)



###################################################################




add_library( qgd_CH MODULE
    ${EXT_DIR}/qgd_CH.cpp
)


ADD_DEPENDENCIES (qgd_CH qgd)

target_link_libraries (qgd_CH qgd ${BLAS_LIBRARIES})

python_extension_module(qgd_CH)


# adding compile options
target_compile_options(qgd_CH PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CH PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CH PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CH LIBRARY 
         DESTINATION qgd_python/gates)


###################################################################



add_library( qgd_Gates_Block MODULE
    ${EXT_DIR}/qgd_Gates_Block.cpp
)


ADD_DEPENDENCIES (qgd_Gates_Block qgd)

target_link_libraries (qgd_Gates_Block qgd ${BLAS_LIBRARIES})


python_extension_module(qgd_Gates_Block)

# adding compile options
target_compile_options(qgd_Gates_Block PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_Gates_Block PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/gates/include
                            ${PROJECT_SOURCE_DIR}/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_Gates_Block PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_Gates_Block LIBRARY 
         DESTINATION qgd_python/gates)

