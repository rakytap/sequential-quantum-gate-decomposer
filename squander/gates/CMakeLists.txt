
set(EXT_DIR ${PROJECT_SOURCE_DIR}/squander/gates)


add_library( qgd_Circuit_Wrapper MODULE
    ${EXT_DIR}/qgd_Circuit_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_Circuit_Wrapper qgd)

target_link_libraries (qgd_Circuit_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_Circuit_Wrapper)

# adding compile options
target_compile_options(qgd_Circuit_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_Circuit_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_Circuit_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_Circuit_Wrapper LIBRARY 
         DESTINATION squander/gates)




###################################################################


add_library( gates_Wrapper MODULE
    ${EXT_DIR}/gates_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (gates_Wrapper qgd)

target_link_libraries (gates_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(gates_Wrapper)

# adding compile options
target_compile_options(gates_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(gates_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( gates_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS gates_Wrapper LIBRARY 
         DESTINATION squander/gates)

###################################################################


add_library( qgd_CROT_Wrapper MODULE
    ${EXT_DIR}/qgd_CROT_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_CROT_Wrapper qgd)

target_link_libraries (qgd_CROT_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_CROT_Wrapper)


# adding compile options
target_compile_options(qgd_CROT_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CROT_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CROT_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CROT_Wrapper LIBRARY 
         DESTINATION squander/gates)

