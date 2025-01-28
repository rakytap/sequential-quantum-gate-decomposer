
set(EXT_DIR ${PROJECT_SOURCE_DIR}/squander/gates)


add_library( qgd_U3_Wrapper MODULE
    ${EXT_DIR}/qgd_U3_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_U3_Wrapper qgd)

target_link_libraries (qgd_U3_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_U3_Wrapper)

# adding compile options
target_compile_options(qgd_U3_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_U3_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_U3_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_U3_Wrapper LIBRARY 
         DESTINATION squander/gates)

###################################################################



add_library( qgd_CNOT_Wrapper MODULE
    ${EXT_DIR}/qgd_CNOT_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_CNOT_Wrapper qgd)

target_link_libraries (qgd_CNOT_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_CNOT_Wrapper)


# adding compile options
target_compile_options(qgd_CNOT_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CNOT_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CNOT_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CNOT_Wrapper LIBRARY 
         DESTINATION squander/gates)

###################################################################




add_library( qgd_CZ_Wrapper MODULE
    ${EXT_DIR}/qgd_CZ_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_CZ_Wrapper qgd)

target_link_libraries (qgd_CZ_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_CZ_Wrapper)


# adding compile options
target_compile_options(qgd_CZ_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CZ_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CZ_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CZ_Wrapper LIBRARY 
         DESTINATION squander/gates)



###################################################################




add_library( qgd_CH_Wrapper MODULE
    ${EXT_DIR}/qgd_CH_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_CH_Wrapper qgd)

target_link_libraries (qgd_CH_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_CH_Wrapper)


# adding compile options
target_compile_options(qgd_CH_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CH_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})

set_target_properties( qgd_CH_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS qgd_CH_Wrapper LIBRARY 
         DESTINATION squander/gates)


###################################################################



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



add_library( qgd_SYC MODULE
    ${EXT_DIR}/qgd_SYC.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_SYC qgd)

target_link_libraries (qgd_SYC qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_SYC)

# adding compile options
target_compile_options(qgd_SYC PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_SYC PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_SYC PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_SYC LIBRARY 
         DESTINATION squander/gates)



###################################################################



add_library( qgd_CRY_Wrapper MODULE
    ${EXT_DIR}/qgd_CRY_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_CRY_Wrapper qgd)

target_link_libraries (qgd_CRY_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_CRY_Wrapper)

# adding compile options
target_compile_options(qgd_CRY_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_CRY_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_CRY_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_CRY_Wrapper LIBRARY 
         DESTINATION squander/gates)



###################################################################



add_library( qgd_RY_Wrapper MODULE
    ${EXT_DIR}/qgd_RY_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_RY_Wrapper qgd)

target_link_libraries (qgd_RY_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_RY_Wrapper)

# adding compile options
target_compile_options(qgd_RY_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_RY_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_RY_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_RY_Wrapper LIBRARY 
         DESTINATION squander/gates)





###################################################################



add_library( qgd_RZ_Wrapper MODULE
    ${EXT_DIR}/qgd_RZ_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_RZ_Wrapper qgd)

target_link_libraries (qgd_RZ_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_RZ_Wrapper)

# adding compile options
target_compile_options(qgd_RZ_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_RZ_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_RZ_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_RZ_Wrapper LIBRARY 
         DESTINATION squander/gates)



###################################################################



add_library( qgd_H_Wrapper MODULE
    ${EXT_DIR}/qgd_H_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_H_Wrapper qgd)

target_link_libraries (qgd_H_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_H_Wrapper)

# adding compile options
target_compile_options(qgd_H_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_H_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_H_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_H_Wrapper LIBRARY 
         DESTINATION squander/gates)






###################################################################



add_library( qgd_RX_Wrapper MODULE
    ${EXT_DIR}/qgd_RX_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_RX_Wrapper qgd)

target_link_libraries (qgd_RX_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_RX_Wrapper)

# adding compile options
target_compile_options(qgd_RX_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_RX_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_RX_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_RX_Wrapper LIBRARY 
         DESTINATION squander/gates)





###################################################################



add_library( qgd_X_Wrapper MODULE
    ${EXT_DIR}/qgd_X_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_X_Wrapper qgd)

target_link_libraries (qgd_X_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_X_Wrapper)

# adding compile options
target_compile_options(qgd_X_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_X_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_X_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_X_Wrapper LIBRARY 
         DESTINATION squander/gates)





###################################################################



add_library( qgd_Y_Wrapper MODULE
    ${EXT_DIR}/qgd_Y_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_Y_Wrapper qgd)

target_link_libraries (qgd_Y_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_Y_Wrapper)

# adding compile options
target_compile_options(qgd_Y_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_Y_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_Y_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_Y_Wrapper LIBRARY 
         DESTINATION squander/gates)




###################################################################



add_library( qgd_Z_Wrapper MODULE
    ${EXT_DIR}/qgd_Z_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_Z_Wrapper qgd)

target_link_libraries (qgd_Z_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_Z_Wrapper)

# adding compile options
target_compile_options(qgd_Z_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_Z_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_Z_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_Z_Wrapper LIBRARY 
         DESTINATION squander/gates)





###################################################################



add_library( qgd_SX_Wrapper MODULE
    ${EXT_DIR}/qgd_SX_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (qgd_SX_Wrapper qgd)

target_link_libraries (qgd_SX_Wrapper qgd ${BLAS_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})


python_extension_module(qgd_SX_Wrapper)

# adding compile options
target_compile_options(qgd_SX_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(qgd_SX_Wrapper PRIVATE
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_SX_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS qgd_SX_Wrapper LIBRARY 
         DESTINATION squander/gates)


