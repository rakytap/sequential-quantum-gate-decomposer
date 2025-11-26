
set(EXT_DIR ${PROJECT_SOURCE_DIR}/squander/nn)


add_library( qgd_nn_Wrapper MODULE
    ${EXT_DIR}/qgd_nn_Wrapper.cpp
    ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp        
)

# Windows/MSVC: Apply /bigobj flag directly to numpy_interface.cpp to handle large object files
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/numpy_interface.cpp
        PROPERTIES COMPILE_OPTIONS "/bigobj"
    )
endif()

ADD_DEPENDENCIES (qgd_nn_Wrapper qgd)

target_link_libraries (qgd_nn_Wrapper qgd  ${BLAS_LIBRARIES}  ${LAPACKE_LIBRARIES} ${TBB_LIB} ${TBBMALLOC_LIB})

python_extension_module(qgd_nn_Wrapper)


# adding compile options
target_compile_options(qgd_nn_Wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
    "$<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>"
)


target_include_directories(qgd_nn_Wrapper PRIVATE    
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/nn/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/decomposition/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/gates/include
                            ${PROJECT_SOURCE_DIR}/squander/src-cpp/common/include
                            ${EXTRA_INCLUDES})


set_target_properties( qgd_nn_Wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)
# Set BUILD_RPATH to prioritize conda libraries
if(DEFINED ENV{CONDA_PREFIX})
    set_target_properties(qgd_nn_Wrapper PROPERTIES
        BUILD_RPATH "${CONDA_PREFIX}/lib"
        BUILD_RPATH_USE_ORIGIN TRUE
    )
endif()




install(TARGETS qgd_nn_Wrapper 
        LIBRARY DESTINATION squander/nn
        RUNTIME DESTINATION squander/nn  # Add this for Windows
        COMPONENT python)


