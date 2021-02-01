# CMAKE to create standalone executables


# add tests to the build
add_test(decomposition_test decomposition_test ...)
add_test(custom_gate_structure_test custom_gate_structure_test ...)



# Add executable called "decomposition_test" that is built from the source files
# "decomposition_test.cpp". The extensions are automatically found.
add_executable (decomposition_test decomposition_test.cpp)
add_executable (custom_gate_structure_test custom_gate_structure_test.cpp)





# Link the executable to the qgd library. Since the qgd library has
# public include directories we will use those link directories when building
# decomposition_test
target_link_libraries (decomposition_test LINK_PUBLIC qgd)
target_link_libraries (custom_gate_structure_test LINK_PUBLIC qgd)





