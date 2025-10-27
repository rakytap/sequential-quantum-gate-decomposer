/*
Created on Fri Jun 26 14:13:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Permutation.cpp
    \brief Class for the representation of Permutation gate.
*/
#include "Permutation.h"
#include "apply_dedicated_gate_kernel_to_input.h"
#include "common.h"

Permutation::Permutation(){
    name = "Permutation";
    type = PERMUTATION_OPERATION;
    target_qbits.clear();
    control_qbits.clear();
    parameter_num = 0;
}

Permutation::Permutation(int qbit_num_in, const std::vector<int>& pattern_in) : Gate(qbit_num_in) {
    if (pattern_in.size() != qbit_num_in) {
        std::stringstream sstream;
        sstream << "Permutation: Pattern size " << pattern_in.size() << " is not equal to the number of qubits " << qbit_num_in << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }
    name = "Permutation";
    type = PERMUTATION_OPERATION;
    pattern = pattern_in;
    control_qbits.clear();
    parameter_num = 0;
    target_qbits.resize(qbit_num_in);
    for (int idx=0; idx<qbit_num_in; idx++){
        target_qbits[idx] = idx;
    }
}
Permutation::~Permutation(){
    target_qbits.clear();
    control_qbits.clear();
}

Matrix Permutation::get_matrix(){
    return get_matrix(false);
}

Matrix Permutation::get_matrix(int parallel){
    Matrix permutation_matrix = create_identity(matrix_size);
    apply_to(permutation_matrix, parallel);
    return permutation_matrix;
}

void Permutation::apply_to(Matrix& input, int parallel){
    apply_Permutation_kernel_to_input(input, pattern, matrix_size);
}
void Permutation::apply_to(Matrix& input){
    if (input.rows != matrix_size) {
        std::string err("Permutation::apply_to: Wrong input size in Permutation gate apply");
        throw err;
    }
    apply_Permutation_kernel_to_input(input, pattern, matrix_size);
}

void Permutation::apply_to_list(std::vector<Matrix>& inputs, int parallel){
    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = inputs.size();
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,inputs.size(),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to( *input, parallel );

        }

    });
}


std::vector<int> Permutation::get_target_qbits(){
    return target_qbits;
}

std::vector<int> Permutation::get_control_qbits(){
    return control_qbits;
}

std::vector<int> Permutation::get_pattern(){
    return pattern;
}

void Permutation::set_pattern(const std::vector<int>& pattern_in){
    pattern = pattern_in;
}

std::vector<int> Permutation::get_involved_qubits(bool only_target){
    std::vector<int> involved_qubits;
    for (int i = 0; i < qbit_num; i++) {
        involved_qubits.push_back(i);
    }
    return involved_qubits;
}

Permutation* Permutation::clone(){
    Permutation* ret = new Permutation(qbit_num, pattern);
    ret->set_parameter_start_idx(get_parameter_start_idx());
    ret->set_parents(parents);
    ret->set_children(children);
    return ret;
}

void Permutation::reorder_qubits(std::vector<int> qbit_list){
    Gate::reorder_qubits(qbit_list);
    std::vector<int> new_pattern(qbit_num);
    for (int idx=0; idx<qbit_num; idx++){
        new_pattern[idx] = std::find(qbit_list.begin(), qbit_list.end(), pattern[idx]) - qbit_list.begin();
    }
    pattern = new_pattern;
}