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
/*! \file Permutation.h
    \brief Class for the representation of Permutation gate.
*/

#ifndef PERMUTATION_H
#define PERMUTATION_H

#include "Gate.h"
#include "common.h"
#include "matrix.h"
#include "logging.h"
#include "tbb/tbb.h"

class Permutation : public Gate {

protected:
    std::vector<int> pattern;

public:
    Permutation();
    Permutation(int qbit_num_in, const std::vector<int>& pattern_in);
    ~Permutation();
    Matrix get_matrix();
    Matrix get_matrix(int parallel);
    void apply_to(Matrix& input, int parallel);
    void apply_to(Matrix& input);
    void apply_to_list(std::vector<Matrix>& inputs, int parallel);
    std::vector<int> get_pattern();
    void set_pattern(const std::vector<int>& pattern_in);
    std::vector<int> get_target_qbits();
    std::vector<int> get_control_qbits();
    std::vector<int> get_involved_qubits(bool only_target = false);
    Permutation* clone();
    void reorder_qubits(std::vector<int> qbit_list);
};

#endif //PERMUTATION_H