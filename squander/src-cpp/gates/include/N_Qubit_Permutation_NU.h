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
/*! \file UN.h
    \brief Header file for a class for the representation of general gate operations on the first qbit_num-1 qubits.
*/

#ifndef N_Qubit_Permutation_NU_H
#define N_Qubit_Permutation_NU_H

#include <vector>
#include "common.h"
#include "matrix.h"
#include "matrix_real.h"
#include "Gate.h"

/**
@brief Base class for the representation of general gate operations.
*/
class N_Qubit_Permutation_NU : public Gate { 


protected:
   
   std::vector<std::vector<int>> all_patterns;
   std::vector<double> centers;
   int n_perm;

public:

    // Constructors and Destructor
    N_Qubit_Permutation_NU();
    
    N_Qubit_Permutation_NU(int qbit_num_in);
    
    ~N_Qubit_Permutation_NU();

    // Matrix retrieval
    Matrix get_matrix(Matrix_real& parameters_mtx);
    
    Matrix get_matrix(Matrix_real& parameters_mtx, int parallel);

    // Apply gate to input
    void apply_to(Matrix_real& parameters_mtx, Matrix& input, int parallel);
    
    void apply_from_right(Matrix_real& parameters_mtx, Matrix& input);

    // Utility functions
    void reorder_qubits(std::vector<int> qbit_list);
    
    gate_type get_type();
    
    int get_qbit_num();
    
    std::vector<Matrix> apply_derivate_to( Matrix_real& parameters_mtx, Matrix& input, int parallel );

    // Permutation pattern functions
    std::vector<std::vector<int>> construct_all_possible_patterns(int qbit_num);
    
    Matrix construct_matrix_from_pattern(std::vector<int> pattern);

    // Lagrange basis functions
    double f_k(double x, int k);
    
    double f_k_derivative(double x, int k);
    
    double g_k(double x, int k);
    
    double g_k_derivative(double x, int k);

    // Matrix addition helper
    void matrix_addition(Matrix& lhs, Matrix rhs);

    // Clone function
    N_Qubit_Permutation_NU* clone();
    
    Matrix_real extract_parameters( Matrix_real& parameters );

};


#endif //OPERATION
