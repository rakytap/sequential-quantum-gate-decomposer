/*
Created on Fri Jun 26 14:13:26 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
*/
/*! \file Functor_Get_Gate_Products.cpp
    \brief Methods and classes for the parallelized calculation of the vector containing the products of gates (supporting TBB and OpenMP).
*/

#include "Decomposition_Base.h"
#include "Functor_Get_Operation_Products.h"
#include <tbb/parallel_for.h>


/**
@brief Calculate the list of gate gate matrices such that the i>0-th element in the result list is the product of the gates of all 0<=n<i gates from the input list and the 0th element in the result list is the identity.
@param parameters An array containing the parameters of the gates.
@param gates_it An iterator pointing to the first gate.
@param num_of_gates The number of gates involved in the calculations
@return Returns with a vector of the product matrices.
*/
std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> Decomposition_Base::get_gate_products(double* parameters, std::vector<Gate*>::iterator gates_it, int num_of_gates) {


    // construct the vector of matrix representation of the gates
    std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>> gate_mtxs(num_of_gates);

    // creating identity gate if no gates were involved in the calculations
    if (num_of_gates==0) {
        gate_mtxs.push_back( create_identity(matrix_size) );
        return gate_mtxs;
    }

    // calculate the matrices of the individual block gates
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_of_gates, 1), functor_get_operation_matrices( parameters, gates_it, &gate_mtxs, num_of_gates ));

    // calculate the gates products
    Matrix gate_product_mtx = Matrix(matrix_size, matrix_size);
    for (int idx=1; idx<num_of_gates; idx++) {
        gate_product_mtx = apply_gate(gate_mtxs[idx-1], gate_mtxs[idx] );
        gate_mtxs[idx] = gate_product_mtx;
        gates_it++;
    }

    return gate_mtxs;

}



/**
@brief Constructor of the class.
@param parameters_in An array containing the parameters of the gates.
@param gates_it_in An iterator pointing to the first gate.
@param gate_mtxs_in Pointer to a vector containing the matrix representation of the gates/gate blocks.
@param num_of_gates_in The number of gates in the vector
@return Returns with the instance of the class.
*/
functor_get_operation_matrices::functor_get_operation_matrices( double* parameters_in, std::vector<Gate*>::iterator gates_it_in, std::vector<Matrix, tbb::cache_aligned_allocator<Matrix>>* gate_mtxs_in, int num_of_gates_in ) {

    parameters = parameters_in;
    gates_it = gates_it_in;
    gate_mtxs = gate_mtxs_in;
    num_of_gates = num_of_gates_in;

}

/**
@brief Operator to calculate the matrix representation of gate labeled by i.
@param r Range of indexes labeling the gate in the vector of gates iterated by gates_it.
*/
void functor_get_operation_matrices::operator()( const tbb::blocked_range<size_t> r ) const {

    for ( size_t i = r.begin(); i!=r.end(); i++) {

        // determine the range parameters
        double* parameters_loc = parameters;
        for (size_t idx=0; idx<i; idx++) {
            Gate* gate = *(gates_it+idx);
            parameters_loc = parameters_loc + gate->get_parameter_num();
        }

        // get the matrix representation of th egate
        Gate* gate = *(gates_it+i);

        if (gate->get_type() == CNOT_OPERATION ) {
            CNOT* cnot_gate = static_cast<CNOT*>(gate);
            (*gate_mtxs)[i] = cnot_gate->get_matrix();
        }
        else if (gate->get_type() == GENERAL_OPERATION ) {
            (*gate_mtxs)[i] = gate->get_matrix();
        }
            else if (gate->get_type() == U3_OPERATION ) {
            U3* u3_gate = static_cast<U3*>(gate);
            (*gate_mtxs)[i] = u3_gate->get_matrix(parameters_loc);
        }
        else if (gate->get_type() == BLOCK_OPERATION ) {
            Gates_block* block_gate = static_cast<Gates_block*>(gate);
            (*gate_mtxs)[i] = block_gate->get_matrix(parameters_loc);
        }

    }
}
