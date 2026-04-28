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
/*! \file Gate.cpp
    \brief Class for the representation of general gate operations.
*/


#include "Gate.h"
#include "common.h"
#include "gate_kernel_templates.h"
#include "qgd_math.h"
#include <algorithm>
#include <cmath>
#include <sstream>

#ifdef USE_AVX 
#include "apply_kernel_to_input_AVX.h"
#include "apply_kernel_to_state_vector_input_AVX.h"
#include "apply_large_kernel_to_input_AVX.h"
#endif

#include "apply_kernel_to_input.h"
#include "apply_kernel_to_state_vector_input.h"
#include "apply_large_kernel_to_input.h"
#include "apply_dedicated_gate_kernel_to_input.h"

/**
@brief Deafult constructor of the class.
@return An instance of the class
*/
Gate::Gate() {

    // A string labeling the gate operation
    name = "Gate";
    // number of qubits spanning the matrix of the operation
    qbit_num = -1;
    // The size N of the NxN matrix associated with the operations.
    matrix_size = -1;
    // The type of the operation (see enumeration gate_type)
    type = GENERAL_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // Vector-based qubit storage
    target_qbits.clear();
    control_qbits.clear();
    // the number of free parameters of the operation
    parameter_num = 0;
    // the index in the parameter array (corrensponding to the encapsulated circuit) where the gate parameters begin (if gates are placed into a circuit a single parameter array is used to execute the whole circuit)
    parameter_start_idx = 0;
}



/**
@brief Constructor of the class.
@param qbit_num_in The number of qubits spanning the unitaries
@return An instance of the class
*/
Gate::Gate(int qbit_num_in) {



    // A string labeling the gate operation
    name = "Gate";
    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = GENERAL_OPERATION;
    // The index of the qubit on which the operation acts (target_qbit >= 0)
    target_qbit = -1;
    // The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
    control_qbit = -1;
    // Vector-based qubit storage
    target_qbits.clear();
    control_qbits.clear();
    // The number of parameters
    parameter_num = 0;
    // the index in the parameter array (corrensponding to the encapsulated circuit) where the gate parameters begin (if gates are placed into a circuit a single parameter array is used to execute the whole circuit)
    parameter_start_idx = 0;
}


/**
@brief Constructor of the class with vector-based qubit specification.
@param qbit_num_in The number of qubits spanning the unitaries
@param target_qbits_in Vector of target qubit indices
@param control_qbits_in Vector of control qubit indices
@return An instance of the class
*/
Gate::Gate(int qbit_num_in, const std::vector<int>& target_qbits_in, const std::vector<int>& control_qbits_in) {



    // A string labeling the gate operation
    name = "Gate";
    // number of qubits spanning the matrix of the operation
    qbit_num = qbit_num_in;
    // the size of the matrix
    matrix_size = Power_of_2(qbit_num);
    // A string describing the type of the operation
    type = GENERAL_OPERATION;
    // The number of parameters
    parameter_num = 0;
    // the index in the parameter array (corrensponding to the encapsulated circuit) where the gate parameters begin (if gates are placed into a circuit a single parameter array is used to execute the whole circuit)
    parameter_start_idx = 0;

    // Validate target qubits
    for (int tq : target_qbits_in) {
        if (tq >= qbit_num_in) {
            std::stringstream sstream;
            sstream << "Gate: Target qubit index " << tq << " is larger than or equal to the number of qubits " << qbit_num_in << std::endl;
            print(sstream, 0);
            throw sstream.str();
        }
    }

    // Validate control qubits
    for (int cq : control_qbits_in) {
        if (cq >= qbit_num_in) {
            std::stringstream sstream;
            sstream << "Gate: Control qubit index " << cq << " is larger than or equal to the number of qubits " << qbit_num_in << std::endl;
            print(sstream, 0);
            throw sstream.str();
        }
    }

    target_qbits = target_qbits_in;
    control_qbits = control_qbits_in;
    std::sort(target_qbits.begin(), target_qbits.end());
    std::sort(control_qbits.begin(), control_qbits.end());
    // Set the legacy single-qubit members to first elements for backward compatibility
    target_qbit = target_qbits.empty() ? -1 : target_qbits[0];
    control_qbit = control_qbits.empty() ? -1 : control_qbits[0];
}


/**
@brief Destructor of the class
*/
Gate::~Gate() {
}

/**
@brief Set the number of qubits spanning the matrix of the operation
@param qbit_num_in The number of qubits spanning the matrix
*/
void Gate::set_qbit_num( int qbit_num_in ) {



    if ( qbit_num_in <= target_qbit || qbit_num_in <= control_qbit ) {
        std::string err("Gate::set_qbit_num: The number of qbits is too small, conflicting with either target_qbit os control_qbit"); 
        throw err;   
    }


    // setting the number of qubits
    qbit_num = qbit_num_in;

    // update the size of the matrix
    matrix_size = Power_of_2(qbit_num);

}

/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
Gate::get_matrix( Matrix_real& parameters, int parallel ) {

    Matrix gate_matrix = create_identity(matrix_size);
    apply_to(parameters, gate_matrix, parallel);
    return gate_matrix;

}

Matrix
Gate::get_matrix( Matrix_real& parameters ) {
    return get_matrix(parameters, 0);
}

Matrix
Gate::get_matrix() {
    Matrix gate_matrix = create_identity(matrix_size);
    apply_to(gate_matrix, 0);
    return gate_matrix;
}

Matrix
Gate::get_matrix( int parallel ) {
    Matrix gate_matrix = create_identity(matrix_size);
    apply_to(gate_matrix, parallel);
    return gate_matrix;
}


/**
@brief Float32 overload: retrieve the gate matrix.
Base implementation creates a float32 identity and applies the gate with parallel=0
to avoid any broken AVX path.  Derived classes should override this.
@param parameters Float32 parameter array
@param parallel Unused in the base implementation (always uses sequential path)
@return Returns with a float32 matrix of the gate
*/
Matrix_float
Gate::get_matrix( Matrix_real_float& parameters, int parallel ) {

    Matrix_float gate_matrix = create_identity_float(matrix_size);
    apply_to(parameters, gate_matrix, parallel);
    return gate_matrix;

}

/**
@brief Call to apply the gate on a list of inputs
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to_list( std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = static_cast<int>(inputs.size());
    }
    else {
        work_batch = 1;
    }


    tbb::parallel_for( tbb::blocked_range<int>(0,static_cast<int>(inputs.size()),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 

            Matrix* input = &inputs[idx];

            apply_to( *input, parallel );

        }

    });




}



/**
@brief Abstract function to be overriden in derived classes to be used to transform a list of inputs upon a parametric gate operation
@param parameter_mtx An array conatining the parameters of the gate
@param inputs The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to_list( Matrix_real& parameters_mtx, std::vector<Matrix>& inputs, int parallel ) {

    int work_batch = ( parallel == 0 ) ? static_cast<int>(inputs.size()) : 1;

    tbb::parallel_for( tbb::blocked_range<int>(0,static_cast<int>(inputs.size()),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) {
            apply_to( parameters_mtx, inputs[idx], parallel );
        }
    });

}


/**
@brief Float32 overload: apply gate to a list of float32 inputs without parameters.
@param inputs Float32 input matrices/states
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void
Gate::apply_to_list( std::vector<Matrix_float>& inputs, int parallel ) {

    int work_batch = 1;
    if ( parallel == 0 ) {
        work_batch = static_cast<int>(inputs.size());
    }
    else {
        work_batch = 1;
    }

    tbb::parallel_for( tbb::blocked_range<int>(0,static_cast<int>(inputs.size()),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) {
            Matrix_float* input = &inputs[idx];
            apply_to( *input, parallel );
        }
    });

}


/**
@brief Float32 overload: apply gate to a list of float32 inputs with float32 parameters.
@param parameters_mtx Float32 parameter matrix
@param inputs Float32 input matrices/states
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void
Gate::apply_to_list( Matrix_real_float& parameters_mtx, std::vector<Matrix_float>& inputs, int parallel ) {

    int work_batch = ( parallel == 0 ) ? static_cast<int>(inputs.size()) : 1;

    tbb::parallel_for( tbb::blocked_range<int>(0,static_cast<int>(inputs.size()),work_batch), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) {
            apply_to( parameters_mtx, inputs[idx], parallel );
        }
    });

}



/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to( Matrix& input, int parallel ) {

   if (input.rows != matrix_size ) {
        std::stringstream sstream;
        sstream << "Gate::apply_to: Wrong matrix size in Gate gate apply."
                << " name=" << name
                << " type=" << type
                << " qbit_num=" << qbit_num
                << " matrix_size=" << matrix_size
                << " input.rows=" << input.rows
                << " input.cols=" << input.cols << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    Matrix kernel;
    if (type != GENERAL_OPERATION && type != SWAP_OPERATION && type != CSWAP_OPERATION && type != CCX_OPERATION) {
        Matrix_real empty_params(0, 0);
        kernel = gate_kernel(empty_params);
    }

    apply_kernel_to(kernel, input, false, parallel);
}


void
Gate::apply_to( Matrix_float& input, int parallel ) {

    if (input.rows != matrix_size) {
        std::string err("Gate::apply_to(Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix_float kernel;
    if (type != GENERAL_OPERATION && type != SWAP_OPERATION && type != CSWAP_OPERATION && type != CCX_OPERATION) {
        Matrix_real_float empty_params(0, 0);
        kernel = gate_kernel(empty_params);
    }

    apply_kernel_to(kernel, input, false, parallel);
}


/**
@brief Abstract function to be overriden in derived classes to be used to transform an input upon a parametric gate operation
@param parameter_mtx An array conatining the parameters
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to( Matrix_real& parameter_mtx, Matrix& input, int parallel ) {

    if (input.rows != matrix_size) {
        std::stringstream sstream;
        sstream << "Gate::apply_to(Matrix_real&, Matrix&): Wrong matrix size in gate apply."
                << " name=" << name
                << " type=" << type
                << " qbit_num=" << qbit_num
                << " matrix_size=" << matrix_size
                << " input.rows=" << input.rows
                << " input.cols=" << input.cols
                << " parameter_num=" << parameter_num
                << " provided_params=" << parameter_mtx.size() << std::endl;
        print(sstream, 0);
        throw sstream.str();
    }

    // For zero-parameter gates delegate to the no-param virtual.
    if (parameter_num == 0) {
        apply_to(input, parallel);
        return;
    }

    Matrix_real precomputed_sincos = compute_precomputed_sincos(parameter_mtx);
    apply_to_inner(parameter_mtx, precomputed_sincos, input, parallel);
}


void
Gate::apply_to( Matrix_real_float& parameter_mtx, Matrix_float& input, int parallel ) {

    if (input.rows != matrix_size) {
        std::string err("Gate::apply_to(Matrix_real_float&, Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    // For zero-parameter gates delegate to the no-parameter overload.
    if (parameter_num == 0) {
        apply_to(input, parallel);
        return;
    }

    Matrix_real_float precomputed_sincos = compute_precomputed_sincos(parameter_mtx);
    apply_to_inner(parameter_mtx, precomputed_sincos, input, parallel);
}


void
Gate::apply_to_inner( Matrix_real& parameter_mtx, const Matrix_real& precomputed_sincos, Matrix& input, int parallel ) {

    // Adaptive gate keeps custom branching logic in its own apply_to overload.
    if (type == ADAPTIVE_OPERATION) {
        apply_to(parameter_mtx, input, parallel);
        return;
    }

    if (parameter_num == 0) {
        apply_to(input, parallel);
        return;
    }

    Matrix u3 = gate_kernel(precomputed_sincos);
    Matrix u3_aux = inverse_gate_kernel(precomputed_sincos);
    apply_kernel_to(u3, input, false, parallel, &u3_aux);
}


void
Gate::apply_to_inner( Matrix_real_float& parameter_mtx, const Matrix_real_float& precomputed_sincos, Matrix_float& input, int parallel ) {

    // Adaptive gate keeps custom branching logic in its own apply_to overload.
    if (type == ADAPTIVE_OPERATION) {
        apply_to(parameter_mtx, input, parallel);
        return;
    }

    if (parameter_num == 0) {
        apply_to(input, parallel);
        return;
    }

    Matrix_float u3 = gate_kernel(precomputed_sincos);
    Matrix_float u3_aux = inverse_gate_kernel(precomputed_sincos);
    apply_kernel_to(u3, input, false, parallel, &u3_aux);
}


void
Gate::apply_to( Matrix_any& input, int parallel ) {

    if (input.is_float64()) {
        apply_to(input.as_float64(), parallel);
        return;
    }

    apply_to(input.as_float32(), parallel);
}


void
Gate::apply_to( Matrix_real_any& parameter_mtx, Matrix_any& input, int parallel ) {

    if (parameter_mtx.is_float64() && input.is_float64()) {
        apply_to(parameter_mtx.as_float64(), input.as_float64(), parallel);
        return;
    }

    if (parameter_mtx.is_float32() && input.is_float32()) {
        apply_to(parameter_mtx.as_float32(), input.as_float32(), parallel);
        return;
    }

    std::string err("Gate::apply_to(Matrix_real_any&, Matrix_any&): precision mismatch between parameters and input");
    throw err;
}



/**
@brief Call to evaluate the derivate of the circuit on an inout with respect to all of the free parameters.
@param parameters An array of the input parameters.
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP (NOT IMPLEMENTED YET) and 2 for parallel with TBB (optional)
*/
std::vector<Matrix> 
Gate::apply_derivate_to( Matrix_real& parameters_mtx_in, Matrix& input, int parallel ) {

    const int parameter_count = get_parameter_num();
    std::vector<Matrix> ret;
    ret.reserve(parameter_count);

    if (parameter_count <= 0) {
        return ret;
    }

    Matrix_real precomputed_sincos = precompute_sincos(parameters_mtx_in);
    for (int param_idx = 0; param_idx < parameter_count; ++param_idx) {
        Matrix u3 = derivative_kernel(precomputed_sincos, param_idx);
        Matrix u3_aux = derivative_aux_kernel(precomputed_sincos, param_idx);

        Matrix res = input.copy();
        if (u3_aux.size() > 0) {
            apply_kernel_to(u3, res, true, parallel, &u3_aux);
        } else {
            apply_kernel_to(u3, res, true, parallel);
        }
        ret.push_back(std::move(res));
    }

    return ret;
}


std::vector<Matrix_float>
Gate::apply_derivate_to( Matrix_real_float& parameters_mtx_in, Matrix_float& input, int parallel ) {

    const int parameter_count = get_parameter_num();
    std::vector<Matrix_float> ret;
    ret.reserve(parameter_count);

    if (parameter_count <= 0) {
        return ret;
    }

    Matrix_real_float precomputed_sincos = precompute_sincos(parameters_mtx_in);
    for (int param_idx = 0; param_idx < parameter_count; ++param_idx) {
        Matrix_float u3 = derivative_kernel(precomputed_sincos, param_idx);
        Matrix_float u3_aux = derivative_aux_kernel(precomputed_sincos, param_idx);

        Matrix_float res = input.copy();
        if (u3_aux.size() > 0) {
            apply_kernel_to(u3, res, true, parallel, &u3_aux);
        } else {
            apply_kernel_to(u3, res, true, parallel);
        }
        ret.push_back(std::move(res));
    }

    return ret;
}


std::vector<Matrix>
Gate::apply_to_combined( Matrix_real& parameters_mtx_in, Matrix& input, int parallel ) {

    Matrix_real precomputed_sincos = compute_precomputed_sincos(parameters_mtx_in);
    return apply_to_combined_inner(parameters_mtx_in, precomputed_sincos, input, parallel);
}


std::vector<Matrix>
Gate::apply_to_combined_inner( Matrix_real& parameters_mtx_in, const Matrix_real& precomputed_sincos, Matrix& input, int parallel ) {

    std::vector<Matrix> ret;
    ret.reserve(get_parameter_num() + 1);

    Matrix applied = input.copy();
    apply_to_inner(parameters_mtx_in, precomputed_sincos, applied, parallel);
    ret.push_back(std::move(applied));

    std::vector<Matrix> derivs = apply_derivate_to(parameters_mtx_in, input, parallel);
    for (size_t idx = 0; idx < derivs.size(); ++idx) {
        ret.push_back(std::move(derivs[idx]));
    }

    return ret;
}


std::vector<Matrix_float>
Gate::apply_to_combined( Matrix_real_float& parameters_mtx_in, Matrix_float& input, int parallel ) {

    Matrix_real_float precomputed_sincos = compute_precomputed_sincos(parameters_mtx_in);
    return apply_to_combined_inner(parameters_mtx_in, precomputed_sincos, input, parallel);
}


std::vector<Matrix_float>
Gate::apply_to_combined_inner( Matrix_real_float& parameters_mtx_in, const Matrix_real_float& precomputed_sincos, Matrix_float& input, int parallel ) {

    std::vector<Matrix_float> ret;
    ret.reserve(get_parameter_num() + 1);

    Matrix_float applied = input.copy();
    apply_to_inner(parameters_mtx_in, precomputed_sincos, applied, parallel);
    ret.push_back(std::move(applied));

    std::vector<Matrix_float> derivs = apply_derivate_to(parameters_mtx_in, input, parallel);
    for (size_t idx = 0; idx < derivs.size(); ++idx) {
        ret.push_back(std::move(derivs[idx]));
    }

    return ret;
}



/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param input The input array on which the gate is applied
*/
void 
Gate::apply_from_right( Matrix& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix empty_kernel;
    apply_kernel_from_right(empty_kernel, input);

}


void
Gate::apply_from_right( Matrix_float& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix_float empty_kernel;
    apply_kernel_from_right(empty_kernel, input);
}

/**
@brief Apply the gate on input from the right (Matrix_real version)
@param parameter_mtx The gate parameters.
@param input The input array on which the gate is applied
*/
void 
Gate::apply_from_right( Matrix_real& parameter_mtx, Matrix& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix_real&, Matrix&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix_real precomputed_sincos = compute_precomputed_sincos(parameter_mtx);
    apply_from_right_inner(parameter_mtx, precomputed_sincos, input);

}


void
Gate::apply_from_right( Matrix_real_float& parameter_mtx, Matrix_float& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix_real_float&, Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix_real_float precomputed_sincos = compute_precomputed_sincos(parameter_mtx);
    apply_from_right_inner(parameter_mtx, precomputed_sincos, input);
}


void
Gate::apply_from_right_inner( Matrix_real& parameter_mtx, const Matrix_real& precomputed_sincos, Matrix& input ) {

    // Adaptive gate keeps custom branching logic in its own apply_from_right overload.
    if (type == ADAPTIVE_OPERATION) {
        apply_from_right(parameter_mtx, input);
        return;
    }

    Matrix kernel = gate_kernel(precomputed_sincos);
    Matrix inv_kernel = inverse_gate_kernel(precomputed_sincos);
    apply_kernel_from_right(kernel, input, &inv_kernel);
}


void
Gate::apply_from_right_inner( Matrix_real_float& parameter_mtx, const Matrix_real_float& precomputed_sincos, Matrix_float& input ) {

    // Adaptive gate keeps custom branching logic in its own apply_from_right overload.
    if (type == ADAPTIVE_OPERATION) {
        apply_from_right(parameter_mtx, input);
        return;
    }

    Matrix_float kernel = gate_kernel(precomputed_sincos);
    Matrix_float inv_kernel = inverse_gate_kernel(precomputed_sincos);
    apply_kernel_from_right(kernel, input, &inv_kernel);
}


Matrix_real
Gate::compute_precomputed_sincos(const Matrix_real& parameters) const {

    return precompute_sincos(parameters);
}


Matrix_real_float
Gate::compute_precomputed_sincos(const Matrix_real_float& parameters) const {

    return precompute_sincos(parameters);
}


/**
@brief Call to set the stored matrix in the operation.
@param input The operation matrix to be stored. The matrix is stored by attribute matrix_alloc.
@return Returns with 0 on success.
*/
void
Gate::set_matrix( Matrix input ) {
    matrix_alloc = input;
}


/**
@brief Call to set the control qubit for the gate operation
@param control_qbit_in The control qubit. Should be: 0 <= control_qbit_in < qbit_num
*/
void Gate::set_control_qbit(int control_qbit_in){

    if ( control_qbit_in >= qbit_num ) {
        std::string err("Gate::set_target_qbit: Wrong value of the control qbit: of out of the range given by qbit_num");
        throw err;
    }

    control_qbit = control_qbit_in;

    // Synchronize with vector storage
    if (control_qbits.empty()) {
        control_qbits.push_back(control_qbit_in);
    } else {
        control_qbits[0] = control_qbit_in;
    }
}


/**
@brief Call to set the target qubit for the gate operation
@param target_qbit_in The target qubit on which the gate is applied. Should be: 0 <= target_qbit_in < qbit_num
*/
void Gate::set_target_qbit(int target_qbit_in){

    if ( target_qbit_in >= qbit_num  ) {
        std::string err("Gate::set_target_qbit: Wrong value of the target qbit: out of the range given by qbit_num");
        throw err;
    }

    target_qbit = target_qbit_in;

    // Synchronize with vector storage
    if (target_qbits.empty()) {
        target_qbits.push_back(target_qbit_in);
    } else {
        target_qbits[0] = target_qbit_in;
    }
}

/**
@brief Call to set the control qubits for the gate operation
@param control_qbits_in Vector of control qubit indices
*/
void Gate::set_control_qbits(const std::vector<int>& control_qbits_in) {
    // Validate control qubits
    for (int cq : control_qbits_in) {
        if (cq >= qbit_num) {
            std::stringstream sstream;
            sstream << "Gate::set_control_qbits: Control qubit index " << cq << " is out of range" << std::endl;
            print(sstream, 0);
            throw sstream.str();
        }
    }

    control_qbits = control_qbits_in;
    control_qbit = control_qbits.empty() ? -1 : control_qbits[0];
}

/**
@brief Call to set the target qubits for the gate operation
@param target_qbits_in Vector of target qubit indices
*/
void Gate::set_target_qbits(const std::vector<int>& target_qbits_in) {
    // Validate target qubits
    for (int tq : target_qbits_in) {
        if (tq >= qbit_num) {
            std::stringstream sstream;
            sstream << "Gate::set_target_qbits: Target qubit index " << tq << " is out of range" << std::endl;
            print(sstream, 0);
            throw sstream.str();
        }
    }

    target_qbits = target_qbits_in;
    target_qbit = target_qbits.empty() ? -1 : target_qbits[0];
}

/**
@brief Call to get the vector of control qubits
@return Returns vector of control qubit indices
*/
std::vector<int> Gate::get_control_qbits() const {
    return control_qbits;
}

/**
@brief Call to get the vector of target qubits
@return Returns vector of target qubit indices
*/
std::vector<int> Gate::get_target_qbits() const {
    return target_qbits;
}

/**
@brief Call to reorder the qubits in the matrix of the operation
@param qbit_list The reordered list of qubits spanning the matrix
*/
void Gate::reorder_qubits( std::vector<int> qbit_list ) {

    // check the number of qubits
    if ((int)qbit_list.size() != qbit_num ) {
        std::string err("Gate::reorder_qubits: Wrong number of qubits.");
        throw err;
    }

    int control_qbit_new = control_qbit;
    int target_qbit_new = target_qbit;

    // setting the new value for the target qubit
    for (int idx=0; idx<qbit_num; idx++) {
        if (target_qbit == qbit_list[idx]) {
            target_qbit_new = idx;
        }
        if (control_qbit == qbit_list[idx]) {
            control_qbit_new = idx;
        }
    }

    control_qbit = control_qbit_new;
    target_qbit = target_qbit_new;

    // Reorder control qubits vector
    for (size_t i = 0; i < control_qbits.size(); i++) {
        int old_qbit = control_qbits[i];
        for (int idx = 0; idx < qbit_num; idx++) {
            if (old_qbit == qbit_list[idx]) {
                control_qbits[i] = idx;
                break;
            }
        }
    }

    // Reorder target qubits vector
    for (size_t i = 0; i < target_qbits.size(); i++) {
        int old_qbit = target_qbits[i];
        for (int idx = 0; idx < qbit_num; idx++) {
            if (old_qbit == qbit_list[idx]) {
                target_qbits[i] = idx;
                break;
            }
        }
    }
}


/**
@brief Call to get the index of the target qubit
@return Return with the index of the target qubit (return with -1 if target qubit was not set)
*/
int Gate::get_target_qbit() {
    return target_qbit;
}

/**
@brief Call to get the index of the control qubit
@return Return with the index of the control qubit (return with -1 if control qubit was not set)
*/
int Gate::get_control_qbit()  {
    return control_qbit;
}

/**
@brief Call to get the qubits involved in the gate operation.
@return Return with a list of the involved qubits
*/
std::vector<int> Gate::get_involved_qubits(bool only_target) {

    std::vector<int> involved_qbits;

    // Use vector storage if available, otherwise fall back to single qubits
    if (!target_qbits.empty()) {
        involved_qbits.insert(involved_qbits.end(), target_qbits.begin(), target_qbits.end());
    } else if (target_qbit != -1) {
        involved_qbits.push_back(target_qbit);
    }

    if (!only_target) {
        if (!control_qbits.empty()) {
            involved_qbits.insert(involved_qbits.end(), control_qbits.begin(), control_qbits.end());
        } else if (control_qbit != -1) {
            involved_qbits.push_back(control_qbit);
        }
    }

    return involved_qbits;

}


/**
@brief Call to add a parent gate to the current gate 
@param parent The parent gate of the current gate.
*/
void Gate::add_parent( Gate* parent ) {

    // check if parent already present in th elist of parents
    if ( std::count(parents.begin(), parents.end(), parent) > 0 ) {
        return;
    }
    
    parents.push_back( parent );

}



/**
@brief Call to add a child gate to the current gate 
@param child The parent gate of the current gate.
*/
void Gate::add_child( Gate* child ) {

    // check if parent already present in th elist of parents
    if ( std::count(children.begin(), children.end(), child) > 0 ) {
        return;
    }
    
    children.push_back( child );

}


/**
@brief Call to erase data on children.
*/
void Gate::clear_children() {

    children.clear();

}


/**
@brief Call to erase data on parents.
*/
void Gate::clear_parents() {

    parents.clear();

}



/**
@brief Call to get the parents of the current gate
@return Returns with the list of the parents
*/
std::vector<Gate*> Gate::get_parents() {

    return parents;

}


/**
@brief Call to get the children of the current gate
@return Returns with the list of the children
*/
std::vector<Gate*> Gate::get_children() {

    return children;

}



/**
@brief Call to get the number of free parameters
@return Return with the number of the free parameters
*/
int Gate::get_parameter_num() {
    return parameter_num;
}


/**
@brief Call to get the type of the operation
@return Return with the type of the operation (see gate_type for details)
*/
gate_type Gate::get_type() {
    return type;
}


/**
@brief Call to get the number of qubits composing the unitary
@return Return with the number of qubits composing the unitary
*/
int Gate::get_qbit_num() {
    return qbit_num;
}


/**
@brief Call to create a clone of the present class
@return Return with a pointer pointing to the cloned object
*/
Gate* Gate::clone() {

    Gate* ret = new Gate( qbit_num );
    ret->set_matrix( matrix_alloc );
    
    ret->set_parameter_start_idx( get_parameter_start_idx() );
    ret->set_parents( parents );
    ret->set_children( children );

    return ret;
}



Matrix
Gate::derivative_kernel(const Matrix_real& precomputed_sincos, int param_idx) {

    if (param_idx < 0 || param_idx >= get_parameter_num() || precomputed_sincos.cols < 2) {
        return Matrix();
    }

    Matrix_real shifted = precomputed_sincos.copy();
    const int offset = param_idx * shifted.stride;
    const double sin_theta = shifted[offset + 0];
    const double cos_theta = shifted[offset + 1];

    // sin(theta + pi/2) = cos(theta), cos(theta + pi/2) = -sin(theta)
    shifted[offset + 0] = cos_theta;
    shifted[offset + 1] = -sin_theta;

    return gate_kernel(shifted);
}


Matrix_float
Gate::derivative_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {

    if (param_idx < 0 || param_idx >= get_parameter_num() || precomputed_sincos.cols < 2) {
        return Matrix_float();
    }

    Matrix_real_float shifted = precomputed_sincos.copy();
    const int offset = param_idx * shifted.stride;
    const float sin_theta = shifted[offset + 0];
    const float cos_theta = shifted[offset + 1];

    // sin(theta + pi/2) = cos(theta), cos(theta + pi/2) = -sin(theta)
    shifted[offset + 0] = cos_theta;
    shifted[offset + 1] = -sin_theta;

    return gate_kernel(shifted);
}


Matrix
Gate::derivative_aux_kernel(const Matrix_real& precomputed_sincos, int param_idx) {
    (void)precomputed_sincos;
    (void)param_idx;
    return Matrix();
}


Matrix_float
Gate::derivative_aux_kernel(const Matrix_real_float& precomputed_sincos, int param_idx) {
    (void)precomputed_sincos;
    (void)param_idx;
    return Matrix_float();
}


Matrix_real
Gate::precompute_sincos(const Matrix_real& parameters) const {

    if (parameter_num <= 0) {
        return Matrix_real(0, 0);
    }

    if ((int)parameters.size() < parameter_num) {
        std::string err(name + "::precompute_sincos(double): not enough parameters supplied.");
        throw err;
    }

    Matrix_real sincos(parameter_num, 2);
    for (int idx = 0; idx < parameter_num; ++idx) {
        double sin_theta;
        double cos_theta;
        qgd_sincos<double>((double)parameters[idx], &sin_theta, &cos_theta);

        const int offset = idx * sincos.stride;
        sincos[offset + 0] = sin_theta;
        sincos[offset + 1] = cos_theta;
    }

    return sincos;
}


Matrix_real_float
Gate::precompute_sincos(const Matrix_real_float& parameters) const {

    if (parameter_num <= 0) {
        return Matrix_real_float(0, 0);
    }

    if ((int)parameters.size() < parameter_num) {
        std::string err(name + "::precompute_sincos(float): not enough parameters supplied.");
        throw err;
    }

    Matrix_real_float sincos(parameter_num, 2);
    for (int idx = 0; idx < parameter_num; ++idx) {
        float sin_theta;
        float cos_theta;
        qgd_sincos<float>((float)parameters[idx], &sin_theta, &cos_theta);

        const int offset = idx * sincos.stride;
        sincos[offset + 0] = sin_theta;
        sincos[offset + 1] = cos_theta;
    }

    return sincos;
}



/**
@brief Call to apply the gate kernel on the input state or unitary with optional AVX support
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise (optional)
@param deriv Set true to apply parallel kernels, false otherwise (optional)
@param parallel Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_kernel_to(Matrix& u3_1qbit, Matrix& input, bool deriv, int parallel, const Matrix* alt_kernel) {

    (void)deriv;

    if (type == SWAP_OPERATION || type == CSWAP_OPERATION) {
        switch (parallel) {
            case 0:
                apply_SWAP_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
            case 1:
                apply_SWAP_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
            case 2:
                apply_SWAP_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
            default:
                apply_SWAP_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
        }
        return;
    }

    if (type == CCX_OPERATION) {
        switch (parallel) {
            case 0:
                apply_X_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
            case 1:
                apply_X_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
            case 2:
                apply_X_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
            default:
                apply_X_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
        }
        return;
    }

    if (type == SYC_OPERATION) {
        switch (parallel) {
            case 0:
                apply_SYC_kernel_to_input(input, target_qbit, control_qbit, matrix_size); break;
            case 1:
                apply_SYC_kernel_to_input_omp(input, target_qbit, control_qbit, matrix_size); break;
            case 2:
                apply_SYC_kernel_to_input_tbb(input, target_qbit, control_qbit, matrix_size); break;
            default:
                apply_SYC_kernel_to_input(input, target_qbit, control_qbit, matrix_size); break;
        }
        return;
    }

    if (type == CROT_OPERATION && alt_kernel != nullptr) {
        Matrix branch0 = alt_kernel->copy();
        Matrix branch1 = u3_1qbit.copy();
#ifdef USE_AVX
        if (parallel) {
            apply_crot_kernel_to_matrix_input_AVX_parallel(branch0, branch1, input, target_qbit, control_qbit, input.rows);
        } else {
            apply_crot_kernel_to_matrix_input_AVX(branch0, branch1, input, target_qbit, control_qbit, input.rows);
        }
#else
        apply_crot_kernel_to_matrix_input(branch0, branch1, input, target_qbit, control_qbit, input.rows);
#endif
        return;
    }

    if (type == GENERAL_OPERATION) {
        const std::vector<int> involved_qbits = get_involved_qubits();
        const bool has_any_control = (control_qbit >= 0) || !control_qbits.empty();
        const bool is_state_vector = (input.cols == 1);
        const int involved_qbit_num = static_cast<int>(involved_qbits.size());
        const bool has_full_matrix = matrix_alloc.rows == matrix_size && matrix_alloc.cols == matrix_size;
        const bool has_local_matrix = involved_qbit_num > 0
            && matrix_alloc.rows == Power_of_2(involved_qbit_num)
            && matrix_alloc.cols == Power_of_2(involved_qbit_num);
        const bool can_use_large_kernel = !has_any_control
            && ((is_state_vector && involved_qbit_num >= 2 && involved_qbit_num <= 5)
                || (!is_state_vector && involved_qbit_num == 2));

        if (has_full_matrix) {
            if (can_use_large_kernel) {
                switch (parallel) {
                    case 0:
                        apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
                        break;
                    case 1:
#ifdef USE_AVX
                        apply_large_kernel_to_input_AVX_OpenMP(matrix_alloc, input, involved_qbits, matrix_size);
#else
                        apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
#endif
                        break;
                    case 2:
#ifdef USE_AVX
                        apply_large_kernel_to_input_AVX_TBB(matrix_alloc, input, involved_qbits, matrix_size);
#else
                        apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
#endif
                        break;
                    default:
                        apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
                }
            }
            else {
                Matrix transformed = dot(matrix_alloc, input);
                memcpy(input.get_data(), transformed.get_data(), transformed.size() * sizeof(QGD_Complex16));
            }
            return;
        }

        if (has_local_matrix && matrix_alloc.rows == 2 && matrix_alloc.cols == 2 && involved_qbit_num == 1) {
            u3_1qbit = matrix_alloc;
        }
        else if (has_local_matrix && can_use_large_kernel) {
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
                    break;
                case 1:
#ifdef USE_AVX
                    apply_large_kernel_to_input_AVX_OpenMP(matrix_alloc, input, involved_qbits, matrix_size);
#else
                    apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
#endif
                    break;
                case 2:
#ifdef USE_AVX
                    apply_large_kernel_to_input_AVX_TBB(matrix_alloc, input, involved_qbits, matrix_size);
#else
                    apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
#endif
                    break;
                default:
                    apply_large_kernel_to_input(matrix_alloc, input, involved_qbits, matrix_size);
            }
            return;
        }
        else {
            std::string err("Gate::apply_kernel_to(Matrix&): unsupported GENERAL_OPERATION dispatch for stored matrix size.");
            throw err;
        }
    }

    if (u3_1qbit.rows != 2 || u3_1qbit.cols != 2) {
        const std::vector<int> involved_qbits = get_involved_qubits();
        const bool has_any_control = (control_qbit >= 0) || !control_qbits.empty();
        const bool is_state_vector = (input.cols == 1);
        const int involved_qbit_num = static_cast<int>(involved_qbits.size());
        const bool can_use_large_kernel = !has_any_control
            && ((is_state_vector && involved_qbit_num >= 2 && involved_qbit_num <= 5)
                || (!is_state_vector && involved_qbit_num == 2));

        if (can_use_large_kernel) {
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input(u3_1qbit, input, involved_qbits, matrix_size);
                    break;
                case 1:
#ifdef USE_AVX
                    apply_large_kernel_to_input_AVX_OpenMP(u3_1qbit, input, involved_qbits, matrix_size);
#else
                    apply_large_kernel_to_input(u3_1qbit, input, involved_qbits, matrix_size);
#endif
                    break;
                case 2:
#ifdef USE_AVX
                    apply_large_kernel_to_input_AVX_TBB(u3_1qbit, input, involved_qbits, matrix_size);
#else
                    apply_large_kernel_to_input(u3_1qbit, input, involved_qbits, matrix_size);
#endif
                    break;
                default:
                    apply_large_kernel_to_input(u3_1qbit, input, involved_qbits, matrix_size);
            }
            return;
        }

        if (u3_1qbit.rows == matrix_size && u3_1qbit.cols == matrix_size) {
            Matrix transformed = dot(u3_1qbit, input);
            memcpy(input.get_data(), transformed.get_data(), transformed.size() * sizeof(QGD_Complex16));
            return;
        }

        std::string err("Gate::apply_kernel_to(Matrix&): unsupported non-2x2 kernel dispatch for this configuration.");
        throw err;
    }

#ifdef USE_AVX

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num<14 || !parallel) ) {
        apply_kernel_to_state_vector_input_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        if ( parallel == 1 ) {
            apply_kernel_to_state_vector_input_parallel_OpenMP_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        }
        else if ( parallel == 2 ) {
            apply_kernel_to_state_vector_input_parallel_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        }
        else {
            std::string err("Gate::apply_kernel_to: the argument parallel should be either 0,1 or 2. Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB"); 
            throw err;
        }
        return;
    }



    // unitary transform kernels
    if ( qbit_num < 4 ) {
        apply_kernel_to_input_AVX_small(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( qbit_num < 10 || !parallel) {
        apply_kernel_to_input_AVX(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
     }
    else {
        apply_kernel_to_input_AVX_parallel(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
     }


#else

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num<10 || !parallel) ) {
        apply_kernel_to_state_vector_input(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        apply_kernel_to_state_vector_input_parallel(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }


    // apply kernel on unitary
    apply_kernel_to_input(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size); 


   


#endif // USE_AVX


}


void
Gate::apply_kernel_to(Matrix_float& u3_1qbit, Matrix_float& input, bool deriv, int parallel, const Matrix_float* alt_kernel) {

    (void)deriv;

    if (type == SWAP_OPERATION || type == CSWAP_OPERATION) {
        switch (parallel) {
            case 0:
                apply_SWAP_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
            case 1:
                apply_SWAP_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
            case 2:
                apply_SWAP_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
            default:
                apply_SWAP_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
        }
        return;
    }

    if (type == CCX_OPERATION) {
        switch (parallel) {
            case 0:
                apply_X_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
            case 1:
                apply_X_kernel_to_input_omp(input, target_qbits, control_qbits, matrix_size); break;
            case 2:
                apply_X_kernel_to_input_tbb(input, target_qbits, control_qbits, matrix_size); break;
            default:
                apply_X_kernel_to_input(input, target_qbits, control_qbits, matrix_size); break;
        }
        return;
    }

    if (type == SYC_OPERATION) {
        switch (parallel) {
            case 0:
                apply_SYC_kernel_to_input(input, target_qbit, control_qbit, matrix_size); break;
            case 1:
                apply_SYC_kernel_to_input_omp(input, target_qbit, control_qbit, matrix_size); break;
            case 2:
                apply_SYC_kernel_to_input_tbb(input, target_qbit, control_qbit, matrix_size); break;
            default:
                apply_SYC_kernel_to_input(input, target_qbit, control_qbit, matrix_size); break;
        }
        return;
    }

    if (type == CROT_OPERATION && alt_kernel != nullptr) {
        Matrix_float branch0 = alt_kernel->copy();
        Matrix_float branch1 = u3_1qbit.copy();
        if (input.cols != 1) {
            for (int col_idx = 0; col_idx < input.cols; ++col_idx) {
                Matrix_float col_mtx(input.rows, 1);
                for (int row_idx = 0; row_idx < input.rows; ++row_idx) {
                    col_mtx[row_idx * col_mtx.stride] = input[row_idx * input.stride + col_idx];
                }

                Matrix_float col_branch0 = branch0.copy();
                Matrix_float col_branch1 = branch1.copy();
#ifdef USE_AVX
                if (parallel) {
                    apply_crot_kernel_to_matrix_input_AVX_parallel32(col_branch0, col_branch1, col_mtx, target_qbit, control_qbit, col_mtx.rows);
                } else {
                    apply_crot_kernel_to_matrix_input_AVX32(col_branch0, col_branch1, col_mtx, target_qbit, control_qbit, col_mtx.rows);
                }
#else
                apply_crot_kernel_to_matrix_input(col_branch0, col_branch1, col_mtx, target_qbit, control_qbit, col_mtx.rows);
#endif

                for (int row_idx = 0; row_idx < input.rows; ++row_idx) {
                    input[row_idx * input.stride + col_idx] = col_mtx[row_idx * col_mtx.stride];
                }
            }
        } else {
#ifdef USE_AVX
            if (parallel) {
                apply_crot_kernel_to_matrix_input_AVX_parallel32(branch0, branch1, input, target_qbit, control_qbit, input.rows);
            } else {
                apply_crot_kernel_to_matrix_input_AVX32(branch0, branch1, input, target_qbit, control_qbit, input.rows);
            }
#else
            apply_crot_kernel_to_matrix_input(branch0, branch1, input, target_qbit, control_qbit, input.rows);
#endif
        }
        return;
    }

    if (type == GENERAL_OPERATION) {
        const std::vector<int> involved_qbits = get_involved_qubits();
        const bool has_any_control = (control_qbit >= 0) || !control_qbits.empty();
        const bool is_state_vector = (input.cols == 1);
        const int involved_qbit_num = static_cast<int>(involved_qbits.size());
        const bool has_full_matrix = matrix_alloc.rows == matrix_size && matrix_alloc.cols == matrix_size;
        const bool has_local_matrix = involved_qbit_num > 0
            && matrix_alloc.rows == Power_of_2(involved_qbit_num)
            && matrix_alloc.cols == Power_of_2(involved_qbit_num);
        const bool can_use_large_kernel = !has_any_control
            && ((is_state_vector && involved_qbit_num >= 2 && involved_qbit_num <= 5)
                || (!is_state_vector && involved_qbit_num == 2));
        Matrix_float matrix_alloc32 = matrix_alloc.to_float32();

        if (has_full_matrix) {
            if (can_use_large_kernel) {
#ifdef USE_AVX
                switch (parallel) {
                    case 0:
                        apply_large_kernel_to_input_AVX32(matrix_alloc32, input, involved_qbits, matrix_size);
                        break;
                    case 1:
                        apply_large_kernel_to_input_AVX_OpenMP32(matrix_alloc32, input, involved_qbits, matrix_size);
                        break;
                    case 2:
                        apply_large_kernel_to_input_AVX_TBB32(matrix_alloc32, input, involved_qbits, matrix_size);
                        break;
                    default:
                        apply_large_kernel_to_input_AVX32(matrix_alloc32, input, involved_qbits, matrix_size);
                }
#else
                apply_large_kernel_to_input(matrix_alloc32, input, involved_qbits, matrix_size);
#endif
            }
            else {
                Matrix_float transformed = dot(matrix_alloc32, input);
                memcpy(input.get_data(), transformed.get_data(), transformed.size() * sizeof(QGD_Complex8));
            }
            return;
        }

        if (has_local_matrix && matrix_alloc32.rows == 2 && matrix_alloc32.cols == 2 && involved_qbit_num == 1) {
            u3_1qbit = matrix_alloc32;
        }
        else if (has_local_matrix && can_use_large_kernel) {
#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(matrix_alloc32, input, involved_qbits, matrix_size);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(matrix_alloc32, input, involved_qbits, matrix_size);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(matrix_alloc32, input, involved_qbits, matrix_size);
                    break;
                default:
                    apply_large_kernel_to_input_AVX32(matrix_alloc32, input, involved_qbits, matrix_size);
            }
#else
            apply_large_kernel_to_input(matrix_alloc32, input, involved_qbits, matrix_size);
#endif
            return;
        }
        else {
            std::string err("Gate::apply_kernel_to(Matrix_float&): unsupported GENERAL_OPERATION dispatch for stored matrix size.");
            throw err;
        }
    }

    if (u3_1qbit.rows != 2 || u3_1qbit.cols != 2) {
        const std::vector<int> involved_qbits = get_involved_qubits();
        const bool has_any_control = (control_qbit >= 0) || !control_qbits.empty();
        const bool is_state_vector = (input.cols == 1);
        const int involved_qbit_num = static_cast<int>(involved_qbits.size());
        const bool can_use_large_kernel = !has_any_control
            && ((is_state_vector && involved_qbit_num >= 2 && involved_qbit_num <= 5)
                || (!is_state_vector && involved_qbit_num == 2));

        if (can_use_large_kernel) {
#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u3_1qbit, input, involved_qbits, matrix_size);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u3_1qbit, input, involved_qbits, matrix_size);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u3_1qbit, input, involved_qbits, matrix_size);
                    break;
                default:
                    apply_large_kernel_to_input_AVX32(u3_1qbit, input, involved_qbits, matrix_size);
            }
#else
            apply_large_kernel_to_input(u3_1qbit, input, involved_qbits, matrix_size);
#endif
            return;
        }

        if (u3_1qbit.rows == matrix_size && u3_1qbit.cols == matrix_size) {
            Matrix_float transformed = dot(u3_1qbit, input);
            memcpy(input.get_data(), transformed.get_data(), transformed.size() * sizeof(QGD_Complex8));
            return;
        }

        std::string err("Gate::apply_kernel_to(Matrix_float&): unsupported non-2x2 kernel dispatch for this configuration.");
        throw err;
    }

#ifdef USE_AVX

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num<14 || !parallel) ) {
        apply_kernel_to_state_vector_input_AVX32(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        if ( parallel == 1 ) {
            apply_kernel_to_state_vector_input_parallel_OpenMP_AVX32(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        }
        else if ( parallel == 2 ) {
            apply_kernel_to_state_vector_input_parallel_AVX32(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        }
        else {
            std::string err("Gate::apply_kernel_to(Matrix_float&): the argument parallel should be either 0,1 or 2.");
            throw err;
        }
        return;
    }

    // unitary transform kernels
    if ( qbit_num < 4 ) {
        apply_kernel_to_input_AVX_small32(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( qbit_num < 10 || !parallel) {
        apply_kernel_to_input_AVX32(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
     }
    else {
        apply_kernel_to_input_AVX_parallel32(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
     }

#else

    // apply kernel on state vector
    if ( input.cols == 1 && (qbit_num < 10 || !parallel) ) {
        apply_kernel_to_state_vector_input(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }
    else if ( input.cols == 1 ) {
        apply_kernel_to_state_vector_input_parallel(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);
        return;
    }

    apply_kernel_to_input(u3_1qbit, input, deriv, target_qbit, control_qbit, matrix_size);

#endif // USE_AVX

}





/**
@brief Call to apply the gate kernel on the input state or unitary from right (no AVX support)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
*/
void 
Gate::apply_kernel_from_right( Matrix& u3_1qbit, Matrix& input, const Matrix* alt_kernel ) {

    if (type == SYC_OPERATION) {
        apply_SYC_kernel_from_right(input, target_qbit, control_qbit, matrix_size);
        return;
    }

    if (u3_1qbit.rows == 0 || u3_1qbit.cols == 0) {
        Matrix gate_matrix = create_identity(matrix_size);
        apply_to(gate_matrix, 0);
        Matrix ret = dot(input, gate_matrix);
        memcpy(input.get_data(), ret.get_data(), ret.size() * sizeof(QGD_Complex16));
        return;
    }

    if (type == SWAP_OPERATION || type == CSWAP_OPERATION || type == CCX_OPERATION || type == CROT_OPERATION || u3_1qbit.rows != 2 || u3_1qbit.cols != 2) {
        Matrix gate_matrix = create_identity(matrix_size);
        Matrix kernel_copy = u3_1qbit.copy();
        if (alt_kernel != nullptr) {
            Matrix alt_copy = alt_kernel->copy();
            apply_kernel_to(kernel_copy, gate_matrix, false, 0, &alt_copy);
        } else {
            apply_kernel_to(kernel_copy, gate_matrix, false, 0);
        }
        Matrix ret = dot(input, gate_matrix);
        memcpy(input.get_data(), ret.get_data(), ret.size() * sizeof(QGD_Complex16));
        return;
    }

   
    ::apply_kernel_from_right(u3_1qbit, input, target_qbit, control_qbit, matrix_size);


}


void
Gate::apply_kernel_from_right( Matrix_float& u3_1qbit, Matrix_float& input, const Matrix_float* alt_kernel ) {

    if (type == SYC_OPERATION) {
        apply_SYC_kernel_from_right(input, target_qbit, control_qbit, matrix_size);
        return;
    }

    if (u3_1qbit.rows == 0 || u3_1qbit.cols == 0) {
        Matrix_float gate_matrix = create_identity_float(matrix_size);
        apply_to(gate_matrix, 0);
        Matrix_float ret = dot(input, gate_matrix);
        memcpy(input.get_data(), ret.get_data(), ret.size() * sizeof(QGD_Complex8));
        return;
    }

    if (type == SWAP_OPERATION || type == CSWAP_OPERATION || type == CCX_OPERATION || type == CROT_OPERATION || u3_1qbit.rows != 2 || u3_1qbit.cols != 2) {
        Matrix_float gate_matrix = create_identity_float(matrix_size);
        Matrix_float kernel_copy = u3_1qbit.copy();
        if (alt_kernel != nullptr) {
            Matrix_float alt_copy = alt_kernel->copy();
            apply_kernel_to(kernel_copy, gate_matrix, false, 0, &alt_copy);
        } else {
            apply_kernel_to(kernel_copy, gate_matrix, false, 0);
        }
        Matrix_float ret = dot(input, gate_matrix);
        memcpy(input.get_data(), ret.get_data(), ret.size() * sizeof(QGD_Complex8));
        return;
    }

    ::apply_kernel_from_right(u3_1qbit, input, target_qbit, control_qbit, matrix_size);

}

/**
@brief Call to set the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@param start_idx The starting index
*/
void 
Gate::set_parameter_start_idx(int start_idx) {

    parameter_start_idx = start_idx;

}

/**
@brief Call to set the parents of the current gate
@param parents_ the list of the parents
*/
void 
Gate::set_parents( std::vector<Gate*>& parents_ ) {

    parents = parents_;

}


/**
@brief Call to set the children of the current gate
@param children_ the list of the children
*/
void 
Gate::set_children( std::vector<Gate*>& children_ ) {

    children = children_;

}


/**
@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated
@param start_idx The starting index
*/
int 
Gate::get_parameter_start_idx() {

    return parameter_start_idx;
    
}


/**
@brief Default gate_kernel: throws if not overridden.
*/
Matrix
Gate::gate_kernel(const Matrix_real& /*precomputed_sincos*/) {
    std::string err(name + "::gate_kernel(double) not implemented");
    throw err;
    return Matrix(0, 0);
}

Matrix_float
Gate::gate_kernel(const Matrix_real_float& /*precomputed_sincos*/) {
    std::string err(name + "::gate_kernel(float) not implemented");
    throw err;
    return Matrix_float(0, 0);
}

Matrix
Gate::inverse_gate_kernel(const Matrix_real& precomputed_sincos) {
    Matrix fwd = gate_kernel(precomputed_sincos);
    Matrix inv(fwd.rows, fwd.cols);
    for (int row_idx = 0; row_idx < fwd.rows; ++row_idx) {
        const int row_offset = row_idx * fwd.stride;
        for (int col_idx = 0; col_idx < fwd.cols; ++col_idx) {
            const QGD_Complex16& src = fwd[row_offset + col_idx];
            QGD_Complex16& dst = inv[col_idx * inv.stride + row_idx];
            dst.real = src.real;
            dst.imag = -src.imag;
        }
    }
    return inv;
}

Matrix_float
Gate::inverse_gate_kernel(const Matrix_real_float& precomputed_sincos) {
    Matrix_float fwd = gate_kernel(precomputed_sincos);
    Matrix_float inv(fwd.rows, fwd.cols);
    for (int row_idx = 0; row_idx < fwd.rows; ++row_idx) {
        const int row_offset = row_idx * fwd.stride;
        for (int col_idx = 0; col_idx < fwd.cols; ++col_idx) {
            const QGD_Complex8& src = fwd[row_offset + col_idx];
            QGD_Complex8& dst = inv[col_idx * inv.stride + row_idx];
            dst.real = src.real;
            dst.imag = -src.imag;
        }
    }
    return inv;
}

/**
@brief Returns the per-parameter multipliers relative to 2π.
       Default implementation: empty (zero-parameter gate).
*/
std::vector<double>
Gate::get_parameter_multipliers() const {
    return {};
}


Matrix
Gate::calc_one_qubit_u3(double ThetaOver2, double Phi, double Lambda) {
    double sin_theta, cos_theta;
    double sin_phi, cos_phi;
    double sin_lambda, cos_lambda;
    qgd_sincos<double>(ThetaOver2, &sin_theta, &cos_theta);
    qgd_sincos<double>(Phi, &sin_phi, &cos_phi);
    qgd_sincos<double>(Lambda, &sin_lambda, &cos_lambda);
    return calc_one_qubit_u3_from_trig<Matrix, double>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}


Matrix_float
Gate::calc_one_qubit_u3(float ThetaOver2, float Phi, float Lambda) {
    float sin_theta, cos_theta;
    float sin_phi, cos_phi;
    float sin_lambda, cos_lambda;
    qgd_sincos<float>(ThetaOver2, &sin_theta, &cos_theta);
    qgd_sincos<float>(Phi, &sin_phi, &cos_phi);
    qgd_sincos<float>(Lambda, &sin_lambda, &cos_lambda);
    return calc_one_qubit_u3_from_trig<Matrix_float, float>(sin_theta, cos_theta, sin_phi, cos_phi, sin_lambda, cos_lambda);
}


/**
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is embedded.
       Uses get_parameter_multipliers() to apply fmod wrapping generically.
       Multiplier m → extracted[i] = fmod(m * params[start+i], m * 2π).
*/
Matrix_real 
Gate::extract_parameters( Matrix_real& parameters ) {

    const std::vector<double> mults = get_parameter_multipliers();

    if (mults.empty()) {
        return Matrix_real(0, 0);
    }

    const int n = static_cast<int>(mults.size());
    if ( get_parameter_start_idx() + n > (int)parameters.size() ) {
        std::string err(name + "::extract_parameters: Can't extract parameters, input array has not enough elements.");
        throw err;
    }

    Matrix_real extracted_parameters(1, n);
    const int start = get_parameter_start_idx();
    for (int i = 0; i < n; ++i) {
        const double m = mults[i];
        extracted_parameters[i] = std::fmod(m * parameters[start + i], m * 2.0 * M_PI);
    }

    return extracted_parameters;

}


/**
@brief Float32 overload of extract_parameters. Uses get_parameter_multipliers() identically.
       Multiplier m → extracted[i] = fmodf(m * params[start+i], m * 2π).
*/
Matrix_real_float
Gate::extract_parameters( Matrix_real_float& parameters ) {

    const std::vector<double> mults = get_parameter_multipliers();

    if (mults.empty()) {
        return Matrix_real_float(0, 0);
    }

    const int n = static_cast<int>(mults.size());
    if ( get_parameter_start_idx() + n > (int)parameters.size() ) {
        std::string err(name + "::extract_parameters: Can't extract parameters, input array has not enough elements.");
        throw err;
    }

    Matrix_real_float extracted_parameters(1, n);
    const int start = get_parameter_start_idx();
    for (int i = 0; i < n; ++i) {
        const float m = static_cast<float>(mults[i]);
        extracted_parameters[i] = std::fmod(m * parameters[start + i], m * static_cast<float>(2.0 * M_PI));
    }

    return extracted_parameters;

}

/**
@brief Call to get the name label of the gate.
@return Returns with the name label of the gate
*/
std::string 
Gate::get_name() {

    return name;

}

