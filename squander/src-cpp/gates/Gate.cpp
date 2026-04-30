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
#include "qgd_math.h"
#include <sstream>

#ifdef USE_AVX 
#include "apply_kernel_to_input_AVX.h"
#include "apply_kernel_to_state_vector_input_AVX.h"
#include "apply_large_kernel_to_input_AVX.h"
#endif

#include "apply_kernel_to_input.h"
#include "apply_kernel_to_state_vector_input.h"
#include "apply_large_kernel_to_input.h"

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
@brief Call to retrieve the operation matrix
@return Returns with a matrix of the operation
*/
Matrix
Gate::get_matrix() {

    return matrix_alloc;
}


/**
@brief Call to retrieve the operation matrix
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with the matrix of the operation
*/
Matrix
Gate::get_matrix(int parallel) {

    std::string err("Gate::get_matrix: Unimplemented function");
    throw err;     

}


/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@return Returns with a matrix of the gate
*/
Matrix
Gate::get_matrix(Matrix_real& parameters ) {

    std::string err("Gate::get_matrix: Unimplemented function");
    throw err;  
   
}



/**
@brief Call to retrieve the gate matrix
@param parameters An array of parameters to calculate the matrix of the U3 gate.
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
@return Returns with a matrix of the gate
*/
Matrix
Gate::get_matrix( Matrix_real& parameters, int parallel ) {

    std::string err("Gate::get_matrix: Unimplemented function");
    throw err;     

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

    return;

}



/**
@brief Call to apply the gate on the input array/matrix
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to( Matrix& input, int parallel ) {

   if (input.rows != matrix_size ) {
        std::string err("Gate::apply_to: Wrong matrix size in Gate gate apply.");
        throw err;    
    }

    Matrix ret = dot(matrix_alloc, input);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );
    //input = ret;
}


void
Gate::apply_to( Matrix_float& input, int parallel ) {

    if (input.rows != matrix_size) {
        std::string err("Gate::apply_to(Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    // Stabilized path: execute via the validated float64 implementation and cast back.
    Matrix input_float64 = input.to_float64();
    this->apply_to(input_float64, parallel);
    for (int idx = 0; idx < input.size(); ++idx) {
        input[idx].real = static_cast<float>(input_float64[idx].real);
        input[idx].imag = static_cast<float>(input_float64[idx].imag);
    }
    return;

    Matrix_float u3_1qbit(2, 2);

    const float inv_sqrt2 = 0.7071067811865475f;
    const float inv2 = 0.5f;
    const float pi_over_4 = 0.7853981633974483f;

    for (int idx = 0; idx < 4; ++idx) {
        u3_1qbit[idx].real = 0.0f;
        u3_1qbit[idx].imag = 0.0f;
    }

    switch (type) {
        case X_OPERATION:
        case CNOT_OPERATION:
            u3_1qbit[1].real = 1.0f;
            u3_1qbit[2].real = 1.0f;
            break;

        case Y_OPERATION:
            u3_1qbit[1].imag = -1.0f;
            u3_1qbit[2].imag = 1.0f;
            break;

        case Z_OPERATION:
        case CZ_OPERATION:
            u3_1qbit[0].real = 1.0f;
            u3_1qbit[3].real = -1.0f;
            break;

        case H_OPERATION:
        case CH_OPERATION:
            u3_1qbit[0].real = inv_sqrt2;
            u3_1qbit[1].real = inv_sqrt2;
            u3_1qbit[2].real = inv_sqrt2;
            u3_1qbit[3].real = -inv_sqrt2;
            break;

        case S_OPERATION:
            u3_1qbit[0].real = 1.0f;
            u3_1qbit[3].imag = 1.0f;
            break;

        case SDG_OPERATION:
            u3_1qbit[0].real = 1.0f;
            u3_1qbit[3].imag = -1.0f;
            break;

        case T_OPERATION:
            u3_1qbit[0].real = 1.0f;
            qgd_sincos<float>(pi_over_4, &u3_1qbit[3].imag, &u3_1qbit[3].real);
            break;

        case TDG_OPERATION:
            u3_1qbit[0].real = 1.0f;
            qgd_sincos<float>(-pi_over_4, &u3_1qbit[3].imag, &u3_1qbit[3].real);
            break;

        case SX_OPERATION:
            u3_1qbit[0].real = inv2;
            u3_1qbit[0].imag = inv2;
            u3_1qbit[1].real = inv2;
            u3_1qbit[1].imag = -inv2;
            u3_1qbit[2].real = inv2;
            u3_1qbit[2].imag = -inv2;
            u3_1qbit[3].real = inv2;
            u3_1qbit[3].imag = inv2;
            break;

        case SXDG_OPERATION:
            u3_1qbit[0].real = inv2;
            u3_1qbit[0].imag = -inv2;
            u3_1qbit[1].real = inv2;
            u3_1qbit[1].imag = inv2;
            u3_1qbit[2].real = inv2;
            u3_1qbit[2].imag = inv2;
            u3_1qbit[3].real = inv2;
            u3_1qbit[3].imag = -inv2;
            break;

        case SWAP_OPERATION: {
            if (target_qbits.size() != 2) {
                throw std::string("Gate::apply_to(Matrix_float&): SWAP expects exactly 2 target qubits.");
            }

            Matrix_float u_2qbit(4, 4);
            for (int idx = 0; idx < 16; ++idx) {
                u_2qbit[idx].real = 0.0f;
                u_2qbit[idx].imag = 0.0f;
            }

            u_2qbit[0].real = 1.0f;
            u_2qbit[1 * 4 + 2].real = 1.0f;
            u_2qbit[2 * 4 + 1].real = 1.0f;
            u_2qbit[3 * 4 + 3].real = 1.0f;

#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u_2qbit, input, target_qbits, input.rows);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u_2qbit, input, target_qbits, input.rows);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u_2qbit, input, target_qbits, input.rows);
                    break;
                default:
                    throw std::string("Gate::apply_to(Matrix_float&): invalid parallel mode");
            }
#else
            throw std::string("Gate::apply_to(Matrix_float&): SWAP float32 requires AVX float32 kernels.");
#endif
            return;
        }

        case CSWAP_OPERATION: {
            if (target_qbits.size() != 2) {
                throw std::string("Gate::apply_to(Matrix_float&): CSWAP expects exactly 2 target qubits.");
            }
            if (control_qbits.size() != 1) {
                throw std::string("Gate::apply_to(Matrix_float&): CSWAP expects exactly 1 control qubit.");
            }

            Matrix_float u_3qbit(8, 8);
            for (int idx = 0; idx < 64; ++idx) {
                u_3qbit[idx].real = 0.0f;
                u_3qbit[idx].imag = 0.0f;
            }

            for (int idx = 0; idx < 8; ++idx) {
                u_3qbit[idx * 8 + idx].real = 1.0f;
            }
            u_3qbit[5 * 8 + 5].real = 0.0f;
            u_3qbit[6 * 8 + 6].real = 0.0f;
            u_3qbit[5 * 8 + 6].real = 1.0f;
            u_3qbit[6 * 8 + 5].real = 1.0f;

            std::vector<int> involved_qbits = {control_qbits[0], target_qbits[0], target_qbits[1]};

#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u_3qbit, input, involved_qbits, input.rows);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u_3qbit, input, involved_qbits, input.rows);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u_3qbit, input, involved_qbits, input.rows);
                    break;
                default:
                    throw std::string("Gate::apply_to(Matrix_float&): invalid parallel mode");
            }
#else
            throw std::string("Gate::apply_to(Matrix_float&): CSWAP float32 requires AVX float32 kernels.");
#endif
            return;
        }

        case CCX_OPERATION: {
            if (target_qbits.size() != 1) {
                throw std::string("Gate::apply_to(Matrix_float&): CCX expects exactly 1 target qubit.");
            }
            if (control_qbits.size() != 2) {
                throw std::string("Gate::apply_to(Matrix_float&): CCX expects exactly 2 control qubits.");
            }

            Matrix_float u_3qbit(8, 8);
            for (int idx = 0; idx < 64; ++idx) {
                u_3qbit[idx].real = 0.0f;
                u_3qbit[idx].imag = 0.0f;
            }

            for (int idx = 0; idx < 8; ++idx) {
                u_3qbit[idx * 8 + idx].real = 1.0f;
            }
            u_3qbit[6 * 8 + 6].real = 0.0f;
            u_3qbit[7 * 8 + 7].real = 0.0f;
            u_3qbit[6 * 8 + 7].real = 1.0f;
            u_3qbit[7 * 8 + 6].real = 1.0f;

            std::vector<int> involved_qbits = {control_qbits[0], control_qbits[1], target_qbits[0]};

#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u_3qbit, input, involved_qbits, input.rows);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u_3qbit, input, involved_qbits, input.rows);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u_3qbit, input, involved_qbits, input.rows);
                    break;
                default:
                    throw std::string("Gate::apply_to(Matrix_float&): invalid parallel mode");
            }
#else
            throw std::string("Gate::apply_to(Matrix_float&): CCX float32 requires AVX float32 kernels.");
#endif
            return;
        }

        case SYC_OPERATION: {
            Matrix_float u_2qbit(4, 4);
            for (int idx = 0; idx < 16; ++idx) {
                u_2qbit[idx].real = 0.0f;
                u_2qbit[idx].imag = 0.0f;
            }

            // |00> -> |00>
            u_2qbit[0].real = 1.0f;
            // |01> <-> |10> with factor -i
            u_2qbit[1 * 4 + 2].imag = -1.0f;
            u_2qbit[2 * 4 + 1].imag = -1.0f;
            // |11> -> (sqrt(3)/2 - i/2) |11>
            u_2qbit[3 * 4 + 3].real = 0.8660254037844386f;
            u_2qbit[3 * 4 + 3].imag = -0.5f;

            std::vector<int> involved_qbits = {control_qbit, target_qbit};

#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u_2qbit, input, involved_qbits, input.rows);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u_2qbit, input, involved_qbits, input.rows);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u_2qbit, input, involved_qbits, input.rows);
                    break;
                default:
                    throw std::string("Gate::apply_to(Matrix_float&): invalid parallel mode");
            }
#else
            throw std::string("Gate::apply_to(Matrix_float&): SYC float32 requires AVX float32 kernels.");
#endif
            return;
        }

        default: {
            std::string err("Gate::apply_to(Matrix_float&): Float32 gate path is not implemented for gate type " + name);
            throw err;
        }
    }

    apply_kernel_to(u3_1qbit, input, false, parallel);
}


/**
@brief Abstract function to be overriden in derived classes to be used to transform an input upon a parametric gate operation
@param parameter_mtx An array conatining the parameters
@param input The input array on which the gate is applied
@param parallel Set 0 for sequential execution, 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_to( Matrix_real& parameter_mtx, Matrix& input, int parallel ) {

    std::string err("Unimplemented abstract function apply_to");
    throw( err );

    return;
}


void
Gate::apply_to( Matrix_real_float& parameter_mtx, Matrix_float& input, int parallel ) {

    if (input.rows != matrix_size) {
        std::string err("Gate::apply_to(Matrix_real_float&, Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    // Stabilized path: execute via validated float64 parameterized implementation and cast back.
    Matrix input_float64 = input.to_float64();
    Matrix_real parameter_float64(parameter_mtx.rows, parameter_mtx.cols);
    for (int idx = 0; idx < parameter_mtx.size(); ++idx) {
        parameter_float64[idx] = static_cast<double>(parameter_mtx[idx]);
    }

    this->apply_to(parameter_float64, input_float64, parallel);

    for (int idx = 0; idx < input.size(); ++idx) {
        input[idx].real = static_cast<float>(input_float64[idx].real);
        input[idx].imag = static_cast<float>(input_float64[idx].imag);
    }
    return;

    Matrix_float u3_1qbit(2, 2);

    for (int idx = 0; idx < 4; ++idx) {
        u3_1qbit[idx].real = 0.0f;
        u3_1qbit[idx].imag = 0.0f;
    }

    float theta_over_2 = 0.0f;
    float phi = 0.0f;
    float lambda = 0.0f;
    float global_phase = 0.0f;

    switch (type) {
        case U3_OPERATION:
        case CU_OPERATION:
            if (parameter_mtx.size() < 3) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): U3/CU expects at least 3 parameters.");
            }
            theta_over_2 = parameter_mtx[0];
            phi = parameter_mtx[1];
            lambda = parameter_mtx[2];
            if (type == CU_OPERATION) {
                if (parameter_mtx.size() < 4) {
                    throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): CU expects 4 parameters.");
                }
                global_phase = parameter_mtx[3];
            }
            break;

        case U2_OPERATION:
            if (parameter_mtx.size() < 2) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): U2 expects 2 parameters.");
            }
            theta_over_2 = static_cast<float>(M_PI / 4.0);
            phi = parameter_mtx[0];
            lambda = parameter_mtx[1];
            break;

        case U1_OPERATION:
        case CP_OPERATION:
        case RZ_OPERATION:
        case CRZ_OPERATION:
        case RZ_P_OPERATION:
            if (parameter_mtx.size() < 1) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): U1/CP/RZ-family expects 1 parameter.");
            }
            theta_over_2 = 0.0f;
            phi = 0.0f;
            lambda = parameter_mtx[0];
            break;

        case CZ_NU_OPERATION:
            if (parameter_mtx.size() < 1) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): CZ_NU expects 1 parameter.");
            }
            u3_1qbit[0].real = 1.0f;
            u3_1qbit[0].imag = 0.0f;
            u3_1qbit[1].real = 0.0f;
            u3_1qbit[1].imag = 0.0f;
            u3_1qbit[2].real = 0.0f;
            u3_1qbit[2].imag = 0.0f;
            u3_1qbit[3].real = qgd_cos<float>(parameter_mtx[0]);
            u3_1qbit[3].imag = 0.0f;
            apply_kernel_to(u3_1qbit, input, false, parallel);
            return;

        case RX_OPERATION:
        case CRX_OPERATION:
            if (parameter_mtx.size() < 1) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): RX-family expects 1 parameter.");
            }
            theta_over_2 = parameter_mtx[0];
            phi = -static_cast<float>(M_PI / 2.0);
            lambda = static_cast<float>(M_PI / 2.0);
            break;

        case RY_OPERATION:
        case CRY_OPERATION:
        case ADAPTIVE_OPERATION:
            if (parameter_mtx.size() < 1) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): RY-family expects 1 parameter.");
            }
            theta_over_2 = parameter_mtx[0];
            phi = 0.0f;
            lambda = 0.0f;
            break;

        case R_OPERATION:
        case CR_OPERATION:
            if (parameter_mtx.size() < 2) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): R-family expects 2 parameters.");
            }
            theta_over_2 = parameter_mtx[0];
            phi = parameter_mtx[1] - static_cast<float>(M_PI / 2.0);
            lambda = -parameter_mtx[1] + static_cast<float>(M_PI / 2.0);
            break;

        case CROT_OPERATION: {
            if (parameter_mtx.size() < 2) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): CROT expects 2 parameters.");
            }

            int crot_target_qbit = target_qbit;
            int crot_control_qbit = control_qbit;

            if (target_qbits.size() == 1) {
                crot_target_qbit = target_qbits[0];
            }
            else if (crot_target_qbit < 0) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): CROT expects exactly 1 target qubit.");
            }

            if (control_qbits.size() == 1) {
                crot_control_qbit = control_qbits[0];
            }
            else if (crot_control_qbit < 0) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): CROT expects exactly 1 control qubit.");
            }

            float theta = parameter_mtx[0];
            float phi_crot = parameter_mtx[1];

            float c = qgd_cos<float>(theta);
            float s = qgd_sin<float>(theta);
            float s_neg = qgd_sin<float>(-theta);
            float sin_phi = qgd_sin<float>(phi_crot);
            float cos_phi = qgd_cos<float>(phi_crot);

            Matrix_float u_2qbit(4, 4);
            for (int idx = 0; idx < 16; ++idx) {
                u_2qbit[idx].real = 0.0f;
                u_2qbit[idx].imag = 0.0f;
            }

            u_2qbit[0].real = c;
            u_2qbit[2].real = s * sin_phi;
            u_2qbit[2].imag = s * cos_phi;
            u_2qbit[1 * 4 + 3].real = -s_neg * sin_phi;
            u_2qbit[1 * 4 + 3].imag = -s_neg * cos_phi;
            u_2qbit[1 * 4 + 1].real = c;
            u_2qbit[2 * 4 + 2].real = c;
            u_2qbit[2 * 4].real = -s * sin_phi;
            u_2qbit[2 * 4].imag = s * cos_phi;
            u_2qbit[3 * 4 + 3].real = c;
            u_2qbit[3 * 4 + 1].real = s_neg * sin_phi;
            u_2qbit[3 * 4 + 1].imag = -s_neg * cos_phi;

            std::vector<int> involved_qbits = {crot_control_qbit, crot_target_qbit};

#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u_2qbit, input, involved_qbits, input.rows);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u_2qbit, input, involved_qbits, input.rows);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u_2qbit, input, involved_qbits, input.rows);
                    break;
                default:
                    throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): invalid parallel mode");
            }
#else
            throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): CROT float32 requires AVX float32 kernels.");
#endif
            return;
        }

        case RXX_OPERATION:
        case RYY_OPERATION:
        case RZZ_OPERATION: {
            if (parameter_mtx.size() < 1) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): RXX/RYY/RZZ expects 1 parameter.");
            }
            if (target_qbits.size() != 2) {
                throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): RXX/RYY/RZZ expects exactly 2 target qubits.");
            }

            float theta = parameter_mtx[0];
            Matrix_float u_2qbit(4, 4);
            for (int idx = 0; idx < 16; ++idx) {
                u_2qbit[idx].real = 0.0f;
                u_2qbit[idx].imag = 0.0f;
            }

            float c = qgd_cos<float>(theta);
            float s = qgd_sin<float>(theta);

            if (type == RXX_OPERATION) {
                u_2qbit[0].real = c;
                u_2qbit[3].imag = -s;
                u_2qbit[1 * 4 + 1].real = c;
                u_2qbit[1 * 4 + 2].imag = -s;
                u_2qbit[2 * 4 + 2].real = c;
                u_2qbit[2 * 4 + 1].imag = -s;
                u_2qbit[3 * 4 + 3].real = c;
                u_2qbit[3 * 4].imag = -s;
            }
            else if (type == RYY_OPERATION) {
                u_2qbit[0].real = c;
                u_2qbit[3].imag = s;
                u_2qbit[1 * 4 + 1].real = c;
                u_2qbit[1 * 4 + 2].imag = -s;
                u_2qbit[2 * 4 + 2].real = c;
                u_2qbit[2 * 4 + 1].imag = -s;
                u_2qbit[3 * 4 + 3].real = c;
                u_2qbit[3 * 4].imag = s;
            }
            else {
                u_2qbit[0].real = c;
                u_2qbit[0].imag = -s;
                u_2qbit[1 * 4 + 1].real = c;
                u_2qbit[1 * 4 + 1].imag = s;
                u_2qbit[2 * 4 + 2].real = c;
                u_2qbit[2 * 4 + 2].imag = s;
                u_2qbit[3 * 4 + 3].real = c;
                u_2qbit[3 * 4 + 3].imag = -s;
            }

#ifdef USE_AVX
            switch (parallel) {
                case 0:
                    apply_large_kernel_to_input_AVX32(u_2qbit, input, target_qbits, input.rows);
                    break;
                case 1:
                    apply_large_kernel_to_input_AVX_OpenMP32(u_2qbit, input, target_qbits, input.rows);
                    break;
                case 2:
                    apply_large_kernel_to_input_AVX_TBB32(u_2qbit, input, target_qbits, input.rows);
                    break;
                default:
                    throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): invalid parallel mode");
            }
#else
            throw std::string("Gate::apply_to(Matrix_real_float&, Matrix_float&): RXX/RYY/RZZ float32 requires AVX float32 kernels.");
#endif
            return;
        }

        default: {
            std::string err("Gate::apply_to(Matrix_real_float&, Matrix_float&): Float32 parametric gate path is not implemented for gate type " + name);
            throw err;
        }
    }

    float c = qgd_cos<float>(theta_over_2);
    float s = qgd_sin<float>(theta_over_2);

    float sin_phi, cos_phi;
    float sin_lambda, cos_lambda;
    float sin_phl, cos_phl;
    qgd_sincos<float>(phi, &sin_phi, &cos_phi);
    qgd_sincos<float>(lambda, &sin_lambda, &cos_lambda);
    qgd_sincos<float>(phi + lambda, &sin_phl, &cos_phl);

    // [ c, -exp(i*lambda) s ; exp(i*phi) s, exp(i*(phi+lambda)) c ]
    u3_1qbit[0].real = c;
    u3_1qbit[0].imag = 0.0f;
    u3_1qbit[1].real = -cos_lambda * s;
    u3_1qbit[1].imag = -sin_lambda * s;
    u3_1qbit[2].real = cos_phi * s;
    u3_1qbit[2].imag = sin_phi * s;
    u3_1qbit[3].real = cos_phl * c;
    u3_1qbit[3].imag = sin_phl * c;

    if (type == CU_OPERATION && global_phase != 0.0f) {
        float gp_sin, gp_cos;
        qgd_sincos<float>(global_phase, &gp_sin, &gp_cos);
        for (int idx = 0; idx < 4; ++idx) {
            float real_new = gp_cos * u3_1qbit[idx].real - gp_sin * u3_1qbit[idx].imag;
            float imag_new = gp_sin * u3_1qbit[idx].real + gp_cos * u3_1qbit[idx].imag;
            u3_1qbit[idx].real = real_new;
            u3_1qbit[idx].imag = imag_new;
        }
    }

    apply_kernel_to(u3_1qbit, input, false, parallel);
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

    std::vector<Matrix> ret;
    return ret;

}


std::vector<Matrix_float>
Gate::apply_derivate_to( Matrix_real_float& parameters_mtx_in, Matrix_float& input, int parallel ) {

    (void)parameters_mtx_in;
    (void)input;
    (void)parallel;

    std::vector<Matrix_float> ret;
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

    Matrix gate_matrix = create_identity(matrix_size);
    apply_to(gate_matrix, 0);

    Matrix ret = dot(input, gate_matrix);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );

}


void
Gate::apply_from_right( Matrix_float& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix_float gate_matrix = create_identity_float(matrix_size);
    apply_to(gate_matrix, 0);

    Matrix_float ret = dot(input, gate_matrix);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex8) );
}


/**
@brief Call to apply the gate on the input array/matrix by input*Gate
@param parameter_mtx An array of the input parameters.
@param input The input array on which the gate is applied
*/
void 
Gate::apply_from_right( Matrix_real& parameter_mtx, Matrix& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix_real&, Matrix&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix gate_matrix = create_identity(matrix_size);
    apply_to(parameter_mtx, gate_matrix, 0);

    Matrix ret = dot(input, gate_matrix);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex16) );

    return;

}


void
Gate::apply_from_right( Matrix_real_float& parameter_mtx, Matrix_float& input ) {

    if (input.cols != matrix_size ) {
        std::string err("Gate::apply_from_right(Matrix_real_float&, Matrix_float&): Wrong matrix size in gate apply.");
        throw err;
    }

    Matrix_float gate_matrix = create_identity_float(matrix_size);
    apply_to(parameter_mtx, gate_matrix, 0);

    Matrix_float ret = dot(input, gate_matrix);
    memcpy( input.get_data(), ret.get_data(), ret.size()*sizeof(QGD_Complex8) );
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



/**
@brief Call to apply the gate kernel on the input state or unitary with optional AVX support
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise (optional)
@param deriv Set true to apply parallel kernels, false otherwise (optional)
@param parallel Set 0 for sequential execution (default), 1 for parallel execution with OpenMP and 2 for parallel with TBB (optional)
*/
void 
Gate::apply_kernel_to(Matrix& u3_1qbit, Matrix& input, bool deriv, int parallel) {

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
Gate::apply_kernel_to(Matrix_float& u3_1qbit, Matrix_float& input, bool deriv, int parallel) {

#ifdef USE_AVX

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
    std::string err("Gate::apply_kernel_to(Matrix_float&): Float32 path requires USE_AVX kernels.");
    throw err;
#endif

}





/**
@brief Call to apply the gate kernel on the input state or unitary from right (no AVX support)
@param u3_1qbit The 2x2 kernel of the gate operation
@param input The input matrix on which the transformation is applied
@param deriv Set true to apply derivate transformation, false otherwise
*/
void 
Gate::apply_kernel_from_right( Matrix& u3_1qbit, Matrix& input ) {

   
    int index_step_target = 1 << target_qbit;
    int current_idx = 0;
    int current_idx_pair = current_idx+index_step_target;

//std::cout << "target qbit: " << target_qbit << std::endl;

    while ( current_idx_pair < input.cols ) {

        for(int idx=0; idx<index_step_target; idx++) { 
        //tbb::parallel_for(0, index_step_target, 1, [&](int idx) {  

            int current_idx_loc = current_idx + idx;
            int current_idx_pair_loc = current_idx_pair + idx;

            // determine the action according to the state of the control qubit
            if ( control_qbit<0 || ((current_idx_loc >> control_qbit) & 1) ) {

                for ( int row_idx=0; row_idx<matrix_size; row_idx++) {

                    int row_offset = row_idx*input.stride;


                    int index      = row_offset+current_idx_loc;
                    int index_pair = row_offset+current_idx_pair_loc;

                    QGD_Complex16 element      = input[index];
                    QGD_Complex16 element_pair = input[index_pair];

                    QGD_Complex16 tmp1 = mult(u3_1qbit[0], element);
                    QGD_Complex16 tmp2 = mult(u3_1qbit[2], element_pair);
                    input[index].real = tmp1.real + tmp2.real;
                    input[index].imag = tmp1.imag + tmp2.imag;

                    tmp1 = mult(u3_1qbit[1], element);
                    tmp2 = mult(u3_1qbit[3], element_pair);
                    input[index_pair].real = tmp1.real + tmp2.real;
                    input[index_pair].imag = tmp1.imag + tmp2.imag;

                }

            }
            else {
                // leave the state as it is
                continue;
            }        


//std::cout << current_idx_target << " " << current_idx_target_pair << std::endl;


        //});
        }


        current_idx = current_idx + (index_step_target << 1);
        current_idx_pair = current_idx_pair + (index_step_target << 1);


    }


}

/**
@brief Calculate the matrix of a U3 gate gate corresponding to the given parameters acting on a single qbit space.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix Gate::calc_one_qubit_u3(double ThetaOver2, double Phi, double Lambda ) {

    Matrix u3_1qbit = Matrix(2,2); 
#ifdef DEBUG
    	if (isnan(ThetaOver2)) {
            std::stringstream sstream;
	    sstream << "Matrix U3::calc_one_qubit_u3: ThetaOver2 is NaN." << std::endl;
            print(sstream, 1);	    
        }
    	if (isnan(Phi)) {
            std::stringstream sstream;
	    sstream << "Matrix U3::calc_one_qubit_u3: Phi is NaN." << std::endl;
            print(sstream, 1);	     
        }
     	if (isnan(Lambda)) {
            std::stringstream sstream;
	    sstream << "Matrix U3::calc_one_qubit_u3: Lambda is NaN." << std::endl;
            print(sstream, 1);	   
        }
#endif // DEBUG
		
		double cos_theta = 1.0, sin_theta = 0.0;
		double cos_phi = 1.0, sin_phi = 0.0;
		double cos_lambda = 1.0, sin_lambda = 0.0;

        if (ThetaOver2!=0.0) qgd_sincos<double>(ThetaOver2, &sin_theta, &cos_theta);
        if (Phi!=0.0) qgd_sincos<double>(Phi, &sin_phi, &cos_phi);
        if (Lambda!=0.0) qgd_sincos<double>(Lambda, &sin_lambda, &cos_lambda);

        // the 1,1 element
        u3_1qbit[0].real = cos_theta;
        u3_1qbit[0].imag = 0;
        // the 1,2 element
        u3_1qbit[1].real = -cos_lambda*sin_theta;
        u3_1qbit[1].imag = -sin_lambda*sin_theta;
        // the 2,1 element
        u3_1qbit[2].real = cos_phi*sin_theta;
        u3_1qbit[2].imag = sin_phi*sin_theta;
        // the 2,2 element
        //cos(a+b)=cos(a)cos(b)-sin(a)sin(b)
        //sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
        u3_1qbit[3].real = (cos_phi*cos_lambda-sin_phi*sin_lambda)*cos_theta;
        u3_1qbit[3].imag = (sin_phi*cos_lambda+cos_phi*sin_lambda)*cos_theta;
        //u3_1qbit[3].real = cos(Phi+Lambda)*cos_theta;
        //u3_1qbit[3].imag = sin(Phi+Lambda)*cos_theta;


  return u3_1qbit;

}

/**
@brief Calculate the matrix of the constans gates.
@return Returns with the matrix of the one-qubit matrix.
*/
Matrix Gate::calc_one_qubit_u3( ) {

    std::string err("Gate::calc_one_qubit_u3: Unimplemented abstract function"); 
    throw err;   

    Matrix u3_1qbit = Matrix(2,2); 
    return u3_1qbit;

}

/**
@brief Set static values for the angles and constans parameters for calculating the matrix of the gates.
@param ThetaOver2 Real parameter standing for the parameter theta.
@param Phi Real parameter standing for the parameter phi.
@param Lambda Real parameter standing for the parameter lambda.
*/
void
Gate::parameters_for_calc_one_qubit(double& ThetaOver2, double& Phi, double& Lambda  ) {

 return;

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
@brief Call to extract parameters from the parameter array corresponding to the circuit, in which the gate is incorporated in.
@return Returns with the array of the extracted parameters.
*/
Matrix_real 
Gate::extract_parameters( Matrix_real& parameters ) {

    return Matrix_real(0,0);

}


/**
@brief Call to get the name label of the gate
@return Returns with the name label of the gate
*/
std::string 
Gate::get_name() {

    return name;

}

