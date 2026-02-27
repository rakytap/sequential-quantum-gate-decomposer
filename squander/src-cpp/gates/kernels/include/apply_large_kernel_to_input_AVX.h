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
/*! \file apply_large_kernel_to_input_AVX.h
    \brief Header file for AVX-optimized implementations for applying multi-qubit gate kernels to quantum state vectors and matrices
*/


#ifndef apply_large_kernel_to_input_AVX_H
#define apply_large_kernel_to_input_AVX_H

#include <immintrin.h>
#include "matrix.h"
#include "common.h"

#include "apply_large_kernel_to_input.h"

//AVX auxiliary functions

/**
@brief Helper function to load and prepare AVX vectors with outer and inner elements for complex multiplication
@param element_outer Pointer to the outer element (real and imaginary parts)
@param element_inner Pointer to the inner element (real and imaginary parts)
@return Prepared AVX vector for complex multiplication
*/
__m256d get_AVX_vector(double* element_outer, double* element_inner);

/**
@brief Perform complex multiplication using AVX intrinsics
@param input_vec AVX vector containing the input complex number (real and imaginary parts)
@param unitary_row_vec AVX vector containing the unitary matrix row element (real and imaginary parts)
@param neg AVX vector containing sign pattern for complex multiplication
@return AVX vector containing the result of complex multiplication
*/
__m256d complex_mult_AVX(__m256d input_vec, __m256d unitary_row_vec, __m256d neg);

//main function to call

/**
@brief Apply multi-qubit gate kernel to an input matrix using AVX optimization
@param unitary The 2^Nx2^N unitary matrix representing the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_large_kernel_to_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply multi-qubit gate kernel to an input matrix using AVX optimization and OpenMP parallelization
@param unitary The 2^Nx2^N unitary matrix representing the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_large_kernel_to_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply multi-qubit gate kernel to an input matrix using AVX optimization and TBB parallelization
@param unitary The 2^Nx2^N unitary matrix representing the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_large_kernel_to_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// general N qubit kernel functions

/**
@brief Compute the indices for a given block in the quantum state
@param N The number of qubits
@param tq The target qubits involved in the gate operation
@param non_targets The qubits not involved in the gate operation
@param iter_idx The current iteration index
@param indices The computed indices for the current block (output parameter)
*/
inline void get_block_indices(int N, const std::vector<int> &tq, const std::vector<int> &non_targets,int iter_idx, std::vector<int> &indices);

/**
@brief Efficiently compute the indices for a given block using precomputed patterns
@param iter_idx The current iteration index
@param target_qubits The qubits involved in the gate operation
@param non_targets The qubits not involved in the gate operation
@param block_pattern The precomputed index mapping
@param indices The computed indices for the current block (output parameter)
*/
inline void get_block_indices_fast(int iter_idx, const std::vector<int>& target_qubits, const std::vector<int>& non_targets, const std::vector<int>& block_pattern, std::vector<int>& indices);

/**
@brief Precompute the index mapping for target and non-target qubits
@param target_qubits The qubits involved in the gate operation
@param non_targets The qubits not involved in the gate operation
@param block_pattern The precomputed index mapping (output parameter)
*/
void precompute_index_mapping(const std::vector<int>& target_qubits, const std::vector<int>& non_targets, std::vector<int>& block_pattern);

/**
@brief Write the computed block back to the input matrix
@param input The input matrix to be updated
@param new_block_real The real parts of the new block
@param new_block_imag The imaginary parts of the new block
@param indices The indices where the new block should be written
*/
inline void write_out_block(Matrix& input, const std::vector<double>& new_block_real, const std::vector<double>& new_block_imag, const std::vector<int>& indices);

/**
@brief Perform complex multiplication and accumulation using AVX for a specific row and column
@param mv_xy Precomputed AVX vectors for the unitary matrix
@param rdx The row index of the unitary matrix
@param cdx The column index of the unitary matrix
@param indices The indices of the input matrix for the current block
@param input The input matrix
@param result The accumulated result of the multiplication (output parameter)
*/
inline void complex_prod_AVX(const __m256d* mv_xy, int rdx, int cdx,  const std::vector<int>& indices, const Matrix& input, __m256d& result);

/**
@brief Precompute AVX vectors for the unitary matrix to optimize complex multiplication
@param gate_kernel_unitary The unitary matrix of the gate operation
@param matrix_size The size of the unitary matrix
@return Pointer to the precomputed AVX vectors
*/
inline __m256d* construct_mv_xy_vectors(const Matrix& gate_kernel_unitary, const int& matrix_size);

/**
@brief Apply an n-qubit unitary operation to the input matrix using AVX optimization
@param gate_kernel_unitary The unitary matrix of the gate operation
@param input The input matrix to be transformed
@param involved_qbits The qubits involved in the operation
@param matrix_size The size of the input matrix
*/
void apply_nqbit_unitary_AVX( Matrix& gate_kernel_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size );

/**
@brief Apply multi-qubit gate kernel to an input matrix using AVX optimization and parallel processing
@param gate_kernel_unitary The 2^Nx2^N unitary matrix representing the gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_nqbit_unitary_parallel_AVX( Matrix& gate_kernel_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size );

// 2 qubit kernel functions

/**
@brief Apply two-qubit gate kernel to a state vector using AVX optimization
@param two_qbit_unitary The 4x4 unitary matrix representing the two-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (2 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_2qbit_kernel_to_state_vector_input_AVX(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply two-qubit gate kernel to a state vector using AVX optimization and OpenMP parallelization
@param two_qbit_unitary The 4x4 unitary matrix representing the two-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (2 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_2qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits,  const int& matrix_size);

/**
@brief Apply two-qubit gate kernel to a state vector using AVX optimization and TBB parallelization
@param two_qbit_unitary The 4x4 unitary matrix representing the two-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (2 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_2qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply two-qubit gate kernel to an input matrix using AVX optimization and OpenMP parallelization
@param two_qbit_unitary The 4x4 unitary matrix representing the two-qubit gate operation
@param input The input matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (2 qubits)
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_2qbit_kernel_to_matrix_input_parallel_AVX_OpenMP(Matrix& two_qbit_unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// 3 qubit kernel functions

/**
@brief Apply a 3-qubit quantum gate (unitary matrix) to a state vector using AVX intrinsics
@param unitary The 8x8 unitary matrix representing the 3-qubit gate
@param input The state vector to which the gate is applied
@param involved_qbits A vector of three integers indicating the qubit indices the gate acts on
@param matrix_size The size of the state vector (should be a power of 2)
*/
void apply_3qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply three-qubit gate kernel to a state vector using AVX optimization and OpenMP parallelization
@param unitary The 8x8 unitary matrix representing the three-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (3 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_3qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply three-qubit gate kernel to a state vector using AVX optimization and TBB parallelization
@param unitary The 8x8 unitary matrix representing the three-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (3 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_3qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// 4 qubit kernel functions

/**
@brief Apply four-qubit gate kernel to a state vector using AVX optimization
@param unitary The 16x16 unitary matrix representing the four-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (4 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_4qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply four-qubit gate kernel to a state vector using AVX optimization and OpenMP parallelization
@param unitary The 16x16 unitary matrix representing the four-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (4 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_4qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply four-qubit gate kernel to a state vector using AVX optimization and TBB parallelization
@param unitary The 16x16 unitary matrix representing the four-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (4 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_4qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// 5 qubit kernel functions

/**
@brief Apply five-qubit gate kernel to a state vector using AVX optimization
@param unitary The 32x32 unitary matrix representing the five-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (5 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_5qbit_kernel_to_state_vector_input_AVX(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply five-qubit gate kernel to a state vector using AVX optimization and OpenMP parallelization
@param unitary The 32x32 unitary matrix representing the five-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (5 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_5qbit_kernel_to_state_vector_input_AVX_OpenMP(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

/**
@brief Apply five-qubit gate kernel to a state vector using AVX optimization and TBB parallelization
@param unitary The 32x32 unitary matrix representing the five-qubit gate operation
@param input The input state vector matrix on which the transformation is applied
@param involved_qbits The qubits involved in the gate operation (5 qubits)
@param matrix_size The size of the input state vector (should be a power of 2)
*/
void apply_5qbit_kernel_to_state_vector_input_AVX_TBB(Matrix& unitary, Matrix& input, std::vector<int> involved_qbits, const int& matrix_size);

// CROT Kernels

/**
@brief Apply controlled rotation (CROT) kernel to a matrix input using AVX optimization and parallel processing
@param u3_1qbit1 The U3 gate matrix for the first qubit
@param u3_1qbit2 The U3 gate matrix for the second qubit
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit index
@param control_qbit The control qubit index
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_crot_kernel_to_matrix_input_AVX_parallel(Matrix& u3_1qbit1,Matrix& u3_1qbit2,Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);

/**
@brief Apply controlled rotation (CROT) kernel to a matrix input using AVX optimization
@param u3_1qbit1 The U3 gate matrix for the first qubit
@param u3_qbit2 The U3 gate matrix for the second qubit
@param input The input matrix on which the transformation is applied
@param target_qbit The target qubit index
@param control_qbit The control qubit index
@param matrix_size The size of the input matrix (should be a power of 2)
*/
void apply_crot_kernel_to_matrix_input_AVX(Matrix& u3_1qbit1, Matrix& u3_qbit2, Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
#endif
