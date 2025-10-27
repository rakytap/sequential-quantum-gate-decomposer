#ifndef APPLY_DEDICATED_GATE_KERNEL_TO_INPUT_H
#define APPLY_DEDICATED_GATE_KERNEL_TO_INPUT_H

#include "matrix.h"
#include "common.h"

/**
 * @brief Applies the X gate kernel to the input matrix.
 *
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbits The target qubits (must contain exactly 1 element).
 * @param control_qbits The control qubits (empty vector if no control qubits).
 * @param matrix_size The size of the input.
 */
void apply_X_kernel_to_input(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size);

/**
 * @brief Applies the Y gate kernel to the input matrix.
 * 
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbit The target qubit on which the transformation should be applied.
 * @param control_qbit The control qubit (-1 if there is no control qubit).
 * @param matrix_size The size of the input.
 */
void apply_Y_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);

/**
 * @brief Applies the Z gate kernel to the input matrix.
 * 
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbit The target qubit on which the transformation should be applied.
 * @param control_qbit The control qubit (-1 if there is no control qubit).
 * @param matrix_size The size of the input.
 */
void apply_Z_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);

/**
 * @brief Applies the H (Hadamard) gate kernel to the input matrix.
 * 
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbit The target qubit on which the transformation should be applied.
 * @param control_qbit The control qubit (-1 if there is no control qubit).
 * @param matrix_size The size of the input.
 */
void apply_H_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);

/**
 * @brief Applies the S gate kernel to the input matrix.
 * 
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbit The target qubit on which the transformation should be applied.
 * @param control_qbit The control qubit (-1 if there is no control qubit).
 * @param matrix_size The size of the input.
 */
void apply_S_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);

/**
 * @brief Applies the T gate kernel to the input matrix.
 * 
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbit The target qubit on which the transformation should be applied.
 * @param control_qbit The control qubit (-1 if there is no control qubit).
 * @param matrix_size The size of the input.
 */
void apply_T_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);

/**
 * @brief Applies the SWAP gate kernel to the input matrix.
 *
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbits The target qubits (must contain exactly 2 elements).
 * @param control_qbits The control qubits (empty vector if no control qubits).
 * @param matrix_size The size of the input.
 */
void apply_SWAP_kernel_to_input(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size);

/**
 * @brief Applies the Permutation gate kernel to the input matrix.
 *
 * @param input The input matrix on which the transformation is applied.
 * @param pattern The pattern of the permutation.
 * @param matrix_size The size of the input.
 */
void apply_Permutation_kernel_to_input(Matrix& input, const std::vector<int>& pattern, const int& matrix_size);

// TBB Parallelized versions
void apply_X_kernel_to_input_tbb(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size);
void apply_Y_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_Z_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_H_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_S_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_T_kernel_to_input_tbb(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_SWAP_kernel_to_input_tbb(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size);

// OpenMP Parallelized versions
void apply_X_kernel_to_input_omp(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size);
void apply_Y_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_Z_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_H_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_S_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_T_kernel_to_input_omp(Matrix& input, const int& target_qbit, const int& control_qbit, const int& matrix_size);
void apply_SWAP_kernel_to_input_omp(Matrix& input, const std::vector<int>& target_qbits, const std::vector<int>& control_qbits, const int& matrix_size);

#endif // APPLY_DEDICATED_GATE_KERNEL_TO_INPUT_H