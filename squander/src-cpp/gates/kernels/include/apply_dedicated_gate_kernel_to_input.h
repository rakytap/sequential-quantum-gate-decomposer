#ifndef APPLY_DEDICATED_GATE_KERNEL_TO_INPUT_H
#define APPLY_DEDICATED_GATE_KERNEL_TO_INPUT_H

#include "matrix.h"
#include "common.h"

/**
 * @brief Applies the X gate kernel to the input matrix.
 * 
 * @param input The input matrix on which the transformation is applied.
 * @param target_qbit The target qubit on which the transformation should be applied.
 * @param control_qbit1 The first control qubit (-1 if there is no control qubit).
 * @param control_qbit2 The second control qubit (-1 if there is no control qubit).
 * @param matrix_size The size of the input.
 */
void apply_X_kernel_to_input(Matrix& input, const int& target_qbit, const int& control_qbit1, const int& control_qbit2, const int& matrix_size);

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

void apply_SWAP_kernel_to_input(Matrix& input, const int& target_qbit1, const int& target_qbit2, const int& control_qbit, const int& matrix_size);


#endif // APPLY_DEDICATED_GATE_KERNEL_TO_INPUT_H