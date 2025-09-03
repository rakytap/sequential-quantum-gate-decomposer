#include <iostream>
#include <random>
#include <cassert>
#include "matrix.h"
#include "common.h"
#include "apply_large_kernel_to_input.h"

class ApplyKernelTestSuite {
    std::mt19937 rng;
    
    Matrix generateRandomState(int num_qubits){
        int size = 1 << num_qubits;
        Matrix state(size, 1);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        double norm_sq = 0.0;
        for (int i = 0; i < size; i++) {
            state[i].real = dist(rng);
            state[i].imag = dist(rng);
            norm_sq += state[i].real*state[i].real + state[i].imag*state[i].imag;
        }
        
        double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (int i = 0; i < size; i++){
            state[i].real *= inv_norm;
            state[i].imag *= inv_norm;
        }
        return state;
    }

    double fidelity(const Matrix& s1, const Matrix& s2) {
        double real_sum = 0.0, imag_sum = 0.0;
        for (int i = 0; i < s1.rows; i++) {
            real_sum += s1[i].real*s2[i].real + s1[i].imag*s2[i].imag;
            imag_sum += s1[i].real*s2[i].imag - s1[i].imag*s2[i].real;
        }
        return std::sqrt(real_sum*real_sum + imag_sum*imag_sum);
    }

public:

void test2QubitGate() {
    int num_qubits = 3;
    Matrix state = generateRandomState(num_qubits);
    Matrix identity = create_identity(4);
    Matrix test_state = state.copy();

    apply_2qbit_kernel_to_state_vector_input(identity, test_state, 0, 1, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "2-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);
}

void test3QubitGate() {
    int num_qubits = 4;
    Matrix state = generateRandomState(num_qubits);
    Matrix identity = create_identity(8);
    Matrix test_state = state.copy();
    std::vector<int> qubits = {0,1,2};

    apply_3qbit_kernel_to_state_vector_input(identity, test_state, qubits, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "3-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);
}

void test4QubitGate() {
    int num_qubits = 4;
    Matrix state = generateRandomState(num_qubits);
    Matrix identity = create_identity(16);
    Matrix test_state = state.copy();
    std::vector<int> qubits = {0,1,2,3};

    apply_4qbit_kernel_to_state_vector_input(identity, test_state, qubits, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "4-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);
}

void test5QubitGate() {
    int num_qubits = 5;
    Matrix state = generateRandomState(num_qubits);
    Matrix identity = create_identity(32);
    Matrix test_state = state.copy();
    std::vector<int> qubits = {0,1,2,3,4};

    apply_5qbit_kernel_to_state_vector_input(identity, test_state, qubits, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "5-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);
}


};

int main() {
    ApplyKernelTestSuite suite;
    suite.test2QubitGate();
    suite.test3QubitGate();
    suite.test4QubitGate();
    suite.test5QubitGate();
    return 0;
}