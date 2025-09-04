#include <iostream>
#include <random>
#include <cassert>
#include "matrix.h"
#include "common.h"
#include "apply_large_kernel_to_input.h"
#include "apply_large_kernel_to_input_AVX.h"
#include "Gates_block.h"



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

void test5QubitGate_Parallel_GHZ() {
    int num_qubits = 20;
    Matrix state = generateRandomState(num_qubits);
    Matrix_real params = Matrix_real(1,1);
    Matrix test_state = state.copy();
    Matrix test_state2 = state.copy();
    std::vector<int> qubits = {1,2,3,4};
    Matrix Umtx = create_identity(1<<qubits.size());
    memset(params.get_data(), 0.0, params.size()*sizeof(double) );  
    
    Gates_block GHZ_circ = Gates_block(20);
    GHZ_circ.add_h(1);
    GHZ_circ.add_cnot(2,1);
    GHZ_circ.add_cnot(3,1);
    GHZ_circ.add_cnot(4,1);
    GHZ_circ.apply_to(params,state);
    
    Gates_block GHZ_circ_mini = Gates_block(4);
    GHZ_circ_mini.add_h(0);
    GHZ_circ_mini.add_cnot(1,0);
    GHZ_circ_mini.add_cnot(2,0);
    GHZ_circ_mini.add_cnot(3,0);
    GHZ_circ_mini.apply_to(params,Umtx);
    
    double nqbit_kernel_time = 0.0;
    tbb::tick_count nqbit_kernel_start = tbb::tick_count::now();
    
    apply_nqbit_unitary_AVX(Umtx, test_state, qubits, 1 << num_qubits);
    
    tbb::tick_count nqbit_kernel_end = tbb::tick_count::now();
    nqbit_kernel_time  = nqbit_kernel_time + (nqbit_kernel_end-nqbit_kernel_start).seconds();
    
    double fid = fidelity(state, test_state);
    std::cout << num_qubits <<"-qubit GHZ gate fidelity: " << fid << std::endl;
    
    double fqbit_kernel_time = 0.0;
    tbb::tick_count fqbit_kernel_start = tbb::tick_count::now();
    
    apply_4qbit_kernel_to_state_vector_input_parallel_AVX(Umtx, test_state2, qubits, 1 << num_qubits);
    
    tbb::tick_count fqbit_kernel_end = tbb::tick_count::now();
    fqbit_kernel_time  = fqbit_kernel_time + (fqbit_kernel_end-fqbit_kernel_start).seconds();
    
    double fid2 = fidelity(state, test_state2);
    std::cout << num_qubits <<"-qubit GHZ gate fidelity: " << fid2 << std::endl;
    
    std::cout << "4 qubit kernel simulation time: " << fqbit_kernel_time <<" N qubit kernel simulation time: " << nqbit_kernel_time<< std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);

}



};

int main() {
    ApplyKernelTestSuite suite;
    suite.test2QubitGate();
    suite.test3QubitGate();
    suite.test4QubitGate();
    suite.test5QubitGate();
    suite.test5QubitGate_Parallel_GHZ();
    return 0;
}
