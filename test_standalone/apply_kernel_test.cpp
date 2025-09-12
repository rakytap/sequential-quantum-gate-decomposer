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
    int num_qubits = 6;
    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();
    Matrix test_state_avx = state.copy();
    Matrix_real params = Matrix_real(1,1);

    std::vector<int> qubits = {0,3};
    Matrix Umtx = create_identity(1<<qubits.size());
    memset(params.get_data(), 0.0, params.size()*sizeof(double) );  
    
    Gates_block GHZ_circ = Gates_block(6);
    GHZ_circ.add_h(0);
    GHZ_circ.add_cnot(3,0);
    GHZ_circ.apply_to(params,state);
    
    Gates_block GHZ_circ_mini = Gates_block(2);
    GHZ_circ_mini.add_h(0);
    GHZ_circ_mini.add_cnot(1,0);
    GHZ_circ_mini.apply_to(params,Umtx);

    apply_2qbit_kernel_to_state_vector_input(Umtx, test_state, 0, 3, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "2-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);

    #ifdef USE_AVX 
    apply_2qbit_kernel_to_state_vector_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
    double fid_avx = fidelity(state, test_state_avx);
    std::cout << "2-qubit identity gate fidelity AVX: " << fid_avx << std::endl;
    #endif
}

void test3QubitGate() {
    int num_qubits = 10;
    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();
    Matrix test_state_avx = state.copy();
    Matrix_real params = Matrix_real(1,1);

    std::vector<int> qubits = {0,4,7};
    Matrix Umtx = create_identity(1<<qubits.size());
    memset(params.get_data(), 0.0, params.size()*sizeof(double) );  

    // Prepare a GHZ-like circuit for 3 qubits
    Gates_block GHZ_circ = Gates_block(num_qubits);
    GHZ_circ.add_h(qubits[0]);
    GHZ_circ.add_cnot(qubits[1], qubits[0]);
    GHZ_circ.add_cnot(qubits[2], qubits[0]);
    GHZ_circ.apply_to(params, state);

    Gates_block GHZ_circ_mini = Gates_block(3);
    GHZ_circ_mini.add_h(0);
    GHZ_circ_mini.add_cnot(1,0);
    GHZ_circ_mini.add_cnot(2,0);
    GHZ_circ_mini.apply_to(params, Umtx);

    apply_3qbit_kernel_to_state_vector_input(Umtx, test_state, qubits, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "3-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);

    #ifdef USE_AVX 
    apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
    double fid_avx = fidelity(state, test_state_avx);
    std::cout << "3-qubit identity gate fidelity AVX: " << fid_avx << std::endl;
    #endif
}

void test4QubitGate() {
    int num_qubits = 20;
    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();
    Matrix test_state_avx = state.copy();
    Matrix_real params = Matrix_real(1,1);

    std::vector<int> qubits = {0,4,6,9};
    Matrix Umtx = create_identity(1<<qubits.size());
    memset(params.get_data(), 0.0, params.size()*sizeof(double) );  

    // Prepare a GHZ-like circuit for 4 qubits
    Gates_block GHZ_circ = Gates_block(num_qubits);
    GHZ_circ.add_h(qubits[0]);
    GHZ_circ.add_cnot(qubits[1], qubits[0]);
    GHZ_circ.add_cnot(qubits[2], qubits[0]);
    GHZ_circ.add_cnot(qubits[3], qubits[0]);
    GHZ_circ.apply_to(params, state);

    Gates_block GHZ_circ_mini = Gates_block(4);
    GHZ_circ_mini.add_h(0);
    GHZ_circ_mini.add_cnot(1,0);
    GHZ_circ_mini.add_cnot(2,0);
    GHZ_circ_mini.add_cnot(3,0);
    GHZ_circ_mini.apply_to(params, Umtx);

    apply_4qbit_kernel_to_state_vector_input(Umtx, test_state, qubits, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "4-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);

    #ifdef USE_AVX 
    apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
    double fid_avx = fidelity(state, test_state_avx);
    std::cout << "4-qubit identity gate fidelity AVX: " << fid_avx << std::endl;
    #endif
}

void test5QubitGate() {
    int num_qubits = 10;
    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();
    Matrix test_state_avx = state.copy();
    Matrix_real params = Matrix_real(1,1);

    std::vector<int> qubits = {1,3,5,7,9};
    Matrix Umtx = create_identity(1<<qubits.size());
    memset(params.get_data(), 0.0, params.size()*sizeof(double) );  

    // Prepare a GHZ-like circuit for 5 qubits
    Gates_block GHZ_circ = Gates_block(num_qubits);
    GHZ_circ.add_h(qubits[0]);
    GHZ_circ.add_cnot(qubits[1], qubits[0]);
    GHZ_circ.add_cnot(qubits[2], qubits[0]);
    GHZ_circ.add_cnot(qubits[3], qubits[0]);
    GHZ_circ.add_cnot(qubits[4], qubits[0]);
    GHZ_circ.apply_to(params, state);

    Gates_block GHZ_circ_mini = Gates_block(5);
    GHZ_circ_mini.add_h(0);
    GHZ_circ_mini.add_cnot(1,0);
    GHZ_circ_mini.add_cnot(2,0);
    GHZ_circ_mini.add_cnot(3,0);
    GHZ_circ_mini.add_cnot(4,0);
    GHZ_circ_mini.apply_to(params, Umtx);

    apply_5qbit_kernel_to_state_vector_input(Umtx, test_state, qubits, 1 << num_qubits);

    double fid = fidelity(state, test_state);
    std::cout << "5-qubit identity gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);

    #ifdef USE_AVX 
    apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
    double fid_avx = fidelity(state, test_state_avx);
    std::cout << "5-qubit identity gate fidelity AVX: " << fid_avx << std::endl;
    #endif 
}

void testNQubitGate_Parallel_GHZ() {
    int num_qubits = 10;
    Matrix state = generateRandomState(num_qubits);
    Matrix_real params = Matrix_real(1,1);
    Matrix test_state = state.copy();
    Matrix test_state2 = state.copy();
    std::vector<int> qubits = {0,4,7};
    Matrix Umtx = create_identity(1<<qubits.size());
    memset(params.get_data(), 0.0, params.size()*sizeof(double) );  
    
    Gates_block GHZ_circ = Gates_block(10);
    GHZ_circ.add_h(qubits[0]);
    GHZ_circ.add_cnot(qubits[1], qubits[0]);
    GHZ_circ.add_cnot(qubits[2], qubits[0]);
    GHZ_circ.apply_to(params,state);
    
    Gates_block GHZ_circ_mini = Gates_block(3);
    GHZ_circ_mini.add_h(0);
    GHZ_circ_mini.add_cnot(1,0);
    GHZ_circ_mini.add_cnot(2,0);

    GHZ_circ_mini.apply_to(params,Umtx);
    apply_nqbit_unitary_AVX(Umtx, test_state, qubits, 1 << num_qubits);
    double fid = fidelity(state, test_state);
    std::cout << num_qubits <<"-qubit GHZ gate fidelity: " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);

}

void testNQubit_Gate_speed() {
    int num_qubits = 20;
    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();
    Matrix test_state2 = state.copy();
    int offset = 4;
    for (int n=2; n<6; n++){
        std::vector<int> qubits;
        for (int qubit=offset;qubit<offset+n;qubit++){
            qubits.push_back(qubit);
        }
        Matrix Umtx = create_identity(1<<qubits.size());

        int samples = 500;
        double dedicated_kernel_time = 0.0;
        tbb::tick_count dedicated_kernel_start = tbb::tick_count::now();
        for (int idx=0; idx<samples;idx++){
            apply_large_kernel_to_input(Umtx, test_state, qubits, 1 << num_qubits);
        }
        tbb::tick_count dedicated_kernel_end = tbb::tick_count::now();
        dedicated_kernel_time  = (dedicated_kernel_end-dedicated_kernel_start).seconds()/samples;
        std::cout << qubits.size()<<" qubit dedicated kernel time "<< dedicated_kernel_time << std::endl;


        #ifdef USE_AVX
        double dedicated_kernel_AVX_time = 0.0;
        tbb::tick_count dedicated_kernel_AVX_start = tbb::tick_count::now();
        for (int idx=0; idx<samples;idx++){
            apply_large_kernel_to_input_AVX(Umtx, test_state, qubits, 1 << num_qubits);
        }
        tbb::tick_count dedicated_kernel_AVX_end = tbb::tick_count::now();

        dedicated_kernel_AVX_time  = (dedicated_kernel_AVX_time + (dedicated_kernel_AVX_end-dedicated_kernel_AVX_start).seconds())/samples;

        std::cout << qubits.size()<<" qubit dedicated AVX kernel time "<< dedicated_kernel_AVX_time << std::endl;
        #endif
    }
}


};

int main() {
    ApplyKernelTestSuite suite;
    suite.test2QubitGate();
    suite.test3QubitGate();
    suite.test4QubitGate();
    suite.test5QubitGate();
    #ifdef USE_AVX
    suite.testNQubitGate_Parallel_GHZ();
    //suite.testNQubit_Gate_speed();
    #endif
    return 0;
}
