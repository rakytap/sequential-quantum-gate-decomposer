#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include "matrix.h"
#include "Gates_block.h"
#include "common.h"
#include "apply_large_kernel_to_input.h"
#include "apply_large_kernel_to_input_AVX.h"
#include "apply_kernel_to_input.h"
#include "apply_dedicated_gate_kernel_to_input.h"
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

std::vector<std::vector<int>> generate_combinations(int n, int k) {
    std::vector<std::vector<int>> combinations;
    std::vector<bool> selector(n);
    std::fill(selector.end() - k, selector.end(), true);
    
    do {
        std::vector<int> combination;
        for (int i = 0; i < n; ++i) {
            if (selector[i]) {
                combination.push_back(i);
            }
        }
        // Ensure combination is sorted (should already be due to iteration order)
        std::sort(combination.begin(), combination.end());
        combinations.push_back(combination);
    } while (std::next_permutation(selector.begin(), selector.end()));
    
    // Sort combinations lexicographically for consistent ordering
    std::sort(combinations.begin(), combinations.end());
    return combinations;
}

public:

void test2QubitGate() {
    const int num_qubits = 10;
    const int k = 2;
    
    std::cout << "Testing all " << k << "-qubit gates on " << num_qubits << "-qubit system..." << std::endl;
    
    auto combinations = generate_combinations(num_qubits, k);
    std::cout << "Total combinations to test: " << combinations.size() << std::endl;
    
    int passed_regular = 0;
    int failed_regular = 0;
    int passed_avx = 0;
    int failed_avx = 0;
    
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        
        Matrix state = generateRandomState(num_qubits);
        Matrix test_state = state.copy();
        Matrix test_state_avx = state.copy();
        Matrix_real params = Matrix_real(1, 1);
        
        Matrix Umtx = create_identity(1 << k);
        memset(params.get_data(), 0.0, params.size() * sizeof(double));
        
        // Apply GHZ circuit to full system
        Gates_block GHZ_circ = Gates_block(num_qubits);
        GHZ_circ.add_h(qubits[0]);
        GHZ_circ.add_cnot(qubits[1], qubits[0]);
        GHZ_circ.apply_to(params, state);
        
        // Create corresponding unitary matrix
        Gates_block GHZ_circ_mini = Gates_block(k);
        GHZ_circ_mini.add_h(0);
        GHZ_circ_mini.add_cnot(1, 0);
        GHZ_circ_mini.apply_to(params, Umtx);
        
        // Test regular kernel
        apply_2qbit_kernel_to_state_vector_input(Umtx, test_state, qubits[0], qubits[1], 1 << num_qubits);
        
        double fid = fidelity(state, test_state);
        
        if (std::abs(fid - 1.0) < 1e-10) {
            passed_regular++;
        } else {
            failed_regular++;
            std::cout << "REGULAR FAILED: Qubits {" << qubits[0] << "," << qubits[1] 
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
        
        #ifdef USE_AVX
        // Test AVX kernel
        apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
        double fid_avx = fidelity(state, test_state_avx);
        
        if (std::abs(fid_avx - 1.0) < 1e-10) {
            passed_avx++;
        } else {
            failed_avx++;
            std::cout << "AVX FAILED: Qubits {" << qubits[0] << "," << qubits[1] 
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid_avx << std::endl;
        }
        #endif
    }
    
    std::cout << "2-qubit gate test results:" << std::endl;
    std::cout << "  Regular kernel: " << passed_regular << " passed, " << failed_regular << " failed" << std::endl;
    #ifdef USE_AVX
    std::cout << "  AVX kernel:     " << passed_avx << " passed, " << failed_avx << " failed" << std::endl;
    #endif
    
    assert(failed_regular == 0);
    #ifdef USE_AVX
    assert(failed_avx == 0);
    #endif
}

void test3QubitGate() {
    const int num_qubits = 10;
    const int k = 3;
    
    std::cout << "Testing all " << k << "-qubit gates on " << num_qubits << "-qubit system..." << std::endl;
    
    auto combinations = generate_combinations(num_qubits, k);
    std::cout << "Total combinations to test: " << combinations.size() << std::endl;
    
    int passed_regular = 0;
    int failed_regular = 0;
    int passed_avx = 0;
    int failed_avx = 0;
    
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        
        Matrix state = generateRandomState(num_qubits);
        Matrix test_state = state.copy();
        Matrix test_state_avx = state.copy();
        Matrix_real params = Matrix_real(1, 1);
        
        Matrix Umtx = create_identity(1 << k);
        memset(params.get_data(), 0.0, params.size() * sizeof(double));
        
        // Apply GHZ circuit to full system
        Gates_block GHZ_circ = Gates_block(num_qubits);
        GHZ_circ.add_h(qubits[0]);
        GHZ_circ.add_cnot(qubits[1], qubits[0]);
        GHZ_circ.add_cnot(qubits[2], qubits[0]);
        GHZ_circ.apply_to(params, state);
        
        // Create corresponding unitary matrix
        Gates_block GHZ_circ_mini = Gates_block(k);
        GHZ_circ_mini.add_h(0);
        GHZ_circ_mini.add_cnot(1, 0);
        GHZ_circ_mini.add_cnot(2, 0);
        GHZ_circ_mini.apply_to(params, Umtx);
        
        // Test regular kernel
        apply_3qbit_kernel_to_state_vector_input(Umtx, test_state, qubits, 1 << num_qubits);
        
        double fid = fidelity(state, test_state);
        
        if (std::abs(fid - 1.0) < 1e-10) {
            passed_regular++;
        } else {
            failed_regular++;
            std::cout << "REGULAR FAILED: Qubits {" << qubits[0] << "," << qubits[1] << "," << qubits[2]
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
        
        #ifdef USE_AVX
        // Test AVX kernel
        apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
        double fid_avx = fidelity(state, test_state_avx);
        
        if (std::abs(fid_avx - 1.0) < 1e-10) {
            passed_avx++;
        } else {
            failed_avx++;
            std::cout << "AVX FAILED: Qubits {" << qubits[0] << "," << qubits[1] << "," << qubits[2]
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid_avx << std::endl;
        }
        #endif
    }
    
    std::cout << "3-qubit gate test results:" << std::endl;
    std::cout << "  Regular kernel: " << passed_regular << " passed, " << failed_regular << " failed" << std::endl;
    #ifdef USE_AVX
    std::cout << "  AVX kernel:     " << passed_avx << " passed, " << failed_avx << " failed" << std::endl;
    #endif
    
    assert(failed_regular == 0);
    #ifdef USE_AVX
    assert(failed_avx == 0);
    #endif
}

void test4QubitGate() {
    const int num_qubits = 10;
    const int k = 4;
    
    std::cout << "Testing all " << k << "-qubit gates on " << num_qubits << "-qubit system..." << std::endl;
    
    auto combinations = generate_combinations(num_qubits, k);
    std::cout << "Total combinations to test: " << combinations.size() << std::endl;
    
    int passed_regular = 0;
    int failed_regular = 0;
    int passed_avx = 0;
    int failed_avx = 0;
    
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        
        Matrix state = generateRandomState(num_qubits);
        Matrix test_state = state.copy();
        Matrix test_state_avx = state.copy();
        Matrix_real params = Matrix_real(1, 1);
        
        Matrix Umtx = create_identity(1 << k);
        memset(params.get_data(), 0.0, params.size() * sizeof(double));
        
        // Apply GHZ circuit to full system
        Gates_block GHZ_circ = Gates_block(num_qubits);
        GHZ_circ.add_h(qubits[0]);
        for (int i = 1; i < k; ++i) {
            GHZ_circ.add_cnot(qubits[i], qubits[0]);
        }
        GHZ_circ.apply_to(params, state);
        
        // Create corresponding unitary matrix
        Gates_block GHZ_circ_mini = Gates_block(k);
        GHZ_circ_mini.add_h(0);
        for (int i = 1; i < k; ++i) {
            GHZ_circ_mini.add_cnot(i, 0);
        }
        GHZ_circ_mini.apply_to(params, Umtx);
        
        // Test regular kernel
        apply_4qbit_kernel_to_state_vector_input(Umtx, test_state, qubits, 1 << num_qubits);
        
        double fid = fidelity(state, test_state);
        
        if (std::abs(fid - 1.0) < 1e-10) {
            passed_regular++;
        } else {
            failed_regular++;
            std::cout << "REGULAR FAILED: Qubits {";
            for (size_t i = 0; i < qubits.size(); ++i) {
                std::cout << qubits[i];
                if (i < qubits.size() - 1) std::cout << ",";
            }
            std::cout << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
        #ifdef USE_AVX
        // Test AVX kernel
        apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
        double fid_avx = fidelity(state, test_state_avx);
        
        if (std::abs(fid_avx - 1.0) < 1e-10) {
            passed_avx++;
        } else {
            failed_avx++;
            std::cout << "AVX FAILED: Qubits {";
            for (size_t i = 0; i < qubits.size(); ++i) {
                std::cout << qubits[i];
                if (i < qubits.size() - 1) std::cout << ",";
            }
            std::cout << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid_avx << std::endl;
        }
        #endif
        
    }
    
    std::cout << "4-qubit gate test results:" << std::endl;
    std::cout << "  Regular kernel: " << passed_regular << " passed, " << failed_regular << " failed" << std::endl;
    #ifdef USE_AVX
    std::cout << "  AVX kernel:     " << passed_avx << " passed, " << failed_avx << " failed" << std::endl;
    #endif
    
    assert(failed_regular == 0);
    #ifdef USE_AVX
    assert(failed_avx == 0);
    #endif
}

void test5QubitGate() {
    const int num_qubits = 10;
    const int k = 5;
    
    std::cout << "Testing all " << k << "-qubit gates on " << num_qubits << "-qubit system..." << std::endl;
    
    auto combinations = generate_combinations(num_qubits, k);
    std::cout << "Total combinations to test: " << combinations.size() << std::endl;
    
    int passed_regular = 0;
    int failed_regular = 0;
    int passed_avx = 0;
    int failed_avx = 0;
    
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        
        Matrix state = generateRandomState(num_qubits);
        Matrix test_state = state.copy();
        Matrix test_state_avx = state.copy();
        Matrix_real params = Matrix_real(1, 1);
        
        Matrix Umtx = create_identity(1 << k);
        memset(params.get_data(), 0.0, params.size() * sizeof(double));
        
        // Apply GHZ circuit to full system
        Gates_block GHZ_circ = Gates_block(num_qubits);
        GHZ_circ.add_h(qubits[0]);
        for (int i = 1; i < k; ++i) {
            GHZ_circ.add_cnot(qubits[i], qubits[0]);
        }
        GHZ_circ.apply_to(params, state);
        
        // Create corresponding unitary matrix
        Gates_block GHZ_circ_mini = Gates_block(k);
        GHZ_circ_mini.add_h(0);
        for (int i = 1; i < k; ++i) {
            GHZ_circ_mini.add_cnot(i, 0);
        }
        GHZ_circ_mini.apply_to(params, Umtx);
        
        // Test regular kernel
        apply_5qbit_kernel_to_state_vector_input(Umtx, test_state, qubits, 1 << num_qubits);
        
        double fid = fidelity(state, test_state);
        
        if (std::abs(fid - 1.0) < 1e-10) {
            passed_regular++;
        } else {
            failed_regular++;
            std::cout << "REGULAR FAILED: Qubits {";
            for (size_t i = 0; i < qubits.size(); ++i) {
                std::cout << qubits[i];
                if (i < qubits.size() - 1) std::cout << ",";
            }
            std::cout << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
        
        #ifdef USE_AVX
        // Test AVX kernel
        apply_large_kernel_to_input_AVX(Umtx, test_state_avx, qubits, 1 << num_qubits);
        double fid_avx = fidelity(state, test_state_avx);
        
        if (std::abs(fid_avx - 1.0) < 1e-10) {
            passed_avx++;
        } else {
            failed_avx++;
            std::cout << "AVX FAILED: Qubits {";
            for (size_t i = 0; i < qubits.size(); ++i) {
                std::cout << qubits[i];
                if (i < qubits.size() - 1) std::cout << ",";
            }
            std::cout << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid_avx << std::endl;
        }
        #endif
        

    }
    
    std::cout << "5-qubit gate test results:" << std::endl;
    std::cout << "  Regular kernel: " << passed_regular << " passed, " << failed_regular << " failed" << std::endl;
    #ifdef USE_AVX
    std::cout << "  AVX kernel:     " << passed_avx << " passed, " << failed_avx << " failed" << std::endl;
    #endif
    
    assert(failed_regular == 0);
    #ifdef USE_AVX
    assert(failed_avx == 0);
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

void testSmallCircuit() {
    int num_qubits = 4;

    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();

    Gates_block circuit(num_qubits);
    Gates_block* circuit_inner = new Gates_block(num_qubits);
    Gates_block* circuit_inner_2 = new Gates_block(num_qubits);
    
    circuit_inner->add_u3(2);
    circuit_inner->add_u3(0);
    circuit_inner->add_u3(1);

    circuit_inner->add_cry(0, 1);
    circuit_inner->add_cry(1, 2);
    circuit_inner_2->add_cry(2, 3);

    circuit_inner_2->add_u3(3);
    circuit_inner_2->add_u3(0);

    // circuit.fragment_circuit();
    circuit.add_gate(circuit_inner);
    circuit.add_gate(circuit_inner_2);

    int num_params = circuit.get_parameter_num();
    Matrix_real parameters(num_params, 1);
    for (int i = 0; i < num_params; i++) {
        parameters[i] = (i+1) / (M_PI*2);
    }

    circuit.set_min_fusion(-1);
    circuit_inner->set_min_fusion(-1);
    circuit_inner_2->set_min_fusion(-1);

    circuit.apply_to(parameters, state);

    circuit.set_min_fusion(0);
    circuit_inner->set_min_fusion(0);
    circuit_inner_2->set_min_fusion(0);

    circuit.apply_to(parameters, test_state);

    double fid = fidelity(state, test_state);
    std::cout << "Identity circuit fidelity (should be 1.0): " << fid << std::endl;
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
        /*double dedicated_kernel_time = 0.0;
        tbb::tick_count dedicated_kernel_start = tbb::tick_count::now();
        for (int idx=0; idx<samples;idx++){
            apply_large_kernel_to_input(Umtx, test_state, qubits, 1 << num_qubits);
        }
        tbb::tick_count dedicated_kernel_end = tbb::tick_count::now();
        dedicated_kernel_time  = (dedicated_kernel_end-dedicated_kernel_start).seconds()/samples;
        std::cout << qubits.size()<<" qubit dedicated kernel time "<< dedicated_kernel_time << std::endl;*/


        #ifdef USE_AVX
        double dedicated_kernel_AVX_time = 0.0;
        tbb::tick_count dedicated_kernel_AVX_start = tbb::tick_count::now();
        for (int idx=0; idx<samples;idx++){
            apply_large_kernel_to_input_AVX_TBB(Umtx, test_state, qubits, 1 << num_qubits);
        }
        tbb::tick_count dedicated_kernel_AVX_end = tbb::tick_count::now();

        dedicated_kernel_AVX_time  = (dedicated_kernel_AVX_time + (dedicated_kernel_AVX_end-dedicated_kernel_AVX_start).seconds())/samples;

        std::cout << qubits.size()<<" qubit dedicated AVX kernel time "<< dedicated_kernel_AVX_time << std::endl;
        #endif
    }
}

void testXgateKernel(){

    //test single qubit X gate
    int num_qubits = 10;
    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();
    Matrix test_state2 = state.copy();
    auto combinations = generate_combinations(num_qubits, 1);
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        std::vector<int> target_qbits = {qubits[0]};
        std::vector<int> control_qbits = {};
        apply_X_kernel_to_input(test_state, target_qbits, control_qbits, 1 << num_qubits);
        Gates_block X_gate = Gates_block(num_qubits);
        X_gate.add_x(qubits[0]);
        Matrix_real params = Matrix_real(1, 1);
        X_gate.apply_to(params, test_state2);
        double fid = fidelity(test_state, test_state2);
        if (std::abs(fid - 1.0) >= 1e-10) {
            std::cout << "X gate FAILED: Qubit {" << qubits[0] 
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
        assert(std::abs(fid - 1.0) < 1e-10);

    }
    //test 2 qubit X gate
    combinations = generate_combinations(num_qubits, 2);
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        std::vector<int> target_qbits = {qubits[0]};
        std::vector<int> control_qbits = {qubits[1]};
        apply_X_kernel_to_input(test_state, target_qbits, control_qbits, 1 << num_qubits);
        Gates_block X_gate = Gates_block(num_qubits);
        X_gate.add_cnot(qubits[0],qubits[1]);
        Matrix_real params = Matrix_real(1, 1);
        X_gate.apply_to(params, test_state2);
        double fid = fidelity(test_state, test_state2);
        if (std::abs(fid - 1.0) >= 1e-10) {
            std::cout << "2 qubit X gate FAILED: Qubits {" << qubits[0] << "," << qubits[1]
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
    }
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        std::vector<int> target_qbits = {qubits[1]};
        std::vector<int> control_qbits = {qubits[0]};
        apply_X_kernel_to_input(test_state, target_qbits, control_qbits, 1 << num_qubits);
        Gates_block X_gate = Gates_block(num_qubits);
        X_gate.add_cnot(qubits[1],qubits[0]);
        Matrix_real params = Matrix_real(1, 1);
        X_gate.apply_to(params, test_state2);
        double fid = fidelity(test_state, test_state2);
        if (std::abs(fid - 1.0) >= 1e-10) {
            std::cout << "2 qubit X gate FAILED: Qubits {" << qubits[0] << "," << qubits[1]
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
    }
    int failed = 0;
    // test troffoli X gate
    combinations = generate_combinations(num_qubits, 3);
    for (size_t combo_idx = 0; combo_idx < combinations.size(); ++combo_idx) {
        const auto& qubits = combinations[combo_idx];
        std::vector<int> target_qbits = {qubits[0]};
        std::vector<int> control_qbits = {qubits[1], qubits[2]};
        apply_X_kernel_to_input(test_state, target_qbits, control_qbits, 1 << num_qubits);
        Matrix Umtx = create_identity(8);
        Umtx[6*8 + 6].real = 0.0;
        Umtx[7*8 + 7].real = 0.0;
        Umtx[6*8 + 7].real = 1.0;
        Umtx[7*8 + 6].real = 1.0;
        apply_3qbit_kernel_to_state_vector_input(Umtx, test_state2, qubits, 1 << num_qubits);
        double fid = fidelity(test_state, test_state2);
        if (std::abs(fid - 1.0) >= 1e-10) {
            failed++;
            std::cout << "Toffoli X gate FAILED: Qubits {" << qubits[0] << "," << qubits[1] << "," << qubits[2]
                      << "} - Fidelity: " << std::fixed << std::setprecision(12) << fid << std::endl;
        }
    }
    std::cout << "Toffoli X gate failed cases: " << failed/combinations.size() << std::endl;
}


};
int main() {
    ApplyKernelTestSuite suite;
    suite.test2QubitGate();
    suite.test3QubitGate();
    suite.test4QubitGate();
    suite.test5QubitGate();
    #ifdef USE_AVX
    //suite.testNQubitGate_Parallel_GHZ();
    //suite.testNQubit_Gate_speed();
    #endif
    suite.testSmallCircuit();
    suite.testXgateKernel();
    return 0;
}
