#include <iostream>
#include <random>
#include <cassert>
#include "matrix.h"
#include "Gates_block.h"
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

void testNQubitGate(int num_qubits) {

    Matrix state = generateRandomState(num_qubits);
    Matrix test_state = state.copy();

    Gates_block circuit(num_qubits);

    for (int i = 0; i < num_qubits; i++){
        circuit.add_u3(i);
    }
    for (int i = 0; i < num_qubits; i++){
        for (int j = i+1; j < num_qubits; j++){
            circuit.add_cry(i, j);
        }
    }
    
    int num_params = circuit.get_parameter_num();
    Matrix_real parameters(num_params, 1);
    for (int i = 0; i < num_params; i++) {
        parameters[i] = (i+1) / (M_PI*2);
    }

    Matrix identity = create_identity(1 << num_qubits);

    circuit.apply_to(parameters, identity);

    circuit.apply_to(parameters, state);
 
    switch (num_qubits) {
        case 2: apply_2qbit_kernel_to_state_vector_input(identity, test_state, 0, 1, 1 << num_qubits); break;
        case 3: apply_3qbit_kernel_to_state_vector_input(identity, test_state, {0,1,2}, 1 << num_qubits); break;
        case 4: apply_4qbit_kernel_to_state_vector_input(identity, test_state, {0,1,2,3}, 1 << num_qubits); break;
        case 5: apply_5qbit_kernel_to_state_vector_input(identity, test_state, {0,1,2,3,4}, 1 << num_qubits); break;
        default: throw std::invalid_argument("Unsupported number of qubits for state vector.");
    }

    double fid = fidelity(state, test_state);
    std::cout << num_qubits << "-qubit identity gate fidelity: " << fid << std::endl;
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

    circuit.apply_to(parameters, state);

    // circuit.set_max_fusion(0);
    circuit_inner->set_max_fusion(0);
    circuit_inner_2->set_max_fusion(0);

    circuit.apply_to(parameters, test_state);

    double fid = fidelity(state, test_state);
    std::cout << "Identity circuit fidelity (should be 1.0): " << fid << std::endl;
    assert(std::abs(fid - 1.0) < 1e-10);
}


};

int main() {
    ApplyKernelTestSuite suite;
    suite.testSmallCircuit();
    suite.testNQubitGate(2);
    suite.testNQubitGate(3);
    suite.testNQubitGate(4);
    suite.testNQubitGate(5);
    return 0;
}