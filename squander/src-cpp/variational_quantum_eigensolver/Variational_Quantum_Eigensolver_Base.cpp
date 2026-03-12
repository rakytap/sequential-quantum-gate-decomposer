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
/*! \file Variational_Quantum_Eigensolver_Base.cpp
    \brief Class to solve VQE problems
*/
#include "Variational_Quantum_Eigensolver_Base.h"

#include "../density_matrix/include/density_matrix.h"
#include "../density_matrix/include/noisy_circuit.h"

#include <nlohmann/json.hpp>
#include <random>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <tbb/tbb.h>
#include "../../gates/include/H.h"
#include "../../gates/include/SDG.h"
using json = nlohmann::json;

static tbb::spin_mutex my_mutex;

namespace {

const char* VQE_BACKEND_MODE_CONFIG_KEY = "backend_mode";

const char*
density_noise_type_to_label(density_noise_type noise_type) {
    switch (noise_type) {
        case LOCAL_DEPOLARIZING_NOISE:
            return "local_depolarizing";
        case AMPLITUDE_DAMPING_NOISE:
            return "amplitude_damping";
        case PHASE_DAMPING_NOISE:
            return "phase_damping";
        default:
            return "unknown_density_noise";
    }
}

const char*
vqe_circuit_source_to_label(vqe_circuit_source_type circuit_source) {
    switch (circuit_source) {
        case VQE_CIRCUIT_SOURCE_UNSET:
            return "unset";
        case VQE_CIRCUIT_SOURCE_GENERATED_HEA:
            return "generated_hea";
        case VQE_CIRCUIT_SOURCE_GENERATED_HEA_ZYZ:
            return "generated_hea_zyz";
        case VQE_CIRCUIT_SOURCE_BINARY_IMPORT:
            return "binary_import";
        case VQE_CIRCUIT_SOURCE_CUSTOM_GATE_STRUCTURE:
            return "custom_gate_structure";
        default:
            return "unknown_circuit_source";
    }
}

vqe_backend_type
parse_vqe_backend_mode(std::map<std::string, Config_Element>& config) {

    long long backend_mode_value = STATE_VECTOR_BACKEND;
    if (config.count(VQE_BACKEND_MODE_CONFIG_KEY) > 0) {
        config[VQE_BACKEND_MODE_CONFIG_KEY].get_property(backend_mode_value);
    }

    switch (backend_mode_value) {
        case STATE_VECTOR_BACKEND:
            return STATE_VECTOR_BACKEND;
        case DENSITY_MATRIX_BACKEND:
            return DENSITY_MATRIX_BACKEND;
        default:
            throw std::string(
                "Variational_Quantum_Eigensolver_Base::Variational_Quantum_Eigensolver_Base: unsupported backend_mode value"
            );
    }
}

}

/**
@brief A base class to solve VQE problems
This class can be used to approximate the ground state of the input Hamiltonian (sparse format) via a quantum circuit
*/
Variational_Quantum_Eigensolver_Base::Variational_Quantum_Eigensolver_Base() {

    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;


    // error of the unitarity of the final decomposition
    decomposition_error = DBL_MAX;


    // The current minimum of the optimization problem
    current_minimum = DBL_MAX;
    
    global_target_minimum = -DBL_MAX;

    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;


    // The maximal allowed error of the optimization problem
    optimization_tolerance = -DBL_MAX;

    // The convergence threshold in the optimization process
    convergence_threshold = -DBL_MAX;
    
    alg = AGENTS;
    

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
    cost_fnc = VQE;
    
    ansatz = HEA;
    backend_mode = STATE_VECTOR_BACKEND;
    circuit_source = VQE_CIRCUIT_SOURCE_UNSET;
    density_noise_specs.clear();
}


void Variational_Quantum_Eigensolver_Base::set_density_noise_specs(
    const std::vector<DensityNoiseSpec>& density_noise_specs_in) {

    for (const auto& spec : density_noise_specs_in) {
        if (spec.target_qbit < 0 || spec.target_qbit >= qbit_num) {
            throw std::string(
                "Variational_Quantum_Eigensolver_Base::set_density_noise_specs: target_qbit out of range"
            );
        }

        if (spec.after_gate_index < 0) {
            throw std::string(
                "Variational_Quantum_Eigensolver_Base::set_density_noise_specs: after_gate_index must be non-negative"
            );
        }

        if (spec.value < 0.0 || spec.value > 1.0) {
            throw std::string(
                "Variational_Quantum_Eigensolver_Base::set_density_noise_specs: noise values must be in [0, 1]"
            );
        }
    }

    density_noise_specs = density_noise_specs_in;
}


bool Variational_Quantum_Eigensolver_Base::density_optimizer_supported() const {

    return alg == BAYES_OPT || alg == COSINE;
}


void Variational_Quantum_Eigensolver_Base::validate_density_anchor_support(
    bool require_optimizer_support, bool require_gradient_support) {

    if (backend_mode == STATE_VECTOR_BACKEND) {
        if (!density_noise_specs.empty()) {
            throw std::string(
                "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: state_vector backend does not support density_matrix-only noise configuration"
            );
        }
        return;
    }

    int expected_dimension = Power_of_2(qbit_num);
    if (Hamiltonian.rows != expected_dimension || Hamiltonian.cols != expected_dimension) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: density_matrix backend requires a square Hamiltonian matching the configured qubit count"
        );
    }

    if (require_gradient_support) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: density_matrix backend does not support gradient-based optimization in Story 3"
        );
    }

    if (ansatz != HEA) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: density_matrix backend currently supports only the HEA ansatz"
        );
    }

    if (require_optimizer_support && !density_optimizer_supported()) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: density_matrix backend currently supports only BAYES_OPT or COSINE optimization traces"
        );
    }

    if (circuit_source == VQE_CIRCUIT_SOURCE_UNSET) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: density_matrix backend requires a generated HEA circuit"
        );
    }

    if (circuit_source != VQE_CIRCUIT_SOURCE_GENERATED_HEA) {
        std::string error(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: unsupported circuit source in density backend path: "
        );
        error += vqe_circuit_source_to_label(circuit_source);
        throw error;
    }

    int gate_num = get_gate_num();
    if (gate_num <= 0) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: density_matrix backend requires a generated HEA circuit"
        );
    }

    for (int gate_idx = 0; gate_idx < gate_num; ++gate_idx) {
        Gate* gate = get_gate(gate_idx);
        switch (gate->get_type()) {
            case U3_OPERATION:
            case CNOT_OPERATION:
                break;
            default: {
                std::string error(
                    "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: first unsupported gate in density backend path is "
                );
                error += gate->get_name();
                throw error;
            }
        }
    }

    for (size_t spec_idx = 0; spec_idx < density_noise_specs.size(); ++spec_idx) {
        const auto& spec = density_noise_specs[spec_idx];
        if (spec.after_gate_index >= gate_num) {
            std::string error(
                "Variational_Quantum_Eigensolver_Base::validate_density_anchor_support: unsupported density-noise insertion at index "
            );
            error += std::to_string(spec_idx);
            error += " because after_gate_index exceeds generated gate count";
            throw error;
        }
    }
}


void Variational_Quantum_Eigensolver_Base::append_density_noise_for_gate_index(
    squander::density::NoisyCircuit& circuit, int gate_index) const {

    for (const auto& spec : density_noise_specs) {
        if (spec.after_gate_index != gate_index) {
            continue;
        }

        switch (spec.type) {
            case LOCAL_DEPOLARIZING_NOISE:
                circuit.add_local_depolarizing(spec.target_qbit, spec.value);
                break;
            case AMPLITUDE_DAMPING_NOISE:
                circuit.add_amplitude_damping(spec.target_qbit, spec.value);
                break;
            case PHASE_DAMPING_NOISE:
                circuit.add_phase_damping(spec.target_qbit, spec.value);
                break;
            default: {
                std::string error(
                    "Variational_Quantum_Eigensolver_Base::append_density_noise_for_gate_index: unsupported density-noise type "
                );
                error += density_noise_type_to_label(spec.type);
                throw error;
            }
        }
    }
}


void Variational_Quantum_Eigensolver_Base::lower_anchor_circuit_to_noisy_circuit(
    squander::density::NoisyCircuit& circuit) {

    validate_density_anchor_support();

    int gate_num = get_gate_num();
    for (int gate_idx = 0; gate_idx < gate_num; ++gate_idx) {
        Gate* gate = get_gate(gate_idx);

        switch (gate->get_type()) {
            case U3_OPERATION:
                circuit.add_U3(gate->get_target_qbit());
                break;
            case CNOT_OPERATION:
                circuit.add_CNOT(gate->get_target_qbit(), gate->get_control_qbit());
                break;
            default: {
                std::string error(
                    "Variational_Quantum_Eigensolver_Base::lower_anchor_circuit_to_noisy_circuit: unsupported gate in density backend path: "
                );
                error += gate->get_name();
                throw error;
            }
        }

        append_density_noise_for_gate_index(circuit, gate_idx);
    }
}


double Variational_Quantum_Eigensolver_Base::expectation_value_of_density_energy_real(
    const squander::density::DensityMatrix& rho) const {

    if (Hamiltonian.rows != rho.get_dim() || Hamiltonian.cols != rho.get_dim()) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::expectation_value_of_density_energy_real: dimension mismatch"
        );
    }

    double energy_real = 0.0;
    for (int row = 0; row < Hamiltonian.rows; ++row) {
        for (int idx = Hamiltonian.indptr[row]; idx < Hamiltonian.indptr[row + 1];
             ++idx) {
            int col = Hamiltonian.indices[idx];
            const QGD_Complex16& h_val = Hamiltonian.data[idx];
            const QGD_Complex16& rho_val = rho(col, row);

            energy_real += h_val.real * rho_val.real - h_val.imag * rho_val.imag;
        }
    }

    if (!std::isfinite(energy_real)) {
        throw std::string(
            "Variational_Quantum_Eigensolver_Base::expectation_value_of_density_energy_real: produced a non-finite energy"
        );
    }

    return energy_real;
}


double Variational_Quantum_Eigensolver_Base::evaluate_density_matrix_backend(
    Matrix_real& parameters) {

    squander::density::DensityMatrix rho =
        (initial_state.size() == 0)
            ? squander::density::DensityMatrix(qbit_num)
            : squander::density::DensityMatrix(initial_state);

    squander::density::NoisyCircuit circuit(qbit_num);
    lower_anchor_circuit_to_noisy_circuit(circuit);
    circuit.apply_to(parameters, rho);

    return expectation_value_of_density_energy_real(rho);
}


/**
@brief Constructor of the class.
@param Hamiltonian_in The Hamiltonian describing the physical system
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param config_in A map that can be used to set hyperparameters during the process
@return An instance of the class
*/
Variational_Quantum_Eigensolver_Base::Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in, int accelerator_num) : Optimization_Interface(Matrix(Power_of_2(qbit_num_in),1), qbit_num_in, false, config_in, RANDOM, accelerator_num) {

	Hamiltonian = Hamiltonian_in;
    // config maps
    config   = config_in;
    backend_mode = parse_vqe_backend_mode(config);
    // logical value describing whether the decomposition was finalized or not
    decomposition_finalized = false;


    // error of the unitarity of the final decomposition
    decomposition_error = DBL_MAX;


    // The current minimum of the optimization problem
    current_minimum = DBL_MAX;


    // logical value describing whether the optimization problem was solved or not
    optimization_problem_solved = false;

    // override optimization parameters governing the convergence used in gate decomposition applications
    global_target_minimum  = -DBL_MAX;
    optimization_tolerance = -DBL_MAX;
    convergence_threshold  = -DBL_MAX;
    
   

    random_shift_count_max = 100;
    
    adaptive_eta = false;
    
    qbit_num = qbit_num_in;
	

	
    std::random_device rd;  
    
    gen = std::mt19937(rd());
    
    alg = BAYES_OPT;
    
    cost_fnc = VQE;
    
    ansatz = HEA;
    circuit_source = VQE_CIRCUIT_SOURCE_UNSET;
    density_noise_specs.clear();
    
}




/**
@brief Destructor of the class
*/
Variational_Quantum_Eigensolver_Base::~Variational_Quantum_Eigensolver_Base(){

}




/**
@brief Call to start solving the VQE problem to get the approximation for the ground state  
*/ 
void Variational_Quantum_Eigensolver_Base::start_optimization(){

    // initialize the initial state if it was not given
    if ( initial_state.size() == 0 ) {
        initialize_zero_state();
    }


    if (gates.size() == 0 ) {
        std::string error("Variational_Quantum_Eigensolver_Base::Get_ground_state: for VQE process the circuit needs to be initialized");
        throw error;
    }

    int num_of_parameters =  optimized_parameters_mtx.size();
    if ( num_of_parameters == 0 ) {
        std::string error("Variational_Quantum_Eigensolver_Base::Get_ground_state: No intial parameters were given");
        throw error;
    }


    if ( num_of_parameters != get_parameter_num() ) {
        std::string error("Variational_Quantum_Eigensolver_Base::Get_ground_state: The number of initial parameters does not match with the number of parameters in the circuit");
        throw error;
    }    

    validate_density_anchor_support(true, false);


    // start the VQE process
    Matrix_real solution_guess = optimized_parameters_mtx.copy();
    solve_layer_optimization_problem(num_of_parameters, solution_guess);


    return;
}



/**
@brief Call to evaluate the expectation value of the energy  <State_left| H | State_right>.
@param State_left The state on the let for which the expectation value is evaluated. It is a column vector. In the sandwich product it is transposed and conjugated inside the function.
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated expectation value
*/
QGD_Complex16 Variational_Quantum_Eigensolver_Base::Expectation_value_of_energy( Matrix& State_left, Matrix& State_right ) {


    if ( State_left.rows != State_right.rows || Hamiltonian.rows != State_right.rows) {
        std::string error("Variational_Quantum_Eigensolver_Base::Expectation_value_of_energy: States on the right and left should be of the same dimension as the Hamiltonian");
        throw error;
    }

    Matrix tmp = mult(Hamiltonian, State_right);
    QGD_Complex16 Energy;
    Energy.real = 0.0;
    Energy.imag = 0.0;    

    
    tbb::combinable<double> priv_partial_energy_real{[](){return 0.0;}};
    tbb::combinable<double> priv_partial_energy_imag{[](){return 0.0;}};    

    tbb::parallel_for( tbb::blocked_range<int>(0, State_left.rows, 1024), [&](tbb::blocked_range<int> r) { 

        double& energy_local_real = priv_partial_energy_real.local();
        double& energy_local_imag = priv_partial_energy_imag.local();        

        for (int idx=r.begin(); idx<r.end(); idx++){
	    energy_local_real += State_left[idx].real*tmp[idx].real + State_left[idx].imag*tmp[idx].imag;
	    energy_local_imag += State_left[idx].real*tmp[idx].imag - State_left[idx].imag*tmp[idx].real;	    
        }

    });

    // calculate the final cost function
    priv_partial_energy_real.combine_each([&Energy](double a) {
        Energy.real = Energy.real + a;
    });
    
    priv_partial_energy_imag.combine_each([&Energy](double a) {
        Energy.imag = Energy.imag + a;
    });    
 

    tmp.release_data(); // TODO: this is not necessary, beacause it is called with the destructor of the class Matrix

    {
        tbb::spin_mutex::scoped_lock my_lock{my_mutex};

        number_of_iters++;
        
    }

    return Energy;
}


/**
@brief Call to evaluate the expectation value of the energy  <State_left| H | State_right>. Calculates only the real part of the expectation value.
@param State_left The state on the let for which the expectation value is evaluated. It is a column vector. In the sandwich product it is transposed and conjugated inside the function.
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated expectation value
*/
double Variational_Quantum_Eigensolver_Base::Expectation_value_of_energy_real( Matrix& State_left, Matrix& State_right ) {


    if ( State_left.rows != State_right.rows || Hamiltonian.rows != State_right.rows) {
        std::string error("Variational_Quantum_Eigensolver_Base::Expectation_value_of_energy_real: States on the right and left should be of the same dimension as the Hamiltonian");
        throw error;
    }

    Matrix tmp = mult(Hamiltonian, State_right);
    double Energy = 0.0;

    
    tbb::combinable<double> priv_partial_energy{[](){return 0.0;}};

    tbb::parallel_for( tbb::blocked_range<int>(0, State_left.rows, 1024), [&](tbb::blocked_range<int> r) { 

        double& energy_local = priv_partial_energy.local();

        for (int idx=r.begin(); idx<r.end(); idx++){
	    energy_local += State_left[idx].real*tmp[idx].real + State_left[idx].imag*tmp[idx].imag;
        }

    });

    // calculate the final cost function
    priv_partial_energy.combine_each([&Energy](double a) {
        Energy = Energy + a;
    });
 

    tmp.release_data(); // TODO: this is not necessary, beacause it is called with the destructor of the class Matrix

    {
        tbb::spin_mutex::scoped_lock my_lock{my_mutex};

        number_of_iters++;
        
    }

    return Energy;
}


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
/**
 * Shot-noise estimator for the expectation value of the Hamiltonian
 * <H> = ⟨ψ|H|ψ⟩ including readout errors and finite-shot sampling.
 *
 * Description:
 *  - Given the input state vector `State` (size N = 2^n_qubits), this
 *    routine performs Monte–Carlo sampling to estimate the mean and
 *    variance of the Hamiltonian measurement under a projective measurement
 *    model with independent per-qubit readout flips.
 *
 * Measurement model / supported terms:
 *  - `zz_terms`: list of (i, j, coeff) representing coeff * Z_i Z_j
 *  - `z_terms` : list of pairs (i, coeff) representing coeff * Z_i
 *  - `xx_terms`: list of (i, j, coeff) representing coeff * X_i X_j
 *               (implemented by rotating the required qubits to the X basis
 *                with Hadamard gates and measuring in Z)
 *  - `yy_terms`: list of (i, j, coeff) representing coeff * Y_i Y_j
 *               (implemented by applying S† then H on required qubits)
 *
 * Algorithm summary:
 *  1. Compute Z-basis probabilities from the supplied state |ψ⟩.
 *  2. If XX terms are present, build a single global X-basis distribution
 *     by applying H to every qubit that appears in any XX term and
 *     computing the rotated probabilities.
 *  3. If YY terms are present, build a single global Y-basis distribution
 *     by applying S† then H to every qubit that appears in any YY term.
 *  4. For each Monte–Carlo shot:
 *       - sample one Z-outcome (for ZZ and Z contributions), optionally
 *         flip bits according to `p_readout` (independent per-qubit flips),
 *         and accumulate the corresponding term contributions.
 *       - sample one global X-outcome (for all XX terms) and accumulate.
 *       - sample one global Y-outcome (for all YY terms) and accumulate.
 *  5. Compute sample mean, variance and standard error (std_error = sqrt(var/shots)).
 *
 * Parameters
 * ----------
 *  State : Matrix&
 *    State vector |ψ⟩ as the project's Matrix/QGD_Complex16 representation.
 *  shots : int
 *    Number of Monte–Carlo measurement shots to draw.
 *  seed : uint64_t
 *    RNG seed; if zero a random device is used to seed the engine.
 *  p_readout : double
 *    Probability [0,1] that a given qubit measurement is flipped (bit-flip
 *    readout error) and applied independently per measured qubit.
 *
 * Returns
 * -------
 *  SimulationResult
 *    Struct with fields `{ mean, variance, std_error }` containing the
 *    estimated expectation value, its sample variance, and the estimated
 *    standard error of the mean.
 *
 * Notes
 * -----
 *  - The routine throws a std::string on dimension or index errors.
 *  - A single global X (and Y) distribution is built and reused for all
 *    XX (and YY) terms; this is correct because all XX (YY) terms are
 *    measured in the same rotated basis.
 *  - The implementation deliberately copies the state for basis rotations
 *    so the input `State` is not modified.
 *  - Thread-safety: the function uses a local RNG instance; only the final
 *    increment of `number_of_iters` is protected by the `my_mutex` spinlock.
 */
SimulationResult Variational_Quantum_Eigensolver_Base::Expectation_value_with_shot_noise_real(
    Matrix &State, int shots, uint64_t seed, double p_readout
) {

if (State.rows != Hamiltonian.rows) {
    throw std::string("Expectation_value_with_shot_noise_real: dimension mismatch");
}

const int N = State.rows;
const int n_qubits = static_cast<int>(std::round(std::log2(N)));
if ((1 << n_qubits) != N) {
    throw std::string("Expectation_value_with_shot_noise_real: State.rows not power of two");
}

// -------------------------------------------------------
// 1) Z-basis probabilities from |ψ⟩  (for Z and ZZ terms)
// -------------------------------------------------------
std::vector<double> probsZ(N);
double sum_pZ = 0.0;
for (int idx = 0; idx < N; ++idx) {
    double re = State[idx].real;
    double im = State[idx].imag;
    double p  = re * re + im * im;
    probsZ[idx] = p;
    sum_pZ += p;
}
if (std::abs(sum_pZ - 1.0) > 1e-6) {
    // Slight numerical mismatch is OK; normalize anyway
    for (auto &v : probsZ) v /= sum_pZ;
} else {
    for (auto &v : probsZ) v /= sum_pZ;
}

std::mt19937_64 rng(seed ? seed : std::random_device{}());
std::discrete_distribution<int> distZ(probsZ.begin(), probsZ.end());
std::uniform_real_distribution<double> unif01(0.0, 1.0);

// -------------------------------------------------------
// 2) Global X-basis distribution for ALL XX terms
//    We apply H on every qubit that appears in any XX term
// -------------------------------------------------------
bool has_xx = !xx_terms.empty();
std::discrete_distribution<int> distX;  // will be initialized if has_xx == true

if (has_xx) {

    // Which qubits need H for XX measurement?
    std::vector<bool> need_H(n_qubits, false);
    for (const auto &t : xx_terms) {
        int i, j;
        double coeff;
        std::tie(i, j, coeff) = t;
        if (i < 0 || i >= n_qubits || j < 0 || j >= n_qubits) {
            throw std::string("XX term index out of range in Expectation_value_with_shot_noise_real");
        }
        need_H[i] = true;
        need_H[j] = true;
    }

    // Rotated state |ψ_X⟩ = (⊗_q∈S H_q) |ψ⟩,
    // where S is the set of qubits that appear in any XX term.
    Matrix psi_rot = State;  // copy original state |ψ⟩
    for (int q = 0; q < n_qubits; ++q) {
        if (need_H[q]) {
            H h_gate(n_qubits, q);
            h_gate.apply_to(psi_rot, 0);
        }
    }

    // Build X-basis probabilities
    std::vector<double> probsX(N);
    double sum_pX = 0.0;
    for (int idx = 0; idx < N; ++idx) {
        double re = psi_rot[idx].real;
        double im = psi_rot[idx].imag;
        double p  = re * re + im * im;
        probsX[idx] = p;
        sum_pX += p;
    }
    if (sum_pX <= 0.0) {
        throw std::string("X-rotated state has zero norm in Expectation_value_with_shot_noise_real");
    }
    for (auto &v : probsX) v /= sum_pX;

    // One global X-basis distribution for all XX terms
    distX = std::discrete_distribution<int>(probsX.begin(), probsX.end());
}

// -------------------------------------------------------
// 2b) Global Y-basis distribution for ALL YY terms
//    We apply S† (SDG) then H on every qubit that appears in any YY term
//    This rotates Y eigenstates to Z eigenstates for measurement
// -------------------------------------------------------
bool has_yy = !yy_terms.empty();
std::discrete_distribution<int> distY;  // will be initialized if has_yy == true

if (has_yy) {

    // Which qubits need S†H for YY measurement?
    std::vector<bool> need_SDG_H(n_qubits, false);
    for (const auto &t : yy_terms) {
        int i, j;
        double coeff;
        std::tie(i, j, coeff) = t;
        if (i < 0 || i >= n_qubits || j < 0 || j >= n_qubits) {
            throw std::string("YY term index out of range in Expectation_value_with_shot_noise_real");
        }
        need_SDG_H[i] = true;
        need_SDG_H[j] = true;
    }

    // Rotated state |ψ_Y⟩ = (⊗_q∈S H_q S†_q) |ψ⟩,
    // where S is the set of qubits that appear in any YY term.
    // S† rotates Y to X, then H rotates X to Z
    Matrix psi_rot_y = State;  // copy original state |ψ⟩
    for (int q = 0; q < n_qubits; ++q) {
        if (need_SDG_H[q]) {
            // Apply S† (SDG) first to rotate Y to X
            SDG sdg_gate(n_qubits, q);
            sdg_gate.apply_to(psi_rot_y, 0);
            // Then apply H to rotate X to Z
            H h_gate(n_qubits, q);
            h_gate.apply_to(psi_rot_y, 0);
        }
    }

    // Build Y-basis probabilities (after S†H rotation, Z-eigenvalues encode Y-eigenvalues)
    std::vector<double> probsY(N);
    double sum_pY = 0.0;
    for (int idx = 0; idx < N; ++idx) {
        double re = psi_rot_y[idx].real;
        double im = psi_rot_y[idx].imag;
        double p  = re * re + im * im;
        probsY[idx] = p;
        sum_pY += p;
    }
    if (sum_pY <= 0.0) {
        throw std::string("Y-rotated state has zero norm in Expectation_value_with_shot_noise_real");
    }
    for (auto &v : probsY) v /= sum_pY;

    // One global Y-basis distribution for all YY terms
    distY = std::discrete_distribution<int>(probsY.begin(), probsY.end());
}

// -------------------------------------------------------
// 3) Monte–Carlo sampling
//    For each "shot":
//      - sample one Z-basis outcome for ZZ and Z terms
//      - sample one X-basis outcome for ALL XX terms
//      - sample one Y-basis outcome for ALL YY terms
// -------------------------------------------------------
double sum     = 0.0;
double sum_sq  = 0.0;

for (int s = 0; s < shots; ++s) {

    double E = 0.0;

    // --------- Z-basis measurement: ZZ + Z ----------
    int idx = distZ(rng);

    if (p_readout > 0.0) {
        int observed = idx;
        for (int q = 0; q < n_qubits; ++q) {
            if (unif01(rng) < p_readout) {
                observed ^= (1 << q);
            }
        }
        idx = observed;
    }

    // ZZ terms: coeff * Z_i Z_j
    for (const auto &t : zz_terms) {
        int i, j;
        double coeff;
        std::tie(i, j, coeff) = t;

        const int zi = ((idx >> i) & 1) ? -1 : +1;
        const int zj = ((idx >> j) & 1) ? -1 : +1;
        E += coeff * (zi * zj);
    }

    // Z terms: h_i * Z_i
    for (const auto &z : z_terms) {
        const int i = z.first;
        const double h = z.second;
        const int zi = ((idx >> i) & 1) ? -1 : +1;
        E += h * zi;
    }

    // --------- X-basis measurement: ALL XX terms -----
    if (has_xx) {
        int idx_x = distX(rng);

        if (p_readout > 0.0) {
            int observed = idx_x;
            for (int q = 0; q < n_qubits; ++q) {
                if (unif01(rng) < p_readout) {
                    observed ^= (1 << q);
                }
            }
            idx_x = observed;
        }

        // After H-rotation, Z-eigenvalues encode X-eigenvalues
        for (const auto &t : xx_terms) {
            int i, j;
            double coeff;
            std::tie(i, j, coeff) = t;

            const int xi = ((idx_x >> i) & 1) ? -1 : +1;
            const int xj = ((idx_x >> j) & 1) ? -1 : +1;
            E += coeff * (xi * xj);
        }
    }

    // --------- Y-basis measurement: ALL YY terms -----
    if (has_yy) {
        int idx_y = distY(rng);

        if (p_readout > 0.0) {
            int observed = idx_y;
            for (int q = 0; q < n_qubits; ++q) {
                if (unif01(rng) < p_readout) {
                    observed ^= (1 << q);
                }
            }
            idx_y = observed;
        }

        // After S†H-rotation, Z-eigenvalues encode Y-eigenvalues
        for (const auto &t : yy_terms) {
            int i, j;
            double coeff;
            std::tie(i, j, coeff) = t;

            const int yi = ((idx_y >> i) & 1) ? -1 : +1;
            const int yj = ((idx_y >> j) & 1) ? -1 : +1;
            E += coeff * (yi * yj);
        }
    }

    sum    += E;
    sum_sq += E * E;
}

const double mean     = sum / static_cast<double>(shots);
const double mean_sq  = sum_sq / static_cast<double>(shots);
double variance       = mean_sq - mean * mean;
if (variance < 0.0 && variance > -1e-12) {
    variance = 0.0;
}
const double std_error = (variance > 0.0)
                         ? std::sqrt(variance / static_cast<double>(shots))
                         : 0.0;

{
    tbb::spin_mutex::scoped_lock my_lock{my_mutex};
    number_of_iters++;
}

return {mean, variance, std_error};
}


//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------



/**
@brief The optimization problem of the final optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Variational_Quantum_Eigensolver_Base::optimization_problem_non_static(Matrix_real parameters, void* void_instance){

    double Energy=0.0;

    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    instance->validate_density_anchor_support(false, false);

    if (instance->backend_mode == DENSITY_MATRIX_BACKEND) {
        return instance->evaluate_density_matrix_backend(parameters);
    }

    Matrix State = instance->initial_state.copy();

    instance->apply_to(parameters, State);
    Energy = instance->Expectation_value_of_energy_real(State, State);

    return Energy;
}


#ifdef __GROQ__
/**
@brief The optimization problem of the final optimization implemented to be run on Groq hardware
@param parameters An array of the free parameters to be optimized.
@param chosen_device Indicate the device on which the state vector emulation is performed
@return Returns with the cost function.
*/
double Variational_Quantum_Eigensolver_Base::optimization_problem_Groq(Matrix_real& parameters, int chosen_device)  {



    Matrix State;


    //tbb::tick_count t0_DFE = tbb::tick_count::now();
    std::vector<int> target_qbits;
    std::vector<int> control_qbits;
    std::vector<Matrix> u3_qbit;
    extract_gate_kernels_target_and_control_qubits(u3_qbit, target_qbits, control_qbits, parameters);
        

    // initialize the initial state on the chip if it was not given
    if ( initial_state.size() == 0 ) {
        
        Matrix State_zero(0,0);  
        apply_to_groq_sv(accelerator_num, chosen_device, qbit_num, u3_qbit, target_qbits, control_qbits, State_zero, id); 
            
        State = State_zero;
            
    }
    else {
	
        State = initial_state.copy();
        // apply state transformation via the Groq chip
        apply_to_groq_sv(accelerator_num, chosen_device, qbit_num, u3_qbit, target_qbits, control_qbits, State, id);

    }
        
        
/*        
    //////////////////////////////
    Matrix State_copy = State.copy();


    Decomposition_Base::apply_to(parameters, State_copy );


    double diff = 0.0;
    for( int64_t idx=0; idx<State_copy.size(); idx++ ) {

        QGD_Complex16 element = State[idx];
        QGD_Complex16 element_copy = State_copy[idx];
        QGD_Complex16 element_diff;
        element_diff.real = element.real - element_copy.real;
        element_diff.imag = element.imag - element_copy.imag;
 
        double diff_increment = element_diff.real*element_diff.real + element_diff.imag*element_diff.imag;
        diff = diff + diff_increment;
    }
       
    std::cout << "Variational_Quantum_Eigensolver_Base::apply_to checking diff: " << diff << std::endl;


    if ( diff > 1e-4 ) {
        std::string error("Groq and CPU results do not match");
        throw(error);
    }

    //////////////////////////////
*/


	
    double Energy = Expectation_value_of_energy_real(State, State);
	
    return Energy;

}

#endif


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters associated with the circuit. 
@return Returns with the cost function.
*/
double Variational_Quantum_Eigensolver_Base::optimization_problem(Matrix_real& parameters)  {


    validate_density_anchor_support(false, false);

    if (backend_mode == DENSITY_MATRIX_BACKEND) {
        return evaluate_density_matrix_backend(parameters);
    }

    Matrix State;

#ifdef __GROQ__
    if ( accelerator_num > 0 ) {
    
        
        return optimization_problem_Groq( parameters, 0 );
            
    }
    else {
#endif
        // initialize the initial state if it was not given
        if ( initial_state.size() == 0 ) {
            initialize_zero_state();
        }
	
        State = initial_state.copy();
        apply_to(parameters, State);

#ifdef __GROQ__
    }
#endif



	
    double Energy = Expectation_value_of_energy_real(State, State);
	
    return Energy;
}



/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Variational_Quantum_Eigensolver_Base::optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) {

    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    instance->validate_density_anchor_support(false, true);

    // initialize the initial state if it was not given
    if ( instance->initial_state.size() == 0 ) {
        instance->initialize_zero_state();
    }

    // the number of free parameters
    int parameter_num_loc = parameters.size();

    Matrix_real cost_function_terms;

    // vector containing gradients of the transformed matrix
    std::vector<Matrix> State_deriv;
    Matrix State;

    int parallel = get_parallel_configuration();

    tbb::parallel_invoke(
        [&]{
            State = instance->initial_state.copy();
            instance->apply_to(parameters, State);
            *f0 = instance->Expectation_value_of_energy_real(State, State);
            
        },
        [&]{
            Matrix State_loc = instance->initial_state.copy();

            State_deriv = instance->apply_derivate_to( parameters, State_loc, parallel );
            State_loc.release_data();
    });

    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,2), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            grad[idx] = 2*instance->Expectation_value_of_energy_real(State_deriv[idx], State);
        }
    });
    
    /*
    double delta = 0.0000001;
    
    tbb::parallel_for( tbb::blocked_range<int>(0,parameter_num_loc,1), [&](tbb::blocked_range<int> r) {
        for (int idx=r.begin(); idx<r.end(); ++idx) { 
            Matrix_real param_tmp = parameters.copy();
            param_tmp[idx] += delta;
            
            Matrix State = instance->initial_state.copy();
            instance->apply_to(param_tmp, State);
            double f_loc = instance->Expectation_value_of_energy_real(State, State);
            
            
            grad[idx] = (f_loc-*f0)/delta;
        }
    });    

*/
    return;
}






/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
void Variational_Quantum_Eigensolver_Base::optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad )  {

    optimization_problem_combined_non_static( parameters, this, f0, grad );
    return;
}




/**
@brief Calculate the derivative of the cost function with respect to the free parameters.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad Array containing the calculated gradient components.
*/
void Variational_Quantum_Eigensolver_Base::optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad ) {

    double f0;
    Variational_Quantum_Eigensolver_Base* instance = reinterpret_cast<Variational_Quantum_Eigensolver_Base*>(void_instance);
    instance->optimization_problem_combined_non_static(parameters, void_instance, &f0, grad);
    return;

}




/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double Variational_Quantum_Eigensolver_Base::optimization_problem( double* parameters ) {

    // get the transformed matrix with the gates in the list
    Matrix_real parameters_mtx(parameters, 1, parameter_num );
    
    return optimization_problem( parameters_mtx );


}




/**
@brief Initialize the state used in the quantun circuit. All qubits are initialized to state 0
*/
void Variational_Quantum_Eigensolver_Base::initialize_zero_state( ) {

    int initialize_state;
    if ( config.count("initialize_state") > 0 ) { 
         long long value;                   
         config["initialize_state"].get_property( value );  
         initialize_state = (int) value;
    }
    else {
        initialize_state = 1;
         
    }
    
    if( initialize_state == 0 ) {
        initial_state = Matrix(0, 0);
        return;
    }

    initial_state = Matrix( 1 << qbit_num , 1);


    initial_state[0].real = 1.0;
    initial_state[0].imag = 0.0;
    memset(initial_state.get_data()+2, 0, (initial_state.size()*2-2)*sizeof(double) );    

    initial_state[1].real = 0.0;
    initial_state[1].imag = 0.0;  


    return;
}





/**
@brief Call to set the ansatz type. Currently imp
@param ansatz_in The ansatz type . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
*/ 
void Variational_Quantum_Eigensolver_Base::set_ansatz(ansatz_type ansatz_in){

    ansatz = ansatz_in;
    
    return;
}





/**
@brief Call to generate the circuit ansatz
@param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT)
@param inner_blocks The number of U3-CNOT repetition within a single layer
*/
void Variational_Quantum_Eigensolver_Base::generate_circuit( int layers, int inner_blocks ) {


    switch (ansatz){
    
        case HEA:
        {

            release_gates();

            if ( qbit_num < 2 ) {
                std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: number of qubits should be at least 2");
                throw error;
            }

            for (int layer_idx=0; layer_idx<layers ;layer_idx++){

                for( int idx=0; idx<inner_blocks; idx++) {
                    add_u3(1);                          
                    add_u3(0);
                    add_cnot(1,0);
                }


                for (int control_qbit=1; control_qbit<qbit_num-1; control_qbit=control_qbit+2){
                    if (control_qbit+2<qbit_num){

                        for( int idx=0; idx<inner_blocks; idx++) {
                            add_u3(control_qbit+1);
                            add_u3(control_qbit+2); 
                        
                            add_cnot(control_qbit+2,control_qbit+1);
                        }

                    }

                    for( int idx=0; idx<inner_blocks; idx++) {
                        add_u3(control_qbit+1);  
                        add_u3(control_qbit);  

                        add_cnot(control_qbit+1,control_qbit);

                    }

                }
            }


            circuit_source = VQE_CIRCUIT_SOURCE_GENERATED_HEA;
            return;
        }
        case HEA_ZYZ:
        {

            release_gates();

            if ( qbit_num < 2 ) {
                std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: number of qubits should be at least 2");
                throw error;
            }

            for (int layer_idx=0; layer_idx<layers ;layer_idx++){

                for( int idx=0; idx<inner_blocks; idx++) {
                    Gates_block* block_1 = new Gates_block( qbit_num );
                    Gates_block* block_2 = new Gates_block( qbit_num );
 
                    // single qubit gates in a ablock are then merged into a single gate kernel.
                    block_1->add_rz(1);
                    block_1->add_ry(1);
                    block_1->add_rz(1);     

                    add_gate( block_1 );                           

                    block_2->add_rz(0);
                    block_2->add_ry(0);
                    block_2->add_rz(0);         

                    add_gate( block_2 );           
                
                    add_cnot(1,0);
                }

                for (int control_qbit=1; control_qbit<qbit_num-1; control_qbit=control_qbit+2){
                    if (control_qbit+2<qbit_num){

                        for( int idx=0; idx<inner_blocks; idx++) {

                            Gates_block* block_1 = new Gates_block( qbit_num );
                            Gates_block* block_2 = new Gates_block( qbit_num );

                            block_1->add_rz(control_qbit+1);
                            block_1->add_ry(control_qbit+1);
                            block_1->add_rz(control_qbit+1); 
                            add_gate( block_1 );                                

                            block_2->add_rz(control_qbit+2);
                            block_2->add_ry(control_qbit+2);
                            block_2->add_rz(control_qbit+2);  
                            add_gate( block_2 );                              

                            add_cnot(control_qbit+2,control_qbit+1);
                            
                        }

                    }

                    for( int idx=0; idx<inner_blocks; idx++) {

                        Gates_block* block_1 = new Gates_block( qbit_num );
                        Gates_block* block_2 = new Gates_block( qbit_num );

                        block_1->add_rz(control_qbit+1);
                        block_1->add_ry(control_qbit+1);
                        block_1->add_rz(control_qbit+1); 
                        add_gate( block_1 );                      

                        block_2->add_rz(control_qbit);
                        block_2->add_ry(control_qbit);
                        block_2->add_rz(control_qbit);     
                        add_gate( block_2 );                  

                        add_cnot(control_qbit+1,control_qbit);
                    }

                }
            }


            circuit_source = VQE_CIRCUIT_SOURCE_GENERATED_HEA_ZYZ;
            return;
        }        
        default:
            std::string error("Variational_Quantum_Eigensolver_Base::generate_initial_circuit: ansatz not implemented");
            throw error;
    }

}

void 
Variational_Quantum_Eigensolver_Base::set_gate_structure( std::string filename ) {

    release_gates();
    Matrix_real optimized_parameters_mtx_tmp;
    Gates_block* gate_structure_tmp = import_gate_list_from_binary(optimized_parameters_mtx_tmp, filename);

    if ( gates.size() > 0 ) {
        gate_structure_tmp->combine( static_cast<Gates_block*>(this) );

        release_gates();
        combine( gate_structure_tmp );


        Matrix_real optimized_parameters_mtx_tmp2( 1, optimized_parameters_mtx_tmp.size() + optimized_parameters_mtx.size() );
        memcpy( optimized_parameters_mtx_tmp2.get_data(), optimized_parameters_mtx_tmp.get_data(), optimized_parameters_mtx_tmp.size()*sizeof(double) );
        memcpy( optimized_parameters_mtx_tmp2.get_data()+optimized_parameters_mtx_tmp.size(), optimized_parameters_mtx.get_data(), optimized_parameters_mtx.size()*sizeof(double) );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp2;
    }
    else {
        combine( gate_structure_tmp );
        optimized_parameters_mtx = optimized_parameters_mtx_tmp;
            
        std::stringstream sstream;
        // get the number of gates used in the decomposition
        std::map<std::string, int>&& gate_nums = get_gate_nums();
    	
        for( auto it=gate_nums.begin(); it != gate_nums.end(); it++ ) {
            sstream << it->second << " " << it->first << " gates" << std::endl;
        } 
        print(sstream, 1);	
    }

    circuit_source = VQE_CIRCUIT_SOURCE_BINARY_IMPORT;
}

void
Variational_Quantum_Eigensolver_Base::set_custom_gate_structure(
    Gates_block* gate_structure_in ) {

    Optimization_Interface::set_custom_gate_structure( gate_structure_in );
    circuit_source = VQE_CIRCUIT_SOURCE_CUSTOM_GATE_STRUCTURE;
}




/**
@brief Call to set the initial quantum state in the VQE iterations
@param initial_state_in A vector containing the amplitudes of the initial state.
*/
void Variational_Quantum_Eigensolver_Base::set_initial_state( Matrix initial_state_in ) {



    // check the size of the input state
    if ( initial_state_in.size() != 1 << qbit_num ) {
        std::string error("Variational_Quantum_Eigensolver_Base::set_initial_state: teh number of elements in the input state does not match with the number of qubits.");
        throw error;   
    }

    initial_state = Matrix( 1 << qbit_num, 1 );
    memcpy( initial_state.get_data(), initial_state_in.get_data(), initial_state_in.size()*sizeof( QGD_Complex16 ) );

}




