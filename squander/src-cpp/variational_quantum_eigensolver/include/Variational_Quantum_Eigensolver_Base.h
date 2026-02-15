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
/*! \file Variational_Quantum_Eigensolver_Base.h
    \brief Class to solve VQE problems
*/


#ifndef VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H
#define  VARIATIONAL_QUANTUM_EIGENSOLVER_BASE_H

#include "Optimization_Interface.h"

//=====================================================================================================================================================================================
//=====================================================================================================================================================================================
//=====================================================================================================================================================================================

// --- SimulationResult struct ---
/**
 * @brief Container for statistics returned by shot-noise simulations.
 *
 * Fields represent Monte–Carlo estimates produced by
 * `Expectation_value_with_shot_noise_real`:
 *  - `mean` : sample mean of the estimated energy ⟨H⟩ over the performed
 *             measurement shots.
 *  - `variance` : sample variance of the per-shot energy estimator.
 *  - `std_error` : estimated standard error of the mean (typically
 *                  sqrt(variance / shots)).
 *
 * This plain-aggregate struct is returned by the wrapper into Python as a
 * dictionary with the same keys.
 */
struct SimulationResult {
    double mean;
    double variance;
    double std_error;
};
//=====================================================================================================================================================================================
//=====================================================================================================================================================================================
//=====================================================================================================================================================================================


/// @brief Type definition of the fifferent types of ansatz
typedef enum ansatz_type {HEA, HEA_ZYZ} ansatz_type;
    



/**
@brief A base class to solve VQE problems
This class can be used to approximate the ground state of the input Hamiltonian (sparse format) via a quantum circuit
*/
class Variational_Quantum_Eigensolver_Base : public Optimization_Interface {
public:
//================================================================================================================================================================================
//================================================================================================================================================================================
//================================================================================================================================================================================

    /**
     * @brief Return a copy of the stored initial state used for VQE runs.
     *
     * Returns a `Matrix` copy to avoid exposing internal storage; callers
     * may mutate the returned object without affecting the class internals.
     */
    Matrix get_initial_state() const { 
            return initial_state.copy(); 
        }
//================================================================================================================================================================================
//================================================================================================================================================================================
//================================================================================================================================================================================


private:

    /// The Hamiltonian of the system
    Matrix_sparse Hamiltonian;
    
    /// Quantum state used as an initial state in the VQE iterations
    Matrix initial_state;
    
    /// Ansatz type (HEA stands for hardware efficient ansatz)
    ansatz_type ansatz;

public:

//================================================================================================================================================================================
//================================================================================================================================================================================
//================================================================================================================================================================================

    /**
     * @name Hamiltonian term containers
     *
     * Public members used by the Python wrapper to pass measured-term
     * lists into the C++ implementation. Each element is stored in a
     * compact form and interpreted as follows:
     *  - `zz_terms`: tuples `(i, j, coeff)` representing `coeff * Z_i Z_j`.
     *  - `xx_terms`: tuples `(i, j, coeff)` representing `coeff * X_i X_j`.
     *  - `yy_terms`: tuples `(i, j, coeff)` representing `coeff * Y_i Y_j`.
     *  - `z_terms` : pairs `(i, coeff)` representing local field `coeff * Z_i`.
     *
     * In all tuples, `i` and `j` are qubit indices (0-based) and `coeff` is a
     * double precision coefficient. The wrapper populates these vectors from
     * Python dictionaries of the form `{ "i": int, "j": int, "coeff": float }`.
     */
    /// Coupling terms of the form coeff * Z_i * Z_j
    std::vector<std::tuple<int, int, double>> zz_terms;

    /// Local magnetic field terms of the form coeff * Z_i
    std::vector<std::pair<int, double>> z_terms;

    /// Coupling terms of the form coeff * X_i * X_j
    std::vector<std::tuple<int, int, double>> xx_terms;

    /// Coupling terms of the form coeff * Y_i * Y_j
    std::vector<std::tuple<int, int, double>> yy_terms;


//================================================================================================================================================================================
//================================================================================================================================================================================
//================================================================================================================================================================================

/**
@brief Nullary constructor of the class.
@return An instance of the class
*/
Variational_Quantum_Eigensolver_Base();


/**
@brief Constructor of the class.
@param Hamiltonian_in The Hamiltonian describing the physical system
@param qbit_num_in The number of qubits spanning the unitary Umtx
@param config_in A map that can be used to set hyperparameters during the process
@return An instance of the class
*/
Variational_Quantum_Eigensolver_Base( Matrix_sparse Hamiltonian_in, int qbit_num_in, std::map<std::string, Config_Element>& config_in, int accelerator_num=0);

/**
@brief Destructor of the class
*/
virtual ~Variational_Quantum_Eigensolver_Base();

/**
@brief Call to evaluate the expectation value of the energy  <State_left| H | State_right>.
@param State_left The state on the let for which the expectation value is evaluated. It is a column vector. In the sandwich product it is transposed and conjugated inside the function.
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated expectation value
*/
QGD_Complex16 Expectation_value_of_energy(Matrix& State_left, Matrix& State_right);


/**
@brief Call to evaluate the expectation value of the energy  <State_left| H | State_right>. Calculates only the real part of the expectation value.
@param State_left The state on the let for which the expectation value is evaluated. It is a column vector. In the sandwich product it is transposed and conjugated inside the function.
@param State_right The state on the right for which the expectation value is evaluated. It is a column vector.
@return The calculated expectation value
*/
double Expectation_value_of_energy_real(Matrix& State_left, Matrix& State_right);





//=====================================================================================================================================================================================
//=====================================================================================================================================================================================
//=====================================================================================================================================================================================

/**
 * @brief Evaluate the expectation value of the energy with simulated shot noise and readout error.
 * @param State The quantum state (column vector).
 * @param shots Number of measurement shots to simulate.
 * @param seed Random seed for reproducibility.
 * @param p_readout Probability of bit-flip readout error per qubit.
 * @return A SimulationResult struct containing mean, variance, and standard error.
 */
SimulationResult Expectation_value_with_shot_noise_real(Matrix &State, int shots, uint64_t seed, double p_readout);


//=====================================================================================================================================================================================
//=====================================================================================================================================================================================
//=====================================================================================================================================================================================


/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
virtual double optimization_problem(Matrix_real& parameters) override;


/**
@brief The optimization problem of the final optimization
@param parameters Array containing the parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
virtual double optimization_problem_non_static( Matrix_real parameters, void* void_instance) override;


#ifdef __GROQ__
/**
@brief The optimization problem of the final optimization implemented to be run on Groq hardware
@param parameters An array of the free parameters to be optimized.
@param chosen_device Indicate the device on which the state vector emulation is performed
@return Returns with the cost function.
*/
double optimization_problem_Groq(Matrix_real& parameters, int chosen_device) ;
#endif



/**
@brief The optimization problem of the final optimization
@param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
@return Returns with the cost function. (zero if the qubits are desintangled.)
*/
double optimization_problem( double* parameters);


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
virtual void optimization_problem_combined_non_static( Matrix_real parameters, void* void_instance, double* f0, Matrix_real& grad ) override;


/**
@brief Call to calculate both the cost function and the its gradient components.
@param parameters Array containing the free parameters to be optimized.
@param f0 The value of the cost function at x0.
@param grad Array containing the calculated gradient components.
*/
virtual void optimization_problem_combined( Matrix_real parameters, double* f0, Matrix_real grad ) override;


/**
@brief Calculate the derivative of the cost function with respect to the free parameters.
@param parameters Array containing the free parameters to be optimized.
@param void_instance A void pointer pointing to the instance of the current class.
@param grad Array containing the calculated gradient components.
*/
static void optimization_problem_grad_vqe( Matrix_real parameters, void* void_instance, Matrix_real& grad );


/**
@brief Initialize the state used in the quantun circuit. All qubits are initialized to state 0
*/
void initialize_zero_state();


/**
@brief Call to start solving the VQE problem to get the approximation for the ground state  
*/ 
void start_optimization();


/**
@brief Call to set the ansatz type. Currently imp
@param ansatz_in The ansatz type . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
*/ 
void set_ansatz(ansatz_type ansatz_in);


/**
@brief Call to generate the circuit ansatz
@param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT)
@param inner_blocks The number of U3-CNOT repetition within a single layer
*/
void generate_circuit( int layers, int inner_blocks );


/**
@brief Call to set custom layers to the gate structure that are intended to be used in the VQE process.
@param filename The path to the binary file
*/
void set_gate_structure( std::string filename );


/**
@brief Call to set the initial quantum state in the VQE iterations
@param initial_state_in A vector containing the amplitudes of the initial state.
*/
void set_initial_state( Matrix initial_state_in );



};

#endif
