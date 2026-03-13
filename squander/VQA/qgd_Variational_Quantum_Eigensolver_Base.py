## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

"""

## \file qgd_N_Qubit_Decomposition.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.


import numpy as np
from os import path
from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base_Wrapper import qgd_Variational_Quantum_Eigensolver_Base_Wrapper
from squander.gates.qgd_Circuit import qgd_Circuit

_VQE_BACKEND_NAME_TO_CODE = {
    "state_vector": 0,
    "density_matrix": 1,
}
_VQE_DEFAULT_BACKEND = "state_vector"
_VQE_BACKEND_CONFIG_KEY = "backend_mode"
_DENSITY_NOISE_CHANNEL_SPECS = {
    "depolarizing": ("local_depolarizing", "error_rate"),
    "local_depolarizing": ("local_depolarizing", "error_rate"),
    "amplitude_damping": ("amplitude_damping", "gamma"),
    "phase_damping": ("phase_damping", "lambda"),
    "dephasing": ("phase_damping", "lambda"),
}


def _normalize_vqe_backend_name(backend):

    if backend is None:
        return _VQE_DEFAULT_BACKEND

    if not isinstance(backend, str):
        raise TypeError(
            "backend should be one of 'state_vector', 'density_matrix', or None"
        )

    normalized_backend = backend.strip()
    if normalized_backend not in _VQE_BACKEND_NAME_TO_CODE:
        raise ValueError(
            "Unsupported backend '{}'. Supported backends are 'state_vector' and "
            "'density_matrix'.".format(backend)
        )

    return normalized_backend


def _normalize_density_noise_spec(density_noise):

    if density_noise is None:
        return []

    if not isinstance(density_noise, (list, tuple)):
        raise TypeError("density_noise should be a list or tuple of dictionaries")

    normalized_density_noise = []
    for item_idx, noise_item in enumerate(density_noise):
        if not isinstance(noise_item, dict):
            raise TypeError(
                "density_noise[{}] should be a dictionary".format(item_idx)
            )

        channel = noise_item.get("channel", noise_item.get("type"))
        if not isinstance(channel, str):
            raise TypeError(
                "density_noise[{}] should define a string 'channel'".format(
                    item_idx
                )
            )

        channel_name = channel.strip()
        if channel_name not in _DENSITY_NOISE_CHANNEL_SPECS:
            raise ValueError(
                "Unsupported density-noise channel '{}'. Supported channels are "
                "'local_depolarizing', 'amplitude_damping', and "
                "'phase_damping'.".format(channel)
            )

        canonical_channel, value_key = _DENSITY_NOISE_CHANNEL_SPECS[channel_name]
        raw_value = noise_item.get("value", noise_item.get(value_key))
        if canonical_channel == "phase_damping" and raw_value is None:
            raw_value = noise_item.get("lambda_param")

        if raw_value is None:
            raise ValueError(
                "density_noise[{}] is missing the '{}' value".format(
                    item_idx, value_key
                )
            )

        target = noise_item.get("target")
        after_gate_index = noise_item.get("after_gate_index")
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError(
                "density_noise[{}].target should be an integer".format(item_idx)
            )
        if isinstance(after_gate_index, bool) or not isinstance(after_gate_index, int):
            raise TypeError(
                "density_noise[{}].after_gate_index should be an integer".format(
                    item_idx
                )
            )

        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(
                "density_noise[{}] has a non-finite value".format(item_idx)
            )

        normalized_density_noise.append(
            {
                "channel": canonical_channel,
                "target": int(target),
                "after_gate_index": int(after_gate_index),
                "value": value,
            }
        )

    return normalized_density_noise



##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_Variational_Quantum_Eigensolver_Base(qgd_Variational_Quantum_Eigensolver_Base_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix to be decomposed.
# @param config Dictionary describing optimization hyperparameters.
# @param accelerator_num Optional accelerator identifier.
# @param backend Optional backend selector. Supported values are
#   "state_vector" and "density_matrix". When omitted, the VQE keeps the
#   legacy state-vector behavior. Explicit `backend="density_matrix"` activates
#   the supported exact noisy mixed-state energy path for the Phase 2 anchor
#   workflow.
# @param density_noise Optional ordered list of fixed local density-noise
#   insertions. Each entry must define a channel, target qubit,
#   after_gate_index, and fixed noise value. The canonical Phase 2 local-noise
#   channels are `local_depolarizing`, `amplitude_damping`, and
#   `phase_damping`; the aliases `depolarizing` and `dephasing` normalize to
#   the local canonical names. Phase 2 only supports this surface together
#   with `backend="density_matrix"` on the supported `HEA` anchor circuit.
# @return An instance of the class
    def __init__(
        self,
        Hamiltonian,
        qbit_num,
        config=None,
        accelerator_num=0,
        *,
        backend=None,
        density_noise=None,
    ):
    

        if config is None:
            config = {}

        # config
        if not isinstance(config, dict):
            print("Input parameter config should be a dictionary describing the following hyperparameters:") #TODO
            return

        normalized_backend = _normalize_vqe_backend_name(backend)
        config_copy = dict(config)
        config_copy[_VQE_BACKEND_CONFIG_KEY] = _VQE_BACKEND_NAME_TO_CODE[normalized_backend]

        # call the constructor of the wrapper class
        super(qgd_Variational_Quantum_Eigensolver_Base, self).__init__(Hamiltonian.data, Hamiltonian.indices, Hamiltonian.indptr, qbit_num, config=config_copy, accelerator_num=accelerator_num)
        self.qbit_num = qbit_num
        # Keep the selected backend visible on the Python object so tests,
        # validation harnesses, and reproducibility artifacts can attribute
        # which execution path was requested.
        self.backend = normalized_backend
        self.density_noise = []
        self.set_Density_Matrix_Noise(density_noise)


## 
# @brief Call to set the optimizer used in the VQE process
# @param optimizer String indicating the optimizer. Possible values: "BFGS" ,"ADAM", "BFGS2", "ADAM_BATCHED", "AGENTS", "COSINE", "AGENTS_COMBINED".
    def set_Optimizer(self, alg):    

        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimizer(alg)

## 
# @brief Call to start solving the VQE problem to get the approximation for the ground state   
    def Start_Optimization(self):

	# call the C wrapper function
        super(qgd_Variational_Quantum_Eigensolver_Base, self).Start_Optimization()

## 
# @brief Call to get the optimized parameters set in numpy array
# @return Returns with the optimized parameters    
    def get_Optimized_Parameters(self):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Optimized_Parameters()


## 
# @brief Call to set the parameters which are used as a starting point in the optimization
# @param A numpy array containing the parameters. The number of parameters can be  retrieved with method get_Parameter_Num        
    def set_Optimized_Parameters(self, new_params):
        
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimized_Parameters(new_params)



# TODO should be deleted!        
    def set_Optimization_Tolerance(self, tolerance):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Optimization_Tolerance(tolerance)


##
# @brief Call to set the name of the SQUANDER project
# @param project_name_new new project name      
    def set_Project_Name(self, project_name):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Project_Name(project_name)

##
# @brief Call to set custom layers to the gate structure that are intended to be used in the decomposition from a binary file created from SQUANDER
# @param filename String containing the filename        
    def set_Gate_Structure_from_Binary(self, filename):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Gate_Structure_From_Binary(filename)


##
# @brief Call to set the ansatz type. Currently imp
# @param ansatz_new String of the ansatz . Possible values: "HEA" (hardware efficient ansatz with U3 and CNOT gates).
    def set_Ansatz(self, ansatz_new):
        
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Ansatz(ansatz_new)
        
##
# @brief Call to generate the circuit ansatz
# @param layers The number of layers. The depth of the generated circuit is 2*layers+1 (U3-CNOT-U3-CNOT...CNOT)
# @param inner_blocks The number of U3-CNOT repetition within a single layer
    def Generate_Circuit(self, layers, inner_blocks=1):
    
        super(qgd_Variational_Quantum_Eigensolver_Base, self).Generate_Circuit( layers, inner_blocks )
        
## 
# @brief Call to evaluate the VQE energy.
# @param parameters A float64 numpy array. The number of parameters can be  retrieved with method get_Parameter_Num 
    def Optimization_Problem(self, parameters):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).Optimization_Problem(parameters)

## 
# @brief Call to evaluate the VQE energy.
# @param parameters A float64 numpy array. The number of parameters can be  retrieved with method get_Parameter_Num 
    def Optimization_Problem_Grad(self, parameters):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).Optimization_Problem_Grad(parameters)

##
# @brief Call to get the second Rényi entropy
# @param parameters A float64 numpy array
# @param input_state A complex array storing the input state. If None |0> is created.
# @param qubit_list A subset of qubits for which the Rényi entropy should be calculated.
# @Return Returns with the calculated entropy
    def get_Second_Renyi_Entropy(self, parameters=None, input_state=None, qubit_list=None ):

        qbit_num = self.get_Qbit_Num()

        qubit_list_validated = list()
        if isinstance(qubit_list, list) or isinstance(qubit_list, tuple):
            for item in qubit_list:
                if isinstance(item, int):
                    qubit_list_validated.append(item)
                    qubit_list_validated = list(set(qubit_list_validated))
                else:
                    print("Elements of qbit_list should be integers")
                    return
        elif qubit_list == None:
            qubit_list_validated = [ x for x in range(qbit_num) ]

        else:
            print("Elements of qbit_list should be integers")
            return
        

        if parameters is None:
            print( "get_Second_Renyi_entropy: array of input parameters is None")
            return None


        if input_state is None:
            matrix_size = 1 << qbit_num
            input_state = np.zeros( (matrix_size,1) )
            input_state[0] = 1

        # evaluate the entropy
        entropy = super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Second_Renyi_Entropy( parameters, input_state, qubit_list_validated)  


        return entropy


##
# @brief Call to get the number of qubits in the circuit
# @return Returns with the number of qubits
    def get_Qbit_Num(self):
    
        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Qbit_Num()


##
# @brief Call to get the number of free parameters in the gate structure used for the decomposition
    def get_Parameter_Num( self ):

        return super(qgd_Variational_Quantum_Eigensolver_Base, self).get_Parameter_Num()
        
        

#@brief Call to apply the gate operation on the input state
#@param parameters_mtx Python array ontaining the parameter set
#@param state_to_be_transformed Numpy array storing the state on which the transformation should be applied
    def apply_to( self, parameters_mtx, state_to_be_transformed):

	# call the C wrapper function
        super().apply_to( parameters_mtx, state_to_be_transformed )


##
# @brief Call to retrieve the incorporated quantum circuit (Squander format)
# @return Return with a Qiskit compatible quantum circuit.
    def get_Circuit( self ):
        
        # call the C wrapper function
        return super().get_Circuit()



##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.        
    def get_Qiskit_Circuit(self):

        from squander import Qiskit_IO
        
        squander_circuit = self.get_Circuit()
        parameters       = self.get_Optimized_Parameters()
        
        return Qiskit_IO.get_Qiskit_Circuit( squander_circuit, parameters )


##
# @brief Call to get the number of free parameters in the gate structure used for the decomposition
    def set_Initial_State( self, initial_state ):

        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Initial_State( initial_state )


##
# @brief Configure ordered fixed local-noise insertions for the density backend.
# @param density_noise A list of dictionaries with channel, target,
#   after_gate_index, and noise value metadata. The supported required
#   Phase 2 local-noise vocabulary is `local_depolarizing`,
#   `amplitude_damping`, and `phase_damping`; `depolarizing` and `dephasing`
#   are accepted aliases that normalize to the canonical local names. Phase 2
#   treats this as a mixed-state-only surface and rejects it on
#   `state_vector` workflows. The supported positive path is the exact noisy
#   `HEA` anchor workflow on the `density_matrix` backend.
    def set_Density_Matrix_Noise(self, density_noise):

        normalized_density_noise = _normalize_density_noise_spec(density_noise)
        super(qgd_Variational_Quantum_Eigensolver_Base, self).set_Density_Matrix_Noise(
            normalized_density_noise
        )
        self.density_noise = [dict(item) for item in normalized_density_noise]


##
# @brief Return reviewable metadata for the currently supported density bridge.
# @return A dictionary describing the generated source, ordered bridge
#   operations, and fixed local-noise insertions used by the density path.
    def describe_density_bridge(self):

        bridge_metadata = super(
            qgd_Variational_Quantum_Eigensolver_Base, self
        ).get_Density_Matrix_Bridge_Metadata()
        bridge_metadata["density_noise"] = [dict(item) for item in self.density_noise]
        return bridge_metadata


##
# @brief Call to set custom gate structure to used in the decomposition
# @param Gate_structure An instance of SQUANDER Circuit
    def set_Gate_Structure( self, Gate_structure ):  

        if not isinstance(Gate_structure, qgd_Circuit) :
            raise Exception("Input parameter Gate_structure should be a an instance of Circuit")
                    
                    
        return super().set_Gate_Structure( Gate_structure )        
        
            

