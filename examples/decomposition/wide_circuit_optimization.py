# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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
## \file wide_circuit_optimization.py
## \brief Simple example python code demonstrating a wide circuit optimization

from squander import Wide_Circuit_Optimization
from squander import N_Qubit_Decomposition_Tree_Search
from squander import N_Qubit_Decomposition_Tabu_Search

import numpy as np
from qiskit import QuantumCircuit


from squander.partitioning.partition import (
    get_qubits,
    qasm_to_partitioned_circuit
)



squander_config = {  
            'strategy': "Tabu_search", 
            'parallel': 1,
            'verbosity': 0, 
         }

filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"

max_partition_size = 3
partitined_circuit, parameters = qasm_to_partitioned_circuit( filename, max_partition_size )

qbit_num_orig_circuit = partitined_circuit.get_Qbit_Num()


subcircuits = partitined_circuit.get_Gates()

# method to do the decomposition of the partitions
def decompose_partition( Umtx ):

    if squander_config["strategy"] == "Tree_search":
        cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=squander_config, accelerator_num=0 )
    elif squander_config["strategy"] == "Tabu_search":
        cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=squander_config, accelerator_num=0 )

            
            
    cDecompose.set_Verbose( squander_config["verbosity"] )
    cDecompose.set_Cost_Function_Variant( 3)	
    

    # adding new layer to the decomposition until threshold
    cDecompose.set_Optimizer( "BFGS" )

    # starting the decomposition
    cDecompose.Start_Decomposition()
            

    squander_circuit = cDecompose.get_Circuit()
    parameters       = cDecompose.get_Optimized_Parameters()


    print( "Decomposition error: ", cDecompose.get_Decomposition_Error() )

    return squander_circuit, parameters


#  code for iterate over partitions and optimize them
for subcircuit in subcircuits:

    

    # isolate the parameters corresponding to the given sub-circuit
    start_idx = subcircuit.get_Parameter_Start_Index()
    end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
    subcircuit_parameters = parameters[ start_idx:end_idx ]
    

    involved_qbits = subcircuit.get_Qbits()
    qbit_num = len( involved_qbits )

    # create qbit map:
    qbit_map = {}
    for idx in range( len(involved_qbits) ):
        qbit_map[ involved_qbits[idx] ] = idx

    # remap the subcircuit to a smaller qubit register
    remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )

    # get the unitary representing the circuit
    unitary = remapped_subcircuit.get_Matrix( subcircuit_parameters )

    # decompose a small unitary into a new circuit
    decomposed_circuit, decomposed_parameters = decompose_partition( unitary )


    # create inverse qbit map:
    inverse_qbit_map = {}
    for key, value in qbit_map.items():
        inverse_qbit_map[ value ] = key

    # remap the decomposed circuit in order to insert it into a large circuit
    inverse_remapped_subcircuit = decomposed_circuit.Remap_Qbits( inverse_qbit_map, qbit_num_orig_circuit )
    
    print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
    print( "reoptimized subcircuit: ", inverse_remapped_subcircuit.get_Gate_Nums()) 

    ##################################xx
    # testing the correctness of the original sub circuit and remapped->decomposed->inverse remapped circuit
    matrix_size = 1 << qbit_num_orig_circuit
    initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state = initial_state_real + initial_state_imag*1j
    initial_state = initial_state/np.linalg.norm(initial_state)
    


    transformed_state_1 = initial_state.copy()
    transformed_state_2 = initial_state.copy()    
    
    subcircuit.apply_to( subcircuit_parameters, transformed_state_1 )
    inverse_remapped_subcircuit.apply_to( decomposed_parameters, transformed_state_2)    
    
    overlap = transformed_state_1.transpose().conj() @ transformed_state_2
    print( "overlap: ", np.abs(overlap) )
    ##################################xx    
 
