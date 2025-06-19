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
from squander import Circuit

import numpy as np
from qiskit import QuantumCircuit

from typing import List
from collections.abc import Callable

import multiprocessing as mp
from multiprocessing import Process, Pool
import os


from squander.partitioning.partition import (
    get_qubits,
    qasm_to_partitioned_circuit
)

import time
    

# subcircuit_decomposition_tolerance
tolerance = 1e-8

test_subcircuit_decomposition = True

squander_config = {  
            'strategy': "Tree_search", 
            'parallel': 0,
            'verbosity': 0, 
         }

filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"

max_partition_size = 3


# method to do the decomposition of the partitions
def DecomposePartition( Umtx: np.ndarray ) -> Circuit:
    """
    Call to run the decomposition of a given unitary Umtx, typically associated with the circuit 
    partition to be optimized

    
    Args:

        Umtx (np.ndarray) A complex typed unitary to be decomposed


    Return:

        Returns with the the decoposed circuit structure and with the corresponding gate parameters

    
    """ 
    if squander_config["strategy"] == "Tree_search":
        cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=squander_config, accelerator_num=0 )
    elif squander_config["strategy"] == "Tabu_search":
        cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=squander_config, accelerator_num=0 )

            
            
    cDecompose.set_Verbose( squander_config["verbosity"] )
    cDecompose.set_Cost_Function_Variant( 3 )	
    cDecompose.set_Optimization_Tolerance( tolerance )
    

    # adding new layer to the decomposition until threshold
    cDecompose.set_Optimizer( "BFGS" )

    # starting the decomposition
    cDecompose.Start_Decomposition()
            

    squander_circuit = cDecompose.get_Circuit()
    parameters       = cDecompose.get_Optimized_Parameters()


    print( "Decomposition error: ", cDecompose.get_Decomposition_Error() )

    if tolerance < cDecompose.get_Decomposition_Error():
        return None, None


    return squander_circuit, parameters



def CNOTGateCount( circ: Circuit ) -> int :
    """
    Call to get the number of CNOT gates in the circuit

    
    Args:

        circ (Circuit) A squander circuit representation


    Return:

        Returns with the CNOT gate count

    
    """ 

    if not isinstance(circ, Circuit ):
        raise Exception("The input parameters should be an instance of Squander Circuit")

    gate_counts = circ.get_Gate_Nums()

    return gate_counts.get('CNOT', 0)



def CompareAndPickCircuits( circs: List[Circuit], parameter_arrs: [List[np.ndarray]], metric : Callable[ [Circuit], int ] = CNOTGateCount ) -> Circuit:
    """
    Call to pick the most optimal circuit corresponding a specific metric. Looks for the circuit
    with the minimal metric value.

    
    Args:

        circs ( List[Circuit] ) A list of Squander circuits to be compared

        parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

        metric (optional) The metric function to decide which input circuit is better.


    Return:

        Returns with the chosen circuit and the corresponding parameter array

    
    """ 

    if not isinstance( circs, list ):
        raise Exception("First argument should be a list of squander circuits")

    if not isinstance( parameter_arrs, list ):
        raise Exception("Second argument should be a list of numpy arrays")

    if len(circs) != len(parameter_arrs) :
        raise Exception("The first two arguments should be of the same length")


    metrics = [metric( circ ) for circ in circs]

    metrics = np.array( metrics )

    min_idx = np.argmin( metrics )

    return circs[ min_idx ], parameter_arrs[ min_idx ]
    


def ConstructCircuitFromPartitions( circs: List[Circuit], parameter_arrs: [List[np.ndarray]] ) -> (Circuit, np.ndarray):
    """
    Call to construct the wide quantum circuit from the partitions.

    
    Args:

        circs ( List[Circuit] ) A list of Squander circuits to be compared

        parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

    Return:

        Returns with the constructed circuit and the corresponding parameter array

    
    """ 

    if not isinstance( circs, list ):
        raise Exception("First argument should be a list of squander circuits")

    if not isinstance( parameter_arrs, list ):
        raise Exception("Second argument should be a list of numpy arrays")

    if len(circs) != len(parameter_arrs) :
        raise Exception("The first two arguments should be of the same length")


    qbit_num = circs[0].get_Qbit_Num()



    wide_parameters = np.concatenate( parameter_arrs, axis=0 ) 


    wide_circuit = Circuit( qbit_num )

    for circ in circs:
        wide_circuit.add_Circuit( circ )


    assert wide_circuit.get_Parameter_Num() == wide_parameters.size, \
            f"Mismatch in the number of parameters: {wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"



    return wide_circuit, wide_parameters



def TestSubcircuitDecomposition( circ1: Circuit, parameters1: np.ndarray, circ2: Circuit, parameters2: np.ndarray) :
    """
    Call to test if the two circuits give the same state starnsformation upon a random input state

    
    Args:

        circ1 ( Circuit ) A circuit

        parameters1 ( np.ndarray ) A parameter array associated with the input circuit

        circ2 ( Circuit ) A circuit

        parameters2 ( np.ndarray ) A parameter array associated with the input circuit

    
    Return

        Returns with True if the two circuits give identical results.
    """ 


    qbit_num1 = circ1.get_Qbit_Num()
    qbit_num2 = circ2.get_Qbit_Num()

    if qbit_num1 != qbit_num2:
        raise Exception( "The two compared circuits should have the same number of qubits." )
    
    matrix_size = 1 << qbit_num1
    initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
    initial_state = initial_state_real + initial_state_imag*1j
    norm = np.sum(initial_state_real * initial_state_real + initial_state_imag*initial_state_imag)
    initial_state = initial_state/np.sqrt(norm)
    


    transformed_state_1 = initial_state.copy()
    transformed_state_2 = initial_state    
    
    circ1.apply_to( parameters1, transformed_state_1, parallel=squander_config['parallel'] )
    circ2.apply_to( parameters2, transformed_state_2, parallel=squander_config['parallel'])    
    
    overlap = np.sum( transformed_state_1.conj() * transformed_state_2 )
    #print( "overlap: ", np.abs(overlap) )

    assert( (np.abs(overlap)-1) < 1e-3 )



def PartitionDecompositionProcess( subcircuit: Circuit, subcircuit_parameters: np.ndarray ) -> (Circuit, np.ndarray):
    """
    Implements an asynchronous process to decompose a unitary associated with a partition in a large 
    quantum circuit

    
    Args:

        circ ( Circuit ) A subcircuit representing a partition

        parameters ( np.ndarray ) A parameter array associated with the input circuit

    
    """ 
    

    qbit_num_orig_circuit = subcircuit.get_Qbit_Num()
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
    decomposed_circuit, decomposed_parameters = DecomposePartition( unitary )

    if decomposed_circuit is None:
        decomposed_circuit = subcircuit
        decomposed_parameters = subcircuit_parameters


    # create inverse qbit map:
    inverse_qbit_map = {}
    for key, value in qbit_map.items():
        inverse_qbit_map[ value ] = key

    # remap the decomposed circuit in order to insert it into a large circuit
    new_subcircuit = decomposed_circuit.Remap_Qbits( inverse_qbit_map, qbit_num_orig_circuit )


    if test_subcircuit_decomposition:
        TestSubcircuitDecomposition( subcircuit, subcircuit_parameters, new_subcircuit, decomposed_parameters )
        


    new_subcircuit = new_subcircuit.get_Flat_Circuit()

    return new_subcircuit, decomposed_parameters
    

def OptimizeWideCircuit() -> (Circuit, np.ndarray):

    partitined_circuit, parameters = qasm_to_partitioned_circuit( filename, max_partition_size )


    qbit_num_orig_circuit = partitined_circuit.get_Qbit_Num()


    subcircuits = partitined_circuit.get_Gates()




    # the list of optimized subcircuits
    optimized_subcircuits = [None] * len(subcircuits)

    # the list of parameters associated with the optimized subcircuits
    optimized_parameter_list = [None] * len(subcircuits)

    # list of AsyncResult objects
    async_results = [None] * len(subcircuits)

    with Pool(processes=mp.cpu_count()) as pool:

        #  code for iterate over partitions and optimize them
        for partition_idx, subcircuit in enumerate( subcircuits ):
        

            # isolate the parameters corresponding to the given sub-circuit
            start_idx = subcircuit.get_Parameter_Start_Index()
            end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
            subcircuit_parameters = parameters[ start_idx:end_idx ]
   
    
            
            # callback function done on the master process to compare the new decomposed and the original suncircuit
            callback_fnc = lambda  x : CompareAndPickCircuits( [subcircuit, x[0]], [subcircuit_parameters, x[1]] )

            # call a process to decompose a subcircuit
            async_results[partition_idx]  = pool.apply_async( PartitionDecompositionProcess, (subcircuit, subcircuit_parameters,), callback=callback_fnc )



        #  code for iterate over async results and retrieve the new subcircuits
        for partition_idx, subcircuit in enumerate( subcircuits ):
    
            new_subcircuit, new_parameters = async_results[partition_idx].get( timeout = 1800 )

            '''
            if subcircuit != new_subcircuit:

                print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
                print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 
            '''

            optimized_subcircuits[ partition_idx ] = new_subcircuit
            optimized_parameter_list[ partition_idx ] = new_parameters


    # construct the wide circuit from the optimized suncircuits
    wide_circuit, wide_parameters = ConstructCircuitFromPartitions( optimized_subcircuits, optimized_parameter_list )


    print( "original circuit:    ", partitined_circuit.get_Gate_Nums()) 
    print( "reoptimized circuit: ", wide_circuit.get_Gate_Nums()) 



    if test_subcircuit_decomposition:
        TestSubcircuitDecomposition( partitined_circuit, parameters, wide_circuit, wide_parameters )
        


if __name__ == '__main__':

    start_time = time.time()

    OptimizeWideCircuit()

    print("--- %s seconds elapsed during optimization ---" % (time.time() - start_time))


