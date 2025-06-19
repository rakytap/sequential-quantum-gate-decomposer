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

# subcircuit_decomposition_tolerance
tolerance = 1e-8

test_subcircuit_decomposition = True

squander_config = {  
            'strategy': "Tabu_search", 
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
        # testing the correctness of the original sub circuit and decomposed circuit
        matrix_size = 1 << qbit_num_orig_circuit
        initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
        initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
        initial_state = initial_state_real + initial_state_imag*1j
        initial_state = initial_state/np.linalg.norm(initial_state)
    


        transformed_state_1 = initial_state.copy()
        transformed_state_2 = initial_state.copy()    
    
        subcircuit.apply_to( subcircuit_parameters, transformed_state_1, parallel=0 )
        new_subcircuit.apply_to( decomposed_parameters, transformed_state_2, parallel=0)    
    
        overlap = transformed_state_1.transpose().conj() @ transformed_state_2
        print( "overlap: ", np.abs(overlap) )

        assert( (np.abs(overlap)-1) < 1e-3 )



    new_subcircuit = new_subcircuit.get_Flat_Circuit()

    gates = new_subcircuit.get_Gates()
    print( gates )


    print( "ccccc ", new_subcircuit.get_Parameter_Num() == decomposed_parameters.size, decomposed_parameters.size, new_subcircuit.get_Parameter_Num() )

    return new_subcircuit, decomposed_parameters
    

def OptimizeWideCircuit() -> (Circuit, np.ndarray):

    partitined_circuit, parameters = qasm_to_partitioned_circuit( filename, max_partition_size )


    qbit_num_orig_circuit = partitined_circuit.get_Qbit_Num()


    subcircuits = partitined_circuit.get_Gates()




    # the list of optimized subcircuits
    optimized_subcircuits = [None] * len(subcircuits)

    # the list of parameters associated with the optimized subcircuits
    optimized_parameter_list = [None] * len(subcircuits)


    #with Pool(processes=1) as pool:

    #  code for iterate over partitions and optimize them
    for partition_idx, subcircuit in enumerate( subcircuits ):
        
        print( partition_idx )

        # isolate the parameters corresponding to the given sub-circuit
        start_idx = subcircuit.get_Parameter_Start_Index()
        end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
        subcircuit_parameters = parameters[ start_idx:end_idx ]

        print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqq ")
            #gates = subcircuit.get_Gates()
            #print( gates )

   
    
        '''
            # call a process to decompose a subcircuit
            res = pool.apply_async( PartitionDecompositionProcess, (subcircuit, subcircuit_parameters,) )
            new_subcircuit, decomposed_parameters = res.get( timeout = 1800 )

            print( "ppppp ", new_subcircuit.get_Parameter_Num() == decomposed_parameters.size, decomposed_parameters.size, new_subcircuit.get_Parameter_Num() )

            gates = new_subcircuit.get_Gates()
            print( gates )
        '''
            #continue
        new_subcircuit, decomposed_parameters = PartitionDecompositionProcess( subcircuit, subcircuit_parameters )

 
        # pick up the better decomposition of the partition
        new_subcircuit, new_parameters = CompareAndPickCircuits( [subcircuit, new_subcircuit], [subcircuit_parameters, decomposed_parameters] )


        if subcircuit != new_subcircuit:

            print( "original subcircuit:    ", subcircuit.get_Gate_Nums()) 
            print( "reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums()) 


        optimized_subcircuits[ partition_idx ] = new_subcircuit
        optimized_parameter_list[ partition_idx ] = new_parameters


    # construct the wide circuit from the optimized suncircuits
    wide_circuit, wide_parameters = ConstructCircuitFromPartitions( optimized_subcircuits, optimized_parameter_list )


    print( "original circuit:    ", partitined_circuit.get_Gate_Nums()) 
    print( "reoptimized circuit: ", wide_circuit.get_Gate_Nums()) 



    if test_subcircuit_decomposition:
        # testing the correctness of the original sub circuit and decomposed circuit
        matrix_size = 1 << qbit_num_orig_circuit
        initial_state_real = np.random.uniform(-1.0,1.0, (matrix_size,) )
        initial_state_imag = np.random.uniform(-1.0,1.0, (matrix_size,) )
        initial_state = initial_state_real + initial_state_imag*1j
        initial_state = initial_state/np.linalg.norm(initial_state)
    


        transformed_state_1 = initial_state.copy()
        transformed_state_2 = initial_state.copy()    
    
        partitined_circuit.apply_to( parameters, transformed_state_1 )
        wide_circuit.apply_to( wide_parameters, transformed_state_2)    
    
        overlap = transformed_state_1.transpose().conj() @ transformed_state_2
        print( "overlap: ", np.abs(overlap) )

        assert( (np.abs(overlap)-1) < 1e-3 )


import multiprocessing as mp
from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)


if __name__ == '__main__':

    OptimizeWideCircuit()

    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

    print( f"hhhhhhhhhhhhhh {os.cpu_count()}" )



