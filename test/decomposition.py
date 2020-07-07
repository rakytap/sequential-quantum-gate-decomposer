#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:23:58 2020

@author: rakytap
"""

from qiskit import *
from qiskit import Aer
import numpy as np

def two_qubit_decomposition():
 
    from decomposition.Two_Qubit_Decomposition import Two_Qubit_Decomposition
    
    print('****************************************')
    print('Test of two qubit decomposition')
    print(' ')
    
    # cerate unitary q-bit matrix
    from scipy.stats import unitary_group

    
    # the number of qubits
    qbit_num = 2
    
    Umtx = unitary_group.rvs(int(2**qbit_num))
    #Umtx = np.identity(4)
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')

    cDecomposition = Two_Qubit_Decomposition( Umtx )
    
    #start the decomposition
    cDecomposition.start_decomposition()
    
    print('')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operation_inverses()
    
    print(' ')
    print('Constructing quantum circuit:')
    print(' ')
    quantum_circuit = cDecomposition.get_quantum_circuit()
    
    print(quantum_circuit)
    
    # test the decomposition of the matrix
    #Changing the simulator 
    backend = Aer.get_backend('unitary_simulator')
    
    #job execution and getting the result as an object
    job = execute(quantum_circuit, backend)
    result = job.result()
    
    #get the unitary matrix from the result object
    decomposed_matrix = result.get_unitary(quantum_circuit)
    
    # get the error of the decomposition
    product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
    decomposition_error =  np.linalg.norm(product_matrix - np.identity(2**qbit_num)*product_matrix[0,0], 2)
    
    print('The error of the decomposition is ' + str(decomposition_error))
    
    
    
    
def three_qubit_decomposition():
 
    from decomposition.N_Qubit_Decomposition import N_Qubit_Decomposition
    
    print('****************************************')
    print('Test of three-qubit decomposition')
    print(' ')
    
    # cerate unitary q-bit matrix
    from scipy.stats import unitary_group
        
    # the number of qubits
    qbit_num = 3
    
    Umtx = unitary_group.rvs(int(2**qbit_num))
    
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')
    #print(np.dot(Umtx, Umtx.conj().T))

    cDecomposition = N_Qubit_Decomposition( Umtx, optimize_layer_num=False, identical_blocks=2 )
    
    #start the decomposition
    cDecomposition.start_decomposition()
    
    print(' ')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operation_inverses()  
    
    print(' ')
    print('Constructing quantum circuit:')
    print(' ')
    quantum_circuit = cDecomposition.get_quantum_circuit()
    
    #print(quantum_circuit)
    
    # test the decomposition of the matrix
    #Changing the simulator 
    backend = Aer.get_backend('unitary_simulator')
    
    #job execution and getting the result as an object
    job = execute(quantum_circuit, backend)
    result = job.result()
        
    #get the unitary matrix from the result object
    decomposed_matrix = result.get_unitary(quantum_circuit)
    
    # get the error of the decomposition
    product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
    decomposition_error =  np.linalg.norm(product_matrix - np.identity(2**qbit_num)*product_matrix[0,0], 2)
    
    print('The error of the decomposition is ' + str(decomposition_error))
    
    
    
def four_qubit_decomposition():
 
    from decomposition.N_Qubit_Decomposition import N_Qubit_Decomposition
    
    print('****************************************')
    print('Test of four-qubit decomposition')
    print(' ')
    
    # cerate unitary q-bit matrix
    from scipy.stats import unitary_group
    
    # the number of qubits
    qbit_num = 4
    
    Umtx = unitary_group.rvs(int(2**qbit_num))
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')

    cDecomposition = N_Qubit_Decomposition( Umtx, parallel = True, identical_blocks=1 )
    
    # Maximal number of iteartions in the optimalization process
    cDecomposition.set_max_iteration( int(1e6) )
    
    cDecomposition.optimalization_tolerance = 1e-13
    
    #start the decomposition
    cDecomposition.start_decomposition()
        
    print(' ')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operation_inverses() 
    
    print(' ')
    print('Constructing quantum circuit:')
    print(' ')
    quantum_circuit = cDecomposition.get_quantum_circuit()
    
    #print(quantum_circuit)
    
    # test the decomposition of the matrix
    #Changing the simulator 
    backend = Aer.get_backend('unitary_simulator')
    
    #job execution and getting the result as an object
    job = execute(quantum_circuit, backend)
    result = job.result()
    
    #get the unitary matrix from the result object
    decomposed_matrix = result.get_unitary(quantum_circuit)
    
    # get the error of the decomposition
    product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
    decomposition_error =  np.linalg.norm(product_matrix - np.identity(2**qbit_num)*product_matrix[0,0], 2)
    
    print('The error of the decomposition is ' + str(decomposition_error))




def IBM_challenge_decomposition():
 
    from decomposition.N_Qubit_Decomposition import N_Qubit_Decomposition
    
    print('****************************************')
    print('Test of four-qubit decomposition')
    print(' ')
    
    # cerate unitary q-bit matrix
    import numpy as np
    import numpy.linalg as LA
    from scipy.io import loadmat
    
    # load the matrix from file
    data = loadmat('test/Umtx.mat')    
    Umtx = data['Umtx']
    
    
    # The unitary given in the 4th problem of IBM chellenge 2020

    print('The test matrix to be decomposed is:')
    #print(Umtx)

    # set the number of successive identical blocks in the optimalization of disentanglement of the n-th qubits
    identical_blocks = { '4':2, '3':1}

    cDecomposition = N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=True, identical_blocks=identical_blocks )
    cDecomposition.max_layer_num = 9
    cDecomposition.optimalization_block = 1
    
    cDecomposition.reorder_qubits([3, 2, 1, 0])
    
    #start the decomposition
    cDecomposition.start_decomposition()
        
    print(' ')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operation_inverses() 

    print(' ')
    print('Constructing quantum circuit:')
    print(' ')
    quantum_circuit = cDecomposition.get_quantum_circuit_decomposition()
    
    #print(quantum_circuit)
    
    # test the decomposition of the matrix
    #Changing the simulator 
    backend = Aer.get_backend('unitary_simulator')
    
    #job execution and getting the result as an object
    job = execute(quantum_circuit, backend)
    result = job.result()
    
    #get the unitary matrix from the result object
    decomposed_matrix = result.get_unitary(quantum_circuit)
    
    # get the error of the decomposition
    product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
    decomposition_error =  np.linalg.norm(product_matrix - np.identity(16)*product_matrix[0,0], 2)
    
    print('The error of the decomposition is ' + str(decomposition_error))
        