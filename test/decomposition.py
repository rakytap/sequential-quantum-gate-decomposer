#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:23:58 2020
Copyright (C) 2020 Peter Rakyta, Ph.D.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

from qiskit import execute
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
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')
    
    # Creating the class to decompose the 2-qubit unitary
    # as the input the hermitian conjugate id given ti the class 
    # (The decomposition procedure brings the input matrix into identity)
    cDecomposition = Two_Qubit_Decomposition( Umtx.conj().T )
    
    #start the decomposition
    cDecomposition.start_decomposition()
    
    print('')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operations()
    
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
    print('This might take about 1 mins')
    print(' ')
    
    # cerate unitary q-bit matrix
    from scipy.stats import unitary_group
        
    # the number of qubits
    qbit_num = 3
    
    # generating 3-qubit general random unitary
    Umtx = unitary_group.rvs(int(2**qbit_num))
    
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')

    # Creating the class to decompose the 3-qubit unitary
    # as the input the hermitian conjugate id given ti the class 
    # (The decomposition procedure brings the input matrix into identity)
    cDecomposition = N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=False, identical_blocks={'3':1} )
    
    #start the decomposition
    cDecomposition.start_decomposition()
    
    print(' ')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operations()  
    
    # Constructing quantum circuit
    quantum_circuit = cDecomposition.get_quantum_circuit()
        
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
    print('Test of four-qubit decomposition: IBM challenge')
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

    # creating class to decompose the matrix
    cDecomposition = N_Qubit_Decomposition( Umtx.conj().T, optimize_layer_num=True, identical_blocks=identical_blocks, initial_guess= 'zeros' )
    cDecomposition.max_layer_num = 9
    
    #start the decomposition
    cDecomposition.start_decomposition()
        
    print(' ')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operations() 

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
    decomposition_error =  LA.norm(product_matrix - np.identity(16)*product_matrix[0,0], 2)
    
    print('The error of the decomposition is ' + str(decomposition_error))
    
    
    
def four_qubit_decomposition():
 
    from decomposition.N_Qubit_Decomposition import N_Qubit_Decomposition
    
    print('****************************************')
    print('Test of four-qubit decomposition')
    print('This might take about 1.5 hour')
    print(' ')
    
    # cerate unitary q-bit matrix
    from scipy.stats import unitary_group
    
    # the number of qubits
    qbit_num = 4
    
    Umtx = unitary_group.rvs(int(2**qbit_num))
    print('The test matrix to be decomposed is:')
    print(Umtx)
    print(' ')

    cDecomposition = N_Qubit_Decomposition( Umtx.conj().T, parallel = True, identical_blocks={'4':2, '3':1} )
    
    # Maximal number of iteartions in the optimalization process
    cDecomposition.set_max_iteration( int(1e6) )
    
    # Set the tolerance of the minimum of the cost function during the optimalization
    cDecomposition.optimalization_tolerance = 1e-7
    
    #start the decomposition
    cDecomposition.start_decomposition()
        
    print(' ')
    print('The matrix can be decomposed into operations:')
    print(' ')
    cDecomposition.list_operations() 
    
    # Constructing quantum circuit
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
        