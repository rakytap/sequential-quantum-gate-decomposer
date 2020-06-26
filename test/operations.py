# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:14:12 2020

@author: rakytap
"""

## @package pyexample
#  Documentation for this module.
#
#  More details.



def test_general_operation():
 
    from operations.Operation import Operation
    import numpy as np

    # define the nmumber of qubits spanning the matrices
    qbit_num = 4

    # creating gereal operation
    op = Operation( qbit_num )    
    
    # reorder qubits
    qbit_array = np.array([2, 1, 0, 3])
    op.reorder_qubits( qbit_array )
    
    
    
def test_U3_operation():
 
    from operations.U3 import U3
    import numpy as np

    # define the nmumber of qubits spanning the matrices
    qbit_num = 2
    
    # the target qbit of the U3 operation
    target_qbit = 1
        

    # creating gereal operation
    op = U3( qbit_num, target_qbit, Theta = True, Lambda = True )    
    
    # print the parameter identifiers
    print( op.parameters )
    print(' ')
    
    # check the matrix
    matrix = op.matrix(1,2,3)
    print(matrix)
    print(' ')
    
    # reorder qubits, and test the modified target qubit
    qbit_array = np.array([0, 1])
    op.reorder_qubits( qbit_array )
    if op.target_qbit != qbit_array[-target_qbit-1]:
        raise('Reordering qubits does not work properly')
        
    
    # check the reordered matrix
    matrix = op.matrix(1,2,3)
    print(matrix)
    print(' ')
    
    
    
def test_CNOT_operation():
 
    from operations.CNOT import CNOT
    import numpy as np

    # define the nmumber of qubits spanning the matrices
    qbit_num = 2
    
    # the target qbit of the U3 operation
    target_qbit = 0
    
    # the control qbit of the U3 operation
    control_qbit = 1
        

    # creating gereal operation
    op = CNOT( qbit_num, control_qbit, target_qbit )  
    
    # check the matrix
    matrix = op.matrix
    print(matrix)
    print(' ')
    
    # reorder qubits, and test the modified target qubit
    qbit_array = np.array([0, 1])
    op.reorder_qubits( qbit_array )
    if op.target_qbit != qbit_array[-target_qbit-1]:
        raise('Reordering qubits does not work properly')
        
    
    # check the reordered matrix
    matrix = op.matrix
    print(matrix)
    print(' ')    
    
    
    
