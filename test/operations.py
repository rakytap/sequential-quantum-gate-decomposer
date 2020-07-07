# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:14:12 2020
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

## @package pyexample
#  Documentation for this module.
#
#  More details.



def test_general_operation():
 
    from operations.Operation import Operation
    import numpy as np
    
    print('****************************************')
    print('Test of general operation')
    print(' ')

    # define the nmumber of qubits spanning the matrices
    qbit_num = 4

    # creating gereal operation
    op = Operation( qbit_num )    
    
    # reorder qubits
    qbit_list = [2, 1, 0, 3]
    op.reorder_qubits( qbit_list )
    
    
    
def test_U3_operation():
 
    from operations.U3 import U3
    import numpy as np
    
    print('****************************************')
    print('Test of operation U3')
    print(' ')

    # define the nmumber of qubits spanning the matrices
    qbit_num = 2
    
    # the target qbit of the U3 operation
    target_qbit = 1
        

    # creating gereal operation
    op = U3( qbit_num, target_qbit, ['Theta', 'Lambda'] )    
    
    # print the parameter identifiers
    print('The parameetrs of the U3 operation:')
    print( op.parameters )
    print(' ')
    
    # check the matrix
    print('The matrix of ' + str(qbit_num) + ' qubit U3 operator acting on target qubit ' + str(op.target_qbit) )
    matrix = op.matrix([1,2])
    print(matrix)
    print(' ')
    
    # reorder qubits, and test the modified target qubit
    qbit_list = [0, 1]
    op.reorder_qubits( qbit_list )
    if op.target_qbit != qbit_list[-target_qbit-1]:
        raise('Reordering qubits does not work properly')
        
    
    # check the reordered matrix
    print('The matrix of ' + str(qbit_num) + ' qubit U3 operator acting on target qubit ' + str(op.target_qbit) )
    matrix = op.matrix([1,2])
    print(matrix)
    print(' ')
    
    
    
def test_CNOT_operation():
 
    from operations.CNOT import CNOT
    import numpy as np
    
    print('****************************************')
    print('Test of operation CNOT')
    print(' ')

    # define the nmumber of qubits spanning the matrices
    qbit_num = 3
    
    # the target qbit of the U3 operation
    target_qbit = 0
    
    # the control qbit of the U3 operation
    control_qbit = 1
        

    # creating gereal operation
    op = CNOT( qbit_num, control_qbit, target_qbit )  
    
    # check the CNOT matrix
    matrix = op.matrix
    print('The matrix of ' + str(qbit_num) + ' qubit CNOT operator with control qubit ' + str(op.control_qbit) + ' and target qubit ' + str(op.target_qbit) )
    print(matrix)
    print(' ')
    
    # reorder qubits, and test the modified target qubit
    qbit_list = [2, 0, 1]
    op.reorder_qubits( qbit_list )
    if op.target_qbit != qbit_list[-target_qbit-1]:
        raise('Reordering qubits does not work properly')
        
    
    # check the reordered CNOT matrix
    matrix = op.matrix
    print('The matrix of ' + str(qbit_num) + ' qubit CNOT operator with control qubit ' + str(op.control_qbit) + ' and target qubit ' + str(op.target_qbit) )
    print(matrix)
    print(' ')    
    
    
    
def test_operations():
 
    from operations.Operations import Operations
    import numpy as np
    
    print('****************************************')
    print('Test of operations')
    print(' ')
    

    # define the nmumber of qubits spanning the matrices
    qbit_num = 3
    
    # create class intance storing quantum gate operations
    operations = Operations( qbit_num )
    
    # adding operations to the list
    operations.add_u3_to_end(1, ['Theta', 'Lambda'])
    operations.add_cnot_to_end(1, 2)
    
    # get the number of parameters
    print( 'The number of parameters in the list of operations is ' + str(operations.parameter_num))
    print(' ')
    
   
    
    
