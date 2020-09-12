#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
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

## \addtogroup python Python Interface
## @{
#
# ***********MOOOOOOOOORRRRRREEEEEEEEEEEEEE DEEEEEEEEEEEEETAAAAAAAAAAAAAAAAILSSSSSSSSSSSSSS HEEEEEEEEEEEEEEREEEEEEEEEEE ****************


import ctypes
import numpy as np
from os import path



#load qgd C library for the decomposition

if ( path.exists('.libs/libqgd.so') ):
    library_path = '.libs/libqgd.so'
elif ( path.exists('lib64/libqgd.so') ):
    library_path = 'lib64/libqgd.so'
elif ( path.exists('lib/libqgd.so') ):
    library_path = 'lib/libqgd.so'
else:
    print("Quantum Gate Decomposer library not found.")
    exit()

_qgd_library = ctypes.cdll.LoadLibrary(library_path)  




# defining the input/output arguments of the interface methods
_qgd_library.iface_new_N_Qubit_Decomposition.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool, ctypes.c_int,)
_qgd_library.iface_new_N_Qubit_Decomposition.restype = ctypes.c_void_p
_qgd_library.iface_start_decomposition.argtypes = (ctypes.c_void_p,)
_qgd_library.iface_delete_N_Qubit_Decomposition.argtypes = (ctypes.c_void_p,)
_qgd_library.iface_set_identical_blocks.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int,)
_qgd_library.iface_set_iteration_loops.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int,)
_qgd_library.iface_set_max_layer_num.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int,)
_qgd_library.iface_list_operations.argtypes = (ctypes.c_void_p, ctypes.c_int,)
_qgd_library.iface_get_operation_num.argtypes = ( ctypes.c_void_p, )
_qgd_library.iface_get_operation_num.restype = ctypes.c_int
_qgd_library.iface_get_operation.argtypes = ( ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), )
_qgd_library.iface_get_operation.restype = ctypes.c_int


##
# @brief A class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class N_Qubit_Decomposition:
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix
# @param initial_guess
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False, initial_guess="zeros" ):

        # determine the number of qubits
        self.qbit_num = int(round( np.log2( len(Umtx) ) ))

        matrix_size = int(2**self.qbit_num)


        # arranging all the elements of the matrix into one row (row major order)
        Umtx_real = np.real(Umtx).reshape(matrix_size*matrix_size)
        Umtx_imag = np.imag(Umtx).reshape(matrix_size*matrix_size)

        # definig ctypes array storing the real and imaginary parts of the unitary
        array_type = ctypes.c_double * (matrix_size*matrix_size)
        string_length = len(initial_guess)

        # convertin intial guess into numerical input
        if initial_guess=="zeros":
            initial_guess_num = 0
        elif initial_guess=="random":
             initial_guess_num = 1
        elif initial_guess=="close_to_zero":
             initial_guess_num = 2
        else:
             initial_guess_num = 0
             
        # create and store the instance of the N-Qubit decomposition class
        self.c_instance = _qgd_library.iface_new_N_Qubit_Decomposition( array_type(*Umtx_real), array_type(*Umtx_imag), ctypes.c_int(self.qbit_num), ctypes.c_bool(optimize_layer_num), ctypes.c_int(initial_guess_num) )


## 
# @brief Destructor of the class
    def __del__(self):

        # call to release the instance of  the class
        _qgd_library.iface_delete_N_Qubit_Decomposition( self.c_instance )
        
                

##
# @brief Start the disentanglig process of the least significant two qubit unitary
# @param finalize_decomposition Optional logical parameter. If true (default), the decoupled qubits are rotated into
# state |0> when the disentangling of the qubits is done. Set to False to omit this procedure
    def start_decomposition(self, finalize_decomposition=True):

        # start the decomposition routine in the C-class
        _qgd_library.iface_start_decomposition( self.c_instance )


##
# @brief Set the maximal number of layers used in the subdecomposition of the qbit-th qubit.
# @param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.
# @param max_layer_num The maximal number of the operation layers used in the subdecomposition.
    def set_max_layer_num(self, qbit, max_layer_num ):

        # Set the maximal number of layers throug the C-interface
        _qgd_library.iface_set_max_layer_num( self.c_instance, ctypes.c_int(qbit), ctypes.c_int(max_layer_num) )

##
# @brief Set the number of iteration loops during the subdecomposition of the qbit-th qubit.
# @param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.,
# @param iteration_loops The number of iteration loops in each sted of the subdecomposition.
    def set_iteration_loops(self, qbit, iteration_loops ):

        # Set the number of iteration loops throug the C-interface
        _qgd_library.iface_set_iteration_loops( self.c_instance, ctypes.c_int(qbit), ctypes.c_int(iteration_loops) )


##
# @brief Set the number of identical successive blocks during the subdecomposition of the qbit-th qubit.
# @param qbit The number of qubits for which the maximal number of layers should be used in the subdecomposition.
# @param identical_blocks The number of successive identical layers used in the subdecomposition.
    def set_identical_blocks(self, qbit, identical_blocks ):

        # Set the number of identical successive blocks  throug the C-interface
        _qgd_library.iface_set_identical_blocks( self.c_instance, ctypes.c_int(qbit), ctypes.c_int(identical_blocks) )


##
# @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
# @param start_index The index of the first inverse operation
    def list_operations(self, start_index=1 ):

        _qgd_library.iface_list_operations( self.c_instance, ctypes.c_int(start_index) );


##
# @brief Export the unitary decomposition into Qiskit format.
# @param start_index The index of the first inverse operation
    def get_quantum_circuit( self ):

        from qiskit import QuantumCircuit

        # creating Qiskit quantum circuit
        circuit = QuantumCircuit(self.qbit_num)

        # fill up the quantum circuit witj operations

        # create Ctypes compatible wrapper variables for the operation parameters
        operation_type = ctypes.c_int()
        target_qbit = ctypes.c_int()
        control_qbit = ctypes.c_int()

        array_type = ctypes.c_double * 3
        parameters = array_type(*np.array([0,0,0]))
      
        number_of_decomposing_operations = _qgd_library.iface_get_operation_num( self.c_instance )
        op_idx = ctypes.c_int(number_of_decomposing_operations-1)

        while True:

            # retrive the parameters of the op_idx-th operation
            status = _qgd_library.iface_get_operation( self.c_instance, op_idx, ctypes.byref(operation_type), ctypes.byref(target_qbit), ctypes.byref(control_qbit), parameters )

            if not ( status == 0 ) :
                break
       
            #print( "op_type python: " + str(operation_type.value))
            #print("status: " + str(status))
            #if ( operation_type.value == 2 ) :
            #    for i in parameters: print(i, end=" ")
            #    print(' ')
            #    print( parameters[0] )

            if operation_type.value == 1:
                # adding CNOT operation to the quantum circuit
                circuit.cx(control_qbit.value, target_qbit.value)

            elif operation_type.value == 2:
                # adding U3 operation to the quantum circuit
                circuit.u3(parameters[0], parameters[1], parameters[2], target_qbit.value)
                pass

            op_idx.value = op_idx.value - 1


        return circuit


            
## @}

 



        
        
            

