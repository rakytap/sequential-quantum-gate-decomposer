# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:13:26 2020

@author: Dr. Peter Rakyta
"""
##
# @brief A base class responsible for constructing matrices of C-NOT, U3
# gates acting on the N-qubit space


class Operation():
    
    ##
    # @brief Constructor of the class.
    # @param qbit_num The number of qubits in the unitaries
    # @return An instance of the class
    def __init__(self, qbit_num):
        # number of qubits spanning the matrix of the operation
        self.qbit_num = qbit_num
        # A string describing the type of the operation
        self.type = 'general'
        # A list of parameter names which are used to evaluate the matrix of the operation
        self.parameters = list()
        # The index of the qubit on which the operation acts (target_qbit >= 0) 
        self.target_qbit = None
        # The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        self.control_qbit = None      
        # The matrix (or function handle to generate the matrix) of the operation
        self.matrix = None
   
    ##
    # @brief Set the number of qubits spanning the matrix of the operation
    # @param qbit_num The number of qubits spanning the matrix
    def set_qbit_num( self, qbit_num ):
        # setting the number of qubits
        self.qbit_num = qbit_num
     
    ##
    # @brief Call to reored the qubits in the matrix of the operation
    # @param qbit_list The list of qubits spanning the matrix
    def reorder_qubits( self, qbit_list ):
        
        # check the number of qubits
        if len( qbit_list ) != self.qbit_num:
            raise BaseException('Wrong number of qubits')
        
        # setting the new value for the target qubit
        if self.target_qbit != None :
            self.target_qbit = qbit_list[-self.target_qbit-1]
        