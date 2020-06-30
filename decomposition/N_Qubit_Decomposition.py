#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020

@author: rakytap
"""

from .Decomposition_Base import def_layer_num
from .Decomposition_Base import Decomposition_Base
from .Two_Qubit_Decomposition import Two_Qubit_Decomposition
from .Sub_Matrix_Decomposition import Sub_Matrix_Decomposition
import numpy as np
from numpy import linalg as LA

##
# @brief A class for the decomposition of N-qubit unitaries into U3 and CNOT gates.

class N_Qubit_Decomposition(Decomposition_Base):
    
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False ):  
        
        Decomposition_Base.__init__( self, Umtx ) 
            
        
        # logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
        self.optimize_layer_num  = optimize_layer_num
        
        # number of iteratrion loops in the finale optimalization
        self.iteration_loops = 5
        
        # The maximal allowed error of the optimalization problem
        self.optimalization_tolerance = 1e-7
        
        # Maximal number of iterations in the optimalization process
        self.max_iterations = int(1e4)
    
        # number of operators in one sub-layer of the optimalization process
        self.operation_layer = 1       
        
        # The number of gate blocks used in the decomposition
        self.max_layer_num = def_layer_num.get(str(self.qbit_num), 200)
        
        
                        
    
    
    
    
## start_decomposition
# @brief Start the disentanglig process of the least significant two qubit unitary
# @param varargin Cell array of optional parameters:
# @param 'param_num_layer' number of parameters in one sub-layer of the disentangling process
# @param 'operation_layer' number of operators in one sub-layer of the disentangling process
# @param 'max_iterations' Maximal number of iterations in the disentangling process
# @param 'max_layer_num' The maximal number of C-NOT gates allowed in the disentangling process
# ??????????????????????????????????
    def start_decomposition(self, operation_layer=None, max_iterations=None, max_layer_num=None, finalize_decomposition=True):
        
        
        if operation_layer is None:
            operation_layer = self.operation_layer
            
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        if max_layer_num is None:
            max_layer_num = self.max_layer_num
            
        
        print('***************************************************************')
        print('Starting to disentangle ' + str(self.qbit_num) + '-qubit matrix')
        print('***************************************************************')
        
            
        # create an instance of class to disentangle the given qubit pair
        cSub_decomposition = Sub_Matrix_Decomposition(self.Umtx, optimize_layer_num=self.optimize_layer_num)   
        
        # The maximal error of the optimalization problem
        cSub_decomposition.optimalization_tolerance = self.optimalization_tolerance
        
        
        # ---------- setting the decomposition parameters  --------
        
        # setting the maximal number of iterations in the disentangling process
        cSub_decomposition.operation_layer = operation_layer
        
        # setting the number of operators in one sub-layer of the disentangling process
        cSub_decomposition.max_iterations = max_iterations
        
        # setting the number of parameters in one sub-layer of the disentangling process
        cSub_decomposition.max_layer_num = max_layer_num
            
        # start to disentangle the qubit pair
        cSub_decomposition.disentangle_submatrices()
                                
        if not cSub_decomposition.subdisentaglement_done:
            return
        
        
        # saving the subunitarization operations
        self.save_subdecomposition_results( cSub_decomposition )
        
        # decompose the qubits in the disentangled submatrices
        self.decompose_submatrix()
            
        if finalize_decomposition:
            # finalizing the decompostition
            self.finalize_decomposition()
            
            # final tuning of the decomposition parameters
            self.final_optimalization()
            
            matrix_new = self.get_transformed_matrix(self.optimized_parameters, self.operations )    

            # calculating the final error of the decomposition
            self.decomposition_error = LA.norm(matrix_new*np.exp(np.complex(0,-np.angle(matrix_new[0,0]))) - np.identity(len(matrix_new))*abs(matrix_new[0,0]))
            
            # get the number of gates used in the decomposition
            gates_num = self.get_gate_nums()
            print( 'In the decomposition with error = ' + str(self.decomposition_error) + ' were used ' + str(self.layer_num) + ' layers with '  + str(gates_num['u3']) + ' U3 operations and ' + str(gates_num['cnot']) + ' CNOT gates.' )        
            
        
        
        
    
        
    
    
## save_subdecomposition_results
# @brief stores the calculated parameters and operations of the sub-decomposition processes
# @param cSub_decomposition An instance of class Sub_Two_Qubit_Decomposition used to disentangle qubit pairs from the others.
# @param qbits_reordered A permutation of qubits that was applied on the initial unitary in prior of the sub decomposition. (This is needed to restore the correct qubit indices.)
    def save_subdecomposition_results( self, cSub_decomposition ):
                
        # get the unitarization operations
        operations = cSub_decomposition.operations
        
        # get the unitarization parameters
        parameters = cSub_decomposition.optimized_parameters
        
        # create new operations in the original, not reordered qubit list
        for idx in range(len(operations)-1,-1,-1):
            operation = operations[idx]
            self.add_operation_to_front( operation )   
            
        self.optimized_parameters = np.concatenate((parameters, self.optimized_parameters))
            
            
    
    
## start_decomposition
# @brief Start the decompostion process of the two-qubit unitary
    def decompose_submatrix(self):
        
        if self.decomposition_finalized:
            print('Decomposition was already finalized')
            return
                       
        
        # obtaining the subdecomposed submatrices
        subdecomposed_mtx = self.get_transformed_matrix( self.optimized_parameters, self.operations )        
        
        # get the most unitary submatrix
        # get the number of 2qubit submatrices
        submatrices_num_row = 2#2^(self.qbit_num-2)
        
        # get the size of the submatrix
        submatrix_size = int(len(subdecomposed_mtx)/2)
        
        # fill up the submatrices
        unitary_error_min = None
        most_unitary_submatrix = []
        for idx in range(0,submatrices_num_row):
            for jdx in range(0,submatrices_num_row):
                submatrix = subdecomposed_mtx[ (idx*submatrix_size):((idx+1)*submatrix_size+1), (jdx*submatrix_size):((jdx+1)*submatrix_size+1) ]
                
                submatrix_prod = np.dot(submatrix, submatrix.conj().T)
                unitary_error = LA.norm( submatrix_prod - np.identity(len(submatrix_prod))*submatrix_prod[0,0] )
                if (unitary_error_min is None) or unitary_error < unitary_error_min:
                    unitary_error_min = unitary_error
                    most_unitary_submatrix = submatrix
                
            
        
        
        # if the qubit number in the submatirx is greater than 2 new N-qubit decomposition is started
        if len(most_unitary_submatrix) > 4:
            cdecomposition = N_Qubit_Decomposition(most_unitary_submatrix, optimize_layer_num=False)

            # starting the decomposition of the random unitary
            cdecomposition.start_decomposition(max_iterations=int(1e5), operation_layer=1, finalize_decomposition=False)
        else:

            # decompose the chosen 2-qubit unitary
            cdecomposition = Two_Qubit_Decomposition(most_unitary_submatrix)
        
            # starting the decomposition of the random unitary
            cdecomposition.start_decomposition()
        
                
        # saving the decomposition operations
        self.save_subdecomposition_results( cdecomposition )
        
        
     
    
    
    
##
# @brief final optimalization procedure improving the accuracy of the decompositin when all the qubits were already disentangled.
    def final_optimalization( self ):
        
        print(' ')
        print('Final fine tuning of the parameters')
        
        # setting the global minimum
        self.global_target_minimum = 0
        
        self.solve_optimalization_problem( optimalization_problem=self.final_optimalization_problem, solution_guess=self.optimized_parameters) 
        
        
    
    
    
    
## final_optimalization_problem
# @brief The optimalization problem to be solved in order to disentangle the two least significant qubits
# @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
# @return Returns with the value representing the difference between the decomposed matrix and the unity.
    def final_optimalization_problem( self, parameters ):
               
        # get the transformed matrix with the operations in the list
        matrix_new = self.get_transformed_matrix( parameters, self.operations, initial_matrix=self.Umtx)
                
        matrix_new = matrix_new - np.identity(len(matrix_new))*matrix_new[0,0]
        
        cost_function = np.sum( np.multiply(matrix_new, matrix_new.conj() ) )
        
        return cost_function
        
       
        
      
    
    
    
    
    


