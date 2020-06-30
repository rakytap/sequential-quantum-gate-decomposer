#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:53:10 2020

@author: rakytap
"""

from .Decomposition_Base import Decomposition_Base
from operations.operation_block import operation_block
import numpy as np


##
# @brief A class for the decomposition of two-qubit unitaries.

class Two_Qubit_Decomposition( Decomposition_Base ):
    
    
    

##
# @brief Constructor of the class.
# @param Umtx The unitary matrix
# ?????????????
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False ):
        
        Decomposition_Base.__init__( self, Umtx )
         
        
        # logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
        self.optimize_layer_num  = optimize_layer_num        
        
        
        # The global minimum of the optimalization problem
        self.global_target_minimum = 0
        
        # number of iteratrion loops in the finale optimalization
        self.iteration_loops = 3
        
        # The maximal allowed error of the optimalization problem
        self.optimalization_tolerance = 1e-7
        
        # Maximal number of iteartions in the optimalization process
        self.max_iterations = int(1e4)
    
        # number of operators in one sub-layer of the optimalization process
        self.operation_layer = int(1)
        
        # The maximal number of C-NOT gates allowed in the decomposition
        self.max_layer_num = 3
        
        
        
    
    
    
## start_decomposition
# @brief Start the decompostion process of the two-qubit unitary
# @param operation_layer ?????????????????    
    def start_decomposition(self, operation_layer=None):
        
        if self.decomposition_finalized:
            print('Decomposition was already finalized')
            return
        

        if not (operation_layer is None):
            self.operation_layer = operation_layer
        
        
        
        # setting the global target minimum
        self.global_target_minimum = 0
        
        # check whether the problem can be solved without optimalization
        if not self.test_indepency():
            
            # Do the optimalization of the parameters
            while self.layer_num < self.max_layer_num :  
                
                # creating block of operations
                block = operation_block( self.qbit_num )
                    
                # add CNOT gate to the block
                block.add_cnot_to_end(1, 0)      
                    
                # adding U3 operation to the block
                block.add_u3_to_end(1, Theta=True, Lambda=True) 
                block.add_u3_to_end(0, Theta=True, Lambda=True)
                    
                # adding the opeartion block to the operations
                self.add_operation_to_end( block )
                
                # set the number of layers in the optimalization
                self.operation_layer = self.layer_num
                
                # Do the optimalization
                if self.optimize_layer_num or self.layer_num >= self.max_layer_num :
                    # solve the optzimalization problem to find the correct mninimum
                    self.solve_optimalization_problem( optimalization_problem = self.optimalization_problem)

                    if self.check_optimalization_solution():
                        break
                    
                
        
        
        # check the solution
        if self.check_optimalization_solution():
                
            # logical value describing whether the first optimalization problem was solved or not
            self.optimalization_problem_solved = True
            
        else:
            # setting the logical variable to true even if no optimalization was needed
            self.optimalization_problem_solved = False
        
                
        
    
        
        

    
        
## optimalization_problem
# @brief The optimalization problem to be solved in order to disentangle the qubits
# @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
# @param operations_post A matrix of the product of operations which are applied after the operations to be optimalized in the sub-layer optimalization problem.
# @param operations_pre A matrix of the product of operations which are applied in prior the operations to be optimalized in the sub-layer optimalization problem.
# @return Returns with the value representing the entaglement of the qubits. (gives zero if the two qubits are decoupled.)
    def optimalization_problem( self, parameters ):
               
        # get the transformed matrix with the operations in the list
        matrix_new = self.get_transformed_matrix( parameters, self.operations, initial_matrix = self.Umtx )
                
        submatrices = list()
        submatrices.append(matrix_new[0:2, 0:2])
        submatrices.append(matrix_new[0:2, 2:4])
        submatrices.append(matrix_new[2:4, 0:2])
        submatrices.append(matrix_new[2:4, 2:4])
        
        cost_function = 0;
        
        for idx in range(0,4):
            for jdx in range(idx,4):
                
                submatrix_prod = np.dot(submatrices[idx],submatrices[jdx].conj().T)
                #print(submatrix_prod)
                #print(' ')
                submatrix_prod = submatrix_prod - np.identity(len(submatrix_prod))*submatrix_prod[0,0]
                cost_function = cost_function + np.sum( np.multiply(submatrix_prod, submatrix_prod.conj() ) )
                
                
        return np.real(cost_function)
                
                                        
                    

    
## test_indepency
# @brief Check whether qubits are indepent or not
# @returns Return with true if qubits are disentangled, or false otherwise.
    def test_indepency( self ):
       
        self.current_minimum = self.optimalization_problem( self.optimized_parameters )
        
        ret = self.check_optimalization_solution()
        
        return ret      
        
        
        
    

   
    
