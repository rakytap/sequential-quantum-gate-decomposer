#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:31:58 2020

@author: rakytap
"""

from .Decomposition_Base import Decomposition_Base
from operations.operation_block import operation_block
import numpy as np
import time

class Sub_Matrix_Decomposition(Decomposition_Base):
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix
# @param optimize_layer_num A logical value. Set true in order to optimize the number of C-NOT gates in the decomposition (this can significantly slow down the decomposition process), or false to use the maximal number of C-NOT gates.
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False ):  
        
        Decomposition_Base.__init__( self, Umtx ) 
            
        
        # logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
        self.optimize_layer_num  = optimize_layer_num
        
        # number of iteratrion loops in the finale optimalization
        self.iteration_loops = 3
        
        # The maximal allowed error of the optimalization problem
        self.optimalization_tolerance = 1e-7
        
        # Maximal number of iteartions in the optimalization process
        self.max_iterations = int(1e4)
        
        # number of operators in one sub-layer of the optimalization process
        self.operation_layer = 1     
        
        # logical value indicating whether the quasi-unitarization of the submatrices was done or not 
        self.subdisentaglement_done = False
        
        # The subunitarized matrix
        self.subunitarized_mtx = None
                        
    
    
    
    
##
# @brief Start the optimalization process to disentangle the most significant qubit from the others. The optimized parameters and operations are stored in the attributes @optimized_parameters and @operations.
    def disentangle_submatrices(self):
        
        if self.subdisentaglement_done:
            print('Sub-disentaglement already done.')
            return 
        
        
        print(' ')
        print('Disentagling submatrices.')
        
        # setting the global target minimum
        self.global_target_minimum = 0   
        
        
        # check if it needed to do the subunitarization
        #if self.subdisentaglement_problem( [], eye(size(self.Umtx)), eye(size(self.Umtx)) ) <= self.optimalization_tolerance 
        if self.subdisentaglement_problem( [] ) <= self.optimalization_tolerance :
            print('Disentanglig not needed')
            self.subunitarized_mtx = self.Umtx
            self.subdisentaglement_done = True
            return
        
                  
        
        
        if not self.check_optimalization_solution():
            # Adding the operations of the successive layers
            
            #measure the time for the decompositin        
            start_time = time.time()
            
            # variable for local counting of C-NOT gates
            while self.layer_num < self.max_layer_num  : 
                
                control_qbit = self.qbit_num-1
                
                for target_qbit in range(self.qbit_num-2,-1,-1):
                    
                    # creating block of operations
                    block = operation_block( self.qbit_num )
                    
                    # add CNOT gate to the block
                    block.add_cnot_to_end(control_qbit, target_qbit)       
                    
                    # adding U3 operation to the block
                    block.add_u3_to_end(target_qbit, Theta=True, Lambda=True) 
                    block.add_u3_to_end(control_qbit, Theta=True, Lambda=True) 
                    
                    # adding the opeartion block to the operations
                    self.add_operation_to_end( block )
                    
                
                
                # get the number of blocks
                self.layer_num = len(self.operations)
                                                 
                # Do the optimalization
                if self.optimize_layer_num and ((self.layer_num % self.operation_layer) == 0) or self.layer_num >= self.max_layer_num:
                    # solve the optzimalization problem to find the correct mninimum
                    self.solve_optimalization_problem( optimalization_problem=self.subdisentaglement_problem)   

                    if self.check_optimalization_solution():
                        break
                    
                
            print("--- %s seconds elapsed during the decomposition ---" % (time.time() - start_time))            
                
           

                       
        
        
        if self.check_optimalization_solution():   
            
            print('Sub-disentaglement was succesfull')
            print(' ')
        else:
            print('Sub-disentaglement did not reach the tolerance limit.')
            print(' ')

        
        
        #print('Eliminating parametrized C-NOT gates.')
        #print(' ')
        # eliminate_parametrized_cnot
        #self.eliminate_parametrized_cnot()
        
        # indicate that the unitarization of the sumbatrices was done
        self.subdisentaglement_done = True
        
        # The subunitarized matrix
        self.subunitarized_mtx = self.get_transformed_matrix( self.optimized_parameters, self.operations )
        
       

## subdisentaglement_problem
# @brief The optimalization problem to be solved in order to disentangle the two least significant qubits
# @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
# @return Returns with the value representing the entaglement of the most significant qubit from the others. (gives zero if the most significant qubit is decoupled from the others.)
    def subdisentaglement_problem( self, parameters ):
       
        
        # get the transformed matrix with the operations in the list
        matrix_new = self.get_transformed_matrix( parameters, self.operations, initial_matrix=self.Umtx)
                
        # get the number of 2qubit submatrices
        submatrices_num_row = 2
        submatrices_num = submatrices_num_row*submatrices_num_row
        
        # get the size of the submatrix
        submatrix_size = int(2**(self.qbit_num-1))
        
        # allocating cell array for submatrices
        submatrices = []
        
        # fill up the submatrices
        for idx in range(0,submatrices_num_row):
            for jdx in range(0,submatrices_num_row):
                submatrices.append( matrix_new[ (idx*submatrix_size):((idx+1)*submatrix_size), (jdx*submatrix_size):((jdx+1)*submatrix_size)] )
            
        
        
        # create the cost function
        cost_function = 0   

        for idx in range(0,submatrices_num):
            for jdx in range(0,submatrices_num_row):
                submatrix_prod = np.dot(submatrices[idx],submatrices[jdx].conj().T)
                submatrix_prod= submatrix_prod - np.identity(len(submatrix_prod))*submatrix_prod[0,0]
                cost_function = cost_function + np.sum( np.multiply(submatrix_prod, submatrix_prod.conj() ) )
            
        return cost_function
        
        
       
        
      
    
    
    
    
    

    


