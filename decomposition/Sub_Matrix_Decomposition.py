#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:31:58 2020
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

from .Decomposition_Base import def_layer_num
from .Decomposition_Base import Decomposition_Base
from operations.operation_block import operation_block
import numpy as np
import time


class Sub_Matrix_Decomposition(Decomposition_Base):
    
## 
# @brief Constructor of the class.
# @param Umtx The unitary matrix
# @param optimize_layer_num Optional logical value. If true, then the optimalization tries to determine the lowest number of the layers needed for the decomposition. If False (default), the optimalization is performed for the maximal number of layers.
# @param method Optional string value labeling the optimalization method used in the calculations. Deafult is L-BFGS-B. For details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# @param identical_blocks A dictionary {'n': integer} indicating that how many identical succesive blocks should be used in the disentanglement of the nth qubit from the others
# @param iteration_loops A dictionary {'n': integer} giving the number of optimalization subloops done for each step in the optimalization process during the disentanglement of the n-th qubit from the others. (For general matrices 1 works fine, higher value increase both the convergence tendency, but also the running time.)
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
# @return An instance of the class
    def __init__( self, Umtx, optimize_layer_num=False, max_layer_num=def_layer_num, method='L-BFGS-B',
                  identical_blocks=dict(), initial_guess= 'zeros', iteration_loops=dict() ):
        
        Decomposition_Base.__init__( self, Umtx, method=method, initial_guess=initial_guess )
            
        
        # logical value. Set true if finding the minimum number of operation layers is required (default), or false when the maximal number of CNOT gates is used (ideal for general unitaries).
        self.optimize_layer_num  = optimize_layer_num
        
        # number of iteratrion loops in the finale optimalization
        self.iteration_loops = iteration_loops
        
        # The maximal allowed error of the optimalization problem
        self.optimalization_tolerance = 1e-7
        
        # Maximal number of iteartions in the optimalization process
        self.max_iterations = int(1e8)
        
        # number of operators in one sub-layer of the optimalization process
        self.optimalization_block = 1     
        
        # logical value indicating whether the quasi-unitarization of the submatrices was done or not 
        self.subdisentaglement_done = False
        
        # The subunitarized matrix
        self.subunitarized_mtx = None
                
        # The number of successive identical blocks in one leyer
        self.identical_blocks = identical_blocks

        # The number of gate blocks used in the decomposition
        self.max_layer_num = max_layer_num
    
    
    
    
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
        if self.subdisentaglement_problem() <= self.optimalization_tolerance :
            print('Disentanglig not needed')
            self.subunitarized_mtx = self.Umtx
            self.subdisentaglement_done = True
            return
        
                  
        
        
        if not self.check_optimalization_solution():
            # Adding the operations of the successive layers
            
            #measure the time for the decompositin        
            start_time = time.time()
            
            # variable for local counting of C-NOT gates
            while self.layer_num < self.max_layer_num.get(str(self.qbit_num), 200)  :
                
                control_qbit = self.qbit_num-1
                
                for target_qbit in range(self.qbit_num-2,-1,-1):
                    
                    
                    
                    for idx in range(0,self.identical_blocks.get(str(self.qbit_num),200)):
                        
                        # creating block of operations
                        block = operation_block( self.qbit_num )
                    
                        # add CNOT gate to the block
                        block.add_cnot_to_end(control_qbit, target_qbit)       
                    
                        # adding U3 operation to the block
                        block.add_u3_to_end(target_qbit, ['Theta', 'Lambda']) 
                        block.add_u3_to_end(control_qbit, ['Theta', 'Lambda']) 
                    
                        # adding the opeartion block to the operations
                        self.add_operation_to_end( block )                    
                
                
                # get the number of blocks
                self.layer_num = len(self.operations)
                                                 
                # Do the optimalization
                if self.optimize_layer_num or self.layer_num >= self.max_layer_num.get(str(self.qbit_num), 200):
                    # solve the optzimalization problem to find the correct mninimum
                    self.solve_optimalization_problem( optimalization_problem=self.subdisentaglement_problem, solution_guess=self.optimized_parameters)   

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
    def subdisentaglement_problem( self, parameters=None ):
       
        if parameters is None:
            matrix_new = self.Umtx
        else:
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
                submatrix_prod = self.apply_operation(submatrices[idx],submatrices[jdx].conj().T)
                submatrix_prod= submatrix_prod - np.identity(len(submatrix_prod))*submatrix_prod[0,0]
                cost_function = cost_function + np.sum( self.multiply(submatrix_prod, submatrix_prod.conj() ) )

        return np.real(cost_function)






       
        
      
    
    
    
    
    

    


