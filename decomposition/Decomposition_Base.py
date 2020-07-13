# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:57:35 2020
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

import numpy as np
from numpy import linalg as LA
from operations.Operations import Operations
from operations.Operation import Operation
from operations.U3 import U3
from operations.operation_block import operation_block
from scipy.optimize import minimize
from optimparallel import minimize_parallel
import time
from functools import reduce
import multiprocessing as mp


# default number of layers in the decomposition as a function of number of qubits
def_layer_num = { '2': 3, '3':20, '4':60, '5':100 }



##
# @brief A class containing basic methods for the decomposition process.

class Decomposition_Base( Operations ):
 
    

## Contructor of the class
# @brief Constructor of the class.
# @param Umtx The unitary matrix to be decomposed
# @param parallel Optional logical value. I true, parallelized optimalization id used in the decomposition. The parallelized optimalization is efficient if the number of blocks optimized in one shot (given by attribute @optimalization_block) is at least 10). For False (default) sequential optimalization is applied
# @param method Optional string value labeling the optimalization method used in the calculations. Deafult is L-BFGS-B. For details see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# @param initial_guess String indicating the method to guess initial values for the optimalization. Possible values: 'zeros' (deafult),'random', 'close_to_zero'
# @return An instance of the class
    def __init__( self, Umtx, parallel= False, method='L-BFGS-B', initial_guess= 'zeros' ):
        
        # determine the number of qubits
        qbit_num = int(round( np.log2( len(Umtx) ) ))
        Operations.__init__( self, qbit_num )
        
        # the unitary operator to be decomposed
        self.Umtx = Umtx   

        # the optimalization method used in the calculations (default is L-BFGS-B)
        self.method = method
        
        # logical value to use paralllelized optimalization
        self.parallel = parallel
        
        # The corrent optimized parameters for the operations
        self.optimized_parameters = np.empty(0)
        
        # logical value describing whether the decomposition was finalized or not
        self.decomposition_finalized = False        
        
        # error of the unitarity of the final decomposition
        self.decomposition_error = None
        
        # number of finalizing (deterministic) opertaions counted from the top of the array of operations
        self.finalizing_operations_num = 0
        
        # the number of the finalizing (deterministic) parameters counted from the top of the optimized_parameters list
        self.finalizing_parameters_num = 0
        
        # The current minimum of the optimalization problem
        self.current_minimum = None                       
        
        # The global minimum of the optimalization problem
        self.global_target_minimum = None
        
        # logical value describing whether the optimalization problem was solved or not
        self.optimalization_problem_solved = False
        
        # number of iteratrion loops in the finale optimalization
        self.iteration_loops = None
        
        # The maximal allowed error of the optimalization problem
        self.optimalization_tolerance = None
        
        # Maximal number of iteartions in the optimalization process
        self.max_iterations = int(1e4)
    
        # number of operators in one sub-layer of the optimalization process
        self.optimalization_block = None
        
        # The maximal number of C-NOT gates allowed in the decomposition
        self.max_layer_num = None
        
        # method to guess initial values for the optimalization. POssible values: 'zeros', 'random', 'close_to_zero'
        self.initial_guess = initial_guess
    
        
        
     
##   
# @brief Call to set the number of operation layers to optimize in one shot
# @param optimalization_block The number of operation blocks to optimize in one shot 
    def set_optimalization_blocks( self, optimalization_block):
        self.optimalization_block = optimalization_block
        
##   
# @brief Call to set the maximal number of the iterations in the optimalization process
# @param max_iterations aximal number of iteartions in the optimalization process
    def set_max_iteration( self, max_iterations):
        self.max_iterations = max_iterations

##   
# @brief Call to set whether to use parallel or sequential calculations in the optimalization
# @param parallel Logical value to use paralllelized optimalization
    def set_parallel( self, parallel):
        self.parallel = parallel       
    
    
## 
# @brief After the main optimalization problem is solved, the indepent qubits can be rotated into state |0> by this def. The constructed operations are added to the array of operations needed to the decomposition of the input unitary.
    def finalize_decomposition(self):       
            
        # get the transformed matrix resulted by the operations in the list
        matrix_new = self.get_transformed_matrix(self.optimized_parameters, self.operations ) 
          
        # obtaining the final operations of the decomposition
        finalizing_operations, finalizing_parameters, matrix_new = self.get_finalizing_operations( matrix_new )
            
        # adding the finalizing operations to the list of operations
        # adding the opeartion block to the operations
        self.add_operation_to_front( finalizing_operations )
        self.optimized_parameters = np.concatenate((finalizing_parameters, self.optimized_parameters))
        self.parameter_num = len(self.optimized_parameters)
        self.finalizing_operations_num = len(finalizing_operations.operations)
        self.finalizing_parameters_num = len(finalizing_parameters)
        
        # indicat that the decomposition was finalized    
        self.decomposition_finalized = True
            
        # calculating the final error of the decomposition
        self.decomposition_error = LA.norm(matrix_new*np.exp(np.complex(0,-np.angle(matrix_new[0,0]))) - np.identity(len(matrix_new))*abs(matrix_new[0,0]), 2)
            
        # get the number of gates used in the decomposition
        gates_num = self.get_gate_nums()
        print( 'The error of the decomposition after finalyzing operations is ' + str(self.decomposition_error) + ' with ' + str(self.layer_num) + ' layers containing '  + str(gates_num.get('u3',0)) + ' U3 operations and ' + str(gates_num.get('cnot',0)) + ' CNOT gates.' )
        print(' ')
            

##
# @brief Lists the operations decomposing the initial unitary. (These operations are the inverse operations of the operations bringing the intial matrix into unity.)
# @param start_index The index of the first inverse operation
    def list_operations( self, start_index = 1 ):
       
        Operations.list_operations(self, self.optimized_parameters, start_index = start_index )
       

                
##
# @brief This method determine the operations needed to rotate the indepent qubits into the state |0>
# @param mtx The unitary describing indepent qubits.
# @return [1] The operations needed to rotate the qubits into the state |0>
# @return [2] The parameters of the U3 operations needed to rotate the qubits into the state |0>
# @return [3] The resulted diagonalized matrix.
    def get_finalizing_operations( self, mtx ):
        
        # creating block of operations to store the finalization operations
        finalizing_operations = operation_block( self.qbit_num )
                                    
        finalizing_parameters = np.empty(0)     
               
        mtx_new = mtx
        
        for target_qbit in range(0,self.qbit_num):
                        
            # contructing the list of reordered qubits
            qbits_reordered = list( range(self.qbit_num-1,-1,-1) )
            del qbits_reordered[len(qbits_reordered)-target_qbit-1]
            qbits_reordered = qbits_reordered + [target_qbit]
            
            # contruct the permutation to get the basis for the reordered qbit list
            bases_reorder_indexes = self.get_basis_of_reordered_qubits( qbits_reordered )
            
            # construct the 2x2 submatrix for the given qubit to be rorarted to axis z
            # first reorder the matrix elements to get the submatrix representing the given qubit into the corner
            matrix_reordered = mtx[:, bases_reorder_indexes][bases_reorder_indexes]
            submatrix = matrix_reordered[0:2, 0:2]
            
            
            # finalize the 2x2 submatrix with z-y-z rotation
            cos_theta_2 = abs(submatrix[0,0])/np.sqrt(abs(submatrix[0,0])**2 + abs(submatrix[0,1])**2)
            Theta = 2*np.arccos( cos_theta_2 )
            
            if abs(submatrix[0,0]) < 1e-7:
                Phi = np.angle( submatrix[1,0] )
                Lambda = np.angle( -submatrix[0,1] )
            elif abs(submatrix[1,0]) < 1e-7:
                Phi = 0
                Lambda = np.angle( submatrix[1,1]*np.conj(submatrix[0,0]))
            else:            
                Phi = np.angle( submatrix[1,0]*np.conj(submatrix[0,0]))
                Lambda = np.angle( -submatrix[0,1]*np.conj(submatrix[0,0]))
                
            parameters_loc = np.array([Theta, np.pi-Lambda, np.pi-Phi])
            u3_loc = U3( self.qbit_num, target_qbit, ['Theta', 'Phi', 'Lambda'])           
            
            # adding the new operation to the list of finalizing operations
            finalizing_parameters = np.concatenate((parameters_loc, finalizing_parameters))
            finalizing_operations.add_operation_to_front( u3_loc )
            
            
            # get the new matrix
            mtx_new = self.apply_operation( u3_loc.matrix(parameters_loc), mtx_new )
            
        return finalizing_operations,finalizing_parameters, mtx_new
            
        
                
    
        
    
    

    
## solve_optimalization_problem
# @brief This method can be used to solve the main optimalization problem which is devidid into sub-layer optimalization processes. (The aim of the optimalization problem is to disentangle one or more qubits) The optimalized parameters are stored in attribute @optimized_parameters.
# @param varargin Cell array of optional parameters:
# @param 'optimalization_problem' def handle of the cost def to be optimalized
# @param 'solution_guess' Array of guessed parameters
    def solve_optimalization_problem(self, optimalization_problem = None, solution_guess=None): 
        
        if len(self.operations) == 0:
            return
        
        if optimalization_problem is None:
            optimalization_problem = self.optimalization_problem
            
        
        
        
        # array containing minimums to check convergence of the solution
        minimum_vec = [0]*40
               
        # store the operations
        operations = self.operations
        
        # store the number of parameters
        parameter_num = self.parameter_num  
        
        # storing the initial computational parameters
        optimalization_block = self.optimalization_block
        parallel = self.parallel
        
        # store the optimized parameters
        if solution_guess is None:
            if self.initial_guess=='zeros':
                optimized_parameters = np.zeros(self.parameter_num)
            elif self.initial_guess=='random':
                optimized_parameters = (2*np.random.rand(self.parameter_num)-1)*2*np.pi
            elif self.initial_guess=='close_to_zero':
                optimized_parameters = (2*np.random.rand(self.parameter_num)-1)*2*np.pi/100
            else:
                raise BaseException('bad value for initial guess')
        else:     
            if self.initial_guess=='zeros':
                optimized_parameters = np.concatenate(( np.zeros(self.parameter_num-len(solution_guess)), solution_guess ))
            elif self.initial_guess=='random':
                optimized_parameters = np.concatenate(( (2*np.random.rand(self.parameter_num-len(solution_guess))-1)*2*np.pi, solution_guess ))
            elif self.initial_guess=='close_to_zero':
                optimized_parameters = np.concatenate(( (2*np.random.rand(self.parameter_num-len(solution_guess))-1)*2*np.pi/100, solution_guess ))
            else:
                raise BaseException('bad value for initial guess')

        # starting number of operation block applied prior to the optimalized operation blocks
        pre_operation_parameter_num = 0

        # starting index of the block group to be optimalized
        block_idx_start = len(operations)

        # Determine the starting indexes of the optimalization iterations if the initial guess of the new parameters is set to zero

        if self.initial_guess=='zeros':
            if not (solution_guess is None) and len(solution_guess) < parameter_num:
                while pre_operation_parameter_num < len(solution_guess):
                    pre_operation_parameter_num = pre_operation_parameter_num + len(operations[block_idx_start-1].parameters)
                    block_idx_start = block_idx_start - 1

        
        #measure the time for the decompositin        
        start_time = time.time()
        
        for iter_idx in range(0,self.max_iterations+1):
            
            # determine the index of the current block under optimalization
            #block_idx = int((iter_idx % (len(operations)/self.optimalization_block) ) + 1)
            
            #determine the range of blocks to be optimalized togedther
            block_idx_end = block_idx_start - self.optimalization_block
            if block_idx_end < 0 :
                block_idx_end = 0
                
            ## determine the number of free parameters to be optimized
            #block_parameter_num = 0
            #for jdx in range(1,self.optimalization_block+1):
            #    block_parameter_num = block_parameter_num + len( operations[-(jdx+block_idx-2)-1].parameters )
            
            # determine the number of free parameters to be optimized
            block_parameter_num = 0
            for block_idx in range(block_idx_start-1,block_idx_end-1,-1):
                block_parameter_num = block_parameter_num + len( operations[block_idx].parameters )
                    
            
            
            #get the fixed operations
            # matrix of the fixed operations aplied befor the operations to be varied
            fixed_parameters_pre = optimized_parameters[ len(optimized_parameters)-pre_operation_parameter_num+1-1 : len(optimized_parameters)]
            #fixed_operations_pre = operations[ len(operations)-(block_idx-1)*self.optimalization_block+1-1 : len(operations) ]
            fixed_operations_pre = operations[ block_idx_start : len(operations) ]
            operations_mtx_pre = self.get_transformed_matrix( fixed_parameters_pre, fixed_operations_pre, initial_matrix = np.identity(len(self.Umtx)) )
                        
            fixed_operation_pre = Operation( self.qbit_num )
            fixed_operation_pre.matrix = operations_mtx_pre
            
            # matrix of the fixed operations aplied after the operations to be varied
            fixed_parameters_post = optimized_parameters[ 0:len(optimized_parameters)-pre_operation_parameter_num-block_parameter_num ]
            #fixed_operations_post = operations[ 0:len(operations)-block_idx*self.optimalization_block ]
            fixed_operations_post = operations[ 0:block_idx_end ]
            operations_mtx_post = self.get_transformed_matrix( fixed_parameters_post, fixed_operations_post, initial_matrix = np.identity(len(self.Umtx)) )
            
            
            fixed_operation_post = Operation( self.qbit_num )
            fixed_operation_post.matrix = operations_mtx_post
                        
            # operations in the optimalization process
            #self.operations = [fixed_operation_post] + operations[len(operations)-block_idx*self.optimalization_block:len(operations)-(block_idx-1)*self.optimalization_block] + [fixed_operation_pre]
            self.operations = [fixed_operation_post] + operations[block_idx_end:block_idx_start] + [fixed_operation_pre]
            
            
            
            # solve the optimalization problem of the block            
            solution_guess = optimized_parameters[ len(optimized_parameters)-pre_operation_parameter_num-block_parameter_num : len(optimized_parameters)-pre_operation_parameter_num]
            self.solve_layer_optimalization_problem( optimalization_problem=optimalization_problem, solution_guess=solution_guess)
            
                        
            # add the current minimum to the array of minimums
            minimum_vec = minimum_vec[1:] + [self.current_minimum]
            
            # store the obtained optimalized parameters for the block
            #optimized_parameters( -block_idx*self.optimalization_block*block_parameter_num+1:-(block_idx-1)*self.optimalization_block*block_parameter_num ) = self.optimized_parameters;
            optimized_parameters[ len(optimized_parameters)-pre_operation_parameter_num-block_parameter_num+1-1 : len(optimized_parameters)-pre_operation_parameter_num ] = self.optimized_parameters
            
            # update the index of paramateres corresponding to the operations applied before the operations to be optimalized in the iteration cycle
            #if block_idx == len(operations)/self.optimalization_block:
            #    pre_operation_parameter_num = 0
            #else:
            #    pre_operation_parameter_num = pre_operation_parameter_num + block_parameter_num
            
            if block_idx_end == 0:
                block_idx_start = len(operations)
                pre_operation_parameter_num = 0
            else:
                block_idx_start = block_idx_start - self.optimalization_block
                pre_operation_parameter_num = pre_operation_parameter_num + block_parameter_num
                
            
            # optimalization result is displayed in each 10th iteration
            if iter_idx % 5 == 0 and self.parallel:
                print('The minimum with ' + str(self.layer_num) + ' layers after ' + str(iter_idx) + ' iterations is ' + str(self.current_minimum))
                print("--- %s seconds elapsed during the decomposition cycle ---" % (time.time() - start_time))            
                start_time = time.time()
            elif iter_idx % 50 == 0 and not self.parallel:
                print('The minimum with ' + str(self.layer_num) + ' layers after ' + str(iter_idx) + ' iterations is ' + str(self.current_minimum))
            
            # conditions to break the iteration cycles
            if np.std(minimum_vec[30:40])/minimum_vec[39] < self.optimalization_tolerance or \
                np.std(minimum_vec[20:40])/minimum_vec[39] < self.optimalization_tolerance*1e2 or \
                np.std(minimum_vec[10:40])/minimum_vec[39] < self.optimalization_tolerance*1e3 or \
                np.std(minimum_vec[0:40])/minimum_vec[39] < self.optimalization_tolerance*1e4:

                print('The iterations converged to minimum ' + str(self.current_minimum) + ' after ' + str(iter_idx) + ' iterations with ' + str(self.layer_num) + ' layers '  )
                print(' ')
                break
            elif self.check_optimalization_solution():
                print('The minimum with ' + str(self.layer_num) + ' layers after ' + str(iter_idx) + ' iterations is ' + str(self.current_minimum))
                print(' ')
                break
            
            # the convergence at low minimums is much faster if only one layer is considered in the optimalization at once
            if self.current_minimum < 1e-0:
                self.optimalization_block = 1
                self.parallel = False
            
        
        
        if iter_idx == self.max_iterations:
            print('Reached maximal number of iterations')
            print(' ')
        
        # restoring the parameters to originals
        self.optimalization_block = optimalization_block
        self.parallel = parallel
        
        # store the obtained optimalizated parameters
        self.operations = operations
        self.optimized_parameters = optimized_parameters
        self.parameter_num = parameter_num
        
        

    
##
# @brief This method can be used to solve a single sub-layer optimalization problem. The optimalized parameters are stored in attribute @optimized_parameters.
# @param 'optimalization_problem' def handle of the cost def to be optimalized
# @param 'solution_guess' Array of guessed parameters
    def solve_layer_optimalization_problem(self, optimalization_problem = None, solution_guess = None): 
                
        if len(self.operations) == 0:
            return
        
        if optimalization_problem is None:
            optimalization_problem = self.optimalization_problem
            
        if solution_guess is None:
            solution_guess = np.zeros(self.parameter_num)
        
        
        self.current_minimum = None
        optimized_parameters = None
        for idx  in range(1,self.iteration_loops+1):
            
            if self.parallel:
                res = minimize_parallel(optimalization_problem, solution_guess, options={'disp': False})
            else:
                res = minimize(optimalization_problem, solution_guess, method=self.method, options={'disp': False})    
            ##res = minimize(optimalization_problem, solution_guess, method='nelder-mead', options={'xatol': 1e-7, 'disp': False})
            #res = minimize(optimalization_problem, solution_guess, method='BFGS', options={'disp': False})
            #res = minimize(optimalization_problem, solution_guess, method='L-BFGS-B', options={'disp': False})
            ##res = minimize(optimalization_problem, solution_guess, method='CG', options={'disp': False})
            solution = res.x
            minimum = res.fun        
                        
            if self.current_minimum is None or self.current_minimum > minimum:
                self.current_minimum = minimum
                optimized_parameters = solution
                solution_guess = solution + (2*np.random.rand(len(solution))-1)*np.pi/10  
            else:
                solution_guess = solution_guess + (2*np.random.rand(len(solution))-1)*np.pi 
              
                      
        

        
        # storing the solution of the optimalization problem
        self.optimized_parameters = optimized_parameters
        
             
                
        
    
    
    
    
## optimalization_problem
# @brief This is an abstact def giving the cost def measuring the entaglement of the qubits. When the qubits are indepent, teh cost def should be zero.
# @param parameters An array of the free parameters to be optimized. (The number of teh free paramaters should be equal to the number of parameters in one sub-layer)
    def optimalization_problem( self, parameters ):       
        return None
        
        
       
    
    
    
## check_optimalization_solution
# @brief Checks the convergence of the optimalization problem.
# @return Returns with true if the target global minimum was reached during the optimalization process, or false otherwise.
    def check_optimalization_solution(self):
        
        if (not (self.current_minimum is None )):
            return (abs(self.current_minimum - self.global_target_minimum) < self.optimalization_tolerance)
        
        return False
        
            
        
## get_transformed_matrix
# @brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
# @param parameters An array containing the parameters of the U3 operations.
# @param operations The array of the operations to be applied on a unitary
# @param initial_matrix The initial matrix wich is transformed by the given operations. (by deafult it is set to the attribute @Umtx)
# @return Returns with the transformed matrix.
    def get_transformed_matrix(self, parameters, operations, initial_matrix = None ):
                
        
        parameter_idx = 0
        
        if initial_matrix is None:
            initial_matrix = self.Umtx
        
        # construct the list of matrix representation of the gates
        operation_mtxs = list()

        for idx in range(0,len(operations)):
            
            operation = operations[idx]
            
            if operation.type == 'cnot':
                operation_mtx = operation.matrix
                
            elif operation.type == 'u3':
                
                if len(operation.parameters) == 1:
                    operation_mtx = operation.matrix( parameters[parameter_idx] )
                    parameter_idx = parameter_idx + 1
                    
                elif len(operation.parameters) == 2:
                    operation_mtx = operation.matrix( parameters[parameter_idx:parameter_idx+2] )
                    parameter_idx = parameter_idx +- 2
                    
                elif len(operation.parameters) == 3:
                    operation_mtx = operation.matrix( parameters[parameter_idx:parameter_idx+3] )
                    parameter_idx = parameter_idx + 3
                else:
                    print('The U3 operation has wrong number of parameters')
                
                                
            elif operation.type == 'block':
                parameters_num = len(operation.parameters)
                operation_mtx = operation.matrix( parameters[parameter_idx:parameter_idx+parameters_num] )
                parameter_idx = parameter_idx + parameters_num
                
            elif operation.type == 'general':
                operation_mtx = operation.matrix

            operation_mtxs.append(operation_mtx)

        operation_mtxs.append(initial_matrix)
        return self.apply_operations( operation_mtxs, mp.cpu_count() )
    
    
    
##
# @brief Calculate the transformed matrix resulting by an array of operations on a given initial matrix.
# @return Returns with the decomposed matrix.
    def get_decomposed_matrix(self):
        
        return self.get_transformed_matrix( self.optimized_parameters, self.operations )
        
            
    
##
# @brief Gives an array of permutation indexes that can be used to permute the basis in the N-qubit unitary according to the flip in the qubit order.
# @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
# @retrun Returns with the reordering indexes of the basis     
    def get_basis_of_reordered_qubits(self, qbit_list):
        
        bases_reorder_indexes  = list()
        
        # generate the reordered  basis set
        for idx in range(0,2**self.qbit_num):
            reordered_state = bin(idx)
            reordered_state = reordered_state[2:].zfill(self.qbit_num)
            reordered_state = [int(i) for i in reordered_state ]
            #reordered_state.reverse()
            bases_reorder_indexes.append(int(np.dot( [2**power for power in qbit_list], reordered_state)))
        
        #print(qbit_list)
        #print(bases_reorder_indexes)
        return bases_reorder_indexes
        
     
##
# @brief Call to reorder the qubits in the unitary to be decomposed (the qubits become reordeerd in the operations a well)        
# @param qbit_list A list of the permutation of the qubits (for example [1 3 0 2])
    def reorder_qubits(self, qbit_list):
        
        # contruct the permutation to get the basis for the reordered qbit list
        bases_reorder_indexes = self.get_basis_of_reordered_qubits( qbit_list )
            
        # reordering the matrix elements
        self.Umtx = self.Umtx[:, bases_reorder_indexes][bases_reorder_indexes]
        
        # reordering the matrix eleemnts of the operations
        Operations.reorder_qubits( self, qbit_list)
    
       
##
# @brief Call to contruct Qiskit compatible quantum circuit from the operations
    def get_quantum_circuit_inverse(self, circuit=None):
        return Operations.get_quantum_circuit_inverse( self, self.optimized_parameters, circuit=circuit)
    
##
# @brief Call to contruct Qiskit compatible quantum circuit from the operations that brings the original unitary into identity
    def get_quantum_circuit(self, circuit=None):    
        return Operations.get_quantum_circuit( self, self.optimized_parameters, circuit=circuit)


##
# @brief Apply an operations on the input matrix
# @param operation_mtx The matrix of the operation.
# @param input_matrix The input matrix to be transformed.
# @return Returns with the transformed matrix
    @staticmethod
    def apply_operation( operation_mtx, input_matrix ):

        # Getting the transformed state upon the transformation given by operation
        return np.dot(operation_mtx, input_matrix)


##
# @brief Apply a list of operations on the input matrix
# @param operation_mtxs The list containing matrix representations of the gates.
# @param input_matrix The input matrix to be transformed.
# @return Returns with the transformed matrix
    def apply_operations(self, operation_mtxs, numCPUs, connection=None):


        # Getting the transformed state upon the transformation given by operation
        #return reduce(np.dot, operation_mtxs)



        if numCPUs == 1 or len(operation_mtxs) <= 100:
            returnVal = reduce(np.dot, operation_mtxs)
            if connection != None:
                connection.send(returnVal)
            return returnVal

        parent1, child1 = mp.Pipe()
        parent2, child2 = mp.Pipe()
        p1 = mp.Process(target=self.apply_operations, args=(operation_mtxs[:len(operation_mtxs) // 2], numCPUs // 2, child1,))
        p2 = mp.Process(target=self.apply_operations, args=(operation_mtxs[len(operation_mtxs) // 2:], numCPUs // 2 + numCPUs % 2, child2,))
        p1.start()
        p2.start()
        leftReturn, rightReturn = parent1.recv(), parent2.recv()
        p1.join()
        p2.join()
        returnVal = np.dot(leftReturn, rightReturn)
        if connection != None:
            connection.send(returnVal)

        return returnVal

