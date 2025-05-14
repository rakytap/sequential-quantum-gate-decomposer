## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

## \file qgd_N_Qubit_Decomposition_Tree_Search.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.


import numpy as np
from os import path
from squander.decomposition.qgd_N_Qubit_Decomposition_Tree_Search_Wrapper import qgd_N_Qubit_Decomposition_Tree_Search_Wrapper
from squander.gates.qgd_Circuit import qgd_Circuit



##
# @brief A QGD Python interface class for the decomposition of N-qubit unitaries into U3 and CNOT gates.
class qgd_N_Qubit_Decomposition_Tree_Search(qgd_N_Qubit_Decomposition_Tree_Search_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param Umtx_in The unitary matrix to be decomposed
# @param qbit_num_in The number of qubits spanning the unitary Umtx
# @param level_limit_max The maximal number of two-qubit gates in the decomposition
# @param topology_in A list of <target_qubit, control_qubit> pairs describing the connectivity between qubits.
# @param config std::map conatining custom config parameters
# @param accelerator_num The number of DFE accelerators used in the calculations
# @return An instance of the class
    def __init__( self, Umtx, level_limit_max=8, topology=None, config={}, accelerator_num=0 ):

        ## the number of qubits
        self.qbit_num = int(round( np.log2( len(Umtx) ) ))

        # validate input parameters

        topology_validated = list()
        if isinstance(topology, list) or isinstance(topology, tuple):
            for item in topology:
                if isinstance(item, tuple) and len(item) == 2:
                    item_validated = (np.intc(item[0]), np.intc(item[1]))
                    topology_validated.append(item_validated)
                else:
                    print("Elements of topology should be two-component tuples (int, int)")
                    return
        elif topology == None:
            pass
        else:
            print("Input parameter topology should be a list of (int, int) describing the connected qubits in the topology")
            return
        

        # config
        if not( type(config) is dict):
            print("Input parameter config should be a dictionary describing the following hyperparameters:") #TODO
            return

        # call the constructor of the wrapper class
        super().__init__(Umtx, self.qbit_num, level_limit_max, topology=topology_validated, config=config, accelerator_num=accelerator_num)

        
##
# @brief Wrapper function to call the start_decomposition method of C++ class
    def Start_Decomposition(self):

	# call the C wrapper function
        super().Start_Decomposition()
        


    
##
# @brief Call to reorder the qubits in the matrix of the gate
# @param qbit_list The reordered list of qubits spanning the matrix
    def Reorder_Qubits( self, qbit_list ):

	# call the C wrapper function
        super().Reorder_Qubits(qbit_list)


##
# @brief @brief Call to print the gates decomposing the initial unitary. These gates brings the intial matrix into unity.
    def List_Gates(self):

	# call the C wrapper function
        super().List_Gates()

##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Qiskit_Circuit( self ):

        from squander import Qiskit_IO
        
        squander_circuit = self.get_Circuit()
        parameters       = self.get_Optimized_Parameters()
        
        return Qiskit_IO.get_Qiskit_Circuit( squander_circuit, parameters )




 
##
# @brief Call to set unitary matrix from a numpy array
# @param Umtx_arr numpy complex array 
    def set_Unitary( self, Umtx_arr ):  

        return super().set_Unitary( Umtx_arr )


##
# @brief Call to set the error tolerance of the decomposition
# @param tolerance Error tolerance given as a floating point number
    def set_Optimization_Tolerance( self, tolerance ):  

        return super().set_Optimization_Tolerance( tolerance )
        
##
# @brief Call to get unitary matrix
    def get_Unitary( self ):

        return super().get_Unitary()
        
##
# @brief Call to export unitary matrix to binary file
# @param filename string
    def export_Unitary( self, filename ):

        return super().export_Unitary(filename)


##
# @brief Call to get the number of free parameters in the gate structure used for the decomposition
    def get_Parameter_Num( self ):

        return super().get_Parameter_Num()

##
# @brief Call to get global phase
    def get_Global_Phase( self ):
	
        return super().get_Global_Phase()

##
# @brief Call to set global phase
# @param new_global_phase New global phase (in radians)
    def set_Global_Phase( self, new_global_phase ):
	
        return super().set_Global_Phase(new_global_phase)
##
# @brief Call to get the name of the SQUANDER project
    def get_Project_Name( self ):
	
        return super().get_Project_Name()

##
# @brief Call to set the name of the SQUANDER project
# @param project_name_new new project name
    def set_Project_Name( self, project_name_new ):
	
        return super().set_Project_Name(project_name_new)
##
# @brief Call to apply global phase on Unitary matrix
    def apply_Global_Phase_Factor( self ):
	
        return super().apply_Global_Phase_Factor()


## 
# @brief Call to set the optimizer used in the gate synthesis process
# @param optimizer String indicating the optimizer. Possible values: "BFGS" ,"ADAM", "BFGS2", "ADAM_BATCHED", "AGENTS", "COSINE", "AGENTS_COMBINED".
    def set_Optimizer( self, optimizer="BFGS" ):

        # Set the optimizer
        super().set_Optimizer(optimizer)  


##
# @brief Call to retrieve the unitary of the circuit
# @param parameters A float64 numpy array
    def get_Matrix( self, parameters = None ):

  
        if parameters is None:
            print( "get_Matrix: arary of input parameters is None")
            return None

        return super().get_Matrix( parameters )
        
## 
# @brief Call to set the optimizer used in the gate synthesis process
# @param costfnc Variant of the cost function. Input argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2
    def set_Cost_Function_Variant( self, costfnc=0 ):

        # Set the optimizer
        super().set_Cost_Function_Variant(costfnc=costfnc)  



## 
# @brief Call to set the trace offset used in the cost function. In this case Tr(A) = sum_(i-offset=j) A_{ij}
# @param trace_offset The trace offset to be set
    def set_Trace_Offset( self, trace_offset=0 ):

        # Set the trace offset
        super().set_Trace_Offset(trace_offset=trace_offset)  


## 
# @brief Call to get the trace offset used in the cost function. In this case Tr(A) = sum_(i-offset=j) A_{ij}
# @return Returns with the trace offset
    def get_Trace_Offset( self ):

        # Set the optimizer
        return super().get_Trace_Offset()  


## 
# @brief Call to get the optimized parameters set in numpy array
# @return Returns with the optimized parameters
    def get_Optimized_Parameters(self):
    
        return super().get_Optimized_Parameters()


## 
# @brief Call to set the parameters which are used as a starting point in the optimization
# @param A numpy array containing the parameters. The number of parameters can be  retrieved with method get_Parameter_Num
    def set_Optimized_Parameters(self, new_params):
        
        super().set_Optimized_Parameters(new_params)


## 
# @brief Call to evaluate the cost function.
# @param parameters A float64 numpy array. The number of parameters can be  retrieved with method get_Parameter_Num 
    def Optimization_Problem( self, parameters=None ):

        if parameters is None:
            print( "Optimization_Problem: array of input parameters is None")
            return None

        # evaluate the cost function and gradients
        cost_function = super().Optimization_Problem(parameters)  


        return cost_function


## 
# @brief Call to evaluate the gradient components.
# @param parameters A float64 numpy array
    def Optimization_Problem_Grad( self, parameters=None ):

        if parameters is None:
            print( "Optimization_Problem: array of input parameters is None")
            return None

        # evaluate the cost function and gradients
        grad = super().Optimization_Problem_Grad(parameters)  

        grad = grad.reshape( (-1,))

        return grad


## 
# @brief Call to evaluate the cost function and the gradient components.
# @param parameters A float64 numpy array
    def Optimization_Problem_Combined( self, parameters=None ):

        if parameters is None:
            print( "Optimization_Problem_Combined: array of input parameters is None")
            return None

        # evaluate the cost function and gradients
        cost_function, grad = super().Optimization_Problem_Combined(parameters)  

        grad = grad.reshape( (-1,))

        return cost_function, grad

##
# @brief Call to get the number of iterations  
    def get_Num_of_Iters(self):
    
        return super().get_Num_of_Iters()
    
##
# @brief Call to set the maximum number of iterations for each optimization loop
# @param max_iters int number of maximum iterations each loop
    def set_Max_Iterations(self, max_iters):
        
        super().set_Max_Iterations(max_iters)
    
##
# @brief call to set the cost function type of the optimization problem
# @param cost_func int argument 0 stands for FROBENIUS_NORM, 1 for FROBENIUS_NORM_CORRECTION1, 2 for FROBENIUS_NORM_CORRECTION2, 3 for HILBERT_SCHMIDT_TEST, 4 for HILBERT_SCHMIDT_TEST_CORRECTION1, 5 for HILBERT_SCHMIDT_TEST_CORRECTION2 see more at: https://arxiv.org/abs/2210.09191
    def set_Cost_Function_Variant(self, cost_func):
    
        super().set_Cost_Function_Variant(cost_func)


##
# @brief Call to get the error of the decomposition. (i.e. the final value of the cost function)
# @return Returns with the error of the decmposition
    def get_Decomposition_Error(self):
    
        return super().get_Decomposition_Error()



##
# @brief Call to get the second Rényi entropy
# @param parameters A float64 numpy array
# @param input_state A complex array storing the input state. If None |0> is created.
# @param qubit_list A subset of qubits for which the Rényi entropy should be calculated.
# @Return Returns with the calculated entropy
    def get_Second_Renyi_Entropy(self, parameters=None, input_state=None, qubit_list=None ):

        qbit_num = self.get_Qbit_Num()

        qubit_list_validated = list()
        if isinstance(qubit_list, list) or isinstance(qubit_list, tuple):
            for item in qubit_list:
                if isinstance(item, int):
                    qubit_list_validated.append(item)
                    qubit_list_validated = list(set(qubit_list_validated))
                else:
                    print("Elements of qbit_list should be integers")
                    return
        elif qubit_list == None:
            qubit_list_validated = [ x for x in range(qbit_num) ]

        else:
            print("Elements of qbit_list should be integers")
            return
        

        if parameters is None:
            print( "get_Second_Renyi_entropy: array of input parameters is None")
            return None


        if input_state is None:
            matrix_size = 1 << qbit_num
            input_state = np.zeros( (matrix_size,1) )
            input_state[0] = 1

        # evaluate the entropy
        entropy = super().get_Second_Renyi_Entropy( parameters, input_state, qubit_list_validated)  


        return entropy



##
# @brief Call to get the number of qubits in the circuit
# @return Returns with the number of qubits
    def get_Qbit_Num(self):
    
        return super().get_Qbit_Num()
