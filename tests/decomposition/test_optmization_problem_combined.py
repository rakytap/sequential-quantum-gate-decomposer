# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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
## \file test_optimization_problem_combined.py
## \brief Simple example python code demonstrating the basic usage of the Python interface of the Quantum Gate Decomposer package

from squander import N_Qubit_Decomposition_adaptive       


#from squander import nn

import numpy as np
import random
import scipy.linalg
import time
from scipy.fft import fft

import time

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


np.set_printoptions(linewidth=200) 


# number of qubits
qbit_num = 8

# cost function variant
cost_function_variant = 0


# matrix size of the unitary
matrix_size = 1 << qbit_num #pow(2, qbit_num )
dim_over_2 = 1 << (qbit_num-1) #pow(2, qbit_num-1)

# the number of basis in the space of 2^n x 2^n Hermitian matrices
num_of_basis = 1 << 2*qbit_num

# number of adaptive levels
levels = 4

# set true to limit calcualtions to real numbers
real=False


##
# @brief Call to construct random parameter, with limited number of non-trivial adaptive layers
# @param num_of_parameters The number of parameters
def create_randomized_parameters( num_of_parameters, real=False ):


    parameters = np.zeros(num_of_parameters)

    # the number of adaptive layers in one level
    num_of_adaptive_layers = int(qbit_num*(qbit_num-1)/2 * levels)
    
    if (real):
        
        for idx in range(qbit_num):
            parameters[idx*3] = np.random.rand(1)*2*np.pi

    else:
        parameters[0:3*qbit_num] = np.random.rand(3*qbit_num)*np.pi
        pass

    
    nontrivial_adaptive_layers = np.zeros( (num_of_adaptive_layers ))
    
    for layer_idx in range(num_of_adaptive_layers) :

        nontrivial_adaptive_layer = random.randint(0,1)
        nontrivial_adaptive_layers[layer_idx] = nontrivial_adaptive_layer

        if (nontrivial_adaptive_layer) :
        
            # set the radom parameters of the chosen adaptive layer
            start_idx = qbit_num*3 + layer_idx*7
            
            if (real):
                parameters[start_idx]   = np.random.rand(1)*2*np.pi
                parameters[start_idx+1] = np.random.rand(1)*2*np.pi
                parameters[start_idx+4] = np.random.rand(1)*2*np.pi
            else:
                end_idx = start_idx + 7
                parameters[start_idx:end_idx] = np.random.rand(7)*2*np.pi
         
        
    
    #print( parameters )
    return parameters, nontrivial_adaptive_layers



class Test_Decomposition:
    """This is a test class of the python iterface to test the trace offset, and the optimized problem"""

    def test_N_Qubit_Decomposition_creation(self):


        ###########################################
        # create the unitary




        # creating a class to decompose the unitary
        cDecompose_createUmtx = N_Qubit_Decomposition_adaptive( np.eye(matrix_size, dtype=np.complex128), level_limit_max=5, level_limit_min=0, accelerator_num=0 )


        # adding decomposing layers to the gat structure
        for idx in range(levels):
            cDecompose_createUmtx.add_Adaptive_Layers()

        cDecompose_createUmtx.add_Finalyzing_Layer_To_Gate_Structure()


        # get the number of free parameters
        num_of_parameters = cDecompose_createUmtx.get_Parameter_Num()


        # create randomized parameters
        parameters, nontrivial_adaptive_layers = create_randomized_parameters( num_of_parameters, real=real )



        Umtx = cDecompose_createUmtx.get_Matrix( parameters )


        # cut the matrixt by trace offset
        trace_offset = 80
        Umtx = Umtx[trace_offset:240, :]


	###################################################################################

	# test cost function with trace offset 



        # creating a class to decompose the unitary
        cDecompose_CPU = N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=0, accelerator_num=0 )

        # set the trace offset
        cDecompose_CPU.set_Trace_Offset( trace_offset )

        # adding decomposing layers to the gat structure
        for idx in range(levels):
            cDecompose_CPU.add_Adaptive_Layers()

        cDecompose_CPU.add_Finalyzing_Layer_To_Gate_Structure()

        # setting the cost function variant
        cDecompose_CPU.set_Cost_Function_Variant(cost_function_variant)

        t0 = time.time()
        f0_CPU, grad_CPU = cDecompose_CPU.Optimization_Problem_Combined( parameters )

        assert( np.abs( f0_CPU ) < 1e-8 )
        
        
        

    def test_grad_batch_unitary_funcs(self):
        # creating a class to decompose the unitary
        cDecompose = N_Qubit_Decomposition_adaptive( np.eye(matrix_size, dtype=np.complex128), level_limit_max=5, level_limit_min=0, accelerator_num=0 )


        # adding decomposing layers to the gat structure
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()

        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()


        # get the number of free parameters
        num_of_parameters = cDecompose.get_Parameter_Num()


        # create randomized parameters
        parameters, nontrivial_adaptive_layers = create_randomized_parameters( num_of_parameters, real=real )



        Umtx = cDecompose.get_Matrix( parameters )
        mat, mat_deriv = cDecompose.Optimization_Problem_Combined_Unitary(parameters)
        assert np.allclose(Umtx, mat)
        
        cost = cDecompose.Optimization_Problem(parameters)
        assert np.allclose(np.array([cost, cost, cost]), cDecompose.Optimization_Problem_Batch(np.vstack([parameters, parameters, parameters])))
        grad = cDecompose.Optimization_Problem_Grad(parameters)        
        f0_CPU, grad_CPU = cDecompose.Optimization_Problem_Combined( parameters )
        assert np.allclose(grad, grad_CPU)
        assert np.isclose(f0_CPU, cost)






