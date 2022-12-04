# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
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
## \file test_decomposition.py
## \brief Functionality test cases for the qgd_N_Qubit_Decomposition class.



# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np
import random

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


##
# @brief Call to construct random parameter, with limited number of non-trivial adaptive layers
# @param num_of_parameters The number of parameters
def create_randomized_parameters( num_of_parameters, qbit_num, levels ):


    parameters = np.zeros(num_of_parameters)


    # the number of adaptive layers in one level
    num_of_adaptive_layers = int(qbit_num*(qbit_num-1)/2 * levels)

    parameters[0:qbit_num*3] = (2*np.random.rand(qbit_num*3)-1)*2*np.pi
    #parameters[2*qbit_num:3*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
    #parameters[3*qbit_num-1] = 0
    #parameters[3*qbit_num-2] = 0
    
    nontrivial_adaptive_layers = np.zeros( (num_of_adaptive_layers ))

    for layer_idx in range(num_of_adaptive_layers) :

        nontrivial_adaptive_layer = random.randint(0,1)
        nontrivial_adaptive_layers[layer_idx] = nontrivial_adaptive_layer

        if (nontrivial_adaptive_layer) :
        
            # set the radom parameters of the chosen adaptive layer
            start_idx = qbit_num*3 + layer_idx*7

            end_idx = start_idx + 7
            parameters[start_idx:end_idx] = np.random.rand(7)*2*np.pi
        
        
    

    return parameters, nontrivial_adaptive_layers


class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""

    def test_get_nn_chanels(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of class N_Qubit_Decomposition.

        """

        from qgd_python.nn.qgd_nn import qgd_nn
        from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive      

        # number of qubits
        qbit_num = 3

        # matrix size of the unitary
        matrix_size = pow(2, qbit_num )

        # number of adaptive levels
        levels = 0

        # creating a class to decompose the unitary
        cDecompose = qgd_N_Qubit_Decomposition_adaptive( np.eye(matrix_size), level_limit_max=5, level_limit_min=0 )
        
        # adding decomposing layers to the gat structure
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()        

        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()


        # get the number of free parameters
        num_of_parameters = cDecompose.get_Parameter_Num()

        # create randomized parameters having number of nontrivial adaptive blocks determined by the parameter nontrivial_ratio
        parameters, nontrivial_adaptive_layers = create_randomized_parameters( num_of_parameters, qbit_num, levels )

        # getting the unitary corresponding to quantum circuit
        unitary = cDecompose.get_Matrix( parameters )
        
        print(parameters)

        # retrieve the chanels
        qgd_nn().get_NN_Chanels( unitary )
        
