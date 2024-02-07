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

    parameters[0:qbit_num*3] = np.random.rand(qbit_num*3)*2*np.pi
    #parameters[2*qbit_num:3*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
    #parameters[qbit_num:2*qbit_num] = np.random.rand(qbit_num)*2*np.pi/4
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

        from squander.nn.qgd_nn import qgd_nn
        from squander.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive      

        # number of qubits
        qbit_num = 3

        # matrix size of the unitary
        matrix_size = pow(2, qbit_num )

        # number of adaptive levels
        levels = 1

        # retrieve the chanels
        nn_class = qgd_nn()
        #nn_class.get_NN_Chanels( unitary )
        #nn_class.get_NN_Chanels( qbit_num=qbit_num, levels=levels )        
        chanels, nontrivial_adaptive_layers = nn_class.get_NN_Chanels( qbit_num=qbit_num, levels=levels, samples_num=4 )   

        #print( chanels.shape )
        #print( chanels )
        #print( nontrivial_adaptive_layers )

        
