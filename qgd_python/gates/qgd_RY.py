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

## \file qgd_N_Qubit_Decomposition.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.


import numpy as np
from os import path
from qgd_python.gate.qgd_RY_Wrapper import qgd_RY_Wrapper



##
# @brief A QGD Python interface class for the Gates_Block.
class qgd_RY(qgd_RY_Wrapper):
    
    
## 
# @brief Constructor of the class.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param args A tuple of the input arguments: qbit_num (integer)
#qbit_num: the number of qubits spanning the operations
#@param kwds A tuple of keywords
# @return An instance of the class

    def __init__( self ):

        # initiate variables for input arguments
        #int  qbit_num = -1; 
        # call the constructor of the wrapper class
        super(qgd_python.gate.qgd_RY_Wrapper, self).__init__()

#@brief Call to add a U3 gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), Theta (bool), Phi (bool), Lambda (bool).

    def get_Matrix( self, parameters_mtx ):

	# call the C wrapper function
        super(qgd_python.gate.qgd_RY_Wrapper, self).get_Matrix( parameters_mtx )

#@brief Call to get the parameters of the matrices. 
#@param self A pointer pointing to an instance of the class qgd_Circuit.

    def get_Gate_Kernel( self, ThetaOver2, Phi, Lambda):

	# call the C wrapper function
        super(qgd_python.gate.qgd_RY_Wrapper, self).calc_one_qubit_u3(ThetaOver2, Phi, Lambda)

#@brief Call to apply the gate operation on the input matrix
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: parameters_mtx, unitary_mtx.

    def apply_to( self, parameters_mtx, unitary_mtx):

	# call the C wrapper function
        super(qgd_python.gate.qgd_RY_Wrapper self).apply_to( parameters_mtx, unitary_mtx  )
