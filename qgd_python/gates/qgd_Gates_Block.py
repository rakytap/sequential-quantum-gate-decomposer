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
from qgd_python.gate.qgd_Gates_Block_Wrapper import qgd_Gates_Block_Wrapper



##
# @brief A QGD Python interface class for the Gates_Block.
class qgd_Gates_Block(qgd_Gates_Block_Wrapper):
    
    
## 
# @brief Constructor of the class.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param args A tuple of the input arguments: qbit_num (integer)
#qbit_num: the number of qubits spanning the operations
#@param kwds A tuple of keywords
# @return An instance of the class

    def __init__( self ):

        # initiate variables for input arguments
        #int  qbit_num = -1; 
        # call the constructor of the wrapper class
        super(qgd_python.gate.qgd_Gates_Block, self).__init__()


#@brief Call to add a U3 gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), Theta (bool), Phi (bool), Lambda (bool).

    def add_U3( self, target_qbit, Theta, Phi, Lambda):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_U3(target_qbit, Theta, Phi, Lambda)


#@brief Call to add a RX gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int).

    def add_RX( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_RX(target_qbit)


#@brief Call to add a RY gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int).

    def add_RY( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_RY(target_qbit)

#@brief Call to add a RZ gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int).

    def add_RZ( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_RZ(target_qbit)

#@brief Call to add a CNOT gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CNOT( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_CNOT(target_qbit, control_qbit)

#@brief Call to add a CZ gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CZ( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_CZ(target_qbit, control_qbit)

#@brief Call to add a CH gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CH( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_CH(target_qbit, control_qbit)

#@brief Call to add a SYC gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_SYC( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_SYC(target_qbit, control_qbit)

#@brief Call to add a X gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int)

    def add_X( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_X(target_qbit)

#@brief Call to add a Y gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int).

    def add_Y( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_Y(target_qbit)

#@brief Call to add a Z gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int).

    def add_Z( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_Z(target_qbit)

#@brief Call to add a SX gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int).

    def add_SX( self, target_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_SX(target_qbit)

#@brief Call to add adaptive gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_adaptive( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).add_adaptive(target_qbit, control_qbit)

#@brief Call to add adaptive gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_Gates_Block( self, gate):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block_Wrapper, self).add_adaptive(gate) 

#@brief Call to retrieve the matrix of the operation. 
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: parameters_mtx.

    def get_Matrix( self, parameters_mtx):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).get_Matrix(parameters_mtx)


#@brief Call to get the parameters of the matrices. 
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.

    def get_Parameter_Num( self):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).get_Parameter_Num()

#@brief Call to apply the gate operation on the input matrix
#@param self A pointer pointing to an instance of the class qgd_Gates_Block.
#@param Input arguments: parameters_mtx, unitary_mtx.

    def apply_to( self, parameters_mtx, unitary_mtx):

	# call the C wrapper function
        super(qgd_python.gate.qgd_Gates_Block, self).apply_to( parameters_mtx, unitary_mtx )

