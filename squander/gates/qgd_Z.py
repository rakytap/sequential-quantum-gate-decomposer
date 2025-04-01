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
from .qgd_Z_Wrapper import qgd_Z_Wrapper



##
# @brief A QGD Python interface class for the qgd_Z.
class qgd_Z(qgd_Z_Wrapper):
    
    
## 
# @brief Constructor of the class.
#@param self A pointer pointing to an instance of the class qgd_Z.
#@param args A tuple of the input arguments: qbit_num (integer)
#qbit_num: the number of qubits spanning the operations
#@param kwds A tuple of keywords
# @return An instance of the class

    def __init__( self, qbit_num, target_qbit ):

        # call the constructor of the wrapper class
        super().__init__(qbit_num, target_qbit)

#@brief  Call to retrieve the gate matrix
#@param self A pointer pointing to an instance of the class qgd_Z. 

    def get_Matrix( self ):

	# call the C wrapper function
        return super().get_Matrix( )

#@brief Call to get the parameters of the matrices. 
#@param self A pointer pointing to an instance of the class qgd_Z.

    def get_Gate_Kernel( self):

	# call the C wrapper function
        return super().calc_one_qubit_u3()

#@brief Call to apply the gate operation on the input matrix
#@param self A pointer pointing to an instance of the class qgd_Z.
#@param Input arguments: parameters_mtx, unitary_mtx.

    def apply_to( self, unitary_mtx):

	# call the C wrapper function
        super().apply_to( unitary_mtx )


#@brief Call to get the number of free parameters in the gate.
    def get_Parameter_Num( self):

	# call the C wrapper function
        return super().get_Parameter_Num()


#@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated.
    def get_Parameter_Start_Index( self):

	# call the C wrapper function
        return super().get_Parameter_Start_Index()


#@brief Call to get the target qbit.
    def get_Target_Qbit( self ):

	# call the C wrapper function
        return super().get_Target_Qbit()


#@brief Call to get the control qbit (returns with -1 if no control qbit is used in the gate).
    def get_Control_Qbit( self ):

	# call the C wrapper function
        return super().get_Control_Qbit()

#@brief Call to set the target qbit.
    def set_Target_Qbit( self, target_qbit_in ):

	# call the C wrapper function
        super().set_Target_Qbit(target_qbit_in)


#@brief Call to set the control qbit (does nothing if no control qbit is used in the gate).
    def set_Control_Qbit( self, control_qbit_in ):

	# call the C wrapper function
        return
#@brief Call to extract the paramaters corresponding to the gate, from a parameter array associated to the circuit in which the gate is embedded.
    def Extract_Parameters( self, parameters_circuit ):

	# call the C wrapper function
        parameters_gate = super().Extract_Parameters( parameters_circuit )

        parameters_gate = np.reshape( parameters_gate, (parameters_gate.size,) )

        return parameters_gate
