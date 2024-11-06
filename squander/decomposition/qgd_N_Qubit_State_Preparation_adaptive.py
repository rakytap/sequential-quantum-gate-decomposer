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
from squander.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive

##
# @brief A QGD Python interface class for the decomposition of N-qubit state into U3 and CNOT gates.
class qgd_N_Qubit_State_Preparation_adaptive(qgd_N_Qubit_Decomposition_adaptive):

    def __init__( self, State, level_limit_max=8, level_limit_min=0, topology=None, config={} ):

        # check input quantum state

        if type(State) != np.ndarray:
            raise Exception("Initial state should be a numpy array")


        if State.dtype != np.complex128:
            raise Exception("Initial state should be made of complex values")


        if not State.data.c_contiguous :
            raise Exception("Initial state should be contiguous in memory")

        if len(State.shape) == 1:
            State = State.reshape( (State.size, 1,) )


        if len(State.shape) == 2 and State.shape[1]==1:
            super().__init__( State, level_limit_max, level_limit_min, topology=topology, config=config )
        else:
            raise Exception("Initial state not properly formatted. Input state must be a column vector")



##
# @brief Export the unitary decomposition into Qiskit format.
# @return Return with a Qiskit compatible quantum circuit.
    def get_Qiskit_Circuit( self ):
    
        from squander import Qiskit_IO
        
        squander_circuit = self.get_Circuit()
        parameters       = self.get_Optimized_Parameters()
        
        return Qiskit_IO.get_Qiskit_Circuit_inverse( squander_circuit, parameters )
        
        
              
