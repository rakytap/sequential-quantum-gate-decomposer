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
from squander.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper
from squander.gates.qgd_U3 import qgd_U3 
from squander.gates.qgd_H import qgd_H 
from squander.gates.qgd_X import qgd_X  
from squander.gates.qgd_Y import qgd_Y  
from squander.gates.qgd_Z import qgd_Z  
from squander.gates.qgd_CH import qgd_CH   
from squander.gates.qgd_CNOT import qgd_CNOT  
from squander.gates.qgd_CZ import qgd_CZ  
from squander.gates.qgd_RX import qgd_RX  
from squander.gates.qgd_RY import qgd_RY  
from squander.gates.qgd_RZ import qgd_RZ   
from squander.gates.qgd_SX import qgd_SX  
from squander.gates.qgd_SYC import qgd_SYC   
from squander.gates.qgd_CRY import qgd_CRY 




##
# @brief A QGD Python interface class for the Gates_Block.
class qgd_Circuit(qgd_Circuit_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param qbit_num: the number of qubits spanning the operations
# @return An instance of the class

    def __init__( self, qbit_num ):
        self.qbit_num = qbit_num
        # call the constructor of the wrapper class
        super(qgd_Circuit, self).__init__( qbit_num )


#@brief Call to add a U3 gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), Theta (bool), Phi (bool), Lambda (bool).

    def add_U3( self, target_qbit, Theta=True, Phi=True, Lambda=True):

	# call the C wrapper function
        super(qgd_Circuit, self).add_U3(target_qbit, Theta, Phi, Lambda)


#@brief Call to add a RX gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_RX( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_RX(target_qbit)


#@brief Call to add a RY gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_RY( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_RY(target_qbit)

#@brief Call to add a RZ gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_RZ( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_RZ(target_qbit)

#@brief Call to add a CNOT gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CNOT( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_CNOT(target_qbit, control_qbit)

#@brief Call to add a CZ gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CZ( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_CZ(target_qbit, control_qbit)

#@brief Call to add a CH gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CH( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_CH(target_qbit, control_qbit)

#@brief Call to add a SYC gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_SYC( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_SYC(target_qbit, control_qbit)


#@brief Call to add a Hadamard gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int)

    def add_H( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_H(target_qbit)

#@brief Call to add a X gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int)

    def add_X( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_X(target_qbit)

#@brief Call to add a Y gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_Y( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_Y(target_qbit)

#@brief Call to add a Z gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_Z( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_Z(target_qbit)

#@brief Call to add a SX gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_SX( self, target_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_SX(target_qbit)

#@brief Call to add adaptive gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_adaptive( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_adaptive(target_qbit, control_qbit)
        


#@brief Call to add adaptive gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_Circuit( self, gate):

	# call the C wrapper function
        super(qgd_Circuit, self).add_Circuit(gate) 

#@brief Call to retrieve the matrix of the operation. 
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: parameters_mtx.

    def get_Matrix( self, parameters_mtx):

	# call the C wrapper function
        return super(qgd_Circuit, self).get_Matrix(parameters_mtx)


#@brief Call to get the parameters of the matrices. 
#@param self A pointer pointing to an instance of the class qgd_Circuit.

    def get_Parameter_Num( self):

	# call the C wrapper function
        return super(qgd_Circuit, self).get_Parameter_Num()



#@brief Call to apply the gate operation on the input matrix
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: parameters_mtx, unitary_mtx.

    def apply_to( self, parameters_mtx, unitary_mtx):

	# call the C wrapper function
        super(qgd_Circuit, self).apply_to( parameters_mtx, unitary_mtx )



##
# @brief Call to get the second Rényi entropy
# @param parameters A float64 numpy array
# @param input_state A complex array storing the input state. If None |0> is created.
# @param qubit_list A subset of qubits for which the Rényi entropy should be calculated.
# @Return Returns with the calculated entropy
    def get_Second_Renyi_Entropy(self, parameters=None, input_state=None, qubit_list=None ):

        # validate input parameters

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
            input_state = np.zeros( (matrix_size,1), dtype=np.complex128 )
            input_state[0] = 1

        # evaluate the entropy
        entropy = super(qgd_Circuit, self).get_Second_Renyi_Entropy( parameters, input_state, qubit_list_validated)  


        return entropy



##
# @brief Call to get the number of qubits in the circuit
# @return Returns with the number of qubits
    def get_Qbit_Num(self):
    
        return super(qgd_Circuit, self).get_Qbit_Num()



#@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated.
    def get_Parameter_Start_Index( self, unitary_mtx):

	# call the C wrapper function
        super().get_Parameter_Start_Index()


#@brief Method to get the list of parent gate indices. Then the parent gates can be obtained from the list of gates involved in the circuit.
    def get_Parents( self, gate):

	# call the C wrapper function
        return super().get_Parents( gate )


#@brief Method to get the list of child gate indices. Then the children gates can be obtained from the list of gates involved in the circuit.
    def get_Children( self, gate):

	# call the C wrapper function
        return super().get_Children( gate )
        
    def add_Gate(self,qgd_gate):
        gate_type = qgd_gate
        if isinstance(qgd_gate,qgd_H):
            self.add_H(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_X):
            self.add_X(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_Y):
            self.add_Y(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_Z):
            self.add_Z(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_CH):
            self.add_CH(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,qgd_CZ):
            self.add_CZ(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,qgd_RX):
            self.add_RX(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_RY):
            self.add_RY(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_RZ):
            self.add_RZ(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_SX):
            self.add_SX(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,qgd_U3):
            self.add_U3(qgd_gate.get_Target_Qbit(),qgd_gate.Theta,qgd_gate.Phi,qgd_gate.Lambda)
        elif isinstance(qgd_gate,qgd_CRY):
            self.add_CRY(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,qgd_CNOT):
            self.add_CNOT(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        else:
            raise Exception("Cannot add gate: unimplemented gate type")
