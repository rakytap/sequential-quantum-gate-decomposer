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

## \file qgd_Circuit.py
##    \brief A Python interface class representing  Squander circuit.


import numpy as np
from os import path
from squander.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper


from squander.gates.gates_Wrapper import (
    U1,
    U2,
    U3,
    H,
    X,
    Y,
    Z,
    T,
    Tdg,
    CH,
    CNOT,
    CZ,
    R,
    RX,
    RY,
    RZ,
    SX,
    SYC,
    CRY,
    CR,
    CROT )


##
# @brief A QGD Python interface class for the Gates_Block.
class qgd_Circuit(qgd_Circuit_Wrapper):
    
    
## 
# @brief Constructor of the class.
# @param qbit_num: the number of qubits spanning the operations
# @return An instance of the class

    def __init__( self, qbit_num ):

        # call the constructor of the wrapper class
        super().__init__( qbit_num )

    """
    def __getstate__(self):
        # Return a dictionary of the object's state

        return super().__getstate__()

    def __setstate__(self, state):

        print( state )
        super().__setstate__( state )
    

    def __new__(cls, *args, **kwargs):

        print( "NEEEEEEEEEEEEEW", args)
        return super().__new__(cls, *args, **kwargs)
    """

    def __copy__(self):
        cCircuit = qgd_Circuit(self.get_Qbit_Num())
        for gate in self.get_Gates():
            cCircuit.add_Gate(gate)
        return cCircuit
#@brief Call to add a U1 gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input argument: target_qbit (int)

    def add_U1( self, target_qbit):

	# call the C wrapper function
        super().add_U1(target_qbit)

#@brief Call to add a U2 gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input argument: target_qbit (int)

    def add_U2( self, target_qbit):

	# call the C wrapper function
        super().add_U2(target_qbit)

#@brief Call to add a U3 gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input argument: target_qbit (int)

    def add_U3( self, target_qbit):

	# call the C wrapper function
        super().add_U3(target_qbit)

#@brief Call to add a RX gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_RX( self, target_qbit):

	# call the C wrapper function
        super().add_RX(target_qbit)

#@brief Call to add a R gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).
    def add_R( self, target_qbit):

	# call the C wrapper function
        super().add_R(target_qbit)

#@brief Call to add a RY gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_RY( self, target_qbit):

	# call the C wrapper function
        super().add_RY(target_qbit)

#@brief Call to add a RZ gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_RZ( self, target_qbit):

	# call the C wrapper function
        super().add_RZ(target_qbit)

#@brief Call to add a CNOT gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CNOT( self, target_qbit, control_qbit):

	# call the C wrapper function
        super().add_CNOT(target_qbit, control_qbit)

#@brief Call to add a CZ gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CZ( self, target_qbit, control_qbit):

	# call the C wrapper function
        super().add_CZ(target_qbit, control_qbit)

#@brief Call to add a CH gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CH( self, target_qbit, control_qbit):

	# call the C wrapper function
        super().add_CH(target_qbit, control_qbit)

#@brief Call to add a SYC gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_SYC( self, target_qbit, control_qbit):

	# call the C wrapper function
        super().add_SYC(target_qbit, control_qbit)


#@brief Call to add a Hadamard gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int)

    def add_H( self, target_qbit):

	# call the C wrapper function
        super().add_H(target_qbit)

#@brief Call to add a X gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int)

    def add_X( self, target_qbit):

	# call the C wrapper function
        super().add_X(target_qbit)

#@brief Call to add a Y gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_Y( self, target_qbit):

	# call the C wrapper function
        super().add_Y(target_qbit)

#@brief Call to add a Z gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_Z( self, target_qbit):

	# call the C wrapper function
        super().add_Z(target_qbit)

#@brief Call to add a SX gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_SX( self, target_qbit):

	# call the C wrapper function
        super().add_SX(target_qbit)

#@brief Call to add a T gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_T( self, target_qbit):

	# call the C wrapper function
        super().add_T(target_qbit)

#@brief Call to add a T gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_Tdg( self, target_qbit):

	# call the C wrapper function
        super().add_Tdg(target_qbit)

#@brief Call to add adaptive gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_adaptive( self, target_qbit, control_qbit):

	# call the C wrapper function
        super().add_adaptive(target_qbit, control_qbit)
        
#@brief Call to add a CROT gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int).

    def add_CROT( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_CROT(target_qbit, control_qbit)

#@brief Call to add a CR gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_CR( self, target_qbit, control_qbit):

	# call the C wrapper function
        super(qgd_Circuit, self).add_CR(target_qbit, control_qbit)

#@brief Call to add adaptive gate to the front of the gate structure.
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: target_qbit (int), control_qbit (int).

    def add_Circuit( self, gate):

	# call the C wrapper function
        super().add_Circuit(gate) 

#@brief Call to retrieve the matrix of the operation. 
#@param self A pointer pointing to an instance of the class qgd_Circuit.
#@param Input arguments: parameters_mtx.

    def get_Matrix( self, parameters_mtx):

	# call the C wrapper function
        return super().get_Matrix(parameters_mtx)


#@brief Call to get the parameters of the matrices. 
#@param self A pointer pointing to an instance of the class qgd_Circuit.

    def get_Parameter_Num( self):

	# call the C wrapper function
        return super().get_Parameter_Num()



#@brief Call to apply the gate operation on the input matrix
#@param Input arguments: parameters_mtx, unitary_mtx.
    def apply_to( self, parameters_mtx, unitary_mtx, parallel=1):

	# call the C wrapper function
        super().apply_to( parameters_mtx, unitary_mtx, parallel=parallel )



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
        entropy = super().get_Second_Renyi_Entropy( parameters, input_state, qubit_list_validated)  


        return entropy



##
# @brief Call to get the number of qubits in the circuit
# @return Returns with the number of qubits
    def get_Qbit_Num(self):
    
        return super().get_Qbit_Num()


##
# @brief Call to get the list of qubits involved in the circuit
# @return Returns with the list of qubits
    def get_Qbits(self):
    
        return super().get_Qbits()
        
        
##
# @brief Call to get the list of gates (or subcircuits) in the circuit
# @return Returns with the list of gates
    def get_Gates(self):
    
        return super().get_Gates()  
        
        
##
# @brief Call to get statisctics on the gate counts in the circuit.
# @return Returns with the dictionary containing the gate counts
    def get_Gate_Nums(self):
    
        return super().get_Gate_Nums()                


##
# @brief Call to remap the qubits in the circuit. 
# @param qbit_map The map conatining the qbit map in a form of dict: {int(initial_qbit): int(remapped_qbit)}. 
# @param qbit_num The number of qbits in the remaped circuits (Can be different than in the original circuit) 
# @return Returns with the newly created, remapped circuit
    def Remap_Qbits(self, qbit_map, qbit_num=None):

        if qbit_num == None:
            qbit_num = self.get_Qbit_Num()


        return super().Remap_Qbits( qbit_map, qbit_num)
    



#@brief Call to get the starting index of the parameters in the parameter array corresponding to the circuit in which the current gate is incorporated.
    def get_Parameter_Start_Index( self ):

	# call the C wrapper function
        return super().get_Parameter_Start_Index()


#@brief Method to get the list of parent gate indices. Then the parent gates can be obtained from the list of gates involved in the circuit.
    def get_Parents( self, gate):

	# call the C wrapper function
        return super().get_Parents( gate )


#@brief Method to get the list of child gate indices. Then the children gates can be obtained from the list of gates involved in the circuit.
    def get_Children( self, gate):

	# call the C wrapper function
        return super().get_Children( gate )


#@brief TODO: Jakab
    def get_Circuit_Depth(self):
        used_gates_idx = []
        gates = self.get_Gates()
        depth = 0
        gate_start_idx = 0
        gate_groups = []
        while (len(used_gates_idx) != len(gates)):
            involved_qbits=[]
            gate_start_idx_prev = gate_start_idx
            gate_idx = gate_start_idx
            depth += 1
            control_last_single=[False]*self.qbit_num
            gate_group=[]
            while((len(involved_qbits)<self.qbit_num) and gate_idx<len(gates)):
                if gate_idx not in used_gates_idx:
                    target_qbit = gates[gate_idx].get_Target_Qbit()
                    control_qbit = gates[gate_idx].get_Control_Qbit()
                    gate = gates[gate_idx]
                    if isinstance( gate, qgd_CROT ):
                        if ((control_qbit in involved_qbits) and control_last_single[control_qbit]==False) and (target_qbit not in involved_qbits):
                            involved_qbits.append(target_qbit)
                            used_gates_idx.append(gate_idx)
                            gate_group.append(gate_idx)
                            control_last_single[control_qbit]=False
                        elif ((control_qbit in involved_qbits) and control_last_single[control_qbit]==True) and (target_qbit not in involved_qbits):
                            involved_qbits.append(target_qbit)
                            if (gate_start_idx == gate_start_idx_prev):
                                gate_start_idx=gate_idx
                        elif (control_qbit not in involved_qbits) and (target_qbit not in involved_qbits):
                            involved_qbits.append(target_qbit)
                            involved_qbits.append(control_qbit)
                            used_gates_idx.append(gate_idx)
                            gate_group.append(gate_idx)
                            control_last_single[control_qbit]=False
                        elif (gate_start_idx == gate_start_idx_prev):
                            gate_start_idx=gate_idx
                    else:
                        if control_qbit!=-1:
                            if (control_qbit not in involved_qbits) and (target_qbit not in involved_qbits):
                                involved_qbits.append(target_qbit)
                                involved_qbits.append(control_qbit)
                                used_gates_idx.append(gate_idx)
                                gate_group.append(gate_idx)
                            elif (gate_start_idx == gate_start_idx_prev):
                                gate_start_idx=gate_idx
                        else:
                            control_last_single[target_qbit]=True
                            if target_qbit not in involved_qbits:
                                involved_qbits.append(target_qbit)
                                used_gates_idx.append(gate_idx)
                                control_last_single[target_qbit]=True
                                gate_group.append(gate_idx)
                            elif (gate_start_idx == gate_start_idx_prev):
                                gate_start_idx=gate_idx
                gate_idx+=1
            gate_groups.append(gate_group)
        return depth
        

#@brief Add a generic gate to the circuit
    def add_Gate(self,qgd_gate):


        if isinstance(qgd_gate,H):
            self.add_H(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,X):
            self.add_X(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,Y):
            self.add_Y(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,Z):
            self.add_Z(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,CH):
            self.add_CH(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,CZ):
            self.add_CZ(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,RX):
            self.add_RX(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,RY):
            self.add_RY(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,RZ):
            self.add_RZ(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,SX):
            self.add_SX(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,U1):
            self.add_U1(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,U2):
            self.add_U2(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,U3):
            self.add_U3(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,CRY):
            self.add_CRY(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,CNOT):
            self.add_CNOT(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,T):
            self.add_T(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,Tdg):
            self.add_Tdg(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,R):
            self.add_R(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate,CROT):
            self.add_CROT(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate,CR):
            self.add_CR(qgd_gate.get_Target_Qbit(),qgd_gate.get_Control_Qbit())
        else:
            raise Exception("Cannot add gate: unimplemented gate type")
