## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
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

## \file qgd_N_Qubit_Decomposition.py
##    \brief A QGD Python interface class for the decomposition of N-qubit unitaries into a set of two-qubit and one-qubit gates.

import numpy as np
from os import path
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive

##
# @brief A QGD Python interface class for the decomposition of N-qubit state into U3 and CNOT gates.
class qgd_N_Qubit_State_Preparation_adaptive(qgd_N_Qubit_Decomposition_adaptive):

	def __init__( self, State, level_limit_max=8, level_limit_min=0, topology=None, config={} ):
		print(State.shape)
		if ( (type(State) == np.ndarray) and (State.shape[1]==1) ):
			super().__init__( State, level_limit_max, level_limit_min, topology=topology, config=config )
		else:
			raise Exception("Initial state not properly formatted. Input state must be a column vector")
			
	def get_Quantum_Circuit( self ):

		from qiskit import QuantumCircuit

        # creating Qiskit quantum circuit
		circuit = QuantumCircuit(self.qbit_num)

        # retrive the list of decomposing gate structure
		gates = self.get_Gates()

        # constructing quantum circuit
		for idx in range(len(gates)):

			gate = gates[idx]

			if gate.get("type") == "CNOT":
                # adding CNOT gate to the quantum circuit
				circuit.cx(gate.get("control_qbit"), gate.get("target_qbit"))

			elif gate.get("type") == "CZ":
                # adding CZ gate to the quantum circuit
				circuit.cz(gate.get("control_qbit"), gate.get("target_qbit"))

			elif gate.get("type") == "CH":
                # adding CZ gate to the quantum circuit
				circuit.ch(gate.get("control_qbit"), gate.get("target_qbit"))

			elif gate.get("type") == "SYC":
                # Sycamore gate
				print("Unsupported gate in the circuit export: Sycamore gate")
				return None;

			elif gate.get("type") == "U3":
                # adding U3 gate to the quantum circuit
				circuit.u(-gate.get("Theta"), -gate.get("Lambda"), -gate.get("Phi"), gate.get("target_qbit"))

			elif gate.get("type") == "RX":
                # RX gate
				circuit.rx(-gate.get("Theta"), gate.get("target_qbit"))

			elif gate.get("type") == "RY":
                # RY gate
				circuit.ry(-gate.get("Theta"), gate.get("target_qbit"))

			elif gate.get("type") == "RZ":
                # RZ gate
				circuit.rz(-gate.get("Phi"), gate.get("target_qbit"))

			elif gate.get("type") == "X":
                # X gate
				circuit.x(gate.get("target_qbit"))

			elif gate.get("type") == "SX":
                # SX gate
				circuit.sx(gate.get("target_qbit"))

		return circuit

