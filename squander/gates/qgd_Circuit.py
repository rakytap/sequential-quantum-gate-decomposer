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
See the License for the specific language governing permissions andP
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""

"""A Python interface class representing Squander circuit."""


import numpy as np
from squander.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper


from squander.gates.gates_Wrapper import (
    U1,
    U2,
    U3,
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
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
    CRZ,
    CRX,
    CP,
    CR,
    CROT,
    CCX,
    CSWAP,
    SWAP,
    RXX
)


class qgd_Circuit(qgd_Circuit_Wrapper):
    """A QGD Python interface class for the Gates_Block."""

    def __init__(self, qbit_num):
        """Constructor of the class.

        Args:
            qbit_num: The number of qubits spanning the operations
        """

        # call the constructor of the wrapper class
        super().__init__(qbit_num)

    def add_U1(self, target_qbit):
        """Add a U1 gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_U1(target_qbit)

    def add_U2(self, target_qbit):
        """Add a U2 gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_U2(target_qbit)

    def add_U3(self, target_qbit):
        """Add a U3 gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_U3(target_qbit)

    def add_RX(self, target_qbit):
        """Add a RX gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_RX(target_qbit)

    def add_R(self, target_qbit):
        """Add a R gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_R(target_qbit)

    def add_RY(self, target_qbit):
        """Add a RY gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_RY(target_qbit)

    def add_RZ(self, target_qbit):
        """Add a RZ gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_RZ(target_qbit)

    def add_CNOT(self, target_qbit, control_qbit):
        """Add a CNOT gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super().add_CNOT(target_qbit, control_qbit)

    def add_CZ(self, target_qbit, control_qbit):
        """Add a CZ gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super().add_CZ(target_qbit, control_qbit)

    def add_CH(self, target_qbit, control_qbit):
        """Add a CH gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super().add_CH(target_qbit, control_qbit)

    def add_CU(self, target_qbit, control_qbit):
        """Add a CU gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super().add_CU(target_qbit, control_qbit)

    def add_SYC(self, target_qbit, control_qbit):
        """Add a SYC gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super().add_SYC(target_qbit, control_qbit)

    def add_H(self, target_qbit):
        """Add a Hadamard gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_H(target_qbit)

    def add_X(self, target_qbit):
        """Add a X gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_X(target_qbit)

    def add_Y(self, target_qbit):
        """Add a Y gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper functionerror
        super().add_Y(target_qbit)

    def add_Z(self, target_qbit):
        """Add a Z gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_Z(target_qbit)

    def add_SX(self, target_qbit):
        """Add a SX gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_SX(target_qbit)

    def add_S(self, target_qbit):
        """Add a S gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_S(target_qbit)

    def add_Sdg(self, target_qbit):
        """Add a Sdg gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_Sdg(target_qbit)

    def add_T(self, target_qbit):
        """Add a T gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_T(target_qbit)

    def add_Tdg(self, target_qbit):
        """Add a Tdg gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
        """

        # call the C wrapper function
        super().add_Tdg(target_qbit)

    def add_adaptive(self, target_qbit, control_qbit):
        """Add an adaptive gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super().add_adaptive(target_qbit, control_qbit)

    def add_CROT(self, target_qbit, control_qbit):
        """Add a CROT gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super(qgd_Circuit, self).add_CROT(target_qbit, control_qbit)

    def add_CR(self, target_qbit, control_qbit):
        """Add a CR gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super(qgd_Circuit, self).add_CR(target_qbit, control_qbit)

    def add_CRY(self, target_qbit, control_qbit):
        """Add a CRY gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super(qgd_Circuit, self).add_CRY(target_qbit, control_qbit)

    def add_CRZ(self, target_qbit, control_qbit):
        """Add a CRZ gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super(qgd_Circuit, self).add_CRZ(target_qbit, control_qbit)

    def add_CRX(self, target_qbit, control_qbit):
        """Add a CRX gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super(qgd_Circuit, self).add_CRX(target_qbit, control_qbit)

    def add_CP(self, target_qbit, control_qbit):
        """Add a CP gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbit: Control qubit index (int)
        """

        # call the C wrapper function
        super(qgd_Circuit, self).add_CP(target_qbit, control_qbit)

    def add_SWAP(self, target_qbits, target_qbit2=-1):
        """Add a SWAP gate to the front of the gate structure.

        Args:
            target_qbits: List of target qubits (list of int) - at least 2 qubits required
            target_qbit2: Optional second qubit if target_qbits is a single int
        """
        # Ensure target_qbits is a list
        if isinstance(target_qbits, (list, tuple)):
            super(qgd_Circuit, self).add_SWAP(list(target_qbits))
        if isinstance(target_qbits, int) and target_qbit2 != -1:
            super(qgd_Circuit, self).add_SWAP(list([target_qbits, target_qbit2]))
    
    def add_RXX(self, target_qbits, target_qbit2=-1):
        """Add a SWAP gate to the front of the gate structure.

        Args:
            target_qbits: List of target qubits (list of int) - at least 2 qubits required
            target_qbit2: Optional second qubit if target_qbits is a single int
        """
        # Ensure target_qbits is a list
        if isinstance(target_qbits, (list, tuple)):
            super(qgd_Circuit, self).add_RXX(list(target_qbits))
        if isinstance(target_qbits, int) and target_qbit2 != -1:
            super(qgd_Circuit, self).add_RXX(list([target_qbits, target_qbit2]))

    def add_CSWAP(self, target_qbits, control_qbits):
        """Add a CSWAP (Fredkin) gate to the front of the gate structure.

        Args:
            target_qbits: Target qubits (list of int) - exactly 2 for standard CSWAP
            control_qbits: Control qubit(s) (int or list of int) - exactly 1 for standard CSWAP

        Note:
            Accepts both list and single integer inputs for control_qbits. Examples:
            - add_CSWAP([0,1], 2) -> control_qbits becomes [2] (standard Fredkin gate)
            - add_CSWAP([0,1], [2]) -> control_qbits stays [2] (standard Fredkin gate)
            Currently only supports exactly 1 control qubit (standard Fredkin gate).
        """
        # Convert target_qbits to list if needed
        if not isinstance(target_qbits, (list, tuple)):
            raise TypeError("target_qbits must be a list or tuple")
        target_qbits = list(target_qbits)

        # Convert control_qbits to list if it's a single integer
        if isinstance(control_qbits, int):
            control_qbits = [control_qbits]
        elif isinstance(control_qbits, (list, tuple)):
            control_qbits = list(control_qbits)
        else:
            raise TypeError("control_qbits must be an int, list, or tuple")

        # call the C wrapper function
        super(qgd_Circuit, self).add_CSWAP(target_qbits, control_qbits)

    def add_CCX(self, target_qbit, control_qbits):
        """Add a CCX gate to the front of the gate structure.

        Args:
            target_qbit: Target qubit index (int)
            control_qbits: Control qubits (list of int or tuple) - at least 2 control qubits required

        Note:
            control_qbits can be a list or tuple. Example:
            - add_CCX(0, [1,2]) -> standard CCX with 2 controls
        """
        # Convert control_qbits to list if needed
        if isinstance(control_qbits, int):
            raise TypeError(
                "control_qbits must be a list or tuple (CCX requires at least 2 control qubits)"
            )
        elif isinstance(control_qbits, (list, tuple)):
            control_qbits = list(control_qbits)
        else:
            raise TypeError("control_qbits must be a list or tuple")

        # call the C wrapper function
        super(qgd_Circuit, self).add_CCX(target_qbit, control_qbits)

    def add_Circuit(self, gate):
        """Add a block of operations (subcircuit) to the front of the gate structure.

        Args:
            gate: A qgd_Circuit instance representing the subcircuit to add
        """

        # call the C wrapper function
        super().add_Circuit(gate)

    def get_Matrix(self, parameters_mtx):
        """Retrieve the matrix representation of the circuit operation.

        Args:
            parameters_mtx: Parameter array (numpy array) for parametric gates

        Returns:
            numpy.ndarray: The matrix representation of the circuit
        """

        # call the C wrapper function
        return super().get_Matrix(parameters_mtx)

    def get_Parameter_Num(self):
        """Get the number of free parameters in the gate structure.

        Returns:
            int: The number of free parameters
        """

        # call the C wrapper function
        return super().get_Parameter_Num()

    def apply_to(self, parameters_mtx, unitary_mtx, parallel=1):
        """Apply the gate circuit operation on the input matrix.

        Args:
            parameters_mtx: Parameter array (numpy array) for parametric gates
            unitary_mtx: Input matrix (numpy array) to be transformed
            parallel: Parallel execution mode (int, optional, default=1)
        """

        # call the C wrapper function
        super().apply_to(parameters_mtx, unitary_mtx, parallel=parallel)

    def get_Second_Renyi_Entropy(
        self, parameters=None, input_state=None, qubit_list=None
    ):
        """Calculate the second Rényi entropy of the quantum circuit.

        Args:
            parameters: Parameter array (float64 numpy array, optional)
            input_state: Input quantum state (complex numpy array, optional). If None, |0> is created
            qubit_list: Subset of qubits for which the Rényi entropy should be calculated (list, optional)

        Returns:
            float: The calculated second Rényi entropy
        """

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
            qubit_list_validated = [x for x in range(qbit_num)]

        else:
            print("Elements of qbit_list should be integers")
            return

        if parameters is None:
            print("get_Second_Renyi_entropy: array of input parameters is None")
            return None

        if input_state is None:
            matrix_size = 1 << qbit_num
            input_state = np.zeros((matrix_size, 1), dtype=np.complex128)
            input_state[0] = 1

        # evaluate the entropy
        entropy = super().get_Second_Renyi_Entropy(
            parameters, input_state, qubit_list_validated
        )

        return entropy

    def get_Qbit_Num(self):
        """Get the number of qubits in the circuit.

        Returns:
            int: The number of qubits
        """

        return super().get_Qbit_Num()

    def get_Qbits(self):
        """Get the list of qubits involved in the circuit.

        Returns:
            list: List of qubit indices involved in the circuit
        """

        return super().get_Qbits()

    def set_min_fusion(self, min_fusion):
        """Set the minimum fusion parameter in the circuit.

        Args:
            min_fusion: Minimum fusion value (int)
        """

        super().set_min_fusion(min_fusion)

    def get_Gates(self):
        """Get the list of gates (or subcircuits) in the circuit.

        Returns:
            list: List of gate objects in the circuit
        """

        return super().get_Gates()

    def get_Gate_Nums(self):
        """Get statistics on the gate counts in the circuit.

        Returns:
            dict: Dictionary containing the gate type counts
        """

        return super().get_Gate_Nums()

    def Remap_Qbits(self, qbit_map, qbit_num=None):
        """Remap the qubits in the circuit.

        Args:
            qbit_map: Dictionary mapping initial qubit indices to remapped qubit indices
                     Format: {int(initial_qbit): int(remapped_qbit)}
            qbit_num: Number of qubits in the remapped circuit (int, optional).
                     Can be different from the original circuit. If None, uses the original number

        Returns:
            qgd_Circuit: A newly created, remapped circuit instance
        """

        if qbit_num == None:
            qbit_num = self.get_Qbit_Num()

        return super().Remap_Qbits(qbit_map, qbit_num)

    def get_Parameter_Start_Index(self):
        """Get the starting index of the parameters in the parameter array.

        The starting index corresponds to the circuit in which the current gate is incorporated.

        Returns:
            int: The starting index of parameters
        """

        # call the C wrapper function
        return super().get_Parameter_Start_Index()

    def get_Parents(self, gate):
        """Get the list of parent gate indices.

        The parent gates can be obtained from the list of gates involved in the circuit.

        Args:
            gate: Gate index (int) for which to retrieve parent gates

        Returns:
            list: List of parent gate indices
        """

        # call the C wrapper function
        return super().get_Parents(gate)

    def get_Children(self, gate):
        """Get the list of child gate indices.

        The children gates can be obtained from the list of gates involved in the circuit.

        Args:
            gate: Gate index (int) for which to retrieve child gates

        Returns:
            list: List of child gate indices
        """

        # call the C wrapper function
        return super().get_Children(gate)

    def add_Gate(self, qgd_gate):
        """Add a generic gate to the circuit.

        Args:
            qgd_gate: A gate object from the gates_Wrapper module

        Raises:
            Exception: If the gate type is not implemented
        """

        if isinstance(qgd_gate, H):
            self.add_H(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, X):
            self.add_X(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, Y):
            self.add_Y(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, Z):
            self.add_Z(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, CH):
            self.add_CH(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, CZ):
            self.add_CZ(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, RX):
            self.add_RX(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, RY):
            self.add_RY(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, RZ):
            self.add_RZ(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, SX):
            self.add_SX(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, U1):
            self.add_U1(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, U2):
            self.add_U2(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, U3):
            self.add_U3(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, CRY):
            self.add_CRY(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, CNOT):
            self.add_CNOT(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, S):
            self.add_S(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, Sdg):
            self.add_Sdg(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, T):
            self.add_T(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, Tdg):
            self.add_Tdg(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, R):
            self.add_R(qgd_gate.get_Target_Qbit())
        elif isinstance(qgd_gate, CROT):
            self.add_CROT(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, CR):
            self.add_CR(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, SYC):
            self.add_SYC(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, CRZ):
            self.add_CRZ(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, CRX):
            self.add_CRX(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, CP):
            self.add_CP(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbit())
        elif isinstance(qgd_gate, SWAP):
            self.add_SWAP(qgd_gate.get_Target_Qbits())
        elif isinstance(qgd_gate, RXX):
            self.add_RXX(qgd_gate.get_Target_Qbits())
        elif isinstance(qgd_gate, CSWAP):
            self.add_CSWAP(qgd_gate.get_Target_Qbits(), qgd_gate.get_Control_Qbits())
        elif isinstance(qgd_gate, CCX):
            self.add_CCX(qgd_gate.get_Target_Qbit(), qgd_gate.get_Control_Qbits())
        else:
            raise Exception("Cannot add gate: unimplemented gate type")
