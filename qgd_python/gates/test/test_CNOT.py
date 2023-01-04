import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit

pi=np.pi


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""
#SQUANDER#

    def test_CNOT_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        from qgd_python.gates.qgd_CNOT import qgd_CNOT

        # number of qubits
        qbit_num = 2

        # target qbit
        target_qbit = 0

        # control qbit
        control_qbit = 1
     
        # creating an instance of the C++ class
        CNOT = qgd_CNOT( qbit_num, target_qbit, control_qbit )
                
        CNOT_squander = CNOT.get_Matrix( )
        
        #print(CNOT_squander)

#QISKIT


        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the CNOT gate on control qbit and target qbit
        circuit.cx( control_qbit, target_qbit )
                       
        # the unitary matrix from the result object
        CNOT_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CNOT_qiskit = np.asarray(CNOT_qiskit)
        
        # Draw the circuit        
        #print(CNOT_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CNOT_squander-CNOT_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 




