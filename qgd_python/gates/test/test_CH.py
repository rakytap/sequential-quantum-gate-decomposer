import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit

pi=np.pi

#SQUANDER

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_CH_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        from qgd_python.gates.qgd_CH import qgd_CH

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1

        # creating an instance of the C++ class
        CH = qgd_CH( qbit_num, target_qbit, control_qbit )

        CH_squander = CH.get_Matrix(  )
        
        #print(CH_squander)

#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the CH gate 
        circuit.ch(control_qbit, target_qbit)
        
        # the unitary matrix from the result object
        CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CH_qiskit = np.asarray(CH_qiskit)
        
        # Draw the circuit        
        #print(CH_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CH_squander-CH_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )




