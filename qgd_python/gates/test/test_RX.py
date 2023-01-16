import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit

pi=np.pi

#SQUANDER
class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_RX_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        from qgd_python.gates.qgd_RX import qgd_RX

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # creating an instance of the C++ class
        RX = qgd_RX( qbit_num, target_qbit )

        parameters = np.array( [pi/2*0.32] )
        
        RX_squander = RX.get_Matrix( parameters )
        
        #print(RX_squander)

#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.rx(parameters[0]*2, target_qbit)

        # the unitary matrix from the result object
        RX_qiskit = get_unitary_from_qiskit_circuit( circuit )
        RX_qiskit = np.asarray(RX_qiskit)
        
        # Draw the circuit        
        #print(RX_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=RX_squander-RX_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )        
 


