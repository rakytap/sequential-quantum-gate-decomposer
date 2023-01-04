import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit

pi=np.pi

#SQUANDER
class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_RY_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        from qgd_python.gates.qgd_RY import qgd_RY

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # creating an instance of the C++ class
        RY = qgd_RY( qbit_num, target_qbit )

        parameters = np.array( [pi/2*0.32] )
        
        RY_squander= RY.get_Matrix( parameters )
        
        #print(RY_squander)


#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.ry(parameters[0]*2, target_qbit)
        
        # the unitary matrix from the result object
        RY_qiskit = get_unitary_from_qiskit_circuit( circuit )
        RY_qiskit = np.asarray(RY_qiskit)
        
        # Draw the circuit        
        #print(RY_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=RY_squander-RY_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 


