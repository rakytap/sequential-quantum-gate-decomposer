import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit

pi=np.pi


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""
#SQUANDER#

    def test_SX_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        from qgd_python.gates.qgd_SX import qgd_SX

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # creating an instance of the C++ class
        SX = qgd_SX( qbit_num, target_qbit )
                
        SX_squander = SX.get_Matrix( )

	#print(SX_squander)
        
 
#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the CNOT gate on control qbit and target qbit
        circuit.sx( target_qbit )

        # the unitary matrix from the result object
        SX_qiskit = get_unitary_from_qiskit_circuit( circuit )
        SX_qiskit = np.asarray(SX_qiskit)

        # Draw the circuit        
        #print(SX_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=SX_squander-SX_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 





