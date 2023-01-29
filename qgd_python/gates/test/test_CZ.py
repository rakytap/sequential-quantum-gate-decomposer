import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_CZ import qgd_CZ


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""

    pi=np.pi

    # number of qubits
    qbit_num = 2

    # target qbit
    target_qbit = 0

    # control_qbit
    control_qbit = 1

    # creating an instance of the C++ class
    CZ = qgd_CZ( qbit_num, target_qbit, control_qbit )


    def test_CZ_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

	#SQUANDER#

        # get the matrix                     
        CZ_squander = self.CZ.get_Matrix( )
        
        #print(CZ_squander)

	#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the CZ gate on control qbit and target qbit
        circuit.cz( self.control_qbit, self.target_qbit )
        
        # the unitary matrix from the result object
        CZ_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CZ_qiskit = np.asarray(CZ_qiskit)
        
        # Draw the circuit        
        #print(CZ_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CZ_squander-CZ_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 

    def test_CZ_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

	#SQUANDER

        # get the matrix              
        CZ_squander = self.CZ.get_Matrix(  )

        # apply the gate on the input array/matrix                
        self.CZ.apply_to(CZ_squander )
        
        #print(CZ_squander)

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the CZ gate on control qbit and target qbit
        circuit.cz( self.control_qbit, self.target_qbit )
        
        # the unitary matrix from the result object
        CZ_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CZ_qiskit = np.asarray(CZ_qiskit)

        # the CZ gate 
        cz_gate=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0.,1., 0.], [0., 0., 0.,-1.]])

        # apply the gate on the input array/matrix 
        CZ_qiskit_apply_gate=np.matmul(CZ_qiskit, cz_gate)

        #print(np.around(CH_qiskit_apply_gate,1))

        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CZ_squander-CZ_qiskit_apply_gate

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 


