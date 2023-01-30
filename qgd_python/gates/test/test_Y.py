import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_Y import qgd_Y


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""

    pi=np.pi

    # number of qubits
    qbit_num = 1

    # target qbit
    target_qbit = 0

    # creating an instance of the C++ class
    Y = qgd_Y( qbit_num, target_qbit )

    def test_Y_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

	#SQUANDER#

        # get the matrix                
        Y_squander = self.Y.get_Matrix( )

        #print(Y_squander)

	#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the CNOT gate on control qbit and target qbit
        circuit.y( self.target_qbit )

        # the unitary matrix from the result object
        Y_qiskit = get_unitary_from_qiskit_circuit( circuit )
        Y_qiskit = np.asarray(Y_qiskit)
        
        # Draw the circuit        
        #print(Y_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=Y_squander-Y_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 

    def test_Y_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of X gate and compare with qiskit.
        """
	#SQUANDER

        # get the matrix                  
        Y_squander = self.Y.get_Matrix(  )

        # apply the gate on the input array/matrix              
        self.Y.apply_to(Y_squander )
              
        #print(Y_squander)             

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)
      
        # Add the Y gate on the target qbit
        circuit.y( self.target_qbit )

        # the unitary matrix from the result object
        Y_qiskit = get_unitary_from_qiskit_circuit( circuit )
        Y_qiskit = np.asarray(Y_qiskit)

        # the Y gate 
        y_gate=np.array([[0., 0.-1.j], [0.+1.j, 0.]])

        # apply the gate on the input array/matrix  
        y_qiskit_apply_gate=np.matmul(Y_qiskit, y_gate)

        #print(Y_qiskit_apply_gate)

        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=Y_squander-y_qiskit_apply_gate

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 




