import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_X import qgd_X



class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""

    pi=np.pi

    # number of qubits
    qbit_num = 1

    # target qbit
    target_qbit = 0

    # creating an instance of the C++ class
    X = qgd_X( qbit_num, target_qbit )

    def test_X_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of X gate and compare with qiskit.
        """
	#SQUANDER#

        # get the matrix                
        X_squander = self.X.get_Matrix( )

	#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the CNOT gate on control qbit and target qbit
        circuit.x( self.target_qbit )

        # the unitary matrix from the result object
        X_qiskit = get_unitary_from_qiskit_circuit( circuit )
        X_qiskit = np.asarray(X_qiskit)
        
        # Draw the circuit        
        #print(X_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=X_squander-X_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 

    def test_X_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of X gate and compare with qiskit.
        """
	#SQUANDER
              
        # get the matrix             
        X_squander = self.X.get_Matrix(  )

        # apply the gate on the input array/matrix                
        self.X.apply_to(X_squander)
              
        #print(X_squander)             

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)
      
        # Add the X gate on the target qbit
        circuit.x( self.target_qbit )

        # the unitary matrix from the result object
        X_qiskit = get_unitary_from_qiskit_circuit( circuit )
        X_qiskit = np.asarray(X_qiskit)

        # the X gate 
        x_gate=np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])

        # apply the gate on the input array/matrix  
        X_qiskit_apply_gate=np.matmul(X_qiskit, x_gate)

        #print(X_qiskit_apply_gate)

        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=X_squander-X_qiskit_apply_gate

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 





