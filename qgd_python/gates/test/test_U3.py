import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_U3 import qgd_U3
import math


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""

    pi=np.pi

    # number of qubits
    qbit_num = 1

    # target qbit
    target_qbit = 0

    # # set the free parameters
    Theta = True
    Phi = True
    Lambda = True

    parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

    # creating an instance of the C++ class
    U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )

    def test_U3_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
	#SQUANDER

        # get the matrix          
        U3_squander = self.U3.get_Matrix( self.parameters )
        
        #print(U3_squander)

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.u(self.parameters[0]*2, self.parameters[1], self.parameters[2], self.target_qbit)             
      
        # the unitary matrix from the result object
        U3_qiskit = get_unitary_from_qiskit_circuit( circuit )
        U3_qiskit = np.asarray(U3_qiskit)
        
        # Draw the circuit        
        #print(U3_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=U3_squander-U3_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )


    def test_U3_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

	#SQUANDER
              
        # get the matrix             
        U3_squander = self.U3.get_Matrix(self.parameters  )

        # apply the gate on the input array/matrix                
        self.U3.apply_to(self.parameters, U3_squander)
              
        #print(X_squander)             

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)
      
        # Add the X gate on the target qbit
        circuit.u( self.parameters[0]*2, self.parameters[1], self.parameters[2], self.target_qbit )

        # the unitary matrix from the result object
        U3_qiskit = get_unitary_from_qiskit_circuit( circuit )
        U3_qiskit = np.asarray(U3_qiskit)

        # the U3 gate 

        u3_gate=np.array([[math.cos(self.parameters[0]),-np.exp(1.j*self.parameters[2])*math.sin(self.parameters[0])], [np.exp(1.j*self.parameters[1])*math.sin(self.parameters[0]), np.exp(1.j*(self.parameters[1]+self.parameters[2]))*math.cos(self.parameters[0])]]) 

        # apply the gate on the input array/matrix  
        U3_qiskit_apply_gate=np.matmul(U3_qiskit, u3_gate)

        #print(X_qiskit_apply_gate)

        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=U3_squander-U3_qiskit_apply_gate

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 

      










