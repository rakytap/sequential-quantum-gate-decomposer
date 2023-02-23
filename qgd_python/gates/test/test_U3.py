import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_U3 import qgd_U3
import math
from scipy.stats import unitary_group

class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""

    pi=np.pi


    # # set the free parameters
    Theta = True
    Phi = True
    Lambda = True

    parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

    def test_U3_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """
        global U3_qiskit
        U3_qiskit = [0]*6

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            U3 = qgd_U3( qbit_num, target_qbit, self.Theta, self.Phi, self.Lambda )

	    #SQUANDER

            # get the matrix              
            U3_squander = U3.get_Matrix( self.parameters )
        
            #print(U3_squander)

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the RX gate on qubit pi, pi,
            circuit.u(self.parameters[0]*2, self.parameters[1], self.parameters[2], target_qbit)

            # the unitary matrix from the result object
            U3_qiskit[qbit_num-1] = get_unitary_from_qiskit_circuit( circuit )
            U3_qiskit[qbit_num-1] = np.asarray(U3_qiskit[qbit_num-1])
        
            # Draw the circuit        
            #print(RX_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=U3_squander-U3_qiskit[qbit_num-1]

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_U3_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            U3 = qgd_U3( qbit_num, target_qbit, self.Theta, self.Phi, self.Lambda )

            #create text matrix 
            test_m = unitary_group.rvs(((2**qbit_num)))           
            test_matrix = np.dot(test_m, test_m.conj().T)

	    #QISKIT      

            # apply the gate on the input array/matrix 
            U3_qiskit_apply_gate=np.matmul(U3_qiskit[qbit_num-1], test_matrix)

	    #SQUANDER

            U3_squander=test_matrix

            # apply the gate on the input array/matrix                
            U3.apply_to(self.parameters, U3_squander )
        
            #print(RX_squander)

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=U3_squander-U3_qiskit_apply_gate

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )










