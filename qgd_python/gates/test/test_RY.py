import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_RY import qgd_RY
import math
from scipy.stats import unitary_group

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""

    pi=np.pi

    # parameters
    parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

    def test_RY_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """
        global RY_qiskit
        RY_qiskit = [0]*6

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            RY = qgd_RY( qbit_num, target_qbit )

	    #SQUANDER

            # get the matrix              
            RY_squander = RY.get_Matrix( self.parameters )
        
            #print(RX_squander)

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the ry gate on qubit pi, pi,
            circuit.ry(self.parameters[0]*2, target_qbit)

            # the unitary matrix from the result object
            RY_qiskit[qbit_num-1] = get_unitary_from_qiskit_circuit( circuit )
            RY_qiskit[qbit_num-1] = np.asarray(RY_qiskit[qbit_num-1])
        
            # Draw the circuit        
            #print(RY_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=RY_squander-RY_qiskit[qbit_num-1]

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_RY_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            RY = qgd_RY( qbit_num, target_qbit )

            #create text matrix 
            test_m = unitary_group.rvs(((2**qbit_num)))           
            test_matrix = np.dot(test_m, test_m.conj().T)

	    #QISKIT      

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # apply the gate on the input array/matrix 
            RY_qiskit_apply_gate=np.matmul(RY_qiskit[qbit_num-1], test_matrix)

	    #SQUANDER

            RY_squander=test_matrix

            # apply the gate on the input array/matrix                
            RY.apply_to(self.parameters, RY_squander )
        
            #print(RX_squander)

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=RY_squander-RY_qiskit_apply_gate

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 




