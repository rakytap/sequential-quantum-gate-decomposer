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



    def test_RY_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        pi=np.pi

        # parameters
        parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            RY = qgd_RY( qbit_num, target_qbit )

	    #SQUANDER

            # get the matrix              
            RY_squander = RY.get_Matrix( parameters )

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the ry gate on qubit pi, pi,
            circuit.ry(parameters[0]*2, target_qbit)

            # the unitary matrix from the result object
            RY_qiskit = get_unitary_from_qiskit_circuit( circuit )
            RY_qiskit = np.asarray(RY_qiskit)

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=RY_squander-RY_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_RY_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """


        pi=np.pi

        # parameters
        parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            RY = qgd_RY( qbit_num, target_qbit )

            #create text matrix 
            test_matrix= np.eye( int( pow(2,qbit_num) ))

	    #QISKIT      
            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the ry gate on qubit pi, pi,
            circuit.ry(parameters[0]*2, target_qbit)

            # the unitary matrix from the result object
            RY_qiskit = get_unitary_from_qiskit_circuit( circuit )
            RY_qiskit = np.asarray(RY_qiskit)

            # apply the gate on the input array/matrix 
            RY_qiskit_apply_gate=np.matmul(RY_qiskit[qbit_num-1], test_matrix)

	    #SQUANDER

            RY_squander=test_matrix

            # apply the gate on the input array/matrix                
            RY.apply_to(parameters, RY_squander )

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=RY_squander-RY_qiskit_apply_gate

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 




