import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_CH import qgd_CH
import math
from scipy.stats import unitary_group        

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""


    pi=np.pi


    def test_CH_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CH gate.
        """
        global CH_qiskit
        CH_qiskit = [0]*6

        for qbit_num in range(2,7):

            # target qbit
            target_qbit = qbit_num-2

            # control qbit
            control_qbit = qbit_num-1

            # creating an instance of the C++ class
            CH = qgd_CH( qbit_num, target_qbit, control_qbit )

	    #SQUANDER

            # get the matrix              
            CH_squander = CH.get_Matrix(  )
        
            #print(CH_squander)

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the CH gate on qubit pi, pi,
            circuit.ch(control_qbit, target_qbit)

            # the unitary matrix from the result object
            CH_qiskit[qbit_num-1] = get_unitary_from_qiskit_circuit( circuit )
            CH_qiskit[qbit_num-1] = np.asarray(CH_qiskit[qbit_num-1])
        
            # Draw the circuit        
            #print(CH_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=CH_squander-CH_qiskit[qbit_num-1]

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_CH_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
        for qbit_num in range(2,7):

            # target qbit
            target_qbit = qbit_num-2

            # control qbit
            control_qbit = qbit_num-1

            # creating an instance of the C++ class
            CH = qgd_CH( qbit_num, target_qbit, control_qbit )

            #create text matrix 
            test_m = unitary_group.rvs(((2**qbit_num)))           
            test_matrix = np.dot(test_m, test_m.conj().T)

	    #QISKIT      

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # apply the gate on the input array/matrix 
            CH_qiskit_apply_gate=np.matmul(CH_qiskit[qbit_num-1], test_matrix)

	    #SQUANDER

            CH_squander=test_matrix

            # apply the gate on the input array/matrix                
            CH.apply_to(CH_squander )
        
            #print(CH_squander)

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=CH_squander-CH_qiskit_apply_gate

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 







