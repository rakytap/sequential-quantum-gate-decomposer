import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_CH import qgd_CH
import math
        

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""


    pi=np.pi

    # number of qubits
    qbit_num = 2

    # target qbit
    target_qbit = 0

    # control_qbit
    control_qbit = 1

    # creating an instance of the C++ class
    CH = qgd_CH( qbit_num, target_qbit, control_qbit )

    def test_CH_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """
	#SQUANDER

        # get the matrix     
        CH_squander = self.CH.get_Matrix(  )
        
        #print(CH_squander)

	#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the CH gate 
        circuit.ch(self.control_qbit, self.target_qbit)
        
        # the unitary matrix from the result object
        CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CH_qiskit = np.asarray(CH_qiskit)
        
        # Draw the circuit        
        #print(CH_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CH_squander-CH_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )

    def test_CH_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

	#SQUANDER

        # get the matrix          
        CH_squander = self.CH.get_Matrix(  )

        # apply the gate on the input array/matrix                   
        self.CH.apply_to(CH_squander )
        
        #print(CH_squander)

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)
      
        # Add the X gate on the target qbit
        circuit.ch(self.control_qbit, self.target_qbit)

        # the unitary matrix from the result object
        CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CH_qiskit = np.asarray(CH_qiskit)

        # the CH gate 
        ch_gate=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1./math.sqrt(2), 1./math.sqrt(2)], [0., 0., 1./math.sqrt(2),-1./math.sqrt(2)]])

        # apply the gate on the input array/matrix 
        CH_qiskit_apply_gate=np.matmul(CH_qiskit, ch_gate)

        #print(np.around(CH_qiskit_apply_gate,1))

        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CH_squander-CH_qiskit_apply_gate

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 


