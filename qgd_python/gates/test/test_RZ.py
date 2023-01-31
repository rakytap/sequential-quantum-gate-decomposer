import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_RZ import qgd_RZ
import math


class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""

    pi=np.pi

    # number of qubits
    qbit_num = 1

    # target qbit
    target_qbit = 0

    # parameters
    parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

    # creating an instance of the C++ class
    RZ = qgd_RZ( qbit_num, target_qbit )



    def test_RZ_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RZ gate.
        """

        #SQUANDER

        # get the matrix              
        RZ_squander = self.RZ.get_Matrix( self.parameters )
        
        #print(RX_squander)
        
        #QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.rz(self.parameters[0], self.target_qbit)
        
        # the unitary matrix from the result object
        RZ_qiskit = get_unitary_from_qiskit_circuit( circuit )
        RZ_qiskit = np.asarray(RZ_qiskit)
      
        # the unitary matrix from the result object
        product_matrix = np.dot(RZ_squander, RZ_qiskit.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(2)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        error = (np.real(np.trace(product_matrix)))/2
       
        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )

    def test_RZ_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        #SQUANDER
 
        # get the matrix               
        RZ_squander = self.RZ.get_Matrix( self.parameters )

        # apply the gate on the input array/matrix                
        self.RZ.apply_to(self.parameters, RZ_squander )
        
        #print(RZ_squander)

	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.rz(self.parameters[0], self.target_qbit)

        # the unitary matrix from the result object
        RZ_qiskit = get_unitary_from_qiskit_circuit( circuit )
        RZ_qiskit = np.asarray(RZ_qiskit)

        # the R gate 
        rz_gate=np.array([[np.exp(-1.j*(self.parameters[2]/2)), 0.], [0. ,np.exp(1.j*(self.parameters[2]/2))] ])

        #print(rz_gate)
 
        # apply the gate on the input array/matrix 
        RZ_qiskit_apply_gate=np.matmul(RZ_qiskit, rz_gate)

        # the unitary matrix from the result object
        product_matrix = np.dot(RZ_squander, RZ_qiskit_apply_gate.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)

        # the error of the decomposition
        error = (np.real(np.trace(product_matrix)))/2
        #error = (np.real(np.trace(RZ_qiskit_apply_gate-RZ_squander)))/2       
        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        #assert( error < 1e-3 )) 


 
a=Test_operations_squander()
a.test_RZ_get_matrix()
a.test_RZ_apply_to()
