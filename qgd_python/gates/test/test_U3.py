import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit

pi=np.pi

#SQUANDER
class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""



    def test_U3_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        from qgd_python.gates.qgd_U3 import qgd_U3

        # number of qubits
        qbit_num = 2

        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        

        # creating an instance of the C++ class
        U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )
        
        parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )
        
        U3_squander = U3.get_Matrix( parameters )
        
        #print(U3_squander)

#QISKIT

        

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.u(parameters[0]*2, parameters[1], parameters[2], target_qbit)             
      
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
 










