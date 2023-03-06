import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_CZ import qgd_CZ
from scipy.stats import unitary_group


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""

    def test_CZ_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
        pi=np.pi

        for qbit_num in range(2,7):

            # target qbit
            target_qbit = qbit_num-2

            # control_qbit
            control_qbit = qbit_num-1

            # creating an instance of the C++ class
            CZ = qgd_CZ( qbit_num, target_qbit, control_qbit )

	    #SQUANDER#

            # get the matrix                     
            CZ_squander = CZ.get_Matrix( )

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the CZ gate on control qbit and target qbit
            circuit.cz( control_qbit, target_qbit )
        
            # the unitary matrix from the result object
            CZ_qiskit = get_unitary_from_qiskit_circuit( circuit )
            CZ_qiskit= np.asarray(CZ_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=CZ_squander-CZ_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 

    def test_CZ_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
        pi=np.pi

        for qbit_num in range(2,7):

            # target qbit
            target_qbit = qbit_num-2

            # control_qbit
            control_qbit = qbit_num-1

            # creating an instance of the C++ class
            CZ = qgd_CZ( qbit_num, target_qbit, control_qbit )

            #create text matrix 
            test_matrix= np.eye( int( pow(2,qbit_num) ))         

	    #QISKIT    
  
            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the CZ gate on control qbit and target qbit
            circuit.cz( control_qbit, target_qbit )
        
            # the unitary matrix from the result object
            CZ_qiskit = get_unitary_from_qiskit_circuit( circuit )
            CZ_qiskit= np.asarray(CZ_qiskit)

            # apply the gate on the input array/matrix 
            CZ_qiskit_apply_gate=np.matmul(CZ_qiskit, test_matrix)

            #SQUANDER

            CZ_squander=test_matrix

            # apply the gate on the input array/matrix                
            CZ.apply_to(CZ_squander )

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=CZ_squander-CZ_qiskit_apply_gate

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 



