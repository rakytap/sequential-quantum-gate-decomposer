import numpy as np
import random

from qiskit import QuantumRegister, ClassicalRegister, BasicAer
from qiskit import QuantumCircuit, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer

from scipy.stats import unitary_group

pi=np.pi


class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""
#SQUANDER#

    def test_SYC_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        from qgd_python.gates.qgd_SYC import qgd_SYC

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # control qbit
        control_qbit = 1
     
        # creating an instance of the C++ class
        SYC = qgd_SYC( qbit_num, target_qbit, control_qbit )
                
        SYC_squander = SYC.get_Matrix( )
        
        print(SYC_squander)

#QISKIT

        backend = Aer.get_backend('unitary_simulator')


        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the CNOT gate on control qbit and target qbit
        circuit.cy( control_qbit, target_qbit )
                
        # job execution and getting the result as an object
        job = execute(circuit, backend)
        
        # the result of the Qiskit job
        result=job.result()  
        
        # the unitary matrix from the result object
        SYC_qiskit = result.get_unitary(circuit)
        SYC_qiskit = np.asarray(SYC_qiskit)
        
        # Draw the circuit        
        print(SYC_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=SYC_squander-SYC_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 ) 


