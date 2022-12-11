import numpy as np
import random

from qiskit import QuantumRegister, ClassicalRegister, BasicAer
from qiskit import QuantumCircuit, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer

from scipy.stats import unitary_group

pi=np.pi

#SQUANDER

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_CH_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        from qgd_python.gates.qgd_CH import qgd_CH

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1

        # creating an instance of the C++ class
        CH = qgd_CH( qbit_num, target_qbit, control_qbit )

        CH_squander = CH.get_Matrix(  )
        
        print(CH_squander)

#QISKIT

        backend = Aer.get_backend('unitary_simulator')


        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the CH gate 
        circuit.ch(control_qbit, target_qbit)
                
        # job execution and getting the result as an object
        job = execute(circuit, backend)
        
        # the result of the Qiskit job
        result=job.result()  
        
        # the unitary matrix from the result object
        CH_qiskit = result.get_unitary(circuit)
        CH_qiskit = np.asarray(CH_qiskit)
        
        # Draw the circuit        
        print(CH_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CH_squander-CH_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )



