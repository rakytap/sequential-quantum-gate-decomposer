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



    def test_RZ_squander(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        from qgd_python.gates.qgd_RZ import qgd_RZ

        # number of qubits
        qbit_num = 1

        # target qbit
        target_qbit = 0

        # creating an instance of the C++ class
        RZ = qgd_RZ( qbit_num, target_qbit )

        parameters = np.array( [pi/2*0.32 ] )
        
        RZ_squander= RZ.get_Matrix( parameters )
        
#QISKIT

        backend = Aer.get_backend('aer_simulator')


        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(qbit_num)

        # Add the u3 gate on qubit pi, pi,
        circuit.rz(parameters[0], target_qbit)
        circuit.save_unitary()
                
        # job execution and getting the result as an object
        job = execute(circuit, backend)
        
        # the result of the Qiskit job
        result=job.result()  
        
        # the unitary matrix from the result object
        RZ_qiskit = result.get_unitary(circuit)
        RZ_qiskit = np.asarray(RZ_qiskit)
        

        # the unitary matrix from the result object
        product_matrix = np.dot(RZ_squander, RZ_qiskit.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(2)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        error = (np.real(np.trace(product_matrix)))/2
       
        #print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )

