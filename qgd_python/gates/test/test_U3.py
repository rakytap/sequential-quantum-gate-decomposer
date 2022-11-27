import numpy as np
import random

from qiskit import QuantumRegister, ClassicalRegister, BasicAer
from qiskit import QuantumCircuit, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer
from qgd_python.gates.qgd_U3 import qgd_U3


from scipy.stats import unitary_group

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
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        

        # creating an instance of the C++ class
        U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )
        
        parameters = np.array( [pi/2, pi, pi/2] )
        
        U3_matrix = U3.get_Matrix( parameters )
        
        print(np.around(U3_matrix,2))

#QISKIT

        backend = Aer.get_backend('unitary_simulator')

        q=QuantumRegister(3,'q')
        c=ClassicalRegister(3, 'c')

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(q, c)

        # Add the u3 gate on qubit pi, pi,
        circuit.u(pi, pi, pi/2, q[0])
                
        # job execution and getting the result as an object
        job = execute(circuit, backend)
        
        # the result of the Qiskit job
        result=job.result()  
        
        # the unitary matrix from the result object
        decomposed_matrix = result.get_unitary(circuit,20)
        decomposed_matrix = np.asarray(decomposed_matrix)
        
        # Draw the circuit        
        print(result.get_unitary(circuit,2))
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=U3_matrix-decomposed_matrix

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )        
 








