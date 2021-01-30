# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np


class Test_Example:
    """This is an example class to demonstrate how to interface with a C++ part of the piquasso project."""

    def test_N_Qubit_Decomposition_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of class N_Qubit_Decomposition.

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)
    
        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx, False, "RANDOM" )


    def etest_N_Qubit_Decomposition_3qubit(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 3-qubit unitary

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)

        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx, False, "ZEROS" )

        # start the decomposition
        decomp.Start_Decomposition(True, True)



    def test_N_Qubit_Decomposition_Qiskit_export(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 2-qubit unitary and retrive the corresponding Qiskit circuit

        """

        from qgd_python.qgd_N_Qubit_Decomposition import qgd_N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 2

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        Umtx = unitary_group.rvs(matrix_size)

        # creating an instance of the C++ class
        decomp = qgd_N_Qubit_Decomposition( Umtx.conj().T, False, "ZEROS" )

        # start the decomposition
        decomp.Start_Decomposition(True, True)

        # get the decomposing operations
        quantum_circuit = decomp.get_Quantum_Circuit()

        # print the list of decomposing operations
        decomp.List_Operations()

        # print the quantum circuit
        print(quantum_circuit)

        from qiskit import execute
        from qiskit import Aer
        import numpy.linalg as LA
    
        # test the decomposition of the matrix
        # Qiskit backend for simulator
        backend = Aer.get_backend('unitary_simulator')
     
        # job execution and getting the result as an object
        job = execute(quantum_circuit, backend)
        # the result of the Qiskit job
        result = job.result()
    
        # the unitary matrix from the result object
        decomposed_matrix = result.get_unitary(quantum_circuit)

    
        # the Umtx*Umtx' matrix
        product_matrix = np.dot(Umtx, decomposed_matrix.conj().T)
        # the error of the decomposition
        decomposition_error =  LA.norm(product_matrix - np.identity(4)*product_matrix[0,0], 2)

        print('The error of the decomposition is ' + str(decomposition_error))




      

