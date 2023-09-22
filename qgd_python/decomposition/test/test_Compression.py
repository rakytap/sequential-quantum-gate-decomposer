# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:42:56 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: Peter Rakyta, Ph.D.
"""
## \file test_decomposition.py
## \brief Functionality test cases for the qgd_N_Qubit_Decomposition class.



# cerate unitary q-bit matrix
from scipy.stats import unitary_group
import numpy as np
from squander import utils


try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False



class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""

    def test_IBM_Chellenge_full(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 4-qubit unitary of the IBM chellenge

        """

        from squander import N_Qubit_Decomposition_adaptive       
        from scipy.io import loadmat
    
        # load the unitary from file
        data = loadmat('Umtx.mat')  
        # The unitary to be decomposed  
        Umtx = data['Umtx']

        # generate config structure
        config = { 'max_outer_iterations': 10, 
		'max_inner_iterations': 300000, 	
		'max_inner_iterations_compression': 10000, 
		'max_inner_iterations_final': 1000, 	
	        'randomized_adaptive_layers': 1,
	        'export_circuit_2_binary': 1,
		'optimization_tolerance': 1e-8 }

        
        

        # creating a class to decompose the unitary
        cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config )


        # setting the verbosity of the decomposition
        cDecompose.set_Verbose( 3 )

	# adding decomposing layers to the gate structure
        levels = 2
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()

        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
	


        # starting the decomposition
        cDecompose.get_Initial_Circuit()

        # compress the initial gate structure
        cDecompose.Compress_Circuit()

        # finalize the gate structur (replace CRY gates with CNOT gates)
        cDecompose.Finalize_Circuit()

        # list the decomposing operations
        cDecompose.List_Gates()

        # get the decomposing operations
        quantum_circuit = cDecompose.get_Quantum_Circuit()

        # print the quantum circuit
        print(quantum_circuit)

        import numpy.linalg as LA
    
        # the unitary matrix from the result object
        decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(16)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error = (np.real(np.trace(product_matrix)))/2
       
        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )

       
    def test_IBM_Chellenge_no_compression(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 4-qubit unitary of the IBM chellenge

        """

        from squander import N_Qubit_Decomposition_adaptive       
        from scipy.io import loadmat
    
        # load the unitary from file
        data = loadmat('Umtx.mat')  
        # The unitary to be decomposed  
        Umtx = data['Umtx']
        
        # generate config structure
        config = { 'max_outer_iterations': 10, 
		'max_inner_iterations': 300000, 	
		'max_inner_iterations_compression': 10000, 
		'max_inner_iterations_final': 1000, 	
	        'randomized_adaptive_layers': 1,
	        'export_circuit_2_binary': 1,
		'optimization_tolerance': 1e-8 }


        # creating a class to decompose the unitary
        cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config )


        # setting the verbosity of the decomposition
        cDecompose.set_Verbose( 3 )


	# adding decomposing layers to the gate structure
        levels = 3
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()

        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()


        # starting the decomposition
        cDecompose.get_Initial_Circuit()

        # finalize the gate structur (replace CRY gates with CNOT gates)
        cDecompose.Finalize_Circuit()

        # list the decomposing operations
        cDecompose.List_Gates()

        # get the decomposing operations
        quantum_circuit = cDecompose.get_Quantum_Circuit()

        # print the quantum circuit
        print(quantum_circuit)

        import numpy.linalg as LA
    
        # the unitary matrix from the result object
        decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(16)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error = (np.real(np.trace(product_matrix)))/2
       
        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )


    def test_IBM_Chellenge_compression_only(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 4-qubit unitary of the IBM chellenge

        """

        from squander import N_Qubit_Decomposition_adaptive       
        from scipy.io import loadmat
    
        # load the unitary from file
        data = loadmat('Umtx.mat')  
        # The unitary to be decomposed  
        Umtx = data['Umtx']


        # generate config structure
        config = { 'max_outer_iterations': 10, 
		'max_inner_iterations': 300000, 	
		'max_inner_iterations_compression': 10000, 
		'max_inner_iterations_final': 1000, 	
	        'randomized_adaptive_layers': 1,
	        'export_circuit_2_binary': 1,
		'optimization_tolerance': 1e-8 }
        

        # creating a class to decompose the unitary
        cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config )


        # setting the verbosity of the decomposition
        cDecompose.set_Verbose( 3 )



        # importing circuit from a binary
        cDecompose.set_Gate_Structure_From_Binary("circuit_squander.binary")
        
        # starting compression iterations
        cDecompose.Compress_Circuit()

        # finalize the gate structur (replace CRY gates with CNOT gates)
        cDecompose.Finalize_Circuit()

        # list the decomposing operations
        cDecompose.List_Gates()

        # get the decomposing operations
        quantum_circuit = cDecompose.get_Quantum_Circuit()

        # print the quantum circuit
        print(quantum_circuit)

        import numpy.linalg as LA
    
        # the unitary matrix from the result object
        decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(16)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error = (np.real(np.trace(product_matrix)))/2
       
        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )


    def test_IBM_Chellenge_multiple_optim(self):
        r"""
        This method is called by pytest. 
        Test to decompose a 4-qubit unitary of the IBM chellenge

        """

        from squander import N_Qubit_Decomposition_adaptive       
        from scipy.io import loadmat
    
        # load the unitary from file
        data = loadmat('Umtx.mat')  
        # The unitary to be decomposed  
        Umtx = data['Umtx']
        config = { 'max_outer_iterations': 1, 
		'max_inner_iterations_agent': 25000, 
		'max_inner_iterations_compression': 10000,
		'max_inner_iterations' : 500,
		'max_inner_iterations_final': 5000, 		
		'Randomized_Radius': 0.3, 
                'randomized_adaptive_layers': 1,
		'optimization_tolerance_agent': 1e-4,
		'optimization_tolerance': 1e-5,

                'agent_num': 10}

        # creating a class to decompose the unitary
        cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, config=config )


        # setting the verbosity of the decomposition
        cDecompose.set_Verbose( 3 )


	# adding decomposing layers to the gate structure
        levels = 2
        for idx in range(levels):
            cDecompose.add_Adaptive_Layers()

        cDecompose.add_Finalyzing_Layer_To_Gate_Structure()
        


        # setting intial parameter set
        parameter_num = cDecompose.get_Parameter_Num()
        parameters = np.zeros( (parameter_num,1), dtype=np.float64 )
        cDecompose.set_Optimized_Parameters( parameters )

        # setting optimizer
        cDecompose.set_Optimizer("AGENTS")

        # starting the decomposition
        cDecompose.get_Initial_Circuit()

        # setting optimizer
        cDecompose.set_Optimizer("BFGS")

        # continue the decomposition with a second optimizer method
        cDecompose.get_Initial_Circuit()

        # starting compression iterations
        cDecompose.Compress_Circuit()

        # finalize the gate structur (replace CRY gates with CNOT gates)
        cDecompose.Finalize_Circuit()

        # list the decomposing operations
        cDecompose.List_Gates()

        # get the decomposing operations
        quantum_circuit = cDecompose.get_Quantum_Circuit()

        # print the quantum circuit
        print(quantum_circuit)

        import numpy.linalg as LA
    
        # the unitary matrix from the result object
        decomposed_matrix = utils.get_unitary_from_qiskit_circuit( quantum_circuit )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(16)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error = (np.real(np.trace(product_matrix)))/2
       
        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )
    
        


