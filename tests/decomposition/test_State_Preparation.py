#!/usr/bin/python
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

## \file test_State_Preparation.py
## \brief Functionality test cases for the N_Qubit_State_Preparation_adaptive class.


from scipy.stats import unitary_group
import numpy as np
import pytest
from squander import utils

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False
    
    
import qiskit
qiskit_version = qiskit.version.get_version_info()

from qiskit import QuantumCircuit

    
if qiskit_version[0] == '1' or qiskit_version[0] == '2':
    from qiskit import transpile
    import qiskit_aer as Aer    
else :
    from qiskit import execute
    from qiskit import Aer        


class Test_State_Preparation:

    def test_State_Preparation_adaptive_false(self):  # atnevezni
        r"""
        This method is called by pytest. 
        Test to decompose a 4-qubit unitary State

        """

        from squander import N_Qubit_State_Preparation_adaptive
        from scipy.io import loadmat

        # load the unitary from file

        data = loadmat('data/Umtx.mat')

        # The unitary to be decomposed

        Umtx = data['Umtx'].conj().T

        # creating a class to decompose the unitary

        with pytest.raises(Exception):
            cDecompose = N_Qubit_State_Preparation_adaptive(Umtx,
                    level_limit_max=5, level_limit_min=0)

    def State_Preparation_adaptive_base(self, optimizer, cost_func, compression_enabled=1):

        from squander import N_Qubit_State_Preparation_adaptive
        from scipy.io import loadmat

        # load the unitary from file

        data = loadmat('data/Umtx.mat')

        # The unitary to be decomposed

        Umtx = data['Umtx']
        State = Umtx[:, 0].reshape(16, 1)
        norm = np.sqrt( State.conj().T @ State )
        State = State/norm
       

        config = { 'max_outer_iterations': 1, 
		        'max_inner_iterations': 10000, 
		        'max_inner_iterations_compression': 10000, 
		        'max_inner_iterations_final': 1000, 
		        'randomization_threshold': int(1e4),  			
		        'Randomized_Radius': 0.3, 
	            'randomized_adaptive_layers': 1,
		        'optimization_tolerance_agent': 1e-4,
		        'optimization_tolerance': 1e-4,
		        'compression_enabled': compression_enabled,
		        'number_of_agents': 4}


        # creating a class to decompose the unitary

        cDecompose = N_Qubit_State_Preparation_adaptive(State,
                level_limit_max=5, level_limit_min=0, config = config)

        # setting the verbosity of the decomposition

        cDecompose.set_Verbose(3)

        # setting the verbosity of the decomposition

        cDecompose.set_Cost_Function_Variant(cost_func)
        
        #set Optimizer
        
        cDecompose.set_Optimizer(optimizer)

        # set initial parameters
        rng = np.random.default_rng( 42 )
        num_of_parameters = cDecompose.get_Parameter_Num()
        parameters = rng.random(num_of_parameters)*2*np.pi
        
        cDecompose.set_Optimized_Parameters( parameters )
        
        # starting the decomposition

        cDecompose.Start_Decomposition()

        # list the decomposing operations

        cDecompose.List_Gates()

        # get the decomposing operations

        circuit_qiskit = cDecompose.get_Qiskit_Circuit()

        # print the quantum circuit

        print (circuit_qiskit)
        # the unitary matrix from the result object

        decomp_error = cDecompose.Optimization_Problem_Combined(cDecompose.get_Optimized_Parameters())[0]
        assert decomp_error < 1e-4
        print(f"DECOMPOSITION ERROR: {decomp_error} ")
        
        # Execute and get the state vector
        if qiskit_version[0] == '1' or qiskit_version[0] == '2':
	
            circuit_qiskit.save_statevector()
	
            backend = Aer.AerSimulator(method='statevector')
            compiled_circuit = transpile(circuit_qiskit, backend)
            result = backend.run(compiled_circuit).result()
		
            transformed_state = result.get_statevector(compiled_circuit)		
       
        
        elif qiskit_version[0] == '0':
	
            # Select the StatevectorSimulator from the Aer provider
            simulator = Aer.get_backend('statevector_simulator')	
		
            backend = Aer.get_backend('aer_simulator')
            result = execute(circuit_qiskit, simulator).result()
		
            transformed_state = result.get_statevector(circuit_qiskit)
        
        overlap = np.abs( np.asarray(transformed_state).conj().T @ State )
	
        print( 'Overlap integral with the initial state: ', overlap )        
        assert( np.abs(overlap - 1) < 1e-4 )
        
        
        

    def test_State_Preparation_BFGS(self):
        r"""
        This method is called by pytest. 
        Test for a 4 qubit state preparation using the BFGS optimizer 

        """

        self.State_Preparation_adaptive_base('BFGS', 0)	

    def test_State_Preparation_HS(self):
        r"""
        This method is called by pytest. 
        Test for a 4 qubit state preparation using the Hilbert Schmidt test

        """

        self.State_Preparation_adaptive_base('BFGS', 3)
