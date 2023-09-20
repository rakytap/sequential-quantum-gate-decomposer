'''
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

'''

import numpy as np
import time
import random






from qiskit import QuantumCircuit, transpile
import numpy as np


from qiskit import execute

from squander import utils
from squander import N_Qubit_Decomposition_custom



try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


##
# @brief ???????????
# @return ???????????
def pauli_exponent( alpha=0.6217*np.pi ):
	# creating Qiskit quantum circuit
	qc_orig = QuantumCircuit(5)
	qc_orig.h(1)
	qc_orig.cx(1,2)

	qc_orig.rx(np.pi/2,0)
	qc_orig.rx(np.pi/2,1)
	qc_orig.cx(2,4)
	qc_orig.cx(0,1)

	qc_orig.rx(np.pi/2,0)
	qc_orig.h(2)
	qc_orig.cx(0,2)


	qc_orig.rx(np.pi/2,0)
	qc_orig.h(3)
	qc_orig.rz(alpha,4)
	qc_orig.cx(0,3)

	qc_orig.h(0)
	qc_orig.rz(-alpha,1)
	qc_orig.cx(2,4)

	qc_orig.cx(2,1)
	qc_orig.rz(-alpha,4)
	qc_orig.cx(3,1)

	qc_orig.rz(alpha,1)
	qc_orig.cx(0,1)
	qc_orig.cx(3,1)
	qc_orig.cx(4,1)

	qc_orig.rz(-alpha,1)
	qc_orig.cx(2,1)

	qc_orig.rz(alpha,1)
	qc_orig.cx(3,1)
	qc_orig.cx(4,1)

	qc_orig.rz(alpha,1)
	qc_orig.cx(2,4)
	qc_orig.cx(0,1)

	qc_orig.h(0)
	qc_orig.cx(3,1)
	qc_orig.cx(0,3)

	qc_orig.rx(-np.pi/2,0)
	qc_orig.h(3)
	qc_orig.cx(0,2)

	qc_orig.rx(-np.pi/2,0)
	qc_orig.h(2)
	qc_orig.cx(0,1)

	qc_orig.rx(-np.pi/2,0)
	qc_orig.rx(-np.pi/2,1)
	qc_orig.cx(2,4)
	qc_orig.cx(1,2)
	qc_orig.h(1)

	return qc_orig



##
# @brief Calcuates the distance between two unitaries according to Eq.(3) of Ref. ....
# @param Umtx1 The first unitary
# @param Umtx2 The second unitary
# @return Returns with the calculated distance
def get_unitary_distance( Umtx1, Umtx2 ):

	product_matrix = np.dot(Umtx1, Umtx2.conj().T)
	phase = np.angle(product_matrix[0,0])
	product_matrix = product_matrix*np.exp(-1j*phase)
	product_matrix = np.eye(len(Umtx1))*2 - product_matrix - product_matrix.conj().T
	distance =  (np.real(np.trace(product_matrix)))/2

	return distance


##
# @brief ???????????
# @return ???????????
def get_optimized_circuit( alpha, optimizer='BFGS', optimized_parameters=None ):
	
	filename = 'qgd_python/decomposition/test/19CNOT.qasm'
	qc_trial = QuantumCircuit.from_qasm_file( filename )
	qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
	#print(qc_trial)

	##### getting the alpha dependent unitary
	qc_orig = pauli_exponent(alpha )
	qc_orig = transpile(qc_orig, optimization_level=3, basis_gates=['cx', 'u3'], layout_method='sabre')
	#print('global phase: ', qc_orig.global_phase)

	Umtx_orig = utils.get_unitary_from_qiskit_circuit( qc_orig )
        
	iteration_max = 10
	
	for jdx in range(iteration_max):
        
		cDecompose = N_Qubit_Decomposition_custom( Umtx_orig.conj().T )

		# setting the tolerance of the optimization process. The final error of the decomposition would scale with the square root of this value.
		cDecompose.set_Optimization_Tolerance( 1e-5 )

		# importing the quantum circuit
		cDecompose.import_Qiskit_Circuit(qc_trial)
		
		# set the number of successive identical blocks in the optimalization of disentanglement of the n-th qubits
		cDecompose.set_Optimization_Blocks( 200 )

                # set the optimizer
		cDecompose.set_Optimizer( optimizer )

		# set the initial parameters if given
		if not ( optimized_parameters is None ):
			cDecompose.set_Optimized_Parameters( optimized_parameters )

		# set verbosity
		cDecompose.set_Verbose( 4 )

		# starting the decomposition
		cDecompose.Start_Decomposition()
		
		# getting the new optimized parameters
		optimized_parameters_loc = cDecompose.get_Optimized_Parameters()

		qc_final = cDecompose.get_Quantum_Circuit()

		# get the unitary of the final circuit
		Umtx_recheck = utils.get_unitary_from_qiskit_circuit( qc_final )

		# get the decomposition error
		decomposition_error =  get_unitary_distance(Umtx_orig, Umtx_recheck)
		print('recheck decomposition error: ', decomposition_error)

		if decomposition_error < 1e-3:
			break
		
	            
	assert( decomposition_error < 1e-3 )
	if decomposition_error < 1e-3:
		return qc_final, optimized_parameters_loc
	else:
		return None, None
	



class Test_parametric_circuit:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""


    def test_optimizer(self):


        # determine random parameter value alpha
        alpha = 1.823631161607293

        # determine the quantum circuit at parameter value alpha with BFGS2 optimizer
        qc, optimized_parameters = get_optimized_circuit( alpha, optimizer='BFGS2' )

        # determine the quantum circuit at parameter value alpha with BFGS optimizer
        qc, optimized_parameters_tmp = get_optimized_circuit( alpha+0.005, optimizer='BFGS', optimized_parameters=optimized_parameters )

        # determine the quantum circuit at parameter value alpha with ADAM optimizer
        qc, optimized_parameters_tmp = get_optimized_circuit( alpha+0.005, optimizer='GRAD_DESCEND', optimized_parameters=optimized_parameters )

        # determine the quantum circuit at parameter value alpha with ADAM optimizer
        qc, optimized_parameters_tmp = get_optimized_circuit( alpha+0.005, optimizer='ADAM', optimized_parameters=optimized_parameters )





