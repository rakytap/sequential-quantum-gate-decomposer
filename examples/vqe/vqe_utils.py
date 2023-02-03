from qiskit import QuantumCircuit, transpile
import numpy as np

from pauli_exponent import pauli_exponent

from qiskit import execute
from qiskit import Aer

from qgd_python.decomposition.qgd_N_Qubit_Decomposition_custom import qgd_N_Qubit_Decomposition_custom
from qgd_python.utils import get_unitary_from_qiskit_circuit

from scipy.interpolate import interp1d



## Qiskit backend for simulator
backend = Aer.get_backend('unitary_simulator')


# 
optimized_parameters_mtx = None
alpha_vec = None


##
# @brief Call load precosntructed data for inerpolated optimization
# @param filename1 The filename containing the preconstructed optimizated parameters
# @param filename2 The filename containing the parameter values
def load_preconstructed_data( filename1, filename2 ):
	global alpha_vec
	global optimized_parameters_mtx

	optimized_parameters_mtx = np.loadtxt(filename1)
	alpha_vec = np.loadtxt(filename2)




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
def get_optimized_circuit( alpha, optimized_parameters_in=None ):
	
	filename = '19CNOT.qasm'
	qc_trial = QuantumCircuit.from_qasm_file( filename )
	qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
	#print(qc_trial)

	##### getting the alpha dependent unitary
	qc_orig = pauli_exponent(alpha )
	qc_orig = transpile(qc_orig, optimization_level=3, basis_gates=['cx', 'u3'], layout_method='sabre')
	#print('global phase: ', qc_orig.global_phase)

	Umtx_orig = get_unitary_from_qiskit_circuit( qc_orig )

	iteration_max = 10
	for jdx in range(iteration_max):

		cDecompose = qgd_N_Qubit_Decomposition_custom( Umtx_orig.conj().T )

		# setting the tolerance of the optimization process. The final error of the decomposition would scale with the square root of this value.
		cDecompose.set_Optimization_Tolerance( 1e-8 )

		# importing the quantum circuit
		cDecompose.import_Qiskit_Circuit(qc_trial)

		#if isinstance(optimized_parameters_in, (np.ndarray, np.generic) ) :
		#	cDecompose.set_Optimized_Parameters( optimized_parameters_in )

		# set the number of successive identical blocks in the optimalization of disentanglement of the n-th qubits
		cDecompose.set_Optimization_Blocks( 200 )

		# turning off verbosity
		cDecompose.set_Verbose( False )

		# starting the decomposition
		cDecompose.Start_Decomposition()
		
		# getting the new optimized parameters
		optimized_parameters_loc = cDecompose.get_Optimized_Parameters()

		qc_final = cDecompose.get_Quantum_Circuit()

		# get the unitary of the final circuit
		Umtx_recheck = get_unitary_from_qiskit_circuit( qc_final )

		# get the decomposition error
		decomposition_error =  get_unitary_distance(Umtx_orig, Umtx_recheck)
		print('recheck decomposition error: ',decomposition_error)

		if decomposition_error < 1e-6:
			break

	if decomposition_error < 1e-6:
		return qc_final, optimized_parameters_loc
	else:
		return None, None


##
# @brief ???????????
# @return ???????????
def get_interpolated_circuit( alpha ):

	# interpolate the optimized parameters to obtain the best initial set of parameters to be furthe roptimized
	itp = interp1d(alpha_vec, optimized_parameters_mtx, axis=0, kind='linear')
	initial_optimized_parameters = itp(alpha)

	qc = get_optimized_circuit( alpha, initial_optimized_parameters )
	return qc

