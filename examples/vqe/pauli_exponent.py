from qiskit import QuantumCircuit
import numpy as np

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
