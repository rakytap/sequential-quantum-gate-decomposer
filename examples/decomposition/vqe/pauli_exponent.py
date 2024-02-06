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
'''

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
