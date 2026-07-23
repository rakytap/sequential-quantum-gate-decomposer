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
## \brief Functionality test cases for the N_Qubit_Decomposition class.



import numpy as np
from scipy.stats import unitary_group

from squander import utils


class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""

    def test_N_Qubit_Decomposition_QX2(self):
        r"""
        Test a custom QX2 gate structure in a four-qubit decomposition.

        Both the target and optimizer are seeded because convergence time for
        this strong numerical test otherwise varies by thousands of optimizer
        iterations.
        """

        from squander import N_Qubit_Decomposition

        qbit_num = 4
        matrix_size = 2**qbit_num

        # Keep both halves of the stochastic workload reproducible.
        Umtx = unitary_group.rvs(matrix_size, random_state=0)
        decomp = N_Qubit_Decomposition(
            Umtx.conj().T,
            config={
                "random_seed": 1,
                "max_outer_iterations": 600,
                # The explicit iteration cap bounds runtime; do not let the
                # looser stagnation heuristic pre-empt the 1e-7 tolerance.
                "convergence_threshold": 0.0,
            },
        )

        reordered_qbits = (2, 3, 1, 0)
        decomp.Reorder_Qubits(reordered_qbits)
        decomp.set_Gate_Structure(
            {
                4: self.create_custom_gate_structure_QX2(4),
                3: self.create_custom_gate_structure_QX2(3),
            }
        )
        decomp.set_Max_Layer_Num({4: 60, 3: 16})
        decomp.set_Optimization_Blocks(20)
        decomp.set_Optimization_Tolerance(1e-7)
        decomp.Start_Decomposition()

        assert decomp.get_Decomposition_Error() < 1e-7

        revert_qbits = (3, 2, 0, 1)
        decomp.Reorder_Qubits(revert_qbits)
        quantum_circuit = decomp.get_Qiskit_Circuit()

        decomposed_matrix = np.asarray(
            utils.get_unitary_from_qiskit_circuit(quantum_circuit)
        )
        product_matrix = Umtx @ decomposed_matrix.conj().T
        phase = np.angle(product_matrix[0, 0])
        product_matrix *= np.exp(-1j * phase)
        product_matrix = (
            np.eye(matrix_size) * 2
            - product_matrix
            - product_matrix.conj().T
        )
        decomposition_error = np.real(np.trace(product_matrix)) / 2

        assert decomposition_error < 1e-3

    def test_custom_gate_structure_QX2_couplings(self):
        """Validate every QX2 coupling independently of optimizer convergence."""

        gate_structure = self.create_custom_gate_structure_QX2(4)
        cnot_qbits = [
            tuple(gate.get_Involved_Qbits())
            for layer in gate_structure.get_Gates()
            for gate in layer.get_Gates()
            if gate.get_Name() == "CNOT"
        ]

        assert cnot_qbits == [(0, 3), (0, 1), (2, 3)]


    def create_custom_gate_structure_QX2(self, qbit_num):
        r"""
        This method is called to create custom gate structure for the decomposition on IBM QX2

        """

        from squander import Circuit

        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )

        disentangle_qbit = qbit_num - 1

        for qbit in range(0, disentangle_qbit ):

            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )

            if qbit == 0:

                # add U3 fate to the block  
                Layer.add_U3( 0 )                 
                Layer.add_U3( disentangle_qbit ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, disentangle_qbit)

            elif qbit == 1:

                # add U3 fate to the block     
                Layer.add_U3( 0 )                 
                Layer.add_U3( 1 ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, 1)



            elif qbit == 2:

                # add U3 fate to the block    
                Layer.add_U3( 2 )                 
                Layer.add_U3( disentangle_qbit ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 2, disentangle_qbit )

            Circuit_ret.add_Circuit( Layer )

        return Circuit_ret
