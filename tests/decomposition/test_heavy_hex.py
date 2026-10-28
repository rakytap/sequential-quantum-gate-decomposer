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
## \file test_heavy_hex.py
## \brief Functionality test cases for the N_Qubit_Decomposition class.



# cerate unitary q-bit matrix
import os

import pytest
from qiskit import QuantumCircuit
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

    def test_N_Qubit_Decomposition_Heavy_Hex(self):
        """Exercise every Heavy-Hex coupling with a bounded entangling target."""
        from squander import N_Qubit_Decomposition

        qbit_num = 4
        target = QuantumCircuit(qbit_num)
        for qbit in range(qbit_num):
            target.u(
                0.2 + 0.13 * qbit,
                -0.3 + 0.07 * qbit,
                0.4 - 0.09 * qbit,
                qbit,
            )
        target.cx(0, 3)
        target.rz(0.31, 3)
        target.cx(0, 1)
        target.ry(-0.27, 1)
        target.cx(0, 2)
        Umtx = np.asarray(utils.get_unitary_from_qiskit_circuit(target))

        decomp = N_Qubit_Decomposition(
            Umtx.conj().T,
            config={"random_seed": 1, "max_outer_iterations": 150},
        )
        decomp.set_Gate_Structure(
            {
                4: self.create_custom_gate_structure_heavy_hex_4(4),
                3: self.create_custom_gate_structure_heavy_hex_3(3),
            }
        )
        decomp.set_Max_Layer_Num({4: 6, 3: 6})
        decomp.set_Optimization_Blocks(20)
        decomp.set_Optimization_Tolerance(1e-5)
        decomp.Start_Decomposition()

        decomposed = np.asarray(
            utils.get_unitary_from_qiskit_circuit(decomp.get_Qiskit_Circuit())
        )
        product = Umtx @ decomposed.conj().T
        product *= np.exp(-1j * np.angle(product[0, 0]))
        error = np.real(np.trace(2 * np.eye(1 << qbit_num) - product - product.conj().T)) / 2
        assert error < 1e-3

    @pytest.mark.skipif(
        os.environ.get("SQUANDER_RUN_HAAR_STRESS_TESTS") != "1",
        reason="Set SQUANDER_RUN_HAAR_STRESS_TESTS=1 to run the slow Haar stress case",
    )
    def test_N_Qubit_Decomposition_Heavy_Hex_Haar_stress(self):
        r"""
        This method is called by pytest. 
        Test to define custom gate structure in the decomposition

        """

        from squander import N_Qubit_Decomposition

        # the number of qubits spanning the unitary
        qbit_num = 4

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating a random unitary to be decomposed
        # This is a numerical functionality test, not a stochastic optimizer
        # benchmark.  An unseeded Haar target occasionally needs thousands of
        # outer iterations and can consume the entire CI job timeout.
        Umtx = unitary_group.rvs(matrix_size, random_state=0)
    
        # creating an instance of the C++ class
        decomp = N_Qubit_Decomposition(
            Umtx.conj().T,
            config={
                "random_seed": 1,
                "max_outer_iterations": 150,
            },
        )


        # create custom gate structure
        gate_structure = { 4: self.create_custom_gate_structure_heavy_hex_4(4), 3: self.create_custom_gate_structure_heavy_hex_3(3)}        


        # adding custom gate structure to the decomposition
        decomp.set_Gate_Structure( gate_structure )


        # set the maximal number of layers in the decomposition
        decomp.set_Max_Layer_Num( {4: 60, 3:16} )

        # set the number of block to be optimized in one shot
        decomp.set_Optimization_Blocks( 20 )

        # The assertion below is 1e-3; targeting one order of magnitude below
        # it avoids spending minutes polishing toward the library's much
        # tighter general-purpose default.
        decomp.set_Optimization_Tolerance(1e-5)

        # starting the decomposition
        decomp.Start_Decomposition()   


        # get the decomposing operations
        quantum_circuit = decomp.get_Qiskit_Circuit()

        import numpy.linalg as LA
    
    
        # the unitary matrix from the result object
        decomposed_matrix = np.asarray( utils.get_unitary_from_qiskit_circuit( quantum_circuit ) )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
    
        product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error =  (np.real(np.trace(product_matrix)))/2
       
        print('The error of the decomposition is ' + str(decomposition_error))

        assert( decomposition_error < 1e-3 )




    def create_custom_gate_structure_heavy_hex_3(self, qbit_num):
        r"""
        This method is called to create custom gate structure for the decomposition on IBM QX2

        """

        from squander import Circuit

        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )

        disentangle_qubit = qbit_num - 1 

        

        for qbit in range(0, disentangle_qubit ):

            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )


            if qbit == 0:

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Layer.add_U3( 0 )                 
                Layer.add_U3( disentangle_qubit ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, disentangle_qubit)

            elif qbit == 1:

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Layer.add_U3( 0 )                 
                Layer.add_U3( 1 ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, 1)



            elif qbit == 2:

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Layer.add_U3( 2 )                 
                Layer.add_U3( disentangle_qubit ) 

                # add CNOT gate to the block
                Layer.add_CNOT( disentangle_qubit, 2 )

         

            Circuit_ret.add_Circuit( Layer )

        return Circuit_ret





    def create_custom_gate_structure_heavy_hex_4(self, qbit_num):
        r"""
        This method is called to create custom gate structure for the decomposition on IBM QX2

        """

        from squander import Circuit

        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )

        disentangle_qubit = qbit_num - 1 

        

        for qbit in range(0, disentangle_qubit ):

            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )


            if qbit == 0:

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Layer.add_U3( 0 )                 
                Layer.add_U3( disentangle_qubit ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, disentangle_qubit)

            elif qbit == 1:

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Layer.add_U3( 0 )                 
                Layer.add_U3( 1 ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, 1)



            elif qbit == 2:

                # add U3 fate to the block
                Theta = True
                Phi = False
                Lambda = True      
                Layer.add_U3( 2 )                 
                Layer.add_U3( 0 ) 

                # add CNOT gate to the block
                Layer.add_CNOT( 0, 2)

         

            Circuit_ret.add_Circuit( Layer )

        return Circuit_ret
    
