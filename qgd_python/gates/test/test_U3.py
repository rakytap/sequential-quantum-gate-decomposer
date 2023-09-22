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
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_U3 import qgd_U3
import math
from scipy.stats import unitary_group

class Test_operations_squander:
    """This is a test class of the python iterface to compare the SQUANDER and the qiskit decomposition"""



    def test_U3_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """
        pi=np.pi

        # # set the free parameters
        Theta = True
        Phi = True
        Lambda = True

        parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )

	    #SQUANDER

            # get the matrix              
            U3_squander = U3.get_Matrix( parameters )

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the RX gate on qubit pi, pi,
            circuit.u(parameters[0]*2, parameters[1], parameters[2], target_qbit)

            # the unitary matrix from the result object
            U3_qiskit= get_unitary_from_qiskit_circuit( circuit )
            U3_qiskit = np.asarray(U3_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=U3_squander-U3_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_U3_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        pi=np.pi


        # # set the free parameters
        Theta = True
        Phi = True
        Lambda = True

        parameters = np.array( [pi/2*0.32, pi*1.2, pi/2*0.89] )

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )

            #create text matrix 
            test_matrix= np.identity( 2**qbit_num, dtype=complex )

	    #QISKIT      

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the RX gate on qubit pi, pi,
            circuit.u(parameters[0]*2, parameters[1], parameters[2], target_qbit)

            # the unitary matrix from the result object
            U3_qiskit= get_unitary_from_qiskit_circuit( circuit )
            U3_qiskit = np.asarray(U3_qiskit)

            # apply the gate on the input array/matrix 
            #U3_qiskit_apply_gate=np.matmul(U3_qiskit, test_matrix)

	    #SQUANDER

            U3_squander=test_matrix

            # apply the gate on the input array/matrix                
            U3.apply_to(parameters, U3_squander )        

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=U3_squander-U3_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )







