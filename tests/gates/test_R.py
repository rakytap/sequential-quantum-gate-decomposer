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

from squander.utils import get_unitary_from_qiskit_circuit
from squander.gates.qgd_R import qgd_R
import math
from scipy.stats import unitary_group

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_R_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        pi=np.pi

        # parameters
        parameters = np.array( [pi/2*0.32,np.pi/2*0.32] )

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            R = qgd_R( qbit_num, target_qbit )

	    #SQUANDER

            # get the matrix              
            R_squander = R.get_Matrix( parameters )

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the RX gate on qubit pi, pi,
            circuit.r(parameters[0]*2,parameters[1], target_qbit)

            # the unitary matrix from the result object
            R_qiskit = get_unitary_from_qiskit_circuit( circuit )
            R_qiskit = np.asarray(R_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=R_squander-R_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_R_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """

        pi=np.pi

        # parameters
        parameters = np.array( [pi/2*0.32,np.pi/2*0.32] )

        for qbit_num in range(1,7):

            # target qbit
            target_qbit = qbit_num-1

            # creating an instance of the C++ class
            R = qgd_R( qbit_num, target_qbit )

            #create text matrix 
            test_matrix= np.identity( 2**qbit_num, dtype=complex )

	    #QISKIT      

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the RX gate on qubit pi, pi,
            circuit.r(parameters[0]*2,parameters[1], target_qbit)

            # the unitary matrix from the result object
            R_qiskit = get_unitary_from_qiskit_circuit( circuit )
            R_qiskit = np.asarray(R_qiskit)

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # apply the gate on the input array/matrix 
            #RX_qiskit_apply_gate=np.matmul(RX_qiskit, test_matrix)

	    #SQUANDER

            R_squander=test_matrix

            # apply the gate on the input array/matrix                
            R.apply_to(parameters, R_squander )

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=R_squander-R_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 
