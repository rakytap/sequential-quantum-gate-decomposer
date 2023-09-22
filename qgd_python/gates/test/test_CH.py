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
from qgd_python.gates.qgd_CH import qgd_CH
import math
from scipy.stats import unitary_group        

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""

    def test_CH_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CH gate.
        """
        pi=np.pi

        for qbit_num in range(2,7):

            # target qbit
            target_qbit = qbit_num-2

            # control qbit
            control_qbit = qbit_num-1

            # creating an instance of the C++ class
            CH = qgd_CH( qbit_num, target_qbit, control_qbit )

	    #SQUANDER

            # get the matrix              
            CH_squander = CH.get_Matrix(  )

	    #QISKIT

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the CH gate on qubit pi, pi,
            circuit.ch(control_qbit, target_qbit)

            # the unitary matrix from the result object
            CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
            CH_qiskit = np.asarray(CH_qiskit)
        
            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=CH_squander-CH_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Get_matrix: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 )        
 
    def test_CH_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """
        pi=np.pi

        for qbit_num in range(2,7):

            # target qbit
            target_qbit = qbit_num-2

            # control qbit
            control_qbit = qbit_num-1

            # creating an instance of the C++ class
            CH = qgd_CH( qbit_num, target_qbit, control_qbit )

            #create text matrix 
            test_matrix= np.identity( 2**qbit_num, dtype=complex )    

	    #QISKIT      

            # Create a Quantum Circuit acting on the q register
            circuit = QuantumCircuit(qbit_num)

            # Add the CH gate on qubit pi, pi,
            circuit.ch(control_qbit, target_qbit)

            # the unitary matrix from the result object
            CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
            CH_qiskit = np.asarray(CH_qiskit)

            # apply the gate on the input array/matrix 
            #CH_qiskit_apply_gate=np.matmul(CH_qiskit, test_matrix)

	    #SQUANDER

            CH_squander=test_matrix

            # apply the gate on the input array/matrix                
            CH.apply_to(CH_squander )

            #the difference between the SQUANDER and the qiskit result        
            delta_matrix=CH_squander-CH_qiskit

            # compute norm of matrix
            error=np.linalg.norm(delta_matrix)

            #print("Apply_to: The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
            assert( error < 1e-3 ) 







