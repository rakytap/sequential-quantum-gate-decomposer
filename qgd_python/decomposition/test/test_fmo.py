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
import scipy.constants as cnst
import scipy.linalg as linalg

from squander import N_Qubit_Decomposition
from squander import N_Qubit_Decomposition_custom

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit import assemble,Aer
from qiskit import execute
from squander import utils

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""


    def create_custom_gate_structure_lin_3qubit(self, layer_num=2):
        """
        Add layers to disentangle the 3rd qubit from the others
        linear chain with IBM native operations

        """

        from squander import Circuit

        qbit_num = 3

        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )


        for idx in range(0,layer_num,2):

            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )

#Rz Ry Rz = Rz Sx Rz Sx^+ Rz
            Layer.add_U3(2, True, True, True)
            #Layer.add_RZ( 2 ) 
            #Layer.add_RY( 2 ) 
            ##Layer.add_SX( 2 ) 
            #Layer.add_RZ( 2 )
            ##Layer.add_X( 2 )
            ##Layer.add_SX( 2 ) 

            Layer.add_U3(1, True, True, True)
            #Layer.add_RZ( 1 ) 
            #Layer.add_RY( 1 ) 
            ##Layer.add_SX( 1 ) 
            #Layer.add_RZ( 1 )
            ##Layer.add_X( 1 )  
            ##Layer.add_SX( 1 )

            # add CNOT gate to the block
            Layer.add_CNOT( 1, 2)

            Circuit_ret.add_Circuit( Layer )


            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )

            Layer.add_U3(1, True, True, True)
            #Layer.add_RZ( 1 ) 
            #Layer.add_RY( 1 ) 
            ##Layer.add_SX( 1 ) 
            #Layer.add_RZ( 1 )
            ##Layer.add_X( 1 )  
            ##Layer.add_SX( 1 )

            Layer.add_U3(0, True, True, True)
            #Layer.add_RZ( 0 ) 
            #Layer.add_RY( 0 ) 
            ##Layer.add_SX( 0 ) 
            #Layer.add_RZ( 0 )
            ##Layer.add_X( 0 )   
            ##Layer.add_SX( 0 )

            # add CNOT gate to the block
            Layer.add_CNOT( 0, 1)

            Circuit_ret.add_Circuit( Layer )



        return Circuit_ret



    def create_custom_gate_structure_lin_2qubit(self, layer_num=1, Circuit_ret=None):
        """
        Add layers to disentangle the final 2 qubits
        linear chain with IBM native operations

        """

        from squander import Circuit


        qbit_num = 2

        if Circuit_ret == None:
            # creating an instance of the wrapper class Circuit
            Circuit_ret = Circuit( qbit_num )


        for idx in range(layer_num):

            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )

            Layer.add_U3(1, True, True, True)
            #Layer.add_RZ( 1 ) 
            #Layer.add_RY( 1 ) 
            ##Layer.add_SX( 1 ) 
            #Layer.add_RZ( 1 )
            ##Layer.add_X( 1 )  
            ##Layer.add_SX( 1 )

            Layer.add_U3(0, True, True, True)
            #Layer.add_RZ( 0 ) 
            #Layer.add_RY( 0 ) 
            ##Layer.add_SX( 0 ) 
            #Layer.add_RZ( 0 )
            ##Layer.add_X( 0 )   
            ##Layer.add_SX( 0 )

            # add CNOT gate to the block
            Layer.add_CNOT( 0, 1)

            Circuit_ret.add_Circuit( Layer )



        return Circuit_ret


    def create_custom_gate_structure_finalyzing(self, qbit_num, Circuit_ret=None):
        """
        Rotating qubits into state |0>

        """

        from squander import Circuit



        if Circuit_ret == None:
            # creating an instance of the wrapper class Circuit
            Circuit_ret = Circuit( qbit_num )


        for idx in range(qbit_num):

            # creating an instance of the wrapper class Circuit
            Layer = Circuit( qbit_num )

            Layer.add_U3(idx, True, True, True)
            #Layer.add_RZ( idx ) 
            #Layer.add_RY( idx ) 
            ##Layer.add_SX( idx ) 
            #Layer.add_RZ_P( idx )
            ##Layer.add_X( idx )  
            ##Layer.add_SX( idx )

            Circuit_ret.add_Circuit( Layer )



        return Circuit_ret




    def get_Unitary(self, H): # H is the Hamiltonian to be exponentiated, t is the timestamp
        X = -(2j)*cnst.pi*H #the Hamiltonian is in 1/cm
        U = linalg.expm(X)
        U = np.hstack((U,np.zeros((7,1))))
        U = np.vstack((U,np.zeros((1,8))))
        U[-1][-1] = 1
        #print( np.dot(U, U.conj().T))
        return U



    def test_N_Qubit_Decomposition_3qubit(self):

        # the number of qubits spanning the unitary
        qbit_num = 3

        # determine the soze of the unitary to be decomposed
        matrix_size = int(2**qbit_num)
   
        # creating my unitary to be decomposed
        # first is the hamiltonian
        H = np.array(
        [[240, -87.7,  5.5, -5.9,   6.7,  -13.7, -9.9],
        [-87.7, 315,   30.8, 8.2,   0.7,   11.8,  4.3],
        [ 5.5,  30.8,  0,   -53.5, -2.2,  -9.6,   6.0],
        [-5.9,  8.2,  -53.5, 130,  -70.7, -17.0, -63.3],
        [ 6.7,  0.7,  -2.2, -70.7,  285,   81.1, -1.3],
        [-13.7, 11.8, -9.6, -17.0,  81.1,  435,   39.7],
        [-9.9,  4.3,   6.0, -63.3, -1.3,   39.7,  245]])


        Umtx = self.get_Unitary(H)
        
        # create custom gate structure for the decomposition
        gate_structure = self.create_custom_gate_structure_lin_3qubit(14)
        gate_structure = self.create_custom_gate_structure_lin_2qubit(3, gate_structure)
        gate_structure = self.create_custom_gate_structure_finalyzing(qbit_num, gate_structure)



        #Python map containing hyper-parameters
        config = { 'max_outer_iterations': 1,  
                'max_inner_iterations' : 500,	
                'optimization_tolerance': 1e-8,
                'agent_num': 20}



        # creating an instance of the C++ class
        decomp = N_Qubit_Decomposition_custom( Umtx.conj().T, initial_guess="zeros", config=config )

        # adding custom gate structure to the decomposition
        decomp.set_Gate_Structure( gate_structure )


        # setting the tolerance of the optimization process. The final error of the decomposition would scale with the square root of this value.
        decomp.set_Optimization_Tolerance( 1e-6 )

        # set the number of block to be optimized in one shot
        decomp.set_Optimization_Blocks( 50 )

        decomp.set_Cost_Function_Variant( 3 )

        decomp.set_Optimizer("BFGS")

        # starting the decomposition
        decomp.Start_Decomposition(prepare_export=True)




        # list the decomposing operations
        #decomp.List_Gates()
	
	
        # get the decomposing operations
        quantum_circuit = decomp.get_Quantum_Circuit()
        #print(quantum_circuit )
	
        # test the decomposition of the matrix
        # the unitary matrix from the result object
        decomposed_matrix = np.asarray( utils.get_unitary_from_qiskit_circuit( quantum_circuit ) )
        product_matrix = np.dot(Umtx,decomposed_matrix.conj().T)
        phase = np.angle(product_matrix[0,0])
        product_matrix = product_matrix*np.exp(-1j*phase)
		
        product_matrix = np.eye(matrix_size)*2 - product_matrix - product_matrix.conj().T
        # the error of the decomposition
        decomposition_error =  (np.real(np.trace(product_matrix)))/2
	   
        print('The error of the decomposition is ' + str(decomposition_error))
	




















##END OF CODE
