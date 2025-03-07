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

from qiskit import QuantumCircuit
from squander import utils
from squander import Qiskit_IO

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


class Test_Decomposition:
    """This is a test class of the python iterface to the decompsition classes of the QGD package"""


    def create_flat_circuit(self, qbit_num, layer_num=2):
        """
        Add layers to disentangle the 3rd qubit from the others
        linear chain with IBM native operations

        """

        from squander import Circuit


        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )


        for idx in range(0,layer_num,2):


            for qbit_idx in range(qbit_num-1):
#Rz Ry Rz = Rz Sx Rz Sx^+ Rz
                #Layer.add_U3(2, True, True, True)
                Circuit_ret.add_RZ( qbit_idx+1 ) 
                Circuit_ret.add_SX( qbit_idx+1 ) 
                Circuit_ret.add_RZ( qbit_idx+1 )
                Circuit_ret.add_X( qbit_idx+1 )
                Circuit_ret.add_SX( qbit_idx+1 ) 
                Circuit_ret.add_RZ( qbit_idx+1 )

                #Layer.add_U3(1, True, True, True)
                Circuit_ret.add_RZ( qbit_idx ) 
                #Layer.add_RY( 1 ) 
                Circuit_ret.add_SX( qbit_idx ) 
                Circuit_ret.add_RZ( qbit_idx )
                Circuit_ret.add_X( qbit_idx )  
                Circuit_ret.add_SX( qbit_idx )
                Circuit_ret.add_RZ( qbit_idx )

                # add CNOT gate to the block
                Circuit_ret.add_CNOT( qbit_idx, qbit_idx+1)




        return Circuit_ret



    def create_circuit(self, qbit_num, layer_num=2):
        """
        Add layers to disentangle the 3rd qubit from the others
        linear chain with IBM native operations

        """

        from squander import Circuit


        # creating an instance of the wrapper class Circuit
        Circuit_ret = Circuit( qbit_num )


        for idx in range(0,layer_num,2):


            for qbit_idx in range(qbit_num-1):
#Rz Ry Rz = Rz Sx Rz Sx^+ Rz
                Layer = Circuit( qbit_num )

                Layer.add_RZ( qbit_idx+1 ) 
                Layer.add_SX( qbit_idx+1 ) 
                Layer.add_RZ( qbit_idx+1 )
                Layer.add_X( qbit_idx+1 )
                Layer.add_SX( qbit_idx+1 ) 
                Layer.add_RZ( qbit_idx+1 )


                Layer.add_RZ( qbit_idx ) 
                #Layer.add_RY( 1 ) 
                Layer.add_SX( qbit_idx ) 
                Layer.add_RZ( qbit_idx )
                Layer.add_X( qbit_idx )  
                Layer.add_SX( qbit_idx )
                Layer.add_RZ( qbit_idx )

                # add CNOT gate to the block
                Layer.add_CNOT( qbit_idx, qbit_idx+1)

                Circuit_ret.add_Circuit( Layer )






        return Circuit_ret






    def test_dependency_graph(self):

        # the number of qubits spanning the unitary
        qbit_num = 4

        
        # create custom gate structure for the decomposition
        squander_circuit = self.create_flat_circuit( qbit_num, 2 )


        # cerate random parameter array for the circuit
        parameter_num = squander_circuit.get_Parameter_Num()
        parameters = np.random.randn( parameter_num ) *2*np.pi
        
        Qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit( squander_circuit, parameters )

        print( ' ' )
        print( Qiskit_circuit )

        flat_circuit = squander_circuit.get_Flat_Circuit()

        #assert( decomposition_error < 1e-3 )  





    def test_flat_conversion(self):

        # the number of qubits spanning the unitary
        qbit_num = 4

        
        # create custom gate structure for the decomposition
        squander_circuit = self.create_circuit( qbit_num, 2 )

        flat_circuit = squander_circuit.get_Flat_Circuit()


        # cerate random parameter array for the circuit
        parameter_num = flat_circuit.get_Parameter_Num()
        parameters = np.random.randn( parameter_num ) *2*np.pi

        
        Qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit( flat_circuit, parameters )

        print( ' ' )
        print( Qiskit_circuit )


        #assert( decomposition_error < 1e-3 )  




    def test_parents(self):

        # the number of qubits spanning the unitary
        qbit_num = 4

        
        # create custom gate structure for the decomposition
        squander_circuit = self.create_flat_circuit( qbit_num, 2 )


        gates = squander_circuit.get_Gates()


        chosen_gate_idx = 18
        chosen_gate = gates[ chosen_gate_idx ]


        print( "The chosen gate is: " + str(chosen_gate) )

        parents_indices = squander_circuit.get_Parents( chosen_gate )
        parent_gate = gates[ parents_indices[0] ]
   

        print("The parents gate is: " + str( parent_gate )  )



    def test_children(self):

        # the number of qubits spanning the unitary
        qbit_num = 4

        
        # create custom gate structure for the decomposition
        squander_circuit = self.create_flat_circuit( qbit_num, 2 )


        gates = squander_circuit.get_Gates()


        chosen_gate_idx = 18
        chosen_gate = gates[ chosen_gate_idx ]


        print( "The chosen gate is: " + str(chosen_gate) )

        children_indices = squander_circuit.get_Children( chosen_gate )
        child_gate = gates[ children_indices[0] ]
   

        print("The child gate is: " + str( child_gate )  )


















##END OF CODE
