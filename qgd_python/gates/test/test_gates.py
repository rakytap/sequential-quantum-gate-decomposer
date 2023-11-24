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


class Test_operations:
    """This is a test class of the python iterface to the gates of the QGD package"""

    def test_CNOT_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CNOT gate.

        """

        from qgd_python.gates.qgd_CNOT import qgd_CNOT


        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1        

        # creating an instance of the C++ class
        CNOT = qgd_CNOT( qbit_num, target_qbit, control_qbit )


    def test_CZ_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CZ gate.

        """

        from qgd_python.gates.qgd_CZ import qgd_CZ


        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1        

        # creating an instance of the C++ class
        CZ = qgd_CZ( qbit_num, target_qbit, control_qbit )



    def test_CH_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CH gate.

        """

        from qgd_python.gates.qgd_CH import qgd_CH


        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1        

        # creating an instance of the C++ class
        CH = qgd_CH( qbit_num, target_qbit, control_qbit )


    def test_U3_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_U3 import qgd_U3

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        

        # creating an instance of the C++ class
        U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )



    def test_RY_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_RY import qgd_RY

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0
    

        # creating an instance of the C++ class
        RY = qgd_RY( qbit_num, target_qbit )



    def test_RX_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_RX import qgd_RX

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0
    

        # creating an instance of the C++ class
        RX = qgd_RX( qbit_num, target_qbit )



    def test_RZ_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_RZ import qgd_RZ

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0
    

        # creating an instance of the C++ class
        RZ = qgd_RZ( qbit_num, target_qbit )


    def test_X_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_X import qgd_X

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0
    

        # creating an instance of the C++ class
        X = qgd_X( qbit_num, target_qbit )



    def test_SX_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_SX import qgd_SX

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0
    

        # creating an instance of the C++ class
        SX = qgd_SX( qbit_num, target_qbit )


    def test_SYC_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of CH gate.

        """

        from qgd_python.gates.qgd_SYC import qgd_SYC


        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 2

        # control_qbit
        control_qbit = 1       

        # creating an instance of the C++ class
        SYC = qgd_SYC( qbit_num, target_qbit, control_qbit )


    def test_Operation_Block_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of Operation_Block class.

        """

        from qgd_python.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        cCircuit = qgd_Circuit_Wrapper( qbit_num )



    def test_Operation_Block_add_operations(self):
        r"""
        This method is called by pytest. 
        Test to add operations to a block of gates

        """

        from qgd_python.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        cCircuit = qgd_Circuit_Wrapper( qbit_num )



        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        


        # add U3 gate to the block
        cCircuit.add_U3( target_qbit, Theta, Phi, Lambda )


        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1  

        # add CNOT gate to the block
        cCircuit.add_CNOT( target_qbit, control_qbit )





    def test_Operation_Block_add_block(self):
        r"""
        This method is called by pytest. 
        Test to add operations to a block of gates

        """

        from qgd_python.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        layer = qgd_Circuit_Wrapper( qbit_num )

        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        


        # add U3 gate to the block
        layer.add_U3( target_qbit, Theta, Phi, Lambda )



        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1  

        # add CNOT gate to the block
        layer.add_CNOT( control_qbit, target_qbit )


        # add RX gate to the block
        layer.add_RX( target_qbit )

        # creating an instance of the C++ class
        cCircuit = qgd_Circuit_Wrapper( qbit_num )   

        # add inner operation block to the outher operation block
        cCircuit.add_Circuit( layer )










      

