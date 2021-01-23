


class Test_Example:
    """This is an example class to demonstrate how to interface with a C++ part of the piquasso project."""

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


    def test_Operation_Block_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of Operation_Block class.

        """

        from qgd_python.gates.qgd_Operation_Block import qgd_Operation_Block        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        Operation_Block = qgd_Operation_Block( qbit_num )



    def test_Operation_Block_add_operations(self):
        r"""
        This method is called by pytest. 
        Test to add operations to a block of gates

        """

        from qgd_python.gates.qgd_Operation_Block import qgd_Operation_Block        

        # number of qubits
        qbit_num = 3
     

        # creating an instance of the C++ class
        Operation_Block = qgd_Operation_Block( qbit_num )



        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        


        # add U3 gate to the block
        Operation_Block.add_U3_To_End( target_qbit, Theta, Phi, Lambda )


        # target qbit
        target_qbit = 0

        # control_qbit
        control_qbit = 1  

        # add CNOT gate to the block
        Operation_Block.add_CNOT_To_End( target_qbit, control_qbit )
      

