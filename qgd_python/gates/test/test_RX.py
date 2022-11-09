import numpy as np
import random


class Test_operations:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_RX_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """

        from qgd_python.gates.qgd_RX import qgd_RX

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # creating an instance of the C++ class
        RX = qgd_RX( qbit_num, target_qbit )

        parameters = np.array( [random.random(), random.random(), random.random()] )
        
        RX_matrix = RX.get_Matrix( parameters )
        
        print(RX_matrix)




