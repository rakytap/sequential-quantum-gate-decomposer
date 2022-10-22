import numpy as np
import random


class Test_operations:
    """This is a test class of the python iterface to the gates of the QGD package"""



    def test_U3(self):
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
        
        parameters = np.array( [random.random(), random.random(), random.random()] )
        
        U3_matrix = U3.get_Matrix( parameters )
        
        print(U3_matrix)



      

