import numpy as np
import random

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

from qgd_python.utils import get_unitary_from_qiskit_circuit
from qgd_python.gates.qgd_CH import qgd_CH
import math
        

class Test_operations_squander:
    """This is a test class of the python iterface to the gates of the QGD package"""


    pi=np.pi

    # number of qubits
    qbit_num = 2

    # target qbit
    target_qbit = 0

    # control_qbit
    control_qbit = 1

    # creating an instance of the C++ class
    CH = qgd_CH( qbit_num, target_qbit, control_qbit )

    def test_CH_get_matrix(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of RX gate.
        """
	#SQUANDER

        # get the matrix     
        CH_squander = self.CH.get_Matrix(  )
        
        #print(CH_squander)

	#QISKIT

        # Create a Quantum Circuit acting on the q register
        circuit = QuantumCircuit(self.qbit_num)

        # Add the CH gate 
        circuit.ch(self.control_qbit, self.target_qbit)
        
        # the unitary matrix from the result object
        CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
        CH_qiskit = np.asarray(CH_qiskit)
        
        # Draw the circuit        
        #print(CH_qiskit)
        
        #the difference between the SQUANDER and the qiskit result        
        delta_matrix=CH_squander-CH_qiskit

        # compute norm of matrix
        error=np.linalg.norm(delta_matrix)

        print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        assert( error < 1e-3 )

    def test_CH_apply_to(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate and compare with qiskit.
        """


        test_matrix = np.array([[1., 0., 0., 1.], [0., 1., 1., 1.], [1., 0., 0., 1.], [0., 0., 1., 1.]])
        print("ťest_matrix: ")
        print(test_matrix) 

        # the CH gate 
        ch_gate=np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1./math.sqrt(2), 1./math.sqrt(2)], [0., 0., 1./math.sqrt(2),-1./math.sqrt(2)]])

        # apply the gate on the input array/matrix 
        CH_qiskit_apply_gate=np.matmul(test_matrix, ch_gate)

        print("qiskit appply_to: ")   
        print(CH_qiskit_apply_gate)

	#SQUANDER

        CH_squander = test_matrix
        # get the matrix          
        #CH_squander = self.CH.get_Matrix(  )

        # apply the gate on the input array/matrix                   
        #self.CH.apply_to(CH_squander )
        self.CH.apply_to(test_matrix)    
        print("squander apply_to: ")                
        print(CH_squander)             


	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        #circuit = QuantumCircuit(self.qbit_num)
      
        # Add the X gate on the target qbit
        #circuit.ch(self.control_qbit, self.target_qbit)

        # the unitary matrix from the result object
        #CH_qiskit = get_unitary_from_qiskit_circuit( circuit )
        #CH_qiskit = np.asarray(CH_qiskit)



        #print(np.around(CH_qiskit_apply_gate,1))

        #the difference between the SQUANDER and the qiskit result        
        #delta_matrix=CH_squander-CH_qiskit_apply_gate

        # compute norm of matrix
        #error=np.linalg.norm(delta_matrix)

        #print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        #assert( error < 1e-3 ) 

a=Test_operations_squander()
a.test_CH_get_matrix()
a.test_CH_apply_to()


#qbit_num legyen 5, vagy 6, ne 1. Illetve egy for ciklusban le kellene ellenőrizni az össze target qubitet 0-qbit_num-1 között.
#Ezrt nem kell az osztályban létrehozni egy külön X kaput atribútumként, ezt az összes test függvényben csináljuk külön.
#Illetve az apply_to teszt függvényben ne olvassuk ki a mátrixot a kapuból. X_squander = self.X.get_Matrix(  ) helyett megteszi egy sima egységmátrix is, amit a numpy generál:
#test_matrix = np.eye( matrix_size ), ahol matrix_size = int( pow(2,qbit_num) )


	#QISKIT      

        # Create a Quantum Circuit acting on the q register
        #circuit = QuantumCircuit(self.qbit_num)
      
        # Add the X gate on the target qbit
        #circuit.x( self.target_qbit )

        # the unitary matrix from the result object
        #X_qiskit = get_unitary_from_qiskit_circuit( circuit )
        #X_qiskit = np.asarray(X_qiskit)
        
        # the X gate 
        #x_gate=np.array([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]])
        #x_gate=np.array([[0., 1.], [1., 0.]])
        # apply the gate on the input array/matrix  
        #X_qiskit_apply_gate=np.matmul(x_gate, test_matrix)

        #print(X_qiskit_apply_gate)

        #the difference between the SQUANDER and the qiskit result        
        #delta_matrix=X_squander-X_qiskit_apply_gate

        # compute norm of matrix
        #error=np.linalg.norm(delta_matrix)

        #print("The difference between the SQUANDER and the qiskit result is: " , np.around(error,2))
        #assert( error < 1e-3 ) 


