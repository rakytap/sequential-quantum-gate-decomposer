# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:12:18 2020

@author: rakytap
"""

from .Operation import Operation
import numpy as np

##
# @brief A class responsible for constructing matrices of U3
# gates acting on the N-qubit space

class U3( Operation ):
    
    # Pauli x matrix
    pauli_x = np.array([[0,1],[1,0]])
    # Pauli y matrix
    pauli_y = np.array([[0,np.complex(0,-1)],[np.complex(0,1),0]])
    # Pauli z matrix
    pauli_z = np.array([[1,0],[0,-1]])

    ##
    # @brief Constructor of the class.
    # @param qbit_num The number of qubits in the unitaries
    # @return An instance of the class
    def __init__(self, qbit_num, target_qbit, Theta=False, Phi=False, Lambda=False):
        # number of qubits spanning the matrix of the operation
        self.qbit_num = qbit_num
        # A string describing the type of the operation
        self.type = 'u3'
        # A list of parameter names which are used to evaluate the matrix of the operation
        self.parameters = list()
        # The index of the qubit on which the operation acts (target_qbit >= 0) 
        self.target_qbit = target_qbit
        # The index of the qubit which acts as a control qubit (control_qbit >= 0) in controlled operations
        self.control_qbit = None     
        
        if Theta and Phi and Lambda:
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta_Phi_Lambda
            self.parameters = ["Theta", "Phi", "Lambda"]
            
        elif (not Theta) and Phi and Lambda:
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Phi_Lambda
            self.parameters = ["phi", "Lambda"]
            
        elif Theta and (not Phi) and Lambda:
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta_Lambda
            self.parameters = ["Theta", "Lambda"]
           
        elif Theta and Phi and (not Lambda)  :          
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta_Phi
            self.parameters = ["Theta", "Phi"]
            
        elif (not Theta) and (not Phi) and Lambda:
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Lambda
            self.parameters = ["Lambda"]
             
        elif (not Theta) and Phi and (not Lambda):
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Phi
            self.parameters = ["Phi"]
        
        elif Theta and (not Phi) and (not Lambda):
            # function handle to calculate the operation on the target qubit
            self.matrix = self.composite_u3_Theta
            self.parameters = ["Theta"]
            
        else:
            raise BaseException('Input error in the ceration of the operator U3')
        
        # set the function to calculate the matrix of the operation as a function of a given parameters
        self.matrix = self.composite_u3
        
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    def composite_u3_Theta_Phi_Lambda(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = parameters[1], Lambda = parameters[2] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Phi_Lambda(self, parameters ):
        return self.composite_u3( Theta = 0, Phi = parameters[0], Lambda = parameters[1] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Theta_Lambda(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = 0, Lambda = parameters[1] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Theta_Phi(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = parameters[1], Lambda = 0 )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Lambda(self, parameters ):
        return self.composite_u3( Theta = 0, Phi = 0, Lambda = parameters[0] )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Phi(self, parameters ):
        return self.composite_u3( Theta = 0, Phi = parameters[0], Lambda = 0 )
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param parameters Three component array conating the parameters in order Theta, Phi, Lambda
    # @return Returns with the matrix of the U3 gate.
    def composite_u3_Theta(self, parameters ):
        return self.composite_u3( Theta = parameters[0], Phi = 0, Lambda = 0 )
        
        
    ##    
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on the space of qbit_num qubits.
    # @param Theta Real parameter standing for the parameter theta.
    # @param Phi Real parameter standing for the parameter phi.
    # @param Lambda Real parameter standing for the parameter lambda.
    # @return Returns with the matrix of the U3 gate.
    def composite_u3(self, Theta, Phi, Lambda ):
        
        ret = None
        for idx in range(0, self.qbit_num):
            if idx == self.target_qbit:
                if ret is None:
                    ret = self.u3( Theta, Phi, Lambda )
                else:
                    ret = np.kron(self.u3(Theta, Phi, Lambda), ret)
            else:
                if ret is None:
                    ret = np.identity(2)
                else:
                    ret = np.kron(np.identity(2), ret)
                    
        return ret
    
    
    ##   
    # @brief Calculate the matrix of a U3 gate operation corresponding corresponding to the given parameters acting on a single qbit space.
    # @param Theta Real parameter standing for the parameter theta.
    # @param Phi Real parameter standing for the parameter phi.
    # @param Lambda Real parameter standing for the parameter lambda.
    # @return Returns with the matrix of the U3 gate.
    @staticmethod
    def u3(Theta=0, Phi=0, Lambda=0 ):
        
        return np.array([[np.cos(Theta/2), -np.exp(np.complex(0,Lambda))*np.sin(Theta/2)],
                        [np.exp(np.complex(0,Phi))*np.sin(Theta/2), np.exp(np.complex(0,Lambda+Phi))*np.cos(Theta/2)]])
                   