from scipy.stats import unitary_group
import numpy as np
from qgd_python.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base import qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver_Base
from qgd_python import utils as utils
import time
import sys
from qgd_python.gates.qgd_Circuit_Wrapper import qgd_Circuit_Wrapper as Circuit
import scipy as sp
import pytest


sigmax = sp.sparse.csr_matrix(np.array([[0,1],
                   [1,0]])+0.j)
sigmay = sp.sparse.csr_matrix(np.array([[0,0+-1j],
                   [0+1j,0]])+0.j)
sigmaz = sp.sparse.csr_matrix(np.array([[1,0],
                   [0,-1]])+0.j)


def generate_hamiltonian(topology, n):

    Hamiltonian = sp.sparse.coo_matrix((2**n, 2**n), dtype=np.complex128)
    for i in topology:
        if i[0]==0:
            lhs_1 = sp.sparse.eye(1,format='coo')
        else:
            lhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(i[0]-1):
            lhs_1 = sp.sparse.kron(lhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        if i[0]==n-1:
            rhs_1 = sp.sparse.eye(1,format='coo')
        else:
            rhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(n-i[0]-2):
	        rhs_1 = sp.sparse.kron(rhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        if i[1]==0:
            lhs_2 = sp.sparse.eye(1,format='coo')
        else:
            lhs_2 = sp.sparse.eye(2,format='coo')
        for k in range(i[1]-1):
	        lhs_2 = sp.sparse.kron(lhs_2,sp.sparse.eye(2,format='coo'),format='coo')
        if i[1]==n-1:
            rhs_2 = sp.sparse.eye(1,format='coo')
        else:
            rhs_2 = sp.sparse.eye(2,format='coo')
        for k in range(n-i[1]-2):
	        rhs_2 = sp.sparse.kron(rhs_2,sp.sparse.eye(2,format='coo'),format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmax,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax,format='coo'),rhs_2 ,format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmay,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmay,format='coo'),rhs_2 ,format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmaz,format='coo'),rhs_2 ,format='coo')
    for i in range(n):
    
        if i==0:
            lhs_1 = sp.sparse.eye(1,format='coo')
        else:
            lhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(i-1):
            lhs_1 = sp.sparse.kron(lhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        if i==n-1:
            rhs_1 = sp.sparse.eye(1,format='coo')
        else:
            rhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(n-i-2):
	        rhs_1 = sp.sparse.kron(rhs_1,sp.sparse.eye(2,format='coo'),format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')
        

    return Hamiltonian.tocsr()






class Test_VQE:

    def ctest_VQE_Identity(self):
        layers = 1
        blocks = 1

        qbit_num = 7
        Hamiltonian = sp.sparse.eye(2**qbit_num,format="csr")

        config = {"agent_lifetime":500,
                  "optimization_tolerance": -7.1,
                  "max_inner_iterations":10,
                  "max_iterations":50,
                  "learning_rate": 2e-1,
                  "agent_num":64,
                  "agent_exploration_rate":0.3,
                  "max_inner_iterations_adam":50000}
        
        # initiate the VQE object with the Hamiltonian
        VQE_eye = Variational_Quantum_Eigensolver_Base(Hamiltonian, qbit_num, config)

        # set the optimization engine
        VQE_eye.set_Optimizer("GRAD_DESCEND")

        # set the ansatz variant (U3 rotations and CNOT gates)
        VQE_eye.set_Ansatz("HEA")

        # generate the circuit ansatz for the optimization
        VQE_eye.Generate_Circuit(layers, blocks)

        # create initial parameters 
        param_num  = VQE_eye.get_Parameter_Num()
        parameters = np.zeros( (param_num,) )

        VQE_eye.set_Optimized_Parameters(parameters)

        # start an etap of the optimization (max_inner_iterations iterations)
        VQE_eye.Start_Optimization()
        
        # retrieve QISKIT format of the optimized circuit
        quantum_circuit = VQE_eye.get_Quantum_Circuit()

        # retrieve the optimized parameter
        parameters = VQE_eye.get_Optimized_Parameters()

        # evaluate the VQE energy at the optimized parameters
        Energy = VQE_eye.Optimization_Problem( parameters )


        assert (abs(Energy-1)<1e-4)
    
    def test_Heisenberg_XX(self):
    
        layers = 1
        blocks = 1

        qbit_num = 9
        topology = []
        for idx in range(qbit_num-1):
            topology.append( (idx, idx+1) )
            
        Hamiltonian = generate_hamiltonian(topology, qbit_num)
        
        config = {"agent_lifetime":10,
                  "max_inner_iterations":1000,
                  "max_iterations":1000,
                  "agent_num":3,
                  "agent_exploration_rate":0.5,
                  "max_inner_iterations_adam":500,
                  "convergence_length": 300}
        
        # initiate the VQE object with the Hamiltonian
        VQE_Heisenberg = Variational_Quantum_Eigensolver_Base(Hamiltonian, qbit_num, config)

        # set the optimization engine
        VQE_Heisenberg.set_Optimizer("AGENTS")

        # set the ansatz variant (U3 rotations and CNOT gates)
        VQE_Heisenberg.set_Ansatz("HEA_ZYZ")

        # generate the circuit ansatz for the optimization        
        VQE_Heisenberg.Generate_Circuit(layers, blocks)

        # create initial parameters 
        param_num  = VQE_Heisenberg.get_Parameter_Num()
        parameters = np.zeros( (param_num,) )

        VQE_Heisenberg.set_Optimized_Parameters(parameters)

        # start an etap of the optimization (max_inner_iterations iterations)
        VQE_Heisenberg.Start_Optimization()

        # retrieve QISKIT format of the optimized circuit
        quantum_circuit = VQE_Heisenberg.get_Quantum_Circuit()
        
        # print the quantum circuit
        lambdas, vecs = sp.sparse.linalg.eigs(Hamiltonian)

        # retrieve the optimized parameter
        parameters = VQE_Heisenberg.get_Optimized_Parameters()

        # evaluate the VQE energy at the optimized parameters
        Energy = VQE_Heisenberg.Optimization_Problem( parameters )
        
        print(lambdas[0])
        assert ((Energy - np.real(lambdas[0]))<1e-2)






