from scipy.stats import unitary_group
import numpy as np
from qgd_python.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base import qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver_Base
from qgd_python import utils as utils
import time
import sys
from qgd_python.gates.qgd_Gates_Block_Wrapper import qgd_Gates_Block_Wrapper as Gates_Block
import scipy as sp
import pytest
sigmax = sp.sparse.csr_matrix(np.array([[0,1],
                   [1,0]])+0.j)
sigmay = sp.sparse.csr_matrix(np.array([[0,0+-1j],
                   [0+-1j,0]])+0.j)
sigmaz = sp.sparse.csr_matrix(np.array([[1,0],
                   [0,-1]])+0.j)
def generate_hamiltonian(topology, n):
    Hamiltonian = sp.sparse.coo_array((2**n, 2**n), dtype=np.complex128)
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
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmay,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax,format='coo'),rhs_2 ,format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax,format='coo'),rhs_2 ,format='coo')
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')
    return Hamiltonian.tocsr()

class Test_VQE:

    def test_VQE_Identity(self):
        layers = 1
        blocks = 1
        rot_layers = 1
        n = 10
        Hamiltonian = sp.sparse.eye(2**n,format="csr")
        config = {"agent_lifetime":500,
        "optimization_tolerance": -7.1,
        "max_inner_iterations":1000,
        "max_iterations":5000,
        "learning_rate": 2e-1,
        "agent_num":64,
        "agent_exploration_rate":0.3,
        "max_inner_iterations_adam":50000}
        VQE_eye = Variational_Quantum_Eigensolver_Base(Hamiltonian, n, config)
        VQE_eye.set_Optimizer("BFGS")
        VQE_eye.set_Ansatz("HEA")
        VQE_eye.Generate_initial_circuit(layers, blocks, rot_layers)
        VQE_eye.get_Ground_State()
        quantum_circuit = VQE_eye.get_Quantum_Circuit()
        # print the quantum circuit
        print(quantum_circuit)
        Energy = VQE_eye.Optimization_Problem(VQE_eye.get_Optimized_Parameters())
        assert (abs(Energy-1)<1e-4)
    
    def test_Heisenberg_XXX(self):
        layers = 1
        blocks = 1
        rot_layers = 1
        n = 6
        topology = []
        for idx in range(n-1):
            topology.append( (idx, idx+1) )
        Hamiltonian = generate_hamiltonian(topology,n)
        config = {"agent_lifetime":100,
        "max_inner_iterations":100,
        "max_iterations":100,
        "learning_rate": 2e-1,
        "agent_num":32,
        "agent_exploration_rate":0.5,
        "max_inner_iterations_adam":50000,
        "convergence_length": 300}
        VQE_Heisenberg = Variational_Quantum_Eigensolver_Base(Hamiltonian, n, config)
        VQE_Heisenberg.set_Optimizer("BFGS")
        VQE_Heisenberg.set_Ansatz("HEA")
        VQE_Heisenberg.Generate_initial_circuit(layers, blocks, rot_layers)
        VQE_Heisenberg.get_Ground_State()
        quantum_circuit = VQE_Heisenberg.get_Quantum_Circuit()
        # print the quantum circuit
        print(quantum_circuit)
        lambdas, vecs = sp.sparse.linalg.eigs(Hamiltonian)
        Energy = VQE_Heisenberg.Optimization_Problem(VQE_Heisenberg.get_Optimized_Parameters())
        print(np.real(np.min(lambdas)))
        assert (Energy < np.real(np.min(lambdas)))
