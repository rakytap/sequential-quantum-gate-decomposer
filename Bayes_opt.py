from scipy.stats import unitary_group
import numpy as np
from qgd_python.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base import qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver_Base
from qgd_python import utils as utils
import time
import sys
from qgd_python.gates.qgd_Gates_Block_Wrapper import qgd_Gates_Block_Wrapper as Gates_Block
import scipy as sp
import pytest
from bayes_opt import BayesianOptimization
np.random.seed(31415)
sigmax = sp.sparse.csr_array(np.array([[0,1],
                   [1,0]])+0.j)
sigmay = sp.sparse.csr_array(np.array([[0,0+-1j],
                   [0+1j,0]])+0.j)
sigmaz = sp.sparse.csr_array(np.array([[1,0],
                   [0,-1]])+0.j)
def generate_hamiltonian(n):
    topology = []
    edges = np.zeros(n)
    options = list(range(n))
    for qbit in range(n):
        if edges[qbit]!=3:
            options.pop(np.argwhere(np.array(options)==qbit)[0,0])
            if len(options) > 0:
                targets = np.random.choice(options,min(len(options),int(3-edges[qbit])),False)
                for i in targets:
                    edges[qbit]+=1
                    edges[i]+=1
                    topology.append((qbit,i))
                    if edges[i]==3:
                        options.pop(np.argwhere(np.array(options)==i)[0,0])
    #print(len(topology))
    Hamiltonian = sp.sparse.coo_array((2**n, 2**n), dtype=np.complex128)
    for i in topology:
        #print(i)
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
        Hamiltonian -= sp.sparse.kron(sp.sparse.kron(lhs_1,sigmax,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax,format='coo'),rhs_2 ,format='coo')
        Hamiltonian -= sp.sparse.kron(sp.sparse.kron(lhs_1,sigmay,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmay,format='coo'),rhs_2 ,format='coo')
        Hamiltonian -= sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmaz,format='coo'),rhs_2 ,format='coo')
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
        Hamiltonian -= sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz,format='coo'),rhs_1,format='coo')
        

    return Hamiltonian.tocsr()
layers = 1
blocks = 1
rot_layers = 1
n = 10
topology = []
for idx in range(n-1):
    topology.append( (idx, idx+1) )
Hamiltonian = generate_hamiltonian(n)
config = {"agent_lifetime":11,
"max_inner_iterations":39,
"max_iterations":1000,
"learning_rate": 2e-3,
"agent_num":3,
"agent_exploration_rate":0.5,
"max_inner_iterations_adam":500,
"convergence_length": 300}
"""
VQE_Heisenberg = Variational_Quantum_Eigensolver_Base(Hamiltonian, n, config)
VQE_Heisenberg.set_Ansatz("HEA")
VQE_Heisenberg.set_Optimization_Tolerance(-8.5)
VQE_Heisenberg.Generate_initial_circuit(layers, blocks, rot_layers)
variable1 = 1
variable2 = 0
parameters = VQE_Heisenberg.get_Optimized_Parameters()
def cost_func(x,y):
    parameters[0,variable1]=x
    parameters[0,variable2]=y
    return -1*VQE_Heisenberg.Optimization_Problem(parameters)

iters = 100
pbounds = {'x':(-np.pi,np.pi),'y':(-np.pi,np.pi)}
cost_funcs = []
for it in range(iters):
    optimizer = BayesianOptimization(
    f=cost_func,
    pbounds=pbounds,
    random_state=1,
    )
    optimizer.maximize(
    init_points=5,
    n_iter=5,
    )
    parameters[0,variable1] = optimizer.max['params']['x']
    parameters[0,variable2] = optimizer.max['params']['y']
    VQE_Heisenberg.set_Optimized_Parameters(parameters)
    variable1,variable2 = np.random.choice(range(parameters.shape[1]),2,replace=False)
    cost_funcs.append(VQE_Heisenberg.Optimization_Problem(parameters))
del VQE_Heisenberg"""
lambdas, vecs = sp.sparse.linalg.eigs(Hamiltonian)
#Energy = VQE_Heisenberg.Optimization_Problem(VQE_Heisenberg.get_Optimized_Parameters())
print(lambdas)
config = {"agent_lifetime":50,
"max_inner_iterations":30,
"max_inner_iterations_adam":1250,
"max_inner_iterations_grad_descend":75,
"max_iteration_loops_grad_descend":5,
"max_inner_iterations_bfgs":750,
"max_iteration_loops_bfgs":5,
"learning_rate": 2e-3,
"agent_num":64,
"agent_exploration_rate":0.2,
"convergence_length": 10}


VQE_Heisenberg = Variational_Quantum_Eigensolver_Base(Hamiltonian, n, config)
VQE_Heisenberg.set_Optimizer("BAYES_AGENTS")
VQE_Heisenberg.set_Ansatz("HEA")
VQE_Heisenberg.set_Optimization_Tolerance(-60)
VQE_Heisenberg.set_Project_Name("AGENTS")
VQE_Heisenberg.Generate_initial_circuit(layers, blocks, rot_layers)
parameters = np.random.randn(VQE_Heisenberg.get_Optimized_Parameters().shape[1])*2*np.pi
VQE_Heisenberg.set_Optimized_Parameters(parameters)
VQE_Heisenberg.get_Ground_State()

