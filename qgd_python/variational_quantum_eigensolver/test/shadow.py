from scipy.stats import unitary_group
import numpy as np
from qgd_python.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base import qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver_Base
from qgd_python import utils as utils
import time
import sys
from qgd_python.gates.qgd_Gates_Block import qgd_Gates_Block as Gates_Block
import scipy as sp
n = 8
l = 1
def create_circuit(n,l,b,topology):
    Gates_Block_ret = Gates_Block( n )
    #Create initial rotation gates
    rot_layer = Gates_Block( n )
    for idx in range(l):
        for qbits in topology:
            for bdx in range(b):
                rot_layer.add_CNOT(qbits[0],qbits[1])
        for qbit in range(n):
            rot_layer.add_RZ(qbit)
            rot_layer.add_RY(qbit)
            #rot_layer.add_RZ(qbit)
    Gates_Block_ret.add_Gates_Block( rot_layer )
    return Gates_Block_ret
topology = []
for idx in range(n-1):
    topology.append( (idx, idx+1) )
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
            lhs_1 = np.array([[1]])
        else:
            lhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(i[0]-1):
            lhs_1 = sp.sparse.kron(lhs_1,sp.sparse.eye(2,format='coo'))
        if i[0]==n-1:
            rhs_1 = np.array([[1]])
        else:
            rhs_1 = sp.sparse.eye(2,format='coo')
        for k in range(n-i[0]-2):
	        rhs_1 = sp.sparse.kron(rhs_1,sp.sparse.eye(2,format='coo'))
        if i[1]==0:
            lhs_2 = np.array([[1]])
        else:
            lhs_2 = sp.sparse.eye(2,format='coo')
        for k in range(i[1]-1):
	        lhs_2 = sp.sparse.kron(lhs_2,sp.sparse.eye(2,format='coo'))
        if i[1]==n-1:
            rhs_2 = np.array([[1]])
        else:
            rhs_2 = sp.sparse.eye(2,format='coo')
        for k in range(n-i[1]-2):
	        rhs_2 = sp.sparse.kron(rhs_2,sp.sparse.eye(2))
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmax),rhs_1)@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax),rhs_2 )
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmay),rhs_1)@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax),rhs_2 )
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz),rhs_1)@sp.sparse.kron(sp.sparse.kron(lhs_2 ,sigmax),rhs_2 )
        Hamiltonian += -0.5*sp.sparse.kron(sp.sparse.kron(lhs_1,sigmaz),rhs_1)
    return Hamiltonian.tocsr()
gate_structure = create_circuit(n,l,1,topology)
gate_structure.export_gate_list_to_binary(np.zeros(n*3*l),"squander.binary")
State = (np.zeros((2**n,1))).astype(np.complex128)
State[0] = 1+0j
parameters = np.random.randn(n*3)*2*np.pi
gate_structure.apply_to(parameters,State)
config = {"agent_lifetime":100,
        "optimization_tolerance": -7.1,
        "max_inner_iterations":10000,
        "max_iterations":500,
        "learning_rate": 2e-1,
        "agent_num":64,
        "agent_exploration_rate":0.3,
        "max_inner_iterations_adam":50000}
Hamiltonian = generate_hamiltonian(topology,n)#sp.sparse.eye(2**n,format="csr")
cDecompose = Variational_Quantum_Eigensolver_Base(Hamiltonian, n, config)
cDecompose.set_Gate_Structure_from_Binary("squander.binary")
cDecompose.set_Optimizer("BFGS")
cDecompose.get_Ground_State()
# get the decomposing operations
quantum_circuit = cDecompose.get_Quantum_Circuit()
# print the quantum circuit
print(quantum_circuit)
print(cDecompose.get_Optimized_Parameters())



