from scipy.stats import unitary_group
import numpy as np
from squander import Variational_Quantum_Eigensolver
from squander import utils as utils
import time
import sys
import scipy as sp
import pytest
from networkx.generators.random_graphs import random_regular_graph
from qiskit.quantum_info import SparsePauliOp

np.set_printoptions(linewidth=200) 

import pickle



topology = []

def generate_hamiltonian_tmp(n):

    topology = [[0,1],[0,4],[0,8],
                [1,2],[1,5],
                [2,3],[2,4],
                [3,6],[3,9],
                [4,5],
                [5,6],
                [6,7],
                [7,8],[7,9],
                [8,9]]
    
    oplist = []
    for i in topology:
        oplist.append(("XX",[i[0],i[1]],1))
        oplist.append(("YY",[i[0],i[1]],1))
        oplist.append(("ZZ",[i[0],i[1]],1))
    for i in range(n):
        oplist.append(("Z",[i],1))
    return SparsePauliOp.from_sparse_list(oplist,num_qubits=n).to_matrix(True)



def generate_hamiltonian(n):
    topology = random_regular_graph(3,n,seed=31415).edges
    oplist = []
    for i in topology:
        oplist.append(("XX",[i[0],i[1]],1))
        oplist.append(("YY",[i[0],i[1]],1))
        oplist.append(("ZZ",[i[0],i[1]],1))
    for i in range(n):
        oplist.append(("Z",[i],1))
    return SparsePauliOp.from_sparse_list(oplist,num_qubits=n).to_matrix(True)





# The number of circuit layers
layers = 2

# the number of subblocks in a single layer
inner_blocks = 1

# The number of qubits
qbit_num = 17



# generate the Hamiltonian
Hamiltonian = generate_hamiltonian_tmp( qbit_num )


# obtain the groud state energy of the Hamitonian
[eigvals, eigvecs] = sp.sparse.linalg.eigs( Hamiltonian, k=10, which='SR' )
eigval = np.real(eigvals[0])
eigvec = eigvecs[:,0]

print( 'The target eigenvalue is: ', eigval )


# generate configuration dictionary for the solver
config = {"max_inner_iterations":800, 
	"batch_size": 128,
	"convergence_length": 20}

# initiate the VQE object with the Hamiltonian
VQE_Heisenberg = Variational_Quantum_Eigensolver(Hamiltonian, qbit_num, config, accelerator_num=1)

# set the optimization engine to agents
VQE_Heisenberg.set_Optimizer("COSINE")

# set the ansatz variant (U3 rotations and CNOT gates)
VQE_Heisenberg.set_Ansatz("HEA_ZYZ")

# generate the circuit ansatz for the optimization
VQE_Heisenberg.Generate_Circuit( layers, inner_blocks)

# create random initial parameters 
param_num  = VQE_Heisenberg.get_Parameter_Num()
print('The number of free parameters is: ', str(param_num) )


parameters = np.random.randn( param_num )*2*np.pi#np.zeros( (param_num,) )
VQE_Heisenberg.set_Optimized_Parameters(parameters)

#VQE_Heisenberg.set_Initial_State( eigvec )


# calculate the entropy of the exact ground state
page_entropy                = 2 * np.log(2.0) - 1.0/( pow(2, qbit_num-2*2+1) )
entropy_exact_gs            = VQE_Heisenberg.get_Second_Renyi_Entropy( parameters=np.array([]), qubit_list=[0,1], input_state=eigvec ) 
normalized_entropy_exact_gs = entropy_exact_gs/page_entropy
print('The normalized entropy of the exact ground state evaluated on qubits 0 and 1 is:', normalized_entropy_exact_gs)
print(' ')
print(' ')
print(' ', flush=True)

for iter_idx in range(400):

    # start an etap of the optimization (max_inner_iterations iterations)
    VQE_Heisenberg.Start_Optimization()

    # retrieve the current parameter set
    parameters = VQE_Heisenberg.get_Optimized_Parameters()

    # retrive the current VQE energy after max_inner_iterations iterations
    VQE_energy = VQE_Heisenberg.Optimization_Problem( parameters ) 

    # calculate the Renyi entropy after max_inner_iterations iterations on the subsystem made of the 0-th and the 1st qubits
    qubit_list = [0,1]

    page_entropy       = len(qubit_list) * np.log(2.0) - 1.0/( pow(2, qbit_num-2*len(qubit_list)+1) )
    entropy            = VQE_Heisenberg.get_Second_Renyi_Entropy( parameters=parameters, qubit_list=qubit_list ) 
    normalized_entropy = entropy/page_entropy


    print('Current VQE energy: ', VQE_energy, ' normalized entropy: ', normalized_entropy)

    np.save( 'Heisenberg_VQE_data.npy', parameters, topology ) 
    
    initial_state = np.zeros( (1 << qbit_num), dtype=np.complex128 )
    initial_state[0] = 1.0 + 0j        
        
    
    state_to_transform = initial_state.copy()    
    VQE_Heisenberg.apply_to( parameters, state_to_transform );   
    
    overlap      = state_to_transform.transpose().conjugate() @ eigvecs
    overlap_norm = np.real(overlap * overlap.conjugate())

    for idx in range( overlap_norm.size) :
        print('The overlap integral with the exact eigenstates of energy ', eigvals[idx], ' is: ', overlap_norm[idx] )
    
    print('The sum of the calculated overlaps: ', np.sum(overlap_norm ) )  
              

    if ( VQE_energy < 0.99*eigval):
        break
        
       
