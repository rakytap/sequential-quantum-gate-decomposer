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
np.random.seed(31415)
np.set_printoptions(linewidth=200) 

import pickle



topology = []

def generate_hamiltonian_tmp(n):

    topology = random_regular_graph(3,n,seed=31415).edges
    
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




def generate_circuit_ansatz( layers, inner_blocks, qbit_num):

    from squander import Circuit

    ret = Circuit( qbit_num )


    for layer_idx in range(layers) :


        for int in range(inner_blocks):
            # we organize gates into blocks that can be merged and thus faster to execute
            block1 = Circuit( qbit_num )
            block2 = Circuit( qbit_num )

            block1.add_RZ( 1 )
            block1.add_RY( 1 )
            block1.add_RZ( 1 )

            block2.add_RZ( 0 )
            block2.add_RY( 0 )
            block2.add_RZ( 0 )

            # add the partitions to the circuit
            ret.add_Circuit( block1 )
            ret.add_Circuit( block2 )

            ret.add_CNOT(1,0)


        for control_qbit in range(1, qbit_num-1, 2):
            
            if control_qbit+2<qbit_num :

                for int in range(inner_blocks):


                    # we organize gates into blocks that can be merged and thus faster to execute
                    block1 = Circuit( qbit_num )
                    block2 = Circuit( qbit_num )

                    block1.add_RZ( control_qbit+1 )
                    block1.add_RY( control_qbit+1 )
                    block1.add_RZ( control_qbit+1 )

                    block2.add_RZ( control_qbit+2 )
                    block2.add_RY( control_qbit+2 )
                    block2.add_RZ( control_qbit+2 )

                    # add the partitions to the circuit
                    ret.add_Circuit( block1 )
                    ret.add_Circuit( block2 )

                    ret.add_CNOT(control_qbit+2,control_qbit+1);
        


            for int in range(inner_blocks):


                # we organize gates into blocks that can be merged and thus faster to execute
                block1 = Circuit( qbit_num )
                block2 = Circuit( qbit_num )

                block1.add_RZ( control_qbit+1 )
                block1.add_RY( control_qbit+1 )
                block1.add_RZ( control_qbit+1 )

                block2.add_RZ( control_qbit )
                block2.add_RY( control_qbit )
                block2.add_RZ( control_qbit )

                # add the partitions to the circuit
                ret.add_Circuit( block1 )
                ret.add_Circuit( block2 )

                ret.add_CNOT(control_qbit+1,control_qbit);           


    return ret


# The number of circuit layers
layers = 1000

# the number of subblocks in a single layer
inner_blocks = 1

# The number of qubits
qbit_num = 16



# generate the Hamiltonian
Hamiltonian = generate_hamiltonian_tmp( qbit_num )


# generate custom circuit ansatz
squander_circuit = generate_circuit_ansatz( layers, inner_blocks, qbit_num)

# obtain the groud state energy of the Hamitonian
[eigvals, eigvecs] = sp.sparse.linalg.eigs( Hamiltonian, k=10, which='SR' )
eigval = np.real(eigvals[0])
eigvec = eigvecs[:,0]

print( 'The target eigenvalue is: ', eigval )


# generate configuration dictionary for the solver (if permforming further training)
config = {"max_inner_iterations":800, 
	"batch_size": 128,
	"convergence_length": 20}

# initiate the VQE object with the Hamiltonian
VQE_Heisenberg = Variational_Quantum_Eigensolver(Hamiltonian, qbit_num, config)

# set the optimization engine to agents
VQE_Heisenberg.set_Optimizer("COSINE")

# set the ansatz variant (U3 rotations and CNOT gates)
VQE_Heisenberg.set_Gate_Structure( squander_circuit )

# load pretrained parameters 
param_num  = VQE_Heisenberg.get_Parameter_Num()
print('The number of free parameters is: ', str(param_num) )

parameters = np.load( 'COSINE_1000x1_layers_Heisenberg_16_3point_zero_init_3overlap0.8602661822824491.npy' )

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


# retrive the current VQE energy after max_inner_iterations iterations
VQE_energy = VQE_Heisenberg.Optimization_Problem( parameters ) 

# calculate the Renyi entropy after max_inner_iterations iterations on the subsystem made of the 0-th and the 1st qubits
qubit_list = [0,1]

page_entropy       = len(qubit_list) * np.log(2.0) - 1.0/( pow(2, qbit_num-2*len(qubit_list)+1) )
entropy            = VQE_Heisenberg.get_Second_Renyi_Entropy( parameters=parameters, qubit_list=qubit_list ) 
normalized_entropy = entropy/page_entropy


print('Current VQE energy: ', VQE_energy, ' normalized entropy: ', normalized_entropy)


initial_state = np.zeros( (1 << qbit_num), dtype=np.complex128 )
initial_state[0] = 1.0 + 0j        
        
    
state_to_transform = initial_state.copy()    
VQE_Heisenberg.apply_to( parameters, state_to_transform );  
# or equivalently:  squander_circuit.apply_to( parameters, state_to_transform );   

    
overlap      = state_to_transform.transpose().conjugate() @ eigvecs
overlap_norm = np.real(overlap * overlap.conjugate())

for idx in range( overlap_norm.size) :
    print('The overlap integral with the exact eigenstates of energy ', eigvals[idx], ' is: ', overlap_norm[idx] )
   
print('The sum of the calculated overlaps: ', np.sum(overlap_norm ) )  
 
        
       
