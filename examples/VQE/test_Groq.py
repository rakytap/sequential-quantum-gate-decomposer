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
from scipy.optimize import minimize

# mpirun -n 2 --bind-to numa --map-by numa python ~/vqe_project/100_layers.py
# hwloc-bind --membind node:0 --cpubind node:0 -- python 100_layers.py

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False

#MPI_imported = False

if MPI_imported:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()



topology = []

def generate_hamiltonian_tmp(n):
    
    topology = random_regular_graph(3,n,seed=31415).edges
    
    if MPI_imported:
        topology = comm.bcast(topology, root=0)
    '''
    
    topology = [[0,1],[0,4],[0,8],
                [1,2],[1,5],
                [2,3],[2,4],
                [3,6],[3,9],
                [4,5],
                [5,6],
                [6,7],
                [7,8],[7,9],
                [8,9]]
    
    '''
    '''
    topology = [[0,1],
                [1,2],
                [2,3],
                [3,4],
                [4,5],
                [5,6],
                [6,7],
                [7,8],
                [8,9]]
    '''
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

layers = 500
inner_blocks = 1

qbit_num = 20



# generate the Hamiltonian
Hamiltonian = generate_hamiltonian_tmp( qbit_num )
project_name = 'Test_GROQ'

with open(project_name + '_Hamiltonian.dat', 'wb') as file:
    pickle.dump(Hamiltonian, file)
    pickle.dump(topology, file)


# generate configuration dictionary for the solver
config = {"max_inner_iterations":800, #29000,
	"batch_size": 100, #128,
        "eta": 0.0001,
        "adaptive_eta": 0,
        "use_line_search": 0,
        "output_periodicity": 1,
	"convergence_length": 20}


########################## with GROQ acceelrator #############################################

# initiate the VQE object with the Hamiltonian
VQE_Heisenberg = Variational_Quantum_Eigensolver(Hamiltonian, qbit_num, config, accelerator_num=1)


# set the ansatz variant (U3 rotations and CNOT gates)
VQE_Heisenberg.set_Ansatz("HEA_ZYZ")

# Set the name for the project
VQE_Heisenberg.set_Project_Name( project_name )

# generate the circuit ansatz for the optimization
VQE_Heisenberg.Generate_Circuit( layers, inner_blocks)

# create random initial parameters 
param_num  = VQE_Heisenberg.get_Parameter_Num()
print('The number of free parameters: ', param_num)
parameters = np.random.randn( param_num ) *2*np.pi

if MPI_imported:
    parameters = comm.bcast(parameters, root=0)



start_time = time.time()

cost_fnc_GROQ = VQE_Heisenberg.Optimization_Problem( parameters )

print("--- %s seconds elapsed during cost function evaluation with Groq ---" % (time.time() - start_time))

########################## only CPU #############################################




# initiate the VQE object with the Hamiltonian
VQE_Heisenberg = Variational_Quantum_Eigensolver(Hamiltonian, qbit_num, config, accelerator_num=0)


# set the ansatz variant (U3 rotations and CNOT gates)
VQE_Heisenberg.set_Ansatz("HEA_ZYZ")

# Set the name for the project
VQE_Heisenberg.set_Project_Name( project_name )

# generate the circuit ansatz for the optimization
VQE_Heisenberg.Generate_Circuit( layers, inner_blocks)

# create random initial parameters 
param_num  = VQE_Heisenberg.get_Parameter_Num()
print('The number of free parameters: ', param_num)
#parameters = np.random.randn( param_num ) *2*np.pi

if MPI_imported:
    parameters = comm.bcast(parameters, root=0)

start_time = time.time()

cost_fnc_CPU = VQE_Heisenberg.Optimization_Problem( parameters )

print("--- %s seconds elapsed during cost function evaluation with CPU ---" % (time.time() - start_time))

print( cost_fnc_GROQ, cost_fnc_CPU )
assert abs((cost_fnc_GROQ-cost_fnc_CPU)/cost_fnc_CPU) < 1e-3

