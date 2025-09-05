from squander import Generative_Quantum_Machine_Learning
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import dataset_generator

n_nodes = 9
graph_type = "grid"
dataset_size = 4000

training_set, target_distribution, cliques = dataset_generator.generate_MRF_dataset(n_nodes, graph_type, dataset_size)
print(cliques)
# generate configuration dictionary for the solver
config = {"max_inner_iterations":800, 
	"batch_size": 2,
    "check_for_convergence": True,
	"convergence_length": 20,
    "output_periodicity": 50}
qbit_num = n_nodes
sigma = 10
x = np.astype(training_set, np.int32)
P_star = target_distribution
# print(P_star)
mrf_samples = np.array(range(2**qbit_num))
use_lookup_table = True

GQML = Generative_Quantum_Machine_Learning(x, P_star, sigma, qbit_num, use_lookup_table, cliques, config)


# set the optimization engine to agents
GQML.set_Optimizer("COSINE")

# set the ansatz variant (U3 rotations and CNOT gates)
GQML.set_Ansatz("QCMRF")

GQML.Generate_Circuit(4, 1)
param_num  = GQML.get_Parameter_Num()
print(param_num)


parameters = np.zeros(param_num)
# print("MMD", GQML.Optimization_Problem(parameters))
GQML.set_Optimized_Parameters(parameters)
print(GQML.get_Qiskit_Circuit())


initial_state = np.zeros( (1 << qbit_num), dtype=np.complex128 )
initial_state[0] = 1.0 + 0j        
state_to_transform = initial_state.copy()    
GQML.apply_to( parameters, state_to_transform );   
P_theta = np.abs(state_to_transform)**2
print("TV",np.sum(np.abs(P_theta - P_star))/2)

# param_copy = parameters.copy()
# print(parameters, param_copy)
# thetas = np.linspace(0, 1*np.pi, 100)
# mmds = []
# for i in range(4):
#     mmds.append([])
#     for fi in thetas:
#         parameters[i] = fi
#         mmds[i].append(GQML.Optimization_Problem(parameters))
#     parameters = param_copy.copy() 
#     print(parameters[i], mmds[i][0])
#     plt.scatter(parameters[i], mmds[i][0])
#     plt.plot(thetas, mmds[i])


parameters = np.zeros(param_num)
print("MMD", GQML.Optimization_Problem(parameters))
GQML.set_Optimized_Parameters(parameters)
GQML.Start_Optimization()
parameters = GQML.get_Optimized_Parameters()
initial_state = np.zeros( (1 << qbit_num), dtype=np.complex128 )
initial_state[0] = 1.0 + 0j        
state_to_transform = initial_state.copy()    
GQML.apply_to( parameters, state_to_transform );   
P_theta = np.abs(state_to_transform)**2
print("TV", np.sum(np.abs(P_theta - P_star))/2)
   
# plt.plot(P_star)
# plt.plot(P_theta)
# plt.show()

# for i in range(2**qbit_num):
#     print(every_sample[i], P_star[i], P_theta[i])

