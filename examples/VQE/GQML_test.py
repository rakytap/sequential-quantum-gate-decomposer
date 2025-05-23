from squander import Generative_Quantum_Machine_Learning
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import dataset_generator

n_nodes = 5
graph_type = "8grid"
dataset_size = 100

training_set, target_distribution = dataset_generator.generate_MRF_dataset(n_nodes, graph_type, dataset_size)
# generate configuration dictionary for the solver
config = {"max_inner_iterations":800, 
	"batch_size": 128,
	"convergence_length": 20}
qbit_num = n_nodes
sigma = 10
x = np.astype(training_set, np.int32)
P_star = target_distribution


GQML = Generative_Quantum_Machine_Learning(x, P_star, sigma, qbit_num, config)


# set the optimization engine to agents
GQML.set_Optimizer("COSINE")

# set the ansatz variant (U3 rotations and CNOT gates)
GQML.set_Ansatz("HEA_ZYZ")

GQML.Generate_Circuit(50, 1)
param_num  = GQML.get_Parameter_Num()

thetas = np.linspace(0, 1*np.pi, 100)
mmds = []
parameters = np.zeros(param_num)
print(GQML.Optimization_Problem(parameters))
GQML.set_Optimized_Parameters(parameters)

GQML.Start_Optimization()
parameters = GQML.get_Optimized_Parameters()

initial_state = np.zeros( (1 << qbit_num), dtype=np.complex128 )
initial_state[0] = 1.0 + 0j        
state_to_transform = initial_state.copy()    
GQML.apply_to( parameters, state_to_transform );   
P_theta = np.abs(state_to_transform)**2
   
plt.plot(P_star)
plt.plot(P_theta)
plt.show()

for i in range(2**qbit_num):
    print(P_theta[i], P_star[i])

