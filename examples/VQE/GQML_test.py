from squander import Generative_Quantum_Machine_Learning
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib
import dataset_generator
import networkx as nx


def generate_MRF_dataset(n_nodes, graph_type, dataset_size, path = None, G=None):
    if graph_type != "custom":
        mrf = dataset_generator.GeneralBinaryMRF(graph_type, n_nodes)
    else:
        mrf = dataset_generator.GeneralBinaryMRF(graph_type, n_nodes, G=G)
    mrf_samples = np.random.choice(
        range(2**mrf.n_vertices), size=dataset_size, p=mrf.distribution
    )
    training_set = np.array(
        [
            np.array(list(format(i, "b").zfill(mrf.n_vertices))).astype(int)
            for i in mrf_samples
        ]
    )

    if path is not None:
        mrf.save(path)

    return training_set, mrf.distribution, list(nx.find_cliques(mrf.graph))

n_nodes = 5
graph_type = "custom"
dataset_size = 1000

G = nx.Graph()
G.add_nodes_from(range(n_nodes))
edges = [(x, x+1) for x in range(n_nodes-1)]
edges.append((n_nodes-1, 0))
# edges = [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9], [9, 1]]
G.add_edges_from(edges)
training_set, target_distribution, cliques = generate_MRF_dataset(n_nodes, graph_type, dataset_size, G=G)


# generate configuration dictionary for the solver
config = {"max_inner_iterations":8000, 
	"batch_size": 3,
    "check_for_convergence": True,
	"convergence_length": 20,
    "output_periodicity": 500}
qbit_num = n_nodes
sigma = [0.25, 10, 1000]
x = training_set.astype(np.int32)
P_star = target_distribution
use_lookup_table = True
use_exact  = True
print(cliques)

GQML = Generative_Quantum_Machine_Learning(x, P_star, sigma, qbit_num, use_lookup_table, cliques, use_exact, config)


# set the optimization engine to agents
GQML.set_Optimizer("COSINE")

# set the ansatz variant (U3 rotations and CNOT gates)
GQML.set_Ansatz("QCMRF")

GQML.Generate_Circuit(10, 1)
param_num  = GQML.get_Parameter_Num()
print(param_num)


parameters = np.zeros(param_num)
GQML.set_Optimized_Parameters(parameters)
print(GQML.get_Qiskit_Circuit())


initial_state = np.zeros( (1 << qbit_num), dtype=np.complex128 )
initial_state[0] = 1.0 + 0j        
state_to_transform = initial_state.copy()    
GQML.apply_to( parameters, state_to_transform );   
P_theta = np.abs(state_to_transform)**2
print("TV",np.sum(np.abs(P_theta - P_star))/2)


tvs_qcmrf = []
for i in range(1):
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
    tvs_qcmrf.append(np.sum(np.abs(P_theta - P_star))/2)
       
    plt.plot(P_star)
    plt.plot(P_theta)
    plt.show()

tvs_hea = []
for i in range(0):
    GQML.set_Ansatz("HEA")

    GQML.Generate_Circuit(1, 1)
    param_num  = GQML.get_Parameter_Num()
    parameters = np.zeros(param_num)
    print(param_num)
    print(GQML.get_Qiskit_Circuit())
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
    tvs_hea.append(np.sum(np.abs(P_theta - P_star))/2)
       
    plt.plot(P_star)
    plt.plot(P_theta)
    plt.show()
