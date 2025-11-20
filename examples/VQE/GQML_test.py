from squander import Generative_Quantum_Machine_Learning
import time
from squander.partitioning.partition import PartitionCircuitQasm
from squander.partitioning.partition import PartitionCircuit
import qml_pennylane
import os
import numpy as np
import matplotlib.pyplot as plt
import dataset_generator
import networkx as nx
from qiskit.qasm2 import dumps

print("works")

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

def plot_distributions(pennylane_dist, original):
    n_points = len(pennylane_dist)
    initial_state = np.zeros(n_points, dtype=np.complex128)
    initial_state[0] = 1.0+0j
    state_to_transform = initial_state.copy()
    params = GQML.get_Optimized_Parameters()
    GQML.apply_to(params, state_to_transform)
    P = abs(state_to_transform)**2
    plt.figure()
    plt.plot(P, alpha=0.5, label="sqander")
    plt.plot(original, alpha=0.5, label="original")
    plt.legend()
    plt.grid()
    plt.savefig("dist_sq.pdf")
    plt.figure()
    plt.plot(pennylane_dist, alpha=0.5, label="pennylane")
    plt.plot(original, alpha=0.5, label="original")
    plt.legend()
    plt.grid()
    plt.savefig("dist_pl.pdf")

def run_squander(n_run):
    print("MMD", GQML.Optimization_Problem(parameters))
    iters_sq_best = []
    mmd_sq_best = []
    tv_sq_best = []
    for i in range(n_run):
        GQML.set_Optimized_Parameters(params0)
        GQML.set_Project_Name(f"qml_test{i}")
        try:
            os.remove(f"qml_test{i}_costfuncs_entropy_and_tv.txt")
        except:
            pass
        t0 = time.time()
        GQML.Start_Optimization()
        print("squander time", time.time()-t0)
        iters_sq, mmd_sq, _, tv_sq = np.loadtxt(f"qml_test{i}_costfuncs_entropy_and_tv.txt").T
        if len(iters_sq_best) == 0:
            iters_sq_best = iters_sq[:]
            mmd_sq_best = mmd_sq[:]
            tv_sq_best = tv_sq[:]
        else:
            if tv_sq[-1] < tv_sq_best[-1]:
                iters_sq_best = iters_sq[:]
                mmd_sq_best = mmd_sq[:]
                tv_sq_best = tv_sq[:]
    print(tv_sq_best[-1])
    return iters_sq_best, mmd_sq_best, tv_sq_best

def run_pennylane(n_run):
    iters_pl_best = []
    mmd_pl_best = []
    P_theta_pl_best = []
    for i in range(n_run):
        # P_theta_pl, mmd_pl, tv_pl = qml_pennylane.run_pennylane(P_star, qbit_num, 6, iters, cliques, sigma, params0, ansatz_type="QCMRF", circuit_input=circuit) 
        P_theta_pl, mmd_pl, tv_pl = qml_pennylane.run_pennylane(P_star, qbit_num, 6, iters, cliques, sigma, params0) 
        if len(iters_pl_best) == 0:
            P_theta_pl_best = P_theta_pl[:]
            mmd_pl_best = mmd_pl[:]
            tv_pl_best = tv_pl[:]
        else:
            if tv_pl[-1] < tv_pl_best[-1]:
                P_theta_pl_best = P_theta_pl[:]
                mmd_pl_best = mmd_pl[:]
                tv_pl_best = tv_pl[:]
    return mmd_pl_best, tv_pl_best, P_theta_pl_best

def plot_cost_fnx(iters, sqander_cf, pennylane_cf, cf_name):
    plt.figure()
    plt.title(cf_name)
    plt.plot( sqander_cf, label="squander")
    plt.plot(pennylane_cf, label="pennylane")
    plt.legend()
    plt.savefig(f"{cf_name}.pdf")

n_nodes = 9
graph_type = "grid"
dataset_size = 1000

G = nx.Graph()
G.add_nodes_from(range(n_nodes))
edges = [(x, x+1) for x in range(n_nodes-1)]
edges.append((n_nodes-1, 0))
# edges = [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9], [9, 1]]
# edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes)]
G.add_edges_from(edges)
training_set, target_distribution, cliques = generate_MRF_dataset(n_nodes, graph_type, dataset_size, G=G)
p_num = np.sum([2**(len(i)-1) for i in cliques])+3*n_nodes

pos = nx.spring_layout(G, seed=42)  # Layout for nicer visualization
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=700, edge_color="gray", font_weight="bold")
plt.savefig("graph.png")
plt.figure()

cliques = sorted([sorted(x) for x in cliques])

iters = 100
bs = 1
full_iters = int(iters*(p_num//bs))
print("iterations", full_iters)
op = int(full_iters//iters)
print("op",op)
# generate configuration dictionary for the solver
config = {"max_inner_iterations": iters, 
	"batch_size": bs,
    "eta": .1,
    "check_for_convergence": False,
	"convergence_length": 20,
    "output_periodicity": int(iters//100)}
qbit_num = n_nodes
median_dist = int(2**n_nodes*(1-1/np.sqrt(2)))/150
print("median distance",median_dist)
sigma = np.array([0.25, 8, 1024])
sigma = np.array([x*median_dist for x in sigma])
x = training_set.astype(np.int32)
P_star = target_distribution
use_lookup_table = True
use_exact  = True

GQML = Generative_Quantum_Machine_Learning(x, P_star, sigma, qbit_num, use_lookup_table, cliques, use_exact, config, accelerator_num=0)
# set the optimization engine to agents
GQML.set_Optimizer("ADAM")
# set the ansatz variant (U3 rotations and CNOT gates)
GQML.set_Ansatz("QCMRF")
# os.remove("qml_test_costfuncs_entropy_and_tv.txt")
GQML.Generate_Circuit(5, 1)
param_num  = GQML.get_Parameter_Num()
print("param num", param_num)


# parameters = np.random.random(param_num)
parameters = np.zeros(param_num)
params0 = parameters.copy()
GQML.set_Optimized_Parameters(parameters)
circuit = GQML.get_Qiskit_Circuit()
max_partition_size = 4

t0 = time.time()
print("squander mmd", GQML.Optimization_Problem(parameters))
print("squander time", time.time()-t0)
# with open('tmp.qasm', 'w') as file:
#     qasm_c = dumps(circuit)
#     qasm_c = qasm_c.replace("u(", "u3(")
#     file.write(qasm_c)
# partitioned_circuit, params_reord, _ = PartitionCircuitQasm("tmp.qasm", max_partition_size, "ilp")
#
# GQML.set_Gate_Structure(partitioned_circuit)
# GQML.set_Optimized_Parameters(params_reord)
# t0 = time.time()
# print("MMD", GQML.Optimization_Problem(parameters))
# print("part", time.time()-t0)

pl_mmd, pl_tv, pl_P = run_pennylane(1)
sq_iters, sq_mmd, sq_tv = run_squander(1)                                 

plot_cost_fnx(sq_iters, sq_tv, pl_tv, "tv")
plot_cost_fnx(sq_iters, sq_mmd, pl_mmd, "mmd")
plot_distributions(pl_P, P_star)
