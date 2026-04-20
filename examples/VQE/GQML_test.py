from squander import Generative_Quantum_Machine_Learning
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import dataset_generator
import networkx as nx
import matplotlib
import scipy.stats as stats
matplotlib.use('Agg')

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

def plot_distributions(original, GQML, pennylane_dist=None, name="dist_sq"):
    n_points = len(original)
    initial_state = np.zeros(n_points, dtype=np.complex128)
    initial_state[0] = 1.0+0j
    state_to_transform = initial_state.copy()
    params = GQML.get_Optimized_Parameters()
    GQML.apply_to(params, state_to_transform)
    P = abs(state_to_transform)**2
    bar_width = 1
    P_hist = np.array([P[i*bar_width:(i+1)*bar_width].sum() for i in range(int(n_points/bar_width))])/bar_width
    plt.figure()
    plt.bar(range(0, n_points, bar_width), P_hist, width=bar_width, alpha=0.5, color="orange",label="sqander")
    plt.plot(original, alpha=0.5, label="original")
    plt.legend()
    plt.grid()
    plt.savefig(f"{name}.pdf")
    plt.figure()
    if pennylane_dist is not None:
        plt.plot(pennylane_dist, alpha=0.5, label="pennylane")
        plt.plot(original, alpha=0.5, label="original")
        plt.legend()
        plt.grid()
        plt.savefig("dist_pl.pdf")

def run_squander(qbit_num, P_star, cliques, config, optimizer="COSINE", circuit=("HEA_ZYZ", (5, 1)), use_exact=False, project_name="qml_test"):
    qbit_num = n_nodes
    median_dist = int(2**n_nodes*(1-1/np.sqrt(2)))/150
    print("Median distance: ",median_dist)

    sigma = np.array([0.25, 8, 1024])
    sigma = np.array([x*median_dist for x in sigma])
    print("Sigma: ", sigma)

    x = training_set.astype(np.int32)
    P_star = target_distribution
    use_lookup_table = True

    print("Initializing GQML...")
    GQML = Generative_Quantum_Machine_Learning(x, P_star, sigma, qbit_num, use_lookup_table, cliques, use_exact, config, accelerator_num=0)
    GQML.set_Optimizer(optimizer)
    GQML.set_Ansatz(circuit[0])
    GQML.Generate_Circuit(circuit[1][0], circuit[1][1])
    param_num  = GQML.get_Parameter_Num()
    print("Numbers of parameteris in the circuit: ", param_num)

    # parameters = np.zeros(param_num)
    parameters = np.random.normal(0, 1/np.sqrt(n_nodes), param_num)
    GQML.set_Optimized_Parameters(parameters)
    plot_distributions(target_distribution, GQML, name="init")

    print(f"\nInitialized QQML, with {qbit_num} qubits and {param_num} parameters.\n\
{optimizer} optimizer will be used for optimization.\n\
Cirucuit type: {circuit[0]} with depth {circuit[1][0]} and repetitions {circuit[1][1]}.\n")

    t0 = time.time()
    print("Initial MMD", GQML.Optimization_Problem(parameters))
    print("One MMD time:", time.time()-t0)

    GQML.set_Project_Name(project_name)
    try:
        os.remove(f"{project_name}_costfuncs_entropy_and_tv.txt")
    except:
        pass

    print("Starting optimization...")
    t0 = time.time()
    GQML.Start_Optimization()
    print("Optimization time: ", opt_time := time.time()-t0)
    iters_sq, mmd_sq, _, tv_sq = np.loadtxt(f"{project_name}_costfuncs_entropy_and_tv.txt").T

    return iters_sq, mmd_sq, tv_sq, opt_time, GQML

def plot_cost_fnx(iters, sqander_cf, cosine_cf,cf_name, pennylane_cf=None, ):
    plt.figure()
    plt.title(cf_name)
    plt.plot( sqander_cf, label="bfgs2")
    plt.plot( cosine_cf, label="cosine")
    if pennylane_cf is not None:
        plt.plot(pennylane_cf, label="pennylane adam")
    plt.legend()
    plt.savefig(f"{cf_name}.pdf")

n_nodes = 10
graph_type = "custom"
dataset_size = 100

G = nx.Graph()
G.add_nodes_from(range(n_nodes))
edges = [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9], [9, 1]]
G.add_edges_from(edges)
training_set, target_distribution, cliques = generate_MRF_dataset(n_nodes, graph_type, dataset_size, G=G)
p_num = np.sum([2**(len(i)-1) for i in cliques])+3*n_nodes

cliques = sorted([sorted(x) for x in cliques])
print(cliques)
dist = stats.norm(loc=2**(n_nodes)*2/7, scale=2**(n_nodes-3))
dist2 = stats.norm(loc=2**(n_nodes)*5/7, scale=2**(n_nodes-3))
target_distribution = dist.pdf(range(0, 2**n_nodes))+dist2.pdf(range(0, 2**n_nodes))
# target_distribution = stats.norm.pdf(range(2**n_nodes), loc=2**(n_nodes-1), scale=2**n_nodes/4)
target_distribution = np.random.random(2**n_nodes)
target_distribution = target_distribution/np.sum(target_distribution)
print(np.sum(target_distribution))


iters = 4000
output_num = 100
bs = 12
circuit = ("HEA_ZYZ", (10, 1))
# generate configuration dictionary for the solver
config = {
    "max_inner_iterations": iters, 
    "batch_size": bs,
    "check_for_convergence": False,
    "da_maxiter": iters,
    "da_maxfun": iters,
    "da_no_local_search": 1,
    "output_periodicity": int(iters//output_num)
}

repatitions = 1
tvs = []
for i in range(repatitions):
    sq_iters, sq_mmd, sq_tv, sq_time, GQML = run_squander(qbit_num=n_nodes, P_star=target_distribution, cliques=cliques, config=config, use_exact=True, circuit=circuit, optimizer="BFGS2")
    tvs.append(sq_tv)
plot_distributions(target_distribution, GQML, name="bfgs2")

iters = 3000
output_num = 100
config = {
    "max_inner_iterations": iters, 
    "da_maxiter": iters,
    "da_maxfun": iters,
    "da_no_local_search": 1,
    "eta": 0.1,
    "adaptive_eta": False,
    "randomization_threshold": 10,
	"batch_size": bs,
    "batch_size_stochastic_gradient": 4,
    "check_for_convergence": False,
    "sampling_rate": 1000,
    "output_periodicity": int(iters//output_num)
}

tvs_stoch = []
for i in range(1):
    sq_iters_stoch, sq_mmd_stoch, sq_tv_stoch, sq_time_stoch, GQML = run_squander(qbit_num=n_nodes, P_star=target_distribution, cliques=cliques, use_exact=True, config=config, circuit=circuit, optimizer="COSINE")
    tvs_stoch.append(sq_tv_stoch)
plot_distributions(target_distribution, GQML, name="cosine")

print("\n" + "#"*50)
print("Exact optimization tv distance: ", sq_tv[-1], ", elapsed time: ", sq_time)
print("Stochastic optimization tv distance: ", sq_tv_stoch[-1], ", elapsed time: ", sq_time_stoch)

plot_cost_fnx(sq_iters, sq_tv, sq_tv_stoch,  "tv")
plot_cost_fnx(sq_iters, sq_mmd, sq_mmd_stoch,  "mmd")
