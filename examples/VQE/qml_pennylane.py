import jax
import time
from itertools import chain, combinations
import optax
import numpy as np
import jax.numpy as jnp
from functools import partial
import pennylane as qml
from qiskit.circuit import ParameterVector, QuantumCircuit

jax.config.update("jax_enable_x64", True)

class MMD:
    def __init__(self, scales, space):
        gammas = 1/(2*scales)
        sq_dist = jnp.abs(space[:, None] - space[None, :]) **2
        self.K = sum(jnp.exp(-gamma*sq_dist) for gamma in gammas) / len(scales)
        self.scales = scales

    def k_expval(self, px, py):
        return px @ self.K @ py

    def __call__(self, px, py):
        pxy = px - py
        return self.k_expval(pxy, pxy)

class QCBM:
    def __init__(self, circ, mmd, py):
        self.circ = circ
        self.mmd = mmd
        self.py = py

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        px = self.circ(params)
        return self.mmd(px, self.py), px

def ansatz(weights, qubits, cliques):
    for q in qubits:
        qml.Hadamard(q)

    all_subsets = []
    w_idx = [0]
    for i in range(len(cliques)):
        subset = []
        generate_clique_circuit(0, cliques[i], all_subsets, subset, w_idx, weights)
    print(all_subsets)
    for i in range(len(qubits)):
        qml.U3(weights[w_idx[0]], weights[w_idx[0]+1], weights[w_idx[0]+2],wires=qubits[i])
        w_idx[0] += 3

def multiRZ(weights, qubits, w_idx):
    for i in range(len(qubits)-1):
        qml.CNOT([qubits[i+1], qubits[i]])
    qml.RZ(weights[w_idx[0]], wires=qubits[len(qubits)-1])
    w_idx[0] += 1
    for i in range(len(qubits)-1, 0, -1):
        qml.CNOT([qubits[i], qubits[i-1]])

def generate_clique_circuit(i, arr, res, subset, w_idx, weights):
    if i == len(arr):
        if len(subset) != 0:
            if subset not in res:
                res.append(subset[:])
                multiRZ(weights, subset, w_idx)
        return

    subset.append(arr[i])
    generate_clique_circuit(i+1, arr, res, subset, w_idx, weights)

    subset.pop()
    generate_clique_circuit(i+1, arr, res, subset, w_idx, weights)


def all_unique_nonempty_subsets(cliques):
    """
    Given a list of cliques (each a list or set of variables),
    returns a list of all unique non-empty subsets across all cliques.
    """
    def nonempty_subsets(clique):
        clique = list(clique)
        # all non-empty subsets
        return chain.from_iterable(combinations(clique, r) for r in range(1, len(clique) + 1))

    seen = set()
    unique_subsets = []
    for clique in cliques:
        for subset in nonempty_subsets(clique):
            s = frozenset(subset)  # use frozenset for hashable set comparison
            if s not in seen:
                seen.add(s)
                unique_subsets.append(sorted(list(subset)))

    return unique_subsets


def make_parametric_circuit(circuit):
    new_circuit = QuantumCircuit(circuit.num_qubits)
    param_idx = 0
    params = []
    angles = ParameterVector("angles", 0)

    for instr, quargs, cargs in circuit.data:
        if instr.name == "rz":
            angles.resize(param_idx+1)
            # p = Parameter(f"t_P{param_idx}")
            # params.append(p)
            new_circuit.rz(angles[param_idx], quargs[0])
            param_idx += 1
        if instr.name == "ry":
            angles.resize(param_idx+1)
            # p = Parameter(f"t_P{param_idx}")
            # params.append(p)
            new_circuit.ry(angles[param_idx], quargs[0])
            param_idx += 1
        elif instr.name == "u":
            angles.resize(param_idx+3)
            new_circuit.u(angles[param_idx], angles[param_idx+1], angles[param_idx+2], quargs[0])
            param_idx += 3
        else:
            new_circuit.append(instr, quargs, cargs)
    return new_circuit, params
    

def run_pennylane(target_distribution, n_qubits, n_layers, n_iter, cliques, bandwith, params=None, ansatz_type="QCMRF", circuit_input=None):
    dev = qml.device("default.qubit", wires=n_qubits)

    if ansatz_type != "QCMRF":
        whshape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    else:
        whshape = len(all_unique_nonempty_subsets(cliques))+3*n_qubits
    if params is not None:
        weights = params
    else:
        weights = np.random.random(whshape)
    print(weights.shape)

    circuit_input_p, params = make_parametric_circuit(circuit_input)
    circuit_qk = qml.from_qiskit(circuit_input_p)
    @qml.qnode(dev)
    def circuit(weights):
        if circuit_input is not None:
            circuit_qk(weights, wires=range(n_qubits))
        else:
            if ansatz_type != "QCMRF":
                qml.StronglyEntanglingLayers( weights=weights, ranges=[1] * n_layers, wires=range(n_qubits))
            else:
                ansatz(weights, range(n_qubits), cliques)
        return qml.probs(wires=range(n_qubits))

    jit_circuit = jax.jit(circuit)

    bandwith = jnp.array(bandwith)
    space = jnp.arange(2**n_qubits)
    mmd = MMD(bandwith, space)
    qcbm = QCBM(jit_circuit, mmd, target_distribution)

    opt = optax.adam(learning_rate=0.1)

    opt_state = opt.init(weights)

    history = []
    divs = []
    n_iterations = n_iter

    
    @jax.jit
    def update_step(params, opt_state):
        (loss_val, qcbm_probs), grads = jax.value_and_grad(qcbm.mmd_loss, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        total_div = jnp.sum(jnp.abs(qcbm_probs-qcbm.py))/2
        return params, opt_state, loss_val, total_div, grads

    op = int(n_iterations // 10)
    t0 = time.time()
    for i in range(n_iterations):
        weights, opt_state, loss_val, kl_div, grads = update_step(weights, opt_state)

        if i % op == 0:
            grad_norm = jnp.linalg.norm(grads)
            print(f"Step: {i} Loss: {loss_val:.4f} TV-dist: {kl_div:.4f} Grad norm: {grad_norm}")


        history.append(loss_val)
        divs.append(kl_div)
    print("pennylane time", time.time() - t0)
    qcbm_probs = qcbm.circ(weights)

    # drawer = qml.draw(circuit, level="device")
    # print(drawer(weights))
    return qcbm_probs, history, divs

