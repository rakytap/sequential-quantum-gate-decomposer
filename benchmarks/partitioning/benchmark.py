import itertools
from squander.partitioning.partition import PartitionCircuitQasm
from qiskit import QuantumCircuit
from squander.partitioning.tools import qiskit_to_squander_name, get_qubits, total_float_ops
from squander import utils

import timeit
import glob
import os

USE_ILP = True

MAX_GATES_ALLOWED = 1024**2 

METHOD_NAMES = [
    "kahn", 
    "tdag",
    "gtqcp",
    "qiskit",
    "qiskit-fusion",
    "bqskit-Quick",
    # "bqskit-Greedy", 
    # "bqskit-Scan",
    # "bqskit-Cluster", 
] + (["ilp", "ilp-fusion", #"ilp-fusion-ca"
      ] if USE_ILP else [])

from squander.gates import gates_Wrapper as gate
SUPPORTED_GATES = {x for n in dir(gate) for x in (getattr(gate, n),) if not n.startswith("_") and issubclass(x, gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES = {n for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES.remove("S") #delete this if S gate is properly supported

def purity_analysis():
    """
    Run a small symbolic experiment to study “purity” and “sparsity” of control sets.

    This routine builds a set of helper functions and 1q/2q/3q gate factories with SymPy,
    then:
      1) Checks which qubit subsets act as pure/sparse controls for some canonical gates
         (e.g., U3 on 1q, CRY on 2q, CCX on 3q).
      2) Composes short circuits (e.g., U3 on qubit i followed by CRY(0,1)) and reports
         which control subsets remain pure/sparse after composition.
      3) Prints the identified “pure” and “sparsity” control sets for each experiment.

    Notes:
      - Endianness is governed by the local variable `little_endian` (default: True).
      - `apply_to` operates on full operators (2^n × 2^n), not statevectors.
      - Requires SymPy. The helper `apply_to` also uses `itertools.product`.

    Args:
        None

    Returns:
        None
            Results are printed to stdout; the function is intended as a diagnostic/demo.
    """
    import sympy, functools, operator
    from sympy.combinatorics import Permutation
    theta, phi, lbda, gamma = sympy.Symbol("θ"), sympy.Symbol("ϕ"), sympy.Symbol("λ"), sympy.Symbol("γ")
    theta2 = sympy.Symbol("θ2")
    little_endian = True
    def find_control_qubits(psi, num_qubits):
        """
        Identify control-qubit sets that make a unitary ‘pure-controlled’ or ‘sparse-controlled’.

        A subset S of qubits is:
          - pure-controlled if, when control pattern S is not fully satisfied in basis index j,
            row j of the unitary equals the corresponding computational basis row (identity row).
          - sparse-controlled if, when control pattern S is not fully satisfied in j,
            row j has zeros everywhere except possibly in columns k that satisfy S.

        Args:
            psi (sympy.Matrix): 2**num_qubits × 2**num_qubits unitary matrix.
            num_qubits (int): Total number of qubits.

        Returns:
            tuple[list[list[int]], list[list[int]]]:
                pure_controls:  list of control index lists (little-endian by default) making psi pure-controlled.
                sparsity_controls: list of control index lists making psi sparse-controlled.
        """
        pure_controls = []
        sparsity_controls = []
        for i in range(1, 1<<num_qubits):
            is_pure, is_sparse = True, True
            for j in range(1<<num_qubits):
                if i & j != i:
                    if not all(psi[j, k] == (1 if j == k else 0) for k in range(1<<num_qubits)):
                        is_pure = False
                    if not all(psi[j, k] == 0 for k in range(1<<num_qubits) if i & k == i):
                        is_sparse = False
            if is_pure: pure_controls.append([j if little_endian else num_qubits-1-j for j in range(num_qubits) if i & (1<<j)])
            if is_sparse: sparsity_controls.append([j if little_endian else num_qubits-1-j for j in range(num_qubits) if i & (1<<j)])
        return pure_controls, sparsity_controls
    def apply_to(psi, num_qubits, gate, gate_qubits):
        """
        Left-apply a k-qubit gate to designated positions inside an n-qubit operator.

        Embeds the k-qubit `gate` on `gate_qubits` (indices in [0..n-1]) under the
        current endianness, and returns (gate ⊗ I_rest) · psi with the correct qubit
        wiring. Works for psi as a full 2^n×2^n operator (not a statevector).

        Args:
            psi (sympy.Matrix): 2**num_qubits × 2**num_qubits operator to transform.
            num_qubits (int): Total number of qubits n.
            gate (sympy.Matrix): 2**k × 2**k unitary to embed.
            gate_qubits (Iterable[int]): The k target qubit indices where `gate` acts.

        Returns:
            sympy.Matrix: New 2**num_qubits × 2**num_qubits matrix after applying `gate`.
        """
        pos = [(q if little_endian else num_qubits-1-q) for q in gate_qubits]
        k = len(gate_qubits)
        deltas, combo = [1<<p for p in pos], []
        for u in range (1 << k):
            m = 0; x = u; b = 0
            while x:
                if x & 1: m ^= deltas[b]
                x >>= 1; b += 1
            combo.append(m)
        pos_set = set(pos)
        rest = [p for p in range(num_qubits) if not p in pos_set]
        output = sympy.zeros(1<<num_qubits, 1<<num_qubits)
        for rest_bits in itertools.product((0, 1), repeat=len(rest)):
            base = 0
            for bit, p in zip(rest_bits, rest):
                if bit: base |= 1<<p
            mat, idx = sympy.zeros(0, 1<<num_qubits), []
            for u in range(1 << k):
                idx.append(base ^ combo[u])
                mat = mat.col_join(psi[idx[u],:])
            prod = gate @ mat
            for u in range(1 << k):
                output[idx[u],:] = prod[u,:]
        return output
    def compile_gates(num_qubits, gates):
        """
        Compose a list of embedded gates into a full n-qubit unitary.

        Args:
            num_qubits (int): Total number of qubits n.
            gates (Iterable[tuple[sympy.Matrix, Iterable[int]]]):
                Sequence of (gate, gate_qubits) pairs. Each `gate` is 2**k×2**k,
                applied to the provided `gate_qubits`.

        Returns:
            sympy.Matrix: The resulting 2**n × 2**n unitary product.
        """
        Umtx = sympy.eye(2**num_qubits)
        for gate in gates: Umtx = apply_to(Umtx, num_qubits, *gate)
        Umtx.simplify()
        return Umtx
    def make_controlled(gate, gate_qubits, gateother=None): #control is first qubit, gate qubits come after
        """
        Build a 1-control controlled version of `gate` (control is the first qubit by convention).

        Constructs block-diagonal diag(gateother or I, gate) and applies a permutation so that,
        under little-endian, the control is the least significant qubit.

        Args:
            gate (sympy.Matrix): 2**m × 2**m target unitary.
            gate_qubits (int): Number of target qubits m (excluding the control qubit).
            gateother (sympy.Matrix | None): Optional block for the control-off subspace.
                If None, identity of size 2**m is used.

        Returns:
            sympy.Matrix: A 2**(m+1) × 2**(m+1) controlled-gate matrix.
        """
        res = sympy.diag(gateother if not gateother is None else sympy.eye(1<<gate_qubits), gate)
        P = sympy.eye(1<<(gate_qubits+1))[:, [2*x+y for y in (0, 1) for x in range(1<<gate_qubits)]]
        return P * res * P.T if little_endian else res
    def make_inverse(g): return g**-1
    def make_sqrt(g): return g**(1/2)
    def gen_I(): return sympy.eye(2)
    def gen_Rx(theta): return sympy.exp(-sympy.I*theta/2*gen_X()).simplify().nsimplify()
    def gen_Ry(theta): return sympy.exp(-sympy.I*theta/2*gen_Y()).simplify().nsimplify()
    def gen_Rz(phi): return sympy.exp(-sympy.I*phi/2*gen_Z()).simplify().nsimplify()
    def gen_GP(theta): return sympy.exp(theta*sympy.I)
    def gen_H(): return sympy.Matrix([[1, 1], [1, -1]])/sympy.sqrt(2)
    def gen_X(): return sympy.Matrix([[0, 1], [1, 0]])
    def gen_Y(): return sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])
    def gen_Z(): return sympy.Matrix([[1, 0], [0, -1]])
    def gen_S(): return sympy.Matrix([[1, 0], [0, sympy.I]])
    def gen_Sdg(): return sympy.Matrix([[1, 0], [0, -sympy.I]])
    def gen_SX(): return sympy.nsimplify(make_sqrt(gen_X()), rational=True)
    def gen_CZPowGate(t): return make_controlled(sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*sympy.pi*t)]]), 1)
    def gen_fSim(theta, phi): return gen_iSWAP()**(-2*theta/sympy.pi) * gen_CZPowGate(-phi/sympy.pi)
    def gen_SYC(): return gen_fSim(sympy.pi/2, sympy.pi/6)
    def gen_T(): return sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*sympy.pi/4)]])
    def gen_Tinv(): return make_inverse(gen_T())
    def gen_Tdg(): return sympy.Matrix([[1, 0], [0, sympy.exp(-sympy.I*sympy.pi/4)]])
    def gen_P(theta): return sympy.Matrix([[1, 0], [0, sympy.exp(sympy.I*theta)]])
    def gen_U1(theta): return gen_P(theta)
    def gen_CH(): return sympy.Matrix(make_controlled(gen_H(), 1))
    def gen_CX(): return sympy.Matrix(make_controlled(gen_X(), 1))
    def gen_CNOT(): return gen_CX()
    def gen_CY(): return sympy.Matrix(make_controlled(gen_Y(), 1))
    def gen_CZ(): return sympy.Matrix(make_controlled(gen_Z(), 1))
    def gen_U(theta, phi, lbda): u = (gen_Rz(phi) @ gen_Ry(theta) @ gen_Rz(lbda) * gen_GP((phi+lbda)/2)); u.simplify(); return u
    def gen_U2(phi, lbda): return gen_U(sympy.pi/2, phi, lbda)
    def gen_U3(theta, phi, lbda): return gen_U(theta, phi, lbda)
    def gen_R(theta, phi): return gen_U(theta, phi-sympy.pi/2, -phi+sympy.pi/2)
    def gen_CR(theta, phi): return make_controlled(gen_R(theta, phi), 1)
    def gen_CROT(theta, phi): return make_controlled(gen_R(theta, phi), 1, gen_R(-theta, phi))
    def gen_CRY(theta): return make_controlled(gen_Ry(theta), 1)
    def gen_CCX(): return make_controlled(gen_CNOT(), 2)
    def gen_Toffoli(): return gen_CCX()
    def gen_CCZ(): return make_controlled(gen_CZ(), 2)
    def gen_SWAP(): return sympy.nsimplify(functools.reduce(operator.add, (compile_gates(2, [(gen(), [0]), (gen(), [1])]) for gen in (gen_I, gen_X, gen_Y, gen_Z))) / 2, rational=True)
    def gen_CSWAP(): return make_controlled(gen_SWAP(), 2)
    def gen_iSWAP(): return gen_Rxy(-sympy.pi)
    def gen_CP(phi): return make_controlled(gen_P(phi), 1)
    def gen_CR(phi): return gen_CP(phi)
    def gen_CS(): return make_controlled(gen_S(), 1)
    def gen_CU1(theta): return gen_CP(theta)
    def gen_CU(theta, phi, lbda, gamma): return make_controlled(gen_U(theta, phi, lbda) * gen_GP(gamma), 1)
    def gen_CU3(theta, phi, lbda): return gen_CU(theta, phi, lbda, 0)
    def gen_Rxx(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])])).simplify() #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_X(), [0]), (gen_X(), [1])])
    def gen_Ryy(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])])).simplify() #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Y(), [0]), (gen_Y(), [1])])
    def gen_Rzz(theta): return sympy.exp(-sympy.I*theta/2*compile_gates(2, [(gen_Z(), [0]), (gen_Z(), [1])])).simplify() #compile_gates(2, [(sympy.cos(theta/2)*gen_I(), [0]), (gen_I(), [1])]) - compile_gates(2, [(sympy.I*sympy.sin(theta/2)*gen_Z(), [0]), (gen_Z(), [1])])
    def gen_Rxy(phi): return sympy.exp(-sympy.I*phi/4*sympy.simplify(compile_gates(2, [(gen_X(), [0]), (gen_X(), [1])]) + compile_gates(2, [(gen_Y(), [0]), (gen_Y(), [1])]))).simplify().nsimplify() 
    def gen_CCZ_decomp(): return compile_gates(3, [(gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tinv(), [1]), (gen_CNOT(), [0, 1])])
    def gen_CCX_decomp(): return compile_gates(3, [(gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tinv(), [1]), (gen_CNOT(), [0, 1])])
    def gen_SWAP_decomp(): return compile_gates(2, [(gen_CNOT(), [0, 1]), (gen_CNOT(), [1, 0]), (gen_CNOT(), [0, 1])])
    def gen_CSWAP_decomp(): return compile_gates(3, [(gen_CNOT(), [2, 1]), (gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2]), (gen_CNOT(), [0, 1]), (gen_T(), [0]), (gen_Tinv(), [1]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [2, 1])])
    def gen_iSWAP_decomp(): return sympy.nsimplify(compile_gates(2, [(gen_S(), [0]), (gen_S(), [1]), (gen_H(), [0]), (gen_CNOT(), [0, 1]), (gen_CNOT(), [1, 0]), (gen_H(), [1])]), rational=True)
    def gen_SX_test(): return sympy.nsimplify(sympy.Matrix([[1+sympy.I, 1-sympy.I], [1-sympy.I, 1+sympy.I]])/2, rational=True)
    def gen_Rx_test(theta): return sympy.nsimplify(sympy.Matrix([[sympy.cos(theta/2), -sympy.I*sympy.sin(theta/2)], [-sympy.I*sympy.sin(theta/2), sympy.cos(theta/2)]], rational=True))
    def gen_SYC_test(): return sympy.Matrix([[1, 0, 0, 0], [0, 0, -sympy.I, 0], [0, -sympy.I, 0, 0], [0, 0, 0, sympy.exp(-sympy.I*sympy.pi/6)]])
    #print(compile_gates(3, [(gen_H(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [2]), (gen_CNOT(), [1, 2]), (gen_Tinv(), [2]), (gen_CNOT(), [0, 2]), (gen_T(), [1]), (gen_T(), [2]), (gen_H(), [2])]))
    assert gen_Rx(theta) == gen_Rx_test(theta)
    assert gen_SX() == gen_SX_test()
    assert gen_SYC() == gen_SYC_test()
    assert gen_CCZ() == gen_CCZ_decomp()
    assert gen_CCX() == gen_CCX_decomp()
    assert gen_SWAP() == gen_SWAP_decomp()
    assert gen_CSWAP() == gen_CSWAP_decomp()
    assert gen_iSWAP() == gen_iSWAP_decomp()
    print(find_control_qubits(gen_U3(theta, phi, lbda), 1), find_control_qubits(gen_CRY(theta), 2), find_control_qubits(gen_CCX(), 3))
    for i in range(3): #this proves any single qubit chain removes all purity, and converts aligning control to target
        print(f"U3({i})@CRY(0, 1) pure, sparse control:", find_control_qubits(compile_gates(3, [(gen_U3(theta, phi, lbda), [i]), (gen_CRY(theta2), (0, 1))]), 3))
    for i in range(3):
        for j in range(3):
            if i == j : continue
            print(f"CRY({i}, {j})@CRY(0, 1) pure, sparse control:", find_control_qubits(compile_gates(3, [(gen_CRY(theta), [i, j]), (gen_CRY(theta2), (0, 1))]), 3))

#purity_analysis(); assert False

def test_partitions(max_qubits = 4):
    """
    Benchmark partitioning strategies on QASM circuits
    Args:
        
        max_qubits: Max qubits per partition
    """
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print(f"Total QASM: {len(files)}, max qubits: {max_qubits}")
    allfiles = {}
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        qc_gates_names = {qiskit_to_squander_name(inst.operation.name) for inst in qc.data}
        num_gates = len(qc.data)
        fname = os.path.basename(filename)
        if num_gates > MAX_GATES_ALLOWED or not qc_gates_names.issubset(SUPPORTED_GATES_NAMES) or filename.endswith("_qsearch.qasm") or filename.endswith("_squander.qasm"):
            print(f"Skipping {fname}; qubits {qc.num_qubits} gates {qc_gates_names} num_gates {num_gates}")
            continue
        circ, _ = utils.qasm_to_squander_circuit(filename)
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        gate_to_qubit = { i: get_qubits(g) for i, g in gate_dict.items() }
        gate_to_tqubit = { i: g.get_Target_Qbit() for i, g in gate_dict.items() }
        print(fname, qc.num_qubits, num_gates, f"k={max_qubits}",
              "Max. float ops:", total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, None, [[i] for i in gate_dict]),
              "Control-aware Max. float ops:", total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, gate_to_tqubit, [[i] for i in gate_dict]))
        res = {}
        for method in METHOD_NAMES:
            print(method)
            ls = []
            def f():
                ls.extend(PartitionCircuitQasm( filename, max_qubits, method ))
            t = timeit.timeit(f, number=1)
            partitioned_circuit, parameters, L = ls
            res[method] = len(partitioned_circuit.get_Gates()), t, total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, None, L), total_float_ops(qc.num_qubits, max_qubits, gate_to_qubit, gate_to_tqubit, L)
        allfiles[fname] = (qc.num_qubits, num_gates, res)
        print(fname, allfiles[fname])
    import json
    print(json.dumps(allfiles, indent=2))    
    import matplotlib.pyplot as plt
    sorted_items = sorted(allfiles.items(), key=lambda item: (item[1][0], item[1][1]))
    circuits_sorted = [name for name, _ in sorted_items]
    circuits_labels = [name.replace(".qasm", f" ({x[0]}, {x[1]})") for name, x in sorted_items]
    print("Circuit Name & Qubit Count & Gate Count & " + " & ".join(METHOD_NAMES) + r"\\")
    if USE_ILP:
        ilpbest = [name for name in circuits_sorted if allfiles[name][2]["ilp"][0] != min(allfiles[name][2][method][0] for method in METHOD_NAMES if method != "ilp")]
        ilptie = [name for name in circuits_sorted if allfiles[name][2]["ilp"][0] == min(allfiles[name][2][method][0] for method in METHOD_NAMES if method != "ilp")]
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in ilpbest for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in ilptie for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
    else:
        print("\n".join(" & ".join((name.replace(".qasm", "").replace("_", r"\_"), str(allfiles[name][0]), str(allfiles[name][1]),
                                    *((r"\textbf{" if allfiles[name][2][method][0] == m else "") + str(allfiles[name][2][method][0]) + (r"}" if allfiles[name][2][method][0] == m else "") for method in METHOD_NAMES))) + r"\\" for name in circuits_sorted for m in (min(allfiles[name][2][method][0] for method in METHOD_NAMES),)))
    markers = ["o", "*", "D", "s", "+", "<", ">", "v", "^"]
    for j in range(4):
        title = ["Total Partitions", "Runtime Performance", "Total Float Ops for Fusion", "Total Float Ops for Control-Aware Fusion"][j]
        y_label = ["Partition Count", "Time in seconds", "Number of Float Ops", "Number of Float Ops"][j]
        for i, strat in enumerate(METHOD_NAMES):
            y = [allfiles[name][2][strat][j] for name in circuits_sorted]
            plt.plot(circuits_labels, y, marker=markers[i], label=strat)
        plt.xlabel("Circuit (sorted by qubits, gates)")
        plt.ylabel(y_label, fontsize=6)
        plt.title(f"{y_label} - Maximum k={max_qubits} Qubits per Partition")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{title.replace(' ', '_')}-{max_qubits}-max_qubit.svg", format="svg", transparent=True)
        plt.clf()

if __name__ == "__main__":
    for max_qubits in range(3, 6):
        test_partitions(max_qubits)
