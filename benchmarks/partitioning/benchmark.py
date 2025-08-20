from squander.partitioning.partition import PartitionCircuitQasm
from qiskit import QuantumCircuit
from squander.partitioning.tools import qiskit_to_squander_name

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
    "bqskit-Quick",
    # "bqskit-Greedy", 
    # "bqskit-Scan",
    # "bqskit-Cluster", 
] + (["ilp"] if USE_ILP else [])

from squander.gates import gates_Wrapper as gate
SUPPORTED_GATES = {x for n in dir(gate) for x in (getattr(gate, n),) if not n.startswith("_") and issubclass(x, gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES = {n for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES.remove("S")

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
        print(fname, qc.num_qubits, num_gates, f"k={max_qubits}")
        res = {}
        for method in METHOD_NAMES:
            print(method)
            ls = []
            def f():
                ls.extend(PartitionCircuitQasm( filename, max_qubits, method ))
            t = timeit.timeit(f, number=1)
            partitioned_circuit, parameters = ls
            res[method] = len(partitioned_circuit.get_Gates()), t
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
    for perf in (True, False):
        title = "Partition Count" if not perf else "Runtime Performance"
        y_label = "Partition Count" if not perf else "Time in seconds"
        for i, strat in enumerate(METHOD_NAMES):
            y = [allfiles[name][2][strat][1 if perf else 0] for name in circuits_sorted]
            plt.plot(circuits_labels, y, marker=markers[i], label=strat)
        plt.xlabel("Circuit (sorted by qubits, gates)")
        plt.ylabel(y_label, fontsize=6)
        plt.title(f"{y_label} - Maximum k={max_qubits} Qubits per Circuit")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{title.replace(' ', '_')}-{max_qubits}-max_qubit.svg", format="svg", transparent=True)
        plt.clf()

if __name__ == "__main__":
    for max_qubits in range(3, 6):
        test_partitions(max_qubits)
