from squander.partitioning.partition import PartitionCircuitQasm
from qiskit import QuantumCircuit

import timeit
import glob
import os

USE_ILP = False

MAX_GATES_ALLOWED = 1024

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

def test_partitions():
    max_partition_size = 4
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print("Total QASM:", len(files))
    allfiles = {}
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        num_gates = len(qc.data)
        fname = os.path.basename(filename)      
        if num_gates > MAX_GATES_ALLOWED:
            print(f"Skipping {fname}; qubits {qc.num_qubits} gates {num_gates}")
            continue
        print(fname)
        res = {}
        for method in METHOD_NAMES:
            ls = []
            def f():
                ls.extend(PartitionCircuitQasm( filename, max_partition_size, method ))
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
    markers = ["o", "*", "D", "s", "+", "<", ">", "v", "^"]
    for i, strat in enumerate(METHOD_NAMES):
        y = [allfiles[name][2][strat][0] for name in circuits_sorted]
        plt.plot(circuits_sorted, y, marker=markers[i], label=strat)
    plt.xlabel("Circuit (sorted by qubits, gates)")
    plt.ylabel("Partition Count")
    plt.title("Partition Count per Circuit by Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("partition_counts.svg", format="svg", transparent=True)

if __name__ == "__main__":
    test_partitions()
