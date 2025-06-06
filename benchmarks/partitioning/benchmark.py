from squander.partitioning.partition import qasm_to_partitioned_circuit
import glob
import os
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CollectMultiQBlocks, ConsolidateBlocks

import asyncio
from bqskit import Circuit
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner #too slow
from bqskit.passes.partitioning.cluster import ClusteringPartitioner #does a bad job at minimizing partitions
from bqskit.passes.partitioning.scan import ScanPartitioner

async def test_partitions():
    allowed_gates = {'u', 'u3', 'cx', 'cry', 'cz', 'ch', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z', 'sx'}
    max_partition_size = 4
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print("Total QASM:", len(files))
    allfiles = {}
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        num_gates = len(qc.data)
        fname = os.path.basename(filename)      
        print(f"{fname} qubits {qc.num_qubits} gates {num_gates}")
        if num_gates > 4096:
            continue
        res = {}
        partitioned_circuit, parameters = qasm_to_partitioned_circuit( filename, max_partition_size )
        res["Greedy"] = len(partitioned_circuit.get_Gates())
        print("ILP")
        partitioned_circuit_ilp, parameters_ilp = qasm_to_partitioned_circuit( filename, max_partition_size, True )
        res["ILP"] = len(partitioned_circuit_ilp.get_Gates())
        pm = PassManager([
            CollectMultiQBlocks(max_block_size=max_partition_size),
            ConsolidateBlocks(basis_gates=allowed_gates, approximation_degree=0.0, #approximation_degree (float) – a float between [0.0,1.0]. Lower approximates more.
                            force_consolidate=True)
        ])
        try:
            optimized_circuit = pm.run(qc)
        except Exception as e:
            print(fname, e)
            continue
        res["Qiskit"] = len(optimized_circuit.data)
        for name, partitioner in {"BQSKit-Quick": QuickPartitioner(block_size=max_partition_size),
                            "BQSKit-Scan": ScanPartitioner(block_size=max_partition_size), "BQSKit-Greedy": GreedyPartitioner(block_size=max_partition_size),
                            "BQSKit-Cluster": ClusteringPartitioner(block_size=max_partition_size)}.items():
            if name not in {"BQSKit-Quick"}: continue
            bq_circuit = Circuit.from_file(filename)
            #print("Gates:", len(bq_circuit))
            # Apply partitioning (returns a DAG of blocks)
            print(name)
            await partitioner.run(bq_circuit, None)
            # Count number of blocks (partitions)
            num_partitions = sum(bq_circuit.gate_counts[x] for x in bq_circuit.gate_counts)
            res[name] = num_partitions
        allfiles[fname] = (qc.num_qubits, num_gates, res)
        print(fname, allfiles[fname])
    print(allfiles)
    import matplotlib.pyplot as plt
    sorted_items = sorted(allfiles.items(), key=lambda item: (item[1][0], item[1][1]))
    circuits_sorted = [name for name, _ in sorted_items]
    markers = ["o", "*", "D", "s"]
    for i, strat in enumerate(("Greedy", "ILP",""
                  "Qiskit", "BQSKit-Quick")):
        y = [allfiles[name][2][strat] for name in circuits_sorted]
        plt.plot(circuits_sorted, y, marker=markers[i], label=strat)
    plt.xlabel("Circuit (sorted by qubits, gates)")
    plt.ylabel("Partition Count")
    plt.title("Partition Count per Circuit by Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("partition_counts.svg", format="svg")

if __name__ == "__main__":
    asyncio.run(test_partitions())
