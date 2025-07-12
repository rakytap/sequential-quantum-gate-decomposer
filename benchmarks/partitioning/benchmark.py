from squander.partitioning.partition import PartitionCircuitQasm, kahn_partition, translate_param_order
import glob
import os
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CollectMultiQBlocks, ConsolidateBlocks
from squander import utils

import asyncio
from bqskit import Circuit
import bqskit.ir
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner #too slow
from bqskit.passes.partitioning.cluster import ClusteringPartitioner #does a bad job at minimizing partitions
from bqskit.passes.partitioning.scan import ScanPartitioner

bqs_to_squander = {bqskit.ir.gates.constant.cx.CNOTGate: "CNOT",
                   bqskit.ir.gates.constant.t.TGate: "T",
                   bqskit.ir.gates.constant.h.HGate: "H",
                   bqskit.ir.gates.constant.tdg.TdgGate: "TDG",
                   bqskit.ir.gates.constant.x.XGate: "X",
                   bqskit.ir.gates.constant.y.YGate: "Y",
                   bqskit.ir.gates.constant.z.ZGate: "Z",
                   bqskit.ir.gates.constant.sx.SXGate: "SX",
                   bqskit.ir.gates.constant.ch.CHGate: "CH",
                   bqskit.ir.gates.constant.cz.CZGate: "CZ",
                   bqskit.ir.gates.parameterized.cry.CRYGate: "CRY",
                   bqskit.ir.gates.parameterized.u3.U3Gate: "U3",
                   bqskit.ir.gates.parameterized.rz.RZGate: "RZ",
                   bqskit.ir.gates.parameterized.ry.RYGate: "RY",
                   bqskit.ir.gates.parameterized.rx.RXGate: "RX",}

def get_qiskit_partitions(circ, qc, max_partition_size):
    pm = PassManager([
        CollectMultiQBlocks(max_block_size=max_partition_size),
    ])
    pm.run(qc)
    blocks = pm.property_set['block_list']
    L = [{(frozenset(qc.find_bit(x)[0] for x in dagop.qargs), dagop.name.upper().replace("CX", "CNOT")) for dagop in block} for block in blocks]
    assert len(qc.data) == sum(map(len, blocks))
    return kahn_partition(circ, max_partition_size, L)

def do_get_qiskit_partitions(filename, max_partition_size):
    circ, parameters, qc = utils.qasm_to_squander_circuit(filename, True)
    partitioned_circ, param_order, _ = get_qiskit_partitions(circ, qc, max_partition_size)
    param_reordered = translate_param_order(parameters, param_order)
    return partitioned_circ, param_reordered

def get_bqskit_partitions(filename, partitioner, max_partition_size):
    if partitioner == "Quick":
        partitioner = QuickPartitioner(block_size=max_partition_size)
    elif partitioner == "Scan":
        partitioner = ScanPartitioner(block_size=max_partition_size)
    elif partitioner == "Greedy":
        partitioner = GreedyPartitioner(block_size=max_partition_size)
    elif partitioner == "Cluster":
        partitioner = ClusteringPartitioner(block_size=max_partition_size)
    bq_circuit = Circuit.from_file(filename)
    asyncio.run(partitioner.run(bq_circuit, None))
    # Count number of blocks (partitions)
    circ, parameters, qc = utils.qasm_to_squander_circuit(filename, True)
    L = [{(frozenset(curloc.location[x] for x in op.location), bqs_to_squander[type(op.gate)]) for op in curloc.gate._circuit.operations()} for curloc in bq_circuit.operations()]
    return kahn_partition(circ, max_partition_size, L)
def test_partitions():
    max_partition_size = 4
    files = glob.glob("benchmarks/partitioning/test_circuit/*.qasm")
    print("Total QASM:", len(files))
    allfiles = {}
    for filename in files:
        qc = QuantumCircuit.from_qasm_file(filename)
        num_gates = len(qc.data)
        fname = os.path.basename(filename)      
        print(f"{fname} qubits {qc.num_qubits} gates {num_gates}")
        if num_gates > 1024:
            continue
        res = {}
        partitioned_circuit, parameters = PartitionCircuitQasm( filename, max_partition_size )
        res["Greedy"] = len(partitioned_circuit.get_Gates())
        print("ILP")
        partitioned_circuit_ilp, parameters_ilp = PartitionCircuitQasm( filename, max_partition_size, True )
        res["ILP"] = len(partitioned_circuit_ilp.get_Gates())
        res["Qiskit"] = do_get_qiskit_partitions(filename, max_partition_size)
        for name in ("Quick", "Greedy", "Scan", "Cluster"):
            print(name)
            res[f"BQSKit-{name}"] = get_bqskit_partitions(filename, name, max_partition_size)
        allfiles[fname] = (qc.num_qubits, num_gates, res)
        print(fname, allfiles[fname])
    print(allfiles)
    import matplotlib.pyplot as plt
    sorted_items = sorted(allfiles.items(), key=lambda item: (item[1][0], item[1][1]))
    circuits_sorted = [name for name, _ in sorted_items]
    markers = ["o", "*", "D", "s"]
    for i, strat in enumerate(("Greedy","ILP","Qiskit", "BQSKit-Quick")):
        y = [allfiles[name][2][strat] for name in circuits_sorted]
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
