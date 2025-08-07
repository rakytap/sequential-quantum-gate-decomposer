from typing import Set, List, Tuple, Dict
import numpy as np

from squander import utils
from squander.gates.gates_Wrapper import Gate
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

import asyncio
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CollectMultiQBlocks

def get_qubits(gate: Gate) -> Set[int]:
    """
    Retrieves qubit indices used by a SQUANDER gate.
    
    Args:
        gate: The SQUANDER gate
        
    Returns:
        Set of qubit indices used by the gate
    """
    return {gate.get_Target_Qbit()} | ({control} if (control := gate.get_Control_Qbit()) != -1 else set())



def translate_param_order(params: np.ndarray, param_order: List[Tuple[int,int]]) -> np.ndarray:
    """
    Call to reorder circuit parameters based on partitioned execution order
    
    Args:

        params ( np.ndarray ) Original parameter array

        param_order (List[int] ) Tuples specifying new parameter positions: source_idx, dest_idx, param_count
    
    Return:

        Returns with the reordered Reordered parameter array
    """ 
    reordered = np.empty_like(params)

    for s_idx, n_idx, n_params in param_order:
        reordered[n_idx:n_idx + n_params] = params[s_idx:s_idx + n_params]

    return reordered



def build_dependency(c: Circuit) -> Tuple[Dict[int, Gate], Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Build dependency graphs for circuit gates.
    
    Args:
        c: SQUANDER Circuit
        
    Returns:
        tuple: (gate_dict, forward_graph, reverse_graph)
    """
    gate_dict = {i: gate for i, gate in enumerate(c.get_Gates())}
    g, rg = {i: set() for i in gate_dict}, {i: set() for i in gate_dict}
    
    for gate in gate_dict:
        for child in c.get_Children(gate_dict[gate]):
            g[gate].add(child)
            rg[child].add(gate)
    
    return gate_dict, g, rg


def qiskit_to_squander_name(qiskit_name):
    name = qiskit_name.upper()
    if name == "CX":
        return "CNOT"
    elif name == "U":
        return "U3"
    elif name == "TDG":
        return "Tdg"
    else:
        return name

def gate_desc_to_gate_index(circ, preparts):
    gate_dict, g, rg = build_dependency(circ)
    L, S = [], {m for m in rg if len(rg[m]) == 0}
    
    curr_partition = set()
    curr_idx = 0
    total = 0
    parts = [[]]

    while S:
        Scomp = {(frozenset(get_qubits(gate_dict[x])), gate_dict[x].get_Name()): x for x in S}
        rev_Scomp = { y: x for x, y in Scomp.items()}
        n = next(iter(Scomp.keys() & preparts[len(parts)-1]), None)
        if n is not None: n = Scomp[n]

        if n is None:
            total += len(parts[-1])
            curr_partition = set()
            parts.append([])
            n = next(iter(Scomp.keys() & preparts[len(parts)-1]), None)
            if n is not None: n = Scomp[n]
        
        preparts[len(parts)-1].remove(rev_Scomp[n])
        parts[-1].append(n)
        curr_partition |= get_qubits(gate_dict[n])
        curr_idx += gate_dict[n].get_Parameter_Num()

        # Update dependencies
        L.append(n)
        S.remove(n)
        assert len(rg[n]) == 0
        for child in set(g[n]):
            g[n].remove(child)
            rg[child].remove(n)
            if not rg[child]:
                S.add(child)

    # Add the last partition
    total += len(parts[-1])
    assert total == len(gate_dict)
    # print(parts)
    return parts

def get_qiskit_partitions(filename, max_partition_size):
    circ, parameters, qc = utils.qasm_to_squander_circuit(filename, True)
    pm = PassManager([
        CollectMultiQBlocks(max_block_size=max_partition_size),
    ])

    pm.run(qc)
    blocks = pm.property_set['block_list'] # is not in topological order

    L = [[(frozenset(qc.find_bit(x)[0] for x in dagop.qargs), 
           qiskit_to_squander_name(dagop.name)) for dagop in block] for block in blocks]
    assert len(qc.data) == sum(map(len, blocks))
    from squander.partitioning.kahn import kahn_partition_preparts
    partitioned_circ, param_order, parts = kahn_partition_preparts(circ, max_partition_size, gate_desc_to_gate_index(circ, L))
    return parameters, partitioned_circ, param_order, parts


def get_bqskit_partitions(filename, max_partition_size, partitioner):
    try:
        from bqskit import Circuit
        from bqskit.passes.partitioning.quick import QuickPartitioner
        from bqskit.passes.partitioning.greedy import GreedyPartitioner #too slow
        from bqskit.passes.partitioning.cluster import ClusteringPartitioner #does a bad job at minimizing partitions
        from bqskit.passes.partitioning.scan import ScanPartitioner

        import bqskit.ir

        bqs_to_squander = {
            bqskit.ir.gates.constant.cx.CNOTGate: "CNOT",
            bqskit.ir.gates.constant.t.TGate: "T",
            bqskit.ir.gates.constant.h.HGate: "H",
            bqskit.ir.gates.constant.tdg.TdgGate: "Tdg",
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
            bqskit.ir.gates.parameterized.rx.RXGate: "RX",
        }
    
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
        L = [[(frozenset(curloc.location[x] for x in op.location), bqs_to_squander[type(op.gate)]) for op in curloc.gate._circuit.operations()] for curloc in bq_circuit.operations()]
        from squander.partitioning.kahn import kahn_partition_preparts
        partitioned_circ, param_order, parts = kahn_partition_preparts(circ, max_partition_size, gate_desc_to_gate_index(circ, L))
        return parameters, partitioned_circ, param_order, parts

    except ImportError as e:
        raise ImportError(
            f"bqskit is not installed: bqskit is required for bqskit-{partitioner} partitioning."
        ) from e
