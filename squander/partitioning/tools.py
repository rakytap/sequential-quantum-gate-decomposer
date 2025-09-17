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
    Get qubit indices used by a gate
    Args:
        
        gate: SQUANDER gate
    Returns:
        
        Set of qubit indices
    """
    return {gate.get_Target_Qbit()} | ({control} if (control := gate.get_Control_Qbit()) != -1 else set())

def get_float_ops(num_qubit, gate_qubits, control_qubits, is_pure=False, io_penalty=32):
    """
    Compute the number of floating-point operations (FLOPs) required 
    for simulating a quantum gate acting on a set of qubits.

    Args:
        num_qubit (int): Total number of qubits in the system.
        gate_qubits (int): Number of qubits the gate acts on (including controls).
        control_qubits (int): Number of control qubits for the gate.
        is_pure (bool, optional): Whether the gate is a "pure" controlled gate 
            (i.e., all controlled gates share the same target qubit). Defaults to False.

    Returns:
        int: Estimated number of floating-point operations required.
    """
    g_size = 2**(gate_qubits-control_qubits)
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i => 6 ops for 4m2a
    return 2**(num_qubit-(control_qubits if is_pure else 0)) * (g_size * (4 + 2) + 2 * (g_size - 1) + io_penalty)

def parts_to_float_ops(num_qubit, gate_to_qubit, gate_to_tqubit, allparts):
    """
    Compute FLOPs for each partition of gates in a quantum circuit.

    Args:
        num_qubit (int): Total number of qubits in the system.
        gate_to_qubit (dict): Mapping from gate ID to the set of qubits 
            it acts on.
        gate_to_tqubit (dict or None): Mapping from gate ID to its target qubit 
            (used for distinguishing control vs. target qubits). If None, 
            control qubits are assumed to be 0.
        allparts (list[list]): Partitioning of gates, where each part is a 
            collection (e.g., list or set) of gate IDs.

    Returns:
        list[int]: FLOP counts for each partition in `allparts`.
    """
    weights = []
    for part in allparts:
        qubits = set.union(*(gate_to_qubit[x] for x in part))
        if gate_to_tqubit is not None:
            tqubits = {gate_to_tqubit[x] for x in part}
            is_pure = len({frozenset(gate_to_qubit[x]-{gate_to_tqubit[x]}) for x in part}) == 1
            weights.append(get_float_ops(num_qubit, len(qubits), len(qubits)-len(tqubits), is_pure))
        else: weights.append(get_float_ops(num_qubit, len(qubits), 0, False))
    return weights

def total_float_ops(num_qubit, max_qubits_per_partition, gate_to_qubit, gate_to_tqubit, allparts):
    """
    Compute the total FLOPs across all partitions of a quantum circuit,
    scaled by the number of qubits outside the maximum partition.

    Args:
        num_qubit (int): Total number of qubits in the system.
        max_qubits_per_partition (int): Maximum number of qubits any partition 
            can act on.
        gate_to_qubit (dict): Mapping from gate ID to the set of qubits 
            it acts on.
        gate_to_tqubit (dict or None): Mapping from gate ID to its target qubit. 
            If None, control qubits are assumed to be 0.
        allparts (list[list]): Partitioning of gates, where each part is a 
            collection of gate IDs.

    Returns:
        int: Total number of floating-point operations across all partitions.
    """
    weights = parts_to_float_ops(max_qubits_per_partition, gate_to_qubit, gate_to_tqubit, allparts)
    return 2**(num_qubit-max_qubits_per_partition) * sum(weights)

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
    Build dependency graphs for circuit gates
    Args:
        
        c: SQUANDER Circuit.
    Returns:
        
        Gate dict, forward graph, reverse graph, qubit mapping, start set.
    """
    gate_dict = {i: gate for i, gate in enumerate(c.get_Gates())}
    gate_to_qubit = { i: get_qubits(g) for i, g in gate_dict.items() }
    g, rg = {i: set() for i in gate_dict}, {i: set() for i in gate_dict}
    
    for gate in gate_dict:
        for child in c.get_Children(gate_dict[gate]):
            g[gate].add(child)
            rg[child].add(gate)
    
    S = {m for m in rg if len(rg[m]) == 0}

    return gate_dict, g, rg, gate_to_qubit, S


def qiskit_to_squander_name(qiskit_name):
    """
    Convert Qiskit gate name to SQUANDER name
    Args:
        
        qiskit_name: Qiskit gate name
    Returns:
        
        SQUANDER gate name
    """
    name = qiskit_name.upper()
    if name == "CX":
        return "CNOT"
    elif name == "U":
        return "U3"
    elif name == "TDG":
        return "Tdg"
    else:
        return name

def gate_desc_to_gate_index(circ, preparts, qubit_groups_only=False):
    """
    Map gate descriptions to indices for partitioning
    Args:
        
        circ: SQUANDER Circuit
        
        preparts: Partition descriptions
    Returns:
        
        Partitioned gate indices
    """
    gate_dict, g, rg, gate_to_qubit, S = build_dependency(circ)
    L = []
    
    curr_partition = set()
    curr_idx = 0
    total = 0
    parts = [[]]

    while S:
        if qubit_groups_only:
            n = next(iter(x for x in S if gate_to_qubit[x] <= preparts[len(parts)-1]), None)
        else:
            Scomp = {(frozenset(gate_to_qubit[x]), gate_dict[x].get_Name()): x for x in S}
            rev_Scomp = { y: x for x, y in Scomp.items()}
            n = next(iter(Scomp.keys() & preparts[len(parts)-1]), None)
            if n is not None: n = Scomp[n]

        while n is None:
            total += len(parts[-1])
            curr_partition = set()
            parts.append([])
            if qubit_groups_only:
                n = next(iter(x for x in S if gate_to_qubit[x] <= preparts[len(parts)-1]), None)
            else:
                n = next(iter(Scomp.keys() & preparts[len(parts)-1]), None)
                if n is not None: n = Scomp[n]
            
        if not qubit_groups_only: preparts[len(parts)-1].remove(rev_Scomp[n])
        parts[-1].append(n)
        curr_partition |= gate_to_qubit[n]
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
    """
    Partition circuit using Qiskit multi-qubit blocks
    Args:
        
        filename: QASM file path
        
        max_partition_size: Max qubits per partition
    Returns:
        
        Parameters, partitioned circuit, parameter order (source_idx, dest_idx, param_count), partitions
    """
    circ, parameters, qc = utils.qasm_to_squander_circuit(filename, True)
    pm = PassManager([
        CollectMultiQBlocks(max_block_size=max_partition_size),
    ])

    pm.run(qc)
    blocks = pm.property_set['block_list'] # is not in topological order

    L = [[(frozenset(qc.find_bit(x)[0] for x in dagop.qargs), 
           qiskit_to_squander_name(dagop.name)) for dagop in block] for block in blocks]
    #L = [frozenset({qc.find_bit(x)[0] for dagop in block for x in dagop.qargs}) for block in blocks]
    assert len(qc.data) == sum(map(len, blocks))
    from squander.partitioning.kahn import kahn_partition_preparts
    partitioned_circ, param_order, parts = kahn_partition_preparts(circ, max_partition_size, gate_desc_to_gate_index(circ, L))
    return parameters, partitioned_circ, param_order, parts

def get_qiskit_fusion_partitions(filename, max_partition_size):
    """
    Generate circuit partitions from a QASM file using Qiskit's fusion 
    metadata and Squander's partitioning utilities.

    This function:
      1. Parses the QASM file into a Squander circuit.
      2. Runs the circuit on Qiskit's AerSimulator with gate fusion enabled.
      3. Extracts the fusion metadata (grouped qubit operations).
      4. Uses Squander's Kahn-based partitioning to produce circuit partitions.

    Args:
        filename (str): Path to the QASM file to be parsed.
        max_partition_size (int): Maximum number of qubits allowed in each 
            fusion partition (i.e., `fusion_max_qubit` for Qiskit).

    Returns:
        tuple:
            parameters (list): Parameter list extracted from the Squander circuit.
            partitioned_circ (object): Partitioned Squander circuit object.
            param_order (list): Order of parameters corresponding to the partitioned circuit.
            parts (list[list]): Partitioning of the circuit into gate groups, 
                represented as lists of gate indices.
    """
    circ, parameters, qc = utils.qasm_to_squander_circuit(filename, True)
    qc.save_statevector()
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method="statevector", fusion_enable=True, fusion_verbose=True, fusion_max_qubit=max_partition_size, fusion_threshold=1, shots=1)
    tcirc = transpile(qc, backend=backend, optimization_level=0)
    job = backend.run(tcirc, shots=1)
    res = job.result()
    meta = res.results[0].metadata.get("fusion", {})
    qubits = [frozenset(x["qubits"]) for x in meta["output_ops"][:-1]] #could try to determine control qubits by looking 
    from squander.partitioning.kahn import kahn_partition_preparts
    partitioned_circ, param_order, parts = kahn_partition_preparts(circ, max_partition_size, gate_desc_to_gate_index(circ, qubits, qubit_groups_only=True))
    return parameters, partitioned_circ, param_order, parts


def get_bqskit_partitions(filename, max_partition_size, partitioner):
    """
    Partition circuit using BQSKit partitioners
    Args:
        
        filename: QASM file path
        
        max_partition_size: Max qubits per partition
        
        partitioner: BQSKit Partitioning strategy
    Returns:
        
        Parameters, partitioned circuit, parameter order (source_idx, dest_idx, param_count), partitions
    """
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
            bqskit.ir.gates.constant.s.SGate: "S",
            bqskit.ir.gates.constant.sdg.SdgGate: "Sdg",
            bqskit.ir.gates.constant.r.RGate: "R",
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
