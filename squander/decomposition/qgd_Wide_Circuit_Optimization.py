"""
Wide-circuit optimization: partition large circuits into subcircuits, re-decompose
them, and optionally route or fuse results according to configuration.
"""

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander import N_Qubit_Decomposition_custom, N_Qubit_Decomposition
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits

import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable, Tuple, Optional, Set, Dict, Any, cast, Union

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os, contextlib, collections, time


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import translate_param_order, build_dependency
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE


def extract_subtopology(involved_qbits, qbit_map, config):
    """Return topology edges restricted to ``involved_qbits``, with indices remapped via ``qbit_map``.

    Args:
        involved_qbits: Qubit labels present in a partition.
        qbit_map: Maps original qubit index to local index (0..n-1).
        config: Configuration dict containing ``topology`` as a list of edges.

    Returns:
        List of ``(u, v)`` pairs in local indices, each edge fully inside the partition.
    """
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]], qbit_map[edge[1]]))
    return mini_topology


CNOT_COUNT_DICT = {
    "CNOT": 1,
    "CH": 1,
    "CZ": 1,
    "SYC": 3,
    "CRY": 2,
    "CU": 2,
    "CR": 2,
    "CROT": 2,
    "CRX": 2,
    "CRZ": 2,
    "CP": 2,
    "CCX": 6,
    "CSWAP": 7,
    "SWAP": 3,
    "RXX": 2,
    "RYY": 2,
    "RZZ": 2,
}


def CNOTGateCount(circ: Circuit, max_gates: int = 0) -> int:
    """Compute weighted two-qubit gate count for a circuit.

    The base count is the CNOT-equivalent cost derived from ``CNOT_COUNT_DICT``.
    When ``max_gates > 0``, the function returns a lexicographic-style score:
    ``two_qubit_cost * max_gates + single_qubit_gate_count``.

    Args:
        circ: Squander circuit representation.
        max_gates: Weight multiplier for the two-qubit cost term.

    Returns:
        Integer gate-cost score used by optimization heuristics.
    """

    assert isinstance(
        circ, Circuit
    ), "The input parameters should be an instance of Squander Circuit"

    gate_counts = circ.get_Gate_Nums()
    num_cnots = sum(
        CNOT_COUNT_DICT.get(gate, 0) * count for gate, count in gate_counts.items()
    )

    if max_gates > 0:
        return num_cnots * max_gates + sum(
            y for x, y in gate_counts.items() if x not in CNOT_COUNT_DICT
        )
    return num_cnots

def _topology_le_to_be(n_qubits, topology):
    """Convert a topology from squander LE convention to bqskit BE convention."""
    return [(n_qubits - 1 - i, n_qubits - 1 - j) for i, j in topology]


def generate_squander_seqpam(squander_config, block_size):
    """Build a bqskit SeqPAM workflow using Squander as the inner synthesis engine with ILP partitioning.

    Args:
        squander_config: Config dict passed to SquanderSynthesisPass (bqskit-squander keys:
            ``strategy`` ("Tree_search"/"Tabu_search"), ``verbosity``,
            ``optimization_tolerance``, ``optimizer_engine``, etc.).
        block_size: Maximum block size for ILP partitioning and SubtopologySelectionPass.

    Returns:
        bqskit Workflow implementing the two-stage permutation-aware mapping.
    """
    from bqskit.passes import (
        SquanderSynthesisPass,
        ForEachBlockPass,
        EmbedAllPermutationsPass,
        PAMRoutingPass,
        PAMLayoutPass,
        PAMVerificationSequence,
        SubtopologySelectionPass,
        ApplyPlacement,
        UnfoldPass,
        ExtractModelConnectivityPass,
        RestoreModelConnectivityPass,
        LogPass,
    )
    from bqskit.passes.control import IfThenElsePass
    from bqskit.passes.control.predicates import NotPredicate, WidthPredicate
    from bqskit.compiler import Workflow, BasePass

    class SquanderILPPartitioner(BasePass):
        """Partition a bqskit circuit using squander's ILP."""

        def __init__(self, block_size):
            self.block_size = block_size

        async def run(self, circuit, data):
            from bqskit.ir import Circuit as BQCircuit
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import QuantumCircuit as QkCircuit, qasm2 as qasm2_module
            from squander import Qiskit_IO
            from squander.partitioning.ilp import (
                get_all_partitions, _get_topo_order, ilp_global_optimal,
            )

            # Unfold any CircuitGate blocks (e.g. from a prior SubtopologySelectionPass)
            # so that bqskit op indices align 1:1 with squander gate indices after the
            # QASM roundtrip.  unfold_all() is a no-op on already-flat circuits.
            flat_circuit = circuit.copy()
            flat_circuit.unfold_all()

            qasm_str = OPENQASM2Language().encode(flat_circuit)
            qk_circ = QkCircuit.from_qasm_str(qasm_str)
            sqdr_circ, _ = Qiskit_IO.convert_Qiskit_to_Squander(qk_circ)

            allparts, g, go, rgo, sq_chains, gate_to_qubit, _ = \
                get_all_partitions(sqdr_circ, self.block_size)
            gate_dict = {i: gate for i, gate in enumerate(sqdr_circ.get_Gates())}

            L_parts, _ = ilp_global_optimal(allparts, g)

            bqskit_ops = list(flat_circuit.operations_with_cycles())

            sqc_pre     = {x[0]: x for x in sq_chains if rgo[x[0]]}
            sqc_post    = {x[-1]: x for x in sq_chains if go[x[-1]]}
            sqc_prepost = {x[0]: x for x in sq_chains
                           if x[0] in sqc_pre and x[-1] in sqc_post}

            # Build expanded gate_idxs per ILP partition (include surrounding 1q gates)
            expanded = {}
            for i in L_parts:
                part = allparts[i]
                surrounded = {
                    t for s in part for t in go[s]
                    if t in sqc_prepost
                    and go[sqc_prepost[t][-1]]
                    and next(iter(go[sqc_prepost[t][-1]])) in part
                }
                gate_idxs = frozenset.union(part, *(sqc_prepost[v] for v in surrounded))
                expanded[i] = gate_idxs

            # Further expand: include ALL intermediate gates on partition qubits
            for i in L_parts:
                gate_idxs = expanded[i]
                part_qubits = set()
                for gi in gate_idxs:
                    part_qubits.update(gate_dict[gi].get_Involved_Qbits())
                lo = min(gate_idxs)
                hi = max(gate_idxs)
                extra = set()
                for gi in range(lo, hi + 1):
                    if gi not in gate_idxs:
                        gq = set(gate_dict[gi].get_Involved_Qbits())
                        if gq & part_qubits:
                            extra.add(gi)
                if extra:
                    expanded[i] = gate_idxs | frozenset(extra)

            # Sort partitions by their minimum gate index to preserve original order
            seen_parts = set()
            sorted_parts = []
            for i in L_parts:
                gate_idxs = expanded[i]
                part_key = min(gate_idxs)
                if part_key not in seen_parts:
                    seen_parts.add(part_key)
                    sorted_parts.append((part_key, gate_idxs))
            sorted_parts.sort(key=lambda x: x[0])

            # Map gate_idx -> sorted partition index
            gate_to_part = {}
            for pidx, (_, gate_idxs) in enumerate(sorted_parts):
                for gi in gate_idxs:
                    gate_to_part[gi] = pidx

            # Build partitioned circuit by iterating gates in original order
            partitioned = BQCircuit(circuit.num_qudits, circuit.radixes)
            built_parts = set()

            for gi, (_, op) in enumerate(bqskit_ops):
                pidx = gate_to_part.get(gi, -1)

                if pidx >= 0 and pidx not in built_parts:
                    built_parts.add(pidx)
                    _, gate_idxs = sorted_parts[pidx]
                    global_qudits = sorted({
                        q for ggi in gate_idxs
                        for q in gate_dict[ggi].get_Involved_Qbits()
                    })
                    local_map = {gq: l for l, gq in enumerate(global_qudits)}

                    topo = _get_topo_order(
                        {x: go[x] & gate_idxs for x in gate_idxs},
                        {x: rgo[x] & gate_idxs for x in gate_idxs},
                        gate_to_qubit,
                    )
                    sub = BQCircuit(len(global_qudits))
                    for ggi in topo:
                        _, gop = bqskit_ops[ggi]
                        sub.append_gate(gop.gate, [local_map[q] for q in gop.location], gop.params)
                    partitioned.append_circuit(sub, global_qudits, as_circuit_gate=True)

                elif pidx < 0:
                    sub_1q = BQCircuit(1)
                    sub_1q.append_gate(op.gate, [0], op.params)
                    partitioned.append_circuit(sub_1q, list(op.location), as_circuit_gate=True)

            circuit.become(partitioned, False)

    class SetPAMInitialPlacementPass(BasePass):
        """Set the placement used as the starting point for the final PAM layout."""

        def __init__(self, placement):
            self.placement = None if placement is None else list(placement)

        async def run(self, circuit, data):
            if self.placement is None:
                return
            if len(self.placement) != circuit.num_qudits:
                raise ValueError(
                    "PAM initial placement length must match circuit width."
                )
            data.placement = list(self.placement)

    squander    = SquanderSynthesisPass(squander_config=squander_config)
    partitioner = SquanderILPPartitioner(block_size)
    post_pam_seq: BasePass = PAMVerificationSequence(8)
    num_layout_passes = int(squander_config.get("num_layout_passes", 100))
    pam_initial_placement = squander_config.get("pam_initial_placement", None)

    return Workflow(
        IfThenElsePass(
            NotPredicate(WidthPredicate(2)),
            [
                LogPass("Caching permutation-aware synthesis results."),
                ExtractModelConnectivityPass(),
                partitioner,
                ForEachBlockPass(
                    EmbedAllPermutationsPass(
                        inner_synthesis=squander,
                        input_perm=True,
                        output_perm=False,
                        vary_topology=False,
                    ),
                ),
                LogPass("Preoptimizing with permutation-aware mapping."),
                PAMRoutingPass(),
                post_pam_seq,
                UnfoldPass(),
                RestoreModelConnectivityPass(),
                LogPass("Recaching permutation-aware synthesis results."),
                SubtopologySelectionPass(block_size),
                partitioner,
                ForEachBlockPass(
                    EmbedAllPermutationsPass(
                        inner_synthesis=squander,
                        input_perm=False,
                        output_perm=True,
                        vary_topology=True,
                    ),
                ),
                LogPass("Performing permutation-aware mapping."),
                ApplyPlacement(),
                SetPAMInitialPlacementPass(pam_initial_placement),
                PAMLayoutPass(num_layout_passes),
                PAMRoutingPass(0.1),
                post_pam_seq,
                ApplyPlacement(),
                UnfoldPass(),
            ],
        ),
        name="SeqPAM Mapping",
    )


class qgd_Wide_Circuit_Optimization:
    """Optimize wide (many-qubit) circuits via partitioning and subcircuit decomposition.

    Supports multiple decomposition strategies, optional global recombination (ILP),
    and routing when the circuit does not match the target topology.

    """

    def __init__(self, config):
        """Validate and store wide-circuit optimization ``config`` (strategy, topology, partitioning, tolerances)."""

        config.setdefault("strategy", "TreeSearch")
        config.setdefault("parallel", 0)
        config.setdefault("verbosity", 0)
        config.setdefault("tolerance", 1e-8)
        config.setdefault("test_subcircuits", False)
        config.setdefault("test_final_circuit", True)
        config.setdefault("max_partition_size", 3)
        config.setdefault("topology", None)
        config.setdefault("partition_strategy", "ilp")

        # testing the fields of config
        strategy = config["strategy"]
        allowed_startegies = [
            "TreeSearch",
            "TabuSearch",
            "Adaptive",
            "TreeGuided",
            "qiskit",
            "bqskit",
            "seqpam_PartAM",
        ]
        if not strategy in allowed_startegies:
            raise Exception(
                f"The decomposition startegy should be either of {allowed_startegies}, got {strategy}."
            )

        parallel = config["parallel"]
        allowed_parallel = [0, 1, 2]
        if not parallel in allowed_parallel:
            raise Exception(
                f"The parallel configuration should be either of {allowed_parallel}, got {parallel}."
            )

        verbosity = config["verbosity"]
        if not isinstance(verbosity, int):
            raise Exception(f"The verbosity parameter should be an integer.")

        tolerance = config["tolerance"]
        if not isinstance(tolerance, float):
            raise Exception(f"The tolerance parameter should be a float.")

        test_subcircuits = config["test_subcircuits"]
        if not isinstance(test_subcircuits, bool):
            raise Exception(f"The test_subcircuits parameter should be a bool.")

        test_final_circuit = config["test_final_circuit"]
        if not isinstance(test_final_circuit, bool):
            raise Exception(f"The test_final_circuit parameter should be a bool.")

        max_partition_size = config["max_partition_size"]
        if not isinstance(max_partition_size, int):
            raise Exception(f"The max_partition_size parameter should be an integer.")

        self.config = config

        self.max_partition_size = max_partition_size

    def ConstructCircuitFromPartitions(
        self, circs: List[Circuit], parameter_arrs: List[List[np.ndarray]]
    ) -> Tuple[Circuit, np.ndarray]:
        """Concatenate optimized partition circuits into a single wide circuit.

        Args:
            circs: Partition circuits in execution order.
            parameter_arrs: Parameter arrays corresponding to ``circs``.

        Returns:
            Tuple of ``(wide_circuit, wide_parameters)``.
        """

        if not isinstance(circs, list):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance(parameter_arrs, list):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs):
            raise Exception("The first two arguments should be of the same length")

        qbit_num = circs[0].get_Qbit_Num()

        wide_parameters = np.concatenate(parameter_arrs, axis=0)

        wide_circuit = Circuit(qbit_num)

        for circ in circs:
            wide_circuit.add_Circuit(circ)

        assert (
            wide_circuit.get_Parameter_Num() == wide_parameters.size
        ), f"Mismatch in the number of parameters: {wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"

        return wide_circuit, wide_parameters

    @staticmethod
    def DecomposePartition(
        Umtx: np.ndarray, config: dict, mini_topology=None, structure=None
    ) -> list[tuple[Circuit, np.ndarray]]:
        """
        Decompose a unitary ``Umtx`` (e.g. from a partition) using ``config['strategy']``.

        Args:
            Umtx: Complex unitary matrix.
            config: Must include ``strategy``, ``tolerance``, ``verbosity``, etc.
            mini_topology: Optional hardware couplers for topology-aware decomposers.
            structure: Required gate structure when ``strategy == "Custom"``.

        Returns:
            List of ``(squander_circuit, parameters)`` on success, or ``[]`` if error exceeds tolerance.
        """
        strategy = config["strategy"]
        if strategy == "TreeSearch":
            cDecompose = N_Qubit_Decomposition_Tree_Search(
                Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology
            )
        elif strategy == "TabuSearch":
            cDecompose = N_Qubit_Decomposition_Tabu_Search(
                Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology
            )
        elif strategy == "Adaptive":
            cDecompose = N_Qubit_Decomposition_adaptive(
                Umtx.conj().T,
                level_limit_max=5,
                level_limit_min=1,
                topology=mini_topology,
            )
        elif strategy == "Custom":
            cDecompose = N_Qubit_Decomposition_custom(
                Umtx.conj().T, config=config, accelerator_num=0
            )
            assert (
                structure is not None
            ), "Custom decomposition strategy requires a gate structure to be provided."
            cDecompose.set_Gate_Structure(structure)
        else:
            raise Exception(f"Unsupported decomposition type: {strategy}")

        tolerance = config["tolerance"]
        cDecompose.set_Verbose(config["verbosity"])
        cDecompose.set_Cost_Function_Variant(3)
        cDecompose.set_Optimization_Tolerance(tolerance)

        # adding new layer to the decomposition until threshold
        cDecompose.set_Optimizer("BFGS")

        # starting the decomposition
        try:
            cDecompose.Start_Decomposition()
        except Exception as e:
            # print(e)
            raise e
            # return []
        if not config.get("stop_first_solution", True):
            return cDecompose.all_solutions

        squander_circuit = cDecompose.get_Circuit()
        parameters = cDecompose.get_Optimized_Parameters()
        assert parameters is not None

        if strategy == "Custom":
            err = cDecompose.Optimization_Problem(parameters)
            it = 0
            while err > tolerance and it < 20:
                cDecompose.set_Optimized_Parameters(
                    np.random.rand(cDecompose.get_Parameter_Num()) * (2 * np.pi)
                )
                cDecompose.Start_Decomposition()
                parameters = cDecompose.get_Optimized_Parameters()
                err = cDecompose.Optimization_Problem(parameters)
                it += 1
            if (err > tolerance or it != 0) and config.get("verbosity", 0) >= 1:
                print("Decomposition error: ", err, it)
        else:
            err = cDecompose.get_Decomposition_Error()
        # print( "Decomposition error: ", err )
        if tolerance < err:
            # raise Exception(f"Decomposition error {err} exceeds the tolerance {tolerance}.")
            return []

        return [(squander_circuit, parameters)]

    @staticmethod
    def CompareAndPickCircuits(
        circs: List[Circuit],
        parameter_arrs: List[np.ndarray],
        metric: Callable[[Circuit], int] = CNOTGateCount,
    ) -> tuple[Circuit, np.ndarray]:
        """
        Call to pick the most optimal circuit corresponding a specific metric. Looks for the circuit
        with the minimal metric value.


        Args:

            circs ( List[Circuit] ) A list of Squander circuits to be compared

            parameter_arrs ( List[np.ndarray] ) A list of parameter arrays associated with the sqaunder circuits

            metric (optional) The metric function to decide which input circuit is better.


        Return:

            Returns with the chosen circuit and the corresponding parameter array


        """

        if not isinstance(circs, list):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance(parameter_arrs, list):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs):
            raise Exception("The first two arguments should be of the same length")

        metrics = [metric(circ) for circ in circs]

        metrics = np.array(metrics)

        min_idx = np.argmin(metrics)

        return circs[min_idx], parameter_arrs[min_idx]

    @staticmethod
    def PartitionDecompositionProcess(
        subcircuit: Circuit,
        subcircuit_parameters: np.ndarray,
        config: dict,
        structure=None,
    ) -> Tuple[Circuit, np.ndarray]:
        """
        Worker-friendly entry: decompose a partition subcircuit (optionally nested for TreeGuided).

        Args:
            subcircuit: Subcircuit acting on a subset of the wide register.
            subcircuit_parameters: Flat parameter vector slice for ``subcircuit``.
            config: Same keys as wide optimization (``strategy``, ``topology``, etc.).
            structure: Optional fixed gate structure when ``strategy == "Custom"``.

        Returns:
            List of ``(Circuit, parameters)`` pairs (or empty list on failure), remapped to the original register.
        """

        qbit_num_orig_circuit = subcircuit.get_Qbit_Num()

        involved_qbits = subcircuit.get_Qbits()

        qbit_num = len(involved_qbits)

        # create qbit map:
        qbit_map = {}
        for idx in range(len(involved_qbits)):
            qbit_map[involved_qbits[idx]] = idx
        mini_topology = None
        if config["topology"] is not None:
            mini_topology = extract_subtopology(involved_qbits, qbit_map, config)
        # remap the subcircuit to a smaller qubit register
        remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, qbit_num)

        if not structure is None:
            structure = structure.Remap_Qbits(qbit_map, qbit_num)

        # get the unitary representing the circuit
        unitary = remapped_subcircuit.get_Matrix(subcircuit_parameters)

        # decompose a small unitary into a new circuit
        all_decomposed = qgd_Wide_Circuit_Optimization.DecomposePartition(
            unitary, config, mini_topology, structure=structure
        )
        # create inverse qbit map:
        inverse_qbit_map = {}
        for key, value in qbit_map.items():
            inverse_qbit_map[value] = key
        result = []
        for decomposed_circuit, decomposed_parameters in all_decomposed:

            # remap the decomposed circuit in order to insert it into a large circuit
            new_subcircuit = decomposed_circuit.Remap_Qbits(
                inverse_qbit_map, qbit_num_orig_circuit
            )

            if config["test_subcircuits"]:
                CompareCircuits(
                    subcircuit,
                    subcircuit_parameters,
                    new_subcircuit,
                    decomposed_parameters,
                    parallel=config["parallel"],
                )

            new_subcircuit = new_subcircuit.get_Flat_Circuit()
            result.append((new_subcircuit, decomposed_parameters))
        return tuple(result)

    @staticmethod
    def build_partition_topo_deps(allparts):
        """Topological sort of partition gate-sets; returns ordered partitions and reverse-dependency map."""
        gate_to_parts = {}
        for i, part in enumerate(allparts):
            for gate in part:
                gate_to_parts.setdefault(gate, set()).add(i)
        g = {i: set() for i in range(len(allparts))}
        rg = {i: set() for i in range(len(allparts))}
        for i, part in enumerate(allparts):
            for gate in part:
                for other_part in gate_to_parts[gate]:
                    if other_part != i and (
                        len(part & allparts[other_part]) > 0
                        and (len(part) < len(allparts[other_part]))
                        or part < allparts[other_part]
                    ):
                        g[i].add(other_part)
                        rg[other_part].add(i)
        rg_ret = {i: set(rg[i]) for i in range(len(allparts))}
        S = collections.deque(m for m in rg if len(rg[m]) == 0)
        L = []
        while S:
            n = S.popleft()
            L.append(n)
            for m in set(g[n]):
                g[n].remove(m)
                rg[m].remove(n)
                if len(rg[m]) == 0:
                    S.append(m)
        if len(L) != len(allparts):
            raise ValueError("Dependency graph is not a DAG")
        neworder = {old: new for new, old in enumerate(L)}
        rg_ret = {
            neworder[i]: set(neworder[j] for j in rg_ret[i])
            for i in range(len(allparts))
        }
        return [
            allparts[i] for i in L
        ], rg_ret  # return partitions in dependency order and dependencies

    @staticmethod
    def make_all_partition_circuit(circ, orig_parameters, max_partition_size):
        """ILP-based partitioning: flatten ``circ`` into a circuit of sub-circuits with concatenated parameters.

        Returns:
            ``(partitioned_circuit, parameters, recombine_info, part_deps)`` for later fusion in
            ``recombine_all_partition_circuit``.
        """
        from squander.partitioning.ilp import get_all_partitions, _get_topo_order

        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = (
            get_all_partitions(circ, max_partition_size)
        )
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {
            x[0]: x
            for x in single_qubit_chains
            if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post
        }
        partitioned_circuit = Circuit(qbit_num_orig_circuit)
        params = []
        allparts, part_deps = qgd_Wide_Circuit_Optimization.build_partition_topo_deps(
            allparts
        )
        for part in allparts:
            surrounded_chains = {
                t
                for s in part
                for t in go[s]
                if t in single_qubit_chains_prepost
                and go[single_qubit_chains_prepost[t][-1]]
                and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part
            }
            gates = frozenset.union(
                part, *(single_qubit_chains_prepost[v] for v in surrounded_chains)
            )
            # topo sort part + surrounded chains
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in _get_topo_order(
                {x: go[x] & gates for x in gates},
                {x: rgo[x] & gates for x in gates},
                gate_to_qubit,
            ):
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(
                    orig_parameters[
                        start : start + gate_dict[gate_idx].get_Parameter_Num()
                    ]
                )
            partitioned_circuit.add_Circuit(c)
        for chain in single_qubit_chains:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in chain:
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(
                    orig_parameters[
                        start : start + gate_dict[gate_idx].get_Parameter_Num()
                    ]
                )
            partitioned_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)
        return (
            partitioned_circuit,
            parameters,
            (allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit),
            part_deps,
        )

    @staticmethod
    def strip_single_qubit_head_tails(circ, params):
        """Remove single-qubit gates that are purely at the head/tail of the dependency graph."""
        gate_dict, g, rg, gate_to_qubit, _ = build_dependency(circ)
        newcirc = Circuit(circ.get_Qbit_Num())
        new_params = []
        for i in gate_dict:
            gate = gate_dict[i]
            if len(gate_to_qubit[i]) == 1 and (len(g[i]) == 0 or len(rg[i]) == 0):
                continue
            newcirc.add_Gate(gate)
            start_idx = gate.get_Parameter_Start_Index()
            new_params.append(params[start_idx : start_idx + gate.get_Parameter_Num()])
        return newcirc, (
            np.empty((0,), dtype=np.float64)
            if len(new_params) == 0
            else np.concatenate(new_params, axis=0)
        )

    @staticmethod
    def get_fingerprint(circ, params):
        """Hashable signature of gate types, qubits, and parameters (for decomposition caching)."""
        return tuple(
            (gate.get_Name(), tuple(gate.get_Involved_Qbits()))
            for gate in circ.get_Gates()
        ) + tuple(params)

    @staticmethod
    def recombine_all_partition_circuit(
        circ, optimized_subcircuits, optimized_parameter_list, recombine_info
    ):
        """Reorder partition results to satisfy global dependencies.

        Uses ILP-based ordering and a final topological sort, then returns
        reordered subcircuits and parameter arrays aligned by structure index.
        """
        from squander.partitioning.ilp import (
            topo_sort_partitions,
            ilp_global_optimal,
            recombine_single_qubit_chains,
        )

        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = (
            recombine_info
        )
        max_gates = sum(
            sum(y for x, y in c.get_Gate_Nums().items() if x not in CNOT_COUNT_DICT)
            for c in optimized_subcircuits[: len(allparts)]
        )
        weights = [
            CNOTGateCount(circ, max_gates)
            for circ in optimized_subcircuits[: len(allparts)]
        ]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        struct_idxs = list(L)
        parts = recombine_single_qubit_chains(
            go,
            rgo,
            single_qubit_chains,
            gate_to_tqubit,
            [allparts[i] for i in L],
            fusion_info,
            surrounded_only=True,
        )
        single_qubit_chain_idx = {
            frozenset(chain): idx + len(allparts)
            for idx, chain in enumerate(single_qubit_chains)
        }
        for extrapart in parts[len(struct_idxs) :]:
            struct_idxs.append(single_qubit_chain_idx[frozenset(extrapart)])
        L = topo_sort_partitions(circ, parts)
        return [optimized_subcircuits[struct_idxs[i]] for i in L], [
            optimized_parameter_list[struct_idxs[i]] for i in L
        ]

    def OptimizeWideCircuit(
        self, circ: Circuit, parameters: np.ndarray
    ) -> Tuple[Circuit, np.ndarray]:
        """Top-level wide-circuit pass: optional routing, then Qiskit / BQSKit / Squander partition optimization.

        Sets ``self.config`` timing and intermediate circuit keys (e.g. ``routed_circuit``, ``optimization_time``).
        """
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(
            circ, self.config["topology"]
        ):

            if self.config["verbosity"] >= 1:
                print("fixing topology in the circuit")
            topo = self.config["topology"]
            self.config["topology"] = None
            strat = self.config["strategy"]
            self.config["strategy"] = self.config["pre-opt-strategy"]

            if self.config["verbosity"] >= 1:
                print("Optimizing circuit with all-to-all (a2a) connectivity")
            circ, parameters = self.OptimizeWideCircuit(circ, parameters)
            self.config["all_to_all_optimization_time"] = self.config[
                "optimization_time"
            ]
            self.config["all_to_all_circuit"] = circ
            self.config["all_to_all_parameters"] = parameters
            self.config["strategy"] = strat
            self.config["topology"] = topo
            start_time = time.time()

            if self.config["verbosity"] >= 1:
                print("Routing circuit to fix the topology")
            circ, parameters = self.route_circuit(circ, parameters)
            self.config["routing_time"] = time.time() - start_time
            self.config["routed_circuit"] = circ
            self.config["routed_parameters"] = parameters
        else:
            if self.config["verbosity"] >= 1:
                print("No additional routing is needed on the circuit")

        start_time = time.time()
        if self.config["strategy"] == "bqskit":
            if self.config["verbosity"] >= 1:
                print("Optimizing circuit with BQSkit")
            from squander import Qiskit_IO
            from bqskit import compile

            from bqskit.compiler.machine import MachineModel
            from bqskit.compiler import Compiler
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            from bqskit.passes import SetModelPass
            from bqskit.compiler.compile import (
                build_multi_qudit_retarget_workflow,
                build_resynthesis_optimization_workflow,
                build_single_qudit_retarget_workflow,
                build_gate_deletion_optimization_workflow,
                LogErrorPass,
            )

            # Build BQSKit machine model from your topology
            model = MachineModel(circ.get_Qbit_Num(), _topology_le_to_be(circ.get_Qbit_Num(), self.config["topology"]))

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, parameters)

            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))

            compilation_workflow = [
                SetModelPass(model),  # attach hardware model to circuit
                build_multi_qudit_retarget_workflow(
                    4, max_synthesis_size=self.max_partition_size
                ),
                build_resynthesis_optimization_workflow(
                    4, max_synthesis_size=self.max_partition_size, iterative=True
                ),
                build_single_qudit_retarget_workflow(
                    4, max_synthesis_size=self.max_partition_size
                ),
                build_gate_deletion_optimization_workflow(
                    4, max_synthesis_size=self.max_partition_size, iterative=True
                ),
                LogErrorPass(),
            ]

            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(
                    bqskit_circ, compilation_workflow, True
                )

                default = list(range(bqskit_circ.num_qudits))
                initial_map = pass_data.get("initial_mapping", default)
                final_map = pass_data.get("final_mapping", default)

            # Convert back: BQSKit → Qiskit → Squander
            circuit_qiskit = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(
                circuit_qiskit
            )

            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            if self.config["verbosity"] >= 2:
                print("OptimizeWideCircuit::check_compare_circuits")
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters

        elif self.config["strategy"] == "seqpam_PartAM":
            if self.config["verbosity"] >= 1:
                print("Optimizing circuit with BQSKit SeqPAM + Squander (PartAM ILP weights)")
            from squander import Qiskit_IO
            from bqskit.compiler import Compiler
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from bqskit.passes import SetModelPass
            from qiskit import qasm2, QuantumCircuit

            strategy_map = {"TreeSearch": "Tree_search", "TabuSearch": "Tabu_search"}
            squander_config = {
                "strategy": strategy_map.get(self.config.get("strategy", "TreeSearch"), "Tree_search"),
                "optimization_tolerance": self.config.get("tolerance", 1e-8),
                "verbosity": self.config.get("verbosity", 0),
                "optimizer_engine": self.config.get("optimizer_engine", "BFGS"),
                "Cost_Function_Variant": self.config.get("Cost_Function_Variant", 3),
                "size_density_weight": True,
                "sparse_penalty": self.config.get("sparse_penalty", 3.0),
                "max_partition_size": self.max_partition_size,
            }
            block_size = self.max_partition_size

            model = MachineModel(circ.get_Qbit_Num(), _topology_le_to_be(circ.get_Qbit_Num(), self.config["topology"]))
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, parameters)
            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))

            workflow = generate_squander_seqpam(squander_config, block_size)

            with Compiler() as compiler:
                routed_bqskit_circ = compiler.compile(
                    bqskit_circ, [SetModelPass(model), workflow]
                )

            circuit_qiskit = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit)

            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            if self.config["verbosity"] >= 2:
                print("OptimizeWideCircuit::check_compare_circuits")
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters

        elif self.config["strategy"] == "qiskit":
            if self.config["verbosity"] >= 1:
                print("Optimizing circuit with Qiskit")
            from squander import Qiskit_IO
            from qiskit import transpile
            from qiskit.transpiler import CouplingMap
            from squander.gates import gates_Wrapper as gate

            SUPPORTED_GATES_NAMES = {
                n.lower().replace("cnot", "cx")
                for n in dir(gate)
                if not n.startswith("_")
                and issubclass(getattr(gate, n), gate.Gate)
                and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")
            }
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, parameters)
            coupling_map = (
                None
                if self.config["topology"] is None
                else CouplingMap([[i, j] for i, j in self.config["topology"]])
            )
            circuit_qiskit = transpile(
                circo,
                basis_gates=SUPPORTED_GATES_NAMES,
                coupling_map=coupling_map,
                optimization_level=3,
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(
                circuit_qiskit
            )
            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            if self.config["verbosity"] >= 2:
                print("OptimizeWideCircuit::check_compare_circuits")
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters
        else:

            if self.config["verbosity"] >= 1:
                print("Optimizing circuit with Squander")
            part_size_start = self.max_partition_size
            part_size_end = self.config.get("part_size_end",self.max_partition_size)
            count = CNOTGateCount(circ, 0)
            fingerprint_dict = {}
            for max_part_size in range(part_size_start, part_size_end + 1):
                # instantiate the object for optimizing wide circuits
                wide_circuit_optimizer = qgd_Wide_Circuit_Optimization(
                    {**self.config, "max_partition_size": max_part_size}
                )
                while True:
                    # run circuit optimization
                    circ_flat, parameters = (
                        wide_circuit_optimizer.InnerOptimizeWideCircuit(
                            circ, parameters, fingerprint_dict=fingerprint_dict
                        )
                    )
                    circ = circ_flat.get_Flat_Circuit()
                    newcount = CNOTGateCount(circ, 0)
                    no_improve = newcount >= count
                    count = newcount
                    if no_improve:
                        break
        self.config["optimization_time"] = time.time() - start_time
        return circ, parameters

    def InnerOptimizeWideCircuit(
        self, circ: Circuit, orig_parameters: np.ndarray, fingerprint_dict=None
    ) -> Tuple[Circuit, np.ndarray]:
        """Optimize one pass of wide-circuit partition decomposition.

        The circuit is converted to a CNOT basis, partitioned, each partition is
        optimized (possibly in parallel), and then reconstructed into one circuit.

        Args:
            circ: Input circuit to optimize.
            orig_parameters: Parameter array associated with ``circ``.
            fingerprint_dict: Optional decomposition cache shared across passes.

        Returns:
            Tuple of ``(optimized_circuit, optimized_parameters)``.
        """
        from squander.utils import circuit_to_CNOT_basis

        circ, orig_parameters = circuit_to_CNOT_basis(circ, orig_parameters)
        max_gates = sum(
            y for x, y in circ.get_Gate_Nums().items() if x not in CNOT_COUNT_DICT
        )

        global_min = self.config.get("global_min", True)
        if global_min:
            partitioned_circuit, parameters, recombine_info, part_deps = (
                qgd_Wide_Circuit_Optimization.make_all_partition_circuit(
                    circ, orig_parameters, self.max_partition_size
                )
            )

        else:
            partitioned_circuit, parameters, _ = PartitionCircuit(
                circ,
                orig_parameters,
                self.max_partition_size,
                strategy=self.config["partition_strategy"],
            )
            part_deps = None

        subcircuits = partitioned_circuit.get_Gates()

        # subcircuits = subcircuits[9:10]

        in_parent = parent_process() is not None

        if not in_parent and self.config["verbosity"] >= 1:
            print(len(subcircuits), "partitions found to optimize")

        # the list of optimized subcircuits
        optimized_subcircuits: List[Optional[Circuit]] = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list: List[Optional[List[np.ndarray]]] = [None] * len(
            subcircuits
        )

        # list of AsyncResult objects
        async_results = [None] * len(subcircuits)

        total_opt = [0]

        def process_result(partition_idx):
            """Finalize async decomposition for partition ``partition_idx`` and update caches / lists."""
            if optimized_subcircuits[partition_idx] is not None:
                return
            subcircuit = subcircuits[partition_idx]
            # callback function done on the master process to compare the new decomposed and the original suncircuit
            start_idx = subcircuit.get_Parameter_Start_Index()
            subcircuit_parameters = parameters[
                start_idx : start_idx + subcircuit.get_Parameter_Num()
            ]
            fingerprint = (
                None
                if fingerprint_dict is None
                else qgd_Wide_Circuit_Optimization.get_fingerprint(
                    subcircuit, subcircuit_parameters
                )
            )
            callback_fnc = lambda x: self.CompareAndPickCircuits(
                [subcircuit, *(z[0] for z in x)],
                [subcircuit_parameters, *(z[1] for z in x)],
                lambda c: CNOTGateCount(c, max_gates),
            )
            if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                new_subcircuit, new_parameters = fingerprint_dict[fingerprint]
            else:
                new_subcircuit, new_parameters = callback_fnc(
                    async_results[partition_idx][0](*async_results[partition_idx][1])
                    if in_parent
                    else async_results[partition_idx].get(timeout=None)
                )

                if subcircuit != new_subcircuit and self.config["verbosity"] >= 2:
                    print(
                        "original subcircuit:    ",
                        subcircuit.get_Gate_Nums(),
                        partition_idx,
                    )
                    print("reoptimized subcircuit: ", new_subcircuit.get_Gate_Nums())
                if fingerprint_dict is not None:
                    fingerprint_dict[fingerprint] = (new_subcircuit, new_parameters)
                    fingerprint_dict[
                        qgd_Wide_Circuit_Optimization.get_fingerprint(
                            new_subcircuit, new_parameters
                        )
                    ] = (new_subcircuit, new_parameters)
                    trim_subcirc, trim_parameters = (
                        qgd_Wide_Circuit_Optimization.strip_single_qubit_head_tails(
                            new_subcircuit, new_parameters
                        )
                    )
                    fingerprint_dict[
                        qgd_Wide_Circuit_Optimization.get_fingerprint(
                            trim_subcirc, trim_parameters
                        )
                    ] = (trim_subcirc, trim_parameters)
            if total_opt[0] % 100 == 99 and self.config["verbosity"] >= 1:
                print(total_opt[0] + 1, "partitions optimized")
            total_opt[0] += 1
            optimized_subcircuits[partition_idx] = new_subcircuit
            optimized_parameter_list[partition_idx] = new_parameters

        with (
            contextlib.nullcontext()
            if in_parent
            else Pool(processes=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else mp.cpu_count())
        ) as pool:
            remaining = list(range(len(subcircuits)))
            while remaining:
                still_remaining = []
                #  code for iterate over partitions and optimize them
                for partition_idx in remaining:
                    subcircuit = subcircuits[partition_idx]

                    # isolate the parameters corresponding to the given sub-circuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx = start_idx + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[start_idx:end_idx]

                    fingerprint = (
                        None
                        if fingerprint_dict is None
                        else qgd_Wide_Circuit_Optimization.get_fingerprint(
                            subcircuit, subcircuit_parameters
                        )
                    )
                    if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                        (
                            optimized_subcircuits[partition_idx],
                            optimized_parameter_list[partition_idx],
                        ) = fingerprint_dict[fingerprint]
                        continue
                    if part_deps is not None and partition_idx in part_deps:
                        any_optimized, any_remaining = False, False
                        for dep_idx in part_deps[partition_idx]:
                            if optimized_subcircuits[dep_idx] is None and (
                                async_results[dep_idx] is None
                                or not isinstance(async_results[dep_idx], tuple)
                                and not async_results[dep_idx].ready()
                            ):
                                any_remaining = True
                                continue
                            elif optimized_subcircuits[dep_idx] is None:
                                process_result(dep_idx)

                            optimized_subcircuits_loc = optimized_subcircuits[dep_idx]
                            assert isinstance(optimized_subcircuits_loc, Circuit)
                            assert optimized_subcircuits_loc is not None

                            if CNOTGateCount(optimized_subcircuits_loc) < CNOTGateCount(
                                subcircuits[dep_idx]
                            ):  # if the dependency partition was optimized, skip
                                any_optimized = True
                                break
                        if any_optimized:
                            optimized_subcircuits[partition_idx] = subcircuit
                            optimized_parameter_list[partition_idx] = (
                                subcircuit_parameters
                            )
                            continue
                        if any_remaining:
                            still_remaining.append(partition_idx)
                            continue
                    # call a process to decompose a subcircuit
                    config = {
                        **self.config,
                        "tree_level_max": max(0, CNOTGateCount(subcircuit, 0) - 1),
                    }
                    fargs = (
                        self.PartitionDecompositionProcess,
                        (subcircuit, subcircuit_parameters, config, None),
                    )
                    # print("Dispatching", subcircuit.get_Involved_Qubits(), "qubits with", CNOGateCount(subcircuit, 0), "CNOT gates, partition ", partition_idx)
                    async_results[partition_idx] = (
                        fargs
                        if in_parent
                        else pool.apply_async(*fargs)
                    )
                if len(remaining) == len(still_remaining):
                    time.sleep(0.1)
                remaining = still_remaining
            #  code for iterate over async results and retrieve the new subcircuits
            for partition_idx in range(len(subcircuits)):
                process_result(partition_idx)

        # construct the wide circuit from the optimized suncircuits
        if global_min:
            optimized_subcircuits, optimized_parameter_list = (
                qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(
                    circ,
                    optimized_subcircuits,
                    optimized_parameter_list,
                    recombine_info,
                )
            )

        if any(c is None for c in optimized_subcircuits) or any(
            p is None for p in optimized_parameter_list
        ):
            raise RuntimeError(
                "Internal error: some partitions were not optimized before reconstruction."
            )
        wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions(
            cast(List[Circuit], optimized_subcircuits),
            cast(List[List[np.ndarray]], optimized_parameter_list),
        )

        if not in_parent and self.config["verbosity"] >= 1:
            print("original circuit:    ", circ.get_Gate_Nums())
            print("reoptimized circuit: ", wide_circuit.get_Gate_Nums())

        qgd_Wide_Circuit_Optimization.check_valid_routing(
            wide_circuit, self.config["topology"]
        )
        if self.config["verbosity"] >= 2:
            print("InnerOptimizeWideCircuit: check_compare_circuits")
        self.check_compare_circuits(
            circ, orig_parameters, wide_circuit, wide_parameters
        )

        return wide_circuit, wide_parameters

    @staticmethod
    def all_to_all_topology(num_qubits):
        """Undirected all-to-all coupler list for ``num_qubits`` qubits."""
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

    @staticmethod
    def linear_topology(num_qubits):
        """Path graph couplers ``(i, i+1)``."""
        return [(i, i + 1) for i in range(num_qubits - 1)]

    @staticmethod
    def star_topology(num_qubits):
        """Star graph: hub qubit ``0`` connected to all others."""
        return [(0, i) for i in range(1, num_qubits)]

    @staticmethod
    def ring_topology(num_qubits):
        """Ring couplers including wrap-around ``(n-1, 0)``."""
        return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]

    @staticmethod
    def lattice_topology(x_qbits, y_qbits):
        """2D grid of size ``x_qbits`` by ``y_qbits`` with nearest-neighbor horizontal and vertical edges."""
        return [
            (i * x_qbits + j, i * x_qbits + (j + 1))
            for i in range(y_qbits)
            for j in range(x_qbits - 1)
        ] + [
            (i * x_qbits + j, (i + 1) * x_qbits + j)
            for i in range(y_qbits - 1)
            for j in range(x_qbits)
        ]

    @staticmethod
    def heavy_hexagonal_topology(rows, cols):
        """
        Finite heavy-hex patch.

        rows, cols describe the underlying honeycomb 'brick-wall' patch.
        The first rows*cols qubits are the original honeycomb vertices.
        Every original edge gets one inserted degree-2 qubit.

        Returns:
            list[(u, v)]  undirected couplers
        """

        def vid(r, c):
            """Linear index for honeycomb vertex at row ``r``, column ``c``."""
            return r * cols + c

        # Underlying honeycomb / brick-wall edges
        base_edges = []

        for r in range(rows):
            for c in range(cols):
                # Vertical brick-wall edges
                if r + 1 < rows:
                    base_edges.append((vid(r, c), vid(r + 1, c)))

                # Alternating horizontal edges
                if c + 1 < cols and ((r + c) % 2 == 0):
                    base_edges.append((vid(r, c), vid(r, c + 1)))

        # Subdivide every honeycomb edge by inserting a qubit
        next_id = rows * cols
        heavy_edges = []

        for u, v in base_edges:
            w = next_id
            next_id += 1
            heavy_edges.append((u, w))
            heavy_edges.append((w, v))

        return heavy_edges

    @staticmethod
    def sycamore_topology():
        """Approximate Sycamore-like 6x9 grid topology (simplified; ignores known dead qubits)."""
        return qgd_Wide_Circuit_Optimization.lattice_topology(
            6, 9
        )  # there is a defective qubit at (0, 3) in the sycamore chip, but we ignore it here for simplicity

    @staticmethod
    def is_valid_routing(wide_circuit, topo):
        """True if every multi-qubit gate's qubits lie in a connected subgraph of undirected ``topo``."""
        if topo is None:
            return True

        import itertools

        topo_set = {frozenset(edge) for edge in topo}

        def qubits_connected(qubits):
            """Whether pairwise couplers in ``topo_set`` connect all qubits in ``qubits``."""
            if len(qubits) <= 1:
                return True
            edges = {
                frozenset((q1, q2))
                for q1, q2 in itertools.combinations(qubits, 2)
                if frozenset((q1, q2)) in topo_set
            }
            if len(edges) == 0:
                return False
            cur_set = set(edges.pop())
            while edges:
                next_edge = next((e for e in edges if len(e & cur_set) > 0), None)
                if next_edge is None:
                    return False
                cur_set |= next_edge
                edges.remove(next_edge)
            return set(qubits) <= cur_set

        return all(
            qubits_connected(gate.get_Involved_Qbits())
            for gate in wide_circuit.get_Flat_Circuit().get_Gates()
            if len(gate.get_Involved_Qbits()) > 1
        )

    @staticmethod
    def check_valid_routing(wide_circuit, topo):
        """Assert ``is_valid_routing``; raises if any gate violates ``topo``."""
        assert qgd_Wide_Circuit_Optimization.is_valid_routing(
            wide_circuit, topo
        ), "Final circuit contains gates that do not respect the routing constraints."

    def check_compare_circuits(
        self, circ, orig_parameters, wide_circuit, wide_parameters, routing=False, forced_test=False,
    ):
        """If ``test_final_circuit``, numerically compare unitaries (optional initial/final layout for routing)."""
        if self.config["test_final_circuit"] or forced_test:
            if (
                routing
                and self.config.get("initial_mapping", None) is not None
                and self.config.get("final_mapping", None) is not None
            ):
                CompareCircuits(
                    circ,
                    orig_parameters,
                    wide_circuit,
                    wide_parameters,
                    initial_mapping=self.config["initial_mapping"],
                    final_mapping=self.config["final_mapping"],
                    parallel=0,
                )
            else:
                CompareCircuits(circ, orig_parameters, wide_circuit, wide_parameters)

    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):
        """Map ``circ`` onto ``self.config['topology']`` using BQSKit SeQPAM, Qiskit SABRE, or Squander SABRE."""
        strategy = self.config.get("routing-strategy", "seqpam-ilp")

        if strategy == "seqpam-ilp":
            from squander import Qiskit_IO
            from squander.decomposition.qgd_Wide_Circuit_Optimization import generate_squander_seqpam
            from bqskit.compiler import Compiler
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from bqskit.passes import SetModelPass
            from qiskit import qasm2, QuantumCircuit

            model = MachineModel(circ.get_Qbit_Num(), _topology_le_to_be(circ.get_Qbit_Num(), self.config["topology"]))
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, orig_parameters)
            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))

            strategy_map = {"TreeSearch": "Tree_search", "TabuSearch": "Tabu_search"}
            squander_config = {
                "strategy": strategy_map.get(self.config.get("strategy", "TreeSearch"), "Tree_search"),
                "optimization_tolerance": self.config.get("tolerance", 1e-8),
                "verbosity": self.config.get("verbosity", 0),
                "optimizer_engine": self.config.get("optimizer_engine", "BFGS"),
                "Cost_Function_Variant": self.config.get("Cost_Function_Variant", 3),
                "size_density_weight": True,
                "sparse_penalty": self.config.get("sparse_penalty", 3.0),
                "max_partition_size": self.max_partition_size,
            }
            block_size = self.max_partition_size

            workflow = generate_squander_seqpam(squander_config, block_size)

            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(
                    bqskit_circ, [SetModelPass(model), workflow], True
                )

            circuit_qiskit_routed = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_routed)
            )
            Squander_remapped_circuit = Squander_remapped_circuit.Remap_Qbits(
                {i: j for i, j in enumerate(pass_data.placement)}
            )
            self.config["initial_mapping"] = list(
                pass_data.placement[x] for x in pass_data.initial_mapping
            )
            self.config["final_mapping"] = list(
                pass_data.placement[x] for x in pass_data.final_mapping
            )

        elif strategy in ("seqpam-quick", "bqskit-sabre"):
            from squander import Qiskit_IO
            from bqskit import Circuit as BQSKitCircuit, compile
            from bqskit.compiler import Compiler
            from bqskit.compiler.compile import (
                build_seqpam_mapping_optimization_workflow,
            )
            from bqskit.compiler.basepass import BasePass

            class SquanderPartitioner(BasePass):
                """BQSKit pass: replace circuit body with Squander ILP partition blocks (QASM round-trip)."""

                def __init__(self, max_partition_size):
                    super().__init__()
                    self.max_partition_size = max_partition_size

                async def run(self, circuit: BQSKitCircuit, data=None):
                    from squander import Qiskit_IO
                    from squander.partitioning.partition import PartitionCircuit

                    circ_qiskit = QuantumCircuit.from_qasm_str(
                        OPENQASM2Language().encode(circuit)
                    )
                    circ, orig_parameters = Qiskit_IO.convert_Qiskit_to_Squander(
                        circ_qiskit
                    )
                    partitioned_circuit, parameters, _ = PartitionCircuit(
                        circ, orig_parameters, self.max_partition_size, strategy="ilp"
                    )
                    partitioned_circuit_qiskit = Qiskit_IO.get_Qiskit_Circuit(
                        partitioned_circuit, parameters
                    )
                    partitioned_circuit_bqskit = OPENQASM2Language().decode(
                        qasm2.dumps(partitioned_circuit_qiskit)
                    )
                    circuit.become(partitioned_circuit_bqskit, False)

            from bqskit.passes import (
                GeneralizedSabreLayoutPass,
                GeneralizedSabreRoutingPass,
                SetModelPass,
                IfThenElsePass,
                QuickPartitioner,
            )
            from bqskit.ir.gates import CNOTGate  # example; extend as needed
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            # Build BQSKit machine model from your topology
            model = MachineModel(circ.get_Qbit_Num(), _topology_le_to_be(circ.get_Qbit_Num(), self.config["topology"]))

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, orig_parameters)

            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))
            # Customizable knobs

            # Routing-only pass pipeline — NO optimization passes
            mainflow = build_seqpam_mapping_optimization_workflow(
                block_size=self.config["max_partition_size"]
            )

            routing_workflow = [
                SetModelPass(model),  # attach hardware model to circuit
                *(
                    (build_seqpam_mapping_optimization_workflow(),)
                    if strategy != "bqskit-sabre"
                    else (
                        GeneralizedSabreLayoutPass(),  # SABRE-style layout
                        GeneralizedSabreRoutingPass(),
                    )
                ),  # SABRE-style routing
            ]

            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(
                    bqskit_circ, routing_workflow, True
                )

            # Convert back: BQSKit → Qiskit → Squander
            circuit_qiskit_routed = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_routed)
            )
            Squander_remapped_circuit = Squander_remapped_circuit.Remap_Qbits(
                {i: j for i, j in enumerate(pass_data.placement)}
            )
            self.config["initial_mapping"] = list(
                pass_data.placement[x] for x in pass_data.initial_mapping
            )
            self.config["final_mapping"] = list(
                pass_data.placement[x] for x in pass_data.final_mapping
            )

        elif strategy == "seqpam_partam":
            from squander import Qiskit_IO
            from squander.decomposition.qgd_Wide_Circuit_Optimization import generate_squander_seqpam
            from bqskit.compiler import Compiler
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from bqskit.passes import SetModelPass
            from qiskit import qasm2, QuantumCircuit

            model = MachineModel(circ.get_Qbit_Num(), _topology_le_to_be(circ.get_Qbit_Num(), self.config["topology"]))
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, orig_parameters)
            bqskit_circ = OPENQASM2Language().decode(qasm2.dumps(circo))

            squander_config = {
                'strategy': 'Tree_search',
                'optimization_tolerance': self.config.get('tolerance', 1e-8),
                'verbosity': self.config.get('verbosity', 0),
                'optimizer_engine': self.config.get('optimizer_engine', 'BFGS'),
                'size_density_weight': True,
                'sparse_penalty': self.config.get('sparse_penalty', 3.0),
                'max_partition_size': self.max_partition_size,
                'use_osr':0,
                'use_graph_search':0,
            }
            workflow = generate_squander_seqpam(squander_config, self.max_partition_size)

            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(
                    bqskit_circ, [SetModelPass(model), workflow], True
                )

            circuit_qiskit_routed = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(routed_bqskit_circ)
            )
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_routed)
            )
            Squander_remapped_circuit = Squander_remapped_circuit.Remap_Qbits(
                {i: j for i, j in enumerate(pass_data.placement)}
            )
            self.config["initial_mapping"] = list(
                pass_data.placement[x] for x in pass_data.initial_mapping
            )
            self.config["final_mapping"] = list(
                pass_data.placement[x] for x in pass_data.final_mapping
            )

        elif strategy == "light-sabre":
            from squander import Qiskit_IO
            from qiskit import transpile
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            from qiskit.transpiler.passes import SabreLayout, SabreSwap
            from qiskit.transpiler import PassManager, CouplingMap
            from squander.gates import gates_Wrapper as gate

            # SUPPORTED_GATES_NAMES = {n.lower().replace("cnot", "cx") for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")}
            circo = Qiskit_IO.get_Qiskit_Circuit(circ, orig_parameters)
            coupling_map = [[i, j] for i, j in self.config["topology"]]
            # circuit_qiskit_sabre = transpile(circo, basis_gates=SUPPORTED_GATES_NAMES, coupling_map=coupling_map, optimization_level=0)
            coupling_map = CouplingMap(coupling_map)
            # Customizable SABRE parameters
            sabre_seed = self.config.get("sabre_seed", 42)
            sabre_trials = self.config.get("sabre_trials", 5)  # layout trials
            swap_trials = self.config.get("sabre_swap_trials", sabre_trials)
            heuristic = self.config.get(
                "sabre_heuristic", "decay"
            )  # "basic" | "lookahead" | "decay"

            layout_pass = SabreLayout(
                coupling_map,
                seed=sabre_seed,
                max_iterations=sabre_trials,
                swap_trials=swap_trials,
            )
            swap_pass = SabreSwap(
                coupling_map,
                heuristic=heuristic,
                seed=sabre_seed,
                trials=swap_trials,
            )

            pm = PassManager(
                [
                    layout_pass,  # find initial qubit mapping via SABRE
                    swap_pass,  # insert SWAP gates for routing
                ]
            )
            circuit_qiskit_sabre = pm.run(circo)
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_sabre)
            )
            self.config["initial_mapping"] = (
                circuit_qiskit_sabre.layout.initial_index_layout()
            )
            self.config["final_mapping"] = (
                circuit_qiskit_sabre.layout.final_index_layout()
            )
        elif strategy == "sabre":
            sabre = SABRE(circ, self.config["topology"])
            (
                Squander_remapped_circuit,
                parameters_remapped_circuit,
                pi,
                final_pi,
                swap_count,
            ) = sabre.map_circuit(orig_parameters)
            self.config["initial_mapping"] = pi
            self.config["final_mapping"] = final_pi
        qgd_Wide_Circuit_Optimization.check_valid_routing(
            Squander_remapped_circuit, self.config["topology"]
        )

        if self.config["verbosity"] >= 2:
            print("cheking circuit after routing")
        self.check_compare_circuits(
            circ,
            orig_parameters,
            Squander_remapped_circuit,
            parameters_remapped_circuit,
            routing=True,
        )
        return Squander_remapped_circuit, parameters_remapped_circuit
