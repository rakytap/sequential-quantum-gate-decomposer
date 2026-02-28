"""
This is an implementation of Partition Aware Mapping.
"""
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from squander.partitioning.ilp import (
    get_all_partitions,
    _get_topo_order,
    topo_sort_partitions,
    ilp_global_optimal,
    recombine_single_qubit_chains,
)

import numpy as np

from typing import Callable, Dict, List, Optional, Set, Tuple, FrozenSet
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import Process, Pool
import os
import logging
from tqdm import tqdm
from collections import deque, defaultdict

from squander.synthesis.PartAM_utils import (
    get_subtopologies_of_type,
    get_unique_subtopologies,
    get_canonical_form,
    get_node_mapping,
    SingleQubitPartitionResult,
    PartitionSynthesisResult,
    PartitionCandidate,
    check_circuit_compatibility,
    construct_swap_circuit,
    calculate_dist_small,
    group_into_two_qubit_blocks,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class PartitionScoreData:
    mini_topologies: Tuple[Tuple[Tuple[int, int], ...], ...]
    topology_candidates: Tuple[Tuple[Tuple[int, int], ...], ...]
    permutations_pairs: Tuple[
        Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...], ...
    ]
    circuit_structures: Tuple[Tuple[Tuple[int, ...], ...], ...]
    qubit_map: Dict[int, int]
    involved_qbits: Tuple[int, ...]


# ============================================================================
# Parallel Processing Setup
# ============================================================================

_WORKER_SCORING_PARTITIONS: Optional[List[Optional[PartitionScoreData]]] = None
_WORKER_DISTANCE_MATRIX: Optional[np.ndarray] = None
_WORKER_SWAP_CACHE: Optional[Dict] = None
_WORKER_VIRTUAL_E: Optional[List[Tuple[int, int]]] = None


def _init_scoring_worker(scoring_partitions, distance_matrix, virtual_E=None):
    """Initializer for process-based scoring workers."""
    global _WORKER_SCORING_PARTITIONS, _WORKER_DISTANCE_MATRIX, _WORKER_SWAP_CACHE, _WORKER_VIRTUAL_E
    _WORKER_SCORING_PARTITIONS = scoring_partitions
    _WORKER_DISTANCE_MATRIX = distance_matrix
    _WORKER_SWAP_CACHE = {}
    _WORKER_VIRTUAL_E = virtual_E


def _score_candidate_worker(payload):
    """
    Worker wrapper that reconstructs scoring inputs from a lightweight payload.
    Payload format: (PartitionCandidate, F_snapshot, pi_snapshot[, free_routing])
    """
    if (
        _WORKER_SCORING_PARTITIONS is None
        or _WORKER_DISTANCE_MATRIX is None
    ):
        raise RuntimeError("Scoring worker not initialized with shared data.")
    partition_candidate, F_snapshot, pi_snapshot = payload[:3]
    free_routing = payload[3] if len(payload) > 3 else False
    return qgd_Partition_Aware_Mapping.score_partition_candidate(
        partition_candidate,
        F_snapshot,
        pi_snapshot,
        _WORKER_SCORING_PARTITIONS,
        _WORKER_DISTANCE_MATRIX,
        _WORKER_SWAP_CACHE,
        _WORKER_VIRTUAL_E,
        free_routing=free_routing,
    )


# ============================================================================
# Main Class: qgd_Partition_Aware_Mapping
# ============================================================================

class qgd_Partition_Aware_Mapping:

    # ------------------------------------------------------------------------
    # Initialization & Configuration
    # ------------------------------------------------------------------------

    def __init__(self, config):
        self.topology = config['topology']
        self.config = config
        self.config.setdefault('strategy', 'TreeSearch')
        self.config.setdefault('parallel', 0 )
        self.config.setdefault('verbosity', 0 )
        self.config.setdefault('tolerance', 1e-8 )
        self.config.setdefault('test_subcircuits', False )
        self.config.setdefault('test_final_circuit', True )
        self.config.setdefault('max_partition_size', 3 )
        self.config.setdefault('topology', None)
        self.config.setdefault('routed', False)
        self.config.setdefault('partition_strategy','ilp')
        self.config.setdefault('optimizer', 'BFGS2')
        self.config.setdefault('use_basin_hopping', 1)
        self.config.setdefault('bh_T', 1.0)
        self.config.setdefault('bh_stepsize', 0.5)
        self.config.setdefault('bh_interval', 50)
        self.config.setdefault('bh_target_accept_rate', 0.5)
        self.config.setdefault('bh_stepwise_factor', 0.9)
        self.config.setdefault('hs_score_workers', os.cpu_count() or 1)
        self.config.setdefault('window_size', 0)  # 0 = full circuit (backward compat)
        self.config.setdefault('use_osr',0)
        strategy = self.config['strategy']
        allowed_strategies = ['TreeSearch', 'TabuSearch', 'Adaptive']
        if not strategy in allowed_strategies:
            raise Exception(f"The strategy should be either of {allowed_strategies}, got {strategy}.")
        
        # Initialize caches for performance optimization
        self._topology_cache = {}  # {frozenset(edges): [topology_candidates]}
        self._swap_cache = {}     # {(pi_tuple, qbit_map_frozen): (swaps, output_perm)}

    # ------------------------------------------------------------------------
    # Scoring Methods
    # ------------------------------------------------------------------------

    def compute_routing_aware_weight(self, result, pi_init, D, E, dag_start=0.0, dag_end=1.0):
        """
        Compute a routing-aware ILP weight for a partition synthesis result.

        For each (topology, P_I, P_O, node_mapping) combination:
        - Routing cost is computed using the specific P_I and node_mapping, so the
          cost reflects where each partition qubit actually needs to go.
        - E penalty is computed on the output layout after applying P_O, so it
          correctly penalises layouts that leave future gates far from each other.

        DAG-position-dependent weighting:
        - Early partitions (dag_start~0): routing cost weighted higher
        - Late partitions (dag_end~1): E penalty weighted higher

        Args:
            result: PartitionSynthesisResult or SingleQubitPartitionResult
            pi_init: Current qubit layout (logical -> physical mapping).
                     None when routing is free (first window).
            D: Distance matrix between physical qubits
            E: List of (q_a, q_b) tuples for virtual outgoing 2-qubit gates
            dag_start: Float in [0, 1] — earliest DAG level of the partition
            dag_end: Float in [0, 1] — latest DAG level of the partition

        Returns:
            float: Combined weight (lower is better)
        """
        if isinstance(result, SingleQubitPartitionResult):
            return 0

        routing_weight = 1.0 - dag_start
        e_weight = dag_end

        N = len(D)
        k = result.N
        qbit_map_inv = {v: q for q, v in result.qubit_map.items()}  # q* → circuit qubit q

        best_score = np.inf

        for tdx, mini_topology in enumerate(result.mini_topologies):
            topology_candidates = self._get_subtopologies_of_type_cached(mini_topology)
            if not topology_candidates:
                continue

            # Precompute node_mappings (Q* → Q) once per topology candidate — independent of pdx
            node_mappings = [get_node_mapping(mini_topology, tc) for tc in topology_candidates]
            node_mappings = [nm for nm in node_mappings if nm]
            if not node_mappings:
                continue

            for pdx, (P_i, P_o) in enumerate(result.permutations_pairs[tdx]):
                cnot_count = result.cnot_counts[tdx][pdx]
                P_i_list = list(P_i)
                P_o_list = list(P_o)
                P_i_inv = [P_i_list.index(i) for i in range(k)]

                for node_mapping in node_mappings:
                    # --- Routing cost: bring each partition qubit from pi_init to its
                    # target physical position determined by P_I and node_mapping ---
                    routing_cost = 0
                    if pi_init is not None:
                        for q_star, q in qbit_map_inv.items():
                            target_Q = node_mapping[P_i_inv[q_star]]
                            dist = D[int(pi_init[q])][target_Q]
                            if not np.isinf(dist):
                                routing_cost += max(0, dist - 1) * 3

                    # --- Output layout: start from pi_init then apply P_O ---
                    if pi_init is not None:
                        pi_out = [int(x) for x in pi_init]
                    else:
                        pi_out = list(range(N))
                    for q_star in range(len(P_o_list)):
                        if q_star in qbit_map_inv:
                            q = qbit_map_inv[q_star]
                            pi_out[q] = node_mapping[P_o_list[q_star]]

                    # --- E penalty computed on the output layout after P_O ---
                    e_penalty = 0.0
                    if E:
                        involved = set(result.involved_qbits)
                        for (q_a, q_b) in E:
                            if q_a in involved or q_b in involved:
                                dist = D[pi_out[q_a]][pi_out[q_b]]
                                if not np.isinf(dist):
                                    e_penalty += max(0, (dist - 1)) * 3
                                else:
                                    e_penalty += 3.0

                    score = cnot_count + routing_weight * routing_cost + e_weight * e_penalty
                    best_score = min(best_score, score)

        if np.isinf(best_score):
            best_score = 0

        return best_score

    def compute_transition_cost(self, result_pred, result_succ, pi_init, D):
        """
        Compute the minimum transition cost between two partitions over all
        (topology, P_o, node_mapping) configs of pred and (topology, P_i, node_mapping)
        configs of succ.

        The cost measures how far each qubit involved in succ needs to travel
        from its position after pred's output to where succ's input requires it.

        Args:
            result_pred: PartitionSynthesisResult for the predecessor partition
            result_succ: PartitionSynthesisResult for the successor partition
            pi_init: Current qubit layout (logical -> physical), or None
            D: Distance matrix between physical qubits

        Returns:
            float: Minimum transition cost (lower is better)
        """
        if isinstance(result_pred, SingleQubitPartitionResult) or isinstance(result_succ, SingleQubitPartitionResult):
            return 0

        N = len(D)
        involved_pred = set(result_pred.involved_qbits)
        involved_succ = set(result_succ.involved_qbits)
        qmap_pred_inv = {v: k for k, v in result_pred.qubit_map.items()}  # q* -> q
        qmap_succ_inv = {v: k for k, v in result_succ.qubit_map.items()}  # q* -> q
        k_succ = result_succ.N

        # Precompute all output positions for pred: list of dicts {q: physical_pos}
        pred_outputs = []
        for tdx, mini_topo in enumerate(result_pred.mini_topologies):
            topo_candidates = self._get_subtopologies_of_type_cached(mini_topo)
            if not topo_candidates:
                continue
            node_mappings = [get_node_mapping(mini_topo, tc) for tc in topo_candidates]
            node_mappings = [nm for nm in node_mappings if nm]
            if not node_mappings:
                continue
            for pdx, (_, P_o) in enumerate(result_pred.permutations_pairs[tdx]):
                P_o_list = list(P_o)
                for nm in node_mappings:
                    out_pos = {}
                    for q_star, q in qmap_pred_inv.items():
                        out_pos[q] = nm[P_o_list[q_star]]
                    pred_outputs.append(out_pos)

        # Precompute all input target positions for succ: list of dicts {q: physical_pos}
        succ_inputs = []
        for tdx, mini_topo in enumerate(result_succ.mini_topologies):
            topo_candidates = self._get_subtopologies_of_type_cached(mini_topo)
            if not topo_candidates:
                continue
            node_mappings = [get_node_mapping(mini_topo, tc) for tc in topo_candidates]
            node_mappings = [nm for nm in node_mappings if nm]
            if not node_mappings:
                continue
            for pdx, (P_i, _) in enumerate(result_succ.permutations_pairs[tdx]):
                P_i_list = list(P_i)
                P_i_inv = [P_i_list.index(i) for i in range(k_succ)]
                for nm in node_mappings:
                    in_pos = {}
                    for q_star, q in qmap_succ_inv.items():
                        in_pos[q] = nm[P_i_inv[q_star]]
                    succ_inputs.append(in_pos)

        if not pred_outputs or not succ_inputs:
            return 0

        # Find minimum transition cost over all (pred_output, succ_input) pairs
        best_cost = np.inf
        for out_pos in pred_outputs:
            for in_pos in succ_inputs:
                cost = 0
                for q, target in in_pos.items():
                    if q in out_pos:
                        current = out_pos[q]
                    elif pi_init is not None:
                        current = int(pi_init[q])
                    else:
                        current = q
                    dist = D[current][target]
                    if not np.isinf(dist):
                        cost += max(0, dist - 1) * 3
                    if cost >= best_cost:
                        break
                best_cost = min(best_cost, cost)

        return best_cost if not np.isinf(best_cost) else 0

    # ------------------------------------------------------------------------
    # Caching Methods
    # ------------------------------------------------------------------------

    def _get_subtopologies_of_type_cached(self, mini_topology):
        """
        Cached version of get_subtopologies_of_type.
        Uses canonical form of mini_topology as cache key.
        """
        
        # Create canonical form key
        target_qubits = set()
        for u, v in mini_topology:
            target_qubits.add(u)
            target_qubits.add(v)
        if not target_qubits:
            return []
        
        # Use canonical form as cache key
        canonical_key = get_canonical_form(target_qubits, mini_topology)
        
        if canonical_key not in self._topology_cache:
            self._topology_cache[canonical_key] = get_subtopologies_of_type(self.topology, mini_topology)
        
        return self._topology_cache[canonical_key]

    @staticmethod
    def _compute_ideal_pi_for_candidate(candidate, N):
        """
        Compute the ideal pi_initial such that the given candidate needs zero
        SWAPs for input routing, plus the resulting pi_output after the partition.

        Returns:
            pi_initial: np.ndarray — layout where partition qubits are already
                        at their required physical positions.
            pi_output:  np.ndarray — layout after the partition circuit (P_o applied).
        """
        P_i_inv = [candidate.P_i.index(i) for i in range(len(candidate.P_i))]

        # Required physical position for each partition qubit
        required = {}
        for k, v in candidate.qbit_map.items():
            required[k] = candidate.node_mapping[P_i_inv[v]]

        pi_initial = np.zeros(N, dtype=int)
        used_physical = set(required.values())

        for k, p in required.items():
            pi_initial[k] = p

        remaining_physical = sorted(p for p in range(N) if p not in used_physical)
        remaining_logical = sorted(q for q in range(N) if q not in required)
        for q, p in zip(remaining_logical, remaining_physical):
            pi_initial[q] = p

        # Apply P_o to get output permutation (mirrors transform_pi logic)
        pi_output = np.array(pi_initial, dtype=int)
        qbit_map_inverse = {v: k for k, v in candidate.qbit_map.items()}
        for q_star in range(len(candidate.P_o)):
            if q_star in qbit_map_inverse:
                k = qbit_map_inverse[q_star]
                pi_output[k] = candidate.node_mapping[candidate.P_o[q_star]]

        return pi_initial, pi_output

    def _build_scoring_partitions(self, optimized_partitions) -> List[Optional[PartitionScoreData]]:
        """
        Create lightweight, picklable views of partitions that contain only the
        data required during heuristic scoring.
        """
        scoring_partitions: List[Optional[PartitionScoreData]] = []
        for partition in optimized_partitions:
            if isinstance(partition, SingleQubitPartitionResult):
                scoring_partitions.append(None)
                continue

            mini_topologies = tuple(
                tuple(tuple(edge) for edge in mini_topology)
                for mini_topology in partition.mini_topologies
            )

            topology_candidates = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                if hasattr(partition, "get_topology_candidates"):
                    candidates = partition.get_topology_candidates(tdx)
                else:
                    candidates = self._get_subtopologies_of_type_cached(mini_topology)
                topology_candidates.append(
                    tuple(tuple(edge) for edge in candidates)
                )

            permutations_pairs = tuple(
                tuple((tuple(P_i), tuple(P_o)) for (P_i, P_o) in partition.permutations_pairs[tdx])
                for tdx in range(len(partition.mini_topologies))
            )

            circuit_structures = tuple(
                tuple(tuple(struct) for struct in partition.circuit_structures[tdx])
                for tdx in range(len(partition.mini_topologies))
            )

            scoring_partitions.append(
                PartitionScoreData(
                    mini_topologies=mini_topologies,
                    topology_candidates=tuple(topology_candidates),
                    permutations_pairs=permutations_pairs,
                    circuit_structures=circuit_structures,
                    qubit_map=dict(partition.qubit_map),
                    involved_qbits=tuple(partition.involved_qbits),
                )
            )
        return scoring_partitions

    @staticmethod
    def _group_circuit_for_levels(circuit):
        """
        Group a flat circuit into 2-qubit blocks for coarser DAG level generation.

        Returns:
            grouped_circ: Circuit with 2-qubit blocks as top-level elements
            block_gate_orders: List of lists mapping each block index to the
                               original flat gate indices it contains
        """
        gates = circuit.get_Gates()

        # Track gate indices following group_into_two_qubit_blocks logic
        pending = defaultdict(list)
        block_gate_orders = []
        last_block_for_qubit = {}

        for gate_idx, gate in enumerate(gates):
            qubits = gate.get_Involved_Qbits()
            if len(qubits) == 1:
                pending[qubits[0]].append(gate_idx)
            else:
                q0, q1 = qubits[0], qubits[1]
                block_order = list(pending[q0]) + list(pending[q1]) + [gate_idx]
                pending[q0].clear()
                pending[q1].clear()
                block_idx = len(block_gate_orders)
                block_gate_orders.append(block_order)
                last_block_for_qubit[q0] = block_idx
                last_block_for_qubit[q1] = block_idx

        # Trailing single-qubit gates
        for q, gate_indices in pending.items():
            if not gate_indices:
                continue
            if q in last_block_for_qubit:
                block_gate_orders[last_block_for_qubit[q]].extend(gate_indices)
            else:
                # Qubit only has single-qubit gates — standalone block
                block_gate_orders.append(list(gate_indices))

        grouped_circ = group_into_two_qubit_blocks(circuit)

        return grouped_circ, block_gate_orders

    # ------------------------------------------------------------------------
    # Partition Decomposition Methods
    # ------------------------------------------------------------------------

    @staticmethod
    def DecomposePartition_Sequential(Partition_circuit: Circuit, Partition_parameters: np.ndarray, config: dict, topologies, involved_qbits, qbit_map) -> PartitionSynthesisResult:
        """
        Call to decompose a partition sequentially
        """
        N = Partition_circuit.get_Qbit_Num()
        if N !=1:
            perumations_all = list(permutations(range(N)))
            result = PartitionSynthesisResult(N, topologies, involved_qbits, qbit_map, Partition_circuit)
            # Sequential permutation search
            for topology_idx in range(len(topologies)):
                mini_topology = topologies[topology_idx]
                P_o_initial = perumations_all[np.random.choice(range(len(perumations_all)))]
                for P_i in perumations_all:
                    Partition_circuit_tmp = Circuit(N)
                    Partition_circuit_tmp.add_Permutation(list(P_i))  # Must convert tuple to list
                    Partition_circuit_tmp.add_Circuit(Partition_circuit)
                    Partition_circuit_tmp.add_Permutation(list(P_o_initial))  # Must convert tuple to list
                    synthesised_circuit, synthesised_parameters = qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                    result.add_result((P_i, P_o_initial), synthesised_circuit, synthesised_parameters, topology_idx)

                P_i_best, _ = result.get_best_result(topology_idx)[0]
                for P_o in perumations_all:
                    Partition_circuit_tmp = Circuit(N)
                    Partition_circuit_tmp.add_Permutation(list(P_i_best))  # Must convert tuple to list
                    Partition_circuit_tmp.add_Circuit(Partition_circuit)
                    Partition_circuit_tmp.add_Permutation(list(P_o))  # Must convert tuple to list
                    synthesised_circuit, synthesised_parameters = qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                    result.add_result((P_i_best, P_o), synthesised_circuit, synthesised_parameters, topology_idx)
        else:
            result = SingleQubitPartitionResult(Partition_circuit,Partition_parameters)
        return result

    @staticmethod
    def DecomposePartition_Full(Partition_circuit: Circuit, Partition_parameters: np.ndarray, config: dict, topologies, involved_qbits, qbit_map) -> PartitionSynthesisResult:
        """
        Call to decompose a partition sequentially
        """
        N = Partition_circuit.get_Qbit_Num()
        if N !=1:
            permutations_all = list(permutations(range(N)))
            result = PartitionSynthesisResult(N, topologies, involved_qbits, qbit_map, Partition_circuit)
            # Sequential permutation search
            for topology_idx in range(len(topologies)):
                mini_topology = topologies[topology_idx]
                for P_i in permutations_all:
                    for P_o in permutations_all:
                        Partition_circuit_tmp = Circuit(N)
                        Partition_circuit_tmp.add_Permutation(list(P_i))  # Must convert tuple to list
                        Partition_circuit_tmp.add_Circuit(Partition_circuit)
                        Partition_circuit_tmp.add_Permutation(list(P_o))  # Must convert tuple to list
                        synthesised_circuit, synthesised_parameters = qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                        result.add_result((P_i, P_o), synthesised_circuit, synthesised_parameters, topology_idx)
        else:
            result = SingleQubitPartitionResult(Partition_circuit,Partition_parameters)
        return result

    @staticmethod
    def DecomposePartition_and_Perm(Umtx: np.ndarray, config: dict, mini_topology = None, max_retries: int = 5) -> Circuit:
        """
        Call to decompose a partition. Retries up to max_retries times if the
        decomposition error exceeds the configured tolerance.
        """
        tolerance = config["tolerance"]
        strategy = config["strategy"]

        for attempt in range(max_retries):
            if strategy == "TreeSearch":
                cDecompose = N_Qubit_Decomposition_Tree_Search( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
            elif strategy == "TabuSearch":
                cDecompose = N_Qubit_Decomposition_Tabu_Search( Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology )
            elif strategy == "Adaptive":
                cDecompose = N_Qubit_Decomposition_adaptive( Umtx.conj().T, level_limit_max=5, level_limit_min=1, topology=mini_topology )
            else:
                raise Exception(f"Unsupported decomposition type: {strategy}")
            cDecompose.set_Verbose( config["verbosity"] )
            cDecompose.set_Cost_Function_Variant( 3 )
            cDecompose.set_Optimization_Tolerance( tolerance )
            cDecompose.set_Optimizer( config["optimizer"] )
            cDecompose.Start_Decomposition()

            err = cDecompose.get_Decomposition_Error()
            if err <= tolerance:
                break

            if attempt >= max_retries - 1:
                break

        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
        return squander_circuit, parameters

    # ------------------------------------------------------------------------
    # Circuit Synthesis
    # ------------------------------------------------------------------------

    def SynthesizeWideCircuit(self, circ, orig_parameters, pi_init=None, E=None,
                              DAG_start=0, DAG_end=0, window_gate_indices=None):
        """
        Partition and synthesize a circuit, optionally restricted to a window.

        Args:
            circ: The full quantum circuit (must be flat — no subcircuit blocks)
            orig_parameters: Parameters for circ
            pi_init: Current qubit permutation (logical->physical). When provided,
                     enables routing-aware ILP scoring.
            E: Virtual outgoing gates as List[(q_a, q_b)]. Computed automatically
               when None and a window is active.
            DAG_start: First DAG level to process (inclusive)
            DAG_end: Last DAG level to process (exclusive). 0 means all levels.
            window_gate_indices: Optional list of gate indices (into circ) to
                process. When provided, overrides DAG_start/DAG_end.

        Returns:
            optimized_partitions: List of PartitionSynthesisResult / SingleQubitPartitionResult
        """
        # ---- Phase 0: Window extraction ----
        all_gates = circ.get_Gates()
        qbit_num = circ.get_Qbit_Num()

        if window_gate_indices is not None:
            # Window specified by explicit gate indices
            window_topo_order = list(window_gate_indices)
            window_gate_set = set(window_topo_order)
            full_circuit_mode = (len(window_gate_set) == len(all_gates))
            has_gates_beyond_window = len(window_gate_set) < len(all_gates)
        else:
            # Window specified by DAG level range
            levels = self.generate_DAG_levels(circ)
            total_levels = len(levels)

            if DAG_start == 0 and DAG_end == 0:
                effective_end = total_levels
            else:
                effective_end = min(DAG_end, total_levels)
            effective_start = DAG_start

            if effective_start >= total_levels or effective_start >= effective_end:
                self._last_synthesis_metadata = {
                    'E': E if E is not None else [],
                    'window_gates': 0,
                    'total_gates': len(all_gates),
                }
                return []

            window_topo_order = []
            for level_idx in range(effective_start, effective_end):
                window_topo_order.extend(levels[level_idx])
            window_gate_set = set(window_topo_order)
            full_circuit_mode = (len(window_gate_set) == len(all_gates))
            has_gates_beyond_window = effective_end < total_levels

        if full_circuit_mode:
            working_circ = circ
            working_parameters = orig_parameters
        else:
            # Build sub-circuit from window gates
            working_circ = Circuit(qbit_num)
            working_params_list = []
            for orig_idx in window_topo_order:
                gate = all_gates[orig_idx]
                working_circ.add_Gate(gate)
                start = gate.get_Parameter_Start_Index()
                working_params_list.append(
                    orig_parameters[start:start + gate.get_Parameter_Num()]
                )
            if working_params_list:
                working_parameters = np.concatenate(working_params_list, axis=0)
            else:
                working_parameters = np.array([])

        # ---- Phase 0b: Identify virtual outgoing gates (E) ----
        if E is None and not full_circuit_mode and has_gates_beyond_window:
            E = []
            for orig_idx in window_topo_order:
                gate = all_gates[orig_idx]
                children = circ.get_Children(gate)
                for child_idx in children:
                    if child_idx not in window_gate_set:
                        child_gate = all_gates[child_idx]
                        child_qubits = child_gate.get_Involved_Qbits()
                        if len(child_qubits) == 2:
                            E.append((child_qubits[0], child_qubits[1]))
            E = list(set(E))
        elif E is None:
            E = []

        # ---- Phase 0c: Compute distance matrix ----
        D = self.compute_distances_bfs(qbit_num)

        # ---- Phase 1: Partition enumeration ----
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(working_circ, self.config["max_partition_size"])
        qbit_num_orig_circuit = working_circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(working_circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}
        partitioned_circuit = Circuit( qbit_num_orig_circuit )
        params = []
        for part in allparts:
            surrounded_chains = {t for s in part for t in go[s] if t in single_qubit_chains_prepost and go[single_qubit_chains_prepost[t][-1]] and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
            gates = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded_chains))
            #topo sort part + surrounded chains
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in _get_topo_order({x: go[x] & gates for x in gates}, {x: rgo[x] & gates for x in gates}):
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)
        # Only add single-qubit chains as separate partitions if minimum_partition_size allows it
        for chain in single_qubit_chains:
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in chain:
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)

        # ---- Phase 2: Stage 1 synthesis (Sequential) ----
        subcircuits = partitioned_circuit.get_Gates()
        optimized_results = [None] * len(subcircuits)

        with Pool(processes=mp.cpu_count()) as pool:
            for partition_idx, subcircuit in enumerate( subcircuits ):

                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
                involved_qbits = subcircuit.get_Qbits()

                qbit_num_sub = len( involved_qbits )
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num_sub)
                qbit_map = {}
                for idx in range( len(involved_qbits) ):
                    qbit_map[ involved_qbits[idx] ] = idx
                remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num_sub )
                optimized_results[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="First Synthesis",disable=self.config.get('progressbar', 0) == False) ):
                optimized_results[partition_idx] = optimized_results[partition_idx].get()

        # ---- Phase 3: ILP partition selection with routing-aware weights ----
        gate_to_level = self.get_gate_DAG_level_map(working_circ)
        max_level = max(gate_to_level.values()) if gate_to_level else 0

        weights = []
        for idx, result in enumerate(optimized_results[:len(allparts)]):
            partition_gates = allparts[idx]
            part_start = min(gate_to_level.get(g, 0) for g in partition_gates)
            part_end = max(gate_to_level.get(g, 0) for g in partition_gates)
            dag_start = part_start / max_level if max_level > 0 else 0.0
            dag_end = part_end / max_level if max_level > 0 else 1.0
            weights.append(
                self.compute_routing_aware_weight(result, pi_init, D, E, dag_start, dag_end)
            )

        # ---- Phase 3b: Compute inter-partition transition costs ----
        transition_weight = self.config.setdefault('transition_weight', 1.0)
        transition_costs = {}
        if transition_weight > 0:
            gate_to_part = {}
            for idx, part in enumerate(allparts):
                for gate in part:
                    gate_to_part.setdefault(gate, []).append(idx)

            # Build directed DAG-neighbor partition pairs (pred -> succ)
            directed_neighbors = set()
            for gate_u, successors in g.items():
                for part_u in gate_to_part.get(gate_u, []):
                    for gate_v in successors:
                        for part_v in gate_to_part.get(gate_v, []):
                            if part_u != part_v:
                                directed_neighbors.add((part_u, part_v))

            # Compute transition cost for each directed pair, keyed by (min, max)
            seen_pairs = set()
            for pred_idx, succ_idx in directed_neighbors:
                pair_key = (min(pred_idx, succ_idx), max(pred_idx, succ_idx))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                if allparts[pred_idx] & allparts[succ_idx]:
                    continue
                result_pred = optimized_results[pred_idx]
                result_succ = optimized_results[succ_idx]
                # Compute both directions and take the minimum
                cost_fwd = self.compute_transition_cost(result_pred, result_succ, pi_init, D)
                cost_rev = self.compute_transition_cost(result_succ, result_pred, pi_init, D)
                cost = min(cost_fwd, cost_rev)
                if cost > 0:
                    transition_costs[pair_key] = cost * transition_weight

        L_parts, fusion_info = ilp_global_optimal(allparts, g, weights=weights, transition_costs=transition_costs)
        parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L_parts], fusion_info)
        L = topo_sort_partitions(working_circ, self.config["max_partition_size"], parts)
        from squander.partitioning.kahn import kahn_partition_preparts
        from squander.partitioning.tools import translate_param_order
        partitioned_circuit, param_order, _ = kahn_partition_preparts(working_circ, self.config["max_partition_size"], [parts[i] for i in L])
        parameters = translate_param_order(working_parameters, param_order)

        # ---- Phase 4: Stage 2 synthesis (Full) ----
        subcircuits = partitioned_circuit.get_Gates()
        optimized_partitions = [None] * len(subcircuits)

        with Pool(processes=mp.cpu_count()) as pool:
            for partition_idx, subcircuit in enumerate( subcircuits ):

                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
                involved_qbits = subcircuit.get_Qbits()

                qbit_num_sub = len( involved_qbits )
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num_sub)
                qbit_map = {}
                for idx in range( len(involved_qbits) ):
                    qbit_map[ involved_qbits[idx] ] = idx
                remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num_sub )
                optimized_partitions[partition_idx] = pool.apply_async( self.DecomposePartition_Full, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="Second Synthesis",disable=self.config.get('progressbar', 0) == False) ):
                optimized_partitions[partition_idx] = optimized_partitions[partition_idx].get()

        # ---- Phase 5: Store metadata and return ----
        self._last_synthesis_metadata = {
            'E': E,
            'window_gates': len(window_gate_set),
            'total_gates': len(all_gates),
        }

        return optimized_partitions

    # ------------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------------

    def Partition_Aware_Mapping(self, circ: Circuit, orig_parameters: np.ndarray):
        N = circ.get_Qbit_Num()

        # Pre-process: group circuit for coarser DAG levels (window boundaries)
        has_2q_gates = any(len(g.get_Involved_Qbits()) >= 2 for g in circ.get_Gates())
        if has_2q_gates:
            grouped_circ, block_gate_orders = self._group_circuit_for_levels(circ)
        else:
            grouped_circ = circ
            block_gate_orders = None

        window_size = self.config.get('window_size', 0)
        grouped_levels = self.generate_DAG_levels(grouped_circ)
        total_levels = len(grouped_levels)

        # ---- Full-circuit path (backward compat) ----
        if window_size <= 0 or window_size >= total_levels:
            # Pass the original flat circuit — no grouping needed for full circuit
            optimized_partitions = self.SynthesizeWideCircuit(circ, orig_parameters)

            for partition in optimized_partitions:
                if isinstance(partition, PartitionSynthesisResult):
                    partition._topology = self.topology
                    partition._topology_cache = self._topology_cache

            DAG, IDAG = self.construct_DAG_and_IDAG(optimized_partitions)

            D = self.compute_distances_bfs(N)
            pi = np.arange(N)  # Dummy — free_initial_routing will derive pi_initial

            F = self.get_initial_layer(IDAG, N, optimized_partitions)
            scoring_partitions = self._build_scoring_partitions(optimized_partitions)

            partition_order, pi, pi_initial = self.Heuristic_Search(F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, free_initial_routing=True)

            final_circuit, final_parameters = self.Construct_circuit_from_HS(partition_order, optimized_partitions, N)

            return final_circuit, final_parameters, pi_initial, pi

        # ---- Windowed mode ----
        D = self.compute_distances_bfs(N)
        pi = np.arange(N)  # Dummy for first window — free_initial_routing derives pi_initial
        pi_initial = None

        all_window_circuits = []
        all_window_params = []

        for window_start in range(0, total_levels, window_size):
            window_end = min(window_start + window_size, total_levels)
            is_first_window = (window_start == 0)

            # Expand grouped block indices to flat gate indices
            window_gate_indices = []
            for level_idx in range(window_start, window_end):
                for block_idx in grouped_levels[level_idx]:
                    if block_gate_orders is not None:
                        window_gate_indices.extend(block_gate_orders[block_idx])
                    else:
                        window_gate_indices.append(block_idx)

            # a. Synthesize this window (pass original flat circuit)
            window_partitions = self.SynthesizeWideCircuit(
                circ, orig_parameters,
                pi_init=pi if not is_first_window else None,
                window_gate_indices=window_gate_indices
            )

            # Skip empty windows
            if not window_partitions:
                continue

            # Retrieve virtual outgoing gates computed by SynthesizeWideCircuit
            virtual_E = self._last_synthesis_metadata.get('E', []) or []

            # b. Set topology info on partition results
            for partition in window_partitions:
                if isinstance(partition, PartitionSynthesisResult):
                    partition._topology = self.topology
                    partition._topology_cache = self._topology_cache

            # c. Build per-window structures
            DAG, IDAG = self.construct_DAG_and_IDAG(window_partitions)
            F = self.get_initial_layer(IDAG, N, window_partitions)
            scoring_partitions = self._build_scoring_partitions(window_partitions)

            # d. Heuristic search for this window (pi carries forward)
            partition_order, pi, window_pi_initial = self.Heuristic_Search(
                F, pi.copy(), DAG, IDAG,
                window_partitions, scoring_partitions, D,
                virtual_E=virtual_E if virtual_E else None,
                free_initial_routing=is_first_window,
            )

            if is_first_window:
                pi_initial = window_pi_initial

            # e. Construct window circuit
            window_circuit, window_params = self.Construct_circuit_from_HS(
                partition_order, window_partitions, N
            )

            # f. Append results
            all_window_circuits.append(window_circuit)
            all_window_params.append(window_params)

        # Concatenate all window circuits and parameters
        final_circuit = Circuit(N)
        for wc in all_window_circuits:
            final_circuit.add_Circuit(wc)

        if all_window_params:
            final_parameters = np.concatenate(all_window_params, axis=0)
        else:
            final_parameters = np.array([])

        if pi_initial is None:
            pi_initial = np.arange(N)

        return final_circuit, final_parameters, pi_initial, pi

    # ------------------------------------------------------------------------
    # Heuristic Search
    # ------------------------------------------------------------------------

    def Heuristic_Search(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, virtual_E=None, free_initial_routing=False):
        pi_initial = pi.copy()

        resolved_partitions = [False] * len(DAG)
        partition_order = []
        step = 0
        first_routing_done = not free_initial_routing
        buffered_single_qubit = []

        for partition_idx in list(F):
            if isinstance(optimized_partitions[partition_idx], SingleQubitPartitionResult):
                F.remove(partition_idx)
                single_qubit_part = optimized_partitions[partition_idx]

                if free_initial_routing and not first_routing_done:
                    # Buffer — will remap after pi_initial is determined
                    buffered_single_qubit.append(single_qubit_part)
                else:
                    qubit = single_qubit_part.circuit.get_Qbits()[0]
                    single_qubit_part.circuit.Remap_Qbits({int(qubit): int(pi[qubit])},max(D.shape))
                    partition_order.append(single_qubit_part)

                resolved_partitions[partition_idx] = True
                children = list(DAG[partition_idx])
                while len(children) !=0:
                    child = children.pop(0)
                    parents_resolved = True
                    for parent in IDAG[child]:
                        parents_resolved *= resolved_partitions[parent]
                    if parents_resolved:
                        F.append(child)


        # Initialize progress bar
        total_partitions = len(DAG)
        pbar = tqdm(total=total_partitions, desc="Heuristic Search",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} resolved',
                   disable=self.config.get('progressbar', 0) == False)

        while len(F) != 0:
                partition_candidates = self.obtain_partition_candidates(F,optimized_partitions)
                if len(partition_candidates) == 0:
                    break
                F_snapshot = tuple(F)
                use_free_routing = not first_routing_done
                scores = [
                        self.score_partition_candidate(
                            partition_candidate,
                            F_snapshot,
                            pi,
                            scoring_partitions,
                            D,
                            self._swap_cache,
                            virtual_E,
                            free_routing=use_free_routing,
                        )
                        for partition_candidate in partition_candidates
                    ]
                min_idx = np.argmin(scores)
                min_partition_candidate = partition_candidates[min_idx]

                F.remove(min_partition_candidate.partition_idx)
                resolved_partitions[min_partition_candidate.partition_idx] = True
                resolved_count = sum(resolved_partitions)
                pbar.n = resolved_count
                pbar.refresh()

                if not first_routing_done:
                    # Derive pi_initial from chosen candidate — no SWAPs needed
                    pi_initial, pi = self._compute_ideal_pi_for_candidate(
                        min_partition_candidate, len(pi)
                    )
                    first_routing_done = True

                    # Remap and insert buffered single-qubit partitions
                    for sq_part in buffered_single_qubit:
                        qubit = sq_part.circuit.get_Qbits()[0]
                        sq_part.circuit.Remap_Qbits({int(qubit): int(pi_initial[qubit])}, max(D.shape))
                        partition_order.append(sq_part)
                    buffered_single_qubit = []
                else:
                    swap_order, pi = min_partition_candidate.transform_pi(pi, D, self._swap_cache)
                    if len(swap_order)!=0:
                        partition_order.append(construct_swap_circuit(swap_order, len(pi)))

                partition_order.append(min_partition_candidate)
                children = list(DAG[min_partition_candidate.partition_idx])
                step += 1
                while len(children) != 0:
                    child = children.pop(0)
                    parents_resolved = True
                    for parent in IDAG[child]:
                        parents_resolved *= resolved_partitions[parent]
                    if (not resolved_partitions[child] and child not in F) and parents_resolved:
                        if isinstance(optimized_partitions[child], SingleQubitPartitionResult):
                            child_partition = optimized_partitions[child]
                            qubit = child_partition.circuit.get_Qbits()[0]
                            child_partition.circuit.Remap_Qbits({int(qubit): int(pi[qubit])},max(D.shape))
                            partition_order.append(child_partition)
                            resolved_partitions[child] = True
                            resolved_count = sum(resolved_partitions)
                            pbar.n = resolved_count
                            pbar.refresh()
                            children.extend(DAG[child])
                        else:
                            F.append(child)

        # If no multi-qubit partition was resolved, flush buffered single-qubit parts
        if buffered_single_qubit:
            for sq_part in buffered_single_qubit:
                qubit = sq_part.circuit.get_Qbits()[0]
                sq_part.circuit.Remap_Qbits({int(qubit): int(pi[qubit])}, max(D.shape))
                partition_order.append(sq_part)

        pbar.close()
        return partition_order, pi, pi_initial

    # ------------------------------------------------------------------------
    # Circuit Construction
    # ------------------------------------------------------------------------

    def Construct_circuit_from_HS(self, partition_order, optimized_partitions,N):
        final_circuit = Circuit(N)
        final_parameters = []
        perm_count = 0
        partition_count = 0
        
        for part in partition_order:
            if isinstance(part, Circuit):
                final_circuit.add_Circuit(part)
                perm_count += 1
            elif isinstance(part, SingleQubitPartitionResult):
                final_circuit.add_Circuit(part.circuit)
                final_parameters.append(part.parameters)
                partition_count += 1
            else:
                part_circ, part_parameters = part.get_final_circuit(optimized_partitions,N)
                final_circuit.add_Circuit(part_circ)
                final_parameters.append(part_parameters)
                partition_count += 1
        
        if final_parameters:
            final_parameters = np.concatenate([np.atleast_1d(p).ravel() for p in final_parameters], axis=0)
        else:
            final_parameters = np.array([])
        if not check_circuit_compatibility(final_circuit,self.topology):
            print("ERROR: Final circuit is not compatible with device topology!")
        return final_circuit, final_parameters
    
    # ------------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------------

    @staticmethod
    def score_partition_candidate(partition_candidate, F, pi, scoring_partitions, D, swap_cache, virtual_E=None, free_routing=False):
        score_F = 0
        swap_weight = 1
        swaps, output_perm = partition_candidate.transform_pi(pi, D, swap_cache, free_routing=free_routing)
        if not free_routing:
            score_F += swap_weight * len(swaps) * 3
        score_F += 0.1*len(partition_candidate.circuit_structure)

        for partition_idx in F:
            partition = scoring_partitions[partition_idx]
            if partition is None or partition_idx == partition_candidate.partition_idx:
                continue
            qbit_map_inv = {v: q for q, v in partition.qubit_map.items()}  # q* → circuit qubit
            mini_scores = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                for pdx, (P_i, P_o) in enumerate(partition.permutations_pairs[tdx]):
                    cnot_count = len(partition.circuit_structures[tdx][pdx])
                    if mini_topology:
                        routing_cost = swap_weight * 3 * sum(
                            max(0, D[int(output_perm[qbit_map_inv[P_i[u]]])][int(output_perm[qbit_map_inv[P_i[v]]])] - 1)
                            for u, v in mini_topology
                        )
                    else:
                        routing_cost = 0
                    mini_scores.append(routing_cost + cnot_count)
            if mini_scores:
                score_F += np.min(mini_scores)

        # Virtual outgoing gate penalty: cross-window look-ahead (always active)
        virtual_e_score = 0.0
        if virtual_E:
            for (q_a, q_b) in virtual_E:
                dist = D[int(output_perm[q_a])][int(output_perm[q_b])]
                if not np.isinf(dist):
                    virtual_e_score += max(0, (dist - 1)) * 3

        E_score = 0.3 * virtual_e_score if virtual_e_score > 0.0 else 0.0
        F_score = 0.7 * score_F

        return E_score + F_score

    # ------------------------------------------------------------------------
    # Candidate Generation
    # ------------------------------------------------------------------------

    def obtain_partition_candidates(self, F, optimized_partitions):
        partition_candidates = []
        for partition_idx in F:
            partition = optimized_partitions[partition_idx]
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                # Use pre-computed topology candidates if available, otherwise compute and cache
                if hasattr(partition, 'get_topology_candidates'):
                    topology_candidates = partition.get_topology_candidates(tdx)
                else:
                    topology_candidates = self._get_subtopologies_of_type_cached(mini_topology)
                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(partition.permutations_pairs[tdx]):
                        partition_candidates.append(PartitionCandidate(partition_idx,tdx,pdx,partition.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition.qubit_map,partition.involved_qbits))
        return partition_candidates

    # ------------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------------
        
    def get_initial_layer(self, IDAG, N, optimized_partitions):
        initial_layer = []
        active_qbits = list(range(N))
        for idx in range(len(IDAG)):
            if len(IDAG[idx]) == 0:
                initial_layer.append(idx)
                for qbit in optimized_partitions[idx].involved_qbits:
                    active_qbits.remove(qbit)
            if len(active_qbits) == 0:
                break
        return initial_layer
            
    def construct_DAG_and_IDAG(self, optimized_partitions):
        DAG = []
        IDAG = []
        for idx in range(len(optimized_partitions)):
            parents = []
            children = []
            if idx != len(optimized_partitions)-1:
                involved_qbits_current = optimized_partitions[idx].involved_qbits.copy()
                for next_idx in range(idx+1, len(optimized_partitions)):
                    involved_qbits_next = optimized_partitions[next_idx].involved_qbits
                    intersection = [i for i in involved_qbits_current if i in involved_qbits_next]
                    if len(intersection) > 0:
                        children.append(next_idx)
                    for intersection_qbit in intersection:
                        involved_qbits_current.remove(intersection_qbit)
                    if len(involved_qbits_current) == 0:
                        break
            if idx != 0:
                involved_qbits_current = optimized_partitions[idx].involved_qbits.copy()
                for prev_idx in range(idx-1, -1, -1):
                    involved_qbits_prev = optimized_partitions[prev_idx].involved_qbits
                    intersection = [i for i in involved_qbits_current if i in involved_qbits_prev]
                    if len(intersection) > 0:
                        parents.append(prev_idx)
                    for intersection_qbit in intersection:
                        involved_qbits_current.remove(intersection_qbit)
                    if len(involved_qbits_current) == 0:
                        break
            DAG.append(children)
            IDAG.append(parents)
        return DAG, IDAG
    
    def construct_sDAG(self, optimized_partitions):
        sDAG = [[] for _ in range(len(optimized_partitions))]
        
        for idx in range(len(optimized_partitions)):
            # Skip single-qubit partitions
            if len(optimized_partitions[idx].involved_qbits) <= 1:
                continue
                
            children = []
            
            if idx != len(optimized_partitions)-1:
                involved_qbits_current = optimized_partitions[idx].involved_qbits.copy()
                for next_idx in range(idx+1, len(optimized_partitions)):
                    # Skip single-qubit partitions when searching for children
                    if len(optimized_partitions[next_idx].involved_qbits) <= 1:
                        continue
                        
                    involved_qbits_next = optimized_partitions[next_idx].involved_qbits
                    intersection = [i for i in involved_qbits_current if i in involved_qbits_next]
                    if len(intersection) > 0:
                        children.append(next_idx)
                        for intersection_qbit in intersection:
                            involved_qbits_current.remove(intersection_qbit)
                    if len(involved_qbits_current) == 0:
                        break                        
            sDAG[idx] = children
            
        return sDAG

    # ------------------------------------------------------------------------
    # Distance & Layout
    # ------------------------------------------------------------------------

    def compute_distances_bfs(self, N):
        """BFS distance computation - faster than Floyd-Warshall."""
        D = np.ones((N, N)) * np.inf
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in self.config['topology']:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for start in range(N):
            D[start][start] = 0
            queue = deque([(start, 0)])
            visited = {start}
            
            while queue:
                node, dist = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        D[start][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        
        return D #multiply by 3 to make it CNOT cost instead of SWAP cost


    def _compute_smart_initial_layout(self, circuit, N, D):
        """
        Compute initial layout using interaction graph + simulated annealing.
        Much better than the greedy approach.
        """
        # Build interaction graph: weight = number of CNOTs between qubits
        interaction_graph = defaultdict(int)
        gates = circuit.get_Gates()
        
        for gate in gates:
            if gate.get_Control_Qbit() != -1:
                q1, q2 = sorted([gate.get_Target_Qbit(), gate.get_Control_Qbit()])
                if q1 < N and q2 < N:
                    interaction_graph[(q1, q2)] += 1
        
        # If no 2-qubit gates, return identity
        if not interaction_graph:
            return np.arange(N)
        
        # Start with greedy mapping as baseline
        pi_greedy = self._greedy_initial_layout(interaction_graph, N, D)
        best_pi = pi_greedy.copy()
        best_score = self._evaluate_layout_score(best_pi, interaction_graph, D)
        
        # Simulated annealing to improve
        current_pi = best_pi.copy()
        current_score = best_score
        
        # Temperature schedule
        max_iter = 100 * N
        for iteration in range(max_iter):
            temp = 1.0 - (iteration / max_iter)
            
            # Propose swap of two physical qubits
            p1, p2 = np.random.choice(N, 2, replace=False)
            new_pi = current_pi.copy()
            new_pi[p1], new_pi[p2] = new_pi[p2], new_pi[p1]  # Swap assignments
            
            # Evaluate new layout
            new_score = self._evaluate_layout_score(new_pi, interaction_graph, D)
            
            # Accept if better or with probability
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / (temp + 1e-6)):
                current_pi = new_pi
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_pi = current_pi.copy()
        
        return best_pi
    
    def _greedy_initial_layout(self, interaction_graph, N, D):
        """Greedy baseline mapping - much simpler and reliable"""
        pi = np.arange(N)
        placed_logical = set()
        placed_physical = set()
        
        # Sort interactions by weight (descending)
        sorted_interactions = sorted(
            interaction_graph.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Place highest interaction pair first
        if sorted_interactions:
            (q1, q2), _ = sorted_interactions[0]
            # Find closest physical pair
            min_dist = float('inf')
            best_pair = None
            for p1 in range(N):
                for p2 in range(p1 + 1, N):
                    if D[p1][p2] < min_dist:
                        min_dist = D[p1][p2]
                        best_pair = (p1, p2)
            
            if best_pair:
                p1, p2 = best_pair
                pi[q1] = p1
                pi[q2] = p2
                placed_logical = {q1, q2}
                placed_physical = {p1, p2}
        
        # Place remaining qubits
        remaining_logical = [q for q in range(N) if q not in placed_logical]
        for q in remaining_logical:
            best_p = None
            best_cost = float('inf')
            
            for p in range(N):
                if p in placed_physical:
                    continue
                
                # Cost = sum of distances to already placed interacting qubits
                cost = 0
                for other_q in placed_logical:
                    weight = interaction_graph.get(tuple(sorted((q, other_q))), 0)
                    if weight > 0:
                        other_p = pi[other_q]
                        cost += D[p][other_p] * weight
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p
            
            if best_p is not None:
                pi[q] = best_p
                placed_logical.add(q)
                placed_physical.add(best_p)
        
        return pi
    
    def _evaluate_layout_score(self, pi, interaction_graph, D):
        """
        Evaluate layout quality: lower score is better.
        Score = sum(distance(physical_q1, physical_q2) * interaction_weight)
        """
        score = 0.0
        for (q1, q2), weight in interaction_graph.items():
            p1, p2 = pi[q1], pi[q2]
            distance = D[p1][p2]
            if np.isinf(distance):
                return float('inf')  # Invalid layout
            score += distance * weight
        
        return score
    
    def generate_DAG_levels(self, circuit):
        """
        Generate DAG levels - groups gates by their topological level.
        
        Args:
            circuit: The quantum circuit to analyze
            
        Returns:
            List of lists, where each inner list contains gate indices at the same DAG level.
            Level 0 contains gates with no parents, level 1 contains gates whose parents
            are all at level 0, etc.
        """ 
        gates = circuit.get_Gates()
        num_gates = len(gates)
        
        # Build parent count for each gate
        parent_counts = [0] * num_gates
        children_map = [[] for _ in range(num_gates)]
        
        for gate_idx in range(num_gates):
            gate = gates[gate_idx]
            parents = circuit.get_Parents(gate)
            parent_counts[gate_idx] = len(parents)
            
            # Build children map
            children = circuit.get_Children(gate)
            for child_idx in children:
                children_map[gate_idx].append(child_idx)
        
        # Initialize level 0 with gates that have no parents
        levels = []
        current_level = []
        processed = [False] * num_gates
        
        # Find gates with no parents (level 0)
        for gate_idx in range(num_gates):
            if parent_counts[gate_idx] == 0:
                current_level.append(gate_idx)
                processed[gate_idx] = True
        
        # Process levels using BFS
        while current_level:
            levels.append(current_level)
            next_level = []
            
            # Process all gates in current level
            for gate_idx in current_level:
                # Decrement parent counts for children
                for child_idx in children_map[gate_idx]:
                    parent_counts[child_idx] -= 1
                    # If all parents are processed, add to next level
                    if parent_counts[child_idx] == 0 and not processed[child_idx]:
                        next_level.append(child_idx)
                        processed[child_idx] = True
            
            current_level = next_level
        
        return levels

    def get_gate_DAG_level(self, circuit, gate_idx):
        """
        Find the DAG level a specific gate belongs to.

        Args:
            circuit: The quantum circuit to analyze
            gate_idx: Index of the gate within the circuit's gate list

        Returns:
            int: The DAG level the gate belongs to (0-indexed), or -1 if not found.
        """
        levels = self.generate_DAG_levels(circuit)
        for level_idx, level_gates in enumerate(levels):
            if gate_idx in level_gates:
                return level_idx
        return -1

    def get_gate_DAG_level_map(self, circuit):
        """
        Build a mapping from gate index to its DAG level.

        Args:
            circuit: The quantum circuit to analyze

        Returns:
            dict: Mapping {gate_idx: level} for every gate in the circuit.
        """
        levels = self.generate_DAG_levels(circuit)
        gate_to_level = {}
        for level_idx, level_gates in enumerate(levels):
            for gate_idx in level_gates:
                gate_to_level[gate_idx] = level_idx
        return gate_to_level
