"""
This is an implementation of Partition Aware Mapping.
"""
import logging
import multiprocessing as mp
import os
import time
from collections import deque, defaultdict
from itertools import permutations
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.ilp import (
    get_all_partitions,
    _get_topo_order,
    topo_sort_partitions,
    ilp_global_optimal,
)
# Module-level globals for pool workers (set via Pool initializer)
_worker_config = None

def _init_decompose_worker(config):
    global _worker_config
    _worker_config = config

def _decompose_one(Umtx, mini_topology):
    """Pool worker function. Uses config set once by initializer instead of
    pickling it per task."""
    from squander.synthesis.PartAM import qgd_Partition_Aware_Mapping
    return qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(
        Umtx, _worker_config, mini_topology
    )

def _available_cpus():
    """Return the number of CPUs available to this process.

    Respects affinity masks set by taskset, cgroups, SLURM, etc.
    Falls back to mp.cpu_count() on platforms without sched_getaffinity.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return mp.cpu_count()


from squander.synthesis.PartAM_utils import (
    get_subtopologies_of_type,
    get_unique_subtopologies,
    get_canonical_form,
    get_node_mapping,
    compute_automorphisms,
    derive_result_from_automorphism,
    SingleQubitPartitionResult,
    PartitionSynthesisResult,
    PartitionCandidate,
    PartitionScoreData,
    check_circuit_compatibility,
    construct_swap_circuit,
)

_routing_worker_state = None


def _init_layout_trial_worker(state):
    global _routing_worker_state
    from squander.synthesis.PartAM import qgd_Partition_Aware_Mapping

    worker_config = dict(state["config"])
    worker_config["progressbar"] = False

    mapper = qgd_Partition_Aware_Mapping(worker_config)
    mapper._adj = [list(neighbors) for neighbors in state["adj"]]
    mapper._swap_cache = {}

    _routing_worker_state = {
        "mapper": mapper,
        "seeded_pi": np.asarray(state["seeded_pi"]),
        "DAG": state["DAG"],
        "IDAG": state["IDAG"],
        "layout_partitions": state["layout_partitions"],
        "scoring_partitions": state["scoring_partitions"],
        "D": np.asarray(state["D"]),
        "candidate_cache": state["candidate_cache"],
        "n_iterations": state["n_iterations"],
        "n_trials": state["n_trials"],
        "random_seed": state["random_seed"],
    }


def _run_layout_trial_worker(trial_idx):
    state = _routing_worker_state
    mapper = state["mapper"]

    return mapper._run_single_layout_trial(
        trial_idx=trial_idx,
        seeded_pi=state["seeded_pi"],
        DAG=state["DAG"],
        IDAG=state["IDAG"],
        layout_partitions=state["layout_partitions"],
        scoring_partitions=state["scoring_partitions"],
        D=state["D"],
        candidate_cache=state["candidate_cache"],
        n_iterations=state["n_iterations"],
        n_trials=state["n_trials"],
        random_seed=state["random_seed"],
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
        self.config.setdefault('optimizer', 'BFGS')
        self.config.setdefault('use_osr', 0)
        self.config.setdefault("use_graph_search", 0)
        self.config.setdefault('n_layout_trials', 1)
        self.config.setdefault('random_seed', 42)
        self.config.setdefault('cleanup', True)
        self.config.setdefault('prefilter_top_k', 50)
        self.config.setdefault('prefilter_min_per_partition', 2)
        self.config.setdefault('prefilter_min_3q', 12)
        self.config.setdefault('cleanup_top_k', 3)
        self.config.setdefault('decay_delta', 0.001)  # Qiskit LightSABRE DECAY_RATE
        self.config.setdefault('swap_burst_budget', 5)  # Qiskit LightSABRE DECAY_RESET_INTERVAL
        self.config.setdefault('path_tiebreak_weight', 0.2)
        # The neighbor heuristic is normalized to [0, 1] and added to A*'s f-value.
        # g-deltas are integer and h-deltas are half-integer, so preserving
        # swap-count optimality requires weight < 0.5.
        if self.config['path_tiebreak_weight'] >= 0.5:
            logging.warning(
                "path_tiebreak_weight=%.3f ≥ 0.5 may override SWAP-count "
                "optimality; clamping to 0.49.",
                self.config['path_tiebreak_weight'],
            )
            self.config['path_tiebreak_weight'] = 0.49
        self.config.setdefault('cnot_cost', 1.0 / 3.0)  # 1 SWAP = 3 CNOTs
        self.config.setdefault('three_qubit_exit_weight', 1.0)
        self.config.setdefault('boundary_beam_width', 1)
        self.config.setdefault('boundary_beam_depth', 1)
        self.config.setdefault('size_density_weight', False)
        self.config.setdefault('sparse_penalty', 3.0)
        self.config.setdefault('partition_weight_model', 'density')
        strategy = self.config['strategy']
        self.config.setdefault('parallel_layout_trials', False)
        self.config.setdefault('layout_trial_workers', 0)
        allowed_strategies = ['TreeSearch', 'TabuSearch', 'Adaptive']
        if not strategy in allowed_strategies:
            raise Exception(f"The strategy should be either of {allowed_strategies}, got {strategy}.")
        
        # Initialize caches for performance optimization
        self._topology_cache = {}  # {frozenset(edges): [topology_candidates]}
        self._swap_cache = {}     # {(pi_tuple, qbit_map_frozen): (swaps, output_perm)}
        self._adj = None          # Precomputed adjacency list (built by compute_distances_bfs)
        self._decomp_cache = {}   # {(rounded unitary bytes, topology): synthesis result}

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

    # ------------------------------------------------------------------------
    # Static Synthesis Helpers (extracted from SynthesizeWideCircuit)
    # ------------------------------------------------------------------------

    @staticmethod
    def _parts_to_density_weights(allparts, gate_dict, sparse_penalty=3.0):
        """Per-part ILP weights that penalise sparse 3-qubit partitions.

        Penalty by active-pair count for a 3q partition:
          1 pair  -> sparse_penalty        (e.g. 3 -> total ILP cost 4)
          2 pairs -> sparse_penalty / 3    (e.g. 1 -> total ILP cost 2)
          3 pairs -> 0                     (no penalty)
        For 2q (or 1q) partitions the weight is always 0.
        """
        N = max(len(allparts), 1)
        weights = []
        for part in allparts:
            qubits_in_part = set()
            for gate_idx in part:
                gate = gate_dict.get(gate_idx)
                if gate is not None:
                    qubits_in_part.update(gate.get_Involved_Qbits())
            if len(qubits_in_part) != 3:
                weights.append(0.0)
                continue
            active_pairs = set()
            for gate_idx in part:
                gate = gate_dict.get(gate_idx)
                if gate is None:
                    continue
                qbs = list(gate.get_Involved_Qbits())
                for a in range(len(qbs)):
                    for b in range(a + 1, len(qbs)):
                        active_pairs.add((min(qbs[a], qbs[b]), max(qbs[a], qbs[b])))
            n_pairs = len(active_pairs)
            if n_pairs >= 3:
                penalty = 0.0
            elif n_pairs == 2:
                penalty = sparse_penalty / 3.0
            else:
                penalty = sparse_penalty
            weights.append(penalty / N)
        return weights

    @staticmethod
    def _part_support_and_active_pairs(part, gate_dict):
        qubits_in_part = set()
        active_pairs = set()
        for gate_idx in part:
            gate = gate_dict.get(gate_idx)
            if gate is None:
                continue
            qbs = list(gate.get_Involved_Qbits())
            qubits_in_part.update(qbs)
            if len(qbs) < 2:
                continue
            for a in range(len(qbs)):
                for b in range(a + 1, len(qbs)):
                    active_pairs.add(
                        (min(qbs[a], qbs[b]), max(qbs[a], qbs[b]))
                    )
        return frozenset(qubits_in_part), frozenset(active_pairs)

    @staticmethod
    def _turnover_between_supports(support_a, support_b):
        if len(support_a) < 2 or len(support_b) < 2:
            return None
        return min(len(support_a), len(support_b)) - len(support_a & support_b)

    @staticmethod
    def _average_turnover(part_idx, part, neighbor_gate_sets,
                          gate_to_parts, allparts, supports):
        turnovers = []
        for gate_set in neighbor_gate_sets:
            for gate_idx in gate_set - part:
                for other_idx in gate_to_parts.get(gate_idx, ()):
                    if other_idx == part_idx:
                        continue
                    other_part = allparts[other_idx]
                    if part & other_part:
                        continue
                    turnover = (
                        qgd_Partition_Aware_Mapping._turnover_between_supports(
                            supports[part_idx],
                            supports[other_idx],
                        )
                    )
                    if turnover is not None:
                        turnovers.append(turnover)
        if not turnovers:
            return None
        return sum(turnovers) / len(turnovers)

    @staticmethod
    def _parts_to_window_turnover_weights(allparts, gate_dict, g):
        """Linear ILP weights for 3q window continuity.

        Dense 3q blocks are only routing-friendly when their local qubit window
        persists into adjacent work.  A block like (0, i, j) followed by
        (0, k, l) replaces two qubits in the 3q window, which is exactly the
        expensive pattern on a line.  This cost keeps 2q parts at conceptual
        cost one and charges 3q parts for active-pair count plus average
        predecessor/successor window turnover.
        """
        N = max(len(allparts), 1)
        supports = []
        active_pairs_by_part = []
        for part in allparts:
            support, active_pairs = (
                qgd_Partition_Aware_Mapping._part_support_and_active_pairs(
                    part,
                    gate_dict,
                )
            )
            supports.append(support)
            active_pairs_by_part.append(active_pairs)

        gate_to_parts = defaultdict(list)
        for part_idx, part in enumerate(allparts):
            for gate_idx in part:
                gate_to_parts[gate_idx].append(part_idx)

        rg = defaultdict(set)
        for src, dsts in g.items():
            for dst in dsts:
                rg[dst].add(src)

        weights = []
        for part_idx, part in enumerate(allparts):
            support = supports[part_idx]
            active_pairs = active_pairs_by_part[part_idx]
            if len(support) < 3:
                weights.append(0.0)
                continue

            succ_gate_sets = [g.get(gate_idx, set()) for gate_idx in part]
            pred_gate_sets = [rg.get(gate_idx, set()) for gate_idx in part]
            succ_turnover = qgd_Partition_Aware_Mapping._average_turnover(
                part_idx,
                part,
                succ_gate_sets,
                gate_to_parts,
                allparts,
                supports,
            )
            pred_turnover = qgd_Partition_Aware_Mapping._average_turnover(
                part_idx,
                part,
                pred_gate_sets,
                gate_to_parts,
                allparts,
                supports,
            )
            boundary_turnover = len(support)
            if succ_turnover is None:
                succ_turnover = boundary_turnover
            if pred_turnover is None:
                pred_turnover = boundary_turnover
            conceptual_cost = (
                max(len(support), len(active_pairs), 1)
                + succ_turnover
                + pred_turnover
            )
            weights.append((conceptual_cost - 1.0) / N)
        return weights

    @staticmethod
    def _side_window_turnover_cnot_cost(support, neighbor_support):
        if len(support) < 3 or len(neighbor_support) < 2:
            return None
        entering_or_leaving = len(support - neighbor_support)
        if entering_or_leaving == 0:
            return 0.0

        # A new qubit in a 3q window implies at least one SWAP on a line.
        # If both sides are 3q candidates the boundary is seen from both
        # candidate scores, so each side pays half of the 3-CNOT SWAP cost.
        cnot_per_window_qubit = 1.5 if len(neighbor_support) >= 3 else 3.0
        return cnot_per_window_qubit * entering_or_leaving

    @staticmethod
    def _average_window_cnot_cost(part_idx, part, neighbor_gate_sets,
                                  gate_to_parts, allparts, supports):
        costs = []
        support = supports[part_idx]
        turnover_cost = (
            qgd_Partition_Aware_Mapping._side_window_turnover_cnot_cost
        )
        for gate_set in neighbor_gate_sets:
            for gate_idx in gate_set - part:
                for other_idx in gate_to_parts.get(gate_idx, ()):
                    if other_idx == part_idx:
                        continue
                    other_part = allparts[other_idx]
                    if part & other_part:
                        continue
                    cost = turnover_cost(support, supports[other_idx])
                    if cost is not None:
                        costs.append(cost)
        if not costs:
            return 0.0
        return sum(costs) / len(costs)

    @staticmethod
    def _parts_to_window_turnover_cnot_costs(allparts, gate_dict, g):
        supports = []
        for part in allparts:
            support, _ = (
                qgd_Partition_Aware_Mapping._part_support_and_active_pairs(
                    part,
                    gate_dict,
                )
            )
            supports.append(support)

        gate_to_parts = defaultdict(list)
        for part_idx, part in enumerate(allparts):
            for gate_idx in part:
                gate_to_parts[gate_idx].append(part_idx)

        rg = defaultdict(set)
        for src, dsts in g.items():
            for dst in dsts:
                rg[dst].add(src)

        costs = []
        for part_idx, part in enumerate(allparts):
            support = supports[part_idx]
            if len(support) < 3:
                costs.append(0.0)
                continue
            succ_gate_sets = [g.get(gate_idx, set()) for gate_idx in part]
            pred_gate_sets = [rg.get(gate_idx, set()) for gate_idx in part]
            costs.append(
                qgd_Partition_Aware_Mapping._average_window_cnot_cost(
                    part_idx,
                    part,
                    pred_gate_sets,
                    gate_to_parts,
                    allparts,
                    supports,
                )
                + qgd_Partition_Aware_Mapping._average_window_cnot_cost(
                    part_idx,
                    part,
                    succ_gate_sets,
                    gate_to_parts,
                    allparts,
                    supports,
                )
            )
        return costs

    @staticmethod
    def _subcircuit_from_gate_set(gates, gate_dict, parameters, go, rgo,
                                  gate_to_qubit, qbit_num):
        subcircuit = Circuit(qbit_num)
        subparams = []
        ordered_gates = _get_topo_order(
            {gate_idx: go[gate_idx] & gates for gate_idx in gates},
            {gate_idx: rgo[gate_idx] & gates for gate_idx in gates},
            gate_to_qubit,
        )
        for gate_idx in ordered_gates:
            gate = gate_dict[gate_idx]
            subcircuit.add_Gate(gate)
            start = gate.get_Parameter_Start_Index()
            stop = start + gate.get_Parameter_Num()
            subparams.append(parameters[start:stop])
        return subcircuit, np.concatenate(subparams, axis=0)

    def _meta_from_gate_set(self, gates, gate_dict, parameters, go, rgo,
                            gate_to_qubit, qbit_num):
        subcircuit, subparams = self._subcircuit_from_gate_set(
            gates,
            gate_dict,
            parameters,
            go,
            rgo,
            gate_to_qubit,
            qbit_num,
        )
        involved_qbits = subcircuit.get_Qbits()
        qbit_num_sub = len(involved_qbits)
        qbit_map = {
            involved_qbits[idx]: idx for idx in range(len(involved_qbits))
        }
        remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, qbit_num_sub)
        return {
            'N': qbit_num_sub,
            'circuit': remapped_subcircuit,
            'params': subparams,
            'mini_topologies': get_unique_subtopologies(
                self.topology,
                qbit_num_sub,
            ),
            'involved_qbits': involved_qbits,
            'qbit_map': qbit_map,
            'original_cnot_count': subcircuit.get_Gate_Nums().get('CNOT', 0),
        }

    @staticmethod
    def _synthesis_score_pairs(N):
        identity = tuple(range(N))
        pairs = []
        seen = set()
        for perm in permutations(range(N)):
            for pair in ((identity, tuple(perm)), (tuple(perm), identity)):
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
        return pairs

    def _synthesis_score_fallback(self, meta):
        best_cost = float('inf')
        for mini_topology in meta['mini_topologies']:
            fb_circuit, _ = self._qiskit_routing_fallback(meta, mini_topology)
            if fb_circuit is not None:
                best_cost = min(
                    best_cost,
                    fb_circuit.get_Gate_Nums().get('CNOT', 0),
                )
        if best_cost < float('inf'):
            return best_cost
        return max(1, int(meta.get('original_cnot_count', 1)))

    def _parts_to_synthesis_cnot_weights(self, allparts, gate_dict, parameters,
                                         go, rgo, gate_to_qubit, qbit_num,
                                         gate_dag=None,
                                         include_window_route_cost=False):
        """Linear ILP weights from measured SeqPAM CNOT cost.

        Each candidate partition is synthesized over the input-identity and
        output-identity boundary sweeps, i.e. up to 2*N! local decompositions
        per local topology.  The selected partitions are later fully
        enumerated by _run_parallel_synthesis, reusing the shared decomposition
        cache populated here.

        The ILP objective always keeps the original one-unit partition cost.
        The measured CNOT score is an additional cost, not a replacement for
        partition count; otherwise small local CNOT savings can fragment the
        circuit into many routing boundaries.
        """
        N_parts = max(len(allparts), 1)
        metas = []
        scores = [None] * len(allparts)

        for part_idx, part in enumerate(allparts):
            meta = self._meta_from_gate_set(
                part,
                gate_dict,
                parameters,
                go,
                rgo,
                gate_to_qubit,
                qbit_num,
            )
            metas.append(meta)
            if meta['N'] < 2:
                scores[part_idx] = 0

        disable_pbar = self.config.get('progressbar', 0) == False
        futures = []
        cached = []
        n_cpus = _available_cpus()

        with Pool(processes=n_cpus, initializer=_init_decompose_worker,
                  initargs=(self.config,)) as pool:
            for part_idx, meta in enumerate(metas):
                if scores[part_idx] is not None:
                    continue
                pairs = self._synthesis_score_pairs(meta['N'])
                for topology_idx, mini_topology in enumerate(
                    meta['mini_topologies']
                ):
                    for P_i, P_o in pairs:
                        Umtx = self._build_permuted_unitary(meta, P_i, P_o)
                        ck = self._cache_key(Umtx, mini_topology)
                        if ck in self._decomp_cache:
                            cached.append((part_idx, ck))
                        else:
                            future = pool.apply_async(
                                _decompose_one,
                                (Umtx, mini_topology),
                            )
                            futures.append((part_idx, ck, future))

            for part_idx, ck in cached:
                _, _, synth_err = self._decomp_cache[ck]
                if synth_err <= self.config['tolerance']:
                    synth_circuit, _, _ = self._decomp_cache[ck]
                    cnot_count = synth_circuit.get_Gate_Nums().get('CNOT', 0)
                    if scores[part_idx] is None:
                        scores[part_idx] = cnot_count
                    else:
                        scores[part_idx] = min(scores[part_idx], cnot_count)

            for part_idx, ck, future in tqdm(
                futures,
                desc="Partition Weight Synthesis",
                disable=disable_pbar,
            ):
                synth_circuit, synth_params, synth_err = future.get()
                self._decomp_cache[ck] = (
                    synth_circuit,
                    synth_params,
                    synth_err,
                )
                if synth_err <= self.config['tolerance']:
                    cnot_count = synth_circuit.get_Gate_Nums().get('CNOT', 0)
                    if scores[part_idx] is None:
                        scores[part_idx] = cnot_count
                    else:
                        scores[part_idx] = min(scores[part_idx], cnot_count)

        for part_idx, score in enumerate(scores):
            if score is None:
                scores[part_idx] = self._synthesis_score_fallback(
                    metas[part_idx],
                )

        if include_window_route_cost and gate_dag is not None:
            window_route_costs = self._parts_to_window_turnover_cnot_costs(
                allparts,
                gate_dict,
                gate_dag,
            )
            scores = [
                float(score) + window_route_costs[part_idx]
                for part_idx, score in enumerate(scores)
            ]

        self._partition_synthesis_cnot_scores = list(scores)
        return [float(score) / N_parts for score in scores]

    @staticmethod
    def _topo_key(mini_topology):
        return tuple(sorted(tuple(sorted(e)) for e in mini_topology))

    @staticmethod
    def _cache_key(Umtx, mini_topology):
        topo_key = tuple(sorted(tuple(sorted(e)) for e in mini_topology))
        return (np.round(Umtx, decimals=10).tobytes(), topo_key)

    @staticmethod
    def _get_auts(mini_topo, aut_cache):
        key = tuple(sorted(tuple(sorted(e)) for e in mini_topo))
        if key not in aut_cache:
            aut_cache[key] = compute_automorphisms(mini_topo)
        return aut_cache[key]

    @staticmethod
    def _build_permuted_unitary(meta, P_i, P_o):
        N = meta['N']
        circ_tmp = Circuit(N)
        circ_tmp.add_Permutation(list(P_i))
        circ_tmp.add_Circuit(meta['circuit'])
        circ_tmp.add_Permutation(list(P_o))
        return circ_tmp.get_Matrix(meta['params'])

    @staticmethod
    def _add_result_with_auts(result, perm_pair, synth_circuit, synth_params,
                              topology_idx, N, mini_topology, known_pairs, pair_key,
                              use_auts, aut_cache):
        """Add a synthesis result and derive automorphism equivalents."""
        result.add_result(perm_pair, synth_circuit, synth_params, topology_idx)
        if use_auts:
            if pair_key not in known_pairs:
                known_pairs[pair_key] = set()
            known_pairs[pair_key].add(perm_pair)
            P_i, P_o = perm_pair
            auts = qgd_Partition_Aware_Mapping._get_auts(mini_topology, aut_cache)
            identity = tuple(range(N))
            for sigma in auts:
                if sigma == identity:
                    continue
                new_P_i, new_P_o, new_circ, new_params = derive_result_from_automorphism(
                    sigma, P_i, P_o, synth_circuit, synth_params, N
                )
                if (new_P_i, new_P_o) not in known_pairs[pair_key]:
                    result.add_result((new_P_i, new_P_o), new_circ, new_params, topology_idx)
                    known_pairs[pair_key].add((new_P_i, new_P_o))

    @staticmethod
    def _qiskit_routing_fallback(meta, mini_topology):
        """Route original partition circuit on mini_topology using Qiskit transpiler.

        Called when unitary synthesis fails to reach tolerance.  Routes the
        original (un-permuted) circuit and returns it with identity P_i/P_o.
        Returns (circuit, params) or (None, None) if Qiskit is unavailable or
        routing fails.
        """
        try:
            from squander.IO_interfaces.Qiskit_IO import get_Qiskit_Circuit, convert_Qiskit_to_Squander
            from qiskit.compiler import transpile
            from qiskit.transpiler import CouplingMap
        except ImportError:
            return None, None

        try:
            qk_circ = get_Qiskit_Circuit(meta['circuit'], meta['params'])
            edges = []
            for u, v in mini_topology:
                edges.append([u, v])
                edges.append([v, u])
            coupling_map = CouplingMap(couplinglist=edges)
            qk_routed = transpile(
                qk_circ,
                coupling_map=coupling_map,
                optimization_level=1,
                basis_gates=['cx', 'u3'],
            )
            return convert_Qiskit_to_Squander(qk_routed)
        except Exception as exc:
            logging.warning("Qiskit routing fallback failed: %s", exc)
            return None, None

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
            cnot_counts = tuple(
                tuple(int(cnot) for cnot in partition.cnot_counts[tdx])
                for tdx in range(len(partition.mini_topologies))
            )

            scoring_partitions.append(
                PartitionScoreData(
                    mini_topologies=mini_topologies,
                    topology_candidates=tuple(topology_candidates),
                    permutations_pairs=permutations_pairs,
                    circuit_structures=circuit_structures,
                    cnot_counts=cnot_counts,
                    qubit_map=dict(partition.qubit_map),
                    involved_qbits=tuple(partition.involved_qbits),
                )
            )
        return scoring_partitions
    @staticmethod
    def _partition_is_single(partition):
        if isinstance(partition, dict):
            return partition.get("is_single", False)
        return isinstance(partition, SingleQubitPartitionResult)


    @staticmethod
    def _partition_involved_qbits(partition):
        if isinstance(partition, dict):
            return partition["involved_qbits"]
        return partition.involved_qbits


    @staticmethod
    def _build_layout_partition_info(optimized_partitions):
        return [
            {
                "is_single": isinstance(
                    partition, SingleQubitPartitionResult
                ),
                "involved_qbits": tuple(partition.involved_qbits),
            }
            for partition in optimized_partitions
        ]
    def _build_partition_candidate_cache(self, scoring_partitions):
        """
        Precompute all PartitionCandidate objects once, grouped by partition_idx.

        Returns:
            tuple where candidate_cache[partition_idx] is a tuple of
            PartitionCandidate objects for that partition. Single-qubit
            partitions get an empty tuple.
        """
        candidate_cache = []

        for partition_idx, partition in enumerate(scoring_partitions):
            if partition is None:
                candidate_cache.append(())
                continue

            cached_candidates = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                topology_candidates = partition.topology_candidates[tdx]
                permutation_pairs = partition.permutations_pairs[tdx]
                circuit_structures = partition.circuit_structures[tdx]
                cnot_counts = partition.cnot_counts[tdx]

                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(permutation_pairs):
                        circuit_structure = circuit_structures[pdx]
                        cached_candidates.append(
                            PartitionCandidate(
                                partition_idx,
                                tdx,
                                pdx,
                                circuit_structure,
                                permutation_pair[0],
                                permutation_pair[1],
                                topology_candidate,
                                mini_topology,
                                partition.qubit_map,
                                partition.involved_qbits,
                                cnot_count=cnot_counts[pdx],
                            )
                        )

            candidate_cache.append(tuple(cached_candidates))

        return tuple(candidate_cache)
    # ------------------------------------------------------------------------
    # Partition Decomposition Methods
    # ------------------------------------------------------------------------

    @staticmethod
    def DecomposePartition_and_Perm(Umtx: np.ndarray, config: dict, mini_topology = None, max_retries: int = 5) -> Circuit:
        """
        Call to decompose a partition. Retries up to max_retries times if the
        decomposition error exceeds the configured tolerance.  Returns the
        best-error attempt across all retries and logs a warning when no
        attempt reaches ``config["tolerance"]``.
        """
        tolerance = config["tolerance"]
        strategy = config["strategy"]

        best_err = float('inf')
        best_circuit = None
        best_params = None

        for attempt in range(max_retries):
            if strategy == "TreeSearch":
                cDecompose = N_Qubit_Decomposition_Tree_Search(Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
            elif strategy == "TabuSearch":
                cDecompose = N_Qubit_Decomposition_Tabu_Search(Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology)
            elif strategy == "Adaptive":
                cDecompose = N_Qubit_Decomposition_adaptive(Umtx.conj().T, level_limit_max=5, level_limit_min=1, topology=mini_topology)
            else:
                raise Exception(f"Unsupported decomposition type: {strategy}")
            cDecompose.set_Verbose(config["verbosity"])
            cDecompose.set_Cost_Function_Variant(3)
            cDecompose.set_Optimization_Tolerance(tolerance)
            cDecompose.set_Optimizer(config["optimizer"])
            cDecompose.Start_Decomposition()

            err = cDecompose.get_Decomposition_Error()
            if err < best_err:
                best_err = err
                best_circuit = cDecompose.get_Circuit()
                best_params = cDecompose.get_Optimized_Parameters()

            if best_err <= tolerance:
                break

        return best_circuit, best_params, best_err

    # ------------------------------------------------------------------------
    # Circuit Synthesis
    # ------------------------------------------------------------------------

    def SynthesizeWideCircuit(self, circ, orig_parameters):
        """
        Partition and synthesize a full circuit.

        Flow:
            1) Enumerate candidate partitions.
            2) ILP-select the minimum-count non-overlapping cover (uniform weights).
            3) Synthesize only the selected partitions via SeqPAM (two-stage P_i/P_o
               sweep over mini_topologies, executed by _run_parallel_synthesis).

        Args:
            circ: The full quantum circuit (must be flat — no subcircuit blocks)
            orig_parameters: Parameters for circ

        Returns:
            optimized_partitions: List of PartitionSynthesisResult /
                SingleQubitPartitionResult, in topological order.
        """
        working_circ = circ
        working_parameters = orig_parameters
        qbit_num = circ.get_Qbit_Num()

        # ---- Phase 0: Compute distance matrix ----
        D = self.compute_distances_bfs(qbit_num)

        # ---- Phase 1: Partition enumeration ----
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(working_circ, self.config["max_partition_size"])
        qbit_num_orig_circuit = working_circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(working_circ.get_Gates())}

        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}

        # ---- Phase 2: ILP partition selection ----
        # By default this minimizes partition count. Optional weight models can
        # replace that unit cost with a routing-oriented conceptual cost while
        # preserving a linear ILP objective.
        ilp_weights = None
        partition_weight_model = self.config.get(
            'partition_weight_model',
            'density',
        )
        if partition_weight_model == 'window_turnover':
            ilp_weights = self._parts_to_window_turnover_weights(
                allparts,
                gate_dict,
                g,
            )
        elif partition_weight_model in (
            'synthesis_cnot',
            'synthesis_route_cnot',
        ):
            ilp_weights = self._parts_to_synthesis_cnot_weights(
                allparts,
                gate_dict,
                working_parameters,
                go,
                rgo,
                gate_to_qubit,
                qbit_num_orig_circuit,
                gate_dag=g,
                include_window_route_cost=(
                    partition_weight_model == 'synthesis_route_cnot'
                ),
            )
        elif self.config.get('size_density_weight', False):
            sparse_penalty = float(self.config.get('sparse_penalty', 3.0))
            ilp_weights = self._parts_to_density_weights(
                allparts, gate_dict, sparse_penalty=sparse_penalty
            )
        L_parts, _ = ilp_global_optimal(allparts, g, weights=ilp_weights)

        # ---- Phase 3: Build gate sets for selected partitions (+ standalone chains) ----
        selected_surrounded_starts = set()
        selected_parts_gates = []
        for i in L_parts:
            part = allparts[i]
            surrounded = {t for s in part for t in go[s]
                          if t in single_qubit_chains_prepost
                          and go[single_qubit_chains_prepost[t][-1]]
                          and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
            gates = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded))
            selected_parts_gates.append(gates)
            selected_surrounded_starts.update(surrounded)

        standalone_chains = []
        for chain in single_qubit_chains:
            if chain[0] not in selected_surrounded_starts:
                selected_parts_gates.append(frozenset(chain))
                standalone_chains.append(chain)

        n_multi = len(L_parts)

        size_counts = {}
        for gates in selected_parts_gates:
            involved = set()
            for g in gates:
                involved.update(gate_dict[g].get_Involved_Qbits())
            size = len(involved)
            size_counts[size] = size_counts.get(size, 0) + 1
        self._selected_partition_counts = dict(size_counts)
        if self.config.get('verbosity', 0) > 0:
            selected_multi = sum(
                count for size, count in size_counts.items() if size > 1
            )
            print(
                "Selected partitions: "
                f"2-qubit={size_counts.get(2, 0)}, "
                f"3-qubit={size_counts.get(3, 0)}, "
                f"total_multi={selected_multi}"
            )

        # ---- Phase 4: Assemble partitioned circuit from selected partitions only ----
        partitioned_circuit = Circuit(qbit_num_orig_circuit)
        params = []

        for gates in selected_parts_gates[:n_multi]:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in _get_topo_order({x: go[x] & gates for x in gates},
                                            {x: rgo[x] & gates for x in gates},
                                            gate_to_qubit):
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)

        for chain in standalone_chains:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in chain:
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)

        parameters = np.concatenate(params, axis=0)

        # ---- Phase 5: SeqPAM synthesis on selected partitions only ----
        subcircuits = partitioned_circuit.get_Gates()
        optimized_results = [None] * len(subcircuits)
        partition_meta = []
        for partition_idx, subcircuit in enumerate(subcircuits):
            start_idx = subcircuit.get_Parameter_Start_Index()
            end_idx = start_idx + subcircuit.get_Parameter_Num()
            subcircuit_parameters = parameters[start_idx:end_idx]
            involved_qbits = subcircuit.get_Qbits()
            qbit_num_sub = len(involved_qbits)
            qbit_map = {involved_qbits[idx]: idx for idx in range(len(involved_qbits))}
            remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, qbit_num_sub)

            if qbit_num_sub == 1:
                optimized_results[partition_idx] = SingleQubitPartitionResult(
                    remapped_subcircuit, subcircuit_parameters,
                    original_qubits=list(involved_qbits)
                )
                partition_meta.append(None)
            else:
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num_sub)
                partition_meta.append({
                    'N': qbit_num_sub,
                    'circuit': remapped_subcircuit,
                    'params': subcircuit_parameters,
                    'mini_topologies': mini_topologies,
                    'involved_qbits': involved_qbits,
                    'qbit_map': qbit_map,
                })

        results_map = self._run_parallel_synthesis(partition_meta)
        for partition_idx, result in results_map.items():
            optimized_results[partition_idx] = result

        # ---- Phase 6: Topologically order selected partitions ----
        L = topo_sort_partitions(working_circ, selected_parts_gates)
        return [optimized_results[idx] for idx in L]

    def _run_parallel_synthesis(self, partition_meta):
        """Phase 2: Run parallel synthesis for all multi-qubit partitions.

        Args:
            partition_meta: List of per-partition dicts (None for single-qubit partitions).

        Returns:
            results_map: Dict mapping partition_idx to PartitionSynthesisResult.
        """
        n_cpus = _available_cpus()
        use_auts = self.config.get('use_automorphisms', True)
        disable_pbar = self.config.get('progressbar', 0) == False
        aut_cache = {}
        decomp_cache = self._decomp_cache

        with Pool(processes=n_cpus, initializer=_init_decompose_worker,
                  initargs=(self.config,)) as pool:
            # Initialize PartitionSynthesisResult for each multi-qubit partition
            results_map = {}
            for partition_idx, meta in enumerate(partition_meta):
                if meta is None:
                    continue
                results_map[partition_idx] = PartitionSynthesisResult(
                    meta['N'], meta['mini_topologies'], meta['involved_qbits'],
                    meta['qbit_map'],
                )

            # ---- Stage 1: sweep all boundary permutations for small partitions ----
            # For N<=3 the full (P_i, P_o) space is at most 36 pairs.  Routing
            # needs that complete boundary-state set; otherwise 3q partitions
            # expose less layout freedom than 2q partitions.
            stage1_futures = []
            stage1_cached = []
            known_pairs = {}
            full_enum_keys = set()  # (partition_idx, topology_idx) fully covered in S1

            for partition_idx, meta in enumerate(partition_meta):
                if meta is None:
                    continue
                N = meta['N']
                perms_all = list(permutations(range(N)))
                for topology_idx, mini_topology in enumerate(meta['mini_topologies']):
                    if N <= 3:
                        full_enum_keys.add((partition_idx, topology_idx))
                        po_sweep = perms_all
                    else:
                        po_sweep = [perms_all[np.random.choice(len(perms_all))]]
                    for P_o in po_sweep:
                        for P_i in perms_all:
                            Umtx = self._build_permuted_unitary(meta, P_i, P_o)
                            ck = self._cache_key(Umtx, mini_topology)
                            if ck in decomp_cache:
                                stage1_cached.append((partition_idx, topology_idx, P_i, P_o, ck))
                            else:
                                future = pool.apply_async(
                                    _decompose_one, (Umtx, mini_topology)
                                )
                                stage1_futures.append((partition_idx, topology_idx, P_i, P_o, ck, future))

            # Process Stage 1 cache hits immediately
            for partition_idx, topology_idx, P_i, P_o, ck in stage1_cached:
                meta = partition_meta[partition_idx]
                N = meta['N']
                mini_topology = meta['mini_topologies'][topology_idx]
                synth_circuit, synth_params, synth_err = decomp_cache[ck]
                if synth_err <= self.config['tolerance']:
                    pair_key = (partition_idx, topology_idx)
                    self._add_result_with_auts(
                        results_map[partition_idx], (P_i, P_o),
                        synth_circuit, synth_params, topology_idx,
                        N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                    )

            # Collect Stage 1 pool results
            cache_hits_s1 = len(stage1_cached)
            for partition_idx, topology_idx, P_i, P_o, ck, future in tqdm(
                stage1_futures, desc=f"Stage 1 Synthesis ({cache_hits_s1} cached)",
                disable=disable_pbar
            ):
                synth_circuit, synth_params, synth_err = future.get()
                decomp_cache[ck] = (synth_circuit, synth_params, synth_err)
                meta = partition_meta[partition_idx]
                N = meta['N']
                mini_topology = meta['mini_topologies'][topology_idx]
                if synth_err <= self.config['tolerance']:
                    pair_key = (partition_idx, topology_idx)
                    self._add_result_with_auts(
                        results_map[partition_idx], (P_i, P_o),
                        synth_circuit, synth_params, topology_idx,
                        N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                    )

            # ---- Stage 2: fix top-k P_i from Stage 1, sweep all P_o ----
            # Skipped for partitions already fully enumerated in Stage 1
            # (currently all N<=3 partitions).
            top_k_pi = self.config.get('top_k_pi', 1)
            stage2_futures = []
            stage2_cached = []

            for partition_idx, meta in enumerate(partition_meta):
                if meta is None:
                    continue
                N = meta['N']
                perms_all = list(permutations(range(N)))
                result = results_map[partition_idx]
                for topology_idx, mini_topology in enumerate(meta['mini_topologies']):
                    if (partition_idx, topology_idx) in full_enum_keys:
                        continue
                    pair_key = (partition_idx, topology_idx)
                    kp = known_pairs.get(pair_key, set()) if use_auts else set()
                    for P_i_cand in result.get_top_k_results(topology_idx, top_k_pi):
                        for P_o in perms_all:
                            if use_auts and (tuple(P_i_cand), P_o) in kp:
                                continue
                            Umtx = self._build_permuted_unitary(meta, P_i_cand, P_o)
                            ck = self._cache_key(Umtx, mini_topology)
                            if ck in decomp_cache:
                                stage2_cached.append((partition_idx, topology_idx, P_i_cand, P_o, ck))
                            else:
                                future = pool.apply_async(
                                    _decompose_one, (Umtx, mini_topology)
                                )
                                stage2_futures.append((partition_idx, topology_idx, P_i_cand, P_o, ck, future))

            # Process Stage 2 cache hits
            for partition_idx, topology_idx, P_i_cand, P_o, ck in stage2_cached:
                meta = partition_meta[partition_idx]
                N = meta['N']
                mini_topology = meta['mini_topologies'][topology_idx]
                synth_circuit, synth_params, synth_err = decomp_cache[ck]
                if synth_err <= self.config['tolerance']:
                    pair_key = (partition_idx, topology_idx)
                    self._add_result_with_auts(
                        results_map[partition_idx], (tuple(P_i_cand), P_o),
                        synth_circuit, synth_params, topology_idx,
                        N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                    )

            # Collect Stage 2 pool results
            cache_hits_s2 = len(stage2_cached)
            for partition_idx, topology_idx, P_i_cand, P_o, ck, future in tqdm(
                stage2_futures, desc=f"Stage 2 Synthesis ({cache_hits_s2} cached)",
                disable=disable_pbar
            ):
                synth_circuit, synth_params, synth_err = future.get()
                decomp_cache[ck] = (synth_circuit, synth_params, synth_err)
                meta = partition_meta[partition_idx]
                N = meta['N']
                mini_topology = meta['mini_topologies'][topology_idx]
                if synth_err <= self.config['tolerance']:
                    pair_key = (partition_idx, topology_idx)
                    self._add_result_with_auts(
                        results_map[partition_idx], (tuple(P_i_cand), P_o),
                        synth_circuit, synth_params, topology_idx,
                        N, mini_topology, known_pairs, pair_key, use_auts, aut_cache
                    )

        # Qiskit routing fallback: for any (partition, topology) pair where all
        # synthesis attempts failed (no results stored), route the original circuit
        # with Qiskit and add the result with identity P_i/P_o permutations.
        qiskit_fallback_cache = {}
        for partition_idx, meta in enumerate(partition_meta):
            if meta is None:
                continue
            N = meta['N']
            for topology_idx, mini_topology in enumerate(meta['mini_topologies']):
                if results_map[partition_idx].permutations_pairs[topology_idx]:
                    continue
                fkey = (partition_idx, topology_idx)
                if fkey not in qiskit_fallback_cache:
                    fb_circuit, fb_params = self._qiskit_routing_fallback(meta, mini_topology)
                    qiskit_fallback_cache[fkey] = (fb_circuit, fb_params)
                fb_circuit, fb_params = qiskit_fallback_cache[fkey]
                if fb_circuit is None:
                    logging.warning(
                        "Partition %d topology_idx %d: synthesis failed and Qiskit "
                        "fallback unavailable; no result for this combination.",
                        partition_idx, topology_idx,
                    )
                    continue
                identity = tuple(range(N))
                results_map[partition_idx].add_result(
                    (identity, identity), fb_circuit, fb_params, topology_idx
                )

        return results_map

    # ------------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------------
    def _run_single_layout_trial(
        self,
        trial_idx,
        seeded_pi,
        DAG,
        IDAG,
        layout_partitions,
        scoring_partitions,
        D,
        candidate_cache,
        n_iterations,
        n_trials,
        random_seed,
    ):
        N = len(seeded_pi)
        rng = (
            np.random.RandomState(random_seed + trial_idx)
            if n_trials > 1
            else None
        )
        pi = self._sample_initial_layout(trial_idx, n_trials, seeded_pi, rng)

        for iteration in range(n_iterations):
            F_rev = self.get_final_layer(DAG, N, layout_partitions)
            pi, _ = self._heuristic_search_layout_only(
                F_rev,
                pi,
                IDAG,
                DAG,
                layout_partitions,
                scoring_partitions,
                D,
                rng=rng,
                reverse=True,
                candidate_cache=candidate_cache,
            )

            if iteration < n_iterations - 1:
                F_fwd = self.get_initial_layer(IDAG, N, layout_partitions)
                pi, _ = self._heuristic_search_layout_only(
                    F_fwd,
                    pi,
                    DAG,
                    IDAG,
                    layout_partitions,
                    scoring_partitions,
                    D,
                    rng=rng,
                    candidate_cache=candidate_cache,
                )

        F_eval = self.get_initial_layer(IDAG, N, layout_partitions)
        _, cost = self._heuristic_search_layout_only(
            F_eval,
            pi.copy(),
            DAG,
            IDAG,
            layout_partitions,
            scoring_partitions,
            D,
            rng=None,
            candidate_cache=candidate_cache,
        )
        return cost, pi


    def _run_layout_trials(
        self,
        seeded_pi,
        DAG,
        IDAG,
        layout_partitions,
        scoring_partitions,
        D,
        candidate_cache,
        n_iterations,
        n_trials,
        random_seed,
    ):
        use_cpp = self.config.get('use_cpp_router', True)
        if use_cpp:
            return self._run_layout_trials_cpp(
                seeded_pi, DAG, IDAG, layout_partitions,
                scoring_partitions, D, candidate_cache,
                n_iterations, n_trials, random_seed,
            )

        trial_indices = list(range(max(1, n_trials)))
        use_parallel = (
            self.config.get("parallel_layout_trials", False)
            and len(trial_indices) > 1
        )

        if not use_parallel:
            return [
                self._run_single_layout_trial(
                    trial_idx=trial_idx,
                    seeded_pi=seeded_pi,
                    DAG=DAG,
                    IDAG=IDAG,
                    layout_partitions=layout_partitions,
                    scoring_partitions=scoring_partitions,
                    D=D,
                    candidate_cache=candidate_cache,
                    n_iterations=n_iterations,
                    n_trials=n_trials,
                    random_seed=random_seed,
                )
                for trial_idx in trial_indices
            ]

        workers = self.config.get("layout_trial_workers", 0)
        if workers <= 0:
            workers = min(len(trial_indices), _available_cpus())

        worker_state = {
            "config": dict(self.config),
            "adj": tuple(tuple(neighbors) for neighbors in self._adj),
            "seeded_pi": np.asarray(seeded_pi),
            "DAG": DAG,
            "IDAG": IDAG,
            "layout_partitions": layout_partitions,
            "scoring_partitions": scoring_partitions,
            "D": np.asarray(D),
            "candidate_cache": candidate_cache,
            "n_iterations": n_iterations,
            "n_trials": n_trials,
            "random_seed": random_seed,
        }

        with Pool(
            processes=workers,
            initializer=_init_layout_trial_worker,
            initargs=(worker_state,),
        ) as pool:
            return pool.map(_run_layout_trial_worker, trial_indices)

    def _run_layout_trials_cpp(
        self,
        seeded_pi,
        DAG,
        IDAG,
        layout_partitions,
        scoring_partitions,
        D,
        candidate_cache,
        n_iterations,
        n_trials,
        random_seed,
    ):
        from squander.synthesis._sabre_router import SabreRouter, SabreConfig

        cfg = SabreConfig()
        cfg.prefilter_top_k = self.config.get('prefilter_top_k', 50)
        if hasattr(cfg, 'prefilter_min_per_partition'):
            cfg.prefilter_min_per_partition = self.config.get(
                'prefilter_min_per_partition', 2
            )
        if hasattr(cfg, 'prefilter_min_3q'):
            cfg.prefilter_min_3q = self.config.get('prefilter_min_3q', 12)
        cfg.max_E_size = self.config.get('max_E_size', 20)
        cfg.max_lookahead = self.config.get('max_lookahead', 4)
        cfg.E_weight = self.config.get('E_weight', 0.5)
        cfg.E_alpha = self.config.get('E_alpha', 1.0)
        cfg.cnot_cost = self.config.get('cnot_cost', 1.0 / 3.0)
        cfg.sabre_iterations = n_iterations
        cfg.n_layout_trials = max(1, n_trials)
        cfg.random_seed = random_seed
        cfg.decay_delta = self.config.get('decay_delta', 0.001)
        cfg.swap_burst_budget = self.config.get('swap_burst_budget', 5)
        cfg.path_tiebreak_weight = self.config.get(
            'path_tiebreak_weight', 0.2
        )
        if hasattr(cfg, 'three_qubit_exit_weight'):
            cfg.three_qubit_exit_weight = self.config.get(
                'three_qubit_exit_weight', 1.0
            )
        if hasattr(cfg, 'boundary_beam_width'):
            cfg.boundary_beam_width = self.config.get(
                'boundary_beam_width', 1
            )
        if hasattr(cfg, 'boundary_beam_depth'):
            cfg.boundary_beam_depth = self.config.get(
                'boundary_beam_depth', 1
            )
        canonical_fwd = self._build_canonical_neighbor_data(
            scoring_partitions, reverse=False
        )
        canonical_rev = self._build_canonical_neighbor_data(
            scoring_partitions, reverse=True
        )

        # Convert candidate_cache: list of tuples -> list of lists
        candidate_cache_lists = [list(cands) for cands in candidate_cache]

        # Convert layout_partitions: list of dicts with tuple involved_qbits
        layout_partitions_lists = [
            {'is_single': lp['is_single'], 'involved_qbits': list(lp['involved_qbits'])}
            for lp in layout_partitions
        ]

        router = SabreRouter(
            cfg, D, self._adj, DAG, IDAG,
            candidate_cache_lists, layout_partitions_lists,
            canonical_fwd, canonical_rev,
        )

        seeded_pi_list = [int(x) for x in seeded_pi]
        n_trials_actual = max(1, n_trials)
        trial_indices = list(range(n_trials_actual))

        use_parallel = (
            self.config.get("parallel_layout_trials", False)
            and n_trials_actual > 1
        )

        if not use_parallel:
            trial_results = [
                router.run_trial(idx, seeded_pi_list, n_iterations, n_trials_actual)
                for idx in trial_indices
            ]
        else:
            from concurrent.futures import ThreadPoolExecutor
            workers = self.config.get("layout_trial_workers", 0)
            if workers <= 0:
                workers = min(n_trials_actual, _available_cpus())

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(router.run_trial, idx, seeded_pi_list, n_iterations, n_trials_actual)
                    for idx in trial_indices
                ]
                trial_results = [f.result() for f in futures]

        heuristic_ranked = sorted(trial_results, key=lambda x: x[0])
        actual_rank_default = min(
            max(1, self.config.get("cleanup_top_k", 3) * 2),
            n_trials_actual,
        )
        actual_rank_top_k = self.config.get(
            "actual_routing_rank_top_k", actual_rank_default
        )
        if actual_rank_top_k is None or actual_rank_top_k <= 0:
            actual_rank_top_k = len(heuristic_ranked)
        actual_rank_top_k = min(int(actual_rank_top_k), len(heuristic_ranked))

        ranked = []
        for heuristic_cost, trial_pi in heuristic_ranked[:actual_rank_top_k]:
            actual_cnot, pi_out, pi_init, steps = router.route_forward(
                [int(x) for x in trial_pi]
            )
            ranked.append((actual_cnot, pi_out, heuristic_cost, pi_init, steps))
        ranked.sort(key=lambda x: (x[0], x[2]))
        ranked.extend(
            (float("inf"), pi, cost, None, None)
            for cost, pi in heuristic_ranked[actual_rank_top_k:]
        )
        return ranked
        
    @staticmethod
    def _snapshot_single_qubit_circuits(optimized_partitions):
        return {
            i: p.circuit.copy()
            for i, p in enumerate(optimized_partitions)
            if isinstance(p, SingleQubitPartitionResult)
        }

    @staticmethod
    def _restore_single_qubit_circuits(optimized_partitions, saved_circuits):
        for idx, orig in saved_circuits.items():
            optimized_partitions[idx].circuit = orig.copy()

    @staticmethod
    def _partition_order_cnot_breakdown(partition_order):
        routing_cnot = 0
        partition_cnot = 0
        for part in partition_order:
            if isinstance(part, Circuit):
                routing_cnot += part.get_Gate_Nums().get('CNOT', 0)
            elif isinstance(part, SingleQubitPartitionResult):
                continue
            else:
                partition_cnot += int(getattr(part, 'cnot_count', 0))
        return routing_cnot, partition_cnot

    def _partition_order_from_cpp_steps(
        self, steps, optimized_partitions, candidate_cache, N
    ):
        partition_order = []
        for step in steps:
            kind = step[0]
            if kind == "swap":
                swaps = [(int(u), int(v)) for u, v in step[1]]
                if swaps:
                    partition_order.append(construct_swap_circuit(swaps, N))
            elif kind == "partition":
                partition_idx = int(step[1])
                candidate_idx = int(step[2])
                partition_order.append(
                    candidate_cache[partition_idx][candidate_idx]
                )
            elif kind == "single":
                partition_idx = int(step[1])
                physical_qubit = int(step[2])
                part = optimized_partitions[partition_idx]
                circuit_qubit = int(part.circuit.get_Qbits()[0])
                part.circuit = part.circuit.Remap_Qbits(
                    {circuit_qubit: physical_qubit}, N
                )
                partition_order.append(part)
        return partition_order


    def _rank_layout_trials_by_actual_routing(
        self,
        trial_results,
        DAG,
        IDAG,
        optimized_partitions,
        scoring_partitions,
        D,
        candidate_cache,
        rank_top_k=None,
    ):
        """Reroute a bounded candidate set and rank it by actual CNOT count."""
        if trial_results and len(trial_results[0]) >= 5:
            return sorted(trial_results, key=lambda x: (x[0], x[2]))
        heuristic_ranked = sorted(trial_results, key=lambda x: x[0])
        if rank_top_k is None or rank_top_k <= 0:
            rank_top_k = len(heuristic_ranked)
        rank_top_k = min(int(rank_top_k), len(heuristic_ranked))
        actual_candidates = heuristic_ranked[:rank_top_k]
        heuristic_tail = heuristic_ranked[rank_top_k:]

        saved_sq_circuits = self._snapshot_single_qubit_circuits(
            optimized_partitions
        )
        ranked_results = []
        old_progressbar = self.config.get("progressbar", 0)
        self.config["progressbar"] = False
        try:
            for heuristic_cost, trial_pi in actual_candidates:
                self._restore_single_qubit_circuits(
                    optimized_partitions, saved_sq_circuits
                )
                F_trial = self.get_initial_layer(
                    IDAG, len(trial_pi), optimized_partitions
                )
                partition_order, _, _ = self.Heuristic_Search(
                    F_trial,
                    np.asarray(trial_pi, dtype=np.int64).copy(),
                    DAG,
                    IDAG,
                    optimized_partitions,
                    scoring_partitions,
                    D,
                    candidate_cache=candidate_cache,
                )
                trial_circuit, _ = self.Construct_circuit_from_HS(
                    partition_order, optimized_partitions, len(trial_pi)
                )
                actual_cnot = trial_circuit.get_Gate_Nums().get("CNOT", 0)
                ranked_results.append((actual_cnot, trial_pi, heuristic_cost, None, None))
        finally:
            if old_progressbar is None:
                self.config.pop("progressbar", None)
            else:
                self.config["progressbar"] = old_progressbar
            self._restore_single_qubit_circuits(
                optimized_partitions, saved_sq_circuits
            )

        ranked_results.sort(key=lambda x: (x[0], x[2]))
        ranked_results.extend(
            (float("inf"), pi, cost, None, None) for cost, pi in heuristic_tail
        )
        return ranked_results

    def Partition_Aware_Mapping(
        self, circ: Circuit, orig_parameters: np.ndarray
    ):
        N = circ.get_Qbit_Num()

        optimized_partitions = self.SynthesizeWideCircuit(circ, orig_parameters)

        for partition in optimized_partitions:
            if isinstance(partition, PartitionSynthesisResult):
                partition._topology = self.topology
                partition._topology_cache = self._topology_cache

        DAG, IDAG = self.construct_DAG_and_IDAG(optimized_partitions)

        D = self.compute_distances_bfs(N)
        scoring_partitions = self._build_scoring_partitions(optimized_partitions)
        candidate_cache = self._build_partition_candidate_cache(
            scoring_partitions
        )
        layout_partitions = self._build_layout_partition_info(
            optimized_partitions
        )
        seeded_pi = self._compute_seeded_layout(
            optimized_partitions, D, N, circ
        )

        n_iterations = self.config.get('sabre_iterations', 1)
        n_trials = self.config.get('n_layout_trials', 1)
        random_seed = self.config.get('random_seed', 42)
        do_cleanup = self.config.get('cleanup', True)

        routing_start = time.time()
        routing_swap_cnot = 0
        partition_body_cnot = 0
        routing_elapsed_before_cleanup = None
        cleanup_total = 0.0

        if n_iterations == 0:
            F = self.get_initial_layer(IDAG, N, optimized_partitions)
            partition_order, pi, pi_initial = self.Heuristic_Search(
                F,
                pi=seeded_pi.copy(),
                DAG=DAG,
                IDAG=IDAG,
                optimized_partitions=optimized_partitions,
                scoring_partitions=scoring_partitions,
                D=D,
                candidate_cache=candidate_cache,
            )
            final_circuit, final_parameters = self.Construct_circuit_from_HS(
                partition_order, optimized_partitions, N
            )
            routing_swap_cnot, partition_body_cnot = (
                self._partition_order_cnot_breakdown(partition_order)
            )

        else:
            trial_results = self._run_layout_trials(
                seeded_pi=seeded_pi,
                DAG=DAG,
                IDAG=IDAG,
                layout_partitions=layout_partitions,
                scoring_partitions=scoring_partitions,
                D=D,
                candidate_cache=candidate_cache,
                n_iterations=n_iterations,
                n_trials=max(1, n_trials),
                random_seed=random_seed,
            )
            actual_rank_default = min(
                max(1, self.config.get("cleanup_top_k", 3) * 2),
                max(1, n_trials),
            )
            actual_rank_top_k = self.config.get(
                "actual_routing_rank_top_k", actual_rank_default
            )
            trial_results = self._rank_layout_trials_by_actual_routing(
                trial_results,
                DAG,
                IDAG,
                optimized_partitions,
                scoring_partitions,
                D,
                candidate_cache,
                rank_top_k=actual_rank_top_k,
            )
            routing_elapsed_before_cleanup = time.time() - routing_start

            if do_cleanup:
                from squander.decomposition.qgd_Wide_Circuit_Optimization import (
                    qgd_Wide_Circuit_Optimization,
                )

                cleanup_config = dict(self.config)
                cleanup_config['topology'] = self.topology
                cleanup_config['routed'] = True
                cleanup_config['test_subcircuits'] = False
                cleanup_config['test_final_circuit'] = False
                cleanup_config['global_min'] = True
                cleanup_config['use_osr'] = 0
                cleanup_config['use_graph_search'] = 0
                cleanup_config['part_size_end'] = 3
                cleanup_config['max_partition_size'] = 3

                wco = qgd_Wide_Circuit_Optimization(cleanup_config)

                saved_sq_circuits = self._snapshot_single_qubit_circuits(
                    optimized_partitions
                )

                cleanup_top_k = self.config.get('cleanup_top_k', 3)
                top_layouts = trial_results[:cleanup_top_k]

                best_circuit = None
                best_params = None
                best_pi_init = None
                best_pi = None
                best_cost = float('inf')
                best_pre_cleanup = None
                best_routing_swap_cnot = 0
                best_partition_body_cnot = 0

                for _, trial_pi, _, trace_pi_init, route_steps in top_layouts:
                    self._restore_single_qubit_circuits(
                        optimized_partitions, saved_sq_circuits
                    )
                    if route_steps is not None:
                        partition_order = self._partition_order_from_cpp_steps(
                            route_steps, optimized_partitions, candidate_cache, N
                        )
                        pi_out = np.asarray(trial_pi, dtype=np.int64)
                        pi_init = np.asarray(trace_pi_init, dtype=np.int64)
                    else:
                        F_trial = self.get_initial_layer(
                            IDAG, N, optimized_partitions
                        )
                        partition_order, pi_out, pi_init = self.Heuristic_Search(
                            F_trial,
                            trial_pi.copy(),
                            DAG,
                            IDAG,
                            optimized_partitions,
                            scoring_partitions,
                            D,
                            candidate_cache=candidate_cache,
                        )

                    trial_circuit, trial_params = self.Construct_circuit_from_HS(
                        partition_order, optimized_partitions, N
                    )
                    pre_cleanup_cnots = trial_circuit.get_Gate_Nums().get(
                        'CNOT', 0
                    )
                    trial_routing_cnot, trial_partition_cnot = (
                        self._partition_order_cnot_breakdown(partition_order)
                    )

                    cleanup_t0 = time.time()
                    cleaned_circuit, cleaned_params = wco.OptimizeWideCircuit(
                        trial_circuit.get_Flat_Circuit(),
                        trial_params,
                    )
                    cleaned_cost = cleaned_circuit.get_Gate_Nums().get(
                        'CNOT', 0
                    )
                    cleanup_total += time.time() - cleanup_t0

                    if cleaned_cost < best_cost:
                        best_cost = cleaned_cost
                        best_pre_cleanup = pre_cleanup_cnots
                        best_circuit = cleaned_circuit
                        best_params = cleaned_params
                        best_pi_init = pi_init
                        best_pi = pi_out
                        best_routing_swap_cnot = trial_routing_cnot
                        best_partition_body_cnot = trial_partition_cnot

                final_cleanup_config = dict(cleanup_config)
                final_cleanup_config['use_osr'] = 1
                final_cleanup_config['use_graph_search'] = 1
                final_cleanup_config['part_size_end'] = 4

                wco = qgd_Wide_Circuit_Optimization(final_cleanup_config)

                cleanup_t0 = time.time()
                final_circuit, final_parameters = wco.OptimizeWideCircuit(
                    best_circuit.get_Flat_Circuit(),
                    best_params,
                )
                cleanup_total += time.time() - cleanup_t0
                pi_initial = best_pi_init
                pi = best_pi
                routing_swap_cnot = best_routing_swap_cnot
                partition_body_cnot = best_partition_body_cnot

            else:
                _, best_pi, _, trace_pi_init, route_steps = trial_results[0]

                if route_steps is not None:
                    saved_sq_circuits = self._snapshot_single_qubit_circuits(
                        optimized_partitions
                    )
                    self._restore_single_qubit_circuits(
                        optimized_partitions, saved_sq_circuits
                    )
                    partition_order = self._partition_order_from_cpp_steps(
                        route_steps, optimized_partitions, candidate_cache, N
                    )
                    pi = np.asarray(best_pi, dtype=np.int64)
                    pi_initial = np.asarray(trace_pi_init, dtype=np.int64)
                else:
                    F = self.get_initial_layer(IDAG, N, optimized_partitions)
                    partition_order, pi, pi_initial = self.Heuristic_Search(
                        F,
                        best_pi.copy(),
                        DAG,
                        IDAG,
                        optimized_partitions,
                        scoring_partitions,
                        D,
                        candidate_cache=candidate_cache,
                    )
                final_circuit, final_parameters = self.Construct_circuit_from_HS(
                    partition_order, optimized_partitions, N
                )
                routing_swap_cnot, partition_body_cnot = (
                    self._partition_order_cnot_breakdown(partition_order)
                )

        if do_cleanup and n_iterations > 0:
            self._routing_time = routing_elapsed_before_cleanup
            self._cleanup_time = cleanup_total
            self._cnot_pre_cleanup = best_pre_cleanup
        else:
            self._routing_time = time.time() - routing_start
            self._cleanup_time = 0.0
            self._cnot_pre_cleanup = final_circuit.get_Gate_Nums().get(
                'CNOT', 0
            )

            if self.config.get('cleanup', True):
                from squander.decomposition.qgd_Wide_Circuit_Optimization import (
                    qgd_Wide_Circuit_Optimization,
                )

                cleanup_config = dict(self.config)
                cleanup_config['topology'] = self.topology
                cleanup_config['routed'] = True
                cleanup_config['test_subcircuits'] = False
                cleanup_config['test_final_circuit'] = False
                cleanup_config['global_min'] = True
                wco = qgd_Wide_Circuit_Optimization(cleanup_config)

                final_circuit, final_parameters = wco.OptimizeWideCircuit(
                    final_circuit.get_Flat_Circuit(), final_parameters
                )

        self._routing_swap_cnot = routing_swap_cnot
        self._partition_body_cnot = partition_body_cnot

        return final_circuit, final_parameters, pi_initial, pi

    # ------------------------------------------------------------------------
    # Heuristic Search
    # ------------------------------------------------------------------------

    def _select_best_candidate(self, partition_candidates, scores, rng=None):
        """Select the lowest-scoring candidate deterministically."""
        del rng
        scores_array = np.array(scores)
        return partition_candidates[np.argmin(scores_array)]

    def _prefilter_candidates(
        self,
        partition_candidates,
        pi,
        D,
        top_k,
        F=None,
        E=None,
        candidate_cache=None,
        layout_partitions=None,
        reverse=False,
        W=0.5,
        alpha=1.0,
        canonical_data=None,
    ):
        """Pre-filter candidates using cheap swap-count estimate before full A* scoring."""
        if top_k <= 0:
            return []
        if len(partition_candidates) <= top_k:
            return partition_candidates
        cnot_cost = self.config.get('cnot_cost', 1.0 / 3.0)
        estimates = np.array([
            (
                self._routing_objective(
                    pc.estimate_swap_count(pi, D, reverse=reverse),
                    pc.cnot_count,
                    cnot_cost,
                )
                + self._future_context_cost(
                    pc.partition_idx,
                    self._estimate_candidate_output_layout(
                        pc, pi, reverse=reverse
                    ),
                    F or (),
                    E or (),
                    D,
                    candidate_cache,
                    reverse=reverse,
                    cnot_cost=cnot_cost,
                    W=W,
                    alpha=alpha,
                    layout_partitions=layout_partitions,
                    canonical_data=canonical_data,
                )
            )
            for pc in partition_candidates
        ])
        selected = set()
        min_per_partition = int(
            self.config.get('prefilter_min_per_partition', 0) or 0
        )
        min_3q = int(self.config.get('prefilter_min_3q', 0) or 0)
        if min_per_partition > 0 or min_3q > 0:
            by_partition = defaultdict(list)
            for idx, pc in enumerate(partition_candidates):
                by_partition[pc.partition_idx].append(idx)
            for indices in by_partition.values():
                sample = partition_candidates[indices[0]]
                quota = min_per_partition
                if len(sample.involved_qbits) >= 3:
                    quota = max(quota, min_3q)
                if quota <= 0:
                    continue
                ranked = sorted(indices, key=lambda i: estimates[i])
                selected.update(ranked[:min(quota, len(ranked))])

        remaining = max(0, top_k - len(selected))
        if remaining > 0:
            ranked_global = np.argsort(estimates)
            for idx in ranked_global:
                selected.add(int(idx))
                if len(selected) >= top_k:
                    break

        if not selected:
            top_k_indices = np.argpartition(estimates, top_k)[:top_k]
            selected.update(int(i) for i in top_k_indices)

        return [
            partition_candidates[i]
            for i in sorted(selected, key=lambda idx: estimates[idx])
        ]

    @staticmethod
    def _decay_factor_for_swaps(swaps, decay):
        if not swaps:
            return 1.0
        return max(max(decay[u], decay[v]) for u, v in swaps)

    @staticmethod
    def _routing_objective(
        route_cost,
        cnot_count,
        cnot_cost,
        cnot_weight=1.0,
        decay_factor=1.0,
    ):
        return decay_factor * (
            float(route_cost)
            + cnot_weight * cnot_cost * float(cnot_count)
        )

    def _apply_decay_for_swaps(self, swaps, decay):
        delta = self.config.get("decay_delta", 0.001)
        if delta <= 0:
            return
        for u, v in swaps:
            decay[u] += delta
            decay[v] += delta

    @staticmethod
    def _reset_decay(decay):
        for idx in range(len(decay)):
            decay[idx] = 1.0

    @staticmethod
    def _apply_swaps_to_pi(pi, swaps):
        pi_new = [int(x) for x in pi]
        n = len(pi_new)
        p2v = [0] * n
        for q in range(n):
            p2v[pi_new[q]] = q
        for P1, P2 in swaps:
            q1, q2 = p2v[P1], p2v[P2]
            p2v[P1], p2v[P2] = q2, q1
            pi_new[q1], pi_new[q2] = P2, P1
        return pi_new

    def _perturb_layout(self, base_pi, num_swaps, rng):
        if num_swaps <= 0 or rng is None or not self._adj:
            return np.asarray(base_pi, dtype=np.int64).copy()

        swaps = []
        N = len(base_pi)
        for _ in range(num_swaps):
            phys = int(rng.randint(N))
            retries = 0
            while not self._adj[phys] and retries < N:
                phys = (phys + 1) % N
                retries += 1
            if not self._adj[phys]:
                break
            nb = int(self._adj[phys][rng.randint(len(self._adj[phys]))])
            swaps.append((min(phys, nb), max(phys, nb)))

        if not swaps:
            return np.asarray(base_pi, dtype=np.int64).copy()

        return np.asarray(
            self._apply_swaps_to_pi(base_pi, swaps), dtype=np.int64
        )

    def _sample_initial_layout(self, trial_idx, n_trials, seeded_pi, rng):
        seeded_pi = np.asarray(seeded_pi, dtype=np.int64)
        if n_trials <= 1 or rng is None:
            return seeded_pi.copy()

        mirrored_pi = (len(seeded_pi) - 1) - seeded_pi

        if trial_idx == 0:
            return seeded_pi.copy()
        if trial_idx == 1:
            return mirrored_pi.copy()

        local_cutoff = max(3, int(np.ceil(n_trials * 0.6)))
        if trial_idx < local_cutoff:
            local_idx = trial_idx - 2
            band_idx = local_idx // 2
            local_budget = max(1, local_cutoff - 2)
            phase = band_idx / max(1, local_budget // 2)
            num_swaps = (
                1 + (band_idx % 3)
                if phase < 0.5
                else 4 + (band_idx % 5)
            )
            base = seeded_pi if local_idx % 2 == 0 else mirrored_pi
            return self._perturb_layout(base, num_swaps, rng)

        return rng.permutation(len(seeded_pi))

    def _bfs_shortest_path(self, src, dst):
        if src == dst:
            return [src]
        parent = {src: None}
        q = deque([src])
        while q:
            node = q.popleft()
            for nb in self._adj[node]:
                if nb in parent:
                    continue
                parent[nb] = node
                if nb == dst:
                    path = [dst]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(nb)
        return []

    @staticmethod
    def _entry_future_cost(entry, output_perm_arr, D_arr):
        eu = entry.get("edges_u")
        if eu is None:
            return 0.0
        phys_u = output_perm_arr[eu]
        phys_v = output_perm_arr[entry["edges_v"]]
        return float(np.maximum(0, D_arr[phys_u, phys_v] - 1).sum())

    @staticmethod
    def _estimate_candidate_output_layout(partition_candidate, pi, reverse=False):
        P_exit = partition_candidate.P_i if reverse else partition_candidate.P_o
        pi_output = [int(x) for x in pi]
        qbit_map_inverse = {
            v: k for k, v in partition_candidate.qbit_map.items()
        }
        for q_star in range(len(P_exit)):
            if q_star in qbit_map_inverse:
                k = qbit_map_inverse[q_star]
                pi_output[k] = partition_candidate.node_mapping[P_exit[q_star]]
        return pi_output

    @staticmethod
    def _future_context_cost(
        exclude_partition_idx,
        pi,
        F,
        E,
        D,
        candidate_cache,
        reverse=False,
        cnot_cost=1.0 / 3.0,
        W=0.5,
        alpha=1.0,
        layout_partitions=None,
        canonical_data=None,
    ):
        del cnot_cost, layout_partitions

        # Candidate-aware lower bound: for each future partition, use the best
        # available candidate entry cost under this layout.  This preserves the
        # monotone distance signal while allowing 3q line blocks to distinguish
        # which logical qubit should sit on the path center.
        pi_arr = np.asarray(pi, dtype=np.intp)
        D_arr = np.asarray(D)

        def partition_cost(p_idx):
            if candidate_cache is not None and 0 <= p_idx < len(candidate_cache):
                candidates = candidate_cache[p_idx]
                if candidates and len(candidates[0].involved_qbits) >= 3:
                    return min(
                        cand.estimate_swap_count(pi, D, reverse=reverse)
                        for cand in candidates
                    )
            if canonical_data is None:
                return None
            entry = canonical_data.get(p_idx)
            if entry is None:
                return None
            return qgd_Partition_Aware_Mapping._entry_future_cost(
                entry, pi_arr, D_arr
            )

        f_sum = 0.0
        n_other = 0
        for p_idx in F:
            if p_idx == exclude_partition_idx:
                continue
            cost = partition_cost(p_idx)
            if cost is None:
                continue
            f_sum += cost
            n_other += 1
        score = f_sum / n_other if n_other > 0 else 0.0

        if E:
            e_sum = 0.0
            e_count = 0
            for p_idx, depth in E:
                if p_idx == exclude_partition_idx:
                    continue
                cost = partition_cost(p_idx)
                if cost is None:
                    continue
                e_sum += (alpha ** depth) * cost
                e_count += 1
            if e_count:
                score += W * e_sum / e_count
        return score

    def _release_valve(self, F, pi, D, canonical_data):
        pi_arr = np.asarray(pi, dtype=np.intp)
        D_arr = np.asarray(D)
        best = None
        for p_idx in F:
            entry = canonical_data.get(p_idx)
            if entry is None:
                continue
            eu = entry.get("edges_u")
            if eu is None:
                continue
            ev = entry["edges_v"]
            phys_u = pi_arr[eu]
            phys_v = pi_arr[ev]
            dists = D_arr[phys_u, phys_v]
            if dists.size == 0:
                continue
            worst_idx = int(np.argmax(dists))
            worst_d = float(dists[worst_idx])
            if worst_d <= 1:
                continue
            if best is None or worst_d > best[0] or (
                worst_d == best[0] and p_idx < best[1]
            ):
                best = (worst_d, p_idx, int(eu[worst_idx]), int(ev[worst_idx]))

        if best is None:
            return [], list(pi)

        _, _, u, v = best
        path = self._bfs_shortest_path(int(pi[u]), int(pi[v]))
        if len(path) < 2:
            return [], list(pi)

        k = len(path) - 1
        m = k // 2
        swaps = []
        for i in range(m):
            swaps.append((path[i], path[i + 1]))
        for i in range(k, m + 1, -1):
            swaps.append((path[i], path[i - 1]))

        return swaps, self._apply_swaps_to_pi(pi, swaps)

    @staticmethod
    def _build_neighbor_info(
        partition_idx,
        F,
        E,
        pi,
        canonical_data,
        weight=0.2,
        W=0.5,
        alpha=0.9,
        layout_partitions=None,
    ):
        if weight <= 0 or layout_partitions is None:
            return None

        edge_weights = {}
        qubits = set()

        def add_edges(target_idx, edge_weight):
            if target_idx == partition_idx or edge_weight <= 0:
                return
            if target_idx >= len(layout_partitions):
                return
            entry = canonical_data.get(target_idx) if canonical_data else None
            if entry is not None and entry.get("edges_u") is not None:
                for u, v in zip(entry["edges_u"], entry["edges_v"]):
                    u = int(u)
                    v = int(v)
                    qubits.add(u)
                    qubits.add(v)
                    key = (u, v) if u <= v else (v, u)
                    edge_weights[key] = (
                        edge_weights.get(key, 0.0) + edge_weight
                    )
                return

            involved = qgd_Partition_Aware_Mapping._partition_involved_qbits(
                layout_partitions[target_idx]
            )
            for i, u in enumerate(involved):
                for v in involved[i + 1:]:
                    u = int(u)
                    v = int(v)
                    qubits.add(u)
                    qubits.add(v)
                    key = (u, v) if u <= v else (v, u)
                    edge_weights[key] = (
                        edge_weights.get(key, 0.0) + edge_weight
                    )

        for future_idx in F:
            add_edges(future_idx, 1.0)
        if E:
            for future_idx, depth in E:
                add_edges(future_idx, W * (alpha ** depth))

        if not edge_weights:
            return None

        neighbor_vqs = sorted(qubits)
        q_to_idx = {q: idx for idx, q in enumerate(neighbor_vqs)}
        edges = [
            (q_to_idx[u], q_to_idx[v], edge_weight)
            for (u, v), edge_weight in edge_weights.items()
        ]
        return {
            "neighbor_vqs": neighbor_vqs,
            "initial_pos": tuple(int(pi[q]) for q in neighbor_vqs),
            "edges": edges,
            "weight": weight,
        }

    def _advance_layout_frontier(
        self,
        selected_partition_idx,
        F,
        resolved_partitions,
        DAG,
        IDAG,
        optimized_partitions,
    ):
        """Advance a copied frontier without mutating circuits.

        This mirrors the layout-only single-qubit elision logic and is used by
        the boundary beam rollout.  It intentionally tracks only dependency
        state and layout; final circuit construction still happens through the
        concrete chosen route.
        """
        F_next = list(F)
        resolved_next = list(resolved_partitions)

        if selected_partition_idx in F_next:
            F_next.remove(selected_partition_idx)
        resolved_next[selected_partition_idx] = True

        stack = deque(DAG[selected_partition_idx])
        while stack:
            child = stack.popleft()
            if resolved_next[child] or child in F_next:
                continue
            if not all(resolved_next[parent] for parent in IDAG[child]):
                continue
            if self._partition_is_single(optimized_partitions[child]):
                resolved_next[child] = True
                stack.extend(DAG[child])
            else:
                F_next.append(child)

        return tuple(F_next), tuple(resolved_next)

    def _boundary_beam_select_index(
        self,
        partition_candidates,
        scores,
        cached_swaps,
        cached_pi,
        F_snapshot,
        resolved_partitions,
        DAG,
        IDAG,
        optimized_partitions,
        scoring_partitions,
        D,
        candidate_cache,
        canonical_data,
        reverse=False,
        W=0.5,
        alpha=1.0,
        cnot_cost=1.0 / 3.0,
        adj=None,
    ):
        """Choose the next candidate by rolling out boundary-layout states.

        The ordinary SABRE selector commits to the locally best candidate. This
        keeps a small beam of possible boundary layouts across several future
        partitions, then returns the first candidate from the best rollout.
        """
        beam_width = int(self.config.get("boundary_beam_width", 1) or 1)
        beam_depth = int(self.config.get("boundary_beam_depth", 1) or 1)
        fallback_idx = int(np.argmin(np.asarray(scores)))
        if beam_width <= 1 or beam_depth <= 1 or len(partition_candidates) <= 1:
            return fallback_idx
        if not any(len(cand.involved_qbits) >= 3 for cand in partition_candidates):
            return fallback_idx

        max_E_size = self.config.get("max_E_size", 20)
        max_lookahead = self.config.get("max_lookahead", 4)
        top_k = self.config.get("prefilter_top_k", 50)
        path_weight = self.config.get("path_tiebreak_weight", 0.2)
        three_q_weight = self.config.get("three_qubit_exit_weight", 1.0)

        def transition_cost(cand, swaps):
            return self._routing_objective(
                len(swaps or ()),
                cand.cnot_count,
                cnot_cost,
            )

        states = []
        for idx, cand in enumerate(partition_candidates):
            if cached_pi[idx] is None:
                continue
            trans_cost = transition_cost(cand, cached_swaps[idx])
            F_next, resolved_next = self._advance_layout_frontier(
                cand.partition_idx,
                F_snapshot,
                resolved_partitions,
                DAG,
                IDAG,
                optimized_partitions,
            )
            states.append(
                (
                    float(scores[idx]),
                    float(trans_cost),
                    tuple(int(x) for x in cached_pi[idx]),
                    F_next,
                    resolved_next,
                    idx,
                )
            )

        if not states:
            return fallback_idx

        states.sort(key=lambda item: (item[0], item[5]))
        states = states[:beam_width]

        for _ in range(1, beam_depth):
            expanded = []
            for _, total_cost, pi_state, F_state, resolved_state, first_idx in states:
                if not F_state:
                    expanded.append(
                        (total_cost, total_cost, pi_state, F_state, resolved_state, first_idx)
                    )
                    continue

                resolved_list = list(resolved_state)
                F_list = list(F_state)
                E = self.generate_extended_set(
                    F_list,
                    DAG,
                    IDAG,
                    resolved_list,
                    optimized_partitions,
                    max_E_size=max_E_size,
                    max_lookahead=max_lookahead,
                )
                candidates = self.obtain_partition_candidates(
                    F_list,
                    optimized_partitions,
                    candidate_cache=candidate_cache,
                )
                if not candidates:
                    expanded.append(
                        (total_cost, total_cost, pi_state, F_state, resolved_state, first_idx)
                    )
                    continue
                candidates = self._prefilter_candidates(
                    candidates,
                    list(pi_state),
                    D,
                    top_k,
                    F=F_state,
                    E=E,
                    candidate_cache=candidate_cache,
                    layout_partitions=optimized_partitions,
                    reverse=reverse,
                    W=W,
                    alpha=alpha,
                    canonical_data=canonical_data,
                )

                for cand in candidates:
                    neighbor_info = self._build_neighbor_info(
                        cand.partition_idx,
                        F_state,
                        E,
                        pi_state,
                        canonical_data,
                        weight=path_weight,
                        W=W,
                        alpha=alpha,
                        layout_partitions=optimized_partitions,
                    )
                    score, swaps, output_perm = self.score_partition_candidate(
                        cand,
                        F_state,
                        list(pi_state),
                        scoring_partitions,
                        D,
                        self._swap_cache,
                        E=E,
                        W=W,
                        alpha=alpha,
                        reverse=reverse,
                        canonical_data=canonical_data,
                        adj=adj,
                        cnot_cost=cnot_cost,
                        path_tiebreak_weight=path_weight,
                        cached_neighbor_info=neighbor_info,
                        candidate_cache=candidate_cache,
                        layout_partitions=optimized_partitions,
                        return_transforms=True,
                        three_qubit_exit_weight=three_q_weight,
                    )
                    trans_cost = transition_cost(cand, swaps)
                    future_cost = float(score) - trans_cost
                    new_total = total_cost + trans_cost
                    rank_cost = new_total + future_cost
                    F_next, resolved_next = self._advance_layout_frontier(
                        cand.partition_idx,
                        F_state,
                        resolved_state,
                        DAG,
                        IDAG,
                        optimized_partitions,
                    )
                    expanded.append(
                        (
                            rank_cost,
                            new_total,
                            tuple(int(x) for x in output_perm),
                            F_next,
                            resolved_next,
                            first_idx,
                        )
                    )

            if not expanded:
                break
            expanded.sort(key=lambda item: (item[0], item[5]))
            states = expanded[:beam_width]

        if not states:
            return fallback_idx
        return int(min(states, key=lambda item: (item[0], item[5]))[5])

    def Heuristic_Search(
        self,
        F,
        pi,
        DAG,
        IDAG,
        optimized_partitions,
        scoring_partitions,
        D,
        candidate_cache=None,
    ):
        pi_initial = pi.copy()
        F = list(F)

        resolved_partitions = [False] * len(DAG)
        partition_order = []
        resolved_count = 0

        queue = deque(
            p
            for p in F
            if isinstance(optimized_partitions[p], SingleQubitPartitionResult)
        )
        while queue:
            partition_idx = queue.pop()
            if resolved_partitions[partition_idx]:
                continue
            if partition_idx in F:
                F.remove(partition_idx)

            single_qubit_part = optimized_partitions[partition_idx]
            original_qubit = int(single_qubit_part.involved_qbits[0])
            circuit_qubit = int(single_qubit_part.circuit.get_Qbits()[0])
            single_qubit_part.circuit = single_qubit_part.circuit.Remap_Qbits(
                {circuit_qubit: int(pi[original_qubit])},
                max(D.shape),
            )
            partition_order.append(single_qubit_part)
            resolved_partitions[partition_idx] = True
            resolved_count += 1

            for child in DAG[partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if isinstance(
                            optimized_partitions[child],
                            SingleQubitPartitionResult,
                        ):
                            queue.append(child)
                        else:
                            F.append(child)

        total_partitions = len(DAG)
        pbar = tqdm(
            total=total_partitions,
            desc="Heuristic Search",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} resolved"
            ),
            disable=self.config.get("progressbar", 0) is False,
            mininterval=0.2,
        )
        if resolved_count:
            pbar.update(resolved_count)

        max_E_size = self.config.get("max_E_size", 20)
        max_lookahead = self.config.get("max_lookahead", 4)
        E_W = self.config.get("E_weight", 0.5)
        E_alpha = self.config.get("E_alpha", 1.0)
        swap_burst_budget = self.config.get("swap_burst_budget", 5)

        canonical_data = self._build_canonical_neighbor_data(
            scoring_partitions, reverse=False
        )
        decay = [1.0] * len(pi)
        swap_heavy_partitions = 0

        while F:
            if (
                swap_burst_budget > 0
                and swap_heavy_partitions >= swap_burst_budget
            ):
                valve_swaps, pi_bridged = self._release_valve(
                    F, pi, D, canonical_data
                )
                if valve_swaps:
                    partition_order.append(
                        construct_swap_circuit(valve_swaps, len(pi))
                    )
                    self._apply_decay_for_swaps(valve_swaps, decay)
                    pi = np.asarray(pi_bridged)
                    swap_heavy_partitions = 0
                    continue
                self._reset_decay(decay)
                swap_heavy_partitions = 0

            F_snapshot = tuple(F)
            E = self.generate_extended_set(
                F,
                DAG,
                IDAG,
                resolved_partitions,
                optimized_partitions,
                max_E_size=max_E_size,
                max_lookahead=max_lookahead,
            )

            partition_candidates = self.obtain_partition_candidates(
                F,
                optimized_partitions,
                candidate_cache=candidate_cache,
            )
            if not partition_candidates:
                break

            top_k = self.config.get("prefilter_top_k", 50)
            partition_candidates = self._prefilter_candidates(
                partition_candidates,
                pi,
                D,
                top_k,
                F=F_snapshot,
                E=E,
                candidate_cache=candidate_cache,
                layout_partitions=optimized_partitions,
                W=E_W,
                alpha=E_alpha,
                canonical_data=canonical_data,
            )

            # Group candidates by partition_idx to reuse _build_neighbor_info
            candidate_order = sorted(
                range(len(partition_candidates)),
                key=lambda i: partition_candidates[i].partition_idx
            )
            scores = [0.0] * len(partition_candidates)
            cached_swaps = [None] * len(partition_candidates)
            cached_pi = [None] * len(partition_candidates)
            prev_partition_idx = None
            cached_neighbor_info = None
            for ci in candidate_order:
                cand = partition_candidates[ci]
                if cand.partition_idx != prev_partition_idx:
                    cached_neighbor_info = self._build_neighbor_info(
                        cand.partition_idx,
                        F_snapshot,
                        E,
                        pi,
                        canonical_data,
                        weight=self.config.get("path_tiebreak_weight", 0.2),
                        W=E_W,
                        alpha=E_alpha,
                        layout_partitions=optimized_partitions,
                    )
                    prev_partition_idx = cand.partition_idx
                score, swaps, output_perm = self.score_partition_candidate(
                    cand,
                    F_snapshot,
                    pi,
                    scoring_partitions,
                    D,
                    self._swap_cache,
                    E=E,
                    W=E_W,
                    alpha=E_alpha,
                    canonical_data=canonical_data,
                    adj=self._adj,
                    cnot_cost=self.config.get("cnot_cost", 1.0 / 3.0),
                    path_tiebreak_weight=self.config.get(
                        "path_tiebreak_weight", 0.2
                    ),
                    decay=decay,
                    cached_neighbor_info=cached_neighbor_info,
                    candidate_cache=candidate_cache,
                    layout_partitions=optimized_partitions,
                    return_transforms=True,
                    three_qubit_exit_weight=self.config.get(
                        "three_qubit_exit_weight", 1.0
                    ),
                )
                scores[ci] = score
                cached_swaps[ci] = swaps
                cached_pi[ci] = output_perm

            best_idx = self._boundary_beam_select_index(
                partition_candidates,
                scores,
                cached_swaps,
                cached_pi,
                F_snapshot,
                resolved_partitions,
                DAG,
                IDAG,
                optimized_partitions,
                scoring_partitions,
                D,
                candidate_cache,
                canonical_data,
                W=E_W,
                alpha=E_alpha,
                cnot_cost=self.config.get("cnot_cost", 1.0 / 3.0),
                adj=self._adj,
            )
            min_partition_candidate = partition_candidates[best_idx]

            F.remove(min_partition_candidate.partition_idx)
            resolved_partitions[min_partition_candidate.partition_idx] = True
            resolved_count += 1
            pbar.update(1)

            swap_order, pi = cached_swaps[best_idx], cached_pi[best_idx]
            if swap_order:
                partition_order.append(construct_swap_circuit(swap_order, len(pi)))
                self._apply_decay_for_swaps(swap_order, decay)
                swap_heavy_partitions += 1
            else:
                swap_heavy_partitions = 0
                self._reset_decay(decay)

            partition_order.append(min_partition_candidate)

            children = deque(DAG[min_partition_candidate.partition_idx])
            while children:
                child = children.popleft()
                parents_resolved = all(
                    resolved_partitions[parent] for parent in IDAG[child]
                )
                if (not resolved_partitions[child] and child not in F) and (
                    parents_resolved
                ):
                    if isinstance(
                        optimized_partitions[child], SingleQubitPartitionResult
                    ):
                        child_partition = optimized_partitions[child]
                        original_qubit = int(child_partition.involved_qbits[0])
                        circuit_qubit = int(child_partition.circuit.get_Qbits()[0])
                        child_partition.circuit = child_partition.circuit.Remap_Qbits(
                            {circuit_qubit: int(pi[original_qubit])},
                            max(D.shape),
                        )
                        partition_order.append(child_partition)
                        resolved_partitions[child] = True
                        resolved_count += 1
                        pbar.update(1)
                        children.extend(DAG[child])
                    else:
                        F.append(child)

        pbar.close()
        return partition_order, pi, pi_initial

    def _heuristic_search_layout_only(
        self,
        F,
        pi,
        DAG,
        IDAG,
        optimized_partitions,
        scoring_partitions,
        D,
        rng=None,
        reverse=False,
        candidate_cache=None,
    ):
        """Run heuristic search but only track layout (pi). No circuit modification.

        Args:
            reverse: When True, swap P_i/P_o roles in scoring and layout
                    updates (used for backward passes in SABRE iterations).

        Returns:
            (pi, total_cost): final layout and layout-only heuristic score.
            Trial ranking reroutes returned layouts and sorts by actual
            constructed-circuit CNOT count; this score is only a tie-breaker.
        """
        F = list(F)
        resolved_partitions = [False] * len(DAG)
        total_cost = 0.0

        queue = deque(
            p for p in F if self._partition_is_single(optimized_partitions[p])
        )
        while queue:
            partition_idx = queue.pop()
            if resolved_partitions[partition_idx]:
                continue
            if partition_idx in F:
                F.remove(partition_idx)
            resolved_partitions[partition_idx] = True

            for child in DAG[partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if self._partition_is_single(optimized_partitions[child]):
                            queue.append(child)
                        else:
                            F.append(child)

        max_E_size = self.config.get("max_E_size", 20)
        max_lookahead = self.config.get("max_lookahead", 4)
        E_W = self.config.get("E_weight", 0.5)
        E_alpha = self.config.get("E_alpha", 1.0)
        cnot_cost = self.config.get("cnot_cost", 1.0 / 3.0)
        swap_burst_budget = self.config.get("swap_burst_budget", 5)

        canonical_data = self._build_canonical_neighbor_data(
            scoring_partitions, reverse=reverse
        )
        decay = [1.0] * len(pi)
        swap_heavy_partitions = 0

        while F:
            if (
                swap_burst_budget > 0
                and swap_heavy_partitions >= swap_burst_budget
            ):
                valve_swaps, pi = self._release_valve(F, pi, D, canonical_data)
                if valve_swaps:
                    total_cost += self._routing_objective(
                        len(valve_swaps),
                        0,
                        cnot_cost,
                        decay_factor=self._decay_factor_for_swaps(
                            valve_swaps, decay
                        ),
                    )
                    self._apply_decay_for_swaps(valve_swaps, decay)
                    swap_heavy_partitions = 0
                    continue
                self._reset_decay(decay)
                swap_heavy_partitions = 0

            F_snapshot = tuple(F)
            E = self.generate_extended_set(
                F,
                DAG,
                IDAG,
                resolved_partitions,
                optimized_partitions,
                max_E_size=max_E_size,
                max_lookahead=max_lookahead,
            )

            partition_candidates = self.obtain_partition_candidates(
                F,
                optimized_partitions,
                candidate_cache=candidate_cache,
            )
            if not partition_candidates:
                break

            top_k = self.config.get("prefilter_top_k", 50)
            partition_candidates = self._prefilter_candidates(
                partition_candidates,
                pi,
                D,
                top_k,
                F=F_snapshot,
                E=E,
                candidate_cache=candidate_cache,
                layout_partitions=optimized_partitions,
                reverse=reverse,
                W=E_W,
                alpha=E_alpha,
                canonical_data=canonical_data,
            )

            # Group candidates by partition_idx to reuse _build_neighbor_info
            candidate_order = sorted(
                range(len(partition_candidates)),
                key=lambda i: partition_candidates[i].partition_idx
            )
            scores = [0.0] * len(partition_candidates)
            cached_swaps = [None] * len(partition_candidates)
            cached_pi = [None] * len(partition_candidates)
            prev_partition_idx = None
            cached_neighbor_info = None
            for ci in candidate_order:
                cand = partition_candidates[ci]
                if cand.partition_idx != prev_partition_idx:
                    cached_neighbor_info = self._build_neighbor_info(
                        cand.partition_idx,
                        F_snapshot,
                        E,
                        pi,
                        canonical_data,
                        weight=self.config.get("path_tiebreak_weight", 0.2),
                        W=E_W,
                        alpha=E_alpha,
                        layout_partitions=optimized_partitions,
                    )
                    prev_partition_idx = cand.partition_idx
                score, swaps, output_perm = self.score_partition_candidate(
                    cand,
                    F_snapshot,
                    pi,
                    scoring_partitions,
                    D,
                    self._swap_cache,
                    E=E,
                    W=E_W,
                    alpha=E_alpha,
                    reverse=reverse,
                    canonical_data=canonical_data,
                    adj=self._adj,
                    cnot_cost=cnot_cost,
                    path_tiebreak_weight=self.config.get(
                        "path_tiebreak_weight", 0.2
                    ),
                    decay=decay,
                    cached_neighbor_info=cached_neighbor_info,
                    candidate_cache=candidate_cache,
                    layout_partitions=optimized_partitions,
                    return_transforms=True,
                    three_qubit_exit_weight=self.config.get(
                        "three_qubit_exit_weight", 1.0
                    ),
                )
                scores[ci] = score
                cached_swaps[ci] = swaps
                cached_pi[ci] = output_perm

            best_idx = self._boundary_beam_select_index(
                partition_candidates,
                scores,
                cached_swaps,
                cached_pi,
                F_snapshot,
                resolved_partitions,
                DAG,
                IDAG,
                optimized_partitions,
                scoring_partitions,
                D,
                candidate_cache,
                canonical_data,
                reverse=reverse,
                W=E_W,
                alpha=E_alpha,
                cnot_cost=cnot_cost,
                adj=self._adj,
            )
            best = partition_candidates[best_idx]
            F.remove(best.partition_idx)
            resolved_partitions[best.partition_idx] = True

            swaps, pi = cached_swaps[best_idx], cached_pi[best_idx]
            decay_factor = 1.0
            if swaps:
                decay_factor = self._decay_factor_for_swaps(swaps, decay)
            total_cost += self._routing_objective(
                len(swaps),
                best.cnot_count,
                cnot_cost,
                decay_factor=decay_factor,
            )
            if swaps:
                self._apply_decay_for_swaps(swaps, decay)
                swap_heavy_partitions += 1
            else:
                swap_heavy_partitions = 0
                self._reset_decay(decay)

            for child in DAG[best.partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if self._partition_is_single(optimized_partitions[child]):
                            resolved_partitions[child] = True
                            stack = deque(DAG[child])
                            while stack:
                                gc = stack.pop()
                                if not resolved_partitions[gc] and gc not in F:
                                    if all(
                                        resolved_partitions[p]
                                        for p in IDAG[gc]
                                    ):
                                        if self._partition_is_single(
                                            optimized_partitions[gc]
                                        ):
                                            resolved_partitions[gc] = True
                                            stack.extend(DAG[gc])
                                        else:
                                            F.append(gc)
                        else:
                            F.append(child)

        return pi, total_cost    
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

    def _build_canonical_neighbor_data(self, scoring_partitions, reverse=False):
        """Build a compact future-routing surrogate per partition.

        For each partition, pick the edge pattern with the lowest CNOT count;
        the router uses this as a canonical "best still-available option" when
        scoring future partitions.
        """
        data = {}
        for idx, partition in enumerate(scoring_partitions):
            if partition is None:
                continue
            qbit_map_inv = {v: q for q, v in partition.qubit_map.items()}
            variant_map = {}
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                for pdx, (P_i, P_o) in enumerate(partition.permutations_pairs[tdx]):
                    cnot = partition.cnot_counts[tdx][pdx]
                    P_route = P_o if reverse else P_i
                    if mini_topology:
                        edge_key = tuple(
                            sorted(
                                tuple(
                                    sorted(
                                        (
                                            qbit_map_inv[P_route[u]],
                                            qbit_map_inv[P_route[v]],
                                        )
                                    )
                                )
                                for u, v in mini_topology
                            )
                        )
                    else:
                        edge_key = tuple()
                    prev_cnot = variant_map.get(edge_key)
                    if prev_cnot is None or cnot < prev_cnot:
                        variant_map[edge_key] = cnot
            if not variant_map:
                continue
            edge_key, cnot = min(
                variant_map.items(),
                key=lambda item: (item[1], len(item[0]), item[0]),
            )
            if edge_key:
                eu = np.array([e[0] for e in edge_key], dtype=np.intp)
                ev = np.array([e[1] for e in edge_key], dtype=np.intp)
            else:
                eu = ev = None
            data[idx] = {"edges_u": eu, "edges_v": ev, "cnot": cnot}
        return data

    @staticmethod
    def score_partition_candidate(partition_candidate, F, pi, scoring_partitions, D, swap_cache,
                                  E=None, W=0.5, alpha=0.9, reverse=False,
                                  canonical_data=None, adj=None,
                                  cnot_cost=1.0 / 3.0,
                                  path_tiebreak_weight=0.2, decay=None,
                                  cached_neighbor_info=None,
                                  candidate_cache=None,
                                  layout_partitions=None,
                                  return_transforms=False,
                                  three_qubit_exit_weight=1.0):
        """LightSABRE-style relative scoring (arXiv:2409.08368, eq. 1).

        H = |swaps|
          + cnot_cost * cand.cnot_count
          + (1/|F'|) * average routing cost over F \\ {cand}
          + (W/|E|)  * alpha^d-decayed routing cost over E
        """
        if cached_neighbor_info is not None:
            neighbor_info = cached_neighbor_info
        else:
            neighbor_info = qgd_Partition_Aware_Mapping._build_neighbor_info(
                partition_candidate.partition_idx,
                F,
                E,
                pi,
                canonical_data,
                weight=path_tiebreak_weight,
                W=W,
                alpha=alpha,
                layout_partitions=layout_partitions,
            )
        swaps, output_perm = partition_candidate.transform_pi(
            pi,
            D,
            swap_cache,
            reverse=reverse,
            adj=adj,
            neighbor_info=neighbor_info,
        )
        decay_factor = 1.0
        if decay is not None and swaps:
            decay_factor = qgd_Partition_Aware_Mapping._decay_factor_for_swaps(
                swaps, decay
            )
        score = qgd_Partition_Aware_Mapping._routing_objective(
            len(swaps),
            partition_candidate.cnot_count,
            cnot_cost,
            decay_factor=decay_factor,
        )

        if candidate_cache is None:
            if return_transforms:
                return score, swaps, output_perm
            return score

        cand_idx = partition_candidate.partition_idx
        future_score = qgd_Partition_Aware_Mapping._future_context_cost(
            cand_idx,
            output_perm,
            F,
            E,
            D,
            candidate_cache,
            reverse=reverse,
            cnot_cost=cnot_cost,
            W=W,
            alpha=alpha,
            layout_partitions=layout_partitions,
            canonical_data=canonical_data,
        )
        if len(partition_candidate.involved_qbits) >= 3:
            future_score *= three_qubit_exit_weight
        score += future_score

        if return_transforms:
            return score, swaps, output_perm
        return score

    # ------------------------------------------------------------------------
    # Extended Set
    # ------------------------------------------------------------------------

    @staticmethod
    def generate_extended_set(
        F,
        DAG,
        IDAG,
        resolved_partitions,
        optimized_partitions,
        max_E_size=20,
        max_lookahead=4,
    ):
        """
        Generate SABRE-style extended set: multi-qubit partitions near the
        front layer, up to ``max_lookahead`` levels deep and ``max_E_size``
        entries. Returns list of (partition_idx, depth) tuples.
        """
        E = []
        E_set = set()
        F_set = set(F)

        for front_idx in F:
            if len(E) >= max_E_size:
                break

            queue = deque((child, 1) for child in DAG[front_idx])

            while queue and len(E) < max_E_size:
                child_idx, depth = queue.popleft()
                if depth > max_lookahead:
                    continue
                if child_idx in E_set or child_idx in F_set:
                    continue
                if resolved_partitions[child_idx]:
                    continue

                parents_resolved = all(
                    resolved_partitions[p] or p in F_set for p in IDAG[child_idx]
                )
                if not parents_resolved:
                    continue

                if qgd_Partition_Aware_Mapping._partition_is_single(
                    optimized_partitions[child_idx]
                ):
                    for grandchild in DAG[child_idx]:
                        queue.append((grandchild, depth))
                    continue

                E.append((child_idx, depth))
                E_set.add(child_idx)

                if depth < max_lookahead:
                    for grandchild in DAG[child_idx]:
                        queue.append((grandchild, depth + 1))

        return E

    # ------------------------------------------------------------------------
    # Candidate Generation
    # ------------------------------------------------------------------------

    def obtain_partition_candidates(
        self,
        F,
        optimized_partitions=None,
        candidate_cache=None,
    ):
        if candidate_cache is not None:
            partition_candidates = []
            for partition_idx in F:
                cached = candidate_cache[partition_idx]
                if cached:
                    partition_candidates.extend(cached)
            return partition_candidates

        partition_candidates = []
        for partition_idx in F:
            partition = optimized_partitions[partition_idx]
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                if hasattr(partition, 'get_topology_candidates'):
                    topology_candidates = partition.get_topology_candidates(tdx)
                else:
                    topology_candidates = self._get_subtopologies_of_type_cached(
                        mini_topology
                    )
                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(
                        partition.permutations_pairs[tdx]
                    ):
                        partition_candidates.append(
                            PartitionCandidate(
                                partition_idx,
                                tdx,
                                pdx,
                                partition.circuit_structures[tdx][pdx],
                                permutation_pair[0],
                                permutation_pair[1],
                                topology_candidate,
                                mini_topology,
                                partition.qubit_map,
                                partition.involved_qbits,
                                cnot_count=partition.cnot_counts[tdx][pdx],
                            )
                        )
        return partition_candidates

    # ------------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------------
        
    def get_initial_layer(self, IDAG, N, optimized_partitions):
        del N, optimized_partitions
        return [idx for idx in range(len(IDAG)) if not IDAG[idx]]


    def get_final_layer(self, DAG, N, optimized_partitions):
        del N, optimized_partitions
        return [idx for idx in range(len(DAG) - 1, -1, -1) if not DAG[idx]]
                
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
        
        # Store adjacency list for reuse by A* routing
        self._adj = [list(adj[i]) for i in range(N)]

        return D

    def _compute_seeded_layout(self, optimized_partitions, D, N, circ):
        """VF2Layout + SabrePreLayout seeded initial layout (LightSABRE §II.3).

        The interaction graph is built from the circuit's two-qubit gate pairs
        (matching the paper's gate-level approach), not from partition cliques.
        Partition-level weights are used only for the greedy fallback.

        Steps:
        1. VF2Layout: subgraph isomorphism of gate interaction graph into
           hardware topology.  If a mapping exists, every gate qubit pair
           lands on adjacent physical qubits (zero SWAPs).
        2. SabrePreLayout: augment topology with distance-d edges (d=2),
           retry VF2 — handles "almost perfect" embeddings.
        3. Fallback: greedy weighted-distance placement from partition weights.
        """
        from collections import defaultdict
        from squander.synthesis.PartAM_utils import PartitionSynthesisResult, SingleQubitPartitionResult

        if not self.topology:
            return np.arange(N)

        # --- build gate-level interaction graph from circuit CNOT pairs ---
        gate_edges = set()
        for g in circ.get_Gates():
            gname = str(type(g).__name__)
            if 'CNOT' in gname or 'CX' in gname:
                ctrl = g.get_Control_Qbit()
                tgt = g.get_Target_Qbit()
                gate_edges.add((min(ctrl, tgt), max(ctrl, tgt)))

        if not gate_edges:
            return np.arange(N)

        # --- try rustworkx VF2 approaches ---
        try:
            import rustworkx as rx
        except ImportError:
            return self._greedy_seeded_layout(optimized_partitions, D, N)

        G_int = rx.PyGraph()
        G_int.add_nodes_from(range(N))
        for u, v in gate_edges:
            G_int.add_edge(u, v, None)

        G_hw = rx.PyGraph()
        G_hw.add_nodes_from(range(N))
        for u, v in self.topology:
            G_hw.add_edge(u, v, None)

        # Step 1: VF2Layout — exact subgraph isomorphism
        pi = self._try_vf2_layout(G_int, G_hw, N)
        if pi is not None:
            return pi

        # Step 2: SabrePreLayout — augment topology with distance-2 edges
        G_aug = rx.PyGraph()
        G_aug.add_nodes_from(range(N))
        seen = set()
        for u, v in self.topology:
            G_aug.add_edge(u, v, None)
            seen.add((min(u, v), max(u, v)))
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in seen and D[i][j] <= 2:
                    G_aug.add_edge(i, j, None)
                    seen.add((i, j))

        pi = self._try_vf2_layout(G_int, G_aug, N)
        if pi is not None:
            return pi

        # Step 3: greedy fallback using partition-level weights
        return self._greedy_seeded_layout(optimized_partitions, D, N)

    def _try_vf2_layout(self, G_int, G_hw, N):
        """Try VF2 subgraph isomorphism of G_int into G_hw.

        Returns pi (logical->physical mapping) or None if no embedding exists.
        Uses induced=False to allow non-edges in the interaction graph to
        correspond to edges in the hardware graph (monotone subgraph iso).
        """
        import rustworkx as rx

        try:
            vf2_iter = rx.vf2_mapping(G_hw, G_int, subgraph=True, induced=False)
            mapping = next(vf2_iter)  # {hw_node: int_node}
        except StopIteration:
            return None

        # Invert: pi[logical_q] = physical_q
        pi = np.zeros(N, dtype=int)
        inv = {v: k for k, v in mapping.items()}
        used = set(inv.values())
        free = [p for p in range(N) if p not in used]
        fi = 0
        for q in range(N):
            if q in inv:
                pi[q] = inv[q]
            else:
                pi[q] = free[fi]
                fi += 1
        return pi

    def _greedy_seeded_layout(self, optimized_partitions, D, N):
        """Greedy weighted-distance placement (fallback when VF2 fails)."""
        from collections import defaultdict
        from squander.synthesis.PartAM_utils import PartitionSynthesisResult, SingleQubitPartitionResult

        # Build interaction weights from partitions
        interaction_weight = defaultdict(float)
        for partition in optimized_partitions:
            if isinstance(partition, SingleQubitPartitionResult):
                continue
            if not isinstance(partition, PartitionSynthesisResult):
                continue
            involved = list(partition.involved_qbits)
            if len(involved) < 2:
                continue
            best_cnot = float('inf')
            for tdx in range(len(partition.cnot_counts)):
                if not partition.cnot_counts[tdx]:
                    continue
                cnot_min = min(partition.cnot_counts[tdx])
                if cnot_min < best_cnot:
                    best_cnot = cnot_min
            if best_cnot == float('inf'):
                continue
            for i in range(len(involved)):
                for j in range(i + 1, len(involved)):
                    key = (min(involved[i], involved[j]),
                           max(involved[i], involved[j]))
                    interaction_weight[key] += best_cnot

        if not interaction_weight:
            return np.arange(N)

        pi = np.arange(N)
        placed_logical = set()
        placed_physical = set()

        (q1, q2), _ = max(interaction_weight.items(), key=lambda x: x[1])
        p1, p2 = self.topology[0]

        holder1 = np.where(pi == p1)[0][0]
        pi[q1], pi[holder1] = p1, pi[q1]
        holder2 = np.where(pi == p2)[0][0]
        pi[q2], pi[holder2] = p2, pi[q2]
        placed_logical.update([q1, q2])
        placed_physical.update([p1, p2])

        remaining = [q for q in range(N) if q not in placed_logical]

        def _score(q):
            return sum(
                interaction_weight.get((min(q, pq), max(q, pq)), 0.0)
                for pq in placed_logical
            )

        remaining.sort(key=_score, reverse=True)

        for logical_q in remaining:
            best_physical = None
            best_dist = float('inf')

            for physical_q in range(N):
                if physical_q in placed_physical:
                    continue

                total_dist = 0.0
                total_w = 0.0
                for other_q in placed_logical:
                    key = (min(logical_q, other_q), max(logical_q, other_q))
                    w = interaction_weight.get(key, 0.0)
                    if w > 0:
                        total_dist += D[physical_q][pi[other_q]] * w
                        total_w += w

                avg = total_dist / total_w if total_w > 0 else 0.0
                if avg < best_dist:
                    best_dist = avg
                    best_physical = physical_q

            if best_physical is not None:
                holder = np.where(pi == best_physical)[0][0]
                pi[logical_q], pi[holder] = best_physical, pi[logical_q]
                placed_logical.add(logical_q)
                placed_physical.add(best_physical)

        return pi


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
