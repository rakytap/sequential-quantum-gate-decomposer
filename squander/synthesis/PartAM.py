"""
This is an implementation of Partition Aware Mapping.
"""
from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from itertools import permutations
from squander.partitioning.ilp import (
    get_all_partitions,
    _get_topo_order,
    topo_sort_partitions,
    ilp_global_optimal,
)

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
import os
import logging
from tqdm import tqdm
from collections import deque, defaultdict

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
    check_circuit_compatibility,
    construct_swap_circuit,
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
        self.config.setdefault('use_osr', 0)
        self.config.setdefault('n_layout_trials', 1)
        self.config.setdefault('score_tolerance', 0.05)
        self.config.setdefault('random_seed', 42)
        self.config.setdefault('cleanup', True)
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

    # ------------------------------------------------------------------------
    # Partition Decomposition Methods
    # ------------------------------------------------------------------------

    @staticmethod
    def DecomposePartition_Sequential(Partition_circuit: Circuit, Partition_parameters: np.ndarray, config: dict, topologies, involved_qbits, qbit_map) -> PartitionSynthesisResult:
        """
        Call to decompose a partition sequentially.
        When use_automorphisms is enabled in config, derives equivalent
        decompositions from topology automorphisms to skip redundant work.
        """
        N = Partition_circuit.get_Qbit_Num()
        if N !=1:
            perumations_all = list(permutations(range(N)))
            result = PartitionSynthesisResult(N, topologies, involved_qbits, qbit_map, Partition_circuit)
            use_auts = config.get('use_automorphisms', True)

            for topology_idx in range(len(topologies)):
                mini_topology = topologies[topology_idx]
                if use_auts:
                    auts = compute_automorphisms(mini_topology)
                    identity = tuple(range(N))
                    known_pairs = set()

                # Stage 1: fix P_o, sweep all P_i
                P_o_initial = perumations_all[np.random.choice(range(len(perumations_all)))]
                for P_i in perumations_all:
                    Partition_circuit_tmp = Circuit(N)
                    Partition_circuit_tmp.add_Permutation(list(P_i))
                    Partition_circuit_tmp.add_Circuit(Partition_circuit)
                    Partition_circuit_tmp.add_Permutation(list(P_o_initial))
                    synthesised_circuit, synthesised_parameters = qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                    result.add_result((P_i, P_o_initial), synthesised_circuit, synthesised_parameters, topology_idx)
                    if use_auts:
                        known_pairs.add((P_i, P_o_initial))
                        for sigma in auts:
                            if sigma == identity:
                                continue
                            new_P_i, new_P_o, new_circuit, new_params = derive_result_from_automorphism(sigma, P_i, P_o_initial, synthesised_circuit, synthesised_parameters, N)
                            if (new_P_i, new_P_o) not in known_pairs:
                                result.add_result((new_P_i, new_P_o), new_circuit, new_params, topology_idx)
                                known_pairs.add((new_P_i, new_P_o))

                # Stage 2: fix P_i_best, sweep all P_o
                P_i_best, _ = result.get_best_result(topology_idx)[0]
                for P_o in perumations_all:
                    if use_auts and (tuple(P_i_best), P_o) in known_pairs:
                        continue
                    Partition_circuit_tmp = Circuit(N)
                    Partition_circuit_tmp.add_Permutation(list(P_i_best))
                    Partition_circuit_tmp.add_Circuit(Partition_circuit)
                    Partition_circuit_tmp.add_Permutation(list(P_o))
                    synthesised_circuit, synthesised_parameters = qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                    result.add_result((P_i_best, P_o), synthesised_circuit, synthesised_parameters, topology_idx)
                    if use_auts:
                        known_pairs.add((tuple(P_i_best), P_o))
                        for sigma in auts:
                            if sigma == identity:
                                continue
                            new_P_i, new_P_o, new_circuit, new_params = derive_result_from_automorphism(sigma, P_i_best, P_o, synthesised_circuit, synthesised_parameters, N)
                            if (new_P_i, new_P_o) not in known_pairs:
                                result.add_result((new_P_i, new_P_o), new_circuit, new_params, topology_idx)
                                known_pairs.add((new_P_i, new_P_o))
        else:
            result = SingleQubitPartitionResult(Partition_circuit,Partition_parameters)
        return result

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

    def SynthesizeWideCircuit(self, circ, orig_parameters):
        """
        Partition and synthesize a full circuit.

        Args:
            circ: The full quantum circuit (must be flat — no subcircuit blocks)
            orig_parameters: Parameters for circ

        Returns:
            optimized_partitions: List of PartitionSynthesisResult / SingleQubitPartitionResult
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

        # Config with parallel=1 for large partitions (use internal C++ parallelism)
        large_partition_config = dict(self.config)
        large_partition_config['parallel'] = 1

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

                if qbit_num_sub == 2:
                    optimized_results[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )
                elif qbit_num_sub >= 3:
                    optimized_results[partition_idx] = self.DecomposePartition_Sequential(remapped_subcircuit, subcircuit_parameters, large_partition_config, mini_topologies, involved_qbits, qbit_map)
                else:
                    optimized_results[partition_idx] = self.DecomposePartition_Sequential(remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map)

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="First Synthesis",disable=self.config.get('progressbar', 0) == False) ):
                result = optimized_results[partition_idx]
                if isinstance(result, AsyncResult):
                    optimized_results[partition_idx] = result.get()
                # else: already a resolved result (sequential or single-qubit)

        # ---- Phase 3: ILP partition selection with synthesis-cost weights ----
        weights = []
        for idx, result in enumerate(optimized_results[:len(allparts)]):
            if isinstance(result, SingleQubitPartitionResult):
                weights.append(0)
            else:
                weights.append(result.get_partition_synthesis_score())

        L_parts, fusion_info = ilp_global_optimal(allparts, g, weights=weights)

        # ---- Phase 4: Reuse Phase 2 results (no re-synthesis) ----
        # Build non-overlapping parts from selected allparts + standalone chains.
        # Phase 2 already synthesized each allpart (with surrounded chains included),
        # so we reuse those results directly.
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

        # Non-surrounded chains become standalone SingleQubitPartitionResult entries
        standalone_chains = []
        for chain in single_qubit_chains:
            if chain[0] not in selected_surrounded_starts:
                selected_parts_gates.append(frozenset(chain))
                standalone_chains.append(chain)

        L = topo_sort_partitions(working_circ, self.config["max_partition_size"], selected_parts_gates)

        n_selected = len(L_parts)
        optimized_partitions = []
        for part_idx in L:
            if part_idx < n_selected:
                # Multi-qubit partition — reuse Phase 2 PartitionSynthesisResult
                optimized_partitions.append(optimized_results[L_parts[part_idx]])
            else:
                # Standalone single-qubit chain
                chain = standalone_chains[part_idx - n_selected]
                c = Circuit(qbit_num_orig_circuit)
                chain_params = []
                for gate_idx in chain:
                    c.add_Gate(gate_dict[gate_idx])
                    start = gate_dict[gate_idx].get_Parameter_Start_Index()
                    chain_params.append(working_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
                chain_parameters = np.concatenate(chain_params) if chain_params else np.array([])
                optimized_partitions.append(SingleQubitPartitionResult(c, chain_parameters))

        return optimized_partitions

    # ------------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------------

    def Partition_Aware_Mapping(self, circ: Circuit, orig_parameters: np.ndarray):
        N = circ.get_Qbit_Num()


        optimized_partitions = self.SynthesizeWideCircuit(circ, orig_parameters)

        for partition in optimized_partitions:
            if isinstance(partition, PartitionSynthesisResult):
                partition._topology = self.topology
                partition._topology_cache = self._topology_cache

        DAG, IDAG = self.construct_DAG_and_IDAG(optimized_partitions)

        D = self.compute_distances_bfs(N)
        scoring_partitions = self._build_scoring_partitions(optimized_partitions)

        n_iterations = self.config.get('sabre_iterations', 1)
        n_trials = self.config.get('n_layout_trials', 1)
        random_seed = self.config.get('random_seed', 42)
        routing_start = time.time()
        if n_iterations == 0:
            # Single forward pass from identity layout
            F = self.get_initial_layer(IDAG, N, optimized_partitions)
            partition_order, pi, pi_initial = self.Heuristic_Search(
                F, pi=np.arange(N), DAG=DAG, IDAG=IDAG,
                optimized_partitions=optimized_partitions,
                scoring_partitions=scoring_partitions, D=D,
            )
        else:
            best_pi = None
            best_cost = float('inf')

            for trial in range(max(1, n_trials)):
                rng = np.random.RandomState(random_seed + trial) if n_trials > 1 else None
                pi = np.arange(N)

                for iteration in range(n_iterations):
                    # Reverse pass: walk DAG backwards (swap DAG↔IDAG)
                    F_rev = self.get_final_layer(DAG, N, optimized_partitions)
                    pi, _ = self._heuristic_search_layout_only(
                        F_rev, pi, IDAG, DAG, optimized_partitions, scoring_partitions, D,
                        rng=rng,
                        reverse=True,
                    )

                    # Forward layout-only pass (skip on last iteration — real pass follows)
                    if iteration < n_iterations - 1:
                        F_fwd = self.get_initial_layer(IDAG, N, optimized_partitions)
                        pi, _ = self._heuristic_search_layout_only(
                            F_fwd, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D,
                            rng=rng,
                        )

                # Score this trial: deterministic forward layout-only pass
                F_eval = self.get_initial_layer(IDAG, N, optimized_partitions)
                _, cost = self._heuristic_search_layout_only(
                    F_eval, pi.copy(), DAG, IDAG, optimized_partitions, scoring_partitions, D,
                    rng=None,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_pi = pi.copy()

            # Final forward pass — builds actual circuits
            F = self.get_initial_layer(IDAG, N, optimized_partitions)
            partition_order, pi, pi_initial = self.Heuristic_Search(
                F, best_pi, DAG, IDAG, optimized_partitions, scoring_partitions, D,
            )

        final_circuit, final_parameters = self.Construct_circuit_from_HS(partition_order, optimized_partitions, N)
        self._routing_time = time.time() - routing_start
        self._cnot_pre_cleanup = final_circuit.get_Gate_Nums().get('CNOT', 0)

        # Cleanup phase: re-partition and resynthesize to eliminate
        # redundancies at SWAP-partition boundaries
        if self.config.get('cleanup', True):
            from squander.decomposition.qgd_Wide_Circuit_Optimization import qgd_Wide_Circuit_Optimization
            cleanup_config = dict(self.config)
            cleanup_config['topology'] = self.topology
            cleanup_config['routed'] = True
            cleanup_config['test_subcircuits'] = False
            cleanup_config['test_final_circuit'] = False
            wco = qgd_Wide_Circuit_Optimization(cleanup_config)
            final_circuit, final_parameters = wco.OptimizeWideCircuit(
                final_circuit.get_Flat_Circuit(), final_parameters
            )

        return final_circuit, final_parameters, pi_initial, pi

    # ------------------------------------------------------------------------
    # Heuristic Search
    # ------------------------------------------------------------------------

    def _select_best_candidate(self, partition_candidates, scores, rng=None):
        """Select best candidate, with optional stochastic tie-breaking."""
        scores_array = np.array(scores)
        min_score = np.min(scores_array)
        tolerance = self.config.get('score_tolerance', 0.05)

        if rng is not None and min_score > 0:
            threshold = min_score * (1 + tolerance)
            close_indices = np.where(scores_array <= threshold)[0]
            if len(close_indices) > 1:
                return partition_candidates[rng.choice(close_indices)]
            return partition_candidates[close_indices[0]]
        else:
            return partition_candidates[np.argmin(scores_array)]

    def Heuristic_Search(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D):
        pi_initial = pi.copy()

        resolved_partitions = [False] * len(DAG)
        partition_order = []
        step = 0

        for partition_idx in list(F):
            if isinstance(optimized_partitions[partition_idx], SingleQubitPartitionResult):
                F.remove(partition_idx)
                single_qubit_part = optimized_partitions[partition_idx]
                qubit = single_qubit_part.circuit.get_Qbits()[0]
                single_qubit_part.circuit = single_qubit_part.circuit.Remap_Qbits({int(qubit): int(pi[qubit])},max(D.shape))
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

        max_E_size = self.config.get('max_E_size', 20)
        max_lookahead = self.config.get('max_lookahead', 4)
        E_W = self.config.get('E_weight', 0.5)
        E_alpha = self.config.get('E_alpha', 0.9)

        while len(F) != 0:
                partition_candidates = self.obtain_partition_candidates(F,optimized_partitions)
                if len(partition_candidates) == 0:
                    break
                F_snapshot = tuple(F)

                E = self.generate_extended_set(
                    F, DAG, IDAG, resolved_partitions, optimized_partitions,
                    max_E_size=max_E_size, max_lookahead=max_lookahead
                )

                scores = [
                        self.score_partition_candidate(
                            partition_candidate,
                            F_snapshot,
                            pi,
                            scoring_partitions,
                            D,
                            self._swap_cache,
                            E=E,
                            W=E_W,
                            alpha=E_alpha,
                        )
                        for partition_candidate in partition_candidates
                    ]
                min_partition_candidate = self._select_best_candidate(partition_candidates, scores, rng=None)

                F.remove(min_partition_candidate.partition_idx)
                resolved_partitions[min_partition_candidate.partition_idx] = True
                resolved_count = sum(resolved_partitions)
                pbar.n = resolved_count
                pbar.refresh()

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
                            child_partition.circuit = child_partition.circuit.Remap_Qbits({int(qubit): int(pi[qubit])},max(D.shape))
                            partition_order.append(child_partition)
                            resolved_partitions[child] = True
                            resolved_count = sum(resolved_partitions)
                            pbar.n = resolved_count
                            pbar.refresh()
                            children.extend(DAG[child])
                        else:
                            F.append(child)

        pbar.close()
        return partition_order, pi, pi_initial

    def _heuristic_search_layout_only(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, rng=None, reverse=False):
        """Run heuristic search but only track layout (pi). No circuit modification.

        Args:
            reverse: When True, swap P_i/P_o roles in scoring and layout
                     updates (used for backward passes in SABRE iterations).

        Returns:
            (pi, total_swaps): final layout and total number of SWAPs accumulated.
        """
        resolved_partitions = [False] * len(DAG)
        total_swaps = 0

        # Resolve initial single-qubit partitions
        for partition_idx in list(F):
            if isinstance(optimized_partitions[partition_idx], SingleQubitPartitionResult):
                F.remove(partition_idx)
                resolved_partitions[partition_idx] = True
                for child in DAG[partition_idx]:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        F.append(child)

        max_E_size = self.config.get('max_E_size', 20)
        max_lookahead = self.config.get('max_lookahead', 4)
        E_W = self.config.get('E_weight', 0.5)
        E_alpha = self.config.get('E_alpha', 0.9)

        while F:
            partition_candidates = self.obtain_partition_candidates(F, optimized_partitions)
            if not partition_candidates:
                break

            F_snapshot = tuple(F)

            E = self.generate_extended_set(
                F, DAG, IDAG, resolved_partitions, optimized_partitions,
                max_E_size=max_E_size, max_lookahead=max_lookahead
            )

            scores = [
                self.score_partition_candidate(
                    pc, F_snapshot, pi, scoring_partitions, D,
                    self._swap_cache,
                    E=E, W=E_W, alpha=E_alpha,
                    reverse=reverse,
                )
                for pc in partition_candidates
            ]

            best = self._select_best_candidate(partition_candidates, scores, rng=rng)
            F.remove(best.partition_idx)
            resolved_partitions[best.partition_idx] = True

            swaps, pi = best.transform_pi(pi, D, self._swap_cache, reverse=reverse)
            total_swaps += len(swaps)

            # Promote children
            for child in DAG[best.partition_idx]:
                if not resolved_partitions[child] and child not in F:
                    if all(resolved_partitions[p] for p in IDAG[child]):
                        if isinstance(optimized_partitions[child], SingleQubitPartitionResult):
                            resolved_partitions[child] = True
                            stack = list(DAG[child])
                            while stack:
                                gc = stack.pop()
                                if not resolved_partitions[gc] and gc not in F:
                                    if all(resolved_partitions[p] for p in IDAG[gc]):
                                        if isinstance(optimized_partitions[gc], SingleQubitPartitionResult):
                                            resolved_partitions[gc] = True
                                            stack.extend(DAG[gc])
                                        else:
                                            F.append(gc)
                        else:
                            F.append(child)

        return pi, total_swaps

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
    def score_partition_candidate(partition_candidate, F, pi, scoring_partitions, D, swap_cache,
                                  E=None, W=0.5, alpha=0.9, reverse=False):
        score = 0
        swap_weight = 1
        swaps, output_perm = partition_candidate.transform_pi(pi, D, swap_cache, reverse=reverse)
        score += swap_weight * len(swaps) * 3
        score += 0.1*len(partition_candidate.circuit_structure)

        for partition_idx in F:
            partition = scoring_partitions[partition_idx]
            if partition is None or partition_idx == partition_candidate.partition_idx:
                continue
            qbit_map_inv = {v: q for q, v in partition.qubit_map.items()}
            mini_scores = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                for pdx, (P_i, P_o) in enumerate(partition.permutations_pairs[tdx]):
                    cnot_count = len(partition.circuit_structures[tdx][pdx])
                    # In reverse pass, the "entry" side of neighbor partitions
                    # is their output (P_o), not their input (P_i).
                    P_route = P_o if reverse else P_i
                    if mini_topology:
                        routing_cost = swap_weight * 3 * sum(
                            max(0, D[int(output_perm[qbit_map_inv[P_route[u]]])][int(output_perm[qbit_map_inv[P_route[v]]])] - 1)
                            for u, v in mini_topology
                        )
                    else:
                        routing_cost = 0
                    mini_scores.append(routing_cost + cnot_count)
            if mini_scores:
                score += np.min(mini_scores)

        # Extended set look-ahead scoring
        if E:
            e_score = 0
            for partition_idx, depth in E:
                partition = scoring_partitions[partition_idx]
                if partition is None or partition_idx == partition_candidate.partition_idx:
                    continue
                qbit_map_inv = {v: q for q, v in partition.qubit_map.items()}
                mini_scores = []
                for tdx, mini_topology in enumerate(partition.mini_topologies):
                    for pdx, (P_i, P_o) in enumerate(partition.permutations_pairs[tdx]):
                        cnot_count = len(partition.circuit_structures[tdx][pdx])
                        P_route = P_o if reverse else P_i
                        if mini_topology:
                            routing_cost = swap_weight * 3 * sum(
                                max(0, D[int(output_perm[qbit_map_inv[P_route[u]]])][int(output_perm[qbit_map_inv[P_route[v]]])] - 1)
                                for u, v in mini_topology
                            )
                        else:
                            routing_cost = 0
                        mini_scores.append(routing_cost + cnot_count)
                if mini_scores:
                    e_score += np.min(mini_scores) * (alpha ** depth)
            if len(E) > 0:
                score += W * e_score / len(E)

        return score

    # ------------------------------------------------------------------------
    # Extended Set
    # ------------------------------------------------------------------------

    @staticmethod
    def generate_extended_set(F, DAG, IDAG, resolved_partitions, optimized_partitions,
                              max_E_size=20, max_lookahead=4):
        """
        Generate SABRE-style extended set: multi-qubit partitions near the
        front layer, up to ``max_lookahead`` levels deep and ``max_E_size``
        entries.  Returns list of (partition_idx, depth) tuples.
        """
        E = []
        E_set = set()
        F_set = set(F)

        for front_idx in F:
            if len(E) >= max_E_size:
                break

            # BFS from front_idx through DAG children
            queue = []  # (child_idx, depth)
            for child in DAG[front_idx]:
                queue.append((child, 1))

            while queue and len(E) < max_E_size:
                child_idx, depth = queue.pop(0)
                if depth > max_lookahead:
                    continue
                if child_idx in E_set or child_idx in F_set:
                    continue
                if resolved_partitions[child_idx]:
                    continue

                # Check all parents resolved (except those still in F)
                parents_resolved = all(
                    resolved_partitions[p] or p in F_set
                    for p in IDAG[child_idx]
                )
                if not parents_resolved:
                    continue

                # Skip single-qubit partitions — follow through them
                if isinstance(optimized_partitions[child_idx], SingleQubitPartitionResult):
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

    def get_final_layer(self, DAG, N, optimized_partitions):
        final_layer = []
        active_qbits = list(range(N))
        for idx in range(len(DAG) - 1, -1, -1):
            if len(DAG[idx]) == 0:
                final_layer.append(idx)
                for qbit in optimized_partitions[idx].involved_qbits:
                    if qbit in active_qbits:
                        active_qbits.remove(qbit)
            if len(active_qbits) == 0:
                break
        return final_layer
            
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
        
        return D #multiply by 3 to make it CNOT cost instead of SWAP cost


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

