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
    SingleQubitPartitionResult,
    PartitionSynthesisResult,
    PartitionCandidate,
    check_circuit_compatibility,
    construct_swap_circuit,
    calculate_dist_small,
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
_WORKER_S_DAG: Optional[List[List[int]]] = None
_WORKER_DISTANCE_MATRIX: Optional[np.ndarray] = None
_WORKER_SWAP_CACHE: Optional[Dict] = None


def _init_scoring_worker(scoring_partitions, sdag, distance_matrix):
    """Initializer for process-based scoring workers."""
    global _WORKER_SCORING_PARTITIONS, _WORKER_S_DAG, _WORKER_DISTANCE_MATRIX, _WORKER_SWAP_CACHE
    _WORKER_SCORING_PARTITIONS = scoring_partitions
    _WORKER_S_DAG = sdag
    _WORKER_DISTANCE_MATRIX = distance_matrix
    _WORKER_SWAP_CACHE = {}


def _score_candidate_worker(payload):
    """
    Worker wrapper that reconstructs scoring inputs from a lightweight payload.
    Payload format: (PartitionCandidate, F_snapshot, pi_snapshot)
    """
    if (
        _WORKER_SCORING_PARTITIONS is None
        or _WORKER_S_DAG is None
        or _WORKER_DISTANCE_MATRIX is None
    ):
        raise RuntimeError("Scoring worker not initialized with shared data.")
    partition_candidate, F_snapshot, pi_snapshot, lookahead_gates = payload
    return qgd_Partition_Aware_Mapping.score_partition_candidate(
        partition_candidate,
        F_snapshot,
        pi_snapshot,
        _WORKER_SCORING_PARTITIONS,
        _WORKER_S_DAG,
        _WORKER_DISTANCE_MATRIX,
        _WORKER_SWAP_CACHE,
        lookahead_gates
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
        self.config.setdefault('hs_score_workers', os.cpu_count() or 1)
        strategy = self.config['strategy']
        allowed_strategies = ['TreeSearch', 'TabuSearch', 'Adaptive']
        if not strategy in allowed_strategies:
            raise Exception(f"The strategy should be either of {allowed_strategies}, got {strategy}.")
        
        # Initialize caches for performance optimization
        self._topology_cache = {}  # {frozenset(edges): [topology_candidates]}
        self._swap_cache = {}     # {(pi_tuple, qbit_map_frozen): (swaps, output_perm)}

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
        print(N)
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
    def DecomposePartition_and_Perm(Umtx: np.ndarray, config: dict, mini_topology = None) -> Circuit:
        """
        Call to decompose a partition
        """
        strategy = config["strategy"]
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
        cDecompose.set_Optimization_Tolerance( config["tolerance"] )
        cDecompose.set_Optimizer( config["optimizer"] )
        cDecompose.Start_Decomposition()
        squander_circuit = cDecompose.get_Circuit()
        parameters       = cDecompose.get_Optimized_Parameters()
        return squander_circuit, parameters

    # ------------------------------------------------------------------------
    # Circuit Synthesis
    # ------------------------------------------------------------------------

    def SynthesizeWideCircuit(self, circ, orig_parameters):
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(circ, self.config["max_partition_size"])
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
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
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)
        # Only add single-qubit chains as separate partitions if minimum_partition_size allows it
        for chain in single_qubit_chains:
            c = Circuit( qbit_num_orig_circuit )
            for gate_idx in chain:
                c.add_Gate( gate_dict[gate_idx] )
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(orig_parameters[start:start + gate_dict[gate_idx].get_Parameter_Num()])
            partitioned_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)

        qbit_num_orig_circuit = circ.get_Qbit_Num()


        subcircuits = partitioned_circuit.get_Gates()

        optimized_results = [None] * len(subcircuits)

        with Pool(processes=mp.cpu_count()) as pool:
            for partition_idx, subcircuit in enumerate( subcircuits ):

                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
                qbit_num_orig_circuit = subcircuit.get_Qbit_Num()
                involved_qbits = subcircuit.get_Qbits()

                qbit_num = len( involved_qbits )
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num)
                qbit_map = {}
                for idx in range( len(involved_qbits) ):
                    qbit_map[ involved_qbits[idx] ] = idx
                remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )
                optimized_results[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="First Synthesis",disable=self.config.get('progressbar', 0) == False) ):
                optimized_results[partition_idx] = optimized_results[partition_idx].get()
        
        weights = [result.get_partition_synthesis_score() for result in optimized_results[:len(allparts)]]
        L_parts, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L_parts], fusion_info)
        L = topo_sort_partitions(circ, self.config["max_partition_size"], parts)
        from squander.partitioning.kahn import kahn_partition_preparts
        from squander.partitioning.tools import translate_param_order
        partitioned_circuit, param_order, _ = kahn_partition_preparts(circ, self.config["max_partition_size"], [parts[i] for i in L])
        parameters = translate_param_order(orig_parameters, param_order)

        subcircuits = partitioned_circuit.get_Gates()

        # the list of optimized subcircuits
        optimized_partitions = [None] * len(subcircuits)

        with Pool(processes=mp.cpu_count()) as pool:
            for partition_idx, subcircuit in enumerate( subcircuits ):

                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
                qbit_num_orig_circuit = subcircuit.get_Qbit_Num()
                involved_qbits = subcircuit.get_Qbits()

                qbit_num = len( involved_qbits )
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num)
                qbit_map = {}
                for idx in range( len(involved_qbits) ):
                    qbit_map[ involved_qbits[idx] ] = idx
                remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )
                optimized_partitions[partition_idx] = pool.apply_async( self.DecomposePartition_Full, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="Second Synthesis",disable=self.config.get('progressbar', 0) == False) ):
                optimized_partitions[partition_idx] = optimized_partitions[partition_idx].get()
                
        
        return optimized_partitions

    # ------------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------------

    def Partition_Aware_Mapping(self, circ: Circuit, orig_parameters: np.ndarray):
        
        optimized_partitions = self.SynthesizeWideCircuit(circ, orig_parameters)
        
        # Initialize topology candidates in PartitionSynthesisResult objects
        for partition in optimized_partitions:
            if isinstance(partition, PartitionSynthesisResult):
                partition._topology = self.topology
                partition._topology_cache = self._topology_cache

        
        DAG, IDAG = self.construct_DAG_and_IDAG(optimized_partitions)
        sDAG = self.construct_sDAG(optimized_partitions)
        
        D = self.compute_distances_bfs(circ.get_Qbit_Num())
        pi = self._compute_smart_initial_layout(circ, circ.get_Qbit_Num(), D)
        
        F = self.get_initial_layer(IDAG, circ.get_Qbit_Num(),optimized_partitions)
        scoring_partitions = self._build_scoring_partitions(optimized_partitions)
        
        partition_order, pi, pi_initial = self.Heuristic_Search(F,pi.copy(),DAG,IDAG, optimized_partitions,scoring_partitions,D, sDAG)
        
        final_circuit, final_parameters = self.Construct_circuit_from_HS(partition_order,optimized_partitions, circ.get_Qbit_Num())
        
        return final_circuit, final_parameters, pi_initial, pi

    # ------------------------------------------------------------------------
    # Heuristic Search
    # ------------------------------------------------------------------------

    def Heuristic_Search(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, sDAG):
        pi_initial = pi.copy()

        resolved_partitions = [False] * len(DAG)
        partition_order = []
        step = 0
        for partition_idx in F:
            if isinstance(optimized_partitions[partition_idx], SingleQubitPartitionResult):
                F.remove(partition_idx)
                single_qubit_part = optimized_partitions[partition_idx]
                qubit = single_qubit_part.circuit.get_Qbits()[0]
                single_qubit_part.circuit.Remap_Qbits({int(qubit): int(pi[qubit])},max(D.shape))
                partition_order.append(single_qubit_part)

                resolved_partitions[partition_idx] = True
                children = DAG[partition_idx]
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
        
        configured_workers = self.config.get('hs_score_workers', os.cpu_count() or 1)
        score_workers = max(1, configured_workers if configured_workers else 1)
        executor: Optional[ProcessPoolExecutor] = None
        if score_workers > 1:
            try:
                executor = ProcessPoolExecutor(
                    max_workers=score_workers,
                    initializer=_init_scoring_worker,
                    initargs=(scoring_partitions, sDAG, D),
                )
            except Exception as exc:
                logging.warning(
                    "Falling back to sequential heuristic scoring: %s",
                    exc,
                )
                executor = None

        try:
            while len(F) != 0:
                lookahead_gates = None
                partition_candidates = self.obtain_partition_candidates(F,optimized_partitions)
                if len(partition_candidates) == 0:
                    break
                F_snapshot = tuple(F)
                if executor is not None:
                    pi_snapshot = tuple(int(x) for x in pi)
                    payloads = [
                        (partition_candidate, F_snapshot, pi_snapshot, lookahead_gates)
                        for partition_candidate in partition_candidates
                    ]
                    scores = list(executor.map(_score_candidate_worker, payloads))
                else:
                    scores = [
                        self.score_partition_candidate(
                            partition_candidate,
                            F_snapshot,
                            pi,
                            scoring_partitions,
                            sDAG,
                            D,
                            self._swap_cache,
                            lookahead_gates
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
                pi_prev = pi # Save previous pi state for filtering
                swap_order, pi = min_partition_candidate.transform_pi(pi, D, self._swap_cache)
                if len(swap_order)!=0:
                    partition_order.append(construct_swap_circuit(swap_order, len(pi)))
                
                
                partition_order.append(min_partition_candidate)
                children = DAG[min_partition_candidate.partition_idx]
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
                            print(int(qubit),int(pi[qubit]))
                            child_partition.circuit.Remap_Qbits({int(qubit): int(pi[qubit])},max(D.shape))
                            partition_order.append(child_partition)
                            resolved_partitions[child] = True
                            resolved_count = sum(resolved_partitions)
                            pbar.n = resolved_count
                            pbar.refresh()
                            children.extend(DAG[child])
                        else:
                            F.append(child)
        finally:
            if executor is not None:
                executor.shutdown()
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
            final_parameters = np.concatenate(final_parameters,axis=0)
        else:
            final_parameters = np.array([])
        if not check_circuit_compatibility(final_circuit,self.topology):
            print("ERROR: Final circuit is not compatible with device topology!")
        return final_circuit, final_parameters
    
    # ------------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------------

    @staticmethod
    def score_partition_candidate(partition_candidate, F,  pi, scoring_partitions, sDAG, D, swap_cache, lookahead_gates=None):
        score_F = 0
        score_E = 0
        E_partitions = set()  # Changed to set for O(1) membership checks
        E_partitions_1 = set()
        E_partitions_2 = set()
        swap_weight = 4
        swaps, output_perm = partition_candidate.transform_pi(pi, D, swap_cache, lookahead_gates)
        score_F += swap_weight*len(swaps)*3
        score_F += len(partition_candidate.circuit_structure)

        # Safety check: ensure partition_idx is valid for sDAG
        if partition_candidate.partition_idx < len(sDAG):
            for partition_idx in sDAG[partition_candidate.partition_idx]:
                if partition_idx in E_partitions:
                    continue
                E_partitions.add(partition_idx)


        for partition_idx in F:
            partition = scoring_partitions[partition_idx]
            if partition is None or partition_idx == partition_candidate.partition_idx:
                continue
            mini_scores = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                dist_placeholder = swap_weight*3*calculate_dist_small(mini_topology,partition.qubit_map,D,output_perm)
                circuit_length = np.min([len(circ) for circ in partition.circuit_structures[tdx]])
                score = dist_placeholder + circuit_length
                mini_scores.append(score)
            if mini_scores:
                score_F += np.min(mini_scores)

            # Safety check: ensure partition_idx is valid for sDAG
            if partition_idx < len(sDAG):
                for partition_idx_E in sDAG[partition_idx]:
                    if partition_idx_E in E_partitions:
                        continue
                    E_partitions.add(partition_idx_E)

        #check the secondary children            
        for partition_idx in E_partitions: 
            if partition_idx < len(sDAG):
                for partition_idx_E in sDAG[partition_idx]:
                    if partition_idx_E in E_partitions or partition_idx_E in E_partitions_1:
                        continue
                    E_partitions_1.add(partition_idx_E)
        #score all
        for partition_idx in E_partitions:
            mini_scores = []
            partition_result = scoring_partitions[partition_idx]
            if partition_result is None:
                continue
            for tdx, mini_topology in enumerate(partition_result.mini_topologies):
                dist_placeholder = 3*calculate_dist_small(mini_topology,partition_result.qubit_map,D,output_perm)
                circuit_length = np.min([len(circ) for circ in partition_result.circuit_structures[tdx]])
                score = dist_placeholder + circuit_length
                mini_scores.append(score)
            if mini_scores:
                score_E += np.min(mini_scores)

        coeff_E = 0.3
        if len(E_partitions) == 0:
            E_score = 0.0
        else:
            E_score = coeff_E * score_E 
        
        F_score = 0.7*score_F
        
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