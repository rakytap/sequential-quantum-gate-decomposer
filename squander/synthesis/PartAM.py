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
from squander.partitioning.ilp import get_all_partitions, _get_topo_order, topo_sort_partitions, ilp_global_optimal, recombine_single_qubit_chains

import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable

import multiprocessing as mp
from multiprocessing import Process, Pool
import os
from typing import List, Set, Tuple, FrozenSet
from tqdm import tqdm
from collections import deque, defaultdict
import numpy as np

from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE
from squander.synthesis.PartAM_utils import (get_subtopologies_of_type, get_unique_subtopologies, get_canonical_form,
SingleQubitPartitionResult, PartitionSynthesisResult, 
PartitionCandidate, check_circuit_compatibility, construct_swap_circuit)

def _score_partition_candidate_helper(partition_candidate, F, pi, optimized_partitions, sDAG, D, swap_cache, topology_cache, topology):
    """
    Helper function for parallel candidate scoring.
    This is a module-level function to enable pickling for multiprocessing.
    """
    score_F = 0
    score_E = 0
    E_visited_partitions = set()
    
    # Use swap cache if provided
    swaps, output_perm = partition_candidate.transform_pi(pi, D, swap_cache)
    score_F += len(swaps)*3
    score_F += len(partition_candidate.circuit_structure)

    # Cache for transform_pi results to avoid redundant computation
    transform_cache = {}

    # Helper function to get topology candidates
    def get_topology_candidates_cached(mini_topology):
        if topology_cache is not None:
            from squander.synthesis.PartAM_utils import get_canonical_form
            target_qubits = set()
            for u, v in mini_topology:
                target_qubits.add(u)
                target_qubits.add(v)
            if target_qubits:
                canonical_key = get_canonical_form(target_qubits, mini_topology)
                if canonical_key in topology_cache:
                    return topology_cache[canonical_key]
                else:
                    candidates = get_subtopologies_of_type(topology, mini_topology)
                    topology_cache[canonical_key] = candidates
                    return candidates
        return get_subtopologies_of_type(topology, mini_topology)

    for partition_idx in sDAG[partition_candidate.partition_idx]:
        if partition_idx in E_visited_partitions:
            continue
        E_visited_partitions.add(partition_idx)
        mini_scores = []
        partition_result = optimized_partitions[partition_idx]
        for tdx, mini_topology in enumerate(partition_result.mini_topologies):
            if hasattr(partition_result, 'get_topology_candidates'):
                topology_candidates = partition_result.get_topology_candidates(tdx)
            else:
                topology_candidates = get_topology_candidates_cached(mini_topology)
            for topology_candidate in topology_candidates:
                for pdx, permutation_pair in enumerate(partition_result.permutations_pairs[tdx]):
                    cache_key = (partition_idx, tdx, pdx, tuple(sorted(topology_candidate)), tuple(output_perm))
                    if cache_key not in transform_cache:
                        new_cand = PartitionCandidate(partition_idx,tdx,pdx,partition_result.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition_result.qubit_map,partition_result.involved_qbits)
                        swap_count = len(new_cand.transform_pi(output_perm,D, swap_cache)[0])
                        transform_cache[cache_key] = swap_count * 3 + len(new_cand.circuit_structure)
                    mini_scores.append(transform_cache[cache_key])
        if mini_scores:
            score_E += min(mini_scores)

    for partition_idx in F:
        partition = optimized_partitions[partition_idx]
        mini_scores = []
        for tdx, mini_topology in enumerate(partition.mini_topologies):
            if hasattr(partition, 'get_topology_candidates'):
                topology_candidates = partition.get_topology_candidates(tdx)
            else:
                topology_candidates = get_topology_candidates_cached(mini_topology)
            for topology_candidate in topology_candidates:
                for pdx, permutation_pair in enumerate(partition.permutations_pairs[tdx]):
                    cache_key = (partition_idx, tdx, pdx, tuple(sorted(topology_candidate)), tuple(output_perm))
                    if cache_key not in transform_cache:
                        new_cand = PartitionCandidate(partition_idx,tdx,pdx,partition.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition.qubit_map,partition.involved_qbits)
                        swap_count = len(new_cand.transform_pi(output_perm,D, swap_cache)[0])
                        transform_cache[cache_key] = swap_count * 3 + len(new_cand.circuit_structure)
                    mini_scores.append(transform_cache[cache_key])
        if mini_scores:
            score_F += min(mini_scores)

        for partition_idx_E in sDAG[partition_idx]:
            if partition_idx_E in E_visited_partitions:
                continue
            E_visited_partitions.add(partition_idx_E)
            mini_scores = []
            partition_result_E = optimized_partitions[partition_idx_E]
            for tdx, mini_topology in enumerate(partition_result_E.mini_topologies):
                if hasattr(partition_result_E, 'get_topology_candidates'):
                    topology_candidates = partition_result_E.get_topology_candidates(tdx)
                else:
                    topology_candidates = get_topology_candidates_cached(mini_topology)
                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(partition_result_E.permutations_pairs[tdx]):
                        cache_key = (partition_idx_E, tdx, pdx, tuple(sorted(topology_candidate)), tuple(output_perm))
                        if cache_key not in transform_cache:
                            new_cand = PartitionCandidate(partition_idx_E,tdx,pdx,partition_result_E.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition_result_E.qubit_map,partition_result_E.involved_qbits)
                            swap_count = len(new_cand.transform_pi(output_perm,D, swap_cache)[0])
                            transform_cache[cache_key] = swap_count * 3 + len(new_cand.circuit_structure)
                        mini_scores.append(transform_cache[cache_key])
            if mini_scores:
                score_E += min(mini_scores)
    
    # Safety check for division by zero
    if len(E_visited_partitions) == 0:
        E_score = 0.0
    else:
        E_score = 0.2 * score_E / len(E_visited_partitions)
    
    if len(F) == 0:
        F_score = 0.0
    else:
        F_score = score_F / len(F)
    
    return E_score + F_score

class qgd_Partition_Aware_Mapping:

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
        strategy = self.config['strategy']
        allowed_strategies = ['TreeSearch', 'TabuSearch', 'Adaptive']
        if not strategy in allowed_strategies:
            raise Exception(f"The strategy should be either of {allowed_strategies}, got {strategy}.")
        
        # Initialize caches for performance optimization
        self._topology_cache = {}  # {frozenset(edges): [topology_candidates]}
        self._swap_cache = {}     # {(pi_tuple, qbit_map_frozen): (swaps, output_perm)}

    def _get_subtopologies_of_type_cached(self, mini_topology):
        """
        Cached version of get_subtopologies_of_type.
        Uses canonical form of mini_topology as cache key.
        """
        from squander.synthesis.PartAM_utils import get_canonical_form
        
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
                optimized_partitions[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="Second Synthesis",disable=self.config.get('progressbar', 0) == False) ):
                optimized_partitions[partition_idx] = optimized_partitions[partition_idx].get()
                
        
        return optimized_partitions

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
        pi_list = pi.tolist() if hasattr(pi, 'tolist') else list(pi)
        
        F = self.get_initial_layer(IDAG, circ.get_Qbit_Num(),optimized_partitions)
        
        partition_order, pi_final = self.Heuristic_Search(F,pi.copy(),DAG,IDAG, optimized_partitions,D, sDAG)
        pi_final_list = pi_final.tolist() if hasattr(pi_final, 'tolist') else list(pi_final)
        
        final_circuit, final_parameters = self.Construct_circuit_from_HS(partition_order,optimized_partitions, circ.get_Qbit_Num())
        
        return final_circuit, final_parameters, pi, pi_final

    def Heuristic_Search(self, F, pi, DAG, IDAG, optimized_partitions, D, sDAG):
        resolved_partitions = [False] * len(DAG)
        partition_order = []
        step = 0
        
        # Determine number of processes for parallel scoring
        num_processes = self.config.get('parallel_scoring_processes', min(mp.cpu_count(), 64))
        use_parallel = num_processes > 1 and len(F) > 1
        
        # Initialize progress bar
        total_partitions = len(DAG)
        pbar = tqdm(total=total_partitions, desc="Heuristic Search", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} resolved', 
                   disable=self.config.get('progressbar', 0) == False)
        
        while len(F) != 0:
            scores = []
            partition_candidates = self.obtain_partition_candidates(F,optimized_partitions)
            if len(partition_candidates) != 0:
                if use_parallel and len(partition_candidates) > 1:
                    # Parallel scoring
                    with Pool(processes=num_processes) as pool:
                        score_args = [(cand, F, pi, optimized_partitions, sDAG, D, self._swap_cache, self._topology_cache, self.topology) 
                                     for cand in partition_candidates]
                        scores = pool.starmap(_score_partition_candidate_helper, score_args)
                else:
                    # Sequential scoring
                    for partition_candidate in partition_candidates:
                        score = self.score_partition_candidate(partition_candidate, F, pi, optimized_partitions, sDAG, D)
                        scores.append(score)
            if len(scores) == 0:
                break
            min_idx = np.argmin(scores)
            min_partition_candidate = partition_candidates[min_idx]
            
            F.remove(min_partition_candidate.partition_idx)
            resolved_partitions[min_partition_candidate.partition_idx] = True
            resolved_count = sum(resolved_partitions)
            pbar.n = resolved_count
            pbar.refresh()
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
                        child_partition.circuit.map_circuit({qubit: pi[qubit]})
                        partition_order.append(child_partition)
                        resolved_partitions[child] = True
                        resolved_count = sum(resolved_partitions)
                        pbar.n = resolved_count
                        pbar.refresh()
                        children.extend(DAG[child])
                    else:
                        F.append(child)
        pbar.close()
        return partition_order, pi

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
        
        return final_circuit, final_parameters

    def score_partition_candidate(self, partition_candidate, F,  pi, optimized_partitions, sDAG, D):
        score_F = 0
        score_E = 0
        E_visited_partitions = set()  # Changed to set for O(1) membership checks
        swaps, output_perm = partition_candidate.transform_pi(pi, D, self._swap_cache)
        score_F += len(swaps)*3
        score_F += len(partition_candidate.circuit_structure)

        # Cache for transform_pi results to avoid redundant computation
        # Key: (partition_idx, topology_idx, permutation_idx, topology_candidate_tuple, output_perm_tuple)
        transform_cache = {}

        for partition_idx in sDAG[partition_candidate.partition_idx]:
            if partition_idx in E_visited_partitions:
                continue
            E_visited_partitions.add(partition_idx)
            mini_scores = []
            partition_result = optimized_partitions[partition_idx]
            for tdx, mini_topology in enumerate(partition_result.mini_topologies):
                # Use pre-computed topology candidates if available, otherwise compute and cache
                if hasattr(partition_result, 'get_topology_candidates'):
                    topology_candidates = partition_result.get_topology_candidates(tdx)
                else:
                    topology_candidates = self._get_subtopologies_of_type_cached(mini_topology)
                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(partition_result.permutations_pairs[tdx]):
                        # Create cache key for this candidate's transform_pi result
                        cache_key = (partition_idx, tdx, pdx, tuple(sorted(topology_candidate)), tuple(output_perm))
                        if cache_key not in transform_cache:
                            new_cand = PartitionCandidate(partition_idx,tdx,pdx,partition_result.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition_result.qubit_map,partition_result.involved_qbits)
                            swap_count = len(new_cand.transform_pi(output_perm,D, self._swap_cache)[0])
                            transform_cache[cache_key] = swap_count * 3 + len(new_cand.circuit_structure)
                        mini_scores.append(transform_cache[cache_key])
            if mini_scores:
                score_E += min(mini_scores)

        for partition_idx in F:
            partition = optimized_partitions[partition_idx]
            mini_scores = []
            for tdx, mini_topology in enumerate(partition.mini_topologies):
                # Use pre-computed topology candidates if available, otherwise compute and cache
                if hasattr(partition, 'get_topology_candidates'):
                    topology_candidates = partition.get_topology_candidates(tdx)
                else:
                    topology_candidates = self._get_subtopologies_of_type_cached(mini_topology)
                for topology_candidate in topology_candidates:
                    for pdx, permutation_pair in enumerate(partition.permutations_pairs[tdx]):
                        # Create cache key for this candidate's transform_pi result
                        cache_key = (partition_idx, tdx, pdx, tuple(sorted(topology_candidate)), tuple(output_perm))
                        if cache_key not in transform_cache:
                            new_cand = PartitionCandidate(partition_idx,tdx,pdx,partition.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition.qubit_map,partition.involved_qbits)
                            swap_count = len(new_cand.transform_pi(output_perm,D, self._swap_cache)[0])
                            transform_cache[cache_key] = swap_count * 3 + len(new_cand.circuit_structure)
                        mini_scores.append(transform_cache[cache_key])
            if mini_scores:
                score_F += min(mini_scores)

            for partition_idx_E in sDAG[partition_idx]:
                if partition_idx_E in E_visited_partitions:
                    continue
                E_visited_partitions.add(partition_idx_E)
                mini_scores = []
                partition_result_E = optimized_partitions[partition_idx_E]
                for tdx, mini_topology in enumerate(partition_result_E.mini_topologies):
                    # Use pre-computed topology candidates if available, otherwise compute and cache
                    if hasattr(partition_result_E, 'get_topology_candidates'):
                        topology_candidates = partition_result_E.get_topology_candidates(tdx)
                    else:
                        topology_candidates = self._get_subtopologies_of_type_cached(mini_topology)
                    for topology_candidate in topology_candidates:
                        for pdx, permutation_pair in enumerate(partition_result_E.permutations_pairs[tdx]):
                            # Create cache key for this candidate's transform_pi result
                            cache_key = (partition_idx_E, tdx, pdx, tuple(sorted(topology_candidate)), tuple(output_perm))
                            if cache_key not in transform_cache:
                                new_cand = PartitionCandidate(partition_idx_E,tdx,pdx,partition_result_E.circuit_structures[tdx][pdx],permutation_pair[0],permutation_pair[1],topology_candidate,mini_topology,partition_result_E.qubit_map,partition_result_E.involved_qbits)
                                swap_count = len(new_cand.transform_pi(output_perm,D, self._swap_cache)[0])
                                transform_cache[cache_key] = swap_count * 3 + len(new_cand.circuit_structure)
                            mini_scores.append(transform_cache[cache_key])
                if mini_scores:
                    score_E += min(mini_scores)
        # Safety check for division by zero
        if len(E_visited_partitions) == 0:
            E_score = 0.0
        else:
            E_score = 0.2 * score_E / len(E_visited_partitions)
        
        if len(F) == 0:
            F_score = 0.0
        else:
            F_score = score_F / len(F)
        
        return E_score + F_score

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
        sDAG = []
        
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
            sDAG.append(children)
            
        return sDAG

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

        # Count interactions between qubits
        interaction_count = defaultdict(int)
        gates = circuit.get_Gates()
        
        for gate in gates:
            if gate.get_Control_Qbit() != -1:
                q1 = gate.get_Target_Qbit()
                q2 = gate.get_Control_Qbit()
                if q1 < N and q2 < N:
                    key = (min(q1, q2), max(q1, q2))
                    interaction_count[key] += 1
        
        if not interaction_count:
            # No 2-qubit gates, use trivial mapping
            return np.arange(N)
        
        # Find most interacting qubit pair
        most_connected = max(interaction_count.items(), key=lambda x: x[1])
        q1, q2 = most_connected[0]
        
        # Find physical qubits that are connected
        # Start with an arbitrary connected pair
        for edge in self.config['topology']:
            p1, p2 = edge
            break  # Just take first edge
        
        # Initialize mapping
        pi = np.arange(N)
        
        # Place most interacting qubits on connected physical qubits
        pi[q1] = p1
        pi[q2] = p2
        
        # Place other qubits using greedy approach
        placed_logical = {q1, q2}
        placed_physical = {p1, p2}
        
        # For each remaining logical qubit, find where to place it
        remaining_logical = [q for q in range(N) if q not in placed_logical]
        
        # Sort by how much they interact with already placed qubits
        def interaction_score(q):
            score = 0
            for placed_q in placed_logical:
                key = (min(q, placed_q), max(q, placed_q))
                score += interaction_count.get(key, 0)
            return score
        
        remaining_logical.sort(key=interaction_score, reverse=True)
        
        # Place them near their interacting partners
        for logical_q in remaining_logical:
            # Find best physical location
            best_physical = None
            best_score = float('inf')
            
            for physical_q in range(N):
                if physical_q not in placed_physical:
                    # Calculate average distance to interacting qubits
                    total_dist = 0
                    count = 0
                    for other_q in placed_logical:
                        key = (min(logical_q, other_q), max(logical_q, other_q))
                        weight = interaction_count.get(key, 0)
                        if weight > 0:
                            other_physical = pi[other_q]
                            total_dist += D[physical_q][other_physical] * weight
                            count += weight
                    
                    if count > 0:
                        avg_dist = total_dist / count
                    else:
                        avg_dist = 0
                    
                    if avg_dist < best_score:
                        best_score = avg_dist
                        best_physical = physical_q
            
            if best_physical is not None:
                pi[logical_q] = best_physical
                placed_logical.add(logical_q)
                placed_physical.add(best_physical)
        
        return pi
