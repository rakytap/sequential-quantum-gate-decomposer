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
    recombine_single_qubit_chains,
)

import numpy as np

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import multiprocessing as mp
from multiprocessing import Pool
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

    def compute_transition_cost(self, result_pred, result_succ, D):
        """
        Compute the minimum transition cost between two partitions over all
        (topology, P_o, node_mapping) configs of pred and (topology, P_i, node_mapping)
        configs of succ.

        The cost measures how far each qubit involved in succ needs to travel
        from its position after pred's output to where succ's input requires it.

        Args:
            result_pred: PartitionSynthesisResult for the predecessor partition
            result_succ: PartitionSynthesisResult for the successor partition
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

        # ---- Phase 3: ILP partition selection with synthesis-cost weights ----
        weights = []
        for idx, result in enumerate(optimized_results[:len(allparts)]):
            if isinstance(result, SingleQubitPartitionResult):
                weights.append(0)
            else:
                weights.append(result.get_partition_synthesis_score())

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
                cost_fwd = self.compute_transition_cost(result_pred, result_succ, D)
                cost_rev = self.compute_transition_cost(result_succ, result_pred, D)
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
        pi = np.arange(N)

        F = self.get_initial_layer(IDAG, N, optimized_partitions)
        scoring_partitions = self._build_scoring_partitions(optimized_partitions)

        partition_order, pi, pi_initial = self.Heuristic_Search(F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, free_initial_routing=True)

        final_circuit, final_parameters = self.Construct_circuit_from_HS(partition_order, optimized_partitions, N)

        return final_circuit, final_parameters, pi_initial, pi

    # ------------------------------------------------------------------------
    # Heuristic Search
    # ------------------------------------------------------------------------

    def Heuristic_Search(self, F, pi, DAG, IDAG, optimized_partitions, scoring_partitions, D, free_initial_routing=False):
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

        max_E_size = self.config.get('max_E_size', 20)
        max_lookahead = self.config.get('max_lookahead', 4)
        E_W = self.config.get('E_weight', 0.5)
        E_alpha = self.config.get('E_alpha', 0.9)

        while len(F) != 0:
                partition_candidates = self.obtain_partition_candidates(F,optimized_partitions)
                if len(partition_candidates) == 0:
                    break
                F_snapshot = tuple(F)
                use_free_routing = not first_routing_done

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
                            free_routing=use_free_routing,
                            E=E,
                            W=E_W,
                            alpha=E_alpha,
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
    def score_partition_candidate(partition_candidate, F, pi, scoring_partitions, D, swap_cache,
                                  free_routing=False, E=None, W=0.5, alpha=0.9):
        score = 0
        swap_weight = 1
        swaps, output_perm = partition_candidate.transform_pi(pi, D, swap_cache, free_routing=free_routing)
        if not free_routing:
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
                    if mini_topology:
                        routing_cost = swap_weight * 3 * sum(
                            max(0, D[int(output_perm[qbit_map_inv[P_i[u]]])][int(output_perm[qbit_map_inv[P_i[v]]])] - 1)
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
                        if mini_topology:
                            routing_cost = swap_weight * 3 * sum(
                                max(0, D[int(output_perm[qbit_map_inv[P_i[u]]])][int(output_perm[qbit_map_inv[P_i[v]]])] - 1)
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

