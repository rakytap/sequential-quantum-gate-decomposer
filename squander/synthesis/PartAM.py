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

from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import get_qubits
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE
from itertools import product
from squander.synthesis.PartAM_utils import get_all_subtopologies, get_unique_subtopologies, SingleQubitPartitionResult, PartitionSynthesisResult

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

    @staticmethod
    def DecomposePartition_Sequential(Partition_circuit: Circuit, Partition_parameters: np.ndarray, config: dict, topologies, involved_qbits, qbit_map) -> PartitionSynthesisResult:
        """
        Call to decompose a partition sequentially
        """
        N = Partition_circuit.get_Qbit_Num()
        if N !=1:
            perumations_all = list(permutations(range(N)))
            result = PartitionSynthesisResult(N, topologies, involved_qbits, qbit_map)
            # Sequential permutation search
            for topology_idx in range(len(topologies)):
                mini_topology = topologies[topology_idx]
                P_o_initial = perumations_all[np.random.choice(range(len(perumations_all)))]
                for P_i in perumations_all:
                    Partition_circuit_tmp = Circuit(N)
                    Partition_circuit_tmp.add_Permutation(P_i)
                    Partition_circuit_tmp.add_Circuit(Partition_circuit)
                    Partition_circuit_tmp.add_Permutation(P_o_initial)
                    synthesised_circuit, synthesised_parameters = qgd_Partition_Aware_Mapping.DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                    result.add_result((P_i, P_o_initial), synthesised_circuit, synthesised_parameters, topology_idx)

                P_i_best, _ = result.get_best_result(topology_idx)[0]
                for P_o in perumations_all:
                    Partition_circuit_tmp = Circuit(N)
                    Partition_circuit_tmp.add_Permutation(P_i_best)
                    Partition_circuit_tmp.add_Circuit(Partition_circuit)
                    Partition_circuit_tmp.add_Permutation(P_o)
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
                k = subcircuit.get_Qbit_Num()
                qbit_num_orig_circuit = subcircuit.get_Qbit_Num()
                involved_qbits = subcircuit.get_Qbits()

                qbit_num = len( involved_qbits )
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num)
                qbit_map = {}
                for idx in range( len(involved_qbits) ):
                    qbit_map[ involved_qbits[idx] ] = idx
                remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )
                optimized_results[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="First Synthesis") ):
                optimized_results[partition_idx] = optimized_results[partition_idx].get()

        weights = [result.get_partition_synthesis_score() for result in optimized_results[:len(allparts)]]
        L_parts, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        parts = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, [allparts[i] for i in L_parts], fusion_info)
        L = topo_sort_partitions(circ, self.max_partition_size, parts)
        from squander.partitioning.kahn import kahn_partition_preparts
        from squander.partitioning.tools import translate_param_order
        partitioned_circuit, param_order, _ = kahn_partition_preparts(circ, self.max_partition_size, [parts[i] for i in L])
        parameters = translate_param_order(orig_parameters, param_order)

        subcircuits = partitioned_circuit.get_Gates()

        # the list of optimized subcircuits
        optimized_results = [None] * len(subcircuits)

        with Pool(processes=mp.cpu_count()) as pool:
            for partition_idx, subcircuit in enumerate( subcircuits ):

                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
                k = subcircuit.get_Qbit_Num()
                qbit_num_orig_circuit = subcircuit.get_Qbit_Num()
                involved_qbits = subcircuit.get_Qbits()

                qbit_num = len( involved_qbits )
                mini_topologies = get_unique_subtopologies(self.topology, qbit_num)
                qbit_map = {}
                for idx in range( len(involved_qbits) ):
                    qbit_map[ involved_qbits[idx] ] = idx
                remapped_subcircuit = subcircuit.Remap_Qbits( qbit_map, qbit_num )
                optimized_results[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (remapped_subcircuit, subcircuit_parameters, self.config, mini_topologies, involved_qbits, qbit_map) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="Second Synthesis") ):
                optimized_results[partition_idx] = optimized_results[partition_idx].get()
        return optimized_results

    def Partition_Aware_Mapping(self, circ: Circuit, orig_parameters: np.ndarray):
        optimized_partitions, preparation_parts = self.SynthesizeWideCircuit(circ, orig_parameters)
        print(preparation_parts)
        DAG, IDAG = self.construct_DAG_and_IDAG(optimized_partitions)
        D = self.compute_distances_bfs(circ.get_Qbit_Num())
        pi = self._compute_smart_initial_layout(circ, circ.get_Qbit_Num(), D)

    def get_initial_layer(self, IDAG, N):
        initial_layer = []
        active_qbits = list(range(N))
        for idx in range(len(IDAG)):
            if len(IDAG[idx][1]) == 0:
                initial_layer.append(idx)
                for qbit in IDAG[idx][0].involved_qbits:
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
            DAG.append([idx, children])
            IDAG.append([idx, parents])
        return DAG, IDAG

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
        
        return D*3 #multiply by 3 to make it CNOT cost instead of SWAP cost

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
