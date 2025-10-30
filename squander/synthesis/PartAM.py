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


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import get_qubits
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE
from itertools import product

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

def get_all_subtopologies(edges: List[Tuple[int, int]], k: int) -> List[List[Tuple[int, int]]]:
    """
    Find ALL connected subtopologies with exactly k qubits using DFS.
    
    Args:
        edges: List of edges representing the quantum hardware topology
        k: Number of qubits in the desired subtopologies
    
    Returns:
        List of all subtopologies, where each subtopology is a list of edges
    """
    if k <= 0:
        return []
    
    # Build adjacency list
    adj_list = {}
    for u, v in edges:
        if u not in adj_list:
            adj_list[u] = set()
        if v not in adj_list:
            adj_list[v] = set()
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    all_qubits = sorted(adj_list.keys())
    
    if k == 1:
        return [[] for _ in all_qubits]
    
    def get_induced_edges(qubit_subset: Set[int]) -> List[Tuple[int, int]]:
        induced = []
        for edge in edges:
            if edge[0] in qubit_subset and edge[1] in qubit_subset:
                induced.append(edge)
        return induced
    
    subtopologies = []
    seen = set()
    
    def dfs(current_qubits: Set[int], candidates: Set[int]):
        """Enumerate connected subgraphs using DFS."""
        if len(current_qubits) == k:
            frozen = frozenset(current_qubits)
            if frozen not in seen:
                seen.add(frozen)
                subtopologies.append(get_induced_edges(current_qubits))
            return
        
        # Prune if we can't reach k qubits
        if len(current_qubits) + len(candidates) < k:
            return
        
        for node in sorted(candidates):
            # Add node and explore
            new_qubits = current_qubits | {node}
            
            # New candidates: neighbors of new_qubits not yet included
            new_candidates = set()
            for q in new_qubits:
                for neighbor in adj_list[q]:
                    if neighbor not in new_qubits and neighbor > node:
                        new_candidates.add(neighbor)
            
            dfs(new_qubits, new_candidates)
    
    # Start DFS from each qubit
    for start in all_qubits:
        candidates = {n for n in adj_list[start] if n > start}
        dfs({start}, candidates)
    
    return subtopologies


def get_canonical_form(qubit_subset: Set[int], induced_edges: List[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    """
    Convert a subgraph to canonical form for isomorphism checking.
    Relabels nodes as 0,1,2,...,k-1 and returns the lexicographically smallest edge set.
    """
    qubits = sorted(qubit_subset)
    n = len(qubits)
    
    # Try all permutations and find lexicographically smallest
    best_edges = None
    
    for perm in permutations(range(n)):
        # Create mapping: qubits[i] -> perm[i]
        mapping = {qubits[i]: perm[i] for i in range(n)}
        
        # Relabel edges
        relabeled = []
        for u, v in induced_edges:
            new_u, new_v = mapping[u], mapping[v]
            # Normalize edge direction
            relabeled.append(tuple(sorted([new_u, new_v])))
        
        relabeled = tuple(sorted(relabeled))
        
        if best_edges is None or relabeled < best_edges:
            best_edges = relabeled
    
    return frozenset(best_edges)


def get_unique_subtopologies(edges: List[Tuple[int, int]], k: int) -> List[List[Tuple[int, int]]]:
    """
    Find all UNIQUE subtopology structures with k qubits using DFS.
    Returns one example of each non-isomorphic connected subgraph.
    
    Args:
        edges: List of edges representing the quantum hardware topology
        k: Number of qubits in the desired subtopologies
    
    Returns:
        List of unique subtopologies (one representative per isomorphism class)
    """
    if k <= 0:
        return []
    
    # Build adjacency list
    adj_list = {}
    for u, v in edges:
        if u not in adj_list:
            adj_list[u] = set()
        if v not in adj_list:
            adj_list[v] = set()
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    all_qubits = sorted(adj_list.keys())
    
    if k == 1:
        return [[]]  # Single qubit has no edges
    
    def get_induced_edges(qubit_subset: Set[int]) -> List[Tuple[int, int]]:
        induced = []
        for edge in edges:
            if edge[0] in qubit_subset and edge[1] in qubit_subset:
                induced.append(edge)
        return induced
    
    # Track unique canonical forms and their examples
    canonical_forms = {}
    seen = set()
    
    def dfs(current_qubits: Set[int], candidates: Set[int]):
        """Enumerate connected subgraphs using DFS."""
        if len(current_qubits) == k:
            frozen = frozenset(current_qubits)
            if frozen not in seen:
                seen.add(frozen)
                induced = get_induced_edges(current_qubits)
                
                # Get canonical form
                canonical = get_canonical_form(current_qubits, induced)
                
                # Store first example of each canonical form
                if canonical not in canonical_forms:
                    canonical_forms[canonical] = induced
            return
        
        # Prune if we can't reach k qubits
        if len(current_qubits) + len(candidates) < k:
            return
        
        for node in sorted(candidates):
            # Add node and explore
            new_qubits = current_qubits | {node}
            
            # New candidates: neighbors of new_qubits not yet included
            new_candidates = set()
            for q in new_qubits:
                for neighbor in adj_list[q]:
                    if neighbor not in new_qubits and neighbor > node:
                        new_candidates.add(neighbor)
            
            dfs(new_qubits, new_candidates)
    
    # Start DFS from each qubit
    for start in all_qubits:
        candidates = {n for n in adj_list[start] if n > start}
        dfs({start}, candidates)
    
    return list(canonical_forms.values())


def extract_subtopology(involved_qbits, qbit_map, config ):
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology

class PartitionSynthesisResult:
    def __init__(self, N , mini_topologies):
        self.mini_topologies = mini_topologies
        self.topology_count = len(mini_topologies)
        self.N = N
        self.permutations_pairs = [[] for _ in range(len(mini_topologies))]
        self.synthesised_circuits = [[] for _ in range(len(mini_topologies))]
        self.synthesised_parameters = [[] for _ in range(len(mini_topologies))]
        self.cnot_counts = [[] for _ in range(len(mini_topologies))]
    
    def add_result(self, permutations_pair, synthesised_circuit, synthesised_parameters, topology_idx):
        self.permutations_pairs[topology_idx].append(permutations_pair)
        self.synthesised_circuits[topology_idx].append(synthesised_circuit)
        self.synthesised_parameters[topology_idx].append(synthesised_parameters)
        self.cnot_counts[topology_idx].append(synthesised_circuit.get_Gate_Nums().get('CNOT', 0))
    
    def get_best_result(self, topology_idx):
        best_index = np.argmin(self.cnot_counts[topology_idx])
        return self.permutations_pairs[topology_idx][best_index], self.synthesised_circuits[topology_idx][best_index], self.synthesised_parameters[topology_idx][best_index]
    
    def get_partition_synthesis_score(self):
        score = 0
        for topology_idx in range(self.topology_count):
            cnot_count_topology = np.mean(self.cnot_counts[topology_idx])*0.1 + np.min(self.cnot_counts[topology_idx])*0.9
            if len(self.mini_topologies[topology_idx]) == self.N*(self.N-1)/2:
                score += cnot_count_topology*0.3/self.topology_count
            else:
                score += cnot_count_topology*0.7/self.topology_count
        return score 

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
        parallel = self.config['parallel']
        allowed_parallel = [0, 1, 2]
        if not parallel in allowed_parallel:
            raise Exception(f"The parallel configuration should be either of {allowed_parallel}, got {parallel}.")
        verbosity = self.config['verbosity']
        if not isinstance(verbosity, int):
            raise Exception(f"The verbosity parameter should be an integer.")

        self.max_partition_size = self.config['max_partition_size']
        if not isinstance(self.max_partition_size, int):
            raise Exception(f"The max_partition_size parameter should be an integer.")
        self.topology = self.config['topology']
        if not isinstance(self.topology, list):
            raise Exception(f"The topology parameter should be a list.")
        self.routed = self.config['routed']
        if not isinstance(self.routed, bool):
            raise Exception(f"The routed parameter should be a bool.")
        self.partition_strategy = self.config['partition_strategy']
        allowed_partition_strategies = ['ilp', 'tdag', 'kahn', 'qiskit', 'qiskit-fusion', 'bqskit-Quick', 'bqskit-Scan', 'bqskit-Greedy', 'bqskit-Cluster']

    @staticmethod
    def DecomposePartition_Sequential(Partition_circuit: Circuit, Partition_parameters: np.ndarray, config: dict, topologies) -> PartitionSynthesisResult:
        """
        Call to decompose a partition sequentially
        """
        N = Partition_circuit.get_Qbit_Num()
        perumations_all = list(permutations(range(N)))
        result = PartitionSynthesisResult(N, topologies)
        # Sequential permutation search
        for topology_idx in range(len(topologies)):
            mini_topology = topologies[topology_idx]
            P_o_initial = perumations_all[np.random.choice(range(len(perumations_all)))]
            for P_i in perumations_all:
                Partition_circuit_tmp = Circuit(N)
                Partition_circuit_tmp.add_Permutation(P_i)
                Partition_circuit_tmp.add_Circuit(Partition_circuit)
                Partition_circuit_tmp.add_Permutation(P_o_initial)
                synthesised_circuit, synthesised_parameters = DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                result.add_result((P_i, P_o_initial), synthesised_circuit, synthesised_parameters, topology_idx)

            P_i_best, _ = result.get_best_result(topology_idx)[0]
            for P_o in perumations_all:
                Partition_circuit_tmp = Circuit(N)
                Partition_circuit_tmp.add_Permutation(P_i_best)
                Partition_circuit_tmp.add_Circuit(Partition_circuit)
                Partition_circuit_tmp.add_Permutation(P_o)
                synthesised_circuit, synthesised_parameters = DecomposePartition_and_Perm(Partition_circuit_tmp.get_Matrix(Partition_parameters), config, mini_topology)
                result.add_result((P_i_best, P_o), synthesised_circuit, synthesised_parameters, topology_idx)
        return result

    def SynthesizeWideCircuit(self, circ, orig_parameters):
        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = get_all_partitions(circ, self.max_partition_size)
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
            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="Synthesizing partitions") ):

                start_idx = subcircuit.get_Parameter_Start_Index()
                end_idx   = subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
                subcircuit_parameters = parameters[ start_idx:end_idx ]
                k = subcircuit.get_Qbit_Num()
                mini_topologies = get_unique_subtopologies(self.topology, k)
                optimized_results[partition_idx] = pool.apply_async( self.DecomposePartition_Sequential, (subcircuit, subcircuit_parameters, self.config, mini_topologies) )

            for partition_idx, subcircuit in enumerate( tqdm(subcircuits, desc="Processing partitions") ):
                optimized_results[partition_idx] = optimized_results[partition_idx].get()

        weights = [result.get_partition_synthesis_score() for result in optimized_results[:len(allparts)]]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        # Create a mapping from partition frozensets to their indices in allparts
        partition_to_idx = {frozenset(part): i for i, part in enumerate(allparts)}

        # Convert the returned partitions to indices
        L_indices = [partition_to_idx[frozenset(part)] for part in L_parts]

        # Now directly select the already-optimized subcircuits using the indices
        selected_optimized_subcircuits = [optimized_subcircuits[i] for i in L_indices]

  max_gates = max(len(c.get_Gates()) for c in optimized_subcircuits)
  def to_cost(d): return d.get('CNOT', 0)*max_gates + sum(d[x] for x in d if x != 'CNOT')
  weights = [to_cost(circ.get_Gate_Nums()) for circ in optimized_subcircuits[:len(allparts)]]

  # ilp_global_optimal returns the selected partitions as frozensets
  L_parts, fusion_info = ilp_global_optimal(allparts, g, weights=weights)

  # Create a mapping from partition frozensets to their indices in allparts
  partition_to_idx = {frozenset(part): i for i, part in enumerate(allparts)}

  # Convert the returned partitions to indices
  L_indices = [partition_to_idx[frozenset(part)] for part in L_parts]

  # Now directly select the already-optimized subcircuits using the indices
  selected_optimized_subcircuits = [optimized_subcircuits[i] for i in L_indices]
  selected_parameters = [optimized_parameter_list[i] for i in L_indices]

  # Construct the final circuit from the optimized subcircuits
  wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions(
      selected_optimized_subcircuits,
      selected_parameters
  )
