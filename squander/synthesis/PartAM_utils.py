import numpy as np
from typing import List, Tuple, Set, FrozenSet
from itertools import permutations

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

class SingleQubitPartitionResult:
    def __init__(self,circuit_in,parameters_in):
        self.circuit = circuit_in
        self.parameters = parameters_in
    def get_partition_synthesis_score(self):
        return 0

class PartitionSynthesisResult:
    def __init__(self, N , mini_topologies, involved_qbits, qubit_map):
        self.mini_topologies = mini_topologies
        self.topology_count = len(mini_topologies)
        self.N = N
        self.permutations_pairs = [[] for _ in range(len(mini_topologies))]
        self.synthesised_circuits = [[] for _ in range(len(mini_topologies))]
        self.synthesised_parameters = [[] for _ in range(len(mini_topologies))]
        self.cnot_counts = [[] for _ in range(len(mini_topologies))]
        self.involved_qbits = involved_qbits
        self.qubit_map = qubit_map
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