import numpy as np
from typing import List, Tuple, Set, FrozenSet
from itertools import permutations
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
import heapq

def _build_adj_list(edges: List[Tuple[int, int]]) -> dict:
    adj_list = {}
    for u, v in edges:
        if u not in adj_list:
            adj_list[u] = set()
        if v not in adj_list:
            adj_list[v] = set()
        adj_list[u].add(v)
        adj_list[v].add(u)
    return adj_list

def _get_induced_edges(edges: List[Tuple[int, int]], qubit_subset: Set[int]) -> List[Tuple[int, int]]:
    return [edge for edge in edges if edge[0] in qubit_subset and edge[1] in qubit_subset]

def _dfs_enumerate(adj_list: dict, k: int, callback):
    all_qubits = sorted(adj_list.keys())
    seen = set()
    def dfs(current_qubits: Set[int], candidates: Set[int]):
        if len(current_qubits) == k:
            frozen = frozenset(current_qubits)
            if frozen not in seen:
                seen.add(frozen)
                callback(current_qubits)
            return
        if len(current_qubits) + len(candidates) < k:
            return
        for node in sorted(candidates):
            new_qubits = current_qubits | {node}
            new_candidates = {neighbor for q in new_qubits for neighbor in adj_list[q] 
                            if neighbor not in new_qubits and neighbor > node}
            dfs(new_qubits, new_candidates)
    for start in all_qubits:
        dfs({start}, {n for n in adj_list[start] if n > start})

def get_canonical_form(qubit_subset: Set[int], induced_edges: List[Tuple[int, int]]) -> FrozenSet[Tuple[int, int]]:
    qubits = sorted(qubit_subset)
    n = len(qubits)
    best_edges = None
    for perm in permutations(range(n)):
        mapping = {qubits[i]: perm[i] for i in range(n)}
        relabeled = tuple(sorted([tuple(sorted([mapping[u], mapping[v]])) for u, v in induced_edges]))
        if best_edges is None or relabeled < best_edges:
            best_edges = relabeled
    return frozenset(best_edges)

def get_unique_subtopologies(edges: List[Tuple[int, int]], k: int) -> List[List[Tuple[int, int]]]:
    if k <= 0:
        return []
    adj_list = _build_adj_list(edges)
    if k == 1:
        return [[]]
    canonical_forms = {}
    def process(qubits):
        induced = _get_induced_edges(edges, qubits)
        canonical = get_canonical_form(qubits, induced)
        if canonical not in canonical_forms:
            canonical_forms[canonical] = induced
    _dfs_enumerate(adj_list, k, process)
    return list(canonical_forms.values())

def get_subtopologies_of_type(edges: List[Tuple[int, int]], target_topology: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    target_qubits = set()
    for u, v in target_topology:
        target_qubits.add(u)
        target_qubits.add(v)
    k = len(target_qubits) if target_qubits else 1
    if k <= 0:
        return []
    adj_list = _build_adj_list(edges)
    if k == 1:
        return [[] for _ in adj_list.keys()]
    target_canonical = get_canonical_form(target_qubits, target_topology)
    matches = []
    def process(qubits):
        induced = _get_induced_edges(edges, qubits)
        canonical = get_canonical_form(qubits, induced)
        if canonical == target_canonical:
            matches.append(induced)
    _dfs_enumerate(adj_list, k, process)
    return matches

def get_node_mapping(topology1: List[Tuple[int, int]], topology2: List[Tuple[int, int]]) -> dict:
    qubits1 = set()
    for u, v in topology1:
        qubits1.add(u)
        qubits1.add(v)
    qubits2 = set()
    for u, v in topology2:
        qubits2.add(u)
        qubits2.add(v)
    if len(qubits1) != len(qubits2):
        return {}
    sorted_qubits1 = sorted(qubits1)
    sorted_qubits2 = sorted(qubits2)
    n = len(sorted_qubits1)
    for perm in permutations(range(n)):
        mapping = {sorted_qubits1[i]: sorted_qubits2[perm[i]] for i in range(n)}
        mapped_edges = set()
        for u, v in topology1:
            mapped_edges.add(tuple(sorted([mapping[u], mapping[v]])))
        original_edges = set(tuple(sorted([u, v])) for u, v in topology2)
        if mapped_edges == original_edges:
            return mapping
    return {}

def find_constrained_swaps_partial(pi_A, pi_B_dict, dist_matrix):
    """
    Find SWAP sequence to route subset of virtual qubits to targets.
    
    Args:
        pi_A: List [Q0, Q1, ...] where pi_A[q] = Q (complete initial mapping)
        pi_B_dict: Dict {q: Q} specifying only qubits that need routing
        dist_matrix: Pre-computed distance matrix dist[i][j] between physical qubits
    
    Returns:
        swaps: List of (i, j) SWAP operations on adjacent physical qubits
        final_permutation: List showing final virtual→physical mapping
    """
    n = len(pi_A)
    
    # Build adjacency list from distance matrix
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i][j] == 1:  # Adjacent in topology
                adj[i].add(j)
                adj[j].add(i)
    
    # Use physical-to-virtual representation for easier SWAP handling
    # state[P] = q means physical qubit P contains virtual qubit q
    def to_phys_to_virt(virt_to_phys):
        """Convert virtual→physical list to physical→virtual list"""
        p2v = [0] * n
        for q in range(n):
            P = virt_to_phys[q]
            p2v[P] = q
        return p2v
    
    def to_virt_to_phys(phys_to_virt):
        """Convert physical→virtual list to virtual→physical list"""
        v2p = [0] * n
        for P in range(n):
            q = phys_to_virt[P]
            v2p[q] = P
        return v2p
    
    start_state = tuple(to_phys_to_virt(pi_A))
    
    def is_goal(state):
        """Check if target qubits are in correct physical positions"""
        for q, target_P in pi_B_dict.items():
            if state[target_P] != q:  # Physical position target_P should contain virtual q
                return False
        return True
    
    def heuristic(state):
        """Lower bound: sum of distances for qubits needing routing"""
        total = 0
        for q, target_P in pi_B_dict.items():
            # Find where virtual qubit q currently is
            current_P = state.index(q)
            total += dist_matrix[current_P][target_P]
        return total // 2  # Optimistic: each SWAP helps 2 qubits
    
    # A* search
    heap = [(heuristic(start_state), 0, start_state, [])]
    visited = {start_state: 0}
    
    while heap:
        f, g, current, path = heapq.heappop(heap)
        
        if is_goal(current):
            # Convert final state back to virtual→physical mapping
            final_permutation = to_virt_to_phys(current)
            return path, final_permutation
        
        if visited.get(current, float('inf')) < g:
            continue
        
        # Try all valid SWAPs on adjacent physical qubits
        current_list = list(current)
        for i in range(n):
            for j in adj[i]:
                if i < j:  # Avoid duplicate (i,j) and (j,i)
                    # SWAP physical qubits i and j
                    new_state = current_list[:]
                    new_state[i], new_state[j] = new_state[j], new_state[i]
                    new_state_tuple = tuple(new_state)
                    
                    new_g = g + 1
                    
                    if visited.get(new_state_tuple, float('inf')) > new_g:
                        visited[new_state_tuple] = new_g
                        new_f = new_g + heuristic(new_state_tuple)
                        new_path = path + [(i, j)]
                        heapq.heappush(heap, (new_f, new_g, new_state_tuple, new_path))
    
    return None, None  # No solution found

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
# Virtual qubits q, reduced virtual qubits (the remapped circuit only up to partition_size) q*
# Physical qubits Q, reduced physical qubits Q* 
class PartitionSynthesisResult:
    
    def __init__(self, N , mini_topologies, involved_qbits, qubit_map, original_circuit):
        #The physical mini_topology of the partition q*
        self.mini_topologies = mini_topologies
        #number of topologies
        self.topology_count = len(mini_topologies)
        #Qubit num of the partition
        self.N = N
        # P_i in q*->Q* permutation pattern: [q*1 q*0 q*2] where q*1 goes to Q* qubit 0 and etc 
        # P_o in Q*->q* permutation pattern [Q*1 Q*0 Q*2] This means that the current output of Q*1 is equal to q*0
        self.permutations_pairs = [[] for _ in range(len(mini_topologies))]
        # results of synthesis
        self.synthesised_circuits = [[] for _ in range(len(mini_topologies))]
        self.synthesised_parameters = [[] for _ in range(len(mini_topologies))]
        self.cnot_counts = [[] for _ in range(len(mini_topologies))]
        self.circuit_structures = [[] for _ in range(len(mini_topologies))]
        # Involved q qubits on the circuit
        self.involved_qbits = involved_qbits
        # {q:q*}
        self.qubit_map = qubit_map
        # the original circuit
        self.original_circuit = original_circuit

    def add_result(self, permutations_pair, synthesised_circuit, synthesised_parameters, topology_idx):
        self.permutations_pairs[topology_idx].append(permutations_pair)
        self.synthesised_circuits[topology_idx].append(synthesised_circuit)
        self.synthesised_parameters[topology_idx].append(synthesised_parameters)
        self.cnot_counts[topology_idx].append(synthesised_circuit.get_Gate_Nums().get('CNOT', 0))
        self.circuit_structures[topology_idx].append(self.extract_circuit_structure(synthesised_circuit))
    
    def extract_circuit_structure(self, circuit):
        circuit_structure = []
        for gate in circuit.get_Gates():
            involved_qbits = gate.get_Involved_Qbits()
            if len(involved_qbits) != 1:
                circuit_structure.append(involved_qbits)
        return circuit_structure

    def get_best_result(self, topology_idx):
        best_index = np.argmin(self.cnot_counts[topology_idx])
        return self.permutations_pairs[topology_idx][best_index], self.synthesised_circuits[topology_idx][best_index], self.synthesised_parameters[topology_idx][best_index]
    
    #get the circuit structure in q 
    def get_original_circuit_structure(self):
        #q*->q
        qbit_map_inverse = {v:k for k,v in self.qubit_map.items()}
        circuit_structure = []
        for gate in self.original_circuit.get_Gates():
            involved_qbits = gate.get_Involved_Qbits()
            if len(involved_qbits) != 1:
                circuit_structure.append((qbit_map_inverse[involved_qbits[0]],qbit_map_inverse[involved_qbits[1]]))
        return circuit_structure
        
    def get_partition_synthesis_score(self):
        score = 0
        for topology_idx in range(self.topology_count):
            cnot_count_topology = np.mean(self.cnot_counts[topology_idx])*0.5 + np.min(self.cnot_counts[topology_idx])*0.5
            score += cnot_count_topology/self.topology_count
        return score

class PartitionCandidate:
    
    def __init__(self, partition_idx, topology_idx, permutation_idx, circuit_structure, P_i, P_o, topology, mini_topology, qbit_map, involved_qbits):
        #Which partition does this belong to
        self.partition_idx = partition_idx
        #the index of the Q* topology
        self.topology_idx = topology_idx
        #the index of the P_i and P_o pair
        self.permutation_idx = permutation_idx
        # the structure of the circuit in Q*
        self.circuit_structure = circuit_structure
        # P_i in q*->Q* permutation pattern: [q*1 q*0 q*2] where q*1 goes to Q* qubit 0 and etc 
        self.P_i = P_i
        # P_o in Q*->q* permutation pattern [Q*1 Q*0 Q*2] This means that the current output of Q*1 is equal to q*0
        self.P_o = P_o
        #The mini_topology in Q
        self.topology = topology
        #The mini topology in Q*
        self.mini_topology = mini_topology
        # {q:q*}
        self.qbit_map = qbit_map
        # q belonging to the original circuit
        self.involved_qbits = involved_qbits
        # {Q*:Q}
        self.node_mapping = get_node_mapping(mini_topology, topology)

    def transform_pi(self, pi, D):
        # Fixed: Use P_i^{-1} instead of P_i for input routing
        # The synthesized circuit S implements: add_Permutation(P_i) -> Original -> add_Permutation(P_o)
        # For Original to see logical qubit q* at partition position q*, we need:
        # - After P_i, position q* should have logical qubit q*'s data
        # - Before P_i (= input to S), position P_i^{-1}[q*] should have logical qubit q*'s data
        # So we route logical qubit k (with qbit_map[k] = q*) to partition position P_i^{-1}[q*]
        P_i_inv = [self.P_i.index(i) for i in range(len(self.P_i))]  # Compute inverse
        qbit_map_input = {k : self.node_mapping[P_i_inv[v]] for k,v in self.qbit_map.items()}
        # Convert pi to plain Python list of ints (may contain np.int64)
        pi_list = [int(x) for x in pi]
        swaps, pi_init = find_constrained_swaps_partial(pi_list, qbit_map_input, D)
        
        pi_output = pi_init.copy()
        # Fixed: P_o should be indexed by partition virtual index q*, not physical index Q*
        # After the circuit, logical qubit k with qbit_map[k] = q* ends up at 
        # physical position node_mapping[P_o[q*]]
        qbit_map_inverse = {v: k for k, v in self.qbit_map.items()}
        for q_star in range(len(self.P_o)):
            if q_star in qbit_map_inverse:
                k = qbit_map_inverse[q_star]
                pi_output[k] = self.node_mapping[self.P_o[q_star]]
        return swaps, pi_output
    
    def get_final_circuit(self,optimized_partitions,N):
        partition = optimized_partitions[self.partition_idx]
        part_parameters = partition.synthesised_parameters[self.topology_idx][self.permutation_idx]
        part_circuit = partition.synthesised_circuits[self.topology_idx][self.permutation_idx]
        part_circuit = part_circuit.Remap_Qbits(self.node_mapping, N)
        return part_circuit, part_parameters

def check_circuit_compatibility(circuit: Circuit, topology):
    circuit_topology = []
    gates = circuit.get_Gates()
    for gate in gates:
        qubits = gate.get_Involved_Qbits()
        if len(qubits) == 1:
            continue
        elif len(qubits) == 2:
            qubits = tuple(qubits)
            if qubits not in circuit_topology and qubits[::-1] not in circuit_topology:
                circuit_topology.append(qubits)
        else:
            gates_new = gate.get_Gates()
            for gate_new in gates_new:
                qubits_new = gate_new.get_Involved_Qbits()
                if len(qubits_new)==1:
                    continue
                qubits_new = tuple(qubits_new)
                if qubits_new not in circuit_topology and qubits_new[::-1] not in circuit_topology:
                    circuit_topology.append(qubits_new)
    for qubits in circuit_topology:
        if qubits not in topology and qubits[::-1] not in topology:
            return False
    return True

def construct_swap_circuit(swap_order, N):
    swap_circ = Circuit(N)
    for swap in swap_order:
        swap_circ.add_CNOT(swap[0],swap[1])
        swap_circ.add_CNOT(swap[1],swap[0])
        swap_circ.add_CNOT(swap[0],swap[1])
    return swap_circ