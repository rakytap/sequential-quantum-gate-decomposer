import numpy as np
from typing import List, Tuple, Set, FrozenSet
from itertools import permutations
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
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

def min_cnots_between_permutations(A, B):
    n = len(A)
    inv_B = [0] * n
    for pos, qubit in enumerate(B):
        inv_B[qubit] = pos
    
    P = [inv_B[A[i]] for i in range(n)]
    visited = [False] * n
    total_cnots = 0
    
    for i in range(n):
        if not visited[i]:
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = P[j]
                cycle_len += 1
            if cycle_len >= 2:
                total_cnots += 2 * cycle_len - 3
    
    return total_cnots

def permutation_to_cnot_circuit(A, B):
    n = len(A)
    inv_B = {qubit: pos for pos, qubit in enumerate(B)}
    P = [inv_B[A[i]] for i in range(n)]
    
    visited = [False] * n
    cnot_circuit = Circuit(n)
    
    for i in range(n):
        if not visited[i]:
            # Extract cycle
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j]
            
            # Convert cycle to CNOTs
            k = len(cycle)
            if k == 2:
                cnot_circuit.add_CNOT(cycle[1], cycle[0])
            elif k >= 3:
                # Forward pass
                for idx in range(k - 1):
                    cnot_circuit.add_CNOT(cycle[idx + 1], cycle[idx])
                # Backward pass
                for idx in range(k - 2, 0, -1):
                    cnot_circuit.add_CNOT(cycle[idx + 1], cycle[idx])
    
    return cnot_circuit

def find_best_permutation_with_constraints(A, constraints, strategy='greedy'):
    n = len(A)
    B = [None] * n
    used_qubits = set()
    
    # Apply constraints
    for pos, qubit in constraints.items():
        B[pos] = qubit
        used_qubits.add(qubit)
    
    # Fill unconstrained positions
    available_qubits = [q for q in range(n) if q not in used_qubits]
    unconstrained_positions = [i for i in range(n) if B[i] is None]

    for pos in unconstrained_positions:
        if A[pos] in available_qubits:
            B[pos] = A[pos]
            available_qubits.remove(A[pos])
    
    # Fill remaining positions with remaining qubits
    j = 0
    for pos in unconstrained_positions:
        if B[pos] is None:
            B[pos] = available_qubits[j]
            j += 1
    
    
    return B


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
        # P_o and P_i pairs q*->Q*
        self.permutations_pairs = [[] for _ in range(len(mini_topologies))]
        # results of synthesis
        self.synthesised_circuits = [[] for _ in range(len(mini_topologies))]
        self.synthesised_parameters = [[] for _ in range(len(mini_topologies))]
        self.cnot_counts = [[] for _ in range(len(mini_topologies))]
        self.circuit_structures = [[] for _ in range(len(mini_topologies))]
        # Involved q qubits on the circuit
        self.involved_qbits = involved_qbits
        # q->q*
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
            cnot_count_topology = np.mean(self.cnot_counts[topology_idx])*0.1 + np.min(self.cnot_counts[topology_idx])*0.9
            if len(self.mini_topologies[topology_idx]) == self.N*(self.N-1)/2:
                score += cnot_count_topology*0.3/self.topology_count
            else:
                score += cnot_count_topology*0.7/self.topology_count
        return score 

class PartitionCandidate:
    
    def __init__(self, partition_idx, topology_idx, permutation_idx, circuit_structure, P_i, P_o, topology, mini_topology, qbit_map, involved_qbits):
        #Which partition does this belong to
        self.partition_idx = partition_idx
        #the index of the q* topology
        self.topology_idx = topology_idx
        #the index of the P_i and P_o pair
        self.permutation_idx = permutation_idx
        # the structure of the circuit in q*
        self.circuit_structure = circuit_structure
        # permutations in q*->Q*
        self.P_i = P_i
        self.P_o = P_o
        #The mini_topology in Q
        self.topology = topology
        #The mini topology in q*
        self.mini_topology = mini_topology
        # q->q*
        self.qbit_map = qbit_map
        # q belonging to the original circuit
        self.involved_qbits = involved_qbits
        # q->Q*
        self.node_mapping = get_node_mapping(mini_topology, topology)

    def transform_pi_input(self, pi):
        #Q->q
        qbit_map_swapped = {self.node_mapping[self.P_i.index(v)]: k for k, v in self.qbit_map.items()}
        return find_best_permutation_with_constraints(pi, qbit_map_swapped)

    def transform_pi_output(self, pi):
        #Q->q
        qbit_map_swapped = {self.node_mapping[self.P_o.index(v)]: k for k, v in self.qbit_map.items()}
        return find_best_permutation_with_constraints(pi, qbit_map_swapped)

    def get_final_circuit(self,optimized_partitions,N):
        print(self.node_mapping,self.qbit_map,self.involved_qbits)
        partition = optimized_partitions[self.partition_idx]
        part_parameters = partition.synthesised_parameters[self.topology_idx][self.permutation_idx]
        part_circuit = partition.synthesised_circuits[self.topology_idx][self.permutation_idx]
        qbit_map_swapped = {v : self.node_mapping[self.P_i.index(v)] for k, v in self.qbit_map.items()}
        part_circuit.Remap_Qbits(qbit_map_swapped,N)
        return part_circuit, part_parameters

def check_circuit_compatibility(circuit: Circuit, topology):
    circuit_topology = []
    gates = circuit.get_Gates()
    for gate in gates:
        qubits = gate.get_Involved_Qbits()
        if len(qubits) != 1:
            qubits = tuple(qubits)
            if qubits not in circuit_topology and qubits[::-1] not in circuit_topology:
                circuit_topology.append(qubits)
    for qubits in circuit_topology:
        if qubits not in topology and qubits[::-1] not in topology:
            return False
    return True