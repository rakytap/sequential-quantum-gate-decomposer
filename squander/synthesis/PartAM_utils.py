import numpy as np
from typing import List, Tuple, Set, FrozenSet
from itertools import permutations, combinations
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
import heapq
import math
import logging
import pulp
from collections import defaultdict


# ============================================================================
# SWAP Routing Algorithms
# ============================================================================
def find_constrained_swaps_A_star(pi_A, pi_B_dict, dist_matrix):
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
        total = 0.0
        for q, target_P in pi_B_dict.items():
            # Find where virtual qubit q currently is
            current_P = state.index(q)
            distance = dist_matrix[current_P][target_P]
            if np.isinf(distance):
                logging.warning(
                    "Encountered unreachable qubit pair (%s, %s) in routing heuristic; returning inf cost.",
                    current_P,
                    target_P,
                )
                return math.inf
            total += float(distance)
        return math.floor(total / 2)  # Optimistic: each SWAP helps 2 qubits
    
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

def find_constrained_swaps_partial(pi_A, pi_B_dict, dist_matrix):
    """
    Route partition qubits to their target physical positions using A* over
    the k-dimensional state space of partition qubit positions only.

    For k partition qubits on an n-node topology the state space has at most
    n^k entries (n*(n-1)*...*(n-k+1) distinct states).  For the typical case
    of k=2 or k=3 and n≤20 this is tiny (≤2744 states) so the search
    completes in microseconds while still finding an optimal SWAP sequence.

    The original full-state A* had O(n!) state space which was exponentially
    slow.  The naive greedy replacement oscillated when two adjacent partition
    qubits needed to move in the same direction.  This implementation avoids
    both problems.

    Args:
        pi_A        : List[int], pi_A[q] = current physical position of virtual qubit q.
        pi_B_dict   : Dict {q: target_physical} for the qubits that need routing.
        dist_matrix : n×n distance/cost matrix; dist[i][j]==1 means i and j are adjacent.

    Returns:
        swaps            : List of (P1, P2) adjacent-qubit SWAP operations (optimal).
        final_permutation: Updated virtual→physical mapping after all SWAPs.
    """
    n = len(pi_A)

    # Build adjacency list from dist_matrix
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] == 1:
                adj[i].append(j)
                adj[j].append(i)

    partition_qubits = sorted(pi_B_dict.keys())
    k = len(partition_qubits)

    initial_positions = tuple(int(pi_A[q]) for q in partition_qubits)
    target_positions  = tuple(int(pi_B_dict[q]) for q in partition_qubits)

    if initial_positions == target_positions:
        return [], list(pi_A)

    def heuristic(positions):
        # Admissible lower bound: sum of individual distances / 2
        return sum(dist_matrix[positions[i]][target_positions[i]] for i in range(k)) / 2

    # A* over k-dimensional state space.
    # Each state is a tuple of physical positions, one per partition qubit.
    # Paths are reconstructed via a parent-pointer dict to avoid copying lists
    # on every heap push (which would be O(depth²) total).
    counter = 0  # tiebreak counter so tuples never compare paths
    parent = {}  # state → (parent_state, swap) for path reconstruction
    parent[initial_positions] = None

    heap = []
    heapq.heappush(heap, (heuristic(initial_positions), 0, counter, initial_positions))
    visited = {initial_positions: 0}

    while heap:
        f, g, _, positions = heapq.heappop(heap)

        if positions == target_positions:
            # Reconstruct swap path via parent pointers
            path = []
            state = positions
            while parent[state] is not None:
                prev_state, swap = parent[state]
                path.append(swap)
                state = prev_state
            path.reverse()

            # Replay swaps on the full mapping to get final virt→phys
            final_v2p = list(pi_A)
            final_p2v = [0] * n
            for q_idx in range(n):
                final_p2v[int(final_v2p[q_idx])] = q_idx
            for P1, P2 in path:
                q1, q2 = final_p2v[P1], final_p2v[P2]
                final_p2v[P1], final_p2v[P2] = q2, q1
                final_v2p[q1], final_v2p[q2] = P2, P1
            return path, final_v2p

        if visited.get(positions, float('inf')) < g:
            continue

        # Quick lookup: physical position → index within partition_qubits list
        pos_to_k_idx = {p: i for i, p in enumerate(positions)}

        # Expand: try every SWAP that moves at least one partition qubit
        for i, p in enumerate(positions):
            for nb in adj[p]:
                new_positions = list(positions)
                new_positions[i] = nb
                # If the neighbor also holds a partition qubit, swap it too
                if nb in pos_to_k_idx:
                    j = pos_to_k_idx[nb]
                    new_positions[j] = p
                new_positions = tuple(new_positions)

                new_g = g + 1
                if visited.get(new_positions, float('inf')) <= new_g:
                    continue

                visited[new_positions] = new_g
                swap_key = (min(p, nb), max(p, nb))
                parent[new_positions] = (positions, swap_key)
                counter += 1
                heapq.heappush(heap, (new_g + heuristic(new_positions), new_g,
                                      counter, new_positions))

    logging.warning(
        "find_constrained_swaps_partial: failed to route %s → %s",
        initial_positions, target_positions,
    )
    return [], list(pi_A)

def calculate_swaps_quick(P_i, qbit_map, node_mapping, pi, D, swap_cache=None):
    P_i_inv = [P_i.index(i) for i in range(len(P_i))]  # Compute inverse
    qbit_map_input = {k : node_mapping[P_i_inv[v]] for k,v in qbit_map.items()}
    # Convert pi to plain Python list of ints (may contain np.int64)
    pi_list = [int(x) for x in pi]

    # Check cache if provided
    cache_key = None
    if swap_cache is not None:
        # Create cache key: (pi_tuple, frozenset of qbit_map_input items)
        pi_tuple = tuple(pi_list)
        qbit_map_frozen = frozenset(qbit_map_input.items())
        cache_key = (pi_tuple, qbit_map_frozen)
        if cache_key in swap_cache:
            swaps, pi_init = swap_cache[cache_key]
        else:
            swaps, pi_init = find_constrained_swaps_partial(pi_list, qbit_map_input, D)
            swap_cache[cache_key] = (swaps, pi_init)
    else:
        swaps, pi_init = find_constrained_swaps_partial(pi_list, qbit_map_input, D)
    return len(swaps)


# ============================================================================
# Topology Utilities
# ============================================================================

def _get_induced_edges(edges: List[Tuple[int, int]], qubit_subset: Set[int]) -> List[Tuple[int, int]]:
    return [edge for edge in edges if edge[0] in qubit_subset and edge[1] in qubit_subset]

def _is_connected(nodes: Set[int], edges: List[Tuple[int, int]]) -> bool:
    if len(nodes) <= 1:
        return True
    adj = defaultdict(set)
    for u, v in edges:
        if u in nodes and v in nodes:
            adj[u].add(v)
            adj[v].add(u)
    start = next(iter(nodes))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited == nodes

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
    """Return one representative locally-labeled (0..k-1) edge list per unique k-node
    connected subgraph isomorphism class found in the graph defined by *edges*."""
    if k <= 0:
        return []
    if k == 1:
        return [[]]
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    nodes = sorted(nodes)
    if len(nodes) < k:
        return []
    canonical_forms = {}
    for subset in combinations(nodes, k):
        subset_set = set(subset)
        induced = _get_induced_edges(edges, subset_set)
        if not _is_connected(subset_set, induced):
            continue
        canonical = get_canonical_form(subset_set, induced)
        if canonical not in canonical_forms:
            # Store locally-labeled edges (0..k-1) so the decomposer always
            # receives a valid k-qubit topology regardless of global qubit indices.
            canonical_forms[canonical] = sorted(canonical)
    return list(canonical_forms.values())

def get_subtopologies_of_type(edges: List[Tuple[int, int]], target_topology: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Return all connected k-node subgraphs of *edges* that are isomorphic to
    *target_topology*, each expressed with the original global qubit labels
    (needed for physical routing decisions)."""
    target_qubits = set()
    for u, v in target_topology:
        target_qubits.add(u)
        target_qubits.add(v)
    k = len(target_qubits) if target_qubits else 1
    if k <= 0:
        return []
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    if k == 1:
        return [[] for _ in nodes]
    nodes = sorted(nodes)
    if len(nodes) < k:
        return []
    target_canonical = get_canonical_form(target_qubits, target_topology)
    matches = []
    for subset in combinations(nodes, k):
        subset_set = set(subset)
        induced = _get_induced_edges(edges, subset_set)
        if not _is_connected(subset_set, induced):
            continue
        canonical = get_canonical_form(subset_set, induced)
        if canonical == target_canonical:
            matches.append(induced)  # global labels retained for routing
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

def extract_subtopology(involved_qbits, qbit_map, config ):
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology


# ============================================================================
# Distance & Cost Calculations
# ============================================================================

def calculate_dist_small(mini_topology, qbit_map, dist_matrix, pi):
    """Estimate the routing cost needed to bring the partition qubits adjacent.

    Minimises over all assignments of circuit qubits to local topology
    positions so that, e.g., the hub qubit in a 3-qubit star topology is
    always matched to the physically most central circuit qubit rather than
    whichever qubit happened to be assigned local label 0 by the ILP.

    Returns sum-of-edge-distances * 3 (three gates per SWAP) for the best
    qubit-to-position assignment.
    """
    if not mini_topology:
        return 0
    # Build ordered list: circuit_qubits[j] = circuit qubit at local position j
    k = len(qbit_map)
    qbit_map_inv = {local: circ for circ, local in qbit_map.items()}
    circuit_qubits = [qbit_map_inv[j] for j in range(k)]

    # Try all k! permutations to find the cheapest qubit→position assignment
    best = float('inf')
    for perm in permutations(range(k)):
        cost = 0
        for u, v in mini_topology:
            cost += (dist_matrix[pi[circuit_qubits[perm[u]]]][pi[circuit_qubits[perm[v]]]] - 1) * 3
        if cost < best:
            best = cost
    return best

# ============================================================================
# Data Classes
# ============================================================================

class SingleQubitPartitionResult:
    
    def __init__(self,circuit_in,parameters_in):
        self.circuit = circuit_in
        self.parameters = parameters_in
        self.involved_qbits = circuit_in.get_Qbits()
    
    def get_partition_synthesis_score(self):
        return 0

# Virtual qubits q, reduced virtual qubits (the remapped circuit only up to partition_size) q*
# Physical qubits Q, reduced physical qubits Q* 
class PartitionSynthesisResult:
    
    def __init__(self, N , mini_topologies, involved_qbits, qubit_map, original_circuit, topology=None, topology_cache=None):
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
        # Pre-computed topology candidates for each mini_topology (lazy initialization)
        self._topology_candidates = [None] * len(mini_topologies)
        self._topology = topology  # Full topology for computing candidates
        self._topology_cache = topology_cache  # Cache to use for lookups

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
        score = np.inf
        for topology_idx in range(self.topology_count):
            cnot_count_topology = np.min(self.cnot_counts[topology_idx])#np.mean(self.cnot_counts[topology_idx])*0.5 + np.min(self.cnot_counts[topology_idx])*0.5
            score = min(cnot_count_topology,score)
        return score
    
    def get_topology_candidates(self, topology_idx):
        """
        Get topology candidates for a given topology index, using cache if available.
        """
        if self._topology_candidates[topology_idx] is None:
            mini_topology = self.mini_topologies[topology_idx]
            if self._topology_cache is not None:
                # Use cached version if available
                target_qubits = set()
                for u, v in mini_topology:
                    target_qubits.add(u)
                    target_qubits.add(v)
                if target_qubits:
                    canonical_key = get_canonical_form(target_qubits, mini_topology)
                    if canonical_key in self._topology_cache:
                        self._topology_candidates[topology_idx] = self._topology_cache[canonical_key]
                    else:
                        # Compute and cache
                        if self._topology is not None:
                            candidates = get_subtopologies_of_type(self._topology, mini_topology)
                            self._topology_cache[canonical_key] = candidates
                            self._topology_candidates[topology_idx] = candidates
                        else:
                            self._topology_candidates[topology_idx] = []
                else:
                    self._topology_candidates[topology_idx] = []
            else:
                # No cache, compute directly
                if self._topology is not None:
                    self._topology_candidates[topology_idx] = get_subtopologies_of_type(self._topology, mini_topology)
                else:
                    self._topology_candidates[topology_idx] = []
        return self._topology_candidates[topology_idx]



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

    def transform_pi(self, pi, D, swap_cache=None, free_routing=False):
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
        n = len(pi_list)

        if free_routing:
            # Routing is free: build the ideal pi_init that places partition qubits
            # at their target physical positions with zero SWAPs, then assign the
            # remaining virtual qubits to the remaining physical positions.
            used_physical = set(qbit_map_input.values())
            pi_init = [0] * n
            for k, target_P in qbit_map_input.items():
                pi_init[k] = target_P
            remaining_physical = sorted(p for p in range(n) if p not in used_physical)
            remaining_logical  = sorted(q for q in range(n) if q not in qbit_map_input)
            for q, p in zip(remaining_logical, remaining_physical):
                pi_init[q] = p
            swaps = []
        else:
            # Check cache if provided
            if swap_cache is not None:
                pi_tuple = tuple(pi_list)
                qbit_map_frozen = frozenset(qbit_map_input.items())
                cache_key = (pi_tuple, qbit_map_frozen)
                if cache_key in swap_cache:
                    swaps, pi_init = swap_cache[cache_key]
                else:
                    swaps, pi_init = find_constrained_swaps_partial(pi_list, qbit_map_input, D)
                    swap_cache[cache_key] = (swaps, pi_init)
            else:
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
    
    def estimate_swap_count(self, pi, D) -> int:
        """O(n) lower-bound on the number of SWAPs needed to route this
        partition's virtual qubits to their target physical positions.
        Uses the same admissible heuristic as the A* search internaly:
            floor(sum_of_distances / 2)
        """
        P_i_inv = [self.P_i.index(i) for i in range(len(self.P_i))]
        total = 0.0
        for k, v in self.qbit_map.items():
            target_P = self.node_mapping[P_i_inv[v]]
            current_P = int(pi[k])
            d = D[current_P][target_P]
            if not np.isinf(d):
                total += d
        return int(total / 2)

    def get_final_circuit(self,optimized_partitions,N):
        partition = optimized_partitions[self.partition_idx]
        part_parameters = partition.synthesised_parameters[self.topology_idx][self.permutation_idx]
        part_circuit = partition.synthesised_circuits[self.topology_idx][self.permutation_idx]
        part_circuit = part_circuit.Remap_Qbits(self.node_mapping, N)
        return part_circuit, part_parameters


# ============================================================================
# Circuit Utilities
# ============================================================================

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

def group_into_two_qubit_blocks(circuit: Circuit) -> Circuit:
    """
    Takes a flat circuit and returns an equivalent circuit whose top-level
    elements are all 2-qubit Circuit blocks, each containing exactly one
    2-qubit gate.

    Single-qubit gates are buffered and flushed into the next 2-qubit block
    on that qubit. Trailing single-qubit gates (after the last 2-qubit gate
    on a qubit) are appended to the last block that involved that qubit.

    Assumes the circuit contains only 1- and 2-qubit gates.

    Args:
        circuit: Flat input circuit with individual gates

    Returns:
        Circuit: Equivalent circuit whose top-level elements are all 2-qubit blocks
    """
    N = circuit.get_Qbit_Num()

    pending = defaultdict(list)  # pending[q] = single-qubit gates waiting for next block on q
    blocks = []                  # accumulated Circuit block objects
    last_block_for_qubit = {}    # last_block_for_qubit[q] = index into blocks

    for gate in circuit.get_Gates():
        qubits = gate.get_Involved_Qbits()
        if len(qubits) == 1:
            pending[qubits[0]].append(gate)
        else:  # 2-qubit gate
            q0, q1 = qubits[0], qubits[1]
            block = Circuit(N)
            for g in pending[q0]:
                block.add_Gate(g)
            for g in pending[q1]:
                block.add_Gate(g)
            pending[q0].clear()
            pending[q1].clear()
            block.add_Gate(gate)
            idx = len(blocks)
            blocks.append(block)
            last_block_for_qubit[q0] = idx
            last_block_for_qubit[q1] = idx

    # Append trailing single-qubit gates to the last block that touched that qubit
    for q, gates_list in pending.items():
        if not gates_list:
            continue
        if q in last_block_for_qubit:
            block = blocks[last_block_for_qubit[q]]
            for g in gates_list:
                block.add_Gate(g)
        else:
            # Qubit only has single-qubit gates — create a standalone block
            block = Circuit(N)
            for g in gates_list:
                block.add_Gate(g)
            blocks.append(block)

    result = Circuit(N)
    for block in blocks:
        result.add_Circuit(block)
    return result
