import numpy as np
from typing import List, Tuple, Set, FrozenSet
from itertools import permutations
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
import heapq
import math
import logging
import pulp
from collections import defaultdict

def solve_min_swaps(perm, edges, T=None, use_gurobi=True):
    """
    Compute globally optimal minimum SWAPs to route permutation 'perm'
    to identity under connectivity 'edges'.

    perm[i] = logical qubit currently at physical node i.
    edges   = list of undirected edges (u,v).
    T       = time horizon (number of layers); if None, use n^2.
    """

    n = len(perm)
    nodes = list(range(n))
    tokens = list(range(n))

    # Time horizon
    if T is None:
        T = n * n   # safe-ish upper bound for small n

    # Undirected edges => directed arcs for movement variables
    undirected = {tuple(sorted(e)) for e in edges}
    neighbors = {u: set() for u in nodes}
    for u, v in undirected:
        neighbors[u].add(v)
        neighbors[v].add(u)
    directed_arcs = [(u, v) for u, v in undirected for (u, v) in ((u, v), (v, u))]

    # ILP model
    prob = pulp.LpProblem("TokenSwapping", pulp.LpMinimize)

    # x[t][v][q]: token q at node v at time t
    x = {
        (t, v, q): pulp.LpVariable(f"x_t{t}_v{v}_q{q}", cat="Binary")
        for t in range(T + 1)
        for v in nodes
        for q in tokens
    }

    # m[t][u][v][q]: token q moves from u to v between t and t+1
    m = {
        (t, u, v, q): pulp.LpVariable(f"m_t{t}_{u}_{v}_q{q}", cat="Binary")
        for t in range(T)
        for (u, v) in directed_arcs
        for q in tokens
    }

    # Initial positions: tokens are at positions given by 'perm'
    # perm[i] = token currently at physical node i
    for v in nodes:
        for q in tokens:
            prob += x[(0, v, q)] == (1 if v == q else 0), f"init_t0_v{v}_q{q}"

    # Final positions: identity mapping (token q at node q)
    for v in nodes:
        for q in tokens:
            prob += x[(T, v, q)] == (1 if perm[v] == q else 0), f"final_tT_v{v}_q{q}"

    # Each token at exactly one node at each time
    for t in range(T + 1):
        for q in tokens:
            prob += (
                pulp.lpSum(x[(t, v, q)] for v in nodes) == 1,
                f"one_node_t{t}_q{q}",
            )

    # Each node holds exactly one token at each time
    for t in range(T + 1):
        for v in nodes:
            prob += (
                pulp.lpSum(x[(t, v, q)] for q in tokens) == 1,
                f"one_token_t{t}_v{v}",
            )

    # Introduce swap decision per time over undirected edges (single swap per time)
    y = {
        (t, u, v): pulp.LpVariable(f"y_t{t}_e{u}_{v}", cat="Binary")
        for t in range(T)
        for (u, v) in undirected
    }

    # At most one swap per time step
    for t in range(T):
        prob += (
            pulp.lpSum(y[(t, u, v)] for (u, v) in undirected) <= 1,
            f"one_swap_per_time_t{t}",
        )

    # Flow constraints for token movement
    for t in range(T):
        for u in nodes:
            for q in tokens:
                outbound = pulp.lpSum(
                    m[(t, u, v, q)] for v in neighbors[u]
                )
                inbound = pulp.lpSum(
                    m[(t, v, u, q)] for v in neighbors[u]
                )
                prob += (
                    x[(t, u, q)] == x[(t + 1, u, q)] + outbound - inbound,
                    f"flow_t{t}_u{u}_q{q}",
                )

    # Link moves to selected swap edge and enforce swap semantics
    for t in range(T):
        for (u, v) in undirected:
            # Only allow moves along (u,v) at time t if this edge is selected
            for q in tokens:
                prob += m[(t, u, v, q)] <= y[(t, u, v)], f"link_m_y_t{t}_{u}_{v}_q{q}_uv"
                prob += m[(t, v, u, q)] <= y[(t, u, v)], f"link_m_y_t{t}_{u}_{v}_q{q}_vu"

            # If edge selected, exactly one token moves each direction (a swap)
            prob += (
                pulp.lpSum(m[(t, u, v, q)] for q in tokens) == y[(t, u, v)],
                f"one_token_uv_t{t}_{u}_{v}",
            )
            prob += (
                pulp.lpSum(m[(t, v, u, q)] for q in tokens) == y[(t, u, v)],
                f"one_token_vu_t{t}_{u}_{v}",
            )

    # Objective: minimize number of swaps (sum of y)
    total_swaps = pulp.lpSum(y.values())
    total_moves = pulp.lpSum(m.values())  # optional, for reporting
    prob += total_swaps

    # Choose solver
    if use_gurobi:
        try:
            solver = pulp.GUROBI(msg=False, manageEnv=True, Threads=1)
        except Exception:
            # Fallback if GUROBI not properly installed with PuLP wrapper
            solver = pulp.PULP_CBC_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)

    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"Solver did not find optimal solution, status = {status}")

    # Extract move/swap counts
    moves_value = int(pulp.value(total_moves))
    swap_value = int(pulp.value(total_swaps))

    # Build per-time-step SWAP schedule:
    # For each t, look at directed moves and turn them into undirected edges.
    swap_layers = []
    for t in range(T):
        layer = []
        for (u, v) in undirected:
            if int(pulp.value(y[(t, u, v)])) == 1:
                layer.append((u, v))
        # At most one edge per layer by construction
        if layer: swap_layers.append(layer)

    return {
        "swap_count": swap_value,
        "moves": moves_value,
        "layers": swap_layers,
        "status": status,
    }

def apply_swaps(perm, layers):
    """
    Apply a sequence of SWAP layers to a permutation.

    perm: initial permutation (list)
    layers: list of SWAP layers, each layer is a list of edges (u,v)

    Returns the resulting permutation after applying all SWAPs.
    """
    current_perm = perm[:]
    for layer in layers:
        for (u, v) in layer:
            # Swap tokens at positions u and v
            current_perm[u], current_perm[v] = current_perm[v], current_perm[u]
    return current_perm

def find_constrained_swaps_ILP(pi_A, pi_B_dict, dist_matrix, use_gurobi=True):
    """
    Find SWAP sequence to route subset of virtual qubits to targets using ILP.
    
    Args:
        pi_A: List [P0, P1, ...] where pi_A[q] = P (virtual q at physical P)
        pi_B_dict: Dict {q: P} specifying only qubits that need routing
        dist_matrix: Pre-computed distance matrix dist[i][j] between physical qubits
    
    Returns:
        swaps: List of (i, j) SWAP operations on adjacent physical qubits
        final_permutation: List showing final virtual→physical mapping
    """
    n = len(pi_A)
    
    # Build edges from distance matrix (adjacent = distance 1)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i][j] == 1:
                edges.append((i, j))
    
    # === Step 1: Complete eta (target permutation) ===
    # eta[q] = target physical position for virtual qubit q
    assigned_physical = set(pi_B_dict.values())
    unassigned_logical = [q for q in range(n) if q not in pi_B_dict]
    available_physical = set(P for P in range(n) if P not in assigned_physical)
    
    eta = dict(pi_B_dict)  # Start with required assignments
    
    # Try to keep unassigned qubits in place if their position is available
    still_unassigned = []
    for q in unassigned_logical:
        current_P = pi_A[q]
        if current_P in available_physical:
            eta[q] = current_P
            available_physical.remove(current_P)
        else:
            still_unassigned.append(q)
    
    # Assign remaining qubits to remaining positions
    remaining_physical = sorted(available_physical)
    for q, P in zip(still_unassigned, remaining_physical):
        eta[q] = P
    
    # Convert to list
    eta_list = [eta[q] for q in range(n)]
    
    # === Step 2: Compute inverse permutations ===
    # pi_A_inv[P] = q means physical P has virtual q
    pi_A_inv = [0] * n
    for q in range(n):
        pi_A_inv[pi_A[q]] = q
    
    # eta_inv[P] = q means we want physical P to have virtual q
    eta_inv = [0] * n
    for q in range(n):
        eta_inv[eta_list[q]] = q
    
    # === Step 3: Construct perm for solve_min_swaps ===
    # To route from state A to state B using swaps S where S(identity) = perm:
    # We need: A[perm[P]] = B[P], so perm[P] = A^{-1}[B[P]]
    # Here: A = pi_A_inv, B = eta_inv, A^{-1} = pi_A
    # So: perm[P] = pi_A[eta_inv[P]]
    perm = [pi_A[eta_inv[P]] for P in range(n)]
    
    # Check if already at target (perm is identity)
    if perm == list(range(n)):
        return [], eta_list
    
    # === Step 4: Solve using ILP ===
    result = solve_min_swaps(perm, edges, use_gurobi=use_gurobi)
    
    if result['status'] != 'Optimal':
        return None, None
    
    # Extract swaps from layers (flatten)
    swaps = []
    for layer in result['layers']:
        for swap in layer:
            swaps.append(swap)
    
    # === Step 5: Compute final permutation ===
    # Apply swaps to pi_A to get final virtual→physical mapping
    # Maintain both directions for O(1) swap operations
    final_perm = list(pi_A)
    phys_to_virt = list(pi_A_inv)
    
    for (i, j) in swaps:
        # Get virtual qubits at physical positions i and j
        q_i = phys_to_virt[i]
        q_j = phys_to_virt[j]
        
        # Swap their physical positions
        final_perm[q_i] = j
        final_perm[q_j] = i
        phys_to_virt[i] = q_j
        phys_to_virt[j] = q_i
    
    return swaps, final_perm

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
    
    # Build adjacency list if not provided
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i][j] == 1:
                adj_list[i].add(j)
                adj_list[j].add(i)
    
    # Convert to physical-to-virtual for SWAP handling
    # Also maintain virtual-to-physical for O(1) lookup
    def init_state(virt_to_phys):
        p2v = [0] * n
        v2p = [0] * n
        for q in range(n):
            P = virt_to_phys[q]
            p2v[P] = q
            v2p[q] = P
        return tuple(p2v), tuple(v2p)
    
    start_p2v, start_v2p = init_state(pi_A)
    
    def is_goal(v2p):
        """Check if target qubits are at correct positions"""
        for q, target_P in pi_B_dict.items():
            if v2p[q] != target_P:
                return False
        return True
    
    def heuristic(v2p):
        """
        Improved heuristic: sum of distances without over-optimistic division.
        Each qubit needs at least ceil(dist/1) swaps to move dist positions.
        But swaps can help at most 2 qubits, so we use a tighter bound.
        """
        distances = []
        for q, target_P in pi_B_dict.items():
            current_P = v2p[q]  # O(1) lookup now!
            d = dist_matrix[current_P][target_P]
            if np.isinf(d):
                return math.inf
            distances.append(int(d))
        
        if not distances:
            return 0
        
        # Tighter heuristic: max distance is a lower bound
        # Also sum/2 but with ceiling, and take max of both
        total = sum(distances)
        max_dist = max(distances)
        
        # A single SWAP reduces total distance by at most 2
        # So we need at least ceil(total/2) swaps
        # But we also need at least max_dist swaps for the furthest qubit
        return max(max_dist, (total + 1) // 2)
    
    if is_goal(start_v2p):
        return [], list(pi_A)
    
    # A* search with improved state representation
    # State: (p2v_tuple, v2p_tuple) - we track both for efficiency
    start_state = (start_p2v, start_v2p)
    h0 = heuristic(start_v2p)
    
    # heap: (f, g, state, path)
    heap = [(h0, 0, start_state, [])]
    visited = {start_state: 0}
    
    max_iterations = 100000  # Safety limit
    iterations = 0
    
    while heap and iterations < max_iterations:
        iterations += 1
        f, g, (p2v, v2p), path = heapq.heappop(heap)
        
        if is_goal(v2p):
            # Convert back to virtual->physical list
            return path, list(v2p)
        
        if visited.get((p2v, v2p), float('inf')) < g:
            continue
        
        # Try all valid SWAPs on adjacent physical qubits
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    # SWAP physical qubits i and j
                    # Get virtual qubits at these positions
                    q_i = p2v[i]
                    q_j = p2v[j]
                    
                    # Create new state
                    new_p2v = list(p2v)
                    new_v2p = list(v2p)
                    
                    new_p2v[i], new_p2v[j] = q_j, q_i
                    new_v2p[q_i], new_v2p[q_j] = j, i
                    
                    new_state = (tuple(new_p2v), tuple(new_v2p))
                    new_g = g + 1
                    
                    if visited.get(new_state, float('inf')) > new_g:
                        visited[new_state] = new_g
                        new_h = heuristic(tuple(new_v2p))
                        if new_h < math.inf:
                            new_f = new_g + new_h
                            heapq.heappush(heap, (new_f, new_g, new_state, path + [(i, j)]))
    
    logging.warning(f"SWAP routing did not converge after {iterations} iterations")
    return None, None

def calculate_dist_small(mini_topology, qbit_map, dist_matrix,pi):
    dist_placeholder = 0
    qbit_map_inv = { k:v for v,k in qbit_map.items()}
    for u,v in mini_topology:
        dist_placeholder += (dist_matrix[pi[qbit_map_inv[u]]][pi[qbit_map_inv[v]]]-1)*3
    return dist_placeholder

def extract_subtopology(involved_qbits, qbit_map, config ):
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]],qbit_map[edge[1]]))
    return mini_topology
    
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
        cache_key = (pi_tuple, qbit_map_frozen)
        if cache_key in swap_cache:
            swaps, pi_init = swap_cache[cache_key]
        else:
            swaps, pi_init = find_constrained_swaps_partial(pi_list, qbit_map_input, D)
            swap_cache[cache_key] = (swaps, pi_init)
    else:
        swaps, pi_init = find_constrained_swaps_partial(pi_list, qbit_map_input, D)
    return len(swaps)

class SingleQubitPartitionResult:
    
    def __init__(self,circuit_in,parameters_in):
        self.circuit = circuit_in
        self.parameters = parameters_in
    
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
        score = 0
        for topology_idx in range(self.topology_count):
            cnot_count_topology = np.min(self.cnot_counts[topology_idx])#np.mean(self.cnot_counts[topology_idx])*0.5 + np.min(self.cnot_counts[topology_idx])*0.5
            score += cnot_count_topology/self.topology_count
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

    def transform_pi(self, pi, D, swap_cache=None):
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

def calculate_swap_cost(swaps, current_pi, used_qubits):
    """
    Calculate swap cost. Swaps involving unused qubits are costless (0).
    unused qubits are those not in used_qubits set.
    """
    cost = 0
    temp_pi = list(current_pi)
    # Build inverse map for O(1) lookup: physical -> logical
    phys_to_logical = {p: l for l, p in enumerate(temp_pi)}

    for p1, p2 in swaps:
        l1 = phys_to_logical[p1]
        l2 = phys_to_logical[p2]

        is_l1_unused = (l1 not in used_qubits)
        is_l2_unused = (l2 not in used_qubits)

        if is_l1_unused and is_l2_unused:
            step_cost = 0
        else:
            step_cost = 3

        cost += step_cost

        # Update state
        temp_pi[l1] = p2
        temp_pi[l2] = p1
        phys_to_logical[p1] = l2
        phys_to_logical[p2] = l1

    return cost

def filter_required_swaps(swaps, current_pi, pi_initial, used_qubits):
    """
    Filter swaps that are effectively 'costless' (involve unused qubits).
    Returns filtered swaps and the updated pi_initial.
    """
    required_swaps = []
    temp_pi = list(current_pi)
    
    # pi_initial might be numpy array, convert to list for mutation if needed, 
    # but we'll return a new list/array to be safe.
    updated_pi_initial = list(pi_initial)
    
    # Build inverse map for O(1) lookup: physical -> logical
    phys_to_logical = {p: l for l, p in enumerate(temp_pi)}

    for p1, p2 in swaps:
        l1 = phys_to_logical[p1]
        l2 = phys_to_logical[p2]

        is_l1_unused = (l1 not in used_qubits)
        is_l2_unused = (l2 not in used_qubits)

        if not (is_l1_unused and is_l2_unused):
            required_swaps.append((p1, p2))
        else:
            # If unused, we update the initial mapping to reflect this swap
            # effectively retconning that they started in these positions.
            # pi_initial maps logical -> physical.
            # We swap the physical locations for these logical qubits.
            # Note: updated_pi_initial[l1] should track where l1 'started'.
            # If we swap l1 and l2 physically, and it's costless, 
            # it means l1 is now 'initially' at p2, and l2 at p1.
            # But wait, temp_pi[l1] is currently p1. After swap it is p2.
            # So we update pi_initial to match the new temp_pi.
            updated_pi_initial[l1] = p2
            updated_pi_initial[l2] = p1

        # Always update the tracking state
        temp_pi[l1] = p2
        temp_pi[l2] = p1
        phys_to_logical[p1] = l2
        phys_to_logical[p2] = l1

    return required_swaps, updated_pi_initial

def construct_swap_circuit(swap_order, N):
    swap_circ = Circuit(N)
    for swap in swap_order:
        swap_circ.add_CNOT(swap[0],swap[1])
        swap_circ.add_CNOT(swap[1],swap[0])
        swap_circ.add_CNOT(swap[0],swap[1])
    return swap_circ
