"""
Simplified SABRE improvements focusing on key quality enhancements
without breaking existing functionality.
"""

import numpy as np
from collections import deque, defaultdict
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.gates.gates_Wrapper import CNOT

class qgd_SABRE:
    def __init__(self, quantum_circuit, topology, max_lookahead=4, max_E_size=20, 
                 W=0.5, alpha=0.9, initial_layout='smart', score_tolerance=0.01, stochastic_selection=False, random_seed=None):
        """
        Parameters match original with addition of:
        initial_layout: 'smart', 'random', or 'trivial'
        """
        self.circuit_qbit_num = quantum_circuit.get_Qbit_Num()
        self.qbit_num = max(max(topology)) + 1
        self.initial_circuit = quantum_circuit
        
        # Initialize RNG
        self.rng = np.random.RandomState(random_seed)

        self.score_tolerance = score_tolerance
        self.stochastic_selection = stochastic_selection

        # Build topology first
        self.topology = topology
        self.D = self.compute_distances_bfs()
        self.possible_connections_control, self.neighbours = self.generate_possible_connections()
        
        # Smart initial mapping (KEY IMPROVEMENT)
        if initial_layout == 'smart':
            self.pi = self._compute_smart_initial_layout(quantum_circuit)
        elif initial_layout == 'random':
            self.pi = np.arange(self.qbit_num)
            self.rng.shuffle(self.pi)
        else:  # 'trivial'
            self.pi = np.arange(self.qbit_num)
        
        self.max_E_size = max_E_size
        self.W = W
        self.alpha = alpha
        self.max_lookahead = max_lookahead
    
    def _compute_smart_initial_layout(self, circuit):
        """
        Smart initial layout based on circuit connectivity.
        This is the MOST IMPORTANT improvement for reducing SWAPs.
        """
        # Count interactions between qubits
        interaction_count = defaultdict(int)
        gates = circuit.get_Gates()
        
        for gate in gates:
            if gate.get_Control_Qbit() != -1:
                q1 = gate.get_Target_Qbit()
                q2 = gate.get_Control_Qbit()
                if q1 < self.circuit_qbit_num and q2 < self.circuit_qbit_num:
                    key = (min(q1, q2), max(q1, q2))
                    interaction_count[key] += 1
        
        if not interaction_count:
            # No 2-qubit gates, use trivial mapping
            return np.arange(self.qbit_num)
        
        # Find most interacting qubit pair
        most_connected = max(interaction_count.items(), key=lambda x: x[1])
        q1, q2 = most_connected[0]
        
        # Find physical qubits that are connected
        # Start with an arbitrary connected pair
        for edge in self.topology:
            p1, p2 = edge
            break  # Just take first edge
        
        # Initialize mapping
        pi = np.arange(self.qbit_num)
        
        # Place most interacting qubits on connected physical qubits
        pi[q1] = p1
        pi[q2] = p2
        
        # Place other qubits using greedy approach
        placed_logical = {q1, q2}
        placed_physical = {p1, p2}
        
        # For each remaining logical qubit, find where to place it
        remaining_logical = [q for q in range(self.circuit_qbit_num) if q not in placed_logical]
        
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
            
            for physical_q in range(self.qbit_num):
                if physical_q not in placed_physical:
                    # Calculate average distance to interacting qubits
                    total_dist = 0
                    count = 0
                    for other_q in placed_logical:
                        key = (min(logical_q, other_q), max(logical_q, other_q))
                        weight = interaction_count.get(key, 0)
                        if weight > 0:
                            other_physical = pi[other_q]
                            total_dist += self.D[physical_q][other_physical] * weight
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
    
    def compute_distances_bfs(self):
        """BFS distance computation - faster than Floyd-Warshall."""
        D = np.ones((self.qbit_num, self.qbit_num)) * np.inf
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in self.topology:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex
        for start in range(self.qbit_num):
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
        
        return D
    
    def generate_possible_connections(self):
        """Generate possible connections - using sets for O(1) lookup."""
        possible_connections_control = [set() for _ in range(self.qbit_num)]
        neighbours = [set() for _ in range(self.qbit_num)]
        
        for qbit1, qbit2 in self.topology:
            possible_connections_control[qbit2].add(qbit1)
            neighbours[qbit1].add(qbit2)
            neighbours[qbit2].add(qbit1)
        
        return possible_connections_control, neighbours
    
    def is_gate_possible(self, pi, q_target, q_control):
        """Check if gate is possible (unchanged from original)."""
        if q_control == -1:
            return True, 0
        else:
            Q_control = pi[q_control]
            Q_target = pi[q_target]
            if Q_target in self.possible_connections_control[Q_control]:
                return True, 0
            elif Q_control in self.possible_connections_control[Q_target]:
                return True, 1
        return False, 0
    
    def H_basic(self, pi, q1, q2):
        """Basic cost function (unchanged)."""
        if q2 == -1:
            return 0
        return self.D[pi[q1]][pi[q2]]
    
    def update_pi(self, pi, SWAP):
        """Update mapping after swap."""
        q1, q2 = SWAP
        pi[q1], pi[q2] = pi[q2], pi[q1]
    
    def get_inverse_pi(self, pi):
        """Get inverse mapping - optimized."""
        inverse_pi = np.empty(len(pi), dtype=int)
        inverse_pi[pi] = np.arange(len(pi))
        return inverse_pi
    
    def obtain_SWAPS(self, F, pi, DAG):
        """Get candidate SWAPs - optimized with sets."""
        swaps = set()
        inverse_pi = self.get_inverse_pi(pi)
        involved_qbits = set()
        
        for gate_idx in F:
            gate = DAG[gate_idx][0]
            q_control = gate.get_Control_Qbit()
            q_target = gate.get_Target_Qbit()
            if q_control == -1:
                continue
            involved_qbits.add(q_control)
            involved_qbits.add(q_target)
        
        for qbit in involved_qbits:
            Q = pi[qbit]
            for Q_cand in self.neighbours[Q]:
                q_cand = inverse_pi[Q_cand]
                swap = (min(qbit, q_cand), max(qbit, q_cand))
                swaps.add(swap)
        
        return list(swaps)
    
    def generate_E(self, F, DAG, IDAG, resolved_gates):
        """Generate extended set - simplified version."""
        E = []
        E_set = set()
        
        for front_gate in F:
            if len(E) >= self.max_E_size:
                break
            
            involved_qbits = [DAG[front_gate][0].get_Target_Qbit(), 
                            DAG[front_gate][0].get_Control_Qbit()]
            lookahead_count = 0
            children = list(DAG[front_gate][1])
            
            while lookahead_count < self.max_lookahead and children:
                for gate_idx in children[:]:  # Copy to avoid modification issues
                    gate = DAG[gate_idx][0]
                    
                    if gate.get_Control_Qbit() == -1:
                        children.extend(DAG[gate_idx][1])
                    else:
                        gate_depth = self.calculate_gate_depth(gate_idx, IDAG, resolved_gates)
                        if (len(E) < self.max_E_size and 
                            (gate.get_Target_Qbit() in involved_qbits or 
                             gate.get_Control_Qbit() in involved_qbits) and 
                            gate_idx not in E_set and gate_depth < 6):
                            E.append(gate_idx)
                            E_set.add(gate_idx)
                            lookahead_count += 1
                            if lookahead_count < self.max_lookahead:
                                children.extend(DAG[gate_idx][1])
                    
                    children.remove(gate_idx)
        
        return E
    
    def calculate_gate_depth(self, gate_idx, IDAG, resolved_gates):
        """Calculate gate depth - iterative to avoid stack issues."""
        depth = 1
        visited = set()
        stack = list(IDAG[gate_idx][1])
        
        while stack:
            parent_idx = stack.pop()
            if parent_idx in visited:
                continue
            visited.add(parent_idx)
            
            if not resolved_gates[parent_idx]:
                parent_gate = IDAG[parent_idx][0]
                if parent_gate.get_Control_Qbit() != -1:
                    depth += 1
                stack.extend(IDAG[parent_idx][1])
        
        return depth
    
    def check_dependencies(self, gate_idx, IDAG, resolved_gates):
        """Check dependencies - iterative."""
        visited = set()
        stack = list(IDAG[gate_idx][1])
        
        while stack:
            parent_idx = stack.pop()
            if parent_idx in visited:
                continue
            visited.add(parent_idx)
            
            if not resolved_gates[parent_idx]:
                return False
            stack.extend(IDAG[parent_idx][1])
        
        return True
    
    def score_swap_improved(self, swap, F, E, pi, DAG, IDAG, resolved_gates):
        """
        Improved scoring that rewards immediately executable gates.
        """
        pi_temp = pi.copy()
        self.update_pi(pi_temp, swap)
        
        score = 0
        gates_resolved = 0
        
        # Score gates in F with bonus for resolution
        for gate_idx in F:
            gate = DAG[gate_idx][0]
            q_target, q_control = gate.get_Target_Qbit(), gate.get_Control_Qbit()
            
            if q_control != -1:
                dist = self.H_basic(pi_temp, q_target, q_control)
                if dist == 1:  # This swap makes the gate executable
                    gates_resolved += 1
                    score -= 5  # Bonus for resolving gates
                else:
                    score += dist
        
        if len(F) > 0:
            score /= len(F)
        
        # Score extended set
        if len(E) != 0:
            score_temp = 0
            for gate_idx in E:
                gate = DAG[gate_idx][0]
                q_target, q_control = gate.get_Target_Qbit(), gate.get_Control_Qbit()
                depth = self.calculate_gate_depth(gate_idx, IDAG, resolved_gates)
                score_temp += self.H_basic(pi_temp, q_target, q_control) * (self.alpha ** depth)
            score_temp *= self.W / len(E)
            score += score_temp
        
        # Penalty if no gates are resolved
        if gates_resolved == 0:
            score += 1
        
        return score
    
    def Heuristic_search(self, F, pi, DAG, IDAG):
        """Main heuristic search - with improved scoring."""
        swap_count = 0
        resolved_gates = [False] * len(DAG)
        flags = [0] * len(DAG)
        swap_list = []
        gate_order = []
        E = self.generate_E(F, DAG, IDAG, resolved_gates)
        swap_prev = None
        
        while len(F) != 0:
            execute_gate_list = []
            
            # Check which gates can be executed
            for gate_idx in F:
                gate = DAG[gate_idx][0]
                possible, flag = self.is_gate_possible(pi, gate.get_Target_Qbit(), 
                                                      gate.get_Control_Qbit())
                if possible:
                    execute_gate_list.append(gate_idx)
                    resolved_gates[gate_idx] = True
                    flags[gate_idx] = flag
            
            if len(execute_gate_list) != 0:
                # Execute gates
                for gate_idx in execute_gate_list:
                    F.remove(gate_idx)
                    gate_order.append(gate_idx)
                    successors = DAG[gate_idx][1]
                    for new_gate_idx in successors:
                        resolved = self.check_dependencies(new_gate_idx, IDAG, resolved_gates)
                        if resolved and new_gate_idx not in F:
                            F.append(new_gate_idx)
                
                E = self.generate_E(F, DAG, IDAG, resolved_gates)
                continue
            else:
                # Need to find a SWAP
                scores = []
                swaps = self.obtain_SWAPS(F, pi, DAG)
                
                if len(swaps) != 0:
                    for SWAP in swaps:
                        if SWAP == swap_prev:
                            scores.append(np.inf)
                            continue
                        
                        # Use improved scoring
                        score = self.score_swap_improved(SWAP, F, E, pi, DAG, IDAG, resolved_gates)
                        scores.append(score)
                    if self.stochastic_selection:
                        min_swap = self.select_swap_stochastic(swaps, scores)
                    else:
                        min_idx = np.argmin(scores)
                        min_swap = swaps[min_idx]
                    swap_prev = min_swap
                    gate_order.append(min_swap)
                    self.update_pi(pi, min_swap)
                    swap_count += 1
                else:
                    print("ERROR: circuit cannot be mapped please check topology")
                    break
        
        return gate_order, flags, swap_count
    
    # Keep all the original helper methods unchanged
    def get_reverse_circuit(self, circuit):
        gates = circuit.get_Gates()
        reverse_circuit = Circuit(self.circuit_qbit_num)
        for idx in range(len(gates)):
            gate_idx = len(gates) - 1 - idx
            gate = gates[gate_idx]
            reverse_circuit.add_Gate(gate)
        return reverse_circuit
    
    def generate_DAG(self, circuit):
        DAG = []
        gates = circuit.get_Gates()
        for gate in gates:
            DAG.append([gate, circuit.get_Children(gate)])
        return DAG
    
    def generate_inverse_DAG(self, circuit):
        inverse_DAG = []
        gates = circuit.get_Gates()
        for gate in gates:
            inverse_DAG.append([gate, circuit.get_Parents(gate)])
        return inverse_DAG
    
    def get_initial_layer(self, circuit):
        initial_layer = []
        gates = circuit.get_Gates()
        for gate_idx in range(len(gates)):
            gate = gates[gate_idx]
            if len(circuit.get_Parents(gate)) == 0:
                initial_layer.append(gate_idx)
        return initial_layer
    
    def get_mapped_circuit(self, circuit, init_pi, gate_order, flags, parameters):
        circuit_mapped = Circuit(self.qbit_num)
        gates = circuit.get_Gates()
        parameters_new = np.array([])
        
        for gate_idx in gate_order:
            if isinstance(gate_idx, int):
                gate = gates[gate_idx]
                if gate.get_Parameter_Num() != 0:
                    start_idx = gate.get_Parameter_Start_Index()
                    params = parameters[start_idx:start_idx + gate.get_Parameter_Num()]
                    parameters_new = np.append(parameters_new, params)
                
                q_target, q_control = gate.get_Target_Qbit(), gate.get_Control_Qbit()
                Q_target = init_pi[q_target]
                Q_control = init_pi[q_control] if q_control != -1 else -1
                
                gate.set_Target_Qbit(Q_target)
                if q_control != -1:
                    gate.set_Control_Qbit(Q_control)
                
                if flags[gate_idx] == 1:
                    gate.set_Target_Qbit(Q_control)
                    gate.set_Control_Qbit(Q_target)
                    if isinstance(gate, CNOT):
                        circuit_mapped.add_H(Q_target)
                        circuit_mapped.add_H(Q_control)
                
                circuit_mapped.add_Gate(gate)
                
                if flags[gate_idx] == 1 and isinstance(gate, CNOT):
                    circuit_mapped.add_H(Q_target)
                    circuit_mapped.add_H(Q_control)
            else:
                q1, q2 = gate_idx[0], gate_idx[1]
                Q1, Q2 = init_pi[q1], init_pi[q2]
                circuit_mapped.add_SWAP(Q1, Q2)
                self.update_pi(init_pi, [q1, q2])
        
        return circuit_mapped, parameters_new
    
    def select_swap_stochastic(self, swaps, scores):
        """Stochastic swap selection."""
        scores_array = np.array(scores)
        valid_mask = ~np.isinf(scores_array)
        
        if not np.any(valid_mask):
            return swaps[0]
        
        valid_scores = scores_array[valid_mask]
        min_score = np.min(valid_scores)
        
        if self.stochastic_selection and min_score > 0:
            threshold = min_score * (1 + self.score_tolerance)
            close_indices = np.where((scores_array <= threshold) & valid_mask)[0]
            
            if len(close_indices) > 1:
                selected_idx = self.rng.choice(close_indices)
                return swaps[selected_idx]
            else:
                return swaps[close_indices[0]]
        else:
            min_idx = np.argmin(scores_array)
            return swaps[min_idx]

    def map_circuit(self, parameters=np.array([]), iter_num=1):
        """Main entry point - unchanged interface."""
        init_pi = self.pi.copy()
        circuit = self.initial_circuit
        reverse_circuit = self.get_reverse_circuit(circuit)
        DAG = self.generate_DAG(circuit)
        IDAG = self.generate_inverse_DAG(circuit)
        first_layer = self.get_initial_layer(circuit)
        DAG_reverse = self.generate_DAG(reverse_circuit)
        IDAG_reverse = self.generate_inverse_DAG(reverse_circuit)
        first_layer_reverse = self.get_initial_layer(reverse_circuit)
        
        for idx in range(iter_num):
            self.Heuristic_search(first_layer.copy(), init_pi, DAG, IDAG)
            self.Heuristic_search(first_layer_reverse.copy(), init_pi, DAG_reverse, IDAG_reverse)
        
        final_pi = init_pi.copy()
        gate_order, flags, swap_count = self.Heuristic_search(first_layer.copy(), final_pi, DAG, IDAG)
        circuit_mapped, parameters_new = self.get_mapped_circuit(circuit, init_pi.copy(), 
                                                                gate_order, flags, parameters)
        
        return circuit_mapped, parameters_new, init_pi, final_pi, swap_count