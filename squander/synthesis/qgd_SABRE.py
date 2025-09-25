import numpy as np
from collections import deque, defaultdict
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.gates.gates_Wrapper import CNOT

class qgd_SABRE:
    def __init__(self, quantum_circuit, topology, max_lookahead=4, max_E_size=20, W=0.5, alpha=0.9):
        self.circuit_qbit_num = quantum_circuit.get_Qbit_Num()
        self.qbit_num = max(max(topology)) + 1
        self.initial_circuit = quantum_circuit
        self.pi = np.arange(self.qbit_num)
        np.random.shuffle(self.pi)
        self.topology = topology
        
        # Use BFS-based distance calculation instead of Floyd-Warshall
        self.D = self.compute_distances_bfs()
        
        self.max_E_size = max_E_size
        self.W = W
        self.alpha = alpha
        self.possible_connections_control, self.neighbours = self.generate_possible_connections()
        self.max_lookahead = max_lookahead
        
        # Cache for gate depths and dependencies
        self._depth_cache = {}
        self._dependency_cache = {}
    
    def compute_distances_bfs(self):
        """Compute distances using BFS - O(n²) for sparse graphs instead of O(n³)"""
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
        """Optimized with sets for O(1) membership testing"""
        possible_connections_control = [set() for _ in range(self.qbit_num)]
        neighbours = [set() for _ in range(self.qbit_num)]
        
        for qbit1, qbit2 in self.topology:
            possible_connections_control[qbit2].add(qbit1)
            neighbours[qbit1].add(qbit2)
            neighbours[qbit2].add(qbit1)
        
        return possible_connections_control, neighbours
    
    def calculate_gate_depth_cached(self, gate_idx, IDAG, resolved_gates):
        """Cached version of gate depth calculation"""
        if gate_idx in self._depth_cache:
            return self._depth_cache[gate_idx]
        
        depth = self._calculate_gate_depth_impl(gate_idx, IDAG, resolved_gates)
        self._depth_cache[gate_idx] = depth
        return depth
    
    def _calculate_gate_depth_impl(self, gate_idx, IDAG, resolved_gates):
        """Iterative implementation with visited set to avoid redundant traversals"""
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
    
    def check_dependencies_cached(self, gate_idx, IDAG, resolved_gates):
        """Cached dependency checking"""
        # Create a cache key based on resolved gates state
        cache_key = (gate_idx, tuple(resolved_gates))
        if cache_key in self._dependency_cache:
            return self._dependency_cache[cache_key]
        
        result = self._check_dependencies_impl(gate_idx, IDAG, resolved_gates)
        self._dependency_cache[cache_key] = result
        return result
    
    def _check_dependencies_impl(self, gate_idx, IDAG, resolved_gates):
        """Iterative implementation with early termination"""
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
    
    def generate_E_optimized(self, F, DAG, IDAG, resolved_gates):
        """Optimized E generation with sets and early termination"""
        E = set()
        E_list = []  # Maintain order for compatibility
        
        for front_gate in F:
            if len(E) >= self.max_E_size:
                break
                
            involved_qbits = {
                DAG[front_gate][0].get_Target_Qbit(),
                DAG[front_gate][0].get_Control_Qbit()
            }
            
            lookahead_count = 0
            children = deque(DAG[front_gate][1])
            visited = {front_gate}
            
            while lookahead_count < self.max_lookahead and children and len(E) < self.max_E_size:
                gate_idx = children.popleft()
                
                if gate_idx in visited:
                    continue
                visited.add(gate_idx)
                
                gate = DAG[gate_idx][0]
                
                if gate.get_Control_Qbit() == -1:
                    children.extend(DAG[gate_idx][1])
                else:
                    gate_qbits = {gate.get_Target_Qbit(), gate.get_Control_Qbit()}
                    if gate_qbits & involved_qbits:  # Set intersection
                        gate_depth = self.calculate_gate_depth_cached(gate_idx, IDAG, resolved_gates)
                        if gate_depth < 6 and gate_idx not in E:
                            E.add(gate_idx)
                            E_list.append(gate_idx)
                            lookahead_count += 1
                            if lookahead_count < self.max_lookahead:
                                children.extend(DAG[gate_idx][1])
        
        return E_list
    
    def obtain_SWAPS_optimized(self, F, pi, DAG):
        """Optimized SWAP generation using sets"""
        swaps = set()
        inverse_pi = self.get_inverse_pi(pi)
        involved_qbits = set()
        
        for gate_idx in F:
            gate = DAG[gate_idx][0]
            q_control = gate.get_Control_Qbit()
            if q_control == -1:
                continue
            involved_qbits.add(q_control)
            involved_qbits.add(gate.get_Target_Qbit())
        
        for qbit in involved_qbits:
            Q = pi[qbit]
            for Q_cand in self.neighbours[Q]:
                q_cand = inverse_pi[Q_cand]
                swap = (min(qbit, q_cand), max(qbit, q_cand))
                swaps.add(swap)
        
        return list(swaps)
    
    def score_swap_vectorized(self, swap, F, E, pi, DAG, IDAG, resolved_gates):
        """Vectorized scoring when possible"""
        pi_temp = pi.copy()
        self.update_pi(pi_temp, swap)
        
        # Vectorize F scoring
        f_score = 0
        if F:
            for gate_idx in F:
                gate = DAG[gate_idx][0]
                q_t, q_c = gate.get_Target_Qbit(), gate.get_Control_Qbit()
                if q_c != -1:
                    f_score += self.D[pi_temp[q_t], pi_temp[q_c]]
            f_score /= len(F)
        
        # Vectorize E scoring with precomputed depths
        e_score = 0
        if E and self.W > 0:
            for gate_idx in E:
                gate = DAG[gate_idx][0]
                q_t, q_c = gate.get_Target_Qbit(), gate.get_Control_Qbit()
                if q_c != -1:
                    depth = self.calculate_gate_depth_cached(gate_idx, IDAG, resolved_gates)
                    e_score += self.D[pi_temp[q_t], pi_temp[q_c]] * (self.alpha ** depth)
            e_score *= self.W / len(E)
        
        return f_score + e_score
    
    def Heuristic_search(self, F, pi, DAG, IDAG):
        """Optimized heuristic search with all improvements"""
        swap_count = 0
        resolved_gates = [False] * len(DAG)
        flags = [0] * len(DAG)
        gate_order = []
        
        # Clear caches for new search
        self._depth_cache.clear()
        self._dependency_cache.clear()
        
        E = self.generate_E_optimized(F, DAG, IDAG, resolved_gates)
        swap_prev = None
        
        while F:
            execute_gate_list = []
            
            # Check executable gates
            for gate_idx in F:
                gate = DAG[gate_idx][0]
                q_t, q_c = gate.get_Target_Qbit(), gate.get_Control_Qbit()
                
                if q_c == -1:
                    possible, flag = True, 0
                else:
                    Q_c, Q_t = pi[q_c], pi[q_t]
                    if Q_t in self.possible_connections_control[Q_c]:
                        possible, flag = True, 0
                    elif Q_c in self.possible_connections_control[Q_t]:
                        possible, flag = True, 1
                    else:
                        possible, flag = False, 0
                
                if possible:
                    execute_gate_list.append(gate_idx)
                    resolved_gates[gate_idx] = True
                    flags[gate_idx] = flag
            
            if execute_gate_list:
                for gate_idx in execute_gate_list:
                    F.remove(gate_idx)
                    gate_order.append(gate_idx)
                    
                    # Update front layer
                    for new_gate_idx in DAG[gate_idx][1]:
                        if self.check_dependencies_cached(new_gate_idx, IDAG, resolved_gates) and new_gate_idx not in F:
                            F.append(new_gate_idx)
                
                # Clear dependency cache when resolved gates change
                self._dependency_cache.clear()
                E = self.generate_E_optimized(F, DAG, IDAG, resolved_gates)
            else:
                # Need to insert SWAP
                swaps = self.obtain_SWAPS_optimized(F, pi, DAG)
                
                if swaps:
                    # Score swaps in parallel if possible
                    scores = []
                    for swap in swaps:
                        if swap == swap_prev:
                            scores.append(np.inf)
                        else:
                            score = self.score_swap_vectorized(swap, F, E, pi, DAG, IDAG, resolved_gates)
                            scores.append(score)
                    
                    min_idx = np.argmin(scores)
                    min_swap = swaps[min_idx]
                    swap_prev = min_swap
                    gate_order.append(min_swap)
                    self.update_pi(pi, min_swap)
                    swap_count += 1
                else:
                    print("ERROR: circuit cannot be mapped to topology")
                    break
        
        return gate_order, flags, swap_count
    
    # Keep the remaining methods from original implementation
    def get_reverse_circuit(self, circuit):
        gates = circuit.get_Gates()
        reverse_circuit = Circuit(self.circuit_qbit_num)
        for idx in range(len(gates)):
            gate_idx = len(gates) - 1 - idx
            reverse_circuit.add_Gate(gates[gate_idx])
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
            if len(circuit.get_Parents(gates[gate_idx])) == 0:
                initial_layer.append(gate_idx)
        return initial_layer
    
    def update_pi(self, pi, SWAP):
        q1, q2 = SWAP
        pi[q1], pi[q2] = pi[q2], pi[q1]
    
    def get_inverse_pi(self, pi):
        inverse_pi = np.empty(len(pi), dtype=int)
        inverse_pi[pi] = np.arange(len(pi))
        return inverse_pi
    
    def get_mapped_circuit(self, circuit, init_pi, gate_order, flags, parameters):
        # Same as original implementation
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
                Q_target, Q_control = init_pi[q_target], init_pi[q_control]
                gate.set_Target_Qbit(Q_target)
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
    
    def map_circuit(self, parameters=np.array([]), iter_num=1):
        # Same as original implementation
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
        circuit_mapped, parameters_new = self.get_mapped_circuit(circuit, init_pi.copy(), gate_order, flags, parameters)
        
        return circuit_mapped, parameters_new, init_pi, final_pi, swap_count