#Source paper: https://arxiv.org/pdf/1809.02573

from os import path
import numpy as np 
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.gates.qgd_CNOT import qgd_CNOT as CNOT

class qgd_SABRE:
    
    def __init__(self,quantum_circuit,topology, max_E_size=20,W=0.5):
        self.circuit_qbit_num = quantum_circuit.get_Qbit_Num()
        self.qbit_num = max(max(topology))+1
        self.initial_circuit = quantum_circuit
        self.pi = np.arange(self.qbit_num)
        #np.random.shuffle(self.pi)
        self.topology = topology
        self.D = self.Floyd_Warshall(topology)
        self.max_E_size = max_E_size
        self.W = W
        self.possible_connections_control,self.neighbours = self.geneterate_possible_connections(topology)

        
    def Floyd_Warshall(self,topology):
        D = np.ones((self.qbit_num,self.qbit_num))*np.inf
        for vertex in topology:
            u,v = vertex
            D[u][v] = 1
            D[v][u] = 1
        for idx in range(self.qbit_num):
            D[idx][idx] = 0
        for k in range(self.qbit_num):
            for i in range(self.qbit_num):
                for j in range(self.qbit_num):
                    if D[i][j] > D[i][k] + D[k][j]:
                        D[i][j] = D[i][k] + D[k][j]
        return D
        
    def geneterate_possible_connections(self,topology):
        possible_connections_control = [ [] for _ in range(self.qbit_num) ]
        neighbours = [ [] for _ in range(self.qbit_num) ]
        for connection in topology:
            qbit1,qbit2 = connection
            possible_connections_control[qbit1].append(qbit2)
            if qbit2 not in neighbours[qbit1]:
                neighbours[qbit1].append(qbit2)
            if qbit1 not in neighbours[qbit2]:
                neighbours[qbit2].append(qbit1)
        return possible_connections_control,neighbours
    
    def get_reverse_circuit(self,circuit):
        gates = circuit.get_Gates()
        inverse_circuit = Circuit(self.circuit_qbit_num)
        for idx in range(len(gates)):
            gate_idx = len(gates)-1-idx
            gate = gates[gate_idx]
            inverse_circuit.add_Gate(gate)
        return inverse_circuit
    
    def generate_DAG(self,circuit):
        DAG = []
        gates = circuit.get_Gates()
        for gate in gates:
            DAG.append([gate,circuit.get_Children(gate)])
        return DAG
        
    def generate_inverse_DAG(self,circuit):
        inverse_DAG = []
        gates = circuit.get_Gates()
        for gate in gates:
            inverse_DAG.append([gate,circuit.get_Parents(gate)])
        return inverse_DAG
        
    def get_initial_layer(self,circuit):
        initial_layer = []
        gates = circuit.get_Gates()
        for gate_idx in range(len(gates)):
            gate = gates[gate_idx]
            if len(circuit.get_Parents(gate))==0:
                initial_layer.append(gate_idx)
        return initial_layer
    
    def is_connection_possible(self,pi,q_target,q_control):
        if q_control == -1:
            return True,0
        else:
            Q_control = pi[q_control]
            Q_target = pi[q_target]
            if Q_target in self.possible_connections_control[Q_control]:
                return True,0
            elif Q_control in self.possible_connections_control[Q_target]:
                return True,1
                
        return False,0
        
    def H_basic(self,pi,q1,q2):
        if q2==-1:
            return 0 
        return self.D[pi[q1]][pi[q2]]
         
    def update_pi(self, pi, SWAP):
        q1,q2 = SWAP
        placeholder = pi[q2]
        pi[q2] = pi[q1]
        pi[q1] = placeholder
        
    def get_inverse_pi(self, pi):
        inverse_pi = list(range(len(pi)))
        for i in range(len(pi)):
            Q = pi[i]
            inverse_pi[Q] = i 
        return inverse_pi
        
    def obtain_SWAPS(self, F, pi,DAG):
        swaps = []
        inverse_pi = self.get_inverse_pi(pi)
        for gate_idx in F:
            gate = DAG[gate_idx][0]
            q_control = gate.get_Control_Qbit()
            q_target = gate.get_Target_Qbit()
            if q_control == -1:
                continue
            candidates_control = self.neighbours[pi[q_control]]
            candidates_target = self.neighbours[pi[q_target]]
            for Q_cand in candidates_control:
                swaps.append([[min(q_control,inverse_pi[Q_cand]),max(q_control,inverse_pi[Q_cand])],gate_idx])
            for Q_cand in candidates_target:
                swaps.append([[min(q_target,inverse_pi[Q_cand]),max(q_target,inverse_pi[Q_cand])],gate_idx])
        return swaps
    
    def Heuristic_search(self,F,pi,D,DAG,IDAG):
        swap_count = 0
        resolved_gates = [False]*len(DAG)
        flags = [0]*len(DAG)
        swap_list = []
        gate_order = []
        E = []
        swap_prev = []
        for gate_idx in F: 
            successors = DAG[gate_idx][1]
            for successor in successors:
                if successor not in E and len(E)<self.max_E_size:
                    E.append(successor)
            if len(E)<self.max_E_size:
                for successor in successors:
                    children = DAG[gate_idx][1]
                    for child in children:
                        if child not in E and len(E)<self.max_E_size:
                            E.append(child)
        while len(F)!=0:
            execute_gate_list = []
            for gate_idx in F:
                gate = DAG[gate_idx][0]
                possible, flag = self.is_connection_possible(pi,gate.get_Target_Qbit(),gate.get_Control_Qbit()) 
                if possible:
                    execute_gate_list.append(gate_idx)
                    resolved_gates[gate_idx] = True 
                    flags[gate_idx] = flag
            if len(execute_gate_list ) != 0:
                for gate_idx in execute_gate_list:
                    F.remove(gate_idx)
                    gate_order.append(gate_idx)
                    successors = DAG[gate_idx][1]
                    for new_gate_idx in successors:
                        dependencies = IDAG[new_gate_idx][1]
                        resolved = 1
                        for dependency in dependencies:
                            resolved *= resolved_gates[dependency]
                        if resolved and new_gate_idx not in F:
                            F.append(new_gate_idx)
                E = []
                for gate_idx in F: 
                    successors = DAG[gate_idx][1]
                    for successor in successors:
                        if successor not in E and len(E)<self.max_E_size:
                            E.append(successor)
                    if len(E)<self.max_E_size:
                        for successor in successors:
                            children = DAG[gate_idx][1]
                            for child in children:
                                if child not in E and len(E)<self.max_E_size:
                                    E.append(child)
                continue
            else:
                scores = []
                swaps = self.obtain_SWAPS(F,pi,DAG)
                if len(swaps)!=0:
                    for SWAP_idx in range(len(swaps)):
                        SWAP = swaps[SWAP_idx][0]
                        if SWAP == swap_prev:
                            scores.append(np.inf)
                            continue
                        pi_temp = pi.copy()
                        self.update_pi(pi_temp,SWAP)
                        score = 0
                        for gate_idx in F: 
                            gate = DAG[gate_idx][0]
                            q_target, q_control = gate.get_Target_Qbit(),gate.get_Control_Qbit()
                            score += self.H_basic(pi,q_target, q_control)
                        score *= 1/len(F)
                        if len(E) != 0:
                            score_temp = 0
                            for gate_idx in E:
                                gate = DAG[gate_idx][0]
                                q_target, q_control = gate.get_Target_Qbit(),gate.get_Control_Qbit()
                                score_temp += self.H_basic(pi,q_target, q_control)
                            score_temp *= self.W/len(E)
                            score+=score_temp
                        scores.append(score)
                    min_idx = np.argmin(np.array(scores))
                    min_swap = swaps[min_idx][0]
                    swap_prev = min_swap
                    swap_gate_idx = swaps[min_idx][1]
                    gate_order.append(min_swap)
                    self.update_pi(pi,min_swap)
                    swap_count +=1
                else:
                    print("ERORR: circuit cannot be mapped please check topology")
                    break
        return gate_order,flags,swap_count
        
    def get_mapped_circuit(self,circuit,init_pi,gate_order,flags,parameters):
        circuit_mapped = Circuit(self.circuit_qbit_num)
        gates = circuit.get_Gates()
        parameters_new = np.array([])
        for gate_idx in gate_order:
            if isinstance(gate_idx,int):
                gate=gates[gate_idx]
                if gate.get_Parameter_Num() != 0:
                    start_idx = gate.get_Parameter_Start_Index()
                    params = parameters[start_idx:start_idx + gate.get_Parameter_Num()]
                    parameters_new = np.append(parameters_new,params)
                q_target,q_control = gate.get_Target_Qbit(),gate.get_Control_Qbit()
                Q_target,Q_control = init_pi[q_target],init_pi[q_control]
                gate.set_Target_Qbit(Q_target)
                gate.set_Control_Qbit(Q_control)
                if flags[gate_idx] == 1:
                    gate.set_Target_Qbit(Q_control)
                    gate.set_Control_Qbit(Q_target)
                    if isinstance(gate,CNOT):
                        circuit_mapped.add_H(Q_target)
                        circuit_mapped.add_H(Q_control)
                circuit_mapped.add_Gate(gate)
                if flags[gate_idx] == 1 and isinstance(gate,CNOT):
                    circuit_mapped.add_H(Q_target)
                    circuit_mapped.add_H(Q_control)
            else:
                q1,q2 = gate_idx[0],gate_idx[1]
                Q1,Q2 = init_pi[q1],init_pi[q2]
                circuit_mapped.add_CNOT(Q1,Q2)
                circuit_mapped.add_CNOT(Q2,Q1)
                circuit_mapped.add_CNOT(Q1,Q2)
                self.update_pi(init_pi,[q1,q2])
                
        return circuit_mapped,parameters_new
        
    def map_circuit(self,parameters=np.array([]),iter_num=1):
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
            self.Heuristic_search(first_layer.copy(),init_pi,self.D,DAG,IDAG)
            self.Heuristic_search(first_layer_reverse.copy(),init_pi,self.D,DAG_reverse,IDAG_reverse)
        final_pi = init_pi.copy()
        gate_order,flags,swap_count = self.Heuristic_search(first_layer.copy(),final_pi,self.D,DAG,IDAG)
        circuit_mapped,parameters_new = self.get_mapped_circuit(circuit,init_pi.copy(),gate_order,flags,parameters)
        return circuit_mapped,parameters_new,init_pi,final_pi,swap_count
