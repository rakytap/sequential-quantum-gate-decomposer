#Source paper: https://arxiv.org/pdf/1809.02573
## #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:44:26 2020
Copyright 2020 Peter Rakyta, Ph.D.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author: Peter Rakyta, Ph.D.
"""
from os import path
import numpy as np 
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.gates.qgd_CNOT import qgd_CNOT as CNOT

## \file qgd_SABRE.py
##    \brief A QGD Python class for mapping and routing circuits to different topologies.


class qgd_SABRE:
## 
# @brief Constructor of the class.
# @param quantum_circuit SQUANDER circuit to be mapped and routed
# @param topology target topology for circuit to be routed to
# @param max_E_size Maximum lookahead size for more see at:  https://arxiv.org/pdf/1809.02573
# @param W lookahead cost function weight for more see at:  https://arxiv.org/pdf/1809.02573
    def __init__(self,quantum_circuit,topology, max_lookahead=4, max_E_size=20,W=0.5,alpha=0.9):
        self.circuit_qbit_num = quantum_circuit.get_Qbit_Num() # number of qubits in circuit to be mapped and routed
        self.qbit_num = max(max(topology))+1 # number of qubits in system to be mapped to
        self.initial_circuit = quantum_circuit 
        self.pi = np.arange(self.qbit_num) #initial mapping
        np.random.shuffle(self.pi)
        self.topology = topology
        self.D = self.Floyd_Warshall()
        self.max_E_size = max_E_size
        self.W = W
        self.alpha = alpha
        self.possible_connections_control,self.neighbours = self.geneterate_possible_connections()
        self.max_lookahead = max_lookahead

##
## @brief Floyd_warshall algorithm to calculate distance matrix
    def Floyd_Warshall(self):
        D = np.ones((self.qbit_num,self.qbit_num))*np.inf
        for vertex in self.topology:
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
##
## @brief generate list of possible connections based on inputted topology
    def geneterate_possible_connections(self):
        possible_connections_control = [ [] for _ in range(self.qbit_num) ]
        neighbours = [ [] for _ in range(self.qbit_num) ]
        for connection in self.topology:
            qbit1,qbit2 = connection
            possible_connections_control[qbit2].append(qbit1)
            if qbit2 not in neighbours[qbit1]:
                neighbours[qbit1].append(qbit2)
            if qbit1 not in neighbours[qbit2]:
                neighbours[qbit2].append(qbit1)
        return possible_connections_control,neighbours
##
## @brief generates reverse circuit
## @param circuit Circuit to be reversed 
    def get_reverse_circuit(self, circuit):
        gates = circuit.get_Gates()
        reverse_circuit = Circuit(self.circuit_qbit_num)
        for idx in range(len(gates)):
            gate_idx = len(gates)-1-idx
            gate = gates[gate_idx]
            reverse_circuit.add_Gate(gate)
        return reverse_circuit
##
## @brief generates DAG of input circuit
## @param cirucit Circuit to generate DAG of
    def generate_DAG(self,circuit):
        DAG = []
        gates = circuit.get_Gates()
        for gate in gates:
            DAG.append([gate,circuit.get_Children(gate)])
        return DAG
##
## @brief generates inverse DAG of input circuit
## @param cirucit Circuit to generate inverse DAG of
    def generate_inverse_DAG(self,circuit):
        inverse_DAG = []
        gates = circuit.get_Gates()
        for gate in gates:
            inverse_DAG.append([gate,circuit.get_Parents(gate)])
        return inverse_DAG
##
## @brief generates initial layer of input circuit for more see at: https://arxiv.org/pdf/1809.02573
## @param cirucit Circuit to generate intial layer of
    def get_initial_layer(self,circuit):
        initial_layer = []
        gates = circuit.get_Gates()
        for gate_idx in range(len(gates)):
            gate = gates[gate_idx]
            if len(circuit.get_Parents(gate))==0:
                initial_layer.append(gate_idx)
        return initial_layer
##
## @brief checks if connection between two virtual qubits q_target and q_control is possible, if second return is 1 it means gate might need to be inverted (with H gates in the case of CNOT) 
## @param pi mapping between virtual qubits q and physical qubits Q
## @param q_target virtual qubit that is the target of the gate
## @param q_control virtual qubit that is the control of the gate
    def is_gate_possible(self,pi,q_target,q_control):
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

##
## @brief basic cost function that returns distance of two virtual qubits q1 and q2 
## @param pi mapping between virtual qubits q and physical qubits Q
## @param q1 virtual qubit that is the target of the gate
## @param q2 virtual qubit that is the control of the gate
    def H_basic(self,pi,q1,q2):
        if q2==-1:
            return 0 
        return self.D[pi[q1]][pi[q2]]
##
## @brief basic cost function that returns distance of two virtual qubits q1 and q2 
## @param pi mapping between virtual qubits q and physical qubits Q
## @param SWAP virtual qubits to be swapped
    def update_pi(self, pi, SWAP):
        q1,q2 = SWAP
        placeholder = pi[q2]
        pi[q2] = pi[q1]
        pi[q1] = placeholder

##
## @brief returns inverse pi mapping physical qubits Q to virtual qubits q
## @param pi mapping between virtual qubits q and physical qubits Q
    def get_inverse_pi(self, pi):
        inverse_pi = list(range(len(pi)))
        for i in range(len(pi)):
            Q = pi[i]
            inverse_pi[Q] = i 
        return inverse_pi

##
## @brief Get all possible swaps between all involved qubits in F
## @breif F front layer containing unresolved gates 
## @param pi mapping between virtual qubits q and physical qubits Q
## @param DAG DAG of circuit
    def obtain_SWAPS(self, F, pi, DAG):
        swaps = []
        inverse_pi = self.get_inverse_pi(pi)
        involved_qbits = []
        for gate_idx in F:
            gate = DAG[gate_idx][0]
            q_control = gate.get_Control_Qbit()
            q_target = gate.get_Target_Qbit()
            if q_control == -1:
                continue
            if q_control not in involved_qbits:
                involved_qbits.append(q_control)
            if q_target not in involved_qbits:
                involved_qbits.append(q_target)
        for qbit in involved_qbits:
            candidates = self.neighbours[pi[qbit]]
            for Q_cand in candidates:
                swap = [min(qbit,inverse_pi[Q_cand]),max(qbit,inverse_pi[Q_cand])]
                if swap not in swaps:
                    swaps.append(swap)
        return swaps
        
##
## @brief Generates lookahead extended layer containing all upcoming two qubit gates
## @param F front layer containing unresolved gates
## @param DAG dag to walk
    def generate_E(self,F,DAG,IDAG,resolved_gates):
        E = []
        for front_gate in F:
            if len(E) < self.max_E_size:
                involved_qbits = [DAG[front_gate][0].get_Target_Qbit(),DAG[front_gate][0].get_Control_Qbit()]
                lookahead_count = 0
                children = list(DAG[front_gate][1]).copy()
                while (lookahead_count<self.max_lookahead and len(children)>0):
                    for gate_idx in children.copy():
                        gate = DAG[gate_idx][0]
                        if gate.get_Control_Qbit() == -1:
                            children.extend(DAG[gate_idx][1])
                        else:
                            gate_depth = self.calculate_gate_depth(gate_idx,IDAG,resolved_gates)
                            if len(E)< self.max_E_size and (gate.get_Target_Qbit() in involved_qbits or gate.get_Control_Qbit() in involved_qbits) and gate_idx not in E and gate_depth<6:
                                E.append(gate_idx)
                                lookahead_count +=1
                                if lookahead_count<self.max_lookahead:
                                    children.extend(DAG[gate_idx][1])
                        children.remove(gate_idx)
        return E
        
        
##
## @brief Calculates the number of predecessing unresolved gates for a gate 
## @param gate_idx gate to be calculated "distance for"
## @param DAG circuit DAG
## @param resolved_gates indicates whether gate at index is resolved or not
## @param F front layer 
    def calculate_gate_depth(self,gate_idx,IDAG,resolved_gates):
        depth = 1
        gate = IDAG[gate_idx][0]
        parents = list(IDAG[gate_idx][1])
        while (len(parents)>0):
            for parent_idx in parents.copy():
                parent_gate = IDAG[parent_idx][0]
                if resolved_gates[parent_idx]==False:
                    if parent_gate.get_Control_Qbit() != -1: 
                        depth += 1
                    parents.extend(IDAG[parent_idx][1])
                parents.remove(parent_idx)
        return depth
##
## @brief Checks if gate dependencies have been resolved
## @param gate_idx gate to check
## @param IDAG inverse DAG of circuit
## @param resolved_gates indicates whether gate at index is resolved or not
    def check_dependencies(self,gate_idx,IDAG,resolved_gates):
        resolved = True
        gate = IDAG[gate_idx][0]
        parents = list(IDAG[gate_idx][1])
        while (len(parents)>0 and resolved!=False):
            for parent_idx in parents.copy():
                parent_gate = IDAG[parent_idx][0]
                resolved *= resolved_gates[parent_idx]
                parents.extend(IDAG[parent_idx][1])
                parents.remove(parent_idx)
        return resolved
    

##
## @brief Heuristic search to map circuit to topology based on initial mapping pi this is an implementation of Algorithm 1 as seen in: https://arxiv.org/pdf/1809.02573
## @param F front layer containing unresolved gates 
## @param pi mapping between virtual qubits q and physical qubits Q
## @param DAG DAG of circuit
## @param IDAG inverse DAG of circuit

    def Heuristic_search(self,F,pi,DAG,IDAG):
        swap_count = 0
        resolved_gates = [False]*len(DAG)
        flags = [0]*len(DAG)
        swap_list = []
        gate_order = []
        E = self.generate_E(F,DAG,IDAG,resolved_gates)
        swap_prev = []
        while len(F)!=0:
            execute_gate_list = []
            for gate_idx in F:
                gate = DAG[gate_idx][0]
                possible, flag = self.is_gate_possible(pi,gate.get_Target_Qbit(),gate.get_Control_Qbit()) 
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
                        resolved = self.check_dependencies(new_gate_idx,IDAG,resolved_gates)
                        if resolved and new_gate_idx not in F:
                            F.append(new_gate_idx)
                E = self.generate_E(F,DAG,IDAG,resolved_gates)

                continue
            else:
                scores = []
                swaps = self.obtain_SWAPS(F,pi,DAG)
                if len(swaps)!=0:
                    for SWAP_idx in range(len(swaps)):
                        SWAP = swaps[SWAP_idx]
                        if SWAP == swap_prev:
                            scores.append(np.inf)
                            continue
                        pi_temp = pi.copy()
                        self.update_pi(pi_temp,SWAP)
                        score = 0
                        for gate_idx in F: 
                            gate = DAG[gate_idx][0]
                            q_target, q_control = gate.get_Target_Qbit(),gate.get_Control_Qbit()
                            score += self.H_basic(pi_temp,q_target, q_control)
                        score *= 1/len(F)
                        if len(E) != 0:
                            score_temp = 0
                            for gate_idx in E:
                                gate = DAG[gate_idx][0]
                                q_target, q_control = gate.get_Target_Qbit(),gate.get_Control_Qbit()
                                score_temp += self.H_basic(pi_temp,q_target, q_control)*(self.alpha**self.calculate_gate_depth(gate_idx,IDAG,resolved_gates))
                            score_temp *= self.W/len(E)
                            score+=score_temp
                        scores.append(score)
                    min_idx = np.argmin(np.array(scores))
                    min_swap = swaps[min_idx]
                    swap_prev = min_swap
                    gate_order.append(min_swap)
                    self.update_pi(pi,min_swap)
                    swap_count +=1
                else:
                    print("ERORR: circuit cannot be mapped please check topology")
                    break
        return gate_order,flags,swap_count
   
##
## @brief Returns mapped circuit on physical qubits Q with swaps included 
## @param circuit circuit to be mapped 
## @param init_pi intial mapping between virtual qubits q and physical qubits Q
## @param gate_order order of gates to be excecuted on physical hardware (swaps included)
## @param flags sign if flipping target and control in gate is necessary 
## @param parameters parameters belonging to the initial circuit
    def get_mapped_circuit(self,circuit,init_pi,gate_order,flags,parameters):
        circuit_mapped = Circuit(self.qbit_num)
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
        
##
## @brief Returns mapped circuit on physical qubits Q with swaps included 
## @param parameters parameters belonging to initial circuit defined in init 
## @param iter_num number of iterations to execute during reversal traversal to find optimal mapping, for more see at: https://arxiv.org/pdf/1809.02573
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
            self.Heuristic_search(first_layer.copy(),init_pi,DAG,IDAG)
            self.Heuristic_search(first_layer_reverse.copy(),init_pi,DAG_reverse,IDAG_reverse)
        final_pi = init_pi.copy()
        gate_order,flags,swap_count = self.Heuristic_search(first_layer.copy(),final_pi,DAG,IDAG)
        circuit_mapped,parameters_new = self.get_mapped_circuit(circuit,init_pi.copy(),gate_order,flags,parameters)
        return circuit_mapped,parameters_new,init_pi,final_pi,swap_count
