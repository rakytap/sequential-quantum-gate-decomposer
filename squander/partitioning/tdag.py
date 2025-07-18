# TDAG (Tree-based Directed Acyclic Graph)
from squander.gates.gates_Wrapper import CNOT, CH, CZ, CRY
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from collections import deque

def get_qubits(gate):
    return ({gate.get_Target_Qbit(), gate.get_Control_Qbit()}
            if isinstance(gate, (CH, CRY, CNOT, CZ)) else {gate.get_Target_Qbit()})

def _build_dependency(c):
    gate_dict = {i: gate for i, gate in enumerate(c.get_Gates())}
    g, rg = {i: set() for i in gate_dict}, {i: set() for i in gate_dict}
    for gate in gate_dict:
        for child in c.get_Children(gate_dict[gate]):
            g[gate].add(child)
            rg[child].add(gate)
    return gate_dict, g, rg

def _build_tree(root, g, vis, gate_dict, max_qubit): # BFS
    tree, q, local_vis, tree_qubits = [], deque([root]), set(), set()
    while q:
        curr = q.popleft()
        if curr in vis or curr in local_vis:
            continue
        # check if adding this gate would exceed qubit limit
        gate_qubits = get_qubits(gate_dict[curr])
        if len(tree_qubits | gate_qubits) > max_qubit:
            continue
        # add gate to tree
        tree.append(curr)
        local_vis.add(curr)
        tree_qubits.update(gate_qubits)
        # add children to queue for exploration
        for child in g[curr]:  # Use forward dependencies
            if child not in vis and child not in local_vis:
                q.append(child)
    # mark all gates in this tree as visited
    vis.update(local_vis)
    return tree, tree_qubits

def _score_tree(tree, tree_qubits, max_qubit):
    ng, nq = len(tree), len(tree_qubits) # num gates, num qubits
    # Formula: base_score (gate count) + bonus (gates per qubit) + bonus (reward for using more qubits)
    return ng * 100 + (ng / nq) * 10 + (nq / max_qubit) * 20 if nq else 0

def _greedy_tree_selection(scored_trees, max_qubit):
    parts, used_gates = [], set()
    while scored_trees:
        part_gates, part_qubits, remaining = set(), set(), []
        for score, tree, tree_qubits in scored_trees:
            tree_set = set(tree)
            # Check if tree can fit in current partition
            if not tree_set & used_gates and len(part_qubits | tree_qubits) <= max_qubit:
                part_gates.update(tree_set)
                part_qubits.update(tree_qubits)
                used_gates.update(tree_set)
            else:
                remaining.append((score, tree, tree_qubits))
        if part_gates:
            parts.append(part_gates)
        elif remaining:
            # Force add highest scoring tree if no progress
            tree_set = set(remaining.pop(0)[1])
            if not tree_set & used_gates:
                parts.append(tree_set)
                used_gates.update(tree_set)
        scored_trees = remaining
    return parts, used_gates

def _handle_remaining(remaining_gates, gate_dict, max_qubit):
    parts, rem_gates = [], list(remaining_gates)
    while rem_gates:
        part_gates, part_qubits, i = set(), set(), 0
        while i < len(rem_gates):
            gate_id = rem_gates[i]
            gate_qubits = get_qubits(gate_dict[gate_id])
            if len(part_qubits | gate_qubits) <= max_qubit:
                part_gates.add(gate_id)
                part_qubits.update(gate_qubits)
                rem_gates.pop(i)
            else:
                i += 1
        if part_gates:
            parts.append(part_gates)
        elif rem_gates:
            # force add at least one gate to avoid infinite loop
            gate_id = rem_gates.pop(0)
            parts.append({gate_id})
    return parts

def tdag_max_partitions(c, max_qubit):
    gate_dict, g, rg = _build_dependency(c)
    if not gate_dict:
        return c, [], []
    # Build and score trees
    roots = [i for i in gate_dict if not rg[i]]  # (gates with no dependencies
    trees, vis = [], set()
    for root in roots: # build tree from roots
        if root not in vis:
            tree, tree_qubits = _build_tree(root, g, vis, gate_dict, max_qubit)
            if tree:
                score = _score_tree(tree, tree_qubits, max_qubit)
                trees.append((score, tree, tree_qubits))
    trees.sort(reverse=True, key=lambda x: x[0]) # sort trees by score

    parts, used_gates = _greedy_tree_selection(trees, max_qubit)

    # handle remaining gates
    remaining = set(range(len(gate_dict))) - used_gates
    if remaining:
        parts.extend(_handle_remaining(remaining, gate_dict, max_qubit))

    # ordered_parts = [set(part) for part in parts]
    # from squander.partitioning.partition import kahn_partition
    # return kahn_partition(c, max_qubit, ordered_parts)

    # build circuit
    top_circuit = Circuit(c.get_Qbit_Num())
    param_order = []
    curr_idx = 0
    for part in parts:
        subcircuit = Circuit(c.get_Qbit_Num())
        part_rg = {i: sum(1 for parent in rg[i] if parent in part) for i in part}
        queue = [gate_id for gate_id, deps in part_rg.items() if deps == 0]
        while queue:
            gate_id = queue.pop(0)
            gate = gate_dict[gate_id]
            subcircuit.add_Gate(gate)

            param_order.append((
                gate.get_Parameter_Start_Index(),
                curr_idx,
                gate.get_Parameter_Num()
            ))
            curr_idx += gate.get_Parameter_Num()
            
            # update dependencies within partition
            for child in g[gate_id]:
                if child in part:
                    part_rg[child] -= 1
                    if part_rg[child] == 0:
                        queue.append(child)
        
        top_circuit.add_Circuit(subcircuit)
    
    return top_circuit, param_order, [list(part) for part in parts]