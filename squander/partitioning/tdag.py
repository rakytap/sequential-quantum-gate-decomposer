from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.tools import get_qubits, build_dependency



def tdag_max_partitions(c, max_qubit):
    # TDAG (Tree-based Directed Acyclic Graph)
    gate_dict, g, rg = build_dependency(c)
    L = []
    while len(g) != 0:
        groups = _enumerate_groups(c, gate_dict, g, rg, max_qubit)
        L.append(_remove_best_partition(groups, gate_dict, g, rg))
    return L



def _get_gate_dependencies(c, gate_dict, g, rg, max_qubit):
    deps, rev_deps = {x : set() for x in g}, {x : [] for x in range(c.get_Qbit_Num())}
    for qubit in range(c.get_Qbit_Num()):
        gates = [gate for gate in g if qubit in get_qubits(gate_dict[gate]) and not any(qubit in get_qubits(gate_dict[x]) for x in rg[gate]) ]
        _mark_dependency(g, gates, qubit, deps, rev_deps, max_qubit)
    return deps, rev_deps



def _mark_dependency(g, gates, gate_qubit, deps, rev_deps, max_qubit):
    if max_qubit == 0:
        return
    for gate in gates:
        deps[gate].add(gate_qubit)
        rev_deps[gate_qubit].append(gate)
        _mark_dependency(g, g[gate], gate_qubit, deps, rev_deps, max_qubit-1)



def _enumerate_groups(c, gate_dict, g, rg, max_qubit):
    deps, rev_deps = _get_gate_dependencies(c, gate_dict, g, rg, max_qubit)
    result = []
    for qubit in range(c.get_Qbit_Num()):
        if len(rev_deps[qubit]) == 0:
            continue
        target_gate = rev_deps[qubit][0]
        result.append(_enumerate(target_gate, qubit, {frozenset({qubit})}, deps, rev_deps, gate_dict, max_qubit))
    return result



def _enumerate(target_gate, target_qubit, input_groups, deps, rev_deps, gate_dict, max_qubit):
    orig_output, result = set(), set() # qubit groups
    for group in input_groups:
        qubits = frozenset.union(group, deps[target_gate])
        if len(qubits) <= max_qubit:
            result.add(qubits)
            if not deps[target_gate] <= group: # is not subset
                orig_output.add(qubits) 
    
    for next_gate in rev_deps[target_qubit]:
        if next_gate == target_gate:
            continue
        output = orig_output
        for group in output:
            if len(group) > max_qubit:
                continue
            next_qubit = next(iter(get_qubits(gate_dict[next_gate]) - {target_qubit})) 
            input_groups = output
            output = _enumerate(next_gate, next_qubit, input_groups, deps, rev_deps, gate_dict, max_qubit)
            result |= output
    return result



def _remove_best_partition(qubit_results, gate_dict, g, rg):
    S = {m for m in rg if len(rg[m]) == 0}
    gate_info = []
    for result in (y for x in qubit_results for y in x):
        pos_gates, gates = S, set()
        while True:
            t = {x for x in pos_gates if get_qubits(gate_dict[x]) <= result}
            if len(t) == 0:
                break
            pos_gates |= {child for n in t for child in g[n]}
            pos_gates -= t
            gates |= t
        gate_info.append(gates)
    best_part = max(gate_info, key=lambda x: len(x))
    
    for gate in best_part:
        for child in g[gate]:
            rg[child].remove(gate)
        for child in rg[gate]:
            g[child].remove(gate)
    for gate in best_part:
        del g[gate]
        del rg[gate]

    return best_part



def _test_tdag():
    K = 4

    c = Circuit(8)
    c.add_CNOT(3, 4)
    c.add_CNOT(2, 3)
    c.add_CNOT(4, 5)
    c.add_CNOT(1, 2)
    c.add_CNOT(5, 6)
    c.add_CNOT(0, 3)
    c.add_CNOT(4, 7)
    
    expected = ({0: {3, 4}, 1: {2, 3, 4}, 2: {3, 4, 5}, 3: {1, 2, 3, 4}, 4: {3, 4, 5, 6}, 5: {0, 2, 3, 4}, 6: {3, 4, 5, 7}}, {0: [5], 1: [3], 2: [1, 3, 5], 3: [0, 1, 3, 5, 2, 4, 6], 4: [0, 1, 3, 5, 2, 4, 6], 5: [2, 4, 6], 6: [4], 7: [6]})
   
    expected_groups = {frozenset({3, 4}), frozenset({3, 4, 5, 6}), frozenset({2, 3, 4}), frozenset({0, 2, 3, 4}), frozenset({2, 3, 4, 5}), frozenset({1, 2, 3, 4}), frozenset({3, 4, 5}), frozenset({3, 4, 5, 7})}
    
    expected_index = 3

    gate_dict, g, rg = build_dependency(c)

    result = _get_gate_dependencies(c, gate_dict, g, rg, K)

    assert result == expected, (result, expected)

    assert _enumerate_groups(c, gate_dict, g, rg, K)[expected_index] == expected_groups

    partition = tdag_max_partitions(c, K)

    print(partition)



if __name__ == "__main__":
    _test_tdag()