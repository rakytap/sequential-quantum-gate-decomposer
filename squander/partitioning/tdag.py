# TDAG (Tree-based Directed Acyclic Graph)
# https://www.osti.gov/servlets/purl/1985363
# https://dl.acm.org/doi/10.1145/3583781.3590234
# https://arxiv.org/pdf/2410.02901
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.tools import build_dependency

def _get_topo_order(g, rg, S):
    g = { x: set(y) for x, y in g.items() }
    rg = { x: set(y) for x, y in rg.items() }
    S = set(S)
    L = []
    while S:
        n = S.pop()
        assert not rg[n]
        for m in set(g[n]):
            g[n].remove(m)
            rg[m].remove(n)
            if not rg[m]:
                S.add(m)
        L.append(n)
    return L

def _get_starting_gates(L, g, gate_to_qubit):
    start_qubit = {}
    for gate in L:    
        if gate not in g:
            continue
        for qubit in gate_to_qubit[gate]:
            if qubit not in start_qubit:
                start_qubit[qubit] = gate
    return start_qubit


def tdag_max_partitions(c, max_qubit, use_gtqcp=False):
    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)
    L = []
    topo = _get_topo_order(g, rg, S)
    start_qubit = _get_starting_gates(topo, g, gate_to_qubit)

    while len(g) != 0:
        groups = _enumerate_groups(c, g, gate_to_qubit, start_qubit, max_qubit, use_gtqcp)
        L.append(_remove_best_partition(groups, g, rg, gate_to_qubit, S, start_qubit, topo))
    assert sum(len(x) for x in L) == len(gate_dict), (sum(len(x) for x in L), len(gate_dict))
    from squander.partitioning.kahn import kahn_partition_preparts
    return kahn_partition_preparts(c, max_qubit, preparts=L)


def _get_gate_dependencies(c, g, gate_to_qubit, S, max_qubit):
    deps = {x : set(gate_to_qubit[x]) for x in g}
    next_level = set()
    for qubit in range(c.get_Qbit_Num()):
        if qubit not in S:
            continue
        next_level.add(S[qubit])
    while next_level:
        level = next_level
        next_level = set()
        for curr_gate in level:
            for gate in g[curr_gate]:
                if len(deps[gate]) > max_qubit:
                    continue
                deps[gate] |= deps[curr_gate]
                next_level.add(gate)
    return deps


def _enumerate_groups(c, g, gate_to_qubit, S, max_qubit, use_gtqcp):
    deps = _get_gate_dependencies(c, g, gate_to_qubit, S, max_qubit)
    result = set()
    func = _enumerate if not use_gtqcp else _enumerate_gtqcp
    for qubit in range(c.get_Qbit_Num()):
        if qubit not in S:
            continue
        result |= func(S[qubit], qubit, {qubit} if use_gtqcp else {frozenset({qubit})}, deps, g, gate_to_qubit, S, max_qubit)
    return result


def _enumerate(target_gate, target_qubit, input_groups, deps, g, gate_to_qubit, S, max_qubit):
    output, result = set(), set() # qubit groups
    for group in input_groups:
        qubits = frozenset.union(group, deps[target_gate])
        if len(qubits) <= max_qubit:
            result.add(qubits)
            if not deps[target_gate] <= group: # is not subset
                output.add(qubits) 
    
    while any(len(group) < max_qubit for group in output):
        target_gate = next(iter(x for x in g[target_gate] if target_qubit in gate_to_qubit[x]), None)
        if target_gate is None:
            break
        input_groups = output
        next_qubit = next(iter(gate_to_qubit[target_gate] - {target_qubit}), None)
        if next_qubit is None:
            continue
        output = _enumerate(target_gate, next_qubit, input_groups, deps, g, gate_to_qubit, S, max_qubit)
        result |= output
    
    return result

def _enumerate_gtqcp(target_gate, target_qubit, input_groups, deps, g, gate_to_qubit, S, max_qubit):
    result = set()
    gate = target_gate
    while True:
        next_gate = next(iter(x for x in g[gate] if target_qubit in gate_to_qubit[x]), None)
        if next_gate is None or len(input_groups | deps[next_gate]) > max_qubit:
            break
        gate = next_gate
    if len(input_groups | deps[gate]) > max_qubit:
        return result
    group = frozenset(input_groups) | frozenset(deps[gate])
    if group not in result:
        result.add(group)
        if len(group) < max_qubit:
            for qubit in group - input_groups:
                if qubit not in S:
                    continue
                _enumerate_gtqcp(S[qubit], qubit, input_groups | frozenset({qubit}), deps, g, gate_to_qubit, S, max_qubit)
    return result


def _remove_best_partition(qubit_results, g, rg, gate_to_qubit, S, start_qubit, topo):
    gate_info = []
    for result in qubit_results:
        pos_gates, gates = set(S), set()
        while True:
            t = {x for x in pos_gates if gate_to_qubit[x] <= result}
            if len(t) == 0:
                break
            gates |= t
            pos_gates |= {child for n in t for child in g[n] if not (rg[child] - gates)}
            pos_gates -= t
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
    
    S.clear()
    for m in rg:
        if len(rg[m]) == 0:
            S.add(m)

    start_qubit.clear()
    start_qubit.update( _get_starting_gates(topo, g, gate_to_qubit) )

    return best_part

def _test_tdag_qasm(use_gtqcp=False):
    K = 3
    # filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm" 
    filename = "benchmarks/partitioning/test_circuit/cm42a_207_squander.qasm"
    from squander import utils
    
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    partition = tdag_max_partitions(circ, K, use_gtqcp)

    print(partition[2])


def _test_tdag_single_qubit():
    K = 4

    c = Circuit(2)
    c.add_H(0)
    c.add_H(1)
    c.add_CNOT(0, 1)

    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)

    result = _get_gate_dependencies(c, g, gate_to_qubit, S, K)

    # print("gate_dependencies: ", result)

    _enumerate_groups(c, g, gate_to_qubit, S, K, False)

    partition = tdag_max_partitions(c, K)

    # print(partition)

def _test_gtqcp():
    K = 5

    c = Circuit(6)
    c.add_CNOT(0, 1)
    c.add_CNOT(2, 3)
    c.add_CNOT(3, 4)
    c.add_CNOT(2, 3)
    c.add_CNOT(4, 5)
    c.add_CNOT(1, 2)

    expected = {0: {0, 1}, 1: {2, 3}, 2: {2, 3, 4}, 3: {2, 3, 4}, 4: {2, 3, 4, 5}, 5: {0, 1, 2, 3, 4}}
   
    expected_groups = {frozenset({2, 3, 4}), frozenset({0, 1, 2, 3, 4}), frozenset({0, 1}), frozenset({2, 3, 4, 5})} # we don't consider starting qubits in the middle of topological sort
    
    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)
    
    topo = _get_topo_order(g, rg, S)
    start_qubit = _get_starting_gates(topo, g, gate_to_qubit)

    result = _get_gate_dependencies(c, g, gate_to_qubit, start_qubit, K)

    assert result == expected, (result, expected)

    result_groups = _enumerate_groups(c, g, gate_to_qubit, start_qubit, K, True)

    assert result_groups == expected_groups, (result_groups, expected_groups)

    partition = tdag_max_partitions(c, K)

    print(partition[2])


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

    
    expected = {0: {3, 4}, 1: {2, 3, 4}, 2: {3, 4, 5}, 3: {1, 2, 3, 4}, 4: {3, 4, 5, 6}, 5: {0, 2, 3, 4}, 6: {3, 4, 5, 7}}
   
    expected_groups = {frozenset({3, 4}), frozenset({3, 4, 5, 6}), frozenset({2, 3, 4}), frozenset({0, 2, 3, 4}), frozenset({1, 2, 3, 4}), frozenset({3, 4, 5}), frozenset({3, 4, 5, 7})}
    
    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)

    topo = _get_topo_order(g, rg, S)
    start_qubit = _get_starting_gates(topo, g, gate_to_qubit)

    result = _get_gate_dependencies(c, g, gate_to_qubit, start_qubit, K)

    assert result == expected, (result, expected)

    result_groups = _enumerate_groups(c, g, gate_to_qubit, start_qubit, K, False)

    assert result_groups == expected_groups, (result_groups, expected_groups)

    partition = tdag_max_partitions(c, K)

    print(partition[2])



if __name__ == "__main__":
    _test_tdag()
    _test_gtqcp()
    _test_tdag_qasm(False)
    _test_tdag_qasm(True)