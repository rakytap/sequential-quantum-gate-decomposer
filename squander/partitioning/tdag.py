# TDAG (Tree-based Directed Acyclic Graph)
# https://www.osti.gov/servlets/purl/1985363
# https://dl.acm.org/doi/10.1145/3583781.3590234
# https://arxiv.org/pdf/2410.02901
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.tools import get_qubits, build_dependency



def tdag_max_partitions(c, max_qubit, use_gtqcp=False):
    gate_dict, g, rg = build_dependency(c)
    L = []
    while len(g) != 0:
        groups = _enumerate_groups(c, gate_dict, g, rg, max_qubit, use_gtqcp)
        L.append(_remove_best_partition(groups, gate_dict, g, rg))
    from squander.partitioning.kahn import kahn_partition_preparts
    return kahn_partition_preparts(c, max_qubit, preparts=L)



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


def _enumerate_groups(c, gate_dict, g, rg, max_qubit, use_gtqcp):
    deps, rev_deps = _get_gate_dependencies(c, gate_dict, g, rg, max_qubit)
    result = []
    func = _enumerate if not use_gtqcp else _enumerate_gtqcp
    for qubit in range(c.get_Qbit_Num()):
        if len(rev_deps[qubit]) == 0:
            continue
        for target_gate in rev_deps[qubit]:
            result.append(func(target_gate, qubit, {frozenset({qubit})}, deps, rev_deps, gate_dict, max_qubit))
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
            # next_qubit = next(iter(get_qubits(gate_dict[next_gate]) - {target_qubit}))
            for next_qubit in get_qubits(gate_dict[next_gate]) - {target_qubit}:                
                input_groups = output
                output = _enumerate(next_gate, next_qubit, input_groups, deps, rev_deps, gate_dict, max_qubit)
                result |= output
    return result


def _enumerate_gtqcp(target_gate, target_qubit, input_groups, deps, rev_deps, gate_dict, max_qubit):
    for next_gate in rev_deps[target_qubit]:
        if next_gate == target_gate:
            continue
        if len(input_groups | deps[next_gate]) > max_qubit:
            break
        target_gate = next_gate
    group = frozenset(input_groups) | frozenset(deps[target_gate])
    result = set()
    if group not in result:
        result.add(group)
        if len(group) < max_qubit:
            for qubit in group - input_groups:
                for next_gate in rev_deps[qubit]:
                    _enumerate_gtqcp(next_gate, qubit, input_groups | frozenset({qubit}), deps, rev_deps, gate_dict, max_qubit)
    return result


def _remove_best_partition(qubit_results, gate_dict, g, rg):
    S = {m for m in rg if len(rg[m]) == 0}
    gate_info = []
    for result in (y for x in qubit_results for y in x):
        pos_gates, gates = set(S), set()
        while True:
            t = {x for x in pos_gates if get_qubits(gate_dict[x]) <= result}
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

    return best_part

def _test_tdag_qasm():
    K = 4
    filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    from squander import utils
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    partition = tdag_max_partitions(circ, K, True)


def _test_tdag_single_qubit():
    K = 4

    c = Circuit(2)
    c.add_H(0)
    c.add_H(1)
    c.add_CNOT(0, 1)

    gate_dict, g, rg = build_dependency(c)

    result = _get_gate_dependencies(c, gate_dict, g, rg, K)

    # print("gate_dependencies: ", result)

    _enumerate_groups(c, gate_dict, g, rg, K)

    partition = tdag_max_partitions(c, K)

    # print(partition)


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

    # print(partition)



if __name__ == "__main__":
    # _test_tdag()
    _test_tdag_qasm()