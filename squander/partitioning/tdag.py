# TDAG (Tree-based Directed Acyclic Graph)
# https://www.osti.gov/servlets/purl/1985363
# https://dl.acm.org/doi/10.1145/3583781.3590234
# https://arxiv.org/pdf/2410.02901
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.tools import build_dependency

def _get_starting_gates(g, rg, gate_to_qubit, S):
    """
    Get starting gates for each qubit in the circuit

    Args:
        
        g: Forward dependency graph
        
        rg: Reverse dependency graph
        
        gate_to_qubit: Mapping of gates to qubits
        
        S: Set of initial gates

    Returns:
        
        Dictionary mapping qubits to their starting gates
    """
    g = { x: set(y) for x, y in g.items() }
    rg = { x: set(y) for x, y in rg.items() }
    S = set(S)
    start_qubit = {}
    while S:
        n = S.pop()
        assert not rg[n]
        for m in set(g[n]):
            g[n].remove(m)
            rg[m].remove(n)
            if not rg[m]:
                S.add(m)
        for qubit in gate_to_qubit[n]:
            if not qubit in start_qubit: start_qubit[qubit] = {}
            start_qubit[qubit][n] = None
    return start_qubit

def tdag_max_partitions(c, max_qubit, use_gtqcp=False):
    """
    Partition a circuit using TDAG algorithm

    Args:
        
        c: SQUANDER Circuit to partition
        
        max_qubit: Max qubits per partition
        
        use_gtqcp: Use GTQCP variant

    Returns:
       
        Partitioned circuit, parameter order (source_idx, dest_idx, param_count), partition assignments
    """
    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)
    L = []
    start_qubit = _get_starting_gates(g, rg, gate_to_qubit, S)
    while g:
        groups, deps = _enumerate_groups(g, rg, gate_to_qubit, start_qubit, max_qubit, use_gtqcp)
        L.append(_remove_best_partition(groups, g, rg, gate_to_qubit, start_qubit, deps))
    assert sum(len(x) for x in L) == len(gate_dict), (sum(len(x) for x in L), len(gate_dict))
    from squander.partitioning.kahn import kahn_partition_preparts
    return kahn_partition_preparts(c, max_qubit, preparts=L)


def _get_gate_dependencies(g, rg, gate_to_qubit, S, max_qubit):
    """
    Get gate dependencies for each gate within qubit constraints

    Args:
        
        g: Forward dependency graph
        
        rg: Reverse dependency graph
        
        gate_to_qubit: Mapping of gates to qubits
        
        S: Starting gates per qubit
        
        max_qubit: Max qubits per partition

    Returns:
        
        Dictionary of gate dependencies
    """
    deps, visited = {}, set()
    level, next_level = set(), set()
    for qubit in S:
        level.add(next(iter(S[qubit])))
    while level:
        for curr_gate in level:
            if not rg[curr_gate] <= visited: continue            
            deps[curr_gate] = set.union(gate_to_qubit[curr_gate], *(deps[gate] for gate in rg[curr_gate]))
            if len(deps[curr_gate]) <= max_qubit:
                next_level |= g[curr_gate]
                visited.add(curr_gate)
            else: del deps[curr_gate]
        level, next_level = next_level, level
        next_level.clear()
    return deps


def _enumerate_groups(g, rg, gate_to_qubit, S, max_qubit, use_gtqcp):
    """
    Enumerate possible gate groups for partitioning

    Args:
        
        g: Forward dependency graph
        
        rg: Reverse dependency graph
        
        gate_to_qubit: Mapping of gates to qubits
        
        S: Starting gates per qubit
        
        max_qubit: Max qubits per partition
        
        use_gtqcp: Use GTQCP variant

    Returns:
       
        Set of gate groups and their dependencies
    """
    deps = _get_gate_dependencies(g, rg, gate_to_qubit, S, max_qubit)
    result = set()
    func = _enumerate if not use_gtqcp else _enumerate_gtqcp
    for qubit in S:
        result |= func(next(iter(S[qubit])), qubit, {qubit} if use_gtqcp else {frozenset({qubit})}, deps, g, gate_to_qubit, S, max_qubit)
    return result, deps


def _enumerate(target_gate, target_qubit, input_groups, deps, g, gate_to_qubit, S, max_qubit):
    """
    Recursively enumerate gate groups for a target gate

    Args:
        
        target_gate: Gate to start enumeration
        
        target_qubit: Qubit index
        
        input_groups: Current qubit groups
        
        deps: Gate dependencies
        
        g: Forward dependency graph
        
        gate_to_qubit: Mapping of gates to qubits
    
        S: Starting gates per qubit
        
        max_qubit: Max qubits per partition

    Returns:
        
        Set of valid qubit groups
    """
    output, result = set(), set() # qubit groups
    if not target_gate in deps: return result
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
    """
    Recursively enumerate gate groups for a target gate using GTQCP variant.

    Args:
        
        target_gate: Gate to start enumeration
        
        target_qubit: Qubit index
        
        input_groups: Current qubit groups
        
        deps: Gate dependencies
        
        g: Forward dependency graph
        
        gate_to_qubit: Mapping of gates to qubits
        
        S: Starting gates per qubit
        
        max_qubit: Max qubits per partition

    Returns:
        
        Set of valid qubit groups
    """
    result = set()
    if not target_gate in deps: return result
    gate = target_gate
    while True:
        next_gate = next(iter(x for x in g[gate] if target_qubit in gate_to_qubit[x]), None)
        if next_gate is None or next_gate not in deps or len(input_groups | deps[next_gate]) > max_qubit:
            break
        gate = next_gate
    group = frozenset(input_groups) | frozenset(deps[gate])
    if len(group) > max_qubit:
        return result
    if group not in result:
        result.add(group)
        if len(group) < max_qubit:
            for qubit in group - input_groups:
                if qubit not in S:
                    continue
                gate = next(iter(S[qubit]))
                _enumerate_gtqcp(gate, qubit, input_groups | frozenset({qubit}), deps, g, gate_to_qubit, S, max_qubit)
    return result


def _remove_best_partition(qubit_results, g, rg, gate_to_qubit, start_qubit, deps):
    """
    Remove the best partition from the dependency graphs

    Args:
        
        qubit_results: Candidate qubit groups
        
        g: Forward dependency graph
        
        rg: Reverse dependency graph
        
        gate_to_qubit: Mapping of gates to qubits
        
        start_qubit: Starting gates per qubit
        
        deps: Gate dependencies

    Returns:
        
        List of gate indices in the best partition
    """
    gate_info = [[x for x, y in deps.items() if y <= result] for result in qubit_results]
    best_part = max(gate_info, key=lambda x: len(x))
    for gate in best_part:
        for child in g[gate]:
            rg[child].remove(gate)
        for child in rg[gate]:
            g[child].remove(gate)
    for gate in best_part:
        del g[gate]
        del rg[gate]
        for qubit in gate_to_qubit[gate]:
            del start_qubit[qubit][gate]
            if len(start_qubit[qubit]) == 0: del start_qubit[qubit]

    return best_part

def _test_tdag_qasm(use_gtqcp=False):
    """
    Test TDAG partitioning on a QASM file

    Args:
        
        use_gtqcp: Use GTQCP variant
    """
    K = 3
    # filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm" 
    filename = "benchmarks/partitioning/test_circuit/9symml_195.qasm"
    from squander import utils
    
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    partition = tdag_max_partitions(circ, K, use_gtqcp)

    print(partition[2])


def _test_tdag_single_qubit():
    """
    Test TDAG partitioning on a simple two-qubit circuit
    """
    K = 4

    c = Circuit(2)
    c.add_H(0)
    c.add_H(1)
    c.add_CNOT(0, 1)

    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)

    result = _get_gate_dependencies(g, rg, gate_to_qubit, S, K)

    # print("gate_dependencies: ", result)

    _enumerate_groups(g, gate_to_qubit, S, K, False)

    partition = tdag_max_partitions(c, K)

    # print(partition)

def _test_gtqcp():
    """
    Test GTQCP variant for TDAG partitioning
    """
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
    
    start_qubit = _get_starting_gates(g, rg, gate_to_qubit, S)

    result = _get_gate_dependencies(g, rg, gate_to_qubit, start_qubit, K)

    assert result == expected, (result, expected)

    result_groups, _ = _enumerate_groups(g, rg, gate_to_qubit, start_qubit, K, True)

    assert result_groups == expected_groups, (result_groups, expected_groups)

    partition = tdag_max_partitions(c, K)

    print(partition[2])


def _test_tdag():
    """
    Test TDAG partitioning on a sample circuit
    """
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

    start_qubit = _get_starting_gates(g, rg, gate_to_qubit, S)

    result = _get_gate_dependencies(g, rg, gate_to_qubit, start_qubit, K)

    assert result == expected, (result, expected)

    result_groups, _ = _enumerate_groups(g, rg, gate_to_qubit, start_qubit, K, False)

    assert result_groups == expected_groups, (result_groups, expected_groups)

    partition = tdag_max_partitions(c, K)

    print(partition[2])



if __name__ == "__main__":
    _test_tdag()
    _test_gtqcp()
    _test_tdag_qasm(False)
    _test_tdag_qasm(True)
