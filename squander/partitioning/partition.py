from squander import Circuit, CNOT, CH, CZ, CRY
from squander import utils
from itertools import dropwhile
import numpy as np


#@brief Retrieves qubit indices used by a SQUANDER gate
#@param gate The SQUANDER gate
#@return Set of qubit indices used by the gate
def get_qubits(gate):
    return ({gate.get_Target_Qbit(), gate.get_Control_Qbit()}
            if isinstance(gate, (CH, CRY, CNOT, CZ)) else {gate.get_Target_Qbit()})





#@brief Partitions a flat circuit into subcircuits using Kahn's algorithm
#@param c The SQUANDER circuit to be partitioned
#@param max_qubit Maximum number of qubits allowed per partition
#@param preparts Optional prefedefined partitioning scheme
#@return Tuple: 
#   - Partitioned 2-level circuit
#   - Tuples specifying new parameter positions: source_idx, dest_idx, param_count
#   - Partition assignments
def kahn_partition(c, max_qubit, preparts=None):
    top_circuit = Circuit(c.get_Qbit_Num())

    # Build dependency graphs
    gate_dict = {i: gate for i, gate in enumerate(c.get_Gates())}
    g, rg = { i: set() for i in gate_dict }, { i: set() for i in gate_dict }

    for gate in gate_dict:
        for child in c.get_Children(gate_dict[gate]):
            g[gate].add(child)
            rg[child].add(gate)
    
    L, S = [], {m for m in rg if len(rg[m]) == 0}
    param_order = []

    def partition_condition(gate):
        return len(get_qubits(gate_dict[gate]) | curr_partition) > max_qubit
    
    c = Circuit(c.get_Qbit_Num())
    curr_partition = set()
    curr_idx = 0
    total = 0
    parts = [[]]

    while S:
        if preparts is None:
            n = next(dropwhile(partition_condition, S), None)
        else:
            n = next(iter(S & preparts[len(parts)-1]), None)
            assert (n is None) == (len(preparts[len(parts)-1]) == len(parts[-1])) #sanity check valid partitioning

        if n is None:  # partition cannot be expanded
            # Add partition to circuit
            top_circuit.add_Circuit(c)
            total += len(c.get_Gates())
            # Reset for next partition
            curr_partition = set()
            c = Circuit(c.get_Qbit_Num())
            parts.append([])
            n = next(iter(S)) if preparts is None else next(iter(S & preparts[len(parts)-1]))


        # Add gate to current partition
        parts[-1].append(n)
        curr_partition |= get_qubits(gate_dict[n])
        c.add_Gate(gate_dict[n])
        param_order.append((
            gate_dict[n].get_Parameter_Start_Index(), 
            curr_idx, 
            gate_dict[n].get_Parameter_Num()
        ))
        curr_idx += gate_dict[n].get_Parameter_Num()

        # Update dependencies
        L.append(n)
        S.remove(n)
        assert len(rg[n]) == 0
        for child in set(g[n]):
            g[n].remove(child)
            rg[child].remove(n)
            if not rg[child]:
                S.add(child)

    # Add the last partition
    top_circuit.add_Circuit(c)
    total += len(c.get_Gates())
    assert total == len(gate_dict)
    return top_circuit, param_order, parts


#@brief Finds the next biggest optimal partition using ILP
#@param c The SQUANDER circuit to be partitioned
#@param max_qubits_per_partition Maximum qubits allowed per partition
#@param prevparts Previously determined partitions to exclude
#@return gates Set of gate indices forming next biggest partition
def find_next_biggest_partition(c, max_qubits_per_partition, prevparts=None):
    import pulp
    gatedict = {i: gate for i, gate in enumerate(c.get_Gates())}
    all_qubits = set(range(c.get_Qbit_Num()))

    num_gates = len(gatedict)
    prob = pulp.LpProblem("MaxSinglePartition", pulp.LpMinimize)
    a = pulp.LpVariable.dicts("a", (i for i in range(num_gates)), cat="Binary") #is gate i in previous partition to p
    x = pulp.LpVariable.dicts("x", (i for i in range(num_gates)), cat="Binary") #is gate i in partition p
    z = pulp.LpVariable.dicts("z", (q for q in all_qubits), cat="Binary") #is qubit q in paritition p
    # Constraint 1: not gate is in both a and x
    for i in gatedict:
        prob += x[i] + a[i] <= 1
    # Constraint 2: if gate g_i uses qubit q, then z_{q} = 1
    for i in gatedict:
        gate_qubits = get_qubits(gatedict[i])
        for q in gate_qubits:
            prob += x[i] <= z[q]
    # Constraint 3: the partition uses at most k qubits
    prob += pulp.lpSum(z[q] for q in all_qubits) <= max_qubits_per_partition
    # Constraint 4: previous partitions are filtered out, and are none or all included if in pre-partition
    for part in prevparts:
        for i in part:
            prob += x[i] == 0
            prob += a[i] == a[next(iter(part))]
    # Constraint 5: ordering constraints for dependencies
    for j in gatedict:
        for i in c.get_Children(gatedict[j]): #there exists a topological ordering of all the gates enabled in x
            prob += a[j] >= a[i]
            prob += x[j] + a[j] >= x[i]
    prob.setObjective(-pulp.lpSum(x[i] for i in range(num_gates)))
    from gurobilic import get_gurobi_options
    prob.solve(pulp.GUROBI(manageEnv=True, msg=False, envOptions=get_gurobi_options()))
    #prob.solve(pulp.PULP_CBC_CMD(msg=False))
    gates = {i for i in range(num_gates) if int(pulp.value(x[i]))}
    qubits = set.union(*(get_qubits(gatedict[i]) for i in gates))
    print(f"Status: {pulp.LpStatus[prob.status]}  Found partition with {len(gates)} gates: {gates} and {len(qubits)} qubits: {qubits}")
    return gates


#@brief Partitions a circuit using ILP to maximize gates per partition
#@param c The SQUANDER circuit to be partitioned
#@param max_qubits_per_partition Maximum qubits allowed per partition
#@return Tuple: 
#   - Partitioned 2-level circuit
#   - Tuples specifying new parameter positions: source_idx, dest_idx, param_count
#   - Partition assignments
def ilp_max_partitions(c, max_qubits_per_partition):
    gatedict = {i: gate for i, gate in enumerate(c.get_Gates())}
    num_gates = len(gatedict)
    parts = []
    while sum(len(x) for x in parts) != num_gates:
        parts.append(find_next_biggest_partition(c, max_qubits_per_partition, parts))
    gatedict = {i: gate for i, gate in enumerate(c.get_Gates())}
    partdict = {gate: i for i, part in enumerate(parts) for gate in part}
    g, rg = {i: set() for i in range(len(parts))}, {i: set() for i in range(len(parts))}
    for gate in gatedict:
        for child in c.get_Children(gatedict[gate]):
            if partdict[gate] == partdict[child]: continue
            g[partdict[gate]].add(partdict[child])
            rg[partdict[child]].add(partdict[gate])
    L, S = [], {m for m in rg if len(rg[m]) == 0}
    while len(S) != 0:
        n = S.pop()
        L.append(parts[n])
        assert len(rg[n]) == 0
        for m in set(g[n]):
            g[n].remove(m)
            rg[m].remove(n)
            if len(rg[m]) == 0:
                S.add(m)
    assert len(L) == len(parts)
    return kahn_partition(c, max_qubits_per_partition, L)


#@brief Reorders circuit parameters based on partitioned execution order
#@param params Original parameter array
#@param param_order Tuples specifying new parameter positions: source_idx, dest_idx, param_count
#@return reordered Reordered parameter array
def translate_param_order(params, param_order):
    reordered = np.empty_like(params)
    for s_idx, n_idx, n_params in param_order:
        reordered[n_idx:n_idx + n_params] = params[s_idx:s_idx + n_params]
    return reordered


#@brief Converts a QASM file to a partitioned SQUANDER circuit with reordered parameters
#@param filename Path to the QASM file
#@param max_qubit Maximum qubits allowed per partition
#@param use_ilp Flag to use ILP-based partitioning
#@return Tuple: Partitioned SQUANDER circuit, Reordered parameter array
def qasm_to_partitioned_circuit(filename, max_qubit, use_ilp=False):

    c, param = utils.qasm_to_squander_circuit(filename)
    top_c, param_order, _ = ilp_max_partitions(c, max_qubit) if use_ilp else kahn_partition(c, max_qubit)
    param_reordered = translate_param_order(param, param_order)
    return top_c, param_reordered


if __name__ == "__main__":
    qasm_to_partitioned_circuit("examples/partitioning/qasm_samples/heisenberg-16-20.qasm", 4, True)
