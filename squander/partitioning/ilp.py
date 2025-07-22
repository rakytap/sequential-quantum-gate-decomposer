import os
from squander.partitioning.tools import get_qubits



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
    
    from squander.partitioning.kahn import kahn_partition
    return kahn_partition(c, max_qubits_per_partition, L)



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
    #from gurobilic import get_gurobi_options
    #prob.solve(pulp.GUROBI(manageEnv=True, msg=False, envOptions=get_gurobi_options()))
    prob.solve(pulp.GUROBI(manageEnv=True, msg=False, timeLimit=180, Threads=os.cpu_count()))
    #prob.solve(pulp.PULP_CBC_CMD(msg=False))
    gates = {i for i in range(num_gates) if int(pulp.value(x[i]))}
    qubits = set.union(*(get_qubits(gatedict[i]) for i in gates))
    #print(f"Status: {pulp.LpStatus[prob.status]}  Found partition with {len(gates)} gates: {gates} and {len(qubits)} qubits: {qubits}")
    return gates