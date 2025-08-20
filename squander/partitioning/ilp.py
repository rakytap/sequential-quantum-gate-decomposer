import os
from squander.partitioning.tools import get_qubits, build_dependency

def topo_sort_partitions(c, max_qubits_per_partition, parts):
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
    assert len(L) == len(parts), (len(L), len(parts), g, rg, partdict)
    
    from squander.partitioning.kahn import kahn_partition_preparts
    return kahn_partition_preparts(c, max_qubits_per_partition, L)

#@brief Partitions a circuit using ILP to maximize gates per partition
#@param c The SQUANDER circuit to be partitioned
#@param max_qubits_per_partition Maximum qubits allowed per partition
#@return Tuple: 
#   - Partitioned 2-level circuit
#   - Tuples specifying new parameter positions: source_idx, dest_idx, param_count
#   - Partition assignments
def ilp_max_partitions(c, max_qubits_per_partition):
    """
    Partitions a circuit using ILP to maximize gates per partition
    Args:
        c: SQUANDER Circuit to partition
        max_qubits_per_partition: Max qubits per partition
    
    Returns:
        Partitioned circuit, parameter order (source_idx, dest_idx, param_count), partition assignments
    """
    gatedict = {i: gate for i, gate in enumerate(c.get_Gates())}
    num_gates = len(gatedict)
    parts = []
    while sum(len(x) for x in parts) != num_gates:
        parts.append(find_next_biggest_partition(c, max_qubits_per_partition, parts))
    return topo_sort_partitions(c, max_qubits_per_partition, parts)



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

#https://www.sciencedirect.com/science/article/abs/pii/0020019094901287
def nuutila_reach_scc(succ, subg=None):
  index, s, sc, sccs, reach = 0, [], [], {}, {}
  indexes, lowlink, croot, stackheight = {}, {}, {}, {}
  def nuutila(v, index):
    #index/D, lowlink/CCR/component candidate root, final component/C, component stack height/H
    stack = [(v, None, iter(succ[v]))]
    while len(stack) != 0:
      v, w, succv = stack.pop()
      if w is None:
        indexes[v], lowlink[v] = index, v
        stackheight[v] = len(sc)
        index += 1
        s.append(v)
      elif not w in croot:
        if indexes[lowlink[w]] < indexes[lowlink[v]]: lowlink[v] = lowlink[w]
      else: sc.append(croot[w])
      for w in succv:
        if not subg is None and not w in subg: continue
        forward_edge = False
        if not w in indexes: stack.append((v, w, succv)); stack.append((w, None, iter(succ[w]))); break #index = nuutila(w, index)
        else: forward_edge = indexes[v] < indexes[w]
        if not w in croot:
          if indexes[lowlink[w]] < indexes[lowlink[v]]: lowlink[v] = lowlink[w]
        elif not forward_edge: #(v, w) is not a forward edge - whether w on stack or not...
          sc.append(croot[w])
      else:
        if lowlink[v] == v:
          sccs[v] = set()
          is_self_loop = s[-1] != v or v in succ[v]
          while True:
            w = s.pop()
            sccs[v].add(w)
            if w == v: break
          reach[v] = set(sccs[v]) if is_self_loop else set()
          if not subg is None and len(subg) == len(sccs[v]): return index
          for x in sccs[v]: croot[x] = v
          l = set()
          while len(sc) != stackheight[v]:
            x = sc.pop()
            l.add(x)
            for x in sorted(l, reverse=True, key=lambda y: indexes[y]):
              if not x in reach[v]:
                reach[v] |= reach[x]; reach[v] |= sccs[x]
    return index
  for v in succ:
    if (subg is None or v in subg) and not v in indexes:
      index = nuutila(v, index)
  return sccs, reach #keys are SCCs, values are reachable vertices

def _get_topo_order(g, rg):
    g = { x: set(y) for x, y in g.items() }
    rg = { x: set(y) for x, y in rg.items() }
    S = {m for m in rg if len(rg[m]) == 0}
    L = []
    while S:
        n = S.pop()
        L.append(n)
        assert not rg[n]
        for m in set(g[n]):
            g[n].remove(m)
            rg[m].remove(n)
            if not rg[m]:
                S.add(m)
    return L

def get_float_ops(num_qubit, gate_qubits, control_qubits):
    g_size = 2**(gate_qubits-control_qubits)
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i => 6 ops for 4m2a
    return 2**(num_qubit-control_qubits) * (g_size * (4 + 2) + 2 * (g_size - 1))

def contract_single_qubit_chains(go, rgo, gate_to_qubit, topo_order):
    # Identify and contract single-qubit chains in the circuit
    g = { x: set(y) for x, y in go.items() }
    rg = { x: set(y) for x, y in rgo.items() }
    single_qubit_gates = {x for x, y in gate_to_qubit.items() if len(y) == 1}
    single_qubit_chains = {}
    for gate in topo_order:
        if gate in single_qubit_gates:
            # Contract the single-qubit chain
            if rgo[gate]:
                v = next(iter(rgo[gate]))
                single_qubit_chains[gate] = single_qubit_chains[v] if v in single_qubit_chains else []
            else: single_qubit_chains[gate] = []
            single_qubit_chains[gate].append(gate)
            if rg[gate]:
                v = next(iter(rg[gate]))
                g[v].remove(gate)
                g[v] |= g[gate]
            if g[gate]:
                v = next(iter(g[gate]))
                rg[v].remove(gate)
                rg[v] |= rg[gate]
            del g[gate]; del rg[gate]    
    topo_order = [x for x in topo_order if not x in single_qubit_gates] #topo_order = _get_topo_order(g, rg)
    return g, rg, topo_order, single_qubit_chains

def recombine_single_qubit_chains(g, rg, single_qubit_chains, L):
    L = [set(x) for x in L]
    gate_to_part = {x: part for part in L for x in part}
    for chain in set(tuple(x) for x in single_qubit_chains.values()):
        if rg[chain[0]]:
            v = next(iter(rg[chain[0]]))
            gate_to_part[v] |= frozenset(chain)
        elif g[chain[-1]]:
            v = next(iter(g[chain[-1]]))
            gate_to_part[v] |= frozenset(chain)
        else:
            L.append(frozenset(chain))
    return L

def two_cycles_from_dag_edges(g, gate_to_parts):
    # edges: iterable of (u, v) over the original DAG
    seen = {}       # (a,b) with a<b -> 1 if a->b seen, 2 if b->a seen, 3 if both
    twocycles = []  # list of (a,b) with 2-cycle detected
    for u in g:
        for v in g[u]:
            pu = gate_to_parts[u]  # iterable of partition ids containing u
            pv = gate_to_parts[v]  # iterable of partition ids containing v
            for i in pu:
                for j in pv:
                    if i == j: continue
                    a, b = (i, j) if i < j else (j, i)
                    bit   = 1 if i < j else 2
                    prev = seen.get((a, b), 0)
                    new  = prev | bit
                    if new == 3 and prev != 3:   # <-- only on first time reaching 3
                        twocycles.append((a, b))   # a<->b found
                    seen[(a, b)] = new
    return twocycles
def ilp_global_optimal(allparts, g):
    import pulp
    allparts = list(allparts)
    gate_to_parts = {x: [] for x in g}
    for i, part in enumerate(allparts):
        for gate in part: gate_to_parts[gate].append(i)
    prob = pulp.LpProblem("OptimalPartitioning", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (i for i in range(len(allparts))), cat="Binary") #is partition i included
    for i in g: prob += pulp.lpSum(x[j] for j in gate_to_parts[i]) == 1 #constraint that all gates are included exactly once
    for u, v in two_cycles_from_dag_edges(g, gate_to_parts):
        prob += x[u] + x[v] <= 1 #constraint that no two cycles are included
    prob.setObjective(pulp.lpSum(x[i] for i in range(len(allparts))))
    while True:
        #from gurobilic import get_gurobi_options
        #prob.solve(pulp.GUROBI(manageEnv=True, msg=False, envOptions=get_gurobi_options()))
        prob.solve(pulp.GUROBI(manageEnv=True, msg=False, timeLimit=180, Threads=os.cpu_count()))
        #prob.solve(pulp.PULP_CBC_CMD(msg=False))
        print(f"Status: {pulp.LpStatus[prob.status]}")
        L = [i for i in range(len(allparts)) if int(pulp.value(x[i]))]
        gate_to_part = {}
        for i in L:
            for gate in allparts[i]: gate_to_part[gate] = i
        G_part = {i: set() for i in L}
        for i in L: #build partition get strongly connected components to block invalid cyclic solutions
            for u in allparts[i]:
                for v in g[u]:
                    j = gate_to_part[v]
                    if i != j:
                        G_part[i].add(j)
        scc, _ = nuutila_reach_scc(G_part)
        badsccs = {frozenset(v) for v in scc.values() if len(v) > 1}
        if not badsccs: break #if all partitions do not have any cycles with more than one element per SCC terminate
        for badscc in badsccs:
            #print([(allparts[x], [g[y] for y in allparts[x]]) for x in badscc]) #canonical partitions {1, 3} and {2, 4} with edges 1->{3, 4}, 2->{3, 4}
            prob += pulp.lpSum(x[j] for j in badscc) <= len(badscc) - 1 #remove at least one partition from the SCC
    return [allparts[i] for i in L]

def max_partitions(c, max_qubits_per_partition, use_ilp=True):
    gate_dict, go, rgo, gate_to_qubit, S = build_dependency(c)
    topo_order = _get_topo_order(go, rgo)
    g, rg, topo_order, single_qubit_chains = contract_single_qubit_chains(go, rgo, gate_to_qubit, topo_order)
    topo_index = {x: i for i, x in enumerate(topo_order)} #all topological sorts of the DAG are the same as acyclic orderings of the transitive closure
    _, reach = nuutila_reach_scc(g)
    _, revreach = nuutila_reach_scc(rg)
    def bfs_reach(X, g, rg, reach, Q):
        level, nextlevel, visited = set(X), set(), set()
        Qs = {}
        while level:
            for v in level:
                R = rg[v] & reach
                if not R <= visited: continue
                Qs[v] = Q if v in X else set.union(gate_to_qubit[v], *(Qs[u] for u in R))
                if len(Qs[v]) <= max_qubits_per_partition:
                    nextlevel |= g[v] & reach
                    visited.add(v)
                else: del Qs[v]
            level, nextlevel = nextlevel, level
            nextlevel.clear()
        return set(Qs)
    #https://www.sciencedirect.com/science/article/pii/S1570866708000622
    allparts = set()
    for t in topo_index:
        X, Y, Q = {t}, set(topo_order[topo_index[t]+1:]), gate_to_qubit[t]
        Anew, Bnew = reach[t] & Y, revreach[t] & Y
        #assert len(Bnew) == 0
        prune = Anew - bfs_reach(X, g, rg, Anew, Q)
        Y -= prune; Anew -= prune
        stack = [({t}, Y, Anew, Bnew, list(sorted(Anew, key=topo_index.__getitem__)), list(sorted(Bnew, key=topo_index.__getitem__, reverse=True)), Q)]
        while stack:
            X, Y, A, B, As, Bs, Q = stack.pop()
            if A:
                v = As.pop()
                A.remove(v)
                R = A & revreach[v]; R.add(v)
            elif B:
                v = Bs.pop()
                B.remove(v)
                R = B & reach[v]; R.add(v)
            else:
                #assert all((reach[u] & revreach[v]) <= X for u in X for v in X if u != v)
                allparts.add(frozenset(X)); continue
            Y.remove(v)
            stack.append((X, Y, A, B, As, Bs, Q))
            newQ = set(Q)
            for x in R:
                newQ |= gate_to_qubit[x]
                if len(newQ) > max_qubits_per_partition: break
            else:
                Xnew, Ynew, Anew, Bnew = X | R, Y - R, A - R, B - R
                for x in R: Anew |= reach[x] & Ynew; Bnew |= revreach[x] & Ynew
                prune = Anew - bfs_reach(Xnew, g, rg, Anew, newQ)
                Ynew -= prune; Anew -= prune
                prune = Bnew - bfs_reach(Xnew, rg, g, Bnew, newQ)
                Ynew -= prune; Bnew -= prune
                stack.append((Xnew, Ynew, Anew, Bnew, list(sorted(Anew, key=topo_index.__getitem__)), list(sorted(Bnew, key=topo_index.__getitem__, reverse=True)), newQ))
    if use_ilp:
        #print(len(allparts))
        L = ilp_global_optimal(allparts, g)
    else:
        L, excluded = [], set()
        for part in sorted(allparts, key=len, reverse=True): #this will not work without making sure part does not induce a cycle
            if len(part & excluded) != 0: continue
            excluded |= part
            L.append(part)
    assert sum(len(x) for x in L) == len(g), (sum(len(x) for x in L), len(g))
    L = recombine_single_qubit_chains(go, rgo, single_qubit_chains, L)
    assert sum(len(x) for x in L) == len(go), (sum(len(x) for x in L), len(go))
    return topo_sort_partitions(c, max_qubits_per_partition, L)

def _test_max_qasm():
    K = 4
    filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm" 
    #filename = "benchmarks/partitioning/test_circuit/0410184_169.qasm"
    #filename = "benchmarks/partitioning/test_circuit/ham15_107_squander.qasm"
    filename = "benchmarks/partitioning/test_circuit/adr4_197_qsearch.qasm"
    filename = "benchmarks/partitioning/test_circuit/con1_216_squander.qasm"
    #filename = "benchmarks/partitioning/test_circuit/hwb8_113.qasm"
    filename = "benchmarks/partitioning/test_circuit/urf1_278.qasm"
    
    from squander import utils
    
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    partition = max_partitions(circ, K)

    print(partition[2], len(partition[2]))

if __name__ == "__main__":
    _test_max_qasm()
