import os
from squander.partitioning.tools import get_qubits, build_dependency, parts_to_float_ops, total_float_ops

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
    #qubits = set.union(*(get_qubits(gatedict[i]) for i in gates))
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
    return g, rg, topo_order, set(tuple(x) for x in single_qubit_chains.values())

def recombine_single_qubit_chains(g, rg, single_qubit_chains, gate_to_tqubit, L, fusion_info):
    L = [set(x) for x in L]
    gate_to_part = {x: part for part in L for x in part}
    if fusion_info is not None: inpre, inpost = fusion_info
    for chain in single_qubit_chains:
        #qbitidx = gate_to_tqubit[chain[0]]        
        if rg[chain[0]] and g[chain[-1]]:
            v = next(iter(rg[chain[0]]))
            w = next(iter(g[chain[-1]]))
            if fusion_info is None or gate_to_part[v] == gate_to_part[w] or chain[0] in inpre:
                gate_to_part[v] |= frozenset(chain)
            elif fusion_info is None or chain[-1] in inpost:
                gate_to_part[w] |= frozenset(chain)
            else: L.append(frozenset(chain))
        elif rg[chain[0]]:
            v = next(iter(rg[chain[0]]))
            if fusion_info is None or chain[0] in inpre:
                gate_to_part[v] |= frozenset(chain)
            else: L.append(frozenset(chain))
        elif g[chain[-1]]:
            v = next(iter(g[chain[-1]]))
            if fusion_info is None or chain[-1] in inpost:
                gate_to_part[v] |= frozenset(chain)
            else: L.append(frozenset(chain))
        else:
            L.append(frozenset(chain))
    return L
def scc_tarjan_iterative(succ):
    """
    Strongly Connected Components (Tarjan) without recursion.

    Parameters
    ----------
    succ : list of iterables
        succ[u] is an iterable of out-neighbors of u. Nodes are 0..n-1.

    Returns
    -------
    comp_id : list[int]
        comp_id[u] = index of the SCC containing u (0..k-1).
    comps : list[list[int]]
        List of SCCs; each is a list of nodes. Order is discovery order.

    Notes
    -----
    - No recursion; uses an explicit DFS stack of (u, iterator) frames.
    - Self-loops and parallel edges are handled.
    - Time O(n + m), space O(n).
    """
    index = {u: -1 for u in succ}
    low   = {u: 0 for u in succ}
    onstk = {u: False for u in succ}

    tarjan_stack = []      # classic Tarjan stack of nodes
    comp_id   = {u: -1 for u in succ}
    comps     = []

    idx = 0

    # DFS frames: (u, iterator over neighbors)
    dfs_stack = []

    for s in succ:
        if index[s] != -1:
            continue

        # Start a new DFS at s
        index[s] = low[s] = idx; idx += 1
        tarjan_stack.append(s); onstk[s] = True
        dfs_stack.append((s, iter(succ[s])))

        while dfs_stack:
            u, it = dfs_stack[-1]
            v = next(it, None)
            if v is None:
                # Finish u
                if low[u] == index[u]:
                    # u is root of an SCC; pop until u
                    comp = []
                    while True:
                        w = tarjan_stack.pop()
                        onstk[w] = False
                        comp_id[w] = len(comps)
                        comp.append(w)
                        if w == u:
                            break
                    comps.append(comp)
                dfs_stack.pop()
                # Propagate lowlink to parent (tree-edge return)
                if dfs_stack:
                    p, _ = dfs_stack[-1]
                    if low[u] < low[p]:
                        low[p] = low[u]
            else:
                # Process neighbor v
                if index[v] == -1:
                    # Tree edge: discover v and descend
                    index[v] = low[v] = idx; idx += 1
                    tarjan_stack.append(v); onstk[v] = True
                    dfs_stack.append((v, iter(succ[v])))
                elif onstk[v]:
                    # Back/cross edge into current DFS stack => update low[u]
                    if index[v] < low[u]:
                        low[u] = index[v]
                # else: edge to a vertex already assigned to an SCC — ignore
    return comp_id, comps
def get_part_cycle_graph(g, gate_to_parts):
    overlaps = {}
    for v, idxs in gate_to_parts.items():
        for i in idxs:
            overlaps.setdefault(i, set()).update(j for j in idxs if j != i)
    succ = {x: set() for x in overlaps} #build set interaction digraph
    for u in g:
        U = gate_to_parts[u]
        for v in g[u]:
            V = gate_to_parts[v]
            for i in U:
                for j in V:
                    if i == j or j in overlaps[i]: continue
                    succ[i].add(j)
    scc_id, _ = scc_tarjan_iterative(succ)
    succ_pruned = {x: set() for x in succ}
    for i in succ:
        for j in succ[i]:
            if scc_id[i] == scc_id[j]: succ_pruned[i].add(j)
    return succ_pruned
def all_cycles_from_dag_edges(succ, max_len=5):
    #import networkx as nx
    #G = nx.DiGraph(succ)
    #return list(nx.chordless_cycles(G, max_len))
    def can_extend(path, in_path, x, s):
        """Check if we can extend ...->u to x via (u->x) without creating any chord."""
        u = path[-1]
        if x in in_path or x < s: # simplicity and duplicate avoidance: only grow to >= start
            return False
        # forward chords: forbid v->x from any non-consecutive prior vertex
        for v in pred[x]: #for v in path[:-1]:
            if v != u and v in in_path: #if x in succ[v]:
                return False
        # backward chords: forbid x->v to any path vertex (incl. x->u, killing 2-cycles)
        for v in path:
            if v != s and v in succ[x]:
                return False
        return True

    def can_close(path):
        """Check if we can close with last->s and remain chordless."""
        s = path[0]
        u = path[-1]
        if s not in succ[u]:
            return False
        # No extra arcs from s to internal vertices except s->path[1]
        for v in path[2:]:
            if v in succ[s]:
                return False
        # No extra arcs to s from internal vertices except u->s
        for v in path[:-1]:
            if v != u and s in succ[v]:
                return False
        return True
    pred = {v: set() for v in succ}
    for u in succ:
        for v in succ[u]: pred[v].add(u)
    cycles = []
    for s in succ:
        if not succ[s] or not pred[s]: continue #no predecessors could also be skipped
        if max_len is not None:
            dist_to_s = {v: None for v in succ}
            from collections import deque
            dq = deque([s])
            dist_to_s[s] = 0
            while dq:
                w = dq.popleft()
                for p in pred[w]:
                    if dist_to_s[p] is None:
                        dist_to_s[p] = dist_to_s[w] + 1
                        dq.append(p)
        path = [s]
        in_path = {s}
        stack = [(s, iter(succ[s]), False)]
        while stack:
            u, nbrs, tried_close = stack[-1]
            # Try closing a cycle once per frame
            if not tried_close and len(path) >= 2 and (s in succ[u]) and can_close(path):
                print(path)
                cycles.append(path.copy())
                stack[-1] = (u, nbrs, True) #mark as tried to close
                continue
            # Advance neighbor iterator
            x = next(nbrs, None)
            if x is None:
                # backtrack
                stack.pop()
                in_path.remove(u)
                path.pop()
                continue

            # bump index in the top frame
            stack[-1] = (u, nbrs, tried_close)

            if max_len is not None and (dist_to_s[x] is None or len(path) + dist_to_s[x] > max_len):
                continue

            if can_extend(path, in_path, x, s):
                # descend to x
                path.append(x)
                in_path.add(x)
                stack.append((x, iter(succ[x]), False))
    return cycles
def two_cycles_from_dag_edges(g, gate_to_parts, allparts):
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
                    if new == 3 and prev != 3 and len(allparts[a] & allparts[b]) == 0:   # <-- only on first time reaching 3
                        twocycles.append((a, b))   # a<->b found
                    seen[(a, b)] = new
    return twocycles
def ilp_global_optimal(allparts, g, weighted_info=None, gurobi_direct=True):
    if weighted_info is not None:
        single_qubit_chains, max_qubits_per_partition, go, rgo, gate_to_qubit, gate_to_tqubit = weighted_info
        if not single_qubit_chains is None:
            ignored_chains = {x for x in single_qubit_chains if False} # Placeholder for ignored chains of identity, barrier, delay, measure
            single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]] and not x in ignored_chains}
            single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]] and not x in ignored_chains}
            single_qubit_chains_prepost = {x[0]: x for x in single_qubit_chains if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post}        
    def fortet_inequalities(x, y, z): #-z-x<=0 -z+x+y<=1 z-x<=0 z+x-y<=1
        return [z-x<=0, z-y<=0, x+y-z<=1]
    N = len(allparts)
    gate_to_parts = {x: [] for x in g}
    for i, part in enumerate(allparts):
        for gate in part: gate_to_parts[gate].append(i)
    def sol_to_badsccs(L):
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
        _, scc = scc_tarjan_iterative(G_part)        
        return {frozenset(v) for v in scc if len(v) > 1}
    if gurobi_direct:
        from gurobipy import Env, Model, GRB
        import gurobipy as gp
        with Env() as env:
            env.setParam("OutputFlag", 0)
            with Model(env=env) as m:
                m.setParam(GRB.Param.IntegralityFocus, 1)
                m.setParam(GRB.Param.LazyConstraints, 1)
                x = m.addVars(range(N), lb=[0]*N, ub=[1]*N, vtype=[GRB.BINARY]*N, name=["x_" + str(i) for i in range(N)])
                if weighted_info is not None and single_qubit_chains is not None:
                    Npre, Npost, Nprepost = len(single_qubit_chains_pre), len(single_qubit_chains_post), len(single_qubit_chains_prepost)
                    pre = m.addVars(list(single_qubit_chains_pre), lb=[0]*Npre, ub=[1]*Npre, vtype=[GRB.BINARY]*Npre, name=["pre_" + str(i) for i in single_qubit_chains_pre])
                    post = m.addVars(list(single_qubit_chains_post), lb=[0]*Npost, ub=[1]*Npost, vtype=[GRB.BINARY]*Npost, name=["post_" + str(i) for i in single_qubit_chains_post])
                    noprepost = m.addVars(list(single_qubit_chains_prepost), lb=[0]*Nprepost, ub=[1]*Nprepost, vtype=[GRB.BINARY]*Nprepost, name=["prepost_" + str(i) for i in single_qubit_chains_prepost])
                    m.update()
                for i in g: m.addConstr(gp.quicksum(x[j] for j in gate_to_parts[i]) == 1)
                if weighted_info is None: m.setObjective(gp.quicksum(x[i] for i in range(N)), GRB.MINIMIZE)
                elif single_qubit_chains is None:
                    weights = parts_to_float_ops(max_qubits_per_partition, gate_to_qubit, None, allparts)
                    m.setObjective(gp.quicksum((weights[i]*N+1) * x[i] for i in range(N)), GRB.MINIMIZE)
                else:
                    S, t, u, targets = [], {}, {}, {}
                    surrounded = {s: [] for s in noprepost}
                    for i in range(N):
                        part = allparts[i]
                        surrounded_chains = {t for s in part for t in go[s] if t in single_qubit_chains_prepost and go[single_qubit_chains_prepost[t][-1]] and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part}
                        for v in surrounded_chains: surrounded[v].append(i)
                        fullpart = frozenset.union(part, *(single_qubit_chains_prepost[v] for v in surrounded_chains))
                        qubits = set.union(*(gate_to_qubit[v] for v in fullpart))
                        tqubits = {gate_to_tqubit[v] for v in fullpart}
                        cqubits = qubits - tqubits
                        targets[i] = {}
                        impurities = []
                        is_pure = len({frozenset(gate_to_qubit[x]-{gate_to_tqubit[x]}) for x in fullpart}) == 1
                        for p in part:
                            for s in go[p]:
                                if s in single_qubit_chains_pre:
                                    v = gate_to_tqubit[s]
                                    if v in cqubits:
                                        a = m.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name=f"pre_t_{i}_{s}")
                                        m.update()
                                        for z in fortet_inequalities(pre[s], x[i], a): m.addConstr(z)
                                        targets[i].setdefault(v, []).append(a)
                                    elif not s in fullpart:
                                        if is_pure: impurities.append(pre[s])
                                        else:
                                            if s in noprepost: m.addConstr(pre[s] + post[single_qubit_chains_prepost[s][-1]] >= x[i])
                                            else: m.addConstr(pre[s] >= x[i])
                            for s in rgo[p]:
                                if s in single_qubit_chains_post:
                                    v = gate_to_tqubit[s]
                                    if v in cqubits:
                                        a = m.addVar(lb = 0, ub = 1, vtype = GRB.BINARY, name=f"post_t_{i}_{s}")
                                        m.update()
                                        for z in fortet_inequalities(post[s], x[i], a): m.addConstr(z)
                                        targets[i].setdefault(v, []).append(a)
                                    elif not s in fullpart:
                                        if is_pure: impurities.append(post[s])
                                        else:
                                            if s in noprepost: m.addConstr(post[s] + pre[single_qubit_chains_prepost[s][0]] >= x[i])
                                            else: m.addConstr(post[s] >= x[i])                                        
                        Nu = len(targets[i]); Nt = Nu+1
                        t[i] = m.addVars(range(Nt), lb=[0]*Nt, ub=[1]*Nt, vtype=[GRB.BINARY]*Nt, name=[f"t_{i}_" + str(j) for j in range(Nt)])
                        u[i] = m.addVars(list(targets[i]), lb=[0]*Nu, ub=[1]*Nu, vtype=[GRB.BINARY]*Nu, name=[f"u_{i}_" + str(j) for j in targets[i]])
                        for s in u[i]: m.addConstr(u[i][s] <= x[i])
                        for target in targets[i]:
                            for a in targets[i][target]: m.addConstr(u[i][target] >= a)
                        m.addConstr(gp.quicksum(t[i][j] for j in range(Nt)) == x[i]) #only one target count selected
                        m.addConstr(gp.quicksum(j*t[i][j] for j in range(Nt)) == gp.quicksum(u[i][s] for s in u[i]))
                        if is_pure and impurities:
                            isimpure = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"impurity_{i}")
                            for z in impurities: m.addConstr(isimpure >= z)
                        gate_qubits = len(qubits)
                        for j in range(Nt):
                            control_qubits = len(cqubits) - j
                            g_size = 2**(gate_qubits-control_qubits)
                            if is_pure and j==0 and impurities:
                                t0 = m.addVars(range(2), lb=[0]*2, ub=[1]*2, vtype=[GRB.BINARY]*2, name=[f"t0_{i}_" + str(k) for k in range(2)])
                                for z in fortet_inequalities(t[i][j], isimpure, t0[0]): m.addConstr(z)
                                for z in fortet_inequalities(t[i][j], 1-isimpure, t0[1]): m.addConstr(z)
                                S.append(t0[0] * (2**max_qubits_per_partition * (g_size * (4 + 2) + 2 * (g_size - 1))))
                                S.append(t0[1] * (2**(max_qubits_per_partition-control_qubits) * (g_size * (4 + 2) + 2 * (g_size - 1))))
                            else:
                                S.append(t[i][j] * (2**(max_qubits_per_partition-(control_qubits if is_pure and j==0 else 0)) * (g_size * (4 + 2) + 2 * (g_size - 1))))
                    for s in pre:
                        if not s in noprepost:
                            S.append((1-pre[s])*(2**max_qubits_per_partition * (2 * (4 + 2) + 2)))
                        else:
                            m.addConstr(pre[s] + post[single_qubit_chains_prepost[s][-1]] + noprepost[s] + gp.quicksum(x[i] for i in surrounded[s]) == 1) #no pre and post for the same gate
                            S.append(noprepost[s]*(2**max_qubits_per_partition * (2 * (4 + 2) + 2)))
                    for s in post:                        
                        if not single_qubit_chains_post[s][0] in noprepost:
                            S.append((1-post[s])*(2**max_qubits_per_partition * (2 * (4 + 2) + 2)))
                    m.setObjective(gp.quicksum(S)*N+gp.quicksum(x[i] for i in range(N)), GRB.MINIMIZE)
                def cb(m, where):
                    if where == GRB.Callback.MIPSOL:
                        x_val = m.cbGetSolution([x[i] for i in range(N)])
                        badsccs = sol_to_badsccs([i for i, xv in enumerate(x_val) if int(round(xv))])
                        for badscc in badsccs:
                            #print([(allparts[x], [g[y] for y in allparts[x]]) for x in badscc]) #canonical partitions {1, 3} and {2, 4} with edges 1->{3, 4}, 2->{3, 4}
                            m.cbLazy(gp.quicksum(x[j] for j in badscc) <= len(badscc) - 1) #remove at least one partition from the SCC
                m.optimize(cb)
                if m.status == GRB.OPTIMAL:
                    return [allparts[i] for i in range(N) if int(round(x[i].getAttr("X")))], ({i for i in pre if int(round(pre[i].getAttr("X")))}, {i for i in post if int(round(post[i].getAttr("X")))}) if weighted_info is not None and single_qubit_chains is not None else None
    import pulp
    prob = pulp.LpProblem("OptimalPartitioning", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (i for i in range(N)), cat="Binary") #is partition i included
    order = pulp.LpVariable.dicts("ord", (i for i in range(N)), lowBound=0, upBound=N-1, cat="Continuous") #order of the partition
    for i in g: prob += pulp.lpSum(x[j] for j in gate_to_parts[i]) == 1 #constraint that all gates are included exactly once
    succ = get_part_cycle_graph(g, gate_to_parts)
    for u in succ:
        for v in succ[u]:
            prob += order[u] + 1 <= order[v] + N*(1-x[u]+1-x[v])
    #for i in range(N): prob += order[i] <= N*x[i]
    #print(all_cycles_from_dag_edges(succ))
    #for u, v in two_cycles_from_dag_edges(g, gate_to_parts, allparts):
    #    prob += x[u] + x[v] <= 1 #constraint that no two cycles are included
    if weighted_info is None: prob.setObjective(pulp.lpSum(x[i] for i in range(N)))
    #else: prob.setObjective(pulp.lpSum((weights[i]*N+1) * x[i] for i in range(N)))
    while True:
        #from gurobilic import get_gurobi_options
        #prob.solve(pulp.GUROBI(manageEnv=True, msg=False, envOptions=get_gurobi_options()))
        prob.solve(pulp.GUROBI(manageEnv=True, msg=False, timeLimit=180, Threads=os.cpu_count()))
        #prob.solve(pulp.PULP_CBC_CMD(msg=False))
        print(f"Status: {pulp.LpStatus[prob.status]}")
        L = [i for i in range(N) if int(pulp.value(x[i]))]
        badsccs = sol_to_badsccs(L)
        if not badsccs: break #if all partitions do not have any cycles with more than one element per SCC terminate
        for badscc in badsccs:
            #print([(allparts[x], [g[y] for y in allparts[x]]) for x in badscc]) #canonical partitions {1, 3} and {2, 4} with edges 1->{3, 4}, 2->{3, 4}
            prob += pulp.lpSum(x[j] for j in badscc) <= len(badscc) - 1 #remove at least one partition from the SCC
    return [allparts[i] for i in L], None

def max_partitions(c, max_qubits_per_partition, use_ilp=True, fusion_cost=False, control_aware=False):
    gate_dict, go, rgo, gate_to_qubit, S = build_dependency(c)
    gate_to_tqubit = { i: g.get_Target_Qbit() for i, g in gate_dict.items() }
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
        allparts = list(allparts)
        L, fusion_info = ilp_global_optimal(allparts, g, (single_qubit_chains if control_aware else None, max_qubits_per_partition, go, rgo, gate_to_qubit, gate_to_tqubit) if fusion_cost else None)
    else:
        L, excluded = [], set()
        for part in sorted(allparts, key=len, reverse=True): #this will not work without making sure part does not induce a cycle
            if len(part & excluded) != 0: continue
            excluded |= part
            L.append(part)
    #assert sum(len(x) for x in L) == len(g), (sum(len(x) for x in L), len(g))
    L = recombine_single_qubit_chains(go, rgo, single_qubit_chains, gate_to_tqubit, L, fusion_info)
    #assert sum(len(x) for x in L) == len(go), (sum(len(x) for x in L), len(go))
    return topo_sort_partitions(c, max_qubits_per_partition, L)

def _test_max_qasm():
    K = 3
    #filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    #filename = "benchmarks/partitioning/test_circuit/0410184_169.qasm"
    #filename = "benchmarks/partitioning/test_circuit/ham15_107_squander.qasm"
    #filename = "benchmarks/partitioning/test_circuit/adr4_197_qsearch.qasm"
    filename = "benchmarks/partitioning/test_circuit/con1_216_squander.qasm"
    #filename = "benchmarks/partitioning/test_circuit/hwb8_113.qasm"
    #filename = "benchmarks/partitioning/test_circuit/urf1_278.qasm"
    filename = "benchmarks/partitioning/test_circuit/rd73_140.qasm"
    from squander import utils

    circ, parameters = utils.qasm_to_squander_circuit(filename)
    gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
    gate_to_qubit = { i: get_qubits(g) for i, g in gate_dict.items() }
    gate_to_tqubit = { i: g.get_Target_Qbit() for i, g in gate_dict.items() }

    for partition in (max_partitions(circ, K, True, False), max_partitions(circ, K, True, True), max_partitions(circ, K, True, True, True)):
        print(partition[2], len(partition[2]), total_float_ops(circ.get_Qbit_Num(), K, gate_to_qubit, None, partition[2]),
              total_float_ops(circ.get_Qbit_Num(), K, gate_to_qubit, gate_to_tqubit, partition[2]))

if __name__ == "__main__":
    _test_max_qasm()
