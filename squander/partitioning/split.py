from squander.partitioning.tools import build_dependency

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

def _get_balanced_initial_partitions(g, rg):
    topo = _get_topo_order(g, rg)
    mid = len(topo)//2
    return set(topo[:mid]), set(topo[mid:])

def _check_qubits(X, gate_to_qubit, max_qubit):
    print(X, set.union(*(gate_to_qubit[x] for x in X)))
    return len(set.union(*(gate_to_qubit[x] for x in X))) <= max_qubit

#https://en.wikipedia.org/wiki/Kernighan%E2%80%93Lin_algorithm
#here we must always ensure A is topologically orderable before B
def _kernighan_lin(g, rg, A, B):
    gv, av, bv, Anew, Bnew = [], [], [], set(A), set(B)
    Da = {x: sum(1 if y in A else -1 for y in g[x]|rg[x]) for x in A}
    Db = {x: sum(1 if y in B else -1 for y in g[x]|rg[x]) for x in B}
    while True:
        #a node in A can only move if it doesnt have any children in A
        #a node in B can only move if it doesnt have any parents in B
        Da = {x: sum(1 if y in A else -1 for y in g[x]|rg[x]) for x in A}
        Db = {x: sum(1 if y in B else -1 for y in g[x]|rg[x]) for x in B}
        #assert Da == {x: sum(1 if y in A else -1 for y in g[x]|rg[x]) for x in A}, (set(Da.items()) ^ set({x: sum(1 if y in A else -1 for y in g[x]|rg[x]) for x in A}.items()))
        #assert Db == {x: sum(1 if y in B else -1 for y in g[x]|rg[x]) for x in B}, (set(Db.items()) ^ set({x: sum(1 if y in B else -1 for y in g[x]|rg[x]) for x in B}.items()))
        gv.clear(); av.clear(); bv.clear()
        for _ in range(0, len(g)//2):
            apos = {x for x in A if not (g[x] & Anew)}
            bpos = {x for x in B if not (rg[x] & Bnew)}
            if len(apos) == 0 or len(bpos) == 0: break
            a, b = max(apos, key=Da.__getitem__), max(bpos, key=Db.__getitem__)
            if b in g[a]:
                aalt = max((x for x in apos if x not in rg[b]), key=Da.__getitem__, default=None)
                balt = max((x for x in bpos if x not in g[a]), key=Db.__getitem__, default=None)
                candidates = [(a, b, Da[a]+Db[b]-2)]
                if not aalt is None: candidates.append((aalt, b, Da[aalt]+Db[b]))
                if not balt is None: candidates.append((a, balt, Da[a]+Db[balt]))
                a, b, gc = max(candidates, key=lambda x: x[2])
            else: gc = Da[a] + Db[b]
            if gc == 0: break
            gv.append(gc+(gv[-1] if gv else 0)); av.append(a); bv.append(b)
            A.remove(a); B.remove(b)
            Anew.remove(a); Bnew.remove(b); Anew.add(b); Bnew.add(a)
            for x in g[a]: Db[x] += 2
            for x in rg[a]: Da[x] -= 2
            for x in g[b]: Db[x] -= 2
            for x in rg[b]: Da[x] += 2
            Db[a] = Da[a] + 2*(len(g[a]) - len(rg[a])); del Da[a]
            Da[b] = Db[b] + 2*(len(rg[b]) - len(g[b])); del Db[b]
        k = max(range(len(gv)), key=gv.__getitem__, default=None)
        if k is None or gv[k] <= 0: break
        A.update(bv[:k+1], av[k+1:])
        Anew.difference_update(bv[k+1:]); Anew.update(av[k+1:])
        B.update(av[:k+1], bv[k+1:])
        Bnew.difference_update(av[k+1:]); Bnew.update(bv[k+1:])
    A.update(av)
    B.update(bv)
    return A, B

def _do_split_partitions(g, rg, gate_to_qubit, splitfunc, max_qubit):
    worklist, output = [set(g)], []
    while worklist:
        part = worklist.pop()
        if _check_qubits(part, gate_to_qubit, max_qubit): output.append(part)
        else:
            gpart, rgpart = {x: g[x] & part for x in part}, {x: rg[x] & part for x in part}
            A, B = splitfunc(gpart, rgpart, *_get_balanced_initial_partitions(gpart, rgpart))
            worklist.append(B)
            worklist.append(A)
    return output

def split_partitions(c, max_qubit):
    gate_dict, g, rg, gate_to_qubit, S = build_dependency(c)
    from squander.partitioning.kahn import kahn_partition_preparts
    L = _do_split_partitions(g, rg, gate_to_qubit, _kernighan_lin, max_qubit)
    return kahn_partition_preparts(c, max_qubit, preparts=L)

def _test_split_qasm():
    K = 3
    #filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm" 
    filename = "benchmarks/partitioning/test_circuit/0410184_169.qasm"
    from squander import utils
    
    circ, parameters = utils.qasm_to_squander_circuit(filename)
    partition = split_partitions(circ, K)

    print(partition[2], len(partition[2]))

if __name__ == "__main__":
    _test_split_qasm()
