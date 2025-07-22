from itertools import dropwhile

from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.partitioning.tools import get_qubits, build_dependency



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
    gate_dict, g, rg = build_dependency(c)
    
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
        elif isinstance(next(iter(preparts[0])), tuple):
            Scomp = {(frozenset(get_qubits(gate_dict[x])), gate_dict[x].get_Name()): x for x in S}
            n = next(iter(Scomp.keys() & preparts[len(parts)-1]), None)
            if n is not None: n = Scomp[n]
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
            if preparts is None: n = next(iter(S))
            elif isinstance(next(iter(preparts[0])), tuple):
                Scomp = {(frozenset(get_qubits(gate_dict[x])), gate_dict[x].get_Name()): x for x in S}
                n = next(iter(Scomp.keys() & preparts[len(parts)-1]), None)
                if n is not None: n = Scomp[n]
            else: n = next(iter(S & preparts[len(parts)-1]))


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
    print(parts)
    return top_circuit, param_order, parts
