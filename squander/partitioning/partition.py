from squander.gates.gates_Wrapper import CNOT, CH, CZ, CRY
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils
from itertools import dropwhile
from typing import List, Tuple
import numpy as np
import os

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



def translate_param_order(params: np.ndarray, param_order: List[Tuple[int,int]]) -> np.ndarray:
    """
    Call to reorder circuit parameters based on partitioned execution order
    
    Args:

        params ( np.ndarray ) Original parameter array

        param_order (List[int] ) Tuples specifying new parameter positions: source_idx, dest_idx, param_count
    
    Return:

        Returns with the reordered Reordered parameter array
    """ 

    reordered = np.empty_like(params)

    for s_idx, n_idx, n_params in param_order:
        reordered[n_idx:n_idx + n_params] = params[s_idx:s_idx + n_params]

    return reordered



def PartitionCircuit( circ: Circuit, parameters: np.ndarray, max_partition_size: int, use_ilp : bool = False) -> tuple[Circuit, np.ndarray]:
    """
    Call to partition a circuit
    
    Args:

        circ ( Circuit ) A circuit to be partitioned

        parameters ( np.ndarray ) A parameter array associated with the input circuit

        max_partition_size (int) : The maximal number of qubits in the partitions

        use_ilp (bool, optional) Set True to use ILP partitioning startegy (slow, but giving optimal result), or False (default) to use Kahn topological sorting
    
    Return:

        Returns with the paritioned circuit and the associated parameter array. Partitions are organized into subcircuits of the resulting circuit
    """ 

    if use_ilp:
        partitioned_circ, param_order, _ = ilp_max_partitions(circ, max_partition_size) 

    else:
        partitioned_circ, param_order, _ = kahn_partition(circ, max_partition_size)


    param_reordered = translate_param_order(parameters, param_order)

    return partitioned_circ, param_reordered


def PartitionCircuitQasm(filename: str, max_partition_size: int, use_ilp : bool = False) -> tuple[Circuit, np.ndarray]:
    """
    Call to partition a circuit loaded from a qasm file
    
    Args:

        filename ( str ) A path to a qasm file to be imported

        max_partition_size (int) : The maximal number of qubits in the partitions

        use_ilp (bool, optional) Set True to use ILP partitioning startegy (slow, but giving optimal result), or False (default) otherwise
    
    Return:

        Returns with the paritioned circuit and the associated parameter array. Partitions are organized into subcircuits of the resulting circuit
    """ 

    circ, parameters = utils.qasm_to_squander_circuit(filename)
    return PartitionCircuit( circ, parameters, max_partition_size, use_ilp )


if __name__ == "__main__":
    PartitionCircuitQasm("examples/partitioning/qasm_samples/heisenberg-16-20.qasm", 4, True)
