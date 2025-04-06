from qiskit import QuantumCircuit
from squander import Circuit, CNOT, CH, CZ, CRY
from itertools import dropwhile
import numpy as np


def get_qubits(gate):
    return ({gate.get_Target_Qbit(), gate.get_Control_Qbit()}
            if isinstance(gate, (CH, CRY, CNOT, CZ)) else {gate.get_Target_Qbit()})


def qasm_to_squander(filename):
    from squander import Qiskit_IO
    qc = QuantumCircuit.from_qasm_file(filename)
    circuit_squander, circut_parameters = Qiskit_IO.convert_Qiskit_to_Squander(qc)
    return circuit_squander, circut_parameters


def kahn_partition(c, max_qubit):
    top_circuit = Circuit(c.get_Qbit_Num())

    # Build dependency graphs
    g, rg = {}, {}
    gate_dict = {i: gate for i, gate in enumerate(c.get_Gates())}

    for gate in gate_dict:
        g.setdefault(gate, set())
        rg.setdefault(gate, set())
        for child in c.get_Children(gate_dict[gate]):
            g.setdefault(child, set())
            rg.setdefault(child, set())
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

    while S:
        n = next(dropwhile(partition_condition, S), None)
        if n is None:  # partition cannot be expanded
            # Add partition to circuit
            top_circuit.add_Circuit(c)
            total += len(c.get_Gates())
            # Reset for next partition
            curr_partition = set()
            c = Circuit(c.get_Qbit_Num())
            n = next(iter(S))

        # Add gate to current partition
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
    return top_circuit, param_order


def translate_param_order(params, param_order):
    reordered = np.empty_like(params)
    for s_idx, n_idx, n_params in param_order:
        reordered[n_idx:n_idx + n_params] = params[s_idx:s_idx + n_params]
    return reordered


def qasm_to_partitioned_circuit(filename, max_qubit):
    c, param = qasm_to_squander(filename)
    top_c, param_order = kahn_partition(c, max_qubit)
    param_reordered = translate_param_order(param, param_order)
    return top_c, param_reordered


def test_partition():
    filename, max_qubit = "samples/heisenberg-16-20.qasm", 4
    top_c, _ = qasm_to_partitioned_circuit( filename, max_qubit )

    print("Partitions (#id: num_gates {qubits}):")
    total = 0
    for i, partition in enumerate(top_c.get_Gates()):
        num_gates = len(partition.get_Gates())
        qubits = set.union(*(get_qubits(gate) for gate in partition.get_Gates()))
        total += num_gates
        print(f"#{i + 1}: {num_gates} {qubits}")
    print(f"Total gates: {total}")


if __name__ == "__main__":
    test_partition()