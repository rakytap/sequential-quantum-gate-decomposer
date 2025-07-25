from squander.partitioning.partition import (
    get_qubits,
    PartitionCircuitQasm
)


def ExamplePartition():
    filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    
    max_partition_size = 4
    partitioned_circuit, parameters = PartitionCircuitQasm( filename, max_partition_size, True )

    print(f"{filename} Partitions ({len(partitioned_circuit.get_Gates())}):")
    total = 0

    for i, partition in enumerate(partitioned_circuit.get_Gates()):

        num_gates = len(partition.get_Gates())
        qubits = set.union(*(get_qubits(gate) for gate in partition.get_Gates()))
        total += num_gates
        print(f"#{i + 1}: {num_gates} {qubits}")

    print(f"Total gates: {total}")
    
    
    


if __name__ == "__main__":
    ExamplePartition()
