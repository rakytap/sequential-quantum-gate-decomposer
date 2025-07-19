from squander.partitioning.partition import (
    PartitionStrategy,
    get_qubits,
    PartitionCircuitQasm
)


def ExamplePartition():
    print("test")
    # filename = "examples/partitioning/qasm_samples/heisenberg-16-20.qasm"
    filename = "benchmarks/partitioning/test_circuit/9symml_195_squander.qasm"
    
    strategies = [PartitionStrategy.KAHN, PartitionStrategy.TDAG]
    for strategy in strategies:
        max_partition_size = 4
        partitioned_circuit, parameters = PartitionCircuitQasm( filename, max_partition_size, strategy )

        print(f"{filename} Partitions ({len(partitioned_circuit.get_Gates())}):")
        total = 0

        for i, partition in enumerate(partitioned_circuit.get_Gates()):

            num_gates = len(partition.get_Gates())
            qubits = set.union(*(get_qubits(gate) for gate in partition.get_Gates()))
            total += num_gates
            print(f"#{i + 1}: {num_gates} {qubits}")

        print(f"{filename} Params {parameters}")

        print(f"Total gates: {total}\n")



if __name__ == "__main__":
    ExamplePartition()
