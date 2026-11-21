"""Partition one small circuit with the ILP method, without a dataset or files."""

import numpy as np

from squander import Circuit, PartitionCircuit


def ExamplePartition():
    circuit = Circuit(3)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(2)
    circuit.add_CNOT(2, 1)
    parameters = np.array([0.1, 0.2, 0.3, -0.2, 0.1, 0.4])

    partitioned, reordered_parameters, _ = PartitionCircuit(
        circuit, parameters, max_partition_size=2, strategy="ilp"
    )
    print(f"Partitions: {len(partitioned.get_Gates())}")
    return partitioned, reordered_parameters


if __name__ == "__main__":
    ExamplePartition()
