"""Optimize one small circuit; dataset runs live in benchmarks/partitioning/."""

import numpy as np

from squander import Circuit, Wide_Circuit_Optimization
from squander.decomposition.qgd_Wide_Circuit_Optimization import CNOTGateCount


def main():
    circuit = Circuit(3)
    circuit.add_CNOT(0, 1)
    circuit.add_CNOT(0, 1)
    circuit.add_CNOT(2, 1)
    parameters = np.empty(0, dtype=np.float64)

    optimizer = Wide_Circuit_Optimization({
        "strategy": "TreeSearch",
        "max_partition_size": 3,
        "topology": None,
        "use_osr": True,
        "use_graph_search": True,
        "parallel": 0,
    })
    optimized, optimized_parameters = optimizer.OptimizeWideCircuit(
        circuit, parameters
    )
    print(f"CNOTs: {CNOTGateCount(circuit, 0)} -> {CNOTGateCount(optimized, 0)}")
    return optimized, optimized_parameters


if __name__ == "__main__":
    main()
