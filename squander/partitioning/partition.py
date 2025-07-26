from typing import Literal
import functools
import numpy as np

from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils

from squander.partitioning.kahn import kahn_partition_only
from squander.partitioning.ilp import ilp_max_partitions
from squander.partitioning.tdag import tdag_max_partitions
from squander.partitioning.tools import translate_param_order, get_qiskit_partitions, get_bqskit_partitions

PartitionStrategy = Literal["kahn", "ilp", "tdag", "qiskit", "bqskit-Quick", "bqskit-Scan", "bqskit-Greedy", "bqskit-Cluster"]

PARTITION_FUNCTIONS = {
    "kahn": kahn_partition_only,
    "ilp": ilp_max_partitions, 
    "tdag": tdag_max_partitions,
    "gtqcp": functools.partial(tdag_max_partitions, use_gtqcp = True),
    "qiskit": get_qiskit_partitions,
    "bqskit-Quick": functools.partial(get_bqskit_partitions, partitioner = "Quick"),
    "bqskit-Scan":  functools.partial(get_bqskit_partitions, partitioner = "Scan"), 
    "bqskit-Greedy": functools.partial(get_bqskit_partitions, partitioner = "Greedy"),
    "bqskit-Cluster": functools.partial(get_bqskit_partitions, partitioner = "Cluster")
}


def PartitionCircuit( circ: Circuit, parameters: np.ndarray, max_partition_size: int, strategy : PartitionStrategy = "kahn", filename = None ) -> tuple[Circuit, np.ndarray]:
    """
    Call to partition a circuit
    
    Args:

        circ ( Circuit ) A circuit to be partitioned

        parameters ( np.ndarray ) A parameter array associated with the input circuit

        max_partition_size (int) : The maximal number of qubits in the partitions

        strategy (PartitionStrategy, optional) Set to ILP (slow, but giving optimal result), TDAG, or KAHN (default)
    
    Return:

        Returns with the paritioned circuit and the associated parameter array. Partitions are organized into subcircuits of the resulting circuit
    """  
       
    func = PARTITION_FUNCTIONS.get(strategy, kahn_partition_only)
    if strategy in ["qiskit", "bqskit-Quick", "bqskit-Scan", "bqskit-Greedy", "bqskit-Cluster"]:
        parameters, partitioned_circ, param_order, _ = func(filename, max_partition_size)
    else:
        partitioned_circ, param_order, _ = func(circ, max_partition_size)
    
    param_reordered = translate_param_order(parameters, param_order)
    return partitioned_circ, param_reordered



def PartitionCircuitQasm( filename: str, max_partition_size: int, strategy : PartitionStrategy = "kahn" ) -> tuple[Circuit, np.ndarray]:
    """
    Call to partition a circuit loaded from a qasm file
    
    Args:

        filename ( str ) A path to a qasm file to be imported

        max_partition_size (int) : The maximal number of qubits in the partitions

        strategy (PartitionStrategy, optional) Set to ilp (slow, but giving optimal result), tdag, or kahn (default)
    
    Return:

        Returns with the paritioned circuit and the associated parameter array. Partitions are organized into subcircuits of the resulting circuit
    """ 

    circ, parameters = utils.qasm_to_squander_circuit(filename)
    return PartitionCircuit( circ, parameters, max_partition_size, strategy, filename )




if __name__ == "__main__":
    PartitionCircuitQasm("examples/partitioning/qasm_samples/heisenberg-16-20.qasm", 4)
