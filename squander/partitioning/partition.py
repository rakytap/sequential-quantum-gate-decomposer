from typing import Literal
import numpy as np

from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander import utils

from squander.partitioning.kahn import kahn_partition
from squander.partitioning.ilp import ilp_max_partitions
from squander.partitioning.tools import translate_param_order

PartitionStrategy = Literal["kahn", "ilp", "tdag"]

PARTITION_FUNCTIONS = {
    "kahn": kahn_partition,
    "ilp": ilp_max_partitions, 
}



def PartitionCircuit( circ: Circuit, parameters: np.ndarray, max_partition_size: int, strategy : PartitionStrategy = "kahn" ) -> tuple[Circuit, np.ndarray]:
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
       
    func = PARTITION_FUNCTIONS.get(strategy, kahn_partition)
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
    return PartitionCircuit( circ, parameters, max_partition_size, strategy )



if __name__ == "__main__":
    PartitionCircuitQasm("examples/partitioning/qasm_samples/heisenberg-16-20.qasm", 4)
