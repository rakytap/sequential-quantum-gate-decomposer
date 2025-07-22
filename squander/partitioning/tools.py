from typing import Set, List, Tuple, Dict
import numpy as np

from squander.gates.gates_Wrapper import Gate
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit



def get_qubits(gate: Gate) -> Set[int]:
    """
    Retrieves qubit indices used by a SQUANDER gate.
    
    Args:
        gate: The SQUANDER gate
        
    Returns:
        Set of qubit indices used by the gate
    """
    return {gate.get_Target_Qbit()} | ({control} if (control := gate.get_Control_Qbit()) != -1 else set())



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



def build_dependency(c: Circuit) -> Tuple[Dict[int, Gate], Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Build dependency graphs for circuit gates.
    
    Args:
        c: SQUANDER Circuit
        
    Returns:
        tuple: (gate_dict, forward_graph, reverse_graph)
    """
    gate_dict = {i: gate for i, gate in enumerate(c.get_Gates())}
    g, rg = {i: set() for i in gate_dict}, {i: set() for i in gate_dict}
    
    for gate in gate_dict:
        for child in c.get_Children(gate_dict[gate]):
            g[gate].add(child)
            rg[child].add(gate)
    
    return gate_dict, g, rg