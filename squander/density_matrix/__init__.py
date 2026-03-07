"""
Density Matrix Module for SQUANDER

Provides mixed-state quantum simulation with noise modeling.
Integrated as a subpackage of SQUANDER.

Usage:
    from squander.density_matrix import DensityMatrix, NoisyCircuit
    from squander.density_matrix import DepolarizingChannel
    
Example:
    # Create 2-qubit density matrix
    rho = DensityMatrix(qbit_num=2)
    
    # Create circuit
    circuit = NoisyCircuit(2)
    circuit.add_H(0)
    circuit.add_CNOT(1, 0)
    
    # Apply circuit
    import numpy as np
    circuit.apply_to(np.array([]), rho)
    
    # Add noise
    noise = DepolarizingChannel(qbit_num=2, error_rate=0.01)
    noise.apply(rho)
    
    print(f"Purity: {rho.purity()}")
"""

__version__ = "1.0.0"

# Import C++ bindings
from ._density_matrix_cpp import (
    DensityMatrix,
    NoisyCircuit,
    NoiseChannel,
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
)

__all__ = [
    # Core classes
    "DensityMatrix",
    "NoisyCircuit",
    
    # Noise channels
    "NoiseChannel",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
]

