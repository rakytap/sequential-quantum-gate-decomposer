#!/usr/bin/env python3
"""
Basic Usage Example for SQUANDER Density Matrix Module

Demonstrates:
- Creating density matrices
- Applying quantum circuits
- Adding noise
- Computing quantum properties
"""

import numpy as np
import sys

try:
    from squander.density_matrix import (
        DensityMatrix,
        NoisyCircuit,
        DepolarizingChannel,
        AmplitudeDampingChannel,
    )
except ImportError as e:
    print(f"Error: Could not import density matrix module: {e}")
    print("Please build the module first: python setup.py build_ext")
    sys.exit(1)


def example_1_pure_state_evolution():
    """Example 1: Pure state evolution."""
    print("\n" + "="*60)
    print("Example 1: Pure State Evolution")
    print("="*60)
    
    # Create 2-qubit density matrix (initialized to |00⟩)
    rho = DensityMatrix(qbit_num=2)
    print(f"Initial state: |00⟩⟨00|")
    print(f"  Purity: {rho.purity():.4f} (pure state)")
    print(f"  Entropy: {rho.entropy():.4f} bits")
    
    # Create Bell state circuit
    circuit = NoisyCircuit(2)
    circuit.add_H(0)          # Hadamard on qubit 0
    circuit.add_CNOT(1, 0)    # CNOT with control=0, target=1
    
    # Apply circuit
    circuit.apply_to(np.array([]), rho)
    
    print(f"\nAfter Bell state circuit:")
    print(f"  Purity: {rho.purity():.4f} (still pure)")
    print(f"  Entropy: {rho.entropy():.4f} bits")
    
    # Get density matrix
    rho_np = rho.to_numpy()
    print(f"\nDensity matrix shape: {rho_np.shape}")
    print(f"Diagonal elements: {np.diag(rho_np).real}")


def example_2_noise_simulation():
    """Example 2: Simulation with noise."""
    print("\n" + "="*60)
    print("Example 2: Noise Simulation")
    print("="*60)
    
    # Create circuit
    circuit = NoisyCircuit(2)
    circuit.add_H(0)
    circuit.add_CNOT(1, 0)
    
    # Start with pure state
    rho = DensityMatrix(qbit_num=2)
    circuit.apply_to(np.array([]), rho)
    
    print(f"After circuit (no noise):")
    print(f"  Purity: {rho.purity():.4f}")
    
    # Add depolarizing noise
    noise = DepolarizingChannel(qbit_num=2, error_rate=0.05)
    noise.apply(rho)
    
    print(f"\nAfter 5% depolarizing noise:")
    print(f"  Purity: {rho.purity():.4f} (< 1, now mixed)")
    print(f"  Entropy: {rho.entropy():.4f} bits (> 0)")
    
    # Add more noise
    for i in range(10):
        noise.apply(rho)
    
    print(f"\nAfter repeated noise application:")
    print(f"  Purity: {rho.purity():.4f}")
    print(f"  Approaching maximally mixed (purity → 0.25)")


def example_3_t1_t2_noise():
    """Example 3: T1 and T2 noise."""
    print("\n" + "="*60)
    print("Example 3: T1 and T2 Noise")
    print("="*60)
    
    # Create superposition state |+⟩ on qubit 0
    psi = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    rho = DensityMatrix(psi)
    
    rho_np = rho.to_numpy()
    coherence_initial = abs(rho_np[0, 1])
    
    print(f"Initial state: |+0⟩")
    print(f"  Coherence ρ(0,1): {coherence_initial:.4f}")
    print(f"  Purity: {rho.purity():.4f}")
    
    # Apply T1 noise (amplitude damping)
    t1_noise = AmplitudeDampingChannel(target_qbit=0, gamma=0.1)
    t1_noise.apply(rho)
    
    print(f"\nAfter T1 noise (10% decay):")
    print(f"  Purity: {rho.purity():.4f}")
    
    # Apply T2 noise (phase damping) would further reduce coherence
    # (Will be fully implemented in Phase 1)


def example_4_maximally_mixed():
    """Example 4: Maximally mixed state."""
    print("\n" + "="*60)
    print("Example 4: Maximally Mixed State")
    print("="*60)
    
    # Create maximally mixed state
    rho = DensityMatrix.maximally_mixed(qbit_num=2)
    
    print(f"Maximally mixed state (2 qubits):")
    print(f"  Purity: {rho.purity():.4f} (= 1/2² = 0.25)")
    print(f"  Entropy: {rho.entropy():.4f} bits (= log₂(4) = 2)")
    
    # Eigenvalues
    eigs = rho.eigenvalues()
    print(f"  Eigenvalues: {eigs} (all equal for maximally mixed)")


def example_5_partial_trace():
    """Example 5: Partial trace."""
    print("\n" + "="*60)
    print("Example 5: Partial Trace")
    print("="*60)
    
    # Create Bell state
    circuit = NoisyCircuit(2)
    circuit.add_H(0)
    circuit.add_CNOT(1, 0)
    
    rho_full = DensityMatrix(qbit_num=2)
    circuit.apply_to(np.array([]), rho_full)
    
    print(f"Full Bell state (2 qubits):")
    print(f"  Purity: {rho_full.purity():.4f} (pure)")
    
    # Trace out qubit 1
    rho_reduced = rho_full.partial_trace([1])
    
    print(f"\nReduced state (qubit 0 only):")
    print(f"  Purity: {rho_reduced.purity():.4f} (maximally mixed)")
    print(f"  Entropy: {rho_reduced.entropy():.4f} bits")
    
    print(f"\nThis demonstrates entanglement: tracing out one qubit")
    print(f"of an entangled pair gives a maximally mixed state.")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("SQUANDER Density Matrix Module - Basic Usage Examples")
    print("="*60)
    
    try:
        example_1_pure_state_evolution()
        example_2_noise_simulation()
        example_3_t1_t2_noise()
        example_4_maximally_mixed()
        example_5_partial_trace()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

