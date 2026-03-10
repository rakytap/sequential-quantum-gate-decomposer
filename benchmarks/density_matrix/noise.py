from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a 3-qubit quantum circuit
qc = QuantumCircuit(3, 3)

# Add some quantum gates
qc.h(0)  # Hadamard on qubit 0
qc.cx(0, 1)  # CNOT from qubit 0 to qubit 1
qc.cx(1, 2)  # CNOT from qubit 1 to qubit 2

# Add some more gates to make it interesting
qc.rx(0.5, 0)  # Rotation around X on qubit 0
qc.ry(0.3, 1)  # Rotation around Y on qubit 1
qc.rz(0.7, 2)  # Rotation around Z on qubit 2

# Measure all qubits
qc.measure([0, 1, 2], [0, 1, 2])

print("Circuit:")
print(qc.draw())

# ============ Create a Noise Model ============

noise_model = NoiseModel()

# 1. Depolarizing error on single-qubit gates
# This adds random Pauli errors (X, Y, Z) with some probability
single_qubit_error_rate = 0.01  # 1% error rate
single_qubit_error = depolarizing_error(single_qubit_error_rate, 1)

# 2. Depolarizing error on two-qubit gates (typically higher)
two_qubit_error_rate = 0.05  # 5% error rate
two_qubit_error = depolarizing_error(two_qubit_error_rate, 2)

# 3. Thermal relaxation error (T1/T2 decoherence)
# T1: relaxation time, T2: dephasing time, gate_time: how long the gate takes
t1 = 50e3  # 50 microseconds
t2 = 30e3  # 30 microseconds (must be <= 2*T1)
gate_time_1q = 50  # 50 ns for single-qubit gates
gate_time_2q = 300  # 300 ns for two-qubit gates

thermal_error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
thermal_error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
    thermal_relaxation_error(t1, t2, gate_time_2q)
)

# Add errors to specific gates
noise_model.add_all_qubit_quantum_error(
    single_qubit_error, ['h', 'rx', 'ry', 'rz'])
noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

# Optionally add thermal relaxation errors too
# noise_model.add_all_qubit_quantum_error(thermal_error_1q, ['h', 'rx', 'ry', 'rz'])
# noise_model.add_all_qubit_quantum_error(thermal_error_2q, ['cx'])

print("\nNoise Model:")
print(noise_model)

# ============ Run Simulations ============

shots = 4096

# Ideal simulation (no noise)
ideal_simulator = AerSimulator()
transpiled_ideal = transpile(qc, ideal_simulator)
ideal_result = ideal_simulator.run(transpiled_ideal, shots=shots).result()
ideal_counts = ideal_result.get_counts()

# Noisy simulation
noisy_simulator = AerSimulator(noise_model=noise_model)
transpiled_noisy = transpile(qc, noisy_simulator)
noisy_result = noisy_simulator.run(transpiled_noisy, shots=shots).result()
noisy_counts = noisy_result.get_counts()

# ============ Display Results ============

print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)

print("\nIdeal (noiseless) counts:")
for state, count in sorted(ideal_counts.items(), key=lambda x: -x[1]):
    print(f"  |{state}>: {count} ({100*count/shots:.2f}%)")

print("\nNoisy counts:")
for state, count in sorted(noisy_counts.items(), key=lambda x: -x[1]):
    print(f"  |{state}>: {count} ({100*count/shots:.2f}%)")

# Plot histograms side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot ideal results
ax1 = axes[0]
states = sorted(ideal_counts.keys())
ideal_values = [ideal_counts.get(s, 0) for s in states]
ax1.bar(states, ideal_values, color='steelblue')
ax1.set_title('Ideal (Noiseless) Simulation', fontsize=14)
ax1.set_xlabel('Quantum State', fontsize=12)
ax1.set_ylabel('Counts', fontsize=12)
ax1.tick_params(axis='x', rotation=45)

# Plot noisy results
ax2 = axes[1]
all_states = sorted(set(ideal_counts.keys()) | set(noisy_counts.keys()))
noisy_values = [noisy_counts.get(s, 0) for s in all_states]
ax2.bar(all_states, noisy_values, color='coral')
ax2.set_title('Noisy Simulation', fontsize=14)
ax2.set_xlabel('Quantum State', fontsize=12)
ax2.set_ylabel('Counts', fontsize=12)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('noisy_simulation_results.png', dpi=150)
plt.show()

print("\nPlot saved to 'noisy_simulation_results.png'")
