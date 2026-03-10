from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error

# 1. Create the circuit
qc = QuantumCircuit(2)

# --- FIRST INSTANCE (Noisy) ---
qc.h(0)
# Manually append the error instruction immediately after the gate
error_1q = depolarizing_error(0.2, 1)
qc.append(error_1q, [0])  # This effectively makes the previous H gate noisy

# --- SECOND INSTANCE (Ideal) ---
qc.h(0)
# We do NOT append an error here, so this H remains ideal

qc.measure_all()

# 2. Simulate
# Note: We do NOT need to pass a noise_model argument because
# the noise is already baked into the circuit instructions.
sim = AerSimulator()
result = sim.run(qc).result()
counts = result.get_counts()

print("Circuit with manually inserted noise:")
print(qc)
