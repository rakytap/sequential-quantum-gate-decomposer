"""Heisenberg VQE example and shot-noise energy estimator.

This example constructs a Heisenberg-style Hamiltonian (ZZ/XX/YY couplings
plus local Z fields), builds an instance of the C++-backed
`qgd_Variational_Quantum_Eigensolver_Base` wrapper, applies a parameterized
ansatz, and demonstrates both exact (ideal) and shot-noise energy estimation.

The file shows how to format the term lists passed to the C++ wrapper:
`zz_terms`, `xx_terms`, `yy_terms` each contain items of the form
`{"i": int, "j": int, "coeff": float}` and `z_terms` contains
`{"i": int, "coeff": float}`. The shot-noise API expects a dictionary
containing these lists together with `shots`, `p_readout`, and `seed`.

This script is intended for interactive runs and manual inspection. For
lightweight automated checks that do not require compiling the native
extension, see `tests/VQE/test_shot_noise_measurement.py` which provides
fallback implementations and fast unit tests.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import scipy as sp
from qiskit.quantum_info import SparsePauliOp
from networkx.generators.random_graphs import random_regular_graph

np.set_printoptions(linewidth=200)

# =======================================
#  Generate ZZ Hamiltonian + field terms
# =======================================

# Import the C++-backed VQE class
import numpy as np
import scipy as sp
from qiskit.quantum_info import SparsePauliOp

from squander.variational_quantum_eigensolver.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base as Variational_Quantum_Eigensolver,
)

def generate_zz_xx_hamiltonian(n_qubits, h=0.5, topology=None, Jz=1.0, Jx=1.0, Jy=1.0):
    """Create a Heisenberg-like Hamiltonian and a Pauli-format oplist.

    The Hamiltonian built is:
        H = sum_{(i,j) in topology} [ Jz Z_i Z_j + Jx X_i X_j + Jy Y_i Y_j ]
            + sum_i h_i Z_i

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    h : float or array-like, optional
        Local Z field magnitude (scalar for uniform field or array-like for
        per-qubit fields). Default 0.5.
    topology : sequence of (i, j) pairs, optional
        Edge list describing which qubit pairs are coupled. If ``None``, a
        nearest-neighbour ring topology is used.
    Jz, Jx, Jy : float, optional
        Coupling strengths for the ZZ, XX and YY terms respectively.

    Returns
    -------
    H_sparse : scipy.sparse.csr_matrix
        The full Hamiltonian as a sparse matrix in the computational (Z)
        basis. Suitable for small-system exact calculations.
    oplist : list
        A list of tuples describing Pauli terms in the form expected by the
        wrapper and by :class:`qiskit.quantum_info.SparsePauliOp`:
        ("ZZ", [i, j], coeff), ("XX", [i, j], coeff), ("YY", [i, j], coeff),
        and ("Z", [i], coeff).

    Notes
    -----
    The returned ``oplist`` is used to build dictionaries passed to the C++
    wrapper shot-noise routine (see ``zz_terms_for_dict`` etc. in
    :func:`main`). The sparse Hamiltonian is convenient for exact energy
    evaluation using ``scipy.sparse.linalg.eigsh`` for small systems.
    """

    if topology is None:
        topology = [[i, (i + 1) % n_qubits] for i in range(n_qubits)]

    # normalize h into a vector
    if np.isscalar(h):
        h_vec = np.full(n_qubits, float(h))
    else:
        h_vec = np.asarray(h, dtype=float)

    oplist = []

    # --- ZZ, XX, and YY couplings ---
    for i, j in topology:
        # ZZ term
        oplist.append(("ZZ", [i, j], float(Jz)))

        # XX term
        oplist.append(("XX", [i, j], float(Jx)))
        
        # YY term
        oplist.append(("YY", [i, j], float(Jy)))

    # --- Magnetic field (Z terms) ---
    for i in range(n_qubits):
        oplist.append(("Z", [i], float(h_vec[i])))

    # Convert to SparsePauliOp
    H_sparse = SparsePauliOp.from_sparse_list(oplist, num_qubits=n_qubits)
    H_sparse = H_sparse.to_matrix(sparse=True).tocsr()

    return H_sparse, oplist


def main():
    """Run a full VQE example and shot-noise energy estimation.

    Workflow:
    - Build a Heisenberg Hamiltonian and decompose it to term lists
      consumable by the C++ wrapper.
    - Instantiate the wrapped VQE object and configure optimizer/ansatz.
    - Generate and apply the circuit to obtain the final state |psi>.
    - Compute exact (ideal) energies and per-term contributions.
    - Run the shot-noise estimator via the wrapper and print a summary.

    This function prints human-readable diagnostics. For automated tests
    prefer using the unit-tests under ``tests/VQE`` which include
    lightweight fallbacks when the native extension is unavailable.
    """

    # ----------------------
    #   Simulation config
    # ----------------------
    n_qubits = 3
    h = 0.5
    topology = None  # ring if None

    Hamiltonian, oplist = generate_zz_xx_hamiltonian(n_qubits, h=h, topology=topology)

    # ----------------------
    #   Split term lists
    # ----------------------
    zz_terms_for_dict = [
        {"i": t[1][0], "j": t[1][1], "coeff": float(t[2])}
        for t in oplist if t[0] == "ZZ"
    ]

    xx_terms_for_dict = [
        {"i": t[1][0], "j": t[1][1], "coeff": float(t[2])}
        for t in oplist if t[0] == "XX"
    ]

    yy_terms_for_dict = [
        {"i": t[1][0], "j": t[1][1], "coeff": float(t[2])}
        for t in oplist if t[0] == "YY"
    ]

    z_terms_for_dict = [
        {"i": t[1][0], "coeff": float(t[2])}
        for t in oplist if t[0] == "Z"
    ]

    # --- Build separate Hamiltonians for ZZ, XX, YY, Z parts (for analysis) ---
    zz_ops = [(t[0], t[1], t[2]) for t in oplist if t[0] == "ZZ"]
    xx_ops = [(t[0], t[1], t[2]) for t in oplist if t[0] == "XX"]
    yy_ops = [(t[0], t[1], t[2]) for t in oplist if t[0] == "YY"]
    z_ops  = [(t[0], t[1], t[2]) for t in oplist if t[0] == "Z"]

    H_zz = (
        SparsePauliOp.from_sparse_list(zz_ops, num_qubits=n_qubits)
        .to_matrix(sparse=True)
        .tocsr()
        if zz_ops else None
    )
    H_xx = (
        SparsePauliOp.from_sparse_list(xx_ops, num_qubits=n_qubits)
        .to_matrix(sparse=True)
        .tocsr()
        if xx_ops else None
    )
    H_yy = (
        SparsePauliOp.from_sparse_list(yy_ops, num_qubits=n_qubits)
        .to_matrix(sparse=True)
        .tocsr()
        if yy_ops else None
    )
    H_z = (
        SparsePauliOp.from_sparse_list(z_ops, num_qubits=n_qubits)
        .to_matrix(sparse=True)
        .tocsr()
        if z_ops else None
    )

    # Reference ground energy (small n). Use eigsh since H is Hermitian.
    # 'SA' = smallest algebraic
    eigvals, _ = sp.sparse.linalg.eigsh(Hamiltonian, k=1, which="SA")
    target_e = float(np.real(eigvals[0]))

    # ----------------------
    #   Build & apply VQE
    # ----------------------
    config = {"max_inner_iterations": 200, "batch_size": 64, "convergence_length": 10}
    VQE = Variational_Quantum_Eigensolver(Hamiltonian, n_qubits, config, accelerator_num=0)
    VQE.set_Optimizer("COSINE")
    VQE.set_Ansatz("HEA_ZYZ")
    VQE.Generate_Circuit(layers=2, inner_blocks=1)

    param_num = VQE.get_Parameter_Num()
    parameters = np.random.randn(param_num) * 2 * np.pi
    VQE.set_Optimized_Parameters(parameters)

    # Initial |0...0> state
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[0] = 1.0

    # set the internal initial state used by the shot-noise path
    VQE.set_Initial_State(state)

    # Apply circuit in-place (C++ mutates `state`) to get |ψ>
    VQE.apply_to(parameters, state)

    # Sanity checks
    prob_sum = float(np.sum(np.abs(state) ** 2))
    if not np.isclose(prob_sum, 1.0, atol=1e-9):
        raise RuntimeError(f"State not normalized after apply_to: sum|amp|^2 = {prob_sum}")

    if Hamiltonian.shape != (len(state), len(state)):
        raise RuntimeError(
            f"Dimension mismatch: H is {Hamiltonian.shape}, state length {len(state)} "
            f"(expected {(1<<n_qubits)})."
        )

    # ---------------------------------------------------
    #  Show the final VQE state |ψ> in the Z basis
    # ---------------------------------------------------
    print("\n================== Final VQE state |ψ⟩ (Z basis) ==================")
    print(f"(Only amplitudes with |amp| > 1e-3 are shown; norm^2 = {prob_sum:.12f})\n")

    threshold = 1e-3
    for idx, amp in enumerate(state):
        if np.abs(amp) > threshold:
            bitstring = format(idx, f"0{n_qubits}b")
            print(f"|{bitstring}> : {amp.real:+.6f} + {amp.imag:+.6f}i")
    print("==================================================================\n")

    # ===================================================
    #   Ideal (exact) energy with this |ψ>
    # ===================================================
    ideal_energy = VQE.Expectation_value_of_energy_real(state, state)
    energy_ref = float(np.vdot(state, Hamiltonian.dot(state)).real)
    diff = ideal_energy - energy_ref

    # --- Per-part ideal energies: ZZ, XX, YY, Z ---
    def exp_val(part_H):
        if part_H is None:
            return 0.0
        return float(np.vdot(state, part_H.dot(state)).real)

    E_zz = exp_val(H_zz)
    E_xx = exp_val(H_xx)
    E_yy = exp_val(H_yy)
    E_z  = exp_val(H_z)
    E_sum_parts = E_zz + E_xx + E_yy + E_z

    print("\n============================================================")
    print("🧮  Quantum Heisenberg VQE Simulation Results")
    print("============================================================")
    print(f"Target ground state eigenvalue: {target_e:.8f}")
    print(f"Ideal energy ⟨ψ|H|ψ⟩ (wrapper): {ideal_energy:.8f}  | direct: {energy_ref:.8f}  | Δ = {diff:.2e}")

    print("\n--- Decomposition of ⟨ψ|H|ψ⟩ into ZZ / XX / YY / Z parts (ideal) ---")
    print(f"E_ZZ = {E_zz: .8f}")
    print(f"E_XX = {E_xx: .8f}")
    print(f"E_YY = {E_yy: .8f}")
    print(f"E_Z  = {E_z: .8f}")
    print(f"E_ZZ + E_XX + E_YY + E_Z = {E_sum_parts: .8f}")

    # ===================================================
    #   Shot-noise run setup
    # ===================================================
    shots = 10000
    p_readout = 0.0
    seed = 42

    input_dict = {
        "shots": shots,
        "p_readout": p_readout,
        "zz_terms": zz_terms_for_dict,
        "xx_terms": xx_terms_for_dict,
        "yy_terms": yy_terms_for_dict,
        "z_terms": z_terms_for_dict,
        "seed": seed,
    }

    # ===================================================
    #   Debug info
    # ===================================================
    max_zz_idx = max(max(t["i"], t["j"]) for t in zz_terms_for_dict) if zz_terms_for_dict else -1
    max_xx_idx = max(max(t["i"], t["j"]) for t in xx_terms_for_dict) if xx_terms_for_dict else -1
    max_yy_idx = max(max(t["i"], t["j"]) for t in yy_terms_for_dict) if yy_terms_for_dict else -1
    max_z_idx = max(t["i"] for t in z_terms_for_dict) if z_terms_for_dict else -1

    print("------------------------------------------------------------")
    print(f"Qubits: {n_qubits}")
    print(f"ZZ terms: {len(zz_terms_for_dict)}  (max index: {max_zz_idx})")
    print(f"XX terms: {len(xx_terms_for_dict)}  (max index: {max_xx_idx})")
    print(f"YY terms: {len(yy_terms_for_dict)}  (max index: {max_yy_idx})")
    print(f"Z  terms: {len(z_terms_for_dict)}  (max index: {max_z_idx})")
    print(f"State shape: {state.shape}  |  Hamiltonian shape: {Hamiltonian.shape}")
    print(f"Shots per measurement basis (Z, X, and Y): {shots}")


    # ===================================================
    #   Run shot-noise energy estimation
    # ===================================================
    result = VQE.Expectation_Value_Shot_Noise(input_dict)

    mean_energy = result["mean"]
    variance = result["variance"]
    std_error = result["std_error"]
    delta = mean_energy - ideal_energy

    print("------------------------------------------------------------")
    print(f"Shot-noise energy ({shots} shots, {p_readout*100:.2f}% readout error): {mean_energy:.6f}")
    print(f"Variance: {variance:.6f}  |  Std. error: {std_error:.6f}")
    print(f"Difference (noisy − ideal): {delta:+.6f}")

    # ===================================================
    #   Summary interpretation
    # ===================================================
    rel_change = (delta / ideal_energy) * 100 if abs(ideal_energy) > 1e-10 else 0.0
    direction = "decrease" if delta < 0 else "increase"
    print("------------------------------------------------------------")
    print(f"Summary: Noise introduced a {direction} of {abs(delta):.6f} "
          f"in energy ({abs(rel_change):.2f}% relative change).")
    print("============================================================\n")


if __name__ == "__main__":
    main()

