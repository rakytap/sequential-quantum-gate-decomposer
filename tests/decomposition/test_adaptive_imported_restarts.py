"""Regression cases for bounded adaptive imported-structure restarts."""

import numpy as np
import pytest
from scipy.io import loadmat

from squander import N_Qubit_Decomposition_adaptive, utils


@pytest.mark.parametrize("seed", [0, 4])
def test_randomized_imported_layers_recover_from_bad_initial_fit(seed):
    target = loadmat("data/Umtx.mat")["Umtx"]
    config = {
        "max_outer_iterations": 10,
        "max_inner_iterations": 300000,
        "max_inner_iterations_compression": 10000,
        "max_inner_iterations_final": 1000,
        "randomized_adaptive_layers": 1,
        "random_seed": seed,
        "optimization_tolerance": 1e-8,
    }
    decomposition = N_Qubit_Decomposition_adaptive(target.conj().T, config=config)
    for _ in range(3):
        decomposition.add_Adaptive_Layers()
    decomposition.add_Finalyzing_Layer_To_Gate_Structure()

    parameters = np.random.default_rng(42).random(decomposition.get_Parameter_Num())
    decomposition.set_Optimized_Parameters(parameters * (2 * np.pi))
    decomposition.get_Initial_Circuit()
    decomposition.Finalize_Circuit()

    circuit = decomposition.get_Qiskit_Circuit()
    reconstructed = np.asarray(utils.get_unitary_from_qiskit_circuit(circuit))
    product = target @ reconstructed.conj().T
    aligned = product * np.exp(-1j * np.angle(product[0, 0]))
    error = np.real(np.trace(2 * np.eye(16) - aligned - aligned.conj().T)) / 2
    assert error < 1e-3
