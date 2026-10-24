"""SABRE must resolve routing cycles without dropping or changing gates."""

import importlib

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from squander import Qiskit_IO
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.synthesis.qgd_SABRE import qgd_SABRE


backend = importlib.import_module(
    "squander.decomposition.qgd_Wide_Circuit_Optimization"
)


def assert_exact_route(before, mapped, parameters, initial, final, topology):
    after = backend._squander_audit_representation(
        mapped, parameters, range(mapped.get_Qbit_Num())
    )
    actions, replayed = backend._qiskit_sabre_actions(
        backend._qasm_exact_state(before), backend._qasm_exact_state(after),
        list(map(int, initial)), topology,
    )
    width = mapped.get_Qbit_Num()
    assert sorted(initial) == list(range(width))
    assert sorted(final) == list(range(width))
    assert replayed == list(map(int, final))
    assert actions


def assert_dense_equivalence(target, mapped, parameters, initial, final):
    width = mapped.get_Qbit_Num()

    def basis_indices(mapping):
        return [sum(((basis >> logical) & 1) << int(physical)
                    for logical, physical in enumerate(mapping))
                for basis in range(1 << width)]

    actual = Operator(Qiskit_IO.get_Qiskit_Circuit(mapped, parameters)).data
    actual = actual[np.ix_(basis_indices(final), basis_indices(initial))]
    overlap = np.vdot(target, actual)
    phase = overlap / abs(overlap)
    assert np.linalg.norm(actual - phase * target, "fro") < 1e-10


@pytest.mark.parametrize("width", [6, 64])
def test_four_swap_cycle_forces_a_front_gate_to_be_executable(width, monkeypatch):
    circuit = qgd_Circuit(width)
    circuit.add_U3(2)
    circuit.add_CNOT(width - 1, 0)
    parameters = np.array([0.17, -0.03, 0.29])
    before = backend._squander_audit_representation(circuit, parameters, range(width))
    target = (Operator(Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)).data
              if width <= 6 else None)
    topology = [(q, q + 1) for q in range(width - 1)]
    router = qgd_SABRE(circuit, topology, initial_layout="trivial")
    cycle = {
        (0, width - 1): (0, 1),
        (1, width - 1): (width - 2, width - 1),
        (1, width - 2): (0, 1),
        (0, width - 2): (width - 2, width - 1),
    }

    def cyclic_score(swap, F, E, pi, DAG, IDAG, resolved_gates):
        preferred = cycle[(int(pi[0]), int(pi[width - 1]))]
        return 0.0 if swap == preferred else 100.0

    monkeypatch.setattr(router, "score_swap_improved", cyclic_score)
    original = router.obtain_SWAPS
    calls = []

    def bounded_obtain(F, pi, DAG):
        # Fail quickly against the old infinite loop, not by a CI timeout.
        assert len(calls) < 16, "SABRE repeated a four-SWAP cycle without progress"
        calls.append(tuple(map(int, pi)))
        return original(F, pi, DAG)

    monkeypatch.setattr(router, "obtain_SWAPS", bounded_obtain)
    mapped, values, initial, final, swaps = router.map_circuit(parameters, iter_num=0)
    assert 0 < swaps <= width + 8
    assert_exact_route(before, mapped, values, initial, final, topology)
    if target is not None:
        assert_dense_equivalence(target, mapped, values, initial, final)


@pytest.mark.parametrize("width", [2, 3, 4, 6])
def test_already_executable_gates_do_not_use_cycle_escape(width, monkeypatch):
    circuit = qgd_Circuit(width)
    circuit.add_U3(0)
    for q in range(width - 1):
        circuit.add_CNOT(q + 1, q)
    parameters = np.array([0.23, 0.11, -0.37])
    target = Operator(Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)).data
    before = backend._squander_audit_representation(circuit, parameters, range(width))
    topology = [(q, q + 1) for q in range(width - 1)]
    router = qgd_SABRE(circuit, topology, initial_layout="trivial")

    def unexpected_escape(*args):
        raise AssertionError("An executable route must not need a cycle escape")

    monkeypatch.setattr(router, "_route_front_gate", unexpected_escape)
    mapped, values, initial, final, swaps = router.map_circuit(parameters)
    assert swaps == 0
    assert_exact_route(before, mapped, values, initial, final, topology)
    assert_dense_equivalence(target, mapped, values, initial, final)


@pytest.mark.parametrize("graph", ["path", "star", "complete"])
def test_routed_parameterized_circuits_preserve_the_full_unitary(graph):
    width = 5
    circuit = qgd_Circuit(width)
    rng = np.random.default_rng(20261015)
    values = []
    for _ in range(12):
        target, control = map(int, rng.choice(width, 2, replace=False))
        circuit.add_U3(target)
        values.extend(rng.uniform(-1, 1, 3))
        circuit.add_CNOT(target, control)
    parameters = np.array(values)
    before = backend._squander_audit_representation(circuit, parameters, range(width))
    target = Operator(Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)).data
    topology = ({"path": [(q, q + 1) for q in range(width - 1)],
                 "star": [(0, q) for q in range(1, width)],
                 "complete": [(q, r) for q in range(width) for r in range(q + 1, width)]}
                [graph])
    router = qgd_SABRE(circuit, topology, random_seed=20261015)
    mapped, routed_values, initial, final, _ = router.map_circuit(parameters)
    assert_exact_route(before, mapped, routed_values, initial, final, topology)
    assert_dense_equivalence(target, mapped, routed_values, initial, final)


def test_disconnected_routing_fails_instead_of_returning_a_partial_circuit():
    circuit = qgd_Circuit(4)
    circuit.add_CNOT(3, 0)
    router = qgd_SABRE(circuit, [(0, 1), (2, 3)], initial_layout="trivial")
    with pytest.raises(ValueError, match="disconnected|cannot be mapped"):
        router.map_circuit(np.array([]), iter_num=0)
