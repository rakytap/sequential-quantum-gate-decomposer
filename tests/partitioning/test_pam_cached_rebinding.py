"""Resolved PAM programs remain usable in symmetry-equivalent physical frames."""
from dataclasses import replace

import numpy as np
import pytest
from bqskit.passes import PAMLayoutPass

from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing


def _swap_alternative():
    source = qgd_Circuit(2)
    source.add_CNOT(1, 0)
    source.add_CNOT(0, 1)
    source.add_CNOT(1, 0)
    parameters = np.empty(0, dtype=np.float64)
    payload = routing.SynthesizedRoutingPayload(
        circuit=source, parameters=parameters, topology=((0, 1),),
        input_assignment=(0, 1), output_assignment=(0, 1),
        source_circuit=source, source_parameters=parameters,
    )
    return routing.RoutingAlternative(
        0, (0, 1), (0, 1), (0, 1), 3, payload=payload,
    )


def test_pam_retains_relabelled_cached_program_and_registers_exact_column(monkeypatch):
    original = _swap_alternative()
    # Distinct ownership of the same resolved unitary at the reflected boundary.
    reflected = replace(
        original, input_physical=(1, 0), output_physical=(1, 0),
        payload=replace(original.payload,
                        circuit=original.payload.circuit.get_Flat_Circuit(),
                        parameters=original.payload.parameters.copy(),
                        input_assignment=(1, 0), output_assignment=(1, 0)),
    )
    alternatives = {0: (original, reflected)}
    registered = []

    def register(column):
        registered.append(column)
        alternatives[column.partition] += (column,)

    # Hold the initial physical frame fixed; exercise the actual PAM table lookup.
    monkeypatch.setattr(PAMLayoutPass, 'forward_pass', lambda *args: None)
    monkeypatch.setattr(PAMLayoutPass, 'backward_pass', lambda *args: None)
    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0,), gate_predecessors={0: (), 1: (0,), 2: (1,)},
        partitions=(frozenset((0, 1, 2)),), alternatives=alternatives,
        logical_qubit_count=2, topology=((0, 1),), path=(0, 1),
        initial_mapping=(1, 0), layout_passes=1,
        register_alternative=register,
    )
    assert result is not None
    assert result.initial_mapping == result.final_mapping == (1, 0)
    assert result.cnot_count == 3
    chosen = result.selections[0].alternative
    assert len(registered) == 1 and registered[0] is chosen
    assert chosen.payload is original.payload
    assert chosen is not reflected
    fitted, parameters = routing.construct_exact_routed_circuit(result, 2)
    np.testing.assert_allclose(
        fitted.get_Matrix(parameters),
        original.payload.circuit.get_Matrix(original.payload.parameters), atol=1e-14,
    )
    # The generic flow master requires the identical column object in its model.
    import squander.partitioning.ilp as ilp
    monkeypatch.setattr(ilp, '_GUROBI_AVAILABLE', False)
    monkeypatch.setattr(ilp, '_GUROBI_FALLBACK_WARNING_PRINTED', True)
    exact = routing._solve_exact_routing_ilp_flow_restricted(
        gate_predecessors={0: (), 1: (0,), 2: (1,)},
        partitions=(frozenset((0, 1, 2)),), alternatives=alternatives,
        logical_qubit_count=2, topology=((0, 1),),
        initial_mapping=(1, 0), warm_start=result, timeout_seconds=5,
    )
    assert exact.cnot_count == 3 and exact.optimal


@pytest.mark.parametrize('before,after,topology,message', [
    ((1, 0), (0, 1), ((0, 1),), 'boundary permutations'),
    ((0, 2), (0, 2), ((0, 1), (1, 2)), 'physical topology'),
])
def test_cached_pam_rebinding_rejects_incompatible_frame(before, after, topology, message):
    with pytest.raises(RuntimeError, match=message):
        routing._rebind_cached_pam_alternative(
            _swap_alternative(), (0, 1), before, after, topology,
        )


def test_cached_pam_abandons_unproductive_swap_trajectory(monkeypatch):
    alternative = _swap_alternative()
    attempts = []
    monkeypatch.setattr(PAMLayoutPass, '_get_best_swap',
                        lambda *args, **kwargs: attempts.append(1) or (0, 1))

    def stuck_pass(self, *args):
        for _ in range(10_000):
            self._get_best_swap(None, (0,))

    monkeypatch.setattr(PAMLayoutPass, 'forward_pass', stuck_pass)
    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0,), gate_predecessors={0: (), 1: (0,), 2: (1,)},
        partitions=(frozenset((0, 1, 2)),), alternatives={0: (alternative,)},
        logical_qubit_count=2, topology=((0, 1),), path=(0, 1), layout_passes=1,
    )
    assert result is None
    assert len(attempts) < 1000
