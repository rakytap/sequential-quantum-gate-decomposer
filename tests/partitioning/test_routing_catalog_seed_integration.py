"""Exercise catalog guidance through the production routing entry point."""
from dataclasses import replace

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from squander import Qiskit_IO
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing


def _problem():
    circuit = qgd_Circuit(3)
    circuit.add_CNOT(2, 0)
    return circuit, np.empty(0), ((0, 1), (1, 2))


def _config(**overrides):
    config = {
        "exact_routing_global_search": False,
        "exact_routing_mapping_aware_seeds": True,
        "exact_routing_lazy_osr": False,
        "exact_routing_light_sabre_seed_count": 1,
        "exact_routing_light_guided_cover_count": 1,
        "exact_routing_minimum_cover_seed_count": 1,
        "exact_routing_cover_seed_timeout_seconds": 0.1,
        "exact_routing_post_catalog_pam_seed_count": 0,
        "exact_routing_precomputed_pam_seeds": False,
        "exact_routing_token_seed_timeout_seconds": 0.0,
        "exact_routing_layout_portfolio_timeout_seconds": 0.1,
        "exact_routing_synthesis_restarts": 1,
        "routing_synthesis_workers": 1,
        "parallel": 0,
        "random_seed": 20260926,
        "synthesis_acceptance_tolerance": 1e-10,
        "circuit_validation_tolerance": 1e-8,
        "use_osr": True,
        "use_graph_search": True,
    }
    config.update(overrides)
    return config


def _assert_unitary(circuit, parameters, result):
    original = Operator(Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)).data
    fitted = Qiskit_IO.get_Qiskit_Circuit(result.circuit, result.parameters)
    physical = Operator(fitted).data

    def permutation(mapping):
        assert sorted(mapping) == list(range(3))
        return [sum(((column >> q) & 1) << mapping[q] for q in range(3))
                for column in range(8)]

    logical = physical[np.ix_(permutation(result.solution.final_mapping),
                              permutation(result.solution.initial_mapping))]
    overlap = np.vdot(original, logical)
    logical = logical * np.conj(overlap) / abs(overlap)
    np.testing.assert_allclose(logical, original, atol=1e-10, rtol=0)
    for instruction in fitted.data:
        if len(instruction.qubits) > 1:
            assert instruction.operation.name == "cx"
            wires = [fitted.find_bit(q).index for q in instruction.qubits]
            assert abs(wires[0] - wires[1]) == 1


def test_full_router_uses_complete_canonical_catalog_guidance(monkeypatch):
    calls = []

    def guidance(catalog, **kwargs):
        calls.append((catalog, kwargs))
        return ()

    monkeypatch.setattr(routing, "mapping_aware_catalog_seeds", guidance)
    circuit, parameters, topology = _problem()
    result = routing.route_circuit_exact(circuit, parameters, topology, _config())
    assert len(calls) == 1
    catalog, kwargs = calls[0]
    assert isinstance(catalog, routing.LoadedRoutingOSRCatalog)
    assert len(catalog.partitions) == len(catalog.gate_orders)
    assert catalog.alternatives
    assert all(0 <= index < len(catalog.partitions)
               for index in catalog.alternatives)
    assert all(len(alternative.logical_qubits) <= 3
               for values in catalog.alternatives.values() for alternative in values)
    assert kwargs["recombine_covers"] is False
    assert kwargs["initial_mapping"] is None
    assert result.solution.cnot_count == 1
    assert result.timed_out is False
    _assert_unitary(circuit, parameters, result)


def test_full_router_can_explicitly_disable_catalog_guidance(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled catalog guidance was called.")

    monkeypatch.setattr(routing, "mapping_aware_catalog_seeds", forbidden)
    circuit, parameters, topology = _problem()
    result = routing.route_circuit_exact(
        circuit, parameters, topology,
        _config(exact_routing_mapping_aware_seeds=False))
    assert result.solution.cnot_count == 1
    _assert_unitary(circuit, parameters, result)


def test_catalog_guidance_preserves_incumbent_on_cost_ties(monkeypatch):
    circuit, parameters, topology = _problem()
    baseline = routing.route_circuit_exact(
        circuit, parameters, topology,
        _config(exact_routing_mapping_aware_seeds=False))
    calls = []

    def guidance(catalog, **kwargs):
        calls.append(True)
        return (replace(baseline.solution, master_backend="000-guidance-tie"),)

    monkeypatch.setattr(routing, "mapping_aware_catalog_seeds", guidance)
    result = routing.route_circuit_exact(circuit, parameters, topology, _config())
    assert calls
    assert not result.solution.master_backend.startswith("000-guidance-tie")
    assert result.solution.cnot_count == baseline.solution.cnot_count
    _assert_unitary(circuit, parameters, result)


def test_catalog_guidance_solver_limit_keeps_original_route(monkeypatch):
    def limited(*args, **kwargs):
        raise routing.ExactRoutingLimitExceeded("Controlled guidance solver limit.")

    monkeypatch.setattr(routing, "mapping_aware_catalog_seeds", limited)
    circuit, parameters, topology = _problem()
    result = routing.route_circuit_exact(circuit, parameters, topology, _config())
    assert result.solution.cnot_count == 1
    assert result.timed_out is False
    _assert_unitary(circuit, parameters, result)


def test_catalog_guidance_receives_fixed_initial_mapping(monkeypatch):
    calls = []

    def guidance(catalog, **kwargs):
        calls.append(kwargs)
        return ()

    monkeypatch.setattr(routing, "mapping_aware_catalog_seeds", guidance)
    circuit = qgd_Circuit(3)
    circuit.add_CNOT(1, 0)
    parameters = np.empty(0)
    fixed = (0, 1, 2)
    result = routing.route_circuit_exact(
        circuit, parameters, ((0, 1), (1, 2)),
        _config(exact_routing_initial_mapping=fixed))
    assert len(calls) == 1
    assert calls[0]["initial_mapping"] == fixed
    assert tuple(result.solution.initial_mapping) == fixed
    _assert_unitary(circuit, parameters, result)


@pytest.mark.parametrize("value", [1, "false", None])
def test_catalog_guidance_requires_boolean_enable_flag(value):
    circuit, parameters, topology = _problem()
    with pytest.raises(ValueError, match="exact_routing_mapping_aware_seeds must be a bool"):
        routing.route_circuit_exact(
            circuit, parameters, topology,
            _config(exact_routing_mapping_aware_seeds=value))


@pytest.mark.parametrize("requested", [None, (0, 1, 2)])
def test_reference_layout_does_not_overwrite_caller_mapping(monkeypatch, requested):
    from types import SimpleNamespace

    circuit, parameters, topology = _problem()
    catalog = routing.LoadedRoutingOSRCatalog(
        circuit=circuit, parameters=parameters, topology=topology,
        partitions=(frozenset({0}),), gate_orders=((0,),),
        alternatives={0: (SimpleNamespace(logical_qubits=(0, 2)),)},
        configuration={},
    )
    reference = (0, 2, 1)
    trajectory_mappings = []
    solve_mappings = []
    monkeypatch.setattr(routing, "_light_sabre_route",
                        lambda *args, **kwargs: (None, None, reference, None, object()))

    def trajectory(trace, mapping, gate_count):
        trajectory_mappings.append(mapping)
        return {}, {}

    def solve(**kwargs):
        solve_mappings.append(kwargs["initial_mapping"])
        return None

    monkeypatch.setattr(routing, "_mapping_trajectory_from_light_sabre", trajectory)
    monkeypatch.setattr(routing, "mapping_aware_partition_weights",
                        lambda **kwargs: {0: 1})
    monkeypatch.setattr(routing, "_minimum_partition_cover_selection",
                        lambda **kwargs: (0,))
    monkeypatch.setattr(routing, "_cover_selection_warm_start", solve)
    assert routing.mapping_aware_catalog_seeds(
        catalog, reference_seed_count=1, refinement_rounds=1,
        initial_mapping=requested, recombine_covers=False) == ()
    assert trajectory_mappings == [reference]
    assert solve_mappings == [requested, requested]
