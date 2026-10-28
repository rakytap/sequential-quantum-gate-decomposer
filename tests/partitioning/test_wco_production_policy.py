"""Production OSR policy and auditable constructive routing."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    _append_exact_osr_routing_event,
    _verify_exact_osr_routing_replay,
    qgd_Wide_Circuit_Optimization,
)
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing


POLICY = {
    "random_seed": 20260926,
    "osr_certified_terminal_completion": True,
    "osr_geometric_completion_motifs": True,
    "osr_physical_cover_motifs": True,
    "osr_geometric_completion_basin_hops": 2,
    "osr_certified_cnot_deflation": True,
    "osr_cnot_deflation_basin_hops": 1,
}


@pytest.mark.parametrize("width", (3, 4))
@pytest.mark.parametrize("use_float", (False, True))
def test_wco_uses_certified_policy_and_constructive_routing(width, use_float):
    config = qgd_Wide_Circuit_Optimization({
        "max_partition_size": width,
        "use_osr": True,
        "use_graph_search": True,
        "use_float": use_float,
    }).config
    assert {key: config[key] for key in POLICY} == POLICY
    assert config["exact_routing_global_search"] is False
    assert config["exact_routing_lazy_osr"] is False
    assert config["synthesis_acceptance_tolerance"] == 1e-10


def test_wco_respects_explicit_replay_overrides():
    overrides = {
        "random_seed": 17,
        "osr_certified_cnot_deflation": False,
        "osr_geometric_completion_basin_hops": 5,
        "exact_routing_global_search": True,
    }
    config = qgd_Wide_Circuit_Optimization(dict(overrides)).config
    assert {key: config[key] for key in overrides} == overrides


def test_production_metadata_records_the_actual_policy():
    path = Path(__file__).resolve().parents[2] / "examples/decomposition/wide_circuit_optimization.py"
    spec = importlib.util.spec_from_file_location("wco_policy_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = qgd_Wide_Circuit_Optimization({"max_partition_size": 4}).config
    snapshot = module.result_configuration(config, 5)
    assert {key: snapshot[key] for key in POLICY} == POLICY
    assert snapshot["exact_routing_global_search"] is False
    assert snapshot["partition_size_schedule"] == [4]


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("partition_size", (3, 4))
def test_constructive_route_bypasses_global_models_and_replays(
    monkeypatch, tmp_path, lazy, partition_size
):
    def forbidden_global_model(*args, **kwargs):
        pytest.fail("Constructive routing must not enter the global master/flow model")

    def fixed_cover_only(*args, **kwargs):
        # Local refinement shares the solver entry point with the global
        # master, but its candidates are one disjoint, already-selected cover.
        # The global model retains overlapping competing partitions.
        parts = tuple(frozenset(part) for part in kwargs["partitions"])
        gates = frozenset(kwargs["gate_predecessors"])
        if (
            args
            or not parts
            or frozenset().union(*parts) != gates
            or sum(len(part) for part in parts) != len(gates)
            or kwargs.get("allow_suboptimal") is not True
            or kwargs.get("warm_start") is None
        ):
            forbidden_global_model(*args, **kwargs)
        assert len(kwargs["warm_start"].selections) == len(parts)
        return kwargs["warm_start"]

    def fallback_pricing(targets, config, topology, *, target_configs=None):
        assert all(target.shape[0] <= 8 for target in targets)
        return (None,) * len(targets)

    monkeypatch.setattr(routing, "solve_exact_routing", fixed_cover_only)
    monkeypatch.setattr(routing, "_solve_exact_routing_ilp_flow_restricted", forbidden_global_model)
    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", fallback_pricing)
    audit_path = tmp_path / "constructive.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(audit_path))
    circuit = qgd_Circuit(3)
    circuit.add_U3(0)
    circuit.add_CNOT(2, 0)
    circuit.add_CNOT(1, 2)
    parameters = np.array([0.123, -0.234, 0.345])
    topology = ((0, 1), (1, 2))
    result = routing.route_circuit_exact(circuit, parameters, topology, {
        "strategy": "TreeSearch",
        "max_partition_size": partition_size,
        "exact_routing_global_search": False,
        "exact_routing_mapping_aware_seeds": True,
        "exact_routing_lazy_osr": lazy,
        "exact_routing_light_sabre_seed_count": 2,
        "exact_routing_cover_seed_timeout_seconds": 0.1,
        "exact_routing_layout_portfolio_timeout_seconds": 0.1,
        "exact_routing_minimum_cover_seed_count": 2,
        "exact_routing_post_catalog_pam_seed_count": 2,
        "exact_routing_layout_seed_count": 2,
        "exact_routing_layout_total_passes": 2,
    })
    assert result.solution.selections
    assert result.solution.master_backend.endswith("-constructive")
    assert result.solution.optimal is False
    assert result.solution.cnot_optimal is False
    assert result.solution.solver_bound is None
    assert result.timed_out is False
    _append_exact_osr_routing_event(
        "routing", 0, circuit, parameters, result.circuit, result.parameters,
        result, topology, 1e-10,
    )
    event = json.loads(audit_path.read_text().splitlines()[-1])
    _verify_exact_osr_routing_replay(event)


def test_global_search_remains_available_when_explicitly_enabled(monkeypatch):
    calls = []

    def capture_master(*, warm_start, **kwargs):
        calls.append("master")
        return replace(warm_start, optimal=True)

    def capture_flow(*, warm_start, **kwargs):
        calls.append("flow")
        return warm_start

    monkeypatch.setattr(routing, "solve_exact_routing", capture_master)
    monkeypatch.setattr(routing, "_solve_exact_routing_ilp_flow_restricted", capture_flow)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    result = routing.route_circuit_exact(circuit, np.empty((0,)), ((0, 1),), {
        "exact_routing_global_search": True,
        "exact_routing_lazy_osr": False,
        "exact_routing_flow_seed": True,
        "exact_routing_light_sabre_seed_count": 1,
    })
    # Fixed-cover seeds can call the solver too. The optional global phase
    # starts with the flow probe and then invokes the full routing master.
    assert calls[-2:] == ["flow", "master"]
    assert calls.count("flow") == 1
    assert result.solution.optimal is True


def test_global_search_rejects_ambiguous_nonboolean_settings():
    with pytest.raises(ValueError, match="exact_routing_global_search must be a bool"):
        routing.route_circuit_exact(qgd_Circuit(2), np.empty((0,)), ((0, 1),), {
            "exact_routing_global_search": "false",
        })


@pytest.mark.parametrize("overrides, expected", [
    ({}, False),
    ({"exact_routing_mapping_aware_seeds": True}, True),
    ({"exact_routing_mapping_aware_seeds": False}, False),
])
def test_metadata_records_catalog_guidance_default_and_overrides(overrides, expected):
    path = Path(__file__).resolve().parents[2] / "examples/decomposition/wide_circuit_optimization.py"
    spec = importlib.util.spec_from_file_location("wco_catalog_policy_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = qgd_Wide_Circuit_Optimization(
        {"max_partition_size": 4, **overrides}).config
    snapshot = module.result_configuration(config, 5)
    assert snapshot["exact_routing_mapping_aware_seeds"] is expected
