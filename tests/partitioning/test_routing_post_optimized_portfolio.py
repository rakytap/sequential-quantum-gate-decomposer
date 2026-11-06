"""Guarded routing portfolios preserve quality and a single causal audit."""
from dataclasses import replace
import importlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from squander import Qiskit_IO
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing
from squander.decomposition._audit_composition import (
    certificate_scope, current_certificate,
)
from squander.decomposition._routing_portfolio import (
    optimize_routing_portfolio, routing_outcome_key,
)

backend = importlib.import_module("squander.decomposition.qgd_Wide_Circuit_Optimization")


def config(**overrides):
    result = {
        "strategy": "TreeSearch", "pre-opt-strategy": "TreeSearch",
        "routing-strategy": "exact-osr", "topology": ((0, 1), (1, 2)),
        "max_partition_size": 3, "partition_workers": 1, "parallel": 0,
        "use_osr": True, "use_graph_search": True, "use_float": False,
        "random_seed": 20260926, "synthesis_acceptance_tolerance": 1e-10,
        "circuit_validation_tolerance": 1e-8, "test_final_circuit": True,
        "exact_routing_post_optimized_portfolio": True,
        "exact_routing_mapping_aware_seeds": False,
        "exact_routing_global_search": False, "exact_routing_lazy_osr": False,
        "exact_routing_master": "ilp", "exact_routing_fixed_cover_backend": "ilp",
        "exact_routing_light_sabre_seed_count": 1,
        "exact_routing_minimum_cover_seed_count": 1,
        "exact_routing_cover_seed_timeout_seconds": 0.1,
        "exact_routing_post_catalog_pam_seed_count": 0,
        "exact_routing_precomputed_pam_seeds": False,
        "exact_routing_token_seed_timeout_seconds": 0.0,
        "exact_routing_layout_portfolio_timeout_seconds": 0.1,
        "routing_synthesis_workers": 1,
    }
    result.update(overrides)
    return result


def problem():
    circuit = qgd_Circuit(3)
    circuit.add_CNOT(2, 0)
    return circuit, np.empty(0)


def test_final_cost_not_raw_cost_selects_the_route():
    anchor = {"final_cnots": 13, "is_anchor": True, "index": 1, "raw_cnots": 18}
    cheaper = {"final_cnots": 15, "is_anchor": False, "index": 0, "raw_cnots": 17}
    assert min((cheaper, anchor), key=routing_outcome_key) is anchor
    better = dict(cheaper, final_cnots=12)
    assert min((anchor, better), key=routing_outcome_key) is better


def test_final_cnot_ties_keep_the_original_anchor():
    anchor = {"final_cnots": 22, "is_anchor": True, "index": 4}
    alternate = {"final_cnots": 22, "is_anchor": False, "index": 0}
    assert min((alternate, anchor), key=routing_outcome_key) is anchor


def test_certificate_scope_restores_nested_parent_on_failure():
    parent, child = SimpleNamespace(pid=os.getpid()), SimpleNamespace(pid=os.getpid())
    initial = current_certificate()
    with certificate_scope(parent):
        assert current_certificate() is parent
        with pytest.raises(RuntimeError, match="controlled"):
            with certificate_scope(child):
                assert current_certificate() is child
                raise RuntimeError("controlled")
        assert current_certificate() is parent
    assert current_certificate() is initial


@pytest.mark.parametrize("value", [1, "false", None])
def test_routing_portfolio_flag_must_be_boolean(value):
    circuit, parameters = problem()
    owner = backend.qgd_Wide_Circuit_Optimization(config(
        exact_routing_post_optimized_portfolio=value))
    with pytest.raises(ValueError, match="must be a bool"):
        owner.OptimizeWideCircuit(circuit, parameters)


def test_failed_candidate_never_changes_parent_audit_or_certificate(monkeypatch, tmp_path):
    circuit, parameters = problem()
    owner = backend.qgd_Wide_Circuit_Optimization(config())
    parent_path = tmp_path / "parent.jsonl"
    parent_path.write_text("existing parent audit\n")
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(parent_path))

    def fail(*args, **kwargs):
        raise RuntimeError("controlled routing failure")

    monkeypatch.setattr(type(owner), "route_circuit", fail)
    parent = SimpleNamespace(pid=os.getpid())
    with certificate_scope(parent):
        with pytest.raises(RuntimeError, match="controlled routing failure"):
            optimize_routing_portfolio(owner, circuit, parameters, backend)
        assert current_certificate() is parent
    assert os.environ["SQUANDER_REWRITE_AUDIT_JSONL"] == str(parent_path)
    assert parent_path.read_text() == "existing parent audit\n"


@pytest.mark.parametrize("extra_swap_pair", [False, True])
def test_full_wco_retains_distinct_routes_and_replays_only_winner(
    monkeypatch, tmp_path, extra_swap_pair,
):
    circuit, parameters = problem()
    original_router = routing.route_circuit_exact
    offered_backends = []

    def offer_tied_routes(*args, **kwargs):
        result = original_router(*args, **kwargs)
        original = result.solution
        # A different, equally costly layout is a real permutation-valid route.
        alternative = replace(original, master_backend="distinct-equal-cost-layout",
                              initial_mapping=tuple(2 - v for v in original.initial_mapping),
                              final_mapping=tuple(2 - v for v in original.final_mapping),
                              selections=tuple(
                                  replace(selection, alternative=replace(
                                      selection.alternative,
                                      input_physical=tuple(2 - v for v in selection.alternative.input_physical),
                                      output_physical=tuple(2 - v for v in selection.alternative.output_physical)))
                                  for selection in original.selections))
        if extra_swap_pair:
            # Two identical physical SWAPs compose to the identity. This is
            # a valid, strictly more expensive raw route for the same input.
            swaps = list(alternative.transition_swaps or
                         tuple(() for _ in range(len(alternative.selections) + 1)))
            swaps[0] = ((0, 1), (0, 1)) + tuple(swaps[0])
            alternative = replace(
                alternative, master_backend="distinct-higher-raw-cost-layout",
                transition_swaps=tuple(swaps),
                cnot_count=alternative.cnot_count + 6,
            )
        offered_backends.extend((original.master_backend, alternative.master_backend))
        return replace(result, constructive_candidates=(original, alternative, alternative))

    monkeypatch.setattr(routing, "route_circuit_exact", offer_tied_routes)
    parent_path = tmp_path / "parent.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(parent_path))
    owner = backend.qgd_Wide_Circuit_Optimization(config())
    fitted, fitted_parameters = owner.OptimizeWideCircuit(circuit, parameters)
    assert backend.CNOTGateCount(fitted, 0) == 1
    portfolio = owner.config["routing_portfolio"]
    assert len(portfolio["candidates"]) == 2
    assert portfolio["duplicate_routes"] >= 1
    assert portfolio["winner_only_audit"] is True
    assert portfolio["raw_shortlist_is_heuristic"] is False
    assert portfolio["screened_higher_raw_cost"] == 0
    if extra_swap_pair:
        anchor_cost = portfolio["raw_incumbent_bound"]
        assert any(candidate["raw_cnots"] > anchor_cost
                   for candidate in portfolio["candidates"])
    chosen = portfolio["candidates"][portfolio["selected_index"]]
    assert chosen["is_anchor"] is True

    input_representation = backend._squander_audit_representation(
        circuit, parameters, range(3))
    output_representation = backend._squander_audit_representation(
        fitted, fitted_parameters, range(3))
    metadata = {
        "input_circuit": input_representation, "output_circuit": output_representation,
        "input_circuit_sha256": backend._exact_state_sha256(
            backend._qasm_exact_state(input_representation)),
        "output_circuit_sha256": backend._exact_state_sha256(
            backend._qasm_exact_state(output_representation)),
        "initial_mapping": owner.config["initial_mapping"],
        "final_mapping": owner.config["final_mapping"],
    }
    audit, digest = owner.save_rewrite_audit(
        parent_path, tmp_path / "audit.json.gz", metadata)
    verification = owner.verify_rewrite_audit(
        audit, expected_sha256=digest, audit_path=tmp_path / "audit.json.gz")
    assert verification["replayed_exact_osr_routing_passes"] == 1
    events = [json.loads(line) for line in parent_path.read_text().splitlines()]
    assert sum(event.get("event_type") == "exact-osr-routing" for event in events) <= 1
    target = Operator(Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)).data
    output = Qiskit_IO.get_Qiskit_Circuit(fitted, fitted_parameters)
    physical = Operator(output).data
    def permutation(mapping):
        return [sum(((column >> q) & 1) << mapping[q] for q in range(3))
                for column in range(8)]
    logical = physical[np.ix_(permutation(owner.config["final_mapping"]),
                              permutation(owner.config["initial_mapping"]))]
    overlap = np.vdot(target, logical)
    np.testing.assert_allclose(logical * np.conj(overlap) / abs(overlap),
                               target, atol=1e-10, rtol=0)
    for instruction in output.data:
        if len(instruction.qubits) > 1:
            wires = [output.find_bit(q).index for q in instruction.qubits]
            assert instruction.operation.name == "cx" and abs(wires[0] - wires[1]) == 1
