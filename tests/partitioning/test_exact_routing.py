import json
import itertools
import time
from dataclasses import replace

import numpy as np
import pytest

from squander import utils
import squander.partitioning.routing as routing
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    _append_exact_osr_routing_event,
    _cnot_aware_pam_routing_class,
    _verify_exact_osr_routing_replay,
    qgd_Wide_Circuit_Optimization,
)
from squander.partitioning.routing import (
    ExactRoutingResult,
    RoutingAlternative,
    RoutingSelection,
    _process_infidelity,
    cnot_schmidt_lower_bound,
    permuted_partition_target,
    route_circuit_exact,
    solve_exact_routing,
    solve_exact_routing_branch_and_bound,
    solve_exact_routing_ilp,
    synthesize_partition_alternatives,
    symmetry_reduced_assignment_orbits,
    topology_automorphisms,
)


def test_cnot_aware_pam_preserves_native_entangler_ratio_at_default_cost():
    class StubPAM:
        def __init__(self):
            self.gate_count_weight = 0.1

        def _score_perm(self, circuit, frontier, pi, distances, perm, extended):
            return 2.0

    pam_type = _cnot_aware_pam_routing_class(
        StubPAM, {"pam_swap_cnot_cost": 3.0}
    )
    pam = pam_type()

    assert pam.gate_count_weight == 0.1
    assert pam._score_perm(None, {0, 1, 2, 3}, None, None, None, None) == 2.0


def test_exact_osr_is_the_wide_router_default():
    optimizer = qgd_Wide_Circuit_Optimization({})

    assert optimizer.config["routing-strategy"] == "exact-osr"
    assert optimizer.config["routing_column_synthesis_mode"] == "topology-osr"
    assert optimizer.config["exact_routing_master"] == "benders"
    assert optimizer.config["exact_routing_lazy_osr"] is False
    assert optimizer.config["exact_routing_synthesis_restarts"] == 1
    assert optimizer.config["exact_routing_light_sabre_seed_count"] == 32
    assert optimizer.config["exact_routing_light_sabre_trials_per_seed"] == 1
    assert optimizer.config["exact_routing_light_guided_cover_count"] == 8
    assert optimizer.config["exact_routing_cover_seed_translation_limit"] == 8
    assert optimizer.config["exact_routing_flow_seed"] is True
    assert optimizer.config["exact_routing_flow_seed_timeout_seconds"] == 30.0
    assert optimizer.config["exact_routing_flow_seed_max_terms"] == 500_000
    assert optimizer.config["exact_routing_fixed_cover_backend"] == "sat"
    assert optimizer.config["exact_routing_sat_solver"] == "glucose42"
    assert (
        optimizer.config["exact_routing_sat_max_estimated_clauses"]
        == 5_000_000
    )
    assert optimizer.config["exact_routing_timeout_seconds"] == 20 * 60
    assert (
        optimizer.config["exact_routing_benders_zero_swap_probe_seconds"]
        == 5.0
    )
    assert optimizer.config["exact_routing_benders_master_slice_seconds"] == 30.0
    assert (
        optimizer.config["exact_routing_benders_subproblem_slice_seconds"]
        == 60.0
    )
    assert optimizer.config["exact_routing_benders_stagnation_seconds"] == 120.0
    assert optimizer.config["exact_routing_cover_seed_timeout_seconds"] == 10.0
    assert optimizer.config["exact_routing_cover_pool_timeout_seconds"] == 10.0
    assert optimizer.config["exact_routing_minimum_cover_seed_count"] == 8
    assert optimizer.config["exact_routing_post_catalog_pam_seed_count"] == 12
    assert optimizer.config["exact_routing_cover_seed_beam_width"] == 64
    assert optimizer.config["exact_routing_precomputed_pam_seeds"] is True
    assert optimizer.config["exact_routing_pam_layout_passes"] == 3
    assert optimizer.config["exact_routing_pam_swap_cnot_costs"] == (3.0,)
    assert optimizer.config["exact_routing_pam_cover_strategies"] == (
        "kahn", "ilp", "ilp-routing"
    )
    assert optimizer.config["exact_routing_require_tiebreaker_proof"] is False
    assert optimizer.config["routing_synthesis_workers"] is None
    assert optimizer.config["routing_synthesis_worker_memory_limit_gib"] == 16.0
    assert optimizer.config["routing_minimum_available_memory_fraction"] == 0.25
    assert (
        optimizer.config["exact_routing_fixed_cover_max_triangle_constraints"]
        == 500_000
    )
    assert "exact_routing_total_timeout_seconds" not in optimizer.config


@pytest.mark.parametrize("target,control", [(2, 0), (0, 2)])
def test_distance_two_cnot_uses_exact_four_cnot_bridge(target, control):
    source = qgd_Circuit(3)
    source.add_CNOT(target, control)

    routed, parameters = routing._route_source_circuit_on_local_topology(
        source, np.empty((0,)), ((0, 1), (1, 2)), 3
    )

    assert len(routed.get_Gates()) == 4
    assert _process_infidelity(
        source.get_Matrix(np.empty((0,)), is_f32=False),
        routed.get_Matrix(parameters, is_f32=False),
    ) < 1e-15


@pytest.mark.parametrize(
    "routing_strategy",
    (
        "exact-osr",
        "pam-osr",
        "seqpam-ilp",
        "seqpam-quick",
        "bqskit-sabre",
        "light-sabre",
        "sabre",
    ),
)
def test_every_router_still_runs_explicit_topology_optimization(
    monkeypatch, routing_strategy
):
    topology = ((0, 1), (1, 2))
    source = qgd_Circuit(3)
    source.add_CNOT(2, 0)
    parameters = np.empty((0,), dtype=np.float64)
    observed_topologies = []

    def fake_inner(self, circuit, values, **_kwargs):
        observed_topologies.append(self.config.get("topology"))
        return circuit, values

    def fake_route(self, circuit, values):
        return routing._route_source_circuit_on_local_topology(
            circuit, values, topology, 3
        )

    monkeypatch.setattr(
        qgd_Wide_Circuit_Optimization,
        "InnerOptimizeWideCircuit",
        fake_inner,
    )
    monkeypatch.setattr(
        qgd_Wide_Circuit_Optimization,
        "route_circuit",
        fake_route,
    )
    optimizer = qgd_Wide_Circuit_Optimization(
        {
            "strategy": "TreeSearch",
            "pre-opt-strategy": "TreeSearch",
            "routing-strategy": routing_strategy,
            "topology": topology,
            "test_final_circuit": False,
        }
    )

    optimizer.OptimizeWideCircuit(source, parameters)

    assert observed_topologies == [None, topology]
    assert "topology_optimization_skipped" not in optimizer.config


def test_three_qubit_topology_symmetry_counts():
    path = [(0, 1), (1, 2)]
    triangle = [(0, 1), (1, 2), (0, 2)]

    assert len(topology_automorphisms(path, 3)) == 2
    assert len(topology_automorphisms(triangle, 3)) == 6
    assert len(symmetry_reduced_assignment_orbits(path, 3)) == 18
    assert len(symmetry_reduced_assignment_orbits(triangle, 3)) == 6


def test_precomputed_osr_pam_seed_uses_resolved_catalog_without_synthesis(
    monkeypatch,
):
    local = qgd_Circuit(2)
    local.add_CNOT(1, 0)
    empty = np.empty((0,), dtype=np.float64)
    payload = routing.SynthesizedRoutingPayload(
        circuit=local,
        parameters=empty,
        topology=((0, 1),),
        input_assignment=(0, 1),
        output_assignment=(0, 1),
        source_circuit=local,
        source_parameters=empty,
    )
    alternatives = {}
    for partition, logicals in ((0, (0, 2)), (1, (1, 2))):
        alternatives[partition] = tuple(
            RoutingAlternative(
                partition,
                logicals,
                embedding,
                embedding,
                1,
                payload=payload,
            )
            for edge in ((0, 1), (1, 2))
            for embedding in (edge, edge[::-1])
        )

    def reject_synthesis(*args, **kwargs):
        raise AssertionError("A precomputed PAM seed attempted synthesis.")

    monkeypatch.setattr(routing, "_call_shared_synthesis", reject_synthesis)
    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", reject_synthesis)
    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0, 1),
        gate_predecessors={0: (), 1: (0,)},
        partitions=({0}, {1}),
        alternatives=alternatives,
        logical_qubit_count=3,
        topology=((0, 1), (1, 2)),
        path=(0, 1, 2),
    )

    assert result is not None
    assert result.cnot_count == 2
    assert {selection.partition for selection in result.selections} == {0, 1}
    assert all(
        selection.alternative in alternatives[selection.partition]
        for selection in result.selections
    )
    replay, parameters = routing.construct_exact_routed_circuit(result, 3)
    assert replay.get_Gate_Nums() == {"CNOT": 2}
    assert parameters.size == 0


def test_precomputed_pam_catalog_prices_swap_as_three_cnots():
    swap = qgd_Circuit(2)
    swap.add_SWAP((0, 1))

    converted = routing._squander_to_bqskit(
        swap,
        np.empty((0,), dtype=np.float64),
        compensate_endianness=True,
    )

    assert sum(
        count
        for gate, count in converted.gate_counts.items()
        if gate.num_qudits >= 2
    ) == 3


def test_precomputed_pam_keys_sparse_circuit_by_declared_full_topology():
    local = qgd_Circuit(3)
    local.add_CNOT(1, 0)
    empty = np.empty((0,), dtype=np.float64)
    payload = routing.SynthesizedRoutingPayload(
        circuit=local,
        parameters=empty,
        topology=((0, 1), (1, 2)),
        input_assignment=(0, 1, 2),
        output_assignment=(0, 1, 2),
        source_circuit=local,
        source_parameters=empty,
    )
    alternative = RoutingAlternative(
        0, (0, 1, 2), (0, 1, 2), (0, 1, 2), 1, payload=payload
    )

    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0,),
        gate_predecessors={0: ()},
        partitions=({0},),
        alternatives={0: (alternative,)},
        logical_qubit_count=3,
        topology=((0, 1), (1, 2)),
        path=(0, 1, 2),
    )

    assert result is not None
    assert result.cnot_count == 1


def test_precomputed_pam_replays_full_mapping_when_swaps_displace_other_tokens():
    local = qgd_Circuit(2)
    local.add_CNOT(1, 0)
    empty = np.empty((0,), dtype=np.float64)
    payload = routing.SynthesizedRoutingPayload(
        circuit=local,
        parameters=empty,
        topology=((0, 1),),
        input_assignment=(0, 1),
        output_assignment=(0, 1),
        source_circuit=local,
        source_parameters=empty,
    )
    logical_pairs = ((0, 1), (0, 2), (0, 3))
    alternatives = {
        partition: tuple(
            RoutingAlternative(
                partition, logicals, embedding, embedding, 1, payload=payload
            )
            for edge in ((0, 1), (1, 2), (2, 3))
            for embedding in (edge, edge[::-1])
        )
        for partition, logicals in enumerate(logical_pairs)
    }

    result = routing._precomputed_osr_pam_warm_start(
        selected_partitions=(0, 1, 2),
        gate_predecessors={0: (), 1: (0,), 2: (1,)},
        partitions=({0}, {1}, {2}),
        alternatives=alternatives,
        logical_qubit_count=4,
        topology=((0, 1), (1, 2), (2, 3)),
        path=(0, 1, 2, 3),
    )

    assert result is not None
    assert sum(map(len, result.transition_swaps)) >= 1
    assert sorted(result.initial_mapping) == list(range(4))
    assert sorted(result.final_mapping) == list(range(4))
    replay, _parameters = routing.construct_exact_routed_circuit(result, 4)
    routing._assert_local_topology(replay, ((0, 1), (1, 2), (2, 3)))


def test_exact_audit_remap_canonicalizes_embedded_swap_operands():
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        _remap_exact_state,
    )

    remapped = _remap_exact_state(
        {
            "qubits": 2,
            "operations": [
                {"name": "swap", "qubits": [0, 1], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []},
            ],
        },
        (5, 4),
        7,
    )

    assert remapped["operations"][0]["qubits"] == [4, 5]
    # Directed gates must retain their operand order.
    assert remapped["operations"][1]["qubits"] == [5, 4]


def test_exact_solver_propagates_partition_output_mapping():
    alternatives = {
        0: [
            RoutingAlternative(0, (0, 1), (0, 1), (1, 0), 2),
            RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 4),
        ],
        1: [RoutingAlternative(1, (0, 2), (1, 2), (1, 2), 3)],
    }
    result = solve_exact_routing(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}],
        alternatives=alternatives,
        logical_qubit_count=3,
    )

    assert result.cnot_count == 5
    assert result.initial_mapping == (0, 1, 2)
    assert result.final_mapping == (1, 0, 2)
    assert result.master_backend.startswith("pulp-")

    branch_and_bound = solve_exact_routing_branch_and_bound(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}],
        alternatives=alternatives,
        logical_qubit_count=3,
    )
    assert (
        result.cnot_count,
        result.single_qubit_count,
        result.initial_mapping,
        result.final_mapping,
    ) == (
        branch_and_bound.cnot_count,
        branch_and_bound.single_qubit_count,
        branch_and_bound.initial_mapping,
        branch_and_bound.final_mapping,
    )


def test_benders_master_enforces_positive_swap_cost_when_gurobi_available():
    try:
        from squander.partitioning.ilp import _check_gurobi_available

        _check_gurobi_available()
    except Exception as exc:
        pytest.skip(f"Gurobi is unavailable: {exc}")

    result = solve_exact_routing(
        backend="benders",
        gate_predecessors={0: ()},
        partitions=[{0}],
        alternatives={
            0: [RoutingAlternative(0, (0, 2), (0, 1), (0, 1), 2)]
        },
        logical_qubit_count=3,
        topology=[(0, 1), (1, 2)],
        initial_mapping=(0, 1, 2),
        timeout_seconds=20,
        allow_suboptimal=False,
    )

    assert result.optimal is True
    assert result.cnot_count == 5
    assert result.solver_bound == 5
    assert result.transition_swaps == (((1, 2),),)

    warm_alternative = RoutingAlternative(
        0, (0, 2), (0, 1), (0, 1), 2
    )
    warm_start = ExactRoutingResult(
        selections=(RoutingSelection(0, warm_alternative),),
        cnot_count=5,
        single_qubit_count=0,
        initial_mapping=(0, 1, 2),
        final_mapping=(0, 2, 1),
        explored_states=0,
        master_backend="test-warm-start",
        optimal=False,
        transition_swaps=(((1, 2),),),
    )
    cnot_proof = solve_exact_routing(
        backend="benders",
        gate_predecessors={0: ()},
        partitions=[{0}],
        alternatives={0: [warm_alternative]},
        logical_qubit_count=3,
        topology=[(0, 1), (1, 2)],
        initial_mapping=(0, 1, 2),
        timeout_seconds=20,
        allow_suboptimal=True,
        warm_start=warm_start,
        stop_when_cnot_optimal=True,
    )
    assert cnot_proof.cnot_optimal is True
    assert cnot_proof.cnot_count == 5


def test_benders_zero_swap_probe_timeout_preserves_staged_incumbent(monkeypatch):
    try:
        from squander.partitioning.ilp import _check_gurobi_available

        _check_gurobi_available()
    except Exception as exc:
        pytest.skip(f"Gurobi is unavailable: {exc}")

    probe_timeouts = []

    def timeout_zero_swap_probe(*args, timeout_seconds=None, **kwargs):
        probe_timeouts.append(timeout_seconds)
        raise routing.ExactRoutingLimitExceeded("test probe timeout")

    def timeout_staged_oracle(*args, **kwargs):
        raise routing.ExactRoutingLimitExceeded("test staged timeout")

    monkeypatch.setattr(
        routing,
        "_solve_exact_routing_ilp_flow_restricted",
        timeout_zero_swap_probe,
    )
    monkeypatch.setattr(routing, "solve_exact_routing_ilp", timeout_staged_oracle)

    alternative = RoutingAlternative(0, (0, 2), (0, 1), (0, 1), 2)
    warm_start = ExactRoutingResult(
        selections=(RoutingSelection(0, alternative),),
        cnot_count=8,
        single_qubit_count=0,
        initial_mapping=(0, 1, 2),
        final_mapping=(2, 1, 0),
        explored_states=0,
        master_backend="test-warm-start",
        optimal=False,
        transition_swaps=(((1, 2), (0, 1)),),
    )
    result = routing.solve_exact_routing_benders(
        gate_predecessors={0: ()},
        partitions=[{0}],
        alternatives={0: [alternative]},
        logical_qubit_count=3,
        topology=[(0, 1), (1, 2)],
        initial_mapping=(0, 1, 2),
        timeout_seconds=20,
        allow_suboptimal=True,
        warm_start=warm_start,
        zero_swap_probe_seconds=0.01,
    )

    assert probe_timeouts and max(probe_timeouts) <= 0.01
    assert result.cnot_count == 5
    assert result.cnot_count < warm_start.cnot_count


def test_benders_oracle_receives_every_permutation_for_selected_partition(
    monkeypatch,
):
    try:
        from squander.partitioning.ilp import _check_gurobi_available

        _check_gurobi_available()
    except Exception as exc:
        pytest.skip(f"Gurobi is unavailable: {exc}")

    cheap = RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 0)
    compatible = RoutingAlternative(0, (0, 1), (1, 2), (1, 2), 1)
    warm_start = ExactRoutingResult(
        selections=(RoutingSelection(0, cheap),),
        cnot_count=3,
        single_qubit_count=0,
        initial_mapping=(0, 1, 2),
        final_mapping=(0, 1, 2),
        explored_states=0,
        master_backend="test-warm-start",
        optimal=False,
        transition_swaps=(((0, 1),),),
    )
    observed = []

    def timeout_zero_swap_probe(*args, **kwargs):
        raise routing.ExactRoutingLimitExceeded("force staged oracle")

    def inspect_staged_oracle(*, alternatives, **kwargs):
        observed.append(tuple(alternatives[0]))
        chosen = alternatives[0][1]
        return ExactRoutingResult(
            selections=(RoutingSelection(0, chosen),),
            cnot_count=1,
            single_qubit_count=0,
            initial_mapping=(0, 1, 2),
            final_mapping=(0, 1, 2),
            explored_states=0,
            master_backend="test-oracle",
            optimal=True,
            transition_swaps=((),),
        )

    monkeypatch.setattr(
        routing,
        "_solve_exact_routing_ilp_flow_restricted",
        timeout_zero_swap_probe,
    )
    monkeypatch.setattr(routing, "solve_exact_routing_ilp", inspect_staged_oracle)

    result = routing.solve_exact_routing_benders(
        gate_predecessors={0: ()},
        partitions=[{0}],
        alternatives={0: (cheap, compatible)},
        logical_qubit_count=3,
        topology=[(0, 1), (1, 2)],
        timeout_seconds=20,
        allow_suboptimal=True,
        warm_start=warm_start,
    )

    assert observed == [(cheap, compatible)]
    assert result.cnot_count == 1


def test_local_cover_seed_prefers_synthesized_multi_gate_column():
    try:
        from squander.partitioning.ilp import _check_gurobi_available

        _check_gurobi_available()
    except Exception as exc:
        pytest.skip(f"Gurobi is unavailable: {exc}")

    alternatives = {
        0: [RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 2)],
        1: [RoutingAlternative(1, (1, 2), (1, 2), (1, 2), 2)],
        2: [RoutingAlternative(2, (0, 1, 2), (0, 1, 2), (0, 1, 2), 3)],
    }
    result = routing._local_cost_cover_warm_start(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}, {0, 1}],
        alternatives=alternatives,
        logical_qubit_count=3,
        path=(0, 1, 2),
        timeout_seconds=5,
    )

    assert result is not None
    assert result.cnot_count == 3
    assert [selection.partition for selection in result.selections] == [2]
    assert result.master_backend == "local-cover-greedy-mip-start"


def test_minimum_partition_seed_uses_existing_exact_cover_ilp():
    try:
        from squander.partitioning.ilp import _check_gurobi_available

        _check_gurobi_available()
    except Exception as exc:
        pytest.skip(f"Gurobi is unavailable: {exc}")

    alternatives = {
        0: [RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 2)],
        1: [RoutingAlternative(1, (1, 2), (1, 2), (1, 2), 2)],
        2: [RoutingAlternative(2, (0, 1, 2), (0, 1, 2), (0, 1, 2), 5)],
    }
    result = routing._minimum_partition_cover_warm_start(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}, {0, 1}],
        alternatives=alternatives,
        logical_qubit_count=3,
        path=(0, 1, 2),
        exact_refine_timeout_seconds=5,
    )

    assert result.cnot_count == 5
    assert [selection.partition for selection in result.selections] == [2]
    assert result.master_backend == "minimum-partition-fixed-cover-mip-start"


def test_greedy_seed_uses_output_permutation_to_avoid_downstream_swap():
    alternatives = {
        0: [
            RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 0),
            RoutingAlternative(0, (0, 1), (0, 1), (1, 0), 1),
        ],
        1: [RoutingAlternative(1, (0, 2), (0, 1), (0, 1), 1)],
    }
    result = routing._greedy_selected_path_warm_start(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[frozenset({0}), frozenset({1})],
        alternatives=alternatives,
        logical_qubit_count=3,
        path=(0, 1, 2),
        initial_mapping=(0, 1, 2),
    )

    assert result.cnot_count == 2
    assert result.transition_swaps == ((), ())
    assert result.selections[0].alternative.output_physical == (1, 0)


def test_partition_aware_layout_recovers_multiply_style_initial_placement():
    blocks = (
        (0, 2, 5),
        (1, 2, 6),
        (0, 3, 7),
        (6, 7, 11),
        (1, 3, 8),
        (0, 4, 9),
        (1, 4, 10),
        (8, 9, 12),
    )
    last_on_qubit = {}
    predecessors = {}
    alternatives = {}
    for partition, logicals in enumerate(blocks):
        predecessors[partition] = {
            last_on_qubit[logical]
            for logical in logicals
            if logical in last_on_qubit
        }
        for logical in logicals:
            last_on_qubit[logical] = partition
        values = []
        for start in range(11):
            embedding = (start, start + 1, start + 2)
            for input_physical in itertools.permutations(embedding):
                values.append(
                    RoutingAlternative(
                        partition,
                        logicals,
                        input_physical,
                        input_physical,
                        8,
                    )
                )
                values.append(
                    RoutingAlternative(
                        partition,
                        logicals,
                        input_physical,
                        (
                            input_physical[0],
                            input_physical[2],
                            input_physical[1],
                        ),
                        7,
                    )
                )
        alternatives[partition] = tuple(values)

    layout = routing._partition_aware_layout_seed(
        gate_predecessors=predecessors,
        partitions=[{partition} for partition in range(len(blocks))],
        alternatives=alternatives,
        logical_qubit_count=13,
        topology=tuple((qubit, qubit + 1) for qubit in range(12)),
    )

    assert layout == (2, 4, 1, 7, 10, 0, 3, 6, 8, 11, 9, 5, 12)


def test_mapping_beam_gives_each_initial_layout_its_full_quota():
    alternatives = {
        0: (
            RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 1),
        ),
        1: (
            RoutingAlternative(1, (1, 2), (1, 2), (1, 2), 1),
        ),
    }
    arguments = {
        "gate_predecessors": {0: (), 1: (0,)},
        "partitions": ({0}, {1}),
        "alternatives": alternatives,
        "logical_qubit_count": 3,
        "path": (0, 1, 2),
        "beam_width": 1,
    }
    layouts = ((0, 1, 2), (2, 1, 0))
    combined = routing._beam_selected_path_warm_start(
        **arguments, initial_mapping_candidates=layouts
    )
    separate = min(
        (
            routing._beam_selected_path_warm_start(
                **arguments, initial_mapping_candidates=(layout,)
            )
            for layout in layouts
        ),
        key=lambda result: (
            result.cnot_count,
            result.single_qubit_count,
            result.initial_mapping,
        ),
    )

    assert combined == separate


def test_fixed_cover_seed_obeys_an_expired_wall_clock_deadline():
    alternatives = {
        0: (
            RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 1),
        ),
    }
    result = routing._cover_selection_warm_start(
        selected_partitions=(0,),
        gate_predecessors={0: ()},
        partitions=({0},),
        alternatives=alternatives,
        logical_qubit_count=2,
        path=(0, 1),
        initial_mapping=None,
        master_backend="deadline-test",
        beam_width=1,
        deadline=time.monotonic() - 1.0,
    )

    assert result is None


def test_path_mapping_swap_sequence_reaches_target():
    source = (4, 1, 3, 0, 2)
    target = (2, 4, 0, 3, 1)
    mapping = list(source)
    for left, right in routing._path_mapping_swap_sequence(
        source, target, tuple(range(5))
    ):
        left_logical = mapping.index(left)
        right_logical = mapping.index(right)
        mapping[left_logical], mapping[right_logical] = (
            mapping[right_logical],
            mapping[left_logical],
        )

    assert tuple(mapping) == target


def test_ilp_encodes_and_propagates_a_fixed_initial_permutation():
    alternatives = {
        0: [RoutingAlternative(0, (0, 1), (1, 0), (0, 1), 1)],
        1: [RoutingAlternative(1, (0, 2), (0, 1), (0, 1), 1)],
    }
    result = solve_exact_routing_ilp(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}],
        alternatives=alternatives,
        logical_qubit_count=3,
        initial_mapping=(1, 0, 2),
    )

    assert result.initial_mapping == (1, 0, 2)
    assert result.cnot_count == 5
    assert len(result.transition_swaps) == 2
    assert len(result.transition_swaps[1]) == 1
    assert [selection.partition for selection in result.selections] == [0, 1]


@pytest.mark.parametrize(
    "initial_mapping",
    ((0, 1, 2), (2, 0, 1), None),
)
def test_fixed_cover_sat_matches_exact_ilp_oracle(initial_mapping):
    """The SAT replacement must preserve the former oracle's exact optimum."""
    alternatives = {
        0: [RoutingAlternative(0, (0, 1), (0, 1), (1, 0), 2)],
        1: [RoutingAlternative(1, (1, 2), (0, 1), (0, 1), 1)],
        2: [RoutingAlternative(2, (0, 2), (1, 0), (0, 1), 2)],
    }
    arguments = {
        "gate_predecessors": {0: (), 1: (0,), 2: (0,)},
        "partitions": [{0}, {1}, {2}],
        "alternatives": alternatives,
        "logical_qubit_count": 3,
        "topology": [(0, 1), (1, 2)],
        "initial_mapping": initial_mapping,
    }

    sat = solve_exact_routing_ilp(**arguments, fixed_cover_backend="sat")
    ilp = solve_exact_routing_ilp(**arguments, fixed_cover_backend="ilp")

    assert sat.cnot_count == ilp.cnot_count
    assert sat.single_qubit_count == ilp.single_qubit_count
    assert sat.optimal and ilp.optimal
    assert sat.master_backend.startswith("pysat-glucose42-")
    assert sum(map(len, sat.transition_swaps)) == (sat.cnot_count - 5) // 3
    assert sorted(selection.partition for selection in sat.selections) == [0, 1, 2]


def test_fixed_cover_sat_matches_ilp_for_every_three_block_boundary_profile():
    logical_pairs = ((0, 1), (1, 2), (0, 2))
    local_orders = ((0, 1), (1, 0))
    for transitions in itertools.product(
        itertools.product(local_orders, local_orders), repeat=3
    ):
        alternatives = {
            partition: [
                RoutingAlternative(
                    partition,
                    logical_pairs[partition],
                    input_order,
                    output_order,
                    partition + 1,
                )
            ]
            for partition, (input_order, output_order) in enumerate(transitions)
        }
        arguments = {
            "gate_predecessors": {0: (), 1: (0,), 2: (0,)},
            "partitions": [{0}, {1}, {2}],
            "alternatives": alternatives,
            "logical_qubit_count": 3,
            "topology": [(0, 1), (1, 2)],
            "initial_mapping": (0, 1, 2),
        }
        sat = solve_exact_routing_ilp(
            **arguments, fixed_cover_backend="sat"
        )
        ilp = solve_exact_routing_ilp(
            **arguments, fixed_cover_backend="ilp"
        )

        assert sat.cnot_count == ilp.cnot_count


def test_global_path_ilp_materializes_and_audits_interpartition_swap(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "global-path-routing.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(audit_path))
    local = qgd_Circuit(2)
    local.add_CNOT(1, 0)
    payload = routing.SynthesizedRoutingPayload(
        circuit=local,
        parameters=np.empty((0,), dtype=np.float64),
        topology=((0, 1),),
        input_assignment=(0, 1),
        output_assignment=(0, 1),
        source_circuit=local,
        source_parameters=np.empty((0,), dtype=np.float64),
    )
    alternatives = {
        0: [RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 1, payload=payload)],
        1: [RoutingAlternative(1, (0, 2), (1, 2), (1, 2), 1, payload=payload)],
    }

    result = solve_exact_routing_ilp(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}],
        alternatives=alternatives,
        logical_qubit_count=3,
        topology=[(0, 1), (1, 2)],
        initial_mapping=(0, 1, 2),
    )
    circuit, parameters = routing.construct_exact_routed_circuit(result, 3)

    assert result.cnot_count == 5
    assert result.transition_swaps[0] == ()
    assert len(result.transition_swaps[1]) == 1
    assert frozenset(result.transition_swaps[1][0]) in {
        frozenset((0, 1)),
        frozenset((1, 2)),
    }
    assert circuit.get_Gate_Nums() == {"CNOT": 2, "SWAP": 1}
    assert parameters.size == 0

    source = qgd_Circuit(3)
    source.add_CNOT(1, 0)
    source.add_CNOT(2, 0)
    exact_route = routing.ExactCircuitRoutingResult(
        circuit=circuit,
        parameters=parameters,
        solution=result,
        candidate_gate_sets=((0,), (1,)),
        candidate_gate_orders=((0,), (1,)),
    )
    topology = [(0, 1), (1, 2)]
    _append_exact_osr_routing_event(
        "routing",
        0,
        source,
        np.empty((0,), dtype=np.float64),
        circuit,
        parameters,
        exact_route,
        topology,
        1e-10,
    )
    event = json.loads(audit_path.read_text().splitlines()[-1])

    assert event["selections"][1]["swaps_before"]
    # The router may record an undirected edge opposite to the operand order
    # canonicalized by QASM. SWAP replay must not treat that orientation as
    # semantically or byte-stream significant.
    event["selections"][1]["swaps_before"][0].reverse()
    _verify_exact_osr_routing_replay(event)


def test_global_path_ilp_can_swap_after_a_fixed_layout_before_first_block():
    result = solve_exact_routing_ilp(
        gate_predecessors={0: ()},
        partitions=[{0}],
        alternatives={
            0: [RoutingAlternative(0, (0, 2), (0, 1), (0, 1), 1)]
        },
        logical_qubit_count=3,
        topology=[(0, 1), (1, 2)],
        initial_mapping=(0, 1, 2),
    )

    assert result.initial_mapping == (0, 1, 2)
    assert result.transition_swaps == (((1, 2),),)
    assert result.cnot_count == 4
    assert result.final_mapping == (0, 2, 1)


def test_line_topology_breaks_the_initial_mapping_reflection_symmetry():
    alternatives = {
        0: [
            RoutingAlternative(0, (0, 1, 2, 3), mapping, mapping, 1)
            for mapping in ((0, 1, 2, 3), (3, 2, 1, 0))
        ]
    }
    result = solve_exact_routing_ilp(
        gate_predecessors={0: ()},
        gate_qubits={0: (0, 1, 2, 3)},
        partitions=[{0}],
        alternatives=alternatives,
        logical_qubit_count=4,
        topology=[(0, 1), (1, 2), (2, 3)],
    )

    assert result.initial_mapping == (0, 1, 2, 3)


def test_ilp_submits_a_feasible_sabre_incumbent_as_a_mip_start(monkeypatch):
    import pulp
    from squander.partitioning import ilp as partitioning_ilp

    captured = {}

    def solve_with_cbc(prob, _pulp, callback=None, **kwargs):
        captured["warm_start"] = kwargs.get("warmStart")
        captured["values"] = {
            variable.name: variable.varValue for variable in prob.variables()
        }
        captured["unset"] = [
            variable.name
            for variable in prob.variables()
            if variable.varValue is None
        ]
        captured["violated"] = [
            name
            for name, constraint in prob.constraints.items()
            if not constraint.valid(1e-7)
        ]
        prob.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))
        return "cbc"

    monkeypatch.setattr(
        partitioning_ilp, "_solve_pulp_with_gurobi_or_cbc", solve_with_cbc
    )
    local = RoutingAlternative(0, (0, 1), (0, 1), (0, 1), 1)
    sabre = RoutingAlternative(1, (0, 1), (0, 1), (0, 1), 3)
    warm_start = ExactRoutingResult(
        selections=(RoutingSelection(1, sabre),),
        cnot_count=3,
        single_qubit_count=0,
        initial_mapping=(0, 1),
        final_mapping=(0, 1),
        explored_states=0,
        master_backend="sabre-mip-start",
        optimal=False,
    )

    result = solve_exact_routing_ilp(
        gate_predecessors={0: ()},
        gate_qubits={0: (0, 1)},
        partitions=[{0}, {0}],
        alternatives={0: [local], 1: [sabre]},
        logical_qubit_count=2,
        warm_start=warm_start,
    )

    assert captured["warm_start"] is True
    assert captured["unset"] == []
    assert captured["violated"] == []
    assert captured["values"]["selected_0"] == 0
    assert captured["values"]["selected_1"] == 1
    assert captured["values"]["config_1_0"] == 1
    assert captured["values"]["stage_1_0"] == 1
    assert captured["values"]["active_0"] == 1
    assert captured["values"]["map_in_0_0_0"] == 1
    assert captured["values"]["map_in_0_1_1"] == 1
    assert captured["values"]["map_out_0_0_0"] == 1
    assert captured["values"]["map_out_0_1_1"] == 1
    # The seed supplies the bound; the master remains free to improve it.
    assert result.cnot_count == 1


def test_ilp_rejects_a_cyclic_partition_quotient():
    partitions = [{0}, {1}, {2}, {3}, {0, 3}, {1, 2}]
    alternatives = {
        partition: [
            RoutingAlternative(
                partition,
                (0, 1),
                (0, 1),
                (0, 1),
                0 if partition >= 4 else 1,
            )
        ]
        for partition in range(len(partitions))
    }
    arguments = {
        "gate_predecessors": {0: (), 1: (0,), 2: (1,), 3: (2,)},
        "partitions": partitions,
        "alternatives": alternatives,
        "logical_qubit_count": 2,
    }

    ilp = solve_exact_routing_ilp(**arguments)
    branch_and_bound = solve_exact_routing_branch_and_bound(**arguments)

    assert ilp.cnot_count == branch_and_bound.cnot_count == 2
    assert {selection.partition for selection in ilp.selections} != {4, 5}


def test_cbc_global_stages_reject_a_convex_cyclic_cover(monkeypatch):
    import pulp
    from squander.partitioning import ilp as partitioning_ilp

    def solve_with_cbc(prob, _pulp, callback=None, **_kwargs):
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return "cbc"

    monkeypatch.setattr(
        partitioning_ilp, "_solve_pulp_with_gurobi_or_cbc", solve_with_cbc
    )
    partitions = [{0}, {1}, {2}, {3}, {0, 2}, {1, 3}]
    partition_qubits = [(0,), (1,), (1,), (0,), (0, 1), (0, 1)]
    alternatives = {
        partition: [
            RoutingAlternative(
                partition,
                partition_qubits[partition],
                partition_qubits[partition],
                partition_qubits[partition],
                0 if partition >= 4 else 1,
            )
        ]
        for partition in range(len(partitions))
    }
    result = solve_exact_routing_ilp(
        gate_predecessors={0: (), 1: (), 2: (1,), 3: (0,)},
        gate_qubits={0: (0,), 1: (1,), 2: (1,), 3: (0,)},
        partitions=partitions,
        alternatives=alternatives,
        logical_qubit_count=2,
    )

    assert {selection.partition for selection in result.selections} != {4, 5}


def test_schmidt_bound_prices_two_qubit_boundary_permutations():
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    unitary = circuit.get_Matrix(np.empty((0,)))

    assert cnot_schmidt_lower_bound(unitary, [(0, 1)]) == 1
    assert cnot_schmidt_lower_bound(
        permuted_partition_target(unitary, (0, 1), (1, 0)),
        [(0, 1)],
    ) == 2


def test_two_qubit_nonidentity_transition_is_synthesized_not_swap_wrapped():
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    transition = ((0, 1), (1, 0))

    alternatives = synthesize_partition_alternatives(
        partition=0,
        unitary=circuit.get_Matrix(np.empty((0,)), is_f32=False),
        logical_qubits=(0, 1),
        topology=[(0, 1)],
        physical_qubit_count=2,
        config={
            "strategy": "TreeSearch",
            "use_osr": True,
            "use_graph_search": True,
            "use_float": True,
            "tolerance": 1e-14,
            "osr_optimization_tolerance": 1e-6,
            "synthesis_acceptance_tolerance": 1e-10,
            "parallel": 0,
            "verbosity": 0,
            "exact_routing_eager_synthesis": True,
        },
        source_circuit=circuit,
        source_parameters=np.empty((0,)),
        requested_transitions={transition},
    )
    synthesized = next(
        alternative
        for alternative in alternatives
        if (
            alternative.payload.input_assignment,
            alternative.payload.output_assignment,
        )
        == transition
    )

    # A raw CNOT surrounded by a SWAP needs four CNOTs. OSR absorbs the
    # boundary permutation and finds the two-CNOT implementation instead.
    assert synthesized.cnot_count == 2


def test_lazy_osr_is_default_and_matches_explicit_exhaustive_pricing():
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    common = {
        "max_partition_size": 2,
        "strategy": "TreeSearch",
        "use_osr": True,
        "use_graph_search": True,
        "use_float": True,
        "tolerance": 1e-14,
        "osr_optimization_tolerance": 1e-6,
        "synthesis_acceptance_tolerance": 1e-10,
        "parallel": 0,
        "verbosity": 0,
        "exact_routing_master": "ilp",
    }

    exhaustive = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {**common, "exact_routing_lazy_osr": False},
    )
    lazy = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        common,
    )

    assert exhaustive.lazy_rounds == 0
    assert exhaustive.synthesized_partitions == 1
    assert lazy.synthesized_partitions == 0
    assert exhaustive.solution.cnot_count == lazy.solution.cnot_count == 1


def test_exhaustive_osr_reuses_identical_partition_synthesis():
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    circuit.add_CNOT(1, 0)
    result = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {
            "max_partition_size": 2,
            "strategy": "TreeSearch",
            "use_osr": True,
            "use_graph_search": True,
            "use_float": True,
            "tolerance": 1e-14,
            "osr_optimization_tolerance": 1e-6,
            "synthesis_acceptance_tolerance": 1e-10,
            "parallel": 0,
            "verbosity": 0,
            "exact_routing_master": "ilp",
            "exact_routing_lazy_osr": False,
        },
    )

    assert result.synthesized_partitions == 3
    assert result.synthesis_cache_hits >= 1
    assert result.solution.cnot_count == 0


def test_synthesis_cache_deduplicates_and_remembers_failures(monkeypatch):
    calls = []

    def fail_once(targets, config, topology, *, target_configs=None):
        calls.append(len(targets))
        return (None,) * len(targets)

    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", fail_once)
    target = np.eye(4, dtype=np.complex128)
    cache = {}
    stats = [0]
    first = routing._call_shared_synthesis_batch_cached(
        (target, target),
        {"strategy": "TreeSearch", "tree_level_max": 2},
        [(0, 1)],
        cache=cache,
        cache_stats=stats,
    )
    second = routing._call_shared_synthesis_batch_cached(
        (target,),
        {"strategy": "TreeSearch", "tree_level_max": 2},
        [(0, 1)],
        cache=cache,
        cache_stats=stats,
    )

    assert calls == [1]
    assert first == (None, None)
    assert second == (None,)
    assert stats[0] == 2


def test_exact_routing_rejects_repeated_synthesis_attempts():
    with pytest.raises(ValueError, match="exactly once"):
        routing._call_shared_synthesis_batch_cached(
            (np.eye(4, dtype=np.complex128),),
            {
                "strategy": "TreeSearch",
                "tree_level_max": 3,
                "exact_routing_synthesis_restarts": 2,
            },
            [(0, 1)],
        )


def test_global_synthesis_batch_combines_partitions_and_deduplicates(monkeypatch):
    calls = []

    def synthesize(targets, config, topology, *, target_configs=None):
        calls.append((len(targets), tuple(topology)))
        return tuple(f"result-{index}" for index in range(len(targets)))

    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", synthesize)
    config = {"strategy": "TreeSearch"}
    batch = routing._GlobalRoutingSynthesisBatch(config, {}, [0])
    first_results = []
    second_results = []
    first = np.eye(2, dtype=np.complex128)
    shared = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    last = np.diag([1, -1]).astype(np.complex128)
    batch.enqueue(
        [(0, 1)],
        (first, shared),
        (config, config),
        first_results.extend,
    )
    batch.enqueue(
        [(0, 1)],
        (shared, last),
        (config, config),
        second_results.extend,
    )

    batch.run()

    assert calls == [(3, ((0, 1),))]
    assert first_results[1] == second_results[0]


def test_failed_orbit_representatives_are_not_retried(monkeypatch):
    calls = []
    seeds = []

    def reject_all(targets, config, topology, *, target_configs=None):
        calls.append(len(targets))
        seeds.extend(value["random_seed"] for value in target_configs)
        return (None,) * len(targets)

    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", reject_all)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    alternatives = synthesize_partition_alternatives(
        partition=0,
        unitary=circuit.get_Matrix(np.empty((0,)), is_f32=False),
        logical_qubits=(0, 1),
        topology=[(0, 1)],
        physical_qubit_count=2,
        config={
            "strategy": "TreeSearch",
            "max_partition_size": 2,
            "exact_routing_eager_synthesis": True,
            "synthesis_acceptance_tolerance": 1e-10,
        },
        source_circuit=circuit,
        source_parameters=np.empty((0,)),
    )

    # The identity-placement CNOT fallback already saturates its one-CNOT
    # Schmidt lower bound. Only the nontrivial boundary transition needs OSR.
    assert calls == [1]
    assert len(seeds) == len(set(seeds)) == 1
    assert len(alternatives) == 4


def test_routing_search_depth_is_one_below_each_naive_fallback(monkeypatch):
    captured_depths = []

    def reject_all(targets, config, topology, *, target_configs=None):
        captured_depths.extend(
            target_config["tree_level_max"] for target_config in target_configs
        )
        return (None,) * len(targets)

    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", reject_all)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    synthesize_partition_alternatives(
        partition=0,
        unitary=circuit.get_Matrix(np.empty((0,)), is_f32=False),
        logical_qubits=(0, 1),
        topology=[(0, 1)],
        physical_qubit_count=2,
        config={
            "strategy": "TreeSearch",
            "max_partition_size": 2,
            "exact_routing_eager_synthesis": True,
            "synthesis_acceptance_tolerance": 1e-10,
            # A generic cap must not make exact routing miss a strict
            # improvement over its known topology-valid fallback.
            "tree_level_max": 0,
        },
        source_circuit=circuit,
        source_parameters=np.empty((0,)),
    )

    # The one-CNOT representative saturates its rigorous lower bound and is
    # skipped. The four-CNOT transition is searched only through depth three.
    assert captured_depths == [3]


def test_two_qubit_passthrough_route_replays_exactly():
    circuit = qgd_Circuit(3)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(1)
    circuit.add_CNOT(2, 1)
    parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    result = route_circuit_exact(
        circuit,
        parameters,
        [(0, 1), (1, 2)],
        {
            "max_partition_size": 2,
            "exact_routing_timeout_seconds": 5,
        },
    )

    assert result.solution.initial_mapping == (0, 1, 2)
    assert result.solution.final_mapping == (0, 1, 2)
    assert _process_infidelity(
        circuit.get_Matrix(parameters),
        result.circuit.get_Matrix(result.parameters),
    ) < 1e-14
    assert not any(
        isinstance(gate, qgd_Circuit) for gate in result.circuit.get_Gates()
    )


def test_reversed_two_qubit_placement_is_a_free_relabeling():
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    alternatives = routing.two_qubit_passthrough_alternatives(
        partition=0,
        circuit=circuit,
        parameters=np.empty((0,)),
        logical_qubits=(0, 1),
        topology=[(0, 1)],
        physical_qubit_count=2,
    )
    reversed_placement = next(
        alternative
        for alternative in alternatives
        if alternative.input_physical == (1, 0)
        and alternative.output_physical == (1, 0)
    )

    assert reversed_placement.cnot_count == 1
    target = permuted_partition_target(
        circuit.get_Matrix(np.empty((0,)), is_f32=False),
        reversed_placement.payload.input_assignment,
        reversed_placement.payload.output_assignment,
    )
    assert _process_infidelity(
        target,
        reversed_placement.payload.circuit.get_Matrix(
            reversed_placement.payload.parameters
        ),
    ) < 1e-14

    synthesized_alternatives = synthesize_partition_alternatives(
        partition=0,
        unitary=circuit.get_Matrix(np.empty((0,)), is_f32=False),
        logical_qubits=(0, 1),
        topology=[(0, 1)],
        physical_qubit_count=2,
        config={"exact_routing_eager_synthesis": False},
        source_circuit=circuit,
        source_parameters=np.empty((0,)),
    )
    synthesized_reversed = next(
        alternative
        for alternative in synthesized_alternatives
        if alternative.input_physical == (1, 0)
        and alternative.output_physical == (1, 0)
    )
    assert synthesized_reversed.cnot_count == 1


def test_nested_routing_blocks_convert_to_a_flat_cnot_basis():
    inner = qgd_Circuit(2)
    inner.add_SWAP([0, 1])
    outer = qgd_Circuit(2)
    outer.add_Circuit(inner)

    converted, parameters = utils.circuit_to_CNOT_basis(
        outer, np.empty((0,))
    )

    assert converted.get_Gate_Nums() == {"CNOT": 3}
    assert not any(
        isinstance(gate, qgd_Circuit) for gate in converted.get_Gates()
    )
    assert parameters.size == 0
    assert _process_infidelity(
        outer.get_Matrix(np.empty((0,))), converted.get_Matrix(parameters)
    ) < 1e-14


def test_fallback_routes_nonlocal_source_gates_on_the_local_subtopology():
    circuit = qgd_Circuit(3)
    circuit.add_U3(1)
    circuit.add_CNOT(2, 1)
    circuit.add_U3(2)
    parameters = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    alternatives = synthesize_partition_alternatives(
        partition=0,
        unitary=circuit.get_Matrix(parameters, is_f32=False),
        logical_qubits=(0, 1, 2),
        topology=[(0, 1), (1, 2)],
        physical_qubit_count=3,
        config={
            "max_partition_size": 3,
            "exact_routing_eager_synthesis": False,
            "synthesis_acceptance_tolerance": 1e-10,
        },
        source_circuit=circuit,
        source_parameters=parameters,
    )

    for alternative in alternatives:
        allowed = {frozenset(edge) for edge in alternative.payload.topology}
        for gate in alternative.payload.circuit.get_Flat_Circuit().get_Gates():
            qubits = tuple(gate.get_Involved_Qbits())
            if len(qubits) == 2:
                assert frozenset(qubits) in allowed


def test_exact_routing_audit_metric_replays_bit_identically(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "exact-routing.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(audit_path))
    circuit = qgd_Circuit(3)
    circuit.add_U3(0)
    circuit.add_CNOT(2, 0)
    parameters = np.array([0.123456789, 0.234567891, 0.345678912])
    topology = [(0, 1), (1, 2)]
    exact_route = route_circuit_exact(
        circuit,
        parameters,
        topology,
        {"max_partition_size": 2},
    )
    _append_exact_osr_routing_event(
        "routing",
        0,
        circuit,
        parameters,
        exact_route.circuit,
        exact_route.parameters,
        exact_route,
        topology,
        1e-10,
    )
    event = json.loads(audit_path.read_text().splitlines()[-1])

    _verify_exact_osr_routing_replay(event)


def test_exact_incumbent_guarantees_and_audits_a_feasible_route(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "exact-routing-sabre.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(audit_path))
    circuit = qgd_Circuit(4)
    circuit.add_CNOT(0, 1)
    circuit.add_CNOT(0, 2)
    circuit.add_CNOT(0, 3)
    parameters = np.empty((0,))
    topology = [(0, 1), (1, 2), (2, 3)]

    exact_route = route_circuit_exact(
        circuit,
        parameters,
        topology,
        {
            "max_partition_size": 2,
            "exact_routing_master": "branch-and-bound",
        },
    )
    assert exact_route.solution.selections

    _append_exact_osr_routing_event(
        "routing",
        1,
        circuit,
        parameters,
        exact_route.circuit,
        exact_route.parameters,
        exact_route,
        topology,
        1e-10,
    )
    event = json.loads(audit_path.read_text())
    assert event["cnot_count"] == exact_route.solution.cnot_count
    assert event["master_backend"] == exact_route.solution.master_backend
    assert event["transition_swap_count"] == sum(
        len(swaps) for swaps in (exact_route.solution.transition_swaps or ())
    )
    _verify_exact_osr_routing_replay(event)


def test_lazy_pam_osr_route_reaches_auditable_catalog_fixed_point(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "pam-osr-fixed-point.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(audit_path))
    circuit = qgd_Circuit(3)
    circuit.add_CNOT(0, 2)
    parameters = np.empty((0,))
    topology = [(0, 1), (1, 2)]

    result = route_circuit_exact(
        circuit,
        parameters,
        topology,
        {
            "max_partition_size": 2,
            "pam_osr_only": True,
            "strategy": "TreeSearch",
            "use_osr": True,
            "use_graph_search": True,
            "use_float": True,
            "parallel": 0,
            "tolerance": 1e-14,
            "synthesis_acceptance_tolerance": 1e-10,
            "routing_synthesis_workers": 2,
            "exact_routing_pam_swap_cnot_costs": (3.0,),
        },
    )

    assert result.timed_out is False
    assert result.lazy_rounds >= 1
    assert result.solution.master_backend.endswith("-fixed-point")
    qgd_Wide_Circuit_Optimization.check_valid_routing(
        result.circuit, topology
    )
    _append_exact_osr_routing_event(
        "routing",
        2,
        circuit,
        parameters,
        result.circuit,
        result.parameters,
        result,
        topology,
        1e-10,
    )
    _verify_exact_osr_routing_replay(json.loads(audit_path.read_text()))


def test_route_wide_timeout_returns_verified_light_sabre_incumbent():
    circuit = qgd_Circuit(4)
    circuit.add_CNOT(0, 3)
    result = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1), (1, 2), (2, 3)],
        {
            "max_partition_size": 3,
            "exact_routing_timeout_seconds": 1e-12,
            "exact_routing_cover_seed_timeout_seconds": 1e-12,
            "exact_routing_flow_seed": False,
        },
    )

    assert result.timed_out is True
    assert result.solution.optimal is False
    assert result.solution.master_backend.endswith("-timeout-incumbent")
    assert (
        result.solution.selections[0].alternative.payload.certificate_kind
        == "unitary"
    )


def test_light_sabre_fallback_audits_original_serialized_gate_order(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "light-sabre-source-order.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL", str(audit_path))
    circuit = qgd_Circuit(5)
    # Independent one-qubit gates make multiple topological orders valid,
    # while the structural fallback deliberately preserves this exact order.
    circuit.add_H(4)
    circuit.add_H(2)
    circuit.add_CNOT(0, 4)
    circuit.add_H(1)
    circuit.add_CNOT(0, 2)
    circuit.add_CNOT(0, 1)
    parameters = np.empty((0,))
    topology = tuple((qubit, qubit + 1) for qubit in range(4))
    result = route_circuit_exact(
        circuit,
        parameters,
        topology,
        {
            "max_partition_size": 3,
            "exact_routing_timeout_seconds": 1e-12,
            "exact_routing_cover_seed_timeout_seconds": 1e-12,
            "exact_routing_flow_seed": False,
            "exact_routing_precomputed_pam_seeds": False,
            "exact_routing_initial_mapping": tuple(range(5)),
        },
    )

    fallback = result.solution.selections[0].partition
    assert result.candidate_gate_orders[fallback] == tuple(
        range(len(circuit.get_Gates()))
    )
    # Reproduce the legacy metadata bug: the first two independent gates may
    # be topologically exchanged even though the archived source circuit keeps
    # its original serialized order. The writer must resolve this before save.
    legacy_orders = list(result.candidate_gate_orders)
    legacy_orders[fallback] = (1, 0, *range(2, len(circuit.get_Gates())))
    result = replace(result, candidate_gate_orders=tuple(legacy_orders))
    _append_exact_osr_routing_event(
        "routing",
        0,
        circuit,
        parameters,
        result.circuit,
        result.parameters,
        result,
        topology,
        1e-10,
    )
    event = json.loads(audit_path.read_text())

    assert event["selections"][0]["source_gate_indices"] == list(
        range(len(circuit.get_Gates()))
    )
    _verify_exact_osr_routing_replay(event)
    assert (
        result.solution.selections[0].alternative.payload.certificate_kind
        == "sabre"
    )


def test_osr_pricing_time_does_not_consume_the_master_budget(monkeypatch):
    captured = {}

    def delayed_pricing(targets, config, topology, *, target_configs=None):
        assert "_exact_routing_deadline" not in config
        time.sleep(0.02)
        return (None,) * len(targets)

    def capture_master(*, alternatives, timeout_seconds, **kwargs):
        captured["timeout_seconds"] = timeout_seconds
        captured["warm_start"] = kwargs["warm_start"]
        fallback_partition = max(alternatives)
        captured["fallback_partition"] = fallback_partition
        fallback = alternatives[fallback_partition][0]
        return ExactRoutingResult(
            selections=(RoutingSelection(fallback_partition, fallback),),
            cnot_count=fallback.cnot_count,
            single_qubit_count=fallback.single_qubit_count,
            initial_mapping=fallback.input_physical,
            final_mapping=fallback.output_physical,
            explored_states=0,
            master_backend="test-master",
            optimal=True,
        )

    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", delayed_pricing)
    monkeypatch.setattr(routing, "solve_exact_routing", capture_master)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {
            "strategy": "TreeSearch",
            "max_partition_size": 2,
            "exact_routing_master": "benders",
            "exact_routing_timeout_seconds": 7.0,
            "exact_routing_flow_seed": False,
        },
    )

    # The compact cover seed is solver work and is charged to this budget;
    # the deliberately delayed OSR pricing above is not.
    assert 6.5 < captured["timeout_seconds"] <= 7.0
    assert captured["warm_start"].master_backend.endswith("mip-start")
    assert all(
        selection.partition != captured["fallback_partition"]
        for selection in captured["warm_start"].selections
    )


def test_flow_seed_infeasibility_does_not_force_a_global_swap(monkeypatch):
    captured = {}

    def reject_flow_seed(*args, **kwargs):
        raise ValueError("test flow-seed infeasibility")

    def capture_master(**kwargs):
        captured["minimum_transition_cnot_cost"] = kwargs.get(
            "minimum_transition_cnot_cost"
        )
        return kwargs["warm_start"]

    monkeypatch.setattr(
        routing,
        "_solve_exact_routing_ilp_flow_restricted",
        reject_flow_seed,
    )
    monkeypatch.setattr(routing, "solve_exact_routing", capture_master)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    result = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {
            "strategy": "TreeSearch",
            "max_partition_size": 2,
            "exact_routing_master": "benders",
            "exact_routing_timeout_seconds": 7.0,
        },
    )

    assert captured["minimum_transition_cnot_cost"] == 0
    assert sum(map(len, result.solution.transition_swaps)) == 0


def test_flow_seed_nonfinite_bound_is_not_used_as_proof(monkeypatch):
    captured = {}

    def return_unproven_flow_seed(*args, warm_start, **kwargs):
        return replace(
            warm_start,
            master_backend="test-flow-seed",
            optimal=False,
            solver_bound=float("inf"),
        )

    def capture_master(**kwargs):
        captured["zero_swap_local_cnot_lower_bound"] = kwargs.get(
            "zero_swap_local_cnot_lower_bound"
        )
        return kwargs["warm_start"]

    monkeypatch.setattr(
        routing,
        "_solve_exact_routing_ilp_flow_restricted",
        return_unproven_flow_seed,
    )
    monkeypatch.setattr(routing, "solve_exact_routing", capture_master)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    result = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {
            "strategy": "TreeSearch",
            "max_partition_size": 2,
            "exact_routing_master": "benders",
            "exact_routing_timeout_seconds": 7.0,
        },
    )

    assert captured["zero_swap_local_cnot_lower_bound"] == 0
    assert result.solution.selections


def test_routing_synthesis_worker_default_is_memory_bounded(monkeypatch):
    monkeypatch.setattr(routing.mp, "cpu_count", lambda: 128)

    assert routing._routing_synthesis_worker_count({}, 1000) == 128
    assert (
        routing._routing_synthesis_worker_count(
            {"routing_synthesis_workers": 5}, 1000
        )
        == 5
    )
    assert (
        routing._routing_synthesis_worker_count(
            {"partition_workers": 7}, 1000
        )
        == 7
    )
    assert routing._routing_synthesis_worker_count({}, 3) == 3


def test_oversized_auxiliary_flow_seed_is_skipped(monkeypatch, capsys):
    captured = {}

    def reject_unexpected_flow_seed(*args, **kwargs):
        raise AssertionError("oversized auxiliary flow seed was constructed")

    def capture_master(**kwargs):
        captured["warm_start"] = kwargs["warm_start"]
        return kwargs["warm_start"]

    monkeypatch.setattr(
        routing,
        "_flow_seed_model_term_count",
        lambda *args, **kwargs: 500_001,
    )
    monkeypatch.setattr(
        routing,
        "_solve_exact_routing_ilp_flow_restricted",
        reject_unexpected_flow_seed,
    )
    monkeypatch.setattr(routing, "solve_exact_routing", capture_master)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)

    result = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {
            "strategy": "TreeSearch",
            "max_partition_size": 2,
            "exact_routing_master": "benders",
            "exact_routing_timeout_seconds": 7.0,
        },
    )

    assert result.solution.selections == captured["warm_start"].selections
    assert result.solution.cnot_count == captured["warm_start"].cnot_count
    assert "Skipping auxiliary routing flow seed" in capsys.readouterr().out
