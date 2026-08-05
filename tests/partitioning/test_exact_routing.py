import json

import numpy as np

from squander import utils
import squander.partitioning.routing as routing
from squander.gates.qgd_Circuit import qgd_Circuit
from squander.decomposition.qgd_Wide_Circuit_Optimization import (
    _append_exact_osr_routing_event,
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


def test_exact_osr_ilp_is_the_wide_router_default():
    optimizer = qgd_Wide_Circuit_Optimization({})

    assert optimizer.config["routing-strategy"] == "exact-osr"
    assert optimizer.config["exact_routing_master"] == "ilp"
    assert optimizer.config["exact_routing_lazy_osr"] is False
    assert optimizer.config["exact_routing_total_timeout_seconds"] == 20 * 60


def test_three_qubit_topology_symmetry_counts():
    path = [(0, 1), (1, 2)]
    triangle = [(0, 1), (1, 2), (0, 2)]

    assert len(topology_automorphisms(path, 3)) == 2
    assert len(topology_automorphisms(triangle, 3)) == 6
    assert len(symmetry_reduced_assignment_orbits(path, 3)) == 18
    assert len(symmetry_reduced_assignment_orbits(triangle, 3)) == 6


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


def test_ilp_encodes_and_propagates_a_fixed_initial_permutation():
    alternatives = {
        0: [RoutingAlternative(0, (0, 1), (1, 0), (0, 1), 1)],
        1: [RoutingAlternative(1, (0, 2), (0, 2), (0, 2), 1)],
    }
    result = solve_exact_routing_ilp(
        gate_predecessors={0: (), 1: (0,)},
        partitions=[{0}, {1}],
        alternatives=alternatives,
        logical_qubit_count=3,
        initial_mapping=(1, 0, 2),
    )

    assert result.initial_mapping == (1, 0, 2)
    assert result.final_mapping == (0, 1, 2)
    assert [selection.partition for selection in result.selections] == [0, 1]


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
    assert captured["values"]["cfg_0"] == 0
    assert captured["values"]["cfg_1"] == 1
    assert captured["values"]["location_q0_b0_p0"] == 1
    assert captured["values"]["location_q1_b0_p1"] == 1
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


def test_cbc_cycle_cuts_reject_a_convex_cyclic_cover(monkeypatch):
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

    assert result.explored_states >= 1
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


def test_exhaustive_osr_is_default_and_lazy_pricing_is_explicit():
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
        circuit, np.empty((0,)), [(0, 1)], common
    )
    lazy = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1)],
        {**common, "exact_routing_lazy_osr": True},
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

    def reject_all(targets, config, topology, *, target_configs=None):
        calls.append(len(targets))
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

    assert calls == [2]
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

    # The two symmetry representatives have naive fallback costs one and
    # four CNOTs respectively, hence exact strict-improvement depths 0 and 3.
    assert sorted(captured_depths) == [0, 3]


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


def test_sabre_incumbent_guarantees_and_audits_a_feasible_route(
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
    assert len(exact_route.solution.selections) == 1
    assert (
        exact_route.solution.selections[0].alternative.payload.certificate_kind
        == "sabre"
    )

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
    assert event["selections"][0]["certificate_kind"] == "sabre"
    assert any(
        action["kind"] == "swap"
        for action in event["selections"][0]["actions"]
    )
    _verify_exact_osr_routing_replay(event)


def test_route_wide_timeout_returns_verified_light_sabre_incumbent():
    circuit = qgd_Circuit(4)
    circuit.add_CNOT(0, 3)
    result = route_circuit_exact(
        circuit,
        np.empty((0,)),
        [(0, 1), (1, 2), (2, 3)],
        {
            "max_partition_size": 3,
            "exact_routing_total_timeout_seconds": 1e-12,
        },
    )

    assert result.timed_out is True
    assert result.solution.optimal is False
    assert result.solution.master_backend == "light-sabre-timeout-incumbent"
    assert (
        result.solution.selections[0].alternative.payload.certificate_kind
        == "sabre"
    )
