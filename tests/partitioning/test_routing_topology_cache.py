"""Exercise actual topology enumeration and fallback construction, not a mock."""

import itertools
import math

import numpy as np
import pytest

from squander.partitioning import routing


def _reference_embeddings(topology, size, width):
    device_edges = routing._normalize_edges(topology)
    grouped = {}
    for vertices in itertools.combinations(range(int(size)), width):
        canonical = routing._canonical_labeled_topology(vertices, device_edges)
        if canonical is not None:
            edges, embedding = canonical
            grouped.setdefault(edges, set()).add(embedding)
    return {edges: tuple(sorted(embeddings))
            for edges, embeddings in sorted(grouped.items())}


def _reference_boundary_keys(topology, size, width):
    assignments = tuple(itertools.permutations(range(int(width))))
    expected = set()
    for embeddings in _reference_embeddings(topology, size, width).values():
        for embedding in embeddings:
            for incoming, outgoing in itertools.product(assignments, repeat=2):
                before = [0] * width
                after = [0] * width
                for physical, logical in enumerate(incoming):
                    before[logical] = embedding[physical]
                for physical, logical in enumerate(outgoing):
                    after[logical] = embedding[physical]
                expected.add((tuple(before), tuple(after)))
    return expected


@pytest.fixture(autouse=True)
def clear_topology_caches():
    routing._connected_subtopology_embeddings_cached.cache_clear()
    routing._expected_boundary_transition_keys_cached.cache_clear()
    yield
    routing._connected_subtopology_embeddings_cached.cache_clear()
    routing._expected_boundary_transition_keys_cached.cache_clear()


def test_all_four_vertex_graphs_match_uncached_enumeration_and_order():
    edges = tuple(itertools.combinations(range(4), 2))
    for mask in range(1 << len(edges)):
        graph = tuple(edge for index, edge in enumerate(edges) if mask & (1 << index))
        for width in (1, 2, 3):
            expected = _reference_embeddings(graph, 4, width)
            actual = routing.connected_subtopology_embeddings(graph, 4, width)
            assert list(actual.items()) == list(expected.items())
            assert routing._expected_boundary_transition_keys(graph, 4, width) == (
                _reference_boundary_keys(graph, 4, width))


def test_128_qubit_device_is_scanned_once_across_both_helpers(monkeypatch):
    graph = tuple((wire, wire + 1) for wire in range(127))
    original = routing._canonical_labeled_topology
    calls = [0]

    def count(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(routing, "_canonical_labeled_topology", count)
    embeddings = routing.connected_subtopology_embeddings(graph, 128, 3)
    keys = routing._expected_boundary_transition_keys(graph, 128, 3)
    assert sum(map(len, embeddings.values())) == 126
    assert len(keys) == 126 * 36
    assert calls == [math.comb(128, 3)]

    reversed_graph = tuple((right, left) for left, right in reversed(graph))
    for _ in range(100):
        assert routing.connected_subtopology_embeddings(reversed_graph, 128, 3) == embeddings
        assert routing._expected_boundary_transition_keys(reversed_graph, 128, 3) == keys
    assert calls == [math.comb(128, 3)]


def test_callers_cannot_mutate_cached_geometry():
    graph = [(0, 1), (1, 2), (2, 3)]
    expected_embeddings = _reference_embeddings(graph, 4, 3)
    expected_keys = _reference_boundary_keys(graph, 4, 3)
    routing.connected_subtopology_embeddings(graph, 4, 3).clear()
    routing._expected_boundary_transition_keys(graph, 4, 3).clear()
    assert routing.connected_subtopology_embeddings(graph, 4, 3) == expected_embeddings
    assert routing._expected_boundary_transition_keys(graph, 4, 3) == expected_keys
    graph.append((0, 3))
    assert routing.connected_subtopology_embeddings(graph, 4, 3) == (
        _reference_embeddings(graph, 4, 3))


def test_cache_keys_include_physical_size_width_and_normalized_edges():
    graph = ((0, 1), (1, 2), (2, 3))
    for size, width in ((4, 2), (4, 3), (5, 3)):
        assert routing.connected_subtopology_embeddings(graph, size, width) == (
            _reference_embeddings(graph, size, width))
        assert routing._expected_boundary_transition_keys(graph, size, width) == (
            _reference_boundary_keys(graph, size, width))
    assert routing._connected_subtopology_embeddings_cached.cache_info().misses == 3
    assert routing._expected_boundary_transition_keys_cached.cache_info().misses == 3
    equivalent = ((3, 2), (1, 0), (2, 1), (0, 1), (2, 2))
    assert routing.connected_subtopology_embeddings(equivalent, 4, 3) == (
        _reference_embeddings(graph, 4, 3))
    assert routing._connected_subtopology_embeddings_cached.cache_info().misses == 3


def test_topology_caches_have_bounded_entry_counts():
    for size in range(2, 42):
        graph = tuple((wire, wire + 1) for wire in range(size - 1))
        routing.connected_subtopology_embeddings(graph, size, 1)
        routing._expected_boundary_transition_keys(graph, size, 1)
    for cache in (routing._connected_subtopology_embeddings_cached,
                  routing._expected_boundary_transition_keys_cached):
        info = cache.cache_info()
        assert info.maxsize == 32
        assert info.currsize == 32


@pytest.mark.parametrize("helper", [
    "connected_subtopology_embeddings", "_expected_boundary_transition_keys",
])
def test_nonintegral_width_still_rejected_after_cache_is_warm(helper):
    function = getattr(routing, helper)
    function(((0, 1), (1, 2)), 3, 3)
    with pytest.raises(TypeError):
        function(((0, 1), (1, 2)), 3, 3.0)


def _alternative_signature(alternatives):
    matrices = {}
    rows = []
    for alternative in alternatives:
        payload = alternative.payload
        identity = id(payload)
        if identity not in matrices:
            matrices[identity] = np.asarray(payload.circuit.get_Matrix(
                payload.parameters, is_f32=False)).tobytes()
        rows.append((
            alternative.input_physical, alternative.output_physical,
            alternative.cnot_count, alternative.single_qubit_count,
            payload.topology, payload.input_assignment, payload.output_assignment,
            matrices[identity],
        ))
    return tuple(rows)


def _real_128q_fallback_case():
    from squander.gates.qgd_Circuit import qgd_Circuit

    circuit = qgd_Circuit(3)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(2)
    circuit.add_CNOT(2, 1)
    parameters = np.array([0.123, -0.234, 0.345, 0.456, -0.567, 0.678])
    return dict(
        partition=0, unitary=circuit.get_Matrix(parameters, is_f32=False),
        logical_qubits=(0, 1, 2),
        topology=tuple((wire, wire + 1) for wire in range(127)),
        physical_qubit_count=128,
        config={"exact_routing_eager_synthesis": False,
                "synthesis_acceptance_tolerance": 1e-10},
        source_circuit=circuit, source_parameters=parameters,
    )


def test_real_128q_fallback_catalog_matches_uncached_counts_mappings_and_matrices(
    monkeypatch,
):
    case = _real_128q_fallback_case()
    with monkeypatch.context() as original:
        original.setattr(routing, "connected_subtopology_embeddings", _reference_embeddings)
        original.setattr(routing, "_expected_boundary_transition_keys", _reference_boundary_keys)
        reference = routing.synthesize_partition_alternatives(**case)
    cold = routing.synthesize_partition_alternatives(**case)
    warm = routing.synthesize_partition_alternatives(**case)
    assert len(reference) == len(cold) == len(warm) == 126 * 36
    expected = _alternative_signature(reference)
    assert _alternative_signature(cold) == expected
    assert _alternative_signature(warm) == expected

