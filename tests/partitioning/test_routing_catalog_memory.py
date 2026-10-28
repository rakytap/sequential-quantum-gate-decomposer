"""Bounded catalog preparation must preserve the exact synthesis workload."""

import weakref

import numpy as np
import pytest

from squander.partitioning import routing


@pytest.mark.parametrize("runtime_key", sorted(routing._ROUTING_SYNTHESIS_RUNTIME_KEYS))
def test_cache_keys_ignore_only_worker_ignored_runtime_artifacts(runtime_key):
    target = np.eye(4, dtype=np.complex128)
    topology = ((0, 1),)
    config = {
        "strategy": "TreeSearch",
        "tree_level_max": 3,
        "synthesis_acceptance_tolerance": 1e-10,
        "weights": np.array([1.0, 2.0]),
    }
    expected = routing._synthesis_target_cache_key(target, config, topology)
    runtime_array = np.arange(65536, dtype=np.float64)
    actual = routing._synthesis_target_cache_key(
        target, {**config, runtime_key: runtime_array}, topology
    )
    assert actual == expected
    assert routing._synthesis_target_cache_key(
        target, {**config, "tree_level_max": 4}, topology
    ) != expected
    assert routing._synthesis_target_cache_key(
        target, {**config, "weights": np.array([1.0, 3.0])}, topology
    ) != expected


def test_cache_reuses_successes_and_failures_when_runtime_artifacts_change(monkeypatch):
    calls = []

    def synthesize(targets, config, topology, *, target_configs=None):
        for task_config in target_configs:
            assert not routing._ROUTING_SYNTHESIS_RUNTIME_KEYS.intersection(task_config)
            assert task_config["tree_level_max"] == 3
            assert task_config["synthesis_acceptance_tolerance"] == 1e-10
        calls.append(len(targets))
        return ("verified", None)

    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", synthesize)
    monkeypatch.setattr(routing, "_check_routing_memory_pressure", lambda config: None)
    targets = (np.eye(2, dtype=np.complex128),
               np.array([[0, 1], [1, 0]], dtype=np.complex128))
    config = {
        "strategy": "TreeSearch",
        "tree_level_max": 3,
        "synthesis_acceptance_tolerance": 1e-10,
        "all_to_all_parameters": np.arange(65536, dtype=np.float64),
    }
    cache, stats = {}, [0]
    first = routing._call_shared_synthesis_batch_cached(
        targets, config, ((0, 1),), cache=cache, cache_stats=stats
    )
    second = routing._call_shared_synthesis_batch_cached(
        targets, {**config, "all_to_all_parameters": np.zeros(65536)},
        ((0, 1),), cache=cache, cache_stats=stats
    )
    assert first == second == ("verified", None)
    assert calls == [2]
    assert stats == [2]


@pytest.mark.parametrize("chunk_size", [1, 64, 20000])
def test_16028_partition_queue_preserves_targets_seeds_and_cached_failures(
    monkeypatch, chunk_size
):
    count = 16028
    topology = ((0, 1),)
    config = {
        "strategy": "TreeSearch",
        "exact_routing_random_seed": 23,
        "tree_level_max": 3,
        "synthesis_acceptance_tolerance": 1e-10,
    }
    metadata = {
        index: (np.array([[index % 7]], dtype=np.complex128),
                (0,), None, np.empty(0))
        for index in range(count)
    }
    live = weakref.WeakSet()
    peak = [0]
    solved = []

    class Deferred:
        def __init__(self):
            self.results = []
            live.add(self)
            peak[0] = max(peak[0], len(live))

        def resolve(self):
            return tuple(self.results)

    def prepare(*, unitary, config, synthesis_batch, **kwargs):
        result = Deferred()
        synthesis_batch.enqueue(topology, (unitary,), (config,), result.results.extend)
        return result

    def synthesize(targets, config, local_topology, *, target_configs=None):
        results = []
        for target, task_config in zip(targets, target_configs):
            index = int(target[0, 0].real)
            expected_seed = routing._deterministic_synthesis_seed(
                target, local_topology, 23
            )
            assert task_config["random_seed"] == expected_seed
            assert task_config["tree_level_max"] == 3
            assert task_config["synthesis_acceptance_tolerance"] == 1e-10
            solved.append(index)
            results.append(None if index == 3 else (index, expected_seed))
        return tuple(results)

    monkeypatch.setattr(routing, "_ROUTING_CATALOG_PREPARATION_BATCH_SIZE", chunk_size)
    monkeypatch.setattr(routing, "_check_routing_memory_pressure", lambda config: None)
    monkeypatch.setattr(routing, "synthesize_partition_alternatives", prepare)
    monkeypatch.setattr(routing, "_call_shared_synthesis_batch", synthesize)
    cache, stats = {}, [0]
    rows = list(routing._iter_refined_routing_catalog(
        list(range(count)), metadata, config, topology, 2, cache, stats
    ))
    assert len(rows) == count
    assert solved == list(range(7))
    assert stats == [count - 7]
    assert peak[0] <= min(chunk_size, count) + 1
    for expected_index, (partition, results) in enumerate(rows):
        assert partition == expected_index
        target = metadata[partition][0]
        seed = routing._deterministic_synthesis_seed(target, topology, 23)
        expected = None if partition % 7 == 3 else (partition % 7, seed)
        assert results == (expected,)


@pytest.mark.parametrize("headroom_mib", [0, 1, 128, 512])
def test_process_limit_guard_triggers_even_when_host_has_free_memory(
    monkeypatch, headroom_mib
):
    limit = 64 * 1024 ** 3
    monkeypatch.setattr(routing, "_available_memory_fraction", lambda: 0.9)
    monkeypatch.setattr(
        routing, "_process_address_space_usage",
        lambda: (limit - headroom_mib * 1024 ** 2, limit),
    )
    with pytest.raises(routing.RoutingMemoryPressure, match="process address-space"):
        routing._check_routing_memory_pressure({})


@pytest.mark.parametrize("usage", [None, (62 * 1024 ** 3, 64 * 1024 ** 3)])
def test_process_limit_guard_allows_adequate_or_unlimited_headroom(
    monkeypatch, usage
):
    monkeypatch.setattr(routing, "_available_memory_fraction", lambda: 0.9)
    monkeypatch.setattr(routing, "_process_address_space_usage", lambda: usage)
    routing._check_routing_memory_pressure({})


def test_catalog_checks_memory_before_preparing_another_partition(monkeypatch):
    limit = 64 * 1024 ** 3
    monkeypatch.setattr(routing, "_available_memory_fraction", lambda: 0.9)
    monkeypatch.setattr(
        routing, "_process_address_space_usage", lambda: (limit - 128 * 1024, limit)
    )

    def forbidden_preparation(**kwargs):
        pytest.fail("Catalog preparation ran without recovery headroom")

    monkeypatch.setattr(routing, "synthesize_partition_alternatives", forbidden_preparation)
    with pytest.raises(routing.RoutingMemoryPressure, match="process address-space"):
        list(routing._iter_refined_routing_catalog(
            [0], {0: (np.eye(2), (0,), None, np.empty(0))},
            {}, ((0, 1),), 2, {}, [0]
        ))

