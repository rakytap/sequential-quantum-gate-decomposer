"""Small exact controls for catalog-only path routing priorities."""
from dataclasses import dataclass
import itertools
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from squander.partitioning import ilp, routing


@dataclass(frozen=True)
class Alternative:
    partition: int
    logical_qubits: tuple
    input_physical: tuple
    output_physical: tuple
    cnot_count: int
    single_qubit_count: int
    word: tuple


def _word(incoming, outgoing):
    width = len(incoming)
    control, target = incoming[:2]
    if abs(control - target) == 1:
        result = [(control, target)]
    else:
        middle = next(q for q in range(width) if q not in (control, target))
        result = [(control, middle), (middle, target), (control, middle), (middle, target)]
    labels = [None] * width
    desired = [None] * width
    for logical, physical in enumerate(incoming):
        labels[physical] = logical
    for logical, physical in enumerate(outgoing):
        desired[physical] = logical
    for position in range(width):
        found = labels.index(desired[position])
        while found > position:
            a, b = found - 1, found
            first = result[-1] if result and set(result[-1]) == {a, b} else (a, b)
            result.extend((first, first[::-1], first))
            labels[a], labels[b] = labels[b], labels[a]
            found -= 1
    reduced = []
    for edge in result:
        if reduced and reduced[-1] == edge:
            reduced.pop()
        else:
            reduced.append(edge)
    return tuple(reduced)


def _alternatives(partition, block, path):
    values = []
    for incoming in itertools.permutations(range(len(block))):
        for outgoing in itertools.permutations(range(len(block))):
            word = _word(incoming, outgoing)
            values.append(Alternative(
                partition, block, tuple(path[q] for q in incoming),
                tuple(path[q] for q in outgoing), len(word), 0, word))
    return tuple(values)


def _problem(width, seed):
    rng = np.random.default_rng(993 + width + seed)
    path = tuple(10 + 3 * bit for bit in range(width))
    blocks = [tuple(int(q) for q in rng.choice(
        width, size=(3 if part % 4 == seed % 4 else 2), replace=False))
        for part in range(4)]
    predecessors = {}
    last = {}
    for part, block in enumerate(blocks):
        predecessors[part] = frozenset(last[q] for q in block if q in last)
        for q in block:
            last[q] = part
    args = dict(
        partitions=tuple(frozenset([part]) for part in range(4)),
        gate_predecessors=predecessors,
        alternatives={part: _alternatives(part, block, path)
                      for part, block in enumerate(blocks)},
        logical_qubit_count=width, path=path, initial_mapping=None, beam_width=512)
    return args, blocks


def _verify(result, args, blocks):
    width = args["logical_qubit_count"]
    position = {q: index for index, q in enumerate(args["path"])}
    target = QuantumCircuit(width)
    for block in blocks:
        target.cx(block[0], block[1])
    actual = QuantumCircuit(width)
    assert len(result.selections) == len(result.transition_swaps)
    for selected, swaps in zip(result.selections, result.transition_swaps):
        for a, b in swaps:
            left, right = position[a], position[b]
            actual.cx(left, right)
            actual.cx(right, left)
            actual.cx(left, right)
        alternative = selected.alternative
        start = min(position[q] for q in alternative.input_physical)
        for a, b in alternative.word:
            actual.cx(start + a, start + b)
    assert int(actual.count_ops().get("cx", 0)) == result.cnot_count

    def permutation(mapping):
        assert sorted(mapping) == sorted(args["path"])
        return [sum(((state >> q) & 1) << position[physical]
                    for q, physical in enumerate(mapping))
                for state in range(1 << width)]

    logical = Operator(target).data
    expected = np.empty_like(logical)
    expected[np.ix_(permutation(result.final_mapping),
                    permutation(result.initial_mapping))] = logical
    assert np.linalg.norm(Operator(actual).data - expected) == 0


def test_path_packing_bound_matches_exhaustive_windows():
    for width in range(1, 9):
        for count in range(1, min(width, 5) + 1):
            for positions in itertools.combinations(range(width), count):
                exact = min(sum(abs(position - start - index)
                                for index, position in enumerate(positions))
                            for start in range(width - count + 1))
                assert routing._minimum_path_block_gather_swaps(positions) == exact
    assert routing._minimum_path_block_gather_swaps(()) == 0
    assert routing._minimum_path_block_gather_swaps((0, 1, 98, 99)) == 192


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_guided_priority_cache_preserves_full_unitary(width, seed):
    args, blocks = _problem(width, seed)
    cache = {}
    anchor = routing._beam_selected_path_warm_start(
        **args, use_packing_bound=False, priority_cache=cache,
        priority_cache_key=(width, seed))
    guided = routing._beam_selected_path_warm_start(
        **args, use_packing_bound=True, priority_cache=cache,
        priority_cache_key=(width, seed))
    direct = routing._beam_selected_path_warm_start(**args, use_packing_bound=True)
    assert guided == direct
    assert anchor.cnot_count == guided.cnot_count
    for result in (anchor, guided, direct):
        _verify(result, args, blocks)


def test_identical_priorities_reuse_exact_raw_result():
    args, _ = _problem(3, 0)
    args.update(partitions=(frozenset([0]),), gate_predecessors={0: frozenset()},
                alternatives={0: args["alternatives"][0]})
    cache = {}
    anchor = routing._beam_selected_path_warm_start(
        **args, use_packing_bound=False, priority_cache=cache,
        priority_cache_key=("one-block",))
    guided = routing._beam_selected_path_warm_start(
        **args, use_packing_bound=True, priority_cache=cache,
        priority_cache_key=("one-block",))
    assert guided is anchor


def test_route_fingerprint_includes_all_transport():
    a = SimpleNamespace(
        initial_mapping=(0, 1, 2), final_mapping=(0, 1, 2), selections=(),
        transition_swaps=(((0, 1), (1, 2)),))
    b = SimpleNamespace(
        initial_mapping=(0, 1, 2), final_mapping=(0, 1, 2), selections=(),
        transition_swaps=(((1, 2), (0, 1)),))
    assert routing._catalog_route_trajectory_key(a) != routing._catalog_route_trajectory_key(b)
    b.transition_swaps = a.transition_swaps
    b.final_mapping = (1, 0, 2)
    assert routing._catalog_route_trajectory_key(a) != routing._catalog_route_trajectory_key(b)


def test_mapping_caches_retain_only_current_frontier():
    width = 8
    path = tuple(range(width))
    blocks = [(q, q + 1) for q in range(width - 1)]
    captured = {}
    previous = sys.getprofile()

    def profile(frame, event, value):
        if event == "return" and frame.f_code.co_name == "_beam_selected_path_warm_start":
            for name in ("cached_complete_mapping", "source_mapping_metadata",
                         "complete_mapping_and_swap_count", "block_mapping_swap_count"):
                cache_info = getattr(frame.f_locals.get(name), "cache_info", None)
                if cache_info is not None:
                    captured[name] = cache_info().currsize

    try:
        sys.setprofile(profile)
        routing._beam_selected_path_warm_start(
            partitions=tuple(frozenset([part]) for part in range(len(blocks))),
            gate_predecessors={part: (() if part == 0 else (part - 1,))
                               for part in range(len(blocks))},
            alternatives={part: _alternatives(part, block, path)
                          for part, block in enumerate(blocks)},
            logical_qubit_count=width, path=path, initial_mapping=path, beam_width=1)
    finally:
        sys.setprofile(previous)
    assert any(name in captured for name in (
        "complete_mapping_and_swap_count", "block_mapping_swap_count"))
    assert all(size <= 2 * (width - 1) + len(blocks)
               for size in captured.values())



@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("order_index", range(6))
@pytest.mark.parametrize("translation_limit", [None, 1, 2])
@pytest.mark.parametrize("fixed_initial", [False, True])
def test_boundary_contraction_matches_explicit_path_transport(
        width, order_index, translation_limit, fixed_initial):
    """Check exact boundary prices, selected words and physical permutations."""
    path = (17, 5, 29, 11, 3)
    block = tuple(range(width))
    orders = tuple(itertools.permutations(range(width)))
    order = orders[order_index % len(orders)]
    slots = (0, 2) if width == 2 else (0, 2, 4)
    initial = [None] * len(path)
    for logical in block:
        initial[logical] = path[slots[order[logical]]]
    remaining = iter(physical for physical in path if physical not in initial)
    for logical in range(width, len(path)):
        initial[logical] = next(remaining)
    initial = tuple(initial) if fixed_initial else None
    values = _alternatives(0, block, path)
    arguments = dict(
        partitions=(frozenset([0]),), gate_predecessors={0: ()},
        alternatives={0: values}, logical_qubit_count=len(path),
        path=path, initial_mapping=initial, beam_width=64,
        translation_limit=translation_limit)
    positions = {physical: index for index, physical in enumerate(path)}
    expected = {}
    for alternative in values:
        incoming = tuple(positions[p] for p in alternative.input_physical)
        outgoing = tuple(positions[p] for p in alternative.output_physical)
        starts = tuple(range(len(path) - width + 1))
        if translation_limit is not None and len(starts) > translation_limit:
            if initial is None:
                last = len(starts) - 1
                starts = tuple(sorted({
                    round(index * last / (translation_limit - 1))
                    if translation_limit > 1 else last // 2
                    for index in range(translation_limit)}))
            else:
                starts = tuple(sorted(starts, key=lambda start: (
                    sum(abs(positions[initial[q]] - start - incoming[q])
                        for q in block), start))[:translation_limit])
        for start in starts:
            desired_in = tuple(path[start + index] for index in incoming)
            desired_out = tuple(path[start + index] for index in outgoing)
            mapping_in = [None] * len(path)
            for logical, physical in zip(block, desired_in):
                mapping_in[logical] = physical
            outside = (list(range(width, len(path))) if initial is None else
                       [initial.index(physical) for physical in path
                        if initial.index(physical) not in block])
            available = (physical for physical in path
                         if physical not in desired_in)
            for logical, physical in zip(outside, available):
                mapping_in[logical] = physical
            mapping_in = tuple(mapping_in)
            crossings = 0 if initial is None else sum(
                (positions[initial[left]] < positions[initial[right]]) !=
                (positions[mapping_in[left]] < positions[mapping_in[right]])
                for left in range(len(path))
                for right in range(left + 1, len(path)))
            mapping_out = list(mapping_in)
            for logical, physical in zip(block, desired_out):
                mapping_out[logical] = physical
            mapping_out = tuple(mapping_out)
            candidate = (alternative.cnot_count + 3 * crossings,
                         alternative.single_qubit_count, mapping_out,
                         mapping_in if initial is None else initial,
                         alternative.word, desired_in, desired_out, crossings)
            previous = expected.get(mapping_out)
            if previous is None or candidate[:2] < previous[:2]:
                expected[mapping_out] = candidate
    chosen = min(expected.values(), key=lambda item: item[:3])
    result = routing._beam_selected_path_warm_start(**arguments)
    assert (result.cnot_count, result.single_qubit_count,
            result.final_mapping, result.initial_mapping) == chosen[:4]
    selected = result.selections[0].alternative
    assert (selected.word, selected.input_physical,
            selected.output_physical) == chosen[4:7]
    assert sum(map(len, result.transition_swaps)) == chosen[7]
    _verify(result, arguments, [block])


def _cover_graph(monkeypatch, successors, starts, costs, fail_first=False):
    class Catalog:
        pass

    class Circuit:
        def get_Qbit_Num(self):
            return 3

        def get_Gates(self):
            return (0, 1, 2)

    catalog = Catalog()
    catalog.circuit = Circuit()
    catalog.parameters = ()
    catalog.topology = ((0, 1), (1, 2))
    catalog.partitions = (frozenset([0]), frozenset([1]), frozenset([2]))
    catalog.alternatives = {0: (SimpleNamespace(logical_qubits=(0, 1)),)}
    monkeypatch.setattr(routing, "LoadedRoutingOSRCatalog", Catalog)
    monkeypatch.setattr(routing, "augment_catalog_with_recombined_covers", lambda value: value)
    monkeypatch.setattr(routing, "_path_topology_order", lambda *args: (0, 1, 2))
    monkeypatch.setattr(ilp, "get_all_partitions",
                        lambda *args: (None, None, None, {0: (), 1: (), 2: ()}, None, None, None))
    monkeypatch.setattr(routing, "_light_sabre_route",
                        lambda circuit, params, topology, config:
                        (None, None, None, None, starts[config["sabre_seed"]]))
    monkeypatch.setattr(routing, "_mapping_trajectory_from_light_sabre",
                        lambda trace, initial, count: (trace, trace))
    monkeypatch.setattr(routing, "mapping_aware_partition_weights",
                        lambda **kwargs: kwargs["gate_before_mapping"])
    monkeypatch.setattr(routing, "_minimum_partition_cover_selection",
                        lambda **kwargs: (kwargs["weights"],))
    calls = []

    def solve(**kwargs):
        node = kwargs["selected_partitions"][0]
        mode = kwargs["use_packing_bound"]
        calls.append((node, mode))
        if fail_first and len(calls) == 1:
            return None
        next_node = successors[node, mode]
        output = tuple((q + next_node) % 3 for q in range(3))
        alt = SimpleNamespace(logical_qubits=(0, 1, 2),
                              input_physical=(0, 1, 2), output_physical=output)
        return SimpleNamespace(
            node=node, next_node=next_node, cnot_count=costs[node, mode],
            single_qubit_count=0, master_backend=kwargs["master_backend"],
            initial_mapping=(0, 1, 2), final_mapping=output, transition_swaps=(),
            selections=(SimpleNamespace(partition=node, alternative=alt),))

    monkeypatch.setattr(routing, "_cover_selection_warm_start", solve)
    monkeypatch.setattr(routing, "_mapping_trajectory_from_solution",
                        lambda route, parts: (route.next_node, route.next_node))
    return catalog, calls


def test_cached_tail_is_not_a_cycle(monkeypatch):
    successors = {(0, False): 1, (1, False): 2, (2, False): 2}
    costs = {(0, False): 12, (1, False): 10, (2, False): 7}
    catalog, calls = _cover_graph(monkeypatch, successors, [0, 1, 0], costs)
    result = routing.mapping_aware_catalog_seeds(
        catalog, reference_seed_count=3, refinement_rounds=2, use_packing_bound=False)
    assert calls == [(0, False), (1, False), (2, False)]
    assert result[0].cnot_count == 7


def test_real_cycle_terminates_and_failed_cover_can_retry(monkeypatch):
    successors = {(0, False): 1, (1, False): 0}
    costs = {(0, False): 9, (1, False): 8}
    catalog, calls = _cover_graph(monkeypatch, successors, [0, 1], costs)
    routing.mapping_aware_catalog_seeds(
        catalog, reference_seed_count=2, refinement_rounds=100, use_packing_bound=False)
    assert calls == [(0, False), (1, False)]
    catalog, calls = _cover_graph(monkeypatch, {(0, False): 0}, [0, 0],
                                  {(0, False): 8}, fail_first=True)
    result = routing.mapping_aware_catalog_seeds(
        catalog, reference_seed_count=2, refinement_rounds=2, use_packing_bound=False)
    assert calls == [(0, False), (0, False)]
    assert result[0].cnot_count == 8


@pytest.mark.parametrize("guided_cost", [7, 71])
def test_packing_cannot_evict_anchor_result(monkeypatch, guided_cost):
    successors = {(0, False): 0, (0, True): 0}
    costs = {(0, False): 69, (0, True): guided_cost}
    catalog, calls = _cover_graph(monkeypatch, successors, [0], costs)
    result = routing.mapping_aware_catalog_seeds(
        catalog, reference_seed_count=1, refinement_rounds=2)
    assert calls == [(0, False), (0, True)]
    assert any(route.cnot_count == 69 for route in result)
    assert result[0].cnot_count == min(69, guided_cost)
