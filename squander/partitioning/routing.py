"""Exact permutation-aware routing over pre-synthesized partition alternatives.

This module deliberately separates expensive local synthesis from global routing.
For each feasible gate partition, a caller supplies topology-valid alternatives
with an input placement, output placement, and exact CNOT cost.  The solver then
chooses an exact cover, an acyclic execution order, and a compatible mapping
trajectory with minimum total CNOT count.

The default global solver is a PuLP exact-cover model with mapping-compatible
flow along each logical wire. A preserved dependency-ready branch-and-bound
backend is available both as a fallback and as a correctness cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import collections
import functools
import hashlib
import itertools
import multiprocessing as mp
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Permutation = tuple[int, ...]
Edge = tuple[int, int]


def _isolated_synthesis_worker(connection, target, config, topology):
    """Run the shared native callback behind a process-crash boundary."""
    try:
        from squander.decomposition.qgd_Wide_Circuit_Optimization import (
            synthesize_partition_with_squander,
        )

        result = synthesize_partition_with_squander(
            target,
            config,
            mini_topology=topology,
        )
        if result is None:
            connection.send(("none",))
        else:
            from qiskit import qasm2
            from squander import Qiskit_IO

            portable_circuit = Qiskit_IO.get_Qiskit_Circuit(
                result.circuit.get_Flat_Circuit(), result.parameters
            )
            connection.send(
                (
                    "result_qasm",
                    qasm2.dumps(portable_circuit),
                    result.config,
                    result.topology,
                )
            )
    except BaseException as error:
        connection.send(("error", repr(error)))
    finally:
        connection.close()


def _call_shared_synthesis(target, config, topology):
    """Call native synthesis directly or in an isolated worker process."""
    if not config.get("exact_routing_isolate_synthesis", True):
        from squander.decomposition.qgd_Wide_Circuit_Optimization import (
            synthesize_partition_with_squander,
        )

        return synthesize_partition_with_squander(
            target, config, mini_topology=topology
        )

    parent_connection, child_connection = mp.Pipe(duplex=False)
    process = mp.Process(
        target=_isolated_synthesis_worker,
        args=(child_connection, target, dict(config), list(topology)),
    )
    process.start()
    child_connection.close()
    timeout = float(config.get("routing_synthesis_timeout_seconds", 300.0))
    deadline = time.monotonic() + timeout
    message = None
    while time.monotonic() < deadline:
        if parent_connection.poll(0.1):
            message = parent_connection.recv()
            break
        if not process.is_alive():
            break
    if process.is_alive():
        process.terminate()
    process.join()
    parent_connection.close()
    if message is None:
        return None
    return _decode_synthesis_message(message)


def _decode_synthesis_message(message):
    """Reconstruct a fresh native circuit from a portable worker response."""
    kind, *payload = message
    if kind == "error":
        raise RuntimeError(f"Isolated Squander synthesis failed: {payload[0]}")
    if kind == "none":
        return None
    if kind != "result_qasm":
        raise RuntimeError(f"Unknown isolated synthesis response {kind!r}.")

    from qiskit import QuantumCircuit
    from squander import Qiskit_IO
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        SquanderPartitionSynthesisResult,
    )

    qiskit_circuit = QuantumCircuit.from_qasm_str(payload[0])
    circuit, parameters = Qiskit_IO.convert_Qiskit_to_Squander(qiskit_circuit)
    return SquanderPartitionSynthesisResult(
        circuit=circuit,
        parameters=np.asarray(parameters, dtype=np.float64),
        config=payload[1],
        topology=payload[2],
    )


def _call_shared_synthesis_batch(
    targets, config, topology, *, target_configs=None
):
    """Run independent OSR targets concurrently with crash isolation."""
    targets = tuple(targets)
    if not targets:
        return ()
    configurations = (
        tuple(dict(config) for _ in targets)
        if target_configs is None
        else tuple(dict(value) for value in target_configs)
    )
    if len(configurations) != len(targets):
        raise ValueError("One synthesis config is required per target.")
    if not config.get("exact_routing_isolate_synthesis", True):
        return tuple(
            _call_shared_synthesis(target, target_config, topology)
            for target, target_config in zip(targets, configurations)
        )

    configured_workers = config.get("routing_synthesis_workers")
    if configured_workers is None:
        configured_workers = config.get("partition_workers")
    if configured_workers is None:
        configured_workers = mp.cpu_count()
    worker_count = max(1, min(int(configured_workers), len(targets)))
    timeout = float(config.get("routing_synthesis_timeout_seconds", 300.0))
    route_deadline = config.get("_exact_routing_deadline")
    if route_deadline is not None:
        route_deadline = float(route_deadline)
    results = [None] * len(targets)
    pending = iter(enumerate(zip(targets, configurations)))
    active = {}

    def launch(index, task):
        target, target_config = task
        parent_connection, child_connection = mp.Pipe(duplex=False)
        process = mp.Process(
            target=_isolated_synthesis_worker,
            args=(
                child_connection,
                target,
                target_config,
                list(topology),
            ),
        )
        process.start()
        child_connection.close()
        active[index] = (
            process,
            parent_connection,
            min(
                time.monotonic() + timeout,
                route_deadline if route_deadline is not None else float("inf"),
            ),
        )

    for _ in range(worker_count):
        try:
            launch(*next(pending))
        except StopIteration:
            break

    while active:
        if route_deadline is not None and time.monotonic() >= route_deadline:
            for process, connection, _deadline in active.values():
                if process.is_alive():
                    process.terminate()
                process.join()
                connection.close()
            break
        progressed = False
        for index, (process, connection, deadline) in list(active.items()):
            message = None
            if connection.poll():
                message = connection.recv()
            elif process.is_alive() and time.monotonic() < deadline:
                continue
            if process.is_alive():
                process.terminate()
            process.join()
            connection.close()
            if message is not None:
                results[index] = _decode_synthesis_message(message)
            del active[index]
            progressed = True
            if route_deadline is None or time.monotonic() < route_deadline:
                try:
                    launch(*next(pending))
                except StopIteration:
                    pass
        if not progressed:
            time.sleep(0.01)
    return tuple(results)


def _freeze_synthesis_cache_value(value):
    """Return an exact, hashable representation for a synthesis setting."""
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, float):
        return ("float64", np.float64(value).tobytes())
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return (
            "ndarray",
            contiguous.dtype.str,
            contiguous.shape,
            contiguous.view(np.uint8).tobytes(),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (str(key), _freeze_synthesis_cache_value(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(_freeze_synthesis_cache_value(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        return (
            type(value).__name__,
            tuple(
                sorted(_freeze_synthesis_cache_value(item) for item in value)
            ),
        )
    # Unknown objects are deliberately distinguished by identity. This can
    # forgo a cache hit, but can never alias two materially different inputs.
    return (
        "object",
        type(value).__module__,
        type(value).__qualname__,
        id(value),
    )


def _synthesis_target_cache_key(target, config, topology):
    """Key one exact synthesis request without unsafe semantic aliasing."""
    target = np.ascontiguousarray(target, dtype=np.complex128)
    return (
        target.shape,
        target.view(np.uint8).tobytes(),
        tuple(sorted(tuple(sorted((int(u), int(v)))) for u, v in topology)),
        _freeze_synthesis_cache_value(config),
    )


def _deterministic_synthesis_seed(target, topology, base_seed):
    """Derive one stable native RNG seed from an exact synthesis target."""
    target = np.ascontiguousarray(target, dtype=np.complex128)
    digest = hashlib.sha256()
    digest.update(np.int64(int(base_seed)).tobytes())
    digest.update(np.asarray(target.shape, dtype=np.int64).tobytes())
    digest.update(target.view(np.uint8).tobytes())
    digest.update(
        np.asarray(
            sorted(tuple(sorted((int(u), int(v)))) for u, v in topology),
            dtype=np.int64,
        ).tobytes()
    )
    return int.from_bytes(digest.digest()[:4], "little", signed=False)


def _call_shared_synthesis_batch_cached(
    targets,
    config,
    topology,
    *,
    target_configs=None,
    cache=None,
    cache_stats=None,
):
    """Synthesize each distinct exact target once, caching failures as final."""
    targets = tuple(targets)
    configurations = (
        tuple(dict(config) for _ in targets)
        if target_configs is None
        else tuple(dict(value) for value in target_configs)
    )
    if len(configurations) != len(targets):
        raise ValueError("One synthesis config is required per target.")
    base_seed = config.get("exact_routing_random_seed")
    if base_seed is not None:
        configurations = tuple(
            {
                **target_config,
                "random_seed": target_config.get(
                    "random_seed",
                    _deterministic_synthesis_seed(
                        target, topology, base_seed
                    ),
                ),
            }
            for target, target_config in zip(targets, configurations)
        )
    if cache is None:
        cache = {}
    if cache_stats is None:
        cache_stats = [0]

    keys = tuple(
        _synthesis_target_cache_key(target, target_config, topology)
        for target, target_config in zip(targets, configurations)
    )
    missing_keys = []
    missing_targets = []
    missing_configs = []
    scheduled = set()
    for key, target, target_config in zip(keys, targets, configurations):
        if key in cache or key in scheduled:
            cache_stats[0] += 1
            continue
        scheduled.add(key)
        missing_keys.append(key)
        missing_targets.append(target)
        missing_configs.append(target_config)

    missing_results = (
        _call_shared_synthesis_batch(
            missing_targets,
            config,
            topology,
            target_configs=missing_configs,
        )
        if missing_targets
        else ()
    )
    for key, result in zip(missing_keys, missing_results):
        # None is intentional: a failed OSR request is final for this route.
        cache[key] = result
    return tuple(cache[key] for key in keys)


def _sorted_routing_alternatives(alternatives_by_transition):
    return tuple(
        sorted(
            alternatives_by_transition.values(),
            key=lambda value: (
                value.cnot_count,
                value.single_qubit_count,
                value.input_physical,
                value.output_physical,
            ),
        )
    )


class _DeferredRoutingAlternatives:
    """Alternatives populated after a shared synthesis batch completes."""

    def __init__(self, alternatives_by_transition):
        self.alternatives_by_transition = alternatives_by_transition

    def resolve(self):
        return _sorted_routing_alternatives(self.alternatives_by_transition)


class _GlobalRoutingSynthesisBatch:
    """One deduplicated worker queue per canonical local topology."""

    def __init__(self, config, cache, cache_stats):
        self.config = dict(config)
        self.cache = cache
        self.cache_stats = cache_stats
        self.requests = collections.defaultdict(list)

    def enqueue(self, topology, targets, target_configs, consume):
        topology = tuple(sorted(tuple(map(int, edge)) for edge in topology))
        self.requests[topology].append(
            (tuple(targets), tuple(target_configs), consume)
        )

    def run(self):
        for topology, requests in sorted(self.requests.items()):
            flat_targets = tuple(
                target
                for targets, _configs, _consume in requests
                for target in targets
            )
            flat_configs = tuple(
                target_config
                for _targets, configs, _consume in requests
                for target_config in configs
            )
            flat_results = _call_shared_synthesis_batch_cached(
                flat_targets,
                self.config,
                topology,
                target_configs=flat_configs,
                cache=self.cache,
                cache_stats=self.cache_stats,
            )
            offset = 0
            for targets, _configs, consume in requests:
                count = len(targets)
                consume(flat_results[offset : offset + count])
                offset += count


def _normalize_edges(edges: Iterable[Sequence[int]]) -> frozenset[frozenset[int]]:
    """Return an undirected, loop-free edge set."""
    return frozenset(
        frozenset((int(edge[0]), int(edge[1])))
        for edge in edges
        if int(edge[0]) != int(edge[1])
    )


def topology_automorphisms(
    edges: Iterable[Sequence[int]],
    width: int,
) -> tuple[Permutation, ...]:
    """Enumerate wire relabelings that preserve an undirected topology."""
    width = int(width)
    if width < 1:
        raise ValueError("Topology width must be positive.")
    normalized = _normalize_edges(edges)
    automorphisms = []
    for permutation in itertools.permutations(range(width)):
        relabeled = frozenset(
            frozenset((permutation[u], permutation[v]))
            for u, v in (tuple(edge) for edge in normalized)
        )
        if relabeled == normalized:
            automorphisms.append(tuple(permutation))
    return tuple(automorphisms)


def _permutation_swaps(
    permutation: Sequence[int], edges: Iterable[Sequence[int]]
) -> tuple[Edge, ...]:
    """Return a shortest topology-valid SWAP sequence for a permutation."""
    permutation = tuple(int(value) for value in permutation)
    identity = tuple(range(len(permutation)))
    if tuple(sorted(permutation)) != identity:
        raise ValueError("Invalid permutation.")
    normalized_edges = tuple(
        sorted(tuple(sorted(tuple(edge))) for edge in _normalize_edges(edges))
    )
    queue = collections.deque([identity])
    previous = {identity: None}
    previous_swap = {}
    while queue:
        state = queue.popleft()
        if state == permutation:
            swaps = []
            while previous[state] is not None:
                swaps.append(previous_swap[state])
                state = previous[state]
            return tuple(reversed(swaps))
        for left, right in normalized_edges:
            updated = list(state)
            updated[left], updated[right] = updated[right], updated[left]
            updated = tuple(updated)
            if updated not in previous:
                previous[updated] = state
                previous_swap[updated] = (left, right)
                queue.append(updated)
    raise ValueError("Permutation cannot be realized on a disconnected topology.")


def _inverse_permutation(permutation: Sequence[int]) -> Permutation:
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[int(value)] = index
    return tuple(inverse)


def _shortest_topology_path(
    source: int, target: int, edges: Iterable[Sequence[int]]
) -> tuple[int, ...]:
    """Return a shortest undirected path between two local topology wires."""
    source, target = int(source), int(target)
    adjacency = collections.defaultdict(set)
    for edge in _normalize_edges(edges):
        left, right = tuple(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    queue = collections.deque([source])
    previous = {source: None}
    while queue:
        current = queue.popleft()
        if current == target:
            path = []
            while current is not None:
                path.append(current)
                current = previous[current]
            return tuple(reversed(path))
        for neighbour in sorted(adjacency[current]):
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)
    raise ValueError(
        f"No local topology path connects wires {source} and {target}."
    )


def _assert_local_topology(circuit, edges: Iterable[Sequence[int]]) -> None:
    """Reject any synthesized or fallback gate outside its local topology."""
    allowed = _normalize_edges(edges)
    for gate in circuit.get_Flat_Circuit().get_Gates():
        qubits = tuple(sorted(map(int, gate.get_Involved_Qbits())))
        if len(qubits) <= 1:
            continue
        if len(qubits) != 2 or frozenset(qubits) not in allowed:
            raise AssertionError(
                "Routing alternative violates its local topology: "
                f"{type(gate).__name__} on {qubits}, edges="
                f"{tuple(sorted(tuple(sorted(edge)) for edge in allowed))}."
            )


def _route_source_circuit_on_local_topology(circuit, parameters, edges, width):
    """Implement a source circuit using only local edges and restored SWAPs.

    A nonadjacent two-qubit gate is conjugated by a shortest-path SWAP chain.
    Undoing that chain preserves the source circuit's wire assignment, making
    this a safe boundary-permutation fallback rather than a mapping heuristic.
    """
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

    routed = Circuit(int(width))
    allowed = _normalize_edges(edges)
    for gate in circuit.get_Flat_Circuit().get_Gates():
        qubits = tuple(sorted(map(int, gate.get_Involved_Qbits())))
        if len(qubits) <= 1 or (
            len(qubits) == 2 and frozenset(qubits) in allowed
        ):
            routed.add_Gate(gate)
            continue
        if len(qubits) != 2:
            raise ValueError(
                "Exact-routing fallback supports primitive one- and two-qubit "
                f"gates, got {type(gate).__name__} on {qubits}."
            )
        left, right = qubits
        path = _shortest_topology_path(left, right, edges)
        movement = tuple(zip(path[:-2], path[1:-1]))
        for edge in movement:
            routed.add_SWAP(list(edge))
        remapping = {wire: wire for wire in range(int(width))}
        remapping[left] = path[-2]
        gate_circuit = Circuit(int(width))
        gate_circuit.add_Gate(gate)
        routed.add_Circuit(gate_circuit.Remap_Qbits(remapping, int(width)))
        for edge in reversed(movement):
            routed.add_SWAP(list(edge))
    routed = routed.get_Flat_Circuit()
    _assert_local_topology(routed, edges)
    error = _process_infidelity(
        circuit.get_Flat_Circuit().get_Matrix(parameters, is_f32=False),
        routed.get_Matrix(parameters, is_f32=False),
    )
    if error >= 1e-12:
        raise AssertionError(
            "Topology-routed source fallback changed its unitary: "
            f"infidelity={error:.3e}."
        )
    return routed


def symmetry_reduced_permutation_pairs(
    edges: Iterable[Sequence[int]],
    width: int,
    *,
    input_permutations: Iterable[Sequence[int]] | None = None,
    output_permutations: Iterable[Sequence[int]] | None = None,
) -> tuple[tuple[Permutation, Permutation], ...]:
    """Return representatives modulo simultaneous topology automorphisms.

    Permutations use BQSKit/EAPP's ordering convention.  Relabeling physical
    topology wires by an automorphism ``a`` maps a pair ``(pi, po)`` to
    ``(a[pi], a[po])``.  This action is free, giving 18 representatives for a
    three-vertex path and 6 for a triangle when both sides range over ``S3``.
    """
    width = int(width)
    all_permutations = tuple(itertools.permutations(range(width)))
    inputs = tuple(
        tuple(int(value) for value in permutation)
        for permutation in (
            all_permutations if input_permutations is None else input_permutations
        )
    )
    outputs = tuple(
        tuple(int(value) for value in permutation)
        for permutation in (
            all_permutations if output_permutations is None else output_permutations
        )
    )
    expected = tuple(range(width))
    for permutation in inputs + outputs:
        if tuple(sorted(permutation)) != expected:
            raise ValueError(f"Invalid width-{width} permutation: {permutation}.")

    automorphisms = topology_automorphisms(edges, width)
    allowed_pairs = set(itertools.product(inputs, outputs))
    representatives = []
    while allowed_pairs:
        representative = min(allowed_pairs)
        orbit = {
            (
                tuple(automorphism[value] for value in representative[0]),
                tuple(automorphism[value] for value in representative[1]),
            )
            for automorphism in automorphisms
        }
        allowed_pairs.difference_update(orbit)
        representatives.append(representative)
    return tuple(representatives)


def symmetry_reduced_assignment_orbits(
    edges: Iterable[Sequence[int]],
    width: int,
) -> tuple[
    tuple[
        tuple[Permutation, Permutation],
        tuple[tuple[tuple[Permutation, Permutation], Permutation], ...],
    ],
    ...,
]:
    """Return physical-wire assignment-pair orbits and relabeling witnesses.

    An assignment tuple stores the logical wire occupying each physical wire.
    Relabeling physical wire ``old`` to ``a[old]`` transforms an assignment by
    ``new[a[old]] = old_assignment[old]``.  Each orbit member includes the
    automorphism needed to derive its circuit from the representative circuit.
    """
    width = int(width)
    permutations = tuple(itertools.permutations(range(width)))
    automorphisms = topology_automorphisms(edges, width)

    def relabel(assignment: Permutation, automorphism: Permutation) -> Permutation:
        transformed = [0] * width
        for old, logical in enumerate(assignment):
            transformed[automorphism[old]] = logical
        return tuple(transformed)

    unseen = set(itertools.product(permutations, permutations))
    result = []
    while unseen:
        representative = min(unseen)
        members = {}
        for automorphism in automorphisms:
            member = (
                relabel(representative[0], automorphism),
                relabel(representative[1], automorphism),
            )
            members.setdefault(member, automorphism)
        unseen.difference_update(members)
        result.append(
            (
                representative,
                tuple(sorted(members.items())),
            )
        )
    return tuple(result)


def assignment_matrix(assignment: Sequence[int]) -> np.ndarray:
    """Return Squander's LSB-first basis permutation for a wire assignment.

    Squander numbers qubit zero at the least-significant tensor factor.  Using
    the opposite convention silently maps a topology automorphism to its
    bit-reversed permutation, which is especially easy to miss on a symmetric
    three-qubit path.
    """
    assignment = tuple(int(value) for value in assignment)
    width = len(assignment)
    if tuple(sorted(assignment)) != tuple(range(width)):
        raise ValueError(f"Invalid wire assignment {assignment}.")
    dimension = 1 << width
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for logical_index in range(dimension):
        logical_bits = [
            (logical_index >> qubit) & 1
            for qubit in range(width)
        ]
        physical_index = 0
        for physical_wire in range(width):
            physical_index |= (
                logical_bits[assignment[physical_wire]] << physical_wire
            )
        matrix[physical_index, logical_index] = 1.0
    return matrix


def permuted_partition_target(
    unitary: np.ndarray,
    input_assignment: Sequence[int],
    output_assignment: Sequence[int],
) -> np.ndarray:
    """Express a logical unitary between chosen input/output wire assignments."""
    unitary = np.asarray(unitary, dtype=np.complex128)
    input_matrix = assignment_matrix(input_assignment)
    output_matrix = assignment_matrix(output_assignment)
    if unitary.shape != input_matrix.shape:
        raise ValueError("Partition unitary and assignment widths do not match.")
    return output_matrix @ unitary @ input_matrix.T


def _process_infidelity(left: np.ndarray, right: np.ndarray) -> float:
    """Return Squander's global-phase-insensitive ``1 - F**2`` cost."""
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise ValueError("Unitary matrices must have the same square shape.")
    overlap = abs(np.vdot(left, right)) / left.shape[0]
    return float(1.0 - overlap * overlap)


def cnot_schmidt_lower_bound(
    unitary: np.ndarray,
    topology: Iterable[Sequence[int]],
    *,
    rank_tolerance: float = 1e-8,
) -> int:
    """Return a topology-aware lower bound on the required CNOT count.

    Across a one-qubit-versus-rest operator cut, one incident CNOT can at most
    double operator Schmidt rank.  The resulting per-vertex requirements form
    a tiny integer edge-cover problem on a local partition topology.
    """
    unitary = np.asarray(unitary, dtype=np.complex128)
    width = int(round(np.log2(unitary.shape[0])))
    if unitary.shape != (1 << width, 1 << width):
        raise ValueError("Unitary dimension is not a power-of-two square.")
    if width <= 1:
        return 0
    tensor = unitary.reshape((2,) * (2 * width))
    requirements = []
    for qubit in range(width):
        output_axis = width - 1 - qubit
        input_axis = 2 * width - 1 - qubit
        remaining = [
            axis
            for axis in range(2 * width)
            if axis not in (output_axis, input_axis)
        ]
        cut_matrix = np.transpose(
            tensor, (output_axis, input_axis, *remaining)
        ).reshape(4, -1)
        singular_values = np.linalg.svd(cut_matrix, compute_uv=False)
        threshold = max(float(rank_tolerance), np.finfo(float).eps) * max(
            1.0, float(singular_values[0])
        )
        rank = max(1, int(np.count_nonzero(singular_values > threshold)))
        requirements.append(int(np.ceil(np.log2(rank))))

    edges = tuple(
        sorted(tuple(sorted(tuple(edge))) for edge in _normalize_edges(topology))
    )
    if not edges:
        if any(requirements):
            raise ValueError(
                "Disconnected topology cannot meet Schmidt requirements."
            )
        return 0

    def compositions(total, count, prefix=()):
        if count == 1:
            yield prefix + (total,)
            return
        for value in range(total + 1):
            yield from compositions(total - value, count - 1, prefix + (value,))

    for total in range(sum(requirements) + 1):
        for multiplicities in compositions(total, len(edges)):
            degrees = [0] * width
            for (left, right), count in zip(edges, multiplicities):
                degrees[left] += count
                degrees[right] += count
            if all(
                degree >= requirement
                for degree, requirement in zip(degrees, requirements)
            ):
                return total
    raise AssertionError("Failed to solve local Schmidt edge-cover bound.")


def _canonical_labeled_topology(
    physical_vertices: Sequence[int],
    device_edges: frozenset[frozenset[int]],
) -> tuple[tuple[Edge, ...], tuple[int, ...]] | None:
    """Canonicalize one induced connected physical subtopology."""
    physical_vertices = tuple(int(value) for value in physical_vertices)
    width = len(physical_vertices)
    local_edges = {
        frozenset((left, right))
        for left in range(width)
        for right in range(left + 1, width)
        if frozenset((physical_vertices[left], physical_vertices[right]))
        in device_edges
    }
    if width > 1:
        seen = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for edge in local_edges:
                if current not in edge:
                    continue
                neighbour = next(iter(edge - {current}))
                if neighbour not in seen:
                    seen.add(neighbour)
                    frontier.append(neighbour)
        if len(seen) != width:
            return None

    best = None
    for old_to_new in itertools.permutations(range(width)):
        relabeled_edges = tuple(
            sorted(
                tuple(sorted((old_to_new[u], old_to_new[v])))
                for u, v in (tuple(edge) for edge in local_edges)
            )
        )
        canonical_vertices = [0] * width
        for old, new in enumerate(old_to_new):
            canonical_vertices[new] = physical_vertices[old]
        candidate = (relabeled_edges, tuple(canonical_vertices))
        if best is None or candidate < best:
            best = candidate
    return best


def connected_subtopology_embeddings(
    topology: Iterable[Sequence[int]],
    physical_qubit_count: int,
    width: int,
) -> dict[tuple[Edge, ...], tuple[tuple[int, ...], ...]]:
    """Group connected induced physical embeddings by canonical local graph."""
    device_edges = _normalize_edges(topology)
    grouped: dict[tuple[Edge, ...], set[tuple[int, ...]]] = {}
    for vertices in itertools.combinations(range(int(physical_qubit_count)), width):
        canonical = _canonical_labeled_topology(vertices, device_edges)
        if canonical is not None:
            edges, embedding = canonical
            grouped.setdefault(edges, set()).add(embedding)
    return {
        edges: tuple(sorted(embeddings))
        for edges, embeddings in sorted(grouped.items())
    }


@dataclass(frozen=True)
class SynthesizedRoutingPayload:
    """Local Squander circuit attached to a routing alternative."""

    circuit: Any
    parameters: np.ndarray
    topology: tuple[Edge, ...]
    input_assignment: Permutation
    output_assignment: Permutation
    source_circuit: Any = None
    source_parameters: Any = None
    certificate_kind: str = "unitary"


def synthesize_partition_alternatives(
    *,
    partition: int,
    unitary: np.ndarray,
    logical_qubits: Sequence[int],
    topology: Iterable[Sequence[int]],
    physical_qubit_count: int,
    config: Mapping[str, Any],
    source_circuit=None,
    source_parameters=None,
    requested_transitions=None,
    synthesis_cache=None,
    synthesis_cache_stats=None,
    synthesis_batch=None,
) -> tuple[RoutingAlternative, ...] | _DeferredRoutingAlternatives:
    """Synthesize every symmetry-distinct local routing alternative.

    OSR is invoked only for orbit representatives.  Circuits for the remaining
    members are obtained by topology-automorphism wire relabeling, while each
    physical embedding becomes a compatible global-routing alternative.
    """
    logical_qubits = tuple(int(value) for value in logical_qubits)
    width = len(logical_qubits)
    if width < 1 or len(set(logical_qubits)) != width:
        raise ValueError("A partition must contain distinct logical qubits.")
    if width > int(config.get("max_partition_size", width)):
        raise ValueError("Partition exceeds the configured maximum width.")
    unitary = np.asarray(unitary, dtype=np.complex128)
    dimension = 1 << width
    if unitary.shape != (dimension, dimension):
        raise ValueError("Partition unitary shape does not match its width.")

    # Lazy imports avoid a module cycle during normal partitioning imports.
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        CNOTGateCount,
        SingleQubitGateCount,
        _synthesis_acceptance_tolerance,
    )
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

    embeddings_by_graph = connected_subtopology_embeddings(
        topology, physical_qubit_count, width
    )
    alternatives_by_transition = {}
    synthesis_config = dict(config)
    synthesis_config["topology"] = None
    validation_tolerance = float(
        _synthesis_acceptance_tolerance(synthesis_config)
    )
    assignments = tuple(itertools.permutations(range(width)))
    requested_transitions = (
        None
        if requested_transitions is None
        else {
            (tuple(input_assignment), tuple(output_assignment))
            for input_assignment, output_assignment in requested_transitions
        }
    )
    targets = {
        (input_assignment, output_assignment): permuted_partition_target(
            unitary, input_assignment, output_assignment
        )
        for input_assignment in assignments
        for output_assignment in assignments
    }
    for local_edges, embeddings in embeddings_by_graph.items():
        def record_local_circuit(
            local_circuit,
            local_parameters,
            matching_transitions,
        ):
            _assert_local_topology(local_circuit, local_edges)
            cnot_count = CNOTGateCount(local_circuit, 0)
            single_qubit_count = SingleQubitGateCount(local_circuit)
            for input_assignment, output_assignment in matching_transitions:
                payload = SynthesizedRoutingPayload(
                    circuit=local_circuit,
                    parameters=local_parameters,
                    topology=tuple(local_edges),
                    input_assignment=input_assignment,
                    output_assignment=output_assignment,
                    source_circuit=source_circuit,
                    source_parameters=(
                        None
                        if source_parameters is None
                        else np.asarray(source_parameters, dtype=np.float64)
                    ),
                )
                for embedding in embeddings:
                    input_physical = [0] * width
                    output_physical = [0] * width
                    for physical_wire, logical_wire in enumerate(input_assignment):
                        input_physical[logical_wire] = embedding[physical_wire]
                    for physical_wire, logical_wire in enumerate(output_assignment):
                        output_physical[logical_wire] = embedding[physical_wire]
                    alternative = RoutingAlternative(
                        partition=int(partition),
                        logical_qubits=logical_qubits,
                        input_physical=tuple(input_physical),
                        output_physical=tuple(output_physical),
                        cnot_count=cnot_count,
                        single_qubit_count=single_qubit_count,
                        payload=payload,
                    )
                    key = (
                        alternative.input_physical,
                        alternative.output_physical,
                    )
                    previous = alternatives_by_transition.get(key)
                    if previous is None or (
                        alternative.cnot_count,
                        alternative.single_qubit_count,
                    ) < (
                        previous.cnot_count,
                        previous.single_qubit_count,
                    ):
                        alternatives_by_transition[key] = alternative

        fallback_costs = {}
        if source_circuit is not None and source_parameters is not None:
            source_flat = source_circuit.get_Flat_Circuit()
            source_parameters_array = np.asarray(
                source_parameters, dtype=np.float64
            )
            topology_valid_source = _route_source_circuit_on_local_topology(
                source_flat,
                source_parameters_array,
                local_edges,
                width,
            )
            for input_assignment, output_assignment in targets:
                fallback = Circuit(width)
                for edge in _permutation_swaps(
                    _inverse_permutation(input_assignment), local_edges
                ):
                    fallback.add_SWAP(list(edge))
                fallback.add_Circuit(topology_valid_source)
                for edge in _permutation_swaps(output_assignment, local_edges):
                    fallback.add_SWAP(list(edge))
                fallback = fallback.get_Flat_Circuit()
                error = _process_infidelity(
                    fallback.get_Matrix(source_parameters_array, is_f32=False),
                    targets[(input_assignment, output_assignment)],
                )
                if error >= validation_tolerance:
                    raise AssertionError(
                        "Topology-valid routing fallback has incorrect unitary: "
                        f"infidelity={error:.3e}."
                    )
                transition = (input_assignment, output_assignment)
                fallback_costs[transition] = CNOTGateCount(fallback, 0)
                record_local_circuit(
                    fallback,
                    source_parameters_array,
                    (transition,),
                )

        if not config.get("exact_routing_eager_synthesis", True):
            continue

        all_assignment_orbits = symmetry_reduced_assignment_orbits(
            local_edges, width
        )
        assignment_orbits = (
            all_assignment_orbits
            if requested_transitions is None
            else tuple(
                (representative, orbit)
                for representative, orbit in all_assignment_orbits
                if any(
                    declared_member in requested_transitions
                    for declared_member, _automorphism in orbit
                )
            )
        )
        representative_targets = tuple(
            permuted_partition_target(unitary, representative[0], representative[1])
            for representative, _orbit in assignment_orbits
        )
        representative_results = [None] * len(assignment_orbits)
        active_indices = []
        active_targets = []
        active_configs = []
        for index, ((representative, _orbit), target) in enumerate(
            zip(assignment_orbits, representative_targets)
        ):
            fallback_cost = fallback_costs.get(representative)
            if fallback_cost == 0:
                continue
            target_config = dict(synthesis_config)
            if fallback_cost is not None:
                # A synthesized routing alternative is useful only when it
                # strictly improves on the topology-valid naive fallback.
                # This bound is therefore intrinsic to the routing problem,
                # not a generic tree-search tuning cap.
                target_config["tree_level_max"] = fallback_cost - 1
            active_indices.append(index)
            active_targets.append(target)
            active_configs.append(target_config)
        def consume_results(
            active_results,
            *,
            active_indices=tuple(active_indices),
            assignment_orbits=assignment_orbits,
            representative_results=representative_results,
            record_local_circuit=record_local_circuit,
            targets=targets,
            width=width,
            validation_tolerance=validation_tolerance,
        ):
            for index, result in zip(active_indices, active_results):
                representative_results[index] = result
            for (_representative, orbit), synthesis_result in zip(
                assignment_orbits, representative_results
            ):
                if synthesis_result is None:
                    continue
                representative_circuit = synthesis_result.circuit
                representative_parameters = synthesis_result.parameters
                for _declared_member, automorphism in orbit:
                    relabeling = {
                        old: automorphism[old] for old in range(width)
                    }
                    local_circuit = representative_circuit.Remap_Qbits(
                        relabeling, width
                    )
                    # Identify the exact transition represented by a
                    # symmetry-derived circuit from its small matrix rather
                    # than relying on a fragile endian convention.
                    local_matrix = local_circuit.get_Matrix(
                        representative_parameters, is_f32=False
                    )
                    matching_transitions = tuple(
                        transition
                        for transition, expected in targets.items()
                        if _process_infidelity(local_matrix, expected)
                        < validation_tolerance
                    )
                    if not matching_transitions:
                        error = min(
                            _process_infidelity(local_matrix, expected)
                            for expected in targets.values()
                        )
                        raise AssertionError(
                            "Symmetry-derived routing circuit does not implement "
                            "any input/output assignment transition: "
                            f"infidelity={error:.3e}."
                        )
                    record_local_circuit(
                        local_circuit,
                        representative_parameters,
                        matching_transitions,
                    )

        if synthesis_batch is None:
            consume_results(
                _call_shared_synthesis_batch_cached(
                    active_targets,
                    synthesis_config,
                    list(local_edges),
                    target_configs=active_configs,
                    cache=synthesis_cache,
                    cache_stats=synthesis_cache_stats,
                )
            )
        else:
            synthesis_batch.enqueue(
                local_edges,
                active_targets,
                active_configs,
                consume_results,
            )

        # Every requested transition was priced through its canonical topology
        # automorphism orbit above. A rejected representative keeps all of its
        # topology-valid fallbacks; there is deliberately no direct retry.
    if synthesis_batch is not None:
        return _DeferredRoutingAlternatives(alternatives_by_transition)
    return _sorted_routing_alternatives(alternatives_by_transition)


# Compatibility name retained for callers of the initial three-qubit API.
synthesize_three_qubit_alternatives = synthesize_partition_alternatives


def single_qubit_passthrough_alternatives(
    *,
    partition: int,
    circuit,
    parameters: np.ndarray,
    logical_qubit: int,
    physical_qubit_count: int,
) -> tuple[RoutingAlternative, ...]:
    """Place a one-qubit chain without invoking OSR or changing its gates."""
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        CNOTGateCount,
        SingleQubitGateCount,
    )

    parameters = np.asarray(parameters, dtype=np.float64)
    payload = SynthesizedRoutingPayload(
        circuit=circuit,
        parameters=parameters,
        topology=(),
        input_assignment=(0,),
        output_assignment=(0,),
        source_circuit=circuit,
        source_parameters=parameters,
    )
    return tuple(
        RoutingAlternative(
            partition=int(partition),
            logical_qubits=(int(logical_qubit),),
            input_physical=(physical,),
            output_physical=(physical,),
            cnot_count=CNOTGateCount(circuit, 0),
            single_qubit_count=SingleQubitGateCount(circuit),
            payload=payload,
        )
        for physical in range(int(physical_qubit_count))
    )


def two_qubit_passthrough_alternatives(
    *,
    partition: int,
    circuit,
    parameters: np.ndarray,
    logical_qubits: Sequence[int],
    topology: Iterable[Sequence[int]],
    physical_qubit_count: int,
) -> tuple[RoutingAlternative, ...]:
    """Realize a two-qubit block with exact boundary SWAP permutations."""
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        CNOTGateCount,
        SingleQubitGateCount,
    )
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

    logical_qubits = tuple(int(value) for value in logical_qubits)
    if len(logical_qubits) != 2:
        raise ValueError("Expected a two-qubit passthrough block.")
    parameters = np.asarray(parameters, dtype=np.float64)
    assignments = tuple(itertools.permutations(range(2)))
    alternatives_by_transition = {}
    for local_edges, embeddings in connected_subtopology_embeddings(
        topology, physical_qubit_count, 2
    ).items():
        for input_assignment, output_assignment in itertools.product(
            assignments, repeat=2
        ):
            local_circuit = Circuit(2)
            if input_assignment != (0, 1):
                local_circuit.add_SWAP([0, 1])
            local_circuit.add_Circuit(circuit)
            if output_assignment != (0, 1):
                local_circuit.add_SWAP([0, 1])
            local_circuit = local_circuit.get_Flat_Circuit()
            payload = SynthesizedRoutingPayload(
                circuit=local_circuit,
                parameters=parameters,
                topology=tuple(local_edges),
                input_assignment=input_assignment,
                output_assignment=output_assignment,
                source_circuit=circuit,
                source_parameters=parameters,
            )
            for embedding in embeddings:
                input_physical = [0, 0]
                output_physical = [0, 0]
                for physical_wire, logical_wire in enumerate(input_assignment):
                    input_physical[logical_wire] = embedding[physical_wire]
                for physical_wire, logical_wire in enumerate(output_assignment):
                    output_physical[logical_wire] = embedding[physical_wire]
                alternative = RoutingAlternative(
                    partition=int(partition),
                    logical_qubits=logical_qubits,
                    input_physical=tuple(input_physical),
                    output_physical=tuple(output_physical),
                    cnot_count=CNOTGateCount(local_circuit, 0),
                    single_qubit_count=SingleQubitGateCount(local_circuit),
                    payload=payload,
                )
                key = (
                    alternative.input_physical,
                    alternative.output_physical,
                )
                previous = alternatives_by_transition.get(key)
                if previous is None or (
                    alternative.cnot_count,
                    alternative.single_qubit_count,
                ) < (
                    previous.cnot_count,
                    previous.single_qubit_count,
                ):
                    alternatives_by_transition[key] = alternative
    return tuple(alternatives_by_transition.values())


@dataclass(frozen=True)
class RoutingAlternative:
    """One topology-valid realization of a gate partition.

    ``logical_qubits``, ``input_physical``, and ``output_physical`` are aligned.
    The alternative is executable when each logical qubit currently occupies
    its corresponding input physical qubit.  After execution, the same logical
    qubits occupy ``output_physical``.  Both physical tuples must contain the
    same vertices; output permutations therefore absorb local SWAPs without
    moving unrelated logical qubits.
    """

    partition: int
    logical_qubits: tuple[int, ...]
    input_physical: tuple[int, ...]
    output_physical: tuple[int, ...]
    cnot_count: int
    single_qubit_count: int = 0
    payload: Any = None

    def __post_init__(self) -> None:
        width = len(self.logical_qubits)
        if width == 0:
            raise ValueError("A routing alternative must touch at least one qubit.")
        if len(set(self.logical_qubits)) != width:
            raise ValueError("Routing alternative logical qubits are not unique.")
        if len(self.input_physical) != width or len(self.output_physical) != width:
            raise ValueError("Routing alternative placement width mismatch.")
        if len(set(self.input_physical)) != width:
            raise ValueError("Routing alternative input placement is not injective.")
        if set(self.input_physical) != set(self.output_physical):
            raise ValueError(
                "Routing alternative output must permute its input vertices."
            )
        if self.cnot_count < 0 or self.single_qubit_count < 0:
            raise ValueError("Routing alternative gate counts must be nonnegative.")


@dataclass(frozen=True)
class RoutingSelection:
    """A selected partition alternative at one point in the exact schedule."""

    partition: int
    alternative: RoutingAlternative


@dataclass(frozen=True)
class ExactRoutingResult:
    """Minimum-cost exact-cover result over the supplied alternatives."""

    selections: tuple[RoutingSelection, ...]
    cnot_count: int
    single_qubit_count: int
    initial_mapping: tuple[int, ...]
    final_mapping: tuple[int, ...]
    explored_states: int
    master_backend: str = "unknown"
    optimal: bool = True


@dataclass(frozen=True)
class ExactCircuitRoutingResult:
    """Materialized circuit plus its replayable combinatorial certificate."""

    circuit: Any
    parameters: np.ndarray
    solution: ExactRoutingResult
    candidate_gate_sets: tuple[tuple[int, ...], ...]
    candidate_gate_orders: tuple[tuple[int, ...], ...]
    lazy_rounds: int = 0
    synthesized_partitions: int = 0
    synthesis_cache_hits: int = 0
    timed_out: bool = False


class ExactRoutingLimitExceeded(RuntimeError):
    """Raised when an explicitly configured exact-search limit is reached."""


def _gate_mask(gates: Iterable[int]) -> int:
    mask = 0
    for gate in gates:
        gate = int(gate)
        if gate < 0:
            raise ValueError("Gate indices must be nonnegative.")
        mask |= 1 << gate
    return mask


def solve_exact_routing_branch_and_bound(
    *,
    gate_predecessors: Mapping[int, Iterable[int]],
    partitions: Sequence[Iterable[int]],
    alternatives: Mapping[int, Sequence[RoutingAlternative]],
    logical_qubit_count: int,
    gate_qubits: Mapping[int, Iterable[int]] | None = None,
    physical_qubit_count: int | None = None,
    initial_mapping: Sequence[int] | None = None,
    timeout_seconds: float | None = None,
    max_states: int | None = None,
) -> ExactRoutingResult:
    """Solve routing with the original dependency-ready branch-and-bound.

    This solver is exponential by design but avoids enumerating all global
    initial mappings.  With ``initial_mapping=None``, logical qubits are assigned
    lazily on first use.  Any injective partial mapping can be extended to a full
    permutation, so untouched qubits do not contribute a factorial upfront cost.
    """
    logical_qubit_count = int(logical_qubit_count)
    physical_qubit_count = int(
        logical_qubit_count
        if physical_qubit_count is None
        else physical_qubit_count
    )
    if logical_qubit_count < 1 or physical_qubit_count < logical_qubit_count:
        raise ValueError("Physical width must cover every logical qubit.")

    gate_indices = sorted(int(gate) for gate in gate_predecessors)
    if gate_indices != list(range(len(gate_indices))):
        raise ValueError("Gate indices must be contiguous from zero.")
    gate_count = len(gate_indices)
    all_gates_mask = (1 << gate_count) - 1
    predecessor_masks = tuple(
        _gate_mask(gate_predecessors[gate]) for gate in gate_indices
    )

    partition_masks = tuple(_gate_mask(partition) for partition in partitions)
    if any(mask == 0 for mask in partition_masks):
        raise ValueError("Empty routing partitions are not allowed.")
    for partition, partition_alternatives in alternatives.items():
        if partition < 0 or partition >= len(partitions):
            raise ValueError(f"Unknown partition alternative key {partition}.")
        if any(value.partition != partition for value in partition_alternatives):
            raise ValueError("Routing alternative partition id mismatch.")
    gate_to_partitions = [[] for _ in gate_indices]
    for partition_index, mask in enumerate(partition_masks):
        for gate in gate_indices:
            if mask & (1 << gate):
                gate_to_partitions[gate].append(partition_index)
    if any(not values for values in gate_to_partitions):
        raise ValueError("Every gate must belong to at least one partition.")

    if initial_mapping is None:
        start_mapping = (-1,) * logical_qubit_count
    else:
        start_mapping = tuple(int(value) for value in initial_mapping)
        if len(start_mapping) != logical_qubit_count:
            raise ValueError("Initial mapping width mismatch.")
        assigned = [value for value in start_mapping if value >= 0]
        if (
            len(set(assigned)) != len(assigned)
            or any(value >= physical_qubit_count for value in assigned)
        ):
            raise ValueError("Initial mapping is not an injective partial mapping.")

    executable_external_masks = []
    for partition_mask in partition_masks:
        external = 0
        for gate in gate_indices:
            if partition_mask & (1 << gate):
                external |= predecessor_masks[gate] & ~partition_mask
        executable_external_masks.append(external)

    started = time.monotonic()
    explored_states = 0
    memo: dict[
        tuple[int, tuple[int, ...]],
        tuple[tuple[int, int], tuple[RoutingSelection, ...]] | None,
    ] = {}

    def check_limits() -> None:
        nonlocal explored_states
        explored_states += 1
        if max_states is not None and explored_states > int(max_states):
            raise ExactRoutingLimitExceeded(
                f"Exact routing exceeded {max_states} states."
            )
        if (
            timeout_seconds is not None
            and time.monotonic() - started > float(timeout_seconds)
        ):
            raise ExactRoutingLimitExceeded(
                f"Exact routing exceeded {timeout_seconds} seconds."
            )

    def apply_alternative(
        mapping: tuple[int, ...], alternative: RoutingAlternative
    ) -> tuple[int, ...] | None:
        if any(
            logical < 0 or logical >= logical_qubit_count
            for logical in alternative.logical_qubits
        ):
            raise ValueError("Alternative references an unknown logical qubit.")
        if any(
            physical < 0 or physical >= physical_qubit_count
            for physical in alternative.input_physical
            + alternative.output_physical
        ):
            raise ValueError("Alternative references an unknown physical qubit.")

        involved = set(alternative.logical_qubits)
        occupied_outside = {
            physical
            for logical, physical in enumerate(mapping)
            if logical not in involved and physical >= 0
        }
        if occupied_outside & set(alternative.input_physical):
            return None
        for logical, physical in zip(
            alternative.logical_qubits, alternative.input_physical
        ):
            if mapping[logical] >= 0 and mapping[logical] != physical:
                return None

        updated = list(mapping)
        for logical, physical in zip(
            alternative.logical_qubits, alternative.output_physical
        ):
            updated[logical] = physical
        assigned = [physical for physical in updated if physical >= 0]
        if len(set(assigned)) != len(assigned):
            return None
        return tuple(updated)

    def search(
        consumed: int, mapping: tuple[int, ...]
    ) -> tuple[tuple[int, int], tuple[RoutingSelection, ...]] | None:
        key = (consumed, mapping)
        if key in memo:
            return memo[key]
        check_limits()
        if consumed == all_gates_mask:
            result = ((0, 0), ())
            memo[key] = result
            return result

        best = None
        for partition_index, partition_mask in enumerate(partition_masks):
            if partition_mask & consumed:
                continue
            if executable_external_masks[partition_index] & ~consumed:
                continue
            for alternative in alternatives.get(partition_index, ()):
                next_mapping = apply_alternative(mapping, alternative)
                if next_mapping is None:
                    continue
                suffix = search(consumed | partition_mask, next_mapping)
                if suffix is None:
                    continue
                cost = (
                    alternative.cnot_count + suffix[0][0],
                    alternative.single_qubit_count + suffix[0][1],
                )
                selections = (
                    RoutingSelection(partition_index, alternative),
                ) + suffix[1]
                candidate = (cost, selections)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        memo[key] = best
        return best

    optimum = search(0, start_mapping)
    if optimum is None:
        raise ValueError("No compatible exact-cover routing solution exists.")

    discovered_initial = list(start_mapping)
    current = tuple(start_mapping)
    for selection in optimum[1]:
        alternative = selection.alternative
        for logical, physical in zip(
            alternative.logical_qubits, alternative.input_physical
        ):
            if discovered_initial[logical] < 0:
                discovered_initial[logical] = physical
        next_mapping = apply_alternative(current, alternative)
        if next_mapping is None:
            raise AssertionError("Stored exact-routing solution no longer replays.")
        current = next_mapping
    unused_physical = iter(
        sorted(set(range(physical_qubit_count)) - set(discovered_initial))
    )
    for logical, physical in enumerate(discovered_initial):
        if physical < 0:
            discovered_initial[logical] = next(unused_physical)
    final_mapping = list(current)
    for logical, physical in enumerate(final_mapping):
        if physical < 0:
            final_mapping[logical] = discovered_initial[logical]

    return ExactRoutingResult(
        selections=optimum[1],
        cnot_count=optimum[0][0],
        single_qubit_count=optimum[0][1],
        initial_mapping=tuple(discovered_initial),
        final_mapping=tuple(final_mapping),
        explored_states=explored_states,
        master_backend="branch-and-bound",
    )


def _routing_gate_ancestors(
    gate_predecessors: Mapping[int, Iterable[int]],
) -> tuple[frozenset[int], ...]:
    """Return transitive predecessor sets, rejecting a malformed cyclic DAG."""
    count = len(gate_predecessors)
    visiting = set()
    completed = {}

    def visit(gate):
        if gate in completed:
            return completed[gate]
        if gate in visiting:
            raise ValueError("Routing gate dependency graph contains a cycle.")
        visiting.add(gate)
        result = set()
        for predecessor in gate_predecessors[gate]:
            predecessor = int(predecessor)
            if predecessor < 0 or predecessor >= count:
                raise ValueError("Routing dependency references an unknown gate.")
            result.add(predecessor)
            result.update(visit(predecessor))
        visiting.remove(gate)
        completed[gate] = frozenset(result)
        return completed[gate]

    return tuple(visit(gate) for gate in range(count))


def solve_exact_routing_ilp_dense(
    *,
    gate_predecessors: Mapping[int, Iterable[int]],
    partitions: Sequence[Iterable[int]],
    alternatives: Mapping[int, Sequence[RoutingAlternative]],
    logical_qubit_count: int,
    gate_qubits: Mapping[int, Iterable[int]] | None = None,
    physical_qubit_count: int | None = None,
    initial_mapping: Sequence[int] | None = None,
    timeout_seconds: float | None = None,
    max_states: int | None = None,
) -> ExactRoutingResult:
    """Solve exact cover with an explicitly materialized mapping-flow MILP.

    Each synthesized partition configuration is a node. For every logical
    qubit, selected nodes form one source-to-sink path in circuit order, and
    an arc exists only when the predecessor's output physical location equals
    the successor's input location. Initial and final placement variables are
    injective assignments. Continuous order potentials exclude cycles in the
    quotient partition graph; they are not schedule or time indices. This
    reference formulation is useful for validation, but the sparse cut-based
    formulation below is the operational ILP backend.
    """
    if max_states is not None:
        raise ValueError(
            "exact_routing_max_states applies only to the branch-and-bound "
            "master; use exact_routing_timeout_seconds for the ILP master."
        )

    import pulp
    from squander.partitioning.ilp import _solve_pulp_with_gurobi_or_cbc

    logical_qubit_count = int(logical_qubit_count)
    physical_qubit_count = int(
        logical_qubit_count
        if physical_qubit_count is None
        else physical_qubit_count
    )
    if logical_qubit_count < 1 or physical_qubit_count < logical_qubit_count:
        raise ValueError("Physical width must cover every logical qubit.")

    gate_indices = sorted(int(gate) for gate in gate_predecessors)
    if gate_indices != list(range(len(gate_indices))):
        raise ValueError("Gate indices must be contiguous from zero.")
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    if any(not part for part in partition_sets):
        raise ValueError("Empty routing partitions are not allowed.")
    if any(
        gate < 0 or gate >= len(gate_indices)
        for part in partition_sets
        for gate in part
    ):
        raise ValueError("Routing partition references an unknown gate.")

    nodes = []
    nodes_by_partition = [[] for _ in partition_sets]
    for partition, values in sorted(alternatives.items()):
        partition = int(partition)
        if partition < 0 or partition >= len(partition_sets):
            raise ValueError(f"Unknown partition alternative key {partition}.")
        for alternative in values:
            if alternative.partition != partition:
                raise ValueError("Routing alternative partition id mismatch.")
            if any(
                logical < 0 or logical >= logical_qubit_count
                for logical in alternative.logical_qubits
            ):
                raise ValueError("Alternative references an unknown logical qubit.")
            if any(
                physical < 0 or physical >= physical_qubit_count
                for physical in (
                    alternative.input_physical + alternative.output_physical
                )
            ):
                raise ValueError("Alternative references an unknown physical qubit.")
            node = len(nodes)
            nodes.append(alternative)
            nodes_by_partition[partition].append(node)
    if not nodes:
        raise ValueError("No routing alternatives were supplied.")

    gate_to_partitions = [[] for _ in gate_indices]
    for partition, gate_set in enumerate(partition_sets):
        for gate in gate_set:
            gate_to_partitions[gate].append(partition)
    if any(not covering for covering in gate_to_partitions):
        raise ValueError("Every gate must belong to at least one partition.")

    ancestors = _routing_gate_ancestors(gate_predecessors)
    partition_before = set()
    for left, left_gates in enumerate(partition_sets):
        for right, right_gates in enumerate(partition_sets):
            if left == right:
                continue
            if any(
                left_gate in ancestors[right_gate]
                for left_gate in left_gates
                for right_gate in right_gates
            ):
                partition_before.add((left, right))

    prob = pulp.LpProblem("ExactPermutationAwareRouting", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("cfg", range(len(nodes)), cat="Binary")
    selected_partition = {
        partition: pulp.lpSum(x[node] for node in node_ids)
        for partition, node_ids in enumerate(nodes_by_partition)
    }

    for gate in gate_indices:
        prob += (
            pulp.lpSum(
                x[node]
                for partition in gate_to_partitions[gate]
                for node in nodes_by_partition[partition]
            )
            == 1,
            f"cover_gate_{gate}",
        )

    # A selected partition has exactly one synthesized configuration. The
    # gate-cover constraints imply this for nonempty partitions; retaining the
    # explicit upper bound makes that invariant visible to the model.
    for partition, expression in selected_partition.items():
        prob += expression <= 1, f"one_configuration_{partition}"

    order_bound = max(1, len(partition_sets))
    order = pulp.LpVariable.dicts(
        "partition_order",
        range(len(partition_sets)),
        lowBound=0,
        upBound=order_bound - 1,
        cat="Continuous",
    )
    for left, right in sorted(partition_before):
        prob += (
            order[left] + 1
            <= order[right]
            + order_bound
            * (2 - selected_partition[left] - selected_partition[right]),
            f"partition_precedence_{left}_{right}",
        )

    node_qubit_index = {
        (node, logical): alternative.logical_qubits.index(logical)
        for node, alternative in enumerate(nodes)
        for logical in alternative.logical_qubits
    }
    nodes_on_qubit = {
        logical: [
            node
            for node, alternative in enumerate(nodes)
            if logical in alternative.logical_qubits
        ]
        for logical in range(logical_qubit_count)
    }

    source = {}
    sink = {}
    arcs = {}
    incoming_arcs = collections.defaultdict(list)
    outgoing_arcs = collections.defaultdict(list)
    for logical, qubit_nodes in nodes_on_qubit.items():
        for node in qubit_nodes:
            source[logical, node] = pulp.LpVariable(
                f"source_q{logical}_n{node}", cat="Binary"
            )
            sink[logical, node] = pulp.LpVariable(
                f"sink_q{logical}_n{node}", cat="Binary"
            )
        for left in qubit_nodes:
            left_alternative = nodes[left]
            left_index = node_qubit_index[left, logical]
            output_physical = left_alternative.output_physical[left_index]
            for right in qubit_nodes:
                if left == right:
                    continue
                right_alternative = nodes[right]
                if (
                    (left_alternative.partition, right_alternative.partition)
                    not in partition_before
                ):
                    continue
                # Interleaving candidate partitions induce precedence in both
                # directions and cannot coexist in an acyclic exact cover.
                if (
                    (right_alternative.partition, left_alternative.partition)
                    in partition_before
                ):
                    continue
                right_index = node_qubit_index[right, logical]
                if output_physical != right_alternative.input_physical[right_index]:
                    continue
                variable = pulp.LpVariable(
                    f"flow_q{logical}_n{left}_n{right}", cat="Binary"
                )
                arcs[logical, left, right] = variable
                outgoing_arcs[logical, left].append(variable)
                incoming_arcs[logical, right].append(variable)

    initial = pulp.LpVariable.dicts(
        "initial_mapping",
        (range(logical_qubit_count), range(physical_qubit_count)),
        cat="Binary",
    )
    final = pulp.LpVariable.dicts(
        "final_mapping",
        (range(logical_qubit_count), range(physical_qubit_count)),
        cat="Binary",
    )
    idle = pulp.LpVariable.dicts(
        "idle_mapping",
        (range(logical_qubit_count), range(physical_qubit_count)),
        cat="Binary",
    )
    for logical in range(logical_qubit_count):
        prob += (
            pulp.lpSum(initial[logical][physical] for physical in range(physical_qubit_count))
            == 1,
            f"initial_logical_{logical}",
        )
        prob += (
            pulp.lpSum(final[logical][physical] for physical in range(physical_qubit_count))
            == 1,
            f"final_logical_{logical}",
        )
        for physical in range(physical_qubit_count):
            source_at_physical = [
                source[logical, node]
                for node in nodes_on_qubit[logical]
                if nodes[node].input_physical[node_qubit_index[node, logical]]
                == physical
            ]
            sink_at_physical = [
                sink[logical, node]
                for node in nodes_on_qubit[logical]
                if nodes[node].output_physical[node_qubit_index[node, logical]]
                == physical
            ]
            prob += (
                initial[logical][physical]
                == idle[logical][physical] + pulp.lpSum(source_at_physical),
                f"initial_source_q{logical}_p{physical}",
            )
            prob += (
                final[logical][physical]
                == idle[logical][physical] + pulp.lpSum(sink_at_physical),
                f"final_sink_q{logical}_p{physical}",
            )
    for physical in range(physical_qubit_count):
        prob += (
            pulp.lpSum(initial[logical][physical] for logical in range(logical_qubit_count))
            <= 1,
            f"initial_physical_{physical}",
        )
        prob += (
            pulp.lpSum(final[logical][physical] for logical in range(logical_qubit_count))
            <= 1,
            f"final_physical_{physical}",
        )

    if initial_mapping is not None:
        fixed_mapping = tuple(int(value) for value in initial_mapping)
        if len(fixed_mapping) != logical_qubit_count:
            raise ValueError("Initial mapping width mismatch.")
        assigned = [physical for physical in fixed_mapping if physical >= 0]
        if (
            len(set(assigned)) != len(assigned)
            or any(physical >= physical_qubit_count for physical in assigned)
        ):
            raise ValueError("Initial mapping is not an injective partial mapping.")
        for logical, physical in enumerate(fixed_mapping):
            if physical >= 0:
                prob += initial[logical][physical] == 1

    for logical, qubit_nodes in nodes_on_qubit.items():
        for node in qubit_nodes:
            prob += (
                source[logical, node]
                + pulp.lpSum(incoming_arcs[logical, node])
                == x[node],
                f"flow_in_q{logical}_n{node}",
            )
            prob += (
                sink[logical, node]
                + pulp.lpSum(outgoing_arcs[logical, node])
                == x[node],
                f"flow_out_q{logical}_n{node}",
            )

    max_single_qubit_cost = 1 + sum(
        max((nodes[node].single_qubit_count for node in node_ids), default=0)
        for node_ids in nodes_by_partition
    )
    prob.setObjective(
        pulp.lpSum(
            (
                alternative.cnot_count * max_single_qubit_cost
                + alternative.single_qubit_count
            )
            * x[node]
            for node, alternative in enumerate(nodes)
        )
    )

    solve_kwargs = {}
    if timeout_seconds is not None:
        solve_kwargs["timeLimit"] = float(timeout_seconds)
    solver_name = _solve_pulp_with_gurobi_or_cbc(
        prob, pulp, **solve_kwargs
    )
    if pulp.LpStatus[prob.status] != "Optimal":
        if timeout_seconds is not None:
            raise ExactRoutingLimitExceeded(
                f"The {solver_name} routing ILP did not prove optimality "
                f"within {timeout_seconds} seconds; status="
                f"{pulp.LpStatus[prob.status]}."
            )
        raise ValueError(
            "No optimal compatible exact-cover routing solution exists: "
            f"status={pulp.LpStatus[prob.status]}."
        )

    selected_nodes = {
        node for node in range(len(nodes)) if int(round(pulp.value(x[node])))
    }
    selected_gate_partition = {}
    for node in selected_nodes:
        partition = nodes[node].partition
        for gate in partition_sets[partition]:
            if gate in selected_gate_partition:
                raise AssertionError("ILP result covers a gate more than once.")
            selected_gate_partition[gate] = partition

    selected_partitions = {nodes[node].partition for node in selected_nodes}
    successors = {partition: set() for partition in selected_partitions}
    indegree = {partition: 0 for partition in selected_partitions}
    for gate, predecessors in gate_predecessors.items():
        right = selected_gate_partition[int(gate)]
        for predecessor in predecessors:
            left = selected_gate_partition[int(predecessor)]
            if left != right and right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    ready = [partition for partition, degree in indegree.items() if degree == 0]
    ready.sort()
    partition_order = []
    while ready:
        partition = ready.pop(0)
        partition_order.append(partition)
        for successor in sorted(successors[partition]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(partition_order) != len(selected_partitions):
        raise AssertionError("ILP returned a cyclic partition quotient.")
    selected_by_partition = {
        nodes[node].partition: node for node in selected_nodes
    }
    selections = tuple(
        RoutingSelection(
            partition,
            nodes[selected_by_partition[partition]],
        )
        for partition in partition_order
    )
    initial_result = tuple(
        next(
            physical
            for physical in range(physical_qubit_count)
            if int(round(pulp.value(initial[logical][physical])))
        )
        for logical in range(logical_qubit_count)
    )
    final_result = tuple(
        next(
            physical
            for physical in range(physical_qubit_count)
            if int(round(pulp.value(final[logical][physical])))
        )
        for logical in range(logical_qubit_count)
    )

    # Replay the selected topological order independently of the flow model.
    replay_mapping = list(initial_result)
    for selection in selections:
        alternative = selection.alternative
        for logical, physical in zip(
            alternative.logical_qubits, alternative.input_physical
        ):
            if replay_mapping[logical] != physical:
                raise AssertionError(
                    "ILP qubit-fixing flow does not replay in dependency order."
                )
        for logical, physical in zip(
            alternative.logical_qubits, alternative.output_physical
        ):
            replay_mapping[logical] = physical
        if len(set(replay_mapping)) != len(replay_mapping):
            raise AssertionError("ILP routing mapping ceased to be injective.")
    if tuple(replay_mapping) != final_result:
        raise AssertionError("ILP final mapping does not match flow replay.")

    return ExactRoutingResult(
        selections=selections,
        cnot_count=sum(selection.alternative.cnot_count for selection in selections),
        single_qubit_count=sum(
            selection.alternative.single_qubit_count for selection in selections
        ),
        initial_mapping=initial_result,
        final_mapping=final_result,
        explored_states=0,
        master_backend="dense-ilp",
    )


def solve_exact_routing_ilp_lazy_cuts(
    *,
    gate_predecessors: Mapping[int, Iterable[int]],
    partitions: Sequence[Iterable[int]],
    alternatives: Mapping[int, Sequence[RoutingAlternative]],
    logical_qubit_count: int,
    gate_qubits: Mapping[int, Iterable[int]] | None = None,
    physical_qubit_count: int | None = None,
    initial_mapping: Sequence[int] | None = None,
    timeout_seconds: float | None = None,
    max_states: int | None = None,
) -> ExactRoutingResult:
    """Solve mapping flow by sparse, iteratively generated PuLP constraints.

    The dense formulation has one successor variable for every compatible
    configuration pair. This equivalent branch-and-cut master starts with the
    exact-cover objective and adds only violated mapping-flow constraints:
    consecutive selected configurations on a logical wire must agree on its
    physical location, and first configurations must form an injective initial
    placement. Each solve remains a global MILP optimum over all columns.
    """
    if max_states is not None:
        raise ValueError(
            "exact_routing_max_states applies only to the branch-and-bound "
            "master; use exact_routing_timeout_seconds for the ILP master."
        )
    import pulp
    from squander.partitioning.ilp import (
        _solve_pulp_with_gurobi_or_cbc,
        sol_to_badsccs,
    )

    logical_qubit_count = int(logical_qubit_count)
    physical_qubit_count = int(
        logical_qubit_count
        if physical_qubit_count is None
        else physical_qubit_count
    )
    if logical_qubit_count < 1 or physical_qubit_count < logical_qubit_count:
        raise ValueError("Physical width must cover every logical qubit.")
    gate_indices = sorted(int(gate) for gate in gate_predecessors)
    if gate_indices != list(range(len(gate_indices))):
        raise ValueError("Gate indices must be contiguous from zero.")
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    if any(not part for part in partition_sets):
        raise ValueError("Empty routing partitions are not allowed.")
    if any(
        gate < 0 or gate >= len(gate_indices)
        for part in partition_sets
        for gate in part
    ):
        raise ValueError("Routing partition references an unknown gate.")

    nodes = []
    nodes_by_partition = [[] for _ in partition_sets]
    for partition, values in sorted(alternatives.items()):
        partition = int(partition)
        if partition < 0 or partition >= len(partition_sets):
            raise ValueError(f"Unknown partition alternative key {partition}.")
        for alternative in values:
            if alternative.partition != partition:
                raise ValueError("Routing alternative partition id mismatch.")
            if any(
                logical < 0 or logical >= logical_qubit_count
                for logical in alternative.logical_qubits
            ):
                raise ValueError("Alternative references an unknown logical qubit.")
            if any(
                physical < 0 or physical >= physical_qubit_count
                for physical in (
                    alternative.input_physical + alternative.output_physical
                )
            ):
                raise ValueError("Alternative references an unknown physical qubit.")
            node = len(nodes)
            nodes.append(alternative)
            nodes_by_partition[partition].append(node)
    if not nodes:
        raise ValueError("No routing alternatives were supplied.")

    gate_to_partitions = [[] for _ in gate_indices]
    for partition, gate_set in enumerate(partition_sets):
        for gate in gate_set:
            gate_to_partitions[gate].append(partition)
    if any(not values for values in gate_to_partitions):
        raise ValueError("Every gate must belong to at least one partition.")
    gate_successors = {gate: set() for gate in gate_indices}
    for gate, predecessors in gate_predecessors.items():
        for predecessor in predecessors:
            gate_successors[int(predecessor)].add(int(gate))
    ancestors = _routing_gate_ancestors(gate_predecessors)

    @functools.lru_cache(maxsize=None)
    def partition_precedes(left, right):
        return any(
            left_gate in ancestors[right_gate]
            for left_gate in partition_sets[left]
            for right_gate in partition_sets[right]
        )

    prob = pulp.LpProblem("SparseExactPermutationAwareRouting", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("cfg", range(len(nodes)), cat="Binary")
    selected_partition = {
        partition: pulp.lpSum(x[node] for node in node_ids)
        for partition, node_ids in enumerate(nodes_by_partition)
    }
    for gate in gate_indices:
        prob += (
            pulp.lpSum(
                x[node]
                for partition in gate_to_partitions[gate]
                for node in nodes_by_partition[partition]
            )
            == 1,
            f"cover_gate_{gate}",
        )
    for partition, node_ids in enumerate(nodes_by_partition):
        if node_ids:
            prob += selected_partition[partition] <= 1

    max_single_qubit_cost = 1 + sum(
        max((nodes[node].single_qubit_count for node in node_ids), default=0)
        for node_ids in nodes_by_partition
    )
    prob.setObjective(
        pulp.lpSum(
            (
                alternative.cnot_count * max_single_qubit_cost
                + alternative.single_qubit_count
            )
            * x[node]
            for node, alternative in enumerate(nodes)
        )
    )

    fixed_mapping = None
    if initial_mapping is not None:
        fixed_mapping = tuple(int(value) for value in initial_mapping)
        if len(fixed_mapping) != logical_qubit_count:
            raise ValueError("Initial mapping width mismatch.")
        assigned = [physical for physical in fixed_mapping if physical >= 0]
        if (
            len(set(assigned)) != len(assigned)
            or any(physical >= physical_qubit_count for physical in assigned)
        ):
            raise ValueError("Initial mapping is not an injective partial mapping.")

    nodes_on_qubit = {
        logical: tuple(
            node
            for node, alternative in enumerate(nodes)
            if logical in alternative.logical_qubits
        )
        for logical in range(logical_qubit_count)
    }
    node_qubit_index = {
        (node, logical): alternative.logical_qubits.index(logical)
        for node, alternative in enumerate(nodes)
        for logical in alternative.logical_qubits
    }

    def before_nodes(node, logical):
        partition = nodes[node].partition
        return tuple(
            candidate
            for candidate in nodes_on_qubit[logical]
            if candidate != node
            and partition_precedes(nodes[candidate].partition, partition)
            and not partition_precedes(partition, nodes[candidate].partition)
        )

    def between_nodes(left, right, logical):
        left_partition = nodes[left].partition
        right_partition = nodes[right].partition
        return tuple(
            candidate
            for candidate in nodes_on_qubit[logical]
            if candidate not in (left, right)
            and partition_precedes(
                left_partition, nodes[candidate].partition
            )
            and partition_precedes(
                nodes[candidate].partition, right_partition
            )
            and not partition_precedes(
                nodes[candidate].partition, left_partition
            )
            and not partition_precedes(
                right_partition, nodes[candidate].partition
            )
        )

    started = time.monotonic()
    cut_signatures = set()
    cut_rounds = 0
    solver_name = "unknown"

    def violated_cut_descriptors(selected_nodes):
        """Return valid sparse cuts violated by one integer incumbent."""
        selected_nodes = set(selected_nodes)
        selected_partitions = {nodes[node].partition for node in selected_nodes}
        descriptors = []
        bad_sccs = sol_to_badsccs(
            gate_successors, partition_sets, selected_partitions
        )
        for scc in bad_sccs:
            scc_nodes = tuple(
                node
                for partition in scc
                for node in nodes_by_partition[partition]
            )
            descriptors.append(
                (("cycle", tuple(sorted(scc_nodes))), scc_nodes, (), len(scc) - 1)
            )
        if descriptors:
            return descriptors

        selected_gate_partition = {
            gate: nodes[node].partition
            for node in selected_nodes
            for gate in partition_sets[nodes[node].partition]
        }
        successors = {partition: set() for partition in selected_partitions}
        indegree = {partition: 0 for partition in selected_partitions}
        for gate, predecessors in gate_predecessors.items():
            right = selected_gate_partition[int(gate)]
            for predecessor in predecessors:
                left = selected_gate_partition[int(predecessor)]
                if left != right and right not in successors[left]:
                    successors[left].add(right)
                    indegree[right] += 1
        ready = sorted(
            partition for partition, degree in indegree.items() if degree == 0
        )
        ordered_partitions = []
        while ready:
            partition = ready.pop(0)
            ordered_partitions.append(partition)
            for successor in sorted(successors[partition]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort()
        if len(ordered_partitions) != len(selected_partitions):
            # sol_to_badsccs should have supplied a stronger SCC cut.
            all_nodes = tuple(sorted(selected_nodes))
            return [
                (("cycle_fallback", all_nodes), all_nodes, (), len(all_nodes) - 1)
            ]
        selected_by_partition = {
            nodes[node].partition: node for node in selected_nodes
        }
        ordered_nodes = [
            selected_by_partition[partition] for partition in ordered_partitions
        ]
        selected_on_qubit = {
            logical: [
                node
                for node in ordered_nodes
                if logical in nodes[node].logical_qubits
            ]
            for logical in range(logical_qubit_count)
        }
        for logical, qubit_nodes in selected_on_qubit.items():
            for left, right in zip(qubit_nodes, qubit_nodes[1:]):
                left_index = node_qubit_index[left, logical]
                right_index = node_qubit_index[right, logical]
                if (
                    nodes[left].output_physical[left_index]
                    == nodes[right].input_physical[right_index]
                ):
                    continue
                intermediate = between_nodes(left, right, logical)
                signature = (
                    "continuity", logical, left, right, intermediate
                )
                descriptors.append(
                    (signature, (left, right), intermediate, 1)
                )
        first_nodes = {
            logical: qubit_nodes[0]
            for logical, qubit_nodes in selected_on_qubit.items()
            if qubit_nodes
        }
        if fixed_mapping is not None:
            for logical, node in first_nodes.items():
                physical = fixed_mapping[logical]
                if physical < 0:
                    continue
                index = node_qubit_index[node, logical]
                if nodes[node].input_physical[index] != physical:
                    predecessors = before_nodes(node, logical)
                    signature = (
                        "fixed_initial", logical, node, predecessors
                    )
                    descriptors.append(
                        (signature, (node,), predecessors, 0)
                    )
        first_by_physical = collections.defaultdict(list)
        for logical, node in first_nodes.items():
            index = node_qubit_index[node, logical]
            first_by_physical[nodes[node].input_physical[index]].append(
                (logical, node)
            )
        for colliding in first_by_physical.values():
            for (left_logical, left), (right_logical, right) in itertools.combinations(
                colliding, 2
            ):
                if left == right:
                    raise AssertionError(
                        "A routing alternative has a noninjective input mapping."
                    )
                earlier_left = before_nodes(left, left_logical)
                earlier_right = before_nodes(right, right_logical)
                negative = earlier_left + earlier_right
                signature = (
                    "initial_collision",
                    left_logical,
                    left,
                    right_logical,
                    right,
                    earlier_left,
                    earlier_right,
                )
                descriptors.append(
                    (signature, (left, right), negative, 1)
                )
        return descriptors

    try:
        from gurobipy import GRB as _GRB
        import gurobipy as _gp

        callback_cut_signatures = set()

        def gurobi_lazy_mapping_cuts(model, where):
            if where != _GRB.Callback.MIPSOL:
                return
            model_variables = [
                model.getVarByName(x[node].name) for node in range(len(nodes))
            ]
            values = model.cbGetSolution(model_variables)
            selected = {
                node for node, value in enumerate(values) if int(round(value))
            }
            for signature, positive, negative, rhs in violated_cut_descriptors(
                selected
            ):
                if signature in callback_cut_signatures:
                    continue
                model.cbLazy(
                    _gp.quicksum(model_variables[node] for node in positive)
                    - _gp.quicksum(model_variables[node] for node in negative)
                    <= rhs
                )
                callback_cut_signatures.add(signature)

    except Exception:
        gurobi_lazy_mapping_cuts = None

    while True:
        remaining = None
        if timeout_seconds is not None:
            remaining = float(timeout_seconds) - (time.monotonic() - started)
            if remaining <= 0:
                raise ExactRoutingLimitExceeded(
                    f"The routing ILP exceeded {timeout_seconds} seconds."
                )
        solve_kwargs = {} if remaining is None else {"timeLimit": remaining}
        if gurobi_lazy_mapping_cuts is not None:
            solve_kwargs.update(
                {"IntegralityFocus": 1, "LazyConstraints": 1}
            )
        solver_name = _solve_pulp_with_gurobi_or_cbc(
            prob,
            pulp,
            callback=gurobi_lazy_mapping_cuts,
            **solve_kwargs,
        )
        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            if timeout_seconds is not None:
                raise ExactRoutingLimitExceeded(
                    f"The {solver_name} routing ILP did not prove optimality "
                    f"within {timeout_seconds} seconds; status={status}."
                )
            raise ValueError(
                "No optimal compatible exact-cover routing solution exists: "
                f"status={status}."
            )
        selected_nodes = {
            node
            for node in range(len(nodes))
            if int(round(pulp.value(x[node])))
        }
        selected_partitions = {nodes[node].partition for node in selected_nodes}

        bad_sccs = sol_to_badsccs(
            gate_successors, partition_sets, selected_partitions
        )
        added_cut = False
        for scc in bad_sccs:
            scc_nodes = tuple(
                node
                for partition in scc
                for node in nodes_by_partition[partition]
            )
            signature = ("cycle", tuple(sorted(scc_nodes)))
            if signature not in cut_signatures:
                prob += pulp.lpSum(x[node] for node in scc_nodes) <= len(scc) - 1
                cut_signatures.add(signature)
                added_cut = True
        if added_cut:
            cut_rounds += 1
            continue

        selected_gate_partition = {
            gate: nodes[node].partition
            for node in selected_nodes
            for gate in partition_sets[nodes[node].partition]
        }
        successors = {partition: set() for partition in selected_partitions}
        indegree = {partition: 0 for partition in selected_partitions}
        for gate, predecessors in gate_predecessors.items():
            right = selected_gate_partition[int(gate)]
            for predecessor in predecessors:
                left = selected_gate_partition[int(predecessor)]
                if left != right and right not in successors[left]:
                    successors[left].add(right)
                    indegree[right] += 1
        ready = sorted(
            partition for partition, degree in indegree.items() if degree == 0
        )
        ordered_partitions = []
        while ready:
            partition = ready.pop(0)
            ordered_partitions.append(partition)
            for successor in sorted(successors[partition]):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort()
        if len(ordered_partitions) != len(selected_partitions):
            raise AssertionError("Cycle cuts failed to produce an acyclic cover.")
        selected_by_partition = {
            nodes[node].partition: node for node in selected_nodes
        }
        ordered_nodes = [
            selected_by_partition[partition] for partition in ordered_partitions
        ]
        selected_on_qubit = {
            logical: [
                node
                for node in ordered_nodes
                if logical in nodes[node].logical_qubits
            ]
            for logical in range(logical_qubit_count)
        }

        for logical, qubit_nodes in selected_on_qubit.items():
            for left, right in zip(qubit_nodes, qubit_nodes[1:]):
                left_index = node_qubit_index[left, logical]
                right_index = node_qubit_index[right, logical]
                if (
                    nodes[left].output_physical[left_index]
                    == nodes[right].input_physical[right_index]
                ):
                    continue
                intermediate = between_nodes(left, right, logical)
                signature = (
                    "continuity",
                    logical,
                    left,
                    right,
                    intermediate,
                )
                if signature not in cut_signatures:
                    prob += (
                        x[left] + x[right]
                        <= 1 + pulp.lpSum(x[node] for node in intermediate)
                    )
                    cut_signatures.add(signature)
                    added_cut = True

        first_nodes = {
            logical: qubit_nodes[0]
            for logical, qubit_nodes in selected_on_qubit.items()
            if qubit_nodes
        }
        if fixed_mapping is not None:
            for logical, node in first_nodes.items():
                physical = fixed_mapping[logical]
                if physical < 0:
                    continue
                index = node_qubit_index[node, logical]
                if nodes[node].input_physical[index] == physical:
                    continue
                predecessors = before_nodes(node, logical)
                signature = ("fixed_initial", logical, node, predecessors)
                if signature not in cut_signatures:
                    prob += x[node] <= pulp.lpSum(
                        x[predecessor] for predecessor in predecessors
                    )
                    cut_signatures.add(signature)
                    added_cut = True

        first_by_physical = collections.defaultdict(list)
        for logical, node in first_nodes.items():
            index = node_qubit_index[node, logical]
            first_by_physical[nodes[node].input_physical[index]].append(
                (logical, node)
            )
        for colliding in first_by_physical.values():
            for (left_logical, left), (right_logical, right) in itertools.combinations(
                colliding, 2
            ):
                if left == right:
                    raise AssertionError(
                        "A routing alternative has a noninjective input mapping."
                    )
                earlier_left = before_nodes(left, left_logical)
                earlier_right = before_nodes(right, right_logical)
                signature = (
                    "initial_collision",
                    left_logical,
                    left,
                    right_logical,
                    right,
                    earlier_left,
                    earlier_right,
                )
                if signature not in cut_signatures:
                    prob += (
                        x[left] + x[right]
                        <= 1
                        + pulp.lpSum(x[node] for node in earlier_left)
                        + pulp.lpSum(x[node] for node in earlier_right)
                    )
                    cut_signatures.add(signature)
                    added_cut = True
        if added_cut:
            cut_rounds += 1
            continue
        break

    discovered_initial = [-1] * logical_qubit_count
    if fixed_mapping is not None:
        discovered_initial[:] = fixed_mapping
    for logical, node in first_nodes.items():
        index = node_qubit_index[node, logical]
        physical = nodes[node].input_physical[index]
        if discovered_initial[logical] >= 0 and discovered_initial[logical] != physical:
            raise AssertionError("Fixed initial mapping was not enforced.")
        discovered_initial[logical] = physical
    used = {physical for physical in discovered_initial if physical >= 0}
    unused = iter(sorted(set(range(physical_qubit_count)) - used))
    for logical, physical in enumerate(discovered_initial):
        if physical < 0:
            discovered_initial[logical] = next(unused)

    selections = tuple(
        RoutingSelection(partition, nodes[selected_by_partition[partition]])
        for partition in ordered_partitions
    )
    replay_mapping = list(discovered_initial)
    for selection in selections:
        alternative = selection.alternative
        for logical, physical in zip(
            alternative.logical_qubits, alternative.input_physical
        ):
            if replay_mapping[logical] != physical:
                raise AssertionError(
                    "Sparse ILP mapping constraints do not replay."
                )
        for logical, physical in zip(
            alternative.logical_qubits, alternative.output_physical
        ):
            replay_mapping[logical] = physical
        if len(set(replay_mapping)) != len(replay_mapping):
            raise AssertionError("Sparse ILP mapping ceased to be injective.")

    return ExactRoutingResult(
        selections=selections,
        cnot_count=sum(selection.alternative.cnot_count for selection in selections),
        single_qubit_count=sum(
            selection.alternative.single_qubit_count for selection in selections
        ),
        initial_mapping=tuple(discovered_initial),
        final_mapping=tuple(replay_mapping),
        explored_states=cut_rounds,
        master_backend="lazy-cut-ilp",
    )


def solve_exact_routing_ilp(
    *,
    gate_predecessors: Mapping[int, Iterable[int]],
    partitions: Sequence[Iterable[int]],
    alternatives: Mapping[int, Sequence[RoutingAlternative]],
    logical_qubit_count: int,
    gate_qubits: Mapping[int, Iterable[int]] | None = None,
    physical_qubit_count: int | None = None,
    topology: Iterable[Sequence[int]] | None = None,
    initial_mapping: Sequence[int] | None = None,
    timeout_seconds: float | None = None,
    max_states: int | None = None,
    allow_suboptimal: bool = False,
    _model_cache: dict | None = None,
    _model_cache_key=None,
) -> ExactRoutingResult:
    """Solve exact routing with compact logical-wire boundary mapping flow.

    For logical qubit ``q``, ``location[q,b,p]`` states that it occupies
    physical qubit ``p`` at boundary ``b`` in q's gate sequence. Selecting a
    partition configuration fixes the locations at its first and last boundary
    on every participating wire. Exact gate coverage tiles these intervals, so
    consecutive selected partitions share the same boundary variable without
    pairwise successor arcs or global time indices.
    """
    if max_states is not None:
        raise ValueError(
            "exact_routing_max_states applies only to the branch-and-bound "
            "master; use exact_routing_timeout_seconds for the ILP master."
        )
    import pulp
    from squander.partitioning.ilp import (
        _solve_pulp_with_gurobi_or_cbc,
        sol_to_badsccs,
    )

    logical_qubit_count = int(logical_qubit_count)
    physical_qubit_count = int(
        logical_qubit_count
        if physical_qubit_count is None
        else physical_qubit_count
    )
    if logical_qubit_count < 1 or physical_qubit_count < logical_qubit_count:
        raise ValueError("Physical width must cover every logical qubit.")
    gate_indices = sorted(int(gate) for gate in gate_predecessors)
    if gate_indices != list(range(len(gate_indices))):
        raise ValueError("Gate indices must be contiguous from zero.")
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    if any(not part for part in partition_sets):
        raise ValueError("Empty routing partitions are not allowed.")
    if any(
        gate < 0 or gate >= len(gate_indices)
        for part in partition_sets
        for gate in part
    ):
        raise ValueError("Routing partition references an unknown gate.")

    nodes = []
    nodes_by_partition = [[] for _ in partition_sets]
    for partition, values in sorted(alternatives.items()):
        partition = int(partition)
        if partition < 0 or partition >= len(partition_sets):
            raise ValueError(f"Unknown partition alternative key {partition}.")
        expected_logical_qubits = None
        for alternative in values:
            if alternative.partition != partition:
                raise ValueError("Routing alternative partition id mismatch.")
            if any(
                logical < 0 or logical >= logical_qubit_count
                for logical in alternative.logical_qubits
            ):
                raise ValueError("Alternative references an unknown logical qubit.")
            if any(
                physical < 0 or physical >= physical_qubit_count
                for physical in (
                    alternative.input_physical + alternative.output_physical
                )
            ):
                raise ValueError("Alternative references an unknown physical qubit.")
            logical_set = frozenset(alternative.logical_qubits)
            if expected_logical_qubits is None:
                expected_logical_qubits = logical_set
            elif logical_set != expected_logical_qubits:
                raise ValueError(
                    "Alternatives of one partition disagree on logical qubits."
                )
            node = len(nodes)
            nodes.append(alternative)
            nodes_by_partition[partition].append(node)
    if not nodes:
        raise ValueError("No routing alternatives were supplied.")

    gate_to_partitions = [[] for _ in gate_indices]
    for partition, gate_set in enumerate(partition_sets):
        for gate in gate_set:
            gate_to_partitions[gate].append(partition)
    if any(not values for values in gate_to_partitions):
        raise ValueError("Every gate must belong to at least one partition.")

    if gate_qubits is None:
        inferred_gate_qubits = {}
        for gate in gate_indices:
            containing_sets = [
                set(nodes[nodes_by_partition[partition][0]].logical_qubits)
                for partition in gate_to_partitions[gate]
                if nodes_by_partition[partition]
            ]
            if not containing_sets:
                raise ValueError(f"Cannot infer logical qubits for gate {gate}.")
            inferred_gate_qubits[gate] = frozenset.intersection(
                *map(frozenset, containing_sets)
            )
        gate_qubits = inferred_gate_qubits
    else:
        gate_qubits = {
            int(gate): frozenset(map(int, qubits))
            for gate, qubits in gate_qubits.items()
        }
    if set(gate_qubits) != set(gate_indices):
        raise ValueError("gate_qubits must describe every routing gate.")
    if any(
        not qubits
        or any(logical < 0 or logical >= logical_qubit_count for logical in qubits)
        for qubits in gate_qubits.values()
    ):
        raise ValueError("gate_qubits contains an invalid logical-qubit set.")

    gate_successors = {gate: set() for gate in gate_indices}
    remaining_predecessors = {
        gate: {int(value) for value in gate_predecessors[gate]}
        for gate in gate_indices
    }
    for gate, predecessors in remaining_predecessors.items():
        for predecessor in predecessors:
            if predecessor not in gate_successors:
                raise ValueError("Routing dependency references an unknown gate.")
            gate_successors[predecessor].add(gate)
    ready = sorted(gate for gate in gate_indices if not remaining_predecessors[gate])
    topological_gates = []
    while ready:
        gate = ready.pop(0)
        topological_gates.append(gate)
        for successor in sorted(gate_successors[gate]):
            remaining_predecessors[successor].remove(gate)
            if not remaining_predecessors[successor]:
                ready.append(successor)
                ready.sort()
    if len(topological_gates) != len(gate_indices):
        raise ValueError("Routing gate dependency graph contains a cycle.")
    wire_gates = {
        logical: tuple(
            gate
            for gate in topological_gates
            if logical in gate_qubits[gate]
        )
        for logical in range(logical_qubit_count)
    }
    wire_position = {
        (logical, gate): position
        for logical, gates in wire_gates.items()
        for position, gate in enumerate(gates)
    }
    wire_boundary_count = {
        logical: max(2, len(gates) + 1)
        for logical, gates in wire_gates.items()
    }

    partition_boundaries = {}
    nonconvex_partitions = set()
    spectator_partitions = collections.defaultdict(set)
    for partition, node_ids in enumerate(nodes_by_partition):
        if not node_ids:
            continue
        alternative = nodes[node_ids[0]]
        boundaries = {}
        for logical in alternative.logical_qubits:
            positions = sorted(
                wire_position[logical, gate]
                for gate in partition_sets[partition]
                if logical in gate_qubits[gate]
            )
            if not positions:
                if (
                    wire_gates[logical]
                    or partition_sets[partition] != frozenset(gate_indices)
                ):
                    raise ValueError(
                        "Only a whole-circuit fallback may move a globally "
                        "idle spectator qubit."
                    )
                boundaries[logical] = (0, 1)
                spectator_partitions[logical].add(partition)
            else:
                first, last = positions[0], positions[-1]
                interval_gates = set(wire_gates[logical][first : last + 1])
                if not interval_gates <= partition_sets[partition]:
                    nonconvex_partitions.add(partition)
                boundaries[logical] = (first, last + 1)
        partition_boundaries[partition] = boundaries

    prob = pulp.LpProblem("WireBoundaryExactRouting", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("cfg", range(len(nodes)), cat="Binary")
    selected_partition = {
        partition: pulp.lpSum(x[node] for node in node_ids)
        for partition, node_ids in enumerate(nodes_by_partition)
    }
    for gate in gate_indices:
        prob += (
            pulp.lpSum(
                x[node]
                for partition in gate_to_partitions[gate]
                for node in nodes_by_partition[partition]
            )
            == 1,
            f"cover_gate_{gate}",
        )
    for partition, node_ids in enumerate(nodes_by_partition):
        if node_ids:
            prob += selected_partition[partition] <= 1
            if partition in nonconvex_partitions:
                prob += selected_partition[partition] == 0

    location = {}
    for logical, gates in wire_gates.items():
        for boundary in range(wire_boundary_count[logical]):
            variables = []
            for physical in range(physical_qubit_count):
                variable = pulp.LpVariable(
                    f"location_q{logical}_b{boundary}_p{physical}",
                    cat="Binary",
                )
                location[logical, boundary, physical] = variable
                variables.append(variable)
            prob += pulp.lpSum(variables) == 1
    for physical in range(physical_qubit_count):
        prob += (
            pulp.lpSum(
                location[logical, 0, physical]
                for logical in range(logical_qubit_count)
            )
            <= 1
        )
        prob += (
            pulp.lpSum(
                location[logical, wire_boundary_count[logical] - 1, physical]
                for logical in range(logical_qubit_count)
            )
            <= 1
        )
    for logical, partitions_with_spectator in spectator_partitions.items():
        movement_enabled = pulp.lpSum(
            selected_partition[partition]
            for partition in partitions_with_spectator
        )
        for physical in range(physical_qubit_count):
            prob += (
                location[logical, 0, physical]
                - location[logical, 1, physical]
                <= movement_enabled
            )
            prob += (
                location[logical, 1, physical]
                - location[logical, 0, physical]
                <= movement_enabled
            )

    if initial_mapping is not None:
        fixed_mapping = tuple(int(value) for value in initial_mapping)
        if len(fixed_mapping) != logical_qubit_count:
            raise ValueError("Initial mapping width mismatch.")
        assigned = [physical for physical in fixed_mapping if physical >= 0]
        if (
            len(set(assigned)) != len(assigned)
            or any(physical >= physical_qubit_count for physical in assigned)
        ):
            raise ValueError("Initial mapping is not an injective partial mapping.")
        for logical, physical in enumerate(fixed_mapping):
            if physical >= 0:
                prob += location[logical, 0, physical] == 1
    elif topology is not None and physical_qubit_count > 1:
        # A path has one nontrivial device automorphism: reflection. Every
        # feasible route and its mirror have identical costs, so choose the
        # representative placing logical qubit zero in the first half. For an
        # odd path, qubit zero may occupy the fixed center; qubit one then
        # breaks the remaining reflection tie.
        adjacency = {physical: set() for physical in range(physical_qubit_count)}
        for edge in _normalize_edges(topology):
            left, right = tuple(edge)
            if left in adjacency and right in adjacency:
                adjacency[left].add(right)
                adjacency[right].add(left)
        endpoints = sorted(
            physical for physical, neighbours in adjacency.items()
            if len(neighbours) == 1
        )
        is_path = (
            len(endpoints) == 2
            and sum(len(neighbours) for neighbours in adjacency.values())
            == 2 * (physical_qubit_count - 1)
            and all(1 <= len(neighbours) <= 2 for neighbours in adjacency.values())
        )
        if is_path:
            path = [endpoints[0]]
            previous = None
            while len(path) < physical_qubit_count:
                candidates = adjacency[path[-1]] - (
                    set() if previous is None else {previous}
                )
                if not candidates:
                    break
                previous, current = path[-1], min(candidates)
                path.append(current)
            if len(path) == physical_qubit_count:
                path_position = {
                    physical: position for position, physical in enumerate(path)
                }
                midpoint = (physical_qubit_count - 1) / 2.0
                initial_position_zero = pulp.lpSum(
                    path_position[physical] * location[0, 0, physical]
                    for physical in range(physical_qubit_count)
                )
                prob += initial_position_zero <= midpoint
                if physical_qubit_count % 2 == 1 and logical_qubit_count > 1:
                    center_physical = path[physical_qubit_count // 2]
                    initial_position_one = pulp.lpSum(
                        path_position[physical] * location[1, 0, physical]
                        for physical in range(physical_qubit_count)
                    )
                    prob += initial_position_one <= midpoint + (
                        physical_qubit_count - 1
                    ) * (1 - location[0, 0, center_physical])

    for node, alternative in enumerate(nodes):
        boundaries = partition_boundaries[alternative.partition]
        for index, logical in enumerate(alternative.logical_qubits):
            first, after_last = boundaries[logical]
            prob += (
                x[node]
                <= location[
                    logical, first, alternative.input_physical[index]
                ]
            )
            prob += (
                x[node]
                <= location[
                    logical, after_last, alternative.output_physical[index]
                ]
            )

    max_single_qubit_cost = 1 + sum(
        max((nodes[node].single_qubit_count for node in node_ids), default=0)
        for node_ids in nodes_by_partition
    )
    objective_coefficients = tuple(
        alternative.cnot_count * max_single_qubit_cost
        + alternative.single_qubit_count
        for alternative in nodes
    )
    prob.setObjective(
        pulp.lpSum(
            objective_coefficients[node] * x[node]
            for node in range(len(nodes))
        )
    )

    structure_signature = tuple(
        (
            alternative.partition,
            alternative.logical_qubits,
            alternative.input_physical,
            alternative.output_physical,
        )
        for alternative in nodes
    )
    persistent_entry = (
        None
        if _model_cache is None
        else _model_cache.get(_model_cache_key)
    )
    if persistent_entry is not None:
        if persistent_entry["structure_signature"] != structure_signature:
            raise AssertionError("Persistent routing-master structure changed.")
        prob = persistent_entry["prob"]
        x = persistent_entry["x"]
        location = persistent_entry["location"]
        for node, coefficient in enumerate(objective_coefficients):
            x[node].solverVar.Obj = coefficient
        prob.solverModel.update()

    try:
        from gurobipy import GRB as _GRB
        import gurobipy as _gp
        callback_signatures = set()

        def cycle_callback(model, where):
            if where != _GRB.Callback.MIPSOL:
                return
            model_variables = [
                model.getVarByName(x[node].name) for node in range(len(nodes))
            ]
            values = model.cbGetSolution(model_variables)
            selected_nodes = {
                node for node, value in enumerate(values) if int(round(value))
            }
            selected_partitions = {
                nodes[node].partition for node in selected_nodes
            }
            for scc in sol_to_badsccs(
                gate_successors, partition_sets, selected_partitions
            ):
                scc_nodes = tuple(
                    node
                    for partition in scc
                    for node in nodes_by_partition[partition]
                )
                signature = tuple(sorted(scc_nodes))
                if signature in callback_signatures:
                    continue
                model.cbLazy(
                    _gp.quicksum(model_variables[node] for node in scc_nodes)
                    <= len(scc) - 1
                )
                callback_signatures.add(signature)

    except Exception:
        cycle_callback = None

    started = time.monotonic()
    cut_rounds = 0
    while True:
        remaining = None
        if timeout_seconds is not None:
            remaining = float(timeout_seconds) - (time.monotonic() - started)
            if remaining <= 0:
                raise ExactRoutingLimitExceeded(
                    f"The routing ILP exceeded {timeout_seconds} seconds."
                )
        solve_kwargs = {} if remaining is None else {"timeLimit": remaining}
        if cycle_callback is not None:
            solve_kwargs.update({"IntegralityFocus": 1, "LazyConstraints": 1})
        if persistent_entry is None:
            solver_name = _solve_pulp_with_gurobi_or_cbc(
                prob, pulp, callback=cycle_callback, **solve_kwargs
            )
            if (
                _model_cache is not None
                and str(solver_name).lower().startswith("gurobi")
            ):
                persistent_entry = {
                    "structure_signature": structure_signature,
                    "prob": prob,
                    "x": x,
                    "location": location,
                }
                _model_cache[_model_cache_key] = persistent_entry
        else:
            solver_name = "gurobi-persistent"
            model = prob.solverModel
            if remaining is not None:
                model.setParam("TimeLimit", remaining)
            model.setParam("IntegralityFocus", 1)
            model.setParam("LazyConstraints", 1)
            model.optimize(cycle_callback)
            prob.solver.findSolutionValues(prob)
        if (
            str(solver_name).lower().startswith("gurobi")
            and prob.solverModel.Status == _GRB.INTERRUPTED
        ):
            raise KeyboardInterrupt
        status = pulp.LpStatus[prob.status]
        proven_optimal = status == "Optimal"
        has_gurobi_incumbent = (
            str(solver_name).lower().startswith("gurobi")
            and prob.solverModel.SolCount >= 1
        )
        if not proven_optimal and not (
            allow_suboptimal and has_gurobi_incumbent
        ):
            if status == "Infeasible":
                raise ValueError(
                    "No compatible exact-cover routing solution exists."
                )
            if timeout_seconds is not None:
                raise ExactRoutingLimitExceeded(
                    f"The {solver_name} routing ILP did not prove optimality "
                    f"within {timeout_seconds} seconds; status={status}."
                )
            raise ValueError(
                "No optimal compatible exact-cover routing solution exists: "
                f"status={status}."
            )
        selected_nodes = {
            node
            for node in range(len(nodes))
            if int(round(pulp.value(x[node])))
        }
        selected_partitions = {nodes[node].partition for node in selected_nodes}
        bad_sccs = sol_to_badsccs(
            gate_successors, partition_sets, selected_partitions
        )
        if not bad_sccs:
            break
        if not proven_optimal:
            raise ExactRoutingLimitExceeded(
                "The routing ILP timed out with only cyclic incumbents."
            )
        for scc in bad_sccs:
            scc_nodes = tuple(
                node
                for partition in scc
                for node in nodes_by_partition[partition]
            )
            if persistent_entry is not None:
                prob.solverModel.addConstr(
                    _gp.quicksum(x[node].solverVar for node in scc_nodes)
                    <= len(scc) - 1
                )
                prob.solverModel.update()
            else:
                prob += (
                    pulp.lpSum(x[node] for node in scc_nodes)
                    <= len(scc) - 1
                )
        cut_rounds += 1

    selected_gate_partition = {
        gate: nodes[node].partition
        for node in selected_nodes
        for gate in partition_sets[nodes[node].partition]
    }
    successors = {partition: set() for partition in selected_partitions}
    indegree = {partition: 0 for partition in selected_partitions}
    for gate, predecessors in gate_predecessors.items():
        right = selected_gate_partition[int(gate)]
        for predecessor in predecessors:
            left = selected_gate_partition[int(predecessor)]
            if left != right and right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    ready = sorted(
        partition for partition, degree in indegree.items() if degree == 0
    )
    ordered_partitions = []
    while ready:
        partition = ready.pop(0)
        ordered_partitions.append(partition)
        for successor in sorted(successors[partition]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(ordered_partitions) != len(selected_partitions):
        raise AssertionError("Cycle cuts failed to produce an acyclic cover.")
    selected_by_partition = {
        nodes[node].partition: node for node in selected_nodes
    }
    selections = tuple(
        RoutingSelection(partition, nodes[selected_by_partition[partition]])
        for partition in ordered_partitions
    )
    initial_result = tuple(
        next(
            physical
            for physical in range(physical_qubit_count)
            if int(round(pulp.value(location[logical, 0, physical])))
        )
        for logical in range(logical_qubit_count)
    )
    final_result = tuple(
        next(
            physical
            for physical in range(physical_qubit_count)
            if int(
                round(
                    pulp.value(
                        location[
                            logical, wire_boundary_count[logical] - 1, physical
                        ]
                    )
                )
            )
        )
        for logical in range(logical_qubit_count)
    )
    replay_mapping = list(initial_result)
    for selection in selections:
        alternative = selection.alternative
        for logical, physical in zip(
            alternative.logical_qubits, alternative.input_physical
        ):
            if replay_mapping[logical] != physical:
                raise AssertionError(
                    "Wire-boundary ILP mapping does not replay."
                )
        for logical, physical in zip(
            alternative.logical_qubits, alternative.output_physical
        ):
            replay_mapping[logical] = physical
        if len(set(replay_mapping)) != len(replay_mapping):
            raise AssertionError("Wire-boundary ILP mapping ceased to be injective.")
    if tuple(replay_mapping) != final_result:
        raise AssertionError("Wire-boundary ILP final mapping does not replay.")

    return ExactRoutingResult(
        selections=selections,
        cnot_count=sum(selection.alternative.cnot_count for selection in selections),
        single_qubit_count=sum(
            selection.alternative.single_qubit_count for selection in selections
        ),
        initial_mapping=initial_result,
        final_mapping=final_result,
        explored_states=cut_rounds,
        master_backend=f"pulp-{solver_name}",
        optimal=proven_optimal,
    )


def solve_exact_routing(*, backend="ilp", **kwargs) -> ExactRoutingResult:
    """Dispatch to the PuLP mapping-flow master or preserved branch-and-bound."""
    backend = str(backend).lower().replace("_", "-")
    if backend == "auto":
        try:
            from squander.partitioning.ilp import _check_gurobi_available

            _check_gurobi_available()
            backend = "ilp"
        except Exception:
            backend = "branch-and-bound"
    if backend in ("ilp", "pulp"):
        return solve_exact_routing_ilp(**kwargs)
    if backend in ("branch-and-bound", "bnb"):
        kwargs.pop("allow_suboptimal", None)
        kwargs.pop("topology", None)
        return solve_exact_routing_branch_and_bound(**kwargs)
    raise ValueError(
        "Unknown exact-routing master backend "
        f"{backend!r}; expected 'auto', 'ilp', or 'branch-and-bound'."
    )


def construct_exact_routed_circuit(
    result: ExactRoutingResult,
    physical_qubit_count: int,
):
    """Materialize an exact-routing selection as a Squander circuit.

    Local payload circuits are expressed on canonical subtopology wires.  This
    routine embeds each one at the selected physical vertices and concatenates
    its parameters without changing their bit patterns.
    """
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit

    physical_qubit_count = int(physical_qubit_count)
    routed_circuit = Circuit(physical_qubit_count)
    parameter_blocks = []
    for selection in result.selections:
        alternative = selection.alternative
        payload = alternative.payload
        if not isinstance(payload, SynthesizedRoutingPayload):
            raise TypeError(
                "A materialized exact route requires synthesized payloads."
            )
        if len(payload.input_assignment) != len(alternative.logical_qubits):
            raise AssertionError("Routing payload width mismatch.")

        # input_physical is indexed by local logical wire, while the payload's
        # assignment is indexed by canonical physical wire.
        embedding = {
            local_physical: alternative.input_physical[local_logical]
            for local_physical, local_logical in enumerate(
                payload.input_assignment
            )
        }
        # Remap_Qbits does not recursively relabel a nested BLOCK gate; flatten
        # before embedding so every primitive gate receives the physical map.
        embedded = payload.circuit.get_Flat_Circuit().Remap_Qbits(
            embedding, physical_qubit_count
        )
        routed_circuit.add_Circuit(embedded)
        parameter_blocks.append(
            np.asarray(payload.parameters, dtype=np.float64)
        )

    # add_Circuit stores each selected rewrite as a nested BLOCK. Downstream
    # basis conversion accepts primitive gates, so materialize the route as an
    # actually flat circuit rather than merely flattening each input payload.
    routed_circuit = routed_circuit.get_Flat_Circuit()
    parameters = (
        np.concatenate(parameter_blocks)
        if parameter_blocks
        else np.empty((0,), dtype=np.float64)
    )
    if routed_circuit.get_Parameter_Num() != parameters.size:
        raise AssertionError("Routed circuit parameter stream is inconsistent.")
    return routed_circuit, parameters


def route_circuit_exact(
    circuit,
    parameters: np.ndarray,
    topology: Iterable[Sequence[int]],
    config: Mapping[str, Any],
) -> ExactCircuitRoutingResult:
    """Route a Squander circuit by exact cover and permutation-aware OSR.

    All convex partitions produced by the existing all-partition enumerator
    are retained.  Contracted one-qubit chains are either absorbed where the
    enumerator permits or represented as standalone candidates, so the exact
    cover is over the original gate stream rather than an approximation of it.

    The combinatorial solve is exact over generated alternatives, but its
    worst-case state space is exponential and OSR is a numerical pricing
    oracle. This strategy is consequently experimental and should be bounded
    with the candidate, state, and time limits for larger circuits.
    """
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
    from squander.partitioning.ilp import get_all_partitions, _get_topo_order

    config = dict(config)
    routing_started = time.monotonic()
    total_timeout = config.get("exact_routing_total_timeout_seconds", 20 * 60)
    if total_timeout is not None:
        total_timeout = float(total_timeout)
        if total_timeout <= 0:
            raise ValueError("The exact-routing total timeout must be positive.")
        routing_deadline = routing_started + total_timeout
        config["_exact_routing_deadline"] = routing_deadline
    else:
        routing_deadline = None

    parameters = np.asarray(parameters, dtype=np.float64)
    max_partition_size = int(config.get("max_partition_size", 3))
    (
        allparts,
        _contracted_graph,
        successors,
        predecessors,
        single_qubit_chains,
        gate_to_qubit,
        _gate_to_target,
    ) = get_all_partitions(circuit, max_partition_size)
    allparts = sorted(
        (frozenset(int(gate) for gate in part) for part in allparts),
        key=lambda part: (len(part), tuple(sorted(part))),
    )

    pre_chains = {
        chain[0]: chain for chain in single_qubit_chains if predecessors[chain[0]]
    }
    post_chains = {
        chain[-1]: chain for chain in single_qubit_chains if successors[chain[-1]]
    }
    internal_chains = {
        chain[0]: chain
        for chain in single_qubit_chains
        if chain[0] in pre_chains and chain[-1] in post_chains
    }
    candidate_sets = []
    for part in allparts:
        surrounded = {
            child
            for gate in part
            for child in successors[gate]
            if child in internal_chains
            and successors[internal_chains[child][-1]]
            and next(iter(successors[internal_chains[child][-1]])) in part
        }
        candidate_sets.append(
            frozenset().union(
                part,
                *(frozenset(internal_chains[start]) for start in surrounded),
            )
        )
    candidate_sets.extend(frozenset(chain) for chain in single_qubit_chains)
    candidate_sets = sorted(
        set(candidate_sets), key=lambda part: (len(part), tuple(sorted(part)))
    )

    candidate_limit = config.get("exact_routing_max_candidates")
    if candidate_limit is not None and len(candidate_sets) > int(candidate_limit):
        raise ExactRoutingLimitExceeded(
            f"Exact routing generated {len(candidate_sets)} candidates, "
            f"above the configured limit {candidate_limit}."
        )

    local_candidate_sets = candidate_sets
    fallback_partition = len(local_candidate_sets)
    fallback_gate_set = frozenset(range(len(circuit.get_Gates())))
    candidate_sets = [*local_candidate_sets, fallback_gate_set]
    candidate_gate_orders = []
    for gate_set in local_candidate_sets:
        local_successors = {
            gate: successors[gate] & gate_set for gate in gate_set
        }
        local_predecessors = {
            gate: predecessors[gate] & gate_set for gate in gate_set
        }
        candidate_gate_orders.append(
            tuple(
                _get_topo_order(
                    local_successors,
                    local_predecessors,
                    gate_to_qubit,
                )
            )
        )
    candidate_gate_orders.append(
        tuple(_get_topo_order(successors, predecessors, gate_to_qubit))
    )

    gate_dict = {index: gate for index, gate in enumerate(circuit.get_Gates())}
    feasible_alternatives = {}
    optimistic_alternatives = {}
    synthesis_metadata = {}
    synthesis_cache = {}
    target_synthesis_cache = {}
    target_synthesis_cache_stats = [0]
    synthesis_cache_hits = 0
    synthesized_partitions = set()
    lazy_osr = bool(config.get("exact_routing_lazy_osr", False))
    global_synthesis_batch = _GlobalRoutingSynthesisBatch(
        config,
        target_synthesis_cache,
        target_synthesis_cache_stats,
    )
    deferred_refinements = {}

    def synthesis_cache_key(unitary, fallback_alternatives, requested):
        fallback_signature = tuple(
            sorted(
                (
                    alternative.payload.input_assignment,
                    alternative.payload.output_assignment,
                    alternative.cnot_count,
                    alternative.single_qubit_count,
                )
                for alternative in fallback_alternatives
            )
        )
        return (
            unitary.shape,
            np.ascontiguousarray(unitary).view(np.uint8).tobytes(),
            tuple(
                sorted(
                    tuple(sorted((int(u), int(v))))
                    for u, v in topology
                )
            ),
            _freeze_synthesis_cache_value(config),
            fallback_signature,
            None if requested is None else tuple(sorted(requested)),
        )

    def rebind_cached_alternatives(
        cached, partition_index, involved_qubits, compact, subparameters
    ):
        return tuple(
            replace(
                alternative,
                partition=partition_index,
                logical_qubits=involved_qubits,
                payload=replace(
                    alternative.payload,
                    source_circuit=compact,
                    source_parameters=np.asarray(
                        subparameters, dtype=np.float64
                    ),
                ),
            )
            for alternative in cached
        )

    # Prepare the guaranteed incumbent before expensive OSR pricing so the
    # route-wide deadline can always return a valid, audited result.
    from squander.decomposition.qgd_Wide_Circuit_Optimization import (
        CNOTGateCount,
        SingleQubitGateCount,
    )
    from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE

    sabre = SABRE(
        circuit,
        topology,
        random_seed=config.get("exact_routing_random_seed"),
    )
    requested_initial_mapping = config.get("exact_routing_initial_mapping")
    if requested_initial_mapping is not None:
        requested_initial_mapping = tuple(map(int, requested_initial_mapping))
        if len(requested_initial_mapping) != circuit.get_Qbit_Num():
            raise ValueError("Initial mapping width mismatch.")
        assigned = [value for value in requested_initial_mapping if value >= 0]
        if (
            len(set(assigned)) != len(assigned)
            or any(value >= circuit.get_Qbit_Num() for value in assigned)
        ):
            raise ValueError("Initial mapping is not an injective partial mapping.")
        available = iter(
            physical
            for physical in range(circuit.get_Qbit_Num())
            if physical not in assigned
        )
        sabre.pi = np.asarray(
            [
                next(available) if value < 0 else value
                for value in requested_initial_mapping
            ],
            dtype=int,
        )
    (
        sabre_circuit,
        sabre_parameters,
        sabre_initial_mapping,
        sabre_final_mapping,
        _swap_count,
    ) = sabre.map_circuit(parameters)
    sabre_circuit = sabre_circuit.get_Flat_Circuit()
    sabre_parameters = np.asarray(sabre_parameters, dtype=np.float64)
    _assert_local_topology(sabre_circuit, topology)
    sabre_initial_mapping = tuple(map(int, sabre_initial_mapping))
    sabre_final_mapping = tuple(map(int, sabre_final_mapping))
    if (
        tuple(sorted(sabre_initial_mapping)) != tuple(range(circuit.get_Qbit_Num()))
        or tuple(sorted(sabre_final_mapping)) != tuple(range(circuit.get_Qbit_Num()))
    ):
        raise AssertionError("SABRE fallback returned an invalid mapping.")
    fallback_logical_qubits = tuple(range(circuit.get_Qbit_Num()))
    fallback_alternative = RoutingAlternative(
        partition=fallback_partition,
        logical_qubits=fallback_logical_qubits,
        input_physical=sabre_initial_mapping,
        output_physical=sabre_final_mapping,
        cnot_count=CNOTGateCount(sabre_circuit, 0),
        single_qubit_count=SingleQubitGateCount(sabre_circuit),
        payload=SynthesizedRoutingPayload(
            circuit=sabre_circuit,
            parameters=sabre_parameters,
            topology=tuple(
                sorted(tuple(sorted((int(u), int(v)))) for u, v in topology)
            ),
            input_assignment=_inverse_permutation(sabre_initial_mapping),
            output_assignment=_inverse_permutation(sabre_final_mapping),
            source_circuit=circuit,
            source_parameters=parameters,
            certificate_kind="sabre",
        ),
    )
    feasible_alternatives[fallback_partition] = (fallback_alternative,)
    optimistic_alternatives[fallback_partition] = (fallback_alternative,)

    def finish_with_timeout(solution=None):
        if solution is None:
            solution = ExactRoutingResult(
                selections=(
                    RoutingSelection(fallback_partition, fallback_alternative),
                ),
                cnot_count=fallback_alternative.cnot_count,
                single_qubit_count=fallback_alternative.single_qubit_count,
                initial_mapping=sabre_initial_mapping,
                final_mapping=sabre_final_mapping,
                explored_states=0,
                master_backend="sabre-timeout-incumbent",
                optimal=False,
            )
        else:
            solution = replace(
                solution,
                master_backend=f"{solution.master_backend}-timeout-incumbent",
                optimal=False,
            )
        routed_circuit, routed_parameters = construct_exact_routed_circuit(
            solution, circuit.get_Qbit_Num()
        )
        return ExactCircuitRoutingResult(
            circuit=routed_circuit,
            parameters=routed_parameters,
            solution=solution,
            candidate_gate_sets=tuple(
                tuple(sorted(part)) for part in candidate_sets
            ),
            candidate_gate_orders=tuple(candidate_gate_orders),
            synthesized_partitions=len(synthesized_partitions),
            synthesis_cache_hits=(
                synthesis_cache_hits + target_synthesis_cache_stats[0]
            ),
            timed_out=True,
        )

    for partition_index, gate_set in enumerate(local_candidate_sets):
        if routing_deadline is not None and time.monotonic() >= routing_deadline:
            return finish_with_timeout()
        involved_qubits = tuple(
            sorted(set().union(*(gate_to_qubit[gate] for gate in gate_set)))
        )
        if len(involved_qubits) > max_partition_size:
            raise AssertionError("Partition enumerator exceeded its width bound.")
        subcircuit = Circuit(circuit.get_Qbit_Num())
        parameter_blocks = []
        gate_order = candidate_gate_orders[partition_index]
        for gate in gate_order:
            source_gate = gate_dict[gate]
            subcircuit.add_Gate(source_gate)
            start = source_gate.get_Parameter_Start_Index()
            parameter_blocks.append(
                parameters[start : start + source_gate.get_Parameter_Num()]
            )
        subparameters = (
            np.concatenate(parameter_blocks)
            if parameter_blocks
            else np.empty((0,), dtype=np.float64)
        )
        compact_map = {
            logical: local for local, logical in enumerate(involved_qubits)
        }
        compact = subcircuit.Remap_Qbits(compact_map, len(involved_qubits))
        if len(involved_qubits) == 1:
            feasible_alternatives[partition_index] = (
                single_qubit_passthrough_alternatives(
                    partition=partition_index,
                    circuit=compact,
                    parameters=subparameters,
                    logical_qubit=involved_qubits[0],
                    physical_qubit_count=circuit.get_Qbit_Num(),
                )
            )
        else:
            unitary = compact.get_Matrix(subparameters, is_f32=False)
            fallback_config = {
                **dict(config),
                "exact_routing_eager_synthesis": False,
            }
            fallback_alternatives = synthesize_partition_alternatives(
                partition=partition_index,
                unitary=unitary,
                logical_qubits=involved_qubits,
                topology=topology,
                physical_qubit_count=circuit.get_Qbit_Num(),
                config=fallback_config,
                source_circuit=compact,
                source_parameters=subparameters,
                synthesis_cache=target_synthesis_cache,
                synthesis_cache_stats=target_synthesis_cache_stats,
            )
            feasible_alternatives[partition_index] = fallback_alternatives
            if "strategy" in config:
                if lazy_osr:
                    optimistic_alternatives[partition_index] = tuple(
                        replace(
                            alternative,
                            cnot_count=cnot_schmidt_lower_bound(
                                permuted_partition_target(
                                    unitary,
                                    alternative.payload.input_assignment,
                                    alternative.payload.output_assignment,
                                ),
                                alternative.payload.topology,
                                rank_tolerance=float(
                                    config.get("routing_rank_tolerance", 1e-8)
                                ),
                            ),
                            single_qubit_count=0,
                            payload=None,
                        )
                        for alternative in fallback_alternatives
                    )
                    synthesis_metadata[partition_index] = (
                        unitary,
                        involved_qubits,
                        compact,
                        subparameters,
                    )
                else:
                    eager_config = {
                        **dict(config),
                        "exact_routing_eager_synthesis": True,
                    }
                    deferred_refinements[partition_index] = (
                        synthesize_partition_alternatives(
                            partition=partition_index,
                            unitary=unitary,
                            logical_qubits=involved_qubits,
                            topology=topology,
                            physical_qubit_count=circuit.get_Qbit_Num(),
                            config=eager_config,
                            source_circuit=compact,
                            source_parameters=subparameters,
                            synthesis_cache=target_synthesis_cache,
                            synthesis_cache_stats=target_synthesis_cache_stats,
                            synthesis_batch=global_synthesis_batch,
                        )
                    )

        if partition_index not in optimistic_alternatives:
            optimistic_alternatives[partition_index] = (
                feasible_alternatives[partition_index]
            )

    global_synthesis_batch.run()
    for partition_index, deferred in deferred_refinements.items():
        refined = deferred.resolve()
        feasible_alternatives[partition_index] = refined
        optimistic_alternatives[partition_index] = refined
        synthesized_partitions.add(partition_index)

    if routing_deadline is not None and time.monotonic() >= routing_deadline:
        return finish_with_timeout()
    remaining_total = (
        None
        if routing_deadline is None
        else max(0.0, routing_deadline - time.monotonic())
    )
    configured_master_timeout = config.get("exact_routing_timeout_seconds")
    if configured_master_timeout is None:
        master_timeout = remaining_total
    elif remaining_total is None:
        master_timeout = float(configured_master_timeout)
    else:
        master_timeout = min(float(configured_master_timeout), remaining_total)

    solver_arguments = {
        "backend": config.get("exact_routing_master", "ilp"),
        "gate_predecessors": {
            gate: predecessors[gate] for gate in range(len(gate_dict))
        },
        "gate_qubits": {
            gate: gate_to_qubit[gate] for gate in range(len(gate_dict))
        },
        "partitions": candidate_sets,
        "logical_qubit_count": circuit.get_Qbit_Num(),
        "physical_qubit_count": circuit.get_Qbit_Num(),
        "topology": topology,
        "initial_mapping": config.get("exact_routing_initial_mapping"),
        "timeout_seconds": master_timeout,
        "max_states": config.get("exact_routing_max_states"),
        "allow_suboptimal": True,
    }
    refined_transition_groups = set()
    lazy_rounds = 0
    refinement_limit = int(
        config.get(
            "exact_routing_max_refinement_rounds",
            max(1, 4 * len(candidate_sets)),
        )
    )
    try:
        incumbent_solution = solve_exact_routing(
            alternatives=feasible_alternatives, **solver_arguments
        )
    except ExactRoutingLimitExceeded:
        return finish_with_timeout()
    if not incumbent_solution.optimal:
        return finish_with_timeout(incumbent_solution)
    solution = incumbent_solution
    lower_master_cache = {}
    lower_solver_arguments = dict(solver_arguments)
    if str(lower_solver_arguments["backend"]).lower() in ("ilp", "pulp"):
        lower_solver_arguments.update(
            {
                "_model_cache": lower_master_cache,
                "_model_cache_key": "optimistic",
            }
        )
    while lazy_osr:
        if routing_deadline is not None:
            remaining = routing_deadline - time.monotonic()
            if remaining <= 0:
                return finish_with_timeout(incumbent_solution)
            configured_timeout = config.get("exact_routing_timeout_seconds")
            lower_solver_arguments["timeout_seconds"] = (
                remaining
                if configured_timeout is None
                else min(float(configured_timeout), remaining)
            )
        try:
            lower_bound_solution = solve_exact_routing(
                alternatives=optimistic_alternatives,
                **lower_solver_arguments,
            )
        except ExactRoutingLimitExceeded:
            return finish_with_timeout(incumbent_solution)
        if not lower_bound_solution.optimal:
            return finish_with_timeout(incumbent_solution)
        incumbent_cost = (
            incumbent_solution.cnot_count,
            incumbent_solution.single_qubit_count,
        )
        lower_bound_cost = (
            lower_bound_solution.cnot_count,
            lower_bound_solution.single_qubit_count,
        )
        if lower_bound_cost >= incumbent_cost:
            solution = incumbent_solution
            break
        to_refine = {}
        for selection in lower_bound_solution.selections:
            partition_index = selection.partition
            if partition_index not in synthesis_metadata:
                continue
            transition_key = (
                selection.alternative.input_physical,
                selection.alternative.output_physical,
            )
            fallback = next(
                (
                    alternative
                    for alternative in feasible_alternatives[partition_index]
                    if (
                        alternative.input_physical,
                        alternative.output_physical,
                    )
                    == transition_key
                ),
                None,
            )
            if fallback is None or fallback.payload is None:
                raise AssertionError("Lazy routing transition has no fallback.")
            transition = (
                fallback.payload.input_assignment,
                fallback.payload.output_assignment,
            )
            if (partition_index, transition) not in refined_transition_groups:
                to_refine.setdefault(partition_index, set()).add(transition)
        if not to_refine:
            raise AssertionError(
                "Lazy exact-routing lower bound cannot be closed."
            )
        lazy_rounds += 1
        if lazy_rounds > refinement_limit:
            raise ExactRoutingLimitExceeded(
                "Exact routing exceeded its lazy refinement-round limit."
            )
        for partition_index, requested_transitions in sorted(to_refine.items()):
            unitary, involved_qubits, compact, subparameters = (
                synthesis_metadata[partition_index]
            )
            priced_transitions = set()
            local_topologies = {
                alternative.payload.topology
                for alternative in feasible_alternatives[partition_index]
                if alternative.payload is not None
            }
            for local_edges in local_topologies:
                for _representative, orbit in symmetry_reduced_assignment_orbits(
                    local_edges, len(involved_qubits)
                ):
                    orbit_transitions = {
                        member for member, _automorphism in orbit
                    }
                    if orbit_transitions & requested_transitions:
                        priced_transitions.update(orbit_transitions)
            cache_key = synthesis_cache_key(
                unitary,
                feasible_alternatives[partition_index],
                priced_transitions,
            )
            cached = synthesis_cache.get(cache_key)
            if cached is None:
                eager_config = {
                    **dict(config),
                    "exact_routing_eager_synthesis": True,
                }
                refined = synthesize_partition_alternatives(
                    partition=partition_index,
                    unitary=unitary,
                    logical_qubits=involved_qubits,
                    topology=topology,
                    physical_qubit_count=circuit.get_Qbit_Num(),
                    config=eager_config,
                    source_circuit=compact,
                    source_parameters=subparameters,
                    requested_transitions=priced_transitions,
                    synthesis_cache=target_synthesis_cache,
                    synthesis_cache_stats=target_synthesis_cache_stats,
                )
                synthesis_cache[cache_key] = refined
            else:
                synthesis_cache_hits += 1
                refined = rebind_cached_alternatives(
                    cached,
                    partition_index,
                    involved_qubits,
                    compact,
                    subparameters,
                )
            refined_by_key = {
                (alternative.input_physical, alternative.output_physical): alternative
                for alternative in refined
                if (
                    alternative.payload.input_assignment,
                    alternative.payload.output_assignment,
                )
                in priced_transitions
            }
            feasible_by_key = {
                (alternative.input_physical, alternative.output_physical): alternative
                for alternative in feasible_alternatives[partition_index]
            }
            for key, alternative in refined_by_key.items():
                previous = feasible_by_key[key]
                if (
                    alternative.cnot_count,
                    alternative.single_qubit_count,
                ) < (
                    previous.cnot_count,
                    previous.single_qubit_count,
                ):
                    feasible_by_key[key] = alternative
            feasible_alternatives[partition_index] = tuple(
                feasible_by_key.values()
            )
            priced_keys = {
                (alternative.input_physical, alternative.output_physical)
                for alternative in feasible_by_key.values()
                if alternative.payload is not None
                and (
                    alternative.payload.input_assignment,
                    alternative.payload.output_assignment,
                )
                in priced_transitions
            }
            optimistic_alternatives[partition_index] = tuple(
                feasible_by_key.get(
                    (alternative.input_physical, alternative.output_physical),
                    alternative,
                )
                if (
                    alternative.input_physical,
                    alternative.output_physical,
                )
                in priced_keys
                else alternative
                for alternative in optimistic_alternatives[partition_index]
            )
            for transition in priced_transitions:
                refined_transition_groups.add((partition_index, transition))
            synthesized_partitions.add(partition_index)

        actual_selections = tuple(
            RoutingSelection(
                selection.partition,
                next(
                    alternative
                    for alternative in feasible_alternatives[
                        selection.partition
                    ]
                    if (
                        alternative.input_physical,
                        alternative.output_physical,
                    )
                    == (
                        selection.alternative.input_physical,
                        selection.alternative.output_physical,
                    )
                ),
            )
            for selection in lower_bound_solution.selections
        )
        actual_solution = replace(
            lower_bound_solution,
            selections=actual_selections,
            cnot_count=sum(
                selection.alternative.cnot_count
                for selection in actual_selections
            ),
            single_qubit_count=sum(
                selection.alternative.single_qubit_count
                for selection in actual_selections
            ),
        )
        if (
            actual_solution.cnot_count,
            actual_solution.single_qubit_count,
        ) < incumbent_cost:
            incumbent_solution = actual_solution

    routed_circuit, routed_parameters = construct_exact_routed_circuit(
        solution, circuit.get_Qbit_Num()
    )
    return ExactCircuitRoutingResult(
        circuit=routed_circuit,
        parameters=routed_parameters,
        solution=solution,
        candidate_gate_sets=tuple(
            tuple(sorted(part)) for part in candidate_sets
        ),
        candidate_gate_orders=tuple(candidate_gate_orders),
        lazy_rounds=lazy_rounds,
        synthesized_partitions=len(synthesized_partitions),
        synthesis_cache_hits=(
            synthesis_cache_hits + target_synthesis_cache_stats[0]
        ),
    )
