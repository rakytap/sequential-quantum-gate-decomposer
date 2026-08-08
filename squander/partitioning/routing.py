"""Exact permutation-aware routing over pre-synthesized partition alternatives.

This module deliberately separates expensive local synthesis from global routing.
For each feasible gate partition, a caller supplies topology-valid alternatives
with an input placement, output placement, and exact CNOT cost.  The solver then
chooses an exact cover, an acyclic execution order, and a compatible mapping
trajectory with minimum total CNOT count.

The publication router uses a compact PuLP/Gurobi logic-based Benders master.
It selects synthesis columns while a fixed-cover staged oracle prices complete
logical-to-physical mapping trajectories and arbitrary adjacent SWAPs.  The
monolithic staged ILP and a dependency-ready branch-and-bound backend remain
available as correctness cross-checks.
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
_DEFAULT_ROUTING_SYNTHESIS_WORKERS = 16


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
    configured_timeout = config.get("routing_synthesis_timeout_seconds")
    deadline = (
        None
        if configured_timeout is None
        else time.monotonic() + float(configured_timeout)
    )
    message = None
    while deadline is None or time.monotonic() < deadline:
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
        # Exact routing consumes only the portable circuit, parameters, and
        # local topology. Retaining one complete resolved WCO configuration
        # per OSR target duplicates runtime circuit/configuration artifacts
        # thousands of times on benchmark-scale routes.
        config={},
        topology=payload[1],
    )


def _routing_synthesis_worker_count(config, task_count):
    """Choose bounded process concurrency for memory-heavy native OSR calls."""
    configured_workers = config.get("routing_synthesis_workers")
    if configured_workers is None:
        configured_workers = config.get("partition_workers")
    if configured_workers is None:
        configured_workers = min(
            mp.cpu_count(), _DEFAULT_ROUTING_SYNTHESIS_WORKERS
        )
    return max(1, min(int(configured_workers), int(task_count)))


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

    worker_count = _routing_synthesis_worker_count(config, len(targets))
    configured_timeout = config.get("routing_synthesis_timeout_seconds")
    timeout = (
        None if configured_timeout is None else float(configured_timeout)
    )
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
            None if timeout is None else time.monotonic() + timeout,
        )

    for _ in range(worker_count):
        try:
            launch(*next(pending))
        except StopIteration:
            break

    while active:
        progressed = False
        for index, (process, connection, deadline) in list(active.items()):
            message = None
            if connection.poll():
                message = connection.recv()
            elif process.is_alive() and (
                deadline is None or time.monotonic() < deadline
            ):
                continue
            if process.is_alive():
                process.terminate()
            process.join()
            connection.close()
            if message is not None:
                results[index] = _decode_synthesis_message(message)
            del active[index]
            progressed = True
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
    restart_count = int(config.get("exact_routing_synthesis_restarts", 1))
    if restart_count != 1:
        raise ValueError(
            "Exact routing synthesizes each canonical OSR target exactly once; "
            "exact_routing_synthesis_restarts must be 1."
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

    missing_results = [None] * len(missing_targets)
    if missing_targets:
        configured_seed = config.get("exact_routing_random_seed", 0)
        if configured_seed is None:
            configured_seed = 0
        seeded_configs = []
        for target, target_config in zip(missing_targets, missing_configs):
            target_config = dict(target_config)
            target_config["random_seed"] = _deterministic_synthesis_seed(
                target,
                topology,
                int(configured_seed),
            )
            seeded_configs.append(target_config)
        missing_results = list(
            _call_shared_synthesis_batch(
                missing_targets,
                config,
                topology,
                target_configs=seeded_configs,
            )
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
        # Detach the queue before executing it so the batch object cannot keep
        # every target/config/consumer closure alive after its topology has
        # been consumed. This matters when thousands of routing columns are
        # priced in one exhaustive round.
        queued_requests = self.requests
        self.requests = collections.defaultdict(list)
        for topology, requests in sorted(queued_requests.items()):
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
            requests.clear()


def _normalize_edges(edges: Iterable[Sequence[int]]) -> frozenset[frozenset[int]]:
    """Return an undirected, loop-free edge set."""
    return frozenset(
        frozenset((int(edge[0]), int(edge[1])))
        for edge in edges
        if int(edge[0]) != int(edge[1])
    )


def _path_topology_order(
    edges: Iterable[Sequence[int]], width: int
) -> tuple[int, ...] | None:
    """Return a deterministic endpoint-to-endpoint order for a path topology."""
    width = int(width)
    if width < 2:
        return None
    adjacency = {physical: set() for physical in range(width)}
    for edge in _normalize_edges(edges):
        left, right = tuple(edge)
        if left not in adjacency or right not in adjacency:
            return None
        adjacency[left].add(right)
        adjacency[right].add(left)
    endpoints = sorted(
        physical
        for physical, neighbours in adjacency.items()
        if len(neighbours) == 1
    )
    if (
        len(endpoints) != 2
        or sum(map(len, adjacency.values())) != 2 * (width - 1)
        or any(not 1 <= len(neighbours) <= 2 for neighbours in adjacency.values())
    ):
        return None
    path = [endpoints[0]]
    previous = None
    while len(path) < width:
        candidates = adjacency[path[-1]] - (
            set() if previous is None else {previous}
        )
        if len(candidates) != 1:
            return None
        previous, current = path[-1], next(iter(candidates))
        path.append(current)
    return tuple(path)


def _canonicalize_path_reflection(
    circuit,
    initial_mapping: Sequence[int],
    final_mapping: Sequence[int],
    topology: Iterable[Sequence[int]],
):
    """Mirror a path route when needed to satisfy the ILP symmetry anchor."""
    initial_mapping = tuple(map(int, initial_mapping))
    final_mapping = tuple(map(int, final_mapping))
    width = len(initial_mapping)
    path = _path_topology_order(topology, width)
    if path is None:
        return circuit, initial_mapping, final_mapping
    position = {physical: index for index, physical in enumerate(path)}
    midpoint = (width - 1) / 2.0
    mirror = position[initial_mapping[0]] > midpoint
    if (
        not mirror
        and width % 2 == 1
        and width > 1
        and position[initial_mapping[0]] == midpoint
    ):
        mirror = position[initial_mapping[1]] > midpoint
    if not mirror:
        return circuit, initial_mapping, final_mapping
    reflection = {
        physical: path[-1 - index] for index, physical in enumerate(path)
    }
    return (
        circuit.Remap_Qbits(reflection, width).get_Flat_Circuit(),
        tuple(reflection[physical] for physical in initial_mapping),
        tuple(reflection[physical] for physical in final_mapping),
    )


def _light_sabre_route(
    circuit,
    parameters: np.ndarray,
    topology: Iterable[Sequence[int]],
    config: Mapping[str, Any],
):
    """Return LightSABRE's route plus a source-gate/mapping trajectory."""
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import SabreLayout, SabreSwap
    from squander import Qiskit_IO

    qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit(
        circuit, np.asarray(parameters, dtype=np.float64)
    )
    source_gate_count = len(circuit.get_Gates())
    if len(qiskit_circuit.data) != source_gate_count:
        raise AssertionError(
            "Qiskit export is not one instruction per Squander source gate."
        )
    label_prefix = "squander_exact_source_"
    for gate_index, instruction in enumerate(tuple(qiskit_circuit.data)):
        operation = instruction.operation.to_mutable()
        operation.label = f"{label_prefix}{gate_index}"
        qiskit_circuit.data[gate_index] = instruction.replace(
            operation=operation
        )
    coupling_map = CouplingMap(
        [[int(left), int(right)] for left, right in topology]
    )
    seed = config.get(
        "sabre_seed", config.get("exact_routing_random_seed", 42)
    )
    if seed is None:
        seed = 42
    layout_trials = int(config.get("sabre_trials", 5))
    swap_trials = int(config.get("sabre_swap_trials", layout_trials))
    max_iterations = int(config.get("sabre_max_iterations", 3))
    if layout_trials < 1 or swap_trials < 1 or max_iterations < 1:
        raise ValueError("LightSABRE trial and iteration counts must be positive.")
    pass_manager = PassManager(
        [
            SabreLayout(
                coupling_map,
                seed=int(seed),
                max_iterations=max_iterations,
                swap_trials=swap_trials,
                layout_trials=layout_trials,
            ),
            SabreSwap(
                coupling_map,
                heuristic=config.get("sabre_heuristic", "decay"),
                seed=int(seed),
                trials=swap_trials,
            ),
        ]
    )
    routed_qiskit = pass_manager.run(qiskit_circuit)
    if routed_qiskit.layout is None:
        raise AssertionError("LightSABRE did not return routing-layout metadata.")
    initial_mapping = tuple(
        map(int, routed_qiskit.layout.initial_index_layout())
    )
    final_mapping = tuple(
        map(int, routed_qiskit.layout.final_index_layout())
    )
    trace = []
    seen_source_gates = set()
    for instruction in routed_qiskit.data:
        label = instruction.operation.label
        physical = tuple(
            int(routed_qiskit.find_bit(qubit).index)
            for qubit in instruction.qubits
        )
        if isinstance(label, str) and label.startswith(label_prefix):
            gate_index = int(label[len(label_prefix) :])
            if gate_index in seen_source_gates:
                raise AssertionError("LightSABRE duplicated a source gate.")
            seen_source_gates.add(gate_index)
            trace.append(("gate", gate_index))
        elif instruction.operation.name == "swap":
            if len(physical) != 2:
                raise AssertionError("LightSABRE emitted a malformed SWAP.")
            trace.append(("swap", physical))
        else:
            raise AssertionError(
                "LightSABRE emitted an untraceable routing instruction: "
                f"{instruction.operation.name!r}."
            )
    if seen_source_gates != set(range(source_gate_count)):
        raise AssertionError("LightSABRE trace does not cover every source gate.")
    routed_circuit, routed_parameters = Qiskit_IO.convert_Qiskit_to_Squander(
        routed_qiskit
    )
    return (
        routed_circuit.get_Flat_Circuit(),
        np.asarray(routed_parameters, dtype=np.float64),
        initial_mapping,
        final_mapping,
        tuple(trace),
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
                fallback = None
                if input_assignment == output_assignment:
                    # P U P^-1 is a free relabeling whenever P preserves the
                    # chosen local topology. Do not encode it as physical
                    # SWAPs on both boundaries.
                    relabeling = {
                        logical_wire: physical_wire
                        for physical_wire, logical_wire in enumerate(
                            input_assignment
                        )
                    }
                    relabeled = topology_valid_source.Remap_Qbits(
                        relabeling, width
                    ).get_Flat_Circuit()
                    try:
                        _assert_local_topology(relabeled, local_edges)
                    except AssertionError:
                        pass
                    else:
                        fallback = relabeled
                if fallback is None:
                    fallback = Circuit(width)
                    for edge in _permutation_swaps(
                        _inverse_permutation(input_assignment), local_edges
                    ):
                        fallback.add_SWAP(list(edge))
                    fallback.add_Circuit(topology_valid_source)
                    for edge in _permutation_swaps(
                        output_assignment, local_edges
                    ):
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
            rigorous_lower_bound = cnot_schmidt_lower_bound(
                target,
                local_edges,
                rank_tolerance=float(
                    config.get("routing_rank_tolerance", 1e-8)
                ),
            )
            if (
                fallback_cost is not None
                and fallback_cost <= rigorous_lower_bound
            ):
                # The validated topology fallback already saturates a
                # rigorous entangling-gate lower bound. OSR cannot strictly
                # improve its CNOT count, so pricing this target is pointless.
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
            if input_assignment == output_assignment == (1, 0):
                # Conjugating a two-wire block by SWAP only relabels its
                # wires.  Materializing SWAP-U-SWAP charges six fictitious
                # CNOTs and badly corrupts routing MIP starts.
                local_circuit = circuit.Remap_Qbits({0: 1, 1: 0}, 2)
            else:
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
    transition_swaps: tuple[tuple[Edge, ...], ...] = ()
    solver_nodes: float | None = None
    solver_bound: float | None = None
    solver_gap: float | None = None
    solver_solutions: int | None = None
    cnot_optimal: bool | None = None


def _finite_nonnegative_solver_bound(value) -> float | None:
    """Return a usable minimization bound, or ``None`` when unavailable."""
    if value is None:
        return None
    try:
        bound = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(bound):
        return None
    # Every routing objective used here is nonnegative.  Some solvers can
    # expose a tiny negative dual bound because of numerical tolerances.
    return max(0.0, bound)


def _flow_seed_model_term_count(partitions, alternatives) -> int:
    """Estimate dominant PuLP terms before building the auxiliary flow model."""
    partition_sets = tuple(frozenset(partition) for partition in partitions)
    coverage_terms = sum(
        len(partition_sets[partition]) * len(values)
        for partition, values in alternatives.items()
    )
    # Each configuration is linked to both the input and output boundary for
    # every logical wire it acts upon.
    placement_terms = 2 * sum(
        len(alternative.logical_qubits)
        for values in alternatives.values()
        for alternative in values
    )
    return int(coverage_terms + placement_terms)


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


def _raise_if_seed_deadline_expired(deadline, phase):
    """Interrupt optional seed construction at its wall-clock deadline."""
    if deadline is not None and time.monotonic() >= float(deadline):
        raise ExactRoutingLimitExceeded(
            f"Exact-routing {phase} seed exceeded its wall-clock budget."
        )


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


def _solve_exact_routing_ilp_flow_restricted(
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
    warm_start: ExactRoutingResult | None = None,
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
        path = _path_topology_order(topology, physical_qubit_count)
        if path is not None:
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

    if warm_start is not None:
        node_by_identity = {
            id(alternative): node for node, alternative in enumerate(nodes)
        }
        warm_nodes = []
        for selection in warm_start.selections:
            node = node_by_identity.get(id(selection.alternative))
            if node is None:
                raise ValueError(
                    "Exact-routing warm start contains an unknown alternative."
                )
            warm_nodes.append(node)
        warm_node_set = set(warm_nodes)
        if len(warm_node_set) != len(warm_nodes):
            raise ValueError("Exact-routing warm start repeats an alternative.")
        covered = collections.Counter(
            gate
            for node in warm_nodes
            for gate in partition_sets[nodes[node].partition]
        )
        if any(covered[gate] != 1 for gate in gate_indices):
            raise ValueError("Exact-routing warm start is not an exact cover.")
        for node in range(len(nodes)):
            x[node].setInitialValue(int(node in warm_node_set))
        for logical in range(logical_qubit_count):
            initial_physical = int(warm_start.initial_mapping[logical])
            final_physical = int(warm_start.final_mapping[logical])
            last_boundary = wire_boundary_count[logical] - 1
            for physical in range(physical_qubit_count):
                location[logical, 0, physical].setInitialValue(
                    int(physical == initial_physical)
                )
                location[logical, last_boundary, physical].setInitialValue(
                    int(physical == final_physical)
                )

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
        if warm_start is not None and persistent_entry is None:
            solve_kwargs["warmStart"] = True
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
                error = ExactRoutingLimitExceeded(
                    f"The {solver_name} routing ILP did not prove optimality "
                    f"within {timeout_seconds} seconds; status={status}."
                )
                if str(solver_name).lower().startswith("gurobi"):
                    error.solver_bound = _finite_nonnegative_solver_bound(
                        float(prob.solverModel.ObjBound)
                        / max_single_qubit_cost
                    )
                raise error
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
        solver_nodes=(
            None
            if not str(solver_name).lower().startswith("gurobi")
            else float(prob.solverModel.NodeCount)
        ),
        solver_bound=(
            None
            if not str(solver_name).lower().startswith("gurobi")
            else _finite_nonnegative_solver_bound(
                float(prob.solverModel.ObjBound) / max_single_qubit_cost
            )
        ),
        solver_gap=(
            None
            if not str(solver_name).lower().startswith("gurobi")
            else float(prob.solverModel.MIPGap)
        ),
        solver_solutions=(
            None
            if not str(solver_name).lower().startswith("gurobi")
            else int(prob.solverModel.SolCount)
        ),
    )


def _path_mapping_swap_sequence(
    source: Sequence[int], target: Sequence[int], path: Sequence[int]
) -> tuple[Edge, ...]:
    """Return a minimum adjacent-SWAP sequence between two path mappings."""
    source = tuple(map(int, source))
    target = tuple(map(int, target))
    path = tuple(map(int, path))
    if sorted(source) != sorted(target) or sorted(source) != sorted(path):
        raise ValueError("Path mappings must be permutations of the device vertices.")
    physical_to_logical = {
        physical: logical for logical, physical in enumerate(source)
    }
    target_physical_to_logical = {
        physical: logical for logical, physical in enumerate(target)
    }
    target_at_position = [target_physical_to_logical[vertex] for vertex in path]
    current_at_position = [physical_to_logical[vertex] for vertex in path]
    current_position = {
        logical: position for position, logical in enumerate(current_at_position)
    }
    swaps = []
    for destination, logical in enumerate(target_at_position):
        source_position = current_position[logical]
        while source_position > destination:
            left = source_position - 1
            swaps.append((path[left], path[source_position]))
            displaced = current_at_position[left]
            current_at_position[left], current_at_position[source_position] = (
                current_at_position[source_position],
                current_at_position[left],
            )
            current_position[logical] = left
            current_position[displaced] = source_position
            source_position = left
    if current_at_position != target_at_position:
        raise AssertionError("Adjacent-SWAP construction did not reach its target.")
    return tuple(swaps)


def _light_sabre_structural_warm_start(
    *,
    trace: Sequence[tuple[str, Any]],
    initial_mapping: Sequence[int],
    partitions: Sequence[frozenset[int]],
    alternatives: Mapping[int, Sequence[RoutingAlternative]],
    topology: Iterable[Sequence[int]],
) -> ExactRoutingResult:
    """Express a traced LightSABRE route as a complete singleton trajectory."""
    singleton_partition = {}
    for partition, gate_set in enumerate(partitions):
        if len(gate_set) == 1:
            singleton_partition.setdefault(next(iter(gate_set)), partition)
    mapping = list(map(int, initial_mapping))
    seed_initial_mapping = tuple(mapping)
    allowed_edges = {
        frozenset((int(left), int(right))) for left, right in topology
    }
    selections = []
    transition_swaps = []
    pending_swaps = []
    seen_gates = set()
    initial_mapping = None
    final_mapping = seed_initial_mapping
    for kind, payload in trace:
        if kind == "swap":
            left, right = map(int, payload)
            if frozenset((left, right)) not in allowed_edges:
                raise AssertionError("LightSABRE warm-start SWAP violates topology.")
            left_logical = mapping.index(left)
            right_logical = mapping.index(right)
            mapping[left_logical], mapping[right_logical] = (
                mapping[right_logical],
                mapping[left_logical],
            )
            pending_swaps.append((left, right))
            continue
        if kind != "gate":
            raise AssertionError(f"Unknown LightSABRE trace entry {kind!r}.")
        gate = int(payload)
        if gate in seen_gates or gate not in singleton_partition:
            raise AssertionError("Invalid LightSABRE source-gate trajectory.")
        seen_gates.add(gate)
        partition = singleton_partition[gate]
        candidates = []
        for alternative in alternatives[partition]:
            placement = tuple(mapping[q] for q in alternative.logical_qubits)
            if (
                alternative.input_physical == placement
                and alternative.output_physical == placement
            ):
                candidates.append(alternative)
        if not candidates:
            raise AssertionError(
                "A LightSABRE gate placement has no singleton routing column."
            )
        alternative = min(
            candidates,
            key=lambda value: (value.cnot_count, value.single_qubit_count),
        )
        selections.append(RoutingSelection(partition, alternative))
        if initial_mapping is None:
            # The master permits a free initial layout. Any LightSABRE SWAPs
            # before its first source gate are absorbed into that layout.
            initial_mapping = tuple(mapping)
            transition_swaps.append(())
        else:
            transition_swaps.append(tuple(pending_swaps))
        pending_swaps.clear()
        final_mapping = tuple(mapping)
    if seen_gates != set(singleton_partition):
        raise AssertionError("LightSABRE warm start does not cover every gate.")
    # A final mapping is unconstrained, so trailing routing SWAPs can always be
    # dropped. LightSABRE normally emits none, but they must not burden the seed.
    return ExactRoutingResult(
        selections=tuple(selections),
        cnot_count=(
            sum(value.alternative.cnot_count for value in selections)
            + 3 * sum(len(swaps) for swaps in transition_swaps)
        ),
        single_qubit_count=sum(
            value.alternative.single_qubit_count for value in selections
        ),
        initial_mapping=(
            seed_initial_mapping if initial_mapping is None else initial_mapping
        ),
        final_mapping=final_mapping,
        explored_states=0,
        master_backend="light-sabre-structural-mip-start",
        optimal=False,
        transition_swaps=tuple(transition_swaps),
    )


def _greedy_selected_path_warm_start(
    *,
    gate_predecessors: Mapping[int, Iterable[int]],
    partitions: Sequence[frozenset[int]],
    alternatives: Mapping[int, Sequence[RoutingAlternative]],
    logical_qubit_count: int,
    path: Sequence[int],
    initial_mapping: Sequence[int] | None = None,
    lookahead: bool = True,
    deadline=None,
) -> ExactRoutingResult:
    """Construct a feasible staged seed for one fixed column cover."""
    path = tuple(map(int, path))
    _raise_if_seed_deadline_expired(deadline, "fixed-cover greedy")
    path_position = {physical: index for index, physical in enumerate(path)}
    selected_gate_partition = {
        gate: partition
        for partition, gate_set in enumerate(partitions)
        for gate in gate_set
    }
    successors = {partition: set() for partition in range(len(partitions))}
    indegree = {partition: 0 for partition in range(len(partitions))}
    for gate, predecessors in gate_predecessors.items():
        right = selected_gate_partition[int(gate)]
        for predecessor in map(int, predecessors):
            left = selected_gate_partition[predecessor]
            if left != right and right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    ready = {partition for partition, degree in indegree.items() if degree == 0}
    current = (
        None if initial_mapping is None else tuple(map(int, initial_mapping))
    )
    initial_result = current
    selections = []
    transition_swaps = []
    relative_alternatives = {}
    for partition, values in alternatives.items():
        grouped = {}
        for alternative in values:
            input_positions = tuple(
                path_position[physical]
                for physical in alternative.input_physical
            )
            output_positions = tuple(
                path_position[physical]
                for physical in alternative.output_physical
            )
            base = min(input_positions)
            key = (
                tuple(value - base for value in input_positions),
                tuple(value - base for value in output_positions),
            )
            previous = grouped.get(key)
            if previous is None or (
                alternative.cnot_count,
                alternative.single_qubit_count,
            ) < (
                previous.cnot_count,
                previous.single_qubit_count,
            ):
                grouped[key] = alternative
        relative_alternatives[partition] = tuple(sorted(grouped.items()))

    def complete_mapping(source, logicals, desired):
        fixed = dict(zip(logicals, desired))
        if source is None:
            remaining_logicals = [
                logical
                for logical in range(logical_qubit_count)
                if logical not in fixed
            ]
        else:
            physical_to_logical = {
                physical: logical for logical, physical in enumerate(source)
            }
            remaining_logicals = [
                physical_to_logical[physical]
                for physical in path
                if physical_to_logical[physical] not in fixed
            ]
        remaining_physical = [
            physical for physical in path if physical not in set(desired)
        ]
        result = [-1] * logical_qubit_count
        for logical, physical in fixed.items():
            result[logical] = physical
        for logical, physical in zip(remaining_logicals, remaining_physical):
            result[logical] = physical
        return tuple(result)

    def choices_for(partition, source_mapping):
        choices = []
        for (input_offsets, output_offsets), alternative in (
            relative_alternatives[partition]
        ):
            width = len(alternative.logical_qubits)
            for start in range(logical_qubit_count - width + 1):
                desired_input = tuple(
                    path[start + value] for value in input_offsets
                )
                mapping_in = complete_mapping(
                    source_mapping, alternative.logical_qubits, desired_input
                )
                swaps = (
                    ()
                    if source_mapping is None
                    else _path_mapping_swap_sequence(
                        source_mapping, mapping_in, path
                    )
                )
                desired_output = tuple(
                    path[start + value] for value in output_offsets
                )
                mapping_out = list(mapping_in)
                for logical, physical in zip(
                    alternative.logical_qubits, desired_output
                ):
                    mapping_out[logical] = physical
                translated = replace(
                    alternative,
                    input_physical=desired_input,
                    output_physical=desired_output,
                )
                choices.append(
                    (
                        alternative.cnot_count + 3 * len(swaps),
                        alternative.single_qubit_count,
                        len(swaps),
                        partition,
                        start,
                        translated,
                        mapping_in,
                        tuple(mapping_out),
                        swaps,
                    )
                )
        return choices

    while ready:
        _raise_if_seed_deadline_expired(deadline, "fixed-cover greedy")
        choices = []
        for partition in sorted(ready):
            for choice in choices_for(partition, current):
                mapping_out = choice[7]
                # Only score direct successors unlocked by this choice.  A
                # full ready-frontier lookahead is quadratic in independent
                # blocks and became more expensive than OSR on wide circuits.
                # Direct successors capture the output-permutation benefit
                # without coupling unrelated ready work.
                future_ready = (
                    {
                        successor
                        for successor in successors[partition]
                        if indegree[successor] == 1
                    }
                    if lookahead
                    else set()
                )
                lookahead = min(
                    (
                        next_choice[0]
                        for next_partition in future_ready
                        for next_choice in choices_for(
                            next_partition, mapping_out
                        )
                    ),
                    default=0,
                )
                choices.append((choice[0] + lookahead, *choice))
        if not choices:
            raise ValueError("Selected partition quotient contains a cycle.")
        (
            _score_with_lookahead,
            _incremental_cnot_count,
            _single_qubit_count,
            _swap_count,
            partition,
            _start,
            alternative,
            mapping_in,
            mapping_out,
            swaps,
        ) = min(choices, key=lambda value: value[:6])
        if initial_result is None:
            initial_result = mapping_in
        selections.append(RoutingSelection(partition, alternative))
        transition_swaps.append(tuple(swaps))
        current = mapping_out
        ready.remove(partition)
        for successor in successors[partition]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.add(successor)
    if len(selections) != len(partitions):
        raise ValueError("Selected partition quotient contains a cycle.")
    return ExactRoutingResult(
        selections=tuple(selections),
        cnot_count=(
            sum(value.alternative.cnot_count for value in selections)
            + 3 * sum(len(swaps) for swaps in transition_swaps)
        ),
        single_qubit_count=sum(
            value.alternative.single_qubit_count for value in selections
        ),
        initial_mapping=tuple(initial_result),
        final_mapping=tuple(current),
        explored_states=0,
        master_backend="greedy-selected-path-mip-start",
        optimal=False,
        transition_swaps=tuple(transition_swaps),
    )


def _beam_selected_path_warm_start(
    *,
    gate_predecessors,
    partitions,
    alternatives,
    logical_qubit_count,
    path,
    initial_mapping=None,
    initial_mapping_candidates=None,
    beam_width=64,
    translation_limit=None,
    deadline=None,
):
    """Route one fixed cover with a bounded beam over complete mappings."""
    beam_width = int(beam_width)
    if beam_width < 1:
        raise ValueError("The fixed-cover mapping beam width must be positive.")
    if translation_limit is not None:
        translation_limit = int(translation_limit)
        if translation_limit < 1:
            raise ValueError("The seed translation limit must be positive.")
    _raise_if_seed_deadline_expired(deadline, "fixed-cover beam")
    if initial_mapping_candidates is not None:
        distinct_initial_mappings = tuple(
            dict.fromkeys(
                None if value is None else tuple(map(int, value))
                for value in initial_mapping_candidates
            )
        )
        if len(distinct_initial_mappings) > 1:
            # Each layout hypothesis needs the full beam quota.  Mixing them
            # in one global beam lets a cheap prefix from one layout evict a
            # different layout that has the better complete route.
            return min(
                (
                    _beam_selected_path_warm_start(
                        gate_predecessors=gate_predecessors,
                        partitions=partitions,
                        alternatives=alternatives,
                        logical_qubit_count=logical_qubit_count,
                        path=path,
                        initial_mapping=initial_mapping,
                        initial_mapping_candidates=(candidate,),
                        beam_width=beam_width,
                        translation_limit=translation_limit,
                        deadline=deadline,
                    )
                    for candidate in distinct_initial_mappings
                ),
                key=lambda result: (
                    result.cnot_count,
                    result.single_qubit_count,
                    result.initial_mapping,
                ),
            )
    path = tuple(map(int, path))
    path_position = {physical: index for index, physical in enumerate(path)}
    selected_gate_partition = {
        gate: partition
        for partition, gate_set in enumerate(partitions)
        for gate in gate_set
    }
    successors = {partition: set() for partition in range(len(partitions))}
    indegree = {partition: 0 for partition in range(len(partitions))}
    for gate, predecessors in gate_predecessors.items():
        right = selected_gate_partition[int(gate)]
        for predecessor in map(int, predecessors):
            left = selected_gate_partition[predecessor]
            if left != right and right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    predecessor_masks = [0] * len(partitions)
    for partition, values in successors.items():
        for successor in values:
            predecessor_masks[successor] |= 1 << partition
    all_completed = (1 << len(partitions)) - 1
    if not any(degree == 0 for degree in indegree.values()):
        raise ValueError("Selected partition quotient contains a cycle.")

    relative = {}
    for partition, values in alternatives.items():
        _raise_if_seed_deadline_expired(deadline, "fixed-cover beam")
        grouped = {}
        for alternative_index, alternative in enumerate(values):
            if alternative_index % 256 == 0:
                _raise_if_seed_deadline_expired(deadline, "fixed-cover beam")
            input_positions = tuple(
                path_position[value] for value in alternative.input_physical
            )
            output_positions = tuple(
                path_position[value] for value in alternative.output_physical
            )
            base = min(input_positions)
            key = (
                tuple(value - base for value in input_positions),
                tuple(value - base for value in output_positions),
            )
            previous = grouped.get(key)
            if previous is None or (
                alternative.cnot_count,
                alternative.single_qubit_count,
            ) < (
                previous.cnot_count,
                previous.single_qubit_count,
            ):
                grouped[key] = alternative
        relative[partition] = tuple(sorted(grouped.items()))

    trivial_alternatives = {}
    for partition, values in alternatives.items():
        _raise_if_seed_deadline_expired(deadline, "fixed-cover beam")
        if values and len(values[0].logical_qubits) == 1:
            trivial_alternatives[partition] = min(
                values,
                key=lambda value: (
                    value.cnot_count,
                    value.single_qubit_count,
                    value.input_physical,
                    value.output_physical,
                ),
            )

    def complete_mapping(source, logicals, desired):
        fixed = dict(zip(logicals, desired))
        if source is None:
            remaining_logicals = [
                logical
                for logical in range(logical_qubit_count)
                if logical not in fixed
            ]
        else:
            physical_to_logical = {
                physical: logical for logical, physical in enumerate(source)
            }
            remaining_logicals = [
                physical_to_logical[physical]
                for physical in path
                if physical_to_logical[physical] not in fixed
            ]
        remaining_physical = [
            physical for physical in path if physical not in set(desired)
        ]
        result = [-1] * logical_qubit_count
        for logical, physical in fixed.items():
            result[logical] = physical
        for logical, physical in zip(remaining_logicals, remaining_physical):
            result[logical] = physical
        return tuple(result)

    # cost, one-qubit tie, mapping, initial mapping, selections, swaps,
    # completed-partition mask.  The completed mask is part of the beam state:
    # choosing one fixed topological order misses useful schedules of
    # independent blocks and can introduce avoidable mapping transitions.
    if initial_mapping_candidates is None:
        initial_mapping_candidates = (initial_mapping,)

    def drain_ready_single_qubit_partitions(state):
        """Execute every ready mapping-neutral block without spending beam width."""
        cost, single, current, initial, selections, swap_blocks, completed = state
        while True:
            _raise_if_seed_deadline_expired(deadline, "fixed-cover beam")
            ready_trivial = [
                partition
                for partition in range(len(partitions))
                if not completed & (1 << partition)
                and predecessor_masks[partition] & ~completed == 0
                and partition in trivial_alternatives
            ]
            if not ready_trivial:
                break
            partition = min(
                ready_trivial,
                key=lambda value: (min(partitions[value]), value),
            )
            alternative = trivial_alternatives[partition]
            logical = alternative.logical_qubits[0]
            physical = current[logical]
            translated = replace(
                alternative,
                input_physical=(physical,),
                output_physical=(physical,),
            )
            cost += alternative.cnot_count
            single += alternative.single_qubit_count
            selections = (*selections, RoutingSelection(partition, translated))
            swap_blocks = (*swap_blocks, ())
            completed |= 1 << partition
        return (
            cost,
            single,
            current,
            initial,
            selections,
            swap_blocks,
            completed,
        )

    beam = []
    for candidate_mapping in initial_mapping_candidates:
        candidate_mapping = (
            tuple(range(logical_qubit_count))
            if candidate_mapping is None
            else tuple(map(int, candidate_mapping))
        )
        beam.append(drain_ready_single_qubit_partitions(
            (
                0,
                0,
                candidate_mapping,
                candidate_mapping,
                (),
                (),
                0,
            )
        ))
    while beam[0][6] != all_completed:
        _raise_if_seed_deadline_expired(deadline, "fixed-cover beam")
        next_by_state = {}
        for (
            cost,
            single,
            current,
            initial,
            selections,
            swap_blocks,
            completed,
        ) in beam:
            ready = (
                partition
                for partition in range(len(partitions))
                if not completed & (1 << partition)
                and predecessor_masks[partition] & ~completed == 0
            )
            for partition in ready:
                for (input_offsets, output_offsets), alternative in relative[partition]:
                    width = len(alternative.logical_qubits)
                    starts = range(logical_qubit_count - width + 1)
                    if (
                        translation_limit is not None
                        and len(starts) > translation_limit
                    ):
                        current_positions = tuple(
                            path_position[current[logical]]
                            for logical in alternative.logical_qubits
                        )
                        starts = sorted(
                            starts,
                            key=lambda start: (
                                sum(
                                    abs(
                                        current_positions[index]
                                        - (start + input_offsets[index])
                                    )
                                    for index in range(width)
                                ),
                                start,
                            ),
                        )[:translation_limit]
                    for start in starts:
                        _raise_if_seed_deadline_expired(
                            deadline, "fixed-cover beam"
                        )
                        desired_input = tuple(
                            path[start + value] for value in input_offsets
                        )
                        mapping_in = complete_mapping(
                            current, alternative.logical_qubits, desired_input
                        )
                        swaps = (
                            ()
                            if current is None
                            else _path_mapping_swap_sequence(
                                current, mapping_in, path
                            )
                        )
                        desired_output = tuple(
                            path[start + value] for value in output_offsets
                        )
                        mapping_out = list(mapping_in)
                        for logical, physical in zip(
                            alternative.logical_qubits, desired_output
                        ):
                            mapping_out[logical] = physical
                        mapping_out = tuple(mapping_out)
                        translated = replace(
                            alternative,
                            input_physical=desired_input,
                            output_physical=desired_output,
                        )
                        next_completed = completed | (1 << partition)
                        candidate = drain_ready_single_qubit_partitions((
                            cost + alternative.cnot_count + 3 * len(swaps),
                            single + alternative.single_qubit_count,
                            mapping_out,
                            mapping_in if initial is None else initial,
                            (*selections, RoutingSelection(partition, translated)),
                            (*swap_blocks, tuple(swaps)),
                            next_completed,
                        ))
                        next_completed = candidate[6]
                        mapping_out = candidate[2]
                        state_key = (next_completed, mapping_out)
                        previous = next_by_state.get(state_key)
                        if previous is None or candidate[:2] < previous[:2]:
                            next_by_state[state_key] = candidate
        beam = sorted(
            next_by_state.values(),
            key=lambda value: (value[0], value[1], value[6], value[2]),
        )[:beam_width]
        if not beam:
            raise ValueError("The fixed-cover mapping beam became empty.")
    best = beam[0]
    if best[6] != all_completed:
        raise ValueError("Selected partition quotient contains a cycle.")
    return ExactRoutingResult(
        selections=best[4],
        cnot_count=best[0],
        single_qubit_count=best[1],
        initial_mapping=best[3],
        final_mapping=best[2],
        explored_states=0,
        master_backend="fixed-cover-mapping-beam-mip-start",
        optimal=False,
        transition_swaps=best[5],
    )


def _partition_aware_layout_seed(
    *,
    gate_predecessors,
    partitions,
    alternatives,
    logical_qubit_count,
    topology,
    deadline=None,
):
    """Build a native bidirectional layout from already-priced OSR columns.

    This is deliberately a consumer of ``alternatives`` only.  In particular,
    layout construction must never trigger synthesis: exhaustive exact routing
    prices and caches every symmetry representative before calling this helper.
    """
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    _raise_if_seed_deadline_expired(deadline, "partition-aware layout")
    gate_partition = {
        gate: partition
        for partition, gate_set in enumerate(partition_sets)
        for gate in gate_set
    }
    successors = {partition: set() for partition in range(len(partition_sets))}
    indegree = {partition: 0 for partition in range(len(partition_sets))}
    for gate, dependencies in gate_predecessors.items():
        right = gate_partition[int(gate)]
        for dependency in map(int, dependencies):
            left = gate_partition[dependency]
            if left != right and right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    predecessors = {partition: set() for partition in range(len(partition_sets))}
    for left, right_values in successors.items():
        for right in right_values:
            predecessors[right].add(left)
    if not alternatives:
        return tuple(range(logical_qubit_count))

    adjacency = [set() for _ in range(logical_qubit_count)]
    edges = []
    for left, right in topology:
        left, right = int(left), int(right)
        adjacency[left].add(right)
        adjacency[right].add(left)
        edges.append((min(left, right), max(left, right)))
    edges = tuple(sorted(set(edges)))
    infinity = logical_qubit_count + 1
    distances = [[infinity] * logical_qubit_count for _ in range(logical_qubit_count)]
    for source in range(logical_qubit_count):
        _raise_if_seed_deadline_expired(deadline, "partition-aware layout")
        distances[source][source] = 0
        queue = collections.deque((source,))
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if distances[source][neighbor] == infinity:
                    distances[source][neighbor] = distances[source][current] + 1
                    queue.append(neighbor)
    if any(infinity in row for row in distances):
        raise ValueError("Routing topology must be connected.")

    logicals = {}
    input_sets = {}
    for partition, values in alternatives.items():
        _raise_if_seed_deadline_expired(deadline, "partition-aware layout")
        logicals[partition] = tuple(map(int, values[0].logical_qubits))
        physical_sets = set()
        for alternative_index, value in enumerate(values):
            if alternative_index % 256 == 0:
                _raise_if_seed_deadline_expired(
                    deadline, "partition-aware layout"
                )
            physical_sets.add(frozenset(map(int, value.input_physical)))
        input_sets[partition] = frozenset(physical_sets)

    def block_distance(partition, mapping):
        locations = tuple(mapping[logical] for logical in logicals[partition])
        if len(locations) <= 1:
            return 0.0
        # For a connected k-vertex block this hub-distance score reaches k-1.
        return float(min(
            sum(distances[center][other] for other in locations if other != center)
            for center in locations
        ))

    def extended_set(frontier, following, limit=20):
        answer = set()
        queue = collections.deque(sorted(frontier))
        seen = set(frontier)
        while queue and len(answer) < limit:
            current = queue.popleft()
            for neighbor in sorted(following[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                answer.add(neighbor)
                queue.append(neighbor)
                if len(answer) >= limit:
                    break
        return answer

    def placement_score(mapping, frontier, following):
        active = tuple(sorted(frontier))
        front_score = (
            sum(block_distance(partition, mapping) for partition in active)
            / len(active)
            if active else 0.0
        )
        extended = tuple(sorted(extended_set(frontier, following)))
        lookahead_score = (
            0.5
            * sum(block_distance(partition, mapping) for partition in extended)
            / len(extended)
            if extended else 0.0
        )
        return front_score + lookahead_score

    def swap_mapping(mapping, edge):
        answer = list(mapping)
        left_token = answer.index(edge[0])
        right_token = answer.index(edge[1])
        answer[left_token], answer[right_token] = edge[1], edge[0]
        return answer

    def executable(partition, mapping):
        if len(logicals[partition]) <= 1:
            return True
        occupied = frozenset(mapping[logical] for logical in logicals[partition])
        return occupied in input_sets[partition]

    def traverse(mapping, dependencies, following, permute_blocks):
        remaining = set(range(len(partition_sets)))
        frontier = {
            partition for partition in remaining if not dependencies[partition]
        }
        decay = [1.0] * logical_qubit_count
        swap_count = 0
        visited_since_progress = set()
        while frontier:
            _raise_if_seed_deadline_expired(deadline, "partition-aware layout")
            runnable = tuple(
                partition for partition in sorted(frontier)
                if executable(partition, mapping)
            )
            if runnable:
                for partition in runnable:
                    frontier.remove(partition)
                    remaining.remove(partition)
                for partition in runnable:
                    for neighbor in following[partition]:
                        if neighbor in remaining and not (
                            dependencies[neighbor] & remaining
                        ):
                            frontier.add(neighbor)
                if permute_blocks:
                    for partition in runnable:
                        if len(logicals[partition]) <= 1:
                            continue
                        occupied = frozenset(
                            mapping[logical] for logical in logicals[partition]
                        )
                        candidates = []
                        for alternative_index, alternative in enumerate(
                            alternatives[partition]
                        ):
                            if alternative_index % 64 == 0:
                                _raise_if_seed_deadline_expired(
                                    deadline, "partition-aware layout"
                                )
                            if frozenset(alternative.input_physical) != occupied:
                                continue
                            candidate_mapping = list(mapping)
                            for logical, physical in zip(
                                alternative.logical_qubits,
                                alternative.output_physical,
                            ):
                                candidate_mapping[logical] = physical
                            gate_weight = (
                                0.3 * alternative.cnot_count / len(frontier)
                                if frontier else alternative.cnot_count
                            )
                            candidates.append((
                                placement_score(candidate_mapping, frontier, following)
                                + gate_weight,
                                alternative.cnot_count,
                                alternative.single_qubit_count,
                                alternative_index,
                                tuple(candidate_mapping),
                            ))
                        if not candidates:
                            raise RuntimeError(
                                "No priced OSR column matches an executable block."
                            )
                        mapping[:] = min(candidates)[4]
                decay[:] = [1.0] * logical_qubit_count
                swap_count = 0
                visited_since_progress.clear()
                continue

            state = tuple(mapping)
            visited_since_progress.add(state)
            active_physical = {
                mapping[logical]
                for partition in frontier
                for logical in logicals[partition]
            }
            candidate_edges = tuple(
                edge for edge in edges
                if edge[0] in active_physical or edge[1] in active_physical
            ) or edges
            swaps = []
            for edge in candidate_edges:
                _raise_if_seed_deadline_expired(
                    deadline, "partition-aware layout"
                )
                if any(
                    edge[0] in {
                        mapping[logical] for logical in logicals[partition]
                    }
                    and edge[1] in {
                        mapping[logical] for logical in logicals[partition]
                    }
                    for partition in frontier
                ):
                    continue
                candidate_mapping = swap_mapping(mapping, edge)
                candidate_state = tuple(candidate_mapping)
                swaps.append((
                    candidate_state in visited_since_progress,
                    max(decay[edge[0]], decay[edge[1]])
                    * placement_score(candidate_mapping, frontier, following),
                    edge,
                    candidate_state,
                ))
            if not swaps:
                raise RuntimeError("No topology edge can advance native PAM layout.")
            selected = min(swaps)
            mapping[:] = selected[3]
            decay[selected[2][0]] += 0.001
            decay[selected[2][1]] += 0.001
            swap_count += 1
            if swap_count % 5 == 0:
                decay[:] = [1.0] * logical_qubit_count
        if remaining:
            raise ValueError("Selected partition quotient contains a cycle.")

    mapping = list(range(logical_qubit_count))
    traverse(mapping, predecessors, successors, True)
    traverse(mapping, successors, predecessors, False)
    return tuple(map(int, mapping))


def _cover_selection_warm_start(
    *,
    selected_partitions,
    gate_predecessors,
    partitions,
    alternatives,
    logical_qubit_count,
    path,
    initial_mapping,
    master_backend,
    lookahead=True,
    exact_refine_timeout_seconds=None,
    beam_width=None,
    beam_initial_mappings=None,
    translation_limit=None,
    deadline=None,
    refine_backend="ilp",
):
    """Route one selected cover greedily and restore original partition ids."""
    selected_partitions = tuple(map(int, selected_partitions))
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    reduced_partitions = tuple(
        partition_sets[partition] for partition in selected_partitions
    )
    try:
        reduced_alternatives = {}
        for reduced, original in enumerate(selected_partitions):
            reduced_values = []
            for alternative_index, value in enumerate(alternatives[original]):
                if alternative_index % 256 == 0:
                    _raise_if_seed_deadline_expired(deadline, "fixed-cover")
                reduced_values.append(replace(value, partition=reduced))
            reduced_alternatives[reduced] = tuple(reduced_values)
    except ExactRoutingLimitExceeded:
        return None
    seed_arguments = {
        "gate_predecessors": gate_predecessors,
        "partitions": reduced_partitions,
        "alternatives": reduced_alternatives,
        "logical_qubit_count": logical_qubit_count,
        "path": path,
        "initial_mapping": initial_mapping,
        "deadline": deadline,
    }
    try:
        _raise_if_seed_deadline_expired(deadline, "fixed-cover")
        if beam_width is None:
            reduced = _greedy_selected_path_warm_start(
                **seed_arguments, lookahead=lookahead
            )
        else:
            forward_initial_mappings = tuple(beam_initial_mappings or (None,))
            if initial_mapping is None:
                partition_layout = _partition_aware_layout_seed(
                    gate_predecessors=gate_predecessors,
                    partitions=reduced_partitions,
                    alternatives=reduced_alternatives,
                    logical_qubit_count=logical_qubit_count,
                    topology=tuple(zip(path, path[1:])),
                    deadline=deadline,
                )
                if partition_layout is not None:
                    forward_initial_mappings = tuple(
                        dict.fromkeys(
                            (*forward_initial_mappings, partition_layout)
                        )
                    )
            reduced = _beam_selected_path_warm_start(
                **seed_arguments,
                beam_width=beam_width,
                initial_mapping_candidates=forward_initial_mappings,
                translation_limit=translation_limit,
            )
    except ExactRoutingLimitExceeded:
        return None
    if exact_refine_timeout_seconds is not None:
        refine_seconds = float(exact_refine_timeout_seconds)
        if deadline is not None:
            refine_seconds = min(
                refine_seconds, max(0.0, float(deadline) - time.monotonic())
            )
        if refine_seconds <= 0:
            return replace(
                reduced,
                selections=tuple(
                    RoutingSelection(
                        selected_partitions[selection.partition],
                        replace(
                            selection.alternative,
                            partition=selected_partitions[selection.partition],
                        ),
                    )
                    for selection in reduced.selections
                ),
                master_backend=master_backend,
            )
        try:
            refined = solve_exact_routing(
                backend=refine_backend,
                gate_predecessors=gate_predecessors,
                partitions=reduced_partitions,
                alternatives=reduced_alternatives,
                logical_qubit_count=logical_qubit_count,
                physical_qubit_count=logical_qubit_count,
                topology=tuple(zip(path, path[1:])),
                initial_mapping=initial_mapping,
                timeout_seconds=refine_seconds,
                allow_suboptimal=True,
                warm_start=reduced,
            )
            if (
                refined.cnot_count,
                refined.single_qubit_count,
            ) < (
                reduced.cnot_count,
                reduced.single_qubit_count,
            ):
                reduced = refined
        except ExactRoutingLimitExceeded:
            pass
    return replace(
        reduced,
        selections=tuple(
            RoutingSelection(
                selected_partitions[selection.partition],
                replace(
                    selection.alternative,
                    partition=selected_partitions[selection.partition],
                ),
            )
            for selection in reduced.selections
        ),
        master_backend=master_backend,
    )


def _local_cost_cover_warm_start(
    *,
    gate_predecessors,
    partitions,
    alternatives,
    logical_qubit_count,
    path,
    initial_mapping=None,
    timeout_seconds=10.0,
    deadline=None,
):
    """Find a cheap dependency-valid cover before solving mapping flow.

    This compact model has one binary variable per partition, rather than one
    per routing configuration and stage.  Its only purpose is to give the
    complete Benders model a synthesis-aware feasible incumbent promptly.
    """
    try:
        import gurobipy as gp
    except (ImportError, ModuleNotFoundError):
        return None

    if deadline is None and timeout_seconds is not None:
        deadline = time.monotonic() + float(timeout_seconds)
    _raise_if_seed_deadline_expired(deadline, "local-cost cover")
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    gate_indices = tuple(sorted(map(int, gate_predecessors)))
    gate_to_partitions = {gate: [] for gate in gate_indices}
    usable = []
    for partition, values in sorted(alternatives.items()):
        if not values or partition >= len(partition_sets):
            continue
        usable.append(int(partition))
        for gate in partition_sets[partition]:
            gate_to_partitions[gate].append(int(partition))
    if any(not values for values in gate_to_partitions.values()):
        return None

    successors = {gate: set() for gate in gate_indices}
    for gate, predecessors in gate_predecessors.items():
        for predecessor in map(int, predecessors):
            successors[predecessor].add(int(gate))
    costs = {
        partition: min(
            (value.cnot_count, value.single_qubit_count)
            for value in alternatives[partition]
        )
        for partition in usable
    }
    max_single = 1 + sum(value[1] for value in costs.values())
    try:
        env = gp.Env(params={"OutputFlag": 0})
    except gp.GurobiError:
        return None
    with env, gp.Model(env=env) as model:
        model.Params.LazyConstraints = 1
        model.Params.IntegralityFocus = 1
        solver_seconds = timeout_seconds
        if deadline is not None:
            solver_seconds = max(0.0, float(deadline) - time.monotonic())
        if solver_seconds is not None:
            if solver_seconds <= 0:
                return None
            model.Params.TimeLimit = float(solver_seconds)
        selected = model.addVars(usable, vtype=gp.GRB.BINARY, name="cover")
        for gate in gate_indices:
            model.addConstr(
                gp.quicksum(selected[p] for p in gate_to_partitions[gate]) == 1
            )
        model.setObjective(
            gp.quicksum(
                (costs[p][0] * max_single + costs[p][1]) * selected[p]
                for p in usable
            ),
            gp.GRB.MINIMIZE,
        )
        singleton_by_gate = {
            next(iter(partition_sets[p])): p
            for p in usable
            if len(partition_sets[p]) == 1
        }
        if len(singleton_by_gate) == len(gate_indices):
            singleton_ids = set(singleton_by_gate.values())
            for partition in usable:
                selected[partition].Start = int(partition in singleton_ids)

        def reject_dependency_cycles(callback_model, where):
            if where != gp.GRB.Callback.MIPSOL:
                return
            values = callback_model.cbGetSolution(
                [selected[p] for p in usable]
            )
            chosen = [
                partition
                for partition, value in zip(usable, values)
                if int(round(value))
            ]
            from squander.partitioning.ilp import sol_to_badsccs

            for component in sol_to_badsccs(
                successors, partition_sets, chosen
            ):
                callback_model.cbLazy(
                    gp.quicksum(selected[p] for p in component)
                    <= len(component) - 1
                )

        model.optimize(reject_dependency_cycles)
        if model.SolCount == 0:
            return None
        chosen = [p for p in usable if selected[p].X > 0.5]

    return _cover_selection_warm_start(
        selected_partitions=chosen,
        gate_predecessors=gate_predecessors,
        partitions=partition_sets,
        alternatives=alternatives,
        logical_qubit_count=logical_qubit_count,
        path=path,
        initial_mapping=initial_mapping,
        master_backend="local-cover-greedy-mip-start",
        lookahead=False,
        deadline=deadline,
    )


def _minimum_partition_cover_selection(
    *,
    gate_predecessors,
    partitions,
    weights=None,
):
    """Return the project's exact dependency-valid minimum cover."""
    from squander.partitioning.ilp import ilp_global_optimal

    successors = {int(gate): set() for gate in gate_predecessors}
    for gate, predecessors in gate_predecessors.items():
        for predecessor in map(int, predecessors):
            successors[predecessor].add(int(gate))
    try:
        selected, _fusion = ilp_global_optimal(
            list(partitions),
            successors,
            gurobi_direct=True,
            weights=weights,
        )
    except (ImportError, ModuleNotFoundError):
        selected, _fusion = ilp_global_optimal(
            list(partitions), successors, weights=weights
        )
    except Exception as exc:
        try:
            import gurobipy as gp
        except (ImportError, ModuleNotFoundError):
            gp = None
        if gp is None or not isinstance(exc, gp.GurobiError):
            raise
        selected, _fusion = ilp_global_optimal(
            list(partitions), successors, weights=weights
        )
    return tuple(map(int, selected))


def _dependency_ordered_cover(
    selected_partitions,
    gate_predecessors,
    partitions,
):
    """Return a deterministic quotient-DAG order for one exact cover."""
    selected = tuple(map(int, selected_partitions))
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    gate_partition = {
        gate: partition
        for partition in selected
        for gate in partition_sets[partition]
    }
    if set(gate_partition) != set(map(int, gate_predecessors)):
        raise ValueError("Selected PAM seed partitions are not an exact cover.")
    successors = {partition: set() for partition in selected}
    indegree = {partition: 0 for partition in selected}
    for gate, predecessors in gate_predecessors.items():
        right = gate_partition[int(gate)]
        for predecessor in map(int, predecessors):
            left = gate_partition[predecessor]
            if left != right and right not in successors[left]:
                successors[left].add(right)
                indegree[right] += 1
    ready = sorted(partition for partition, degree in indegree.items() if degree == 0)
    ordered = []
    while ready:
        partition = ready.pop(0)
        ordered.append(partition)
        for successor in sorted(successors[partition]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(ordered) != len(selected):
        raise ValueError("Selected PAM seed partition quotient contains a cycle.")
    return tuple(ordered)


def _squander_to_bqskit(circuit, parameters, *, compensate_endianness):
    """Convert a resolved Squander circuit without invoking synthesis."""
    from bqskit.ir.lang.qasm2 import OPENQASM2Language
    from qiskit import qasm2
    from squander import Qiskit_IO

    flat = circuit.get_Flat_Circuit()
    parameters = np.asarray(parameters, dtype=np.float64)
    if compensate_endianness:
        # PAM's private objective counts multi-qudit operations. Normalize
        # resolved routing columns to the U3+CNOT basis so that count is the
        # exact CNOT cost used by the global router (in particular SWAP=3).
        from squander.utils import circuit_to_CNOT_basis

        flat, parameters = circuit_to_CNOT_basis(flat, parameters)
        flat = flat.get_Flat_Circuit()
    qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit(
        flat, parameters
    )
    converted = OPENQASM2Language().decode(qasm2.dumps(qiskit_circuit))
    width = flat.get_Qbit_Num()
    if compensate_endianness and width > 1:
        converted.renumber_qudits([width - 1 - index for index in range(width)])
    return converted


def _precomputed_osr_pam_warm_start(
    *,
    selected_partitions,
    gate_predecessors,
    partitions,
    alternatives,
    logical_qubit_count,
    topology,
    path,
    layout_passes=3,
    swap_cnot_cost=3.0,
    master_backend="precomputed-osr-pam-mip-start",
):
    """Route one Squander cover with PAM over resolved OSR columns only.

    BQSKit supplies only its permutation-aware mapping heuristic here.  The
    partition blocks, every permutation circuit, and every objective count are
    taken from the exact router's precomputed Squander catalog.  Consequently
    this helper cannot invoke a BQSKit partitioner or synthesis algorithm.
    """
    from bqskit import Circuit as BQSKitCircuit
    from bqskit.ir.point import CircuitPoint
    from bqskit.passes import PAMLayoutPass, PAMRoutingPass
    from bqskit.qis.graph import CouplingGraph

    ordered = _dependency_ordered_cover(
        selected_partitions, gate_predecessors, partitions
    )
    partitioned = BQSKitCircuit(int(logical_qubit_count))
    permutation_data = {}
    table_partition = {}
    transition_lookup = {}

    for partition in ordered:
        values = tuple(alternatives[int(partition)])
        if not values:
            return None
        exemplar = values[0]
        payload = exemplar.payload
        if not isinstance(payload, SynthesizedRoutingPayload):
            return None
        source = _squander_to_bqskit(
            payload.source_circuit,
            payload.source_parameters,
            compensate_endianness=False,
        )
        location = tuple(map(int, exemplar.logical_qubits))
        cycle = partitioned.append_circuit(source, location, True, False)
        point = CircuitPoint(cycle, location[0])
        width = len(location)
        table = {}
        # Physical embeddings duplicate the same local OSR result. Convert
        # each canonical payload once and expand only the tiny local relabeling
        # orbit required by PAM (at most 3! for publication runs).
        canonical_payloads = {}
        for alternative in values:
            alt_payload = alternative.payload
            if not isinstance(alt_payload, SynthesizedRoutingPayload):
                continue
            canonical_key = (
                tuple(map(int, alt_payload.input_assignment)),
                tuple(map(int, alt_payload.output_assignment)),
                tuple(tuple(map(int, edge)) for edge in alt_payload.topology),
                id(alt_payload.circuit),
            )
            canonical_payloads.setdefault(canonical_key, alt_payload)
            transition_lookup[
                (int(partition), alternative.input_physical,
                 alternative.output_physical)
            ] = alternative
        for alt_payload in canonical_payloads.values():
            base = _squander_to_bqskit(
                alt_payload.circuit,
                alt_payload.parameters,
                compensate_endianness=True,
            )
            input_perm = tuple(
                width - 1 - int(value) for value in alt_payload.input_assignment
            )
            output_perm = tuple(
                width - 1 - int(value) for value in alt_payload.output_assignment
            )
            for relabeling in itertools.permutations(range(width)):
                renamed = base.copy()
                renamed.renumber_qudits(relabeling)
                # PAM queries this table with the complete induced hardware
                # subgraph, not merely the subset of edges used by a chosen
                # circuit.  A topology-valid circuit may use only one edge of
                # a three-vertex path; keying by ``renamed.coupling_graph``
                # made that circuit invisible and silently discarded the
                # entire grouped PAM seed.
                graph = CouplingGraph(
                    [
                        (
                            relabeling[int(left)],
                            relabeling[int(right)],
                        )
                        for left, right in alt_payload.topology
                    ],
                    num_qudits=width,
                )
                perms = (
                    tuple(relabeling[index] for index in input_perm),
                    tuple(relabeling[index] for index in output_perm),
                )
                previous = table.setdefault(graph, {}).get(perms)
                if previous is None or (
                    sum(
                        count for gate, count in renamed.gate_counts.items()
                        if gate.num_qudits >= 2
                    ),
                    sum(
                        count for gate, count in renamed.gate_counts.items()
                        if gate.num_qudits == 1
                    ),
                ) < (
                    sum(
                        count for gate, count in previous.gate_counts.items()
                        if gate.num_qudits >= 2
                    ),
                    sum(
                        count for gate, count in previous.gate_counts.items()
                        if gate.num_qudits == 1
                    ),
                ):
                    table[graph][perms] = renamed
        permutation_data[point] = table
        table_partition[id(table)] = int(partition)

    coupling_graph = CouplingGraph(
        [tuple(map(int, edge)) for edge in topology],
        num_qudits=int(logical_qubit_count),
    )
    class _CNOTAwarePAMMixin:
        def _score_perm(self, circuit, frontier, pi, distances, perm, extended):
            mapping_score = super()._score_perm(
                circuit, frontier, pi, distances, perm, extended
            )
            if not frontier:
                return 0.0
            return float(swap_cnot_cost) * mapping_score / len(frontier)

    class _CNOTAwarePAMLayout(_CNOTAwarePAMMixin, PAMLayoutPass):
        def __init__(self, total_passes):
            super().__init__(total_passes=total_passes)
            self.gate_count_weight = 1.0

    layout = _CNOTAwarePAMLayout(max(1, int(layout_passes)))
    initial_mapping = list(range(int(logical_qubit_count)))
    try:
        for _ in range(layout.total_passes):
            layout.forward_pass(
                partitioned, initial_mapping, coupling_graph, permutation_data
            )
            layout.backward_pass(partitioned, initial_mapping, coupling_graph)
    except RuntimeError:
        return None

    class _RecordingPAMRoutingPass(_CNOTAwarePAMMixin, PAMRoutingPass):
        def __init__(self):
            super().__init__()
            self.gate_count_weight = 1.0
            self.selected = []

        def _get_best_perm(self, circuit, block_data, cg, frontier, pi,
                           distances, extended, qudits):
            result = super()._get_best_perm(
                circuit, block_data, cg, frontier, pi, distances, extended, qudits
            )
            partition = table_partition[id(block_data)]
            before = list(pi)
            self._apply_perm(result[0], before)
            input_physical = tuple(before[int(logical)] for logical in qudits)
            full_input_mapping = tuple(map(int, before))
            self._apply_perm(result[2], before)
            output_physical = tuple(before[int(logical)] for logical in qudits)
            full_output_mapping = tuple(map(int, before))
            key = (partition, input_physical, output_physical)
            if key not in transition_lookup:
                raise RuntimeError("PAM selected a transition absent from OSR catalog.")
            self.selected.append((
                partition,
                transition_lookup[key],
                full_input_mapping,
                full_output_mapping,
            ))
            return result

    router = _RecordingPAMRoutingPass()
    routed_mapping = list(initial_mapping)
    try:
        router.forward_pass(
            partitioned, routed_mapping, coupling_graph, permutation_data, False
        )
    except RuntimeError:
        return None

    replay_mapping = list(initial_mapping)
    selections = []
    transition_swaps = []
    total_cnots = 0
    total_single_qubit = 0
    for (
        partition,
        alternative,
        full_input_mapping,
        full_output_mapping,
    ) in router.selected:
        # PAM may insert SWAPs involving logical qubits outside this block.
        # Retain its complete boundary mapping rather than assigning only the
        # block wires, which would duplicate physical vertices occupied by the
        # displaced tokens and discard an otherwise valid seed.
        swaps = _path_mapping_swap_sequence(
            replay_mapping, full_input_mapping, path
        )
        transition_swaps.append(swaps)
        replay_mapping[:] = full_output_mapping
        selections.append(RoutingSelection(partition, alternative))
        total_cnots += 3 * len(swaps) + alternative.cnot_count
        total_single_qubit += alternative.single_qubit_count
    if len(selections) != len(ordered):
        return None
    return ExactRoutingResult(
        selections=tuple(selections),
        cnot_count=int(total_cnots),
        single_qubit_count=int(total_single_qubit),
        initial_mapping=tuple(map(int, initial_mapping)),
        final_mapping=tuple(map(int, replay_mapping)),
        explored_states=0,
        master_backend=str(master_backend),
        optimal=False,
        transition_swaps=tuple(transition_swaps),
    )


def _minimum_partition_cover_warm_start(
    *,
    gate_predecessors,
    partitions,
    alternatives,
    logical_qubit_count,
    path,
    initial_mapping=None,
    exact_refine_timeout_seconds=10.0,
    weights=None,
    master_backend="minimum-partition-fixed-cover-mip-start",
    beam_width=64,
    beam_initial_mappings=None,
):
    """Route an exact minimum-cardinality cover as a compact seed."""
    selected = _minimum_partition_cover_selection(
        gate_predecessors=gate_predecessors,
        partitions=partitions,
        weights=weights,
    )
    return _cover_selection_warm_start(
        selected_partitions=selected,
        gate_predecessors=gate_predecessors,
        partitions=partitions,
        alternatives=alternatives,
        logical_qubit_count=logical_qubit_count,
        path=path,
        initial_mapping=initial_mapping,
        master_backend=master_backend,
        lookahead=True,
        exact_refine_timeout_seconds=exact_refine_timeout_seconds,
        beam_width=beam_width,
        beam_initial_mappings=beam_initial_mappings,
    )


def _solve_fixed_cover_path_order_ilp(
    *,
    gate_predecessors,
    partition_sets,
    templates,
    logical_qubit_count,
    path,
    initial_mapping,
    timeout_seconds,
    allow_suboptimal,
    warm_start,
):
    """Price one exact cover using only total orders on a path.

    A complete mapping onto a path is a permutation, equivalently a transitive
    tournament over logical-qubit pairs. A local synthesis block occupies one
    contiguous interval, permutes only the order inside that interval, and a
    routing transition costs the Kendall distance between consecutive orders.
    This formulation avoids the weak pair of n-by-n assignment matrices at
    every stage in the general model.
    """
    import pulp
    from squander.partitioning.ilp import _solve_pulp_with_gurobi_or_cbc

    partitions = tuple(sorted(templates))
    partition_count = len(partitions)
    stages = range(partition_count)
    logicals = range(logical_qubit_count)
    pairs = tuple(itertools.combinations(logicals, 2))
    triples = tuple(itertools.combinations(logicals, 3))
    representatives = {
        partition: templates[partition][0] for partition in partitions
    }
    gate_partition = {
        gate: partition
        for partition in partitions
        for gate in partition_sets[partition]
    }

    prob = pulp.LpProblem("FixedCoverPathOrderRouting", pulp.LpMinimize)
    at_stage = pulp.LpVariable.dicts(
        "order_stage", (partitions, stages), cat="Binary"
    )
    for partition in partitions:
        prob += pulp.lpSum(at_stage[partition][stage] for stage in stages) == 1
    for stage in stages:
        prob += pulp.lpSum(at_stage[partition][stage] for partition in partitions) == 1

    partition_stage = {
        partition: pulp.lpSum(
            stage * at_stage[partition][stage] for stage in stages
        )
        for partition in partitions
    }
    precedence_edges = set()
    for gate, predecessors in gate_predecessors.items():
        right = gate_partition[int(gate)]
        for predecessor in map(int, predecessors):
            left = gate_partition[predecessor]
            if left != right:
                precedence_edges.add((left, right))
    for left, right in precedence_edges:
        prob += partition_stage[left] + 1 <= partition_stage[right]

    before_in = {
        (stage, left, right): pulp.LpVariable(
            f"order_in_{stage}_{left}_{right}", cat="Binary"
        )
        for stage in stages
        for left, right in pairs
    }
    before_out = {
        (stage, left, right): pulp.LpVariable(
            f"order_out_{stage}_{left}_{right}", cat="Binary"
        )
        for stage in stages
        for left, right in pairs
    }

    def order(values, stage, left, right):
        if left < right:
            return values[stage, left, right]
        return 1 - values[stage, right, left]

    # Triangle facets are sufficient to make every integer tournament a total
    # order and materially strengthen the LP relaxation used for proof.
    for stage in stages:
        for values in (before_in, before_out):
            for left, middle, right in triples:
                cycle = (
                    values[stage, left, middle]
                    + values[stage, middle, right]
                    - values[stage, left, right]
                )
                prob += cycle >= 0
                prob += cycle <= 1

    def conditional_equal(left, right, chosen):
        prob.addConstraint(left - right <= 1 - chosen)
        prob.addConstraint(right - left <= 1 - chosen)

    for partition in partitions:
        input_offsets, output_offsets, representative = representatives[partition]
        involved = tuple(map(int, representative.logical_qubits))
        involved_set = set(involved)
        spectators = tuple(q for q in logicals if q not in involved_set)
        for stage in stages:
            chosen = at_stage[partition][stage]
            # Fix the synthesized input/output order inside the interval.
            for local_left, local_right in itertools.combinations(
                range(len(involved)), 2
            ):
                left = involved[local_left]
                right = involved[local_right]
                expected_in = int(
                    input_offsets[local_left] < input_offsets[local_right]
                )
                expected_out = int(
                    output_offsets[local_left] < output_offsets[local_right]
                )
                input_order = order(before_in, stage, left, right)
                output_order = order(before_out, stage, left, right)
                if expected_in:
                    prob += input_order >= chosen
                else:
                    prob += input_order <= 1 - chosen
                if expected_out:
                    prob += output_order >= chosen
                else:
                    prob += output_order <= 1 - chosen

            # No spectator may lie inside the selected interval: it must have
            # the same side relationship to every involved logical qubit.
            anchor = involved[0]
            for spectator in spectators:
                anchor_side = order(
                    before_in, stage, spectator, anchor
                )
                for participant in involved[1:]:
                    conditional_equal(
                        order(before_in, stage, spectator, participant),
                        anchor_side,
                        chosen,
                    )

            # A local rewrite changes no order relation except those between
            # two participating qubits.
            for left, right in pairs:
                if left in involved_set and right in involved_set:
                    continue
                conditional_equal(
                    order(before_out, stage, left, right),
                    order(before_in, stage, left, right),
                    chosen,
                )

    reversals = {}
    for stage in range(partition_count - 1):
        for left, right in pairs:
            reversal = pulp.LpVariable(
                f"order_reverse_{stage}_{left}_{right}", cat="Binary"
            )
            reversals[stage, left, right] = reversal
            first = before_out[stage, left, right]
            second = before_in[stage + 1, left, right]
            prob += reversal >= first - second
            prob += reversal >= second - first

    initial_reversals = {}
    fixed_initial_before = {}
    if initial_mapping is not None:
        fixed_mapping = tuple(map(int, initial_mapping))
        if sorted(fixed_mapping) != sorted(path):
            raise ValueError("Initial mapping is not a path permutation.")
        path_position = {physical: index for index, physical in enumerate(path)}
        for left, right in pairs:
            fixed = int(
                path_position[fixed_mapping[left]]
                < path_position[fixed_mapping[right]]
            )
            fixed_initial_before[left, right] = fixed
            reversal = pulp.LpVariable(
                f"order_reverse_initial_{left}_{right}", cat="Binary"
            )
            initial_reversals[left, right] = reversal
            first = before_in[0, left, right]
            prob += reversal >= first - fixed
            prob += reversal >= fixed - first
    elif logical_qubit_count > 1:
        # Break the path-reflection symmetry by putting logical zero in the
        # first half of the initial total order.
        rank_zero = pulp.lpSum(
            order(before_in, 0, other, 0)
            for other in logicals
            if other != 0
        )
        prob += rank_zero <= (logical_qubit_count - 1) // 2

    local_cnot = sum(value[2].cnot_count for value in representatives.values())
    local_single = sum(
        value[2].single_qubit_count for value in representatives.values()
    )
    swap_count = pulp.lpSum(
        tuple(initial_reversals.values()) + tuple(reversals.values())
    )
    scale = local_single + 1
    prob.setObjective((local_cnot + 3 * swap_count) * scale + local_single)

    if warm_start is not None:
        warm_records = []
        mapping = list(map(int, warm_start.initial_mapping))
        swaps_by_stage = warm_start.transition_swaps or tuple(
            () for _ in warm_start.selections
        )
        for swaps, selection in zip(swaps_by_stage, warm_start.selections):
            for physical_left, physical_right in swaps:
                logical_left = mapping.index(int(physical_left))
                logical_right = mapping.index(int(physical_right))
                mapping[logical_left], mapping[logical_right] = (
                    mapping[logical_right], mapping[logical_left]
                )
            mapping_in = tuple(mapping)
            for logical, physical in zip(
                selection.alternative.logical_qubits,
                selection.alternative.output_physical,
            ):
                mapping[logical] = physical
            warm_records.append(
                (selection.partition, mapping_in, tuple(mapping))
            )
        path_position = {physical: index for index, physical in enumerate(path)}
        record_by_partition = {
            partition: stage
            for stage, (partition, _mapping_in, _mapping_out)
            in enumerate(warm_records)
        }
        for partition in partitions:
            for stage in stages:
                at_stage[partition][stage].setInitialValue(
                    int(record_by_partition[partition] == stage)
                )
        for stage, (_partition, mapping_in, mapping_out) in enumerate(warm_records):
            for left, right in pairs:
                before_in[stage, left, right].setInitialValue(
                    int(path_position[mapping_in[left]] < path_position[mapping_in[right]])
                )
                before_out[stage, left, right].setInitialValue(
                    int(path_position[mapping_out[left]] < path_position[mapping_out[right]])
                )
        prob += local_cnot + 3 * swap_count <= int(warm_start.cnot_count)

    solve_kwargs = {}
    if timeout_seconds is not None:
        solve_kwargs["timeLimit"] = float(timeout_seconds)
    if warm_start is not None:
        solve_kwargs["warmStart"] = True
    solver_name = _solve_pulp_with_gurobi_or_cbc(
        prob, pulp, callback=None, **solve_kwargs
    )
    model = getattr(prob, "solverModel", None)
    status = pulp.LpStatus[prob.status]
    proven_optimal = status == "Optimal"
    has_incumbent = bool(
        model is not None and getattr(model, "SolCount", 0) >= 1
    )
    if not proven_optimal and not (allow_suboptimal and has_incumbent):
        if status == "Infeasible":
            raise ValueError("The fixed-cover path-order model is infeasible.")
        raise ExactRoutingLimitExceeded(
            f"The {solver_name} path-order oracle did not prove optimality; "
            f"status={status}."
        )

    def mapping_from_order(values, stage):
        ranks = []
        for logical in logicals:
            rank = sum(
                int(round(pulp.value(order(values, stage, other, logical))))
                for other in logicals
                if other != logical
            )
            ranks.append(rank)
        if sorted(ranks) != list(logicals):
            raise AssertionError("The path-order solution is not a permutation.")
        return tuple(path[ranks[logical]] for logical in logicals)

    selections = []
    mappings_in = []
    mappings_out = []
    for stage in stages:
        partition = next(
            p for p in partitions if pulp.value(at_stage[p][stage]) > 0.5
        )
        input_offsets, output_offsets, representative = representatives[partition]
        mapping_in = mapping_from_order(before_in, stage)
        mapping_out = mapping_from_order(before_out, stage)
        alternative = replace(
            representative,
            input_physical=tuple(
                mapping_in[logical] for logical in representative.logical_qubits
            ),
            output_physical=tuple(
                mapping_out[logical] for logical in representative.logical_qubits
            ),
        )
        selections.append(RoutingSelection(partition, alternative))
        mappings_in.append(mapping_in)
        mappings_out.append(mapping_out)

    initial_result = (
        mappings_in[0]
        if initial_mapping is None
        else tuple(map(int, initial_mapping))
    )
    transition_swaps = []
    previous = initial_result
    for mapping_in, mapping_out in zip(mappings_in, mappings_out):
        transition_swaps.append(
            _path_mapping_swap_sequence(previous, mapping_in, path)
        )
        previous = mapping_out
    cnot_count = local_cnot + 3 * sum(map(len, transition_swaps))
    objective_bound = (
        None if model is None else float(getattr(model, "ObjBound", 0)) / scale
    )
    return ExactRoutingResult(
        selections=tuple(selections),
        cnot_count=cnot_count,
        single_qubit_count=local_single,
        initial_mapping=initial_result,
        final_mapping=mappings_out[-1],
        explored_states=int(getattr(model, "NodeCount", 0) or 0),
        master_backend=f"pulp-{solver_name}-fixed-cover-path-order",
        optimal=proven_optimal,
        transition_swaps=tuple(transition_swaps),
        solver_nodes=(None if model is None else float(getattr(model, "NodeCount", 0))),
        solver_bound=objective_bound,
        solver_gap=(None if model is None else float(getattr(model, "MIPGap", 0))),
        solver_solutions=(None if model is None else int(getattr(model, "SolCount", 0))),
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
    warm_start: ExactRoutingResult | None = None,
    _model_cache: dict | None = None,
    _model_cache_key=None,
) -> ExactRoutingResult:
    """Solve globally routed exact cover on a path, including arbitrary SWAPs.

    Selected gate partitions are assigned to global execution stages.  Each
    stage has a complete input and output permutation.  A local synthesis
    changes only its participating logical qubits; the minimum transition cost
    between consecutive complete mappings is their Kendall-tau distance, which
    is exactly the adjacent-SWAP distance on a path.
    """
    del gate_qubits, _model_cache, _model_cache_key
    if max_states is not None:
        raise ValueError(
            "exact_routing_max_states applies only to branch-and-bound; "
            "use exact_routing_timeout_seconds for the ILP master."
        )
    import pulp
    from squander.partitioning.ilp import _solve_pulp_with_gurobi_or_cbc

    logical_qubit_count = int(logical_qubit_count)
    physical_qubit_count = int(
        logical_qubit_count
        if physical_qubit_count is None
        else physical_qubit_count
    )
    if physical_qubit_count != logical_qubit_count:
        raise NotImplementedError(
            "Globally exact path routing currently requires equal logical and "
            "physical widths."
        )
    if topology is None:
        path = tuple(range(physical_qubit_count))
        topology = tuple(zip(path, path[1:]))
    else:
        topology = tuple(tuple(map(int, edge)) for edge in topology)
        path = _path_topology_order(topology, physical_qubit_count)
        if path is None:
            raise NotImplementedError(
                "Globally exact SWAP-cost routing is currently implemented for "
                "path topologies only."
            )
    path = tuple(path)
    path_position = {physical: position for position, physical in enumerate(path)}

    gate_indices = sorted(map(int, gate_predecessors))
    if gate_indices != list(range(len(gate_indices))):
        raise ValueError("Gate indices must be contiguous from zero.")
    gate_count = len(gate_indices)
    if gate_count == 0:
        identity = tuple(path)
        return ExactRoutingResult((), 0, 0, identity, identity, 0, "pulp-path")
    partition_sets = tuple(frozenset(map(int, part)) for part in partitions)
    if any(not part for part in partition_sets):
        raise ValueError("Empty routing partitions are not allowed.")
    gate_to_partitions = [[] for _ in gate_indices]
    for partition, gate_set in enumerate(partition_sets):
        for gate in gate_set:
            if gate < 0 or gate >= gate_count:
                raise ValueError("Routing partition references an unknown gate.")
            gate_to_partitions[gate].append(partition)

    # Collapse translated embeddings of the same local synthesis transition.
    # The path interval start remains a compact integer decision variable.
    templates = {}
    logicals_by_partition = {}
    for partition, values in sorted(alternatives.items()):
        partition = int(partition)
        if partition < 0 or partition >= len(partition_sets):
            raise ValueError(f"Unknown partition alternative key {partition}.")
        grouped = {}
        logical_set = None
        for alternative in values:
            if alternative.partition != partition:
                raise ValueError("Routing alternative partition id mismatch.")
            logical_qubits = tuple(map(int, alternative.logical_qubits))
            if logical_set is None:
                logical_set = logical_qubits
            elif logical_set != logical_qubits:
                raise ValueError(
                    "Alternatives of one partition disagree on logical-qubit order."
                )
            input_positions = tuple(
                path_position[int(value)] for value in alternative.input_physical
            )
            output_positions = tuple(
                path_position[int(value)] for value in alternative.output_physical
            )
            occupied = sorted(input_positions)
            if occupied != list(range(occupied[0], occupied[0] + len(occupied))):
                raise ValueError("A path alternative does not occupy one interval.")
            if sorted(output_positions) != occupied:
                raise ValueError("A path alternative changes its physical support.")
            key = (
                tuple(value - occupied[0] for value in input_positions),
                tuple(value - occupied[0] for value in output_positions),
            )
            previous = grouped.get(key)
            if previous is None or (
                alternative.cnot_count,
                alternative.single_qubit_count,
            ) < (previous.cnot_count, previous.single_qubit_count):
                grouped[key] = alternative
        if grouped:
            logicals_by_partition[partition] = logical_set
            templates[partition] = tuple(
                (key[0], key[1], alternative)
                for key, alternative in sorted(grouped.items())
            )
    if any(not any(partition in templates for partition in choices) for choices in gate_to_partitions):
        raise ValueError("Every gate must have a synthesized routing alternative.")

    partitions_with_templates = tuple(sorted(templates))
    available_gate_partitions = {
        gate: tuple(
            partition
            for partition in gate_to_partitions[gate]
            if partition in templates
        )
        for gate in gate_indices
    }
    fixed_exact_cover = (
        all(len(values) == 1 for values in available_gate_partitions.values())
        and {
            values[0] for values in available_gate_partitions.values()
        }
        == set(partitions_with_templates)
    )
    if fixed_exact_cover and all(
        len(templates[partition]) == 1
        for partition in partitions_with_templates
    ):
        return _solve_fixed_cover_path_order_ilp(
            gate_predecessors=gate_predecessors,
            partition_sets=partition_sets,
            templates=templates,
            logical_qubit_count=logical_qubit_count,
            path=path,
            initial_mapping=initial_mapping,
            timeout_seconds=timeout_seconds,
            allow_suboptimal=allow_suboptimal,
            warm_start=warm_start,
        )
    # A Benders routing oracle supplies an already-selected exact cover. Its
    # schedule has one stage per partition, not one stage per source gate.
    # This removes the dominant O(gates * qubits^2) mapping variables without
    # changing the complete joint model used by the standalone backend.
    stage_count = (
        len(partitions_with_templates) if fixed_exact_cover else gate_count
    )
    stages = range(stage_count)
    prob = pulp.LpProblem("GlobalPathExactRouting", pulp.LpMinimize)
    selected = pulp.LpVariable.dicts(
        "selected", list(partitions_with_templates), cat="Binary"
    )
    at_stage = pulp.LpVariable.dicts(
        "stage",
        (partitions_with_templates, stages),
        cat="Binary",
    )
    config = {
        (partition, index): pulp.LpVariable(
            f"config_{partition}_{index}", cat="Binary"
        )
        for partition in partitions_with_templates
        for index in range(len(templates[partition]))
    }
    interval_start = {
        partition: pulp.LpVariable(
            f"interval_{partition}",
            lowBound=0,
            upBound=physical_qubit_count - len(logicals_by_partition[partition]),
            cat="Integer",
        )
        for partition in partitions_with_templates
    }
    active = pulp.LpVariable.dicts("active", stages, cat="Binary")

    for partition in partitions_with_templates:
        prob += pulp.lpSum(at_stage[partition][stage] for stage in stages) == selected[partition]
        prob += pulp.lpSum(
            config[partition, index]
            for index in range(len(templates[partition]))
        ) == selected[partition]
        prob += interval_start[partition] <= (
            physical_qubit_count - len(logicals_by_partition[partition])
        ) * selected[partition]
        if fixed_exact_cover:
            prob += selected[partition] == 1
    for stage in stages:
        prob += pulp.lpSum(
            at_stage[partition][stage] for partition in partitions_with_templates
        ) == active[stage]
        if stage:
            prob += active[stage - 1] >= active[stage]
        if fixed_exact_cover:
            prob += active[stage] == 1
    prob += active[0] == 1
    for gate in gate_indices:
        prob += pulp.lpSum(
            selected[partition]
            for partition in gate_to_partitions[gate]
            if partition in templates
        ) == 1

    gate_stage = {
        gate: pulp.lpSum(
            stage * at_stage[partition][stage]
            for partition in gate_to_partitions[gate]
            if partition in templates
            for stage in stages
        )
        for gate in gate_indices
    }
    for gate, predecessors in gate_predecessors.items():
        gate = int(gate)
        for predecessor in map(int, predecessors):
            same_block = pulp.lpSum(
                selected[partition]
                for partition in partitions_with_templates
                if predecessor in partition_sets[partition]
                and gate in partition_sets[partition]
            )
            prob += (
                gate_stage[predecessor] + 1
                <= gate_stage[gate] + stage_count * same_block
            )

    location_in = pulp.LpVariable.dicts(
        "map_in",
        (stages, range(logical_qubit_count), range(physical_qubit_count)),
        cat="Binary",
    )
    location_out = pulp.LpVariable.dicts(
        "map_out",
        (stages, range(logical_qubit_count), range(physical_qubit_count)),
        cat="Binary",
    )
    location_start = None
    position_start = {}
    if initial_mapping is not None:
        location_start = pulp.LpVariable.dicts(
            "map_start",
            (range(logical_qubit_count), range(physical_qubit_count)),
            cat="Binary",
        )
        for logical in range(logical_qubit_count):
            prob += pulp.lpSum(location_start[logical]) == 1
            position_start[logical] = pulp.lpSum(
                path_position[physical] * location_start[logical][physical]
                for physical in range(physical_qubit_count)
            )
        for physical in range(physical_qubit_count):
            prob += pulp.lpSum(
                location_start[logical][physical]
                for logical in range(logical_qubit_count)
            ) == 1
    position_in = {}
    position_out = {}
    for stage in stages:
        for logical in range(logical_qubit_count):
            prob += pulp.lpSum(location_in[stage][logical]) == 1
            prob += pulp.lpSum(location_out[stage][logical]) == 1
            position_in[stage, logical] = pulp.lpSum(
                path_position[physical] * location_in[stage][logical][physical]
                for physical in range(physical_qubit_count)
            )
            position_out[stage, logical] = pulp.lpSum(
                path_position[physical] * location_out[stage][logical][physical]
                for physical in range(physical_qubit_count)
            )
        for physical in range(physical_qubit_count):
            prob += pulp.lpSum(
                location_in[stage][logical][physical]
                for logical in range(logical_qubit_count)
            ) == 1
            prob += pulp.lpSum(
                location_out[stage][logical][physical]
                for logical in range(logical_qubit_count)
            ) == 1
        # Inactive suffix stages have no routing meaning. Canonicalize both
        # permutations to eliminate (n!)^2 equivalent assignments per stage.
        for logical in range(logical_qubit_count):
            canonical_physical = path[logical]
            for physical in range(physical_qubit_count):
                canonical = int(physical == canonical_physical)
                prob += (
                    location_in[stage][logical][physical]
                    <= active[stage] + canonical
                )
                prob += (
                    location_out[stage][logical][physical]
                    <= active[stage] + canonical
                )

    big_m = physical_qubit_count
    partition_position_in = {}
    partition_position_out = {}
    for partition in partitions_with_templates:
        involved = set(logicals_by_partition[partition])
        for logical in involved:
            partition_position_in[partition, logical] = pulp.LpVariable(
                f"partition_in_{partition}_{logical}",
                lowBound=0,
                upBound=physical_qubit_count - 1,
                cat="Integer",
            )
            partition_position_out[partition, logical] = pulp.LpVariable(
                f"partition_out_{partition}_{logical}",
                lowBound=0,
                upBound=physical_qubit_count - 1,
                cat="Integer",
            )
            # Likewise, an unselected partition has no placement. Pinning its
            # auxiliary positions removes a large family of symmetric values.
            prob += partition_position_in[partition, logical] <= (
                physical_qubit_count - 1
            ) * selected[partition]
            prob += partition_position_out[partition, logical] <= (
                physical_qubit_count - 1
            ) * selected[partition]
        for config_index, (
            input_offsets,
            output_offsets,
            _alternative,
        ) in enumerate(templates[partition]):
            chosen = config[partition, config_index]
            for local, logical in enumerate(logicals_by_partition[partition]):
                slack = big_m * (1 - chosen)
                prob += partition_position_in[partition, logical] - interval_start[partition] - input_offsets[local] <= slack
                prob += interval_start[partition] + input_offsets[local] - partition_position_in[partition, logical] <= slack
                prob += partition_position_out[partition, logical] - interval_start[partition] - output_offsets[local] <= slack
                prob += interval_start[partition] + output_offsets[local] - partition_position_out[partition, logical] <= slack
        for stage in stages:
            for logical in involved:
                slack = big_m * (1 - at_stage[partition][stage])
                prob += position_in[stage, logical] - partition_position_in[partition, logical] <= slack
                prob += partition_position_in[partition, logical] - position_in[stage, logical] <= slack
                prob += position_out[stage, logical] - partition_position_out[partition, logical] <= slack
                prob += partition_position_out[partition, logical] - position_out[stage, logical] <= slack

    for stage in stages:
        for logical in range(logical_qubit_count):
            touches_logical = pulp.lpSum(
                at_stage[partition][stage]
                for partition in partitions_with_templates
                if logical in logicals_by_partition[partition]
            )
            movement_slack = big_m * (1 - active[stage] + touches_logical)
            prob += position_out[stage, logical] - position_in[stage, logical] <= movement_slack
            prob += position_in[stage, logical] - position_out[stage, logical] <= movement_slack

    if initial_mapping is not None:
        fixed_mapping = tuple(map(int, initial_mapping))
        if len(fixed_mapping) != logical_qubit_count:
            raise ValueError("Initial mapping width mismatch.")
        if any(
            physical < -1 or physical >= physical_qubit_count
            for physical in fixed_mapping
        ):
            raise ValueError("Initial mapping contains an invalid physical qubit.")
        assigned = tuple(physical for physical in fixed_mapping if physical >= 0)
        if len(set(assigned)) != len(assigned):
            raise ValueError("Initial mapping assigns one physical qubit twice.")
        for logical, physical in enumerate(fixed_mapping):
            if physical >= 0:
                prob += location_start[logical][physical] == 1
    elif physical_qubit_count > 1:
        prob += position_in[0, 0] <= (physical_qubit_count - 1) // 2

    pairs = tuple(itertools.combinations(range(logical_qubit_count), 2))
    before_start = {}
    before_in = {}
    before_out = {}
    if location_start is not None:
        for left, right in pairs:
            before_start[left, right] = pulp.LpVariable(
                f"before_start_{left}_{right}", cat="Binary"
            )
            before = before_start[left, right]
            prob += position_start[left] - position_start[right] <= -1 + big_m * (1 - before)
            prob += position_start[right] - position_start[left] <= -1 + big_m * before
    for stage in stages:
        for left, right in pairs:
            before_in[stage, left, right] = pulp.LpVariable(
                f"before_in_{stage}_{left}_{right}", cat="Binary"
            )
            before_out[stage, left, right] = pulp.LpVariable(
                f"before_out_{stage}_{left}_{right}", cat="Binary"
            )
            for before, positions in (
                (before_in[stage, left, right], position_in),
                (before_out[stage, left, right], position_out),
            ):
                prob += positions[stage, left] - positions[stage, right] <= -1 + big_m * (1 - before)
                prob += positions[stage, right] - positions[stage, left] <= -1 + big_m * before

    initial_reversals = {}
    if location_start is not None:
        for left, right in pairs:
            reversal = pulp.LpVariable(
                f"reverse_initial_{left}_{right}", cat="Binary"
            )
            initial_reversals[left, right] = reversal
            first = before_start[left, right]
            second = before_in[0, left, right]
            prob += reversal >= first - second
            prob += reversal >= second - first

    reversals = {}
    for stage in range(stage_count - 1):
        for left, right in pairs:
            reversal = pulp.LpVariable(
                f"reverse_{stage}_{left}_{right}", cat="Binary"
            )
            reversals[stage, left, right] = reversal
            first = before_out[stage, left, right]
            second = before_in[stage + 1, left, right]
            prob += reversal >= first - second - (1 - active[stage + 1])
            prob += reversal >= second - first - (1 - active[stage + 1])
            prob += reversal <= active[stage + 1]

    max_single_qubit_cost = 1 + sum(
        max(
            (template[2].single_qubit_count for template in templates[partition]),
            default=0,
        )
        for partition in partitions_with_templates
    )
    synthesized_cnot_cost = pulp.lpSum(
        template[2].cnot_count * config[partition, index]
        for partition in partitions_with_templates
        for index, template in enumerate(templates[partition])
    )
    swap_cnot_cost = 3 * pulp.lpSum(
        tuple(initial_reversals.values()) + tuple(reversals.values())
    )
    single_qubit_cost = pulp.lpSum(
        template[2].single_qubit_count * config[partition, index]
        for partition in partitions_with_templates
        for index, template in enumerate(templates[partition])
    )
    prob.setObjective(
        (synthesized_cnot_cost + swap_cnot_cost) * max_single_qubit_cost
        + single_qubit_cost
    )

    if warm_start is not None:
        # This is a redundant but much tighter relaxation than merely passing
        # a MIP start: no optimum can be worse than a verified feasible seed.
        prob += synthesized_cnot_cost + swap_cnot_cost <= int(
            warm_start.cnot_count
        )
        warm_swaps = warm_start.transition_swaps or tuple(
            () for _selection in warm_start.selections
        )
        if len(warm_swaps) != len(warm_start.selections):
            raise ValueError("Warm-start SWAP trajectory length mismatch.")
        warm_mapping = list(map(int, warm_start.initial_mapping))
        if sorted(warm_mapping) != list(range(physical_qubit_count)):
            raise ValueError("Warm start has an invalid initial mapping.")
        warm_records = []
        for swaps_before, selection in zip(
            warm_swaps, warm_start.selections
        ):
            for left, right in swaps_before:
                left_logical = warm_mapping.index(int(left))
                right_logical = warm_mapping.index(int(right))
                warm_mapping[left_logical], warm_mapping[right_logical] = (
                    warm_mapping[right_logical],
                    warm_mapping[left_logical],
                )
            alternative = selection.alternative
            mapping_in = tuple(warm_mapping)
            if any(
                mapping_in[logical] != physical
                for logical, physical in zip(
                    alternative.logical_qubits,
                    alternative.input_physical,
                )
            ):
                raise ValueError(
                    "Warm-start partition input does not match its mapping."
                )
            for logical, physical in zip(
                alternative.logical_qubits,
                alternative.output_physical,
            ):
                warm_mapping[logical] = physical
            mapping_out = tuple(warm_mapping)
            warm_records.append(
                (selection.partition, alternative, mapping_in, mapping_out)
            )
        if tuple(warm_mapping) != tuple(warm_start.final_mapping):
            raise ValueError("Warm-start final mapping does not replay.")
        if len(warm_records) > stage_count:
            raise ValueError("Warm start uses more stages than the ILP permits.")

        warm_by_partition = {
            partition: (stage, alternative, mapping_in, mapping_out)
            for stage, (
                partition,
                alternative,
                mapping_in,
                mapping_out,
            ) in enumerate(warm_records)
        }
        if len(warm_by_partition) != len(warm_records):
            raise ValueError("Warm start selects one partition more than once.")
        inactive_mapping = tuple(path)
        for partition in partitions_with_templates:
            record = warm_by_partition.get(partition)
            chosen_alternative = None if record is None else record[1]
            selected[partition].setInitialValue(int(record is not None))
            for stage in stages:
                expected = record is not None and record[0] == stage
                at_stage[partition][stage].setInitialValue(int(expected))
            chosen_config = None
            chosen_start = 0
            if chosen_alternative is not None:
                input_positions = tuple(
                    path_position[physical]
                    for physical in chosen_alternative.input_physical
                )
                output_positions = tuple(
                    path_position[physical]
                    for physical in chosen_alternative.output_physical
                )
                chosen_start = min(input_positions)
                chosen_key = (
                    tuple(value - chosen_start for value in input_positions),
                    tuple(value - chosen_start for value in output_positions),
                )
                chosen_config = next(
                    index
                    for index, (input_offsets, output_offsets, _alternative)
                    in enumerate(templates[partition])
                    if (input_offsets, output_offsets) == chosen_key
                )
            interval_start[partition].setInitialValue(chosen_start)
            for index in range(len(templates[partition])):
                config[partition, index].setInitialValue(
                    int(index == chosen_config)
                )
            for logical in logicals_by_partition[partition]:
                if record is None:
                    input_value = output_value = 0
                else:
                    input_value = path_position[record[2][logical]]
                    output_value = path_position[record[3][logical]]
                partition_position_in[partition, logical].setInitialValue(
                    input_value
                )
                partition_position_out[partition, logical].setInitialValue(
                    output_value
                )

        for stage in stages:
            active[stage].setInitialValue(int(stage < len(warm_records)))
            if stage < len(warm_records):
                mapping_in = warm_records[stage][2]
                mapping_out = warm_records[stage][3]
            else:
                mapping_in = mapping_out = inactive_mapping
            for logical in range(logical_qubit_count):
                for physical in range(physical_qubit_count):
                    location_in[stage][logical][physical].setInitialValue(
                        int(mapping_in[logical] == physical)
                    )
                    location_out[stage][logical][physical].setInitialValue(
                        int(mapping_out[logical] == physical)
                    )
            for left, right in pairs:
                before_in[stage, left, right].setInitialValue(
                    int(path_position[mapping_in[left]] < path_position[mapping_in[right]])
                )
                before_out[stage, left, right].setInitialValue(
                    int(path_position[mapping_out[left]] < path_position[mapping_out[right]])
                )

        if location_start is not None:
            start_mapping = tuple(map(int, warm_start.initial_mapping))
            for logical in range(logical_qubit_count):
                for physical in range(physical_qubit_count):
                    location_start[logical][physical].setInitialValue(
                        int(start_mapping[logical] == physical)
                    )
            for left, right in pairs:
                start_before = (
                    path_position[start_mapping[left]]
                    < path_position[start_mapping[right]]
                )
                first_before = (
                    path_position[warm_records[0][2][left]]
                    < path_position[warm_records[0][2][right]]
                )
                before_start[left, right].setInitialValue(int(start_before))
                initial_reversals[left, right].setInitialValue(
                    int(start_before != first_before)
                )
        for stage in range(stage_count - 1):
            for left, right in pairs:
                if stage + 1 >= len(warm_records):
                    reversed_order = False
                else:
                    output_mapping = warm_records[stage][3]
                    next_input_mapping = warm_records[stage + 1][2]
                    reversed_order = (
                        path_position[output_mapping[left]]
                        < path_position[output_mapping[right]]
                    ) != (
                        path_position[next_input_mapping[left]]
                        < path_position[next_input_mapping[right]]
                    )
                reversals[stage, left, right].setInitialValue(
                    int(reversed_order)
                )

    solve_kwargs = {}
    if timeout_seconds is not None:
        solve_kwargs["timeLimit"] = float(timeout_seconds)
    if warm_start is not None:
        solve_kwargs["warmStart"] = True
    solver_name = _solve_pulp_with_gurobi_or_cbc(
        prob, pulp, callback=None, **solve_kwargs
    )
    status = pulp.LpStatus[prob.status]
    model = getattr(prob, "solverModel", None)
    has_incumbent = bool(
        model is not None and getattr(model, "SolCount", 0) >= 1
    )
    proven_optimal = status == "Optimal"
    if not proven_optimal and not (allow_suboptimal and has_incumbent):
        if status == "Infeasible":
            raise ValueError("No globally routed exact-cover solution exists.")
        raise ExactRoutingLimitExceeded(
            f"The {solver_name} global routing ILP did not prove optimality; "
            f"status={status}."
        )

    selected_stages = []
    for stage in stages:
        if pulp.value(active[stage]) < 0.5:
            break
        partition = next(
            partition
            for partition in partitions_with_templates
            if pulp.value(at_stage[partition][stage]) > 0.5
        )
        config_index = next(
            index
            for index in range(len(templates[partition]))
            if pulp.value(config[partition, index]) > 0.5
        )
        input_offsets, output_offsets, representative = templates[partition][config_index]
        start = int(round(pulp.value(interval_start[partition])))
        input_physical = tuple(
            path[start + input_offsets[local]]
            for local in range(len(representative.logical_qubits))
        )
        output_physical = tuple(
            path[start + output_offsets[local]]
            for local in range(len(representative.logical_qubits))
        )
        alternative = replace(
            representative,
            input_physical=input_physical,
            output_physical=output_physical,
        )
        mapping_in = tuple(
            next(
                physical
                for physical in range(physical_qubit_count)
                if pulp.value(location_in[stage][logical][physical]) > 0.5
            )
            for logical in range(logical_qubit_count)
        )
        mapping_out = tuple(
            next(
                physical
                for physical in range(physical_qubit_count)
                if pulp.value(location_out[stage][logical][physical]) > 0.5
            )
            for logical in range(logical_qubit_count)
        )
        selected_stages.append((partition, alternative, mapping_in, mapping_out))

    selections = []
    transition_swaps = []
    if location_start is None:
        initial_result = selected_stages[0][2]
    else:
        initial_result = tuple(
            next(
                physical
                for physical in range(physical_qubit_count)
                if pulp.value(location_start[logical][physical]) > 0.5
            )
            for logical in range(logical_qubit_count)
        )
    previous_mapping = initial_result
    for partition, alternative, mapping_in, mapping_out in selected_stages:
        swaps = _path_mapping_swap_sequence(
            previous_mapping, mapping_in, path
        )
        transition_swaps.append(swaps)
        selections.append(RoutingSelection(partition, alternative))
        replayed = list(mapping_in)
        for logical, physical in zip(
            alternative.logical_qubits, alternative.output_physical
        ):
            replayed[logical] = physical
        if tuple(replayed) != mapping_out:
            raise AssertionError("Selected partition output mapping does not replay.")
        previous_mapping = mapping_out

    cnot_count = sum(value.alternative.cnot_count for value in selections) + 3 * sum(
        len(value) for value in transition_swaps
    )
    return ExactRoutingResult(
        selections=tuple(selections),
        cnot_count=cnot_count,
        single_qubit_count=sum(
            value.alternative.single_qubit_count for value in selections
        ),
        initial_mapping=initial_result,
        final_mapping=selected_stages[-1][3],
        explored_states=int(getattr(model, "NodeCount", 0) or 0),
        master_backend=f"pulp-{solver_name}-global-path",
        optimal=proven_optimal,
        transition_swaps=tuple(transition_swaps),
        solver_nodes=(None if model is None else float(getattr(model, "NodeCount", 0))),
        solver_bound=(None if model is None else float(getattr(model, "ObjBound", 0)) / max_single_qubit_cost),
        solver_gap=(None if model is None else float(getattr(model, "MIPGap", 0))),
        solver_solutions=(None if model is None else int(getattr(model, "SolCount", 0))),
    )


def solve_exact_routing_benders(
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
    warm_start: ExactRoutingResult | None = None,
    subproblem_slice_seconds: float | None = None,
    master_slice_seconds: float | None = 30.0,
    zero_swap_probe_seconds: float | None = 5.0,
    stagnation_seconds: float | None = 120.0,
    minimum_transition_cnot_cost: int = 0,
    zero_swap_local_cnot_lower_bound: int = 0,
    stop_when_cnot_optimal: bool = False,
    **_unused,
) -> ExactRoutingResult:
    """Solve exact routing by lazy synthesis-column/path-routing separation.

    The compact master selects one relative placement/permutation column for
    every exact-cover partition. Gurobi callbacks add dependency-cycle cuts;
    between persistent master resumes, the staged path oracle prices an integer
    cover and adds an exact conditional SWAP-cost cut. Thus Gurobi never starts
    a nested optimization inside its callback, and convergence preserves the
    same global optimum as the monolithic formulation.
    """
    del gate_qubits
    if max_states is not None:
        raise ValueError(
            "exact_routing_max_states does not apply to the Benders master."
        )
    import pulp
    import gurobipy as gp
    from squander.partitioning.ilp import (
        _check_gurobi_available,
        sol_to_badsccs,
    )

    _check_gurobi_available()
    logical_qubit_count = int(logical_qubit_count)
    physical_qubit_count = int(
        logical_qubit_count
        if physical_qubit_count is None
        else physical_qubit_count
    )
    if physical_qubit_count != logical_qubit_count:
        raise NotImplementedError(
            "The exact Benders router currently requires equal device width."
        )
    if topology is None:
        path = tuple(range(physical_qubit_count))
        topology = tuple(zip(path, path[1:]))
    else:
        topology = tuple(tuple(map(int, edge)) for edge in topology)
        path = _path_topology_order(topology, physical_qubit_count)
        if path is None:
            raise NotImplementedError(
                "The exact Benders router currently supports path topologies."
            )
    path = tuple(path)
    path_position = {physical: index for index, physical in enumerate(path)}

    gate_indices = sorted(map(int, gate_predecessors))
    if gate_indices != list(range(len(gate_indices))):
        raise ValueError("Gate indices must be contiguous from zero.")
    gate_count = len(gate_indices)
    if gate_count == 0:
        identity = tuple(path)
        return ExactRoutingResult((), 0, 0, identity, identity, 0, "benders")
    partition_sets = tuple(frozenset(map(int, value)) for value in partitions)
    gate_successors = {gate: set() for gate in gate_indices}
    for gate, predecessors in gate_predecessors.items():
        for predecessor in map(int, predecessors):
            gate_successors[predecessor].add(int(gate))

    # Physical translations of one relative path transition have identical
    # synthesis cost. Keep one master column and let the routing oracle choose
    # its interval, exactly as the staged model does.
    nodes = []
    nodes_by_partition = collections.defaultdict(list)
    node_key = {}
    for partition, values in sorted(alternatives.items()):
        grouped = {}
        for alternative in values:
            positions_in = tuple(
                path_position[int(value)] for value in alternative.input_physical
            )
            positions_out = tuple(
                path_position[int(value)] for value in alternative.output_physical
            )
            start = min(positions_in)
            key = (
                tuple(map(int, alternative.logical_qubits)),
                tuple(value - start for value in positions_in),
                tuple(value - start for value in positions_out),
            )
            previous = grouped.get(key)
            if previous is None or (
                alternative.cnot_count,
                alternative.single_qubit_count,
            ) < (
                previous.cnot_count,
                previous.single_qubit_count,
            ):
                grouped[key] = alternative
        for key, alternative in sorted(grouped.items()):
            node = len(nodes)
            nodes.append(alternative)
            nodes_by_partition[int(partition)].append(node)
            node_key[int(partition), key] = node
    gate_to_nodes = {gate: [] for gate in gate_indices}
    for node, alternative in enumerate(nodes):
        for gate in partition_sets[alternative.partition]:
            gate_to_nodes[gate].append(node)
    if any(not values for values in gate_to_nodes.values()):
        raise ValueError("Every gate needs at least one Benders routing column.")

    minimum_transition_cnot_cost = int(minimum_transition_cnot_cost)
    zero_swap_local_cnot_lower_bound = int(
        zero_swap_local_cnot_lower_bound
    )
    if (
        minimum_transition_cnot_cost < 0
        or minimum_transition_cnot_cost % 3
    ):
        raise ValueError(
            "The path-transition lower bound must be a nonnegative multiple "
            "of three CNOTs."
        )
    if zero_swap_local_cnot_lower_bound < 0:
        raise ValueError("The zero-SWAP local-cost bound cannot be negative.")
    prob = pulp.LpProblem("ExactRoutingBenders", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("column", range(len(nodes)), cat="Binary")
    swap_cost = pulp.LpVariable(
        "transition_cnot_cost",
        lowBound=minimum_transition_cnot_cost,
        cat="Integer",
    )
    swap_count = pulp.LpVariable(
        "transition_swap_count",
        lowBound=minimum_transition_cnot_cost // 3,
        cat="Integer",
    )
    prob += swap_cost == 3 * swap_count
    for gate in gate_indices:
        prob += pulp.lpSum(x[node] for node in gate_to_nodes[gate]) == 1
    max_single = 1 + sum(
        max(
            (nodes[node].single_qubit_count for node in values),
            default=0,
        )
        for values in nodes_by_partition.values()
    )
    local_cnot = pulp.lpSum(
        alternative.cnot_count * x[node]
        for node, alternative in enumerate(nodes)
    )
    local_single = pulp.lpSum(
        alternative.single_qubit_count * x[node]
        for node, alternative in enumerate(nodes)
    )
    if zero_swap_local_cnot_lower_bound:
        # Disjunctive lower envelope: a zero-transition route costs at least Z
        # locally; any positive path transition already costs >=3 CNOTs. This
        # imports the compact flow model's dual certificate without excluding
        # more expensive zero-SWAP routes.
        prob += (
            local_cnot
            + (zero_swap_local_cnot_lower_bound / 3.0) * swap_cost
            >= zero_swap_local_cnot_lower_bound
        )
    prob.setObjective((local_cnot + swap_cost) * max_single + local_single)

    def relative_key(alternative):
        positions_in = tuple(
            path_position[int(value)] for value in alternative.input_physical
        )
        positions_out = tuple(
            path_position[int(value)] for value in alternative.output_physical
        )
        start = min(positions_in)
        return (
            tuple(map(int, alternative.logical_qubits)),
            tuple(value - start for value in positions_in),
            tuple(value - start for value in positions_out),
        )

    warm_nodes = set()
    warm_swap_cost = 0
    if warm_start is not None:
        for selection in warm_start.selections:
            key = (selection.partition, relative_key(selection.alternative))
            if key not in node_key:
                raise ValueError("A warm-start routing column is unavailable.")
            warm_nodes.add(node_key[key])
        warm_swap_cost = 3 * sum(
            len(swaps) for swaps in (warm_start.transition_swaps or ())
        )
        for node in range(len(nodes)):
            x[node].setInitialValue(int(node in warm_nodes))
        swap_cost.setInitialValue(warm_swap_cost)
        swap_count.setInitialValue(warm_swap_cost // 3)
        prob += local_cnot + swap_cost <= (
            sum(nodes[node].cnot_count for node in warm_nodes)
            + warm_swap_cost
        )

    if (
        subproblem_slice_seconds is not None
        and subproblem_slice_seconds <= 0
    ):
        raise ValueError("The Benders subproblem slice must be positive.")
    if master_slice_seconds is not None and master_slice_seconds <= 0:
        raise ValueError("The Benders master slice must be positive.")
    if zero_swap_probe_seconds is not None and zero_swap_probe_seconds <= 0:
        raise ValueError("The Benders zero-SWAP probe must be positive.")
    if stagnation_seconds is not None and stagnation_seconds <= 0:
        raise ValueError("The Benders stagnation limit must be positive.")
    solve_deadline = (
        None
        if timeout_seconds is None
        else time.monotonic() + float(timeout_seconds)
    )
    callback_cache = {}
    callback_lower_bounds = {}
    callback_cycle_cuts = set()
    callback_failure = [None]
    cnot_bound_closed = [False]
    cnot_proof_bound = [None]
    incumbent_cost = [
        (
            int(warm_start.cnot_count),
            int(warm_start.single_qubit_count),
        )
        if warm_start is not None
        else (float("inf"), float("inf"))
    ]
    incumbent_result = [warm_start]
    last_improvement = [time.monotonic()]

    def seconds_remaining():
        now = time.monotonic()
        limits = []
        if solve_deadline is not None:
            limits.append(solve_deadline - now)
        if stagnation_seconds is not None:
            limits.append(
                last_improvement[0] + float(stagnation_seconds) - now
            )
        return None if not limits else max(0.0, min(limits))

    def record_incumbent(result):
        value = (int(result.cnot_count), int(result.single_qubit_count))
        if value < incumbent_cost[0]:
            incumbent_cost[0] = value
            incumbent_result[0] = result
            last_improvement[0] = time.monotonic()

    def remap_oracle_result(reduced_result, original_partitions):
        remapped_selections = tuple(
            RoutingSelection(
                original_partitions[selection.partition],
                replace(
                    selection.alternative,
                    partition=original_partitions[selection.partition],
                ),
            )
            for selection in reduced_result.selections
        )
        return replace(
            reduced_result,
            selections=remapped_selections,
            master_backend="benders-routing-oracle",
        )

    def solve_selected_subproblem(signature):
        if signature in callback_cache:
            return callback_cache[signature]
        selected_nodes = tuple(signature)
        original_partitions = tuple(nodes[node].partition for node in selected_nodes)
        reduced_partitions = tuple(
            partition_sets[partition] for partition in original_partitions
        )
        reduced_alternatives = {
            reduced: (
                replace(nodes[node], partition=reduced),
            )
            for reduced, node in enumerate(selected_nodes)
        }
        # The compact flow oracle does not translate relative path columns on
        # its own, so explicitly expose every interval only for this selected
        # cover. This remains tiny compared with the global master.
        expanded_alternatives = {}
        for reduced, node in enumerate(selected_nodes):
            alternative = nodes[node]
            positions_in = tuple(
                path_position[physical]
                for physical in alternative.input_physical
            )
            positions_out = tuple(
                path_position[physical]
                for physical in alternative.output_physical
            )
            base = min(positions_in)
            input_offsets = tuple(value - base for value in positions_in)
            output_offsets = tuple(value - base for value in positions_out)
            width = len(alternative.logical_qubits)
            expanded_alternatives[reduced] = tuple(
                replace(
                    alternative,
                    partition=reduced,
                    input_physical=tuple(
                        path[start + value] for value in input_offsets
                    ),
                    output_physical=tuple(
                        path[start + value] for value in output_offsets
                    ),
                )
                for start in range(logical_qubit_count - width + 1)
            )
        selected_local = sum(nodes[node].cnot_count for node in selected_nodes)
        selected_single = sum(
            nodes[node].single_qubit_count for node in selected_nodes
        )

        # Zero-SWAP compatibility is exactly the old compact mapping-flow
        # problem. It is both a fast certificate and a rigorous one-SWAP
        # lower bound when infeasible.
        remaining = seconds_remaining()
        if remaining is not None and remaining <= 0:
            raise ExactRoutingLimitExceeded(
                "The Benders routing budget was exhausted in its flow oracle."
            )
        probe_seconds = remaining
        if zero_swap_probe_seconds is not None:
            probe_seconds = (
                float(zero_swap_probe_seconds)
                if remaining is None
                else min(float(zero_swap_probe_seconds), remaining)
            )
        try:
            flow_result = _solve_exact_routing_ilp_flow_restricted(
                gate_predecessors=gate_predecessors,
                partitions=reduced_partitions,
                alternatives=expanded_alternatives,
                logical_qubit_count=logical_qubit_count,
                physical_qubit_count=physical_qubit_count,
                topology=topology,
                initial_mapping=initial_mapping,
                timeout_seconds=probe_seconds,
                allow_suboptimal=False,
            )
        except (ExactRoutingLimitExceeded, ValueError):
            # This is only a fast zero-transition certificate.  A difficult
            # proof must not consume the complete routing budget before the
            # staged oracle has constructed its synthesis-aware incumbent.
            flow_result = None
        if flow_result is not None:
            result = remap_oracle_result(flow_result, original_partitions)
            callback_cache[signature] = (0, result)
            record_incumbent(result)
            return callback_cache[signature]

        one_swap_lower_bound = (selected_local + 3, selected_single)
        if one_swap_lower_bound >= incumbent_cost[0]:
            callback_cache[signature] = ("pruned", None)
            return callback_cache[signature]

        # First reject the zero-cost master incumbent with the universally
        # valid one-SWAP lower bound.  This lets the master eliminate many
        # covers before paying for the complete staged routing oracle.
        if signature not in callback_lower_bounds:
            callback_lower_bounds[signature] = 3
            return 3, None

        reduced_warm_start = _greedy_selected_path_warm_start(
            gate_predecessors=gate_predecessors,
            partitions=reduced_partitions,
            alternatives=reduced_alternatives,
            logical_qubit_count=logical_qubit_count,
            path=path,
            initial_mapping=initial_mapping,
        )
        oracle_warm_start = remap_oracle_result(
            reduced_warm_start, original_partitions
        )
        record_incumbent(oracle_warm_start)
        remaining = seconds_remaining()
        if remaining is not None and remaining <= 0:
            raise ExactRoutingLimitExceeded(
                "The Benders routing budget was exhausted in its oracle."
            )
        if subproblem_slice_seconds is None:
            oracle_seconds = remaining
        elif remaining is None:
            oracle_seconds = float(subproblem_slice_seconds)
        else:
            oracle_seconds = min(float(subproblem_slice_seconds), remaining)
        reduced_result = solve_exact_routing_ilp(
            gate_predecessors=gate_predecessors,
            partitions=reduced_partitions,
            alternatives=reduced_alternatives,
            logical_qubit_count=logical_qubit_count,
            physical_qubit_count=physical_qubit_count,
            topology=topology,
            initial_mapping=initial_mapping,
            timeout_seconds=oracle_seconds,
            allow_suboptimal=True,
            warm_start=reduced_warm_start,
        )
        result = remap_oracle_result(reduced_result, original_partitions)
        transition_cost = result.cnot_count - selected_local
        if transition_cost < 0 or transition_cost % 3:
            raise AssertionError("Routing oracle returned an invalid SWAP cost.")
        record_incumbent(result)
        if result.optimal:
            callback_cache[signature] = (transition_cost, result)
            return callback_cache[signature]

        # An unproven combinatorial subproblem cannot validate a Benders
        # incumbent. Preserve its replayable primal route, terminate the
        # master, and report a bounded (not falsely optimal) result.
        raise ExactRoutingLimitExceeded(
            "A Benders routing oracle exhausted its solve-time budget."
        )

    def cycle_callback(model, where):
        if where == gp.GRB.Callback.MIP:
            if stop_when_cnot_optimal and incumbent_result[0] is not None:
                objective_bound = (
                    model.cbGet(gp.GRB.Callback.MIP_OBJBND) / max_single
                )
                if int(np.floor(objective_bound + 1e-9)) >= int(
                    incumbent_result[0].cnot_count
                    ):
                        cnot_bound_closed[0] = True
                        cnot_proof_bound[0] = objective_bound
                        model.terminate()
            return
        if where != gp.GRB.Callback.MIPSOL:
            return
        try:
            remaining = seconds_remaining()
            if remaining is not None and remaining <= 0:
                model.terminate()
                return
            model_x = [
                model.getVarByName(x[node].name) for node in range(len(nodes))
            ]
            values = model.cbGetSolution(model_x)
            signature = tuple(
                node for node, value in enumerate(values) if int(round(value))
            )
            selected_partitions = {nodes[node].partition for node in signature}
            bad_sccs = sol_to_badsccs(
                gate_successors, partition_sets, selected_partitions
            )
            if bad_sccs:
                for scc in bad_sccs:
                    scc_nodes = tuple(
                        node
                        for partition in scc
                        for node in nodes_by_partition[partition]
                    )
                    cut_signature = tuple(sorted(scc_nodes))
                    if cut_signature in callback_cycle_cuts:
                        continue
                    model.cbLazy(
                        gp.quicksum(model_x[node] for node in scc_nodes)
                        <= len(scc) - 1
                    )
                    callback_cycle_cuts.add(cut_signature)
        except BaseException as exc:
            callback_failure[0] = exc
            model.terminate()

    def decorate_result(result, model, *, optimal):
        solver_bound = (
            float(getattr(model, "ObjBound", 0) or 0) / max_single
        )
        if not np.isfinite(solver_bound):
            solver_bound = (
                float(result.cnot_count)
                + float(result.single_qubit_count) / max_single
                if optimal
                else float(cnot_proof_bound[0])
                if cnot_proof_bound[0] is not None
                else 0.0
            )
        incumbent_objective = (
            float(result.cnot_count)
            + float(result.single_qubit_count) / max_single
        )
        global_gap = (
            0.0
            if optimal
            else max(
                0.0,
                (incumbent_objective - solver_bound)
                / max(abs(incumbent_objective), 1e-12),
            )
        )
        cnot_optimal = bool(
            optimal
            or cnot_bound_closed[0]
            or int(np.floor(solver_bound + 1e-9))
            >= int(result.cnot_count)
        )
        return replace(
            result,
            explored_states=int(getattr(model, "NodeCount", 0) or 0),
            master_backend="pulp-gurobi-benders-path",
            optimal=bool(optimal),
            solver_nodes=float(getattr(model, "NodeCount", 0) or 0),
            solver_bound=solver_bound,
            solver_gap=global_gap,
            solver_solutions=int(getattr(model, "SolCount", 0) or 0),
            cnot_optimal=cnot_optimal,
        )

    solve_kwargs = {
        "manageEnv": True,
        "msg": False,
        "LazyConstraints": 1,
        "IntegralityFocus": 1,
    }
    if timeout_seconds is not None:
        solve_kwargs["timeLimit"] = min(
            float(timeout_seconds),
            float(master_slice_seconds)
            if master_slice_seconds is not None
            else float(timeout_seconds),
        )
    elif master_slice_seconds is not None:
        solve_kwargs["timeLimit"] = float(master_slice_seconds)
    if warm_start is not None:
        solve_kwargs["warmStart"] = True
    prob.solve(pulp.GUROBI(**solve_kwargs), callback=cycle_callback)
    if callback_failure[0] is not None:
        raise callback_failure[0]
    model = prob.solverModel
    model_x = [model.getVarByName(x[node].name) for node in range(len(nodes))]
    model_swap_cost = model.getVarByName(swap_cost.name)
    permanent_cycle_cuts = set()

    def resume_master():
        remaining = seconds_remaining()
        if remaining is not None and remaining <= 0:
            return False
        master_seconds = remaining
        if master_slice_seconds is not None:
            master_seconds = (
                float(master_slice_seconds)
                if remaining is None
                else min(float(master_slice_seconds), remaining)
            )
        if master_seconds is not None:
            model.setParam("TimeLimit", master_seconds)
        model.optimize(cycle_callback)
        return True

    # Logic-based Benders loop. The PuLP-built Gurobi model is retained across
    # resumes; only the compact master is reoptimized. Gurobi does not support
    # safely launching another optimization from a MIPSOL callback, so the
    # exact fixed-cover oracle runs between resumes while the callback remains
    # responsible for lazy dependency-cycle cuts.
    while True:
        if callback_failure[0] is not None:
            raise callback_failure[0]
        master_optimal = model.Status == gp.GRB.OPTIMAL
        has_master_incumbent = int(getattr(model, "SolCount", 0) or 0) > 0
        if not master_optimal and not (
            model.Status == gp.GRB.TIME_LIMIT and has_master_incumbent
        ):
            if (
                model.Status == gp.GRB.INFEASIBLE
                and incumbent_result[0] is not None
            ):
                # Covers whose rigorous lower bound cannot improve the verified
                # incumbent are removed by no-good cuts. If those cuts exhaust
                # the master, the incumbent is lexicographically optimal.
                return decorate_result(
                    incumbent_result[0], model, optimal=True
                )
            if allow_suboptimal and incumbent_result[0] is not None:
                return decorate_result(
                    incumbent_result[0], model, optimal=False
                )
            if model.Status == gp.GRB.INFEASIBLE:
                raise ValueError("No exact Benders routing solution exists.")
            raise ExactRoutingLimitExceeded(
                "The Benders master exhausted its solve-time budget."
            )

        signature = tuple(
            node for node, variable in enumerate(model_x) if variable.X > 0.5
        )
        selected_partitions = {nodes[node].partition for node in signature}
        bad_sccs = sol_to_badsccs(
            gate_successors, partition_sets, selected_partitions
        )
        if bad_sccs:
            for scc in bad_sccs:
                scc_nodes = tuple(
                    node
                    for partition in scc
                    for node in nodes_by_partition[partition]
                )
                cut_signature = tuple(sorted(scc_nodes))
                if cut_signature in permanent_cycle_cuts:
                    raise AssertionError(
                        "A permanent Benders cycle cut was not enforced."
                    )
                model.addConstr(
                    gp.quicksum(model_x[node] for node in scc_nodes)
                    <= len(scc) - 1
                )
                permanent_cycle_cuts.add(cut_signature)
            if resume_master():
                continue
            if allow_suboptimal and incumbent_result[0] is not None:
                return decorate_result(
                    incumbent_result[0], model, optimal=False
                )
            raise ExactRoutingLimitExceeded(
                "The Benders routing budget expired after cycle separation."
            )
        try:
            transition_cost, result = solve_selected_subproblem(signature)
            if transition_cost != "pruned" and result is None:
                # The first positive-SWAP response is deliberately only the
                # rigorous +3 lower bound. If the master already satisfies it
                # (for example via the global zero-SWAP envelope), perform the
                # exact fixed-cover pricing now before accepting the cover.
                transition_cost, result = solve_selected_subproblem(signature)
        except ExactRoutingLimitExceeded:
            if allow_suboptimal and incumbent_result[0] is not None:
                return decorate_result(
                    incumbent_result[0], model, optimal=False
                )
            raise

        indicator = gp.quicksum(model_x[node] for node in signature)
        if transition_cost == "pruned":
            model.addConstr(indicator <= len(signature) - 1)
        elif model_swap_cost.X + 0.5 < transition_cost:
            model.addConstr(
                model_swap_cost
                >= transition_cost * (indicator - len(signature) + 1)
            )
        else:
            if result is None:
                raise AssertionError(
                    "A Benders incumbent met only an unproven routing bound."
                )
            if master_optimal:
                return decorate_result(result, model, optimal=True)
            # The sliced master has supplied a valid but not globally proven
            # cover. Preserve its exact routed result, exclude that fully
            # priced cover, and use the next slice to search for a better one.
            record_incumbent(result)
            model.addConstr(indicator <= len(signature) - 1)

        if not resume_master():
            if allow_suboptimal and incumbent_result[0] is not None:
                return decorate_result(
                    incumbent_result[0], model, optimal=False
                )
            raise ExactRoutingLimitExceeded(
                "The Benders routing budget expired before proving optimality."
            )


def solve_exact_routing(*, backend="ilp", **kwargs) -> ExactRoutingResult:
    """Dispatch to the global path ILP or preserved branch-and-bound model."""
    backend = str(backend).lower().replace("_", "-")
    if backend == "auto":
        try:
            from squander.partitioning.ilp import _check_gurobi_available

            _check_gurobi_available()
            backend = "benders"
        except Exception:
            backend = "branch-and-bound"
    if backend in ("ilp", "pulp"):
        return solve_exact_routing_ilp(**kwargs)
    if backend in ("benders", "lazy-benders"):
        return solve_exact_routing_benders(**kwargs)
    if backend in ("branch-and-bound", "bnb"):
        kwargs.pop("allow_suboptimal", None)
        kwargs.pop("topology", None)
        kwargs.pop("warm_start", None)
        return solve_exact_routing_branch_and_bound(**kwargs)
    raise ValueError(
        "Unknown exact-routing master backend "
        f"{backend!r}; expected 'auto', 'ilp', 'benders', or "
        "'branch-and-bound'."
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
    transition_swaps = result.transition_swaps or tuple(
        () for _selection in result.selections
    )
    if len(transition_swaps) != len(result.selections):
        raise AssertionError("Routing transition-SWAP certificate length mismatch.")
    for swaps_before, selection in zip(transition_swaps, result.selections):
        for edge in swaps_before:
            routed_circuit.add_SWAP(list(edge))
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

    On a path topology, the ILP includes arbitrary adjacent SWAPs between
    selected partitions and minimizes their three-CNOT cost jointly with the
    synthesized blocks.  It is exact over the generated OSR alternatives;
    OSR remains a numerical pricing oracle.  The worst-case state space is
    exponential, so the ILP solve should have a time limit on larger circuits.
    """
    from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
    from squander.partitioning.ilp import get_all_partitions, _get_topo_order

    config = dict(config)
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
    # A gate-by-gate LightSABRE trajectory must always be representable as a
    # complete master solution, independently of which wider convex
    # partitions the enumerator produces.
    candidate_sets.extend(
        frozenset((gate,)) for gate in range(len(circuit.get_Gates()))
    )
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
    # The whole-circuit LightSABRE fallback retains the source circuit's exact
    # serialized gate order.  A different valid topological order is suitable
    # for synthesized local candidates, but would associate this fallback's
    # archived source stream with the wrong input indices during audit replay.
    candidate_gate_orders.append(tuple(range(len(circuit.get_Gates()))))

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

    native_sabre = SABRE(
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
        native_sabre.pi = np.asarray(
            [
                next(available) if value < 0 else value
                for value in requested_initial_mapping
            ],
            dtype=int,
        )
    (
        native_circuit,
        native_parameters,
        native_initial_mapping,
        native_final_mapping,
        _swap_count,
    ) = native_sabre.map_circuit(parameters)
    seed_candidates = [
        (
            "sabre",
            native_circuit.get_Flat_Circuit(),
            np.asarray(native_parameters, dtype=np.float64),
            tuple(map(int, native_initial_mapping)),
            tuple(map(int, native_final_mapping)),
            None,
        )
    ]
    # A caller-fixed input placement must be preserved exactly. LightSABRE's
    # bidirectional layout search deliberately chooses its own initial layout,
    # so it joins the normal unconstrained seed portfolio only.
    if requested_initial_mapping is None:
        seed_candidates.append(
            (
                "light-sabre",
                *_light_sabre_route(circuit, parameters, topology, config),
            )
        )

    normalized_candidates = []
    expected_mapping = tuple(range(circuit.get_Qbit_Num()))
    for (
        seed_strategy,
        seed_circuit,
        seed_parameters,
        seed_initial_mapping,
        seed_final_mapping,
        seed_trace,
    ) in seed_candidates:
        seed_circuit = seed_circuit.get_Flat_Circuit()
        seed_parameters = np.asarray(seed_parameters, dtype=np.float64)
        seed_initial_mapping = tuple(map(int, seed_initial_mapping))
        seed_final_mapping = tuple(map(int, seed_final_mapping))
        _assert_local_topology(seed_circuit, topology)
        if (
            tuple(sorted(seed_initial_mapping)) != expected_mapping
            or tuple(sorted(seed_final_mapping)) != expected_mapping
        ):
            raise AssertionError(
                f"{seed_strategy} seed returned an invalid mapping."
            )
        if requested_initial_mapping is None:
            unreflected_initial_mapping = seed_initial_mapping
            (
                seed_circuit,
                seed_initial_mapping,
                seed_final_mapping,
            ) = _canonicalize_path_reflection(
                seed_circuit,
                seed_initial_mapping,
                seed_final_mapping,
                topology,
            )
            if seed_trace is not None:
                reflection = dict(
                    zip(unreflected_initial_mapping, seed_initial_mapping)
                )
                seed_trace = tuple(
                    (
                        kind,
                        (
                            tuple(reflection[int(value)] for value in payload)
                            if kind == "swap"
                            else payload
                        ),
                    )
                    for kind, payload in seed_trace
                )
            _assert_local_topology(seed_circuit, topology)
        normalized_candidates.append(
            (
                CNOTGateCount(seed_circuit, 0),
                SingleQubitGateCount(seed_circuit),
                seed_strategy,
                seed_circuit,
                seed_parameters,
                seed_initial_mapping,
                seed_final_mapping,
                seed_trace,
            )
        )

    (
        _seed_cnot_count,
        _seed_single_qubit_count,
        seed_strategy,
        sabre_circuit,
        sabre_parameters,
        sabre_initial_mapping,
        sabre_final_mapping,
        _selected_seed_trace,
    ) = min(normalized_candidates, key=lambda candidate: candidate[:3])
    light_sabre_seed = next(
        (
            candidate
            for candidate in normalized_candidates
            if candidate[2] == "light-sabre"
        ),
        None,
    )
    fallback_logical_qubits = tuple(range(circuit.get_Qbit_Num()))
    fallback_alternative = RoutingAlternative(
        partition=fallback_partition,
        logical_qubits=fallback_logical_qubits,
        input_physical=sabre_initial_mapping,
        output_physical=sabre_final_mapping,
        cnot_count=_seed_cnot_count,
        single_qubit_count=_seed_single_qubit_count,
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
    sabre_warm_start = ExactRoutingResult(
        selections=(RoutingSelection(fallback_partition, fallback_alternative),),
        cnot_count=fallback_alternative.cnot_count,
        single_qubit_count=fallback_alternative.single_qubit_count,
        initial_mapping=sabre_initial_mapping,
        final_mapping=sabre_final_mapping,
        explored_states=0,
        master_backend=f"{seed_strategy}-mip-start",
        optimal=False,
    )

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
                master_backend=f"{seed_strategy}-timeout-incumbent",
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

    def finish_with_cnot_optimum(solution):
        solution = replace(
            solution,
            master_backend=f"{solution.master_backend}-cnot-optimal",
            optimal=False,
            cnot_optimal=True,
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
            timed_out=False,
        )

    for partition_index, gate_set in enumerate(local_candidate_sets):
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

    master_warm_start = sabre_warm_start
    if light_sabre_seed is not None:
        master_warm_start = _light_sabre_structural_warm_start(
            trace=light_sabre_seed[7],
            initial_mapping=light_sabre_seed[5],
            partitions=tuple(candidate_sets),
            alternatives=feasible_alternatives,
            topology=topology,
        )

    master_backend = str(config.get("exact_routing_master", "ilp")).lower()
    configured_master_timeout = config.get(
        "exact_routing_timeout_seconds", 20 * 60
    )
    if configured_master_timeout is None:
        master_seconds_remaining = None
    else:
        master_seconds_remaining = float(configured_master_timeout)
        if master_seconds_remaining <= 0:
            raise ValueError("The exact-routing ILP timeout must be positive.")

    configured_cover_seed_seconds = float(
        config.get("exact_routing_cover_seed_timeout_seconds", 10.0)
    )
    if configured_cover_seed_seconds <= 0:
        raise ValueError("The exact-routing cover-seed timeout must be positive.")
    cover_seed_seconds = configured_cover_seed_seconds
    cover_seed_started = time.monotonic()
    cover_seed_deadline = cover_seed_started + cover_seed_seconds

    def cover_seed_seconds_remaining():
        return max(0.0, cover_seed_deadline - time.monotonic())

    path = _path_topology_order(topology, circuit.get_Qbit_Num())
    if path is not None:
        try:
            cover_seed_arguments = {
                "gate_predecessors": {
                    gate: predecessors[gate] for gate in range(len(gate_dict))
                },
                "partitions": local_candidate_sets,
                "alternatives": feasible_alternatives,
                "logical_qubit_count": circuit.get_Qbit_Num(),
                "path": path,
                "initial_mapping": config.get("exact_routing_initial_mapping"),
            }
            # The synthesis-cost cover is useful but redundant with the two
            # exact minimum-cover objectives below. Give it only the first
            # quarter of the bounded portfolio so it cannot starve them.
            local_cover_deadline = min(
                cover_seed_deadline,
                time.monotonic() + cover_seed_seconds / 4.0,
            )
            local_cost_seed = _local_cost_cover_warm_start(
                gate_predecessors={
                    gate: predecessors[gate] for gate in range(len(gate_dict))
                },
                partitions=local_candidate_sets,
                alternatives=feasible_alternatives,
                logical_qubit_count=circuit.get_Qbit_Num(),
                path=path,
                initial_mapping=config.get("exact_routing_initial_mapping"),
                timeout_seconds=max(
                    0.0, local_cover_deadline - time.monotonic()
                ),
                deadline=local_cover_deadline,
            )
            from squander.partitioning.ilp import routing_partition_weights

            plain_cover = _minimum_partition_cover_selection(
                gate_predecessors=cover_seed_arguments["gate_predecessors"],
                partitions=local_candidate_sets,
            )
            routing_weights = routing_partition_weights(
                local_candidate_sets, successors, gate_to_qubit
            )
            routing_cover = _minimum_partition_cover_selection(
                gate_predecessors=cover_seed_arguments["gate_predecessors"],
                partitions=local_candidate_sets,
                weights=routing_weights,
            )
            pam_cover_strategies = tuple(
                config.get(
                    "exact_routing_pam_cover_strategies",
                    ("kahn", "ilp", "ilp-routing"),
                )
            )
            distinct_covers = []
            if "ilp" in pam_cover_strategies:
                distinct_covers.append(("minimum-partition", plain_cover))
            if (
                "ilp-routing" in pam_cover_strategies
                and not any(
                    cover == routing_cover for _name, cover in distinct_covers
                )
            ):
                distinct_covers.append(("routing-weighted", routing_cover))
            if "kahn" in pam_cover_strategies:
                from squander.partitioning.kahn import kahn_partition

                _partitioned, _parameter_order, kahn_parts = kahn_partition(
                    circuit, max_partition_size
                )
                candidate_by_gate_set = {
                    frozenset(map(int, gate_set)): index
                    for index, gate_set in enumerate(local_candidate_sets)
                }
                # Contracted one-qubit chains can make a raw Kahn block differ
                # from the canonical all-partition representation. Preserve
                # the Kahn schedule in that case by expanding only that block
                # into its guaranteed singleton columns; never synthesize a
                # partition outside the precomputed catalog.
                kahn_cover = []
                for part in kahn_parts:
                    gate_set = frozenset(map(int, part))
                    if gate_set in candidate_by_gate_set:
                        kahn_cover.append(candidate_by_gate_set[gate_set])
                    else:
                        kahn_cover.extend(
                            candidate_by_gate_set[frozenset((gate,))]
                            for gate in map(int, part)
                        )
                kahn_cover = tuple(kahn_cover)
                if kahn_cover not in [cover for _name, cover in distinct_covers]:
                    distinct_covers.append(("kahn", kahn_cover))
            precomputed_pam_seeds = []
            if bool(config.get("exact_routing_precomputed_pam_seeds", True)):
                pam_layout_passes = int(
                    config.get("exact_routing_pam_layout_passes", 3)
                )
                if pam_layout_passes <= 0:
                    raise ValueError("PAM layout passes must be positive.")
                for cover_name, selected_cover in distinct_covers:
                    candidate = _precomputed_osr_pam_warm_start(
                        selected_partitions=selected_cover,
                        gate_predecessors=cover_seed_arguments[
                            "gate_predecessors"
                        ],
                        partitions=local_candidate_sets,
                        alternatives=feasible_alternatives,
                        logical_qubit_count=circuit.get_Qbit_Num(),
                        topology=topology,
                        path=path,
                        layout_passes=pam_layout_passes,
                        swap_cnot_cost=float(
                            config.get("pam_swap_cnot_cost", 3.0)
                        ),
                        master_backend=(
                            f"{cover_name}-precomputed-osr-pam-mip-start"
                        ),
                    )
                    if candidate is not None:
                        precomputed_pam_seeds.append(candidate)
            configured_cover_beam_width = int(
                config.get("exact_routing_cover_seed_beam_width", 64)
            )
            cover_beam_width = min(
                configured_cover_beam_width,
                max(1, 1024 // circuit.get_Qbit_Num()),
            )
            translation_limit = int(
                config.get("exact_routing_cover_seed_translation_limit", 8)
            )
            if requested_initial_mapping is None:
                cover_initial_mappings = tuple(
                    dict.fromkeys(
                        candidate[5] for candidate in normalized_candidates
                    )
                ) or (None,)
            else:
                cover_initial_mappings = (
                    tuple(map(int, requested_initial_mapping)),
                )
            minimum_partition_seeds = []
            for cover_index, (cover_name, selected_cover) in enumerate(
                distinct_covers
            ):
                covers_left = len(distinct_covers) - cover_index
                cover_deadline = time.monotonic() + (
                    cover_seed_seconds_remaining() / covers_left
                )
                candidate = _cover_selection_warm_start(
                    selected_partitions=selected_cover,
                    **cover_seed_arguments,
                    master_backend=(
                        f"{cover_name}-fixed-cover-mip-start"
                    ),
                    lookahead=True,
                    exact_refine_timeout_seconds=max(
                        0.0, cover_deadline - time.monotonic()
                    ),
                    beam_width=cover_beam_width,
                    beam_initial_mappings=cover_initial_mappings,
                    translation_limit=translation_limit,
                    deadline=cover_deadline,
                    refine_backend=master_backend,
                )
                if candidate is not None:
                    minimum_partition_seeds.append(candidate)
            master_warm_start = min(
                (
                    candidate
                    for candidate in (
                        master_warm_start,
                        local_cost_seed,
                        *precomputed_pam_seeds,
                        *minimum_partition_seeds,
                    )
                    if candidate is not None
                ),
                key=lambda candidate: (
                    candidate.cnot_count,
                    candidate.single_qubit_count,
                    candidate.master_backend,
                ),
            )
        except ExactRoutingLimitExceeded:
            # LightSABRE is already a complete replayable incumbent. Optional
            # cover seeds improve it only when they finish promptly.
            pass

    global_transition_cnot_lower_bound = 0
    zero_swap_local_cnot_lower_bound = 0
    flow_seed_requested = bool(config.get("exact_routing_flow_seed", True))
    flow_seed_supported = flow_seed_requested and master_backend in (
        "ilp",
        "pulp",
        "benders",
        "lazy-benders",
    )
    flow_seed_max_terms = int(
        config.get("exact_routing_flow_seed_max_terms", 500_000)
    )
    if flow_seed_max_terms <= 0:
        raise ValueError("The flow-seed model term limit must be positive.")
    flow_seed_model_terms = (
        _flow_seed_model_term_count(candidate_sets, feasible_alternatives)
        if flow_seed_supported
        else 0
    )
    flow_seed_allowed = (
        flow_seed_supported and flow_seed_model_terms <= flow_seed_max_terms
    )
    if flow_seed_supported and not flow_seed_allowed:
        print(
            "Skipping auxiliary routing flow seed: "
            f"estimated {flow_seed_model_terms:,} PuLP terms exceeds "
            f"the safe limit {flow_seed_max_terms:,}.",
            flush=True,
        )
    if flow_seed_allowed:
        configured_flow_seconds = float(
            config.get("exact_routing_flow_seed_timeout_seconds", 30.0)
        )
        if configured_flow_seconds <= 0:
            raise ValueError("The flow-seed ILP timeout must be positive.")
        flow_seconds = (
            configured_flow_seconds
            if master_seconds_remaining is None
            else min(configured_flow_seconds, master_seconds_remaining)
        )
        flow_started = time.monotonic()
        try:
            flow_seed = _solve_exact_routing_ilp_flow_restricted(
                gate_predecessors={
                    gate: predecessors[gate] for gate in range(len(gate_dict))
                },
                gate_qubits={
                    gate: gate_to_qubit[gate] for gate in range(len(gate_dict))
                },
                partitions=candidate_sets,
                alternatives=feasible_alternatives,
                logical_qubit_count=circuit.get_Qbit_Num(),
                physical_qubit_count=circuit.get_Qbit_Num(),
                topology=topology,
                initial_mapping=config.get("exact_routing_initial_mapping"),
                timeout_seconds=flow_seconds,
                allow_suboptimal=True,
                warm_start=master_warm_start,
            )
            if (
                flow_seed.cnot_count,
                flow_seed.single_qubit_count,
            ) < (
                master_warm_start.cnot_count,
                master_warm_start.single_qubit_count,
            ):
                master_warm_start = replace(
                    flow_seed,
                    master_backend=(
                        f"{flow_seed.master_backend}-global-mip-start"
                    ),
                    optimal=False,
                )
            flow_bound = _finite_nonnegative_solver_bound(
                flow_seed.solver_bound
            )
            if flow_bound is not None:
                zero_swap_local_cnot_lower_bound = max(
                    zero_swap_local_cnot_lower_bound,
                    int(np.floor(flow_bound)),
                )
        except ExactRoutingLimitExceeded as exc:
            flow_bound = _finite_nonnegative_solver_bound(
                getattr(exc, "solver_bound", None)
            )
            if flow_bound is not None:
                zero_swap_local_cnot_lower_bound = max(
                    zero_swap_local_cnot_lower_bound,
                    int(np.floor(flow_bound)),
                )
        except ValueError:
            # Mapping-flow is only an auxiliary seed model.  In particular,
            # it can reject a replayable zero-transition route whose mapping
            # changes are absorbed by synthesized block permutations.  Its
            # infeasibility is therefore not a valid global +3 CNOT bound.
            pass
        finally:
            if master_seconds_remaining is not None:
                master_seconds_remaining = max(
                    0.0,
                    master_seconds_remaining
                    - (time.monotonic() - flow_started),
                )

    solver_arguments = {
        "backend": master_backend,
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
        "timeout_seconds": master_seconds_remaining,
        "max_states": config.get("exact_routing_max_states"),
        "allow_suboptimal": True,
        "warm_start": master_warm_start,
    }
    if master_backend in ("benders", "lazy-benders"):
        solver_arguments["subproblem_slice_seconds"] = config.get(
            "exact_routing_benders_subproblem_slice_seconds", 60.0
        )
        solver_arguments["master_slice_seconds"] = config.get(
            "exact_routing_benders_master_slice_seconds", 30.0
        )
        solver_arguments["zero_swap_probe_seconds"] = config.get(
            "exact_routing_benders_zero_swap_probe_seconds", 5.0
        )
        solver_arguments["stagnation_seconds"] = config.get(
            "exact_routing_benders_stagnation_seconds", 120.0
        )
        solver_arguments["minimum_transition_cnot_cost"] = (
            global_transition_cnot_lower_bound
        )
        solver_arguments["zero_swap_local_cnot_lower_bound"] = (
            zero_swap_local_cnot_lower_bound
        )
        solver_arguments["stop_when_cnot_optimal"] = (
            not bool(
                config.get("exact_routing_require_tiebreaker_proof", False)
            )
        )
    refined_transition_groups = set()
    lazy_rounds = 0
    refinement_limit = int(
        config.get(
            "exact_routing_max_refinement_rounds",
            max(1, 4 * len(candidate_sets)),
        )
    )

    def solve_master(alternatives, arguments):
        """Charge only PuLP/Gurobi work against the shared master budget."""
        nonlocal master_seconds_remaining
        if (
            master_seconds_remaining is not None
            and master_seconds_remaining <= 0
        ):
            raise ExactRoutingLimitExceeded(
                "The exact-routing ILP exhausted its solver-time budget."
            )
        call_arguments = dict(arguments)
        call_arguments["timeout_seconds"] = master_seconds_remaining
        master_started = time.monotonic()
        try:
            return solve_exact_routing(
                alternatives=alternatives, **call_arguments
            )
        finally:
            if master_seconds_remaining is not None:
                master_seconds_remaining = max(
                    0.0,
                    master_seconds_remaining
                    - (time.monotonic() - master_started),
                )

    if lazy_osr:
        # LightSABRE already supplies a fully replayable incumbent. Start with
        # the optimistic master and price only columns that can improve it;
        # solving the fallback-only master first can consume the entire budget
        # without contributing a single useful OSR synthesis target.
        incumbent_solution = master_warm_start
        solution = incumbent_solution
    else:
        try:
            incumbent_solution = solve_master(
                feasible_alternatives, solver_arguments
            )
        except ExactRoutingLimitExceeded:
            return finish_with_timeout(master_warm_start)
        if not incumbent_solution.optimal:
            if incumbent_solution.cnot_optimal:
                return finish_with_cnot_optimum(incumbent_solution)
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
        try:
            lower_bound_solution = solve_master(
                optimistic_alternatives, lower_solver_arguments
            )
        except ExactRoutingLimitExceeded:
            return finish_with_timeout(incumbent_solution)
        if not lower_bound_solution.optimal:
            if lower_bound_solution.cnot_optimal:
                incumbent_solution = replace(
                    incumbent_solution,
                    solver_bound=lower_bound_solution.solver_bound,
                    solver_gap=lower_bound_solution.solver_gap,
                    solver_nodes=lower_bound_solution.solver_nodes,
                    solver_solutions=lower_bound_solution.solver_solutions,
                )
                return finish_with_cnot_optimum(incumbent_solution)
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
