"""
Wide-circuit optimization: partition large circuits into subcircuits, re-decompose
them, and optionally route or fuse results according to configuration.
"""

from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
    qgd_N_Qubit_Decomposition_adaptive as N_Qubit_Decomposition_adaptive,
    qgd_N_Qubit_Decomposition_Tree_Search as N_Qubit_Decomposition_Tree_Search,
    qgd_N_Qubit_Decomposition_Tabu_Search as N_Qubit_Decomposition_Tabu_Search,
)
from squander import N_Qubit_Decomposition_custom, N_Qubit_Decomposition
from squander.gates.qgd_Circuit import qgd_Circuit as Circuit
from squander.utils import CompareCircuits

import numpy as np
from qiskit import QuantumCircuit

from typing import List, Callable, Tuple, Optional, Set, Dict, Any, cast, Union

import multiprocessing as mp
from multiprocessing import Process, Pool, parent_process
import os, contextlib, collections, time
import gzip
import hashlib
import json
import platform
import struct
import sys
import uuid
from dataclasses import dataclass


from squander.partitioning.partition import PartitionCircuit
from squander.partitioning.tools import translate_param_order, build_dependency
from squander.synthesis.qgd_SABRE import qgd_SABRE as SABRE

try:
    from bqskit.compiler.basepass import BasePass as _BQSKitBasePass
    from bqskit.passes.synthesis.synthesis import SynthesisPass as _BQSKitSynthesisPass
except Exception:
    _BQSKitBasePass = object
    _BQSKitSynthesisPass = object


_SQUANDER_BQSKIT_SYNTHESIS_CONFIG = None

_REWRITE_AUDIT_PATH_ENV = "SQUANDER_REWRITE_AUDIT_JSONL"
REWRITE_AUDIT_SCHEMA_VERSION = 2

_SQUANDER_NATIVE_STRATEGIES = frozenset(
    ("TreeSearch", "TabuSearch", "Adaptive", "Custom")
)

# Squander's optimizer target and the common accepted-rewrite budget both use
# the process-infidelity metric ``1 - F**2``, but serve different purposes.
SQUANDER_FLOAT64_TOLERANCE = 1e-14
# ``use_float`` selects the faster float32 OSR search, but the final
# Hilbert-Schmidt refinement is deliberately performed in float64.
SQUANDER_FLOAT32_TOLERANCE = SQUANDER_FLOAT64_TOLERANCE
# OSR minimizes squared tail singular values. Its relative singular-value rank
# cutoff is derived as sqrt(OSR_OPTIMIZATION_TOLERANCE) in C++, keeping the
# optimizer and classifier on one explicit scale.
OSR_OPTIMIZATION_TOLERANCE = 1e-6
SYNTHESIS_ACCEPTANCE_TOLERANCE = 1e-10
# Whole-circuit state-vector discrepancies accumulate across independently
# accepted partition rewrites. Keep this looser than the per-partition budget;
# it is a validation threshold, not a synthesis acceptance threshold.
CIRCUIT_FLOAT64_VALIDATION_TOLERANCE = 1e-8
CIRCUIT_FLOAT32_VALIDATION_TOLERANCE = 1e-8


def _config_uses_float32(config):
    return bool(config.get("use_float", False))


def _default_squander_tolerance(config):
    return (
        SQUANDER_FLOAT32_TOLERANCE
        if _config_uses_float32(config)
        else SQUANDER_FLOAT64_TOLERANCE
    )


def _default_circuit_validation_tolerance(config):
    return (
        CIRCUIT_FLOAT32_VALIDATION_TOLERANCE
        if _config_uses_float32(config)
        else CIRCUIT_FLOAT64_VALIDATION_TOLERANCE
    )


def _synthesis_acceptance_tolerance(config):
    """Return the common block-rewrite budget in the ``1 - F**2`` metric."""

    return config.get(
        "synthesis_acceptance_tolerance",
        SYNTHESIS_ACCEPTANCE_TOLERANCE,
    )


def _circuit_validation_tolerance(config):
    """Return the allowed whole-circuit infidelity for state-vector checks."""

    return config.get(
        "circuit_validation_tolerance",
        _default_circuit_validation_tolerance(config),
    )


def _trace_infidelity_from_process_infidelity(process_infidelity):
    """Convert ``1 - F**2`` to ``1 - F`` without cancellation."""

    process_infidelity = float(process_infidelity)
    if not 0.0 <= process_infidelity <= 1.0:
        raise ValueError(
            "Process infidelity must be between zero and one, got "
            f"{process_infidelity}."
        )
    return process_infidelity / (
        1.0 + np.sqrt(1.0 - process_infidelity)
    )


def _bqskit_synthesis_epsilon(config):
    """Return BQSKit's exactly equivalent synthesis-success threshold.

    Squander cost-function variant 3 and the rewrite audit use
    ``1 - F**2``. BQSKit's ``HilbertSchmidtCost`` uses ``1 - F``. Here
    ``F = |Tr(U^dagger V)| / d``, so the conversion is exact and independent
    of partition width. BQSKit calls this threshold ``synthesis_epsilon``;
    its numerical optimizer uses separate, tighter ftol/gtol stopping tests.
    """

    return _trace_infidelity_from_process_infidelity(
        _synthesis_acceptance_tolerance(config)
    )


def _copy_bqskit_synthesis_config(config):
    """Copy only plain data needed by BQSKit worker processes."""

    def copy_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            copied = [copy_value(item) for item in value]
            return tuple(item for item in copied if item is not _SKIP_CONFIG_VALUE)
        if isinstance(value, list):
            copied = [copy_value(item) for item in value]
            return [item for item in copied if item is not _SKIP_CONFIG_VALUE]
        if isinstance(value, dict):
            copied = {}
            for key, item in value.items():
                copied_item = copy_value(item)
                if copied_item is not _SKIP_CONFIG_VALUE:
                    copied[key] = copied_item
            return copied
        return _SKIP_CONFIG_VALUE

    copied_config = {}
    for key, value in config.items():
        copied_value = copy_value(value)
        if copied_value is not _SKIP_CONFIG_VALUE:
            copied_config[key] = copied_value
    return copied_config


_SKIP_CONFIG_VALUE = object()


def rewrite_audit_software_versions():
    """Return the execution identity required for bit-exact metric replay."""
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for package in ("numpy", "qiskit", "bqskit", "squander"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }


def _json_complex_matrix(matrix):
    """Return a compact, JSON-safe representation of a complex matrix."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    return {
        "dimension": int(matrix.shape[0]),
        "real": matrix.real.ravel().tolist(),
        "imag": matrix.imag.ravel().tolist(),
    }


def _complex_matrix_from_json(value):
    """Decode a matrix produced by :func:`_json_complex_matrix`."""
    dimension = int(value["dimension"])
    real = np.asarray(value["real"], dtype=np.float64)
    imag = np.asarray(value["imag"], dtype=np.float64)
    return (real + 1j * imag).reshape((dimension, dimension))


def _unitary_audit_metrics(before, after):
    """Compute phase-insensitive equivalence metrics in float64."""
    before = np.asarray(before, dtype=np.complex128)
    after = np.asarray(after, dtype=np.complex128)
    if before.shape != after.shape or before.ndim != 2 or before.shape[0] != before.shape[1]:
        raise AssertionError(
            f"Audit unitary shape mismatch: {before.shape} versus {after.shape}."
        )
    dimension = before.shape[0]
    trace = np.trace(before.conj().T @ after)
    overlap = min(1.0, float(abs(trace) / dimension))
    phase = np.conj(trace) / abs(trace) if abs(trace) else 1.0 + 0.0j
    difference = before - phase * after
    return {
        "trace_overlap": overlap,
        "process_infidelity": max(0.0, 1.0 - overlap * overlap),
        "phase_aligned_frobenius": float(np.linalg.norm(difference, "fro")),
        "phase_aligned_spectral": float(np.linalg.norm(difference, 2)),
    }


def _squander_audit_representation(circuit, parameters, involved_qbits=None):
    """Serialize a Squander block compactly enough for independent replay."""
    from qiskit import qasm2
    from squander import Qiskit_IO

    if involved_qbits is None:
        involved_qbits = sorted(circuit.get_Qbits())
    involved_qbits = list(involved_qbits)
    qbit_map = {qbit: index for index, qbit in enumerate(involved_qbits)}
    compact = circuit.Remap_Qbits(qbit_map, len(involved_qbits)).get_Flat_Circuit()
    parameters = np.asarray(parameters, dtype=np.float64)
    qiskit_circuit = Qiskit_IO.get_Qiskit_Circuit(compact, parameters)
    return {
        "format": "squander_openqasm2",
        "qasm": qasm2.dumps(qiskit_circuit),
        "gate_counts": compact.get_Gate_Nums(),
        "qubits": len(involved_qbits),
    }


def _float64_bits(value):
    """Encode one real number by its IEEE-754 binary64 bits."""
    return struct.pack(">d", float(value)).hex()


def _qasm_exact_state(representation):
    """Return the exact ordered gate/parameter stream represented by QASM."""
    from qiskit import QuantumCircuit

    qasm = representation["qasm"] if isinstance(representation, dict) else representation
    circuit = QuantumCircuit.from_qasm_str(qasm)
    operations = []
    for instruction in circuit.data:
        operation = instruction.operation
        params = []
        for parameter in operation.params:
            value = complex(parameter)
            params.append(
                [_float64_bits(value.real), _float64_bits(value.imag)]
            )
        operations.append(
            {
                "name": operation.name,
                "qubits": [
                    int(circuit.find_bit(qubit).index)
                    for qubit in instruction.qubits
                ],
                "params": params,
            }
        )
    return {"qubits": int(circuit.num_qubits), "operations": operations}


def _exact_state_sha256(state):
    """Hash a canonical exact circuit state."""
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _remap_exact_state(state, qubits, total_qubits):
    """Embed a compact exact state onto global ``qubits``."""
    qubits = [int(qubit) for qubit in qubits]
    if state["qubits"] != len(qubits):
        raise AssertionError(
            f"Cannot embed {state['qubits']}-qubit state on {len(qubits)} qubits."
        )
    def remapped_operation(operation):
        physical_qubits = [
            qubits[index] for index in operation["qubits"]
        ]
        # Squander/QASM serialization canonicalizes the operands of the
        # symmetric SWAP gate after embedding.  Apply the same normalization
        # to archived compact payloads so a reversed local embedding remains
        # bit-identical to the materialized global stream.
        if operation["name"] == "swap":
            physical_qubits.sort()
        return {**operation, "qubits": physical_qubits}

    return {
        "qubits": int(total_qubits),
        "operations": [remapped_operation(operation) for operation in state["operations"]],
    }


def _verify_native_round_replay(event, event_by_id):
    """Rebuild one native round solely from its recorded selected blocks."""
    input_state = _qasm_exact_state(event["input"])
    output_state = _qasm_exact_state(event["output"])
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("Native round input state hash does not match its QASM.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("Native round output state hash does not match its QASM.")

    expected_operations = []
    consumed_indices = []
    for selection in event["selections"]:
        rewrite = event_by_id.get(selection["rewrite_event_id"])
        if rewrite is None or rewrite.get("kind") != "rewrite":
            raise AssertionError(
                f"Missing selected rewrite {selection['rewrite_event_id']}."
            )
        if not rewrite.get("accepted", False):
            raise AssertionError("A native round selected an unaccepted rewrite.")
        source_indices = [int(index) for index in selection["source_gate_indices"]]
        if any(index < 0 or index >= len(input_state["operations"]) for index in source_indices):
            raise AssertionError("Native source gate index is outside the round input.")
        qubits = selection["qubits"]
        before_state = _remap_exact_state(
            _qasm_exact_state(rewrite["before"]), qubits, input_state["qubits"]
        )
        source_operations = [
            input_state["operations"][index] for index in source_indices
        ]
        if source_operations != before_state["operations"]:
            raise AssertionError(
                f"Selected partition {rewrite.get('partition_index')} does not "
                "match its claimed source gates."
            )
        after_state = _remap_exact_state(
            _qasm_exact_state(rewrite["after"]), qubits, input_state["qubits"]
        )
        expected_operations.extend(after_state["operations"])
        consumed_indices.extend(source_indices)

    if sorted(consumed_indices) != list(range(len(input_state["operations"]))):
        raise AssertionError(
            "Native round selections do not cover every input gate exactly once."
        )
    expected_state = {
        "qubits": input_state["qubits"],
        "operations": expected_operations,
    }
    if expected_state != output_state:
        raise AssertionError(
            "Replaying selected native partitions did not reproduce the exact "
            "round output gate and parameter stream."
        )
    return input_state, output_state


def _verify_squander_basis_replay(event):
    """Rerun CNOT-basis conversion and require the exact recorded output."""
    from qiskit import QuantumCircuit
    from squander import Qiskit_IO
    from squander.utils import circuit_to_CNOT_basis

    input_state = _qasm_exact_state(event["input"])
    output_state = _qasm_exact_state(event["output"])
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("Basis conversion input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("Basis conversion output hash mismatch.")
    qiskit_circuit = QuantumCircuit.from_qasm_str(event["input"]["qasm"])
    circuit, parameters = Qiskit_IO.convert_Qiskit_to_Squander(qiskit_circuit)
    converted, converted_parameters = circuit_to_CNOT_basis(circuit, parameters)
    replay_representation = _squander_audit_representation(
        converted,
        converted_parameters,
        range(converted.get_Qbit_Num()),
    )
    if _qasm_exact_state(replay_representation) != output_state:
        raise AssertionError(
            "Replaying CNOT-basis conversion did not reproduce the exact output."
        )
    return input_state, output_state


def _verify_bqskit_to_squander_replay(event):
    """Rerun BQSKit-QASM loading through Qiskit_IO exactly."""
    from qiskit import QuantumCircuit
    from squander import Qiskit_IO

    input_state = _qasm_exact_state(event["input_qasm"])
    output_state = _qasm_exact_state(event["output"])
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("Framework conversion input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("Framework conversion output hash mismatch.")
    qiskit_circuit = QuantumCircuit.from_qasm_str(event["input_qasm"])
    circuit, parameters = Qiskit_IO.convert_Qiskit_to_Squander(qiskit_circuit)
    replay = _squander_audit_representation(
        circuit, parameters, range(circuit.get_Qbit_Num())
    )
    if _qasm_exact_state(replay) != output_state:
        raise AssertionError(
            "BQSKit-to-Squander conversion did not reproduce its exact output."
        )
    return input_state, output_state


def _circuit_dependencies(state):
    """Return predecessor/successor sets for an exact operation stream."""
    predecessors = [set() for _ in state["operations"]]
    successors = [set() for _ in state["operations"]]
    last_on_qubit = {}
    for index, operation in enumerate(state["operations"]):
        for qubit in operation["qubits"]:
            previous = last_on_qubit.get(int(qubit))
            if previous is not None:
                predecessors[index].add(previous)
                successors[previous].add(index)
            last_on_qubit[int(qubit)] = index
    return predecessors, successors


def _qiskit_sabre_actions(input_state, output_state, initial_mapping, topology):
    """Explain a SABRE output as reordered source gates plus physical SWAPs."""
    initial_mapping = [int(qubit) for qubit in initial_mapping]
    width = int(input_state["qubits"])
    if sorted(initial_mapping) != list(range(width)):
        raise AssertionError("SABRE initial mapping is not a permutation.")
    if int(output_state["qubits"]) != width:
        raise AssertionError("SABRE changed the circuit width.")

    topology_edges = {
        frozenset((int(u), int(v))) for u, v in topology
    }
    predecessors, successors = _circuit_dependencies(input_state)
    remaining_predecessors = [len(values) for values in predecessors]
    ready = {
        index
        for index, count in enumerate(remaining_predecessors)
        if count == 0
    }
    mapping = list(initial_mapping)
    actions = []
    consumed = set()

    output_operations = output_state["operations"]

    def consume_source(source_index, action):
        ready.remove(source_index)
        consumed.add(source_index)
        actions.append(action)
        for successor in successors[source_index]:
            remaining_predecessors[successor] -= 1
            if remaining_predecessors[successor] == 0:
                ready.add(successor)

    output_index = 0
    while output_index < len(output_operations):
        output_operation = output_operations[output_index]
        if output_operation["name"] == "swap":
            physical = [int(q) for q in output_operation["qubits"]]
            if (
                len(physical) != 2
                or frozenset(physical) not in topology_edges
                or output_operation["params"]
            ):
                raise AssertionError("SABRE emitted an invalid physical SWAP.")
            logical_a = mapping.index(physical[0])
            logical_b = mapping.index(physical[1])
            mapping[logical_a], mapping[logical_b] = (
                mapping[logical_b], mapping[logical_a]
            )
            actions.append({"kind": "swap", "qubits": physical})
            output_index += 1
            continue

        # Squander's internal SABRE represents SWAP physically as the exact
        # primitive sequence CX(a,b) CX(b,a) CX(a,b).  Recognize it before
        # attempting to match those CNOTs to source gates.
        decomposed_swap = output_operations[output_index : output_index + 3]
        if (
            len(decomposed_swap) == 3
            and [operation["name"] for operation in decomposed_swap]
            == ["cx", "cx", "cx"]
            and not any(operation["params"] for operation in decomposed_swap)
            and not any(
                input_state["operations"][source_index]["name"] == "cx"
                and [
                    mapping[int(q)]
                    for q in input_state["operations"][source_index]["qubits"]
                ]
                == output_operation["qubits"]
                for source_index in ready
            )
        ):
            first = [int(q) for q in decomposed_swap[0]["qubits"]]
            middle = [int(q) for q in decomposed_swap[1]["qubits"]]
            last = [int(q) for q in decomposed_swap[2]["qubits"]]
            if (
                len(first) == 2
                and middle == list(reversed(first))
                and last == first
                and frozenset(first) in topology_edges
            ):
                logical_a = mapping.index(first[0])
                logical_b = mapping.index(first[1])
                mapping[logical_a], mapping[logical_b] = (
                    mapping[logical_b], mapping[logical_a]
                )
                actions.append(
                    {"kind": "swap", "qubits": sorted(first)}
                )
                output_index += 3
                continue

        # Squander's internal SABRE may reverse a directed CNOT with the exact
        # identity H(a) H(b) CX(b,a) H(a) H(b). Treat this as one replayable
        # source-gate action rather than five unexplained output gates.
        direction_rewrite = output_operations[output_index : output_index + 5]
        if (
            len(direction_rewrite) == 5
            and [operation["name"] for operation in direction_rewrite]
            == ["h", "h", "cx", "h", "h"]
            and not any(operation["params"] for operation in direction_rewrite)
        ):
            first_pair = [
                int(direction_rewrite[0]["qubits"][0]),
                int(direction_rewrite[1]["qubits"][0]),
            ]
            last_pair = [
                int(direction_rewrite[3]["qubits"][0]),
                int(direction_rewrite[4]["qubits"][0]),
            ]
            reversed_cnot = [
                int(q) for q in direction_rewrite[2]["qubits"]
            ]
            rewrite_matches = []
            for source_index in ready:
                source = input_state["operations"][source_index]
                mapped = [mapping[int(q)] for q in source["qubits"]]
                if (
                    source["name"] == "cx"
                    and not source["params"]
                    and len(mapped) == 2
                    and first_pair == last_pair
                    and set(first_pair) == set(mapped)
                    and reversed_cnot == list(reversed(mapped))
                    and frozenset(mapped) in topology_edges
                ):
                    rewrite_matches.append(source_index)
            if rewrite_matches:
                source_index = min(rewrite_matches)
                consume_source(
                    source_index,
                    {
                        "kind": "cnot_direction_rewrite",
                        "source_index": source_index,
                    },
                )
                output_index += 5
                continue

        physical = [int(q) for q in output_operation["qubits"]]
        if len(physical) > 1:
            wanted = set(physical)
            seen = {physical[0]}
            frontier = [physical[0]]
            while frontier:
                current = frontier.pop()
                for edge in topology_edges:
                    if current not in edge:
                        continue
                    neighbour = next(iter(edge - {current}))
                    if neighbour in wanted and neighbour not in seen:
                        seen.add(neighbour)
                        frontier.append(neighbour)
            if seen != wanted:
                raise AssertionError(
                    "SABRE emitted a multi-qubit gate outside the topology."
                )

        matches = []
        for source_index in ready:
            source = input_state["operations"][source_index]
            if (
                source["name"] == output_operation["name"]
                and source["params"] == output_operation["params"]
                and [mapping[int(q)] for q in source["qubits"]]
                == output_operation["qubits"]
            ):
                matches.append(source_index)
        if not matches:
            raise AssertionError(
                "SABRE output contains a gate that cannot be matched to a "
                "dependency-ready input gate."
            )
        source_index = min(matches)
        consume_source(
            source_index, {"kind": "gate", "source_index": source_index}
        )
        output_index += 1

    if consumed != set(range(len(input_state["operations"]))):
        raise AssertionError("SABRE output did not consume every input gate.")
    return actions, mapping


def _verify_qiskit_sabre_replay(event):
    """Replay SABRE solely as source-gate scheduling, placement, and SWAPs."""
    input_state = _qasm_exact_state(event["input_qasm"])
    output_state = _qasm_exact_state(event["output_qasm"])
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("SABRE input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("SABRE output hash mismatch.")

    recorded_actions = event["actions"]
    replayed_actions, replayed_final_mapping = _qiskit_sabre_actions(
        input_state,
        output_state,
        event["initial_mapping"],
        event["topology"],
    )
    if replayed_actions != recorded_actions:
        raise AssertionError("Replayed SABRE actions are not bit-identical.")
    if replayed_final_mapping != [int(q) for q in event["final_mapping"]]:
        raise AssertionError("Replayed SABRE final mapping does not match.")
    return input_state, output_state


def _verify_bqskit_foreach_replay(event, event_by_id):
    """Replay one BQSKit ForEachBlockPass from its parent operation list."""
    input_state = event["input"]
    output_state = event["output"]
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("BQSKit invocation input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("BQSKit invocation output hash mismatch.")

    from bqskit import Circuit as BQSKitCircuit

    expected_circuit = BQSKitCircuit(int(input_state["qubits"]))
    for parent_operation in event["parent_operations"]:
        representation = parent_operation["before"]
        rewrite_id = parent_operation.get("rewrite_event_id")
        if parent_operation.get("collected", False):
            rewrite = event_by_id.get(rewrite_id)
            if rewrite is None:
                raise AssertionError(
                    f"Missing BQSKit block rewrite {rewrite_id}."
                )
            recorded_before = _bqskit_representation_exact_state(
                rewrite["before"]
            )
            parent_before = _bqskit_representation_exact_state(representation)
            if recorded_before != parent_before:
                raise AssertionError(
                    "BQSKit rewrite does not match its parent input operation."
                )
            if rewrite.get("accepted", False):
                representation = rewrite["after"]
        elif rewrite_id is not None:
            raise AssertionError("Uncollected BQSKit operation has a rewrite id.")

        local_circuit = _bqskit_circuit_from_representation(representation)
        expected_circuit.append_circuit(
            local_circuit,
            parent_operation["location"],
            as_circuit_gate=True,
        )
    expected_state = _bqskit_exact_state(expected_circuit)
    expected_operations = expected_state["operations"]
    if expected_state != output_state:
        mismatch_index = next(
            (
                index
                for index, (expected, actual) in enumerate(
                    zip(expected_operations, output_state["operations"])
                )
                if expected != actual
            ),
            min(len(expected_operations), len(output_state["operations"])),
        )
        raise AssertionError(
            "Replaying BQSKit's accepted block replacements did not reproduce "
            "the exact invocation output gate and parameter stream for "
            f"{event.get('event_id')}; first mismatch {mismatch_index}, "
            f"expected length {len(expected_operations)}, output length "
            f"{len(output_state['operations'])}."
        )
    return input_state, output_state


def _apply_recorded_permutation(permutation, mapping):
    """Apply BQSKit PAM's logical permutation update to ``mapping``."""
    permutation = [int(q) for q in permutation]
    updated = {
        qubit: mapping[permutation[index]]
        for index, qubit in enumerate(sorted(permutation))
    }
    for qubit in permutation:
        mapping[qubit] = updated[qubit]


def _verify_bqskit_pam_replay(event, event_by_id):
    """Replay PAM routing actions, topology/mapping changes, and chosen blocks."""
    from bqskit import Circuit as BQSKitCircuit

    input_state = event["input"]
    output_state = event["output"]
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("PAM input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("PAM output hash mismatch.")

    blocks = {
        tuple(block["point"]): block for block in event["input_blocks"]
    }
    consumed = set()
    for choice in event["choices"]:
        point = tuple(choice["input_point"])
        block = blocks.get(point)
        if block is None or point in consumed:
            raise AssertionError(f"PAM consumed invalid input block {point}.")
        predecessors = {tuple(value) for value in block["predecessors"]}
        if not predecessors <= consumed:
            raise AssertionError(
                f"PAM executed block {point} before its dependencies."
            )
        choice_mapping = [int(value) for value in choice["mapping_before"]]
        _apply_recorded_permutation(
            choice["pre_perm_global"], choice_mapping
        )
        physical_location = [
            choice_mapping[int(logical)]
            for logical in choice["logical_location"]
        ]
        if physical_location != choice["physical_location"]:
            raise AssertionError(
                f"PAM block {point} has inconsistent physical placement."
            )
        rewrite = event_by_id.get(choice["rewrite_event_id"])
        if rewrite is None or not rewrite.get("accepted", False):
            raise AssertionError("PAM choice lacks an accepted rewrite certificate.")
        from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

        replayed_original = _audit_representation_unitary(block["before"])
        has_recorded_original = "original_unitary" in block
        if has_recorded_original:
            recorded_original = _complex_matrix_from_json(
                block["original_unitary"]
            )
            original_metrics = _unitary_audit_metrics(
                replayed_original, recorded_original
            )
            stored_metrics = block.get("original_unitary_metrics", {})
            for metric_name, metric_value in original_metrics.items():
                stored_value = stored_metrics.get(metric_name)
                stored_bits = stored_metrics.get(f"{metric_name}_bits")
                if (
                    stored_value is None
                    or _float64_bits(stored_value) != stored_bits
                    or stored_bits != _float64_bits(metric_value)
                ):
                    raise AssertionError(
                        f"Stored PAM input-block {metric_name} for {point} "
                        "is not bit-identical when recomputed."
                    )
            if original_metrics["process_infidelity"] > float(
                rewrite["tolerance"]
            ):
                raise AssertionError(
                    f"PAM original unitary for input block {point} does not "
                    "match its replayed gate stream within synthesis tolerance."
                )
            original_unitary = UnitaryMatrix(recorded_original)
        else:
            # Schema-v2 audits written before original_unitary was added can
            # only reconstruct BQSKit's cached value from the block circuit.
            original_unitary = UnitaryMatrix(replayed_original)
        reconstructed_target = _pam_exact_target(
            original_unitary,
            choice["pre_perm_local"],
            choice["post_perm_local"],
        )
        stored_target = _audit_representation_unitary(rewrite["before"])
        if not np.array_equal(reconstructed_target, stored_target):
            if has_recorded_original:
                raise AssertionError(
                    f"PAM target for input block {point} is not bit-identical."
                )
            legacy_metrics = _unitary_audit_metrics(
                reconstructed_target, stored_target
            )
            if legacy_metrics["process_infidelity"] > float(
                rewrite["tolerance"]
            ):
                raise AssertionError(
                    f"Legacy PAM target for input block {point} does not "
                    "match its replayed gate stream within synthesis tolerance."
                )
        consumed.add(point)
    if consumed != set(blocks):
        raise AssertionError("PAM replay did not consume every input block exactly once.")

    trace_mapping = list(range(int(input_state["qubits"])))
    for trace in event["mapping_trace"]:
        if trace_mapping != trace["before"]:
            raise AssertionError("PAM mapping trace is not contiguous.")
        if trace["kind"] == "swap":
            physical_a, physical_b = trace["swap"]
            logical_a = trace_mapping.index(physical_a)
            logical_b = trace_mapping.index(physical_b)
            trace_mapping[logical_a], trace_mapping[logical_b] = (
                trace_mapping[logical_b],
                trace_mapping[logical_a],
            )
        else:
            _apply_recorded_permutation(trace["permutation"], trace_mapping)
        if trace_mapping != trace["after"]:
            raise AssertionError("PAM mapping trace operation is invalid.")
    if trace_mapping != event["local_final_mapping"]:
        raise AssertionError("PAM local final mapping mismatch.")
    composed_mapping = [
        trace_mapping[index] for index in event["mapping_before_pass"]
    ]
    if composed_mapping != event["mapping_after_pass"]:
        raise AssertionError("PAM pass-data mapping composition mismatch.")

    expected_circuit = BQSKitCircuit(int(input_state["qubits"]))
    for action in event["actions"]:
        if action["kind"] == "gate":
            local_circuit = _bqskit_circuit_from_representation(action["circuit"])
            expected_circuit.append_circuit(
                local_circuit, action["location"], as_circuit_gate=True
            )
            continue

        point = tuple(action["input_point"])
        rewrite = event_by_id.get(action["rewrite_event_id"])
        if rewrite is None or not rewrite.get("accepted", False):
            raise AssertionError("PAM action lacks an accepted block certificate.")
        selected = _bqskit_circuit_from_representation(rewrite["after"])
        expected_circuit.append_circuit(
            selected, action["physical_location"], as_circuit_gate=True
        )
    expected_state = _bqskit_exact_state(expected_circuit)
    if expected_state != output_state:
        raise AssertionError(
            "PAM action replay did not reproduce the exact routed gate and "
            "parameter stream."
        )
    return input_state, output_state


def _verify_bqskit_placement_replay(event):
    """Replay an ApplyPlacement wire renumbering exactly."""
    from bqskit import Circuit as BQSKitCircuit

    input_state = event["input"]
    output_state = event["output"]
    placement = [int(q) for q in event["placement"]]
    if len(placement) < input_state["qubits"]:
        raise AssertionError("Placement does not cover every input qubit.")
    expected_circuit = BQSKitCircuit(int(output_state["qubits"]))
    for parent_operation in event["parent_operations"]:
        local_circuit = _bqskit_circuit_from_representation(
            parent_operation["circuit"]
        )
        expected_circuit.append_circuit(
            local_circuit,
            [placement[q] for q in parent_operation["location"]],
            as_circuit_gate=True,
        )
    expected_state = _bqskit_exact_state(expected_circuit)
    if expected_state != output_state:
        raise AssertionError("ApplyPlacement replay did not reproduce its output.")
    for mapping_name in ("initial", "final"):
        expected_mapping = [
            placement[q] for q in event["mapping_before"][mapping_name]
        ]
        if expected_mapping != event["mapping_after"][mapping_name]:
            raise AssertionError(
                f"ApplyPlacement {mapping_name} mapping mismatch."
            )
    return input_state, output_state


def _verify_bqskit_stage_replay(stage_event, all_events, event_by_id):
    """Chain all root compiler mutations between exact QASM boundaries."""
    from bqskit.ir.lang.qasm2 import OPENQASM2Language

    input_qasm_state = _qasm_exact_state(stage_event["input_qasm"])
    output_qasm_state = _qasm_exact_state(stage_event["output_qasm"])
    if _exact_state_sha256(input_qasm_state) != stage_event["input_qasm_sha256"]:
        raise AssertionError("BQSKit stage input QASM hash mismatch.")
    if _exact_state_sha256(output_qasm_state) != stage_event["output_qasm_sha256"]:
        raise AssertionError("BQSKit stage output QASM hash mismatch.")
    decoded_input = OPENQASM2Language().decode(stage_event["input_qasm"])
    decoded_output = OPENQASM2Language().decode(stage_event["output_qasm"])
    if _bqskit_exact_state(decoded_input) != stage_event["input"]:
        raise AssertionError("BQSKit stage input QASM does not match its IR state.")
    if _bqskit_exact_state(decoded_output) != stage_event["output"]:
        raise AssertionError("BQSKit stage output QASM does not match its IR state.")

    replay_kinds = {
        "bqskit_foreach",
        "bqskit_pam_routing",
        "bqskit_apply_placement",
    }
    internal_events = [
        event
        for event in all_events
        if event.get("kind") in replay_kinds
        and event.get("stage") == stage_event.get("stage")
        and stage_event["started_ns"] <= event.get("started_ns", 0)
        <= stage_event["finished_ns"]
        and not (
            event.get("kind") == "bqskit_foreach"
            and event.get("parent_event_id") is not None
        )
    ]
    internal_events.sort(key=lambda event: event["started_ns"])
    current_state = stage_event["input"]
    for event in internal_events:
        if event["input"] != current_state:
            raise AssertionError(
                f"BQSKit replay chain breaks before {event.get('event_id')}."
            )
        if event["kind"] == "bqskit_foreach":
            _, current_state = _verify_bqskit_foreach_replay(event, event_by_id)
        elif event["kind"] == "bqskit_pam_routing":
            _, current_state = _verify_bqskit_pam_replay(event, event_by_id)
        else:
            _, current_state = _verify_bqskit_placement_replay(event)
    if current_state != stage_event["output"]:
        raise AssertionError(
            "BQSKit internal replay did not reach the exact stage output."
        )
    return input_qasm_state, output_qasm_state, len(internal_events)


def _verify_full_replay_chain(audit, event_by_id):
    """Replay every whole-circuit transition from archived input to output."""
    run = audit.get("run", {})
    if "input_circuit" not in run or "output_circuit" not in run:
        raise AssertionError(
            "Audit is missing its archived input/output circuit states."
        )
    input_state = _qasm_exact_state(run["input_circuit"])
    output_state = _qasm_exact_state(run["output_circuit"])
    if _exact_state_sha256(input_state) != run["input_circuit_sha256"]:
        raise AssertionError("Archived input circuit hash mismatch.")
    if _exact_state_sha256(output_state) != run["output_circuit_sha256"]:
        raise AssertionError("Archived output circuit hash mismatch.")

    transition_kinds = {
        "squander_basis_conversion",
        "round",
        "bqskit_stage",
        "qiskit_sabre_routing",
        "squander_sabre_routing",
        "exact_osr_routing",
        "bqskit_to_squander",
    }
    transitions = [
        event
        for event in audit["events"]
        if event.get("kind") in transition_kinds
    ]
    transitions.sort(key=lambda event: event["started_ns"])
    current_state = input_state
    for transition in transitions:
        if transition["kind"] == "squander_basis_conversion":
            transition_input, transition_output = _verify_squander_basis_replay(
                transition
            )
        elif transition["kind"] == "round":
            transition_input, transition_output = _verify_native_round_replay(
                transition, event_by_id
            )
        elif transition["kind"] == "bqskit_stage":
            transition_input, transition_output, _ = _verify_bqskit_stage_replay(
                transition, audit["events"], event_by_id
            )
        elif transition["kind"] == "qiskit_sabre_routing":
            transition_input, transition_output = _verify_qiskit_sabre_replay(
                transition
            )
        elif transition["kind"] == "exact_osr_routing":
            transition_input, transition_output = (
                _verify_exact_osr_routing_replay(transition)
            )
        elif transition["kind"] == "squander_sabre_routing":
            transition_input, transition_output = (
                _verify_squander_sabre_replay(transition)
            )
        else:
            transition_input, transition_output = (
                _verify_bqskit_to_squander_replay(transition)
            )
        if transition_input != current_state:
            raise AssertionError(
                f"Whole-circuit replay chain breaks before "
                f"{transition.get('event_id')}."
            )
        current_state = transition_output
    if current_state != output_state:
        raise AssertionError(
            "Whole-circuit replay did not finish at the exact archived output "
            "gate and parameter stream."
        )
    return len(transitions), _exact_state_sha256(current_state)


def _bqskit_audit_representation(circuit):
    """Serialize a BQSKit block, falling back to its unitary when necessary."""
    from bqskit.ir.lang.qasm2 import OPENQASM2Language

    representation = {
        "gate_counts": dict(collections.Counter(op.gate.name for op in circuit)),
        "qubits": int(circuit.num_qudits),
    }
    try:
        representation.update(
            format="openqasm2", qasm=OPENQASM2Language().encode(circuit)
        )
    except Exception:
        representation.update(
            format="unitary", unitary=_json_complex_matrix(circuit.get_unitary())
        )
    return representation


def _bqskit_full_qasm(circuit):
    """Encode a fully unfolded BQSKit circuit as OpenQASM 2."""
    from bqskit.ir.lang.qasm2 import OPENQASM2Language

    circuit = circuit.copy()
    circuit.unfold_all()
    return OPENQASM2Language().encode(circuit)


def _append_bqskit_stage_event(
    stage,
    started_ns,
    input_qasm,
    input_circuit,
    output_qasm,
    output_circuit,
    initial_mapping,
    final_mapping,
):
    """Record a complete compiler-stage boundary around internal replay events."""
    input_state = _bqskit_exact_state(input_circuit)
    output_state = _bqskit_exact_state(output_circuit)
    input_qasm_state = _qasm_exact_state(input_qasm)
    output_qasm_state = _qasm_exact_state(output_qasm)
    _append_rewrite_audit_event(
        {
            "kind": "bqskit_stage",
            "component": "bqskit_stage_replay",
            "stage": stage,
            "started_ns": int(started_ns),
            "finished_ns": time.time_ns(),
            "input_qasm": input_qasm,
            "output_qasm": output_qasm,
            "input_qasm_sha256": _exact_state_sha256(input_qasm_state),
            "output_qasm_sha256": _exact_state_sha256(output_qasm_state),
            "input": input_state,
            "output": output_state,
            "input_sha256": _exact_state_sha256(input_state),
            "output_sha256": _exact_state_sha256(output_state),
            "initial_mapping": [int(q) for q in initial_mapping],
            "final_mapping": [int(q) for q in final_mapping],
        }
    )


def _append_bqskit_to_squander_event(
    stage, started_ns, input_qasm, output_circuit, output_parameters
):
    """Record the exact BQSKit-QASM to Squander IR conversion boundary."""
    output = _squander_audit_representation(
        output_circuit,
        output_parameters,
        range(output_circuit.get_Qbit_Num()),
    )
    input_state = _qasm_exact_state(input_qasm)
    output_state = _qasm_exact_state(output)
    _append_rewrite_audit_event(
        {
            "kind": "bqskit_to_squander",
            "component": "framework_conversion_replay",
            "stage": stage,
            "started_ns": int(started_ns),
            "input_qasm": input_qasm,
            "output": output,
            "input_sha256": _exact_state_sha256(input_state),
            "output_sha256": _exact_state_sha256(output_state),
        }
    )


def _append_qiskit_sabre_event(
    stage,
    started_ns,
    input_qasm,
    output_qasm,
    initial_mapping,
    final_mapping,
    topology,
):
    """Record a replayable placement-and-SWAP certificate for light SABRE."""
    input_state = _qasm_exact_state(input_qasm)
    output_state = _qasm_exact_state(output_qasm)
    actions, replayed_final_mapping = _qiskit_sabre_actions(
        input_state,
        output_state,
        initial_mapping,
        topology,
    )
    final_mapping = [int(q) for q in final_mapping]
    if replayed_final_mapping != final_mapping:
        raise AssertionError(
            "SABRE output actions do not reproduce its reported final mapping."
        )
    _append_rewrite_audit_event(
        {
            "kind": "qiskit_sabre_routing",
            "component": "qiskit_sabre_replay",
            "stage": stage,
            "started_ns": int(started_ns),
            "input_qasm": input_qasm,
            "output_qasm": output_qasm,
            "input_sha256": _exact_state_sha256(input_state),
            "output_sha256": _exact_state_sha256(output_state),
            "initial_mapping": [int(q) for q in initial_mapping],
            "final_mapping": final_mapping,
            "topology": [[int(u), int(v)] for u, v in topology],
            "actions": actions,
        }
    )


def _append_squander_sabre_event(
    stage,
    started_ns,
    input_circuit,
    input_parameters,
    output_circuit,
    output_parameters,
    initial_mapping,
    final_mapping,
    topology,
):
    """Record internal Squander SABRE as source gates and physical SWAPs."""
    input_representation = _squander_audit_representation(
        input_circuit, input_parameters, range(input_circuit.get_Qbit_Num())
    )
    output_representation = _squander_audit_representation(
        output_circuit, output_parameters, range(output_circuit.get_Qbit_Num())
    )
    input_state = _qasm_exact_state(input_representation)
    output_state = _qasm_exact_state(output_representation)
    actions, replayed_final_mapping = _qiskit_sabre_actions(
        input_state, output_state, initial_mapping, topology
    )
    final_mapping = [int(q) for q in final_mapping]
    if replayed_final_mapping != final_mapping:
        raise AssertionError("Internal SABRE final mapping does not replay.")
    _append_rewrite_audit_event(
        {
            "kind": "squander_sabre_routing",
            "component": "squander_sabre_replay",
            "stage": stage,
            "started_ns": int(started_ns),
            "input": input_representation,
            "output": output_representation,
            "input_sha256": _exact_state_sha256(input_state),
            "output_sha256": _exact_state_sha256(output_state),
            "initial_mapping": [int(q) for q in initial_mapping],
            "final_mapping": final_mapping,
            "topology": [[int(u), int(v)] for u, v in topology],
            "actions": actions,
        }
    )


def _verify_squander_sabre_replay(event):
    """Replay the internal SABRE gate schedule and SWAP sequence exactly."""
    input_state = _qasm_exact_state(event["input"])
    output_state = _qasm_exact_state(event["output"])
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("Internal SABRE input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("Internal SABRE output hash mismatch.")
    actions, final_mapping = _qiskit_sabre_actions(
        input_state,
        output_state,
        event["initial_mapping"],
        event["topology"],
    )
    if actions != event["actions"]:
        raise AssertionError("Internal SABRE actions are not bit-identical.")
    if final_mapping != [int(q) for q in event["final_mapping"]]:
        raise AssertionError("Internal SABRE final mapping mismatch.")
    return input_state, output_state


def _append_exact_osr_routing_event(
    stage,
    started_ns,
    input_circuit,
    input_parameters,
    output_circuit,
    output_parameters,
    exact_route,
    topology,
    tolerance,
):
    """Record a replayable exact-cover, mapping, and local-unitary certificate."""
    from squander.partitioning.routing import permuted_partition_target

    input_representation = _squander_audit_representation(
        input_circuit, input_parameters, range(input_circuit.get_Qbit_Num())
    )
    output_representation = _squander_audit_representation(
        output_circuit, output_parameters, range(output_circuit.get_Qbit_Num())
    )
    input_state = _qasm_exact_state(input_representation)
    selections = []
    transition_swaps = exact_route.solution.transition_swaps or tuple(
        () for _selection in exact_route.solution.selections
    )
    if len(transition_swaps) != len(exact_route.solution.selections):
        raise AssertionError("Exact routing SWAP certificate length mismatch.")
    for swaps_before, selection in zip(
        transition_swaps, exact_route.solution.selections
    ):
        alternative = selection.alternative
        payload = alternative.payload
        if payload.source_circuit is None or payload.source_parameters is None:
            raise AssertionError("Exact routing audit is missing its source block.")
        logical_qubits = tuple(map(int, alternative.logical_qubits))
        source = _squander_audit_representation(
            payload.source_circuit,
            payload.source_parameters,
            range(len(alternative.logical_qubits)),
        )
        synthesized = _squander_audit_representation(
            payload.circuit,
            payload.parameters,
            range(len(alternative.logical_qubits)),
        )
        embedding = [
            int(alternative.input_physical[local_logical])
            for local_logical in payload.input_assignment
        ]
        certificate_kind = getattr(payload, "certificate_kind", "unitary")
        source_gate_indices = tuple(
            exact_route.candidate_gate_orders[selection.partition]
        )
        remapped_source_state = _remap_exact_state(
            _qasm_exact_state(source), logical_qubits, input_state["qubits"]
        )

        def source_matches(indices):
            return remapped_source_state["operations"] == [
                input_state["operations"][index] for index in indices
            ]

        if not source_matches(source_gate_indices):
            # The whole-circuit structural fallback preserves serialization
            # order, while local synthesis candidates use their dependency-
            # valid extraction order. Resolve this distinction before writing
            # the archive; the verifier must never need to guess it later.
            source_order = tuple(
                sorted(exact_route.candidate_gate_sets[selection.partition])
            )
            if source_matches(source_order):
                source_gate_indices = source_order
            else:
                raise AssertionError(
                    "Exact routing source block cannot be associated with its "
                    "input gate indices."
                )
        archived_selection = {
                "partition": int(selection.partition),
                "source_gate_indices": list(source_gate_indices),
                "logical_qubits": list(logical_qubits),
                "input_physical": [int(q) for q in alternative.input_physical],
                "output_physical": [int(q) for q in alternative.output_physical],
                "input_assignment": [int(q) for q in payload.input_assignment],
                "output_assignment": [int(q) for q in payload.output_assignment],
                "embedding": embedding,
                "source": source,
                "synthesized": synthesized,
                "certificate_kind": certificate_kind,
                "swaps_before": [
                    [int(left), int(right)] for left, right in swaps_before
                ],
                "tolerance": float(tolerance),
            }
        if certificate_kind == "sabre":
            actions, replayed_final_mapping = _qiskit_sabre_actions(
                _qasm_exact_state(source),
                _qasm_exact_state(synthesized),
                alternative.input_physical,
                topology,
            )
            if replayed_final_mapping != list(alternative.output_physical):
                raise AssertionError(
                    "Exact-router SABRE fallback mapping does not replay."
                )
            archived_selection["actions"] = actions
        elif certificate_kind == "unitary":
            # Compute the archived metric from the archived representations.
            # The verifier reconstructs these same QASM streams; using live
            # objects can differ in the final float64 bits after QASM gate
            # normalization despite representing the same unitary.
            source_unitary = _audit_representation_unitary(source)
            target = permuted_partition_target(
                source_unitary,
                payload.input_assignment,
                payload.output_assignment,
            )
            synthesized_unitary = _audit_representation_unitary(synthesized)
            metrics = _unitary_audit_metrics(target, synthesized_unitary)
            archived_selection["metrics"] = {
                    **metrics,
                    **{
                        f"{name}_bits": _float64_bits(value)
                        for name, value in metrics.items()
                    },
            }
        else:
            raise AssertionError(
                f"Unknown exact-routing certificate {certificate_kind!r}."
            )
        selections.append(archived_selection)
    output_state = _qasm_exact_state(output_representation)
    _append_rewrite_audit_event(
        {
            "kind": "exact_osr_routing",
            "component": "exact_osr_routing_replay",
            "stage": stage,
            "started_ns": int(started_ns),
            "input": input_representation,
            "output": output_representation,
            "input_sha256": _exact_state_sha256(input_state),
            "output_sha256": _exact_state_sha256(output_state),
            "initial_mapping": list(exact_route.solution.initial_mapping),
            "final_mapping": list(exact_route.solution.final_mapping),
            "topology": [[int(u), int(v)] for u, v in topology],
            "cnot_count": int(exact_route.solution.cnot_count),
            "single_qubit_count": int(
                exact_route.solution.single_qubit_count
            ),
            "master_backend": str(exact_route.solution.master_backend),
            "transition_swap_count": int(
                sum(
                    len(swaps)
                    for swaps in (exact_route.solution.transition_swaps or ())
                )
            ),
            "explored_states": int(exact_route.solution.explored_states),
            "solver_nodes": exact_route.solution.solver_nodes,
            "solver_bound": exact_route.solution.solver_bound,
            "solver_gap": exact_route.solution.solver_gap,
            "solver_solutions": exact_route.solution.solver_solutions,
            "optimal": bool(exact_route.solution.optimal),
            "cnot_optimal": bool(
                exact_route.solution.optimal
                or exact_route.solution.cnot_optimal
            ),
            "timed_out": bool(exact_route.timed_out),
            "selections": selections,
        }
    )


def _verify_exact_osr_routing_replay(event):
    """Replay an exact OSR route from source indices and local certificates."""
    from squander.partitioning.routing import permuted_partition_target

    input_state = _qasm_exact_state(event["input"])
    output_state = _qasm_exact_state(event["output"])
    if _exact_state_sha256(input_state) != event["input_sha256"]:
        raise AssertionError("Exact OSR routing input hash mismatch.")
    if _exact_state_sha256(output_state) != event["output_sha256"]:
        raise AssertionError("Exact OSR routing output hash mismatch.")

    predecessors, _ = _circuit_dependencies(input_state)
    consumed = set()
    expected_operations = []
    mapping = [int(q) for q in event["initial_mapping"]]
    topology = {frozenset((int(u), int(v))) for u, v in event["topology"]}
    for selection in event["selections"]:
        for edge in selection.get("swaps_before", []):
            left, right = map(int, edge)
            if frozenset((left, right)) not in topology:
                raise AssertionError("Exact OSR transition SWAP violates topology.")
            left_logical = mapping.index(left)
            right_logical = mapping.index(right)
            mapping[left_logical], mapping[right_logical] = (
                mapping[right_logical],
                mapping[left_logical],
            )
            # SWAP is symmetric, and QASM serialization canonicalizes its
            # operands even when the router records the traversed edge in the
            # opposite orientation. Match the serialized gate stream while
            # retaining the archived orientation for mapping replay above.
            expected_operations.append(
                {
                    "name": "swap",
                    "qubits": sorted((left, right)),
                    "params": [],
                }
            )
        source_indices = [int(index) for index in selection["source_gate_indices"]]
        if any(index in consumed for index in source_indices):
            raise AssertionError("Exact OSR selections overlap.")
        source_set = set(source_indices)
        for index in source_indices:
            if not predecessors[index] - source_set <= consumed:
                raise AssertionError("Exact OSR selected a dependency-blocked part.")

        logical_qubits = [int(q) for q in selection["logical_qubits"]]
        source_state = _remap_exact_state(
            _qasm_exact_state(selection["source"]),
            logical_qubits,
            input_state["qubits"],
        )
        if source_state["operations"] != [
            input_state["operations"][index] for index in source_indices
        ]:
            raise AssertionError("Exact OSR source block does not match its gates.")

        input_physical = [int(q) for q in selection["input_physical"]]
        output_physical = [int(q) for q in selection["output_physical"]]
        if any(mapping[q] != p for q, p in zip(logical_qubits, input_physical)):
            raise AssertionError("Exact OSR input mapping transition mismatch.")
        for logical, physical in zip(logical_qubits, output_physical):
            mapping[logical] = physical

        certificate_kind = selection.get("certificate_kind", "unitary")
        if certificate_kind == "sabre":
            actions, replayed_final_mapping = _qiskit_sabre_actions(
                _qasm_exact_state(selection["source"]),
                _qasm_exact_state(selection["synthesized"]),
                input_physical,
                event["topology"],
            )
            if actions != selection["actions"]:
                raise AssertionError(
                    "Exact-router SABRE actions are not bit-identical."
                )
            if replayed_final_mapping != output_physical:
                raise AssertionError(
                    "Exact-router SABRE final mapping does not replay."
                )
        elif certificate_kind == "unitary":
            source_unitary = _audit_representation_unitary(selection["source"])
            synthesized_unitary = _audit_representation_unitary(
                selection["synthesized"]
            )
            target = permuted_partition_target(
                source_unitary,
                selection["input_assignment"],
                selection["output_assignment"],
            )
            metrics = _unitary_audit_metrics(target, synthesized_unitary)
            if metrics["process_infidelity"] > float(selection["tolerance"]):
                raise AssertionError("Exact OSR selected an inaccurate synthesis.")
            for name, value in metrics.items():
                if _float64_bits(value) != selection["metrics"][f"{name}_bits"]:
                    raise AssertionError("Exact OSR metric is not bit-identical.")
        else:
            raise AssertionError(
                f"Unknown exact-routing certificate {certificate_kind!r}."
            )

        synthesized_state = _remap_exact_state(
            _qasm_exact_state(selection["synthesized"]),
            selection["embedding"],
            input_state["qubits"],
        )
        for operation in synthesized_state["operations"]:
            if len(operation["qubits"]) > 1:
                used = operation["qubits"]
                if len(used) == 2 and frozenset(used) not in topology:
                    raise AssertionError("Exact OSR output violates topology.")
        expected_operations.extend(synthesized_state["operations"])
        consumed.update(source_indices)

    if consumed != set(range(len(input_state["operations"]))):
        raise AssertionError("Exact OSR selections are not an exact gate cover.")
    if mapping != [int(q) for q in event["final_mapping"]]:
        raise AssertionError("Exact OSR final mapping mismatch.")
    expected_state = {
        "qubits": input_state["qubits"],
        "operations": expected_operations,
    }
    if expected_state != output_state:
        raise AssertionError("Exact OSR replay did not reproduce exact output.")
    return input_state, output_state


def _bqskit_exact_state(circuit):
    """Return BQSKit's exact flattened gate/parameter stream."""
    circuit = circuit.copy()
    circuit.unfold_all()
    return {
        "qubits": int(circuit.num_qudits),
        "operations": [
            {
                "name": op.gate.name,
                "qubits": [int(qubit) for qubit in op.location],
                "params": [
                    [_float64_bits(complex(parameter).real), _float64_bits(complex(parameter).imag)]
                    for parameter in op.params
                ],
            }
            for op in circuit
        ],
    }


def _bqskit_representation_exact_state(representation):
    """Decode an audit block and return its exact BQSKit operation stream."""
    return _bqskit_exact_state(
        _bqskit_circuit_from_representation(representation)
    )


def _bqskit_circuit_from_representation(representation):
    """Reconstruct a BQSKit circuit from a replay representation."""
    from bqskit import Circuit as BQSKitCircuit
    from bqskit.ir.gates import ConstantUnitaryGate
    from bqskit.ir.lang.qasm2 import OPENQASM2Language

    if representation["format"] == "openqasm2":
        circuit = OPENQASM2Language().decode(representation["qasm"])
    elif representation["format"] == "unitary":
        unitary = _complex_matrix_from_json(representation["unitary"])
        circuit = BQSKitCircuit(int(representation["qubits"]))
        circuit.append_gate(
            ConstantUnitaryGate(unitary), range(circuit.num_qudits)
        )
    else:
        raise AssertionError(
            f"Unsupported BQSKit audit representation {representation['format']}."
        )
    return circuit


def _audit_representation_unitary(representation):
    """Reconstruct a block unitary from an audit representation."""
    if representation["format"] == "unitary":
        return _complex_matrix_from_json(representation["unitary"])
    if representation["format"] == "squander_openqasm2":
        from qiskit import QuantumCircuit
        from squander import Qiskit_IO

        qiskit_circuit = QuantumCircuit.from_qasm_str(representation["qasm"])
        circuit, parameters = Qiskit_IO.convert_Qiskit_to_Squander(qiskit_circuit)
        return np.asarray(
            circuit.get_Matrix(np.asarray(parameters, dtype=np.float64)),
            dtype=np.complex128,
        )
    if representation["format"] != "openqasm2":
        raise AssertionError(f"Unknown audit representation: {representation['format']}")
    from bqskit.ir.lang.qasm2 import OPENQASM2Language
    return np.asarray(
        OPENQASM2Language().decode(representation["qasm"]).get_unitary(),
        dtype=np.complex128,
    )


def _append_rewrite_audit_event(event):
    """Append one process-safe JSONL audit event when auditing is enabled."""
    path = os.environ.get(_REWRITE_AUDIT_PATH_ENV)
    if not path:
        return None
    import fcntl

    event = dict(event)
    event.setdefault("event_id", uuid.uuid4().hex)
    event.setdefault("timestamp_ns", time.time_ns())
    event.setdefault("pid", os.getpid())
    encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
    with open(path, "a", encoding="utf-8") as audit_file:
        fcntl.flock(audit_file.fileno(), fcntl.LOCK_EX)
        audit_file.write(encoded + "\n")
        audit_file.flush()
        fcntl.flock(audit_file.fileno(), fcntl.LOCK_UN)
    return event["event_id"]


def _rewrite_audit_enabled():
    """Return whether the current process should emit rewrite events."""
    return bool(os.environ.get(_REWRITE_AUDIT_PATH_ENV))


def _make_rewrite_event(component, before, after, tolerance, **metadata):
    """Build a replayable event without slowing the optimization hot path."""
    event = {
        "kind": "rewrite",
        "component": component,
        "before": before,
        "after": after,
        "tolerance": float(tolerance),
    }
    event.update(metadata)
    return event


def load_rewrite_audit(path):
    """Load a compressed rewrite audit generated by the benchmark driver."""
    with gzip.open(path, "rt", encoding="utf-8") as audit_file:
        return json.load(audit_file)


def verify_rewrite_audit(audit_or_path, tolerance=None, expected_sha256=None):
    """Replay a complete optimization certificate and hard-fail any mismatch.

    The replay starts with the archived input gate/parameter stream, reruns
    basis and framework conversions, applies every selected native partition,
    replays BQSKit parent replacements, PAM dependencies/permutations/swaps and
    placements, and requires the exact archived output stream and file digest.
    Every local unitary metric is independently recomputed in complex128 and
    compared by its IEEE-754 binary64 bits with the stored value.
    """
    audit_path = None
    if isinstance(audit_or_path, (str, os.PathLike)):
        audit_path = os.fspath(audit_or_path)
        if expected_sha256 is not None:
            with open(audit_or_path, "rb") as audit_file:
                actual_sha256 = hashlib.sha256(audit_file.read()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise AssertionError(
                    f"Rewrite audit digest mismatch: {actual_sha256} != "
                    f"{expected_sha256}."
                )
        audit = load_rewrite_audit(audit_or_path)
    else:
        if expected_sha256 is not None:
            raise ValueError("A SHA-256 digest can only verify an on-disk audit.")
        audit = audit_or_path
    if audit.get("schema_version") != REWRITE_AUDIT_SCHEMA_VERSION:
        raise AssertionError(
            f"Unsupported rewrite audit schema {audit.get('schema_version')}."
        )
    recorded_software = audit.get("run", {}).get("software")
    current_software = rewrite_audit_software_versions()
    if recorded_software != current_software:
        raise AssertionError(
            "Bit-exact replay requires the recorded software/platform identity; "
            f"recorded={recorded_software}, current={current_software}."
        )
    verified = 0
    accepted = 0
    tolerance_misses = 0
    worst = 0.0
    worst_accepted = 0.0
    accepted_spectral_sum = 0.0
    event_by_id = {
        event["event_id"]: event
        for event in audit.get("events", [])
        if event.get("event_id") is not None
    }
    for event in audit.get("events", []):
        if event.get("kind") != "rewrite":
            continue
        before = _audit_representation_unitary(event["before"])
        after = _audit_representation_unitary(event["after"])
        metrics = _unitary_audit_metrics(before, after)
        limit = float(event["tolerance"] if tolerance is None else tolerance)
        error = metrics["process_infidelity"]
        within_tolerance = error <= limit
        if not within_tolerance:
            tolerance_misses += 1
        if event.get("accepted", False):
            if not within_tolerance:
                raise AssertionError(
                    f"Accepted rewrite {event.get('event_id', '<unknown>')} in "
                    f"{event.get('component', '<unknown>')} has process "
                    f"infidelity {error:.3e}, exceeding {limit:.3e}."
                )
        stored = event.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            stored_value = stored.get(metric_name)
            stored_bits = stored.get(f"{metric_name}_bits")
            recomputed_bits = _float64_bits(metric_value)
            if (
                stored_value is None
                or _float64_bits(stored_value) != stored_bits
                or stored_bits != recomputed_bits
            ):
                raise AssertionError(
                    f"Stored {metric_name} is not bit-identical when recomputed "
                    f"for rewrite {event.get('event_id')}."
                )
        verified += 1
        worst = max(worst, error)
        if event.get("accepted", False):
            accepted += 1
            worst_accepted = max(worst_accepted, error)
            accepted_spectral_sum += metrics["phase_aligned_spectral"]
    native_rounds = 0
    bqskit_invocations = 0
    pam_routing_passes = 0
    placement_passes = 0
    qiskit_sabre_passes = 0
    exact_osr_routing_passes = 0
    squander_sabre_passes = 0
    for event in audit.get("events", []):
        if event.get("kind") == "round" and event.get("component") == "squander_wide_optimization":
            _verify_native_round_replay(event, event_by_id)
            native_rounds += 1
        elif event.get("kind") == "bqskit_foreach":
            _verify_bqskit_foreach_replay(event, event_by_id)
            bqskit_invocations += 1
        elif event.get("kind") == "bqskit_pam_routing":
            _verify_bqskit_pam_replay(event, event_by_id)
            pam_routing_passes += 1
        elif event.get("kind") == "bqskit_apply_placement":
            _verify_bqskit_placement_replay(event)
            placement_passes += 1
        elif event.get("kind") == "qiskit_sabre_routing":
            _verify_qiskit_sabre_replay(event)
            qiskit_sabre_passes += 1
        elif event.get("kind") == "exact_osr_routing":
            _verify_exact_osr_routing_replay(event)
            exact_osr_routing_passes += 1
        elif event.get("kind") == "squander_sabre_routing":
            _verify_squander_sabre_replay(event)
            squander_sabre_passes += 1
    bqskit_stages = 0
    chained_bqskit_events = 0
    for event in audit.get("events", []):
        if event.get("kind") == "bqskit_stage":
            _, _, chained = _verify_bqskit_stage_replay(
                event, audit["events"], event_by_id
            )
            bqskit_stages += 1
            chained_bqskit_events += chained
    replayed_transitions, final_circuit_sha256 = _verify_full_replay_chain(
        audit, event_by_id
    )
    if audit_path is not None and audit.get("run", {}).get("output_file"):
        archived_output_path = os.path.join(
            os.path.dirname(audit_path),
            os.path.basename(audit["run"]["output_file"]),
        )
        if os.path.exists(archived_output_path):
            with open(archived_output_path, "rb") as output_file:
                output_bytes = output_file.read()
            output_digest = hashlib.sha256(output_bytes).hexdigest()
            if output_digest != audit["run"].get("output_file_sha256"):
                raise AssertionError("Archived output QASM byte hash mismatch.")
            archived_state = _qasm_exact_state(output_bytes.decode("utf-8"))
            if archived_state != _qasm_exact_state(audit["run"]["output_circuit"]):
                raise AssertionError(
                    "Archived output QASM is not the exact replayed gate and "
                    "parameter stream."
                )
        input_relative_path = audit["run"].get("input_file")
        if input_relative_path:
            search_directory = os.path.abspath(os.path.dirname(audit_path))
            archived_input_path = None
            while True:
                candidate = os.path.join(search_directory, input_relative_path)
                if os.path.exists(candidate):
                    archived_input_path = candidate
                    break
                parent = os.path.dirname(search_directory)
                if parent == search_directory:
                    break
                search_directory = parent
            if archived_input_path is not None:
                with open(archived_input_path, "rb") as input_file:
                    input_bytes = input_file.read()
                input_digest = hashlib.sha256(input_bytes).hexdigest()
                if input_digest != audit["run"].get("input_file_sha256"):
                    raise AssertionError("Archived input QASM byte hash mismatch.")
                from squander import utils

                source_circuit, source_parameters, _ = (
                    utils.qasm_to_squander_circuit(archived_input_path)
                )
                converted_input = _squander_audit_representation(
                    source_circuit,
                    source_parameters,
                    range(source_circuit.get_Qbit_Num()),
                )
                if _qasm_exact_state(converted_input) != _qasm_exact_state(
                    audit["run"]["input_circuit"]
                ):
                    raise AssertionError(
                        "Reparsing the archived input QASM did not reproduce "
                        "the replay input gate and parameter stream."
                    )
    return {
        "verified_rewrites": verified,
        "accepted_rewrites": accepted,
        "worst_process_infidelity": worst,
        "worst_accepted_process_infidelity": worst_accepted,
        "candidate_tolerance_misses": tolerance_misses,
        "accepted_spectral_error_bound": accepted_spectral_sum,
        "replayed_native_rounds": native_rounds,
        "replayed_bqskit_invocations": bqskit_invocations,
        "replayed_pam_routing_passes": pam_routing_passes,
        "replayed_placement_passes": placement_passes,
        "replayed_qiskit_sabre_passes": qiskit_sabre_passes,
        "replayed_exact_osr_routing_passes": exact_osr_routing_passes,
        "replayed_squander_sabre_passes": squander_sabre_passes,
        "replayed_bqskit_stages": bqskit_stages,
        "chained_bqskit_events": chained_bqskit_events,
        "replayed_whole_circuit_transitions": replayed_transitions,
        "final_circuit_sha256": final_circuit_sha256,
    }


def save_rewrite_audit(jsonl_path, output_path, run_metadata=None):
    """Compact process-safe JSONL events into one atomic ``.json.gz`` ledger."""
    events = []
    jsonl_path = os.fspath(jsonl_path)
    output_path = os.fspath(output_path)
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as audit_file:
            for line_number, line in enumerate(audit_file, 1):
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise AssertionError(
                            f"Corrupt audit event at {jsonl_path}:{line_number}."
                        ) from error
    events.sort(
        key=lambda event: (
            event.get("timestamp_ns", 0),
            event.get("event_id", ""),
        )
    )
    for event in events:
        if event.get("kind") == "rewrite":
            event["metrics"] = _unitary_audit_metrics(
                _audit_representation_unitary(event["before"]),
                _audit_representation_unitary(event["after"]),
            )
            for metric_name, metric_value in tuple(event["metrics"].items()):
                event["metrics"][f"{metric_name}_bits"] = _float64_bits(
                    metric_value
                )
            event["within_tolerance"] = (
                event["metrics"]["process_infidelity"]
                <= float(event["tolerance"])
            )
        elif event.get("kind") == "bqskit_pam_routing":
            for block in event.get("input_blocks", []):
                if "original_unitary" not in block:
                    continue
                block["original_unitary_metrics"] = _unitary_audit_metrics(
                    _audit_representation_unitary(block["before"]),
                    _complex_matrix_from_json(block["original_unitary"]),
                )
                for metric_name, metric_value in tuple(
                    block["original_unitary_metrics"].items()
                ):
                    block["original_unitary_metrics"][
                        f"{metric_name}_bits"
                    ] = _float64_bits(metric_value)
    summary = {
        "events": len(events),
        "rewrites": sum(event.get("kind") == "rewrite" for event in events),
        "accepted_rewrites": sum(
            event.get("kind") == "rewrite" and event.get("accepted", False)
            for event in events
        ),
        "failed_synthesis_attempts": sum(
            event.get("kind") == "synthesis_attempt"
            and not event.get("accepted", False)
            for event in events
        ),
        "fallbacks": sum(bool(event.get("fallback_used", False)) for event in events),
        "candidate_tolerance_misses": sum(
            event.get("kind") == "rewrite"
            and not event.get("within_tolerance", True)
            and not event.get("accepted", False)
            for event in events
        ),
    }
    run = dict(run_metadata or {})
    run.setdefault("software", rewrite_audit_software_versions())
    audit = {
        "schema_version": REWRITE_AUDIT_SCHEMA_VERSION,
        "run": run,
        "summary": summary,
        "events": events,
    }
    temporary_path = output_path + ".tmp"
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as audit_file:
            json.dump(audit, audit_file, separators=(",", ":"), allow_nan=False)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    with open(output_path, "rb") as audit_file:
        digest = hashlib.sha256(audit_file.read()).hexdigest()
    return audit, digest


# ---------------------------------------------------------------------------
# Helper: insert a SWAP as 3 CNOTs so BQSKit's scoring function weights
# them honestly (3 two-qubit ops instead of 1).  SEQPAM then avoids
# unnecessary SWAP insertions.
# ---------------------------------------------------------------------------
def _add_swap_as_cnots(circuit, a, b):
    """Append CNOT(a,b); CNOT(b,a); CNOT(a,b) — equivalent to SWAP(a,b)."""
    from bqskit.ir.gates import CNOTGate
    circuit.append_gate(CNOTGate(), [a, b])
    circuit.append_gate(CNOTGate(), [b, a])
    circuit.append_gate(CNOTGate(), [a, b])


# ---------------------------------------------------------------------------
# Module-level EAPP monkey-patch for SWAP fallback.
# BQSKit's Compiler starts a runtime server via Popen([sys.executable, ...]),
# a fresh Python process.  Class-level monkey-patches applied in the parent
# are invisible there.  We use an environment variable that the Popen child
# inherits; when this module is imported inside a worker, the env var triggers
# the patch.
# ---------------------------------------------------------------------------
def _append_topology_safe(new_c, op, topo_edges, width):
    """Append *op* to *new_c*, using SWAP bridges for edges not in *topo_edges*.

    For gates with ≥3 qubits, decomposes via :func:`squander.utils.circuit_to_CNOT_basis`
    and recurses on each resulting gate.
    """

    loc = list(op.location)
    gate = op.gate
    params = list(op.params) if op.params else None

    if gate.num_qudits == 1:
        if params:
            new_c.append_gate(gate, loc, params)
        else:
            new_c.append_gate(gate, loc)
        return

    if gate.num_qudits == 2:
        u, v = loc[0], loc[1]
        if (u, v) in topo_edges:
            if params:
                new_c.append_gate(gate, [u, v], params)
            else:
                new_c.append_gate(gate, [u, v])
            return
        # Edge not in topology — find shortest SWAP path u↔v via BFS.
        adj = {i: set() for i in range(width)}
        for a, b in topo_edges:
            adj[a].add(b)
            adj[b].add(a)
        from collections import deque
        parent = {v: None}
        q = deque([v])
        while q:
            node = q.popleft()
            if node == u:
                break
            for nb in adj.get(node, set()):
                if nb not in parent:
                    parent[nb] = node
                    q.append(nb)
        if u not in parent:
            # Cannot bridge this edge on the given topology.
            raise ValueError(f"Cannot bridge ({u},{v}) on topology")
        # Reconstruct path v -> ... -> u, then SWAP v along the path until it
        # is adjacent to u, apply the gate, and unwind those same SWAPs.
        path = [u]
        node = u
        while parent[node] is not None:
            node = parent[node]
            path.append(node)
        path = list(reversed(path))
        swaps = list(zip(path[:-2], path[1:-1]))
        cur = v
        for a, b in swaps:
            _add_swap_as_cnots(new_c, a, b)
            cur = b
        if params:
            new_c.append_gate(gate, [u, cur], params)
        else:
            new_c.append_gate(gate, [u, cur])
        for a, b in reversed(swaps):
            _add_swap_as_cnots(new_c, a, b)
        return

    # gate.num_qudits >= 3: decompose to CNOT basis via Squander's utility
    from bqskit.ir.lang.qasm2 import OPENQASM2Language
    from qiskit import qasm2

    # 1) Build a minimal BQSKit circuit containing just this gate
    from bqskit import Circuit as _BQCircuit
    tmp_bq = _BQCircuit(width)
    if params:
        tmp_bq.append_gate(gate, loc, params)
    else:
        tmp_bq.append_gate(gate, loc)

    # 2) Encode to QASM, then decode via Squander
    qasm_str = OPENQASM2Language().encode(tmp_bq)
    from squander import Qiskit_IO as _QIO
    qiskit_tmp = qasm2.loads(qasm_str)
    sq_tmp, sq_params = _QIO.convert_Qiskit_to_Squander(qiskit_tmp)

    # 3) Decompose to CNOT basis
    from squander.utils import circuit_to_CNOT_basis
    sq_decomp, sq_decomp_params = circuit_to_CNOT_basis(sq_tmp, sq_params)

    # 4) Convert back to BQSKit and recurse on each gate
    qiskit_decomp = _QIO.get_Qiskit_Circuit(sq_decomp, sq_decomp_params)
    bq_decomp = OPENQASM2Language().decode(qasm2.dumps(qiskit_decomp))
    for bq_op in bq_decomp:
        _append_topology_safe(new_c, bq_op, topo_edges, width)


def _bqskit_location_respects_topology(location, topo_edges):
    """Return true if ``location`` can be hosted by ``topo_edges``."""
    loc = tuple(int(q) for q in location)
    if len(loc) <= 1:
        return True
    if len(loc) == 2:
        return (loc[0], loc[1]) in topo_edges or (loc[1], loc[0]) in topo_edges

    wanted = set(loc)
    seen = {loc[0]}
    stack = [loc[0]]
    adjacency = {q: set() for q in wanted}
    for u, v in topo_edges:
        if u in wanted and v in wanted:
            adjacency[u].add(v)
            adjacency[v].add(u)
    while stack:
        cur = stack.pop()
        for nxt in adjacency.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return wanted <= seen


def _assert_circuit_respects_topology(circuit, topo_edges):
    """Raise AssertionError if ``circuit`` violates ``topo_edges``.

    Topology violations indicate a critical logic bug — the circuit cannot
    physically execute on the target hardware.  Execution must stop
    immediately so the root cause can be investigated and fixed.
    """
    for op in circuit:
        if op.gate.num_qudits <= 1:
            continue
        if not _bqskit_location_respects_topology(op.location, topo_edges):
            raise AssertionError(
                f"BUG: circuit contains {op.gate.name} on {list(op.location)}, "
                f"outside topology {sorted(topo_edges)}."
            )


def _fallback_circuit_for_permutation(original_circuit, graph, pi, po):
    """Build a topology-valid fallback for ``Po.T @ U @ Pi``.

    ``original_circuit`` is the block circuit passed into BQSKit's
    EmbedAllPermutationsPass.  ``graph`` is the block-local coupling graph
    selected by EAPP for this synthesis attempt.
    """
    from bqskit import Circuit as _BQCircuit

    width = original_circuit.num_qudits
    if len(pi) != width or len(po) != width:
        raise _SquanderSynthesisFailed(
            f"Permutation width mismatch for fallback: {pi}, {po}, width={width}."
        )

    topo_edges = set()
    for u, v in graph:
        topo_edges.add((u, v))
        topo_edges.add((v, u))

    if tuple(pi) == tuple(po):
        # Equal boundary permutations are a change of wire labels, not a
        # physical permutation that must be implemented twice.  In
        # particular, reversing a two-qubit CNOT remains one CNOT; encoding
        # it as SWAP-CNOT-SWAP poisons routing costs by six CNOTs.
        relabeled = original_circuit.copy()
        relabeled.renumber_qudits(tuple(pi))
        try:
            _assert_circuit_respects_topology(relabeled, topo_edges)
        except AssertionError:
            pass
        else:
            return relabeled

    fallback = _BQCircuit(width, original_circuit.radixes)

    for a, b in _topo_perm_to_swaps(pi, topo_edges, width):
        if (a, b) not in topo_edges:
            raise _SquanderSynthesisFailed(
                f"Cannot realize input permutation {pi} on topology {sorted(topo_edges)}."
            )
        _add_swap_as_cnots(fallback, a, b)

    for op in original_circuit:
        _append_topology_safe(fallback, op, topo_edges, width)

    po_inv = tuple(po.index(k) for k in range(width))
    for a, b in _topo_perm_to_swaps(po_inv, topo_edges, width):
        if (a, b) not in topo_edges:
            raise _SquanderSynthesisFailed(
                f"Cannot realize output permutation {po} on topology {sorted(topo_edges)}."
            )
        _add_swap_as_cnots(fallback, a, b)

    _assert_circuit_respects_topology(fallback, topo_edges)
    return fallback


async def _squander_synthesize_or_fallback(
    inner_synthesis,
    target,
    target_data,
    fallback,
):
    """Run Squander synthesis, falling back only for explicit Squander misses."""
    used_fallback = False
    try:
        synthesized = await inner_synthesis.synthesize(target, target_data)
    except _SquanderSynthesisFailed:
        synthesized = fallback
        used_fallback = True
    if _rewrite_audit_enabled():
        before = {
            "format": "unitary",
            "unitary": _json_complex_matrix(np.asarray(target)),
            "qubits": int(target.num_qudits),
        }
        after = _bqskit_audit_representation(synthesized)
        config = getattr(inner_synthesis, "config", {})
        _append_rewrite_audit_event(
            _make_rewrite_event(
                "seqpam_permutation_synthesis",
                before,
                after,
                _synthesis_acceptance_tolerance(config),
                accepted=False,
                candidate=True,
                stage=config.get("_rewrite_audit_stage", "routing"),
                fallback_used=used_fallback,
                input_permutation=target_data.get("_squander_input_permutation"),
                output_permutation=target_data.get("_squander_output_permutation"),
                subtopology=target_data.get("_squander_subtopology"),
            )
        )
    return synthesized


def _patch_eapp_if_needed():
    """Monkey-patch EAPP.run to catch Squander OSR failures per permutation.

    IMPORTANT: This patch fully replaces ``EmbedAllPermutationsPass.run``.
    It was written against BQSKit's internal EAPP implementation as of
    the pip-installed version (see pyproject.toml / requirements for the
    exact version).  If BQSKit changes its EAPP internals (scoring function,
    subtopology selection, permutation handling, or pass data keys), this
    patch may silently diverge and should be re-audited against the new
    BQSKit source.
    """
    import os as _os
    if not _os.environ.get('_SQUANDER_EAPP_FALLBACK_PATCH'):
        return

    from bqskit.passes.mapping.embed import EmbedAllPermutationsPass as __EAPP
    if getattr(__EAPP.run, "_squander_fallback_patch", False):
        return

    async def __patched_eapp_run(self, circuit, data):
        import copy as _copy
        import itertools as _it
        import logging as _logging
        from bqskit.compiler.machine import MachineModel as _MachineModel
        from bqskit.passes.mapping.topology import SubtopologySelectionPass as _STSP
        from bqskit.qis.graph import CouplingGraph as _CouplingGraph
        from bqskit.qis.permutation import PermutationMatrix as _PermutationMatrix
        from bqskit.runtime import get_runtime as _get_runtime

        _logger = _logging.getLogger("bqskit.passes.mapping.embed")
        utry = data.target

        if not all(r == utry.radixes[0] for r in utry.radixes):
            raise NotImplementedError(
                'PermutationAwareSynthesisPass only supports unitaries '
                'with the same radix on all qudits currently.',
            )

        width = utry.num_qudits
        perms = list(_it.permutations(range(width)))
        no_perm = [tuple(range(width))]
        Pis = [
            _PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]
        Pos = [
            _PermutationMatrix.from_qudit_location(width, utry.radixes[0], p)
            for p in perms
        ]

        if self.input_perm and self.output_perm:
            permsbyperms = list(_it.product(perms, perms))
            targets = [Po.T @ utry @ Pi for Pi, Po in _it.product(Pis, Pos)]
        elif self.input_perm:
            permsbyperms = list(_it.product(perms, no_perm))
            targets = [utry @ Pi for Pi in Pis]
        elif self.output_perm:
            permsbyperms = list(_it.product(no_perm, perms))
            targets = [Po.T @ utry for Po in Pos]
        else:
            _logger.warning('No permutation is being used in PAS.')
            permsbyperms = list(_it.product(no_perm, no_perm))
            targets = [utry]

        if self.vary_topology and width != 1:
            if _STSP.key not in data:
                raise RuntimeError(
                    'Cannot find subtopologies, try running a'
                    ' SubtopologySelectionPass first.',
                )
            if width not in data[_STSP.key]:
                raise RuntimeError(
                    'Subtopology information for block size'
                    f' {width} is not available.',
                )
            graphs = data[_STSP.key][width]
        else:
            graphs = [_CouplingGraph.all_to_all(width)]

        datas = []
        for graph in graphs:
            model = _MachineModel(
                circuit.num_qudits, graph,
                data.gate_set, data.model.radixes,
            )
            target_data = _copy.deepcopy(data)
            target_data.model = model
            datas.append(target_data)

        extended_targets = []
        extended_datas = []
        extended_graphs = []
        extended_perms = []
        fallback_circuits = []
        for target_index, target in enumerate(targets):
            for graph_index, graph in enumerate(graphs):
                fallback = _fallback_circuit_for_permutation(
                    circuit,
                    graph,
                    permsbyperms[target_index][0],
                    permsbyperms[target_index][1],
                )
                # EAPP uses this fallback unless OSR finds a strictly cheaper
                # circuit. Bound the search to circuits that can win instead
                # of retaining the tree search's unrelated default depth 14.
                fallback_entanglers = sum(
                    1 for op in fallback if op.gate.num_qudits >= 2
                )
                target_data = _copy.deepcopy(datas[graph_index])
                target_data['_squander_tree_level_max'] = max(
                    0, fallback_entanglers - 1
                )
                target_data['_squander_input_permutation'] = list(
                    permsbyperms[target_index][0]
                )
                target_data['_squander_output_permutation'] = list(
                    permsbyperms[target_index][1]
                )
                target_data['_squander_subtopology'] = [
                    [int(u), int(v)] for u, v in graph
                ]
                extended_targets.append(target)
                extended_datas.append(target_data)
                extended_graphs.append(graph)
                extended_perms.append(permsbyperms[target_index])
                fallback_circuits.append(fallback)

        circuits = await _get_runtime().map(
            _squander_synthesize_or_fallback,
            [self.inner_synthesis] * len(extended_targets),
            extended_targets,
            extended_datas,
            fallback_circuits,
        )

        perm_data = {}
        all_perms = list(_it.permutations(range(width)))
        for i, synthesized in enumerate(circuits):
            graph = extended_graphs[i]
            perm = extended_perms[i]

            if graph not in perm_data:
                perm_data[graph] = {}

            if perm in perm_data[graph]:
                s1 = self.scoring_fn(perm_data[graph][perm])
                s2 = self.scoring_fn(synthesized)
                if s2 < s1:
                    perm_data[graph][perm] = synthesized
            else:
                perm_data[graph][perm] = synthesized

            for univ_perm in all_perms[1:]:
                renumber_c = synthesized.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_pi = tuple(univ_perm[j] for j in perm[0])
                new_pf = tuple(univ_perm[j] for j in perm[1])
                new_graph = renumber_c.coupling_graph
                if new_graph not in perm_data:
                    perm_data[new_graph] = {}

                new_perm = (new_pi, new_pf)
                if new_perm not in perm_data[new_graph]:
                    perm_data[new_graph][new_perm] = renumber_c
                else:
                    s1 = self.scoring_fn(perm_data[new_graph][new_perm])
                    s2 = self.scoring_fn(renumber_c)
                    if s2 < s1:
                        perm_data[new_graph][new_perm] = renumber_c

        if circuit.gate_set.issubset(data.model.gate_set):
            for univ_perm in _it.permutations(range(width)):
                uperm = (univ_perm, univ_perm)
                renumber_c = circuit.copy()
                renumber_c.renumber_qudits(univ_perm)
                new_graph = renumber_c.coupling_graph
                new_score = self.scoring_fn(renumber_c)
                for graph, graph_data in perm_data.items():
                    if all(e in graph for e in new_graph):
                        if uperm not in graph_data:
                            graph_data[uperm] = renumber_c
                        elif new_score < self.scoring_fn(graph_data[uperm]):
                            graph_data[uperm] = renumber_c

        data['permutation_data'] = perm_data

    __patched_eapp_run._squander_fallback_patch = True
    __EAPP.run = __patched_eapp_run


_patch_eapp_if_needed()


class SquanderPartitioner(_BQSKitBasePass):
    """BQSKit pass: replace the body with selected Squander partition blocks."""

    def __init__(self, max_partition_size):
        super().__init__()
        self.max_partition_size = max_partition_size

    async def run(self, circuit, data=None):
        from qiskit import qasm2, QuantumCircuit
        from squander import Qiskit_IO
        from bqskit import Circuit as BQSKitCircuit
        from bqskit.ir.lang.qasm2 import OPENQASM2Language

        try:
            circ_qiskit = QuantumCircuit.from_qasm_str(
                OPENQASM2Language().encode(circuit)
            )
        except Exception:
            # Circuit contains gates that can't be QASM-encoded (e.g.
            # ConstantUnitaryGate from a prior pass).  Keep as-is.
            return

        circ, orig_parameters = Qiskit_IO.convert_Qiskit_to_Squander(circ_qiskit)
        cfg = _SQUANDER_BQSKIT_SYNTHESIS_CONFIG
        if not cfg:
            import os as _os, json as _json
            serialized = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
            if serialized:
                cfg = _json.loads(serialized)
        partition_strategy = (cfg or {}).get(
            "routing_partition_strategy", "ilp"
        )
        partitioned_circuit, parameters, _ = PartitionCircuit(
            circ,
            orig_parameters,
            self.max_partition_size,
            strategy=partition_strategy,
        )
        partitioned_circuit_bqskit = BQSKitCircuit(circ.get_Qbit_Num())
        for subcircuit in partitioned_circuit.get_Gates():
            if not isinstance(subcircuit, Circuit):
                raise RuntimeError(
                    "Squander partitioning returned a non-block gate; "
                    "BQSKit SEQPAM requires partition blocks."
                )

            involved_qbits = sorted(subcircuit.get_Qbits())
            qbit_map = {qbit: idx for idx, qbit in enumerate(involved_qbits)}
            subcircuit_parameters = parameters[
                subcircuit.get_Parameter_Start_Index() :
                subcircuit.get_Parameter_Start_Index() + subcircuit.get_Parameter_Num()
            ]
            remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, len(involved_qbits))
            subcircuit_qiskit = Qiskit_IO.get_Qiskit_Circuit(
                remapped_subcircuit.get_Flat_Circuit(),
                np.asarray(subcircuit_parameters, dtype=np.float64),
            )
            subcircuit_bqskit = OPENQASM2Language().decode(qasm2.dumps(subcircuit_qiskit))
            partitioned_circuit_bqskit.append_circuit(
                subcircuit_bqskit,
                involved_qbits,
                True,
                True,
            )
        circuit.become(partitioned_circuit_bqskit, False)


def _coupling_graph_embeds_in_model(edges, width, model_edges, model_width):
    """Return whether ``edges`` has a subgraph embedding in the device model."""

    if width > model_width:
        return False

    pattern_adjacency = [set() for _ in range(width)]
    for u, v in edges:
        u, v = int(u), int(v)
        pattern_adjacency[u].add(v)
        pattern_adjacency[v].add(u)

    device_adjacency = [set() for _ in range(model_width)]
    for u, v in model_edges:
        u, v = int(u), int(v)
        if u != v:
            device_adjacency[u].add(v)
            device_adjacency[v].add(u)

    # Map high-degree pattern vertices first. Candidate locations are then the
    # intersection of already mapped neighbours, avoiding the O(N**width)
    # permutation scan that is prohibitive on Eagle-sized devices.
    order = sorted(
        range(width),
        key=lambda vertex: (-len(pattern_adjacency[vertex]), vertex),
    )
    placement = {}
    used = set()

    def search(index):
        if index == width:
            return True
        vertex = order[index]
        mapped_neighbours = [
            placement[neighbour]
            for neighbour in pattern_adjacency[vertex]
            if neighbour in placement
        ]
        if mapped_neighbours:
            candidates = set(device_adjacency[mapped_neighbours[0]])
            for neighbour in mapped_neighbours[1:]:
                candidates.intersection_update(device_adjacency[neighbour])
        else:
            candidates = range(model_width)

        for location in candidates:
            if location in used:
                continue
            if len(device_adjacency[location]) < len(pattern_adjacency[vertex]):
                continue
            placement[vertex] = location
            used.add(location)
            if search(index + 1):
                return True
            used.remove(location)
            del placement[vertex]
        return False

    return search(0)


class SquanderSubtopologySelectionPass(_BQSKitBasePass):
    """Select genuinely embeddable block topologies for SEQPAM.

    BQSKit's ``filter_compatible_subgraphs`` may return non-embeddable
    supergraphs. For example, it returns a triangle for a linear device.
    Preserve labeled copies because EAPP needs them to cover every combination
    of local topology and boundary permutation, but reject graphs that cannot
    actually occur on the target device.
    """

    def __init__(self, block_size):
        super().__init__()
        self.block_size = int(block_size)
        if self.block_size <= 1:
            raise ValueError("Expected block_size > 1.")

    async def run(self, circuit, data):
        from bqskit.passes.mapping.topology import (
            SubtopologySelectionPass,
            all_coupling_graphs_of_size,
        )
        from bqskit.qis.graph import CouplingGraph

        model = data.model
        model_graph = model.coupling_graph
        model_edges = tuple(model_graph)
        model_width = int(model.num_qudits)
        model_is_all_to_all = (
            len(model_edges) == model_width * (model_width - 1) // 2
        )
        topologies = {}
        for width in range(2, self.block_size + 1):
            if model_is_all_to_all:
                topologies[width] = [CouplingGraph.all_to_all(width)]
                continue

            representatives = {}
            for candidate in all_coupling_graphs_of_size(width):
                candidate_edges = tuple(candidate)
                normalized = tuple(
                    sorted(
                        tuple(sorted((int(u), int(v))))
                        for u, v in candidate_edges
                    )
                )
                if normalized in representatives:
                    continue
                if _coupling_graph_embeds_in_model(
                    normalized,
                    width,
                    model_edges,
                    model_width,
                ):
                    representatives[normalized] = CouplingGraph(
                        normalized, width
                    )
            topologies[width] = [
                representatives[key]
                for key in sorted(
                    representatives,
                    key=lambda item: (len(item), item),
                )
            ]
            if not topologies[width]:
                raise RuntimeError(
                    f"Device topology has no connected {width}-qubit subgraph."
                )

        data[SubtopologySelectionPass.key] = topologies


@dataclass(frozen=True)
class SquanderPartitionSynthesisResult:
    """Validated native result returned by the shared synthesis callback."""

    circuit: Circuit
    parameters: np.ndarray
    config: dict
    topology: Optional[tuple]


def synthesize_partition_with_squander(
    target_matrix,
    config,
    *,
    mini_topology=None,
    tree_level_max=None,
):
    """Run one topology-aware Squander partition synthesis consistently.

    This is the native callback shared by BQSKit's permutation-aware routing
    adapter and Squander's exact router.  It owns configuration resolution,
    the optional routing search bound, candidate selection, parameter dtype,
    and the acceptance check performed by ``DecomposePartition``.  Conversion
    to a foreign circuit representation deliberately remains in the adapter.
    """
    resolved_config = {
        **dict(config or {}),
        "topology": mini_topology,
    }
    resolved_config = qgd_Wide_Circuit_Optimization(
        resolved_config
    ).config
    if tree_level_max is not None:
        # Routing supplies the strict-improvement bound of one fewer CNOT
        # than its topology-valid fallback.  Do not let a generic synthesis
        # cap silently make that exact routing search incomplete.
        resolved_config["tree_level_max"] = int(tree_level_max)

    target_matrix = np.asarray(target_matrix, dtype=np.complex128)
    candidates = qgd_Wide_Circuit_Optimization.DecomposePartition(
        target_matrix,
        resolved_config,
        mini_topology=mini_topology,
    )
    if not candidates:
        return None

    optimized_circuit, optimized_parameters = (
        qgd_Wide_Circuit_Optimization.CompareAndPickCircuits(
            [candidate[0] for candidate in candidates],
            [candidate[1] for candidate in candidates],
        )
    )
    return SquanderPartitionSynthesisResult(
        circuit=optimized_circuit,
        parameters=np.asarray(optimized_parameters, dtype=np.float64),
        config=resolved_config,
        topology=(
            None
            if mini_topology is None
            else tuple((int(u), int(v)) for u, v in mini_topology)
        ),
    )


class SquanderSynthesisPass(_BQSKitSynthesisPass):
    """BQSKit synthesis pass: optimize partition blocks with Squander.

    Raises _SquanderSynthesisFailed when the configured Squander synthesis
    strategy cannot produce a valid circuit for the requested subtopology. The
    monkey-patched EmbedAllPermutationsPass catches this and installs a
    SWAP-correct original-block fallback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        cfg = _SQUANDER_BQSKIT_SYNTHESIS_CONFIG
        if not cfg:
            # Workers spawned via Popen inherit env vars but not Python
            # globals.  The main process serializes the config to
            # _SQUANDER_BQSKIT_CONFIG before spawning workers.
            import os as _os, json as _json
            _env = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
            if _env:
                cfg = _json.loads(_env)
        self.config = dict(cfg or {})

    @staticmethod
    def _data_topology(data, qbit_num):
        """Return block subtopology from *data*.

        BQSKit labels are reversed when circuits are converted through
        Squander/Qiskit, so the topology supplied to Squander is reversed too.
        """
        if data is None or getattr(data, "model", None) is None:
            return None

        edges = []
        for u, v in data.model.coupling_graph:
            if u == v:
                continue
            edges.append((qbit_num - 1 - int(u), qbit_num - 1 - int(v)))

        all_edges = {
            frozenset((i, j))
            for i in range(qbit_num)
            for j in range(i + 1, qbit_num)
        }
        edge_set = {frozenset(edge) for edge in edges}
        if edge_set == all_edges:
            return None
        return edges

    @staticmethod
    def _topology_edges_from_data(data):
        """Return directed topology edges from BQSKit pass data."""
        if data is None or getattr(data, "model", None) is None:
            return None
        topo_edges = set()
        for u, v in data.model.coupling_graph:
            topo_edges.add((int(u), int(v)))
            topo_edges.add((int(v), int(u)))
        return topo_edges

    async def synthesize(self, target, data=None):
        from qiskit import qasm2
        from squander import Qiskit_IO
        from bqskit.ir.lang.qasm2 import OPENQASM2Language
        from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

        target_matrix = np.asarray(target)
        qbit_num = target.num_qudits
        mini_topology = self._data_topology(data, qbit_num)

        routing_level_max = None
        if data is not None and '_squander_tree_level_max' in data:
            routing_level_max = int(data['_squander_tree_level_max'])

        result = synthesize_partition_with_squander(
            target_matrix,
            self.config,
            mini_topology=mini_topology,
            tree_level_max=routing_level_max,
        )
        if result is None:
            tolerance = self.config.get(
                "tolerance", _default_squander_tolerance(self.config)
            )
            raise _SquanderSynthesisFailed(
                f"Squander synthesis failed for {qbit_num}-qubit block "
                f"at tolerance {tolerance}."
            )

        optimized_qiskit = Qiskit_IO.get_Qiskit_Circuit(
            result.circuit.get_Flat_Circuit(),
            result.parameters,
        )
        synthesized = OPENQASM2Language().decode(qasm2.dumps(optimized_qiskit))

        # The QASM round-trip preserves qubit labels but changes the physical
        # interpretation (Squander MSB=0 → BQSKit LSB=0).  Renumber qudits to
        # compensate: Squander qubit k (MSB=0) → BQSKit qubit (qbit_num-1-k).
        if qbit_num > 1:
            synthesized.renumber_qudits(
                [qbit_num - 1 - i for i in range(qbit_num)]
            )

        topo_edges = self._topology_edges_from_data(data)
        if topo_edges is not None:
            _assert_circuit_respects_topology(synthesized, topo_edges)

        if self.config.get("bqskit_distance_test", False):
            target_unitary = UnitaryMatrix(target)
            distance = target_unitary.get_distance_from(synthesized.get_unitary())
            process_tolerance = _synthesis_acceptance_tolerance(
                self.config
            )
            distance_tolerance = np.sqrt(process_tolerance)
            if distance > distance_tolerance:
                raise _SquanderSynthesisFailed(
                    "BQSKit synthesis validation failed: "
                    f"{distance:.2e} > {distance_tolerance:.2e}"
                )

        return synthesized


class _SquanderSynthesisFailed(Exception):
    """Raised when Squander cannot synthesize a partition block."""


def _topo_perm_to_swaps(pi, topo_edges, width):
    """Decompose permutation *pi* into SWAPs using only edges in *topo_edges*.

    Uses BFS on the topology graph to find a SWAP sequence that implements
    the permutation.  Returns a list of (u, v) pairs valid in *topo_edges*.
    """
    # Build adjacency list from topo_edges (undirected)
    adj = {i: set() for i in range(width)}
    for u, v in topo_edges:
        adj[u].add(v)
        adj[v].add(u)

    # Greedy: for each position i, bring the target qubit pi[i] to position i
    # by routing through the topology graph.
    current = list(range(width))  # current[pos] = which qubit is at pos
    swaps = []
    for i in range(width):
        target = pi[i]
        if current[i] == target:
            continue
        # Find where target currently is
        target_pos = current.index(target)
        # BFS from target_pos to i, finding shortest path of SWAPs
        from collections import deque
        parent = {target_pos: None}
        q = deque([target_pos])
        while q:
            u = q.popleft()
            if u == i:
                break
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        # Reconstruct path and apply SWAPs
        if i not in parent:
            raise _SquanderSynthesisFailed(
                f"Cannot realize permutation {pi} on disconnected topology "
                f"{sorted(topo_edges)}."
            )
        path = []
        v = i
        while parent[v] is not None:
            path.append(v)
            v = parent[v]
        path.append(target_pos)
        # Apply SWAPs along the path (reverse order to bring target to i)
        for k in range(len(path) - 1, 0, -1):
            a, b = path[k], path[k - 1]
            swaps.append((a, b))
            # Update current positions
            current[a], current[b] = current[b], current[a]
    return swaps


def _audited_foreach_class(base_class, config):
    """Wrap BQSKit's block control pass without copying its implementation."""
    from bqskit.passes.control.foreach import gen_replace_filter
    from bqskit.ir.gates import CircuitGate
    from bqskit import Circuit as BQSKitCircuit
    configured_audit = _copy_bqskit_synthesis_config(config)

    class AuditedForEachBlockPass(base_class):
        async def run(self, circuit, data):
            started_ns = time.time_ns()
            configured_filter = self.replace_filter
            replace_filter = (
                gen_replace_filter(configured_filter, data.model)
                if isinstance(configured_filter, str)
                else configured_filter
            )

            invocation = getattr(self, "_squander_audit_invocation", 0)
            self._squander_audit_invocation = invocation + 1
            block_index = [0]
            invocation_id = uuid.uuid4().hex
            parent_key = self.pass_down_key_prefix + "squander_audit_parent"
            parent_id = data.get(parent_key)
            previous_parent = data.get(parent_key)
            data[parent_key] = invocation_id
            input_state = _bqskit_exact_state(circuit)
            parent_operations = []
            collected_operations = []
            for cycle, operation in circuit.operations_with_cycles():
                if isinstance(operation.gate, CircuitGate):
                    local_circuit = operation.gate._circuit.copy()
                    local_circuit.set_params(operation.params)
                else:
                    local_circuit = BQSKitCircuit.from_operation(operation)
                record = {
                    "cycle": int(cycle),
                    "location": [int(q) for q in operation.location],
                    "before": _bqskit_audit_representation(local_circuit),
                    "collected": bool(self.collection_filter(operation)),
                    "rewrite_event_id": None,
                }
                parent_operations.append(record)
                if record["collected"]:
                    collected_operations.append(record)

            def audited_replace_filter(candidate, operation):
                current_block = block_index[0]
                block_index[0] += 1
                accepted = bool(replace_filter(candidate, operation))
                if isinstance(operation.gate, CircuitGate):
                    original = operation.gate._circuit.copy()
                    original.set_params(operation.params)
                else:
                    original = BQSKitCircuit.from_operation(operation)
                before = _bqskit_audit_representation(original)
                after = _bqskit_audit_representation(candidate)
                tolerance = _synthesis_acceptance_tolerance(
                    configured_audit
                )
                event_id = _append_rewrite_audit_event(
                    _make_rewrite_event(
                        "bqskit_partition_rewrite",
                        before,
                        after,
                        tolerance,
                        accepted=accepted,
                        stage=configured_audit.get("_rewrite_audit_stage"),
                        invocation=int(invocation),
                        block_index=int(current_block),
                        location=[int(q) for q in operation.location],
                        workflow=getattr(self.workflow, "name", type(self.workflow).__name__),
                    )
                )
                collected_operations[current_block]["rewrite_event_id"] = event_id
                return accepted

            self.replace_filter = audited_replace_filter
            try:
                await super().run(circuit, data)
            finally:
                self.replace_filter = configured_filter
                if previous_parent is None:
                    data.pop(parent_key, None)
                else:
                    data[parent_key] = previous_parent
            if block_index[0] != len(collected_operations):
                raise AssertionError(
                    "BQSKit audit did not observe every collected block."
                )
            output_state = _bqskit_exact_state(circuit)
            _append_rewrite_audit_event(
                {
                    "event_id": invocation_id,
                    "kind": "bqskit_foreach",
                    "component": "bqskit_partition_replay",
                    "stage": configured_audit.get("_rewrite_audit_stage"),
                    "parent_event_id": parent_id,
                    "started_ns": started_ns,
                    "invocation": int(invocation),
                    "workflow": getattr(
                        self.workflow, "name", type(self.workflow).__name__
                    ),
                    "input": input_state,
                    "input_sha256": _exact_state_sha256(input_state),
                    "output": output_state,
                    "output_sha256": _exact_state_sha256(output_state),
                    "parent_operations": parent_operations,
                }
            )

    AuditedForEachBlockPass.__name__ = "AuditedForEachBlockPass"
    return AuditedForEachBlockPass


def _circuit_point_json(point):
    """Serialize a BQSKit CircuitPoint as two integers."""
    return [int(point[0]), int(point[1])]


def _pam_exact_target(original_unitary, pre_perm, post_perm):
    """Build BQSKit's exact local PAM target for recorded permutations."""
    from bqskit import Circuit as BQSKitCircuit
    from bqskit.ir.gates import ConstantUnitaryGate, PermutationGate

    width = int(original_unitary.num_qudits)
    opposite_pre = tuple(pre_perm.index(i) for i in range(width))
    opposite_post = tuple(post_perm.index(i) for i in range(width))
    exact = BQSKitCircuit(width)
    exact.append_gate(PermutationGate(width, opposite_pre), range(width))
    exact.append_gate(ConstantUnitaryGate(original_unitary), range(width))
    exact.append_gate(PermutationGate(width, opposite_post), range(width))
    return np.asarray(exact.get_unitary(), dtype=np.complex128)


def _cnot_aware_pam_routing_class(base_class, config):
    """Scale PAM's native entangler-aware mapping heuristic if requested.

    BQSKit's PAM objective is::

        mapping_score + two_qubit_gates * gate_count_weight / len(front)

    BQSKit already counts only multi-qudit gates in the competing gate term;
    for Squander's U3+CNOT candidates this is exactly the local CNOT count.
    Its layout and routing passes intentionally use different gate weights
    (0.3 and 0.1), hence different look-ahead/local-cost tradeoffs.  Replacing
    both with one unit gate weight destroyed that calibration and produced
    severe routing blowups.  A cost of three is therefore the neutral/native
    setting; other values scale only the mapping term while preserving each
    pass's native entangler weight.
    """

    swap_cnot_cost = float(config.get("pam_swap_cnot_cost", 3.0))
    class CNOTAwarePAMPass(base_class):
        def _score_perm(self, circuit, F, pi, D, perm, E):
            mapping_score = super()._score_perm(circuit, F, pi, D, perm, E)
            if not F:
                return 0.0
            return (swap_cnot_cost / 3.0) * mapping_score

    CNOTAwarePAMPass.__name__ = f"CNOTAware{base_class.__name__}"
    return CNOTAwarePAMPass


def _audited_pam_class(base_class, config):
    """Wrap PAM routing with a complete, replayable action certificate."""
    configured_audit = _copy_bqskit_synthesis_config(config)

    class AuditedPAMRoutingPass(base_class):
        def _get_best_perm(self, circuit, perm_data, *args, **kwargs):
            mapping_before = list(args[2])
            result = super()._get_best_perm(circuit, perm_data, *args, **kwargs)
            input_point = self._squander_point_by_perm_data[id(perm_data)]
            operation = circuit[input_point]
            self._squander_choices.append(
                {
                    "input_point": _circuit_point_json(input_point),
                    "logical_location": [int(q) for q in operation.location],
                    "pre_perm_global": [int(q) for q in result[0]],
                    "post_perm_global": [int(q) for q in result[2]],
                    "mapping_before": [int(q) for q in mapping_before],
                    "selected": _bqskit_audit_representation(result[1]),
                }
            )
            return result

        def _apply_swap(self, swap, mapping, decay):
            before = list(mapping)
            result = super()._apply_swap(swap, mapping, decay)
            if hasattr(self, "_squander_mapping_trace"):
                self._squander_mapping_trace.append(
                    {
                        "kind": "swap",
                        "swap": [int(q) for q in swap],
                        "before": [int(q) for q in before],
                        "after": [int(q) for q in mapping],
                    }
                )
            return result

        def _apply_perm(self, permutation, mapping):
            before = list(mapping)
            result = super()._apply_perm(permutation, mapping)
            if hasattr(self, "_squander_mapping_trace"):
                self._squander_mapping_trace.append(
                    {
                        "kind": "permutation",
                        "permutation": [int(q) for q in permutation],
                        "before": [int(q) for q in before],
                        "after": [int(q) for q in mapping],
                    }
                )
            return result

        async def run(self, circuit, data):
            from bqskit.passes.control.foreach import ForEachBlockPass

            started_ns = time.time_ns()
            input_circuit = circuit.copy()
            input_state = _bqskit_exact_state(input_circuit)
            block_datas = data[ForEachBlockPass.key][-1]
            self._squander_point_by_perm_data = {
                id(block_data["permutation_data"]): block_data["point"]
                for block_data in block_datas
            }
            self._squander_choices = []
            self._squander_mapping_trace = []
            mapping_before_pass = [int(q) for q in data.final_mapping]
            input_blocks = []
            for point in sorted(self._squander_point_by_perm_data.values()):
                operation = input_circuit[point]
                predecessors = sorted(input_circuit.prev(point))
                if hasattr(operation.gate, "_circuit"):
                    source_block = operation.gate._circuit.copy()
                    source_block.set_params(operation.params)
                else:
                    source_block = type(input_circuit).from_operation(operation)
                input_blocks.append(
                    {
                        "point": _circuit_point_json(point),
                        "location": [int(q) for q in operation.location],
                        "predecessors": [
                            _circuit_point_json(predecessor)
                            for predecessor in predecessors
                        ],
                        "before": _bqskit_audit_representation(
                            source_block
                        ),
                    }
                )
            input_block_by_point = {
                tuple(block["point"]): block for block in input_blocks
            }

            await super().run(circuit, data)
            out_data = data[self.out_data_key]
            if len(self._squander_choices) != len(out_data):
                raise AssertionError("PAM audit choice/output count mismatch.")

            routed_blocks = {}
            for choice, (output_point, block_data) in zip(
                self._squander_choices, out_data.items()
            ):
                # PAM creates original_utry during its run and may evaluate it
                # through a numerically different path than source_block.
                # Preserve the exact target input and later certify its
                # fidelity to the independently replayable input gate stream.
                input_block_by_point[tuple(choice["input_point"])][
                    "original_unitary"
                ] = _json_complex_matrix(block_data["original_utry"])
                output_operation = circuit[output_point]
                selected = (
                    output_operation.gate._circuit.copy()
                    if hasattr(output_operation.gate, "_circuit")
                    else type(circuit).from_operation(output_operation)
                )
                selected.set_params(output_operation.params)
                selected_representation = _bqskit_audit_representation(selected)
                if (
                    _bqskit_representation_exact_state(choice["selected"])
                    != _bqskit_representation_exact_state(selected_representation)
                ):
                    raise AssertionError("PAM selected circuit changed before routing output.")
                target = {
                    "format": "unitary",
                    "unitary": _json_complex_matrix(
                        _pam_exact_target(
                            block_data["original_utry"],
                            block_data["pre_perm"],
                            block_data["post_perm"],
                        )
                    ),
                    "qubits": int(block_data["original_utry"].num_qudits),
                }
                rewrite_id = _append_rewrite_audit_event(
                    _make_rewrite_event(
                        "bqskit_pam_selected_block",
                        target,
                        selected_representation,
                        _synthesis_acceptance_tolerance(configured_audit),
                        accepted=True,
                        stage=configured_audit.get("_rewrite_audit_stage"),
                        input_point=choice["input_point"],
                        output_point=_circuit_point_json(output_point),
                        input_permutation=list(block_data["pre_perm"]),
                        output_permutation=list(block_data["post_perm"]),
                    )
                )
                choice.update(
                    {
                        "output_point": _circuit_point_json(output_point),
                        "physical_location": [
                            int(q) for q in output_operation.location
                        ],
                        "pre_perm_local": list(block_data["pre_perm"]),
                        "post_perm_local": list(block_data["post_perm"]),
                        "rewrite_event_id": rewrite_id,
                    }
                )
                routed_blocks[tuple(output_point)] = choice

            actions = []
            for cycle, operation in circuit.operations_with_cycles():
                point = (int(cycle), int(operation.location[0]))
                if point in routed_blocks:
                    actions.append({"kind": "block", **routed_blocks[point]})
                else:
                    local = type(circuit).from_operation(operation)
                    actions.append(
                        {
                            "kind": "gate",
                            "location": [int(q) for q in operation.location],
                            "circuit": _bqskit_audit_representation(local),
                        }
                    )

            output_state = _bqskit_exact_state(circuit)
            _append_rewrite_audit_event(
                {
                    "kind": "bqskit_pam_routing",
                    "component": "bqskit_pam_replay",
                    "stage": configured_audit.get("_rewrite_audit_stage"),
                    "started_ns": started_ns,
                    "input": input_state,
                    "input_sha256": _exact_state_sha256(input_state),
                    "output": output_state,
                    "output_sha256": _exact_state_sha256(output_state),
                    "input_blocks": input_blocks,
                    "actions": actions,
                    "choices": self._squander_choices,
                    "local_final_mapping": (
                        self._squander_mapping_trace[-1]["after"]
                        if self._squander_mapping_trace
                        else list(range(circuit.num_qudits))
                    ),
                    "mapping_before_pass": mapping_before_pass,
                    "mapping_after_pass": [int(q) for q in data.final_mapping],
                    "mapping_trace": self._squander_mapping_trace,
                }
            )

    AuditedPAMRoutingPass.__name__ = "AuditedPAMRoutingPass"
    return AuditedPAMRoutingPass


def _audited_apply_placement_class(base_class, config):
    """Wrap ApplyPlacement with an exact wire-renumbering certificate."""
    configured_audit = _copy_bqskit_synthesis_config(config)

    class AuditedApplyPlacement(base_class):
        async def run(self, circuit, data):
            started_ns = time.time_ns()
            input_state = _bqskit_exact_state(circuit)
            parent_operations = []
            for _, operation in circuit.operations_with_cycles():
                if hasattr(operation.gate, "_circuit"):
                    local_circuit = operation.gate._circuit.copy()
                    local_circuit.set_params(operation.params)
                else:
                    local_circuit = type(circuit).from_operation(operation)
                parent_operations.append(
                    {
                        "location": [int(q) for q in operation.location],
                        "circuit": _bqskit_audit_representation(local_circuit),
                    }
                )
            placement = [int(q) for q in data.placement]
            mapping_before = {
                "initial": [int(q) for q in data.initial_mapping],
                "final": [int(q) for q in data.final_mapping],
            }
            await super().run(circuit, data)
            output_state = _bqskit_exact_state(circuit)
            _append_rewrite_audit_event(
                {
                    "kind": "bqskit_apply_placement",
                    "component": "bqskit_placement_replay",
                    "stage": configured_audit.get("_rewrite_audit_stage"),
                    "started_ns": started_ns,
                    "placement": placement,
                    "parent_operations": parent_operations,
                    "mapping_before": mapping_before,
                    "mapping_after": {
                        "initial": [int(q) for q in data.initial_mapping],
                        "final": [int(q) for q in data.final_mapping],
                    },
                    "input": input_state,
                    "input_sha256": _exact_state_sha256(input_state),
                    "output": output_state,
                    "output_sha256": _exact_state_sha256(output_state),
                }
            )

    AuditedApplyPlacement.__name__ = "AuditedApplyPlacement"
    return AuditedApplyPlacement


@contextlib.contextmanager
def patched_seqpam_workflow_classes(bqskit_compile_module, use_squander_partitioner, config):
    """Patch BQSKit workflow factories to use Squander passes.

    Replaces QSearch/LEAP with ``SquanderSynthesisPass`` only when the selected
    decomposition strategy is Squander-native. External strategies such as
    ``bqskit`` and ``qiskit`` keep BQSKit's synthesis passes; otherwise they
    would be forwarded to Squander's ``DecomposePartition`` and fail as
    unsupported. Squander failures are caught by the EAPP patch and replaced
    with SWAP-correct fallbacks.
    """

    global _SQUANDER_BQSKIT_SYNTHESIS_CONFIG

    import os as _os, json as _json

    original_quick = bqskit_compile_module.QuickPartitioner
    original_qsearch = bqskit_compile_module.QSearchSynthesisPass
    original_leap = bqskit_compile_module.LEAPSynthesisPass
    original_foreach = bqskit_compile_module.ForEachBlockPass
    original_pam = bqskit_compile_module.PAMRoutingPass
    original_subtopology = bqskit_compile_module.SubtopologySelectionPass
    original_apply_placement = bqskit_compile_module.ApplyPlacement
    original_config = _SQUANDER_BQSKIT_SYNTHESIS_CONFIG
    original_config_env = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
    try:
        cfg = _copy_bqskit_synthesis_config(config)
        _SQUANDER_BQSKIT_SYNTHESIS_CONFIG = cfg
        # Also store in env var so worker processes (Popen) inherit it
        _os.environ['_SQUANDER_BQSKIT_CONFIG'] = _json.dumps(cfg)
        # BQSKit may report non-embeddable block graphs (for example, a
        # triangle on a linear device). Use the same exact topology filter for
        # both ILP and Quick partitioning so the comparison remains fair.
        bqskit_compile_module.SubtopologySelectionPass = (
            SquanderSubtopologySelectionPass
        )
        if use_squander_partitioner:
            bqskit_compile_module.QuickPartitioner = SquanderPartitioner
            # Placement does not emit gates and its bidirectional search needs
            # BQSKit's mapping-only score. Replace only the final routing
            # choice, where synthesized blocks and inserted SWAPs contribute
            # directly to the emitted CNOT count.
            bqskit_compile_module.PAMRoutingPass = _cnot_aware_pam_routing_class(
                original_pam, config
            )
        if config.get("strategy") in _SQUANDER_NATIVE_STRATEGIES:
            bqskit_compile_module.QSearchSynthesisPass = SquanderSynthesisPass
            bqskit_compile_module.LEAPSynthesisPass = SquanderSynthesisPass
        if _os.environ.get(_REWRITE_AUDIT_PATH_ENV):
            bqskit_compile_module.ForEachBlockPass = _audited_foreach_class(
                original_foreach, config
            )
            bqskit_compile_module.PAMRoutingPass = _audited_pam_class(
                bqskit_compile_module.PAMRoutingPass, config
            )
            bqskit_compile_module.ApplyPlacement = (
                _audited_apply_placement_class(original_apply_placement, config)
            )
        yield
    finally:
        bqskit_compile_module.QuickPartitioner = original_quick
        bqskit_compile_module.QSearchSynthesisPass = original_qsearch
        bqskit_compile_module.LEAPSynthesisPass = original_leap
        bqskit_compile_module.ForEachBlockPass = original_foreach
        bqskit_compile_module.PAMRoutingPass = original_pam
        bqskit_compile_module.SubtopologySelectionPass = original_subtopology
        bqskit_compile_module.ApplyPlacement = original_apply_placement
        _SQUANDER_BQSKIT_SYNTHESIS_CONFIG = original_config
        if original_config_env is None:
            _os.environ.pop('_SQUANDER_BQSKIT_CONFIG', None)
        else:
            _os.environ['_SQUANDER_BQSKIT_CONFIG'] = original_config_env


def _remove_seqpam_preoptimization(workflow):
    """Drop BQSKit's redundant A2A SeqPAM phase from a routing workflow.

    WCO has already optimized the circuit with all-to-all connectivity before
    invoking its router. BQSKit's stock workflow nevertheless partitions,
    permutation-synthesizes, and unfolds the circuit once on extracted A2A
    connectivity before repeating those operations for the actual topology.
    Keep the mapping half beginning at ``SubtopologySelectionPass``.
    """

    root_passes = getattr(workflow, "_passes", None)
    if not isinstance(root_passes, list) or len(root_passes) != 1:
        raise RuntimeError("Unexpected BQSKit SeqPAM workflow root structure.")
    conditional = root_passes[0]
    on_true = getattr(conditional, "on_true", None)
    passes = getattr(on_true, "_passes", None)
    if not isinstance(passes, list):
        raise RuntimeError("Unexpected BQSKit SeqPAM conditional structure.")

    mapping_start = next(
        (
            index
            for index, pass_object in enumerate(passes)
            if type(pass_object).__name__ == "SubtopologySelectionPass"
            or isinstance(pass_object, SquanderSubtopologySelectionPass)
        ),
        None,
    )
    if mapping_start is None:
        raise RuntimeError(
            "BQSKit SeqPAM workflow has no SubtopologySelectionPass."
        )
    on_true._passes = passes[mapping_start:]
    return workflow


def extract_subtopology(involved_qbits, qbit_map, config):
    """Return topology edges restricted to ``involved_qbits``, with indices remapped via ``qbit_map``.

    Args:
        involved_qbits: Qubit labels present in a partition.
        qbit_map: Maps original qubit index to local index (0..n-1).
        config: Configuration dict containing ``topology`` as a list of edges.

    Returns:
        List of ``(u, v)`` pairs in local indices, each edge fully inside the partition.
    """
    mini_topology = []
    for edge in config["topology"]:
        if edge[0] in involved_qbits and edge[1] in involved_qbits:
            mini_topology.append((qbit_map[edge[0]], qbit_map[edge[1]]))
    return mini_topology


# Universal gate decomposition dictionary.
# Each gate maps to its exact breakdown into {CNOT, H, RX, RY, RZ, ...} basis
# as defined by circuit_to_CNOT_basis in squander/utils.py.
# Native single-qubit gates and CNOT map to themselves with count 1.
_GATE_DECOMPOSITION = {
    # --- native gates (do not decompose) ---
    "CNOT":  {"CNOT": 1},
    "H":     {"H": 1},
    "X":     {"X": 1},
    "Y":     {"Y": 1},
    "Z":     {"Z": 1},
    "S":     {"S": 1},
    "Sdg":   {"Sdg": 1},
    "T":     {"T": 1},
    "Tdg":   {"Tdg": 1},
    "SX":    {"SX": 1},
    "SXdg":  {"SXdg": 1},
    "RX":    {"RX": 1},
    "RY":    {"RY": 1},
    "RZ":    {"RZ": 1},
    "R":     {"R": 1},
    "U1":    {"U1": 1},
    "U2":    {"U2": 1},
    "U3":    {"U3": 1},
    # --- decomposed gates (counts from circuit_to_CNOT_basis) ---
    "CH":    {"CNOT": 1, "RY": 2},                                    # RY + CNOT + RY
    "CZ":    {"CNOT": 1, "H": 2},                                      # H + CNOT + H
    "SYC":   {"CNOT": 3, "U1": 3},                                     # U1 + U1 + CNOT + U1 + CNOT + CNOT
    "CRY":   {"CNOT": 2, "RY": 2},                                     # CNOT + RY + CNOT + RY
    "CU":    {"CNOT": 2, "U1": 1, "RZ": 3, "RY": 2},                  # U1 + RZ + RY + CNOT + RY + RZ + CNOT + RZ
    "CR":    {"CNOT": 2, "RZ": 2, "RY": 2},                            # RZ + CNOT + RY + CNOT + RY + RZ
    "CROT":  {"CNOT": 2, "RZ": 3, "RY": 2},                            # RZ + RY + CNOT + RZ + CNOT + RY + RZ
    "CRX":   {"CNOT": 2, "H": 2, "RZ": 2},                             # H + CNOT + RZ + CNOT + RZ + H
    "CRZ":   {"CNOT": 2, "RZ": 2},                                     # CNOT + RZ + CNOT + RZ
    "CP":    {"CNOT": 2, "U1": 3},                                     # U1 + CNOT + U1 + CNOT + U1
    "CCX":   {"CNOT": 6, "H": 2, "T": 4, "Tdg": 3},                   # standard Toffoli: 7 CNOTs + 8 single-qubit
    "CSWAP": {"CNOT": 7, "H": 1, "T": 5, "Tdg": 2, "SX": 1, "Sdg": 1, "S": 1},  # Fredkin
    "SWAP":  {"CNOT": 3},                                              # CNOT + CNOT + CNOT
    "RXX":   {"CNOT": 2, "RX": 1},                                     # CNOT + RX + CNOT
    "RYY":   {"CNOT": 2, "RX": 4, "RZ": 1},                            # RX + RX + CNOT + RZ + CNOT + RX + RX
    "RZZ":   {"CNOT": 2, "RZ": 1},                                     # CNOT + RZ + CNOT
}

# Backward-compatible: CNOT-equivalent cost (number of CNOTs in decomposition).
CNOT_COUNT_DICT = {g: d.get("CNOT", 0) for g, d in _GATE_DECOMPOSITION.items()}


def CNOTGateCount(circ: Circuit, max_gates: int = 0) -> int:
    """Compute weighted two-qubit gate count for a circuit.

    The base count is the CNOT-equivalent cost derived from ``CNOT_COUNT_DICT``.
    When ``max_gates > 0``, the function returns a weighted scalar score:
    ``two_qubit_cost * max_gates + single_qubit_gate_count``.
    This is lexicographic only when ``max_gates`` is strictly greater than the
    largest possible difference in single-qubit counts.

    Args:
        circ: Squander circuit representation.
        max_gates: Weight multiplier for the two-qubit cost term.

    Returns:
        Integer gate-cost score used by optimization heuristics.
    """
    assert isinstance(circ, Circuit), \
        "The input parameters should be an instance of Squander Circuit"
    gate_counts = circ.get_Gate_Nums()
    num_cnots = sum(
        CNOT_COUNT_DICT.get(gate, 0) * count for gate, count in gate_counts.items()
    )
    if max_gates > 0:
        return num_cnots * max_gates + sum(
            y for x, y in gate_counts.items() if CNOT_COUNT_DICT.get(x, -1) <= 0
        )
    return num_cnots


def SingleQubitGateCount(circ: Circuit) -> int:
    """Count single-qubit gates in a circuit (U3, H, RX, RY, RZ, etc.).

    Uses _GATE_DECOMPOSITION to count non-CNOT gates in each gate's breakdown.

    Args:
        circ: Squander circuit representation.

    Returns:
        Total number of single-qubit gate operations when fully decomposed.
    """
    gate_counts = circ.get_Gate_Nums()
    total = 0
    for gate, count in gate_counts.items():
        decomp = _GATE_DECOMPOSITION.get(gate, {})
        total += count * sum(v for k, v in decomp.items() if k != "CNOT")
    return total


def TotalRawGateCount(circ: Circuit) -> int:
    """Total number of raw gate operations (single-qubit + multi-qubit).

    Args:
        circ: Squander circuit representation.

    Returns:
        Total gate operation count.
    """
    return sum(circ.get_Gate_Nums().values())


def CircuitGateStats(circ: Circuit) -> dict:
    """Return comprehensive gate statistics for a circuit.

    Uses _GATE_DECOMPOSITION to compute fully-decomposed gate counts.

    Returns dict with keys: cnot_equiv, single_qubit, total_raw, qubits,
    and gate_breakdown (per-gate-type raw counts).
    """
    gate_counts = circ.get_Gate_Nums()
    cnot_equiv = sum(
        CNOT_COUNT_DICT.get(g, 0) * c for g, c in gate_counts.items()
    )
    single = 0
    for g, c in gate_counts.items():
        decomp = _GATE_DECOMPOSITION.get(g, {})
        single += c * sum(v for k, v in decomp.items() if k != "CNOT")
    total = sum(gate_counts.values())
    return {
        "cnot_equiv": cnot_equiv,
        "single_qubit": single,
        "total_raw": total,
        "qubits": circ.get_Qbit_Num(),
        "gate_breakdown": dict(gate_counts),
    }


class qgd_Wide_Circuit_Optimization:
    """Optimize wide (many-qubit) circuits via partitioning and subcircuit decomposition.

    Supports multiple decomposition strategies, optional global recombination (ILP),
    and routing when the circuit does not match the target topology.
    """

    save_rewrite_audit = staticmethod(save_rewrite_audit)
    load_rewrite_audit = staticmethod(load_rewrite_audit)
    verify_rewrite_audit = staticmethod(verify_rewrite_audit)

    def __init__(self, config):
        """Validate and store wide-circuit optimization ``config`` (strategy, topology, partitioning, tolerances)."""

        config.setdefault("strategy", "TreeSearch")
        config.setdefault("parallel", 0)
        config.setdefault("verbosity", 0)
        config.setdefault("use_float", False)
        config.setdefault("tolerance", _default_squander_tolerance(config))
        config.setdefault(
            "osr_optimization_tolerance",
            OSR_OPTIMIZATION_TOLERANCE,
        )
        # Jointly optimize the complete exact-CNOT coverage profile. The
        # eight-hop BFGS2 ceiling is needed by hard seeds; easy fidelity solves
        # can terminate after their initial local minimization.
        config.setdefault("osr_profile_temperature", 0.1)
        config.setdefault("osr_cut_smoothmax_temperature", 0.1)
        config.setdefault("optimizer", "BFGS2")
        config.setdefault("use_basin_hopping", True)
        config.setdefault("use_differential_evolution", False)
        config.setdefault("use_dual_annealing", False)
        config.setdefault("max_iteration_loops", 8)
        config.setdefault("max_inner_iterations_bfgs2", 1000)
        config.setdefault(
            "circuit_validation_tolerance",
            _default_circuit_validation_tolerance(config),
        )
        config.setdefault(
            "synthesis_acceptance_tolerance",
            SYNTHESIS_ACCEPTANCE_TOLERANCE,
        )
        config["bqskit_synthesis_epsilon"] = _bqskit_synthesis_epsilon(config)
        config.setdefault("test_subcircuits", False)
        config.setdefault("test_final_circuit", True)
        config.setdefault("max_partition_size", 3)
        config.setdefault("topology", None)
        config.setdefault("partition_strategy", "ilp")
        # Keep the exact minimum-partition ILP as the routing default.  The
        # optional ``ilp-routing`` objective is experimental and can select
        # materially worse SEQPAM blocks despite preserving minimum cardinality.
        config.setdefault("routing_partition_strategy", "ilp")
        # The default router couples the all-partition exact-cover master to
        # permutation-aware OSR columns and exact SAT/Gurobi mapping flow. It
        # returns that routed synthesis directly: no foreign synthesis and no
        # post-routing cleanup are included in the reported route.
        config.setdefault("routing-strategy", "exact-osr")
        # Use the compact PuLP/Gurobi Benders master. It separates exact-cover
        # column choice from the complete mapping/SWAP trajectory, while the
        # monolithic ILP remains selectable as a cross-check.
        config.setdefault("exact_routing_master", "benders")
        # Build the complete symmetry-reduced topology-aware OSR catalog
        # before invoking PAM or the exact master. This makes routing quality
        # independent of optimistic column order and keeps the 20-minute
        # budget strictly a solver budget.
        config.setdefault("exact_routing_lazy_osr", False)
        # Routing prices each symmetry-distinct permutation exactly once.
        # Additional randomized retries multiply the dominant OSR cost and
        # make the exhaustive global router impractical on benchmark-scale
        # circuits. The topology-valid fallback already supplies a strict
        # per-target CNOT ceiling to the single attempt.
        config.setdefault("exact_routing_synthesis_restarts", 1)
        # Routing targets need the same deterministic basin schedule as normal
        # OSR synthesis. A two-start shortcut both missed easy 2-CNOT columns
        # and ran longer on multiply_n13 than the full eight-start schedule.
        config.setdefault(
            "routing_column_max_iteration_loops",
            int(config["max_iteration_loops"]),
        )
        config.setdefault("routing_column_synthesis_mode", "topology-osr")
        config.setdefault("exact_routing_catalog_progress", True)
        config.setdefault("exact_routing_catalog_progress_interval", 25)
        config.setdefault("exact_routing_light_sabre_seed_count", 32)
        config.setdefault("exact_routing_light_sabre_trials_per_seed", 1)
        config.setdefault("exact_routing_light_guided_cover_count", 8)
        # The former mapping-flow model is incomplete as a router because it
        # cannot insert arbitrary inter-block SWAPs, but any solution it does
        # find is a valid, compact MIP start for the complete staged model.
        config.setdefault("exact_routing_flow_seed", True)
        config.setdefault("exact_routing_flow_seed_timeout_seconds", 30.0)
        # PuLP model construction occurs before the solver time limit applies.
        # Bound the auxiliary flow seed by its dominant expression terms so a
        # large route cannot exhaust host memory while merely building a MIP
        # start. The complete Benders router and its LightSABRE incumbent are
        # unaffected when this optional seed is skipped.
        config.setdefault("exact_routing_flow_seed_max_terms", 500_000)
        # PuLP builds the staged path-order oracle in Python before Gurobi's
        # time limit starts. Its O(partitions * qubits^3) triangle system can
        # otherwise exhaust a large host. Above this bound Benders safely
        # returns the best replayable LightSABRE/PAM incumbent.
        config.setdefault(
            "exact_routing_fixed_cover_max_triangle_constraints", 500_000
        )
        config.setdefault("exact_routing_fixed_cover_backend", "sat")
        config.setdefault("exact_routing_sat_solver", "glucose42")
        config.setdefault(
            "exact_routing_sat_max_estimated_clauses", 5_000_000
        )
        # Native OSR calls run in isolated processes, but isolation alone does
        # not bound their aggregate memory. Keep both concurrency and each
        # worker's address space bounded, and stop well before host recovery
        # services such as SSH are threatened.
        # Match the normal WCO pool: None inherits partition_workers and then
        # falls back to every host CPU. An explicit integer remains available
        # for smaller machines.
        config.setdefault("routing_synthesis_workers", None)
        # The native three-qubit OSR backend reserves more virtual address
        # space than its resident set. A 12-GiB RLIMIT_AS silently rejected
        # valid columns; one 16-GiB worker remains safely bounded.
        config.setdefault("routing_synthesis_worker_memory_limit_gib", 16.0)
        config.setdefault("routing_minimum_available_memory_fraction", 0.25)
        # Bound cumulative routing-ILP time (compact master, flow certificates,
        # and fixed-cover oracles). Partition enumeration and OSR pricing must
        # finish so the master sees the complete column set; they are excluded.
        config.setdefault("exact_routing_timeout_seconds", 20 * 60)
        # A zero-interstage-SWAP check is a useful certificate, but some fixed
        # covers make that auxiliary ILP unexpectedly difficult.  Keep it a
        # short probe so the complete staged oracle always gets time to build
        # a synthesis-aware incumbent.
        config.setdefault(
            "exact_routing_benders_zero_swap_probe_seconds", 5.0
        )
        # Do not let the compact cover master monopolize the whole budget at
        # its root relaxation. Price its best incumbent between short slices
        # so routing quality improves even when global proof remains hard.
        config.setdefault("exact_routing_benders_master_slice_seconds", 30.0)
        config.setdefault(
            "exact_routing_benders_subproblem_slice_seconds", 60.0
        )
        config.setdefault("exact_routing_benders_stagnation_seconds", 120.0)
        config.setdefault("exact_routing_cover_pool_timeout_seconds", 10.0)
        config.setdefault("exact_routing_minimum_cover_seed_count", 8)
        config.setdefault("exact_routing_post_catalog_pam_seed_count", 12)
        config.setdefault("exact_routing_cover_seed_timeout_seconds", 10.0)
        # The exact token-order seed is useful diagnostically, but on the
        # benchmark outlier it did not improve a strong structural incumbent
        # in three minutes. Keep it opt-in so seed construction cannot consume
        # routing time without progress.
        config.setdefault("exact_routing_token_seed_timeout_seconds", 0.0)
        config.setdefault(
            "exact_routing_token_seed_cover_strategies",
            ("light-structural",),
        )
        config.setdefault("exact_routing_cover_seed_beam_width", 64)
        # Reuse the resolved all-partition/all-permutation OSR catalog in PAM
        # to obtain strong mapping incumbents. BQSKit contributes only its PAM
        # mapping heuristic; no BQSKit partitioner or synthesizer is involved.
        config.setdefault("exact_routing_precomputed_pam_seeds", True)
        config.setdefault("exact_routing_pam_layout_passes", 3)
        # Three is the neutral scale for PAM's native entangler-aware score.
        # Its averaged look-ahead distance is only a routing surrogate, so use
        # a small deterministic scale portfolio and retain the route with the
        # lowest actual CNOT count.  These passes reuse the completed OSR
        # catalog and perform no additional synthesis.
        config.setdefault(
            "exact_routing_pam_swap_cnot_costs",
            (1.5, 3.0, 6.0, 12.0),
        )
        config.setdefault(
            "exact_routing_pam_cover_strategies",
            ("kahn", "ilp", "ilp-routing"),
        )
        # A fixed-cover warm start needs only a few promising translations of
        # each relative OSR column. Exhaustively expanding every path interval
        # made this optional seed superlinear and unbounded on wide circuits.
        config.setdefault("exact_routing_cover_seed_translation_limit", 8)
        # CNOT count is the primary publication metric. Stop once its global
        # lower bound closes; proving the single-qubit tie is optional.
        config.setdefault("exact_routing_require_tiebreaker_proof", False)
        config.setdefault("seqpam_preoptimization", False)
        # PAM's mapping-distance score estimates future SWAP pressure.  Each
        # actual SWAP is emitted as three CNOTs, so compare it against local
        # synthesis using the same CNOT-equivalent unit.
        config.setdefault("pam_swap_cnot_cost", 3.0)
        config.setdefault("partition_workers", None)
        config.setdefault("auto_expand_partition_size", False)
        config.setdefault("force_small_circuit_validation", True)

        # testing the fields of config
        strategy = config["strategy"]
        allowed_startegies = [
            "TreeSearch",
            "TabuSearch",
            "Adaptive",
            "qiskit",
            "bqskit",
        ]
        if not strategy in allowed_startegies:
            raise Exception(
                f"The decomposition startegy should be either of {allowed_startegies}, got {strategy}."
            )

        parallel = config["parallel"]
        allowed_parallel = [0, 1, 2]
        if not parallel in allowed_parallel:
            raise Exception(
                f"The parallel configuration should be either of {allowed_parallel}, got {parallel}."
            )

        verbosity = config["verbosity"]
        if not isinstance(verbosity, int):
            raise Exception(f"The verbosity parameter should be an integer.")

        tolerance = config["tolerance"]
        if not isinstance(tolerance, float):
            raise Exception(f"The tolerance parameter should be a float.")
        if not 0.0 <= tolerance <= 1.0:
            raise Exception(
                "The tolerance parameter should be between zero and one."
            )

        osr_optimization_tolerance = config[
            "osr_optimization_tolerance"
        ]
        if not isinstance(osr_optimization_tolerance, float):
            raise Exception(
                "The osr_optimization_tolerance parameter should be a float."
            )
        if not 0.0 <= osr_optimization_tolerance <= 1.0:
            raise Exception(
                "The osr_optimization_tolerance parameter should be between "
                "zero and one."
            )

        use_float = config["use_float"]
        if not isinstance(use_float, bool):
            raise Exception(f"The use_float parameter should be a bool.")

        synthesis_acceptance_tolerance = config[
            "synthesis_acceptance_tolerance"
        ]
        if not isinstance(synthesis_acceptance_tolerance, float):
            raise Exception(
                "The synthesis_acceptance_tolerance parameter should be a float."
            )
        if not 0.0 <= synthesis_acceptance_tolerance <= 1.0:
            raise Exception(
                "The synthesis_acceptance_tolerance parameter should "
                "be between zero and one."
            )
        if synthesis_acceptance_tolerance < tolerance:
            raise Exception(
                "The synthesis_acceptance_tolerance parameter should not be "
                "tighter than the optimization tolerance."
            )

        circuit_validation_tolerance = config["circuit_validation_tolerance"]
        if not isinstance(circuit_validation_tolerance, float):
            raise Exception(
                "The circuit_validation_tolerance parameter should be a float."
            )
        if not 0.0 <= circuit_validation_tolerance <= 1.0:
            raise Exception(
                "The circuit_validation_tolerance parameter should be between "
                "zero and one."
            )

        test_subcircuits = config["test_subcircuits"]
        if not isinstance(test_subcircuits, bool):
            raise Exception(f"The test_subcircuits parameter should be a bool.")

        test_final_circuit = config["test_final_circuit"]
        if not isinstance(test_final_circuit, bool):
            raise Exception(f"The test_final_circuit parameter should be a bool.")

        max_partition_size = config["max_partition_size"]
        if not isinstance(max_partition_size, int):
            raise Exception(f"The max_partition_size parameter should be an integer.")

        partition_workers = config["partition_workers"]
        if partition_workers is not None and (
            not isinstance(partition_workers, int) or partition_workers <= 0
        ):
            raise Exception(
                "The partition_workers parameter should be a positive integer or None."
            )
        routing_synthesis_workers = config["routing_synthesis_workers"]
        if routing_synthesis_workers is not None and (
            not isinstance(routing_synthesis_workers, int)
            or isinstance(routing_synthesis_workers, bool)
            or routing_synthesis_workers <= 0
        ):
            raise ValueError(
                "The routing_synthesis_workers parameter should be a positive "
                "integer or None."
            )
        exact_routing_flow_seed_max_terms = config[
            "exact_routing_flow_seed_max_terms"
        ]
        if (
            not isinstance(exact_routing_flow_seed_max_terms, int)
            or isinstance(exact_routing_flow_seed_max_terms, bool)
            or exact_routing_flow_seed_max_terms <= 0
        ):
            raise ValueError(
                "The exact_routing_flow_seed_max_terms parameter should be a "
                "positive integer."
            )
        minimum_cover_seed_count = config[
            "exact_routing_minimum_cover_seed_count"
        ]
        if (
            not isinstance(minimum_cover_seed_count, int)
            or isinstance(minimum_cover_seed_count, bool)
            or minimum_cover_seed_count <= 0
        ):
            raise ValueError(
                "The exact_routing_minimum_cover_seed_count parameter should "
                "be a positive integer."
            )
        light_guided_cover_count = config[
            "exact_routing_light_guided_cover_count"
        ]
        if (
            not isinstance(light_guided_cover_count, int)
            or isinstance(light_guided_cover_count, bool)
            or light_guided_cover_count < 0
        ):
            raise ValueError(
                "The exact_routing_light_guided_cover_count parameter should "
                "be a nonnegative integer."
            )
        post_catalog_seed_count = config[
            "exact_routing_post_catalog_pam_seed_count"
        ]
        if (
            not isinstance(post_catalog_seed_count, int)
            or isinstance(post_catalog_seed_count, bool)
            or post_catalog_seed_count < 0
        ):
            raise ValueError(
                "The exact_routing_post_catalog_pam_seed_count parameter "
                "should be a nonnegative integer."
            )
        if not isinstance(config["exact_routing_precomputed_pam_seeds"], bool):
            raise ValueError(
                "The exact_routing_precomputed_pam_seeds parameter should be a bool."
            )
        pam_layout_passes = config["exact_routing_pam_layout_passes"]
        if (
            not isinstance(pam_layout_passes, int)
            or isinstance(pam_layout_passes, bool)
            or pam_layout_passes <= 0
        ):
            raise ValueError(
                "The exact_routing_pam_layout_passes parameter should be a "
                "positive integer."
            )
        pam_swap_cnot_costs = config["exact_routing_pam_swap_cnot_costs"]
        if (
            not isinstance(pam_swap_cnot_costs, (tuple, list))
            or not pam_swap_cnot_costs
            or any(
                not isinstance(cost, (int, float))
                or isinstance(cost, bool)
                or not np.isfinite(cost)
                or cost <= 0.0
                for cost in pam_swap_cnot_costs
            )
        ):
            raise ValueError(
                "The exact_routing_pam_swap_cnot_costs parameter should "
                "contain positive finite numbers."
            )
        pam_cover_strategies = config["exact_routing_pam_cover_strategies"]
        if (
            not isinstance(pam_cover_strategies, (tuple, list))
            or not pam_cover_strategies
            or any(
                strategy not in ("kahn", "ilp", "ilp-routing")
                for strategy in pam_cover_strategies
            )
        ):
            raise ValueError(
                "The exact_routing_pam_cover_strategies parameter should contain "
                "only 'kahn', 'ilp', and 'ilp-routing'."
            )

        if config["routing_partition_strategy"] not in (
            "kahn", "ilp", "ilp-routing"
        ):
            raise Exception(
                "The routing_partition_strategy parameter should be 'kahn', "
                "'ilp', or 'ilp-routing'."
            )

        if not isinstance(config["seqpam_preoptimization"], bool):
            raise Exception(
                "The seqpam_preoptimization parameter should be a bool."
            )

        pam_swap_cnot_cost = config["pam_swap_cnot_cost"]
        if not isinstance(pam_swap_cnot_cost, float):
            raise Exception("The pam_swap_cnot_cost parameter should be a float.")
        if pam_swap_cnot_cost <= 0.0:
            raise Exception("The pam_swap_cnot_cost parameter should be positive.")

        self.config = config

        self.max_partition_size = max_partition_size

    @staticmethod
    def partition_tree_level_max(config, subcircuit, reduction=1):
        """Return the tree-search depth used for partition-local rewrites."""

        target_depth = max(0, CNOTGateCount(subcircuit, 0) - reduction)
        configured_limit = config.get("partition_tree_level_max", None)
        if configured_limit is None:
            configured_limit = target_depth
        return min(target_depth, int(configured_limit))

    def ConstructCircuitFromPartitions(
        self, circs: List[Circuit], parameter_arrs: List[List[np.ndarray]]
    ) -> Tuple[Circuit, np.ndarray]:
        """Concatenate optimized partition circuits into a single wide circuit.

        Args:
            circs: Partition circuits in execution order.
            parameter_arrs: Parameter arrays corresponding to ``circs``.

        Returns:
            Tuple of ``(wide_circuit, wide_parameters)``.
        """

        if not isinstance(circs, list):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance(parameter_arrs, list):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs):
            raise Exception("The first two arguments should be of the same length")

        qbit_num = circs[0].get_Qbit_Num()

        wide_parameters = np.concatenate(parameter_arrs, axis=0)

        wide_circuit = Circuit(qbit_num)

        for circ in circs:
            wide_circuit.add_Circuit(circ)

        assert (
            wide_circuit.get_Parameter_Num() == wide_parameters.size
        ), f"Mismatch in the number of parameters: {wide_circuit.get_Parameter_Num()} vs {wide_parameters.size}"

        return wide_circuit, wide_parameters

    @staticmethod
    def DecomposePartition(
        Umtx: np.ndarray, config: dict, mini_topology=None, structure=None
    ) -> list[tuple[Circuit, np.ndarray]]:
        """Decompose a unitary ``Umtx`` (e.g. from a partition) using ``config['strategy']``.

        Args:
            Umtx: Complex unitary matrix.
            config: Must include ``strategy``, ``tolerance``, ``verbosity``, etc.
            mini_topology: Optional hardware couplers for topology-aware decomposers.
            structure: Required gate structure when ``strategy == "Custom"``.

        Returns:
            Normally ``[(circuit, parameters)]`` on success, or ``[]`` if the
            decomposition error exceeds ``tolerance``. If
            ``config.get('stop_first_solution')`` is false, returns
            ``cDecompose.all_solutions`` from the underlying decomposer instead of
            a single best pair.
        """
        strategy = config["strategy"]
        if strategy == "TreeSearch":
            cDecompose = N_Qubit_Decomposition_Tree_Search(
                Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology
            )
        elif strategy == "TabuSearch":
            cDecompose = N_Qubit_Decomposition_Tabu_Search(
                Umtx.conj().T, config=config, accelerator_num=0, topology=mini_topology
            )
        elif strategy == "Adaptive":
            cDecompose = N_Qubit_Decomposition_adaptive(
                Umtx.conj().T,
                level_limit_max=5,
                level_limit_min=1,
                topology=mini_topology,
            )
        elif strategy == "Custom":
            cDecompose = N_Qubit_Decomposition_custom(
                Umtx.conj().T, config=config, accelerator_num=0
            )
            assert (
                structure is not None
            ), "Custom decomposition strategy requires a gate structure to be provided."
            cDecompose.set_Gate_Structure(structure)
        else:
            raise Exception(f"Unsupported decomposition type: {strategy}")

        optimization_tolerance = config["tolerance"]
        acceptance_tolerance = _synthesis_acceptance_tolerance(config)
        cDecompose.set_Verbose(config["verbosity"])
        cDecompose.set_Cost_Function_Variant(3)
        cDecompose.set_Optimization_Tolerance(optimization_tolerance)

        cDecompose.set_Optimizer(config.get("optimizer", "BFGS2"))

        # starting the decomposition
        try:
            cDecompose.Start_Decomposition()
        except Exception as e:
            # print(e)
            raise e
            # return []
        if not config.get("stop_first_solution", True):
            return cDecompose.all_solutions

        squander_circuit = cDecompose.get_Circuit()
        if strategy == "TreeSearch" and config.get("use_float", False):
            parameters = cDecompose.get_Optimized_Parameters_Double()
        else:
            parameters = cDecompose.get_Optimized_Parameters()
        assert parameters is not None

        if strategy == "Custom":
            err = cDecompose.Optimization_Problem(parameters)
            it = 0
            while err > optimization_tolerance and it < 20:
                cDecompose.set_Optimized_Parameters(
                    np.random.rand(cDecompose.get_Parameter_Num()) * (2 * np.pi)
                )
                cDecompose.Start_Decomposition()
                parameters = cDecompose.get_Optimized_Parameters()
                err = cDecompose.Optimization_Problem(parameters)
                it += 1
            if err > optimization_tolerance or it != 0:
                print("Decomposition error: ", err, it)
        else:
            err = cDecompose.get_Decomposition_Error()

        # Accept the circuit that Python actually receives, not merely the
        # optimizer's internal error estimate. At tight tolerances, rebuilding
        # the returned gate stream can expose rounding differences that matter
        # to the independently replayed rewrite audit.
        returned_unitary = squander_circuit.get_Matrix(
            np.asarray(parameters, dtype=np.float64)
        )
        optimizer_reported_error = err
        err = _unitary_audit_metrics(Umtx, returned_unitary)[
            "process_infidelity"
        ]
        if acceptance_tolerance < err:
            if _rewrite_audit_enabled():
                _append_rewrite_audit_event(
                    {
                        "kind": "synthesis_attempt",
                        "component": "squander_partition_synthesis",
                        "accepted": False,
                        "target": {
                            "format": "unitary",
                            "unitary": _json_complex_matrix(Umtx),
                            "qubits": int(round(np.log2(Umtx.shape[0]))),
                        },
                        "reported_error": float(err),
                        "optimizer_reported_error": float(
                            optimizer_reported_error
                        ),
                        "tolerance": float(acceptance_tolerance),
                        "optimization_tolerance": float(
                            optimization_tolerance
                        ),
                        "strategy": strategy,
                        "stage": config.get("_rewrite_audit_stage"),
                    }
                )
            # raise Exception(f"Decomposition error {err} exceeds the tolerance {tolerance}.")
            return []

        if _rewrite_audit_enabled():
            target_representation = {
                "format": "unitary",
                "unitary": _json_complex_matrix(Umtx),
                "qubits": int(round(np.log2(Umtx.shape[0]))),
            }
            candidate_representation = _squander_audit_representation(
                squander_circuit, parameters
            )
            _append_rewrite_audit_event(
                _make_rewrite_event(
                    "squander_partition_synthesis",
                    target_representation,
                    candidate_representation,
                    acceptance_tolerance,
                    accepted=False,
                    candidate=True,
                    reported_error=float(err),
                    optimizer_reported_error=float(optimizer_reported_error),
                    optimization_tolerance=float(optimization_tolerance),
                    strategy=strategy,
                    stage=config.get("_rewrite_audit_stage"),
                    subtopology=(
                        [[int(u), int(v)] for u, v in mini_topology]
                        if mini_topology is not None
                        else None
                    ),
                )
            )

        return [(squander_circuit, parameters)]

    @staticmethod
    def CompareAndPickCircuits(
        circs: List[Circuit],
        parameter_arrs: List[np.ndarray],
        metric: Callable[[Circuit], Any] = CNOTGateCount,
    ) -> tuple[Circuit, np.ndarray]:
        """Select the circuit with the lowest ``metric`` value.

        Args:
            circs: Candidate Squander circuits (same length as ``parameter_arrs``).
            parameter_arrs: Parameter vectors aligned with ``circs``.
            metric: Comparable cost value; lower is better. Defaults to
                ``CNOTGateCount``. Tuples may be used for lexicographic ordering.

        Returns:
            ``(best_circuit, best_parameters)`` for the minimizing index.
        """

        if not isinstance(circs, list):
            raise Exception("First argument should be a list of squander circuits")

        if not isinstance(parameter_arrs, list):
            raise Exception("Second argument should be a list of numpy arrays")

        if len(circs) != len(parameter_arrs):
            raise Exception("The first two arguments should be of the same length")

        min_idx = min(range(len(circs)), key=lambda idx: metric(circs[idx]))

        return circs[min_idx], parameter_arrs[min_idx]

    @staticmethod
    def PartitionDecompositionProcess(
        subcircuit: Circuit,
        subcircuit_parameters: np.ndarray,
        config: dict,
        structure=None,
    ) -> Tuple[Circuit, np.ndarray]:
        """Decompose one partition subcircuit (multiprocessing-safe entry point).

        Args:
            subcircuit: Subcircuit acting on a subset of the wide register.
            subcircuit_parameters: Flat parameter vector slice for ``subcircuit``.
            config: Same keys as wide optimization (``strategy``, ``topology``, etc.).
            structure: Optional fixed gate structure when ``strategy == "Custom"``.

        Returns:
            Tuple of ``(decomposed_circuit, decomposed_parameters)`` pairs, each
            remapped back to the original qubit indices of ``subcircuit``.
        """

        qbit_num_orig_circuit = subcircuit.get_Qbit_Num()

        involved_qbits = subcircuit.get_Qbits()

        qbit_num = len(involved_qbits)

        # create qbit map:
        qbit_map = {}
        for idx in range(len(involved_qbits)):
            qbit_map[involved_qbits[idx]] = idx
        mini_topology = None
        if config["topology"] is not None:
            mini_topology = extract_subtopology(involved_qbits, qbit_map, config)
        # remap the subcircuit to a smaller qubit register
        remapped_subcircuit = subcircuit.Remap_Qbits(qbit_map, qbit_num)

        if not structure is None:
            structure = structure.Remap_Qbits(qbit_map, qbit_num)

        # get the unitary representing the circuit
        unitary = remapped_subcircuit.get_Matrix(
            np.asarray(subcircuit_parameters, dtype=np.float64)
        )

        # decompose a small unitary into a new circuit
        all_decomposed = qgd_Wide_Circuit_Optimization.DecomposePartition(
            unitary, config, mini_topology, structure=structure
        )
        # create inverse qbit map:
        inverse_qbit_map = {}
        for key, value in qbit_map.items():
            inverse_qbit_map[value] = key
        result = []
        for decomposed_circuit, decomposed_parameters in all_decomposed:

            # remap the decomposed circuit in order to insert it into a large circuit
            new_subcircuit = decomposed_circuit.Remap_Qbits(
                inverse_qbit_map, qbit_num_orig_circuit
            )

            if config["test_subcircuits"]:
                CompareCircuits(
                    subcircuit,
                    subcircuit_parameters,
                    new_subcircuit,
                    decomposed_parameters,
                    parallel=config["parallel"],
                    tolerance=_circuit_validation_tolerance(config),
                    report_overlap=config.get("verbosity", 0) >= 2,
                )

            new_subcircuit = new_subcircuit.get_Flat_Circuit()
            result.append((new_subcircuit, decomposed_parameters))
        return tuple(result)

    @staticmethod
    def build_partition_topo_deps(allparts):
        """Order partition gate-sets by dependencies and build a reverse-dependency map.

        Args:
            allparts: List of sets of gate indices, one per partition.

        Returns:
            ``(ordered_parts, rg_new)`` where ``ordered_parts`` lists partitions in
            topological order and ``rg_new`` maps each new index to predecessors.
        """
        gate_to_parts = {}
        for i, part in enumerate(allparts):
            for gate in part:
                gate_to_parts.setdefault(gate, set()).add(i)
        g = {i: set() for i in range(len(allparts))}
        rg = {i: set() for i in range(len(allparts))}
        for i, part in enumerate(allparts):
            for gate in part:
                for other_part in gate_to_parts[gate]:
                    if other_part != i and (
                        len(part & allparts[other_part]) > 0
                        and (len(part) < len(allparts[other_part]))
                        or part < allparts[other_part]
                    ):
                        g[i].add(other_part)
                        rg[other_part].add(i)
        rg_ret = {i: set(rg[i]) for i in range(len(allparts))}
        S = collections.deque(m for m in rg if len(rg[m]) == 0)
        L = []
        while S:
            n = S.popleft()
            L.append(n)
            for m in set(g[n]):
                g[n].remove(m)
                rg[m].remove(n)
                if len(rg[m]) == 0:
                    S.append(m)
        if len(L) != len(allparts):
            raise ValueError("Dependency graph is not a DAG")
        neworder = {old: new for new, old in enumerate(L)}
        rg_ret = {
            neworder[i]: set(neworder[j] for j in rg_ret[i])
            for i in range(len(allparts))
        }
        return [
            allparts[i] for i in L
        ], rg_ret  # return partitions in dependency order and dependencies

    @staticmethod
    def make_all_partition_circuit(circ, orig_parameters, max_partition_size):
        """ILP-based partitioning: flatten ``circ`` into a circuit of sub-circuits with concatenated parameters.

        Returns:
            ``(partitioned_circuit, parameters, recombine_info, part_deps)`` for later fusion in
            ``recombine_all_partition_circuit``.
        """
        from squander.partitioning.ilp import get_all_partitions, _get_topo_order

        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = (
            get_all_partitions(circ, max_partition_size)
        )
        qbit_num_orig_circuit = circ.get_Qbit_Num()
        gate_dict = {i: gate for i, gate in enumerate(circ.get_Gates())}
        single_qubit_chains_pre = {x[0]: x for x in single_qubit_chains if rgo[x[0]]}
        single_qubit_chains_post = {x[-1]: x for x in single_qubit_chains if go[x[-1]]}
        single_qubit_chains_prepost = {
            x[0]: x
            for x in single_qubit_chains
            if x[0] in single_qubit_chains_pre and x[-1] in single_qubit_chains_post
        }
        partitioned_circuit = Circuit(qbit_num_orig_circuit)
        params = []
        allparts, part_deps = qgd_Wide_Circuit_Optimization.build_partition_topo_deps(
            allparts
        )
        for part in allparts:
            surrounded_chains = {
                t
                for s in part
                for t in go[s]
                if t in single_qubit_chains_prepost
                and go[single_qubit_chains_prepost[t][-1]]
                and next(iter(go[single_qubit_chains_prepost[t][-1]])) in part
            }
            gates = frozenset.union(
                part, *(single_qubit_chains_prepost[v] for v in surrounded_chains)
            )
            # topo sort part + surrounded chains
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in _get_topo_order(
                {x: go[x] & gates for x in gates},
                {x: rgo[x] & gates for x in gates},
                gate_to_qubit,
            ):
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(
                    orig_parameters[
                        start : start + gate_dict[gate_idx].get_Parameter_Num()
                    ]
                )
            partitioned_circuit.add_Circuit(c)
        for chain in single_qubit_chains:
            c = Circuit(qbit_num_orig_circuit)
            for gate_idx in chain:
                c.add_Gate(gate_dict[gate_idx])
                start = gate_dict[gate_idx].get_Parameter_Start_Index()
                params.append(
                    orig_parameters[
                        start : start + gate_dict[gate_idx].get_Parameter_Num()
                    ]
                )
            partitioned_circuit.add_Circuit(c)
        parameters = np.concatenate(params, axis=0)
        return (
            partitioned_circuit,
            parameters,
            (allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit),
            part_deps,
        )

    @staticmethod
    def strip_single_qubit_head_tails(circ, params):
        """Drop single-qubit gates that sit only at the head or tail of the dependency DAG.

        Args:
            circ: Input circuit.
            params: Flat parameter array for ``circ``.

        Returns:
            ``(new_circuit, new_params)`` with head/tail single-qubit gates removed.
        """
        gate_dict, g, rg, gate_to_qubit, _ = build_dependency(circ)
        newcirc = Circuit(circ.get_Qbit_Num())
        new_params = []
        for i in gate_dict:
            gate = gate_dict[i]
            if len(gate_to_qubit[i]) == 1 and (len(g[i]) == 0 or len(rg[i]) == 0):
                continue
            newcirc.add_Gate(gate)
            start_idx = gate.get_Parameter_Start_Index()
            new_params.append(params[start_idx : start_idx + gate.get_Parameter_Num()])
        return newcirc, (
            np.empty((0,), dtype=np.float64)
            if len(new_params) == 0
            else np.concatenate(new_params, axis=0)
        )

    @staticmethod
    def get_fingerprint(circ, params):
        """Hashable signature of gate layout and parameters (for decomposition caching).

        Args:
            circ: Squander circuit.
            params: Parameter array associated with ``circ``.

        Returns:
            Tuple usable as a dict key for memoizing decompositions.
        """
        return (circ.get_Qbit_Num(),) + tuple(
            (gate.get_Name(), tuple(gate.get_Involved_Qbits()))
            for gate in circ.get_Gates()
        ) + tuple(params)

    @staticmethod
    def recombine_all_partition_circuit(
        circ,
        optimized_subcircuits,
        optimized_parameter_list,
        recombine_info,
        return_selection=False,
    ):
        """Reorder optimized partitions to respect global gate dependencies.

        Args:
            circ: Original flat circuit (for topological ordering context).
            optimized_subcircuits: One optimized subcircuit per partition slot.
            optimized_parameter_list: Parameter lists aligned with ``optimized_subcircuits``.
            recombine_info: Tuple from ``make_all_partition_circuit`` (ILP metadata).

        Returns:
            ``(reordered_circuits, reordered_parameter_lists)`` in execution order.
        """
        from squander.partitioning.ilp import (
            topo_sort_partitions,
            ilp_global_optimal,
            recombine_single_qubit_chains,
        )

        allparts, g, go, rgo, single_qubit_chains, gate_to_qubit, gate_to_tqubit = (
            recombine_info
        )
        # One additional CNOT must cost more than every possible difference in
        # the selected partitions' combined single-qubit count.
        cnot_weight = 1 + sum(
            sum(y for x, y in c.get_Gate_Nums().items() if CNOT_COUNT_DICT.get(x, -1) <= 0)
            for c in optimized_subcircuits[: len(allparts)]
        )
        weights = [
            CNOTGateCount(circ, cnot_weight)
            for circ in optimized_subcircuits[: len(allparts)]
        ]
        L, fusion_info = ilp_global_optimal(allparts, g, weights=weights)
        struct_idxs = list(L)
        parts = recombine_single_qubit_chains(
            go,
            rgo,
            single_qubit_chains,
            gate_to_tqubit,
            [allparts[i] for i in L],
            fusion_info,
            surrounded_only=True,
        )
        single_qubit_chain_idx = {
            frozenset(chain): idx + len(allparts)
            for idx, chain in enumerate(single_qubit_chains)
        }
        for extrapart in parts[len(struct_idxs) :]:
            struct_idxs.append(single_qubit_chain_idx[frozenset(extrapart)])
        L = topo_sort_partitions(circ, parts)
        selected_indices = [struct_idxs[i] for i in L]
        selected_gate_sets = [sorted(int(index) for index in parts[i]) for i in L]
        result = ([optimized_subcircuits[i] for i in selected_indices], [
            optimized_parameter_list[struct_idxs[i]] for i in L
        ])
        return (
            (*result, selected_indices, selected_gate_sets)
            if return_selection
            else result
        )

    def OptimizeWideCircuit(
        self, circ: Circuit, parameters: np.ndarray
    ) -> Tuple[Circuit, np.ndarray]:
        """Top-level wide-circuit pass: optional routing, then Qiskit / BQSKit / Squander partition optimization.

        Sets ``self.config`` timing and intermediate circuit keys (e.g. ``routed_circuit``, ``optimization_time``).
        """
        self.config.setdefault(
            "_rewrite_audit_stage",
            "all_to_all"
            if self.config["topology"] is None
            else "topology_optimization",
        )
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(
            circ, self.config["topology"]
        ):

            topo = self.config["topology"]
            self.config["topology"] = None
            strat = self.config["strategy"]
            self.config["strategy"] = self.config["pre-opt-strategy"]
            self.config["_rewrite_audit_stage"] = "all_to_all"

            circ, parameters = self.OptimizeWideCircuit(circ, parameters)
            self.config["all_to_all_optimization_time"] = self.config[
                "optimization_time"
            ]
            self.config["all_to_all_circuit"] = circ
            self.config["all_to_all_parameters"] = parameters
            self.config["strategy"] = strat
            self.config["topology"] = topo
            self.config["_rewrite_audit_stage"] = "routing"
            start_time = time.time()
            routing_input_cnot_count = CNOTGateCount(circ, 0)

            circ, parameters = self.route_circuit(circ, parameters)
            self.config["routing_time"] = time.time() - start_time
            print(
                f"Routing ({self.config.get('routing-strategy', 'exact-osr')}): "
                f"{routing_input_cnot_count} -> {CNOTGateCount(circ, 0)} CNOTs; "
                f"{self.config['routing_time']:.2f} s",
                flush=True,
            )
            self.config["routed_circuit"] = circ
            self.config["routed_parameters"] = parameters
            # Routing quality is archived above before the normal explicit
            # topology-constrained optimization stage. Routing implementations
            # must not hide cleanup internally, but this separately timed WCO
            # stage is required for every routing strategy.
            self.config.pop("topology_optimization_skipped", None)
            self.config["_rewrite_audit_stage"] = "topology_optimization"
        start_time = time.time()
        optimization_input_cnot_count = CNOTGateCount(circ, 0)
        if _rewrite_audit_enabled() and self.config["strategy"] == "qiskit":
            raise NotImplementedError(
                "Exact schema-v2 replay currently supports Squander-native and "
                "BQSKit synthesis, not opaque Qiskit transpiler mutations."
            )
        if self.config["strategy"] == "bqskit":
            from squander import Qiskit_IO
            from bqskit import compile
            import bqskit.compiler.compile as bqskit_compile_module

            from bqskit.compiler.machine import MachineModel
            from bqskit.compiler import Compiler
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            from bqskit.passes import SetModelPass
            from bqskit.compiler.compile import (
                build_multi_qudit_retarget_workflow,
                build_resynthesis_optimization_workflow,
                build_single_qudit_retarget_workflow,
                build_gate_deletion_optimization_workflow,
                LogErrorPass,
            )

            # Build BQSKit machine model from your topology
            model = MachineModel(circ.get_Qbit_Num(), self.config["topology"])
            synthesis_epsilon = _bqskit_synthesis_epsilon(self.config)

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(parameters, dtype=np.float64)
            )

            bqskit_input_qasm = qasm2.dumps(circo)
            bqskit_circ = OPENQASM2Language().decode(bqskit_input_qasm)

            with patched_seqpam_workflow_classes(
                bqskit_compile_module,
                use_squander_partitioner=False,
                config=self.config,
            ):
                compilation_workflow = [
                    SetModelPass(model),  # attach hardware model to circuit
                    build_multi_qudit_retarget_workflow(
                        4,
                        synthesis_epsilon=synthesis_epsilon,
                        max_synthesis_size=self.max_partition_size,
                    ),
                    build_resynthesis_optimization_workflow(
                        4,
                        synthesis_epsilon=synthesis_epsilon,
                        max_synthesis_size=self.max_partition_size,
                        iterative=True,
                    ),
                    build_single_qudit_retarget_workflow(
                        4,
                        synthesis_epsilon=synthesis_epsilon,
                        max_synthesis_size=self.max_partition_size,
                    ),
                    build_gate_deletion_optimization_workflow(
                        4,
                        synthesis_epsilon=synthesis_epsilon,
                        max_synthesis_size=self.max_partition_size,
                        iterative=True,
                    ),
                    LogErrorPass(),
                ]

            bqskit_stage_started_ns = time.time_ns()
            with Compiler() as compiler:
                routed_bqskit_circ, pass_data = compiler.compile(
                    bqskit_circ, compilation_workflow, True
                )

                default = list(range(bqskit_circ.num_qudits))
                initial_map = pass_data.get("initial_mapping", default)
                final_map = pass_data.get("final_mapping", default)

            bqskit_output_qasm = _bqskit_full_qasm(routed_bqskit_circ)
            if _rewrite_audit_enabled():
                _append_bqskit_stage_event(
                    self.config.get("_rewrite_audit_stage"),
                    bqskit_stage_started_ns,
                    bqskit_input_qasm,
                    bqskit_circ,
                    bqskit_output_qasm,
                    routed_bqskit_circ,
                    initial_map,
                    final_map,
                )

            # Convert back: BQSKit → Qiskit → Squander
            conversion_started_ns = time.time_ns()
            circuit_qiskit = QuantumCircuit.from_qasm_str(
                bqskit_output_qasm
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(
                circuit_qiskit
            )
            if _rewrite_audit_enabled():
                _append_bqskit_to_squander_event(
                    self.config.get("_rewrite_audit_stage"),
                    conversion_started_ns,
                    bqskit_output_qasm,
                    newcirc,
                    newparameters,
                )

            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters

        elif self.config["strategy"] == "qiskit":
            from squander import Qiskit_IO
            from qiskit import transpile
            from qiskit.transpiler import CouplingMap
            from squander.gates import gates_Wrapper as gate

            SUPPORTED_GATES_NAMES = {
                n.lower().replace("cnot", "cx")
                for n in dir(gate)
                if not n.startswith("_")
                and issubclass(getattr(gate, n), gate.Gate)
                and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")
            }
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(parameters, dtype=np.float64)
            )
            coupling_map = (
                None
                if self.config["topology"] is None
                else CouplingMap([[i, j] for i, j in self.config["topology"]])
            )
            circuit_qiskit = transpile(
                circo,
                basis_gates=SUPPORTED_GATES_NAMES,
                coupling_map=coupling_map,
                optimization_level=3,
            )
            newcirc, newparameters = Qiskit_IO.convert_Qiskit_to_Squander(
                circuit_qiskit
            )
            qgd_Wide_Circuit_Optimization.check_valid_routing(
                newcirc, self.config["topology"]
            )
            self.check_compare_circuits(circ, parameters, newcirc, newparameters)
            circ, parameters = newcirc, newparameters
        else:
            part_size_start = self.max_partition_size
            part_size_end = self.max_partition_size
            if self.config.get("auto_expand_partition_size", False) and (
                self.config.get("use_osr", False)
                or self.config.get("use_graph_search", False)
            ):
                part_size_end = min(4, circ.get_Qbit_Num())
            count = CNOTGateCount(circ, 0)
            fingerprint_dict = {}
            audit_round = 0
            for max_part_size in range(part_size_start, part_size_end + 1):
                # instantiate the object for optimizing wide circuits
                wide_circuit_optimizer = qgd_Wide_Circuit_Optimization(
                    {**self.config, "max_partition_size": max_part_size}
                )
                while True:
                    # run circuit optimization
                    circ_flat, parameters = (
                        wide_circuit_optimizer.InnerOptimizeWideCircuit(
                            circ,
                            parameters,
                            fingerprint_dict=fingerprint_dict,
                            audit_round=audit_round,
                        )
                    )
                    audit_round += 1
                    circ = circ_flat.get_Flat_Circuit()
                    newcount = CNOTGateCount(circ, 0)
                    no_improve = newcount >= count
                    count = newcount
                    if no_improve:
                        break
        self.config["optimization_time"] = time.time() - start_time
        if self.config["strategy"] in ("bqskit", "qiskit"):
            stage_name = (
                "All-to-all optimization"
                if self.config.get("_rewrite_audit_stage") == "all_to_all"
                else "Topology optimization"
            )
            strategy_name = (
                "BQSKit" if self.config["strategy"] == "bqskit" else "Qiskit"
            )
            print(
                f"{stage_name} ({strategy_name}): "
                f"{optimization_input_cnot_count} -> {CNOTGateCount(circ, 0)} CNOTs; "
                f"{self.config['optimization_time']:.2f} s",
                flush=True,
            )
        return circ, parameters

    def InnerOptimizeWideCircuit(
        self,
        circ: Circuit,
        orig_parameters: np.ndarray,
        fingerprint_dict=None,
        audit_round=None,
    ) -> Tuple[Circuit, np.ndarray]:
        """Optimize one pass of wide-circuit partition decomposition.

        The circuit is converted to a CNOT basis, partitioned, each partition is
        optimized (possibly in parallel), and then reconstructed into one circuit.

        Args:
            circ: Input circuit to optimize.
            orig_parameters: Parameter array associated with ``circ``.
            fingerprint_dict: Optional decomposition cache shared across passes.

        Returns:
            Tuple of ``(optimized_circuit, optimized_parameters)``.
        """
        from squander.utils import circuit_to_CNOT_basis

        progress_started = time.perf_counter()
        round_input_cnot_count = CNOTGateCount(circ, 0)
        audit_enabled = _rewrite_audit_enabled()
        if audit_enabled:
            basis_started_ns = time.time_ns()
            basis_input_representation = _squander_audit_representation(
                circ,
                orig_parameters,
                range(circ.get_Qbit_Num()),
            )
        circ, orig_parameters = circuit_to_CNOT_basis(circ, orig_parameters)
        if audit_enabled:
            basis_output_representation = _squander_audit_representation(
                circ,
                orig_parameters,
                range(circ.get_Qbit_Num()),
            )
            _append_rewrite_audit_event(
                {
                    "kind": "squander_basis_conversion",
                    "component": "squander_basis_replay",
                    "stage": self.config.get("_rewrite_audit_stage"),
                    "started_ns": basis_started_ns,
                    "input": basis_input_representation,
                    "output": basis_output_representation,
                    "input_sha256": _exact_state_sha256(
                        _qasm_exact_state(basis_input_representation)
                    ),
                    "output_sha256": _exact_state_sha256(
                        _qasm_exact_state(basis_output_representation)
                    ),
                }
            )
        round_started_ns = time.time_ns()
        round_input_representation = None
        round_input_state = None
        round_input_sha256 = None
        if audit_enabled:
            round_input_representation = _squander_audit_representation(
                circ,
                orig_parameters,
                range(circ.get_Qbit_Num()),
            )
            round_input_state = _qasm_exact_state(round_input_representation)
            round_input_sha256 = _exact_state_sha256(round_input_state)
        global_min = self.config.get("global_min", True)
        if audit_enabled and not global_min:
            raise NotImplementedError(
                "Exact rewrite replay currently requires global_min=True so "
                "every output block has explicit source gate indices."
            )
        if global_min:
            partitioned_circuit, parameters, recombine_info, part_deps = (
                qgd_Wide_Circuit_Optimization.make_all_partition_circuit(
                    circ, orig_parameters, self.max_partition_size
                )
            )

        else:
            partitioned_circuit, parameters, _ = PartitionCircuit(
                circ,
                orig_parameters,
                self.max_partition_size,
                strategy=self.config["partition_strategy"],
            )
            part_deps = None

        subcircuits = partitioned_circuit.get_Gates()

        # subcircuits = subcircuits[9:10]

        in_parent = parent_process() is not None

        # the list of optimized subcircuits
        optimized_subcircuits: List[Optional[Circuit]] = [None] * len(subcircuits)

        # the list of parameters associated with the optimized subcircuits
        optimized_parameter_list: List[Optional[List[np.ndarray]]] = [None] * len(
            subcircuits
        )

        # list of AsyncResult objects
        async_results = [None] * len(subcircuits)

        def process_result(partition_idx):
            """Finalize async decomposition for partition ``partition_idx`` and update caches / lists."""
            if optimized_subcircuits[partition_idx] is not None:
                return
            subcircuit = subcircuits[partition_idx]
            # callback on the master process to compare the decomposed and original subcircuit
            start_idx = subcircuit.get_Parameter_Start_Index()
            subcircuit_parameters = parameters[
                start_idx : start_idx + subcircuit.get_Parameter_Num()
            ]
            fingerprint = (
                None
                if fingerprint_dict is None
                else qgd_Wide_Circuit_Optimization.get_fingerprint(
                    subcircuit, subcircuit_parameters
                )
            )
            callback_fnc = lambda x: self.CompareAndPickCircuits(
                [subcircuit, *(z[0] for z in x)],
                [subcircuit_parameters, *(z[1] for z in x)],
                lambda c: (CNOTGateCount(c), SingleQubitGateCount(c)),
            )
            if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                new_subcircuit, new_parameters = fingerprint_dict[fingerprint]
            else:
                new_subcircuit, new_parameters = callback_fnc(
                    async_results[partition_idx][0](*async_results[partition_idx][1])
                    if in_parent
                    else async_results[partition_idx].get(timeout=None)
                )

                if fingerprint_dict is not None:
                    fingerprint_dict[fingerprint] = (new_subcircuit, new_parameters)
                    fingerprint_dict[
                        qgd_Wide_Circuit_Optimization.get_fingerprint(
                            new_subcircuit, new_parameters
                        )
                    ] = (new_subcircuit, new_parameters)
                    trim_subcirc, trim_parameters = (
                        qgd_Wide_Circuit_Optimization.strip_single_qubit_head_tails(
                            new_subcircuit, new_parameters
                        )
                    )
                    fingerprint_dict[
                        qgd_Wide_Circuit_Optimization.get_fingerprint(
                            trim_subcirc, trim_parameters
                        )
                    ] = (trim_subcirc, trim_parameters)
            optimized_subcircuits[partition_idx] = new_subcircuit
            optimized_parameter_list[partition_idx] = new_parameters

        worker_count = self.config.get("partition_workers")
        if worker_count is None:
            worker_count = mp.cpu_count()
        worker_count = max(1, min(worker_count, len(subcircuits)))
        with (
            contextlib.nullcontext() if in_parent else Pool(processes=worker_count)
        ) as pool:
            remaining = list(range(len(subcircuits)))
            while remaining:
                still_remaining = []
                #  code for iterate over partitions and optimize them
                for partition_idx in remaining:
                    subcircuit = subcircuits[partition_idx]

                    # isolate the parameters corresponding to the given sub-circuit
                    start_idx = subcircuit.get_Parameter_Start_Index()
                    end_idx = start_idx + subcircuit.get_Parameter_Num()
                    subcircuit_parameters = parameters[start_idx:end_idx]

                    fingerprint = (
                        None
                        if fingerprint_dict is None
                        else qgd_Wide_Circuit_Optimization.get_fingerprint(
                            subcircuit, subcircuit_parameters
                        )
                    )
                    if fingerprint_dict is not None and fingerprint in fingerprint_dict:
                        (
                            optimized_subcircuits[partition_idx],
                            optimized_parameter_list[partition_idx],
                        ) = fingerprint_dict[fingerprint]
                        continue
                    # With a required reduction of one CNOT, a CNOT-basis block
                    # containing at most one CNOT has no search depth to explore.
                    # Local one-qubit gates cannot reduce the operator Schmidt
                    # rank of a single CNOT, so avoid an exact-synthesis call and
                    # cache this mathematically irreducible result immediately.
                    if CNOTGateCount(subcircuit) <= 1:
                        optimized_subcircuits[partition_idx] = subcircuit
                        optimized_parameter_list[partition_idx] = (
                            subcircuit_parameters
                        )
                        if fingerprint_dict is not None:
                            fingerprint_dict[fingerprint] = (
                                subcircuit,
                                subcircuit_parameters,
                            )
                        continue
                    if part_deps is not None and partition_idx in part_deps:
                        any_optimized, any_remaining = False, False
                        for dep_idx in part_deps[partition_idx]:
                            if optimized_subcircuits[dep_idx] is None and (
                                async_results[dep_idx] is None
                                or not isinstance(async_results[dep_idx], tuple)
                                and not async_results[dep_idx].ready()
                            ):
                                any_remaining = True
                                continue
                            elif optimized_subcircuits[dep_idx] is None:
                                process_result(dep_idx)

                            optimized_subcircuits_loc = optimized_subcircuits[dep_idx]
                            assert isinstance(optimized_subcircuits_loc, Circuit)
                            assert optimized_subcircuits_loc is not None

                            if CNOTGateCount(optimized_subcircuits_loc) < CNOTGateCount(
                                subcircuits[dep_idx]
                            ):  # if the dependency partition was optimized, skip
                                any_optimized = True
                                break
                        if any_optimized:
                            optimized_subcircuits[partition_idx] = subcircuit
                            optimized_parameter_list[partition_idx] = (
                                subcircuit_parameters
                            )
                            continue
                        if any_remaining:
                            still_remaining.append(partition_idx)
                            continue
                    # call a process to decompose a subcircuit
                    config = {
                        **self.config,
                        "tree_level_max": qgd_Wide_Circuit_Optimization.partition_tree_level_max(
                            self.config, subcircuit
                        ),
                    }
                    fargs = (
                        self.PartitionDecompositionProcess,
                        (subcircuit, subcircuit_parameters, config, None),
                    )
                    # print("Dispatching", subcircuit.get_Involved_Qubits(), "qubits with", CNOGateCount(subcircuit, 0), "CNOT gates, partition ", partition_idx)
                    async_results[partition_idx] = (
                        fargs if in_parent else pool.apply_async(*fargs)  # type: ignore[union-attr]
                    )
                if len(remaining) == len(still_remaining):
                    time.sleep(0.1)
                remaining = still_remaining
            #  code for iterate over async results and retrieve the new subcircuits
            for partition_idx in range(len(subcircuits)):
                process_result(partition_idx)

        # construct the wide circuit from the optimized subcircuits
        if global_min:
            (
                optimized_subcircuits,
                optimized_parameter_list,
                selected_indices,
                selected_gate_sets,
            ) = (
                qgd_Wide_Circuit_Optimization.recombine_all_partition_circuit(
                    circ,
                    optimized_subcircuits,
                    optimized_parameter_list,
                    recombine_info,
                    return_selection=True,
                )
            )
        else:
            selected_indices = list(range(len(subcircuits)))
            selected_gate_sets = [None] * len(selected_indices)

        round_selections = []
        if audit_enabled:
            for output_index, partition_idx in enumerate(selected_indices):
                if partition_idx >= len(subcircuits):
                    raise AssertionError(
                        f"Selected partition {partition_idx} has no source block."
                    )
                before_circuit = subcircuits[partition_idx]
                after_circuit = optimized_subcircuits[output_index]
                start_idx = before_circuit.get_Parameter_Start_Index()
                before_parameters = parameters[
                    start_idx : start_idx + before_circuit.get_Parameter_Num()
                ]
                involved_qbits = sorted(before_circuit.get_Qbits())
                before = _squander_audit_representation(
                    before_circuit, before_parameters, involved_qbits
                )
                after = _squander_audit_representation(
                    after_circuit,
                    optimized_parameter_list[output_index],
                    involved_qbits,
                )
                source_gate_set = selected_gate_sets[output_index]
                before_global_state = _remap_exact_state(
                    _qasm_exact_state(before),
                    involved_qbits,
                    round_input_state["qubits"],
                )
                unused_source_indices = set(source_gate_set)
                source_gate_indices = []
                for operation in before_global_state["operations"]:
                    matching_index = next(
                        (
                            index
                            for index in sorted(unused_source_indices)
                            if round_input_state["operations"][index] == operation
                        ),
                        None,
                    )
                    if matching_index is None:
                        raise AssertionError(
                            "Partition operation has no matching source gate."
                        )
                    source_gate_indices.append(int(matching_index))
                    unused_source_indices.remove(matching_index)
                if unused_source_indices:
                    raise AssertionError(
                        "Partition did not consume every selected source gate."
                    )
                event_id = _append_rewrite_audit_event(
                    _make_rewrite_event(
                        "squander_selected_partition_rewrite",
                        before,
                        after,
                        _synthesis_acceptance_tolerance(self.config),
                        accepted=True,
                        stage=self.config.get("_rewrite_audit_stage"),
                        round=audit_round,
                        changed=before_circuit != after_circuit,
                        partition_index=int(partition_idx),
                        source_gate_indices=source_gate_indices,
                        qubits=[int(q) for q in involved_qbits],
                        max_partition_size=int(self.max_partition_size),
                    )
                )
                round_selections.append(
                    {
                        "rewrite_event_id": event_id,
                        "source_gate_indices": source_gate_indices,
                        "qubits": [int(q) for q in involved_qbits],
                    }
                )

        if any(c is None for c in optimized_subcircuits) or any(
            p is None for p in optimized_parameter_list
        ):
            raise RuntimeError(
                "Internal error: some partitions were not optimized before reconstruction."
            )
        wide_circuit, wide_parameters = self.ConstructCircuitFromPartitions(
            cast(List[Circuit], optimized_subcircuits),
            cast(List[List[np.ndarray]], optimized_parameter_list),
        )

        stage_name = (
            "All-to-all optimization"
            if self.config.get("_rewrite_audit_stage") == "all_to_all"
            else "Topology optimization"
        )
        round_label = (
            f" round {audit_round + 1}"
            if audit_round is not None
            else ""
        )
        print(
            f"{stage_name}{round_label} "
            f"({self.max_partition_size}-qubit partitions): "
            f"{round_input_cnot_count} -> {CNOTGateCount(wide_circuit, 0)} CNOTs; "
            f"{len(subcircuits)} candidates; "
            f"{time.perf_counter() - progress_started:.2f} s",
            flush=True,
        )

        qgd_Wide_Circuit_Optimization.check_valid_routing(
            wide_circuit, self.config["topology"]
        )
        self.check_compare_circuits(
            circ,
            orig_parameters,
            wide_circuit,
            wide_parameters,
        )

        if audit_enabled:
            round_output_representation = _squander_audit_representation(
                wide_circuit.get_Flat_Circuit(),
                wide_parameters,
                range(wide_circuit.get_Qbit_Num()),
            )
            round_output_state = _qasm_exact_state(round_output_representation)
            round_output_sha256 = _exact_state_sha256(round_output_state)
            _append_rewrite_audit_event(
                {
                    "kind": "round",
                    "component": "squander_wide_optimization",
                    "started_ns": round_started_ns,
                    "round": audit_round,
                    "stage": self.config.get("_rewrite_audit_stage"),
                    "max_partition_size": int(self.max_partition_size),
                    "input_sha256": round_input_sha256,
                    "output_sha256": round_output_sha256,
                    "input": round_input_representation,
                    "output": round_output_representation,
                    "selections": round_selections,
                    "input_gate_counts": circ.get_Gate_Nums(),
                    "output_gate_counts": wide_circuit.get_Gate_Nums(),
                }
            )

        return wide_circuit, wide_parameters

    @staticmethod
    def all_to_all_topology(num_qubits):
        """Undirected all-to-all coupler list for ``num_qubits`` qubits."""
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

    @staticmethod
    def linear_topology(num_qubits):
        """Path graph couplers ``(i, i+1)``."""
        return [(i, i + 1) for i in range(num_qubits - 1)]

    @staticmethod
    def star_topology(num_qubits):
        """Star graph: hub qubit ``0`` connected to all others."""
        return [(0, i) for i in range(1, num_qubits)]

    @staticmethod
    def ring_topology(num_qubits):
        """Ring couplers including wrap-around ``(n-1, 0)``."""
        return [(i, (i + 1) % num_qubits) for i in range(num_qubits)]

    @staticmethod
    def lattice_topology(x_qbits, y_qbits):
        """2D grid of size ``x_qbits`` by ``y_qbits`` with nearest-neighbor horizontal and vertical edges."""
        return [
            (i * x_qbits + j, i * x_qbits + (j + 1))
            for i in range(y_qbits)
            for j in range(x_qbits - 1)
        ] + [
            (i * x_qbits + j, (i + 1) * x_qbits + j)
            for i in range(y_qbits - 1)
            for j in range(x_qbits)
        ]

    @staticmethod
    def heavy_hexagonal_topology(rows, cols):
        """Build a finite heavy-hex coupling list (honeycomb with subdivided edges).

        Args:
            rows: Number of rows in the brick-wall honeycomb patch.
            cols: Number of columns in the patch.

        Returns:
            List of undirected edges ``(u, v)``. The first ``rows * cols`` qubit
            indices are honeycomb vertices; each original edge introduces one
            additional degree-2 qubit on the subdivided link.
        """

        def vid(r, c):
            """Linear index for honeycomb vertex at row ``r``, column ``c``."""
            return r * cols + c

        # Underlying honeycomb / brick-wall edges
        base_edges = []

        for r in range(rows):
            for c in range(cols):
                # Vertical brick-wall edges
                if r + 1 < rows:
                    base_edges.append((vid(r, c), vid(r + 1, c)))

                # Alternating horizontal edges
                if c + 1 < cols and ((r + c) % 2 == 0):
                    base_edges.append((vid(r, c), vid(r, c + 1)))

        # Subdivide every honeycomb edge by inserting a qubit
        next_id = rows * cols
        heavy_edges = []

        for u, v in base_edges:
            w = next_id
            next_id += 1
            heavy_edges.append((u, w))
            heavy_edges.append((w, v))

        return heavy_edges

    @staticmethod
    def sycamore_topology():
        """Approximate Sycamore-like 6x9 grid topology (simplified; ignores known dead qubits)."""
        return qgd_Wide_Circuit_Optimization.lattice_topology(
            6, 9
        )  # there is a defective qubit at (0, 3) in the sycamore chip, but we ignore it here for simplicity

    @staticmethod
    def is_valid_routing(wide_circuit, topo):
        """True if every multi-qubit gate's qubits lie in a connected subgraph of undirected ``topo``."""
        if topo is None:
            return True

        import itertools

        topo_set = {frozenset(edge) for edge in topo}

        def qubits_connected(qubits):
            """Whether pairwise couplers in ``topo_set`` connect all qubits in ``qubits``."""
            if len(qubits) <= 1:
                return True
            edges = {
                frozenset((q1, q2))
                for q1, q2 in itertools.combinations(qubits, 2)
                if frozenset((q1, q2)) in topo_set
            }
            if len(edges) == 0:
                return False
            cur_set = set(edges.pop())
            while edges:
                next_edge = next((e for e in edges if len(e & cur_set) > 0), None)
                if next_edge is None:
                    return False
                cur_set |= next_edge
                edges.remove(next_edge)
            return set(qubits) <= cur_set

        return all(
            qubits_connected(gate.get_Involved_Qbits())
            for gate in wide_circuit.get_Flat_Circuit().get_Gates()
            if len(gate.get_Involved_Qbits()) > 1
        )

    @staticmethod
    def check_valid_routing(wide_circuit, topo):
        """Assert ``is_valid_routing``; raises if any gate violates ``topo``."""
        if not qgd_Wide_Circuit_Optimization.is_valid_routing(wide_circuit, topo):
            import itertools, sys
            topo_set = {frozenset(e) for e in topo}
            for gate in wide_circuit.get_Flat_Circuit().get_Gates():
                qbits = gate.get_Involved_Qbits()
                if len(qbits) <= 1:
                    continue
                edges = {frozenset((q1,q2)) for q1,q2 in itertools.combinations(qbits,2) if frozenset((q1,q2)) in topo_set}
                if not edges:
                    sys.stderr.write(f'ROUTING_VIOLATION: {type(gate).__name__} on {qbits} topo={topo}\n')
                    sys.stderr.flush()
                    break
            raise AssertionError("Final circuit contains gates that do not respect the routing constraints.")

    def check_compare_circuits(
        self,
        circ,
        orig_parameters,
        wide_circuit,
        wide_parameters,
        routing=False,
        forced_test=False,
        label=None,
    ):
        """Optionally verify equivalence of ``circ`` and ``wide_circuit`` via ``CompareCircuits``.

        Args:
            circ: Original circuit.
            orig_parameters: Parameters for ``circ``.
            wide_circuit: Optimized or routed circuit.
            wide_parameters: Parameters for ``wide_circuit``.
            routing: If true and initial/final mappings exist in ``self.config``,
                pass them to ``CompareCircuits`` for layout-aware comparison.
            forced_test: If true, run the comparison for circuits of at most 12
                qubits even when ``test_final_circuit`` is false in config. Large
                comparisons require ``test_final_circuit=True`` explicitly.

        ``self.config['circuit_validation_tolerance']`` bounds
        ``1 - |<psi_original|psi_optimized>|`` for this whole-circuit random
        state-vector check. It is deliberately separate from
        ``self.config['tolerance']``, the per-block process infidelity
        ``1 - |Tr(U^dagger V)/d|^2`` used by Squander and converted exactly for
        BQSKit synthesis.
        """
        forced_test = circ.get_Qbit_Num() <= 12 and (
            forced_test
            or self.config.get("force_small_circuit_validation", True)
        )
        if self.config["test_final_circuit"] or forced_test:
            tolerance = _circuit_validation_tolerance(self.config)
            if (
                routing
                and self.config.get("initial_mapping", None) is not None
                and self.config.get("final_mapping", None) is not None
            ):
                CompareCircuits(
                    circ,
                    orig_parameters,
                    wide_circuit,
                    wide_parameters,
                    initial_mapping=self.config["initial_mapping"],
                    final_mapping=self.config["final_mapping"],
                    tolerance=tolerance,
                    parallel=0,
                    report_overlap=self.config.get("verbosity", 0) >= 2,
                )
            else:
                CompareCircuits(
                    circ,
                    orig_parameters,
                    wide_circuit,
                    wide_parameters,
                    tolerance=tolerance,
                    report_overlap=self.config.get("verbosity", 0) >= 2,
                )

    def route_circuit(self, circ: Circuit, orig_parameters: np.ndarray):
        """Map ``circ`` onto ``self.config['topology']`` using the configured router.

        The strategy is ``self.config['routing-strategy']``, e.g. ``pam-osr``, ``exact-osr``,
        ``seqpam-ilp``, ``seqpam-quick``, ``bqskit-sabre``, ``light-sabre``
        (Qiskit), or ``sabre`` (Squander). Writes ``initial_mapping`` and ``final_mapping`` into
        ``self.config`` when the backend provides them.

        Args:
            circ: Circuit before routing.
            orig_parameters: Parameter vector for ``circ``.

        Returns:
            ``(routed_circuit, routed_parameters)`` laid out for ``self.config['topology']``.
        """
        strategy = self.config.get("routing-strategy", "pam-osr")

        if _rewrite_audit_enabled() and strategy not in (
            "exact-osr",
            "pam-osr",
            "seqpam-ilp",
            "seqpam-quick",
            "light-sabre",
            "sabre",
        ):
            raise NotImplementedError(
                "Exact schema-v2 routing replay currently supports pam-osr, "
                "exact-osr, seqpam-ilp, seqpam-quick, light-sabre, and sabre only."
            )

        if strategy in ("pam-osr", "exact-osr"):
            from squander.partitioning.routing import route_circuit_exact

            exact_route_started_ns = time.time_ns()
            routing_config = dict(self.config)
            routing_config["pam_osr_only"] = strategy == "pam-osr"
            exact_route = route_circuit_exact(
                circ,
                orig_parameters,
                self.config["topology"],
                routing_config,
            )
            Squander_remapped_circuit = exact_route.circuit
            parameters_remapped_circuit = exact_route.parameters
            self.config["initial_mapping"] = list(
                exact_route.solution.initial_mapping
            )
            self.config["final_mapping"] = list(
                exact_route.solution.final_mapping
            )
            self.config["exact_routing_cnot_count"] = (
                exact_route.solution.cnot_count
            )
            self.config["exact_routing_single_qubit_count"] = (
                exact_route.solution.single_qubit_count
            )
            self.config["exact_routing_explored_states"] = (
                exact_route.solution.explored_states
            )
            self.config["exact_routing_master"] = self.config.get(
                "exact_routing_master", "benders"
            )
            self.config["exact_routing_master_used"] = (
                exact_route.solution.master_backend
            )
            self.config["exact_routing_candidate_count"] = len(
                exact_route.candidate_gate_sets
            )
            self.config["exact_routing_lazy_rounds"] = exact_route.lazy_rounds
            self.config["exact_routing_synthesized_partitions"] = (
                exact_route.synthesized_partitions
            )
            self.config["exact_routing_synthesis_cache_hits"] = (
                exact_route.synthesis_cache_hits
            )
            self.config["exact_routing_optimal"] = bool(
                exact_route.solution.optimal
            )
            self.config["exact_routing_cnot_optimal"] = bool(
                exact_route.solution.optimal
                or exact_route.solution.cnot_optimal
            )
            self.config["exact_routing_timed_out"] = bool(
                exact_route.timed_out
            )
            self.config["exact_routing_solver_nodes"] = (
                exact_route.solution.solver_nodes
            )
            self.config["exact_routing_solver_bound"] = (
                exact_route.solution.solver_bound
            )
            self.config["exact_routing_solver_gap"] = (
                exact_route.solution.solver_gap
            )
            self.config["exact_routing_solver_solutions"] = (
                exact_route.solution.solver_solutions
            )
            if _rewrite_audit_enabled():
                _append_exact_osr_routing_event(
                    self.config.get("_rewrite_audit_stage"),
                    exact_route_started_ns,
                    circ,
                    orig_parameters,
                    Squander_remapped_circuit,
                    parameters_remapped_circuit,
                    exact_route,
                    self.config["topology"],
                    _synthesis_acceptance_tolerance(self.config),
                )

        elif strategy in ("seqpam-ilp", "seqpam-quick", "bqskit-sabre"):
            from squander import Qiskit_IO
            import bqskit.compiler.compile as bqskit_compile_module
            from bqskit.compiler import Compiler
            from bqskit.compiler.compile import (
                build_sabre_mapping_workflow,
                build_seqpam_mapping_optimization_workflow,
            )

            from bqskit.passes import (
                SetModelPass,
            )
            from bqskit.compiler.machine import MachineModel
            from bqskit.ir.lang.qasm2 import OPENQASM2Language
            from qiskit import qasm2, QuantumCircuit

            # Build BQSKit machine model from your topology
            model = MachineModel(circ.get_Qbit_Num(), self.config["topology"])
            synthesis_epsilon = _bqskit_synthesis_epsilon(self.config)

            # Convert squander circuit → qiskit → BQSKit
            # (BQSKit has a from_qiskit helper if you go via Qiskit IR)
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(orig_parameters, dtype=np.float64)
            )

            bqskit_input_qasm = qasm2.dumps(circo)
            bqskit_circ = OPENQASM2Language().decode(bqskit_input_qasm)
            # Customizable knobs

            if strategy == "seqpam-ilp":
                # Routing-only SEQPAM pass pipeline. Patch the classes BQSKit's
                # workflow factory instantiates, so we do not depend on the private
                # shape of the returned Workflow.
                with patched_seqpam_workflow_classes(
                    bqskit_compile_module,
                    use_squander_partitioner=True,
                    config=self.config,
                ):
                    mainflow = build_seqpam_mapping_optimization_workflow(
                        synthesis_epsilon=synthesis_epsilon,
                        block_size=3,  # SEQPAM uses 3-qubit blocks only
                    )
                    if not self.config["seqpam_preoptimization"]:
                        mainflow = _remove_seqpam_preoptimization(mainflow)
            elif strategy == "seqpam-quick":
                # Keep BQSKit's QuickPartitioner. QSearch/LEAP are replaced
                # only when the configured optimizer is Squander-native.
                with patched_seqpam_workflow_classes(
                    bqskit_compile_module,
                    use_squander_partitioner=False,
                    config=self.config,
                ):
                    mainflow = build_seqpam_mapping_optimization_workflow(
                        synthesis_epsilon=synthesis_epsilon,
                        block_size=3,  # SEQPAM uses 3-qubit blocks only
                    )
                    if not self.config["seqpam_preoptimization"]:
                        mainflow = _remove_seqpam_preoptimization(mainflow)
            elif strategy == "bqskit-sabre":
                mainflow = build_sabre_mapping_workflow()
            else:
                raise ValueError(f"Unsupported BQSKit routing strategy: {strategy}")

            routing_workflow = [
                SetModelPass(model),  # attach hardware model to circuit
                mainflow,
            ]

            # EAPP monkey-patch catches Squander OSR failures per permutation
            # and installs a SWAP-correct fallback in BQSKit worker processes.
            import os as _os, json as _json
            old_patch_env = _os.environ.get('_SQUANDER_EAPP_FALLBACK_PATCH')
            old_config_env = _os.environ.get('_SQUANDER_BQSKIT_CONFIG')
            _os.environ['_SQUANDER_EAPP_FALLBACK_PATCH'] = '1'
            _os.environ['_SQUANDER_BQSKIT_CONFIG'] = _json.dumps(
                _copy_bqskit_synthesis_config(self.config)
            )
            _patch_eapp_if_needed()
            bqskit_stage_started_ns = time.time_ns()
            try:
                with Compiler() as compiler:
                    routed_bqskit_circ, pass_data = compiler.compile(
                        bqskit_circ, routing_workflow, True
                    )
            finally:
                if old_patch_env is None:
                    _os.environ.pop('_SQUANDER_EAPP_FALLBACK_PATCH', None)
                else:
                    _os.environ['_SQUANDER_EAPP_FALLBACK_PATCH'] = old_patch_env
                if old_config_env is None:
                    _os.environ.pop('_SQUANDER_BQSKIT_CONFIG', None)
                else:
                    _os.environ['_SQUANDER_BQSKIT_CONFIG'] = old_config_env

            bqskit_output_qasm = _bqskit_full_qasm(routed_bqskit_circ)
            if _rewrite_audit_enabled():
                _append_bqskit_stage_event(
                    self.config.get("_rewrite_audit_stage"),
                    bqskit_stage_started_ns,
                    bqskit_input_qasm,
                    bqskit_circ,
                    bqskit_output_qasm,
                    routed_bqskit_circ,
                    pass_data.initial_mapping,
                    pass_data.final_mapping,
                )

            # Convert back: BQSKit → Qiskit → Squander
            conversion_started_ns = time.time_ns()
            circuit_qiskit_routed = QuantumCircuit.from_qasm_str(
                bqskit_output_qasm
            )
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_routed)
            )
            if _rewrite_audit_enabled():
                _append_bqskit_to_squander_event(
                    self.config.get("_rewrite_audit_stage"),
                    conversion_started_ns,
                    bqskit_output_qasm,
                    Squander_remapped_circuit,
                    parameters_remapped_circuit,
                )
            self.config["initial_mapping"] = list(pass_data.initial_mapping)
            self.config["final_mapping"] = list(pass_data.final_mapping)

        elif strategy == "light-sabre":
            from squander import Qiskit_IO
            from qiskit import qasm2
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
            from qiskit.transpiler.passes import SabreLayout, SabreSwap
            from qiskit.transpiler import PassManager, CouplingMap
            from squander.gates import gates_Wrapper as gate

            # SUPPORTED_GATES_NAMES = {n.lower().replace("cnot", "cx") for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n not in ("Gate", "CROT", "CR", "SYC", "CCX", "CSWAP")}
            circo = Qiskit_IO.get_Qiskit_Circuit(
                circ, np.asarray(orig_parameters, dtype=np.float64)
            )
            coupling_map = [[i, j] for i, j in self.config["topology"]]
            # circuit_qiskit_sabre = transpile(circo, basis_gates=SUPPORTED_GATES_NAMES, coupling_map=coupling_map, optimization_level=0)
            coupling_map = CouplingMap(coupling_map)
            # Customizable SABRE parameters
            sabre_seed = self.config.get("sabre_seed", 42)
            sabre_trials = self.config.get("sabre_trials", 5)  # layout trials
            swap_trials = self.config.get("sabre_swap_trials", sabre_trials)
            sabre_max_iterations = self.config.get("sabre_max_iterations", 3)
            heuristic = self.config.get(
                "sabre_heuristic", "decay"
            )  # "basic" | "lookahead" | "decay"

            layout_pass = SabreLayout(
                coupling_map,
                seed=sabre_seed,
                max_iterations=sabre_max_iterations,
                swap_trials=swap_trials,
                layout_trials=sabre_trials,
            )
            swap_pass = SabreSwap(
                coupling_map,
                heuristic=heuristic,
                seed=sabre_seed,
                trials=swap_trials,
            )

            pm = PassManager(
                [
                    layout_pass,  # find initial qubit mapping via SABRE
                    swap_pass,  # insert SWAP gates for routing
                ]
            )
            sabre_started_ns = time.time_ns()
            sabre_input_qasm = qasm2.dumps(circo)
            circuit_qiskit_sabre = pm.run(circo)
            sabre_output_qasm = qasm2.dumps(circuit_qiskit_sabre)
            initial_mapping = (
                circuit_qiskit_sabre.layout.initial_index_layout()
            )
            final_mapping = (
                circuit_qiskit_sabre.layout.final_index_layout()
            )
            if _rewrite_audit_enabled():
                _append_qiskit_sabre_event(
                    self.config.get("_rewrite_audit_stage"),
                    sabre_started_ns,
                    sabre_input_qasm,
                    sabre_output_qasm,
                    initial_mapping,
                    final_mapping,
                    self.config["topology"],
                )
            conversion_started_ns = time.time_ns()
            Squander_remapped_circuit, parameters_remapped_circuit = (
                Qiskit_IO.convert_Qiskit_to_Squander(circuit_qiskit_sabre)
            )
            if _rewrite_audit_enabled():
                _append_bqskit_to_squander_event(
                    self.config.get("_rewrite_audit_stage"),
                    conversion_started_ns,
                    sabre_output_qasm,
                    Squander_remapped_circuit,
                    parameters_remapped_circuit,
                )
            self.config["initial_mapping"] = initial_mapping
            self.config["final_mapping"] = final_mapping
        elif strategy == "sabre":
            sabre_started_ns = time.time_ns()
            sabre = SABRE(circ, self.config["topology"])
            (
                Squander_remapped_circuit,
                parameters_remapped_circuit,
                pi,
                final_pi,
                swap_count,
            ) = sabre.map_circuit(orig_parameters)
            self.config["initial_mapping"] = pi
            self.config["final_mapping"] = final_pi
            if _rewrite_audit_enabled():
                _append_squander_sabre_event(
                    self.config.get("_rewrite_audit_stage"),
                    sabre_started_ns,
                    circ,
                    orig_parameters,
                    Squander_remapped_circuit,
                    parameters_remapped_circuit,
                    pi,
                    final_pi,
                    self.config["topology"],
                )
        qgd_Wide_Circuit_Optimization.check_valid_routing(
            Squander_remapped_circuit, self.config["topology"]
        )

        self.check_compare_circuits(
            circ,
            orig_parameters,
            Squander_remapped_circuit,
            parameters_remapped_circuit,
            routing=True,
            label="route_circuit",
        )
        return Squander_remapped_circuit, parameters_remapped_circuit
