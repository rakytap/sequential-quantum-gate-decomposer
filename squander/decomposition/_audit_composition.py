"""Bounded-memory, numerical composition certificates for native OSR rewrites.

Local phase-aligned spectral and normalized Frobenius errors compose by the
triangle inequality for unitaries. Exact gate-stream replay and causal ordering
establish that the checked local maps are the maps actually used. Permutations
and spectator identities preserve both norms (normalize Frobenius by sqrt(d)).
Either squared sum bounds whole-circuit process infidelity; keep both bounds
and use the smaller one. Within a consecutive layer of disjoint supports,
normalized squared Frobenius errors add rather than distances: the normalized
trace overlap factors over tensor products. Exact entanglers remain in the
support sequence, and opaque exact blocks terminate a layer. Unitarity defects
supply polar-projection allowances.
These are complex128 numerical certificates, not interval proofs.
"""
import collections
import contextvars
import functools
import json
import math
import os
import sqlite3
import sys
import tempfile

import numpy as np


STATE_VALIDATION_MAX_QUBITS = 20
_ACTIVE_CERTIFICATE = contextvars.ContextVar("native_osr_composition", default=None)
_nextafter = getattr(math, "nextafter", lambda x, y: float(np.nextafter(x, y)))


def current_certificate():
    certificate = _ACTIVE_CERTIFICATE.get()
    if certificate is not None and certificate.pid == os.getpid():
        return certificate
    return None


def _up_add(left, right):
    return _nextafter(left + right, math.inf) if right else left


def _permutation(values, width, label):
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise AssertionError(label + " must contain integer vertices.")
    if sorted(values) != list(range(width)):
        raise AssertionError(label + " must be a permutation.")
    return values


class NativeAuditComposition:
    """Incrementally replay native events; keep the rewrite ledger on disk."""

    def __init__(self, backend, initial, tolerance, local_tolerance, topology=None):
        self.backend = backend
        self.pid = os.getpid()
        self.tolerance = float(tolerance)
        self.local_tolerance = float(local_tolerance)
        if not (0 < self.tolerance < 1 and 0 < self.local_tolerance < 1):
            raise ValueError("Composition tolerances must be finite and in (0, 1).")
        self.state = backend._qasm_exact_state(initial)
        self.width = self.state["qubits"]
        self.initial_digest = backend._exact_state_sha256(self.state)
        self.topology = None if topology is None else {
            frozenset(map(int, edge)) for edge in topology
        }
        self.spectral_sum = 0.0
        self.frobenius_sum = 0.0
        self.disjoint_error_layers = 0
        self.defect_sum = 0.0
        self.worst_local_error = 0.0
        self.selected_rewrites = 0
        self.selected_routing_columns = 0
        self.changed_basis_conversions = 0
        self.transitions = 0
        self.last_input_digest = None
        self.last_route = None
        self.frames = []
        self.matrix_cache = collections.OrderedDict()
        self.basis_cache = collections.OrderedDict()
        self.temporary = tempfile.TemporaryDirectory(prefix="squander-composition-")
        self.database = sqlite3.connect(os.path.join(self.temporary.name, "rewrites.sqlite"))
        self.database.execute("PRAGMA cache_size=-2048")
        self.database.execute("PRAGMA journal_mode=OFF")
        self.database.execute("PRAGMA synchronous=OFF")
        self.database.execute(
            "CREATE TABLE rewrites (id TEXT PRIMARY KEY, event TEXT, "
            "delta REAL, defect REAL, width INTEGER, frobenius_delta REAL)"
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.database.close()
        self.temporary.cleanup()

    def get(self, event_id):
        row = self.database.execute(
            "SELECT event FROM rewrites WHERE id=?", (event_id,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def _digest(self, state):
        return self.backend._exact_state_sha256(state)

    def require_current(self, representation):
        state = self.backend._qasm_exact_state(representation)
        if state != self.state:
            raise AssertionError(
                "Composition certificate does not reach the exact circuit gate stream."
            )

    def enter_frame(self, representation):
        self.require_current(representation)
        self.frames.append(self._digest(self.state))

    def leave_frame(self):
        self.frames.pop()

    def assert_comparison(self, before, after, routing=False, initial=None, final=None):
        self.require_current(after)
        before_digest = self._digest(self.backend._qasm_exact_state(before))
        if before_digest not in self.frames + [self.last_input_digest]:
            raise AssertionError("Comparison input is outside the certified transition.")
        if routing:
            if self.last_route is None:
                raise AssertionError("Routing comparison lacks a selected-route certificate.")
            if (tuple(initial or ()) != tuple(self.last_route["initial_mapping"])
                    or tuple(final or ()) != tuple(self.last_route["final_mapping"])):
                raise AssertionError("Routing comparison mappings differ from its certificate.")
        self.report()

    def _matrix(self, representation):
        if representation.get("format") != "squander_openqasm2":
            raise AssertionError("Composition requires native OpenQASM block representations.")
        width = representation.get("qubits")
        if type(width) is not int or not 1 <= width <= 4:
            raise AssertionError("Composition local matrices are restricted to 1-4 qubits.")
        key = (width, representation["qasm"])
        cached = self.matrix_cache.get(key)
        if cached is not None:
            self.matrix_cache.move_to_end(key)
            return cached
        from qiskit import qasm2
        from qiskit.circuit.library import UGate
        from qiskit.quantum_info import Operator
        instructions = [item for item in qasm2.LEGACY_CUSTOM_INSTRUCTIONS if item.name != "u"] + [
            qasm2.CustomInstruction("u", 3, 1, UGate, builtin=True)
        ]
        circuit = qasm2.loads(representation["qasm"], custom_instructions=instructions)
        if circuit.num_qubits != width or circuit.num_clbits:
            raise AssertionError("Local representation has inconsistent or classical wires.")
        matrix = np.asarray(Operator(circuit).data, dtype=np.complex128)
        if not np.isfinite(matrix).all():
            raise AssertionError("Local unitary contains nonfinite entries.")
        matrix.setflags(write=False)
        self.matrix_cache[key] = matrix
        if len(self.matrix_cache) > 512:
            self.matrix_cache.popitem(last=False)
        return matrix

    def _score(self, before, after, tolerance):
        limit = float(tolerance)
        if not (0 < limit <= self.local_tolerance):
            raise AssertionError("Local certificate cannot relax synthesis acceptance.")
        if before.shape != after.shape:
            raise AssertionError("Local matrix dimensions differ.")
        overlap = np.vdot(before, after)
        if not np.isfinite(overlap) or not abs(overlap):
            raise AssertionError("Local phase alignment is undefined.")
        difference = after - overlap / abs(overlap) * before
        delta = float(np.linalg.norm(difference, 2))
        chord = float(np.vdot(difference, difference).real / before.shape[0])
        error = chord * (1.0 - chord / 4.0)
        identity = np.eye(before.shape[0], dtype=np.complex128)
        before_defect = float(np.linalg.norm(before.conj().T @ before - identity, 2))
        after_defect = float(np.linalg.norm(after.conj().T @ after - identity, 2))
        if not (0 <= chord <= 4 and 0 <= error <= limit):
            raise AssertionError("Selected local synthesis exceeds acceptance tolerance.")
        if max(before_defect, after_defect) > limit:
            raise AssertionError("Selected local matrices are not unitary within tolerance.")
        self.worst_local_error = max(self.worst_local_error, error)
        frobenius_delta = _nextafter(math.sqrt(chord), math.inf) if chord else 0.0
        return delta, _up_add(before_defect, after_defect), frobenius_delta

    def _add_error(self, delta, defect, frobenius_delta=None):
        self.spectral_sum = _up_add(self.spectral_sum, delta)
        # A spectral distance is also a valid normalized Frobenius bound when
        # only the legacy two-distance interface supplies an error allowance.
        self.frobenius_sum = _up_add(
            self.frobenius_sum, delta if frobenius_delta is None else frobenius_delta
        )
        self.defect_sum = _up_add(self.defect_sum, defect)

    def _add_error_layers(self, errors):
        """Compose only consecutive disjoint supports, preserving gate order.

        For phase-aligned unitary pairs, chord_i = distance_i**2 and
        overlap_i = 1 - chord_i/2. A tensor layer has overlap equal to the
        product of its local overlaps, so its chord is at most sum(chord_i).
        Every block's support counts, including blocks with zero error:
        an exact entangler can transport earlier errors onto later supports.
        None is a conservative barrier for an opaque exactly replayed block.
        """
        occupied = set()
        layer_square = 0.0

        def finish_layer():
            if not occupied:
                return
            if layer_square:
                distance = _nextafter(math.sqrt(layer_square), math.inf)
                self.frobenius_sum = _up_add(self.frobenius_sum, distance)
            self.disjoint_error_layers += 1

        for error in errors:
            if error is None:
                finish_layer()
                occupied.clear()
                layer_square = 0.0
                continue
            delta, defect, frobenius_delta, qubits = error
            support = set(qubits)
            if occupied.intersection(support):
                finish_layer()
                occupied.clear()
                layer_square = 0.0
            occupied.update(support)
            self.spectral_sum = _up_add(self.spectral_sum, delta)
            self.defect_sum = _up_add(self.defect_sum, defect)
            if frobenius_delta:
                square = _nextafter(frobenius_delta * frobenius_delta, math.inf)
                layer_square = _up_add(layer_square, square)
        finish_layer()

    def _transition_states(self, event):
        before = self.backend._qasm_exact_state(event["input"])
        after = self.backend._qasm_exact_state(event["output"])
        if self._digest(before) != event["input_sha256"]:
            raise AssertionError("Composition input hash mismatch.")
        if self._digest(after) != event["output_sha256"]:
            raise AssertionError("Composition output hash mismatch.")
        if before != self.state or after["qubits"] != self.width:
            raise AssertionError("Composition gate-stream chain is broken.")
        return before, after

    def _commit(self, before, after):
        self.last_input_digest = self._digest(before)
        self.state = after
        self.transitions += 1
        self.report()

    def _topology_check(self, state):
        if self.topology is None:
            raise AssertionError("Topology transition lacks a device graph.")
        for operation in state["operations"]:
            wires = operation["qubits"]
            if len(wires) > 1 and (
                len(wires) != 2 or frozenset(wires) not in self.topology
            ):
                raise AssertionError("Composition output violates device topology.")

    def _basis_conversion(self, event, before, after):
        if before == after:
            return
        if not event.get("conversion_applied", True):
            raise AssertionError("Skipped basis conversion changed the gate stream.")
        from qiskit import QuantumCircuit, qasm2
        from qiskit.circuit.library import UGate
        from squander import Qiskit_IO, utils
        from squander.decomposition._audit_qasm import dump_native_audit_qasm
        instructions = [item for item in qasm2.LEGACY_CUSTOM_INSTRUCTIONS if item.name != "u"] + [
            qasm2.CustomInstruction("u", 3, 1, UGate, builtin=True)
        ]
        source = qasm2.loads(event["input"]["qasm"], custom_instructions=instructions)
        if source.num_qubits != self.width or source.num_clbits:
            raise AssertionError("Basis conversion has inconsistent or classical wires.")
        expected = QuantumCircuit(self.width)
        errors = []
        for item in source.data:
            operation = item.operation
            width = len(item.qubits)
            if item.clbits or not 1 <= width <= 4:
                raise AssertionError("Basis conversion needs a compact gate identity.")
            if width == 1 or operation.name == "cx":
                # The production converter copies these gates and parameters
                # verbatim. Certify that identity by exact output-stream replay,
                # rather than round-tripping unchanged gates through an importer.
                expected.append(operation, [source.find_bit(q).index for q in item.qubits])
                errors.append((0.0, 0.0, 0.0,
                               tuple(source.find_bit(q).index for q in item.qubits)))
                continue
            key = (operation.name, width, tuple(complex(x) for x in operation.params))
            cached = self.basis_cache.get(key)
            if cached is None:
                compact = QuantumCircuit(width)
                compact.append(operation, range(width))
                native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(compact)
                converted, converted_parameters = utils.circuit_to_CNOT_basis(
                    native, parameters
                )
                converted_qiskit = Qiskit_IO.get_Qiskit_Circuit(
                    converted, np.asarray(converted_parameters, dtype=np.float64)
                )
                representations = [
                    {"format": "squander_openqasm2", "qubits": width,
                     "qasm": dump_native_audit_qasm(c)}
                    for c in (compact, converted_qiskit)
                ]
                delta, defect, frobenius_delta = self._score(
                    self._matrix(representations[0]),
                    self._matrix(representations[1]),
                    self.local_tolerance,
                )
                cached = (converted_qiskit, delta, defect, frobenius_delta)
                self.basis_cache[key] = cached
                if len(self.basis_cache) > 256:
                    self.basis_cache.popitem(last=False)
            else:
                self.basis_cache.move_to_end(key)
            converted_qiskit, delta, defect, frobenius_delta = cached
            expected.compose(
                converted_qiskit,
                qubits=[source.find_bit(q).index for q in item.qubits],
                inplace=True,
            )
            errors.append((delta, defect, frobenius_delta,
                           tuple(source.find_bit(q).index for q in item.qubits)))
        replayed = self.backend._qasm_exact_state({
            "format": "squander_openqasm2", "qubits": self.width,
            "qasm": dump_native_audit_qasm(expected),
        })
        if replayed != after:
            raise AssertionError("Compact basis identities do not reproduce exact output.")
        self._add_error_layers(errors)
        self.changed_basis_conversions += 1

    def _native_round(self, event, before, after):
        replay_before, replay_after = self.backend._verify_native_round_replay(event, self)
        if (replay_before, replay_after) != (before, after):
            raise AssertionError("Native replay and composition states disagree.")
        predecessors, _ = self.backend._circuit_dependencies(before)
        consumed = set()
        errors = []
        for selection in event["selections"]:
            for index in selection["source_gate_indices"]:
                if not predecessors[index] <= consumed:
                    raise AssertionError("Native selections violate causal gate order.")
                consumed.add(index)
            if selection.get("passthrough") is True:
                # Keep exact entanglers in the support ordering even though
                # an unchanged gate stream contributes zero rewrite error.
                errors.append((0.0, 0.0, 0.0, selection["qubits"]))
                self.selected_rewrites += 1
                continue
            row = self.database.execute(
                "SELECT delta,defect,width,frobenius_delta FROM rewrites WHERE id=?",
                (selection["rewrite_event_id"],),
            ).fetchone()
            if row is None or row[2] != len(selection["qubits"]):
                raise AssertionError("Native selection lacks its compact error certificate.")
            errors.append((row[0], row[1], row[3], selection["qubits"]))
            self.selected_rewrites += 1
        stage = event.get("stage")
        # A direct public InnerOptimizeWideCircuit call need not have the
        # top-level stage label. Its actual absent topology is unrestricted.
        if stage != "all_to_all" and not (stage is None and self.topology is None):
            self._topology_check(after)
        self._add_error_layers(errors)

    def _routing(self, event, before, after):
        for name in ("initial_mapping", "final_mapping"):
            _permutation(event[name], self.width, name)
        graph = {frozenset(map(int, edge)) for edge in event["topology"]}
        if graph != self.topology:
            raise AssertionError("Routing certificate uses a different device graph.")
        errors = []
        for selection in event["selections"]:
            kind = selection.get("certificate_kind", "unitary")
            if kind == "sabre":
                # The exact replayer below verifies each preserved gate and SWAP.
                errors.append(None)
                continue
            if kind != "unitary":
                raise AssertionError("Unsupported routing certificate.")
            width = selection["source"]["qubits"]
            source = self._matrix(selection["source"])
            synthesized = self._matrix(selection["synthesized"])
            pi = _permutation(selection["input_assignment"], width, "input assignment")
            po = _permutation(selection["output_assignment"], width, "output assignment")
            for name in ("logical_qubits", "input_physical", "output_physical", "embedding"):
                values = selection[name]
                if (len(values) != width or len(set(values)) != width
                        or any(type(q) is not int or not 0 <= q < self.width for q in values)):
                    raise AssertionError("Routing block has invalid vertices.")
            embedding = selection["embedding"]
            if embedding != [selection["input_physical"][q] for q in pi]:
                raise AssertionError("Routing input assignment does not match its embedding.")
            if embedding != [selection["output_physical"][q] for q in po]:
                raise AssertionError("Routing output assignment does not match its embedding.")
            # Assignments map local physical positions to local logical positions.
            # Reconstruct both basis-index maps independently of the native router.
            def indices(assignment):
                return [
                    sum(((basis >> physical) & 1) << logical
                        for physical, logical in enumerate(assignment))
                    for basis in range(1 << width)
                ]
            target = source[np.ix_(indices(po), indices(pi))]
            delta, defect, frobenius_delta = self._score(
                target, synthesized, selection["tolerance"]
            )
            # Undoing the checked boundary permutations places each column
            # in its source logical frame. Disjointness must use those logical
            # supports, not physical windows moved by intervening SWAPs.
            errors.append((delta, defect, frobenius_delta, selection["logical_qubits"]))
            self.selected_routing_columns += 1
        replay_before, replay_after = self.backend._verify_exact_osr_routing_replay(event)
        if (replay_before, replay_after) != (before, after):
            raise AssertionError("Routing replay and composition states disagree.")
        self._topology_check(after)
        self._add_error_layers(errors)
        self.last_route = event

    def consume(self, event):
        kind = event.get("kind")
        if kind == "synthesis_attempt":
            return
        if kind == "rewrite":
            if not event.get("accepted", False):
                return
            before = self._matrix(event["before"])
            after = self._matrix(event["after"])
            delta, defect, frobenius_delta = self._score(before, after, event["tolerance"])
            self.database.execute(
                "INSERT INTO rewrites VALUES (?,?,?,?,?,?)",
                (event["event_id"], json.dumps(event, allow_nan=False),
                 delta, defect, event["before"]["qubits"], frobenius_delta),
            )
            return
        before, after = self._transition_states(event)
        if kind == "squander_basis_conversion":
            self._basis_conversion(event, before, after)
        elif kind == "round":
            self._native_round(event, before, after)
        elif kind == "exact_osr_routing":
            self._routing(event, before, after)
        else:
            raise AssertionError("Unsupported whole-circuit composition transition: " + str(kind))
        self._commit(before, after)

    def report(self):
        spectral_total = _up_add(self.spectral_sum, self.defect_sum)
        frobenius_total = _up_add(self.frobenius_sum, self.defect_sum)
        spectral_bound = (
            _nextafter(spectral_total * spectral_total, math.inf)
            if spectral_total else 0.0
        )
        frobenius_bound = (
            _nextafter(frobenius_total * frobenius_total, math.inf)
            if frobenius_total else 0.0
        )
        # For phase-aligned unitaries, process infidelity is at most
        # ||U - phase*V||_F**2 / d. Triangle composition preserves this
        # normalized norm even for differently supported spectator embeddings.
        bound = min(spectral_bound, frobenius_bound)
        if not math.isfinite(bound) or bound > self.tolerance:
            raise AssertionError("Accumulated composition bound exceeds whole-circuit tolerance.")
        return {
            "status": "certified",
            "scope": "complex128 numerical composition, not interval arithmetic",
            "qubits": self.width,
            "input_circuit_sha256": self.initial_digest,
            "output_circuit_sha256": self._digest(self.state),
            "selected_rewrites": self.selected_rewrites,
            "selected_routing_columns": self.selected_routing_columns,
            "changed_basis_conversions": self.changed_basis_conversions,
            "replayed_transitions": self.transitions,
            "phase_aligned_spectral_sum": self.spectral_sum,
            "phase_aligned_normalized_frobenius_sum": self.frobenius_sum,
            "disjoint_error_layers": self.disjoint_error_layers,
            "error_composition_policy": "spectral-triangle-and-disjoint-frobenius-layers",
            "unitarity_defect_sum": self.defect_sum,
            "global_process_infidelity_spectral_bound": spectral_bound,
            "global_process_infidelity_normalized_frobenius_bound": frobenius_bound,
            "global_process_infidelity_bound": bound,
            "whole_circuit_tolerance": self.tolerance,
            "worst_selected_local_process_infidelity": self.worst_local_error,
            "initial_mapping": None if self.last_route is None else self.last_route["initial_mapping"],
            "final_mapping": None if self.last_route is None else self.last_route["final_mapping"],
        }


class certificate_scope:
    """Temporarily isolate a candidate's certificate from its caller's chain.

    A rejected routing candidate must never advance the parent certificate.
    ContextVar tokens also restore the caller when candidate validation raises.
    """

    def __init__(self, certificate=None):
        self.certificate = certificate
        self.token = None

    def __enter__(self):
        if self.token is not None:
            raise RuntimeError("A certificate scope cannot be entered twice.")
        self.token = _ACTIVE_CERTIFICATE.set(self.certificate)
        return self.certificate

    def __exit__(self, *exc):
        _ACTIVE_CERTIFICATE.reset(self.token)
        self.token = None
        return False


def native_composition_guard(method):
    """Replace large native random-state validation with an active certificate."""
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        circuit = args[0] if args else kwargs.get("circ")
        parameters = args[1] if len(args) > 1 else kwargs.get(
            "parameters", kwargs.get("orig_parameters"))
        if circuit is None:
            return method(self, *args, **kwargs)
        native_osr = self.config.get("strategy") == "TreeSearch" and (
            self.config.get("use_osr", False) or self.config.get("use_graph_search", False))
        if (not native_osr or circuit.get_Qbit_Num() <= STATE_VALIDATION_MAX_QUBITS
                or not self.config.get("test_final_circuit", True)):
            return method(self, *args, **kwargs)
        backend = sys.modules[method.__module__]
        eligible = (
            self.config.get("strategy") == "TreeSearch"
            and (self.config.get("use_osr", False) or self.config.get("use_graph_search", False))
            and 1 <= self.max_partition_size <= 4
            and self.config.get("global_min", True)
        )
        needs_routing = method.__name__ == "route_circuit" or (
            method.__name__ == "OptimizeWideCircuit"
            and not self.is_valid_routing(circuit, self.config.get("topology"))
        )
        if (not eligible or (needs_routing and (
                self.config.get("routing-strategy") != "exact-osr"
                or self.config.get("pre-opt-strategy", "TreeSearch") != "TreeSearch"))):
            raise NotImplementedError(
                "Large-circuit validation requires replayable native 1-4-qubit "
                "OSR partitions and exact-osr routing; no exponential state is allocated."
            )
        initial = backend._squander_audit_representation(
            circuit, parameters, range(circuit.get_Qbit_Num())
        )
        active = current_certificate()
        if active is not None:
            active.enter_frame(initial)
            try:
                result = method(self, *args, **kwargs)
                active.require_current(backend._squander_audit_representation(
                    result[0], result[1], range(result[0].get_Qbit_Num())
                ))
                active.report()
                return result
            finally:
                active.leave_frame()
        self.config.pop("circuit_validation_certificate", None)
        with NativeAuditComposition(
            backend, initial, backend._circuit_validation_tolerance(self.config),
            backend._synthesis_acceptance_tolerance(self.config),
            topology=self.config.get("topology"),
        ) as certificate:
            token = _ACTIVE_CERTIFICATE.set(certificate)
            certificate.enter_frame(initial)
            try:
                result = method(self, *args, **kwargs)
                certificate.require_current(backend._squander_audit_representation(
                    result[0], result[1], range(result[0].get_Qbit_Num())
                ))
                self.config["circuit_validation_certificate"] = certificate.report()
                return result
            finally:
                certificate.leave_frame()
                _ACTIVE_CERTIFICATE.reset(token)
    return guarded
