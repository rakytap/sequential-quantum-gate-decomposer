"""Independent positive and fail-closed controls for composition validation."""
import importlib

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2

from squander.decomposition import _audit_composition as composition

backend = importlib.import_module("squander.decomposition.qgd_Wide_Circuit_Optimization")


def representation(circuit):
    return {"format": "squander_openqasm2", "qubits": circuit.num_qubits,
            "qasm": qasm2.dumps(circuit)}


def transition(kind, before, after, **extra):
    event = {"kind": kind, "input": representation(before), "output": representation(after)}
    for key in ("input", "output"):
        event[key + "_sha256"] = backend._exact_state_sha256(
            backend._qasm_exact_state(event[key]))
    event.update(extra)
    return event


def rewrite(before, after, event_id="selected", tolerance=1e-10):
    return {"kind": "rewrite", "accepted": True, "event_id": event_id,
            "before": representation(before), "after": representation(after),
            "tolerance": tolerance}


def selection(event_id, indices, qubits):
    return {"rewrite_event_id": event_id, "source_gate_indices": indices, "qubits": qubits}


def certificate(circuit, tolerance=1e-8, topology=None):
    return composition.NativeAuditComposition(
        backend, representation(circuit), tolerance, 1e-10, topology)


def test_large_identity_rewrite_is_certified_without_a_global_matrix():
    full = QuantumCircuit(30)
    full.cx(0, 29)
    local = QuantumCircuit(2)
    local.cx(0, 1)
    with certificate(full) as proof:
        proof.consume(rewrite(local, local))
        proof.consume(transition("round", full, full, stage="all_to_all",
            selections=[selection("selected", [0], [0, 29])]))
        proof.require_current(representation(full))
        report = proof.report()
        assert report["selected_rewrites"] == 1
        assert report["global_process_infidelity_bound"] == 0
        assert all(key[0] <= 4 for key in proof.matrix_cache)


def test_selected_rewrite_occurrences_are_counted_not_only_unique_ids():
    before = QuantumCircuit(21)
    before.x(0)
    before.x(1)
    after = QuantumCircuit(21)
    after.rx(np.pi, 0)
    after.rx(np.pi, 1)
    local_before = QuantumCircuit(1)
    local_before.x(0)
    local_after = QuantumCircuit(1)
    local_after.rx(np.pi, 0)
    with certificate(before) as proof:
        proof.consume(rewrite(local_before, local_after))
        delta = proof.database.execute("SELECT delta FROM rewrites").fetchone()[0]
        proof.consume(transition("round", before, after, stage="all_to_all", selections=[
            selection("selected", [0], [0]), selection("selected", [1], [1])]))
        assert proof.report()["selected_rewrites"] == 2
        assert proof.spectral_sum == pytest.approx(2 * delta, rel=1e-14, abs=1e-30)


def test_causal_reordering_of_noncommuting_gates_is_rejected():
    before = QuantumCircuit(21)
    before.h(0)
    before.x(0)
    after = QuantumCircuit(21)
    after.x(0)
    after.h(0)
    h = QuantumCircuit(1)
    h.h(0)
    x = QuantumCircuit(1)
    x.x(0)
    with certificate(before) as proof:
        proof.consume(rewrite(h, h, "h"))
        proof.consume(rewrite(x, x, "x"))
        with pytest.raises(AssertionError, match="causal gate order"):
            proof.consume(transition("round", before, after, stage="all_to_all", selections=[
                selection("x", [1], [0]), selection("h", [0], [0])]))


def test_disjoint_gate_reordering_is_allowed():
    before = QuantumCircuit(21)
    before.h(0)
    before.x(20)
    after = QuantumCircuit(21)
    after.x(20)
    after.h(0)
    h = QuantumCircuit(1)
    h.h(0)
    x = QuantumCircuit(1)
    x.x(0)
    with certificate(before) as proof:
        proof.consume(rewrite(h, h, "h"))
        proof.consume(rewrite(x, x, "x"))
        proof.consume(transition("round", before, after, stage="all_to_all", selections=[
            selection("x", [1], [20]), selection("h", [0], [0])]))
        proof.require_current(representation(after))


@pytest.mark.parametrize("damage", ["matrix", "tolerance", "width", "classical"])
def test_bad_selected_rewrite_fails_closed(damage):
    before = QuantumCircuit(1)
    after = QuantumCircuit(1)
    after.ry(0.01, 0)
    event = rewrite(before, after)
    if damage == "tolerance":
        event["tolerance"] = 1e-4
    elif damage == "width":
        event["before"]["qubits"] = 5
    elif damage == "classical":
        classical = QuantumCircuit(1, 1)
        event["before"] = representation(classical)
    with certificate(QuantumCircuit(21)) as proof:
        with pytest.raises(AssertionError):
            proof.consume(event)


def test_nonunitary_matrix_is_rejected_even_if_small_chord_passes():
    with certificate(QuantumCircuit(21)) as proof:
        with pytest.raises(AssertionError, match="not unitary"):
            proof._score(np.eye(2), (1 + 1e-6) * np.eye(2), 1e-10)


def test_accumulated_error_cannot_relax_the_global_tolerance():
    before = QuantumCircuit(21)
    before.id(0)
    after = QuantumCircuit(21)
    after.ry(2e-6, 0)
    local_before = QuantumCircuit(1)
    local_before.id(0)
    local_after = QuantumCircuit(1)
    local_after.ry(2e-6, 0)
    with certificate(before, tolerance=1e-14) as proof:
        proof.consume(rewrite(local_before, local_after))
        with pytest.raises(AssertionError, match="Accumulated composition bound"):
            proof.consume(transition("round", before, after, stage="all_to_all",
                selections=[selection("selected", [0], [0])]))


@pytest.mark.parametrize("damage", ["input_hash", "output_hash", "chain", "missing_rewrite"])
def test_missing_or_broken_evidence_fails_closed(damage):
    full = QuantumCircuit(21)
    full.x(0)
    local = QuantumCircuit(1)
    local.x(0)
    event = transition("round", full, full, stage="all_to_all",
                       selections=[selection("selected", [0], [0])])
    initial = full
    if damage == "input_hash":
        event["input_sha256"] = "0" * 64
    elif damage == "output_hash":
        event["output_sha256"] = "0" * 64
    elif damage == "chain":
        initial = QuantumCircuit(21)
    with certificate(initial) as proof:
        if damage != "missing_rewrite":
            proof.consume(rewrite(local, local))
        with pytest.raises(AssertionError):
            proof.consume(event)


def test_topology_empty_graph_is_not_treated_as_unrestricted():
    full = QuantumCircuit(21)
    full.cx(0, 1)
    local = QuantumCircuit(2)
    local.cx(0, 1)
    with certificate(full, topology=[]) as proof:
        proof.consume(rewrite(local, local))
        with pytest.raises(AssertionError, match="device topology"):
            proof.consume(transition("round", full, full, stage="topology_optimization",
                selections=[selection("selected", [0], [0, 1])]))


def routed_event():
    before = QuantumCircuit(21)
    before.cx(0, 20)
    after = QuantumCircuit(21)
    after.cx(4, 5)
    source = QuantumCircuit(2)
    source.cx(0, 1)
    synthesized = QuantumCircuit(2)
    synthesized.cx(1, 0)
    mapping = list(range(21))
    mapping[0], mapping[4] = mapping[4], mapping[0]
    mapping[20], mapping[5] = mapping[5], mapping[20]
    from squander.partitioning.routing import permuted_partition_target
    target = permuted_partition_target(
        backend._audit_representation_unitary(representation(source)), [1, 0], [1, 0])
    metrics = backend._unitary_audit_metrics(
        target, backend._audit_representation_unitary(representation(synthesized)))
    metrics.update({name + "_bits": backend._float64_bits(value)
                    for name, value in tuple(metrics.items())})
    selected = {"certificate_kind": "unitary", "logical_qubits": [0, 20],
        "input_physical": [4, 5], "output_physical": [4, 5],
        "input_assignment": [1, 0], "output_assignment": [1, 0],
        "embedding": [5, 4], "source_gate_indices": [0],
        "source": representation(source), "synthesized": representation(synthesized),
        "tolerance": 1e-10, "metrics": metrics, "swaps_before": []}
    event = transition("exact_osr_routing", before, after, topology=[[4, 5]],
        initial_mapping=mapping, final_mapping=list(mapping), selections=[selected])
    return before, after, event


def test_nontrivial_routing_permutation_is_independently_certified():
    before, after, event = routed_event()
    with certificate(before, topology=[[4, 5]]) as proof:
        proof.consume(event)
        proof.require_current(representation(after))
        report = proof.report()
        assert report["selected_routing_columns"] == 1
        assert report["global_process_infidelity_bound"] == 0
        assert report["initial_mapping"] == event["initial_mapping"]


@pytest.mark.parametrize("damage", ["mapping", "assignment", "embedding", "topology"])
def test_bad_routing_evidence_fails_closed(damage):
    before, _, event = routed_event()
    if damage == "mapping":
        event["initial_mapping"][0] = event["initial_mapping"][1]
    elif damage == "assignment":
        event["selections"][0]["output_assignment"] = [0, 1]
    elif damage == "embedding":
        event["selections"][0]["embedding"] = [4, 5]
    elif damage == "topology":
        event["topology"] = [[0, 1]]
    with certificate(before, topology=[[4, 5]]) as proof:
        with pytest.raises(AssertionError):
            proof.consume(event)


def test_basis_conversion_is_checked_as_compact_gate_identities():
    from squander import Qiskit_IO, utils
    from qiskit.circuit.library import U3Gate
    original = QuantumCircuit(30)
    original.append(U3Gate(0.83, -0.2, 0.71), [7])
    original.ccx(0, 20, 7)
    original.append(U3Gate(-1.13, 0.27, -0.61), [20])
    native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(original)
    original = Qiskit_IO.get_Qiskit_Circuit(native, parameters)
    converted, converted_parameters = utils.circuit_to_CNOT_basis(native, parameters)
    after = Qiskit_IO.get_Qiskit_Circuit(converted, converted_parameters)
    with certificate(original) as proof:
        proof.consume(transition("squander_basis_conversion", original, after,
                                 conversion_applied=True))
        proof.require_current(representation(after))
        assert proof.report()["changed_basis_conversions"] == 1
        assert all(key[0] <= 4 for key in proof.matrix_cache)


def test_unsupported_transition_and_unrecorded_output_fail_closed():
    full = QuantumCircuit(21)
    after = QuantumCircuit(21)
    after.x(0)
    with certificate(full) as proof:
        with pytest.raises(AssertionError, match="Unsupported whole-circuit"):
            proof.consume(transition("opaque_compiler", full, full))
        with pytest.raises(AssertionError, match="exact circuit gate stream"):
            proof.require_current(representation(after))


def test_forked_child_cannot_use_its_parents_certificate(monkeypatch):
    with certificate(QuantumCircuit(21)) as proof:
        token = composition._ACTIVE_CERTIFICATE.set(proof)
        try:
            assert composition.current_certificate() is proof
            monkeypatch.setattr(composition.os, "getpid", lambda: proof.pid + 1)
            assert composition.current_certificate() is None
        finally:
            composition._ACTIVE_CERTIFICATE.reset(token)

def native_wco_config(**extra):
    config = {
        "strategy": "TreeSearch", "use_osr": True, "use_graph_search": True,
        "parallel": 0, "partition_workers": 1, "max_partition_size": 3,
        "max_equal_cnot_optimization_rounds": 1, "topology": None,
        "test_final_circuit": True, "verbosity": 0,
    }
    config.update(extra)
    return config


@pytest.mark.parametrize("width", [21, 30, 36])
@pytest.mark.parametrize("keyword_api", [False, True])
def test_large_wco_default_validation_never_allocates_global_states(width, keyword_api, monkeypatch):
    from squander import Qiskit_IO
    monkeypatch.delenv(backend._REWRITE_AUDIT_PATH_ENV, raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Large native validation attempted a global random-state comparison.")
    monkeypatch.setattr(backend, "CompareCircuits", forbidden)
    original = QuantumCircuit(width)
    original.cx(0, width - 1)
    native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(original)
    optimizer = backend.qgd_Wide_Circuit_Optimization(native_wco_config())
    if keyword_api:
        output, output_parameters = optimizer.OptimizeWideCircuit(circ=native, parameters=parameters)
    else:
        output, output_parameters = optimizer.OptimizeWideCircuit(native, parameters)
    assert output.get_Qbit_Num() == width
    assert backend.CNOTGateCount(output) == 1
    report = optimizer.config["circuit_validation_certificate"]
    assert report["status"] == "certified"
    assert report["qubits"] == width
    assert report["selected_rewrites"] > 0
    assert report["global_process_infidelity_bound"] <= optimizer.config["circuit_validation_tolerance"]
    assert composition.current_certificate() is None


def test_large_wco_can_emit_the_regular_external_audit_too(tmp_path, monkeypatch):
    import json
    from squander import Qiskit_IO
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv(backend._REWRITE_AUDIT_PATH_ENV, str(path))
    monkeypatch.setattr(backend, "CompareCircuits",
                        lambda *args, **kwargs: pytest.fail("Global state allocated."))
    original = QuantumCircuit(21)
    original.cx(0, 20)
    native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(original)
    optimizer = backend.qgd_Wide_Circuit_Optimization(native_wco_config())
    optimizer.OptimizeWideCircuit(native, parameters)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(e["kind"] == "round" for e in events)
    assert any(
        (e["kind"] == "rewrite" and e.get("accepted"))
        or (e["kind"] == "round" and any(
            selection.get("passthrough") is True
            and selection["source_gate_indices"] == [0]
            and selection["qubits"] == [0, 20]
            for selection in e.get("selections", [])
        ))
        for e in events
    )
    assert optimizer.config["circuit_validation_certificate"]["status"] == "certified"


def test_small_wco_retains_the_existing_state_comparison(monkeypatch):
    from squander import Qiskit_IO
    monkeypatch.delenv(backend._REWRITE_AUDIT_PATH_ENV, raising=False)
    calls = []
    def existing_check(*args, **kwargs):
        assert args[0].get_Qbit_Num() == 3
        calls.append(True)
    monkeypatch.setattr(backend, "CompareCircuits", existing_check)
    original = QuantumCircuit(3)
    original.cx(0, 2)
    native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(original)
    optimizer = backend.qgd_Wide_Circuit_Optimization(native_wco_config())
    optimizer.OptimizeWideCircuit(native, parameters)
    assert calls
    assert "circuit_validation_certificate" not in optimizer.config
    assert composition.current_certificate() is None


def test_large_wco_exception_cleans_up_its_certificate(monkeypatch):
    from squander import Qiskit_IO
    monkeypatch.delenv(backend._REWRITE_AUDIT_PATH_ENV, raising=False)
    original = QuantumCircuit(21)
    original.cx(0, 20)
    native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(original)
    def injected(*args, **kwargs):
        raise RuntimeError("injected partition failure")
    monkeypatch.setattr(backend.qgd_Wide_Circuit_Optimization,
                        "make_all_partition_circuit", injected)
    optimizer = backend.qgd_Wide_Circuit_Optimization(native_wco_config())
    with pytest.raises(RuntimeError, match="injected partition failure"):
        optimizer.OptimizeWideCircuit(native, parameters)
    assert composition.current_certificate() is None
    assert "circuit_validation_certificate" not in optimizer.config


def test_unsupported_large_native_width_fails_closed_before_optimization(monkeypatch):
    from squander.gates.qgd_Circuit import qgd_Circuit
    monkeypatch.setattr(backend, "CompareCircuits",
                        lambda *args, **kwargs: pytest.fail("Global state allocated."))
    optimizer = backend.qgd_Wide_Circuit_Optimization(
        native_wco_config(max_partition_size=5))
    with pytest.raises(NotImplementedError, match="Large-circuit validation"):
        optimizer.OptimizeWideCircuit(qgd_Circuit(21), np.array([], dtype=np.float64))


def test_manual_large_comparison_requires_an_active_certificate(monkeypatch):
    from squander.gates.qgd_Circuit import qgd_Circuit
    monkeypatch.setattr(backend, "CompareCircuits",
                        lambda *args, **kwargs: pytest.fail("Global state allocated."))
    optimizer = backend.qgd_Wide_Circuit_Optimization(native_wco_config())
    circuit = qgd_Circuit(21)
    parameters = np.array([], dtype=np.float64)
    with pytest.raises(AssertionError, match="active composition certificate"):
        optimizer.check_compare_circuits(circuit, parameters, circuit, parameters)

def test_large_inner_pass_preserves_orig_parameters_keyword(monkeypatch):
    from squander import Qiskit_IO
    monkeypatch.delenv(backend._REWRITE_AUDIT_PATH_ENV, raising=False)
    monkeypatch.setattr(backend, "CompareCircuits",
                        lambda *args, **kwargs: pytest.fail("Global state allocated."))
    original = QuantumCircuit(21)
    original.cx(0, 20)
    native, parameters = Qiskit_IO.convert_Qiskit_to_Squander(original)
    optimizer = backend.qgd_Wide_Circuit_Optimization(native_wco_config())
    output, _ = optimizer.InnerOptimizeWideCircuit(circ=native, orig_parameters=parameters)
    assert backend.CNOTGateCount(output) == 1
    assert optimizer.config["circuit_validation_certificate"]["status"] == "certified"
    assert composition.current_certificate() is None


@pytest.mark.parametrize("left,right", [(0.0, 1e-12), (0.1, 0.2), (1.0, 2.0)])
def test_python38_numpy_rounding_fallback_is_outward_and_equivalent(left, right, monkeypatch):
    expected = composition._up_add(left, right)
    monkeypatch.setattr(composition, "_nextafter", lambda x, y: float(np.nextafter(x, y)))
    assert composition._up_add(left, right) == expected
    assert composition._up_add(left, right) > left + right
    with certificate(QuantumCircuit(21)) as proof:
        proof._add_error(1e-6, 0.0)
        report = proof.report()
        assert report["global_process_infidelity_bound"] > proof.spectral_sum ** 2
        assert report["global_process_infidelity_bound"] <= proof.tolerance


def _stable_matrix_infidelity(before, after):
    overlap = np.vdot(before, after)
    difference = after - overlap / abs(overlap) * before
    chord = float(np.vdot(difference, difference).real / before.shape[0])
    return chord * (1.0 - chord / 4.0)


def test_normalized_frobenius_certificate_avoids_a_spectral_false_rejection():
    from qiskit import transpile
    from qiskit.quantum_info import Operator

    local_before = QuantumCircuit(2)
    local_before.cx(0, 1)
    local_after = local_before.copy()
    local_after.cp(2e-5, 0, 1)
    local_after = transpile(local_after, basis_gates=["u3", "cx"], optimization_level=0)
    before = QuantumCircuit(21)
    after = QuantumCircuit(21)
    compact_before = QuantumCircuit(2)
    compact_after = QuantumCircuit(2)
    for _ in range(10):
        before.compose(local_before, qubits=[0, 20], inplace=True)
        after.compose(local_after, qubits=[0, 20], inplace=True)
        compact_before.compose(local_before, inplace=True)
        compact_after.compose(local_after, inplace=True)
    with certificate(before) as proof:
        proof.consume(rewrite(local_before, local_after))
        proof.consume(transition("round", before, after, stage="all_to_all",
            selections=[selection("selected", [index], [0, 20]) for index in range(10)]))
        proof.require_current(representation(after))
        report = proof.report()
    actual = _stable_matrix_infidelity(
        Operator(compact_before).data, Operator(compact_after).data)
    assert report["global_process_infidelity_spectral_bound"] > 1e-8
    assert actual <= report["global_process_infidelity_bound"] < 1e-8
    assert report["global_process_infidelity_bound"] == report[
        "global_process_infidelity_normalized_frobenius_bound"]


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_normalized_frobenius_distance_is_invariant_under_spectators(width):
    from qiskit.quantum_info import Operator

    before = QuantumCircuit(width)
    before.h(0)
    after = before.copy()
    after.rz(2e-6, width - 1)
    left, right = Operator(before).data, Operator(after).data
    with certificate(QuantumCircuit(21)) as proof:
        _, _, local_distance = proof._score(left, right, 1e-10)
        _, _, extended_distance = proof._score(
            np.kron(np.eye(2), left), np.kron(np.eye(2), right), 1e-10)
    assert extended_distance == pytest.approx(local_distance, rel=1e-12)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_frobenius_composition_bounds_independent_noncommuting_matrix_products(width):
    rng = np.random.default_rng(20261008 + width)
    dimension = 1 << width
    for _ in range(12):
        before_product = np.eye(dimension, dtype=np.complex128)
        after_product = np.eye(dimension, dtype=np.complex128)
        with certificate(QuantumCircuit(21)) as proof:
            for _ in range(8):
                sample = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
                    (dimension, dimension))
                before, _ = np.linalg.qr(sample)
                vector = rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension)
                vector /= np.linalg.norm(vector)
                perturbation = np.eye(dimension) + np.expm1(2e-5j) * np.outer(
                    vector, vector.conj())
                after = perturbation @ before
                delta, defect, frobenius_delta = proof._score(before, after, 1e-10)
                proof._add_error(delta, defect, frobenius_delta)
                before_product = before @ before_product
                after_product = after @ after_product
            report = proof.report()
        actual = _stable_matrix_infidelity(before_product, after_product)
        assert actual <= report["global_process_infidelity_bound"]
        assert report["global_process_infidelity_bound"] <= 1e-8


def test_frobenius_bound_still_rejects_coherent_accumulated_error():
    before = np.eye(4, dtype=np.complex128)
    after = np.diag([1, 1, 1, np.exp(2e-5j)])
    actual = _stable_matrix_infidelity(before, np.linalg.matrix_power(after, 12))
    assert actual > 1e-8
    with certificate(QuantumCircuit(21)) as proof:
        delta, defect, frobenius_delta = proof._score(before, after, 1e-10)
        for _ in range(12):
            proof._add_error(delta, defect, frobenius_delta)
        with pytest.raises(AssertionError, match="Accumulated composition bound"):
            proof.report()


def test_disjoint_rewrites_use_a_tensor_layer_not_a_linear_error_sum():
    from qiskit.quantum_info import Operator

    local_before = QuantumCircuit(1)
    local_before.id(0)
    local_after = QuantumCircuit(1)
    local_after.ry(1.8e-5, 0)
    before = QuantumCircuit(21)
    before.id(0)
    before.id(20)
    after = QuantumCircuit(21)
    after.ry(1.8e-5, 0)
    after.ry(1.8e-5, 20)
    with certificate(before, tolerance=2e-10) as proof:
        proof.consume(rewrite(local_before, local_after))
        proof.consume(transition("round", before, after, stage="all_to_all", selections=[
            selection("selected", [0], [0]), selection("selected", [1], [20])]))
        report = proof.report()
    compact = QuantumCircuit(2)
    compact.ry(1.8e-5, 0)
    compact.ry(1.8e-5, 1)
    actual = _stable_matrix_infidelity(np.eye(4), Operator(compact).data)
    assert actual <= report["global_process_infidelity_bound"] < 2e-10
    assert report["disjoint_error_layers"] == 1
    assert report["global_process_infidelity_spectral_bound"] > 2e-10


def test_zero_error_entangler_cannot_hide_coherent_error_transport():
    from qiskit.quantum_info import Operator

    local_before = QuantumCircuit(1)
    local_before.id(0)
    local_after = QuantumCircuit(1)
    local_after.ry(1.8e-5, 0)
    swap = QuantumCircuit(2)
    swap.cx(0, 1)
    swap.cx(1, 0)
    swap.cx(0, 1)
    before = QuantumCircuit(21)
    before.id(0)
    before.compose(swap, qubits=[0, 20], inplace=True)
    before.id(20)
    after = QuantumCircuit(21)
    after.ry(1.8e-5, 0)
    after.compose(swap, qubits=[0, 20], inplace=True)
    after.ry(1.8e-5, 20)
    compact = QuantumCircuit(2)
    compact.ry(1.8e-5, 0)
    compact.compose(swap, inplace=True)
    compact.ry(1.8e-5, 1)
    actual = _stable_matrix_infidelity(Operator(swap).data, Operator(compact).data)
    assert actual > 2e-10
    with certificate(before, tolerance=2e-10) as proof:
        proof.consume(rewrite(local_before, local_after))
        proof.consume(rewrite(swap, swap, "transport"))
        with pytest.raises(AssertionError, match="Accumulated composition bound"):
            proof.consume(transition("round", before, after, stage="all_to_all", selections=[
                selection("selected", [0], [0]),
                selection("transport", [1, 2, 3], [0, 20]),
                selection("selected", [4], [20])]))


def test_opaque_replay_block_ends_a_disjoint_error_layer():
    with certificate(QuantumCircuit(21), tolerance=2e-10) as proof:
        proof._add_error_layers([
            (9e-6, 0.0, 9e-6, [0]), None, (9e-6, 0.0, 9e-6, [20])])
        with pytest.raises(AssertionError, match="Accumulated composition bound"):
            proof.report()


@pytest.mark.parametrize("pattern", [
    ((0, 1), (2, 3), (1, 2), (0, 3)) * 4,
    ((0,), (1,), (2,), (3,), (0, 2), (1, 3)) * 3,
    ((0, 1), (2, 3), (0, 1, 2), (3,)) * 4,
])
def test_disjoint_layer_bounds_independent_overlapping_dense_products(pattern):
    rng = np.random.default_rng(20261009)

    def embed(matrix, qubits):
        spectator_mask = sum(1 << q for q in range(4) if q not in qubits)
        indices = [sum(((basis >> q) & 1) << i for i, q in enumerate(qubits))
                   for basis in range(16)]
        result = np.zeros((16, 16), dtype=np.complex128)
        for row in range(16):
            for column in range(16):
                if (row & spectator_mask) == (column & spectator_mask):
                    result[row, column] = matrix[indices[row], indices[column]]
        return result

    for _ in range(20):
        before_product = np.eye(16, dtype=np.complex128)
        after_product = before_product.copy()
        errors = []
        with certificate(QuantumCircuit(21)) as proof:
            for qubits in pattern:
                dimension = 1 << len(qubits)
                sample = rng.standard_normal((dimension, dimension)) + 1j * rng.standard_normal(
                    (dimension, dimension))
                before, _ = np.linalg.qr(sample)
                vector = rng.standard_normal(dimension) + 1j * rng.standard_normal(dimension)
                vector /= np.linalg.norm(vector)
                after = (np.eye(dimension) + np.expm1(1e-5j) * np.outer(
                    vector, vector.conj())) @ before
                delta, defect, frobenius_delta = proof._score(before, after, 1e-10)
                errors.append((delta, defect, frobenius_delta, qubits))
                before_product = embed(before, qubits) @ before_product
                after_product = embed(after, qubits) @ after_product
            proof._add_error_layers(errors)
            report = proof.report()
        actual = _stable_matrix_infidelity(before_product, after_product)
        assert actual <= report["global_process_infidelity_bound"] < 1e-8
