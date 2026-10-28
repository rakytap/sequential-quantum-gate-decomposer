"""Lossless benchmark exports and fail-closed archive recovery controls."""

import ast
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator

from squander import Qiskit_IO
from squander.decomposition._audit_qasm import dump_native_audit_qasm
from squander.gates.qgd_Circuit import qgd_Circuit


ROOT = Path(__file__).resolve().parents[2]
backend = importlib.import_module(
    "squander.decomposition.qgd_Wide_Circuit_Optimization"
)


def load_exporter():
    # Load only the helper, without starting the example's benchmark driver.
    source = ROOT / "examples/decomposition/wide_circuit_optimization.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "save_qasm2")
    namespace = {"Qiskit_IO": Qiskit_IO, "np": np, "os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace["save_qasm2"]


def load_recovery():
    source = ROOT / "benchmarks/partitioning/recover_wco_qasm_exports.py"
    spec = importlib.util.spec_from_file_location("wco_export_recovery", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("angle", [
    -8.112385435312549e-17, np.nextafter(np.pi / 4, 0),
    np.nextafter(np.pi / 4, np.inf), np.nextafter(2 * np.pi, 0),
])
def test_benchmark_export_matches_lossless_audit_bits(tmp_path, angle):
    circuit = qgd_Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    parameters = np.array([0.37, angle, -np.pi])
    path = tmp_path / "output.qasm"
    load_exporter()(circuit, parameters, path)
    expected = backend._squander_audit_representation(circuit, parameters, range(2))
    assert backend._qasm_exact_state(path.read_text()) == backend._qasm_exact_state(expected)
    restored = QuantumCircuit.from_qasm_str(path.read_text())
    native = Qiskit_IO.get_Qiskit_Circuit(circuit, parameters)
    assert np.array_equal(Operator(restored).data, Operator(native).data)
    assert not path.with_suffix(".qasm.tmp").exists()


def archive_fixture(tmp_path):
    circuit = QuantumCircuit(2)
    circuit.u(0.7, -8.112385435312549e-17,
              np.nextafter(np.pi / 4, np.inf), 0)
    circuit.cx(0, 1)
    representation = {"format": "squander_openqasm2", "qubits": 2,
                      "qasm": dump_native_audit_qasm(circuit)}
    output = tmp_path / "example.qasm"
    output.write_text(qasm2.dumps(circuit))
    audit_path = tmp_path / "example.audit.json.gz"
    run = {"output_file": "example.qasm", "input_circuit": representation,
           "output_circuit": representation,
           "input_circuit_sha256": backend._exact_state_sha256(
               backend._qasm_exact_state(representation)),
           "output_circuit_sha256": backend._exact_state_sha256(
               backend._qasm_exact_state(representation)),
           "output_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
           "software": backend.rewrite_audit_software_versions()}
    audit = {"schema_version": backend.REWRITE_AUDIT_SCHEMA_VERSION,
             "run": run, "summary": {"events": 0}, "events": []}
    with gzip.open(audit_path, "wt") as stream:
        json.dump(audit, stream)
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"example.qasm": {
        "status": "failed", "output_file": "example.qasm",
        "failure": load_recovery().EXPORT_FAILURE,
        "timing": {"total": None}}}))
    return results, output, audit_path, audit


def test_recovery_preserves_originals_and_requires_complete_replay(tmp_path):
    results, output, audit_path, audit = archive_fixture(tmp_path)
    originals = {path.name: path.read_bytes() for path in (results, output, audit_path)}
    report = load_recovery().recover_one(results, "example.qasm", apply=True)
    assert report["applied"]
    assert report["final_cnots"] == 1
    for name, content in originals.items():
        assert (Path(report["backup"]) / name).read_bytes() == content
    entry = json.loads(results.read_text())["example.qasm"]
    assert entry["status"] == "completed"
    assert entry["timing"] == {"a2a": None, "routing": None,
                               "optimization": None, "total": None}
    assert entry["export_recovery"]["synthesis_rerun"] is False
    assert entry["source_fingerprint"] if "source_fingerprint" in entry else True
    with gzip.open(audit_path, "rt") as stream:
        repaired = json.load(stream)
    assert repaired["events"] == audit["events"]
    assert repaired["run"]["output_circuit"] == audit["run"]["output_circuit"]
    backend.verify_rewrite_audit(audit_path, expected_sha256=entry["rewrite_audit"]["sha256"])
    assert backend._qasm_exact_state(output.read_text()) == backend._qasm_exact_state(
        audit["run"]["output_circuit"])


def test_recovery_check_only_does_not_change_archives(tmp_path):
    results, output, audit_path, _ = archive_fixture(tmp_path)
    originals = [path.read_bytes() for path in (results, output, audit_path)]
    report = load_recovery().recover_one(results, "example.qasm")
    assert report["status"] == "verified"
    assert not report["applied"]
    assert originals == [path.read_bytes() for path in (results, output, audit_path)]


@pytest.mark.parametrize("corruption", ["file", "wrong_stream", "audit_chain"])
def test_recovery_rejects_corruption_without_changes(tmp_path, corruption):
    results, output, audit_path, audit = archive_fixture(tmp_path)
    if corruption in ("file", "wrong_stream"):
        output.write_text(output.read_text().replace("u(0.7,", "u(0.9,"))
        if corruption == "wrong_stream":
            audit["run"]["output_file_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    else:
        audit["run"]["output_circuit_sha256"] = "0" * 64
    with gzip.open(audit_path, "wt") as stream:
        json.dump(audit, stream)
    originals = [path.read_bytes() for path in (results, output, audit_path)]
    with pytest.raises(AssertionError):
        load_recovery().recover_one(results, "example.qasm", apply=True)
    assert originals == [path.read_bytes() for path in (results, output, audit_path)]
