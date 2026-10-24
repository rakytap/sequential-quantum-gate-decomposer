"""Storage-only audit changes retain strict replay and bound memory."""

import gzip
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tracemalloc

import numpy as np
import pytest

from squander.decomposition import _rewrite_audit_stream as stream
from squander.gates.qgd_Circuit import qgd_Circuit

backend = importlib.import_module(
    "squander.decomposition.qgd_Wide_Circuit_Optimization")


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 65536])
def test_incremental_json_handles_all_token_boundaries(monkeypatch, chunk):
    monkeypatch.setattr(stream, "CHUNK_CHARS", chunk)
    expected = {
        "schema_version": 2,
        "run": {"value": 1.23456789e-120, "large": 1234567890123456789,
                "text": 'escaped "quote", slash \\ and unicode \u03b1'},
        "events": [{"event_id": "a", "value": -1.25e+120},
                   {"event_id": "b", "value": 1.0}],
        "summary": {"events": 2},
    }

    class BoundedReader(io.StringIO):
        def read(self, size=-1):
            assert 0 < size <= chunk
            return super().read(size)

    events = []
    header = stream.read_archive(BoundedReader(json.dumps(expected)), events.append)
    assert events == expected["events"]
    assert header == {key: value for key, value in expected.items() if key != "events"}


@pytest.mark.parametrize("text", [
    '{"events":[{},]}', '{"events":[],"events":[]}',
    '{"events":[{}]', '{"events":[]} trailing', '{"run":{}}',
])
def test_incremental_json_rejects_corrupt_envelopes(text):
    with pytest.raises(ValueError):
        stream.read_archive(io.StringIO(text), lambda event: None)


def fixture(tmp_path, *, missing_reference=False):
    circuit = qgd_Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    parameters = np.array([0.123, -0.234, 0.345])
    representation = backend._squander_audit_representation(circuit, parameters, range(2))
    state = backend._qasm_exact_state(representation)
    digest = backend._exact_state_sha256(state)
    rewrite = {
        "event_id": "rewrite", "kind": "rewrite", "timestamp_ns": 2,
        "component": "squander_partition", "before": representation,
        "after": representation, "accepted": True, "tolerance": 1e-10,
    }
    round_event = {
        "event_id": "round", "kind": "round", "timestamp_ns": 3,
        "component": "squander_wide_optimization", "started_ns": 1,
        "input": representation, "output": representation,
        "input_sha256": digest, "output_sha256": digest,
        "selections": [{
            "rewrite_event_id": "missing" if missing_reference else "rewrite",
            "source_gate_indices": list(range(len(state["operations"]))),
            "qubits": [0, 1],
        }],
    }
    jsonl = tmp_path / "source.jsonl"
    jsonl.write_text("\n".join(json.dumps(event) for event in (round_event, rewrite)))
    run = {
        "input_circuit": representation, "output_circuit": representation,
        "input_circuit_sha256": digest, "output_circuit_sha256": digest,
    }
    return jsonl, run


def test_disk_replay_matches_original_checks_and_summaries(tmp_path, monkeypatch):
    jsonl, run = fixture(tmp_path)
    small, small_digest = backend.save_rewrite_audit(jsonl, tmp_path / "small.gz", run)
    reference = backend.verify_rewrite_audit(
        small, expected_sha256=small_digest, audit_path=tmp_path / "small.gz")
    monkeypatch.setattr(stream, "MAX_MEMORY_BYTES", 1)
    large, digest = backend.save_rewrite_audit(jsonl, tmp_path / "large.gz", run)
    try:
        assert isinstance(large["events"], stream.DiskAuditEvents)
        assert list(large["events"]) == small["events"]
        assert large["summary"] == small["summary"]
        assert large["events"].get("rewrite") == small["events"][0]
        assert backend.verify_rewrite_audit(
            large, expected_sha256=digest, audit_path=tmp_path / "large.gz") == reference
        assert backend.verify_rewrite_audit(
            tmp_path / "large.gz", expected_sha256=digest) == reference
    finally:
        stream.close_audit(large)
    assert not list(tmp_path.glob(".rewrite-audit-index-*"))


def test_disk_replay_still_rejects_missing_selected_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "MAX_MEMORY_BYTES", 1)
    jsonl, run = fixture(tmp_path, missing_reference=True)
    audit, _ = backend.save_rewrite_audit(jsonl, tmp_path / "audit.gz", run)
    try:
        with pytest.raises(AssertionError, match="Missing selected rewrite"):
            backend.verify_rewrite_audit(audit)
    finally:
        stream.close_audit(audit)


def test_disk_replay_still_rejects_modified_metric_bits(tmp_path, monkeypatch):
    jsonl, run = fixture(tmp_path)
    audit, _ = backend.save_rewrite_audit(jsonl, tmp_path / "original.gz", run)
    audit["events"][0]["metrics"]["process_infidelity_bits"] = "0000000000000001"
    corrupted = tmp_path / "corrupted.gz"
    with gzip.open(corrupted, "wt") as output:
        json.dump(audit, output)
    monkeypatch.setattr(stream, "MAX_MEMORY_BYTES", 1)
    with pytest.raises(AssertionError, match="not bit-identical"):
        backend.verify_rewrite_audit(corrupted)
    assert not list(tmp_path.glob(".rewrite-audit-index-*"))


def test_index_preserves_timestamp_ties_and_rejects_duplicate_ids(tmp_path):
    events = stream.DiskAuditEvents(tmp_path)
    try:
        for seq, (timestamp, event_id, started) in enumerate([
            (4, "b", 3), (2, "z", 2), (4, "a", 1),
        ]):
            event = dict(event_id=event_id, kind="round", timestamp_ns=timestamp,
                         started_ns=started, seq=seq)
            events.add(event, json.dumps(event))
        events.finish()
        assert [event["event_id"] for event in events] == ["z", "a", "b"]
        assert [event["event_id"] for event in
                stream.iter_audit_events(events, {"round"}, ordered=True)] == ["a", "z", "b"]
        assert events[-1]["event_id"] == "b"
        assert events.get("missing") is None
        with pytest.raises(AssertionError, match="Duplicate"):
            events.add({"event_id": "a"}, '{"event_id":"a"}')
    finally:
        events.close()


def test_large_compressible_archive_does_not_materialize_event_array(tmp_path, monkeypatch):
    monkeypatch.setattr(stream, "MAX_MEMORY_BYTES", 1024)
    path = tmp_path / "large.gz"
    count = 2200
    payload = "x" * 8192
    with gzip.open(path, "wt") as output:
        output.write('{"schema_version":2,"run":{},"events":[')
        for index in range(count):
            if index:
                output.write(",")
            json.dump(dict(event_id=str(index), kind="probe", timestamp_ns=index,
                           payload=payload), output)
        output.write('],"summary":{"events":2200}}')
    tracemalloc.start()
    audit = None
    try:
        audit = backend.load_rewrite_audit(path)
        assert isinstance(audit["events"], stream.DiskAuditEvents)
        assert len(audit["events"]) == count
        assert sum(1 for _ in audit["events"]) == count
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 6 * 1024 * 1024
    finally:
        tracemalloc.stop()
        if audit is not None:
            stream.close_audit(audit)


def test_complete_transition_chain_is_not_materialized(tmp_path, monkeypatch):
    jsonl, run = fixture(tmp_path)
    source = json.loads(jsonl.read_text().splitlines()[0])
    state = backend._qasm_exact_state(run["input_circuit"])
    monkeypatch.setattr(backend, "_verify_native_round_replay",
                        lambda event, index: (state, state))
    events = stream.DiskAuditEvents(tmp_path)
    count = 600
    for index in range(count):
        event = dict(source, event_id=str(index), timestamp_ns=index,
                     started_ns=index, unused_payload="x" * 32768)
        events.add(event, json.dumps(event))
    events.finish()
    tracemalloc.start()
    try:
        report = backend._verify_full_replay_chain(
            {"run": run, "events": events}, events)
        assert report == (count, run["output_circuit_sha256"])
        _, peak = tracemalloc.get_traced_memory()
        assert peak < 6 * 1024 * 1024
    finally:
        tracemalloc.stop()
        events.close()

