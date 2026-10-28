from examples.decomposition.wide_circuit_optimization import (
    clear_invalidated_result_artifacts,
)


def test_missing_result_entry_cleanup_removes_only_matching_relics(tmp_path):
    output = tmp_path / "example.qasm"
    relics = (
        output,
        tmp_path / "example.qasm.tmp",
        tmp_path / "example.audit.jsonl",
        tmp_path / "example.audit.json.gz",
        tmp_path / "example.routing-catalog.json.gz",
        tmp_path / "example.failure.json",
    )
    preserved = (
        tmp_path / "example_other.qasm",
        tmp_path / "other.audit.json.gz",
    )
    for path in (*relics, *preserved):
        path.write_bytes(b"test")

    removed = clear_invalidated_result_artifacts(output)

    assert set(removed) == set(relics)
    assert all(not path.exists() for path in relics)
    assert all(path.exists() for path in preserved)
