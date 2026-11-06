#!/usr/bin/env python3
"""Freeze current benchmark results and package their large evidence for a release."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


DATASETS = ("IBMEagle", "QASMBenchmarks")
METHODS = ("TreeSearch", "bqskit")
ASSET_TARGET_BYTES = 7 * 1024**3 // 4
GITHUB_ASSET_LIMIT = 2 * 1024**3


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {relative}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or non-regular input: {path}")
    return path


def record(path, root, role, location="git", digest=None):
    return {
        "path": path.relative_to(root).as_posix(),
        "role": role,
        "location": location,
        "bytes": path.stat().st_size,
        "sha256": digest or sha256(path),
    }


def copy_snapshot(path, root, snapshot, role):
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while snapshotting: {path}")
    destination = snapshot / path.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return record(destination, snapshot, role), content


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def write_asset(root, output, name, entries):
    destination = output / name
    partial = output / (name + ".partial")
    records = []
    try:
        with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
            for path, role, expected in entries:
                before = path.stat()
                info = tarfile.TarInfo(path.relative_to(root).as_posix())
                info.size = before.st_size
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as stream:
                    reader = DigestReader(stream)
                    archive.addfile(info, reader)
                digest = reader.digest.hexdigest()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Evidence changed while reading: {path}")
                if digest != expected:
                    raise RuntimeError(f"Recorded SHA-256 does not match {path}")
                entry = record(path, root, role, "release", digest)
                entry["asset"] = name
                records.append(entry)
        if partial.stat().st_size >= GITHUB_ASSET_LIMIT:
            raise RuntimeError(f"Release asset exceeds GitHub's limit: {name}")
        partial.rename(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return records, {
        "name": name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "file_count": len(records),
    }


def prepare(root, output, release_tag):
    root = root.resolve()
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Keep release assets outside the Git checkout")
    if output.exists():
        raise ValueError(f"Use a new, empty output directory: {output}")
    output.mkdir(parents=True)
    snapshot = output / "snapshot"
    assets = output / "assets"
    assets.mkdir()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "packaging_source_commit": commit,
        "release_tag": release_tag,
        "release_url": "https://github.com/rakytap/sequential-quantum-gate-decomposer/releases/tag/" + release_tag,
        "scope": "Eight current result directories; completed outputs and their referenced evidence. All recorded statuses are retained in results.json.",
        "exclusions": [
            "old/reference result directories",
            "temporary SQLite indexes",
            "export-recovery backups",
            "unreferenced or incomplete audit streams and cached artifacts",
        ],
        "groups": [], "files": [], "assets": [],
    }
    input_paths = set()
    metadata_digests = {}
    for dataset in DATASETS:
        for width in (3, 4):
            for method in METHODS:
                folder = root / "benchmarks/partitioning" / f"{dataset}_results_{width}qbit_{method}"
                metadata = safe_path(root, (folder / "results.json").relative_to(root).as_posix())
                item, content = copy_snapshot(metadata, root, snapshot, "result_metadata")
                metadata_digests[metadata] = item["sha256"]
                manifest["files"].append(item)
                results = json.loads(content)
                group = {
                    "path": folder.relative_to(root).as_posix(),
                    "dataset": dataset, "partition_size": width,
                    "method": "OSR" if method == "TreeSearch" else "BQSKit",
                    "statuses": dict(Counter(row["status"] for row in results.values())),
                    "source_fingerprints": sorted({row["source_fingerprint"] for row in results.values()}),
                    "completed_circuits": [],
                }
                evidence = {}
                for basename, row in results.items():
                    source = safe_path(root, f"benchmarks/partitioning/{dataset}/{basename}")
                    input_paths.add(source)
                    if row["status"] != "completed":
                        continue
                    output_path = safe_path(root, row["output_file"])
                    if output_path.parent != folder:
                        raise ValueError(f"Output does not belong to its result group: {output_path}")
                    item, _ = copy_snapshot(output_path, root, snapshot, "final_qasm")
                    manifest["files"].append(item)
                    group["completed_circuits"].append(basename)
                    for key, role in [("rewrite_audit", "rewrite_audit"), ("routing_osr_catalog", "routing_catalog")]:
                        info = row.get(key)
                        if not info:
                            if key == "rewrite_audit":
                                raise ValueError(f"Completed result has no audit: {basename}")
                            continue
                        path = safe_path(root, info["file"])
                        if path.parent != folder:
                            raise ValueError(f"Evidence does not belong to its result group: {path}")
                        expected = info["sha256"]
                        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                            raise ValueError(f"Invalid SHA-256 for {path}")
                        evidence[path] = (role, expected)
                chunks = [[]]
                chunk_size = 10240
                for path, (role, expected) in sorted(evidence.items()):
                    estimated = 2048 + ((path.stat().st_size + 511) // 512) * 512
                    if estimated + 10240 > ASSET_TARGET_BYTES:
                        raise ValueError(f"An individual artifact needs further splitting: {path}")
                    if chunks[-1] and chunk_size + estimated > ASSET_TARGET_BYTES:
                        chunks.append([])
                        chunk_size = 10240
                    chunks[-1].append((path, role, expected))
                    chunk_size += estimated
                for part, entries in enumerate(chunks, 1):
                    if not entries:
                        continue
                    name = f"{dataset}-{group['method']}-{width}q-evidence-{part:02d}.tar"
                    records, asset = write_asset(root, assets, name, entries)
                    manifest["files"].extend(records)
                    manifest["assets"].append(asset)
                    print(f"Prepared {name}: {asset['bytes'] / 1024**3:.3f} GiB; recorded hashes match", flush=True)
                manifest["groups"].append(group)
    for path in sorted(input_paths):
        item, _ = copy_snapshot(path, root, snapshot, "source_qasm")
        manifest["files"].append(item)
    for path, expected in metadata_digests.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Result metadata changed during preparation; repeat the snapshot: {path}")
    manifest["files"].sort(key=lambda entry: entry["path"])
    manifest["totals"] = {
        "git_files": sum(item["location"] == "git" for item in manifest["files"]),
        "git_bytes": sum(item["bytes"] for item in manifest["files"] if item["location"] == "git"),
        "evidence_files": sum(item["location"] == "release" for item in manifest["files"]),
        "evidence_bytes": sum(item["bytes"] for item in manifest["files"] if item["location"] == "release"),
        "release_assets": len(manifest["assets"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (assets / "SHA256SUMS").write_text("".join(
        f"{item['sha256']}  {item['name']}\n" for item in manifest["assets"]
    ))
    print(json.dumps(manifest["totals"], indent=2), flush=True)


def verify(root, manifest_path, include_evidence=False):
    manifest = json.loads(manifest_path.read_text())
    checked = 0
    for entry in manifest["files"]:
        if entry["location"] == "release" and not include_evidence:
            continue
        path = safe_path(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Archive checksum mismatch: {entry['path']}")
        checked += 1
    print(f"Verified {checked} files against {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("prepare", help="create a frozen snapshot and release assets")
    build.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--release-tag", required=True)
    check = sub.add_parser("verify", help="check published file hashes without rerunning synthesis")
    check.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--include-evidence", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.output_dir, args.release_tag)
    else:
        verify(args.root.resolve(), args.manifest, args.include_evidence)


if __name__ == "__main__":
    main()
