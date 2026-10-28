# Wide-circuit benchmark datasets

The wide-circuit optimization example reads the two source datasets here and
writes optimized OpenQASM 2 circuits with identical filenames into
`<dataset>_results_<width>qbit_<strategy>/`, with metadata in each directory's
`results.json`. The datasets are `IBMEagle` and `QASMBenchmarks`, the maximum
A2A/final partition widths are 3 and 4, and the archived strategies are
`TreeSearch` (OSR) and `bqskit`. Routing uses at most three-qubit blocks.

- `IBMEagle` contains 51 unitary circuits from the IBM Eagle/QMill benchmark
  corpus. They were converted from the `njross/optimizer` Quipper sources with
  [`onestruggler/qasm-quipper`](https://github.com/onestruggler/qasm-quipper).
  The source `_before` marker was removed from the archived filenames. Nine
  generated adder circuits containing non-unitary resets are deliberately
  excluded.
- `QASMBenchmarks` contains the 26 original circuits from the curated
  `processed_qasm` benchmark set used in the BQSKit comparison experiments.
  That set is primarily derived from [PNNL
  QASMBench](https://github.com/pnnl/QASMBench), with additional
  [VeriQBench](https://github.com/Veri-Q/Benchmark) and
  [BQSKit](https://github.com/BQSKit/bqskit) example circuits. Generated
  `_u3cx`, `_squander`, and `_qsearch` results are deliberately excluded.

The two input directories contain 77 circuits in total. The eight current
result directories are populated by
`examples/decomposition/wide_circuit_optimization.py` and the BQSKit comparison
workflow. Directories with `_old` suffixes and scratch experiments are not part
of the archived comparison.

## Frozen result archive: 28 October 2026

The Git archive contains the eight current `results.json` files and the final
OpenQASM exports for every result recorded as completed. All recorded failures
and timeouts are retained in the metadata, without treating an unrun circuit
as a failure. The archive includes these completed results:

| Dataset | OSR 3q | OSR 4q | BQSKit 3q | BQSKit 4q |
| --- | ---: | ---: | ---: | ---: |
| IBM Eagle | 35 | 35 | 38 | 20 |
| QASMBenchmarks | 24 | 24 | 25 | 18 |
| Total | 59 | 59 | 63 | 38 |

The paired end-to-end comparisons contain 59 three-qubit and 38 four-qubit
circuits. Twenty-one additional four-qubit OSR results have no completed
BQSKit counterpart. Extra results available from only one method remain
available for analysis and are not counted as CNOT-quality wins.

[`results_archive_20261028/manifest.json`](results_archive_20261028/manifest.json)
records byte sizes and SHA-256 hashes for the result records, source and final
circuits, and referenced audit evidence. The source inputs are already tracked
in this repository. The manifest also records the packaging source commit and
each measurement's source fingerprint. The packaging commit identifies this
snapshot; it does not imply that every measurement used that implementation.
Per-circuit configurations preserve the actual seeds, precision settings,
partition sizes, synthesis thresholds, and routing choices. Stage times are
the recorded measurements, not portable predictions for another machine.

The completed runs reference approximately 8.82 GB of compressed rewrite
audits and 1.89 GB of OSR routing catalogs. These files are distributed as
assets of the
[`benchmark-results-20261028` release](https://github.com/rakytap/sequential-quantum-gate-decomposer/releases/tag/benchmark-results-20261028),
rather than as Git blobs. Each asset is an uncompressed tar containing the
original gzip files and is smaller than 2 GiB. This preserves the recorded
artifact hashes without forcing ordinary clones to download the full traces.
The routing catalogs are analysis/debug caches; their metadata explicitly
distinguishes them from the rewrite-audit certificates.

Temporary SQLite indexes, export-recovery backups, incomplete/unreferenced
audit streams, and the old reference directories are excluded. Their local
files are left in place. A recorded failure or timeout remains in `results.json`
even when its incomplete trace is excluded from the completed-run evidence.

### Reproduce the tables without rerunning synthesis

From the repository root:

```bash
python benchmarks/partitioning/archive_results.py verify \
  --manifest benchmarks/partitioning/results_archive_20261028/manifest.json
python benchmarks/partitioning/compare_osr_bqskit.py --no-fetch --include-4q \
  --output /tmp/comparison-20261028.tex
```

The generated tables are also stored in
[`results_archive_20261028/comparison.tex`](results_archive_20261028/comparison.tex).
The comparison uses matching recorded per-block process-infidelity tolerances
and matching input qubit/CNOT counts. It compares the complete workflows:
OSR uses `exact-osr` routing, while the BQSKit records use `seqpam-quick`.
It does not isolate the local synthesis algorithm or assume identical routers.
Use the snapshot's own table file when citing its results; an earlier manuscript
cutoff or the separate frozen 772-target local-synthesis study is a distinct
cohort.

### Obtain and inspect the full evidence

Run these commands in a fresh checkout of the archive commit. Download the
release assets to an empty directory outside the checkout:

```bash
gh release download benchmark-results-20261028 \
  --repo rakytap/sequential-quantum-gate-decomposer \
  --dir /tmp/benchmark-results-20261028 --pattern '*.tar' --pattern SHA256SUMS
(cd /tmp/benchmark-results-20261028 && sha256sum --check SHA256SUMS)
for asset in /tmp/benchmark-results-20261028/*.tar; do
  tar -xf "$asset" -C .
done
python benchmarks/partitioning/archive_results.py verify --include-evidence \
  --manifest benchmarks/partitioning/results_archive_20261028/manifest.json
```

Checksum verification establishes archive integrity; it does not rerun the
numerical correctness checks. The result records retain the original audit
verification summaries. To independently replay an audit with a built SQUANDER
environment, use its recorded digest:

```python
import json
from pathlib import Path
from squander.decomposition.qgd_Wide_Circuit_Optimization import verify_rewrite_audit

folder = Path("benchmarks/partitioning/IBMEagle_results_3qbit_TreeSearch")
entry = json.loads((folder / "results.json").read_text())["tof_3.qasm"]
audit = entry["rewrite_audit"]
report = verify_rewrite_audit(audit["file"], expected_sha256=audit["sha256"])
print(report)
```

Large audit replays can take substantial time and temporary disk space.
The archived output circuits and audit traces support inspection without
rerunning the optimization benchmarks. The separate local-synthesis experiment
can sample these archived targets with its documented size limits; those
filters do not remove evidence from this archive.

To prepare another snapshot, choose a new output directory outside the checkout
and a new release tag:

```bash
python benchmarks/partitioning/archive_results.py prepare \
  --output-dir /path/outside/checkout/new-benchmark-snapshot \
  --release-tag benchmark-results-YYYYMMDD
```

The preparation command freezes the small files into `snapshot/`, verifies the
recorded evidence digests while creating `assets/*.tar`, and writes a manifest
and `assets/SHA256SUMS`. It stops if the result metadata changes during capture.
No benchmark is rerun or interrupted.
