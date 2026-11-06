# Symbolic gate reference comparisons

`reference_comparison_20261106.json.gz` records one explicit parameter binding
for each of the 327 catalog definitions against all 32 basis presets. It is a
reference dataset for developing the symbolic search, not a report of new
optimal symbolic synthesis results.

The eleven supported Qiskit adapters provide 3,597 matrix-verified numerical
upper bounds. The other 6,867 cells explicitly report `unsupported_sdk_basis`.
Every selected circuit passed verification, every supported comparison had a
valid circuit, and no selected result exceeded an applicable fixed-gate
literature bound in the curated ledger (24 comparisons).

The run uses Qiskit 2.2.3, SymPy 1.14.0, NumPy as recorded in the JSON, and seed
219. Connectivity is all-to-all with zero ancillas. All operations, parameters,
global phases, errors, attempted inputs, source checksums, and compiler failures
are retained. Scalar-phase corrections are explicit operations on the delivered
circuit's global phase; the verifier does not ignore phase differences.
Inputs are arbitrary states, and both input and output wire orders are fixed.
Register structure is retained; virtual output permutations are materialized
through Qiskit's permutation-preserving compilation path and charged.

| Gate | Selected CNOT upper bound |
| --- | ---: |
| Margolus | 3 |
| RCCX | 3 |
| Peres | 5 |
| Toffoli / CCX | 6 |
| Fredkin / CSWAP | 7 |

The original parameter-sensitive score is preserved in the symbolic search.
These numerical compiler outputs do not have that score: SDK parameter-slot
counts are separate diagnostics. A successful sample is not a symbolic proof
for all parameter values. Proved minima and parameter-dependent exact tests
are documented in [the symbolic guide](../../docs/symbolic_decomposition.md)
and the source-backed reference ledger.

Regenerate from the repository root with the symbolic dependencies and Qiskit:

```bash
python examples/decomposition/symbolic_reference_comparison.py \
  --basis all --output /tmp/symbolic-reference-comparison.json
```

The command exits nonzero for a compiler failure or a regression against an
applicable recorded literature count. Interrupted runs retain a `.partial.json`
checkpoint. Source checksums identify the exact implementation used even when
the recorded Git revision was the parent of the commit containing this data.

Read the archived report without extracting it:

```python
import gzip
import json

with gzip.open("benchmarks/symbolic/reference_comparison_20261106.json.gz", "rt") as stream:
    report = json.load(stream)
print(report["summary"])
```

## Symbolic phase policies

[`synthesis_20261106.json.gz`](synthesis_20261106.json.gz) is the first bounded
symbolic construction archive (594,120 bytes compressed). It enumerates all
10,464 cells using eight seconds per phase mode. The original pass and the
32-cell MCPhase integer-constructor correction are preserved as separate
`synthesis_runs`, with per-row run indices, source hashes and previous failed
attempts. This is a construction/development dataset, not an all-optimal result.

| Outcome | Phase exact | Projective |
| --- | ---: | ---: |
| Symbolically verified | 6,584 | 8,819 |
| Proved obstruction, under the recorded scope/condition | 2,586 | 264 |
| Time limit | 465 | 393 |
| Unresolved construction | 829 | 988 |
| Compiler errors after correction | 0 | 0 |

There are 4,058 cells where an exact circuit is verified and the retained
projective circuit is strictly cheaper. Matched fixed-target references certify
225 retained exact and 69 retained projective entangler minima; these counts
exclude the separate zero-entangler one-qubit certificates. No minimum local
score follows from matching an entangler lower bound.

Among eligible fixed-target SDK comparisons, 2,434 tie its entangler upper bound
and 38 use more entanglers. Those 38 remain optimization work; no SDK improvement
or novelty is claimed. They occur in the iSWAP or tunable XX/ZZ/ZX bases, where
replacing each CNOT independently leaves avoidable interactions. Unsupported
SDK bases and unverified constructions contribute no win/loss outcome.

The OSR update is archived separately in
[`synthesis_osr_20261106.json.gz`](synthesis_osr_20261106.json.gz). It retains all
10,464 cells, with **284 distinct cells rerun** using the implementation at
`1076988f`; the remaining cells retain their earlier constructions and source
snapshots. The raw reruns are
[`synthesis_osr_regressions_20261106.json.gz`](synthesis_osr_regressions_20261106.json.gz)
(16 targets in four bases) and
[`synthesis_osr_families_20261106.json.gz`](synthesis_osr_families_20261106.json.gz)
(Canonical, CP, CRX, fSim, PhasedFSim, SWAP and iSWAP in all 32 presets).
The four overlapping SWAP cells retain both attempt histories.

| Outcome after the OSR update | Phase exact | Projective |
| --- | ---: | ---: |
| Symbolically verified | 6,622 | 8,837 |
| Proved obstruction, under the recorded scope/condition | 2,588 | 264 |
| Time limit | 425 | 378 |
| Unresolved construction | 829 | 985 |
| Compiler errors | 0 | 0 |

All 38 former fixed-target SDK entangler regressions now tie the baseline.
There are 2,481 eligible fixed-target ties and zero worse counts; no SDK win or
novelty claim is made. Among rerun modes with verified old and new circuits,
81 phase-exact and 129 projective circuits have strictly lower costs, and none
has a higher cost. Newly verified modes are additional to those improvements.
These use the original lexicographic entangler/parameter/local-count objective;
they are not a claim of globally minimum local score.

Every projective mode in the two raw reruns either has an exact symbolic
construction or a proved obstruction. Six phase-exact family cases still hit
the 30-second limit; they were already timeouts in the earlier archive.
The merged archive has 4,087 cells where a verified exact construction is more
expensive than the retained projective construction. Each row resolves its
`synthesis_run_index` in the flattened `synthesis_runs`, including indices in
its `previous_attempts`; appending a rerun preserves older corrections' hashes.

Reproduce the merge and fixed-target analysis with:

```bash
python examples/decomposition/symbolic_archive_analysis.py \
  --input benchmarks/symbolic/synthesis_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_osr_regressions_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_osr_families_20261106.json.gz \
  --output /tmp/synthesis-osr-with-references.json.gz
```

The OSR guide in [`docs/symbolic_decomposition.md`](../../docs/symbolic_decomposition.md)
describes the opaque-matrix API, exact cut/minor certificates, native Pauli
networks, phase handling and scoped optimality claims. The planned reference
atlas will sort targets alphabetically, index them, and place the 32 basis
panels on each target's page. Verified circuits, proved obstructions and open
cases must be distinguished, and exact/projective variants must retain their
phase and certificate labels. Atlas rendering is not part of these data files.

The newer `symbolic_synthesis_archive.py` runner preserves free real parameters
and forbids physical GP operations. Its phase-exact results require literal
matrix equality; its projective results automatically record a virtual scalar
at zero physical cost. When exact is no more expensive, the duplicate
projective circuit is omitted while its observed status remains in the row.

```bash
python examples/decomposition/symbolic_synthesis_archive.py \
  --basis all --workers 4 --seconds-per-mode 10 \
  --output /tmp/symbolic-synthesis.json.gz
```

There are 327 target definitions and 32 presets, or 10,464 cells and 20,928 phase
checks. Fixed-width parameters, including declared optional global shifts, are
free real symbols. Register constructors have recorded width/matrix fixtures.
A timeout is an open case, and a verified circuit is an upper bound unless its
own certificate establishes optimality. The one-qubit H/S certificate covers
all 192 exact Clifford phases; continuous-alphabet searches state their more
limited word-search scope explicitly.

Runs checkpoint every 100 cells. Add `--resume` to continue an interrupted run
using the same output, implementation and time budget. The full source hashes
identify the implementation even when the recorded Git revision precedes the
commit containing the files. Wall times include the caches available in that
worker and are diagnostic, not matched scientific performance measurements.

UniversalQCompiler is deferred by the user, with zero executed comparisons. Its
optional Wolfram exchange adapter has expression/wire/sign unit tests but has
not been run against a licensed kernel. It contributes no wins, losses or
minimum-count claims. A future native analytical run can be shared across
compatible bases; rebased output must be identified and reverified separately.

Apply the fixed-target reference analysis with:

```bash
python examples/decomposition/symbolic_archive_analysis.py \
  --input /tmp/symbolic-synthesis.json.gz \
  --output /tmp/symbolic-synthesis-with-references.json.gz
```

It attaches applicable published CNOT lower bounds only when each fixed native
primitive is itself proved locally equivalent to one CNOT. A matching verified
construction then certifies the entangler count, without certifying its local
score. The numerical SDK comparison is restricted to fixed targets and the
projective phase policy. Free-parameter families are not compared to one
sampled SDK binding, and an improvement over an SDK output is not called novel.
Compare the parameter-sensitive score within a supplied alphabet: native angle
units and macro granularity differ between presets.

To reproduce the two-pass construction workflow of this archive:

```bash
python examples/decomposition/symbolic_synthesis_archive.py \
  --basis all --workers 24 --seconds-per-mode 8 --output /tmp/synthesis.json.gz
python examples/decomposition/symbolic_synthesis_archive.py \
  --gate MCPhase --basis all --workers 24 --seconds-per-mode 8 \
  --output /tmp/mcphase.json.gz
python examples/decomposition/symbolic_archive_analysis.py \
  --input /tmp/synthesis.json.gz --supplement /tmp/mcphase.json.gz \
  --output /tmp/synthesis-with-references.json.gz
```

The fixed adapter in the current source should eliminate the original control
count exception even in a single full run. Time-limited outcomes can vary with
available compute and caches; every accepted circuit still requires its exact
identity proof. `--supplement` validates identical targets and phase contracts
before replacing selected cells, and keeps each earlier attempt and revision.
