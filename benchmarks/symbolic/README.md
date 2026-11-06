# Symbolic gate reference comparisons

## Current motif/OSR reference edition

[`synthesis_reference_20261106.json.gz`](synthesis_reference_20261106.json.gz)
merges all 327 × 32 cells, retaining the least-cost verified incumbent for each
phase policy across thirteen implementation snapshots. The indexed
[gate and Clifford PDFs](../../docs/symbolic_atlas/README.md) and their sources
are generated from this archive. The earlier datasets below remain historical
snapshots.

| Current outcome | Phase exact | Projective |
| --- | ---: | ---: |
| Symbolically verified | 7,702 | 10,196 |
| Proved obstruction, with recorded scope/condition | 2,762 | 268 |
| Time limit | 0 | 0 |
| Unresolved construction | 0 | 0 |
| Retained compiler errors | 0 | 0 |

The construction history includes a complete bounded run and the targeted
supplements below:

| Raw file | Cells | Budget per mode | Source revision |
| --- | ---: | ---: | --- |
| `synthesis_full_osr_20261106.json.gz` | 10,464 | 30 s | `b391dd11` |
| `synthesis_motifs_20261106.json.gz` | 928 | 60 s | `c21e0494` |
| `synthesis_motif_refinement_20261106.json.gz` | 192 | 90 s | `62b44f65` |
| `synthesis_phase_proofs_20261106.json.gz` | 6 | 120 s | `d3d3c4b8` |
| `synthesis_completion_20261106.json.gz` | 288 | 120 s | `2c39f362` |
| `synthesis_fixed_native_20261106.json.gz` | 96 | 45 s | `2c39f362` |
| `synthesis_fixed_pivots_20261106.json.gz` | 9 | 90 s | `0bdf752f` |
| `synthesis_final_exact_20261106.json.gz` | 2 | 120 s | `5f2c8f6b` |
| `synthesis_sdk_locals_20261106.json.gz` | 235 | 45 s | `c5e84e60` |

The complete raw run includes 16 register-constructor compiler failures; the
motif supplement resolves all of them. Raw outcomes are preserved even when
the merged report has a successful replacement. Source hashes identify the
actual implementation. `mode_source_run_indices` records the origin of each
retained circuit; `synthesis_run_index` is the latest attempted run. Historical
attempts and their indices are retained recursively.

The completion pass reruns only the 288 cells containing the former 360 open
phase modes. The fixed-native pass exposed seven algebraic pivot exceptions,
which the nine-cell pivot rerun resolves in both modes. The final two-cell
pass resolves the exact Rzz/U3 C4X and C3SX cases by preserving simple
root-of-unity phases. These raw outcomes remain separate from the successful
incumbents; there are no compiler errors in the merged phase statuses.

The fixed-target local refinement proves 101 literal-phase and 138 projective
SDK lifts. Unproved proposals and one proof timeout retain their earlier
verified incumbents. It removes 137 of the former 235 equal-entangler local
count regressions in the original emitted SDK circuits. The stricter local
audit first applies native one-qubit SDK cleanup, including removal of adapter
identity rotations: against that baseline the gaps fall from 254 to 129.
The 138 native template cells are also available
to the normal compiler, which rechecks each exact identity. The full
[local audit and cited Sycamore comparison](../../docs/symbolic_notable_results/notable_results.pdf)
record local counts and distinguish them from the parameter-sensitive score.

There are 4,636 verified exact cells with a strictly cheaper projective result.
Matched fixed-target lower bounds certify 287 exact and 153 projective
entangler minima, excluding separate one-qubit certificates. The eligible
fixed-target Qiskit comparisons have 11 lower counts, 2,662 ties and zero higher
counts. The lower counts concern fFredkin in different bases, including five
CNOTs versus the SDK upper bound of seven and six iSWAPs versus eight. No
novelty or global local-cost claim follows from these comparisons.

All 20,928 phase modes are now classified. The header-only
[`open_cases.csv`](../../docs/symbolic_atlas/open_cases.csv) records that no
construction mode remains open. This does not establish optimality for every
verified circuit: the audit of the two-B and three-SQiSW uniform bounds still
finds 158 exact and 158 projective optimization gaps, listed in
[`known_optimization_gaps.csv`](../../docs/symbolic_atlas/known_optimization_gaps.csv).
The Cartan trace/parity analysis adds conditional CNOT minima, including four
for exact SU(2)-local synthesis versus three projectively where the trace
condition is nonzero. The Clifford reference has all 192
phase targets in each of 12 distinct local alphabets: 1,488 reachable circuits
and 816 phases excluded by the one-wire determinant group. The H/S basis has a
global local-cost certificate; other alphabet certificates retain their more
limited search scope.

Reproduce the current merge and reference analysis:

```bash
python examples/decomposition/symbolic_archive_analysis.py \
  --input benchmarks/symbolic/synthesis_reference_20261106_classified.json.gz \
  --supplement benchmarks/symbolic/synthesis_sdk_locals_20261106.json.gz \
  --output /tmp/synthesis-reference.json.gz
```

`synthesis_reference_20261106_classified.json.gz` preserves the twelve-run
classification edition before local refinement; the earlier
`synthesis_reference_20261106_initial.json.gz` preserves its eight-run input.

`reference_locals_20261106.json.gz` preserves the numerical SDK baseline after
native one-qubit cleanup and phase-sensitive operator replay. Its separate
normalization provenance identifies the original reference file and cleanup
source. It changes no entangler counts and makes no symbolic-family claim:

```bash
python examples/decomposition/symbolic_local_baselines.py \
  --input benchmarks/symbolic/reference_comparison_20261106.json.gz \
  --output /tmp/reference-locals.json.gz
python examples/decomposition/symbolic_notable_results.py \
  --archive benchmarks/symbolic/synthesis_reference_20261106.json.gz \
  --baselines /tmp/reference-locals.json.gz --output-dir /tmp/notable-results
```

## Initial numerical reference and construction snapshots

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
networks, phase handling and scoped optimality claims. The current reference
atlas sorts targets alphabetically, indexes them, and places 32 basis panels
on each target's page. Verified circuits, proved obstructions and open cases
have distinct labels, with exact/projective phase policies and certificate
scopes preserved. Its PDFs and manifests are separate artifacts linked above.

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
