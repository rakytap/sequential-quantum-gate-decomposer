# Symbolic gate reference comparisons

## Current motif/OSR reference edition

[`synthesis_reference_20261106.json.gz`](synthesis_reference_20261106.json.gz)
merges all 327 × 32 cells and both phase policies across 29 source snapshots.
Each retained circuit minimizes the recorded lexicographic objective among
verified candidates: entanglers, original parameter-sensitive score, local count.
This is **not a claim of global optimality**. The indexed
[gate and Clifford PDFs and editable LaTeX](../../docs/symbolic_atlas/README.md)
are generated from this archive.

| Current outcome | Literal phase | Projective |
| --- | ---: | ---: |
| Symbolically verified | 7,702 | 10,196 |
| Proved obstruction with recorded scope/condition | 2,762 | 268 |
| Retained timeout, unresolved construction or compiler error | 0 | 0 |

There are 4,702 cells with a verified literal circuit and a strictly cheaper
projective circuit. No physical GP operation is allowed. Register constructors
retain their width/matrix fixtures; generic obstructions retain their conditions.
All 20,928 phase modes are classified, while optimality remains a separate task.

The successful constructions still exceed the published two-B/three-SQiSW
uniform caps in **15 literal and 15 projective modes**, down from 158 each.
The [gap CSV](../../docs/symbolic_atlas/known_optimization_gaps.csv) lists them.
Among eligible fixed-target SDK comparisons, entanglers are lower in 11 cases,
equal in 2,662, and higher in zero. All eleven improvements concern fFredkin
in distinct basis presets. The cleaned matched local-count audit has
180 lower, 2,476 equal and 6 higher counts at equal entangler count.
C3X/Rzz/U3 retains 28 locals because its score 115.594 beats the proved
24-local alternative's 116.37. A count gap need not be a score regression.
See the [cited comparison appendix](../../docs/symbolic_notable_results/notable_results.pdf),
[local-gap CSV](../../docs/symbolic_notable_results/local_optimization_gaps.csv),
and [native layer opportunities](../../docs/symbolic_notable_results/native_local_layer_opportunities.csv).
Layer count opportunities are not certified score improvements.

New native proofs include controlled SU(2)/phase charts, B echoes, SQiSW
exchange and fixed motifs, a parity-block Sycamore chart, a two-pulse CZS
matchgate chart, continuous IonQ MS control frames, and exact local rewrites.
Numerical SDK output only proposes fixed-target circuits. Rational-π and
explicit algebraic Euler-angle proposals must pass a complete exact identity.
Raw losing candidates and failed attempts remain available for audit.
The normal compiler also has 313 fixed-target template cells;
it checks full identities again before accepting these templates.

Matched curated lower bounds certify 287 literal and 158 projective entangler
minima, excluding separate one-qubit certificates. Nonzero-entangler minimum
flags total 671 literal and 857 projective; conditional flags total
342 and 411, respectively. Local-score proofs are separate.
The Clifford volume has 192 phase targets in each of 12 local alphabets:
1,488 reachable and 816 determinant-group exclusions. H/S has a global
shortest-path certificate; other alphabets keep their narrower search scopes.
UniversalQCompiler remains deferred with zero executed comparisons.

The raw full run and all construction refinements are preserved:

| Raw file | Cells | Budget per mode | Source revision |
| --- | ---: | ---: | --- |
| `synthesis_full_osr_20261106.json.gz` | 10,464 | 30.0 s | `b391dd11` |
| `synthesis_motifs_20261106.json.gz` | 928 | 60.0 s | `c21e0494` |
| `synthesis_motif_refinement_20261106.json.gz` | 192 | 90.0 s | `62b44f65` |
| `synthesis_phase_proofs_20261106.json.gz` | 6 | 120.0 s | `d3d3c4b8` |
| `synthesis_completion_20261106.json.gz` | 288 | 120.0 s | `2c39f362` |
| `synthesis_fixed_native_20261106.json.gz` | 96 | 45.0 s | `2c39f362` |
| `synthesis_fixed_pivots_20261106.json.gz` | 9 | 90.0 s | `0bdf752f` |
| `synthesis_final_exact_20261106.json.gz` | 2 | 120.0 s | `5f2c8f6b` |
| `synthesis_sdk_locals_20261106.json.gz` | 235 | 45.0 s | `c5e84e60` |
| `synthesis_cleaned_locals_20261106.json.gz` | 129 | 120.0 s | `0d4979ac` |
| `synthesis_local_blocks_20261106.json.gz` | 98 | 90.0 s | `928fcf7c` |
| `synthesis_native_echoes_20261106.json.gz` | 174 | 90.0 s | `40f98fe3` |
| `synthesis_native_controlled_20261106.json.gz` | 174 | 120.0 s | `603df966` |
| `synthesis_sdk_gauges_20261106.json.gz` | 24 | 180.0 s | `603df966` |
| `synthesis_native_ms_axes_20261106.json.gz` | 6 | 120.0 s | `ba8e01b4` |
| `synthesis_sdk_gauge_scalars_20261106.json.gz` | 11 | 180.0 s | `ba8e01b4` |
| `synthesis_fixed_transfers_20261106.json.gz` | 51 | 120.0 s | `66f807ca` |
| `synthesis_local_blocks_deep_20261106.json.gz` | 11 | 600.0 s | `9b1542a2` |
| `synthesis_native_sqrt_swap_20261106.json.gz` | 2 | 180.0 s | `f03198c2` |
| `synthesis_local_periods_20261106.json.gz` | 9 | 120.0 s | `bba8ccc7` |
| `synthesis_native_local_layers_20261106.json.gz` | 34 | 90.0 s | `c5640588` |
| `synthesis_local_axes_20261106.json.gz` | 9 | 120.0 s | `24dcf58b` |
| `synthesis_native_phase_absorption_20261106.json.gz` | 8 | 60.0 s | `6beb0a60` |
| `synthesis_native_fixed_matchgates_20261106.json.gz` | 21 | 120.0 s | `cbc150cf` |
| `synthesis_sdk_algebraic_frames_20261107.json.gz` | 8 | 300.0 s | `c6bf1ffd` |

Source hashes identify the implementation of each run. `mode_source_run_indices`
identifies the origin of each retained phase result; `synthesis_run_index`
identifies the latest attempt. `previous_attempts` and `raw_attempt` preserve
history, including exact alternatives that lose under the declared score.
The [source-validation report](../../docs/symbolic_atlas/source-validation.json)
checks each listed new refinement against its recorded Git blobs. Earlier
historical snapshots retain their own provenance and scopes.

Reproduce the merge and reference analysis from the repository root:

```bash
python examples/decomposition/symbolic_archive_analysis.py \
  --input benchmarks/symbolic/synthesis_reference_20261106_classified.json.gz \
  --supplement benchmarks/symbolic/synthesis_sdk_locals_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_cleaned_locals_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_local_blocks_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_echoes_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_controlled_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_sdk_gauges_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_ms_axes_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_sdk_gauge_scalars_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_fixed_transfers_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_local_blocks_deep_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_sqrt_swap_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_local_periods_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_local_layers_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_local_axes_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_phase_absorption_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_native_fixed_matchgates_20261106.json.gz \
  --supplement benchmarks/symbolic/synthesis_sdk_algebraic_frames_20261107.json.gz \
  --output /tmp/synthesis-reference.json.gz
```

The classified input preserves the twelve-run classification edition; the
`initial` reference archive preserves its earlier eight-run input. Original raw
compiler failures remain in history even when a later verified result wins.

The cleaned numerical SDK baseline is `reference_locals_20261106.json.gz`.
Its native block cleanup and phase-sensitive replay remove adapter identity
rotations; they make no symbolic-family claim. Regenerate the local audit with:

```bash
python examples/decomposition/symbolic_notable_results.py \
  --archive benchmarks/symbolic/synthesis_reference_20261106.json.gz \
  --baselines benchmarks/symbolic/reference_locals_20261106.json.gz \
  --output-dir /tmp/notable-results
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
