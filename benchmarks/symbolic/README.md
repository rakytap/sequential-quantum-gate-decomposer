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
