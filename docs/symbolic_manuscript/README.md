# Theory manuscript and complete review package

[Manuscript](manuscript.pdf) · [Supplement](supplement.pdf) ·
[Response to the second report](referee-response.pdf) ·
[Editable main LaTeX](manuscript.tex) · [Paper sources](sources.zip) ·
[Full review package](review-package.zip) · [Release manifest](release-manifest.json)

Authors: Gregory Morse and Peter Rakyta. This is an internal review draft;
no submission has been performed. Author affiliations and individual
contributions still need author confirmation; see [editorial status](editorial-status.json).

The paper classifies controlled balanced-involution interaction costs and
extends the normal forms to all spectral strata. Trace-±2 reflections have
proved interior bounds of 4–8 CNOTs and 3–6 unrestricted interactions; their
interior optimum remains open. The SQiSW proposition expresses the known
two-pulse region as real polynomial inequalities suitable for symbolic families.

All compiler comparisons, ablations, application timings, and the negative
peephole experiment are preserved in the standalone supplement. The work
claims neither all-grid optimality nor compiler or literature dominance.

The native corrections include an exact two-SYC controlled-phase chart and
one-TK2 DCX. Remaining SYC/√SWAP gaps are flagged on atlas panels against
separately replayed numerical witnesses. Family samples are labelled as
samples. Remaining TK2 gaps are flagged against its known Cartan capacity.
These annotations do not replace exact words with numerical circuits.

The [optimality ledger](generated/optimality_ledger.csv) resolves each
positive-entangler effective cell to matched proof sources. The
[native-column gap ledger](generated/native_column_gaps.csv) gives exact
counts, shorter count leads, scopes, and provenance. The numerical audit
contains the actual witness operations and errors. Boolean optimality flags
alone do not count as lower-bound payloads.

For Prism, copy **`manuscript.tex`, `references.tex`, and `generated/`** from
`sources.zip`, preserving relative paths. Copy **`supplement.tex`** as well
to compile the separate supplement. Each uses the shared references and
generated directory. The response is a standalone `referee-response.tex`.

The full review ZIP contains the main paper, supplement, both responses,
both complete atlas PDFs, and editable paper/atlas source ZIPs. Atlas sources
contain `gates.tex`, `cliffords.tex`, `circuits/`, and `programs/`.

The current scientific grid is
`benchmarks/symbolic/synthesis_reference_20261107_referee.json.gz`.
The dated original archives remain immutable. New exact native runs and
numerical gap proposals are separately identified by hashes. Exact identity
checks, numerical replays, analytical capacities, and minimum proofs have
different scopes throughout.

Reproduction entry points:

```bash
python3 -m pytest tests/symbolic -q
python3 examples/decomposition/verify_symbolic_native_gaps.py
python3 examples/decomposition/build_symbolic_referee_revision.py
python3 examples/decomposition/verify_symbolic_manuscript_release.py --help
tectonic manuscript.tex
tectonic supplement.tex
tectonic referee-response.tex
```

The source ZIP includes current pure-symbolic modules, tests, numerical audit
scripts, and pinned environments. The independent SDK and C4X-route harnesses
use the recorded Python-3.12 environment. Quartz/QUESO require their recorded
external builds; their records retain commands, rules, and harness patch.
UniversalQCompiler remains unrun because a licensed Wolfram kernel was unavailable.
