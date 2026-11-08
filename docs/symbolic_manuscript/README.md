# Theory manuscript and complete review package

[Manuscript](manuscript.pdf) · [Quantum-template draft](quantum-submission.pdf) ·
[Supplement](supplement.pdf) · [Response to the third report](referee-response.pdf) ·
[Editable main LaTeX](manuscript.tex) · [Paper sources](sources.zip) ·
[Full review package](review-package.zip) · [Release manifest](release-manifest.json) · [Research reproduction map](REPRODUCE.md)

Authors: Gregory Morse and Péter Rakyta. Affiliations and contributions are
confirmed in [editorial status](editorial-status.json) and the main manuscript.
Gregory carried out the research, software, analysis, validation, experiments,
data, visualization, and original draft. Péter contributes scientific
discussions, drafting, reviewing, and editing. No external submission has been performed.

The paper classifies controlled balanced-involution interaction costs and
extends the normal forms to all spectral strata. Trace-±2 reflections now
have a literal **five-unrestricted-gate construction with the angle free**.
The proved interior bounds are 4–7 CNOTs and 3–5 unrestricted interactions.
A literal seven-CNOT family word is now proved, with only two varying
local angles. The matching interior lower bounds remain open; five/seven
minimality is explicitly a conjecture. Seven CNOTs is best known
for Fredkin in the literature audit, not a proved minimum.

The SQiSW proposition expresses the known two-pulse region as real polynomial
inequalities suitable for symbolic families. Compiler comparisons, ablations,
application timings, and the negative peephole experiment remain in the
standalone supplement. Neither all-grid optimality nor compiler dominance is claimed.

The exact monodromy audit now certifies minimum counts for all **64
fixed-target/native pairs** in the audited SYC/√SWAP subset. Its 112 proved
records also include named family samples. Thirty proof deadlines and eight
uncertified coordinate conversions remain inconclusive. Exact Makhlin
invariant checks and rational primal/Farkas witnesses support every accepted
count. A proved count is an existence statement with arbitrary local frames;
it does not replace a longer displayed circuit or prove local-gate optimality.
SYC's five-√SWAP minimum, versus the stored six-pulse word, remains visible.
Family samples and minimum uniform family budgets have separate labels.
The remaining TK2 gaps retain their known one-macro capacity annotations.

The [optimality ledger](generated/optimality_ledger.csv) resolves each
positive-entangler effective cell to matched proof sources. The
[native-column ledger](generated/native_column_gaps.csv) gives construction
counts, proved minima or numerical leads, scopes, and provenance.
[Monodromy verification](monodromy-verification.json) and
[reflection replay](reflection-verification.json) bind the new audits to hashes.

For Prism, copy **`manuscript.tex`, `references.bib`, and `generated/`** from
`sources.zip`, preserving relative paths. Copy **`supplement.tex`** to compile
the separate supplement. For the official Quantum format, also copy
**`quantum-submission.tex` and `quantumarticle.cls`**, and compile
`quantum-submission.tex`. It inputs the same manuscript in unpublished-review
format. The source ZIP also includes matching `.bbl` files for builds without BibTeX. Class provenance and the original license header are included.
The response is a standalone `referee-response.tex`.

The full review ZIP contains both main-paper formats, the supplement,
the current and earlier response PDFs, both complete atlas PDFs, and editable source ZIPs.
Atlas sources contain `gates.tex`, `cliffords.tex`, `circuits/`, and `programs/`.

The current grid is `benchmarks/symbolic/synthesis_reference_20261107_referee.json.gz`.
Dated original archives remain immutable. Exact identity checks, numerical
replays, analytical capacities, and minimum proofs have distinct scopes.

Reproduction entry points:

```bash
python3 -m pytest tests/symbolic -q
python3 examples/decomposition/verify_symbolic_third_referee.py
python3 examples/decomposition/verify_symbolic_native_gaps.py
python3 examples/decomposition/build_symbolic_referee_revision.py
python3 examples/decomposition/verify_symbolic_manuscript_release.py --help
tectonic manuscript.tex
tectonic quantum-submission.tex
tectonic supplement.tex
tectonic referee-response.tex
```

The source ZIP includes pure-symbolic modules, tests, audit scripts, licensed
monodromy data, exact certificates, numerical witnesses, and pinned environments.
Independent SDK, reflection fitting, and C4X-route harnesses use the recorded
Python-3.12 environment. Quartz/QUESO require their recorded external builds;
the records retain commands, rules, and harness patch. UniversalQCompiler
remains unrun because a licensed Wolfram kernel was unavailable.
