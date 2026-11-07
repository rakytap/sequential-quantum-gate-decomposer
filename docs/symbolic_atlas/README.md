# Symbolic circuit atlas — 7 November 2026

[Gate atlas](gates.pdf) · [All-phase Clifford atlas](cliffords.pdf) ·
[Gate preview](preview-gates.pdf) · [Clifford preview](preview-cliffords.pdf)

[Notable results and local-gate comparisons](../symbolic_notable_results/notable_results.pdf) ·
[Editable comparison LaTeX](../symbolic_notable_results/notable_results.tex) ·
[Comparison source bundle](../symbolic_notable_results/sources.zip)

[Global minimum proofs and objective audit](../symbolic_optimality/proofs.pdf) ·
[Editable proof LaTeX](../symbolic_optimality/proofs.tex) ·
[Exact certificate replays](../symbolic_optimality/verification.json)

The approved objective revision clamps negative denominator penalties. An exact
audit of 1,390 retained angle expressions preserved every score at migration.
The archive now includes 372 literal and 1,144 projective global minimum
certificates from primitive/support floors, independently of the all-phase
Clifford proofs. These counts can overlap other certificates and do not
establish global optimality of the full grid.

An independent native-count audit now supplies 3,290 matching certificates
after combining cut/spin and exchange proofs: 689 literal and 936 projective
fixed-target minima, plus 710 literal and 955 projective minimum uniform
family budgets. There are 1,224 phase modes that previously had no minimum
certificate. A uniform family budget is necessary to handle a certified hard
member; special parameter values can use fewer pulses. These certificates
do not establish minimum local scores. The original circuits, scores and
source-run indices are unchanged. See the
[coverage list](../symbolic_optimality/entangler_coverage.csv),
[validation summary](../symbolic_optimality/native_bounds_summary.json), and
[unmatched lower-bound attempts](../symbolic_optimality/unmatched_native_bounds.csv).
The last list records additional proof searches, independently of the complete
construction classification below.

The comparison LaTeX is self-contained, with inline circuit drawings: copy
`notable_results.tex` alone into Prism. Compile it locally with
`tectonic notable_results.tex`.
The [local-savings list](../symbolic_notable_results/local_savings.csv) contains
all 184 equal-entangler improvements against the cleaned SDK baseline.

The gate volume contains 327 alphabetically indexed target pages, each with
32 preserved basis presets. The separate Clifford volume contains 24 circuits
per phase page: 192 phase targets in each of the 12 distinct local alphabets
shared by those presets. An exact matrix key fixes the 24 representative
conventions. Both volumes use Qiskit's vector LaTeX circuit drawings, clickable
indices and bookmarks. Gate pages are A1 landscape; Clifford pages are A3.

Every archived pair now has a verified construction or a scoped mathematical
obstruction in both phase modes. **Complete classification does not mean all
circuits are globally optimal.** The current merged archive has these outcomes:

| Outcome | Exact phase | Projective |
| --- | ---: | ---: |
| Symbolically verified construction | 7,702 | 10,196 |
| Proved obstruction within its recorded scope | 2,762 | 268 |
| Time limit | 0 | 0 |
| Construction unresolved | 0 | 0 |
| Retained compiler errors | 0 | 0 |

Exact mode permits no physical GP gate or free scalar correction. Projective
mode records its scalar explicitly. Both circuits are shown when the
projective result is strictly cheaper; this occurs in 4,716 cells with a
verified exact construction. Obstructions for parameterized families may be
generic or conditional; their conditions remain in the data. Register
constructors retain their recorded width or matrix fixture, rather than
claiming a circuit for every register size.

A minimum label concerns entanglers only unless a separate local certificate
says otherwise. Gate-panel labels explicitly say **E minimum** or **E upper
bound**. The ordering is entanglers, parameter-sensitive local
score with nonnegative denominator penalties, then local gate count. There are 287 exact and 158 projective matches
to the curated fixed-target entangler lower bounds, in addition to the
one-qubit and OSR certificates. The eligible fixed-target Qiskit comparison
contains 11 lower counts, 2,662 ties and zero higher counts. The 11 lower counts
are fFredkin in different bases, including five CNOTs versus the stored SDK
upper bound of seven and six iSWAPs versus eight. These comparisons establish
neither novelty nor global optimality.

The archive currently marks 671 exact and 857 projective constructions with
nonzero entanglers as having unconditional minimum-count certificates. It also
records 342 exact and 411 projective conditional certificates. Most other
minimum-count flags concern zero-entangler circuits. These are entangler
certificates; the parameter-sensitive local objective has a separate scope.

The literature audit also checks universal two-qubit upper bounds: two
[B gates](https://arxiv.org/abs/quant-ph/0312193), or three
[SQiSW gates](https://arxiv.org/abs/2105.06074), with the corresponding arbitrary
local alphabets. There are still **10 exact and 10 projective modes above
those published bounds**. These are known optimization gaps among successful
constructions, listed separately and marked on the relevant PDF panels. A
count below a uniform worst-case bound is not a new per-target result.

Two phase effects have exact matching certificates in
[`notable_examples.json.gz`](notable_examples.json.gz). A Cartan kernel with
`a=b=c=pi/8` needs three CNOTs projectively and four for literal phase with
SU(2)-only Rz/Ry locals; determinant parity excludes an odd exact count. More
generally that distinction is certified where
`sin(2a) sin(2b) sin(2c) != 0`. In the Rxx/Rz/Ry basis, `Rxx(pi)` needs zero
entanglers projectively and one for literal phase. These examples demonstrate
resource-policy differences; no novelty claim is made.

The Clifford tables contain 1,488 reachable exact circuits and 816 phases
excluded by the alphabet's one-wire determinant group. The H/S table has a
global shortest-path certificate over its full 192-element phase group.
H/S/T/Tdg now has an exhaustive exact-word proof including non-Clifford
intermediates: 628 matrices through radius seven certify all 192 targets,
with maximum minimum length 14. Continuous alphabets retain the narrower
scope printed in the volume.

The five formerly over-cap B families now use two B gates in both phase modes.
The remaining ten cap gaps per phase belong to the five free families in the
two SQiSW presets.

New exact native constructions include two SQiSWs and three U3 rotations for
B with literal phase, three SQiSWs for Sycamore and both square-root-SWAPs,
two-pulse DCX frames, and a six-local two-pulse matchgate chart for the
entire real-parameter CZS family. Component identities and explicit polar
branches are proved before substitution.

The construction engine combines exact OSR constraints with native Pauli
networks and constructive motifs: controlled rotations, local frames,
exchange echoes, Gray-code multiplexors, Walsh phase polynomials and binary
linear networks. DEAP is retained as an experimental alternative. Every
accepted free-parameter circuit needs an exact identity proof. Numerical
matches can propose candidates and never certify them.

## Data and reproduction

The authoritative merged data are
[`synthesis_reference_20261106.json.gz`](../../benchmarks/symbolic/synthesis_reference_20261106.json.gz).
Thirty-five source snapshots and all earlier attempts are preserved. Each retained
phase result identifies its actual source run, even when a later attempt
failed. The raw full run and motif/phase supplements are described in the
[archive README](../../benchmarks/symbolic/README.md).

- [`manifest.json.gz`](manifest.json.gz) maps every displayed panel to the
  archive and source runs and stores exact expression aliases.
- [`summary.json`](summary.json) records counts, provenance validation and
  the synthesis test result; [`build-report.json`](build-report.json) records
  PDF hashes, page counts and repaired index links.
- [`open_cases.csv`](open_cases.csv) is now header-only: no phase mode remains
  unclassified. [`known_optimization_gaps.csv`](known_optimization_gaps.csv)
  lists the 20 phase modes above the audited published uniform bounds. These
  are optimization gaps among verified circuits, rather than missing proofs
  of their matrix identities.
- [`sources.zip`](sources.zip) contains both LaTeX volumes and every referenced
  Qiskit circuit fragment, plus the manifest, build report, gap list and exact
  phase examples. Its top-level `gates.tex` and `cliffords.tex` are the LaTeX
  sources of the two PDFs; keep their `circuits/` directory together.

The separate local audit has 184 lower counts, 2,477 ties and 1 higher count
at equal entangler count among eligible fixed targets, after native SDK local
cleanup removes adapter identity rotations. Against this cleaned baseline,
the exact refinements reduced the local-count gaps from 254 to 1 (and from
6 in the preceding release).
These counts do not compare the secondary symbolic score when the SDK angles
have no proved exact lift. See the
[full local comparison CSV](../symbolic_notable_results/local_comparisons.csv)
and [remaining local-count gaps](../symbolic_notable_results/local_optimization_gaps.csv).
The remaining count gap has a verified secondary-score tradeoff: C3X/Rzz/U3
uses 28 locals with score 115.594; its verified 24-local alternative scores
116.37. The declared objective retains the lower score.
The [native local-layer audit](../symbolic_notable_results/native_local_layer_opportunities.csv)
also records opportunities beyond the SDK-supported bases; these count
bounds do not certify an improvement of the parameter-sensitive score.
The [source validation](source-validation.json) checks every listed new raw
refinement hash against its recorded Git revision.

The newest local improvements include C3SX/iSWAP at 40 locals versus SDK 56,
C4X/iSWAP at 78 versus 113, C4X/CZ/Rz/Rx at 69 versus 72, and
C4X/IBM-CZ at 130 versus 143. Peres/Rxx/Rz/Ry matches 35.
These are proved constructions, not unrestricted local-minimum claims.

The Sycamore section compares three local gates with 27 in the cited published
figure and 13 from independent Qiskit synthesis in Rz/SX/CX; both comparisons
retain three CNOTs and use the projective policy for that alphabet.

To edit or rebuild the existing LaTeX without regenerating the circuits,
extract `sources.zip` into a directory and open `gates.tex` or `cliffords.tex`.
From that directory, with Tectonic and `pypdf` installed, run:

```bash
python build_symbolic_atlas.py --source-dir . --workers 16
```

Generate the sources and compile the PDFs from the repository root:

```bash
python examples/decomposition/symbolic_circuit_atlas.py \
  --archive benchmarks/symbolic/synthesis_reference_20261106.json.gz \
  --output-dir /tmp/symbolic-atlas
python examples/decomposition/build_symbolic_atlas.py \
  --source-dir /tmp/symbolic-atlas --workers 16
```

The builder requires Tectonic and `pypdf`; rendering requires Qiskit,
`pylatexenc` and the symbolic dependencies. It compiles sheets independently,
checks that each fits one page, restores index destinations and bookmarks,
and uses separately compiled vector fragments if a dense sheet exceeds
TeX's memory limit. Long circuits continue in time order and exact expressions
use a glossary when needed. No operations are omitted to make a page fit.

UniversalQCompiler remains deferred with zero executed comparisons.
