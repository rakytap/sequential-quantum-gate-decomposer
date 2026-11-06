# Symbolic circuit atlas — 6 November 2026

[Gate atlas](gates.pdf) · [All-phase Clifford atlas](cliffords.pdf) ·
[Gate preview](preview-gates.pdf) · [Clifford preview](preview-cliffords.pdf)

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
projective result is strictly cheaper; this occurs in 4,615 cells with a
verified exact construction. Obstructions for parameterized families may be
generic or conditional; their conditions remain in the data. Register
constructors retain their recorded width or matrix fixture, rather than
claiming a circuit for every register size.

A minimum label concerns entanglers only unless a separate local certificate
says otherwise. Gate-panel labels explicitly say **E minimum** or **E upper
bound**. The original ordering is entanglers, parameter-sensitive local
score, then local gate count. There are 287 exact and 145 projective matches
to the curated fixed-target entangler lower bounds, in addition to the
one-qubit and OSR certificates. The eligible fixed-target Qiskit comparison
contains 11 lower counts, 2,662 ties and zero higher counts. The 11 lower counts
are fFredkin in different bases, including five CNOTs versus the stored SDK
upper bound of seven and six iSWAPs versus eight. These comparisons establish
neither novelty nor global optimality.

The archive currently marks 634 exact and 815 projective constructions with
nonzero entanglers as having unconditional minimum-count certificates. It also
records 253 exact and 322 projective conditional certificates. Most other
minimum-count flags concern zero-entangler circuits. These are entangler
certificates; the parameter-sensitive local objective has a separate scope.

The literature audit also checks universal two-qubit upper bounds: two
[B gates](https://arxiv.org/abs/quant-ph/0312193), or three
[SQiSW gates](https://arxiv.org/abs/2105.06074), with the corresponding arbitrary
local alphabets. There are still **158 exact and 158 projective modes above
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
global shortest-path certificate over its full 192-element phase group. For
continuous and Clifford+T alphabets, the listed word search has the narrower
scope printed in the volume; its local cost is not asserted globally minimal.

The construction engine combines exact OSR constraints with native Pauli
networks and constructive motifs: controlled rotations, local frames,
exchange echoes, Gray-code multiplexors, Walsh phase polynomials and binary
linear networks. DEAP is retained as an experimental alternative. Every
accepted free-parameter circuit needs an exact identity proof. Numerical
matches can propose candidates and never certify them.

## Data and reproduction

The authoritative merged data are
[`synthesis_reference_20261106.json.gz`](../../benchmarks/symbolic/synthesis_reference_20261106.json.gz).
Twelve source snapshots and all earlier attempts are preserved. Each retained
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
  lists the 316 phase modes above the audited published uniform bounds. These
  are optimization gaps among verified circuits, rather than missing proofs
  of their matrix identities.
- [`sources.zip`](sources.zip) contains both LaTeX volumes and every referenced
  Qiskit circuit fragment, plus the manifest, build report, gap list and exact
  phase examples. Its top-level `gates.tex` and `cliffords.tex` are the LaTeX
  sources of the two PDFs; keep their `circuits/` directory together.

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
