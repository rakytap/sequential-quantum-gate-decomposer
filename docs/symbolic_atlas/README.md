# Symbolic circuit atlas — 6 November 2026

[Gate atlas](gates.pdf) · [All-phase Clifford atlas](cliffords.pdf) ·
[Gate preview](preview-gates.pdf) · [Clifford preview](preview-cliffords.pdf)

The gate volume contains 327 alphabetically indexed target pages, each with
32 preserved basis presets. The separate Clifford volume contains 24 circuits
per phase page: 192 phase targets in each of the 12 distinct local alphabets
shared by those presets. An exact matrix key fixes the 24 representative
conventions. Both volumes use Qiskit's vector LaTeX circuit drawings, clickable
indices and bookmarks. Gate pages are A1 landscape; Clifford pages are A3.

This is a complete enumeration and a working reference edition, **not a claim
that every feasible pair is solved or globally optimal**. The current merged
archive has these outcomes:

| Outcome | Exact phase | Projective |
| --- | ---: | ---: |
| Symbolically verified construction | 7,598 | 10,069 |
| Proved obstruction within its recorded scope | 2,637 | 264 |
| Time limit | 97 | 98 |
| Construction unresolved | 132 | 33 |
| Retained compiler errors | 0 | 0 |

Exact mode permits no physical GP gate or free scalar correction. Projective
mode records its scalar explicitly. Both circuits are shown when the
projective result is strictly cheaper; this occurs in 4,547 cells with a
verified exact construction. Obstructions for parameterized families may be
generic or conditional; their conditions remain in the data. Register
constructors retain their recorded width or matrix fixture, rather than
claiming a circuit for every register size.

A minimum label concerns entanglers only unless a separate local certificate
says otherwise. The original ordering is entanglers, parameter-sensitive local
score, then local gate count. There are 287 exact and 141 projective matches
to the curated fixed-target entangler lower bounds, in addition to the
one-qubit and OSR certificates. The eligible fixed-target Qiskit comparison
contains 11 lower counts, 2,637 ties and zero higher counts. The 11 lower counts
are fFredkin in different bases, including five CNOTs versus the stored SDK
upper bound of seven and six iSWAPs versus eight. These comparisons establish
neither novelty nor global optimality.

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
Eight source snapshots and all earlier attempts are preserved. Each retained
phase result identifies its actual source run, even when a later attempt
failed. The raw full run and motif/phase supplements are described in the
[archive README](../../benchmarks/symbolic/README.md).

- [`manifest.json.gz`](manifest.json.gz) maps every displayed panel to the
  archive and source runs and stores exact expression aliases.
- [`summary.json`](summary.json) records counts, provenance validation and
  the synthesis test result; [`build-report.json`](build-report.json) records
  PDF hashes, page counts and repaired index links.
- [`open_cases.csv`](open_cases.csv) lists all 360 open phase modes: 229 exact
  and 131 projective. They remain work for the next solver iteration.
- [`sources.zip`](sources.zip) contains both LaTeX volumes and every referenced
  Qiskit circuit fragment, plus the manifest and build report.

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
