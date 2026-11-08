# Symbolic circuit atlases — revised review edition

[Gate atlas](gates.pdf) · [All-phase Clifford atlas](cliffords.pdf) ·
[Editable LaTeX and exact programs](sources.zip) ·
[Revised manuscript](../symbolic_manuscript/manuscript.pdf) ·
[Full review bundle](../symbolic_manuscript/review-package.zip)

The current archive is `synthesis_reference_20261107_referee.json.gz`.
The gate volume has 327 indexed target sheets with 32 presets per target;
its expression glossaries add continuation pages (see `build-report.json` for the current total). The separate Clifford
volume has 192 phase-resolved one-qubit matrices in 12 local alphabets, 24
circuits per sheet, and 98 pages including its index.

Controlled fermionic SWAP now has a literal four-CNOT, six-Ry construction.
BQSKit independently matches its interaction/local counts. The mathematical
contribution is the lower bound and controlled balanced-involution
classification, not an advantage over every compiler.

The complete grid has exact constructions or scoped obstructions, including
many one-qubit, phase-multiple, and determinant cases. This does not imply
all-grid entangler or local optimality. Twenty shared-expression scores remain
unevaluated. Finite-alphabet Clifford shortest paths do not establish
continuous-alphabet minima. See the manuscript's width-resolved tables and
`generated/optimality_ledger.csv` for the union of explicit matched proof
payloads; legacy Boolean flags alone are excluded from that union.

Stronger Qiskit, tket, BQSKit, Quartz, and QUESO comparisons retain atlas
losses and failed runs. The older broad SDK audit includes source-fed routes
and is not a claim of universal compiler dominance. The 27-to-3 tomography
figure comparison has been withdrawn.

Extract `sources.zip` to obtain `gates.tex`, `cliffords.tex`, `circuits/`, and
`programs/`. Keep relative paths when copying to Prism. Diagrams use Qiskit's
vector LaTeX drawer, fixed wire order, exact-expression glossaries, clickable
indices, and bookmarks. A compact alias is an exact linked definition, not
an additional free angle. Literal mode uses no free physical GP; projective
mode records a scalar. Both words are retained when projective is cheaper or
secondary costs are incomparable.

Historical preview PDFs and older archives remain references. The full review
ZIP includes the actual current gate and Clifford PDFs and their source ZIP.
[Verification](../symbolic_manuscript/referee-verification.json) separates
changed exact identity/certificate replays from unchanged payload comparisons.

The second review edition replaces the long SYC controlled-phase fallback
with an exact two-pulse chart and gives DCX a one-TK2 word. Remaining
SYC/√SWAP count gaps are annotated on their panels against separate
numerical witnesses. A family sample is explicitly labelled and proves
neither a free-parameter identity nor a minimum. Remaining TK2 gaps are
flagged against its known one-macro Cartan capacity. The numerical words,
errors, parameter substitutions, failures, and full gap ledger accompany the
[standalone supplement](../symbolic_manuscript/supplement.pdf).
