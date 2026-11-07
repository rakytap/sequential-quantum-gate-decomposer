# Symbolic circuit atlases — manuscript edition, 7 November 2026

[Gate atlas](gates.pdf) · [All-phase Clifford atlas](cliffords.pdf) ·
[Gate preview](preview-gates.pdf) · [Clifford preview](preview-cliffords.pdf) ·
[Editable LaTeX and circuit/program sources](sources.zip)

[Referee manuscript](../symbolic_manuscript/manuscript.pdf) ·
[Manuscript sources](../symbolic_manuscript/sources.zip) ·
[Notable comparisons](../symbolic_notable_results/notable_results.pdf) ·
[Earlier proof ledger](../symbolic_optimality/proofs.pdf) ·
[SQiSW chart proof](../symbolic_optimality/sqrt_iswap_cartan.pdf)

The current source is `synthesis_reference_20261107_manuscript.json.gz`.
The older dated archive is preserved. All 20,928 requested phase modes have
an exact construction or a scoped obstruction. There are no retained grid
synthesis timeouts or errors. This does not imply global optimality or the
absence of timed-out proof attempts and held-out experiments.

Twenty three-SQiSW family circuits now attain the known minimum uniform
budget of three, replacing six-pulse incumbents. Full branch definitions
and linked scalar-program glossaries are included. Their secondary scores
remain unresolved. Unknown equal-entangler scores are incomparable and both
phase modes are retained. Zero excesses over the audited two-B/three-SQiSW
uniform caps remain.

Controlled fermionic SWAP now uses four CNOTs with a matching lower bound
against arbitrary Ising interactions; the manuscript gives the proof.
The independent native/exchange audits plus these new certificates cover
712 literal and 959 projective fixed-target minima, and 720 literal and
965 projective minimum uniform budgets. Other OSR, Clifford and secondary
floor certificates are separate and can overlap. A family budget is not a
minimum at every parameter specialization.

The broad matched SDK audit has 11 smaller entangler counts and 2,662 ties.
At equal entangler count, locals are smaller in 553 cases, equal in 2,108,
and larger in one. The exception is the documented C3X/Rzz/U3 score/count
tradeoff. Some earlier SDK inputs were source-fed. The manuscript supplies
independent Qiskit/Cirq comparisons for Sycamore and fFredkin. These results
do not establish superiority to every compiler route or publication.

The gate volume has 327 alphabetically indexed target sheets with 32
presets per target, A1 landscape. The separate Clifford volume has 192
exact-phase matrices in each of 12 distinct local alphabets, 24 circuits
per sheet, A3 landscape. Diagrams use Qiskit's vector LaTeX drawer, fixed
wire order, clickable indices and bookmarks. H/S and H/S/T/Tdg have global
shortest-word certificates; the latter allows non-Clifford intermediates.
Continuous alphabets retain narrower proof scopes.

Extract `sources.zip` to get `gates.tex`, `cliffords.tex`, `circuits/` and
`programs/`, plus build instructions. Copy these files and directories into
Prism with their relative paths intact. Compact diagram symbols resolve to
exact glossary definitions and machine-readable programs; they are not
additional free angles or approximations.

Literal mode permits neither physical GP nor free scalar correction.
Projective mode stores the scalar as metadata. Both circuits are shown if
projective is cheaper or their secondary comparison is unresolved. There
are 4,701 strictly cheaper projective circuits alongside literal witnesses,
plus ten cells with unknown comparisons and both modes retained. Catalog
aliases and fixed-width register fixtures are explicitly scoped.

[Release verification](../symbolic_manuscript/release-verification.json)
separates new exact replays from unchanged construction payloads and records
source-hash resolution for early runs made before committing their code.
The [release manifest](summary.json) records hashes, page counts and tests.
The manuscript edition supersedes earlier dated open-construction counts.
