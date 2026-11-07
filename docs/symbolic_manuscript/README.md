# Revised theory manuscript and review package

[Manuscript](manuscript.pdf) · [Point-by-point response](referee-response.pdf) ·
[Editable LaTeX](manuscript.tex) · [Paper sources](sources.zip) ·
[Full review package](review-package.zip) · [Coverage and experiments](referee-revision.json) ·
[Exact replays](referee-verification.json) · [Release manifest](release-manifest.json)

Authors: Gregory Morse and Peter Rakyta. Affiliations and individual contribution
statements remain for the authors to supply before submission. This is a review
draft; it has not been submitted to a journal.

The paper now leads with a complete CNOT-cost classification for controlled
balanced two-qubit involutions and the invariant iff rewriting of the known
two-SQiSW region. The literal four-CNOT fermionic-SWAP circuit now uses six Ry
rotations. The one-qubit Clifford atlas and full grid are reference resources.

The revision executes BQSKit, tket, Quartz, and QUESO as well as Qiskit. It reports
BQSKit's matching fFredkin circuit, the atlas's C4X/C3X/SSWAP losses, failed
baselines, the negative peephole-suite result, and the chart-dominated ablation.
The numerical 18-CX/40-local/depth-25 application circuit beats the symbolic
18/42/31 result in locals and depth. No all-grid optimality or universal compiler
dominance is claimed.

`generated/optimality_ledger.csv` resolves every positive-entangler effective
cell to its matched proof sources. Zero-entangler cells, fixed-target minima,
uniform-family minima, unproved constructions, and obstruction types are
reported separately by physical width. Boolean flags alone do not count as
proof payloads.

For Prism, copy **`manuscript.tex` and the entire `generated/` directory** from
`sources.zip`, preserving their relative paths. Compile with
`tectonic manuscript.tex`. The response uses its own standalone
`referee-response.tex`. The full review ZIP also contains both complete atlas
PDFs and `atlas-sources.zip`, with `gates.tex`, `cliffords.tex`, `circuits/`, and
`programs/`.

The active scientific archive is
`benchmarks/symbolic/synthesis_reference_20261107_referee.json.gz`.
Historical archives remain immutable. The scientific view removes coordination
metadata while preserving exact constructions, source hashes, and proof scopes;
its fFredkin updates come from the separately archived 32-preset rerun.

Current reproduction scripts:

```bash
python3 -m pytest tests/symbolic -q
python3 examples/decomposition/symbolic_referee_components.py --output /tmp/components.json.gz
python3 examples/decomposition/symbolic_referee_peephole.py
python3 examples/decomposition/build_symbolic_referee_revision.py
```

The stronger SDK harness uses a separate Python 3.12 environment. Quartz/QUESO
require the recorded external builds; their raw records include commands,
revisions, rules, and the Quartz harness patch. UniversalQCompiler remains unrun
because a licensed Wolfram kernel was unavailable. Exact identity checks,
numerical regression replays, and analytical proof components are distinguished.
