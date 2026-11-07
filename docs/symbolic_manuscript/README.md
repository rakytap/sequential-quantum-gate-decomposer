# Symbolic synthesis: referee package

[Manuscript PDF](manuscript.pdf) · [Editable LaTeX](manuscript.tex) ·
[Complete source bundle](sources.zip) · [Build/data manifest](build-manifest.json) ·
[Release verification](release-verification.json)

This is an anonymous draft for scientific review, not a journal submission.
Names, affiliations and human contribution allocations need author approval
before submission; the draft already discloses AI use.

The main new mathematical result is a literal four-CNOT controlled fermionic
SWAP construction with a matching four-Ising lower bound. The proof permits
arbitrary local unitaries and interaction angles, no ancillas, all-to-all
coupling, and arbitrary scalar phase. It is an analytical proof with checked
algebraic components, not a proof-assistant formalization. Ordinary Fredkin
is a different operator.

The grid classifies all 327 × 32 cells under both phase policies: 7,702
literal constructions / 2,762 obstructions and 10,196 projective constructions
/ 268 obstructions. Family obstructions and minimum uniform budgets retain
their specialization caveats. Twenty new three-SQiSW circuit scores remain
unresolved; their counts, identities and uniform budgets are certified. The
paper does not claim unrestricted global local-score optimality or superiority
to every published circuit.

The bounded 36-job opaque-family experiment retains one full-method timeout
at 20 seconds: 11/12 full-method successes, 3/12 with native charts disabled,
12/12 source-lowering successes. The latter receives a source word and is
not the same input problem. The six-block four-qubit application uses 18
instead of 36 CNOTs, 42 instead of 66 locals, and depth 31 instead of 55
after the same SDK pass. Cold symbolic synthesis takes 5.628 seconds here;
it does not give a cold compilation speedup.

Independent Qiskit/Cirq inputs and raw operations are archived. Quartz,
QUESO and UniversalQCompiler are discussed as prior art but were not run.
The Sycamore 27-to-3 local comparison concerns one cited figure, not every
publication. Counts are not hardware latency or fidelity measurements.

For Prism, extract `sources.zip` and copy `manuscript.tex` plus the entire
`generated/` directory, preserving relative paths. Those files suffice to
compile the paper using `tectonic manuscript.tex`. Other bundle files support
reproduction and review. The [atlas bundle](../symbolic_atlas/sources.zip)
separately contains `gates.tex`, `cliffords.tex`, `circuits/` and `programs/`.
The dated [older proof ledgers](../symbolic_optimality) remain supporting
artifacts; this manuscript supersedes their old construction-gap status.

```bash
OPENBLAS_NUM_THREADS=1 python3 examples/decomposition/symbolic_manuscript_experiments.py \
  --output /tmp/manuscript-experiments.json.gz --workers 4 --seconds 20 --bindings 1000
python3 examples/decomposition/build_symbolic_manuscript.py
OPENBLAS_NUM_THREADS=1 python3 examples/decomposition/verify_symbolic_manuscript_release.py
python3 -m pytest tests/symbolic -q
```

The authoritative new archive is
`benchmarks/symbolic/synthesis_reference_20261107_manuscript.json.gz`.
The older `synthesis_reference_20261106.json.gz` remains immutable.
Some early raw Git revisions precede uncommitted source edits. The verifier
resolves the recorded content hashes from Git history and lists each
exception; it does not pretend those bytes were in the recorded old HEAD.
Unchanged payload comparisons and newly replayed identities are counted
separately in the release report.
