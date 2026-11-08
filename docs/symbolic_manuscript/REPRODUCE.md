# Research artifacts for the involution-cost paper

The paper's analytical proofs stand independently of numerical fitting. Exact
matrix and polynomial checks support their algebraic components; this repository
is not a proof-assistant formalization. Numerical compiler runs are supporting
experiments and regression evidence.

| Result | Source or record | Check |
| --- | --- | --- |
| Literal five-unrestricted-gate reflection family | `squander/decomposition/symbolic_reflection_involution.py`, `five_arbitrary_matrices` | `test_five_unrestricted_reflection_word_with_angle_free` |
| Literal seven-CNOT reflection family, both signs | Same module, `seven_cnot_source`; `benchmarks/symbolic/reflection_seven_exact_20261108.json.gz` | `test_seven_cnot_reflection_word_is_literal_with_angle_free` |
| Native minimum-count certificates | `squander/decomposition/symbolic_monodromy.py`; `benchmarks/symbolic/native_monodromy_20261108.json.gz` and `native_monodromy_deep_20261108.json.gz` | `examples/decomposition/verify_symbolic_third_referee.py` |
| Quantum Littlewood–Richardson data | `symbolic_monodromy_qlr.json` beside the module; source revision, attribution, and license retained | Rational primal/Farkas reconstruction; exact Makhlin invariant equality |
| Numerical reflection discovery witnesses | `benchmarks/symbolic/reflection_seven_numeric_20261108.json.gz` | Independent Qiskit literal matrix replays in the same verifier |
| Current synthesis grid and proof scopes | `benchmarks/symbolic/synthesis_reference_20261107_referee.json.gz` | `generated/optimality_ledger.csv`; `referee-verification.json` separates 144 exact replays from 20,784 unchanged payload comparisons |
| Complete atlases | `docs/symbolic_atlas/gates.pdf`, `cliffords.pdf`, `sources.zip` | `build-report.json` binds each PDF to its full LaTeX source hash |

The seven-CNOT word has eighteen physical U3 gates, two varying polar angles,
and no GP gate. It proves an upper bound for every real angle. It does not prove
the conjectured interior minima of seven CNOTs or five unrestricted gates.

The native audit proves 112 distinct count records, including all 64 fixed-target
pairs in its SYC/√SWAP subset. Family samples apply only to the named substitution;
a matching full-family upper construction can separately establish a minimum
uniform budget. Thirty proof deadlines and eight uncertified coordinate
conversions remain inconclusive. Count/existence certificates do not supply the
missing shortest symbolic local frames or local-cost minima.

Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/symbolic -q
python examples/decomposition/verify_symbolic_third_referee.py
```

Pure-symbolic and independent SDK environments are pinned in the paper source
package. SDK experiments and numerical reflection fitting use the separately
recorded Python-3.12 environment. No licensed UniversalQCompiler run is claimed.
The existing Apache-2.0 repository license applies to the supplied code; upstream
notices, including the monodromy data license, remain alongside their sources.
