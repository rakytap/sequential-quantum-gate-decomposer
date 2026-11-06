# Exact symbolic gate catalog and basis selection

The symbolic modules preserve the gate definitions and Clifford analysis from
PR #219 and provide a catalog for the symbolic OSR decomposition work. The
catalog itself needs SymPy, with no numerical optimizer or native extension.
The original `sympy_param` search additionally needs DEAP; both are in the
`symbolic` optional dependency group:

```bash
python -m pip install '.[symbolic]'
python examples/decomposition/symbolic_gate_catalog.py --gate ECR --gate PhasedFSimGate
python examples/decomposition/symbolic_gate_catalog.py --list-gates --output /tmp/symbolic-gates.json
```

The example can also be run from a source checkout with just SymPy installed,
before building SQUANDER. It imports the symbolic modules directly. In an
installed SQUANDER environment, use the normal package imports:

```python
import sympy as sp
from squander.decomposition.symbolic_gates import make_gate, compile_gates, controlled_gate
from squander.decomposition.symbolic_bases import get_basis

theta, phi = sp.symbols("theta phi", real=True)
target = make_gate("FSimGate", theta, phi)
basis = get_basis("ibm_ecr")
print(basis.gates)
print(target)
```

## Coverage and conventions

[`symbolic_gate_coverage.json`](../squander/decomposition/symbolic_gate_coverage.json)
maps every named unitary row in the
[Wikipedia gate list](https://en.wikipedia.org/wiki/List_of_quantum_logic_gates)
as retrieved on 6 November 2026 to its catalog entry. It also records the
[Qiskit library](https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_library)
class-name aliases. Google's additional parameterized families follow the
[Cirq Gate Zoo](https://quantumai.google/cirq/gatezoo).

The catalog includes all 46 named rows from that Wikipedia snapshot, the Qiskit
standard unitary gates, and common variable-width gate constructors:

| Family | Examples |
| --- | --- |
| Single-qubit gates | I, X/Y/Z, H, S/T and adjoints, square roots, Rx/Ry/Rz, P, R, U/U1/U2/U3, rotation vectors |
| Controlled gates | Open CX, CX/CY/CZ/CH/CS/CSdg/CSX/CT, controlled rotations, CU, Toffoli, Fredkin, C3X/C4X/C3SX, relative-phase Toffolis |
| Interaction and swap gates | Rxx/Ryy/Rzz/Rzx, XX±YY, SWAP, sqrt(SWAP), iSWAP and its roots, fermionic SWAP, DCX, ECR |
| Other named gates | Barenco, Berkeley B, canonical N(a,b,c), Dagwood Bumstead, Givens, Magic, CZS/CCZS, Deutsch, Margolus, Peres, fermionic Fredkin |
| Google/Cirq families | fSim/Sycamore, PhasedFSim, PhasedXPow, PhasedXZ, Pauli/H/CX/CZ/SWAP/iSWAP/CCX/CCZ powers, PhasedISWAPPow |
| Register constructors | Arbitrary controlled unitaries, MCX/MCPhase/MCMT, global MS, QFT and inverse, Walsh-Hadamard, Pauli words/rotations, diagonal/unitary matrices, wire permutations, binary linear functions, uniformly controlled rotations |

Parameterized families and generic controls, adjoints, and matrix inputs extend
the catalog beyond a finite list of names. Measurement, reset, classical
control flow, and compiler directives are not unitary targets for this API.

All matrices use **local wire 0 as the least-significant bit**, matching Qiskit.
Cirq's matrices are converted to this convention. Pauli string labels use
Qiskit's most-significant-wire-first notation. Gate lists are in circuit order:
`[("H", (0,)), ("CX", (0, 1))]` means H first, then CX with control 0.
Matrices returned by `make_gate` are immutable.

Angles such as theta and phi are in radians. Cirq `Pow` exponents and axis-phase
exponents are in half turns, as in Cirq. Their optional `global_shift` is
preserved. Canonical N(a,b,c) uses `exp(i(a XX + b YY + c ZZ))`, the convention
in the named-gate table, rather than a rotation with an implicit factor of -1/2.

Global phase is retained in every matrix. `make_gate("GP", theta)` returns a
scalar 1×1 global-phase operation; pass a register width as the second argument
to obtain `exp(i theta) I` for that register. Controlling a matrix retains its
phase, which then affects the controlled operation.

Names are case sensitive to preserve the original library. In particular,
legacy `SX` is a named Clifford, while `Sx`, `SXGate`, `sqrt(X)`'s alias `SqrtX`,
and lowercase `sx` identify the square-root-X gate. Use the SDK class names or
`gate_definition(name)` to avoid ambiguous shorthand.

## Complete single-qubit Clifford analysis

The original 192 gate variants and their 183 stored decompositions are retained.
They form 24 classes up to global phase, with eight distinct eighth-root phases
per class. The tests verify the stored decompositions with their actual phase.

```python
from squander.decomposition.symbolic_clifford import (
    analyze_single_qubit_cliffords, is_clifford, pauli_action,
)

analysis = analyze_single_qubit_cliffords()
assert len(analysis.representatives) == 24
assert len(analysis.elements) == 192
```

The analysis includes the multiplication table, inverse class indices, and a
phase table. If `k = analysis.multiplication_phases[i][j]`, then
`U_i @ U_j = exp(i*pi*k/4) U_result`. Matrix multiplication order is explicit;
it is the reverse of writing those two operations in circuit order.
`pauli_action` also checks multi-qubit matrices by exact conjugation of the
Pauli generators. It returns `None` when Clifford membership is not proved.
No numerical tolerance is used for this classification.

To export the full tables:

```bash
python examples/decomposition/symbolic_gate_catalog.py --cliffords --output /tmp/cliffords.json
```

## Bases with one entangler family

Each preset has one two-qubit primitive or parameterized family. This keeps its
entangler cost explicit and avoids silently mixing CNOTs, CZs, and hardware
gates in the same objective.

| Basis presets | Entangler | Single-qubit gates |
| --- | --- | --- |
| `cx_u`, `cx_rz_ry`, `cx_h_phase` | CX | U3; Rz/Ry; H/P |
| `ibm_cx`, `ibm_ecr`, `ibm_cz` | CX, ECR, or CZ respectively | Rz/Sx/X |
| `cz_u`, `cz_rz_rx` | CZ | U3 or Rz/Rx |
| `iswap_u`, `sqrt_iswap_u`, `sqrt_swap_u` | iSWAP, sqrt(iSWAP), or sqrt(SWAP) | U3 |
| `xx90_u`, `xx_rz_ry` | Fixed Rxx(pi/2) or Rxx(theta) | U3 or Rz/Ry |
| `zz_u`, `zx_u` | Rzz(theta) or Rzx(theta) | U3 |
| `google_cz`, `google_syc`, `google_fsim` | CZ, Sycamore, or fSim(theta,phi) | PhasedXZ |
| `google_sqrt_iswap` | sqrt(iSWAP) | PhasedXZ |
| `berkeley_b` | B | U3 |
| `ionq_aria`, `ionq_aria_full`, `ionq_forte` | MS, fully entangling MS, or ZZ | GPi/GPi2 |
| `quantinuum_zzmax`, `quantinuum_zzphase` | ZZMax or ZZPhase | PhasedX/Rz |
| `tket_tk2` | General TK2 compiler macro | PhasedX/Rz |
| `rigetti_cz`, `aqt_rxx`, `iqm_cz` | CZ, Rxx, or CZ respectively | Rz/Sx/X; R/Rz; R |
| `clifford_t_cx`, `clifford_t_cz` | CX or CZ | H/S/T/Tdg |
| `clifford_only` | CX | H/S; analysis only, not universal |

The continuous presets are universal up to global phase. Literal equality can
be impossible in the supplied basis; the synthesis contract below forbids a
physical GP gate and records this obstruction. Clifford+T is dense
universal, but an arbitrary symbolic angle generally has no exact finite
Clifford+T representation. These modes are separate in `BasisSet.universality`.
Custom `make_basis` calls validate widths and names; any claimed universality
of a custom combination is the caller's assertion.

`basis.entangler_count(sequence)` counts that basis's primitive operations and
rejects gates from other bases. It measures a cost; it does not prove that the
cost is minimal. An OSR lower bound must use the chosen entangler's actual
cut-crossing capacity, rather than assuming every entangler has CNOT's rank.

The matrix-based OSR engine below adds synthesis and lower-bound certificates.
It is not a certified optimal compiler for every catalog/basis pair. Minimum
counts require an exhaustive search in its declared scope or a matching
mathematical lower bound.

## Phase-aware symbolic constructions and archives

`symbolic_synthesis.synthesize` constructs circuits in the supplied basis and
proves their matrix identities with free real parameters. It orders candidates
by native entangler count, the unchanged original parameter-sensitive score,
then single-qubit gate count. These constructions are upper bounds; they do not
yet establish optimal synthesis for all 327 definitions in all 32 presets.
Analytical identities and native CNOT frames remain incumbent constructions.
The matrix-based engine also discovers native interaction networks and
resynthesizes two-wire sections of larger circuits. Attributed SDK definitions
can supply incumbents; their use is recorded separately from OSR discovery.

```python
from squander.decomposition.symbolic_synthesis import synthesize_phase_pair

a, b, c = sp.symbols("a b c", real=True)
result = synthesize_phase_pair("Canonical", (a, b, c), "cx_rz_ry")
```

For an opaque operator, use `synthesize_matrix`; no catalog gate name or source
circuit is passed to the search:

```python
from squander.decomposition.symbolic_gates import make_gate
from squander.decomposition.symbolic_synthesis import synthesize_matrix

theta = sp.Symbol("theta", real=True)
target = make_gate("CP", theta)
result = synthesize_matrix(target, "zz_u", phase_exact=False, seconds=5)
```

The engine realigns the operator across every nontrivial bipartition and
computes exact operator-Schmidt ranks and rank-one factors. It uses rational
functions of formal exponentials, without floating SVD thresholds or angle
sampling. A nonzero minor witnesses each rank lower bound; parameter-dependent
rank drops remain explicit conditions. Native pulse proposals are accepted
only after recursive OSR factorization recovers their residual local factors
and a complete circuit identity is proved.

For commuting two-qubit interactions, binary Pauli transport discovers native
two/three-pulse skeletons. Exact OSR recovers their local inverse. A verified
Bell-spectrum calculation and local Z gauges cover Cartan, fSim and PhasedFSim
families with free parameters. Their phase ratios propose angles; they do not
replace the final identity proof, including zero-amplitude specializations.
Contiguous two-wire source windows receive the same matrix-based treatment
before native rebasing, with each replacement's scalar propagated into the
enclosing circuit.

Examples checked against complete exact matrices include:

| Target | Native basis | Entanglers | Certificate scope |
| --- | --- | ---: | --- |
| Opaque three-wire CNOT chain | CX/U | 2 | Matching OSR connectivity bound |
| SWAP | iSWAP/U | 3 | CNOT invariant exclusion and mirror-gate theorem |
| SWAP | sqrt(SWAP)/U | 2 | OSR nonlocality and one-pulse invariant exclusion |
| iSWAP | sqrt(iSWAP)/U | 2 | OSR nonlocality and one-pulse invariant exclusion |
| Toffoli | iSWAP/U | 10 | Verified upper bound |
| Toffoli | tunable ZZ/U | 5 | Verified upper bound |
| Fredkin | iSWAP/U | 12 | Verified upper bound |
| Fredkin | tunable ZZ/U | 6 | Verified upper bound |

The finite two-qubit Clifford search closes over 20 local cosets and certifies
its distance with Clifford local frames. Its scope alone does not prove a
minimum with arbitrary continuous locals. The SWAP/iSWAP certificate uses the
[mirror-gate theorem](https://arxiv.org/abs/quant-ph/0407108) together with the
[CNOT invariant criterion](https://arxiv.org/abs/quant-ph/0308045) to exclude two
iSWAP uses under arbitrary locals. One-pulse exclusions compare exact
phase-normalized [local invariants](https://arxiv.org/abs/quant-ph/0002045).
These certificates concern entangler count; local-score optimality remains a
separate question.

`seconds` checks the OSR proposal-search budget between algebra operations.
An individual algebra operation, native Clifford setup or Cartan extraction
can exceed that interval. The archive runner adds a hard per-mode Unix alarm
around the entire synthesis and proof, retaining timeouts as open cases.
Its proposal budget is at most one third of the per-mode allowance, capped at
ten seconds. Search exhaustion is never recorded as mathematical impossibility.

Phase-exact circuits contain only the allowed physical primitives, with no GP
operation or free scalar repair. Projective circuits include the exact scalar
as automatic `virtual_gp` metadata at zero physical cost. Exact circuits are
preferred when no more expensive; a projective circuit is retained alongside
them only when strictly cheaper under the same objective.

This distinction can affect entanglers as well as local cost. The three-CNOT
Cartan identity uses a fixed scalar; the four-CNOT construction has literal
equality using SU(2) rotations. Every CNOT has determinant -1, so an exact SU(4)
target with SU(2)-only locals requires even CNOT parity. A projective minimum
therefore cannot be transferred unchanged to the phase-exact problem.

The compiler reports determinant-group, Clifford and number-field
obstructions without adding gates to the basis. A `generic_family`
impossibility record does not assert that every special parameter value is
impossible: its exact nonzero condition and witness are retained. Unsupported
constructions and timeouts remain open cases, never impossibility proofs.

For one-qubit Cliffords, a shortest-path certificate contains all attainable
phase-resolved states, exact edge circuits, costs, predecessor witnesses and
Bellman inequalities. H/S exhausts all 192 variants and proves their minimum
local score in that supplied alphabet. With continuous primitives or
non-Clifford intermediates, the certificate covers the explicitly enumerated
Clifford-preserving words only. Those more limited certificates do not claim
global local-cost optimality. The frozen group table is checked against the
independent exact Clifford analysis in the test suite.

Generate a bounded, checkpointed archive with:

```bash
python examples/decomposition/symbolic_synthesis_archive.py \
  --basis all --workers 4 --seconds-per-mode 10 \
  --output /tmp/symbolic-synthesis.json.gz
```

The archive enumerates 10,464 gate/basis cells, evaluates both phase policies,
and retains free real parameters, including optional `global_shift`. Register
constructors have explicitly listed integer widths and matrix/list fixtures;
they are not proofs for every register size or arbitrary matrix. Records
distinguish exact verification, proved impossibility, unresolved cases,
timeouts and compiler errors. Source hashes, package versions, setup
certificates and per-mode wall times are included. `--resume` accepts an
interrupted checkpoint only when its source hashes, selection and budget
match. A complete enumeration does not mean that every synthesis problem has
been solved.

The numerical Qiskit archive remains a separate sample baseline. Its permitted
SDK scalar correction differs from the new physical phase-exact policy. A
free-parameter identity must not be called an improvement over a matched
analytical competitor merely because it beats one sampled SDK output.

UniversalQCompiler comparisons are deferred at the user's request: no licensed
Wolfram kernel is available and none is being purchased. It has zero benchmark
runs and is excluded from competitor outcomes. `symbolic_uqc.py` and the
optional native `.wls` adapter preserve exact expression trees, native wire
order and rotation signs, but the Wolfram integration has not been executed.
Any future comparison should run its native analytical method once per target
family, verify the returned free-parameter identity, then separately label
rebasing into other compatible alphabets. It is not necessary to run its
native compiler independently for every gate/basis cell.

There are 32 presets. Hardware sources and unit conventions are recorded in
[`symbolic_reference_ledger.json`](../squander/decomposition/symbolic_reference_ledger.json).
The [IonQ definitions](https://docs.ionq.com/features/getting-started-with-native-gates)
use turns, with the first MS phase attached to local wire 0. `IonQMS` is distinct
from Qiskit's multi-qubit `MS` constructor; its hardware interaction range is
0 through 1/4. The mathematical matrix also accepts values outside that range.
[TKET definitions](https://docs.quantinuum.com/tket/api-docs/optype.html) use half
turns for PhasedX, ZZPhase, and TK2. Our existing Rz remains in radians.
One TK2 macro is not necessarily one physical entangling pulse. Device routing,
transport, pulse duration, and hardware angle precision are separate resources.

## Published bounds and reproducible reference circuits

The reference ledger distinguishes a proved minimum, a tight worst-case bound,
a published construction, and a family upper bound. Each record specifies the
basis, parameter domain, ancillary qubits, connectivity, and phase convention.
It is a curated research ledger; it does not claim an exhaustive literature
review for every gate/basis pair. An absent record is an open comparison.

With arbitrary local unitaries and no ancillas, the initial CNOT targets include:

| Target | CNOT count | Evidence |
| --- | --- | --- |
| CX/CY/CZ/CH/ECR | 1 | Exact local-invariant classification |
| DCX, iSWAP, sqrt(iSWAP), B | 2 | Exact local-invariant classification |
| SWAP, sqrt(SWAP), Sycamore | 3 | Exact local-invariant classification |
| Toffoli / CCZ | 6 | [Shende–Markov optimality proof](https://arxiv.org/abs/0803.2316) |
| Margolus | 3 | [Song–Klappenecker optimality proof](https://arxiv.org/abs/quant-ph/0312225) |
| Fredkin | at most 7 | [Cruz–Murta construction](https://arxiv.org/abs/2305.18128); no matching lower bound asserted |

The original reference sequences are now available directly as
`decomposition_sequence(name, *parameters)`, including the seven-CNOT Fredkin.
Margolus has an exact three-CNOT/four-Ry reference. Peres uses five CNOTs by
cancelling the final CNOT of the Toffoli reference. Its matching lower bound
follows because adding one CNOT recovers Toffoli. Local phase conjugation also
provides three-CNOT RCCX and six-CNOT CCY references. These arguments concern
entangler count; they do not certify the secondary local-gate score.

For any exact two-qubit input, the
[Shende–Bullock–Markov invariant test](https://arxiv.org/abs/quant-ph/0308045)
is implemented without numerical tolerance:

```python
from squander.decomposition.symbolic_minimality import cnot_cost_certificate

theta = sp.Symbol("theta", real=True)
family = make_gate("Rzz", theta)
conditional = cnot_cost_certificate(family)
assert conditional.minimum is None  # special angles have different costs
assert conditional.upper_bound == 2
assert cnot_cost_certificate(family.subs(theta, sp.pi/4)).minimum == 2
assert cnot_cost_certificate(family.subs(theta, sp.pi/2)).minimum == 1
```

`at_most` retains exact conditions for costs zero, one, and two; the remaining
case costs three. This certifies the interaction count, not a decomposition or
the parameter-sensitive tie breaker. The result is specific to CNOT with free
local unitaries and must not be transferred to an arbitrary native entangler.

The numerical reference harness covers every catalog definition at an explicit
binding, plus register-constructor fixtures. Eleven presets have Qiskit adapters:

```bash
python examples/decomposition/symbolic_reference_comparison.py \
  --basis all --output /tmp/symbolic-reference-comparison.json
```

The report records SDK versions, source hashes, seed, binding, operations,
global phase, verification error, and compiler failures. Other presets produce
`unsupported_sdk_basis` cells. The compiler tries named-gate structure, generic
matrix synthesis, and stored circuits, selecting the smallest verified result.
For example, it can retain the stored seven-CNOT Fredkin when generic SDK
synthesis is worse. An SDK scalar-phase omission is repaired in the delivered
circuit and recorded explicitly, then matrix equality is checked again.
Relative-phase errors are rejected. A nonzero exit status reports a compiler
failure or a regression against an applicable recorded literature bound.
Register constructors retain their integer widths and tensor-product structure.
The compiler assumes arbitrary input states and fixes both input and output
wire order. Qiskit's [virtual output permutations](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.2/qiskit.transpiler.TranspileLayout)
are therefore not free SWAPs: the adapter retains and charges their physical
implementation, recording when it uses the permutation-preserving level 1 path.

A numerical binding is an upper-bound witness for that binding, never a proof
for a whole symbolic family. SDK parameter-slot counts are diagnostic only;
floating angles are not converted to invented exact fractions to score them.
`verify_symbolic_candidate` checks the exact matrix identity first, rejects a
worse matched reference count, and computes the original `get_gatecost` for an
exact circuit, including a separately supplied global-phase operation. Both
legacy search methods remain available and their original gate score is
unchanged.

The search acceptance check now requires an exact symbolic identity. A small
sampled numerical error only triggers that check; it cannot certify a solution.
Both methods accept an explicit `allow_global_phase` policy. The optimistic
search also removes the old unconditional Fredkin seed, keeps the cheapest
goal when `find_min=True`, handles priority ties without ordering circuit
objects, and closes its worker pool on return. Pass `workers=1` to run it
without multiprocessing. Real single-qubit searches are regression-tested.
`find_min` concerns the searched ansatz and original gate score; it is not a
proof that every entangler structure or universal basis has been exhausted.

## Validation

```bash
python -m pytest tests/symbolic -q
```

Checks include exact identities in the free parameters, numerical unitarity for
every fixed-width definition, independent Qiskit/Cirq matrix comparisons with
global phase and wire order preserved, all stored Clifford decompositions, and
phase-aware group closure and associativity. SDK comparisons are optional when
the corresponding package is not installed. Importing the legacy symbolic
tool runs neither its old diagnostic nor its Fredkin search.
