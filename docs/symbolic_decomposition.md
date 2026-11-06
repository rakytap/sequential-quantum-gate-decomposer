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

The continuous presets are universal up to global phase. Include an explicit
global phase when exact matrix equality is required. Clifford+T is dense
universal, but an arbitrary symbolic angle generally has no exact finite
Clifford+T representation. These modes are separate in `BasisSet.universality`.
Custom `make_basis` calls validate widths and names; any claimed universality
of a custom combination is the caller's assertion.

`basis.entangler_count(sequence)` counts that basis's primitive operations and
rejects gates from other bases. It measures a cost; it does not prove that the
cost is minimal. An OSR lower bound must use the chosen entangler's actual
cut-crossing capacity, rather than assuming every entangler has CNOT's rank.

This catalog/basis milestone does **not** turn the legacy CNOT-guided heuristic
into a certified optimal compiler for all these bases. The next synthesis step
must retain symbolic equality checks and report minimum counts only when an
exhaustive search or a matching mathematical lower bound certifies them.

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
