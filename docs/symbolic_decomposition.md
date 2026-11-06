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
| `berkeley_b` | B | U3 |
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
