# Pre-Implementation Completion Checklist (Phase 3.1)

This document validates that `DETAILED_PLANNING_PHASE_3_1.md` and
`ADRs_PHASE_3_1.md` are detailed enough to start implementation, and maps open
items to the decisions that must close them.

Layer 1 documents this checklist validates (see spec-driven-development skill,
Layer 1 primary deliverables + Step 2 gap list):

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`

This file is the **implementation gate** (P31-G-1): it is part of the phase-3-1
working contract set alongside the two documents above and the publication
surfaces listed in `DETAILED_PLANNING_PHASE_3_1.md` Tier 4.

## Current Readiness Verdict

**Implementation-ready** for the frozen Phase 3.1 v1 contract.

All `P31-C-01 .. P31-C-09` rows are now closed below, with explicit trade-offs
and a staged default-pipeline migration policy.

Planning, paper, and implementation work may now proceed under Gate `P31-G-1`.

## Checklist Items (Closure Status)

| ID | Contract item | What must be decided | Primary record | Status |
|---|---|---|---|---|
| P31-C-01 | Primary fusion representation | **Closed by** `P31-ADR-004`: primary counted-claim representation is `kraus_bundle`; Liouville / superoperator form is allowed only as an internal apply/cache optimization after equivalence to the Kraus bundle is demonstrated on the same block. | `P31-ADR-004`; detailed planning §7, §10.6 | Closed |
| P31-C-02 | Support matrix | Gate names, noise channels, max fused support, partition patterns in vs out. **Closed by** `P31-ADR-007`: contiguous 1- and 2-qubit mixed gate+noise motifs built from `U3` / `CNOT` plus local single-qubit depolarizing / amplitude-damping / phase-damping channels on the same support, allowing multiple successive gates and multiple local noise insertions; each eligible block contains at least one noise operation. | `P31-ADR-007`; detailed planning §3, §7 | Closed |
| P31-C-03 | Numerical correctness thresholds | **Closed by** `P31-ADR-008`: Phase 3 exactness tolerances retained (`<= 1e-10` Frobenius / trace-validity family, `<= 1e-8` continuity energy) plus representation-level CPTP-invariant policy (`<= 1e-10` equality residuals, `>= -1e-12` positivity floors). | `P31-ADR-008`; detailed planning §10.1; Task 1 / Task 3 mini-specs | Closed |
| P31-C-04 | Mandatory correctness case set | **Closed by** `P31-ADR-009`: four stable v1 mixed-motif microcases, bounded continuity anchors at `q4` and `q6`, and every counted performance case from `P31-ADR-010`; inherited Phase 3 microcases may remain regression-only outside the counted claim surface. | `P31-ADR-009`; detailed planning §10.2 | Closed |
| P31-C-05 | External reference slice | **Closed by** `P31-ADR-011`: Qiskit Aer is required on all four `phase31_microcase_*` cases plus `phase2_xxz_hea_q4_continuity`; not required on `q6` continuity or counted 8/10-qubit performance cases. | `P31-ADR-011`; detailed planning §10.4 | Closed |
| P31-C-06 | Performance case set | **Closed by** `P31-ADR-010`: counted motif-dense families `phase31_pair_repeat` and `phase31_alternating_ladder` at `q ∈ {8,10}`, patterns `{periodic,dense}`, seeds `{20260318,20260319,20260320}`, plus sparse `layered_nearest_neighbor` control cases; required `break_even_table` / `justification_map`; positive-method threshold measured versus the Phase 3 fused baseline. | `P31-ADR-010`; detailed planning §10.3 | Closed |
| P31-C-07 | Mode naming and API surface | **Closed by** `P31-ADR-012`: planner `requested_mode` stays `partitioned_density`; counted runtime path uses `execute_partitioned_density_channel_native(...)` and runtime-path label `phase31_channel_native`; existing exception families and categories remain, with Phase 3.1 `first_unsupported_condition` vocabulary added. | `P31-ADR-012`; detailed planning §10.5 | Closed |
| P31-C-08 | Evidence bundle schema | **Closed by** `P31-ADR-013`: keep top-level `correctness_evidence` / `performance_evidence` trees, add Phase 3.1 fields and required slices (`channel_invariants`, `break_even_table`), and require version-bumped successor schemas when mandatory fields expand. | `P31-ADR-013`; detailed planning §10.6 | Closed |
| P31-C-09 | Host performance build and runtime policy | **Closed by** `P31-ADR-014`: counted v1 builds are scalar-only and must record `build_policy_id = "phase31_scalar_only_v1"`, `build_flavor = "scalar"`, `simd_enabled = false`, `tbb_enabled = false`, `thread_count = 1`, and `counted_claim_build = true`. Optional host acceleration remains a later non-counted branch under Task 6. | `P31-ADR-014`; detailed planning §10.7; `task-6/TASK_6_MINI_SPEC.md` | Closed |

## Closure Map

| Contract item | Decision that closes it | Status |
|---|---|---|
| Representation primary | `P31-ADR-004` Kraus-bundle primary representation with Liouville/PTM as non-primary views only | Closed |
| Support matrix | `P31-ADR-007` frozen v1 slice plus written unsupported tiers and final max-support wording | Closed |
| Thresholds | `P31-ADR-008` frozen numeric policy plus representation-level invariant policy | Closed |
| Correctness slice | `P31-ADR-009` frozen counted case list and IDs, including v1 mixed-motif cases | Closed |
| External reference slice | `P31-ADR-011` frozen Aer slice on four `phase31_microcase_*` cases plus `phase2_xxz_hea_q4_continuity` | Closed |
| Performance slice | `P31-ADR-010` frozen counted case list and IDs plus `break_even_table` / `justification_map` | Closed |
| Mode naming / API surface | `P31-ADR-012` planner-mode continuity plus `phase31_channel_native` runtime identity | Closed |
| Evidence packaging | `P31-ADR-013` schema and emission rules agreed | Closed |
| Host build / threading / SIMD | `P31-ADR-014` scalar-only counted-build policy for v1; Task 6 deferred to later non-counted branch | Closed |

## Closure Details (This Pass)

### P31-C-02: Support Matrix

Status: `Closed`

Closure decision:

- Phase 3.1 v1 keeps the Phase 3 primitive surface (`U3`, `CNOT`, local
  depolarizing, amplitude damping, phase damping / dephasing).
- The richer claim surface is the **fused eligibility class**, not a broader
  primitive gate/noise surface.
- Counted eligible blocks are contiguous 1- and 2-qubit mixed gate+noise motifs
  on the same support, with multiple successive gates and multiple local noise
  insertions allowed, and with **at least one noise operation** in each Phase
  3.1 fused block.
- Support beyond 2 qubits, spectator-qubit effects, correlated noise, and
  arbitrary unbounded CPTP fusion remain outside the v1 minimum claim.

Trade-off recorded:

- This is richer than Phase 3 exactly where the current fused runtime stops.
- It stays narrow enough to validate and benchmark as a positive methods paper.

### P31-C-03: Numerical Correctness Thresholds

Status: `Closed`

Closure decision:

- Internal/external exactness remains on the Phase 3 `<= 1e-10` Frobenius-norm
  scale.
- State validity remains `|Tr(rho) - 1| <= 1e-10` and
  `rho.is_valid(tol=1e-10)`.
- Continuity-anchor energy checks remain `<= 1e-8` where counted.
- Representation-level invariant policy is now explicit:
  - equality-style residuals `<= 1e-10`,
  - positivity-style eigenvalue floors `>= -1e-12`.

Trade-off recorded:

- Reusing Phase 3 exactness thresholds keeps the scientific story stable.
- The positivity floor is strict without pretending floating-point eigensolvers
  are exact-symbolic objects.

### P31-C-04: Mandatory Correctness Slice

Status: `Closed`

Closure decision:

- Counted microcases are:
  - `phase31_microcase_1q_u3_local_noise_chain`,
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`.
- Counted continuity anchors are:
  - `phase2_xxz_hea_q4_continuity`,
  - `phase2_xxz_hea_q6_continuity`.
- Every counted performance case is also part of the internal correctness
  matrix.
- Older Phase 3 microcases may remain regression checks, but they do not define
  the counted v1 Phase 3.1 paper surface unless promoted later.

Trade-off recorded:

- The counted slice now proves the richer mixed-motif claim directly.
- Continuity remains present, but does not dominate the methods narrative.

### P31-C-06: Performance Slice

Status: `Closed`

Closure decision:

- Counted positive-method families:
  - `phase31_pair_repeat`,
  - `phase31_alternating_ladder`.
- Counted control family:
  - `layered_nearest_neighbor`.
- The counted performance matrix is:
  - primary families at `q ∈ {8, 10}`, patterns `{periodic, dense}`, seeds
    `{20260318, 20260319, 20260320}`,
  - control family at `q ∈ {8, 10}`, pattern `sparse`, seed `20260318`.
- Positive methods closure requires at least one representative primary-family
  case to show `>= 1.2x` median wall-clock speedup or `>= 15%` peak-memory
  reduction **versus the Phase 3 fused baseline**, with no correctness loss.
- Every counted performance package must include a `break_even_table` or
  `justification_map`.

Trade-off recorded:

- The performance slice is now explicitly win-seeking rather than broad-neutral.
- Control cases keep the paper honest about regions where Phase 3 may already be
  sufficient.

### P31-C-01: Primary Fusion Representation

Status: `Closed`

Closure decision:

- The counted Phase 3.1 claim surface uses **Kraus bundles** as the primary
  exact representation for fused noisy blocks on bounded support.
- Ordered composition is defined at the Kraus-operator level.
- Completeness residuals and derived Choi positivity checks provide the required
  invariant suite under the frozen `P31-ADR-008` thresholds.
- Liouville / superoperator form is permitted only as an internal apply/cache
  optimization after equivalence to the Kraus bundle is demonstrated on the
  same block.

Trade-off recorded:

- Kraus bundles make the exact CPTP contract reviewer-visible and easy to state.
- A faster matrix apply backend remains available later without changing the
  public scientific claim.

### P31-C-05: External Reference Slice

Status: `Closed`

Closure decision:

- Qiskit Aer is required on:
  - all four `phase31_microcase_*` counted mixed-motif cases,
  - `phase2_xxz_hea_q4_continuity`.
- Qiskit Aer is not required on:
  - `phase2_xxz_hea_q6_continuity`,
  - counted 8/10-qubit performance cases,
  - optional later Task 6 build variants.

Trade-off recorded:

- The external slice now tests the new fused object directly.
- The package stays bounded instead of turning into a broad simulator bake-off.

### P31-C-07: Mode Naming And API Surface

Status: `Closed`

Closure decision:

- Planner and descriptor entry keep `requested_mode = "partitioned_density"`.
- The counted Phase 3.1 runtime path is named `phase31_channel_native`.
- The canonical convenience helper is
  `execute_partitioned_density_channel_native(...)`.
- Existing exception families and high-level runtime categories remain; Phase
  3.1 extends the `first_unsupported_condition` vocabulary rather than creating
  a new error framework.

Trade-off recorded:

- This preserves continuity with the shipped Phase 3 planner contract.
- The new runtime identity is still explicit enough for audits, bundles, and
  papers.

### P31-C-08: Evidence Packaging

Status: `Closed`

Closure decision:

- Phase 3.1 keeps the existing top-level evidence families:
  - `correctness_evidence`,
  - `performance_evidence`.
- Required counted-case metadata now includes:
  - `claim_surface_id = "phase31_bounded_mixed_motif_v1"`,
  - `runtime_class`,
  - `representation_primary = "kraus_bundle"`,
  - `fused_block_support_qbits`,
  - `contains_noise`,
  - `counted_phase31_case`.
- Required new slices:
  - `channel_invariants` under correctness,
  - `break_even_table` under performance.
- Schema versions must bump when mandatory Phase 3.1 fields are added.

Trade-off recorded:

- Phase 3.1 stays continuous with the mature Phase 3 evidence machinery.
- The richer exact-channel claim becomes machine-reviewable rather than
  free-text-only.

### P31-C-09: Host Build / Threading / SIMD

Status: `Closed`

Closure decision:

- Counted Phase 3.1 v1 builds are frozen to the scalar-only policy:
  - `build_policy_id = "phase31_scalar_only_v1"`,
  - `build_flavor = "scalar"`,
  - `simd_enabled = false`,
  - `tbb_enabled = false`,
  - `thread_count = 1`,
  - `counted_claim_build = true`.
- Counted bundles must record those fields directly in their metadata once the
  schema changes from `P31-C-08` are implemented.
- Task 6 remains a later non-counted branch and may not alter the counted v1
  paper surface without checklist amendment.

Trade-off recorded:

- The main Phase 3.1 claim stays about channel-native fusion rather than host
  acceleration.
- Optional kernel work remains possible later without destabilizing v1 evidence.

## Go / No-Go Rule

- **Go** for implementation when all P31-C-01 through P31-C-09 are `Closed` in
  the closure map above, with traceability to mini-specs.
- **No-Go** if representation or support matrix remains ambiguous, or if
  correctness baselines are undefined.

## Notes

- **P31-C-09 (v1):** Closed without requiring Task 6 work. Reopening SIMD/TBB
  counted evidence requires checklist amendment and `task-6/TASK_6_MINI_SPEC.md`
  scope.
- **Closed this pass:** `P31-C-01` to `P31-C-09` are now closed through
  `P31-ADR-004`, `P31-ADR-007` to `P31-ADR-015`, and the matching
  detailed-planning sections.
- **Default-pipeline migration policy:** Phase 3.1 may become the default
  evidence surface later, but historical Phase 3 must remain available through
  explicit legacy scripts/functions rather than flags.
- ADR-007 at program level remains `Deferred` for *code* until this checklist
  reaches **Go** and implementation begins; Phase 3.1 documentation **owns**
  the gate specification.
- Re-read `DETAILED_PLANNING_PHASE_3_1.md` after each closure pass for internal
  consistency.
