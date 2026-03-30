# Phase 3.1 ADRs

This document records architecture and scope decisions that apply specifically
to Phase 3.1 (channel-native / superoperator fusion follow-on).

It refines, but does not override, the upstream planning decisions in:

- `docs/density_matrix_project/planning/ADRs.md`
- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`
- `docs/density_matrix_project/phases/phase-3/ADRs_PHASE_3.md`

If this document appears to conflict with upstream planning, resolve in favor of
upstream unless this document explicitly tightens scope for Phase 3.1 only.

## Status Legend

- `Accepted`: part of the Phase 3.1 contract once implementation starts.
- `Proposed`: planning default pending checklist closure.
- `Rejected for Phase 3.1`: explicitly excluded.

## Phase 3.1 Decision Summary

| ADR | Title | Status |
|---|---|---|
| P31-ADR-001 | Use spec-driven Phase 3.1 documentation as the implementation contract | Accepted |
| P31-ADR-002 | Keep Phase 3.1 additive relative to Phase 3; do not rewrite Paper 2 claims | Accepted |
| P31-ADR-003 | Require exact agreement with sequential `NoisyCircuit` for any channel-native claim | Accepted |
| P31-ADR-004 | Choose a primary exact superoperator/channel representation for the baseline fusion claim | Accepted |
| P31-ADR-005 | Treat performance claims as comparative and evidence-bound versus sequential and Phase 3 fused baselines | Accepted |
| P31-ADR-006 | Target Phase 3.1 as a bounded positive-methods contribution first | Accepted |
| P31-ADR-007 | Freeze a provisional v1 scientific slice around bounded mixed noisy motifs | Accepted |
| P31-ADR-008 | Freeze Phase 3.1 numeric exactness and CPTP-invariant thresholds | Accepted |
| P31-ADR-009 | Freeze the mandatory correctness slice around v1 mixed-motif cases plus bounded integration anchors | Accepted |
| P31-ADR-010 | Freeze the performance slice and positive-method success rule around motif-dense workloads | Accepted |
| P31-ADR-011 | Freeze the external reference slice around Phase 3.1 microcases plus a bounded continuity subset | Accepted |
| P31-ADR-012 | Keep `partitioned_density` as the planner mode and add a distinct Phase 3.1 runtime/API surface | Accepted |
| P31-ADR-013 | Extend existing evidence families with Phase 3.1-specific fields and required slices | Accepted |
| P31-ADR-014 | Freeze the scalar-only counted-build policy for Phase 3.1 v1 | Accepted |
| P31-ADR-015 | Promote Phase 3.1 to the default evidence pipelines after stabilization while preserving Phase 3 as explicit legacy scripts/functions | Accepted |

## P31-ADR-001: Use Spec-Driven Phase 3.1 Documentation As The Implementation Contract

### Status

`Accepted`

### Context

Phase 3.1 is more invasive than unitary-island fusion. Without a documentation
contract, implementation risks unclear semantics, silent fallbacks, or
publication claims that exceed validation.

### Decision

Phase 3.1 work must be guided by explicit documentation-first contracts in
`docs/density_matrix_project/phases/phase-3-1/`, including this ADR set,
`DETAILED_PLANNING_PHASE_3_1.md`,
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` (Gate P31-G-1), and task mini-specs
created before each implementation task.

### Rationale

- Channel-native fusion touches CPTP semantics and numerical linear algebra at
  the core of trust in exact noisy simulation.
- Phase 3 established a successful pattern: contract, evidence, then claims.

### Consequences

- Unsupported channel-native requests must be explicit outcomes.
- Evidence bundle design is planned alongside architecture.

### Rejected Alternatives

- Prototype-first without frozen representation and support matrix.

### Upstream Alignment and Traceability

- Program spec-driven workflow; `P3-ADR-001` pattern.

## P31-ADR-002: Keep Phase 3.1 Additive Relative To Phase 3

### Status

`Accepted`

### Context

Phase 3 is closed with a bounded Paper 2 claim. Phase 3.1 explores a deferred
branch described in `P3-ADR-010` and PLANNING.md §5.1.

### Decision

Phase 3.1 must not retroactively redefine what Phase 3 delivered. Phase 3.1
claims are **additional** and must be traceable to Phase 3.1 evidence only.

### Rationale

- Preserves auditability of published Phase 3 materials.
- Avoids mixed messages for reviewers.

### Consequences

- Documentation and benchmarks should label Phase 3.1 modes distinctly.
- Comparison to Phase 3 fused baseline is part of the scientific story, not a
  replacement for Phase 3 closure artifacts.

### Rejected Alternatives

- Merge Phase 3.1 results into Phase 3 claim language without versioning.

### Upstream Alignment and Traceability

- `P3-ADR-002`, `P3-ADR-010`; PLANNING.md Phase 3 exit criteria.

## P31-ADR-003: Require Exact Agreement With Sequential Density Baseline

### Status

`Accepted`

### Context

The PhD program treats exact dense density matrices as the reference engine for
early and middle phases.

### Decision

Any execution mode advertised as part of Phase 3.1’s supported channel-native
surface must match the sequential `NoisyCircuit` density result within frozen
numerical thresholds on the mandatory correctness slice.

### Rationale

- Exactness is the differentiator versus approximate branches.
- Matches Phase 3 validation culture (`P3-ADR-008`).

### Consequences

- External reference checks (e.g. Qiskit Aer) remain part of the evidence story
  where Phase 3 did the same.
- Performance wins never justify semantic slack.

### Rejected Alternatives

- Statistical or sampling-based correctness for the core Phase 3.1 claim.

### Upstream Alignment and Traceability

- `ADR-005`, `P3-ADR-008`.

## P31-ADR-004: Choose A Primary Exact Representation For Baseline Fusion

### Status

`Accepted`

### Context

Channel-native fusion can be expressed as Kraus sums, Pauli transfer matrices,
Liouville superoperators, or other equivalent linear-algebraic forms. The
codebase needs one **primary** representation for the baseline implementation
and paper narrative to avoid ambiguity.

### Decision

Phase 3.1 v1 uses **Kraus bundles on bounded local support** as the **primary
representation** for the counted methods claim.

Specifically:

- every counted fused block is defined first as a Kraus bundle on 1- or 2-qubit
  support,
- unitary sub-steps are represented as single-operator Kraus channels inside the
  same composition rule,
- composition is defined by ordered Kraus-operator multiplication across the
  contiguous mixed motif,
- trace preservation is checked through the completeness residual
  `||Σ_i K_i† K_i - I||_F`,
- positivity-aware validation is checked through a derived Choi-matrix test on
  the mandatory microcases under the frozen `P31-ADR-008` thresholds.

Alternative encodings are allowed only as **internal optimizations**:

- Liouville / superoperator matrices may be cached or used as the apply backend
  once they are shown equivalent to the primary Kraus bundle on the same block,
- PTM-style views may be used for audit or analysis but are not the baseline
  public claim surface.

### Rationale

- Reviewers and maintainers need a single anchor for “what is fused.”
- Multiple public representations without a primary invite inconsistent
  thresholds and bugs.

### Consequences

- `P31-C-01` is closed by this ADR once planning and checklist cite the decision.
- Task 1 and Task 3 must require invariant-level acceptance evidence in
  addition to end-state agreement checks.
- The paper can speak about one exact fused object without ambiguity while still
  permitting a faster internal apply form later.

### Rejected Alternatives

- “Any equivalent representation” without a documented primary.

### Upstream Alignment and Traceability

- `ADR-007` context (IR-first / superoperator discussion).
- `P31-ADR-008`.

## P31-ADR-005: Evidence-Bound Comparative Performance Claims

### Status

`Accepted`

### Context

Phase 3 closed performance through diagnosis-grounded evidence when speedups
were absent. Phase 3.1 should not repeat over-claiming.

### Decision

Phase 3.1 performance statements must compare against:

1. sequential `NoisyCircuit` density execution, and
2. the Phase 3 partitioned path with shipped unitary-island fusion (where
   applicable),

and must allow diagnosis-only closure if positive thresholds are not met.

### Rationale

- Scientific honesty and continuity with Phase 3 narrative.
- Answers the Side Paper A question: “when does channel-native fusion matter?”

### Consequences

- Performance bundles name baselines and case IDs explicitly.

### Rejected Alternatives

- Claiming universal speedup from channel-native fusion without comparative
  evidence.

### Upstream Alignment and Traceability

- Phase 3 performance rule; PLANNING.md Phase 3 evidence findings.

## P31-ADR-006: Target Phase 3.1 As A Bounded Positive-Methods Contribution First

### Status

`Accepted`

### Context

Phase 3.1 can be framed either as a broad decision study or as a narrower
positive-methods contribution. The strongest v1 paper opportunity is a bounded
methods result that extends Phase 3's exact noisy runtime beyond unitary-island
fusion on a scientifically chosen slice, while still allowing diagnosis-grounded
negative closure if that slice does not produce a defensible win.

### Decision

Phase 3.1 v1 should optimize for a **positive methods claim first**:

- define a narrow support slice where exact channel-native fusion has a
  realistic chance to outperform the shipped Phase 3 fused baseline,
- preserve a **strict** motif-proof runtime for fully eligible mixed-motif
  cases,
- add an explicit **hybrid** whole-workload runtime for continuity and
  structured cases where some partitions are Phase-3-supported but not
  Phase-3.1-eligible,
- design correctness and performance bundles around that strict-plus-hybrid
  split,
- and retain the broader decision-study framing as a **fallback closure mode**
  rather than the primary publication target.

### Rationale

- A positive methods paper is more valuable if the support slice is narrow and
  the win is clear.
- The fallback path preserves scientific honesty if the win does not
  materialize.

### Consequences

- Support-matrix, correctness-slice, and performance-slice decisions should be
  **win-seeking but honest**, not maximally broad.
- Task 4 still needs a decision artifact (`break_even_table` /
  `justification_map`) so the paper answers when the richer method is actually
  warranted.

### Rejected Alternatives

- Keep Phase 3.1 primarily as a broad neutral decision study from the start.
- Broaden the gate/noise surface aggressively before a bounded positive-methods
  slice exists.

### Upstream Alignment and Traceability

- `PUBLICATIONS.md` Side Paper A.
- `P31-ADR-005`.

## P31-ADR-007: Freeze A Provisional V1 Scientific Slice Around Bounded Mixed Noisy Motifs

### Status

`Accepted`

### Context

Phase 3 already freezes the primitive support surface (`U3`, `CNOT`, and local
single-qubit depolarizing / amplitude-damping / phase-damping channels) but
defers channel-native noisy-block fusion. For a positive Phase 3.1 methods
paper, the scientifically meaningful expansion is not immediate broadening of
the primitive gate/noise families; it is widening the **fused eligibility
class** to bounded mixed gate+noise motifs that Phase 3 cannot absorb.

### Decision

Phase 3.1 v1 adopts the following **provisional scientific slice** as the
planning default:

- primitive gates remain `U3` and `CNOT`,
- primitive noise remains local single-qubit depolarizing, local amplitude
  damping, and local phase damping / dephasing,
- the richer eligibility class is **contiguous mixed gate+noise motifs** on a
  total support of **1 or 2 qubits**,
- each eligible fused block must contain **at least one noise operation**;
  purely unitary islands remain part of the shipped Phase 3 fused path rather
  than part of the new Phase 3.1 claim surface,
- multiple successive gates and multiple local noise insertions inside one
  fused block are allowed when all operations remain on those same qubits,
- spectator-qubit effects, correlated multi-qubit noise, support beyond 2
  qubits, and arbitrary unbounded CPTP fusion remain outside the v1 minimum
  claim.

This is the default slice for closing `P31-C-02`, `P31-C-04`, and `P31-C-06`.

### Rationale

- It extends Phase 3 exactly where the shipped fused runtime currently stops:
  noise boundaries breaking otherwise local fused structure.
- It is narrow enough to validate exactly and broad enough to support a real
  methods claim.

### Consequences

- Task 1 must define a representation and invariant suite for this bounded
  mixed-motif class.
- Task 3 and Task 4 must include mandatory cases that exercise multi-noise,
  same-support motifs rather than only primitive microcases.

### Rejected Alternatives

- Stay strictly on the shipped Phase 3 fused eligibility class.
- Broaden the primitive gate/noise surface first and leave fused eligibility
  vague.
- Make 3-qubit motifs, correlated noise, or arbitrary CPTP regions part of the
  v1 minimum claim.

### Upstream Alignment and Traceability

- `P3-ADR-007`.
- `P3-ADR-010`.
- `PUBLICATIONS.md` Side Paper A.

## P31-ADR-008: Freeze Phase 3.1 Numeric Exactness And CPTP-Invariant Thresholds

### Status

`Accepted`

### Context

Phase 3.1 needs thresholds that are strict enough to support an exact methods
claim while remaining compatible with the Phase 3 numeric culture. The richer
channel-native claim also needs representation-level invariant thresholds, not
only end-state regression thresholds.

### Decision

Phase 3.1 freezes the following numeric policy for the v1 counted claim surface:

- internal semantic-preservation exactness:
  - maximum Frobenius-norm density difference `<= 1e-10` between Phase 3.1 and
    sequential `NoisyCircuit` execution on every counted supported case,
- external micro-validation exactness:
  - maximum Frobenius-norm density difference `<= 1e-10` against Qiskit Aer on
    the external slice once `P31-C-05` is closed,
- state validity:
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)` on recorded required outputs,
- continuity-anchor observable agreement (only on counted continuity cases):
  - maximum absolute energy error `<= 1e-8`,
- representation-level channel invariants:
  - equality-style invariants (for example Kraus completeness or explicit
    trace-preservation residuals) must have residual `<= 1e-10`,
  - positivity-style invariants checked through eigenvalue floors (for example
    Choi positivity) must satisfy minimum eigenvalue `>= -1e-12`.

### Rationale

- Reusing Phase 3 exactness tolerances keeps the scientific story consistent.
- A small numerical floor for positivity checks reflects floating-point noise
  without weakening the exact-channel claim.

### Consequences

- `P31-C-03` is closed by this policy once planning and checklist cite it.
- Task 1 and Task 3 must emit invariant-aware evidence, not only final-state
  agreement.

### Rejected Alternatives

- Loosen exactness thresholds because the fused object is “more complicated.”
- Require exact symbolic positivity with no numerical floor.

### Upstream Alignment and Traceability

- `P3-ADR-008`.
- `P31-ADR-004`.

## P31-ADR-009: Freeze The Mandatory Correctness Slice Around V1 Mixed-Motif Cases Plus Bounded Integration Anchors

### Status

`Accepted`

### Context

Phase 3.1 needs a counted correctness slice that proves the new positive-method
claim surface, rather than only replaying the full Phase 3 matrix. At the same
time, the slice must stay connected to the delivered workflow and runtime
contracts so the new method does not become a disconnected micro-benchmark.

### Decision

The counted Phase 3.1 correctness slice is frozen as:

- **v1 mixed-motif microcases** (all counted, all internally exact):
  - `phase31_microcase_1q_u3_local_noise_chain`
  - `phase31_microcase_2q_cnot_local_noise_pair`
  - `phase31_microcase_2q_multi_noise_entangler_chain`
  - `phase31_microcase_2q_dense_same_support_motif`
- **bounded continuity anchors** (counted integration cases):
  - `phase2_xxz_hea_q4_continuity`
  - `phase2_xxz_hea_q6_continuity`
- **performance-carry-forward rule**:
  - every case counted in the frozen Phase 3.1 performance slice
    (`P31-ADR-010`) is also part of the mandatory internal correctness matrix.

Interpretation rules:

- the counted slice is the one used for the positive Phase 3.1 methods claim,
- the four counted mixed-motif microcases run through the **strict**
  `phase31_channel_native` path,
- the counted continuity anchors and counted performance carry-forward rows run
  through the explicit **hybrid** `phase31_channel_native_hybrid` path,
- inherited Phase 3 microcases outside this slice may remain as regression or
  unsupported-boundary checks, but they are not part of the counted v1 methods
  surface unless explicitly promoted later.

### Rationale

- The new mixed-motif microcases directly exercise the richer fused eligibility
  class rather than only the Phase 3 primitive surface.
- The 4- and 6-qubit continuity anchors keep the new path scientifically tied to
  the existing exact noisy workflow without letting continuity dominate the
  methods paper.

### Consequences

- `P31-C-04` is closed by this counted-slice rule once planning and checklist
  cite these IDs.
- Task 3 must preserve these exact case IDs in artifacts and paper traceability.

### Rejected Alternatives

- Count only inherited Phase 3 microcases and hope they expose the richer class.
- Reuse the full Phase 3 correctness matrix as the counted v1 paper surface.

### Upstream Alignment and Traceability

- `P31-ADR-007`.
- `P3-ADR-008`.

## P31-ADR-010: Freeze The Performance Slice And Positive-Method Success Rule Around Motif-Dense Workloads

### Status

`Accepted`

### Context

A positive Phase 3.1 methods paper should optimize for cases where channel-native
fusion has a realistic chance to outperform the shipped Phase 3 fused baseline.
That means the performance slice should be narrower and more motif-focused than
the neutral Phase 3 benchmark matrix, while still recording control cases that
show where the older method is already sufficient.

### Decision

The counted Phase 3.1 performance slice is frozen as:

- **primary positive-method families** (motif-dense, counted):
  - family `phase31_pair_repeat`
  - family `phase31_alternating_ladder`
- **control family** (counted, inherited-style baseline sanity):
  - family `layered_nearest_neighbor`
- qubit counts:
  - `8`, `10`
- noise-pattern policy:
  - primary families use `periodic` and `dense`,
  - control family uses `sparse`,
- seed policy:
  - primary families use seeds `20260318`, `20260319`, `20260320`,
  - control family uses seed `20260318`.

This yields:

- primary positive-method counted cases:
  - `2 families × 2 qbit counts × 2 noise patterns × 3 seeds = 24 cases`,
- control counted cases:
  - `1 family × 2 qbit counts × 1 noise pattern × 1 seed = 2 cases`.

Positive-method threshold rule:

- at least one representative counted `8`- or `10`-qubit case from the
  **primary** families must show either:
  - median wall-clock speedup `>= 1.2x` versus the Phase 3 partitioned+fused
    baseline, or
  - peak-memory reduction `>= 15%` versus the Phase 3 partitioned+fused
    baseline,
  - with no correctness loss under `P31-ADR-008`.

All counted performance cases must also record:

- the sequential `NoisyCircuit` baseline,
- the Phase 3 partitioned+fused baseline,
- the Phase 3.1 **hybrid** channel-native baseline,
- and a `break_even_table` / `justification_map` classifying cases as:
  - `phase3_sufficient`,
  - `phase31_justified`,
  - or `phase31_not_justified_yet`.

Execution interpretation for the counted performance slice:

- the counted whole-workload performance families are **not** carried by the
  strict `phase31_channel_native` path, because many counted partitions remain
  Phase-3-supported but Phase-3.1-ineligible,
- the counted Phase 3.1 performance baseline is therefore the explicit hybrid
  runtime `phase31_channel_native_hybrid`, which routes:
  - eligible partitions through the Phase 3.1 channel-native executor,
  - Phase-3-supported but Phase-3.1-ineligible partitions through the shipped
    Phase 3 exact path with route attribution.

### Rationale

- The primary families are chosen to maximize exposure of the richer same-support
  mixed-motif class.
- The control family keeps the paper honest about cases where Phase 3 may
  already be enough.

### Consequences

- `P31-C-06` is closed by this counted-slice and threshold rule once planning
  and checklist cite it.
- The main positive Phase 3.1 paper claim is explicitly **against the shipped
  Phase 3 fused baseline**, not only against sequential density execution.

### Rejected Alternatives

- Reuse the entire Phase 3 structured-family matrix as the counted Phase 3.1
  performance claim.
- Define success only versus the sequential baseline and not versus Phase 3.

### Upstream Alignment and Traceability

- `P31-ADR-005`.
- `P31-ADR-006`.
- `P31-ADR-007`.

## P31-ADR-011: Freeze The External Reference Slice Around Phase 3.1 Microcases Plus A Bounded Continuity Subset

### Status

`Accepted`

### Context

Phase 3.1 needs an external exact reference slice strong enough to support the
new methods claim without turning the evidence package into a broad simulator
bake-off. The new claim surface is defined by small mixed-motif microcases, so
the external slice should emphasize those directly.

### Decision

Phase 3.1 uses **Qiskit Aer density-matrix simulation** as the required external
reference on the following counted slice:

- all four counted Phase 3.1 mixed-motif microcases:
  - `phase31_microcase_1q_u3_local_noise_chain`,
  - `phase31_microcase_2q_cnot_local_noise_pair`,
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`,
- one bounded continuity anchor:
  - `phase2_xxz_hea_q4_continuity`.

Execution interpretation:

- the four `phase31_microcase_*` cases use the strict
  `phase31_channel_native` path,
- `phase2_xxz_hea_q4_continuity` uses the explicit hybrid
  `phase31_channel_native_hybrid` path.

The following are **not** part of the required external slice:

- `phase2_xxz_hea_q6_continuity`,
- counted 8- and 10-qubit performance cases,
- any optional later-branch host-acceleration variants.

### Rationale

- The external slice now tests the actual new claim-bearing motifs directly.
- The single small continuity anchor keeps Phase 3.1 externally connected to the
  delivered workflow without bloating the package.

### Consequences

- `P31-C-05` is closed by this slice once planning and checklist cite it.
- Task 3 must record Aer version metadata and preserve stable case IDs on these
  five required external-reference cases.

### Rejected Alternatives

- Reuse the whole Phase 3 external slice as the counted Phase 3.1 external
  package.
- Drop the external reference because the internal oracle already exists.

### Upstream Alignment and Traceability

- `P3-ADR-008`.
- `P31-ADR-009`.

## P31-ADR-012: Keep `partitioned_density` As The Planner Mode And Add A Distinct Phase 3.1 Runtime/API Surface

### Status

`Accepted`

### Context

The Phase 3 planner and descriptor contract already hinges on
`PARTITIONED_DENSITY_MODE` (`"partitioned_density"`). Reopening that top-level
mode would blur the continuity of the contract. Phase 3.1 needs a distinct
execution identity, but it does not need a different planner-entry mode.

### Decision

Phase 3.1 freezes the following API and naming policy:

- planner and descriptor preflight continue to use
  **`PARTITIONED_DENSITY_MODE`** (`"partitioned_density"`),
- the Phase 3 unitary-island baseline remains identified by the existing helper
  **`execute_partitioned_density_fused(...)`** and its runtime labels,
- the strict motif-proof Phase 3.1 path is exposed as:
  - convenience helper: **`execute_partitioned_density_channel_native(...)`**,
  - runtime-path label: **`phase31_channel_native`**,
- the hybrid whole-workload Phase 3.1 path is exposed as:
  - convenience helper:
    **`execute_partitioned_density_channel_native_hybrid(...)`**,
  - runtime-path label: **`phase31_channel_native_hybrid`**.

Error taxonomy policy:

- keep the existing exception families:
  - `NoisyPlannerValidationError`,
  - `NoisyRuntimeValidationError`,
- keep the existing high-level runtime categories:
  - `runtime_request`,
  - `descriptor_to_runtime_mismatch`,
  - `unsupported_runtime_operation`,
  - `unsupported_runtime_execution`,
- add Phase 3.1-specific `first_unsupported_condition` vocabulary where needed,
  including:
  - `channel_native_support_surface`,
  - `channel_native_noise_presence`,
  - `channel_native_qubit_span`,
  - `channel_native_representation`,
  - `channel_native_invariant_failure`,
  - `channel_native_runtime_execution`.

Hybrid-route policy:

- strict `phase31_channel_native` keeps the current **no-fallback** rule:
  any Phase-3.1-ineligible partition is a hard error,
- hybrid `phase31_channel_native_hybrid` may route a partition to the shipped
  Phase 3 exact path only when the partition is:
  - Phase-3-supported,
  - Phase-3.1-ineligible for a documented support-surface reason,
  - and not failing due to representation, invariant, or runtime-execution
    errors,
- representation mismatch, invariant failure, or channel-native execution
  failure on a partition that should have executed channel-natively remain hard
  errors, not reroute conditions.

Frozen hybrid route reasons:

- `eligible_channel_native_motif`,
- `pure_unitary_partition`,
- `channel_native_noise_presence`,
- `channel_native_qubit_span`,
- `channel_native_support_surface`.

### Rationale

- This preserves one planner contract while still making strict and hybrid Phase
  3.1 evidence distinguishable from the shipped Phase 3 fused baseline.
- Reusing the existing structured error schema keeps audits and bundles aligned
  with current runtime behavior.

### Consequences

- `P31-C-07` is closed by this naming and error-policy decision once planning
  and checklist cite it.
- A later `API_REFERENCE_PHASE_3_1.md` should document the strict and hybrid
  helper names, runtime-path labels, hybrid route reasons, and Phase 3.1
  `first_unsupported_condition` vocabulary.

### Rejected Alternatives

- Add a second planner-entry `requested_mode` string for Phase 3.1.
- Reuse the generic Phase 3 fused helper name and distinguish the paths only in
  prose.

### Upstream Alignment and Traceability

- `P31-ADR-001`.
- `P31-ADR-002`.

## P31-ADR-013: Extend Existing Evidence Families With Phase 3.1-Specific Fields And Required Slices

### Status

`Accepted`

### Context

Phase 3 already has mature `correctness_evidence` and `performance_evidence`
artifact trees. Phase 3.1 should build on those surfaces rather than inventing a
third top-level evidence family, but the richer claim needs extra fields and new
required slices.

### Decision

Phase 3.1 keeps the existing top-level evidence families:

- `benchmarks/density_matrix/artifacts/correctness_evidence/`
- `benchmarks/density_matrix/artifacts/performance_evidence/`

and closes `P31-C-08` by requiring additive Phase 3.1 extensions:

- no new top-level evidence tree,
- version-bumped successor schemas for any case or package record that gains
  mandatory Phase 3.1 fields,
- required counted-case metadata:
  - `claim_surface_id = "phase31_bounded_mixed_motif_v1"`,
  - `runtime_class ∈ {"plain_partitioned_baseline", "phase3_unitary_island_fused", "phase31_channel_native", "phase31_channel_native_hybrid"}`,
  - `representation_primary = "kraus_bundle"`,
  - `fused_block_support_qbits`,
  - `contains_noise`,
  - `counted_phase31_case`,
- required correctness-specific additions:
  - mandatory slice `channel_invariants`,
  - invariant summary fields tied to `P31-ADR-004` / `P31-ADR-008`,
  - `partition_route_summary` for hybrid-counted cases,
- required performance-specific additions:
  - mandatory slice `break_even_table`,
  - decision-class field
    `phase3_sufficient | phase31_justified | phase31_not_justified_yet`,
  - explicit metrics versus the Phase 3 fused baseline,
  - hybrid route-coverage metadata:
    - `channel_native_partition_count`,
    - `phase3_routed_partition_count`,
    - `channel_native_member_count`,
    - `phase3_routed_member_count`,
  - partition-route records with:
    - `partition_runtime_class ∈ {"phase31_channel_native",
      "phase3_unitary_island_fused", "phase3_supported_unfused"}`,
    - `partition_route_reason` drawn from the frozen hybrid route reasons in
      `P31-ADR-012`.

### Rationale

- Phase 3.1 remains auditable as a follow-on rather than an unrelated evidence
  universe.
- The richer exact-channel and hybrid-routing claims become machine-reviewable
  instead of existing only in prose.

### Consequences

- `P31-C-08` is closed by this schema-and-emission rule once planning and
  checklist cite it.
- Implementation should extend existing bundle builders rather than fork a new
  evidence framework.

### Rejected Alternatives

- Create a completely separate `phase31_evidence/` tree.
- Reuse the old fields and attempt to infer Phase 3.1 semantics from free-text
  notes.

### Upstream Alignment and Traceability

- `P31-ADR-004`.
- `P31-ADR-010`.

## P31-ADR-014: Freeze The Scalar-Only Counted-Build Policy For Phase 3.1 V1

### Status

`Accepted`

### Context

Phase 3.1 v1 intentionally keeps host acceleration out of the counted paper
claim. To make that restriction machine-reviewable rather than a prose-only
promise, the build policy for counted bundles must be frozen explicitly.

### Decision

Phase 3.1 v1 counted correctness and performance bundles are generated only from
the following frozen build policy:

- `build_policy_id = "phase31_scalar_only_v1"`,
- `build_flavor = "scalar"`,
- `simd_enabled = false`,
- `tbb_enabled = false`,
- `thread_count = 1`,
- `counted_claim_build = true`.

Counted Phase 3.1 artifacts must record these fields directly in package-level
or case-level metadata once `P31-C-08` schema updates are implemented.

Task 6 remains a **later branch**:

- any TBB/SIMD evidence is non-counted for v1,
- any future counted use of host acceleration requires re-opening this policy
  and amending the checklist.

### Rationale

- This keeps the counted methods claim focused on channel-native fusion rather
  than host-side kernel tuning.
- It makes the exclusion of TBB/SIMD rows auditable in artifacts and papers.

### Consequences

- `P31-C-09` is closed by this policy once planning and checklist cite it.
- The default counted Phase 3.1 bundle surface is scalar-only until an explicit
  later ADR changes that rule.

### Rejected Alternatives

- Leave build flavor implicit and rely on prose.
- Let optional host acceleration appear inside counted v1 bundles.

### Upstream Alignment and Traceability

- `P31-ADR-013`.
- `task-6/TASK_6_MINI_SPEC.md`.

## P31-ADR-015: Promote Phase 3.1 To The Default Evidence Pipelines After Stabilization While Preserving Phase 3 As Explicit Legacy Scripts/Functions

### Status

`Accepted`

### Context

Phase 3.1 is intended to become the next default evidence surface for this
project once the new runtime and bundle builders are implemented and validated.
At the same time, the historical Phase 3 bundle flow must remain reproducible
for paper auditability and regression comparisons.

### Decision

Phase 3.1 adopts a **two-stage evidence migration**:

- **Stage A: sibling implementation**
  - add Phase 3.1-specific case builders, record builders, package builders,
    validation slices, and validation pipelines beside the current Phase 3
    entrypoints,
- **Stage B: default switch**
  - once the Phase 3.1 bundle flow is validated, make the default bundle and
    validation-pipeline entrypoints resolve to the Phase 3.1 implementations.

Historical Phase 3 surfaces are preserved through **explicit legacy
scripts/functions**, not flags. The target naming pattern is:

- bundle builders:
  - `build_phase3_correctness_package_payload()`,
  - `build_phase3_performance_benchmark_package_payload()`,
- validation pipelines:
  - `phase3_validation_pipeline.py` in each evidence tree, or equivalent
    explicit legacy module names.

The default entrypoints eventually become the Phase 3.1 ones:

- `build_correctness_package_payload()`,
- `build_performance_evidence_benchmark_package_payload()`,
- default `validation_pipeline.py` scripts.

### Rationale

- Default surfaces should reflect the current scientific frontier.
- Explicit legacy entrypoints are clearer for reproducibility than a mode flag
  or hidden configuration switch.

### Consequences

- Implementation should avoid mutating the current Phase 3 defaults until the
  Phase 3.1 sibling flow is complete.
- Documentation and future API references should describe Phase 3 as the legacy
  explicit surface once the default switch happens.

### Rejected Alternatives

- Use a `--contract phase3|phase31` flag on the same default pipeline.
- Keep Phase 3 defaults forever and leave Phase 3.1 as a permanent sibling
  branch.

### Upstream Alignment and Traceability

- `P31-ADR-013`.
- `P31-ADR-014`.
