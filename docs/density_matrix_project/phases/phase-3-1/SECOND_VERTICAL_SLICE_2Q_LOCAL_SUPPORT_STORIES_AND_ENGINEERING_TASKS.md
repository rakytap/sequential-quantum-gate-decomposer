# Phase 3.1 - Second Vertical Slice (2q / Local-Support Increment)

This document is the **implementation planning slice** for the next Phase 3.1
increment after the completed 1-qubit channel-native slice. It keeps the Layer
2 mini-specs authoritative and defines the smallest Tasks 1-2 expansion plus
the smallest Task 3 subset needed to make the frozen **2-qubit / local-support**
contract real.

**Do not** treat this as full Task 3 closure (`P31-ADR-009`, `P31-ADR-011`,
full `correctness_evidence` schema per `P31-ADR-013`) or as Task 4 performance
closure (`P31-ADR-010`). Those remain later increments after this slice proves
that the runtime can execute a bounded 2-qubit motif on local support inside a
larger density-state workload.

## Source mini-specs (Layer 2)

| Task | Document |
|------|----------|
| 1 | [`task-1/TASK_1_MINI_SPEC.md`](task-1/TASK_1_MINI_SPEC.md) |
| 2 | [`task-2/TASK_2_MINI_SPEC.md`](task-2/TASK_2_MINI_SPEC.md) |
| 3 (slice only) | [`task-3/TASK_3_MINI_SPEC.md`](task-3/TASK_3_MINI_SPEC.md) |

## Slice boundary

**In**

- Primary representation `kraus_bundle` generalized from the completed 1-qubit
  slice to **1- and 2-qubit** bounded objects, with completeness and
  positivity checks evaluated at the correct support dimension under
  `P31-ADR-008`.
- Ordered lowering for the smallest frozen counted 2-qubit motif:
  `U3` / `CNOT` plus local single-qubit noise on the same 2-qubit support,
  consistent with `P31-ADR-007`.
- Channel-native application on **local support inside a larger global density
  state** so the runtime no longer requires `descriptor_set.qbit_num == 1` to
  execute a counted fused block.
- Runtime path identity `phase31_channel_native` /
  `execute_partitioned_density_channel_native(...)` with the same no-silent-
  fallback rule already frozen in `P31-ADR-012`.
- **One** frozen counted 2-qubit microcase end-to-end:
  `phase31_microcase_2q_cnot_local_noise_pair`.
- **One** non-counted larger-workload local-support smoke case proving that a
  bounded 2-qubit fused block can execute inside a workload with spectator
  qubits. Recommended stable ID for this slice:
  `phase31_local_support_q4_spectator_embedding_smoke`.

**Out (deferred to full Task 3 / later slices)**

- Remaining counted 2-qubit microcases:
  `phase31_microcase_2q_multi_noise_entangler_chain` and
  `phase31_microcase_2q_dense_same_support_motif`.
- Bounded continuity anchors:
  `phase2_xxz_hea_q4_continuity` and `phase2_xxz_hea_q6_continuity`.
- Full external Aer matrix from `P31-ADR-011`.
- Version-bumped production `correctness_evidence` bundles and CI regeneration
  policy at full `P31-C-08` breadth.
- Counted structured performance families and `break_even_table` packaging from
  Task 4 / `P31-ADR-010`.

**Recommended counted case ID for the slice:**  
`phase31_microcase_2q_cnot_local_noise_pair`

**Recommended non-counted local-support smoke ID for the slice:**  
`phase31_local_support_q4_spectator_embedding_smoke`

If implementation order forces a different bounded embedding smoke case, record
the replacement ID here and keep the same acceptance intent: one eligible
2-qubit fused block must execute inside a workload with spectator qubits.

---

### Story P31-S04: Two-qubit channel-native blocks are exact on local support

**User/Research value**

- Moves Phase 3.1 beyond a whole-workload 1-qubit proof-of-concept into the
  minimum scientifically meaningful 2-qubit object frozen by `P31-ADR-007`.
- Proves that the primary mathematical object remains exact when the fused block
  acts on a local subset of the global density state rather than on the full
  workload width.

**Given / When / Then**

- **Given** a contiguous eligible 2-qubit mixed motif built from `U3`, `CNOT`,
  and local depolarizing / amplitude-damping / phase-damping channels on those
  same qubits, plus a density state whose total workload width may exceed the
  motif width.
- **When** the motif is lowered into the primary Phase 3.1 representation and
  applied on its local support.
- **Then** the global output density matrix matches sequential
  `NoisyCircuit` semantics within **P31-C-03**, spectator qubits remain correct
  because the full global density matrix agrees with the reference, and the
  representation-level invariant suite passes at the correct support dimension.

**Scope**

- **In:** support-aware representation shape, 4x4 channel invariants, local to
  global support mapping, direct apply tests against sequential reference.
- **Out:** full runtime selection / failure matrix (Story P31-S05); full case
  registry and bundle schema closure (Story P31-S06 expansion).

**Acceptance signals**

- Direct lowering / apply evidence for
  `phase31_microcase_2q_cnot_local_noise_pair` matches sequential reference at
  **P31-C-03** thresholds.
- Representation-level invariants for the composed 2-qubit object satisfy
  **P31-ADR-008** thresholds:
  completeness-style residuals `<= 1e-10`,
  positivity-style eigenvalue floors `>= -1e-12`.
- A larger-workload local-support smoke case
  (`phase31_local_support_q4_spectator_embedding_smoke`) matches the sequential
  reference on the **full** density matrix, demonstrating spectator preservation
  by equality rather than by a weaker partial check.
- One short note in code comments or docs freezes the 2-qubit local ordering
  convention used for `CNOT`, local support, and `partition.local_to_global_qbits`.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 1; **P31-C-01**,
  **P31-C-03**, **P31-C-04** (subset).
- ADRs: `P31-ADR-004`, `P31-ADR-007`, `P31-ADR-008`, `P31-ADR-003`.

#### Engineering tasks (Story P31-S04)

##### Engineering Task P31-S04-E01: Generalize the primary representation and invariant helpers to support size-aware 1q/2q blocks

**Implements story**

- Story P31-S04

**Change type**

- code | tests | docs

**Definition of done**

- The primary representation and its invariant helpers no longer assume `2x2`
  / 1-qubit shape; they validate both 1-qubit and 2-qubit bounded objects using
  the correct support dimension.
- The channel-native apply path accepts explicit local-support information and a
  local-to-global mapping contract suitable for workloads larger than the fused
  block itself.
- Unsupported support sizes still fail explicitly rather than being accepted
  accidentally.

**Execution checklist**

- [ ] Replace hard-coded 1-qubit / `2x2` assumptions in representation and
      invariant helpers with support-aware dimension logic.
- [ ] Freeze and document the canonical 2-qubit local ordering convention used
      by `CNOT` lowering and local-support embedding.
- [ ] Keep support widths above 2 qubits as explicit hard failures under the
      Phase 3.1 unsupported taxonomy.

**Evidence produced**

- Focused unit tests proving invariant checks work on a 4x4 bounded object.
- A reviewable note in code comments or docs stating the local-support ordering
  rule used by the slice.
- Concrete file-level implementation plan:
  `task-1/ENGINEERING_TASK_P31_S04_E01_IMPLEMENTATION_PLAN.md`.

**Risks / rollback**

- Risk: reshape / ordering mistakes can preserve matrix dimensions while still
  changing semantics.
- Rollback/mitigation: validate the generalized helpers first on tiny synthetic
  motifs and the smallest counted 2-qubit microcase before wiring larger
  workloads.

---

##### Engineering Task P31-S04-E02: Add `CNOT` lowering and ordered 2-qubit composition tests

**Implements story**

- Story P31-S04

**Change type**

- code | tests

**Definition of done**

- `CNOT` can be lowered into the primary representation in the canonical local
  ordering used by the runtime.
- Ordered compositions of `U3`, `CNOT`, and local single-qubit noise on either
  qubit agree with sequential reference on bounded 2-qubit motifs.

**Execution checklist**

- [ ] Implement `CNOT` lowering for the channel-native path using the same local
      support convention documented in P31-S04-E01.
- [ ] Add direct tests covering ordered composition for gate-noise-gate and
      noise-on-either-qubit variants.
- [ ] Assert that descriptor order, not an internal shortcut order, defines the
      resulting fused object.

**Evidence produced**

- Focused lowering / composition pytest coverage rooted in the smallest counted
  2-qubit motif.
- Reviewable direct-comparison evidence against sequential density evolution
  before full partition-runtime wiring.

**Risks / rollback**

- Risk: `CNOT` control / target orientation may be inverted relative to the
  sequential oracle.
- Rollback/mitigation: freeze one explicit control-target convention and test
  both lowering and final-state agreement on deterministic fixtures.

---

### Story P31-S05: Channel-native runtime admits 2-qubit same-support motifs inside larger workloads and stays non-silent at the boundaries

**User/Research value**

- Makes the counted Phase 3.1 path usable beyond a 1-qubit full-workload toy
  case and removes the blocker that currently prevents all counted 2-qubit,
  continuity, and structured-family follow-on work.
- Keeps the comparative story honest: either the bounded 2-qubit local-support
  path ran, or the runtime failed with a documented unsupported condition.

**Given / When / Then**

- **Given** a partitioned execution request with channel-native fusion on and a
  partition containing an eligible 2-qubit same-support mixed motif.
- **When** the runtime validates the motif and applies it on its local support
  inside the global density state.
- **Then** the runtime reports `phase31_channel_native`, preserves global
  ordering, emits auditable fused-region metadata, and hard-fails on patterns
  that exceed the frozen 2-qubit/local-support boundary.

**Scope**

- **In:** runtime gating based on local motif support rather than whole-workload
  width, mapped application on global density matrices, positive and negative
  integration tests.
- **Out:** full correctness bundle emission, Aer rows, and performance package
  generation.

**Acceptance signals**

- Positive integration test:
  `phase31_microcase_2q_cnot_local_noise_pair` executes through
  `execute_partitioned_density_channel_native(...)` with runtime-path label
  `phase31_channel_native` and at least one
  `PHASE31_FUSION_KIND_CHANNEL_NATIVE_MOTIF` fused region.
- Positive larger-workload smoke test:
  `phase31_local_support_q4_spectator_embedding_smoke` executes with channel-
  native fusion on a 2-qubit local support inside a 4-qubit workload.
- Negative matrix:
  support `> 2` qubits -> `channel_native_qubit_span`,
  pure unitary motif -> `channel_native_noise_presence`,
  unsupported operation or off-support pattern -> `channel_native_support_surface`.
- Positive tests assert fused-region target-qubit metadata so the applied local
  support is reviewable and not only inferred from final-state equality.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 2; **P31-C-02**,
  **P31-C-07**, **P31-C-08** (path attribution direction).
- ADRs: `P31-ADR-012`, `P31-ADR-007`, `P31-ADR-003`, `P31-ADR-005`.

#### Engineering tasks (Story P31-S05)

##### Engineering Task P31-S05-E01: Replace whole-workload gating with local-support eligibility and mapped application

**Implements story**

- Story P31-S05

**Change type**

- code | tests

**Definition of done**

- Channel-native validation no longer requires `descriptor_set.qbit_num == 1`
  to admit a supported fused block.
- The runtime applies 1-qubit or 2-qubit channel-native blocks to the correct
  local support inside a larger global density state using explicit mapping.
- Runtime-path identity and fused-region audit fields remain stable for the new
  supported slice.

**Execution checklist**

- [ ] Remove the whole-workload-width assumption from channel-native motif
      validation and replace it with local-support validation.
- [ ] Thread local-support to global-support mapping explicitly through the apply
      path and fused-region record.
- [ ] Keep pure-unitary islands and broader unsupported motifs as explicit hard
      failures when channel-native mode is requested.

**Evidence produced**

- Positive integration test on
  `phase31_microcase_2q_cnot_local_noise_pair`.
- Positive integration test on
  `phase31_local_support_q4_spectator_embedding_smoke`.

**Risks / rollback**

- Risk: a correct 2-qubit object may still be applied to the wrong global qubit
  pair when local/global mapping is threaded through the runtime.
- Rollback/mitigation: assert fused-region target metadata and compare against
  the full sequential reference on the smoke workload before expanding further.

---

##### Engineering Task P31-S05-E02: Add the local-support negative matrix and audit assertions for the second slice

**Implements story**

- Story P31-S05

**Change type**

- tests | docs

**Definition of done**

- At least three deterministic local-support boundary failures are covered by
  tests with stable `first_unsupported_condition` assertions.
- Positive tests assert the fused-region target support, member indices, and
  operation names so runtime auditability is part of the acceptance evidence.

**Execution checklist**

- [ ] Add a `> 2` local-support negative case and assert
      `channel_native_qubit_span`.
- [ ] Add a pure-unitary 2-qubit negative case and assert
      `channel_native_noise_presence`.
- [ ] Add one unsupported off-support or unsupported-operation negative case and
      assert `channel_native_support_surface`.
- [ ] Tighten positive tests so fused-region metadata is asserted, not merely
      inspected manually.

**Evidence produced**

- Focused pytest negative matrix for the second slice.
- Reviewable runtime-audit assertions proving which local support actually fused.

**Risks / rollback**

- Risk: broad negative tests can over-freeze behavior that should remain
  implementation-local.
- Rollback/mitigation: assert only the frozen taxonomy and local-support
  boundary, not incidental wording outside the documented contract.

---

### Story P31-S06: One counted 2-qubit microcase plus one larger-workload smoke case close the second-slice correctness loop

**User/Research value**

- Produces the minimum reproducible proof that Phase 3.1 is no longer limited to
  a 1-qubit full-workload slice and can now execute a bounded 2-qubit motif on
  local support inside a larger density-state workflow.
- Creates a clean handoff into the remaining Task 3 / Task 4 work without
  pretending the full counted matrix or evidence schema is already done.

**Given / When / Then**

- **Given** Stories P31-S04 and P31-S05 implemented and the frozen counted case
  ID for this slice.
- **When** the second-slice correctness gate runs.
- **Then** the counted 2-qubit microcase and the larger-workload local-support
  smoke case both match the sequential oracle within **P31-C-03**, and the doc
  record makes the remaining deferred Phase 3.1 evidence work explicit.

**Scope**

- **In:** one counted 2-qubit microcase, one non-counted larger-workload smoke
  case, deterministic pytest evidence, documentation hook for what remains.
- **Out:** full `P31-ADR-009` closure, Aer rows, bundle schema version bump,
  structured performance measurements.

**Acceptance signals**

- `phase31_microcase_2q_cnot_local_noise_pair` passes full end-to-end internal
  correctness checks:
  Frobenius difference `<= 1e-10`,
  `|Tr(rho) - 1| <= 1e-10`,
  `rho.is_valid(tol=1e-10)`,
  and representation invariants at `P31-ADR-008` thresholds.
- `phase31_local_support_q4_spectator_embedding_smoke` passes the same internal
  exactness checks on the full density matrix, while remaining explicitly
  **non-counted** for the publication claim.
- The doc record names the remaining deferred counted IDs:
  `phase31_microcase_2q_multi_noise_entangler_chain`,
  `phase31_microcase_2q_dense_same_support_motif`,
  `phase2_xxz_hea_q4_continuity`,
  `phase2_xxz_hea_q6_continuity`,
  plus the full external and bundle work from Task 3.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Tasks 2-3; **P31-C-03**,
  **P31-C-04** (subset), **P31-C-08** (directional).
- ADRs: `P31-ADR-009` (subset), `P31-ADR-011` (future external slice),
  `P31-ADR-013` (future bundle alignment).

#### Engineering tasks (Story P31-S06)

##### Engineering Task P31-S06-E01: Register the second-slice fixtures and end-to-end comparisons

**Implements story**

- Story P31-S06

**Change type**

- code | tests

**Definition of done**

- The counted 2-qubit case ID remains
  `phase31_microcase_2q_cnot_local_noise_pair`.
- One stable larger-workload local-support smoke fixture exists for this slice;
  recommended ID:
  `phase31_local_support_q4_spectator_embedding_smoke`.
- End-to-end tests compare channel-native execution against the sequential
  oracle on both cases and assert the frozen numeric thresholds.

**Execution checklist**

- [ ] Keep the counted 2-qubit case ID unchanged so later Task 3 evidence can
      reuse it directly.
- [ ] Add or document one stable larger-workload local-support smoke case ID for
      the slice.
- [ ] Add end-to-end tests asserting Frobenius, trace-validity, and invariant
      checks on both cases.
- [ ] Document in the test module what remains intentionally uncovered after the
      second slice.

**Evidence produced**

- Recommended command:
  `conda run -n qgd python -m pytest tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py -q`
- Stable fixture IDs ready for later merge into the broader Task 3 harness.

**Risks / rollback**

- Risk: the smoke case can drift into a de facto continuity benchmark if it is
  too large or too workflow-specific.
- Rollback/mitigation: keep the smoke case synthetic and bounded; defer real
  continuity anchors to the next Task 3 increment.

---

##### Engineering Task P31-S06-E02: Record second-slice completion and the remaining Task 3 / Task 4 expansion hooks

**Implements story**

- Story P31-S06

**Change type**

- docs

**Definition of done**

- This file records the counted 2-qubit slice case and the larger-workload smoke
  case, then marks the second vertical slice complete when true.
- `FIRST_VERTICAL_SLICE_STORIES_AND_ENGINEERING_TASKS.md` and
  `DETAILED_PLANNING_PHASE_3_1.md` point to this file as the next increment
  after the completed 1-qubit slice.
- The remaining deferred counted IDs and full evidence-work items are named
  explicitly so reviewers can see what this slice does **not** claim.

**Execution checklist**

- [ ] Update the **Slice implementation status** section in this file when the
      second slice is implemented.
- [ ] Keep Layer 1 contract closure in
      `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` separate from slice progress.
- [ ] Add or maintain pointers from the first-slice and detailed-planning docs.

**Evidence produced**

- Traceability note for reviewers linking the completed first slice to this
  second slice and then to the remaining Task 3 / Task 4 work.

**Risks / rollback**

- Risk: readers may mistake the second slice for full Phase 3.1 Task 3 closure.
- Rollback/mitigation: name the remaining counted IDs and bundle work
  explicitly in the status section and deferred list.

---

## Story-to-engineering map

| Story | Engineering tasks |
|-------|-------------------|
| P31-S04 | P31-S04-E01, P31-S04-E02 |
| P31-S05 | P31-S05-E01, P31-S05-E02 |
| P31-S06 | P31-S06-E01, P31-S06-E02 |

## Suggested implementation order

1. P31-S04-E01 -> P31-S04-E02
2. P31-S05-E01 -> P31-S05-E02
3. P31-S06-E01 -> P31-S06-E02

Parallelism:

- P31-S04-E02 can prototype against direct lowering tests once P31-S04-E01
  freezes the local ordering convention.
- P31-S05-E02 can be prepared in parallel with P31-S05-E01 once the frozen
  unsupported-condition taxonomy is clear.

## Slice completion checklist

- [ ] P31-S04-E01: support-aware representation and invariant helpers for 1q/2q
      bounded objects are in place.
- [ ] P31-S04-E02: `CNOT` lowering and ordered 2-qubit composition tests pass.
- [ ] P31-S05-E01: runtime local-support gating and mapped application work for
      2-qubit motifs inside larger workloads.
- [ ] P31-S05-E02: negative local-support matrix and fused-region audit
      assertions are in place.
- [ ] P31-S06-E01: counted 2-qubit case plus larger-workload smoke case pass
      end-to-end sequential-reference comparison.
- [ ] P31-S06-E02: implementation status and deferred follow-on hooks are
      recorded in docs.

---

## Slice implementation status

Update this section when P31-S06-E02 completes (implementation progress only; do
not fold into `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`).

| Field | Value |
|-------|-------|
| Counted slice case ID | `phase31_microcase_2q_cnot_local_noise_pair` |
| Non-counted local-support smoke ID | `phase31_local_support_q4_spectator_embedding_smoke` (recommended) |
| Second vertical slice implementation | **Planned** |
| Recommended evidence module | `tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py` |
| Deferred counted correctness IDs after this slice | `phase31_microcase_2q_multi_noise_entangler_chain`, `phase31_microcase_2q_dense_same_support_motif`, `phase2_xxz_hea_q4_continuity`, `phase2_xxz_hea_q6_continuity`, plus every counted performance case from `P31-ADR-010` |
| Deferred full Task 3 work | Aer per `P31-ADR-011`; version-bumped `correctness_evidence` rows and `channel_invariants` packaging per `P31-ADR-013` |
| Deferred Task 4 work | `phase31_pair_repeat`, `phase31_alternating_ladder`, and control-family performance packaging with `break_even_table` per `P31-ADR-010` |
