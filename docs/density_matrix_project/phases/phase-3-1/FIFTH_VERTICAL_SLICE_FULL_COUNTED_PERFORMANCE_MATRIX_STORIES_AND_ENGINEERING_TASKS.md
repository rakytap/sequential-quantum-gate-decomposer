# Phase 3.1 - Fifth Vertical Slice (Full Counted Performance Matrix Closure)

This document is the **implementation planning slice** for the next Phase 3.1
increment after the completed fourth-slice correctness closure. It keeps the
Layer 2 mini-specs authoritative and defines the Layer 3 / Layer 4 plan needed
to close the frozen counted Task 4 performance obligations on the Phase 3.1 v1
slice.

**Do not** treat this as publication closure (Task 5), as Task 6 host
acceleration, or as an opportunity to broaden the frozen support surface. This
slice exists to finish the full counted performance matrix and the route-aware
decision artifact so the later publication decision is made from emitted
artifacts rather than inference.

## Source mini-specs (Layer 2)

| Task | Document |
|------|----------|
| 2 (runtime dependency) | [`task-2/TASK_2_MINI_SPEC.md`](task-2/TASK_2_MINI_SPEC.md) |
| 4 | [`task-4/TASK_4_MINI_SPEC.md`](task-4/TASK_4_MINI_SPEC.md) |

## Slice boundary

**In**

- The full frozen `P31-ADR-010` counted performance matrix:
  - primary families:
    - `phase31_pair_repeat`
    - `phase31_alternating_ladder`
  - control family:
    - `layered_nearest_neighbor`
  - qubit bands:
    - `q ∈ {8, 10}`
  - primary-family noise patterns:
    - `periodic`
    - `dense`
  - primary seeds:
    - `20260318`
    - `20260319`
    - `20260320`
  - control-family sparse seed:
    - `20260318`
- Baseline trio on every counted row:
  - sequential density reference,
  - Phase 3 partitioned+fused baseline,
  - explicit Phase 3.1 hybrid baseline.
- Hybrid route-coverage emission on every counted row:
  - `channel_native_partition_count`,
  - `phase3_routed_partition_count`,
  - `channel_native_member_count`,
  - `phase3_routed_member_count`,
  - partition-level route records.
- `decision_class` on every counted row.
- Matrix-wide `break_even_table` / `justification_map`.
- Scalar-build metadata required by `P31-C-09`.

**Out (deferred to later closure steps)**

- Task 5 publication rewrites.
- Program-level sync in `PLANNING.md`, `PUBLICATIONS.md`, or `CHANGELOG.md`.
- Task 6 host acceleration or any reopened SIMD / TBB contract.
- Any broadening of the frozen counted family inventory or support surface.

## Frozen counted matrix summary

**Primary positive-method families**

- `phase31_pair_repeat`
- `phase31_alternating_ladder`

**Control family**

- `layered_nearest_neighbor`

**Counted rows**

- Primary families:
  - `2 families × 2 qbit counts × 2 noise patterns × 3 seeds = 24`
- Control family:
  - `1 family × 2 qbit counts × 1 sparse pattern × 1 seed = 2`

**Total counted rows**

- `26`

---

### Story P31-S11: The frozen performance matrix yields route-aware decision rows

**User/Research value**

- Answers the whole-workload "when does channel-native fusion matter?" question
  on the frozen slice rather than on one pilot row.
- Separates local exactness success from workload-level justification.

**Given / When / Then**

- **Given** the bounded counted correctness slice is already green on the frozen
  Phase 3.1 v1 surface and the explicit hybrid runtime is available.
- **When** the full `P31-ADR-010` counted matrix runs through the Phase 3.1
  performance harness.
- **Then** each counted row records the baseline trio, route coverage, and a
  measured `decision_class`, and the matrix emits a route-aware decision
  artifact without yet changing publication prose.

**Scope**

- **In:** all remaining primary-family and control rows from planning §10.3,
  scalar-build metadata, route-coverage emission, matrix-wide decision
  classification, and the required `break_even_table` / `justification_map`.
- **Out:** publication framing, Task 6, and any widening of the counted matrix.

**Acceptance signals**

- All 26 frozen counted rows are emitted with stable IDs.
- Each row records sequential, Phase 3 fused, and Phase 3.1 hybrid timings.
- Each row records hybrid route coverage and `decision_class`.
- The matrix emits a `break_even_table` or `justification_map` tied to the same
  case IDs.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §6, §8 Task 4, §10.3, §10.6, §10.7.
- ADRs: `P31-ADR-005`, `P31-ADR-006`, `P31-ADR-010`, `P31-ADR-013`,
  `P31-ADR-014`.

#### Engineering tasks (Story P31-S11)

##### Engineering Task P31-S11-E01: Expand the current hybrid pilot harness into the full frozen counted matrix

**Implementation plan:** [`task-4/ENGINEERING_TASK_P31_S11_E01_IMPLEMENTATION_PLAN.md`](task-4/ENGINEERING_TASK_P31_S11_E01_IMPLEMENTATION_PLAN.md)

**Implements story**

- Story P31-S11

**Change type**

- benchmark harness | tests

**Definition of done**

- The current pilot runner expands to the full counted matrix from planning
  §10.3.
- Each emitted row carries the baseline trio and scalar-build metadata required
  by planning §10.7.
- The counted matrix inventory is protected by regression tests.

**Execution checklist**

- Add the remaining primary-family rows from `phase31_pair_repeat` and
  `phase31_alternating_ladder`.
- Add the control-family `layered_nearest_neighbor` rows.
- Preserve the frozen case IDs, seeds, patterns, and qubit bands from planning
  §10.3.
- Guard the inventory and iterator helpers with regression tests.

**Evidence produced**

- Reproducible benchmark rows for the full frozen counted matrix.
- Regression tests protecting the counted matrix inventory.

**Risks / rollback**

- Risk: harness expansion drifts from the frozen slice and reintroduces
  selection bias.
- Rollback/mitigation: derive rows from the frozen inventory and test the exact
  counts.

##### Engineering Task P31-S11-E02: Emit the route-aware decision artifact and machine-readable matrix summary

**Implementation plan:** [`task-4/ENGINEERING_TASK_P31_S11_E02_IMPLEMENTATION_PLAN.md`](task-4/ENGINEERING_TASK_P31_S11_E02_IMPLEMENTATION_PLAN.md)

**Implements story**

- Story P31-S11

**Change type**

- validation automation | docs

**Definition of done**

- The matrix emits `break_even_table` / `justification_map` directly from the
  measured rows.
- A machine-readable summary table exists for later publication use without
  requiring manual reconstruction.
- Negative or mixed outcomes remain explicit rather than being masked by one
  favorable row.

**Execution checklist**

- Emit `decision_class` for every counted performance row.
- Emit route-coverage counters and partition-level route records.
- Build the `break_even_table` / `justification_map` from measured results.
- Record explicit negative or mixed outcomes without rewriting the frozen
  success rule.

**Evidence produced**

- Full performance bundle with decision artifact.
- Review-ready case table: Case ID -> baselines -> route coverage -> decision.

**Risks / rollback**

- Risk: one favorable row is mistaken for matrix closure.
- Rollback/mitigation: keep the matrix-level artifact mandatory and include the
  control family in the same summary.

---

## Story-to-engineering map

| Story | Engineering tasks |
|-------|-------------------|
| P31-S11 | P31-S11-E01, P31-S11-E02 |

## Suggested implementation order

1. `P31-S11-E01`
2. `P31-S11-E02`

Parallelism:

- Inventory expansion and decision-artifact shaping can overlap, but
  `P31-S11-E02` should not be treated as claim-bearing until the full counted
  row set from `P31-S11-E01` is stable.

---

## Slice implementation status

Update this section when the fifth-slice engineering tasks land.

| Field | Value |
|-------|-------|
| Counted matrix rows | 26 frozen rows (`24` primary + `2` control) |
| Fifth vertical slice implementation | **Planned** |
| Immediate successor after this slice | `P31-S12` pre-publication evidence review |
| Deferred beyond this slice | Task 5 publication closure, program sync, Task 6 |
