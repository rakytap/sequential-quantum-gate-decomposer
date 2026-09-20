# Phase 3.1 - Fourth Vertical Slice (Remaining Counted Correctness Closure)

This document is the **implementation planning slice** for the next Phase 3.1
increment after the completed strict-plus-hybrid thin slice. It keeps the Layer
2 mini-specs authoritative and defines the smallest Task 3 expansion needed to
close the remaining counted correctness obligations on the frozen Phase 3.1
slice before the full Task 4 performance matrix.

**Do not** treat this as Task 4 performance closure (`P31-ADR-010`) or as
publication closure (Task 5). This slice exists to finish the remaining
claim-bearing correctness surface and the versioned correctness package needed
before the performance matrix can be interpreted confidently.

## Source mini-specs (Layer 2)


| Task                   | Document                                                   |
| ---------------------- | ---------------------------------------------------------- |
| 2 (runtime dependency) | `[task-2/TASK_2_MINI_SPEC.md](task-2/TASK_2_MINI_SPEC.md)` |
| 3                      | `[task-3/TASK_3_MINI_SPEC.md](task-3/TASK_3_MINI_SPEC.md)` |


## Slice boundary

**In**

- The remaining counted **strict** mixed-motif microcases from the frozen
correctness slice:
  - `phase31_microcase_2q_multi_noise_entangler_chain`
  - `phase31_microcase_2q_dense_same_support_motif`
- The remaining counted **hybrid** continuity anchor:
  - `phase2_xxz_hea_q6_continuity`
- The required external-reference rows from `P31-ADR-011` / planning §10.4:
  - `phase31_microcase_1q_u3_local_noise_chain`
  - `phase31_microcase_2q_cnot_local_noise_pair`
  - `phase31_microcase_2q_multi_noise_entangler_chain`
  - `phase31_microcase_2q_dense_same_support_motif`
  - `phase2_xxz_hea_q4_continuity`
- Versioned `correctness_evidence` emission with the required Phase 3.1
metadata plus:
  - `channel_invariants`
  - `partition_route_summary` for hybrid-counted rows
- Existing completed strict and hybrid tests remain regression anchors while the
remaining counted correctness rows are promoted into claim-bearing gates.

**Out (deferred to later slices / closure steps)**

- The full counted structured performance matrix from `P31-ADR-010`
- `break_even_table` / `justification_map` closure
- Task 5 publication closure and paper reframing
- Any Task 6 host-acceleration work
- Any broadening of the frozen v1 support surface or counted case list

## Recommended frozen IDs for this slice

**Remaining counted strict microcases**

- `phase31_microcase_2q_multi_noise_entangler_chain`
- `phase31_microcase_2q_dense_same_support_motif`

**Remaining counted hybrid continuity anchor**

- `phase2_xxz_hea_q6_continuity`

**Required external slice rows**

- `phase31_microcase_1q_u3_local_noise_chain`
- `phase31_microcase_2q_cnot_local_noise_pair`
- `phase31_microcase_2q_multi_noise_entangler_chain`
- `phase31_microcase_2q_dense_same_support_motif`
- `phase2_xxz_hea_q4_continuity`

---

### Story P31-S10: The remaining counted correctness slice closes under the frozen strict-plus-hybrid contract

**User/Research value**

- Converts Phase 3.1 exactness from a strong partial result into the full frozen
counted correctness slice.
- Removes the biggest remaining ambiguity before the full performance matrix:
whether negative or mixed performance rows are being interpreted on top of an
incompletely validated correctness surface.

**Given / When / Then**

- **Given** the completed strict motif-proof slice, the completed hybrid `q4`
continuity anchor, and the thin structured pilot already implemented.
- **When** the remaining counted correctness rows and required external
micro-validation rows run through the Phase 3.1 correctness harness.
- **Then** every frozen correctness case is either green with stable bundle
metadata or explicitly recorded as a blocking failure on the frozen slice, and
the correctness package exposes the required Phase 3.1 invariant and route
summary fields.

**Scope**

- **In:** remaining strict counted microcases, counted hybrid
`phase2_xxz_hea_q6_continuity`, required Aer subset, and the versioned
`correctness_evidence` / `channel_invariants` / `partition_route_summary`
migration.
- **Out:** performance matrix expansion, publication framing, and Task 6.

**Acceptance signals**

- `phase31_microcase_2q_multi_noise_entangler_chain` passes through the strict
path under the frozen §10.1 thresholds.
- `phase31_microcase_2q_dense_same_support_motif` passes through the strict path
under the frozen §10.1 thresholds.
- `phase2_xxz_hea_q6_continuity` passes through the hybrid path with stable
route-summary assertions.
- The external slice from §10.4 is emitted with stable case IDs and versioned
schema.
- The counted correctness bundle contains the required Phase 3.1 fields from
planning §10.6.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §6, §8 Task 3, §10.1, §10.2, §10.4,
§10.6
- ADRs: `P31-ADR-003`, `P31-ADR-008`, `P31-ADR-009`, `P31-ADR-011`,
`P31-ADR-013`

#### Engineering tasks (Story P31-S10)

##### Engineering Task P31-S10-E01: Promote the remaining counted strict microcases and `q6` hybrid continuity row into claim-bearing correctness gates

**Implementation plan:** `[ENGINEERING_TASK_P31_S10_E01_IMPLEMENTATION_PLAN.md](task-3/ENGINEERING_TASK_P31_S10_E01_IMPLEMENTATION_PLAN.md)`

**Implements story**

- Story P31-S10

**Change type**

- code | tests | validation automation

**Definition of done**

- The remaining counted strict microcases and the counted `q6` hybrid continuity
row are green in reproducible pytest or harness form.
- Hybrid `q6` route-summary assertions are stable and reviewer-auditable.
- Any counted correctness blocker is recorded explicitly rather than hidden by
slice narrowing.

**Execution checklist**

- Promote `phase31_microcase_2q_multi_noise_entangler_chain` to a
claim-bearing exactness gate.
- Promote `phase31_microcase_2q_dense_same_support_motif` to a
claim-bearing exactness gate.
- Add the counted `phase2_xxz_hea_q6_continuity` hybrid exactness gate with
route-summary assertions.
- Keep existing `q4` hybrid and strict-slice rows as regression anchors.
- Record explicit blockers if any counted correctness row fails.

**Evidence produced**

- Passing tests or harness outputs for the remaining counted correctness rows.
- Updated counted-case registry or manifest inputs tied to the frozen IDs.

**Risks / rollback**

- Risk: `q6` continuity exposes a routing or scale behavior that weakens the
current partial-confidence boundary.
- Rollback/mitigation: keep the failure explicit and do not broaden the support
surface to compensate.

##### Engineering Task P31-S10-E02: Emit the versioned correctness package with Aer rows and Phase 3.1 slices

**Implementation plan:** `[ENGINEERING_TASK_P31_S10_E02_IMPLEMENTATION_PLAN.md](task-3/ENGINEERING_TASK_P31_S10_E02_IMPLEMENTATION_PLAN.md)`

**Implements story**

- Story P31-S10

**Change type**

- validation automation | docs

**Definition of done**

- The required Aer subset is emitted for the frozen external slice only.
- The versioned `correctness_evidence` successor includes the required Phase
3.1 fields plus `channel_invariants` and `partition_route_summary`.
- A stable Case ID -> internal pass -> Aer pass summary exists for review.

**Execution checklist**

- Wire the required Aer rows from planning §10.4 into the correctness
package.
- Version-bump the package or case schema where Phase 3.1 fields are added.
- Emit `channel_invariants` for the counted strict microcases.
- Emit `partition_route_summary` for the counted hybrid rows.
- Produce a review-ready Case ID -> internal pass -> Aer pass summary.

**Evidence produced**

- Regeneratable correctness bundle for the full frozen counted correctness
surface.
- Review-ready case summary table aligned to the same frozen IDs.

**Risks / rollback**

- Risk: partial schema migration obscures which rows are truly Phase 3.1-counted.
- Rollback/mitigation: prefer explicit successor schema names over silent reuse
of Phase 3 package shapes.

---

## Story-to-engineering map


| Story   | Engineering tasks        |
| ------- | ------------------------ |
| P31-S10 | P31-S10-E01, P31-S10-E02 |


## Suggested implementation order

1. `P31-S10-E01`
2. P31-S10-E02

Parallelism:

- Aer harness and schema migration can begin in parallel, but should not be
treated as claim-bearing until the remaining counted correctness rows are
green or explicitly blocked.
- Existing completed strict / hybrid rows should remain active as regression
anchors throughout the slice.

---

## Slice implementation status

Update this section when the fourth-slice engineering tasks land.


| Field                                          | Value                                                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Remaining strict counted microcases            | `phase31_microcase_2q_multi_noise_entangler_chain`, `phase31_microcase_2q_dense_same_support_motif`      |
| Remaining hybrid continuity anchor             | `phase2_xxz_hea_q6_continuity`                                                                           |
| Required external slice rows in this increment | four `phase31_microcase_`* rows plus `phase2_xxz_hea_q4_continuity`                                      |
| Fourth vertical slice implementation           | **Planned**                                                                                              |
| Immediate successor after this slice           | `P31-S11` full counted performance matrix                                                                |
| Deferred beyond this slice                     | full `P31-ADR-010` matrix closure, `break_even_table` / `justification_map`, publication closure, Task 6 |


