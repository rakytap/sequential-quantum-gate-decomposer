# Phase 3.1 - Third Vertical Slice (Strict Plus Hybrid Whole-Workload Increment)

This document is the **implementation planning slice** for the next Phase 3.1
increment after the completed bounded 2-qubit local-support slice. It keeps the
Layer 2 mini-specs authoritative and defines the smallest Tasks 2-4 expansion
needed to make the frozen **strict-plus-hybrid** runtime contract real.

**Do not** treat this as full Task 3 closure (`P31-ADR-009`, `P31-ADR-011`,
full `correctness_evidence` schema per `P31-ADR-013`) or as full Task 4
closure (`P31-ADR-010`). This slice is the first thin end-to-end increment for
the new whole-workload **hybrid** interpretation.

## Source mini-specs (Layer 2)


| Task | Document                                                   |
| ---- | ---------------------------------------------------------- |
| 2    | `[task-2/TASK_2_MINI_SPEC.md](task-2/TASK_2_MINI_SPEC.md)` |
| 3    | `[task-3/TASK_3_MINI_SPEC.md](task-3/TASK_3_MINI_SPEC.md)` |
| 4    | `[task-4/TASK_4_MINI_SPEC.md](task-4/TASK_4_MINI_SPEC.md)` |


## Slice boundary

**In**

- The explicit **hybrid** Phase 3.1 whole-workload runtime:
  - helper `execute_partitioned_density_channel_native_hybrid(...)`,
  - runtime-path label `phase31_channel_native_hybrid`.
- Frozen hybrid route-policy reasons from `P31-ADR-012`:
  - `eligible_channel_native_motif`,
  - `pure_unitary_partition`,
  - `channel_native_noise_presence`,
  - `channel_native_qubit_span`,
  - `channel_native_support_surface`.
- One counted **hybrid** continuity anchor:
  - `phase2_xxz_hea_q4_continuity`.
- One representative **hybrid** structured pilot case from the frozen
performance family inventory:
  - frozen pilot ID:
  `phase31_pair_repeat_q8_periodic_seed20260318` (replaced from the dense row
  so the pilot exercises mixed channel-native + Phase 3 routing).
- Route-attribution and route-coverage metadata sufficient to support later
`partition_route_summary` and performance route summaries.
- Existing **strict** path remains unchanged as the motif-proof regression
anchor.

**Out (deferred to later slices / full Task 3 / Task 4 closure)**

- `phase2_xxz_hea_q6_continuity`.
- Full external Aer slice from `P31-ADR-011`.
- Remaining counted strict microcases if not already promoted elsewhere:
  - `phase31_microcase_2q_multi_noise_entangler_chain`,
  - `phase31_microcase_2q_dense_same_support_motif`.
- Full version-bumped `correctness_evidence` / `performance_evidence` builder
migration under `P31-ADR-013`.
- Full counted structured matrix across all families, seeds, and noise patterns.
- `break_even_table` / `justification_map` closure across the whole frozen
matrix.
- Any Task 6 host-acceleration work.

## Recommended frozen IDs for this slice

**Counted hybrid continuity anchor**
`phase2_xxz_hea_q4_continuity`

**Representative hybrid structured pilot**
`phase31_pair_repeat_q8_periodic_seed20260318`

If implementation order forces a different structured pilot row, record the
replacement ID here and preserve the same intent: one case from the frozen
counted performance families must exercise the hybrid path with route coverage
and the baseline trio.

---

### Story P31-S07: Hybrid runtime executes whole workloads through explicit partition routing

**User/Research value**

- Makes the counted continuity and structured workload surface executable
without violating the no-silent-fallback contract.
- Preserves scientific auditability: reviewers can see where bounded
channel-native fusion actually applied and where the shipped Phase 3 exact
path still carried the workload.

**Given / When / Then**

- **Given** a partitioned noisy execution request with the frozen hybrid Phase
3.1 mode enabled, plus a workload containing both:
  - partitions that are fully eligible for bounded channel-native execution,
  - and partitions that are Phase-3-supported but Phase-3.1-ineligible.
- **When** the runtime executes that workload.
- **Then** eligible partitions execute through the channel-native path,
documented ineligible partitions execute through the shipped Phase 3 exact
path, unsupported-by-both partitions fail loudly, and route attribution is
emitted at partition granularity.

**Scope**

- **In:** explicit hybrid runtime entrypoint, route-policy enforcement,
partition-level route records, one positive continuity case, one representative
structured pilot, one unsupported-by-both negative case.
- **Out:** full correctness bundle wiring and full structured benchmark matrix.

**Acceptance signals**

- `phase2_xxz_hea_q4_continuity` executes through
`execute_partitioned_density_channel_native_hybrid(...)` with runtime-path
label `phase31_channel_native_hybrid`.
- The continuity result records at least:
  - one partition routed as `eligible_channel_native_motif`,
  - and one partition routed through the shipped Phase 3 exact path for a frozen
  route reason.
- One representative structured pilot case executes through the hybrid path and
records the same route-policy vocabulary.
- One unsupported-by-both case still fails loudly rather than being absorbed
into hybrid routing.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 2; **P31-C-02**, **P31-C-07**,
**P31-C-08**.
- ADRs: `P31-ADR-012`, `P31-ADR-013`, `P31-ADR-003`, `P31-ADR-005`.

#### Engineering tasks (Story P31-S07)

##### Engineering Task P31-S07-E01: Add explicit hybrid runtime entrypoint and partition-route policy

**Implementation plan:** `[ENGINEERING_TASK_P31_S07_E01_IMPLEMENTATION_PLAN.md](task-2/ENGINEERING_TASK_P31_S07_E01_IMPLEMENTATION_PLAN.md)`

**Implements story**

- Story P31-S07

**Change type**

- code | tests

**Definition of done**

- A distinct public helper and runtime label expose the hybrid whole-workload
path.
- Hybrid execution routes eligible partitions through the channel-native path
and Phase-3-supported but Phase-3.1-ineligible partitions through the shipped
Phase 3 exact path.
- Unsupported-by-both partitions still fail.

**Execution checklist**

- Add hybrid runtime identity without reopening `requested_mode`.
- Freeze and emit partition-route reasons under the `P31-ADR-012` contract.
- Keep strict `phase31_channel_native` behavior unchanged.

**Evidence produced**

- Positive hybrid execution on the `q4` continuity anchor.
- One negative hybrid failure that proves unsupported-by-both still errors.

**Risks / rollback**

- Risk: hybrid routing becomes an undocumented silent downgrade.
- Rollback/mitigation: require explicit partition-route records and keep strict
mode unchanged.

---

### Story P31-S08: One hybrid continuity anchor closes the next correctness loop

**User/Research value**

- Proves that the new strict-plus-hybrid contract is not only an API split but a
semantically exact whole-workload execution model.
- Establishes the first counted continuity row under the hybrid interpretation
before broader correctness-package migration.

**Given / When / Then**

- **Given** the hybrid runtime from Story P31-S07 and the frozen counted
continuity anchor `phase2_xxz_hea_q4_continuity`.
- **When** the next correctness gate runs.
- **Then** the hybrid output matches the sequential oracle within **P31-C-03**,
route attribution remains reviewable, and the slice explicitly records what
remains deferred (`q6`, Aer, full bundle migration, full matrix closure).

**Scope**

- **In:** one counted hybrid continuity anchor, internal exactness checks,
route-summary checks, documentation hook for remaining Task 3 work.
- **Out:** full `P31-ADR-011` Aer closure, `q6` continuity, full
`correctness_evidence` schema migration.

**Acceptance signals**

- `phase2_xxz_hea_q4_continuity` passes:
  - Frobenius difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)`,
  - stable hybrid route summary.
- The doc record names the deferred correctness work:
  - `phase2_xxz_hea_q6_continuity`,
  - Aer on the frozen external slice,
  - full `correctness_evidence` package migration.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 3; **P31-C-03**,
**P31-C-04** (subset), **P31-C-05** (future), **P31-C-08** (directional).
- ADRs: `P31-ADR-009`, `P31-ADR-011`, `P31-ADR-013`.

#### Engineering tasks (Story P31-S08)

##### Engineering Task P31-S08-E01: Add the counted q4 hybrid continuity correctness gate

**Implementation plan:** `[ENGINEERING_TASK_P31_S08_E01_IMPLEMENTATION_PLAN.md](task-3/ENGINEERING_TASK_P31_S08_E01_IMPLEMENTATION_PLAN.md)`

**Implements story**

- Story P31-S08

**Change type**

- code | tests | docs

**Definition of done**

- The counted `q4` continuity anchor executes through the hybrid path and
matches the sequential oracle under the frozen numeric policy.
- Hybrid route attribution is asserted as part of the correctness evidence.
- Deferred Task 3 items remain explicit.

**Execution checklist**

- Add one end-to-end hybrid correctness module or section for
`phase2_xxz_hea_q4_continuity`.
- Assert full-density exactness and route-summary stability.
- Record what remains deferred after this gate.

**Evidence produced**

- Passing hybrid continuity test or harness output.
- Reviewer-facing status note naming deferred Task 3 items.

**Risks / rollback**

- Risk: continuity execution passes numerically but route attribution is too weak
to support later publication claims.
- Rollback/mitigation: require route-summary assertions in the same gate.

---

### Story P31-S09: One representative hybrid structured pilot closes the next performance-design loop

**User/Research value**

- Converts the hybrid runtime from a correctness-only concept into the first
measured whole-workload performance object.
- Provides the minimum evidence needed to design the broader Task 4 matrix
honestly instead of speculating about performance closure.

**Given / When / Then**

- **Given** Stories P31-S07 and P31-S08 implemented and one representative pilot
row from the frozen counted performance families.
- **When** the structured pilot benchmark runs.
- **Then** the case records the baseline trio, route coverage, and one initial
diagnosis tag, while the docs keep full Task 4 closure explicitly out of
scope.

**Scope**

- **In:** one representative primary-family hybrid pilot row,
baseline-trio timing, route coverage, diagnosis tag, doc record of what
remains for full Task 4 closure.
- **Out:** full counted structured matrix, full `break_even_table`, and final
positive-method closure.

**Acceptance signals**

- `phase31_pair_repeat_q8_periodic_seed20260318` (or a documented replacement
frozen ID) runs through the hybrid path with nonzero channel-native coverage.
- The pilot row records:
  - sequential baseline,
  - Phase 3 fused baseline,
  - hybrid Phase 3.1 baseline,
  - route coverage,
  - one diagnosis classification.
- The doc record names the remaining deferred Task 4 work:
  - the remaining counted family/seed/pattern rows,
  - full `break_even_table` / `justification_map`,
  - control-family closure.

**Traceability**

- Phase: `DETAILED_PLANNING_PHASE_3_1.md` §8 Task 4; **P31-C-06**,
**P31-C-08** (directional), **P31-C-09**.
- ADRs: `P31-ADR-010`, `P31-ADR-013`, `P31-ADR-014`.

#### Engineering tasks (Story P31-S09)

##### Engineering Task P31-S09-E01: Add one representative hybrid structured benchmark pilot

**Implementation plan:** `[ENGINEERING_TASK_P31_S09_E01_IMPLEMENTATION_PLAN.md](task-4/ENGINEERING_TASK_P31_S09_E01_IMPLEMENTATION_PLAN.md)`

**Implements story**

- Story P31-S09

**Change type**

- benchmark harness | docs

**Definition of done**

- One representative primary-family structured case executes through the hybrid
path and records the baseline trio plus route coverage.
- The result is explicitly framed as a pilot row, not full Task 4 closure.

**Execution checklist**

- Freeze the pilot case ID (frozen:
`phase31_pair_repeat_q8_periodic_seed20260318`) or document the replacement.
- Record timings for sequential, Phase 3 fused, and hybrid Phase 3.1.
- Emit route-coverage counters and one diagnosis tag.
- Document the remaining full Task 4 matrix work.

**Evidence produced**

- One reproducible pilot benchmark row.
- Reviewer-facing note explaining what remains deferred for Task 4.

**Risks / rollback**

- Risk: the pilot row is mistaken for full performance closure.
- Rollback/mitigation: keep one frozen pilot ID and explicitly name the
remaining counted matrix work in the status section.

---

## Story-to-engineering map


| Story   | Engineering tasks |
| ------- | ----------------- |
| P31-S07 | P31-S07-E01       |
| P31-S08 | P31-S08-E01       |
| P31-S09 | P31-S09-E01       |


## Suggested implementation order

1. P31-S07-E01
2. P31-S08-E01
3. P31-S09-E01

Parallelism:

- Task 3 design can begin once the hybrid route-policy vocabulary stabilizes.
- The structured pilot harness can be prepared in parallel with hybrid runtime
implementation, but should not be treated as claim-bearing until the hybrid
continuity gate is green.

---

## Slice implementation status

In-scope third vertical slice (Stories P31-S07–S09) is **complete** in code and
tests. Full Task 3 / Task 4 closure items in the table below remain explicitly
out of scope for this slice.


| Field                                          | Value                                                                                                                                                          |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hybrid continuity anchor                       | `phase2_xxz_hea_q4_continuity`                                                                                                                                 |
| Hybrid structured pilot                        | `phase31_pair_repeat_q8_periodic_seed20260318` (frozen)                                                                                                        |
| Third vertical slice implementation (in-scope) | **Complete**                                                                                                                                                   |
| Primary hybrid correctness + routing evidence  | `tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py`                                                                                   |
| Primary hybrid pilot evidence                  | `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`; CLI `benchmarks/density_matrix/performance_evidence/phase31_hybrid_pilot_validation.py` |
| Deferred Task 3 work after this slice          | `phase2_xxz_hea_q6_continuity`, full Aer slice, full `correctness_evidence` migration                                                                          |
| Deferred Task 4 work after this slice          | remaining counted structured rows, control-family closure, full `break_even_table` / `justification_map`                                                       |


