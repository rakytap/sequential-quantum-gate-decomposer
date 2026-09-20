# Pre-Publication Evidence Review for Phase 3.1

This document is the formal pre-publication evidence review required by
`CLOSURE_PLAN_PHASE_3_1.md` Package C and by `task-5/TASK_5_MINI_SPEC.md`.

Its role is to record the current review state of the frozen Phase 3.1 v1 slice
before any venue-ready publication closure is attempted. It is a phase-local
evidence artifact, not a paper draft.

## Review scope and authority

This review is authoritative only for the frozen Phase 3.1 v1 support slice
defined by:

- `DETAILED_PLANNING_PHASE_3_1.md`,
- `ADRs_PHASE_3_1.md`,
- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
- `task-3/TASK_3_MINI_SPEC.md`,
- `task-4/TASK_4_MINI_SPEC.md`,
- `task-5/TASK_5_MINI_SPEC.md`,
- and `CLOSURE_PLAN_PHASE_3_1.md`.

It does **not** reopen:

- the support matrix,
- the positive-method threshold,
- the required external slice,
- or the counted workload inventory.

## Review state

**Recorded closure state:** `decision-study-ready`

## Why this is the current state

Phase 3.1 is no longer a planning-only branch. The repo already contains a
substantial implementation-backed bounded result:

- the strict `phase31_channel_native` path exists,
- the hybrid `phase31_channel_native_hybrid` path exists,
- the bounded counted correctness package is present on the frozen six-row
  counted slice,
- the required five-row external slice is present on the current Stage-A path,
- and one structured hybrid pilot row already provides initial whole-workload
  decision evidence.

However, the phase still does **not** satisfy the closure conditions for
`positive-methods-ready`, because the full frozen counted matrix now shows no
`phase31_justified` rows. The emitted matrix instead supports a bounded
decision-study closure.

Therefore the current review outcome is:

- **not a positive-methods closure,**
- **yes a bounded decision-study closure,**
- with **correctness green on the frozen bounded slice and matrix-level
  performance evidence showing where Phase 3 remains sufficient and where Phase
  3.1 is still not justified yet.**

## Closure-state rubric check

| Closure state | Required condition | Current verdict | Reason |
|---|---|---|---|
| `positive-methods-ready` | Mandatory correctness slice green; required external slice green; full counted performance matrix emitted; `break_even_table` / `justification_map` emitted; at least one representative primary-family row meets the frozen positive threshold | **No** | The counted matrix is now emitted, but it contains `0` `phase31_justified` rows and therefore does not satisfy the frozen positive-method rule |
| `decision-study-ready` | Mandatory correctness slice green; full counted performance matrix emitted; decision artifact emitted; evidence supports a bounded negative or mixed whole-workload conclusion | **Yes** | The counted matrix and decision artifact are now emitted; the current result shows `17` `phase3_sufficient` rows, `9` `phase31_not_justified_yet` rows, and `0` `phase31_justified` rows |
| `not-ready-yet` | Any mandatory correctness or performance obligation remains incomplete, contradictory, or blocked | **No** | The full `P31-ADR-010` matrix and its machine-readable decision artifact are now present |

## Review table: frozen slice status

### A. Correctness and external-reference closure

| Evidence area | Frozen requirement | Current status | Primary evidence anchor | Review note |
|---|---|---|---|---|
| Counted bounded correctness package | 4 strict microcases + 2 hybrid continuity anchors | **Green** | `tests/partitioning/evidence/test_phase31_correctness_evidence.py` | Test asserts `len(ctxs) == 6`, `microcases == 4`, `continuity_cases == 2`, and `counted_supported_cases == 6` |
| Counted case metadata and schema | Phase 3.1 case/package/summary schema present | **Green** | same file | Current tests assert `CORRECTNESS_EVIDENCE_PHASE31_CASE_SCHEMA_VERSION`, package schema, and summary schema |
| Required external slice | 4 strict microcases + `phase2_xxz_hea_q4_continuity` | **Green** | same file | Test asserts external bundle `total_cases == 5`, `microcases == 4`, `continuity_cases == 1` |
| Channel invariants and route summaries | Strict microcases carry invariants; hybrid rows carry route summary | **Green** | same file | Tests assert `channel_invariants` on microcases and `partition_route_summary` on continuity rows |
| Strict runtime exactness interpretation | strict path carries bounded fused-object claim | **Green** | publication surfaces + current tests | Current evidence supports the strict object on the frozen counted microcase surface |
| Hybrid continuity exactness interpretation | hybrid path carries counted continuity rows | **Green** | `SHORT_PAPER_PHASE_3_1.md`, current tests | Both `q4` and `q6` continuity anchors are now treated as implemented and validated on the bounded slice |

### B. Whole-workload performance closure

| Evidence area | Frozen requirement | Current status | Primary evidence anchor | Review note |
|---|---|---|---|---|
| Full `P31-ADR-010` counted performance matrix | 26 counted Phase 3.1 rows | **Green** | `tests/partitioning/evidence/test_phase31_counted_matrix_validation.py`; `phase31_counted_matrix_bundle.json` | Focused validation now asserts `26` total rows with the frozen `24` primary + `2` control split |
| Baseline trio on all counted rows | sequential, Phase 3 fused, Phase 3.1 hybrid | **Green** | same file | The bundle summary asserts baseline-trio presence across all counted rows |
| Route-coverage metadata on counted rows | route counters and route records | **Green** | same file | The bundle summary asserts route-field presence across all counted rows |
| Decision artifact | `break_even_table` / `justification_map` | **Green** | same file; `phase31_counted_matrix_bundle.json` | The bundle now emits both matrix-wide artifacts and a machine-readable decision summary |
| Matrix-level closure mode | positive methods or decision study | **Decision-study-ready** | this review + counted matrix bundle | Matrix-level counts are `17` `phase3_sufficient`, `9` `phase31_not_justified_yet`, `0` `phase31_justified` |

## Case-indexed evidence summary

### Current counted correctness boundary

| Case class | Case IDs / slice | Runtime interpretation | Current review status |
|---|---|---|---|
| Strict counted microcases | `phase31_microcase_1q_u3_local_noise_chain`, `phase31_microcase_2q_cnot_local_noise_pair`, `phase31_microcase_2q_multi_noise_entangler_chain`, `phase31_microcase_2q_dense_same_support_motif` | strict `phase31_channel_native` | **Implemented and validated** |
| Hybrid counted continuity anchors | `phase2_xxz_hea_q4_continuity`, `phase2_xxz_hea_q6_continuity` | hybrid `phase31_channel_native_hybrid` | **Implemented and validated** |
| Required external-reference slice | four strict microcases + `phase2_xxz_hea_q4_continuity` | strict + hybrid | **Implemented and validated** |

### Current whole-workload performance boundary

| Evidence unit | Workload / slice | Runtime interpretation | Current review status |
|---|---|---|---|
| Structured pilot row | `phase31_pair_repeat_q8_periodic_seed20260318` | hybrid `phase31_channel_native_hybrid` | **Implemented and validated as one member of the counted matrix** |
| Full counted matrix inventory | 26 counted rows from the frozen `P31-ADR-010` slice | hybrid `phase31_channel_native_hybrid` | **Implemented and validated** |

## Current evidence-backed answer to Side Paper A

The Side Paper A question is:

> When does the Phase 3 noise-aware partitioning baseline stop being enough, and
> when would more invasive channel-native fusion become justified?

### Current answer status

**Bounded decision-study answer; claim-closing for the decision-study mode, but
not for the positive-methods mode.**

### What the current evidence already supports

- Phase 3 is not sufficient to express the bounded exact fused object on the
  frozen mixed gate+noise motif slice, because repeated local noise insertions
  can fragment a same-support motif that still admits exact channel-native
  composition.
- That bounded fused object is already **implemented and validated** under the
  strict path on the counted microcase surface.
- The hybrid path is already **implemented and validated** on the counted
  continuity anchors and is therefore not only hypothetical.
- The full counted matrix now indicates that mathematical feasibility does
  **not** automatically imply workload-level advantage over the shipped Phase 3
  fused baseline.
- The current matrix-wide result is mixed and negative-to-non-proceed for the
  stronger methods claim:
  - `17` rows classify as `phase3_sufficient`,
  - `9` rows classify as `phase31_not_justified_yet`,
  - `0` rows classify as `phase31_justified`.

### What the current evidence does not yet support

- A positive-methods claim against the Phase 3 fused baseline.
- Any statement stronger than the current bounded decision-study outcome on the
  frozen slice.

## Blockers to publication closure

The remaining blockers are now narrower:

1. **Positive-methods threshold not met**
   - The emitted counted matrix shows `0` `phase31_justified` rows under the
     frozen rule.

2. **Task 5 publication closure still pending**
   - The review state is now decision-study-ready, but the paper surfaces still
     need their final Task 5 decision-study closure wording.

3. **Program-level sync still pending**
   - Top-level docs still need to be updated after Task 5 so the repo tells one
     consistent story.

## Recommended next actions

The next actions now are:

1. **Proceed to Task 5 publication closure**
   - The recorded state is `decision-study-ready`, so the publication surfaces
     may now be tightened to the bounded decision-study mode.

2. **Keep the positive-methods branch closed unless new evidence changes the matrix**
   - Any attempt to reopen the stronger claim would require new emitted evidence,
     not reinterpretation of the current matrix.

3. **Then execute program-level sync**
   - Update `PLANNING.md`, `CHANGELOG.md`, and any publication-strategy pointers
     so the repo reflects the recorded decision-study-ready state.

## Publication-writing implications of the current review state

Because the current recorded state is `decision-study-ready`:

- the publication surfaces may now describe the bounded exactness result and the
  full counted matrix-level decision-study outcome,
- they may state explicitly that the stronger positive-methods closure was not
  met on the frozen slice,
- they may proceed to Task 5 decision-study framing,
- but they must **not** present Phase 3.1 as a positive-methods success,
- and they must **not** upgrade the result beyond the emitted decision-study
  evidence.

## Review evidence anchors

- `docs/density_matrix_project/phases/phase-3-1/CLOSURE_PLAN_PHASE_3_1.md`
- `docs/density_matrix_project/phases/phase-3-1/task-5/TASK_5_MINI_SPEC.md`
- `tests/partitioning/evidence/test_phase31_correctness_evidence.py`
- `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`
- `tests/partitioning/evidence/test_phase31_counted_matrix_validation.py`
- `benchmarks/density_matrix/artifacts/correctness_evidence/phase31_stage_a/correctness_package/phase31_correctness_package_bundle.json`
- `benchmarks/density_matrix/artifacts/correctness_evidence/phase31_stage_a/external_correctness/phase31_external_correctness_bundle.json`
- `benchmarks/density_matrix/artifacts/performance_evidence/phase31_counted_matrix/phase31_counted_matrix_bundle.json`

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`
- `task-3/TASK_3_MINI_SPEC.md`
- `task-4/TASK_4_MINI_SPEC.md`
- `task-5/TASK_5_MINI_SPEC.md`
- `CLOSURE_PLAN_PHASE_3_1.md`
