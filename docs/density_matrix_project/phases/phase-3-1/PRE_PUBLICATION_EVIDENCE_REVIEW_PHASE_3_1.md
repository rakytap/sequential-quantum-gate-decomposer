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

**Recorded closure state:** `not-ready-yet`

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

However, the phase does **not** yet satisfy the closure conditions for either
`positive-methods-ready` or `decision-study-ready`, because the full frozen
performance matrix and its decision artifact are still missing.

Therefore the current review outcome is:

- **not a positive-methods closure,**
- **not yet a decision-study closure,**
- but **an implementation-backed bounded result with correctness substantially
  ahead of performance closure.**

## Closure-state rubric check

| Closure state | Required condition | Current verdict | Reason |
|---|---|---|---|
| `positive-methods-ready` | Mandatory correctness slice green; required external slice green; full counted performance matrix emitted; `break_even_table` / `justification_map` emitted; at least one representative primary-family row meets the frozen positive threshold | **No** | The correctness and external slices are present, but the full counted performance matrix and decision artifact are not yet emitted |
| `decision-study-ready` | Mandatory correctness slice green; full counted performance matrix emitted; decision artifact emitted; evidence supports a bounded negative or mixed whole-workload conclusion | **No** | One pilot row exists, but one row is not the full matrix and cannot close the whole-workload question |
| `not-ready-yet` | Any mandatory correctness or performance obligation remains incomplete, contradictory, or blocked | **Yes** | The full `P31-ADR-010` matrix and its decision artifact remain outstanding |

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
| Full `P31-ADR-010` counted performance matrix | 26 counted Phase 3.1 rows | **Blocked / incomplete** | `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py` | Inventory helpers define the full 26-row matrix, but the current dedicated emitted artifact is still the single pilot row |
| Baseline trio on all counted rows | sequential, Phase 3 fused, Phase 3.1 hybrid | **Partial** | same file | The pilot row records the full baseline trio, but matrix-wide emission is not yet present |
| Route-coverage metadata on counted rows | route counters and route records | **Partial** | same file | Present on the pilot row; not yet present across the full counted matrix |
| Decision artifact | `break_even_table` / `justification_map` | **Missing** | planning and mini-specs | No emitted matrix-wide decision artifact has been recorded yet |
| Matrix-level closure mode | positive methods or decision study | **Not decidable yet** | this review | One pilot row is informative but insufficient for matrix-level closure |

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
| Structured pilot row | `phase31_pair_repeat_q8_periodic_seed20260318` | hybrid `phase31_channel_native_hybrid` | **Implemented and partially validated** |
| Full counted matrix inventory | 26 counted rows from the frozen `P31-ADR-010` slice | hybrid `phase31_channel_native_hybrid` | **Planned inventory exists, emitted closure package still missing** |

## Current evidence-backed answer to Side Paper A

The Side Paper A question is:

> When does the Phase 3 noise-aware partitioning baseline stop being enough, and
> when would more invasive channel-native fusion become justified?

### Current answer status

**Partial answer only; not yet claim-closing.**

### What the current evidence already supports

- Phase 3 is not sufficient to express the bounded exact fused object on the
  frozen mixed gate+noise motif slice, because repeated local noise insertions
  can fragment a same-support motif that still admits exact channel-native
  composition.
- That bounded fused object is already **implemented and validated** under the
  strict path on the counted microcase surface.
- The hybrid path is already **implemented and validated** on the counted
  continuity anchors and is therefore not only hypothetical.
- The first structured hybrid pilot row already indicates that mathematical
  feasibility does **not** automatically imply workload-level advantage over the
  shipped Phase 3 fused baseline.

### What the current evidence does not yet support

- A matrix-level statement about where Phase 3.1 is justified across the frozen
  8/10-qubit workload families.
- A positive-methods claim against the Phase 3 fused baseline.
- A closed decision-study claim across the full frozen performance slice.

## Blockers to publication closure

The current blockers are:

1. **Full counted performance matrix not yet emitted**
   - The frozen 26-row matrix is defined, but the current dedicated emitted
     evidence surface still centers on the pilot row.

2. **Decision artifact not yet emitted**
   - `break_even_table` / `justification_map` is still required to classify:
     - `phase3_sufficient`,
     - `phase31_justified`,
     - `phase31_not_justified_yet`.

3. **Closure-state decision cannot yet be upgraded**
   - Without the full matrix and decision artifact, the phase cannot be promoted
     to either `positive-methods-ready` or `decision-study-ready`.

## Recommended next actions

The next actions remain exactly those specified by the closure playbook:

1. **Finish Package B / Story P31-S11**
   - emit the full frozen 26-row counted matrix,
   - emit route-aware summaries on the counted rows,
   - emit the matrix-wide decision artifact.

2. **Re-run this review**
   - upgrade the closure state only after the matrix-level artifact exists.

3. **Then execute Task 5 publication closure**
   - only if the revised review state is:
     - `positive-methods-ready`, or
     - `decision-study-ready`.

## Publication-writing implications of the current review state

Because the current recorded state is `not-ready-yet`:

- the publication surfaces may describe the bounded exactness result and the
  current pilot row,
- they may describe the current state as a **pre-closure boundary-sync draft**,
- they may explicitly state that the full structured matrix and decision
  artifact remain the claim-closing gate,
- but they must **not** present Phase 3.1 as submission-ready,
- and they must **not** claim either a closed positive-methods outcome or a
  closed matrix-level decision-study outcome.

## Review evidence anchors

- `docs/density_matrix_project/phases/phase-3-1/CLOSURE_PLAN_PHASE_3_1.md`
- `docs/density_matrix_project/phases/phase-3-1/task-5/TASK_5_MINI_SPEC.md`
- `tests/partitioning/evidence/test_phase31_correctness_evidence.py`
- `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`
- `benchmarks/density_matrix/artifacts/correctness_evidence/phase31_stage_a/correctness_package/phase31_correctness_package_bundle.json`
- `benchmarks/density_matrix/artifacts/correctness_evidence/phase31_stage_a/external_correctness/phase31_external_correctness_bundle.json`

## Traceability

- `DETAILED_PLANNING_PHASE_3_1.md`
- `ADRs_PHASE_3_1.md`
- `task-3/TASK_3_MINI_SPEC.md`
- `task-4/TASK_4_MINI_SPEC.md`
- `task-5/TASK_5_MINI_SPEC.md`
- `CLOSURE_PLAN_PHASE_3_1.md`
