# Engineering Task P31-S12-E01 Implementation Plan

## Engineering Task Being Implemented

`Engineering Task P31-S12-E01: Produce the pre-publication evidence review pack and go/no-go summary`

This is a Layer 4 file-level implementation plan for the engineering task under
Story `P31-S12` from
`../SIXTH_VERTICAL_SLICE_PRE_PUBLICATION_EVIDENCE_REVIEW_STORIES_AND_ENGINEERING_TASKS.md`.
It turns the closure-review wording into a concrete plan against the existing
Phase 3.1 evidence and publication surfaces in:

- `task-5/TASK_5_MINI_SPEC.md`,
- `../CLOSURE_PLAN_PHASE_3_1.md`,
- `../PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`,
- `../SHORT_PAPER_PHASE_3_1.md`,
- `../SHORT_PAPER_NARRATIVE.md`,
- `../ABSTRACT_PHASE_3_1.md`,
- `../PAPER_PHASE_3_1.md`,
- and, for current evidence anchors only,
  `tests/partitioning/evidence/test_phase31_correctness_evidence.py`,
  `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`,
  and the current Stage-A correctness bundles under
  `benchmarks/density_matrix/artifacts/correctness_evidence/phase31_stage_a/`.

## Scope

This engineering task produces the **review artifact** that determines whether
Phase 3.1 can proceed to publication closure:

- summarize the current frozen counted correctness boundary,
- summarize the current whole-workload performance boundary,
- record the current closure state using the frozen rubric:
  - `positive-methods-ready`,
  - `decision-study-ready`,
  - `not-ready-yet`,
- distinguish:
  - implemented and validated,
  - implemented but not yet claim-closing,
  - still missing for closure,
- tie the recorded state to concrete evidence anchors and blockers,
- and define the next permitted action under the closure playbook.

Out of scope for this engineering task:

- rewriting the paper surfaces beyond synchronization wording,
- inventing new evidence that does not exist in emitted artifacts or stable test
  surfaces,
- upgrading the closure state without the required matrix-level performance
  artifact,
- and program-level doc sync (`PLANNING.md`, `CHANGELOG.md`, `PUBLICATIONS.md`).

## Current Gap To Close

The repo now has a formal closure playbook and aligned publication surfaces, but
before this task lands there is no single reviewer-facing artifact that states:

- what the current closure state is,
- why it is that state,
- which frozen obligations are green,
- which obligations remain open,
- and what work is allowed next.

That creates two risks:

1. publication surfaces drift into interpretation without a recorded go / no-go
   state,
2. correctness progress and performance incompleteness are discussed in prose but
   not captured in one artifact-indexed review pack.

`P31-S12-E01` closes that gap.

## Dependencies And Assumptions

- The source-of-truth contract remains:
  - `TASK_5_MINI_SPEC.md`,
  - `../CLOSURE_PLAN_PHASE_3_1.md`,
  - `../DETAILED_PLANNING_PHASE_3_1.md`,
  - `../ADRs_PHASE_3_1.md`.
- The current state after the bounded correctness work is:
  - correctness substantially ahead of performance closure,
  - one emitted hybrid pilot row present,
  - full counted performance matrix and decision artifact not yet emitted.
- This task must treat the current review as **artifact-indexed**, not as a free
  narrative summary.
- If the full matrix is not yet emitted, the review must remain
  `not-ready-yet`.

## Target Files And Responsibilities

### Primary review artifact: `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`

This file becomes the formal Package C artifact.

#### What it should contain

- an explicit recorded closure state,
- a rubric table checking all three closure states,
- a correctness/external review table,
- a performance review table,
- a case-indexed summary of the current bounded evidence,
- the current partial answer to the Side Paper A question,
- blockers to closure,
- recommended next actions,
- and publication-writing implications of the recorded state.

#### What it must avoid

- venue-specific selling language,
- implementation detail that does not change the review state,
- or claims not traceable to the evidence anchors.

### Synchronized publication surfaces

The technical short paper, full paper, abstract, and narrative companion may
need **minor synchronization edits** after the review file lands, but only to:

- acknowledge that the review now exists,
- state the recorded closure state correctly,
- and keep the current surfaces boundary-synchronized rather than venue-ready.

They should **not** be rewritten as if the phase had already passed the review.

## Evidence Model For This Task

### Correctness evidence to summarize

Use the current bounded correctness evidence as the basis for the correctness
side of the review:

- `tests/partitioning/evidence/test_phase31_correctness_evidence.py`,
- the bounded Stage-A correctness package,
- and the bounded external slice bundle.

Key facts this task should make reviewable:

- bounded counted correctness package = 6 rows,
- 4 strict microcases + 2 hybrid continuity anchors,
- external required slice = 5 rows,
- invariants and route summaries present on the correct row classes.

### Performance evidence to summarize

Use the current pilot-only performance evidence as the basis for the performance
side of the review:

- `tests/partitioning/evidence/test_phase31_hybrid_pilot_validation.py`,
- the frozen pilot workload ID,
- the frozen 26-row inventory helper,
- and the absence of a matrix-wide decision artifact.

Key facts this task should make reviewable:

- matrix inventory exists,
- pilot row exists and is informative,
- matrix-wide emission is still missing,
- therefore closure cannot be upgraded.

## Recommended Writing Structure

1. **State the review role** and frozen authority.
2. **Record the closure state** immediately and unambiguously.
3. **Explain why** in one short section.
4. **Run the rubric check** against all three states.
5. **Summarize correctness vs. performance** in separate tables.
6. **Answer the Side Paper A question partially**, not conclusively.
7. **List blockers** as actionable items.
8. **State the next permitted action**.
9. **Describe publication-writing implications** of the current state.

## Definition Of Done Mapping

This engineering task is done when:

- `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md` exists and records exactly one
  of the allowed closure states,
- the state is backed by concrete evidence anchors and blocker logic,
- the publication surfaces can point to that state without contradiction,
- and the task does not quietly smuggle in publication closure before the review
  says it is allowed.

## Risks / Rollback

- Risk: the review drifts into manuscript argument instead of evidence review.
  - Mitigation: keep all sections artifact-indexed and case-ID-indexed.
- Risk: the state is upgraded prematurely from one pilot row.
  - Mitigation: keep the rubric explicit that matrix-wide emission is required.
- Risk: the review duplicates the papers instead of governing them.
  - Mitigation: keep publication-writing implications short and subordinate to
    the evidence tables.
