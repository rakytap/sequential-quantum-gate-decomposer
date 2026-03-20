# Story 6 Implementation Plan

## Story Being Implemented

Story 6: Unsupported-Boundary Evidence Preserves Stage Separation

This is a Layer 4 engineering plan for implementing the sixth behavioral slice
from `TASK_6_STORIES.md`.

## Scope

This story turns Task 6 negative evidence into one explicit stage-separated
boundary surface:

- planner-entry, descriptor-generation, and runtime-stage unsupported or
  deferred cases remain explicitly distinguishable,
- failure stage, unsupported category, first unsupported condition, provenance,
  and exclusion reason are preserved in a machine-reviewable way,
- negative evidence remains structurally comparable to positive Task 6 evidence
  where fields overlap,
- and Story 6 closes the contract for "how unsupported-boundary evidence is
  represented" without yet claiming full-package assembly or summary-consistency
  closure.

Out of scope for this story:

- correctness-matrix inventory already owned by Story 1,
- internal sequential-baseline gating already owned by Story 2,
- the bounded Qiskit Aer slice already owned by Story 3,
- output-integrity and continuity-energy emphasis already owned by Story 4,
- runtime and fusion classification comparability already owned by Story 5,
- full correctness-package assembly already owned by Story 7,
- and counted-status propagation into later summaries already owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 5 already define the positive Task 6 surfaces Story 6 must
  protect and complement.
- The frozen source-of-truth contract is `TASK_6_MINI_SPEC.md`,
  `TASK_6_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`,
  `P3-ADR-004`, `P3-ADR-005`, and `P3-ADR-008`.
- Earlier Phase 3 tasks already provide stage-specific negative-evidence
  surfaces Story 6 should align rather than replace:
  - planner-entry unsupported evidence in
    `benchmarks/density_matrix/planner_surface/unsupported_planner_validation.py`,
  - descriptor-generation unsupported evidence in
    `benchmarks/density_matrix/planner_surface/unsupported_descriptor_validation.py`,
  - and runtime-stage unsupported evidence in
    `benchmarks/density_matrix/partitioned_runtime/unsupported_runtime_validation.py`.
- Earlier task plans also establish the structured unsupported vocabulary
  precedent Story 6 should preserve where practical:
  - `NoisyPlannerValidationError`,
  - `NoisyDescriptorValidationError`,
  - and the Task 3 runtime-stage failure taxonomy.
- The current implementation learning is that Story 6 should normalize those
  existing Task 1 through Task 3 negative records into one shared Task 6
  unsupported-boundary surface, instead of regenerating a separate negative case
  matrix from scratch.
- Story 6 should prefer one shared cross-stage negative-evidence record over one
  incompatible artifact shape per failure stage.
- Negative evidence is a required output of Task 6, not an appendix to positive
  cases.
- The natural implementation home for Task 6 unsupported-boundary validators is
  the new `benchmarks/density_matrix/correctness_evidence/` package, with
  `unsupported_boundary_validation.py` reading and normalizing the earlier
  stage-specific negative record surfaces.

## Engineering Tasks

### Engineering Task 1: Freeze The Cross-Stage Unsupported-Boundary Taxonomy For Task 6

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- docs | validation automation

**Definition of done**
- Story 6 defines one stable unsupported-boundary taxonomy spanning the three
  Task 6 stage categories.
- The taxonomy is explicit enough that later Task 7 and Task 8 work can reuse
  it safely.
- Stage separation remains visible rather than collapsing into one generic
  failure bucket.

**Execution checklist**
- [ ] Freeze the minimum Task 6 stage categories: planner-entry unsupported,
      descriptor-generation unsupported or lossy, and runtime-stage unsupported
      or deferred.
- [ ] Freeze one shared vocabulary for failure stage, unsupported category, first
      unsupported condition, provenance, and exclusion reason.
- [ ] Align overlapping category meanings with the Task 1, Task 2, and Task 3
      negative-evidence vocabularies where practical.
- [ ] Keep full-package assembly and summary-consistency semantics outside the
      Story 6 bar.

**Evidence produced**
- One stable Task 6 unsupported-boundary taxonomy.
- One reviewable cross-stage failure vocabulary for later consumers.

**Risks / rollback**
- Risk: later reviewers may be unable to tell why a case was excluded if Story 6
  leaves negative evidence spread across incompatible task-level vocabularies.
- Rollback/mitigation: freeze one Task 6 cross-stage taxonomy before packaging
  negative evidence.

### Engineering Task 2: Reuse The Existing Task 1 Through Task 3 Negative-Evidence Surfaces As The Base

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- docs | code

**Definition of done**
- Story 6 reuses the existing Task 1, Task 2, and Task 3 negative-evidence
  surfaces where they already match the contract.
- The Task 6 unsupported-boundary surface remains auditable back to the actual
  stage where a case failed.
- Story 6 avoids inventing a detached Task 6-only negative language.

**Execution checklist**
- [ ] Reuse overlapping fields from the Task 1 planner-entry, Task 2
      descriptor-generation, and Task 3 runtime-stage negative records where
      they already fit Story 6.
- [ ] Keep stage identity explicit rather than flattening all earlier failures
      into one Task 6 status.
- [ ] Add only the Task 6-specific exclusion and carry-forward fields needed for
      later package assembly.
- [ ] Document where Story 6 intentionally extends earlier negative-evidence
      vocabularies.

**Evidence produced**
- One reviewable mapping from Task 1 through Task 3 negative surfaces to the
  Task 6 unsupported-boundary surface.
- One explicit boundary between reused negative-evidence fields and Task 6-
  specific extensions.

**Risks / rollback**
- Risk: Story 6 may create a disconnected negative-evidence language that later
  reviewers must translate manually against the real failure stage.
- Rollback/mitigation: align Story 6 with earlier negative surfaces wherever
  practical.

### Engineering Task 3: Define A Shared Task 6 Unsupported-Boundary Record And Validation Harness

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- code | validation automation

**Definition of done**
- Story 6 defines one shared Task 6 unsupported-boundary record shape.
- The record carries stage identity, unsupported category, first unsupported
  condition, provenance, exclusion reason, and join keys into the positive Task
  6 case surface.
- Story 6 has one reusable harness for validating that structure.

**Execution checklist**
- [ ] Add a shared Task 6 unsupported-boundary record in
      `benchmarks/density_matrix/correctness_evidence/records.py` or the
      smallest auditable successor.
- [ ] Add a dedicated Story 6 validation driver under
      `benchmarks/density_matrix/correctness_evidence/`, with
      `unsupported_boundary_validation.py` as the primary checker.
- [ ] Keep the record compatible with the shared Task 6 case identity where
      fields overlap.
- [ ] Keep the harness rooted in machine-reviewable negative records rather than
      in prose-only interpretation.

**Evidence produced**
- One stable Story 6 unsupported-boundary record shape.
- One reusable Story 6 negative-evidence harness.

**Risks / rollback**
- Risk: negative evidence may remain structurally inconsistent across stages and
  hard to package honestly later.
- Rollback/mitigation: define one shared record and harness before broadening
  Story 6 matrices.

### Engineering Task 4: Align Negative Records With The Shared Positive Task 6 Provenance Surface

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- code | tests

**Definition of done**
- Unsupported-boundary records align with the shared positive Task 6 provenance
  surface where fields overlap.
- Later Story 7 package assembly can compare counted positive and excluded
  negative cases without manual relabeling.
- Story 6 avoids creating an error-only reporting format that cannot be joined
  back to the matrix.

**Execution checklist**
- [ ] Reuse shared fields such as workload family, workload ID, requested mode,
      source type, and entry route on negative records where they still apply.
- [ ] Add explicit exclusion-reason and stage-specific unsupported fields
      alongside the shared provenance tuple.
- [ ] Document how unsupported-boundary records relate to the shared Story 1
      case inventory and Story 2 counted-status surface.
- [ ] Add focused regression checks for join-key and provenance stability.

**Evidence produced**
- One aligned Task 6 negative-evidence record shape.
- One explicit mapping between positive and negative Task 6 record surfaces.

**Risks / rollback**
- Risk: unsupported-boundary evidence may remain plausible in isolation while
  staying incomparable to the positive Task 6 package.
- Rollback/mitigation: align overlapping provenance fields from the start.

### Engineering Task 5: Add A Representative Cross-Stage Unsupported-Boundary Matrix

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- tests | validation automation

**Definition of done**
- Story 6 covers representative excluded cases across planner-entry,
  descriptor-generation, and runtime-stage boundaries.
- The matrix is broad enough to prove the Task 6 exclusion surface is truly
  stage-separated.
- The matrix remains representative and contract-driven rather than exhaustive
  over every impossible request.

**Execution checklist**
- [ ] Include at least one representative planner-entry unsupported case.
- [ ] Include at least one representative descriptor-generation unsupported or
      lossy case.
- [ ] Include at least one representative runtime-stage unsupported or deferred
      case.
- [ ] Keep the matrix focused on stage separation and explicit negative evidence
      rather than on later summary-consistency logic.

**Evidence produced**
- One representative Story 6 cross-stage unsupported-boundary matrix.
- One review surface for explicit exclusion reasons across all three stages.

**Risks / rollback**
- Risk: later package consumers may see only one stage of negative evidence and
  miss the real claim boundary.
- Rollback/mitigation: freeze a small but stage-spanning negative matrix early.

### Engineering Task 6: Emit A Stable Story 6 Unsupported-Boundary Bundle Or Rerunnable Checker

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- validation automation | docs

**Definition of done**
- Story 6 emits one stable machine-reviewable unsupported-boundary bundle or
  rerunnable checker.
- The bundle records cross-stage negative evidence through one stable schema.
- The output is stable enough for Story 7 full-package assembly and Story 8
  summary-consistency guardrails.

**Execution checklist**
- [ ] Add a dedicated Story 6 artifact location
      (for example
      `benchmarks/density_matrix/artifacts/phase3_task6/story6_unsupported_boundary/`).
- [ ] Emit planner-entry, descriptor-generation, and runtime-stage negative
      cases through one stable schema with stage identity explicit.
- [ ] Record rerun commands, software metadata, and exclusion semantics with the
      emitted bundle.
- [ ] Keep the bundle summary explicit about stage counts and exclusion reasons.

**Evidence produced**
- One stable Story 6 unsupported-boundary bundle or checker.
- One reusable negative-evidence surface for later Task 6 stories.

**Risks / rollback**
- Risk: prose-only Story 6 closure will make later reviewers unable to tell how
  stage-separated exclusion was actually enforced.
- Rollback/mitigation: emit one machine-reviewable negative bundle directly.

### Engineering Task 7: Document The Story 6 Negative-Evidence Rule And Run The Gate

**Implements story**
- `Story 6: Unsupported-Boundary Evidence Preserves Stage Separation`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the Task 6 stage-separated unsupported-boundary
  rule.
- The Story 6 negative-evidence harness and bundle run successfully.
- Story 6 makes clear that full-package assembly and summary-consistency remain
  assigned to later stories.

**Execution checklist**
- [ ] Document the Task 6 stage categories and the shared failure vocabulary.
- [ ] Explain how excluded negative evidence remains part of the publishable
      claim boundary.
- [ ] Run focused Story 6 regression coverage and verify
      `benchmarks/density_matrix/correctness_evidence/unsupported_boundary_validation.py`.
- [ ] Record stable references to the Story 6 tests, checker, and emitted
      bundle.

**Evidence produced**
- Passing Story 6 unsupported-boundary regression checks.
- One stable Story 6 unsupported-boundary bundle or checker reference.

**Risks / rollback**
- Risk: later reviewers may confuse Story 6 negative-evidence closure with the
  full Story 7 package or Story 8 summary logic.
- Rollback/mitigation: document the Story 6 handoff to later stories explicitly.

## Exit Criteria

Story 6 is complete only when all of the following are true:

- one explicit Task 6 stage-separated unsupported-boundary taxonomy exists,
- negative records preserve stable failure stage, unsupported category, first
  unsupported condition, provenance, and exclusion reason,
- representative excluded cases exist across planner-entry, descriptor-
  generation, and runtime-stage boundaries,
- one stable Story 6 unsupported-boundary bundle or rerunnable checker exists
  for later reuse,
- and full-package assembly plus summary-consistency guardrails remain clearly
  assigned to later stories.

## Implementation Notes

- Prefer one shared cross-stage negative-evidence record over one incompatible
  artifact shape per failure stage.
- Keep Story 6 focused on explicit exclusion evidence, not yet on full-package
  assembly.
- Treat negative evidence as a required scientific output, not as an appendix to
  positive cases.
