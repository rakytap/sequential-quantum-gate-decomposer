# Story 8 Implementation Plan

## Story Being Implemented

Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across
The Paper Package

This is a Layer 4 engineering plan for implementing the eighth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one final package-consistency and reviewer-
navigation guardrail for Paper 2:

- key terminology, reviewer-entry paths, and rolled-up summary wording stay
  stable across abstract, technical short-paper, narrative short-paper,
  full-paper, lower-story outputs, and the top-level reviewer manifest,
- current counts, diagnosis-grounded interpretation, and limitation summaries do
  not drift across publication-facing and evidence-facing surfaces,
- the final Task 8 package can be reviewed as one coherent publication bundle
  with stable reviewer entry and stable summary semantics,
- and Story 8 closes the contract for "how the full Paper 2 package stays
  coherent" without reopening lower-story scope, claim, or evidence decisions.

Out of scope for this story:

- freezing the main claim and non-claims, which is owned by Story 1,
- keeping publication surfaces aligned in role and depth, which is owned by
  Story 2,
- claim-to-source traceability owned by Story 3,
- evidence-floor and threshold-or-diagnosis closure owned by Story 4,
- supported-path and benchmark-honesty wording owned by Story 5,
- manifest-driven reviewer packaging owned by Story 6,
- and future-work positioning owned by Story 7.

## Dependencies And Assumptions

- Stories 1 through 7 are already expected to freeze the claim package, align
  surfaces, define traceability, define evidence closure, define supported-path
  honesty, produce one manifest-driven reviewer package, and freeze future-work
  framing. Story 8 should enforce coherence across those surfaces rather than
  redefine them.
- Story 6 is expected to emit one top-level reviewer manifest under
  `benchmarks/density_matrix/artifacts/publication_evidence/manifest/`.
- The emitted Task 6 and Task 7 bundle families already provide summary-
  consistency precedent:
  - `benchmarks/density_matrix/artifacts/correctness_evidence/summary_consistency/`,
  - and `benchmarks/density_matrix/artifacts/performance_evidence/summary_consistency/`.
- Story 8 should reuse that precedent where it strengthens auditability rather
  than inventing a detached publication-only summary logic.
- Story 8 should treat the current implementation-backed counts as contract-
  sensitive when they are cited:
  - Task 6 `25` counted supported cases,
  - `4` required external Task 6 cases,
  - `17` explicit Task 6 boundary cases,
  - Task 7 `34` counted supported benchmark cases,
  - and `6` representative review cases with diagnosis-grounded closure.
- The natural implementation home for Task 8 final consistency validation is the
  same `benchmarks/density_matrix/publication_evidence/` package, with
  `package_consistency_validation.py` as the Story 8 validation surface and
  emitted artifacts rooted in `benchmarks/density_matrix/artifacts/publication_evidence/`.
- Story 8 should enforce coherence and summary consistency only. It should not
  become a general style linter or a substitute for lower-story validation.

## Engineering Tasks

### Engineering Task 1: Freeze The Task 8 Terminology Inventory And Package-Consistency Rule

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- docs | validation automation

**Definition of done**
- Story 8 defines one explicit terminology inventory for the final Paper 2
  package.
- Story 8 defines one explicit package-consistency rule for reviewer entry and
  summary wording.
- The rule is stable enough that later consumers can rely on the full package
  coherently.

**Execution checklist**
- [ ] Freeze one canonical terminology inventory covering exact noisy mixed-state
      circuits, canonical noisy planner surface, partitioned density runtime,
      real fused path, counted supported, diagnosis-grounded closure, required /
      optional / deferred / unsupported, and reproducibility bundle.
- [ ] Freeze one explicit package-consistency rule for reviewer-entry stability,
      count stability, and limitation-summary consistency.
- [ ] Distinguish mandatory Task 8 terminology from optional stylistic
      variation.
- [ ] Keep general editorial polish outside the Story 8 terminology inventory.

**Evidence produced**
- One stable Story 8 terminology inventory.
- One stable Story 8 package-consistency rule.

**Risks / rollback**
- Risk: without a frozen terminology and consistency rule, the final package can
  drift while still looking locally reasonable in each file.
- Rollback/mitigation: freeze the vocabulary and consistency rule before final
  package validation.

### Engineering Task 2: Reuse The Story 6 Reviewer Manifest And Task 6 / Task 7 Summary-Consistency Precedent As The Base

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- docs | code

**Definition of done**
- Story 8 derives its final consistency checks directly from the Story 6
  manifest and existing Task 6 / Task 7 summary-consistency precedent.
- Story 8 preserves stable reviewer entry and count semantics across the full
  package.
- Story 8 avoids creating a detached publication-only summary vocabulary.

**Execution checklist**
- [ ] Reuse the Story 6 top-level reviewer manifest as the direct package-entry
      substrate for Story 8.
- [ ] Reuse the Task 6 and Task 7 summary-consistency precedent where it matches
      Story 8 needs.
- [ ] Reuse lower-story field names and count semantics wherever practical.
- [ ] Document where Story 8 intentionally extends lower-story consistency
      semantics.

**Evidence produced**
- One reviewable mapping from the Story 6 manifest and prior summary precedent to
  the final Story 8 package-consistency surface.
- One explicit boundary between reused lower-level semantics and Story 8-
  specific final-package fields.

**Risks / rollback**
- Risk: Story 8 may produce polished summaries that still drift from the emitted
  package and earlier summary-consistency surfaces.
- Rollback/mitigation: anchor Story 8 directly on the Story 6 manifest and the
  existing summary-consistency precedent.

### Engineering Task 3: Build The Story 8 Package-Consistency Validation Harness

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- code | validation automation

**Definition of done**
- Story 8 has one reusable final package-consistency checker.
- The checker records terminology use, reviewer-entry stability, count
  consistency, and limitation-summary stability through one machine-reviewable
  schema.
- The checker stays focused on package coherence rather than on lower-story claim
  discovery.

**Execution checklist**
- [ ] Add a Story 8 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `package_consistency_validation.py` as the primary validation surface.
- [ ] Define one stable Story 8 package-consistency schema.
- [ ] Record terminology inventory usage, reviewer-entry references, count
      references, and limitation-summary fields.
- [ ] Keep lower-story evidence-closure and supported-path semantics outside the
      Story 8 checker except where they are reused as final consistency inputs.

**Evidence produced**
- One reusable Story 8 package-consistency checker.
- One stable Story 8 schema for final Task 8 review.

**Risks / rollback**
- Risk: Story 8 can collapse into vague editorial review if it lacks one
  structured final-package surface.
- Rollback/mitigation: define one explicit final package-consistency checker and
  schema.

### Engineering Task 4: Define Explicit Count-Stability, Reviewer-Entry-Stability, And Limitation-Summary Semantics

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- code | tests

**Definition of done**
- Story 8 defines explicit machine-reviewable semantics for stable count usage,
  stable reviewer entry, and stable limitation summaries.
- Current implementation-backed counts cannot drift silently across paper and
  artifact surfaces when cited.
- Diagnosis-grounded limitation wording remains stable across the package.

**Execution checklist**
- [ ] Add explicit fields or checks for current Task 6 and Task 7 count usage
      when cited by the package.
- [ ] Add explicit fields or checks for one stable reviewer-entry path through
      the Story 6 manifest.
- [ ] Add explicit fields or checks for limitation-summary stability, especially
      diagnosis-grounded closure and the current bottleneck wording.
- [ ] Add focused regression checks for missing or conflicting final-package
      semantics.

**Evidence produced**
- One explicit Story 8 count-stability, reviewer-entry-stability, and
  limitation-summary rule.
- Regression coverage for required final-package fields.

**Risks / rollback**
- Risk: later publication edits may preserve broad honesty while still allowing
  subtle count or limitation drift across surfaces.
- Rollback/mitigation: attach the highest-risk final-package fields directly to
  the Story 8 validation surface.

### Engineering Task 5: Add A Representative Final-Package Consistency Matrix Across Paper Surfaces And Reviewer-Entry Outputs

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- tests | validation automation

**Definition of done**
- Story 8 covers representative publication-facing and evidence-facing surfaces
  together.
- The matrix is broad enough to show that one consistency rule spans abstract,
  technical short-paper, narrative short-paper, full-paper, and the top-level
  reviewer manifest.
- The matrix remains representative and contract-driven rather than exhaustive
  over every sentence and field.

**Execution checklist**
- [ ] Include all four publication-facing paper surfaces.
- [ ] Include the Story 6 reviewer manifest.
- [ ] Include at least one lower-story output carrying claim or evidence
      semantics.
- [ ] Include at least one Task 6 or Task 7 summary-consistency input when the
      package cites current counts or limitation summaries.

**Evidence produced**
- One representative Story 8 final-package consistency matrix.
- One review surface for cross-package terminology, reviewer-entry, and count
  stability.

**Risks / rollback**
- Risk: Story 8 may appear correct on the reviewer manifest while drift persists
  in one of the manuscript surfaces, or vice versa.
- Rollback/mitigation: freeze a small but cross-package consistency matrix early.

### Engineering Task 6: Add Focused Regression Checks For Terminology Drift, Broken Reviewer Entry, Or Count Drift

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- tests

**Definition of done**
- Fast checks catch terminology drift, broken reviewer-entry paths, conflicting
  count usage, or limitation-summary drift.
- Negative cases prove Story 8 fails when the final package stops reading as one
  coherent auditable paper package.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_publication_evidence.py` or a
      tightly related successor for Story 8 final-package consistency.
- [ ] Add negative checks for conflicting terminology across publication-facing
      surfaces.
- [ ] Add negative checks for broken reviewer-entry references.
- [ ] Add negative checks for cited counts or limitation summaries drifting from
      the emitted package and lower-story outputs.

**Evidence produced**
- Focused regression coverage for Story 8 consistency failures.
- Reviewable failures for terminology drift, reviewer-entry breakage, or count
  and limitation drift.

**Risks / rollback**
- Risk: final-package drift can survive manual review because each file remains
  individually plausible.
- Rollback/mitigation: add targeted checks for the highest-risk coherence
  regressions.

### Engineering Task 7: Emit A Stable Story 8 Package-Consistency Bundle

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- validation automation | docs

**Definition of done**
- Story 8 emits one stable machine-reviewable package-consistency bundle or one
  stable rerunnable checker output.
- The output records terminology inventory status, reviewer-entry status, count
  consistency, and limitation-summary consistency through one stable schema.
- The output is stable enough for direct Task 8 closure and later publication
  review.

**Execution checklist**
- [ ] Add one stable Story 8 output location under
      `benchmarks/density_matrix/artifacts/publication_evidence/package_consistency/`.
- [ ] Emit one artifact such as `package_consistency_bundle.json`.
- [ ] Record generation command, software metadata, and package-consistency
      summary in the output.
- [ ] Keep the output focused on final package coherence rather than on new
      scope or evidence decisions.

**Evidence produced**
- One stable Story 8 package-consistency bundle or rerunnable checker output.
- One reusable Story 8 output schema for final Task 8 review.

**Risks / rollback**
- Risk: prose-only Story 8 closure will make later reviewers unable to tell
  whether the final package actually remained coherent across all required
  surfaces.
- Rollback/mitigation: emit one machine-reviewable final package-consistency
  surface directly.

### Engineering Task 8: Document The Final Task 8 Package Rule And Run The Story 8 Gate

**Implements story**
- `Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 8 validates, how to rerun it, and
  why it is the final coherence gate for Task 8.
- The Story 8 checker and emitted artifact run successfully.
- Story 8 completion is backed by rerunnable final-package validation rather
  than by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 8 terminology inventory, reviewer-entry rule, and
      count-consistency rule.
- [ ] Make the Story 8 rule explicit:
      the full Paper 2 package must read as one coherent auditable bundle across
      paper surfaces, lower-story outputs, and the top-level reviewer manifest.
- [ ] Explain that Story 8 closes package consistency but does not reopen lower-
      story scope, claim, or evidence decisions.
- [ ] Run focused Story 8 regression checks and verify the emitted Story 8
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 8 final-package consistency.
- One stable Story 8 output proving coherent Task 8 package closure.

**Risks / rollback**
- Risk: Story 8 can look complete while still allowing subtle drift between the
  reviewer manifest, emitted counts, and paper surfaces.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 8.

## Exit Criteria

Story 8 is complete only when all of the following are true:

- one explicit terminology inventory and package-consistency rule define the
  final Task 8 coherence surface,
- reviewer entry remains stable through the top-level Story 6 manifest,
- cited counts and diagnosis-grounded limitation summaries remain consistent with
  lower-story outputs and emitted Task 6 / Task 7 package surfaces,
- terminology drift, broken reviewer entry, or count drift fail focused Story 8
  checks,
- one stable Story 8 bundle or rerunnable checker captures the final package-
  consistency surface,
- and no lower-story claim, evidence, support-surface, or roadmap decision is
  reopened by the final coherence layer.

## Implementation Notes

- Story 8 is the final coherence guardrail, not a place to rediscover the Paper
  2 claim or evidence package. Those are already frozen by the earlier stories.
- Treat counts and limitation summaries as contract-sensitive. Small wording
  drift can still weaken reviewer trust materially.
- Keep reviewer entry explicit and stable. A coherent package is not only one
  with consistent terms, but one reviewers can actually navigate.
- Story 8 should remain thin and validating. Its job is to prove coherence, not
  to replace the rest of Task 8 with a new monolithic package layer.
