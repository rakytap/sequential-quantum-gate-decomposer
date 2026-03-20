# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded
Planner Claim, And Current Benchmark Surface Honestly

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit supported-path and benchmark-honesty
surface for Paper 2:

- the guaranteed supported path for Paper 2 is stated precisely and stays
  aligned with the frozen Phase 3 contract,
- the no-fallback rule remains explicit in publication-facing wording,
- the Task 5 planner claim remains bounded to the delivered benchmark-calibrated
  `max_partition_qubits` span-budget surface instead of broad planner-family
  closure,
- the current Task 6 and Task 7 count surfaces remain honest when cited,
- and Story 5 closes the contract for "what the delivered Phase 3 baseline
  actually supports and what the current benchmark package actually shows"
  without taking ownership of the evidence-closure rule or top-level manifest
  packaging.

Out of scope for this story:

- freezing the main claim and non-claims, which is owned by Story 1,
- keeping publication surfaces aligned, which is owned by Story 2,
- claim-to-source traceability owned by Story 3,
- evidence-floor and threshold-or-diagnosis closure owned by Story 4,
- manifest-driven reviewer packaging owned by Story 6,
- future-work positioning owned by Story 7,
- and package-level terminology, reviewer-entry, and summary-consistency
  guardrails owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 4 are already expected to freeze the claim package, align
  surfaces, map claims to sources, and define evidence closure. Story 5 should
  refine supported-path and current-benchmark wording within that frozen frame.
- The authoritative supported-path and no-fallback rules are already frozen in:
  - `DETAILED_PLANNING_PHASE_3.md`,
  - `ADRs_PHASE_3.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - and `TASK_8_MINI_SPEC.md`.
- The current delivered Task 5 result is already narrowed to a benchmark-
  calibrated selection rule over auditable `max_partition_qubits` span-budget
  settings on the existing noisy planner surface.
- The current emitted Task 6 bundle family already records:
  - `25` counted supported cases,
  - `4` required external-reference cases,
  - and `17` explicit unsupported-boundary cases.
- The current emitted Task 7 bundle family already records:
  - `34` counted supported benchmark cases,
  - `6` representative review cases,
  - and diagnosis-grounded rather than positive-threshold closure on the current
    representative review set.
- Story 5 should treat those counts as implementation-backed facts only when the
  paper actually cites them. Story 5 should validate accuracy and joinability,
  not force all surfaces to enumerate every count.
- The natural implementation home for Task 8 supported-path and benchmark-
  honesty validation is the same `benchmarks/density_matrix/publication_evidence/`
  package, with `supported_path_validation.py` as the Story 5 validation surface
  and emitted artifacts rooted in `benchmarks/density_matrix/artifacts/phase3_task8/`.

## Engineering Tasks

### Engineering Task 1: Freeze The Supported-Path, No-Fallback, And Bounded-Planner Wording Inventory

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one explicit wording inventory for the supported path,
  no-fallback rule, and bounded planner claim.
- The inventory is stable enough that later Task 8 stories can package it
  directly.
- The inventory stays aligned with the frozen Phase 3 contract.

**Execution checklist**
- [ ] Freeze one explicit supported-path description for the canonical noisy
      mixed-state planner surface plus the documented exact lowering for the
      frozen Phase 2 continuity workflow and the required structured benchmark
      families.
- [ ] Freeze one explicit no-fallback statement for `partitioned_density`
      evidence.
- [ ] Freeze one explicit bounded-planner statement for the current Task 5
      `max_partition_qubits` span-budget surface.
- [ ] Keep broader planner-family closure outside the Story 5 wording inventory.

**Evidence produced**
- One stable Story 5 supported-path and bounded-planner wording inventory.
- One explicit boundary between supported baseline behavior and broader planner
  design-space discussion.

**Risks / rollback**
- Risk: without a frozen wording inventory, Paper 2 can quietly imply broader
  planner or workflow support than the delivered baseline actually provides.
- Rollback/mitigation: freeze the supported-path rule before broad publication
  polishing.

### Engineering Task 2: Reuse The Frozen Phase 3 Contract And Current Emitted Bundle Counts As The Base

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- docs | code

**Definition of done**
- Story 5 derives supported-path and current-benchmark honesty directly from the
  frozen Phase 3 contract and current emitted bundle surfaces.
- Story 5 preserves direct joinability to the current Task 6 and Task 7 records.
- Story 5 avoids inventing a second publication-only benchmark vocabulary.

**Execution checklist**
- [ ] Reuse the supported-path and no-fallback wording already frozen in the
      planning, ADR, checklist, and Task 8 mini-spec surfaces.
- [ ] Reuse the current Task 5 bounded-planner interpretation instead of
      promoting broader planner-family language.
- [ ] Reuse the current Task 6 and Task 7 counts and interpretation only through
      direct references to emitted bundle surfaces.
- [ ] Treat broken or ambiguous joins to current bundle counts as a Story 5
      failure condition.

**Evidence produced**
- One reviewable mapping from frozen contract wording and current emitted bundle
  facts to Story 5 publication language.
- One explicit boundary between contract-backed statements and optional context.

**Risks / rollback**
- Risk: Story 5 may use true-sounding phrases that no longer line up with the
  emitted evidence or the frozen contract.
- Rollback/mitigation: anchor Story 5 directly on the contract and emitted
  bundle surfaces.

### Engineering Task 3: Define The Story 5 Supported-Path And Benchmark-Honesty Record Schema And Checker

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 has one reusable checker for supported-path and benchmark-honesty
  wording.
- The checker records supported-path fields, no-fallback presence, bounded-
  planner wording, and current-count references through one stable schema.
- The checker stays focused on honest supported-path description rather than on
  evidence closure or top-level manifest packaging.

**Execution checklist**
- [ ] Add a Story 5 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `supported_path_validation.py` as the primary validation surface.
- [ ] Define one stable Story 5 supported-path record schema.
- [ ] Record whether the current publication surface cites counts and, if so,
      which emitted bundle references support them.
- [ ] Keep evidence-closure flags, future-work framing, and package-consistency
      logic outside the Story 5 checker.

**Evidence produced**
- One reusable Story 5 supported-path and benchmark-honesty checker.
- One stable Story 5 record schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 5 can become vague editorial review if it lacks one structured
  validation surface.
- Rollback/mitigation: validate one machine-reviewable supported-path and count-
  honesty record set directly.

### Engineering Task 4: Add Explicit Checks For No-Fallback, Bounded-Planner, And Current-Benchmark Interpretation Honesty

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- code | tests

**Definition of done**
- Story 5 checks the highest-risk honesty statements directly.
- No-fallback, bounded-planner, and current benchmark interpretation wording are
  each represented explicitly.
- Diagnosis-grounded benchmark closure cannot be misread as a positive-threshold
  win when cited.

**Execution checklist**
- [ ] Add explicit record fields or checks for the no-fallback rule.
- [ ] Add explicit record fields or checks for bounded Task 5 planner-claim
      wording.
- [ ] Add explicit record fields or checks for the current Task 7 diagnosis-
      grounded benchmark interpretation.
- [ ] Add focused regression checks for missing or inflated benchmark-honesty
      wording.

**Evidence produced**
- One explicit Story 5 honesty rule for supported-path, no-fallback, planner-
  claim, and benchmark interpretation wording.
- Regression coverage for required honesty-field stability.

**Risks / rollback**
- Risk: later publication edits may preserve the claim boundary in spirit while
  still overstating the current planner or benchmark result in detail.
- Rollback/mitigation: attach the key honesty fields directly to the Story 5
  validation surface.

### Engineering Task 5: Add A Representative Supported-Path And Benchmark-Honesty Matrix Across Publication Surfaces

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 covers representative supported-path and benchmark-honesty statements
  across the Paper 2 package.
- The matrix is broad enough to show that one honesty rule spans abstract,
  technical short-paper, narrative short-paper, and full-paper surfaces.
- The matrix remains representative and contract-driven rather than exhaustive
  over every sentence.

**Execution checklist**
- [ ] Include at least one supported-path statement.
- [ ] Include at least one no-fallback statement.
- [ ] Include at least one bounded Task 5 planner-claim statement.
- [ ] Include at least one current Task 6 or Task 7 count or diagnosis
      interpretation statement where the relevant surface cites current results.

**Evidence produced**
- One representative Story 5 supported-path and benchmark-honesty matrix.
- One review surface for cross-surface honesty coverage.

**Risks / rollback**
- Risk: Story 5 may look correct in one file while another file quietly broadens
  the supported path or misstates the current benchmark outcome.
- Rollback/mitigation: freeze a small but cross-surface honesty matrix early.

### Engineering Task 6: Add Focused Regression Checks For Overstated Support Or Inflated Benchmark Claims

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- tests

**Definition of done**
- Fast checks catch overstated support-surface language or inflated benchmark
  claims.
- Negative cases prove Story 5 fails when Paper 2 implies broader planner
  closure, hidden fallback, or positive-threshold success that is not supported
  by the current emitted package.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task8.py` or a
      tightly related successor for Story 5 honesty validation.
- [ ] Add negative checks for broader planner-family closure language.
- [ ] Add negative checks for silent-fallback-compatible wording.
- [ ] Add negative checks for diagnosis-grounded benchmark results being written
      as positive-threshold wins.

**Evidence produced**
- Focused regression coverage for Story 5 overstatement failures.
- Reviewable failures for inflated support-surface or benchmark claims.

**Risks / rollback**
- Risk: overstatement can survive manual review because the paper still sounds
  technically plausible.
- Rollback/mitigation: add targeted checks for the highest-risk honesty
  regressions.

### Engineering Task 7: Emit A Stable Story 5 Supported-Path Scope Bundle

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable supported-path scope bundle or one
  stable rerunnable checker output.
- The output records supported-path wording, no-fallback presence, bounded-
  planner interpretation, and current-result references through one stable
  schema.
- The output is stable enough for later manifest and package-consistency stories
  to consume directly.

**Execution checklist**
- [ ] Add one stable Story 5 output location under
      `benchmarks/density_matrix/artifacts/phase3_task8/story5_supported_path_scope/`.
- [ ] Emit one artifact such as `supported_path_scope_bundle.json`.
- [ ] Record generation command, software metadata, and honesty-summary fields in
      the output.
- [ ] Keep the output focused on supported-path and benchmark-honesty semantics
      rather than on full reviewer-entry packaging.

**Evidence produced**
- One stable Story 5 supported-path scope bundle or rerunnable checker output.
- One reusable Story 5 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 5 closure will make later reviewers unable to tell
  whether Paper 2 really preserved the frozen support boundary and current
  benchmark interpretation.
- Rollback/mitigation: emit one machine-reviewable honesty surface directly.

### Engineering Task 8: Document Story 5 Honesty Rules And Run The Story 5 Gate

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 5 validates, how to rerun it, and
  how it hands off to later Task 8 stories.
- The Story 5 checker and emitted artifact run successfully.
- Story 5 completion is backed by rerunnable supported-path and benchmark-
  honesty validation rather than by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 5 supported-path and benchmark-honesty rules.
- [ ] Make the Story 5 rule explicit:
      Paper 2 must describe the delivered baseline and current benchmark outcome
      honestly, not as a broader planner or speedup result.
- [ ] Explain how Story 5 hands off top-level reviewer packaging to Story 6 and
      future-work framing to Story 7.
- [ ] Run focused Story 5 regression checks and verify the emitted Story 5
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 5 supported-path and benchmark honesty.
- One stable Story 5 output proving honest Paper 2 support-surface language.

**Risks / rollback**
- Risk: Story 5 can look complete while still allowing subtle overstatement in
  later publication edits.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 5.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one explicit supported-path, no-fallback, and bounded-planner wording
  inventory defines the honest Paper 2 baseline,
- the current benchmark interpretation remains diagnosis-grounded rather than
  being inflated into a positive-threshold claim,
- inaccurate support-surface or benchmark wording fails focused Story 5 checks,
- one stable Story 5 bundle or rerunnable checker captures the supported-path
  honesty surface,
- and manifest packaging, future-work framing, and package-level consistency
  remain clearly assigned to Stories 6 through 8.

## Implementation Notes

- Story 5 is where the paper proves it understands its own delivered baseline.
  Honest support-surface wording is part of the scientific contribution.
- Keep the bounded Task 5 planner-claim wording explicit. Avoid sliding from
  "supported benchmark-calibrated rule" into "broad planner-family conclusion."
- When current counts are cited, treat them as evidence-backed facts that must
  remain joinable to emitted bundles, not as prose-only summary numbers.
- Story 5 should stay close to the frozen contract and emitted artifacts.
  Reviewer trust depends on that closeness.
# Story 5 Implementation Plan

## Story Being Implemented

Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded
Planner Claim, And Current Benchmark Surface Honestly

This is a Layer 4 engineering plan for implementing the fifth behavioral slice
from `TASK_8_STORIES.md`.

## Scope

This story turns Task 8 into one explicit supported-path and benchmark-honesty
surface for Paper 2:

- the guaranteed supported path for Paper 2 is stated precisely and stays
  aligned with the frozen Phase 3 contract,
- the no-fallback rule remains explicit in publication-facing wording,
- the Task 5 planner claim remains bounded to the delivered benchmark-calibrated
  `max_partition_qubits` span-budget surface instead of broad planner-family
  closure,
- the current Task 6 and Task 7 count surfaces remain honest when cited,
- and Story 5 closes the contract for "what the delivered Phase 3 baseline
  actually supports and what the current benchmark package actually shows"
  without taking ownership of the evidence-closure rule or top-level manifest
  packaging.

Out of scope for this story:

- freezing the main claim and non-claims, which is owned by Story 1,
- keeping publication surfaces aligned, which is owned by Story 2,
- claim-to-source traceability owned by Story 3,
- evidence-floor and threshold-or-diagnosis closure owned by Story 4,
- manifest-driven reviewer packaging owned by Story 6,
- future-work positioning owned by Story 7,
- and package-level terminology, reviewer-entry, and summary-consistency
  guardrails owned by Story 8.

## Dependencies And Assumptions

- Stories 1 through 4 are already expected to freeze the claim package, align
  surfaces, map claims to sources, and define evidence closure. Story 5 should
  refine supported-path and current-benchmark wording within that frozen frame.
- The authoritative supported-path and no-fallback rules are already frozen in:
  - `DETAILED_PLANNING_PHASE_3.md`,
  - `ADRs_PHASE_3.md`,
  - `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
  - and `TASK_8_MINI_SPEC.md`.
- The current delivered Task 5 result is already narrowed to a benchmark-
  calibrated selection rule over auditable `max_partition_qubits` span-budget
  settings on the existing noisy planner surface.
- The current emitted Task 6 bundle family already records:
  - `25` counted supported cases,
  - `4` required external-reference cases,
  - and `17` explicit unsupported-boundary cases.
- The current emitted Task 7 bundle family already records:
  - `34` counted supported benchmark cases,
  - `6` representative review cases,
  - and diagnosis-grounded rather than positive-threshold closure on the current
    representative review set.
- Story 5 should treat those counts as implementation-backed facts only when the
  paper actually cites them. Story 5 should validate accuracy and joinability,
  not force all surfaces to enumerate every count.
- The natural implementation home for Task 8 supported-path and benchmark-
  honesty validation is the same `benchmarks/density_matrix/publication_evidence/`
  package, with `supported_path_validation.py` as the Story 5 validation surface
  and emitted artifacts rooted in `benchmarks/density_matrix/artifacts/phase3_task8/`.

## Engineering Tasks

### Engineering Task 1: Freeze The Supported-Path, No-Fallback, And Bounded-Planner Wording Inventory

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- docs | validation automation

**Definition of done**
- Story 5 defines one explicit wording inventory for the supported path,
  no-fallback rule, and bounded planner claim.
- The inventory is stable enough that later Task 8 stories can package it
  directly.
- The inventory stays aligned with the frozen Phase 3 contract.

**Execution checklist**
- [ ] Freeze one explicit supported-path description for the canonical noisy
      mixed-state planner surface plus the documented exact lowering for the
      frozen Phase 2 continuity workflow and the required structured benchmark
      families.
- [ ] Freeze one explicit no-fallback statement for `partitioned_density`
      evidence.
- [ ] Freeze one explicit bounded-planner statement for the current Task 5
      `max_partition_qubits` span-budget surface.
- [ ] Keep broader planner-family closure outside the Story 5 wording inventory.

**Evidence produced**
- One stable Story 5 supported-path and bounded-planner wording inventory.
- One explicit boundary between supported baseline behavior and broader planner
  design-space discussion.

**Risks / rollback**
- Risk: without a frozen wording inventory, Paper 2 can quietly imply broader
  planner or workflow support than the delivered baseline actually provides.
- Rollback/mitigation: freeze the supported-path rule before broad publication
  polishing.

### Engineering Task 2: Reuse The Frozen Phase 3 Contract And Current Emitted Bundle Counts As The Base

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- docs | code

**Definition of done**
- Story 5 derives supported-path and current-benchmark honesty directly from the
  frozen Phase 3 contract and current emitted bundle surfaces.
- Story 5 preserves direct joinability to the current Task 6 and Task 7 records.
- Story 5 avoids inventing a second publication-only benchmark vocabulary.

**Execution checklist**
- [ ] Reuse the supported-path and no-fallback wording already frozen in the
      planning, ADR, checklist, and Task 8 mini-spec surfaces.
- [ ] Reuse the current Task 5 bounded-planner interpretation instead of
      promoting broader planner-family language.
- [ ] Reuse the current Task 6 and Task 7 counts and interpretation only through
      direct references to emitted bundle surfaces.
- [ ] Treat broken or ambiguous joins to current bundle counts as a Story 5
      failure condition.

**Evidence produced**
- One reviewable mapping from frozen contract wording and current emitted bundle
  facts to Story 5 publication language.
- One explicit boundary between contract-backed statements and optional context.

**Risks / rollback**
- Risk: Story 5 may use true-sounding phrases that no longer line up with the
  emitted evidence or the frozen contract.
- Rollback/mitigation: anchor Story 5 directly on the contract and emitted
  bundle surfaces.

### Engineering Task 3: Define The Story 5 Supported-Path And Benchmark-Honesty Record Schema And Checker

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- code | validation automation

**Definition of done**
- Story 5 has one reusable checker for supported-path and benchmark-honesty
  wording.
- The checker records supported-path fields, no-fallback presence, bounded-
  planner wording, and current-count references through one stable schema.
- The checker stays focused on honest supported-path description rather than on
  evidence closure or top-level manifest packaging.

**Execution checklist**
- [ ] Add a Story 5 checker under
      `benchmarks/density_matrix/publication_evidence/`, with
      `supported_path_validation.py` as the primary validation surface.
- [ ] Define one stable Story 5 supported-path record schema.
- [ ] Record whether the current publication surface cites counts and, if so,
      which emitted bundle references support them.
- [ ] Keep evidence-closure flags, future-work framing, and package-consistency
      logic outside the Story 5 checker.

**Evidence produced**
- One reusable Story 5 supported-path and benchmark-honesty checker.
- One stable Story 5 record schema for later Task 8 reuse.

**Risks / rollback**
- Risk: Story 5 can become vague editorial review if it lacks one structured
  validation surface.
- Rollback/mitigation: validate one machine-reviewable supported-path and count-
  honesty record set directly.

### Engineering Task 4: Add Explicit Checks For No-Fallback, Bounded-Planner, And Current-Benchmark Interpretation Honesty

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- code | tests

**Definition of done**
- Story 5 checks the highest-risk honesty statements directly.
- No-fallback, bounded-planner, and current benchmark interpretation wording are
  each represented explicitly.
- Diagnosis-grounded benchmark closure cannot be misread as a positive-threshold
  win when cited.

**Execution checklist**
- [ ] Add explicit record fields or checks for the no-fallback rule.
- [ ] Add explicit record fields or checks for bounded Task 5 planner-claim
      wording.
- [ ] Add explicit record fields or checks for the current Task 7 diagnosis-
      grounded benchmark interpretation.
- [ ] Add focused regression checks for missing or inflated benchmark-honesty
      wording.

**Evidence produced**
- One explicit Story 5 honesty rule for supported-path, no-fallback, planner-
  claim, and benchmark interpretation wording.
- Regression coverage for required honesty-field stability.

**Risks / rollback**
- Risk: later publication edits may preserve the claim boundary in spirit while
  still overstating the current planner or benchmark result in detail.
- Rollback/mitigation: attach the key honesty fields directly to the Story 5
  validation surface.

### Engineering Task 5: Add A Representative Supported-Path And Benchmark-Honesty Matrix Across Publication Surfaces

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- tests | validation automation

**Definition of done**
- Story 5 covers representative supported-path and benchmark-honesty statements
  across the Paper 2 package.
- The matrix is broad enough to show that one honesty rule spans abstract,
  technical short-paper, narrative short-paper, and full-paper surfaces.
- The matrix remains representative and contract-driven rather than exhaustive
  over every sentence.

**Execution checklist**
- [ ] Include at least one supported-path statement.
- [ ] Include at least one no-fallback statement.
- [ ] Include at least one bounded Task 5 planner-claim statement.
- [ ] Include at least one current Task 6 or Task 7 count or diagnosis
      interpretation statement where the relevant surface cites current results.

**Evidence produced**
- One representative Story 5 supported-path and benchmark-honesty matrix.
- One review surface for cross-surface honesty coverage.

**Risks / rollback**
- Risk: Story 5 may look correct in one file while another file quietly broadens
  the supported path or misstates the current benchmark outcome.
- Rollback/mitigation: freeze a small but cross-surface honesty matrix early.

### Engineering Task 6: Add Focused Regression Checks For Overstated Support Or Inflated Benchmark Claims

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- tests

**Definition of done**
- Fast checks catch overstated support-surface language or inflated benchmark
  claims.
- Negative cases prove Story 5 fails when Paper 2 implies broader planner
  closure, hidden fallback, or positive-threshold success that is not supported
  by the current emitted package.
- Regression coverage remains narrow and publication-package focused.

**Execution checklist**
- [ ] Add focused checks in `tests/partitioning/test_phase3_task8.py` or a
      tightly related successor for Story 5 honesty validation.
- [ ] Add negative checks for broader planner-family closure language.
- [ ] Add negative checks for silent-fallback-compatible wording.
- [ ] Add negative checks for diagnosis-grounded benchmark results being written
      as positive-threshold wins.

**Evidence produced**
- Focused regression coverage for Story 5 overstatement failures.
- Reviewable failures for inflated support-surface or benchmark claims.

**Risks / rollback**
- Risk: overstatement can survive manual review because the paper still sounds
  technically plausible.
- Rollback/mitigation: add targeted checks for the highest-risk honesty
  regressions.

### Engineering Task 7: Emit A Stable Story 5 Supported-Path Scope Bundle

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- validation automation | docs

**Definition of done**
- Story 5 emits one stable machine-reviewable supported-path scope bundle or one
  stable rerunnable checker output.
- The output records supported-path wording, no-fallback presence, bounded-
  planner interpretation, and current-result references through one stable
  schema.
- The output is stable enough for later manifest and package-consistency stories
  to consume directly.

**Execution checklist**
- [ ] Add one stable Story 5 output location under
      `benchmarks/density_matrix/artifacts/phase3_task8/story5_supported_path_scope/`.
- [ ] Emit one artifact such as `supported_path_scope_bundle.json`.
- [ ] Record generation command, software metadata, and honesty-summary fields in
      the output.
- [ ] Keep the output focused on supported-path and benchmark-honesty semantics
      rather than on full reviewer-entry packaging.

**Evidence produced**
- One stable Story 5 supported-path scope bundle or rerunnable checker output.
- One reusable Story 5 output schema for later Task 8 handoffs.

**Risks / rollback**
- Risk: prose-only Story 5 closure will make later reviewers unable to tell
  whether Paper 2 really preserved the frozen support boundary and current
  benchmark interpretation.
- Rollback/mitigation: emit one machine-reviewable honesty surface directly.

### Engineering Task 8: Document Story 5 Honesty Rules And Run The Story 5 Gate

**Implements story**
- `Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain what Story 5 validates, how to rerun it, and
  how it hands off to later Task 8 stories.
- The Story 5 checker and emitted artifact run successfully.
- Story 5 completion is backed by rerunnable supported-path and benchmark-
  honesty validation rather than by editorial confidence alone.

**Execution checklist**
- [ ] Document the Story 5 supported-path and benchmark-honesty rules.
- [ ] Make the Story 5 rule explicit:
      Paper 2 must describe the delivered baseline and current benchmark outcome
      honestly, not as a broader planner or speedup result.
- [ ] Explain how Story 5 hands off top-level reviewer packaging to Story 6 and
      future-work framing to Story 7.
- [ ] Run focused Story 5 regression checks and verify the emitted Story 5
      bundle or checker output.

**Evidence produced**
- Passing focused checks for Story 5 supported-path and benchmark honesty.
- One stable Story 5 output proving honest Paper 2 support-surface language.

**Risks / rollback**
- Risk: Story 5 can look complete while still allowing subtle overstatement in
  later publication edits.
- Rollback/mitigation: require passing checks plus one stable emitted output
  before closing Story 5.

## Exit Criteria

Story 5 is complete only when all of the following are true:

- one explicit supported-path, no-fallback, and bounded-planner wording
  inventory defines the honest Paper 2 baseline,
- the current benchmark interpretation remains diagnosis-grounded rather than
  being inflated into a positive-threshold claim,
- inaccurate support-surface or benchmark wording fails focused Story 5 checks,
- one stable Story 5 bundle or rerunnable checker captures the supported-path
  honesty surface,
- and manifest packaging, future-work framing, and package-level consistency
  remain clearly assigned to Stories 6 through 8.

## Implementation Notes

- Story 5 is where the paper proves it understands its own delivered baseline.
  Honest support-surface wording is part of the scientific contribution.
- Keep the bounded Task 5 planner-claim wording explicit. Avoid sliding from
  "supported benchmark-calibrated rule" into "broad planner-family conclusion."
- When current counts are cited, treat them as evidence-backed facts that must
  remain joinable to emitted bundles, not as prose-only summary numbers.
- Story 5 should stay close to the frozen contract and emitted artifacts.
  Reviewer trust depends on that closeness.
