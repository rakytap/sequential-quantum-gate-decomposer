# Story 3 Implementation Plan

## Story Being Implemented

Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader
Or Deferred Capability

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_7_STORIES.md`.

## Scope

This story turns the frozen bridge and support-matrix boundary into one explicit
documentation classification layer:

- the guaranteed generated-`HEA` VQE-facing density path is separated cleanly
  from broader manual circuit or standalone `NoisyCircuit` capability,
- required, optional, deferred, and unsupported support labels are made explicit
  for gates, noise models, and circuit-source expectations,
- documentation makes clear that broader capability does not automatically imply
  Phase 2 workflow support,
- and the resulting support-surface layer stays narrow enough that Story 4 can
  explain the mandatory evidence bar without redoing the classification work.

Out of scope for this story:

- redefining the backend-selection or canonical-workflow wording owned by Story
  2,
- widening the frozen Phase 2 support surface,
- evidence-bar and threshold interpretation owned by Story 4,
- future-work and non-goal separation owned by Story 5,
- and final terminology bundle closure owned by Story 6.

## Dependencies And Assumptions

- Story 2 already closes the supported-entry and canonical-workflow wording.
  Story 3 should build on that surface rather than restating it.
- Story 1 and Story 2 should now share one stable documentation entry surface in
  `docs/density_matrix_project/phases/phase-2/PHASE_2_DOCUMENTATION_INDEX.md`.
  Story 3 should extend that same reader-facing surface with support-surface
  classification instead of creating a competing front door for Phase 2 docs.
- Bridge semantics are already frozen in:
  - `TASK_3_MINI_SPEC.md`,
  - `P2-ADR-011`,
  - and the bridge closure in
    `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Support-matrix semantics are already frozen in:
  - `TASK_4_MINI_SPEC.md`,
  - `P2-ADR-012`,
  - `DETAILED_PLANNING_PHASE_2.md`,
  - and the support-matrix closure in the checklist.
- The current distinction between required, optional, deferred, and unsupported
  behavior already appears across Phase 2 docs and should be preserved rather
  than replaced with a second taxonomy.
- The standalone `NoisyCircuit` module may remain broader than the guaranteed
  VQE-facing density path. Story 3 must keep those two surfaces visibly distinct.
- Story 3 is a documentation-classification layer only. It does not add support
  for new gates, new noise models, or full `qgd_Circuit` parity.

## Engineering Tasks

### Engineering Task 1: Freeze The Canonical Story 3 Support-Surface Inventory And Classification Matrix

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 names one canonical inventory of support-surface categories that must
  be documented.
- The inventory covers bridge source, gate families, noise models, unsupported
  behavior, and standalone-versus-VQE-facing distinction.
- The inventory remains aligned with the frozen Phase 2 support matrix.

**Execution checklist**
- [ ] Freeze one canonical Story 3 inventory covering source surface, gate
      support, noise support, optional extensions, deferred classes, and
      unsupported behavior.
- [ ] Mark which categories are required, optional, deferred, or unsupported in
      the Phase 2 contract.
- [ ] Include explicit distinction between guaranteed VQE-facing support and
      broader standalone `NoisyCircuit` capability.
- [ ] Record the classification inventory in one stable implementation-facing
      location.

**Evidence produced**
- One stable Story 3 support-surface inventory and classification matrix.
- One reviewable mapping from support categories to frozen contract decisions.

**Risks / rollback**
- Risk: if support categories remain implicit, broader standalone capability can
  be mistaken for guaranteed workflow support.
- Rollback/mitigation: freeze one explicit support-surface matrix and reuse it
  everywhere.

### Engineering Task 2: Reuse Existing Bridge And Support-Matrix Vocabulary Without Forking The Taxonomy

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 reuses existing bridge and support-matrix vocabulary where practical.
- Readers can trace classification labels directly to Task 3 and Task 4
  documents without schema translation.
- Story 3 remains a documentation-boundary layer, not a new support taxonomy.

**Execution checklist**
- [ ] Reuse generated-`HEA`, `qgd_Circuit`, `NoisyCircuit`, required / optional /
      deferred / unsupported, and no-silent-fallback language from current
      Phase 2 docs where practical.
- [ ] Reuse the existing required gate and noise labels
      (`U3`, `CNOT`, local depolarizing, local phase damping / dephasing,
      local amplitude damping) exactly.
- [ ] Reuse optional and deferred labels for whole-register depolarizing,
      generalized amplitude damping, coherent over-rotation, correlated noise,
      readout noise, calibration-aware models, and non-Markovian noise.
- [ ] Avoid Story 3-only synonyms when a frozen Task 3 or Task 4 label already
      exists.

**Evidence produced**
- One Story 3 classification layer rooted in the canonical Task 3 and Task 4
  vocabulary.
- Reviewable traceability from support labels to the underlying contract docs.

**Risks / rollback**
- Risk: a forked taxonomy makes the support boundary harder to review and easier
  to overstate.
- Rollback/mitigation: preserve Task 3 and Task 4 vocabulary whenever practical
  and add only the minimum derived labels needed for clarity.

### Engineering Task 3: Add Explicit Completeness Rules For Required, Optional, Deferred, And Unsupported Support Labels

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 defines which support classifications must appear explicitly in the
  documentation surface.
- Missing or flattened classifications fail Story 3 review.
- The classification rules are stable enough for structured checks.

**Execution checklist**
- [ ] Mark required support, optional extensions, deferred capability, and
      unsupported behavior as explicit mandatory labels in Story 3 outputs.
- [ ] Add completeness rules so a support surface cannot pass if any mandatory
      classification bucket is absent.
- [ ] Require explicit labeling of standalone-breadth versus guaranteed
      VQE-facing support.
- [ ] Keep the completeness rules machine-checkable where practical.

**Evidence produced**
- One explicit completeness rule set for Story 3 support classification.
- One reviewable list of mandatory support labels and distinctions.

**Risks / rollback**
- Risk: documentation can mention many support details while still collapsing key
  distinctions such as optional versus required or standalone versus VQE-facing.
- Rollback/mitigation: treat missing classification buckets as a Story 3 failure.

### Engineering Task 4: Preserve Boundary Examples And No-Parity Semantics In Structured Story 3 Outputs

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- docs | validation automation

**Definition of done**
- Story 3 outputs preserve explicit examples of where the guaranteed support
  surface stops.
- The outputs make clear that full `qgd_Circuit` parity is not implied and that
  broader `NoisyCircuit` breadth does not upgrade the VQE-facing contract.
- Silent promotion of optional or deferred capability is prevented at the
  documentation layer.

**Execution checklist**
- [ ] Preserve one explicit generated-`HEA` positive path example in Story 3
      outputs.
- [ ] Preserve one explicit example that full `qgd_Circuit` parity is not part
      of the Phase 2 guarantee.
- [ ] Preserve one explicit example that optional or standalone-only capability
      does not count as guaranteed workflow support.
- [ ] Keep the examples structured enough that later Story 4 and Story 6 outputs
      can reuse them.

**Evidence produced**
- Story 3 outputs with explicit boundary examples and no-parity semantics.
- Reviewable structured examples showing where the guaranteed surface ends.

**Risks / rollback**
- Risk: classification labels alone may still be overread if boundary examples
  are missing.
- Rollback/mitigation: keep a small number of canonical examples attached to the
  structured support-surface summary.

### Engineering Task 5: Add Focused Regression Checks For Story 3 Support-Surface Semantics

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- tests

**Definition of done**
- Fast checks catch missing required classifications, missing VQE-facing versus
  standalone distinction, or wording that implies full parity.
- Negative cases show that Story 3 fails if support labels are flattened or
  broadened.
- Regression coverage remains documentation-focused and lightweight.

**Execution checklist**
- [ ] Add focused Story 3 checks in `tests/density_matrix/test_phase2_docs.py`
      or a tightly related successor.
- [ ] Add negative checks for missing required / optional / deferred /
      unsupported labels.
- [ ] Add at least one negative check for wording that implies full
      `qgd_Circuit` parity or treats standalone `NoisyCircuit` breadth as
      guaranteed workflow support.
- [ ] Keep lower-level bridge and noise execution validation outside this fast
      documentation test layer.

**Evidence produced**
- Focused regression coverage for Story 3 support-surface semantics.
- Reviewable failure messages for missing or overstated support classifications.

**Risks / rollback**
- Risk: support-surface inflation may happen in docs without changing any runtime
  behavior.
- Rollback/mitigation: lock the classification semantics down with targeted
  documentation checks.

### Engineering Task 6: Emit One Stable Story 3 Support-Surface Reference Bundle Or Rerunnable Checker

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- validation automation | docs

**Definition of done**
- Story 3 can emit one stable machine-readable support-surface summary or one
  stable rerunnable checker.
- The output records the classification matrix, key boundary examples, and the
  VQE-facing versus standalone distinction.
- The output is stable enough for Story 4 and Story 6 to reference directly.

**Execution checklist**
- [ ] Add one Story 3 command, script, or checker
      (for example under `benchmarks/density_matrix/`) for support-surface
      summary emission.
- [ ] Emit one stable artifact in a Task 7 artifact directory
      (for example `benchmarks/density_matrix/artifacts/phase2_task7/`).
- [ ] Record source references, generation command, and classification metadata
      with the emitted output.
- [ ] Keep the output narrow to support-surface semantics rather than broader
      evidence-bar interpretation.

**Evidence produced**
- One stable Task 7 Story 3 support-surface reference bundle or checker.
- One reusable Story 3 output schema for later Task 7 handoffs.

**Risks / rollback**
- Risk: support-surface guidance can remain scattered across many docs and become
  hard to audit.
- Rollback/mitigation: emit one thin structured Story 3 surface that later
  stories can reuse directly.

### Engineering Task 7: Document Story 3 Boundary Semantics And Handoff To Story 4

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- docs | validation automation

**Definition of done**
- Developer-facing notes explain what Story 3 validates, how to rerun it, and
  why it is the canonical support-surface boundary gate.
- The notes make clear that Story 3 closes support classification but not the
  evidence-bar interpretation owned by Story 4.
- The documentation stays aligned with the frozen Task 3 and Task 4 contract.

**Execution checklist**
- [ ] Document the Story 3 summary or checker and its relationship to Task 3 and
      Task 4 sources.
- [ ] Make the main Story 3 rule explicit:
      required, optional, deferred, unsupported, and standalone-only breadth
      must remain visibly distinct.
- [ ] Explain how Story 3 hands off evidence-bar interpretation to Story 4.
- [ ] Keep broader future-work expansion clearly outside Story 3 closure.

**Evidence produced**
- Updated developer-facing guidance for the Task 7 Story 3 boundary gate.
- One stable place where Story 3 scope and rerun instructions are documented.

**Risks / rollback**
- Risk: if Story 3 is poorly documented, later evidence notes may silently
  compensate for unclear support boundaries.
- Rollback/mitigation: make Story 3 the explicit classification gate and document
  the handoff boundary to Story 4.

### Engineering Task 8: Run Story 3 Validation And Confirm Support-Surface Boundary Clarity

**Implements story**
- `Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 3 classification checks pass.
- The Story 3 support-surface summary or checker runs successfully and emits
  stable output.
- Story 3 closure is backed by rerunnable documentation evidence rather than by
  prose-only explanations.

**Execution checklist**
- [ ] Run focused Story 3 regression checks for support-classification coverage.
- [ ] Run the Story 3 summary or checker command and verify emitted output.
- [ ] Confirm required / optional / deferred / unsupported labels and
      VQE-facing-versus-standalone distinctions are present and complete.
- [ ] Record stable test and artifact references for Story 4 and later Task 7
      work.

**Evidence produced**
- Passing focused checks for Story 3 support-surface semantics.
- One stable Story 3 output proving the support boundary is explicit and
  reviewable.

**Risks / rollback**
- Risk: Story 3 can appear complete while still letting broader capability look
  mandatory.
- Rollback/mitigation: require both passing checks and one stable structured
  Story 3 output before closure.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- one stable support-surface inventory defines the categories that must be
  documented,
- required, optional, deferred, unsupported, and standalone-only breadth are
  visibly distinct,
- full `qgd_Circuit` parity is not implied anywhere inside the guaranteed
  VQE-facing support surface,
- broader standalone `NoisyCircuit` capability is not allowed to masquerade as
  guaranteed workflow support,
- missing or flattened support classifications fail Story 3 completeness checks,
- one stable Story 3 output or rerunnable checker captures the support-surface
  reference layer,
- and evidence-bar interpretation, future-work separation, and terminology
  bundle closure remain clearly assigned to Stories 4 to 6.

## Implementation Notes

- Keep Story 3 tightly aligned with Task 3 and Task 4. It is a documentation
  boundary layer over already-frozen support decisions.
- Reuse `tests/density_matrix/test_phase2_docs.py` as the default fast
  documentation-regression surface unless Story 3 reveals a compelling reason to
  split it.
- Prefer explicit support labels plus a few canonical examples over long prose.
  Readers need to understand where the guaranteed surface stops.
- Treat standalone module breadth carefully. It is useful context, but it must
  never inflate the VQE-facing Phase 2 guarantee.
- If Story 3 becomes vague, Story 4 evidence-bar guidance will also become
  ambiguous. Support classification should be closed before evidence
  interpretation is packaged.
