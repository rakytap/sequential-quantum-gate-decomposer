# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And
In-Bounds

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns optional legacy-source reuse into a bounded exact-lowering
path rather than an implied parity claim:

- selected `qgd_Circuit` or `Gates_block` sources may enter the canonical
  planner surface when they lower exactly into the frozen Phase 3 support
  matrix,
- the lowering path is auditable enough to prove that support came from exact
  in-bounds translation rather than from heuristic rewriting,
- the optional path reuses existing legacy circuit introspection where possible
  without redefining the canonical planner contract around those older source
  types,
- and unsupported legacy-source features remain explicit non-coverage rather
  than silent partial support.

Out of scope for this story:

- continuity-anchor closure owned by Story 1,
- mandatory methods-workload coverage owned by Story 2,
- richer planner-entry audit schema owned by Story 3,
- unsupported planner-request closure owned by Story 5,
- and full direct parity for every `qgd_Circuit`, `Gates_block`, `Composite`, or
  `Adaptive` structure.

## Dependencies And Assumptions

- Story 4 depends on the canonical planner-entry contract already established by
  Stories 1 through 3.
- The target lowering surface is the existing Python-side canonical planner
  module `squander/partitioning/noisy_planner.py`; Story 4 should lower legacy
  sources into that shared contract rather than inventing a second legacy-only
  planner representation.
- Stories 1 and 2 have already frozen workload IDs, source-type labels, and
  planner-surface payload structure under
  `benchmarks/density_matrix/artifacts/planner_surface/`; Story 4 should reuse
  that schema for positive and negative legacy-source evidence.
- Story 3 confirmed that the planner-side schema is already rich enough for Task
  1 auditability, so Story 4 should stay entirely on that planner-side path
  unless a concrete legacy-source gap proves otherwise.
- The main existing legacy-source introspection surfaces are:
  - `qgd_Circuit.get_Gates()` and related gate metadata in
    `squander/gates/qgd_Circuit.py`,
  - gate-level helpers such as `get_Name()`, `get_Involved_Qbits()`,
    `get_Parameter_Start_Index()`, and `get_Parameter_Num()` exposed through the
    gate wrappers,
  - and dependency utilities in `squander/partitioning/tools.py`.
- The underlying C++ legacy source is `Gates_block`, with structural helpers in
  `squander/src-cpp/gates/include/Gates_block.h` and
  `squander/src-cpp/gates/Gates_block.cpp`.
- Story 4 should not infer noise that is absent from a legacy source. If noise
  is attached to a legacy-source request, it must be explicit and independently
  validated against the frozen support matrix.
- The required optional-positive slice is intentionally narrow: exact lowering
  of in-bounds `U3` / `CNOT` content into the same canonical planner surface
  already used by required Phase 3 workloads.

## Engineering Tasks

### Engineering Task 1: Freeze The Optional Legacy-Source Eligibility Rules

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 names exactly which legacy-source situations are eligible for optional
  lowering.
- The eligibility rules distinguish exact in-bounds support from deferred or
  unsupported parity.
- The rules are tight enough that later reviewers can tell why a legacy-source
  case was accepted or rejected.

**Execution checklist**
- [ ] Freeze one eligibility rule set for optional legacy-source lowering.
- [ ] Require exact lowering into the frozen `U3` / `CNOT` support surface.
- [ ] Require explicit handling for any attached noise schedule rather than
      implicit or heuristic noise insertion.
- [ ] Record the explicit non-goals:
      full source parity, heuristic rewriting, partial lowering, and hidden
      support expansion.

**Evidence produced**
- One stable Story 4 eligibility rule set.
- One explicit boundary between optional exact support and deferred parity work.

**Risks / rollback**
- Risk: vague optional-lowering wording can be misread as a promise of broad
  legacy-source compatibility.
- Rollback/mitigation: define a narrow eligibility rule set before any legacy
  path is implemented.

### Engineering Task 2: Add A Legacy-Source Classifier And Introspection Path

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- code | tests

**Definition of done**
- Supported legacy-source requests are classified explicitly before lowering.
- The classification path can inspect gate names, qubit support, and parameter
  spans from `qgd_Circuit` or `Gates_block`.
- The classification path produces stable source labels for planner-entry
  provenance.

**Execution checklist**
- [ ] Reuse `qgd_Circuit.get_Gates()` and gate metadata helpers as the primary
      Python-side source walker when possible.
- [ ] Add a narrow `Gates_block` inspection path or adapter only where the
      Python-side path is insufficient.
- [ ] Record stable source labels such as `legacy_qgd_circuit_exact` or
      `legacy_gates_block_exact` only for eligible cases.
- [ ] Reject ambiguous legacy-source cases before they reach canonical lowering.

**Evidence produced**
- One reviewable legacy-source classification path.
- Regression coverage for legacy-source identification and eligibility checks.

**Risks / rollback**
- Risk: without explicit classification, exact and inexact legacy-source cases
  can become indistinguishable in later artifacts.
- Rollback/mitigation: classify source type and eligibility before any lowering
  occurs.

### Engineering Task 3: Lower In-Bounds Legacy Gates Into The Canonical Planner Surface Without Heuristic Rewriting

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- code | tests

**Definition of done**
- Eligible `qgd_Circuit` or `Gates_block` content lowers into the canonical
  planner surface through one explicit exact-lowering path.
- The lowering path preserves gate order, qubit support, and parameter spans for
  in-bounds `U3` / `CNOT` cases.
- The implementation does not silently rewrite unsupported gates into supported
  substitutes.

**Execution checklist**
- [ ] Build a narrow exact-lowering walker over the inspected legacy gate list.
- [ ] Map only the frozen supported gate families into canonical planner
      operations using the shared constructors in
      `squander/partitioning/noisy_planner.py`.
- [ ] Preserve source gate order and parameter spans explicitly.
- [ ] Fail immediately when an unsupported gate family, unsupported control
      structure, or unsupported circuit form appears.

**Evidence produced**
- One exact-lowering path for eligible legacy-source gate content.
- Focused tests proving order and parameter stability on in-bounds cases.

**Risks / rollback**
- Risk: heuristic or partial lowering can make unsupported cases appear
  supported long enough to contaminate later validation.
- Rollback/mitigation: implement only exact gate-family mappings and hard-fail
  the rest.

### Engineering Task 4: Attach Explicit In-Bounds Noise Metadata Only When Provided And Supported

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- code | tests

**Definition of done**
- Legacy-source lowering does not invent noise metadata that the source never
  specified.
- If an explicit supported local-noise schedule accompanies an eligible legacy
  source, that schedule can be attached to the canonical planner surface with
  the same audit rules used elsewhere.
- Unsupported or malformed noise schedules fail before planner entry.

**Execution checklist**
- [ ] Define whether optional legacy-source cases can arrive gate-only or with a
      separate explicit supported noise schedule.
- [ ] Reuse the existing supported local-noise normalization and validation
      rules from the shared planner module rather than inventing a second noise
      vocabulary.
- [ ] Attach explicit in-bounds noise metadata only after gate eligibility
      passes.
- [ ] Reject malformed indices, unsupported channels, or unsupported targets
      before canonical planner entry.

**Evidence produced**
- One reviewable legacy-source noise-attachment rule for supported cases.
- Focused tests proving no implicit noise inference occurs.

**Risks / rollback**
- Risk: optional legacy-source support can accidentally become an undocumented
  noise-construction surface.
- Rollback/mitigation: require explicit supported schedules and reuse the same
  validation logic used for other canonical planner inputs.

### Engineering Task 5: Add Positive And Negative Validation For The Optional Legacy Slice

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 has at least one positive in-bounds legacy-source case and
  representative negative out-of-bounds cases.
- Positive cases prove exact entry into the canonical planner surface.
- Negative cases prove that unsupported legacy-source features do not acquire
  accidental support.

**Execution checklist**
- [ ] Add fast regression tests for at least one eligible in-bounds legacy
      source.
- [ ] Add negative cases for unsupported gates, unsupported source forms, or
      unsupported noise attachments.
- [ ] Reuse `benchmarks/density_matrix/bridge_scope/unsupported_bridge_validation.py`
      patterns where they provide a good model for first-unsupported-condition
      reporting.
- [ ] Keep the Story 4 validation layer distinct from the broader Story 5
      unsupported-planner bundle.

**Evidence produced**
- Focused Story 4 regression coverage for optional legacy-source support.
- Reviewable negative evidence showing that unsupported parity is not implied.

**Risks / rollback**
- Risk: without negative evidence, a single positive exact-lowering case can be
  over-read as broad source compatibility.
- Rollback/mitigation: pair every positive slice with representative negative
  cases from the same source family.

### Engineering Task 6: Emit A Stable Story 4 Legacy-Lowering Artifact Bundle

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable bundle or rerunnable checker for the optional
  legacy-source slice.
- The bundle distinguishes positive exact-lowering cases from unsupported or
  deferred parity cases.
- The emitted output is reusable by Story 5 and later publication work.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example `benchmarks/density_matrix/artifacts/phase3_task1/story4_legacy/`).
- [ ] Record source label, eligibility status, first unsupported condition when
      applicable, and canonical planner-entry summary metadata.
- [ ] Keep the emitted record shape aligned with the existing planner-surface
      bundles from Stories 1 and 2 wherever fields overlap.
- [ ] Keep the artifact focused on exact in-bounds lowering rather than runtime
      execution outcomes.
- [ ] Record rerun commands and software metadata with the bundle.

**Evidence produced**
- One stable Story 4 legacy-lowering bundle or checker.
- One reusable output schema for later unsupported-boundary and paper packaging.

**Risks / rollback**
- Risk: if Story 4 evidence stays ad hoc, later readers may not be able to tell
  which legacy-source cases were actually supported.
- Rollback/mitigation: emit one stable artifact that records eligibility and
  unsupported reasons explicitly.

### Engineering Task 7: Document And Run The Story 4 Exact-Lowering Gate

**Implements story**
- `Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the optional legacy-source boundary precisely.
- Fast Story 4 regression checks and the Story 4 bundle run successfully.
- Story 4 closes with explicit evidence that optional support is narrow and
  exact.

**Execution checklist**
- [ ] Document the optional nature of legacy-source lowering and its exact
      in-bounds boundary.
- [ ] Explain that unsupported legacy-source features remain deferred or
      unsupported rather than best-effort supported.
- [ ] Run focused Story 4 regression coverage and verify the emitted bundle.
- [ ] Record stable references for Story 5 and later Phase 3 tasks.

**Evidence produced**
- Passing Story 4 regression checks.
- One stable Story 4 bundle or checker reference.

**Risks / rollback**
- Risk: Story 4 can inadvertently overstate support if its documentation focuses
  only on the positive slice.
- Rollback/mitigation: always document the optional slice together with the
  explicit non-goals and negative evidence.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- at least one eligible `qgd_Circuit` or `Gates_block` path lowers exactly into
  the canonical planner surface,
- legacy-source eligibility is explicit and tightly bounded to the frozen Phase
  3 support matrix,
- no implicit noise inference or heuristic gate rewriting is part of the
  supported Story 4 slice,
- representative negative cases prove unsupported legacy-source features do not
  gain accidental support,
- one stable Story 4 legacy-lowering bundle or checker exists for review and
  later reuse,
- and unsupported planner-boundary closure remains clearly assigned to Story 5.

## Implementation Notes

- Prefer exact gate-list walking over higher-level circuit rewriting for Story 4.
- Reuse the same canonical planner schema already established by Stories 1
  through 3 so optional legacy-source support does not invent a second planner
  contract.
- Treat every positive exact-lowering case as evidence for a narrow slice, not
  for general legacy-source parity.
