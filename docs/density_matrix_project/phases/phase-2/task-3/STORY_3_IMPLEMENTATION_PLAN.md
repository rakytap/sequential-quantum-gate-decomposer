# Story 3 Implementation Plan

## Story Being Implemented

Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution
With The First Unsupported Condition

This is a Layer 4 engineering plan for implementing the third behavioral slice
from `TASK_3_STORIES.md`.

## Scope

This story turns the frozen unsupported bridge contract for Task 3 into
explicit, auditable runtime and validation behavior:

- unsupported circuit sources, unsupported lowered gates, fused blocks, and
  unsupported noise insertions fail before execution,
- the first unsupported bridge condition is reported deterministically and
  consistently,
- unsupported requests do not silently rewrite, omit, reroute, or fall through
  to a standalone-only density path,
- and validation tooling records unsupported bridge outcomes as structured
  evidence instead of generic crashes or silent disappearance.

Out of scope for this story:

- widening the bridge support surface beyond the frozen Phase 2 contract,
- 1 to 3 qubit supported micro-validation owned by Story 2,
- workflow-scale 4/6/8/10 bridge completion owned by Story 4,
- the final provenance and publication bundle owned by Story 5,
- and backend-selection fallback rules already frozen primarily under Task 1
  except where they intersect directly with bridge behavior.

## Dependencies And Assumptions

- Story 1 is already in place: the supported positive bridge slice exists on the
  VQE path and now exposes stable bridge metadata through
  `describe_density_bridge()` plus the Story 1 `bridge_*` artifact vocabulary.
- Story 2 is already in place: the mandatory supported bridge surface is defined
  and locally validated.
- The current implementation already contains several relevant unsupported-case
  hooks in `validate_density_anchor_support()`,
  `lower_anchor_circuit_to_noisy_circuit()`, and
  `append_density_noise_for_gate_index()`.
- `benchmarks/density_matrix/story2_vqe_density_validation.py` already emits at
  least one structured unsupported artifact and therefore provides the natural
  pattern for Story 3 negative evidence.
- Current implementation-backed constraint from Story 1: unsupported bridge
  classification should prefer validator-driven failures before any
  parameter-sensitive `get_Qiskit_Circuit()` export path is needed.
- The frozen source-of-truth decisions remain `P2-ADR-011`, `P2-ADR-012`,
  `TASK_3_MINI_SPEC.md`, and the bridge/support-matrix closure decisions in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Story 3 should complete unsupported behavior without reopening phase-level
  bridge or support-matrix decisions.

## Engineering Tasks

### Engineering Task 1: Centralize Bridge Support-Surface Preflight Validation

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- code

**Definition of done**
- One shared preflight validator owns the Task 3 bridge support-surface decision.
- The same bridge checks are applied consistently to fixed-parameter execution,
  optimization entry points, and validation helper paths that can reach the
  density bridge.
- Unsupported requests fail before lowering, density evolution, or external
  comparison begins.

**Execution checklist**
- [ ] Refactor or extend the current bridge validation so
      `validate_density_anchor_support()` remains the single support-surface
      authority for Task 3.
- [ ] Ensure all bridge-relevant exact-noisy entry points use the same validator
      rather than duplicating partial checks.
- [ ] Keep the validator phase-contract driven and avoid embedding ad hoc bridge
      scope decisions in downstream helpers.

**Evidence produced**
- One reviewable support-surface authority for Task 3 bridge requests.
- Clear pre-execution failures instead of downstream accidental exceptions.

**Risks / rollback**
- Risk: split or duplicated validation logic can drift and produce inconsistent
  unsupported behavior across bridge entry points.
- Rollback/mitigation: keep one validator as the only source of Task 3
  support-matrix truth and call it everywhere.

### Engineering Task 2: Reject Unsupported Circuit Sources Explicitly And Deterministically

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- code | tests

**Definition of done**
- Unsupported circuit sources fail explicitly before bridge execution.
- The guaranteed source remains the generated default `HEA` circuit, with any
  optional extension cases kept clearly outside the mandatory support claim.
- Unsupported source failures are specific enough to tell the caller why the
  bridge request is out of scope.

**Execution checklist**
- [ ] Tighten source validation for unsupported `HEA_ZYZ`, missing generated
      circuit state, binary-imported gate lists, and unsupported manual-circuit
      paths where relevant.
- [ ] Preserve the distinction between guaranteed generated-`HEA` behavior and
      any optional future extension boundary.
- [ ] Add focused negative tests that assert unsupported source failures occur
      before lowering begins.

**Evidence produced**
- Focused negative tests for unsupported circuit-source categories.
- Stable unsupported-source error wording tied to the frozen bridge contract.

**Risks / rollback**
- Risk: ambiguous source validation can make unsupported manual or alternate
  ansatz requests appear partially supported.
- Rollback/mitigation: validate source identity up front and hard-error anything
  outside the guaranteed bridge surface.

### Engineering Task 3: Detect And Report The First Unsupported Gate Or Lowering Condition

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- code | tests

**Definition of done**
- Unsupported lowered gates or fused-block conditions fail deterministically
  before execution.
- Error reporting identifies the first unsupported gate or lowering category.
- Silent omission, partial lowering, or bridge-surface rewriting is not possible
  on the mandatory Task 3 path.

**Execution checklist**
- [ ] Extend current gate-surface validation so the first unsupported lowered
      gate is surfaced predictably.
- [ ] Keep required support, optional documented extensions, and deferred
      behavior visibly distinct.
- [ ] Add focused negative tests that assert the first unsupported gate or
      lowering condition appears in the failure output.

**Evidence produced**
- Reviewable first-failure reporting for unsupported gate and lowering
  violations.
- Negative regression tests that lock down deterministic first-unsupported
  behavior.

**Risks / rollback**
- Risk: vague aggregate failures make debugging and auditability weak and may
  hide which bridge boundary actually failed first.
- Rollback/mitigation: report the first unsupported gate or lowering condition
  consistently so tests can pin the behavior down.

### Engineering Task 4: Detect And Report Unsupported Ordered-Noise Insertions And Noise Types

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- code | tests

**Definition of done**
- Unsupported noise insertions fail deterministically before execution.
- The first unsupported noise condition is reported clearly, including whether
  the defect is an unsupported type or an invalid ordered insertion.
- Required local-noise behavior remains distinct from optional or deferred noise
  families.

**Execution checklist**
- [ ] Tighten `append_density_noise_for_gate_index()` and related validation so
      unsupported density-noise types and invalid insertion positions are
      surfaced predictably.
- [ ] Preserve the frozen required local-noise contract and avoid broadening it
      in Story 3.
- [ ] Add focused negative tests for unsupported noise-channel and
      ordered-insertion defects.

**Evidence produced**
- Reviewable first-failure reporting for unsupported noise conditions.
- Negative tests locking down deterministic noise-boundary behavior.

**Risks / rollback**
- Risk: unsupported noise cases may slip through as generic execution failures
  or produce misleading partially lowered outputs.
- Rollback/mitigation: validate noise type and insertion semantics before bridge
  execution begins.

### Engineering Task 5: Ban Silent Reroute, Silent Rewriting, And Standalone-Only Bypass Paths

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- code | tests

**Definition of done**
- Unsupported bridge requests cannot silently bypass the documented VQE bridge
  path.
- Unsupported requests do not degrade into partial bridge execution or a
  standalone-only density route while still appearing successful.
- Bridge failure remains a hard yes/no contract for the mandatory Phase 2 path.

**Execution checklist**
- [ ] Audit the bridge call path to ensure unsupported requests cannot continue
      through hidden fallback or alternative density helpers.
- [ ] Add focused negative regression coverage for reroute or silent-rewrite
      risks where practical.
- [ ] Keep Story 3 aligned with Task 1 no-fallback rules without turning it into
      a second backend-selection contract.

**Evidence produced**
- Negative tests showing unsupported bridge requests do not reroute or degrade.
- Clear reproducible errors showing bridge failure occurred before execution.

**Risks / rollback**
- Risk: unsupported requests may still appear to work if a nearby density path is
  silently invoked.
- Rollback/mitigation: perform explicit bridge support checks before any density
  execution helper is called.

### Engineering Task 6: Emit Structured Unsupported Bridge Artifacts

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Unsupported bridge cases are emitted as structured, reviewable artifacts.
- Validation tooling distinguishes unsupported bridge outcomes from supported
  executions and from numeric exactness failures owned elsewhere.
- Unsupported bridge output is stable enough to be cited in later docs and
  provenance bundles.

**Execution checklist**
- [ ] Extend the unsupported artifact pattern in
      `benchmarks/density_matrix/story2_vqe_density_validation.py` for bridge
      boundary cases.
- [ ] Use stable fields for unsupported category, first unsupported condition,
      source type, backend, and case identity.
- [ ] Keep the negative artifact vocabulary compatible with the Story 1
      supported fields where that makes comparison easier, especially
      `bridge_source_type` and stable case identity fields.
- [ ] Keep unsupported bridge outcomes explicit rather than disappearing into
      generic script failures or silent skips.

**Evidence produced**
- Stable structured unsupported-case artifacts for relevant Task 3 negative
  cases.
- Reviewable negative evidence distinct from supported bridge bundles.

**Risks / rollback**
- Risk: unsupported bridge cases may become untraceable if they are mixed into
  supported bundles or emitted only as generic crashes.
- Rollback/mitigation: keep unsupported status typed, intentional, and
  separately auditable.

### Engineering Task 7: Add Focused Regression Coverage And A Small Unsupported Taxonomy

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- tests | docs | validation automation

**Definition of done**
- The standard regression surface contains representative negative cases for the
  major Task 3 unsupported categories.
- Developer-facing guidance explains which bridge requests are guaranteed,
  optional, deferred, or unsupported in Phase 2.
- Error categories are stable enough to serve as test targets and review
  evidence.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` with negative bridge cases for unsupported
      source, gate/lowering, and noise conditions.
- [ ] Record a small supported/unsupported taxonomy in one stable bridge-facing
      location near the Story 3 workflow.
- [ ] Keep validation outputs and test expectations aligned with the same
      taxonomy rather than ad hoc strings.
- [ ] Reuse the Story 1 bridge metadata vocabulary in test helpers and artifacts
      so supported and unsupported evidence can be compared without a schema
      translation step.
- [ ] Avoid implying broader source or gate support in the developer-facing
      notes.

**Evidence produced**
- Negative pytest coverage across the main unsupported Task 3 categories.
- Stable unsupported-case taxonomy usable by tests and review.

**Risks / rollback**
- Risk: drifting or inconsistent error text weakens reproducibility and makes
  later provenance work harder.
- Rollback/mitigation: keep a small explicit taxonomy tied directly to the
  frozen bridge support surface and reuse it everywhere.

### Engineering Task 8: Run Story 3 Validation And Confirm Unsupported Outcomes Are Explicit

**Implements story**
- `Story 3: Out-Of-Scope Circuit, Gate, And Noise Requests Fail Before Execution With The First Unsupported Condition`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 3 negative regression tests pass.
- Representative unsupported bridge cases are emitted through stable validation
  paths as structured unsupported outcomes.
- Story 3 completion is backed by reviewable negative evidence rather than by
  code changes alone.

**Execution checklist**
- [ ] Run the focused Story 3 regression tests.
- [ ] Run the relevant validation commands that emit structured unsupported
      bridge results.
- [ ] Verify unsupported bridge cases fail before execution and are not counted
      as supported bridge passes.
- [ ] Record stable test and artifact references for later Task 3 docs and
      bundles.

**Evidence produced**
- Passing focused Story 3 negative pytest coverage.
- Stable unsupported-case artifact references from validation runs.

**Risks / rollback**
- Risk: Story 3 can appear complete while still lacking auditable proof that
  unsupported bridge outcomes are explicit and reproducible.
- Rollback/mitigation: treat negative artifact references and focused test runs
  as part of the exit gate, not optional cleanup.

## Exit Criteria

Story 3 is complete only when all of the following are true:

- bridge requests outside the frozen Phase 2 support matrix fail before
  execution,
- unsupported circuit-source, gate/lowering, and noise conditions are all
  covered by deterministic failures,
- the first unsupported bridge condition is reported clearly and reproducibly,
- unsupported requests do not silently rewrite, reroute, or fall through to a
  standalone-only density path,
- and regression plus validation tooling can distinguish unsupported bridge
  outcomes from supported executions and from unrelated numeric failures.

## Implementation Notes

- The current `validate_density_anchor_support()` path already covers several
  Story 3 categories and should evolve into the full bridge support-surface
  authority rather than being replaced by a second framework.
- `lower_anchor_circuit_to_noisy_circuit()` and
  `append_density_noise_for_gate_index()` are the natural places to preserve
  first-unsupported reporting for gate and noise failures.
- Story 1 already proved that supported bridge cases can be described through a
  stable VQE-side metadata surface. Story 3 should extend the same vocabulary to
  negative artifacts rather than inventing a separate unsupported-only schema.
- `benchmarks/density_matrix/story2_vqe_density_validation.py` already emits
  structured unsupported artifacts for at least one density-backend mismatch
  case. Story 3 should extend that pattern where it adds Task 3 negative
  evidence.
- Story 1 also showed that parameter-sensitive export paths are a poor primary
  mechanism for unsupported classification. Story 3 should prefer validator- and
  lowering-path failures that happen before Qiskit export is needed.
- Optional extensions should stay clearly separated from guaranteed behavior so
  Story 3 strengthens unsupported-case clarity without accidentally widening the
  bridge contract.
