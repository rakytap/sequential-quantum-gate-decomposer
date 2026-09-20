# Story 4 Implementation Plan

## Story Being Implemented

Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise
Boundaries

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the first positive fused path into an exact-first semantic
claim:

- fused execution remains faithful to descriptor-defined qubit remapping,
  parameter routing, and gate or noise order,
- explicit noise operations remain hard semantic boundaries for the minimum
  fused baseline unless a separately validated exact rule is introduced,
- supported fused cases are checked against the sequential density reference with
  the frozen Task 4 exactness thresholds,
- and Story 4 closes positive semantic-preservation for fused execution without
  yet claiming full classification closure or phase-level performance closure.

Out of scope for this story:

- descriptor-local eligibility definition already owned by Story 1,
- the first positive structured fused-runtime slice already owned by Story 2,
- shared fused-capable reuse across continuity and microcases already owned by
  Story 3,
- explicit fused versus unfused versus deferred classification closure owned by
  Story 5,
- stable fused-output and provenance packaging owned by Story 6,
- and threshold-or-diagnosis benchmark closure owned by Story 7.

## Dependencies And Assumptions

- Stories 1 through 3 already define the eligibility rule, positive structured
  fused path, and shared fused-capable reuse surface that Story 4 must validate
  semantically.
- The frozen source-of-truth contract is `TASK_4_MINI_SPEC.md`,
  `TASK_4_STORIES.md`, `DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-004`,
  `P3-ADR-005`, and `P3-ADR-008`.
- Task 2 already froze the descriptor semantic contract Story 4 must honor in
  `squander/partitioning/noisy_planner.py`, including ordered `members`,
  `canonical_operation_indices`, remapping fields, and `parameter_routing`.
- Task 3 already provides the exact reference and positive runtime semantic
  substrate Story 4 should reuse through:
  - `execute_partitioned_density()`,
  - `execute_sequential_density_reference()`,
  - `execute_partitioned_with_reference()`,
  - `runtime_semantics_validation.py`,
  - and the emitted bundle under
    `benchmarks/density_matrix/artifacts/partitioned_runtime/story4_semantics/`.
- The fused baseline remains conservative unitary-island fusion inside noisy
  partitions. Story 4 should not require channel-native noisy-block fusion.
- The most direct real fused path is to build descriptor-local 1- and 2-qubit
  unitary kernels and apply them through the density backend's local-unitary
  primitive rather than through a separate fused `NoisyCircuit` abstraction.
- The frozen exactness thresholds for fused supported cases remain:
  - maximum Frobenius-norm density difference `<= 1e-10`,
  - `|Tr(rho) - 1| <= 1e-10`,
  - `rho.is_valid(tol=1e-10)`,
  - and maximum absolute energy error `<= 1e-8` where continuity observable
    comparison is exercised.
- Story 4 should prefer auditable semantic fidelity to aggressive fused-region
  enlargement.

## Engineering Tasks

### Engineering Task 1: Freeze The Positive Fused Semantic-Preservation Rule

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- docs | validation automation

**Definition of done**
- Story 4 defines one explicit positive semantic-preservation rule for supported
  fused execution.
- The rule distinguishes faithful fused execution from fused execution that only
  appears plausible numerically.
- The rule stays narrow enough that later classification and benchmark work can
  build on it cleanly.

**Execution checklist**
- [ ] Freeze the supported meaning of exact fused execution for the minimum
      baseline around descriptor-defined remapping, parameter routing, and
      explicit noise boundaries.
- [ ] Define what counts as faithful handling of explicit noise members in the
      minimum unitary-island fused path.
- [ ] Freeze the minimum semantic review fields needed for Story 4 artifacts.
- [ ] Keep classification closure and phase-level performance closure outside the
      Story 4 bar.

**Evidence produced**
- One stable Story 4 fused semantic-preservation rule.
- One clear boundary between positive semantic validation and later
  classification or benchmark closure.

**Risks / rollback**
- Risk: if Story 4 defines fused semantics too loosely, later exactness claims
  will rest on ambiguous runtime meaning.
- Rollback/mitigation: freeze the positive fused semantic rule before broadening
  benchmark interpretation.

### Engineering Task 2: Lower Eligible Spans While Preserving Descriptor-Defined Remapping And Routing

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- code | tests

**Definition of done**
- Supported fused execution applies descriptor-defined remapping and parameter
  routing explicitly.
- Fused lowering remains auditable back to the descriptor contract.
- The implementation does not depend on hidden fused-only qubit-layout or
  parameter-order assumptions.

**Execution checklist**
- [ ] Reuse `local_to_global_qbits`, `global_to_local_qbits`,
      `requires_remap`, and `parameter_routing` or the smallest auditable
      successors when building fused runtime work.
- [ ] Ensure fused execution does not bypass or privately reinterpret descriptor
      routing metadata.
- [ ] Add focused fixtures where remapping or multi-parameter `U3` routing is
      nontrivial inside or adjacent to a fused span.
- [ ] Add focused checks proving fused lowering stays rooted in descriptor
      metadata.

**Evidence produced**
- One reviewable fused remapping layer rooted in descriptor metadata.
- One reviewable fused parameter-routing layer rooted in descriptor metadata.

**Risks / rollback**
- Risk: fused results may look numerically plausible while still relying on
  implicit local indexing or parameter-order assumptions.
- Rollback/mitigation: make fused remapping and routing auditable from descriptor
  metadata directly.

### Engineering Task 3: Enforce Explicit Noise Boundaries In The Minimum Fused Baseline

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- code | tests

**Definition of done**
- Explicit supported noise operations remain hard semantic boundaries in the
  minimum fused baseline.
- Fused execution does not silently absorb, skip, or reorder noise across those
  boundaries.
- Boundary-sensitive supported cases remain auditable at execution time.

**Execution checklist**
- [ ] Keep fused-region construction bounded by explicit noise members unless a
      separately validated exact rule is added later.
- [ ] Preserve the relative position of explicit noise with respect to neighboring
      fused and unfused gate regions.
- [ ] Add focused checks showing the minimum fused baseline does not absorb
      supported noise operations into a unitary-only fused block.
- [ ] Reuse sparse, periodic, and dense local-noise placements when selecting
      boundary-sensitive fixtures.

**Evidence produced**
- One explicit fused-boundary rule for supported noise operations.
- Focused regression coverage for noise-boundary fidelity.

**Risks / rollback**
- Risk: a fused runtime may appear fast while silently weakening the exact noisy
  semantics Phase 3 is supposed to preserve.
- Rollback/mitigation: make explicit noise boundaries contract-defining for the
  minimum fused baseline.

### Engineering Task 4: Add Boundary-Sensitive Fused Fixtures Across Representative Workloads

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 includes supported cases that stress fused spans near explicit noise
  members, remapping boundaries, and parameter-routing boundaries.
- The fixtures are strong enough to expose silent semantic drift.
- The fixture set remains smaller than the full Story 7 benchmark inventory.

**Execution checklist**
- [ ] Select at least one microcase and at least one structured case with
      explicit noise adjacent to a candidate fused span.
- [ ] Include at least one case where remapping or parameter routing is
      nontrivial near a fused span.
- [ ] Reuse continuity observable comparison only where it helps validate fused
      semantics rather than broadening the fixture set unnecessarily.
- [ ] Record which fixtures later Story 7 work can reuse directly.

**Evidence produced**
- Boundary-sensitive fused semantic fixtures.
- Reviewable evidence that supported fused execution does not silently change
  descriptor semantics.

**Risks / rollback**
- Risk: semantic bugs near explicit noise boundaries may remain hidden if Story
  4 tests only easy fused regions.
- Rollback/mitigation: include boundary-sensitive fused fixtures explicitly.

### Engineering Task 5: Add A Focused Story 4 Fused-Semantics Validation Gate

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- tests | validation automation

**Definition of done**
- Story 4 has a rerunnable validation layer dedicated to fused semantic
  preservation.
- The validator checks the frozen exactness thresholds on representative fused
  cases.
- The validator remains narrower than the later threshold-or-diagnosis benchmark
  package.

**Execution checklist**
- [ ] Add focused Story 4 checks in `tests/partitioning/test_partitioned_runtime.py`.
- [ ] Add a Story 4 validator under
      `benchmarks/density_matrix/partitioned_runtime/`, with
      `fused_semantics_validation.py` as the primary checker.
- [ ] Reuse `execute_partitioned_with_reference()` or the smallest auditable
      successor to compare fused execution against the sequential density
      baseline.
- [ ] Assert the frozen density and trace thresholds, and where continuity
      observable comparison is exercised assert the frozen energy threshold.

**Evidence produced**
- One rerunnable Story 4 fused-semantics validation surface.
- Fast regression coverage for fused semantic fidelity.

**Risks / rollback**
- Risk: Story 4 may close on informal spot checks rather than on a repeatable
  exactness gate.
- Rollback/mitigation: require one dedicated fused-semantics validator before
  closure.

### Engineering Task 6: Emit A Stable Story 4 Fused-Semantics Bundle

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- validation automation | docs

**Definition of done**
- Story 4 emits one stable machine-reviewable bundle or rerunnable checker for
  fused semantic preservation.
- The bundle records the exactness metrics and fused-boundary evidence needed to
  review supported cases.
- The output is reusable by later classification and performance work.

**Execution checklist**
- [ ] Add a dedicated Story 4 artifact location
      (for example `benchmarks/density_matrix/artifacts/partitioned_runtime/fused_semantics/`).
- [ ] Emit representative fused cases through one stable schema.
- [ ] Record provenance, fused-path labels, exactness metrics, trace validity,
      and any boundary-review metadata needed for semantic audit.
- [ ] Record rerun commands and software metadata with the emitted bundle.

**Evidence produced**
- One stable Story 4 fused-semantics bundle or checker.
- One reusable semantic-audit surface for later Task 4 work.

**Risks / rollback**
- Risk: if Story 4 emits only ephemeral debug output, later validation and paper
  preparation will not have a stable semantic bundle to cite.
- Rollback/mitigation: emit one machine-reviewable fused-semantics bundle and
  keep it narrow.

### Engineering Task 7: Document And Run The Story 4 Fused-Semantics Gate

**Implements story**
- `Story 4: Fused Execution Preserves Exact Noisy Semantics Around Explicit Noise Boundaries`

**Change type**
- docs | tests | validation automation

**Definition of done**
- Developer-facing notes explain the supported Story 4 fused semantic rules.
- Fast regression checks and the Story 4 fused-semantics bundle run
  successfully.
- Story 4 closes with a stable review path for positive fused semantic
  preservation.

**Execution checklist**
- [ ] Document the explicit fused semantic-preservation rule and the minimum
      noise-boundary handling rule.
- [ ] Explain how Story 4 differs from Story 1 eligibility and Story 2 positive
      fused execution.
- [ ] Run focused Story 4 regression coverage and verify
      `benchmarks/density_matrix/partitioned_runtime/fused_semantics_validation.py`.
- [ ] Record stable test and artifact references for Stories 5 through 7 and
      later Phase 3 tasks.

**Evidence produced**
- Passing Story 4 fused-semantics regression checks.
- One stable Story 4 semantic-bundle or checker reference.

**Risks / rollback**
- Risk: Story 4 may appear complete while still leaving implementers unsure how
  fused semantic preservation is reviewed consistently.
- Rollback/mitigation: document the rules and require a rerunnable semantic
  bundle.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- supported fused execution applies descriptor-defined remapping and parameter
  routing explicitly,
- explicit noise boundaries remain first-class semantic boundaries for the
  minimum fused baseline,
- representative fused cases satisfy the frozen exactness thresholds against the
  sequential density reference,
- one stable Story 4 fused-semantics bundle or checker exists for later reuse,
- and explicit classification closure, stable output packaging, and phase-level
  performance interpretation remain clearly assigned to later stories.

## Implementation Notes

- Prefer conservative fused-region boundaries that are easy to audit over
  aggressive fused-region growth that is hard to justify semantically.
- Treat the sequential density reference as the correctness oracle, not as a
  hidden fallback.
- Keep Story 4 focused on positive semantic fidelity rather than on the full
  negative taxonomy.
