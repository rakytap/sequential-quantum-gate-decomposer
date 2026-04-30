# Story 4 Implementation Plan

## Story Being Implemented

`Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

This is a Layer 4 engineering plan for implementing the fourth behavioral slice
from `TASK_1_STORIES.md`.

## Scope

This story turns backend provenance into a first-class validation and
reproducibility outcome for the frozen Phase 2 backend-selection contract:

- tests, benchmark outputs, and machine-readable artifacts explicitly state
  which backend executed each case,
- completed and unsupported cases preserve backend provenance rather than losing
  it in generic status output,
- reproducibility artifacts are structured so reviewers can reconstruct the
  selected backend, workflow surface, and support-matrix context,
- and the resulting evidence is publication-ready enough to support later Paper
  1 packaging.

Out of scope for this story:

- adding new backend functionality beyond the already frozen Story 1 to Story 3
  behavior,
- expanding the support matrix,
- replacing the existing benchmark logic with a new benchmark framework,
- and broader Phase 2 closure items such as 8/10-qubit evidence, runtime and
  peak-memory characterization, or the full final reproducibility bundle.

## Dependencies And Assumptions

- Story 1 already provides explicit backend selection and preserves the legacy
  `state_vector` default.
- Story 2 already provides a narrow positive `density_matrix` anchor path and
  emits initial JSON artifacts with backend fields for completed cases.
- Story 3 already provides structured `unsupported` benchmark artifacts and
  explicit hard-error unsupported behavior.
- The current provenance surface is uneven: some completed Story 2 artifacts
  already include `backend`, while some unsupported artifacts still need fuller
  provenance context to be publication-robust.

## Engineering Tasks

### Engineering Task 1: Define A Stable Backend-Provenance Artifact Schema

**Implements story**
- `Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

**Change type**
- docs | validation automation

**Definition of done**
- A single stable schema exists for machine-readable backend provenance across
  completed, failed, and unsupported validation cases.
- Required provenance fields are explicit rather than inferred from file names
  or ad hoc log messages.
- The schema is narrow enough for Phase 2 use, but stable enough for later
  benchmark expansion.

**Execution checklist**
- [ ] Define the required artifact fields for all Story 4 outputs, including at
      least backend, status, qubit count, topology, ansatz, layers,
      noise schedule, Hamiltonian metadata, and error or unsupported reason.
- [ ] Distinguish required fields shared by all cases from optional
      case-specific fields such as optimization trace content or Aer comparison
      metrics.
- [ ] Record the schema in a stable developer-facing location near the benchmark
      tooling.

**Evidence produced**
- A documented provenance schema for Phase 2 validation artifacts.
- Machine-readable artifacts that conform to the same field contract.

**Risks / rollback**
- Risk: artifact formats drift by script or case type, making later review and
  publication assembly brittle.
- Rollback/mitigation: define one shared schema first and make all Story 4
  outputs conform to it.

### Engineering Task 2: Make Completed Validation Artifacts Backend-Explicit Everywhere

**Implements story**
- `Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

**Change type**
- code | validation automation

**Definition of done**
- Every completed benchmark or validation artifact clearly names the backend
  that executed it.
- Backend provenance is preserved in both console summaries and JSON or other
  stored outputs.
- Reviewers do not need to infer the backend from code knowledge, defaults, or
  file naming conventions.

**Execution checklist**
- [ ] Audit current benchmark and validation scripts for completed-case outputs
      that already include backend versus those that do not.
- [ ] Extend completed-case artifacts so backend provenance is always serialized
      explicitly.
- [ ] Ensure the printed summaries for fixed-parameter cases and optimization
      traces also identify the executing backend.

**Evidence produced**
- Completed benchmark artifacts with explicit backend fields.
- Console or log summaries that name the backend used for each case.

**Risks / rollback**
- Risk: partial provenance makes positive results easy to misattribute during
  review or paper drafting.
- Rollback/mitigation: require backend fields in all completed artifacts and
  fail validation generation when they are absent.

### Engineering Task 3: Preserve Backend Provenance For Unsupported Outcomes

**Implements story**
- `Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

**Change type**
- code | validation automation

**Definition of done**
- Unsupported artifacts preserve the requested backend and the relevant workflow
  context, not just `status` and an error string.
- Unsupported outputs remain directly comparable with completed outputs at the
  metadata level.
- Reviewers can tell whether an unsupported outcome came from a requested
  `state_vector` or `density_matrix` workflow.

**Execution checklist**
- [ ] Extend the current unsupported artifact path so it captures the requested
      backend together with status, unsupported category, and reason.
- [ ] Preserve minimal workflow metadata such as qubit count, topology, ansatz,
      and noise schedule where available even on unsupported outcomes.
- [ ] Keep unsupported-case schema compatible with the completed-case schema.

**Evidence produced**
- Unsupported JSON artifacts with explicit backend provenance.
- Validation output that distinguishes unsupported backend intent from generic
  runtime failure.

**Risks / rollback**
- Risk: unsupported artifacts that omit backend context weaken the scientific
  value of Story 3’s hard-error behavior.
- Rollback/mitigation: treat backend and workflow metadata as required even for
  unsupported cases.

### Engineering Task 4: Propagate Backend Provenance Into Test And Validation Summaries

**Implements story**
- `Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

**Change type**
- tests | validation automation

**Definition of done**
- Fast regression outputs and benchmark summaries expose backend provenance in a
  stable, human-readable way.
- The backend naming used in test helpers, benchmark scripts, and serialized
  artifacts is consistent.
- Story 4 evidence is visible both to automated consumers and to human reviewers
  reading logs.

**Execution checklist**
- [ ] Identify where pytest-level validation summaries should expose backend
      context explicitly.
- [ ] Align benchmark-summary wording with the machine-readable backend labels.
- [ ] Ensure the same backend vocabulary is used across tests, benchmark
      summaries, and artifact payloads.

**Evidence produced**
- Human-readable validation output with backend-specific wording.
- Consistent backend label usage across regression and benchmark surfaces.

**Risks / rollback**
- Risk: inconsistent naming across logs and JSON artifacts can make provenance
  reconciliation error-prone.
- Rollback/mitigation: keep one canonical backend-label vocabulary and reuse it
  everywhere.

### Engineering Task 5: Strengthen Reproducibility Bundle Content Around Backend Choice

**Implements story**
- `Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

**Change type**
- code | docs | validation automation

**Definition of done**
- The reproducibility bundle for the current Story 1 to Story 3 slice records
  enough context to rerun the same backend path intentionally.
- Backend provenance is coupled with the specific workflow configuration that
  made the result possible.
- The current bundle is explicit about what is already included versus what
  remains open for full Phase 2 closure.

**Execution checklist**
- [ ] Include backend provenance alongside ansatz, layers, topology, noise
      schedule, Hamiltonian parameters, fixed parameter vectors or seeds, and
      status metadata.
- [ ] Record the current limitations of the reproducibility bundle so later
      phases do not mistake it for final Phase 2 closure.
- [ ] Add or update a lightweight artifact manifest or notes describing the
      current Story 4 bundle contents.

**Evidence produced**
- A more complete Story 4 reproducibility bundle or artifact manifest.
- Structured metadata sufficient to rerun the current implemented slice.

**Risks / rollback**
- Risk: backend provenance without the surrounding workflow metadata is not
  enough to reproduce results meaningfully.
- Rollback/mitigation: bundle backend choice together with the minimum workflow
  configuration required to replay the case.

### Engineering Task 6: Align Publication-Facing And Developer-Facing Notes With The Actual Artifact Surface

**Implements story**
- `Story 4: Validation And Reproducibility Artifacts Name The Backend Used`

**Change type**
- docs

**Definition of done**
- Developer-facing and publication-facing notes accurately describe the current
  provenance surface delivered by Stories 1 to 4.
- Documentation distinguishes the implemented artifact slice from the broader
  final Phase 2 reproducibility package.
- Paper-support notes can point to stable backend-explicit artifacts without
  overstating completeness.

**Execution checklist**
- [ ] Update the relevant benchmark or developer notes to describe the current
      artifact schema and backend-provenance behavior.
- [ ] Ensure publication-facing notes explain that backend provenance is now
      explicit, while broader benchmark-floor and reproducibility targets remain
      open Phase 2 work.
- [ ] Keep Story 4 wording aligned with the frozen contract and avoid implying
      that provenance work alone closes the whole benchmark package.

**Evidence produced**
- Updated documentation for backend-explicit validation artifacts.
- Publication-facing notes that correctly scope the current delivered slice.

**Risks / rollback**
- Risk: docs may overstate provenance completeness and confuse the current slice
  with full Phase 2 closure.
- Rollback/mitigation: document both the delivered provenance surface and the
  remaining open reproducibility work explicitly.

## Exit Criteria

Story 4 is complete only when all of the following are true:

- completed and unsupported validation artifacts explicitly name the backend
  involved,
- machine-readable outputs use a stable provenance schema,
- human-readable summaries use the same backend labels as the stored artifacts,
- reproducibility metadata is sufficient to identify and rerun the current
  backend-specific workflow slice,
- and reviewers can determine from stored evidence whether a case came from the
  legacy `state_vector` path or the Phase 2 `density_matrix` path.

## Implementation Notes

- Story 4 should build on the current Story 2 and Story 3 artifact surfaces
  rather than creating a second provenance mechanism.
- The backend field already exists in some positive Story 2 artifacts, and the
  `unsupported` status path already exists for at least one Story 3 case; Story
  4 should unify and complete that surface.
- Keep the distinction between standalone density-module capability and
  VQE-backed density-workflow provenance visible in the artifacts when useful.
- The full final Phase 2 benchmark floor, runtime and peak-memory package, and
  broader reproducibility bundle remain later work even after Story 4 is
  delivered.
