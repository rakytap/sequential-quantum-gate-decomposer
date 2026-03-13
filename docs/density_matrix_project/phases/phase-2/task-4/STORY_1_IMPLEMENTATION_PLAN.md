# Story 1 Implementation Plan

## Story Being Implemented

Story 1: Required Local Noise Models Execute On The Supported VQE Path As
Explicit Ordered Operations

This is a Layer 4 engineering plan for implementing the first behavioral slice
from `TASK_4_STORIES.md`.

## Scope

This story turns the frozen required local-noise baseline into explicit,
auditable runtime behavior on the existing VQE-facing density path:

- the required local models can be configured through the documented
  `density_matrix` plus `density_noise` surface on
  `qgd_Variational_Quantum_Eigensolver_Base`,
- each required model lowers into explicit ordered `NoiseOperation` content on
  the supported generated-`HEA` bridge path rather than through a hidden or
  reconstructed mechanism,
- supported positive cases visibly use the requested required local model rather
  than silently substituting whole-register depolarizing,
- and the resulting positive slice is reviewable enough that later stories can
  add micro-validation thresholds, optional-versus-required classification,
  unsupported-case hard errors, workflow-scale sufficiency, and publication
  packaging without replacing the noise path itself.

Out of scope for this story:

- the 1 to 3 qubit exactness and mixed-sequence validation matrix owned by Story
  2,
- optional-versus-required labeling closure owned by Story 3,
- unsupported and deferred noise hard-error closure owned by Story 4,
- the 4/6/8/10 workflow-scale sufficiency package owned by Story 5,
- the full reproducibility and publication bundle owned by Story 6,
- whole-register depolarizing as a scientific baseline,
- and any attempt to treat generalized amplitude damping, coherent
  over-rotation, correlated noise, readout noise, calibration-aware noise, or
  non-Markovian noise as part of the Story 1 required slice.

## Dependencies And Assumptions

- Task 1 already provides explicit backend selection and the no-fallback
  contract for `density_matrix`.
- Task 3 already established the supported generated-`HEA` bridge path through
  `validate_density_anchor_support()`,
  `lower_anchor_circuit_to_noisy_circuit()`,
  `append_density_noise_for_gate_index()`,
  `evaluate_density_matrix_backend()`, and `describe_density_bridge()`.
- The current Python-side noise surface already exists in
  `_normalize_density_noise_spec()` and `set_Density_Matrix_Noise()` inside
  `squander/VQA/qgd_Variational_Quantum_Eigensolver_Base.py`.
- The current density-matrix substrate already exposes the required model-specific
  operations:
  `NoisyCircuit::add_local_depolarizing()`,
  `NoisyCircuit::add_amplitude_damping()`, and
  `NoisyCircuit::add_phase_damping()`.
- Existing positive fixtures in `tests/VQE/test_VQE.py` and validation scaffolds
  in `benchmarks/density_matrix/story2_vqe_density_validation.py` are the right
  starting points for Story 1 evidence.
- Story 1 should harden and expose the required positive local-noise slice; it
  should not reopen the phase-level support split in `P2-ADR-012`, the anchor
  workflow in `P2-ADR-013`, or the benchmark and threshold package in
  `P2-ADR-014` and `P2-ADR-015`.

## Engineering Tasks

### Engineering Task 1: Stabilize The Canonical Required Local-Noise Configuration Surface

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- code | tests

**Definition of done**
- The VQE-facing `density_noise` surface accepts the required local models in
  one canonical vocabulary for the supported Phase 2 path.
- Normalization preserves ordered noise metadata needed by the bridge:
  model identity, target qubit, `after_gate_index`, and fixed value.
- Story 1 positive cases can rely on one stable normalized representation rather
  than ad hoc per-test noise dictionaries.

**Execution checklist**
- [ ] Review `_normalize_density_noise_spec()` and `set_Density_Matrix_Noise()`
      as the canonical Python-side boundary for required local-noise
      configuration.
- [ ] Keep canonical names aligned with the frozen Task 4 contract:
      `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- [ ] Preserve order and fixed-value metadata through the wrapper boundary so the
      bridge receives a stable schedule vocabulary.
- [ ] Add or tighten focused tests proving normalized required-model schedules
      are stable and reviewable.

**Evidence produced**
- One stable normalized representation for required local-noise schedules on the
  VQE surface.
- Focused tests proving the canonical required-model vocabulary is preserved.

**Risks / rollback**
- Risk: aliasing or normalization drift can make the bridge appear correct while
  different layers disagree about the actual requested model names.
- Rollback/mitigation: keep one canonical representation at the Python boundary
  and validate it before relying on downstream bridge evidence.

### Engineering Task 2: Add Deterministic Per-Model Supported Fixtures For The Story 1 Positive Slice

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- code | tests | benchmark harness

**Definition of done**
- Story 1 has deterministic supported fixtures for each required local-noise
  model on the generated-`HEA` density path.
- Each fixture is narrow enough to localize regressions to one required model
  rather than only to a mixed schedule.
- Fixture metadata is stable enough to reuse across tests and validation
  commands.

**Execution checklist**
- [ ] Refine `build_story2_noise()` or add a tightly related per-model fixture
      helper so Story 1 can exercise each required local model separately.
- [ ] Keep fixtures on the supported anchor VQE path with explicit
      `density_matrix` selection and generated `HEA` circuitry.
- [ ] Give each supported fixture a stable identity and a clear statement of
      which required local model it is meant to exercise.
- [ ] Leave mixed required-model schedules to Story 2 rather than overloading the
      Story 1 fixture set.

**Evidence produced**
- Reusable deterministic Story 1 fixtures for
  `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- Stable per-fixture metadata for use in tests and artifacts.

**Risks / rollback**
- Risk: relying only on the existing mixed-noise fixture can hide which required
  model regressed.
- Rollback/mitigation: introduce one stable per-model positive fixture layer
  before broadening validation breadth.

### Engineering Task 3: Keep Required Local-Noise Lowering Explicit And Model-Specific On The Bridge Path

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- code | tests

**Definition of done**
- Each required local-noise request lowers through the supported bridge into the
  matching `NoisyCircuit` operation rather than a hidden substitute.
- Ordered noise insertion stays attached to the same bridged gate-index
  vocabulary the VQE workflow executes.
- No positive Story 1 case uses whole-register depolarizing as a stand-in for a
  required local-noise model.

**Execution checklist**
- [ ] Review `append_density_noise_for_gate_index()` and its call site inside
      `lower_anchor_circuit_to_noisy_circuit()`.
- [ ] Confirm the required model mapping remains explicit:
      `local_depolarizing` -> `add_local_depolarizing()`,
      `amplitude_damping` -> `add_amplitude_damping()`,
      `phase_damping` -> `add_phase_damping()`.
- [ ] Keep ordered noise insertion tied to `after_gate_index` rather than any
      heuristic or reconstructed placement rule.
- [ ] Add focused checks that required positive cases do not surface
      `add_depolarizing()` or a whole-register `depolarizing` operation name on
      the supported bridge path.

**Evidence produced**
- One explicit, reviewable lowering path for each required local-noise model.
- Focused checks proving the required models are not silently replaced by a
  whole-register baseline.

**Risks / rollback**
- Risk: the density workflow can return plausible energies while silently using
  the wrong noise primitive or wrong insertion location.
- Rollback/mitigation: treat operation-name and placement inspection as part of
  Story 1 closure, not optional debugging detail.

### Engineering Task 4: Expose Reviewable Metadata For Required Noise Identity, Order, And Placement

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- code | tests | validation automation

**Definition of done**
- Supported Story 1 cases expose machine-readable metadata showing which
  required local-noise model was lowered, in what order, and on which target
  location.
- The inspection surface is narrow and bridge-focused rather than a new generic
  public API for arbitrary circuit introspection.
- Reviewers can tell from stored metadata that the positive path used the
  requested local model explicitly.

**Execution checklist**
- [ ] Reuse `describe_density_bridge()` and the current `bridge_operations`
      vocabulary first.
- [ ] Add the smallest derived metadata needed for Story 1 review, such as
      model-specific noise sequences or placement summaries, only if the current
      artifact shape is too indirect.
- [ ] Ensure supported Story 1 metadata preserves model name,
      `source_gate_index`, `target_qbit`, and fixed noise value.
- [ ] Add focused assertions that required positive cases expose auditable noise
      identity and placement.

**Evidence produced**
- A reviewable metadata surface for required local-noise identity, order, and
  placement.
- Focused tests proving Story 1 can audit the positive local-noise path.

**Risks / rollback**
- Risk: if local-noise behavior is visible only through final scalar energies,
  Story 1 will be hard to distinguish from later exactness stories.
- Rollback/mitigation: keep one narrow machine-readable inspection vocabulary and
  reuse it across tests and validation artifacts.

### Engineering Task 5: Add Focused Regression Coverage For Each Required Local-Noise Model On The Supported VQE Path

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- tests

**Definition of done**
- Fast automated tests cover the positive supported path for each required local
  noise model individually.
- Test evidence shows the request reaches the density-backed VQE path, produces
  finite output, and exposes the correct required local-noise operation name in
  bridge metadata.
- Regression coverage remains small and deterministic enough to run regularly.

**Execution checklist**
- [ ] Extend `tests/VQE/test_VQE.py` with per-model supported-path cases for
      `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- [ ] Assert that each supported case exposes the requested required model in
      `describe_density_bridge()` metadata.
- [ ] Assert that Story 1 positive cases do not advertise whole-register
      `depolarizing` as the active bridge noise model.
- [ ] Keep full threshold closure and mixed-sequence validation out of this fast
      regression layer.

**Evidence produced**
- Focused pytest coverage for the required local-noise positive slice.
- Reviewable failures that localize regressions to a specific required local
  model.

**Risks / rollback**
- Risk: weak smoke tests can prove only that some noisy path ran, not that the
  requested required model executed.
- Rollback/mitigation: assert model-specific bridge metadata, not only final
  energy finiteness.

### Engineering Task 6: Emit A Stable Story 1 Validation Artifact Or Rerunnable Command For Required Local-Noise Cases

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- benchmark harness | validation automation

**Definition of done**
- Story 1 can emit at least one stable artifact or rerunnable command that shows
  the supported positive path for each required local-noise model.
- The artifact records the requested local model, ordered bridge metadata, and
  enough placement information to audit the positive slice outside the test
  layer.
- The output format is stable enough that later Task 4 stories can extend it
  rather than replace it.

**Execution checklist**
- [ ] Extend `benchmarks/density_matrix/story2_vqe_density_validation.py` or a
      tightly related successor with Story 1 per-model output.
- [ ] Record backend, source type, required local-noise identity, ordered bridge
      metadata, and target placement information for each supported case.
- [ ] Keep the Story 1 artifact narrow to positive required-model evidence rather
      than the full workflow or publication bundle.
- [ ] Make the artifact or command stable enough to be cited again by later Task
      4 stories.

**Evidence produced**
- One stable Story 1 artifact or rerunnable command for each required local
  noise model.
- A reusable artifact schema for later Task 4 evidence packaging.

**Risks / rollback**
- Risk: ad hoc console inspection will make it hard to prove that the positive
  local-noise slice was ever reviewed.
- Rollback/mitigation: define one small machine-readable output now and extend it
  incrementally.

### Engineering Task 7: Update Developer-Facing Notes For The Required Local-Noise Entry Surface

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- docs

**Definition of done**
- Developer-facing notes identify the supported required local-noise entry
  surface on the VQE path.
- The notes make clear that Story 1 closes only the positive required-model
  slice, not the optional, deferred, unsupported, or full benchmark package.
- The documented support boundary matches `TASK_4_MINI_SPEC.md` and
  `TASK_4_STORIES.md`.

**Execution checklist**
- [ ] Update the most relevant VQE density-backend docstrings, examples, or
      developer notes near `density_noise` configuration and bridge inspection.
- [ ] Keep wording aligned with the frozen required baseline:
      `local_depolarizing`, `amplitude_damping`, and `phase_damping`.
- [ ] Refer threshold closure, optional classification, unsupported cases, and
      workflow-scale sufficiency to later Task 4 stories rather than overclaiming
      completion here.

**Evidence produced**
- Updated developer-facing notes for the Story 1 required local-noise slice.
- One stable place where the supported required-model boundary is documented.

**Risks / rollback**
- Risk: documentation can easily imply that the whole Task 4 contract is closed
  when only the positive required-model path is ready.
- Rollback/mitigation: tie notes directly to the same per-model fixtures and
  artifacts used by Story 1 validation.

### Engineering Task 8: Run Story 1 Validation And Confirm The Required Local-Noise Positive Slice

**Implements story**
- `Story 1: Required Local Noise Models Execute On The Supported VQE Path As Explicit Ordered Operations`

**Change type**
- tests | validation automation

**Definition of done**
- Focused Story 1 regression tests pass for each required local-noise model.
- At least one rerunnable validation command or artifact demonstrates the
  required-model positive path outside the fast test layer.
- Story 1 exits with reviewable local-noise evidence rather than code changes
  alone.

**Execution checklist**
- [ ] Run the focused Story 1 pytest coverage for each required local-noise
      model.
- [ ] Run the Story 1 validation artifact or command emission path.
- [ ] Confirm the resulting evidence is stored or named stably enough for later
      Task 4 stories to reuse.
- [ ] Verify that no positive Story 1 artifact shows whole-register depolarizing
      standing in for a required local-noise case.

**Evidence produced**
- Passing focused Story 1 pytest coverage.
- One stable artifact or command reference for the required local-noise positive
  slice.

**Risks / rollback**
- Risk: Story 1 can look complete while still lacking any stored evidence that
  the requested required models actually ran on the supported path.
- Rollback/mitigation: treat model-specific artifacts and metadata as part of the
  exit gate, not optional follow-up.

## Exit Criteria

Story 1 is complete only when all of the following are true:

- the required local-noise models `local_depolarizing`,
  `amplitude_damping`, and `phase_damping` can each be configured through the
  supported VQE-facing density-noise surface,
- supported generated-`HEA` density-path fixtures using each required local
  model execute through the documented VQE runtime,
- bridge metadata for supported Story 1 cases makes the required model identity,
  insertion order, target placement, and fixed value auditable,
- no positive Story 1 case relies on whole-register `depolarizing` as a
  substitute for a required local-noise model,
- at least one stable artifact or rerunnable command proves the required local
  positive slice outside the unit-test layer,
- and mixed-sequence validation, optional-versus-required closure,
  unsupported-case closure, workflow-scale sufficiency, and final provenance
  packaging remain clearly assigned to later Task 4 stories.

## Implementation Notes

- `_normalize_density_noise_spec()` and `set_Density_Matrix_Noise()` already
  provide the natural Python-side boundary for Story 1. Implementation should
  harden this path rather than introduce a second density-noise API.
- `append_density_noise_for_gate_index()` and
  `lower_anchor_circuit_to_noisy_circuit()` already provide the natural ordered
  lowering path for required local-noise models. Story 1 should keep one
  explicit model-to-operation mapping and reuse it.
- `describe_density_bridge()` and the current `bridge_operations` vocabulary in
  `benchmarks/density_matrix/story2_vqe_density_validation.py` are the most
  natural existing inspection surfaces for Story 1. Extend them minimally.
- The current mixed required-noise fixture in `build_story2_noise()` is useful
  background evidence, but Story 1 should add deterministic per-model fixtures
  so regressions can be localized to one required local-noise model at a time.
