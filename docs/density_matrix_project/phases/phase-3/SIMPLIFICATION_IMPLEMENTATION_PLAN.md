# Phase 3 Simplicity Implementation Plan

This document turns the current branch simplicity review into a concrete
implementation plan.

Important framing:

- The current codebase is the source of truth.
- This is a behavior-preserving simplification plan, not a request to realign the
code with older planning docs.
- Ranking below is the recommended implementation order after considering payoff,
coupling, blast radius, and validation cost.

## Purpose

Phase 3 now contains a real planner surface, descriptor layer, runtime, evidence
pipelines, emitted artifacts, and publication-facing validation surfaces. The
main simplicity goal is to reduce accidental complexity without broadening the
supported path, changing the paper claim, or weakening machine-reviewable
evidence.

The plan focuses on six recurring complexity patterns in the current code:

1. duplicated evidence predicates and record-building logic,
2. many validation modules with nearly identical scaffolding,
3. repeated runtime validation and gate-lowering logic,
4. parallel planner, descriptor, and runtime schemas for the same concepts,
5. tests coupled to benchmark infrastructure instead of API behavior,
6. thin aliases and ambiguous names that expand the public surface without adding
  behavior.

## Non-Goals

- changing the supported noisy partitioning contract,
- broadening the planner or runtime support matrix,
- weakening or removing emitted artifact bundles,
- replacing the current evidence model with narrative-only checks,
- or reopening Phase 3 claim-boundary decisions.

## Ranking Summary


| Rank | Story                                                                        | Why this rank                                                                     | Risk       | Primary areas                                                                                      |
| ---- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------- |
| 1    | Shared evidence core replaces duplicated predicates and record assembly      | Highest simplicity gain per line changed; low behavior risk                       | Low        | `benchmarks/density_matrix/correctness_evidence`, `benchmarks/density_matrix/performance_evidence` |
| 2    | Validation slices express only unique behavior                               | Removes repetitive bundle boilerplate and shrinks test duplication                | Low-Medium | `benchmarks/density_matrix/*_validation.py`, `tests/partitioning/test_*evidence.py`                |
| 3    | Runtime contract checking and lowering flow become single-path               | Reduces repeated validation, repeated gate semantics, and confusing path handling | Medium     | `squander/partitioning/noisy_runtime.py`                                                           |
| 4    | Planner, descriptor, and runtime records carry one minimal model per concept | Largest structural simplification, but widest schema impact                       | High       | `squander/partitioning/noisy_planner.py`, `squander/partitioning/noisy_runtime.py`                 |
| 5    | Partitioning tests validate APIs directly                                    | Makes deeper refactors safer and the test surface easier to understand            | Medium     | `tests/partitioning/**`, shared test helpers                                                       |
| 6    | Public names and payloads expose one concept one way                         | Good cleanup value after structural work settles                                  | Low        | planner/runtime helpers, payload serialization, compatibility aliases                              |


## Story 1  (COMPLETED): Correctness And Performance Evidence Derive From One Shared Core

**Why ranked #1**

This story removes the most obvious duplication with the lowest semantic risk.
It should land first because it creates shared helpers that later stories can
reuse and test against.

**User/Research value**

- Evidence remains machine-reviewable, but the rule for what counts as
supported, correct, and reusable lives in one place.
- Future changes to evidence thresholds or required flags stop requiring parallel
edits in multiple pipelines.

**Given / When / Then**

- Given the current correctness-evidence and performance-evidence record
builders.
- When both pipelines need the same supported-case rule or the same runtime
result field set.
- Then they should consume one shared implementation rather than re-state the
rule in separate modules.

**Scope**

- In: shared supported-case predicate, shared runtime-result field extraction,
shared JSON artifact writing, and removal of no-behavior aliases in evidence
common modules.
- Out: changing case selection, thresholds, benchmark inventory, or artifact
filenames.

**Current complexity anchors**

- `benchmarks/density_matrix/correctness_evidence/records.py`
- `benchmarks/density_matrix/performance_evidence/records.py`
- `benchmarks/density_matrix/correctness_evidence/common.py`
- `benchmarks/density_matrix/performance_evidence/common.py`

**Acceptance signals**

- There is exactly one implementation of the counted-supported predicate.
- There is one shared helper for materializing common runtime result fields used
by correctness and performance records.
- Evidence common modules do not each carry their own copy of
`write_artifact_bundle`.
- Compatibility wrappers that add no behavior are either removed or clearly
marked as temporary adapters.

**Engineering tasks**

### Engineering Task 1.1: Extract one shared counted-supported rule

**Change type**

- code | tests

**Definition of done**

- Both evidence stacks call the same predicate implementation.
- No parallel copies of the same five-condition rule remain.

**Execution checklist**

- Move the counted-supported rule into one shared helper module.
- Update correctness and performance record builders to import that helper.
- Add regression coverage that proves both callers classify the same records
identically.

### Engineering Task 1.2: Extract one shared runtime-result field builder

**Change type**

- code | tests

**Definition of done**

- Shared runtime result fields are assembled by one helper.
- Performance fallback logic no longer hand-builds a near-copy of correctness
record fields.

**Execution checklist**

- Identify the common runtime payload, density metric, and integrity fields.
- Create one shared builder or adapter for those fields.
- Keep caller-specific extras outside the shared helper.
- Add snapshot-style checks for the shared field set.

### Engineering Task 1.3: Collapse duplicate evidence utility helpers

**Change type**

- code

**Definition of done**

- JSON bundle writing and other identical helpers are shared.
- No-op compatibility aliases are reduced to the minimum required for transition.

**Execution checklist**

- Share `write_artifact_bundle`.
- Review compatibility aliases in `performance_evidence/common.py`.
- Remove aliases that do not protect a real consumer boundary.

**Evidence produced**

- Updated evidence tests passing on unchanged artifacts.
- A smaller evidence-core API with one supported-case rule and one shared field
builder.

**Risks / rollback**

- Risk: subtle schema drift while deduplicating common fields.
- Rollback/mitigation: freeze current field-level expectations in tests before
refactor and keep artifact filenames unchanged.

## Story 2 (COMPLETED): Validation Slices Describe Only What Is Unique About Each Slice

**Why ranked #2**

The current validation surface repeats the same module skeleton many times. This
story removes mechanical repetition without changing evidence semantics.

**User/Research value**

- Reviewers keep the same emitted bundles and slice names.
- Developers see fewer near-identical validators and fewer repetitive tests.

**Given / When / Then**

- Given the current correctness-evidence and performance-evidence validation
modules.
- When a new validation slice is added or an existing one changes.
- Then the implementation should define only the filter, summary logic, and
status rule unique to that slice.

**Scope**

- In: shared slice registry or shared bundle-builder scaffolding, parameterized
tests, stable artifact names and emitted fields.
- Out: deleting artifact families, changing the evidence floor, or removing CLI
entry points outright.

**Current complexity anchors**

- `benchmarks/density_matrix/correctness_evidence/*_validation.py`
- `benchmarks/density_matrix/performance_evidence/*_validation.py`
- `tests/partitioning/test_correctness_evidence.py`
- `tests/partitioning/test_performance_evidence.py`

**Acceptance signals**

- Validation modules declare only slice-specific behavior.
- Shared core-field checks and bundle-writing logic live in one framework layer.
- Repetitive test cases are parameterized over a slice registry or equivalent.
- Existing artifact filenames and top-level bundle shapes remain stable.

**Engineering tasks**

### Engineering Task 2.1: Introduce a slice specification model

**Change type**

- code

**Definition of done**

- A validator can be described by metadata plus a case selector and summary rule.
- Shared concerns no longer need to be restated in every slice module.

**Execution checklist**

- Define a slice spec dataclass or registry structure.
- Move shared bundle field validation into the framework layer.
- Keep bundle filenames and suite names explicit in the spec.

### Engineering Task 2.2: Convert repeated validators onto the shared scaffold

**Change type**

- code | tests

**Definition of done**

- At least the most repetitive correctness-evidence validators run through the
common scaffold.
- Output bundle shapes remain unchanged.

**Execution checklist**

- Migrate sequential, external, output-integrity, and similar slices first.
- Preserve current summary semantics.
- Keep thin CLI wrappers only where a stable entry point is still useful.

### Engineering Task 2.3: Parameterize repetitive evidence tests

**Change type**

- tests

**Definition of done**

- Tests assert slice-specific semantics without repeating the same core-field
structure checks in many hand-written functions.

**Execution checklist**

- Introduce a table-driven or parameterized test matrix for slice bundles.
- Keep targeted bespoke tests only where the slice behavior is genuinely
unique.

**Evidence produced**

- Fewer validation modules carrying identical scaffolding.
- Parameterized bundle tests that still preserve current artifact guarantees.

**Risks / rollback**

- Risk: over-generalizing the framework until slice-specific behavior becomes
harder to read.
- Rollback/mitigation: stop at a small registry plus shared bundle builder; do
not force every detail into one meta-language.

## Story 3 (COMPLETED): Partitioned Runtime Has One Contract-Checking Path And One Lowering Source

**Why ranked #3**

This story targets the most visible complexity in the runtime without yet
changing the planner or descriptor schemas. It simplifies behavior around
validation, lowering, and runtime-path reporting.

**User/Research value**

- Runtime behavior stays exact, but the implementation becomes easier to audit.
- Gate and noise semantics stop being repeated in multiple execution paths.

**Given / When / Then**

- Given the current baseline and fused runtime paths.
- When the runtime validates members, lowers them to circuits, or builds fused
kernels.
- Then there should be one clear contract path and one source of truth for
operation semantics.

**Scope**

- In: runtime member validation, circuit-shape validation, gate lowering,
runtime-path reporting, internal helper structure.
- Out: expanding fused support beyond the current contract.

**Current complexity anchors**

- `squander/partitioning/noisy_runtime.py`

**Acceptance signals**

- Support validation for members happens once at the runtime boundary, not again
during circuit append.
- `_validate_runtime_circuit_shape` and `_validate_runtime_member_sequence`
become one shared validator with a small policy parameter.
- Gate semantics are defined once and reused by both baseline lowering and fused
kernel construction.
- Runtime results expose a clear meaning for requested versus actual runtime
path.

**Engineering tasks**

### Engineering Task 3.1: Collapse repeated runtime validators

**Change type**

- code | tests

**Definition of done**

- One helper validates operation ordering, normalized names, parameter counts,
and parameter positions.
- Segment-level and whole-circuit validation differ only by an explicit policy.

**Execution checklist**

- Merge `_validate_runtime_circuit_shape` and
`_validate_runtime_member_sequence` into one shared validator.
- Keep error reporting stable enough for existing unsupported-path tests.
- Add focused tests for both local-param and segment-param modes.

### Engineering Task 3.2: Validate support once, then lower without re-checking

**Change type**

- code

**Definition of done**

- `_append_member_to_circuit` no longer repeats boundary validation that already
happened in `validate_runtime_request`.
- Internal helpers assume validated input unless explicitly documented otherwise.

**Execution checklist**

- Remove duplicate `_validate_supported_member` calls from internal lowering
paths.
- Keep one clear contract boundary for invalid member rejection.

### Engineering Task 3.3: Define one lowering description for supported gates

**Change type**

- code | tests

**Definition of done**

- U3 and CNOT semantics are defined in one reusable structure or helper family.
- Baseline circuit append and fused matrix building consume the same meaning.

**Execution checklist**

- Introduce a shared gate semantics table or helper layer.
- Reuse it in `_append_member_to_circuit`.
- Reuse it in `_build_gate_matrix_for_member`.
- Add regression coverage that baseline and fused handling stay aligned.

### Engineering Task 3.4: Clarify runtime-path reporting

**Change type**

- code | docs | tests

**Definition of done**

- The result schema clearly distinguishes requested execution strategy from
realized execution strategy, or explicitly documents one canonical field.

**Execution checklist**

- Decide whether to emit `requested_runtime_path` and `actual_runtime_path`,
or to keep one field with strict documented semantics.
- Update tests and artifact builders accordingly.

**Evidence produced**

- Smaller runtime helper surface.
- Clearer runtime-path semantics and fewer duplicate contract checks.

**Risks / rollback**

- Risk: runtime simplification changes unsupported-path diagnostics or fused-path
reporting in subtle ways.
- Rollback/mitigation: lock current runtime behavior with targeted tests before
helper consolidation.

## Story 4 (COMPLETED): Planner, Descriptor, And Runtime Surfaces Carry One Minimal Model Per Concept

**Why ranked #4**

This is the most important conceptual cleanup, but it has the highest schema
blast radius. It should come after Stories 1 through 3 have already reduced
local duplication and strengthened tests.

**User/Research value**

- The code stops representing the same operation and partition data in parallel
shapes.
- Validation logic becomes structural instead of field-by-field reconciliation of
copied data.

**Given / When / Then**

- Given the canonical planner surface, partition descriptor members, and runtime
partition records.
- When the system hands off one operation or partition concept between layers.
- Then each layer should add only its layer-specific data rather than copying the
full upstream object.

**Scope**

- In: `CanonicalNoisyPlannerOperation`,
`NoisyPartitionDescriptorMember`, `NoisyPartitionDescriptor`,
`NoisyRuntimePartitionRecord`, serialization helpers, validators.
- Out: changing supported workloads or removing auditability from emitted
payloads.

**Current complexity anchors**

- `squander/partitioning/noisy_planner.py`
- `squander/partitioning/noisy_runtime.py`

**Acceptance signals**

- Descriptor members no longer duplicate every canonical operation field just so
validators can compare them back to the source.
- Partition-level duplicated fields such as `canonical_operation_indices` are
derived or produced by serialization helpers rather than stored redundantly.
- Runtime partition records keep runtime-only data and reuse descriptor
serialization where possible.
- Validation checks layer-specific invariants rather than re-diffing copied
canonical state.

**Engineering tasks**

### Engineering Task 4.1: Split canonical operation data from descriptor-local data

**Change type**

- code | tests

**Definition of done**

- Descriptor members retain only local/remapped/runtime-relevant fields plus a
reference to the canonical operation identity.
- Serialization still emits the fields needed by existing artifact consumers,
either directly or through an adapter.

**Execution checklist**

- Define the minimal descriptor-local field set.
- Replace copied canonical fields with canonical references where feasible.
- Keep compatibility serialization until downstream consumers are updated.

### Engineering Task 4.2: Derive partition-level duplicate fields

**Change type**

- code

**Definition of done**

- Fields like `canonical_operation_indices` and `partition_qubit_span` are not
independently stored if they can be derived from members or remap data.

**Execution checklist**

- Convert duplicated partition fields into computed properties or serializer
outputs.
- Remove redundant validation that exists only to keep duplicated fields in
sync.

### Engineering Task 4.3: Slim runtime partition records to runtime-specific data

**Change type**

- code | tests

**Definition of done**

- Runtime partition records stop mirroring the full descriptor schema.
- Shared serialization logic exists where emitted payloads still need the same
view.

**Execution checklist**

- Separate descriptor-derived data from runtime-only measurements.
- Reuse a shared serializer for common partition fields if artifacts still
need them.
- Update runtime audit and artifact builders accordingly.

### Engineering Task 4.4: Replace copied-field reconciliation with structural validation

**Change type**

- code | tests

**Definition of done**

- Validation logic checks structural integrity, local remap correctness, and
operation ordering without re-comparing every copied canonical field.

**Execution checklist**

- Remove field-by-field canonical re-diff loops that exist only because data
is copied across layers.
- Keep the invariants that actually protect correctness.

**Evidence produced**

- Fewer parallel dataclasses carrying the same information.
- Simpler planner-to-descriptor-to-runtime handoff logic.

**Risks / rollback**

- Risk: schema changes ripple into artifact bundles and tests.
- Rollback/mitigation: introduce compatibility serializers first, migrate
consumers, then remove redundant storage.

## Story 5: Partitioning Tests Validate API Behavior Directly

**Why ranked #5**

This story supports all deeper refactors by making tests easier to understand
and less coupled to the benchmark package layout.

**User/Research value**

- Tests become a clearer specification of planner/runtime behavior.
- Failures point to the API layer first, not to benchmark support plumbing.

**Given / When / Then**

- Given the current partitioning tests.
- When a developer reads or updates them.
- Then the tests should reveal planner/runtime behavior directly instead of
requiring navigation through benchmark-evidence helpers for core cases.

**Scope**

- In: test helpers, shared fixtures, direct API tests, clearer separation between
API tests and artifact-package tests.
- Out: deleting artifact validation tests that still matter for publication
evidence.

**Current complexity anchors**

- `tests/partitioning/test_partitioned_runtime.py`
- `tests/partitioning/test_partitioned_runtime_fusion.py`
- `tests/partitioning/test_planner_surface_entry.py`
- `tests/partitioning/test_planner_surface_descriptors.py`
- `tests/partitioning/test_correctness_evidence.py`
- `tests/partitioning/test_performance_evidence.py`

**Acceptance signals**

- Core planner/runtime tests import primarily from `squander.partitioning` and
test-local helpers.
- Benchmark/evidence imports remain only in dedicated artifact integration tests.
- Repeated bundle-core-field checks are parameterized rather than copied.

**Engineering tasks**

### Engineering Task 5.1: Introduce test-local partitioning fixtures and builders

**Change type**

- tests

**Definition of done**

- Core tests no longer rely on benchmark modules for simple workload setup.

**Execution checklist**

- Add test-local helpers for representative descriptor sets and parameters.
- Move benchmark-only workload builders behind integration-style tests.

### Engineering Task 5.2: Split API tests from evidence-package tests

**Change type**

- tests

**Definition of done**

- Files testing planner/runtime behavior are distinct from files testing emitted
bundle behavior.

**Execution checklist**

- Keep planner/runtime invariants in API-focused tests.
- Keep artifact shape and evidence semantics in dedicated bundle tests.

### Engineering Task 5.3: Parameterize repetitive bundle assertions

**Change type**

- tests

**Definition of done**

- Core-field stability assertions are table-driven where appropriate.

**Execution checklist**

- Introduce a shared assertion helper or parameterized bundle matrix.
- Keep only slice-specific assertions as bespoke tests.

**Evidence produced**

- Smaller, clearer test graph for partitioning code.
- Better protection for Stories 3 and 4 refactors.

**Risks / rollback**

- Risk: moving too much logic into test helpers can hide what each test proves.
- Rollback/mitigation: keep helpers small and focused on setup, not assertions.

## Story 6: Public Names And Payloads Expose One Concept One Way

**Why ranked #6**

This is cleanup with real value, but it should happen after structural changes
stabilize so the team does not rename things twice.

**User/Research value**

- Readers and consumers learn fewer names for the same concept.
- Public payloads become easier to interpret.

**Given / When / Then**

- Given the current planner/runtime helper names and serialized payloads.
- When a user or maintainer reads the API or emitted JSON.
- Then each concept should appear once with clear semantics and minimal aliasing.

**Scope**

- In: thin aliases, ambiguous property names, duplicate summary fields,
inconsistent string literals for operation kinds.
- Out: breaking external consumers without a compatibility path.

**Current complexity anchors**

- `squander/partitioning/noisy_planner.py`
- `squander/partitioning/noisy_runtime.py`
- `benchmarks/density_matrix/performance_evidence/common.py`

**Acceptance signals**

- Thin wrappers that add no behavior are removed or clearly deprecated.
- Ambiguous names such as `partition_qubit_span` are either renamed or explicitly
documented as compatibility surfaces.
- Runtime payloads do not duplicate top-level facts inside `summary` unless
required for compatibility.
- Runtime code uses shared planner kind constants instead of repeating raw
string literals where possible.

**Engineering tasks**

### Engineering Task 6.1: Audit and classify aliases

**Change type**

- code | docs

**Definition of done**

- Each alias is classified as keep, deprecate, or remove.

**Execution checklist**

- Review thin aliases in planner/runtime and evidence common modules.
- Keep only aliases that protect a real compatibility boundary.

### Engineering Task 6.2: Clarify ambiguous property and payload names

**Change type**

- code | docs | tests

**Definition of done**

- Public names match actual semantics, or compatibility names are documented as
legacy views.

**Execution checklist**

- Resolve `partition_qubit_span` naming ambiguity.
- Review summary duplication in runtime result payloads.
- Standardize use of planner operation kind constants in runtime code.

**Evidence produced**

- Smaller public helper surface.
- Clearer serialized payload semantics.

**Risks / rollback**

- Risk: small renames create noisy downstream churn.
- Rollback/mitigation: prefer compatibility adapters and staged removal.

## Recommended Execution Waves

### Wave 1: Low-risk deduplication

- Story 1
- Story 2

Goal: remove obvious duplication in evidence code and tests without touching the
planner/runtime contract.

### Wave 2: Runtime simplification with stronger tests

- Story 5 (foundational test cleanup as needed)
- Story 3

Goal: make runtime refactors safer, then consolidate repeated validation and
lowering logic.

### Wave 3: Structural schema simplification

- Story 4
- Story 6

Goal: reduce parallel models and then clean the remaining naming and
compatibility surface once the new structure is settled.

## Definition Of Done For The Full Simplification Pass

The simplification effort is complete when all of the following are true:

- the evidence layer has one shared supported-case rule and one shared common
runtime field builder,
- validation slices no longer repeat full module scaffolding for every bundle,
- runtime validation and lowering logic has one clear contract path,
- planner, descriptor, and runtime layers no longer store copied upstream data
unless a compatibility serializer requires it,
- partitioning tests read as API behavior tests first and benchmark artifact
tests second,
- and remaining aliases, duplicated payload fields, and ambiguous names are
either removed or intentionally documented as compatibility shims.

## Notes For Implementation

- Preserve emitted artifact filenames and top-level shapes until consumer
migration is complete.
- Favor compatibility serializers over simultaneous schema and behavior changes.
- Land each story behind passing tests and unchanged supported-path behavior.
- If a story uncovers a need to change the scientific claim or supported surface,
stop and record that as a separate contract decision rather than treating it as
simplification.

