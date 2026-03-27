# Task 2: Partition Descriptor And Semantic-Preservation Contract

**Implementation Status: COMPLETE**

This mini-spec defined the Phase 3 Task 2 implementation contract. The partition
descriptor and semantic-preservation contract is now implemented in
`squander/partitioning/noisy_planner.py` as `NoisyPartitionDescriptorSet`,
`NoisyPartitionDescriptor`, and `NoisyPartitionDescriptorMember` dataclasses.

This document inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-003`, `P3-ADR-004`, `P3-ADR-007`, `P3-ADR-008`, and `P3-ADR-009`,
plus the closed semantic-preservation, support-matrix, validation-baseline,
and benchmark-anchor items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen
planner-surface scope, runtime-minimum, cost-model sequencing, or broader
Phase 4 workflow-growth decisions.

## Given / When / Then
- Given a supported Phase 3 canonical noisy planner surface and a request that
  claims `partitioned_density` behavior inside the frozen support matrix.
- When the planner emits partition descriptors for execution, validation, or
  benchmarking.
- Then each descriptor preserves auditable exact noisy semantics by retaining
  stable references to canonical operation order, explicit noise placement,
  qubit support, parameter-routing metadata, and any partition-local remapping
  needed for faithful execution, and unsupported transformations fail
  explicitly before they can produce a misleading runtime result.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner surface as an ordered noisy
  mixed-state operation sequence equivalent to `NoisyCircuit` content built
  from `GateOperation` and `NoiseOperation`.
- Task 2 defines the minimum descriptor contract between that canonical surface
  and later runtime, validation, and benchmark consumers. It does not by
  itself deliver end-to-end partitioned execution, fused kernels, or
  density-aware heuristic calibration.
- Task 3 depends on Task 2 by consuming partition descriptors whose semantics
  are explicit enough to execute without hidden fallback, hidden reindexing, or
  source-specific assumptions.
- Task 4 depends on Task 2 by using descriptor boundaries that make any fused
  substructure auditable against the surrounding exact gate/noise order.
- Task 6 depends on Task 2 by turning descriptor-level semantic obligations
  into correctness checks against the sequential `NoisyCircuit` baseline and
  the required external micro-validation package.
- The sequential `NoisyCircuit` executor remains the internal exact semantic
  oracle that partition descriptors must preserve.
- The required continuity workload is the frozen Phase 2 noisy XXZ `HEA`
  workflow. The required methods workloads remain the 2 to 4 qubit
  micro-validation cases plus the structured noisy `U3` / `CNOT` families.
- The frozen mandatory gate surface is `U3` and `CNOT`; the frozen mandatory
  noise surface is local single-qubit depolarizing, local amplitude damping,
  and local phase damping or dephasing.
- Existing state-vector partition outputs such as partition membership,
  partition-local qubit grouping, or parameter-order tuples may be reused as
  scaffolding, but Task 2 must make noisy mixed-state semantics explicit rather
  than inheriting state-vector-only assumptions.
- Task 1 implementation already froze a concrete planner-entry provenance
  vocabulary and schema-versioned audit surface. Task 2 should extend that
  concrete audit vocabulary into descriptor metadata rather than replacing it
  with a looser descriptor-local naming scheme.
- Task 1 implementation also established that explicit hard failure is not
  sufficient on its own for negative evidence. Unsupported descriptor-generation
  outcomes should use one stable machine-reviewable vocabulary for failure
  category, first unsupported condition, and failure stage so later validation
  and publication artifacts remain comparable.
- Partition-local indexing, auxiliary DAG nodes, or runtime-only schedule
  details may exist internally, but the contract-defining descriptor surface
  must remain auditable back to the canonical operation indices and global
  qubit labels.
- Reordering across noise boundaries is not assumed. If an exact equivalence
  rule is not separately documented and validated, the descriptor contract must
  preserve the original order.
- This task does not widen the frozen support matrix to correlated multi-qubit
  noise, readout or shot-noise workflows, calibration-aware noise, approximate
  scaling methods, or broader noisy VQE/VQA feature growth.

## Required behavior
- Task 2 freezes a minimum partition descriptor contract as the semantic
  boundary between the canonical noisy planner surface and all later
  `partitioned_density` runtime or validation behavior.
- A supported partition descriptor set must identify stable partition ordering
  and stable references to the canonical noisy operation sequence so that the
  relationship between the planner input and the partitioned execution plan is
  auditable.
- Each partition descriptor must retain exact within-partition operation order
  for both gates and noise operations. Partition membership alone is
  insufficient.
- Noise operations remain first-class partition contents. They must not be
  reduced to boundary markers, out-of-band annotations, or undocumented
  side-channel metadata.
- Each supported descriptor must preserve operation-level qubit support and any
  partition-level qubit-span information needed to execute and benchmark the
  partitioned case faithfully.
- If partition-local execution uses a different qubit indexing scheme than the
  canonical surface, the descriptor must record invertible remapping metadata
  sufficient to reconstruct the original global semantics exactly.
- Each supported descriptor must preserve parameter-routing metadata sufficient
  to map the global parameter vector into partition-local execution order
  without ambiguity. An equivalent mapping contract is acceptable, but implicit
  or implementation-only parameter assumptions are not.
- Descriptor generation must preserve the exact relative position of noise with
  respect to neighboring gates unless an exact equivalence rule is separately
  documented, validated, and explicitly included in the supported contract.
- The descriptor contract must be rich enough that a later runtime or audit
  tool can reconstruct or verify the full ordered noisy execution intent from
  the canonical planner surface plus the descriptor set without consulting
  hidden planner state.
- Descriptor validation must reject unsupported or lossy transformations before
  runtime execution begins, including dropped operations, ambiguous parameter
  routing, incomplete remapping, or reordering across unsupported noise
  boundaries.
- Unsupported descriptor-generation failures must expose one stable structured
  diagnostic vocabulary rather than generic exception text alone. At minimum,
  the emitted failure record must make the unsupported category, first
  unsupported condition, and failure stage machine-reviewable, and it should
  remain alignable with the Task 1 planner-entry unsupported vocabulary unless a
  documented Task 2-specific refinement is required.
- Supported descriptor artifacts must expose enough provenance to remain
  reproducible in tests and benchmark outputs. At minimum, the case-level
  provenance must retain a stable tuple equivalent to descriptor schema version,
  requested mode, source type, entry route, workload family, and workload ID so
  supported descriptor outputs stay comparable across continuity, microcase,
  structured-family, and optional exact legacy-lowering paths.
- Supported descriptor artifacts should reuse the Task 1 planner-entry
  provenance vocabulary where possible and add descriptor-specific metadata on
  top of it, rather than redefining provenance separately at the descriptor
  layer.
- Task 2 must produce one stable descriptor-audit surface, such as a
  machine-reviewable artifact bundle or rerunnable checker, that later runtime,
  validation, and benchmark tasks can cite directly instead of relying on ad
  hoc debug output.
- Task 2 completion means descriptor semantics, semantic-preservation rules, and
  negative boundaries are explicit and testable. It does not by itself require
  the full partitioned runtime, real fused execution, or performance claims to
  be complete.

## Unsupported behavior
- Using partition membership alone as the claimed semantic contract for noisy
  mixed-state execution.
- Treating noise placement as partition-boundary-only metadata, opaque planner
  annotations, or another non-first-class representation outside the descriptor
  set.
- Emitting descriptors that require hidden planner state to recover operation
  order, qubit ownership, parameter routing, or remapping details.
- Allowing ambiguous, partial, or lossy parameter-routing behavior, including
  cases where partition-local parameter order cannot be traced back to the
  canonical global parameter order.
- Reordering, merging, or splitting operations across noise boundaries without a
  separately documented exact rule that is part of the supported contract.
- Silently omitting gate operations, silently omitting noise operations, or
  silently inserting compatibility transformations during descriptor creation.
- Warning-only or best-effort handling for unsupported descriptor generation
  requests when the request claims supported `partitioned_density` behavior.
- Claiming that Task 2 is complete because the planner can emit descriptors even
  if those descriptors do not support later exactness auditing against the
  sequential density baseline.
- Using Task 2 to imply that Task 3 runtime delivery, Task 4 fused execution,
  or Task 5 density-aware heuristic calibration is already complete.
- Expanding the required support surface beyond the frozen `U3` / `CNOT` plus
  local-noise baseline as part of Task 2 closure.

## Acceptance evidence
- Descriptor-spec or integration evidence shows that the frozen Phase 2 noisy
  XXZ `HEA` continuity workload, required 2 to 4 qubit microcases, and at least
  one required structured noisy `U3` / `CNOT` family instance all produce
  partition descriptors inside one auditable contract surface.
- Positive descriptor-audit tests show that supported descriptor sets retain
  canonical operation references, exact within-partition gate/noise order,
  operation-level qubit support, parameter-routing metadata, and remapping
  metadata when partition-local indexing differs from the canonical surface.
- Round-trip or audit evidence shows that later consumers can reconstruct or
  verify the intended ordered noisy execution from the canonical planner surface
  plus descriptor metadata without consulting hidden planner state.
- Boundary-focused tests show that descriptor generation keeps noise placement
  explicit on cases where partitions begin or end near noise operations and on
  cases with multi-parameter `U3` gates that stress parameter-routing logic.
- Negative tests show that unsupported descriptor requests fail explicitly when
  metadata would be lossy or ambiguous, including dropped noise operations,
  unsupported reorder attempts, incomplete remapping, or ambiguous parameter
  routing, and that the emitted failure evidence records one stable unsupported
  category, first unsupported condition, and failure stage vocabulary rather
  than only free-form error text.
- Reproducibility artifacts for mandatory cases record the stable case-level
  provenance tuple, including descriptor schema version, requested mode, source
  type, entry route, workload family, and workload ID, plus partition count,
  partition qubit span, and a descriptor-audit summary that makes semantic
  preservation claims inspectable.
- One stable descriptor-audit artifact bundle or rerunnable checker exists for
  supported mandatory cases and records descriptor provenance, ordered
  membership or operation references, partition-span summaries, and
  descriptor-level semantic-preservation metadata in a machine-reviewable form.
- Traceability target: satisfy the Phase 3 Task 2 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring
  partition descriptors to preserve exact gate/noise order and parameter
  routing strongly enough for the frozen Section 10.1 thresholds to be
  validated later by Task 3 and Task 6.
- Traceability target: satisfy the canonical planner-surface dependency in
  `P3-ADR-003`, the semantic-preservation decision in `P3-ADR-004`, the frozen
  support-matrix boundary in `P3-ADR-007`, the validation-baseline rule in
  `P3-ADR-008`, and the continuity-plus-structured-workload anchor in
  `P3-ADR-009`.

## Affected interfaces
- `CanonicalNoisyPlannerSurface` and `CanonicalNoisyPlannerOperation` or the
  equivalent canonical noisy planner-input boundary produced by Task 1.
- Any planner output type, partition-descriptor schema, or serialization surface
  that claims `partitioned_density` semantics for noisy mixed-state workloads.
- Existing partitioning metadata such as partition membership, parameter-order
  translation, or partition-local qubit grouping when those surfaces are reused
  or generalized for Phase 3.
- Any planner-to-runtime adapter that converts descriptor sets into
  partition-local execution requests for the later partitioned density runtime.
- Pre-execution validation and error-reporting paths that reject unsupported or
  lossy descriptor-generation requests before runtime execution begins.
- Validation, benchmark, and reproducibility metadata surfaces that must record
  descriptor identity, partition spans, parameter-routing audit information, and
  semantic-preservation evidence.
- Change classification: additive for supported noisy partitioned workloads, but
  stricter for ambiguous descriptor-generation requests, which become explicit
  hard failures rather than undocumented behavior.

## Publication relevance
- Supports Paper 2's core methods claim that Phase 3 partitioning preserves
  exact noisy mixed-state semantics rather than only partition membership or
  planner structure.
- Makes correctness evidence scientifically defensible by forcing gate/noise
  order, parameter routing, and remapping semantics to be explicit before
  runtime and benchmark claims are made.
- Provides the contract needed for the Phase 3 abstract, short paper, and full
  paper sections that discuss partition descriptors, semantic preservation, and
  the later executable runtime.
- Prevents the paper narrative from overstating aggressive transformations or
  source-surface generality by clearly separating the guaranteed descriptor
  semantics from deferred follow-on work.
