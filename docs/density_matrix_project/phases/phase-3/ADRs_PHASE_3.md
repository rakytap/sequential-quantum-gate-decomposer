# Phase 3 ADRs

**Implementation Status: COMPLETE**

This document records the detailed architecture and scope decisions that apply
specifically to Phase 3. All decisions documented here have been implemented in
the current codebase (`noisy_planner.py`, `noisy_runtime.py`, and the evidence
pipelines under `benchmarks/density_matrix/`).

It refines, but does not override, the upstream planning decisions in:

- `docs/density_matrix_project/planning/ADRs.md`
- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md`

If this document appears to conflict with the upstream planning set, the conflict
must be resolved in favor of the upstream planning set unless this Phase 3
document explicitly states that it is tightening the scope for this phase only.

## Status Legend

- `Accepted`: part of the Phase 3 contract.
- `Deferred`: deliberately postponed beyond the core Phase 3 deliverable.
- `Rejected for Phase 3`: explicitly excluded from this phase.

## Phase 3 Decision Summary

| ADR | Title | Status |
|---|---|---|
| P3-ADR-001 | Use spec-driven Phase 3 documentation as the implementation contract | Accepted |
| P3-ADR-002 | Keep Phase 3 focused on noise-aware partitioning and fusion, not broader Phase 4 workflow growth | Accepted |
| P3-ADR-003 | Use a canonical noisy mixed-state planner surface | Accepted |
| P3-ADR-004 | Treat exact gate/noise-order preservation as a first-class contract | Accepted |
| P3-ADR-005 | Require an executable partitioned runtime plus at least one real fused execution mode | Accepted |
| P3-ADR-006 | Sequence structural heuristics before benchmark-calibrated density cost modeling | Accepted |
| P3-ADR-007 | Freeze a workload-driven Phase 3 support matrix centered on the Phase 2 gate/noise surface | Accepted |
| P3-ADR-008 | Use the sequential density path and Qiskit Aer as required validation baselines | Accepted |
| P3-ADR-009 | Anchor Paper 2 on the Phase 2 continuity workflow plus structured noisy partitioning families | Accepted |
| P3-ADR-010 | Defer channel-native fused noisy blocks, broader Phase 4 workflow growth, and approximate scaling beyond core Phase 3 | Deferred |

## P3-ADR-001: Use Spec-Driven Phase 3 Documentation As The Implementation Contract

### Status

`Accepted`

### Context

Phase 3 sits at a methods-heavy transition point:

- Phase 2 already froze an exact noisy workflow contract,
- the density module already provides a sequential `NoisyCircuit` reference,
- and the existing partitioning stack already contains substantial state-vector
  planning and fusion logic.

Without a documentation-first contract, implementation could drift toward:

- planner-only architecture without a real runtime result,
- performance claims detached from exact noisy semantics,
- or publication claims that exceed the delivered workload surface.

### Decision

Phase 3 work must be guided by explicit documentation-first contracts covering:

- scope,
- non-goals,
- deliverable goals,
- semantic-preservation rules,
- benchmark evidence,
- and Paper 2 claim boundaries.

The Phase 3 document set in `docs/density_matrix_project/phases/phase-3/` is the
working contract for the phase.

### Rationale

- Phase 3 is the first density-matrix phase that must produce a standalone
  methods result rather than only an integration result.
- The planner/runtime/fusion split creates enough architectural freedom that
  undocumented intent would lead to rework.
- Paper 2 needs a defensible claim boundary before implementation starts.

### Consequences

- Every major Phase 3 deliverable must map back to a documented acceptance
  criterion.
- Unsupported and deferred mixed-state partitioning cases become explicit
  outcomes, not accidental omissions.
- Publication artifacts can be prepared in parallel with implementation because
  the required claim surface is fixed in advance.

### Rejected Alternatives

- Rely on implementation order or exploratory notes alone.
- Treat the Phase 3 paper as a downstream packaging task after code exists.

### Upstream Alignment

Refines:

- `ADR-001`
- the accepted spec-driven phase workflow established in Phase 2

## P3-ADR-002: Keep Phase 3 Focused On Noise-Aware Partitioning And Fusion, Not Broader Phase 4 Workflow Growth

### Status

`Accepted`

### Context

After Phase 2, the next tempting step would be to keep expanding the noisy
workflow surface immediately:

- broader circuit sources,
- gradients,
- richer optimizer paths,
- and more VQE/VQA-facing controls.

At the same time, the project still lacks a clearly delivered Phase 3 backend
result inside partitioning and fusion.

### Decision

The core Phase 3 result is noise-aware partitioning and limited real fusion for
exact mixed-state circuits.

Phase 3 will:

- freeze the Phase 2 canonical workflow as the continuity anchor,
- use that workflow in benchmarks without broadening the user-facing workflow
  surface,
- and defer broader noisy VQE/VQA growth, gradient routing, and optimizer
  studies to Phase 4 or later.

### Rationale

- It gives the project a clean methods milestone between Phase 2 integration and
  Phase 4 workflow science.
- It keeps Paper 2 centered on a backend contribution rather than mixing backend
  work with new application-surface promises.
- It matches the accepted dependency order in the upstream planning set.

### Consequences

- Phase 3 benchmarks may still be training-relevant, but they should not depend
  on new Phase 4 surface area.
- The frozen Phase 2 workflow remains a required continuity check, not a pretext
  for broader Phase 4 implementation.
- If new workflow-surface requests arise, they should be treated as future scope
  decisions rather than hidden Phase 3 blockers.

### Rejected Alternatives

- Continue broadening VQE/VQA features immediately after Phase 2.
- Merge Phase 3 backend work and Phase 4 workflow growth into one combined
  milestone.

### Upstream Alignment

Refines:

- `ADR-001`
- `PLANNING.md` Phase 3 and Phase 4 separation
- `RESEARCH_ALIGNMENT.md` Phase mapping

## P3-ADR-003: Use A Canonical Noisy Mixed-State Planner Surface

### Status

`Accepted`

### Context

The current codebase has two relevant but separate execution viewpoints:

- the state-vector partitioning stack centered on `qgd_Circuit` /
  `Gates_block`,
- and the exact noisy mixed-state backend centered on `NoisyCircuit`,
  `GateOperation`, and `NoiseOperation`.

The simplest short-term port would leave the planner tied to the state-vector
surface and treat noise as external annotations or barriers. That would reuse
code, but it would not make noisy mixed-state circuits first-class objects.

### Decision

Phase 3 uses a canonical noisy mixed-state planner surface defined by an ordered
operation stream equivalent to `NoisyCircuit` operations.

Additional contract details:

- every mandatory Phase 3 workload must be representable in that canonical
  surface before partition planning begins,
- exact lowering from existing sources such as Phase 2 continuity workloads,
  `qgd_Circuit`, or `Gates_block` is allowed and desirable,
- but the phase claim is judged on the canonical noisy surface rather than on
  full direct parity for every source representation,
- and unsupported `partitioned_density` requests fail explicitly before
  execution rather than silently falling back.

### Rationale

- It makes the density semantics visible where the Phase 3 methods claim lives.
- It keeps the planner aligned with the exact backend rather than with a
  state-vector-first approximation of it.
- It protects the paper narrative from ambiguity about what was actually
  partitioned.

### Consequences

- Phase 3 can reuse state-vector partitioning machinery structurally without
  inheriting its semantics blindly.
- Exact lowering quality becomes part of the support surface and must be
  documented.
- Full direct circuit-source parity is deferred instead of being implied by the
  planner contract.

### Rejected Alternatives

- Keep the planner phase-defining input tied only to `qgd_Circuit`.
- Treat noise channels as opaque barriers external to the partition model.
- Silently substitute the sequential density path when partitioned execution
  cannot represent a case.

### Upstream Alignment

Refines:

- `ADR-002`
- `ARCHITECTURE.md` Phase 3 extension points

## P3-ADR-004: Treat Exact Gate/Noise-Order Preservation As A First-Class Contract

### Status

`Accepted`

### Context

Phase 3 aims to accelerate exact noisy simulation, but in the noisy mixed-state
setting semantics are especially sensitive to:

- gate/noise ordering,
- channel placement,
- parameter routing,
- and qubit remapping.

Barrier-only or loosely specified partition descriptors would weaken both
correctness and publication claims.

### Decision

Phase 3 treats semantic preservation as a first-class contract item.

Partition descriptors must retain:

- exact gate/noise order,
- qubit support,
- parameter-order mapping or equivalent metadata,
- and any remapping information needed to reconstruct the sequential semantics.

Reordering across noise boundaries is not part of the required Phase 3 contract
unless a rule is separately documented as exact and validated.

### Rationale

- Exact noisy semantics are the main scientific value of the density backend.
- Paper 2 is much stronger if every speedup claim is grounded in a strict
  reference contract.
- Later follow-on branches remain possible without retroactively weakening the
  Phase 3 semantics claim.

### Consequences

- Some aggressive optimization opportunities will remain deferred.
- Validation against the sequential density baseline becomes mandatory, not
  optional.
- Unsupported transformations must be surfaced explicitly instead of hidden
  behind partial behavior.

### Rejected Alternatives

- Let implementation details define which ordering information is retained.
- Treat noise as partition-boundary-only metadata.
- Permit undocumented or partially validated gate/noise reordering as a default
  tactic.

### Upstream Alignment

Refines:

- `ADR-002`
- `ADR-005`
- `PLANNING.md` semantic-faithfulness wording for Phase 3

## P3-ADR-005: Require An Executable Partitioned Runtime Plus At Least One Real Fused Execution Mode

### Status

`Accepted`

### Context

It is possible to make real progress on representation and planning without
delivering a runtime result. That would still leave the Phase 3 claim too weak
for a methods paper.

At the same time, requiring fully channel-native fused noisy blocks immediately
would push the phase into a far more invasive architecture than the accepted
critical path requires.

The planner surface and partition-descriptor contract have now also made the
handoff boundary more concrete than it was at initial planning time:

- the planner input is now a real schema-versioned canonical noisy mixed-state
  surface,
- and the semantic-preservation layer is now a real schema-versioned partition
  descriptor contract with explicit order, noise-placement, remapping, and
  parameter-routing metadata.

### Decision

Phase 3 minimum closure requires:

- an executable partitioned density runtime on the mandatory workload matrix,
- and at least one real fused execution mode on eligible substructures inside
  that runtime.

Additional contract details:

- the executable runtime must consume the validated schema-versioned partition
  descriptor contract rather than a second private runtime-only description of
  the same partitioned case,
- eligible fused substructures must be defined against that supported
  descriptor-level contract rather than against opaque planner internals,
- and unsupported or deferred fusion candidates must remain explicit in the
  benchmark and publication evidence instead of silently entering a fused path.

This fused mode:

- may remain unitary-island-based internally,
- must preserve surrounding noisy semantics exactly,
- and must be benchmarked on representative Phase 3 workloads.

Fully channel-native fused noisy blocks are deferred beyond the minimum Phase 3
result.

### Rationale

- Paper 2 should claim more than planner representation.
- The project needs a real backend result before deciding whether more invasive
  channel-native fusion is justified.
- This preserves a strong methods story without overcommitting the architecture.

### Consequences

- Planner-only closure is explicitly disallowed.
- Benchmark harnesses must exercise the real fused path instead of only the
  partitioned schedule.
- Runtime implementation now has an explicit contract surface to consume, which
  reduces ambiguity but also makes private runtime-only reinterpretations of the
  descriptor semantics out of contract.
- Fusion eligibility and fusion deferral both become auditable evidence
  categories rather than informal implementation notes.
- Channel-native fusion remains an explicit future branch instead of an implicit
  unfinished requirement.

### Implementation Note (Fused Runtime)

The current fused-runtime implementation realizes this decision conservatively
through descriptor-local unitary-island fusion on 1- and 2-qubit spans using the
density backend's local-unitary primitive.

It extends the shared partitioned density runtime surface additively rather than
replacing it:

- fused execution is labeled through an explicit fused runtime path,
- fused, supported-but-unfused, and deferred regions are recorded as auditable
  evidence categories,
- and the runtime schema version remains shared with the partitioned-runtime
  baseline rather than splitting into a second private fusion-only schema.

Representative 8- and 10-qubit layered nearest-neighbor sparse workloads now
exercise the real fused path and preserve exact noisy semantics, but the current
benchmark result closes the performance rule through the diagnosis branch rather
than the positive-threshold branch because runtime remains slower and peak
memory does not improve on those representative cases.

### Rejected Alternatives

- Stop at planner/runtime representation without real fused execution.
- Require full channel-native fused noisy blocks as the minimum Phase 3 bar.
- Hide sequential fallback inside a benchmark that claims partitioned execution.

### Upstream Alignment

Refines:

- `ADR-002`
- `ADR-007`
- `PUBLICATIONS.md` Paper 2 minimum publishable scope

## P3-ADR-006: Sequence Structural Heuristics Before Benchmark-Calibrated Density Cost Modeling

### Status

`Accepted`

### Context

The existing partitioning cost model in `squander/partitioning/tools.py` is
state-vector-specific and noise-blind. It reflects:

- state-vector arithmetic assumptions,
- target/control simplifications,
- and no explicit understanding of mixed-state noise placement.

Using it unchanged for Phase 3 would weaken the density-aware claim. However,
trying to jump directly to a fully calibrated density objective before a working
runtime exists would also be risky.

The planner surface and partition-descriptor contract have now frozen the
benchmark side of this decision more concretely than the original planning text:

- the mandatory workload inventory now exists as deterministic continuity,
  microcase, and structured-family evidence surfaces,
- and the implemented workload matrix already carries stable workload IDs,
  seed rules, noise-pattern labels, and reusable machine-reviewable bundles.

### Decision

Phase 3 follows a correctness-first sequencing rule:

1. deliver native noisy-circuit correctness and an executable partitioned
   runtime,
2. implement structural noise-aware planning heuristics,
3. calibrate a density-aware objective or heuristic on the required benchmark
   matrix,
4. and only then make stronger cost-model claims.

Optional kernel tuning is allowed only when profiling shows that it materially
affects the benchmark outcome.

Additional contract details:

- calibration should be anchored to the frozen mandatory workload inventory and
  its stable provenance fields rather than to ad hoc exploratory cases,
- and the calibration package should emit machine-reviewable bundles or
  rerunnable checkers so later benchmark and paper work can consume the result
  directly.
- The first delivered benchmark-facing calibration surface may be narrower than
  the full longer-horizon planner design space. In the current planner-calibration
  result,
  the supported calibration surface is the existing noisy planner with
  auditable `max_partition_qubits` span-budget settings, while broader adapted
  `kahn` / `tdag` / `gtqcp` / `ilp` / `ilp-fusion-ca` families remain
  comparison or design-space references until they are separately implemented
  on the noisy planner path.
- If the benchmark-grounded selection result remains close or rerun-sensitive
  inside that bounded candidate family, the supported claim should be phrased
  against the auditable selection rule and explicit claim boundary rather than
  against one permanently frozen winner identity.

### Rationale

- Correctness-first is the safest scientific strategy.
- Benchmark calibration is more trustworthy than premature analytic optimality
  claims.
- It keeps Phase 3 centered on the backend result, not only on model elegance.

### Consequences

- Phase 3 should avoid "optimal partitioning" language until the calibrated
  model exists.
- The first working runtime may use simpler structural heuristics than the final
  benchmark-facing planner.
- Core heuristic and calibration claims should now be phrased against the
  already-frozen workload IDs, seed rules, and noise-pattern vocabulary instead
  of against a moving benchmark target.
- The first delivered planner-calibration claim may therefore be a benchmark-calibrated rule
  over a bounded noisy-planner candidate family rather than a broad
  algorithm-family closure.
- A rerun-sensitive winner inside that bounded candidate family is still a valid
  implementation finding, provided the selection rule, comparison baselines,
  and claim boundary remain explicit.
- Kernel work becomes evidence-driven support work rather than a separate phase
  definition.

### Rejected Alternatives

- Port the state-vector FLOP model directly and call the result density-aware.
- Delay runtime implementation until a polished density cost model exists.
- Treat AVX-level tuning as the main Phase 3 scientific result.

### Upstream Alignment

Refines:

- `ADR-003`
- `PLANNING.md` benchmark-calibrated heuristic wording

## P3-ADR-007: Freeze A Workload-Driven Phase 3 Support Matrix Centered On The Phase 2 Gate/Noise Surface

### Status

`Accepted`

### Context

Phase 3 can easily drift into a gate-coverage project if the support surface is
left open. The density backend already exposes more gates than the strict Phase 2
minimum, but full parity is not required to make a publishable methods claim.

### Decision

Phase 3 freezes a workload-driven support matrix:

- required gate surface for the minimum claim: `U3` and `CNOT`,
- required noise surface: local single-qubit depolarizing, local amplitude
  damping, and local phase damping or dephasing,
- required input contract: canonical noisy mixed-state operation stream plus
  exact lowering for the Phase 2 continuity anchor,
- optional extensions: additional gates already exposed by `NoisyCircuit` when a
  mandatory microcase or clearly secondary benchmark genuinely needs them,
- deferred items: full `qgd_Circuit` parity, multi-controlled gates as a minimum
  requirement, correlated multi-qubit noise, calibration-aware noise, and
  readout/shot-noise workflow features.

### Rationale

- The methods claim is about native noisy partitioning/fusion, not about total
  gate-surface parity.
- Keeping the minimum support surface narrow makes the benchmark matrix more
  reproducible and the phase more achievable.
- It preserves room for later gate expansion without weakening the Phase 3
  closure story.

### Consequences

- Mandatory Phase 3 benchmarks must remain compatible with the frozen support
  surface.
- Broader gates become documented optional extensions instead of hidden
  assumptions.
- Requests beyond the frozen surface become explicit scope decisions.

### Rejected Alternatives

- Require immediate full gate parity for Phase 3.
- Leave the support surface open and let benchmarks define it ad hoc later.
- Expand noise scope without a workflow-driven reason.

### Upstream Alignment

Refines:

- `ADR-004`
- `ADR-006`

## P3-ADR-008: Use The Sequential Density Path And Qiskit Aer As Required Validation Baselines

### Status

`Accepted`

### Context

Phase 3 needs both:

- an internal exact reference that reflects SQUANDER's delivered semantics,
- and an external reference that keeps correctness claims credible.

Relying only on internal comparison would weaken the methods paper. Relying only
on an external simulator would miss regressions against the exact delivered Phase
2/3 backend contract.

The planner surface and descriptor-generation layers have also shown that
validation evidence now exists in multiple real layers before runtime numerical
checks:

- planner-entry unsupported behavior already has a machine-reviewable evidence
  surface,
- descriptor-generation unsupported and lossy behavior already has a separate
  machine-reviewable evidence surface,
- and positive supported-path audit bundles already exist at the planner and
  descriptor layers.

Correctness evidence now closes these validation layers into one explicit
package:

- a shared `benchmarks/density_matrix/correctness_evidence/` pipeline emits one
  machine-reviewable correctness-evidence surface instead of leaving validation as
  disconnected slice-local checks,
- the current correctness-evidence package records `25` counted supported cases on the
  selected planner-calibration supported candidate surface,
- it records a bounded external slice of `4` Qiskit Aer cases (`3` microcases
  plus the 4-qubit continuity anchor),
- and it keeps `17` explicit negative boundary cases visible across planner-
  entry, descriptor-generation, and runtime-stage evidence rather than
  collapsing them into one generic exclusion bucket.

### Decision

Phase 3 validation uses a two-baseline model:

- sequential `NoisyCircuit` execution is the required internal exact baseline on
  every mandatory Phase 3 case,
- Qiskit Aer density-matrix simulation is the required external exact reference
  on the mandatory microcases and the current required 4-qubit continuity
  subset,
- and profiler artifacts are required when profiling materially affects
  architecture decisions or benchmark interpretation.

Additional contract details:

- planner-entry unsupported evidence, descriptor-generation unsupported
  evidence, runtime-stage unsupported evidence, and runtime numerical
  correctness evidence should remain distinct in the artifact package rather
  than being collapsed into one generic failure bucket,
- the current correctness-evidence package should preserve one shared positive record surface
  plus one shared negative-boundary surface so later benchmark and publication
  consumers do not need to infer counted status or boundary meaning from
  disconnected outputs,
- and rolled-up validation or benchmark summaries that drive architecture or
  publication conclusions must be checked against the underlying per-case
  records.

### Rationale

- The sequential density path is the exact semantics Phase 3 is obligated to
  preserve.
- Qiskit Aer remains the strongest practical external exact reference for the
  chosen scope.
- This split keeps semantic and scientific validation aligned.

### Consequences

- Every acceleration claim must also be a correctness claim against the
  sequential baseline.
- The benchmark package must record both internal and external validation where
  applicable.
- Validation bundles should preserve the difference between pre-runtime and
  runtime-only failures, because those categories now correspond to real
  contract surfaces.
- Artifact summary consistency is now part of trustworthy validation packaging,
  not only an optional documentation nicety.
- Profiling becomes part of the evidence package when it explains the remaining
  bottleneck or motivates follow-on work.

### Rejected Alternatives

- Use only internal sequential comparison.
- Use only external comparison and ignore the frozen internal semantics.
- Treat profiler evidence as optional narrative material when it drives
  architecture conclusions.

### Upstream Alignment

Refines:

- `ADR-005`
- `PUBLICATIONS.md` Paper 2 evidence requirements
- `REFERENCES.md` Phase 3 baseline shortlist

## P3-ADR-009: Anchor Paper 2 On The Phase 2 Continuity Workflow Plus Structured Noisy Partitioning Families

### Status

`Accepted`

### Context

Using only the Phase 2 continuity workflow would make the Phase 3 paper too
narrow as a methods result. Using only synthetic partitioning circuits would
weaken the connection to the broader noisy-training agenda.

The planner surface and partition-descriptor contract have now frozen the benchmark identity layer more concretely:

- the continuity anchor is a real required case family,
- the external micro-validation slice is a real deterministic microcase
  inventory,
- and the structured methods workloads now exist as real deterministic families
  with stable seed rules and sparse/periodic/dense local-noise labels.

### Decision

Paper 2 and the Phase 3 benchmark package will be anchored on both:

- the frozen Phase 2 noisy XXZ `HEA` workflow as the continuity anchor,
- and structured noisy `U3` / `CNOT` partitioning families as the methods
  stress matrix.

Mandatory benchmark structure:

- 2 to 4 qubit external microcases,
- 4, 6, 8, and 10 qubit continuity cases,
- 8 and 10 qubit structured noisy partitioning families,
- at least 3 seed-fixed instances per mandatory structured family and size,
- and sensitivity over sparse, periodic, and dense local-noise placement.

Additional contract details:

- the benchmark and publication package should preserve stable case-level
  provenance for these workload classes instead of reclassifying cases ad hoc in
  later scripts,
- and the Paper 2 evidence package should be manifest-driven over emitted
  supported and unsupported bundles or rerunnable checkers rather than
  prose-only packaging.

### Rationale

- The continuity anchor keeps the methods paper connected to the Phase 2 exact
  workflow story and to later training research.
- The structured families create a publishable methods matrix that goes beyond a
  single application workflow.
- Noise-placement sensitivity is central to the scientific identity of Phase 3.

### Consequences

- Paper 2 can make a stronger claim than "the Phase 2 workflow is a little
  faster."
- Benchmarks remain bounded enough to be reproducible.
- Publication packaging now has a clear deterministic case-identity layer to
  build on, which strengthens traceability but also raises the bar for summary
  correctness and provenance completeness.
- Negative-evidence surfaces become part of the publishable package where they
  define the claim boundary, not just appendices.
- Additional circuit families can still be added later without changing the
  minimum claim.

### Rejected Alternatives

- Use only the Phase 2 continuity workflow as the Phase 3 benchmark set.
- Use only synthetic partitioning circuits with no continuity anchor.
- Leave noise-placement patterns unspecified.

### Upstream Alignment

Refines:

- `PLANNING.md` Phase 3 benchmark wording
- `PUBLICATIONS.md` Paper 2 scope and evidence boundary
- `REFERENCES.md` Phase 3 shortlist

## P3-ADR-010: Defer Channel-Native Fused Noisy Blocks, Broader Phase 4 Workflow Growth, And Approximate Scaling Beyond Core Phase 3

### Status

`Deferred`

### Context

Several attractive branches sit immediately adjacent to Phase 3:

- fully channel-native or superoperator-native fusion,
- broader noisy VQE/VQA growth and gradients,
- calibration-aware workflow features,
- stochastic trajectories,
- MPDO-style approximate scaling,
- and more aggressive architecture generalization.

All are scientifically relevant, but not all belong in the minimum Phase 3
claim.

### Decision

The following are formally deferred beyond the core Phase 3 deliverable:

- fully channel-native fused noisy blocks,
- broader noisy VQE/VQA workflow surface and gradient routing,
- calibration-aware and readout-oriented workflow features as main Phase 3 work,
- stochastic trajectories and MPDO-style scaling branches,
- and any architecture branch justified only after the native baseline benchmark
  decision gate.

Decision rule:

- if the mandatory Phase 3 benchmark package shows that the native baseline is
  still bottlenecked mainly by the lack of channel-native fusion, open a
  dedicated follow-on branch,
- otherwise keep those branches explicitly out of the minimum Phase 3 claim.

### Rationale

- Deferral protects the clarity of Paper 2.
- It keeps the critical path aligned with the accepted publication ladder.
- It avoids spending major effort before benchmark evidence shows the extra
  complexity is justified.

### Consequences

- Phase 3 can close without claiming the most invasive architecture.
- Deferred branches remain visible and legitimate future work instead of hidden
  technical debt.
- Follow-on architecture decisions become benchmark-driven rather than taste-
  driven.

### Rejected Alternatives

- Make channel-native fusion the minimum Phase 3 architecture immediately.
- Let broader Phase 4 workflow growth expand during Phase 3 implementation.
- Introduce approximate scaling methods before the exact partitioned backend is
  benchmarked.

### Upstream Alignment

Refines:

- `ADR-001`
- `ADR-007`
- `ADR-008`
- `PLANNING.md` Phase 3 decision gate wording

## Phase 3 Decision Gate Summary

At the end of Phase 3, the following questions must all have a positive answer:

1. Can noisy mixed-state circuits be partitioned through the canonical Phase 3
   planner surface without reducing noise to barrier-only metadata?
2. Does the partitioned runtime preserve the sequential density semantics within
   the frozen thresholds?
3. Is there an executable partitioned density path plus at least one real fused
   execution mode on representative noisy workloads?
4. Is the benchmark and reproducibility package strong enough to support Paper 2
   honestly, including the places where the native baseline still falls short?

If the answer to any of these questions is no, then Phase 3 is incomplete even
if substantial implementation work already exists.
