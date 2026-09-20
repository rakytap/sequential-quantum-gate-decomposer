# Task 1 Stories

This document decomposes Phase 3 Task 1 into Layer 3 behavioral stories. These
stories inherit the frozen contract from `TASK_1_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`, `P3-ADR-007`, and `P3-ADR-009`.
They describe behavioral slices, not implementation chores.

Story ordering is intentional:

1. establish the Phase 2 continuity anchor on the canonical planner surface,
2. establish shared canonical coverage for the mandatory Phase 3 methods
   workloads,
3. make gate/noise semantics and entry-route provenance auditable at planner
   entry,
4. allow bounded exact lowering from optional legacy source surfaces without
   implying full source parity,
5. reject unsupported requests before execution with no silent fallback.

## Story 1: The Frozen Phase 2 Continuity Workflow Reaches The Canonical Planner Surface

**User/Research value**
- Keeps Phase 3 scientifically connected to the frozen Phase 2 noisy XXZ
  `HEA` workflow rather than redefining the planner around only synthetic
  methods cases.

**Given / When / Then**
- Given the frozen Phase 2 noisy XXZ `HEA` continuity workflow inside the Phase
  3 support surface.
- When the workflow is prepared for `partitioned_density` planning.
- Then it lowers exactly into the canonical ordered noisy mixed-state planner
  surface before partition planning begins, without requiring new Phase 4
  workflow growth.

**Scope**
- In: exact normalization of the continuity anchor into the canonical planner
  surface, auditable continuity-entry routing, and supported continuity-case
  representability.
- Out: partition execution, fused runtime behavior, new VQE feature growth, and
  broader workflow expansion beyond the frozen continuity anchor.

**Acceptance signals**
- Supported 4, 6, 8, and 10 qubit continuity cases normalize into the canonical
  planner surface before partition planning.
- Validation artifacts can show that the canonical planner contract, not a
  source-specific ad hoc adapter, was used for supported continuity cases.

**Traceability**
- Phase requirement(s): Task 1 goal, success-looks-like, and evidence-required
  sections; planner input decision; workflow and benchmark anchor decision in
  `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-009`

## Story 2: Mandatory Microcases And Structured Families Share One Canonical Planner Contract

**User/Research value**
- Gives Paper 2 a coherent methods surface by showing that the required
  micro-validation cases and structured noisy benchmark families enter one
  common planner contract rather than separate adapters.

**Given / When / Then**
- Given the required 2 to 4 qubit micro-validation cases and the mandatory
  structured noisy `U3` / `CNOT` benchmark families inside the frozen support
  matrix.
- When those workloads are prepared for `partitioned_density` planning.
- Then they are representable through the same canonical ordered noisy
  mixed-state planner surface used by the continuity anchor.

**Scope**
- In: canonical representability of required microcases, canonical
  representability of required structured families, and use of one shared
  planner-entry contract across mandatory workload classes.
- Out: broader gate-family expansion, optional benchmark families, runtime
  speedup claims, and cost-model calibration.

**Acceptance signals**
- Required 2 to 4 qubit micro-validation cases can be expressed through the
  canonical planner surface without widening the frozen support matrix.
- At least one seed-fixed instance from each mandatory structured noisy family
  enters the same canonical planner contract used by the continuity anchor.

**Traceability**
- Phase requirement(s): Task 1 goal and evidence-required sections; support
  matrix decision; workflow and benchmark anchor decision; benchmark minimum
  decision in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-007`, `P3-ADR-009`

## Story 3: Gate And Noise Operations Are First-Class And Auditable At Planner Entry

**User/Research value**
- Makes the noisy mixed-state semantics visible where the Phase 3 methods claim
  lives, which is necessary before later tasks can preserve or accelerate those
  semantics credibly.

**Given / When / Then**
- Given a supported Phase 3 request that reaches canonical planner entry.
- When the planner surface is inspected, tested, or recorded for validation.
- Then gate operations and noise operations appear as first-class ordered
  planner objects, and the route by which the workload reached the planner
  surface is auditable.

**Scope**
- In: first-class representation of `GateOperation` and `NoiseOperation`,
  auditable ordered planner input, and entry-route provenance for supported
  cases.
- Out: partition descriptor metadata beyond planner entry, runtime execution
  scheduling, and any transformation that reorders across noise boundaries.

**Acceptance signals**
- Planner-entry inspection or normalization evidence shows explicit ordered gate
  and noise objects rather than boundary-only or side-channel metadata.
- Reproducibility or validation artifacts identify whether a supported case
  reached the planner surface through continuity lowering, structured-family
  construction, or another exact lowering path.

**Traceability**
- Phase requirement(s): Task 1 success-looks-like and evidence-required
  sections; planner input decision; semantic-preservation decision in
  `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-004`

## Story 4: Optional Legacy Source Lowering Is Allowed Only When Exact And In-Bounds

**User/Research value**
- Preserves room to reuse existing source surfaces such as `qgd_Circuit` or
  `Gates_block` without overstating Phase 3 as full source-parity work.

**Given / When / Then**
- Given an optional legacy source surface such as `qgd_Circuit` or
  `Gates_block` that may lower into the frozen Phase 3 gate and noise surface.
- When that source is submitted for `partitioned_density` planning.
- Then it may enter through the canonical planner surface only if the lowering
  is exact, auditable, and remains inside the documented support matrix.

**Scope**
- In: exact optional lowering from legacy source surfaces into the canonical
  planner contract when the lowered result stays in-bounds.
- Out: full direct parity for every circuit source, heuristic rewriting,
  partial lowering, and source-surface growth beyond the frozen support matrix.

**Acceptance signals**
- At least one in-bounds exact lowering path from an optional legacy source can
  be represented through the canonical planner surface without changing the
  supported claim boundary.
- Unsupported legacy-source features do not become implicitly supported merely
  because some exact lowerings are allowed.

**Traceability**
- Phase requirement(s): Task 1 required behavior and unsupported behavior in
  `TASK_1_MINI_SPEC.md`; planner input decision; support matrix decision in
  `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-007`

## Story 5: Unsupported Planner Requests Fail Before Execution With No Silent Fallback

**User/Research value**
- Protects scientific credibility by ensuring unsupported
  `partitioned_density` requests cannot quietly run on a different execution
  path while still being reported as Phase 3 behavior.

**Given / When / Then**
- Given a request whose source, gate family, noise model, or mode falls outside
  the frozen Phase 3 planner contract.
- When the request is normalized for `partitioned_density` planner entry.
- Then it fails before execution with deterministic unsupported-case
  diagnostics and no silent fallback to sequential density, state-vector, or
  hidden non-partitioned execution.

**Scope**
- In: pre-execution unsupported-case detection at planner entry, explicit hard
  failure for out-of-contract requests, and no-fallback interpretation for
  claimed `partitioned_density` behavior.
- Out: runtime failures inside otherwise supported execution, later-task
  semantic-preservation checks, and performance diagnosis on supported cases.

**Acceptance signals**
- Negative tests show unsupported source, gate, noise, or malformed
  `partitioned_density` requests fail before execution.
- Validation and benchmark workflows do not record unsupported requests as
  supported partitioned cases through fallback or silent substitution.

**Traceability**
- Phase requirement(s): Task 1 success-looks-like and evidence-required
  sections; planner input decision; unsupported-behavior rule in
  `DETAILED_PLANNING_PHASE_3.md` and `TASK_1_MINI_SPEC.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-007`
