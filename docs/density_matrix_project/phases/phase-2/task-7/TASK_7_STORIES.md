# Task 7 Stories

This document decomposes Phase 2 Task 7 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_7_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_2.md`, `P2-ADR-001`, `P2-ADR-002`, `P2-ADR-006`,
`P2-ADR-008`, `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`,
`P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`. They describe behavioral slices,
not implementation chores.

Story ordering is intentional:

1. expose one authoritative and citable source-of-truth path for the Phase 2
   documentation contract,
2. make the supported backend-selection and canonical workflow boundary
   understandable without code inspection,
3. distinguish the guaranteed VQE-facing support surface from optional,
   deferred, unsupported, or standalone-only breadth,
4. define the mandatory evidence bar that documentation must preserve for Phase
   2 claims,
5. keep future work and non-goals visibly separate from current commitments,
6. preserve terminology consistency and cross-document traceability across the
   Phase 2 bundle.

## Story 1: The Phase 2 Documentation Contract Is Discoverable Through One Authoritative, Citable Source-Of-Truth Path

**User/Research value**
- This story ensures developers, reviewers, and paper readers can locate the
  authoritative Phase 2 contract without reverse-engineering code or inferring
  scope from scattered notes.

**Given / When / Then**
- Given the frozen Phase 2 planning, ADR, mini-spec, and evidence hierarchy.
- When a reader needs to determine what the density-matrix backend currently
  guarantees and which documents are authoritative for that claim.
- Then the documentation bundle exposes a clear source-of-truth path and at
  least one stable reference that can be cited in reviews, validation artifacts,
  and publication material.

**Scope**
- In: source-of-truth hierarchy, authoritative document roles, stable entry
  points, and citable references for the Phase 2 contract.
- Out: changing the frozen contract decisions themselves or generating new
  validation evidence.

**Acceptance signals**
- The Phase 2 document bundle identifies where backend selection, observable
  scope, bridge scope, support matrix, workflow anchor, benchmark minimum,
  numeric thresholds, and non-goals are authoritatively defined.
- At least one stable documentation entry point can be cited from validation
  artifacts, workflow evidence, or the Paper 1 bundle when reviewers need the
  authoritative Phase 2 support boundary.
- A reader can tell from documentation alone how planning docs, ADRs, task
  mini-specs, and evidence artifacts relate to one another.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 7 goal and evidence
  requirements; source-of-truth hierarchy expectations; `TASK_7_MINI_SPEC.md`
  required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-001`.

## Story 2: The Supported `density_matrix` Entry Surface And Canonical XXZ Workflow Are Explained Unambiguously

**User/Research value**
- This story lets users understand exactly how the supported density-matrix path
  is selected and what canonical Phase 2 workflow that path actually covers.

**Given / When / Then**
- Given the frozen backend-selection, observable, and workflow-anchor
  decisions.
- When a developer or reviewer reads the Phase 2 user-facing or research-facing
  documentation for the supported density-matrix workflow.
- Then they can see that `state_vector` remains the default, `density_matrix`
  requires explicit selection, unsupported requests do not fall back silently,
  and the canonical supported workflow is noisy XXZ VQE with default `HEA`,
  explicit local noise insertion, and exact `Re Tr(H*rho)` evaluation.

**Scope**
- In: backend-selection wording, canonical workflow description, exact-regime
  scale contract, and supported-path explanation for developers and reviewers.
- Out: implementation details of the backend plumbing or benchmark execution
  itself.

**Acceptance signals**
- Documentation states clearly that `state_vector` remains the default,
  `density_matrix` must be selected explicitly for exact noisy mixed-state
  claims, and unsupported density requests fail early instead of falling back.
- Documentation identifies the canonical supported Phase 2 workflow as noisy VQE
  ground-state estimation of a 1D XXZ spin chain with local `Z` field using the
  default `HEA` ansatz, explicit local noise insertion, and exact
  `Re Tr(H*rho)` evaluation.
- Documentation describes the accepted exact-regime contract honestly:
  full end-to-end workflow execution at 4 and 6 qubits, benchmark-ready
  fixed-parameter evaluation at 8 and 10 qubits, and a documented 10-qubit case
  as the acceptance anchor.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` backend-integration and
  exact-observable scope; Task 7 goal and evidence requirements;
  `TASK_7_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-013`.

## Story 3: The Guaranteed VQE-Facing Support Surface Is Distinguished From Broader Or Deferred Capability

**User/Research value**
- This story protects scientific honesty by making it easy to tell which gates,
  noise models, circuit sources, and execution paths are guaranteed in Phase 2
  and which are only optional, deferred, unsupported, or broader standalone
  capabilities.

**Given / When / Then**
- Given the frozen bridge and support-matrix decisions for the Phase 2 density
  path.
- When a reader checks the documentation to understand which requests are
  guaranteed on the VQE-facing density-matrix route.
- Then the documentation distinguishes the generated-`HEA` path from broader
  `qgd_Circuit` parity, keeps required / optional / deferred / unsupported
  labels explicit, and does not let broader standalone `NoisyCircuit` breadth
  masquerade as guaranteed VQE-facing support.

**Scope**
- In: bridge boundary wording, gate and noise support classification, supported
  circuit-source language, and separation of guaranteed workflow support from
  broader standalone capability.
- Out: promoting optional or deferred features into the frozen Phase 2 contract.

**Acceptance signals**
- Documentation makes clear that the guaranteed Phase 2 bridge is the generated
  default `HEA` path and that full `qgd_Circuit` parity is not implied.
- Documentation consistently labels required, optional, deferred, and
  unsupported gate and noise behavior, including keeping whole-register
  depolarizing optional and later extensions non-mandatory unless a new phase
  decision says otherwise.
- A reviewer can tell from documentation alone that broader standalone
  `NoisyCircuit` capability does not automatically equal guaranteed VQE-facing
  Phase 2 support.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` bridge and support-matrix
  decisions; Task 7 goal and evidence requirements; `TASK_7_MINI_SPEC.md`
  required behavior, unsupported behavior, and acceptance evidence.
- ADR decision(s): `P2-ADR-011`, `P2-ADR-012`.

## Story 4: The Mandatory Evidence Bar Is Documented As The Only Basis For Core Phase 2 Claims

**User/Research value**
- This story keeps documentation from turning favorable examples or optional
  extensions into inflated support claims by preserving the real evidence bar
  for Phase 2 completion and Paper 1.

**Given / When / Then**
- Given the frozen validation, benchmark, and reproducibility requirements for
  the Phase 2 density-matrix workflow.
- When documentation summarizes what evidence supports the Phase 2 support
  surface.
- Then it identifies the mandatory Aer-centered validation package, the required
  exact-regime workflow evidence, and the reproducibility bundle clearly enough
  that optional, unsupported, or incomplete evidence cannot be mistaken for the
  main claim.

**Scope**
- In: mandatory evidence-bar wording, distinction between required and optional
  evidence, exactness-threshold summaries, and reproducibility expectations.
- Out: running the benchmarks, generating new evidence, or redefining the
  thresholds themselves.

**Acceptance signals**
- Documentation identifies the mandatory evidence package: Aer-centered
  comparison, 1 to 3 qubit micro-validation, 4 / 6 / 8 / 10 qubit workflow
  matrix, at least one reproducible 4- or 6-qubit optimization trace, a
  documented 10-qubit anchor case, and the reproducibility bundle.
- Documentation preserves the frozen interpretation of numeric thresholds,
  `100%` pass-rate expectations, runtime and peak-memory recording, and
  case-level auditability as required evidence rather than optional extras.
- Documentation states explicitly that favorable examples, optional secondary
  baselines, unsupported boundary cases, or incomplete bundles do not replace
  the mandatory evidence package for the core Phase 2 claim.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 7 evidence
  requirements; Section 12 full-phase acceptance criteria; Section 13 validation
  and benchmark matrix; `TASK_7_MINI_SPEC.md` required behavior and acceptance
  evidence.
- ADR decision(s): `P2-ADR-006`, `P2-ADR-014`, `P2-ADR-015`.

## Story 5: Future Work And Non-Goals Stay Visibly Separated From Current Phase 2 Commitments

**User/Research value**
- This story prevents roadmap drift by keeping future capabilities interesting
  but non-binding, so developers and reviewers do not confuse later-phase work
  with the delivered Phase 2 milestone.

**Given / When / Then**
- Given the frozen Phase 2 focus on exact noisy backend integration and the
  deliberate deferral of later-phase optimization and trainability work.
- When roadmap-facing, developer-facing, or publication-facing documentation
  mentions adjacent work beyond the frozen Phase 2 contract.
- Then those items remain explicitly labeled as future work or non-goals, and
  the roadmap order is preserved instead of blurred.

**Scope**
- In: explicit future-work wording, non-goal visibility, roadmap ordering, and
  alignment between Phase 2 docs and broader roadmap-facing docs.
- Out: changing the roadmap, reopening deferred decisions, or adding new Phase 2
  commitments.

**Acceptance signals**
- Documentation explicitly marks density-aware partitioning, fusion,
  gradient-path completion, approximate scaling, broad optimizer studies, and
  trainability analysis as later-phase work rather than current Phase 2
  commitments.
- Phase 2 documentation preserves roadmap order by describing Phase 2 as the
  exact noisy backend integration milestone and later phases as the place for
  acceleration, broader noisy-VQA integration, and trainability studies.
- Roadmap-facing wording remains aligned across the Phase 2 bundle,
  `RESEARCH_ALIGNMENT.md`, and `CHANGELOG.md` without overstating current Phase
  2 scope.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` out-of-scope,
  why-Phase-2-precedes-Phase-3, Task 7 evidence requirements, and non-goals;
  `TASK_7_MINI_SPEC.md` required behavior, unsupported behavior, and acceptance
  evidence.
- ADR decision(s): `P2-ADR-002`, `P2-ADR-008`.

## Story 6: Terminology, Support Labels, And Cross-References Stay Consistent Across The Phase 2 Bundle

**User/Research value**
- This story makes the documentation reviewable and publishable by preventing
  conflicting labels, ambiguous terminology, or inconsistent claim wording from
  weakening the Phase 2 narrative.

**Given / When / Then**
- Given the full Phase 2 document bundle across planning, ADRs, mini-specs,
  user-facing notes, and publication-facing summaries.
- When a reviewer compares how the same support claims are described across
  those surfaces.
- Then terminology, support labels, and cross-references remain consistent
  enough that the Phase 2 contract reads as one coherent story rather than as a
  set of near-duplicate but conflicting descriptions.

**Scope**
- In: consistent use of backend names, workflow labels, support classifications,
  exact-regime language, acceptance-anchor wording, reproducibility terminology,
  and cross-document references.
- Out: editorial polish that does not affect meaning, and full manuscript
  authoring beyond the Phase 2 contract boundary.

**Acceptance signals**
- Terms such as `density_matrix`, `state_vector`, exact regime, acceptance
  anchor, canonical workflow, reproducibility bundle, and required / optional /
  deferred / unsupported are used consistently across the Phase 2 bundle.
- Cross-references between planning, ADR, mini-spec, and user-facing
  documentation do not conflict about backend defaults, workflow scope, support
  boundaries, or evidence requirements.
- Legacy or broader roadmap wording that could overstate the current support
  surface is either clarified in Phase 2-specific context or prevented from
  overriding the frozen Phase 2 contract.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 7 evidence
  requirements; `TASK_7_MINI_SPEC.md` required behavior and acceptance
  evidence.
- ADR decision(s): `P2-ADR-001`, `P2-ADR-006`.
