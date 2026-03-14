# Task 8 Stories

This document decomposes Phase 2 Task 8 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_8_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_2.md`, `P2-ADR-001`, `P2-ADR-002`, `P2-ADR-006`,
`P2-ADR-008`, `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`,
`P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`. They describe behavioral slices,
not implementation chores.

Story ordering is intentional:

1. freeze one honest Paper 1 claim set with explicit non-claims,
2. keep abstract, short-paper, and full-paper surfaces aligned while allowing
   different review depths,
3. make major paper claims traceable to authoritative Phase 2 contract and
   evidence,
4. preserve the mandatory evidence floor and claim-closure rule as the only
   basis for the main paper claim,
5. communicate the supported VQE-facing path and exact-regime scale honestly,
6. keep future-work boundaries and publication-ladder positioning explicit,
7. preserve terminology consistency and at least one stable reviewer path
   through the publication package.

## Story 1: Paper 1 Has One Stable Main Claim And Explicit Non-Claims

**User/Research value**
- This story protects scientific credibility by freezing one honest Phase 2
  contribution boundary instead of letting the paper drift into a broader
  roadmap claim.

**Given / When / Then**
- Given the frozen Phase 2 integration contract and the Paper 1 positioning in
  `PUBLICATIONS.md`.
- When a reviewer asks what Paper 1 actually claims and what it explicitly does
  not claim.
- Then the paper package presents one stable main claim plus explicit non-claims
  that prevent Phase 3 to Phase 5 work from being read as already delivered.

**Scope**
- In: Paper 1 main claim, supporting claim set, explicit non-claims, and the
  publication contribution boundary.
- Out: adding new scientific results, expanding the frozen support surface, or
  reopening phase-level scope decisions.

**Acceptance signals**
- An abstract-level claim set identifies one stable Paper 1 main claim centered
  on exact noisy backend integration for one canonical noisy XXZ VQE workflow.
- Supporting claims stay limited to explicit backend selection, exact
  `Re Tr(H*rho)` evaluation, the generated-`HEA` bridge, realistic local-noise
  support, and publication-grade validation.
- Explicit non-claims keep density-aware partitioning, fusion, gradient-path
  completion, approximate scaling, broader noisy-VQA studies, and trainability
  analysis outside the current Paper 1 result.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 8 goal, success
  looks like, and evidence required; `TASK_8_MINI_SPEC.md` required behavior and
  acceptance evidence.
- ADR decision(s): `P2-ADR-002`, `P2-ADR-006`, `P2-ADR-008`.

## Story 2: Abstract, Short-Paper, And Full-Paper Surfaces Tell The Same Phase 2 Story At Different Depths

**User/Research value**
- This story lets different review audiences see the same scientific result
  without forcing every surface to carry the same level of implementation detail
  or background history.

**Given / When / Then**
- Given one frozen Phase 2 result and three publication-facing surfaces with
  different depth expectations.
- When authors or reviewers move between the abstract, the compact short-paper
  narrative, and the full-paper draft.
- Then they find the same claim boundary, evidence bar, and limitation
  structure, expressed at audience-appropriate depth rather than as conflicting
  narratives.

**Scope**
- In: alignment across abstract, short-paper, and full-paper surfaces;
  audience-appropriate compression; and reviewer-readable narrative emphasis.
- Out: venue-specific formatting polish, bibliography expansion, or adding new
  evidence to support a broader claim.

**Acceptance signals**
- The abstract, short-paper, and full-paper surfaces do not conflict about the
  main claim, canonical workflow, benchmark floor, thresholds, or support-tier
  meaning.
- The short-paper surface emphasizes the scientific problem, delivered result,
  evidence, and limitations more than repository history or implementation
  chronology.
- The full-paper surface may add detail, comparison, and context, but it does
  not broaden the frozen Phase 2 contribution boundary.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 8 evidence required;
  `TASK_8_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-001`, `P2-ADR-006`.

## Story 3: Major Paper Claims And Sections Are Traceable To Authoritative Phase 2 Sources

**User/Research value**
- This story makes Paper 1 reviewable by ensuring that every important claim can
  be checked against authoritative contract docs and machine-readable evidence
  instead of resting on prose alone.

**Given / When / Then**
- Given the frozen Phase 2 contract docs, the stable documentation entry points,
  and the Task 6 workflow-facing publication bundle.
- When a reviewer asks which artifact or contract source supports a major Paper
  1 claim, section, or limitation statement.
- Then the publication package provides a stable traceability path from the
  paper surface back to the authoritative docs and evidence.

**Scope**
- In: section-level or equivalent claim-to-evidence mapping, citable document
  paths, and linkage between paper surfaces and the Phase 2 evidence surface.
- Out: generating new validation artifacts or replacing authoritative evidence
  with narrative summaries.

**Acceptance signals**
- Major Paper 1 claims and sections can be mapped to the relevant sources in
  `DETAILED_PLANNING_PHASE_2.md`, `ADRs_PHASE_2.md`,
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`, the relevant Task 1 through Task
  7 mini-specs, and `PHASE_2_DOCUMENTATION_INDEX.md`.
- The workflow-facing publication bundle rooted at
  `benchmarks/density_matrix/artifacts/phase2_task6/task6_story6_publication_bundle.json`
  is identified as the canonical machine-readable evidence surface for the
  workflow-backed claim.
- A reviewer can move from a paper-level statement to the underlying contract or
  evidence without relying on source-code inspection.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 8 evidence required;
  `TASK_8_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-001`, `P2-ADR-006`, `P2-ADR-014`.

## Story 4: Only Mandatory, Complete, Supported Evidence Can Close The Main Paper 1 Claim

**User/Research value**
- This story prevents favorable examples, optional baselines, and incomplete
  bundles from inflating the publication claim beyond what Phase 2 actually
  delivered.

**Given / When / Then**
- Given the frozen mandatory evidence floor and the published workflow-facing
  evidence surface.
- When Paper 1 summarizes the evidence supporting the Phase 2 result.
- Then only mandatory, complete, supported evidence is allowed to close the main
  claim, while optional, deferred, unsupported, or incomplete material remains
  clearly labeled as context or boundary evidence.

**Scope**
- In: mandatory evidence-floor wording, claim-closure semantics, required versus
  optional evidence labeling, and completeness expectations for the publication
  package.
- Out: redefining thresholds, relaxing pass/fail semantics, or promoting
  supplemental evidence into the core claim.

**Acceptance signals**
- The paper package states the mandatory evidence floor explicitly: 1 to 3 qubit
  micro-validation, 4 / 6 / 8 / 10 qubit workflow matrix with 10 fixed
  parameter vectors per required size, at least one reproducible 4- or 6-qubit
  optimization trace, one documented 10-qubit anchor case, runtime and
  peak-memory recording, and the reproducibility bundle.
- The paper package preserves the frozen rule that only mandatory, complete,
  supported evidence closes the main Paper 1 claim.
- Optional whole-register depolarizing, optional secondary baselines,
  unsupported boundary cases, and incomplete bundles are labeled supplemental or
  exclusionary rather than counted as positive completion signals.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 8 success looks like
  and evidence required; Section 12 full-phase acceptance criteria; Section 13
  validation and benchmark matrix; `TASK_8_MINI_SPEC.md` required behavior,
  unsupported behavior, and acceptance evidence.
- ADR decision(s): `P2-ADR-006`, `P2-ADR-014`, `P2-ADR-015`.

## Story 5: Paper 1 Describes The Supported VQE-Facing Path And Exact-Regime Scale Honestly

**User/Research value**
- This story helps readers understand exactly what was demonstrated in Phase 2
  and prevents the paper from overstating backend breadth or achievable scale.

**Given / When / Then**
- Given the frozen backend-selection, observable, bridge, support-matrix, and
  workflow-anchor decisions.
- When Paper 1 explains the supported density-matrix result and the validation
  regime behind it.
- Then it describes the guaranteed VQE-facing path and exact-regime scale
  honestly, without implying broader circuit parity, broader workflow breadth,
  or a scaling claim beyond the documented anchor package.

**Scope**
- In: supported-path wording, exact-regime phrasing, workflow identity,
  support-surface honesty, and scale-boundary communication.
- Out: broader standalone `NoisyCircuit` capability claims, full `qgd_Circuit`
  parity claims, or future acceleration results.

**Acceptance signals**
- Paper 1 states that `density_matrix` requires explicit selection, unsupported
  requests do not silently fall back, and the canonical supported workflow is
  noisy XXZ VQE with default `HEA`, explicit local noise insertion, and exact
  `Re Tr(H*rho)` evaluation.
- Paper 1 presents the exact regime as full end-to-end workflow execution at 4
  and 6 qubits plus benchmark-ready fixed-parameter evaluation at 8 and 10
  qubits, with a documented 10-qubit case as the acceptance anchor.
- Paper 1 distinguishes the guaranteed generated-`HEA` VQE-facing path from
  broader standalone `NoisyCircuit` capability and does not imply full
  `qgd_Circuit` parity.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 8 success looks like;
  Section 10 frozen implementation contracts; Section 12 full-phase acceptance
  criteria; `TASK_8_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-009`, `P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`,
  `P2-ADR-013`.

## Story 6: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package

**User/Research value**
- This story keeps Paper 1 scientifically focused and helps reviewers understand
  how the Phase 2 result fits into the larger PhD program without confusing
  current results with later-phase goals.

**Given / When / Then**
- Given the publication strategy that positions Phase 2 as the first major
  integration paper and later phases as the place for acceleration, optimizer
  studies, and trainability science.
- When Paper 1 discusses motivation, significance, and future directions.
- Then it presents Phase 2 as the exact noisy backend integration milestone and
  keeps later-phase work explicitly labeled as future work rather than current
  evidence.

**Scope**
- In: future-work wording, Phase 2 positioning in the publication ladder, and
  alignment between Paper 1 and the broader roadmap.
- Out: reopening roadmap order, introducing new Phase 2 commitments, or
  reframing Paper 1 as the later acceleration or trainability paper.

**Acceptance signals**
- Paper 1 explicitly marks density-aware partitioning, fusion, gradient-path
  completion, approximate scaling, broader optimizer studies, and trainability
  analysis as later work rather than as current Paper 1 results.
- The paper package positions Phase 2 as the validated exact noisy backend
  integration milestone that enables later phases rather than replacing them.
- Publication-facing wording remains aligned with `PUBLICATIONS.md`,
  `RESEARCH_ALIGNMENT.md`, and `CHANGELOG.md` without overstating current Phase
  2 scope.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md`
  why-Phase-2-precedes-Phase-3 rationale, non-goals, and Task 8 evidence
  required;
  `TASK_8_MINI_SPEC.md` required behavior, unsupported behavior, and acceptance
  evidence.
- ADR decision(s): `P2-ADR-002`, `P2-ADR-008`.

## Story 7: Terminology And Reviewer Navigation Stay Consistent Across The Paper Package

**User/Research value**
- This story makes Paper 1 easier to review and cite by preventing terminology
  drift and by preserving at least one stable path from publication surfaces to
  the authoritative Phase 2 contract and evidence.

**Given / When / Then**
- Given the full publication package across abstract, short-paper, full-paper,
  and evidence-facing sources.
- When a reviewer compares wording, support labels, and references across those
  surfaces.
- Then the paper package reads as one coherent story with stable terms and at
  least one reliable path to the authoritative contract and machine-readable
  evidence.

**Scope**
- In: consistent use of key terms, stable reviewer entry points, and
  cross-reference integrity across publication-facing and contract-facing docs.
- Out: stylistic polish that does not affect meaning and repository navigation
  beyond the Phase 2 publication package.

**Acceptance signals**
- Terms such as `density_matrix`, canonical workflow, exact regime, acceptance
  anchor, required / optional / deferred / unsupported, and reproducibility
  bundle are used consistently across publication-facing surfaces.
- At least one stable reviewer path leads from the paper package to
  `PHASE_2_DOCUMENTATION_INDEX.md` and the canonical workflow-facing publication
  bundle.
- Legacy, roadmap-level, or broader wording does not override the frozen Phase 2
  contract when Paper 1 support claims are interpreted.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_2.md` Task 8 evidence
  requirements; `TASK_8_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P2-ADR-001`, `P2-ADR-006`.
