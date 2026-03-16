# Task 8: Paper 1 Evidence Package

This mini-spec turns Phase 2 Task 8 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-001`, `P2-ADR-002`, `P2-ADR-006`, `P2-ADR-008`, `P2-ADR-009`,
`P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`, `P2-ADR-013`, `P2-ADR-014`, and
`P2-ADR-015`, plus the closed backend-selection, observable, bridge,
support-matrix, workflow-anchor, benchmark-minimum, and numeric-threshold items
in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen Phase 2
scientific scope, main-claim closure rules, workflow identity, or later-phase
boundary decisions.

## Given / When / Then
- Given the frozen Phase 2 contract, the delivered canonical workflow evidence
  surface, and the stable documentation entry points that define supported and
  deferred scope.
- When an author, reviewer, or collaborator assembles or audits Paper 1 across
  abstract, short-paper, and full-paper surfaces.
- Then every publication claim can be traced to mandatory Phase 2 evidence,
  optional or deferred material remains labeled as such, and the paper package
  communicates one honest integration milestone rather than a broader future
  roadmap claim.

## Assumptions and dependencies
- Task 1 provides the explicit backend-selection contract, including default
  `state_vector`, explicit `density_matrix`, and no implicit fallback behavior.
- Task 2 provides the exact noisy observable contract `Re Tr(H*rho)` and the
  frozen observable scope that Paper 1 may describe as delivered.
- Task 3 provides the generated-`HEA` bridge contract into `NoisyCircuit`,
  including explicit unsupported-case boundaries for broader circuit sources.
- Task 4 provides the required / optional / deferred / unsupported support split
  for gate and noise coverage, including the mandatory realistic local-noise
  baseline.
- Task 5 provides the mandatory validation minimum, Aer-centered pass/fail
  interpretation, numeric thresholds, reproducibility requirements, and the rule
  that only mandatory complete supported evidence closes the main claim.
- Task 6 provides the canonical noisy XXZ VQE workflow contract plus the
  workflow-facing publication bundle rooted at
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`.
- Task 7 provides the citable documentation boundary and terminology discipline
  that Paper 1 must reuse rather than redefine.
- `docs/density_matrix_project/planning/PUBLICATIONS.md` defines Paper 1 as the
  Phase 2 integration paper, not as the later partitioning, fusion,
  optimization, or trainability paper.
- Publication-facing docs such as `ABSTRACT_PHASE_2.md`,
  `SHORT_PAPER_PHASE_2.md`, and `PAPER_PHASE_2.md` are packaging surfaces for
  the frozen contract and evidence; they must not outrun the delivered evidence
  or broaden the scientific claim set.
- This task defines how Phase 2 evidence is packaged, traced, and narrated for
  publication. It does not require new backend features, new benchmark classes,
  or new scientific claims beyond the frozen Phase 2 contract.

## Required behavior
- Task 8 freezes the Paper 1 publication package for Phase 2 as the
  authoritative mapping from delivered Phase 2 evidence to paper-level claims.
- Paper 1 must have one stable main-claim boundary: SQUANDER's exact
  density-matrix backend is integrated into one canonical noisy XXZ VQE
  workflow through explicit backend selection, exact Hermitian-energy
  evaluation, a generated-`HEA` bridge, realistic local-noise support, and a
  publication-grade validation package.
- The publication package must define explicit non-claims for Paper 1 so later
  work is not pulled into the current contribution boundary. Density-aware
  partitioning, fusion, gradient-path completion, approximate scaling, broader
  noisy-VQA studies, and trainability analysis remain future work rather than
  current Paper 1 results.
- Abstract, short-paper, and full-paper surfaces must describe the same frozen
  Phase 2 result at different levels of detail rather than introduce conflicting
  claims, scales, thresholds, or support labels.
- The publication package must preserve the frozen Phase 2 claim-closure rule:
  only mandatory, complete, supported evidence closes the main Paper 1 claim.
  Optional, deferred, unsupported, or incomplete evidence may provide context,
  but it must not replace the required evidence surface.
- Paper 1 wording must remain consistent with the frozen Phase 2 contract for
  backend selection, observable scope, bridge scope, support matrix, workflow
  anchor, benchmark minimum, numeric thresholds, and non-goals.
- The publication package must present the exact-regime scale honestly: full
  end-to-end workflow execution at 4 and 6 qubits, benchmark-ready
  fixed-parameter evaluation at 8 and 10 qubits, and a documented 10-qubit case
  as the acceptance anchor rather than as a broad scaling claim.
- Paper 1 must distinguish the guaranteed VQE-facing density path from broader
  standalone `NoisyCircuit` capability. Full `qgd_Circuit` parity or broad
  manual-circuit reuse must not be implied.
- The publication package must identify the workflow-facing publication bundle
  as the canonical machine-readable evidence surface and preserve traceability
  to the underlying validation-evidence layers rather than replacing them with
  narrative summaries alone.
- Task 8 must define an abstract-level claim set, a short-paper structure, and a
  full-paper structure that each map clearly to delivered evidence and explicit
  scope boundaries.
- Section-level or equivalent publication traceability must exist from the Phase
  2 deliverables into the paper package so a reviewer can determine which
  artifact supports each major contribution claim, validation statement, and
  limitation statement.
- Publication terminology must remain consistent across abstract, short-paper,
  full-paper, and evidence-facing surfaces for terms such as `density_matrix`,
  canonical workflow, exact regime, acceptance anchor, required / optional /
  deferred / unsupported, and reproducibility bundle.
- Reviewer-facing entry points must remain stable enough that a paper reader can
  find the authoritative contract, workflow evidence surface, and deferred-scope
  boundary without reverse-engineering the repository.
- Task 8 completion means Paper 1 is claim-ready, evidence-traceable, and
  reviewable as an honest Phase 2 integration milestone. It does not require
  venue-specific camera-ready polishing or new evidence outside the frozen
  contract.

## Unsupported behavior
- Drafting Paper 1 in a way that implies Phase 2 already delivers density-aware
  partitioning, gate fusion, gradient support, approximate scaling, broad
  optimizer studies, or trainability analysis.
- Presenting the paper as evidence of broad `qgd_Circuit` parity, broad manual
  circuit ingestion, or generic noisy-backend generality beyond the documented
  generated-`HEA` VQE-facing path.
- Allowing abstract, short-paper, and full-paper variants to disagree about the
  main claim, workflow identity, benchmark floor, thresholds, or support-tier
  meaning.
- Treating optional whole-register depolarizing, optional secondary baselines,
  exploratory cases, or favorable but incomplete examples as sufficient closure
  of the main Paper 1 claim.
- Building the paper package from detached plots or narrative summaries without
  stable traceability back to the required Phase 2 evidence artifacts.
- Replacing the Task 5 and Task 6 evidence package with prose-only summaries that
  hide case identity, pass/fail interpretation, or reproducibility details.
- Reopening phase-frozen decisions about backend semantics, observable scope,
  bridge scope, support tiers, workflow anchor, benchmark minimum, or numeric
  thresholds inside paper-facing docs.
- Framing later-phase roadmap items as current Phase 2 commitments, or downgrading
  delivered required evidence into optional context.
- Treating formatting polish alone as Task 8 completion if the claim set,
  evidence mapping, and deferred-boundary wording remain ambiguous.

## Acceptance evidence
- An abstract-level claim set identifies the one-sentence Paper 1 main claim,
  the required supporting claims, and the explicit non-claims, and it aligns
  with `docs/density_matrix_project/planning/PUBLICATIONS.md`.
- A short-paper structure identifies the required sections for the compact Paper
  1 narrative and preserves the frozen contribution boundary and evidence bar.
- A full-paper structure identifies the required sections for the long-form Paper
  1 narrative and preserves the same claim boundary, evidence bar, and deferred
  scope.
- Publication-facing docs consistently describe backend selection, exact
  `Re Tr(H*rho)` evaluation, the generated-`HEA` bridge, the support matrix, the
  canonical noisy XXZ VQE workflow, the exact regime, and the claim-closure rule
  without contradicting Phase 2 contract docs.
- The top-level workflow-facing evidence bundle at
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`
  is cited or linked as the canonical machine-readable publication evidence
  surface.
- Traceability exists from major Paper 1 claims and sections to the underlying
  Phase 2 sources: `DETAILED_PLANNING_PHASE_2.md`, `ADRs_PHASE_2.md`,
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`, the relevant Task 1 through Task
  7 mini-specs, `PHASE_2_DOCUMENTATION_INDEX.md`, and the Task 6 publication
  bundle.
- The paper package states the mandatory evidence floor explicitly: 1 to 3 qubit
  micro-validation, 4 / 6 / 8 / 10 qubit workflow matrix with 10 fixed
  parameter vectors per size, at least one reproducible 4- or 6-qubit
  optimization trace, one documented 10-qubit anchor case, runtime and
  peak-memory recording, and the reproducibility bundle.
- The paper package preserves the frozen interpretation rule that only
  mandatory, complete, supported evidence closes the main claim, while optional,
  deferred, unsupported, or incomplete material remains labeled as context or
  boundary evidence.
- Negative review evidence shows that out-of-scope topics such as partitioning,
  fusion, gradients, approximate scaling, broader workflows, and trainability
  are explicitly described as future work rather than as current Paper 1
  results.
- A reviewer can follow at least one stable documentation path from the paper
  package to the authoritative Phase 2 contract and to the canonical workflow
  evidence surface without relying on source-code inspection.
- Traceability target: satisfy the Phase 2 Task 8 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: satisfy the full-phase acceptance criterion requiring a
  Paper 1 evidence package complete enough to support abstract, short-paper, and
  full-paper drafting honestly.
- Traceability target: satisfy the publication-readiness and future-work
  decisions frozen in `P2-ADR-001`, `P2-ADR-002`, `P2-ADR-006`, `P2-ADR-008`,
  `P2-ADR-013`, `P2-ADR-014`, and `P2-ADR-015`, while accurately reflecting the
  concrete interface and support-boundary decisions in `P2-ADR-009` through
  `P2-ADR-012`.

## Affected interfaces
- Publication-facing summaries rooted in `ABSTRACT_PHASE_2.md`,
  `SHORT_PAPER_PHASE_2.md`, and `PAPER_PHASE_2.md`.
- Stable documentation entry points such as
  `PHASE_2_DOCUMENTATION_INDEX.md` that reviewers use to locate the authoritative
  contract and evidence.
- The workflow-facing publication manifest
  `benchmarks/density_matrix/artifacts/workflow_evidence/workflow_publication_bundle.json`
  and the linked Task 6 evidence artifacts it references.
- Phase-level contract docs and decisions that the paper package must cite
  accurately: `DETAILED_PLANNING_PHASE_2.md`, `ADRs_PHASE_2.md`, and
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Task-level contract docs, especially `TASK_1_MINI_SPEC.md` through
  `TASK_7_MINI_SPEC.md`, that provide the paper's support-surface, validation,
  workflow, and terminology references.
- Publication-facing claim, section, and traceability surfaces that connect
  deliverables to reviewer-readable paper structure.
- Change classification: additive and clarifying for publication packaging, but
  stricter for overbroad or ambiguous claim wording, which must become explicit
  scope-boundary statements rather than narrative shortcuts.

## Publication relevance
- Directly defines the Phase 2 Paper 1 claim boundary and evidence package so
  the first major publication can be drafted without inflating scope.
- Reduces review risk by making the paper package traceable to delivered Phase 2
  artifacts instead of relying on informal narrative confidence.
- Keeps Paper 1 aligned with the PhD publication ladder by presenting Phase 2 as
  the exact noisy backend integration milestone rather than as a premature
  acceleration or trainability paper.
- Creates a reusable publication handoff that later phases can extend without
  retroactively changing what the Phase 2 result meant.
