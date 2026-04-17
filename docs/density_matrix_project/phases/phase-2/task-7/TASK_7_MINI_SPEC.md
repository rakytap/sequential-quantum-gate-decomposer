# Task 7: Documentation and User-Facing Clarity

This mini-spec turns Phase 2 Task 7 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_2.md`,
`P2-ADR-001`, `P2-ADR-002`, `P2-ADR-006`, `P2-ADR-008`, `P2-ADR-009`,
`P2-ADR-010`, `P2-ADR-011`, `P2-ADR-012`, `P2-ADR-013`, `P2-ADR-014`, and
`P2-ADR-015`, plus the closed backend-selection, observable, bridge,
support-matrix, workflow-anchor, benchmark-minimum, and numeric-threshold items
in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen phase
focus, future-work, support-surface, workflow-anchor, or acceptance-threshold
decisions.

## Given / When / Then
- Given the frozen Phase 2 contract for backend selection, observable scope,
  bridge scope, support matrix, workflow anchor, validation package, and
  deferred later-phase work.
- When a developer, reviewer, or paper reader consults the Phase 2
  documentation bundle to determine what the density-matrix backend currently
  guarantees and what evidence supports those guarantees.
- Then they can identify the supported workflow boundary, the required evidence
  bar, and the explicit future-work boundary without reverse-engineering code or
  inferring scope from partial implementation details.

## Assumptions and dependencies
- Task 1 provides the explicit backend-selection contract, including default
  `state_vector`, explicit `density_matrix`, and no implicit fallback behavior.
- Task 2 provides the exact noisy Hermitian-energy contract
  `Re Tr(H*rho)` for the supported VQE path and the frozen observable semantics
  that documentation must describe consistently.
- Task 3 provides the partial `HEA`-first bridge contract into
  `NoisyCircuit`, including hard-error behavior for unsupported lowering
  requests.
- Task 4 provides the required / optional / deferred support split for gate and
  noise coverage, including the mandatory realistic local-noise baseline.
- Task 5 provides the Aer-centered validation minimum, numeric thresholds,
  reproducibility requirements, and pass/fail interpretation that Task 7 must
  describe without weakening.
- Task 6 provides the canonical noisy XXZ VQE workflow contract and evidence
  bundle that Task 7 must make discoverable and interpretable.
- Upstream roadmap-facing documents such as
  `docs/density_matrix_project/CHANGELOG.md`,
  `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`,
  `docs/density_matrix_project/planning/PLANNING.md`, and
  `docs/density_matrix_project/planning/PUBLICATIONS.md` may mention adjacent or
  later-phase work; Task 7 must align user-facing wording with the frozen Phase
  2 contract rather than with broader roadmap aspirations.
- Task 8 turns the frozen evidence package into paper-ready publication
  material; Task 7 supplies the citable contract and clarity boundary for that
  work, but does not define new publication claims or new evidence thresholds.
- This task defines Phase 2 documentation clarity and traceability. It does not
  add new backend features, broaden support, or replace the implementation and
  evidence duties owned by earlier tasks.

## Required behavior
- Task 7 freezes a coherent documentation contract for Phase 2 so the
  density-matrix backend can be used, reviewed, and discussed without requiring
  direct inspection of implementation internals.
- The Phase 2 document bundle must make the source-of-truth hierarchy explicit:
  upstream planning docs, Phase 2 planning and ADRs, task mini-specs, and
  evidence artifacts must have visible roles instead of appearing as unrelated
  notes.
- Documentation must present a stable and internally consistent description of
  the frozen Phase 2 support surface: backend selection, exact observable path,
  bridge scope, gate and noise support matrix, workflow anchor, benchmark
  minimum, numeric thresholds, and non-goals.
- Documentation must state clearly that `state_vector` remains the default,
  `density_matrix` requires explicit selection for exact noisy mixed-state
  claims, and unsupported density requests fail early instead of falling back.
- Documentation must describe the canonical supported Phase 2 workflow as noisy
  VQE ground-state estimation of a 1D XXZ spin chain with local `Z` field using
  the default `HEA` ansatz, explicit local noise insertion, and exact
  `Re Tr(H*rho)` evaluation.
- Documentation must describe the exact-regime scale contract honestly:
  full end-to-end workflow execution, including at least one reproducible
  optimization trace, at 4 and 6 qubits; benchmark-ready fixed-parameter
  evaluation at 8 and 10 qubits; and a documented 10-qubit anchor case as the
  acceptance anchor rather than a broad scaling claim.
- Documentation must explain the bridge boundary clearly enough that readers can
  distinguish the guaranteed generated-`HEA` path from broader future or
  optional circuit-sourcing possibilities; full `qgd_Circuit` parity must not
  be implied.
- Documentation must distinguish required, optional, deferred, and unsupported
  gate and noise behavior consistently across developer-facing, research-facing,
  and publication-facing surfaces.
- Documentation must keep the guaranteed VQE-facing support surface distinct
  from broader standalone `NoisyCircuit` capabilities so implementation breadth
  is not mistaken for Phase 2 workflow guarantees.
- Documentation must expose the minimum validation and benchmark contract
  clearly enough that a reader can tell what counts as mandatory evidence:
  Aer-centered comparison, micro-validation, workflow-scale cases, numeric
  thresholds, `100%` pass-rate expectations, runtime and peak-memory recording,
  and the reproducibility bundle.
- Documentation must state explicitly that favorable examples, optional
  extensions, or exploratory runs do not replace the frozen mandatory evidence
  package for Phase 2 claims.
- Documentation must make Phase 2 non-goals and future-work boundaries visible,
  including deferral of density-aware partitioning, fusion, gradient-path
  completion, approximate scaling, broad noisy-VQA optimizer studies, and
  trainability analysis to later phases.
- Documentation must preserve roadmap order rather than blur it: Phase 2 is the
  exact noisy backend integration milestone, while optimization, scaling, and
  trainability milestones belong to later phases.
- Documentation terminology must remain consistent across the Phase 2 bundle for
  phrases such as `density_matrix`, exact regime, acceptance anchor, required /
  optional / deferred / unsupported, canonical workflow, and reproducibility
  bundle.
- Documentation must provide at least one stable, citable way for users and
  reviewers to locate the authoritative Phase 2 contract and the artifacts that
  support it.
- Task 7 completion means the Phase 2 support surface and evidence boundaries
  are coherent, discoverable, and citable. It does not require new algorithm
  families, new benchmark classes, or expansion of the frozen publication claim
  boundary.

## Unsupported behavior
- Leaving the effective Phase 2 contract discoverable only by reading code,
  tests, or benchmark scripts.
- Publishing or retaining conflicting documentation about backend defaults,
  fallback behavior, observable scope, bridge scope, support matrix, workflow
  anchor, or benchmark thresholds.
- Implying that full `qgd_Circuit` parity, broad manual circuit reuse, or
  general `NoisyCircuit` breadth is guaranteed on the VQE-facing density path.
- Presenting optional whole-register depolarizing, generalized amplitude
  damping, coherent over-rotation, or other optional extensions as required
  Phase 2 support without an explicit new phase-level decision and matching
  evidence.
- Describing partitioning, fusion, gradient routing, approximate scaling,
  optimizer-comparison studies, or trainability analysis as current Phase 2
  commitments instead of future work.
- Using vague "supports noisy workflows" wording that hides the canonical XXZ +
  `HEA` anchor, the 4 / 6 / 8 / 10 qubit regime, or the exact-evidence
  thresholds.
- Treating one or two favorable examples, or documentation polish alone, as
  sufficient closure if the frozen support boundary and mandatory evidence bar
  remain ambiguous.
- Allowing roadmap-facing wording from older or broader documents to overstate
  the current Phase 2 support surface without Phase 2-specific clarification.
- Rewriting the meaning of required, optional, deferred, unsupported, or
  non-goal classifications at the task-documentation level.

## Acceptance evidence
- The Phase 2 document bundle identifies the authoritative references for
  backend selection, observable scope, bridge scope, support matrix, workflow
  anchor, benchmark minimum, numeric thresholds, and non-goals.
- A developer-facing documentation surface explains how explicit
  `density_matrix` selection is requested, how unsupported requests are
  interpreted, and how the guaranteed VQE-facing density path differs from
  out-of-scope or future-path requests.
- A research-facing documentation surface or equivalent summary describes the
  canonical noisy XXZ VQE workflow, the exact-regime scale contract, and the
  required evidence package without overstating generality.
- Documentation and cross-references consistently distinguish required,
  optional, deferred, and unsupported behavior for gates, noise models, circuit
  sources, workflow classes, and benchmark evidence.
- Documentation explicitly states the later-phase boundary for partitioning,
  fusion, gradients, approximate scaling, optimizer studies, and trainability,
  and keeps that wording aligned with
  `docs/density_matrix_project/RESEARCH_ALIGNMENT.md`,
  `docs/density_matrix_project/CHANGELOG.md`, and the Phase 2 ADR set.
- A terminology and claim pass over the Phase 2 bundle shows consistent use of
  backend names, workflow labels, exactness language, evidence labels, and
  non-goal terminology across planning, ADR, mini-spec, and user-facing
  references.
- At least one stable documentation reference or entry point can be cited from
  validation artifacts, workflow evidence, and the paper bundle when reviewers
  need the authoritative Phase 2 support boundary.
- Negative review evidence shows that representative out-of-scope requests are
  documented as deferred or unsupported rather than left ambiguous.
- Traceability target: satisfy the Phase 2 Task 7 evidence requirements in
  `DETAILED_PLANNING_PHASE_2.md`.
- Traceability target: support the full-phase acceptance criteria that require a
  documented, honest, publication-ready exact noisy workflow boundary.
- Traceability target: satisfy the documentation-contract and future-work
  decisions frozen in `P2-ADR-001` and `P2-ADR-008`, while accurately
  reflecting the concrete contract decisions frozen in `P2-ADR-009` through
  `P2-ADR-015`.

## Affected interfaces
- Developer-facing documentation for
  `qgd_Variational_Quantum_Eigensolver_Base` and equivalent VQE-facing density
  backend configuration surfaces.
- User-facing usage notes, examples, or workflow references that explain how the
  supported density-matrix path is selected and interpreted.
- Phase 2 support-matrix, bridge, workflow, validation, and reproducibility
  documentation surfaces that define the contract users and reviewers rely on.
- Roadmap-facing and research-alignment documents that mention Phase 2 scope and
  must not overstate current commitments.
- Publication-facing summaries and evidence references that need a stable,
  citable support-surface description for Paper 1.
- Change classification: additive and clarifying for documentation structure and
  references, but stricter for ambiguous or inflated claims, which must become
  explicit scope-boundary statements rather than informal wording.

## Publication relevance
- Supports Paper 1 by giving the project a citable, reviewer-readable contract
  for what the Phase 2 density-matrix backend currently guarantees and what it
  deliberately does not claim.
- Reduces publication risk by separating delivered Phase 2 evidence from
  later-phase roadmap work such as acceleration, broader noisy-VQA integration,
  and trainability analysis.
- Makes abstract, short-paper, and full-paper wording more defensible because
  support labels, workflow scope, and evidence thresholds are documented
  consistently before final writing.
- Supplies the documentation boundary that later phases can extend without
  retroactively changing what Phase 2 meant or overstating its delivered scope.
