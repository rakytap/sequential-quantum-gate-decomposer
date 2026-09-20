# Task 6 Stories

This document decomposes Phase 3 Task 6 into Layer 3 behavioral stories. These
stories inherit the frozen contract from `TASK_6_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`,
`P3-ADR-006`, `P3-ADR-007`, `P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`.
They describe behavioral slices, not implementation chores.

Story ordering is intentional:

1. freeze one correctness matrix with stable case identity and explicit
   internal-only versus internal-plus-external coverage,
2. gate every counted supported case on exact agreement with the sequential
   `NoisyCircuit` baseline,
3. keep the mandatory Qiskit Aer validation slice explicit and bounded,
4. preserve density-validity and continuity-anchor energy agreement as
   first-class parts of the correctness verdict,
5. keep fused, supported-but-unfused, and deferred runtime paths correctness-
   comparable with no silent relabeling,
6. keep planner-entry, descriptor-generation, and runtime-stage unsupported
   boundaries explicitly separated,
7. emit one machine-reviewable correctness package reusable by Task 7, Task 8,
   and publication packaging,
8. enforce summary-consistency and counted-status guardrails before any later
   benchmark or publication claim closes.

## Story 1: The Correctness Matrix Uses One Frozen Case-Identity Surface

**User/Research value**
- Prevents Task 6 from drifting into ad hoc validation slices by ensuring that
  all counted correctness evidence is anchored to one shared matrix with stable
  workload identity and explicit slice membership.

**Given / When / Then**
- Given the frozen Phase 3 workload classes, stable workload IDs,
  deterministic seed rules, and noise-pattern labels defined by the phase
  contract.
- When Task 6 defines which cases belong to the required correctness package.
- Then one correctness matrix identifies the mandatory micro-validation,
  continuity, and structured-family cases and marks which cases require only
  internal validation versus both internal and external validation.

**Scope**
- In: stable workload identity, slice membership, continuity / microcase /
  structured-family coverage, and joinability with Task 5 calibration outputs.
- Out: numeric pass verdicts, unsupported-boundary taxonomy, and rolled-up
  benchmark interpretation.

**Acceptance signals**
- Reviewable Task 6 evidence can identify every mandatory counted case by one
  stable workload tuple and can tell whether that case belongs to the internal-
  only or internal-plus-external slice.
- Later validation, benchmark, and publication consumers do not need to
  redefine workload identity or slice membership per script.

**Traceability**
- Phase requirement(s): Task 6 goal, success-looks-like, and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 6 assumptions and
  dependencies, required behavior, and acceptance evidence in
  `TASK_6_MINI_SPEC.md`; benchmark-anchor and validation-matrix sections in
  `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-007`, `P3-ADR-008`, `P3-ADR-009`

## Story 2: Every Counted Supported Case Is Gated By Sequential-Baseline Exactness

**User/Research value**
- Keeps Phase 3 exact-first by ensuring that any counted runtime, fusion, or
  planner-related success is also a semantic-preservation success relative to
  the delivered SQUANDER density baseline.

**Given / When / Then**
- Given supported `partitioned_density` runs on mandatory Phase 3 cases.
- When Task 6 decides whether a case can count as positive correctness
  evidence.
- Then the case is counted only if it carries an explicit sequential
  `NoisyCircuit` comparison verdict and satisfies the frozen internal exactness
  thresholds with no silent fallback.

**Scope**
- In: sequential-baseline comparison, Frobenius agreement, trace preservation,
  density validity, supported runtime-path identity, and exclusion of hidden
  fallback.
- Out: the required external Aer slice, summary rollups, and performance
  conclusions.

**Acceptance signals**
- Every counted supported case records raw internal comparison metrics and an
  explicit pass/fail verdict against the frozen thresholds.
- A case with silent fallback, unlabeled runtime path, or missing sequential
  verdict is excluded from positive Task 6 evidence.

**Traceability**
- Phase requirement(s): Task 6 success-looks-like and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 6 required behavior,
  unsupported behavior, and acceptance evidence in `TASK_6_MINI_SPEC.md`;
  full-phase acceptance criteria and numeric-threshold decision in
  `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-008`

## Story 3: The Mandatory Qiskit Aer Slice Remains Explicit And Bounded

**User/Research value**
- Gives Paper 2 a credible external exactness story without turning Task 6 into
  a broad simulator bake-off or letting ad hoc external comparisons redefine
  the required scope.

**Given / When / Then**
- Given the required 2 to 4 qubit microcases and the representative small
  continuity subset frozen by the Phase 3 contract.
- When Task 6 records external correctness evidence.
- Then those cases are compared against Qiskit Aer under one explicit external
  slice with stable provenance and threshold rules, and any additional
  comparison cases remain clearly secondary rather than redefining the required
  slice.

**Scope**
- In: Qiskit Aer agreement on the required external slice, stable external-case
  provenance, and bounded external-baseline scope.
- Out: optional secondary simulator baselines, broad framework parity claims,
  and performance benchmarking.

**Acceptance signals**
- The required external slice records auditable Qiskit Aer comparison outputs
  and exactness verdicts for the frozen cases.
- External validation remains identifiable as one bounded slice rather than a
  mixture of ad hoc comparison cases.

**Traceability**
- Phase requirement(s): Task 6 evidence-required section in
  `DETAILED_PLANNING_PHASE_3.md`; Task 6 required behavior and acceptance
  evidence in `TASK_6_MINI_SPEC.md`; primary external-baseline and validation-
  target sections in `PAPER_PHASE_3.md`.
- ADR decision(s): `P3-ADR-008`, `P3-ADR-009`

## Story 4: Density Validity And Continuity-Anchor Energy Agreement Stay First-Class

**User/Research value**
- Prevents Task 6 from reducing correctness to one density-difference number by
  keeping state-validity and workflow-anchor observable agreement visible in the
  counted evidence.

**Given / When / Then**
- Given recorded Task 6 outputs for mandatory correctness cases, including the
  frozen 4 / 6 / 8 / 10 qubit continuity-anchor cases.
- When Task 6 interprets whether those outputs satisfy the required correctness
  contract.
- Then density validity metrics and continuity-anchor energy agreement remain
  explicit parts of the verdict rather than hidden derived checks.

**Scope**
- In: `|Tr(rho) - 1|`, `rho.is_valid(tol=1e-10)`, and maximum absolute energy
  error on the continuity-anchor cases.
- Out: broader observable expansion, workflow-growth decisions, and optional
  paper-level interpretation.

**Acceptance signals**
- Counted supported outputs preserve the frozen trace-validity and density-
  validity checks in the emitted correctness evidence.
- The required 4 / 6 / 8 / 10 qubit continuity-anchor cases record explicit
  energy-agreement verdicts against the frozen `<= 1e-8` threshold.

**Traceability**
- Phase requirement(s): numeric-threshold decision, full-phase acceptance
  criteria, and Task 6 evidence-required section in
  `DETAILED_PLANNING_PHASE_3.md`; Task 6 required behavior and acceptance
  evidence in `TASK_6_MINI_SPEC.md`.
- ADR decision(s): `P3-ADR-004`, `P3-ADR-008`, `P3-ADR-009`

## Story 5: Runtime And Fusion Classifications Stay Correctness-Comparable

**User/Research value**
- Keeps Task 6 honest about the real execution path by ensuring that fused
  evidence, supported-but-unfused evidence, and deferred runtime outcomes remain
  comparable without mislabeling one path as another.

**Given / When / Then**
- Given supported cases that may execute through plain partitioned runtime,
  real fused runtime, or supported-but-unfused runtime paths, plus deferred
  fusion situations exposed by earlier tasks.
- When Task 6 records correctness evidence for those cases.
- Then fused-path cases satisfy the same correctness thresholds as other
  supported cases, and runtime-path plus fusion classification remain explicit
  enough that no path is silently relabeled as a different kind of success.

**Scope**
- In: runtime-path classification, fused versus supported-but-unfused versus
  deferred labeling, and direct correctness comparability across those paths.
- Out: performance-benefit interpretation, channel-native fusion, and benchmark
  threshold closure owned by later tasks.

**Acceptance signals**
- Positive fused-path evidence can be compared directly against the same Task 6
  thresholds used for unfused supported cases.
- Supported-but-unfused or deferred fusion situations remain visible in emitted
  evidence and are not misreported as fused correctness success.

**Traceability**
- Phase requirement(s): Task 6 required behavior, unsupported behavior,
  acceptance evidence, and affected interfaces in `TASK_6_MINI_SPEC.md`; Task 4
  acceptance evidence and affected interfaces in `TASK_4_MINI_SPEC.md`.
- ADR decision(s): `P3-ADR-005`, `P3-ADR-008`, `P3-ADR-010`

## Story 6: Unsupported-Boundary Evidence Preserves Stage Separation

**User/Research value**
- Makes negative evidence publishable and reviewable by preserving the
  difference between planner-entry, descriptor-generation, and runtime-stage
  unsupported or deferred behavior.

**Given / When / Then**
- Given negative or excluded cases arising before planning, during descriptor
  generation, or at runtime.
- When Task 6 emits unsupported-boundary evidence as part of the correctness
  package.
- Then the evidence keeps failure stage, unsupported category, first
  unsupported condition, provenance, and exclusion reason explicit instead of
  collapsing all negative outcomes into one generic failure bucket.

**Scope**
- In: planner-entry unsupported cases, descriptor-generation unsupported or
  lossy cases, runtime-stage unsupported or deferred cases, and one stable
  machine-reviewable taxonomy across them.
- Out: widening the support matrix, hiding negative evidence, and summary-only
  exclusion logic with no underlying records.

**Acceptance signals**
- Unsupported-boundary artifacts cover all three stage categories with one
  reviewable failure vocabulary.
- Later benchmark or publication layers can identify why a case was excluded
  without inferring the answer from logs or ad hoc script logic.

**Traceability**
- Phase requirement(s): Task 6 goal, success-looks-like, and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 6 required behavior,
  unsupported behavior, and acceptance evidence in `TASK_6_MINI_SPEC.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-008`

## Story 7: One Machine-Reviewable Correctness Package Joins Positive And Negative Evidence

**User/Research value**
- Makes Task 6 reusable by later Task 7, Task 8, and publication packaging by
  requiring one shared correctness surface instead of disconnected validation
  logs, unsupported-case notes, and manual spreadsheet joins.

**Given / When / Then**
- Given per-case internal comparisons, external comparisons where required,
  continuity-anchor verdicts, runtime-path labels, and unsupported-boundary
  records.
- When Task 6 emits artifacts for review or downstream consumption.
- Then it emits one machine-reviewable correctness package or rerunnable
  checker that preserves stable provenance, verdict fields, and join keys across
  both counted supported evidence and excluded negative evidence.

**Scope**
- In: shared correctness package or checker, stable per-case provenance,
  planner-setting references, runtime-path and fusion labels, verdict fields,
  exclusion fields, and compatibility with Task 5 calibration outputs.
- Out: final benchmark narrative, paper prose, and performance threshold
  interpretation.

**Acceptance signals**
- Later Task 7 and Task 8 consumers can use one stable correctness package
  without manual relabeling of workload identity, planner-setting references, or
  counted-status rules.
- Positive and negative Task 6 evidence share one reviewable field inventory
  rather than separate incompatible artifact shapes.

**Traceability**
- Phase requirement(s): Task 6 required behavior, acceptance evidence, affected
  interfaces, and publication relevance in `TASK_6_MINI_SPEC.md`; Task 6
  evidence-required section in `DETAILED_PLANNING_PHASE_3.md`; publication-
  bundle expectations in `PAPER_PHASE_3.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-008`, `P3-ADR-009`

## Story 8: Later Summaries Close Claims Only From Counted Supported Per-Case Evidence

**User/Research value**
- Prevents overclaiming by ensuring that benchmark summaries, paper tables, and
  publication-facing figures are derived from one auditable counted-status rule
  rather than from hand-filtered or summary-only interpretations.

**Given / When / Then**
- Given a completed Task 6 correctness package with explicit counted versus
  excluded case status and stable per-case provenance.
- When later Task 7 benchmark rollups or Task 8 publication bundles summarize
  Phase 3 correctness evidence.
- Then those summaries are checked against the underlying per-case records, and
  only complete counted supported evidence is allowed to close a benchmark or
  publication claim.

**Scope**
- In: summary-consistency checks, counted versus excluded status, provenance-
  completeness, and explicit carry-forward of unsupported-boundary evidence.
- Out: performance-threshold satisfaction itself, narrative framing choices, and
  new support-scope decisions.

**Acceptance signals**
- A summary-consistency rule or checker verifies that rolled-up counts and
  status tables agree with the underlying Task 6 per-case records.
- Excluded, unsupported, or deferred cases remain visible as claim-boundary
  evidence instead of disappearing from downstream summaries.

**Traceability**
- Phase requirement(s): Task 6 required behavior, acceptance evidence, and
  publication relevance in `TASK_6_MINI_SPEC.md`; Task 7 goal and success-
  looks-like sections plus Task 8 goal and evidence-required sections in
  `DETAILED_PLANNING_PHASE_3.md`; performance-claim-boundary closure in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- ADR decision(s): `P3-ADR-008`, `P3-ADR-009`, `P3-ADR-010`
