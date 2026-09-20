# Task 7 Stories

This document decomposes Phase 3 Task 7 into Layer 3 behavioral stories. These
stories inherit the frozen contract from `TASK_7_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`,
`P3-ADR-006`, `P3-ADR-007`, `P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`.
They describe behavioral slices, not implementation chores.

Story ordering is intentional:

1. freeze one dual-anchor benchmark matrix with stable case identity and an
   explicit representative review set,
2. count only correctness-preserving supported cases as positive benchmark
   evidence,
3. evaluate the positive threshold on one explicit representative structured
   review set against the sequential baseline,
4. keep planner-setting, noise-placement, and workload-identity sensitivity
   visible rather than averaged away,
5. preserve one comparable metric surface for runtime, memory, planning, and
   fused-coverage behavior,
6. keep the diagnosis branch benchmark-grounded and profiler-backed when
   profiling materially affects the conclusion,
7. emit one machine-reviewable benchmark package reusable by Task 8 and paper
   packaging,
8. enforce summary-consistency and bounded-claim guardrails before any later
   publication-facing performance conclusion closes.

## Story 1: The Benchmark Matrix Uses One Frozen Dual-Anchor Case-Identity Surface

**User/Research value**
- Prevents Task 7 from drifting into an ad hoc benchmark zoo by ensuring that
  both the Phase 2 continuity anchor and the structured methods matrix stay
  visible on one shared case-identity surface.

**Given / When / Then**
- Given the frozen Phase 3 workload classes, stable workload IDs,
  deterministic seed rules, and sparse / periodic / dense noise-pattern labels
  defined by the phase contract.
- When Task 7 defines the benchmark matrix and the representative review set it
  will use for performance interpretation.
- Then one benchmark matrix identifies the continuity-anchor slices and the
  representative structured performance slices with stable workload identity,
  explicit review-set membership, and no per-script redefinition of case
  identity.

**Scope**
- In: the dual-anchor benchmark structure, stable workload identity,
  representative structured review-set membership, and continuity-anchor
  visibility.
- Out: numeric threshold closure, bottleneck diagnosis, and final publication
  prose.

**Acceptance signals**
- Reviewable Task 7 evidence can identify every benchmarked case by one stable
  workload tuple and can tell whether that case belongs to the continuity-
  anchor context, the representative structured review set, or another bounded
  benchmark slice.
- Later benchmark and publication consumers do not need to redefine workload
  identity, representative-case selection, or benchmark-slice membership per
  script.

**Traceability**
- Phase requirement(s): Task 7 goal, success-looks-like, and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 7 assumptions and
  dependencies, required behavior, and acceptance evidence in
  `TASK_7_MINI_SPEC.md`; benchmark minimum and benchmark-anchor decisions in
  `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-007`, `P3-ADR-009`

## Story 2: Positive Benchmark Evidence Is Counted Only On Correctness-Preserving Supported Cases

**User/Research value**
- Keeps the Phase 3 performance package scientifically credible by ensuring
  that any counted runtime or memory result is also a Task 6 correctness-
  preserving supported result rather than an unlabeled or fallback outcome.

**Given / When / Then**
- Given Task 7 benchmark cases that reuse the Task 5 calibration surface and the
  Task 6 correctness package.
- When Task 7 decides which cases can count as positive benchmark evidence.
- Then only cases with supported `partitioned_density` execution, explicit Task
  6 counted-status closure, complete provenance, and no silent fallback may
  contribute to a positive benchmark claim.

**Scope**
- In: Task 6 counted-status reuse, correctness-threshold closure, provenance
  completeness, and exclusion of fallback or unlabeled runtime paths.
- Out: the full sensitivity matrix, diagnosis-only bottleneck reporting, and
  publication-facing narrative framing.

**Acceptance signals**
- Every counted supported benchmark case is joinable to one explicit Task 6
  correctness record with stable workload identity, planner-setting reference,
  and runtime-path identity.
- A case that lacks Task 6 correctness closure, required provenance, or the
  supported runtime surface is excluded from positive benchmark evidence.

**Traceability**
- Phase requirement(s): Task 7 required behavior, unsupported behavior, and
  acceptance evidence in `TASK_7_MINI_SPEC.md`; Task 6 required behavior and
  acceptance evidence in `TASK_6_MINI_SPEC.md`; numeric-threshold and
  validation-baseline decisions in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-004`, `P3-ADR-005`, `P3-ADR-008`

## Story 3: The Positive Threshold Uses One Explicit Representative Structured Review Set

**User/Research value**
- Prevents broad or cherry-picked speedup claims by keeping the phase-level
  positive-threshold test tied to one explicit representative structured review
  set judged against the sequential baseline.

**Given / When / Then**
- Given the required representative 8- and 10-qubit structured cases with
  supported `partitioned_density` execution inside the frozen benchmark matrix.
- When Task 7 evaluates whether Phase 3 closes through the measurable-benefit
  path.
- Then the positive threshold is judged only on that explicit representative
  structured review set against the sequential baseline, and any success is
  reported as a bounded result rather than as a universal acceleration claim.

**Scope**
- In: the representative structured review set, comparison against the
  sequential baseline, the `>= 1.2x` speedup or `>= 15%` memory-reduction rule,
  and bounded positive-path interpretation.
- Out: continuity-anchor reproducibility context, optional secondary baselines,
  and the diagnosis explanation for cases that do not meet the threshold.

**Acceptance signals**
- The benchmark package makes the positive-threshold review set explicit and
  records the per-case sequential-baseline comparison needed for threshold
  evaluation.
- If a case is cited as a positive-threshold win, it is identifiable as one of
  the representative structured cases and is not used to imply universal speedup
  across all noisy workloads.

**Traceability**
- Phase requirement(s): Task 7 goal, success-looks-like, and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 7 required behavior and
  acceptance evidence in `TASK_7_MINI_SPEC.md`; performance-claim-boundary
  closure in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- ADR decision(s): `P3-ADR-005`, `P3-ADR-009`, `P3-ADR-010`

## Story 4: Sensitivity Over Planner Settings, Noise Placement, And Workload Identity Stays First-Class

**User/Research value**
- Explains where noise-aware partitioning helps by making sensitivity visible
  across the knobs that define the scientific identity of the Phase 3 methods
  matrix.

**Given / When / Then**
- Given the bounded Task 5 planner-setting surface, the frozen structured
  workload families, and the sparse / periodic / dense local-noise placement
  patterns required by Phase 3.
- When Task 7 records benchmark behavior across the representative benchmark
  matrix.
- Then the package preserves sensitivity across planner-setting choices,
  noise-placement patterns, family, qubit count, and seed-fixed workload
  identity instead of compressing materially different behavior into one average
  claim.

**Scope**
- In: supported planner-setting or partition-size sensitivity inside the bounded
  Task 5 claim surface, noise-placement sensitivity, and workload-identity
  sensitivity across family, qubit count, and seed-fixed instance.
- Out: new support-matrix growth, new planner families outside the bounded Task
  5 claim, and universal extrapolation beyond the frozen workload matrix.

**Acceptance signals**
- Reviewable benchmark evidence shows how runtime, memory, planning-overhead, or
  fused-coverage behavior changes across the required sensitivity dimensions.
- If winners inside the bounded planner-setting family remain close or rerun-
  sensitive, that sensitivity remains explicit instead of being hidden behind
  one permanently frozen winner identity.

**Traceability**
- Phase requirement(s): Task 7 success-looks-like and evidence-required sections
  in `DETAILED_PLANNING_PHASE_3.md`; Task 7 required behavior and acceptance
  evidence in `TASK_7_MINI_SPEC.md`; Paper 2 readiness threshold and metrics
  sections in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-009`

## Story 5: Runtime, Memory, Planning, And Fused-Coverage Metrics Share One Comparable Measurement Surface

**User/Research value**
- Makes the benchmark package auditable by ensuring that runtime, peak memory,
  planning overhead, partition structure, and fused-coverage behavior are
  comparable across cases rather than scattered across incompatible artifact
  formats.

**Given / When / Then**
- Given counted or diagnosis-only benchmark cases on the frozen Phase 3 matrix.
- When Task 7 emits per-case benchmark measurements for review or later summary
  rollups.
- Then each case records one comparable metric surface covering runtime, peak
  memory, planning time, partition context, runtime-path identity,
  fused-coverage behavior, and any repeated-run or median-timing procedure used
  to support the interpretation.

**Scope**
- In: runtime, peak memory, planning time, partition count, qubit span,
  runtime-path classification, fused / supported-but-unfused / deferred
  coverage summaries, and repeated-timing auditability.
- Out: whether a case closes the positive threshold, bottleneck interpretation,
  and publication-facing narrative language.

**Acceptance signals**
- Benchmark records preserve one comparable metric inventory across counted and
  diagnosis-only cases without workload-specific renaming of common fields.
- If Task 7 cites median runtime or median peak-memory behavior, the underlying
  repeated-run evidence or rerunnable median procedure is preserved well enough
  to recompute that interpretation later.

**Traceability**
- Phase requirement(s): benchmark minimum decision, metrics section, and Task 7
  evidence-required section in `DETAILED_PLANNING_PHASE_3.md`; Task 7 required
  behavior, acceptance evidence, and affected interfaces in
  `TASK_7_MINI_SPEC.md`.
- ADR decision(s): `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-009`

## Story 6: The Diagnosis Branch Keeps Bottlenecks Benchmark-Grounded And Profiler-Backed When Material

**User/Research value**
- Allows Phase 3 to close honestly even when the native baseline does not yet
  show a threshold win by requiring explicit bottleneck evidence instead of
  vague "still slow" language.

**Given / When / Then**
- Given representative required cases that preserve correctness but do not meet
  the positive performance threshold.
- When Task 7 closes the phase-level rule through the diagnosis path.
- Then the benchmark package records the dominant bottleneck in benchmark terms,
  includes profiler artifacts when profiling materially affects the follow-on
  architecture conclusion, and keeps deferred follow-on branches explicit
  instead of implying that Phase 3 has an undefined result.

**Scope**
- In: diagnosis-only cases, explicit bottleneck reasons, profiler-backed
  architecture interpretation when material, and mapping to the follow-on
  decision gate.
- Out: implementing channel-native fusion, broad Phase 4 workflow growth, and
  relabeling diagnosis-only cases as positive-threshold wins.

**Acceptance signals**
- Diagnosis-only benchmark evidence records explicit bottleneck or limitation
  reasons for the representative cases that did not meet the positive threshold.
- When profiling materially affects the architecture conclusion, profiler
  artifacts are preserved as part of the diagnosis package rather than left as
  informal notes.

**Traceability**
- Phase requirement(s): Task 7 success-looks-like and evidence-required sections
  in `DETAILED_PLANNING_PHASE_3.md`; Task 7 required behavior, unsupported
  behavior, and acceptance evidence in `TASK_7_MINI_SPEC.md`; Risk 3 and the
  performance-claim-boundary closure in `DETAILED_PLANNING_PHASE_3.md` and
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- ADR decision(s): `P3-ADR-005`, `P3-ADR-008`, `P3-ADR-010`

## Story 7: One Machine-Reviewable Benchmark Package Joins Counted, Diagnosis-Only, And Excluded Evidence

**User/Research value**
- Makes Task 7 reusable by Task 8 and publication packaging by requiring one
  shared benchmark surface instead of disconnected timing notes, profiler files,
  and manual joins across Task 5 and Task 6 artifacts.

**Given / When / Then**
- Given per-case benchmark metrics, Task 5 planner-setting references, Task 6
  correctness joins, diagnosis artifacts where needed, and explicit excluded or
  unsupported cases that define the boundary of the benchmark claim.
- When Task 7 emits artifacts for review or downstream use.
- Then it emits one machine-reviewable benchmark package or rerunnable checker
  that preserves stable provenance, metric fields, counted versus diagnosis-only
  versus excluded status, and join keys across the full Task 7 evidence surface.

**Scope**
- In: one shared benchmark package or checker, stable per-case provenance,
  planner-setting references, correctness-gate references, metric fields,
  diagnosis fields, and excluded-status fields.
- Out: final Paper 2 section prose, narrative framing choices, and new
  support-scope decisions.

**Acceptance signals**
- Later Task 8 and publication consumers can use one stable Task 7 package
  without manual relabeling of workload identity, planner-setting references, or
  counted-status rules.
- Counted, diagnosis-only, and excluded benchmark evidence share one reviewable
  field inventory rather than separate incompatible artifact shapes.

**Traceability**
- Phase requirement(s): Task 7 required behavior, acceptance evidence, affected
  interfaces, and publication relevance in `TASK_7_MINI_SPEC.md`; Task 7
  evidence-required section in `DETAILED_PLANNING_PHASE_3.md`; Task 8 goal and
  evidence-required sections in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-008`, `P3-ADR-009`

## Story 8: Later Summaries Close Claims Only From Counted Representative Per-Case Evidence And Explicit Boundary Language

**User/Research value**
- Prevents overclaiming by ensuring that benchmark tables, sensitivity
  rollups, paper figures, and Task 8 publication packaging derive from one
  auditable counted-status rule and keep limitations plus deferred branches
  visible.

**Given / When / Then**
- Given a completed Task 7 benchmark package with counted, diagnosis-only, and
  excluded status plus stable per-case provenance.
- When Task 7 or Task 8 rolls that package up into summary counts, sensitivity
  summaries, representative-case conclusions, or publication-facing claims.
- Then those summaries are checked against the underlying per-case records,
  positive results remain bounded to the supported review set, diagnosis-only
  results remain labeled as limitations, and deferred branches remain explicit
  rather than hidden inside optimistic wording.

**Scope**
- In: summary-consistency checks, bounded claim language, explicit limitation
  reporting, carry-forward of excluded or unsupported boundary evidence, and
  Task 8 handoff readiness.
- Out: reopening the support matrix, redefining the representative review set,
  and broadening the Phase 3 claim beyond the frozen contract.

**Acceptance signals**
- A summary-consistency rule or checker verifies that rolled-up counts,
  representative-case labels, and sensitivity summaries agree with the
  underlying Task 7 per-case records.
- Excluded, unsupported, deferred, or diagnosis-only cases remain visible as
  claim-boundary evidence instead of disappearing from downstream summaries.

**Traceability**
- Phase requirement(s): Task 7 required behavior, acceptance evidence, and
  publication relevance in `TASK_7_MINI_SPEC.md`; Task 8 goal, success-looks-
  like, and evidence-required sections in `DETAILED_PLANNING_PHASE_3.md`;
  performance-claim-boundary closure in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- ADR decision(s): `P3-ADR-008`, `P3-ADR-009`, `P3-ADR-010`
