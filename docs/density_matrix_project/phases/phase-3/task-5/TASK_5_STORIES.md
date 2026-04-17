# Task 5 Stories

This document decomposes Phase 3 Task 5 into Layer 3 behavioral stories. These
stories inherit the frozen contract from `TASK_5_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`,
`P3-ADR-006`, `P3-ADR-007`, `P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`.
They describe behavioral slices, not implementation chores.

Story ordering is intentional:

1. define one explicit calibration surface with auditable planner candidates and
   settings,
2. anchor calibration to the frozen mandatory workload inventory,
3. show that planner behavior responds to mixed-state cost and explicit noise
   placement rather than to an unchanged state-vector proxy,
4. count only correctness-preserving supported runs as positive calibration
   evidence,
5. define one benchmark-grounded supported planner-claim surface while keeping
   comparison baselines explicit,
6. emit one machine-reviewable calibration bundle reusable by later validation,
   benchmark, and paper work,
7. keep approximation areas, deferred branches, and non-claim exploratory cases
   explicit with no overstatement.

## Story 1: Calibration Uses One Explicit And Auditable Planner-Candidate Surface

**User/Research value**
- Makes the density-aware planning claim scientifically reviewable by ensuring
  that later benchmark and paper work can identify exactly which planner
  candidates, heuristic versions, or setting combinations were calibrated.

**Given / When / Then**
- Given supported Phase 3 planner candidates or planner-setting variants that
  operate on the canonical noisy planner surface inside the frozen support
  matrix.
- When Task 5 evaluates those candidates for the density-aware planning claim.
- Then each candidate is represented through one auditable calibration surface
  that records planner identity, planner settings, and calibration outputs
  rather than through ad hoc tuning notes or hidden per-workload overrides.

**Scope**
- In: planner-candidate identity, planner-setting identity, explicit calibration
  outputs such as scores or rankings, and one shared auditable selection
  surface.
- Out: correctness-threshold verdicts on counted cases, rolled-up benchmark
  interpretation, and post-benchmark architecture decisions.

**Acceptance signals**
- Reviewable calibration evidence can identify which planner candidates or
  setting variants were evaluated and what outputs they produced.
- The supported calibration surface does not depend on hidden workload-specific
  overrides, benchmark-only allowlists, or hand-selected partitions.

**Traceability**
- Phase requirement(s): Task 5 goal, success-looks-like, and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 5 required behavior,
  acceptance evidence, and affected interfaces in `TASK_5_MINI_SPEC.md`; cost
  model decision in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-006`

## Story 2: Calibration Is Anchored To The Frozen Mandatory Workload Inventory

**User/Research value**
- Prevents the Phase 3 density-aware claim from being calibrated only on
  favorable examples by requiring one shared workload matrix that reflects the
  continuity, micro-validation, and structured methods surfaces.

**Given / When / Then**
- Given the frozen mandatory Phase 3 workload inventory with stable workload
  IDs, deterministic seed rules, and noise-pattern labels.
- When Task 5 assembles the calibration matrix for supported planner
  candidates.
- Then the calibration package is anchored to that shared inventory across the
  mandatory continuity, microcase, and structured-workload slices rather than
  to ad hoc exploratory workloads alone.

**Scope**
- In: the required 2 to 4 qubit microcases, the 4, 6, 8, and 10 qubit Phase 2
  noisy XXZ `HEA` continuity cases, the mandatory 8 and 10 qubit structured
  noisy `U3` / `CNOT` families, and stable workload provenance.
- Out: optional broader circuit families, broader gate/noise support, and
  exploratory benchmark branches outside the frozen Phase 3 workload matrix.

**Acceptance signals**
- The calibration matrix records stable workload IDs, seed rules, and
  noise-pattern labels for the mandatory workload classes it uses.
- Calibration evidence remains traceable across continuity, microcase, and
  structured-workload slices without redefining workload identity per planner.

**Traceability**
- Phase requirement(s): Task 5 evidence-required section in
  `DETAILED_PLANNING_PHASE_3.md`; Task 5 assumptions and dependencies,
  required behavior, and acceptance evidence in `TASK_5_MINI_SPEC.md`; workflow
  and benchmark-anchor decision in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-007`, `P3-ADR-009`

## Story 3: Planner Ranking Responds To Mixed-State Cost And Explicit Noise Placement

**User/Research value**
- Gives Paper 2 a real density-aware planning result by showing that planner
  ranking or selection changes for noisy mixed-state reasons rather than merely
  renaming a state-vector cost model.

**Given / When / Then**
- Given supported workloads or supported workload variants whose explicit noise
  placement, mixed-state support cost, or fused-coverage opportunity differs
  inside the frozen Phase 3 matrix.
- When the calibrated planner scores, ranks, or selects among supported
  partitioning alternatives.
- Then the recorded planner behavior changes in a machine-reviewable way that is
  explainable through mixed-state and noise-aware signals rather than through an
  unchanged state-vector FLOP proxy alone.

**Scope**
- In: support size or qubit-span cost, explicit noise placement or density,
  benchmark-observed runtime and peak-memory behavior, planning time, and where
  relevant fused-path coverage as calibration signals.
- Out: claims of global optimality, universal acceleration, or broader support
  beyond the frozen Phase 3 matrix.

**Acceptance signals**
- At least one reproducible comparison slice shows that explicit noise
  placement, mixed-state support cost, or fused-coverage opportunity changes the
  recorded planner score, ranking, or selected configuration.
- Calibration evidence preserves at least one structural-only or
  state-vector-oriented comparison baseline so the density-aware change remains
  visible.

**Traceability**
- Phase requirement(s): Task 5 goal, success-looks-like, and evidence-required
  sections in `DETAILED_PLANNING_PHASE_3.md`; Task 5 required behavior and
  acceptance evidence in `TASK_5_MINI_SPEC.md`; density-aware cost-model
  decision in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-009`

## Story 4: Positive Calibration Evidence Is Counted Only On Correctness-Preserving Supported Runs

**User/Research value**
- Keeps Phase 3 exact-first by ensuring that density-aware calibration is built
  on runs that preserve frozen noisy semantics instead of on planner outcomes
  that only look favorable when correctness is ignored.

**Given / When / Then**
- Given supported calibration runs produced through the documented
  `partitioned_density` runtime surface on mandatory workloads.
- When Task 5 counts those runs as positive evidence for a calibrated planner
  claim.
- Then only runs that preserve the frozen correctness thresholds against the
  sequential density baseline, and the required Aer baseline where applicable,
  contribute to the supported calibration result.

**Scope**
- In: correctness-gated positive evidence, shared runtime-path labeling,
  sequential `NoisyCircuit` comparison, required Aer comparison on the frozen
  external-baseline slices, and exclusion of silent-fallback results.
- Out: the full Task 6 correctness package, unsupported planner-entry policy
  owned by earlier tasks, and final benchmark-rollup interpretation.

**Acceptance signals**
- Counted calibration cases satisfy the frozen internal exactness thresholds and
  use the supported runtime surface with no silent fallback.
- Any microcase or representative small continuity case used as an external
  anchor remains comparable to Qiskit Aer under the frozen exactness rule.

**Traceability**
- Phase requirement(s): Task 5 required behavior and acceptance evidence in
  `TASK_5_MINI_SPEC.md`; Task 5 evidence-required section in
  `DETAILED_PLANNING_PHASE_3.md`; validation-baseline and numeric-threshold
  decisions in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-008`

## Story 5: One Benchmark-Grounded Rule Defines The Supported Phase 3 Planner Claim Surface

**User/Research value**
- Prevents cherry-picking by requiring that the final Phase 3 planner claim come
  from one documented benchmark-grounded selection rule instead of post hoc
  narrative preference.

**Given / When / Then**
- Given multiple supported planner candidates or planner-setting variants
  evaluated on the frozen calibration matrix.
- When Task 5 defines the supported density-aware planner claim for Phase 3.
- Then one documented benchmark-grounded rule selects an auditable supported
  claim surface, keeps comparison baselines explicit, and avoids treating
  exploratory or diagnosis-only cases as part of the supported closure bar even
  if the top-ranked candidate inside the bounded family is close or rerun-
  sensitive.

**Scope**
- In: final supported planner-selection rule, explicit distinction between the
  supported calibrated model and comparison baselines, and one auditable claim
  boundary for later benchmark and paper consumers.
- Out: universal planner superiority claims, hidden manual interpretation, and
  deferred channel-native or broader Phase 4 architecture growth.

**Acceptance signals**
- The supported planner-claim surface is traceable to one documented rule
  applied to benchmark evidence on the frozen workload matrix.
- Comparison baselines and exploratory cases remain visible but are not
  misrepresented as the supported Task 5 claim.

**Traceability**
- Phase requirement(s): Task 5 required behavior, unsupported behavior,
  acceptance evidence, and publication relevance in `TASK_5_MINI_SPEC.md`; Task
  5 success-looks-like and evidence-required sections in
  `DETAILED_PLANNING_PHASE_3.md`; performance-claim-boundary closure in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-010`

## Story 6: Calibration Results And Provenance Are Emitted In One Machine-Reviewable Shared Surface

**User/Research value**
- Makes Task 5 reusable by later Task 6, Task 7, and publication packaging by
  requiring one shared calibration evidence surface instead of planner-specific
  logs or plots that cannot be compared directly.

**Given / When / Then**
- Given supported calibration cases across the frozen workload inventory.
- When Task 5 emits artifacts for review, later validation, or benchmark
  packaging.
- Then it emits one machine-reviewable calibration bundle or rerunnable checker
  that records planner identity, workload provenance, planner settings,
  partition summaries, runtime-path classification, metrics, and calibration
  outputs in a stable shared form.

**Scope**
- In: one calibration manifest or bundle, stable provenance fields, selected-plan
  summaries, runtime, peak memory, planning time, correctness verdicts, and
  calibration outputs such as score or ranking.
- Out: final benchmark narrative, paper prose, and post-benchmark kernel tuning
  recommendations.

**Acceptance signals**
- Later validation or benchmark consumers can read one stable calibration bundle
  without workload-specific parsing rules.
- Calibration artifacts preserve the fields needed to rerun, audit, or compare
  the supported planner claim across workload classes.

**Traceability**
- Phase requirement(s): Task 5 evidence-required section in
  `DETAILED_PLANNING_PHASE_3.md`; Task 5 required behavior, acceptance
  evidence, and affected interfaces in `TASK_5_MINI_SPEC.md`; benchmark minimum
  and reproducibility-artifact decisions in `DETAILED_PLANNING_PHASE_3.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-008`, `P3-ADR-009`

## Story 7: Approximation Areas, Deferred Branches, And Non-Claim Cases Stay Explicit

**User/Research value**
- Keeps the Phase 3 paper narrative honest by ensuring the calibrated planner is
  presented with an explicit approximation boundary instead of being overstated
  as a universal or final solution.

**Given / When / Then**
- Given a benchmark-calibrated planner claim that is still bounded by the frozen
  workload matrix and may remain approximate on some workload classes, noise
  densities, fusion opportunities, or near-tied candidate outcomes.
- When Task 5 summarizes the supported result for later benchmark and paper
  consumers.
- Then approximation areas, diagnosis-only or exploratory cases, and deferred
  follow-on branches remain explicitly documented and are not folded into the
  minimum Task 5 closure claim.

**Scope**
- In: explicit approximation areas, diagnosis-only cases, exploratory cases,
  deferred channel-native fusion, broader Phase 4 workflow growth, and no
  overclaiming language for optimality or generality.
- Out: implementing deferred branches, opening new support-matrix areas, and
  claiming broader noisy workflow parity.

**Acceptance signals**
- The calibration summary records the supported claim boundary plus the main
  approximation areas that remain after calibration.
- If the benchmark-grounded rule yields close or rerun-sensitive winners inside
  the bounded candidate family, that sensitivity is recorded as part of the
  explicit claim boundary rather than hidden behind one permanently frozen
  winner identity.
- Deferred or exploratory branches remain visible future work rather than hidden
  assumptions inside the supported Phase 3 claim.

**Traceability**
- Phase requirement(s): Task 5 unsupported behavior, acceptance evidence, and
  publication relevance in `TASK_5_MINI_SPEC.md`; Task 5 success-looks-like and
  evidence-required sections in `DETAILED_PLANNING_PHASE_3.md`;
  performance-claim-boundary and follow-on branch rule in
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- ADR decision(s): `P3-ADR-006`, `P3-ADR-010`
