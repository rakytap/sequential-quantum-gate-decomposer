# Task 8: Paper 2 Evidence And Documentation Bundle

This mini-spec turns Phase 3 Task 8 into an implementation-ready contract. It
inherits the frozen phase decisions from `DETAILED_PLANNING_PHASE_3.md`,
`P3-ADR-001`, `P3-ADR-002`, `P3-ADR-003`, `P3-ADR-004`, `P3-ADR-005`,
`P3-ADR-006`, `P3-ADR-007`, `P3-ADR-008`, `P3-ADR-009`, and `P3-ADR-010`,
plus the closed canonical planner-surface, semantic-preservation,
runtime-minimum, cost-model sequencing, support-matrix, validation-baseline,
benchmark-anchor, and performance-claim-boundary items in
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. It does not reopen the supported
workload surface, the frozen numeric thresholds, the bounded Task 5 calibrated
planner claim, or the deferred channel-native / broader Phase 4 branches.

## Given / When / Then
- Given the frozen Phase 3 contract, the emitted `phase3_task6` correctness and
  unsupported-boundary bundle family, the emitted `phase3_task7` benchmark and
  diagnosis bundle family, and the current abstract, short-paper, narrative, and
  full-paper draft surfaces for Paper 2.
- When an author, reviewer, or collaborator assembles, audits, or updates the
  Phase 3 publication package across abstract, technical short paper, narrative
  short paper, and full-paper surfaces.
- Then every Paper 2 claim, limitation statement, and follow-on decision must
  remain traceable to mandatory emitted evidence, the supported-path boundary
  must stay explicit, negative boundary evidence must remain visible where it
  defines the claim, and the package must preserve the current
  threshold-or-diagnosis interpretation honestly rather than smoothing it into a
  generic speedup story.

## Assumptions and dependencies
- Task 1 provides the canonical Phase 3 planner-entry contract as a
  schema-versioned `CanonicalNoisyPlannerSurface` with stable provenance fields
  for `requested_mode`, `source_type`, `entry_route`, `workload_family`, and
  `workload_id`, plus explicit planner-entry unsupported evidence.
- Task 2 provides the schema-versioned partition handoff contract through
  `NoisyPartitionDescriptorSet`, `NoisyPartitionDescriptor`, and
  `NoisyPartitionDescriptorMember`, including canonical operation references,
  exact gate/noise order, remapping metadata, and parameter-routing metadata.
- Task 3 provides the executable partitioned density runtime surface and stable
  runtime-path labels that Paper 2 must describe as delivered behavior rather
  than as planned future work.
- Task 4 provides the minimum real fused-execution baseline and explicit fused,
  supported-but-unfused, and deferred-region classifications. Task 8 may cite
  that real fused path, but it must preserve the delivered limitation boundary
  rather than implying full channel-native noisy-block fusion.
- Task 5 provides the benchmark-calibrated planning result on the current
  supported bounded candidate family. Paper 2 must phrase the planner claim as a
  benchmark-grounded selection rule over auditable `max_partition_qubits`
  span-budget settings, not as a permanently settled broad planner-family win.
- Task 6 provides the machine-reviewable correctness and unsupported-boundary
  package under `benchmarks/density_matrix/artifacts/phase3_task6/`. The current
  delivered surface records `25` counted supported cases, `4` required external
  reference cases, and `17` explicit unsupported-boundary cases across
  planner-entry, descriptor-generation, and runtime-stage evidence.
- Task 6 also provides stable bundle entry points that Task 8 must package
  rather than paraphrase loosely, including:
  - `story7_correctness_package/correctness_package_bundle.json`,
  - `story6_unsupported_boundary/unsupported_boundary_bundle.json`,
  - and `story8_summary_consistency/summary_consistency_bundle.json`.
- Task 7 provides the machine-reviewable benchmark and diagnosis package under
  `benchmarks/density_matrix/artifacts/phase3_task7/`. The current delivered
  surface records `34` counted supported benchmark cases, `6` representative
  review cases, diagnosis-grounded closure of the phase-level performance rule,
  and carry-forward of the `17` explicit Task 6 boundary cases into the summary
  layer.
- Task 7 also provides stable bundle entry points that Task 8 must package
  rather than summarize ambiguously, including:
  - `story7_benchmark_package/benchmark_package_bundle.json`,
  - `story6_diagnosis_path/diagnosis_bundle.json`,
  - `story4_sensitivity_matrix/sensitivity_matrix_bundle.json`,
  - and `story8_summary_consistency/summary_consistency_bundle.json`.
- `docs/density_matrix_project/planning/PUBLICATIONS.md` defines Paper 2 as the
  major Phase 3 methods and systems paper, not as a broader optimizer, workflow,
  or approximate-scaling paper.
- Publication-facing docs such as `ABSTRACT_PHASE_3.md`,
  `SHORT_PAPER_PHASE_3.md`, `SHORT_PAPER_NARRATIVE.md`, and
  `PAPER_PHASE_3.md` are packaging surfaces for the frozen contract and emitted
  evidence. They must not outrun the delivered evidence, reopen closed phase
  decisions, or broaden the scientific claim set.
- This task defines how Phase 3 evidence is packaged, traced, and narrated for
  publication. It does not require new backend features, new benchmark classes,
  new correctness thresholds, or new scientific claims beyond the frozen Phase 3
  contract.

## Required behavior
- Task 8 freezes the Paper 2 publication package for Phase 3 as the
  authoritative mapping from delivered Phase 3 evidence to paper-level claims,
  limitation statements, section-level evidence references, and reviewer-entry
  surfaces.
- Paper 2 must have one stable main-claim boundary: SQUANDER extends
  partitioning and limited fusion to exact noisy mixed-state circuits by making
  noisy operations first-class planner inputs, preserving exact gate/noise
  semantics across partition descriptors and runtime execution, and validating
  the resulting partitioned density path on representative noisy workloads.
- The publication package must preserve the frozen supporting-claim boundary:
  - Task 1 and Task 2 close the canonical noisy planner and semantic-
    preservation contract,
  - Task 3 and Task 4 close the executable partitioned runtime plus at least one
    real fused execution path,
  - Task 5 closes the bounded benchmark-calibrated planning rule,
  - Task 6 closes the correctness and unsupported-boundary package,
  - and Task 7 closes the representative benchmark and diagnosis package.
- The publication package must define explicit non-claims for Paper 2 so later
  work is not pulled into the current contribution boundary. Fully
  channel-native or superoperator-native fused noisy blocks, broader noisy
  VQE/VQA workflow growth, density-backend gradients, optimizer-comparison
  studies, approximate scaling methods, and full direct `qgd_Circuit` parity
  remain future work rather than current Paper 2 results.
- Abstract, technical short-paper, narrative short-paper, and full-paper
  surfaces must describe the same frozen Phase 3 result at different levels of
  detail rather than introduce conflicting claims, scales, thresholds, workload
  definitions, or support labels.
- The narrative short-paper surface may emphasize motivation, PhD positioning,
  and scientific arc, but it must not introduce stronger technical claims than
  the technical short-paper or full-paper surfaces can trace to emitted
  evidence.
- The publication package must preserve the frozen Paper 2 evidence-closure
  rule: only mandatory, complete, supported correctness and reproducibility
  evidence, plus either measured benefit or benchmark-grounded limitation
  reporting, closes the main claim. Optional, exploratory, deferred, unsupported,
  or incomplete material may provide context, but it must not replace the
  required evidence surface.
- The publication package must preserve the frozen supported-path boundary: the
  guaranteed Paper 2 path is the canonical noisy mixed-state planner surface
  plus the documented exact lowering needed for the frozen Phase 2 continuity
  workflow and the required Phase 3 structured benchmark families.
- The publication package must preserve the no-fallback rule. No benchmark or
  paper statement may imply that silent sequential substitution is an accepted
  part of `partitioned_density` evidence.
- Supported positive evidence and negative boundary evidence must both appear
  where they define the claim boundary. At minimum, Paper 2 packaging must keep
  planner-entry unsupported cases, descriptor-generation unsupported or lossy
  cases, and runtime-stage unsupported or deferred cases visible instead of
  collapsing them into one generic omission bucket.
- The package must remain manifest-driven over emitted bundles or rerunnable
  checkers rather than prose-only packaging. Reviewer-facing Paper 2 claims must
  be traceable to concrete `phase3_task6` and `phase3_task7` bundle references
  rather than only to narrative summaries or detached figures.
- The publication package must include a stable reviewer-entry surface for Paper
  2, such as one top-level publication manifest or equivalent machine-reviewable
  checker, that identifies:
  - the current main claim and explicit non-claims,
  - the required positive and negative evidence bundle references,
  - the section-level or equivalent claim-to-evidence mapping,
  - the supported-path boundary and no-fallback rule,
  - and the current benchmark interpretation path.
- If the current benchmark layer closes the performance rule through the
  diagnosis branch rather than the positive-threshold branch, Task 8 must state
  that outcome explicitly and map it to the emitted Task 7 diagnosis evidence
  instead of polishing it into an implied speedup claim.
- The package must preserve the current implementation-backed benchmark wording
  accurately when those counts are cited:
  - Task 6 currently contributes `25` counted supported cases,
  - `4` required external-reference cases,
  - and `17` explicit unsupported-boundary cases,
  - while Task 7 currently contributes `34` counted supported benchmark cases,
  - `6` representative review cases,
  - and diagnosis-grounded rather than positive-threshold benchmark closure on
    the representative review set.
- The current dominant limitation surface must remain explicit when cited:
  representative review cases preserve exact semantics and exercise a real fused
  path, but they remain slower than the sequential reference, do not reduce peak
  memory, and point primarily to supported islands left unfused plus
  Python-level fused-path overhead.
- Publication terminology must remain consistent across abstract, short-paper,
  narrative, full-paper, and reviewer-entry surfaces for terms such as exact
  noisy mixed-state circuits, canonical noisy planner surface, partitioned
  density runtime, real fused path, counted supported, diagnosis-grounded
  closure, required / optional / deferred / unsupported, and reproducibility
  bundle.
- Section-level or equivalent publication traceability must exist from major
  Paper 2 claims into the underlying contract and evidence surfaces so a
  reviewer can determine which artifact supports each contribution claim,
  benchmark statement, limitation statement, and future-work boundary.
- Reviewer-facing entry points must remain stable enough that a paper reader can
  find the authoritative Phase 3 contract, the current emitted correctness and
  benchmark bundles, and the deferred-scope boundary without reverse-
  engineering the repository.
- Task 8 completion means Paper 2 is claim-ready, evidence-traceable, and
  reviewable as an honest Phase 3 methods milestone. It does not require venue-
  specific camera-ready polishing, new backend implementation, or new evidence
  outside the frozen contract.

## Unsupported behavior
- Drafting Paper 2 in a way that implies Phase 3 already delivers fully
  channel-native fused noisy blocks, broader noisy VQE/VQA workflow growth,
  density-backend gradients, optimizer studies, approximate scaling, or full
  direct `qgd_Circuit` parity.
- Presenting the current diagnosis-grounded Task 7 benchmark result as if it
  were already a measured speedup or peak-memory win on the representative
  review cases.
- Allowing abstract, technical short-paper, narrative short-paper, and full-
  paper variants to disagree about the main claim, supported-path boundary,
  benchmark floor, thresholds, case counts, non-claims, or performance
  interpretation.
- Treating the bounded Task 5 planning result as proof that a broad noisy
  planner-family competition has already been permanently settled.
- Building the Paper 2 package from detached plots, prose-only summaries, or
  manually edited counts without stable traceability back to the emitted
  `phase3_task6` and `phase3_task7` bundle surfaces.
- Hiding the `17` explicit Task 6 unsupported-boundary cases or collapsing
  planner-entry, descriptor-generation, and runtime-stage exclusions into one
  ambiguous publication bucket.
- Reopening phase-frozen decisions about planner semantics, support matrix,
  benchmark anchors, numeric thresholds, no-fallback behavior, or deferred
  branches inside paper-facing docs.
- Treating optional baselines, exploratory workload extensions, or favorable but
  incomplete slices as sufficient closure of the main Paper 2 claim.
- Implying support for broader circuit-source parity or broader workflow reuse
  than the documented exact lowering path and required benchmark families
  actually provide.
- Treating formatting polish alone as Task 8 completion if the claim set,
  section-level evidence mapping, supported-path boundary, and limitation
  language remain ambiguous.

## Acceptance evidence
- An abstract-level claim set identifies the one-sentence Paper 2 main claim,
  the required supporting claims, the explicit non-claims, and the
  threshold-or-diagnosis interpretation rule, and it aligns with
  `docs/density_matrix_project/planning/PUBLICATIONS.md`.
- A technical short-paper structure identifies the required sections for the
  compact Paper 2 methods narrative and preserves the frozen contribution
  boundary, workload surface, evidence bar, and honest limitation reporting.
- A narrative short-paper structure identifies the required motivation,
  positioning, and research-arc sections for a general PhD-conference audience
  without outrunning the delivered evidence or changing the technical claim.
- A full-paper structure identifies the required sections for the long-form
  Paper 2 narrative and preserves the same claim boundary, evidence bar, and
  deferred-scope language as the abstract and short-paper surfaces.
- One stable reviewer-entry manifest or equivalent machine-reviewable Paper 2
  checker exists and references, at minimum:
  - the current Task 6 correctness package bundle,
  - the current Task 6 unsupported-boundary bundle,
  - the current Task 6 summary-consistency bundle,
  - the current Task 7 benchmark package bundle,
  - the current Task 7 diagnosis bundle,
  - the current Task 7 sensitivity matrix bundle,
  - and the current Task 7 summary-consistency bundle.
- The reviewer-entry surface records section-level or equivalent mappings from
  major Paper 2 claims into the relevant emitted bundles, phase contract docs,
  and task mini-specs instead of relying on prose-only citations.
- Publication-facing docs consistently describe:
  - the canonical noisy mixed-state planner surface,
  - exact gate/noise-order preservation,
  - the executable partitioned runtime plus real fused path,
  - the bounded Task 5 planning claim,
  - the Task 6 two-baseline correctness package,
  - and the Task 7 representative benchmark plus diagnosis package,
  without contradicting Phase 3 contract docs.
- If current implementation-backed counts are cited in the publication package,
  they remain joinable to emitted bundle surfaces and are reported accurately:
  - `25` counted supported Task 6 cases,
  - `4` required external Task 6 cases,
  - `17` explicit Task 6 boundary cases,
  - `34` counted supported Task 7 benchmark cases,
  - and `6` representative review cases with diagnosis-grounded closure.
- The publication package states the honest current performance interpretation
  explicitly: no representative review case currently meets the positive
  threshold, and the emitted diagnosis surface attributes the current bottleneck
  primarily to supported islands left unfused plus Python-level fused-path
  overhead.
- Negative review evidence shows that fully channel-native fusion, broader noisy
  VQE/VQA growth, density-backend gradients, broader planner-family closure, and
  approximate scaling are explicitly described as future work rather than as
  current Paper 2 results.
- A reviewer can follow at least one stable documentation path from the Paper 2
  package to the authoritative Phase 3 contract, the current emitted correctness
  and benchmark bundles, and the current limitation boundary without relying on
  source-code inspection.
- Traceability exists from major Paper 2 claims and sections to the underlying
  Phase 3 sources: `DETAILED_PLANNING_PHASE_3.md`, `ADRs_PHASE_3.md`,
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`, `TASK_1_MINI_SPEC.md` through
  `TASK_7_MINI_SPEC.md`, `ABSTRACT_PHASE_3.md`, `SHORT_PAPER_PHASE_3.md`,
  `SHORT_PAPER_NARRATIVE.md`, and `PAPER_PHASE_3.md`.
- Traceability target: satisfy the Phase 3 Task 8 evidence requirements in
  `DETAILED_PLANNING_PHASE_3.md`.
- Traceability target: support the full-phase acceptance criteria requiring the
  Paper 2 evidence bundle to document both the achieved backend result and the
  remaining limitations honestly.
- Traceability target: satisfy the publication-contract rule in `P3-ADR-001`,
  the Phase 3 scope-boundary rule in `P3-ADR-002`, the canonical planner-surface
  and semantic-preservation dependencies in `P3-ADR-003` and `P3-ADR-004`, the
  executable runtime and real-fused baseline in `P3-ADR-005`, the bounded
  benchmark-calibrated planning boundary in `P3-ADR-006`, the frozen support
  matrix in `P3-ADR-007`, the two-baseline validation rule in `P3-ADR-008`, the
  benchmark-anchor and manifest-driven publication rule in `P3-ADR-009`, and the
  deferred follow-on boundary in `P3-ADR-010`.

## Affected interfaces
- Publication-facing paper surfaces rooted in `ABSTRACT_PHASE_3.md`,
  `SHORT_PAPER_PHASE_3.md`, `SHORT_PAPER_NARRATIVE.md`, and `PAPER_PHASE_3.md`.
- A stable Task 8 reviewer-entry surface, such as a top-level publication
  manifest or equivalent machine-reviewable checker under
  `benchmarks/density_matrix/artifacts/phase3_task8/`.
- The emitted Task 6 bundle family under
  `benchmarks/density_matrix/artifacts/phase3_task6/`, especially the
  correctness-package, unsupported-boundary, and summary-consistency entry
  points.
- The emitted Task 7 bundle family under
  `benchmarks/density_matrix/artifacts/phase3_task7/`, especially the benchmark-
  package, diagnosis, sensitivity-matrix, and summary-consistency entry points.
- Phase-level contract docs and decisions that the Paper 2 package must cite
  accurately: `DETAILED_PLANNING_PHASE_3.md`, `ADRs_PHASE_3.md`, and
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.
- Task-level contract docs, especially `TASK_1_MINI_SPEC.md` through
  `TASK_7_MINI_SPEC.md`, that provide the paper's claim, benchmark, correctness,
  support-boundary, and terminology references.
- Publication-facing claim tables, reviewer-entry maps, reproducibility
  manifests, figure provenance notes, and limitation summaries that connect the
  delivered backend result to reviewer-readable paper structure.
- Change classification: additive and clarifying for publication packaging,
  reviewer-entry, and traceability surfaces, but stricter for claim language
  because every stronger statement must remain backed by emitted counted
  supported evidence and explicit limitation handling.

## Publication relevance
- Directly defines the final Paper 2 claim boundary and evidence package so the
  main Phase 3 publication can be drafted without inflating scope or hiding the
  current benchmark outcome.
- Reduces review risk by making the paper package traceable to emitted Task 6
  and Task 7 artifacts instead of relying on prose-only confidence.
- Keeps Paper 2 aligned with the density-matrix publication ladder by presenting
  Phase 3 as the noise-aware partitioning and limited-fusion methods milestone,
  not as a premature optimizer or approximate-scaling paper.
- Preserves scientific honesty by making diagnosis-grounded closure publishable
  when the representative benchmark package does not yet show positive-threshold
  acceleration.
- Creates a reusable publication handoff that later phases can extend without
  retroactively changing what the Phase 3 result meant.
