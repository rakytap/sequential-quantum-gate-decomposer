# Task 8 Stories

This document decomposes Phase 3 Task 8 into Layer 3 behavioral stories.
These stories inherit the frozen contract from `TASK_8_MINI_SPEC.md`,
`DETAILED_PLANNING_PHASE_3.md`, `P3-ADR-001`, `P3-ADR-002`, `P3-ADR-003`,
`P3-ADR-004`, `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-007`, `P3-ADR-008`,
`P3-ADR-009`, and `P3-ADR-010`. They describe behavioral slices, not
implementation chores.

Story ordering is intentional:

1. freeze one honest Paper 2 main claim with explicit non-claims,
2. keep abstract, technical short-paper, narrative short-paper, and full-paper
   surfaces aligned while preserving their distinct roles,
3. make major paper claims and sections traceable to authoritative Phase 3
   contract docs and emitted Task 6 / Task 7 bundles,
4. preserve the mandatory evidence floor and threshold-or-diagnosis closure rule
   as the only basis for the main paper claim,
5. communicate the supported path, no-fallback rule, bounded Task 5 planner
   claim, and current benchmark surface honestly,
6. keep positive supported evidence, negative boundary evidence, and
   diagnosis-grounded limitation reporting visible in one manifest-driven
   publication package,
7. keep future-work boundaries and publication-ladder positioning explicit,
8. preserve terminology consistency, reviewer entry, and summary consistency
   across the full Paper 2 package.

## Story 1: Paper 2 Has One Stable Main Claim, Supporting Claim Set, And Explicit Non-Claims

**User/Research value**
- This story protects scientific credibility by freezing one honest Phase 3
  contribution boundary instead of letting Paper 2 drift into a broader roadmap
  or optimizer-facing claim.

**Given / When / Then**
- Given the frozen Phase 3 methods contract and the Paper 2 positioning in
  `PUBLICATIONS.md`.
- When a reviewer asks what Paper 2 actually claims and what it explicitly does
  not claim.
- Then the paper package presents one stable main claim, one bounded supporting
  claim set, and explicit non-claims that prevent later-phase work from being
  read as already delivered.

**Scope**
- In: Paper 2 main claim, supporting claim set, explicit non-claims, and the
  publication contribution boundary.
- Out: adding new scientific results, expanding the frozen support surface, or
  reopening phase-level scope decisions.

**Acceptance signals**
- An abstract-level claim set identifies one stable Paper 2 main claim centered
  on noise-aware partitioning and limited fusion for exact noisy mixed-state
  circuits.
- Supporting claims stay limited to the canonical noisy planner surface, exact
  semantic-preservation rules, the executable partitioned runtime plus at least
  one real fused path, the bounded benchmark-calibrated Task 5 planner result,
  and the machine-reviewable Task 6 plus Task 7 evidence package.
- Explicit non-claims keep fully channel-native fused noisy blocks, broader
  noisy VQE/VQA workflow growth, density-backend gradients, optimizer studies,
  approximate scaling, and full direct `qgd_Circuit` parity outside the current
  Paper 2 result.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Task 8 goal, success
  looks like, and evidence required; `TASK_8_MINI_SPEC.md` required behavior and
  acceptance evidence.
- ADR decision(s): `P3-ADR-001`, `P3-ADR-002`, `P3-ADR-009`, `P3-ADR-010`

## Story 2: Abstract, Technical Short-Paper, Narrative Short-Paper, And Full-Paper Surfaces Tell The Same Phase 3 Story At Different Depths

**User/Research value**
- This story lets different review audiences see the same scientific result
  without forcing every publication surface to carry the same level of detail
  or the same narrative emphasis.

**Given / When / Then**
- Given one frozen Phase 3 result and four publication-facing surfaces with
  different depth and audience expectations.
- When authors or reviewers move between the abstract, the technical short
  paper, the narrative short paper, and the full-paper draft.
- Then they find the same claim boundary, evidence bar, and limitation
  structure, expressed at audience-appropriate depth rather than as conflicting
  narratives.

**Scope**
- In: alignment across abstract, technical short-paper, narrative short-paper,
  and full-paper surfaces; audience-appropriate compression; and distinct
  technical versus narrative emphasis.
- Out: venue-specific formatting polish, bibliography expansion, or adding new
  evidence to support a broader claim.

**Acceptance signals**
- `ABSTRACT_PHASE_3.md`, `SHORT_PAPER_PHASE_3.md`, `SHORT_PAPER_NARRATIVE.md`,
  and `PAPER_PHASE_3.md` do not conflict about the main claim, supported-path
  boundary, non-claims, evidence-closure rule, or current benchmark
  interpretation.
- The technical short-paper surface emphasizes methods, validation thresholds,
  and benchmark interpretation, while the narrative short-paper surface
  emphasizes motivation, scientific positioning, and research arc without making
  stronger technical claims.
- The full-paper surface may add detail, comparison, and literature context, but
  it does not broaden the frozen Phase 3 contribution boundary.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Task 8 evidence required;
  `TASK_8_MINI_SPEC.md` required behavior and acceptance evidence; publication
  surface roles in `SHORT_PAPER_PHASE_3.md`, `SHORT_PAPER_NARRATIVE.md`,
  `ABSTRACT_PHASE_3.md`, and `PAPER_PHASE_3.md`.
- ADR decision(s): `P3-ADR-001`, `P3-ADR-002`

## Story 3: Major Paper Claims And Sections Are Traceable To Phase 3 Contract Docs And Emitted Task 6 / Task 7 Bundles

**User/Research value**
- This story makes Paper 2 reviewable by ensuring that every important claim can
  be checked against authoritative contract docs and machine-reviewable emitted
  bundles instead of resting on prose alone.

**Given / When / Then**
- Given the frozen Phase 3 contract docs, the stable task mini-specs, and the
  emitted `phase3_task6` and `phase3_task7` bundle families.
- When a reviewer asks which artifact or contract source supports a major Paper
  2 claim, section, or limitation statement.
- Then the publication package provides a stable traceability path from the
  paper surface back to the authoritative docs and emitted evidence.

**Scope**
- In: section-level or equivalent claim-to-evidence mapping, citable document
  paths, and linkage between paper surfaces and the emitted correctness,
  benchmark, and diagnosis bundles.
- Out: generating new benchmark artifacts or replacing authoritative evidence
  with narrative summaries.

**Acceptance signals**
- Major Paper 2 claims and sections can be mapped to the relevant sources in
  `DETAILED_PLANNING_PHASE_3.md`, `ADRs_PHASE_3.md`,
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`, and the relevant Task 1 through
  Task 7 mini-specs.
- The publication package identifies concrete machine-reviewable evidence entry
  points under `benchmarks/density_matrix/artifacts/phase3_task6/` and
  `benchmarks/density_matrix/artifacts/phase3_task7/`, including the current
  correctness-package, unsupported-boundary, benchmark-package, diagnosis, and
  summary-consistency bundles.
- A reviewer can move from a paper-level statement to the underlying contract or
  emitted evidence without relying on source-code inspection.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Task 8 evidence required;
  `TASK_8_MINI_SPEC.md` required behavior, acceptance evidence, and affected
  interfaces.
- ADR decision(s): `P3-ADR-001`, `P3-ADR-008`, `P3-ADR-009`

## Story 4: Only Mandatory, Complete, Supported Evidence Plus The Frozen Threshold-Or-Diagnosis Rule Can Close The Main Paper 2 Claim

**User/Research value**
- This story prevents favorable examples, optional material, or partial bundles
  from inflating the publication claim beyond what Phase 3 actually delivered.

**Given / When / Then**
- Given the frozen mandatory evidence floor and the current emitted Task 6 and
  Task 7 package surfaces.
- When Paper 2 summarizes the evidence supporting the Phase 3 result.
- Then only mandatory, complete, supported evidence is allowed to close the main
  claim, and the closure path must remain the frozen threshold-or-diagnosis rule
  rather than an informal speedup narrative.

**Scope**
- In: mandatory evidence-floor wording, claim-closure semantics, required versus
  optional evidence labeling, and explicit threshold-or-diagnosis interpretation.
- Out: redefining thresholds, relaxing pass/fail semantics, or promoting
  supplemental material into the core claim.

**Acceptance signals**
- The paper package states that only mandatory, complete, supported correctness
  and reproducibility evidence may close the main Paper 2 claim.
- Optional, exploratory, deferred, unsupported, or incomplete material remains
  labeled as context, boundary evidence, or future-work motivation rather than
  counted claim-closure evidence.
- If the current representative benchmark package closes through the diagnosis
  branch, the paper package states that outcome explicitly instead of implying
  that the positive-threshold path has already been met.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Task 8 success looks like
  and evidence required; Section 10 numeric acceptance thresholds; Section 12
  full-phase acceptance criteria; `TASK_8_MINI_SPEC.md` required behavior,
  unsupported behavior, and acceptance evidence.
- ADR decision(s): `P3-ADR-005`, `P3-ADR-008`, `P3-ADR-009`, `P3-ADR-010`

## Story 5: Paper 2 Describes The Supported Path, No-Fallback Rule, Bounded Planner Claim, And Current Benchmark Surface Honestly

**User/Research value**
- This story helps readers understand exactly what Phase 3 demonstrates and
  prevents the paper from overstating planner breadth, runtime behavior, or
  current benchmark outcomes.

**Given / When / Then**
- Given the frozen planner-surface, semantic-preservation, runtime, support-
  matrix, benchmark-anchor, and performance-boundary decisions.
- When Paper 2 explains the supported partitioned-density result and the
  benchmark regime behind it.
- Then it describes the guaranteed supported path, the no-fallback rule, the
  bounded Task 5 planner claim, and the current representative benchmark outcome
  honestly without implying broader parity or stronger acceleration than the
  emitted evidence supports.

**Scope**
- In: supported-path wording, no-fallback wording, Task 5 planner-claim
  phrasing, benchmark-surface honesty, and current count or diagnosis wording.
- Out: broader planner-family closure, broader circuit-source parity, or future
  acceleration claims not backed by emitted evidence.

**Acceptance signals**
- Paper 2 states that the guaranteed path is the canonical noisy mixed-state
  planner surface plus the documented exact lowering for the frozen Phase 2
  continuity workflow and the required Phase 3 structured benchmark families.
- Paper 2 states that silent sequential substitution is not part of supported
  `partitioned_density` evidence.
- Paper 2 presents the current Task 5 result as a benchmark-calibrated selection
  rule over auditable `max_partition_qubits` span-budget settings on the
  existing noisy planner surface rather than as a broad permanently settled
  planner-family victory.
- If the current implementation-backed counts are cited, they remain accurate and
  joinable to emitted bundles, including the current `25` counted Task 6
  supported cases, `4` required external Task 6 cases, `17` explicit Task 6
  boundary cases, `34` counted Task 7 benchmark cases, and `6` representative
  review cases.

**Traceability**
- Phase requirement(s): Section 10 frozen implementation contracts, Section 12
  full-phase acceptance criteria, and Task 8 evidence required in
  `DETAILED_PLANNING_PHASE_3.md`; `TASK_8_MINI_SPEC.md` required behavior and
  acceptance evidence.
- ADR decision(s): `P3-ADR-003`, `P3-ADR-005`, `P3-ADR-006`, `P3-ADR-007`,
  `P3-ADR-008`, `P3-ADR-009`

## Story 6: One Manifest-Driven Review Package Keeps Positive Evidence, Negative Boundary Evidence, And Diagnosis Visible Together

**User/Research value**
- This story makes Task 8 reusable by reviewers and later publication packaging
  work by requiring one stable reviewer-entry package instead of disconnected
  paper prose, benchmark summaries, and manual joins across Task 6 and Task 7
  artifacts.

**Given / When / Then**
- Given the emitted Task 6 correctness and unsupported-boundary bundles, the
  emitted Task 7 benchmark and diagnosis bundles, and the need for one stable
  Paper 2 reviewer-entry path.
- When Task 8 emits publication-facing artifacts for review or downstream use.
- Then it emits one manifest-driven review package or equivalent machine-
  reviewable checker that keeps positive supported evidence, negative boundary
  evidence, and diagnosis-grounded limitation reporting visible on one shared
  review surface.

**Scope**
- In: one stable reviewer-entry manifest or checker, emitted-bundle references,
  section-level evidence mapping, positive and negative evidence visibility, and
  explicit diagnosis references.
- Out: replacing the underlying Task 6 and Task 7 bundles, hiding excluded
  cases, or turning the manifest into prose-only navigation.

**Acceptance signals**
- One stable reviewer-entry surface references the current emitted bundle entry
  points under `benchmarks/density_matrix/artifacts/phase3_task6/` and
  `benchmarks/density_matrix/artifacts/phase3_task7/`, including the current
  correctness-package, unsupported-boundary, diagnosis, sensitivity-matrix, and
  summary-consistency surfaces.
- Positive supported evidence, explicit boundary evidence, and the current
  diagnosis-grounded limitation path remain visible together instead of being
  collapsed into one optimistic publication summary.
- Later publication consumers can use one stable Task 8 review package without
  manual relabeling of workload identity, counted-status meaning, or current
  limitation interpretation.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Task 8 goal and evidence
  required; `TASK_8_MINI_SPEC.md` required behavior, acceptance evidence, and
  affected interfaces.
- ADR decision(s): `P3-ADR-001`, `P3-ADR-008`, `P3-ADR-009`, `P3-ADR-010`

## Story 7: Future Work And Publication-Ladder Positioning Stay Explicit In The Paper Package

**User/Research value**
- This story keeps Paper 2 scientifically focused and helps reviewers understand
  how the Phase 3 result fits into the larger PhD program without confusing
  current results with later-phase goals.

**Given / When / Then**
- Given the publication strategy that positions Phase 3 as the major methods and
  systems paper between Phase 2 exact-noisy integration and Phase 4 broader
  noisy workflow science.
- When Paper 2 discusses significance, limitations, and future directions.
- Then it presents Phase 3 as the noise-aware partitioning and limited-fusion
  milestone and keeps deferred branches explicitly labeled as future work rather
  than current evidence.

**Scope**
- In: future-work wording, Phase 3 positioning in the publication ladder, and
  alignment between Paper 2 and the broader roadmap.
- Out: reopening roadmap order, introducing new Phase 3 commitments, or
  reframing Paper 2 as the later optimizer or approximate-scaling paper.

**Acceptance signals**
- Paper 2 explicitly marks fully channel-native fused noisy blocks, broader
  noisy VQE/VQA workflow growth, density-backend gradients, optimizer studies,
  and approximate scaling as future work rather than as current Paper 2
  results.
- The paper package positions Phase 3 as the methods milestone that stabilizes
  the backend for later workflow and optimizer science rather than replacing
  those later phases.
- Follow-on architecture decisions are framed as benchmark-driven future work,
  not as hidden incompleteness in the current publication package.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Why Phase 3 Must Precede
  Phase 4, Decision Gates, Non-Goals, and Task 8 evidence required;
  `TASK_8_MINI_SPEC.md` required behavior, unsupported behavior, and acceptance
  evidence.
- ADR decision(s): `P3-ADR-002`, `P3-ADR-009`, `P3-ADR-010`

## Story 8: Terminology, Reviewer Entry, And Summary Consistency Stay Stable Across The Paper Package

**User/Research value**
- This story makes Paper 2 easier to review and cite by preventing terminology
  drift, count drift, and cross-surface inconsistency while preserving one
  reliable path from publication surfaces to the authoritative Phase 3 contract
  and emitted evidence.

**Given / When / Then**
- Given the full publication package across abstract, technical short-paper,
  narrative short-paper, full-paper, and reviewer-entry sources.
- When a reviewer compares wording, counts, support labels, and references
  across those surfaces.
- Then the paper package reads as one coherent story with stable terms, stable
  reviewer entry, and summary counts that agree with the emitted bundle
  surfaces.

**Scope**
- In: consistent use of key terms, stable reviewer entry points, cross-reference
  integrity, and summary-consistency across publication-facing and evidence-
  facing docs.
- Out: stylistic polish that does not affect meaning and repository navigation
  beyond the Phase 3 publication package.

**Acceptance signals**
- Terms such as exact noisy mixed-state circuits, canonical noisy planner
  surface, partitioned density runtime, real fused path, counted supported,
  diagnosis-grounded closure, required / optional / deferred / unsupported, and
  reproducibility bundle are used consistently across publication-facing
  surfaces.
- At least one stable reviewer path leads from the paper package to the
  authoritative Phase 3 contract docs and the current reviewer-entry manifest or
  equivalent machine-reviewable checker.
- Rolled-up counts, representative-case labels, and limitation summaries agree
  with the emitted summary-consistency surfaces and do not drift across abstract,
  short-paper, narrative, and full-paper wording.

**Traceability**
- Phase requirement(s): `DETAILED_PLANNING_PHASE_3.md` Task 8 evidence
  requirements; `TASK_8_MINI_SPEC.md` required behavior and acceptance evidence.
- ADR decision(s): `P3-ADR-001`, `P3-ADR-008`, `P3-ADR-009`
