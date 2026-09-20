# Closure Plan for Phase 3.1

This document defines the evidence-first closure path for Phase 3.1 from the
current implementation-backed state to a reviewer-defensible scientific closeout.
It is a phase-local companion to:

- `DETAILED_PLANNING_PHASE_3_1.md`,
- `ADRs_PHASE_3_1.md`,
- `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`,
- `task-3/TASK_3_MINI_SPEC.md`,
- `task-4/TASK_4_MINI_SPEC.md`,
- and `task-5/TASK_5_MINI_SPEC.md`.

It does **not** reopen the frozen support matrix, thresholds, or counted case
inventory. If this document conflicts with the frozen contract, the contract
documents above win.

## 1. Role in the spec-driven hierarchy

Phase 3.1 closure should continue to follow the spec-driven hierarchy rather
than becoming an ad hoc manuscript sprint.

- **Layer 1 remains frozen:** `DETAILED_PLANNING_PHASE_3_1.md`,
  `ADRs_PHASE_3_1.md`, and
  `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` define the phase contract and the
  accepted evidence boundary.
- **Layer 2 remains task-driven:** closure consumes Task 3, Task 4, and Task 5
  mini-specs; it does not replace them.
- **Layer 3 / Layer 4 closure path is already defined:** the remaining
  behavioral path is `P31-S10 -> P31-S11 -> P31-S12 -> Task 5`.
- **This document is the closure playbook:** it sequences the remaining evidence
  work, states the go / no-go rubric, and defines how the publication surfaces
  are revised once the evidence review is complete.

## 2. Rebaselined starting point

Phase 3.1 should now be treated as an **implementation-backed** phase, not as a
documentation-only branch.

- The strict motif-proof runtime and the explicit hybrid whole-workload runtime
  already exist.
- Bounded correctness artifacts already exist and should be treated as the
  current starting evidence rather than as hypothetical future work.
- At least one structured hybrid pilot row already exists and therefore changes
  the closure problem from "can the method run?" to "what does the frozen
  evidence package justify?"

Because the phase is already implementation-backed, publication sync must obey
the spec-driven publication rule from `spec-driven-development/SKILL.md`:
**code, tests, and emitted artifacts outrank stale planning prose when they
disagree**. Closure work must therefore begin by reclassifying the current Phase
3.1 statements as:

- **implemented and validated,**
- **implemented but only partially validated,**
- or **scaffolded / planned only.**

## 3. Closure objective

The goal is to close Phase 3.1 without broadening the frozen v1 slice and
without weakening its scientific honesty.

Phase 3.1 closes successfully only when it exits in one of two explicit modes:

1. **bounded positive-methods outcome** on the frozen v1 slice, or
2. **benchmark-grounded decision-study outcome** on the same slice.

The phase is **not** closed if it still depends on partial evidence, ambiguous
publication wording, or contradictory top-level documentation.

## 4. Phase-specific closure principles

The remaining closure work must follow these Phase 3.1-specific rules.

1. **Evidence first, prose second.** Finish the counted correctness and
   performance obligations before attempting venue-ready manuscript wording.
2. **Strict and hybrid claims stay separate.**
   - strict `phase31_channel_native` carries the bounded fused-object claim,
   - hybrid `phase31_channel_native_hybrid` carries continuity and
     whole-workload claims.
3. **The frozen v1 slice does not grow during closure.** No new gate families,
   no larger support surface, and no Phase 4 spillover.
4. **Phase 3 remains the shipped baseline.** Phase 3.1 is additive and must not
   rewrite Phase 3's closed claim boundary.
5. **Negative or mixed results are first-class outcomes.** If the full matrix
   does not justify the richer method, Phase 3.1 should close as a decision
   study rather than by inflating weak wins.
6. **Narrative writing stays science-first.** The narrative companion explains
   scientific significance and research arc, not file paths, internal task IDs,
   or repo mechanics.
7. **Manuscript mode is chosen mechanically.** The publication mode is selected
   only after the pre-publication evidence review records one of the accepted
   closure states below.

## 5. Closure states and go / no-go rubric

| Closure state | Required evidence state | Allowed publication stance | Disallowed stance |
|---|---|---|---|
| `positive-methods-ready` | Mandatory correctness slice green; required external slice green; full counted performance matrix emitted; `break_even_table` / `justification_map` emitted; at least one representative primary-family row meets the frozen positive threshold versus the Phase 3 fused baseline | Bounded positive-methods paper on the frozen v1 slice | Broad or universal acceleration language |
| `decision-study-ready` | Mandatory correctness slice green; full counted performance matrix emitted; decision artifact emitted; no row meets the positive threshold or the wins remain too narrow / mixed for the stronger claim | Benchmark-grounded decision study or negative-result methods note | Silent implication that acceleration has already been established |
| `not-ready-yet` | Any mandatory correctness or performance obligation remains incomplete, contradictory, or blocked | Internal boundary-sync draft only; no venue-ready closure claim | Submission-ready abstract or conclusive results language |

**Go rule for Task 5 publication closure**

- Proceed to Task 5 only after the pre-publication evidence review records
  `positive-methods-ready` or `decision-study-ready`.

**No-go rule**

- If the review state is `not-ready-yet`, stop at evidence review, record
  blockers explicitly, and do not write conclusive publication prose.

## 6. Closure work packages

### Package A: Correctness closure (Task 3 / Story P31-S10)

This package completes the **frozen counted correctness** surface.

**In scope**

- the remaining counted strict microcases,
- the counted hybrid `phase2_xxz_hea_q6_continuity` anchor,
- the required Aer subset from `P31-ADR-011`,
- the versioned correctness-package migration from `P31-ADR-013`,
- and stable `channel_invariants` / `partition_route_summary` emission.

**Out of scope**

- the structured performance matrix,
- publication framing,
- and Task 6 host-acceleration work.

**Definition of done**

- Every counted correctness row in the frozen slice is either green with stable
  evidence metadata or is explicitly recorded as a blocker.
- The correctness package exposes the required Phase 3.1 case metadata and
  schema versioning.
- The external slice stays bounded to the required five rows.

### Package B: Performance closure (Task 4 / Story P31-S11)

This package completes the **whole-workload justification** question.

**In scope**

- the full frozen `P31-ADR-010` counted matrix,
- the baseline trio on every counted row,
- hybrid route coverage,
- `decision_class`,
- and `break_even_table` / `justification_map`.

**Out of scope**

- new workload families,
- Task 6 offload or SIMD claims,
- and paper wording.

**Definition of done**

- The full counted matrix is emitted with stable IDs and scalar-build metadata.
- Each counted row records sequential, Phase 3 fused, and Phase 3.1 hybrid
  measurements.
- The decision artifact classifies where Phase 3 is sufficient, where Phase 3.1
  is justified, and where it is not justified yet.

### Package C: Pre-publication evidence review (Story P31-S12)

This package decides whether the phase is ready for manuscript closure.

**Required artifact**

- `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`

**Definition of done**

- The review pack records one of:
  - `positive-methods-ready`,
  - `decision-study-ready`,
  - `not-ready-yet`.
- The review is artifact-indexed and case-ID-indexed.
- The review does not yet rewrite the paper surfaces.

### Package D: Publication sync (Task 5)

This package updates the paper surfaces only after Package C is complete.

**Definition of done**

- The technical short paper, narrative, abstract, and full paper all agree with
  the same closure state.
- Every claim can be mapped to a strict or hybrid evidence path, case IDs, and
  thresholds.
- Limitations are explicit wherever the frozen evidence remains narrow.

### Package E: Program sync

After the phase-local publication surfaces stabilize, update the program-level
documents that point to Phase 3.1 so the repo tells one consistent story.

**Primary targets**

- `docs/density_matrix_project/planning/PLANNING.md`
- `docs/density_matrix_project/planning/PUBLICATIONS.md` (only if the closure
  state changes the recommended publication packaging)
- `docs/density_matrix_project/CHANGELOG.md`

**Optional target**

- `API_REFERENCE_PHASE_3_1.md` if the phase wants an explicit reviewer-facing
  API note for the strict / hybrid surface.

## 7. Academic-writing workflow for Task 5

There is no separate manuscript sprint outside the phase contract. Academic
writing must be treated as a controlled downstream transformation of the
evidence review.

### 7.1 Statement classification

Before editing any publication surface, classify each intended statement as:

- **implemented and validated,**
- **implemented but only partially validated,**
- or **scaffolded / planned only**.

Only the first class may appear as an affirmative results claim. The second
class belongs in limitations or "current evidence" wording. The third class
belongs only in future-work or non-goal language.

### 7.2 Ordered writing sequence

Task 5 should update documents in the following order:

1. **Freeze the review state** in
   `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`.
2. **Revise `SHORT_PAPER_PHASE_3_1.md`** so the methods/result boundary is
   explicit.
3. **Revise `PAPER_PHASE_3_1.md`** as the extended version of the same claim
   boundary.
4. **Derive `ABSTRACT_PHASE_3_1.md` last from the stabilized claim boundary,**
   not from aspirational phrasing.
5. **Revise `SHORT_PAPER_NARRATIVE.md`** in science-first form after the
   technical boundary is stable.
6. **Update top-level pointers** (`PLANNING.md`, `CHANGELOG.md`, optional
   `PUBLICATIONS.md`) only after the phase-local surfaces agree.

### 7.3 Surface-specific writing briefs

#### `SHORT_PAPER_PHASE_3_1.md`

- Lead with the exact scientific object and the frozen support slice.
- Distinguish strict evidence from hybrid evidence.
- Include a claim boundary section and an answer map to the Side Paper A
  question.
- If the closure state is negative or mixed, state that directly rather than
  softening it through vague optimism.

#### `PAPER_PHASE_3_1.md`

- Expand methods, validation, and limitations from the same claim boundary as
  the short paper.
- Include full case-ID-level results discussion where needed.
- Keep the paper about the scientific object, not the development process.

#### `ABSTRACT_PHASE_3_1.md`

- Derive from the final claim boundary only.
- The results sentence must match the selected closure mode exactly.
- Do not promise matrix closure, broader scalability, or performance benefit if
  the evidence review does not support it.

#### `SHORT_PAPER_NARRATIVE.md`

- Stay science-first and research-arc-first.
- Explain why exact noisy motif fusion matters and what methodological lesson
  Phase 3.1 contributes.
- Avoid repo inventory, internal task IDs, and roadmap wording.

### 7.4 Sentence-level rules

- Every whole-workload performance claim must name the **hybrid** interpretation
  if that is the evidence-carrying path.
- Every correctness claim must name the counted slice or the relevant case IDs
  and thresholds.
- Use measured verbs such as **demonstrates**, **establishes on the frozen
  slice**, **indicates**, or **supports a decision-study conclusion**.
- Avoid unbounded verbs such as **solves**, **proves generally**, or
  **outperforms** without a cited counted matrix.
- Keep a visible limitations paragraph in all publication surfaces.
- Treat smoke tests, pilot rows, and non-counted cases as supportive context,
  not as claim-bearing substitutes for the counted slice.

### 7.5 Writing stance by closure state

| Closure state | Writing stance |
|---|---|
| `positive-methods-ready` | Present Phase 3.1 as a bounded exact methods extension beyond Phase 3, with explicit narrow support and explicit positive rows |
| `decision-study-ready` | Present Phase 3.1 as a benchmark-grounded follow-on study showing where exact channel-native fusion is feasible and where it is not justified beyond the shipped baseline |
| `not-ready-yet` | Restrict prose to internal status / limitation wording; do not write a venue-ready abstract |

## 8. Required closure deliverables

Phase 3.1 should not be considered fully closed until the following artifacts
exist and agree:

- updated counted correctness bundles,
- updated counted performance bundles,
- emitted `break_even_table` / `justification_map`,
- completed `PRE_PUBLICATION_EVIDENCE_REVIEW_PHASE_3_1.md`,
- updated:
  - `SHORT_PAPER_PHASE_3_1.md`,
  - `SHORT_PAPER_NARRATIVE.md`,
  - `ABSTRACT_PHASE_3_1.md`,
  - `PAPER_PHASE_3_1.md`,
- and synchronized top-level pointers where required.

## 9. Stop conditions and anti-patterns

Stop and record a blocker instead of proceeding if any of the following occurs:

- the counted correctness slice is still incomplete,
- the performance matrix is partial but already being interpreted as decisive,
- the review state is still `not-ready-yet`,
- top-level docs still describe the phase as unopened while the paper surfaces
  describe it as complete,
- or Task 5 wording starts to exceed the emitted artifacts.

The main anti-patterns for Phase 3.1 closure are:

- treating one pilot row as if it closes the whole-workload question,
- letting the narrative companion become a software changelog,
- and trying to rescue a weak positive-methods claim instead of shipping an
  honest decision study.
