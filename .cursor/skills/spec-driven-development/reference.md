# Spec-Driven Development — Detailed Reference

**Canonical rules** (hierarchy, two-paper model, paths, sub-phases): [SKILL.md](SKILL.md).
This file provides copy-paste prompts and a **historical** Phase 2 closure
table. When SKILL and this file disagree, **SKILL wins** — update this file to
match.

**Path rule:** All phase outputs under `docs/density_matrix_project/phases/...`
only.

## Three-Step Phase Workflow Prompts

Substitute `phase-X` / `Phase X` / `PHASE_X` for the target (e.g. `phase-2`,
Phase 2, `PHASE_2`, or `phase-3-1`, Phase 3.1, `PHASE_3_1`). Sub-phases use the
same artifact set as numeric phases; see SKILL.md for filename patterns.

### Step 1: Create Initial Phase Documents

```
Work out a detailed plan for phase X according to @docs/density_matrix_project/planning/PLANNING.md and according to all the documentation in @docs/density_matrix_project/planning and according to all the findings. Write it into file DETAILED_PLANNING_PHASE_X.md under @docs/density_matrix_project/phases/phase-X. Do not change code. Apply spec driven development principles. Prepare publication outputs per @docs/density_matrix_project/planning/PUBLICATIONS.md in 4 steps: (1) technical short paper in SHORT_PAPER_PHASE_X.md, (2) narrative positioning short paper in SHORT_PAPER_NARRATIVE.md in the same directory, (3) abstract ABSTRACT_PHASE_X.md for PhD conference presentation, (4) full paper PAPER_PHASE_X.md. Break down the phase X implementation into tasks and acceptance criteria. Tasks are goals, not implementations. No code snippets in phase files except for the API reference related ones after the implementation. Document all the phase-X related decisions in detail in ADRs_PHASE_X.md in the same phase directory.
```

Do **not** add `SHORT_PAPER_4PAGE.md`; the two short surfaces are technical
(`SHORT_PAPER_PHASE_X.md`) and narrative (`SHORT_PAPER_NARRATIVE.md`). See
SKILL.md.

### Step 2: Validate Completeness and Implementation Readiness

```
Validate if docs/density_matrix_project/phases/phase-X/DETAILED_PLANNING_PHASE_X.md is detailed and thorough enough to start the implementation. Turn this validation into a concise gap list in docs/density_matrix_project/phases/phase-X/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md.
```

### Step 3: Close Pre-Implementation Checklist Items

```
Close the pre implementation checklist items for phase X to make the phase X contract solid enough. Ask question or decide but record the decision and the trade-offs. Review the Phase X pre-implementation checklist and map each open item to a concrete contract decision or required clarification. Define backend-selection, observable, bridge, support-matrix, workflow-anchor, benchmark-minimum, and acceptance-threshold decisions with trade-offs (adjust this list for phases that do not need every area). Update the Phase X planning, ADR, and checklist docs under docs/density_matrix_project/phases/phase-X to record decisions, trade-offs, and checklist closure. Re-read updated Phase X docs for internal consistency and verify the checklist is solidly closed and ready for implementation.
```

For phases other than Phase 2, adjust the decision areas (backend-selection,
observable, bridge, support-matrix, workflow-anchor, benchmark-minimum,
acceptance-threshold) to match the phase’s contract surface.

## Phase 3.1 Closure Pattern (bounded positive-methods branch)

When a sub-phase such as `phase-3-1` is trying to produce a **bounded positive
methods paper** rather than a broad neutral study, the checklist-closure pass
should typically freeze decisions in this order:

1. **Primary representation** — choose one public exact channel form and define
   invariant checks.
2. **Support slice** — freeze a narrow but publishable eligibility class rather
   than a broad feature surface.
3. **Numeric policy** — reuse prior exactness thresholds where possible; add
   representation-level invariants if the claim surface is richer.
4. **Counted correctness slice** — list stable case IDs that directly test the
   new claim.
5. **External reference slice** — keep Aer or another external simulator only
   where it strengthens the exactness story.
6. **Counted performance slice** — define the cases that could realistically
   show a win, plus control cases.
7. **Mode / API surface** — freeze how the new path is named without reopening
   the whole planner contract.
8. **Evidence schema** — extend existing artifact trees rather than creating a
   parallel evidence universe when continuity matters.

For the current Phase 3.1 branch in this repo, the contract now favors:

- a **bounded exact mixed-motif** claim over broad gate/noise expansion,
- a **positive-methods paper first** with honest fallback to diagnosis,
- and explicit `break_even_table` / `justification_map` evidence instead of
  relying on prose alone.

If a later sub-phase evidence flow is expected to become the **new default**
pipeline after stabilization, prefer this migration policy:

- keep the historical phase available through **explicit legacy
  scripts/functions**,
- let the later phase become the default `validation_pipeline.py` surface only
  after its sibling builders and validation slices are complete,
- avoid a mode flag when reproducibility and paper auditability matter.

## Phase Contract Closure Pattern (Phase 2 — historical example)

The Phase 2 `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` mapped open gaps to
contract decisions as follows (illustrative, not a template for every phase):

| Former open item        | Contract decision that closes it                  | Primary decision record |
|-------------------------|---------------------------------------------------|-------------------------|
| Backend selection       | Explicit modes, default, hard-error on unsupported| P2-ADR-009              |
| Observable contract     | Exact `Re Tr(H*rho)`, XXZ benchmark family        | P2-ADR-010              |
| Circuit-to-density bridge| Partial HEA-first, hard-error unsupported         | P2-ADR-011              |
| Support matrix          | Required / optional / deferred gate and noise      | P2-ADR-012              |
| Workflow anchor         | Noisy XXZ VQE, 4–10 qubit regime                  | P2-ADR-013              |
| Benchmark minimum       | Aer-centered, microcases, workflow cases          | P2-ADR-014              |
| Numeric thresholds      | 4–10 qubit, error/validity limits                 | P2-ADR-015              |

## Task Mini-Spec Template

Before implementing each task, create
`docs/density_matrix_project/phases/phase-X/task-N/TASK_N_MINI_SPEC.md` with:

```markdown
# Task N: [Title]

## Required behavior
- What must be true when done
- Mandatory interfaces and semantics

## Unsupported behavior
- Explicit exclusions and failure modes
- Documented hard-error conditions

## Acceptance evidence
- Tests, benchmarks, or validation checks
- Traceability to phase acceptance criteria

## Affected interfaces
- API, config, or workflow surfaces
- Breaking vs. additive changes

## Publication relevance
- Paper section or claim this task supports (if any)
```

## Story vs. Engineering Task

**Story (behavioral slice):**

- "Backend selection works for noisy VQE workflow"
- "Observable path returns correct energy for XXZ Hamiltonian"
- "Unsupported gate raises clear error with operation name"

**Engineering task (code-focused):**

- "Add backend parameter to VQE constructor"
- "Implement Tr(H*rho) in observable evaluator"
- "Add gate validation before execution"

Stories are verified by tests and benchmarks; engineering tasks are implementation steps.

## Behavioral Story Template (Recommended)

Use this when decomposing a task mini-spec into delivery slices:

```markdown
### Story: [Behavioral outcome]

**User/Research value**
- [Why this behavior matters]

**Given / When / Then**
- Given [starting contract state]
- When [workflow action]
- Then [observable result]

**In scope**
- [Behavior this story covers]

**Out of scope**
- [Excluded behavior]

**Acceptance evidence**
- [Required validation/test evidence]

**Traceability**
- Requirement(s): [planning references]
- ADR(s): [IDs]
```

## Engineering Task Template (Recommended)

Use this for implementation actions under a story:

```markdown
### Engineering Task: [Action-oriented title]

**Implements story**
- [Story reference]

**Change type**
- code | tests | benchmarks | docs | validation automation

**Definition of done**
- [Done condition 1]
- [Done condition 2]

**Execution checklist**
- [ ] Implement scoped change
- [ ] Add/update tests
- [ ] Run validation/benchmark command(s)
- [ ] Update docs or reproducibility artifacts

**Evidence outputs**
- [Test artifact]
- [Benchmark/validation artifact]

**Risk / mitigation**
- Risk: [risk]
- Mitigation: [plan]
```

## Template Validation Checklist

Use this quick quality check before declaring implementation-ready.

### ADR template check

- [ ] ADR has ID, title, and status
- [ ] Context and decision are explicit and non-overlapping
- [ ] Rationale and consequences are distinct
- [ ] Rejected alternatives are documented
- [ ] Upstream alignment / traceability is included

### Detailed phase planning template check

- [ ] Purpose + mission are explicit
- [ ] Source-of-truth hierarchy and traceability matrix exist
- [ ] Scope boundaries and non-goals are explicit
- [ ] Frozen contracts and numeric thresholds are present
- [ ] Task goals are behavioral (not implementation recipe)
- [ ] Validation/benchmark matrix and decision gates are present

### Task mini-spec template check

- [ ] Required behavior is precise and testable
- [ ] Unsupported behavior and hard-error expectations are explicit
- [ ] Acceptance evidence references phase criteria
- [ ] Affected interfaces identify impact surface
- [ ] Publication relevance is clear or explicitly N/A

## Historical template validation (Phase 2)

One-time review of Phase 2 artifacts against the checklist above (not a
standing certification for later phases):

- ADRs (`ADRs_PHASE_2.md`): **Pass** against the checklist.
- Detailed planning (`DETAILED_PLANNING_PHASE_2.md`): **Pass** with strong
  completeness and traceability.
- Task mini-spec (`task-1/TASK_1_MINI_SPEC.md`): **Pass**; aligns with the
  required mini-spec structure.

Recommended incremental improvements (ongoing):

- normalize behavioral phrasing with `Given/When/Then` in story decomposition,
- add a short dependencies/assumptions subsection in mini-specs when needed,
- keep evidence artifact labels stable for easier review and publication reuse.

## Publication outputs (align with SKILL.md and PUBLICATIONS.md)

Per phase directory under `docs/density_matrix_project/phases/phase-X/`:

1. **`SHORT_PAPER_PHASE_X.md`** — technical methods, validation plan, thresholds
2. **`SHORT_PAPER_NARRATIVE.md`** — narrative companion (not a second draft of (1))
3. **`ABSTRACT_PHASE_X.md`** — conference abstract
4. **`PAPER_PHASE_X.md`** — full paper

Detail and venue constraints live in `PUBLICATIONS.md`; do not re-encode word
counts here if they change there.

## When to Close vs. Defer

| Decision scope | Close at | Defer to |
|----------------|----------|----------|
| Backend modes, observable contract, support matrix | Phase level (ADRs) | — |
| Specific gate lowering for one circuit family | — | Task 3 mini-spec |
| Benchmark seed format for reproducibility | Phase level | — |
| Exact error message for unsupported gate X | — | Task 3 mini-spec |
