---
name: spec-driven-development
description: >-
  Guides spec-driven development for the density matrix project in SQUANDER.
  Use when planning phases, creating phase docs, writing DETAILED_PLANNING,
  ADRs, task mini-specs, pre-implementation checklists, or papers; or when the
  user mentions spec-driven development, phase contracts, task-level specs, or
  the density matrix planning workflow.
---

# Spec-Driven Development (Density Matrix Project)

This skill documents the working model for spec-driven development in the
density matrix project. All phase work follows a layered hierarchy:
contracts first, then task mini-specs, then stories, then code.

## Source-of-Truth Documents

Phase work is driven by:

- `docs/density_matrix_project/planning/PLANNING.md` — high-level research plan
- `docs/density_matrix_project/planning/PUBLICATIONS.md` — publication strategy
- `docs/density_matrix_project/planning/` — ADRs, REFERENCES, etc.
- `docs/density_matrix_project/phases/phase-X/` — phase-specific outputs

## Four-Layer Hierarchy

### Layer 1: Phase Contract (whole-phase level)

Define before implementation. Documents:

- **What Phase X is** and what it is not
- **Prerequisites** before implementation can start
- **Success criteria** for the phase

Primary deliverables:

| Document | Purpose |
|----------|---------|
| `DETAILED_PLANNING_PHASE_X.md` | Scope, tasks (as goals), acceptance criteria, traceability |
| `ADRs_PHASE_X.md` | Architecture and scope decisions for the phase |
| `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | Gap list validating that the plan is detailed enough to implement |

**Practical rule:** If a decision affects multiple tasks, close it at the phase level (ADRs). If it affects only one task, defer to that task's mini-spec.

### Layer 2: Task-Level Mini-Spec (before each task)

Before starting each task, define in a mini-spec:

- Required behavior
- Unsupported behavior
- Acceptance evidence
- Affected interfaces
- Publication relevance (if any)

**Location:** `phases/phase-X/task-N/TASK_N_MINI_SPEC.md`

Example: `phases/phase-2/task-1/TASK_1_MINI_SPEC.md`

### Layer 3: Stories (behavioral slices)

Stories describe **behavioral slices**, not code chores. Examples:

- "Backend selection works for workflow X"
- "Observable path supports Hamiltonian form Y"
- "Unsupported gate families fail in a documented way"

### Layer 4: Engineering Tasks

Only here do you create:

- Code changes
- Test additions
- Benchmark runs
- Doc updates
- Validation runs

## Industry Best-Practice Refinements

Apply these practices when authoring or reviewing phase artifacts:

1. **Contract-first, decision-fast:** close cross-task decisions in ADRs early to
   avoid implementation drift.
2. **Behavior over mechanism:** write outcomes users/researchers can observe
   before implementation details.
3. **Testable acceptance language:** each required behavior should map to
   reproducible evidence and a pass/fail criterion.
4. **Traceability by default:** maintain requirement -> decision -> task ->
   evidence links.
5. **Explicit non-goals and unsupported behavior:** ambiguity is treated as
   scope risk.
6. **Publication-aware definitions of done:** claims in abstract/papers must be
   backed by planned validation evidence.
7. **Small vertical slices for delivery:** stories are the behavioral unit;
   engineering tasks are implementation units under each story.

## Behavioral Story Template (Layer 3)

Use this template for story-level behavioral slices (not code chores):

```markdown
### Story: [Behavioral outcome]

**User/Research value**
- Why this behavior matters to the phase objective

**Given / When / Then**
- Given [initial system contract state]
- When [workflow action]
- Then [observable outcome and boundary behavior]

**Scope**
- In: [what this story covers]
- Out: [explicit exclusions]

**Acceptance signals**
- [Signal 1: measurable behavioral evidence]
- [Signal 2: negative/unsupported behavior handling]

**Traceability**
- Phase requirement(s): [IDs or section references]
- ADR decision(s): [ADR IDs]
```

Quality gate for stories:
- independent enough to validate in isolation,
- negotiable in implementation details,
- valuable to workflow outcomes,
- estimable and small enough for one implementation cycle,
- testable via concrete acceptance signals.

## Engineering Task Template (Layer 4)

Use this template for implementation tasks created under a story:

```markdown
### Engineering Task: [Action-oriented title]

**Implements story**
- [Story title/reference]

**Change type**
- code | tests | benchmark harness | docs | validation automation

**Definition of done**
- [Concrete completion condition 1]
- [Concrete completion condition 2]

**Execution checklist**
- [ ] Implement targeted change
- [ ] Add/adjust tests
- [ ] Run validation/benchmarks needed by the story
- [ ] Update docs/reproducibility artifacts

**Evidence produced**
- [Test run reference]
- [Benchmark/validation artifact]

**Risks / rollback**
- Risk: [known risk]
- Rollback/mitigation: [how to revert or isolate impact]
```

Quality gate for engineering tasks:
- maps to exactly one primary story outcome,
- has objective done criteria,
- produces evidence artifacts,
- declares risk and mitigation.

## Phase Workflow (Three Steps)

The first step for any phase is creating the initial phase documents. Then validate, then close the checklist. Use the prompts below (substitute `phase-X` and `Phase X` for the target phase, e.g. phase-2, Phase 2).

### Step 1: Create Initial Phase Documents

Work out a detailed plan per PLANNING.md and all docs in `planning/`, plus findings. Write into `DETAILED_PLANNING_PHASE_X.md`. Do not change code. Apply spec-driven development principles. Prepare the phase paper per PUBLICATIONS.md in 3 steps: (1) ~4-page short paper in `SHORT_PAPER_PHASE_X.md`, (2) abstract in `ABSTRACT_PHASE_X.md` for PhD conference presentation, (3) full paper in `PAPER_PHASE_X.md`. Put all phase-X docs in `phases/phase-X/`. Break down implementation into tasks and acceptance criteria. Tasks are goals, not implementations. No code snippets in phase docs except API-reference ones after implementation. Document all phase-X decisions in `ADRs_PHASE_X.md`.

**Example invocation (phase 2):** "Work out a detailed plan for phase 2 according to @docs/density_matrix_project/planning/PLANNING.md and according to all the documentation in @docs/density_matrix_project/planning and according to all the findings. Write it into file DETAILED_PLANNING_PHASE_2.md. Do not change code. Apply spec driven development principles. Prepare Paper 1 according to @docs/density_matrix_project/planning/PUBLICATIONS.md in 3 steps, first one is 4 page-long short paper in SHORT_PAPER_PHASE_2.md and an abstract ABSTRACT_PHASE_2.md for a phd conference presentation and a 'normal' paper in PAPER_PHASE_2.md. All the new phase-2 related documents should be put in directory @phases/phase-2. Break down the phase 2 implementation into tasks and acceptance criteria. Tasks are goals, not implementations. No code snippets in phase files except for the API reference related ones after the implementation. Document all the phase-2 related decisions in detail in ADRs_PHASE_2.md. Work in phase specific subdirectory phase-2."

### Step 2: Validate Completeness and Implementation Readiness

Validate whether `DETAILED_PLANNING_PHASE_X.md` is detailed and thorough enough to start implementation. Turn the validation into a concise gap list in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`.

**Example invocation (phase 2):** "Validate if DETAILED_PLANNING_PHASE_2.md is detailed and thorough enough to start the implementation. Turn this validation into a concise gap list and add to a separate phase 2 doc as PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md."

### Step 3: Close Pre-Implementation Checklist Items

Close each open checklist item so the phase contract is solid. Ask questions or decide; record the decision and trade-offs. Map each open item to a concrete contract decision or clarification. Define the relevant phase decisions (e.g. backend-selection, observable, bridge, support-matrix, workflow-anchor, benchmark-minimum, acceptance-threshold) with trade-offs. Update planning, ADRs, and checklist. Re-read updated docs for internal consistency and verify the checklist is solidly closed and implementation-ready.

**Example invocation (phase 2):** "Close the pre implementation checklist items for phase 2 to make the phase 2 contract solid enough. Ask question or decide but record the decision and the trade-offs. Review the Phase 2 pre-implementation checklist and map each open item to a concrete contract decision or required clarification. Define backend-selection, observable, bridge, support-matrix, workflow-anchor, benchmark-minimum, and acceptance-threshold decisions with trade-offs. Update the Phase 2 planning, ADR, and checklist docs to record decisions, trade-offs, and checklist closure. Re-read updated Phase 2 docs for internal consistency and verify the checklist is solidly closed and ready for implementation."

Note: Adjust decision areas per phase — Phase 2 needs all listed; later phases may differ.

### Sequence Summary

1. **Step 1** → Creates DETAILED_PLANNING, ADRs, abstract, short paper, full paper in `phases/phase-X/`.
2. **Step 2** → Creates `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` with initial gap list.
3. **Step 3** → Closes each gap; updates planning, ADRs, checklist until implementation-ready.
4. **Implementation** → Create task mini-specs as needed; implement task by task.

## Spec-Driven Principles (from Phase 2)

1. Define contracts, scope, and success criteria **before** implementation.
2. Separate **required behavior** from implementation choices.
3. Maintain traceability from milestone goals to validation evidence.
4. Treat **unsupported** and **deferred** cases as documented outcomes.
5. Use publication evidence requirements to guide "done."
6. Keep task descriptions **goal-oriented**, not implementation-prescriptive.

## Document Structure for Phase Planning

`DETAILED_PLANNING_PHASE_X.md` should include:

- Purpose and mission
- Source-of-truth hierarchy
- Traceability matrix (requirements → phase interpretation)
- In-scope / out-of-scope
- Assumptions
- Success conditions
- Frozen implementation contracts (backend, observable, bridge, support matrix, workflow anchor, benchmark minimum, numeric thresholds)
- Task breakdown (goals, why, success looks like, evidence required)
- Full-phase acceptance criteria
- Validation and benchmark matrix
- Risks and decision gates
- Non-goals
- Expected outcome

## Template Validation Rubric (ADRs, Planning, Mini-Spec)

Use this rubric to validate template quality before implementation starts.

### ADR template must include

- title and unique ADR ID,
- status,
- context,
- decision,
- rationale,
- consequences,
- rejected alternatives,
- upstream alignment and traceability.

### Detailed planning template must include

- purpose/mission and source-of-truth hierarchy,
- traceability matrix,
- in-scope/out-of-scope boundaries,
- assumptions and success conditions,
- frozen implementation contracts and numeric thresholds,
- task breakdown as goals plus evidence expectations,
- acceptance criteria, validation matrix, risks, and decision gates.

### Task mini-spec template must include

- required behavior,
- unsupported behavior,
- acceptance evidence,
- affected interfaces,
- publication relevance.

### Current validation verdict (Phase 2 artifacts)

- **ADRs template:** `Pass` (all core ADR sections present and consistent).
- **Detailed planning template:** `Pass with strength` (contains full contract
  structure, thresholds, tasks, risks, and gates).
- **Task mini-spec template:** `Pass` (required sections present and behavior
  first).

Observed improvements to apply going forward:
- add explicit `Given/When/Then` behavioral wording where helpful,
- add a lightweight assumptions/dependencies subsection to mini-specs when
  external coupling is significant,
- keep evidence references stable (IDs or artifact names) for auditability.

## Pre-Implementation Checklist

`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`:

- Maps each open item to the contract that closes it
- States readiness verdict (implementation-ready or not)
- Records closure decisions and trade-offs
- Includes a **Go / No-Go rule** for when implementation can begin

## Do Not

- Write implementation code in phase docs except API references post-implementation
- Fully story-split the entire phase before starting
- Defer multi-task decisions to task mini-specs (close at phase level)
- Treat papers as a downstream retrofit — prepare them in parallel with planning

## Additional Resources

- For full prompt text (copy-paste), closure patterns, task mini-spec template, and story vs. engineering-task guidance, see [reference.md](reference.md)

## Example Phase 2 References

- `docs/density_matrix_project/phases/phase-2/DETAILED_PLANNING_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/ADRs_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
- `docs/density_matrix_project/phases/phase-2/ABSTRACT_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/SHORT_PAPER_PHASE_2.md`
- `docs/density_matrix_project/phases/phase-2/PAPER_PHASE_2.md`
