# Spec-Driven Development — Detailed Reference

## Three-Step Phase Workflow Prompts

### Step 1: Create Initial Phase Documents (substitute phase-X, Phase X, Paper N)

```
Work out a detailed plan for phase X according to @docs/density_matrix_project/planning/PLANNING.md and according to all the documentation in @docs/density_matrix_project/planning and according to all the findings. Write it into file DETAILED_PLANNING_PHASE_X.md. Do not change code. Apply spec driven development principles. Prepare Paper N according to @docs/density_matrix_project/planning/PUBLICATIONS.md in 3 steps, first one is 4 page-long short paper in SHORT_PAPER_PHASE_X.md and an abstract ABSTRACT_PHASE_X.md for a phd conference presentation and a "normal" paper in PAPER_PHASE_X.md. All the new phase-X related documents should be put in directory @phases/phase-X. Break down the phase X implementation into tasks and acceptance criteria. Tasks are goals, not implementations. No code snippets in phase files except for the API reference related ones after the implementation. Document all the phase-X related decisions in detail in ADRs_PHASE_X.md. Work in phase specific subdirectory phase-X.
```

### Step 2: Validate Completeness and Implementation Readiness

```
Validate if DETAILED_PLANNING_PHASE_X.md is detailed and thorough enough to start the implementation. Turn this validation into a concise gap list and add to a separate phase X doc as PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md.
```

### Step 3: Close Pre-Implementation Checklist Items

```
Close the pre implementation checklist items for phase X to make the phase X contract solid enough. Ask question or decide but record the decision and the trade-offs. Review the Phase X pre-implementation checklist and map each open item to a concrete contract decision or required clarification. Define backend-selection, observable, bridge, support-matrix, workflow-anchor, benchmark-minimum, and acceptance-threshold decisions with trade-offs. Update the Phase X planning, ADR, and checklist docs to record decisions, trade-offs, and checklist closure. Re-read updated Phase X docs for internal consistency and verify the checklist is solidly closed and ready for implementation.
```

For phases other than Phase 2, adjust the decision areas (backend-selection, observable, bridge, support-matrix, workflow-anchor, benchmark-minimum, acceptance-threshold) to match the phase’s contract surface.

## Phase Contract Closure Pattern

The `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` maps each open gap to a contract decision:

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

Before implementing each task, create `phase-X/task-N/TASK_N_MINI_SPEC.md` with:

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

## Current Template Validation (Phase 2)

- ADRs (`ADRs_PHASE_2.md`): **Pass** against the checklist.
- Detailed planning (`DETAILED_PLANNING_PHASE_2.md`): **Pass** with strong
  completeness and traceability.
- Task mini-spec (`task-1/TASK_1_MINI_SPEC.md`): **Pass**; aligns with the
  required mini-spec structure.

Recommended incremental improvements:
- normalize behavioral phrasing with `Given/When/Then` in story decomposition,
- add a short dependencies/assumptions subsection in mini-specs when needed,
- keep evidence artifact labels stable for easier review and publication reuse.

## Publication Prep Sequence (per PUBLICATIONS.md)

1. **Abstract** — PhD conference presentation (~300–500 words)
2. **Short paper** — ~4 pages, workshop or short paper format
3. **Full paper** — Complete journal/venue structure

All prepared in the phase directory, aligned with the frozen contract.

## When to Close vs. Defer

| Decision scope | Close at | Defer to |
|----------------|----------|----------|
| Backend modes, observable contract, support matrix | Phase level (ADRs) | — |
| Specific gate lowering for one circuit family | — | Task 3 mini-spec |
| Benchmark seed format for reproducibility | Phase level | — |
| Exact error message for unsupported gate X | — | Task 3 mini-spec |
