# Templates — Layer 2 mini-spec, Layer 3 delivery story, Layer 4 engineering task

Read during Step 4a when authoring a slice plan. Replace every bracketed field; keep no
sample titles or scenarios in the repo.

## Contents

- Layer 2: task / work-package mini-spec
- Layer 3: delivery story
- Layer 4: engineering task (red-first)

## Layer 2: task / work-package mini-spec

`docs/specs/milestones/<slug>/task-<n>/TASK_<n>_MINI_SPEC.md` — the contract for the
next cohesive work package. It may wrap one vertical slice, one Layer 1 goal, or a
small set of tightly coupled changes. Budget: 250 lines.

```markdown
# Task / Work Package [N]: [Title]

> **Status:** [planning | code-ready | shipped] · **Slice:** [<M#> slice <n>]
> **Traces:** [REQ-* / QA-* / ADR-* this work package serves]

## Required behavior
- [What must be true when done]
- [Mandatory interfaces and semantics]

## Unsupported behavior
- [Explicit exclusions and failure modes]
- [Documented error conditions]

## Acceptance evidence
- [Tests, types, or agreed checks]
- [Traceability to milestone acceptance criteria]

## Evidence matrix
| Trace id | Evidence type | Command / CI gate | Expected result | Owner artifact |
|----------|---------------|-------------------|-----------------|----------------|
| [REQ-001 / QA-001 / ADR-0001] | [unit / integration / contract / acceptance / fitness] | [command or CI job] | [pass condition] | [story / task / ADR] |

## Affected interfaces
- [API, config, CLI, event, workflow surfaces]
- [Breaking vs additive changes]
```

Optional additions when they earn their space: `Given / When / Then` wording; a short
assumptions / dependencies subsection when external coupling is significant; release or
rollback notes when the work changes production behavior.

Every evidence row needs a runnable command, a named CI gate, or an explicitly declared
non-executable evidence type (`doc review`, `sign-off`). A row that names nothing is an
evidence gap; `check_traceability.py` reports it.

## Layer 3: delivery story

`docs/specs/milestones/<slug>/task-<n>/DELIVERY_STORIES.md`. Budget: 200 lines.

```markdown
### Delivery story: [Behavioral outcome]

**Stakeholder / system value**
- [Why this behavior matters for the milestone objective]

**Given / When / Then**
- Given [initial contract state]
- When [action or trigger]
- Then [observable outcome and boundary behavior]

**Scope**
- In: [what this delivery story covers]
- Out: [explicit exclusions]

**Acceptance signals**
- [Signal 1: measurable or observable evidence]
- [Signal 2: negative or unsupported behavior handling]

**Traceability**
- Initial requirement(s): [REQ-001, REQ-002, … from INITIAL_REQUIREMENTS.md]
- Capability / quality attribute: [CAP-* / QA-* the requirement(s) serve]
- Milestone planning reference(s): [section or goal ids in DETAILED_PLANNING_*]
- ADR(s): [ids]
```

**Quality gate:** independent enough to verify in isolation, negotiable in
implementation detail, valuable, small enough for one cycle, and testable through
concrete acceptance signals.

## Layer 4: engineering task (red-first)

`docs/specs/milestones/<slug>/task-<n>/ENGINEERING_TASKS.md`. Budget: 300 lines.
Engineering tasks are TDD-shaped: the failing test comes first, so the acceptance tests
become the executable mirror of the spec.

```markdown
### Engineering Task: [Action-oriented title]

**Implements delivery story**
- [Delivery story title or reference]

**Change type**
- code | tests | docs | tooling

**Definition of done**
- [Concrete completion condition 1]
- [Concrete completion condition 2]

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing acceptance/unit test(s) that encode the task's done criteria and evidence-matrix rows **first**
- [ ] Run them and confirm they fail for the right reason (red)
- [ ] Implement the minimal change to make them pass (green)
- [ ] Refactor without changing behavior; re-run tests
- [ ] Update user-facing or developer docs if behavior or interfaces changed

**Evidence produced**
- [Test command, CI job, or checklist outcome]

**Risks / rollback**
- Risk: [known risk]
- Rollback / mitigation: [how to revert or isolate impact]
```

**Quality gate:** maps to one primary delivery-story outcome (or a clearly stated
subset), has objective done criteria, records risk and mitigation, and ties to
acceptance evidence without inventing out-of-process gates.

A Layer 4 task is code-ready when a code-generation pass can implement it with no
further design decision: done criteria are objective, the tests to write are named, the
evidence-matrix rows exist, and the `QA-*` fitness functions it must satisfy are cited.
