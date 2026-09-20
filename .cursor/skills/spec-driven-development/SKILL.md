---
name: spec-driven-development
description: Plans and implements one milestone under docs/specs as layered contracts — Layer 1 planning and ADRs, Layer 2 mini-spec, Layer 3 delivery stories, Layer 4 engineering tasks — slice by slice, with REQ-* evidence matrices, closeouts, handbacks, and current-state docs. Use to plan, decompose, implement, verify, or close milestone slices. Not for product vision, roadmap sequencing, or paper writing.
---

# Spec-driven development

The specification is the authoritative description of intent and behavior; code and
automated checks prove it. A spec stays short enough to read, debate, and update when
reality changes. Deliver **one roadmap milestone** through Layers 1–4, **slice by slice**,
where every slice is independently shippable: the extension builds, the named test lanes
are green, and the evidence bundles it claims regenerate. Do not skip layers for
non-trivial change; for a trivial change already covered by specs and tests, reduce
ceremony but still update the test or leave a brief note.

## Position in the stack

| Layer | Skill | Artifact |
|-------|-------|----------|
| Product | `create-product-statement` | `docs/specs/PRODUCT_STATEMENT.md` (`CAP-*`, `QA-*`) |
| Roadmap | `create-product-roadmap` | `docs/specs/ROADMAP.md` (`M#`, Now/Next/Later) |
| Milestone input | `create-initreq-for-sdd` | `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) |
| **Layers 1–4** | **this skill** | `milestones/<slug>/…` — plan, ADRs, slices, closeouts |

Traceability spine: `CAP-*/QA-* → M# → REQ-* → delivery story → engineering task →
evidence`. Every link stays intact, and `specs_check.sh` proves it. Full stack diagram and
shared glossary: `docs/sdd-skills-guide.md`. `docs/specs/` is the only spec root; the
phase trees under `docs/density_matrix_project/archive/` are frozen history. Terms: a
**Layer 2 mini-spec** is a work-package contract, not a Layer 4 engineering task; a
**delivery story** is a Layer 3 behavioral slice, not a `REQ-*`; a **product walking
skeleton** is the first roadmap milestone; a **slice tracer** is the first vertical slice
inside any milestone.

## Reading order and context budget

Read just-in-time. Pre-loading the whole milestone tree is what makes later slices drift.

| At this point | Read | Do not read |
|---------------|------|-------------|
| Step 1 (Layer 1) | `PRODUCT_STATEMENT.md`, the milestone row in `ROADMAP.md`, `INITIAL_REQUIREMENTS.md`, `ARCHITECTURE_OVERVIEW.md`, `TECH_STACK.md` | any `task-<n>/` artifact, the archived phase trees |
| Steps 2–3 (readiness) | the Layer 1 contract you just wrote | prior milestones' trees |
| Step 4a (plan a slice) | the Layer 1 contract, plus the **previous slice's `CLOSEOUT.md`** | previous slices' mini-specs, stories, or task files |
| Step 4b (generate code) | this slice's mini-spec, stories, and engineering tasks | the roadmap, the product statement |
| Milestone close | every slice `CLOSEOUT.md`, the Layer 1 acceptance criteria | slice-level task files |

Delegate wide discovery — codebase exploration, artifact audits — to a subagent and take
back the summary. Persist anything the next slice needs in a file, never in conversation:
after a slice closes, `task-<n>/CLOSEOUT.md` is the only thing that carried forward.

## The four layers

| Layer | Artifact | Answers |
|-------|----------|---------|
| 1 — milestone contract | `DETAILED_PLANNING_<MILESTONE_SLUG>.md`, `ADRS_<MILESTONE_SLUG>.md`, `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | what this milestone is and is not, prerequisites, success criteria, work packages as **goals**, decisions spanning work packages, the gap list and readiness verdict |
| 2 — work-package mini-spec | `task-<n>/TASK_<n>_MINI_SPEC.md` | required and unsupported behavior, acceptance evidence, affected interfaces, evidence matrix |
| 3 — delivery stories | `task-<n>/DELIVERY_STORIES.md` | observable behavior and researcher- or system-relevant outcomes, not internal chores |
| 4 — engineering tasks | `task-<n>/ENGINEERING_TASKS.md` | concrete red-first code, test, doc, benchmark, or evidence-pipeline work with objective done criteria |

If a decision affects multiple work packages or delivery stories, close it at milestone
level; if it affects one work package, it may live in that mini-spec. Never fragment a
cross-cutting choice across mini-specs. Templates: `references/templates-layer-2-4.md`;
paths and naming: `references/artifact-map.md`.

## Milestone workflow

Steps 1–3 and Step 4a are **spec-only** — no product code unless a spike is explicitly in
scope. Step 4b is code generation. Then the milestone closes.

### Step 1 — Layer 1 contract

Read `INITIAL_REQUIREMENTS.md` first when it exists and treat it as the authoritative
product intent and acceptance baseline until Layer 1 supersedes it. Lift into
`DETAILED_PLANNING_<MILESTONE_SLUG>.md`: in-scope / out-of-scope, success criteria,
assumptions, milestone-level acceptance from `REQ-*` plus NFRs, and traceability rows
`REQ-* → milestone goals`. Promote unresolved **open questions** into the
pre-implementation checklist as gaps. Merge the three-tier operational boundaries
(Always / Ask first / Never) into the milestone contract or agent runbook — they
complement, not replace, acceptance criteria. If no init-req file exists, derive the same
content from program-level sources or elicit it before locking Layer 1.

Break the milestone into **goals with acceptance criteria**, not step-by-step coding
instructions. Freeze the contracts the implementation must not contradict — support
matrix, numeric thresholds, reference baselines, unsupported-behavior policy. Apply
`references/practices-architecture.md` and `references/practices-testing.md`. Record
whether `ARCHITECTURE_OVERVIEW.md` / `TECH_STACK.md` must be updated by this milestone.
Do **not** split the whole milestone into Layer 3–4 upfront.

### Step 2 — readiness review

Decide whether Layer 1 is sufficient to start implementation and record a concise gap
list in the checklist: open items, missing decisions, ambiguous contracts, missing
current-state documentation.

### Step 3 — close the gaps

For each open item, ask, decide, or escalate, and record the decision and its trade-offs
in an ADR or in planning. Map each item to a concrete contract clarification. Re-read the
updated docs for internal consistency. End with an explicit **implementation-ready /
not-ready** verdict.

Then run one **adversarial critique pass** before locking Layer 1 (`sdd-critic` or
inline): which assumption would invalidate the most scope if wrong; which acceptance
criterion is least testable; which `QA-*` has no fitness function; which goal has no
evidence route; what belongs in a later milestone. Tighten or record every finding.

### Step 4 — deliver slice by slice

The first slice is the **slice tracer**: a deliberately thin end-to-end path that proves
the milestone's contracts, module boundaries, test route, and evidence route before
broadening — one thin slice that validates the new scientific object, not horizontal
infrastructure.

**Step 4a — plan the slice (spec-only).** For the current slice only: Layer 2 mini-spec
as needed, then Layer 3 delivery stories, then Layer 4 engineering tasks. Specify each
engineering task to a **code-ready** standard — objective done criteria, the tests to
write, evidence-matrix rows with the lane they run in, and the `QA-*` fitness functions it
must satisfy — precise enough that code generation needs no further design decision. If
the slice surfaces a cross-slice contract change, a new ADR, or a readiness gap, resolve
it in Layer 1 first. End with an explicit **code-ready / not-ready** verdict.

**Step 4b — generate code for the slice.** Implement the code-ready engineering tasks:
code and tests against the acceptance tests, evidence matrix, and `QA-*` fitness
functions, red-first. Ship the working chunk — builds, green, evidence regenerated,
rollback-aware — and record `task-<n>/CLOSEOUT.md` with status `shipped`, the acceptance
verdicts, and the reproduce commands. This pass makes **no** design, scope, contract, or
ADR decision. If a task is ambiguous, blocked, or reveals a cross-slice gap, stop: record
`task-<n>/STEP_4A_HANDBACK.md` with one question per gap (options, trade-offs,
consequences, owning authority), set `task-<n>/CLOSEOUT.md` to `implementation handback`,
and do not mark the slice shipped. Updating specs and current-state docs because reality
differed is planning work, not code generation.

After a slice ships — or its handback is disposed, re-issues code-ready, and ships —
return to Step 4a for the next slice. Do not pre-plan the remaining slices.

### Milestone close

When every slice is delivered and the outcome is met, produce
`<MILESTONE_ID>_CLOSEOUT.md`: slices delivered, final acceptance status, the `REQ-*`
evidence matrix, learnings, deferred items, and the roadmap handoff. Update
`ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` for any shipped change to architecture,
stack, build, commands, or evidence pipelines. Confirm any `CHANGE_CONTROL.md` sign-off is
recorded. Then hand control back to `create-product-roadmap` for revalidation. If a
learning invalidated a core product assumption, that escalates to
`create-product-statement`. A paper, abstract, or talk drawn from the milestone consumes
the closeout's evidence matrix; it is not a spec artifact and does not live in `docs/specs/`.

Templates for all four close and governance artifacts:
`references/templates-closeout.md`.

## The planning / code-generation seam

All planning — the Layer 1 contract *and* each slice's just-in-time Layer 2/3/4 — is
spec-only work owned by the planning role. Code generation is a separate, narrower pass:
implement already-specified engineering tasks and nothing more. The seam sits **inside
Step 4**, between 4a and 4b. Three committed subagents carry the roles, so the boundary is
a capability rather than a promise:

| Subagent | Enforcement | Owns |
|----------|-------------|------|
| `.cursor/agents/sdd-planner.md` | `readonly: true` — cannot write code | Layers 1–4 planning, verdicts |
| `.cursor/agents/sdd-implementer.md` | write-enabled, must not edit `docs/specs/**` | Step 4b code and tests |
| `.cursor/agents/sdd-critic.md` | `readonly: true` | adversarial critique before a verdict |

Delegate with the Task tool: the planner returns a plan the parent writes to files, and the
implementer reads those files. Never carry the handoff in chat memory. Models are pinned in
the subagent files, not here.

## Size budgets

An artifact past its budget is a slice that is too big. Split it; do not append. Open
every artifact with a context header of at most ten lines — status, milestone or slice,
scope, traces — so a partial read is still decision-useful.

| Artifact | Lines | Artifact | Lines |
|----------|-------|----------|-------|
| `INITIAL_REQUIREMENTS.md` | 300 | `DELIVERY_STORIES.md` | 200 |
| `DETAILED_PLANNING_*` | 400 | `ENGINEERING_TASKS.md` | 300 |
| `ADRS_*` | 400 | `CLOSEOUT.md` (slice) | 200 |
| `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | 250 | `<MILESTONE_ID>_CLOSEOUT.md` | 250 |
| `TASK_<n>_MINI_SPEC.md` | 250 | `STEP_4A_HANDBACK.md` | 200 |

## Verify

Run before any readiness, code-ready, shipped, or delivered claim. The linters are
stdlib-only; the wrapper picks the `qgd` interpreter when conda is available.

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh            # errors must be zero
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict   # unwaived warnings become errors
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/<slug>
```

Fix findings; do not silence them. A finding you intend to keep is waived in
`docs/specs/.sdd-lint.json` **with a reason**, never on an in-flight milestone.
`--no-waivers` shows the debt the waivers suppress.

## Completion criteria

- **Layer 1 done:** the four Layer 1 files exist, the checklist states
  implementation-ready or not-ready, `specs_check.sh` is error-free, and every `REQ-*`
  maps to a milestone goal.
- **Slice code-ready:** mini-spec (when warranted), delivery stories, and engineering
  tasks exist for this slice only; each task names its tests, evidence rows and lanes, and
  `QA-*` fitness functions; the verdict line is explicit.
- **Slice shipped:** acceptance tests and fitness functions green in the named lanes, the
  extension builds, `CLOSEOUT.md` status `shipped` with reproduce commands, current-state
  docs updated if reality changed, `specs_check.sh` error-free.
- **Milestone delivered:** `<MILESTONE_ID>_CLOSEOUT.md` covers every `REQ-*` with evidence,
  any `CHANGE_CONTROL.md` sign-off is recorded, current-state docs are current, and control
  is handed to `create-product-roadmap`.

## Gotchas in this repo

- Tests, benchmarks, and examples run in the **`qgd` conda environment**
  (`conda run -n qgd --no-capture-output pytest …`). C++ or CMake changes need a rebuild
  (`clean-rebuild` skill); `test-density-matrix` runs the suites; `TECH_STACK.md` lists
  the lanes.
- Evidence rows name their lane: fast pytest (`tests/density_matrix`, `tests/partitioning`,
  `tests/VQE`, `-m "not slow"`), `slow`, a benchmark evidence pipeline
  (`benchmarks/density_matrix/*/validation_pipeline.py`), the optional C++ tests
  (`QGD_CTEST=1`), or the Qiskit Aer external reference.
- The sequential `NoisyCircuit` executor is the exact baseline every partitioned, fused,
  or new backend path is validated against; Aer is the external reference. A `QA-*` about
  exactness becomes a fitness test against that baseline.
- The delivered phase trees under `docs/density_matrix_project/archive/` use an earlier
  convention. Read them for history; never extend them or copy their naming into
  `docs/specs/` (mapping: `references/artifact-map.md`). Revisions are forward-only.

## References

Load only what the current step needs:

| Read this | When |
|-----------|------|
| `references/artifact-map.md` — canonical paths, naming, budgets, current-state docs, legacy mapping | creating or naming an artifact |
| `references/templates-layer-2-4.md` — mini-spec, delivery story, engineering task | Step 4a |
| `references/templates-closeout.md` — slice and milestone closeout, handback, change control | slice close, milestone close, or handback |
| `references/practices-architecture.md` — DDD, hexagonal, boundary maps, C4, ADR rubric | writing Layer 1 or an ADR |
| `references/practices-testing.md` — BDD, test pyramid, evidence matrix, fitness functions, Definition of Ready/Done | writing acceptance, or gating a slice |
| `references/rubrics.md` — planning and mini-spec rubrics, principles, anti-patterns | before a readiness or code-ready verdict |

## Do not

- Put implementation code in milestone planning, unless the team allows an interface sketch.
- Make design, scope, contract, or ADR decisions during Step 4b, or invent acceptance
  criteria there. Stop and hand back instead.
- Write paper, abstract, or slide surfaces inside `docs/specs/`, or gate a slice on them.
- Answer a question by starting the workflow: answer from the specs and the codebase.
