# Spec-driven development (SDD) skills — how to use

The density-matrix track of SQUANDER plans and builds from **specifications**, using four
layered project skills in `.cursor/skills/`. Each turns intent into the input for the next,
top-down, and a feedback loop re-plans the roadmap after every milestone. All artifacts live
under `docs/specs/`.

This guide is the **canonical** home for the stack diagram, the shared glossary, and the
end-to-end flow. Each `SKILL.md` points here rather than restating them, so there is one
definition to keep true.

Throughout, **`specs_check.sh`** is shorthand for
`bash .cursor/skills/spec-driven-development/scripts/specs_check.sh`.

## Where this repository stands

Phases 1, 2, 3 and 3.1 were delivered before this stack existed, under a phase-based
convention. Their record is archived read-only under `docs/density_matrix_project/archive/`
(see its `README.md`). The baseline for the new convention is in place:
`docs/specs/ARCHITECTURE_OVERVIEW.md`, `docs/specs/TECH_STACK.md`, and
`docs/specs/.sdd-lint.json`. The next steps, in order, are the product statement, the
roadmap (recording Phases 1–3.1 as Delivered milestones), and the first *Now* milestone's
requirements — the prompts are in *End-to-end flow* below.

## The stack

```
create-product-statement →  docs/specs/PRODUCT_STATEMENT.md   (North Star: CAP-*, QA-*)               [stable]
create-product-roadmap   →  docs/specs/ROADMAP.md             (outcome milestones M#, Now/Next/Later) [living; revalidated per milestone]
create-initreq-for-sdd   →  docs/specs/milestones/<slug>/INITIAL_REQUIREMENTS.md   (REQ-*)
spec-driven-development  →  docs/specs/milestones/<slug>/…    (built slice by slice; every slice deployable)
                         →  docs/specs/ARCHITECTURE_OVERVIEW.md + TECH_STACK.md    (current-state self-documentation)
                            └─ after a milestone ships → revalidate remaining ROADMAP.md ↺
```

**Traceability spine:** `CAP-*/QA-* → M# → REQ-* → delivery story → engineering task →
evidence`. Every milestone, requirement, story, and test traces up to a product capability
(`CAP-*`) or quality attribute (`QA-*`). `specs_check.sh` proves the spine mechanically;
see *Verifying spec work* below.

## Glossary (shared by all four skills)

| Term | Meaning |
|------|---------|
| **Vision** | One or two sentences naming the change the product brings about. |
| **Product capability** (`CAP-001`, …) | A high-level, durable product requirement: an outcome the product must enable. The traced unit at the product layer. |
| **Quality attribute** (`QA-001`, …) | A cross-cutting, product-level non-functional expectation, written as a scenario with a response measure. |
| **Milestone** (`M4`, `M4A`, `M5`, …) | A shippable outcome slice of the product, with a measurable target and a filesystem-safe **`milestone-slug`** reused by every downstream artifact. Delivered Phases 1–3.1 are recorded as milestones too, pointing at the archive. |
| **Outcome** | The measurable change a milestone creates — a result, not a list of features. |
| **Now / Next / Later** | Priority horizons. *Now* is committed and detailed; *Next* is validated and upcoming; *Later* is deliberately low-precision hypotheses. |
| **Requirement** (`REQ-001`, …) | Product intent plus acceptance, in `INITIAL_REQUIREMENTS.md`. **Not** a Layer 3 artifact. |
| **Layer 2 mini-spec** | A small work-package contract for the next implementation unit. It may wrap one vertical slice, one Layer 1 goal, or a tightly coupled task group. **Not** the same as a Layer 4 engineering task. |
| **Delivery story** | A **Layer 3** behavioral slice for implementation: observable outcome, in/out scope, acceptance signals, traceability. |
| **Engineering task** | A **Layer 4** implementation unit under a delivery story: code, tests, docs, tooling, migration, or operational change with objective done criteria. Red-first. |
| **Product walking skeleton** | The first roadmap milestone: a deployable product-thin path that proves the end-to-end architecture and the riskiest integration assumptions, and establishes the first useful current-state docs. |
| **Slice tracer** | The first vertical slice inside any milestone: a deliberately thin path that proves the milestone's contracts, adapters, test route, and deployment route before broadening. |
| **Current-state architecture docs** | `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md`: living, descriptive references for what is currently true. They summarise reality and link ADRs; they never replace decisions, requirements, or roadmap planning. |
| **Evidence matrix** | The table pairing each `REQ-*`/`QA-*`/decision with a test type, a runnable command or CI gate, an expected result, and an owning artifact. |
| **Fitness function** | An automated check enforcing a `QA-*` continuously — performance budget, dependency-rule test, security gate, containment test. |
| **Revalidation** | Re-checking the remaining roadmap after a milestone, using its closeout evidence against the stable product statement. |
| **User-story phrasing** | Optional *"As a … I want …"* wording **inside a requirement**. Do not confuse it with a delivery story. |

## Artifact placement

| Artifact | Path |
|----------|------|
| Product statement, roadmap, current-state docs | `docs/specs/` |
| Requirements baseline | `docs/specs/milestones/<slug>/INITIAL_REQUIREMENTS.md` |
| Layer 1 contract | `docs/specs/milestones/<slug>/DETAILED_PLANNING_<MILESTONE_SLUG>.md`, `ADRS_<MILESTONE_SLUG>.md`, `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` |
| Per-slice Layers 2–4 | `docs/specs/milestones/<slug>/task-<n>/TASK_<n>_MINI_SPEC.md`, `DELIVERY_STORIES.md`, `ENGINEERING_TASKS.md` |
| Slice record | `task-<n>/CLOSEOUT.md`, or `task-<n>/STEP_4A_HANDBACK.md` when code generation hits a contract gap |
| Milestone record | `<MILESTONE_ID>_CLOSEOUT.md`, plus `CHANGE_CONTROL.md` when a deviation needs governance sign-off |
| Lint budgets and waivers | `docs/specs/.sdd-lint.json` |

One tree is historical and deliberate: the delivered Phases 1–3.1 live under
`docs/density_matrix_project/archive/phases/` with their program-level plan and ADRs under
`archive/planning/`, using the earlier `TASK_<n>_STORIES.md` /
`STORY_<k>_IMPLEMENTATION_PLAN.md` naming and per-phase paper surfaces. Revisions are
**forward-only**: delivered phases keep the convention they shipped under and are never
extended. The legacy-to-current mapping lives in
`.cursor/skills/spec-driven-development/references/artifact-map.md`.

Papers, abstracts, and slides are **not** spec artifacts. They consume a milestone
closeout's evidence matrix and live outside `docs/specs/`.

## Size budgets

An artifact that no longer fits in working memory stops being read and starts being skimmed,
so every artifact type has a line budget enforced as a warning by `specs_check.sh --strict`:
300 for `INITIAL_REQUIREMENTS.md`, 400 for planning and ADRs, 250 for the readiness checklist
and each mini-spec, 200 for delivery stories, 300 for engineering tasks, 200 for a slice
closeout, 250 for a milestone closeout. **Over budget means the slice is too big** — split
it, do not append. Budgets apply to every milestone under `docs/specs/`; the archived phases
predate them (their Layer 1 files ran to 1,000–1,300 lines, which is the failure mode the
budgets exist to prevent).

Every artifact opens with a context header of at most ten lines — status, milestone or slice,
scope, traces — so a partial read is still decision-useful.

## When to use which skill

| You want to… | Use | Produces |
| ------------ | --- | -------- |
| Define the durable vision, capabilities, and quality bars | [`create-product-statement`](../.cursor/skills/create-product-statement/SKILL.md) | `PRODUCT_STATEMENT.md` |
| Turn that into prioritized, measurable milestones | [`create-product-roadmap`](../.cursor/skills/create-product-roadmap/SKILL.md) | `ROADMAP.md` |
| Spec one milestone before building it | [`create-initreq-for-sdd`](../.cursor/skills/create-initreq-for-sdd/SKILL.md) | `milestones/<slug>/INITIAL_REQUIREMENTS.md` |
| Plan and implement that milestone slice by slice | [`spec-driven-development`](../.cursor/skills/spec-driven-development/SKILL.md) | planning, ADRs, checklist, delivery stories, tasks, code, slice and milestone closeouts, change control, handbacks |
| Self-document the actual architecture and stack | [`spec-driven-development`](../.cursor/skills/spec-driven-development/SKILL.md) | `ARCHITECTURE_OVERVIEW.md`, `TECH_STACK.md` |

## End-to-end flow

Each step has an example prompt to copy, edit, and paste into Cursor.

1. **Product statement** — run `create-product-statement` once to set the North Star. It
   changes rarely. It also seeds or hands off the current-state docs without mixing
   architecture decisions into product intent.

  ```text
   Use create-product-statement. Read docs/density_matrix_project/README.md,
   RESEARCH_ALIGNMENT.md, the archived program plan and ADRs under
   docs/density_matrix_project/archive/planning/, and the current-state
   docs/specs/ARCHITECTURE_OVERVIEW.md and TECH_STACK.md. Draft
   docs/specs/PRODUCT_STATEMENT.md for the density-matrix track: the research outcome the
   software enables, CAP-* as researcher-reachable outcomes, QA-* as measurable scenarios
   against the sequential NoisyCircuit baseline or Qiskit Aer. Ask me clarifying questions
   first, then validate it with a PR-FAQ and the adversarial critique pass. Do not put
   phase sequencing, architecture, or publication planning into it.
  ```

2. **Roadmap** — run `create-product-roadmap` to decompose it into outcome milestones,
   ordered Now/Next/Later. The **first *Now* milestone is the product walking skeleton** of
   the new convention; delivered Phases 1–3.1 are recorded, not re-planned.

  ```text
   Use create-product-roadmap. From docs/specs/PRODUCT_STATEMENT.md, the archived
   PLANNING.md (dependency order, decision gates, scope cuts) and RESEARCH_ALIGNMENT.md,
   draft docs/specs/ROADMAP.md. Record Phases 1, 2, 3 and 3.1 as Delivered milestones
   with a one-line outcome each and a link into docs/density_matrix_project/archive/.
   Then lay out Now/Next/Later milestones, each tracing to CAP-*/QA-*, with the first Now
   milestone a product walking skeleton that proves one thin end-to-end path through the
   new convention. Run the critique pass, then propose the sequencing rationale.
  ```

3. **Per milestone** — for the current *Now* milestone, run `create-initreq-for-sdd`.

  ```text
   Use create-initreq-for-sdd for the first "Now" milestone in docs/specs/ROADMAP.md.
   Produce its INITIAL_REQUIREMENTS.md with REQ-* traced to CAP-*/QA-*, EARS/BDD
   acceptance including a negative scenario per requirement (the density path is strict:
   name the rejected input and the error), NFRs with response measures and the lane that
   proves them, and a milestone glossary. Ask clarifying questions before drafting, then
   run the adversarial critique pass and specs_check.sh.
  ```

4. **Milestone technical planning** *(spec-only)* — run `spec-driven-development` Steps 1–3
   for the Layer 1 contract. This step may identify the first slice-tracer goal, but must not
   write product code or split the whole milestone into Layer 3–4.

  ```text
   Use spec-driven-development for milestone <slug>, Steps 1-3 only (spec-only; no product
   code). Read docs/specs/PRODUCT_STATEMENT.md, docs/specs/ROADMAP.md,
   docs/specs/milestones/<slug>/INITIAL_REQUIREMENTS.md, and the current-state
   ARCHITECTURE_OVERVIEW.md / TECH_STACK.md. Produce or update the Layer 1 contract:
   DETAILED_PLANNING_<MILESTONE_SLUG>.md, ADRS_<MILESTONE_SLUG>.md, and
   PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md. Close or record gaps, identify the first
   slice-tracer goal and its evidence route, run the critique pass and specs_check.sh,
   and end with an explicit implementation-ready / not-ready verdict. Do not split the
   milestone into all Layer 3 delivery stories or Layer 4 engineering tasks. Ask questions
   whenever a decision needs information; never resolve a trade-off silently.
  ```

5. **Slice tracer — plan** *(spec-only)* — run Step 4a for the first vertical slice only.

  ```text
   Use spec-driven-development for milestone <slug>, Step 4a, first vertical slice only
   (spec-only; no product code). Read the Layer 1 contract first, plus
   ARCHITECTURE_OVERVIEW.md and TECH_STACK.md. Treat the implementation-ready verdict as
   the gate. Produce the slice-tracer plan: Layer 2 mini-spec if warranted, then Layer 3
   delivery stories for this slice only, then Layer 4 engineering tasks, each with
   objective done criteria, the tests to write, evidence-matrix rows, and QA-* fitness
   functions - precise enough to implement without further design decisions. Keep each
   artifact inside its size budget. End with an explicit code-ready / not-ready verdict.
   If the slice needs a cross-slice contract or ADR change, resolve it in Layer 1 first.
  ```

6. **Slice tracer — implement** *(code generation only)* — once the slice is code-ready,
   generate code against the Layer 4 tasks. No design, scope, contract, or ADR decisions.

  ```text
   Use spec-driven-development for milestone <slug>, Step 4b for the first vertical slice.
   Read the slice plan first: the Layer 2 mini-spec, Layer 3 delivery stories, and Layer 4
   engineering tasks, plus the Layer 1 contract and TECH_STACK.md. Work red-first: write
   the failing tests the task names, then implement to green. Satisfy the evidence matrix
   and the named QA-* fitness functions and keep the slice deployable. Do not redesign,
   re-scope, or change contracts/ADRs - if a task is ambiguous or reveals a cross-slice
   gap, stop and hand back questions for Layer 1-4 planning. Then record
   task-<n>/CLOSEOUT.md and run specs_check.sh.
  ```

7. **Remaining slices** — after the tracer ships, continue one vertical slice at a time:
   just-in-time Layer 2/3/4 planning to a code-ready verdict, then code generation. Do not
   pre-plan the remaining slices.

  ```text
   Continue spec-driven-development for milestone <slug>; the slice tracer has shipped.
   Choose the next vertical slice by value x risk toward the milestone acceptance criteria.
   Read the Layer 1 contract and the previous slice's CLOSEOUT.md - not the previous
   slices' mini-specs.

   Planning pass (spec-only): expand Layer 2/3/4 for this slice only, tracing REQ-* ->
   CAP-*/QA-* and naming the tests, evidence-matrix rows, and QA-* fitness functions. End
   with a code-ready verdict.

   Code-generation pass: implement those engineering tasks red-first against the acceptance
   tests, evidence matrix, and fitness functions; keep the slice deployable. Stop and hand
   back if a task is ambiguous or needs a cross-slice decision.

   Update ARCHITECTURE_OVERVIEW.md / TECH_STACK.md if shipped reality changed, run
   specs_check.sh, then report what shipped and which slices remain.
  ```

   Repeat until the milestone outcome is met.

8. **Close and revalidate** — produce the milestone closeout, then hand back to the roadmap.

  ```text
   Milestone <slug> is complete. Use spec-driven-development to produce
   <MILESTONE_ID>_CLOSEOUT.md (slices delivered, final acceptance status, REQ-* evidence
   matrix, learnings, deferred items), confirm ARCHITECTURE_OVERVIEW.md and TECH_STACK.md
   reflect shipped reality, and run specs_check.sh. Then use create-product-roadmap to
   revalidate the remaining roadmap: record the outcome (achieved/partial/missed), update
   assumptions and risks from what we learned, re-sequence Next/Later, and promote the next
   milestone to Now.
  ```

If a milestone's evidence breaks a core assumption, escalate to the product statement:

```text
Milestone <slug> invalidated assumption <A> in docs/specs/PRODUCT_STATEMENT.md.
Use create-product-statement to review and update it, then re-run create-product-roadmap
revalidation so the roadmap realigns with the new North Star.
```

## The feedback loop

- The **roadmap is living**: revalidated after every milestone — re-sequence, re-scope,
  split, merge, add, drop.
- The **product statement is stable**: revisited only when a milestone's evidence invalidates
  a core assumption, escalated up from revalidation.
- The **current-state docs** are updated whenever shipped work changes boundaries, topology,
  integrations, commands, runtime, tooling, or ADR status.

## Roles: planning versus code generation

The stack separates **spec/planning** work from **code generation**, and every handoff is a
durable file under `docs/specs/`. Planning owns the Layer 1 contract *and* each slice's
just-in-time Layer 2/3/4. Code generation implements engineering tasks that are already
fully specified — and nothing more.

The seam sits **inside Step 4**, between a slice's code-ready Layer 4 plan (4a) and its
implementation (4b). It is carried by three committed subagents, so the boundary is a
capability rather than a promise:

| Subagent | Enforcement | Owns |
|----------|-------------|------|
| [`.cursor/agents/sdd-planner.md`](../.cursor/agents/sdd-planner.md) | `readonly: true` — cannot write code | all planning, all verdicts |
| [`.cursor/agents/sdd-implementer.md`](../.cursor/agents/sdd-implementer.md) | write-enabled; must not edit `docs/specs/**` | Step 4b code and tests |
| [`.cursor/agents/sdd-critic.md`](../.cursor/agents/sdd-critic.md) | `readonly: true` | adversarial critique before a verdict |

Model and effort are pinned in those files, not in the skills — a `SKILL.md` should not
carry a model name, because model names go stale. The planner defaults to a
high-reasoning-effort model and the implementer to a faster execution model: pay for
reasoning once, then spend cheaper tokens executing an unambiguous plan. Cursor falls back to
an available model when a pinned one is not on your plan, so treat the pins as intent.

You can also drive the seam manually with the model picker: plan on a
high-effort reasoning model, switch to a coding model only once the slice is code-ready, and
switch back immediately afterwards. Either way, do not let chat memory carry the handoff —
the files are the interface.

```text
Orchestrate milestone <slug> with the sdd-planner and sdd-implementer subagents, handing
off only through docs/specs artifacts.

1. sdd-planner: spec-driven-development Steps 1-3 for milestone <slug>. It must produce the
   Layer 1 contract, return an implementation-ready / not-ready verdict, and write no code.
2. Per slice (tracer first, then one at a time): sdd-planner expands Layer 2/3/4 for that
   slice only, ending with a code-ready verdict. Optionally run sdd-critic first.
3. Once code-ready: sdd-implementer implements that slice's engineering tasks red-first -
   code and tests only, slice deployable, evidence matrix and QA-* fitness functions
   satisfied. If a task is ambiguous or needs a cross-slice change, it stops and hands back.
4. After the milestone outcome is met: sdd-planner runs create-product-roadmap revalidation.

Each agent reads docs/specs as its source of truth; do not rely on chat memory between them.
```

## Verifying spec work

The spine and the artifact structure are checked mechanically. Run these before claiming
spec work is complete:

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh            # spine + structure (errors only)
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict   # every unwaived warning becomes an error
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --no-waivers
```

The wrapper runs `check_artifacts.py` and `check_traceability.py` (stdlib-only) with the
`qgd` conda interpreter when available, or `python3` otherwise (`SDD_PYTHON` overrides).
`--no-waivers` shows the debt — size budgets, stale evidence commands, legacy naming — that
the recorded waivers suppress.

`specs_check.sh` reports, among other things: a `REQ-*` citing no `CAP-*`/`QA-*`, a `REQ-*`
no delivery story picks up, a delivery story citing no `REQ-*`, an evidence row naming no
runnable command or lane, a `tests/…`, `benchmarks/…`, or `examples/…` path that no longer
exists, a milestone closeout contradicting the files on disk, and a missing Layer 1 or slice
artifact. Scope it to one milestone by passing the directory as an argument.

The product suites themselves (`pytest` lanes, benchmark evidence pipelines, optional C++
tests, Qiskit Aer reference) are listed in `docs/specs/TECH_STACK.md` and run in the `qgd`
conda environment; the `test-density-matrix` skill runs them end to end.

Fix findings rather than silencing them. A finding you intend to keep is waived in
`docs/specs/.sdd-lint.json` **with a reason**, and never on an in-flight milestone.

Two hooks in `.cursor/hooks.json` back this up: `afterFileEdit` appends an audit line when a
`docs/specs/` artifact changes, and `stop` re-runs the linters when a session touched
`docs/specs/` and returns a follow-up message if the spine is broken. Both fail open.

## Invoking the skills

Name the skill ("use `create-product-roadmap`") or just describe the task — skills
auto-apply from their descriptions, which is why each description states its trigger *and*
its boundary against its neighbours. Editing anything under `docs/specs/` also activates the
`spec-driven-docs-specs` rule, which routes each file to its owning skill. Editing the SDD
skills or this guide activates `spec-driven-framework-maintenance`.

All project skills live in `.cursor/skills/`. One skill name has exactly one registered copy:
two copies make routing ambiguous. Add project context as a rule under `.cursor/rules/`,
never as a fork. The two non-SDD skills, `clean-rebuild` and `test-density-matrix`, are the
build and test procedures that Step 4b and the evidence lanes rely on.

## If you know another SDD toolchain

The vocabulary maps closely onto the tools that converged on this workflow:

| Here | GitHub Spec Kit | AWS Kiro |
|------|-----------------|----------|
| `AGENTS.md` + `.cursor/rules/` | `constitution` | steering files |
| `PRODUCT_STATEMENT.md` + `ROADMAP.md` | (no equivalent — Spec Kit starts per feature) | (no equivalent) |
| `INITIAL_REQUIREMENTS.md` (`REQ-*`, EARS/BDD) | `/specify` → `spec.md` | `requirements.md` (EARS) |
| Layer 1: `DETAILED_PLANNING_*`, `ADRS_*` | `/plan` → `plan.md` | `design.md` |
| Layer 3/4: delivery stories, engineering tasks | `/tasks` → `tasks.md` | `tasks.md` |
| Step 4b code generation | `/implement` | agent execution |
| Adversarial critique pass | `/clarify`, `/analyze`, `/checklist` | — |
| Evidence matrix + `QA-*` fitness functions | — | hooks |
| Milestone closeout → roadmap revalidation | — | — |

The two rows with no counterpart are the point of this stack: evidence that is executable,
and a plan that is re-derived from what shipped.

## What's baked in

Current SDD practice (EARS/BDD acceptance, living specs, short specs, explicit out-of-scope,
a clarification pass before freeze) plus battle-tested architecture, design, and testing
practice: **DDD** (ubiquitous language, bounded contexts, boundary maps), **clean/hexagonal
architecture** (dependency rule, ports and adapters, vertical slices), **BDD and
specification by example**, a **test pyramid** with **TDD** and contract tests, **evidence
matrices** and `QA-*` **fitness functions**, **ADRs** with C4 diagrams,
product-walking-skeleton-first roadmaps, slice-tracer-first implementation, and
self-documenting current-state references. Layered on top: progressive disclosure in the
skills themselves, deterministic spine checks, per-artifact size budgets, and a
role-and-permission seam between planning and code generation. See each `SKILL.md` for depth.

## Revision history

- **rev E (this repository)** — The four-skill stack was adopted for the SQUANDER
  density-matrix track, replacing the phase-based `spec-driven-development` skill that
  delivered Phases 1–3.1. Those phases and the program-level plan were archived read-only
  under `docs/density_matrix_project/archive/`; `docs/specs/` became the spec root with the
  current-state docs seeded from the former `ARCHITECTURE.md`. `make specs-check` became
  `specs_check.sh` (no root Makefile here); evidence lanes were redefined around the `qgd`
  conda environment, the sequential `NoisyCircuit` baseline, and the benchmark evidence
  pipelines; publication surfaces left the spec tree. The skill-shape evals runner and the
  framework fitness test of the reference stack were not ported.
- **rev D (reference stack)** — Skills restructured for progressive disclosure: each
  `SKILL.md` is a router with per-step `references/`, descriptions are routing keys with
  explicit boundaries, and revision histories moved to per-skill `CHANGELOG.md`. The
  traceability spine, artifact structure, and size budgets became deterministic checks and
  hooks. The planning / code-generation seam became three committed subagents under
  `.cursor/agents/`. This guide became the canonical home for the stack diagram and
  glossary; `AGENTS.md` was added for non-Cursor tools.
- **rev C (reference stack)** — The Layer 4 engineering-task template became red-first
  (TDD), and `create-initreq-for-sdd` gained an adversarial critique pass.
- **rev B (reference stack)** — Documented the named milestone-close and governance
  artifacts: `<MILESTONE_ID>_CLOSEOUT.md`, `task-<n>/CLOSEOUT.md`,
  `task-<n>/STEP_4A_HANDBACK.md`, and `CHANGE_CONTROL.md`. `create-initreq-for-sdd` gained
  an update/evolve path; `create-product-roadmap` began consuming the milestone closeout.
