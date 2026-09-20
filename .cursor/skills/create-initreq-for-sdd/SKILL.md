---
name: create-initreq-for-sdd
description: Creates or evolves a milestone's INITIAL_REQUIREMENTS.md — REQ-* requirements with EARS/BDD acceptance criteria, NFR response measures, a milestone glossary, and always/ask/never agent boundaries. Use to write, tighten, or evolve requirements before technical planning starts. Not for Layer 1 planning, ADRs, slices, or roadmap sequencing.
---

# Create initial requirements for SDD

Produce the **requirements baseline for one milestone**: short enough to load reliably into
an AI context, rich enough to drive implementation and verification, and aligned with
stakeholder intent — the "why" — without prematurely fixing the "how".

**Output:** `docs/specs/milestones/<milestone-slug>/INITIAL_REQUIREMENTS.md`, roughly 300
lines or fewer, carrying `REQ-*` requirements with stable ids.

Use a **PRD/SRS hybrid**: user-centric goals and journeys plus machine-friendly, testable
statements and explicit guardrails. A traditional PRD alone is too narrative for machines;
a cold SRS alone strips out intent.

**First iteration means "this milestone", not the whole product.** Scope the one milestone
handed off from the roadmap. `spec-driven-development` then delivers it slice by slice.
Avoid a monolithic up-front spec: large instruction sets dilute attention and invite the
model to ignore constraints.

## Position in the stack

| Layer | Skill | Artifact |
|-------|-------|----------|
| Product | `create-product-statement` | `docs/specs/PRODUCT_STATEMENT.md` (`CAP-*`, `QA-*`) |
| Roadmap | `create-product-roadmap` | `docs/specs/ROADMAP.md` (`M#`, `milestone-slug`) |
| **Milestone input** | **this skill** | `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) |
| Layers 1–4 | `spec-driven-development` | `milestones/<slug>/…` |

**Upstream:** the milestone row in `ROADMAP.md` and the capabilities, quality attributes,
and glossary in `PRODUCT_STATEMENT.md`. If neither exists, derive equivalent intent from the
user or from `docs/` before drafting, and suggest running the product and roadmap skills.
**Downstream:** this is the first consumable input for `spec-driven-development`, which
seeds Layer 1 from it.

Reuse the **exact `milestone-slug`** from the roadmap so the handoff is mechanical. Create
the milestone directory when writing the file. Full stack diagram and shared glossary:
`docs/sdd-skills-guide.md`.

**You are not duplicating SDD.** Stay at product intent and testable behavior. Do not author
`DETAILED_PLANNING_*`, ADR bodies, or task mini-specs here unless the user explicitly wants
a combined pass.

## Requirement vs delivery story

In `spec-driven-development`, "story" means a **Layer 3 delivery story**: a behavioral slice
used for implementation planning. In this skill, do not use "story" for that. Use
**requirement** with a stable id like `REQ-001`: a traced, testable unit of product intent.

You may still phrase a requirement as *"As a … I want … so that …"* — that is user-story
wording, not an SDD delivery story. One requirement may become several delivery stories, and
several requirements may be implemented in one delivery story. `REQ-*` ids must survive the
handoff either way.

## Principles

1. **Start high-level; let the agent expand.** The human supplies a tight product brief —
   problem, users, outcomes, boundaries. Detailed `REQ-*` and acceptance criteria may be
   drafted in a follow-up, still bounded to this milestone.
2. **What and why first; defer how.** Capture journeys, success criteria, and constraints.
   Exclude new stack choices, schema design, and architecture unless they are already fixed
   organizational standards or current-state facts that constrain this milestone. New
   decisions belong in Layer 1 planning and ADRs.
3. **Structured, testable acceptance.** Prefer EARS and BDD `Given / When / Then` so "done"
   is unambiguous and checkable, and precise enough to seed the downstream evidence matrix.
4. **Three-tier operational boundaries.** Define what the implementation agent must always
   do, must ask before doing, and must never do.
5. **Clarification before freeze.** Run a short elicitation pass on personas, edge cases,
   failure modes, data rules, and compliance before treating the document as final.
6. **Speak the ubiquitous language.** Reuse and extend the product glossary. The same term
   must mean the same thing in requirements, acceptance, tests, and code.

## Workflow

**1. Ingest, elicit, clarify.** Read the target milestone in `ROADMAP.md` — outcome,
measure, `CAP-*`/`QA-*` traces, slug, dependencies, what ships — and the relevant
capabilities, quality attributes, and glossary in `PRODUCT_STATEMENT.md`. Read
`ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` when present, for current-state constraints
and existing commands; do not copy their content into this file. If they are absent and this
is the first product-walking-skeleton milestone, note that `spec-driven-development` must
create them.

Write no application code. If intent, scope, or failure behavior is unclear, ask **3–7
concrete questions**: personas, the primary journey, what is out of scope, error behavior,
data sensitivity, success metrics, integration touchpoints. Prefer waiting for answers; if
told to proceed, state assumptions in an **Assumptions** subsection.

**2. Draft.** Write the artifact using the exact section order in
`references/document-structure.md`.

**3. Critique, then validate.** Run the adversarial pass and take the checkpoint question to
the user: `references/critique-and-update.md`. Then state the handoff to
`spec-driven-development`.

**Updating an existing baseline** follows the same reference: preserve `REQ-*` ids, mark
superseded requirements rather than deleting them, re-walk the spine, and append a Change
log entry.

## Completion criteria

- The file exists at `docs/specs/milestones/<slug>/INITIAL_REQUIREMENTS.md`, using the
  roadmap's exact slug, with all seven sections.
- The milestone id and its `CAP-*`/`QA-*` traces appear in the header.
- Every `REQ-*` has a stable id, an upstream citation, BDD or EARS acceptance, and at least
  one negative or error scenario.
- Every `QA-*`-realizing NFR carries a response measure or an owned `[confirm]` marker.
- Operational boundaries state Always / Ask first / Never.
- Open questions are written so each can be closed by a decision.
- The critique pass is recorded inline, `specs_check.sh` runs clean, and the handoff is
  stated.

## Verify

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh   # orphan REQ-*, missing CAP-*/QA-* citations, structure, size budgets
```

`check_traceability.py` reports a `REQ-*` that cites no `CAP-*`/`QA-*` as an error, and a
`REQ-*` that no delivery story ever picks up as an orphan once slices exist. Fix findings
rather than silencing them.

## Gotchas in this repo

- The linters are stdlib-only; `specs_check.sh` picks the `qgd` conda interpreter when it
  is available. Tests, benchmarks, and examples themselves always run in `qgd`.
- `REQ-*` ids are cited by delivery stories, engineering tasks, evidence matrices, and
  milestone closeouts. Extend, never renumber; mark retired ids
  `~superseded by REQ-0NN~`.
- Evidence must name the lane it runs in: the fast pytest lane (`-m "not slow"`), the
  `slow` lane, a benchmark evidence pipeline under `benchmarks/density_matrix/`, the
  optional C++ tests, or the Qiskit Aer external reference. Exactness requirements are
  stated relative to the sequential `NoisyCircuit` baseline with an explicit tolerance.
- Unsupported behavior is a first-class requirement here: the density path is strict
  (explicit errors, no silent fallback), so every `REQ-*` that widens a support surface
  needs a negative scenario naming the rejected input and the error it raises.
- Delivered Phases 1–3.1 live under `docs/density_matrix_project/archive/phases/` and
  have no `INITIAL_REQUIREMENTS.md`; new milestones go under `docs/specs/milestones/<slug>/`.

## References

- `references/document-structure.md` — the seven required sections, `REQ-*` shape, EARS/BDD
  acceptance, NFR and boundary tiers. Read while drafting.
- `references/critique-and-update.md` — the adversarial critique pass, the validation
  checkpoint, and the rules for evolving an existing baseline. Read before freezing, or when
  updating.

## Usage notes

- Prefer one canonical spec file per milestone; link out to designs or tickets rather than
  duplicating prose.
- When the user already has a PRD, **extract** scope, requirements, and acceptance criteria
  into this structure instead of pasting unstructured text wholesale.
- If the milestone is exploratory, a thin NFR section is acceptable — but keep the boundaries
  strict, because those protect the codebase even when product detail is fuzzy.
