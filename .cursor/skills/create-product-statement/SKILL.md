---
name: create-product-statement
description: Creates or revises docs/specs/PRODUCT_STATEMENT.md — the durable North Star that decides what not to build, with capabilities (CAP-*) and quality attributes (QA-*). Use for product vision, strategy reframing, PR-FAQ, Lean Canvas, guardrails, or a broken core assumption. Not for milestone sequencing, requirements, or implementation planning.
---

# Create a product statement

Produce the **durable product statement**: the North Star every milestone, requirement,
and line of code traces back to. It is short, inspiring, and stable — it changes little
across the product life cycle while the roadmap beneath it stays fluid.

**Output:** `docs/specs/PRODUCT_STATEMENT.md`, roughly 300 lines or fewer, carrying
`CAP-*` capabilities and `QA-*` quality attributes with stable ids.

## Position in the stack

| Layer | Skill | Artifact |
|-------|-------|----------|
| **Product** | **this skill** | `docs/specs/PRODUCT_STATEMENT.md` (`CAP-*`, `QA-*`) — stable |
| Roadmap | `create-product-roadmap` | `docs/specs/ROADMAP.md` (`M#`) — living |
| Milestone input | `create-initreq-for-sdd` | `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) |
| Layers 1–4 | `spec-driven-development` | `milestones/<slug>/…` |

This is the only layer **outside** the per-milestone revalidation loop. Revisit it only on
a deliberate strategic shift, or when a milestone's evidence invalidates a core
assumption — escalated up from roadmap revalidation, never rewritten by it.

One `CAP-*` is advanced by one or more milestones; each milestone's `REQ-*` cites the
`CAP-*`/`QA-*` it serves. Never lose an id in the handoff. Full stack diagram and shared
glossary: `docs/sdd-skills-guide.md`.

## What this is and is not

**Is:** the stable "why" and "for whom"; durable capabilities and quality bars; a decision
filter that answers "does this help us decide what *not* to build?"

**Is not:** a roadmap, a release plan, a feature backlog, or an architecture decision.
Defer the "how". Capture only fixed organizational standards and non-negotiable
constraints, never solution design.

## Principles

1. **North Star, stable.** Aspirational, customer-centric, durable. If it cannot decide
   what not to build, sharpen it.
2. **Customer and problem before solution.** Start with who hurts and why; describe the
   transformation as "Today… / In the future…".
3. **Outcomes over features.** Capabilities are outcomes the product enables, not a
   feature list.
4. **Validate before freezing.** Use a PR-FAQ or Lean Canvas plus an explicit
   assumption/risk map with kill criteria. Drive out the riskiest unknowns first.
5. **Seed the ubiquitous language.** Name core domain concepts now; this is the root of
   DDD strategic design and keeps every downstream layer honest.
6. **Make quality measurable.** Write each quality attribute as a scenario with a response
   measure, so it can become a fitness function and an evidence-matrix row downstream.
7. **Guardrails, not architecture.** Capture product-level Always / Ask first / Never and
   non-negotiable constraints. Leave stack and design to ADRs.
8. **Self-document the current state separately.** Actual architecture and stack facts
   belong in `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md`, never smuggled into the North
   Star.

## Workflow

**1. Elicit.** Write no application code. Read the user's input and any source material
under `docs/`. If intent is unclear, ask **3–7 concrete questions**: target customers and
personas, the core job-to-be-done and its pain, the today → future transformation,
alternatives and differentiation, hard constraints (compliance, platform, data residency),
success signals, and the riskiest assumptions. Prefer waiting for answers; if told to
proceed, record an **Assumptions** subsection.

**2. Draft.** Create `docs/specs/PRODUCT_STATEMENT.md` (and `docs/specs/` if missing)
using the section order in `references/document-structure.md`.

**3. Seed the current-state docs.** Ensure the workflow has somewhere to self-document the
actual system: `docs/specs/ARCHITECTURE_OVERVIEW.md` and `docs/specs/TECH_STACK.md`. If
the stack and architecture already exist and are known from source material, create or
update lightweight current-state stubs. If they are not known yet, record a handoff note
that the first product-walking-skeleton milestone must establish them. Never put
speculative architecture in the product statement.

**4. Validate.** Run at least one lens from `references/validation-lenses.md` and the
adversarial critique pass there, record the result inline, then take the stakeholder
checkpoint question to the user.

**5. Hand off.** State the next step explicitly: invoke `create-product-roadmap` to
decompose `CAP-*` and `QA-*` into outcome milestones. Say whether the current-state docs
were seeded or must be created by the first milestone. Do not author the roadmap or any
milestone's requirements here.

## Completion criteria

- `docs/specs/PRODUCT_STATEMENT.md` exists with all nine sections.
- 5–12 `CAP-*`, each with a why and a measurable success signal.
- Every `QA-*` is a scenario with a **response measure**.
- Guardrails state Always / Ask first / Never.
- Riskiest assumptions carry validation status and kill criteria.
- One validation lens plus the critique pass are recorded inline.
- No stack, schema, or architecture decision appears anywhere in the file.
- `specs_check.sh` runs clean, and the handoff to `create-product-roadmap` is stated.

## Gotchas in this repo

- `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` wraps the spec
  linters (stdlib-only; picks the `qgd` conda interpreter when available).
- `docs/specs/` is the single spec root; program-level artifacts sit at its top level.
  `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` already exist as current-state docs —
  read them for constraints, do not restate them.
- Source material for the North Star: `docs/density_matrix_project/README.md`,
  `RESEARCH_ALIGNMENT.md`, and the archived program plan and ADRs under
  `docs/density_matrix_project/archive/planning/` (`PLANNING.md` §2–§3 and §8,
  `ADRs.md` ADR-002 … ADR-008). Lift durable intent and constraints from them; leave
  phase sequencing to the roadmap.
- The product is a research software track: capabilities are outcomes a researcher can
  reach (exact noisy evaluation, reproducible evidence bundles), quality attributes are
  measurable against the sequential `NoisyCircuit` baseline or Qiskit Aer. Publication
  targets are success signals, not capabilities.
- `CAP-*` and `QA-*` ids are cited by milestones, requirements, evidence matrices, and
  fitness tests. Extend, never renumber.

## References

- `references/document-structure.md` — required section order, `CAP-*`/`QA-*` shape,
  vision templates, change policy. Read while drafting or revising.
- `references/validation-lenses.md` — PR-FAQ, Lean Canvas, assumption/risk map, the
  adversarial critique pass, the stakeholder checkpoint. Read before freezing a draft.

## Anti-patterns

- **Vision as slogan** — inspiring, but it cannot decide what not to build.
- **Feature list masquerading as capabilities** — outputs instead of outcomes.
- **Premature architecture** — stack, schema, or design smuggled into the statement.
- **Architecture facts buried in product intent** — current topology belongs in
  `ARCHITECTURE_OVERVIEW.md` / `TECH_STACK.md`.
- **Unvalidated certainty** — no PR-FAQ, canvas, or assumption map; riskiest unknowns left
  implicit.
- **Unmeasurable quality bars** — a `QA-*` with no response measure, so nothing downstream
  can test it.
- **Statement bloat** — too long to hold in context; push detail down, not up.
- **Renumbered ids** — breaking every downstream citation to save a gap in the sequence.
