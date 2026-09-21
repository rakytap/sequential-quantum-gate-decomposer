# `docs/specs/` — the spec root

> **Status:** baseline established; product statement drafted (v0.3, at stakeholder checkpoint);
> roadmap drafted (v0.6, scientific-yield refinements, at stakeholder checkpoint).
> **How to work here:** `docs/sdd-skills-guide.md`. Editing this tree activates the
> `spec-driven-docs-specs` rule.

| Artifact | State | Produced by |
|----------|-------|-------------|
| `PRODUCT_STATEMENT.md` (`CAP-*`, `QA-*`) | draft v0.3 — `CAP-001…007`, `QA-001…012`, at stakeholder checkpoint | `create-product-statement` |
| `ROADMAP.md` (`M#`, Now/Next/Later) | draft v0.6 — M1–M3A Delivered; M4 `canonical-attributed-energy` is Now walking skeleton; Phase 4 bounded to M4/M6/M7/M8/M9 plus the M5 detectability screen | `create-product-roadmap` |
| `ARCHITECTURE_OVERVIEW.md` | current-state, seeded from delivered Phases 1–3.1 | `spec-driven-development` |
| `TECH_STACK.md` | current-state | `spec-driven-development` |
| `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) | per milestone | `create-initreq-for-sdd` |
| `milestones/<slug>/…` (Layer 1–4, closeouts) | per milestone, slice by slice | `spec-driven-development` |
| `.sdd-lint.json` | linter budgets, slug overrides, waivers | `spec-driven-development` |

Delivered Phases 1, 2, 3 and 3.1 predate this convention and are archived read-only under
`docs/density_matrix_project/archive/`; the roadmap records them as Delivered milestones.

Verify before claiming spec work complete:

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh
```
