# Archive — delivered phases and program-level planning (read-only)

> **Status:** frozen · **Scope:** Phases 1, 2, 3 and 3.1 of the density-matrix track plus
> the program-level plan, ADRs and references they were planned from · **Convention:**
> phase-based spec-driven development (rev A of the `spec-driven-development` skill).
> New spec work lives in `docs/specs/`; see `docs/sdd-skills-guide.md`.

## What is here

| Tree | Contents |
|------|----------|
| `phases/phase-1/` | `API_REFERENCE_PHASE_1.md` — core `squander.density_matrix` API at Phase 1 closure |
| `phases/phase-2/` | Exact noisy backend integration: Layer 1 contract (`DETAILED_PLANNING_PHASE_2.md`, `ADRs_PHASE_2.md`, checklist), eight task mini-specs with stories and per-story implementation plans, paper surfaces, slides |
| `phases/phase-3/` | Noise-aware partitioning and fusion: Layer 1 contract, algorithm landscape, eight task trees, `API_REFERENCE_PHASE_3.md`, paper surfaces |
| `phases/phase-3-1/` | Channel-native / superoperator fusion decision study: Layer 1 contract, six vertical-slice story/engineering-task files, six task trees, closure plan, pre-publication evidence review, paper surfaces |
| `planning/PLANNING.md` | The high-level research and implementation plan (dependency order, phases, decision gates, benchmark matrix, scope cuts) |
| `planning/ADRs.md` | Program-level ADR-001 … ADR-008 — long-horizon rationale (exact dense anchor, local noise priority, workload-driven gate coverage, deferred channel-native fusion and approximate scaling) |
| `planning/REFERENCES.md` | Bibliography used by the phase papers |

Delivery status of these phases is summarised in [`../CHANGELOG.md`](../CHANGELOG.md) and
[`../README.md`](../README.md); the shipped code and evidence pipelines they describe are
current and live under `squander/`, `tests/`, and `benchmarks/density_matrix/`.

## How to use it

- **Cite, do not extend.** A new milestone contract may cite a delivered decision or
  evidence bundle here by path. Do not add, edit, or re-plan anything in this tree; new
  work is planned under `docs/specs/milestones/<slug>/` with the current convention.
- **ADR-001 … ADR-008 remain in force** until a milestone ADR under `docs/specs/`
  supersedes one explicitly. `docs/specs/ARCHITECTURE_OVERVIEW.md` links to them.
- **Source material.** `create-product-statement` and `create-product-roadmap` lift durable
  intent, constraints, and sequencing from `planning/PLANNING.md`, `planning/ADRs.md`, and
  `../RESEARCH_ALIGNMENT.md`; they do not copy them.
- **API references.** `API_REFERENCE_PHASE_1.md` and `API_REFERENCE_PHASE_3.md` describe
  the public API as of those closures. When a milestone changes the API, write the
  refreshed reference outside this tree and outside `docs/specs/`.

## Conventions inside the archive

- Paths quoted as `docs/density_matrix_project/phases/…` or
  `docs/density_matrix_project/planning/…` refer to the pre-archive location; prefix
  `archive/` to resolve them. Relative Markdown links were adjusted where the move broke
  them; no other text was changed.
- Layer naming differs from the current convention: `TASK_<n>_STORIES.md` and
  `<NTH>_VERTICAL_SLICE_*_STORIES_AND_ENGINEERING_TASKS.md` are Layer 3;
  `STORY_<k>_IMPLEMENTATION_PLAN.md` and `ENGINEERING_TASK_*_IMPLEMENTATION_PLAN.md` are
  Layer 4. The mapping table is in
  `.cursor/skills/spec-driven-development/references/artifact-map.md`.
- `planning/PUBLICATIONS.md` and the per-phase paper surfaces belonged to a publication
  planning track that is no longer part of spec-driven development; the paper surfaces are
  kept for provenance only.
