# Just-in-time reading order

Read when planning or implementing a milestone slice. Pre-loading the whole milestone tree
is what makes later slices drift.

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
