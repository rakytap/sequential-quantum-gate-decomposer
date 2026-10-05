# The planning / code-generation seam

Read when delegating Layer 1–4 planning versus Step 4b implementation.

## Contents

- Role boundary
- Subagent table

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
