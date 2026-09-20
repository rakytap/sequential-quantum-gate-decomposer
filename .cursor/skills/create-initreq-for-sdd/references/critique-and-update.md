# Adversarial critique pass and the update path

Read before freezing a draft (critique), or when running this skill on an
`INITIAL_REQUIREMENTS.md` that already exists (update).

## Contents

- Adversarial critique pass
- Validation checkpoint
- Updating an existing requirements baseline

## Adversarial critique pass

Run one adversarial pass against the draft and resolve or explicitly record every finding —
never leave a weak spot implicit. Capture the outcome as a short inline **Critique** note,
or a list of resolved items, in the artifact.

- **Attack the riskiest assumptions.** Which one would invalidate the largest scope if it
  is wrong? Is each tied to a validation milestone or a test?
- **Attack each `REQ-*` acceptance.** Which is least testable, most ambiguous, or missing a
  negative or error scenario? Tighten the EARS/BDD wording, or add the missing
  `Given / When / Then`.
- **Attack the NFRs.** Which `QA-*`-realizing NFR has no **response measure**? Add a number
  and a measurement method, or mark it `[confirm]` with an owner.
- **Attack traceability.** Any `REQ-*` not citing a `CAP-*`/`QA-*` and the milestone? Any
  in-scope `CAP-*`/`QA-*` with no `REQ-*`? Fix or flag.
- **Attack scope.** What is in this milestone that belongs in a later one? Which edge case
  is really a deferred direction?
- **Attack the evidence route.** For each `REQ-*`, name the command or check that would
  prove it. A requirement whose evidence route cannot be named yet is a requirement that
  will be argued about at close.

For each finding: tighten the requirement or acceptance, add the missing scenario, or record
it as an explicit open question. Then run `specs_check.sh` and take the validation
checkpoint to the user. This pass complements stakeholder validation; it does not replace
it.

## Validation checkpoint

> Does this specification capture your intent for **this milestone**? Which acceptance
> criteria are wrong, missing, or need tighter edge-case coverage before we generate a
> technical plan or implementation tasks?

Then state the next step: `spec-driven-development` ingests this file into Layer 1 —
detailed planning, an ADR stub, checklist gaps drawn from the open questions, current-state
doc creation or update needs, and the initial evidence-matrix seeds from `REQ-*` / `QA-*`.
This file **feeds** `DETAILED_PLANNING_*`; it does not replace it.

## Updating an existing requirements baseline

This skill is bidirectional: run it on an existing `INITIAL_REQUIREMENTS.md` to evolve it,
not only to create one. Living requirements change as planning and implementation reveal
intent — update the spec first, then let the downstream layers follow.

- **Preserve stable `REQ-*` ids.** Never renumber an existing `REQ-*`. Extend with new ids
  (`REQ-012`, …), and mark a superseded requirement `~superseded by REQ-0NN~` with a
  one-line reason instead of deleting it, so downstream traceability and evidence stay
  intact. The linter treats a struck-through requirement as retired rather than orphaned.
- **Re-walk the spine.** Every changed or added `REQ-*` must still cite its `CAP-*`/`QA-*`
  and the milestone. Fix any orphaned trace.
- **Update acceptance and NFRs together.** If a `REQ-*` changes, re-check its EARS/BDD
  acceptance and any NFR response measure or `QA-*` it realizes, and keep the
  evidence-matrix seed consistent.
- **Record a Change log entry** at the bottom of the file: date, what changed, why, and the
  triggering artifact — a `STEP_4A_HANDBACK.md` disposition, an ADR, a stakeholder decision.
  This mirrors the product statement's change-log discipline.
- **Do not re-plan Layers 2–4 here.** Evolve requirements only; re-planning belongs to
  `spec-driven-development`, which re-ingests the revised file into Layer 1.
- **Trivial edits** — typos, formatting, link fixes — skip the elicitation pass. Substantive
  requirement changes re-run elicitation, critique, and the checkpoint.

After any update, run `specs_check.sh` and re-state the handoff.
