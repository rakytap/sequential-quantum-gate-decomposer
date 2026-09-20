# Revalidation — the loop that runs after every milestone

Read when `spec-driven-development` reports a milestone delivered, or when evidence
arrives that should reshape the plan.

## Contents

- Inputs
- Steps
- Adversarial critique pass
- Checklist
- Escalation

## Inputs

Revalidation is triggered by the milestone's `<MILESTONE_ID>_CLOSEOUT.md`. Read, in this
order:

1. the closeout — slices delivered, final acceptance status, the `REQ-*` evidence matrix,
   learnings, deferred items;
2. the stable `PRODUCT_STATEMENT.md`;
3. `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md`;
4. the milestone's evidence — validated and invalidated assumptions, new risks, actual
   effort against expectation, and any market or stakeholder shift.

Do not read the milestone's slice-level task files. If the closeout does not carry what
revalidation needs, the closeout is the thing to fix.

If the milestone produced a `CHANGE_CONTROL.md`, confirm its sign-off status before
treating the milestone as Delivered.

## Steps

1. Mark the milestone's outcome **achieved / partially achieved / missed**, and record the
   measure actually hit — not the measure hoped for.
2. **Re-sequence, re-scope, split, merge, add, or drop** remaining milestones in light of
   the evidence.
3. Refresh Now / Next / Later and the dependency graph; promote the next milestone to
   *Now*.
4. Confirm `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` reflect any shipped change to
   architecture, integrations, runtime, tooling, or operations. If they do not, add a
   roadmap risk or a documentation item to the next milestone.
5. Update the assumptions and risks section, and append a dated **revalidation log** entry:
   what we learned, and what changed because of it.
6. Hand the new *Now* milestone to `create-initreq-for-sdd` with its outcome,
   `CAP-*`/`QA-*` traces, slug, and current-state doc expectations.

## Adversarial critique pass

Before publishing the revalidated roadmap, attack it and resolve or record every finding:

- **Outcome honesty:** is any milestone marked achieved on a measure that was quietly
  redefined during delivery?
- **Learning absorbed:** name the concrete roadmap change each learning caused. A learning
  that changed nothing was either already known or is being ignored.
- **Sequencing:** does the next *Now* milestone still validate the riskiest remaining
  assumption? If a later milestone now carries more risk, it should move earlier.
- **Deferred items:** does every item deferred by the closeout have a target milestone, or
  has it silently become invisible work?
- **Deployability:** is each *Now* milestone still shippable on its own, and small enough
  for a few slices?
- **Orphans:** does every remaining milestone still trace to a `CAP-*`/`QA-*` that the
  product statement still holds?
- **Escalation:** did any evidence invalidate a core product assumption? Say so explicitly
  rather than absorbing it into a re-scope.

## Checklist

```
- [ ] Outcome of the delivered milestone recorded (achieved / partial / missed) with the measure hit
- [ ] `<MILESTONE_ID>_CLOSEOUT.md` read as the revalidation input; any `CHANGE_CONTROL.md` sign-off confirmed before marking Delivered
- [ ] Assumptions validated/invalidated captured; new risks added
- [ ] `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` checked and updated, or a follow-up risk recorded
- [ ] Remaining milestones re-sequenced / re-scoped / split / merged / added / dropped as needed
- [ ] Now/Next/Later and dependency graph refreshed; next milestone promoted to Now
- [ ] Critique pass run; every finding resolved or recorded
- [ ] Revalidation log entry appended (dated)
- [ ] Core-assumption breakage? -> escalate to create-product-statement
- [ ] Next Now milestone handed off to create-initreq-for-sdd
```

## Escalation

If a learning invalidates a core product assumption or capability, escalate to
`create-product-statement` before continuing. The product statement is **not** rewritten by
this loop — only flagged for review. Changing strategy inside the roadmap instead of
escalating is silent product drift.
