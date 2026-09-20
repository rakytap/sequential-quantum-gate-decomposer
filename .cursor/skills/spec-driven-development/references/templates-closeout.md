# Templates — slice closeout, milestone closeout, handback, change control

Read at slice close, at milestone close, or when a Step 4b pass must hand a contract gap
back to planning. These four artifacts record evidence and decisions; none of them is new
planning. Keep them tight.

## Contents

- Slice closeout
- Milestone closeout
- Step 4a handback
- Change control

## Slice closeout

`task-<n>/CLOSEOUT.md`. Status is `shipped` when the slice meets its Definition of Done,
or `implementation handback` when Step 4b hit a contract/ADR gap — in that case point to
`STEP_4A_HANDBACK.md` and do not mark the slice shipped. Budget: 200 lines.

```markdown
# <M#> slice <n> closeout — <one-line slice descriptor>
> **Status:** shipped | implementation handback · **Date:** <date> · **Work package:** <WP> · **Scope:** <e.g. test-only / code+tests>

## Summary
<what shipped, or what blocked finalization and where the handback lives>

## Test snapshots
| Lane | Command | Result |

## Acceptance verdicts (vs slice contract)
| Signal / TD / ET | Verdict | Evidence |

## Evidence commands (reproduce)
<commands>

## Remaining slices (indicative, just-in-time)
<only if the milestone continues>
```

The closeout is also the compaction contract: after a slice, the session's memory is
gone, so anything the next slice needs must be in this file.

## Milestone closeout

`<MILESTONE_ID>_CLOSEOUT.md` at the milestone root — the delivery record and the input to
`create-product-roadmap` revalidation. Budget: 250 lines.

```markdown
# <MILESTONE_ID> Closeout — `<milestone-slug>` (<one-line descriptor>)

> **Milestone delivery record.** Traceability: `CAP-*/QA-* → <M#> → REQ-* → DS-* → evidence`.
- **Milestone:** <M#> / `<milestone-slug>`  · **Outcome (roadmap):** <measurable outcome>
- **Close date:** <date>  · **Status:** Shipped | Partial | Missed

## 1. Slices delivered
| Slice | Focus | Key artifacts |

## 2. Milestone acceptance criteria — final status
| # | Criterion | Status | Evidence |

## 3. REQ-* evidence matrix (milestone scope)
| REQ | CAP / QA | Delivered | Evidence command / artifact |

## 4. Assumptions & risks — what this milestone learned
| ID | Learning | Status after milestone |

## 5. Deferred (carry forward)
| Item | Target milestone | Notes |

## 6. Evidence commands (reproduce)
<exact commands that prove the milestone>

## 7. Handoff — create-product-roadmap
- Mark <M#> Delivered; append a revalidation log entry.
- Promote the next milestone to Now (if sequence unchanged).
- Record learnings; confirm ARCHITECTURE_OVERVIEW.md / TECH_STACK.md are current.
- Escalate to PRODUCT_STATEMENT only if a core assumption was invalidated.

## 8. Layer 2 index (milestone artifacts)
| Path | Purpose |
```

Every `REQ-*` in scope must appear in section 3, and claims here must match the files on
disk. `check_artifacts.py` and `check_traceability.py` verify both.

## Step 4a handback

`task-<n>/STEP_4A_HANDBACK.md`, produced by the Step 4b code-generation pass when it hits
a contract/ADR gap and makes **no** design, scope, contract, or ADR decision. Each
finding is a question with options, trade-offs, consequences, and the authority that owns
it. The planning pass disposes each question, amends the cited contract, and re-issues a
code-ready verdict. Superseded when the slice ships. Budget: 200 lines.

```markdown
# Slice <n> Step 4a handback — questions to close before re-issuing code-ready
> **From:** Step 4b code-generation · **To:** Step 4a / Layer 1 planning · **Date:** <date>
> **Slice:** <M#> `<slug>` slice <n> — <one-line> (REQ-*/QA-* touched)
> **Status:** NOT code-ready. Step 4b made no design/scope/contract/ADR decisions and hands back <N> finding(s).

## Context
<what the slice is, what 4b produced, why it is not code-ready>

## Question 1 (blocker?) — <short title>
**Contract (as authored):** <file:line citation of the contract line to amend or affirm>
**What Step 4b found (reality):** <observed facts, diagnosis>
**Question to close:** <one sentence>
**Options (mechanism · pros · cons · consequences):**
- **A.** …  · **B.** …
**Authority:** Layer 1 | Layer 2 | ADR | stakeholder
**To re-issue code-ready:** <what to amend and what 4b must rework>

## What re-issuing code-ready requires (summary)
1. …

## Stop-conditions honored by the 4b pass
- No design/scope/contract/ADR decisions made in code.
- <frozen contracts / FC-* held; no ADR changed; current-state docs corrected>
```

## Change control

`CHANGE_CONTROL.md` at the milestone root, produced when a milestone deviation requires
formal governance sign-off — a baseline or contract deviation tied to a `REQ-*` or
guardrail. Optional per milestone; required when a `REQ-*` or guardrail demands an
approved path. Flag the need in the pre-implementation checklist so it is tracked from
planning through close. Budget: 200 lines.

```markdown
# Change Control — <one-line deviation title>
> **Governance evidence pack** for <REQ-*> (<what it satisfies>). Traces: `REQ-* → CAP-* → ROADMAP risk R# → assumption A# → ADR-####`.
- **Status:** Approved | Approved path | Pending | Refused  · **Request date:** <date>  · **Owners:** <parties>  · **Related ADR:** [ADR-####](ADRS_<MILESTONE_SLUG>.md#…)

## 1. Request summary
| Item | Detail |   (baseline vs proposed vs scope of deviation vs outcomes preserved)

## 2. Evidence supporting the request
| Evidence | Command / artifact | Result |

## 3. Governance outcome
| Field | Value |   (outcome, meaning, formal approval, approved-path steps)

## 4. Fallback if change control refuses
<trigger · response · impact>

## 5. Risks and mitigations
| Risk | Mitigation |

## 6. Sign-off record
| Role | Name | Decision | Date | Change-request ID |

## 7. Traceability
`CAP-* → REQ-* → DS-* → milestone acceptance #N → ADR-#### → A#, R#`
```

A milestone must not be marked Delivered while a `CHANGE_CONTROL.md` sign-off is still
pending.
