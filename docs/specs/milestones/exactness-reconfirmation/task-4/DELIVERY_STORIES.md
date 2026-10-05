# Delivery stories — M-F1a slice C.0 (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect · **Slice:** M-F1a C.0 ·
> **Parent:** `TASK_4_MINI_SPEC.md` · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-001 · QA-001, QA-008 · ADR-F1A-001, ADR-F1A-008, ADR-F1A-009,
> ADR-F1A-010 · **No push/PR** · **Re-close:** `/tmp/c0-step4a/C0_STEP4A_RECLOSE.md`

Wording: baseline route verified for the shipped q4 cell. Slice B claim: q4 baseline
regenerated at clean C2. This inventory is not counted evidence.

## DS-C0-1 — Sixteen-row inventory

**Stakeholder / system value**

- G-03 needs a named denominator before any later counted route group runs.

**Given / When / Then**

- Given ADR-F1A-001's four routes and anchors 4/6/8/10 at HEAD `0e8299e9`.
- When the inventory is read.
- Then it has 16 rows, each with a public entry that exists, an advertisement source,
  a candidate workload or `TBD inventory`, and a realization value that was not guessed.

**Scope**

- In: `ROUTE_INVENTORY.md`.
- Out: counted runs, workload edits, new routes.

**Acceptance signals**

- 16 advertised rows. Zero `entry_missing`. Zero `not_advertised`.
- Strict q6, q8, and q10 are `no_eligible_workload` on historical builders and stay in the denominator. RM-1(a) makes them realizable via a new M-F1a-only family at C.3.

**Traceability**

- Initial requirement(s): REQ-001
- Capability / quality attribute: CAP-001 / QA-001, QA-008
- ADR(s): ADR-F1A-001, ADR-F1A-003

## DS-C0-2 — C-slice order and close shape

**Stakeholder / system value**

- Later slices need an order that puts fused and channel-native groups first, and a
  close shape that matches Slice A.

**Given / When / Then**

- Given the inventory flags.
- When Architect reviews the proposal.
- Then the order is C.1 fused, C.2 hybrid, C.3 strict, C.4 baseline (q6/q8/q10).
  Each of those slices uses ADR-F1A-009 (a)–(g). C.2 and C.3 stay separate. C.1 and
  C.2 do not wait on the C.3 workload.

**Scope**

- In: the order table in the mini-spec and the inventory.
- Out: implementing those slices in this task.

**Acceptance signals**

- q4 baseline is not re-opened as C.4 work.
- q10 time is `TBD — measure`. `test_QX2` stays deselected.

**Traceability**

- Initial requirement(s): REQ-001
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010

## DS-C0-3 — Flags before code-ready

**Stakeholder / system value**

- A missing workload or a route that is not advertised returns to Research Manager
  before Step 4b. This slice proposes no oracle, tolerance, counted-set, or scope change.

**Given / When / Then**

- Given the flag column.
- When this close records C.0 code-ready under ADR-F1A-008.
- Then strict q6, q8, and q10 stay `no_eligible_workload` on historical builders, and
  RM-1(a) records them as realizable via the C.3 M-F1a-only family. This close does
  not authorize C.1 Step 4b.

**Scope**

- In: the flag section of `ROUTE_INVENTORY.md`.
- Out: resolving the flag by dropping the cell.

**Acceptance signals**

- The mini-spec states the flag and states that the counted set is unchanged.

**Traceability**

- Initial requirement(s): REQ-001
- Capability / quality attribute: QA-001
- ADR(s): ADR-F1A-001, ADR-F1A-010
