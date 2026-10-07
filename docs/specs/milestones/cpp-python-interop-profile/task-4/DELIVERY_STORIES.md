# Delivery stories — M-F5a task-4
> **Status:** Step 4a draft · **Verdict:** not-ready · **Slice:** M-F5a task-4 ·
> **Scope:** four attribution routes on the width-4 anchor. No \(O\). No reduction ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 · CAP-004, CAP-007 · QA-008, QA-009 ·
> **Gate:** SDD stage `step-4a`. Milestone not complete ·
> **RM:** ALIGN 2026-10-07. A4 false. CAP-004 hold-the-line. No Step 4b

### Delivery story: DS-1 — Four attribution rows and no overhead ratio

**Stakeholder / system value**
- The counted inventory gains R-base, R-fused, R-strict, and R-hybrid without a second energy API and without an \(O\).

**Given / When / Then**
- Given the task-1 width-4 HEA anchor and an existing planner descriptor of that anchor
- When a later authorized run calls each of the four `execute_partitioned_density*` entries
- Then each row reports orchestration time, the apply component, throughput, and a one-sided 95 % bound on those times
- And the row has no \(O\), no QA-007 ratio, and `milestone_counted=false`

**Scope**
- In: the four ADR-F5A-001 entries. The N-34 launch, carried, not retuned
- Out: widths 6 and 8; R-oracle; a counted run in this draft

**Acceptance signals**
- A fixture that contains \(O\) on a route row fails
- A missing descriptor stops the slice as a handback

**Traceability**
- Initial requirement(s): REQ-001, REQ-004
- Capability / quality attribute: CAP-004, CAP-007 / QA-008
- Milestone planning reference(s): goals G1, G4
- ADR(s): ADR-F5A-001, ADR-F5A-004, ADR-F5A-006

### Delivery story: DS-2 — The routes do not reduce or close the milestone

**Stakeholder / system value**
- The new rows cannot be read as a reduction, a profiled-routes claim, or a finished milestone.

**Given / When / Then**
- Given this slice's change set and the three counted bundles
- When the rows and the diff are reviewed
- Then the three bundles keep their pinned sha256 values and no reduction diff is present

**Scope**
- In: the exclusions in `TASK_4_MINI_SPEC.md` §3
- Out: binding or dispatch edits; current-state docs

**Acceptance signals**
- CAP-004 stays hold-the-line. The reduction is absent
- "M-F5a complete", "reduction shipped", and "attribution routes profiled" are absent from the route rows

**Traceability**
- Initial requirement(s): REQ-005, REQ-006, REQ-008
- Capability / quality attribute: CAP-004 / QA-008
- Milestone planning reference(s): goals G4, G5, G8
- ADR(s): ADR-F5A-001, ADR-F5A-005, ADR-F5A-006

### Delivery story: DS-3 — State-vector behavior stays the default

**Stakeholder / system value**
- The extra rows must not become the default backend or a GitHub Actions gate.

**Given / When / Then**
- Given a later implementation, when non-interference is checked
- Then the legacy default backend is still state vector and density stays opt-in
- And current-state docs still wait for milestone close

**Scope**
- In: the existing state-vector pin
- Out: G-08, G-09, and Demo

**Acceptance signals**
- `test_explicit_state_vector_matches_legacy_default` still passes
- This draft does not mark M-F5a complete

**Traceability**
- Initial requirement(s): REQ-007
- Capability / quality attribute: QA-009
- Milestone planning reference(s): goals G7, G9
- ADR(s): ADR-F5A-007
