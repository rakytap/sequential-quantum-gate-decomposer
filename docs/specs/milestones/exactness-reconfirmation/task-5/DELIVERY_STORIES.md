# Delivery stories — M-F1a slice C.1 (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.1 ·
> **Parent:** `TASK_5_MINI_SPEC.md` · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-001, REQ-002, REQ-004, REQ-006 · QA-001, QA-008 ·
> ADR-F1A-001, ADR-F1A-009, ADR-F1A-010 · **No push/PR**

Wording: baseline route verified for the shipped q4 cell. This slice records
provisional fused evidence only (`milestone_counted` false). It does not close the milestone.

## DS-C1-1 — Four frozen fused cells

**Stakeholder / system value**

- The accepted inventory names the fused route at 4, 6, 8, and 10. Those cells need
  QA-001 evidence with a real fused region.

**Given / When / Then**

- Given the four workload ids in `TASK_5_MINI_SPEC.md` §3.1 and `max_partition_qubits` 2.
- When `execute_partitioned_density_fused` runs each cell against
  `execute_sequential_density_reference`.
- Then each case has `actual_fused_execution` true and `fused_region_count` at least 1,
  and QA-001 holds at the q4 tolerances.

**Scope**

- In: the new fused sibling and its tests.
- Out: C.2, C.3, C.4, and any edit to `workloads.py`.

**Acceptance signals**

- Manifest ids match the inventory exactly, including
  `layered_nearest_neighbor_q8_sparse_seed20260318` and
  `layered_nearest_neighbor_q10_sparse_seed20260318`.
- A zero-fusion result fails the case. The id is not replaced.

**Traceability**

- Initial requirement(s): REQ-001, REQ-002, REQ-006
- Capability / quality attribute: CAP-001 / QA-001
- ADR(s): ADR-F1A-001, ADR-F1A-002, ADR-F1A-005

## DS-C1-2 — Regeneration without touching the q4 allowlist

**Stakeholder / system value**

- Later commits must regenerate the fused bundle when only implementation revisions
  change, without widening the q4 allowlist.

**Given / When / Then**

- Given a prior fused bundle and a current bundle whose only case-field differences
  are the four `cases[i].provenance.implementation_revision` paths.
- When the fused comparator runs.
- Then `regeneration.pass` is true. Any other difference fails. The q4 constant stays
  length 1.

**Scope**

- In: a length-4 allowlist in `mf1a_fused_validation.py` only.
- Out: edits to `Q4_REGENERATION_ALLOWLIST`.

**Acceptance signals**

- A test pins both tuple lengths and the Slice B negatives.
- Findings follow Layer 1 §10 (above `1e-11` on Frobenius, max-abs, or trace; `lambda_min` below `-1e-13`; above `1e-13` is an outside-expected marker). A finding does not change pass/fail.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 Amendment 1

## DS-C1-3 — Two-commit close and one committed bundle

**Stakeholder / system value**

- Counted fused evidence has to land once, and regenerations after that must not be
  committed.

**Given / When / Then**

- Given a clean C1 with no fused bundle in the commit.
- When (c) runs the pipeline and C2 is cut after evidence review.
- Then C2 contains the fused bundle written at (c) plus `CLOSEOUT.md` (ADR-F1A-009 (f)).
  (g) restores both siblings from the C2 sha and commits nothing (ADR-F1A-009 (g);
  Amendment 1 consequence 1).

**Scope**

- In: ADR-F1A-009 (a)–(g) as §3.4–§3.5 of the mini-spec.
- Out: committing historical bundles or a regenerated q4.

**Acceptance signals**

- Eight-path historical diff empty before any restore.
- C2 path list matches mini-spec §3.5 (ADR-F1A-009 (f)/(g)).

**Traceability**

- Initial requirement(s): REQ-001, REQ-004
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 (+ Amendment 1), ADR-F1A-010, ADR-F1A-011
