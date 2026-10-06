# Delivery stories — M-F1a slice C.2 (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.2 ·
> **Parent:** `TASK_6_MINI_SPEC.md` · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-009, ADR-F1A-010 · **No push/PR**

Wording: baseline route verified for the shipped q4 cell. This slice records
provisional hybrid evidence only (`milestone_counted` false). The milestone stays open.

## DS-C2-1 — Four frozen hybrid cells

**Stakeholder / system value**

- The inventory names hybrid at 4, 6, 8, and 10. Each cell needs a witnessed
  channel-native partition and QA-001 against the sequential oracle.

**Given / When / Then**

- Given the four ids in `TASK_6_MINI_SPEC.md` §3.1 and budget 2.
- When `execute_partitioned_density_channel_native_hybrid` runs.
- Then every partition is labelled, every label agrees with its execution witness
  (ADR-F1A-003), at least one class is `phase31_channel_native`, and QA-001 holds.

**Scope**

- In: `mf1a_hybrid_validation.py` and its tests.
- Out: C.3, C.4, and edits to `workloads.py`.

**Acceptance signals**

- Ids match §3.1, including `phase31_pair_repeat_q8_dense_seed20260318` and
  `phase31_pair_repeat_q10_dense_seed20260318`.
- Zero channel-native partitions fails the case. The id is not replaced.

**Traceability**

- Initial requirement(s): REQ-001, REQ-002, REQ-003, REQ-006
- Capability / quality attribute: QA-001, QA-005
- ADR(s): ADR-F1A-001, ADR-F1A-003

## DS-C2-2 — Own allowlist, unchanged q4 and fused tuples

**Stakeholder / system value**

- A later commit can regenerate the hybrid bundle on revision-only differences
  without editing the q4 or fused allowlists.

**Given / When / Then**

- Given four equal current revisions and four equal prior revisions, and no other
  case-field difference.
- When the hybrid predicate runs.
- Then regeneration passes. A single moved case, with the other cases unmoved,
  fails at that case's path. When every case moved and either side is non-uniform,
  `cases[0]` fails.

**Scope**

- In: a length-4 tuple and a new predicate in the hybrid module.
- Out: edits to `Q4_REGENERATION_ALLOWLIST` or the fused tuple.

**Acceptance signals**

- The ET-C2-1 mutant matrix names the failing path for each one-case mutant.
- Findings follow Layer 1 §10 and do not change pass/fail.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 Amendment 1

## DS-C2-3 — C2 commits the hybrid bundle once

**Stakeholder / system value**

- The first hybrid bundle is committed once, with the closeout. Later regeneration
  is restored and not committed.

**Given / When / Then**

- Given a clean C1 with no hybrid bundle.
- When (c) passes and Reviewer finishes (e).
- Then C2 contains the hybrid bundle from (c) plus `CLOSEOUT.md`. (g) restores
  q4, fused, and hybrid from the C2 sha.

**Scope**

- In: ADR-F1A-009 (a)–(g) as in the mini-spec §3.4.
- Out: staging q4, fused, or historical JSON in C2.

**Acceptance signals**

- Eight-path diff empty before restore.
- C2 path list is the hybrid bundle from (c) plus CLOSEOUT.md (mini-spec §3.4).

**Traceability**

- Initial requirement(s): REQ-001, REQ-004
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 (+ Amendment 1), ADR-F1A-010, ADR-F1A-011
