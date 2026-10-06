# Delivery stories — M-F1a slice C.3 (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.3 ·
> **Parent:** `TASK_7_MINI_SPEC.md` · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-005, ADR-F1A-009, ADR-F1A-010 ·
> **No push/PR**

Wording: baseline route verified for the shipped q4 cell. This slice records
provisional strict evidence only (`milestone_counted` false). The milestone stays open.

## DS-C3-1 — Four frozen strict cells

**Stakeholder / system value**

- The inventory names strict at 4, 6, 8, and 10. q6, q8, and q10 need the RM-1(a)
  family. Each cell needs every partition witnessed as channel-native, and QA-001
  against the sequential oracle.

**Given / When / Then**

- Given the four ids in `TASK_7_MINI_SPEC.md` §3.1 and budget 2.
- When `execute_partitioned_density_channel_native` runs.
- Then every partition carries exactly one `channel_native_motif`/`actually_fused` execution
  witness, partition rows carry no runtime label (ADR-F1A-003), and QA-001 holds.

**Scope**

- In: `mf1a_strict_validation.py`, including the family builder, and its tests.
- Out: C.4 and edits to `workloads.py`.

**Acceptance signals**

- q4 id stays `phase31_local_support_q4_spectator_embedding_smoke`.
- A pure-unitary partition makes the strict entry raise `channel_native_noise_presence`;
  no case and no bundle are written; the id is not replaced.

**Traceability**

- Initial requirement(s): REQ-001, REQ-002, REQ-003, REQ-006
- Capability / quality attribute: QA-001, QA-005
- ADR(s): ADR-F1A-001, ADR-F1A-003, ADR-F1A-005

## DS-C3-2 — Own allowlist, siblings unchanged

**Stakeholder / system value**

- A later commit can regenerate the strict bundle on revision-only differences
  without editing the q4, fused, or hybrid allowlists.

**Given / When / Then**

- Given four equal current revisions and four equal prior revisions, and no other
  case-field difference.
- When the strict predicate runs.
- Then regeneration passes. A single moved case, with the other cases unmoved,
  fails at that case's path. When every case moved and either side is non-uniform,
  `cases[0]` fails.

**Scope**

- In: a length-4 tuple and a new predicate in the strict module.
- Out: edits to the q4, fused, or hybrid tuples.

**Acceptance signals**

- The ET-C3-1 matrix names `first_mismatch` for each one-case mutant.
- The outside-expected marker is for matrix residuals only. `lambda_min` is one-sided.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 Amendment 1

## DS-C3-3 — C2 commits the strict bundle once

**Stakeholder / system value**

- The first strict bundle is committed once, with the closeout. Later regeneration
  is restored and not committed.

**Given / When / Then**

- Given a clean C1 with no strict bundle.
- When (c) passes and Reviewer finishes (e).
- Then C2 contains the strict bundle from (c) plus `CLOSEOUT.md`. (g) restores
  q4, fused, hybrid, and strict from the C2 sha.

**Scope**

- In: ADR-F1A-009 (a)–(g) as in the mini-spec §3.4.
- Out: staging q4, fused, hybrid, or historical JSON in C2.

**Acceptance signals**

- Eight-path diff empty before restore.
- C2 path list is the strict bundle from (c) plus CLOSEOUT.md (mini-spec §3.4).

**Traceability**

- Initial requirement(s): REQ-001, REQ-004
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 (+ Amendment 1), ADR-F1A-010
