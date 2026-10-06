# Delivery stories — M-F1a slice C.4 (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.4 ·
> **Parent:** `TASK_8_MINI_SPEC.md` · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-005, ADR-F1A-009, ADR-F1A-010 ·
> **No push/PR**

Wording: baseline route verified for the shipped q4 cell. This slice records
provisional baseline evidence at 6, 8, and 10 only (`milestone_counted` false). The
milestone stays open.

## DS-C4-1 — Three frozen baseline cells

**Stakeholder / system value**

- The inventory names baseline at 6, 8, and 10 on the continuity workloads. q4 is
  already shipped. Each new cell needs a real baseline realization, with fusion not
  executed, and QA-001 against the sequential oracle.

**Given / When / Then**

- Given the three ids in `TASK_8_MINI_SPEC.md` §3.1 and budget 2.
- When `execute_partitioned_density` runs with `allow_fusion=False`.
- Then the realized path is `partitioned_density_descriptor_baseline`,
  `actual_fused_execution` is false, `fused_region_count` is 0, `actually_fused` is
  absent, the record carries no partition rows and no per-partition label (ADR-F1A-003),
  and QA-001 holds.

**Scope**

- In: `mf1a_baseline_validation.py` and its tests.
- Out: edits to the q4 module, `workloads.py`, and `planner_surface/common.py`.

**Acceptance signals**

- q4 bundle sha stays `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` as the historical file. C.4 does not append a case to it.
- Partition counts are 7, 9, and 11. A `> 1` check does not pass the gate.

**Traceability**

- Initial requirement(s): REQ-001, REQ-002, REQ-003, REQ-006
- Capability / quality attribute: QA-001, QA-005
- ADR(s): ADR-F1A-001, ADR-F1A-003, ADR-F1A-005

## DS-C4-2 — Own allowlist, siblings unchanged

**Stakeholder / system value**

- A later commit can regenerate the new baseline bundle on revision-only differences
  without editing the q4, fused, hybrid, or strict allowlists.

**Given / When / Then**

- Given three equal current revisions and three equal prior revisions, and no other
  case-field difference.
- When the baseline predicate runs.
- Then regeneration passes. A single moved case, with the other cases unmoved, fails
  at that case's path. When every case moved and either side is non-uniform,
  `cases[0]` fails.

**Scope**

- In: a length-3 tuple and a new predicate in the baseline module.
- Out: edits to the q4, fused, hybrid, or strict tuples.

**Acceptance signals**

- The ET-C4-1 matrix names `first_mismatch` for each one-case mutant.
- The outside-expected marker is for matrix residuals only. `lambda_min` is one-sided: a finding is only below `-1e-13`, and a positive `lambda_min` is neither a finding nor a marker.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 Amendment 1

## DS-C4-3 — C2 commits the new baseline bundle once

**Stakeholder / system value**

- The first 6/8/10 baseline bundle is committed once, with the closeout. Later
  regeneration is restored and not committed. The q4 bundle is not that artifact.

**Given / When / Then**

- Given a clean C1 with no `mf1a/baseline` bundle.
- When (c) passes and Reviewer finishes (e).
- Then C2 contains the baseline bundle from (c) plus `CLOSEOUT.md`. (g) restores
  q4, fused, hybrid, strict, and baseline from the C2 sha.

**Scope**

- In: ADR-F1A-009 (a)–(g) as in the mini-spec §3.4.
- Out: staging q4, fused, hybrid, strict, or historical JSON in C2.

**Acceptance signals**

- Eight-path diff empty before restore.
- C2 path list is the baseline bundle from (c) plus CLOSEOUT.md (mini-spec §3.4).

**Traceability**

- Initial requirement(s): REQ-001, REQ-004
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 (+ Amendment 1), ADR-F1A-010
