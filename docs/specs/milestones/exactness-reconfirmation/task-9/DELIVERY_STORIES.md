# Delivery stories — M-F1a counted manifest (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 after the Research Manager freeze record · **Slice:** M-F1a task-9 ·
> **Parent:** `TASK_9_MINI_SPEC.md`, `COUNTED_MANIFEST.md` ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, REQ-007 ·
> QA-001, QA-005, QA-008 · ADR-F1A-001, ADR-F1A-003, ADR-F1A-010 ·
> **No push/PR**

Per cell, the wording is "`<route>` route verified at q`<n>`". The milestone stays open. G-03 stays open until the counted bundle passes.

## DS-C9-1 — Sixteen default cells, re-executed

**Stakeholder / system value**

- The denominator is every advertised route at every anchor, on the workload that already realizes that route.

**Given / When / Then**

- Given `COUNTED_MANIFEST.md` rows 1–16, budget 2, and the parameter and seed columns.
- When the counted bundle is built from the five existing `build_cases(provenance=...)` results.
- Then each cell matches that row, including `n` and `seed_policy`, and the realization rule holds: baseline requested and realized with `fused_region_count` 0; fused has at least one `actually_fused` unitary island; strict executes an eligible motif on every partition; hybrid has complete labels and at least one witnessed channel-native partition. QA-001 holds. No workload is swapped.

**Scope**

- In: the 16 manifest cells: the task-1 pin at row 1 and the C-slice defaults at rows 2–16.
- Out: a new family, an edit to `workloads.py`, Aer, and energy.

**Acceptance signals**

- Exact-set failure on a wrong count, a substituted id, a reorder, or a wrong seed or parameter count.
- A route-realization failure on one row of each of the five producing modules.

**Traceability**

- REQ-001, REQ-002, REQ-003. QA-001, QA-005. ADR-F1A-001, ADR-F1A-003. Goal G1, G3, G4.

## DS-C9-2 — Counted only from a clean start

**Stakeholder / system value**

- A dirty tree must not become the counted denominator.

**Given / When / Then**

- Given one shared `capture_provenance()`.
- When `clean_start` and `provenance_pass` are true, every case has `milestone_counted` true and the summary count is 16.
- When the tree is dirty, every case has `milestone_counted` false, the summary count is 0, and the provenance gate fails.
- `completeness_claim` and `claim_boundary` equal the Research Manager-accepted values in `COUNTED_MANIFEST.md` §§4–5.

**Scope**

- In: the restamp rules in the mini-spec §3.
- Out: flipping the flag inside the five sibling modules.

**Acceptance signals**

- A clean fixture counts 16. A dirty fixture counts 0.
- The module takes `capture_provenance` and the QA-001 tolerances from the q4 module, the siblings run the q4 `evaluate_mf1a_qa001` and the `noisy_runtime` oracle, and the source does not import `workloads`.

**Traceability**

- REQ-004, REQ-006. QA-001, QA-008. ADR-F1A-004, ADR-F1A-010. Goal G2, G5.

## DS-C9-3 — Provisional bundles stay historical

**Stakeholder / system value**

- The counted sibling is new evidence. The five provisional shas remain the slice record.

**Given / When / Then**

- Given C1 with the counted module registered at index 5.
- When Tester runs the one pipeline command from a clean tree, after the RM freeze record.
- Then the counted bundle is written, the five provisional siblings are restored to their C1 bytes, and the eight historical bundles are unchanged.
- A counted failure stops. It does not change the oracle, the tolerances, or the manifest.

**Scope**

- In: the Tester gate in the mini-spec §7.
- Out: the checklist §9 deferrals, the Linux CI job, and current-state docs.

**Acceptance signals**

- REQ-007 eight-path diff empty before any restore.
- Allowlist-only differences on the five provisional siblings.
- (g) restores all six siblings and commits nothing.

**Traceability**

- REQ-004, REQ-007. QA-008. ADR-F1A-005, ADR-F1A-009, ADR-F1A-011. Goal G5.
