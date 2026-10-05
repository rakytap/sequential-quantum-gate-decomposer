# M-F1a slice 2 — historical verify-only (rocky planning)

> **Milestone:** M-F1a `exactness-reconfirmation` · **Directory:** `docs/specs/milestones/exactness-reconfirmation/task-2/` ·
> **HEAD:** `43d8f07425359baee41fbbe8db5add5421264d81` · **Parent:** `a2928bf1` (q4 C2) ·
> **Status:** closed code-ready under ADR-F1A-008 on 2026-10-05 · **No push/PR**

## Artifacts in this directory

| File | Layer |
|------|-------|
| `TASK_2_MINI_SPEC.md` | 2 |
| `DELIVERY_STORIES.md` | 3 |
| `ENGINEERING_TASKS.md` | 4 |
| `STEP_4A_HANDBACK.md` | 4a handback (verdict blank) |
| `ARCHITECT_STEP_4A_RULINGS.md` | Architect reference |

## RC-6 (Tech Lead 2026-10-05)

Order **P0 → A → P1 → B**. ADR fold (008 Amend1, 010, 009 Amend1, 011) and G-01 rewrite are on
rocky uncommitted; **Slice A C1** still commits them with code (RC-6). P0b landed at
`5a5168fd`; `ADR_AMENDMENTS_<SLUG>.md` is checker-visible (slug check and the 400-line
budget). P1 stage-aware `check_artifacts` follows Slice A's C1; until P1, Slice A uses the
manual ADR-F1A-008 split as q4 did. The standing eight-bundle restore in
`test-density-matrix` SKILL.md lines 238-240 and ADR-F1A-011 are consistent; the skill edit
landed in P0b.

## Q-1

Base `1cb3d20c` for the eight historical artifact directories (empty diff vs `43d8f074`):

- `benchmarks/density_matrix/artifacts/correctness_evidence/correctness_package/correctness_package_bundle.json`
- `…/output_integrity/output_integrity_bundle.json`
- `…/runtime_classification/runtime_classification_bundle.json`
- `…/sequential_correctness/sequential_correctness_bundle.json`
- `…/external_correctness/external_correctness_bundle.json`
- `…/unsupported_boundary/unsupported_boundary_bundle.json`
- `…/correctness_matrix/correctness_matrix_bundle.json`
- `…/summary_consistency/summary_consistency_bundle.json`

Registry at HEAD: 9 suites (1 sibling + 8 historical).

## Lineage

Table 2 = CSCS 2026 short-paper frozen-26 matrix (17/9/0). None of the six Phase-3 bundles feeds it.
