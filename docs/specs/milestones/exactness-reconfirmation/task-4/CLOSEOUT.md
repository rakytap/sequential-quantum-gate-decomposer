# M-F1a C.0 closeout — advertised-route inventory

> **Status:** shipped · **Date:** 2026-10-05 · **Work package:** task-4 ·
> **Scope:** inventory only · **Inventory revision:** `0e8299e9f48361ece2cc1665d81d4b2a35b4f156` ·
> **Re-close:** `/tmp/c0-step4a/C0_STEP4A_RECLOSE.md` · **Commit:** `91680ec7` ·
> **No push/PR**

## Record

`ROUTE_INVENTORY.md` is accepted at `0e8299e9` with the Architect rulings in
`/tmp/c0-step4a/C0_STEP4A_CODE_READY_REVIEW.md` and this re-close
(`/tmp/c0-step4a/C0_STEP4A_RECLOSE.md`). No code, no counted evidence, and no
pipeline run. The q4 cell remains baseline route verified.

Strict q6, q8, and q10 stay `no_eligible_workload` on historical builders. RM-1(a)
is recorded: an M-F1a-only family at C.3, not left open. O-11 remains
`max_partition_qubits = 2` for every cell of all four routes. ADR-F1A-009 (a)–(g)
does not apply. This file is not a C1/C2 closeout. Order stays C.1 fused, C.2
hybrid, C.3 strict, C.4 baseline. This close does not authorize C.1 Step 4b.

The Tech Lead commit is `91680ec7`.

## Reproduce

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict
```

Clean output from the writer-pass run (both modes, 0 errors and 0 warnings):

```text
== SDD artifact structure (/home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer/docs/specs)
  no findings
-- 0 error(s), 0 warning(s), 0 info, 0 waived
== SDD traceability spine (/home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer/docs/specs)
  no findings
-- 0 error(s), 0 warning(s), 0 info, 0 waived
```
