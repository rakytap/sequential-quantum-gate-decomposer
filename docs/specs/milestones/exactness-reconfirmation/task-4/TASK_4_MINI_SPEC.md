# Task / Work Package 4: advertised-route inventory (Slice C.0)
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect · **Slice:** M-F1a C.0 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Inventory revision:**
> `0e8299e9f48361ece2cc1665d81d4b2a35b4f156` ·
> **Traces:** REQ-001 · QA-001, QA-008 · ADR-F1A-001, ADR-F1A-003, ADR-F1A-008,
> ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 · **No push/PR** ·
> **Re-close:** `/tmp/c0-step4a/C0_STEP4A_RECLOSE.md`

## 1. Purpose

Name, for each of the four ADR-F1A-001 routes at anchors 4, 6, 8, and 10, the public
entry, advertisement source, candidate workload, and whether genuine realization is
shown. The artifact is `ROUTE_INVENTORY.md` in this directory. It is not counted
evidence. q4 stays baseline route verified. Slice B's claim is q4 baseline regenerated
at clean C2. This slice adds no counted cell and does not change the oracle, QA-001,
tolerances, or G-07.

## 2. Scope

### 2.1 In scope

- Read-only inspection at HEAD `0e8299e9`.
- The 16-row inventory and the C.1–C.4 proposal.
- Flag list for Research Manager where a row is `no_eligible_workload` or a route is
  not advertised. None of the four routes failed the advertisement check.

### 2.2 Out of scope

Counted execution, manifest freeze, `validation_pipeline.py`, new bundles, edits to
`workloads.py`, re-labelling historical evidence, oracle or tolerance changes, C.0
code, and a task-4 `CLOSEOUT.md`. `test_QX2` stays deselected and is not fixed.

## 3. Required behavior

### 3.1 Inventory

Sixteen rows. Unknown values are `TBD inventory`. Field meanings are plan §4, recorded
in `ROUTE_INVENTORY.md`. Shared contracts: same QA-001, oracle
`execute_sequential_density_reference`, tolerances, and G-07 as the q4 cell. Fused and
channel-native groups are first in the proposed counted order.

### 3.2 What inspection showed

All four public entries exist and are re-exported. All sixteen rows are advertised
(`yes`) from roadmap M3/M3A and `ARCHITECTURE_OVERVIEW.md`. Seven rows are `shown`,
nine are `not_shown`, none are `contradicted`. Three flags are `no_eligible_workload`
(strict q6, q8, q10) on historical builders. No row is `other`. Details and line
citations are in `ROUTE_INVENTORY.md`.

### 3.3 C-slice proposal

| Slice | Route | Anchors | ADR-F1A-009 (a)–(g) |
|-------|-------|---------|---------------------|
| C.1 | `partitioned_density_descriptor_fused_unitary_islands` | 4, 6, 8, 10 | yes, when that slice is delivered; does not wait on C.3 |
| C.2 | `phase31_channel_native_hybrid` | 4, 6, 8, 10 | yes; does not wait on C.3 |
| C.3 | `phase31_channel_native` | 4, 6, 8, 10 | yes, after the RM-1(a) family is reviewed and frozen |
| C.4 | `partitioned_density_descriptor_baseline` | 6, 8, 10 | yes |

C.2 (hybrid) and C.3 (strict) stay separate (Architect, C.0 review). Do not merge any
slices. Each delivered C-slice uses the Slice A close shape: Reviewer implementation
review, C1, clean-start run, real CLOSEOUT, Reviewer evidence review, C2, clean-C2
regeneration with outputs restored. q10 time is `TBD — measure` before any q10 run.
Targeted `-k` selections only.

C.0 does not run that counted close. This writer pass records code-ready with a real
`CLOSEOUT.md` (status `shipped`). No placeholder CLOSEOUT. No waiver. The stage value
in `ENGINEERING_TASKS.md` is `step-4b-authorized`. Reviewer, then one local commit,
comes before any C.1 write. This close does not authorize C.1 Step 4b.

### 3.4 Research Manager flags

- Strict q6, q8, and q10 have no eligible workload on historical builders. The cells stay
  in the denominator; this is not a counted-set change. Research Manager disposed them
  as RM-1(a): C.3 designs an M-F1a-only family (width-n generalization of the q4
  spectator-embedding motif; U3 and CNOT; local depolarizing / amplitude damping /
  phase damping at 0.10 / 0.05 / 0.07; sequential-oracle comparison; `max_partition_qubits`
  2). The new module does not edit `workloads.py`. RM-1(a) rejects option (b) as the Architect review stated it (record the three cells as not realizable). Narrowing strict to microcase widths is a separate non-option under ADR-F1A-001.
  C.3 Step 4a cannot close until that design is reviewed and frozen. C.1 and C.2 do not
  wait.
- No oracle, tolerance, counted-set, or scope change is proposed. O-11 is pinned at 2
  for all four routes (Architect, C.0 review); a change returns to Research Manager.

A residual near `1e-10` on a later counted row is a finding (plan §6). It does not
change pass/fail or the denominator. This inventory does not measure residuals.

## 4. Unsupported behavior

- Guessing a workload id where none was found.
- Dropping strict q6, or any other anchor, to clear a flag.
- Editing builders, running the M-F1a pipeline, or committing a bundle in this slice.
- Treating Phase-3 or Phase-3.1 tests as M-F1a counted evidence.
- Wording that the milestone exactness claim is closed.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-001 | doc review | `ROUTE_INVENTORY.md` row table | 16 rows; 4 routes × anchors 4/6/8/10; flags as §3.4 | DS-C0-1 |
| REQ-001 | doc review | entry citations in `ROUTE_INVENTORY.md` | four `execute_*` definitions present at the cited lines | DS-C0-1 |
| REQ-001, QA-008 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and `--strict` | task-4 at `step-4a` with no CLOSEOUT: `SLICE_MISSING_CLOSEOUT` warning, kept under `--strict`; after ET-C0-4: both modes clean | this mini-spec |

## 6. Affected interfaces

No code interface changes. Later C-slices will add sibling bundles. This slice does not.

## 7. Release and rollback

Rollback deletes this planning directory's uncommitted files. No generated artifact.
