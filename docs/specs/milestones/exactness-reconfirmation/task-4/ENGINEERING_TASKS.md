# Engineering tasks — M-F1a slice C.0 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect · **Slice:** M-F1a C.0 ·
> **Parent:** `TASK_4_MINI_SPEC.md`, `DELIVERY_STORIES.md` ·
> **Traces:** REQ-001 · QA-001, QA-008 · ADR-F1A-001, ADR-F1A-008,
> ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **Inventory revision:** `0e8299e9f48361ece2cc1665d81d4b2a35b4f156`
> **SDD stage:** step-4b-authorized
> **No push/PR** · baseline route verified for q4 only

C.0's Step 4b is this docs close. Do not share the stage field with a middot.
This close does not authorize C.1 Step 4b. No waiver.

## ADR-F1A-010 planning statement

This slice states each item unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged.
2. QA-001 and the regeneration comparators, apart from the single allowlist entry in ADR-F1A-009 Amendment 1.
3. The counted set unchanged. This slice adds no counted cell. Strict q6, q8, and q10 stay in the denominator.
4. The scope and the G-07 exit rule unchanged. Item 4 is both.

## Rules for every task

No Developer code in this slice. Do not edit `workloads.py`, the three Slice B code
paths, `SKILL.md`, or `validation_pipeline.py`. Do not run the pipeline. Do not rebuild
the extension. `test_QX2` stays deselected.

## ET-C0-1 — Inventory rows (DS-C0-1)

**Implements delivery story**

- DS-C0-1. Traces: REQ-001.

**Change type**

- docs

**Definition of done**

- `ROUTE_INVENTORY.md` has 16 rows, the shared-field table, and the flag section.
- Realization values match the cited tests. Unshown cells are `not_shown` or
  `TBD inventory`, not a guessed pass.

**Execution checklist**

- [x] Read entries, roadmap, architecture overview, and workload builders at this HEAD
- [x] Record 16 rows without a pipeline run
- [ ] Architect reviews the row table before any code-ready claim

**Evidence produced**

- Doc review of `ROUTE_INVENTORY.md`.

**Risks / rollback**

- Risk: a later code read shows a q6 strict builder this pass missed.
- Rollback / mitigation: correct the row and, if the flag changes, return to Research Manager. Do not drop the cell in silence.

## ET-C0-2 — C-slice order (DS-C0-2)

**Implements delivery story**

- DS-C0-2. Traces: REQ-001.

**Change type**

- docs

**Definition of done**

- Order is C.1 fused, C.2 hybrid, C.3 strict, C.4 baseline q6/q8/q10.
- Each of C.1–C.4 states ADR-F1A-009 (a)–(g).
- q10 runtime is `TBD — measure`. No numeric guess.

**Execution checklist**

- [x] Record the order. C.2 (hybrid) and C.3 (strict) stay separate.
- [x] Architect re-close accepts C.1 fused, C.2 hybrid, C.3 strict, C.4 baseline, with C.2 and C.3 separate

**Evidence produced**

- Order table in `TASK_4_MINI_SPEC.md` §3.3 and `ROUTE_INVENTORY.md`.

**Risks / rollback**

- Risk: a later merge of C.2 and C.3 hides the strict flags.
- Rollback / mitigation: they stay separate, and the flags stay on the strict rows.

## ET-C0-3 — RM flag list (DS-C0-3)

**Implements delivery story**

- DS-C0-3. Traces: REQ-001, QA-001.

**Change type**

- docs

**Definition of done**

- Strict q6, q8, and q10 are `no_eligible_workload` on historical builders and are named with the RM-1(a) disposition.
- They are realizable via the C.3 M-F1a-only family, not by an edit to `workloads.py`.
- The four ADR-F1A-010 items above stay a planning statement.

**Execution checklist**

- [x] Write the flag section
- [x] Architect review is recorded and C.1 code has not started

**Evidence produced**

- Flag section of `ROUTE_INVENTORY.md`.

**Risks / rollback**

- Risk: counting strict q6–q10 on an unreviewed workload.
- Rollback / mitigation: C.3 Step 4a includes the RM-1(a) design and cannot close until that design is reviewed and frozen. C.1 and C.2 do not wait.

## ET-C0-4 — Close C.0 (DS-C0-2)

**Implements delivery story**

- DS-C0-2. Traces: REQ-001.

**Change type**

- docs

**Definition of done**

- After the Architect re-close, one Planner writer pass sets `**SDD stage:** step-4b-authorized` (ADR-F1A-008 Amendment 1 bound 4), states the four ADR-F1A-010 items unchanged as the verdict, and updates the "not code-ready" headers.
- The same pass writes `task-4/CLOSEOUT.md`, status `shipped`: `ROUTE_INVENTORY.md` accepted at `0e8299e9` with the Architect rulings; no code, no counted evidence, no pipeline run; the strict q6/q8/q10 `no_eligible_workload` findings on historical builders carried forward; RM-1(a) recorded (M-F1a-only family at C.3), not left open; reproduce with both `specs_check.sh` commands.
- Normal and `--strict` are fully clean (0 errors, 0 warnings).
- Reviewer checks the porcelain (exactly the five task-4 files, the checklist, and the task-3 mini-spec), then Tech Lead makes one local commit staged by exact path list. ADR-F1A-009 does not apply. No push, no PR. This commit lands before any C.1 (task-5) write.

**Execution checklist**

- [x] Architect re-close recorded (`/tmp/c0-step4a/C0_STEP4A_RECLOSE.md`)
- [x] Writer pass: stage, verdict, headers, CLOSEOUT
- [ ] Both lint modes clean; Reviewer; Tech Lead commit

**Evidence produced**

- `task-4/CLOSEOUT.md`; clean normal and `--strict` output.

**Risks / rollback**

- Risk: the stage flips without the closeout and `--strict` fails. Mitigation: one writer pass does both.
