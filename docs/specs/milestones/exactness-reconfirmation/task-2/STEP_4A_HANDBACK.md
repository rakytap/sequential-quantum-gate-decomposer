# Step 4a handback — M-F1a slice 2 checklist

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect
> (Opus confirm bc-06c54d98 run 4) · **Effect when closed:** Step 4b and the Developer/Tester
> handoff under ADR-F1A-010, performed by Tech Lead · **No push/PR**

## 1. ADR-F1A-010 four-item statement (must be explicit)

This slice does not change any of them. Architect confirms each line:

- [x] **Oracle unchanged:** `execute_sequential_density_reference`. Slice A does not call or
  change it.
- [x] **QA-001 and comparators unchanged:** QA-001 per ADR-F1A-002 and the regeneration
  comparators, apart from the single allowlist entry in ADR-F1A-009 Amendment 1.
  q4 regeneration is accepted under option (i) per Q-2; the comparator itself is unchanged
  (RC-4).
- [x] **Counted set unchanged:** ADR-F1A-001 counting and realization rules and, once frozen,
  the reviewed manifest. No counted run precedes the freeze. Slice A adds no counted cell.
- [x] **Scope and G-07 exit rule unchanged:** adding a required M-F1a sibling suite is not a
  G-07 change. Changing an exclusion, dropping, skipping, or demoting an included suite, or
  changing the exit rule is. Slice A does none of those. `g07_exit_passes` is unchanged.

If any line cannot be stated unchanged, Slice A returns to Research Manager.

## 2. Files touched

C1 stages exactly these paths (B-1). Never stage the q4 sibling or any of the eight
historical paths. Before step (c), `git status --porcelain --untracked-files=all` is empty.

- [ ] `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
- [ ] `tests/partitioning/evidence/test_correctness_evidence.py`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-2/TASK_2_MINI_SPEC.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-2/DELIVERY_STORIES.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-2/ENGINEERING_TASKS.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-2/README.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-2/STEP_4A_HANDBACK.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-2/ARCHITECT_STEP_4A_RULINGS.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/ADRS_EXACTNESS_RECONFIRMATION.md`
  (bodies match `43d8f074`; continuation index only)
- [x] `docs/specs/milestones/exactness-reconfirmation/ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-1/TASK_1_MINI_SPEC.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-1/DELIVERY_STORIES.md`
- [x] `docs/specs/milestones/exactness-reconfirmation/task-1/ENGINEERING_TASKS.md`
- [ ] `task-2/CLOSEOUT.md` at step (d) only; it is not a C1 path

No generator, `squander/`, `evidence_io.py`, `artifact_root`, historical JSON, or gitignore.
P0b landed at `5a5168fd`. `ADR_AMENDMENTS_<SLUG>.md` is checker-visible (slug check and
the 400-line budget). Manual line count of
`ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md` is 201 of 400.
`ADRS_EXACTNESS_RECONFIRMATION.md` is the `43d8f074` bodies plus the continuation index.

## 3. Non-goals

- [ ] No regeneration or commit of the six Phase-3 bundles.
- [ ] No historical byte comparison gating the exit code.
- [ ] No in-repo historical output directory and no `clean_start` path exclusion.
- [ ] No change to the out-of-command rewriters (standalone per-suite CLIs,
  `phase31_validation_pipeline.py`, `performance_evidence` pipeline); they are a CLOSEOUT
  known hazard and nobody runs them on rocky.
- [ ] No comparator work (Slice B) and no inventory (C.0) or counted C-slice run before
  Slice A closes.
- [ ] No staging or committing a regenerated q4 sibling in C1 or C2 (Q-2 option (i)).
- [ ] No "historical drift fixed" and no "exactness reconfirmed" wording.

## 4. Code-ready content checks

- [ ] Write allowlist is `entry.mf1a_sibling is True` on the NamedTuple registry (S-1),
  fail-closed, set only on `mf1a_q4_baseline` (ADR-F1A-011 decision 1; Q-7).
- [ ] Verify-only suites built in memory, statuses and G-07 unchanged, reported "verified,
  not written" (decisions 2–3; Q-6 stdout).
- [ ] `--historical-output-dir` refuses in-repo paths per Q-5 (exit 2; exact stderr),
  including via symlink, before building (decision 4).
- [ ] Eight REQ-007 paths listed individually; base/pins per Q-1 (decision 5).
- [ ] Tests (a)–(g) each have an engineering-task acceptance line (decision 7; RC-2; S-7);
  (b), (c), and (e) names contain `mf1a_historical`; `--collect-only -k mf1a_historical`
  collects exactly 7 (B-2 / RC-7).
- [ ] ADR-F1A-009 (a)–(g) close. Restore is Amendment 1 consequence 3: before C1, restore
  the eight after every run; from C1 on, REQ-007 eight-path diff before any restore
  (non-empty is a defect), then the standing restore stays as the safety check until
  Research Manager revises it. That skill rule (SKILL.md lines 238-240) and ADR-F1A-011
  are consistent. Skill edit is P0b. q4 restored until Slice B, never staged.
- [ ] **Acceptance at (c) and (g) (S-3):** exit 1; no traceback; nine stdout lines (eight
  `status=pass | verified, not written`, one q4 `status=fail | written`); q4 copy and
  sha256 under `/tmp/<run>/` before restore; field diff equals only the TST-4 set.
  Extension SHA-256 `05f01747…cc77` before (c) and (g); no rebuild (S-8). Tester signs
  the Q-2 difference-set check (Q-8).
- [ ] CLOSEOUT "Known hazards" section specified; RC-5 entries present (pre-(d) written
  statement, with the diff stat, that no runtime or oracle code path changed;
  per-suite status table; q4 difference-set result; Q-1 base/pins).
- [ ] Line anchors re-checked at HEAD after P0 (Q-1).
- [x] Registry count confirmed at HEAD `43d8f074` (Q-3: 9 = 1 sibling + 8 historical).

## 5. Architect rulings (Q-1…Q-8 CLOSED)

From `ARCHITECT_STEP_4A_RULINGS.md` / `TASK_2_MINI_SPEC.md` §8:

- **Q-1 CLOSED.** `1cb3d20c` only if empty vs `a2928bf1` on the eight; else per-path pins
  (`fb865857` for the six). Recorded: single base `1cb3d20c` (Planner verified `git diff --exit-code 1cb3d20c 43d8f074 -- <eight paths>` empty at rocky HEAD `43d8f07425359baee41fbbe8db5add5421264d81`, 2026-10-05).
- **Q-2 CLOSED.** Option (i). Non-counted hygiene; do not recount q4. At (c)/(g): S-3
  acceptance (exit 1, no traceback, nine stdout lines, q4 copy and sha256 before restore,
  TST-4 field diff only). q4 restore until Slice B, never staged. Eight-path rule is
  ADR-F1A-009 Amendment 1 consequence 3: check before restore from C1 on; standing restore
  remains the safety check.
- **Q-3 CLOSED.** Rule not number; expect 9 = 1 sibling + 8 historical; lineage "seven"
  miscount; test (f) locks registry ↔ REQ-007.
- **Q-4 CLOSED.** Stub builders, fake registry, real suite names. Stub both `build_cases`
  and `build_artifact_bundle`, or use a fake q4 module. A real `build_cases` runs the heavy
  cell and `git status` (`mf1a_q4_baseline_validation.py:209-215`); a real
  `build_artifact_bundle` reads `DEFAULT_OUTPUT_PATH` (`:39`, `:320-326`), which a
  `DEFAULT_OUTPUT_DIR` patch does not move. For
  every suite entry, set that module's `DEFAULT_OUTPUT_DIR` under `tmp_path` (fake modules
  or `monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`) and also patch
  `validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no entry's output path
  resolves inside the repo before `run_pipeline` or `main`. No `artifact_root`. Keyword-only
  `historical_output_dir`; return `(name, status, Path | None)`.
- **Q-5 CLOSED.** Repo root = first ancestor with `.git`; `resolve(strict=False)`; refuse if
  equals/under root; exit 2; exact stderr as in mini-spec §3.4.
- **Q-6 CLOSED.** Mirror relative layout; overwrite; no markers; stdout lines as in
  rulings; tests assert substring `verified, not written`.
- **Q-7 CLOSED.** NamedTuple field `mf1a_sibling`; `entry.mf1a_sibling is True`; only on
  `mf1a_q4_baseline`.
- **Q-8 CLOSED.** Pre-(d) gate is a written statement, with the diff stat, that no runtime
  or oracle code path changed. Tester signs the Q-2 difference-set check.

## 6. Verdict

- [x] Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect (Opus confirm
  bc-06c54d98 run 4). Per ADR-F1A-010, §1 stated unchanged: oracle unchanged; QA-001 and
  comparators unchanged except the ADR-F1A-009 Amendment 1 allowlist entry; counted set
  unchanged; scope and G-07 exit rule unchanged. Effect when closed: Step 4b and the
  Developer/Tester handoff under ADR-F1A-010, performed by Tech Lead. No push/PR.
