# Engineering tasks — M-F1a slice 2 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect
> (Opus confirm bc-06c54d98 run 4); Step 4b under ADR-F1A-010 · **Parent:** `TASK_2_MINI_SPEC.md`,
> `DELIVERY_STORIES.md` · **Start condition:** met: four ADR-F1A-010 items stated unchanged ·
> **SDD stage:** step-4b-authorized
> Tech Lead hands off · **No push/PR**

Line numbers are lineage-report citations at rocky HEAD `43d8f074` (lineage cites `a2928bf1`)
(`../2026-10-05-mf1a-drift-lineage-report.md`). Re-anchor by function name and quoted text at
HEAD before editing; report any mismatch. KB line numbers are a hint only (Q-1).

## ADR-F1A-010 four-item verdict

This slice states each item unchanged. If any item is not unchanged, return to Research Manager before Step 4b.
1. The oracle, `execute_sequential_density_reference`.
2. QA-001 per ADR-F1A-002 and the regeneration comparators, apart from the single allowlist entry in ADR-F1A-009 Amendment 1.
3. The counted set: ADR-F1A-001 counting and realization rules and, once frozen, the reviewed manifest. No counted run precedes the freeze. This slice adds no counted cell.
4. The scope and the G-07 exit rule. Adding a required M-F1a sibling suite is not a G-07 change. Changing an exclusion, dropping, skipping, or demoting an included suite, or changing the exit rule is. This slice does none of those.

## Rules for every task

- Files changed by the Developer: only
  `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` and
  `tests/partitioning/evidence/test_correctness_evidence.py`. No `artifact_root` parameter.
  For every suite entry a test runs, set that module's `DEFAULT_OUTPUT_DIR` under `tmp_path`
  (fake module objects or `monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`), and also patch
  `validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no entry's output path resolves
  inside the repo before `run_pipeline` or `main` is called (Q-4). A change in
  `evidence_io.py` or the builders is not allowed.
- Do not touch generators, `squander/`, `evidence_io.py`, the q4 comparator, or any of the
  eight historical artifacts. Do not regenerate or commit the six Phase-3 bundles. Do not
  stage a regenerated q4 sibling in C1 or C2 (Q-2).
- Do not run standalone per-suite CLIs (`validation_scaffold.py`),
  `phase31_validation_pipeline.py`, or `performance_evidence/validation_pipeline.py` on rocky.
- **Restore (ADR-F1A-009 Amendment 1 consequence 3; ADR-F1A-011):** Before Slice A's C1,
  restore the eight historical paths from HEAD after every run (`git show HEAD:<path> > <path>`;
  TST-3; never stash, reset, checkout, or clean). From C1 on, run the REQ-007 eight-path
  diff before any restore; a non-empty result is a defect; the standing restore then stays
  as the safety check until Research Manager revises it. That standing restore
  (`test-density-matrix` SKILL.md lines 238-240) and ADR-F1A-011 are consistent. The skill
  edit itself is P0b. Restore the q4 sibling from HEAD after every run until Slice B lands;
  never stage it in C1 or C2.
- Wording: "historical suites verified, not written"; never "historical drift fixed".

## ET-A1 — Tests red-first (DS-A4; RC-2, RC-7)

File: `tests/partitioning/evidence/test_correctness_evidence.py`. Names indicative; every
Slice A test name must match the pytest selector `-k mf1a_historical` (RC-7). Each test has
its own acceptance line:

- **(a)** `test_mf1a_historical_suites_bytes_unchanged_after_run` — Acceptance: after a run,
  each of the eight historical bundles is byte-identical to its pre-run bytes. Red before
  ET-A2. Harness: seed `tmp_path` with byte copies of the eight real committed bundles at
  their relative paths; assert sha256 unchanged (Q-4).
- **(b)** `test_mf1a_historical_registered_siblings_still_written` — Acceptance: every registry entry
  with `entry.mf1a_sibling is True` has its bundle written under `tmp_path`. Green before and after
  (regression guard).
- **(c)** `test_mf1a_historical_all_statuses_returned_g07_unchanged` — Acceptance: all registered
  statuses are returned, and the returned status set equals the pre-change set; the G-07
  result equals the pre-change result for the same statuses. Green before and after. Existing
  G-07 tests (lineage cite `:360-400`, including the output-integrity exclusion at
  `:370-371`, `:397-398`) unchanged and green.
- **(d)** `test_mf1a_historical_output_dir_refuses_in_repo_and_symlink` — Acceptance: an
  in-repo path and an out-of-repo symlink resolving into the repo are both refused with exit
  code 2 and the exact stderr from Q-5 before any suite is built (assert no stub builder was
  called). Red before ET-A3.
- **(e)** `test_mf1a_historical_suite_without_sibling_field_not_written` — Acceptance: a registry entry
  without `entry.mf1a_sibling is True` is built and its status returned, but no bundle is written
  under `tmp_path`. A truthy non-bool `mf1a_sibling=1` is also rejected. Red before ET-A2.
- **(f)** `test_mf1a_historical_nonsibling_paths_match_req007` — Acceptance (RC-2 / Q-3): the
  test hard-codes the eight directories and asserts the sibling output-directory set is
  exactly `{mf1a/q4_baseline}`. Red before ET-A2 if the field is absent.
- **(g)** `test_mf1a_historical_output_dir_writes_outside_repo` — Acceptance (S-7): a positive
  `--historical-output-dir` outside the repo writes the mirrored layout
  `<dir>/<suite_dir>/<ARTIFACT_FILENAME>`, overwrites an existing file, creates no marker
  files, and prints a line containing `verified, written outside repo`. Red before ET-A3.

**Collect-only (B-2 / RC-7):** `pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -k mf1a_historical` collects exactly 7 tests: (a)–(g).

**Harness (Q-4):** No pytest runs the real heavy pipeline. Tests call `run_pipeline` and
`main` with a fake registry of stub builders that return small fixed bundles, keeping real
suite names. Stub both `build_cases` and `build_artifact_bundle`, or use a fake q4 module.
A real `build_cases` runs the heavy cell and a plain `git status`
(`mf1a_q4_baseline_validation.py:209-215`) that can refresh `.git/index`; a real
`build_artifact_bundle` reads the real q4 bundle via `DEFAULT_OUTPUT_PATH` (`:39`,
`:320-326`), which a `DEFAULT_OUTPUT_DIR` patch does not move. Neither writes the
worktree; the stubs keep the red-first test honest and avoid index mtime churn.
For every suite entry a test runs, set that module's
`DEFAULT_OUTPUT_DIR` under `tmp_path` (fake module objects or
`monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`), and also patch
`validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no entry's output path resolves
inside the repo before `run_pipeline` or `main` is called. No `artifact_root`. Tests assert
on the substring `verified, not written` where applicable (Q-6).

**RC-2 existing-test check:** Developer also checks whether any existing test asserts that
historical bundles are written. If one does, that is a Step 4a finding. Do not edit it
silently.

## ET-A2 — Verify-only write gate (DS-A1; Q-7)

File: `validation_pipeline.py`. Edit points (lineage cites):

- Suite registry (`:59-64`, `:69-70`): replace the tuples with `typing.NamedTuple` values
  `_CaseSuiteEntry(module, cases_attr, bundle_attr, mf1a_sibling: bool)` and
  `_NullarySuiteEntry(module, bundle_attr, mf1a_sibling: bool = False)`. Set
  `mf1a_sibling=True` on the `mf1a_q4_baseline` entry only. A full positional unpack still
  binds the original fields in today's order. `registered_suite_names()` output stays
  unchanged. Writing requires `entry.mf1a_sibling is True` (identity check on the bool).
  Do not key writing on the suite name, a path prefix, or a default; a false or absent
  flag means verify-only.
- `_write_slice_bundle` (`:52-53`, calls `write_artifact_bundle(bundle,
  module.DEFAULT_OUTPUT_DIR, module.ARTIFACT_FILENAME)`; the write itself is
  `evidence_io.py:15`, unchanged) and `run_pipeline` (`:104-120`, today builds and writes
  every registered suite unconditionally): build every suite; write only entries with
  `entry.mf1a_sibling is True`. Signature is keyword-only
  `historical_output_dir: Path | None = None`. Return `(name, status, Path | None)`: the
  path is the outside-repo copy for a historical suite when the flag is set, and None when
  that suite is verify-only without the flag. Sibling entries still return the written path.
- `main` (`:123-137`): report each verify-only suite as "verified, not written" (stdout
  format per Q-6, keeping `status=` and ASCII `|` separators).
- `g07_exit_passes`: not modified. No comparison against committed historical bytes.
- Acceptance: (a), (b), (c), (e), (f) green.

## ET-A3 — Opt-in `--historical-output-dir` (DS-A2; Q-5, Q-6)

File: `validation_pipeline.py` `main` (`:123-137`).

- Parse `--historical-output-dir <path>`.
- Repo root: walk up from `Path(__file__).resolve()` to the first ancestor that contains
  `.git` (file or directory). If none found, refuse (fail closed).
- Output: `Path(arg).expanduser().resolve(strict=False)`. Refuse if output equals root or
  `root in output.parents`. The refusal runs during argument handling, before
  `DEFAULT_OUTPUT_ROOT.mkdir` (lineage cite `:132`) and before any registry iteration.
- On refuse: exit 2; stderr exactly
  `refused: --historical-output-dir resolves inside the repository (<resolved>; repo <root>)`.
- Otherwise write verify-only bundles under that path after the normal run, mirroring
  relative layout `<dir>/<suite_dir>/<ARTIFACT_FILENAME>`; overwrite; no marker files.
  Never evidence; never committed.
- stdout lines per Q-6, with the `status=` token and ASCII `|` separators
  (siblings / verify-only / verify-only-with-flag).
- Acceptance: (d) and (g) green; (a) still green when the flag is absent.

## ET-A4 — REQ-007 path list + ADR fold / G-01 (DS-A3; planning role; RC-6; Q-1)

- REQ-007 row in `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md` and mini-spec §5: eight paths
  added on the rocky planning turn (HEAD `43d8f074`). Base `1cb3d20c` per Q-1.
- Base revision per Q-1: use `1cb3d20c` only if
  `git diff --exit-code 1cb3d20c a2928bf1 -- <eight>` is empty (Planner records on rocky
  write); else pin each path to last-touching commit (`fb865857` for the six) and list the
  pins in the REQ-007 row.
- **RC-6 (decided, Tech Lead 2026-10-05):** ADR fold and G-01 texts are on rocky uncommitted;
  Slice A C1 commits them with the two code files. P0b makes
  `ADR_AMENDMENTS_<SLUG>.md` checker-visible. Manual ADR-F1A-008 split until P1.
- Acceptance: the extended `git diff --exit-code` and `git status --porcelain
  --untracked-files=all` outputs are empty at C1, after the clean-start run (no counted cell), and at C2 (on the
  eight historical paths; q4 is out of this static set).

## ET-A5 — ADR-F1A-009 close sequence (Developer, Tester, Reviewer, Tech Lead; RC-1, RC-3, RC-7)

1. Developer: focused tests green
   (`conda run -n qgd --no-capture-output pytest
   tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_historical -v`).
   **RC-7:** every Slice A test name must match `-k mf1a_historical`.
2. (a) Reviewer implementation review. Diff limited to the two code files plus the C1
   planning paths in step 3. No `artifact_root` signature.
3. (b) C1, Tech Lead local commit. Staged paths are exactly these, and nothing else.
   Never stage the q4 sibling or any of the eight historical paths.
   - `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
   - `tests/partitioning/evidence/test_correctness_evidence.py`
   - `docs/specs/milestones/exactness-reconfirmation/task-2/TASK_2_MINI_SPEC.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-2/DELIVERY_STORIES.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-2/ENGINEERING_TASKS.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-2/README.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-2/STEP_4A_HANDBACK.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-2/ARCHITECT_STEP_4A_RULINGS.md`
   - `docs/specs/milestones/exactness-reconfirmation/ADRS_EXACTNESS_RECONFIRMATION.md`
   - `docs/specs/milestones/exactness-reconfirmation/ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`
   - `docs/specs/milestones/exactness-reconfirmation/DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md`
   - `docs/specs/milestones/exactness-reconfirmation/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-1/TASK_1_MINI_SPEC.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-1/DELIVERY_STORIES.md`
   - `docs/specs/milestones/exactness-reconfirmation/task-1/ENGINEERING_TASKS.md`
4. (c) Before the run, `git status --porcelain --untracked-files=all` is empty. Tester compares
   `sha256sum` of the loaded `_density_matrix_cpp` `.so` with `extension_identities[0].sha256`
   from `git show HEAD:benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json`
   (`05f01747…cc77` is display only; generalize this pin before Slice B). No rebuild during Slice A. This step is the clean-start run (no counted cell).
   Tester then runs
   `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
   in tmux `tester-sliceA-c` (TST-5), log under `/tmp/<run>/`. TST-3 snapshot before.
   **Acceptance (S-3 / RC-1 / Q-2), all of:**
   - exit 1;
   - no traceback on stderr;
   - nine stdout lines in registry order: eight `status=pass | verified, not written` and
     one q4 `status=fail | written`;
   - the regenerated q4 bundle is copied to `/tmp/<run>/` with its sha256 before restore;
   - the full field diff against `git show HEAD:<q4>` equals exactly the TST-4 set
     (`implementation_revision` plus the derived status, summary, and regeneration fields;
     `regeneration.first_mismatch` is `cases[0].provenance.implementation_revision`);
   - the REQ-007 eight-path diff runs before any restore and is empty; a non-empty result
     is a defect;
   - q4 is then restored from HEAD (`git show HEAD:<path> > <path>`) and recorded; never staged.
   An exit 1 with any other cause is a finding and goes to Tech Lead. Tester performs and
   signs the Q-2 difference-set check (Q-8).
5. (d) Real `CLOSEOUT.md` (ET-A6); `specs_check.sh` normal and `--strict` clean.
6. (e) Reviewer evidence review; REV-1 porcelain check (none of the eight staged; q4 not
   staged).
7. (f) C2, Tech Lead local commit.
8. (g) Same pre-check as step 4: empty porcelain is already true at clean C2. Tester again
   compares `sha256sum` of the loaded `_density_matrix_cpp` `.so` with
   `extension_identities[0].sha256` from
   `git show HEAD:benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json`
   (`05f01747…cc77` is display only; generalize this pin before Slice B), with no rebuild. Clean-C2 regeneration
   has the same acceptance as step 4 (S-3). The REQ-007 eight-path diff runs before any
   restore; a non-empty result is a defect; the standing restore of the eight then still
   runs as the safety check (ADR-F1A-009 Amendment 1 consequence 3). q4 restore continues
   after every run until Slice B lands; q4 is never staged.

A code change after C1 restarts at step (a). No push, no PR.

## ET-A6 — CLOSEOUT required sections (DS-A5; RC-5; Q-8)

- Summary: "historical suites verified, not written" under the M-F1a command; not a counted
  cell; no change to oracle, tolerances, counted set, scope, or G-07 exit set.
- Test results (a)–(g), the collect-only count of exactly 7, and existing G-07 tests.
- REQ-007 extended static diff output (empty on the eight paths).
- Containment record (consequence 3 of ADR-F1A-009 Amendment 1): from C1 on, the REQ-007
  eight-path diff ran before each restore and was empty; each standing safety restore of the
  eight, with run id and paths; every q4 restore (continues until Slice B), with run id.
- **Known hazards:** standalone per-suite CLIs (`validation_scaffold.py:23-53`),
  `phase31_validation_pipeline.py` (`:48`), `performance_evidence/validation_pipeline.py`
  (`:49-83`) still rewrite historical artifacts in place and are outside the M-F1a command.
  Standing rule: nobody runs them on rocky until a later decision.
- Statement that the six Phase-3 bundles were not regenerated or committed, and that
  committed bytes match `a2928bf1`.
- Pointer: lineage report; ADR-F1A-011; `ARCHITECT_STEP_4A_RULINGS.md`.
- **RC-5 required entries:**
  - Pre-(d) gate (ADR-F1A-009 Amendment 1): a written statement, with the diff stat, that no runtime or oracle code path changed
  - the per-suite status table from the (c) run
  - the q4 difference-set check result (Tester-signed; Q-2 / Q-8)
  - the Q-1 base or pins used
