# Task / Work Package 2: historical suites verify-only under the M-F1a command
> **Status:** closed code-ready under ADR-F1A-008 on 2026-10-05 · **Slice:** M-F1a slice 2
> (historical sibling-bundle drift; RM-3) · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Planning-base revision:** `1cb3d20c` · **Parent HEAD:** `a2928bf1` (q4 C2) · **Rocky HEAD:**
> `43d8f074` (P0 planning parent) · **Rocky directory:** `docs/specs/milestones/exactness-reconfirmation/task-2/` · **Scope:** `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
> and `tests/partitioning/evidence/test_correctness_evidence.py` only ·
> **Traces:** REQ-004, REQ-007 · CAP-007 · QA-008 · ADR-F1A-004, ADR-F1A-005, ADR-F1A-008,
> ADR-F1A-009 (+ Amendment 1), ADR-F1A-010, ADR-F1A-011 · G-07 ·
> **Authorization:** Research Manager Slice A go (via Tech Lead, 2026-10-05, "exactly as
> designed", plus ADR-F1A-011); Step 4b only after Architect closes Step 4a code-ready under
> ADR-F1A-008 (ADR-F1A-010); C1 only after Reviewer implementation review · **No push/PR**

## 1. Purpose and slice boundary

Stop the single M-F1a command (`validation_pipeline.py`) from rewriting Phase-3 historical
bundles in place. Historical suites are still built and still feed G-07, but they are
verified, not written.

Lineage (`../2026-10-05-mf1a-drift-lineage-report.md`): HISTORICAL-RECORD BLOCKER. The six
bundles `correctness_package`, `output_integrity`, `runtime_classification`,
`sequential_correctness`, `external_correctness`, `unsupported_boundary` are Phase-3
evidence; Phase 3 papers quote their 25/4/17 counts. None feeds the frozen 26-case matrix
(17/9/0), which is "Table 2" in the CSCS 2026 short paper. Slice A does not touch Table 2
evidence. Committed bytes at rocky HEAD `43d8f074` (lineage cites `a2928bf1`) are intact.

Baseline route verified: N/A here. This slice is pipeline hygiene, not a counted cell. It
adds no case, route, anchor, or counted row and moves nothing toward "exactness reconfirmed".

## 2. Scope

### 2.1 In scope

- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`: suite registry,
  `_write_slice_bundle`, `run_pipeline`, `main` (lineage line refs at rocky HEAD `43d8f074` (lineage cites `a2928bf1`): `:52-53`,
  `:59-64`, `:69-70`, `:104-120`, `:123-137`; re-anchor at HEAD after P0 — KB line numbers
  are a hint only per Q-1).
- `tests/partitioning/evidence/test_correctness_evidence.py`: red-first tests (a)–(g) (§3.5).
- Planning-role edits: REQ-007 row (§3.6); ADR fold in `ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`
  with index in `ADRS_EXACTNESS_RECONFIRMATION.md`; G-01 cites ADR-F1A-010 (on rocky, uncommitted;
  **RC-6:** Slice A C1 commits with code). P0b makes that companion checker-visible. P1
  stage-aware `check_artifacts` follows Slice A C1; until then the manual ADR-F1A-008 split
  holds, as it did for q4 (`README.md`, handback layout note).
- Real `CLOSEOUT.md` with the known-hazard list (§3.7).

### 2.2 Out of scope

- Regenerating or committing the six historical JSON bundles (needs a separate ADR-F1A-005
  decision).
- Any generator, any `squander/` code, `evidence_io.py`, or an `artifact_root` parameter.
  Harness (Q-4): for every suite entry a test runs, set that module's `DEFAULT_OUTPUT_DIR`
  under `tmp_path` (fake module objects or `monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`),
  and also patch `validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no entry's output
  path resolves inside the repo before `run_pipeline` or `main` is called.
- G-07 semantics: `g07_exit_passes`, the exit set, and both exclusions
  (`correctness_evidence_external_correctness`, whole `correctness_evidence_output_integrity`).
- The q4 regeneration comparator and its allowlist (Slice B).
- Route inventory (C.0) and route-group delivery (C.1…C.n).
- The out-of-command rewriters: standalone per-suite CLIs (`validation_scaffold.py:23-53`),
  `phase31_validation_pipeline.py`, `performance_evidence/validation_pipeline.py`. ADR-F1A-011
  does not change them; they are a CLOSEOUT known hazard.
- Gitignore entries or any `clean_start` path exclusion.
- `TECH_STACK.md:43` (milestone close, ADR-F1A-007).

## 3. Required behavior

### 3.1 Write allowlist (ADR-F1A-011 decision 1; Q-7)

The M-F1a command writes a bundle only for suites explicitly registered as M-F1a siblings.
Registry values are `typing.NamedTuple`s: `_CaseSuiteEntry(module, cases_attr, bundle_attr,
mf1a_sibling: bool)` and `_NullarySuiteEntry(module, bundle_attr, mf1a_sibling: bool = False)`.
Original fields stay in that order, with `mf1a_sibling` last, so a full positional unpack
still binds them as today, and a nullary entry constructed as `(module, bundle_attr)` defaults
the flag to false. `registered_suite_names()` output (names and order) stays unchanged.
Writing requires `entry.mf1a_sibling is True` (identity on the bool; truthy strings or ints
do not count). Test (e) also asserts a truthy non-bool `mf1a_sibling=1` is not written.
Set it on the `mf1a_q4_baseline` entry only. A suite name pattern, a path
prefix, or a default does not count. A suite without the field true is verify-only, so new
or unknown suites fail closed. Later route-group siblings are written by setting the field,
with no further change to the write logic.

### 3.2 Verify-only: built, not written (decision 2; Q-3)

Every verify-only suite is still built in memory. Its status is returned exactly as today
and feeds G-07 and `summary_consistency` unchanged. `g07_exit_passes` is not modified. The
command reports each such suite as "verified, not written". The rule decides, not a number:
every registry entry without the sibling field is verify-only. Expected at HEAD: nine
registered outputs — one sibling (`mf1a_q4_baseline`) and eight historical (the six Phase-3
suites plus `correctness_matrix` and `summary_consistency`). The lineage report's "other
seven" is a miscount (its own cited lines `:59-64` and `:69-70` are eight entries). Developer
confirms the actual count at HEAD and records it in the handback. Test (f) hard-codes the
eight directories and asserts the sibling output-directory set is exactly `{mf1a/q4_baseline}`.

### 3.3 No historical comparison (decision 3)

Verify-only suites are not compared against their committed bytes. Drift does not affect
the exit code. Drift is a known pre-existing condition, not an M-F1a finding.

### 3.4 Opt-in out-of-repo output (decision 4; Q-5, Q-6)

`--historical-output-dir <path>` may write verify-only bundles for inspection, only to a path
that resolves outside the repository working tree.

**In-repo detection (Q-5):**
- Repo root: walk up from `Path(__file__).resolve()` to the first ancestor that contains
  `.git` (a file or a directory, so worktrees are safe). If none is found, refuse (fail
  closed).
- Output: `Path(arg).expanduser().resolve(strict=False)`.
- Refuse if the output equals the root or `root in output.parents`. The refusal runs during
  argument handling, before `DEFAULT_OUTPUT_ROOT.mkdir` (today `main` at lineage cite `:132`)
  and before any registry iteration.
- Exit code 2; print on stderr exactly:
  `refused: --historical-output-dir resolves inside the repository (<resolved>; repo <root>)`.
- Out of scope: a symlink race created after the check.

**Layout and stdout (Q-6):**
- Layout under the dir mirrors the in-repo relative path:
  `<dir>/<suite_dir>/<ARTIFACT_FILENAME>`, the same relative path as under
  `artifacts/correctness_evidence/`.
- The command overwrites existing files there and creates no marker files.
- stdout prints one line per registered suite, in registry order. Keep the `status=` token
  and use ASCII separators (no em dash):
  - `<suite>: status=<status> | written <repo-relative path>` for siblings
  - `<suite>: status=<status> | verified, not written` for verify-only suites
  - `<suite>: status=<status> | verified, written outside repo <path>` with the flag
- Tests assert on the substring `verified, not written`.

Without the flag, no verify-only bundle is written anywhere.

### 3.5 Required tests (decision 7; Q-3, Q-4)

| Id | Test (names indicative; final names in Layer 4; every name must match the Layer 4 pytest selector per RC-7) |
|----|-------------------------------------------------|
| (a) | After a run, all eight historical bundles are byte-identical to their pre-run state |
| (b) | Every registered M-F1a sibling is still written |
| (c) | All registered statuses are still returned, with the same G-07 result as before (returned status set equals the pre-change set) |
| (d) | An in-repo `--historical-output-dir`, including through a symlink, is refused before building |
| (e) | A suite without `mf1a_sibling is True` is not written, including truthy non-bool `mf1a_sibling=1` |
| (f) | Hard-coded eight directories; sibling output-directory set is exactly `{mf1a/q4_baseline}` (Q-3 / RC-2) |
| (g) | Positive `--historical-output-dir`: mirrored layout, overwrite, no marker files, and a `verified, written outside repo` line (S-7) |

**Harness (Q-4):** No pytest runs the real heavy pipeline. Tests call `run_pipeline` and
`main` with stub builders and real suite names. Stub both `build_cases` and
`build_artifact_bundle`, or use a fake q4 module. A real `build_cases` runs the cell and
`git status` (`mf1a_q4_baseline_validation.py:209-215`); a real `build_artifact_bundle`
reads `DEFAULT_OUTPUT_PATH` (`:39`, `:320-326`), which a `DEFAULT_OUTPUT_DIR` patch does
not move. For every suite entry, set `DEFAULT_OUTPUT_DIR` under `tmp_path` (fake modules
or `monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`) and also patch
`validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no output path is inside the
repo before `run_pipeline` or `main`. No `artifact_root`. Keyword-only
`historical_output_dir: Path | None = None`; return `(name, status, Path | None)`.
- (a) Seed `tmp_path` with byte copies of the eight real committed bundles at their relative
  paths, run, and assert sha256 is unchanged.
- (b) and (e): assert presence or absence of files under `tmp_path`.
- (d): assert no stub builder was called.
- (f): derive non-sibling output paths from the registry; assert equality with the eight
  REQ-007 paths.
- The real end-to-end proof is the clean-start run (no counted cell) plus the extended REQ-007 static diff.
  Neither is a pytest.
- A change in `evidence_io.py` or the builders is not allowed. Redirect is the monkeypatch
  above, not a new root parameter.

All of (a)–(g) are red first against `a2928bf1` behavior where applicable ((a), (d), (e),
(f), (g) must fail before the change; (b) and (c) are regression guards that stay green). The
existing G-07 tests (lineage cite `test_correctness_evidence.py:360-400`) stay green
unchanged. Developer also checks whether any existing test asserts that historical bundles
are written; if one does, that is a Step 4a finding — do not edit it silently (RC-2).

### 3.6 REQ-007 static diff (decision 5; Q-1)

The eight historical artifact directories are added to the REQ-007 historical-boundary
static diff, both to `git diff --exit-code <base> -- …` and to
`git status --porcelain --untracked-files=all -- …`:

Eight directories (listed individually; not the parent `artifacts/correctness_evidence/` root):
see `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md` REQ-007 row and `README.md` § Q-1.

**Base revision (Q-1 CLOSED):** `1cb3d20c` (`git diff --exit-code 1cb3d20c 43d8f074 -- <eight>` empty at
HEAD `43d8f074`). Registry: 9 suites (1 sibling + 8 historical). Table 2 = CSCS 2026 frozen-26
(17/9/0); none of the six Phase-3 bundles feeds it. Else-pin: `ARCHITECT_STEP_4A_RULINGS.md` Q-1.

### 3.7 Close and CLOSEOUT (ADR-F1A-009 + Amendment 1; ADR-F1A-011; Q-2; RC-3)

- ADR-F1A-009 two-commit close (a)–(g): Reviewer implementation review → C1 → clean-start run
  → real CLOSEOUT, strict clean → Reviewer evidence review → C2 → clean-C2 regeneration with
  outputs restored.
- **Restore (ADR-F1A-009 Amendment 1 consequence 3; ADR-F1A-011):** before Slice A's C1,
  restore the eight historical bundles after every run (`git show HEAD:<path> > <path>`;
  never stash, reset, checkout, or clean) and record it. From C1 on, the pipeline no longer
  writes those eight. The standing restore in `test-density-matrix` `SKILL.md` lines 238-240
  and ADR-F1A-011 are consistent: that restore stays a safety check until Research Manager
  revises it. The skill edit itself is P0b. From C1
  on, the REQ-007 eight-path diff runs before any restore, and a non-empty result is a
  defect. None of the eight is staged in C1 or C2. q4 is restored from HEAD after every run
  until Slice B, and is never staged.
- **Acceptance at (c) and (g) (S-3 / RC-1 / Q-2):** exit 1; no traceback on stderr; nine
  stdout lines in registry order (eight `status=pass | verified, not written` and one q4
  `status=fail | written`); copy the regenerated q4 bundle to `/tmp/<run>/` with its sha256
  before restore; the full field diff against `git show HEAD:<q4>` equals only the TST-4
  set (ET-A5 step 4). Before (c) and (g), confirm the loaded `_density_matrix_cpp` SHA-256
  is `05f01747…cc77` (S-8). No rebuild during Slice A. Any other exit-1 cause or diff is a
  finding and goes to Tech Lead.
- CLOSEOUT section "Known hazards" (Research Manager requirement): the standalone per-suite
  CLIs (`validation_scaffold.py`), `phase31_validation_pipeline.py`, and the
  `performance_evidence` validation pipeline still rewrite historical artifacts in place and
  are outside the M-F1a command. Standing rule: nobody runs them on rocky until a later
  decision.

### 3.8 Wording

"Historical suites verified, not written." Never "historical drift fixed". No
"exactness reconfirmed". No "baseline route verified" for this slice.

## 4. Unsupported behavior

- Writing any suite that lacks the M-F1a-sibling registry field (`mf1a_sibling is True`).
- Any byte comparison of historical suites that gates the exit code.
- Any in-repo historical output directory, with or without a gitignore entry.
- Any change to statuses, summaries, schema ids, or G-07 inclusion/exclusion.
- Regenerating, restaging, or committing any of the eight historical artifacts.
- Staging or committing a regenerated q4 sibling in Slice A's C1 or C2 (Q-2).
- Edits to generators, `squander/`, `evidence_io.py`, or the q4 comparator.
- Running the out-of-command rewriters on rocky.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-004, REQ-007, QA-008 | verify-only tests (a)–(g) | `pytest …/test_correctness_evidence.py -k mf1a_historical` (RC-7; full command in `ENGINEERING_TASKS.md` ET-A5 step 1) | (a)–(g) pass; collect-only collects exactly 7; G-07 tests unchanged | DS-A1…A4; ET-A1…A3 |
| REQ-004, G-07 | pipeline at C1 (c) and C2 (g) | `validation_pipeline.py` (full command in ET-A5) | S-3 acceptance in §3.7 | DS-A1; ET-A5 |
| REQ-007 | static historical review | `git diff --exit-code 1cb3d20c --` paths in `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md` REQ-007 row (includes §3.6 eight dirs) plus matching `git status --porcelain --untracked-files=all --` | both outputs empty | DS-A3; ET-A4 |
| REQ-004, REQ-007 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` (normal and `--strict`) | zero errors; strict clean after the real CLOSEOUT | this mini-spec |

## 6. Affected interfaces

- `validation_pipeline.py`: registry entries become the NamedTuples in §3.1, with
  `mf1a_sibling` true only on `mf1a_q4_baseline`; `run_pipeline` gains keyword-only
  `historical_output_dir` and returns `(name, status, Path | None)`; `main` gains
  `--historical-output-dir`; stdout keeps `status=` (§3.4). No `artifact_root`.
- Unchanged: suite builders, `evidence_io.py`, bundle schemas, `g07_exit_passes`, exit-code
  semantics, the single command name and invocation.
- Breaking changes: none for the M-F1a command. Default invocation writes fewer files.

## 7. Release and rollback

Rollback reverts the `validation_pipeline.py` and test changes only; no artifact or generator change.

## 8. Architect rulings

Q-1…Q-8 CLOSED in `ARCHITECT_STEP_4A_RULINGS.md`; Q-1 record in §3.6; handback §5 mirrors.
