# Slice A — Architect Step 4a rulings and required changes (2026-10-05, refine pass)

> **Status:** Architect refine pass on the Planner pack (README, MINI_SPEC, STORIES, ENGINEERING_TASKS,
> HANDBACK). Not code-ready. Opus 5.5 review follows on the revised pack. Base `a2928bf1`.

## Rulings Q-1…Q-8

**Q-1 — REQ-007 base for the eight paths.** Use a single base, `1cb3d20c`, only if
`git diff --exit-code 1cb3d20c a2928bf1 -- <eight paths>` is empty. Planner checks this read-only
when writing to rocky and records the output. If it is not empty, do not use `1cb3d20c` for these
paths. Instead pin each path to the last commit that touched it at or before `a2928bf1`
(`git log -1 --format=%H a2928bf1 -- <path>`; `fb865857` for the six per lineage), and list the
pins in the REQ-007 row. Rationale: the check must assert "unchanged from the committed historical
record", not hide a pre-existing difference. Line numbers: re-anchor at rocky HEAD after P0. KB
line 172 is a hint only.

**Q-2 — q4 sibling at Slice A runs: option (i), restore, do not commit.** Slice A is non-counted
pipeline hygiene and must not create new q4 counted evidence. At (c) and (g) the q4 sibling is
still written (it is allowlisted). Its regeneration fails only on
`cases[0].provenance.implementation_revision`, so G-07 exit 1 is expected. Acceptance is exactly
the q4 CLOSEOUT step-8 / TST-4 difference set: the revision field plus the fields that follow from
that mismatch (`status` fail, `summary.first_failure` regeneration, `regeneration.prior_present`
true, `regeneration.pass` false, `regeneration.first_mismatch` = the revision path). Any other
difference (numeric, categorical, identity, manifest) is a finding: stop and report to Tech Lead.
Every other suite status must equal the pre-change status. After each run the q4 bundle is
restored from HEAD (`git show HEAD:<path> > <path>`) and recorded. It is never staged in C1 or
C2. The carve-out is ADR-F1A-009 Amendment 1 consequence 2, not a recount under consequence 1:
until Slice B's allowlist lands, in Slice A only, a q4 regeneration failure whose full field
diff equals the TST-4 set is the expected outcome, not a stop. The RM-approved rule exempts
only `implementation_revision`, and Slice B will enforce that in code. That recount path is
not used, because recounting q4 at a hygiene commit adds churn with no evidential gain. q4 restore continues after Slice A's C2, until Slice B lands. For the eight historical
paths, Tech Lead 2026-10-05: before Slice A's C1, restore after every run; from C1 on, the
REQ-007 eight-path diff runs before any restore and a non-empty result is a defect, then the
standing restore stays as the safety check until Research Manager revises it. That skill rule
and ADR-F1A-011 are consistent (SKILL.md lines 238-240). The skill edit is P0b.

**Q-3 — registry count.** The rule decides, not a number: every registry entry without the sibling
field is verify-only. Expected at HEAD: nine registered outputs, one sibling (`mf1a_q4_baseline`)
and eight historical (the six Phase-3 suites plus `correctness_matrix` and
`summary_consistency`). The lineage report's "other seven" is a miscount, because its own cited
lines `:59-64` and `:69-70` are eight entries. Developer confirms at HEAD and records the actual
count in the handback. Test (c) asserts the returned status set equals the pre-change set. A test
asserts that the set of non-sibling output paths derived from the registry equals the eight REQ-007
paths, so the spec list and the code cannot diverge silently.

**Q-4 — test harness.** No pytest runs the real heavy pipeline (long-test rule). Tests call
`run_pipeline` and `main` with a fake registry of stub builders that return small fixed bundles.
For every suite entry a test runs, set that module's `DEFAULT_OUTPUT_DIR` under `tmp_path`
(fake module objects or `monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`), and also patch
`validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no entry's output path resolves
inside the repo before `run_pipeline` or `main` is called. No change outside the two files.
- (a) Seed `tmp_path` with byte copies of the eight real committed bundles at their relative
  paths, run, and assert sha256 is unchanged.
- (b) and (e): assert presence or absence of files under `tmp_path`.
- (d): assert no stub builder was called.
- The real end-to-end proof is the clean-start run (no counted cell) plus the extended REQ-007
  static diff. Neither is a pytest.
- Keep real suite names. Stub both `build_cases` and `build_artifact_bundle`, or use a
  fake q4 module. A real `build_cases` runs the heavy cell and `git status`
  (`mf1a_q4_baseline_validation.py:209-215`); a real `build_artifact_bundle` reads
  `DEFAULT_OUTPUT_PATH` (`:39`, `:320-326`), which a `DEFAULT_OUTPUT_DIR` patch does not
  move. For every suite entry a test runs, set that
  module's `DEFAULT_OUTPUT_DIR` under `tmp_path` (fake module objects or
  `monkeypatch.setattr(mod, "DEFAULT_OUTPUT_DIR", …)`), and also patch
  `validation_pipeline.DEFAULT_OUTPUT_ROOT`. A fixture asserts no entry's output path resolves
  inside the repo before `run_pipeline` or `main` is called. `run_pipeline` takes keyword-only
  `historical_output_dir: Path | None = None` and returns `(name, status, Path | None)`.
  There is no `artifact_root` parameter. A change in `evidence_io.py` or the builders is not
  allowed.

**Q-5 — in-repo detection.**
- Repo root: walk up from `Path(__file__).resolve()` to the first ancestor that contains `.git`
  (a file or a directory, so worktrees are safe). If none is found, refuse (fail closed).
- Output: `Path(arg).expanduser().resolve(strict=False)`.
- Refuse if the output equals the root or `root in output.parents`. The refusal runs during
  argument handling, before `DEFAULT_OUTPUT_ROOT.mkdir` and before any registry iteration.
- Exit code 2, and print on stderr exactly
  `refused: --historical-output-dir resolves inside the repository (<resolved>; repo <root>)`.
- Out of scope: a symlink race created after the check.

**Q-6 — layout and output format.**
- Layout under the dir mirrors the in-repo relative path:
  `<dir>/<suite_dir>/<ARTIFACT_FILENAME>`, the same relative path as under
  `artifacts/correctness_evidence/`.
- The command overwrites existing files there and creates no marker files.
- stdout prints one line per registered suite, in registry order. Keep the `status=` token
  and use an ASCII `|` separator:
  - `<suite>: status=<status> | written <repo-relative path>` for siblings
  - `<suite>: status=<status> | verified, not written` for verify-only suites
  - `<suite>: status=<status> | verified, written outside repo <path>` with the flag
- Tests assert on the substring `verified, not written`.

**Q-7 — field name.** `mf1a_sibling` on a `typing.NamedTuple`: `_CaseSuiteEntry(module,
cases_attr, bundle_attr, mf1a_sibling: bool)` and `_NullarySuiteEntry(module, bundle_attr,
mf1a_sibling: bool = False)`. Writing requires `entry.mf1a_sibling is True`, an identity check
on the bool. Truthy strings or ints do not count, so the check fails closed. Set it on the
`mf1a_q4_baseline` entry only. Positional unpacking of the full entry still binds the original
fields in today's order. `registered_suite_names()` output stays unchanged.

**Q-8 — pre-(d) gate.** Slice A adds no counted cell and no new oracle comparison. The
CLOSEOUT and handback record a written statement, with the diff stat, that no runtime or
oracle code path changed (ADR-F1A-009 Amendment 1). The Q-2 acceptance check is the
scientific guard for this slice; Tester performs and signs it.

## Required changes (RC)

- **RC-1 (mini-spec §5, ET-A5 step 4, handback §4).** Acceptance at (c) and (g) is all of:
  exit 1; no traceback on stderr; nine stdout lines in registry order (eight
  `status=pass | verified, not written` and one q4 `status=fail | written`); the regenerated
  q4 bundle copied to `/tmp/<run>/` with sha256 before restore; the full field diff against
  `git show HEAD:<q4>` equals the TST-4 set. The REQ-007 eight-path diff runs before any
  restore and is empty. An exit 1 with any other cause is a finding.
- **RC-2 (ET-A1).** Add test (f): the set of non-sibling output paths derived from the registry
  equals the eight REQ-007 paths (Q-3). Developer also checks whether any existing test asserts
  that historical bundles are written. If one does, that is a Step 4a finding. Do not edit it
  silently.
- **RC-3 (ET-A5 step 8 and mini-spec §3.7).** Restore is ADR-F1A-009 Amendment 1 consequence 3.
  Before Slice A's C1, restore the eight historical paths after every run. From C1 on, the
  REQ-007 eight-path diff runs before any restore and a non-empty result is a defect; the
  standing restore then stays as the safety check until Research Manager revises it. The q4
  sibling is restored after every run until Slice B closes.
- **RC-4 (handback §1, tolerances line).** Add: "q4 regeneration accepted under option (i) per
  Q-2; no change to the comparator".
- **RC-5 (ET-A6 CLOSEOUT).** Add four entries:
  - a written statement, with the diff stat, that no runtime or oracle code path changed
  - the per-suite status table from the (c) run
  - the q4 difference-set check result
  - the Q-1 base or pins used
- **RC-6 (mini-spec §2.1 and ET-A4).** Write the ADR fold and the G-01 rewrite only once. If P0
  already carries them, Slice A cites P0. Tech Lead decides which commit carries them and records
  that in the README.
- **RC-7 (ET-A5 step 1).** Give the pytest selector in Layer 4, `-k mf1a_historical`.
  Tests (b), (c), and (e) contain `mf1a_historical`. `--collect-only -k mf1a_historical`
  collects exactly 7, which is (a)–(f) plus test (g).

## Verdict

The pack is correctly scoped and matches ADR-F1A-011. It is not code-ready yet: RC-1…RC-7, plus
the Q-1 and Q-3 confirmations at HEAD, are needed first, followed by the Opus review.
