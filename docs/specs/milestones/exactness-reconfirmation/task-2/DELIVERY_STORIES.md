# Delivery stories — M-F1a slice 2 (Layer 3)

> **Status:** Revised for Architect refine rulings (`ARCHITECT_STEP_4A_RULINGS.md`);
> closed code-ready under ADR-F1A-008 on 2026-10-05 · **Parent:** `TASK_2_MINI_SPEC.md` · **ADR:** ADR-F1A-011
> (`ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`) · **No push/PR**

Wording for every story: "historical suites verified, not written". Never "historical drift
fixed". Baseline route verified: N/A (not a counted cell).

## DS-A1 — Verify-only write gate

**As** the M-F1a pipeline owner, **I want** the M-F1a command to write only suites
explicitly registered as M-F1a siblings, **so that** no run rewrites Phase-3 evidence.

- Registry entries are the NamedTuples in the mini-spec §3.1; writing requires
  `entry.mf1a_sibling is True` (Q-7 / S-1). Set only on `mf1a_q4_baseline`. Not a
  hard-coded name check, a name pattern, a path prefix, or a default.
- A suite without the field is built in memory, its status is returned and feeds G-07 and
  `summary_consistency` unchanged, and it is reported "verified, not written".
- No byte comparison of historical suites; drift does not affect the exit code.
- `g07_exit_passes` unchanged.
- Acceptance: tests (a), (b), (c), (e), (f).
- Traces: ADR-F1A-011 decisions 1–3, 6; REQ-004; G-07; Q-3, Q-7.

## DS-A2 — Opt-in historical dump directory with in-repo refusal

**As** a reviewer who wants to inspect drift, **I want** an opt-in
`--historical-output-dir <path>`, **so that** verify-only bundles can be written for
inspection without touching the repo.

- In-repo detection per Q-5: repo root = first ancestor of module with `.git`;
  `resolve(strict=False)`; refuse if equals or under root; exit 2; exact stderr:
  `refused: --historical-output-dir resolves inside the repository (<resolved>; repo <root>)`.
- Layout mirrors in-repo relative path; overwrite; no markers; stdout keeps `status=` and
  uses an ASCII `|` separator (Q-6); tests assert substring `verified, not written`.
- The in-repo refusal runs before `DEFAULT_OUTPUT_ROOT.mkdir`.
- Outputs are never evidence and never committed. Without the flag nothing historical is
  written. `run_pipeline` takes keyword-only `historical_output_dir` and returns
  `(name, status, Path | None)`.
- Acceptance: test (d) and test (g) `test_mf1a_historical_output_dir_writes_outside_repo`
  (mirrored layout, overwrite, no marker files, `verified, written outside repo`).
- Traces: ADR-F1A-011 decision 4; Q-5, Q-6.

## DS-A3 — REQ-007 path list

**As** the milestone reviewer, **I want** the eight historical artifact directories in the
REQ-007 static diff, **so that** any tracked or untracked change to them fails the
historical-boundary check.

- Paths: `benchmarks/density_matrix/artifacts/correctness_evidence/<suite>/` for
  `correctness_package`, `output_integrity`, `runtime_classification`,
  `sequential_correctness`, `external_correctness`, `unsupported_boundary`,
  `correctness_matrix`, `summary_consistency`; listed individually, not the parent root.
- Added to both the `git diff --exit-code` and the `git status --porcelain
  --untracked-files=all` halves of the REQ-007 row in `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md`
  and in this slice's mini-spec §5.
- Planning-role edit, not the implementer. Base revision per Q-1 CLOSED (mini-spec §3.6 /
  §8): `1cb3d20c` only if empty vs `a2928bf1`; else per-path pins (`fb865857` for the six).
- Traces: ADR-F1A-011 decision 5; REQ-007; ADR-F1A-005; Q-1.

## DS-A4 — Tests red-first

**As** the Developer, **I want** tests (a)–(g) written before the change, **so that** the
behavior is pinned.

- (a) eight historical bundles byte-identical after a run; (b)
  `test_mf1a_historical_registered_siblings_still_written`; (c)
  `test_mf1a_historical_all_statuses_returned_g07_unchanged`; (d) in-repo
  `--historical-output-dir`, including via symlink, refused before building and before
  mkdir; (e) `test_mf1a_historical_suite_without_sibling_field_not_written`; (f) hard-codes
  the eight directories and asserts the sibling set is exactly `{mf1a/q4_baseline}`; (g)
  positive out-of-repo `--historical-output-dir` (S-7).
- `--collect-only -k mf1a_historical` collects exactly 7.
- Harness per Q-4: stub builders, fake registry, `tmp_path`; no heavy pipeline in pytest.
- (a), (d), (e), (f), (g) fail before the change; (b), (c) stay green as regression guards.
  Existing G-07 tests stay green unchanged. Developer checks existing tests that assert
  historical bundles written → Step 4a finding if found; do not edit silently (RC-2).
- Only `tests/partitioning/evidence/test_correctness_evidence.py` changes. Every Slice A
  test name must match the Layer 4 selector `-k mf1a_historical` (RC-7).
- Traces: ADR-F1A-011 decision 7; QA-008; Q-3, Q-4; RC-2, RC-7.

## DS-A5 — CLOSEOUT known-hazard standing rule

**As** Research Manager, **I want** the Slice A CLOSEOUT to list the remaining in-place
rewriters and the standing rule, **so that** nobody rewrites Phase-3 evidence outside the
M-F1a command.

- CLOSEOUT section "Known hazards": standalone per-suite CLIs (`validation_scaffold.py`),
  `phase31_validation_pipeline.py`, `performance_evidence` validation pipeline. Standing rule:
  nobody runs them on rocky until a later decision. ADR-F1A-011 does not change them.
- CLOSEOUT records ADR-F1A-009 Amendment 1 consequence 3: before C1, restore the eight
  after every run; from C1 on, the REQ-007 eight-path diff runs before any restore
  (non-empty is a defect) and the standing restore stays as the safety check. The skill
  rule at `test-density-matrix` SKILL.md lines 238-240 and ADR-F1A-011 are consistent.
  The skill edit is P0b. q4 is restored after every run until Slice B and never staged.
- CLOSEOUT states that the six bundles were not regenerated or committed.
- CLOSEOUT also records (RC-5 / Q-8): a written statement, with the diff stat, that no
  runtime or oracle code path changed; per-suite status table from the (c) run; q4
  difference-set check result (Tester-signed); Q-1 base or pins used.
- Traces: ADR-F1A-011 known hazard and consequences; ADR-F1A-009 Amendment 1 consequence 3;
  Q-2, Q-8; RC-3, RC-5.
