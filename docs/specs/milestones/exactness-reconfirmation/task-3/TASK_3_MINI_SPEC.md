# Task / Work Package 3: q4 regeneration comparator allowlist
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a slice 3 (Slice B) ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Planning-base HEAD:**
> `99bf9d519f7aac58d8f1e6502c60912decb85995` · **Lock:** KB
> `2026-10-05-SLICE-B-COMPARATOR-ALLOWLIST-LOCK.md` sha256
> `cbc7c08160f745e1922a1ad4e0b7f71427a1bf3c81bf884bae52b13563118d09` ·
> **Traces:** REQ-004, REQ-006 · QA-008 · ADR-F1A-008, ADR-F1A-009 (+ Amendment 1),
> ADR-F1A-010 · **No push/PR** · **Baseline route verified:** q4 history only

## 1. Purpose

Slice B replaces option (i) for the q4 sibling only. The allowlist constant has length 1.
`_allowlisted_revision_difference` skips `cases[0].provenance.implementation_revision`
only when both values are distinct full lowercase 40-hex git revisions. A pass still
requires every other compared field to match (§3.2). No bundle schema change. The
regenerated q4 bundle is never committed.
Baseline route verified applies to q4 history only. This slice adds no counted cell.

Live bundle at this HEAD:
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json`.
Git blob `92855fdfce7fa838e5efb15100bb9458962b2611`. File sha256
`483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`. `len(cases)` is 1.
Committed `cases[0].provenance.implementation_revision` is
`a50ae79f636afcd97627424f6460a0e552184345`. `regeneration` is
`{prior_present: false, pass: true, first_mismatch: null}`. `status` is `pass`.

## 2. Scope

### 2.1 Developer paths (exactly three)

| Path | Edit |
|------|------|
| `benchmarks/density_matrix/correctness_evidence/mf1a_q4_baseline_validation.py` | `Q4_REGENERATION_ALLOWLIST`, `_is_full_git_revision`, `_allowlisted_revision_difference`, provenance-loop skip in `_regeneration_result` |
| `tests/partitioning/evidence/test_correctness_evidence.py` | §3.5 tests only (lock §5) |
| `.cursor/skills/test-density-matrix/references/regeneration-acceptance.md` | milestone-agnostic replacement (lock §6) |

`.cursor/skills/test-density-matrix/SKILL.md` is unchanged (lines 200–204 stay byte-identical).
No fourth Developer path. Planner owns this directory's Layer 2–4 and the checklist N6
edits. The Developer does not author those during Step 4b. `task-3/README.md` is not
required. No `CHANGE_CONTROL.md` (C-4: the allowlist is Research Manager approved scope).

### 2.2 Out of scope

Any other allowlist entry. Edits to `_QA001_REGENERATION_TOLERANCES` or the `>` residual
compare. G-07 (`g07_exit_passes`, exclusions, included-suite tests). Historical suite
comparators and their JSON. Slice A (`validation_pipeline.py` write allowlist, the eight
historical paths). C.0 and every C-slice. A schema version bump or a new bundle field.
`evidence_io.py`. `validation_pipeline.py`. `squander/` and every C++/CMake file. An
extension rebuild. Committing `mf1a_q4_baseline_bundle.json`. Task-1 and task-2 files,
including their option (i) text. ADR files. `SKILL.md`.
`references/validation-pipeline-restore.md`. `pytest.ini`. A placeholder CLOSEOUT. A
waiver for the expected C1 `SLICE_MISSING_CLOSEOUT`. Widening the allowlist to clear an
acceptance failure.

## 3. Required behavior

### 3.1 Allowlist constant

Module-level tuple in `mf1a_q4_baseline_validation.py`, immediately after
`_QA001_REGENERATION_TOLERANCES` (dict ends at line 63 at the lock HEAD):

```python
Q4_REGENERATION_ALLOWLIST = (
    "cases[0].provenance.implementation_revision",
)
```

Length is 1. The star string `cases[*].provenance.implementation_revision` is not an
entry. Resolved form is `cases[0]` because the live comparator emits `cases[0].…`, the
bundle has one case, and TST-4 / ADR-F1A-009 Amendment 1 consequence 2 name that path.
Later slices do not edit this tuple.

### 3.2 Predicate and provenance loop

`_regeneration_result` (lines 329–423) stays private. The only production caller is
`build_artifact_bundle` (line 464). Add `_is_full_git_revision` and
`_allowlisted_revision_difference` in the same file. No new import. Do not use `re`.

The predicate is true only when `len(Q4_REGENERATION_ALLOWLIST) == 1`, `path` equals
`Q4_REGENERATION_ALLOWLIST[0]`, both current and prior are 40-char lowercase hex, and
they differ. Both sides are checked. No case-folding. No `strip`.

Keep every provenance key, including `implementation_revision`. On a true predicate,
`continue` (do not return). The first later mismatch is reported. A non-revision on
either side is not skipped. `regeneration.prior_present` still flips false → true on a
rerun that finds the committed file. That flip is derived. It is not an allowlist entry.
Do not write `prior_present: false` when a prior file exists.

Check order the tests pin (first reported path wins): `manifest_exact_set` (current
cases only; comparator not called), then `bundle_structure`, then `exact_paths`, then
`extension_identities`, then `input_artifact_identities`, then the revision skip, then
the remaining provenance keys, then categorical qa001, then residual `>` compares.
`summary.first_failure` order: `manifest_exact_set`, `route_realization`, `qa001`,
`provenance`, `regeneration`. A current-side manifest-cell key
(`anchor_qbits`, `workload`, `route`, `planner_setting.max_partition_qubits`) stops at
`manifest_exact_set`. Do not move `capture_provenance`.

### 3.3 Schema

Comparator-only. These strings stay exactly:

- `BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_q4_baseline_bundle_v1"`
- `RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_q4_baseline_case_v1"`
- `MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_q4_baseline_manifest_v1"`

Do not add a JSON field that records allowlist application. `regeneration` keeps exactly
`prior_present`, `pass`, `first_mismatch`. Do not add top-level `manifest`,
`tolerances`, `suite_name`, `status`, `summary`, `regeneration`, or
`non_counted_context` to the comparator in this slice. The §3.8 proof-run diff still requires top-level `schema_version`, `suite_name`, `manifest`, and `tolerances` to equal the committed bundle.

### 3.4 What is not allowlisted

Fail means `regeneration.pass` is false, except a current-side manifest-cell change,
which fails at `manifest_exact_set` before `_regeneration_result`. Not allowlisted:
any second exact field, extension path or sha256, input identity, dependency version,
environment, schema or manifest version, categorical route/realization/workload/seeds,
`clean_start` or `dirty_paths`, residual above the frozen comparators
(`> 1e-10` Frobenius, max-abs, trace deviation; `> 1e-12` `lambda_min`), a revision
value that is not a full lowercase 40-hex SHA on either side, and a second case
(`manifest_exact_set` or `bundle_structure`). A residual inside the comparator plus a
full-SHA revision change passes; the residual is not an allowlist entry. Do not deepen
`dependencies` or `extension_identities` into sub-key paths.

### 3.5 Tests

File: `tests/partitioning/evidence/test_correctness_evidence.py`, after
`test_mf1a_q4_baseline_regeneration_rejects_categorical_or_residual_drift` (ends line
361). Do not change that test or
`test_mf1a_q4_baseline_regeneration_accepts_frozen_residuals`. `pytest.ini` addopts
ignore this directory; every command clears addopts. Do not add a `density_matrix`
marker and do not edit `pytest.ini`.

Every new test passes `prior_bundle=` explicitly and deep-copies both sides (`cases` is
stored by reference). A module-scoped fixture calls
`build_cases(provenance=_clean_mf1a_provenance())` once with revision `"a" * 40`. Do not
refactor the two existing regeneration tests onto that fixture. Sub-cases are loops or
sequential assertions, not `@pytest.mark.parametrize`, unless the Developer states a
different collect-only count in the Layer 4 task and the Reviewer checks that number.
With the locked shape, `--collect-only -k mf1a_q4_baseline_regeneration` collects **14**
(2 existing + 12 new). Names, setups, and asserts are in `ENGINEERING_TASKS.md` ET-B1.

Red before the comparator change (Reviewer records both red and green): `allowlist_is_length_one`; `passes_on_revision_only`; `fails_on_revision_plus_second_field` (the function is red because sub-case 2 expects `clean_start`; sub-cases 1, 3, and 4 would pass in isolation); `fails_on_dependency_version`; `fails_on_environment_identity`; `fails_on_residual_above_comparator_despite_allowlist`; `passes_when_residual_within_comparator`. Guards that pass before and after: extension sha256, input identity, manifest version, `rejects_non_revision_value` (both sides), and `allowlist_does_not_cover_a_second_case`. `rejects_non_revision_value` current side (prior `"a" * 40`): `"g" * 40`, `"c" * 39`, `""`, `"C" * 40`, missing key. Prior side (current `"c" * 40`): `"A" * 40`, missing prior key.

Developer pytest gate (before C1 and again on the C1 tree). These commands do not run
`validation_pipeline.py`:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a_q4_baseline_regeneration" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a_q4_baseline_regeneration"
```

`-k mf1a` covers the ten existing q4 tests, the seven `mf1a_historical` tests, and the
twelve new regeneration tests. Collect-only must report 14.

### 3.6 Skill reference

Replace `references/regeneration-acceptance.md` with the milestone-agnostic text in
ET-B3. When the milestone ADR allowlists `provenance.implementation_revision` (for
M-F1a, ADR-F1A-009 Amendment 1), a difference in only that field passes: exit 0,
`status` pass, `regeneration.pass` true, `first_mismatch` null. Otherwise expect the
non-zero shape that milestone ADR documents. Never widen an allowlist to make a run
pass. q4 pins and the two-path checklist stay in this slice and in §3.8. They do not
move into the reference. Leave `references/validation-pipeline-restore.md` alone.

### 3.7 Close (ADR-F1A-009 (a)–(g), Slice A shape)

The regenerated q4 bundle is never committed. Committed q4 bytes stay blob
`92855fdfce7fa838e5efb15100bb9458962b2611` after (c) and after (g).

- **(a)** Reviewer reviews the uncommitted Developer diff plus task-3 Layer 2–4. An optional planning-docs C0 may precede C1. Headers are synced before (a).
- **(b) C1** contains the three Developer files, task-3 Layer 2–4, the ADR-F1A-010 four-item verdict, and `**SDD stage:** step-4b-authorized`. No q4 bundle. No `CLOSEOUT.md`.
- **(c)** From clean C1, run the §3.8 proof once. Copy the regenerated q4 file to `/tmp/<run>/` before restore. Restore it. Never stage it.
- A change to code, tests, or planning docs between C1 and (c) restarts the slice at (a).
- **Pre-(d)** Written statement, with `git diff --stat <C1 parent> <C1>`, that no runtime or oracle code path changed. Inside `mf1a_q4_baseline_validation.py` the diff is the allowlist constant, `_is_full_git_revision`, `_allowlisted_revision_difference`, and the provenance-loop skip. `evaluate_mf1a_qa001`, `build_cases`, and `capture_provenance` are unchanged. This slice adds no counted cell. This statement is the pre-(d) gate. It is not a new G-10 essay.
- **(d)** Write `task-3/CLOSEOUT.md` citing C1, with the §3.8 table filled from the (c) run. Normal and `--strict` checks are then clean.
- **(e)** Reviewer evidence review.
- **(f) C2** contains the CLOSEOUT plus checklist touch-ups only. No bundle. No code.
- **(g)** From clean C2, rerun §3.8. The new `implementation_revision` equals the C2 SHA. Restore the q4 file. Never stage it.

Between C1 and the CLOSEOUT write, `ENGINEERING_TASKS.md` is `step-4b-authorized` and
`CLOSEOUT.md` is absent. `check_artifacts.py --strict` then reports exactly one finding:
`SLICE_MISSING_CLOSEOUT` for task-3, promoted to an error. That failure is expected and
is recorded. No placeholder CLOSEOUT. No waiver. After (d), normal and
`--strict` are clean.

### 3.8 Acceptance (steps (c) and (g))

Before each proof run, porcelain is empty and `sha256sum` of
`squander/density_matrix/_density_matrix_cpp.cpython-313-x86_64-linux-gnu.so` equals
`05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`. A mismatch stops the
run. Command:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

Compare the regenerated q4 JSON to `git show HEAD:` of that path (parsed JSON; byte
equality is not the acceptance):

| Check | Required |
|-------|----------|
| Process exit | 0 |
| q4 `status` | `pass` |
| `regeneration.pass` | true |
| `regeneration.first_mismatch` | null |
| `summary.first_failure` | null |
| Case-scoped recursive diff | exactly `cases[0].provenance.implementation_revision`, new value equal to `git rev-parse HEAD` |
| Full-file recursive diff | exactly that path, and `regeneration.prior_present` false → true |
| Third path | fail acceptance. Do not add it to the allowlist |
| `schema_version`, `suite_name`, `manifest`, `tolerances` | equal to the committed bundle |
| `non_counted_context` | `{}` |
| Schema ids | case carries `record_schema_version` and `manifest_schema_version`; bundle carries `schema_version`; all three stay the v1 strings in §3.3 |
| `extension_identities` | equal to the committed array, including sha256 `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |

At (g) the new revision equals the C2 SHA. A failed (c) stops before (d). Extension or environment mismatch goes to Tech Lead; any other mismatch, including a within-comparator third path, goes to Research Manager. A failed (g) uses the same routing and does not amend C2. Never widen the allowlist. Restore with `git show HEAD:<q4> > <q4>` (never stash, reset, checkout, or clean) after copying the regenerated JSON to `/tmp/<run>/`. Restored sha256 is `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` and porcelain is empty. A diff on any of the eight historical paths is a defect (ADR-F1A-011), not a Slice B restore. A Python-only C1 does not change the extension SHA unless somebody rebuilds; a rebuild fails at `extension_identities` and goes to Tech Lead. Do not allowlist `dependencies` or `environment`.

## 4. Unsupported behavior

- A second allowlist entry, a star path, or dropping `implementation_revision` from the provenance key tuple.
- Skipping a non-revision, an uppercase hex, or a prior that fails `_is_full_git_revision`.
- Schema version bumps, new bundle fields, tolerance edits, G-07 edits, or committing regenerated q4.
- Editing `SKILL.md`, task-1, task-2 option (i) paragraphs, ADRs, or `validation_pipeline.py`.
- Starting C.0. Widening the allowlist after an acceptance failure.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-004, QA-008 | regeneration unit tests | `pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_q4_baseline_regeneration` | 14 collected; red-before list fails at unmodified HEAD; all 14 pass after the patch | DS-B2; ET-B1 |
| REQ-004, REQ-006 | module gate | `pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a` | existing q4 tests, seven historical tests, and twelve new tests pass | ET-B1 |
| REQ-004, QA-008 | proof runs (c) and (g) | `python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | §3.8 table; extension pin `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`; q4 restored, not committed | DS-B4; ET-B4 |
| REQ-004 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and the same command `--strict` | At `step-4a`, both modes warn `SLICE_MISSING_CLOSEOUT` for task-3. Between C1 and (d), with stage `step-4b-authorized`, `--strict` reports that finding as one error. After (d), both modes are clean. | this mini-spec |

## 6. Affected interfaces and rollback

`_regeneration_result` gains one skip. `references/regeneration-acceptance.md` expected-exit text follows the ADR allowlist. Unchanged: `SKILL.md`, `validation_pipeline.py`, bundle schemas, `g07_exit_passes`, `capture_provenance`, QA-001 tolerances. Rollback reverts the three Developer paths. The committed q4 bundle is not part of the change.
