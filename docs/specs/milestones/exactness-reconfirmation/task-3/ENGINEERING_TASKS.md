# Engineering tasks — M-F1a slice 3 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a slice 3 (Slice B) ·
> **Parent:** `TASK_3_MINI_SPEC.md`, `DELIVERY_STORIES.md` ·
> **Traces:** REQ-004, REQ-006 · QA-008 · ADR-F1A-008, ADR-F1A-009 (+ Amendment 1),
> ADR-F1A-010 · **Planning-base HEAD:** `99bf9d519f7aac58d8f1e6502c60912decb85995`
> **SDD stage:** step-4b-authorized
> **No push/PR** · baseline route verified for q4 history only

Stage is now `step-4b-authorized` so that value lands in C1.
Do not share the stage field with a middot. Between C1 and CLOSEOUT, `--strict` reports exactly one error `SLICE_MISSING_CLOSEOUT` for task-3 (expected; record it; no placeholder; no waiver).

## ADR-F1A-010 four-item verdict

This slice states each item unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged.
2. QA-001 and the regeneration comparators, apart from the single allowlist entry in ADR-F1A-009 Amendment 1.
3. The counted set unchanged. This slice adds no counted cell.
4. The scope and the G-07 exit rule unchanged. Item 4 is both. Stating G-07 alone does not state scope.

No further Research Manager round. Item 2's single exception is the allowlist entry
`cases[0].provenance.implementation_revision`. Scope stays the q4 comparator and the
regeneration-acceptance reference. G-07 is not edited.

## Rules for every task

Developer edits during Step 4b are exactly these three paths:

- `benchmarks/density_matrix/correctness_evidence/mf1a_q4_baseline_validation.py`
- `tests/partitioning/evidence/test_correctness_evidence.py`
- `.cursor/skills/test-density-matrix/references/regeneration-acceptance.md`

`SKILL.md` is unchanged. Planner paths are not Developer edits:
`docs/specs/milestones/exactness-reconfirmation/task-3/TASK_3_MINI_SPEC.md`,
`DELIVERY_STORIES.md`, `ENGINEERING_TASKS.md`, and
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. `task-3/CLOSEOUT.md` is step (d),
committed only in C2. Do not invent a fourth Developer path. Do not start C.0.
Never widen the allowlist to make a run pass. Regenerated q4 is never committed.
Do not edit task-1 or task-2 option (i) paragraphs, ADRs, `validation_pipeline.py`,
`pytest.ini`, or `references/validation-pipeline-restore.md`.

## ET-B1 — Tests red-first (DS-B2)

**Implements delivery story**

- DS-B2. Traces: REQ-004, REQ-006, QA-008.

**Change type**

- tests

**Definition of done**

- Twelve new functions are appended after `test_mf1a_q4_baseline_regeneration_rejects_categorical_or_residual_drift`. The two existing regeneration tests keep their assertions.
- Each new test passes `prior_bundle=` and deep-copies cases before building a prior and again before mutating either side. Fixture revision is `"a" * 40`. Helper revisions `"c" * 40` and `"d" * 40` are full lowercase hex. No `@pytest.mark.parametrize`.
- `--collect-only -k mf1a_q4_baseline_regeneration` collects exactly 14.
- Except `allowlist_is_length_one` (no circuit), `rejects_non_revision_value` (sets its own values), and the `bundle_structure` assertion of `fails_on_manifest_version`, every new test sets the **current** `implementation_revision` to `"c" * 40` and keeps the prior at the fixture `"a" * 40` (lock §5 Setup). In `allowlist_does_not_cover_a_second_case`, current case 1 carries `"c" * 40`. That difference is what makes sub-case 2 of `fails_on_revision_plus_second_field`, `fails_on_dependency_version`, `fails_on_environment_identity`, `fails_on_residual_above_comparator_despite_allowlist`, and `passes_when_residual_within_comparator` red at unmodified HEAD.

**Tests**

| Test | Assert |
|------|--------|
| `test_mf1a_q4_baseline_regeneration_allowlist_is_length_one` | tuple equals `("cases[0].provenance.implementation_revision",)`; `len == 1`; star string absent; three schema ids are the v1 literals; `set(_QA001_REGENERATION_TOLERANCES) == {frobenius_norm_diff, max_abs_diff, trace_abs_deviation, lambda_min}` with values `1e-10`, `1e-10`, `1e-10`, `1e-12` |
| `test_mf1a_q4_baseline_regeneration_passes_on_revision_only` | current `"c" * 40`; `status == "pass"`; `regeneration.pass` true; `first_mismatch` is None; `prior_present` true; `summary.first_failure` is None; regeneration keys are exactly those three; `schema_version` unchanged; case diff is only the revision path |
| `test_mf1a_q4_baseline_regeneration_fails_on_revision_plus_second_field` | (1) current `seed_policy` → `cases[0].seed_policy` and `summary.first_failure == "regeneration"`; (2) current `clean_start` false, `provenance_pass` left true → `cases[0].provenance.clean_start` and `regeneration`; (3) prior `route = "wrong"` → `cases[0].route` and `regeneration`; (4) current `realization.actual_fused_execution = True` → `cases[0].realization` and `summary.first_failure == "route_realization"` |
| `test_mf1a_q4_baseline_regeneration_fails_on_extension_sha256` | prior `extension_identities[0].sha256` is `"e" * 64` → `cases[0].provenance.extension_identities`; second assertion changes only the prior `path` and expects the same mismatch |
| `test_mf1a_q4_baseline_regeneration_fails_on_input_identity` | prior `input_artifact_identities` is one `{path, sha256}` object; current stays `[]` |
| `test_mf1a_q4_baseline_regeneration_fails_on_dependency_version` | prior `dependencies["numpy"]` is `"other"` → `cases[0].provenance.dependencies` |
| `test_mf1a_q4_baseline_regeneration_fails_on_environment_identity` | prior `environment["python_version"]` is `"3.13.1"` → `cases[0].provenance.environment` |
| `test_mf1a_q4_baseline_regeneration_fails_on_manifest_version` | prior bundle `schema_version` `"other"` → `bundle_structure`; prior case `manifest_schema_version` `"other"` → `cases[0].manifest_schema_version` |
| `test_mf1a_q4_baseline_regeneration_fails_on_residual_above_comparator_despite_allowlist` | prior Frobenius increased by `2e-10` → `cases[0].qa001.frobenius_norm_diff` and `summary.first_failure == "regeneration"` |
| `test_mf1a_q4_baseline_regeneration_passes_when_residual_within_comparator` | prior Frobenius increased by `5e-11` → pass, `first_mismatch` is None |
| `test_mf1a_q4_baseline_regeneration_rejects_non_revision_value` | every case: `pass` false and `first_mismatch == "cases[0].provenance.implementation_revision"`. Current side: `"g" * 40`, `"c" * 39`, `""`, `"C" * 40`, missing key. Prior side with current `"c" * 40`: `"A" * 40`, missing prior key |
| `test_mf1a_q4_baseline_regeneration_allowlist_does_not_cover_a_second_case` | two current cases → `manifest_exact_set`; direct `_regeneration_result` with prior `cases` length 2 → `bundle_structure` |

**Red before the comparator change**

These seven fail at unmodified HEAD: `allowlist_is_length_one`, `passes_on_revision_only`, `fails_on_revision_plus_second_field` (sub-case 2; the function is red because of that sub-case), `fails_on_dependency_version`, `fails_on_environment_identity`, `fails_on_residual_above_comparator_despite_allowlist`, `passes_when_residual_within_comparator`.

These pass before and after: extension sha256, input identity, manifest version, `rejects_non_revision_value` (including both prior-side cases), `allowlist_does_not_cover_a_second_case`.

**Execution checklist (TDD: red → green → refactor)**

- [ ] Write the twelve tests first
- [ ] Run them at unmodified HEAD and confirm the seven red-before failures
- [ ] After ET-B2, confirm all 14 collected regeneration tests pass and `-k mf1a` passes
- [ ] Do not run `validation_pipeline.py` as this gate

**Evidence produced**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a_q4_baseline_regeneration" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a_q4_baseline_regeneration"
```

Collect-only reports 14.

**Risks / rollback**

- Risk: parametrizing a sub-case changes the collected count.
- Rollback / mitigation: if the Developer parametrizes, this task states the new count and the Reviewer checks that number. Otherwise the pin stays 14.

## ET-B2 — Comparator skip (DS-B1)

**Implements delivery story**

- DS-B1. Traces: REQ-004, REQ-006.

**Change type**

- code

**Definition of done**

- `Q4_REGENERATION_ALLOWLIST` is the length-1 tuple in the mini-spec §3.1, placed immediately after `_QA001_REGENERATION_TOLERANCES`.
- `_is_full_git_revision` and `_allowlisted_revision_difference` match the lock predicate: both sides full lowercase 40-hex, path equals the single allowlist entry, values differ. No `re`. No new import.
- The provenance loop keeps `implementation_revision` and `continue`s only when the predicate is true.
- `evaluate_mf1a_qa001`, `build_cases`, and `capture_provenance` are unchanged. Tolerances and the `>` residual compare are unchanged. Schemas stay the three v1 strings.

**Execution checklist (TDD: red → green → refactor)**

- [ ] Confirm ET-B1 is red for the seven tests
- [ ] Implement only the constant, the two helpers, and the loop skip
- [ ] Re-run the ET-B1 commands; 14 green; `-k mf1a` green

**Evidence produced**

- Same pytest commands as ET-B1.

**Risks / rollback**

- Risk: dropping the prior-side SHA check skips a bad prior revision.
- Rollback / mitigation: prior-side cases in `rejects_non_revision_value` fail that omission. Revert this file only.

## ET-B3 — Replace regeneration-acceptance.md (DS-B3)

**Implements delivery story**

- DS-B3. Traces: REQ-004.

**Change type**

- docs

**Definition of done**

- Replace `.cursor/skills/test-density-matrix/references/regeneration-acceptance.md` with the lock §6 text verbatim. Do not edit `SKILL.md`.

```markdown
# Regeneration acceptance (revision-only mismatch)

Read when a slice's ADRs or mini-spec require accepting a regeneration run at a commit
after the one that produced the committed evidence bundle. Milestone-specific thresholds
and field lists live in that milestone's ADRs; this file describes the usual shape.

## Contents

- Expected exit status
- Acceptance checklist
- Restore after acceptance

## Expected exit status

The exit status depends on the bundle's comparator. When the milestone ADR allowlists
`provenance.implementation_revision` (for M-F1a, ADR-F1A-009 Amendment 1), a difference
in only that field passes: exit 0, `status` pass, `regeneration.pass` true, and
`first_mismatch` null. Otherwise, expect the non-zero shape that milestone ADR documents.
Never widen an allowlist to make a run pass. Run once with no env overrides.

## Acceptance checklist

Accept only if all of the following hold (adjust metric names to the milestone ADR):

- Metrics are within the frozen comparators.
- Route and label fields are identical.
- Provenance and clean-start flags are as the ADR requires.
- `implementation_revision` equals HEAD.
- The other suites named in the ADR remain `pass`.
- The unfiltered recursive diff shows only the allowlisted field and the derived fields
  the ADR names. For example, `regeneration.prior_present` changes from false to true
  when the committed bundle was written with no prior. Any other difference fails.
- Before the run, the loaded extension sha256 equals the committed identity.

## Restore after acceptance

After acceptance, restore every rewritten file to the HEAD bytes, as in the snapshot and
restore section of `SKILL.md`. Regeneration outputs are never committed.
```

- q4 field lists and sha pins stay in this slice. They do not move into the reference.

**Execution checklist (TDD: red → green → refactor)**

- [ ] Replace the reference file with that contract
- [ ] Doc review against `TASK_3_MINI_SPEC.md` §3.6
- [ ] Confirm `SKILL.md` is unmodified

**Evidence produced**

- Doc review of `references/regeneration-acceptance.md`. Sign-off: three Developer paths only.

**Risks / rollback**

- Risk: putting q4 text back into `SKILL.md` reverses P0b.
- Rollback / mitigation: revert the reference file. Leave `SKILL.md` untouched.

## ET-B4 — Proof runs (c) and (g) (DS-B4)

**Implements delivery story**

- DS-B4. Traces: REQ-004, REQ-006, QA-008. Not a Developer code edit.

**Change type**

- tooling

**Definition of done**

- (c) from clean C1 and (g) from clean C2. Porcelain empty before each run. Extension sha256 `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` or stop.
- Command: `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
- Acceptance table in `TASK_3_MINI_SPEC.md` §3.8: exit 0; q4 `status` pass; `regeneration.pass` true; `first_mismatch` null; `summary.first_failure` null; case diff exactly `cases[0].provenance.implementation_revision` equal to `git rev-parse HEAD`; full-file diff that path plus `regeneration.prior_present` false → true; no third path; schemas and `extension_identities` equal the committed bundle; `non_counted_context` is `{}`.
- Copy regenerated JSON to `/tmp/<run>/` before restore. Restore with `git show HEAD:<q4> > <q4>`. After restore, q4 sha256 is `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` and porcelain is empty. Never stage the bundle.
- At (g) the new revision equals the C2 SHA.
- Pre-(d): `git diff --stat` shows no runtime or oracle path change beyond the allowlist skip. CLOSEOUT is written at (d) only. Between C1 and that write, `--strict` has exactly one finding, `SLICE_MISSING_CLOSEOUT` for task-3, as an error. No placeholder. No waiver.
- Failure routing: extension or environment mismatch → Tech Lead; any other mismatch, including a within-tolerance third path → Research Manager. A failed (c) stops before (d). A failed (g) does not amend C2.

**Execution checklist (TDD: red → green → refactor)**

- [ ] Developer pytest gate is green before (a) and on the C1 tree
- [ ] Tester runs (c), then (g) after C2, and records the §3.8 table
- [ ] Restore q4; do not commit it

**Evidence produced**

- Pipeline command above, plus `sha256sum` of the extension and of the restored q4 path.

**Risks / rollback**

- Risk: a rebuild changes the extension hash and regeneration fails at `extension_identities`.
- Rollback / mitigation: stop and send that mismatch to Tech Lead. Do not add the path to the allowlist.
