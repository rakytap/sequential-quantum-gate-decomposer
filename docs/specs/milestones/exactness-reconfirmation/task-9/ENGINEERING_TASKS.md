# Engineering tasks — M-F1a counted manifest (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 after the Research Manager freeze record · **Slice:** M-F1a task-9 ·
> **Parent:** `TASK_9_MINI_SPEC.md`, `DELIVERY_STORIES.md`, `COUNTED_MANIFEST.md` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, REQ-007 ·
> QA-001, QA-005, QA-008 · ADR-F1A-001, ADR-F1A-010 ·
> **Planning-base HEAD:** `a81be56baba7e97156532524978cc878d347fa87`
> **SDD stage:** step-4b-authorized
> **No push/PR**

Stage is `step-4b-authorized` so the value lands in C1 (ADR-F1A-008 Amendment 1 bound 4). With no `CLOSEOUT.md`, normal mode still warns `SLICE_MISSING_CLOSEOUT` for task-9, and `--strict` promotes that one finding to an error until (d). No placeholder. No waiver.

## ADR-F1A-010 planning statement

Items 1, 2, and 4 are unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged. Imported from the q4 module's runtime import path. Not re-implemented.
2. QA-001 per ADR-F1A-002 unchanged. Tolerances and `evaluate_mf1a_qa001` are imported from `mf1a_q4_baseline_validation.py`. The new bundle's allowlist is `cases[0]`…`cases[15]` `.provenance.implementation_revision` only. Sibling allowlist lengths stay 1, 4, 4, 4, and 3.
3. **Not unchanged. Disposed by Research Manager.** The counted set is the 16-row manifest frozen by the record in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` §10, which names the `COUNTED_MANIFEST.md` sha256. No counted run precedes that record.
4. Scope and the G-07 exit rule unchanged. Adding the counted sibling is not a G-07 change.

`claim_boundary` and `completeness_claim` are the Research Manager-accepted values in `COUNTED_MANIFEST.md` §§4–5. Developer copies them and does not choose them.

## Rules for every task

Developer edits are exactly:

- `benchmarks/density_matrix/correctness_evidence/mf1a_counted_validation.py` (new)
- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` (one import plus index 5)
- `tests/partitioning/evidence/test_correctness_evidence.py` (29 new tests plus two hunks)

Do not edit `workloads.py`, the five sibling modules, or `run_pipeline`'s write rule. Import the counted module inside each new test. A module-level import turns `-k mf1a` into one collection error. Do not add the pipeline import before the module exists.

C1 paths, after the RM record and the code-ready writer pass. No counted bundle. No CLOSEOUT.

1. `benchmarks/density_matrix/correctness_evidence/mf1a_counted_validation.py`
2. `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
3. `tests/partitioning/evidence/test_correctness_evidence.py`
4. `docs/specs/milestones/exactness-reconfirmation/task-9/ENGINEERING_TASKS.md`
5. `docs/specs/milestones/exactness-reconfirmation/task-9/TASK_9_MINI_SPEC.md`
6. `docs/specs/milestones/exactness-reconfirmation/task-9/DELIVERY_STORIES.md`
7. `docs/specs/milestones/exactness-reconfirmation/task-9/COUNTED_MANIFEST.md`
8. `docs/specs/milestones/exactness-reconfirmation/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`

Before Reviewer (a), sweep milestone context headers so none still say Step 4b is blocked. At (a), diff `mf1a_q4_baseline_validation.py`, `mf1a_baseline_validation.py`, `mf1a_fused_validation.py`, `mf1a_strict_validation.py`, and `mf1a_hybrid_validation.py` against `a81be56`. The diff is empty. The sweep does not edit `COUNTED_MANIFEST.md`; its bytes are final before its sha is recorded, and a later edit voids the freeze.

## ET-C9-1 — Red-first tests (DS-C9-1, DS-C9-2)

**Implements delivery story**

- DS-C9-1 and DS-C9-2.

**Change type**

- tests

**Definition of done**

- Twenty-nine functions. Import the counted module inside each test. Fixtures are the 16 committed sibling records, served through spies on the sibling `build_cases` calls. Each spy stamps the `provenance` it receives, as the real builders do; the clean one uses `"a"*40`. Each fixture breaks one guard on one row. Below, `cases[i]` alone means `cases[i].provenance.implementation_revision`.
- Names, in order: `test_mf1a_counted_manifest_exact_set_sixteen`, `test_mf1a_counted_manifest_wrong_count`, `test_mf1a_counted_manifest_substituted_workload`, `test_mf1a_counted_manifest_reordered`, `test_mf1a_counted_manifest_seed_or_param_count`, `test_mf1a_counted_dirty_provenance_not_counted`, `test_mf1a_counted_clean_provenance_counts_sixteen`, `test_mf1a_counted_route_fail_baseline`, `test_mf1a_counted_route_fail_fused`, `test_mf1a_counted_route_fail_strict`, `test_mf1a_counted_route_fail_hybrid`, `test_mf1a_counted_union_equals_sibling_manifests`, `test_mf1a_counted_imports_q4_oracle_and_qa001`, `test_mf1a_counted_allowlist_lengths`, `test_mf1a_counted_does_not_import_workloads`, `test_mf1a_counted_registered_at_index_5`, `test_mf1a_counted_claim_fields_match_rm_record`, `test_mf1a_counted_route_fail_q4_baseline`, `test_mf1a_counted_real_sibling_records_pass`, `test_mf1a_counted_calls_five_builders_with_one_provenance`, `test_mf1a_counted_qa001_failure_fails_bundle`, `test_mf1a_counted_regeneration_revision_only_passes`, `test_mf1a_counted_regeneration_one_case_diverges`, `test_mf1a_counted_regeneration_non_allowlisted_path`, `test_mf1a_counted_regeneration_allowlist_read_at_call_time`, `test_mf1a_counted_regeneration_bundle_structure`, `test_mf1a_counted_regeneration_second_field`, `test_mf1a_counted_finding_bands_one_sided`, `test_mf1a_counted_rejects_non_revision_value`.
- **Unchanged.** Test 1: `manifest.cells` equals `COUNTED_MANIFEST.md` §2 rows 1–16 literally, six fields each, and the 16 real records pass `manifest_exact_set`. Test 2: 15 cases and 17 cases → `manifest_exact_set`, with `regeneration["first_mismatch"] == "manifest_exact_set"`. Test 3: row 7 `workload` `phase2_xxz_hea_q8_continuity` → `manifest_exact_set`. Test 4: rows 12 and 13 swapped → `manifest_exact_set`. Test 7: clean → 16 flags true, count 16, status pass. Test 14: the counted tuple is exactly the 16 paths, the sibling lengths are 1, 4, 4, 4, and 3, and the counted predicate `is not` any sibling predicate. Test 16: `_CASE_SLICE_REGISTRY[5].module` is the counted module with `mf1a_sibling is True`; `[0:5]` is unchanged.
- **Redefine.** Test 5: row 8 `seed_policy` det, and row 15 `parameters` cut to 137 → `manifest_exact_set`. Test 6: dirty, and clean with `provenance_pass` false → `provenance`, all flags false, count 0; row 16 alone false → `provenance`. Tests 8–11, each → `route_realization`: row 4 `partition_count` 10; row 8 `actual_fused_execution` false; row 12 `fused_regions[0].candidate_kind` `unitary_island`; row 16's first channel-native partition relabelled `phase3_unitary_island_fused`/`pure_unitary_partition`, counts recomputed. Test 12: the (route, anchor, workload, budget) multiset equals the five `build_manifest()["cells"]`, and `seed_policy` and `parameter_count` equal the committed records. Test 13 (`is` pins): counted `capture_provenance`, `REGENERATION_COMMAND`, both tolerances, `_QA001_REGENERATION_TOLERANCES`, and `_QA001_VALUE_KEYS` are q4's; each sibling's `evaluate_mf1a_qa001` is q4's, and its `execute_sequential_density_reference` is `noisy_runtime`'s.
- **Add.** 18 `route_fail_q4_baseline`: row 1 `fused_region_count` 1. 19 `real_sibling_records_pass`: pass, with `_ROW_GATE_MODULES` by identity. 20 `calls_five_builders_with_one_provenance`: one capture; each sibling once with `provenance is` that object; each case carries its spy's sentinel, in row order. 21 `qa001_failure_fails_bundle`: row 10. 22 `regeneration_revision_only_passes`: `"c"*40`/`"a"*40` → `{"prior_present": True, "pass": True, "first_mismatch": None}`. 23 `regeneration_one_case_diverges`: current `cases[9]`; prior `cases[14]`; current all `"c"*40` but `cases[3]` → `cases[0]`. 24 `regeneration_non_allowlisted_path`: prior `cases[3].claim_boundary`. 25 `regeneration_allowlist_read_at_call_time`: `()` → `cases[0]`; first 15 paths → `cases[15]`. 26 `regeneration_bundle_structure`: a 15-case prior; prior schema `"x"`. 27 `regeneration_second_field`: uniform move plus `cases[6].workload` → mismatch `cases[6].workload`, failure `manifest_exact_set`. 28 `finding_bands_one_sided`: classifier and findings `is` baseline's; real records → `[]` and `[]`; row 1 `lambda_min` `-5e-13` → one C.4-shaped q4 finding, markers `[]`, no `oracle_lambda_min`. 29 `rejects_non_revision_value`: all-current or all-prior `"g"*40`, `"c"*39`, `""`, `"C"*40` → `cases[0]`; prior `cases[15]` key missing → `cases[15]`.
- **DoD partials (killing test).** 4-field set (5); unordered (4); count-only (3–5, 27); baseline dispatched by route either way (19); paths-only gate (8–11, 18); first row per group (8–11); hybrid skipped (11); `cases[0]`-only QA or provenance (21, 6); literal 16 or a `clean_start`-only flag (6); majority or no uniformity (23); tuple ignored (25); hex unchecked (29); prior length unchecked (26); symmetric or absent `lambda_min` band (28).
- Reviewer runs each at (a). A kill is pytest exit 1 with more than 0 collected; exit 4 or 5, or 0 collected, is not a kill. A survivor is NOT-READY. The red log opens with the test-file sha256, which must equal C1's. Pins: 29 and 137. Red with the module absent: 31 failed, 106 passed.
- Test 15 (`test_mf1a_counted_does_not_import_workloads`) is an AST check, because the siblings import `workloads`.
- Test 17 compares `CLAIM_BOUNDARY` and `COMPLETENESS_CLAIM` to the Research Manager-accepted text in `COUNTED_MANIFEST.md` §§4–5. It does not invent the string.
- Exactly two existing hunks: `sibling_dirs` adds `mf1a/counted` (`test_correctness_evidence.py` set at `:4107`), and `test_mf1a_historical_registered_siblings_still_written` asserts the counted artifact path. A collection error is the wrong red.

**Evidence produced**

- The two pytest commands in the mini-spec §9.

**Risks / rollback**

- Risk: a module-level import makes the red a collection error.
- Rollback: revert the test file.

## ET-C9-2 — Counted module and registry (DS-C9-2, DS-C9-3)

**Implements delivery story**

- DS-C9-2 and DS-C9-3.

**Change type**

- code

**Definition of done**

- The module calls the five `build_cases(provenance=shared)` functions, reorders to rows 1–16, and restamps as the mini-spec §3. It does not execute the cells again.
- Registry entry at index 5 with `mf1a_sibling=True`. `g07_exit_passes` is not edited.
- The source does not import `benchmarks.density_matrix.planner_surface.workloads`.
- `workloads.py` is unchanged against `1cb3d20c`.

**Evidence produced**

- `git diff --exit-code 1cb3d20c -- benchmarks/density_matrix/planner_surface/workloads.py`
- The pytest commands, green.

**Risks / rollback**

- Risk: reading sibling JSON after `run_pipeline` has written it sees a dirty tree. Build from the in-memory `build_cases` results instead. `run_pipeline` builds every suite before it writes.
- Rollback: revert the new module and the pipeline import.

## ET-C9-3 — Tester gate and commits (DS-C9-3)

**Implements delivery story**

- DS-C9-3. Runs only after the RM freeze record and C1.

**Change type**

- docs

**Definition of done**

- (c) follows the mini-spec §7, including the extension pin, the REQ-007 diff before restore, sibling regeneration, and the four-value table for 16 rows.
- Pre-(d) G-10 note states the four independence facts in the mini-spec §6. Baseline agreement is bitwise. Fused, hybrid, and strict agreement is not bitwise.
- (d) CLOSEOUT uses "`<route>` route verified at q`<n>`" per cell.
- (d) CLOSEOUT carries both RM-required disclosures from mini-spec §6 verbatim: the baseline shared-kernel limitation (rows 1–4, q4 per `task-1/CLOSEOUT.md:31-50`) and the strict product-of-pair-states limitation (rows 9–12, q4 included).
- Before (c), Tester records the `sha256sum` of `docs/specs/milestones/exactness-reconfirmation/task-9/COUNTED_MANIFEST.md` and of `git show <C1-sha>:<that path>`. Both must equal the RM freeze-record sha. A mismatch stops before (c) and goes to Research Manager.
- C2 stages the counted bundle, the CLOSEOUT, and the checklist touch. It does not stage the five provisional bundles. The checklist touch does not name the C2 sha.
- (g) restores all six siblings from the C2 sha and commits nothing.
- A counted failure goes to Research Manager with the manifest unchanged.

**Evidence produced**

- The pipeline command. Exit 0 on the passing run. Wall time is recorded and is not a gate. Planning band 105–140 s.

**Risks / rollback**

- Risk: restoring the counted bundle before C2 drops the only copy of (c).
- Rollback: the mini-spec §10. Do not revert the five provisional bundles.
