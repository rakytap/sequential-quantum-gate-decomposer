# Engineering tasks — M-F1a slice C.4 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.4 ·
> **Parent:** `TASK_8_MINI_SPEC.md`, `DELIVERY_STORIES.md` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-005, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **Planning-base HEAD:** `aaf6fcfe8e1a317485bad4d67580d1ad8082d11d`
> **SDD stage:** step-4b-authorized
> **No push/PR** · baseline route verified for q4 only

Stage is `step-4b-authorized` so the value lands in C1 (ADR-F1A-008 Amendment 1 bound 4).
With no `CLOSEOUT.md`, normal mode still warns `SLICE_MISSING_CLOSEOUT` for task-8, and
`--strict` promotes that one finding to an error until (d). No placeholder. No waiver.

## ADR-F1A-010 planning statement

This slice states each item unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged.
2. QA-001 per ADR-F1A-002 and the regeneration comparators unchanged, apart from the single allowlisted field `provenance.implementation_revision` of ADR-F1A-009 Amendment 1, applied once per case record (`cases[0]`…`cases[2]`) in the new baseline bundle; `Q4_REGENERATION_ALLOWLIST` stays length 1 and the fused, hybrid, and strict allowlists stay length 4.
3. The counted denominator (ADR-F1A-001) unchanged. This slice records three provisional baseline cells (`milestone_counted` false, `completeness_claim` false, `summary.milestone_counted_cases` 0). It does not freeze the milestone denominator and adds no other cell.
4. The scope and the G-07 exit rule unchanged; adding this sibling as a required suite is not a G-07 change (ADR-F1A-010 item 4).

## Rules for every task

Developer edits are exactly:

- `benchmarks/density_matrix/correctness_evidence/mf1a_baseline_validation.py` (new)
- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` (the import `mf1a_baseline_validation as mf1a_baseline` plus one registry entry at index 4 only)
- `tests/partitioning/evidence/test_correctness_evidence.py` (24 new `mf1a_baseline` functions plus exactly two existing-test hunks)

Do not edit `workloads.py`, `planner_surface/common.py`, or the q4 module, and do not redesign `run_pipeline`. Import QA-001, `capture_provenance`, and `REGENERATION_COMMAND` from the q4 module. Import the baseline module inside each test. A module-level import turns `-k mf1a` into one collection error. Adding the pipeline import before the module exists turns the whole file into one collection error; do not do that.

## ET-C4-1 — Red-first tests and mutant matrix (DS-C4-1, DS-C4-2)

**Implements delivery story**

- DS-C4-1 and DS-C4-2. Traces: REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, QA-005.

**Change type**

- tests

**Definition of done**

- Twenty-four functions. Collect-only gives `-k mf1a_baseline` 24 and `-k mf1a` 108 (84 at HEAD `aaf6fcfe` plus 24). No parametrize.
- Each gate fixture breaks exactly one guard. A function may hold several isolated fixtures. No fixture breaks two guards.
- Exactly two existing-test hunks. Line 3286 becomes `sibling_dirs == {"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid", "mf1a/strict", "mf1a/baseline"}`, and line 3287 is unchanged. `test_mf1a_historical_registered_siblings_still_written` (`:3165`) also asserts `fake_root / "mf1a" / "baseline" / ARTIFACT_FILENAME`.
- Keep the C.1 comparator rule (mini-spec §3.3) on three cases. Do not use a majority reference.
- Unless a row says otherwise, every regeneration negative asserts `status == "fail"`, `regeneration["pass"] is False`, `summary["first_failure"] == "regeneration"`, and the exact `first_mismatch`. Rows 12, 13, and 24 assert `summary["first_failure"] == "manifest_exact_set"`.
- **Red first.** Write the tests while `validation_pipeline.py` is still at HEAD. `-k mf1a` then collects 108 and reports 26 failed (the 24 new tests on the missing module, plus the two hunked tests) and 82 passed. The module, the import, and the index-4 entry then make all 108 pass.
- `evaluate_mf1a_qa001` is pinned by identity with the q4 function, not by copied numbers. Row 10 lists every identity.
- Each synthetic case carries its own anchor's frozen realization: q6 is 7 partitions and 8 classifications; q8 is 9 partitions and 10 classifications; q10 is 11 partitions and 12 classifications. In each, index 1 is `deferred_or_unsupported_candidate`, the other entries are `supported_but_unfused`, and the key set is the q4 one. A single shared realization sends every regeneration row to `route_realization`.
- Non-revision rows list literal paths in this table. The test asserts those literals. It does not read `first_mismatch` back from the comparator to build the expected path. Row 22's prior and current are all `"a"*40`, and current is a copy of prior except the mutated field.

**Tests.** Each name is one function.

| # | Test | Fixture → expected |
|---:|---|---|
| 1 | `test_mf1a_baseline_manifest_is_the_three_frozen_ids` | Schema id. Anchors `[6, 8, 10]`. Workloads `phase2_xxz_hea_q6_continuity`, `phase2_xxz_hea_q8_continuity`, `phase2_xxz_hea_q10_continuity`. Every cell has the literal `"partitioned_density_descriptor_baseline"` and budget 2. `baseline.ROUTE == "partitioned_density_descriptor_baseline"`. The q4 manifest still has one cell. |
| 2 | `test_mf1a_baseline_builder_calls_match_frozen_ids` | A spy shows each cell calls `build_phase2_continuity_vqe(n)` and `build_phase3_continuity_partition_descriptor_set(..., max_partition_qubits=2)` once. Workload ids are the three frozen ids. `parameter_count` is 30, 42, and 54. `build_initial_parameters is` the partitioned-runtime function. `_seed_policy_for_cell` returns `deterministic_workload_no_random_seed` for all three. `baseline.EXPECTED_PARTITION_COUNTS == {6: 7, 8: 9, 10: 11}`. `len(descriptor_set.partitions)` is 7, 9, and 11 on the real builder at all three anchors (descriptor build only). The module does not import `benchmarks.density_matrix.planner_surface.workloads`. |
| 3 | `test_mf1a_baseline_realization_positive_matches_probed_shape` | **Real.** The baseline entry runs on q6 and q8. Both paths are `partitioned_density_descriptor_baseline`, `exact_output_present` is true, `actual_fused_execution` is false, `fused_region_count` is 0, and `partition_count` is 7 and 9. Classifications are 8 and 10. Index 1 is `deferred_or_unsupported_candidate`. Every other entry is `supported_but_unfused`. `actually_fused` is absent. The realization key set is exactly the q4 key set. q10 is the same shape with `partition_count` 11 and 12 classifications. With `baseline.execute_sequential_density_reference` monkeypatched to return the maximally mixed state of the cell's width, `build_cases(provenance=<clean synthetic>)` gives `qa001_pass` False on every case. |
| 4 | `test_mf1a_baseline_fused_count_fails` | F7: `fused_region_count` 1, classifications unchanged, `actual_fused_execution` false → false, G7 only. |
| 5 | `test_mf1a_baseline_actually_fused_class_fails` | F8a: same-length classifications with `actually_fused` at the last index, `fused_region_count` 0. F8b: `actually_fused` only at index 0, count 0. Each false, G8 only. F8a kills a length-only check and a first-entry-only check. A gate that checks only G7 accepts F8a. |
| 6 | `test_mf1a_baseline_actual_fused_flag_fails` | Each fixture is built from the real q6 record. F6: `actual_fused_execution` true, count 0, `actually_fused` absent → false, G6 only. F6b: `actual_fused_execution` 0 → false, G6 only. |
| 7 | `test_mf1a_baseline_partition_count_fails` | Each fixture is built from the real q6 record. F9: `partition_count` 6, other guards hold → false, G9 only. F9b: `partition_count` 9 (the q8 value) → false, G9 only. F9c: `partition_count` 6, and the last `supported_but_unfused` entry removed (7 classifications) → false, G9 only. A `partition_count > 1` check, an anchor-agnostic `in {7, 9, 11}` check, and `partition_count == len(classifications) - 1` each accept one of these. |
| 8 | `test_mf1a_baseline_case_consistency_guards_fail` | Each fixture is built from the real q6 record. F1 route `partitioned_density_descriptor_fused_unitary_islands`; F2 budget 3; F3 requested `partitioned_density_descriptor_fused_unitary_islands`; F4 realized_path `partitioned_density_descriptor_fused_unitary_islands`; F5 `exact_output_present` false; F5b `exact_output_present` 1 → each false, one guard each. |
| 9 | `test_mf1a_baseline_synthesized_label_fails` | Each fixture is built from the real q6 record. F10: `partition_runtime_class` added on the realization → false, G10 only. F10b: extra realization key `runtime_ms` → false, G10 only. |
| 10 | `test_mf1a_baseline_qa001_tolerances_match_q4` | These are the q4 objects, each `is` the q4 module's object: `evaluate_mf1a_qa001`, `MF1A_QA001_MATRIX_TOL`, `MF1A_QA001_LAMBDA_MIN_FLOOR`, `_QA001_REGENERATION_TOLERANCES`, `_QA001_VALUE_KEYS`, `capture_provenance`, and `REGENERATION_COMMAND`. These are the `squander.partitioning.noisy_runtime` objects: `execute_partitioned_density` and `execute_sequential_density_reference`. |
| 11 | `test_mf1a_baseline_allowlist_is_length_three_and_siblings_stay` | Exact tuple equality. Baseline 3, q4 1, fused 4, hybrid 4, strict 4. The baseline predicate `is not` each of the other four. |
| 12 | `test_mf1a_baseline_rejects_case_count` | A fourth case, and two cases → each `status == "fail"`, `summary["first_failure"] == "manifest_exact_set"`, `regeneration["pass"] is False`, and `first_mismatch == "manifest_exact_set"`. |
| 13 | `test_mf1a_baseline_rejects_substituted_or_reordered_id` | A substituted `cases[1].workload`, and swapped cases 1 and 2 → the same four assertions as row 12. |
| 14 | `test_mf1a_baseline_pipeline_builds_every_sibling_before_writing_any` | Five fakes, every build before the first write. `_CASE_SLICE_REGISTRY[0:5]` modules are q4, fused, hybrid, strict, and baseline, each with `mf1a_sibling is True`. |
| 15 | `test_mf1a_baseline_one_current_revision_diverges` | (i) prior all `"a"*40`; current all `"a"*40` except `cases[2]` is `"d"*40` → `cases[2].provenance.implementation_revision`. (ii) all moved: current `["c"*40, "d"*40, "c"*40]` → `cases[0].provenance.implementation_revision`. |
| 16 | `test_mf1a_baseline_one_prior_revision_diverges` | (i) current all `"a"*40`; prior all `"a"*40` except `cases[1]` is `"d"*40` → `cases[1].provenance.implementation_revision`. (ii) prior `["a"*40, "a"*40, "b"*40]` and current all `"c"*40` → `cases[0].provenance.implementation_revision`. |
| 17 | `test_mf1a_baseline_one_case_non_allowlisted_path` | uniform `"c"*40` / `"a"*40`; only `cases[1].seed_policy` changes → `cases[1].seed_policy`. |
| 18 | `test_mf1a_baseline_swapped_two_cases_revisions_only` | current all `"c"*40`; prior `["a"*40, "b"*40, "a"*40]`, then swap `cases[0]` and `cases[1]`. Before and after → `cases[0].provenance.implementation_revision`. |
| 19 | `test_mf1a_baseline_empty_allowlist_fails` | monkeypatch `BASELINE_REGENERATION_ALLOWLIST = ()`; uniform `"c"*40` / `"a"*40` → `cases[0].provenance.implementation_revision`. |
| 20 | `test_mf1a_baseline_length_two_allowlist_fails` | monkeypatch to the first two paths → `cases[2].provenance.implementation_revision`. |
| 21 | `test_mf1a_baseline_revision_only_passes` | uniform `"c"*40` / `"a"*40` passes with `regeneration == {"prior_present": True, "pass": True, "first_mismatch": None}`. RC-7: for `i` in (0, 1, 2), set only `cases[i].provenance.provenance_pass = False` (prior `None`) → `status == "fail"` and `summary["first_failure"] == "provenance"`. |
| 22 | `test_mf1a_baseline_rejects_non_revision_value` | Prior and current are all `"a"*40`, and current is a copy of prior except the mutated field (template `test_correctness_evidence.py:2833-2895`). For each of `"g"*40`, `"c"*39`, `""`, and `"C"*40`: current `cases[0]` only → `cases[0].provenance.implementation_revision`; prior `cases[1]` only → `cases[1].provenance.implementation_revision`; uniform on the current side → `cases[0].provenance.implementation_revision`; uniform on the prior side → `cases[0].provenance.implementation_revision`. Missing key: current `cases[2]` → `cases[2].provenance.implementation_revision`; prior `cases[2]` → `cases[2].provenance.implementation_revision`; missing on all three, each side → `cases[0].provenance.implementation_revision`. |
| 23 | `test_mf1a_baseline_finding_bands_follow_layer1` | Matrix: `5e-14` expected; `5e-12` marker; `5e-11` finding; `2e-10` QA fail. `lambda_min` is one-sided: a finding is only below `-1e-13`. `-5e-14` none; `-5e-13` finding; `-2e-12` QA fail. `+5.377e-10` and `+2e-13` are neither a finding nor a marker. Bundle level: `cases[0].qa001.lambda_min = -5e-13` gives `status == "pass"`, `summary["first_failure"] is None`, `outside_expected_markers == []`, and `summary["findings"] == [{"route": "partitioned_density_descriptor_baseline", "anchor_qbits": 6, "workload": "phase2_xxz_hea_q6_continuity", "measure": "lambda_min", "value": -5e-13, "cause_hypothesis": "lambda_min below -1e-13 Layer 1 finding band"}]`, with no `oracle_lambda_min` anywhere in `json.dumps(bundle)`. `cases[1].qa001.lambda_min = 5.377e-10` and `cases[1].qa001.lambda_min = 2e-13` each give `findings == []` and `outside_expected_markers == []`. |
| 24 | `test_mf1a_baseline_second_field_fails` | All revisions moved uniformly and `cases[1].workload` substituted → `status == "fail"`, `regeneration["pass"] is False`, `first_mismatch == "cases[1].workload"`, and `summary["first_failure"] == "manifest_exact_set"`. |

**Evidence produced**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a_baseline"
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a"
```

**Risks / rollback**

- Risk: a later edit turns `allow_fusion` on, so the cell fuses.
- Rollback / mitigation: G6, G7, and G8 fail the case. Do not edit `workloads.py` or drop the anchor. Non-counted citation: the probe at `aaf6fcfe` (q6 7/0 fused, 0.037 s; q8 9/0, 0.080 s; q10 11/0, 0.475 s).

## ET-C4-2 — Baseline module and registry insert (DS-C4-1, DS-C4-2)

**Implements delivery story**

- DS-C4-1 and DS-C4-2. Traces: REQ-002, REQ-003, REQ-004, QA-005.

**Change type**

- code

**Definition of done**

- The module implements §3.1–§3.3, including the call-only builders, schema ids, the `claim_boundary`, and `BASELINE_REGENERATION_ALLOWLIST`.
- Registry change is the import plus one `_CaseSuiteEntry` at index 4 only.
- `_route_realization_pass` is exactly G1–G10, one check each, and nothing else. In particular it has no count-versus-tally check, no flag-versus-count check, no classification-length check, and no q4 `partition_count > 1` leftover. Deleting any guard is killed by the function in the table below. `_build_realization` serializes only the q4 baseline keys. A hybrid-copied label gate fails row 3 on real data.
- At (a), Reviewer applies the ten single-guard deletions and the listed partials to the real `_route_realization_pass`. Each must turn its named function red. A survivor is NOT-READY at (a).

| Guard (one check each) | Deleting it lets through | Killing function |
|---|---|---|
| G1 `route == "partitioned_density_descriptor_baseline"` | F1 | 8 |
| G2 `planner_setting.max_partition_qubits == 2` | F2 | 8 |
| G3 `requested_path == ROUTE` | F3 | 8 |
| G4 `realized_path == ROUTE` | F4 | 8 |
| G5 `exact_output_present is True` | F5, F5b | 8 |
| G6 `actual_fused_execution is False` | F6, F6b | 6 |
| G7 `fused_region_count == 0` | F7 | 4 |
| G8 `"actually_fused" not in fused_region_classifications` | F8a, F8b | 5 |
| G9 `partition_count` is an int, not a bool, and equal to `EXPECTED_PARTITION_COUNTS.get(case["anchor_qbits"])`; an unknown anchor returns False | F9, F9b, F9c | 7 |
| G10 realization key set equals the q4 key set | F10, F10b | 9 |

Partial implementations. Each row replaces one guard with the partial and keeps the other nine:

| Guard | Partial | Real positives | Killed by the Planner set | Killed by |
|---|---|---|---|---|
| G1 | prefix `partitioned_density_descriptor` | pass | no | F1b (fused id; replaces F1's value) |
| G4 | `realized_path in {baseline, fused}` | pass | no | F4b (fused id; replaces F4's value) |
| G5 | truthiness; `== True` | pass | no | F5b |
| G6 | `not flag`; `== False` | pass | no | F6b |
| G6 | derived `fused_region_count == 0` | pass | yes | F6 |
| G7 | derived from the classification tally | pass | yes | F7 |
| G7 | `not count` | pass | — | equivalent (correct) |
| G8 | length-only; index 0 only; last index only; skip index 0; count substitute | pass | yes | F8a and/or F8b |
| G9 | q4 style `> 1` | pass | yes | F9 |
| G9 | anchor-agnostic `in {7, 9, 11}` | pass | **no** | F9b |
| G9 | derived `== len(classifications) - 1` | pass | **no** | F9c |
| G9 | formula `== anchor + 1` | pass | — | equivalent; row 2's named-constant pin fixes the literal |
| G10 | superset of the q4 keys | pass | yes | F10 |
| G10 | label-key blacklist | pass | **no** | F10b |

**Execution checklist**

- [ ] In-test import. While the pipeline is at HEAD, `-k mf1a` collects 108 and reports 26 failed and 82 passed
- [ ] Implement the module and the index-4 entry. Do not import it from the pipeline before the module exists
- [ ] `-k mf1a` green; collect pins 24 and 108

**Evidence produced**

- The commands in ET-C4-1.

**Risks / rollback**

- Risk: the baseline predicate is the q4 function with a renamed tuple.
- Rollback / mitigation: the own-function assertion fails. The q4 module is not edited.

## ET-C4-3 — Proof runs and C2 bundle (DS-C4-3)

**Implements delivery story**

- DS-C4-3. Traces: REQ-001, REQ-004, QA-001.

**Change type**

- tooling

**Definition of done**

- (c) and (g) from empty porcelain use the single validation pipeline command.
- Tester time is scheduling guidance, not an acceptance criterion. The 1–25 s per-cell band is not a gate. No timeout and no wall-time assertion. Measured baseline-entry wall, non-counted, at `aaf6fcfe`: q6 0.037 s, q8 0.080 s, q10 0.475 s. C.3's (c) and (g) took 80 s and 77 s. Expect about 80–100 s for C.4's. If a cell or the run passes a band, Tester records the time and continues. Run (c) and (g) in a new tmux session.
- Before each run, loaded `.so` sha256 is `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`.
- After (c), restore q4, fused, hybrid, and strict from `<C1-sha>` to the sha256 values in mini-spec §3.4. Expected diffs are q4 2 paths and fused, hybrid, and strict 5 paths each. After (g), restore all five siblings from the C2 sha. The (g) expected diffs are q4 2 paths, fused, hybrid, and strict 5 paths each, and baseline 4 paths (the three revision paths plus `regeneration.prior_present`). Eight-path diff before any restore.
- CLOSEOUT tabulates the four QA-001 values for each of the three cells, partition counts (7, 9, 11), `fused_region_count` 0, and findings. `lambda_min` is one-sided: a finding is only below `-1e-13`, and a positive `lambda_min` is neither a finding nor a marker. It records the bitwise agreement and its reason in the task-1 pattern, for q6, q8, and q10. If a `lambda_min` finding occurs, the Tester reports the oracle's `lambda_min` from the same `eigenvalues()` call in the CLOSEOUT, outside the bundle (Layer 1 §10). Required limitation:

> The q6, q8, and q10 cells use the existing continuity builders at `max_partition_qubits` 2. The q4 bundle stays the historical one-case record and is not extended. Against the sequential oracle, the cells test that the baseline entry realizes `partitioned_density_descriptor_baseline`, does not set `actual_fused_execution`, and leaves `fused_region_count` at 0 while region classifications stay unfused. They do not test fusion execution, a channel-native motif, or a noise model other than the continuity builder's fixed local depolarizing, amplitude damping, and phase damping on wires 0 and 1. Parameters come from `build_initial_parameters`; no seed is drawn. The cells and the oracle share `_build_runtime_circuit` lowering and the `NoisyCircuit` kernels, so agreement is bitwise and a kernel-level bug would appear on both sides. This oracle cannot detect it.

**Execution checklist**

- [ ] Developer `-k mf1a` gate green before (a)
- [ ] Tester runs (c), then (g) after C2, each in a new tmux session
- [ ] Do not stage q4, fused, hybrid, strict, or historical JSON

**Evidence produced**

- `/tmp/<run>/` copies and sha256 before and after each restore.

**Risks / rollback**

- Risk: C2 omits the new baseline bundle, so (g) has no prior.
- Rollback / mitigation: C2 includes the (c) baseline bytes (ADR-F1A-009 (f)).
