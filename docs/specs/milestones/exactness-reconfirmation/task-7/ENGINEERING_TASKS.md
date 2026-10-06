# Engineering tasks — M-F1a slice C.3 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.3 ·
> **Parent:** `TASK_7_MINI_SPEC.md`, `DELIVERY_STORIES.md` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-005, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **Planning-base HEAD:** `42922382aaf1bdf5d609d169110b75cd72c9a757`
> **SDD stage:** step-4b-authorized
> **No push/PR** · baseline route verified for q4 only

Stage is `step-4b-authorized` so the value lands in C1 (ADR-F1A-008 Amendment 1 bound 4).
With no `CLOSEOUT.md`, normal mode still warns `SLICE_MISSING_CLOSEOUT` for task-7, and
`--strict` promotes that one finding to an error until (d). No placeholder. No waiver.

## ADR-F1A-010 planning statement

This slice states each item unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged.
2. QA-001 per ADR-F1A-002 and the regeneration comparators unchanged, apart from the single allowlisted field `provenance.implementation_revision` of ADR-F1A-009 Amendment 1, applied once per case record (`cases[0]`…`cases[3]`) in the strict bundle; `Q4_REGENERATION_ALLOWLIST` stays length 1 and the fused and hybrid allowlists stay length 4.
3. The counted denominator (ADR-F1A-001) unchanged. This slice records four provisional strict cells (`milestone_counted` false, `completeness_claim` false, `summary.milestone_counted_cases` 0). It does not freeze the milestone denominator and adds no other cell.
4. The scope and the G-07 exit rule unchanged; adding this sibling as a required suite is not a G-07 change (ADR-F1A-010 item 4).

## Rules for every task

Developer edits are exactly:

- `benchmarks/density_matrix/correctness_evidence/mf1a_strict_validation.py` (new; family builder, witness-only record and gate, Layer 1 §10 helper, and the strict allowlist predicate)
- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` (the import `mf1a_strict_validation as mf1a_strict` plus one registry entry at index 3 only)
- `tests/partitioning/evidence/test_correctness_evidence.py` (25 new `mf1a_strict` functions plus exactly two existing-test hunks)

Do not edit `workloads.py` or redesign `run_pipeline`. Import QA-001, `capture_provenance`, and `REGENERATION_COMMAND` from the q4 module. The runtime comes from `squander.partitioning.noisy_runtime`. Import the strict module inside each test or a fixture. A module-level import turns `-k mf1a` into one collection error. Adding the pipeline import before the module exists turns the whole file into one collection error; do not do that.

## ET-C3-1 — Red-first tests and mutant matrix (DS-C3-1, DS-C3-2)

**Implements delivery story**

- DS-C3-1 and DS-C3-2. Traces: REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, QA-005.

**Change type**

- tests

**Definition of done**

- Twenty-five functions. Collect-only gives `-k mf1a_strict` 25 and `-k mf1a` 84 (59 at HEAD `42922382` plus 25). No parametrize.
- Synthetic fixtures are built from the frozen q4 record printed in the Architect review Appendix B, Part 1. They are not built from hybrid's `_q4_hybrid_positive_realization` (`test_correctness_evidence.py:1273-1328`), whose motif reason `eligible_channel_native_motif` is not what the runtime emits.
- Each gate fixture breaks exactly one guard. Fixtures F1–F11 are those of Appendix B, Part 2, plus F8d of Appendix E. "One fixture per guard, no merged guard" means that no fixture breaks two guards. A function may hold several isolated fixtures, as hybrid's realization test does.
- Exactly two existing-test hunks. Line 2296 becomes `sibling_dirs == {"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid", "mf1a/strict"}`, and line 2297 is unchanged. `test_mf1a_historical_registered_siblings_still_written` (`:2179-2195`) also asserts `fake_root / "mf1a" / "strict" / ARTIFACT_FILENAME`.
- Keep the C.1 comparator rule (mini-spec §3.3). Do not use a majority reference.
- Unless a row says otherwise, every regeneration negative asserts `status == "fail"`, `regeneration["pass"] is False`, `summary["first_failure"] == "regeneration"`, and the exact `first_mismatch`. Rows 13, 14, and 25 assert `summary["first_failure"] == "manifest_exact_set"`.
- **Red first.** Write the tests while `validation_pipeline.py` is still at HEAD. `-k mf1a` then collects 84 and reports 27 failed (the 25 new tests on the missing module, plus the two hunked tests) and 57 passed. The module, the import, and the index-3 entry then make all 84 pass.

**Tests.** Each name is one function. Gate rows name the killing guard.

| # | Test | Fixture → expected |
|---:|---|---|
| 1 | `test_mf1a_strict_manifest_is_the_four_frozen_ids` | The schema id. Anchors `[4, 6, 8, 10]`. Workloads `phase31_local_support_q4_spectator_embedding_smoke`, `mf1a_strict_spectator_embed_q6`, `…_q8`, `…_q10`. Every cell has the literal `"phase31_channel_native"` and budget 2. `strict.ROUTE == "phase31_channel_native"`. |
| 2 | `test_mf1a_strict_builder_calls_match_frozen_ids` | A spy shows that the q4 cell calls `workloads.build_phase31_microcase_descriptor_set` once, with the smoke name and `max_partition_qubits=2`; its `source_type` is `microcase_builder`. `mf1a_strict_spectator_operation_specs(4)` equals the smoke's `operation_specs` from `workloads.phase31_microcase_definitions()`. For n = 6, 8, 10: the spec list equals the B3 template written out in the test, and the cell descriptor has id `mf1a_strict_spectator_embed_q{n}`, `source_type` `structured_family_builder`, budget 2, `parameter_count` 9·n/2, n/2 partitions, partition k global `(2k, 2k+1)`, and members `U3, U3, CNOT, amplitude_damping, phase_damping, U3`. Odd n and n < 2 raise `ValueError`. `strict.build_initial_parameters is partitioned_runtime.common.build_initial_parameters`. `_seed_policy_for_cell` returns `deterministic_workload_no_random_seed` for all four cells. |
| 3 | `test_mf1a_strict_realization_positive_is_the_q4_smoke_shape` | **Real.** The strict entry runs on the q4 cell and the q6 cell. `_build_realization` equals the frozen literal: q4 is Appendix B, Part 1; q6 has the same shape with 3 rows and witnesses at `[0, 1]`, `[2, 3]`, `[4, 5]`, `channel_native_motif_kraus_count_4`. `_route_realization_pass` is true for both. |
| 4 | `test_mf1a_strict_dropped_row_is_md` | F8a (row 1 removed) → false, S8 only. |
| 5 | `test_mf1a_strict_duplicate_index_is_me` | F8b (row 1 index set to 0) → false, S8 only. Kills a length-only S8. |
| 6 | `test_mf1a_strict_extra_row_kills_length_and_range_removal` | F8c (extra row `{"partition_index": 1}`) and F8d (row 1 index set to 2, out of range) → each false, S8 only. Kills S8 deletion (the C.2 `S_len_range`), and set-size, set-equality, and length-plus-set-size S8. |
| 7 | `test_mf1a_strict_motif_removed_fails` | F10a (region 1 removed), F10b (duplicate of region 0), and F10c (copy of region 0 at index 99) → each false, S10 only. |
| 8 | `test_mf1a_strict_vocabulary_fails` | F7 (`partition_runtime_class` added to row 0), F9a (region 1 `supported_but_unfused`), F9b (region 1 `unitary_island`), and F9c (region 1 reason `eligible_channel_native_motif`) → each false; one guard each, S7 or S9. |
| 9 | `test_mf1a_strict_pure_unitary_partition_raises` | **Real.** A q4 descriptor is built from `mf1a_strict_spectator_operation_specs(2)` plus pair (2, 3) without noise: U3(2), U3(3), CNOT with control 2 and target 3, U3(2); budget 2. The strict entry raises `NoisyRuntimeValidationError` with the three attributes in mini-spec §3.2. With `_build_cell_descriptor` monkeypatched to return that descriptor, `build_cases(provenance=<clean synthetic>)` raises the same error. |
| 10 | `test_mf1a_strict_case_consistency_guards_fail` | F1 route `phase31_channel_native_hybrid`; F2 budget 3; F3 requested `phase31_channel_native_hybrid`; F4 realized `partitioned_density_descriptor_baseline`; F5 `exact_output_present` false; F6 `partition_count` 0 with empty rows and regions and count 0; F11 `channel_native_partition_count` 3 → each false, one guard each. |
| 11 | `test_mf1a_strict_qa001_tolerances_match_q4` | The five `is` identities, as in hybrid `:1614-1620`. |
| 12 | `test_mf1a_strict_allowlist_is_length_four_and_siblings_stay` | Exact tuple equality. Strict 4, q4 1, fused 4, hybrid 4. The strict predicate `is not` each of the other three. |
| 13 | `test_mf1a_strict_rejects_case_count` | A fifth case, and three cases → each `status == "fail"`, `summary["first_failure"] == "manifest_exact_set"`, `regeneration["pass"] is False`, and `first_mismatch == "manifest_exact_set"`. |
| 14 | `test_mf1a_strict_rejects_substituted_or_reordered_id` | A substituted `cases[2].workload`, and swapped cases 2 and 3 → the same four assertions as row 13. |
| 15 | `test_mf1a_strict_pipeline_builds_every_sibling_before_writing_any` | Four fakes, every build before the first write. `_CASE_SLICE_REGISTRY[0:4]` modules are q4, fused, hybrid, and strict, each with `mf1a_sibling is True`. |
| 16 | `test_mf1a_strict_one_current_revision_diverges` | (i) RC-5: prior all `"a"*40`; current all `"a"*40` except `cases[2]` is `"d"*40` → `cases[2].provenance.implementation_revision`. (ii) all moved: current `["c"*40, "d"*40, "c"*40, "c"*40]` → `cases[0].provenance.implementation_revision` |
| 17 | `test_mf1a_strict_one_prior_revision_diverges` | (i) RC-6: current all `"a"*40`; prior all `"a"*40` except `cases[1]` is `"d"*40` → `cases[1].provenance.implementation_revision`. (ii) all moved: prior `["a"*40, "a"*40, "b"*40, "a"*40]` and current all `"c"*40` → `cases[0].provenance.implementation_revision` |
| 18 | `test_mf1a_strict_one_case_non_allowlisted_path` | uniform `"c"*40` / `"a"*40`; only `cases[1].seed_policy` changes → `cases[1].seed_policy` |
| 19 | `test_mf1a_strict_swapped_two_cases_revisions_only` | current all `"c"*40`; prior `["a"*40, "b"*40, "a"*40, "a"*40]`, then swap `cases[0]` and `cases[1]` to `["b"*40, "a"*40, "a"*40, "a"*40]`. Before and after → `cases[0].provenance.implementation_revision` |
| 20 | `test_mf1a_strict_empty_allowlist_fails` | monkeypatch `STRICT_REGENERATION_ALLOWLIST = ()`; uniform `"c"*40` / `"a"*40` → `cases[0].provenance.implementation_revision` |
| 21 | `test_mf1a_strict_length_three_allowlist_fails` | monkeypatch to the first three paths → `cases[3].provenance.implementation_revision` |
| 22 | `test_mf1a_strict_revision_only_passes` | uniform `"c"*40` / `"a"*40` passes with `first_mismatch is None`. Also asserts `regeneration == {"prior_present": True, "pass": True, "first_mismatch": None}`. RC-7: for `i` in (0, 1, 2, 3), set only `cases[i].provenance.provenance_pass = False` (prior `None`) → `status == "fail"` and `summary["first_failure"] == "provenance"` |
| 23 | `test_mf1a_strict_rejects_non_revision_value` | For each of `"g"*40`, `"c"*39`, `""`, and `"C"*40`: current `cases[0]` only → `cases[0]…`; prior `cases[1]` only → `cases[1]…`; uniform on the current side → `cases[0]…`; uniform on the prior side → `cases[0]…`. Missing key: current `cases[2]` → `cases[2]…`; prior `cases[2]` → `cases[2]…`; missing on all four, each side → `cases[0]…`. Each path is `….provenance.implementation_revision`. Template: `test_correctness_evidence.py:1838-1900`. |
| 24 | `test_mf1a_strict_finding_bands_follow_layer1` | The classifier rows: matrix `5e-14` expected; `5e-12` marker; `5e-11` finding; `2e-10` QA fail. `lambda_min`: `-5e-14` none; `-5e-13` finding; `-2e-12` QA fail; `3e-6` none and not a marker. Bundle level: `cases[0].qa001.lambda_min = -5e-13` gives `status == "pass"`, `summary["first_failure"] is None`, and `summary["findings"] == [{"route": "phase31_channel_native", "anchor_qbits": 4, "workload": "phase31_local_support_q4_spectator_embedding_smoke", "measure": "lambda_min", "value": -5e-13, "cause_hypothesis": "lambda_min below -1e-13 Layer 1 finding band"}]`, with no `oracle_lambda_min` anywhere in `json.dumps(bundle)`. `cases[1].qa001.lambda_min = 3e-6` gives `findings == []` and `outside_expected_markers == []`. |
| 25 | `test_mf1a_strict_second_field_fails` | All revisions moved uniformly and `cases[1].workload` substituted → `status == "fail"`, `regeneration["pass"] is False`, `first_mismatch == "cases[1].workload"`, and `summary["first_failure"] == "manifest_exact_set"`. |

Frozen q4 realization (Architect review Appendix B, Part 1), the literal row 3 matches:

```text
requested_path: phase31_channel_native
realized_path: phase31_channel_native
partition_count: 2
exact_output_present: True
channel_native_partition_count: 2
partitions: [{'partition_index': 0}, {'partition_index': 1}]
fused_regions: [{'partition_index': 0, 'candidate_kind': 'channel_native_motif', 'classification': 'actually_fused', 'reason': 'channel_native_motif_kraus_count_4', 'operation_names': ['U3', 'U3', 'CNOT', 'amplitude_damping', 'phase_damping', 'U3'], 'global_target_qbits': [0, 1]}, {'partition_index': 1, 'candidate_kind': 'channel_native_motif', 'classification': 'actually_fused', 'reason': 'channel_native_motif_kraus_count_4', 'operation_names': ['U3', 'U3', 'CNOT', 'amplitude_damping', 'phase_damping', 'U3'], 'global_target_qbits': [2, 3]}]
```

**Evidence produced**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a_strict"
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a"
```

**Risks / rollback**

- Risk: a later planner change makes a pair partition pure unitary.
- Rollback / mitigation: the strict entry raises `channel_native_noise_presence` and `build_cases` does not catch it (mini-spec §3.2). Do not edit `workloads.py`, swap the id, or drop the anchor. Non-counted citation: the Architect probe at `42922382` (q4 smoke 2/2 at 0.002 s; q6 3/3 at 0.006 s; q8 4/4 at 0.089 s; q10 5/5 at 2.61–2.62 s; Kraus count 4).

## ET-C3-2 — Strict module and registry insert (DS-C3-1, DS-C3-2)

**Implements delivery story**

- DS-C3-1 and DS-C3-2. Traces: REQ-002, REQ-003, REQ-004, QA-005.

**Change type**

- code

**Definition of done**

- The module implements §3.1–§3.3, including the family builder, schema ids, `claim_boundary`, and `STRICT_REGENERATION_ALLOWLIST`.
- Registry change is the import plus one `_CaseSuiteEntry` at index 3 only.
- The gate is S1–S11; deleting any guard is killed by the functions in the §9 table of the Architect review; `_build_realization` serializes partition rows as `{"partition_index"}` only; a hybrid-copied gate fails `…realization_positive_is_the_q4_smoke_shape` on real data.
- Findings follow the one-sided `lambda_min` rule and stay out of `cases[i]`.

| Guard (one check each) | Deleting it lets through | Killing function (B2 numbering) |
|---|---|---|
| S1 `route == "phase31_channel_native"` | F1 | 10 |
| S2 `planner_setting.max_partition_qubits == 2` | F2 | 10 |
| S3 `requested_path == ROUTE` | F3 | 10 |
| S4 `realized_path == ROUTE` | F4 | 10 |
| S5 `exact_output_present is True` | F5 | 10 |
| S6 `partition_count` is an int ≥ 1 | F6 | 10 |
| S7 every partition row's key set is exactly `{"partition_index"}` | F7 | 8 |
| S8 `sorted(row indices) == list(range(partition_count))` | F8a, F8b, F8c, F8d | 4, 5, 6 |
| S9 every region is `channel_native_motif`/`actually_fused` with reason `channel_native_motif_kraus_count_<k>`, k ≥ 1 | F9a, F9b, F9c | 8 |
| S10 `sorted(region partition indices) == list(range(partition_count))` | F10a, F10b, F10c | 7 |
| S11 `channel_native_partition_count == partition_count` | F11 | 10 |

Partial S8 implementations are killed too:

- a length-only check misses F8b and F8d;
- a set-size check misses F8c and F8d;
- length plus set-size, the C.2 shape, misses F8d;
- set equality misses F8c.

Functions 5 and 6 kill all four. F8c is the C.2 `S_len_range` survivor. Copying hybrid's label gate or `_serialize_partitions` fails function 3 on real data.

**Execution checklist**

- [ ] In-test import. While the pipeline is at HEAD, `-k mf1a` collects 84 and reports 27 failed and 57 passed
- [ ] Implement the module and the index-3 entry. Do not import the pipeline module of strict before the module exists
- [ ] `-k mf1a` green; collect pins 25 and 84

**Evidence produced**

- The commands in ET-C3-1.

**Risks / rollback**

- Risk: the strict predicate is the hybrid function with a renamed tuple.
- Rollback / mitigation: the own-function assertion fails. The hybrid module is not edited.

## ET-C3-3 — Proof runs and C2 bundle (DS-C3-3)

**Implements delivery story**

- DS-C3-3. Traces: REQ-001, REQ-004, QA-001.

**Change type**

- tooling

**Definition of done**

- (c) and (g) from empty porcelain use the single validation pipeline command.
- Tester time is scheduling guidance, not an acceptance criterion. The 1–25 s per-cell and 4–100 s four-cell figures (task-5 §6) are conservative rocky bounds. No gate, test, record field, or exit code depends on wall time (Layer 1 §2 puts timing out of scope), and the case record has no `runtime_ms` or `peak_rss_kb`. Measured strict-entry wall, non-counted, at `42922382`, frozen family: q4 0.002 s, q6 0.006 s, q8 0.089 s, q10 2.61–2.62 s. C.2's (c) and (g) took 78 s and 72 s for three siblings; expect about 80–90 s for C.3's. If a cell or the run passes a band, Tester records the measured time in the CLOSEOUT and continues. That is not a fail, a stop, or a Research Manager trigger. Do not add a timeout or a wall-time assertion. Run (c) and (g) in a new tmux session.
- Before each run, loaded `.so` sha256 is `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`.
- After (c), restore q4, fused, and hybrid from `<C1-sha>` to the sha256 values in mini-spec §3.4. Expected diffs are q4 2 paths and fused and hybrid 5 paths each (the revision paths plus `regeneration.prior_present`). After (g), restore all four siblings from the C2 sha. Eight-path diff before any restore.
- CLOSEOUT tabulates four QA-001 values, channel-native counts (2/2, 3/3, 4/4, 5/5), Kraus count 4 per witness, and findings. It does not copy the q4 bitwise paragraph. Required product-state limitation:

> The q6, q8, and q10 workloads are an M-F1a-only family built in `mf1a_strict_validation.py`, not by a Phase-3 or Phase-3.1 builder; no historical builder is strict-eligible at those widths at `max_partition_qubits` 2 (task-4 inventory). Each pair (2k, 2k+1) carries the q4 smoke's block and no operation couples two pairs, so every state in these runs is a product of pair states. Against the sequential oracle, the cells test strict preflight, Kraus composition of the six-member block, the local-to-global remap at every pair offset, and full-space embedding next to spectators that hold either earlier mixed states or |0⟩. They do not test channel application on inputs correlated across a partition boundary, the order of partitions (disjoint channels commute), local depolarizing under the strict entry, or single-wire and odd-aligned motifs. C.2 hybrid exercises correlated inputs and depolarizing on the same channel-native kernel. Parameters come from `build_initial_parameters`; no seed is drawn.

**Execution checklist**

- [ ] Developer `-k mf1a` gate green before (a)
- [ ] Tester runs (c), then (g) after C2, each in a new tmux session
- [ ] Do not stage q4, fused, hybrid, or historical JSON

**Evidence produced**

- `/tmp/<run>/` copies and sha256 before and after each restore.

**Risks / rollback**

- Risk: C2 omits the strict bundle, so (g) has no prior.
- Rollback / mitigation: C2 includes the (c) strict bytes (ADR-F1A-009 (f)).
