# Engineering tasks — M-F1a slice C.1 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.1 ·
> **Parent:** `TASK_5_MINI_SPEC.md`, `DELIVERY_STORIES.md` ·
> **Traces:** REQ-001, REQ-002, REQ-004, REQ-006 · QA-001, QA-008 ·
> ADR-F1A-001, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **Planning-base HEAD:** `91680ec720bc311f28e09021895f0c78900d773b`
> **SDD stage:** step-4b-authorized
> **No push/PR** · baseline route verified for q4 only

Stage is `step-4b-authorized` so the value lands in C1 (ADR-F1A-008 Amendment 1 bound 4).
With no `CLOSEOUT.md`, normal mode still warns `SLICE_MISSING_CLOSEOUT` for task-5, and
`--strict` promotes that one finding to an error until (d). No placeholder. No waiver.

## ADR-F1A-010 planning statement

This slice states each item unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged.
2. QA-001 per ADR-F1A-002 and the regeneration comparators unchanged, apart from the single allowlisted field `provenance.implementation_revision` of ADR-F1A-009 Amendment 1, applied once per case record (`cases[0]`…`cases[3]`) in the fused bundle; `Q4_REGENERATION_ALLOWLIST` stays length 1.
3. The counted denominator (ADR-F1A-001) unchanged. This slice records four provisional fused cells (`milestone_counted` false, `completeness_claim` false, `summary.milestone_counted_cases` 0). It does not freeze the milestone denominator and adds no other cell.
4. The scope and the G-07 exit rule unchanged; adding this sibling as a required suite is not a G-07 change (ADR-F1A-010 item 4).

## Rules for every task

Developer edits are exactly:

- `benchmarks/density_matrix/correctness_evidence/mf1a_fused_validation.py` (new)
- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` (registry index 1 and two-phase `run_pipeline` only)
- `tests/partitioning/evidence/test_correctness_evidence.py` (new tests plus the two hunks in ET-C1-1)

Do not edit `workloads.py` or the q4 module body. Import from it. `test_QX2` is not in this lane.

## ET-C1-1 — Red-first tests (DS-C1-1, DS-C1-2)

**Implements delivery story**

- DS-C1-1 and DS-C1-2. Traces: REQ-001, REQ-002, REQ-004, REQ-006.

**Change type**

- tests

**Definition of done**

- Twelve new functions whose names contain `mf1a_fused`. No parametrize. Import the new module inside the tests or a fixture so a missing module is twelve failures, not an import error that hides the existing `mf1a` tests.
- `--collect-only -k mf1a_fused` collects 12. `--collect-only -k mf1a` collects 41 (29 + 12).
- Two hunks in existing tests, and nothing else there: `sibling_dirs == {"mf1a/q4_baseline", "mf1a/fused"}` with the nonsibling assertion unchanged; `test_mf1a_historical_registered_siblings_still_written` also asserts the path under `fake_root / "mf1a" / "fused"`.

**Tests**

| Test | Assert |
|------|--------|
| `test_mf1a_fused_manifest_is_the_four_frozen_ids` | four ids in anchor order; fused route; `max_partition_qubits` is 2 |
| `test_mf1a_fused_q8_q10_builder_calls_match_frozen_ids` | frozen builder-call tuple and built `workload_id`; "not edited" is the REQ-007 diff, not this test |
| `test_mf1a_fused_realization_requires_a_fused_region` | `fused_region_count >= 1` and `"actually_fused"` in classifications; zero fusion fails |
| `test_mf1a_fused_qa001_tolerances_match_q4` | imported symbols are the q4 objects, not copied numbers |
| `test_mf1a_fused_allowlist_is_length_four_and_q4_stays_length_one` | four revision paths; q4 tuple length 1; predicate is not the q4 function |
| `test_mf1a_fused_revision_only_passes` | four differing full SHAs, equal within the current side and within the prior side; pass |
| `test_mf1a_fused_second_field_fails` | revision plus one earlier field: that earlier path |
| `test_mf1a_fused_finding_bands_follow_layer1` | table below |
| `test_mf1a_fused_rejects_non_revision_value` | `"g"*40`, `"c"*39`, `""`, `"C"*40`, missing key, on either side |
| `test_mf1a_fused_rejects_case_count` | fifth case or missing case fails `manifest_exact_set` or `bundle_structure` |
| `test_mf1a_fused_rejects_substituted_or_reordered_id` | substituted or reordered workload id fails `manifest_exact_set` |
| `test_mf1a_fused_pipeline_builds_every_sibling_before_writing_any` | patch `_CASE_SLICE_REGISTRY` with two fake siblings and log `_write_slice_bundle`; every build precedes the first sibling write; red on today's loop |

| Input | Expected |
|---|---|
| `5e-14` | expected range |
| `5e-12` | outside-expected marker |
| `5e-11` | finding |
| `2e-10` | QA-001 fail |
| `lambda_min = -5e-14` | no finding |
| `lambda_min = -5e-13` | finding |
| `lambda_min = -2e-12` | QA-001 fail |

**Evidence produced**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a_fused"
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a"
```

**Risks / rollback**

- Risk: a cell reports `fused_region_count` 0.
- Rollback / mitigation: fail the case. Do not change the id. A non-counted classification probe (Architect C.1 review) already saw 4, 6, 12, and 20 `actually_fused` islands at q4, q6, q8, and q10 at budget 2. A later zero is a stop under ADR-F1A-010, not a workload edit.

## ET-C1-2 — Sibling contract and two-phase pipeline (DS-C1-1, DS-C1-2)

**Implements delivery story**

- DS-C1-1 and DS-C1-2. Traces: REQ-002, REQ-004.

**Change type**

- code

**Definition of done**

- `run_pipeline`: build every registered suite in registry order, then write `mf1a_sibling is True` bundles, then return results in registry order. Signatures stay `build_cases()`, `build_artifact_bundle(cases)`, and `build_artifact_bundle()`. No path exclusion.
- Fused `_CaseSuiteEntry` at index 1, immediately after q4, `mf1a_sibling=True`.
- `SUITE_NAME = BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_fused_bundle_v1"`. `RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_fused_case_v1"`. `MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_fused_manifest_v1"`. `ARTIFACT_FILENAME = "mf1a_fused_bundle.json"`. `DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "fused"`.
- Claim fields and `seed_policy` as mini-spec §3.1 and §3.3.
- Realization dict: q4 keys `requested_path`, `realized_path`, `partition_count`, `exact_output_present`, `actual_fused_execution`, `fused_region_count`, `fused_region_classifications`, plus `fused_regions` rows (`partition_index`, `candidate_kind`, `classification`, `reason`, `operation_names`, `global_target_qbits`). `actual_fused_execution` is `fused_region_count > 0`.
- Import `evaluate_mf1a_qa001`, `MF1A_QA001_MATRIX_TOL`, `MF1A_QA001_LAMBDA_MIN_FLOOR`, `_QA001_REGENERATION_TOLERANCES`, and `_QA001_VALUE_KEYS` from the q4 module. Do not edit that module.
- Reuse q4 `capture_provenance` and `REGENERATION_COMMAND`. One capture shared by four cases. `input_artifact_identities` is `[]`.
- Finding helper is pure. Summary emission is derived and not compared. No finding key inside `cases[i]`.
- Fused allowlist predicate is new code. Length 4. The four current revisions are equal to each other, and the four prior revisions are equal to each other.
- `_G07_EXCLUDED_SUITES` and `g07_exit_passes` unchanged.

**Execution checklist**

- [ ] Confirm the ordering test is red on today's loop
- [ ] Implement the two-phase loop, the module, and the registry line
- [ ] `-k mf1a` green; collect pins 12 and 41

**Evidence produced**

- The commands in ET-C1-1.

**Risks / rollback**

- Risk: provenance is captured after a sibling write, so `clean_start` is false.
- Rollback / mitigation: the ordering test fails that shape. Revert `run_pipeline` only if the q4 signatures change.

## ET-C1-3 — Proof runs and C2 bundle (DS-C1-3)

**Implements delivery story**

- DS-C1-3. Traces: REQ-001, REQ-004, QA-001.

**Change type**

- tooling

**Definition of done**

- (c) and (g) from empty porcelain:
  `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
- Expect about 60–70 s for the whole (c) run. The 1–25 s per cell band remains the conservative cell bound. New tmux session only.
- Before each run, loaded `.so` sha256 is `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`.
- Commit rule is mini-spec §3.5: C2 is the fused bundle from (c) plus CLOSEOUT (and checklist touch-ups if any). After (c), `git show <C1-sha>:<q4> > <q4>`. After (g), restore both siblings from `<C2-sha>`. Record sha256 before and after. Eight-path diff before any restore.
- CLOSEOUT tabulates four QA-001 values per row, findings per §3.2, and per-cell island composition (CNOT-in-kernel fusion only at q4 and q6).

**Execution checklist**

- [ ] Developer `-k mf1a` gate green before (a)
- [ ] Tester runs (c), then (g) after C2
- [ ] Do not stage q4 or historical JSON

**Evidence produced**

- `/tmp/<run>/` copies and the sha256 pair for each restore.

**Risks / rollback**

- Risk: (g) has no committed fused prior, so regeneration does not check the case.
- Rollback / mitigation: C2 includes the (c) fused bytes (ADR-F1A-009 (f)). Do not omit them.
