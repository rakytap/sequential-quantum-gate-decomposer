# Engineering tasks — M-F1a slice C.2 (Layer 4)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.2 ·
> **Parent:** `TASK_6_MINI_SPEC.md`, `DELIVERY_STORIES.md` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **Planning-base HEAD:** `a006c7e27b9e9f2e400d4379a7890600270b5d54`
> **SDD stage:** step-4b-authorized
> **No push/PR** · baseline route verified for q4 only

Stage is `step-4b-authorized` so the value lands in C1 (ADR-F1A-008 Amendment 1 bound 4).
With no `CLOSEOUT.md`, normal mode still warns `SLICE_MISSING_CLOSEOUT` for task-6, and
`--strict` promotes that one finding to an error until (d). No placeholder. No waiver.

## ADR-F1A-010 planning statement

This slice states each item unchanged.

1. The oracle, `execute_sequential_density_reference`, unchanged.
2. QA-001 per ADR-F1A-002 and the regeneration comparators unchanged, apart from the single allowlisted field `provenance.implementation_revision` of ADR-F1A-009 Amendment 1, applied once per case record (`cases[0]`…`cases[3]`) in the hybrid bundle; `Q4_REGENERATION_ALLOWLIST` stays length 1 and the fused allowlist stays length 4.
3. The counted denominator (ADR-F1A-001) unchanged. This slice records four provisional hybrid cells (`milestone_counted` false, `completeness_claim` false, `summary.milestone_counted_cases` 0). It does not freeze the milestone denominator and adds no other cell.
4. The scope and the G-07 exit rule unchanged; adding this sibling as a required suite is not a G-07 change (ADR-F1A-010 item 4).

## Rules for every task

Developer edits are exactly:

- `benchmarks/density_matrix/correctness_evidence/mf1a_hybrid_validation.py` (new)
- `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` (the import plus one registry entry at index 2 only)
- `tests/partitioning/evidence/test_correctness_evidence.py` (18 new `mf1a_hybrid` functions plus exactly two existing-test hunks)

Do not edit `workloads.py` or redesign `run_pipeline`. Import QA-001 and `capture_provenance` from the q4 module. The hybrid predicate is new code with the C.1 rule in mini-spec §3.3.

## ET-C2-1 — Red-first tests and mutant matrix (DS-C2-1, DS-C2-2)

**Implements delivery story**

- DS-C2-1 and DS-C2-2. Traces: REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, QA-005.

**Change type**

- tests

**Definition of done**

- Import the hybrid module inside each test or a fixture. A module-level import would turn all `-k mf1a` tests into one collection error.
- Eighteen new functions whose names contain `mf1a_hybrid`. No parametrize. Fold sub-cases into those functions.
- `--collect-only -k mf1a_hybrid` collects 18. `--collect-only -k mf1a` collects 59 (41 at HEAD `a006c7e2` plus 18).
- Exactly two existing-test hunks, not new tests. Reviewer (a) checks exactly those two. One sets `sibling_dirs == {"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid"}` with the nonsibling assertion unchanged. `test_mf1a_historical_registered_siblings_still_written` also asserts `fake_root / "mf1a" / "hybrid" / ARTIFACT_FILENAME`.
- Keep the C.1 comparator rule (mini-spec §3.3). Expected `first_mismatch` is what that rule returns. Do not use a majority reference or a rule relative to `cases[0]`.
- C.1 kill-shapes are inside the functions below: RC-5 and RC-6 single-case current and prior moves with the other cases unmoved; RC-7 status-fail assertions; a `provenance_pass` loop over cases; single-case and uniform revision-shape negatives; bundle finding asserts.
- One-case mutants assert the named `first_mismatch` path. They must not be rewritten as "revisions are not all equal".
- Unless the row says otherwise, every regeneration negative asserts all four of `status == "fail"`, `regeneration["pass"] is False`, `summary["first_failure"] == "regeneration"`, and the exact `first_mismatch`.
- The build-before-write ordering half is already green at HEAD (`7cf11a49`). It is not a red-first signal. The live-registry assertion in that same function, plus the in-test import, makes all eighteen functions red before the module and the index-2 insert exist.

**Tests.** Eighteen functions.

| Test | Assert |
|------|--------|
| `test_mf1a_hybrid_manifest_is_the_four_frozen_ids` | ids in §3.1 order; route `phase31_channel_native_hybrid`; budget 2 |
| `test_mf1a_hybrid_q8_q10_builder_calls_match_frozen_ids` | the two `build_phase31_structured_descriptor_set` calls and the built ids |
| `test_mf1a_hybrid_realization_requires_a_channel_native_partition` | `_route_realization_pass` as §3.2. Each synthetic fixture fails: motif region removed from a channel-native partition; a motif region while the label is `phase3_unitary_island_fused`; an unknown class; reason `channel_native_noise_presence`; one partition row dropped; a duplicated partition index; zero `phase31_channel_native` partitions; `phase3_supported_unfused` alongside an `actually_fused` island; an orphan region with an out-of-range `partition_index`. A positive fixture with q4's distribution passes: 2 channel-native partitions with motif witnesses and 3 islands. No new function |
| `test_mf1a_hybrid_qa001_tolerances_match_q4` | imported q4 symbols, not copied numbers |
| `test_mf1a_hybrid_rejects_case_count` | fifth or missing case fails `manifest_exact_set` or `bundle_structure` |
| `test_mf1a_hybrid_rejects_substituted_or_reordered_id` | substituted or reordered id fails `manifest_exact_set` |

**Mutant matrix.** The assertion is the path, not a set-equality of revisions. Monkeypatch `HYBRID_REGENERATION_ALLOWLIST` for the empty and length-three rows.

| Test | Fixture → `first_mismatch` |
|------|----------------------------|
| `test_mf1a_hybrid_one_current_revision_diverges` | (i) RC-5: prior all `"a"*40`; current all `"a"*40` except `cases[2]` is `"d"*40` → `cases[2].provenance.implementation_revision`. (ii) all moved: prior all `"a"*40`; current `["c"*40, "d"*40, "c"*40, "c"*40]` → `cases[0].provenance.implementation_revision` (current side non-uniform; the first moved case fails) |
| `test_mf1a_hybrid_one_prior_revision_diverges` | (i) RC-6: current all `"a"*40`; prior all `"a"*40` except `cases[1]` is `"d"*40` → `cases[1].provenance.implementation_revision`. (ii) all moved: current all `"c"*40`; prior `["a"*40, "a"*40, "b"*40, "a"*40]` → `cases[0].provenance.implementation_revision` |
| `test_mf1a_hybrid_one_case_non_allowlisted_path` | uniform `"c"*40` / `"a"*40`; only current `cases[1].seed_policy` changes → `cases[1].seed_policy` |
| `test_mf1a_hybrid_swapped_two_cases_revisions_only` | current all `"c"*40`; prior `["a"*40, "b"*40, "a"*40, "a"*40]`, then swap the prior revision fields of `cases[0]` and `cases[1]` to give `["b"*40, "a"*40, "a"*40, "a"*40]`. Before and after the swap → `cases[0].provenance.implementation_revision` (prior non-uniform) |
| `test_mf1a_hybrid_empty_allowlist_fails` | monkeypatch `HYBRID_REGENERATION_ALLOWLIST = ()`; uniform `"c"*40` / `"a"*40` → `cases[0].provenance.implementation_revision` |
| `test_mf1a_hybrid_length_three_allowlist_fails` | monkeypatch to the first three paths; uniform `"c"*40` / `"a"*40` → `cases[3].provenance.implementation_revision` |
| `test_mf1a_hybrid_revision_only_passes` | uniform `"c"*40` / `"a"*40` → `status == "pass"` and `regeneration == {"prior_present": True, "pass": True, "first_mismatch": None}`. RC-7: for `i` in (1, 2, 3), set only `cases[i].provenance.provenance_pass = False` (prior `None`) → `status == "fail"` and `summary["first_failure"] == "provenance"` |
| `test_mf1a_hybrid_rejects_non_revision_value` | Same fixture shape as `test_correctness_evidence.py:1034-1124`: other cases stay unmoved. For each of `"g"*40`, `"c"*39`, `""`, and `"C"*40`: current `cases[0]` only → `cases[0].provenance.implementation_revision`; prior `cases[1]` only (current unchanged) → `cases[1].provenance.implementation_revision`; uniform on the current side → `cases[0].provenance.implementation_revision`; uniform on the prior side → `cases[0].provenance.implementation_revision`. Missing key: current `cases[2]` only → `cases[2].provenance.implementation_revision`; prior `cases[2]` only → `cases[2].provenance.implementation_revision`; missing on all four, each side → `cases[0].provenance.implementation_revision` |
| `test_mf1a_hybrid_finding_bands_follow_layer1` | Classifier rows: `5e-14` expected; `5e-12` marker; `5e-11` finding; `2e-10` QA fail; `lambda_min` `-5e-14` none; `-5e-13` finding; `-2e-12` QA fail. Bundle: `cases[0].qa001.lambda_min = -5e-13` → `status == "pass"`, `summary["first_failure"] is None`, and `summary["findings"]` is exactly one record equal to `route` `phase31_channel_native_hybrid`, `anchor_qbits` 4, `workload` `phase2_xxz_hea_q4_continuity`, `measure` `lambda_min`, `value` `-5e-13`, `cause_hypothesis` `lambda_min below -1e-13 Layer 1 finding band`. No key named `oracle_lambda_min` appears anywhere in the bundle |
| `test_mf1a_hybrid_second_field_fails` | all revisions moved uniformly; `cases[1].workload` substituted → `cases[1].workload` |
| `test_mf1a_hybrid_allowlist_is_length_four_and_siblings_stay` | exact tuple equality and length 4; `Q4_REGENERATION_ALLOWLIST` length 1; `FUSED_REGENERATION_ALLOWLIST` length 4; the hybrid predicate `is not` the q4 predicate and `is not` the fused predicate |
| `test_mf1a_hybrid_pipeline_builds_every_sibling_before_writing_any` | three fake siblings; every build precedes the first sibling write. Also the live `_CASE_SLICE_REGISTRY[0:3]` modules are q4, fused, and hybrid, each with `mf1a_sibling is True`. The registry assertion is red until the index-2 insert |

**Evidence produced**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a_hybrid"
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" --collect-only -q -k "mf1a"
```

**Risks / rollback**

- Risk: q8 or q10 has zero witnessed channel-native partitions at budget 2.
- Rollback / mitigation: fail the case. Do not drop the anchor or edit `workloads.py`. A non-counted re-probe at `a006c7e2` found channel-native partitions q4 2/5, q6 2/7, q8 11/34, and q10 18/56, each with one `channel_native_motif` witness. A zero at (c) returns to Research Manager.

## ET-C2-2 — Hybrid module and registry insert (DS-C2-1, DS-C2-2)

**Implements delivery story**

- DS-C2-1 and DS-C2-2. Traces: REQ-002, REQ-003, REQ-004, QA-005.

**Change type**

- code

**Definition of done**

- New module implements mini-spec §3.1–§3.3, including the schema ids, `claim_boundary`, `realization` keys, `HYBRID_REGENERATION_ALLOWLIST`, and `FROZEN_STRUCTURED_BUILDER_CALLS`.
- Registry change is the import plus one entry at index 2 only: `_CaseSuiteEntry` for `mf1a_hybrid` with `build_cases`, `build_artifact_bundle`, and `mf1a_sibling=True`.
- `_route_realization_pass` matches §3.2. Witness agreement, frozen vocabulary, and label count are required. Labels alone do not pass.
- Findings helper matches Layer 1 §10 and stays out of `cases[i]`. The bundle carries no `oracle_lambda_min` key.
- Provenance is captured once inside `build_cases()` before the first cell. The gate is `all(case["provenance"]["provenance_pass"] for case in cases)`.
- Hybrid predicate is not the q4 or fused function. It reads `HYBRID_REGENERATION_ALLOWLIST` at call time. Uniformity is required, and the mutant paths above still fail.

**Execution checklist**

- [ ] Import the hybrid module inside each test or a fixture
- [ ] Eighteen functions are red before the module exists: in-test import, and the live `_CASE_SLICE_REGISTRY[0:3]` assertion. The build-before-write half is already green at `7cf11a49`
- [ ] Implement the module and the index-2 entry
- [ ] `-k mf1a` green; collect pins 18 and 59

**Evidence produced**

- The commands in ET-C2-1.

**Risks / rollback**

- Risk: the hybrid predicate is the fused function with a renamed tuple.
- Rollback / mitigation: the own-function assertion fails. The fused module is not edited. Any diff on it fails Reviewer (a).

## ET-C2-3 — Proof runs and C2 bundle (DS-C2-3)

**Implements delivery story**

- DS-C2-3. Traces: REQ-001, REQ-004, QA-001.

**Change type**

- tooling

**Definition of done**

- (c) and (g) from empty porcelain use the single validation pipeline command.
- Expect about 70–80 s. C.1's (c) was 51 s and (g) 48 s. Measured hybrid-entry wall, non-counted: q4 0.002 s, q6 0.008 s, q8 0.82–0.86 s, q10 19.7–20.0 s. The 1–25 s band holds; q10 near the top is a Tech Lead note, not a fail.
- Before each run, loaded `.so` sha256 is `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`.
- C1 is the seven paths in mini-spec §3.4. C2 path list: hybrid bundle from (c), `task-6/CLOSEOUT.md`, and checklist touch-ups if any. After (c), restore q4 and fused with `git show <C1-sha>:<path> > <path>`. Expected diffs: q4 two paths (`cases[0].provenance.implementation_revision`, `regeneration.prior_present`); fused five paths (the four revisions and `regeneration.prior_present`). After (g), restore all three siblings from the C2 sha. Eight-path diff before any restore.
- CLOSEOUT tabulates four QA-001 values, findings, and per-cell channel-native counts. It pastes the §3.4 independence text. It does not copy the q4 bitwise paragraph or the C.1 partial-share paragraph.

**Execution checklist**

- [ ] Developer `-k mf1a` gate green before (a)
- [ ] Tester runs (c), then (g) after C2
- [ ] Do not stage q4, fused, or historical JSON

**Evidence produced**

- `/tmp/<run>/` copies and sha256 before and after each restore.

**Risks / rollback**

- Risk: C2 omits the hybrid bundle, so (g) has no prior.
- Rollback / mitigation: C2 includes the (c) hybrid bytes (ADR-F1A-009 (f)).
