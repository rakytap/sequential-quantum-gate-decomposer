# Task / Work Package 8: baseline route provisional evidence at 6, 8, and 10 (Slice C.4)
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.4 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Planning-base HEAD:**
> `aaf6fcfe8e1a317485bad4d67580d1ad8082d11d` · **Inventory:** `task-4/ROUTE_INVENTORY.md`
> · **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-005, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **No push/PR**

## 1. Purpose

Record provisional evidence for `partitioned_density_descriptor_baseline` at anchors 6, 8,
and 10 against `execute_sequential_density_reference`. q4 keeps the shipped one-case
bundle. `milestone_counted` stays false. q4 remains baseline route verified. C.1, C.2,
and C.3 stay closed. This slice does not change the oracle, QA-001 tolerances, the
denominator, or G-07 exclusions.

## 2. Scope

### 2.1 Developer paths (exactly three)

| Path | Edit |
|------|------|
| `benchmarks/density_matrix/correctness_evidence/mf1a_baseline_validation.py` | new sibling for anchors 6, 8, and 10. Holds the call-only continuity builder, the witness-only record and gate (§3.2), the Layer 1 §10 helper, the tuple `BASELINE_REGENERATION_ALLOWLIST`, plus a predicate that is its own function. QA-001, `capture_provenance`, and `REGENERATION_COMMAND` come from the q4 module. Builders are the ones the q4 module already calls. The runtime comes from `squander.partitioning.noisy_runtime`. The module does not import `benchmarks.density_matrix.planner_surface.workloads` |
| `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | one import `mf1a_baseline_validation as mf1a_baseline` beside the existing `mf1a_*` imports, plus one `_CaseSuiteEntry(mf1a_baseline, "build_cases", "build_artifact_bundle", mf1a_sibling=True)` at index 4, after strict. Do not redesign `run_pipeline` |
| `tests/partitioning/evidence/test_correctness_evidence.py` | 24 new `mf1a_baseline` functions plus their helpers, and exactly two existing-test hunks (ET-C4-1) |

Sibling directory `mf1a/baseline`, distinct from `mf1a/q4_baseline`. The sibling set becomes
`{"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid", "mf1a/strict", "mf1a/baseline"}`.
Reviewer (a) checks exactly those two hunks and the import-plus-index-4 limit.

### 2.2 q4 retention

The historical bundle
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json`
stays sha256 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` at this
HEAD and is not extended. C.4 does not edit `mf1a_q4_baseline_validation.py`, does not
append a case, and does not change `Q4_REGENERATION_ALLOWLIST` (length 1). Anchors 6, 8,
and 10 are a new sibling bundle, not rows added to the q4 file.

### 2.3 Out of scope

Carry-forward items stay out: the hybrid `S_len_range` extra-row gap, the task-4 CLOSEOUT
nit, and the C.3 partial-S10, `_are_ints`, and reviewer nits. Also out: oracle or
tolerance edits, dropping an anchor, editing `workloads.py`, and committing regenerated
q4, fused, hybrid, strict, or historical bundles. Not touched: `workloads.py`, the q4,
fused, hybrid, and strict module bodies, `planner_surface/common.py`,
`noisy_descriptor.py`, any other `squander/` path, C++, and any other test file.

## 3. Required behavior

### 3.1 Frozen cells

`max_partition_qubits=2` on every builder call (O-11). Entry is
`execute_partitioned_density` (`noisy_runtime_core.py:816`) with `allow_fusion=False`.
`runtime_path` stays `PHASE3_RUNTIME_PATH_BASELINE`
(`partitioned_density_descriptor_baseline`).

| Anchor | Id | Builder |
|---|---|---|
| 6 | `phase2_xxz_hea_q6_continuity` | `build_phase2_continuity_vqe(6)` then `build_phase3_continuity_partition_descriptor_set(vqe, max_partition_qubits=2)` |
| 8 | `phase2_xxz_hea_q8_continuity` | same at 8 |
| 10 | `phase2_xxz_hea_q10_continuity` | same at 10 |

These are the C.0 inventory rows (`task-4/ROUTE_INVENTORY.md` baseline 6, 8, and 10; flag
`none`). The ids are the workload ids those builders already write. The new module calls
them. It does not edit `workloads.py` (ADR-F1A-005, call only; this slice does not call
it). It does not edit `planner_surface/common.py`. Noise stays
`build_continuity_density_noise`: local depolarizing 0.1 on wire 0, amplitude damping
0.05 on wire 1, phase damping 0.07 on wire 0. C.4 does not choose a new noise model.

`seed_policy` is `deterministic_workload_no_random_seed` on all three cases. The
continuity builder draws no seed. Parameters are
`build_initial_parameters(descriptor_set.parameter_count)`, recorded as `parameters`.
The module imports no random-number generator.

> A non-counted probe at `aaf6fcfe` (baseline entry only; no oracle, QA-001, or bundle) found every cell realized on `partitioned_density_descriptor_baseline` with `actual_fused_execution` false, `fused_region_count` 0, and `actually_fused` absent: q6 7 partitions, 8 region records, 30 parameters, 0.037 s; q8 9 partitions, 10 region records, 42 parameters, 0.080 s; q10 11 partitions, 12 region records, 54 parameters, 0.475 s. Partition `partition_runtime_class` and `partition_route_reason` are `None` on every partition. Classification index 1 is `deferred_or_unsupported_candidate`; every other classification is `supported_but_unfused`.

### 3.2 Pass rule (ADR-F1A-003)

Baseline is not evaluation mode. The strict and hybrid label contracts are not copied.
The module writes no per-partition label. Realization keys are exactly the q4 baseline
keys:

- `requested_path` and `realized_path`;
- `partition_count`;
- `exact_output_present`;
- `actual_fused_execution`;
- `fused_region_count`;
- `fused_region_classifications` (the classification string of each `result.fused_regions` entry, and nothing else).

No `partition_runtime_class`, `partition_route_reason`, `runtime_class_counts`,
`route_reason_counts`, `runtime_ms`, or `peak_rss_kb` key appears. No partition-row list
is added. Region indices are not added. `_route_realization_pass` is exactly G1–G10,
one check each, and nothing else. In particular it has no count-versus-tally check,
no flag-versus-count check, no classification-length check, and no q4
`partition_count > 1` leftover. The guard table and the partial table are in ET-C4-2.

G9 is `partition_count` being an int, not a bool, and equal to
`EXPECTED_PARTITION_COUNTS.get(case["anchor_qbits"])`.
`EXPECTED_PARTITION_COUNTS = {6: 7, 8: 9, 10: 11}` is a named module constant, and an
unknown anchor returns False rather than raising.

This slice meets REQ-003 and QA-005 through no silent route substitution (G3, G4, G6–G8).

QA-001 is imported from the q4 module (`evaluate_mf1a_qa001 is` the q4 function). Layer 1
bands, as ruled for C.2 (e): the outside-expected marker applies only to matrix residuals
(Frobenius, max-abs, and absolute trace deviation). A matrix residual above `1e-11` is a
finding; above `1e-10` fails QA-001. `lambda_min` is one-sided: a finding is only below
`-1e-13`, QA-001 floor `-1e-12`, and no upper bound. A positive `lambda_min` is neither
a finding nor a marker. A finding does not change pass/fail, a tolerance, the oracle,
or the counted set. No finding field inside `cases[i]`. No bundle key named
`oracle_lambda_min`.

`claim_boundary` literal. Research Manager approved it via Tech Lead, 2026-10-06. It
names 6, 8, and 10 because this bundle does not contain the q4 cell:

```text
Provisional baseline-route slice evidence for partitioned_density_descriptor_baseline at anchors 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or frozen-matrix claim.
```

### 3.3 Bundle and allowlist

- `SUITE_NAME = BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_baseline_bundle_v1"`
- `RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_baseline_case_v1"`
- `MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_baseline_manifest_v1"`
- `ARTIFACT_FILENAME = "mf1a_baseline_bundle.json"`
- `DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "baseline"`
- `ROUTE = PHASE3_RUNTIME_PATH_BASELINE`
- `MAX_PARTITION_QUBITS = 2`
- `BASELINE_REGENERATION_ALLOWLIST` is the three `cases[i].provenance.implementation_revision` paths, read at call time.

`len(cases)` is 3. All cases `milestone_counted` false and `completeness_claim` false.
`summary.milestone_counted_cases` is 0. q4 `capture_provenance()` runs once inside
`build_cases()`, before the first cell, and is shared. The provenance gate is
`all(case["provenance"]["provenance_pass"] for case in cases)`.
`input_artifact_identities` is `[]`.

`Q4_REGENERATION_ALLOWLIST` stays length 1. Fused, hybrid, and strict allowlists stay
length 4. The baseline predicate is its own function. A revision difference at case `i`
is allowlisted only when the path is in the tuple, both values are full lowercase 40-hex
revisions, current differs from prior, and all three current revisions are equal and all
three prior revisions are equal. A single moved case, with the other cases unmoved,
fails at that case's path. When every case moved and either side is non-uniform,
`cases[0]` fails.

### 3.4 Close

ADR-F1A-009 (a)–(g). No `CLOSEOUT.md` in this pass. C1 is exactly seven paths: the three
Developer paths, the three `task-8/` specs, and this checklist. C1 holds no bundle, no
`CLOSEOUT.md`, no handback, and no q4, fused, hybrid, strict, or historical JSON. (c) is
one pipeline run. `run_pipeline` builds every case slice before the first write. After
(c), restore q4, fused, hybrid, and strict with `git show <C1-sha>:<path> > <path>` and
do not stage them. Committed sha256 values at this HEAD: q4
`483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`; fused
`3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0`; hybrid
`33aaa442f69e533e599e64895346786831a9591ae2a1db83dc80bf0643532ab9`; strict
`b5177cc81b6c6058039ba52310fd19f0311beb8d3247c2b9a13aeeead07edeb3`. Expected sibling
diffs are the allowlisted revision paths plus `regeneration.prior_present` (q4: 2 paths;
fused, hybrid, and strict: 5 paths each). C2 commits the new baseline bundle from (c)
plus CLOSEOUT and checklist touch-ups. (g) restores all five siblings from the C2 sha
and commits nothing. The eight-path historical diff runs before any restore. After (g),
the expected diffs are q4 2 paths, fused, hybrid, and strict 5 paths each, and baseline
4 paths (the three revision paths plus `regeneration.prior_present`). This re-close was at `step-4a` (strict warning, exit 0). After this writer pass the
stage is `step-4b-authorized`. Between C1 and (d), `--strict` reports exactly one
error, `SLICE_MISSING_CLOSEOUT` for task-8. After (d), both modes are clean. No
placeholder. No waiver.

> Baseline and the oracle are separate calls. Each allocates its own `DensityMatrix`, and neither reads the other's output back. Both lower the same canonical operations, in the same order, through `_build_runtime_circuit` and the same C++ `NoisyCircuit` gate and noise kernels. The baseline entry applies each member segment with `_execute_member_sequence` (`noisy_runtime_core.py:766-809`). The oracle applies the flattened circuit once (`:1011-1057`). A non-counted Architect probe at `aaf6fcfe` found the q6, q8, and q10 baseline density matrices equal to the oracle's bit for bit, with zero fused-kernel and zero Kraus calls on the baseline side. Bitwise agreement is expected. The CLOSEOUT records its reason, as task-1 does. Limitation: a kernel-level bug would appear on both sides, and this oracle cannot detect it.

## 4. Unsupported behavior

- Editing `workloads.py`, `planner_surface/common.py`, or the q4 module.
- Appending anchors 6, 8, or 10 to the q4 bundle, or changing its sha as a C.4 product.
- Writing per-partition labels or label counts into the baseline record.
- Treating `fused_region_count == 0` as sufficient when `actually_fused` is in the classification list.
- A `partition_count > 1` check in place of the frozen counts 7, 9, and 11.
- A noise model other than `build_continuity_density_noise`, or any cross-pair retune.
- Setting `milestone_counted` true, a second allowlisted field, or a finding field inside `cases[i]`.
- Changing `run_pipeline`, QA-001, the oracle, or G-07.
- A timeout or wall-time assertion.
- Starting a later slice.
- The hybrid `S_len_range` fix, the task-4 CLOSEOUT nit, and the C.3 carry-forward nits.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-001, REQ-003 | baseline tests | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a` | `-k mf1a_baseline` collects 24; `-k mf1a` collects 108 (84 + 24) | ET-C4-1 |
| REQ-004 | allowlist pin | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_baseline` | baseline length 3; q4 length 1; fused, hybrid, and strict length 4 | ET-C4-1 |
| REQ-002, QA-001 | proof (c) and (g) | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | three provisional cases pass; four existing siblings pass; historical diff empty | ET-C4-3 |
| REQ-001 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and `--strict` | warn at the recorded step-4a review; one strict error SLICE_MISSING_CLOSEOUT for task-8 between C1 and (d); both modes clean after (d) | this mini-spec |

## 6. Tester time

> Tester time is scheduling guidance, not an acceptance criterion. The 1–25 s per-cell band is a conservative rocky bound, not a gate. No gate, test, record field, or exit code depends on wall time, and the case record has no `runtime_ms` or `peak_rss_kb`. Measured baseline-entry wall, non-counted, at `aaf6fcfe`: q6 0.037 s, q8 0.080 s, q10 0.475 s. The oracle was not run. C.3's (c) and (g) took 80 s and 77 s for the four existing siblings. Expect about 80–100 s for C.4's (c) and (g). If a cell or the run passes a band, Tester records the measured time in the CLOSEOUT and continues. That is not a fail, a stop, or a Research Manager trigger. Do not add a timeout or a wall-time assertion. Run (c) and (g) in a new tmux session.

## 7. Rollback

Revert the three Developer paths and, if C2 landed, the baseline sibling bundle only.
Do not revert the q4 bundle.

## 8. Research Manager record

Research Manager approved, via Tech Lead, 2026-10-06, the `claim_boundary` in §3.2, the
C.0 continuity workloads, and the builder noise as-is. O-11 stays 2. The oracle, QA-001,
tolerances, and the counted set are unchanged. No new ADR. ADR-F1A-003 stays closed.
The open questions from the first planning pass are closed. A different workload, rate,
or channel, or a change to the `claim_boundary` wording, would return to Research Manager.
