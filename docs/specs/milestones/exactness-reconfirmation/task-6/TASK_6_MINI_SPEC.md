# Task / Work Package 6: hybrid route provisional evidence (Slice C.2)
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.2 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Planning-base HEAD:**
> `a006c7e27b9e9f2e400d4379a7890600270b5d54` · **Inventory:** `task-4/ROUTE_INVENTORY.md`
> · **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **No push/PR**

## 1. Purpose

Record provisional evidence for `phase31_channel_native_hybrid` at anchors 4, 6, 8,
and 10 against `execute_sequential_density_reference`. Workload ids are frozen.
`milestone_counted` stays false. q4 remains baseline route verified. C.1 stays
closed (fused bundle generated at clean C1; regenerated at clean C2). This slice
does not change the oracle, QA-001 tolerances, the denominator, or G-07 exclusions.

## 2. Scope

### 2.1 Developer paths (exactly three)

| Path | Edit |
|------|------|
| `benchmarks/density_matrix/correctness_evidence/mf1a_hybrid_validation.py` | new sibling; contract in §3.2–§3.3 and ET-C2-2 |
| `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | the import plus one registry entry at index 2, after fused, `mf1a_sibling=True`. Do not redesign the two-phase `run_pipeline` |
| `tests/partitioning/evidence/test_correctness_evidence.py` | 18 new `mf1a_hybrid` functions plus exactly two existing-test hunks in ET-C2-1. Reviewer (a) checks exactly those two |

Sibling directory `mf1a/hybrid`. The exact sibling set becomes
`{"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid"}`. Do not edit `workloads.py`,
the q4 or fused module bodies, or `SKILL.md`. q8 and q10 call `workloads.py` and
are ADR-F1A-005 protected (call only).

### 2.2 Out of scope

C.3 (RM-1(a) family), C.4, oracle or tolerance edits, dropping an anchor,
re-selecting a workload, and committing regenerated q4, fused, or historical bundles.

## 3. Required behavior

### 3.1 Frozen cells

`max_partition_qubits=2` is passed on every builder call. Entry is
`execute_partitioned_density_channel_native_hybrid` (`noisy_runtime_core.py:992`).
Builders come from `benchmarks/density_matrix/planner_surface/{common,workloads}.py`
and `squander/partitioning/noisy_planner.py`, not the `tests/partitioning/fixtures`
mirrors. Record `planner_setting.max_partition_qubits` from
`descriptor_set.max_partition_qubits`.

| Anchor | Workload id | Builder | Protection |
|--------|------------|---------|------------|
| 4 | `phase2_xxz_hea_q4_continuity` | `planner_surface/common.py::build_phase2_continuity_vqe(4)` plus `noisy_planner.py::build_phase3_continuity_partition_descriptor_set` | none |
| 6 | `phase2_xxz_hea_q6_continuity` | same builders at 6 | none |
| 8 | `phase31_pair_repeat_q8_dense_seed20260318` | `planner_surface/workloads.py::build_phase31_structured_descriptor_set("phase31_pair_repeat", qbit_num=8, noise_pattern="dense", seed=20260318, max_partition_qubits=2)` | ADR-F1A-005 |
| 10 | `phase31_pair_repeat_q10_dense_seed20260318` | same builder at `qbit_num=10` | ADR-F1A-005 |

The q8 and q10 ids are the format string in `workloads.py` (`{family}_q{n}_{noise}_seed{seed}`).
`seed_policy` is `deterministic_workload_no_random_seed` on q4 and q6, and
`structured_family_seed_20260318` on q8 and q10. Parameters come from
`benchmarks/density_matrix/partitioned_runtime/common.py::build_initial_parameters`.

A non-counted re-probe at `a006c7e2` (hybrid entry only; no oracle, no QA-001, no
bundle) found channel-native partitions q4 2/5, q6 2/7, q8 11/34, and q10 18/56.
Each witnessed partition has exactly one `channel_native_motif` execution witness,
and static classification equals the executed labels. Zero witnessed channel-native
partitions fails the case. The workload id is not swapped. That stop returns to
Research Manager (ADR-F1A-010). Anchors are not dropped.

### 3.2 Pass rule and findings (ADR-F1A-003)

`_route_realization_pass` is the attribution gate. `requested_path`, `realized_path`,
and `exact_output_present` are consistency checks only, not witnesses.
`fused_region_count` is not the witness. All of the following hold:

1. The cell matches the manifest; `route` is `phase31_channel_native_hybrid`; `planner_setting.max_partition_qubits == 2`.
2. `requested_path == realized_path == ROUTE` and `exact_output_present is True`.
3. `len(partitions) == partition_count`, and the partition indices are exactly `0 … partition_count-1`, each once.
4. Every class is in {`phase31_channel_native`, `phase3_unitary_island_fused`, `phase3_supported_unfused`}, and every reason is in {`eligible_channel_native_motif`, `pure_unitary_partition`, `channel_native_qubit_span`, `channel_native_support_surface`}.
5. For partition `p` with its regions `R_p`: `phase31_channel_native` ⇔ reason `eligible_channel_native_motif`, exactly one `channel_native_motif` / `actually_fused` region, and no `unitary_island` region. `phase3_unitary_island_fused` requires a non-eligible reason, at least one `unitary_island` / `actually_fused` region, and no motif region. `phase3_supported_unfused` requires a non-eligible reason, no `actually_fused` region of either kind, and at least one region classified `supported_but_unfused` or `deferred_or_unsupported_candidate`. No region carries a `partition_index` outside the range.
6. `channel_native_partition_count` equals the number of `phase31_channel_native` labels, and it is at least 1. `runtime_class_counts` and `route_reason_counts` equal the counts recomputed from `partitions`.

QA-001 is imported from the q4 module (call only). Findings follow Layer 1 §10, as
in C.1: matrix residual above `1e-11` is a finding; `lambda_min` below `-1e-13` is
a finding; values above `1e-13` are an outside-expected marker. A finding names
route, anchor, workload, measure, value, and a cause hypothesis. A `lambda_min`
finding also reports the oracle `lambda_min` outside the bundle. Matrix-residual
findings go to Research Manager at slice close. No finding changes pass/fail, a
tolerance, the oracle, or the counted set. No finding field inside `cases[i]`.
No bundle key is named `oracle_lambda_min`.

`claim_boundary` names the route `phase31_channel_native_hybrid`; anchors 4, 6, 8,
and 10 at `max_partition_qubits` 2; provisional slice evidence, not the frozen
milestone denominator; and no complete M-F1a, external-protocol, Aer, energy, or
frozen-matrix claim. The literal is:

```text
Provisional hybrid-route slice evidence for phase31_channel_native_hybrid at anchors 4, 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or frozen-matrix claim.
```

`realization` keys, exactly: `requested_path`, `realized_path`, `partition_count`
(from `len(descriptor_set.partitions)`), `exact_output_present`,
`channel_native_partition_count`, `runtime_class_counts` and `route_reason_counts`
(each `dict(sorted(Counter(...).items()))`), `partitions` rows (`partition_index`,
`partition_runtime_class`, `partition_route_reason`), and `fused_regions` rows
(`partition_index`, `candidate_kind`, `classification`, `reason`,
`operation_names`, `global_target_qbits`). There is no `fused_region_count` key
and no `actual_fused_execution` key.

### 3.3 Bundle

- `SUITE_NAME = BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_hybrid_bundle_v1"`
- `RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_hybrid_case_v1"`
- `MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_hybrid_manifest_v1"`
- `ARTIFACT_FILENAME = "mf1a_hybrid_bundle.json"`
- `DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "hybrid"`
- `ROUTE = PHASE31_RUNTIME_PATH_CHANNEL_NATIVE_HYBRID`
- `MAX_PARTITION_QUBITS = 2`
- `HYBRID_REGENERATION_ALLOWLIST` is the four paths `cases[0].provenance.implementation_revision` through `cases[3].provenance.implementation_revision`. The predicate reads that tuple at call time.
- `FROZEN_STRUCTURED_BUILDER_CALLS` is two dicts: `family_name="phase31_pair_repeat"`, `qbit_num` 8 and 10, `noise_pattern="dense"`, `seed=20260318`, `max_partition_qubits=2`.

`len(cases)` is 4. All cases `milestone_counted` false and `completeness_claim`
false. `summary.milestone_counted_cases` is 0. q4 `capture_provenance()` is called
once, inside `build_cases()`, before the first cell, and shared by the four cases.
The bundle provenance gate is `all(case["provenance"]["provenance_pass"] for case in cases)`.
`input_artifact_identities` is `[]`.

`Q4_REGENERATION_ALLOWLIST` stays length 1. `FUSED_REGENERATION_ALLOWLIST` stays
length 4. The hybrid predicate is its own function, not the q4 predicate and not
the fused predicate. A revision difference at case `i` is allowlisted only when
the path is in `HYBRID_REGENERATION_ALLOWLIST`, both values are full lowercase
40-hex revisions, current differs from prior, all four current revisions are
equal, and all four prior revisions are equal. A single moved case, with the
other cases unmoved, fails at that case's path. When every case moved and either
side is non-uniform, `cases[0]` fails. The ET-C2-1 matrix states each
`first_mismatch`. A uniform all-equal check is not a substitute for a one-case
divergence assertion.

### 3.4 Close

ADR-F1A-009 (a)–(g), same shape as C.1. No `CLOSEOUT.md` in this pass. C1 is
exactly seven paths: the three Developer paths, `task-6/TASK_6_MINI_SPEC.md`,
`task-6/DELIVERY_STORIES.md`, `task-6/ENGINEERING_TASKS.md`, and
`PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`. C1 holds no bundle, no
`CLOSEOUT.md`, no handback, no q4, fused, or historical JSON, and no other path.
(c) is one pipeline run. C2 commits the hybrid bundle from (c) plus CLOSEOUT and
any checklist touch-ups. After (c), restore q4 and fused with
`git show <C1-sha>:<path> > <path>` and do not stage them. Expected sibling diffs:
q4 has `cases[0].provenance.implementation_revision` and `regeneration.prior_present`;
fused has the four revision paths and `regeneration.prior_present`. (g) restores
q4, fused, and hybrid from the C2 sha and commits nothing. The eight-path
historical diff runs before any restore. Between C1 and (d), `--strict` reports
one `SLICE_MISSING_CLOSEOUT` error for task-6. This re-close was at `step-4a`
(strict warning, exit 0). After this writer pass the stage is `step-4b-authorized`.
Between C1 and (d), `--strict` reports exactly one error, `SLICE_MISSING_CLOSEOUT`
for task-6. After (d), both modes are clean. No placeholder. No waiver.

Pre-(d) independence, for the later CLOSEOUT. Hybrid and oracle share no
execution kernel at these four cells.

- Channel-native partitions use `execute_partition_channel_native` (`noisy_runtime_channel_native.py:903-982`). `_member_to_kraus_bundle` builds numpy Kraus bundles for U3, CNOT, local depolarizing, amplitude damping, and phase damping. `_compose_kraus_bundles` composes them, and `_check_kraus_bundle_invariants` checks completeness and Choi positivity. `_apply_kraus_bundle` then applies the sum of K ρ K† in numpy on `rho.to_numpy()` and returns `DensityMatrix.from_numpy` (`:330-454`).
- Pure-unitary partitions use `_build_fused_kernel` (`noisy_runtime_fusion.py:200-261`) and `DensityMatrix.apply_local_unitary`.
- The oracle applies every gate and channel through `_build_runtime_circuit` and the C++ `NoisyCircuit.apply_to` (`noisy_runtime_core.py:1011-1057`).

A non-counted Architect probe found zero hybrid-side calls to
`_execute_member_sequence` at q4, q6, q8, and q10. Unlike C.1, no noise channel
or singleton unitary goes through the oracle's lowering. Shared:
`validate_runtime_request`, descriptor and parameter routing
(`_build_partition_parameter_vector`, `_segment_parameter_vector`), and
`_build_runtime_circuit`, which the hybrid side only builds for alignment
validation and never applies (`:847-861`). The oracle is not the cell's output
read back. Agreement is not bitwise. Do not copy the q4 bitwise paragraph or the
C.1 partial-share paragraph.

## 4. Unsupported behavior

- Editing `workloads.py` or dropping q8 or q10.
- Changing the two-phase pipeline, QA-001, the oracle, or G-07 exclusions (`_G07_EXCLUDED_SUITES`, `g07_exit_passes`).
- Setting `milestone_counted` true, or `summary.milestone_counted_cases` above 0.
- A labels-only pass, or using `fused_region_count` as the channel-native witness.
- A route class or reason outside the §3.2 vocabulary, including `channel_native_noise_presence`.
- A second allowlisted field, a `cases[*]` wildcard, a widened q4 or fused tuple, or a finding field inside `cases[i]`.
- An output-path exclusion from `clean_start`, or a `run_pipeline` write before another sibling builds.
- Weakening a one-case mutant to an all-equal revision check.
- Omitting the hybrid bundle from C2.
- Starting C.3 or C.4.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-001, REQ-003 | hybrid tests | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a` | `-k mf1a_hybrid` collects 18; `-k mf1a` collects 59 (41 + 18) | ET-C2-1 |
| REQ-003, QA-005 | Layer 1 hybrid lane | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -m "not slow" -p no:cacheprovider` | green; the file is unchanged | this mini-spec |
| REQ-004 | allowlist pin | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_hybrid` | hybrid length 4; q4 length 1; fused length 4; mutant matrix red then green | ET-C2-1 |
| REQ-002, QA-001 | proof (c) and (g) | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | four provisional cases pass; q4 and fused pass; historical diff empty | ET-C2-3 |
| REQ-001 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and `--strict` | warn at the recorded `step-4a` review; one strict error `SLICE_MISSING_CLOSEOUT` for task-6 between C1 and (d); both modes clean after (d) | this mini-spec |

## 6. Tester time

HEAD collect-only `-k mf1a` is 41 tests (0.66 s, no cells). Measured hybrid-entry
wall, in memory, non-counted, at `a006c7e2`: q4 0.002 s, q6 0.008 s, q8 0.82–0.86 s,
q10 19.7–20.0 s. The oracle and eigensolver add time per cell. C.1's (c) was 51 s
and (g) 48 s. Expect about 70–80 s for (c) and (g). The 1–25 s per-cell band holds;
q10 sits near its top, and a loaded host can push it past 25 s. That is a Tech Lead
note, not a fail. New tmux session only.

## 7. Rollback

Revert the three Developer paths and, if C2 landed, the hybrid bundle only.
The fused module is not edited.
