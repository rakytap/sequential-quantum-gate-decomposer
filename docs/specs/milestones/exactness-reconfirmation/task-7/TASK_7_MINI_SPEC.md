# Task / Work Package 7: strict route provisional evidence (Slice C.3)
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.3 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Planning-base HEAD:**
> `42922382aaf1bdf5d609d169110b75cd72c9a757` · **Inventory:** `task-4/ROUTE_INVENTORY.md`
> · **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006 · QA-001, QA-005, QA-008 ·
> ADR-F1A-001, ADR-F1A-003, ADR-F1A-005, ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 ·
> **No push/PR**

## 1. Purpose

Record provisional evidence for `phase31_channel_native` at anchors 4, 6, 8, and 10
against `execute_sequential_density_reference`. q4 keeps the inventory workload id.
q6, q8, and q10 use the RM-1(a) family frozen here. `milestone_counted` stays false.
q4 remains baseline route verified. C.1 and C.2 stay closed. This slice does not
change the oracle, QA-001 tolerances, the denominator, or G-07 exclusions.

## 2. Scope

### 2.1 Developer paths (exactly three)

| Path | Edit |
|------|------|
| `benchmarks/density_matrix/correctness_evidence/mf1a_strict_validation.py` | new. Holds the family builder (§3.1), the witness-only record and gate (§3.2), the Layer 1 §10 helper, and `STRICT_REGENERATION_ALLOWLIST` as its own function. QA-001, `capture_provenance`, and `REGENERATION_COMMAND` come from the q4 module. The runtime comes from `squander.partitioning.noisy_runtime` (`PHASE31_RUNTIME_PATH_CHANNEL_NATIVE`, `execute_partitioned_density_channel_native`, `execute_sequential_density_reference`). q4 calls `workloads.build_phase31_microcase_descriptor_set` only |
| `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | one import `mf1a_strict_validation as mf1a_strict` beside the three `mf1a_*` imports, plus one `_CaseSuiteEntry(mf1a_strict, "build_cases", "build_artifact_bundle", mf1a_sibling=True)` at index 3, after hybrid. Do not redesign `run_pipeline` |
| `tests/partitioning/evidence/test_correctness_evidence.py` | 25 new `mf1a_strict` functions plus their `_mf1a_strict_*` helpers, and exactly two existing-test hunks (ET-C3-1) |

Sibling directory `mf1a/strict`. The sibling set becomes
`{"mf1a/q4_baseline", "mf1a/fused", "mf1a/hybrid", "mf1a/strict"}`. Reviewer (a)
checks exactly those two hunks and the import-plus-index-3 limit. Do not edit
`workloads.py`, the q4, fused, or hybrid module bodies, or `SKILL.md`.

### 2.2 Out of scope

C.4, the hybrid `S_len_range` test-gap fix, the task-4 CLOSEOUT nit, oracle or
tolerance edits, dropping an anchor, and committing regenerated q4, fused, hybrid,
or historical bundles. Not touched: `workloads.py`, the q4, fused, and hybrid module
bodies, `planner_surface/common.py`, any `squander/` path, C++, any skill, any JSON,
and any other test file.

## 3. Required behavior

### 3.1 Frozen cells

`max_partition_qubits=2` on every builder call. Entry is
`execute_partitioned_density_channel_native` (`noisy_runtime_core.py:980-989`,
`runtime_path=PHASE31_RUNTIME_PATH_CHANNEL_NATIVE`, `allow_fusion=False`).

| Anchor | Id | Builder |
|---|---|---|
| 4 | `phase31_local_support_q4_spectator_embedding_smoke` | `workloads.build_phase31_microcase_descriptor_set(<id>, max_partition_qubits=2)` (ADR-F1A-005, call only) |
| 6 | `mf1a_strict_spectator_embed_q6` | `build_mf1a_strict_spectator_descriptor_set(6)` |
| 8 | `mf1a_strict_spectator_embed_q8` | `build_mf1a_strict_spectator_descriptor_set(8)` |
| 10 | `mf1a_strict_spectator_embed_q10` | `build_mf1a_strict_spectator_descriptor_set(10)` |

`mf1a_strict_spectator_operation_specs(qbit_num)` returns, for k = 0 … qbit_num/2 − 1
in order and with a = 2k and b = 2k+1, exactly these six specs:

1. `{"kind": "gate", "name": "U3", "target_qbit": a, "param_count": 3}`
2. `{"kind": "gate", "name": "U3", "target_qbit": b, "param_count": 3}`
3. `{"kind": "gate", "name": "CNOT", "target_qbit": b, "control_qbit": a, "param_count": 0}`
4. `{"kind": "noise", "name": "amplitude_damping", "target_qbit": b, "source_gate_index": 4k+2, "fixed_value": 0.05, "param_count": 0}`
5. `{"kind": "noise", "name": "phase_damping", "target_qbit": a, "source_gate_index": 4k+2, "fixed_value": 0.07, "param_count": 0}`
6. `{"kind": "gate", "name": "U3", "target_qbit": a, "param_count": 3}`

The function raises `ValueError` for odd `qbit_num` or `qbit_num < 2`. Here 4k+2 is
that pair's CNOT gate index: gates only, four per pair. Local depolarizing is not
used (Architect freeze; Research Manager preference via Tech Lead, 2026-10-06). The
rates are the `_noise_value` defaults (`workloads.py:67-72`).

`build_mf1a_strict_spectator_descriptor_set(qbit_num)` calls
`build_canonical_planner_surface_from_operation_specs(qbit_num=qbit_num, source_type="structured_family_builder", workload_id=f"mf1a_strict_spectator_embed_q{qbit_num}", operation_specs=...)`,
then `build_partition_descriptor_set(surface, max_partition_qubits=MAX_PARTITION_QUBITS)`.
Both are imported from `squander.partitioning.noisy_planner`, as `workloads.py:6-10`
does. The module does not import `workloads.py`'s private helpers.

At `qbit_num = 4` the list equals the smoke's `operation_specs` (`workloads.py:186-209`)
op for op. The Developer pins this.

`seed_policy` is `deterministic_workload_no_random_seed` on all four cases. Parameters
are `build_initial_parameters(descriptor_set.parameter_count)`, recorded as
`parameters` and compared exactly at regeneration. The module imports no
random-number generator.

> A non-counted Architect probe at `42922382` (strict entry only; no oracle, QA-001, or bundle) found every partition eligible with exactly one `channel_native_motif`/`actually_fused` witness (`channel_native_motif_kraus_count_4`) and no islands: q4 smoke 2/2 (0.002 s); q6 3/3 (0.006 s); q8 4/4 (0.089 s); q10 5/5 (2.61–2.62 s). Witness k targets global (2k, 2k+1); partitions k ≥ 1 are remapped. Runtime partition records carry no class or reason.

### 3.2 Pass rule (ADR-F1A-003)

The strict entry emits no per-partition labels, and this module does not synthesize
its own (ADR-F1A-003). The `realization` keys are exactly:

- `requested_path` and `realized_path`;
- `partition_count`, from `len(descriptor_set.partitions)`;
- `exact_output_present`;
- `channel_native_partition_count`: the number of entries in `result.partitions` with exactly one `channel_native_motif`/`actually_fused` region;
- `partitions`: rows `{"partition_index": record.partition_index}` from `result.partitions`, and nothing else;
- `fused_regions`: rows `partition_index`, `candidate_kind`, `classification`, `reason`, `operation_names`, `global_target_qbits`.

No `partition_runtime_class`, `partition_route_reason`, `runtime_class_counts`,
`route_reason_counts`, `fused_region_count`, `actual_fused_execution`, `runtime_ms`,
or `peak_rss_kb` key appears. The case keys, their order, and the regeneration exact
paths are the hybrid record's (`mf1a_hybrid_validation.py:355-372`, `:537-550`),
with strict values. `planner_setting.max_partition_qubits` is taken from
`descriptor_set.max_partition_qubits`.

`_route_realization_pass` passes only when S1–S11 all hold, one check each. The
guard table is in ET-C3-2. S3–S5 are consistency checks only: the strict branch sets
the realized path and `exact_output_present` unconditionally
(`noisy_runtime_core.py:937-938`, `:958`). The witness is S9 plus S10.

For a pure-unitary partition, the strict entry raises `NoisyRuntimeValidationError`
with `category="unsupported_runtime_operation"`,
`first_unsupported_condition="channel_native_noise_presence"`, and
`failure_stage="runtime_preflight"` (`noisy_runtime_channel_native.py:522-531`).
`build_cases(*, provenance=None)` does not catch it. The error propagates,
`run_pipeline` raises before any sibling write, the command exits nonzero, and no
bundle is written. The id is not swapped, and the stop returns to Research Manager
(ADR-F1A-010).

This slice meets REQ-003 and QA-005 through their negative clauses: no silent route
substitution (S3, S4), and an ineligible strict partition raises a structured error
(QA-005, `PRODUCT_STATEMENT.md:85`). The evaluation-mode label clauses stay with
hybrid (ADR-F1A-003).

QA-001 is imported from the q4 module (call only). Layer 1 bands, as ruled for C.2 (e):
the outside-expected marker applies only to matrix residuals (Frobenius, max-abs, and
absolute trace deviation). A matrix residual above `1e-11` is a finding; above `1e-10`
fails QA-001. `lambda_min` is one-sided: a finding below `-1e-13`, QA-001 floor
`-1e-12`, and no upper bound. A positive `lambda_min` is not a finding and not a marker.
A finding does not change pass/fail, a tolerance, the oracle, or the counted set.
No finding field inside `cases[i]`. No bundle key named `oracle_lambda_min`.

`claim_boundary` literal:

```text
Provisional strict-route slice evidence for phase31_channel_native at anchors 4, 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or frozen-matrix claim.
```

### 3.3 Bundle and allowlist

- `SUITE_NAME = BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_strict_bundle_v1"`
- `RECORD_SCHEMA_VERSION = "correctness_evidence_mf1a_strict_case_v1"`
- `MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_strict_manifest_v1"`
- `ARTIFACT_FILENAME = "mf1a_strict_bundle.json"`
- `DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "strict"`
- `ROUTE = PHASE31_RUNTIME_PATH_CHANNEL_NATIVE`
- `MAX_PARTITION_QUBITS = 2`
- `STRICT_REGENERATION_ALLOWLIST` is the four `cases[i].provenance.implementation_revision` paths, read at call time.

`len(cases)` is 4. All cases `milestone_counted` false and `completeness_claim` false.
`summary.milestone_counted_cases` is 0. q4 `capture_provenance()` runs once inside
`build_cases()`, before the first cell, and is shared. The provenance gate is
`all(case["provenance"]["provenance_pass"] for case in cases)`.
`input_artifact_identities` is `[]`.

`Q4_REGENERATION_ALLOWLIST` stays length 1. Fused and hybrid allowlists stay length 4.
The strict predicate is its own function. A revision difference at case `i` is
allowlisted only when the path is in the tuple, both values are full lowercase 40-hex
revisions, current differs from prior, and all four current revisions are equal and
all four prior revisions are equal. A single moved case, with the other cases unmoved,
fails at that case's path. When every case moved and either side is non-uniform,
`cases[0]` fails.

### 3.4 Close

ADR-F1A-009 (a)–(g). No `CLOSEOUT.md` in this pass. C1 is exactly seven paths: the
three Developer paths, the three `task-7/` specs, and this checklist. C1 holds no
bundle, no `CLOSEOUT.md`, no handback, and no q4, fused, hybrid, or historical JSON.
(c) is one pipeline run. `run_pipeline` builds every case slice before the first write
(`validation_pipeline.py:197-221`). After (c), restore q4, fused, and hybrid with
`git show <C1-sha>:<path> > <path>` and do not stage them. Committed sha256 values at
this HEAD: q4 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`;
fused `3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0`;
hybrid `33aaa442f69e533e599e64895346786831a9591ae2a1db83dc80bf0643532ab9`.
Expected sibling diffs are q4 2 paths and fused and hybrid 5 paths each (the revision
paths plus `regeneration.prior_present`), as in C.2 (c) and (g). C2 commits the strict
bundle from (c) plus CLOSEOUT and checklist touch-ups. (g) restores all four siblings
from the C2 sha and commits nothing. The eight-path historical diff runs before any
restore. This re-close was at `step-4a` (strict warning, exit 0). After this writer pass the
stage is `step-4b-authorized`. Between C1 and (d), `--strict` reports exactly one
error, `SLICE_MISSING_CLOSEOUT` for task-7. After (d), both modes are clean. No
placeholder. No waiver.

> Strict and the oracle share no execution kernel at these four cells. Every strict partition runs `execute_partition_channel_native` (`noisy_runtime_channel_native.py:903-982`): `_member_to_kraus_bundle` builds numpy Kraus bundles for U3, CNOT, amplitude damping, and phase damping; `_compose_kraus_bundles` composes them; `_check_kraus_bundle_invariants` checks completeness and Choi positivity; `_apply_kraus_bundle` embeds each 4×4 operator on global (2k, 2k+1) and applies the sum of K ρ K† in numpy (`:303-454`). The oracle applies every gate and channel through `_build_runtime_circuit` and C++ `NoisyCircuit.apply_to` (`noisy_runtime_core.py:1011-1057`). A non-counted Architect probe at `42922382` counted zero strict-side calls to `_execute_member_sequence`, `_build_fused_kernel`, and `NoisyCircuit.apply_to` at q4–q10. `_build_runtime_circuit` ran once per partition, for alignment validation only (`:847-861`). Shared: `validate_runtime_request`, descriptor and parameter routing (`_build_partition_parameter_vector`, `_segment_parameter_vector`), and that alignment-only build. The oracle is not the cell's output read back. Agreement is not bitwise. Do not copy the q4 bitwise paragraph or the C.1 partial-share paragraph.

## 4. Unsupported behavior

- Editing `workloads.py` or replacing the q4 inventory id.
- A labels-only pass, or a pure-unitary partition treated as strict-eligible.
- Writing per-partition labels or label counts into the strict record.
- Catching the strict preflight error.
- A seed suffix on a new id, or a seeded parameter draw.
- Any motif other than the §3.1 template, or any cross-pair coupling.
- Rates other than the `_noise_value` defaults, or a channel outside the delivered three.
- Any `max_partition_qubits` other than 2, or any change to the `claim_boundary` wording.
- A timeout or wall-time assertion.
- Setting `milestone_counted` true, a second allowlisted field, or a finding field inside `cases[i]`.
- Changing `run_pipeline`, QA-001, the oracle, or G-07.
- Starting C.4.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-001, REQ-003 | strict tests | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a` | `-k mf1a_strict` collects 25; `-k mf1a` collects 84 (59 + 25) | ET-C3-1 |
| REQ-003, QA-005 | Layer 1 strict lane | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/test_partitioned_channel_native_phase31_slice.py tests/partitioning/test_partitioned_channel_native_phase31_second_slice.py -m "not slow" -p no:cacheprovider` | green; both files unchanged; 27 tests collected at HEAD. The second holds `test_phase31_channel_native_public_4q_smoke_matches_sequential` (`:588`) | this mini-spec |
| REQ-004 | allowlist pin | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_strict` | strict length 4; q4 length 1; fused and hybrid length 4 | ET-C3-1 |
| REQ-002, QA-001 | proof (c) and (g) | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | four provisional cases pass; three siblings pass; historical diff empty | ET-C3-3 |
| REQ-001 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and `--strict` | warn at the recorded step-4a review; one strict error SLICE_MISSING_CLOSEOUT for task-7 between C1 and (d); both modes clean after (d) | this mini-spec |

## 6. Tester time

> Tester time is scheduling guidance, not an acceptance criterion. The 1–25 s per-cell and 4–100 s four-cell figures (task-5 §6) are conservative rocky bounds. No gate, test, record field, or exit code depends on wall time (Layer 1 §2 puts timing out of scope), and the case record has no `runtime_ms` or `peak_rss_kb`. Measured strict-entry wall, non-counted, at `42922382`, frozen family: q4 0.002 s, q6 0.006 s, q8 0.089 s, q10 2.61–2.62 s. C.2's (c) and (g) took 78 s and 72 s for three siblings; expect about 80–90 s for C.3's. If a cell or the run passes a band, Tester records the measured time in the CLOSEOUT and continues. That is not a fail, a stop, or a Research Manager trigger. Do not add a timeout or a wall-time assertion. Run (c) and (g) in a new tmux session.

## 7. Rollback

Revert the three Developer paths and, if C2 landed, the strict bundle only.
