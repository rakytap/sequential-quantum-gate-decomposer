# Task / Work Package 5: fused route provisional evidence (Slice C.1)
> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a C.1 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **Planning-base HEAD:**
> `91680ec720bc311f28e09021895f0c78900d773b` · **Inventory:** `task-4/ROUTE_INVENTORY.md`
> at C.0 `91680ec7` · **Traces:** REQ-001, REQ-002, REQ-004, REQ-006 · QA-001, QA-008 ·
> ADR-F1A-001, ADR-F1A-002, ADR-F1A-004, ADR-F1A-005, ADR-F1A-008,
> ADR-F1A-009 (+ Amendment 1), ADR-F1A-010 · **No push/PR**

## 1. Purpose

Record provisional evidence for `partitioned_density_descriptor_fused_unitary_islands`
at anchors 4, 6, 8, and 10 against `execute_sequential_density_reference`. The four
workload ids are frozen. Records are not the frozen milestone denominator
(`milestone_counted` false). q4 stays baseline route verified. The oracle, QA-001
tolerances, and the G-07 exclusion set stay as they are.

## 2. Scope

### 2.1 Developer paths (exactly three)

| Path | Edit |
|------|------|
| `benchmarks/density_matrix/correctness_evidence/mf1a_fused_validation.py` | new sibling; contract in ET-C1-2 |
| `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | fused registry entry at index 1 (after q4), `mf1a_sibling=True`; two-phase `run_pipeline` below; `_G07_EXCLUDED_SUITES` and `g07_exit_passes` unchanged |
| `tests/partitioning/evidence/test_correctness_evidence.py` | new `mf1a_fused` tests, plus two existing-test hunks in §5 |

`run_pipeline` in one run: (1) build every registered suite, case and nullary, in
registry order; (2) write bundles whose `mf1a_sibling is True`; (3) return results in
registry order. Keep signatures `build_cases()`, `build_artifact_bundle(cases)`, and
`build_artifact_bundle()`. No output-path exclusion. Clean-start detection stays
whole-worktree porcelain (ADR-F1A-009). `--historical-output-dir` refusal and
ADR-F1A-011 decisions 1–2 stay as they are.

Do not edit `workloads.py`, the q4 module body, `SKILL.md`, or historical JSON. q8
and q10 call `workloads.py` and are ADR-F1A-005 protected (call only).

### 2.2 Out of scope

C.2, C.3 (including the RM-1(a) family), C.4, oracle or tolerance edits, dropping a
cell, re-selecting a workload, excluding output paths from `clean_start`, and
committing a regenerated q4 or any of the eight historical bundles.

## 3. Required behavior

### 3.1 Frozen cells

`planner_setting.max_partition_qubits` is 2. Pass `max_partition_qubits=2` on every
builder call and record it from `descriptor_set.max_partition_qubits`. Entry is
`execute_partitioned_density_fused` (`noisy_runtime_core.py:968`).

| Anchor | Workload id | Builder | Protection |
|--------|------------|---------|------------|
| 4 | `phase2_xxz_hea_q4_continuity` | `common.py::build_phase2_continuity_vqe(4)` plus `noisy_descriptor.py::build_phase3_continuity_partition_descriptor_set` | none |
| 6 | `phase2_xxz_hea_q6_continuity` | same builders at 6 | none |
| 8 | `layered_nearest_neighbor_q8_sparse_seed20260318` | `workloads.py::build_structured_descriptor_set("layered_nearest_neighbor", qbit_num=8, noise_pattern="sparse", seed=20260318, max_partition_qubits=2)` | ADR-F1A-005 |
| 10 | `layered_nearest_neighbor_q10_sparse_seed20260318` | same builder at `qbit_num=10` | ADR-F1A-005 |

Builder modules: `benchmarks/density_matrix/planner_surface/common.py`,
`squander/partitioning/noisy_descriptor.py`,
`benchmarks/density_matrix/planner_surface/workloads.py`. Parameters:
`benchmarks/density_matrix/partitioned_runtime/common.py:29`. `seed_policy` is
`deterministic_workload_no_random_seed` on q4 and q6, and
`structured_family_seed_20260318` on q8 and q10.

A non-counted classification probe at budget 2 (Architect C.1 review, not counted
evidence) found 4, 6, 12, and 20 `actually_fused` islands at q4, q6, q8, and q10.
q4 and q6 are real fused cells, not vacuous. The C.0 inventory word `not_shown` meant
no prior test called the fused entry. Zero fusion still fails the case. The workload
id is not swapped. A route that is not realizable at its anchor returns to Research
Manager (ADR-F1A-010).

### 3.2 Pass rule

`actual_fused_execution` means `fused_region_count > 0`
(`noisy_runtime_core.py:216-217`). It is that witness, not a second condition. The
gate is in ET-C1-2: manifest cell match, requested path equals realized path equals
the fused route id, `exact_output_present`, `fused_region_count >= 1`, and
`"actually_fused"` in `fused_region_classifications`.

QA-001 is imported from the q4 module (call only): Frobenius, max-abs, and
`|Tr(rho)-1|` at most `1e-10`; `lambda_min` at least `-1e-12` via
`DensityMatrix.eigenvalues()`; every value finite. No symmetrization.

Findings, from Layer 1 §10, do not change pass/fail, a tolerance, the oracle, or the
counted set. A pure helper classifies them. It may emit into bundle `summary`
(derived, recomputed, not compared). Never put a finding field inside `cases[i]`.

- Frobenius, max-abs, or `|Tr(rho)-1|` above `1e-11` is a finding.
- `lambda_min` below `-1e-13` is a finding.
- Values above `1e-13` are an outside-expected marker.
- Each finding names route, anchor, workload, measure, value, and a cause hypothesis.
- A `lambda_min` finding also reports the oracle `lambda_min` from
  `DensityMatrix.eigenvalues()` as a non-counted diagnostic outside the bundle.
- Matrix-residual findings are flagged to Research Manager at slice close.

### 3.3 Bundle identity

One new sibling. Schema ids, `SUITE_NAME`, filename, and directory are pinned in
ET-C1-2. `len(cases)` is 4, anchor order 4, 6, 8, 10. All cases set
`milestone_counted` false and `completeness_claim` false.
`summary.milestone_counted_cases` is 0. `claim_boundary` names the fused route, the
four cells at budget 2, provisional slice evidence (not the frozen denominator), and
no complete M-F1a, protocol, Aer, energy, or frozen-matrix claim.

`Q4_REGENERATION_ALLOWLIST` stays length 1. The fused module has its own length-4
tuple of `cases[i].provenance.implementation_revision` for `i` in 0..3. The predicate
is new code with the Slice B negatives (ET-C1-1). It does not call the q4 predicate.
Capture provenance once per run with q4 `capture_provenance` and
`REGENERATION_COMMAND`, and share that object across the four cases.
`input_artifact_identities` is `[]`.

### 3.4 Close shape (ADR-F1A-009 (a)–(g))

No `CLOSEOUT.md` in this pass.

- **(a)** Reviewer on the uncommitted Developer diff plus this Layer 2–4.
- **(b) C1** the three Developer paths, tests, and planning docs. No bundle. No CLOSEOUT.
- **(c)** Clean C1, one pipeline run. Copy both sibling JSON files to `/tmp/<run>/`
  before restore. Record sha256 before and after restore.
- **Pre-(d) G-10.** Partial sharing. Fused islands use `_build_fused_kernel`
  (`noisy_runtime_fusion.py:200-261`) and `DensityMatrix.apply_local_unitary`. The
  oracle applies each gate through `_build_runtime_circuit` and `circuit.apply_to`.
  Noise and singleton unitaries on the fused route go through `_execute_member_sequence`
  (`noisy_runtime_fusion.py:369-375`, `:418-424`, `:440-446`), the same NoisyCircuit
  lowering as the oracle. `validate_runtime_request` and parameter routing are shared.
  Agreement is not bitwise. Phase-3 non-counted context only: Frobenius about
  `1.02e-15` at q8 and `1.54e-15` at q10. The CLOSEOUT does not copy the q4
  shared-kernel paragraph. It records per-cell island composition, including that
  CNOT-in-kernel fusion is exercised only at q4 and q6.
- **(d)** Real `task-5/CLOSEOUT.md` with the four QA-001 values per row and findings.
- **(e)** Reviewer evidence review. **(f)** and **(g)** follow §3.5.

A QA-001 failure or a q4 regeneration failure at (c) or (g) stops the slice before (d).
An environment or extension mismatch goes to Tech Lead. Anything else goes to Research
Manager (ADR-F1A-005, ADR-F1A-010, Amendment 1 consequence 2). Before (c) and (g), the
loaded `.so` sha256 equals the committed q4 `extension_identities[0].sha256`
(`05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` at this HEAD).

This re-close was at `step-4a` (strict warning, exit 0). After this writer pass the
stage is `step-4b-authorized`. Between C1 and (d), `--strict` reports exactly one
error, `SLICE_MISSING_CLOSEOUT` for task-5. After (d), both modes are clean. No
placeholder. No waiver.

### 3.5 Counted-evidence commit policy

Accepted by the Architect C.1 review (ADR-F1A-009 (f)/(g) and Amendment 1 consequence 1).
Not an open flag.

1. C1 contains no bundle. The C2 path list is the fused bundle written at (c),
   `task-5/CLOSEOUT.md`, and the checklist if the (d) writer touches it. The q4 bundle
   and the eight historical bundles are never staged.
2. The CLOSEOUT records the fused bundle sha256 taken at (c). Reviewer (e) confirms
   the staged bytes equal it.
3. After (c), restore q4 with `git show <C1-sha>:<q4-path> > <q4-path>`. After (g),
   restore both siblings from `<C2-sha>`. Never stash, reset, checkout, or clean.
4. The eight-path historical diff runs before any restore. A non-empty result is a
   defect (Amendment 1 consequence 3; ADR-F1A-011).
5. This policy depends on the two-phase pipeline in §2.1.

## 4. Unsupported behavior

- Editing `workloads.py` or re-selecting a frozen id.
- Using `max_partition_qubits` 4 or excluding sibling paths from porcelain.
- Changing QA-001, the oracle, or `_G07_EXCLUDED_SUITES`.
- Setting `milestone_counted` true, or putting a finding field inside `cases[i]`.
- Hiding a finding, or letting a finding change pass/fail.
- Starting C.2, C.3, or C.4.

## 5. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner artifact |
|----------|---------------|----------------|-----------------|----------------|
| REQ-001, REQ-002 | fused tests | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a` | `-k mf1a_fused` collects 12; `-k mf1a` collects 41 (29 + 12) | DS-C1-1; ET-C1-1 |
| REQ-004 | regeneration pin | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_fused` | fused allowlist length 4; q4 allowlist length 1; Slice B negatives | ET-C1-1 |
| REQ-002, QA-001 | proof runs (c) and (g) | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | four provisional cases pass; q4 passes under Slice B; historical eight-path diff empty | ET-C1-3 |
| REQ-001 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and `--strict` | warn at the recorded `step-4a` review; one strict error `SLICE_MISSING_CLOSEOUT` for task-5 between C1 and (d); both modes clean after (d) | this mini-spec |

## 6. Tester time

The 1–25 s per cell and 4–100 s four-cell bands stay as a conservative rocky bound.
Slice B's full pipeline took 56 s. A classification probe's fused-entry times were
0.001 s at q4 and 0.004 s at q6. Expect about 60–70 s for the whole (c) run on rocky.
Run (c) and (g) in a new tmux session. `test_QX2` is outside `-k mf1a`. No
state-vector lane is required, because no `squander/` or C++ path changes.

## 7. Rollback

Revert the three Developer paths and, if C2 has landed, the fused bundle. Do not revert
q4 or `workloads.py`.
