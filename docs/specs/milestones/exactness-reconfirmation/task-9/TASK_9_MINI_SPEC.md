# Task / Work Package 9: counted manifest for the 16-cell denominator

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-06 by Squander Architect; Step 4b under ADR-F1A-010 after the Research Manager freeze record · **Slice:** M-F1a task-9 ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Planning-base HEAD:** `a81be56baba7e97156532524978cc878d347fa87` ·
> **Manifest:** `task-9/COUNTED_MANIFEST.md` ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-006, REQ-007 ·
> QA-001, QA-005, QA-008 · ADR-F1A-001, ADR-F1A-002, ADR-F1A-003, ADR-F1A-004,
> ADR-F1A-005, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010, ADR-F1A-011 ·
> **No push/PR**

## 1. Purpose

Freeze and then execute the ADR-F1A-001 denominator: 16 cells, four routes at anchors 4, 6, 8, and 10. The workloads, parameter counts, and seed policies are `COUNTED_MANIFEST.md`. This slice re-executes those cells. It does not swap a workload.

Closed code-ready under ADR-F1A-008 on 2026-10-06. Research Manager accepted `claim_boundary` and `completeness_claim` (`COUNTED_MANIFEST.md` §§4–5) and recorded the ADR-F1A-001 freeze in `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` §10. No counted run precedes that record (ADR-F1A-010 item 3).

## 2. What stays unchanged

- Oracle `execute_sequential_density_reference`.
- QA-001 and ADR-F1A-002 tolerances: matrix residuals `1e-10`, `lambda_min` floor `-1e-12`, imported from `mf1a_q4_baseline_validation.py` (`MF1A_QA001_MATRIX_TOL`, `MF1A_QA001_LAMBDA_MIN_FLOOR`, `evaluate_mf1a_qa001`). Not re-implemented.
- G-07 exclusions stay `external_correctness` and `output_integrity`. Adding this sibling is not a G-07 change.
- O-11 `max_partition_qubits` = 2.
- No Aer or energy row in the 16.
- No edit to `workloads.py`, `planner_surface/common.py`, the five sibling modules, `noisy_runtime_core.py`, `correctness_evidence/common.py`, or any REQ-007 path.
- Provisional sibling bundles and their `claim_boundary` strings stay. Their shas stay the historical slice record.
- Checklist §9 deferrals are not in this slice's C1 or C2.
- G-03, G1, and G3 stay open until the counted bundle passes. The Linux CI job (G-04, G6) and current-state docs (G-05, G7) stay at milestone close.

## 3. Counted bundle

Proposed path, not git-ignored, outside the REQ-007 set:

`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/counted/mf1a_counted_bundle.json`

`SUITE_NAME = BUNDLE_SCHEMA_VERSION = "correctness_evidence_mf1a_counted_bundle_v1"`; `MANIFEST_SCHEMA_VERSION = "correctness_evidence_mf1a_counted_manifest_v1"`; `ARTIFACT_FILENAME = "mf1a_counted_bundle.json"`; `DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / "mf1a" / "counted"`; `COUNTED_REGENERATION_ALLOWLIST` is the 16 revision paths, with its own predicate. `build_cases(*, provenance=None)` captures once, calls each sibling `build_cases` through its module attribute, orders rows 1–16, and restamps only `milestone_counted`, `completeness_claim`, and `claim_boundary`. All other fields stay the sibling's, including the record and manifest schema ids. `build_artifact_bundle(cases, *, prior_bundle=_NO_PRIOR, non_counted_context=None)`. `manifest.cells` holds rows 1–16 with `route`, `anchor_qbits`, `workload`, `max_partition_qubits`, `seed_policy`, `parameter_count` (= `len(parameters)`). Gates: `manifest_exact_set`, `route_realization`, `qa001`, `provenance`, `regeneration`. Summary: `total_cases`, `qa001_passes`, `milestone_counted_cases` (summed flags), `completeness_claim`, `first_failure`, and `findings`/`outside_expected_markers` from `mf1a_baseline_validation._derive_summary_findings` by identity. Regeneration is the C.1 rule on 16 cases. A revision difference is allowlisted only if all 16 current and all 16 prior revisions are each one full revision, with no majority reference. A bad prior schema or length gives `bundle_structure`. A count or exact-set miss gives `manifest_exact_set`.

The pipeline still runs the five sibling `build_cases()` calls for the provisional bundles, so one invocation executes the 16 cells twice.

Restamp:

- `milestone_counted` is true on every case only when `clean_start` and `provenance_pass` are both true. A dirty tree sets every case false, fails the provenance gate, and sets `summary.milestone_counted_cases` to the count of true flags (0 when dirty). The summary count is derived, not a literal 16 on a dirty run.
- `completeness_claim` and `claim_boundary` are the Research Manager-accepted values in `COUNTED_MANIFEST.md` §§4–5. Developer copies them and does not invent either value.
- Each row is gated by the `_route_realization_pass` of the module that produced it, held in the 16-tuple `_ROW_GATE_MODULES`: row 1 `mf1a_q4_baseline_validation`; rows 2–4 `mf1a_baseline_validation`; rows 5–8 `mf1a_fused_validation`; rows 9–12 `mf1a_strict_validation`; rows 13–16 `mf1a_hybrid_validation`. The five gates are unchanged and nothing is added to `route_realization`. Hybrid rows keep the full label distribution.
- Regeneration allowlist is the 16 paths `cases[0]`…`cases[15]` `.provenance.implementation_revision`. q4 stays length 1. Fused, hybrid, and strict stay length 4. Baseline stays length 3.

Per-cell closeout wording is "`<route>` route verified at q`<n>`".

## 4. Developer paths

| Path | Edit |
|------|------|
| `benchmarks/density_matrix/correctness_evidence/mf1a_counted_validation.py` | new. Imports the five sibling modules, `capture_provenance`, `REGENERATION_COMMAND`, and the QA-001 tolerances from the q4 module, and the finding helpers from `mf1a_baseline_validation`. QA-001 and the oracle run inside the siblings (test 13). Does not import `workloads` (test 15) |
| `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | one import and one `_CaseSuiteEntry(..., mf1a_sibling=True)` at index 5, after baseline |
| `tests/partitioning/evidence/test_correctness_evidence.py` | 29 new `mf1a_counted` tests and exactly two existing hunks |

C1 is these eight paths. No counted bundle and no `CLOSEOUT.md` in C1.

1. `benchmarks/density_matrix/correctness_evidence/mf1a_counted_validation.py`
2. `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
3. `tests/partitioning/evidence/test_correctness_evidence.py`
4. `docs/specs/milestones/exactness-reconfirmation/task-9/TASK_9_MINI_SPEC.md`
5. `docs/specs/milestones/exactness-reconfirmation/task-9/DELIVERY_STORIES.md`
6. `docs/specs/milestones/exactness-reconfirmation/task-9/ENGINEERING_TASKS.md`
7. `docs/specs/milestones/exactness-reconfirmation/task-9/COUNTED_MANIFEST.md`
8. `docs/specs/milestones/exactness-reconfirmation/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`

Before Reviewer (a), context headers in this milestone match that position. At (a), diff these five sibling modules against `a81be56` and expect an empty diff: `mf1a_q4_baseline_validation.py`, `mf1a_baseline_validation.py`, `mf1a_fused_validation.py`, `mf1a_strict_validation.py`, and `mf1a_hybrid_validation.py`. The sweep does not edit `COUNTED_MANIFEST.md`; its bytes are final before its sha is recorded, and a later edit voids the freeze.

## 5. Process

1. This Step 4a pack.
2. Architect code-ready review. The verdict states oracle, QA-001 and the allowlist rule, and G-07 unchanged. Item 3 is not unchanged: the counted set is this manifest, and it returns to Research Manager.
3. RM freeze record and alignment. `claim_boundary` and `completeness_claim` are `COUNTED_MANIFEST.md` §§4–5. The freeze record is the checklist §10 text and names the manifest sha. It is not stored in the manifest.
4. Code-ready writer pass sets `step-4b-authorized` only after those three marks are cleared. A manifest change between C1 and the counted run restarts the slice at (a).
5. Step 4b: (a) implementation review, C1, Tester (c) from a clean tree, (d) CLOSEOUT, (e), C2, (g).
6. C2 commits the counted bundle, the CLOSEOUT, and the checklist touch. It cannot name its own sha. The next planning pass records that sha. (g) restores all six siblings from the C2 sha and commits nothing.

## 6. Disclosures (non-blockers)

**Baseline shared kernel.** Baseline and the oracle both lower through `_build_runtime_circuit` and the same C++ `NoisyCircuit` kernels (`task-1/CLOSEOUT.md:31-50`, `task-8/CLOSEOUT.md:88-97`). Bitwise agreement is expected. A kernel-level bug appears on both sides, and this oracle cannot detect it. The cells still meet the baseline rule.

**Strict product of pair states, all four anchors.** q4 is two partitions on (0, 1) and (2, 3). q6, q8, and q10 add the same pair block at each (2k, 2k+1). No operation couples two pairs, so every state is a product of pair states (`task-7/CLOSEOUT.md:103`; strict bundle q4 regions at `:78-93` and `:96-111`). These cells do not test channel application on inputs correlated across a partition boundary, the order of partitions (disjoint channels commute), local depolarizing under the strict entry, or single-wire and odd-aligned motifs. C.2 hybrid covers correlated inputs and depolarizing. The strict cells still execute an eligible motif on every partition.

**Other routes, for the G-10 note only.** Fused shares noise and singleton lowering and does not share the island kernel; agreement is not bitwise (`task-5/CLOSEOUT.md:87-89`). Hybrid does not send noise or a singleton through the oracle's lowering; agreement is not bitwise (`task-6/CLOSEOUT.md:102-112`). Strict shares no execution kernel; agreement is not bitwise (`task-7/CLOSEOUT.md:120-132`).

## 7. Tester gate (after the freeze)

A dirty start is non-counted: park and record its outputs, then rerun from a clean C1 via Tech Lead.

From empty `git status --porcelain --untracked-files=all` at C1, in a new tmux session, one command, no timeout:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

Before (c), Tester records the `sha256sum` of `docs/specs/milestones/exactness-reconfirmation/task-9/COUNTED_MANIFEST.md` and of `git show <C1-sha>:<that path>`. Both must equal the RM freeze-record sha. A mismatch stops before (c) and goes to Research Manager. The loaded `_density_matrix_cpp` `.so` sha256 and the `qgd` versions match `extension_identities[0]` in the committed q4 bundle. A mismatch stops the run and goes to Tech Lead. Record sha256 of the five provisional siblings and the eight historical bundles.

After the run, before any restore: the REQ-007 eight-path diff is empty. A non-empty result is a defect. Pre-restore diffs: at (c), q4 2, fused, hybrid, and strict 5 each, baseline 4; at (g), the same plus 17 for counted. Each provisional sibling passes regeneration with only the allowlisted revision differences. Any other sibling failure stops before (d): environment or extension to Tech Lead, anything else to Research Manager. Restore the five provisional files to their C1 bytes. Leave the counted bundle untracked for C2.

The CLOSEOUT tabulates Frobenius, max-abs, `|Tr-1|`, and `lambda_min` for all 16 rows. `lambda_min` is one-sided. A counted QA-001 or realization failure names the first failing route, case, and measure, freezes downstream claims, and goes to Research Manager. No workload swap, no rerun to pass, and no tolerance or denominator change.

Scheduling only, not a gate: the C.4 pipeline was 79.5 s at (c) and 82.1 s at (g). A second pass over the 16 cells puts the counted invocation near 105–140 s.

## 8. Unsupported behavior

- Editing `workloads.py` or any of the five sibling modules.
- Flipping `milestone_counted` inside those five modules.
- A sixth route, a dropped anchor, or a workload that is not in `COUNTED_MANIFEST.md`.
- Aer or energy inside the counted set.
- A counted run before the RM freeze record.
- A placeholder `CLOSEOUT.md`.
- Putting a checklist §9 deferral into C1 or C2.
- Claiming the milestone outcome in this slice.

## 9. Acceptance evidence

| Trace id | Evidence type | Command / gate | Expected result | Owner |
|----------|---------------|----------------|-----------------|-------|
| REQ-001, REQ-002, QA-001 | tests | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a_counted` | collects 29 | ET-C9-1 |
| REQ-001 | tests | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k mf1a` | collects 137 (108 + 29) | ET-C9-1 |
| REQ-004, QA-008 | pipeline | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | exit 0 after the freeze; counted sibling written; five provisional siblings restored | ET-C9-3 |
| REQ-007 | static diff | `git diff --exit-code 1cb3d20c -- benchmarks/density_matrix/planner_surface/workloads.py` | empty | ET-C9-2 |
| QA-008 | spec fitness | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` and `--strict` | at step-4a, the only finding is `SLICE_MISSING_CLOSEOUT` for task-9, a warning in both modes; after the stage flip to `step-4b-authorized`, `--strict` reports that finding as the one error until (d); normal mode keeps the warning; both modes clean after (d) | this mini-spec |

## 10. Rollback

Revert the three Developer paths and, if C2 landed, the counted bundle only. Do not revert the five provisional bundles.

## 11. Verdict

**Closed code-ready under ADR-F1A-008 on 2026-10-06.** Step 4b follows the Research Manager freeze record in the checklist §10 and Tech Lead's ADR-F1A-010 handoff.
