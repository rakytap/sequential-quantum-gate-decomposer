# M-F1a slice C.4 closeout — baseline route provisional evidence at 6, 8, and 10 (task-8)

> **Status:** ready for Reviewer evidence review · **Date:** 2026-10-06 · **Work package:** task-8 ·
> **Scope:** baseline route at anchors 6, 8, and 10; M-F1a remains open ·
> **Claim:** baseline bundle generated at clean C1 ·
> **Revision C1:** `55e3783782144e652d80049ea8269e3f4a96ec25` ·
> **C1 parent:** `aaf6fcfe8e1a317485bad4d67580d1ad8082d11d` ·
> **C2:** this commit ·
> **No push/PR**

## Summary

Baseline bundle generated at clean C1. This closeout records the post-C1 clean-start run
(step (c)). Records stay provisional: every case has `milestone_counted` false and
`summary.milestone_counted_cases` is 0. It does not change the oracle, QA-001
tolerances, the counted denominator, scope, or the G-07 exclusion set. q4 remains
baseline route verified. The q4 bundle is not extended. G-03 stays open.

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| Planning re-close | done | Architect NOT-READY `bc-5d7c4fdb` run 3 (B1–B3); CODE-READY run 4 |
| Docs review | done | code-ready writer APPROVE `bc-d6c39d64` |
| (a) Reviewer implementation review | done | APPROVE `bc-bc146637` |
| (b) C1 Tech Lead local commit | done | C1 `55e37837`; parent `aaf6fcfe`; 7 paths |
| (c) Clean-start pipeline run | done | exit 0; 79.5 s; Tester `bc-9820d3c5` run 17 |
| (d) Real `CLOSEOUT.md` | this file | ET-C4-3; uncommitted |
| (e) Reviewer evidence review | pending | after this write |
| (f) C2 Tech Lead local commit | this commit | baseline bundle from (c) + this CLOSEOUT + checklist touch-ups |
| (g) Clean-C2 regeneration | pending | after C2; restore all five siblings from the C2 sha; do not commit |

## (c) Pipeline run

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `55e3783782144e652d80049ea8269e3f4a96ec25` |
| Porcelain before run | empty |
| Extension `.so` sha256 | `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |
| Wall time | 79.5 s |
| Exit code | **0** |
| Baseline bundle sha256 | `0316a303d03b096dd132d94bd6cfc7742324234cfa141c131b58b10ad8cc8976` |
| Evidence | Tester `bc-9820d3c5` run 17 · `/tmp/c4-c-proof/REPORT.md` |

`-k mf1a_baseline` is 24 passed in 70.29 s. `-k mf1a` is 108 passed in 72.32 s.
The evidence file with no `-k` collects 127 tests and 127 passed in 126.63 s.

## QA-001

Route on every row: `partitioned_density_descriptor_baseline`. `max_partition_qubits` is 2.
`seed_policy` is `deterministic_workload_no_random_seed` on all three cases.
The four QA-001 values for each of the three cells are below. Frobenius and max-abs are
exactly 0. All three `qa001_pass` values are true. `summary.findings` is `[]`.
`summary.outside_expected_markers` is `[]`. The bundle has no `oracle_lambda_min` key.
`summary.first_failure` is null. `status` is `pass`. Every `implementation_revision` is
`55e37837`. The figures are the bundle values at three significant figures. They match
`/tmp/c4-c-proof/REPORT.md`.

| Anchor | Workload | Partitions | Frobenius | Max-abs | Trace abs | lambda_min |
|--------|----------|------------|-----------|---------|-----------|------------|
| 6 | `phase2_xxz_hea_q6_continuity` | 7 | 0 | 0 | 2.23e-16 | -3.74e-16 |
| 8 | `phase2_xxz_hea_q8_continuity` | 9 | 0 | 0 | 4.47e-16 | -7.29e-16 |
| 10 | `phase2_xxz_hea_q10_continuity` | 11 | 0 | 0 | 6.69e-16 | -6.53e-16 |

`lambda_min` is one-sided. A finding is only below `-1e-13`. There is no upper bound.
A positive `lambda_min` is neither a finding nor a marker. These three values are above
`-1e-13`, so they are not findings. The outside-expected marker covers matrix residuals
only. Frobenius and max-abs are 0, so they are not markers.

G1–G10 pass on every cell. The bundle does not store gate results. This write re-derived
each guard from the case fields: route, `planner_setting.max_partition_qubits`,
`requested_path`, `realized_path`, `exact_output_present is True`,
`actual_fused_execution is False`, `fused_region_count == 0`, `actually_fused` absent,
`partition_count` an int equal to `EXPECTED_PARTITION_COUNTS` (7, 9, 11), and the
realization key set equal to the q4 key set.

`claim_boundary` is byte-exact, in the bundle and in the mini-spec:

```text
Provisional baseline-route slice evidence for partitioned_density_descriptor_baseline at anchors 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or frozen-matrix claim.
```

## Bitwise agreement

Baseline and the oracle are separate calls. Each allocates its own `DensityMatrix`, and
neither reads the other's output back. Both lower the same canonical operations, in the
same order, through `_build_runtime_circuit` and the same C++ `NoisyCircuit` gate and
noise kernels. That is why the cells agree with the oracle bit for bit at q6, q8, and
q10, in the task-1 pattern. Frobenius and max-abs are exactly 0.

Limitation: a kernel-level bug would appear on both sides, and this oracle cannot detect
it.

## q4, fused, hybrid, strict, and historical containment

After (c), q4 differs in 2 paths, and fused, hybrid, and strict differ in 5 paths each:
the allowlisted revision paths plus `regeneration.prior_present`. All four were restored
with `git show 55e37837:<path> > <path>`. Restored sha256 values: q4
`483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`; fused
`3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0`; hybrid
`33aaa442f69e533e599e64895346786831a9591ae2a1db83dc80bf0643532ab9`; strict
`b5177cc81b6c6058039ba52310fd19f0311beb8d3247c2b9a13aeeead07edeb3`. The eight-path
historical diff was empty. The baseline file stays untracked until C2.

## Gate history

Architect Step 4a was NOT-READY at `bc-5d7c4fdb` run 3 (B1–B3), then CODE-READY on run 4
(`/tmp/c4-step4a-review/C4_STEP4A_RECLOSE.md`). Reviewer approved the code-ready docs at
`bc-d6c39d64`. Reviewer (a) APPROVE `bc-bc146637`. Tester (c) PASS at `bc-9820d3c5` run 17.

## Mutant evidence

The binding evidence is Reviewer's `mutant_matrix_main.json`
(`/tmp/rev-c4-step4b/mutant_matrix_main.json`, sha256
`59cd20f64bf6ce00c29ea483ee449533e0a2d570454158b36f75903f19fc3747`). It records 10
single-guard deletions, all killed, and 19 non-equivalent ET partials, all killed. The
two ET-equivalent partials survive as expected: G7 as `not count`, and G9 as
`== anchor + 1`. Real positives pass those deletions and partials.

The Developer's own mutant logs are void. Every run exited 4 and zero tests ran. They
are not evidence. The stale red-log header is a nit: it names `de6d20a2` while the
committed test file is `0c9b2c0f`. Reviewer reproduced red-first on the committed test,
with the pipeline still at HEAD and the module absent: 26 failed and 82 passed.

## Carry-forward (not part of C.4; each its own later commit)

- Reviewer (a) nits: unused `Path` import; G9 int and unknown-anchor clauses unpinned; row 2's workloads check is a text match; `claim_boundary` unpinned; extra blank line before `_mf1a_baseline_module`.
- The mutant harness must require pytest exit 1 and a collected count above 0. An exit of 4 with zero tests is not a kill.
- Still open, and still outside C.4: the hybrid extra-row gap, the task-4 CLOSEOUT nit, the C.3 partial-S10 fixture, `_are_ints`, and the cmake skill. This pass does not touch task-4, task-5, task-6, or task-7.

## C2 path list

C2 stages the baseline bundle
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/baseline/mf1a_baseline_bundle.json`
(sha256 `0316a303d03b096dd132d94bd6cfc7742324234cfa141c131b58b10ad8cc8976`), this
`CLOSEOUT.md`, and the checklist touch-ups from this pass. It does not stage q4, fused,
hybrid, strict, or any historical bundle. (g) is still ahead.
