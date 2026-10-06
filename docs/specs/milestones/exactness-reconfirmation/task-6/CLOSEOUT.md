# M-F1a slice C.2 closeout — hybrid route provisional evidence (task-6)

> **Status:** ready for Reviewer evidence review · **Date:** 2026-10-06 · **Work package:** task-6 ·
> **Scope:** hybrid route at anchors 4, 6, 8, and 10; M-F1a remains open ·
> **Claim:** hybrid bundle generated at clean C1 ·
> **Revision C1:** `b95400d54ea5cc31d81a26f5cd43ae3be81e12c4` ·
> **C1 parent:** `a006c7e27b9e9f2e400d4379a7890600270b5d54` ·
> **C2:** pending ·
> **No push/PR**

## Summary

Hybrid bundle generated at clean C1. This closeout records the post-C1 clean-start run
(step (c)). Records stay provisional: every case has `milestone_counted` false and
`summary.milestone_counted_cases` is 0. It does not change the oracle, QA-001
tolerances, the counted denominator, scope, or the G-07 exclusion set. It does not
authorize C.3 or C.4. q4 remains baseline route verified. G-03 stays open.

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| Planning re-close | done | Architect `bc-285a88eb` |
| (a) Reviewer implementation review | done | code-ready writer APPROVE `bc-53aa89e5`; Step 4b NOT-READY `bc-3dad96e3`; re-gate APPROVE `bc-c5e1adf0` |
| (b) C1 Tech Lead local commit | done | C1 `b95400d5`; parent `a006c7e2`; 7 paths |
| (c) Clean-start pipeline run | done | exit 0; 78 s; Tester `bc-9820d3c5` |
| (d) Real `CLOSEOUT.md` | this file | ET-C2-3; uncommitted |
| (e) Reviewer evidence review | pending | after this write |
| (f) C2 Tech Lead local commit | pending | hybrid bundle from (c) + this CLOSEOUT + checklist touch-ups |
| (g) Clean-C2 regeneration | pending | after C2; restore q4, fused, and hybrid from the C2 sha; do not commit |

## (c) Pipeline run

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `b95400d54ea5cc31d81a26f5cd43ae3be81e12c4` |
| Porcelain before run | empty |
| Extension `.so` sha256 | `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |
| Wall time | 78 s |
| Exit code | **0** |
| Hybrid bundle sha256 | `33aaa442f69e533e599e64895346786831a9591ae2a1db83dc80bf0643532ab9` |
| Evidence | Tester `bc-9820d3c5` · `/tmp/c2-c-proof/REPORT.md` |

Developer gate before (c): `-k mf1a` 59 passed, 0 failed; collect 18 (`mf1a_hybrid`) and 59 (`mf1a`).
Stdout: q4, fused, and hybrid `pass | written`; eight historical `pass | verified, not written`.

## QA-001

Route on every row: `phase31_channel_native_hybrid`. `max_partition_qubits` is 2.
All four `qa001_pass` values are true. Max Frobenius is `1.267e-15`. `summary.findings`
is `[]`. The bundle has no `oracle_lambda_min` key. `summary.first_failure` is null.
`status` is `pass`. Every `implementation_revision` is `b95400d5`.

| Anchor | Channel-native / partitions | Frobenius | Max-abs | Trace abs | lambda_min |
|--------|-----------------------------|-----------|---------|-----------|------------|
| 4 | 2/5 | 3.305e-16 | 9.715e-17 | 4.448e-16 | -1.269e-16 |
| 6 | 2/7 | 4.272e-16 | 8.334e-17 | 4.450e-16 | -3.576e-16 |
| 8 | 11/34 | 1.195e-15 | 6.712e-17 | 8.887e-16 | 3.324e-06 |
| 10 | 18/56 | 1.267e-15 | 2.255e-17 | 3.785e-17 | 8.138e-07 |

`claim_boundary` is byte-exact:

```text
Provisional hybrid-route slice evidence for phase31_channel_native_hybrid at anchors 4, 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or frozen-matrix claim.
```

## Observation for Reviewer (e)

Not a finding. `lambda_min` is positive at q8 (`3.324e-06`) and q10 (`8.138e-07`), against
about `-1e-16` at q4 and q6. Tester reads the positive values as the genuine mixed-state
minimum eigenvalue under channel-native noise. Mini-spec §3.2 quotes Layer 1 §10:
matrix residual above `1e-11` is a finding; `lambda_min` below `-1e-13` is a finding.
The outside-expected marker applies only to matrix residuals (Frobenius, max-abs,
|Tr−1|). `lambda_min` is one-sided: a finding below `-1e-13`, QA-001 floor `-1e-12`,
no upper bound. So the module correctly emits `findings` `[]` and markers `[]`.
Positive `lambda_min` at q8 and q10 is expected physics: the dense
`phase31_pair_repeat` workload gives a full-rank ρ, and the oracle `lambda_min`
matches the hybrid to about `1e-18`. q4 and q6 are rank-deficient, so they sit near
`-1e-16`. C.1 fused used sparse q8/q10 workloads, which gives negative `lambda_min`.
The route is not the cause.

## q4, fused, and historical containment

The (c) q4 diff against `git show b95400d5:<q4>` is two paths:
`cases[0].provenance.implementation_revision` (`a50ae79f…` to `b95400d5…`) and
`regeneration.prior_present` (false to true). The fused diff is five paths: the four
`cases[i].provenance.implementation_revision` values (`7cf11a49…` to `b95400d5…`) and
`regeneration.prior_present` (false to true). Both were restored with
`git show b95400d5:<path> > <path>`. Restored sha256 values:
q4 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`;
fused `3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0`.
The eight-path historical diff was empty. The fused module was not edited.
The hybrid file stays untracked until C2.

## Pre-(d) independence

Hybrid and oracle share no execution kernel at these four cells. Channel-native
partitions use `execute_partition_channel_native` (`noisy_runtime_channel_native.py:903-982`):
numpy Kraus bundles, composition, completeness and Choi checks, then K ρ K† on
`rho.to_numpy()`. Pure-unitary partitions use `_build_fused_kernel`
(`noisy_runtime_fusion.py:200-261`) and `DensityMatrix.apply_local_unitary`. The oracle
applies every gate and channel through `_build_runtime_circuit` and C++
`NoisyCircuit.apply_to` (`noisy_runtime_core.py:1011-1057`). A non-counted Architect
probe found zero hybrid-side calls to `_execute_member_sequence` at q4, q6, q8, and q10.
Unlike C.1, no noise channel or singleton unitary goes through the oracle's lowering.
Shared pieces are request validation, parameter routing, and `_build_runtime_circuit`
built only for alignment and never applied. Agreement is not bitwise.

## Reviewer nits (non-blocking; not edited here)

From re-gate `bc-c5e1adf0` (`/tmp/c2-regate-a/REVIEW.md`), for the record:

- The red log was mislabelled. With the hybrid module truly absent, the run fails as a collection error. "18 failed" reproduces only with the HEAD pipeline (`c5e2117e`), not with pipeline `d75e949d`.
- The mutant matrix should record the split mutants `M_d` (row-count guard; dropped-row kill) and `M_e` (duplicate-index guard). The single length, set-size, or sorted-range deletions are equivalent mutants: no test can kill them, and the re-gate closed that split on substance.
- Extra-row test gap: mutant `S_len_range` (removes both the length and the sorted-range checks) survives 59/59. It is not equivalent: it accepts an extra duplicated row. Carry-forward fix: an extra-row fixture.
- Optional: parametrize the realization test.
- Mixed-type partition indices raise `TypeError` and fail closed. Runtime indices are ints.
- Residual survivors from `bc-3dad96e3` still survive and were not re-litigated. The most ADR-F1A-003-relevant name recorded there is `rr_drop_cn_motif_actually_fused`.

Mini-spec §3.4 still has the earlier "Between C1 and (d)" sentence and then the three-window paragraph that restates it. Recorded only. This pass does not edit the mini-spec.

The task-4 CLOSEOUT "Commit: not created yet" nit stays out of this slice. It belongs in its own docs commit. This pass does not touch `task-4/CLOSEOUT.md`.

## C2 path list

C2 stages the hybrid bundle
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/hybrid/mf1a_hybrid_bundle.json`
(sha256 `33aaa442f69e533e599e64895346786831a9591ae2a1db83dc80bf0643532ab9`), this
`CLOSEOUT.md`, and the checklist touch-ups from this pass. It does not stage q4, fused,
or any historical bundle. (g) is still ahead.
