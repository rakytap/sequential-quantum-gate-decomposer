# M-F1a task-9 closeout — counted 16-cell denominator

> **Status:** ready for Reviewer evidence review · **Date:** 2026-10-06 · **Work package:** task-9 ·
> **Scope:** ADR-F1A-001 16-cell counted manifest; M-F1a remains open ·
> **Claim:** counted bundle generated at clean C1 ·
> **Revision C1:** `314fee7cfd536dff79dd9efd16138f57e7ffbd9f` ·
> **C1 parent:** `a81be56baba7e97156532524978cc878d347fa87` ·
> **C2:** this commit ·
> **No push/PR**

## Summary

Counted bundle generated at clean C1. Every case has `milestone_counted` true.
`summary.milestone_counted_cases` is 16. `completeness_claim` is false.
`summary.findings` is `[]`. `summary.outside_expected_markers` is `[]`.
It does not change the oracle, QA-001 tolerances, G-07, or O-11.
`COUNTED_MANIFEST.md` is unchanged. G-03 stays open until milestone close.
Each cell's wording is "`<route>` route verified at q`<n>`".

Wall time was 183.9 s. The planning band was 105–140 s. That band is not a gate.

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| Planning re-close | done | Architect CODE-READY `bc-5d7c4fdb` run 6 |
| (a) Reviewer implementation review | done | before C1 |
| (b) C1 Tech Lead local commit | done | C1 `314fee7c`; parent `a81be56`; 8 paths |
| (c) Clean-start pipeline run | done | exit 0; 183.9 s; `/tmp/t9-c-proof/REPORT.md` |
| (d) Real `CLOSEOUT.md` | this file | ET-C9-3; uncommitted |
| (e) Reviewer evidence review | pending | after this write |
| (f) C2 Tech Lead local commit | this commit | counted bundle from (c) + this CLOSEOUT + checklist touch-ups |
| (g) Clean-C2 regeneration | pending | after C2; restore all six siblings from the C2 sha; do not commit |

## (c) Pipeline run

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `314fee7cfd536dff79dd9efd16138f57e7ffbd9f` |
| Porcelain before run | empty, after restoring `costfuncs_entropy_and_tv.txt` |
| Manifest sha | `a93de87d6b44a8a14d00d2eef73e116997faa635f35952fe6375e620aa363c16` (87 lines); worktree equals `git show HEAD:` and the §10 freeze record |
| Extension `.so` sha256 | `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |
| Wall time | 183.9 s (`real 3m3.882s`) |
| Exit code | **0** |
| Counted bundle sha256 | `a22ee685038170cb0991a9ae2195b20ad4c977b111409b312cc708fa9e6872f9` |
| Evidence | `/tmp/t9-c-proof/REPORT.md` |

`-k mf1a_counted` is 29 passed in 1.98 s. `-k mf1a` is 137 passed in 92.72 s.
`clean_start` and `provenance_pass` are true on all 16 cases. Every
`implementation_revision` is `314fee7cfd536dff79dd9efd16138f57e7ffbd9f`.
`summary.first_failure` is null. `status` is `pass`. `regeneration.pass` is true
and `prior_present` is false. The box path `/workspace/t9-counted-c-proof-314fee7c.md`
is not on this host. The numbers above are the rocky report.

Before the run, porcelain showed `costfuncs_entropy_and_tv.txt` modified. Tester
restored it from HEAD. The run then started from an empty porcelain. After the run,
q4 differed in 2 paths, fused, hybrid, and strict in 5 each, and baseline in 4:
the allowlisted revision paths plus `regeneration.prior_present`. All five passed
regeneration. The eight historical bundles were unchanged. The five siblings were
restored to q4 `483e282d…`, fused `3020ef5a…`, hybrid `33aaa442…`, strict
`b5177cc8…`, and baseline `0316a303…`. Final porcelain is only the untracked
counted bundle.

## QA-001

All 16 `qa001_pass` values are true. Figures are the proof table.

| # | Route | q | Partitions | Frobenius | Max-abs | Trace abs | lambda_min |
|---|-------|---|------------|-----------|---------|-----------|------------|
| 1 | baseline | 4 | 5 | 0 | 0 | 2.92e-17 | -8.00e-17 |
| 2 | baseline | 6 | 7 | 0 | 0 | 2.23e-16 | -3.74e-16 |
| 3 | baseline | 8 | 9 | 0 | 0 | 4.47e-16 | -7.29e-16 |
| 4 | baseline | 10 | 11 | 0 | 0 | 6.69e-16 | -6.53e-16 |
| 5 | fused | 4 | 5 | 2.97e-16 | 1.11e-16 | 2.22e-16 | -1.27e-16 |
| 6 | fused | 6 | 7 | 3.58e-16 | 6.97e-17 | 4.44e-16 | -3.13e-16 |
| 7 | fused | 8 | 24 | 9.78e-16 | 1.00e-16 | 4.47e-16 | -3.16e-16 |
| 8 | fused | 10 | 38 | 1.53e-15 | 1.39e-16 | 1.11e-15 | -5.62e-16 |
| 9 | strict | 4 | 2 | 3.13e-16 | 1.24e-16 | 2.22e-16 | 5.38e-10 |
| 10 | strict | 6 | 3 | 2.82e-16 | 8.43e-17 | 2.22e-16 | 4.49e-14 |
| 11 | strict | 8 | 4 | 3.62e-16 | 1.12e-16 | 1.12e-17 | -4.21e-19 |
| 12 | strict | 10 | 5 | 6.89e-16 | 1.11e-16 | 4.44e-16 | -4.56e-18 |
| 13 | hybrid | 4 | 5 | 3.31e-16 | 9.71e-17 | 4.45e-16 | -1.27e-16 |
| 14 | hybrid | 6 | 7 | 4.27e-16 | 8.33e-17 | 4.45e-16 | -3.58e-16 |
| 15 | hybrid | 8 | 34 | 1.20e-15 | 6.71e-17 | 8.89e-16 | 3.32e-06 |
| 16 | hybrid | 10 | 56 | 1.27e-15 | 2.26e-17 | 3.78e-17 | 8.14e-07 |

`lambda_min` is one-sided. A finding is only below `-1e-13`. The positive values
on strict q4, strict q6, hybrid q8, and hybrid q10 are neither findings nor markers.
`summary.findings` is `[]`. There is no `oracle_lambda_min` key.

## `claim_boundary`

Accepted by Research Manager, 2026-10-06. Byte-exact with `COUNTED_MANIFEST.md` §4
and with every case (437 characters):

```text
Counted M-F1a denominator evidence for the ADR-F1A-001 route-by-anchor set: partitioned_density_descriptor_baseline, partitioned_density_descriptor_fused_unitary_islands, phase31_channel_native, and phase31_channel_native_hybrid at anchors 4, 6, 8, and 10 with max_partition_qubits 2, each against execute_sequential_density_reference under QA-001. No complete M-F1a, state-vector, external-protocol, Aer, energy, or frozen-matrix claim.
```

## `completeness_claim`

Accepted value: `false`. Reason, exact:

The counted bundle is the route-by-anchor denominator only. The Linux CI job (G-04, G6), the current-state docs (G-05, G7), the milestone closeout, the full milestone review, and the Research Manager report are still open. `true` would move the claim, which returns to Research Manager.

## Disclosures

Rows 1–4, verbatim from the mini-spec §6:

**Baseline shared kernel.** Baseline and the oracle both lower through `_build_runtime_circuit` and the same C++ `NoisyCircuit` kernels (`task-1/CLOSEOUT.md:31-50`, `task-8/CLOSEOUT.md:88-97`). Bitwise agreement is expected. A kernel-level bug appears on both sides, and this oracle cannot detect it. The cells still meet the baseline rule.

Rows 9–12, including q4, verbatim from the mini-spec §6:

**Strict product of pair states, all four anchors.** q4 is two partitions on (0, 1) and (2, 3). q6, q8, and q10 add the same pair block at each (2k, 2k+1). No operation couples two pairs, so every state is a product of pair states (`task-7/CLOSEOUT.md:103`; strict bundle q4 regions at `:78-93` and `:96-111`). These cells do not test channel application on inputs correlated across a partition boundary, the order of partitions (disjoint channels commute), local depolarizing under the strict entry, or single-wire and odd-aligned motifs. C.2 hybrid covers correlated inputs and depolarizing. The strict cells still execute an eligible motif on every partition.

G-10 note for the other routes, verbatim from the mini-spec §6:

**Other routes, for the G-10 note only.** Fused shares noise and singleton lowering and does not share the island kernel; agreement is not bitwise (`task-5/CLOSEOUT.md:87-89`). Hybrid does not send noise or a singleton through the oracle's lowering; agreement is not bitwise (`task-6/CLOSEOUT.md:102-112`). Strict shares no execution kernel; agreement is not bitwise (`task-7/CLOSEOUT.md:120-132`).

## C2 path list

C2 stages the counted bundle
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/counted/mf1a_counted_bundle.json`
(sha256 `a22ee685038170cb0991a9ae2195b20ad4c977b111409b312cc708fa9e6872f9`), this
`CLOSEOUT.md`, and the checklist touch-ups from this pass. It does not stage q4,
fused, hybrid, strict, baseline, or `COUNTED_MANIFEST.md`. The §10 freeze-record
bytes stay unchanged. (g) is still ahead.
