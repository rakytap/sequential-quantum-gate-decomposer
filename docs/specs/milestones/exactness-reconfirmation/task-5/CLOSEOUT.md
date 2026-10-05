# M-F1a slice C.1 closeout — fused route provisional evidence (task-5)

> **Status:** ready for Reviewer evidence review · **Date:** 2026-10-05 · **Work package:** task-5 ·
> **Scope:** fused route at anchors 4, 6, 8, and 10; M-F1a remains open ·
> **Claim:** fused bundle generated at clean C1 ·
> **Revision C1:** `7cf11a49b9a89b8793f9afdbe78e1f66ef3c05e7` ·
> **C1 parent:** `91680ec720bc311f28e09021895f0c78900d773b` ·
> **C2:** pending ·
> **No push/PR**

## Summary

Fused bundle generated at clean C1. This closeout records the post-C1 clean-start run
(step (c)). Records stay provisional: every case has `milestone_counted` false and
`summary.milestone_counted_cases` is 0. It does not change the oracle, QA-001
tolerances, the counted denominator, scope, or the G-07 exclusion set. It does not
authorize C.2, C.3, or C.4. q4 remains baseline route verified.

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| (a) Reviewer implementation review | done | before C1 |
| (b) C1 Tech Lead local commit | done | C1 `7cf11a49`; parent `91680ec7`; 7 paths |
| (c) Clean-start pipeline run | done | exit 0; 51 s; see below |
| (d) Real `CLOSEOUT.md` | this file | ET-C1-3; uncommitted |
| (e) Reviewer evidence review | pending | after this write |
| (f) C2 Tech Lead local commit | pending | fused bundle from (c) + this CLOSEOUT + checklist touch-ups |
| (g) Clean-C2 regeneration | pending | after C2; restore both siblings from the C2 sha; do not commit |

## (c) Pipeline run

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `7cf11a49b9a89b8793f9afdbe78e1f66ef3c05e7` |
| Porcelain before run | empty |
| Extension `.so` sha256 | `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |
| Wall time | 51 s |
| Exit code | **0** |
| Fused bundle sha256 | `3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0` |
| Evidence | Tester `bc-9820d3c5` · `/tmp/c1-c-proof/REPORT.md` |

Stdout: q4 `pass | written`; fused `pass | written`; eight historical
`pass | verified, not written`.

## QA-001

Route on every row: `partitioned_density_descriptor_fused_unitary_islands`. All four
`qa001_pass` values are true. `summary.findings` is `[]`. The bundle has no
`oracle_lambda_min` key. `summary.first_failure` is null. `status` is `pass`.

| Anchor | Frobenius | Max-abs | Trace abs | lambda_min |
|--------|-----------|---------|-----------|------------|
| 4 | 2.969e-16 | 1.111e-16 | 2.224e-16 | -1.266e-16 |
| 6 | 3.578e-16 | 6.974e-17 | 4.443e-16 | -3.127e-16 |
| 8 | 9.781e-16 | 1.001e-16 | 4.474e-16 | -3.164e-16 |
| 10 | 1.528e-15 | 1.389e-16 | 1.110e-15 | -5.625e-16 |

No row is above the `1e-11` matrix-residual finding line or below the `-1e-13`
`lambda_min` finding line. No matrix-residual finding is flagged to Research Manager
from this run.

## Island composition

`actually_fused` region counts are 4, 6, 12, and 20 at q4, q6, q8, and q10.
CNOT-in-kernel fusion (`CNOT` inside an `actually_fused` region's
`operation_names`) is only at q4 (3 regions) and q6 (5 regions). q8 and q10 have
none.

## q4 and historical containment

The (c) q4 diff against `git show 7cf11a49:<q4>` is two paths:
`cases[0].provenance.implementation_revision` (`a50ae79f…` to `7cf11a49…`) and
`regeneration.prior_present` (false to true). q4 was restored with
`git show 7cf11a49:<q4> > <q4>` to sha256
`483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`. The eight-path
historical diff was empty. Those eight bytes were not restored because they had not
changed. The fused file stays untracked until C2.

## Pre-(d) note

Fused islands and the sequential oracle do not share the island kernel. Noise and
singleton unitaries on the fused route use the same NoisyCircuit lowering as the
oracle. Agreement is not bitwise. Phase-3 Frobenius figures are non-counted context
only and are not this run's table.

## C2 path list

C2 stages the fused bundle
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/fused/mf1a_fused_bundle.json`
(sha256 `3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0`), this
`CLOSEOUT.md`, and the checklist touch-ups from this pass. It does not stage q4 or
any historical bundle. (g) is still ahead.
