# M-F5a slice 4 closeout — width-4 attribution routes
> **Status:** shipped · **Date:** 2026-10-08 · **Work package:** task-4 ·
> **Scope:** width-4 three-route slice only. M-F5a is not complete ·
> **C1:** `6be6282f` (Q2, B1–B3; Reviewer `a0646acb…`) ·
> **C1-ET4:** `431a5808` (refusal row and counted mode; Reviewer `19c16b71…`) ·
> **C2:** `5f63a9d6` (bundle only; Reviewer `7b48bbc6…`) · (g) PASS ·
> **RM:** `2026-10-08-mf5a-task4-w4-routes-interpret.md` (`788f688d…`) · attribution evidence only ·
> **No push/PR**

## Summary

This closeout records the counted width-4 attribution bundle. C1 `6be6282f` holds the apply-primitive wrap, the hybrid label, and the B3 negatives. C1-ET4 `431a5808` adds the R-strict refusal row and `--attribution-routes` counted mode. C2 `5f63a9d6` commits the bundle and no closeout. This file is the closeout. It does not re-run the counted command.

The Research Manager accepts the bundle as attribution evidence at width 4. That acceptance is not a milestone claim. REQ-004 stays open. CAP-004 stays hold-the-line. `milestone_counted` stays false. No row carries an overhead ratio. R-oracle stays the E1 exclude. R-strict is a `handback_refused` row and carries no timings.

## Verdict

Counted width-4 routes **recorded**. The bundle is attribution evidence only. `milestone_counted=false` is **lawful**. REQ-004 stays **open**. The milestone is **not** complete. Task-5 (widths 6 and 8) is planned in `task-5/`; this file does not make it code-ready.

The Research Manager's quotable sentence, copied verbatim from `2026-10-08-mf5a-task4-w4-routes-interpret.md` (`788f688d…`), is the only route-result wording this file carries:

> On the frozen 4-qubit E-VQE anchor workload (single core, AMD EPYC 7542), all timed attribution routes reproduce the sequential reference to ≤1e-16. Per call, today's fused route takes about 2.5× and the hybrid route about 9× the time of the unfused baseline route. These ratios reflect current implementations (a generic C++ local-unitary routine; Kraus-matrix construction in Python), not intrinsic costs of fusion or channel-native semantics. The strict route refuses under the frozen noise placement, as its contract requires, and is reported without timings.

**Correction 2026-10-08 (forward only; RM 2026-10-08).** The quotation above stays as committed. The Research Manager withdraws the exactness bound in that quotation. The replacement, verbatim, is that the timed routes "agree with the step-by-step (sequential) reference to ≤1.2e-16 and with an independent simulator (Qiskit Aer 0.17.2, given the same circuit and noise specification) to ≤5e-16, i.e. at double-precision machine precision." Both bounds rest on width-4 data here: the sequential bound on the C2 binder probe (`7b48bbc6…` §6.1) and the Aer bound on `/tmp/mf5a-t4-g/REPORT.md` (`6771415c…`). Widths 6 and 8 are recorded in `task-5/CLOSEOUT.md`.

## `milestone_counted=false`

Lawful. The width-4 contract requires the flag false. The row is not a milestone verdict and it does not close REQ-004.

## Chain

| Role | SHA |
|------|-----|
| C0 stamp | `226ffc29` |
| C1 | `6be6282f` |
| Docs before C1-ET4 | `d52f198f` |
| C1-ET4 | `431a5808eac6c6a70a62fb0e1fbcb0191151373a` |
| C2 | `5f63a9d60eba7ab59176addfa0ef071e4865c74a` |

`provenance.implementation_revision` on the bundle is the C1-ET4 sha. Reviewer C1-ET4 binder `/tmp/rev-mf5a-t4-et4-regate3/REVIEW.md` (`19c16b71…`). Reviewer C2 binder `/tmp/rev-mf5a-t4-c2-w4/REVIEW.md` (`7b48bbc6…`). Tester evidence `/tmp/mf5a-t4-counted-w4/`.

## Evidence pins

| Field | Value |
|-------|--------|
| Bundle | `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json` |
| sha256 | `6584be2bc49d55c995c54d1be9a7b7efb2b2dfe4cdd0004fd030b5596f9196aa` |
| Suite | `interop_attribution_routes_task4_w4_v1` |
| `milestone_counted` | false |
| R-strict | `handback_refused`; no timings |
| Handback | `STEP_4A_HANDBACK.md` `98eec857…` unchanged |
| Validator | `validate_attribution_route_bundle` OK (`/tmp/mf5a-t4-counted-w4/validator.txt`) |
| `clean_start` / `provenance_pass` / `dirty_paths` | true / true / `[]` |
| `implementation_revision` | `431a5808eac6c6a70a62fb0e1fbcb0191151373a` (C1-ET4) |
| Protocol | 50 warm-up, then 1000 counted calls per timed route; divisor 3072; `affinity_cpu` 0; four thread values `"1"` |
| Extensions | `libqgd.so` `615610c8…`; VQE wrapper `.so` `c9ed35df…`; compiler `c++ (GCC) 11.5.0`; flags not pinned (N-46) |
| Tester (c) | `/tmp/mf5a-t4-counted-w4/REPORT.md` (`a9b14af852fbbd67f72e388f9c941ab72443743b069bbbef333e82a099084c53`): exit 0, run once, at C1-ET4 from an empty porcelain |

## Acceptance verdicts (vs slice contract)

ET checkboxes stay unchecked, as in tasks 1–3. Lane results are the Reviewer binders'; this docs pass did not rerun them.

| Signal | Verdict | Evidence |
|--------|---------|----------|
| ET-1 route schema, no overhead ratio | pass | C1-ET4 binder `19c16b71…` §5: `test_vqe_interop_bundle_validation.py` 118 passed. C2 binder `7b48bbc6…` §8: the MS §4 pytest rows give 136 passed at the simulated C2 |
| ET-2 width-4 anchor | pass | C2 binder §6.3: 12 operations in 5 partitions; every route executes all 12, so divisor 3072 holds on each |
| ET-3 diff inside attribution | pass | C1-ET4 binder §5: zero diff outside the five paths, `squander/**` included. C2 binder §8: the REQ-005 and REQ-006 rows exit 0 |
| ET-4 refusal row and counted flag | pass | C1-ET4 binder §5: `test_vqe_interop_harness.py` 18 passed. Tester (c) REPORT criteria 1–7 PASS |
| REQ-007, QA-009 | pass | `test_explicit_state_vector_matches_legacy_default` 1 passed (C1-ET4 binder §5; C2 binder §8) |
| Counted bundle | recorded | evidence pins above; `milestone_counted=false` |
| REQ-009 lint | pass at this write | both `specs_check.sh` lines below exit 0; the only finding is task-5's step-4a absent closeout |
| REQ-004 | open | width-4 rows do not close it |
| Milestone | not complete | widths 6 and 8 (task-5), G-08, and G-09 stay later |

## Independence

The task-4 acceptance has no oracle check: the counted bundle carries timings and a refusal row, and no exactness field. No Tester independence note was written, because the counted run had no oracle/cell pair to certify. The exactness clause in the Research Manager's sentence rests on the C2 Reviewer probe (`7b48bbc6…` §6.1), which compared each route's final ρ with a separate `execute_sequential_density_reference` call. R-base and that reference share the `NoisyCircuit` kernels, so their agreement cannot detect a kernel-level bug (G-10). R-fused also runs its unfused U3 and its three noise operations through `NoisyCircuit.apply_to` (`7b48bbc6…` §6.3), so it shares those kernels with the reference too; only the fused-island `apply_local_unitary` kernel and R-hybrid's numpy Kraus path are separate apply code. A later Tester check, not the counted bundle, compares each timed route with Qiskit Aer 0.17.2 at tolerance 1e-10. Report `/tmp/mf5a-t4-g/REPORT.md` (`6771415c…`) records max |Δρ| versus Aer: R-base 4.72e-16, R-fused 4.44e-16, R-hybrid 4.72e-16. R-strict refused and has no state.

## (g) clean-C2 regeneration

PASS. Tester report `/tmp/mf5a-t4-g/REPORT.md` (`6771415c3a3953fde69b1dbb4b0bd5fcb8b4f485156812026f308dc94b998a60`) is a clean clone at C2 `5f63a9d6` with empty porcelain. `test_vqe_interop_bundle_validation.py` 118 passed. `test_vqe_interop_harness.py` 18 passed. The reproduced bundle is `ab4d7fe5e9d8a5cd0dd89c0e858a9854d6daa2eb5892419a1614d66fceb13fe4`. `validate_attribution_route_bundle` on that file is OK. Categorical fields match the committed bundle `6584be2b…`. Timings differ, as a regeneration may. The reproduced `implementation_revision` is C2. The clone built its own extensions, so the two `provenance.extension_identities` sha256 values also differ (`libqgd.so` `86a635fa…`, VQE wrapper `.so` `4447be6d…`). Over all 9081 bundle leaves, only timing leaves, `implementation_revision`, and those two sha256 values differ (Reviewer leaf diff `/tmp/rev-mf5a-t5-codeready/logs/g_bundle_leafdiff.log`, `24966fc0…`). Outputs were not committed. The C2 binder's five repeats (`7b48bbc6…` §7) ran at C1-ET4 and are not this (g). Throughput is not a (g) gate and has no numeric margin (mini-spec §3a).

## Reproduce

The counted command was run for C2. This closeout does not re-run it.

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
unset PYTHONPATH
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 4 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
```

## What this closeout does not say

No method-level speedup. No per-route nanosecond kernel benchmark. No comparison of a route throughput with the E-VQE apply throughput. No generalisation beyond width 4. No orchestration-versus-apply share. No "four-route shipped". No "REQ-004 met". No "M-F5a complete". No "reduction shipped". No VQA. 17/9/0 stays. M-F1b stays closed. No push and no pull request.

## Carries

- C2 N-5 (clock granularity and instrumentation floor, measured in `7b48bbc6…` §6.2 and not restated here) and N-46 (compiler flags not pinned).
- C2 N-7 (host noise; observational).
- ET-4 re-gate 3 N-1 (optional K-c: a `match=` per R-strict guard) and N-4 (run pytest with `-p no:cacheprovider`); optional K-a, K-b, K-d, K-e, and K-f.
- N-7 re-fix N-5, N-8 through N-11, and N-18.
- (g) is PASS, recorded above. G-08, G-09, N8, and Demo No GO. M-F1b stays unopened.
- `STEP_4A_HANDBACK.md` :33 keeps its historical figures, uncited.
- Task-5 W-1…W-5 and N-1…N-7 come from binder `9b8c748f…` §10–§11, and N-a and N-b from `aac125e8…` §10. They are folded into the task-5 docs.

## Next

Task-5's readiness is recorded in `task-5/ENGINEERING_TASKS.md`. This file does not stamp task-5 or authorize its Step 4b. Widths 6 and 8 are not counted here.
