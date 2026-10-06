# M-F1a slice C.3 closeout — strict route provisional evidence (task-7)

> **Status:** ready for Reviewer evidence review · **Date:** 2026-10-06 · **Work package:** task-7 ·
> **Scope:** strict route at anchors 4, 6, 8, and 10; M-F1a remains open ·
> **Claim:** strict bundle generated at clean C1 ·
> **Revision C1:** `a0ac4cbd0044ba97fd136a73cb6739c168270426` ·
> **C1 parent:** `42922382aaf1bdf5d609d169110b75cd72c9a757` ·
> **C2:** this commit ·
> **No push/PR**

## Summary

Strict bundle generated at clean C1. This closeout records the post-C1 clean-start run
(step (c)). Records stay provisional: every case has `milestone_counted` false and
`summary.milestone_counted_cases` is 0. It does not change the oracle, QA-001
tolerances, the counted denominator, scope, or the G-07 exclusion set. It does not
authorize C.4. q4 remains baseline route verified. G-03 stays open.

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| Planning re-close | done | Architect CODE-READY `/tmp/c3-step4a/C3_STEP4A_RECLOSE.md` |
| (a) Reviewer implementation review | done | NOT-READY `bc-340153bd`; APPROVE `bc-101ced81` |
| (b) C1 Tech Lead local commit | done | C1 `a0ac4cbd`; parent `42922382`; 7 paths |
| (c) Clean-start pipeline run | done | exit 0; 80 s; Tester `bc-9820d3c5` run 15 |
| (d) Real `CLOSEOUT.md` | this file | ET-C3-3; uncommitted |
| (e) Reviewer evidence review | pending | after this write |
| (f) C2 Tech Lead local commit | this commit | strict bundle from (c) + this CLOSEOUT + checklist touch-ups |
| (g) Clean-C2 regeneration | pending | after C2; restore q4, fused, hybrid, and strict from the C2 sha; do not commit |

## (c) Pipeline run

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `a0ac4cbd0044ba97fd136a73cb6739c168270426` |
| Porcelain before run | empty |
| Extension `.so` sha256 | `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |
| Wall time | 80 s |
| Exit code | **0** |
| Strict bundle sha256 | `b5177cc81b6c6058039ba52310fd19f0311beb8d3247c2b9a13aeeead07edeb3` |
| Evidence | Tester `bc-9820d3c5` run 15 · `/tmp/c3-c-proof/REPORT.md` |

Developer gate before (c): `-k mf1a` 84 passed; collect 25 (`mf1a_strict`) and 84 (`mf1a`).
Tester, same tree: `-k mf1a_strict` 25 passed; the evidence file with no `-k` collects 103 and
103 passed. Stdout: q4, fused, hybrid, and strict `pass | written`; eight historical
`pass | verified, not written`.

## QA-001

Route on every row: `phase31_channel_native`. `max_partition_qubits` is 2.
`seed_policy` is `deterministic_workload_no_random_seed` on all four cases.
All four `qa001_pass` values are true. Max Frobenius is `6.889e-16`. Max max-abs is
`1.241e-16`. Max absolute trace deviation is `4.441e-16`. Minimum `lambda_min` is
`-4.562e-18`. `summary.findings` is `[]`. `summary.outside_expected_markers` is `[]`.
The bundle has no `oracle_lambda_min` key. `summary.first_failure` is null.
`status` is `pass`. Every `implementation_revision` is `a0ac4cbd`.
The figures are the bundle values at three significant figures. They match
`/tmp/c3-c-proof/REPORT.md`.

| Anchor | Workload | Channel-native / partitions | Frobenius | Max-abs | Trace abs | lambda_min |
|--------|----------|-----------------------------|-----------|---------|-----------|------------|
| 4 | `phase31_local_support_q4_spectator_embedding_smoke` | 2/2 | 3.131e-16 | 1.241e-16 | 2.221e-16 | 5.377e-10 |
| 6 | `mf1a_strict_spectator_embed_q6` | 3/3 | 2.821e-16 | 8.431e-17 | 2.224e-16 | 4.495e-14 |
| 8 | `mf1a_strict_spectator_embed_q8` | 4/4 | 3.615e-16 | 1.119e-16 | 1.123e-17 | -4.209e-19 |
| 10 | `mf1a_strict_spectator_embed_q10` | 5/5 | 6.889e-16 | 1.112e-16 | 4.441e-16 | -4.562e-18 |

Every witness reason is `channel_native_motif_kraus_count_4`. Kraus count is 4 per witness.
Partition rows are `{"partition_index"}` only. S1–S11 pass on every cell. The bundle does
not store gate results. This write re-derived each guard from the case fields: route,
`planner_setting.max_partition_qubits`, `requested_path`, `realized_path`,
`exact_output_present`, `partition_count`, partition-row key sets, sorted partition
indices, region kind, classification, and reason, sorted region indices, and
`channel_native_partition_count`.

`lambda_min` is one-sided. It is a finding only below `-1e-13`. The QA-001 floor is
`-1e-12`. There is no upper bound. The outside-expected marker covers matrix residuals
only (Frobenius, max-abs, and absolute trace deviation). The minimum here,
`-4.562e-18`, is above `-1e-13`. q4 `lambda_min` is `+5.377e-10` and q6 `lambda_min` is
`+4.495e-14`. They are not findings under the one-sided rule (a finding only below
`-1e-13`; no upper bound; per the C.2 (e) ruling, Reviewer `bc-bac6d17d`). The bundle
carries no oracle `lambda_min` to attribute them. Empty `findings` and empty markers
follow that rule.

`claim_boundary` is byte-exact:

```text
Provisional strict-route slice evidence for phase31_channel_native at anchors 4, 6, 8, and 10 with max_partition_qubits 2; not the frozen M-F1a milestone denominator. No complete M-F1a, external-protocol, Aer, energy, or frozen-matrix claim.
```

## Product-state limitation

The four cells evidence per-pair channel-native motif realization and the strict route's
density-matrix agreement with the sequential oracle at widths 4, 6, 8, and 10 under
`max_partition_qubits` 2. They do not evidence entangled cross-pair states,
cross-partition noise propagation, or a general-circuit claim.

> The q6, q8, and q10 workloads are an M-F1a-only family built in `mf1a_strict_validation.py`, not by a Phase-3 or Phase-3.1 builder; no historical builder is strict-eligible at those widths at `max_partition_qubits` 2 (task-4 inventory). Each pair (2k, 2k+1) carries the q4 smoke's block and no operation couples two pairs, so every state in these runs is a product of pair states. Against the sequential oracle, the cells test strict preflight, Kraus composition of the six-member block, the local-to-global remap at every pair offset, and full-space embedding next to spectators that hold either earlier mixed states or |0⟩. They do not test channel application on inputs correlated across a partition boundary, the order of partitions (disjoint channels commute), local depolarizing under the strict entry, or single-wire and odd-aligned motifs. C.2 hybrid exercises correlated inputs and depolarizing on the same channel-native kernel. Parameters come from `build_initial_parameters`; no seed is drawn.

## q4, fused, hybrid, and historical containment

The (c) q4 diff against `git show a0ac4cbd:<q4>` is two paths:
`cases[0].provenance.implementation_revision` (`a50ae79f…` to `a0ac4cbd…`) and
`regeneration.prior_present` (false to true). The fused diff is five paths: the four
`cases[i].provenance.implementation_revision` values (`7cf11a49…` to `a0ac4cbd…`) and
`regeneration.prior_present` (false to true). The hybrid diff is five paths: the four
revision values (`b95400d5…` to `a0ac4cbd…`) and `regeneration.prior_present`
(false to true). All three were restored with `git show a0ac4cbd:<path> > <path>`.
Restored sha256 values:
q4 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`;
fused `3020ef5a92ea7a4bf4ab5dc76cf961e9dfdaff05896981d6ca9b04cd2af46cf0`;
hybrid `33aaa442f69e533e599e64895346786831a9591ae2a1db83dc80bf0643532ab9`.
The eight-path historical diff was empty. The strict file stays untracked until C2.

## Pre-(d) independence

Strict and the oracle share no execution kernel at these four cells. Every strict
partition runs `execute_partition_channel_native`
(`noisy_runtime_channel_native.py:903-982`): numpy Kraus bundles for U3, CNOT,
amplitude damping, and phase damping; composition; completeness and Choi checks; then
each 4×4 operator embedded on global (2k, 2k+1). The oracle applies every gate and
channel through `_build_runtime_circuit` and C++ `NoisyCircuit.apply_to`
(`noisy_runtime_core.py:1011-1057`). A non-counted Architect probe at `42922382`
counted zero strict-side calls to `_execute_member_sequence`, `_build_fused_kernel`,
and `NoisyCircuit.apply_to` at q4–q10. `_build_runtime_circuit` ran once per partition,
for alignment only. Agreement is not bitwise. This closeout does not copy the q4
bitwise paragraph.

## Gate history

Reviewer (a) was NOT-READY at `bc-340153bd` with three blockers: S10 masking, test 11's
QA-001 identity, and test 23's circular paths. The fix is in C1. Reviewer (a) then
APPROVE `bc-101ced81`. Red-first, with the strict module hidden and the pipeline still
at HEAD: `-k mf1a` 27 failed and 57 passed. After the module and the index-3 entry:
`-k mf1a` 84 passed; collect 25 and 84. Thirteen mutants were killed, including a
single-guard deletion of S8 and of S10 (`/tmp/c3-step4b-fix-REPORT.md`).

## Carry-forward (not part of C.3; each its own later commit)

- A partial S10 mutant (length-only, or length plus range) survives. Add a planner fixture with regions `[0, 0]` in ET row 7.
- `_are_ints` is unpinned.
- Reviewer nits, unpinned: cell dispatch; oracle self-compare; channel-native count via descriptor indices; the bool index edge.
- Still open, and still outside C.3: the hybrid `S_len_range` extra-row gap, and the task-4 CLOSEOUT "Commit: not created yet" nit. This pass does not touch `task-4/CLOSEOUT.md`, `task-5/CLOSEOUT.md`, or `task-6/CLOSEOUT.md`.

## C2 path list

C2 stages the strict bundle
`benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/strict/mf1a_strict_bundle.json`
(sha256 `b5177cc81b6c6058039ba52310fd19f0311beb8d3247c2b9a13aeeead07edeb3`), this
`CLOSEOUT.md`, and the checklist touch-ups from this pass. It does not stage q4, fused,
hybrid, or any historical bundle. (g) is still ahead.
