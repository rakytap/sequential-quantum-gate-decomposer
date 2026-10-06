# EXACTNESS_RECONFIRMATION Closeout — `exactness-reconfirmation` (counted 16-cell draft)

> **Status:** draft ready for Reviewer Opus milestone review · **Date:** 2026-10-06 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **completeness_claim:** false · milestone remains open ·
> **HEAD / task-9 C2:** `0a66f9747f61e153a9aea6484310443d63643f08` ·
> **Task-9 C1:** `314fee7cfd536dff79dd9efd16138f57e7ffbd9f` ·
> **No push/PR**

This draft is the input to the full milestone review. It does not close the milestone and it does not interpret the counted result for Research Manager.

## 1. Summary

The frozen 16-cell denominator is recorded. All 16 counted cases pass QA-001. `summary.findings` is `[]`. `summary.milestone_counted_cases` is 16. `completeness_claim` stays false. Provisional sibling evidence through C.4 stays historical. G-04, G-05, the Opus review, the Research Manager report, and the O-10 push are still ahead.

## 2. Slices delivered

| Slice | C2 | What it records |
|-------|----|-----------------|
| task-1 q4 tracer | `a2928bf1` | baseline route verified at q4 |
| Slice A | `2a2f8c137` | historical suites verify-only under the M-F1a command (ADR-F1A-011); historical suites verified, not written |
| Slice B | `0e8299e9` | q4 baseline regenerated at clean C2 |
| C.0 task-4 | `91680ec7` | advertised-route inventory; not a C1/C2 close |
| C.1 task-5 fused | `a006c7e2` | provisional fused 4/6/8/10 |
| C.2 task-6 hybrid | `42922382` | provisional hybrid 4/6/8/10; route labels and witnesses |
| C.3 task-7 strict | `aaf6fcfe` | provisional strict 4/6/8/10 |
| C.4 task-8 baseline | `85f01985` | provisional baseline 6/8/10 |
| task-9 counted | `0a66f974` | counted 16-cell bundle; C1 `314fee7c` |

Task-9 (g) PASS. Regeneration at clean C2 differs from the committed counted bundle in exactly 17 paths: the 16 allowlisted `provenance.implementation_revision` values (`314fee7c` → `0a66f974`) and the derived `regeneration.prior_present` (false → true), which ADR-F1A-009 Amendment 1 recomputes rather than compares. Every other field matches. Each of the five provisional siblings differs only on its allowlisted revision paths plus `prior_present`. All six M-F1a bundles were restored to their C2 bytes. Final porcelain was empty. Evidence: `/tmp/t9-g-proof/REPORT.md`, Tester `bc-9820d3c5` (upload `t9-counted-g-proof-0a66f974`). Wall `real 1m55.897s`. Not a gate.

## 3. Counted evidence

Bundle `benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/counted/mf1a_counted_bundle.json`, sha256 `a22ee685038170cb0991a9ae2195b20ad4c977b111409b312cc708fa9e6872f9`. Sixteen cells, four routes at anchors 4, 6, 8, and 10. Partitions: baseline 5/7/9/11, fused 5/7/24/38, strict 2/3/4/5, hybrid 5/7/34/56 (`task-9/CLOSEOUT.md`). Tests at (c): 29/29 `mf1a_counted` (1.98 s); 137/137 `mf1a` (92.72 s). (c) pipeline 183.9 s, above the 105–140 s planning band. Not a gate. Each cell's wording is "`<route>` route verified at q`<n>`".

`claim_boundary`, verbatim from `task-9/CLOSEOUT.md`:

```text
Counted M-F1a denominator evidence for the ADR-F1A-001 route-by-anchor set: partitioned_density_descriptor_baseline, partitioned_density_descriptor_fused_unitary_islands, phase31_channel_native, and phase31_channel_native_hybrid at anchors 4, 6, 8, and 10 with max_partition_qubits 2, each against execute_sequential_density_reference under QA-001. No complete M-F1a, state-vector, external-protocol, Aer, energy, or frozen-matrix claim.
```

`completeness_claim` is false. Reason, verbatim from `task-9/CLOSEOUT.md`:

The counted bundle is the route-by-anchor denominator only. The Linux CI job (G-04, G6), the current-state docs (G-05, G7), the milestone closeout, the full milestone review, and the Research Manager report are still open. `true` would move the claim, which returns to Research Manager.

## 4. Disclosures

Rows 1–4, verbatim from `task-9/CLOSEOUT.md`:

**Baseline shared kernel.** Baseline and the oracle both lower through `_build_runtime_circuit` and the same C++ `NoisyCircuit` kernels (`task-1/CLOSEOUT.md:31-50`, `task-8/CLOSEOUT.md:88-97`). Bitwise agreement is expected. A kernel-level bug appears on both sides, and this oracle cannot detect it. The cells still meet the baseline rule.

Rows 9–12, including q4, verbatim from `task-9/CLOSEOUT.md`:

**Strict product of pair states, all four anchors.** q4 is two partitions on (0, 1) and (2, 3). q6, q8, and q10 add the same pair block at each (2k, 2k+1). No operation couples two pairs, so every state is a product of pair states (`task-7/CLOSEOUT.md:103`; strict bundle q4 regions at `:78-93` and `:96-111`). These cells do not test channel application on inputs correlated across a partition boundary, the order of partitions (disjoint channels commute), local depolarizing under the strict entry, or single-wire and odd-aligned motifs. C.2 hybrid covers correlated inputs and depolarizing. The strict cells still execute an eligible motif on every partition.

## 5. Freeze record

`task-9/COUNTED_MANIFEST.md` sha256 `a93de87d6b44a8a14d00d2eef73e116997faa635f35952fe6375e620aa363c16`, 87 lines. This pass does not edit that file or any bundle. Checklist §10 holds the record:

```text
Research Manager records that task-9/COUNTED_MANIFEST.md, sha256 a93de87d6b44a8a14d00d2eef73e116997faa635f35952fe6375e620aa363c16, 87 lines, exactly reflects the pinned delivered M3 and M3A advertised routes and the current-state support boundary (ADR-F1A-001): four routes at anchors 4, 6, 8, and 10, 16 cells, each with the workload, parameter source and count, and seed policy its slice pinned. No cell is dropped, merged, or substituted. The oracle, QA-001 tolerances, G-07, and O-11 are unchanged. ADR-F1A-010 item 3 for task-9: the counted set is this frozen manifest; Step 4b may start after the Architect code-ready close and this record. No counted run precedes this record.
```

## 6. REQ acceptance

| REQ | Status in this draft | Evidence |
|-----|----------------------|----------|
| REQ-001, REQ-002 | counted 16/16 QA-001 recorded | `task-9/CLOSEOUT.md`; counted bundle sha above |
| REQ-003 | hybrid labels and witnesses in the counted rows | `task-6/CLOSEOUT.md`; counted hybrid rows |
| REQ-004 | one pipeline command; (c) and (g) PASS | `validation_pipeline.py` |
| REQ-005 | open | Linux CI job not yet run for this head |
| REQ-006 | no counted disagreement in the 16 | `task-9/CLOSEOUT.md` records `summary.findings` `[]` |
| REQ-007 | frozen Phase-3.1 archive, 26-case inventory, schema, and classification inputs (`performance_evidence/`, `workloads.py`, `test_phase31_counted_matrix_validation.py`), and the eight historical bundle directories unchanged against `1cb3d20c`; no M-F1a row merged into or relabelled as the 26-case matrix | detailed plan §9 REQ-007 static diff: exit 0 and path-scoped porcelain empty at `0a66f974` (Reviewer milestone review); eight historical bundles byte-identical at task-9 (c) and (g) (`task-9/CLOSEOUT.md`) |
| REQ-008 | open; doc review later | `docs/specs/ARCHITECTURE_OVERVIEW.md` and `docs/specs/TECH_STACK.md` not updated |

The only archive change in `1cb3d20c..HEAD` is `04107d0c`, a Phase-3 erratum made in place under RM decision A (Reviewer `bc-33b7f7f2`). It is outside the REQ-007 `phase-3-1` path set, touches no bundle, and will ride the O-10 push.

Reproduce the counted pack with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

## 7. Still open

- G-04: the Linux CI job, through the existing `workflow_dispatch` trigger, after Reviewer and the Tech Lead push. No pull request. Do not edit `ci.yml`. The gate definition stays as written.
- G-05 and milestone goal G7: `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` stay unchanged until after the Research Manager report (ADR-F1A-007). Checklist G-07 stays the closed exit contract.
- Roadmap row for M-F1a stays Draft until `create-product-roadmap` revalidation.
- O-10: Tech Lead pushes `feature/dm-perf-tuning` only after Reviewer. No pull request.
- Full Opus milestone review. This file is the draft input.
- Research Manager report. `completeness_claim` stays false until that report, the CI job, and the current-state docs land.

## 8. Deferred

Checklist §9 stays deferred. This draft does not do that work.

| Item | Disposition |
|------|-------------|
| Hybrid `S_len_range` extra-row test | its own test commit |
| C.3 partial-S10 `[0, 0]` fixture and unpinned `_are_ints` | one strict-test commit |
| C.4 nits: unused `Path` import; G9 int and unknown-anchor unpinned; row-2 workloads text-match gap; `claim_boundary` unpinned; blank line | one baseline commit |
| Mutant-harness lesson: pytest exit 1 and collected count above 0 | its own skill commit |
| cmake learning | no action; already in the qgd env rule and `clean-rebuild` |
| task-8 mini-spec §3.4 and the fitness row | recorded only |

## 9. Learnings

- The 105–140 s band and the measured walls (183.9 s at (c), about 116 s at (g)) are scheduling notes, not gates.
- O-11 stays `max_partition_qubits` 2. No Aer or energy row is in the count.
- `workloads.py` was not edited. The counted module calls the five existing builders.
- Baseline agreement can be bitwise because the kernels are shared. Strict q4 through q10 are products of disjoint pair states. Neither fact blocked the 16 QA-001 passes.

## 10. Handoff

Do not mark M-F1a Delivered from this draft. Opus reviews it next. Research Manager interprets after that review. Current-state docs and the roadmap revalidation wait on that report.
