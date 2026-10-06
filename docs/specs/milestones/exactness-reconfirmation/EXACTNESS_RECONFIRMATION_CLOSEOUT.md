# EXACTNESS_RECONFIRMATION Closeout — `exactness-reconfirmation` (counted 16-cell admin close)

> **Status:** admin close recorded (RM `completeness_claim` flip 2026-10-06) · **Date:** 2026-10-06 ·
> **Milestone:** M-F1a `exactness-reconfirmation` · **completeness_claim:** true (RM admin; docs handoff) ·
> **Docs HEAD (G-05):** `11795eedb4715cd014e1647db044f1b6b40d6283` · **Counted C2:** `0a66f974` ·
> **G-04 pin:** `031996f4` rocky-local PASS · **No push/PR**

**M-F1a exactness reconfirmed** for the ADR-F1A-001 counted denominator (16/16 QA-001), with stated
disclosures: baseline shared-kernel agreement with the sequential oracle; strict cells are products
of disjoint pair states. Admin close at docs HEAD `11795eed` (G-04 rocky PASS recorded at
`031996f4`). The frozen counted bundle (`a22ee685…`) may still record `completeness_claim` false
at generation time; this RM admin flip supersedes that JSON flag for milestone handoff.

## 1. Summary

The frozen 16-cell denominator is recorded. All 16 counted cases pass QA-001. `summary.findings` is
`[]`. `summary.milestone_counted_cases` is 16. G-04 rocky-local project CI PASS at HEAD
`031996f4` (1067 passed, 1 QX2 deselected, exit 0, wall 40m56s;
`/tmp/mf1a-g04-rocky-ci/REPORT.md`; pytest.log `/tmp/mf1a-g04-rocky-ci/pytest.log`). G-05
ADR-F1A-007 docs at commit `11795eed`. Research Manager admin flip 2026-10-06 sets
`completeness_claim` true for handoff. Provisional sibling evidence through C.4 stays historical.
Roadmap revalidation and optional O-10 origin sync remain ahead.

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

**Research Manager admin flip (2026-10-06).** `completeness_claim` is **true** for milestone handoff.
Authorized wording (RM brief): **M-F1a exactness reconfirmed** for the ADR-F1A-001 counted
denominator (16/16 QA-001), with stated disclosures: baseline shared-kernel agreement with the
sequential oracle; strict cells are products of disjoint pair states. Admin close at docs HEAD
`11795eed` (G-04 rocky PASS recorded at `031996f4`). Bundle sha
`a22ee685038170cb0991a9ae2195b20ad4c977b111409b312cc708fa9e6872f9` is unchanged QA-001 evidence;
its JSON may still show `completeness_claim` false as recorded at generation—docs supersede for
handoff. Historical task-9 reason (pre-flip): the counted bundle was the route-by-anchor
denominator only until G-04, G-05, and RM admin close completed (`task-9/CLOSEOUT.md`).

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
| REQ-005 | rocky-local G-04 PASS at `031996f4` | `/tmp/mf1a-g04-rocky-ci/REPORT.md` (1067 passed, 1 QX2 deselected, exit 0, wall 40m56s) |
| REQ-006 | no counted disagreement in the 16 | `task-9/CLOSEOUT.md` records `summary.findings` `[]` |
| REQ-007 | frozen Phase-3.1 archive, 26-case inventory, schema, and classification inputs (`performance_evidence/`, `workloads.py`, `test_phase31_counted_matrix_validation.py`), and the eight historical bundle directories unchanged against `1cb3d20c`; no M-F1a row merged into or relabelled as the 26-case matrix | detailed plan §9 REQ-007 static diff: exit 0 and path-scoped porcelain empty at `0a66f974` (Reviewer milestone review); eight historical bundles byte-identical at task-9 (c) and (g) (`task-9/CLOSEOUT.md`) |
| REQ-008 | G-05 docs at `11795eed`; RM flip recorded in this closeout | `docs/specs/ARCHITECTURE_OVERVIEW.md` and `docs/specs/TECH_STACK.md` |

The only archive change in `1cb3d20c..HEAD` is `04107d0c`, a Phase-3 erratum made in place under RM decision A (Reviewer `bc-33b7f7f2`). It is outside the REQ-007 `phase-3-1` path set, touches no bundle, and will ride the O-10 push.

Reproduce the counted pack with:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

## 7. Closed gates and remaining handoff

**Closed (admin close 2026-10-06).**

- G-04: rocky-local project CI PASS at HEAD `031996f4` (1067 passed, 1 QX2 deselected, exit 0,
  wall 40m56s; `/tmp/mf1a-g04-rocky-ci/REPORT.md`; RECIPE/pytest.log under
  `/tmp/mf1a-g04-rocky-ci/`). Not `.github/workflows/ci.yml` `workflow_dispatch`.
- G-05: ADR-F1A-007 current-state docs at commit `11795eed` (`ARCHITECTURE_OVERVIEW.md`,
  `TECH_STACK.md`).
- Research Manager `completeness_claim` admin flip: **true** for handoff (authorized wording in
  §1 and §3); frozen bundle JSON flag may remain false at generation time.

**Still ahead.**

- Roadmap row for M-F1a stays Draft until `create-product-roadmap` revalidation.
- O-10: optional origin sync of `feature/dm-perf-tuning`; no pull request required for this
  admin-close docs record.

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

Hand off to `create-product-roadmap` for revalidation. Do not mark M-F1a Delivered in
`ROADMAP.md` from this file alone. M-F5a planning is not started here. Phase 3.1 17/9/0
decision-study disclosure is unchanged.
