# M-F1a slice 2 closeout — historical verify-only (Slice A)

> **Status:** ready for Reviewer C2 gate · **Date:** 2026-10-05 · **Work package:** task-2 ·
> **Scope:** verify-only historical suites under the M-F1a command; M-F1a remains open ·
> **Claim:** Historical suites verified, not written. ·
> **Revision C1:** `92e95f543262ffd15b92f48c815608ecad835a3b` ·
> **C1 parent:** `5a5168fd191fa05b3af45c8bcf13ea9a49e000d9` ·
> **C2:** not committed; pending Reviewer evidence review, then Tech Lead

## Summary

Historical suites verified, not written under the M-F1a command; Slice A verify-only landed.
This closeout records the post-C1 clean-start run (no counted cell). It does not change the
oracle, QA-001 tolerances, counted set, scope, or G-07 exit set. It does not authorize another
route, anchor, or slice. Wording follows TASK_2_MINI_SPEC §3.8 only (the allowed verify-only claim).

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| (a) Reviewer implementation review | done | Reviewer approve `bc-f1841395` run #2 |
| (b) C1 Tech Lead local commit | done | C1 `92e95f54`; parent `5a5168fd` |
| (c) Clean-start pipeline run | done | exit 1 expected (S-3); see below |
| (d) Real `CLOSEOUT.md` | this file | Planner ET-A6; no commit |
| (e) Reviewer evidence review | pending | Tech Lead sends C2 gate after this write |
| (f) C2 Tech Lead local commit | pending | path list: this CLOSEOUT (+ ET-A6 checklist touch-ups only) |
| (g) Clean-C2 regeneration | pending | after C2; same S-3 acceptance; restore q4; never stage |

## Test results

### Focused tests (ET-A5 step 1 / RC-7)

```bash
conda run -n qgd --no-capture-output pytest \
  tests/partitioning/evidence/test_correctness_evidence.py \
  -o addopts="" -k mf1a_historical -v
```

Result (post-C1, Tester `tester-sliceA-c1`): **7 passed**, 29 deselected, exit 0, ~3 s.
`--collect-only -k mf1a_historical` collects exactly 7. Existing G-07 tests were not modified
by this slice.

### (c) Pipeline run (S-3 / RC-1 / Q-2)

```bash
conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `92e95f543262ffd15b92f48c815608ecad835a3b` |
| Porcelain before run | empty |
| Wall time | 52 s |
| Exit code | **1** (expected: q4 regeneration mismatch drives G-07) |
| Traceback on stderr | none |
| Run id / log | `tester-sliceA-c1` · `/tmp/tester-sliceA-c1/pipeline.log` |
| Evidence source | `/workspace/sliceA-c1-counted-report.md` (pipeline not re-run for this CLOSEOUT) |

**Stdout (nine lines, registry order):**

```text
correctness_evidence_mf1a_q4_baseline_bundle_v1: status=fail | written benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json
correctness_evidence_correctness_matrix: status=pass | verified, not written
correctness_evidence_sequential_correctness: status=pass | verified, not written
correctness_evidence_external_correctness: status=pass | verified, not written
correctness_evidence_output_integrity: status=pass | verified, not written
correctness_evidence_runtime_classification: status=pass | verified, not written
correctness_evidence_unsupported_boundary: status=pass | verified, not written
correctness_evidence_correctness_package: status=pass | verified, not written
correctness_evidence_summary_consistency: status=pass | verified, not written
```

### Per-suite status table (RC-5)

| Suite | Built status | Written? | In G-07 exit set? |
|--------|--------------|----------|-------------------|
| mf1a_q4_baseline | fail | written (sibling) | yes (drives exit 1) |
| correctness_matrix | pass | verified, not written | yes |
| sequential_correctness | pass | verified, not written | yes |
| runtime_classification | pass | verified, not written | yes |
| unsupported_boundary | pass | verified, not written | yes |
| correctness_package | pass | verified, not written | yes |
| summary_consistency | pass | verified, not written | yes |
| external_correctness | pass | verified, not written | no (excluded) |
| output_integrity | pass | verified, not written | no (excluded) |

`g07_exit_passes` is unchanged by Slice A. Exit 1 follows solely from the included sibling
`mf1a_q4_baseline` regeneration fail (S-3).

## REQ-007 eight-path diff (empty)

Before any restore, the REQ-007 eight-path diff against the committed historical bundles was
**EMPTY** (defect = 0). No restore of the eight was required. Paths (under
`benchmarks/density_matrix/artifacts/correctness_evidence/`):

- `correctness_package/correctness_package_bundle.json`
- `output_integrity/output_integrity_bundle.json`
- `runtime_classification/runtime_classification_bundle.json`
- `sequential_correctness/sequential_correctness_bundle.json`
- `external_correctness/external_correctness_bundle.json`
- `unsupported_boundary/unsupported_boundary_bundle.json`
- `correctness_matrix/correctness_matrix_bundle.json`
- `summary_consistency/summary_consistency_bundle.json`

Static base (Q-1): `1cb3d20c` (empty vs `a2928bf1` on the eight at planning). Committed six
Phase-3 bundle bytes were not regenerated or committed in Slice A and still match
`a2928bf1`.

## Containment record (ADR-F1A-009 Amendment 1 consequence 3)

| Item | Run id | Result |
|------|--------|--------|
| REQ-007 eight-path diff before restore | `tester-sliceA-c1` | EMPTY; no defect |
| Standing safety restore of the eight | `tester-sliceA-c1` | not applied (bytes already matched HEAD; skill restore remains standing until Research Manager revises it) |
| q4 restore from HEAD | `tester-sliceA-c1` | applied after copy to `/tmp`; `git show 92e95f54:<q4> > <q4>`; worktree clean |
| Regenerated q4 copy only | `/tmp/tester-sliceA-c1/after/…/mf1a_q4_baseline_bundle.json` | sha256 `fabb4894c8fdd54b6a6c9ff011f8da034c286569dd1c3afbac824fce9b1bd26c` |
| Committed q4 at C1 HEAD (after restore) | repo path | sha256 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` |

None of the eight historical paths and no q4 sibling are staged for C2. C2 stages CLOSEOUT
(and any ET-A6 checklist touch-ups) only.

## q4 difference-set check (Q-2 / Q-8; Tester-signed)

Field diff committed pre-run vs regenerated copy — **six fields, zero unexpected** (TST-4
allowlist only):

```text
cases[0].provenance.implementation_revision: 'a50ae79f636afcd97627424f6460a0e552184345' -> '92e95f543262ffd15b92f48c815608ecad835a3b'
regeneration.first_mismatch: None -> 'cases[0].provenance.implementation_revision'
regeneration.pass: True -> False
regeneration.prior_present: False -> True
status: 'pass' -> 'fail'
summary.first_failure: None -> 'regeneration'
total=6 unexpected=0
```

Q-8 PASS per Tester evidence report. q4 was restored from HEAD before staging (SDD §3.7);
regenerated bytes remain only under `/tmp`.

## Extension pin (S-8)

Loaded `_density_matrix_cpp` `.so` sha256 before and after the (c) run:
`05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`.
Matches `extension_identities[0].sha256` from
`git show HEAD:benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json`.
No rebuild during Slice A.

## Pre-(d) gate (ADR-F1A-009 Amendment 1 / RC-5)

No runtime or oracle code path changed in C1 relative to P0b. Diff stat C1 parent → C1:

```text
git diff --stat 5a5168fd 92e95f54
```

touches `validation_pipeline.py`, `test_correctness_evidence.py`, and planning docs only —
no `squander/`, no C++, no state-vector Python. Oracle
`execute_sequential_density_reference`, QA-001 tolerances, counted set, scope, and
`g07_exit_passes` are unchanged.

## Known hazards

Standalone per-suite CLIs (`validation_scaffold.py:23-53`),
`phase31_validation_pipeline.py` (`:48`), and
`performance_evidence/validation_pipeline.py` (`:49-83`) still rewrite historical artifacts
in place and are outside the M-F1a command. Standing rule: nobody runs them on rocky until a
later decision.

## Pointers

- Lineage / rulings: `task-2/ARCHITECT_STEP_4A_RULINGS.md`
- Sibling allowlist: ADR-F1A-011 (in `ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`)
- Two-commit close: ADR-F1A-009 + Amendment 1
- Evidence pack: `/workspace/sliceA-c1-counted-report.md` (Tester; agent `bc-9820d3c5`)

## Remaining work

M-F1a remains open. Reviewer C2 evidence gate, then Tech Lead C2 (CLOSEOUT only; no evidence
bundles, no q4, no eight historical, no code/test changes). Step (g) follows C2 under the
same S-3 acceptance; q4 restore continues until Slice B. No push, no PR.
