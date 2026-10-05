# M-F1a slice 3 closeout — q4 regeneration comparator allowlist (Slice B)

> **Status:** ready for Reviewer C2 gate · **Date:** 2026-10-05 · **Work package:** task-3 ·
> **Scope:** q4 comparator allowlist only; M-F1a remains open ·
> **Claim:** q4 baseline regenerated at clean C1 ·
> **Revision C1:** `22071ecb0d070646ab0642b68446030b9c17203e` ·
> **C1 parent:** `99bf9d519f7aac58d8f1e6502c60912decb85995` ·
> **C2:** not committed; pending Reviewer evidence review, then Tech Lead

## Summary

Slice B replaces option (i) for the q4 sibling: `_regeneration_result` skips a
difference on `cases[0].provenance.implementation_revision` when both sides are distinct
full lowercase 40-hex git revisions. This closeout records the post-C1 clean-start proof run
(step (c)). It does not change the oracle, QA-001 tolerances, counted set, scope, or G-07 exit
set. It does not authorize another route, anchor, or slice. Wording follows
`TASK_3_MINI_SPEC.md` §3.8 and the lock acceptance table (§10). The regenerated q4 bundle is
not committed.

## ADR-F1A-009 close steps (a)–(g)

| Step | Status | Record |
|------|--------|--------|
| (a) Reviewer implementation review | done | uncommitted §9 diff plus task-3 Layer 2–4 |
| (b) C1 Tech Lead local commit | done | C1 `22071ecb`; parent `99bf9d51` |
| (c) Clean-start pipeline run | done | exit 0; §3.8 table below |
| (d) Real `CLOSEOUT.md` | this file | ET-B4; no commit |
| (e) Reviewer evidence review | pending | Tech Lead sends C2 gate after this write |
| (f) C2 Tech Lead local commit | pending | path list: this CLOSEOUT only (checklist already synced in C1) |
| (g) Clean-C2 regeneration | pending | after C2; same §3.8 acceptance; restore q4; never stage |

## Test results

### Developer pytest gate (ET-B1; pre-(a) and on C1 tree)

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest \
  tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" \
  -k "mf1a_q4_baseline_regeneration" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest \
  tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" -k "mf1a" -v
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest \
  tests/partitioning/evidence/test_correctness_evidence.py -o addopts="" \
  --collect-only -q -k "mf1a_q4_baseline_regeneration"
```

Result on the C1 tree: **14 collected** for `-k mf1a_q4_baseline_regeneration`; **`-k mf1a` green**
(ten existing q4 tests, seven historical tests, twelve new regeneration tests). Red-before /
green-after recorded at implementation review. This CLOSEOUT does not re-run pytest.

### (c) Pipeline run (§3.8 / ET-B4)

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output python \
  benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

| | |
|--|--|
| HEAD before run | `22071ecb0d070646ab0642b68446030b9c17203e` |
| Porcelain before run | empty |
| Extension `.so` sha256 before run | `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` |
| Committed q4 sha256 before run | `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` |
| Wall time | 56 s |
| Exit code | **0** |
| Traceback on stderr | none |
| Run id / log | `tester-sliceB-c1-proof` · `/tmp/slice-b-c1-proof/pipeline.log` |
| Evidence source | `/tmp/slice-b-c1-proof/REPORT.md` (Tester; agent `bc-9820d3c5`) |

**Stdout (nine lines, registry order):**

```text
correctness_evidence_mf1a_q4_baseline_bundle_v1: status=pass | written benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json
correctness_evidence_correctness_matrix: status=pass | verified, not written
correctness_evidence_sequential_correctness: status=pass | verified, not written
correctness_evidence_external_correctness: status=pass | verified, not written
correctness_evidence_output_integrity: status=pass | verified, not written
correctness_evidence_runtime_classification: status=pass | verified, not written
correctness_evidence_unsupported_boundary: status=pass | verified, not written
correctness_evidence_correctness_package: status=pass | verified, not written
correctness_evidence_summary_consistency: status=pass | verified, not written
```

### §3.8 acceptance table (parsed JSON vs `git show HEAD:` q4 path)

| Check | Required | (c) result |
|-------|----------|------------|
| Process exit | 0 | 0 |
| q4 `status` | `pass` | `pass` |
| `regeneration.pass` | true | true |
| `regeneration.first_mismatch` | null | null |
| `summary.first_failure` | null | null |
| Case-scoped recursive diff | exactly `cases[0].provenance.implementation_revision`, new value = `git rev-parse HEAD` | one path; `a50ae79f636afcd97627424f6460a0e552184345` → `22071ecb0d070646ab0642b68446030b9c17203e` |
| Full-file recursive diff | that path plus `regeneration.prior_present` false → true | exactly two paths; no third path |
| `schema_version`, `suite_name`, `manifest`, `tolerances` | equal to committed bundle | equal |
| `non_counted_context` | `{}` | `{}` |
| Schema ids | v1 strings in mini-spec §3.3 | unchanged |
| `extension_identities` | equal, sha256 `05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77` | equal |

Regenerated q4 file sha256 after run (not acceptance): `e8dc67f6866b96df78927cf51d2960cb5f4f22ded6ba4ea4a8edf5418445f5db`.
Copy: `/tmp/slice-b-c1-proof/after/mf1a_q4_baseline_bundle.json`. **Not committed.**

### Per-suite status table

| Suite | Built status | Written? | In G-07 exit set? |
|--------|--------------|----------|-------------------|
| mf1a_q4_baseline | pass | written (sibling) | yes (drives exit 0) |
| correctness_matrix | pass | verified, not written | yes |
| sequential_correctness | pass | verified, not written | yes |
| runtime_classification | pass | verified, not written | yes |
| unsupported_boundary | pass | verified, not written | yes |
| correctness_package | pass | verified, not written | yes |
| summary_consistency | pass | verified, not written | yes |
| external_correctness | pass | verified, not written | no (excluded) |
| output_integrity | pass | verified, not written | no (excluded) |

`g07_exit_passes` is unchanged by Slice B.

## Eight historical bundles (verify-only)

Before restore, all eight historical bundle paths were **byte-identical** to the preflight
snapshot under `/tmp/slice-b-c1-proof/before/` (ADR-F1A-011 verify-only; no restore of the
eight). Historical suites verified, not written.

## Containment record (ADR-F1A-009 Amendment 1)

| Item | Run id | Result |
|------|--------|--------|
| Eight historical bundles | `tester-sliceB-c1-proof` | byte-identical to preflight; no defect |
| q4 restore from HEAD | `tester-sliceB-c1-proof` | applied after copy to `/tmp`; post-restore sha256 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`; `git diff --quiet` OK |
| Regenerated q4 copy only | `/tmp/slice-b-c1-proof/after/mf1a_q4_baseline_bundle.json` | sha256 `e8dc67f6866b96df78927cf51d2960cb5f4f22ded6ba4ea4a8edf5418445f5db` |
| Committed q4 at C1 HEAD (after restore) | repo path | sha256 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` |

None of the eight historical paths, no q4 sibling JSON, and no code are staged for C2. C2
stages this CLOSEOUT only (checklist touch-ups were already in C1).

## q4 difference-set check (§3.8; Tester-signed)

Field diff committed pre-run vs regenerated copy — **two paths, zero unexpected**:

```text
cases[0].provenance.implementation_revision: 'a50ae79f636afcd97627424f6460a0e552184345' -> '22071ecb0d070646ab0642b68446030b9c17203e'
regeneration.prior_present: False -> True
total=2 unexpected=0
```

q4 was restored from HEAD before any staging. Regenerated bytes remain only under `/tmp`.

## Extension pin (B-R3 / §3.8)

Loaded `_density_matrix_cpp` `.so` sha256 before and after the (c) run:
`05f01747e986dabba73073c11c9b00fdb326afdd703e59cd5cfe27af6631cc77`.
Matches `extension_identities[0].sha256` from the committed q4 bundle. No rebuild during
Slice B C1 or the (c) run.

## Pre-(d) gate (ADR-F1A-009 Amendment 1)

No runtime or oracle code path changed in C1 relative to P1. This slice adds no counted
cell. Diff stat C1 parent → C1:

```text
git diff --stat 99bf9d51 22071ecb
```

touches seven paths:

1. `.cursor/skills/test-density-matrix/references/regeneration-acceptance.md`
2. `benchmarks/density_matrix/correctness_evidence/mf1a_q4_baseline_validation.py`
3. `docs/specs/milestones/exactness-reconfirmation/PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md`
4. `docs/specs/milestones/exactness-reconfirmation/task-3/DELIVERY_STORIES.md`
5. `docs/specs/milestones/exactness-reconfirmation/task-3/ENGINEERING_TASKS.md`
6. `docs/specs/milestones/exactness-reconfirmation/task-3/TASK_3_MINI_SPEC.md`
7. `tests/partitioning/evidence/test_correctness_evidence.py`

None of these is a counted cell, a frozen bundle JSON, a historical bundle, or a Phase 3
evidence file. Inside `mf1a_q4_baseline_validation.py` the diff is the allowlist constant,
`_is_full_git_revision`, `_allowlisted_revision_difference`, and the provenance-loop skip only.
`evaluate_mf1a_qa001`, `build_cases`, and `capture_provenance` are unchanged. Oracle
`execute_sequential_density_reference`, QA-001 tolerances, counted set, scope, and
`g07_exit_passes` are unchanged.

## Pointers

- Lock: sha256 `cbc7c08160f745e1922a1ad4e0b7f71427a1bf3c81bf884bae52b13563118d09`
- Mini-spec: `task-3/TASK_3_MINI_SPEC.md` §3.8
- Prior slice closeout: `task-2/CLOSEOUT.md` (historical verify-only; option (i) six-path record)
- Two-commit close: ADR-F1A-009 + Amendment 1
- Evidence pack: `/tmp/slice-b-c1-proof/REPORT.md`

## Remaining work

M-F1a remains open. Reviewer C2 evidence gate, then Tech Lead C2 (this CLOSEOUT only; no
evidence bundles, no q4 JSON, no code or test changes). Step (g) follows C2 under the same
§3.8 acceptance; q4 restore continues. No push, no PR.
