# M-F1a slice 1 closeout — q4 baseline partitioned tracer
> **Status:** shipped · **Date:** 2026-10-05 · **Work package:** task-1 ·
> **Scope:** q4 tracer cell only; M-F1a remains open ·
> **Cell:** `phase2_xxz_hea_q4_continuity` on `partitioned_density_descriptor_baseline` ·
> **Revision C1:** `a50ae79f636afcd97627424f6460a0e552184345` ·
> **C2:** not committed; pending Reviewer, then Tech Lead

## Summary

This closeout records the counted clean-start result for the q4 baseline tracer cell.
It does not complete M-F1a, authorize another route or anchor, or claim C2. Aer and energy
remain non-counted. `milestone_counted_cases` is 0 and `completeness_claim` is false.

## Test snapshots

| Lane | Command | Result |
|------|---------|--------|
| Counted correctness pipeline | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | exit 0; all 9 suites pass |
| State-vector preflight | `PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/gates tests/decomposition --ignore=tests/decomposition/test_wide_circuit_optimization.py --deselect tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2 -p no:cacheprovider -q` | 805 passed, 1 deselected |

## Acceptance verdicts

| Signal | Verdict | Evidence |
|--------|---------|----------|
| QA-001 on the q4 cell | pass | Frobenius `0.0` (tolerance `1e-10`); max-abs `0.0` (tolerance `1e-10`); absolute trace residual `2.9181576996470275e-17` (tolerance `1e-10`); `lambda_min` `-8.000366383372531e-17` (floor `-1e-12`); all finite; `qa001_pass` true |
| Route realization | pass | requested and realized `partitioned_density_descriptor_baseline`; `actual_fused_execution` false; `fused_region_count` 0; `partition_count` 5 |
| Clean-start provenance | pass | `clean_start` true; `dirty_paths` empty; revision `a50ae79f` |
| Bundle identity | recorded | SHA-256 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94` of the bundle file at `benchmarks/density_matrix/artifacts/correctness_evidence/mf1a/q4_baseline/mf1a_q4_baseline_bundle.json` |
| G-10 independence | pass | Tester confirmed independence on the dirty run and the counted run |

## Why the match is bitwise

`execute_sequential_density_reference` and `execute_partitioned_density` with fusion
disabled are separate calls. Each allocates its own `DensityMatrix`, and neither reads the
other call's output back. Both lower the same canonical operations 0 through 11, in the same
order, through the same `NoisyCircuit` kernels.

G-10 passed on both runs:

- Dirty run: the oracle is `noisy_runtime_core.py:1011`, with its density matrix near line
  1040 and its apply near line 1042. The cell is `noisy_runtime_core.py:816`, with its density
  matrix near line 842. Bitwise agreement follows from the same kernels and operations 0
  through 11.
- Counted run: the cell call is `mf1a_q4_baseline_validation.py:260-262`. The oracle call is
  `mf1a_q4_baseline_validation.py:263`, which reaches `noisy_runtime_core.py:1020-1042`.
  Shared lowering is `_build_runtime_circuit` at `noisy_runtime_core.py:538-552`.

**Limitation:** both paths share `_build_runtime_circuit` and the gate and noise kernels, so
a kernel-level bug would appear on both sides and this oracle cannot detect it. That
limitation is inherent to the oracle as defined; this closeout does not redefine it.

## State-vector result

no state-vector regression attributable to this slice; one known flaky SV test listed and deselected

```text
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output pytest tests/gates tests/decomposition --ignore=tests/decomposition/test_wide_circuit_optimization.py --deselect tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2 -p no:cacheprovider -q
```

Result: 805 passed, 1 deselected. `test_QX2` is
`tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2`.
Its residuals were `1.1356e-07` and `1.1462e-07` on failing observations and `9.877e-08` on
a passing observation, against `1e-7`. It is seeded but nondeterministic. It is not
attributable to C1: the diff stat shows no `squander/` or C++ paths and nothing that QX2
imports.

## Historical suite outputs

The pipeline rewrote six historical suite bundles with content drift. The drift covered
unsupported-boundary reason wording and partition, runtime, and residual records. Their
statuses and summaries did not change. Reviewer found the drift predates C1. C1 did not
change the generators or `squander`; it changed only the q4 import, registry, and
`g07_exit_passes` in `validation_pipeline`. The drift is a known item and is not fixed in
this slice. All six bundles were restored to their C1 bytes. Copies and SHA-256 values of
those six drift copies are in `/tmp/tester-mf1a-q4/c1-run/` on rocky. Those copies are not
part of this slice. The counted bundle hash above is of the bundle file at its repo path.

## Step-8 regeneration decision

Tech Lead decision, option (i): the regeneration comparator stays unchanged. The comparator
requires `provenance.implementation_revision` to equal the prior bundle's revision.
Regenerating at C2 therefore differs in that field, and the G-07 exit will be 1.

Step-8 regeneration at C2 (comparator unchanged) is accepted when:

- the q4 QA-001 metrics agree with the committed bundle within 1e-10 (Frobenius, max-abs, |Tr-1|), with lambda_min >= -1e-12;
- the route fields are identical;
- per-case `qa001.qa001_pass` and `provenance.provenance_pass` are true, and clean_start is true at C2;
- the other eight suites keep status=pass;
- every other field is identical apart from `cases[0].provenance.implementation_revision` and the fields that follow from that expected mismatch: `status` fail, `summary.first_failure` regeneration, `regeneration.prior_present` true, `regeneration.pass` false, `regeneration.first_mismatch` `cases[0].provenance.implementation_revision`. The G-07 exit 1 is therefore expected.

Tester reports a field-level diff. Changing the comparator is a possible later framework
item, not part of this slice.

## Evidence commands

```bash
conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

The counted run was revision `a50ae79f636afcd97627424f6460a0e552184345`, exited 0, and
passed all 9 suites.

## Remaining slices

M-F1a remains open. No other route, anchor, workload, or slice is authorized. Local C2 is
pending Reviewer evidence review and then the Tech Lead commit. ADR-F1A-009 step (g)
follows C2 under the step-8 decision above: the comparator stays unchanged, a revision-only
mismatch is expected, generated outputs are restored, and they are not committed.
