# Engineering tasks — M-F5a task-5
> **Status:** not-ready · **Slice:** M-F5a task-5 · routes at widths 6 and 8 ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-010 ·
> **SDD stage:** step-4a
> **Boundary:** docs gate only. No stamp. Width-4 C2 stays `5f63a9d6`. REQ-004 stays open

**Verdict: not-ready.** Step 4b is not authorized. No counted width-6 or width-8 command runs in the Developer change. `task-5/CLOSEOUT.md` stays absent. Lint: task-5's missing closeout is a warning in normal mode and stays a warning under `--strict` because the stage is `step-4a`. Task-4's missing closeout is still the strict error. That error blocks a task-5 code-ready verdict until `task-4/CLOSEOUT.md` is committed. No waiver. No placeholder.

These tasks do not publish an overhead ratio, add a public energy API, change the S-g estimator, take the binding or dispatch reduction, edit kernel, fusion, or AVX code, or open M-F1b.

## ET-1 — Accept widths 6 and 8 on the attribution flag

**Implements delivery story**
- DS-1

**Change type**
- tests | tooling

**Definition of done**
- `--attribution-routes` accepts `--width 6` and `--width 8` and writes only `interop_profile_bundle_routes_w6.json` or `interop_profile_bundle_routes_w8.json` for those widths.
- Any other width still fails. The three counted E-VQE filenames still fail before any route runs. Without the flag, widths 4, 6, and 8 stay the E-VQE path, and a `*_routes_*` name is refused.
- Each bundle names four route ids. R-base, R-fused, and R-hybrid are timed. R-strict is `handback_refused` with no timings. The reason contains `channel_native_noise_presence` and `98eec857`. A reason that contains only `pure_unitary_partition` fails. If `execute_partitioned_density_channel_native` returns, the lane stops and hands back.
- Counted mode discards 50 warm-up calls and records 1000 calls per timed route. `provenance.command` is byte-equal to the width's command in `TASK_5_MINI_SPEC.md` §3. `clean_start` false writes nothing.
- Divisors are 73728 and 1572864. Bridge pins are §2. The width-4 bundle's divisor stays 3072.
- The Developer does not run either 1000-call command.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing width-6 and width-8 flag tests first
- [ ] Confirm they fail because the flag still allows width 4 only
- [ ] Do not run the counted commands

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q`
- REQ-004 and REQ-006

**Risks / rollback**
- Risk: `--width 6` without the flag starts writing a routes file
- Rollback: restore the width-4-only flag. Leave the four pinned bundles in place

## ET-2 — Keep the anchors or hand back

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- Width 6 uses `build_task_evaluator(6)` and expects parameters 30, operations 18, gates 15, noise 3, workload `phase2_xxz_hea_q6_continuity`.
- Width 8 uses `build_task_evaluator(8)` and expects parameters 42, operations 24, gates 21, noise 3, workload `phase2_xxz_hea_q8_continuity`.
- A raise, or a count that disagrees, stops the slice and hands back. It does not invent a divisor.
- R-hybrid's apply label is derived from the executed partition classes. The width-4 class tuple is not copied onto these widths.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the descriptor checks before any timed route call
- [ ] Hand back when the counts disagree
- [ ] Do not run 1000 calls

**Evidence produced**
- doc review of `TASK_5_MINI_SPEC.md` §2
- REQ-004 and REQ-008

**Risks / rollback**
- Risk: a new workload is substituted for the anchor
- Rollback: delete the width-6 and width-8 route calls

## ET-3 — Keep the diff on the allowlist

**Implements delivery story**
- DS-2 and DS-3

**Change type**
- tests | docs

**Definition of done**
- The Developer may touch only:
  - `benchmarks/density_matrix/interop_profile/attribution_route_lane.py`
  - `benchmarks/density_matrix/interop_profile/attribution_route_validation.py`
  - `benchmarks/density_matrix/interop_profile/validation_pipeline.py` (the attribution flag and the width-6/8 output path)
  - `tests/VQE/test_vqe_interop_harness.py`
  - `tests/VQE/test_vqe_interop_bundle_validation.py`
- The Developer does not touch `squander/**`, the three counted E-VQE bundles, `interop_profile_bundle_routes_w4.json`, `performance_evidence/`, `benchmark_perf.py`, `INITIAL_REQUIREMENTS.md`, the archive, the current-state docs, or `tests/VQE/test_VQE.py`.
- A future mutant log records the command line and a mutant-active marker. That note does not block this slice.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Keep the diff on the five paths above
- [ ] Run no counted route command

**Evidence produced**
- The `git diff --exit-code` row in the mini-spec matrix
- REQ-005, REQ-006, REQ-007, and QA-009

**Risks / rollback**
- Risk: a route row tempts a reduction or a C++ change
- Rollback: revert the Python diff. Leave the pinned bundles in place

## ET-4 — Harden the attribution validator

**Implements delivery story**
- DS-1

**Change type**
- tests | tooling

**Developer brief.** `validate_attribution_route_bundle` today does not check primitive-call counts, `clean_start`, affinity, the estimator name, or an exact sample count of 1000. It requires at least 1000 samples when provenance is present. The counted width-4 bundle already satisfies the checks below (`clean_start` true, `affinity_cpu` 0, `estimator` `arithmetic_mean`, `bound` `one_sided_95_orchestration_and_apply`, 1000 samples, primitive calls 8/8/5). The hardened validator must still accept that file (`6584be2b…`). Provenance-absent tracer fixtures may keep the existing minimum of 2 samples. The exact-1000 rule applies only when provenance is present.

**Definition of done**
- Primitive calls. Each counted timed sample has integer `apply_primitive_calls` greater than 0. Missing, zero, or negative fails. An R-strict row that contains `apply_primitive_calls` fails.
- `clean_start`. On a counted bundle, top-level `clean_start` is true, `provenance.clean_start` is true, and `provenance.dirty_paths` is empty. Any of those failing is a validation error. The pipeline still writes nothing when `clean_start` is false.
- Affinity. On a counted bundle, `provenance.affinity_cpu` is 0. Missing, or any other integer, fails. The check reads the recorded field. It does not require the string `taskset` inside argv.
- Estimator. On a counted bundle, `provenance.estimator` is `arithmetic_mean` and `provenance.bound` is `one_sided_95_orchestration_and_apply`. Any other name fails. The published orchestration and apply bounds must equal mean plus `1.644854 * s / sqrt(n)` with `ddof=1`, the `Z_95` constant already used for width 4.
- Sample count. Each counted timed row has exactly 1000 samples. 999 fails. 1001 fails. An R-strict row that contains `samples` fails.
- Also reject a counted bundle that omits `implementation_revision` or `extension_identities`. Both are present on the width-4 bundle. A revision check that requires a particular sha beyond "non-empty 40-hex" is out of scope.
- The phrases "four-route shipped", "REQ-004 met", and "speedup" still fail (N-19).
- One test feeds a live width-6 or width-8 refusal row through the validator, and one test shows the committed width-4 bundle still passes. Do not run the 1000-call commands to build that live row; a single preflight raise is enough for the refusal test.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write one failing test per negative above before the validator change
- [ ] Confirm each fails because the check is absent
- [ ] Re-validate `interop_profile_bundle_routes_w4.json` after the change

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q`
- REQ-001 and REQ-004

**Risks / rollback**
- Risk: the exact-1000 rule rejects the width-4 tracer fixtures that have no provenance
- Rollback: keep the exact-1000 rule behind the provenance-present branch

## ET-5 — Leave the counted runs to a later Tester gate

**Implements delivery story**
- DS-1

**Change type**
- docs

**Definition of done**
- This task does not run the §3 commands. A later Tester gate runs them, with `unset PYTHONPATH` on its own line, from a clean tree.
- Width 6 is a foreground run. Allow 1 minute.
- Width 8 is a background job on CPU 0. Allow 10 minutes.
- Do not copy that Tester time budget into a bundle as a measured route cost.
- No per-route scientific claim is added. Compiler flags stay unpinned (N-46). Clock granularity is not added to the bundle in this slice.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Do not launch either counted command from the Developer change
- [ ] Point the Tester brief at `TASK_5_MINI_SPEC.md` §3 and §4

**Evidence produced**
- doc review of `TASK_5_MINI_SPEC.md` §3 and §4
- REQ-004

**Risks / rollback**
- Risk: a short timeout kills the width-8 job and leaves a partial file
- Rollback: delete any partial routes file. The four pinned bundles stay
