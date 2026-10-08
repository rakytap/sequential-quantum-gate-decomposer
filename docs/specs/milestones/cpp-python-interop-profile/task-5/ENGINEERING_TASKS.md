# Engineering tasks — M-F5a task-5
> **Status:** not-ready · **Slice:** M-F5a task-5 · routes at widths 6 and 8 ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-010 ·
> **SDD stage:** step-4a
> **Boundary:** docs gate only. No stamp. Width-4 C2 stays `5f63a9d6`. REQ-004 stays open

**Verdict: not-ready.** Step 4b is not authorized. No counted width-6 or width-8 command runs in the Developer change. `task-5/CLOSEOUT.md` stays absent. Lint: task-5's missing closeout is a warning in normal mode and stays a warning under `--strict` because the stage is `step-4a`. `task-4/CLOSEOUT.md` clears that slice's strict finding once the file is present. A task-5 code-ready verdict still waits until `task-4/CLOSEOUT.md` is committed. No waiver. No placeholder.

These tasks do not publish an overhead ratio, add a public energy API, change the S-g estimator, take the binding or dispatch reduction, edit kernel, fusion, or AVX code, or open M-F1b.

## ET-1 — Accept widths 6 and 8 on the attribution flag

**Implements delivery story**
- DS-1

**Change type**
- tests | tooling

**Definition of done**
- `--attribution-routes` accepts widths 4, 6, and 8. Every other width fails. Width 6 writes only `interop_profile_bundle_routes_w6.json`. Width 8 writes only `interop_profile_bundle_routes_w8.json`. The default output follows the width. Each routes name is bound to its width: at width 6 or 8, `interop_profile_bundle_routes_w4.json` (the committed `6584be2b…` file) or the other width's routes name fails before any route runs, and at width 4, `interop_profile_bundle_routes_w6.json` or `interop_profile_bundle_routes_w8.json` fails before any route runs.
- The three counted E-VQE filenames still fail before any route runs. Without the flag, widths 4, 6, and 8 stay the E-VQE path, and a `*_routes_*` name is refused. Replacing `test_attribution_routes_refuses_non_width_four` is an authorized contract change, not a weakened test. Width 4's counted command, default output (`interop_profile_bundle_routes_w4.json`), and bundle schema stay as C1-ET4 left them; its only new refusals are the width-6 and width-8 routes names.
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

**Developer brief.** A counted bundle is one with `provenance` present. The committed routes files must carry `provenance`, so the counted rules cannot be skipped by omission. Integer means `type(v) is int`. The bool `True` fails that check. `provenance.implementation_revision` is 40 lowercase hex and `provenance.extension_identities` is a non-empty list; both are read from `provenance`, not the top level. The width-4 bundle (`6584be2b…`) already passes and must still pass.

**Definition of done**
- New red-first negatives, absent today: `apply_primitive_calls` missing, 0, negative, `8.0`, `"8"`, and `True`; `clean_start` false at top level or under `provenance`; a non-empty `dirty_paths`; `affinity_cpu` 1 or missing; a wrong `estimator` name; a wrong `bound` name; 1001 samples; `implementation_revision` missing, empty, or not 40 hex; `extension_identities` missing.
- Already enforced, and not described as absent checks: an R-strict row that carries `apply_primitive_calls` or `samples` (one extra-key guard); 999 samples; a bound that misses `mean + 1.644854 * s / sqrt(n)` with `ddof=1`; the phrases "four-route shipped", "REQ-004 met", and "speedup"; a `qbit_num` the width dispatch does not allow. Those stay green and keep their existing messages.
- On a counted bundle, top-level `clean_start` is true, `provenance.clean_start` is true, and `dirty_paths` is empty. `affinity_cpu` is 0. `estimator` is `arithmetic_mean`. `bound` is `one_sided_95_orchestration_and_apply`. Each timed row has exactly 1000 samples. Each timed sample has `apply_primitive_calls` as an int greater than 0.
- One test feeds a live width-6 or width-8 refusal row through the validator, and one test shows the committed width-4 bundle still passes. Do not run the 1000-call commands to build that live row.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write one failing test for each new red-first negative before the validator change
- [ ] Keep the already-enforced regressions green
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
- This task does not run the §3 commands. A later Tester gate runs them, with `cd` and `unset PYTHONPATH` on their own lines outside the pinned `taskset` string, from a clean tree.
- Order: width 6 from empty porcelain; commit that bundle (the width-6 C2); then width 8 from empty porcelain at that commit. The runs are sequential, never concurrent. `clean_start` detection is not narrowed. The task-5 close, when it comes, records that split: the width-6 bundle commits before the width-8 run, and `task-5/CLOSEOUT.md` follows the width-8 bundle. This file does not write that closeout.
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
