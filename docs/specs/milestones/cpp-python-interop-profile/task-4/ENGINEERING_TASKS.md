# Engineering tasks — M-F5a task-4
> **Status:** task-4 Step 4b closed · C2 `5f63a9d6` · (g) PASS · **Slice:** M-F5a task-4 · attribution routes ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 ·
> CAP-004, CAP-007 · QA-008, QA-009 ·
> **SDD stage:** step-4b-authorized
> **Boundary:** C1 `6be6282f`. C1-ET4 `431a5808`. C2 `5f63a9d6`. Bundle `6584be2b…`. Width-4 rows do not close REQ-004

**Verdict: task-4 Step 4b is closed by `task-4/CLOSEOUT.md`; (g) PASS. Not a milestone close.** Q2 and B1–B3 landed in `6be6282f` (Reviewer `a0646acb…`). Q1 is Option A (`b0edc658…`). Q1a (`5e3c8222…`) is the refusal-row schema; C1-ET4 `431a5808` (Reviewer `19c16b71…`) implements it with counted mode. The counted bundle is C2 `5f63a9d6`.
`task-4/CLOSEOUT.md` is present, so that slice's `SLICE_MISSING_CLOSEOUT` is gone. Task-5 is `step-4a`, so its absent closeout stays a warning in both modes. No waiver. No placeholder. Width-4 rows do not close REQ-004.

These tasks are the draft contract. None may publish \(O\) on a route, add a public
energy API, change the S-g estimator, take the binding or dispatch reduction, or
edit kernel, fusion, or AVX code. A missing descriptor is a handback. A required
C++ edit is a handback.

## ET-1 — Refuse an overhead ratio on a route row

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- A test accepts a separate fixture whose timed rows are only R-base, R-fused,
  and R-hybrid. Those rows carry orchestration time, an apply label, throughput
  in ns per density-matrix entry, and a one-sided bound on orchestration time
  and on the apply component only. The R-strict row is a refusal and has no number.
- The same test rejects a timed R-strict row, \(O\), a QA-007 ratio, "QA-007 met",
  a reduction claim, and an R-oracle row that lacks the E1 sentence.
- Those fixtures are not `interop_profile_bundle.json`,
  `interop_profile_bundle_w6.json`, or `interop_profile_bundle_w8.json`. The
  width-4, width-6, and width-8 validators stay unchanged.
- `milestone_counted` is false. Width-4 rows do not close REQ-004.
- The refusal-row code change is ET-4. This task does not leave that change unowned.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing route-schema tests first
- [ ] Confirm they fail because the route checks are absent
- [ ] Do not run a counted route command

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q`
- REQ-001 and REQ-004

**Risks / rollback**
- Risk: the schema copies the E-VQE \(O\) checks onto a route
- Rollback: delete the route checks. Widths 4, 6, and 8 stay

## ET-2 — Use the width-4 anchor or hand back

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- Unpack `vqe, _hamiltonian = build_task_evaluator(4)` and pass `vqe`, not the
  tuple, to `build_phase3_continuity_partition_descriptor_set`. The surface
  matches `vqe.describe_density_bridge()`: parameters 18, operations 12, gates 9,
  noise 3. The label `phase2_xxz_hea_q4_continuity` is the builder string.
- If that call raises or the counts disagree, the implementation stops and
  records the handback. It does not invent operation specs, a divisor, or a
  lower-boundary twin.
- Throughput numerator is the apply component, as task-1 §7, divided by
  `operation_count * 4^4` = 3072. Orchestration time, \(T_\mathrm{public}\), and
  \(T_\mathrm{lower}\) are not the numerator. The one-sided bound is on
  orchestration time and the apply component only. S-g is not retuned.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the descriptor check before any timed route call
- [ ] Hand back when the descriptor is absent
- [ ] Do not run 1000 calls in this draft

**Evidence produced**
- doc review of `TASK_4_MINI_SPEC.md` §2
- REQ-004 and REQ-008

**Risks / rollback**
- Risk: a new workload is substituted for the anchor
- Rollback: delete the route calls. Leave the three bundles in place

## ET-3 — Keep the diff inside attribution

**Implements delivery story**
- DS-2 and DS-3

**Change type**
- tests | docs

**Definition of done**
- The next Developer step may touch `attribution_route_lane.py`,
  `attribution_route_validation.py`, `validation_pipeline.py` (the new flag only),
  and the two interop test modules named in `TASK_4_MINI_SPEC.md` §3a.
- It does not touch `squander/src-cpp/`, the three counted bundles,
  `performance_evidence/`, `benchmark_perf.py`, `INITIAL_REQUIREMENTS.md`,
  the archive, or the current-state docs.
- At Step 4a, `task-4/CLOSEOUT.md` stayed absent. This docs pass writes it.
- At Step 4a, no counted route bundle was produced in the planning pass. C2 later committed the width-4 bundle.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Keep the later implementation out of the forbidden trees
- [ ] The width-4 counted command ran at C2 `5f63a9d6`. This docs pass does not re-run it

**Evidence produced**
- The `git diff --exit-code` rows in the mini-spec matrix
- REQ-005, REQ-006, REQ-007, and QA-009

**Risks / rollback**
- Risk: a route row tempts a reduction or a C++ change
- Rollback: revert the Python diff. Leave the counted bundles in place

## ET-4 — Implement the R-strict refusal row and the counted flag

**Implements delivery story**
- DS-1 and DS-2

**Change type**
- tests | tooling

**Definition of done**
- This task owns the ADR-F5A-010 code change. It replaces the C1 acceptance in
  `test_validate_attribution_route_bundle_accepts_four_routes`. That replacement
  is authorized. It is not a weakened test. `_minimal_attribution_route_bundle`
  carries an R-strict refusal row. The timed-R-strict variant is the negative
  "a number on R-strict fails".
- A missing R-strict row fails. \(O\) on any row fails. Any numeric field on the
  R-strict row fails, including `samples`, `orchestration`, `apply_component`,
  and `throughput`.
- The refusal row's keys are `route_id`, `entry_symbol`, `apply_label`,
  `status`, and `reason`. `reason` contains `channel_native_noise_presence` and
  `98eec857`. The row is produced by a call in that run which raised
  `NoisyRuntimeValidationError`. If `execute_partitioned_density_channel_native`
  returns, the lane stops and hands back. It does not record apply 0.
- `--attribution-routes` refuses the three counted filenames before any route
  runs. Without the flag, `--width` 4, 6, and 8 are unchanged, and a
  `*_routes_*` output name is refused. This slice's flag refuses widths other
  than 4. Widths 6 and 8 are task-5. This width-4 refusal stays the landed contract until that slice changes the flag.
- Counted mode discards 50 warm-up calls and records 1000 calls per timed
  route. The C1 default of 3 samples is not that mode. The bundle pins
  `provenance.command` to the mini-spec §3a line and refuses any other command.
  `clean_start` false writes nothing. The divisor is 3072. R-hybrid does not
  wrap `apply_to`. Both clock reads sit inside the apply-timer `with`.
- One test feeds a live lane row through the validator.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing refusal-row and flag tests first
- [ ] Confirm the C1 timed-R-strict fixture now fails for the right reason
- [ ] Do not run the 1000-call command

**Evidence produced**
- The two pytest commands in the mini-spec §4 flag and schema rows
- REQ-001, REQ-004, and REQ-006

**Risks / rollback**
- Risk: `--width` 4 starts writing a routes file
- Rollback: delete `--attribution-routes`. Leave the E-VQE path and the three bundles in place
