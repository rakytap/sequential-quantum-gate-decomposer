# Engineering tasks — M-F5a task-4
> **Status:** Step 4a · not-ready · **Slice:** M-F5a task-4 · four attribution routes ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 ·
> CAP-004, CAP-007 · QA-008, QA-009 ·
> **SDD stage:** step-4a
> **Boundary:** not-ready. Width-4 rows do not close REQ-004. No Step 4b

**Verdict: not-ready.** READY-FOR-CODE-READY-REVIEW. RM ACCEPT `c5ea0847…` is
recorded and does not flip the stage. No Developer. No counted route run. No
`task-4/CLOSEOUT.md`. At `step-4a`, normal mode has 0 errors and 1 warning,
`SLICE_MISSING_CLOSEOUT` for task-4, and exits 0. `--strict` keeps that warning,
has 0 errors, and exits 0. No waiver. No placeholder.

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
- A test accepts a separate fixture that names R-base, R-fused, R-strict, and
  R-hybrid and carries orchestration time, an apply label, throughput in ns per
  complex entry per operation, and a one-sided bound on orchestration time and
  on the apply component only.
- The same test rejects \(O\), a QA-007 ratio, "QA-007 met", a reduction claim,
  and an R-oracle row that lacks the E1 sentence.
- Those fixtures are not `interop_profile_bundle.json`,
  `interop_profile_bundle_w6.json`, or `interop_profile_bundle_w8.json`. The
  width-4, width-6, and width-8 validators stay unchanged.
- `milestone_counted` is false. Width-4 rows do not close REQ-004.

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
- The four calls use `build_phase3_continuity_partition_descriptor_set` on
  `build_task_evaluator(4)`. The surface matches `describe_density_bridge()`:
  parameters 18, operations 12, gates 9, noise 3.
- If that call raises or the counts disagree, the implementation stops and
  records the handback. It does not invent operation specs, a divisor, or a
  lower-boundary twin.
- Throughput uses divisor \(4^4\times 12 = 3072\). The one-sided bound is on
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
- A later authorized diff does not touch `squander/src-cpp/`, the three counted
  bundles, `performance_evidence/`, `benchmark_perf.py`, `INITIAL_REQUIREMENTS.md`,
  the archive, or the current-state docs.
- `task-4/CLOSEOUT.md` stays absent during Step 4a.
- No counted route bundle is produced in this planning pass.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Keep the later implementation out of the forbidden trees
- [ ] Leave Step 4b unauthorized

**Evidence produced**
- The `git diff --exit-code` rows in the mini-spec matrix
- REQ-005, REQ-006, REQ-007, and QA-009

**Risks / rollback**
- Risk: a route row tempts a reduction or a C++ change
- Rollback: revert the Python diff. Leave the counted bundles in place
