# Engineering tasks — M-F5a task-3
> **Status:** Step 4a draft · not-ready · **Slice:** M-F5a task-3 · E-VQE at 8 qubits ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **SDD stage:** step-4a
> **Boundary:** draft only. QA-007 stays `[confirm]`. The milestone is not complete

**Verdict: not-ready.** READY-FOR-CODE-READY-REVIEW. RM ACCEPT `39808966…` is
recorded and does not flip the stage. Developer does not start. No counted
width-8 run. No `CLOSEOUT.md`. No C0 stamp. At `step-4a`, normal mode has 0
errors and 1 warning, `SLICE_MISSING_CLOSEOUT` for task-3, and exits 0.
`--strict` keeps that warning, has 0 errors, and exits 0. After a later stage
flip, `--strict` exits 1 on that finding until a real closeout. After a real
closeout, both modes are clean. No waiver. No placeholder.

The stage line moves only when RM ACCEPT, this W-1…W-7 fold, a Reviewer
code-ready re-gate APPROVE, and a separate planning-role stamp-only pass are
all done. This fold keeps `step-4a`.

These tasks are the draft contract. None may add a public energy API, time an
attribution route, apply the 10 % bar, claim an A4 kill or a reduction, drop a
sample, or clip a negative mean. The planned diff is Python only. A required
C++ edit is a handback. Step 4b is not authorized. The two interop test modules
take additions only. No existing test is deleted, skipped, or weakened. The
N-17 golden stays.

## ET-1 — Pin the 8-qubit cell and its Aer oracle

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- A test in `tests/VQE/test_vqe_interop_harness.py` builds the §2 evaluator and
  asserts `parameter_count` 42, `operation_count` 24, `gate_count` 21,
  `noise_count` 3, and Hamiltonian `nnz` 1152.
- The width-8 instance has no `harness_*` attribute and no new energy method.
- Flag-off and flag-on energies on that vector are bit-identical. That check is
  not the independence oracle. The Aer assertion also requires `|imag| ≤ 1e-12`.
- Before `Test_VQE._get_density_backend_aer_reference`, the test calls
  `set_Optimized_Parameters` with the §2 vector. The assertion is
  `|ΔE| ≤ 1e-12 + 1e-5·|E_Aer|`. The Tester shows the check passed rather than
  skipped. No 8-qubit host golden is added.
- The test does not run 1000 pairs and does not write a bundle.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing structural test and the failing Aer check first
- [ ] Run them and confirm the red failure
- [ ] Allow width 8 on the existing lane builder without changing the width-4 or width-6 defaults
- [ ] Do not run the counted command

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q`
- QA-007 (label withheld; bar not applied) and QA-009 (state-vector default unchanged)

**Risks / rollback**
- Risk: the builder retunes depth or noise at width 8
- Rollback: delete the width-8 allowlist. Widths 4 and 6 stay

REQ-002 and REQ-003 are the requirements this task serves.

## ET-2 — Accept a width-8 bundle and refuse the committed filenames

**Implements delivery story**
- DS-1 and DS-2

**Change type**
- tests | tooling

**Definition of done**
- `validate_interop_bundle` stays width 4 and divisor 3072.
  `validate_interop_bundle_w6` stays width 6 and divisor 73728.
- `validate_interop_bundle_w8` enforces that same check set with `qbit_num` 8,
  `operation_count` 24, and divisor 1572864, including recomputed `mean_O`, its
  upper bound, the throughput mean, and the throughput upper bound.
- `provenance.command` is the single-line §4 command. One constant is shared by
  the lane and the validator.
- It also requires `protocol.parameter_count` 42, `workload.hamiltonian_nnz`
  1152, and three `workload.density_noise` entries. A fixture with the width-6
  integers 30 / 18 / 224 fails. `min_O`, `max_O`, `median_O`, and
  `spike_count_abs_wrapper_ns_above_20000` sit under `overhead` and are
  recomputed. The `--width` provenance check is per profile, not a hard-coded
  `"--width 6"`. Width-6 behavior is unchanged.
- Negative fixtures fail for a missing throughput mean, a missing throughput
  bound, a spike count that disagrees with the samples, `qbit_num` 6, divisor
  3072 or 73728 or 65536, an attribution-route label, the no-arg width-4
  command, the width-6 command, the width-8 command with `--width 8` removed,
  "QA-007 met", "A4 kill", "A4 false", "hold-the-line", "reduction taken",
  "reduction justified", "reduction shipped", and "M-F5a complete". Matching is
  case-insensitive and negation-safe. N-78 stays deferred.
- A lawful width-8 fixture passes, including one with "no reduction taken",
  "milestone not complete", and "QA-007 withheld". A second fixture passes with
  a negative mean `O`, a negative wrapper mean, one sample below −0.5, and
  distinct min, max, and median. No check rejects a negative mean.
- `--width 8` runs `run_interop_row(qbit_num=8)` then
  `validate_interop_bundle_w8`. A test replaces `run_interop_row` so no pair
  runs. The live `6 if width == 6 else 4` map is not used for width 8.
  Unsupported widths still raise before any pair.
- Width 8 writes only `interop_profile_bundle_w8.json` and refuses both
  committed names before any pair. Widths 4 and 6 refuse
  `interop_profile_bundle_w8.json`. Refusal tests use `tmp_path`. Smoke runs
  use `--output /tmp/<dir>/interop_profile_bundle_w8.json`. No test writes a
  committed bundle.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing validator and refusal tests first, in `tests/VQE/test_vqe_interop_bundle_validation.py`
- [ ] Confirm they fail because the width-8 entry points are absent
- [ ] Implement the entry points. Do not loosen the width-4 or width-6 checks
- [ ] Do not run the 1000-pair command

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q`
- REQ-001, REQ-002, REQ-004, REQ-005, REQ-006, QA-007, and QA-008

**Risks / rollback**
- Risk: the default pipeline writes width 8 onto a committed bundle path
- Rollback: delete `--width 8` handling and `validate_interop_bundle_w8`

## ET-3 — Keep the diff inside the lane

**Implements delivery story**
- DS-2 and DS-3

**Change type**
- tests | docs

**Definition of done**
- Once a later pass is separately authorized, its diff touches only
  `benchmarks/density_matrix/interop_profile/`, the two interop test modules,
  and at counted close `interop_profile_bundle_w8.json`.
- It does not touch `squander/src-cpp/`, `squander/VQA/`, `squander/partitioning/`,
  `tests/VQE/test_VQE.py`, the archive, `performance_evidence/`,
  `benchmark_perf.py`, `INITIAL_REQUIREMENTS.md`, or the current-state docs.
- `task-3/CLOSEOUT.md` stays absent during Step 4a.
- No counted width-8 bundle is produced in this planning pass.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the refusal test on `tmp_path` before any lane edit
- [ ] Keep the later implementation inside the allowed Python files

**Evidence produced**
- The `git diff --exit-code` rows in the mini-spec matrix
- REQ-005, REQ-007, REQ-008, REQ-009, and QA-009

**Risks / rollback**
- Risk: width 8 tempts a C++ change
- Rollback: revert the Python diff. Leave both committed bundles in place
