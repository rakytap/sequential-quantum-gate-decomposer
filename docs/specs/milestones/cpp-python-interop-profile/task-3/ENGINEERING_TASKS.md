# Engineering tasks — M-F5a task-3
> **Status:** Step 4a draft · not-ready · **Slice:** M-F5a task-3 · E-VQE at 8 qubits ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **SDD stage:** step-4a
> **Boundary:** draft only. QA-007 stays `[confirm]`. The milestone is not complete

**Verdict: not-ready.** READY-FOR-REVIEW. Awaiting Research Manager alignment.
Developer does not start. No counted width-8 run. No `CLOSEOUT.md`. The expected
lint finding is `SLICE_MISSING_CLOSEOUT` for task-3: a warning in normal mode
and the sole error under `--strict`. No waiver.

These tasks are the draft contract. None may add a public energy API, time an
attribution route, apply the 10 % bar, claim an A4 kill or a reduction, drop a
sample, or clip a negative mean. The planned diff is Python only. A required
C++ edit is a handback. Step 4b is not authorized.

## ET-1 — Pin the 8-qubit cell and its Aer oracle

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- A test in `tests/VQE/test_vqe_interop_harness.py` builds the §2 evaluator and
  asserts `parameter_count` 42, `operation_count` 24, `gate_count` 21,
  `noise_count` 3, and Hamiltonian `nnz` 1152.
- Flag-off and flag-on energies on that vector are bit-identical. That check is
  not the independence oracle.
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
- Negative fixtures fail for a missing throughput bound, divisor 3072 or 73728
  or 65536, "QA-007 met", "A4 kill", "hold-the-line", and "reduction taken".
  A bundle whose claim says "no reduction" still passes. No check rejects a
  negative mean `O`.
- Filename refusal is decided before any pair. Width 8 must not resolve to
  `interop_profile_bundle.json` or `interop_profile_bundle_w6.json`.
- Refusal tests use `tmp_path`. No test writes a committed bundle.

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
  the archive, `performance_evidence/`, `benchmark_perf.py`,
  `INITIAL_REQUIREMENTS.md`, or the current-state docs.
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
