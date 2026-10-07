# Engineering tasks — M-F5a task-3
> **Status:** task-3 Step 4b closed · C2 `939d4908` · (g) PASS · **Slice:** M-F5a task-3 · E-VQE at 8 qubits ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **SDD stage:** step-4b-authorized
> **Boundary:** C2 `939d4908`. (e) APPROVE. (g) PASS. QA-007 stays `[confirm]`. The milestone is not complete

**Verdict: task-3 Step 4b is closed.** (g) PASS. `task-3/CLOSEOUT.md` records the row. C1 is `97d726e3`. The next slice is not opened.
Lint with this closeout: normal and `--strict` exit 0 and are clean of `SLICE_MISSING_CLOSEOUT`. No waiver.
The harness is the shipped ADR-F5A-009 timer and `harness_density_lower_ns`, unchanged, with no C++ edit. E1: R-oracle stays excluded.
Width-8 pins are mini-spec §2 and §4: 42 parameters, 21 gates, 24 operations, `nnz` 1152, divisor 1572864. S-g Measure carries. `milestone_counted=false`.
The equal-work pair, the inventory, the no-O rule, and the kernel/fusion/AVX boundary are unchanged.
QA-007 stays `[confirm]`. No "QA-007 met", no A4 kill, no "A4 false", no hold-the-line label, and no reduction justified, taken, or shipped until the Research Manager interprets the counted 4/6/8 set.
The ET-3 allowlist stands. The counted width-8 row is the (c) bundle. It was not re-run here. C0 is `c4f5df9b`. RM ACCEPT did not flip the stage.

These tasks are the C1 contract. None may add a public
energy API, time an attribution route, apply the 10 % bar, claim an A4 kill or a
reduction, drop a sample, or clip a negative mean. The planned diff is Python only.
A required C++ edit is a handback. The Developer pass is C1 `97d726e3`. The two interop
test modules take additions only. No existing test is deleted, skipped, or
weakened. The N-17 golden stays. The counted close stays ADR-F1A-009.

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
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q -rs`
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
  `"--width 6"`. Width-6 behavior is unchanged. Leave
  `_label_contains_forbidden_w6_phrase` and `FORBIDDEN_W6_CLAIM_PHRASES` unchanged.
  Width 8 gets its own matcher or token tuple. Do not add a test that "no A4 kill" fails at width 6.
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
- The authorized diff touched only
  `benchmarks/density_matrix/interop_profile/`, the two interop test modules,
  and at this counted close `interop_profile_bundle_w8.json`.
- It does not touch `squander/src-cpp/`, `squander/VQA/`, `squander/partitioning/`,
  `tests/VQE/test_VQE.py`, the archive, `performance_evidence/`,
  `benchmark_perf.py`, `INITIAL_REQUIREMENTS.md`, or the current-state docs.
- `task-3/CLOSEOUT.md` records the counted row.
- The counted width-8 bundle is the (c) file. (g) restored it. This docs pass does not re-run it.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the refusal test on `tmp_path` before any lane edit
- [ ] Keep the later implementation inside the allowed Python files

**Evidence produced**
- The `git diff --exit-code` rows in the mini-spec matrix
- REQ-005, REQ-007, REQ-008, REQ-009, and QA-009

**Risks / rollback**
- Risk: width 8 tempts a C++ change
- Rollback: revert the Python diff. Leave both committed bundles in place
