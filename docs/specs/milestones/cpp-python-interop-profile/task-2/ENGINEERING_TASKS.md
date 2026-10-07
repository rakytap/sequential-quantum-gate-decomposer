# Engineering tasks — M-F5a task-2
> **Status:** Step 4a draft · not-ready · **Slice:** M-F5a task-2 · E-VQE at 6 qubits ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **SDD stage:** step-4a
> **Boundary:** draft only. QA-007 stays `[confirm]`. The milestone is not complete

**Verdict: not-ready.** READY-FOR-REVIEW. Developer does not start. A code-ready
writer may set `step-4b-authorized` only after the Research Manager accepts this
contract, including the Measure default in `TASK_2_MINI_SPEC.md` §3. This pass does
not authorize Step 4b. No `CLOSEOUT.md` is written. The expected lint finding is
`SLICE_MISSING_CLOSEOUT` for task-2, a warning in both modes. No waiver.

These tasks are the draft contract. None may add a public energy API, time an
attribution route, include width 8, apply the 10 % bar, drop a sample, or take a
reduction. The planned diff is Python only. A required C++ edit is a handback.

## ET-1 — Pin the 6-qubit cell without a counted run

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- A test in `tests/VQE/test_vqe_interop_harness.py` builds the §2 evaluator and
  asserts `parameter_count` 30, `operation_count` 18, `gate_count` 15, `noise_count` 3,
  and Hamiltonian `nnz` 224.
- The same test shows flag-off and flag-on energies on that parameter vector are
  bit-identical, and that the Python class has no new energy method.
- The test does not run 1000 pairs and does not write a bundle.
- The existing 4-qubit golden `-0.7583303034656004` and the Aer node stay in their
  current tests. This task adds no 6-qubit host golden.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing structural test first
- [ ] Run it and confirm the red failure
- [ ] Expose the width-6 builder on the existing lane module without changing the width-4 default
- [ ] Re-run the unchanged Aer node from the mini-spec evidence matrix

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q`
- QA-007 (label still withheld; this task does not apply the bar) and QA-009 (Aer node unchanged)

**Risks / rollback**
- Risk: the builder silently retunes depth or noise when `qbit_num` is 6
- Rollback: delete the width argument path. Width 4 remains `build_task1_evaluator`

REQ-002 and REQ-003 are the requirements this task serves.

## ET-2 — Accept a width-6 bundle and refuse to replace the width-4 file

**Implements delivery story**
- DS-1 and DS-2

**Change type**
- tests | tooling

**Definition of done**
- `validate_interop_bundle` still requires width 4 and divisor 3072.
- `validate_interop_bundle_w6` requires `qbit_num` 6, `operation_count` 18, divisor
  73728, 1000 counted pairs, 50 warm-up pairs, `milestone_counted=false`,
  `estimator.name` `arithmetic_mean_O_i`, `no_sample_dropped` true, the four
  components, and the §3 spike fields.
- A fixture with "QA-007 met", width 8, an attribution-route label, or `O` on such a
  route fails both validators.
- `validation_pipeline.py` with no `--width` still targets `interop_profile_bundle.json`.
  `--width 6` targets `interop_profile_bundle_w6.json`. If width 6 resolves to the
  filename `interop_profile_bundle.json`, the process exits nonzero and writes nothing.
  If width 4 resolves to `interop_profile_bundle_w6.json`, it does the same.
- A test asserts the on-disk task-1 bundle sha256 is
  `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e` and that a refused
  write leaves that file unchanged.
- `assert_mean_o_within_margin(recorded, regenerated, margin=0.02)` returns without
  error when the absolute delta is 0.02 and raises when the delta is greater.
  Fixture tests cover both sides (N-32). The margin is not a 10 % bar check.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing validator and refusal tests first, in `tests/VQE/test_vqe_interop_bundle_validation.py`
- [ ] Confirm they fail because the width-6 entry points are absent
- [ ] Implement the entry points. Do not loosen the width-4 checks
- [ ] Do not run the 1000-pair width-6 command in this task

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q`
- REQ-001, REQ-002, REQ-004, REQ-006, QA-007, and QA-008

**Risks / rollback**
- Risk: the default pipeline starts writing the width-6 row onto the task-1 path
- Rollback: delete `--width` handling and `validate_interop_bundle_w6`

## ET-3 — Record spikes without changing the estimator

**Implements delivery story**
- DS-1

**Change type**
- tests | tooling

**Definition of done**
- The width-6 sample record stores `min_O`, `max_O`, `median_O`, and
  `spike_count_abs_wrapper_ns_above_20000`.
- The spike count equals the number of counted samples with
  `abs(t_public_ns - t_lower_ns) > 20000`. A pure function computes that count and
  has a fixture test that does not need 1000 live pairs.
- The reported `O` remains the arithmetic mean. The median is not substituted.
- The row fails on a non-finite sample or a non-positive `T_public`. It does not
  fail because the spike count is nonzero. It does not drop samples.
- The launch text in the lane module for width 6 is the N-34 environment:
  `taskset -c 0`, `PYTHONDONTWRITEBYTECODE=1`, and the four thread variables set to `1`.
  Affinity stays the lowest allowed CPU inside the process. Nothing in this task sets
  affinity before import.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing spike-count fixture first
- [ ] Confirm it fails for the missing function
- [ ] Add the fields to the width-6 bundle schema and validator
- [ ] Leave the width-4 estimator fields' required set intact

**Evidence produced**
- The fixture test in `tests/VQE/test_vqe_interop_bundle_validation.py`
- QA-008 (categorical protocol pins, margin function from ET-2) and QA-007 (bound present, label absent)

**Risks / rollback**
- Risk: the spike threshold is coded as a sample filter
- Rollback: stop writing the count. Do not add a drop rule in its place

REQ-002 and REQ-006 are the requirements this task serves.

## ET-4 — Keep the diff inside the lane

**Implements delivery story**
- DS-2 and DS-3

**Change type**
- tests | docs

**Definition of done**
- The implementation diff, once Step 4b is separately authorized, touches only
  `benchmarks/density_matrix/interop_profile/`, the two test modules named above, and
  at counted close `interop_profile_bundle_w6.json`.
- It does not touch `squander/src-cpp/`, `squander/VQA/`, `squander/partitioning/`,
  `docs/density_matrix_project/archive/`, `benchmarks/density_matrix/performance_evidence/`,
  `benchmarks/density_matrix/benchmark_perf.py`, `INITIAL_REQUIREMENTS.md`,
  `ARCHITECTURE_OVERVIEW.md`, or `TECH_STACK.md`.
- `test_explicit_state_vector_matches_legacy_default` passes.
- The text "QA-007 met" does not appear in the width-6 schema.
- This slice's close, when it later exists, does not claim QA-007 met and does not
  mark M-F5a complete. `task-2/CLOSEOUT.md` is absent during Step 4a.
- No counted width-6 bundle is produced in the planning commit.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add a review check that the width-6 writer refuses the task-1 filename
- [ ] Confirm a fixture diff under `squander/partitioning` still fails the existing path check
- [ ] Keep the implementation inside the allowed Python files

**Evidence produced**
- `git diff --exit-code` on the forbidden trees named in the mini-spec matrix, from the commit that records this pack
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q`
- REQ-005, REQ-007, REQ-008, REQ-009, and QA-009

**Risks / rollback**
- Risk: a width parameter tempts a C++ change in `evaluate_density_matrix_backend`
- Rollback: revert the Python diff. Leave the task-1 bundle bytes in place
