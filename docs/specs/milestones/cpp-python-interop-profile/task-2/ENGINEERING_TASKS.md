# Engineering tasks — M-F5a task-2
> **Status:** counted (c) PASS · C2-ready · **Slice:** M-F5a task-2 · E-VQE at 6 qubits ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **SDD stage:** step-4b-authorized
> **Boundary:** C1 `d336472f`. C2 is this commit. (e) pending after this write. QA-007 stays `[confirm]`. The milestone is not complete

**Verdict: counted (c) PASS.** C2-ready for Reviewer (e). `task-2/CLOSEOUT.md` records the row. C1 is `d336472f`.
Lint with this closeout: normal and `--strict` exit 0 and are clean of `SLICE_MISSING_CLOSEOUT`. No waiver.
The harness is the shipped ADR-F5A-009 timer and `harness_density_lower_ns`, unchanged, with no C++ edit. E1: R-oracle stays excluded.
Width-6 depth, noise schedule, and the parameter vector are mini-spec §2. The protocol pins are §4.
The equal-work pair, the inventory, the no-O rule, and the kernel/fusion/AVX boundary are unchanged.
QA-007 stays `[confirm]`. No A4 kill, no hold-the-line label, and no reduction on widths 4 and 6 alone.
The ET-4 allowlist stands. The counted width-6 row is the (c) bundle. It was not re-run here.

These tasks are the C1 contract. None may add a public
energy API, time an attribution route, include width 8, apply the 10 % bar, claim
an A4 kill or a reduction on widths 4 and 6, drop a sample, or clip a negative mean.
The planned diff is Python only. A required C++ edit is a handback.

## ET-1 — Pin the 6-qubit cell and its Aer oracle

**Implements delivery story**
- DS-1

**Change type**
- tests

**Definition of done**
- A test in `tests/VQE/test_vqe_interop_harness.py` builds the §2 evaluator and
  asserts `parameter_count` 30, `operation_count` 18, `gate_count` 15, `noise_count` 3,
  and Hamiltonian `nnz` 224. The existing bridge-metadata node already runs at width 6
  (N-54). This test still owns those integers.
- The same test shows flag-off and flag-on energies on that parameter vector are
  bit-identical, and that the Python class has no new energy method. Bit identity
  shares the kernel. It is not the independence oracle.
- The width-6 Aer oracle lives in that same file. It calls
  `Test_VQE._get_density_backend_aer_reference` on this cell and the line Hamiltonian.
  The assertion is `|ΔE| ≤ 1e-12 + 1e-5·|E_Aer|`, about 2.8e-7 here. Before calling
  the helper, the test calls `set_Optimized_Parameters` with the §2 vector, as the
  4-qubit node does (`tests/VQE/test_VQE.py:1843`); the helper exports the circuit
  from the optimized parameters, and without them the process crashes (return −11).
  With the call, `|ΔE|` = 7.5e-16 against that bound. The Tester shows
  the check passed rather than skipped. The frozen 4-qubit Aer node and
  `tests/VQE/test_VQE.py` stay unchanged. No 6-qubit host golden is added.
- The test does not run 1000 pairs and does not write a bundle.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing structural test and the failing Aer check first
- [ ] Run them and confirm the red failure
- [ ] Expose the width-6 builder on the existing lane module without changing the width-4 default
- [ ] Re-run the unchanged 4-qubit Aer node from the mini-spec evidence matrix

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q`
- QA-007 (label still withheld; this task does not apply the bar) and QA-009 (4-qubit Aer node unchanged; width-6 oracle named)

**Risks / rollback**
- Risk: the builder silently retunes depth or noise when `qbit_num` is 6
- Rollback: delete the width argument path. Width 4 remains `build_task1_evaluator`

REQ-002 and REQ-003 are the requirements this task serves.

## ET-2 — Width-6 validator is the full width-4 check set

**Implements delivery story**
- DS-1 and DS-2

**Change type**
- tests | tooling

**Definition of done**
- `validate_interop_bundle` still requires width 4 and divisor 3072.
- `validate_interop_bundle_w6` enforces every check in `validate_interop_bundle`
  (`interop_bundle_validation.py` from the serialized label scan through
  `milestone_counted`), with width-6 constants: `qbit_num` 6, `operation_count` 18,
  divisor 73728. That set is: the "QA-007 met" scan; provenance keys, `clean_start`,
  and `provenance_pass`; `harness_timer_flag`; pairing and protocol counts; estimator
  name `arithmetic_mean_O_i` and `no_sample_dropped`; workload keys and qa008 pins;
  per-sample finiteness, positive `T_public`, six sub-times, and the partition
  tolerance; recomputed `mean_O`, its upper bound, throughput mean, throughput upper
  bound, and the four component means; forbidden labels and symbols; and
  `milestone_counted=false`.
- It also requires, recomputed from the samples and stored under `overhead`:
  `min_O`, `max_O`, `median_O`, and `spike_count_abs_wrapper_ns_above_20000`.
- `provenance.command` must contain `--width 6` and must be the §4 counted command.
  A no-arg width-4 command fails the width-6 row.
- Negative fixtures fail: missing throughput mean, missing throughput upper bound,
  divisor 3072 or 4096 or 65536, a spike count that disagrees with the samples,
  "QA-007 met", a claim of an A4 kill, of CAP-004 hold-the-line, or of a reduction,
  width 8, and an attribution-route label. No check rejects a negative mean `O` or a
  negative wrapper mean.
- `validation_pipeline.py` decides the filename from the arguments before any pair
  runs. No `--width` still targets `interop_profile_bundle.json`. `--width 6` targets
  `interop_profile_bundle_w6.json`. Width 6 resolving to `interop_profile_bundle.json`,
  or width 4 resolving to `interop_profile_bundle_w6.json`, exits nonzero and writes
  nothing, and it does so before `run_interop_row`.
- Refusal tests use a `tmp_path` copy. No test writes the committed task-1 bundle.
- `assert_mean_o_within_margin(0.0, 0.02, margin=0.02)` returns, and a delta above
  0.02 raises. The comparison is `<=` on the absolute difference (N-51). The margin
  is not a 10 % bar check and not an A4 check.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing validator, provenance, A4, and refusal tests first, in `tests/VQE/test_vqe_interop_bundle_validation.py`
- [ ] Confirm they fail because the width-6 entry points are absent
- [ ] Implement the entry points. Do not loosen the width-4 checks
- [ ] Do not run the 1000-pair width-6 command in this task

**Evidence produced**
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q`
- REQ-001, REQ-002, REQ-004, REQ-005, REQ-006, QA-007, and QA-008

**Risks / rollback**
- Risk: the default pipeline starts writing the width-6 row onto the task-1 path
- Rollback: delete `--width` handling and `validate_interop_bundle_w6`

## ET-3 — Record the tail count without changing the estimator

**Implements delivery story**
- DS-1

**Change type**
- tests | tooling

**Definition of done**
- The width-6 `overhead` object stores `min_O`, `max_O`, `median_O`, `mean_O`,
  `upper_bound_95_O`, and `spike_count_abs_wrapper_ns_above_20000`.
- The spike count equals the number of counted samples with
  `abs(t_public_ns - t_lower_ns) > 20000`. A pure function computes that count and
  has a short fixture. The count is an observation. It is not a filter, not a
  failure, and not comparable to the width-4 count of 2 per 1000.
- The reported `O` remains the arithmetic mean, including when that mean is negative.
  The median is not substituted. Samples with `O_i < −0.5` stay in the mean.
- The row fails on a non-finite sample or a non-positive `T_public`. It does not
  fail because the mean is negative, because a sample is below −0.5, or because the
  spike count is nonzero.
- The launch text for width 6 is the §4 command. Affinity stays the lowest allowed
  CPU inside the process. Nothing sets affinity before import.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing spike-count fixture and a fixture whose mean `O` is negative and must still validate
- [ ] Confirm they fail for the missing behavior
- [ ] Add the fields to the width-6 schema
- [ ] Leave the width-4 estimator's required set intact

**Evidence produced**
- The fixture tests in `tests/VQE/test_vqe_interop_bundle_validation.py`
- QA-008 (protocol pins and the margin function from ET-2) and QA-007 (bound present, label absent, bar not applied)

**Risks / rollback**
- Risk: the threshold is coded as a sample filter or a sign check
- Rollback: stop writing the count. Do not add a drop rule or a refusal in its place

REQ-002 and REQ-006 are the requirements this task serves.

## ET-4 — Keep the diff inside the lane

**Implements delivery story**
- DS-2 and DS-3

**Change type**
- tests | docs

**Definition of done**
- Once a later pass is separately authorized, its diff touches only
  `benchmarks/density_matrix/interop_profile/`, `tests/VQE/test_vqe_interop_harness.py`,
  `tests/VQE/test_vqe_interop_bundle_validation.py`, and at counted close
  `interop_profile_bundle_w6.json`.
- This command stays empty. It is the containment row:

```bash
git diff --exit-code cffe2cab7da1f1533584f3972faacd6be3b89392 -- \
  squander/src-cpp squander/VQA squander/partitioning \
  tests/VQE/test_VQE.py \
  benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json
```

- `test_explicit_state_vector_matches_legacy_default` passes.
- The width-6 schema does not claim QA-007 met.
- This slice's close does not claim QA-007 met and does not mark M-F5a
  complete. `task-2/CLOSEOUT.md` records the counted row. C2 is this commit.
- The counted width-6 bundle is the Tester file from (c). This pass does not re-run it.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add the refusal test on `tmp_path` before any lane edit
- [ ] Confirm a fixture diff under `squander/partitioning` still fails the existing path check
- [ ] Keep the later implementation inside the allowed Python files

**Evidence produced**
- The `git diff --exit-code` command above
- `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q`
- REQ-005, REQ-007, REQ-008, REQ-009, and QA-009

**Risks / rollback**
- Risk: a width parameter tempts a C++ change in `evaluate_density_matrix_backend`
- Rollback: revert the Python diff. Leave the task-1 bundle bytes in place
