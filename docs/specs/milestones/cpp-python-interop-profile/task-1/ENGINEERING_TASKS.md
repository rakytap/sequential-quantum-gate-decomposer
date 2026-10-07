# Engineering tasks — M-F5a task-1
> **Status:** code-ready · **Slice:** M-F5a task-1 · E-VQE at 4 qubits only ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **SDD stage:** step-4b-authorized
> **Boundary:** writer stamp only. Step 4b is not started. No Developer handoff

Stage is `step-4b-authorized`. With no `CLOSEOUT.md`, normal mode warns
`SLICE_MISSING_CLOSEOUT`, and `--strict` promotes that one finding to an error.
No placeholder. No waiver.

These tasks are the code-ready contract. ADR-F5A-009 is filed. No task below may add
a public energy API, time an attribution route, include width 6 or 8, or take a
reduction.

## ET-1 — File the timer amendment before any code-ready stamp

**Implements delivery story**
- DS-1

**Change type**
- docs

**Disposition:** completed in this code-ready writer pass. Step 4b does not re-file ADR-F5A-009.

**Definition of done**
- `ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md` exists and states ADR-F5A-009 in the
  words of `TASK_1_MINI_SPEC.md` §6: the flag and six private int64 fields on
  `Variational_Quantum_Eigensolver_Base`, one public C++ setter and one public C++
  getter, the harness functions as their only callers, the same flag on both sides of
  every pair, and clock reads only in `optimization_problem(Matrix_real&)` and
  `evaluate_density_matrix_backend`. Every read is gated on the flag. The
  `support_outer` pair is also gated on the density backend, and that call stays at
  `:1091`.
- The amendment puts no clock inside `lower_anchor_circuit_to_noisy_circuit` and does
  not move `support_outer` into the density branch.
- It requires the §2 energy to be bit-identical with the flag off and with the flag on.
  It states the existing Aer assertion as `atol=1e-12` with NumPy's default `rtol=1e-5`
  (about 7.6e-6 at this cell). It adds no state-vector clock reads and no new energy
  symbol.
- `ADRS_CPP_PYTHON_INTEROP_PROFILE.md` indexes that companion and is not otherwise rewritten.
- The stage line is `step-4b-authorized`. It changed only after ADR-F5A-009 was in the tree and the step-4a lint was clean.

**Execution checklist (TDD: red → green → refactor)**
- [x] This task is a spec edit, not a failing product test
- [x] Confirm the amendment matches §6 before the stage flip
- [x] Do not edit `INITIAL_REQUIREMENTS.md` and do not freeze the 10 % bar

**Evidence produced**
- Doc review of the amendment against mini-spec §6
- Spec lint on this milestone tree

**Risks / rollback**
- Risk: the amendment is read as permission to rewrite `evaluate_density_matrix_backend`
- Rollback: delete the companion file; this pack already specifies that deletion

REQ-002 and REQ-003 are the requirements this task serves.

## ET-2 — Add the harness clock without a new energy entry

**Implements delivery story**
- DS-1

**Change type**
- code | tests

**Implementation path**
- C++ owner: `squander/src-cpp/variational_quantum_eigensolver/include/Variational_Quantum_Eigensolver_Base.h` and `Variational_Quantum_Eigensolver_Base.cpp`
- Wrapper owner: `squander/VQA/qgd_VQE_Base_Wrapper.cpp`, thin callers of the C++ setter and getter

**Definition of done**
- Module-level `harness_density_lower_ns`, `harness_density_set_timer_flag`, and
  `harness_density_subtimes_ns` are on the VQE wrapper extension and absent from
  `qgd_Variational_Quantum_Eigensolver_Base_Wrapper_methods`. None returns an energy.
- `harness_density_lower_ns` returns one positive int64. Its clock wraps
  `self->vqe->optimization_problem` after `Matrix_real` conversion.
- The flag and the six int64 fields are private members of
  `Variational_Quantum_Eigensolver_Base`. The flag defaults off. The public C++ setter
  is `set_harness_density_timer_flag`. The public C++ getter is
  `get_harness_density_subtimes_ns`. `harness_density_set_timer_flag` calls the setter.
  `harness_density_subtimes_ns` calls the getter and returns the six int64s in the §6
  table order. Product Python code does not call the setter.
- Warm-up and counted pairs run with that flag true on both the public call and the
  lower call. The lane reads the getter immediately after each lower call.
- `support_outer` clocks stay on the call at
  `Variational_Quantum_Eigensolver_Base.cpp:1091` and run only when the flag is on and
  `backend_mode == DENSITY_MATRIX_BACKEND`.
- A test on the §2 evaluator shows the shipped energy is bit-identical with the flag
  off and with the flag on, and that the flag-off value equals `-0.7583303034656004`
  at `cdcfe6b1`.
- The same test shows the Python class has no new energy method.
- The existing Aer node still passes unchanged. Its assertion is
  `np.isclose(..., atol=1e-12)` with NumPy's default `rtol=1e-5` (about 7.6e-6 at this
  cell). That node is not the tight timer check.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing test that the three harness symbols are absent, then that the lower call returns nanoseconds, the readout returns six int64s, and neither returns an energy
- [ ] Add the failing bit-identical energy check, flag off versus flag on, on the §2 vector, including equality with `-0.7583303034656004` at `cdcfe6b1`
- [ ] Run them and confirm the red failure
- [ ] Add the private members and the two C++ accessors on the base class, the three module functions as thin callers, and the clocks in the two authorized functions with the §6 gates
- [ ] Rebuild the extension before re-running
- [ ] Re-run the unchanged Aer node from the mini-spec evidence matrix

**Evidence produced**
- The new test, named in the implementation commit, under `tests/VQE/`
- QA-007 and QA-009 as specified in the mini-spec evidence matrix

**Risks / rollback**
- Risk: the symbol is added to `tp_methods` and looks like an energy API
- Rollback: remove the module functions and the C++ members, accessors, and clock reads

REQ-001, REQ-003, and REQ-007 are the requirements this task serves.

## ET-3 — Record the 4-qubit protocol row

**Implements delivery story**
- DS-1

**Change type**
- tests | tooling

**Definition of done**
- One sibling lane under `benchmarks/density_matrix/interop_profile/` runs the §2 cell
  and the §7 protocol and writes under `benchmarks/density_matrix/artifacts/interop_profile/`.
- The row has 1000 counted pairs, 50 discarded warm-up pairs, alternating order, the
  flag on for both sides of every one of those pairs, mean `O`, the one-sided bound,
  four components within the partition tolerance, affinity, thread environment, clock
  implementation, `milestone_counted=false`, and `clean_start=true` on that counted run.
- Throughput is present: mean of per-call `apply_to` nanoseconds divided by 3072, and
  the same one-sided 95 % upper bound. A validator fixture that omits the mean, omits
  the bound, or uses another divisor fails.
- The text "QA-007 met" does not appear.
- `performance_evidence` and `benchmark_perf.py` are unchanged.
- A non-finite sample fails the process.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write a validator that rejects a fixture row with fewer than 1000 pairs, a "QA-007 met" label, a width other than 4, a broken partition, or a throughput mean or bound that is missing or uses a divisor other than 3072
- [ ] Confirm that validator fails on the bad fixture
- [ ] Implement the lane against the frozen cell
- [ ] Do not cite the new pipeline path from an evidence table until the file is on disk

**Evidence produced**
- Validator test plus the lane's own exit status, added to the evidence matrix when the files exist
- REQ-002, REQ-006, QA-007, and QA-008

**Risks / rollback**
- Risk: the lane overwrites Phase 3 performance records
- Rollback: delete the sibling directory and its artifacts

## ET-4 — Keep batch, gradients, and R-oracle out of the row

**Implements delivery story**
- DS-1 and DS-2

**Change type**
- tests

**Definition of done**
- A negative test fails the bundle if it contains `Optimization_Problem_Batch`, an overhead
  ratio on an attribution route, an R-oracle row, or a second public energy symbol.
- The batch exclusion comment cites virtual dispatch, not "no density path".

**Execution checklist (TDD: red → green → refactor)**
- [ ] Write the failing negative tests first
- [ ] Confirm they fail because the checks are missing
- [ ] Add the checks to the validator

**Evidence produced**
- Negative tests in the interop validator module once that module exists
- REQ-001 and REQ-004

**Risks / rollback**
- Risk: the corrected batch note is treated as permission to time batch calls
- Rollback: the exclude list in §3 stays in force

## ET-5 — Leave reduction, attribution, and the archive untouched

**Implements delivery story**
- DS-2 and DS-3

**Change type**
- tests | docs

**Implementation path**
- Allowed C++ tree: `squander/src-cpp/variational_quantum_eigensolver/` (base class header and `Variational_Quantum_Eigensolver_Base.cpp` only)
- Allowed wrapper tree: `squander/VQA/qgd_VQE_Base_Wrapper.cpp`

**Definition of done**
- The diff from `cdcfe6b151e371add2881d7acca45ef47697cdba` has no changes under
  `squander/src-cpp/density_matrix`, `squander/partitioning`, or
  `docs/density_matrix_project/archive`.
- Any change under `squander/src-cpp/variational_quantum_eigensolver/` is only the
  private flag, the six int64 fields, the public setter
  `set_harness_density_timer_flag`, the public getter
  `get_harness_density_subtimes_ns`, and the clock reads in
  `optimization_problem(Matrix_real&)` and `evaluate_density_matrix_backend`, gated as
  in §6. `lower_anchor_circuit_to_noisy_circuit` is untouched. The `support_outer`
  call stays at `:1091`.
- Under `squander/VQA/`, the only additions are `harness_density_lower_ns`,
  `harness_density_set_timer_flag`, `harness_density_subtimes_ns`, and the module
  method table those functions need. The `PyModuleDef` at
  `qgd_VQE_Base_Wrapper.cpp:1725` has no `m_methods` today. The wrapper struct does
  not gain the flag or the six fields.
- `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` are not edited in the tracer implementation.
- This planning pack has no `CLOSEOUT.md`. The later counted run records `clean_start`
  true and is closed under ADR-F1A-009 only after that run.
- Exactness evidence is the unchanged Aer node (`atol=1e-12` with NumPy's default
  `rtol=1e-5`, about 7.6e-6 at this cell) and the flag-off versus flag-on bit-identical
  energy test. The bit-identical test is the tight check. Both are named before the
  C++ edit is treated as done.

**Execution checklist (TDD: red → green → refactor)**
- [ ] Add a review check that lists forbidden paths
- [ ] Confirm a fixture diff of `squander/partitioning` fails that check
- [ ] Keep the implementation inside the allowed files

**Evidence produced**
- `git diff --exit-code` on the forbidden trees, as in the mini-spec matrix
- REQ-005, REQ-008, and REQ-009

**Risks / rollback**
- Risk: a timer patch edits lowering or `apply_to` math
- Rollback: revert the VQE C++ diff. The §2 energy must stay bit-identical to the flag-off value, including `-0.7583303034656004` at `cdcfe6b1`
