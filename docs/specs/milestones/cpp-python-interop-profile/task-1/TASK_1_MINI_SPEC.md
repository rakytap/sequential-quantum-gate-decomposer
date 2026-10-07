# Task 1: E-VQE at 4 qubits, interop tracer
> **Status:** task-1 Step 4b slice closed · **Slice:** M-F5a task-1 ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009 ·
> CAP-004, CAP-007 · QA-007, QA-008, QA-009 · ADR-F5A-001…009 ·
> **Scope:** E-VQE at 4 qubits only. No reduction. No attribution routes. No widths 6 or 8 ·
> **Gate:** task-1 Step 4b slice closed. SDD stage `step-4b-authorized`. C1 `ca5589e2`; QA-007 stays `[confirm]` ·
> **Pair, inventory, no-O rule, and kernel/fusion/AVX boundary:** unchanged

## 1. What this slice is

This tracer freezes the thinnest M-F5a path: one prebuilt E-VQE evaluator at 4 qubits,
the harness-only lower call into the same C++ `optimization_problem` density branch,
1000 warmed paired calls, the overhead ratio and its one-sided bound, and the component
split. It does not deliver the milestone outcome. Widths 6 and 8, R-base, R-fused,
R-strict, R-hybrid, and any binding reduction stay later slices. ADR-F5A-008 is the
slice boundary. An E1 diagnosis row is not a route; this slice decides not to add one.

The equal-work pair is unchanged. The counted inventory is unchanged. Attribution-only
routes still publish no overhead ratio. Kernel, fusion, and AVX stay out. No new public
Python energy API is added. The QA-007 10 % bar stays `[confirm]` (G-06). This cell
withholds "QA-007 met".

## 2. Frozen tracer cell

| Field | Value |
|-------|--------|
| Entry | E-VQE: `Optimization_Problem` on `qgd_Variational_Quantum_Eigensolver_Base`, `backend="density_matrix"` |
| Width | 4 qubits. No 6 or 8 |
| Ansatz | `set_Ansatz("HEA")` then `Generate_Circuit(1, 1)` |
| Hamiltonian | `generate_hamiltonian` on the line topology `[(0,1),(1,2),(2,3)]`, as in `tests/VQE/test_VQE.py` |
| Config | `max_inner_iterations=4`, `max_iterations=1`, `convergence_length=2` |
| Noise, in order | local depolarizing on target 0 after gate 0 at 0.1; amplitude damping on target 1 after gate 2 at 0.05; phase damping on target 0 after gate 4 at 0.07 |
| Parameters | 18 float64 values, `linspace(0.05, 0.05*18, 18)`, reused for every call. No RNG seed |
| Optimizer | not invoked. `Start_Optimization` is out of this slice |
| Lower call | harness-only, same instance, C++ `optimization_problem(Matrix_real&)` density branch |
| Claim | `milestone_counted=false`. This row is not the 4/6/8 QA-007 verdict |

The stored noise records use the normalized `value` field (0.1, 0.05, 0.07). The
26-case matrix is not loaded. 17/9/0 is not edited.

## 3. Inventory (F-1 corrected here)

The five-id denominator is unchanged: E-VQE for a QA-007-shaped row at this width only,
and R-base, R-fused, R-strict, R-hybrid as later attribution-only routes that this
slice does not run. R-oracle stays excluded (section 5).

`Optimization_Problem_Batch` stays excluded. The exclude reason in detailed-plan §5 is
wrong and is corrected here, not by editing that file. `Optimization_Interface::optimization_problem_batched`
has no written density branch, and its CPU path still calls virtual
`optimization_problem` (`Optimization_Interface.cpp:1010`, and `:982` under MPI;
declared virtual at `Optimization_Interface.h:333`). That dispatch hits the VQE override
and the density branch. A reviewer probe at 4 qubits saw batch values match single
density calls. Exclusion holds because QA-007 is a single energy scalar, QA-007 freezes
batching, batch returns an array, and this slice's trials are repeated single calls.
The exclude list is otherwise unchanged: gradients, state-vector
`Expectation_value_of_energy_real`, GQML `Optimization_Problem`, `NoisyCircuit.apply_to`
as its own entry, and the helpers `density_energy` and `hermitian_energy_real`.

## 4. Harness mechanism (G-02)

A lawful in-process path exists. No Research Manager consult.

The C++ object lives in `qgd_Variational_Quantum_Eigensolver_Base_Wrapper` (`qgd_VQE_Base_Wrapper.cpp:50`).
The type method table is `qgd_Variational_Quantum_Eigensolver_Base_Wrapper_methods`
(`:1575`). The shipped energy entry remains `Optimization_Problem` (`:1620`), which calls
`optimization_problem` at `:1076`.

The harness adds module-level functions on that extension. None of them is inserted
into `tp_methods`, none is a method of the Python VQE class, and none returns an energy.
`harness_density_lower_ns` returns one int64 nanosecond count. Callers pass the existing
wrapper object. The function casts that object in the same translation unit, reads
`self->vqe`, converts the already C-contiguous float64 parameter array to `Matrix_real`
before the clock, then times only `vqe->optimization_problem(parameters)`. Both sides
use the same prebuilt instance. Allocate, build,
and the support check at the start of `optimization_problem` stay inside that call on
both sides. The flag and the six sub-times are private state of
`Variational_Quantum_Eigensolver_Base` (§6), reached from the wrapper by one C++ setter
and one C++ getter. The support check inside lowering is not timed as its own interval.

`T_public` is `time.perf_counter_ns` around the shipped Python `Optimization_Problem`.
`T_lower` is `clock_gettime(CLOCK_MONOTONIC)` immediately before and after
`optimization_problem`. On this host `perf_counter` is that clock. The bundle records
`time.get_clock_info("perf_counter").implementation` and fails if it is not
`clock_gettime(CLOCK_MONOTONIC)`. No calibration constant is added to `T_lower`. Adding
one would put the harness crossing into `T_lower` and bias the ratio toward zero.

## 5. R-oracle (G-03)

Not re-included. The inner timer on `circuit.apply_to` inside
`evaluate_density_matrix_backend` labels C++ `apply_to` for this E-VQE cell. An E1
diagnosis row is not required, and this slice does not build one. ADR-F5A-008's ban on
adding a route is untouched.

## 6. Components, flag, and readout (F-2, B1, B2, W-1, W-2)

Four reported components still partition `T_public`. Six sub-times feed them. Clock
reads sit only in `optimization_problem(Matrix_real&)` and
`evaluate_density_matrix_backend`. No clock sits inside
`lower_anchor_circuit_to_noisy_circuit`. Every clock read runs only when the timer
flag is on. The `support_outer` pair also requires
`backend_mode == DENSITY_MATRIX_BACKEND`. That call stays at
`Variational_Quantum_Eigensolver_Base.cpp:1091`, before the density branch at `:1093`.
State vector shares the call, so the call stays outside the branch. State-vector code
takes no new clock reads.

| Sub-time | Where it is clocked |
|----------|---------------------|
| `support_outer` | the existing `validate_density_anchor_support` call at `:1091`, and only when the flag is on and the backend is density |
| `construct` | `DensityMatrix` and `NoisyCircuit` construction inside `evaluate_density_matrix_backend`, flag on |
| `lowering` | the call to `lower_anchor_circuit_to_noisy_circuit`, timed from the caller, flag on. The inner support check stays inside that call and is not a separate sub-time |
| `apply_to` | `circuit.apply_to` inside `evaluate_density_matrix_backend`, flag on |
| `contraction` | `expectation_value_of_density_energy_real` inside `evaluate_density_matrix_backend`, flag on |
| `teardown` | from the return of the contraction until `optimization_problem` returns. The start time is held in the `teardown` field and overwritten with elapsed nanoseconds after `evaluate_density_matrix_backend` returns |

Allocate-and-build is `support_outer + construct + lowering + teardown`. The wrapper
component is `T_public - T_lower` and holds `super().Optimization_Problem`, argument
parse, array handling, `numpy2matrix_real` on the public path, virtual-call overhead,
`Py_DECREF`, and `Py_BuildValue`. `T_lower` is allocate-and-build plus `apply_to` plus
contraction. The row fails when that sum differs from the outer `T_lower` clock by more
than 1 microsecond or 1 % of `T_lower`, whichever is greater. That tolerance is the
timing resolution this cell records for the ADR-F5A-004 partition rule.

The flag and the six int64 fields are private members of
`Variational_Quantum_Eigensolver_Base`, declared in
`include/Variational_Quantum_Eigensolver_Base.h`. The wrapper struct keeps only
`Hamiltonian` and `Variational_Quantum_Eigensolver_Base* vqe`. One public setter,
`set_harness_density_timer_flag(bool)`, and one public getter,
`get_harness_density_subtimes_ns(int64_t out_ns[6])`, follow `set_density_noise_specs`
and `get_initial_state`. The data members stay private. The constructor sets the flag
false, zeroes the six fields, and takes no clock read.

Module-level `harness_density_set_timer_flag` is the only caller of the setter.
Module-level `harness_density_subtimes_ns` is the only caller of the getter. It returns
the six int64 values in the table order and no energy. Product Python code does not
call the setter. The lane sets the flag true once, before the 50 warm-up pairs, and
leaves it true through those pairs and the 1000 counted pairs. Public
`Optimization_Problem` and `harness_density_lower_ns` both enter `optimization_problem`
on that instance, so both sides of every pair run with the flag on. A one-sided flag
is not equal work. With the flag on for both sides, the inner clock reads cancel in
`O`.

When the flag is on and the backend is density, `optimization_problem` zeroes the six
fields at entry and writes them before return. Otherwise the fields stay unchanged and
neither function reads a clock. Inner clocks use `clock_gettime(CLOCK_MONOTONIC)`, the
same clock as `T_lower`. The getter's six values are defined only after
`optimization_problem(Matrix_real&)`. The lane calls `harness_density_subtimes_ns`
immediately after each lower call, before the next public call, and uses that tuple
for the partition check.

ADR-F5A-009 in `ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md` records this section.
The index line is the only edit to `ADRS_CPP_PYTHON_INTEROP_PROFILE.md`. The stage
line is `step-4b-authorized`. C1 tip is `ca5589e2`. The counted close is
`task-1/CLOSEOUT.md`. QA-007 stays `[confirm]`. The amendment
authorizes the private members, the setter and
getter, the three module functions, and these gated clock reads. It leaves the
`support_outer` call in place, puts no clock inside lowering, and adds no reduction,
kernel rewrite, or energy symbol. The §2 energy is bit-identical with the flag off
and with the flag on. The extension is rebuilt before tests. The existing Aer check
in §10 stays unchanged.

## 7. Protocol pins (G-04 and G-05)

G-04 for this slice is section 2. Widths 6 and 8 are not pinned here.

- Paired, not interleaved. Pair `i` runs public then lower when `i` is even, and lower
  then public when `i` is odd. 1000 counted pairs. 50 warm-up pairs, same order, discarded.
- One reused parameter array. No batch API.
- Affinity: the lowest CPU in the process's allowed mask. Record the id. Fail if affinity
  cannot be set.
- Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `NUMEXPR_NUM_THREADS` are `1`. Record the values.
- `O_i = (T_public,i - T_lower,i) / T_public,i`. The reported `O` is the arithmetic mean.
  The one-sided 95 % upper bound is mean plus `1.644854 * s / sqrt(1000)`, with `s` the
  sample standard deviation (`ddof=1`). No sample is dropped. A non-finite sample or a
  non-positive `T_public` fails the row. The median of `O_i` is recorded and is not `O`.
- Throughput uses the per-call `apply_to` sub-time, in nanoseconds, from the lower call
  of each counted pair. The divisor is `operation_count * 256`, where 256 is `4^4` and
  `operation_count` is `describe_density_bridge()["operation_count"]`. For this cell that
  count is 12, so the divisor is 3072. The row reports the mean of those per-call
  throughputs and the same one-sided 95 % upper bound used for `O`. `T_public` and
  `T_lower` are not the numerator.
- QA-008 margin for this cell: categorical labels match exactly, and a rerun mean of `O`
  lies within 0.02 absolute of the recorded mean. That margin is not the QA-007 bar.
- While G-06 is open, the row reports `O` and the bound and does not print "QA-007 met".

The counted bundle is `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json`, and `task-1/CLOSEOUT.md` records it.
"Counted pairs" means the 1000 timing pairs. "Counted inventory" is the milestone
denominator. `milestone_counted=false` means this cell is not the 4/6/8 verdict. The
bundle records `clean_start` true. Its close follows ADR-F1A-009. QA-007 stays
`[confirm]`. This slice close does not complete the milestone.

## 8. Notes that do not change contracts

F-3. ADR-F5A-005 still requires the wrapper component to be strictly the largest at every
width in {4, 6, 8} before a reduction. That rule is stricter than the RM sentence and
will likely close as zero reductions, because `apply_to` grows with `4^n`. The tracer
takes no reduction. No Research Manager consult. An amendment that would allow a
width-4-only reduction is not a task-1 decision.

F-4. Later attribution slices measure orchestration against apply from the harness side,
because partitioned runtime records a total time only. If they need to edit runtime
sources, they hand back. R-fused's apply label is deferred; its islands use
`apply_local_unitary` as well as `NoisyCircuit.apply_to`, and this slice does not label
them. The next Layer 1 touch of the REQ-005 diff adds
`squander/src-cpp/variational_quantum_eigensolver/` so an edit there is visible.
Authorized timer edits, once ADR-F5A-009 exists, are the only permitted diff in that tree.

## 9. Unsupported in this slice

- Widths 6 and 8, and any claim that the milestone QA-007 verdict is complete.
- R-base, R-fused, R-strict, R-hybrid, and R-oracle rows.
- A binding, dispatch, kernel, fusion, AVX, or GPU change.
- A new public Python energy symbol, including a harness function that returns energy.
- `Optimization_Problem_Batch` and `Optimization_Problem_Grad` as timed entries.
- "QA-007 met" while the bar is `[confirm]`.
- Edits to `INITIAL_REQUIREMENTS.md` or the detailed plan. The only edit to `ADRS_CPP_PYTHON_INTEROP_PROFILE.md` is the ADR-F5A-009 index line.

## 10. Evidence matrix

The counted bundle and its reproduce command are in `task-1/CLOSEOUT.md`. Rows below
keep the code-ready checks. The lint row is the post-close expectation.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-001, REQ-003 | doc review | this mini-spec §§2–6 and §11 | five-id list unchanged; batch excluded on the corrected ground; lower harness returns nanoseconds only; the flag and six int64 fields are private members of `Variational_Quantum_Eigensolver_Base`; the wrapper calls one C++ setter and one C++ getter and returns no energy | DS-1 |
| REQ-002, QA-007 | anchor smoke plus doc review | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_density_matrix_backend_anchor_fixed_parameter_smoke -q` | existing anchor test passes; the tracer row withholds "QA-007 met" | DS-1 |
| REQ-002, REQ-003 | Aer exactness | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference -q` | the existing test is unchanged and passes its assertion (`atol=1e-12` with NumPy's default `rtol=1e-5`, about 7.6e-6 at this cell; measured gap 1.6e-15 at `cdcfe6b1`). The tight timer check is the flag-off versus flag-on bit-identical test in ET-2, added to this matrix when that test file exists | DS-1 |
| REQ-004, REQ-005 | doc review | this mini-spec §§1 and 9 | no attribution row and no reduction in this slice | DS-2 |
| REQ-006, QA-008 | repo review | `git diff --exit-code cdcfe6b151e371add2881d7acca45ef47697cdba -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty; M-F1b records untouched by this pack | DS-1 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-008 | repo review | `git diff --exit-code cdcfe6b151e371add2881d7acca45ef47697cdba -- docs/density_matrix_project/archive` | empty | DS-2 |
| REQ-009 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile` | after `task-1/CLOSEOUT.md`: normal and `--strict` are clean of `SLICE_MISSING_CLOSEOUT` and exit 0; no waiver; QA-007 stays `[confirm]` | DS-3 |

## 11. Affected interfaces

The C++ owner is `Variational_Quantum_Eigensolver_Base`
(`include/Variational_Quantum_Eigensolver_Base.h` and
`Variational_Quantum_Eigensolver_Base.cpp`). It gains the private timer flag, the six
private int64 fields, `set_harness_density_timer_flag`, and
`get_harness_density_subtimes_ns`. The wrapper owner is
`squander/VQA/qgd_VQE_Base_Wrapper.cpp`. It gains `harness_density_lower_ns`,
`harness_density_set_timer_flag`, `harness_density_subtimes_ns`, and the module method
table those functions need. The setter and getter module functions are the thin callers
of the C++ accessor pair. The wrapper struct does not store the flag or the fields.
Clock reads are the gated reads in §6. None of this is a public energy API.
ADR-F5A-009 authorizes that edit. Rollback is deleting those members, accessors,
module functions, and clock reads. No new runtime dependency.
