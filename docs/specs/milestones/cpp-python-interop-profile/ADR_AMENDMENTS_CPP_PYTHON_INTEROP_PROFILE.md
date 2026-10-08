# ADR amendments — M-F5a `cpp-python-interop-profile`
> **Status:** accepted · **Milestone:** M-F5a `cpp-python-interop-profile` ·
> **Continues:** `ADRS_CPP_PYTHON_INTEROP_PROFILE.md` (ADR-F5A-001…008) ·
> **Scope:** ADR-F5A-009, the task-1 harness timer; ADR-F5A-010, the R-strict refusal row ·
> **Traces:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-007 · CAP-004, CAP-007 · QA-007, QA-009 ·
> **Baseline:** `cdcfe6b151e371add2881d7acca45ef47697cdba`

## ADR-F5A-009 — Home the harness timer on the C++ base and gate its clocks

**Status:** accepted. Task-1 code-ready contract. Copies `task-1/TASK_1_MINI_SPEC.md` §6
after W-1, W-2, and W-3. Does not start Step 4b and does not issue a Developer handoff.

**Context.** The equal-work pair enters `optimization_problem(Matrix_real&)` on one
prebuilt density instance. Sub-times have to be written by that C++ method. The Python
wrapper struct holds only `Hamiltonian` and `Variational_Quantum_Eigensolver_Base* vqe`
(`qgd_VQE_Base_Wrapper.cpp:50–57`), so the core cannot see wrapper fields.
`validate_density_anchor_support` at `Variational_Quantum_Eigensolver_Base.cpp:1091`
runs before `if (backend_mode == DENSITY_MATRIX_BACKEND)` at `:1093` and is shared with
state vector. The existing Aer node keeps `np.isclose(..., atol=1e-12)` and NumPy's
default `rtol=1e-5`.

**Decision.**

B1 — timer contract. The flag and the six int64 fields are private members of
`Variational_Quantum_Eigensolver_Base`, declared in
`include/Variational_Quantum_Eigensolver_Base.h`. One public setter,
`set_harness_density_timer_flag(bool)`, and one public getter,
`get_harness_density_subtimes_ns(int64_t out_ns[6])`, follow `set_density_noise_specs`
and `get_initial_state`. The data members stay private. The constructor sets the flag
false, zeroes the six fields, and takes no clock read. Module-level
`harness_density_set_timer_flag` is the only caller of the setter. Module-level
`harness_density_subtimes_ns` is the only caller of the getter and returns the six
int64 values in §6 table order and no energy. The wrapper translation unit adds those
functions, `harness_density_lower_ns`, and the module method table they need. The
`PyModuleDef` at `qgd_VQE_Base_Wrapper.cpp:1725` has no `m_methods` today. The wrapper
struct does not store the flag or the fields. Product Python code does not call the
setter. The lane sets the flag true once, before the 50 warm-up pairs, and leaves it
true through those pairs and the 1000 counted pairs, so both sides of every pair run
with the flag on. When the flag is on and the backend is density,
`optimization_problem` zeroes the six fields at entry and writes them before return.
The getter's six values are defined only after `optimization_problem(Matrix_real&)`.
The lane calls the readout immediately after each lower call. Inner clocks use
`clock_gettime(CLOCK_MONOTONIC)`, the same clock as `T_lower`. `teardown` starts in
`evaluate_density_matrix_backend` and ends in `optimization_problem`: the start time
is held in the `teardown` field and overwritten with elapsed nanoseconds after that
call returns, so the crossing adds no seventh field.

B2 — gated clock scope. Clock reads sit only in `optimization_problem(Matrix_real&)`
and `evaluate_density_matrix_backend`. Every clock read runs only when the timer flag
is on. The `support_outer` pair also requires `backend_mode == DENSITY_MATRIX_BACKEND`.
That call stays at `:1091`. State-vector code takes no new clock reads. No clock sits
inside `lower_anchor_circuit_to_noisy_circuit`. `lowering` is timed from its caller.

B3 — exactness. The §2 energy is bit-identical with the flag off and with the flag on.
At `cdcfe6b1` that flag-off value is `-0.7583303034656004`. The existing Aer test is
unchanged. It passes when `|ΔE| ≤ 1e-12 + 1e-5·|E_Aer|`, about 7.6e-6 at this cell.
The measured gap at `cdcfe6b1` is 1.6e-15. The bit-identical flag-off versus flag-on
test is the tight check on the timer edit.

The partition check fails when the six sub-times differ from the outer `T_lower` clock
by more than 1 microsecond or 1 % of `T_lower`, whichever is greater. That tolerance is
the timing resolution the bundle records for the ADR-F5A-004 partition rule on this cell.

**Rationale.** C++ can branch on the flag and write the fields only if both live on the
base class. A public accessor pair keeps those members private and gives the harness a
path the wrapper struct does not have. Gating `support_outer` on the flag and the
density backend covers the shared call without moving it into the density branch or
adding clock reads to state vector. Stating the Aer node's `rtol` keeps the external
row from being read as an atol-only bound.

**Consequences.** Step 4b, when a later handoff starts it, edits the base-class header
and `Variational_Quantum_Eigensolver_Base.cpp` for the members, the two accessors, and
the gated clocks, and edits `qgd_VQE_Base_Wrapper.cpp` for the three module functions
and their method table. The extension is rebuilt before tests. Rollback deletes those
members, accessors, module functions, and clock reads. This ADR adds no public Python
energy symbol, no state-vector clock read, no reduction, and no kernel, fusion, or AVX
edit. QA-007 stays `[confirm]`. `INITIAL_REQUIREMENTS.md` is unchanged.

**Rejected alternatives.**
- Store the flag and the six fields on the Python wrapper struct.
- Expose the six fields as public data members.
- Move `validate_density_anchor_support` into the density branch so its clocks sit inside that branch.
- Authorize clocks in `optimization_problem`'s density branch only, leaving the `:1091` pair uncovered.
- Place a clock inside `lower_anchor_circuit_to_noisy_circuit`.
- Add a seventh field to carry the `teardown` start time.
- Treat the existing Aer node as an atol-only `1e-12` bound, or edit that test in this decision.
- Add a new energy symbol so the harness can return the scalar it timed.

**Upstream alignment:** REQ-001, REQ-002, REQ-003, REQ-007 · CAP-004, CAP-007 · QA-007, QA-009 · goals G1, G2, G3, G7.

## ADR-F5A-010 — R-strict stays a required refusal row

**Status:** planning amendment, 2026-10-07. RM Option A (`b0edc658…`) and Q1a (`5e3c8222…`). Width 4 is in the lane: C1 `6be6282f`, C1-ET4 `431a5808`, C2 `5f63a9d6`, bundle `6584be2b…`. Widths 6 and 8 are not in the lane. Does not close REQ-004. Does not loosen ADR-F5A-006, REQ-005, or CAP-004.

**Context.** `execute_partitioned_density_channel_native` raises on the frozen counted noise (qubits 0 and 1) at widths 4, 6, and 8. STEP_4A_HANDBACK `98eec857…` records that raise. A bundle that omits R-strict, or that invents timings for it, is not an honest inventory.

**Decision.** An attribution bundle names exactly four route ids. R-base, R-fused, and R-hybrid carry orchestration time, the apply component, throughput, and one-sided bounds on those two times. R-strict is required with `status` `handback_refused` and a reason that cites the empty-partition raise under that frozen noise and the handback sha256 above. At width 4 the raise code is `channel_native_noise_presence`. At widths 6 and 8 the recorded code is `pure_unitary_partition` when that is the raise. The R-strict row has no timings, no ns/op, no upper bound, and no \(O\). Any number on that row fails. A missing R-strict row fails. No row carries \(O\) or a lower twin. The bundle field `milestone_counted` is false. The three counted E-VQE bundles stay untouched. After the width-4 counted close, widths 6 and 8 for the three lawful routes are next, each with an R-strict refusal row if it refuses there too. Do not wait on a live R-strict path.

**Wording amendment (2026-10-07, Q1b).** The decision sentence that records `pure_unitary_partition` as the width-6 and width-8 raise is withdrawn. The live strict raise at widths 6 and 8 is `channel_native_noise_presence`, the same code as width 4. `pure_unitary_partition` is the hybrid classifier route reason: `_scan_channel_native_whole_partition_motif` returns it when a partition has no noise (`noisy_runtime_channel_native.py:481–482`), and `_validate_whole_partition_motif` maps that reason to `first_unsupported_condition` `channel_native_noise_presence` (`:523–531`). A read-only probe at `5f63a9d6` saw that raise at both widths. STEP_4A_HANDBACK `98eec857…` line 17 stays the historical probe note and is not rewritten. This paragraph does not change the decision: R-strict stays a required `handback_refused` row with no timings, at each width, and the work does not wait on a live R-strict path.

**Not taken.** A three-id bundle. A stub timing row. Option B's route-only noise schedule is refused. Option C, loosening ADR-F5A-006, REQ-004, REQ-005, or CAP-004, is not taken here. It is escalate-only, after the 4/6/8 three-route rows, through the Research Manager, the PhD Manager, and Zoltán.

**Upstream alignment:** REQ-001, REQ-004, REQ-005 · CAP-004 · ADR-F5A-001, ADR-F5A-006.
