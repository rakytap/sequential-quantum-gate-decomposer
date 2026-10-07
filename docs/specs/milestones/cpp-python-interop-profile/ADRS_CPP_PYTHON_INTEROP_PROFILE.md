# ADRs — M-F5a `cpp-python-interop-profile`
> **Status:** Layer 1 v0.1 — accepted milestone decisions; planning only ·
> **Milestone:** M-F5a `cpp-python-interop-profile` ·
> **Scope:** decisions that span more than one future work package ·
> **Upstream:** `INITIAL_REQUIREMENTS.md` v0.1 · E1, E2, and E3 held ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 ·
> **Program decisions:** ADR-001…008 and ADR-F1A-001…011 stay in force ·
> **Baseline:** `1b123a9a31235dd68d6c0a6ff9ba457c0112cd59`

## ADR-F5A-001 — Count one QA-007 entry and four attribution-only routes

**Status:** accepted.

**Context.** REQ-001 needs a reviewed denominator before any timed claim. At `1b123a9a`
the only shipped Python call that returns a noisy energy scalar and has an equal-work
lower-boundary comparator is E-VQE. The four `execute_partitioned_density*` entries return
\(\rho\). R-oracle is the M-F1a sequential oracle, not an advertised route. RM edit E1
excludes it from the attribution set by default.

**Decision.** The counted set is:

| Id | Public entry | M-F5a role |
|----|--------------|------------|
| E-VQE | `Optimization_Problem` on `qgd_Variational_Quantum_Eigensolver_Base` with `backend="density_matrix"` | QA-007 verdict at widths 4, 6, and 8 |
| R-base | `execute_partitioned_density` | Attribution only. Apply component is C++ `NoisyCircuit.apply_to` |
| R-fused | `execute_partitioned_density_fused` | Attribution only |
| R-strict | `execute_partitioned_density_channel_native` | Attribution only. Apply component is numpy Kraus |
| R-hybrid | `execute_partitioned_density_channel_native_hybrid` | Attribution only. The label follows the executed class |

The advertised denominator is those five ids. R-oracle
(`execute_sequential_density_reference`) is outside it. A later slice may add one
R-oracle row only when the E-VQE split cannot label C++ `apply_to` from the E-VQE pair
alone. That row's claim boundary states that it exists only to label C++ `apply_to` for
the E-VQE diagnosis. The row publishes no overhead ratio and is not an advertised route.
If the pair already labels `apply_to`, R-oracle stays excluded. The inventory validator
accepts the five-id set, or that set plus this one diagnosis row. Any other extra id fails.

Outside the outcome: `Optimization_Problem_Batch`, `Optimization_Problem_Grad`,
state-vector `Expectation_value_of_energy_real`, GQML `Optimization_Problem`,
`NoisyCircuit.apply_to` as its own entry, and the helpers `density_energy` and
`hermitian_energy_real`. Step 4a may correct a symbol location only by citing a call-site
contradiction. The exclude list otherwise stays.

**Rationale.** This is the accepted outcome: a QA-007 verdict only where equal work
exists, and attribution where it does not. E1 keeps the sequential oracle from being
recounted as a fifth route.

**Consequences.** The pre-trial inventory is frozen before counted trials. Validation
accepts the five-id denominator, or that denominator plus the one E1 diagnosis row. An
overhead ratio on an attribution-only route, a missing width, or an R-oracle row without
the E1 sentence fails validation.

**Rejected alternatives.**
- Advertise R-oracle as a fifth route.
- Publish \(O\) on R-base, R-fused, R-strict, or R-hybrid by inventing a lower-boundary twin.
- Fold the batch, gradient, state-vector, or helper symbols into the outcome.
- Treat `NoisyCircuit.apply_to` as its own public energy entry.

**Upstream alignment:** REQ-001, REQ-004 · CAP-004, CAP-007 · QA-007, QA-008 · goals G1, G4.

## ADR-F5A-002 — Keep the E-VQE lower boundary harness-only

**Status:** accepted.

**Context.** QA-007 compares one prebuilt evaluator through a public Python entry and a
lower-boundary invocation where only the language crossing differs. RM edit E3 says the
lower side is a harness-only call into the same C++ density `optimization_problem` branch.
The milestone adds no public Python energy API. At `1b123a9a` the wrapper function begins
at `qgd_VQE_Base_Wrapper.cpp:1041` and calls public C++ `optimization_problem` at `:1076`.
That method runs `validate_density_anchor_support` and then private
`evaluate_density_matrix_backend`, which allocates \(\rho\), builds `NoisyCircuit`, and
calls `apply_to` on every invocation.

**Decision.** \(T_\mathrm{public}\) is shipped Python `Optimization_Problem` on the density
backend, wrapper included. \(T_\mathrm{lower}\) enters
`Variational_Quantum_Eigensolver_Base::optimization_problem(Matrix_real&)` on an instance
whose backend is the density backend, so the call takes the density branch. The harness
excludes the CPython wrapper body: argument parsing, contiguous-array handling,
`numpy2matrix_real`, and `Py_BuildValue`. The same prebuilt evaluator is used on both
sides. In-call allocate, build, and `validate_density_anchor_support` are on both sides
or on neither. A one-sided strip is not equal work: that entry publishes no QA-007 ratio
and is reported attribution-only.

The harness mechanism — how a measurement driver invokes that C++ method — is chosen in
Step 4a. It must not add a shipped Python energy symbol. If no such mechanism exists,
work stops and returns to the Research Manager.

**Rationale.** The crossing under test is the wrapper. Entering at `optimization_problem`
keeps the density branch, the support check, and the in-call allocate and build on both
sides. A new Python energy symbol would change the product surface E3 forbids.

**Consequences.** Step 4a names the mechanism before any timed claim. The pair witness
records that both sides include, or both sides exclude, allocate, build, and the support
check. Batching stays out of \(O\): the trials are repeated single calls.

**Rejected alternatives.**
- Add a public Python energy API so the lower call is convenient.
- Time private `evaluate_density_matrix_backend` alone, dropping `validate_density_anchor_support` on one side.
- Prebuild the circuit on the lower side only.
- Compare the wrapper with `optimization_problem_non_static`, the batch API, or the gradient path.

**Upstream alignment:** REQ-003 · CAP-004 · QA-007 · goal G3.

## ADR-F5A-003 — Leave the QA-007 10 % bar unfrozen

**Status:** accepted.

**Context.** QA-007 in the product statement says the one-sided 95 % upper bound on \(O\)
is at most 10 % `[confirm]`, and that a diagnosis may close a research milestone while
leaving QA-007 unmet. RM edit E2 says this accept does not lock the bar. The product
owner freezes it in `INITIAL_REQUIREMENTS.md`, the first requirements file that cites
QA-007. The A4 kill in the product statement is a different number: an upper bound below
5 % at every 4/6/8 point makes CAP-004 hold-the-line.

**Decision.** Layer 1 does not set the numeric QA-007 bar. Until the product owner records
the freeze in `INITIAL_REQUIREMENTS.md`, every E-VQE row reports \(O\) and its one-sided
95 % upper bound and withholds the label "QA-007 met". After the freeze, a row is met
only when that bound satisfies the frozen bar, and unmet-with-diagnosis otherwise.
Unmet-with-diagnosis may close the research milestone and leaves QA-007 unmet. The A4
kill stays the product-statement 5 % test. Citing it here does not freeze the 10 % bar.

**Rationale.** E2 assigns the bar to the product owner. A Layer 1 number would either
contradict that edit or pretend a confirmation that has not happened. The 5 % kill is
already decided and governs the reduction, not the met label.

**Consequences.** Counted trials that apply a numeric bar wait on the product-owner edit.
Step 4a may still specify the harness, the protocol, and the reported fields. A bundle
that prints "QA-007 met" while the bar is `[confirm]` or unmet fails validation.

**Rejected alternatives.**
- Lock 10 % in this ADR or in the planning file.
- Treat the 5 % A4 kill as the QA-007 pass bar.
- Relabel unmet-with-diagnosis as "QA-007 met".
- Block Step 4a until the bar is frozen.

**Upstream alignment:** REQ-002 · CAP-004 · QA-007 · goal G2.

## ADR-F5A-004 — Measure on a sibling interop lane

**Status:** accepted.

**Context.** QA-008 requires a named lane and pinned provenance. The Phase 3
performance-evidence pipeline stores `timing_mode` `median_3` and a speedup ratio.
`benchmark_perf.py` uses one run and zero warm-up. Both belong to other claims. M-F1b
owns the committed timing record. M-F5a must not rewrite those rows.

**Decision.** The M-F5a lane is a new sibling,
`benchmarks/density_matrix/interop_profile/validation_pipeline.py`, writing
`benchmarks/density_matrix/artifacts/interop_profile/`. A later slice creates both. The
protocol is at least 1000 counted calls after discarded warm-up, on one prebuilt
evaluator per width. Paired means each counted index runs both sides back to back.
Interleaved means the sides alternate in an order the bundle pins. Trials are repeated
single calls. The bundle pins which of those two was used, the warm-up count, affinity,
thread count, the uncertainty estimator, revision, host, CPU, compiler and flags,
dependencies, workload, and claim boundary.

\(O = (T_\mathrm{public} - T_\mathrm{lower}) / T_\mathrm{public}\) on an equal-work pair.
Uncertainty is the one-sided 95 % upper bound on \(O\). The estimator procedure is named
in the bundle before counted trials. Components reported beside \(O\) are the CPython
wrapper, in-call C++ allocate and build, `NoisyCircuit.apply_to`, and the sparse energy
contraction. Those four are non-overlapping and sum to \(T_\mathrm{public}\) within the
recorded timing resolution. On an equal-work pair the wrapper component is
\(T_\mathrm{public}-T_\mathrm{lower}\), and the other three partition \(T_\mathrm{lower}\).
Throughput is nanoseconds per density-matrix entry per operation. The divisor
is \(4^n\) complex elements of \(\rho\), times the operations that timed apply executes,
including ordered local noise. Step 4a may record another divisor before counted trials;
the bundle states the one it used.

`benchmarks/density_matrix/performance_evidence/` and
`benchmarks/density_matrix/benchmark_perf.py` stay byte-identical to `1b123a9a`.

**Rationale.** A sibling schema keeps the M-F5a sampling contract from being read as a
Phase 3 speedup row or as an M-F1b timing commit.

**Consequences.** The lane is absent at Layer 1. Evidence commands that name it become
runnable when the slice that creates it ships. Categorical labels reproduce exactly.
Performance decisions reproduce against the stated margin on every counted row.

**Rejected alternatives.**
- Overload `performance_evidence` or `benchmark_perf.py`.
- Use median-of-three as the M-F5a uncertainty statement.
- Batch the 1000 calls through `Optimization_Problem_Batch`.
- Publish a speedup ratio or an at-least-1.2× sentence from this lane.

**Upstream alignment:** REQ-002, REQ-004, REQ-006 · CAP-004, CAP-007 · QA-007, QA-008 · goals G2, G4, G6.

## ADR-F5A-005 — Allow at most one binding or dispatch reduction

**Status:** accepted.

**Context.** CAP-004 allows a bounded interop reduction only when measurement shows the
crossing is the material term. The roadmap puts kernel rewrites, fusion redesign, AVX,
and GPU outside M-F5a. The A4 kill, when it fires, makes CAP-004 hold-the-line.

**Decision.** After the profile, the change set contains either no interop reduction or
exactly one reduction. That reduction touches only the CPython wrapper path for E-VQE:
argument parsing, array conversion, the call through to `optimization_problem`, or the
boxing of the returned scalar. It does not modify `optimization_problem`,
`evaluate_density_matrix_backend`, `NoisyCircuit`, fusion, the planner, or a kernel.

The reduction is made only when both are true:

1. The A4 kill has not fired: the one-sided 95 % upper bound on \(O\) is not below 5 %
   at every width in {4, 6, 8}.
2. At every counted E-VQE width, the reported CPython-wrapper time is strictly greater
   than each of the other three component times, compared as point estimates. A tie, or
   a component that cannot be separated, means the crossing is not shown to be material.

Otherwise the reduction is absent and the bundle says why. If the material term is
allocate, build, or the kernel, the reduction is still absent: those edits are out of
scope. Meeting both conditions is necessary. Taking the reduction stays ask-first under
the operational boundaries. The width-4 tracer takes no reduction. The decision waits
until widths 4, 6, and 8 have been measured.

**Rationale.** "Material term" has to be checkable at close, or Step 4b will invent a
threshold. Ranking the wrapper against the other reported components uses the component
split REQ-002 already requires. The 5 % kill stays the product-statement rule and is
independent of the unfrozen 10 % bar.

**Consequences.** A closeout diff review fails on a kernel, fusion, planner, AVX, or GPU
edit, on a second reduction, and on a reduction under the A4 kill or under a non-material
wrapper. Zero reductions is a successful close when the bundle records that the reduction
was not justified.

**Rejected alternatives.**
- Leave "material" undefined until implementation.
- Reduce when the wrapper leads at only one width.
- Treat a kernel or fusion edit as the allowed reduction.
- Take the reduction inside the width-4 tracer, before widths 6 and 8 exist.

**Upstream alignment:** REQ-005 · CAP-004 · QA-007 · goal G5.

## ADR-F5A-006 — Freeze the workload class and keep the 26-case matrix historical

**Status:** accepted.

**Context.** QA-007 requires frozen workloads (depth and schedule) at 4, 6, and 8 qubits.
The supported density anchor is generated HEA with U3 and CNOT, plus ordered local
depolarizing, amplitude damping, and phase damping. The frozen 26-case matrix and the
M3A disclosure 17/9/0 are historical. REQ-008 forbids using that matrix as this workload.

**Decision.** Widths 4, 6, and 8 stay in the outcome, including width 8 at at least 1000
calls. The workload class is that generated-HEA density anchor. One depth and one noise
schedule per width, and one parameter vector, are pinned in the slice that first counts
trials, before those trials. The same depth, schedule, and parameter vector apply to the
attribution routes once a descriptor of that anchor exists. The frozen 26-case matrix is
not loaded as the M-F5a workload. The archive and the disclosure 17/9/0 stay unchanged.
No optimizer loop and no VQA campaign produces the parameters.

**Rationale.** The class is fixed by the shipped density anchor. The numeric depth and
schedule are workload data, chosen once and then frozen, not a Layer 1 invention and not
a historical speedup matrix.

**Consequences.** Step 4a of the counting slice records the numbers before trials. A later
slice that cannot represent the same anchor as a planner descriptor hands back. It does
not substitute another circuit.

**Rejected alternatives.**
- Adopt the frozen 26-case matrix as the measured workload.
- Drop width 8 because the tracer starts at 4.
- Train parameters with an optimizer or a VQA campaign.
- Freeze a numeric depth in Layer 1 without the slice that owns the counted trial.

**Upstream alignment:** REQ-002, REQ-008 · CAP-007 · QA-007, QA-008 · goals G2, G8.

## ADR-F5A-007 — Record non-interference on rocky-local CI and move docs at close

**Status:** accepted.

**Context.** QA-009 keeps the state-vector default and requires zero state-vector
regressions at a code close. The semester record for a later M-F5a code close is
rocky-local Tester CI. ADR-F1A-006 still names `workflow_dispatch` and GitHub Actions.
That mismatch is N8. The scope brief defers N8 and forbids an M-F5a edit of ADR-F1A-006.
`ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` describe what is true now. The interop
lane does not exist yet, so those documents cannot honestly name it during planning.

**Decision.** M-F5a does not edit ADR-F1A-006 and does not treat a GitHub Actions green
run as its semester gate. A later code close records rocky-local Tester CI, states that
the default backend is still state vector, and states that density is still opt-in. No
new runtime dependency is added. `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` are
updated at milestone close, after the executable gates pass, and then name the interop
lane, the measured-entry inventory, the claim boundary, and that CI record. `ROADMAP.md`
is left for `create-product-roadmap` after `M_F5A_CLOSEOUT.md`. Planning does not edit
any of the three.

Rollback of the lane is deletion of the sibling tree and its artifacts. Rollback of a
reduction, if one was taken, is revert of that single binding or dispatch diff.

**Rationale.** Writing the lane into current-state docs before it exists would publish
intent as fact. Editing ADR-F1A-006 would reopen N8 inside a milestone that was told to
leave it deferred.

**Consequences.** REQ-007's CI evidence appears at the code close, not at Layer 1.
`tests/VQE/test_VQE.py` remains the existing state-vector pin a close must not break.
`CHANGE_CONTROL.md` is unnecessary unless a later slice proposes to change a frozen
contract in §4 of the detailed plan.

**Rejected alternatives.**
- Rewrite ADR-F1A-006 in this milestone.
- Use GitHub Actions green as the M-F5a close gate.
- Update `ARCHITECTURE_OVERVIEW.md` or `TECH_STACK.md` in Layer 1.
- Add a runtime dependency for timing or statistics.

**Upstream alignment:** REQ-007, REQ-008, REQ-009 · CAP-007 · QA-008, QA-009 · goals G7, G8, G9.

## ADR-F5A-008 — Keep the first slice to E-VQE at 4 qubits

**Status:** accepted.

**Context.** The slice tracer is the first vertical slice inside the milestone. SDD does
not pre-write the remaining slices. The plan's tracer intent is E-VQE at 4 qubits: the
harness, the 1000-call protocol, uncertainty, and the component split, with no reduction.

**Decision.** When the Tech Lead opens Step 4a, the only slice planned is `task-1`: E-VQE
at width 4. It proves the harness-only pair, at least 1000 warmed calls, the uncertainty
statement, and the four components. It takes no reduction, includes no attribution-only
route, and does not claim widths 6 or 8. Layer 1 creates no `task-1/` files. Step 4b for
that slice waits on an explicit code-ready verdict that restates the equal-work pair, the
inventory, the prohibition on \(O\) for attribution-only routes, and the prohibition on
kernel, fusion, and AVX edits. Widths 6 and 8, R-base, R-fused, R-strict, R-hybrid, and
any reduction are later slices, planned only after this one ships or its handback is
disposed.

**Rationale.** A thin energy-entry slice proves the pair, the lane, and the evidence
route before the milestone spends width-8 trials or attribution work. Planning those
slices now would freeze untested workload numbers across packages.

**Consequences.** A Step 4a draft that adds a route, a width, or a reduction to `task-1`
is returned to this ADR. The milestone outcome still requires widths 6 and 8 and the four
attribution routes; they are not dropped by starting at 4.

**Rejected alternatives.**
- Pre-write Layers 2–4 for every future slice in this step.
- Use the tracer to build only shared infrastructure, with no E-VQE measurement.
- Include the binding reduction in the tracer.
- Drop width 8 from the milestone because the tracer is width 4.

**Upstream alignment:** REQ-002, REQ-005 · CAP-004 · QA-007 · goals G2, G5.
