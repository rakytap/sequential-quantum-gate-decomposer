# Task 2: E-VQE at 6 qubits, equal-work extension
> **Status:** Step 4a draft · **Verdict:** not-ready · **Slice:** M-F5a task-2 ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 · ADR-F5A-001…009 ·
> **Scope:** one E-VQE width-6 row on the task-1 harness. No width 8. No attribution routes. No reduction ·
> **Gate:** SDD stage `step-4a`. QA-007 stays `[confirm]`. Milestone not complete ·
> **RM:** this draft returns to the Tech Lead for Research Manager alignment before Step 4b ·
> **Tip:** `f524c2001410c6398f628c5030f95e08a6d9ece9` · task-1 bundle sha256 `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e` stays ·
> **Pair, inventory, no-O rule, kernel/fusion/AVX boundary:** unchanged

## 1. Why this slice is the thinnest next counted row

Task-1 shipped E-VQE at 4 qubits. The milestone outcome still needs widths 6 and 8,
four attribution-only routes, and a lawful QA-007 label once the bar is frozen.
This draft plans one of those rows.

Width 6 is the thinnest counted slice. It reuses the task-1 harness, the equal-work
pair, the 1000-pair protocol, the arithmetic-mean estimator, and the four components.
The new object is the same generated-HEA cell at `qbit_num=6`. A structural build at
this tip (no counted pairs, no bundle write) accepts that cell: `parameter_count` 30,
`operation_count` 18, `gate_count` 15, `noise_count` 3, Hamiltonian `nnz` 224.

Deferred, with the reason each is wider:

| Candidate | Why it waits |
|-----------|----------------|
| Width 8 | Same kind of extension, larger state (`4^8`). ADR-F5A-008 proved the pair before width-8 trials. Width 8 stays in the outcome |
| R-base, R-fused, R-strict, R-hybrid | A new measurement surface. ADR-F5A-006 hands back if the anchor has no planner descriptor. F-4 still defers R-fused apply labels |
| G-06 | Product-owner edit of `INITIAL_REQUIREMENTS.md`. Blocks a "QA-007 met" label and any trial that applies the 10 % bar. Does not block reporting `O` |
| G-08 | Current-state docs at milestone close (ADR-F5A-007) |
| G-09 | Rocky-local Tester CI record at a later code close. N8 stays deferred |
| A binding reduction | ADR-F5A-005 waits until widths 4, 6, and 8 exist. This slice takes none |

## 2. Frozen width-6 cell

Same construction as `task-1/TASK_1_MINI_SPEC.md` §2, with the width changed and the
integers that follow from that change. `max_inner_iterations` stays 4. It is the
task-1 config, not the qubit count.

| Field | Value |
|-------|--------|
| Entry | E-VQE: `Optimization_Problem` on `qgd_Variational_Quantum_Eigensolver_Base`, `backend="density_matrix"` |
| Width | 6 qubits. Width 8 is not built |
| Ansatz | `set_Ansatz("HEA")` then `Generate_Circuit(1, 1)` (`layers=1`, `inner_blocks=1`) |
| Hamiltonian | `generate_hamiltonian` on the line `[(0,1),(1,2),(2,3),(3,4),(4,5)]`. `nnz` 224 |
| Config | `max_inner_iterations=4`, `max_iterations=1`, `convergence_length=2` |
| Noise, in order | local depolarizing on target 0 after gate 0 at 0.1; amplitude damping on target 1 after gate 2 at 0.05; phase damping on target 0 after gate 4 at 0.07 |
| Structural pins | `parameter_count` 30, `gate_count` 15, `operation_count` 18, `noise_count` 3. A build that reports other integers stops |
| Parameters | 30 float64 values, `linspace(0.05, 0.05*30, 30)`, reused. No RNG seed. No optimizer |
| Lower call | existing harness-only `harness_density_lower_ns` into C++ `optimization_problem(Matrix_real&)` |
| Claim | `milestone_counted=false`. This row is not the 4/6/8 QA-007 verdict |
| Suite id | `interop_profile_task2_evqe_6q_v1` |

The stored noise records use the normalized `value` field (0.1, 0.05, 0.07). Gate
index 4 is inside `gate_count` 15. The 26-case matrix is not loaded. 17/9/0 is not
edited. No host golden is pinned for the 6-qubit energy (N-17 stays on the 4-qubit
value). The tight check is flag-off versus flag-on bit identity on this cell.

Throughput divisor is `operation_count * 4^6` = `18 * 4096` = **73728**. A divisor
of 3072, 4096, or `4^8` fails the width-6 row.

## 3. S-g decision (draft)

Single-call spikes of about 20–25 µs on CPU 0 pull the arithmetic mean of `O_i`
low on the width-4 row (mean 0.00711, median 0.0101). No sample was below −0.5.
S-g requires a decision before a width-6 or width-8 counted run. This draft
recommends one default and leaves acceptance to the Research Manager.

| Option | What it changes | Consequence |
|--------|-----------------|-------------|
| Measure (recommended) | Keep the arithmetic mean of `O_i`, drop nothing, keep the N-34 launch (`taskset -c 0`, four `*_NUM_THREADS=1`) | The one-sided bound stays the reported uncertainty. A spike count makes the bias visible. No estimator amendment and no new mask |
| Dispose | Drop, winsorize, or replace `O` with a median or trimmed mean | Ask-first estimator amendment. Contradicts ADR-F5A-004 (median-of-three is not the uncertainty statement) and the task-1 rule that no sample is dropped. Needs an ADR before any counted run |
| Affinity / thread controls | Quieter mask, or affinity and thread env before import (S-e) | Changes the launch relative to the width-4 row. S-e stays its own item |

**Recommended default: Measure.** Sub-time clocks stay `clock_gettime(CLOCK_MONOTONIC)`
reads, the same clock as `T_lower` (N-19 wording). The reported `O` is still the
arithmetic mean. The median is recorded and is not `O`. The bundle also records
`min_O`, `max_O`, and `spike_count_abs_wrapper_ns_above_20000`: the number of counted
samples with `abs(T_public - T_lower) > 20000` ns (20 µs, the low end of the stated
band). That count does not drop a sample, does not fail the row, and does not define
a core mean. The task-1 phrase "core mean" stays a reviewer observation. It is not a
second estimator.

The timer flag remains harness-only and single-threaded. Batched
`optimization_problem` is not timed (N-23). The race on the six fields stays open.

This recommendation is not accepted until the Research Manager aligns. Step 4b does
not start on Dispose or on a new mask unless this pack is revised.

## 4. Protocol that stays

- Paired, not interleaved. Even index: public then lower. Odd index: lower then public. 50 warm-up pairs discarded. 1000 counted pairs.
- Flag on for both sides of every warm-up pair and every counted pair. Read the six sub-times immediately after each lower call.
- Affinity: lowest CPU in the allowed mask, which the N-34 `taskset -c 0` launch makes CPU 0. Fail if affinity cannot be set. Do not move affinity before import.
- Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` are `1`.
- `O_i = (T_public,i - T_lower,i) / T_public,i`. One-sided 95 % upper bound: mean plus `1.644854 * s / sqrt(1000)`, `s` with `ddof=1`. Non-finite `O_i` or non-positive `T_public` fails the row.
- Four components partition `T_public`. The width-6 row fails when the six sub-times differ from the outer `T_lower` clock by more than 1 µs or 1 % of `T_lower`, whichever is greater (S-b, unchanged).
- QA-008 margin stays 0.02 absolute on a rerun mean of `O` (S-a). N-32: the width-6 validator gains `assert_mean_o_within_margin`. Fixture tests apply it. The first counted run has no prior row; a later regeneration uses the function. The margin is not the QA-007 bar.
- While G-06 is open, the row reports `O` and the bound and does not print "QA-007 met". It does not apply the 10 % bar.
- Canonical artifact: `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json`. The task-1 file `interop_profile_bundle.json` stays sha256 `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`. A width-6 write whose destination name is `interop_profile_bundle.json` exits nonzero and writes nothing. The no-arg pipeline remains the width-4 writer.

Counted close, when a later pass reaches it, follows ADR-F1A-009 because `clean_start` is required. This draft writes no `CLOSEOUT.md`.

## 5. Harness boundary

No new public Python energy symbol. No new C++ member, accessor, or clock. The
expected Step 4b diff stays in the interop lane, its validator, the two existing
interop test modules, and the width-6 artifact at close. A need to edit
`squander/src-cpp/` or `squander/VQA/` stops Step 4b and returns here. The equal-work
pair is unchanged: in-call allocate, build, and `validate_density_anchor_support` sit
on both sides. `T_public` is `perf_counter_ns` around `Optimization_Problem`.
`T_lower` is the existing `harness_density_lower_ns` clock.

## 6. Unsupported in this slice

- Width 8, attribution routes, R-oracle, and any claim that the 4/6/8 verdict is complete.
- A binding, dispatch, kernel, fusion, AVX, or GPU change. An optimizer loop or a VQA campaign.
- Dispose, a trimmed mean, or a quieter CPU mask.
- "QA-007 met", freezing the 10 % bar, and `milestone_counted=true`.
- Edits to `INITIAL_REQUIREMENTS.md`, `ARCHITECTURE_OVERVIEW.md`, `TECH_STACK.md`, ADR-F1A-006, the archive, `performance_evidence/`, or `benchmark_perf.py`.
- Overwriting or regenerating the task-1 bundle.
- A placeholder `task-2/CLOSEOUT.md`. The absent-closeout finding stays until a real close.

## 7. Keep-list

| Id | This draft |
|----|------------|
| N-41 | **Closed** by docs commit `f524c200`. C2 was committed before (e). That docs pass went to Reviewer before its commit. Later commits keep that order |
| S-g | Draft default Measure (§3). Open until the Research Manager aligns |
| N-46 | Deferred. Flag text stays exact for `libqgd.so`. The wrapper `.so` still adds `-DCPYTHON`. Pinning flags in the bundle is a lane change or a milestone-review deviation |
| N-16 | Deferred. No MSVC `clock_gettime` branch, until the first pull request into `master` |
| N-17 | Deferred. The 4-qubit golden stays host-pinned. This cell does not add a 6-qubit host golden |
| N-23 | Note restated in §3 (harness-only, single-threaded). The batch race stays open |
| N-32 | In this slice: `assert_mean_o_within_margin` and a fixture test. Closes when that test is green |
| N-35 | Deferred. Wider hand-edit depth stays open |
| N-36 | Deferred. Forbidden-path list gaps stay open. `git diff` remains the containment gate |
| N-37 | Deferred. Isolated B2 tests and a real-bundle fixture stay open |
| N-24 | Deferred. Interop tests stay outside the `density_matrix` marker |
| N-19 | Folded here: sub-time clocks are `clock_gettime` reads |
| N-3, N-6, N-10, N-14, N-21, N-25, N-27, N-28, N-29, N-31, N-40 | Unchanged carries |
| S-a, S-b | Unchanged (margin 0.02, partition tolerance above) |
| S-c, S-e | Deferred. No lag-1 or batch means. No affinity-before-import |
| S-d, S-f | Stay as task-1 left them (closed, disposed) |
| G-06, G-08, G-09 | Open. See §1 |

## 8. Adversarial critique

| Finding | Rank | Disposition |
|---------|------|-------------|
| The width-6 pair might need a C++ edit or a new energy API | blocking if ignored | §5 stop rule. The structural build succeeded. It was not a timed pair. A timed failure returns to planning |
| The width-6 writer could replace the task-1 bundle | blocking if ignored | §4 refusal. Sha256 pin stays the gate |
| Measure could be the wrong S-g call | blocking if ignored | §3 leaves acceptance with the Research Manager. This verdict is not-ready |
| The 10 % bar could be applied to width 6 alone | blocking if ignored | G-06. The row withholds "QA-007 met" and keeps `milestone_counted=false` |
| N-32 could turn this into a second project | non-blocking | One pure function on the same validator. Routes and width 8 stay out |
| QA-007 has no numeric fitness function while `[confirm]` | non-blocking | Live checks are the protocol, the bound's presence, and the label ban |
| Width 8 could be dropped because this slice is width 6 | blocking if ignored | §1. Width 8 stays in the outcome |

Nothing above authorizes Step 4b. The least testable in-scope check is still the
unfrozen 10 % bar, and this slice does not pretend to test it.

## 9. Evidence matrix

The counted width-6 command is specified for a later authorized run. This planning
pass does not run it and does not write the artifact.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-002, QA-007 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_density_matrix_backend_anchor_fixed_parameter_smoke -q` | existing 4-qubit anchor smoke passes; the width-6 row withholds "QA-007 met" and does not apply the 10 % bar | DS-1 |
| REQ-002, REQ-003, QA-009 | Aer exactness | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference -q` | unchanged 4-qubit Aer node passes. Width-6 tight check is flag-off versus flag-on bit identity in `tests/VQE/test_vqe_interop_harness.py` | DS-1 |
| REQ-001, REQ-004 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | a width-6 fixture fails `validate_interop_bundle`; a width-4 fixture fails `validate_interop_bundle_w6`; width 8, an attribution label, and "QA-007 met" fail both | DS-2 |
| REQ-006, QA-008 | repo review | `git diff --exit-code f524c2001410c6398f628c5030f95e08a6d9ece9 -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty. Task-1 bundle sha256 unchanged. Width-6 file is the sibling artifact | DS-2 |
| REQ-003, REQ-005 | doc review | this mini-spec §§4–6 | same harness; no new energy symbol; no reduction; no C++ edit in the planned diff | DS-1 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-008 | repo review | `git diff --exit-code f524c2001410c6398f628c5030f95e08a6d9ece9 -- docs/density_matrix_project/archive` | empty | DS-2 |
| REQ-009, QA-008 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | at `step-4a`, the only finding is `SLICE_MISSING_CLOSEOUT` for task-2, a warning in both modes; no waiver; no placeholder closeout | DS-3 |

## 10. Affected interfaces

Additive. `validation_pipeline.py` gains `--width`. No-arg stays width 4 and the
task-1 path. `--width 6` writes `interop_profile_bundle_w6.json` unless that write
would use the task-1 filename, in which case it writes nothing. The validator gains
`validate_interop_bundle_w6` and `assert_mean_o_within_margin`. `validate_interop_bundle`
stays width 4 and divisor 3072. No new runtime dependency. Rollback is deleting the
width-6 branch of the lane, the new validator entry points, and the width-6 artifact.

## 11. Verdict

**not-ready.** READY-FOR-REVIEW. The cell, the Measure default, and the deferred
set are specific enough for Research Manager alignment. They are not a code-ready
stamp and not Step 4b authorization. After alignment, a later writer pass may
restamp `step-4b-authorized` only if this contract is unchanged.
