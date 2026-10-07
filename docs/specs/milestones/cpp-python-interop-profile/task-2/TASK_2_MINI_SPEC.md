# Task 2: E-VQE at 6 qubits, equal-work extension
> **Status:** task-2 Step 4b closed · **Slice:** M-F5a task-2 ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 · ADR-F5A-001…009 ·
> **Scope:** one E-VQE width-6 row on the task-1 harness. No width 8. No routes. No reduction ·
> **Gate:** SDD stage `step-4b-authorized`. QA-007 stays `[confirm]`. Milestone not complete ·
> **RM:** ACCEPT 2026-10-07 (upload `5c810dac…`). Counted (c) at C1 `d336472f` ·
> **Stamp:** C2 `6707892a`. (e) APPROVE. (g) PASS. No next slice ·
> **Tip:** C2 `6707892ae7aac912a7c9f30f91577f9191b74bff` · task-1 bundle `212f7038…` stays ·
> **Pair, inventory, no-O rule, kernel/fusion/AVX boundary:** unchanged

## 1. Why this slice is the thinnest next counted row

Task-1 shipped E-VQE at 4 qubits and proved the equal-work pair. ADR-F5A-008 only
sequenced that tracer before later widths (N-50). The milestone outcome still needs
widths 6 and 8, four attribution-only routes, and a lawful QA-007 label once the bar
is frozen. This pack plans the width-6 row.

Width 6 reuses the task-1 harness, the pair, the 1000-pair protocol, the arithmetic-mean
estimator, and the four components. The new object is the same generated-HEA cell at
`qbit_num=6`. A structural build (no counted pairs, no bundle write) accepts it:
`parameter_count` 30, `operation_count` 18, `gate_count` 15, `noise_count` 3,
Hamiltonian `nnz` 224.

| Candidate | Why it waits |
|-----------|----------------|
| Width 8 | Larger state (`4^8`). Stays in the outcome |
| R-base, R-fused, R-strict, R-hybrid | New measurement surface. ADR-F5A-006 hands back if the anchor has no planner descriptor |
| G-06 | Product-owner edit of `INITIAL_REQUIREMENTS.md`. Blocks "QA-007 met" and any trial that applies the 10 % bar |
| G-08 | Current-state docs at milestone close |
| G-09 | Rocky-local Tester CI at a later code close. N8 stays deferred |
| Reduction or A4 kill | ADR-F5A-005 needs every width in {4, 6, 8}. Widths 4 and 6 alone cannot kill or reduce |

## 2. Frozen width-6 cell

Same construction as `task-1/TASK_1_MINI_SPEC.md` §2, with the width changed and the
integers that follow. `max_inner_iterations` stays 4. It is the task-1 config, not the
qubit count. The width-6 cell strictly extends the width-4 cell (N-55): the first 12
operations, including noise at gates 0, 2, and 4, match the width-4 sequence, and the
first 18 parameters match the width-4 vector under the 0.05 step.

| Field | Value |
|-------|--------|
| Entry | E-VQE: `Optimization_Problem` on `qgd_Variational_Quantum_Eigensolver_Base`, `backend="density_matrix"` |
| Width | 6 qubits. Width 8 is not built |
| Ansatz | `set_Ansatz("HEA")` then `Generate_Circuit(1, 1)` (`layers=1`, `inner_blocks=1`) |
| Hamiltonian | `generate_hamiltonian` on the line `[(0,1),(1,2),(2,3),(3,4),(4,5)]`. `nnz` 224 |
| Config | `max_inner_iterations=4`, `max_iterations=1`, `convergence_length=2` |
| Noise, in order | local depolarizing on target 0 after gate 0 at 0.1; amplitude damping on target 1 after gate 2 at 0.05; phase damping on target 0 after gate 4 at 0.07 |
| Structural pins | `parameter_count` 30, `gate_count` 15, `operation_count` 18, `noise_count` 3. Another integer stops the row |
| Parameters | 30 float64 values, `linspace(0.05, 0.05*30, 30)`, reused. No RNG seed. No optimizer |
| Lower call | existing harness-only `harness_density_lower_ns` into C++ `optimization_problem(Matrix_real&)` |
| Claim | `milestone_counted=false`. This row is not the 4/6/8 QA-007 verdict |
| Suite id | `interop_profile_task2_evqe_6q_v1` |

Stored noise uses the normalized `value` field (0.1, 0.05, 0.07). Gate index 4 is
inside `gate_count` 15. The 26-case matrix is not loaded. 17/9/0 is not edited. No
6-qubit host golden is pinned (N-17). Flag-off versus flag-on bit identity is the
timer check. It shares the kernel and is not the independence oracle (§9).

Throughput divisor is `operation_count * 4^6` = `18 * 4096` = **73728**. A divisor
of 3072, 4096, or `4^8` fails the width-6 row. The throughput row carries its
one-sided 95 % upper bound, the same estimator as `O`.

## 3. S-g — Measure, with the cause corrected (W-2)

RM ACCEPT 2026-10-07 keeps Measure: arithmetic mean of `O_i`, drop nothing, N-34
launch (`taskset -c 0`, four `*_NUM_THREADS=1`). No trim, winsorize, or median swap.
Any later estimator change is frozen in requirements before a counted run uses it.

The inherited cause is not supported. On the frozen width-4 bundle, 2 of 1000 samples
have `abs(T_public - T_lower) > 20 µs`. Leaving them out moves the mean from 0.00711
to 0.00722. The mean–median gap (median 0.0101) comes from a broad 3–20 µs population
and from ratio asymmetry: a delay `s` on the lower side contributes `−s/T`, and the
same delay on the public side contributes only `s/(T+s)`. The task-1 phrase "core mean"
0.00978 is the mean of samples with `|O_i| ≤ 0.05` (n = 858). It is not an estimator.

`spike_count_abs_wrapper_ns_above_20000` stays the RM-accepted field. It counts
samples with `abs(T_public - T_lower) > 20000` ns. It is an observational tail summary
at an absolute threshold, recorded with that definition. It is not a filter, it does
not fail the row, and it is not comparable across widths. At width 4 the count is 2
per 1000. Uncounted width-6 runs saw 79–144 per 1000, which is ordinary tail
variability, not a spike class. `min_O`, `max_O`, and `median_O` expose the asymmetry.
The per-sample record is what a later freeze would use.

**Pre-registered width-6 outcomes.** A negative or near-zero mean `O` is lawful. Three
uncounted runs saw means −0.0011, −0.0012, and −0.0001. Those figures are not a pass
band. The counted row reports the measured mean. It is not clipped, not relabelled,
not replaced by the median, not a refusal, and not a reason to re-run. Step (c) runs
the counted command once. Samples with `O_i < −0.5` are kept. S-f stays disposed.
The lane does not refuse on that line. The one-sided bound is the reported
uncertainty. No validator check reads the sign of mean `O` or of the wrapper component.

Sub-time clocks stay `clock_gettime(CLOCK_MONOTONIC)` reads (N-19). The timer flag
stays harness-only and single-threaded. Batched `optimization_problem` is not timed
(N-23). The race stays open.

## 4. Protocol that stays

- Paired, not interleaved. Even index: public then lower. Odd index: lower then public. 50 warm-up pairs discarded. 1000 counted pairs.
- Flag on for both sides of every warm-up pair and every counted pair. Read the six sub-times immediately after each lower call.
- Affinity: lowest CPU in the allowed mask. The counted launch pins CPU 0. Fail if affinity cannot be set. Do not move affinity before import.
- Threads: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` are `1`.
- `O_i = (T_public,i - T_lower,i) / T_public,i`. One-sided 95 % upper bound: mean plus `1.644854 * s / sqrt(1000)`, `s` with `ddof=1`. Non-finite `O_i` or non-positive `T_public` fails the row. A negative finite mean does not.
- Four components partition `T_public`. Fail when the six sub-times differ from the outer `T_lower` clock by more than 1 µs or 1 % of `T_lower`, whichever is greater (S-b).
- QA-008 margin stays 0.02 absolute (S-a). `assert_mean_o_within_margin` uses `<=` on the absolute delta. The fixture pair is 0.0 and 0.02 (N-51). The margin is not the QA-007 bar. At width 6 it is about 16 standard errors, so a rerun cannot fail it (N-52). Record that when G-06 is frozen. S-a stays open.
- While G-06 is open, the row reports `O` and the bound and does not print "QA-007 met". It does not apply the 10 % bar.
- Canonical artifact: `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json`. The task-1 file stays sha256 `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`. The no-arg pipeline remains the width-4 writer.

The counted command, recorded verbatim in `provenance.command`, is:

```bash
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output \
  python benchmarks/density_matrix/interop_profile/validation_pipeline.py --width 6
```

`validate_interop_bundle_w6` fails when that command lacks `--width 6`. Filename
refusal is decided from the arguments before any pair runs. Width 6 whose resolved
name is `interop_profile_bundle.json`, and width 4 whose resolved name is
`interop_profile_bundle_w6.json`, exit nonzero and write nothing. Refusal tests use
a `tmp_path` copy. No test writes the committed task-1 bundle. Its bytes are proved
by the §9 `git diff` row, not by a permanent sha assertion inside pytest.

The counted close follows ADR-F1A-009. `task-2/CLOSEOUT.md` records that close.

## 5. Harness boundary

No new public Python energy symbol. No new C++ member, accessor, or clock. The
expected later diff stays in the interop lane, its validator, the two existing
interop test modules, and the width-6 artifact at close. A need to edit
`squander/src-cpp/` or `squander/VQA/` returns to planning. The equal-work pair is
unchanged. `T_public` is `perf_counter_ns` around `Optimization_Problem`. `T_lower`
is the existing `harness_density_lower_ns` clock.

## 6. Unsupported in this slice

- Width 8, attribution routes, R-oracle, and any claim that the 4/6/8 verdict is complete.
- A binding or dispatch reduction, and an A4 kill or a CAP-004 hold-the-line label, on widths 4 and 6 alone. A4 needs the one-sided bound below 5 % at every width in {4, 6, 8} (ADR-F5A-003, ADR-F5A-005).
- A kernel, fusion, AVX, or GPU change. An optimizer loop or a VQA campaign.
- Dispose, a trimmed mean, a median swap, a clip, or a quieter CPU mask.
- "QA-007 met", freezing the 10 % bar, and `milestone_counted=true`.
- Treating a negative mean `O`, or `O_i < −0.5`, as a failed row or as a reason to re-run.
- Edits to `INITIAL_REQUIREMENTS.md`, `ARCHITECTURE_OVERVIEW.md`, `TECH_STACK.md`, ADR-F1A-006, the archive, `performance_evidence/`, or `benchmark_perf.py`.
- Overwriting or regenerating the task-1 bundle.
- A placeholder `task-2/CLOSEOUT.md`.

## 7. Keep-list

| Id | This pack |
|----|-----------|
| N-41 | **Closed** by docs commit `f524c200` |
| S-g | Measure, RM ACCEPT 2026-10-07. The C0 stamp set `step-4b-authorized`. RM ACCEPT did not flip it |
| N-46 | Deferred. Flags stay unpinned in the bundle |
| N-16 | Deferred. No MSVC `clock_gettime` branch until the first pull request into `master` |
| N-17 | Deferred. No 6-qubit host golden |
| N-23 | Note restated in §3. The batch race stays open |
| N-32 | In this slice: `assert_mean_o_within_margin`, fixture 0.0 and 0.02, comparison `<=` |
| N-35, N-36, N-37, N-24 | Deferred, as in the task-1 carries |
| N-19 | Folded: sub-time clocks are `clock_gettime` reads |
| N-3, N-10, N-14 | Task-2 rows are in §9: containment diff, and the width-6 Aer oracle |
| N-6, N-21, N-25, N-27, N-28, N-29, N-31, N-40 | Unchanged carries |
| N-48 | Checklist stays within 250 lines. This fold compresses; it does not append a section |
| N-49 | Checklist §11 points at its own carry table for the N-41 close |
| N-50 | §1: task-1 proved the pair; ADR-F5A-008 sequenced the slices |
| N-51 | Margin fixture is the exact pair 0.0 and 0.02, with `<=` |
| N-52 | §4. S-a stays open. Record the width-6 margin note when G-06 is frozen |
| N-53 | No contract change. A shell where `conda` is only a function is a reproducibility note |
| N-54 | ET-1 still asserts the integers and `nnz` 224. The bridge-metadata node already runs at width 6 |
| N-55 | §2 prefix sentence |
| N-56 | S-c stays deferred. Heavy tails at width 6 do not open lag-1 or batch means here |
| S-a, S-b | Unchanged pins (0.02; 1 µs or 1 %) |
| S-c, S-e | Deferred |
| S-d, S-f | Stay as task-1 left them. S-f disposed: samples below −0.5 are kept |
| G-06, G-08, G-09 | Open |
| N-78 | Open. No test isolates the `provenance.command` equality clause |
| N-79 | Open. No Developer red-first log. Substitute: `/tmp/rev-mf5a-t2-step4b/logs/regate-probe-mutation.txt` |
| N-80 | Open. Fix-pass pytest from the repo root rewrote ignored files. Clean-start porcelain ignores them |

## 8. Adversarial critique

| Finding | Rank | Disposition |
|---------|------|-------------|
| The width-6 pair might need a C++ edit or a new energy API | blocking if ignored | §5 stop rule |
| The width-6 writer could replace the task-1 bundle | blocking if ignored | §4 refusal before any pair. §9 diff proves the committed bytes |
| A negative mean could be clipped or re-run during Step 4b | blocking if ignored | §3 pre-registration. No sign check |
| The 10 % bar or an A4 kill could be applied to widths 4 and 6 | blocking if ignored | §6. A fixture that claims either fails `validate_interop_bundle_w6` |
| RM ACCEPT could be read as a stage flip | blocking if ignored | The C0 stamp set the stage line to `step-4b-authorized`. RM ACCEPT did not |
| Width 8 could be dropped | blocking if ignored | §1 |

The C0 stamp authorized Step 4b. This critique did not. Width 8 stays in the outcome.

## 9. Evidence matrix

This planning pass does not run the counted command and does not write the artifact.
The smoke node is parametrized over widths 4, 6, 8, and 10.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-002, QA-007 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_density_matrix_backend_anchor_fixed_parameter_smoke -q` | smoke passes on its widths, including 6; the width-6 row withholds "QA-007 met" and does not apply the 10 % bar | DS-1 |
| REQ-002, REQ-003, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q` | §2 integers and `nnz` 224; flag-off versus flag-on bit identity; width-6 Aer oracle below passes, and the Tester shows it passed rather than skipped | DS-1 |
| REQ-002, REQ-003 | Aer exactness | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference -q` | frozen 4-qubit Aer node unchanged. The width-6 oracle is a new check in `tests/VQE/test_vqe_interop_harness.py` calling `Test_VQE._get_density_backend_aer_reference`. Bound: `\|ΔE\| ≤ 1e-12 + 1e-5·\|E_Aer\|` (about 2.8e-7 at this cell). Bit identity is not that oracle | DS-1 |
| REQ-001, REQ-004, REQ-005 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | `validate_interop_bundle_w6` applies the full width-4 check set with width-6 constants, including the throughput mean and its upper bound. A missing bound, a bad divisor, a spike count that disagrees with the samples, "QA-007 met", an A4-kill or hold-the-line or reduction claim, width 8, and an attribution label fail | DS-2 |
| REQ-005, REQ-008, QA-009 | repo review | `git diff --exit-code cffe2cab7da1f1533584f3972faacd6be3b89392 -- squander/src-cpp squander/VQA squander/partitioning tests/VQE/test_VQE.py benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json` | empty. This is the no-C++ containment row. It covers the VQE C++ tree (N-10) and the committed task-1 bundle | DS-2 |
| REQ-006, QA-008 | repo review | `git diff --exit-code cffe2cab7da1f1533584f3972faacd6be3b89392 -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty | DS-2 |
| REQ-003 | doc review | this mini-spec §§4–6 | same harness; no new energy symbol; refusal before pairs; `provenance.command` carries `--width 6` | DS-1 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-008 | repo review | `git diff --exit-code cffe2cab7da1f1533584f3972faacd6be3b89392 -- docs/density_matrix_project/archive` | empty | DS-2 |
| REQ-009, QA-008 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | with `task-2/CLOSEOUT.md`: both modes exit 0 and are clean of `SLICE_MISSING_CLOSEOUT`; no waiver; QA-007 stays `[confirm]` | DS-3 |

## 10. Affected interfaces

Additive. `validation_pipeline.py` gains `--width`. No-arg stays width 4 and the
task-1 path. `--width 6` writes `interop_profile_bundle_w6.json` unless the resolved
filename is the task-1 name, in which case it returns before any pair and writes
nothing. The validator gains `validate_interop_bundle_w6` and
`assert_mean_o_within_margin`. `validate_interop_bundle` stays width 4 and divisor
3072. No new runtime dependency. Rollback deletes the width-6 branch, the new
validator entry points, and the width-6 artifact.

## 11. Verdict

**Closed through (g).** C2 is `6707892a`. SDD stage stays `step-4b-authorized`.
(e) is APPROVE. (g) is PASS. QA-007 stays `[confirm]`. No A4 kill, hold-the-line
label, or reduction. `milestone_counted` stays false. The milestone is not
complete. This file does not open the next slice.
