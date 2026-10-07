# Task 3: E-VQE at 8 qubits, equal-work extension
> **Status:** Step 4a draft · **Verdict:** not-ready · **Slice:** M-F5a task-3 ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 · ADR-F5A-001…009 ·
> **Scope:** one E-VQE width-8 row on the task-1/2 harness. No routes. No reduction ·
> **Gate:** SDD stage `step-4a`. QA-007 stays `[confirm]`. Milestone not complete ·
> **RM:** this draft awaits ALIGN ACCEPT. Acceptance is not this file. No Step 4b ·
> **Tip:** `5d93ca07a75154e1e909edcc66d432aea9460889` · w6 and task-1 bundles stay ·
> **Pair, inventory, no-O rule, kernel/fusion/AVX boundary:** unchanged

## 1. Why this slice is the thinnest next counted row

Task-1 and task-2 counted E-VQE at 4 and 6 qubits. The roadmap outcome still
requires E-VQE at 8, the four attribution routes, and a lawful QA-007 label once
the bar is frozen. ADR-F5A-006 keeps width 8 in the outcome at 1000 calls.

Width 8 is the thinnest counted slice that advances that outcome. It reuses the
same harness, equal-work pair, 1000-pair protocol, and Measure estimator. A
structural build at this tip (no counted pairs, no bundle write) accepts the
cell: `parameter_count` 42, `gate_count` 21, `operation_count` 24, `noise_count`
3, Hamiltonian `nnz` 1152. Gate index 4 is inside `gate_count` 21.

| Candidate | Why it waits |
|-----------|----------------|
| R-base, R-fused, R-strict, R-hybrid | A new surface. No equal-work pair. ADR-F5A-006 hands back if the anchor has no planner descriptor. They do not complete the 4/6/8 energy set |
| G-06 | Product-owner edit of `INITIAL_REQUIREMENTS.md`. Blocks "QA-007 met" and any trial that applies the 10 % bar. Does not block reporting `O` |
| G-08 | Current-state docs at milestone close |
| G-09 | Rocky-local Tester CI at a later code close. N8 stays deferred |
| Reduction or an A4 kill | ADR-F5A-005 needs the wrapper ranked at every width in {4, 6, 8}, and A4 needs the bound below 5 % at every one of those widths. This slice does not decide either |
| N-78, N-79, N-80 | Open test notes. They do not add a counted width |

## 2. Frozen width-8 cell

Same construction as the width-6 cell, with the width changed and the integers
that follow. `max_inner_iterations` stays 4.

| Field | Value |
|-------|--------|
| Entry | E-VQE: `Optimization_Problem`, `backend="density_matrix"` |
| Width | 8 qubits |
| Ansatz | `set_Ansatz("HEA")` then `Generate_Circuit(1, 1)` |
| Hamiltonian | `generate_hamiltonian` on the line through `(6,7)`. `nnz` 1152 |
| Config | `max_inner_iterations=4`, `max_iterations=1`, `convergence_length=2` |
| Noise, in order | local depolarizing on target 0 after gate 0 at 0.1; amplitude damping on target 1 after gate 2 at 0.05; phase damping on target 0 after gate 4 at 0.07 |
| Structural pins | `parameter_count` 42, `gate_count` 21, `operation_count` 24, `noise_count` 3. Another integer stops the row |
| Parameters | 42 float64 values, `linspace(0.05, 0.05*42, 42)`, reused. No optimizer |
| Lower call | existing `harness_density_lower_ns` into `optimization_problem(Matrix_real&)` |
| Claim | `milestone_counted=false`. Not a "QA-007 met" claim and not an A4 decision |
| Suite id | `interop_profile_task3_evqe_8q_v1` |

Throughput divisor is `operation_count * 4^8` = `24 * 65536` = **1572864**. A
divisor of 3072, 73728, or 65536 fails the row.

No 8-qubit host golden. The tight timer check is flag-off versus flag-on bit
identity. It shares the kernel. The oracle is Aer, after
`set_Optimized_Parameters` with this vector, bound
`|ΔE| ≤ 1e-12 + 1e-5·|E_Aer|`. The Tester shows that check passed rather than
skipped.

## 3. S-g — carry Measure

RM ACCEPT 2026-10-07 keeps Measure for the counted rows already shipped:
arithmetic mean of `O_i`, drop nothing, N-34 launch (`taskset -c 0`, four
`*_NUM_THREADS=1`). This draft carries that default to width 8 and asks the
Research Manager to confirm the carry. It does not open a new estimator.

The 20 µs count stays observational. It is not a filter and not comparable to
the width-4 count of 2 per 1000 or the width-6 count of 73 per 1000. `min_O`,
`max_O`, and `median_O` stay in the row. A negative or near-zero mean `O` is
lawful, as is `O_i` < −0.5. Neither is a pass band, a clip, a median swap, a
refusal, or a reason to re-run. Step (c), when a later authorization reaches
it, runs the command once. No validator reads the sign of mean `O`.

## 4. Protocol

Paired, not interleaved. 50 warm-up pairs discarded. 1000 counted pairs. Flag
on for both sides. Same partition tolerance: 1 µs or 1 % of `T_lower`,
whichever is greater. QA-008 margin stays 0.02 absolute (S-a). It is not the
QA-007 bar. While G-06 is open the row reports `O` and the bound and does not
print "QA-007 met". It does not apply the 10 % bar and does not apply the 5 %
A4 test.

Canonical artifact:
`benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json`.
The task-1 file stays sha256
`212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`. The width-6
file stays sha256
`5257bad23e9fef4afad3f7b8b61f84c7cebd2ef8f85d95a94a02cb794d139d7d`. Filename
refusal is decided before any pair. A width-8 write aimed at either committed
name exits nonzero and writes nothing.

The counted command, as one line in `provenance.command`, is:

```bash
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --width 8
```

`validate_interop_bundle_w8` fails when that command is not this line. Refusal
tests use `tmp_path`. No test writes a committed bundle. This draft writes no
`CLOSEOUT.md`.

## 5. Unsupported

- Attribution routes, R-oracle, and a claim that the milestone QA-007 verdict is complete.
- A reduction, an A4 kill, or a CAP-004 hold-the-line label. Tokens in `claim_boundary` or `labels` that say "A4 kill", "hold-the-line", or "reduction taken" fail. The words "no reduction" in a lawful claim must still pass.
- A kernel, fusion, AVX, or GPU change. A C++ edit. A new public energy symbol.
- Dispose, a trimmed mean, a median swap, or a quieter CPU mask.
- "QA-007 met", freezing the 10 % bar, and `milestone_counted=true`.
- Overwriting either committed bundle.
- N-78, N-79, and N-80 as work inside this slice.

## 6. Evidence matrix

This planning pass does not run the counted command and does not write the artifact.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-002, QA-007 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q` | width-8 pins 42 / 21 / 24 / `nnz` 1152; flag-off versus flag-on bit identity; Aer oracle after `set_Optimized_Parameters` passes and is not skipped; the row withholds "QA-007 met" | DS-1 |
| REQ-001, REQ-004, REQ-005 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | `validate_interop_bundle_w8` is the full width-6 check set with width-8 constants, including the throughput mean and its upper bound. A bad divisor, a missing bound, "QA-007 met", an A4-kill or hold-the-line or "reduction taken" claim, and an attribution label fail. A lawful "no reduction" claim passes | DS-2 |
| REQ-006, QA-008 | repo review | `git diff --exit-code 5d93ca07a75154e1e909edcc66d432aea9460889 -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty | DS-2 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-008 | repo review | `git diff --exit-code 5d93ca07a75154e1e909edcc66d432aea9460889 -- docs/density_matrix_project/archive squander/src-cpp squander/VQA squander/partitioning` | empty at this draft | DS-2 |
| REQ-009, QA-008 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | at `step-4a`, the only finding is `SLICE_MISSING_CLOSEOUT` for task-3, a warning in normal mode and the sole `--strict` error; no waiver; no placeholder closeout | DS-3 |

## 7. Verdict

**not-ready.** READY-FOR-REVIEW. SDD stage stays `step-4a`. This draft awaits
Research Manager alignment. It does not authorize Step 4b, a Developer, or a
counted width-8 run.
