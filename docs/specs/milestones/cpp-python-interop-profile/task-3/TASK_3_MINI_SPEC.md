# Task 3: E-VQE at 8 qubits, equal-work extension
> **Status:** Step 4a draft · **Verdict:** not-ready · **Slice:** M-F5a task-3 ·
> **Traces:** REQ-001…009 · CAP-004, CAP-007 · QA-007, QA-008, QA-009 · ADR-F5A-001…009 ·
> **Scope:** one E-VQE width-8 row on the task-1/2 harness. No routes. No reduction ·
> **Gate:** SDD stage `step-4a`. QA-007 stays `[confirm]`. Milestone not complete ·
> **RM:** ACCEPT 2026-10-07, upload `2026-10-07-mf5a-task3-align-accept_a41b.md` (`39808966…`). It does not flip the stage ·
> **Tip:** `1058d7029e314e09896151319e697df12dc92534` · pins read at parent `5d93ca07` · bundles stay ·
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
| Claim | `milestone_counted=false`. The flag means this row is the milestone QA-007 verdict. That verdict waits on G-06 and on RM, so the flag stays false even though width 8 completes the width set. `claim_boundary` and `labels` stay per-row, for example "task-3 tracer row; milestone_counted=false". They do not carry the three-width sentence |
| Suite id | `interop_profile_task3_evqe_8q_v1` |

Throughput divisor is `operation_count * 4^8` = `24 * 65536` = **1572864**. A
divisor of 3072, 73728, or 65536 fails the row.

No 8-qubit host golden. The tight timer check is flag-off versus flag-on bit
identity. It shares the kernel. The oracle is Aer, after
`set_Optimized_Parameters` with this vector, bound
`|ΔE| ≤ 1e-12 + 1e-5·|E_Aer|`. The Tester shows that check passed rather than
skipped.

## 3. S-g — Measure carries to width 8

RM ACCEPT 2026-10-07 (`39808966…`) carries Measure to width 8: arithmetic mean
of `O_i`, drop nothing, N-34 launch (`taskset -c 0`, four `*_NUM_THREADS=1`).
No trim, winsorize, median swap, clip, or quieter CPU mask. An estimator change
is frozen in requirements first and then applies to the whole 4/6/8 set. It is
not introduced at width 8.

These facts are information, not a pass band. A mean `O` near zero, of either
sign, is the expected outcome. Uncounted means were +0.00029 to +0.00113, with
SE 0.00044–0.00076. Medians were −0.00005 to +0.00027. The 20 µs count is
expected near saturation, 891–920 per 1000. At this width the threshold is
about 0.1 % of `T_public`. A high count is not a spike class, not an anomaly,
not comparable to the width-4 count of 2 per 1000 or the width-6 count of 73
per 1000, and not a reason to re-run. `T_public` and throughput are host-state
sensitive: between uncounted runs they moved from 16.8 to 25.8 ms and from 12.8
to 16.4 ns per entry per operation. ρ is 1 MiB, larger than the per-core L2.
The throughput bound is within-run only. A between-run throughput change is not
a failure. The QA-008 margin applies to mean `O` only. A negative wrapper mean
is lawful, as are samples below −0.5. No validator reads the sign of mean `O`
or of the wrapper. A validator failure at (c) — a non-finite sample, a
non-positive `T_public`, a partition mismatch, or a wrong integer — is a
stop-and-report to the Tech Lead. It is not a lawful row and not a silent
re-run. Step (c) runs the command once.

## 4. Protocol

`task-2/TASK_2_MINI_SPEC.md` §4 is carried unchanged at width 8, with these pins
restated. Even index: public then lower. Odd index: lower then public. 50
warm-up pairs discarded. 1000 counted pairs. Flag on for both sides. Affinity
is the lowest allowed CPU, pinned in-process, and the row fails if the pin
cannot be set. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
and `NUMEXPR_NUM_THREADS` are `1`. `O_i = (T_public - T_lower) / T_public`. The
one-sided 95 % bound is mean plus `1.644854 * s / sqrt(1000)`, `s` with
`ddof=1`. A non-finite `O_i` or a non-positive `T_public` fails the row. A
negative finite mean does not. The partition tolerance is 1 µs or 1 % of
`T_lower`, whichever is greater (S-b). The QA-008 margin is 0.02 absolute on
mean `O` (S-a). It is not the QA-007 bar and not a throughput gate. While G-06
is open the row reports `O` and the bound and does not print "QA-007 met". It
does not apply the 10 % bar and does not apply the 5 % A4 test.

The counted close follows ADR-F1A-009 in the `two-commit-close.md` order:
(a) Reviewer implementation review; C1; (c) once from a clean C1; the Tester
independence note, naming the width-8 Aer oracle against the cell; (d)
`task-3/CLOSEOUT.md`; (e) Reviewer evidence review; C2; (g). An optional
planning C0 may come first. At (g), regenerate with `--width 8`, restore the
committed width-8 bundle, and check `|Δ mean O|` ≤ 0.02 with
`assert_mean_o_within_margin`. Throughput is not a (g) gate.

Canonical artifact:
`benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json`.
The task-1 file stays sha256
`212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`. The width-6
file stays sha256
`5257bad23e9fef4afad3f7b8b61f84c7cebd2ef8f85d95a94a02cb794d139d7d`. Width 8
writes only `interop_profile_bundle_w8.json` and refuses both committed names
before any pair. Widths 4 and 6 refuse `interop_profile_bundle_w8.json`.
`--width 8` calls `run_interop_row(qbit_num=8)` and then
`validate_interop_bundle_w8`. It does not use the live map that sends every
non-6 width to 4. Smoke runs use `--output /tmp/<dir>/interop_profile_bundle_w8.json`.

The counted command, as one line in `provenance.command`, is:

```bash
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --width 8
```

`validate_interop_bundle_w8` fails when `provenance.command` is not this line.
The `--width` check is parameterized by profile. It is not a hard-coded
`"--width 6"` test. Width-6 behavior stays as it is. Refusal tests use
`tmp_path`. No test writes a committed bundle. This draft writes no
`CLOSEOUT.md`.

## 5. Unsupported

- Direction stays NARROW. 17/9/0 and the 26-case matrix stay untouched. No optimizer loop and no VQA campaign. No push and no pull request. N8 stays deferred.
- Attribution routes and R-oracle.
- Until the Research Manager interprets the counted 4/6/8 set, these sentences are refused: "QA-007 met"; "A4 false / CAP-004 hold-the-line"; "interop reduction justified or shipped"; "M-F5a complete."
- Tokens that fail a width-8 bundle, matched without case and without rejecting a leading "no ": "A4 kill", "A4 false", "hold-the-line", "reduction taken", "reduction justified", "reduction shipped", and "M-F5a complete", plus a full-bundle scan for "QA-007 met". A lawful claim may say "no reduction taken", "milestone not complete", and "QA-007 withheld".
- A kernel, fusion, AVX, or GPU change. A C++ edit. A new public energy symbol.
- Dispose, a trimmed mean, a median swap, a clip, or a quieter CPU mask.
- `milestone_counted=true` on this row.
- Overwriting either committed bundle.
- N-78, N-79, and N-80 as work inside this slice.
- The next RM gate. After the counted width-8 row lands and (g) passes, the 4/6/8 bundles return to the Research Manager for the A4 evaluation (upper bound below 5 % at every width) and for whether to freeze the 10 % bar. That gate is outside this slice.

The allowed sentence, only in `task-3/CLOSEOUT.md` and checklist §13, and only
at C2 or later if the counted row lands clean, is: "E-VQE equal-work interop
cells measured at 4, 6, and 8 qubits under S-g Measure; QA-007 bar still
`[confirm]`/withheld; milestone not complete (attribution routes and close
gates remain)." It does not appear in the bundle, as a demo flag, or in the
current-state docs.

## 6. Evidence matrix

This planning pass does not run the counted command and does not write the artifact.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-002, REQ-003, QA-007 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_harness.py -q -rs` | width-8 pins 42 / 21 / 24 / `nnz` 1152; no `harness_*` attribute and no new energy method; flag-off versus flag-on bit identity; Aer oracle after `set_Optimized_Parameters` passes and is not skipped | DS-1 |
| REQ-001, REQ-004, REQ-005 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | `validate_interop_bundle_w8` is the full width-6 check set with width-8 constants, including throughput mean and upper bound, `parameter_count` 42, `nnz` 1152, and three noise entries. Missing throughput mean or bound, a spike-count mismatch, `qbit_num` 6, a width-6 integer set, a bad divisor, an attribution label, a width-4 or width-6 provenance command, and the §5 refuse tokens fail. "no reduction taken", "milestone not complete", and "QA-007 withheld" pass. A negative mean still validates | DS-2 |
| REQ-006, QA-008 | repo review | `git diff --exit-code 5d93ca07a75154e1e909edcc66d432aea9460889 -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty | DS-2 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-003, REQ-008 | repo review | `git diff --exit-code 5d93ca07a75154e1e909edcc66d432aea9460889 -- docs/density_matrix_project/archive squander/src-cpp squander/VQA squander/partitioning tests/VQE/test_VQE.py` | empty at this draft. `tests/VQE/test_VQE.py` holds the Aer helper and the frozen 4-qubit node | DS-2 |
| REQ-009, QA-008 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | at `step-4a` with no closeout: normal mode 0 errors, 1 warning `SLICE_MISSING_CLOSEOUT` for task-3, exit 0; `--strict` keeps that warning, 0 errors, exit 0. After a later stage flip, `--strict` exits 1 on that finding until a real closeout. After a real closeout, both modes are clean. No waiver. No placeholder | DS-3 |

## 7. Verdict

**not-ready.** READY-FOR-CODE-READY-REVIEW. SDD stage stays `step-4a`. RM ACCEPT
`39808966…` is recorded and does not flip the stage. This fold does not
authorize Step 4b, a Developer, a counted width-8 run, or a C0 stamp.

A later move of the stage line to `step-4b-authorized` requires all four of
these: this RM ACCEPT; W-1…W-7 folded with pins, Measure, the deferred set, the
pair, the inventory, the no-O rule, and the kernel/fusion/AVX boundary
unchanged; a Reviewer code-ready writer re-gate APPROVE; and a separate
planning-role stamp-only pass, then the narrow Reviewer stamp check.

| Finding | Disposition |
|---------|-------------|
| Width 8 might need C++ | §5 stop rule. The structural build needed none |
| Throughput might be read as comparable across hosts | §3. The bound is within-run. QA-008 applies to mean `O` only |
| "Another integer stops the row" might stay unenforced | ET-2 requires parameter count 42, nnz 1152, operation count 24, and three noise entries |
| `--width 8` might be mapped to width 4 | ET-2 forces `qbit_num=8` and a dispatch test that runs no pairs |
| REQ-003 and `tests/VQE/test_VQE.py` might lack evidence | §6 rows |
| The 20 µs count might be read as an anomaly | §3. Saturation is pre-registered |
