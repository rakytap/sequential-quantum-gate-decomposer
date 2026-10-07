# Task 4: four attribution routes, no overhead ratio
> **Status:** stamp draft · **Verdict:** awaiting APPROVE FOR STEP-4B · **Slice:** M-F5a task-4 ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 · CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-005, ADR-F5A-006 ·
> **Scope:** R-base, R-fused, R-strict, R-hybrid on the width-4 E-VQE anchor. No \(O\). No reduction ·
> **Gate:** uncommitted stage line `step-4b-authorized`. Developer not started. Width-4 rows do not close REQ-004 ·
> **RM:** ACCEPT 2026-10-07, upload `2026-10-07-mf5a-task4-align-accept_d0d6.md` (`c5ea0847…`). It does not authorize Step 4b ·
> **Tip:** `95d51da210955b0b3d8c9f26db725632caaf5b26` · binder `86eab037…` · three counted bundles stay ·
> **Pair, inventory, no-O rule, kernel/fusion/AVX boundary:** unchanged

## 1. Why this slice is the thinnest next row

E-VQE at 4, 6, and 8 is counted, and the Research Manager has interpreted that set.
The remaining counted inventory in ADR-F5A-001 is the four attribution routes, as one
set. They share one no-\(O\) rule, one descriptor handback, and the width-4 anchor
already pinned in task-1. One draft covers all four. A route-by-route split would
repeat that handback. Widths 6 and 8 for these routes, G-08, G-09, and the milestone
close stay later.

The authorized planning sentence is: "E-VQE equal-work interop overhead measured at
4/6/8 qubits under S-g Measure; one-sided 95 % UB on \(O\) is below 5 % at every width
(A4 false → CAP-004 hold-the-line); QA-007 10 % bar frozen and met on those cells."
It is not a route-row label, not a bundle phrase, and not "M-F5a complete".

| Candidate | Why it waits |
|-----------|----------------|
| Route rows at 6 and 8 | REQ-004 still needs those widths before milestone close. This slice does not close REQ-004 |
| R-oracle | E1 default-exclude. Task-1 already labels C++ `apply_to` from the E-VQE pair. This slice adds no diagnosis row |
| Binding or dispatch reduction | A4 has fired. CAP-004 is hold-the-line. The reduction is not made |
| G-08, G-09, Demo, full-milestone review | Close-time work. Demo stays No GO. No Opus review in this draft |
| Estimator change | S-g stays the E-VQE rule. This slice does not retune it |

## 2. What a route row is

Width-4 rows do not close REQ-004. Widths 6 and 8 remain for a later slice.
Each row calls one existing public entry on the descriptor built below:

| Id | Entry | Apply label |
|----|-------|-------------|
| R-base | `execute_partitioned_density` | C++ `NoisyCircuit.apply_to` |
| R-fused | `execute_partitioned_density_fused` | the executed apply |
| R-strict | `execute_partitioned_density_channel_native` | numpy Kraus |
| R-hybrid | `execute_partitioned_density_channel_native_hybrid` | the executed class |

The descriptor is the existing
`build_phase3_continuity_partition_descriptor_set` in
`squander/partitioning/noisy_descriptor.py`, which calls
`build_phase3_continuity_planner_surface` in
`squander/partitioning/noisy_planner_surface_builders.py`. Unpack
`vqe, _hamiltonian = build_task_evaluator(4)` from
`benchmarks/density_matrix/interop_profile/interop_lane.py` and pass `vqe`, not
the tuple. The surface is `vqe.describe_density_bridge()` (`source_type`
`generated_hea`), not a hand-built operation list. A structural probe at `6654ede4`
accepted it: parameters 18, operations 12, gates 9, noise 3, gate sequence U3, U3,
CNOT, U3, U3, CNOT, U3, U3, CNOT, and noise `local_depolarizing`,
`amplitude_damping`, `phase_damping`. `max_partition_qubits` stays the default 2.
The builder's default label `phase2_xxz_hea_q4_continuity` is not a second circuit.
If that call raises, or the counts disagree with the bridge, the slice hands back
and does not invent specs.

Throughput is nanoseconds per density-matrix entry. The numerator is the route
apply component, as task-1 §7. The divisor is `operation_count * 4^4` =
\(12\times 256 = 3072\). Orchestration time, \(T_\mathrm{public}\), and
\(T_\mathrm{lower}\) are not the numerator. The one-sided 95 % bound, mean plus
`1.644854 * s / sqrt(n)` with `ddof=1`, applies to orchestration time and to the
apply component only. S-g Measure stays the E-VQE \(O\) estimator and is not
retuned and not applied here. The row publishes no \(O\), no \(T_\mathrm{lower}\)
twin, and no QA-007 ratio. `milestone_counted` stays false. Warm-up, affinity, and
the single-thread launch stay the N-34 rule when a later counted run is separately
authorized. This draft runs no counted route trial.

## 3. Unsupported

- \(O\) on any attribution route. A lower-boundary twin. A QA-007 label on a route row.
- Closing REQ-004 from width-4 rows alone. Widths 6 and 8 remain.
- The binding or dispatch reduction. Kernel, fusion, AVX, or GPU edits. A C++ edit.
- R-oracle, unless a later diagnosis row carries the E1 sentence. This slice does not.
- Estimator change. Dropping samples. A new CPU mask. VQA. The 26-case matrix.
- Overwriting `interop_profile_bundle.json`, `interop_profile_bundle_w6.json`, or `interop_profile_bundle_w8.json`.
- "M-F5a complete", "attribution routes profiled", "reduction shipped", speedup, or GHA.
- Demo GO. A push or a pull request. N8. An edit of ADR-F1A-006.
- `milestone_counted=true`. Width-6 or width-8 route campaigns inside this slice.

## 4. Evidence matrix

This planning pass does not time the routes and does not write a bundle.

| Trace id | Evidence type | Command or gate | Expected result | Owner |
|----------|---------------|-----------------|-----------------|-------|
| REQ-001, REQ-004 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | a separate route fixture with \(O\), a QA-007 ratio, a reduction claim, or an R-oracle row without the E1 sentence fails. Four ids and no \(O\) pass. The three counted bundles are not the fixtures | DS-1 |
| REQ-004, REQ-008 | doc review | this mini-spec §2 | the four entries are the ADR-F5A-001 names. Unpack `vqe` from `build_task_evaluator(4)` before `build_phase3_continuity_partition_descriptor_set`. Width-4 rows do not close REQ-004 | DS-1 |
| REQ-005 | repo review | `git diff --exit-code 1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721 -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json` | empty. No reduction diff | DS-2 |
| REQ-006 | repo review | `git diff --exit-code 1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721 -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty | DS-2 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-009 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | after this uncommitted stamp, no `task-4/CLOSEOUT.md`: normal mode 0 errors, 1 warning `SLICE_MISSING_CLOSEOUT`, exit 0; `--strict` makes that finding the only error, exit 1. No waiver. No placeholder | DS-3 |

## 5. Verdict

**Stamp draft.** The stage line is `step-4b-authorized` in this uncommitted pack.
Reviewer APPROVE FOR STEP-4B has not been given. The Developer is not started.
C0 waits on that gate. RM ACCEPT `c5ea0847…` did not authorize Step 4b. Binder
`/tmp/rev-mf5a-t4-codeready/REVIEW.md` (`86eab037…`) at `95d51da2`. Width-4 rows
do not close REQ-004. No counted route run.

| Finding | Disposition |
|---------|-------------|
| A route row might be given an \(O\) | §3. No lower twin |
| The anchor might have no descriptor | §2. Existing builder; bridge counts 18 / 12 / 9 / 3. A raise is a handback |
| A4 might be read as permission to reduce | RM ALIGN. CAP-004 is hold-the-line |
| The authorized sentence might be copied into a bundle | §1. Planning text only |
| R-oracle might be added to fill the apply label | E1. Task-1 already labels `apply_to` |
| `INITIAL_REQUIREMENTS.md` still says `[confirm]` | The RM upload freezes the bar for the E-VQE cells. The file sentence waits for milestone close |
