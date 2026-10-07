# Task 4: four attribution routes, no overhead ratio
> **Status:** Step 4a draft · **Verdict:** not-ready · **Slice:** M-F5a task-4 ·
> **Traces:** REQ-001, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008 · CAP-004, CAP-007 · QA-008, QA-009 · ADR-F5A-001, ADR-F5A-004, ADR-F5A-005, ADR-F5A-006 ·
> **Scope:** R-base, R-fused, R-strict, R-hybrid on the width-4 E-VQE anchor. No \(O\). No reduction ·
> **Gate:** SDD stage `step-4a`. QA-007 stays met only on the counted E-VQE cells. Milestone not complete ·
> **RM:** ALIGN 2026-10-07. A4 is false. CAP-004 is hold-the-line. The reduction is not made ·
> **Tip:** `1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721` · three counted bundles stay ·
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
| Route rows at 6 and 8 | The width-4 anchor is enough to land the four ids. Repeating widths is a later slice |
| R-oracle | E1 default-exclude. Task-1 already labels C++ `apply_to` from the E-VQE pair. This slice adds no diagnosis row |
| Binding or dispatch reduction | A4 has fired. CAP-004 is hold-the-line. The reduction is not made |
| G-08, G-09, Demo, full-milestone review | Close-time work. Demo stays No GO. No Opus review in this draft |
| Estimator change | S-g stays the E-VQE rule. This slice does not retune it |

## 2. What a route row is

Each row calls one existing public entry on a descriptor of the task-1 width-4 HEA
anchor, if that descriptor already exists:

| Id | Entry | Apply label |
|----|-------|-------------|
| R-base | `execute_partitioned_density` | C++ `NoisyCircuit.apply_to` |
| R-fused | `execute_partitioned_density_fused` | the executed apply |
| R-strict | `execute_partitioned_density_channel_native` | numpy Kraus |
| R-hybrid | `execute_partitioned_density_channel_native_hybrid` | the executed class |

The row publishes orchestration time, the apply component, throughput on \(4^n\) times
the operations that apply executes, and the same one-sided 95 % bound used for those
times. It publishes no \(O\), no \(T_\mathrm{lower}\) twin, and no QA-007 ratio.
`milestone_counted` stays false. Warm-up, affinity, and the single-thread launch stay
the N-34 rule when a later counted run is separately authorized. This draft runs no
counted route trial.

If the existing planner cannot represent that anchor, the slice hands back. It does
not invent a second workload, a new public energy API, or an \(O\) for these entries.

## 3. Unsupported

- \(O\) on any attribution route. A lower-boundary twin. A QA-007 label on a route row.
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
| REQ-001, REQ-004 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_vqe_interop_bundle_validation.py -q` | a route fixture with \(O\), a QA-007 ratio, a reduction claim, or an R-oracle row without the E1 sentence fails. Four ids and no \(O\) pass | DS-1 |
| REQ-004, REQ-008 | doc review | this mini-spec §2 | the four entries are the ADR-F5A-001 names. A missing descriptor is a handback, not a new circuit | DS-1 |
| REQ-005 | repo review | `git diff --exit-code 1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721 -- benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json` | empty. No reduction diff | DS-2 |
| REQ-006 | repo review | `git diff --exit-code 1eb54ddbcf6f7bfdf82c2b7fc75a86687002b721 -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` | empty | DS-2 |
| REQ-007, QA-009 | fast pytest | `conda run -n qgd --no-capture-output pytest tests/VQE/test_VQE.py::Test_VQE::test_explicit_state_vector_matches_legacy_default -q` | state-vector default still matches | DS-3 |
| REQ-009 | spec lint | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile` and the same command with `--strict` | at `step-4a` with no `task-4/CLOSEOUT.md`: normal mode 0 errors, 1 warning `SLICE_MISSING_CLOSEOUT`, exit 0; `--strict` keeps that warning, 0 errors, exit 0. No waiver. No placeholder | DS-3 |

## 5. Verdict

**not-ready.** READY-FOR-TL-RM-CONSULT. SDD stage stays `step-4a`. This draft does
not authorize Step 4b, a Developer, a Tester counted run, or a stamp.

The descriptor mapping is still the detailed-plan §11 handback. This pack does not
prove an existing planner descriptor for the width-4 anchor. That proof, or an
explicit handback, is required before a code-ready verdict.

| Finding | Disposition |
|---------|-------------|
| A route row might be given an \(O\) | §3. No lower twin |
| The anchor might have no descriptor | §2 handback. No second workload |
| A4 might be read as permission to reduce | RM ALIGN. CAP-004 is hold-the-line |
| The authorized sentence might be copied into a bundle | §1. Planning text only |
| R-oracle might be added to fill the apply label | E1. Task-1 already labels `apply_to` |
| `INITIAL_REQUIREMENTS.md` still says `[confirm]` | The RM upload freezes the bar for the E-VQE cells. The file sentence waits for milestone close |
