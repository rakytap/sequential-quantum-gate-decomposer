# M-F5a slice 3 closeout — E-VQE 8-qubit interop row
> **Status:** C2-ready · **Date:** 2026-10-07 · **Work package:** task-3 ·
> **Scope:** counted (c) record. M-F5a is not complete ·
> **C0:** `c4f5df9be22ebd36e0eedd31b9329d57153d6d7f` on base `ced81815` · **C1:** `97d726e383e6846f7cefdccdb8783b45c2884208` ·
> **C2:** this commit · **(e):** pending after this write · **(g):** pending after C2 ·
> **Claim:** width-8 counted row. `milestone_counted=false` is lawful. QA-007 stays `[confirm]` ·
> **No push/PR**

## Summary

This closeout records the counted clean-start width-8 row. Tester (c) passed once
at C1 `97d726e383e6846f7cefdccdb8783b45c2884208`. Evidence is
`/tmp/mf5a-t3-counted/REPORT.md`. The bundle on disk is the Tester file. This
pass did not re-run it. C2 is this commit. (e) is pending after this write.
(g) is pending after C2. The file does not complete the milestone, freeze the
QA-007 bar, or claim "QA-007 met".

## Verdict

Counted row **PASS**. `validate_interop_bundle_w8` **OK**. `milestone_counted=false`
is **lawful** (below). QA-007 stays **withheld** (`[confirm]`, G-06 open). No A4
kill, no "A4 false", no CAP-004 hold-the-line label, and no reduction. The
milestone is **not** complete. C2 is this commit. (e) is pending after this write.

E-VQE equal-work interop cells measured at 4, 6, and 8 qubits under S-g Measure;
QA-007 bar still `[confirm]`/withheld; milestone not complete (attribution routes
and close gates remain). That sentence is allowed here and in checklist §13. It
is not in the bundle.

## `milestone_counted=false`

Lawful. It is not a pack bug and it does not block C2.

The width-8 contract requires the flag false. `TASK_3_MINI_SPEC.md` §2 sets the
claim to `milestone_counted=false` because the row is not the QA-007 verdict.
§5 lists `milestone_counted=true` as unsupported. ET-2 requires the validator to
reject any other value. DS-1 names the same flag. Task-1 and task-2 recorded
false for the same reason: one width is not the milestone verdict. The 4/6/8
set is now on disk, and the flag still means "this row is the milestone QA-007
verdict." That verdict waits on G-06 and on the Research Manager. ADR-F1A-009
governs the clean-start order. `clean_start` is true, which is what that order
needs. The flag is the claim boundary, not the clean-start bit.

## Chain

| Role | SHA |
|------|-----|
| C0 stamp | `c4f5df9be22ebd36e0eedd31b9329d57153d6d7f` |
| C1 implementation | `97d726e383e6846f7cefdccdb8783b45c2884208` |
| C2 | this commit |

`provenance.implementation_revision` is the C1 tip.

## (c) launch

Exit 0. Wall `real` 37.09 s. `affinity_cpu` is 0. All four `*_NUM_THREADS`
values are `"1"`. The command was run once.

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
taskset -c 0 env \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  conda run -n qgd --no-capture-output \
  python benchmarks/density_matrix/interop_profile/validation_pipeline.py --width 8
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
```

The width-8 command is the (c) launch. It was not re-run for this closeout.
The two `specs_check.sh` lines are the lint reproduce commands.

## Evidence pins

| Field | Value |
|-------|--------|
| Bundle | `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json` |
| sha256 | `1712dce97ae567460c9058c288c3a01879524dd3fb9cba71603a5f5785e1b21d` |
| Validator | `validate_interop_bundle_w8` OK |
| `clean_start` / `provenance_pass` | true / true |
| `dirty_paths` | `[]` |
| `implementation_revision` | `97d726e383e6846f7cefdccdb8783b45c2884208` |
| `milestone_counted` | false |
| Suite | `interop_profile_task3_evqe_8q_v1` |
| Counted samples | 1000 |
| `parameter_count` / `operation_count` / `nnz` / divisor | 42 / 24 / 1152 / 1572864 |
| `qa008.mean_O_absolute_margin` | 0.02 |
| Task-1 bundle | `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e` unchanged |
| Width-6 bundle | `5257bad23e9fef4afad3f7b8b61f84c7cebd2ef8f85d95a94a02cb794d139d7d` unchanged |

Harness `gate_count` is 21 and `noise_count` is 3. The bundle stores three
`workload.density_noise` entries. `libqgd.so` sha256
`615610c86ed1152a65b69b7c0ffa78812071c58297f530ee8dea856ae72ff947`.
The VQE wrapper `.so` sha256
`c9ed35dfba7811ff3a83c81dd2c518a943a7910fdd40b0c73eb916aee4480430`.
Compiler `c++ (GCC) 11.5.0`. The bundle still does not pin flags (N-46).

## Throughput and O

These figures are the counted row under Measure. They are not a QA-007 verdict
and not an A4 result. The 5 % A4 test needs the Research Manager to read every
width in {4, 6, 8}. No reduction was taken. A negative mean is lawful.

| Quantity | Record |
|----------|--------|
| `mean_O` | −0.000106 |
| `upper_bound_95_O` | 0.000744 |
| `median_O` | 0.000355 |
| `min_O` / `max_O` | −0.183 / 0.161 |
| spike count | 857 |
| samples | 1000 |
| mean ns per ρ-entry per operation | 10.582 |
| throughput upper bound | 10.592 |
| mean wrapper ns | 299.421 |

Stored values: `mean_O` `−0.000105901432951626`, `upper_bound_95_O`
`0.0007435776878628046`, `spike_count_abs_wrapper_ns_above_20000` `857`,
`throughput.mean_ns_per_op` `10.581956602096557`. The near-zero negative mean
and the saturated 20 µs count were pre-registered. No sample was dropped.
Throughput is within-run and is not a (g) gate.

## Independence

Tester confirmed independence before this closeout. The note is
`/tmp/mf5a-t3-counted/INDEPENDENCE_NOTE.md`, sha256
`992c9c0ee772c0f13a2c920dd05d3f520910137dc6a0abcef165cea9d59adf53`.

The cell is `build_task_evaluator(8)` and `Optimization_Problem` on `libqgd.so`.
The oracle is `Test_VQE._get_density_backend_aer_reference` after
`set_Optimized_Parameters`, in
`test_task3_evqe_8q_cell_pins_timer_identity_and_aer_oracle`. It builds its own
noise and its own trace. It does not read the cell's energy or ρ. Shared inputs
are the HEA circuit export, the §2 parameter vector, and the Hamiltonian. A
defect there would show on both sides. Flag-off and flag-on energies agreed
bitwise in that pytest: the flag only adds `clock_gettime` reads on the same
instance and vector, so the check shares the kernel and cannot see a kernel bug.
Aer agreement is not bitwise. The pytest PASSED and was not skipped
(`/tmp/mf5a-t3-counted/aer_w8_pytest.log`, exit 0, 0.71 s).

## Acceptance verdicts (vs slice contract)

ET checkboxes stay unchecked (N-42). Red-first rows stay unticked (N-79, N-112).

| Signal | Verdict | Evidence |
|--------|---------|----------|
| ET-1 pins, bit identity, Aer oracle | pass | C1 re-gate `/tmp/rev-mf5a-t3-c1-regate/REVIEW.md` (`30cced7d…`) §8: 79 passed, no skip; W8 Aer PASSED. `/tmp/rev-mf5a-t3-c2/REVIEW.md` (`fdb80f4e…`) §6.4 reproduces both. Tester log above, 0.71 s, from `/tmp` |
| ET-2 validator | pass | C1 re-gate `30cced7d…` §8: 79 passed, no skip. Binder `fdb80f4e…` §6.4 reproduces the lane. (c) `validate_interop_bundle_w8` OK |
| ET-3 containment | pass | C1 re-gate §6 and binder `fdb80f4e…` §6.9: both mini-spec §6 diff rows exit 0. Sibling sha256 values stay pinned |
| REQ-007, QA-009 | pass | C1 re-gate §8 and binder `fdb80f4e…` §6.4: `test_explicit_state_vector_matches_legacy_default` PASSED |
| Counted row | pass | `/tmp/mf5a-t3-counted/REPORT.md`; bundle sha256 above; `milestone_counted=false` |
| QA-007 | withheld | G-06 open. Not a "QA-007 met" claim. No A4 kill and no reduction |
| Lint | pass at this write | reproduce commands above; both modes clean once this file exists |
| Milestone | not complete | attribution routes and the bar freeze stay later |

## Notes folded here

N-100. Full RM sha256
`39808966def5055cbedaf7cdd08bffa465121d9c2c0644ec846f37252025d851`.
Upload name `2026-10-07-mf5a-task3-align-accept_a41b.md`.

N-113. Header sync. C0 is `c4f5df9b`. The stamp lines that said C0 was
uncommitted, and that the Developer had not started, are past tense in the
task-3 headers. RM ACCEPT did not flip the stage. The C0 stamp did.

N-98 stays held. This pack does not edit `_label_contains_forbidden_w6_phrase`
or `FORBIDDEN_W6_CLAIM_PHRASES`. Width 8 keeps its own matcher. No test is added
that "no A4 kill" fails at width 6.

N-99 stays held. `claim_boundary` is `task-3 tracer row; milestone_counted=false`.
`labels` stay per-row and do not carry the three-width sentence. No validator
token covers that sentence. Reviewer (e) checks the counted bundle.

N-111 stays open. The width-8 `density_noise` count is pinned from below: the
test rejects 2 entries. Four entries and a missing key reject in the validator
and are untested. Optional later: parametrize that test. Not this closeout.

N-112 stays open. The Developer fix-pass log is thin and a repo-root pytest
rewrote the ignored cache (N-80, N-110). Clean-start porcelain ignores that
file. The (c) launch was from a clean C1. Citing the re-gate mutation run does
not close the note.

N-104 through N-110 stay open as the C1 re-gate left them. N-78, N-79, and N-80
stay open. Direction stays NARROW. 17/9/0 and the 26-case matrix stay untouched.
No VQA.

## Remaining

M-F5a stays open. G-06, G-08, and G-09 stay open. The four attribution routes
and any reduction stay later. This file does not open the next slice. (g) is
pending after C2. No push and no pull request. QA-007 stays `[confirm]`.
