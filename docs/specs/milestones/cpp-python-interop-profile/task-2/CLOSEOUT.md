# M-F5a slice 2 closeout — E-VQE 6-qubit interop row
> **Status:** C2-ready · **Date:** 2026-10-07 · **Work package:** task-2 ·
> **Scope:** counted (c) record. M-F5a is not complete ·
> **C0:** `388a5e5ff7775c496e144edee5b76be85f7caf54` on base `c32b365e` · **C1:** `d336472fa59c514c9eb3a5e6001daf343a4bbb66` ·
> **C2:** this commit · **(e):** pending after this write · **(g):** pending after C2 ·
> **Claim:** width-6 counted row. `milestone_counted=false` is lawful. QA-007 stays `[confirm]` ·
> **No push/PR**

## Summary

This closeout records the counted clean-start width-6 row. Tester (c) passed once
at C1 `d336472fa59c514c9eb3a5e6001daf343a4bbb66`. Evidence is
`/tmp/mf5a-t2-counted/REPORT.md`. The bundle on disk is the Tester file. This
pass did not re-run it. C2 is this commit. (e) is pending after this write.
(g) is pending after C2. The file does not complete the milestone, freeze the
QA-007 bar, or claim "QA-007 met".

## Verdict

Counted row **PASS**. `validate_interop_bundle_w6` **OK**. `milestone_counted=false`
is **lawful** (below). QA-007 stays **withheld** (`[confirm]`, G-06 open). No A4
kill and no reduction. The milestone is **not** complete. C2 is this commit.
(e) is pending after this write.

## `milestone_counted=false`

Lawful. It is not a pack bug and it does not block C2.

The width-6 contract requires the flag false. `TASK_2_MINI_SPEC.md` §2 sets the
claim to `milestone_counted=false` because the row is not the 4/6/8 QA-007
verdict. §6 lists `milestone_counted=true` as unsupported while width 8 and the
bar freeze are still open. ET-2 requires the validator to reject any other
value. DS-1 names the same flag. Task-1 recorded false for the same reason: one
width is not the milestone verdict. ADR-F1A-009 governs the clean-start order.
`clean_start` is true, which is what that order needs. The flag is the claim
boundary, not the clean-start bit.

## Chain

| Role | SHA |
|------|-----|
| C0 stamp | `388a5e5ff7775c496e144edee5b76be85f7caf54` |
| C1 implementation | `d336472fa59c514c9eb3a5e6001daf343a4bbb66` |
| C2 | this commit |

`provenance.implementation_revision` is the C1 tip.

## (c) launch

Exit 0. Wall `real` 3.769 s. `affinity_cpu` is 0. All four `*_NUM_THREADS`
values are `"1"`. The command was run once.

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
taskset -c 0 env \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  conda run -n qgd --no-capture-output \
  python benchmarks/density_matrix/interop_profile/validation_pipeline.py --width 6
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
```

The width-6 command is the (c) launch. It was not re-run for this closeout.
The two `specs_check.sh` lines are the lint reproduce commands.

## Evidence pins

| Field | Value |
|-------|--------|
| Bundle | `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json` |
| sha256 | `5257bad23e9fef4afad3f7b8b61f84c7cebd2ef8f85d95a94a02cb794d139d7d` |
| Validator | `validate_interop_bundle_w6` OK |
| `clean_start` / `provenance_pass` | true / true |
| `dirty_paths` | `[]` |
| `implementation_revision` | `d336472fa59c514c9eb3a5e6001daf343a4bbb66` |
| `milestone_counted` | false |
| Suite | `interop_profile_task2_evqe_6q_v1` |
| Counted samples | 1000 |
| `operation_count` / divisor | 18 / 73728 |
| `qa008.mean_O_absolute_margin` | 0.02 |
| Task-1 bundle | `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e` unchanged |

`libqgd.so` sha256 `615610c86ed1152a65b69b7c0ffa78812071c58297f530ee8dea856ae72ff947`.
The VQE wrapper `.so` sha256 `c9ed35dfba7811ff3a83c81dd2c518a943a7910fdd40b0c73eb916aee4480430`.
Compiler `c++ (GCC) 11.5.0`. The bundle still does not pin flags (N-46).

## Throughput and O

These figures are the counted row under Measure. They are not a QA-007 verdict
and not an A4 result. The one-sided bound is about 0.21 %. The 5 % A4 test needs
every width in {4, 6, 8}. Width 8 is not counted. No reduction was taken.

| Quantity | Record |
|----------|--------|
| `mean_O` | −0.001123 |
| `upper_bound_95_O` | 0.002149 |
| `median_O` | 0.000864 |
| `min_O` / `max_O` | −1.451 / 0.540 |
| spike count | 73 |
| samples | 1000 |
| mean ns per ρ-entry per operation | 11.006 |
| throughput upper bound | 11.042 |
| mean wrapper ns | −333.4 |

Stored values: `mean_O` `−0.0011232300452190054`, `upper_bound_95_O`
`0.0021490045376290055`, `spike_count_abs_wrapper_ns_above_20000` `73`,
`throughput.mean_ns_per_op` `11.006418687608507`. The negative mean is the
pre-registered lawful outcome. Two of 1000 samples have `O_i` < −0.5. They were
kept. The lane did not refuse. S-f stays disposed.

## Independence

Tester confirmed independence before this closeout. The note is
`/tmp/mf5a-t2-counted/INDEPENDENCE_NOTE.md`, sha256
`ab2f66124abef6ed58d4fca76454f78045b2183c1b4bf21125ad6ad956c8201c`.

The cell is `build_task_evaluator(6)` and `Optimization_Problem` on `libqgd.so`.
The oracle is `Test_VQE._get_density_backend_aer_reference` after
`set_Optimized_Parameters`, in
`test_task2_evqe_6q_cell_pins_timer_identity_and_aer_oracle`. It builds its own
noise and its own trace. It does not read the cell's energy or ρ. Shared inputs
are the HEA circuit export, the §2 parameter vector, and the Hamiltonian. A
defect there would show on both sides. Flag-off and flag-on energies agreed
bitwise in that pytest: the flag only adds `clock_gettime` reads on the same
instance and vector, so the check shares the kernel and cannot see a kernel bug.
Aer agreement is not bitwise. The pytest PASSED and was not skipped
(`/tmp/mf5a-t2-counted/aer_w6_pytest.log`, exit 0, 0.65 s). C1 re-gate
`/tmp/rev-mf5a-t2-step4b/REVIEW-regate.md` (`5729f180…`) shows the same node
PASSED and is corroboration.

## Acceptance verdicts (vs slice contract)

ET checkboxes stay unchecked (N-42). Red-first rows stay unticked (N-79).

| Signal | Verdict | Evidence |
|--------|---------|----------|
| ET-1 pins, bit identity, Aer oracle | pass | C1 re-gate `5729f180…` §§6–7: node PASSED, not skipped. Tester R-1 log above, exit 0 |
| ET-2 validator | pass | same re-gate: 46 passed |
| ET-3 Measure row | pass | (c) `/tmp/mf5a-t2-counted/REPORT.md`: mean −0.001123, spike 73, no sample dropped |
| ET-4 containment | pass | same re-gate §§6–7: containment diffs exit 0; 54 and 115 passed on the wider lanes |
| §9 counted row | pass | (c) REPORT; bundle sha256 above; `milestone_counted=false` |
| QA-007 | withheld | G-06 open. Not a "QA-007 met" claim. No A4 kill and no reduction |
| Lint | pass at this write | `/tmp/rev-mf5a-t2-c2/REVIEW.md` §6.7 was 0/0 before this fix. Reproduce commands above |
| Milestone | not complete | width 8, attribution routes, and the bar freeze stay later |

## Header notes folded here

N-66. The mini-spec header again cites upload `5c810dac…`. Full sha256
`5c810dacef780a7b0187f288e2bbadfa0e02bee5d4735c246ecb6338f63c0607`.

N-67. The six `step-4a` leftovers are past tense: the C0 stamp set
`step-4b-authorized`; RM ACCEPT did not flip it; this closeout now exists.

N-78 stays open. No test pins the `provenance.command` equality clause alone.
A mutant that deletes only that clause still passes the current match string.

N-79 stays open. The fix pass has no Developer red-first log. The substitute
record is `/tmp/rev-mf5a-t2-step4b/logs/regate-probe-mutation.txt`. Citing it
does not close the note.

N-80 stays open. The fix-pass pytest ran from the repo root and rewrote ignored
`costfuncs_and_entropy.txt` and `.pytest_cache`. Ignored files are outside
clean-start porcelain. A `/tmp` cwd with `-p no:cacheprovider` avoids a repeat.

## Remaining

M-F5a stays open. G-06, G-08, and G-09 stay open. Width 8, the four attribution
routes, and any reduction stay later. (e) is pending after this write. (g) is
pending after C2. No push and no pull request. QA-007 stays `[confirm]`.
