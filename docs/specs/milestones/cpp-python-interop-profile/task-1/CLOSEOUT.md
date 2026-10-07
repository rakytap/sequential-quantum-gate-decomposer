# M-F5a slice 1 closeout — E-VQE 4-qubit interop tracer
> **Status:** shipped · **Date:** 2026-10-07 · **Work package:** task-1 ·
> **Scope:** task-1 Step 4b slice close only. M-F5a is not complete ·
> **C1 tip:** `ca5589e25036531d599ff963e7d349c6e55b5951` ·
> **C2:** this commit · **No push/PR** ·
> **Claim:** counted attribution row. QA-007 stays `[confirm]`. Not a "QA-007 met" claim

## Summary

This closeout records the counted clean-start interop tracer row for task-1.
C1 is `ca5589e2`. That revision spans three commits (N-38): implementation
`ca2bf7c4df3843e25ab226ccd1157d63d00228e1`, provenance fix
`06b91d6cce63b2d967ad023c4887fa89f8054310`, and planning sync
`ca5589e25036531d599ff963e7d349c6e55b5951`. Reviewer (a) approved C1
(`/tmp/rev-mf5a-c1-rereview/REVIEW.md`, sha256 `d779e54eeaec11ad620bc531d19e7319715d1f9902dc4c5fe7b3d3a40863a478`).
Tester (c) passed at that tip. This file does not complete the milestone, freeze
the QA-007 bar, or authorize the next slice.

N-39 froze checklist §7 ("No-go for a counted run") and the "ahead of the
Reviewer (a) re-gate" lines as true of C1. They were not edited before (c).
This closeout reconciles them: (c) has passed, and task-1 Step 4b is closed here.

## Verdict

Counted row **PASS**. `validate_interop_bundle` **OK**. Task-1 Step 4b slice
**closed**. QA-007 is **not** met (`[confirm]`, G-06 open). The milestone is
**not** complete.

## C1 chain (N-38)

| Role | SHA |
|------|-----|
| C0 planning | `17518a72771d93ac6400217687b5776dff3258d5` |
| C1 implementation | `ca2bf7c4df3843e25ab226ccd1157d63d00228e1` |
| C1 provenance (B1–B3) | `06b91d6cce63b2d967ad023c4887fa89f8054310` |
| C1 tip (B4 sync) | `ca5589e25036531d599ff963e7d349c6e55b5951` |

`provenance.implementation_revision` in the bundle is the C1 tip.

## N-33 build profile

No rebuild. The counted run used the Reviewer-pinned binaries:

| Binary | sha256 |
|--------|--------|
| `squander/libqgd.so` | `615610c86ed1152a65b69b7c0ffa78812071c58297f530ee8dea856ae72ff947` |
| VQE wrapper `.so` | `c9ed35dfba7811ff3a83c81dd2c518a943a7910fdd40b0c73eb916aee4480430` |

Compiler at run time: `c++ (GCC) 11.5.0`. The lane's `build_profile` is its
Release literal. Effective flags, from the Reviewer binder and `build.ninja`,
not re-measured on this close:

```text
-O3 -DNDEBUG -std=gnu++11 -fPIC -DBLAS=2 -Wall -O3 -m64 -ggdb -DNDEBUG -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fpermissive -ftree-vectorize -mavx2 -mfma -DUSE_AVX -fopenmp
```

## N-34 counted launch

Verbatim from the Tester run note. Exit 0. Wall `real` 2.152 s. `affinity_cpu`
is 0. All four `*_NUM_THREADS` values are `"1"`.

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
taskset -c 0 env \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  conda run -n qgd --no-capture-output \
  python benchmarks/density_matrix/interop_profile/validation_pipeline.py
```

## Evidence pins

| Field | Value |
|-------|--------|
| Bundle | `benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json` |
| sha256 | `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e` |
| Validator | `validate_interop_bundle` OK |
| `clean_start` | true |
| `provenance_pass` | true |
| `dirty_paths` | `[]` |
| `implementation_revision` | `ca5589e25036531d599ff963e7d349c6e55b5951` |
| `milestone_counted` | false |
| Counted samples | 1000 |
| `qa008.mean_O_absolute_margin` | 0.02 |

## Throughput and O (attribution only)

These figures attribute the counted row. They are not a QA-007 verdict and they
do not meet QA-007.

| Quantity | Attribution record |
|----------|--------------------|
| `mean_O` | 0.00711 |
| `upper_bound_95_O` | 0.01059 |
| mean ns/op | ≈13.01 |
| samples | 1000 |
| margin pin | 0.02 |

The bundle stores `overhead.mean_O` `0.0071119768188966665`,
`overhead.upper_bound_95_O` `0.010589466306384618`, and
`throughput.mean_ns_per_op` `13.014052083333333`. The rounded row above is the
attribution record for this closeout.

## S-f

The Tech Lead proceeded with the mean-of-trials estimator and the margin pin.
The reported `O` is the arithmetic mean of `O_i`. No sample was dropped. The
lane did not refuse on a stall. Over 1000 samples, min `O_i` is −0.461, max
`O_i` is 0.369, and 0 samples have `O_i` < −0.5. The QA-008 margin pin held.
Stored extremes are min `−0.46106642151229604` and max `0.3694493006993007`.

## Independence (G-10)

Tester confirmed independence before this closeout. The Aer reference energy
and the C++ density-matrix energy are the existing VQE test
`test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference`,
unchanged since C1. The tight timer check is harness flag-off versus flag-on
bit identity in the interop harness tests. This counted row is attribution-only
(`milestone_counted=false`).

## Acceptance verdicts

| Signal | Verdict | Evidence |
|--------|---------|----------|
| Counted clean-start row | pass | bundle sha256 above; `clean_start` and `provenance_pass` true |
| QA-008 margin pin | held | `mean_O_absolute_margin` 0.02; validator OK |
| QA-007 10 % bar | not met | still `[confirm]` (G-06). This closeout does not claim "QA-007 met" |
| S-f stalls | held | no refusal; 0 samples below −0.5; margin held |
| Milestone outcome | not complete | widths 6 and 8, attribution routes, and the bar freeze stay later |
| Slice | closed | task-1 Step 4b only |

## Evidence commands (reproduce)

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
taskset -c 0 env \
  PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  conda run -n qgd --no-capture-output \
  python benchmarks/density_matrix/interop_profile/validation_pipeline.py
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
```

The counted command above is the (c) launch at C1. Step (g) regenerates from
clean C2, checks labels and `|Δ mean O| ≤ 0.02` by hand (N-32), and restores
generated outputs. Those regeneration outputs are not committed.

## Remaining

M-F5a stays open. G-06, G-08, and G-09 stay open. Reviewer evidence review (e)
and clean-C2 regeneration (g) follow this commit. No push and no pull request.
