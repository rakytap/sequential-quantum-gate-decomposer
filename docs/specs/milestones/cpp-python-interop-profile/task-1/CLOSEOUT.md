# M-F5a slice 1 closeout — E-VQE 4-qubit interop tracer
> **Status:** shipped · **Date:** 2026-10-07 · **Work package:** task-1 ·
> **Scope:** task-1 Step 4b slice close only. M-F5a is not complete ·
> **C1 tip:** `ca5589e25036531d599ff963e7d349c6e55b5951` ·
> **C2:** `ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df` · (e) APPROVE · (g) PASS · **No push/PR** ·
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
This closeout reconciles them: (c) has passed, and task-1 Step 4b is closed at
C2 `ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df`. Reviewer (e) approved that
commit (APPROVE FOR C2, bc-b25fac97; binder `/tmp/rev-mf5a-c2/REVIEW.md`,
sha256 `8749d515be86a3f067e9426062bc768275ebc1ec7e6022cbca0062a050f59e8b`).
Step (g) passed. Checklist §11 records (e) and (g) in a later docs pass on that tip. This
file still does not complete the milestone, freeze the QA-007 bar, or open
the next slice.

## Verdict

Counted row **PASS**. `validate_interop_bundle` **OK**. Task-1 Step 4b slice
**closed**. Step (g) **PASS**. QA-007 stays **withheld** (`[confirm]`, G-06
open). The milestone is **not** complete.

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

That line is exact for `libqgd.so`, which holds the timed `apply_to` path.
The wrapper `.so` adds `-DCPYTHON` only. The bundle pins the compiler and both
hashes, not the flags. Putting the flags in the bundle stays open (N-46).

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

These figures attribute the counted row. They are not a QA-007 verdict. The
label stays withheld (G-06 open; bar `[confirm]`; row not milestone-counted).

| Quantity | Attribution record |
|----------|--------------------|
| `mean_O` | 0.00711 |
| `upper_bound_95_O` | 0.01059 |
| mean ns per ρ-entry per operation | ≈13.01 |
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
`O_i` is 0.369, and 0 samples have `O_i` < −0.5. The QA-008 margin is pinned
at 0.02 and was tested at (g). Stored extremes are min
`−0.46106642151229604` and max `0.3694493006993007`.

## Independence

Tester confirmed independence before this closeout. The Aer reference energy
and the C++ density-matrix energy are the existing VQE test
`test_density_matrix_backend_anchor_fixed_parameter_matches_aer_reference`,
unchanged since C1. The tight timer check is harness flag-off versus flag-on
bit identity in the interop harness tests. Those two runs are bitwise equal
because the flag only adds `clock_gettime` reads, on the same instance and
vector, with a deterministic kernel. That check shares the kernel and cannot
see a kernel bug. The Aer node is the independent check, within `atol=1e-12`
and NumPy's default `rtol=1e-5` (about 7.6e-6 at this cell). This counted row
is attribution-only (`milestone_counted=false`).

## Test snapshots

Lanes at C2, from Reviewer (e) §7. This docs pass did not rerun them.

| Lane | Command | Result |
|------|---------|--------|
| Interop harness and bundle validation | `pytest tests/VQE/test_vqe_interop_harness.py tests/VQE/test_vqe_interop_bundle_validation.py -q` | 23 passed |
| Mini-spec §10 smoke, Aer, and state-vector | the §10 nodes, as run in Reviewer (e) §7 | 6 passed |
| VQE fast | `pytest tests/VQE -m "not slow" -q` | 92 passed |
| Density fast | `pytest -m "density_matrix and not slow" -q` | 231 passed, 861 deselected |
| Spec lint | `specs_check.sh` and `--strict` on this milestone | 0 errors, 0 warnings |

## Acceptance verdicts (vs slice contract)

| Signal | Verdict | Evidence |
|--------|---------|----------|
| Counted clean-start row | pass | bundle sha256 above; `clean_start` and `provenance_pass` true |
| QA-008 margin pin | pinned (0.02); tested at (g) | `mean_O_absolute_margin` 0.02; (g) `|Δ mean O|` 0.001209 |
| QA-007 10 % bar | withheld | G-06 open; bar `[confirm]`; row not milestone-counted. Not a "QA-007 met" claim |
| S-f stalls | no refusal at (c) | 0 samples below −0.5; margin tested at (g), not at (c) |
| Mini-spec §10 rows (task-1 scope of REQ-001…009) and ET-1…ET-5 | pass at C2 review | lanes above; containment diffs in Reviewer (e) §7 exit 0. ET checkboxes stay unchecked. REQ-level acceptance not claimed: widths 6 and 8, routes, and reduction later; REQ-006 flags (N-46); REQ-007 CI (G-09); REQ-009 docs (G-08); QA-007 withheld is REQ-002 |
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

The command above is the (c) launch at C1 and the (g) launch at clean C2.

## (g) clean-C2 regeneration

**PASS.** Evidence `/tmp/mf5a-t1-counted-g/`. Exit 0. Wall `real` 2.130 s.
Categorical pins matched. The only revision change was
`ca5589e2` → `ddde49ac`. Regenerated `mean_O` is 0.008321 and
`upper_bound_95_O` is 0.010361, inside [−0.01289, 0.02711]. `|Δ mean O|` is
0.001209, within 0.02. The committed bundle was restored to sha256
`212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`, and
porcelain was empty. Those outputs are not committed. The counted row above
stays the (c) figures.

## Remaining

M-F5a stays open. G-06, G-08, and G-09 stay open. (e) is APPROVE and (g) is
PASS. This file does not open the next slice. No push and no pull request.
QA-007 stays `[confirm]`.

Open carries for the next Step 4a are checklist §11: N-41, N-46, S-g, N-16,
N-17, N-23, N-32, N-35, N-36, N-37, N-24, N-3, N-6, N-10, N-14, N-19, N-21,
N-25, N-27, N-28, N-29, N-31, N-40, S-a, S-b, S-c, and S-e. S-g is the
estimator note: single-call spikes of about 20–25 µs on CPU 0 bias the
arithmetic mean of `O_i` low, with no stall. N-42, N-44, N-45, and the N-47
C2 record are folded in this pass.
