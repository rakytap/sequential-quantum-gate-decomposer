# M-F5a slice 5 closeout — attribution routes at widths 6 and 8
> **Status:** shipped · **Date:** 2026-10-08 · **Work package:** task-5 ·
> **Scope:** widths 6 and 8 only. REQ-004 stays open. M-F5a is not complete ·
> **C1:** `3a8a0d749bf2cb1da6b9cdcdf52382a1469bab8c` ·
> **C2-w6:** `78ba71081aedf455e91674e7f15db7434b60ed23` (bundle only) ·
> **C2-w8:** `3feffb070ba9adfb0ed9437e478db188b1902df6` (bundle only) · (g) PASS at both widths ·
> **No push/PR**

## Summary

This closeout records the counted width-6 and width-8 attribution bundles. C1 `3a8a0d74` holds the lane, the validator, and the tests. C2-w6 `78ba7108` commits the width-6 bundle and no closeout. C2-w8 `3feffb07` commits the width-8 bundle and no closeout. This file is the closeout. It does not re-run either counted command.

REQ-004 stays open. CAP-004 stays hold-the-line. `milestone_counted` stays false. R-strict is a `handback_refused` row and carries no timings. Interpretation of the width-6 and width-8 rows belongs to the Research Manager. The Research Manager has not vetoed N-y; that check stays FYI.

## Verdict

Counted width-6 and width-8 routes **recorded**. `milestone_counted=false` is **lawful**. REQ-004 stays **open**. The milestone is **not** complete. This file does not claim that four routes are delivered.

## Aer limitation

The Aer reference shares the circuit definition and the noise specification with SQUANDER. It does not share the kernels or the lowering. The method is `get_Qiskit_Circuit`, then `Test_VQE._insert_reference_noise`, then `qiskit_aer`. The (g) clone is first on `sys.path`.

The Aer reference shares the circuit definition (`get_Qiskit_Circuit`, through `Qiskit_IO`) and the noise specification (the same three-channel list, written separately in `interop_lane.DENSITY_NOISE` and `Test_VQE._get_density_backend_noise`) with the routes. It does not use SQUANDER's density kernels, `NoisyCircuit`, `DensityMatrix`, or the partitioned runtime lowering. It is independent of the kernels and lowering, not of the circuit and noise definition.

## Six Aer values

`qiskit_aer` 0.17.2. Tolerance 1e-10. Each value below is max |Δρ| against Aer. The short forms are the report figures. The parenthetical floats are the report text.

| Width | Route | max \|Δρ\| |
|-------|-------|------------|
| 6 | R-base | 2.58e-16 (`2.581417751465236e-16`) |
| 6 | R-fused | 2.75e-16 (`2.754662226278516e-16`) |
| 6 | R-hybrid | 2.86e-16 (`2.8609792490763985e-16`) |
| 8 | R-base | 2.22e-16 (`2.2229005699427133e-16`) |
| 8 | R-fused | 1.99e-16 (`1.9869968258640557e-16`) |
| 8 | R-hybrid | 2.10e-16 (`2.100376937328634e-16`) |

Width 6 is `/tmp/mf5a-t5-g-w6-r2/REPORT.md` (`fc10e00a…`). Width 8 is `/tmp/mf5a-t5-g-w8/REPORT.md` (`d9117dfe…`). Each value is below 1e-10. R-strict refused and has no state.

The Research Manager's sentence (RM 2026-10-08), verbatim: the timed routes "agree with the step-by-step (sequential) reference to ≤1.2e-16 and with an independent simulator (Qiskit Aer 0.17.2, given the same circuit and noise specification) to ≤5e-16, i.e. at double-precision machine precision." The ≤1.2e-16 sequential bound rests on width-4 data (task-4 C2 binder `7b48bbc6…` §6.1); this file quotes no sequential figure at widths 6 and 8. At widths 4, 6, and 8 every timed route agrees with Aer to ≤5e-16.

## Width-6 counted launch

The width-6 command was launched twice at C1 `3a8a0d74`, never concurrently: the first launch (2026-10-08 02:04:45 -04:00) stopped about 7.8 s in, before the pipeline's only write, and left no bundle and no timings, only 3 run.log header lines (95 bytes, kept as `/tmp/mf5a-t5-counted-w6-first-launch/run.log.RECONSTRUCTED`, sha256 `cde458753465410a3c73f7cab795943df3851388c7ba4eea4ad6ccef43cffaf1`, reconstructed from the Tester transcript because the second launch truncated the original), and the second launch (02:10:22 -04:00) exited 0 and wrote the counted bundle `212758709a06…`; Reviewer ruled this not a selective rerun (`/tmp/rev-mf5a-t5-c2-w6/REVIEW.md`).

P-2 `NOTE.md` sha256 is `fb9a0670594e8228f82f4f2bd8163190ce8d0a7053e2725b7e6e6cc78f095158`.

## Width-6 (g)

The width-6 (g) was attempted three times, each from a fresh clone of C2-w6 `78ba7108` and never alongside a counted run. Attempt 0 (2026-10-08, about 02:36:50 -04:00) exited 127 before Python started, because `conda` was not on the launcher's PATH; attempt 1 overwrote its log, and its record is attempt 1's report (`/tmp/mf5a-t5-g-w6/REPORT.md`, sha256 `278730f83b0563551ce6d802401aecb1dd665a3a4f3597053b4e041434c91e14`). Attempt 1 (02:39:10 -04:00) ran the pipeline with the `qgd` interpreter directly instead of the `conda run -n qgd` form in `TASK_5_MINI_SPEC.md` §3. It wrote a regenerated bundle (sha256 `680743b11c4a973ee393691f8444a14ef7487e25ea57fcca3875b7dd5eaacfdc`, kept under `/tmp`; its timings are not used), and `qa008_route_categorical_exact` raised at `provenance.environment.conda_default_env` (`base` against `qgd`). A full leaf comparison finds exactly three categorical differences, `conda_default_env`, `conda_prefix`, and `provenance_pass`, all set by the launcher. The Aer step did not run. The Tech Lead authorized one corrective repeat, r2 (02:48:37 -04:00, `/tmp/mf5a-t5-g-w6-r2/REPORT.md`, sha256 `fc10e00a91faedc57e183ad69bc363e76518d9ea123db381e4f72e7f095a3c32`), in the §3 form. `qa008_route_categorical_exact` returned, and max |Δρ| against Qiskit Aer 0.17.2 was 2.58e-16 for R-base, 2.75e-16 for R-fused, and 2.86e-16 for R-hybrid, each below the ET-5 tolerance of 1e-10. Reviewer ruled this not a selective rerun (`/tmp/rev-mf5a-t5-c2-w8/REVIEW.md`).

The r2 run-identity leaves differ from the committed width-6 bundle, as `TASK_5_MINI_SPEC.md` §4 allows: `implementation_revision` `3a8a0d74` → `78ba7108`; `density_matrix_cpp_sha256` `05f01747e986…` → `fb953df3ffa2…`; `squander/libqgd.so` `615610c86ed1…` → `e0f232f47b21…`; the VQE wrapper `.so` `c9ed35dfba78…` → `32efaf11cb59…`. The r2 clone was built with the `qgd` cmake 4.2.0 (`CMakeCache.txt` `CMAKE_COMMAND`), as the main checkout was; the "cmake 3.31.8" line in its log came from the outer shell before `conda run -n qgd`.

The Research Manager accepts the disclosed width-6 relaunch and the width-6 (g) repeat as non-selective (RM, 2026-10-08).

## Width 8

The width-8 command was launched once, at 2026-10-08 02:58:22 -04:00. Wall time was 234.47 s. It wrote the counted bundle. The width-8 (g) was one attempt, from a fresh clone of C2-w8 `3feffb07`. The `--output` path was `/tmp/mf5a-t5-g-w8/interop_profile_bundle_routes_w8.repro.json` (G8-1). `qa008_route_categorical_exact` returned. The categorical diff was empty. Run-identity leaves: `implementation_revision` `78ba7108` → `3feffb07`; `density_matrix_cpp_sha256` `05f01747e986…` → `e86a31117b9b…`; extension 0 `615610c86ed1…` → `7c7dba1fa00a…`; extension 1 `c9ed35dfba78…` → `1f85dd694fb4…`.

## Keep until the milestone review

Leave these unmodified until the milestone review: `/tmp/mf5a-t5-g-w6/`, `/tmp/mf5a-t5-g-w6-r2/`, `/tmp/mf5a-t5-counted-w6-first-launch/`, `/tmp/mf5a-t5-counted-w8/`, `/tmp/mf5a-t5-counted-w6/`, `/tmp/mf5a-t5-g-w8/`, and the two `ADDENDUM-*.md` files in the table below.

`/tmp` is not durable. This closeout carries the sha256 values so the record survives a cleanup. The C2-w6 binder has no §5.5 table. The rows below are the C2-w8 binder §5.5 list, plus the width-8 (g) report, its repro bundle, the tmux addendum, and the r2 cmake addendum.

| Path | sha256 |
|------|--------|
| `/tmp/mf5a-t5-g-w6/REPORT.md` | `278730f83b0563551ce6d802401aecb1dd665a3a4f3597053b4e041434c91e14` |
| `/tmp/mf5a-t5-g-w6/interop_profile_bundle_routes_w6.repro.json` | `680743b11c4a973ee393691f8444a14ef7487e25ea57fcca3875b7dd5eaacfdc` |
| `/tmp/mf5a-t5-g-w6/repro_run.log` | `83cc241537840ab8aa53d5f29f147861c4f8877a4a18df07930af7de2be16cdd` |
| `/tmp/mf5a-t5-g-w6/repro_validator.txt` | `b50b665c4588c7a41876578752233ce20731d217528a23b5eba4a747aa7281ef` |
| `/tmp/mf5a-t5-g-w6/run_gated_steps.sh` | `921575c9a0e8ad5fb214f0feccccd5e0c4cb2d17b8b93df06612adf33fb9fa20` |
| `/tmp/mf5a-t5-g-w6/aer_check_w6.py` | `5a13802c92f431a9d14c1433563d979ce7be1495d98dc20cc6ebfca5fb655fcf` |
| `/tmp/mf5a-t5-g-w6/tests.log` | `a145fceb8062126d84e05f4f9e0a7b02b29c8d5edf799d9718c35d9fb28495c4` |
| `/tmp/mf5a-t5-g-w6/build.log` | `dd242c3319d7bda3620d5d25ddd474ff60eba45ec047a1c49cb3ac4bb5e8f92d` |
| `/tmp/mf5a-t5-g-w6-r2/REPORT.md` | `fc10e00a91faedc57e183ad69bc363e76518d9ea123db381e4f72e7f095a3c32` |
| `/tmp/mf5a-t5-g-w6-r2/run_g.sh` | `605c5b6372387039504cfbeec6e15f9715de24ed6e271cab44dd981b0056bd4d` |
| `/tmp/mf5a-t5-g-w6-r2/run_g.log` | `2448c0561cfca247a8edfafff378116ca7c4e85bb2767fadd8d8b58a2e7777cf` |
| `/tmp/mf5a-t5-g-w6-r2/aer_check_w6.py` | `a42d26d590ea5b455cd39e6098f41aece42e9b26a469d85c7c183e832b503dad` |
| `/tmp/mf5a-t5-g-w6-r2/aer_check_w6.log` | `bc2b5221bf0364de3d74553e0a417fdc7a9d1068e031933099ee856c6ed7ea49` |
| `/tmp/mf5a-t5-g-w6-r2/interop_profile_bundle_routes_w6.repro.json` | `7b9aede8ed77800eb5ece69f78b0b8632fac8fe9e43542be42ee24ae24609484` |
| `/tmp/mf5a-t5-g-w6-r2/repro_validator.txt` | `d79034d457639c4c0d481d6153359dc93863d2a55d19c7c240083e5febc8f70b` |
| `/tmp/mf5a-t5-g-w6-r2/conda_preflight.txt` | `2d7d09b58810babc39bac11dee1489c4fc1907e88d22be1859be789d41368422` |
| `/tmp/mf5a-t5-g-w6-r2.ADDENDUM-cmake.md` | `ddd8bb25f15a105159fe6e20ce85ee24f12857503c657df0832fe3e4193f816c` |
| `/tmp/mf5a-t5-g-w8/REPORT.md` | `d9117dfeace2b2464b06c157dea55bb47b0a8ed21dd214132bec24d379350ade` |
| `/tmp/mf5a-t5-g-w8/interop_profile_bundle_routes_w8.repro.json` | `01fc083cedda88392361f3e795f851ea46e768dc6a73ce6d070d0031ecb0a231` |
| `/tmp/mf5a-t5-counted-w8.ADDENDUM-tmux.md` | `a1fc94504681112b1e10c44533a354e90bbd12100fe4a4c97323141127fa3b21` |
| `/tmp/mf5a-t5-counted-w6-first-launch/run.log.RECONSTRUCTED` | `cde458753465410a3c73f7cab795943df3851388c7ba4eea4ad6ccef43cffaf1` |
| `/tmp/mf5a-t5-counted-w6-first-launch/NOTE.md` | `fb9a0670594e8228f82f4f2bd8163190ce8d0a7053e2725b7e6e6cc78f095158` |
| `/tmp/mf5a-t5-counted-w6/HANDBACK.md` | `4667f0f8d76f9a744d607cbb5a6c13a21d831ffcfbf0cb343e501f068a87b6e2` |
| `/tmp/mf5a-t5-counted-w8/HANDBACK.md` | `97b53f71ee51f907350f35103eaba35b73cdb17ae9240e6176e81e127b66ed26` |

The r2 build used `qgd` cmake 4.2.0. That fact is in the addendum `ddd8bb25…`.

## Timings

Read with `git show <rev>:<path>`. Fields: `rows[].orchestration.mean_ns`, `rows[].apply_component.mean_ns`, `rows[].throughput.mean_ns_per_op`. `ns/op` is `throughput.mean_ns_per_op`. It equals `apply_component.mean_ns / throughput.divisor`. The divisors are 3072, 73728, and 1572864. Width 4 is `5f63a9d6:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w4.json` (`6584be2b…`). Width 6 is `78ba7108:…/interop_profile_bundle_routes_w6.json` (`212758709a06…`). Width 8 is `3feffb07:…/interop_profile_bundle_routes_w8.json` (`a9e375a1c4c7…`). Width 4 is the committed task-4 bundle. Tester handback `orchestration.mean_ns` matches the width-6 and width-8 rows. No mismatch.

| Width | Route | orchestration mean_ns | apply mean_ns | ns/op |
|------:|-------|----------------------:|--------------:|------:|
| 4 | R-base | 284754.706 | 57710.178 | 18.785865234375 |
| 4 | R-fused | 414795.064 | 142730.456 | 46.461736979166666 |
| 4 | R-hybrid | 827898.659 | 518435.171 | 168.76144889322916 |
| 6 | R-base | 407477.964 | 825862.95 | 11.201483154296875 |
| 6 | R-fused | 625104.792 | 2699225.087 | 36.610583319769965 |
| 6 | R-hybrid | 1040973.854 | 5424578.381 | 73.57555312771267 |
| 8 | R-base | 640108.475 | 16421624.405 | 10.440587619145711 |
| 8 | R-fused | 996402.474 | 56348400.768 | 35.82534838867188 |
| 8 | R-hybrid | 1810504.283 | 144894620.291 | 92.12151863797506 |

R-strict is a refusal row at each width. It has no timings. The reason contains `channel_native_noise_presence`. The handback is `98eec857`.

Apply time versus R-base, from `rows[].apply_component.mean_ns`, rounded to 2 decimal places. Current implementation cost, not intrinsic cost; no speed claim. Fused is 2.47×, 3.27×, and 3.43× at widths 4, 6, and 8. Hybrid is 8.98×, 6.57×, and 8.82× at those widths. These are the only ratios in this file. There is no orchestration ratio.

## CPU 0

Counted runs were pinned single-thread to CPU 0. Other users' jobs shared its L3 cache and its hyperthread sibling. That sharing appears as timing tails (Reviewer N-4, w6 #6). This note does not quote the tails.

## Chain

| Role | SHA |
|------|-----|
| C0 | `cbb203ae` |
| C1 | `3a8a0d749bf2cb1da6b9cdcdf52382a1469bab8c` |
| C2-w6 | `78ba71081aedf455e91674e7f15db7434b60ed23` |
| C2-w8 | `3feffb070ba9adfb0ed9437e478db188b1902df6` |

Reviewer C2-w6 binder `/tmp/rev-mf5a-t5-c2-w6/REVIEW.md` (`05afed439100a3e69ea4cb2c8f5a5487df46350009158fcc726ce14cf766fbd2`). Reviewer C2-w8 binder `/tmp/rev-mf5a-t5-c2-w8/REVIEW.md` (`2cd439388994fd5869a01f0aa1b19677cd3bebbfa1518def6bae34f3cbc8c23e`).

## Evidence pins

| Field | Width 6 | Width 8 |
|-------|---------|---------|
| Bundle | `interop_profile_bundle_routes_w6.json` | `interop_profile_bundle_routes_w8.json` |
| sha256 | `212758709a06c2805feef1181111ed980172877ad8b11e66f68bfafb9b11d1be` | `a9e375a1c4c7c39f908ed7136cc461403e315c232ec1a1d4d60fbf4da28fa84f` |
| git blob | `9be88fc08a407b6192ed024f9ee7a677791a1edb` | `33ffba843cbcb90b0a2324ad60c855be0c5750e2` |
| Suite | `interop_attribution_routes_task5_w6_v1` | `interop_attribution_routes_task5_w8_v1` |
| `milestone_counted` | false | false |
| `implementation_revision` | `3a8a0d74` (C1) | `78ba7108` (C2-w6) |
| Divisor | 73728 | 1572864 |
| Protocol | 50 warm-up, then 1000 counted calls per timed route; `affinity_cpu` 0; four thread values `"1"` | same |

R-strict is `handback_refused` at both widths. Handback `STEP_4A_HANDBACK.md` `98eec857…` is unchanged. Tester (c) width 6 is `/tmp/mf5a-t5-counted-w6/HANDBACK.md` (`4667f0f8…`). Tester (c) width 8 is `/tmp/mf5a-t5-counted-w8/HANDBACK.md` (`97b53f71…`).

## Acceptance verdicts (vs slice contract)

ET checkboxes stay unchecked, as in tasks 1–4. Lane results are the Reviewer binders' and the Tester reports. This docs pass did not rerun them.

| Signal | Verdict | Evidence |
|--------|---------|----------|
| ET-1…ET-4 | pass | C1 `3a8a0d74`. (g) pytest: bundle validation 173 passed and harness 48 passed, at r2 and at the width-8 (g) |
| QA-008 | pass | r2 categorical gate returned. Width-8 categorical diff empty |
| REQ-005, REQ-006, REQ-007, REQ-008, QA-009 | pass | MS §6 repo-review rows: both `git diff --exit-code 5f63a9d6` rows empty; the `064a6f4f` allowlist row lawful after C2-w8 (C2-w8 binder `2cd43938…` §1). DS-3 `test_explicit_state_vector_matches_legacy_default` 1 passed (C1 re-gate binder `f9d95db0…`) |
| Aer, both widths | pass | six values above; each below 1e-10; `qiskit_aer` 0.17.2 |
| Counted bundles | recorded | evidence pins; `milestone_counted=false` |
| REQ-009 lint | pass at this write | both `specs_check.sh` lines below exit 0 |
| REQ-004 | open | these rows do not close it |
| Milestone | not complete | G-08 and G-09 stay later |

## (g) clean regeneration

PASS at both widths. Width 6: the passing run is r2, report `fc10e00a…`, clone of C2-w6 `78ba7108`. Attempts 0 and 1 are recorded above and are not the passing run. Width 8: one attempt, report `d9117dfe…`, clone of C2-w8 `3feffb07`, output `/tmp/mf5a-t5-g-w8/interop_profile_bundle_routes_w8.repro.json` (`01fc083c…`). Regenerated files were not committed. Timing leaves differ, as a regeneration may.

## Reproduce

The counted commands were run for C2-w6 and C2-w8. This closeout does not re-run them.

```bash
cd /home/zkegli/work/squander-with-density-matrix/sequential-quantum-gate-decomposer
unset PYTHONPATH
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 6 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w6.json
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py --attribution-routes --width 8 --output benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_routes_w8.json
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
```

## What this closeout does not say

No kernel benchmark. The only ratios are the apply-time figures above, and only with that frame. No orchestration ratio. REQ-004 stays open. CAP-004 stays hold-the-line. `milestone_counted` stays false. This file does not say the milestone is done. No VQA. M-F1b stays closed. No push and no pull request.

## Carries

- Width-6 binder nits N-1 through N-6, N-8, and N-9 stay non-blocking. N-7 is the task-4 quotation; the forward note is on `task-4/CLOSEOUT.md`. RM N-c stays open.
- Width-8 binder nits N-1 through N-6 stay non-blocking. N-5 is the later validator item (the N-35 family).
- N-y stays FYI to the Research Manager.
- N-46 (compiler flags not pinned) stays open from task-4.
- G-08, G-09, and Demo No GO stay later.

## Next

The Research Manager interprets the width-6 and width-8 rows. This file does not do that. M-F5a is not complete.
