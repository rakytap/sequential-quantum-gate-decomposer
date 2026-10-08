# CPP_PYTHON_INTEROP_PROFILE Closeout — `cpp-python-interop-profile` (interop profile)

> **Milestone delivery record.** Traceability: `CAP-*/QA-* → M-F5a → REQ-* → DS-* → evidence`.
- **Milestone:** M-F5a / `cpp-python-interop-profile` · **Outcome (roadmap):** met at 4, 6, and 8: E-VQE QA-007 verdict and three timed attribution routes with the R-strict refusal row (ADR-F5A-011); REQ-009 open until the spec refresh
- **Close date:** 2026-10-08 · **Status:** Partial · REQ-001…REQ-008 closed; REQ-009 (checklist G-08) open until the spec refresh

Status goes to Shipped in Commit C, after RM interpretation and the spec refresh.

Committed at Commit B (parent `2a1dc14e`) with the checklist and `INITIAL_REQUIREMENTS.md` close records. No committed bundle changes: each keeps `milestone_counted` false (section 4). `ROADMAP.md`, `ARCHITECTURE_OVERVIEW.md`, and `TECH_STACK.md` change only in the spec refresh.

## 1. Slices delivered

| Slice | Focus | Key artifacts |
|-------|--------|----------------|
| task-1 | counted E-VQE width 4 | C2 `ddde49ac`; closeout `f524c200` |
| task-2 | counted E-VQE width 6 | C2 `6707892a`; closeout `5d93ca07` |
| task-3 | counted E-VQE width 8 | C2 `939d4908`; closeout `1eb54ddb` |
| task-4 | width-4 attribution routes | C2 `5f63a9d6`; closeout `064a6f4f` |
| task-5 | widths 6 and 8 attribution routes | C2-w6 `78ba7108`; C2-w8 `3feffb07`; closeout `54ac4e14` |

## 2. Milestone acceptance criteria — final status

Layer 1 success conditions, `DETAILED_PLANNING_CPP_PYTHON_INTEROP_PROFILE.md` §3 (goals G1–G9):

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Counted set is E-VQE at 4, 6, and 8 plus R-base, R-fused, R-strict, and R-hybrid; R-oracle excluded (G1) | met | E-VQE in the three E-VQE bundles; the four route ids in each routes bundle; no R-oracle row; `validate_interop_bundle*` and `validate_attribution_route_bundle` OK (slice closeouts) |
| 2 | Each E-VQE width: at least 1000 warmed calls, O, one-sided 95 % UB, four components, ns per ρ-entry per operation, lawful verdict (G2, G3) | met | 50 warm-up and 1000 counted paired calls per width; UB 1.0589 % / 0.2149 % / 0.0744 %; `throughput.mean_ns_per_op` 13.01 / 11.01 / 10.58; QA-007 met (section 3) |
| 3 | Four attribution-only routes report orchestration, apply component, throughput, and uncertainty, and no O (G4) | met; R-strict satisfied by ADR-F5A-011 | R-base, R-fused, and R-hybrid timed at 4, 6, and 8 (50 warm-up, 1000 counted calls each); R-strict refusal row with diagnosis and no numbers (section 3.1); no row carries O |
| 4 | Zero reductions, or one binding or dispatch reduction (G5) | met: zero | A4 kill fired (UB below 5 % at every width); `git diff --exit-code 1b123a9a -- squander/src-cpp/density_matrix squander/partitioning` empty at `2a1dc14e` |
| 5 | One sibling lane regenerates the bundles with REQ-006 provenance; Phase 3 records untouched (G6) | met | `benchmarks/density_matrix/interop_profile/validation_pipeline.py`; clean-start counted runs; (g) PASS for all six bundles; `git diff --exit-code 1b123a9a -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py` empty |
| 6 | State-vector default, density opt-in, rocky-local Tester CI; ADR-F1A-006 unedited (G7) | met | G-09 rocky-local CI PASS at `2a1dc14e` (below), which includes `test_explicit_state_vector_matches_legacy_default`; also recorded in the task-1, task-3, task-4, and task-5 closeouts |
| 7 | 17/9/0 and the archive unchanged (G8); both current-state docs name the lane (G9) | G8 met; G9 open | `git diff --exit-code 1b123a9a -- docs/density_matrix_project/archive` empty; G9 is REQ-009 and checklist G-08 |

Upstream and ADR dispositions:

| Id | Disposition | Evidence |
|----|-------------|----------|
| CAP-004 | hold-the-line | A4 kill fired; no reduction; `CHANGE_CONTROL.md` §6 |
| CAP-007 | met | six regenerable counted bundles, (g) PASS for each; `task-1/CLOSEOUT.md` … `task-5/CLOSEOUT.md` |
| QA-007 | met on the E-VQE cells | QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. UB ≤1.06 % at 4/6/8; `INITIAL_REQUIREMENTS.md` §5 |
| QA-008 | met | categorical regeneration exact at (g) for all six bundles; E-VQE `|Δ mean O|` within the 0.02 margin; interop evidence pipeline |
| QA-009 | met | G-09 rocky-local CI lane PASS at `2a1dc14e` |
| ADR-F5A-001 | held | inventory (row 1); R-oracle excluded |
| ADR-F5A-002 | held | harness-only lower call; see REQ-003 and `tests/VQE/test_vqe_interop_harness.py` |
| ADR-F5A-003 | held | QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. `INITIAL_REQUIREMENTS.md` §5 (checklist G-06) |
| ADR-F5A-004 | held | sibling lane; Phase 3 records untouched (row 5) |
| ADR-F5A-005 | held | zero reductions (row 4) |
| ADR-F5A-006 | held | one frozen generated-HEA anchor per width; the 26-case matrix is not the workload |
| ADR-F5A-007 | held; docs part open | rocky-local CI lane (row 6); ADR-F1A-006 unedited; current-state docs in the spec refresh (REQ-009) |
| ADR-F5A-008 | held | the tracer was task-1, E-VQE at width 4 |
| ADR-F5A-009 | held | harness timer shipped in task-1 (`ca2bf7c4`, `06b91d6c`) |
| ADR-F5A-010 | held | R-strict `handback_refused` row in all three routes bundles; Q1b wording amendment |
| ADR-F5A-011 | held | REQ-004 amendment and success-check deferral in `ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md`; see sections 3.1 and 5 |

Close gates:

| Gate | Status | Record | Notes |
|------|--------|--------|-------|
| Amendment Commit A | passed | `2a1dc14e`; parent `30bb380d` | this tree |
| Reviewer APPROVE | passed | `36d725a6` | Commit A gate |
| Slice closeouts | passed | section 1 | `task-1/` … `task-5/CLOSEOUT.md` |
| Rocky-local CI | PASS | tree `2a1dc14eab926f6a052acb9ae08eaff640f752ac`; `/tmp/mf5a-close-ci/REPORT.md` `077c56d6d2896e178304c7740fe3498b447b29b00efce1d067dd83448974574b` | full `pytest tests/`; exit 0; 1287 passed, 1 skipped, 1 deselected; 588 s |
| Reviewer full milestone review | REQUIRED CHANGES; RC-1…RC-8 applied in this file; APPROVE FOR COMMIT B on the narrow re-gate | `/tmp/rev-mf5a-milestone/REVIEW.md` `3ecb2ff5` (claude-opus-5-5 1m xhigh, 2026-10-08) | re-gate binder named in the Commit B message |
| RM interpretation | later | not in this closeout | after the milestone review |
| Demo | later | not in this closeout | after RM interpretation |
| Spec refresh | later | `ROADMAP.md`, `ARCHITECTURE_OVERVIEW.md`, `TECH_STACK.md` | after Demo |

The CI tree is a clean clone of Commit A at `/tmp/mf5a-close-ci/repo`, docs only on top of `30bb380d`. The command is the full `pytest tests/ -x -v --tb=line` (G-09, REQ-007): the `ci.yml` linux line, which ignores `tests/decomposition/test_wide_circuit_optimization.py`, plus the standing rocky-local deselect of the known-flaky `tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2` (the same deselect G-04 used at `031996f4`), with no `taskset`. PLAN `/tmp/mf5a-close-ci/PLAN.md` sha256 `4b168f9c519f90b529627b4176b7dc7f5e34e94e4774cde156edaed201ee540e` was checked on this host. Timing addendum `/tmp/mf5a-close-ci.ADDENDUM-timing.md` sha256 `73affff7db23c5a1a394c4d053f08a88fb6a05250edb3e863a25551e18f17e6d` (final). Wall time 588 s, 11:54–12:04 CEST (UTC+2), 2026-10-08. JUnit `tests=1288`. The one skip is `tests/gates/test_float32_performance.py::test_float32_apply_to_hot_path_has_expected_speed[U3]`, which skipped itself with "Machine in degraded state — float32 (0.052s) not faster than float64 (0.037s)" (JUnit); it is a non-density-matrix performance self-check, not a failure, and G-04 had 0 skips. Wall time was about 9.8 min this time against G-04's 40m56s, even with 218 more tests (`tests/VQE` interop); the likely cause is BLAS threading or host load, but it cannot be pinned down from the logs, and it is not a gate issue. The first build launch exited before the build started, because `conda` was not on PATH in the non-interactive shell; after the PATH fix the clean build ran once (exit 0, 21.35 s) before the single pytest launch.

## 3. REQ-* evidence matrix (milestone scope)

| REQ | CAP / QA | Delivered | Evidence |
|-----|----------|-----------|----------|
| REQ-001 | CAP-004, CAP-007 · QA-007, QA-008 | met | inventory in all six bundles (section 2, row 1); `benchmarks/density_matrix/interop_profile/interop_bundle_validation.py` and `attribution_route_validation.py`; `task-1/CLOSEOUT.md` … `task-5/CLOSEOUT.md` |
| REQ-002 | CAP-004 · QA-007 | met | `interop_profile_bundle.json`, `interop_profile_bundle_w6.json`, `interop_profile_bundle_w8.json` at `ddde49ac`, `6707892a`, `939d4908` (pins in section 5): 1000 counted paired calls, O, UB, four components, ns/op. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. (checklist G-06, §13); UB 1.0589 % / 0.2149 % / 0.0744 % meets it. The bundles keep their generation-time "QA-007 withheld" labels (section 4) |
| REQ-003 | CAP-004 · QA-007 | met | harness-only lower call into the C++ density branch (ADR-F5A-002, ADR-F5A-009); flag-off versus flag-on bit identity in `tests/VQE/test_vqe_interop_harness.py`; no public Python energy symbol; `task-1/CLOSEOUT.md` |
| REQ-004 | CAP-004, CAP-007 · QA-008 | met for R-base, R-fused, R-hybrid; satisfied for R-strict by ADR-F5A-011 | R-base, R-fused, R-hybrid timed at 4, 6, and 8 (`task-4/CLOSEOUT.md`, `task-5/CLOSEOUT.md`); R-strict refusal row with diagnosis at each width (section 3.1, `CHANGE_CONTROL.md`) |
| REQ-005 | CAP-004 · QA-007 | satisfied as no reduction | A4 kill fired; Python-boundary success-check deferral signed off in `CHANGE_CONTROL.md` §6; REQ-005 diff row empty (section 2, row 4) |
| REQ-006 | CAP-007 · QA-008 | met | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/interop_profile/validation_pipeline.py` (per-width and `--attribution-routes` forms in the slice closeouts); clean-start counted runs; (g) PASS for all six bundles; Phase 3 diff row empty; compiler flags not pinned in the bundles (N-46, section 5) |
| REQ-007 | QA-009 | met | rocky-local CI lane PASS; tree `2a1dc14eab926f6a052acb9ae08eaff640f752ac`; `/tmp/mf5a-close-ci/REPORT.md` `077c56d6d2896e178304c7740fe3498b447b29b00efce1d067dd83448974574b`; ADR-F1A-006 unedited |
| REQ-008 | CAP-007 · QA-008 | met | `git diff --exit-code 1b123a9a -- docs/density_matrix_project/archive` empty; 17/9/0 unchanged; no at-least-1.2× sentence in the bundles or closeouts (`task-4/CLOSEOUT.md`, `task-5/CLOSEOUT.md`, this file) |
| REQ-009 | CAP-007 · QA-008, QA-009 | open | `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` name the lane in the spec refresh (checklist G-08); `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict` |

### 3.1 REQ-004 for R-strict

Where the strict contract refuses under the frozen workload, a required refusal row with recorded diagnosis satisfies REQ-004 for R-strict. The row carries no timings, ns/op, UB or O. Inventing strict timings or changing the anchor workload's noise remains forbidden.

REQ-004 is satisfied for R-strict by that refusal row. The diagnosis is `channel_native_noise_presence`. The rows are `handback_refused` and carry no timings. Handback `98eec857`. Width 4: `task-4/CLOSEOUT.md`, bundle `6584be2bc49d55c995c54d1be9a7b7efb2b2dfe4cdd0004fd030b5596f9196aa` at `5f63a9d6`. Width 6: `task-5/CLOSEOUT.md`, bundle `212758709a06c2805feef1181111ed980172877ad8b11e66f68bfafb9b11d1be` at `78ba7108`. Width 8: `task-5/CLOSEOUT.md`, bundle `a9e375a1c4c7c39f908ed7136cc461403e315c232ec1a1d4d60fbf4da28fa84f` at `3feffb07`.

Zoltán via PhD Manager, 2026-10-08 10:41 CEST (UTC+2): "C1: accept the documented strict refusal, and I sign off the deferral: no reduction, the Python layer is shown not to be a bottleneck."

### 3.2 Exactness and apply-time ratios

The timed routes agree with the step-by-step (sequential) reference to ≤1.2e-16 and with an independent simulator (Qiskit Aer 0.17.2, given the same circuit and noise specification) to ≤5e-16, i.e. at double-precision machine precision (RM N-c wording; brief `2026-10-08-mf5a-task4-w4-routes-interpret.md`, N-c revision `f7f9dca1…`).

Comparisons that ran, by width (max |Δρ| for R-base / R-fused / R-hybrid; tolerance 1e-10):

- Width 4: sequential 0 / 1.11e-16 / 9.71e-17 (task-4 C2 binder `7b48bbc6…` §6.1; `/tmp/mf5a-t4-g/REPORT.md` `6771415c…`). Aer 4.72e-16 / 4.44e-16 / 4.72e-16 (same report).
- Width 6: Aer 2.58e-16 / 2.75e-16 / 2.86e-16 (`/tmp/mf5a-t5-g-w6-r2/REPORT.md` `fc10e00a…`).
- Width 8: Aer 2.22e-16 / 1.99e-16 / 2.10e-16 (`/tmp/mf5a-t5-g-w8/REPORT.md` `d9117dfe…`).

The width-6 and width-8 (g) reports also logged sequential differences as context, at most 8.33e-17, consistent with the ≤1.2e-16 bound; the quoted sequential bound rests on the width-4 data. R-strict refused at every width and has no state. The E-VQE cell energies pass the existing Aer energy nodes at each width (`task-1/CLOSEOUT.md` … `task-3/CLOSEOUT.md`, Independence).

Apply time versus R-base, from `rows[].apply_component.mean_ns` in the three routes bundles (`task-5/CLOSEOUT.md` :113), rounded to 2 decimal places. Current implementation cost, not intrinsic cost; no speed claim. Fused is 2.47×, 3.27×, and 3.43× at widths 4, 6, and 8. Hybrid is 8.98×, 6.57×, and 8.82× at those widths. Per-route ns/op is attribution, not a kernel benchmark, and is not comparable with the E-VQE throughput.

## 4. Assumptions & risks — what this milestone learned

| ID | Learning | Status after milestone |
|----|----------|------------------------|
| A4 | The public Python boundary does not materially limit E-VQE evaluation: UB on O is 1.0589 % / 0.2149 % / 0.0744 % at 4/6/8, below the 5 % kill at every width. C++ `apply_to` is at least 91 % of the summed E-VQE components at every width (`components.mean_apply_to_ns`) | A4 false, as the product statement expected; CAP-004 hold-the-line; no reduction |
| L-1 | Today's fused and hybrid routes spend more apply time than R-base (section 3.2). Mechanisms named at width 4: a generic C++ local-unitary routine on fused regions; Kraus-matrix construction in Python (RM brief `788f688d…`) | implementation cost, not intrinsic; kernel work is out of M-F5a scope |
| L-2 | Under the frozen noise placement (qubits 0 and 1) the channel-native partition is empty, so strict mode refuses at 4, 6, and 8 rather than fall back | by design; REQ-004 satisfied for R-strict by ADR-F5A-011; a strict-route figure needs backlog item C2 |
| L-3 | Counted runs used one core (CPU 0) on a shared host; other users' jobs shared its L3 cache and hyperthread sibling, so timing tails may be inflated. The E-VQE estimator keeps every sample (S-g Measure); negative means at widths 6 and 8 are lawful | means are quoted, tails are not; S-a and S-c deferred (section 5) |

Disclosures carried from the slice closeouts:

- **Oracle independence (G-10).** The E-VQE Aer energy oracle builds its own noise and trace and does not read the cell's energy or ρ; the flag-off versus flag-on bit identity shares the kernel and cannot see a kernel bug (task-1 to task-3 Independence; Tester notes `ab2f6612…`, `992c9c0e…`). The route-versus-sequential comparison is a separate call with its own `DensityMatrix`, but R-base and R-fused's unfused operations share the `NoisyCircuit` kernels with the reference, so it cannot detect a kernel-level bug; only R-fused's `apply_local_unitary` islands and R-hybrid's numpy Kraus path are separate apply code. The task-4 counted run had no oracle/cell pair, so no Tester independence note exists for it (`task-4/CLOSEOUT.md`).
- **Aer limitation.** Aer shares the circuit definition (`get_Qiskit_Circuit`) and the noise specification with SQUANDER. It does not use SQUANDER's density kernels, `NoisyCircuit`, `DensityMatrix`, or the partitioned lowering, so it is independent of kernels and lowering, not of the circuit and noise definition (`task-5/CLOSEOUT.md`). At width 8 Aer's default fusion threshold applies and the harness sets no Aer options, so Aer may have used its own fused path; it is not described as unfused there (RM pack `d038d340…`).
- **Width-6 counted launch.** Two launches at C1 `3a8a0d74`, never concurrent. The first stopped about 7.8 s in, before the pipeline write, and left a headers-only log (`run.log.RECONSTRUCTED` `cde45875…`; NOTE `fb9a0670…`). The second wrote the bundle. Reviewer ruled it not a selective rerun; RM accepted it.
- **Width-6 (g).** Attempt 0 exited 127 before Python started (`conda` not on PATH). Attempt 1 ran the `qgd` interpreter directly; `qa008_route_categorical_exact` failed at `conda_default_env` and Aer did not run (`278730f8…`). The Tech Lead authorized one corrective repeat, r2, in the mini-spec form; it passed (`fc10e00a…`). Reviewer ruled it not a selective rerun; RM accepted it.
- **Build provenance.** The r2 clone was built with the `qgd` cmake 4.2.0; the "cmake 3.31.8" log line came from the outer shell (addendum `ddd8bb25…`).
- **tmux (W8G-6).** Tester addendum A1 (`a1fc9450…`) killed nine idle earlier Tester-role tmux sessions with no child processes. The test-density-matrix skill now limits kills to sessions the current run created (`30bb380d`).
- **Width-8 routes revision.** That bundle's `implementation_revision` is C2-w6 `78ba7108`: the C1 code plus the width-6 bundle.
- **Superseded quotation.** The exactness bound in the width-4 RM quotation at `task-4/CLOSEOUT.md` :22 is withdrawn; the forward correction at :24 carries the replacement (RM N-c).
- **Bundle flags.** All six bundles keep `milestone_counted` false and their generation-time labels ("QA-007 withheld", "milestone not complete"). Both validators reject any other value, and the bundles stay byte-identical. This closeout and the checklist supersede those flags at milestone level (precedent `1b123a9a`).
- **Artifacts.** Only the six counted bundles are committed. The (c) and (g) logs, regenerated bundles, Aer scripts, CI logs, and Reviewer binders live only under `/tmp`, outside the repository. `/tmp` is not durable; `task-5/CLOSEOUT.md` carries the 24-file sha256 keep-list, re-verified at the milestone review.

## 5. Deferred (carry forward)

On 2026-10-08, via PhD Manager and relayed by the Tech Lead, Zoltán signed off: "C1: accept the documented strict refusal, and I sign off the deferral: no reduction, the Python layer is shown not to be a bottleneck." REQ-004 for R-strict is satisfied by ADR-F5A-011. CAP-004 is hold-the-line. REQ-005 is satisfied as no reduction. The percent figure is UB (one-sided 95% upper bound on O, E-VQE bundles) ≤1.06% at 4/6/8: 1.0589% / 0.2149% / 0.0744%. Counted `overhead.upper_bound_95_O` is 0.010589466306384618, 0.0021490045376290055, and 0.0007435776878628046 on `ddde49ac:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle.json` (`212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`), `6707892a:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w6.json` (`5257bad23e9fef4afad3f7b8b61f84c7cebd2ef8f85d95a94a02cb794d139d7d`), and `939d4908:benchmarks/density_matrix/artifacts/interop_profile/interop_profile_bundle_w8.json` (`1712dce97ae567460c9058c288c3a01879524dd3fb9cba71603a5f5785e1b21d`). C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08).

QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. Zoltán, 13:01 CEST (UTC+2), via PhD Manager: "Yes, I ratify the 10% Python-overhead bar as frozen for M-F5a."

| Item | Target | Notes |
|------|--------|-------|
| C2 | backlog, not M-F5a | C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08). |
| Rocky-local CI | this close | PASS; tree `2a1dc14eab926f6a052acb9ae08eaff640f752ac`; report `077c56d6d2896e178304c7740fe3498b447b29b00efce1d067dd83448974574b` |
| Reviewer full milestone review | this close | closed: `/tmp/rev-mf5a-milestone/REVIEW.md` `3ecb2ff5` (section 2) |
| RM interpretation | after the review | not written here |
| Demo | after RM interpretation | not written here |
| Spec refresh | after Demo | REQ-009, checklist G-08; roadmap handoff below |
| N-46 | backlog: lane schema revision | Compiler flags are not pinned in the bundles. Accepted as a disclosed deviation at the milestone review: the bundles pin the compiler and both extension sha256, `task-1/CLOSEOUT.md` records the flags from `build.ninja`, and every (g) clone rebuilt its own extensions and matched every categorical pin |
| N-16, N-17 | before the first pull request into `master` | no MSVC branch for `clock_gettime`; host-pinned ET-2 golden energy (ADR-F5A-007) |
| N-23 | backlog | the six timer fields can race under batched `optimization_problem` with the flag on; the harness is single-threaded by contract |
| Validator hardening | backlog | N-35, N-36, N-37, N-78, N-111, width-8 binder N-5, and the `provenance_pass` / `conda_default_env` code change; the committed bundles passed (e) and (g) |
| N-24 | backlog | interop tests sit outside the `density_matrix` marker; G-09 collected them through the full `pytest tests/` |
| S-a, S-c | backlog | the QA-008 margin 0.02 on mean O is about 16 standard errors at width 6 (N-52), so it checks gross reproducibility only; the UB formula treats the 1000 samples as independent (lag-1 or batch-means uncertainty not applied) |
| N8 | deferred (ADR-F5A-007) | ADR-F1A-006 still names GitHub Actions; not edited in M-F5a |
| Slice nits | Research Manager or backlog | RM N-c and N-y; task-2 N-79 and N-80; task-3 N-104 to N-112 and N-119; task-4 and task-5 Carries. All non-blocking |

REQ-005 is satisfied as no reduction. CAP-004 is hold-the-line. `milestone_counted` stays false in every bundle (section 4).

## 6. Evidence commands (reproduce)

The counted commands are the ones in `task-1/CLOSEOUT.md` through `task-5/CLOSEOUT.md`. This closeout does not re-run them. The milestone-level checks are below; each `git diff` is empty at `2a1dc14e`.

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
git diff --exit-code 1b123a9a -- squander/src-cpp/density_matrix squander/partitioning
git diff --exit-code 1b123a9a -- benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/benchmark_perf.py
git diff --exit-code 1b123a9a -- docs/density_matrix_project/archive
# G-09, clean clone of 2a1dc14e; launcher /tmp/mf5a-close-ci/run_ci.sh adds /usr/bin/time -p and --junitxml
cd /tmp/mf5a-close-ci/repo && unset PYTHONPATH
PYTHONDONTWRITEBYTECODE=1 conda run -n qgd --no-capture-output env PYTHONPATH=/tmp/mf5a-close-ci/repo \
  pytest tests/ -x -v --tb=line \
  --ignore=tests/decomposition/test_wide_circuit_optimization.py \
  --deselect tests/decomposition/test_QX2.py::Test_Decomposition::test_N_Qubit_Decomposition_QX2
```

## 7. Handoff — create-product-roadmap

These steps belong to the spec refresh, after RM interpretation and Demo. This closeout does not perform them.

- Mark M-F5a Delivered in `ROADMAP.md` only in the spec refresh, after REQ-009 (checklist G-08) closes, and append the revalidation log entry there.
- This closeout does not promote, plan, or authorize the next roadmap milestone; that milestone is not started and is out of scope here.
- Record learnings there, and confirm `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` then.
- A4 closed false, as the product statement expected; its pre-registered consequence is CAP-004 hold-the-line, so no `PRODUCT_STATEMENT.md` escalation is required. No other core assumption was invalidated.

## 8. Layer 2 index (milestone artifacts)

| Path | Purpose |
|------|---------|
| `INITIAL_REQUIREMENTS.md` | REQ-001…REQ-009, including the ADR-F5A-011 line (§5, change log). QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. |
| `ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md` | ADR-F5A-011; source of the Deferred paragraph (`:115`) |
| `CHANGE_CONTROL.md` | C1 sign-off |
| `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` | gap list; G-06, G-08, G-09 |
| `task-4/CLOSEOUT.md` | width-4 routes |
| `task-5/CLOSEOUT.md` | widths 6 and 8; ratio line `:113` |
