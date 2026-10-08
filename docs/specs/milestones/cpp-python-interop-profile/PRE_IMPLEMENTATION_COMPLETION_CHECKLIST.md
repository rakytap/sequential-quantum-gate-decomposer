# Pre-implementation completion checklist — M-F5a `cpp-python-interop-profile`
> **Status:** Layer 1 v0.1 · **Verdict:** task-4 closed by `task-4/CLOSEOUT.md` (C2 `5f63a9d6`, w4 routes `6584be2b…`, (g) PASS); task-5 closed by `task-5/CLOSEOUT.md` (C1 `3a8a0d74`, C2-w6 `78ba7108`, C2-w8 `3feffb07`, (g) PASS both widths); REQ-004 for R-strict is satisfied by ADR-F5A-011; `CHANGE_CONTROL.md` records the sign-off; milestone closeout `CPP_PYTHON_INTEROP_PROFILE_CLOSEOUT.md` committed at Commit B; G-06 and G-09 closed; G-08 open until the spec refresh ·
> **Milestone:** M-F5a `cpp-python-interop-profile` ·
> **Owner skill:** `spec-driven-development` Steps 2–4a ·
> **Inputs:** `INITIAL_REQUIREMENTS.md` v0.1,
> `DETAILED_PLANNING_CPP_PYTHON_INTEROP_PROFILE.md`,
> `ADRS_CPP_PYTHON_INTEROP_PROFILE.md`, `ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md` · **Traces:** REQ-001…009 ·
> **Authorization:** task-1 C2 `ddde49ac`; task-2 C2 `6707892a`; task-3 C2 `939d4908`; (e) APPROVE; (g) PASS; QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. ·
> **Boundary:** task-4 C1 `6be6282f`, C1-ET4 `431a5808`, C2 `5f63a9d6`; R-strict refusal row is in the w4 lane; task-5 C1 `3a8a0d74`, C2-w6 `78ba7108`, C2-w8 `3feffb07`; REQ-004 for R-strict is satisfied by ADR-F5A-011; milestone closeout committed; not Delivered until G-08 closes; no push; no pull request ·
> **Baseline:** `1b123a9a31235dd68d6c0a6ff9ba457c0112cd59`

## 1. Readiness rule

**Verdict: task-1, task-2, and task-3 stay closed. Task-4 is closed by `task-4/CLOSEOUT.md`; (g) is PASS. Task-4 C1 is `6be6282f`. C1-ET4 is `431a5808`. C2 is `5f63a9d6`. Width-4 routes are counted. Task-5 C1 is `3a8a0d74`. C2-w6 is `78ba7108`. C2-w8 is `3feffb07`. (g) is PASS at both widths. Task-5 is closed by `task-5/CLOSEOUT.md`. The Research Manager has not vetoed N-y; it stays FYI. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. The milestone closeout is committed. M-F5a is not Delivered until G-08 closes in the spec refresh.**

The Tech Lead opened the tracer. `task-1/` holds the planning pack and `CLOSEOUT.md`.
ADR-F5A-009 is filed. C1 tip is `ca5589e2`. C2 is
`ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df`. Reviewer (e) approved it and step (g)
passed. Section 11 is that record. The counted run recorded `clean_start` true.
QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. This checklist does not mark the milestone
delivered. Section 12 records task-2 through (g). Section 13 records task-3 through (g), the RM ALIGN, and task-4 C2. Width-4 rows do not close REQ-004. Task-4 C1 is `6be6282f`. C1-ET4 is `431a5808`. Task-4 C2 is `5f63a9d6`. The w4 routes bundle is `6584be2b…`. ADR-F5A-010 is in the w4 lane. Widths 6 and 8 are counted in task-5. `task-5/CLOSEOUT.md` records them.
The Developer does not edit `docs/specs/**`.

The code-ready gate, already met before C1, required all of the following:

- the v0.1 requirements, this plan, and ADR-F5A-001…009 are still consistent;
- the Tech Lead has opened Step 4a, and that slice's mini-spec, stories, and engineering
  tasks exist with a code-ready verdict;
- the verdict records the harness mechanism, the E1 label decision, the per-width depth
  and noise schedule, and the protocol pins, and states that the equal-work pair, the
  inventory, the no-\(O\) rule, and the kernel/fusion/AVX boundary are unchanged;
- the product owner has frozen the QA-007 bar in `INITIAL_REQUIREMENTS.md` before any
  counted trial that applies the bar and before any "QA-007 met" label;
- at code-ready, normal `specs_check.sh` had 0 errors and the only finding was absent-closeout `SLICE_MISSING_CLOSEOUT`; after `CLOSEOUT.md`, both modes are clean of that finding;
- any proposal to change a frozen contract returns to planning and, where required, to
  `CHANGE_CONTROL.md`.

## 2. Layer 1 contract review

| Item | Closure artifact | State |
|------|------------------|-------|
| Layer 1 authorization after v0.1 and RM E1–E3 | detailed-plan header; this checklist | closed |
| Purpose, source hierarchy, in/out scope | detailed plan §§1–2 | closed |
| Success conditions and stop rule | detailed plan §3 | closed |
| Counted inventory and E1 exclude | ADR-F5A-001 | closed |
| Harness-only pair; no new public Python energy API | ADR-F5A-002 | closed |
| QA-007 10 % bar left `[confirm]` | ADR-F5A-003 | closed as an explicit non-freeze |
| A4 kill at 5 % separated from that bar | ADR-F5A-003, ADR-F5A-005 | closed |
| Sibling lane; M-F1b records untouched | ADR-F5A-004 | closed |
| Material-term rule for the one reduction | ADR-F5A-005 | closed |
| Workload class; 26-case matrix excluded | ADR-F5A-006 | closed |
| Rocky-local CI; ADR-F1A-006 not edited; docs at close | ADR-F5A-007 | closed |
| Tracer limited to E-VQE at 4; no pre-planned later slices | ADR-F5A-008 | closed |
| Boundary map and dependency direction | detailed plan §6 | closed |
| Every REQ mapped to a goal and an ADR | detailed plan §8 | closed |
| Every REQ and in-scope QA has a named evidence route | detailed plan §9 | closed |
| Operational boundaries | detailed plan §10, adopting requirements §6 | closed |
| Release and rollback | ADR-F5A-007 | closed |
| Security, privacy, accessibility, i18n | no new data obligation; requirements §5 | closed |
| Publication artifacts | not spec artifacts and not a gate | closed |
| `CHANGE_CONTROL.md` | not required at Layer 1 | closed |

## 3. Gap list

| Id | Gap | Closes in | Authority | State |
|----|-----|-----------|-----------|-------|
| G-01 | Tech Lead has not opened the Step 4a tracer | `task-1/` planning pack | Tech Lead | **closed — task-1 code-ready; stage `step-4b-authorized`** |
| G-02 | Harness mechanism that calls C++ `optimization_problem` without a new public Python energy symbol | `task-1/TASK_1_MINI_SPEC.md` §4 and §6; ADR-F5A-009 | Architect, Step 4a | **closed — timer on `Variational_Quantum_Eigensolver_Base`; ADR-F5A-009 filed** |
| G-03 | Whether one R-oracle row is required to label C++ `apply_to` | `task-1/TASK_1_MINI_SPEC.md` §5 | Architect, Step 4a | **closed — excluded; inner `apply_to` timer labels the component** |
| G-04 | Depth and noise schedule per width, and the parameter vector | `task-1/TASK_1_MINI_SPEC.md` §2; `task-2/TASK_2_MINI_SPEC.md` §2; `task-3/TASK_3_MINI_SPEC.md` §2 | Architect, Step 4a | **closed for 4 and 6; width 8 pinned in `task-3` §2, RM ACCEPT 2026-10-07** |
| G-05 | Paired versus interleaved, warm-up count, affinity, thread count, uncertainty estimator, and any divisor other than \(4^n\) | `task-1/TASK_1_MINI_SPEC.md` §7; `task-2/TASK_2_MINI_SPEC.md` §4; `task-3/TASK_3_MINI_SPEC.md` §4 | Architect, Step 4a | **closed for the 4-qubit tracer; task-2 §4 carries these pins at width 6; task-3 §4 carries them unchanged at width 8** |
| G-06 | Product-owner QA-007 numeric bar | an edit of `INITIAL_REQUIREMENTS.md` | Product owner | **closed — QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. Met there (UB 1.0589 % / 0.2149 % / 0.0744 %); recorded in `INITIAL_REQUIREMENTS.md` §5 and its change log at milestone close** |
| G-07 | Sibling interop lane is not on disk | the slice that creates it, after code-ready | Step 4b | **closed — lane and counted bundle are on disk; see `task-1/CLOSEOUT.md`** |
| G-08 | Current-state docs do not yet name the lane | milestone close, ADR-F5A-007 | SDD at close | **open — closes in the spec refresh (REQ-009), after RM interpretation and Demo** |
| G-09 | Rocky-local Tester CI record for a code close | that later close | Tester | **closed — rocky-local CI PASS at `2a1dc14e`: full `pytest tests/`, 1287 passed, 1 skipped, 1 deselected, exit 0, 588 s (`/tmp/mf5a-close-ci/REPORT.md` `077c56d6…`); N8 stays deferred** |
| G-10 | Step 4b authorization for task-1 | code-ready verdict on task-1 | Architect | **closed — task-1 stage `step-4b-authorized`; task-1 closed at C2** |
| G-11 | Step 4b authorization for task-2 | planning-role stamp after the Reviewer writer gate | Planning role | **closed — C2 `6707892a`; (e) APPROVE; (g) PASS; stage `step-4b-authorized`** |
| G-12 | Step 4b authorization for task-3 | planning-role stamp after the Reviewer code-ready writer gate | Planning role | **closed — C2 `939d4908`; (e) APPROVE; (g) PASS; stage `step-4b-authorized`** |
| G-13 | Step 4b authorization for task-4 | planning-role stamp after the Reviewer code-ready writer gate | Planning role | **closed — C0 stamp `226ffc29`; C1 `6be6282f` (Q2, B1–B3); C1-ET4 `431a5808`; C2 `5f63a9d6`; w4 bundle `6584be2b…`; (g) PASS; stage `step-4b-authorized`; Option A; width-4 rows do not close REQ-004** |
| G-14 | Step 4b authorization for task-5 (routes at 6 and 8) | planning-role stamp after a code-ready writer gate | Planning role | **closed — C0 `cbb203ae`; C1 `3a8a0d74`; C2-w6 `78ba7108`; C2-w8 `3feffb07`; (g) PASS both widths; stage `step-4b-authorized`; closed by `task-5/CLOSEOUT.md`** |

No open item changes the equal-work pair, adds an advertised energy entry, publishes \(O\)
on an attribution-only route, or proposes a kernel, fusion, or AVX edit. RM ALIGN 2026-10-07: A4 is false, CAP-004 is hold-the-line, and the reduction is not made (ADR-F5A-003/005).
Task-2 RM ACCEPT is `5c810dac…`. Task-3 RM ACCEPT 2026-10-07 is upload `39808966…` and did not flip the stage. The C0 stamp `c4f5df9b` did.
G-11 is closed by the C0 stamp `388a5e5f`, on base `c32b365e`. C1 is `d336472f`. C2 is `6707892a`. (g) PASS.

## 4. Decision closures and trade-offs

- **E1 default exclude.** Keeps the sequential oracle out of the advertised set. The cost
  is a possible later row whose only job is to label `apply_to`.
- **E2 non-freeze.** Keeps the product owner as the owner of the 10 % bar. The cost is
  that "QA-007 met" is untestable until that edit. Reporting \(O\) and the bound is still
  testable.
- **E3 harness-only entry at `optimization_problem`.** Measures the wrapper and keeps
  allocate, build, and the support check on both sides. The cost is that Step 4a must
  find a driver that is not a public Python energy API.
- **Material-term rule.** Makes REQ-005 decidable from the component split and the 5 %
  kill. The cost is that a wrapper which leads at only one width does not justify a
  reduction.
- **Sibling lane.** Protects M-F1b. The cost is a new pipeline instead of a reused one.
- **Tracer at width 4 only.** Proves the pair before width 8. The cost is that the
  milestone outcome is unfinished when `task-1` ships.
- **Docs at close.** Prevents a planned lane from being described as current truth.

## 5. Adversarial critique

Passed before this verdict. Dispositions:

| Finding | Rank | Disposition |
|---------|------|-------------|
| A4, if the pair cannot be built, removes the only QA-007 ratio | blocking if ignored | ADR-F5A-002 stop rule: no new public Python energy API; return to Research Manager |
| "Material term" was not testable | blocking if ignored | ADR-F5A-005: wrapper strictly largest at every width, and the A4 kill has not fired |
| QA-007 10 % bar has no fitness function while `[confirm]` | non-blocking | ADR-F5A-003: withhold "QA-007 met"; the live checks are the protocol, the bound's presence, and the label ban. G-06 tracks the freeze |
| R-oracle re-include could become a fifth route | blocking if ignored | ADR-F5A-001: one diagnosis row, no \(O\), claim boundary required, default exclude |
| Width 8 could be dropped because the tracer starts at 4 | blocking if ignored | ADR-F5A-006 and ADR-F5A-008: 8 stays in the outcome |
| Citing a pipeline that does not exist yet | non-blocking | detailed plan §9 marks the command as created later; static rows use on-disk paths |
| ADR-F1A-006 names GitHub Actions; M-F5a uses rocky-local CI | non-blocking | ADR-F5A-007: do not edit ADR-F1A-006; N8 stays deferred |
| Descriptor mapping for attribution routes is unproven | non-blocking | detailed plan §11: a later slice hands back rather than inventing a second workload |
| Operation-count definition could drift | non-blocking | ADR-F5A-004: operations the timed apply executes, including ordered local noise, unless Step 4a records another divisor before trials |
| Nested component times would make the material-term ranking ambiguous | blocking if ignored | ADR-F5A-004: the four components partition \(T_\mathrm{public}\); the wrapper is \(T_\mathrm{public}-T_\mathrm{lower}\) |
| Requirements header still says Layer 1 stays closed | non-blocking | That sentence is the requirements slice's own scope. The handoff section authorizes Steps 1–3. Detailed plan §1 records the reading. The requirements file is unchanged |

Nothing in that pass remained blocking for the Step 4a gate. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08.
The task-1 slice close did not do that.

## 6. First-slice gate

| Readiness concern | Layer 1 disposition |
|-------------------|---------------------|
| Slice tracer | task-1 Step 4b slice closed; E-VQE at 4 qubits with no reduction; stage `step-4b-authorized` |
| Boundary decisions | ADR-F5A-001…008 accepted; ADR-F5A-009 filed in the companion |
| Evidence lanes | detailed plan §9 |
| Current-state doc impact | both existing docs update at milestone close |
| Build impact | this step edits no C++ or CMake; a later harness driver or wrapper reduction may, and then requires a rebuild before tests |
| Release / rollback | ADR-F5A-007 |
| Change control | unnecessary unless a frozen contract is proposed to change |
| Operational failure | a bad inventory, a one-sided pair, an unlawful "QA-007 met" label, or an \(O\) on an attribution-only route fails the bundle |

## 7. Go / no-go

**Task-1 Step 4b slice closed.** SDD stage stays `step-4b-authorized`. ADR-F5A-009 stays
filed. C1 tip is `ca5589e25036531d599ff963e7d349c6e55b5951`. C2 is
`ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df` (section 11). The counted bundle and
`task-1/CLOSEOUT.md` are the C2 record. At C1, this section said no-go for a counted
run and for `CLOSEOUT.md` (N-39). That wording was true of C1 and is reconciled here.

**No-go** for marking M-F5a Delivered before G-08 closes, for a reduction, for "attribution routes profiled", for a performance-gain claim, VQA, or GHA claim, and for a push or a pull
request. RM ALIGN 2026-10-07 authorizes the E-VQE 4/6/8 claim in planning text. It does not authorize Step 4b.

A draft that changes the equal-work pair, adds an advertised energy entry, publishes
\(O\) on an attribution-only route, or proposes a kernel, fusion, or AVX change stops
and returns to the Research Manager. That trigger is not met by this layer.

## 8. Spec checks

Run on this milestone tree after the three Layer 1 files exist:

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/cpp-python-interop-profile
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict docs/specs/milestones/cpp-python-interop-profile
```

Result after the code-ready stamp, stage `step-4b-authorized`, with no
`task-1/CLOSEOUT.md`. Normal mode: 0 errors and 1 warning, `SLICE_MISSING_CLOSEOUT`.
`--strict`: that finding is the only error and the exit status is 1. Traceability:
no findings. No waiver. No placeholder closeout. The step-4a lint, before this stamp,
kept the same finding as a warning and exited 0. ADR-F1A-008 forbids a planning-status
closeout, so that finding stayed until this real close.

After `task-1/CLOSEOUT.md`, `task-2/CLOSEOUT.md`, and `task-3/CLOSEOUT.md`, those slices are clean.
`task-5/CLOSEOUT.md` is present, so task-5 `SLICE_MISSING_CLOSEOUT` is gone. Normal and `--strict` both exit 0 with no findings. No waiver. No placeholder. With `CPP_PYTHON_INTEROP_PROFILE_CLOSEOUT.md` committed, both modes still exit 0 with no findings.
Task-3 C1 is `97d726e3`. Task-3 C2 is `939d4908`. (g) is PASS. Width-4 route rows do not close REQ-004. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. Task-4 C1 is `6be6282f`. C1-ET4 is `431a5808`. Task-4 C2 is `5f63a9d6`. Bundle `6584be2b…`. `task-4/CLOSEOUT.md` records the width-4 slice. Task-5 C1 is `3a8a0d74`. C2-w6 is `78ba7108`. C2-w8 is `3feffb07`. (g) is PASS at both widths. `task-5/CLOSEOUT.md` records the slice. The live strict raise at widths 6 and 8 is `channel_native_noise_presence`. `pure_unitary_partition` is the hybrid classifier reason. Handback `98eec857…` is unchanged.

## 9. Task-1 Step 4a opened

Opened at `cdcfe6b151e371add2881d7acca45ef47697cdba`. B1–B5 are in the task-1 pack.
That gate was code-ready. G-02 through G-05 stay closed. G-06 is closed. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08.
ADR-F5A-009 is filed. Stage stays `step-4b-authorized`. The C1 sentence that Step 4b
was ahead of Reviewer (a) was true of C1 (N-39). B5 removed the planning-status
closeout. `task-1/CLOSEOUT.md` closes the counted run under ADR-F1A-009. Task-1 only.

## 10. Ask-first harness sign-off (N-12)

The Tech Lead accepted N-12 for the timer flag, the six sub-times, and the accessors.
That is the ask-first harness path in ADR-F5A-009 and `task-1/TASK_1_MINI_SPEC.md` §4
and §6. The sign-off was recorded here before the Reviewer (a) re-gate. N-12 itself does
not record the counted run. The slice close is `task-1/CLOSEOUT.md`.

## 11. C2 record, (g), and carries for the next Step 4a

Task-1 Step 4b is closed at C2 `ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df`.
Reviewer (e) verdict: **APPROVE FOR C2** (bc-b25fac97). Binder
`/tmp/rev-mf5a-c2/REVIEW.md`, sha256
`8749d515be86a3f067e9426062bc768275ebc1ec7e6022cbca0062a050f59e8b`.
Parent is C1 tip `ca5589e25036531d599ff963e7d349c6e55b5951`. This section does
not mark M-F5a Delivered and does not set the QA-007 bar. The next slice is not
opened in this section. Section 12 records task-2 through (g). Section 13 records task-3 through (g). N-41 is
closed in the carry table in this section.

### (g) PASS

Clean-C2 regeneration used the N-34 launch. Exit 0. Wall `real` 2.130 s.
Evidence is `/tmp/mf5a-t1-counted-g/`.

| Check | Result |
|-------|--------|
| Categorical pins | exact match |
| Revision-only change | `ca5589e25036531d599ff963e7d349c6e55b5951` → `ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df` |
| Regenerated `mean_O` | 0.008321 |
| Regenerated `upper_bound_95_O` | 0.010361 |
| Window | [−0.01289, 0.02711]; inside |
| `\|Δ mean O\|` | 0.001209 ≤ 0.02 |
| Restore | sha256 `212f70386bf2a44711d29956c41bd3f0eea9ee2e284ace9c5403bc3d94ef934e`; porcelain empty |

The committed bundle is still the (c) row: `mean_O` 0.00711, bound 0.01059.
Regeneration outputs are not committed.

### Carries

| Id | State | Note |
|----|-------|------|
| N-41 | **closed** | Docs commit `f524c200`. C2 was committed before (e). That docs pass went to Reviewer before its commit. Later commits keep that order. |
| N-42 | folded | Test snapshots and the slice-contract row are in `task-1/CLOSEOUT.md`. ET checkboxes stay unchecked. |
| N-43 | folded | The lane-reusing carries below are listed so the next slice does not drop them. |
| N-44 | folded | QA-007 wording is withheld, not unmet. The bar stays `[confirm]`. |
| N-45 | folded | Independence sentences are in the closeout. The heading no longer uses gap G-10. |
| N-46 | accepted deviation | The flag line is exact for `libqgd.so`. The wrapper `.so` adds `-DCPYTHON`. The bundle does not pin flags. Accepted as a disclosed deviation at the milestone review (`CPP_PYTHON_INTEROP_PROFILE_CLOSEOUT.md` §5); a flags pin is a backlog lane-schema change. |
| N-47 | folded | This section names C2 `ddde49ac1e248e3ea2b4516f420ca1cbbd14e9df`. |
| S-g | accepted | RM ACCEPT 2026-10-07 (task-2, `5c810dac…`): Measure. Width-4: 2/1000 samples exceed 20 µs; excluding them moves 0.00711 to 0.00722. Width-6 uncounted means −0.0011, −0.0012, −0.0001 are lawful, as is `O_i` < −0.5. The 20 µs count is observational. The C0 stamp set `step-4b-authorized`; RM ACCEPT did not flip it. |
| N-16 | open | No MSVC branch for `clock_gettime`. Carry under ADR-F5A-007 until the first pull request into `master`. |
| N-17 | open | The ET-2 golden energy is host-pinned. Same deadline as N-16. |
| N-23 | open | Batched `optimization_problem` can race the six timer fields if the flag is on. Task-2 §3 restates harness-only, single-threaded. The race stays open. |
| N-32 | **closed** | Task-2 ET-2 adds `assert_mean_o_within_margin`. `test_assert_mean_o_within_margin_fixture_pair` passed in G-09 CI at `2a1dc14e`. Width-4 (g) stays the manual check it was. |
| N-35 | open | Validator depth is still thin for a hand-edited bundle. Discharged for this bundle at (e). Carry for later bundles. |
| N-36 | open | The forbidden-path list omits `performance_evidence/` and `benchmark_perf.py` and includes `docs/specs/`. The `git diff` rows stay the gate. |
| N-37 | open | Some validator tests are weak: both clean-start flags flip together, most B2 rejections have no isolated test, and no test feeds a real lane bundle. |
| N-24 | open | The new interop tests sit outside the `density_matrix` marker (861 deselected at the C2 review). |
| N-19 | folded | Task-2 §3 names sub-time clock reads (`clock_gettime`). |
| N-3, N-6, N-10, N-14, N-21, N-25, N-27, N-28, N-29, N-31, N-40 | open | Task-1 descriptions are in `cffe2cab` checklist §11. For task-2, N-3 and N-10 are the §9 containment diff, and N-14 is the width-6 Aer oracle. |
| S-a, S-b, S-c, S-e | open | S-a: margin 0.02; N-52, about 16 SE at width 6, record at the G-06 freeze. S-b: partition tolerance. S-c: lag-1 or batch means, still deferred at width 6 (N-56). S-e: affinity before import. S-d closed at width 4. S-f disposed: no refusal below −0.5. |

## 12. Task-2 Step 4b closed

C2 `6707892a`. (e) APPROVE (`bc-aaf8e41f`). (g) PASS: regen `mean_O` 0.00075526, `|Δ|` 0.001878 ≤ 0.02. w6 `5257bad2…`. QA-007 `[confirm]`. `milestone_counted` false. N-78–N-80 open.

## 13. Task-3 Step 4b closed

(c) PASS at C1 `97d726e3`. Bundle `1712dce9…`. mean_O −0.000106. UB95 0.000744. Spikes 857. `milestone_counted` false, lawful. Independence gate done: Tester `/tmp/mf5a-t3-counted/INDEPENDENCE_NOTE.md` (`992c9c0e…`); width-8 Aer node PASSED from `/tmp`, not skipped. C2 `939d4908`. (e) APPROVE. (g) PASS: regen mean_O −0.000615593972625157, `|Δ|` 0.000510 ≤ 0.02, window [−0.020106, 0.019894]; restore byte-identical `1712dce9…`. Measure carries. A width-8 mean near zero of either sign is lawful; the 20 µs count may saturate; throughput is host-sensitive and is not a (g) gate. RM ALIGN 2026-10-07: E-VQE equal-work interop overhead measured at 4/6/8 qubits under S-g Measure; one-sided 95 % UB on O is below 5 % at every width (A4 false → CAP-004 hold-the-line); QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. `milestone_counted` stays false. Task-4 C1 is `6be6282f`. C1-ET4 is `431a5808`. Task-4 C2 is `5f63a9d6` and is bundle-only: no CLOSEOUT in that commit, lawful for that gate. Tester evidence is `/tmp/mf5a-t4-counted-w4/`. C2 binder `/tmp/rev-mf5a-t4-c2-w4/REVIEW.md` (`7b48bbc6…`). w4 routes bundle `6584be2b…`. Option A (`b0edc658…`) keeps the raise and times R-base, R-fused, and R-hybrid only. Q1a (`5e3c8222…`) is the required refusal-row schema. Width-4 rows do not close REQ-004. The live strict raise at widths 6 and 8 is `channel_native_noise_presence`, the same code as width 4. `pure_unitary_partition` is the hybrid classifier reason. STEP_4A_HANDBACK `98eec857…` is not rewritten. Task-5 plans the width-6 and width-8 route rows; its readiness is in §1 and G-14. No claim that four routes are shipped. No push. No pull request.
Task-5 parked first launch and (g) record: P-1 `/tmp/mf5a-t5-counted-w6-first-launch/` (`run.log.RECONSTRUCTED` `cde45875…`, P-2 `NOTE.md` `fb9a0670…`) and the width-6 (g) disclosures are in `task-5/CLOSEOUT.md` (C2-w8 binder `2cd43938…` §5.4).
