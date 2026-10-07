# Pre-implementation completion checklist — M-F5a `cpp-python-interop-profile`
> **Status:** Layer 1 v0.1 · **Verdict:** task-1 Step 4b slice closed; SDD stage `step-4b-authorized` ·
> **Milestone:** M-F5a `cpp-python-interop-profile` ·
> **Owner skill:** `spec-driven-development` Steps 2–4a ·
> **Inputs:** `INITIAL_REQUIREMENTS.md` v0.1,
> `DETAILED_PLANNING_CPP_PYTHON_INTEROP_PROFILE.md`,
> `ADRS_CPP_PYTHON_INTEROP_PROFILE.md`, `ADR_AMENDMENTS_CPP_PYTHON_INTEROP_PROFILE.md` · **Traces:** REQ-001…009 ·
> **Authorization:** task-1 Step 4b closed at C2; C1 tip `ca5589e2`; N-12 signed off; QA-007 `[confirm]` ·
> **Boundary:** slice close only; milestone not complete; no push; no pull request ·
> **Baseline:** `1b123a9a31235dd68d6c0a6ff9ba457c0112cd59`

## 1. Readiness rule

**Verdict: task-1 Step 4b is closed. SDD stage stays `step-4b-authorized`. QA-007 stays `[confirm]`. M-F5a is not complete.**

The Tech Lead opened the tracer. `task-1/` holds the planning pack and `CLOSEOUT.md`.
ADR-F5A-009 is filed. C1 tip is `ca5589e2`. The counted run recorded `clean_start`
true. This checklist does not freeze the 10 % bar and does not mark the milestone
delivered. The Developer does not edit `docs/specs/**`.

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
| G-04 | Depth and noise schedule per width, and the parameter vector | `task-1/TASK_1_MINI_SPEC.md` §2 | Architect, Step 4a | **closed for 4 qubits only; widths 6 and 8 stay later slices** |
| G-05 | Paired versus interleaved, warm-up count, affinity, thread count, uncertainty estimator, and any divisor other than \(4^n\) | `task-1/TASK_1_MINI_SPEC.md` §7 | Architect, Step 4a | **closed for the 4-qubit tracer** |
| G-06 | Product-owner freeze of the QA-007 numeric bar | an edit of `INITIAL_REQUIREMENTS.md` | Product owner | **open — blocks a "QA-007 met" label and any counted trial that applies the bar; does not block Step 4a** |
| G-07 | Sibling interop lane is not on disk | the slice that creates it, after code-ready | Step 4b | **closed — lane and counted bundle are on disk; see `task-1/CLOSEOUT.md`** |
| G-08 | Current-state docs do not yet name the lane | milestone close, ADR-F5A-007 | SDD at close | **open — expected** |
| G-09 | Rocky-local Tester CI record for a code close | that later close | Tester | **open — expected; N8 stays deferred** |
| G-10 | Step 4b authorization | code-ready verdict on the open slice | Architect | **closed — stage `step-4b-authorized`; task-1 Step 4b slice closed at C2** |

No open item changes the equal-work pair, adds an advertised energy entry, publishes \(O\)
on an attribution-only route, or proposes a kernel, fusion, or AVX edit. Research Manager
consult is not required to leave these gaps for Step 4a.

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

Nothing in that pass remained blocking for the Step 4a gate. G-06 stays open and still
blocks a "QA-007 met" label. The task-1 slice close does not close G-06.

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
filed. C1 tip is `ca5589e25036531d599ff963e7d349c6e55b5951`. The counted bundle and
`task-1/CLOSEOUT.md` are the C2 record. At C1, this section said no-go for a counted
run and for `CLOSEOUT.md` (N-39). That wording was true of C1 and is reconciled here.

**No-go** for a "QA-007 met" label, for freezing the QA-007 bar, for marking M-F5a
complete, for a push or a pull request, and for opening the next slice from this close.

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

After `task-1/CLOSEOUT.md`, both modes are clean of `SLICE_MISSING_CLOSEOUT` and
exit 0. QA-007 stays `[confirm]`. The milestone is not complete.

## 9. Task-1 Step 4a opened

Opened at HEAD `cdcfe6b151e371add2881d7acca45ef47697cdba`. Reviewer blockers B1 through
B5 are folded into `task-1/TASK_1_MINI_SPEC.md` and the engineering tasks. That pass's
gate was code-ready. G-02 through G-05 stay closed there. G-06 stays open: the
10 % bar is still `[confirm]`, and `INITIAL_REQUIREMENTS.md` is not edited. ADR-F5A-009
is filed in this pass, copying §6 after W-1, W-2, and W-3. The stage line stays
`step-4b-authorized`. At C1, this section said Step 4b had started and C1 was in
flight ahead of the Reviewer (a) re-gate. That sentence was true of C1 (N-39).

B5 deleted the planning-status `task-1/CLOSEOUT.md` before the counted run. The
counted bundle records `clean_start` true. `task-1/CLOSEOUT.md` now closes that
run under ADR-F1A-009. The close is task-1 Step 4b only.

## 10. Ask-first harness sign-off (N-12)

The Tech Lead accepted N-12 for the timer flag, the six sub-times, and the accessors.
That is the ask-first harness path in ADR-F5A-009 and `task-1/TASK_1_MINI_SPEC.md` §4
and §6. The sign-off was recorded here before the Reviewer (a) re-gate. N-12 itself does
not record the counted run. The slice close is `task-1/CLOSEOUT.md`.
