# Pre-implementation completion checklist — M-F1a `exactness-reconfirmation`
> **Status:** Layer 1 v0.1 · **Verdict:** task-1 code-ready; Step 5 CLOSEOUT shipped for
> the q4 tracer cell; milestone remains open; local C2 pending Reviewer then Tech Lead ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Owner skill:** `spec-driven-development` Steps 2–3 ·
> **Inputs:** accepted `INITIAL_REQUIREMENTS.md` v0.3,
> `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md`,
> `ADRS_EXACTNESS_RECONFIRMATION.md` · **Traces:** REQ-001…008 ·
> **Authorization:** Research Manager, 2026-10-05, Step 4b and Developer/Tester handoff for
> the q4 baseline cell only, effective after the Architect code-ready re-close now recorded ·
> **Boundary:** q4 tracer `CLOSEOUT.md` is written; no other route, anchor, workload, or
> slice is authorized, and C2 is not committed

## 1. Readiness rule

Task-1 is code-ready for the q4 baseline cell only. The Research Manager's conditional
Step 4b and Developer/Tester authorization is now effective for that cell because the
Architect code-ready re-close is recorded. The milestone remains open, and every other
route, anchor, workload, or slice remains unauthorized.

Even after authorization, implementation may not begin unless:

- the accepted v0.3 requirements, Layer 1 plan, and accepted ADRs remain internally
  consistent;
- the current slice has objective acceptance, named tests and lanes, a complete evidence
  matrix, rollback expectations, and no unresolved cross-work-package decision;
- both spec checks are clean; and
- any proposed deviation from a frozen Layer 1 contract is returned to planning and, where
  required, governed through `CHANGE_CONTROL.md`.

## 2. Layer 1 contract review

| Item | Closure artifact | State |
|------|------------------|-------|
| Layer 1 authorization after accepted v0.3 baseline | Research Manager instruction, 2026-10-04; detailed-plan header | closed |
| Purpose, source hierarchy, in/out scope, non-goals | detailed plan §§1–2 | closed |
| Exact milestone success and disagreement stop rule | detailed plan §3; ADR-F1A-005 | closed |
| QA-001 predicate exactly preserved | detailed plan §§3–4; ADR-F1A-002 | closed |
| Four-route × four-anchor denominator | ADR-F1A-001 | closed |
| Genuine route realization versus labelled baseline/skip | ADR-F1A-001, ADR-F1A-003 | closed |
| Evaluation-mode label vocabulary and independent witness | ADR-F1A-003 | closed |
| Single regeneration command, provenance, and artifact root | ADR-F1A-004 | closed |
| Aer and energy-continuity rows non-counted | ADR-F1A-002, ADR-F1A-004 | closed |
| Historical packages and frozen 26-case matrix protected | ADR-F1A-005 | closed |
| Local dep/AD/PD and representation policy unchanged | detailed plan §2; ADR-F1A-005 | closed |
| State-vector non-interference gate | ADR-F1A-006 | closed |
| Current-state doc update obligation | ADR-F1A-007 | closed |
| Architecture boundary map and dependency direction | detailed plan §6 | closed |
| Every REQ mapped to goals and decisions | detailed plan §8 | closed |
| Every REQ/QA has a named evidence route | detailed plan §9 | closed |
| Release and rollback expectations | ADR-F1A-007 | closed |
| Security/privacy/accessibility obligations | no new surface or data obligation; v0.3 §5 | closed |
| Publication artifacts | not required and not permitted as spec artifacts | closed |

## 3. Gap list

| Id | Gap | Contract or artifact that closes it | Authority | State |
|----|-----|-------------------------------------|-----------|-------|
| G-01 | Research Manager opened Step 4b and Developer/Tester handoff on 2026-10-05 for the q4 baseline cell only: `phase2_xxz_hea_q4_continuity` on `partitioned_density_descriptor_baseline` | Architect code-ready re-close recorded on 2026-10-05; every other anchor, route, workload, and Step 4a pass for the rest of the 4/6/8/10 inventory requires separate Research Manager authorization | Research Manager / Architect | **authorized and effective — q4 baseline cell only** |
| G-02 | q4 tracer planning review closed as code-ready on 2026-10-05 under ADR-F1A-008 | closed code-ready verdicts in `task-1/TASK_1_MINI_SPEC.md` and `task-1/ENGINEERING_TASKS.md` | Architect / SDD planning role | **closed — code-ready for q4 baseline cell only** |
| G-03 | Existing evidence lacks complete route-anchor coverage and M-F1a provenance | future slice delivery against goals G1–G5 | future implementation evidence | open — implementation outcome |
| G-04 | CI does not trigger automatically on this feature branch | ADR-F1A-006 requires local preflight and the actual Linux CI job through the existing `workflow_dispatch` trigger only at closure; trigger-policy changes remain out of scope | future closeout | closed as gate definition |
| G-05 | Current-state docs do not yet describe M-F1a | ADR-F1A-007 defers truthful updates until milestone close | future closeout | closed as timing decision |
| G-06 | Normal and strict checks after the real q4 closeout | commands in §6 below | SDD planning role | closed as run: normal 0 errors/0 warnings; strict 0 errors/0 warnings; no `SLICE_MISSING_CLOSEOUT`; traceability clean |
| G-07 | Process-exit aggregate adds the required sibling, excludes exactly external correctness and the whole output-integrity suite, retains every other registered suite, preserves excluded statuses, and fails on missing/failing sibling or included suite | Architect's 2026-10-04 exit contract written consistently into the mini-spec, pipeline evidence row, DS-3, and ET-3 | Architect | **closed as slice contract; does not make code-ready** |
| G-08 | Real q4 `CLOSEOUT.md` exists and strict `SLICE_MISSING_CLOSEOUT` is absent, with no waiver and no placeholder | `task-1/CLOSEOUT.md` status `shipped`; normal and strict spec checks fully clean | SDD planning authority | **closed — strict clean after CLOSEOUT** |
| G-09 | Two-commit slice-close order through the counted run and real closeout; C2 is not committed | ADR-F1A-009 steps (a)–(d) are done for the q4 docs and counted run; Reviewer evidence review and the Tech Lead C2 commit remain | Research Manager / Reviewer / Tech Lead | **steps (a)–(d) done; C2 pending** |
| G-10 | Pre-CLOSEOUT scientific independence gate | Tester written independence confirmed on the dirty run and the counted run; `task-1/CLOSEOUT.md` states the bitwise agreement and the shared-kernel limitation | Tester / Reviewer | **closed** |

No requirements-level or cross-work-package design question remains open. G-02 records the
Architect's q4-only code-ready re-close. G-01 is effective for Step 4b and the
Developer/Tester handoff on that cell only. G-08 is closed because the real closeout exists
and strict lint no longer reports a missing closeout. G-09 records steps (a)–(d) as done and
leaves C2 for Reviewer and then Tech Lead. G-10 is closed by the Tester confirmation recorded
in the closeout. G-03 remains open as the implementation outcome; it authorizes no other
slice or scope.

## 4. Decision closures and trade-offs

- **Sibling evidence package:** protects historical schemas and predicates; accepts explicit
  coexistence of labelled historical and M-F1a semantics.
- **Four genuine routes at every anchor:** prevents denominator gaming; requires eligible
  counted workloads rather than treating requested route names as execution.
- **Exact QA-001 only:** preserves the accepted physical-state predicate; gives up any claim
  that a green M-F1a re-closes external protocol or energy agreement.
- **Aer and energy non-counted:** keeps optional context visible while preventing it from
  opening M4 or controlling exactness status.
- **Hybrid-only partition labels:** matches the shipped evaluation surface; does not widen
  strict, baseline, or fused route contracts.
- **CI closure plus local preflight:** the clean Linux CI job proves non-interference; the
  identical local command catches failures early without changing repository trigger policy.
- **Docs at close:** prevents intended behavior from being presented as current truth.

## 5. First-slice authorized scope and remaining gate

| Readiness concern | Layer 1 disposition |
|-------------------|---------------------|
| Slice tracer | task-1 is code-ready and Step 4b-authorized only for `phase2_xxz_hea_q4_continuity` on `partitioned_density_descriptor_baseline`; C1 is `a50ae79f`; C2 awaits Reviewer, then Tech Lead |
| Required boundary decisions | ADR-F1A-001…009 accepted |
| Evidence lanes | detailed plan §9 |
| Current-state doc impact | both existing docs update at milestone close |
| Build impact | no C++/CMake change is planned by Layer 1; any later such proposal requires rebuild |
| Release / rollback | additive sibling package; rollback boundary in ADR-F1A-007 |
| Change control | unnecessary unless a frozen contract is proposed to change |
| Operational failure | counted disagreement fails and freezes downstream claims |

This table records the q4-only Step 4b and Developer/Tester authorization from G-01 and
ADR-F1A-009. C1 is `a50ae79f`. It authorizes no other route, anchor, workload, `task-<n>/`,
push, or pull request. C2 awaits Reviewer, then Tech Lead.

## 6. Verification commands

- `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh`
- `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict`

Before Step 4b, code-ready verification follows ADR-F1A-008: the normal check has 0 errors;
its only allowed warning is `SLICE_MISSING_CLOSEOUT` for a slice that has not reached Step 4b;
traceability is clean; and under `--strict` that same finding is the one recorded known
planning-stage result. It remains visible, is not waived, and is not fixed with a placeholder.
Any other strict finding blocks code-ready.

Step 4b and slice close follow ADR-F1A-009:

1. **(a)** Reviewer completes implementation review of the uncommitted diff.
2. **(b)** After that pass, create local implementation commit C1 containing planning docs,
   implementation, and tests, with no generated artifact and no `CLOSEOUT.md`; an optional
   planning-docs C0 may precede C1.
3. **(c)** From empty `git status --porcelain` at C1, run the single validation pipeline once
   as the counted clean-start run.
4. **Pre-(d) scientific gate G-10:** Tester supplies the written oracle/cell independence
   confirmation. If independence is not established, stop before `CLOSEOUT.md` and do not
   create C2.
5. **(d)** Write the real `CLOSEOUT.md` citing C1 and the bitwise agreement with its reason;
   normal and `--strict` checks and traceability must then be fully clean.
6. **(e)** Reviewer completes evidence review.
7. **(f)** After that pass, create local evidence commit C2.
8. **(g)** Regenerate from clean C2 under the Tech Lead step-8 decision in `task-1/CLOSEOUT.md`: the comparator stays unchanged, a revision-only mismatch exits 1 and is expected, Tester reports a field-level diff, then restore the generated outputs and do not commit them.

## 7. Adversarial critique and disposition

| Attack | Finding | Disposition |
|--------|---------|-------------|
| Riskiest assumption | The sequential oracle and a shipped route may disagree | REQ-006 and ADR-F1A-005 require fail, report, and freeze without retcon |
| Least testable acceptance | “Every advertised route” could be self-defined or requested but unrealized | ADR-F1A-001 derives routes from pinned M3/M3A claims, requires independent manifest review, exact-set validation, and genuine realization |
| QA fitness gap | Existing validity does not equal accepted QA-001; raw `lambda_min(rho)` needs a frozen executable convention | ADR-F1A-002 isolates one predicate, fixes the existing upper-triangle `zheev` convention without symmetrization or a Hermiticity gate, and records every value/failure |
| Attribution gap | A hybrid runtime id or label could exist without channel-native work; supported-unfused lacked a witness rule | ADR-F1A-001/-003 require a witnessed channel-native partition per counted hybrid cell and witness rules for every runtime class |
| Reproducibility gap | Existing metadata does not pin revision, command, denominator, inputs, or clean-start state | ADR-F1A-004 freezes complete provenance and makes a dirty pre-run non-counted or failed |
| Historical-boundary gap | A tracked diff alone misses untracked archive additions; an all-evidence command would execute timing suites | detailed plan §9 and ADR-F1A-005 combine tracked/untracked archive review with targeted correctness regressions only |
| Non-interference gap | A local run cannot replace the clean CI environment | ADR-F1A-006 keeps the actual Linux CI job as closure gate and local `qgd` execution as preflight |
| Authorization drift | Accepted v0.3 predates the later planning authorizations | plan/checklist record Layer 1, q4-only Step 4a, and q4 baseline Step 4b authorization; C1 is recorded and C2 remains pending Reviewer then Tech Lead; every other cell remains unauthorized |
| Scope pressure | Aer, energy, timing, speedup, new channels, and later milestones could enter through existing rows | Scope §2 and ADR-F1A-002/-004/-005 keep them non-counted, historical, or out |
| Documentation timing | Planning could make current-state docs describe unshipped behavior | ADR-F1A-007 defers updates until executable evidence passes |

Every finding is tightened into a contract or recorded boundary; none is silently deferred.

## 8. Verdict

**Planning review remains closed as code-ready by Architect on 2026-10-05 under ADR-F1A-008,
for task-1's q4 baseline cell only. Step 5 `CLOSEOUT.md` is written and shipped for that
cell.** The milestone remains open. G-03 remains open. G-08 is closed because strict lint is
fully clean after the real closeout, with no waiver. G-10 is closed. G-09 steps (a)–(d) are
done; local C2 is pending Reviewer evidence review and then the Tech Lead commit. No other
route, anchor, workload, slice, or milestone scope is authorized.
