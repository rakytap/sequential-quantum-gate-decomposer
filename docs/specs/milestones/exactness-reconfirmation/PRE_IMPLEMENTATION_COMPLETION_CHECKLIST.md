# Pre-implementation completion checklist — M-F1a `exactness-reconfirmation`
> **Status:** Layer 1 v0.1 · **Verdict:** task-1 shipped at C2 `a2928bf1`; task-2 Step 4a
> closed code-ready on 2026-10-05 under ADR-F1A-008; milestone remains open ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Owner skill:** `spec-driven-development` Steps 2–3 ·
> **Inputs:** accepted `INITIAL_REQUIREMENTS.md` v0.3,
> `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md`,
> `ADRS_EXACTNESS_RECONFIRMATION.md` · **Traces:** REQ-001…008 ·
> **Authorization:** Research Manager, 2026-10-05, Step 4b and Developer/Tester handoff for
> the q4 baseline cell only, effective after the Architect code-ready re-close now recorded ·
> **Boundary:** q4 tracer `CLOSEOUT.md` is shipped and C2 is `a2928bf1`; task-2 Step 4a
> is closed code-ready; Step 4b handoff is Tech Lead under ADR-F1A-010

## 1. Readiness rule

Task-1 is shipped for the q4 baseline cell. Task-2 Step 4a is open as planning and is not
code-ready. Every later slice's Step 4b waits on ADR-F1A-010. The milestone remains open.

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
| G-01 | Step 4b for the remaining M-F1a slices | ADR-F1A-010: per slice, Step 4b and the Developer/Tester handoff take effect when Architect closes Step 4a code-ready under ADR-F1A-008 and the verdict states the oracle, QA-001 and comparators (apart from the ADR-F1A-009 Amendment 1 allowlist entry), counted set, and scope and G-07 exit rule unchanged; otherwise Research Manager. q4 baseline cell authorized 2026-10-05 (history) | Research Manager decision record 2026-10-05 §(b) / Architect | **in force per slice** |
| G-02 | q4 tracer planning review closed as code-ready on 2026-10-05 under ADR-F1A-008 | closed code-ready verdicts in `task-1/TASK_1_MINI_SPEC.md` and `task-1/ENGINEERING_TASKS.md` | Architect / SDD planning role | **closed — code-ready for q4 baseline cell only** |
| G-03 | Existing evidence lacks complete route-anchor coverage and M-F1a provenance | future slice delivery against goals G1–G5 | future implementation evidence | open — implementation outcome |
| G-04 | CI does not trigger automatically on this feature branch | ADR-F1A-006 requires local preflight and the actual Linux CI job through the existing `workflow_dispatch` trigger only at closure; trigger-policy changes remain out of scope | future closeout | closed as gate definition |
| G-05 | Current-state docs do not yet describe M-F1a | ADR-F1A-007 defers truthful updates until milestone close | future closeout | closed as timing decision |
| G-06 | Normal and strict checks after the real q4 closeout | commands in §6 below | SDD planning role | closed as run: normal 0 errors/0 warnings; strict 0 errors/0 warnings; no `SLICE_MISSING_CLOSEOUT`; traceability clean |
| G-07 | Process-exit aggregate adds the required sibling, excludes exactly external correctness and the whole output-integrity suite, retains every other registered suite, preserves excluded statuses, and fails on missing/failing sibling or included suite | Architect's 2026-10-04 exit contract written consistently into the mini-spec, pipeline evidence row, DS-3, and ET-3 | Architect | **closed as slice contract; does not make code-ready** |
| G-08 | Real q4 `CLOSEOUT.md` exists; task-2 has no closeout yet, so strict lint reports that slice's `SLICE_MISSING_CLOSEOUT` as the ADR-F1A-008 planning-stage finding | `task-1/CLOSEOUT.md` status `shipped` | SDD planning authority | **q4 closed; task-2 closeout still absent until its step (d)** |
| G-09 | Two-commit close per slice (ADR-F1A-009, Amendment 1) | q4 tracer: C1 `a50ae79f`, C2 `a2928bf1`; later slices record C1 and C2 here in the next planning pass. The q4 step-(g) result ("6 differences, 0 unexpected") exists only in the Tester's off-repo observation and is unverified | Research Manager / Reviewer / Tech Lead | **q4 closed; rule in force per slice** |
| G-10 | Pre-CLOSEOUT scientific independence gate | Tester written independence confirmed on the dirty run and the counted run; `task-1/CLOSEOUT.md` states the bitwise agreement and the shared-kernel limitation | Tester / Reviewer | **closed** |

No requirements-level or cross-work-package design question remains open. G-02 records the
Architect's q4-only code-ready re-close. G-01 retains the q4 authorization as history and cites
ADR-F1A-010 for Step 4b on the remaining M-F1a inventory: each slice's handoff follows that
slice's Step 4a code-ready close with the four ADR-F1A-010 items unchanged. G-08 stays closed for the q4 closeout; task-2's missing closeout stays until step (d), and with stage `step-4b-authorized` it is a strict error. G-09 records q4 C1 `a50ae79f` and C2 `a2928bf1`, and the per-slice rule stays in force. G-10 is closed by the Tester confirmation recorded
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
| Slice tracer | task-1 shipped for `phase2_xxz_hea_q4_continuity` on `partitioned_density_descriptor_baseline`; C1 `a50ae79f`; C2 `a2928bf1`; task-2 Step 4a closed code-ready on 2026-10-05 under ADR-F1A-008 |
| Required boundary decisions | ADR-F1A-001…011 accepted (008 Amend1, 009 Amend1, 010, and 011 in `ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`) |
| Evidence lanes | detailed plan §9 |
| Current-state doc impact | both existing docs update at milestone close |
| Build impact | no C++/CMake change is planned by Layer 1; any later such proposal requires rebuild |
| Release / rollback | additive sibling package; rollback boundary in ADR-F1A-007 |
| Change control | unnecessary unless a frozen contract is proposed to change |
| Operational failure | counted disagreement fails and freezes downstream claims |

G-01 and ADR-F1A-010 govern Step 4b: q4 is historical (task-1 shipped); task-2 and later slices
authorize Step 4b only after each slice's Step 4a code-ready close. ADR-F1A-009 (and Amend1)
govern the per-slice two-commit close. No push or pull request.

P0b landed at `5a5168fd`. `ADR_AMENDMENTS_<SLUG>.md` is checker-visible (slug check and
the 400-line budget). Manual line count of
`ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md` is 201 of 400.
`ADRS_EXACTNESS_RECONFIRMATION.md` is the `43d8f074` bodies plus the continuation index.

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
| Authorization drift | Accepted v0.3 predates the later planning authorizations | plan/checklist record Layer 1, q4 Step 4b as history (C1 `a50ae79f`, C2 `a2928bf1`), and ADR-F1A-010; task-2 Step 4a closed code-ready on 2026-10-05 |
| Scope pressure | Aer, energy, timing, speedup, new channels, and later milestones could enter through existing rows | Scope §2 and ADR-F1A-002/-004/-005 keep them non-counted, historical, or out |
| Documentation timing | Planning could make current-state docs describe unshipped behavior | ADR-F1A-007 defers updates until executable evidence passes |

Every finding is tightened into a contract or recorded boundary; none is silently deferred.

## 8. Verdict

**Planning review remains closed as code-ready by Architect on 2026-10-05 under ADR-F1A-008,
for task-1's q4 baseline cell only. Step 5 `CLOSEOUT.md` is written and shipped for that
cell, and C2 is `a2928bf1`.** The milestone remains open. G-03 remains open. G-08 is closed
for that q4 closeout. Task-2 Step 4a closed code-ready on 2026-10-05 under ADR-F1A-008.
Task-2 has no closeout until step (d), so `SLICE_MISSING_CLOSEOUT` remains, with no waiver.
With `**SDD stage:** step-4b-authorized` that finding is a strict error, not the step-4a
exemption. G-10 is closed. G-09 records the q4 two-commit close as done and keeps the
per-slice rule in force. Step 4b handoff is Tech Lead under ADR-F1A-010.
