# Pre-implementation completion checklist — M-F1a `exactness-reconfirmation`
> **Status:** Layer 1 v0.1 · **Verdict:** task-1 shipped at C2 `a2928bf1`; Slice A C2
> `2a2f8c137`; Slice B C1 `22071ecb`, C2 `0e8299e9` (q4 baseline regenerated at clean C2);
> task-4 (C.0) shipped at `91680ec7`; task-5 (C.1) closed at C2 `a006c7e2`;
> task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS; CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`); milestone remains open ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Owner skill:** `spec-driven-development` Steps 2–3 ·
> **Inputs:** accepted `INITIAL_REQUIREMENTS.md` v0.3,
> `DETAILED_PLANNING_EXACTNESS_RECONFIRMATION.md`,
> `ADRS_EXACTNESS_RECONFIRMATION.md` · **Traces:** REQ-001…008 ·
> **Authorization:** Research Manager, 2026-10-05, Step 4b and Developer/Tester handoff for
> the q4 baseline cell only, effective after the Architect code-ready re-close now recorded ·
> **Boundary:** q4 tracer C2 `a2928bf1`; Slice A C2 `2a2f8c137`; Slice B closed at C2
> `0e8299e9` (parent C1 `22071ecb`). Task-4 C.0 is shipped at `91680ec7`. Task-5 (C.1)
> is closed (C1 `7cf11a49`, C2 `a006c7e2`). Task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS; CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`).
> Step 4b for a later slice waits on that slice's ADR-F1A-010 code-ready close.
> **Task-9:** closed code-ready under ADR-F1A-008 on 2026-10-06; stage `step-4b-authorized`; Research Manager freeze record in §10; Step 4b not started; no `CLOSEOUT.md` yet ·

## 1. Readiness rule

Task-1 is shipped for the q4 baseline cell. Slice A and Slice B are closed (Slice B C2
`0e8299e9`; claim: q4 baseline regenerated at clean C2). Task-4 (Slice C.0) is shipped
at `91680ec7`. Task-5 (C.1) is closed: C1 `7cf11a49`, C2 `a006c7e2`, (c) and (g)
PASS. Claim at C1: fused bundle generated at clean C1. Task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS; CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`). Claim: hybrid bundle generated at clean C1.
The milestone remains open.

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
| G-03 | Existing evidence lacks complete route-anchor coverage and M-F1a provenance | C.1 closed at C2 `a006c7e2` with provisional fused evidence; C.2 closed at C2 `42922382` with provisional hybrid evidence; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`); counted coverage is still later | Planner now; counted evidence later | **open — counted coverage remains** |
| G-04 | CI does not trigger automatically on this feature branch | ADR-F1A-006 requires local preflight and the actual Linux CI job through the existing `workflow_dispatch` trigger only at closure; trigger-policy changes remain out of scope | future closeout | closed as gate definition |
| G-05 | Current-state docs do not yet describe M-F1a | ADR-F1A-007 defers truthful updates until milestone close | future closeout | closed as timing decision |
| G-06 | Normal and strict checks after the real q4 closeout | commands in §6 below | SDD planning role | closed as run: normal 0 errors/0 warnings; strict 0 errors/0 warnings; no `SLICE_MISSING_CLOSEOUT`; traceability clean |
| G-07 | Process-exit aggregate adds the required sibling, excludes exactly external correctness and the whole output-integrity suite, retains every other registered suite, preserves excluded statuses, and fails on missing/failing sibling or included suite | Architect's 2026-10-04 exit contract written consistently into the mini-spec, pipeline evidence row, DS-3, and ET-3 | Architect | **closed as slice contract; does not make code-ready** |
| G-08 | Real closeouts exist for q4, Slice A, Slice B, C.0, C.1, and C.2. Task-7 CLOSEOUT committed at C2 `aaf6fcfe`. Task-8 CLOSEOUT committed at C2 `85f01985` | `task-1/CLOSEOUT.md` shipped; `task-2/CLOSEOUT.md` at Slice A C2 `2a2f8c137`; `task-3/CLOSEOUT.md` at Slice B C2 `0e8299e9`; `task-4/CLOSEOUT.md` at `91680ec7`; `task-5/CLOSEOUT.md` at C.1 C2 `a006c7e2`; `task-6/CLOSEOUT.md` at C.2 C2 `42922382`; `task-7/CLOSEOUT.md` at C.3 C2 `aaf6fcfe`; `task-8/CLOSEOUT.md` at C.4 C2 `85f01985` | SDD planning authority | **q4 through C.2 closeouts committed; task-7 CLOSEOUT committed at C2 `aaf6fcfe`; task-8 CLOSEOUT committed at C2 `85f01985`; milestone remains open** |
| G-09 | Two-commit close per slice (ADR-F1A-009, Amendment 1) | q4: C1 `a50ae79f`, C2 `a2928bf1`. Slice A: C1 `92e95f54` (parent `5a5168fd`), C2 `2a2f8c137`. Slice B: C1 `22071ecb` (parent `99bf9d51`; 7 paths; Reviewer `bc-c0712a04`), C2 `0e8299e9` (Reviewer `bc-6c5c16ac`); (c) PASS Tester `bc-9820d3c5`; (g) PASS, revision equals `0e8299e9`. C.1: C1 `7cf11a49` (parent `91680ec7`), C2 `a006c7e2`; (c) and (g) PASS Tester `bc-9820d3c5`. Task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS at `42922382` (72 s, exit 0, `/tmp/c2-g-proof/REPORT.md`); CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`) | Research Manager / Reviewer / Tech Lead | **q4 through C.2 C1/C2 recorded; task-7 (C.3) closed at C2 `aaf6fcfe`; (c) and (g) PASS; task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`)** |
| G-10 | Pre-CLOSEOUT scientific independence gate | Tester written independence confirmed on the dirty run and the counted run; `task-1/CLOSEOUT.md` states the bitwise agreement and the shared-kernel limitation | Tester / Reviewer | **closed** |

No requirements-level or cross-work-package design question remains open. G-02 records the
Architect's q4-only code-ready re-close. G-01 retains the q4 authorization as history and cites
ADR-F1A-010 for Step 4b on the remaining M-F1a inventory: each slice's handoff follows that
slice's Step 4a code-ready close with the four ADR-F1A-010 items unchanged. G-08 records q4, Slice A C2 `2a2f8c137`, Slice B C2 `0e8299e9`, C.0 at `91680ec7`, C.1 at `a006c7e2`, C.2 C1 `b95400d5` and C2 `42922382`, C.3 C2 `aaf6fcfe`, and C.4 C1 `55e37837` and C2 `85f01985`. Task-4 `CLOSEOUT.md` is present (ET-C0-4); stage `step-4b-authorized`; both lint modes were clean at the C.0 close (`91680ec7`); ADR-F1A-009 does not apply to C.0. task-5 (C.1) is closed at C2 `a006c7e2`. The C.1 finding was a warning at `step-4a`; `--strict` promoted it to the one error after the C1 stage flip, and (d) cleared it. Task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS; CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`). The task-6 `step-4a` review's one finding was a warning kept under `--strict`, and (d) cleared it. task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`). The task-8 `step-4a` review's one finding was a warning kept under `--strict`, and (d) cleared it. The task-7 `step-4a` review's one finding was a warning kept under `--strict`, and after the stage flip `--strict` promotes that finding until (d). O-11: `max_partition_qubits` is pinned at 2 for every M-F1a counted cell (Architect, C.0 Step 4a review, 2026-10-05); a change returns to Research Manager. RM-1(a): strict q6, q8, and q10 stay in the denominator and are realizable via a new M-F1a-only family designed in C.3 (not an edit to `workloads.py`); the RM-1(a) family is frozen in the C.3 Step 4a re-close (`/tmp/c3-step4a/C3_STEP4A_RECLOSE.md`); q4 stays the historical id, call only; q6, q8, and q10 are `mf1a_strict_spectator_embed_q6`, `…_q8`, and `…_q10`; C.1 fused and C.2 hybrid do not wait. Order is C.1 fused, C.2 hybrid, C.3 strict, C.4 baseline. G-09 records q4, Slice A, Slice B C1 `22071ecb` / C2 `0e8299e9`, C.0 at `91680ec7`, C.1 C1 `7cf11a49` / C2 `a006c7e2`, C.2 C1 `b95400d5` and C2 `42922382`, C.3 C2 `aaf6fcfe`, and C.4 C1 `55e37837` and C2 `85f01985`. The per-slice rule stays in force. G-10 is closed by the Tester confirmation recorded
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
| Slice tracer | task-1 shipped, C2 `a2928bf1` (baseline route verified); Slice A C2 `2a2f8c137`; Slice B C1 `22071ecb`, C2 `0e8299e9` (q4 baseline regenerated at clean C2); task-4 C.0 shipped at `91680ec7`; task-5 (C.1) closed at C2 `a006c7e2`; task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS; CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`) |
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
8. **(g)** Regenerate from clean C2 under ADR-F1A-009 Amendment 1. When the only case-field difference is `cases[0].provenance.implementation_revision`, the command exits 0, `status` is pass, and `regeneration.pass` is true. Tester reports the field-level diff, restores the generated outputs, and does not commit them. Task-1 step-8 option (i) text stays the historical record of that slice.

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
for that q4 closeout. Slice A closed at C2 `2a2f8c137`. Slice B closed at C1 `22071ecb` and C2 `0e8299e9`;
the claim is q4 baseline regenerated at clean C2. G-10 is closed. G-09 records those
commits and keeps the per-slice rule in force. Task-4 (C.0) is shipped at `91680ec7`.
Task-5 (C.1) is closed at C1 `7cf11a49` and C2 `a006c7e2`; (c) and (g) PASS. Claim: fused bundle generated at clean C1. Task-6 (C.2) C1 `b95400d5`; (c) PASS; (g) PASS; CLOSEOUT written at (d); closed at C2 `42922382`; task-7 (C.3) closed at C2 `aaf6fcfe`; C1 `a0ac4cbd`; (c) PASS; (g) PASS (77 s, exit 0, `/tmp/c3-g-proof/REPORT.md`); task-8 (C.4) closed at C2 `85f01985`; C1 `55e37837`; (c) PASS; (e) APPROVE `bc-a3270830`; (g) PASS (82.1 s, exit 0, at `85f01985`, Tester `bc-9820d3c5` run 18, `/tmp/c4-g-proof/REPORT.md`). Claim: hybrid bundle generated at clean C1. G-03 stays open. Order remains C.1, C.2, C.3, C.4.
Step 4b for a later slice is Tech Lead under ADR-F1A-010 only after that slice's
code-ready close. Task-9 is the counted-manifest slice, closed code-ready, with its freeze recorded in §10. It is not a fifth
route slice. G-03 stays open.

## 9. Carry-forward

Accepted deferrals. None of these is bundled with the counted-denominator freeze.

| Item | Disposition | Later commit |
|------|-------------|--------------|
| Hybrid `S_len_range` extra-row test | deferred | its own test commit |
| C.3 partial-S10 `[0, 0]` fixture and unpinned `_are_ints` | deferred | one strict-test commit |
| C.4 nits: unused `Path` import; G9 int and unknown-anchor unpinned; row-2 workloads text-match gap; `claim_boundary` unpinned; blank line | deferred | one baseline commit |
| Mutant-harness lesson: require pytest exit 1 and a collected count above 0 | deferred | its own skill commit |
| cmake learning | no action | already covered by `.cursor/rules/qgd-python-env.mdc` and `.cursor/skills/clean-rebuild/SKILL.md` |
| task-8 mini-spec §3.4 and the fitness row | recorded only | historical text; no edit |

## 10. Task-9 counted manifest

Planning-base HEAD `a81be56baba7e97156532524978cc878d347fa87`. Artifacts: `task-9/TASK_9_MINI_SPEC.md`, `task-9/DELIVERY_STORIES.md`, `task-9/ENGINEERING_TASKS.md` (`**SDD stage:** step-4b-authorized`), and `task-9/COUNTED_MANIFEST.md`. No `task-9/CLOSEOUT.md`. The task-9 `step-4a` review's one finding, `SLICE_MISSING_CLOSEOUT`, was a warning kept under `--strict`; after the stage flip `--strict` promotes it to the one error until (d). No waiver and no placeholder closeout.

Closed code-ready under ADR-F1A-008 on 2026-10-06. Research Manager accepted `claim_boundary` and `completeness_claim` (`task-9/COUNTED_MANIFEST.md` §§4–5) and recorded the freeze below, which is not stored in the manifest. Step 4b starts on Tech Lead's ADR-F1A-010 handoff. No counted run before the freeze. Per-cell wording is "`<route>` route verified at q`<n>`". The §9 deferrals stay out of task-9. G-03, G1, and G3 stay open.

**Research Manager freeze record, 2026-10-06, via Tech Lead.**

```text
Research Manager records that task-9/COUNTED_MANIFEST.md, sha256 a93de87d6b44a8a14d00d2eef73e116997faa635f35952fe6375e620aa363c16, 87 lines, exactly reflects the pinned delivered M3 and M3A advertised routes and the current-state support boundary (ADR-F1A-001): four routes at anchors 4, 6, 8, and 10, 16 cells, each with the workload, parameter source and count, and seed policy its slice pinned. No cell is dropped, merged, or substituted. The oracle, QA-001 tolerances, G-07, and O-11 are unchanged. ADR-F1A-010 item 3 for task-9: the counted set is this frozen manifest; Step 4b may start after the Architect code-ready close and this record. No counted run precedes this record.
```
