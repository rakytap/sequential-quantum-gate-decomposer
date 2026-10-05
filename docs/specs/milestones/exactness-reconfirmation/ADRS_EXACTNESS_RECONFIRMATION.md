# ADRs — M-F1a `exactness-reconfirmation`
> **Status:** Layer 1 v0.1 — accepted milestone decisions; planning only ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Scope:** decisions spanning more than one future work package ·
> **Upstream:** accepted `INITIAL_REQUIREMENTS.md` v0.3 ·
> **Traces:** REQ-001…008 · CAP-001, CAP-007 · QA-001, QA-005, QA-008, QA-009 ·
> **Program decisions:** existing ADR-001…008 remain in force

## ADR-F1A-001 — Freeze four advertised routes across four anchors

**Status:** accepted.

**Context.** REQ-001 needs an independently reviewed denominator before counted execution.
The pinned roadmap's delivered M3 row advertises partitioned execution with unitary-island
fusion; its delivered M3A row advertises strict/hybrid channel-native execution. The current
`ARCHITECTURE_OVERVIEW.md` identifies those paths and the sequential reference as shipped
runtime responsibilities. Runtime exports confirm the claimed routes but do not define
advertisement by themselves. Existing evidence does not count every advertised route at
every anchor.

**Decision.** The M-F1a advertised-route inventory is:

| Advertised route id | Public execution entry | Counted realization at 4/6/8/10 |
|---------------------|------------------------|---------------------------------|
| `partitioned_density_descriptor_baseline` | `execute_partitioned_density` | requested and realized baseline |
| `partitioned_density_descriptor_fused_unitary_islands` | `execute_partitioned_density_fused` | requested fused and at least one genuinely fused unitary-island region |
| `phase31_channel_native` | `execute_partitioned_density_channel_native` | strict request succeeds and genuinely executes an eligible channel-native motif |
| `phase31_channel_native_hybrid` | `execute_partitioned_density_channel_native_hybrid` | realized hybrid execution with complete per-partition attribution and at least one genuinely channel-native partition |

The sequential route is the oracle and is not a route under test. Strict eligibility is
about bounded motif support, not register width; strict-eligible workloads must therefore
genuinely execute on registers at each required anchor. A requested fused route that realizes
only baseline does not satisfy the fused cell. A labelled baseline or skip satisfies only its
own attribution and is not channel-native execution.

The VQE density energy entry and its energy-continuity rows are non-counted context because
they do not expose the full candidate density matrix required by QA-001. The legacy standalone
`NoiseChannel` API is not advertised support. Local depolarizing, amplitude-damping, and
phase-damping remain the delivered channels; no support surface widens.

**Rationale.** This is the literal accepted denominator: every shipped advertised execution
route at the established 4/6/8/10 anchors, with genuine realization rather than requested-name
accounting.

**Consequences.** The pre-run manifest must identify a genuinely eligible workload,
parameters, and seed for every route-anchor cell. It is frozen before counted execution and
validated as an exact set. Before the freeze, an independent Research Manager or architect
review records that the manifest exactly reflects the pinned M3/M3A claims and current-state
support boundary.

**Rejected alternatives.**
- Treat every exposed API as advertised support: contradicts the product glossary.
- Count strict only at historical microcase widths: narrows the accepted anchor contract.
- Count a baseline realization as fused or channel-native: violates genuine execution.
- Reuse the frozen 26-case matrix: retcons historical speedup evidence.

**Upstream alignment:** REQ-001, REQ-007 · CAP-001, CAP-007 · QA-001, QA-008 · goals G1, G3.

## ADR-F1A-002 — Isolate the exact M-F1a counted predicate

**Status:** accepted.

**Context.** Existing evidence predicates include older validity, energy, and external
reference gates. The accepted M-F1a baseline restored QA-001 exactly and forbids adding a
Hermiticity requirement or substituting a symmetrized matrix.

**Decision.** A counted M-F1a candidate output `rho` passes only when all four inequalities
hold against the sequential oracle:

- `||delta rho||_F <= 1e-10`;
- `max_ij |delta rho_ij| <= 1e-10`;
- `|Tr(rho) - 1| <= 1e-10`;
- `lambda_min(rho) >= -1e-12`;

and every state entry and every reported residual is finite. This predicate applies on 100%
of counted cases. `lambda_min` is evaluated on `rho` as produced; no symmetrization is
permitted. The frozen fitness interpretation uses the existing
`DensityMatrix.eigenvalues()` contract: LAPACK `zheev` reads the upper triangle of `rho` as
stored and returns real eigenvalues; the minimum returned value is `lambda_min(rho)`.
Finiteness is checked before this call, and a solver failure or non-finite eigenvalue fails
the counted row. This is an eigensolver convention, not a Hermiticity gate. No Hermiticity
gate is part of QA-001 or added by M-F1a. QA-010 is not cited. Each quantity is recorded as
a value and a pass/fail result so a failure is diagnosable.

The same semantic predicate governs the reusable fast fitness lane and the counted bundle.
Older `rho_is_valid`, energy, or external-reference fields may remain visible as context but
do not participate in the M-F1a count or status.

**Rationale.** A sibling predicate prevents accidental widening, weakening, or reinterpretation
of accepted QA-001 while preserving older packages unchanged.

**Consequences.** Historical predicates may coexist with the explicitly versioned M-F1a
predicate. Their statuses are never presented as an M-F1a verdict.

**Rejected alternatives.**
- Reuse `rho_is_valid`: it carries a different contract.
- Add Hermiticity or QA-010: outside accepted scope.
- Evaluate the minimum eigenvalue on `(rho + rho_dagger)/2`: expressly forbidden.
- Gate exactness on energy or Aer: not QA-001 and not M-F1a's claim.

**Upstream alignment:** REQ-002, REQ-006 · CAP-001 · QA-001 · goals G2, G3.

## ADR-F1A-003 — Define evaluation-mode attribution by labels plus witnesses

**Status:** accepted.

**Context.** Hybrid channel-native execution is the shipped evaluation mode. It already emits
per-partition runtime classes and reasons, while fused-region records can independently show
which route actually executed.

**Decision.** Evaluation mode means `phase31_channel_native_hybrid`. Every partition in every
hybrid execution, counted or uncounted, must:

1. carry one runtime-class label and one route-reason label from the frozen shipped
   vocabulary;
2. have label count equal partition count;
3. agree with independently observable fused-region records for that partition; and
4. preserve baseline or skip labels as non-channel-native outcomes.

The frozen runtime-class vocabulary is `phase31_channel_native`,
`phase3_unitary_island_fused`, and `phase3_supported_unfused`. The frozen route-reason
vocabulary is `eligible_channel_native_motif`, `pure_unitary_partition`,
`channel_native_qubit_span`, and `channel_native_support_surface`.

A `phase31_channel_native` label requires a matching `channel_native_motif` execution witness.
A `phase3_unitary_island_fused` label requires a matching genuinely fused unitary-island
witness. A `phase3_supported_unfused` label requires a positive, independently recorded
per-partition baseline/unfused execution witness, the absence of a channel-native or
`actually_fused` unitary-island witness for that partition, and its frozen non-eligible route
reason. Every counted hybrid route-anchor cell requires at least one witnessed
`phase31_channel_native` partition; an all-baseline/all-fused hybrid request does not
establish that route cell. Counted rows report the complete label distribution. Baseline,
fused, and strict entries retain case-level requested/realized path records but are not
evaluation mode and do not gain a new per-partition label contract.

**Rationale.** A label copied without an execution witness is self-attestation. Witness
agreement makes attribution falsifiable without changing route semantics.

**Consequences.** Missing, unknown, inconsistent, or silently substituted labels fail the
attribution gate. A labelled skip is observable but never promoted to channel-native.

**Rejected alternatives.**
- Require labels only on counted executions: weaker than QA-005.
- Infer route from requested path alone: cannot detect substitution.
- Add per-partition labels to non-evaluation routes: unnecessary support widening.

**Upstream alignment:** REQ-003 · CAP-001, CAP-007 · QA-005, QA-008 · goal G4.

## ADR-F1A-004 — Keep one command and require complete provenance

**Status:** accepted.

**Context.** REQ-004 names the existing correctness pipeline. Current bundles do not record
all provenance needed to reproduce the accepted denominator and may consume a pinned planner
selection artifact.

**Decision.** The single M-F1a regeneration entry remains:
`conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`.
It owns a separately versioned M-F1a sibling package under
`benchmarks/density_matrix/artifacts/correctness_evidence/` and exits nonzero for any missing
or failing requirement.

The package records the full revision and dirty-state declaration, `qgd` environment and
interpreter identity, exact command, route inventory, claim boundary, QA-001 values,
workload ids, parameter sources, seeds, and the path and content identity of every consumed
input artifact. Regeneration compares categorical fields exactly and numerical residuals
against their frozen tolerances.

A counted regeneration must begin from a clean tracked and untracked worktree at the recorded
revision. A dirty pre-run state fails validation or is explicitly non-counted. Generated
artifacts are outputs of that clean-start run and do not retroactively invalidate it.

Aer rows and energy-continuity rows may regenerate under the same command as explicitly
non-counted context. Their absence or status cannot add, remove, pass, or fail a counted row.
M-F1a does not re-close protocol (i)–(v) plus Aer.

**Rationale.** One entry point gives a reviewer a mechanical answer while sibling versioning
prevents historical schema mutation.

**Consequences.** Missing provenance, stale partial artifacts, changed route inventory, or a
nonzero counted failure makes the command fail. Optional Aer cannot block counted exactness.

**Rejected alternatives.**
- Add a second top-level command: conflicts with the accepted requirements.
- Continue implicit input consumption: not reproducible.
- Let Aer or energy continuity gate the count: crosses the claim boundary.

**Upstream alignment:** REQ-004, REQ-006, REQ-007 · CAP-007 · QA-008 · goal G5.

## ADR-F1A-005 — Freeze and report disagreement without retcon

**Status:** accepted.

**Context.** The sequential oracle is the semantic baseline, and every downstream claim
depends on shipped paths agreeing with it. Existing Phase-3 and Phase-3.1 evidence has separate
historical meaning.

**Decision.** Any counted QA-001 failure:

- marks the row and M-F1a bundle failed;
- reports route, anchor, case, and first failing quantity;
- marks M-F1a incomplete and freezes downstream claims; and
- preserves the oracle, tolerance table, manifest, and counted denominator.

M-F1a uses a sibling schema and artifact subtree. It does not edit older record schemas,
case sets, predicates, or artifacts. The frozen Phase-3.1 archive remains unchanged against
`1cb3d20c`; the 26-case matrix remains historical 17/9/0 speedup-only evidence and is never
relabelled as M-F1a correctness evidence.

Historical protection covers tracked and untracked changes under the frozen archive, the
entire `benchmarks/density_matrix/performance_evidence/` tree that defines the 26-case
inventory/schema/classification semantics, and
`benchmarks/density_matrix/planner_surface/workloads.py`, which supplies its transitive
family/noise/seed/qubit inventory, plus
`tests/partitioning/evidence/test_phase31_counted_matrix_validation.py`, all against
`1cb3d20c`. The historical-boundary record pins content identities for this complete input
set so a transitive inventory change cannot evade the static review. M-F1a does not execute
the historical performance/timing builder or use its verdict as evidence.

Kraus bundles remain primary. Choi and Liouville remain witnesses and cannot replace the
oracle or counted output.

**Rationale.** Failure must challenge the shipped path or oracle assumption, not trigger
evidence denominator changes.

**Consequences.** Resolution of a real disagreement requires separate authorization and
cannot be hidden inside M-F1a evidence regeneration.

**Rejected alternatives.**
- Loosen a threshold, drop a row, or narrow advertisement after execution: denominator retcon.
- Change oracle semantics in this milestone: violates the frozen baseline.
- Modify old package semantics in place: rewrites historical evidence.

**Upstream alignment:** REQ-006, REQ-007 · CAP-001, CAP-007 · QA-001, QA-008 · goals G1, G2, G5.

## ADR-F1A-006 — Use the upstream Linux test gate for non-interference

**Status:** accepted.

**Context.** QA-009 requires zero state-vector regressions and an unchanged default backend.
The Linux CI job runs the upstream suite, but workflow triggers do not run it on this feature
branch.

**Decision.** The M-F1a closure gate is the actual `.github/workflows/ci.yml`
`build-and-test-linux` job, invoked only through the existing `workflow_dispatch` trigger.
A pull request is not an M-F1a closure path and is not authorized. Before that closure gate,
its test command is reproduced locally in `qgd`:
`conda run -n qgd --no-capture-output pytest tests/ -x -v --tb=line --ignore=tests/decomposition/test_wide_circuit_optimization.py`.
Existing default-backend, explicit-state-vector, and density-opt-in checks are mandatory
parts of that gate. Changing CI branch triggers is outside M-F1a.

**Rationale.** The actual CI job proves the clean build/environment; local reproduction gives
preflight without broadening repository-level CI policy.

**Consequences.** The local run is preflight, not a substitute for CI. Any upstream failure
in either run blocks milestone closure. No density-only subset may stand in for this gate.

**Rejected alternatives.**
- Run only density-marked tests: cannot prove state-vector non-interference.
- Change CI triggers as part of M-F1a: unrelated workflow-policy expansion.
- Invoke the job through a pull request: no pull request is authorized.

**Upstream alignment:** REQ-005 · QA-009 · goal G6.

## ADR-F1A-007 — Update current-state docs only from delivered evidence

**Status:** accepted.

**Context.** Both current-state documents exist and describe the delivered Phase 1–3.1 stack.
M-F1a will add an evidence contract and runnable lane but Layer 1 is not shipped behavior.

**Decision.** Planning does not edit `ARCHITECTURE_OVERVIEW.md` or `TECH_STACK.md`. At
milestone close, after all executable gates pass, both documents are updated—not recreated—to
name the final advertised-route boundary, QA-001 predicate, attribution contract, exactness
fitness lane, regeneration command, artifact root, and state-vector gate. Any stale lane
description encountered in that delivered surface is corrected then.

Release is additive: a separately versioned evidence package and fitness checks. Rollback
removes only the new package registration and its new checks/artifacts; runtime behavior,
public APIs, older bundles, and current channel support remain unchanged. No
`CHANGE_CONTROL.md` is needed unless a future proposal deviates from these frozen contracts.

**Rationale.** Current-state docs describe truth, not intended future state.

**Consequences.** REQ-008 cannot pass before executable evidence. Layer 1 records the update
obligation without prematurely rewriting current state.

**Rejected alternatives.**
- Update current-state docs during planning: would describe unshipped behavior.
- Create replacement docs: violates REQ-008.

**Upstream alignment:** REQ-008 · CAP-007 · QA-008, QA-009 · goal G7.

## ADR-F1A-008 — Slice closeout belongs to slice close, not code-ready

**Status:** accepted (Research Manager, 2026-10-05). Milestone-local to M-F1a. It changes no frozen contract, QA-001, the q4 slice boundary, the G-07 exit contract, or the lint script.

**Correction.** This supersedes the 2026-10-04 Architect ruling that SLICE_MISSING_CLOSEOUT blocks code-ready, because that ruling made code-ready unreachable. These parts stand: no CLOSEOUT.md before delivery, no waiver, and no invented delivery content.

**Context.** The skill's semantic gates do not require CLOSEOUT. They are "Slice code-ready" (SKILL.md:200-208), Definition of Ready (references/practices-testing.md:66-76), and Layer 4 code-ready (references/templates-layer-2-4.md:123-128). CLOSEOUT has only two legitimate statuses, shipped or implementation handback, both written in Step 4b (references/templates-closeout.md:14-41; SKILL.md:113-132). The same template says it is not planning (references/templates-closeout.md:1-4). The finding text names "a shipped or handed-back slice" (scripts/check_artifacts.py:211-218), yet it fires on any task-n directory, and --strict promotes it to an error (scripts/_sddlint.py:120-149). The Verify section (SKILL.md:183-196) runs both checks before any code-ready claim and forbids waivers on an in-flight milestone. Read literally, that is circular for any first slice.

**Decision.**
1. Code-ready gate for a slice requires all of these: the semantic code-ready criteria are met; normal specs_check.sh has 0 errors; its only warnings are SLICE_MISSING_CLOSEOUT for slices that have not reached Step 4b; traceability is clean; the slice's planning gaps are closed; and the Architect review is closed as code-ready. Under --strict, that same finding is the one known planning-stage result. It stays visible and recorded in the checklist. It is not waived, and it is not fixed with a placeholder file.
2. Slice-close gate: after Step 4b ships or hands back, write the real CLOSEOUT.md. Then --strict must be fully clean, and Reviewer checks happen before any local commit.
   **Superseded for Step 4b / slice-close commit ordering by ADR-F1A-009.**
3. Any other strict finding still blocks code-ready. The exception covers exactly one finding code, and only before Step 4b.

**Rationale.** A closeout is an evidence record. Writing one before delivery would assert a verdict that no executed test supports. That is the same failure as counting a requested-but-unrealized route as executed, which ADR-F1A-001 forbids.

**Rejected alternatives.**
- Placeholder or planning-status CLOSEOUT: no such status exists, and it would be false evidence.
- Waive in docs/specs/.sdd-lint.json: forbidden on an in-flight milestone.
- Make the script stage-aware: framework change outside M-F1a. Logged as a framework follow-up for after M-F1a only.
- Keep strict-clean as a code-ready gate: no slice could ever start.

**Consequences.** G-08 is closed as a gate definition. G-01 alone controls Step 4b. Code-ready is not Step 4b, not implementation, and not a Developer or Tester handoff.

**Upstream alignment:** REQ-004, REQ-007 · QA-008 · G-08.

## ADR-F1A-009 — Two-commit slice close for clean-start evidence

**Status:** accepted (Research Manager, 2026-10-05). Milestone-local to M-F1a.

**Upstream:** REQ-004, REQ-007 · QA-008 · ADR-F1A-004, ADR-F1A-008.

**Context.** ADR-F1A-004 and `task-1/TASK_1_MINI_SPEC.md` §3.3 require the counted run to start clean at a recorded implementation revision distinct from the planning base. ADR-F1A-008 decision 2 forbids any local commit before Reviewer after `CLOSEOUT.md`. Together, no counted pass can exist.

**Decision.** For the q4 baseline cell only — `phase2_xxz_hea_q4_continuity` on `partitioned_density_descriptor_baseline` — replace only ADR-F1A-008 decision 2 with this sequence:

1. **(a) Reviewer implementation review:** Reviewer examines the uncommitted implementation diff.
2. **(b) Local implementation commit C1:** after Reviewer passes the implementation review, create local C1 containing planning docs, implementation, and tests, with no generated artifact and no `CLOSEOUT.md`. An optional planning-docs C0 may precede C1.
3. **(c) Counted clean-start run:** from empty `git status --porcelain` at C1, run the single `benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` command once and retain its counted evidence.
4. **(d) Real slice close:** write the real `task-1/CLOSEOUT.md` citing C1, then require normal and `--strict` spec checks fully clean and traceability clean.
5. **(e) Reviewer evidence review:** Reviewer examines the counted evidence, real closeout, and clean verification results.
6. **(f) Local evidence commit C2:** after Reviewer passes the evidence review, create local C2 containing the evidence and closeout.
7. **(g) Clean-C2 regeneration:** from clean C2, rerun regeneration; restore generated outputs afterward and do not commit the regeneration outputs.

Before step (d), Tester must satisfy the checklist's pre-CLOSEOUT scientific gate: confirm in writing that the oracle and the cell are independent, name the code paths and objects used by each side, show that the oracle is not the cell's own output read back, and explain the reason for bitwise agreement. If they are not independent, stop before `CLOSEOUT.md` and do not create C2. The real closeout records the bitwise agreement and its reason.

**Unchanged.** Clean-start semantics and detection remain whole-worktree `git status --porcelain` with no output-path exclusion. `SLICE_MISSING_CLOSEOUT` remains visible until step (d), with no waiver. QA-001, G-07's exit set, ADR-F1A-004, and ADR-F1A-008 decision 1 remain unchanged. No push or pull request is part of this sequence.

**Rationale.** C1 supplies the clean recorded implementation revision required by provenance without inventing a closeout or committing generated evidence. C2 follows the real execution evidence and Reviewer evidence review. The final clean-C2 regeneration checks reproducibility without adding regenerated outputs to history.

**Rejected alternatives.**
- Exclude generated output paths from `clean_start`.
- Commit `CLOSEOUT.md` before the counted run.
- Count the dirty run.

**Consequences.** Research Manager authorized this order and this ADR on 2026-10-05 for the q4 baseline cell only. C1 still waits on the Reviewer pre-C1 implementation pass; writing this ADR does not authorize C1 by itself. The checklist owns the pre-CLOSEOUT scientific gate and records completion before step (d).

**Upstream alignment:** REQ-004, REQ-007 · QA-008 · ADR-F1A-004, ADR-F1A-008 · G-09, G-10.

## Continuation ADRs (2026-10-05)

Amendments and ADRs accepted after the q4 tracer close are final text in
`ADR_AMENDMENTS_EXACTNESS_RECONFIRMATION.md`:

- ADR-F1A-008 Amendment 1 (stage-aware closeout check)
- ADR-F1A-010 (Step 4b authorization for the remaining M-F1a slices)
- ADR-F1A-009 Amendment 1 (per-slice two-commit close)
- ADR-F1A-011 (historical suites verify-only under the M-F1a command)
