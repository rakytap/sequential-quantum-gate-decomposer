# Delivery stories — M-F1a slice 1: q4 baseline partitioned tracer
> **Status:** planning review closed code-ready by Architect on 2026-10-05 under
> ADR-F1A-008 · **Slice:** M-F1a slice 1 ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Scope:** q4 baseline partitioned continuity route against the sequential oracle ·
> **Contract:** `TASK_1_MINI_SPEC.md` ·
> **Traces:** REQ-001, REQ-002, REQ-004, REQ-005, REQ-006, REQ-007 ·
> QA-001, QA-008, QA-009 ·
> **Authorization:** Research Manager G-01 and ADR-F1A-009 authorize Step 4b for the q4
> baseline cell only; C1 is `a50ae79f`; C2 awaits Reviewer, then Tech Lead · **No push/PR**

## DS-1 — A researcher receives the exact q4 baseline verdict

**Stakeholder / system value**
- Establishes that the simplest shipped non-sequential route agrees with the sequential
  semantic oracle under the accepted physical-state predicate before broader coverage.

**Given / When / Then**
- Given the reviewed `phase2_xxz_hea_q4_continuity` tracer cell, shipped default partition
  width, deterministic parameters, and identical descriptor input for route and oracle
- When `execute_partitioned_density` and `execute_sequential_density_reference` run and the
  candidate `rho` is evaluated
- Then the requested and realized paths are baseline, the execution is genuinely
  partitioned and unfused, every QA-001 inequality passes, all state entries and residuals
  are finite, and each value and pass flag is reported.
- Given any path-realization, finiteness, eigensolver, or QA-001 failure
- When the record is built
- Then the record and tracer bundle fail, identify route/anchor/workload and the first
  failing measure, and stop without changing the oracle, predicate, cell, or parameters.

**Scope**
- In: q4 baseline route, sequential oracle, exact QA-001 predicate, fail-and-report.
- Out: fused, strict, hybrid, q6/q8/q10, Hermiticity, QA-010, Aer or energy gating.

**Acceptance signals**
- The q4 baseline acceptance test passes using real runtime and oracle outputs.
- Negative predicate tests pin finite-entry ordering, eigensolver failure, each threshold,
  and absence of symmetrization or a Hermiticity gate.

**Traceability**
- Initial requirements: REQ-002, REQ-006
- Capability / quality attribute: CAP-001 · QA-001
- Milestone planning: success condition 2 and stop rule; goals G2, G3
- ADRs: ADR-F1A-002, ADR-F1A-005

## DS-2 — A reviewer sees a truthful tracer boundary

**Stakeholder / system value**
- Prevents one green q4 row from being presented as a complete M-F1a route-by-anchor result.

**Given / When / Then**
- Given the reviewed slice manifest
- When the q4 baseline record validates
- Then exactly that tracer cell is present, its route realization and planner setting match
  the contract, `milestone_counted` and `completeness_claim` are false, and the claim boundary
  states that no milestone, external protocol, energy, or frozen-matrix claim is made.
- Given an extra route/anchor/workload, a missing tracer record, or a realized fused path
- When the tracer bundle validates
- Then validation fails and identifies the boundary violation.

**Scope**
- In: one-cell manifest, provisional count semantics, exact-set validation, claim boundary.
- Out: the remaining milestone manifest and any other route or anchor.

**Acceptance signals**
- Positive and negative manifest tests pass in the correctness-evidence validator lane.
- Aer and energy fields cannot change `qa001_pass`, tracer status, or count status.

**Traceability**
- Initial requirements: REQ-001, REQ-004
- Capability / quality attribute: CAP-001, CAP-007 · QA-008
- Milestone planning: goals G1, G5
- ADRs: ADR-F1A-001, ADR-F1A-004

## DS-3 — A reproducer regenerates the tracer from the named command

**Stakeholder / system value**
- Makes the first M-F1a evidence object auditable before the broader matrix is planned.

**Given / When / Then**
- Given a clean checkout at the recorded revision in `qgd`
- When the named correctness pipeline command runs
- Then provenance is captured before writes, the separately versioned q4 tracer bundle is
  emitted under the correctness artifact root, all categorical fields are pinned, numerical
  residuals satisfy regeneration policy, and the process-exit aggregate adds the sibling
  result while excluding exactly `correctness_evidence_external_correctness` and the whole
  `correctness_evidence_output_integrity` suite.
- Given the sibling bundle is present with `status = pass` and every other included
  registered suite passes
- When the named command completes
- Then it exits 0; the two excluded suites retain their own statuses, and Aer or energy
  failure cannot change the sibling status or M-F1a acceptance exit.
- Given the sibling is missing or fails provenance, QA-001, or regeneration, or any other
  included suite fails
- When the named command completes
- Then it exits 1 and identifies the failing included result.
- Given a dirty or incompletely described pre-run
- When regeneration runs
- Then the tracer remains non-counted, its provenance gate fails, the tracer bundle fails,
  and it cannot support a milestone claim.
- Given a prior tracer bundle with the same schema/manifest version
- When regeneration compares it
- Then categorical drift or an out-of-tolerance residual fails the tracer with the first
  mismatch identified.

**Scope**
- In: provenance, sibling schema, single-command registration, regeneration comparison.
- Out: changing older package schemas or making older Aer/energy results counted.

**Acceptance signals**
- The pipeline writes a passing clean tracer bundle with complete required provenance.
- Process-exit tests pin sibling presence/pass, the exact two exclusions, retention of every
  other registered suite, exit 1 on sibling/included failure, and unchanged excluded statuses.
- Validator tests reject missing provenance, categorical drift, and excessive residual drift.

**Traceability**
- Initial requirements: REQ-004, REQ-006, REQ-007
- Capability / quality attribute: CAP-007 · QA-008
- Milestone planning: success condition 4; goal G5
- ADRs: ADR-F1A-004, ADR-F1A-005

## DS-4 — A maintainer sees upstream and historical behavior untouched

**Stakeholder / system value**
- Preserves the additive density-track contract while the new evidence lane is introduced.

**Given / When / Then**
- Given the slice change set
- When the upstream suite runs locally in `qgd`
- Then state-vector behavior, default-backend selection, and density opt-in checks have zero
  regressions.
- Given the protected historical paths
- When tracked and untracked state is reviewed against `1cb3d20c`
- Then the archive, performance evidence, transitive workload inventory, and counted-matrix
  test are unchanged and no timing builder executes.
- Given eventual milestone closure
- When CI evidence is requested
- Then only a successful `build-and-test-linux` run invoked through the existing
  `workflow_dispatch` trigger counts as closure evidence; no pull request is used or
  authorized.

**Scope**
- In: local preflight, workflow-dispatch-only closure boundary, static historical review.
- Out: CI trigger changes, pull requests, current-state doc updates, performance execution.

**Acceptance signals**
- Local upstream preflight passes.
- Both historical-review commands have empty output as this slice's partial contribution to
  REQ-007; the durable content-identity record remains a milestone-level obligation.
- No protected path, public runtime interface, or CI configuration changes.

**Traceability**
- Initial requirements: REQ-005, REQ-007
- Capability / quality attribute: CAP-007 · QA-008, QA-009
- Milestone planning: success condition 5; goals G5, G6
- ADRs: ADR-F1A-005, ADR-F1A-006

## Deferred requirement boundary

| Requirement | Reason not served here | Future ownership |
|-------------|------------------------|------------------|
| REQ-003 | Baseline execution is not evaluation mode; no hybrid route is planned in this slice | a separately authorized hybrid slice |
| REQ-008 | Current-state docs update only after delivered evidence and milestone closure | milestone close |

No deferred slice is decomposed or authorized by this table.
