# Initial requirements — M-F1a `exactness-reconfirmation`
> **Milestone:** M-F1a `exactness-reconfirmation` · **Status:** draft v0.3 ·
> **Owner skill:** `create-initreq-for-sdd` ·
> **Upstream:** `PRODUCT_STATEMENT.md` at `ceb469c8` — CAP-001, CAP-007 ·
> QA-001, QA-005, QA-008, QA-009; `ROADMAP.md` at `1cb3d20c` — M-F1a ·
> **Authorized scope:** requirements baseline only; no Layer 1 planning or implementation ·
> **Downstream:** `spec-driven-development` after stakeholder validation

## 1. Milestone scope and vision

- **Milestone:** M-F1a `exactness-reconfirmation`, the product walking skeleton for the
  current spec convention.
- **Upstream traceability:** CAP-001 exact noisy evolution and CAP-007 regenerable
  verification; QA-001 exactness, the route-attribution and no-silent-substitution parts of
  QA-005, QA-008 reproducibility, and QA-009 non-interference.
- **Outcome:** on the revision under test, 100% of counted cases for every shipped advertised
  route meet QA-001 against the sequential `NoisyCircuit` oracle at the 4-, 6-, 8-, and
  10-qubit anchors; every executed evaluation-mode partition is route-labelled; the counted
  bundle regenerates; and the state-vector path has zero regressions.
- **In scope:** exactness, physical-state checks, route attribution, regeneration, a reusable
  exactness fitness lane, and updates to the existing current-state documentation.
- **Out of scope:** new channels; timing; performance or cost selection; optimizer changes;
  AVX; GPU; VQA work; gradients; any speedup or at-least-1.2x claim; opening M-F5a, M-F1b,
  M-F2, M-F-CE, M-F3, M-F4, M4 `canonical-attributed-energy`, Q3, or E7.
- **Target users:** noisy-algorithm researchers who need trustworthy route-attributed
  results; SQUANDER maintainers who protect the additive state-vector path; and external
  reproducers who need a single named regeneration lane.
- **Core problem:** later claims depend on the un-reconfirmed assumption that all currently
  advertised routes still agree with the sequential oracle and preserve physical states on
  the current revision.
- **Success metrics:** all outcome clauses above pass together; zero counted failures, zero
  unlabelled executed evaluation-mode partitions, zero silent route substitutions, and zero
  state-vector regressions. A failure is reported, never converted into success by changing
  the oracle, tolerance, advertised-route inventory, or counted cases after execution.
- **Current-state context:** the sequential executor remains the internal semantic oracle;
  Qiskit Aer remains an optional external reference, not the oracle. Exact execution is CPU
  only on dense complex128 states. All product lanes run in the `qgd` environment.
  `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` already exist and are updated, not recreated.

## 2. User journeys

- **Primary journey:** a researcher checks out the pinned revision, runs the named M-F1a lane
  in `qgd`, and receives a validated bundle. Every counted row identifies its advertised
  route, qubit anchor, workload and seed, reports full-state and physical-state residuals,
  and passes the frozen QA-001 thresholds. Evaluation-mode rows also expose every
  partition's route label.
- **Reproduction journey:** an external reproducer runs the same command from a clean
  checkout and reproduces every categorical result exactly and every numerical result
  within its frozen tolerance.
- **Disagreement path:** if a shipped advertised route disagrees with the oracle or fails a
  physical-state check, the bundle fails, identifies the first failing route/case/measure,
  and freezes downstream claims. The oracle, tolerance, and counted case set remain
  unchanged.
- **Attribution failure path:** if an evaluation-mode partition lacks a route label, claims a
  route inconsistent with its execution record, or silently takes another route, validation
  fails.
- **Non-interference path:** if the state-vector suite regresses or the default backend
  changes, the milestone does not close.

## 3. Ubiquitous language

This milestone inherits the product glossary and uses these narrower terms:

| Term | Meaning in M-F1a |
|------|------------------|
| **Sequential oracle** | The step-by-step `NoisyCircuit` execution used as the internal exact semantic baseline; its semantics are not changed by this milestone. |
| **Shipped advertised route** | A currently shipped execution path named as supported in a claim-bearing support surface at the pinned revision; API exposure alone is not advertisement. The sequential oracle is the comparator, not a non-sequential route under test. |
| **Advertised-route inventory** | The independently reviewed, revision-pinned enumeration of all shipped advertised routes and their support boundaries, frozen before counted regeneration and validated as an exact set rather than against itself. |
| **Counted case** | A claim-bearing tuple of advertised route, 4/6/8/10-qubit anchor, workload, parameters, and seed in the M-F1a bundle. |
| **Physical-state check** | The QA-001 trace and minimum-eigenvalue checks, plus finite full-state output, evaluated independently of route-to-oracle agreement. |
| **Evaluation mode** | A mode in which each partition may execute through one of the already shipped named routes and must expose the route actually taken. It does not imply cost-based selection. |
| **Route label** | Per-partition attribution drawn from a frozen vocabulary and consistent with the execution record. A baseline or skip label is not channel-native execution. |
| **Regeneration** | Re-running the named lane at its pinned revision and reproducing categorical results exactly and numerical residuals within frozen tolerances. |
| **Historical 26-case matrix** | The frozen Phase-3.1 speedup-only evidence at 17/9/0. M-F1a neither edits, relabels, recounts, nor adopts it as its correctness matrix. |
| **Downstream freeze** | No later milestone or claim proceeds after a counted disagreement until the disagreement is resolved without changing the oracle or historical matrix. |

## 4. Requirements and acceptance criteria

### REQ-001 — Complete, frozen advertised-route coverage
- **Upstream:** M-F1a · CAP-001, CAP-007 · QA-001, QA-008.
- **Intent:** establish the denominator before execution so that 100% cannot be achieved by
  omitting a shipped advertised route or anchor.
- **Acceptance:** Given the pinned revision and an independently reviewed pre-run
  route-by-anchor manifest derived from its claim-bearing support surfaces, when the M-F1a
  suite is registered before counted regeneration, then the manifest and executable suite
  form the same exact set: every shipped advertised route genuinely executes at 4, 6, 8, and
  10 qubits, with workloads, parameters, seeds, and support boundaries fixed. A baseline or
  skip execution counts only for its own advertised route.
- **Negative/error:** Given a shipped advertised route or required anchor missing from the
  manifest, a requested route that did not genuinely execute, or a manifest changed after
  counted execution begins, when the bundle validates, then validation fails and no
  completeness claim is emitted.
- **Evidence route:** correctness evidence pipeline exact-set validator plus pre-run manifest
  review.

### REQ-002 — Exact and physical output on every counted case
- **Upstream:** M-F1a · CAP-001 · QA-001.
- **Acceptance:** Given any counted case, when its advertised route and the sequential oracle
  execute the identical ordered operations and parameters, then
  the candidate output `rho` satisfies `||delta rho||_F <= 1e-10`,
  `max_ij |delta rho_ij| <= 1e-10`, `|Tr(rho) - 1| <= 1e-10`,
  and `lambda_min(rho) >= -1e-12`, and every state entry and reported residual is finite.
  All counted cases pass.
- **Negative/error:** Given any non-finite value or threshold violation, when the bundle
  validates, then that row and the bundle fail with the route, anchor, case, and first failing
  measure identified; the row is not dropped and the tolerance is not loosened.
- **Evidence route:** correctness evidence pipeline for counted rows; fast pytest exactness
  fitness tests against the sequential oracle.

### REQ-003 — Complete and truthful evaluation-mode route attribution
- **Upstream:** M-F1a · CAP-001, CAP-007 · QA-005, QA-008.
- **Acceptance:** Given any evaluation-mode execution on the shipped surface, counted or
  uncounted, when it completes, then 100% of executed partitions carry a route label from the
  frozen vocabulary, label count equals executed partition count, each label agrees with an
  independently observable actual-route witness, and counted bundle rows report the per-case
  label distribution.
- **Negative/error:** Given an unlabelled partition, an unknown label, a label inconsistent
  with execution, or a silent route substitution, when validation runs, then the case and
  bundle fail. A labelled baseline/skip remains visible and is never reported as
  channel-native execution.
- **Evidence route:** fast pytest route-attribution/support-boundary tests and correctness
  evidence pipeline route records.

### REQ-004 — Revision-pinned bundle regeneration
- **Upstream:** M-F1a · CAP-007 · QA-008.
- **Acceptance:** Given a clean checkout at the recorded revision in the documented `qgd`
  environment, when
  `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py`
  runs, then it regenerates and validates the complete counted bundle under
  `benchmarks/density_matrix/artifacts/correctness_evidence/`, exits nonzero on any missing
  or failing requirement, reproduces categorical results exactly, and keeps numerical
  residuals within frozen tolerances on 100% of rows. The bundle records revision,
  environment, command, route inventory, claim boundary, tolerances, workloads, parameters,
  and seeds.
- **Negative/error:** Given a missing provenance field, stale or partial artifact, unpinned
  input, or row outside the registered inventory, when validation runs, then the bundle fails
  or is explicitly non-counted and cannot support the milestone outcome.
- **Evidence route:** the named correctness evidence pipeline command and its bundle
  validators, run in `qgd`.

### REQ-005 — State-vector non-interference
- **Upstream:** M-F1a · QA-009.
- **Acceptance:** Given the M-F1a change set, when the documented upstream CI test lane runs,
  then it reports zero state-vector regressions, the default backend remains state-vector,
  and density execution remains opt-in.
- **Negative/error:** Given a changed default backend or any state-vector test regression,
  when the gate runs, then the gate fails and M-F1a does not close.
- **Evidence route:** CI's upstream `pytest tests/` lane, including pinned default-backend and
  explicit-state-vector checks.

### REQ-006 — Immutable oracle and failure freeze
- **Upstream:** M-F1a · CAP-001, CAP-007 · QA-001, QA-008.
- **Acceptance:** If any shipped advertised route disagrees with the sequential oracle, then
  evidence generation reports the disagreement and marks the milestone incomplete; all
  dependent claims remain frozen until a separately authorized resolution preserves the
  frozen oracle contract, tolerances, and registered cases.
- **Negative/error:** Given an attempted recovery that changes oracle semantics, loosens a
  tolerance, suppresses a failing row, or changes the registered denominator after execution,
  when validation or closeout review runs, then the recovery is rejected and the failure
  remains visible.
- **Evidence route:** benchmark validator tests plus the M-F1a evidence matrix and closeout
  review.

### REQ-007 — Historical evidence remains historical
- **Upstream:** M-F1a · CAP-007 · QA-008.
- **Acceptance:** Given the frozen Phase-3.1 26-case matrix, when M-F1a evidence is produced,
  then `docs/density_matrix_project/archive/phases/phase-3-1/` remains unchanged against
  commit `1cb3d20c` and separately identified; M-F1a correctness rows are not merged into it,
  relabelled as it, or used to reinterpret its 17/9/0 result.
- **Negative/error:** Given an M-F1a artifact that edits, recounts, relabels, or adopts a
  frozen-matrix row as a new counted correctness row, when review runs, then validation fails.
- **Evidence route:** repository diff against `1cb3d20c`, benchmark validators, and closeout
  review.

### REQ-008 — Current-state documentation matches the delivered lane
- **Upstream:** M-F1a · CAP-007 · QA-008, QA-009.
- **Acceptance:** Given all executable M-F1a gates pass, when the milestone closes, then the
  existing `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` are updated—not recreated—to name
  the shipped advertised-route boundary, exactness fitness lane, regeneration command,
  QA-001 thresholds, route-attribution contract, and state-vector gate without stale paths.
- **Negative/error:** Given either document cites a missing command/path, omits a delivered
  boundary, or describes behavior inconsistent with the validated bundle, when strict spec
  checks and closeout review run, then M-F1a does not close.
- **Evidence route:** `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh
  --strict` plus current-state-doc review.

## 5. Non-functional requirements

- **Exactness and physical validity — QA-001:** on 100% of counted rows,
  `||delta rho||_F <= 1e-10`, `max_ij |delta rho_ij| <= 1e-10`,
  `|Tr(rho) - 1| <= 1e-10`, and `lambda_min(rho) >= -1e-12`; every state entry and reported
  residual is finite. Hermiticity is not part of QA-001 and is not added by this milestone.
- **Attribution and strictness — QA-005:** 100% of executed evaluation-mode partitions have
  truthful labels; zero silent route substitutions. This milestone does not widen QA-005's
  input or parameter surface.
- **Reproducibility and auditability — QA-008:** 100% of counted rows regenerate with exact
  categorical results and numerical residuals within frozen tolerance; every row has the
  provenance fields in REQ-004.
- **Compatibility and non-interference — QA-009:** zero state-vector regressions; unchanged
  state-vector default; density remains opt-in; no new runtime dependency.
- **Performance / scale:** no timing, throughput, cost, speedup, or at-least-1.2x response
  measure is admitted. The evidence regime is the four anchors from 4 through 10 qubits.
- **Security / privacy / accessibility / i18n:** no new surface or data obligation is created
  by this evidence-only milestone.
- **Architecture and tech-stack documentation:** both existing current-state documents are
  updated only after executable evidence passes.

## 6. Operational boundaries

- **Always do:** run product lanes in `qgd`; freeze the route inventory, cases, oracle, and
  tolerances before counted regeneration; compare full density matrices; report physical
  checks independently; expose every evaluation-mode partition's actual route; run the
  state-vector gate; update both current-state documents at close; run both spec checks.
- **Ask first:** any change to a frozen tolerance, oracle semantics, advertised-route
  boundary, counted-case denominator, public API, runtime dependency, or historical artifact.
- **Never do:** alter the sequential oracle or frozen 26-case matrix to obtain a pass; omit,
  skip, or relabel a failure; silently substitute a route; weaken or delete a test; add a
  channel; perform timing, cost-selection, optimizer, AVX, GPU, gradient, or VQA work; make a
  speedup or at-least-1.2x claim; open or plan another milestone; extend the frozen archive.

## 7. Assumptions and open questions

### Assumptions
- The stakeholder lifted the roadmap's stale hold for M-F1a requirements only. This
  authorization does not open Layer 1, implementation, or any later milestone.
- “At 4–10 qubits” uses the repository's established 4-, 6-, 8-, and 10-qubit correctness
  anchors. It does not claim uncounted intermediate widths.
- The claim-bearing support surfaces at the pinned implementation revision determine the
  complete advertised-route inventory. Layer 1 may record that inventory but may not narrow it
  to avoid a failing route.
- External-reference and energy continuity rows may regenerate as non-counted context, but
  M-F1a claims only the CAP/QA traces in its roadmap row.
- M-F1a uses QA-005 only for route attribution and zero silent route substitution. Its
  broader out-of-domain parameter behavior is not widened by this milestone.

### Open questions
- None at the requirements level. The 2026-10-04 architect instruction confirms exactness,
  physical-state checks, attribution, regeneration, non-interference, the downstream freeze,
  and the scope exclusions. Layer 1 choices may not narrow those contracts.

### Critique
- The largest risk is denominator gaming; REQ-001 freezes every advertised route and anchor
  before execution.
- Oracle agreement alone can hide a shared semantic defect; REQ-002's independent
  physical-state measures catch nonphysical shared failures but do not establish external
  semantics. External-reference continuity remains available but non-counted.
- A route label can overstate channel-native use; REQ-003 requires consistency with the
  execution record and preserves baseline/skip labels.
- A failing route could tempt tolerance or case-set changes; REQ-006 makes such recovery an
  explicit rejection and freezes downstream claims.
- The historical speedup matrix invites accidental retcon and performance scope creep;
  REQ-007 and the operational boundaries keep it frozen and separate.

### Handoff
The create-initreq stakeholder checkpoint is complete for this baseline. A separate
authorization is still required before `spec-driven-development` ingests it into M-F1a
Layer 1. The q4 partitioned route is the roadmap's slice tracer. No other milestone is opened.

### Change log
- **v0.3 (2026-10-04):** restored the exact QA-001 predicate by removing the added
  Hermiticity measure and evaluating `lambda_min` on `rho`.
- **v0.2 (2026-10-04):** recorded the lifted M-F1a requirements hold, closed the requirements
  checkpoint against the architect's constraints, and left Layer 1 and implementation closed.
- **v0.1 (2026-10-04):** initial M-F1a requirements baseline from product statement
  `ceb469c8`, roadmap `1cb3d20c`, and the stakeholder's exactness-only authorization.
