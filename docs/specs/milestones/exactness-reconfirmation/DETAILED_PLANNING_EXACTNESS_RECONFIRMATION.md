# Detailed planning — M-F1a `exactness-reconfirmation`
> **Status:** Layer 1 v0.1 — planning complete; not an implementation handoff ·
> **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Owner skill:** `spec-driven-development` Steps 1–3 ·
> **Upstream:** accepted `INITIAL_REQUIREMENTS.md` v0.3, `PRODUCT_STATEMENT.md` at
> `ceb469c8`, and `ROADMAP.md` at `1cb3d20c` ·
> **Traces:** REQ-001…008 · CAP-001, CAP-007 · QA-001, QA-005, QA-008, QA-009 ·
> **Authorization:** Research Manager, 2026-10-04, Layer 1 only ·
> **Decisions:** `ADRS_EXACTNESS_RECONFIRMATION.md` · **Verdict:** see readiness checklist

## 1. Purpose and authority

M-F1a reconfirms, on a pinned revision, that every shipped advertised density-matrix
execution route agrees with the sequential `NoisyCircuit` oracle under QA-001 at the
4-, 6-, 8-, and 10-qubit anchors. It also makes every evaluation-mode partition's route
observable, regenerates one claim-bearing bundle, and proves that the state-vector path
has zero regressions.

Authority descends in this order:
`PRODUCT_STATEMENT.md` → `ROADMAP.md` M-F1a → accepted `INITIAL_REQUIREMENTS.md` v0.3 →
this plan and its ADRs → future slice artifacts. Layer 1 may clarify but may not weaken
the accepted requirements. No slice or implementation is authorized by these files.

## 2. Scope and non-goals

### In scope
- Exact full-density comparison against `execute_sequential_density_reference`.
- The QA-001 trace and minimum-eigenvalue checks plus finite output and residuals.
- A frozen shipped-advertised-route inventory and counted coverage at every anchor.
- Complete, truthful route attribution for all evaluation-mode partitions.
- Revision-pinned regeneration from the named correctness command.
- A reusable exactness fitness lane and the state-vector non-interference gate.
- Updates, at milestone close only, to the existing `ARCHITECTURE_OVERVIEW.md` and
  `TECH_STACK.md`.

### Out of scope
- New channels; local depolarizing, amplitude-damping, and phase-damping remain the
  delivered channel set.
- Timing, throughput, cost selection, optimizer changes, AVX, GPU, VQA, gradients, or any
  at-least-1.2x claim.
- M-F5a or any later milestone, including M-F3, M-F4, M4, Q3, and E7.
- QA-010, a Hermiticity gate, or any replacement of `lambda_min(rho)` by a symmetrized
  matrix.
- Re-closing the older E1 package or describing M-F1a as protocol (i)–(v) plus Aer.
- Editing, relabelling, recounting, or adopting the frozen 26-case matrix.
- Treating energy-continuity rows as an opening for M4 canonical-attributed-energy.

### Representation boundary
Kraus bundles remain the primary fused channel object. Choi and Liouville forms remain
witnesses only. M-F1a changes neither this representation policy nor runtime semantics.

## 3. Success conditions and stop rule

M-F1a succeeds only when all of the following hold together:

1. The counted manifest and executed suite contain exactly every shipped advertised route
   at each anchor in `{4, 6, 8, 10}`.
2. On 100% of counted cases, the candidate output `rho` satisfies
   `||delta rho||_F <= 1e-10`, `max_ij |delta rho_ij| <= 1e-10`,
   `|Tr(rho) - 1| <= 1e-10`, and `lambda_min(rho) >= -1e-12`, and every state entry and
   reported residual is finite.
3. Every executed evaluation-mode partition is route-labelled, every label agrees with an
   independently observable execution witness, and a labelled skip or baseline is not
   channel-native execution. Every counted hybrid route-anchor cell genuinely executes at
   least one channel-native partition.
4. The named command regenerates the complete bundle with exact categorical results and
   numerical residuals within frozen tolerance.
5. The state-vector gate has zero regressions, the default backend remains state-vector,
   and density execution remains opt-in.
6. Both current-state documents describe the delivered lane and boundary without stale
   commands or paths.

If any shipped advertised route disagrees with the sequential oracle or fails the accepted
predicate, evidence generation stops with a failed bundle, names the first failing
route/case/measure, and freezes downstream claims. The oracle, tolerances, route inventory,
and counted denominator are not changed to obtain a pass.

## 4. Frozen contracts

| Contract | Frozen Layer 1 value | Trace |
|----------|----------------------|-------|
| Oracle | `execute_sequential_density_reference`; runtime id `sequential_density_descriptor_reference`; comparator, not a route under test | REQ-002, REQ-006 |
| QA-001 | The four inequalities and finiteness statement in success condition 2, on `rho` as produced, with no symmetrization and no added Hermiticity gate | REQ-002; QA-001 |
| Anchors | Exactly 4, 6, 8, and 10 qubits; intermediate widths are not claimed | REQ-001 |
| Advertised routes | Baseline partitioned, fused unitary-island, strict channel-native, and hybrid channel-native, derived from the pinned delivered M3/M3A roadmap outcomes and confirmed by current-state architecture; detailed in ADR-F1A-001 | REQ-001 |
| Evaluation mode | Hybrid channel-native only; every partition receives a truthful label and witness check | REQ-003; QA-005 |
| Regeneration command | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | REQ-004 |
| Artifact root | `benchmarks/density_matrix/artifacts/correctness_evidence/`, with an M-F1a sibling package that does not rewrite older schemas | REQ-004, REQ-007 |
| State-vector gate | Actual `.github/workflows/ci.yml` `build-and-test-linux` job at closure; identical local `qgd` command is preflight only | REQ-005; QA-009 |
| Non-counted context | Aer external-reference rows and energy-continuity rows may regenerate and report, but do not gate M-F1a count or status | REQ-002, REQ-004 |
| Historical boundary | Frozen Phase-3.1 archive unchanged against `1cb3d20c`; frozen 26-case matrix remains historical speedup-only evidence | REQ-007 |

## 5. Current-state findings

- The pinned roadmap's delivered M3 outcome advertises partitioned execution with
  unitary-island fusion, and M3A advertises strict/hybrid channel-native execution. The
  current architecture identifies baseline partitioned, fused, strict, hybrid, and
  sequential-reference runtime responsibilities. Runtime exports confirm those claims and
  public results record requested and realized paths; exports alone do not define the
  denominator.
- Hybrid execution already records per-partition runtime class and route reason; fused-region
  records provide an independent execution witness. Other routes are not evaluation mode.
- The main correctness pipeline currently exercises the fused request, while the separate
  Phase-3.1 pipeline has strict microcases and hybrid continuity only at 4 and 6 qubits.
  It does not establish the accepted route-by-anchor denominator.
- The existing shared pass predicate includes `rho_is_valid`, energy continuity, and optional
  Aer gating. Those older package semantics remain untouched; they are not the M-F1a
  predicate.
- Existing bundle software metadata does not yet provide the full revision, command,
  environment, inventory, claim boundary, tolerance, workload, parameter, seed, and pinned
  input provenance required by REQ-004.
- CI's Linux job contains the state-vector gate but does not trigger on this feature branch;
  the identical command is reproduced locally in `qgd` as preflight, and the actual CI job
  remains mandatory at closure.
- `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` exist. Both require an update at close to
  document the final inventory, fitness lane, command, QA-001 predicate, attribution contract,
  and state-vector gate. Planning does not edit either current-state document.

## 6. Architecture boundary map

| Boundary | Owns | Inbound port | Outbound evidence or dependency | Decision |
|----------|------|--------------|--------------------------------|----------|
| Mixed-state core | `DensityMatrix`, `NoisyCircuit`, delivered dep/AD/PD semantics | Python bindings | candidate and oracle density matrices | unchanged |
| Noisy planning | canonical operation surface and partition descriptors | planner surface builders | descriptors to runtime | unchanged |
| Partitioned runtime | baseline, fused, strict, hybrid, sequential execution records | public `execute_*` entries | density output, requested/realized route, partitions, positive per-partition execution witnesses | ADR-F1A-001, ADR-F1A-003 |
| Correctness evidence | manifest, counted records, validators, provenance, bundle status | named validation pipeline | artifacts under the correctness root | ADR-F1A-002, ADR-F1A-004, ADR-F1A-005 |
| External-reference adapter | optional Aer comparison and continuity energy | existing evidence adapters | explicitly non-counted rows | ADR-F1A-002 |
| State-vector path | default backend and upstream behavior | CI test lane | non-interference verdict | ADR-F1A-006 |
| Current-state docs | shipped architecture and runnable lanes | milestone closeout | reader-facing current truth | ADR-F1A-007 |

The dependency direction remains evidence → runtime → planner → bindings → C++ core.
M-F1a introduces no reverse dependency and no new service, storage, API, or deployment
boundary. Its anti-corruption boundary is the sibling evidence schema: historical record
semantics are consumed as context, never mutated into the M-F1a claim.

## 7. Milestone goals

| Goal | Outcome, not implementation | Acceptance evidence | Requirements |
|------|-----------------------------|---------------------|--------------|
| G1 — freeze the denominator | A reviewed, revision-pinned route-by-anchor manifest exactly equals the executed counted suite; every route genuinely executes at every anchor | manifest review and exact-set pipeline validator | REQ-001, REQ-007 |
| G2 — establish the QA-001 fitness function | One reusable predicate reports the four accepted inequalities and finiteness per counted output without importing older gates | fast pytest and per-row pipeline values | REQ-002, REQ-006 |
| G3 — complete route-by-anchor evidence | Baseline, fused, strict, and hybrid routes each have counted passing evidence at 4/6/8/10 | M-F1a bundle coverage matrix | REQ-001, REQ-002 |
| G4 — prove route attribution | Every hybrid partition has a valid label consistent with a positive execution witness, including baseline/unfused; baseline/skip remains visibly non-channel-native | fast negative tests and bundle label distributions | REQ-003 |
| G5 — make regeneration auditable | One command emits a complete revision-pinned bundle, fails on any missing contract, and reproduces under the declared policy | correctness pipeline and validator lane | REQ-004, REQ-006, REQ-007 |
| G6 — protect state-vector behavior | Upstream state-vector tests and backend-selection pins pass with zero regressions | actual Linux CI job at closure; local preflight | REQ-005 |
| G7 — publish current operational truth | Existing architecture and stack documents name the delivered boundary, lane, command, predicate, attribution, and gate | strict spec check and closeout doc review | REQ-008 |

The roadmap's q4 partitioned route remains the future slice tracer. No Layer 2–4 artifact or
slice decomposition is created in Layer 1.

## 8. Traceability

| Requirement | Layer 1 interpretation | Goals | ADRs |
|-------------|------------------------|-------|------|
| REQ-001 | Four advertised routes × four anchors, genuinely executed and exact-set validated | G1, G3 | ADR-F1A-001 |
| REQ-002 | Exact QA-001 predicate only; values and failures reported per row | G2, G3 | ADR-F1A-002 |
| REQ-003 | Hybrid is evaluation mode; labels are complete, vocabulary-bound, and witnessed | G4 | ADR-F1A-003 |
| REQ-004 | Existing named command owns the M-F1a sibling package and full provenance | G5 | ADR-F1A-004 |
| REQ-005 | Actual Linux CI job proves closure; its command is reproduced locally in `qgd` only as preflight; defaults remain pinned | G6 | ADR-F1A-006 |
| REQ-006 | Any counted disagreement fails and freezes downstream claims without retcon | G2, G5 | ADR-F1A-002, ADR-F1A-005 |
| REQ-007 | Older packages and frozen matrix remain separately identified and unchanged | G1, G5 | ADR-F1A-005 |
| REQ-008 | Both current-state docs update only after executable gates pass | G7 | ADR-F1A-007 |

## 9. Milestone evidence matrix

| Trace id | Evidence type | Command or gate | Lane | Expected result |
|----------|---------------|-----------------|------|-----------------|
| REQ-001, REQ-002, REQ-004, QA-001, QA-008 | counted bundle | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py` | correctness evidence pipeline | exit 0; exact route × anchor set present; every counted row passes |
| REQ-002, REQ-006, QA-001 | fitness checks | `conda run -n qgd --no-capture-output pytest tests/partitioning -m "density_matrix and not slow"` | fast pytest | accepted QA-001 predicate passes against the sequential oracle |
| REQ-003, QA-005 | attribution checks | `conda run -n qgd --no-capture-output pytest tests/partitioning/test_partitioned_channel_native_phase31_hybrid_slice.py -m "not slow"` | fast pytest | all hybrid partitions labelled and witness-consistent; each counted hybrid cell has genuine channel-native execution |
| REQ-005, QA-009 | closure CI gate | `.github/workflows/ci.yml` job `build-and-test-linux` via the existing `workflow_dispatch` trigger only; local preflight uses `conda run -n qgd --no-capture-output pytest tests/ -x -v --tb=line --ignore=tests/decomposition/test_wide_circuit_optimization.py` | CI plus local pytest | actual CI job and local preflight pass; state-vector default and density opt-in checks pass |
| REQ-007 | static historical boundary review | `git diff --exit-code 1cb3d20c -- docs/density_matrix_project/archive/phases/phase-3-1 benchmarks/density_matrix/performance_evidence benchmarks/density_matrix/planner_surface/workloads.py tests/partitioning/evidence/test_phase31_counted_matrix_validation.py` plus the same paths under `git status --porcelain --untracked-files=all --` | non-executable repository review | both outputs empty and pinned content identities match; frozen archive and complete transitive 26-case inventory/schema/classification assets unchanged; no timing builder executes |
| REQ-008 | spec fitness and doc review | `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict` | spec lint / doc review | zero errors or warnings; current-state descriptions match evidence |
| Boundary only | external context | `conda run -n qgd --no-capture-output python benchmarks/density_matrix/validate_squander_vs_qiskit.py` | optional Qiskit Aer reference | non-counted context only; does not determine M-F1a status |

## 10. Risks and decision gates

- **Oracle assumption fails:** stop, report the exact disagreement, and freeze downstream
  claims. No alternate oracle or denominator is selected inside M-F1a.
- **Denominator gaming:** the manifest is reviewed and frozen before counted execution, and
  exact-set validation rejects omissions or substitutions.
- **Requested route is not realized:** a route cell counts only when its realized execution
  satisfies ADR-F1A-001; a baseline realization does not count as fused or channel-native.
- **Predicate drift:** the sibling schema owns M-F1a semantics; older package predicates and
  the 26-case matrix are unchanged and separately labelled.
- **Optional Aer unavailable:** counted regeneration remains valid because Aer is non-counted.
- **Branch CI does not run automatically:** local `qgd` execution is preflight only; the
  actual `build-and-test-linux` CI job remains a closure gate through the existing
  `workflow_dispatch` trigger only. Trigger-policy changes are out of scope.
- **Planning turns into delivery:** the readiness verdict remains not-ready until separate
  authorization opens Step 4a. No Developer or Tester handoff exists in this Layer 1 pass.

## 11. Expected outcome

The eventual milestone output is a revision-pinned sibling correctness package, regenerated
by the existing named command, that covers every advertised route at every anchor under the
accepted QA-001 predicate, records complete hybrid attribution and provenance, and carries a
zero-regression state-vector verdict. It updates current-state docs at close and makes no
performance, external-protocol, energy, or later-milestone claim.
