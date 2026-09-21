# Roadmap — SQUANDER density-matrix track

> **Status:** draft v0.5 (scientific correction of v0.4; Phase 4 bounded), at stakeholder checkpoint ·
> **Owner skill:** `create-product-roadmap` ·
> **Last revalidated:** 2026-09-21 — source-verified scientific correction pass (U3 coordinates,
> optimizer support, cost accounting, GAD language); no milestone under the new convention has
> closed yet ·
> **Upstream:** [`PRODUCT_STATEMENT.md`](PRODUCT_STATEMENT.md) v0.3
> (`CAP-001…007`, `QA-001…012`) ·
> **Downstream:** `milestones/<slug>/INITIAL_REQUIREMENTS.md` via
> `create-initreq-for-sdd` ·
> **Sequencing inputs:** archived
> [`PLANNING.md`](../density_matrix_project/archive/planning/PLANNING.md) §3, §6, §8;
> [`RESEARCH_ALIGNMENT.md`](../density_matrix_project/RESEARCH_ALIGNMENT.md);
> [`PHASE4_REQS.md`](PHASE4_REQS.md) ·
> **Not:** requirements, architecture or ADR rationale, dates, or publication planning.

## 1. Summary & horizon

The [product vision](PRODUCT_STATEMENT.md#1-vision) is an exact open-system simulation module
inside a circuit compiler/optimizer that treats noise channels as first-class planner inputs,
never trades exactness, and proves every advertised path under a frozen, regenerable protocol.
The destination is CAP-006 (noisy variational research on the exact backend). The current
direction (Phase 4) ends at **research readiness** and is bounded to attributed exact energies
(M4), a comparable-tier interop posture (M6), a fixed-parameter GAD anchor (M7), a canonical
energy-only readiness surface with explicit refusals (M8), and an \(R=1\)
representation-aware cost decision with a reuse *forecast* (M9). M5 is the roadmap's
scientific-risk screen for A1 and ships evidence only. Gradient product surfaces, a second
task/ansatz family, reuse activation, global strictness, cross-milestone reproduction, and
trainability studies are Later or gated extensions.

| Horizon | Milestones |
|---------|------------|
| Delivered (archived, frozen) | M1 `phase-1` · M2 `phase-2` · M3 `phase-3` · M3A `phase-3-1` |
| **Now** | **M4 `canonical-attributed-energy` — product walking skeleton** → M5 `canonical-sensitivity-screen` |
| Next | M6 `interop-overhead-profile` → M7 `gad-channel-admission` → M8 `study-ready-noisy-workflow` → M9 `representation-aware-selection` |
| Later (hypotheses) | M8A `second-family-workflow` · M10 `strict-parameter-domains` · M11 `clean-checkout-reproduction` · M12 `observed-reuse-amortization` · M13 `noisy-trainability-studies` |
| Conditional | M5A `exact-regime-boundary` — opens only through review gate RG-1 |
| Review gates (not milestones) | RG-1 exact-regime boundary · RG-2 differentiable objective · RG-3 selection inside the optimizer loop |

One milestone tree is in flight at a time. QA-008 is a completion criterion for every
milestone: 100% of that milestone's counted claims must regenerate from named lanes with
tolerances, seeds, revision, and claim boundary pinned. Every exactness claim compares the full
density matrix with the sequential `NoisyCircuit` reference; Qiskit Aer is the external
reference. `QA-005 (scoped)` means strictness is claimed only on that milestone's advertised
support matrix. M10 is the first milestone allowed to claim QA-005 globally. Every code-bearing
milestone carries QA-009 into its requirements and evidence, even when non-interference is not
its primary outcome.

## 2. Strategic themes

| Theme | Serves | `PHASE4_REQS.md` thread | Milestones |
|-------|--------|-------------------------|------------|
| **Trusted exact paths and evidence** — classify every advertised entry, calibrate fixed-parameter GAD under the per-step \((\gamma,p_\mathrm{exc})\) convention against closed forms and Aer, and make claims regenerable | CAP-001, CAP-002, CAP-007 · QA-001/002/003/005/008/010 | Channel Expansion; Verification and Calibration | M4, M7, M10, M11 |
| **Inspectable noise-aware planning and cost** — attribute every route, record complete sequential/candidate build and application costs at \(R=1\), forecast but never credit break-even reuse \(R^*\), activate reuse only at observed \(R_\mathrm{obs}\), and profile interop overhead | CAP-003, CAP-004 · QA-004/006/007/012 | Partitioning and Fusion; Interop Optimization; Literature Positioning as a claim gate | M6, M9, M12 |
| **Noisy variational research on the exact backend** — screen one paired finite-size estimand early, deliver a canonical energy-only readiness surface, extend to a second family, and reach pre-registered studies; gradient product surfaces wait behind RG-2 | CAP-005, CAP-006 · QA-002/005/011 | Readiness for noisy VQA training loops | M5, M5A, M8, M8A, M13 |

## 3. Milestone table

M4 is the product walking skeleton for the new convention. M1–M3A were delivered under the
archived phase convention and are recorded rather than re-planned; no
`milestones/<slug>/` tree is created for them.

| M# | slug | Outcome (measurable) | Horizon | Traces (CAP-*/QA-*) | Depends on | What ships (deployable) | Status |
|----|------|----------------------|---------|---------------------|------------|-------------------------|--------|
| M1 | `phase-1` | Exact mixed-state evolution and ordered noisy-circuit execution with the initial three local channels reproduce the external reference | Delivered | CAP-001, CAP-002 · QA-002, QA-009 | — | C++ core, Python bindings, tests, and Aer comparisons — [`archive/phases/phase-1/`](../density_matrix_project/archive/phases/phase-1/) | Delivered |
| M2 | `phase-2` | The density backend evaluates exact noisy energy in the frozen XXZ/HEA VQE workflow at 4/6/8/10 qubits with machine-checked support boundaries | Delivered | CAP-005, CAP-007 · QA-002, QA-005, QA-008, QA-009 | M1 | Backend selection, exact energy path, bridge metadata, and workflow evidence — [`archive/phases/phase-2/`](../density_matrix_project/archive/phases/phase-2/) | Delivered |
| M3 | `phase-3` | Noisy circuits are first-class planner inputs and execute through a partitioned runtime with exact unitary-island fusion; 34 cases were counted, 0/6 representative cases passed the positive threshold, and 6/6 closed through diagnosis | Delivered | CAP-001, CAP-003, CAP-007 · QA-001, QA-005, QA-008 | M2 | Noisy planner, descriptors, runtime, and correctness/performance evidence — [`archive/phases/phase-3/`](../density_matrix_project/archive/phases/phase-3/) | Delivered |
| M3A | `phase-3-1` | Exact strict/hybrid channel-native fusion closed as a bounded decision study: 17/26 `phase3_sufficient`, 9/26 `phase31_not_justified_yet`, 0/26 `phase31_justified` | Delivered | CAP-001, CAP-003, CAP-007 · QA-001, QA-002, QA-004, QA-005, QA-008 | M3 | Bounded channel-native modes and the 26-row decision bundle — [`archive/phases/phase-3-1/`](../density_matrix_project/archive/phases/phase-3-1/) | Delivered |
| **M4** | **`canonical-attributed-energy`** | **Product walking skeleton.** From one public VQE instance, a researcher obtains exact noisy energy through bridge → planner → attributed runtime → exact core → observable; every route is labelled, every q4/6/8/10 result meets QA-001 on the full state, and energy agrees with the public optimizer path and Aer | Now | CAP-001, CAP-003, CAP-005, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009 | M2 and M3A outcomes | Public attributed-energy entry, route records, entry-point support matrix and negatives, counted state/energy rows, M4 closeout, updated current-state docs | Now — active |
| M5 | `canonical-sensitivity-screen` | A pre-registered paired screen of one primary estimand—mean squared gradient per active stored coordinate \(G=\mathbb{E}_\theta[\lVert\nabla_\theta E\rVert_2^2/K]\)—contrasts fixed local noise with \(p=0\) on common \(\theta\) draws at \(n\in\{4,6,8,10\}\), returning `resolvable-on-screen`, `finite-size-limited`, or `inconclusive` from simultaneous paired intervals; gradients are benchmark-internal, coordinate-aware, and control-validated | Now | CAP-006, CAP-007 · QA-008, QA-009, QA-011 | M4 | Pre-registration, validated gradient instrumentation (not a product surface), sensitivity evidence, machine-readable verdict and diagnosis | Now — queued after M4 |
| M5A | `exact-regime-boundary` | If the M5-triggered review approves it, the practical exact-regime boundary is characterized under a frozen memory/time protocol while QA-001 remains satisfied | Conditional | CAP-001 · QA-001, QA-008 | M5 limiting verdict or exact-size diagnosis, plus review approval | Boundary evidence and escalation record; no approximate implementation | Conditional hypothesis |
| M6 | `interop-overhead-profile` | Public-entry language-boundary overhead is measured at 4/6/8 qubits under QA-007; the confidence bound either meets the frozen bar or a component-level diagnosis resolves A4 | Next | CAP-004, CAP-007 · QA-007, QA-008, QA-009 | M4; M5 verdict recorded | Comparable-tier entries, paired/interleaved harness, profile artifact, and overhead fitness function | Next |
| M7 | `gad-channel-admission` | Fixed-parameter generalized amplitude damping under the per-step \((\gamma,p_\mathrm{exc})\) convention is admitted on its declared core/planner entries; every counted state meets QA-001/002/010 and every malformed class fails before mutation; Gibbs-population wording requires a declared two-level Hamiltonian and basis, with no physical-time claim | Next | CAP-001, CAP-002, CAP-007 · QA-001, QA-002, QA-003, QA-005 (scoped), QA-008, QA-009, QA-010 | M4 support matrix | GAD operation/binding/API, planner and hybrid runtime support, strict-mode refusal, calibration and external evidence, published parameter convention | Next |
| M8 | `study-ready-noisy-workflow` | The canonical XXZ/generated-HEA workflow runs fixed local unital and non-unital noise, including GAD, through one public exact energy evaluator at 4–10 qubits; one bounded supported derivative-free optimizer smoke completes; every gradient and unsupported mode is refused before mutation; all claims regenerate from a clean checkout | Next | CAP-002, CAP-005, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009 | M4; M7; M5 verdict and M6 posture recorded | Canonical energy-only readiness surface, GAD noise specification, support/refusal matrix with negatives, optimizer-smoke evidence, workflow evidence, clean-checkout bundle | Next |
| M9 | `representation-aware-selection` | On the frozen Phase 3.1 families plus a frozen static-subgraph candidate surface, every partition carries decomposed sequential/candidate build and application costs, Choi rank, attributed memory, paired timing, and prediction error; selection satisfies QA-006 at \(R=1\) with no reuse credit and emits \(R^*\) as forecast only; either QA-006 and the competitor gate win or diagnosis kills A2 | Next | CAP-003, CAP-007 · QA-001, QA-004, QA-005 (scoped), QA-006, QA-008, QA-009 | M3A; M6; M8 closed | Frozen cost-policy ADR, \(R=1\) selection mode, cost records with an \(R^*\) forecast, competitor matrix, positive or diagnosis evidence | Next |
| M8A | `second-family-workflow` | One pre-registered, independently motivated additional task/ansatz family lowers through the bridge, evaluates exact energy under its advertised fixed local noise at 4–10 qubits, meets QA-001/002, inherits the M8 refusal discipline, and regenerates from a clean checkout | Later | CAP-002, CAP-005, CAP-006, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009 | M8; M7 if its noise uses GAD | Second-family bridge, workload-ordered gates/noise with negatives, workflow evidence, clean-checkout bundle | Later hypothesis — readiness extension |
| M10 | `strict-parameter-domains` | Every advertised out-of-domain parametric rate is rejected before mutation or uses an explicit caller-selected transform; no silent clamps remain and QA-005 is globally claimable | Later | CAP-001 · QA-003, QA-005, QA-008, QA-009 | M4 support matrix | Domain-semantics ADR, migrated advertised paths, and pinned negatives | Later hypothesis |
| M11 | `clean-checkout-reproduction` | An external reproducer regenerates 100% of closed claim-bearing milestones among M4, M5, M5A, M6, M7, M8, M9, M8A, and M10 from a clean checkout | Later | CAP-007 · QA-008 | Every listed milestone that is closed | Regeneration index, runbook, and cross-milestone bundle | Later hypothesis |
| M12 | `observed-reuse-amortization` | If M9 shows material candidate construction and independent optimizer traces yield \(R_\mathrm{obs}\) beyond forecast \(R^*\) with margin, reuse-conditioned caching is activated and meets QA-001/012; otherwise A3 is dropped and caching stays off | Later | CAP-003 · QA-001, QA-008, QA-009, QA-012 | M9 cost record and forecast; M8/M8A traces; independent of the \(R=1\) verdict | Parameter-dependent cache keys, hit/miss and build/apply accounting, amortization evidence | Later gated hypothesis |
| M13 | `noisy-trainability-studies` | Pre-registered hypotheses are tested across at least two families, unital/non-unital sweeps, and seeded ensembles with effect sizes, uncertainty, null outcomes, and a regenerable dataset; gradient-based training arms require RG-2 | Later | CAP-006, CAP-007 · QA-008, QA-009, QA-011 | M5 verdict; M8A; M11; RG-2 for gradient-based arms | Study pipelines, reusable dataset, and analysis regeneration lane | Later hypothesis — scientific endpoint |

### Review gates

Gates are decisions, not milestones; they carry no traces of their own and open nothing by
themselves.

| Gate | Trigger | Required before any milestone opens | Owner |
|------|---------|-------------------------------------|-------|
| **RG-1 Exact-regime boundary** | M5 `finite-size-limited`, or an `inconclusive` diagnosis that names the exact-size ceiling | Product-statement / ADR-008 review approval; M5A is post-Phase-4 by default, and inserting it earlier requires explicit `PHASE4_REQS.md` plus roadmap revalidation | Product owner via `create-product-statement` |
| **RG-2 Differentiable objective** | Any gradient product surface, `(parameters) → (energy, gradient)` bridge, or gradient-based optimizer trace | Evolve `PRODUCT_STATEMENT.md` with a differentiable-objective `CAP-*` and gradient-fidelity `QA-*`; define coordinate-aware rules, fixed-noise/order boundaries, unsupported/shared-parameter refusal, independent controls, and tolerance; explicitly evolve `PHASE4_REQS.md` or place the work after Phase 4; then add a traced roadmap milestone | Product owner |
| **RG-3 Selection inside the optimizer loop** | M9 ships the positive deployable | M9 revalidation; M12 \(R_\mathrm{obs}\) evidence for any reuse claim | Roadmap revalidation |

Until RG-2 passes, gradients are benchmark-internal, control-validated instrumentation in
evidence pipelines (M5 and, if needed, M13); the product surface refuses gradient modes.

## 4. Per-milestone detail

### M4 — `canonical-attributed-energy` (Now · product walking skeleton)

**Outcome and why now.** The public density-backend optimizer evaluates energy in C++, while the
Python planner/runtime currently executes the same instance's `describe_density_bridge()` only
to a state and route summary. M4 closes those delivered paths at the state and observable levels
without adding a channel or making a performance claim. It is the thinnest positive path through
the variational boundary, planner, runtime, exact core, and evidence system.

**Success measure.**
- For pre-registered parameter vectors at q4, q6, q8, and q10, every advertised attributed
  route—partitioned, unitary-island fused, and hybrid channel-native—meets all QA-001 full-state
  bounds against the sequential reference. The q4 case is the slice tracer.
- Every route's energy agrees with the public `Optimization_Problem` energy within the frozen
  derived tolerance, and at least one energy row per advertised width agrees with Aer per
  QA-002. Every executed partition carries a route label.
- The following support matrix is published, and every rejected class has a structured
  pre-mutation negative test:

| Entry | M4 classification |
|-------|-------------------|
| Direct C++ and Python-bound sequential `NoisyCircuit` | Supported; sequential path is the internal oracle |
| Planner → partitioned runtime | Supported and counted on the canonical surface |
| Planner → unitary-island fused runtime | Supported and counted on the canonical surface |
| Hybrid channel-native runtime | Supported and counted; every partition route-labelled |
| Strict channel-native whole-workload request | Frozen ≤2-qubit motif slice only; canonical whole workload is preflight-rejected |
| Public density-backend VQE energy | Supported on the delivered canonical workflow |
| New attributed energy over planner/runtime | Supported on the counted canonical anchors |
| Legacy standalone `NoiseChannel` API | Exposed but non-claim-bearing; not extended |
| Parametric noise-rate entries that clamp | Exposed but non-claim-bearing until M10 |
| Other gates, noise names, sources, or modes | Preflight-rejected |

**Scope.** In: attributed-energy entry, route records, support matrix, full-state and energy
rows, Aer rows, and current-state docs. Out: new channels, timing or cost selection, clamp
migration, legacy API changes, and optimizer iterations through the Python runtime.

**Riskiest assumption.** The delivered C++ optimizer-energy and Python planner/runtime paths
close on the same exact state and observable across every attributed route, and the new
convention can ship that integration in a few slices.

**Architecture boundary.** The C++ optimizer loop cannot call the Python planner/runtime.
Routing optimizer iterations through selection would require a Python optimizer surface or a C++
planner port, neither justified before M9. M4 therefore walks the product at one public energy
evaluation, not at whole-optimizer-loop granularity.

**Dependencies and handoff.** M2 and M3A outcomes. Handoff slug:
`canonical-attributed-energy`. Update
[`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) with the attributed-energy flow and
support boundary, and [`TECH_STACK.md`](TECH_STACK.md) with the evidence lane.

### M5 — `canonical-sensitivity-screen` (Now · queued)

**Outcome and why now.** A1—whether the exact regime resolves the intended effect rather than
only exhibiting a finite-size example—kills the most scope if wrong. Fitting asymptotic
barren-plateau exponents to four widths is ill-conditioned, so M5 asks one identifiable paired
finite-size question and freezes every decision before data.

**Primary estimand.** Mean squared gradient per active stored coordinate,
\[
G(n,p,L)=\mathbb{E}_{\theta\sim\mu}\!
\left[\lVert\nabla_\theta E(\theta;n,p,L)\rVert_2^2/K\right],
\]
with \(K\) the number of *active* coordinates. The primary contrast compares one fixed local
noise point \(p_1\) with \(p=0\) by the paired difference
\[
\Delta G_n=\mathbb{E}_{\theta\sim\mu_n}\!
\left[\frac{\lVert\nabla E(\theta;n,p_1,L_n)\rVert_2^2-
\lVert\nabla E(\theta;n,0,L_n)\rVert_2^2}{K_n}\right]
\]
at each \(n\in\{4,6,8,10\}\), using common \(\theta\) draws. The depth rule \(L_n\),
noise point, and active-coordinate rule are frozen before data.

**Frozen before data.** Benchmark relevance; distribution \(\mu\) over stored coordinates
(periods follow U3 multipliers); sampling unit = one exact-backend \(\theta\) draw at one width;
per-width sample size from non-counted pilot variance, minimum effect/equivalence bound, and
precision target; simultaneous or multiplicity-controlled paired intervals across all four
widths; verdict boundaries. Widths are design points, not statistical replicates; no exponent
is fitted or extrapolated. Secondary noise/depth points are descriptive unless separately
controlled.

**Verdict (exactly one).**
- `resolvable-on-screen`: at least one multiplicity-controlled interval excludes zero and its
  effect bound clears that width's minimum effect;
- `finite-size-limited`: all four intervals meet the precision target and lie inside their
  pre-registered equivalence bands; label the screened effect finite-size only and open RG-1;
- `inconclusive`: neither rule holds, including any underpowered width; permit one
  pre-registered extension in \(\theta\) draws. M5A stays closed unless diagnosis identifies
  the exact-size ceiling and RG-1 approves.

**Gradient instrumentation (benchmark-internal, not a product surface).** For stored U3
coordinate \(p_j\), derive the two-term rule from the declared multiplier \(m_j\):
\[
\partial_jE=\frac{m_j}{2}
\left[E\!\left(p+\frac{\pi}{2m_j}e_j\right)-
E\!\left(p-\frac{\pi}{2m_j}e_j\right)\right],
\qquad m_j\in\{2,1,1\}.
\]
This is valid only for fixed parameter-independent noise and unchanged operation order. Every
counted gradient must pass a central-finite-difference step study and independent analytic
witnesses covering all three coordinates, including non-zero \(\phi/\lambda\) witnesses.
Deferred to M13 unless rigorously defined there: minimum migration
\(\Delta\theta^*\) (Euclidean parameter distance is gauge- and period-dependent), the
gradient-norm coefficient of variation (previously misnamed SNR), and optimizer-conditioned
optimal depth \(L^*\).

**Scope.** In: pre-registration, gradient-validation tracer, non-counted pilot, sensitivity
lane, raw data, verdict, diagnosis. Out: asymptotic scaling, second family, product gradient
routing, optimizer run, new channels/gates, and performance claims.

**Feasibility.** Python may orchestrate shifted calls to the delivered public energy evaluator,
but "no C++ change" is not a correctness argument: the coordinate rule and controls must pass
before any counted row. The pilot freezes
\(\sum_n 4S_nK_n\) shifted energy evaluations for the paired \(p_1\)/\(p=0\) primary
width set, plus a wall-clock budget. If unaffordable,
shrink descriptive noise/depth points before data; never remove a primary width or reduce
\(S_n\) below its precision/power rule without roadmap and requirements revalidation.

**Riskiest assumption.** A1, only at the frozen design points and estimand.

**Dependencies and handoff.** M4. Handoff slug: `canonical-sensitivity-screen`.
[`TECH_STACK.md`](TECH_STACK.md) gains the named sensitivity lane; no architecture change is
expected.

### M6 — `interop-overhead-profile` (Next)

**Outcome and why next.** Phase 3 identified Python-level overhead but did not compare equivalent
energy tiers. M6 resolves A4 and supplies the comparable-tier protocol M9 needs without blocking
the correctness-only M4.

**Success measure.** At 4/6/8 qubits, at least 1,000 warmed public-entry calls are paired and
interleaved with an equivalent lower-boundary entry differing only in the language crossing.
State reset, allocation, materialization, batching, build, affinity, and warm-up are frozen.
Report component times, uncertainty, and the one-sided 95% upper bound on
`O = (T_public - T_lower) / T_public`. Meeting the QA-007 threshold satisfies the milestone;
missing it closes only through a component-level diagnosis and leaves QA-007 unmet. If overhead
is below the A4 kill threshold at every point, CAP-004 becomes a hold-the-line constraint.

**Scope and assumption.** In: harness, profile, validator, and at most one bounded interop-only
fix. Out: algorithmic fusion, channels, GPU work, and optimizer changes. Riskiest assumption: A4.

**Dependencies and handoff.** M4 and the recorded M5 verdict. Handoff slug:
`interop-overhead-profile`. Update both current-state docs with the lane and measured overhead
posture.

### M7 — `gad-channel-admission` (Next)

**Outcome and why next.** GAD is the non-unital calibration anchor named by CAP-002 and the
contrast to unital depolarizing contraction toward \(I/2^n\). M7 admits it as a per-step
fixed-parameter channel under the \((\gamma,p_\mathrm{exc})\) convention.
“Gibbs-population-calibrated” is used only where requirements declare the two-level Hamiltonian
\(H=\Delta|1\rangle\langle1|\), \(|0\rangle\) ground, \(\Delta>0\), \(\beta\ge0\), and
\(p_\mathrm{exc}=(1+e^{\beta\Delta})^{-1}\). No relaxation-time, \(T_1\), or physical-time
claim is made without a declared \(\gamma(t)\) law or rates, which M7 does not ship.

**Success measure.** Fixed-parameter GAD under the `(gamma, p_exc)` convention:
- matches the QA-010 population/coherence law ($\rho'_{11} = (1-\gamma)\rho_{11} + \gamma p_\mathrm{exc}$),
  coherence scaling $\sqrt{1-\gamma}$, thermal fixed point $\mathrm{diag}(1-p_\mathrm{exc}, p_\mathrm{exc})$,
  convergence for $\gamma > 0$, Hermiticity, and contractivity bounds;
- agrees with Aer at boundary and interior points and with the sequential reference on full
  states at 4–10 qubits for every advertised direct, partitioned, fused-island, and hybrid route;
- rejects every enumerated non-finite, out-of-domain, invalid-target, parametric-mode, and
  legacy-API request before mutation;
- is explicitly refused by strict channel-native mode and takes a labelled baseline route in
  hybrid mode.

**Scope.** In: fixed-parameter circuit-ordered GAD, binding/API, planner/runtime lowering,
calibration, external rows, and published convention. Out: parametric GAD, strict
channel-native GAD bundle, Gibbs convenience mapping, legacy API extension, VQE noise-spec
integration (M8), and physical-time/rate semantics.

**Feasibility.** The four-Kraus form generalizes delivered amplitude damping, so the support
delta is bounded: core operation, binding, planner vocabulary, runtime lowering, sequential
reset validation, and Aer mapping. Population law, fixed point, and contractivity are
closed-form. The milestone ADR—not this roadmap—decides where each implementation piece lives.

**Riskiest assumptions.** A5 (inventory sufficiency) and A6 (the sequential oracle resets
correctly for a new channel).

**Dependencies and handoff.** M4 support matrix. Handoff slug: `gad-channel-admission`.
Update both current-state docs for the admitted inventory and route boundary.

### M8 — `study-ready-noisy-workflow` (Next)

**Outcome and why next.** This is the bounded research-readiness outcome in `PHASE4_REQS.md`.
It is a **canonical, energy-only readiness surface**: the delivered density backend supports
the derivative-free `BAYES_OPT` and `COSINE` traces and refuses gradient entry points. M8 makes
that surface complete, documented, refusal-pinned, GAD-capable, and regenerable. It precedes
M9 so the research path does not depend on a fusion win. It ships neither a second family nor
a gradient product API.

**Success measure.**
- The canonical XXZ/generated-HEA family lowers under advertised fixed local depolarizing,
  phase damping, amplitude damping, and GAD at 4–10 qubits; full states meet QA-001 and energies
  meet QA-002.
- One documented public entry returns exact noisy energy with M4 route attribution.
- **Optimizer smoke—usability, not optimization quality.** One current derivative-free mode
  (`BAYES_OPT` or `COSINE`, frozen in requirements) runs under a bounded iteration and
  wall-clock budget with a pinned seed at q4 and one larger advertised width. It must complete,
  emit finite best-energy history, and reproduce sampled visited energies through the public
  evaluator. No convergence or solution-quality claim is made; upstream optimizer behavior is
  unchanged under QA-009.
- A support matrix preflight-refuses every gradient entry, gradient-based optimizer, non-HEA
  ansatz, non-generated source, unsupported gate/noise name, and parametric-noise request, each
  with a pinned negative.
- 100% of M8's counted claims regenerate from a clean checkout.

**Scope.** In: canonical GAD-capable noise specification, public energy-evaluator
documentation, support/refusal matrix, optimizer smoke, workflow evidence, clean-checkout
proof. Out: second family (M8A), gradient product API or optimizer bridge (RG-2),
trainability conclusions, selection in the optimizer loop, approximate methods, noisy
re-synthesis, new gates, state-vector partitioner changes, and upstream optimizer changes.

**Feasibility.** Energy evaluation and the two derivative-free modes already exist; the support
delta is GAD in the canonical noise specification, complete refusal evidence, a bounded smoke,
and clean-checkout regeneration. The smoke has a frozen iteration/time cap and cannot support
an optimization-quality claim.

**Riskiest assumptions.** A5 (the inventory covers the selected study) and A7 (explicit refusal
is usable).

**Dependencies and handoff.** M4 supplies the public attributed entry; M7 supplies GAD; the M5
verdict and M6 posture are recorded but do not expand scope. Handoff slug:
`study-ready-noisy-workflow`. Update both current-state docs and the non-spec API reference.

### M9 — `representation-aware-selection` (Next)

**Outcome and why next.** M3A found 0/26 channel-native cases justified because composing raw
Kraus bundles at \(R=1\) cost more than sequential application. M9 re-enters fusion with the
unresolved variables explicit—representation, canonicalization, support, construction versus
application cost, and comparable tiers—on the frozen Phase 3.1 families plus one frozen
static-subgraph candidate surface. The candidate surface is a hypothesis, not a prescribed
design. M9 tests A2 after M6 and after M8 secures research readiness.

**Success measure.** Before measurement, a milestone ADR freezes the candidate/motif surface,
representation and canonicalization policy, calibrated work unit, memory ceiling, and
positive-claim thresholds. Every partition records representation/support, logical
transformations, diagnostic Choi rank, sequential/candidate construction and application work,
peak memory, paired timings, and prediction error.
- **Complete \(R=1\) decision.**
  \[
  T_s^{(1)}=B_s+A_s+L_s,\qquad
  T_c^{(1)}=B_c+A_c+L_c,
  \]
  where \(B\) is construction, \(A\) application, and \(L\) lookup/route-selection overhead.
  Sequential build and overhead are measured, not assumed zero. QA-006 selection uses the
  one-sided 95% upper bound of \(T_c^{(1)}/T_s^{(1)}\), with no reuse credit.
- **Break-even forecast only.** The cost record reports both declared scenarios needed to avoid
  hiding reference reuse:
  \[
  T_s^\mathrm{reuse}(R)=B_s+R(A_s+L_s),\qquad
  T_s^\mathrm{rebuild}(R)=R(B_s+A_s+L_s),
  \]
  against the hypothetical cached candidate
  \(T_c^\mathrm{cached}(R)=B_c+R(A_c+L_c)\). Thus
  \[
  R^*_\mathrm{reuse}=\frac{B_c-B_s}{(A_s+L_s)-(A_c+L_c)},\qquad
  R^*_\mathrm{rebuild}=\frac{B_c}{(B_s+A_s+L_s)-(A_c+L_c)}.
  \]
  Each value is defined only for a positive denominator, carries uncertainty, and is rounded up
  for an integer reuse count; otherwise the record says “no break-even.” \(R^*\le1\) predicts
  that no reuse is needed, but an actual win still requires the QA-006 one-sided confidence
  bound at \(R=1\). These are forecast fields only and never enter M9 selection. Runtime/cache
  instrumentation identifies the applicable reuse model; independently collected optimizer
  traces provide \(R_\mathrm{obs}\). Only M12 may activate caching.
- No candidate above sequential \(R=1\) cost or the memory ceiling is selected; every executed
  selection meets QA-001/006, every route is attributed, and every motif class has a QA-004 sentinel.

The positive deployable exposes selection for eligible static subgraphs when the version-pinned
competitor feature matrix supports the claim boundary. The diagnosis deployable keeps the planner
non-selecting, publishes the cost record and failure mechanism, and records A2 as killed. GAD
motifs are excluded.

**Scope.** In: cost-record schema, frozen candidate surface, \(R=1\) selector, \(R^*\) forecast,
performance pipeline, competitor matrix, positive or diagnosis deployable. Out: reuse-conditioned
caching (M12), GAD Kraus bundle, approximate methods, GPU kernels.

**Feasibility.** The Phase 3.1 runtime provides a bounded execution surface but currently
recomposes bundles and lacks build/apply instrumentation. The milestone ADR decides the
candidate representation, canonicalization, instrumentation layer, and implementation
placement; the roadmap prescribes none.

**Riskiest assumption.** A2. The positive arm must beat sequential at \(R=1\) without reuse.
A negative result remains useful through the cost diagnosis and cannot block M8.

**Dependencies and handoff.** M3A, M6, and closed M8. Handoff slug:
`representation-aware-selection`. Update both current-state docs with the cost model, selection
surface, and governing ADR.

### Later — M8A, M10–M13 and conditional M5A

- **M8A `second-family-workflow`.** Add exactly one pre-registered, independently motivated
  task/ansatz family under the M8 refusal discipline and clean-checkout proof. It ships no
  gradient product API.
- **M10 `strict-parameter-domains`.** Replace clamps on advertised parametric paths with
  structured rejection or an explicit recorded transform under an Ask-first ADR. This is the
  global QA-005 milestone.
- **M11 `clean-checkout-reproduction`.** Have the external-reproducer persona regenerate the
  accumulated claim set; turn the regeneration index into the standing QA-008 surface for M13.
- **M12 `observed-reuse-amortization`.** Open only when M9 shows material construction and real
  optimizer traces independently show \(R_\mathrm{obs}\) beyond forecast \(R^*\); it is the only
  milestone allowed to activate reuse-conditioned caching. It remains independent of M9's
  \(R=1\) verdict.
- **M13 `noisy-trainability-studies`.** Execute the pre-registered multi-family scientific
  endpoint. Minimum migration requires a gauge-invariant state/energy definition, the earlier
  “SNR” becomes a gradient-norm coefficient of variation, and \(L^*\) is explicitly
  optimizer-, budget-, and ensemble-conditioned. Gradient-based training arms require RG-2.
- **M5A `exact-regime-boundary`.** Open only through RG-1. It characterizes the exact boundary
  and records escalation; approximate methods or GPU dependency require product-statement revision.

### Feasibility guidance (Now/Next)

Guidance, not design: implementation choices belong to each milestone's ADRs. “Support delta”
is what exists today versus what the outcome needs; “kill” is the pre-registered stop.

| M# | Current support delta | Expected slices | Evidence lanes | Compute budget / kill condition |
|----|-----------------------|-----------------|----------------|---------------------------------|
| M4 | Public C++ density energy and planner→runtime state execution exist; attributed runtime energy, energy route rows, and the support matrix do not | 2–3: q4 partitioned tracer; fused/hybrid routes; q6–q10, Aer, negatives | fast pytest; correctness pipeline; Aer | Existing correctness protocol; any QA-001 route disagreement freezes downstream claims (A6) |
| M5 | No density gradient product exists; benchmark instrumentation must derive U3 coordinate shifts from `{2,1,1}` and pass finite-difference plus analytic controls before counting | 3: q4 gradient tracer; non-counted pilot and threshold freeze; counted screen and verdict | new study-specific sensitivity lane; fast pytest witnesses | \(\sum_n 4S_nK_n\) shifted evaluations for paired noise/no-noise gradients, frozen from pilot timing; shrink descriptive noise/depth points first, never a primary width or \(S_n\) without formal revalidation |
| M6 | No paired/interleaved equivalent-tier harness; legacy performance scripts do not isolate the language crossing | 2: harness/q4; q6/q8, validator, diagnosis | performance pipeline | ≥1000 warmed calls per tier/width plus repeats; inability to create equal-work tiers closes as diagnosis with QA-007 unmet |
| M7 | GAD is absent from core, bindings, planner vocabulary, runtime bundles, and Aer mapping; amplitude damping is the template | 2–3: core/binding/calibration tracer; planner/runtime/refusals; q4–q10 Aer evidence | fast pytest; correctness pipeline; Aer; optional C++ | Existing correctness protocol; sequential–Aer disagreement beyond QA-002 freezes admission |
| M8 | Canonical energy and `BAYES_OPT`/`COSINE` support exist; missing GAD noise specification, complete refusal matrix, bounded optimizer smoke, and clean-checkout bundle | 2–3: GAD-in-spec/matrix; q4–q10/Aer; smoke and clean checkout | fast pytest; workflow pipeline; Aer; clean-checkout lane | Fixed smoke iterations and wall-clock cap; if q10 is unaffordable, M8 is partial/missed unless roadmap and requirements are formally revalidated before data—no silent width reduction |
| M9 | Channel-native runtime recomposes per call; records lack build/apply split, Choi rank, and partition/representation-attributed memory accounting; no candidate surface or competitor matrix is frozen | 3–4: schema/instrumentation over 26-row matrix; candidate surface and \(R=1\) selector; competitor matrix and verdict; split is mandatory if any Layer artifact exceeds its budget | performance pipeline; correctness pipeline; fast no-regression tests | Paired/interleaved trials under memory ceiling; if no candidate upper bound is ≤1 at \(R=1\), ship diagnosis, kill A2, keep planner non-selecting, retain \(R^*\) only as M12 forecast |

## 5. Sequencing rationale

- **Walking skeleton first.** M4 is the smallest useful positive path on the primary persona's
  public object that crosses every currently connectable product boundary. Full-state checks
  prevent energy agreement from hiding a wrong state. GAD-only admission is a channel slice, not
  a product skeleton; interop alone is a measurement.
- **Highest-impact scientific risk early, one identifiable question.** M5 screens A1 with one
  paired estimand at fixed design points, so no exponent is fitted to four widths and no verdict
  is over-read as scaling.
- **Interop → GAD → energy-only readiness → selection.** M6 supplies comparable tiers; M7 the
  bounded \((\gamma,p_\mathrm{exc})\) anchor; M8 the canonical energy-only surface on delivered
  derivative-free traces with refusals pinned; M9 then tests selective fusion without holding
  the research path hostage.
- **\(R=1\) decision first, reuse forecast only.** Phase 3.1's 0/26 was an \(R=1\) result with
  raw bundles. M9 must win or lose at \(R=1\) under complete build/application accounting and
  comparable tiers; \(R^*\) is a forecast, and only M12 may activate caching after
  \(R_\mathrm{obs}\) is observed on real traces.
- **Gradients wait for a contract.** No current `CAP-*`/`QA-*` defines a differentiable
  objective or gradient fidelity. RG-2 makes product-statement evolution the entry ticket.
  U3 multipliers `{2,1,1}` and the stored \(\theta/2\) coordinate make a coordinate-blind
  shift rule a correctness bug, not an implementation detail.
- **Phase 4 stays bounded.** Its delivery outcomes are M4, M6, M7, M8, and M9, with M5 as an
  early scientific-risk screen. M8A, M10–M13, and all review-gated work remain Later or conditional.
- **Archived order and gates remain intact.** M2 stays the exact workflow anchor; M3/M3A are not
  reopened; workflow broadening follows their delivered baseline; trainability follows
  readiness; invasive fusion or scaling work remains evidence- and review-gated.
- **Scope cuts remain protected.** Now/Next contain no noisy re-synthesis, full gate parity,
  approximate simulator, GPU dependency, or scientific claim based on whole-register
  depolarizing. Papers consume closeouts and are not roadmap milestones.

## 6. Assumptions, risks & revalidation log

### Strategic assumptions

| Assumption | Validated by | Consequence if it fails |
|------------|--------------|-------------------------|
| A1 Exact regime resolves the target effect | M5 paired mean-squared-gradient screen; M13 study | Product/ADR review; M5A only on approval; finite-size claim boundary |
| A2 Selective fusion finds paying workloads | M9 QA-006 at \(R=1\), with \(R^*\) forecast but never credited | Ship diagnosis, keep planner non-selecting, do not open selection-in-loop |
| A3 Real traces contain amortizing reuse | M12, gated on M9 cost record and independently observed \(R_\mathrm{obs}\) from M8/M8A traces | Drop reuse and optimize per evaluation only |
| A4 Interop overhead materially limits iteration | M6 | Make CAP-004 a hold-the-line constraint |
| A5 Inventory through GAD covers core studies | M7 GAD anchor, M8 canonical specification, M8A second family | Revisit CAP-002 through Ask-first; no speculative breadth |
| A6 Sequential executor is a trustworthy oracle | M4, M7, every new path | Freeze downstream claims until disagreement is resolved |
| A7 Researchers accept explicit refusal over clamping | M8 in use; M10 migration | Improve diagnostics or explicit transforms; never restore silence |

### Roadmap risks

| Risk | Mitigation |
|------|------------|
| M4 expands into channels, timing, or optimizer integration | Bind it to the support matrix and one-evaluation outcome |
| M5 verdict is over-read as asymptotic scaling | One paired-difference estimand with simultaneous width-wise intervals; widths are design points; no exponent fit; restate the transfer boundary |
| Benchmark gradient applies a coordinate-blind shift | Derive shifts from U3 multipliers `{2,1,1}`; require finite-difference and analytic controls including \(\phi/\lambda\) witnesses; keep gradients benchmark-internal until RG-2 |
| M5 estimand is diluted by structurally inactive coordinates | Freeze the active-coordinate rule before data |
| M5 compute \(\sum_n 4S_nK_n\) exceeds budget | Pilot-timed freeze; shrink descriptive noise/depth points before data, never a primary width or \(S_n\) without formal revalidation |
| Deferred thresholds remain open | Freeze each `[confirm]` value in the first citing `INITIAL_REQUIREMENTS.md` |
| Optional Aer dependency drifts | Pin versions in every QA-002 evidence bundle |
| Scoped QA-005 is mistaken for global compliance | Label every scoped trace; reserve global claim for M10 |
| M9 credits unobserved reuse | Selection and QA-006 remain \(R=1\); \(R^*\) is forecast-only; caching activates only in M12 after \(R_\mathrm{obs}\) |
| M9 candidate family has no eligible static subgraph | Record “no candidate” and use the attributed baseline route; candidate surface remains an ADR hypothesis |
| Phase 4 creeps into gradient or second-family product work | §1 boundary, RG-2, and M8A Later placement |
| GAD receives unconditioned Gibbs or time language | Per-step \((\gamma,p_\mathrm{exc})\); Gibbs wording only with declared Hamiltonian/basis/\(\beta,\Delta\); no time claim without \(\gamma(t)\) |
| M8 optimizer smoke is read as optimization quality | Assert completion, finite/reproducible energies, and refusal behavior only; do not claim convergence |
| M9 revives an unjustified novelty claim | Require the version-pinned competitor matrix before any positive claim |
| Two noise representations drift | Keep legacy `NoiseChannel` non-claim-bearing and unextended |

### Revalidation log

- **2026-09-21 — v0.5, source-verified scientific correction of v0.4.**
  *Learned:* U3 stored coordinates have multipliers `{2,1,1}` with physical \(\theta\) stored
  as \(\theta/2\), so v0.4's universal
  \(\tfrac12[E(p+\pi/2)-E(p-\pi/2)]\) rule is identically wrong on that coordinate; density
  optimizer support is `BAYES_OPT`/`COSINE` only and gradient entry is refused; the
  channel-native runtime recomposes bundles per call and existing records lack build/application
  decomposition; GAD is absent; QA-006 forbids reuse credit at \(R=1\), QA-012 makes
  unobserved reuse forecast-only, and no current `CAP-*`/`QA-*` covers a differentiable product
  objective. *Changed:* M5 now has one paired, coordinate-aware, control-validated estimand;
  \(\Delta\theta^*\), misnamed SNR, and \(L^*\) moved to M13 under definition conditions;
  M8 became canonical energy-only readiness with a derivative-free smoke and refusal matrix;
  M8A adds the second family Later; RG-2 gates any gradient product work; M9 selects only at
  \(R=1\) under complete sequential/candidate costs and emits \(R^*\) as forecast; M12 alone
  activates caching after observed \(R_\mathrm{obs}\); GAD language and the Phase 4 boundary
  were tightened; feasibility guidance and kill conditions were added. Final critique then
  froze M5 to one paired-difference estimand with simultaneous width-wise intervals, made both
  sequential reuse models and lookup overhead explicit in M9 forecasts, prohibited silent M8
  width reduction, and completed QA-008/009 traces.
- **2026-09-21 — v0.4, scientific research enhancement (Phase 4 goals & thesis alignment).**
  *Learned:* Evaluating $n \in \{4,6,8,10\}$ solely against asymptotic barren-plateau scaling
  risks premature false-negative escalation; static $R=1$ fusion testing repeats Phase 3.1's 0/26
  outcome; and legacy C++ VQE code throws on gradient requests.
  *Changed:* M5 enhanced with finite-size landscape observables ($\Delta\theta^*$, SNR, $L^*(p)$);
  M7 explicitly anchored to non-unital thermal relaxation ($p_\mathrm{exc} \leftrightarrow \beta \Delta$);
  M8 equipped with exact 2-point parameter-shift gradient routing and a Python-level SciPy optimizer bridge;
  M9 enhanced with subgraph factoring and a break-even reuse predictor ($R^*$).
- **2026-09-20 — roadmap created after archived Phase 3.1 closure and revalidated through two
  critique revisions (v0.1–v0.3).** Phase 3 had 34 counted supported cases, with 0/6 representative cases
  passing the positive threshold and 6/6 closing through diagnosis; therefore M6 separates
  interop cost before another fusion claim. Phase 3.1 classified 17/26 cases
  `phase3_sufficient`, 9/26 `phase31_not_justified_yet`, and 0/26
  `phase31_justified`; therefore M9 re-enters fusion only through a representation-aware cost
  record and kill criterion, while M12 treats reuse independently. Critique moved A1 forward
  into M5, replaced the original GAD-only skeleton with the positive attributed-energy M4,
  separated GAD from selection, promoted bounded study readiness ahead of selection, scoped
  QA-005 until clamp migration, and made M5A review-gated.

### Critique verdict

- Outcome honesty: delivered milestones record the thresholds actually met, including negative
  performance decisions.
- Sequencing: M4 validates integration/convention risk; M5 screens the highest-impact scientific
  assumption; A4 precedes A2; study readiness precedes and is independent of A2; A3 is gated last.
- Deferred work is visible: gradient product work sits behind RG-2 with its required upstream
  contracts named; the second family is M8A; reuse activation is M12; strict migration is M10;
  M5's removed observables are conditional M13 work.
- Deployability: M4–M8 are bounded to a few slices; M9 must split during requirements if its
  frozen matrix does not fit any Layer artifact budget.
- Orphans: none. Every milestone has non-empty CAP/QA traces.
- Escalation: no current evidence breaks the North Star; M5 is the explicit A1 review trigger.

**Critique verdict:** Ready — scientifically corrected (v0.5); M4 handoff unchanged.

### Stakeholder checkpoint and handoff

> Is this sequence right? Which milestone outcomes, measures, or dependencies are wrong,
> missing, or mis-prioritized before M4 opens?

**Handoff:** invoke `create-initreq-for-sdd` for M4 `canonical-attributed-energy` with its
outcome, support matrix, CAP-001/003/005/007 and QA-001/002/005-scoped/008/009 traces,
dependencies, deployable result, and the instruction to update—not create—the current-state
architecture and stack docs at close.
