# Product statement — SQUANDER density-matrix track

> **Status:** draft v0.3, at stakeholder checkpoint · **Owner skill:** `create-product-statement` ·
> **Scope:** the durable North Star of the density-matrix (noisy mixed-state) track; the current
> emphasis is the direction recorded in `PHASE4_REQS.md` ·
> **Traces down to:** `ROADMAP.md` (`M#`) → `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) ·
> **Not:** sequencing (`ROADMAP.md`), architecture (`ARCHITECTURE_OVERVIEW.md`, ADRs), stack
> (`TECH_STACK.md`), or publication planning ·
> **Sources:** `docs/density_matrix_project/README.md`, `RESEARCH_ALIGNMENT.md`,
> `archive/planning/PLANNING.md` §2–3, §8, `archive/planning/ADRs.md` ADR-001…008, the CSCS 2026
> short paper and talk (`docs/density_matrix_project/cscs2026/`), `PHASE4_REQS.md`.

## 1. Vision

For researchers training variational quantum circuits under realistic noise, the SQUANDER density-matrix track is an **exact open-system simulation module inside a circuit compiler/optimizer** that treats noise channels as first-class objects of partitioning and fusion, so selected noisy workloads execute at lower representation-aware end-to-end cost without ever trading exactness. Every advertised path is proven against a sequential reference under a frozen, auditable protocol, and unsupported input fails loudly instead of degrading silently.

*Decision test:* rules out approximate scaling, breadth-first channel or gate parity, convenience fallbacks, and raw-throughput races; rules in exactness contracts, noise-aware planning, certified channel admission, and regenerable evidence.

## 2. Customers & problem

**Primary persona — the noisy-training researcher.** A PhD or postdoctoral researcher (the SQUANDER group and its collaborators) who studies how variational circuits train under device-motivated local noise in the exact regime. Job to be done: obtain trustworthy noisy energies and states thousands of times inside an optimizer loop, know which model and route produced each number, and defend those numbers to reviewers.

**Secondary personas.** The *SQUANDER maintainer*, who needs the noisy backend to stay additive to the state-vector path; the *external reproducer or reviewer*, who needs to regenerate a published number from a clean checkout.

**Core problem.** Exact mixed-state evolution costs \(2^{2n}\) per operator application, and in the delivered Phase 3 unitary-island baseline each noise channel interrupts fusion; Phase 3.1 crosses only bounded noise motifs without a justified speedup. Existing simulators support noisy channels and some fusion, but their public interfaces do not necessarily expose the representation-aware decisions, route attribution, and regenerable evidence needed here. Silent substitutions and incomplete cost accounting turn a benchmark into a guess; feasibility is not acceleration.

**Today.** The researcher pays the sequential cost per channel and Python dispatch per evaluation, discovers unsupported inputs by trial, and cannot fully attribute wall-clock time.

**In the future.** The researcher declares workload-justified local noise, uses a planner that sees channels and representation cost, gets lower amortized cost on eligible workloads with attributed routes, bounds interop overhead, and regenerates every counted claim. Unsupported work fails before mutating \(\rho\), or undergoes an explicit caller-selected transform.

## 3. Value proposition & differentiation

**Lead differentiator — inspectable noisy planning and evidence.** Noise is a first-class planner input: an eligible ordered gate+noise region may become one exact CPTP object, while eligibility, predicted cost, actual route, and unsupported boundaries remain observable. Strict proof cases validate the object; whole-workload cases evaluate it with per-partition attribution and no silent substitution.

**Supporting differentiators.**
- *One stack.* Compiling, optimizing, and noisily simulating a variational circuit share one
  circuit model and one optimizer loop; the noisy backend is selected, not bolted on.
- *Evidence as a product output.* Counted cases, frozen tolerances, pre-registered thresholds,
  pinned seeds and revisions, and diagnosis-grounded closure when a hypothesis fails.

**Alternatives, honestly.** Aer supports general Kraus errors, superoperator simulation, and fusion for density-matrix/superoperator methods; QuEST and Qulacs accept general channel maps; other frameworks provide broader optimization or physics surfaces. SQUANDER does **not** claim to originate CPTP composition or noisy-operation fusion. Its target distinction combines integrated planning, explicit eligibility, predicted-versus-executed cost, route attribution, strict refusal, and regenerable baseline comparisons. Literature Positioning maintains a version-pinned, cited feature matrix; no priority claim is made without it.

## 4. Product capabilities (`CAP-*`)

Ids are stable: extend, never renumber. Each is an outcome a researcher can reach.

| Id | Capability (durable outcome) | Why / value | Success signal |
|----|------------------------------|-------------|----------------|
| **CAP-001** | **Exact noisy evolution.** A researcher evolves any supported ordered gate+noise circuit on an exact density matrix and obtains a trace-preserving, positive state carrying the ordered open-system semantics, on every shipped execution path. | Exactness disentangles noise effects from approximation error; it is the anchor for every later claim (ADR-005). | 100 % of counted correctness cases on every shipped path meet QA-001; exact-regime coverage at 4–10 qubits is maintained in every milestone closeout. |
| **CAP-002** | **Realistic, extensible noise inventory.** A researcher expresses workload-justified local noise spanning unital dynamics (depolarizing, phase damping) and generally non-unital / finite-temperature dynamics (amplitude damping, generalized amplitude damping), under an explicit parameter convention. | Trainability questions hinge on physically different channel classes; workload-driven growth avoids breadth without scientific purpose (ADR-004, ADR-006). | Each advertised channel meets QA-001 and QA-010 on every path that claims support; all other paths reject it at preflight; generalized amplitude damping (GAD) supplies the next finite-temperature calibration anchor. |
| **CAP-003** | **Representation-aware selective fusion.** A researcher runs a noisy circuit through a planner that lowers total logical transformations only when a frozen cost record predicts no hidden representation, construction, application, or reuse penalty, with every partition's route attributed. | Raw Kraus count is non-unique and logical step reduction can hide greater work; scientific progress requires predicting and measuring the complete trade-off (ADR-002, ADR-003). | QA-006 selection and, where claimed, QA-012 observed-reuse amortization are met; 0 evaluation-mode selections violate the no-regression rule; 100 % route attribution. |
| **CAP-004** | **Low-overhead iterative evaluation.** A researcher evaluates exact noisy energies thousands of times in a loop with language-boundary, construction, allocation, and dispatch costs separately measured and bounded rather than hidden inside total time. | Variational workflows amplify per-call overhead; comparable execution tiers are required to distinguish an algorithmic result from an implementation artifact. | QA-007 met across the exact-regime benchmark points; every performance closeout publishes the component-level profile and comparison method. |
| **CAP-005** | **Exact noisy observables in the variational workflow.** A researcher selects the density backend in the same variational loop that runs state-vector and obtains exact \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\) energies for a supported ansatz and noise specification, with the support surface stated and unsupported requests refused. | Turns the simulator into a usable training backend; the frozen canonical workflow is the proof of usability (ADR-001). | Energies agree with Qiskit Aer per QA-002 at 4–10 qubits; every widening of the surface ships with a pinned negative test. |
| **CAP-006** | **Noisy variational research on the exact backend** *(destination)*. A researcher tests pre-registered hypotheses about optimization and trainability under realistic local noise and distinguishes finite-size observations from scaling claims. | This is the research outcome the module exists to serve; integration alone is not thesis-level scientific evidence (PLANNING §2.1). | QA-011 met across at least two independently motivated task/ansatz families, unital and non-unital noise sweeps, and multiple seeded initializations, with effect sizes, uncertainty, and null outcomes regenerated; depends on CAP-004/005. |
| **CAP-007** | **Regenerable verification and calibration evidence.** A researcher or reviewer regenerates every counted claim — exactness, physical calibration, representation-aware cost, overhead, and scientific inference — from one named lane with tolerances, rates, seeds, revision, and claim boundary pinned. | Reproducibility is a first-class output (PLANNING §2); a claim that cannot be regenerated is not a claim. | QA-008 met: 100 % of counted rows regenerate; every claim row names its lane; no counted claim rests on a non-regenerable artifact. |

**Current-direction trace (`PHASE4_REQS.md`; sequencing belongs in `ROADMAP.md`).** Channel
Expansion → CAP-002, QA-003/010; Partitioning and Fusion → CAP-003, QA-004…006/012; Interop
Optimization → CAP-004, QA-007; Verification and Calibration → CAP-007, QA-001…003/008/010;
Literature Positioning → §3. Its expected deliverable is readiness for CAP-006, not a
trainability conclusion.

## 5. Quality attributes (`QA-*`)

Every bar is a scenario with a response measure, so it becomes a fitness function and an
evidence-matrix row. Baselines: the **sequential `NoisyCircuit` executor** (internal exact
reference), **Qiskit Aer density-matrix** (external reference), and **analytical open-system
models** (closed-form calibration reference). Values marked `[confirm]` are owned by the
product owner (Z. Kégli) and are fixed in the first `INITIAL_REQUIREMENTS.md` that cites the id.

| Id | Scenario | Response measure | Lane |
|----|----------|------------------|------|
| **QA-001 Exactness** | When any non-sequential path (partitioned, fused, channel-native, or a future backend) evolves a supported workload at 4–10 qubits | \(\lVert\Delta\rho\rVert_F \le 10^{-10}\) and \(\lVert\Delta\rho\rVert_{\max} \le 10^{-10}\) vs the sequential reference; \(\lvert\mathrm{Tr}\,\rho - 1\rvert \le 10^{-10}\); \(\lambda_{\min}(\rho) \ge -10^{-12}\); on 100 % of counted cases | correctness evidence pipeline; fast pytest |
| **QA-002 External agreement** | When a channel, support width, or execution path is newly advertised | Its external slice includes boundary and interior parameter points and at least one case per advertised support width/path; \(\lVert\Delta\rho\rVert_F < 10^{-12}\) vs Qiskit Aer and, for the variational surface, \(\lvert\Delta E\rvert \le 10^{-10}\) `[confirm]` | Qiskit Aer external reference |
| **QA-003 Channel admission** | When a built-in channel, or a future public channel representation, is admitted | Parameters and matrices are finite, dimensions/support are valid, and trace-preservation residual is \(\le 10^{-10}\); Kraus input is CP by construction, while any other representation independently establishes CP under its downstream contract; 100 % of the enumerated malformed-input classes raise a structured pre-mutation error | fast pytest (admission and negative classes) |
| **QA-004 Ordered semantics** | When a fused object replaces an ordered motif | Every counted result meets QA-001 against the ordered sequential reference; for each newly admitted motif class, a pinned non-commuting sentinel (fixed motif, parameters, and witness state) differs from its reversed order by the pre-registered map- or state-level threshold `[confirm]` | fast pytest |
| **QA-005 Strictness** | When unsupported input, an ineligible strict-mode partition, or an out-of-domain parameter is presented | A structured error is raised before mutation, unless the caller explicitly selected a named parameter transform recorded in metadata; 0 silent route substitutions or clamps; 100 % of executed evaluation-mode partitions carry a route label | fast pytest (support-boundary suite); planner-surface evidence |
| **QA-006 Representation-aware selection** | When a pre-registered workload executes selective fusion with no unobserved reuse credit (\(R=1\)) | Record representation/support, logical transformations, diagnostic Choi rank (never truncation), construction, a dimension-aware calibrated work unit, peak memory, paired/interleaved trials, and prediction error; select no candidate predicted above sequential cost or the frozen memory ceiling `[confirm]`; every executed selection must have a one-sided 95 % upper confidence bound on \(T_\mathrm{candidate}/T_\mathrm{sequential}\le1\); a positive claim additionally requires \(\le0.70\times\) logical transformations `[confirm]`, no greater executed work, and the bound \(\le\delta_\mathrm{eff}<1\) `[confirm]` | performance evidence pipeline; fast pytest |
| **QA-007 Interop overhead** | When the same prebuilt noisy-energy evaluator is called \(\ge1000\) warmed times through the public Python entry and an equivalent lower-boundary invocation where only the language crossing differs, at 4/6/8 qubits | Freeze state reset, allocation, output materialization, batching, build, threading/affinity, and warm-up; paired/interleaved trials estimate \(O=(T_\mathrm{public}-T_\mathrm{lower})/T_\mathrm{public}\); report components and uncertainty; the one-sided 95 % upper bound on \(O\) is \(\le10\,\%\) `[confirm]`; diagnosis may close a research milestone but leaves QA-007 unmet | performance evidence pipeline |
| **QA-008 Reproducibility** | When a counted claim is cited in a closeout or paper | Categorical classifications reproduce exactly and numerical residuals within frozen tolerance on 100 % of rows; performance decisions reproduce against their stated margin (not identical timings) under pinned CPU, compiler/flags, dependencies, threading/affinity, warm-up and sampling protocol; every bundle pins channel conventions, representation/cost policy, rates, seeds, revision, and claim boundary | benchmark tests; evidence pipelines |
| **QA-009 Non-interference** | When any density-track change lands | The upstream state-vector suite passes with 0 regressions and the density backend remains opt-in (default backend unchanged) | CI |
| **QA-010 Physical calibration** | When an advertised channel acts on pre-registered analytical states | Output populations/coherences match closed form within \(10^{-12}\) `[confirm]`; \(\lVert\rho-\rho^\dagger\rVert_F\le10^{-14}\); trace-distance contractivity holds within \(10^{-12}\) on declared pairs; with \(|0\rangle\) ground, \(\gamma,p_\mathrm{exc}\in[0,1]\), GAD satisfies \(\rho'_{11}=(1-\gamma)\rho_{11}+\gamma p_\mathrm{exc}\), has fixed point \(\mathrm{diag}(1-p_\mathrm{exc},p_\mathrm{exc})\), and converges to it for \(\gamma>0\); “Gibbs” requires \(H=\Delta|1\rangle\langle1|\), \(\Delta>0\), \(\beta\ge0\), and \(p_\mathrm{exc}=(1+e^{\beta\Delta})^{-1}\) | fast pytest; correctness and Aer evidence |
| **QA-011 Scientific inference** | When a noisy-training or trainability conclusion is claimed | 100 % of claim-bearing studies pre-register hypothesis, estimand, comparators, task/ansatz families, size/depth/noise grid, initialization ensemble, sample-count rationale, and finite-size boundary; report effect sizes, uncertainty, and null outcomes; raw results and analysis regenerate | study-specific evidence pipeline |
| **QA-012 Observed-reuse amortization** | When fusion benefit is attributed to reuse | Derive \(R_\mathrm{obs}\) independently from real optimizer traces; identify cache key and parameter dependencies; report hits/misses, build/apply time, memory, and break-even \(R\); unobserved \(R\) is forecast only; at \(R_\mathrm{obs}\), the paired one-sided 95 % upper bound on amortized \(T_\mathrm{candidate}/T_\mathrm{sequential}\) is \(\le\delta_\mathrm{reuse}<1\) `[confirm]` within the frozen memory ceiling | performance evidence pipeline |

## 6. Ubiquitous language (seed glossary)

| Term | Meaning |
|------|---------|
| **Density matrix** \(\rho\) | Exact mixed-state representation of an \(n\)-qubit register; \(2^n \times 2^n\), unit trace, positive semidefinite. |
| **Noisy circuit** | An ordered list of operations, each a unitary gate or a local CPTP channel on a qubit subset; order is semantically binding. |
| **Noise channel / CPTP map** | A completely-positive trace-preserving map; the only admissible form of noise. Kraus form establishes CP by construction; admission still validates shape, finiteness, support, and trace preservation. |
| **Local noise** | A channel on 1–2 qubits; the scientific workload. **Whole-register noise** is a labelled baseline or stress test only. |
| **Motif** | A contiguous sub-sequence of a noisy circuit with bounded support that contains at least one channel. |
| **Fused channel object** | The exact CPTP map of a motif, composed in operation order, in whatever certified representation an ADR selects. |
| **Channel-native fusion** | Executing a motif as its fused channel object instead of step by step. **Unitary-island fusion** fuses gate-only runs between channels. |
| **Logical transformation** | One ordered gate or complete channel application in the circuit semantics. Its count describes structural contraction but is **not** a cost metric by itself because equivalent channel representations can have different execution work. |
| **Canonical channel complexity** | Diagnostic numerical Choi rank under a frozen rank policy; representation-invariant under that policy and never permission to truncate an exact execution. |
| **Execution-cost record** | The declared representation/support, logical transformations, construction/composition work, executed full-state work, memory, measured time, parameter dependencies, reuse horizon, and predicted-versus-executed route. |
| **Sequential reference** | The unfused, step-by-step `NoisyCircuit` execution; the internal exact oracle. **External reference:** Qiskit Aer. **Analytical reference:** a closed-form open-system model. |
| **Strict / hybrid** | Strict: every partition must be eligible or the run fails (proof mode). Hybrid: eligible partitions take the channel-native route, the rest the supported baseline, each with a **route label** (evaluation mode). |
| **Support surface** | The explicitly supported gates, channels, circuit sources, and modes; everything outside raises a structured error. |
| **Advertised support** | A surface named in a claim-bearing support matrix, not merely exposed by an API. The circuit-ordered operation model is authoritative; legacy standalone `NoiseChannel` APIs are excluded unless independently passing QA-001/002/010. |
| **Counted case** | A workload whose result is a claim-bearing row in an evidence bundle; a **continuity anchor** ties a new bundle to an earlier one. |
| **Frozen tolerance / pre-registered threshold** | A numerical policy or decision bar fixed before evidence is generated; changed only through the Ask-first guardrail. |
| **Diagnosis-grounded closure** | Closing a performance hypothesis by evidence of where time or steps go, when the pre-registered arm is not met. |
| **Reuse-heavy workload** | A training-relevant circuit trace in which measured parameter-independent channel components or motifs recur enough to amortize construction; reuse is observed, not assumed from repeated topology. |
| **Exact regime** | Qubit counts at which dense exact simulation is routine: 4–10 today, 12 as stretch `[confirm]`. |
| **Canonical noisy workflow** | The frozen supported variational workflow (generated HEA, ordered fixed local noise, XXZ Hamiltonian) used as the usability proof and continuity anchor. |

Candidate domain areas (high level only): mixed-state core · noise inventory · noisy planning
and execution · variational workflow · evidence.

## 7. Guardrails & constraints

**Always**
- Validate every new or changed execution path against the sequential reference within the
  frozen tolerances *before* any performance statement about it.
- Argue correctness and performance separately; pre-register every decision threshold before
  the matrix runs. A performance statement reports the baseline trio (sequential, existing
  fused, candidate), execution-cost record, comparable execution tiers, uncertainty, and route
  attribution; logical transformation count never stands alone.
- Use realistic local noise as the scientific workload; label whole-register noise as baseline.
- Publish each channel's parameter-to-map convention; use physical-time or Gibbs terminology
  only when the implemented law and Hamiltonian mapping justify it.
- Widen a support surface only with a matching negative test that pins the rejected input and
  the error it raises.
- Keep the state-vector path behaviorally unchanged; the density backend is additive and opt-in.
- Name the lane for every evidence claim and run it in the documented environment.
- Version-pin and cite every competitor capability; do not infer absence from an undocumented
  interface.

**Ask first**
- Changing any frozen tolerance, cost/canonicalization policy, or pre-registered threshold.
- Admitting a channel family outside the local 1–2-qubit CPTP inventory (correlated multi-qubit
  channels, coherent over-rotation, trainable/layer-varying rates, readout or shot noise).
- Introducing an approximate method (stochastic trajectories, MPDO/tensor networks) or a GPU
  dependency into the density path.
- Changing the semantics of the sequential reference itself, or superseding a program ADR.
- Retiring or superseding a `CAP-*` / `QA-*`.

**Never**
- Fall back silently — to the state-vector path, to an unfused route, or to a different
  representation; raise a structured error instead.
- Clamp or otherwise transform an out-of-domain scientific parameter without an explicit,
  caller-selected transform recorded in evidence.
- Truncate a fused channel object or otherwise trade exactness for speed.
- Loosen, skip, or delete a test or tolerance to make a lane pass.
- Claim a speedup without the baseline trio and route attribution, or conflate feasibility with
  acceleration.
- Claim priority, uniqueness, or competitor absence without a version-pinned source matrix.
- Extend the frozen archive, or renumber a `CAP-*` / `QA-*`.

**Fixed constraints.** An additive module of the upstream SQUANDER library that must build and
pass within its toolchain and CI (`TECH_STACK.md`); all product commands run in the documented
`qgd` environment; Phases 1–3.1 are frozen history under `docs/density_matrix_project/archive/`;
exact dense density matrices remain the reference backend (ADR-005); code, configuration, and
evidence are open-source in branch scope.

## 8. Strategic assumptions & risks

Ordered by how much scope dies if wrong. Each carries a validation route and a kill criterion.

| # | Assumption | Status | Validation | Kill criterion |
|---|------------|--------|------------|----------------|
| A1 | The exact regime can resolve the intended trainability effect, not merely exhibit a finite-size example. | **Unvalidated.** ADR-005 selects the exact anchor but is not evidence that \(n\le12\) distinguishes competing scaling laws. | Before a scaling claim, pre-register finite-size sensitivity/power on a known benchmark and the estimand in QA-011. | The target effect cannot be distinguished in the exact regime → label results finite-size only and escalate ADR-008 if the thesis question requires scale. |
| A2 | Representation-aware selection finds workloads where exact fusion lowers cost after channel expansion is counted. | **Unvalidated.** Phase 3.1 found 0/26 justified and used raw composed bundles. | QA-006 on pre-registered families with frozen cost/canonicalization policy and comparable execution tiers. | No target family satisfies QA-006 → fusion leaves the critical path; exact channels, interop, observables, and evidence continue. |
| A3 | Real variational traces contain enough parameter-independent reuse to amortize fused-object construction. | **Unvalidated.** Repeated topology does not imply reusable values; the current path recomposes each execution. | QA-012 at independently observed \(R_\mathrm{obs}\) on real optimizer traces. | Observed reuse never reaches break-even, or useful motifs are parameter-dependent → drop the reuse claim and optimize per-evaluation execution only. |
| A4 | Language-boundary and dispatch overhead materially limits iterative evaluation. | **Partially evidenced.** Python-level overhead exists; the energy loop lacks a comparable-tier profile. | QA-007 at 4/6/8 qubits. | Attributable overhead is already \(<5\,\%\) at every point → CAP-004 becomes a hold-the-line constraint, not an optimization theme. |
| A5 | The built-in local inventory through GAD covers the core studies. | **Partially validated.** Three channels carried delivered evidence; GAD is not delivered. | QA-010 plus the first study's pre-registered noise specification. | A core study requires correlated, coherent, measurement, or trainable noise → revise CAP-002 through Ask-first rather than adding breadth speculatively. |
| A6 | The sequential `NoisyCircuit` executor is a trustworthy internal oracle. | **Validated only on the delivered bounded circuit-ordered three-channel slices**; legacy standalone channels are not claim-bearing. | QA-002/010 reset validation for every new channel/path. | Any counted disagreement beyond tolerance freezes downstream claims until resolved. |
| A7 | Researchers accept explicit rejection or transforms over silent convenience behavior. | **Unvalidated; current parametric paths clamp out-of-range rates.** | Migrate or expose that behavior, pin QA-005 negatives, and collect first external-reproducer feedback. | Users bypass the path because strict input is unusable → improve diagnostics or add explicit transforms, never restore silent behavior. |

**Decisions from elicitation and review (2026-09-20).** Whole-track North Star with the current
direction as emphasis; primary persona the noisy-training researcher; inspectable planning is
the differentiator; GAD is the next calibration anchor while generic breadth is not a goal;
logical count cannot stand alone; the current direction ends at research readiness, while noisy
training remains the destination; finite-size sufficiency is the highest-risk assumption.

## 9. Out of scope / non-goals

- **Approximate scaling** — stochastic trajectories, MPDO or tensor-network mixed states — until
  the exact regime's limit is hit and escalated (ADR-008).
- **GPU kernels** — a separate track; may be integrated per milestone, never a dependency of the
  exactness contract.
- **Noise outside the local CPTP inventory** — readout/measurement/shot noise, correlated
  multi-qubit channels, coherent over-rotation, and trainable/layer-varying rates. Fixed rates
  may be swept between evaluations under their declared domains.
- **Generic custom-map breadth as a product goal** — arbitrary-map admission is justified only
  by a research workload and requires an explicit representation and ownership contract.
- **Full `qgd_Circuit` gate parity** as a goal in itself (ADR-006); coverage grows in workload
  order.
- **Noisy circuit re-synthesis and noise-aware wide-circuit compilation** (ADR-001).
- **A general-purpose noisy simulator** competing with Aer, QuEST, or Qulacs on breadth or
  throughput.
- **Whole-register depolarizing** as a scientific workload.
- **Changing the state-vector partitioners' objective or behavior.**
- **Publication planning** — papers, abstracts, and slides consume closeouts and live outside
  `docs/specs/`.

## Validation

### PR-FAQ

**Press release (simulated).** *Budapest — SQUANDER's density-matrix track now lets researchers
train variational quantum circuits under realistic noise on an exact simulator that treats
noise channels as part of the circuit, not as walls between gates.* Researchers declare local
noise under explicit physical conventions and run circuits through a planner that fuses only
when a representation-aware record predicts that construction and application will pay. Every
advertised path is proven against a sequential reference at \(10^{-10}\), new channel surfaces
are cross-checked against Aer and analytical models, and unsupported work fails before touching
the state. Every counted claim regenerates with cost policy, tolerances, seeds, and revision
pinned. "We stopped asking whether noisy fusion is possible and started measuring, honestly,
when it pays," said the lead researcher.

**FAQ.**
- *Isn't Qiskit Aer enough?* Aer already supplies broad channel, GPU, superoperator, and fusion
  capabilities and remains the external reference. SQUANDER's research value must be shown in
  inspectable planning, strict route contracts, cost prediction, optimizer integration, and
  regenerable comparative evidence — not by claiming Aer lacks channel fusion.
- *Phase 3.1 found zero justified speedups. Why keep fusion?* It exposed the unresolved variables:
  representation expansion, construction, execution tier, parameter dependence, and reuse.
  A2 and A3 test selection and amortization separately, each with a kill criterion.
- *Why not count transformation steps alone?* Because equivalent Kraus representations can have
  different operator counts and a fused map can hide more arithmetic. QA-006 records structural,
  canonical, executed, amortized, memory, and timing evidence before accepting a benefit.
- *What happens when I give it a channel you don't support?* Mathematical validity does not
  imply advertised support. Unsupported channels reject at preflight; public custom-map
  admission needs its own contract.
- *How big can I go?* The delivered exact evidence is 4–10 qubits; 12 is a stretch hypothesis.
  Claims at those sizes are finite-size observations unless QA-011 establishes sensitivity to
  the target effect. Beyond the exact regime requires explicit escalation.
- *Does research readiness equal a trainability conclusion?* No. A calibrated, profiled,
  fusion-aware module enables CAP-006; QA-011 governs the later scientific conclusion.
- *What does a reviewer ask?* "Regenerate the counted bundle." QA-008 makes that a named-lane
  answer, including emitted per-case data rather than only an aggregate. State-vector users
  observe nothing (QA-009).
- *Why not trajectories or GPUs now?* Both answer a scale problem not yet hit inside the exact
  regime the trainability questions need; adding them first would blur the exactness anchor.

### Critique

- **Novelty narrowed.** Official Aer documentation and source show Kraus/superoperator-aware
  fusion; QuEST/Qulacs expose general channel maps. Therefore §3 claims transparent
  SQUANDER-integrated planning and evidence, not invention, uniqueness, or competitor absence.
- **Cost metric repaired.** Kraus decompositions are non-unique, so raw operator count cannot be
  the product bar. CAP-003/QA-006 retain logical reduction only inside a cost record that exposes
  canonical complexity, construction, executed work, memory, and paired timing. A one-sided
  bound tests the directional claim; \(\delta<1\) separates material benefit from no-regression.
- **Selection separated from reuse.** Reuse amortizes construction, not application, and current
  execution recomposes each bundle. QA-006 tests \(R=1\); QA-012 permits a reuse claim only at
  independently observed \(R_\mathrm{obs}\). Either hypothesis may fail independently.
- **Scientific endpoint strengthened.** One extra training run proves integration, not
  trainability. CAP-006/QA-011 require pre-registered multi-family evidence, uncertainty, and
  finite-size boundaries; the current source direction explicitly stops at readiness.
- **Physical claims separated.** QA-003 admits channels; QA-010 calibrates implementations.
  Kraus form makes CP structural, while trace-distance and Hermiticity tests remain regression
  witnesses. GAD uses \(p_\mathrm{exc}\) because it directly names the fixed-point excited
  population and maps unambiguously to Gibbs only after declaring the energy basis.
- **Ordering guard bounded.** Reverse-order separation is not universal for every state or weak
  parameter. QA-004 applies it only to pinned witnesses while universal correctness remains
  ordered agreement with the sequential reference.
- **Scope made feasible.** CAP-002 requires workload-justified channel growth, not generic
  custom-map breadth; representation ownership and all-path policy need downstream contracts.
- **Interop made comparable.** CAP-004 no longer mandates a C++ solution. QA-007 compares the
  same prebuilt evaluator in paired/interleaved trials where only the boundary differs,
  preventing language tier or setup work from masquerading as an algorithmic result.
- **Performance reproduction bounded.** QA-008 pins the environment and reproduces decisions
  against margins because exact timings are neither portable nor scientifically reproducible.
- **Legacy claims bounded.** Public exposure is not advertised support: legacy standalone
  channels remain non-claim-bearing until independently passing exactness and calibration.
- **Strictness reconciled.** Current parametric operations silently clamp. QA-005 now requires
  rejection or an explicit caller-selected transform, making the required behavior change
  visible rather than pretending it is already delivered.
- **Open numbers.** QA-002 energy, QA-004 sentinel, QA-006 reduction/\(\delta_\mathrm{eff}\)/
  memory ceiling, QA-007 overhead, QA-010 analytical tolerance, QA-012
  \(\delta_\mathrm{reuse}\), and the 12-qubit stretch remain `[confirm]`, owned by Z. Kégli
  and frozen in the first citing requirements.

### Stakeholder checkpoint

> Does this capture the durable intent of the track? Which capabilities, quality bars, or
> assumptions are wrong, missing, or too vague to guide a roadmap?

## Handoff

- **Next step:** invoke `create-product-roadmap` to decompose `CAP-001…007` and `QA-001…012` into
  outcome milestones, recording Phases 1–3.1 as Delivered and drawing the *Now* milestones from
  the current direction in `PHASE4_REQS.md`.
- **Current-state docs:** `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` already exist and were
  read as constraints; they are updated at each milestone close, not by this statement.
- **Open `[confirm]` values:** resolved by the product owner in the first `INITIAL_REQUIREMENTS.md`
  citing each id.

## Change log

- **v0.3 (2026-09-20)** — Adversarial scientific/feasibility revision: bounded novelty; representation-aware cost; separate selection/reuse; finite-size risk; separate admission/calibration/inference; current-direction trace separated from roadmap sequencing; rationale recorded in the critique.
- **v0.2 (2026-09-20)** — Added Kraus-count, unital/non-unital, contractivity, Hermiticity, reuse, and competitor-taxonomy refinements later corrected by v0.3.
- **v0.1 (2026-09-20)** — Initial draft from the archived plan/ADRs, Phase 1–3.1 record, CSCS 2026 paper/talk, and `PHASE4_REQS.md`; PR-FAQ and critique recorded.
