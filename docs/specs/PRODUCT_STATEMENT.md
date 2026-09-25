# Product statement — SQUANDER density-matrix track

> **Status:** draft v0.4, at stakeholder checkpoint · **Owner skill:** `create-product-statement` ·
> **Scope:** the durable North Star of the density-matrix (noisy mixed-state) track; the current
> emphasis is the direction recorded in `PHASE4_REQS.md` ·
> **Traces down to:** `ROADMAP.md` (`M#`) → `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) ·
> **Not:** sequencing (`ROADMAP.md`), architecture (`ARCHITECTURE_OVERVIEW.md`, ADRs), stack
> (`TECH_STACK.md`), or publication planning ·
> **Sources:** `docs/density_matrix_project/README.md`, `RESEARCH_ALIGNMENT.md`,
> `archive/planning/PLANNING.md` §2–3, §8, `archive/planning/ADRs.md` ADR-001…008, the CSCS 2026
> short paper and talk, `PHASE4_REQS.md`, and the 2026-09-23 review pilots (`ROADMAP.md` log).

## 1. Vision

For researchers training variational quantum circuits under realistic noise, the SQUANDER density-matrix track is an **exact open-system simulation module inside a circuit compiler/optimizer** that treats noise channels as first-class objects of partitioning, fusion, and controlled scientific comparison, so researchers obtain exact, attributable noisy energies and gradients for trainability research and, where a frozen cost record proves it pays, lower end-to-end cost — without ever trading exactness. Every advertised path is proven against a sequential reference under a frozen, auditable protocol, and unsupported input fails loudly instead of degrading silently.

*Decision test:* rules out approximate scaling, breadth-first channel or gate parity, convenience fallbacks, raw-throughput races, and noise-class comparisons at unmatched fidelity; rules in exactness contracts, noise-aware planning, certified channel admission, controlled unitality experiments, and regenerable evidence.

## 2. Customers & problem

**Primary persona — the noisy-training researcher.** A PhD or postdoctoral researcher (the SQUANDER group and its collaborators) who studies how variational circuits train under device-motivated local noise in the exact regime. Job to be done: obtain trustworthy noisy energies, gradients, and states thousands of times inside an optimizer loop, know which model and route produced each number, and defend those numbers to reviewers.

**Secondary personas.** The *SQUANDER maintainer*, who needs the noisy backend to stay additive to the state-vector path; the *external reproducer or reviewer*, who needs to regenerate a published number from a clean checkout.

**Core problem.** Exact mixed-state evolution costs \(2^{2n}\) per operator application, and in the delivered Phase 3 unitary-island baseline each noise channel interrupts fusion; Phase 3.1 found no justified speedup for its shipped full-dimension Kraus application and left open whether a same-complexity local application changes that verdict. Scientifically, trainability theory now diverges by noise class — unital noise predicts noise-induced barren plateaus, non-unital noise predicts limit sets and effectively shallow circuits — yet a finite-size test needs exact states, channels matched in fidelity, and depth sweeps whose variances fall below any sampling budget. Silent substitutions and incomplete cost accounting turn a benchmark into a guess; feasibility is not acceleration.

**Today.** The researcher pays the sequential cost per channel, cannot obtain density-backend gradients, compares noise models that differ in more than one property, discovers unsupported inputs by trial, and cannot fully attribute wall-clock time.

**In the future.** The researcher declares workload-justified local noise, compares noise classes at matched fidelity along depth, obtains exact gradients at a cost that scales with circuit size, uses a planner that sees channels and representation cost, bounds interop overhead, sees kernel cost attributed, and regenerates every counted claim. Unsupported work fails before mutating \(\rho\), or undergoes an explicit caller-selected transform.

## 3. Value proposition & differentiation

**Lead differentiator — inspectable noisy planning and evidence.** Noise is a first-class planner input: an eligible ordered gate+noise region may become one exact CPTP object, while eligibility, predicted cost, actual route, and unsupported boundaries remain observable. Strict proof cases validate the object; whole-workload cases evaluate it with per-partition attribution and no silent substitution.

**Supporting differentiators.**
- *One stack.* Compiling, optimizing, and noisily simulating a variational circuit share one
  circuit model and one optimizer loop; the noisy backend is selected, not bolted on.
- *Controlled noise-class experiments.* Channels matched in average gate fidelity compare noise
  classes at equal benchmark quality; GAD and its Pauli twirl isolate unitality exactly.
- *Evidence as a product output.* Counted cases, frozen tolerances, pre-registered thresholds,
  pinned seeds and revisions, and diagnosis-grounded closure when a hypothesis fails.

**Alternatives, honestly.** Aer supports general Kraus errors, superoperator simulation, and fusion for density-matrix/superoperator methods; QuEST and Qulacs accept general channel maps; other frameworks provide broader optimization or physics surfaces. SQUANDER does **not** claim to originate CPTP composition or noisy-operation fusion. Its target distinction combines integrated planning, explicit eligibility, predicted-versus-executed cost, route attribution, strict refusal, and regenerable baseline comparisons. Literature Positioning maintains a version-pinned, cited feature matrix; no priority claim is made without it. It also records the scientific frame the destination tests — unital noise-induced plateaus (Wang et al. 2021) versus non-unital limit sets and effective shallowness (Singkanipa and Lidar 2025; Mele et al. 2026); SQUANDER claims instruments for testing that frame, not its results.

## 4. Product capabilities (`CAP-*`)

Ids are stable: extend, never renumber. Each is an outcome a researcher can reach.

| Id | Capability (durable outcome) | Why / value | Success signal |
|----|------------------------------|-------------|----------------|
| **CAP-001** | **Exact noisy evolution.** A researcher evolves any supported ordered gate+noise circuit on an exact density matrix and obtains a trace-preserving, positive state carrying the ordered open-system semantics, on every shipped execution path. | Exactness disentangles noise effects from approximation error; it is the anchor for every later claim (ADR-005). | 100 % of counted correctness cases on every shipped path meet QA-001; exact-regime coverage at 4–10 qubits is maintained in every milestone closeout. |
| **CAP-002** | **Realistic noise inventory with controlled unitality.** A researcher expresses workload-justified local noise spanning unital dynamics (depolarizing, phase damping) and non-unital / finite-temperature dynamics (amplitude damping, generalized amplitude damping) under explicit parameter conventions, and compares noise classes at matched average gate fidelity; at fixed γ, GAD's excited-state population changes only the non-unital displacement \(\nu=\gamma(1-2p_\mathrm{exc})\), while its Pauli twirl and fidelity stay fixed. | Trainability theory diverges by noise class, and a comparison is interpretable only if it changes one property at a time; workload-driven growth avoids breadth without scientific purpose (ADR-004, ADR-006). | Each advertised channel meets QA-001 and QA-010 on every path that claims support; all other paths reject it at preflight; GAD's twirl and fidelity invariance hold as QA-010 witnesses; thermal relaxation \((T_1,T_2)\) is expressible only through a named, Aer-checked transform. |
| **CAP-003** | **Representation-aware selective fusion.** A researcher runs a noisy circuit through a planner that lowers total logical transformations only when a frozen cost record predicts no hidden representation, construction, application, or reuse penalty, with every partition's route attributed. | Raw Kraus count is non-unique and logical step reduction can hide greater work; scientific progress requires predicting and measuring the complete trade-off (ADR-002, ADR-003). | QA-006 selection at equal application complexity and, where claimed, QA-012 observed-reuse amortization are met; 0 evaluation-mode selections violate the no-regression rule; 100 % route attribution. |
| **CAP-004** | **Low-overhead iterative evaluation.** A researcher evaluates exact noisy energies thousands of times in a loop with language-boundary, construction, allocation, and dispatch costs separately measured and bounded, and per-operation kernel costs measured and attributed, rather than hidden inside total time. | Variational workflows amplify per-call overhead, and beyond a few qubits the kernel, not the language crossing, dominates; comparable tiers separate an algorithmic result from an implementation artifact. | QA-007 met on every energy entry that has an equal-work lower-boundary comparator; every performance closeout publishes the component profile, including nanoseconds per density-matrix entry per operation, and the comparison method. |
| **CAP-005** | **Exact noisy observables in the variational workflow.** A researcher selects the density backend in the same variational loop that runs state-vector and obtains exact \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\) energies for a supported ansatz and noise specification, with the support surface stated and unsupported requests refused. | Turns the simulator into a usable training backend; the frozen canonical workflow is the proof of usability (ADR-001). | Energies agree with Qiskit Aer per QA-002 at 4–10 qubits; every widening of the surface ships with a pinned negative test. |
| **CAP-006** | **Noisy variational research on the exact backend** *(destination)*. A researcher tests pre-registered hypotheses about optimization and trainability under realistic local noise, compares noise classes at matched fidelity along depth, and reports depth and noise-class effects, which the exact regime resolves, separately from width trends, which remain finite-size observations over the tested window. | This is the research outcome the module exists to serve; integration alone is not thesis-level scientific evidence (PLANNING §2.1). | QA-011 met across at least two independently motivated task families and two ansatz families, matched-fidelity unital and non-unital sweeps, and multiple seeded initializations, with effect sizes, uncertainty, and null outcomes regenerated; depends on CAP-004/005, and on CAP-008 for gradient-based arms only. |
| **CAP-007** | **Regenerable verification and calibration evidence.** A researcher or reviewer regenerates every counted claim — exactness, physical calibration, representation-aware cost, overhead, and scientific inference — from one named lane with tolerances, rates, seeds, revision, and claim boundary pinned. | Reproducibility is a first-class output (PLANNING §2); a claim that cannot be regenerated is not a claim. | QA-008 met: 100 % of counted rows regenerate; every claim row names its lane; no counted claim rests on a non-regenerable artifact. |
| **CAP-008** | **Exact differentiable noisy objective.** A researcher obtains exact gradients of the noisy energy with respect to every advertised circuit parameter under fixed, parameter-independent noise, inside the same optimizer loop, at a cost bounded by a small multiple of one energy evaluation; parameterizations outside the advertised gradient surface are refused. | Gradient variance and gradient-based training are the core observables of trainability research; parameter shift needs 2K energies per gradient, which is infeasible at 8 qubits and depth 16. | QA-013 met on every advertised parameterization at 4–10 qubits; gradient-based study arms run within their frozen budget. |

**Current-direction trace (`PHASE4_REQS.md`; sequencing belongs in `ROADMAP.md`).** Channel Expansion
→ CAP-002, QA-003/010; Partitioning and Fusion → CAP-003, QA-004…006/012; Interop Optimization →
CAP-004, QA-007; Verification and Calibration → CAP-007, QA-001…003/008/010; Literature Positioning
→ §3 (software and scientific frame). Its deliverable is readiness for CAP-006, not a trainability
conclusion. CAP-008 enables CAP-006 outside the Phase 4 tasks; the roadmap places it after Phase 4.

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
| **QA-006 Representation-aware selection** | When a pre-registered workload executes selective fusion with no unobserved reuse credit (\(R=1\)) | The selection policy is calibrated on one pre-registered workload set and judged on a held-out set; candidate and sequential arms apply each local operation at the same asymptotic complexity in comparable execution tiers; the execution-cost record (fields frozen in the first citing requirements) exposes representation, canonical complexity, construction, application, memory, and prediction error; on the held-out set, simultaneous one-sided 95 % upper bounds on \(T_\mathrm{candidate}/T_\mathrm{sequential}\) from paired, interleaved trials are \(\le1\) for every executed selection, with 0 selections above sequential cost or the frozen memory ceiling `[confirm]`; a positive claim additionally requires the bound \(\le\delta_\mathrm{eff}<1\) `[confirm]`, no greater executed work, and the structural-reduction bar frozen in requirements `[confirm]` | performance evidence pipeline; fast pytest |
| **QA-007 Interop overhead** | When the same prebuilt noisy-energy evaluator is called \(\ge1000\) warmed times through a public Python entry and an equivalent lower-boundary invocation where only the language crossing differs, at 4/6/8 qubits and frozen workloads (depth, schedule), on each advertised energy entry that has such an equal-work invocation | Freeze state reset, allocation, output materialization, batching, build, threading/affinity, and warm-up; paired/interleaved trials estimate \(O=(T_\mathrm{public}-T_\mathrm{lower})/T_\mathrm{public}\); report components, including per-operation kernel throughput, and uncertainty; the one-sided 95 % upper bound on \(O\) is \(\le10\,\%\) `[confirm]`; diagnosis may close a research milestone but leaves QA-007 unmet | performance evidence pipeline |
| **QA-008 Reproducibility** | When a counted claim is cited in a closeout or paper | Categorical classifications reproduce exactly and numerical residuals within frozen tolerance on 100 % of rows; performance decisions reproduce against their stated margin (not identical timings) under pinned CPU, compiler/flags, dependencies, threading/affinity, warm-up and sampling protocol; every bundle pins channel conventions, representation/cost policy, rates, seeds, revision, and claim boundary | benchmark tests; evidence pipelines |
| **QA-009 Non-interference** | When any density-track change lands | The upstream state-vector suite passes with 0 regressions and the density backend remains opt-in (default backend unchanged) | CI |
| **QA-010 Physical calibration** | When an advertised channel acts on pre-registered analytical states | Output populations/coherences match closed form within \(10^{-12}\) `[confirm]`; \(\lVert\rho-\rho^\dagger\rVert_F\le10^{-14}\); trace-distance contractivity holds within \(10^{-12}\) on declared pairs; with \(|0\rangle\) ground, \(\gamma,p_\mathrm{exc}\in[0,1]\), GAD satisfies \(\rho'_{11}=(1-\gamma)\rho_{11}+\gamma p_\mathrm{exc}\), has fixed point \(\mathrm{diag}(1-p_\mathrm{exc},p_\mathrm{exc})\), and after \(k\) applications maps \(\rho_{11}-p_\mathrm{exc}\) to \((1-\gamma)^k(\rho_{11}-p_\mathrm{exc})\); its Bloch translation is \(t_z=\gamma(1-2p_\mathrm{exc})\) with the \(|0\rangle\)-ground sign, and its Pauli-transfer diagonal — hence its twirl GAD(γ, ½) and \(F_\mathrm{avg}=\tfrac12+\mathrm{Tr}\,M/6\) — is independent of \(p_\mathrm{exc}\); “Gibbs” requires \(H=\Delta|1\rangle\langle1|\), \(\Delta>0\), \(\beta\ge0\), and \(p_\mathrm{exc}=(1+e^{\beta\Delta})^{-1}\); a time-based parameterization, with \(T_1,T_2>0\), \(t\ge0\), \(p_\mathrm{exc}\in[0,1]\), and \(T_2\le2T_1\), uses \(\gamma=1-e^{-t/T_1}\) and phase damping \(\lambda=1-e^{-2t/T_\varphi}\), \(1/T_\varphi=1/T_2-1/(2T_1)\) (\(\lambda=0\) at \(T_2=2T_1\)), and matches Aer's thermal-relaxation channel within \(10^{-12}\) `[confirm]` | fast pytest; correctness and Aer evidence |
| **QA-011 Scientific inference** | When a noisy-training or trainability conclusion is claimed | 100 % of claim-bearing studies freeze and archive, before counted data, the hypothesis, estimand, comparators, task/ansatz families, design grid, initialization ensemble, confidence level and multiplicity family, Monte Carlo precision target, decision margins, and finite-size boundary; confirmatory rows use seeds disjoint from any pilot; noise-class comparisons are made at matched average gate fidelity and report noise-weighted depth; results at \(n\le10\) are labelled finite-size, and a sensitivity analysis may only establish discrimination between pre-specified models over the tested window; report effect sizes, uncertainty, and null outcomes; raw results and analysis regenerate | study-specific evidence pipeline |
| **QA-012 Observed-reuse amortization** | When fusion benefit is attributed to reuse | Derive \(R_\mathrm{obs}\) independently from real optimizer traces; identify cache key and parameter dependencies; report hits/misses, build/apply time, memory, and break-even \(R\); unobserved \(R\) is forecast only; at \(R_\mathrm{obs}\), the paired one-sided 95 % upper bound on amortized \(T_\mathrm{candidate}/T_\mathrm{sequential}\) is \(\le\delta_\mathrm{reuse}<1\) `[confirm]` within the frozen memory ceiling | performance evidence pipeline |
| **QA-013 Gradient fidelity** | When a gradient of the noisy energy is reported for an advertised parameterization under fixed, parameter-independent noise | Every component agrees within \(10^{-10}\) `[confirm]` with an independent coordinate-aware two-term shift oracle (shift and coefficient from the gate's declared parameter multiplier) and within the frozen step-study tolerance of central finite differences; one gradient costs at most a frozen multiple of one energy evaluation `[confirm]`; requests outside the advertised gradient surface — initially shared-parameter, parametric-noise, and order-changing ones — raise a structured pre-mutation error; 100 % of counted gradients pass | fast pytest; correctness evidence pipeline |

## 6. Ubiquitous language (seed glossary)

| Term | Meaning |
|------|---------|
| **Density matrix** \(\rho\) | Exact mixed-state representation of an \(n\)-qubit register; \(2^n \times 2^n\), unit trace, positive semidefinite. |
| **Noisy circuit** | An ordered list of operations, each a unitary gate or a local CPTP channel on a qubit subset; order is semantically binding. |
| **Noise channel / CPTP map** | A completely-positive trace-preserving map; the only admissible form of noise. Kraus form establishes CP by construction; admission still validates shape, finiteness, support, and trace preservation. |
| **Local noise** | A channel on 1–2 qubits; the scientific workload. **Whole-register noise** is a labelled baseline or stress test only. |
| **Unitality; non-unital displacement \(\nu\)** | A channel is unital if it maps \(I\) to \(I\). For GAD, \(\nu=\gamma(1-2p_\mathrm{exc})\) is the Bloch translation along \(z\); \(\nu=0\) exactly at \(p_\mathrm{exc}=\tfrac12\). |
| **Pauli twirl; matched fidelity** | The twirl keeps a channel's Pauli-transfer diagonal and drops the rest; GAD(γ, ½) is the twirl of amplitude damping AD(γ), shared by every GAD(γ, \(p_\mathrm{exc}\)). Matched fidelity means equal average gate fidelity, \(F_\mathrm{avg}=\tfrac12+\mathrm{Tr}\,M/6\) for a qubit channel with Bloch linear part \(M\). |
| **Noise-weighted depth; dense schedule** | The control variable for noise-induced effects. Per qubit, exposure \(x_q=\sum_j-\log(1-\gamma_j)\) over its channels, with \(\gamma_j\) the amplitude-damping rate of equal average fidelity; studies order depths by a frozen summary (default: the mean \(\bar x\)) and report the per-qubit range; \(\gamma_\mathrm{eff}L\) is its small-rate form. A dense schedule places a channel on each qubit touched by each gate, or on each qubit per layer. |
| **Late-layer floor; effective trainable depth** | The depth-independent gradient or cost variance final layers retain under non-unital noise; the number of final layers whose gradient variance stays above a frozen threshold. |
| **Cost variance** | Variance of the energy over a declared parameter distribution; diagnoses plateaus as gradient variance does (Arrasmith et al. 2022) at one energy per draw. |
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
| **Exact regime** | Qubit counts at which dense exact simulation is routine: 4–10 today at shallow depth, 12 as stretch `[confirm]`; depth sweeps to hundreds of layers are routine at 4–6 qubits. |
| **Canonical noisy workflow** | The frozen supported variational workflow (generated HEA, ordered fixed local noise, XXZ Hamiltonian) used as the usability proof and continuity anchor; its three-channel schedule is not a trainability noise model. |

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
- Compare noise classes at matched average gate fidelity, using GAD(γ, ½) as the exact twirled
  reference where available, and report each design point's noise-weighted depth.
- Label every width result at \(n\le10\) finite-size; a QA-011 sensitivity analysis may only
  discriminate pre-specified models over the tested window. Depth trends may be fitted across depth.
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
- Claim quantum advantage from noisy trainability results: non-unital trainability coincides
  with effective shallowness and efficient classical estimation.
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
| A1a | In the exact regime, depth and noise-class effects on trainability are resolvable at matched fidelity. | **Partially evidenced (exploratory, non-claim-bearing).** 2026-09-23 SQUANDER pilot: amplitude-damping cost variance exceeded fidelity-matched depolarizing by 8×10³ at depth 16 on 6 qubits. | M5 (channel class) and M5B (isolated unitality) pre-registered screens; M13 study. | No relevant effect detected within the tested finite window → noise-class results are reported as null or inconclusive findings and the thesis emphasis moves to methods (CAP-003/004). |
| A1b | Over \(n\le10\), pre-specified width models of the target effect can be discriminated, even though every result remains finite-size. | **Unvalidated; discrimination at \(n\le10\) is doubtful.** ADR-005 selects the exact anchor but is not evidence that \(n\le12\) separates competing width laws. | Pre-registered sensitivity/power analysis under QA-011 (M13). | Models not discriminated → report finite-size observations only; escalate ADR-008 only if the thesis question requires width scaling. |
| A2 | Representation-aware selection finds workloads where exact fusion lowers cost after channel expansion is counted. | **Unvalidated; prior evidence bounded.** Phase 3.1 validly found no justified speedup (0/26) for its shipped path, which applies composed Kraus terms at full dimension (≈95 ms per 2-qubit term vs ≈15 ms per sequential operation at 10 qubits); whether same-complexity application changes the verdict is open. | QA-006 on held-out pre-registered families with same-complexity application, frozen cost/canonicalization policy, and comparable tiers. | No target family satisfies QA-006 → fusion leaves the critical path; exact channels, interop, observables, and evidence continue. |
| A3 | Real variational traces contain enough parameter-independent reuse to amortize fused-object construction. | **Unvalidated.** Repeated topology does not imply reusable values; the current path recomposes each execution. | QA-012 at independently observed \(R_\mathrm{obs}\) on real optimizer traces. | Observed reuse never reaches break-even, or useful motifs are parameter-dependent → drop the reuse claim and optimize per-evaluation execution only. |
| A8 | Gradient-based trainability studies are feasible because an exact gradient costs a small multiple of one energy, not 2K energies. | **Unvalidated; the alternative is infeasible.** Parameter shift at 8 qubits, depth 16 needs 1,344 energies (≈16 min) per gradient; the density path refuses gradients today. | QA-013 in the differentiable-objective milestone. | No method meets QA-013 within the cost bound → CAP-008 stays unmet, the product statement is revalidated, and studies use cost variance, sampled-coordinate gradients, and derivative-free optimizers. |
| A9 | Noise-induced effects depend on noise-weighted depth, so strong-noise, moderate-depth designs stand in for realistic rates at large depth. | **Split by the pilot (exploratory).** Unital decay collapsed onto \(\gamma L\) within ≈20 % at 4 qubits; the non-unital variance at the last sampled exposure rose with γ and was still falling at small γ, so no floor scaling is established. | Collapse check in M5; realistic-rate floors measured directly in M13 at depths of several \(1/\gamma_\mathrm{eff}\). | No collapse even for the unital reference → every realistic-rate point needs direct deep runs, restricting them to 4–6 qubits. |
| A4 | Language-boundary and dispatch overhead materially limits iterative evaluation. | **Likely false on the C++ energy entry.** Time is linear in operation count with a near-zero intercept, 12–15 ns per density-matrix entry per operation at 4–10 qubits; material costs are kernel throughput and the Python planner/runtime route. | QA-007 at 4/6/8 qubits on the C++ entry; time attribution on the planner/runtime route, which has no equal-work comparator. | The QA-007 upper bound on \(O\) is below \(5\,\%\) at every point → CAP-004 becomes a hold-the-line constraint, not an optimization theme. |
| A5 | The built-in local inventory through GAD covers the core studies. | **Partially validated.** Three channels carried delivered evidence; GAD is not delivered; GAD with phase damping can express thermal relaxation \((T_1,T_2)\) through a named transform. | QA-010 plus the first study's pre-registered noise specification. | A core study requires correlated, coherent, measurement, or trainable noise → revise CAP-002 through Ask-first rather than adding breadth speculatively. |
| A6 | The sequential `NoisyCircuit` executor is a trustworthy internal oracle. | **Validated only on the delivered bounded circuit-ordered three-channel slices**; legacy standalone channels are not claim-bearing. | QA-002/010 reset validation for every new channel/path. | Any counted disagreement beyond tolerance freezes downstream claims until resolved. |
| A7 | Researchers accept explicit rejection or transforms over silent convenience behavior. | **Unvalidated; current parametric paths clamp out-of-range rates.** | Migrate or expose that behavior, pin QA-005 negatives, and collect first external-reproducer feedback. | Users bypass the path because strict input is unusable → improve diagnostics or add explicit transforms, never restore silent behavior. |

**Decisions from elicitation and review (2026-09-20; revised 2026-09-23).** Whole-track North
Star with the current direction as emphasis; primary persona the noisy-training researcher;
inspectable planning is the differentiator; GAD is the next calibration anchor and the unitality
knob, while generic breadth is not a goal; logical count cannot stand alone; the current direction
ends at research readiness, while noisy training remains the destination. v0.4: science is
co-equal with cost; depth and noise class are the exact regime's strength, width laws its risk;
exact gradients enter as CAP-008 behind a roadmap gate.

## 9. Out of scope / non-goals

- **Approximate scaling** — stochastic trajectories, MPDO or tensor-network mixed states — until
  the exact regime's limit is hit and escalated (ADR-008).
- **GPU kernels** — a separate track; may be integrated per milestone, never a dependency of the
  exactness contract.
- **Noise outside the local CPTP inventory** — readout/measurement/shot noise, correlated
  multi-qubit channels, coherent over-rotation, and trainable or time-varying rates. Fixed
  per-insertion rates may differ by gate type or position and may be swept between evaluations.
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
study how variational circuits train under realistic noise on an exact simulator that treats
noise channels as part of the circuit, not as walls between gates.* Researchers compare noise
classes at matched fidelity along circuit depth, obtain exact energies and gradients, and run
circuits through a planner that fuses only when a representation-aware record predicts that
construction and application will pay. Every advertised path is proven against a sequential
reference at \(10^{-10}\), new channels are cross-checked against Aer and analytical models,
and unsupported work fails before touching the state. Every counted claim regenerates with
policy, tolerances, seeds, and revision pinned. "We can now say which property of the noise
decides trainability, and measure honestly when fusion pays," said the lead researcher.

**FAQ.**
- *Isn't Qiskit Aer enough?* Aer remains the external reference and supplies broad channel, GPU,
  and fusion capabilities. SQUANDER's value lies in inspectable planning, strict routes, cost
  prediction, optimizer integration, controlled noise-class designs, and regenerable evidence.
- *Phase 3.1 found zero justified speedups. Why keep fusion?* The verdict holds for its shipped
  path, which applies composed Kraus terms at full dimension; whether same-complexity application
  changes it is open. A2 and A3 test selection and amortization separately, each with a kill criterion.
- *Why compare noise classes at matched fidelity?* Randomized benchmarking reports only average
  fidelity. At equal fidelity, GAD and its Pauli twirl differ only in unitality, so a
  trainability difference between them is attributable to unitality alone.
- *How big can I go?* Delivered exact evidence covers 4–10 qubits (12 is a stretch). Noise acts
  through depth, which exact simulation does not limit, so depth and noise-class questions are
  answerable here; width results stay finite-size, and QA-011 can only discriminate
  pre-specified models over the tested window.
- *Why exact gradients?* Gradient variance and gradient-based training are the core trainability
  observables; parameter shift needs 2K energies per gradient, 1,344 at 8 qubits and depth 16.
- *Does research readiness equal a trainability conclusion?* No. A calibrated, profiled,
  fusion-aware module enables CAP-006; QA-011 governs the later scientific conclusion.
- *What does a reviewer ask?* "Regenerate the counted bundle." QA-008 makes that a named-lane
  answer with per-case data. State-vector users observe nothing (QA-009).
- *Why not trajectories or GPUs now?* Both answer a scale problem the trainability questions
  have not hit; adding them first would blur the exactness anchor.

### Critique

**v0.3 (retained).** Novelty narrowed to integrated planning and evidence, since Aer, QuEST, and
Qulacs already fuse or accept general maps. Raw Kraus count replaced by a cost record with
canonical complexity, construction, executed work, memory, and paired timing. Selection
(QA-006, \(R=1\)) separated from observed reuse (QA-012). Scientific endpoint strengthened to
pre-registered, multi-family evidence. Admission (QA-003) separated from calibration (QA-010);
GAD uses \(p_\mathrm{exc}\), mapped to Gibbs only after declaring the energy basis. Reverse-order
sentinels bounded to pinned witnesses; interop compared at equal work; performance reproduction
bounded to decisions against margins; legacy channels non-claim-bearing; clamping made a visible
required change under QA-005.

**v0.4 (2026-09-23 scientific review).**
- *A negative fusion result could falsify the vision.* Science and cost are now co-equal; the
  vision also rules out noise-class claims at unmatched fidelity and gradients without QA-013.
- *A1 conflated depth with width.* Split into A1a (depth and noise class, resolvable) and A1b
  (width laws, likely finite-size-limited); only A1b can trigger ADR-008 escalation.
- *A2's evidence is bounded.* Phase 3.1's verdict holds for its full-dimension path; QA-006 now
  requires same-complexity application and a held-out validation set.
- *Noise comparisons confounded fidelity with unitality.* CAP-002, QA-010, and QA-011 now require
  matched fidelity and name GAD(γ, ½) as the exact twirl of amplitude damping.
- *No gradient contract.* CAP-008 and QA-013 added; placement after Phase 4 is a roadmap decision.
- *A4 is likely false on the C++ entry.* CAP-004 and QA-007 now report kernel throughput.
- *Physical time.* QA-010 pins the thermal-relaxation mapping; the phase-damping code comment
  \(\lambda=1-e^{-t/T_2}\) contradicts its Kraus form and must not parameterize studies.
- *Realistic rates.* A9 records that strong-noise proxies hold for unital decay only.
- *Critic pass 1.* QA-011 now freezes error levels, multiplicity, precision, and fresh-seed
  confirmation; QA-006 validates on a held-out set; kernels are measured, not bounded; CAP-008 gates
  only gradient-based arms; CAP-006 needs two task and two ansatz families; A9 asserts no floor law.
- *Critic pass 2.* Noise-weighted depth is defined per qubit, since dense schedules expose edge
  qubits half as much; every width result at \(n\le10\) stays finite-size; A4's kill uses the
  QA-007 statistic; QA-013 refuses what lies outside the advertised gradient surface, so tied
  parameters can be added later.
- *Open numbers:* QA-002, QA-004, QA-006, QA-007, QA-010, QA-012, and QA-013 thresholds and the
  12-qubit stretch remain `[confirm]`, owned by Z. Kégli and frozen in the first citing requirements.

### Stakeholder checkpoint

> Does this capture the durable intent of the track? Which capabilities, quality bars, or
> assumptions are wrong, missing, or too vague to guide a roadmap?

## Handoff

- **Next step:** `create-product-roadmap` revalidates `ROADMAP.md` against this v0.4 (v0.7, done
  in the same change); M4 `canonical-attributed-energy` then goes to `create-initreq-for-sdd`
  unchanged. Confirming v0.4 satisfies the product-contract condition of the roadmap's gate RG-2.
- **Current-state docs:** `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` already exist and were
  read as constraints; they are updated at each milestone close, not by this statement.
- **Open `[confirm]` values:** resolved by the product owner in the first `INITIAL_REQUIREMENTS.md`
  citing each id.

## Change log

- **v0.4 (2026-09-23)** — Scientific-review revision with two adversarial critique passes: science co-equal with cost in the vision; per-qubit noise-weighted depth; A1 split into depth/noise class (A1a) and width models (A1b); A2's prior evidence bounded to Phase 3.1's shipped path; unitality at matched fidelity made the CAP-002 axis with GAD twirl, fidelity, and thermal-relaxation witnesses (QA-010); QA-011 made a frozen-protocol fitness function with fresh-seed confirmation; QA-006 held-out and same-complexity; kernel throughput measured in CAP-004/QA-007; CAP-008 and QA-013 for exact gradients; CAP-006 needs two task and two ansatz families; A8–A9 added; advantage-claim guardrail; rates may vary by position.
- **v0.3 (2026-09-20)** — Adversarial scientific/feasibility revision: bounded novelty; representation-aware cost; separate selection/reuse; finite-size risk; separate admission/calibration/inference; current-direction trace separated from roadmap sequencing; rationale recorded in the critique.
- **v0.2 (2026-09-20)** — Added Kraus-count, unital/non-unital, contractivity, Hermiticity, reuse, and competitor-taxonomy refinements later corrected by v0.3.
- **v0.1 (2026-09-20)** — Initial draft from the archived plan/ADRs, Phase 1–3.1 record, CSCS 2026 paper/talk, and `PHASE4_REQS.md`; PR-FAQ and critique recorded.
