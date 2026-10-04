# Product statement — SQUANDER density-matrix track

> **Status:** draft v0.5, aligned for roadmap revalidation · **Owner skill:** `create-product-statement` ·
> **Scope:** the density-matrix track's durable North Star, aligned to the locked Fall 2026 outcome ·
> **Traces down to:** `ROADMAP.md` (`M#`) → `milestones/<slug>/INITIAL_REQUIREMENTS.md` (`REQ-*`) ·
> **Not:** sequencing (`ROADMAP.md`), architecture (`ARCHITECTURE_OVERVIEW.md`, ADRs), stack
> (`TECH_STACK.md`), or publication planning ·
> **Sources:** the four locked Fall briefs and the frozen Phase 1–3.1 record cited below.
## 1. Vision

For researchers preparing exact studies of variational quantum circuits under realistic noise, the SQUANDER density-matrix track is a **verified, fusion-enabled open-system module inside a circuit compiler/optimizer** that treats noise channels as first-class objects of partitioning, execution, calibration, and controlled comparison. It is ready for later noisy VQA loops when every advertised route is exact and attributable; module readiness does not itself claim a VQA campaign, a trainability conclusion, or a full-scale VQA run.

*Decision test:* rules out approximate scaling, breadth-first channel or gate parity, convenience fallbacks, raw-throughput races, committed semester speedups, and noise-class comparisons at unmatched fidelity; rules in exactness contracts, workload-driven channel expansion, noise-aware planning, certified channel admission, calibrated interop, honest literature positioning, and regenerable evidence.
## 2. Customers & problem

**Primary persona — the noisy-algorithm researcher.** A PhD or postdoctoral researcher (the SQUANDER group and its collaborators) preparing later variational studies under device-motivated local noise in the exact regime. Job to be done now: obtain trustworthy noisy states and observables from a calibrated module, know which model and route produced each number, and defend those numbers to reviewers before placing the module in an optimizer loop.

**Secondary personas.** The *SQUANDER maintainer*, who needs the noisy backend to stay additive to the state-vector path; the *external reproducer or reviewer*, who needs to regenerate a published number from a clean checkout.

**Core problem.** Exact mixed-state evolution costs \(2^{2n}\) per operator application, and in the delivered Phase 3 unitary-island baseline each noise channel interrupts fusion; Phase 3.1 found no justified speedup for its shipped full-dimension Kraus application and left open whether a same-complexity local application changes that verdict. Scientifically, trainability theory now diverges by noise class — unital noise predicts noise-induced barren plateaus, non-unital noise predicts limit sets and effectively shallow circuits — yet a finite-size test needs exact states, channels matched in fidelity, and depth sweeps whose variances fall below any sampling budget. Silent substitutions and incomplete cost accounting turn a benchmark into a guess; feasibility is not acceleration.

**Today.** The researcher pays the sequential cost per channel, cannot obtain density-backend gradients, compares noise models that differ in more than one property, discovers unsupported inputs by trial, and cannot fully attribute wall-clock time.

**In the future.** The researcher declares workload-justified local noise beyond the delivered depolarizing, amplitude-damping, and phase-damping set, uses a planner that sees channels and representation cost, bounds C++/Python overhead, sees kernel cost and routes attributed, and regenerates every counted claim. Later milestones may add exact gradients and noisy VQA studies; those are consumers of module readiness, not evidence that the Fall module already establishes trainability.

## 3. Value proposition & differentiation

**Lead differentiator — inspectable noisy planning and evidence.** Noise is a first-class planner input: an eligible ordered gate+noise region may become one exact CPTP object, while eligibility, predicted cost, actual route, and unsupported boundaries remain observable. Strict proof cases validate the object; whole-workload cases evaluate it with per-partition attribution and no silent substitution.

**Supporting differentiators.**
- *One stack.* Compiling, optimizing, and noisily simulating a variational circuit share one
  circuit model and one optimizer loop; the noisy backend is selected, not bolted on.
- *Controlled noise-class experiments.* Channels matched in average gate fidelity compare noise
  classes at equal benchmark quality; GAD and its Pauli twirl isolate unitality exactly.
- *Evidence as a product output.* Counted cases, frozen tolerances, pre-registered thresholds,
  pinned seeds and revisions, and diagnosis-grounded closure when a hypothesis fails.

**Alternatives, honestly.** Aer supports general Kraus errors, superoperator simulation, and fusion for density-matrix/superoperator methods; QuEST and Qulacs accept general channel maps; other frameworks provide broader optimization or physics surfaces. The closest neighbor is TANQ-Sim (Li et al., arXiv:2404.13184), classified **PARTIAL / engine-fusion**: its Kraus-to-Liouville engine merges consecutive same-support C1/C2 operations for throughput. That is not a full scoop, but neither consecutive C1/C2 Liouville merge nor noisy-operation fusion is ours to originate. SQUANDER does **not** claim priority for CPTP composition, noisy-operation fusion, or a “first to fuse” result. Its bounded distinction combines Kraus-primary ordered motifs, explicit strict/hybrid eligibility, predicted-versus-executed cost, route attribution, strict refusal, and regenerable comparisons. Literature Positioning keeps this claim version-pinned and cited; there is no verified CSCS arXiv or DOI to cite. It records the scientific frame that later work may test, while SQUANDER claims instruments, not trainability results.

## 4. Product capabilities (`CAP-*`)

Ids are stable: extend, never renumber. Each is an outcome a researcher can reach.

| Id | Capability (durable outcome) | Why / value | Success signal |
|----|------------------------------|-------------|----------------|
| **CAP-001** | **Exact noisy evolution.** A researcher evolves any supported ordered gate+noise circuit on an exact density matrix and obtains a trace-preserving, positive state carrying the ordered open-system semantics, on every shipped execution path. | Exactness disentangles noise effects from approximation error; it is the anchor for every later claim (ADR-005). | 100 % of counted correctness cases on every shipped path meet QA-001; exact-regime coverage at 4–10 qubits is maintained in every milestone closeout. |
| **CAP-002** | **Workload-driven channel expansion.** A researcher expresses calibrated local CPTP noise beyond the already delivered local depolarizing, amplitude-damping, and phase-damping channels, beginning where justified with generalized amplitude damping (GAD), under explicit parameter conventions and matched-fidelity controls. | Channel growth must serve a scientific comparison rather than repackage delivered work; GAD adds a finite-temperature/non-unital control without pretending amplitude damping is new. | Every newly advertised channel meets QA-001, QA-002, QA-003, and QA-010 on every path that claims support; all other paths reject it at preflight; GAD's twirl and fidelity invariance hold as QA-010 witnesses. |
| **CAP-003** | **Noise-aware partitioning and channel-native fusion.** A researcher runs a noisy circuit through a planner that can route an eligible motif as one exact channel-native object, while exposing representation, construction, application, reuse, skips, and the executed route. | Logical step reduction can hide greater work; scientific progress requires exact routing and honest comparison rather than treating feasibility as acceleration (ADR-002, ADR-003). | Every routed object meets QA-001/004/005; 100 % route attribution; any later justified cell meets QA-006, and reuse claims meet QA-012. |
| **CAP-004** | **Low-overhead iterative evaluation.** A researcher evaluates exact noisy energies thousands of times in a loop with language-boundary, construction, allocation, and dispatch costs separately measured and bounded, and per-operation kernel costs measured and attributed, rather than hidden inside total time. | Variational workflows amplify per-call overhead, and beyond a few qubits the kernel, not the language crossing, dominates; comparable tiers separate an algorithmic result from an implementation artifact. | QA-007 met on every energy entry that has an equal-work lower-boundary comparator; every performance closeout publishes the component profile, including nanoseconds per density-matrix entry per operation, and the comparison method. |
| **CAP-005** | **Module-ready noisy observables.** A researcher obtains exact \(\mathrm{Re}\,\mathrm{Tr}(H\rho)\) energies for a supported ansatz and noise specification through the streamlined public interface, with the support surface stated and unsupported requests refused. | Establishes a verified integration boundary for later noisy VQA loops without turning module delivery into a training campaign. | Energies agree with Qiskit Aer per QA-002 at 4–10 qubits; every widening ships with a pinned negative test; the readiness closeout makes no trainability inference. |
| **CAP-006** | **Noisy variational research on the exact backend** *(destination)*. A researcher tests pre-registered hypotheses about optimization and trainability under realistic local noise, compares noise classes at matched fidelity along depth, and reports depth and noise-class effects, which the exact regime resolves, separately from width trends, which remain finite-size observations over the tested window. | This is the research outcome the module exists to serve; integration alone is not thesis-level scientific evidence (PLANNING §2.1). | QA-011 met across at least two independently motivated task families and two ansatz families, matched-fidelity unital and non-unital sweeps, and multiple seeded initializations, with effect sizes, uncertainty, and null outcomes regenerated; depends on CAP-004/005, and on CAP-008 for gradient-based arms only. |
| **CAP-007** | **Regenerable verification and calibration evidence.** A researcher or reviewer regenerates every counted claim — exactness, physical calibration, representation-aware cost, overhead, and scientific inference — from one named lane with tolerances, rates, seeds, revision, and claim boundary pinned. | Reproducibility is a first-class output (PLANNING §2); a claim that cannot be regenerated is not a claim. | QA-008 met: 100 % of counted rows regenerate; every claim row names its lane; no counted claim rests on a non-regenerable artifact. |
| **CAP-008** | **Exact differentiable noisy objective.** A researcher obtains exact gradients of the noisy energy with respect to every advertised circuit parameter under fixed, parameter-independent noise, inside the same optimizer loop, at a cost bounded by a small multiple of one energy evaluation; parameterizations outside the advertised gradient surface are refused. | Gradient variance and gradient-based training are the core observables of trainability research; parameter shift needs 2K energies per gradient, which is infeasible at 8 qubits and depth 16. | QA-013 met on every advertised parameterization at 4–10 qubits; gradient-based study arms run within their frozen budget. |

**Fall registration alignment.** The five tasks are: channel expansion beyond the delivered local
depolarizing, amplitude-damping, and phase-damping channels (CAP-002); noise-aware partitioning and
channel-native fusion (CAP-003); C++/Python interop profiling and reduction (CAP-004); verification
and calibration (CAP-007); and literature positioning (§3). The deliverable is a verified,
fusion-enabled noise module ready for later noisy VQA loops—not a VQA campaign, a full-scale run,
or a trainability conclusion. A written plan exists for how a future hybrid speedup of at least
1.2× could be earned; it is not a promised Fall result.

**Fall keep-list and deferral boundary.** The keep-list runs through channel expansion (M-F1a → M-F5a → M-F1b → M-F2 → M-F-CE), then defers M-F3 (`reuse_heavy_layered_v0`, the primary two-qubit design) before strict M-F4 (`phase31_bounded_mixed_motif_s3_v0`). M-F4 is a
paper-operations-only, strict-only stretch after M-F3, not a hybrid widen, keep-list item, or
default workload; designing it does not require a prior justified \(\ge1.2\times\) two-qubit cell.
QA-006 applies to the primary two-qubit matrix, not as M-F4's success test. Any later speedup-only
measurement on the stretch surface is optional, is not M-F4's success test, does not make it a
hybrid workload, does not replace the primary design, and is not a promised result. Q3 stays
parked; E7 is dropped; Fall is not redirected to M4.

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
| **QA-006 Channel-native justification** | When a later pre-registered case is classified as a justified channel-native cell | At least one partition genuinely executes the channel-native route (a skip or baseline route does not count); the case passes the standing exactness protocol against the sequential oracle; and median-of-three hybrid wall-clock speedup is \(\ge1.2\times\) versus the **Phase-3 fused baseline**. The two-qubit primary arm uses a new matrix, `reuse_heavy_layered_v0`; the frozen 26-case matrix is not reused or relabelled. A zero-justified-cell result is valid and rejects the hypothesis for that family; this rule is not a cost ratio \(\le1\), a paired-bound rule, or a comparison against the sequential executor. | performance evidence pipeline plus correctness evidence pipeline |
| **QA-007 Interop overhead** | When the same prebuilt noisy-energy evaluator is called \(\ge1000\) warmed times through a public Python entry and an equivalent lower-boundary invocation where only the language crossing differs, at 4/6/8 qubits and frozen workloads (depth, schedule), on each advertised energy entry that has such an equal-work invocation | Freeze state reset, allocation, output materialization, batching, build, threading/affinity, and warm-up; paired/interleaved trials estimate \(O=(T_\mathrm{public}-T_\mathrm{lower})/T_\mathrm{public}\); report components, including per-operation kernel throughput, and uncertainty; the one-sided 95 % upper bound on \(O\) is \(\le10\,\%\) `[confirm]`; diagnosis may close a research milestone but leaves QA-007 unmet | performance evidence pipeline |
| **QA-008 Reproducibility** | When a counted claim is cited in a closeout or paper | Categorical classifications reproduce exactly and numerical residuals within frozen tolerance on 100 % of rows; performance decisions reproduce against their stated margin (not identical timings) under pinned CPU, compiler/flags, dependencies, threading/affinity, warm-up and sampling protocol; every bundle pins channel conventions, representation/cost policy, rates, seeds, revision, and claim boundary | benchmark tests; evidence pipelines |
| **QA-009 Non-interference** | When any density-track change lands | The upstream state-vector suite passes with 0 regressions and the density backend remains opt-in (default backend unchanged) | CI |
| **QA-010 Physical calibration** | When an advertised channel acts on pre-registered analytical states | Output populations/coherences match closed form within \(10^{-12}\) `[confirm]`; \(\lVert\rho-\rho^\dagger\rVert_F\le10^{-14}\); trace-distance contractivity holds within \(10^{-12}\) on declared pairs; with \(|0\rangle\) ground, \(\gamma,p_\mathrm{exc}\in[0,1]\), GAD satisfies \(\rho'_{11}=(1-\gamma)\rho_{11}+\gamma p_\mathrm{exc}\), has fixed point \(\mathrm{diag}(1-p_\mathrm{exc},p_\mathrm{exc})\), and after \(k\) applications maps \(\rho_{11}-p_\mathrm{exc}\) to \((1-\gamma)^k(\rho_{11}-p_\mathrm{exc})\); its Bloch translation is \(t_z=\gamma(1-2p_\mathrm{exc})\) with the \(|0\rangle\)-ground sign, and its Pauli-transfer diagonal — hence its twirl GAD(γ, ½) and \(F_\mathrm{avg}=\tfrac12+\mathrm{Tr}\,M/6\) — is independent of \(p_\mathrm{exc}\); “Gibbs” requires \(H=\Delta|1\rangle\langle1|\), \(\Delta>0\), \(\beta\ge0\), and \(p_\mathrm{exc}=(1+e^{\beta\Delta})^{-1}\); a time-based parameterization, with \(T_1,T_2>0\), \(t\ge0\), \(p_\mathrm{exc}\in[0,1]\), and \(T_2\le2T_1\), uses \(\gamma=1-e^{-t/T_1}\) and phase damping \(\lambda=1-e^{-2t/T_\varphi}\), \(1/T_\varphi=1/T_2-1/(2T_1)\) (\(\lambda=0\) at \(T_2=2T_1\)), and matches Aer's thermal-relaxation channel within \(10^{-12}\) `[confirm]` | fast pytest; correctness and Aer evidence |
| **QA-011 Scientific inference** | When a noisy-training or trainability conclusion is claimed | 100 % of claim-bearing studies freeze and archive, before counted data, the hypothesis, estimand, comparators, task/ansatz families, design grid, initialization ensemble, confidence level and multiplicity family, Monte Carlo precision target, decision margins, and finite-size boundary; confirmatory rows use seeds disjoint from any pilot; noise-class comparisons are made at matched average gate fidelity and report noise-weighted depth; results at \(n\le10\) are labelled finite-size, and a sensitivity analysis may only establish discrimination between pre-specified models over the tested window; report effect sizes, uncertainty, and null outcomes; raw results and analysis regenerate | study-specific evidence pipeline |
| **QA-012 Observed-reuse amortization** | When fusion benefit is attributed to reuse | Derive \(R_\mathrm{obs}\) independently from real traces; identify cache key and parameter dependencies; report hits/misses, build/apply time, memory, and break-even \(R\); unobserved \(R\) is forecast only. Reuse evidence may explain a result but does not replace QA-006: a justified cell still requires genuine routing, exactness, and hybrid speedup \(\ge1.2\times\) versus the Phase-3 fused baseline. | performance evidence pipeline |
| **QA-013 Gradient fidelity** | When a gradient of the noisy energy is reported for an advertised parameterization under fixed, parameter-independent noise | Every component agrees within \(10^{-10}\) `[confirm]` with an independent coordinate-aware two-term shift oracle (shift and coefficient from the gate's declared parameter multiplier) and within the frozen step-study tolerance of central finite differences; one gradient costs at most a frozen multiple of one energy evaluation `[confirm]`; requests outside the advertised gradient surface — initially shared-parameter, parametric-noise, and order-changing ones — raise a structured pre-mutation error; 100 % of counted gradients pass | fast pytest; correctness evidence pipeline |

## 6. Ubiquitous language (seed glossary)

| Term | Meaning |
|------|---------|
| **Density matrix** \(\rho\) | Exact mixed-state representation of an \(n\)-qubit register; \(2^n \times 2^n\), unit trace, positive semidefinite. |
| **Noisy circuit** | An ordered list of operations, each a unitary gate or a local CPTP channel on a qubit subset; order is semantically binding. |
| **Noise channel / CPTP map** | A completely-positive trace-preserving map; the only admissible form of noise. Kraus form establishes CP by construction; admission still validates shape, finiteness, support, and trace preservation. |
| **Local noise / scientific workload** | Workload-justified CPTP noise inserted on bounded qubit subsets. The delivered and primary Fall surface is local 1–2-qubit noise; strict \(|S_M|=3\) on `phase31_bounded_mixed_motif_s3_v0` is a separate paper-operations-only stretch, not the default workload. **Whole-register noise** is a labelled baseline or stress test only. |
| **Unitality; non-unital displacement \(\nu\)** | A channel is unital if it maps \(I\) to \(I\). For GAD, \(\nu=\gamma(1-2p_\mathrm{exc})\) is the Bloch translation along \(z\); \(\nu=0\) exactly at \(p_\mathrm{exc}=\tfrac12\). |
| **Pauli twirl; matched fidelity** | The twirl keeps a channel's Pauli-transfer diagonal and drops the rest; GAD(γ, ½) is the twirl of amplitude damping AD(γ), shared by every GAD(γ, \(p_\mathrm{exc}\)). Matched fidelity means equal average gate fidelity, \(F_\mathrm{avg}=\tfrac12+\mathrm{Tr}\,M/6\) for a qubit channel with Bloch linear part \(M\). |
| **Noise-weighted depth; dense schedule** | The control variable for noise-induced effects. Per qubit, exposure \(x_q=\sum_j-\log(1-\gamma_j)\) over its channels, with \(\gamma_j\) the amplitude-damping rate of equal average fidelity; studies order depths by a frozen summary (default: the mean \(\bar x\)) and report the per-qubit range; \(\gamma_\mathrm{eff}L\) is its small-rate form. A dense schedule places a channel on each qubit touched by each gate, or on each qubit per layer. |
| **Late-layer floor; effective trainable depth** | The depth-independent gradient or cost variance final layers retain under non-unital noise; the number of final layers whose gradient variance stays above a frozen threshold. |
| **Cost variance** | Variance of the energy over a declared parameter distribution; diagnoses plateaus as gradient variance does (Arrasmith et al. 2022) at one energy per draw. |
| **Motif** | A contiguous sub-sequence of a noisy circuit with bounded support that contains at least one channel. |
| **Fused channel object** | The exact CPTP map of a motif, composed in operation order. An ordered **Kraus bundle is primary**; Choi and Liouville forms are correctness/completeness witnesses, not the primary fused object. |
| **Channel-native fusion** | Executing a motif as its fused channel object instead of step by step. **Unitary-island fusion** fuses gate-only runs between channels. |
| **Logical transformation** | One ordered gate or complete channel application in the circuit semantics. Its count describes structural contraction but is **not** a cost metric by itself because equivalent channel representations can have different execution work. |
| **Canonical channel complexity** | Diagnostic numerical Choi rank under a frozen rank policy; representation-invariant under that policy and never permission to truncate an exact execution. |
| **Execution-cost record** | The declared representation/support, logical transformations, construction/composition work, executed full-state work, memory, measured time, parameter dependencies, reuse horizon, and predicted-versus-executed route. |
| **Sequential reference** | The unfused, step-by-step `NoisyCircuit` execution; the internal exact oracle. **External reference:** Qiskit Aer. **Analytical reference:** a closed-form open-system model. |
| **Strict / hybrid** | Strict: every partition must be eligible or the run hard-fails (proof mode). Hybrid: an eligible partition may take the channel-native route when the evaluation policy selects it; the policy may instead make a predicted-cost skip to a labelled baseline. Ineligible or skipped partitions take the supported baseline, and every partition carries a **route label** (evaluation mode). |
| **Support surface** | The explicitly supported gates, channels, circuit sources, and modes; everything outside raises a structured error. |
| **Advertised support** | A surface named in a claim-bearing support matrix, not merely exposed by an API. The circuit-ordered operation model is authoritative; legacy standalone `NoiseChannel` APIs are excluded unless independently passing QA-001/002/010. |
| **Counted case** | A workload whose result is a claim-bearing row in an evidence bundle; a **continuity anchor** ties a new bundle to an earlier one. |
| **Frozen 26-case matrix** | Historical Phase-3.1 speedup-only evidence: 17 baseline-sufficient, 9 genuinely channel-native but not yet justified, and 0 justified at hybrid speedup \(\ge1.2\times\). It is not the baseline or matrix for this phase and must not be mixed with the separate Phase-3 millisecond table. |
| **Phase-3 fused baseline** | The comparator for any later QA-006 hybrid speedup claim. It is distinct from the sequential exact oracle, which establishes correctness rather than the justification denominator. |
| **Frozen tolerance / pre-registered threshold** | A numerical policy or decision bar fixed before evidence is generated; changed only through the Ask-first guardrail. |
| **Diagnosis-grounded closure** | Closing a performance hypothesis by evidence of where time or steps go, when the pre-registered arm is not met. |
| **Reuse-heavy workload** | A training-relevant circuit trace in which measured parameter-independent channel components or motifs recur enough to amortize construction; reuse is observed, not assumed from repeated topology. |
| **Exact regime** | Qubit counts at which dense exact simulation is routine: 4–10 today at shallow depth, 12 as stretch `[confirm]`; depth sweeps to hundreds of layers are routine at 4–6 qubits. |
| **Canonical noisy workflow** | The frozen supported variational workflow (generated HEA, ordered fixed local noise, XXZ Hamiltonian) used as the usability proof and continuity anchor; its three-channel schedule is not a trainability noise model. |

## 7. Guardrails & constraints

**Always**
- Validate every new or changed execution path against the sequential reference within the
  frozen tolerances *before* any performance statement about it.
- Argue correctness and performance separately; pre-register every decision threshold before
  the matrix runs. A performance statement reports the baseline trio (sequential, existing
  fused, candidate), execution-cost record, comparable execution tiers, uncertainty, and route
  attribution; logical transformation count never stands alone.
- Use workload-justified local noise as the default scientific workload; keep the named strict
  \(|S_M|=3\) arm explicitly marked as stretch and label whole-register noise as baseline.
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
- Introducing an approximate method (stochastic trajectories, MPDO/tensor networks), AVX work,
  or any GPU path. AVX and GPU are outside this semester; GPU is neither an active path nor an
  exactness dependency.
- Changing the semantics of the sequential reference itself, or superseding a program ADR.
- Retiring or superseding a `CAP-*` / `QA-*`.

**Never**
- Substitute an execution route silently. A labelled hybrid baseline route is the named evaluation mode, not a silent fallback; silent substitution onto the state-vector path or a different
  representation remains forbidden, and any unlabelled substitution raises a structured error.
- Clamp or otherwise transform an out-of-domain scientific parameter without an explicit,
  caller-selected transform recorded in evidence.
- Truncate a fused channel object or otherwise trade exactness for speed.
- Loosen, skip, or delete a test or tolerance to make a lane pass.
- Claim a speedup without the baseline trio and route attribution, or conflate feasibility with
  acceleration.
- Claim priority, uniqueness, competitor absence, or ownership of consecutive C1/C2 Liouville
  merge; SQUANDER does not originate CPTP composition or noisy-operation fusion.
- Claim quantum advantage from noisy trainability results: non-unital trainability coincides
  with effective shallowness and efficient classical estimation.
- Extend the frozen archive, or renumber a `CAP-*` / `QA-*`.

**Fixed constraints.** An additive module of the upstream SQUANDER library that must build and
pass within its toolchain and CI (`TECH_STACK.md`); all product commands run in the documented
`qgd` environment; Phases 1–3.1 are frozen history under `docs/density_matrix_project/archive/`;
exact dense density matrices remain the reference backend (ADR-005); code, configuration, and
evidence are open-source in branch scope. Fall excludes AVX and GPU; exactness is CPU-verifiable
and has no GPU dependency.

## 8. Strategic assumptions & risks

Ordered by how much scope dies if wrong. Each carries a validation route and a kill criterion.

| # | Assumption | Status | Validation | Kill criterion |
|---|------------|--------|------------|----------------|
| A1a | In the exact regime, depth and noise-class effects on trainability are resolvable at matched fidelity. | **Partially evidenced (exploratory, non-claim-bearing).** 2026-09-23 SQUANDER pilot: amplitude-damping cost variance exceeded fidelity-matched depolarizing by 8×10³ at depth 16 on 6 qubits. | M5 (channel class) and M5B (isolated unitality) pre-registered screens; M13 study. | No relevant effect detected within the tested finite window → noise-class results are reported as null or inconclusive findings and the thesis emphasis moves to methods (CAP-003/004). |
| A1b | Over \(n\le10\), pre-specified width models of the target effect can be discriminated, even though every result remains finite-size. | **Unvalidated; discrimination at \(n\le10\) is doubtful.** ADR-005 selects the exact anchor but is not evidence that \(n\le12\) separates competing width laws. | Pre-registered sensitivity/power analysis under QA-011 (M13). | Models not discriminated → report finite-size observations only; escalate ADR-008 only if the thesis question requires width scaling. |
| A2 | A genuine channel-native route can lower hybrid workload time on a reuse-heavy family without sacrificing exactness. | **Unvalidated; written research plan only.** The frozen historical 26-case matrix remains 17 baseline-sufficient / 9 channel-native but not yet justified / 0 justified at hybrid \(\ge1.2\times\). Separately, Phase 3.1's shipped full-dimension apply costs ≈95 ms per 2-qubit term vs ≈15 ms per sequential operation at 10 qubits; that sentence describes the shipped apply and is not the Phase-3 timing table. | QA-006 on the new two-qubit `reuse_heavy_layered_v0` matrix after an auditable router exists; no Fall result is promised. | No case satisfies QA-006 → reject the hypothesis for that family and report zero justified cells honestly; module readiness and the exact channel work continue. |
| A3 | Representative layered traces contain enough parameter-independent reuse to amortize fused-object construction. | **Unvalidated.** Repeated topology does not imply reusable values; the current path recomposes each execution. | QA-012 at independently observed \(R_\mathrm{obs}\); QA-006 alone decides whether any cell is justified. | Observed reuse never reaches break-even, or useful motifs are parameter-dependent → drop the reuse claim and retain routing only where exactness and honest evidence warrant it. |
| A8 | Gradient-based trainability studies are feasible because an exact gradient costs a small multiple of one energy, not 2K energies. | **Unvalidated; the alternative is infeasible.** Parameter shift at 8 qubits, depth 16 needs 1,344 energies (≈16 min) per gradient; the density path refuses gradients today. | QA-013 in the differentiable-objective milestone. | No method meets QA-013 within the cost bound → CAP-008 stays unmet, the product statement is revalidated, and studies use cost variance, sampled-coordinate gradients, and derivative-free optimizers. |
| A9 | Noise-induced effects depend on noise-weighted depth, so strong-noise, moderate-depth designs stand in for realistic rates at large depth. | **Split by the pilot (exploratory).** Unital decay collapsed onto \(\gamma L\) within ≈20 % at 4 qubits; the non-unital variance at the last sampled exposure rose with γ and was still falling at small γ, so no floor scaling is established. | Collapse check in M5; realistic-rate floors measured directly in M13 at depths of several \(1/\gamma_\mathrm{eff}\). | No collapse even for the unital reference → every realistic-rate point needs direct deep runs, restricting them to 4–6 qubits. |
| A4 | Language-boundary and dispatch overhead materially limits iterative evaluation. | **Likely false on the C++ energy entry.** Time is linear in operation count with a near-zero intercept, 12–15 ns per density-matrix entry per operation at 4–10 qubits; material costs are kernel throughput and the Python planner/runtime route. | QA-007 at 4/6/8 qubits on the C++ entry; time attribution on the planner/runtime route, which has no equal-work comparator. | The QA-007 upper bound on \(O\) is below \(5\,\%\) at every point → CAP-004 becomes a hold-the-line constraint, not an optimization theme. |
| A5 | Expanding beyond the delivered depolarizing, amplitude-damping, and phase-damping channels through GAD covers the next core comparisons. | **Partially validated.** The three local channels are already delivered; GAD is not. GAD with phase damping can express thermal relaxation \((T_1,T_2)\) through a named transform. | QA-002/003/010 plus the first later study's pre-registered noise specification. | A core study requires correlated, coherent, measurement, or trainable noise → revise CAP-002 through Ask-first rather than adding breadth speculatively. |
| A6 | The sequential `NoisyCircuit` executor is a trustworthy internal oracle. | **Validated only on the delivered bounded circuit-ordered three-channel slices**; legacy standalone channels are not claim-bearing. | QA-002/010 reset validation for every new channel/path. | Any counted disagreement beyond tolerance freezes downstream claims until resolved. |
| A7 | Researchers accept explicit rejection or transforms over silent convenience behavior. | **Unvalidated; current parametric paths clamp out-of-range rates.** | Migrate or expose that behavior, pin QA-005 negatives, and collect first external-reproducer feedback. | Users bypass the path because strict input is unusable → improve diagnostics or add explicit transforms, never restore silent behavior. |

**Decisions from elicitation and review (revalidated 2026-10-04).** The current outcome is
module readiness for later noisy VQA loops. The Fall core contains the five registration tasks
and ends with channel expansion; M-F3 then M-F4 remain deferred in that order. The ≥1.2× Track B
artifact is a falsifiable research plan, not a Fall performance promise. Q3 remains parked, E7 is
dropped, and the obsolete 2025 Fall framing is not revived. No priority claim is part of the
product; GAD is expansion beyond the three delivered local channels.

## 9. Out of scope / non-goals

- **Approximate scaling** — stochastic trajectories, MPDO or tensor-network mixed states — until
  the exact regime's limit is hit and escalated (ADR-008).
- **AVX and GPU work this semester** — neither is on the active Fall path; GPU is never a
  dependency of the exactness contract.
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
- **Full-scale noisy VQA runs, training campaigns, or trainability conclusions** — later work
  consumes the ready module and must satisfy QA-011 independently.
- **A committed Fall speedup** — the written ≥1.2× design is a research plan; it promises no
  result and accepts zero justified cells.
- **Publication planning** — papers, abstracts, and slides consume closeouts and live outside
  `docs/specs/`.

## Validation

### PR-FAQ

**Press release (simulated).** *Budapest — SQUANDER's density-matrix track now provides a
verified, fusion-enabled noise module ready for later noisy variational studies.* Researchers
can use channels beyond the delivered depolarizing, amplitude-damping, and phase-damping set,
run noise-aware partitioning with attributable channel-native routes, and call calibrated
C++/Python interfaces. Every advertised path is proven against a sequential reference at
\(10^{-10}\), new channels are cross-checked against Aer and analytical models, and unsupported
work fails before touching the state. The release claims module readiness, not a VQA campaign,
a trainability conclusion, or a promised speedup.

**FAQ.**
- *Isn't Qiskit Aer enough?* Aer remains the external reference and supplies broad channel, GPU,
  and fusion capabilities. SQUANDER's value lies in inspectable planning, strict routes, cost
  prediction, optimizer integration, controlled noise-class designs, and regenerable evidence.
- *Phase 3.1 found zero justified speedups. Why keep fusion?* The historical 26-case matrix stays
  frozen at 17 baseline-sufficient / 9 channel-native but not yet justified / 0 justified at hybrid
  \(\ge1.2\times\). It is not this phase's baseline, and it is not mixed with the Phase-3
  millisecond table. A future QA-006 claim requires a new matrix and speedup versus Phase-3 fused.
- *Why compare noise classes at matched fidelity?* Randomized benchmarking reports only average
  fidelity. At equal fidelity, GAD and its Pauli twirl differ only in unitality, so a
  trainability difference between them is attributable to unitality alone.
- *How big can I go?* Delivered exact evidence covers 4–10 qubits (12 is a stretch). Noise acts
  through depth, which exact simulation does not limit, so depth and noise-class questions are
  answerable here; width results stay finite-size, and QA-011 can only discriminate
  pre-specified models over the tested window.
- *Is the written ≥1.2× plan a Fall commitment?* No. It pre-registers how a later justified
  cell could be earned and treats zero justified cells as a valid outcome.
- *Why exact gradients later?* Gradient variance and gradient-based training are trainability
  observables, but CAP-008 is a later destination capability, not part of Fall module readiness.
- *Does research readiness equal a trainability conclusion?* No. A calibrated, profiled,
  fusion-aware module enables CAP-006; QA-011 governs the later scientific conclusion.
- *What does a reviewer ask?* "Regenerate the counted bundle." QA-008 makes that a named-lane
  answer with per-case data. State-vector users observe nothing (QA-009).
- *Why not AVX or GPU now?* Both are outside this semester. GPU is neither active nor needed to
  establish exactness.

### Critique

**v0.5 (2026-10-04 alignment critique).**
- The prior statement overreached from module delivery to noisy training; the vision, CAP-005,
  PR-FAQ, and scope now stop at module readiness, with CAP-006/008 retained as later destinations.
- QA-006 now uses genuine routing, exactness, and hybrid \(\ge1.2\times\) versus Phase-3 fused;
  the 17/9/0 history and ≈95 ms versus ≈15 ms shipped-apply sentence remain separately bounded.
- Scope/novelty controls retain GAD beyond the three delivered channels, Kraus-primary objects,
  TANQ-Sim PARTIAL / engine-fusion, M-F3→M-F4 deferral, and the AVX/GPU exclusion.

### Stakeholder checkpoint

> Does this capture the durable intent of the track? Which capabilities, quality bars, or
> assumptions are wrong, missing, or too vague to guide a roadmap?

## Handoff

- **Next step:** invoke `create-product-roadmap` separately; preserve the Fall keep-list through
  channel expansion, followed by deferred M-F3 then M-F4. No roadmap edit is authorized here.
- **Current-state docs:** `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` already exist; this
  statement does not change them.
- **Open `[confirm]` values:** the product owner resolves them in the first citing requirements.

## Change log

- **v0.5 (2026-10-04)** — Aligned the North Star to the locked Fall registration record:
  five tasks and module-readiness deliverable; delivered dep/AD/PD boundary; written but
  unpromised ≥1.2× plan; corrected Phase-3-fused justification rule; historical 17/9/0 matrix;
  Kraus-primary representation; TANQ-Sim engine-fusion positioning; M-F3→M-F4 deferral; and
  AVX/GPU, Q3/E7, VQA-campaign, and priority-claim exclusions.
