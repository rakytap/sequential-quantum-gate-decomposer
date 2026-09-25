# Thesis Ideas: Exact Noisy Variational-Circuit Research

> **Aligned with:** [`PRODUCT_STATEMENT.md`](../specs/PRODUCT_STATEMENT.md) v0.4 and
> [`ROADMAP.md`](../specs/ROADMAP.md) v0.7 (2026-09-23); revise this note when either changes.
> **Not a spec artifact:** thesis writing consumes milestone closeouts and never gates a slice.
> Pilot numbers quoted here are exploratory and non-claim-bearing.

## Feasibility as a PhD Basis

The narrowed novelty can support a PhD thesis that stands on two legs, which the product
statement now treats as co-equal:

- **Science.** Current theory diverges by noise class: unital noise predicts noise-induced
  barren plateaus (Wang et al. 2021), while non-unital noise predicts limit sets and effectively
  shallow circuits whose final layers stay trainable (Singkanipa and Lidar 2025; Mele et al.
  2026). A finite-size test of this split needs exact states, channels matched in fidelity, and
  depth sweeps whose variances fall below any sampling budget, which is what the exact backend
  and its calibrated noise inventory supply.
- **Methods.** Complete, representation-aware cost accounting that decides when exact channel
  fusion pays at equal application complexity, and diagnoses where it does not.

The thesis must not claim that SQUANDER originated CPTP-channel composition or noisy-operation
fusion: Qiskit Aer, QuEST, and Qulacs already fuse or accept general channel maps. Nor may it
claim the theory it tests: SQUANDER supplies instruments for testing that frame, and the thesis
contributes controlled, pre-registered finite-size evidence about it. Nor may it read non-unital
trainability as quantum advantage, since such trainability coincides with effective shallowness
and efficient classical estimation.

A noise-versus-noiseless contrast is not a thesis question: its answer is fixed by the chosen
minimum effect, and pilots already show noise lowering gradient variance. The question is which
property of the noise decides trainability when benchmark fidelity is held fixed. It is also
cheap to ask: an exploratory run of M5's full design took 194 s on 120 processes, and its primary
estimand alone needs about 10⁴ energies.

**Reading "scalable" in the thesis theme.** The PhD theme is "scalable methods for training
quantum circuits under realistic noise models" (archived
[`PLANNING.md`](archive/planning/PLANNING.md); [`RESEARCH_ALIGNMENT.md`](RESEARCH_ALIGNMENT.md)).
Read against the product statement, "scalable" is a claim about cost and experimental reach:
exact gradients at a small multiple of one energy rather than 2K energies for K parameters,
fusion only where it lowers end-to-end cost, and the extra optimizer runs and deeper sweeps these
buy. It is not a claim about asymptotic width laws, which exact simulation at \(n\le10\) cannot
establish.

## Candidate Thesis Statement

> This thesis asks which property of realistic local noise decides how variational quantum
> circuits train when benchmark fidelity is held fixed. Using exact open-system simulation inside
> a circuit compiler that treats noise channels as first-class objects, it compares noise classes
> at matched average gate fidelity along noise-weighted depth, isolates unitality exactly through
> generalized amplitude damping and its Pauli twirl, and reports the depth and noise-class effects
> the exact regime resolves, labelling width trends finite-size. Alongside, it establishes under
> complete, representation-aware cost accounting when exact channel fusion lowers end-to-end cost,
> and diagnoses where it does not.

## Evidence Boundary (as of 2026-09-23)

Every thesis claim must stay within its evidence status:

- **Delivered and claim-bearing (archived Phases 1–3.1).** The exact density-matrix core with
  depolarizing, amplitude-damping, and phase-damping channels; exact noisy energies for the
  canonical XXZ/HEA workflow at 4, 6, 8, and 10 qubits; a noisy planner and partitioned runtime
  with unitary-island fusion (34 counted cases; 0/6 representative cases positive, 6/6 closed
  through diagnosis); and a bounded channel-native decision study (0/26 justified).
- **Exploratory and non-claim-bearing (2026-09-23 pilots).** At γ = 0.05, amplitude-damping cost
  variance exceeded fidelity-matched depolarizing by 8.0×10³ (95% interval 5.4×10³–1.2×10⁴) at
  depth 16 on 6 qubits; the C++ energy costs 12–15 ns per density-matrix entry per operation; and
  Phase 3.1's channel-native path costs ≈95 ms per 2-qubit Kraus term against ≈15 ms per
  sequential operation at 10 qubits. These numbers size the plan; none is a thesis result.
- **Planned.** Everything in M4–M13. A thesis claim may rest on a milestone only once its
  closeout evidence matrix supports it.

## Potential Contributions

### 1. Noise-Class Trainability at Matched Fidelity

The scientific core is a three-step ladder. Each step changes one property of the noise and is
pre-registered under QA-011:

1. **Channel class (M5, a confirmation).** Non-unital amplitude damping against unital local
   depolarizing matched in average gate fidelity — the quantity randomized benchmarking reports —
   on the canonical HEA/XXZ workload under a declared dense schedule, along noise-weighted depth
   at 4 and 6 qubits with an 8-qubit confirmatory family. The arms differ in unitality and
   anisotropy together, so this step claims a channel-class difference only.
2. **Isolated unitality (M5B, the novel step).** Amplitude damping AD(γ) against its exact Pauli
   twirl GAD(γ, ½), which shares its average fidelity and Pauli-transfer diagonal and differs only
   in the non-unital displacement \(\nu=\gamma(1-2p_\mathrm{exc})\). A descriptive
   fidelity-matched depolarizing arm splits the channel-class gap into a unitality part and an
   anisotropy part.
3. **Consequences for training (M13).** Does \(\nu\) set effective trainable depth and
   optimization outcomes at realistic relaxation rates, across two task and two ansatz families?
   Candidate follow-ups: how a late-layer floor, if one exists, depends on \(p_\mathrm{exc}\) and
   γ; whether Pauli twirling of \(T_1\)-dominated noise removes a floor; local-minimum
   proliferation under non-unital noise (Fontana et al., arXiv:2011.08763); and the
   noiseless-energy penalty of the noisy optimum.

M5 and M5B use cost variance as the primary observable, since it diagnoses plateaus as gradient
variance does at one energy per draw (Arrasmith et al. 2022). M13 adds layer-resolved gradient
variance, effective trainable depth, and optimization outcomes, and is expected to split into a
gradient-statistics study and an optimization-landscape study. Every claim-bearing study freezes
its design before counted data: matched fidelity, per-qubit noise-weighted depth, confirmatory
seeds disjoint from any pilot, a margin derived from a stated relevance criterion, simultaneous
intervals, and no in-study extension. Each screen returns exactly one verdict — `separated`,
`equivalent-within-margin`, or `inconclusive` — so a null result is reportable. The canonical
three-channel schedule remains a continuity anchor, not a trainability noise model.

### 2. Representation-Aware Selective Fusion at Equal Application Complexity

Phase 3.1's negative verdict holds only for its shipped path, which applies composed Kraus terms
at full dimension; whether fusion pays at equal application complexity is open. The contribution
comprises:

- a local Kraus or superoperator application at the sequential reference's asymptotic complexity
  and execution tier (M9A), which is the precondition for a fair test and is useful on its own;
- an execution-cost record exposing representation, canonical channel complexity, construction
  versus application work, memory, and prediction error, so a logical-transformation count never
  stands alone;
- an \(R=1\) selection policy calibrated on one frozen workload set and judged on a held-out set
  by simultaneous one-sided 95% upper bounds on \(T_\mathrm{candidate}/T_\mathrm{sequential}\)
  from paired, interleaved trials, with 8 and 10 qubits decisive (M9);
- a pass-count cost model (full-matrix passes per operation class, entries, and operation-class
  constants) that predicts held-out costs within a frozen error; transfer beyond SQUANDER stays a
  hypothesis until the model holds on a second implementation or execution tier;
- a mechanism taxonomy (representation growth, construction cost, application cost, memory
  ceiling, interop overhead) that names why each non-selected row lost, and a version-pinned
  competitor matrix that gates both a positive and a diagnosis outcome.

### 3. Reuse Break-Even: Forecast, Then Observation

- Selection never credits reuse. M9 reports the break-even count \(R^*\) as a forecast only,
  under both a reusing and a rebuilding sequential reference.
- Reuse is credited only in M12, when \(R_\mathrm{obs}\), derived independently from real
  optimizer traces, exceeds the forecast with margin and the amortized cost meets QA-012;
  repeated topology does not imply reusable values. Otherwise the reuse claim is dropped (A3).
- This separates per-evaluation benefit from amortization, but it is a gated extension of the
  methods leg, not a leg of its own.

### 4. Calibrated, Differentiable, and Reproducible Exact Reference

- Sequential `NoisyCircuit` execution as the internal exact reference (full-state agreement
  within 10⁻¹⁰), Qiskit Aer as the external reference, and closed-form open-system models as the
  calibration reference — including GAD's translation-sign, Pauli-twirl, and fidelity-invariance
  witnesses and an Aer-checked thermal-relaxation \((T_1,T_2)\) transform, the only route by
  which a noise schedule may be called device-motivated.
- Exact gradients under fixed, parameter-independent noise at a small multiple of one energy
  (CAP-008, QA-013), checked against a coordinate-aware two-term shift oracle and finite
  differences. Until then, benchmark gradients derive their shifts from the U3 multipliers
  {2, 1, 1}; a coordinate-blind shift is a correctness bug.
- Overhead and kernel-throughput profiles at comparable execution tiers, which separate an
  algorithmic result from an implementation artifact (M6).
- Frozen tolerances, channel conventions, seeds, and claim boundaries, with every claim-bearing
  result regenerable from a named lane and, finally, from a clean checkout (M11).

This framework supports the thesis but should not be presented as its sole scientific novelty.

## Relationship to Phase 4 and the Roadmap

| Thesis part | Question it answers | Roadmap sources | Horizon |
|-------------|---------------------|-----------------|---------|
| Exact, calibrated noisy module | Is every advertised path exact, attributed, and physically calibrated? | Phases 1–3.1; M4, M7, M8 | Delivered; Now; Next |
| Representation-aware fusion | When does exact channel fusion pay at equal application complexity? | Phase 3.1; M6, M9A, M9; M12 if it opens | Next; Later (gated) |
| Noise class at matched fidelity | Does channel class, and then unitality alone, change the cost landscape along depth? | M5, M5B | Now; Next (evidence-only) |
| Noisy trainability | Does \(\nu\) set effective trainable depth and optimization outcomes? | M13, with M8A, M11, and M8B for gradient-based arms | Later |

Phase 4 ends at research readiness, not at a trainability conclusion. It is **achieved** when M4
(attributed exact energy), M7 (GAD admission), and M8 (study-ready energy-only workflow) reach
their outcomes, M6 meets QA-007 on the C++ energy entry, M9A ships, and M9 reaches a held-out
QA-006 decision — positive or diagnosis — with its competitor matrix; otherwise it is
**partial**. For the thesis, Phase 4 supplies:

- the methods chapter (M6, M9A, M9), whether or not fusion pays;
- the calibrated instruments for the science: GAD and its twirl (M7), declared dense schedules
  and the thermal-relaxation transform (M8);
- verification evidence and the version-pinned competitor matrix that carries Literature
  Positioning.

Interop optimization will likely end as a hold-the-line constraint rather than a contribution:
the C++ energy entry has a near-zero per-call intercept, and the material costs are kernel
throughput and the Python planner/runtime route (A4).

The evidence-only screens run alongside Phase 4 without gating it: M5 needs only delivered
channels and is handed off after M4, while M5B waits for GAD in the public energy (M8). After
Phase 4, the scientific endpoint M13 needs the M5 and M5B verdicts, a second ansatz family (M8A;
by default the Hamiltonian variational ansatz for TFIM and XXZ, Wiersema et al. 2020),
cross-milestone reproduction (M11), and, for gradient-based arms, exact gradients (M8B, behind
review gate RG-2). Phase 4 alone is therefore unlikely to constitute the complete PhD.

## Main Scientific Risks

### Width, Not Depth, Limits the Exact Regime

The exact regime is 4–10 qubits at shallow depth, with 12 as a stretch, but depth sweeps to
hundreds of layers are routine at 4–6 qubits. Noise acts through depth, so depth and noise-class
effects are resolvable here (A1a, partially evidenced by the pilot), while width laws are
unlikely to be discriminable at \(n\le10\) (A1b). Therefore:

- label every width result at \(n\le10\) finite-size; a pre-registered sensitivity analysis may
  only discriminate pre-specified models over the tested window, whereas depth trends may be
  fitted across depth;
- do not rest the thesis on a width-scaling claim;
- if the thesis question does require width scaling, review gate RG-1 can open an exact-regime
  boundary study (M5A); trajectory, MPDO, tensor-network, or GPU-dependent methods then need an
  Ask-first product-statement revision and validation against the exact reference on overlapping
  problem sizes.

### Gradient Cost Decides Which Studies Are Feasible

Parameter shift needs 2K energies per gradient: 1,344 at 8 qubits and depth 16, about 16 minutes
serially. At 200 optimizer steps per run, a 240-run study would cost about 12,800 CPU-hours,
against about 29 at an estimated three energy-equivalents per exact gradient. Gradient-based arms
therefore wait for M8B, which follows Phase 4 behind RG-2, and tied-parameter gradients for the
second ansatz family need a further M8B extension. Until then, studies use cost variance,
sampled-coordinate gradients, and derivative-free optimizers (A8). For gradient-based arms, exact
gradients are the dominant lever on experimental reach; fusion has yet to show any justified
per-evaluation speedup.

### Realistic Rates Need Direct Deep Runs

Noise-weighted depth licenses strong-noise proxies only for unital decay, which collapsed onto
\(\gamma L\) within about 20% at 4 qubits in the pilot. Non-unital variance did not collapse and
had not reached a floor at small γ, so realistic-rate non-unital points must be measured directly
at depths of several \(1/\gamma_\mathrm{eff}\), which restricts them to 4–6 qubits (A9).

### The Most Novel Result Arrives Late

M5 confirms an effect the exploratory pilot already shows, so its novelty is modest. The
isolated-unitality result (M5B) waits for M8, and the training consequences (M13) come after
Phase 4, so the thesis timeline should not assume either arrives early.

## Positive and Negative Research Outcomes

The two legs carry separate kill criteria — A1a for the science, A2 for fusion — so the thesis
should be planned for all four combinations:

- **A relevant noise-class effect and a paying fusion policy.** The strongest thesis: a
  controlled noise-class result and a validated representation-aware selection method, which
  widens the reach of exact noisy-VQA studies once selection runs inside the optimizer loop
  (review gate RG-3).
- **A relevant noise-class effect, no paying fusion.** The science carries the thesis. If M9A
  cannot ship, M9 does not open, A2 stays untested at equal complexity, and Phase 4 is partial.
  If M9 finds no held-out candidate at \(R=1\), A2 is killed and the methods chapter becomes a
  bounded negative result — the attributed evaluation-mode route, the same-complexity primitive,
  a cost model that predicts held-out costs, a mechanism-labelled account of where fusion loses,
  and the competitor matrix — extending Phase 3.1's full-dimension verdict to equal application
  complexity.
- **No relevant noise-class effect, a paying fusion policy.** The emphasis moves to methods
  (CAP-003/004), as A1a's kill criterion prescribes. An `equivalent-within-margin` verdict is
  still a finding: over the tested window, the contrast changes cost variance by less than the
  relevance margin. An `inconclusive` verdict needs a new pre-registration, not an extension.
- **Neither.** The calibrated module, the reproducible methodology, and the bounded diagnoses are
  unlikely to carry a thesis alone. The direction must then be revisited with the product owner —
  through RG-1 if the question has become width scaling — and never by weakening a frozen
  protocol.

The two screens can also split: a channel-class gap in M5 with an equivalent twirl contrast in
M5B points to anisotropy rather than unitality, which M5B reports only descriptively and a
follow-up study would have to pre-register.

## Minimum Bar for a Strong Thesis

Following the success conditions in archived `PLANNING.md` §2.1 and CAP-006/QA-011:

- At least one substantive, pre-registered conclusion about noisy variational training beyond a
  noise-versus-noiseless contrast — planned as the isolated-unitality screen and the M13
  studies — reported with effect sizes, uncertainty, and null outcomes.
- Two independently motivated task families and two ansatz families, matched-fidelity unital
  and non-unital sweeps, and seeded initialization ensembles with confirmatory seeds disjoint
  from any pilot.
- Realistic local noise under declared schedules, compared at matched average gate fidelity with
  per-qubit noise-weighted depth reported, and depth and noise-class effects kept separate from
  width trends, which stay finite-size.
- A methods result under complete cost accounting at equal application complexity: a held-out
  positive selection at \(R=1\), or a mechanism-labelled diagnosis with a validated cost model.
- Performance work tied to concrete experimental reach: more optimizer runs, deeper sweeps, or
  gradient-based arms that were previously infeasible.
- Novelty stated against a version-pinned competitor matrix, with no priority, uniqueness, or
  quantum-advantage claim.
- A reusable dataset, and every claim-bearing result regenerable from a clean checkout.

## Open Decisions That Shape the Thesis

- **M5's relevance margin δ.** It must come from a stated relevance criterion, not from pilot
  effect sizes, and is frozen in M5's requirements; it is the protocol's key scientific choice.
- **Whether the thesis question needs width scaling.** If it does, RG-1 and a product-statement
  revision must precede any approximate method.
- **The remaining `[confirm]` thresholds**, notably QA-006's positive-claim bound
  \(\delta_\mathrm{eff}\), QA-013's gradient cost multiple, and the 12-qubit stretch, which the
  product owner fixes in the first requirements that cite them.
