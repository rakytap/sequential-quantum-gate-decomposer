# Roadmap — SQUANDER density-matrix track

> **Status:** draft v0.7 (scientific redesign after the 2026-09-23 review, two critique passes), at stakeholder checkpoint ·
> **Owner skill:** `create-product-roadmap` ·
> **Last revalidated:** 2026-09-23 — product statement v0.4 and SQUANDER feasibility pilots; no
> milestone under the new convention has closed yet ·
> **Upstream:** [`PRODUCT_STATEMENT.md`](PRODUCT_STATEMENT.md) v0.4 (`CAP-001…008`, `QA-001…013`) ·
> **Downstream:** `milestones/<slug>/INITIAL_REQUIREMENTS.md` via `create-initreq-for-sdd` ·
> **Sequencing inputs:** archived [`PLANNING.md`](../density_matrix_project/archive/planning/PLANNING.md)
> §3, §6, §8; [`RESEARCH_ALIGNMENT.md`](../density_matrix_project/RESEARCH_ALIGNMENT.md);
> [`PHASE4_REQS.md`](PHASE4_REQS.md) ·
> **Not:** requirements, architecture or ADR rationale, dates, or publication planning.

## 1. Summary & horizon

The [product vision](PRODUCT_STATEMENT.md#1-vision) is an exact open-system simulation module
inside a circuit compiler/optimizer that treats noise channels as first-class objects of
partitioning, fusion, and controlled scientific comparison, never trades exactness, and proves
every advertised path under a frozen, regenerable protocol. The destination is CAP-006. Phase 4
ends at **research readiness**; its delivery outcomes are attributed exact energies (M4), an
overhead and kernel profile at comparable tiers (M6), a twirl-controlled GAD anchor (M7), a
canonical energy-only readiness surface with declared dense noise schedules (M8), a
same-complexity local channel application (M9A), and an \(R=1\) representation-aware cost
decision on held-out workloads (M9). Two evidence-only studies run alongside and do not gate
Phase 4: M5 compares non-unital amplitude damping with unital depolarizing at matched fidelity
along depth, and M5B isolates unitality with GAD's exact Pauli twirl once M8 lands. Exact
gradients (M8B), a second ansatz family, reuse activation, global strictness, cross-milestone
reproduction, and trainability studies are Later or gated.

| Horizon | Milestones |
|---------|------------|
| Delivered (archived, frozen) | M1 `phase-1` · M2 `phase-2` · M3 `phase-3` · M3A `phase-3-1` |
| **Now** | **M4 `canonical-attributed-energy` — product walking skeleton** · M5 `matched-fidelity-depth-screen` (evidence-only; handed off after M4) |
| Next | M6 `interop-overhead-profile` → M7 `gad-channel-admission` → M8 `study-ready-noisy-workflow` → M9A `local-channel-application` → M9 `representation-aware-selection`; evidence-only M5B `twirl-isolation-screen` after M8 |
| Later (hypotheses) | M8B `differentiable-noisy-objective` · M8A `second-family-workflow` · M10 `strict-parameter-domains` · M11 `clean-checkout-reproduction` · M12 `observed-reuse-amortization` · M13 `noisy-trainability-studies` |
| Conditional | M5A `exact-regime-boundary` — opens only through review gate RG-1 |
| Review gates (not milestones) | RG-1 exact-regime boundary · RG-2 differentiable objective · RG-3 selection inside the optimizer loop |

**Standing rules.** At most one code-bearing milestone tree plus one evidence-only study tree are
in flight; a study tree adds evidence pipelines, never product code. Suffixed ids (M3A, M5A, M5B,
M8A, M8B, M9A) record lineage, not order. QA-008 is a completion criterion for every milestone:
100% of that milestone's counted claims must regenerate from named lanes with tolerances, seeds,
revision, and claim boundary pinned. Every exactness claim compares the full density matrix with
the sequential `NoisyCircuit` reference; Qiskit Aer is the external reference. `QA-005 (scoped)`
means strictness is claimed only on that milestone's advertised support matrix; M10 is the first
milestone allowed to claim QA-005 globally. Every code-bearing milestone carries QA-009 into its
requirements and evidence. Every noise-class comparison is made at matched average gate fidelity,
and every claim-bearing study follows QA-011's frozen protocol with fresh confirmatory seeds.

## 2. Strategic themes

| Theme | Serves | `PHASE4_REQS.md` thread | Milestones |
|-------|--------|-------------------------|------------|
| **Trusted exact paths and evidence** — classify every advertised entry; calibrate GAD, including its twirl and fidelity witnesses, against closed forms and Aer; make every claim regenerable | CAP-001, CAP-002, CAP-007 · QA-001/002/003/005/008/010 | Channel Expansion; Verification and Calibration | M4, M7, M10, M11 |
| **Inspectable noise-aware planning and cost** — attribute every route; give fusion a same-complexity application before testing it at \(R=1\) on held-out workloads; forecast but never credit reuse; profile overhead and kernel throughput | CAP-003, CAP-004 · QA-004/006/007/012 | Partitioning and Fusion; Interop Optimization; Literature Positioning (software) | M6, M9A, M9, M12 |
| **Noise-class trainability on the exact backend** — compare channel classes at matched fidelity along depth, isolate unitality with GAD's twirl, add exact scalable gradients, and reach pre-registered multi-family studies | CAP-002, CAP-005, CAP-006, CAP-008 · QA-002/005/011/013 | Readiness for noisy VQA training loops; Literature Positioning (scientific frame) | M5, M5B, M8, M8B, M8A, M13, M5A |

## 3. Milestone table

M4 is the product walking skeleton for the new convention. M1–M3A were delivered under the
archived phase convention and are recorded rather than re-planned; no `milestones/<slug>/` tree
is created for them.

| M# | slug | Outcome (measurable) | Horizon | Traces (CAP-*/QA-*) | Depends on | What ships (deployable) | Status |
|----|------|----------------------|---------|---------------------|------------|-------------------------|--------|
| M1 | `phase-1` | Exact mixed-state evolution and ordered noisy-circuit execution with the initial three local channels reproduce the external reference | Delivered | CAP-001, CAP-002 · QA-002, QA-009 | — | C++ core, Python bindings, tests, and Aer comparisons — [`archive/phases/phase-1/`](../density_matrix_project/archive/phases/phase-1/) | Delivered |
| M2 | `phase-2` | The density backend evaluates exact noisy energy in the frozen XXZ/HEA VQE workflow at 4/6/8/10 qubits with machine-checked support boundaries | Delivered | CAP-005, CAP-007 · QA-002, QA-005, QA-008, QA-009 | M1 | Backend selection, exact energy path, bridge metadata, and workflow evidence — [`archive/phases/phase-2/`](../density_matrix_project/archive/phases/phase-2/) | Delivered |
| M3 | `phase-3` | Noisy circuits are first-class planner inputs and execute through a partitioned runtime with exact unitary-island fusion; 34 cases were counted, 0/6 representative cases passed the positive threshold, and 6/6 closed through diagnosis | Delivered | CAP-001, CAP-003, CAP-007 · QA-001, QA-005, QA-008 | M2 | Noisy planner, descriptors, runtime, and correctness/performance evidence — [`archive/phases/phase-3/`](../density_matrix_project/archive/phases/phase-3/) | Delivered |
| M3A | `phase-3-1` | Exact strict/hybrid channel-native fusion closed as a bounded decision study: 17/26 `phase3_sufficient`, 9/26 `phase31_not_justified_yet`, 0/26 `phase31_justified` | Delivered | CAP-001, CAP-003, CAP-007 · QA-001, QA-002, QA-004, QA-005, QA-008 | M3 | Bounded channel-native modes and the 26-row decision bundle — [`archive/phases/phase-3-1/`](../density_matrix_project/archive/phases/phase-3-1/) | Delivered |
| **M4** | **`canonical-attributed-energy`** | **Product walking skeleton.** From one public VQE instance, a researcher obtains exact noisy energy through bridge → planner → attributed runtime → exact core → observable; every route is labelled, every q4/6/8/10 result meets QA-001 on the full state, and energy agrees with the public optimizer path and Aer | Now | CAP-001, CAP-003, CAP-005, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009 | M2 and M3A outcomes | Public attributed-energy entry, route records, entry-point support matrix and negatives, counted state/energy rows, M4 closeout, updated current-state docs | Now — active |
| M5 | `matched-fidelity-depth-screen` | A pre-registered, confirmatory, evidence-only screen compares non-unital amplitude damping with unital local depolarizing matched in average gate fidelity on the canonical HEA under a declared dense schedule, along noise-weighted depth at 4 and 6 qubits with an 8-qubit confirmatory family; the paired log ratio of cost variance returns `separated`, `equivalent-within-margin`, or `inconclusive` for this channel-class contrast only | Now | CAP-002, CAP-006, CAP-007 · QA-008, QA-009, QA-011 | M2 outcome (delivered public energy and channels); handed off after M4 | Archived pre-registration, sensitivity lane, verdict and diagnosis bundle, descriptive gradient and collapse rows | Now — evidence-only |
| M5A | `exact-regime-boundary` | If RG-1 approves, the practical exact-regime boundary (width, depth, memory, time) is characterized under a frozen protocol while QA-001 remains satisfied | Conditional | CAP-001 · QA-001, QA-008 | RG-1 (A1b) | Boundary evidence and escalation record; no approximate implementation | Conditional hypothesis |
| M6 | `interop-overhead-profile` | On the public C++ energy entry, language-boundary overhead is bounded under QA-007 at 4/6/8 qubits and frozen workloads; on M4's planner/runtime route, which has no equal-work lower comparator, time is attributed to Python orchestration versus C++ kernels; per-operation kernel and parallel throughput are recorded | Next | CAP-004, CAP-007 · QA-007, QA-008, QA-009 | M4 | Comparable-tier harness, route attribution profile, kernel and parallel-throughput profiles, overhead fitness function | Next |
| M7 | `gad-channel-admission` | Fixed-parameter GAD under the per-step \((\gamma,p_\mathrm{exc})\) convention is admitted on its declared core/planner entries; every counted state meets QA-001/002/010, including the translation-sign, Pauli-twirl, and fidelity-invariance witnesses; every malformed class fails before mutation; Gibbs wording needs a declared two-level Hamiltonian and basis | Next | CAP-001, CAP-002, CAP-007 · QA-001, QA-002, QA-003, QA-005 (scoped), QA-008, QA-009, QA-010 | M4 support matrix | GAD operation/binding/API, planner and hybrid support, strict refusal, calibration and twirl witnesses, Aer evidence, published convention | Next |
| M8 | `study-ready-noisy-workflow` | The canonical XXZ/generated-HEA workflow runs declared dense schedules of depolarizing, phase damping, amplitude damping, and GAD through one public exact energy evaluator at 4–10 qubits, with thermal relaxation expressible only through a named, Aer-checked transform; a bounded derivative-free optimizer smoke emits trace-schema rows without a quality claim; every unsupported mode is refused; all claims regenerate from a clean checkout | Next | CAP-002, CAP-005, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009, QA-010 | M4; M7; M6 posture recorded | Energy-only readiness surface, schedule families, thermal-relaxation transform, support/refusal matrix, trace-schema smoke, clean-checkout bundle | Next |
| M5B | `twirl-isolation-screen` | A pre-registered, evidence-only screen isolates unitality: amplitude damping against its exact Pauli twirl GAD(γ, ½), at equal fidelity and twirl, along noise-weighted depth on the canonical workload under the M5 protocol; verdict `separated`, `equivalent-within-margin`, or `inconclusive`; a short \(p_\mathrm{exc}\) sweep is descriptive | Next | CAP-002, CAP-006, CAP-007 · QA-008, QA-009, QA-010, QA-011 | M8 (GAD in the public energy); M5's archived protocol | Archived pre-registration, study lane, verdict bundle, descriptive sweep rows | Next — evidence-only |
| M9A | `local-channel-application` | Every channel-native route on the frozen Phase 3.1 slice executes through local Kraus or superoperator application at the sequential reference's asymptotic complexity and execution tier, meets QA-001 and its QA-004 sentinels at 4–10 qubits, refuses supports above its declared width, and has its per-operation throughput measured against the sequential kernels | Next | CAP-001, CAP-003, CAP-004, CAP-007 · QA-001, QA-004, QA-005 (scoped), QA-008, QA-009 | M3A; M6 | Same-complexity application primitive, channel-native route integration, exactness evidence, throughput profile | Next |
| M9 | `representation-aware-selection` | On M9A's application, a selection policy calibrated on one frozen workload set — the Phase 3.1 families plus a static-subgraph surface — is judged on a held-out set at \(R=1\), with \(R^*\) forecast only and 8–10 qubits decisive; every non-selected row names its mechanism; a pass-count cost model with operation-class constants predicts held-out costs within a frozen error; the competitor matrix gates both outcomes | Next | CAP-001, CAP-003, CAP-007 · QA-001, QA-004, QA-005 (scoped), QA-006, QA-008, QA-009 | M9A; M8 closed | Frozen cost-policy ADR, \(R=1\) selector, cost records, pass-count model, competitor matrix, positive or diagnosis evidence | Next |
| M8B | `differentiable-noisy-objective` | Exact gradients of the noisy energy for fixed-noise U3/CNOT circuits, inside the existing optimizer loop, meet QA-013 at 4–10 qubits at no more than a frozen multiple of one energy evaluation `[confirm]`; unsupported parameterizations are refused | Later | CAP-005, CAP-007, CAP-008 · QA-001, QA-005 (scoped), QA-008, QA-009, QA-013 | RG-2 passed | Exact gradient method chosen by ADR, density-backend gradient entry, lifted refusal for supported modes, gradient-fidelity evidence lane | Later hypothesis — M13 enabler |
| M8A | `second-family-workflow` | An independently motivated second ansatz family — by default the Hamiltonian variational ansatz for TFIM and XXZ, compiled to U3/CNOT with tied parameters — evaluates exact energy under declared dense schedules at 4–10 qubits, meets QA-001/002, inherits the M8 refusal discipline, and regenerates from a clean checkout; the TFIM task on the generated HEA is its tracer slice | Later | CAP-002, CAP-005, CAP-006, CAP-007 · QA-001, QA-002, QA-005 (scoped), QA-008, QA-009 | M8; M7 if its noise uses GAD | Second task and ansatz families, tied-parameter routing, workflow evidence, clean-checkout bundle | Later hypothesis — readiness extension |
| M10 | `strict-parameter-domains` | Every advertised out-of-domain parametric rate is rejected before mutation or uses an explicit caller-selected transform; no silent clamps remain and QA-005 is globally claimable | Later | CAP-001, CAP-002, CAP-005 · QA-003, QA-005, QA-008, QA-009 | M4 support matrix | Domain-semantics ADR, migrated advertised paths, and pinned negatives | Later hypothesis |
| M11 | `clean-checkout-reproduction` | An external reproducer regenerates 100% of closed claim-bearing milestones among M4, M5, M5A, M5B, M6, M7, M8, M9A, M9, M8A, M8B, and M10 from a clean checkout | Later | CAP-007 · QA-008 | Every listed milestone closed when M11 opens; that set is frozen then | Regeneration index, runbook, and cross-milestone bundle | Later hypothesis |
| M12 | `observed-reuse-amortization` | If M9 shows material candidate construction and independent optimizer traces yield \(R_\mathrm{obs}\) beyond forecast \(R^*\) with margin, reuse-conditioned caching is activated and meets QA-001/012; otherwise A3 is dropped and caching stays off | Later | CAP-003 · QA-001, QA-008, QA-009, QA-012 | M9 cost record and forecast; M8/M8A/M8B traces; independent of the \(R=1\) verdict | Parameter-dependent cache keys, hit/miss and build/apply accounting, amortization evidence | Later gated hypothesis |
| M13 | `noisy-trainability-studies` | Pre-registered studies across at least two task families and two ansatz families, matched-fidelity unital and non-unital sweeps, and seeded ensembles report effect sizes, uncertainty, nulls, and a regenerable dataset, with every result at \(n\le10\) labelled finite-size | Later | CAP-006, CAP-007 · QA-008, QA-009, QA-011; CAP-008 and QA-013 for gradient-based arms | M5 and M5B verdicts; M8A; M11; M8B for gradient-based arms | Study pipelines, reusable dataset, and analysis regeneration lane | Later hypothesis — scientific endpoint; expected to split |

### Review gates

Gates are decisions, not milestones; they carry no traces of their own and open nothing by
themselves.

| Gate | Trigger | Required before any milestone opens | Owner |
|------|---------|-------------------------------------|-------|
| **RG-1 Exact-regime boundary** | A1b: a pre-registered width-model analysis fails to discriminate its models and the thesis question requires width scaling, or a study diagnosis names the exact-size ceiling | Product-statement / ADR-008 review approval; M5A is post-Phase-4 by default, and inserting it earlier requires explicit `PHASE4_REQS.md` plus roadmap revalidation | Product owner via `create-product-statement` |
| **RG-2 Differentiable objective** | Opening M8B or any gradient product surface | The owner confirms product statement v0.4 (CAP-008, QA-013); M8 has closed; a pre-registered study needs gradient-based arms or gradient statistics beyond sampled coordinates; a frozen cost and memory budget exists | Product owner |
| **RG-3 Selection inside the optimizer loop** | M9 ships the positive deployable | M9 revalidation; M12 \(R_\mathrm{obs}\) evidence for any reuse claim | Roadmap revalidation |

Until M8B closes, gradients are benchmark-internal, control-validated instrumentation in evidence
pipelines (M5, M5B, M13); the product surface refuses gradient modes.

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

### M5 — `matched-fidelity-depth-screen` (Now · evidence-only)

**Outcome and why now.** Current theory splits by noise class: unital noise predicts
noise-induced barren plateaus (Wang et al. 2021), while non-unital noise predicts limit sets and
effectively shallow circuits whose final layers stay trainable (Singkanipa and Lidar 2025; Mele et
al. 2026). A noise-versus-noiseless contrast cannot test this, because its answer is fixed by the
chosen minimum effect. M5 compares amplitude damping (non-unital, anisotropic) with local
depolarizing (unital, isotropic) matched in average gate fidelity — the quantity randomized
benchmarking reports — along depth at fixed small width, the form of A1 the exact regime can
answer (A1a). The arms differ in unitality and anisotropy together, so M5 claims a channel-class
difference at matched fidelity only; isolating unitality needs GAD's exact twirl and is M5B's
question. M5 needs only delivered channels and the public energy. An exploratory pilot already
shows the effect, so M5 is a confirmation, and its value is a pre-registered, regenerable result
on the production engine that later studies can cite.

**Design (frozen and archived in requirements before counted data; `[confirm]` values there).**
- *Arms.* AD(γ); `local_depolarizing` with \(p=1-\tfrac13\big(2\sqrt{1-\gamma}+1-\gamma\big)\), which
  equalizes \(F_\mathrm{avg}\) because the implemented channel shrinks the Bloch vector by \(1-p\);
  a noiseless reference.
- *Workload.* The canonical generated HEA with XXZ under a declared dense schedule — one channel
  on each qubit touched by each gate, one frozen γ. Edge qubits then receive half the channels of
  interior ones, so exposure is per qubit: \(x_q=\sum_j-\log(1-\gamma_j)\) over qubit \(q\)'s
  channels, identical in both arms at matched fidelity. Depths are ordered by the mean \(\bar x\),
  and every row reports the per-qubit range. The canonical three-channel schedule remains the
  continuity anchor.
- *Families.* Primary: widths 4 and 6 × depths spanning \(\bar x\) from one layer to \(\ge5\) (at
  γ = 0.05, one layer on 6 qubits is \(\bar x\approx0.17\)). Confirmatory: 8 qubits to
  \(\bar x\ge2\). Design points are pre-selected from pilot and analytic evidence so that both arms'
  cost variance stays above a frozen numerical-resolution floor; a counted point that falls below
  it makes its family `inconclusive` and is never excluded afterwards.
- *Sampling.* Seeds disjoint from the pilot; parameter draws shared across arms within each design
  point; stored-coordinate periods follow the U3 multipliers {2,1,1}; sample sizes from the pilot's
  variance and the frozen precision target.

**Estimands and inference.** Primary:
\(\Delta_L=\log\mathrm{Var}_\theta E_\mathrm{AD}(L)-\log\mathrm{Var}_\theta E_\mathrm{dep}(L)\) at each
design point; cost variance diagnoses plateaus as gradient variance does (Arrasmith et al. 2022) at
one energy per draw. A studentized max-statistic paired bootstrap, resampling draws within design
points, gives simultaneous two-sided 90% intervals over each family, so each side is a
simultaneous one-sided 95% bound. The margin \(\delta\) comes from a stated relevance criterion, not
from pilot effect sizes; the precision target is a maximum interval half-width. Secondary and
descriptive only:
last- and first-layer gradient variance from coordinate-aware two-term shifts,
\(\partial_jE=\tfrac{m_j}{2}\big[E(p+\tfrac{\pi}{2m_j}e_j)-E(p-\tfrac{\pi}{2m_j}e_j)\big]\) with
\(m_j\in\{2,1,1\}\), valid only for fixed noise and order and gated by a central finite-difference
check on every row; and whether the depolarizing arm collapses onto \(\bar x\) across two rates
at 4 qubits (A9).

**Verdict (exactly one).**
- `separated`: at every design point beyond a pre-registered \(\bar x\), the lower bound exceeds
  \(+\delta\), in the primary family and, separately, in the 8-qubit confirmatory family;
- `equivalent-within-margin`: every interval in both families lies within \([-\delta,+\delta]\) and
  meets the precision target;
- `inconclusive`: otherwise, including an effect of reversed sign, reported with its sign, and
  any counted point below the resolution floor. There is no in-milestone extension; a follow-up
  needs a new pre-registration.

**What the verdict establishes.** Whether non-unital amplitude damping and unital depolarizing,
at equal average fidelity, differ in cost variance along depth at 4–8 qubits on the canonical
workload. It does not isolate unitality (M5B), discriminate width models (A1b), or establish
training outcomes or GAD dependence (M13).

**Feasibility (non-counted exploratory pilot, 2026-09-23).** At γ = 0.05 per channel, with 300,
150, and 60 paired draws at 4, 6, and 8 qubits and 25 energies per draw, the full design ran in
194 s on 120 processes. Coordinate-aware shifts matched finite differences within 1.7×10⁻¹⁰; the
coordinate-blind rule returned ≈10⁻¹⁷ on a θ/2 coordinate whose true derivative was −7.1×10⁻².
Cost-variance ratios (AD ÷ depolarizing, 95% bootstrap intervals) were 1.34 [1.16, 1.60], 11.9
[8.5, 16.8], and 8.0×10³ [5.4×10³, 1.2×10⁴] at depths 4, 8, and 16 on 6 qubits, and 18 [10, 36]
and 3.1×10⁴ [1.6×10⁴, 5.7×10⁴] at depths 8 and 16 on 8 qubits. The primary estimand alone needs
about 10⁴ energies, minutes of serial compute.

**Scope.** In: the archived pre-registration, sensitivity lane, verdict and diagnosis bundle, and
gradient instrumentation as evidence tooling. Out: product code, GAD, unitality isolation,
width-model claims, optimizer runs, and performance claims.

**Riskiest assumption.** A1a, at 4–8 qubits and the frozen design.

**Dependencies and handoff.** M2's delivered outcome: the public energy and the three channels.
M5 is handed off after M4, so M4 remains the first milestone under the convention, and it runs as
the evidence-only tree. Handoff slug: `matched-fidelity-depth-screen`.
[`TECH_STACK.md`](TECH_STACK.md) gains the sensitivity lane; no architecture change is expected.

### M6 — `interop-overhead-profile` (Next)

**Outcome and why next.** The public C++ energy is linear in operation count with a near-zero
intercept (3.56 µs per operation at 4 qubits) and costs 12–15 ns per density-matrix entry per
operation at 4–10 qubits, so the language crossing is unlikely to matter on that entry. M4's
planner/runtime route keeps its orchestration in Python and has no equal-work lower comparator,
so there M6 attributes time rather than bounding an overhead ratio. Kernel throughput sets every
science budget above 6 qubits. M6 supplies the comparable tiers M9A and M9 need, without blocking
the correctness-only M4.

**Success measure.**
- *C++ entry.* At 4/6/8 qubits and frozen workloads (depth, schedule), at least 1,000 warmed
  public calls are paired and interleaved with a C++-level invocation of the same prebuilt
  evaluator; state reset, allocation, materialization, batching, build, affinity, and warm-up are
  frozen; report components, uncertainty, and the one-sided 95% upper bound on
  `O = (T_public - T_lower) / T_public` under QA-007.
- *Planner/runtime route.* Time split between Python orchestration and C++ kernels per partition
  class, as a diagnosis; no QA-007 claim.
- *Kernels.* Per operation type, throughput in ns per density-matrix entry; aggregate throughput of
  independent evaluations under a frozen parallel protocol.
- Meeting QA-007 on the C++ entry satisfies CAP-004's bar there; missing it closes through a
  component diagnosis with QA-007 unmet; an upper bound below 5% at every width triggers A4's
  kill, and CAP-004 becomes a hold-the-line constraint for that entry.

**Scope and assumption.** In: harness, profiles, validator, and at most one bounded interop-only
fix. Out: kernel rewrites — findings such as the amplitude-damping kernel allocating and zeroing a
full temporary matrix per application become ADR inputs for M9A — plus fusion, channels, GPU work,
and optimizer changes. Riskiest assumption: A4.

**Dependencies and handoff.** M4. Handoff slug: `interop-overhead-profile`. Update both
current-state docs with the lanes and measured posture.

### M7 — `gad-channel-admission` (Next)

**Outcome and why next.** GAD is the non-unital calibration anchor named by CAP-002 and the
controlled-unitality instrument: at fixed γ every GAD shares one Pauli twirl, GAD(γ, ½), and one
average fidelity, so sweeping \(p_\mathrm{exc}\) changes only the non-unital displacement
\(\nu=\gamma(1-2p_\mathrm{exc})\). M7 admits it as a per-step fixed-parameter channel under the
\((\gamma,p_\mathrm{exc})\) convention. “Gibbs-population-calibrated” is used only where
requirements declare \(H=\Delta|1\rangle\langle1|\), \(|0\rangle\) ground, \(\Delta>0\),
\(\beta\ge0\), and \(p_\mathrm{exc}=(1+e^{\beta\Delta})^{-1}\); physical-time semantics arrive only
with M8's named thermal-relaxation transform.

**Success measure.** Fixed-parameter GAD:
- matches the QA-010 population law, coherence scaling \(\sqrt{1-\gamma}\), fixed point
  \(\mathrm{diag}(1-p_\mathrm{exc},p_\mathrm{exc})\), \(k\)-step relaxation law, Hermiticity, and
  contractivity;
- meets the twirl witnesses — Bloch translation \(t_z=\gamma(1-2p_\mathrm{exc})\) with the
  \(|0\rangle\)-ground sign, a Pauli-transfer diagonal independent of \(p_\mathrm{exc}\), and
  \(F_\mathrm{avg}\) invariance — which catch a swapped ground/excited convention;
- agrees with Aer at boundary and interior points and with the sequential reference on full
  states at 4–10 qubits for every advertised direct, partitioned, fused-island, and hybrid route;
- rejects every enumerated non-finite, out-of-domain, invalid-target, parametric-mode, and
  legacy-API request before mutation, is refused by strict channel-native mode, and takes a
  labelled baseline route in hybrid mode.

**Scope.** In: fixed-parameter circuit-ordered GAD, binding/API, planner/runtime lowering,
calibration and twirl witnesses, external rows, and the published convention. Out: parametric
GAD, a strict channel-native GAD bundle, Gibbs convenience mapping, legacy API extension, and the
VQE noise-spec integration and thermal-relaxation transform (both M8).

**Feasibility.** The four-Kraus form generalizes delivered amplitude damping, so the support delta
is bounded: core operation, binding, planner vocabulary, runtime lowering, sequential reset
validation, and the Aer mapping (amplitude damping with an excited-state population). Every
witness is closed-form. The milestone ADR decides where each piece lives.

**Riskiest assumptions.** A5 (inventory sufficiency) and A6 (the sequential oracle resets
correctly for a new channel).

**Dependencies and handoff.** M4 support matrix. Handoff slug: `gad-channel-admission`. Update
both current-state docs for the admitted inventory and route boundary.

### M8 — `study-ready-noisy-workflow` (Next)

**Outcome and why next.** This is the bounded readiness outcome of `PHASE4_REQS.md`: a canonical,
energy-only surface on which pre-registered studies can declare their noise. The delivered
backend supports the derivative-free `BAYES_OPT` and `COSINE` traces and refuses gradient entry
points; M8 makes that surface complete, refusal-pinned, GAD-capable, physically parameterizable,
and regenerable. It precedes M9 so the research path does not depend on a fusion win, and it
ships neither a second family nor a gradient product API.

**Success measure.**
- The canonical XXZ/generated-HEA family lowers under declared dense schedules — one channel on
  each qubit touched by each gate, with fixed rates that may differ by gate type — of depolarizing,
  phase damping, amplitude damping, and GAD at 4–10 qubits; full states meet QA-001, energies
  QA-002. A schedule is called device-motivated only when its rates derive from gate durations and
  \(T_1/T_2\) through the named transform.
- **Thermal relaxation as a named transform.** A caller-selected transform maps
  \((T_1,T_2,t,p_\mathrm{exc})\) to GAD with \(\gamma=1-e^{-t/T_1}\) followed by phase damping with
  \(\lambda=1-e^{-2t/T_\varphi}\), \(1/T_\varphi=1/T_2-1/(2T_1)\), inside QA-010's declared domain;
  it refuses \(T_2>2T_1\), is recorded in metadata, and matches Aer's thermal-relaxation channel.
  The phase-damping code comment stating \(\lambda=1-e^{-t/T_2}\) is corrected in the same change.
- One documented public entry returns exact noisy energy with M4 route attribution.
- **Optimizer smoke — usability, not optimization quality.** One current derivative-free mode
  (`BAYES_OPT` or `COSINE`, frozen in requirements) runs under a bounded iteration and wall-clock
  budget with a pinned seed at 4 qubits and one larger advertised width. It must complete, emit a
  finite best-energy history, and reproduce sampled visited energies through the public
  evaluator. The trace records iteration index, parameter vector, energy, and route labels; it
  fixes the schema later used for \(R_\mathrm{obs}\) and supports no convergence, optimality, or
  reuse claim. Upstream optimizer behavior is unchanged under QA-009.
- A support matrix preflight-refuses every gradient entry, gradient-based optimizer, non-HEA
  ansatz, non-generated source, unsupported gate/noise name, and parametric-noise request, each
  with a pinned negative.
- 100% of M8's counted claims regenerate from a clean checkout.

**Scope.** In: GAD-capable dense schedule families, the thermal-relaxation transform, public
energy-evaluator documentation, support/refusal matrix, trace-schema smoke, workflow evidence,
and clean-checkout proof. Out: scientific contrasts (M5B), the second family (M8A), gradient
product API (M8B), trainability conclusions, selection in the optimizer loop, approximate methods,
noisy re-synthesis, new gates, and changes to state-vector partitioners or upstream optimizers.

**Feasibility.** Energy evaluation, the two derivative-free modes, and a noise-insertion API that
accepts arbitrary ordered lists already exist; the support delta is GAD in the noise
specification, schedule generation, the transform, refusals, the smoke, and clean-checkout
regeneration. A dense schedule costs about twice the canonical lowering per layer (measured at
4–10 qubits). If the milestone exceeds its artifact budgets, the thermal-relaxation transform is
the first candidate to split into its own milestone, named at requirements time.

**Riskiest assumptions.** A5 (the inventory, including thermal relaxation, covers the study
specifications) and A7 (explicit refusal is usable).

**Dependencies and handoff.** M4 supplies the public attributed entry; M7 supplies GAD; M6's
posture is recorded but does not expand scope. Handoff slug: `study-ready-noisy-workflow`. Update
both current-state docs and the non-spec API reference.

### M5B — `twirl-isolation-screen` (Next · evidence-only)

**Outcome and why next.** With GAD in the public energy, unitality can be isolated exactly:
amplitude damping AD(γ) = GAD(γ, 0) and its Pauli twirl GAD(γ, ½) share every Pauli-transfer
diagonal entry and the average fidelity, and differ only in the non-unital displacement. M5B
pre-registers that contrast along depth on the canonical workload — the controlled unitality
experiment that current theory's disagreement calls for.

**Success measure.** The M5 protocol (archived pre-registration, fresh seeds, primary and
confirmatory families, max-statistic paired bootstrap, relevance-derived margin, no in-milestone
extension), with primary estimand the log cost-variance ratio AD ÷ GAD(γ, ½) and the same three
verdicts. Descriptive only: cost variance across a short \(p_\mathrm{exc}\) grid at fixed γ,
last-layer gradient variance, and a fidelity-matched depolarizing arm, so M5's channel-class gap
splits into a unitality part (AD against the twirl) and an anisotropy part (twirl against
depolarizing). The exploratory pilot's twirl–depolarizing agreement, within 10% at every depth,
is not a result here.

**Scope.** In: pre-registration, study lane, verdict bundle, descriptive sweep rows. Out: product
code, γ dependence of floors, width-model claims, and optimizer runs (M13).

**Riskiest assumption.** A1a in its isolated-unitality form.

**Dependencies and handoff.** M8 (GAD in the public energy, with M7's twirl witnesses) and M5's
archived protocol; M5's variances inform its sample sizes, not its margin, and its verdict does
not gate it. Evidence-only tree.
Handoff slug: `twirl-isolation-screen`.

### M9A — `local-channel-application` (Next)

**Outcome and why next.** Phase 3.1 validly found no justified speedup for its shipped
channel-native path, which builds each 2-qubit Kraus term with a Python loop over all \(4^n\)
matrix entries and applies it through dense full-dimension products: at 10 qubits about 95 ms per
term (70.7 ms embedding, 24.5 ms products with this host's multithreaded BLAS) against about 15 ms
per sequential C++ operation. Whether fusion can pay at equal application complexity is open. M9A
provides local Kraus or superoperator application at the sequential reference's asymptotic
complexity and execution tier, so M9 can test A2 fairly; faster exact channel-native execution is
useful on its own.

**Success measure.** Every channel-native route on the frozen Phase 3.1 slice, strict and hybrid,
executes through the new application and meets QA-001 against the sequential reference at 4–10
qubits with its QA-004 sentinels; per-operation throughput (ns per density-matrix entry) is
measured against the sequential kernels under M6's protocol; supports above the declared width are
refused with a structured error.

**Scope.** In: the primitive, starting with supports of at most two qubits, its integration into
the channel-native routes, exactness evidence, and the throughput profile. Out: selection, cost
records, reuse, GAD bundles, and GPU kernels.

**Feasibility.** The main C++ engineering cost of the fusion track. Its ADR decides
representation (Kraus rank versus superoperator), layout, and placement; the existing
local-unitary kernels are the pattern. If it cannot ship, M9A closes through diagnosis, M9 does not
open, A2 stays untested at equal complexity, and the Phase 4 outcome is partial (§5).

**Riskiest assumption.** That same-complexity local application is achievable in the current
execution tier without loss of exactness — A2's precondition.

**Dependencies and handoff.** M3A; M6's comparable-tier protocol. Handoff slug:
`local-channel-application`. Update both current-state docs with the primitive.

### M9 — `representation-aware-selection` (Next)

**Outcome and why next.** With M9A's same-complexity application in place, M9 re-enters fusion
with representation, canonicalization, support, construction versus application cost, and
comparable tiers explicit, on the frozen Phase 3.1 families plus one frozen static-subgraph
candidate surface — a hypothesis, not a prescribed design. It tests A2 after M8 secures research
readiness.

**Success measure.** Before measurement, a milestone ADR freezes the candidate/motif surface,
representation and canonicalization policy, the calibration and held-out workload sets, memory
ceiling, and positive-claim thresholds. The selection policy is calibrated on the first set and
judged on the second. Every partition records representation/support, logical transformations,
canonical complexity, sequential/candidate construction and application work, attributed memory,
paired timings, and prediction error.
- **Complete \(R=1\) decision.**
  \[
  T_s^{(1)}=B_s+A_s+L_s,\qquad
  T_c^{(1)}=B_c+A_c+L_c,
  \]
  where \(B\) is construction, \(A\) application, and \(L\) lookup/route-selection overhead.
  Sequential build and overhead are measured, not assumed zero. QA-006 selection uses
  simultaneous one-sided 95% upper bounds of \(T_c^{(1)}/T_s^{(1)}\) on the held-out set, with no
  reuse credit.
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
  that no reuse is needed, but an actual win still requires the QA-006 bound at \(R=1\). These are
  forecast fields only and never enter M9 selection. Runtime/cache instrumentation identifies the
  applicable reuse model; independently collected optimizer traces provide \(R_\mathrm{obs}\).
  Only M12 may activate caching. These definitions move into M9's requirements at handoff.
- **Pass-count cost model (transferability is a hypothesis).** Full-matrix passes per operation
  class × entries × operation-class constants, with each fused object's Kraus-rank or
  superoperator factor, calibrated on the first set, must predict held-out motifs and qubit counts
  within a frozen relative error `[confirm]`. Transfer beyond SQUANDER is claimed only after the
  model holds on a second implementation or execution tier.
- Decisive design points are 8 and 10 qubits, where full-matrix passes dominate fixed per-call
  and composition costs; 4 and 6 qubits are reported but cannot decide A2.
- No held-out candidate above sequential \(R=1\) cost or the memory ceiling is selected; every
  executed selection meets QA-001/006, every route is attributed, and every motif class has a
  QA-004 sentinel.

Every non-selected or diagnosis row names its mechanism from the taxonomy frozen before
measurement — representation growth, construction cost, application cost, memory ceiling, or
interop overhead. A row names one dominant mechanism only if it exceeds the others by a frozen
factor; otherwise it is labelled mixed, with components listed. The version-pinned competitor
matrix is required for both deployables. The positive deployable exposes selection for eligible
static subgraphs only when that matrix supports the claim boundary. The diagnosis deployable
keeps the planner non-selecting, publishes the cost record, model, and mechanisms, and records A2
as killed. GAD motifs are excluded.

**Scope.** In: cost-record schema, frozen candidate surface and workload sets, mechanism
taxonomy, \(R=1\) selector, \(R^*\) forecast, pass-count model, performance pipeline, competitor
matrix for both outcomes, and the positive or diagnosis deployable. Out: the application
primitive (M9A), reuse-conditioned caching (M12), a GAD Kraus bundle, approximate methods, and
GPU kernels.

**Riskiest assumption.** A2. The positive arm must beat sequential at \(R=1\) without reuse, at
equal application complexity, on held-out workloads. A negative result remains useful through the
cost diagnosis and cannot block M8.

**Dependencies and handoff.** M9A; closed M8. Handoff slug: `representation-aware-selection`.
Update both current-state docs with the cost model, selection surface, and governing ADR.

### Later — M8B, M8A, M10–M13 and conditional M5A

- **M8B `differentiable-noisy-objective`.** Opens only through RG-2. Exact gradients for
  fixed-noise U3/CNOT circuits inside the existing optimizer loop, meeting QA-013 at 4–10 qubits;
  the method is an ADR decision, with adjoint differentiation reusing the existing U3 derivative
  kernels the leading candidate. Budget, assuming 200 optimizer steps per run: parameter shift at
  8 qubits and depth 16 needs 1,344 energies per gradient (≈16 min serial), so a 240-run study
  needs ≈12,800 CPU-hours, against ≈29 at an estimated three energy-equivalents per gradient. Kill
  (A8): CAP-008 stays unmet and M13 uses cost variance, sampled-coordinate gradients, and
  derivative-free optimizers.
- **M8A `second-family-workflow`.** The thesis multi-family bar needs a second ansatz, not only a
  second Hamiltonian. The default is the Hamiltonian variational ansatz — plateau-free without
  noise for TFIM and mildly plateaued for XXZ (Wiersema et al. 2020) — compiled to U3/CNOT with
  tied parameters; the TFIM task on the generated HEA, a Hamiltonian-only change, is the tracer
  slice. Gradients for tied parameters need a chain rule and wait for an M8B extension.
- **M10 `strict-parameter-domains`.** Replace clamps on advertised parametric paths with
  structured rejection or an explicit recorded transform under an Ask-first ADR. This is the
  global QA-005 milestone.
- **M11 `clean-checkout-reproduction`.** Have the external-reproducer persona regenerate the
  accumulated claim set; turn the regeneration index into the standing QA-008 surface for M13.
- **M12 `observed-reuse-amortization`.** Open only when M9 shows material construction and real
  optimizer traces independently show \(R_\mathrm{obs}\) beyond forecast \(R^*\); it is the only
  milestone allowed to activate reuse-conditioned caching. It remains independent of M9's
  \(R=1\) verdict.
- **M13 `noisy-trainability-studies`.** The pre-registered, multi-family scientific endpoint,
  expected to split at requirements time into a gradient-statistics study and an
  optimization-landscape study. Its lead question carries M5B's isolated non-unital displacement
  beyond cost variance: does \(\nu\) set effective trainable depth and optimization outcomes at
  realistic relaxation rates, across both ansatz families? Further candidate questions the pilots
  raise: how a late-layer floor, if
  it exists, depends on \(p_\mathrm{exc}\) and γ (the pilot's small-γ runs had not reached one);
  effective trainable depth; whether Pauli twirling of \(T_1\)-dominated noise removes a floor;
  local-minimum proliferation under non-unital noise (Fontana et al., arXiv:2011.08763); and the
  noiseless-energy penalty of the noisy optimum as a gauge-invariant migration measure.
  Realistic-rate points are measured directly at depths of several \(1/\gamma_\mathrm{eff}\) (A9).
  Results characterize a trainable window, never quantum advantage.
- **M5A `exact-regime-boundary`.** Open only through RG-1 (A1b). It characterizes the exact
  boundary and records escalation; approximate methods or a GPU dependency require
  product-statement revision.

### Feasibility guidance (Now/Next)

Guidance, not design: implementation choices belong to each milestone's ADRs. “Support delta”
is what exists today versus what the outcome needs; “kill” is the pre-registered stop.

| M# | Current support delta | Expected slices | Evidence lanes | Compute budget / kill condition |
|----|-----------------------|-----------------|----------------|---------------------------------|
| M4 | Public C++ density energy and planner→runtime state execution exist; attributed runtime energy, energy route rows, and the support matrix do not | 2–3: q4 partitioned tracer; fused/hybrid routes; q6–q10, Aer, negatives | fast pytest; correctness pipeline; Aer | Existing correctness protocol; any QA-001 route disagreement freezes downstream claims (A6) |
| M5 | Delivered channels, public energy, and the insertion API suffice; instrumentation must derive shifts from U3 multipliers {2,1,1} and pass finite-difference checks | 2–3: q4 tracer with instrumentation checks; archived pre-registration; counted families and verdict | new sensitivity lane; fast pytest witnesses | Pilot: full design with secondary gradients in 194 s on 120 processes; primary estimand ≈10⁴ energies. Descriptive points shrink first; a primary design point is never dropped without a new pre-registration |
| M6 | No paired/interleaved equal-work harness for the C++ entry; no route attribution or per-operation kernel profile | 2: C++-entry harness and kernel profile at q4; q6/q8, route attribution, validator, diagnosis | performance pipeline | ≥1000 warmed calls per tier and width plus repeats; without a C++-level comparator the entry closes as diagnosis with QA-007 unmet |
| M7 | GAD is absent from core, bindings, planner vocabulary, runtime bundles, and Aer mapping; amplitude damping is the template | 2–3: core/binding/calibration and twirl witnesses; planner/runtime/refusals; q4–q10 Aer evidence | fast pytest; correctness pipeline; Aer; optional C++ | Existing correctness protocol; a sequential–Aer disagreement beyond QA-002 or a failed twirl witness freezes admission |
| M8 | Energy, `BAYES_OPT`/`COSINE`, and the insertion API exist; GAD in the specification, schedule families, the thermal transform, the refusal matrix, the smoke, and the clean-checkout bundle do not | 3: schedules, GAD-in-spec, and matrix at q4; transform with Aer and q4–q10; smoke and clean checkout | fast pytest; workflow pipeline; Aer; clean-checkout lane | Dense schedules cost about 2× canonical lowering; if q10 is unaffordable, M8 is partial/missed unless formally revalidated before data; the transform is the first split candidate if budgets are exceeded |
| M5B | Needs M8's GAD in the public energy; otherwise reuses M5's lane | 2: q4 tracer; archived pre-registration, families, and verdict | sensitivity lane | Same order as M5, one extra arm; no in-milestone extension |
| M9A | Channel-native application embeds Kraus terms at full dimension; no same-complexity C++ primitive exists | 2–3: primitive with QA-001 on the strict slice; hybrid routes and sentinels; throughput profile at 8–10 qubits | correctness pipeline; performance pipeline; fast pytest; optional C++ | Existing correctness protocol; if exactness or the declared complexity cannot be met, close as diagnosis; M9 does not open and the Phase 4 outcome is partial |
| M9 | Records lack build/apply split, canonical complexity, and attributed memory; no candidate surface, workload sets, taxonomy, pass-count model, or competitor matrix | 3–4: schema and instrumentation; candidate surface, sets, taxonomy, and selector; model, competitor matrix, and verdict; split mandatory if any Layer artifact exceeds its budget | performance pipeline; correctness pipeline; fast no-regression tests | Paired/interleaved trials under the memory ceiling at 8–10 qubits; if no held-out candidate bound is ≤1 at \(R=1\), ship the mechanism-labelled diagnosis, model, and competitor matrix, kill A2, and keep \(R^*\) as an M12 forecast |

Throughput note: under 120 concurrent processes, per-energy time rose 1.5–2.1× over single-process
runs, so independent evaluations reach about 55–80× serial throughput on the 128-core host.

## 5. Sequencing rationale

- **Walking skeleton first.** M4 is the smallest useful positive path on the primary persona's
  public object that crosses every currently connectable product boundary. Full-state checks
  prevent energy agreement from hiding a wrong state. GAD-only admission is a channel slice, not
  a product skeleton; interop alone is a measurement.
- **The contested scientific question, early and honestly scoped.** M5 asks, with delivered
  channels, whether channel class changes the cost landscape at matched fidelity along depth; it is
  a pre-registered confirmation of an exploratory effect, handed off after M4 and run as the
  evidence-only tree. M5B then isolates unitality with GAD's exact twirl as soon as M8 lands, which
  is where the scientific novelty lies.
- **Profile → knob → readiness → primitive → fair fusion.** M6 profiles the C++ entry, attributes
  the runtime route, and measures kernels; M7 adds the twirl-controlled knob; M8 makes declared and
  thermal noise expressible and Aer-checked; M9A gives fusion a same-complexity application; M9
  then tests selection at \(R=1\) on held-out workloads, at the sizes where passes dominate.
- **\(R=1\) decision first, reuse forecast only.** Phase 3.1's 0/26 holds for its shipped
  full-dimension path. M9 must win or lose at \(R=1\) under complete cost accounting and equal
  application complexity; every negative row names its mechanism, and the competitor matrix
  accompanies both outcomes. \(R^*\) is a forecast; only M12 may activate caching.
- **Gradients planned, not smuggled.** CAP-008 and QA-013 define the contract; M8B follows Phase 4
  behind RG-2, which needs the owner's confirmation, a closed M8, a pre-registered need, and a cost
  and memory budget. Until then, gradients stay benchmark-internal, derived from U3 multipliers
  {2,1,1}, because a coordinate-blind shift is a correctness bug.
- **Phase 4 outcome verdict.** Phase 4 is **achieved** when M4, M7, and M8 achieve their outcomes,
  M6 meets QA-007 on the C++ entry (A4's kill, an upper bound below 5%, implies it), M9A ships the
  same-complexity application, and M9 reaches a held-out QA-006 decision, positive or diagnosis,
  with its version-pinned competitor matrix, which is the Literature Positioning deliverable.
  It is **partial** otherwise: M4, M7, or M8 closes without its outcome, QA-007 is unmet on the
  C++ entry, or M9A cannot ship, in which case M9 does not open. A partial verdict triggers
  roadmap revalidation and, if needed, a `PHASE4_REQS.md` revision. If A2 dies, the fusion
  deliverable is the attributed evaluation-mode route, the primitive, the cost model, and the
  mechanism-labelled diagnosis — not a default speedup. M5 and M5B are evidence-only and do not
  gate Phase 4.
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
| A1a Depth and noise-class effects are resolvable at matched fidelity | M5 (channel class); M5B (isolated unitality); M13 | Noise-class results become finite-size observations; the thesis emphasis moves to methods |
| A1b Pre-specified width models are discriminable at \(n\le10\) | M13 pre-registered sensitivity analysis | Report finite-size observations only; RG-1 only if the thesis question needs width scaling |
| A2 Selective fusion finds paying workloads | M9A same-complexity application, then M9 at \(R=1\) on held-out workloads; \(R^*\) forecast only | Ship diagnosis and cost model; keep the planner non-selecting; no selection-in-loop |
| A3 Real traces contain amortizing reuse | M12, gated on M9's cost record and independently observed \(R_\mathrm{obs}\) from M8/M8A/M8B traces | Drop reuse and optimize per evaluation only |
| A8 An exact gradient costs a small multiple of one energy | M8B (QA-013) | CAP-008 unmet; M13 uses cost variance, sampled-coordinate gradients, and derivative-free optimizers |
| A9 Noise-weighted depth licenses strong-noise proxies | M5 collapse check (unital arm); M13 direct realistic-rate measurements | Realistic-rate points need direct deep runs at 4–6 qubits |
| A4 Interop overhead materially limits iteration | M6 on the C++ entry; route attribution on the runtime path | CAP-004 becomes a hold-the-line constraint for that entry |
| A5 Inventory through GAD covers the core studies | M7; M8 thermal transform; M8A | Revisit CAP-002 through Ask-first; no speculative breadth |
| A6 Sequential executor is a trustworthy oracle | M4, M7, M9A, every new path | Freeze downstream claims until disagreement is resolved |
| A7 Researchers accept explicit refusal over clamping | M8 in use; M10 migration | Improve diagnostics or explicit transforms; never restore silence |

### Roadmap risks

| Risk | Mitigation |
|------|------------|
| M4 expands into channels, timing, or optimizer integration | Bind it to the support matrix and one-evaluation outcome |
| The evidence-only tree drifts into product code | Study trees add pipelines only; any product change moves to a code-bearing milestone |
| M5's channel-class contrast is read as a unitality result | M5 claims channel class only; M5B isolates unitality with GAD(γ, ½) |
| The pilot anticipates M5's verdict | M5 is labelled a confirmation; fresh seeds, an archived analysis plan, and a relevance-derived margin; M5B carries the open question |
| Fidelity matching uses the wrong depolarizing convention | Derive \(p\) from the implemented Bloch shrink \(1-p\); verify equal \(F_\mathrm{avg}\) numerically at every design point |
| Multiplicity or optional stopping inflates error | One simultaneous family per study, a separate confirmatory family, two-sided simultaneous intervals for equivalence, descriptive secondaries, and no in-milestone extension |
| Noise-weighted depth is ambiguous under uneven insertion | Per-qubit exposure \(x_q\), a frozen scalar summary, and the per-qubit range on every row |
| Benchmark gradient applies a coordinate-blind shift | Derive shifts from multipliers {2,1,1}; a finite-difference gate on every row; gradients stay benchmark-internal until M8B |
| Realistic rates need very deep circuits | A9: unital proxies via noise-weighted depth; non-unital points measured directly at 4–6 qubits (≈0.07 and ≈1.7 ms per dense layer) |
| The planner/runtime route has no equal-work comparator | M6 closes it as an attribution diagnosis; QA-007 applies only where a comparator exists |
| Phase 4 is declared closed without its outcomes | The §5 outcome-verdict rule distinguishes achieved from partial |
| M9A's primitive is costly or late | Its ADR starts at supports of at most two qubits; if it cannot ship, M9 does not open and the Phase 4 outcome is partial |
| M9 credits unobserved reuse or overfits its selector | Selection stays at \(R=1\) on a held-out set; \(R^*\) is forecast-only; caching activates only in M12 |
| M9 candidate family has no eligible static subgraph | Record “no candidate” and use the attributed baseline route; the surface remains an ADR hypothesis |
| M9 revives an unjustified novelty or transferability claim | The competitor matrix gates both outcomes; transfer needs a second implementation or tier |
| The second family is only a second Hamiltonian | M8A requires an independent ansatz; TFIM on the HEA is its tracer slice only |
| M8B adjoint memory at 10 qubits | Checkpointing per ADR; RG-2 needs a frozen cost and memory budget |
| Deferred thresholds remain open | Freeze each `[confirm]` value in the first citing `INITIAL_REQUIREMENTS.md` |
| Optional Aer dependency drifts | Pin versions in every QA-002 evidence bundle |
| Scoped QA-005 is mistaken for global compliance | Label every scoped trace; reserve the global claim for M10 |
| GAD receives unconditioned Gibbs or time language | Per-step \((\gamma,p_\mathrm{exc})\); Gibbs only with declared Hamiltonian, basis, \(\beta\), \(\Delta\); time only through M8's named transform |
| Trainability results are read as quantum advantage | Product-statement Never guardrail; effective shallowness is reported beside every floor |
| Two noise representations drift | Keep legacy `NoiseChannel` non-claim-bearing and unextended |

### Revalidation log

- **2026-09-23 — v0.7, scientific redesign after the PhD research review; product statement
  v0.4; two critique passes.** *Learned:* a noise-versus-noiseless screen is settled (exploratory
  NumPy pilot: noise cut gradient variance by ≈20% at depth 1 and ≈70–75% at depth 4). At matched fidelity, non-unital
  amplitude damping keeps cost and last-layer gradient variance orders of magnitude above unital
  references along depth; on SQUANDER itself (non-counted), cost variance exceeded
  fidelity-matched depolarizing by 8.0×10³ at depth 16 on 6 qubits and 3.1×10⁴ on 8 qubits.
  Phase 3.1's shipped channel-native path costs ≈95 ms per 2-qubit Kraus term at 10 qubits against
  ≈15 ms per sequential operation, so whether same-complexity application changes its verdict is
  open. The C++ energy has a near-zero per-call intercept and costs 12–15 ns per entry per
  operation. A parameter-shift gradient at 8 qubits, depth 16 costs 1,344 energies. Unital decay
  collapsed onto noise-weighted depth in the pilot; non-unital variance did not and had not
  plateaued at small γ. The phase-damping comment's \(T_2\) mapping contradicts its Kraus form, and
  the canonical schedule has three channels. *Changed:* M5 redesigned and renamed
  `matched-fidelity-depth-screen` (no tree existed) as a confirmatory, evidence-only channel-class
  screen with a frozen max-statistic protocol, depending only on M2's delivered outcome; M5B
  `twirl-isolation-screen` added to isolate unitality after M8; M6 retargeted to the C++ entry,
  route attribution, and kernels; M7 gains twirl witnesses; M8 gains declared dense schedules and
  the thermal-relaxation transform; M9A `local-channel-application` split from M9, which now
  validates on held-out workloads and treats model transfer as a hypothesis; a Phase 4
  outcome-verdict rule added; M8B added after Phase 4 behind a tightened RG-2; M8A requires an
  independent ansatz; M13 trimmed and expected to split; RG-1 retargeted to A1b; A8–A9 added.
  The second critic pass added two-sided simultaneous equivalence intervals, per-qubit exposure
  (dense HEA schedules give edge qubits half the interior channels), the rule that M9 does not
  open if M9A fails, a deterministic Phase 4 verdict, and M13's lead question.
- **2026-09-21 — v0.6.** M5's positive verdict required a q4-and-q6 pattern; M8 gained the
  trace-schema smoke and a descriptive contrast; M9 gained the mechanism taxonomy and the
  competitor matrix for both outcomes.
- **2026-09-21 — v0.5.** Source-verified correction: coordinate-aware U3 shifts, \(R=1\) selection
  with \(R^*\) forecast only, energy-only M8, M8A and RG-2 added, GAD language bounded.
- **2026-09-21 — v0.4.** Scientific enhancements, later corrected by v0.5 (a universal shift rule
  and reuse-credited selection).
- **2026-09-20 — v0.1–v0.3.** Roadmap created after the Phase 3.1 closure with M4 as walking
  skeleton; Phase 3 (34 counted, 0/6 positive, 6/6 diagnosis) and Phase 3.1 (17/9/0) outcomes
  recorded; A1 screened early; GAD separated from selection; QA-005 scoped until clamp migration.

### Critique verdict

- Outcome honesty: delivered rows are unchanged, and Phase 3.1's verdict is kept within its claim
  boundary; the open question about same-complexity application lives in A2, M9A, and M9.
- Sequencing: M4 validates integration risk; M5 confirms a channel-class effect and M5B isolates
  unitality; A4 and kernel costs are measured and the primitive exists before A2 is tested; study
  readiness precedes and is independent of A2; gradients (A8) precede M13's gradient-based arms;
  A3 is gated last.
- Inference: every claim-bearing study freezes one simultaneous family, a confirmatory family,
  fresh seeds, per-qubit exposure, and a relevance-derived margin before data; equivalence needs
  two-sided simultaneous intervals within the margin; secondaries are descriptive; there is no
  optional stopping.
- Deferred work is visible: gradient product work is M8B behind RG-2; the second ansatz is M8A;
  reuse activation is M12; strict migration is M10; width models are A1b with RG-1.
- Deployability: M4–M9A are bounded to a few slices; M9 must split during requirements if its
  frozen matrix does not fit any Layer artifact budget; M13 is expected to split.
- Orphans: none. Every milestone has non-empty CAP/QA traces; CAP-008 and QA-013 are traced by M8B
  and by M13's gradient-based arms.
- Escalation: this change revises the product statement to v0.4; no further escalation.

**Critique verdict:** Ready for the stakeholder checkpoint. The first adversarial pass raised 14
blocking findings and the second raised 6; all are resolved, and a confirmation pass found no
remaining blocker.

### Stakeholder checkpoint and handoff

> Is this sequence right? Which milestone outcomes, measures, or dependencies are wrong,
> missing, or mis-prioritized before M4 opens?

**Handoff:** invoke `create-initreq-for-sdd` for M4 `canonical-attributed-energy` with its
outcome, support matrix, CAP-001/003/005/007 and QA-001/002/005-scoped/008/009 traces,
dependencies, deployable result, and the instruction to update—not create—the current-state
architecture and stack docs at close. M5 `matched-fidelity-depth-screen` follows through the same
skill as the evidence-only tree once M4 has been handed off.
