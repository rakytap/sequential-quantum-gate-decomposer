# Roadmap — SQUANDER density-matrix track

> **Status:** draft v0.8, aligned to product statement v0.5; stakeholder validation remains open ·
> **Owner skill:** `create-product-roadmap` ·
> **Last revalidated:** 2026-10-08 — M-F5a closeout `918a73a4`, status Shipped (CLOSEOUT wording). The step that promotes a later milestone to Now was not run ·
> **Upstream:** [`PRODUCT_STATEMENT.md`](PRODUCT_STATEMENT.md) v0.5 (`CAP-001…008`, `QA-001…013`) ·
> **Downstream:** `milestones/<slug>/INITIAL_REQUIREMENTS.md` via `create-initreq-for-sdd`; this revision authorizes no handoff or implementation ·
> **Not:** requirements, architecture or ADR rationale, dates, publication planning, or a promise of a Fall speedup.

## 1. Summary & horizon

The current outcome is a verified, fusion-enabled noise module ready for later noisy VQA
loops. It is not a VQA campaign, a full-scale run, or a trainability conclusion. The active
keep-list ends at workload-driven channel expansion:

`M-F1a → M-F5a → M-F1b → M-F2 → M-F-CE`.

Only after that keep-list, `M-F3` and then `M-F4` are deferred stretch hypotheses. No stretch
result is promised. `M4 canonical-attributed-energy` is parked and inactive; this roadmap does
not make it a start.

| Horizon | Milestones |
|---------|------------|
| Delivered (archived, frozen) | M1 `phase-1` · M2 `phase-2` · M3 `phase-3` · M3A `phase-3-1` · M-F5a `cpp-python-interop-profile` |
| **Now** (sequence position only; no handoff authorized) | **M-F1a `exactness-reconfirmation` — product walking skeleton for the new convention** |
| Next (keep-list, strict order) | M-F1b `committed-timings-record` (held: not started until Zoltán approves via PhD Manager) → M-F2 `hybrid-cost-model` → M-F-CE `workload-driven-channel-expansion` |
| Later (deferred stretch) | M-F3 `reuse-heavy-layered-evaluation` → M-F4 `strict-s3-paper-operations` |
| Parked | M4 `canonical-attributed-energy` · Q3 |
| Dropped | E7 |
| Unscheduled destinations | CAP-006 noisy variational research · CAP-008 exact differentiable noisy objective |

**Standing evidence rules.** Exactness always means the full density matrix against the
sequential `NoisyCircuit` oracle; Qiskit Aer is the external reference. Every counted claim
names a regenerable lane and every executed evaluation-mode partition carries a route label.
Kraus bundles are the primary fused objects; Choi and Liouville forms are witnesses. AVX and
GPU work are outside this semester, and GPU is not an exactness dependency.

Where a speedup rule is stated, it is the written QA-006 research plan: a justified cell must
execute at least one genuine channel-native route (a skip or baseline route does not count),
pass exactness against the sequential oracle, and show median-of-three hybrid wall-clock
speedup of at least 1.2× versus the Phase-3 fused baseline. Zero justified cells is a valid
outcome. The frozen 26-case matrix is historical speedup-only evidence at 17/9/0; it is not
this phase's baseline or matrix and is never mixed with the separate Phase-3 millisecond table.

## 2. Strategic themes

| Theme | Serves | Milestones |
|-------|--------|------------|
| **Trusted exact paths and regenerable evidence** — reconfirm the oracle contract, preserve historical provenance, admit channels only with calibrated evidence, and prove the deferred strict-only paper-operations surface | CAP-001, CAP-002, CAP-007 · QA-001/002/003/004/005/008/009/010 | M-F1a, M-F1b, M-F-CE, M-F4 |
| **Inspectable interop and hybrid planning** — bound comparable language-boundary cost and make predicted-versus-executed hybrid routing observable without presenting feasibility as acceleration | CAP-003, CAP-004, CAP-007 · QA-001/004/005/007/008/009/012 | M-F5a, M-F2, M-F3 |

Literature positioning is a cross-cutting evidence obligation, not a priority claim. TANQ-Sim
remains **PARTIAL / engine-fusion**. SQUANDER does not claim to originate consecutive C1/C2
Liouville merge, noisy-operation fusion, or CPTP composition, and does not claim to be first to
fuse. No unverified CSCS arXiv or DOI is introduced.

## 3. Milestone table

M1–M3A were delivered under the archived phase convention and are recorded, not re-planned.
M-F1a is the product walking skeleton for the new convention. “Now” identifies the first
roadmap outcome only; it is not authorization to open requirements or begin development.

| M# | slug | Outcome (measurable) | Horizon | Traces (CAP-*/QA-*) | Depends on | What ships (deployable) | Status |
|----|------|----------------------|---------|---------------------|------------|-------------------------|--------|
| M1 | `phase-1` | Exact mixed-state evolution and ordered noisy-circuit execution with the delivered local depolarizing, amplitude-damping, and phase-damping channels reproduce the external reference | Delivered | CAP-001, CAP-002 · QA-002, QA-009 | — | C++ core, Python bindings, tests, and Aer comparisons — [`archive/phases/phase-1/`](../density_matrix_project/archive/phases/phase-1/) | Delivered |
| M2 | `phase-2` | The density backend evaluates exact noisy energy in the frozen XXZ/HEA VQE workflow at 4/6/8/10 qubits with machine-checked support boundaries | Delivered | CAP-005, CAP-007 · QA-002, QA-005, QA-008, QA-009 | M1 | Backend selection, exact energy path, bridge metadata, and workflow evidence — [`archive/phases/phase-2/`](../density_matrix_project/archive/phases/phase-2/) | Delivered |
| M3 | `phase-3` | Noisy circuits became planner inputs and execute through a partitioned runtime with exact unitary-island fusion; 34 cases were counted, 0/6 representative cases passed the positive threshold, and 6/6 closed through diagnosis | Delivered | CAP-001, CAP-003, CAP-007 · QA-001, QA-005, QA-008 | M2 | Noisy planner, descriptors, runtime, and correctness/performance evidence — [`archive/phases/phase-3/`](../density_matrix_project/archive/phases/phase-3/) | Delivered |
| M3A | `phase-3-1` | Exact strict/hybrid channel-native fusion closed as a bounded decision study: 17/26 baseline-sufficient, 9/26 genuinely channel-native but not yet justified, and 0/26 justified at hybrid speedup ≥1.2× | Delivered | CAP-001, CAP-003, CAP-007 · QA-001, QA-002, QA-004, QA-005, QA-008 | M3 | Bounded channel-native modes and the frozen 26-row historical decision bundle — [`archive/phases/phase-3-1/`](../density_matrix_project/archive/phases/phase-3-1/) | Delivered |
| **M-F1a** | **`exactness-reconfirmation`** | **Product walking skeleton.** On the current revision, 100% of counted cases for every shipped advertised route meet QA-001 against the sequential oracle at 4–10 qubits, all evaluation-mode partitions are route-labelled, and the bundle regenerates with zero state-vector regressions | **Now** | CAP-001, CAP-007 · QA-001, QA-005, QA-008, QA-009 | M3A outcome | Revision-pinned correctness bundle and reusable exactness fitness lane; current-state docs updated, not recreated | Admin close recorded 2026-10-06 (`completeness_claim` true for handoff, `1b123a9a`); horizon unchanged; this revalidation authorizes no handoff |
| M-F5a | `cpp-python-interop-profile` | Shipped (closeout `918a73a4`). Met at 4, 6, and 8: E-VQE QA-007 verdict and three timed attribution routes with the R-strict refusal row (ADR-F5A-011). REQ-009 (checklist G-08) closed per CLOSEOUT and RM ACCEPT 2026-10-08. UB (one-sided 95% upper bound on O, E-VQE bundles) ≤1.06% at 4/6/8: 1.0589% / 0.2149% / 0.0744% | Delivered | CAP-004, CAP-007 · QA-007, QA-008, QA-009 | M-F1a outcome | Interop harness and six counted bundles; R-strict refusal row; no reduction. Closeout `CPP_PYTHON_INTEROP_PROFILE_CLOSEOUT.md` | Shipped |
| M-F1b | `committed-timings-record` | Existing timing evidence is committed with revision, host, method, route, and claim boundary pinned; regenerable current rows are separated from historical rows, and no row retcons or merges the frozen 26-case matrix with the Phase-3 millisecond table | Next | CAP-004, CAP-007 · QA-008 | M-F5a outcome | Validated timing record with provenance and separate historical annotations; no speedup claim | Draft (held: not started until Zoltán approves via PhD Manager) |
| M-F2 | `hybrid-cost-model` | On the hybrid evaluation route only, 100% of partitions record predicted and executed cost/route; held-out prediction meets a threshold frozen in requirements, predicted-cost skips are labelled baseline routes, every executed route remains exact, and zero silent substitutions occur | Next | CAP-003, CAP-007 · QA-001, QA-004, QA-005, QA-008, QA-009 | M-F1a protocol and M-F1b timing record | Execution-cost record, calibrated hybrid-only model, held-out validator, and auditable route/skip policy | Draft |
| M-F-CE | `workload-driven-channel-expansion` | At least one workload-justified local channel beyond the delivered depolarizing, amplitude-damping, and phase-damping set—beginning with GAD where requirements confirm it—meets QA-001/002/003/010 on every advertised path; all other paths reject it at preflight | Next — keep-list endpoint | CAP-001, CAP-002, CAP-007 · QA-001, QA-002, QA-003, QA-005, QA-008, QA-009, QA-010 | M-F2 route surface | Admitted channel, strict refusals, published parameter convention, analytical and Aer calibration, and module-readiness closeout | Draft |
| M-F3 | `reuse-heavy-layered-evaluation` | On the new pre-registered `reuse_heavy_layered_v0` primary two-qubit matrix, every cell is classified under QA-006; a justified cell requires genuine channel-native execution, sequential-oracle exactness, and median-of-three hybrid speedup ≥1.2× versus Phase-3 fused; zero justified cells is valid | Later — deferred stretch | CAP-003, CAP-007 · QA-001, QA-004, QA-005, QA-006, QA-008, QA-009, QA-012 | Keep-list closed; M-F2 model; M-F1b baseline provenance | Pre-registration, new matrix, route-attributed classification bundle, and positive or zero-cell diagnosis | Deferred hypothesis |
| M-F4 | `strict-s3-paper-operations` | On `phase31_bounded_mixed_motif_s3_v0`, strict-only paper operations with \(|S_M|=3\) either fuse as exact ordered Kraus bundles or hard-fail eligibility; every counted route meets QA-001/004 and supports above three are refused | Later — deferred stretch after M-F3 | CAP-001, CAP-003, CAP-007 · QA-001, QA-003, QA-004, QA-005, QA-008, QA-009 | M-F3 precedes it for sequencing only; no justified two-qubit cell is required | Strict three-qubit paper-operations surface, refusal boundary, sentinels, and exactness bundle | Deferred hypothesis |
| M4 | `canonical-attributed-energy` | The previously proposed canonical attributed-energy outcome remains available for future revalidation but is not part of the active Fall sequence | Parked | CAP-001, CAP-003, CAP-005, CAP-007 · QA-001, QA-002, QA-005, QA-008, QA-009 | Explicit future roadmap revalidation | Nothing ships while parked | Parked — inactive; not a start |

## 4. Per-milestone detail

### M-F1a — `exactness-reconfirmation` (Now; not authorized to start)

**Why first and measure.** Reconfirm the semantic baseline before any performance statement.
The q4 partitioned route is the slice tracer; the completed outcome covers all advertised
shipped routes at 4–10 qubits under QA-001 and QA-009, with named lanes and route labels.

**Scope and risk.** In: exactness, physical-state checks, route attribution, regeneration, and
current-state documentation. Out: new channels, timing, cost selection, and optimizer changes.
The riskiest assumption is A6: the sequential executor remains a trustworthy oracle and all
shipped paths still agree. Any disagreement freezes downstream claims. Handoff slug:
`exactness-reconfirmation`; update both current-state docs if authorized and delivered.

### M-F5a — `cpp-python-interop-profile` (Delivered)

**Recorded outcome (closeout status Shipped).** Met at 4, 6, and 8: E-VQE QA-007 verdict and three timed attribution routes with the R-strict refusal row (ADR-F5A-011). REQ-001…REQ-008 are closed. REQ-009 (checklist G-08) is closed per CLOSEOUT and RM ACCEPT 2026-10-08. UB (one-sided 95% upper bound on O, E-VQE bundles) ≤1.06% at 4/6/8: 1.0589% / 0.2149% / 0.0744%. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. Exactness (RM N-c): ≤1.2e-16 vs the sequential reference (w4 data only) and ≤5e-16 vs Qiskit Aer 0.17.2. Apply time versus R-base at widths 6 and 8: fused 3.27× and 3.43×, hybrid 6.57× and 8.82×. Current implementation cost, not intrinsic cost; no speed claim. C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08).

**Scope and risk.** In: the measurement above. Out: a reduction (none was made), kernel rewrites,
fusion redesign, AVX, GPU, and optimizer changes. A4 closed false; CAP-004 is hold-the-line.
Handoff slug: `cpp-python-interop-profile`. `ARCHITECTURE_OVERVIEW.md` and `TECH_STACK.md` name the
lane in this revalidation. Horizon cell is Delivered; the status cell stays Shipped.

### M-F1b — `committed-timings-record` (Next; held: not started until Zoltán approves via PhD Manager)

**Why next and measure.** Commit the evidence that already exists after interop attribution,
without changing what historical rows meant. Current claim-bearing rows must regenerate under
QA-008; historical numbers stay explicitly historical and non-counted if they cannot.

**Scope and risk.** In: provenance, validators, claim boundaries, and a separately identified
Phase-3 fused baseline record. Out: new speedup experiments and reinterpretation of the frozen
26-case result. The ≈95 ms per two-qubit term versus ≈15 ms per sequential operation sentence,
if retained, describes only the shipped full-dimension apply at 10 qubits. Handoff slug:
`committed-timings-record`; update `TECH_STACK.md` only if an evidence lane is added.

### M-F2 — `hybrid-cost-model` (Next)

**Why next and measure.** Make hybrid route choice auditable before adding another channel. The
model separates construction, application, and route/lookup cost, records predicted versus
executed route on every partition, and validates on held-out cases under a pre-registered error
threshold. It may select a labelled baseline skip; a skip is not channel-native execution.

**Scope and risk.** In: hybrid evaluation mode, cost records, a calibrated model, labelled skips,
and literature-positioning evidence. Out: strict-mode widening, caching, a speedup verdict, and
an R=1 or sequential-cost-ratio selection rule. The riskiest assumption is that held-out hybrid
cost is predictable without silent substitution. Handoff slug: `hybrid-cost-model`; update both
current-state docs at close.

### M-F-CE — `workload-driven-channel-expansion` (Next; keep-list endpoint)

**Why next and measure.** Expand beyond the three delivered local channels only after exactness,
interop, timing provenance, and hybrid route attribution are established. GAD is the named
example, not a claim that amplitude damping is new. Every advertised path must meet
QA-001/002/003/010; every other path refuses the channel before mutation.

**Scope and risk.** In: one justified channel, explicit parameter conventions, analytical and
Aer witnesses, strict refusal, and regenerable evidence. Out: speculative breadth, legacy API
expansion, training studies, and a promised fusion win. The riskiest assumptions are A5 and A6.
Handoff slug: `workload-driven-channel-expansion`; update both current-state docs at close.

### Deferred stretch boundary

M-F3 is the first use of the written QA-006 plan and runs only on the new
`reuse_heavy_layered_v0` matrix. M-F4 follows it but is a separate strict-only,
paper-operations-only \(|S_M|=3\) exactness outcome on
`phase31_bounded_mixed_motif_s3_v0`; it is not a hybrid widen and QA-006 is not its success test.
Designing or opening M-F4 does not require a previously justified two-qubit cell. No speedup
measurement is in M-F4 scope.

## 5. Sequencing rationale

- **Exactness before performance.** M-F1a re-establishes the oracle contract at the current
  revision before interop, timing, prediction, or speedup language.
- **Measure before committing the record.** M-F5a separates language-boundary overhead from
  kernel and planner/runtime cost; M-F1b then records existing timings with their real claim
  boundaries.
- **Attribute before expanding.** M-F2 gives hybrid decisions predicted-versus-executed cost
  and route labels before M-F-CE adds a channel to that surface.
- **Stop the keep-list at channel expansion.** This semester delivers the verified,
  fusion-enabled module for later noisy VQA loops. It does not perform those loops or infer
  trainability.
- **Keep stretch hypotheses falsifiable.** M-F3 may honestly return zero justified cells.
  M-F4 proves a strict three-qubit object and does not borrow M-F3's speedup test.
- **Keep scope cuts visible.** Q3 is parked, E7 dropped, and AVX/GPU excluded. M4 stays parked.

The semester outcome is achieved only when every keep-list milestone meets its own stated
measure and its counted evidence regenerates. A diagnosis may close a research question, but an
unmet milestone measure is reported as partial rather than redefined. The outcome claims module
readiness only—never a VQA campaign, trainability result, full-scale run, priority result, or
promised speedup.

## 6. Assumptions, risks & revalidation log

### Strategic assumptions

| Assumption | Validated by | Consequence if it fails |
|------------|--------------|-------------------------|
| A2 A genuine channel-native route can lower hybrid time on a reuse-heavy family | M-F3 under QA-006, after the keep-list | Report zero justified cells and reject the hypothesis for that family; module readiness remains intact |
| A3 Real traces contain enough parameter-independent reuse to amortize construction | M-F3 under QA-012 | Drop the reuse claim; retain only exact, attributable routing |
| A4 Interop overhead materially limits iterative evaluation | Closed false by M-F5a on the E-VQE equal-work cells. CAP-004 hold-the-line | Treat CAP-004 as hold-the-line and focus reporting on kernel/planner cost |
| A5 Expansion through GAD covers the next justified comparison | M-F-CE | Revisit CAP-002 through Ask-first; do not add speculative breadth |
| A6 The sequential executor is a trustworthy oracle | M-F1a and every new channel/path | Freeze downstream claims until disagreement is resolved |
| A7 Researchers accept explicit rejection over silent convenience | M-F-CE refusal surface | Improve diagnostics or explicit transforms; never restore silent behavior |

A1a/A1b, A8, and A9 remain product-level destination assumptions. This Fall roadmap does not
schedule trainability studies or exact-gradient development to validate them.

### Roadmap risks

| Risk | Mitigation |
|------|------------|
| “Now” is mistaken for permission to implement | Header, status, and M-F1a detail state that no handoff or start is authorized |
| M-F1a drifts into performance work | Its success measure and scope contain no timing or selection |
| M-F1b turns historical evidence into a new claim | Separate regenerable counted rows from historical non-counted annotations; preserve the 17/9/0 boundary |
| M-F2 treats a model skip as a fused execution | Record predicted and executed route; skips remain labelled baseline routes |
| M-F3 reuses the historical matrix or wrong denominator | Freeze `reuse_heavy_layered_v0` and the Phase-3 fused baseline in pre-registration |
| M-F4 is read as a hybrid widening or speedup promise | Strict-only scope; QA-006 explicitly excluded as its success test |
| Representation language drifts to Liouville-primary | Kraus bundles primary; Choi/Liouville witnesses only |
| Competitor positioning becomes a novelty claim | TANQ-Sim remains PARTIAL / engine-fusion; no priority or consecutive-C1/C2 ownership claim |

### Revalidation log

- **2026-10-08 — M-F5a closeout `918a73a4`, status Shipped.**
  *Learned:* A4 closed false, as the product statement expected; CAP-004 is hold-the-line and there is no reduction. UB (one-sided 95% upper bound on O, E-VQE bundles) ≤1.06% at 4/6/8: 1.0589% / 0.2149% / 0.0744%. QA-007 bar frozen at 10% (product-statement default), RM ALIGN 2026-10-07, ratified by Zoltán as product owner 2026-10-08. R-strict is a refusal row with diagnosis `channel_native_noise_presence` (ADR-F5A-011), not a timed route. Exactness (RM N-c): ≤1.2e-16 vs the sequential reference (w4 data only) and ≤5e-16 vs Qiskit Aer 0.17.2. Apply time versus R-base at widths 6 and 8: fused 3.27× and 3.43×, hybrid 6.57× and 8.82×. Current implementation cost, not intrinsic cost; no speed claim. C2 (strict-capable side workload): backlog, possible post-supervisor item, not started; not in M-F5a (Zoltán via PhD Manager and RM, 2026-10-08). No core assumption was invalidated, so `PRODUCT_STATEMENT.md` is not escalated.
  *Changed:* M-F5a status is Shipped, the CLOSEOUT's own wording. REQ-009 (checklist G-08) is
  closed per CLOSEOUT and RM ACCEPT 2026-10-08. Current-state docs
  name the lane. M-F1a status cell records the 2026-10-06 admin close; its horizon stays. The
  step that promotes a later milestone to Now was not run. No later keep-list row was edited.
- **2026-10-04 — draft v0.8 aligned to product statement v0.5; no milestone closure.**
  *Learned:* the locked Fall outcome is module readiness and the active sequence ends at channel
  expansion; the ≥1.2× rule is a later falsifiable research plan, not a promised result.
  *Changed:* replaced v0.7's M4-led program with the keep-list M-F1a → M-F5a → M-F1b → M-F2 →
  M-F-CE; parked M4 and Q3; dropped E7; deferred M-F3 then strict-only M-F4; separated the
  historical 26-case matrix from the Phase-3 fused baseline and millisecond evidence; removed
  VQA-campaign, R=1, sequential-cost-ratio, GPU, and priority framing.
- **2026-09-20 to 2026-09-23 — v0.1–v0.7 (superseded sequence).** Recorded the delivered Phase
  1–3.1 outcomes and explored an M4-led research program; retained only evidence that remains
  consistent with product statement v0.5.

### Critique verdict and stakeholder checkpoint

The alignment critique found and corrected the active-order, baseline, speedup-rule,
channel-inventory, representation, competitor-positioning, and semester-outcome conflicts in
v0.7. Open validation remains: milestone outcomes and `[confirm]` thresholds must be accepted by
the product owner before requirements are opened. M-F5a is recorded Shipped at closeout
`918a73a4`. This revalidation does not authorize implementation, a Tech Lead start, a push, or a
pull request, and it does not promote a later milestone to Now.

> Is the keep-list sequence and each milestone's measurable outcome correct before any milestone
> is authorized to open?
