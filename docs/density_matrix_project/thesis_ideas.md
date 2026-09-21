# Thesis Ideas: Exact Noisy Variational-Circuit Research

## Feasibility as a PhD Basis

The narrowed novelty can support a PhD thesis, but the thesis should not claim that SQUANDER
originated CPTP-channel composition or noisy-operation fusion. Existing frameworks, including
Qiskit Aer, already provide related capabilities.

The thesis should instead focus on new, transferable knowledge about when exact noisy-circuit
optimization is beneficial and what it enables researchers to learn about variational training
under realistic noise.

## Candidate Thesis Statement

> This thesis develops and evaluates representation-aware methods for exact simulation of noisy
> variational quantum circuits, establishing when channel-aware partitioning, fusion, and reuse
> reduce end-to-end computational cost, and using the resulting exact backend to characterize
> how physically distinct noise processes affect variational trainability.

## Potential Contributions

### 1. Representation-Aware Selective Fusion

- Develop a cost model and planner that predict when exact noisy fusion is beneficial.
- Account explicitly for channel representation and rank, qubit support, construction cost,
  parameter dependence, memory use, and observed reuse.
- Provide a no-regression selection rule that rejects fusion when predicted or measured cost
  exceeds sequential execution.
- Formulate results so they transfer beyond one simulator implementation.

### 2. Break-Even Analysis for Variational Workloads

- Determine which noisy motifs or channel components remain reusable as variational parameters
  change.
- Separate per-evaluation fusion benefit from amortization through repeated use.
- Measure construction cost, application cost, cache behavior, memory overhead, and the observed
  reuse horizon.
- Produce break-even laws or predictive models rather than only SQUANDER-specific timings.

### 3. Scientific Noisy-VQA Studies

- Compare unital noise with non-unital and finite-temperature noise.
- Study gradient statistics, convergence, optimization quality, and trainability.
- Cover multiple independently motivated task and ansatz families.
- Use multiple seeded initializations and pre-registered size, depth, and noise sweeps.
- Report effect sizes, uncertainty intervals, finite-size limitations, and null results.

### 4. Exact and Reproducible Reference Methodology

- Preserve sequential `NoisyCircuit` execution as the internal exact reference.
- Validate newly advertised channels and paths against Qiskit Aer and analytical models.
- Keep channel conventions, tolerances, cost policies, seeds, build conditions, and claim
  boundaries explicit.
- Regenerate all claim-bearing results through named evidence pipelines.

This reproducibility framework supports the thesis but should not be presented as its sole
scientific novelty.

## Relationship to Phase 4

Phase 4 can provide a strong methods foundation and a substantial thesis chapter through:

- physically calibrated channel expansion, beginning with generalized amplitude damping;
- representation-aware partitioning and selective channel fusion;
- controlled profiling of C++/Python and dispatch overhead;
- analytical, sequential, and external-reference validation;
- and evidence-based positioning relative to existing frameworks.

Phase 4 alone is unlikely to constitute the complete PhD. It should make the backend ready for
the later scientific studies that test noisy-training and trainability hypotheses.

## Main Scientific Risk: Exact-Regime Scale

Exact dense simulation at approximately 4–10 qubits, with 12 qubits as a possible stretch, may
support mechanistic finite-size conclusions. It may not be sufficient to establish asymptotic
barren-plateau or trainability scaling.

Before making a scaling claim:

- define the scientific estimand;
- perform a finite-size sensitivity or power analysis;
- test whether the exact regime distinguishes the competing hypotheses;
- and label results as finite-size observations when it does not.

If the required effect is not identifiable in the exact regime, the research may need a later
trajectory, tensor-network, MPDO, or accelerated-backend branch validated against the exact
reference on overlapping problem sizes.

## Positive and Negative Research Outcomes

### If Selective Fusion Succeeds

The thesis can contribute a new representation-aware optimization method, validated break-even
model, and evidence that it increases the experimental reach of exact noisy-VQA studies.

### If Selective Fusion Does Not Succeed

The negative result may still support a methods paper if it yields:

- a predictive no-regression selector;
- a rigorous explanation of where fusion loses;
- transferable limits on channel representation, support, parameter dependence, and reuse;
- and a reproducible benchmark methodology.

However, a negative fusion result alone is unlikely to carry the full thesis. The principal
positive contribution would then need to come from noisy-VQA science, a validated scaling
method, or another general optimization result.

## Minimum Bar for a Strong Thesis

- A method or predictive model that generalizes beyond one implementation.
- Evaluation across multiple workload, ansatz, and noise families.
- Statistical treatment of performance and scientific outcomes.
- Explicit separation of finite-size observations from scaling claims.
- At least one substantive scientific conclusion about noisy variational training.
- Reproducible artifacts that independently support every major claim.

