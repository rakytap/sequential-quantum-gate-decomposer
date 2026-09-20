# Architecture and design practices for SDD planning

Read when writing Layer 1 planning or an ADR, when a milestone spans more than one module
or context, or when deciding where a boundary belongs. These practices constrain the
*how* without bloating the spec: capture the decision in an ADR, capture the proof in
tests and fitness functions (see `references/practices-testing.md`).

## Contents

- Authoring refinements
- Domain-driven design
- Clean / hexagonal architecture
- Architecture boundary map
- Diagrams and decisions
- ADR content rubric

## Authoring refinements

1. **Contract-first, decision-fast.** Close cross-work-package decisions in ADRs early
   to limit drift.
2. **Behavior over mechanism.** Specify observable outcomes before internal structure
   unless constraints force otherwise.
3. **Testable acceptance language.** Each required behavior maps to reproducible
   evidence and a pass/fail criterion.
4. **Traceability by default.** Maintain requirement or goal → decision → task →
   evidence.
5. **Explicit non-goals and unsupported behavior.** Treat ambiguity as scope risk and
   document deferrals.
6. **Small vertical slices.** Delivery stories are behavioral slices and engineering
   tasks nest under them. Prefer one thin end-to-end slice that exercises the new
   contract over a broad horizontal refactor.

## Domain-driven design

- **Ubiquitous language:** reuse the product and milestone glossary verbatim in specs,
  tests, and code — types, modules, endpoints. Drift in language is a design defect, not
  a naming preference.
- **Strategic design:** identify bounded contexts and draw a context map at
  milestone/ADR level when more than one subdomain is in play. Align module and service
  boundaries to contexts, not to layers.
- **Tactical design:** model aggregates, entities, value objects, and domain events;
  keep invariants inside aggregates. Record non-obvious model choices as ADRs.

## Clean / hexagonal architecture

- Enforce the **dependency rule**: domain and use-cases depend on nothing outward.
  Frameworks, databases, UI, and external services sit behind ports with adapters, so
  the domain stays framework-agnostic and testable in isolation.
- A delivery slice is **vertical**: it crosses adapter → use-case → domain → adapter,
  not one horizontal layer.
- Promote stable, shipped boundary information from the milestone boundary map into
  `ARCHITECTURE_OVERVIEW.md` at milestone close. Rationale stays in the ADR;
  current-state facts go in the overview.

## Architecture boundary map

Add a lightweight boundary map to Layer 1 when the milestone spans more than one module,
bounded context, data owner, external system, or deployment boundary. Include:

- owned concepts and data;
- inbound and outbound ports;
- adapters;
- upstream and downstream dependencies;
- anti-corruption layers;
- domain events.

Link every major boundary choice to an ADR.

## Diagrams and decisions

Use the **C4 model** (context, container, component) at the level the decision needs, and
keep the diagram in or beside the ADR. Every cross-work-package architectural choice is
an ADR with its rejected alternatives recorded.

## ADR content rubric

An ADR entry includes:

- title and unique **ADR id**;
- **status**;
- **context**;
- **decision**;
- **rationale**;
- **consequences**;
- **rejected alternatives**;
- **upstream alignment and traceability** (which `REQ-*`, `CAP-*`/`QA-*`, milestone goal,
  or guardrail it serves).

If a decision affects multiple work packages or delivery stories, it belongs at milestone
level — an ADR or the planning document. If it affects only one work package, it may live
in that Layer 2 mini-spec. Never fragment a cross-cutting choice across mini-specs.
