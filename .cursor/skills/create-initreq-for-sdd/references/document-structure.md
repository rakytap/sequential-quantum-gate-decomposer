# `INITIAL_REQUIREMENTS.md` — required structure

Read while drafting a milestone's requirements baseline. Use this exact section order.
Budget the whole document to about 300 lines: keep the vision section brief and put the
density into requirements, acceptance, and boundaries.

## Contents

- 1. Milestone scope and vision
- 2. User journeys
- 3. Ubiquitous language
- 4. Requirements and acceptance criteria
- 5. Non-functional requirements
- 6. Operational boundaries
- 7. Assumptions and open questions

## 1. Milestone scope and vision

- **Milestone:** [`M#` and `milestone-slug` from the roadmap]
- **Upstream traceability:** [`CAP-*`/`QA-*` this milestone advances; the milestone outcome
  and its measure]
- **In scope:** [what this milestone includes]
- **Out of scope:** [explicit non-goals for this milestone]
- **Target users:** [roles or personas]
- **Core problem:** [what pain or obligation this addresses]
- **Success metrics:** [measurable signals, aligned with the roadmap outcome]
- **Current-state context:** [`ARCHITECTURE_OVERVIEW.md` / `TECH_STACK.md` constraints, or
  "to be established by this milestone"]

The milestone id must appear here; downstream linting looks for it in the header.

## 2. User journeys (narrative, lightweight)

- **Primary journey:** [step by step from the user's perspective]
- **Key alternate paths:** [empty state, permission denied, retry, degraded mode]

## 3. Ubiquitous language (milestone glossary)

Domain terms used in this milestone, inherited and extended from the product statement
glossary, one crisp definition each. Use these exact terms in requirements, acceptance,
tests, and code so meaning stays consistent. Divergent wording downstream is a design
defect, not a style choice.

## 4. Requirements and acceptance criteria

Break the work into small, independently verifiable requirements. Each gets a stable id.

- **Id:** REQ-[nnn]
- **Upstream:** [`CAP-*`/`QA-*` this requirement serves]
- **Intent (optional user-story phrasing):** As a [role], I want [capability] so that
  [outcome]. *(Phrasing only — the traced unit is the requirement, not an SDD delivery
  story.)*
  - **Acceptance (BDD):** Given [context], when [event or action], then [observable
    outcome].
  - **Acceptance (EARS, optional):** "While [state], the system shall [behavior]", "When
    [trigger], the system shall [response]", "If [condition], then the system shall
    [behavior]", "Where [feature is included], the system shall [behavior]".

Every requirement needs at least one negative or error scenario. Acceptance must be
precise enough to seed the downstream evidence matrix, which pairs each `REQ-*`/`QA-*` with
a test type, a command or CI gate, and an expected result.

## 5. Non-functional requirements

State only what matters for this milestone. Where an NFR realizes a product `QA-*`, cite
it and keep the **response measure**, so it can become a testable fitness function.

- **Performance / scale:** [latency, throughput, limits] → [QA-*]
- **Security / privacy:** [authn/authz, PII, secrets, logging redaction] → [QA-*]
- **Reliability / observability:** [errors, metrics, tracing] → [QA-*]
- **Auditability / compliance:** [traceability, retention, regulatory obligations] → [QA-*]
- **Accessibility / i18n:** [if applicable] → [QA-*]
- **Compatibility:** [browsers, APIs, versions — known constraints only]
- **Architecture / tech-stack documentation:** [whether this milestone must create or
  update `ARCHITECTURE_OVERVIEW.md` / `TECH_STACK.md`]

## 6. Operational boundaries (three tiers)

- **Always do:** [run the hermetic suite before claiming done; match existing code style]
- **Ask first:** [destructive migrations, new dependencies, public API changes]
- **Never do:** [commit secrets; bypass auth; disable security checks]

These reduce harmful autonomy. They complement acceptance criteria; they do not replace
them.

## 7. Assumptions and open questions

- **Assumptions:** [listed explicitly]
- **Open questions:** [decisions deferred to product or to a later milestone]

Open questions are the handoff into `spec-driven-development`'s pre-implementation
checklist, so write each one as something that can be closed by a decision.
