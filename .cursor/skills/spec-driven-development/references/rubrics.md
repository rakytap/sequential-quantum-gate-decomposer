# Rubrics, principles, and anti-patterns

Read before treating a milestone as implementation-ready, before treating a slice as
code-ready, or when reviewing someone else's spec.

## Contents

- Detailed milestone planning rubric
- Layer 2 mini-spec rubric
- Pre-implementation completion checklist rubric
- Spec-driven principles
- Working principles
- Anti-patterns

## Detailed milestone planning rubric

`DETAILED_PLANNING_<MILESTONE_SLUG>.md` should include:

- purpose / mission and the source-of-truth hierarchy for this effort;
- traceability matrix or equivalent (`REQ-*` or goals → interpretation in this milestone);
- in-scope / out-of-scope and non-goals;
- assumptions and success conditions;
- **frozen contracts** the implementation must not contradict — interfaces,
  compatibility, error policy, performance or sizing expectations where relevant;
- current-state doc status: whether `ARCHITECTURE_OVERVIEW.md` / `TECH_STACK.md` are
  present, current, and affected by this milestone;
- an architecture boundary map when the milestone spans multiple contexts, modules, data
  owners, external systems, or deployment boundaries;
- the work-package breakdown as **goals** — what success looks like and what evidence
  will show done, not a code recipe;
- milestone acceptance criteria, risks, and decision gates;
- expected outcome.

Prefer no code snippets in milestone planning, except stable interface sketches once the
interfaces exist and the team allows them.

## Layer 2 mini-spec rubric

- required and unsupported behavior;
- acceptance evidence tied to milestone criteria;
- affected interfaces, marked breaking or additive;
- an evidence matrix whenever the work touches user behavior, public interfaces,
  boundaries, or `QA-*` fitness functions.

## Pre-implementation completion checklist rubric

The checklist should:

- map each open item to the contract or artifact that closes it;
- state a clear readiness verdict — implementation-ready or not;
- record closure decisions and their trade-offs;
- include a go / no-go rule for when implementation may begin;
- flag any deviation that will require a `CHANGE_CONTROL.md` governance sign-off, so it
  is tracked from planning through close;
- confirm that required boundary maps, ADRs, evidence-matrix entries, current-state doc
  updates, release/rollback expectations, and operational checks are in place for the
  first slice.

## Spec-driven principles

1. Define contracts, scope, and success criteria **before** implementation.
2. Separate required behavior from implementation choices where possible.
3. Maintain traceability from goals to decisions to tasks to evidence.
4. Treat unsupported and deferred cases as **documented** outcomes.
5. Keep work-package and task descriptions goal-oriented, not unnecessarily prescriptive
   about internals.

## Working principles

- **Human reviewability.** Specs stay short enough to read carefully. Verbosity is a
  defect unless it removes ambiguity.
- **Minimal but complete.** Capture what this slice needs. Avoid long chains of "and" and
  premature edge-case catalogs.
- **Meaningful decomposition.** Slices should be valuable on their own where possible.
  Use INVEST-style and MoSCoW thinking.
- **Context discipline.** Oversized specs drift. Keep the whole slice contract in working
  memory; split rather than append endlessly.
- **Subagents and tools** multiply good decomposition; they do not fix a vague or
  oversized spec.
- **Systems thinking.** Call out cross-feature interactions: load, permissions, retries,
  failure amplification.
- **Parallelism.** Clear interfaces and acceptance criteria are what let work run
  concurrently.

## Anti-patterns

- **Specification theater** — dense documents nobody reads or uses.
- **Premature comprehensiveness** — specifying far beyond the current slice before
  learning anything.
- **Vibe implementation** — coding without acceptance criteria, then retrofitting the
  justification.
- **Spec–implementation drift** — behavior changed but the spec did not.
- **AI spec bloat** — verbose generated specs without curation.
- **Tool obsession** — debating format instead of fixing slice size and ambiguity.
- **Deferring cross-work-package decisions to mini-specs** when they belong in ADRs.
- **Full delivery-story splitting upfront** before any vertical slice is built and
  learned from.
- **Horizontal slicing** — building a whole layer instead of a thin end-to-end vertical
  slice that ships.
- **Quality attributes without fitness functions** — `QA-*` or NFRs asserted but never
  enforced by an automated check.
- **Evidence gaps** — a `REQ-*`, `QA-*`, or ADR decision with no named test command, CI
  gate, or observable verification path.
- **Boundary ambiguity** — code crossing contexts, ports, adapters, or data-ownership
  boundaries that were never mapped or decided.
- **Stale architecture docs** — shipped stack, commands, topology, integrations, or
  boundaries changed but `ARCHITECTURE_OVERVIEW.md` / `TECH_STACK.md` were not updated.
- **Language drift** — code, specs, and tests using different words than the ubiquitous
  language.
- **Oversized slice artifacts** — a mini-spec past its size budget is a slice that is too
  big, not a documentation habit.
- **Waiving a check instead of fixing the finding** — a waiver without a reason, or a
  waiver on an in-flight milestone.
- **Skipping milestone-close revalidation** — finishing a milestone without feeding
  learnings back to the roadmap.
