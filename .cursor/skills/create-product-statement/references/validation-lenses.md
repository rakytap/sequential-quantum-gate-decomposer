# Validation lenses for a product statement

Read during the validation phase. Run at least one lens and record the result inline in
`PRODUCT_STATEMENT.md`, so the reasoning survives the session.

## PR-FAQ (Amazon working-backwards)

Write a one-page simulated launch press release, then an FAQ that answers the hardest
customer, stakeholder, and skeptic questions.

The press release forces customer-benefit clarity: if the benefit cannot be stated as
news, the value proposition is still vague. The FAQ surfaces feasibility risk early —
include the questions you would rather avoid (what does this cost, what breaks, who says
no, what happens at scale, what does the regulator ask).

## Lean Canvas (early-stage)

Fill: Problem, Customer Segments, Unique Value Proposition, Solution (kept high-level),
Channels, Revenue, Cost, Key Metrics, Unfair Advantage.

Keep Solution deliberately thin. A canvas whose Solution box is the longest one has become
a design document.

## Assumption / risk map

List the riskiest assumptions with, for each: what it asserts, why the product fails
without it, current validation status, how it will be validated (and by which milestone),
and its **kill criterion** — the observation that would force a rethink.

Order by "how much scope dies if this is wrong", not by likelihood. The top item should be
what the first milestone is designed to test.

## Adversarial critique pass

Before the user checkpoint, attack the draft and resolve or explicitly record every
finding. Capture the outcome as a short inline **Critique** note in the artifact.

- **Vision:** does it decide anything? Name two plausible features it rules out. If it
  rules out nothing, it is a slogan.
- **Capabilities:** which `CAP-*` is a feature in disguise (an output, not an outcome)?
  Which has no success signal?
- **Quality attributes:** which `QA-*` has no response measure, and therefore cannot
  become a fitness function downstream? Add a number and a measurement method, or mark it
  `[confirm]` with an owner.
- **Differentiation:** what does the strongest alternative do better? Is the claimed
  advantage durable or a head start?
- **Assumptions:** which assumption has no validation milestone and no kill criterion?
- **Scope creep upward:** what solution design, stack choice, or schema has crept into the
  statement? Move it out — it belongs in an ADR, and in the current-state docs once true.
- **Coverage:** is any capability unreachable by any plausible near-term milestone?

## Stakeholder checkpoint

After the critique pass, ask:

> Does this capture the durable intent of the product? Which capabilities, quality bars,
> or assumptions are wrong, missing, or too vague to guide a roadmap?

The critique pass complements stakeholder validation; it does not replace it.
