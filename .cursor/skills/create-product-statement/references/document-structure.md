# `PRODUCT_STATEMENT.md` — required structure and change policy

Read while drafting or revising the product statement. Use this section order.

## Contents

- Section-by-section structure
- Vision templates
- Maintenance and change policy

## Section-by-section structure

**1. Vision** — one or two sentences. See the templates below.

**2. Customers & problem** — target users and personas; jobs-to-be-done; the core
problem; the today → future transformation.

**3. Value proposition & differentiation** — the value delivered and why it beats the
existing alternatives.

**4. Product capabilities (`CAP-*`)** — the high-level, durable requirements. Keep
roughly 5–12. Each one:

- **Id:** CAP-[nnn]
- **Capability:** [durable outcome the product must enable]
- **Why / value:** [the change it creates for the customer or business]
- **Success signal:** [measurable indicator that the capability is delivering value]

**5. Quality attributes (`QA-*`)** — cross-cutting quality bars written as scenarios:

- **Id:** QA-[nnn] — *When [stimulus] under [condition], the product shall [response]
  within/at [response measure].*
- Cover what matters for the domain: security and privacy, reliability and availability,
  performance and scale, auditability and compliance, accessibility, operability. In
  regulated domains treat compliance and auditability as first-class.
- Every `QA-*` needs a **response measure**, because downstream it becomes an automated
  fitness function and an evidence-matrix row. A `QA-*` with no number is not testable.

**6. Ubiquitous language (seed glossary)** — core domain terms with crisp definitions.
Optionally list candidate domain areas or bounded contexts at a high level only, with no
architecture.

**7. Guardrails & constraints** — product-level **Always do / Ask first / Never do**, plus
fixed organizational standards, regulatory and compliance constraints, and other
non-negotiables.

**8. Strategic assumptions & risks** — the riskiest assumptions, their current validation
status, and **kill criteria**: what would force a rethink.

**9. Out of scope / non-goals** — the product-level negative space; what this product
deliberately will not be.

Keep the vision tight and put the density into capabilities, quality attributes, and
guardrails. Budget the whole document to about 300 lines; push detail downward into the
roadmap and milestones rather than upward into the North Star.

## Vision templates

- *For [target customer] who [need], [product] is a [category] that [key benefit]. Unlike
  [alternatives], it [key differentiator].*
- *We believe in a world where [target] can [goal] by/with [differentiator].*

Test the result: does it help decide what **not** to build? If not, sharpen it.

## Maintenance and change policy

Treat the product statement as durable. Update it only when:

- a deliberate strategic pivot occurs, or
- roadmap revalidation after a milestone surfaces evidence that invalidates a core
  assumption or capability.

When changing it, record what changed and why in a short **Change log** entry at the
bottom of the file, preserve existing `CAP-*` / `QA-*` ids (extend rather than renumber,
and mark a retired id superseded with a one-line reason), then re-run
`create-product-roadmap` revalidation so the roadmap realigns with the new North Star.

Never renumber a `CAP-*` or `QA-*`: milestones, `REQ-*` requirements, evidence matrices,
and fitness functions all cite those ids.
