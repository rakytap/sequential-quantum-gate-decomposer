# Task 5 Mini-Spec: Publication and Claim Boundary

**Phase planning traceability:** `DETAILED_PLANNING_PHASE_3_1.md` §8, Task 5.

**Pre-implementation checklist traceability**

| P31-C row | Role in this task |
|-----------|-------------------|
| **P31-C-01–C-09** | **Consume (read-only):** papers must not assert beyond what the closed checklist and emitted bundles support. This task does not substitute for closing C-rows; it **aligns prose to frozen contracts and evidence**. |
| **P31-C-08** | **Traceability:** abstract/paper cite the same artifact types, schema versions, and case ID tables as bundles. |

Upstream: Tasks 3–4 deliverables, `PUBLICATIONS.md` Phase 3.1 / Side Paper A, `P31-ADR-002`, `P31-ADR-005`.

---

## Scientific outcome

A reviewer can trace **each claim** in the Phase 3.1 short paper, narrative, abstract, and full paper skeleton to **specific case IDs, baselines, and thresholds**—or see an explicit **limitations** paragraph where evidence is absent. Phase 3 Paper 2 wording remains untouched except for additive cross-references.

The primary publication target is now a **bounded positive-methods paper** on
the frozen v1 slice. The Side Paper A decision-study question remains
important, but mainly as the check that the stronger methods claim is justified
by evidence rather than assumed.

---

## Given / When / Then

- **Given** frozen **P31-C-01–C-09** and emitted correctness/performance bundles (or a documented decision to ship a **negative-result** narrative with partial bundles).
- **When** publication docs are revised for Phase 3.1 submission or internal review.
- **Then** every methods or results sentence in `SHORT_PAPER_PHASE_3_1.md`, `SHORT_PAPER_NARRATIVE.md`, `ABSTRACT_PHASE_3_1.md`, and `PAPER_PHASE_3_1.md` maps to evidence or is labeled speculation/limitation; Side Paper A’s guiding question (quoted below) has an explicit answer or a scoped “not addressed by current evidence” with limitations.

---

## Assumptions and dependencies

- Evidence artifacts exist or their absence is itself the documented outcome (benchmark-grounded non-proceed decision per planning §9).
- Program editors align with `PUBLICATIONS.md` positioning.

---

## Required behavior

- **Claim boundary** subsection in technical short paper: in-scope contributions, explicit non-goals, what Phase 3 did vs Phase 3.1.
- The technical short paper presents Phase 3.1 **first** as a bounded
  positive-methods contribution on the frozen v1 slice; if the evidence
  does not support that framing, the docs must explicitly fall back to a scoped
  negative-result / decision-study narrative rather than quietly overstating the
  method.
- **Traceability table** (in `DETAILED_PLANNING_PHASE_3_1.md` or paper appendix): claim → case ID / artifact → threshold.
- Narrative doc explains **why** the phase matters without overstating performance (`P31-ADR-005`).
- **Side Paper A alignment:** at least one subsection or paragraph in the technical short paper (or answer map) explicitly addresses the verbatim question under **Acceptance evidence** using evidence-backed language.

---

## Unsupported behavior

- Strengthening Phase 3 claims retroactively (`P31-ADR-002`).
- Performance superlatives without **P31-C-06** row citations.
- Correctness language without **P31-C-04** / **P31-C-03** alignment.

---

## Acceptance evidence

- **Side Paper A question** (`PUBLICATIONS.md`, § “Side Paper A: Superoperator Or Channel-Native Fusion Decision Study”), quoted verbatim—each Phase 3.1 paper surface must tie its contribution to an answer, partial answer, or honest negative result relative to this question:

  > When does the Phase 3 noise-aware partitioning baseline stop being enough, and when would more invasive channel-native fusion become justified?

- A short **answer map** (in the technical short paper or in `DETAILED_PLANNING_PHASE_3_1.md` traceability): for that question, which claims are supported by which case IDs / bundles (`P31-C-04`, `P31-C-06`, `P31-C-08`), and what remains open.
- Checklist pass: independent read-through (self or colleague) with mark-up resolved.
- Version notes in paper headers or CHANGELOG entry pointing to bundle hashes or CI run IDs if the project uses them.

---

## Affected interfaces

- `docs/density_matrix_project/phases/phase-3-1/SHORT_PAPER_PHASE_3_1.md`
- `docs/density_matrix_project/phases/phase-3-1/SHORT_PAPER_NARRATIVE.md`
- `docs/density_matrix_project/phases/phase-3-1/ABSTRACT_PHASE_3_1.md`
- `docs/density_matrix_project/phases/phase-3-1/PAPER_PHASE_3_1.md`
- Optional: `PLANNING.md` / `CHANGELOG.md` one-line pointers if program policy requires.

---

## Publication relevance

- This task **is** the publication relevance slice; success is reviewer-defensible alignment between prose and frozen contracts plus bundles.
