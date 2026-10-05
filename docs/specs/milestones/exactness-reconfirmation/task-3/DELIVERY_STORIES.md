# Delivery stories — M-F1a slice 3 (Layer 3)

> **Status:** Closed code-ready under ADR-F1A-008 on 2026-10-05 by Squander Architect; Step 4b under ADR-F1A-010 · **Slice:** M-F1a slice 3 (Slice B) ·
> **Parent:** `TASK_3_MINI_SPEC.md` · **Milestone:** M-F1a `exactness-reconfirmation` ·
> **Traces:** REQ-004, REQ-006 · QA-008 · ADR-F1A-008, ADR-F1A-009 (+ Amendment 1),
> ADR-F1A-010 · **No push/PR**

Wording: baseline route verified for q4 history only. This slice does not close the
milestone. The regenerated q4 bundle is never committed.

## DS-B1 — Length-1 revision allowlist

**Stakeholder / system value**

- A later commit can regenerate the q4 sibling when the only case-field difference is the implementation revision, without treating that as a comparator failure and without hiding any other field.

**Given / When / Then**

- Given the committed q4 bundle (one case) and `Q4_REGENERATION_ALLOWLIST` equal to `("cases[0].provenance.implementation_revision",)`.
- When `_regeneration_result` compares a current case whose only difference from the prior is that path, and both values are distinct full lowercase 40-hex revisions.
- Then `regeneration.pass` is true, `first_mismatch` is null, `status` is pass, and `summary.first_failure` is null. `prior_present` may flip false → true. That flip is derived and is not an allowlist entry.

**Scope**

- In: the constant, `_is_full_git_revision`, `_allowlisted_revision_difference`, and the provenance-loop `continue` in `mf1a_q4_baseline_validation.py`.
- Out: star paths, a second entry, schema changes, tolerance changes, G-07, `validation_pipeline.py`, `SKILL.md`.

**Acceptance signals**

- Length-1 tuple pin; star string absent; three `*_SCHEMA_VERSION` strings stay the v1 literals.
- A non-revision on either side fails at `cases[0].provenance.implementation_revision`.
- Any earlier exact field, extension identity, input identity, dependency, environment, or residual above the comparator is still reported and is not allowlisted.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: CAP-007 / QA-008
- ADR(s): ADR-F1A-009 Amendment 1, ADR-F1A-010 item 2

## DS-B2 — Red-first regeneration tests

**Stakeholder / system value**

- The allowlist is pinned by tests that fail for the right reason before the skip exists, and by guards that already pass.

**Given / When / Then**

- Given unmodified HEAD `99bf9d519f7aac58d8f1e6502c60912decb85995`.
- When the twelve new tests are added after the existing categorical-drift regeneration test, each passing `prior_bundle=` and deep-copying both sides.
- Then the seven red-before tests fail, the five guards pass, and after the comparator patch `--collect-only -k mf1a_q4_baseline_regeneration` collects 14 and all 14 pass. `-k mf1a` is green.

**Scope**

- In: `tests/partitioning/evidence/test_correctness_evidence.py` only. No parametrize unless the Layer 4 task records a new collected count for Reviewer.
- Out: edits to the two existing regeneration assertions, `pytest.ini`, and `validation_pipeline.py` as a fitness gate.

**Acceptance signals**

- Collect-only pin **14**.
- Red-before: length-one, revision-only pass, second-field (sub-case 2), dependency version, environment identity, residual-above, residual-within.
- `rejects_non_revision_value` covers current-side non-SHAs and prior-side `"A" * 40` plus a missing prior key.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-008, ADR-F1A-009 Amendment 1

## DS-B3 — Milestone-agnostic regeneration reference

**Stakeholder / system value**

- The skill reference stops telling later commits to expect a non-zero revision mismatch, while the skill body stays the P0b pointer.

**Given / When / Then**

- Given `references/regeneration-acceptance.md` lines 14–18 at this HEAD (non-zero expected).
- When the Developer replaces that file with the ET-B3 text.
- Then a revision-only allowlisted difference is described as exit 0, `status` pass, `regeneration.pass` true, and `first_mismatch` null, and the file says never widen an allowlist. `SKILL.md` lines 200–204 are byte-identical.

**Scope**

- In: `.cursor/skills/test-density-matrix/references/regeneration-acceptance.md`.
- Out: `SKILL.md`, `references/validation-pipeline-restore.md`, `references/two-commit-close.md`, q4 sha pins (those stay in Layer 2 and Layer 4).

**Acceptance signals**

- Doc review of the replacement against lock §6. Three Developer paths, not four.

**Traceability**

- Initial requirement(s): REQ-004
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-009 Amendment 1

## DS-B4 — Slice A close shape; q4 never committed

**Stakeholder / system value**

- Slice B closes under ADR-F1A-009 (a)–(g) the way Slice A did, and the proof runs accept a revision-only q4 regeneration without committing it.

**Given / When / Then**

- Given a clean C1 that contains the three Developer paths and this Layer 2–4 tree, with stage `step-4b-authorized` and no CLOSEOUT.
- When steps (c) and (g) run `validation_pipeline.py` once each from an empty porcelain, with the extension sha256 pin matching the committed identity.
- Then exit is 0, q4 `status` is pass, the case diff is only `cases[0].provenance.implementation_revision` equal to HEAD, the full-file diff adds only `regeneration.prior_present` false → true, the file is restored to sha256 `483e282d88e3f5e7f1f235abd755aa2bcaf49c470b95cd63c6226617da354a94`, and it is not staged.

**Scope**

- In: proof runs (c) and (g), pre-(d) diff-stat statement, real CLOSEOUT at (d), C2 of CLOSEOUT plus checklist touch-ups only.
- Out: committing the regenerated bundle, a placeholder CLOSEOUT, a waiver, C.0, and any fourth path.

**Acceptance signals**

- Between C1 and CLOSEOUT, `--strict` reports exactly one `SLICE_MISSING_CLOSEOUT` for task-3 (error). Record it. Do not waive it.
- A third diff path fails acceptance. Extension or environment mismatch goes to Tech Lead. Any other mismatch goes to Research Manager. Never widen the allowlist.
- At (g) the new revision equals the C2 SHA.

**Traceability**

- Initial requirement(s): REQ-004, REQ-006
- Capability / quality attribute: QA-008
- ADR(s): ADR-F1A-008, ADR-F1A-009 (+ Amendment 1), ADR-F1A-010
