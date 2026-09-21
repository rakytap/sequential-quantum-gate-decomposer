# AGENTS.md — how to work in this repository (density-matrix track)

The portable constitution for any agent or engineer changing the density-matrix track of
SQUANDER. Cursor rules under `.cursor/rules/` carry the same content for Cursor sessions;
this file is the copy that other tools read.

## Non-negotiables

1. **Tests, benchmarks, and examples run in the `qgd` conda environment.** Interactively
   `conda activate qgd`; non-interactively `conda run -n qgd --no-capture-output <cmd>`.
   Never assume the system interpreter matches the project. Build and setup procedure:
   `docs/density_matrix_project/SETUP.md`; commands and lanes: `docs/specs/TECH_STACK.md`.
2. **Rebuild after C++ or CMake changes** before testing (`.cursor/skills/clean-rebuild/`).
   A stale extension silently tests old code. C++11 and Python 3.13 are the targets.
3. **Exactness is the contract.** The sequential `NoisyCircuit` executor is the semantic
   baseline for every partitioned, fused, or new backend path; Qiskit Aer is the external
   reference. Never loosen a frozen tolerance or weaken, skip, or delete a test to make a
   lane pass. The density path is strict: unsupported inputs raise explicit errors, they do
   not fall back.
4. **Specs are the source of truth, and the spine is checked.** Work under `docs/specs/`
   follows the spec-driven development stack below and keeps
   `CAP-*/QA-* → M# → REQ-* → delivery story → engineering task → evidence` unbroken. Run
   `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` before claiming spec
   work complete.
5. **The archive is frozen.** `docs/density_matrix_project/archive/` holds the delivered
   Phase 1–3.1 record; cite it, never extend it.
6. **Prefer the existing convention.** Match surrounding C++ and Python style, reuse the
   ubiquitous language from `docs/specs/PRODUCT_STATEMENT.md` once it exists, and reuse
   before abstracting.

## Spec-driven development

Four project skills own the planning stack, top-down. Read
[`docs/sdd-skills-guide.md`](docs/sdd-skills-guide.md) for the full flow, the canonical
stack diagram, the shared glossary, and where this repository currently stands.

| Skill | Produces |
|-------|----------|
| `create-product-statement` | `docs/specs/PRODUCT_STATEMENT.md` — `CAP-*`, `QA-*` (stable) |
| `create-product-roadmap` | `docs/specs/ROADMAP.md` — milestones `M#` (living) |
| `create-initreq-for-sdd` | `milestones/<slug>/INITIAL_REQUIREMENTS.md` — `REQ-*` |
| `spec-driven-development` | `milestones/<slug>/…` — Layer 1–4 artifacts, closeouts, and the current-state docs |

Milestones are built **slice by slice**, every slice shippable (extension builds, named
lanes green, evidence bundles regenerate). Planning and code generation are separate roles:
`.cursor/agents/sdd-planner.md` (read-only) owns all planning,
`.cursor/agents/sdd-implementer.md` writes code and tests only, and
`.cursor/agents/sdd-critic.md` (read-only) attacks a spec before a verdict. Papers and
abstracts are not spec artifacts and never gate a slice.

## Verification commands

```bash
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh             # SDD structure + spine (errors only)
bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict    # + size budgets, stale evidence, legacy naming
conda run -n qgd --no-capture-output pytest -m "density_matrix and not slow"
conda run -n qgd --no-capture-output python benchmarks/density_matrix/correctness_evidence/validation_pipeline.py
```

The `test-density-matrix` skill runs the full workflow (smoke import, pytest lanes, example,
optional Aer comparison, optional C++ tests) and defines the report format.

## Skills and hooks

- `.cursor/skills/` holds the project skills: the four SDD skills plus `clean-rebuild` and
  `test-density-matrix`. One skill name has exactly one copy; project context for a skill
  goes in a rule under `.cursor/rules/`, not a fork.
- `.cursor/hooks.json` runs two fail-open hooks: `afterFileEdit` appends an audit line when a
  `docs/specs/` artifact changes, and `stop` re-runs the spec linters when the session
  touched `docs/specs/` and returns a follow-up if the spine is broken.

## Repository layout (density-matrix track)

| Path | Contents |
|------|----------|
| `squander/src-cpp/density_matrix/` | C++ mixed-state core: `DensityMatrix`, `NoisyCircuit`, operations, noise channels, C++ tests |
| `squander/density_matrix/` | pybind11 bindings (`_density_matrix_cpp`) |
| `squander/VQA/`, `squander/src-cpp/variational_quantum_eigensolver/` | VQE base with the optional density backend and bridge metadata |
| `squander/partitioning/noisy_*.py` | Noisy planner, descriptors, partitioned runtime, fusion, channel-native modes |
| `tests/density_matrix/`, `tests/partitioning/`, `tests/VQE/` | pytest suites (`slow` marker) |
| `benchmarks/density_matrix/` | Evidence pipelines and validators; artifacts under `artifacts/` |
| `examples/density_matrix/` | Runnable examples |
| `docs/specs/` | The single spec root: product statement, roadmap, current-state docs, milestones |
| `docs/density_matrix_project/` | Project docs (`README`, `SETUP`, `CHANGELOG`, `RESEARCH_ALIGNMENT`) and the read-only `archive/` |
