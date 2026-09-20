# Project Requirements and Setup

## Testing and Installation

**Important:** For testing, running and installing anything related to the current feature branch, always use the instructions written in the current `SETUP.md` file.

The `SETUP.md` file (located at `docs/density_matrix_project/SETUP.md`) contains the most up-to-date and accurate instructions for:
- Setting up the development environment
- Installing dependencies
- Building the project
- Running tests
- Troubleshooting common issues

Always refer to `SETUP.md` rather than relying on potentially outdated instructions elsewhere.

## Technical Requirements

**C++ Version:** The project uses C++11 standard.

**Python Version:** The project uses Python 3.13.

USE qgd CONDA ENVIRONMENT FOR ALL BUILD AND TESTING.

## Spec-Driven Status Surfaces

Spec work lives under `docs/specs/` and follows `.cursor/skills/spec-driven-development/SKILL.md`
(entry point: `docs/sdd-skills-guide.md`). When updating implementation status there:

- `task-<n>/CLOSEOUT.md` is the single owner of a slice's implementation progress; `<MILESTONE_ID>_CLOSEOUT.md` owns milestone status.
- Keep `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` as the Layer 1 contract closure surface only.
- Only mark engineering-task checklist rows complete when the current slice explicitly closes them; do not retroactively certify upstream rows because a later slice finished.
- Preserve deferred IDs/work items so bounded slice completion does not read as full milestone closure.
- Run `bash .cursor/skills/spec-driven-development/scripts/specs_check.sh` before claiming spec work complete.

`docs/density_matrix_project/archive/` holds the delivered Phase 1–3.1 record and is read-only.