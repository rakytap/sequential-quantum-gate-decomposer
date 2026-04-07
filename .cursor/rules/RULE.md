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

When updating implementation-status or slice-checklist sections under
`docs/density_matrix_project/phases/`:

- Treat the current story/slice status file as the single owner of implementation progress.
- Keep `PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md` as the Layer 1 contract closure surface only.
- Only mark engineering-task checklist rows complete when the current plan explicitly closes them; do not retroactively certify upstream rows just because a later slice/task finished.
- Preserve deferred IDs/work items so bounded slice completion does not read as full task or phase closure.