#!/usr/bin/env bash
# Run both SDD spec linters (artifact structure + traceability spine) over docs/specs.
#
#   bash .cursor/skills/spec-driven-development/scripts/specs_check.sh                 # errors only
#   bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --strict        # warnings become errors
#   bash .cursor/skills/spec-driven-development/scripts/specs_check.sh docs/specs/milestones/<slug>
#   bash .cursor/skills/spec-driven-development/scripts/specs_check.sh --no-waivers    # show suppressed debt
#
# The linters are standard-library only, so any Python >= 3.10 works. The repository
# convention is the `qgd` conda environment; set SDD_PYTHON to override the interpreter
# (e.g. SDD_PYTHON=python3 for a hook or CI runner without conda).
#
# Exit status: non-zero when either linter reports an unwaived error.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$ROOT"

if [ -n "${SDD_PYTHON:-}" ]; then
  PY="$SDD_PYTHON"
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -q '^qgd[[:space:]]'; then
  PY="conda run -n qgd --no-capture-output python"
else
  PY="python3"
fi

status=0
# shellcheck disable=SC2086
$PY "$HERE/check_artifacts.py" "$@" || status=1
# shellcheck disable=SC2086
$PY "$HERE/check_traceability.py" "$@" || status=1
exit $status
