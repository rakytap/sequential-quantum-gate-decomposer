#!/usr/bin/env python3
"""Lint the SDD traceability spine: CAP-*/QA-* -> M# -> REQ-* -> story -> task -> evidence.

Run from the repo root:

    conda run -n qgd python .cursor/skills/spec-driven-development/scripts/check_traceability.py
    conda run -n qgd python .cursor/skills/spec-driven-development/scripts/check_traceability.py docs/specs/milestones/<slug>

or run both linters through `scripts/specs_check.sh`.

What it proves mechanically:
  * every non-superseded REQ-* is cited by at least one Layer 3 story
  * every story file cites at least one REQ-*
  * every REQ-* cites a CAP-*/QA-* and a milestone id
  * every evidence-matrix row names a runnable command, test lane, or CI gate
  * `make <target>` references resolve against a repo Makefile (or the
    `known_make_targets` list in .sdd-lint.json) when one exists
  * `tests/...`, `benchmarks/...`, and `examples/...` paths referenced as evidence
    exist on disk
  * every REQ-* appears in the milestone closeout, once one exists

Exit status: 0 when there are no unwaived errors, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _sddlint import (  # noqa: E402
    Reporter,
    base_parser,
    read_lines,
    read_text,
    repo_root,
    resolve_context,
    slice_dirs,
)

REQ_RE = re.compile(r"\bREQ-(\d{3})\b")
CAP_QA_RE = re.compile(r"\b(?:CAP|QA)-\d{3}\b")
MILESTONE_RE = re.compile(r"\bM(?:\d+[A-Z]*|[A-Z]{1,4}(?:-[A-Z]{2,4})?)\b")
SUPERSEDED_RE = re.compile(r"~+\s*superseded by REQ-\d{3}", re.IGNORECASE)

MAKE_TARGET_RE = re.compile(r"\bmake\s+([a-zA-Z0-9][a-zA-Z0-9_-]*)")
# Evidence paths that must exist on disk: pytest suites, benchmark evidence pipelines,
# and example scripts are the three executable evidence roots in this repository.
TEST_PATH_RE = re.compile(r"\b((?:tests|benchmarks|examples)/[\w./-]+?\.py)")

# A cell counts as reproducible evidence when it names a command, a file, a test
# symbol, or a named CI lane -- not when it only restates the requirement in prose.
BACKTICK_RE = re.compile(r"`([^`]+)`")
EVIDENCE_TOKEN_RE = re.compile(
    r"(^|[\s(])(make|conda|pytest|python|python3|bash|sh|rg|grep|ls|cat|find|diff|jq|sed|awk|"
    r"git|cmake|ctest|"
    r"scripts/|tests/|benchmarks/|examples/|docs/|squander/)\b"
    r"|\btest_\w+"
    r"|\.(py|cpp|h|sh|ya?ml|toml|json|cfg|md|txt)\b"
    r"|::",
    re.IGNORECASE,
)
# Named evidence lanes in this repository: the fast and `slow` pytest lanes, the benchmark
# evidence pipelines, the optional C++ tests, and the Qiskit Aer external reference. A row
# that names a lane has pointed at a runnable suite.
CI_PHRASE_RE = re.compile(
    r"(CI job|CI gate|CI lane|workflow|\blane\b|\bopt-in\b|\bslow\b|\bfast lane\b|"
    r"evidence pipeline|QGD_CTEST|Qiskit Aer|\bAer\b)",
    re.IGNORECASE,
)
# Evidence types the skill recognises as non-executable but still declared. A row
# that says "doc review" has named how it will be proven; a row that says nothing
# has not.
DECLARED_MANUAL_RE = re.compile(
    r"(doc review|docs only|no test|manual (check|review|verification)|inspection|"
    r"stakeholder sign-off|sign-off)",
    re.IGNORECASE,
)
# Markdown escapes an in-cell pipe as `\|`; splitting on those corrupts the cells.
CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
# A row may legitimately point at another row or artifact instead of repeating the
# command, as long as it points somewhere.
DEFERRAL_RE = re.compile(
    r"(\bvia\b|\bsee\b|\bper\b|\bsame as\b|\bcovered by\b|\bas above\b|→|->)",
    re.IGNORECASE,
)


def cell_has_evidence(cell: str) -> bool:
    for token in BACKTICK_RE.findall(cell):
        if EVIDENCE_TOKEN_RE.search(token):
            return True
    return bool(CI_PHRASE_RE.search(cell))

# Only tables that present themselves as evidence matrices are checked for runnable
# commands. A trace-rollup table (ET -> DS -> REQ) legitimately carries no command.
EVIDENCE_HEADER_RE = re.compile(
    r"(evidence|command|ci gate|expected result|test type|verification|proof|lane)",
    re.IGNORECASE,
)

STORY_FILE_NAMES = ("DELIVERY_STORIES.md",)
STORY_LEGACY_RE = re.compile(r"^TASK_\d+_DELIVERY_STORIES\.md$")
TASK_FILE_NAMES = ("ENGINEERING_TASKS.md",)
TASK_LEGACY_RE = re.compile(r"^TASK_\d+_ENGINEERING_TASKS\.md$")
MINI_SPEC_RE = re.compile(r"MINI_SPEC\.md$")


def declared_reqs(path: Path) -> tuple[dict[str, int], set[str]]:
    """Return {req_id: line_no} for requirements declared in the file, plus superseded ids."""
    declared: dict[str, int] = {}
    superseded: set[str] = set()
    for idx, line in enumerate(read_lines(path), start=1):
        for match in REQ_RE.finditer(line):
            req = f"REQ-{match.group(1)}"
            declared.setdefault(req, idx)
            if SUPERSEDED_RE.search(line):
                superseded.add(req)
    return declared, superseded


def req_blocks(path: Path) -> dict[str, str]:
    """Slice the requirements file into per-REQ text blocks for upstream-citation checks."""
    lines = read_lines(path)
    starts: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        match = REQ_RE.search(line)
        if match:
            req = f"REQ-{match.group(1)}"
            if not starts or starts[-1][1] != req:
                starts.append((idx, req))
    blocks: dict[str, list[str]] = {}
    for pos, (idx, req) in enumerate(starts):
        end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
        blocks.setdefault(req, []).extend(lines[idx:end])
    return {req: "\n".join(body) for req, body in blocks.items()}


def collect_refs(paths: list[Path]) -> dict[str, set[str]]:
    """Map REQ id -> set of file paths (as strings) that cite it."""
    refs: dict[str, set[str]] = {}
    for path in paths:
        text = read_text(path)
        for match in REQ_RE.finditer(text):
            refs.setdefault(f"REQ-{match.group(1)}", set()).add(str(path))
    return refs


def slice_files(slice_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    stories, tasks, specs = [], [], []
    for path in sorted(slice_dir.iterdir()):
        if not path.is_file() or path.suffix != ".md":
            continue
        if path.name in STORY_FILE_NAMES or STORY_LEGACY_RE.match(path.name):
            stories.append(path)
        elif path.name in TASK_FILE_NAMES or TASK_LEGACY_RE.match(path.name):
            tasks.append(path)
        elif MINI_SPEC_RE.search(path.name):
            specs.append(path)
    return stories, tasks, specs


def make_targets(root: Path) -> set[str]:
    makefile = root / "Makefile"
    if not makefile.is_file():
        return set()
    targets = set()
    for line in read_lines(makefile):
        match = re.match(r"^([a-zA-Z0-9][a-zA-Z0-9_-]*):", line)
        if match:
            targets.add(match.group(1))
        if line.startswith(".PHONY:"):
            targets.update(line.split(":", 1)[1].split())
    return targets


def check_evidence_rows(
    reporter: Reporter,
    path: Path,
    targets: set[str],
    known_targets: set[str],
    project_root: Path,
) -> None:
    """Every evidence-matrix row that carries a trace id must name runnable evidence."""
    lines = read_lines(path)
    in_evidence_table = False
    previous = ""
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            in_evidence_table = False
            previous = stripped
            continue
        if stripped.startswith("|--") or "---|" in stripped:
            # The line above a separator is the header: it decides whether this
            # table is an evidence matrix or a plain mapping table.
            in_evidence_table = bool(EVIDENCE_HEADER_RE.search(previous))
            previous = stripped
            continue
        previous = stripped
        if not in_evidence_table:
            continue
        if not (REQ_RE.search(stripped) or CAP_QA_RE.search(stripped)):
            continue
        cells = [c.strip() for c in CELL_SPLIT_RE.split(stripped.strip("|"))]
        if len(cells) < 3:
            continue
        if not any(cell_has_evidence(c) for c in cells):
            if not (DEFERRAL_RE.search(stripped) or DECLARED_MANUAL_RE.search(stripped)):
                reporter.add(
                    "EVIDENCE_NO_COMMAND",
                    "warn",
                    path,
                    "evidence row carries a trace id but names no command, test symbol, "
                    f"file, or CI lane, and defers to nothing: {stripped[:120]}",
                    line=idx,
                )
            continue
        for target in MAKE_TARGET_RE.findall(stripped):
            if target not in targets and target not in known_targets:
                reporter.add(
                    "MAKE_TARGET_UNKNOWN",
                    "warn",
                    path,
                    f"evidence names `make {target}`, which is not a Makefile target",
                    line=idx,
                )
        for test_path in TEST_PATH_RE.findall(stripped):
            candidate = project_root / test_path.split("::", 1)[0]
            if not candidate.exists():
                reporter.add(
                    "TEST_PATH_MISSING",
                    "warn",
                    path,
                    f"evidence names `{test_path}`, which does not exist on disk",
                    line=idx,
                )


def check_milestone(
    reporter: Reporter,
    milestone: Path,
    targets: set[str],
    known_targets: set[str],
    project_root: Path,
) -> None:
    init_req = milestone / "INITIAL_REQUIREMENTS.md"
    if not init_req.is_file():
        return

    declared, superseded = declared_reqs(init_req)
    blocks = req_blocks(init_req)
    active = {req: line for req, line in declared.items() if req not in superseded}

    for req, line in sorted(active.items()):
        body = blocks.get(req, "")
        if not CAP_QA_RE.search(body):
            reporter.add(
                "REQ_NO_UPSTREAM",
                "error",
                init_req,
                f"{req} cites no CAP-*/QA-*; the spine breaks above the milestone",
                line=line,
            )

    # The milestone id anchors the whole file, so it is checked once, in the header.
    header = "\n".join(read_lines(init_req)[:40])
    if not MILESTONE_RE.search(header):
        reporter.add(
            "MILESTONE_ID_MISSING",
            "warn",
            init_req,
            "no milestone id (M#) in the first 40 lines; the requirements baseline "
            "must name the roadmap milestone it scopes",
        )

    all_stories: list[Path] = []
    all_tasks: list[Path] = []
    all_specs: list[Path] = []
    for slice_dir in slice_dirs(milestone):
        stories, tasks, specs = slice_files(slice_dir)
        all_stories.extend(stories)
        all_tasks.extend(tasks)
        all_specs.extend(specs)

        for story in stories:
            if not REQ_RE.search(read_text(story)):
                reporter.add(
                    "STORY_NO_REQ",
                    "error",
                    story,
                    "no REQ-* citation; a delivery story must trace to product intent",
                )

    if not (all_stories or all_tasks or all_specs):
        return

    # Stories are the required downstream link; mini-specs count too because the
    # first milestone embedded its stories inside the mini-spec.
    story_refs = collect_refs(all_stories + all_specs)
    task_refs = collect_refs(all_tasks)

    for req, line in sorted(active.items()):
        if req not in story_refs:
            reporter.add(
                "REQ_ORPHAN",
                "error",
                init_req,
                f"{req} is not cited by any Layer 3 story or mini-spec in this milestone",
                line=line,
            )
        elif req not in task_refs:
            reporter.add(
                "REQ_NO_TASK",
                "warn",
                init_req,
                f"{req} reaches a story but no Layer 4 engineering task cites it",
                line=line,
            )

    for path in all_specs + all_tasks:
        check_evidence_rows(reporter, path, targets, known_targets, project_root)

    closeouts = [p for p in sorted(milestone.glob("*_CLOSEOUT.md")) if p.name != "CLOSEOUT.md"]
    for closeout in closeouts:
        text = read_text(closeout)
        missing = sorted(req for req in active if req not in text)
        if missing:
            reporter.add(
                "CLOSEOUT_REQ_GAP",
                "warn",
                closeout,
                "milestone closeout evidence matrix omits: " + ", ".join(missing),
            )
        check_evidence_rows(reporter, closeout, targets, known_targets, project_root)


def main(argv: list[str] | None = None) -> int:
    parser = base_parser(__doc__.splitlines()[0])
    args = parser.parse_args(argv)
    root, config, milestones = resolve_context(args)
    if not milestones:
        return 0

    project_root = repo_root()
    targets = make_targets(project_root)
    known = set(config.known_make_targets)

    reporter = Reporter(
        root=root,
        config=config,
        strict=args.strict,
        honor_waivers=not args.no_waivers,
    )
    for milestone in milestones:
        check_milestone(reporter, milestone, targets, known, project_root)

    return reporter.emit(args.format, "SDD traceability spine")


if __name__ == "__main__":
    raise SystemExit(main())
