#!/usr/bin/env python3
"""Exercise SLICE_MISSING_CLOSEOUT stage signal and substantive-closeout (P1 / rev F)."""

from __future__ import annotations

import io
import json
import contextlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_artifacts  # noqa: E402

SLUG = "synthetic-stage"
SLUG_UPPER = "SYNTHETIC_STAGE"
ABSENT_MSG = check_artifacts.SLICE_MISSING_CLOSEOUT_ABSENT_MSG
PLACEHOLDER_MSG = check_artifacts.SLICE_MISSING_CLOSEOUT_PLACEHOLDER_MSG

CTX = "> **Status:** planning · **Milestone:** M1 synthetic-stage"


def write_milestone(
    root: Path,
    *,
    tasks_lines: list[str],
    closeout: str | None = None,
    closeout_mode: str = "absent",
    pad_adrs: bool = False,
    drop_stories: bool = False,
    extra_slices: dict[str, list[str]] | None = None,
) -> Path:
    ms = root / "milestones" / SLUG
    task1 = ms / "task-1"
    task1.mkdir(parents=True, exist_ok=True)

    init = f"""# Initial requirements — M1
{CTX}

## REQ-001 — baseline
Upstream: CAP-001.
"""
    (ms / "INITIAL_REQUIREMENTS.md").write_text(init, encoding="utf-8")
    for name in (
        f"DETAILED_PLANNING_{SLUG_UPPER}.md",
        f"ADRS_{SLUG_UPPER}.md",
        "PRE_IMPLEMENTATION_COMPLETION_CHECKLIST.md",
    ):
        (ms / name).write_text(f"# Layer 1\n{CTX}\n", encoding="utf-8")

    if pad_adrs:
        adrs = ms / f"ADRS_{SLUG_UPPER}.md"
        lines = adrs.read_text(encoding="utf-8").splitlines()
        while len(lines) < 401:
            lines.append("padding")
        adrs.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stories = f"""# Delivery stories
{CTX}

### Story
**Traceability**
- Initial requirement(s): REQ-001
"""
    if not drop_stories:
        (task1 / "DELIVERY_STORIES.md").write_text(stories, encoding="utf-8")

    et = "\n".join(tasks_lines) + "\n"
    (task1 / "ENGINEERING_TASKS.md").write_text(et, encoding="utf-8")
    (task1 / "TASK_1_MINI_SPEC.md").write_text(
        f"# Mini spec\n{CTX}\n\nTraces REQ-001.\n", encoding="utf-8"
    )

    co = task1 / "CLOSEOUT.md"
    if closeout_mode == "absent":
        if co.exists():
            co.unlink()
    elif closeout_mode == "empty":
        co.write_text("", encoding="utf-8")
    elif closeout_mode == "ws":
        co.write_text("  \n\t\n", encoding="utf-8")
    elif closeout_mode == "write" and closeout is not None:
        co.write_text(closeout, encoding="utf-8")

    if extra_slices:
        for dirname, et_lines in extra_slices.items():
            tdir = ms / dirname
            tdir.mkdir(parents=True, exist_ok=True)
            num = dirname.split("-", 1)[1]
            (tdir / "DELIVERY_STORIES.md").write_text(stories, encoding="utf-8")
            (tdir / f"TASK_{num}_MINI_SPEC.md").write_text(
                f"# Mini spec\n{CTX}\n\nTraces REQ-001.\n", encoding="utf-8"
            )
            (tdir / "ENGINEERING_TASKS.md").write_text(
                "\n".join(et_lines) + "\n", encoding="utf-8"
            )
            cp = tdir / "CLOSEOUT.md"
            if cp.exists():
                cp.unlink()

    return root


def et_base(*extra: str) -> list[str]:
    lines = ["# Engineering tasks", CTX]
    lines.extend(extra)
    lines.append("")
    lines.append("Implements REQ-001.")
    return lines


def run_artifacts(root: Path, strict: bool = False, fmt: str = "json") -> tuple[int, dict[str, Any]]:
    argv = ["--root", str(root), "--format", fmt]
    if strict:
        argv.append("--strict")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = check_artifacts.main(argv)
    if fmt == "json":
        payload = json.loads(buf.getvalue())
        return code, payload
    return code, {"text": buf.getvalue()}


def absent_stdout_key(payload: dict[str, Any]) -> tuple[tuple[str, str, str, str], ...]:
    """Normal-mode absent closeout lines (path, code, severity, message) for equality check."""
    rows = []
    for f in payload.get("findings", []):
        if f.get("code") != "SLICE_MISSING_CLOSEOUT" or f.get("severity") == "waived":
            continue
        rows.append((f["path"], f["code"], f["severity"], f["message"]))
    return tuple(sorted(rows))


def active_findings(payload: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for f in payload.get("findings", []):
        if f.get("severity") == "waived":
            continue
        out.append(
            {
                "code": f["code"],
                "severity": f["severity"],
                "message": f.get("message", ""),
            }
        )
    return out


def assert_case(
    name: str,
    root: Path,
    *,
    exit_n: int,
    exit_s: int,
    findings_n: list[tuple[str, str]],
    findings_s: list[tuple[str, str]] | None = None,
) -> None:
    code_n, payload_n = run_artifacts(root, strict=False)
    code_s, payload_s = run_artifacts(root, strict=True)
    fn = sorted((f["code"], f["severity"]) for f in active_findings(payload_n))
    fs = sorted((f["code"], f["severity"]) for f in active_findings(payload_s))
    findings_n = sorted(findings_n)
    if findings_s is not None:
        findings_s = sorted(findings_s)
    if code_n != exit_n or fn != findings_n:
        raise AssertionError(f"{name} normal: exit={code_n} findings={fn} expected exit={exit_n} {findings_n}")
    if findings_s is None:
        findings_s = [(c, "error" if exit_s else sev) for c, sev in findings_n]
        if exit_s == 0:
            findings_s = findings_n
    if code_s != exit_s or fs != findings_s:
        raise AssertionError(f"{name} strict: exit={code_s} findings={fs} expected exit={exit_s} {findings_s}")
    print(f"PASS {name}")


def main() -> int:
    absent_finding_keys: list[tuple[tuple[str, str, str, str], ...]] = []
    tmp = Path(tempfile.mkdtemp(prefix="nc-p1-"))
    try:
        # NC-P1-4a-absent
        r = write_milestone(
            tmp / "4a",
            tasks_lines=et_base("> **SDD stage:** step-4a"),
        )
        _, p4a = run_artifacts(r)
        absent_finding_keys.append(absent_stdout_key(p4a))
        assert_case(
            "NC-P1-4a-absent",
            r,
            exit_n=0,
            exit_s=0,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "warn")],
        )

        # NC-P1-4b-absent
        r = write_milestone(
            tmp / "4b",
            tasks_lines=et_base("> **SDD stage:** step-4b-authorized"),
        )
        _, p4b = run_artifacts(r)
        absent_finding_keys.append(absent_stdout_key(p4b))
        assert_case(
            "NC-P1-4b-absent",
            r,
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        # NC-P1-missing-absent
        r = write_milestone(tmp / "missing", tasks_lines=et_base())
        _, pm = run_artifacts(r)
        absent_finding_keys.append(absent_stdout_key(pm))
        assert_case(
            "NC-P1-missing-absent",
            r,
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        # NC-P1-dup-absent
        r = write_milestone(
            tmp / "dup",
            tasks_lines=et_base(
                "> **SDD stage:** step-4a",
                "> **SDD stage:** step-4a",
            ),
        )
        _, pd = run_artifacts(r)
        absent_finding_keys.append(absent_stdout_key(pd))
        if len({tuple(k) for k in absent_finding_keys}) != 1:
            raise AssertionError(
                "absent-stage normal finding output must match across four cases"
            )
        assert_case(
            "NC-P1-dup-absent",
            r,
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        assert_case(
            "NC-P1-other-stage",
            write_milestone(
                tmp / "other",
                tasks_lines=et_base("> **SDD stage:** step-4A"),
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )
        assert_case(
            "NC-P1-empty-value",
            write_milestone(tmp / "emptyval", tasks_lines=et_base("> **SDD stage:**")),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )
        assert_case(
            "NC-P1-shared-middot",
            write_milestone(
                tmp / "middot",
                tasks_lines=et_base("> **SDD stage:** step-4a · **Parent:** x"),
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )
        assert_case(
            "NC-P1-shared-prefix",
            write_milestone(
                tmp / "prefix",
                tasks_lines=et_base("> **Status:** code-ready · **SDD stage:** step-4a"),
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        line13 = ["# Engineering tasks", CTX]
        while len(line13) < 12:
            line13.append("")
        line13.append("> **SDD stage:** step-4a")
        line13.extend(["", "Implements REQ-001."])
        assert_case(
            "NC-P1-line13",
            write_milestone(tmp / "line13", tasks_lines=line13),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        line13x = [
            "# Engineering tasks",
            CTX,
            "",
            "",
            "",
            "> **SDD stage:** step-4a",
            "",
            "",
            "",
            "",
            "",
            "",
            "> **SDD stage:** step-4b-authorized",
            "",
            "Implements REQ-001.",
        ]
        assert_case(
            "NC-P1-line13-extra",
            write_milestone(tmp / "line13x", tasks_lines=line13x),
            exit_n=0,
            exit_s=0,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "warn")],
        )

        assert_case(
            "NC-P1-html",
            write_milestone(
                tmp / "html",
                tasks_lines=et_base("<!-- **SDD stage:** step-4a -->"),
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )
        assert_case(
            "NC-P1-unicode",
            write_milestone(
                tmp / "unicode",
                tasks_lines=et_base("> **SDD stage:** step-4\u0430"),
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )
        assert_case(
            "NC-P1-nbsp",
            write_milestone(
                tmp / "nbsp",
                tasks_lines=et_base("> **SDD stage:** step-4a\u00a0"),
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        assert_case(
            "NC-P1-injected-warn",
            write_milestone(
                tmp / "inj-warn",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                pad_adrs=True,
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SIZE_BUDGET", "warn"),
            ],
            findings_s=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SIZE_BUDGET", "error"),
            ],
        )
        assert_case(
            "NC-P1-injected-warn-4b",
            write_milestone(
                tmp / "inj-4b",
                tasks_lines=et_base("> **SDD stage:** step-4b-authorized"),
                pad_adrs=True,
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SIZE_BUDGET", "warn"),
            ],
            findings_s=[
                ("SLICE_MISSING_CLOSEOUT", "error"),
                ("SIZE_BUDGET", "error"),
            ],
        )
        assert_case(
            "NC-P1-injected-warn-missing",
            write_milestone(tmp / "inj-miss", tasks_lines=et_base(), pad_adrs=True),
            exit_n=0,
            exit_s=1,
            findings_n=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SIZE_BUDGET", "warn"),
            ],
            findings_s=[
                ("SLICE_MISSING_CLOSEOUT", "error"),
                ("SIZE_BUDGET", "error"),
            ],
        )
        assert_case(
            "NC-P1-injected-warn-dup",
            write_milestone(
                tmp / "inj-dup",
                tasks_lines=et_base(
                    "> **SDD stage:** step-4a",
                    "> **SDD stage:** step-4a",
                ),
                pad_adrs=True,
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SIZE_BUDGET", "warn"),
            ],
            findings_s=[
                ("SLICE_MISSING_CLOSEOUT", "error"),
                ("SIZE_BUDGET", "error"),
            ],
        )
        assert_case(
            "NC-P1-injected-error",
            write_milestone(
                tmp / "inj-err",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                drop_stories=True,
            ),
            exit_n=1,
            exit_s=1,
            findings_n=[
                ("SLICE_MISSING_STORIES", "error"),
                ("SLICE_MISSING_CLOSEOUT", "warn"),
            ],
            findings_s=[
                ("SLICE_MISSING_STORIES", "error"),
                ("SLICE_MISSING_CLOSEOUT", "warn"),
            ],
        )

        for label, mode in (("NC-P1-empty", "empty"), ("NC-P1-ws", "ws")):
            assert_case(
                label,
                write_milestone(
                    tmp / label,
                    tasks_lines=et_base("> **SDD stage:** step-4a"),
                    closeout_mode=mode,
                ),
                exit_n=0,
                exit_s=1,
                findings_n=[
                    ("SLICE_MISSING_CLOSEOUT", "warn"),
                    ("CLOSEOUT_NO_STATUS", "warn"),
                ],
                findings_s=[
                    ("SLICE_MISSING_CLOSEOUT", "error"),
                    ("CLOSEOUT_NO_STATUS", "error"),
                ],
            )

        stub = "**Status:** shipped\n"
        for label, stage in (
            ("NC-P1-status-stub", "step-4a"),
            ("NC-P1-status-stub-4b", "step-4b-authorized"),
        ):
            root_case = write_milestone(
                tmp / label,
                tasks_lines=et_base(f"> **SDD stage:** {stage}"),
                closeout_mode="write",
                closeout=stub,
            )
            assert_case(
                label,
                root_case,
                exit_n=0,
                exit_s=1,
                findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
                findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
            )
            _, p = run_artifacts(root_case)
            msgs = [
                f["message"]
                for f in active_findings(p)
                if f["code"] == "SLICE_MISSING_CLOSEOUT"
            ]
            if msgs != [PLACEHOLDER_MSG]:
                raise AssertionError(f"{label} message {msgs}")

        placeholder_co = """# Closeout
> **Slice:** M1 synthetic-stage task-1
**Status:** placeholder

## Summary

```bash
echo ok
```
"""
        assert_case(
            "NC-P1-status-placeholder",
            write_milestone(
                tmp / "ph",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                closeout_mode="write",
                closeout=placeholder_co,
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[("SLICE_MISSING_CLOSEOUT", "warn")],
            findings_s=[("SLICE_MISSING_CLOSEOUT", "error")],
        )

        assert_case(
            "NC-P1-comment",
            write_milestone(
                tmp / "comment",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                closeout_mode="write",
                closeout="<!-- only -->\n",
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[
                ("CLOSEOUT_NO_STATUS", "warn"),
                ("MISSING_CONTEXT_HEADER", "warn"),
                ("SLICE_MISSING_CLOSEOUT", "warn"),
            ],
            findings_s=[
                ("CLOSEOUT_NO_STATUS", "error"),
                ("MISSING_CONTEXT_HEADER", "error"),
                ("SLICE_MISSING_CLOSEOUT", "error"),
            ],
        )

        assert_case(
            "NC-P1-placeholder-handback",
            write_milestone(
                tmp / "phb",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                closeout_mode="write",
                closeout="**Status:** implementation handback\n",
            ),
            exit_n=1,
            exit_s=1,
            findings_n=[
                ("CLOSEOUT_HANDBACK_MISSING", "error"),
                ("SLICE_MISSING_CLOSEOUT", "warn"),
            ],
            findings_s=[
                ("CLOSEOUT_HANDBACK_MISSING", "error"),
                ("SLICE_MISSING_CLOSEOUT", "error"),
            ],
        )

        substantive = """# Closeout
> **Slice:** M1 synthetic-stage task-1

**Status:** shipped

## Summary

```bash
true
```
"""
        assert_case(
            "NC-P1-substantive-4a",
            write_milestone(
                tmp / "sub4a",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                closeout_mode="write",
                closeout=substantive,
            ),
            exit_n=0,
            exit_s=0,
            findings_n=[],
            findings_s=[],
        )
        assert_case(
            "NC-P1-substantive-nostage",
            write_milestone(
                tmp / "sub-ns",
                tasks_lines=et_base(),
                closeout_mode="write",
                closeout=substantive,
            ),
            exit_n=0,
            exit_s=0,
            findings_n=[],
            findings_s=[],
        )
        sub4b = substantive.replace("shipped", "ready for Reviewer C2 gate")
        assert_case(
            "NC-P1-substantive-4b",
            write_milestone(
                tmp / "sub4b",
                tasks_lines=et_base("> **SDD stage:** step-4b-authorized"),
                closeout_mode="write",
                closeout=sub4b,
            ),
            exit_n=0,
            exit_s=0,
            findings_n=[],
            findings_s=[],
        )
        handback_sub = substantive.replace("shipped", "implementation handback")
        assert_case(
            "NC-P1-handback",
            write_milestone(
                tmp / "handback",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                closeout_mode="write",
                closeout=handback_sub,
            ),
            exit_n=1,
            exit_s=1,
            findings_n=[("CLOSEOUT_HANDBACK_MISSING", "error")],
            findings_s=[("CLOSEOUT_HANDBACK_MISSING", "error")],
        )

        assert_case(
            "NC-P1-multi",
            write_milestone(
                tmp / "multi",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                extra_slices={
                    "task-2": et_base("> **SDD stage:** step-4b-authorized"),
                },
            ),
            exit_n=0,
            exit_s=1,
            findings_n=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SLICE_MISSING_CLOSEOUT", "warn"),
            ],
            findings_s=[
                ("SLICE_MISSING_CLOSEOUT", "warn"),
                ("SLICE_MISSING_CLOSEOUT", "error"),
            ],
        )

        assert_case(
            "NC-P1-residual-shaped-stub",
            write_milestone(
                tmp / "residual",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
                closeout_mode="write",
                closeout=substantive.replace("true", "true"),
            ),
            exit_n=0,
            exit_s=0,
            findings_n=[],
            findings_s=[],
        )

        four_a = write_milestone(
            tmp / "specs-wrap",
            tasks_lines=et_base("> **SDD stage:** step-4a"),
        )
        wrap = HERE / "specs_check.sh"
        proc = subprocess.run(
            ["bash", str(wrap), "--root", str(four_a), "--strict"],
            cwd=HERE.parents[3],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"specs_check.sh NC-P1-4a-absent exit {proc.returncode}: {proc.stderr}"
            )
        print("PASS specs_check.sh --root NC-P1-4a-absent")

        # Message checks on absent path
        _, p = run_artifacts(
            write_milestone(
                tmp / "absent-msg",
                tasks_lines=et_base("> **SDD stage:** step-4a"),
            )
        )
        msgs = [f["message"] for f in active_findings(p) if f["code"] == "SLICE_MISSING_CLOSEOUT"]
        if msgs != [ABSENT_MSG]:
            raise AssertionError(f"absent message mismatch: {msgs}")

        print("ALL NC-P1 cases PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
