#!/usr/bin/env python3
"""Fetch three- and four-qubit BQSKit results and emit five LaTeX tables."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Optional, TextIO


ROOT = Path(__file__).resolve().parent
DEFAULT_REMOTE = (
    "157.181.172.111:sequential-quantum-gate-decomposer/results_3qbit.json"
)
DEFAULT_REMOTE_4Q = (
    "157.181.172.111:"
    "sequential-quantum-gate-decomposer/results_4qbit.json"
)
STAGES = ("all_to_all", "routed", "final")
TIME_FIELDS = ("a2a", "routing", "optimization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the in-progress BQSKit results and compare every circuit "
            "completed by both BQSKit and OSR."
        )
    )
    parser.add_argument(
        "--osr",
        type=Path,
        default=ROOT / "results_3qbit.json",
        help="OSR results JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--bqskit",
        type=Path,
        default=ROOT / "results-bqskit_3qbit.json",
        help="local cache for three-qubit BQSKit results (default: %(default)s)",
    )
    parser.add_argument(
        "--osr-4q",
        type=Path,
        default=ROOT / "results_4qbit.json",
        help="four-qubit OSR results JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--bqskit-4q",
        type=Path,
        default=ROOT / "results-bqskit_4qbit.json",
        help="local cache for four-qubit BQSKit results (default: %(default)s)",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help="scp source for three-qubit BQSKit results (default: %(default)s)",
    )
    parser.add_argument(
        "--remote-4q",
        default=DEFAULT_REMOTE_4Q,
        help="scp source for four-qubit BQSKit results (default: %(default)s)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="compare the existing local BQSKit snapshots without running scp",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the LaTeX table here instead of stdout",
    )
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid JSON from {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def fetch_results(remote: str, destination: Path) -> None:
    """Fetch to a temporary file, validate it, then replace the cached snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)

    try:
        subprocess.run(
            [
                "scp",
                "-F",
                "/dev/null",
                "-o",
                "BatchMode=yes",
                remote,
                str(temporary),
            ],
            check=True,
        )
        load_results(temporary)
        os.replace(temporary, destination)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"scp failed with exit status {exc.returncode}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def cnot_count(entry: Dict[str, Any], stage: str) -> Optional[int]:
    section = entry.get(stage)
    if not isinstance(section, dict):
        return None
    value = section.get("cnot_equiv")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{stage}.cnot_equiv is not numeric: {value!r}")
    if int(value) != value:
        raise RuntimeError(f"{stage}.cnot_equiv is not integral: {value!r}")
    return int(value)


def metadata(entry: Dict[str, Any]) -> Optional[tuple[int, int]]:
    initial = entry.get("init")
    if not isinstance(initial, dict):
        return None
    qubits = initial.get("qubits")
    cnots = initial.get("cnot_equiv")
    if not isinstance(qubits, int) or not isinstance(cnots, int):
        return None
    return qubits, cnots


def is_complete(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and metadata(entry) is not None
        and cnot_count(entry, "final") is not None
    )


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def display_name(filename: str) -> str:
    return filename[:-5] if filename.endswith(".qasm") else filename


def display_count(value: Optional[int]) -> str:
    return "--" if value is None else str(value)


def timing_seconds(entry: Dict[str, Any], field: str) -> Optional[float]:
    timing = entry.get("timing")
    if not isinstance(timing, dict):
        return None
    value = timing.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"timing.{field} is not numeric: {value!r}")
    if value < 0:
        raise RuntimeError(f"timing.{field} is negative: {value!r}")
    return float(value)


def display_minutes(seconds: Optional[float]) -> str:
    return "--" if seconds is None else f"{seconds / 60.0:.2f}"


def display_best_pair(
    osr_value: Optional[float],
    bqskit_value: Optional[float],
    formatter,
) -> tuple[str, str]:
    """Format a comparable OSR/BQSKit pair, bolding the lower value.

    Missing values are never treated as winners. Exact ties bold both values.
    Comparisons use the unrounded values, which is especially important for
    runtimes that display only two decimal places.
    """
    osr_display = formatter(osr_value)
    bqskit_display = formatter(bqskit_value)
    if osr_value is None or bqskit_value is None:
        return osr_display, bqskit_display
    if osr_value <= bqskit_value:
        osr_display = rf"\textbf{{{osr_display}}}"
    if bqskit_value <= osr_value:
        bqskit_display = rf"\textbf{{{bqskit_display}}}"
    return osr_display, bqskit_display


def _strip_before(name: str) -> str:
    """Remove ``_before`` (and similar suffixes) from a circuit filename."""
    # Match "_before" immediately before the extension, e.g. "tof_3_before.qasm" → "tof_3.qasm"
    return name.replace("_before.qasm", ".qasm")


def comparable_circuits(
    osr: Dict[str, Dict[str, Any]],
    bqskit: Dict[str, Dict[str, Any]],
) -> Iterable[tuple[str, Dict[str, Any], Dict[str, Any]]]:
    # Iterating BQSKit preserves the order in which its in-progress run finished.
    for name, bqskit_entry in bqskit.items():
        osr_entry = osr.get(name)
        display = name
        if osr_entry is None:
            # BQSKit keys may carry a "_before" suffix from the original
            # benchmark filenames; strip it and try again.
            stripped = _strip_before(name)
            if stripped != name:
                osr_entry = osr.get(stripped)
                if osr_entry is not None:
                    display = stripped
        if not is_complete(bqskit_entry) or not is_complete(osr_entry):
            continue
        assert isinstance(osr_entry, dict)
        assert isinstance(bqskit_entry, dict)
        if metadata(osr_entry) != metadata(bqskit_entry):
            raise RuntimeError(
                f"{display}: OSR metadata {metadata(osr_entry)} does not match "
                f"BQSKit metadata {metadata(bqskit_entry)}"
            )
        yield display, osr_entry, bqskit_entry


def begin_table(stream: TextIO) -> None:
    print(r"\begin{table}[p]", file=stream)
    print(r"\centering", file=stream)
    print(r"\scriptsize", file=stream)


def end_table(stream: TextIO, caption: str, label: str) -> None:
    print(rf"\caption{{{caption}}}", file=stream)
    print(rf"\label{{{label}}}", file=stream)
    print(r"\end{table}", file=stream)


def write_comparison_tables(
    stream: TextIO,
    osr: Dict[str, Dict[str, Any]],
    bqskit: Dict[str, Dict[str, Any]],
    partition_description: str,
    label_suffix: str,
) -> int:
    rows = list(comparable_circuits(osr, bqskit))
    if not rows:
        raise RuntimeError("no circuits have completed in both result files")

    totals = {
        "initial": 0,
        "osr": {stage: 0 for stage in STAGES},
        "bqskit": {stage: 0 for stage in STAGES},
    }
    paired_stage_counts = {stage: 0 for stage in STAGES}

    begin_table(stream)
    print(r"\begin{tabular}{lrr|rrr|rrr}", file=stream)
    print(r"\hline", file=stream)
    print(
        r"Circuit & Qubits & Initial CNOTs"
        r" & \multicolumn{3}{c|}{OSR CNOTs}"
        r" & \multicolumn{3}{c}{BQSKit CNOTs} \\",
        file=stream,
    )
    print(
        r" & & & A2A & Routed & Final & A2A & Routed & Final \\",
        file=stream,
    )
    print(r"\hline", file=stream)

    for name, osr_entry, bqskit_entry in rows:
        row_metadata = metadata(osr_entry)
        assert row_metadata is not None
        qubits, initial = row_metadata
        osr_counts = [cnot_count(osr_entry, stage) for stage in STAGES]
        bqskit_counts = [cnot_count(bqskit_entry, stage) for stage in STAGES]

        totals["initial"] += initial
        for index, stage in enumerate(STAGES):
            if osr_counts[index] is not None and bqskit_counts[index] is not None:
                totals["osr"][stage] += osr_counts[index]
                totals["bqskit"][stage] += bqskit_counts[index]
                paired_stage_counts[stage] += 1

        count_pairs = [
            display_best_pair(osr_value, bqskit_value, display_count)
            for osr_value, bqskit_value in zip(osr_counts, bqskit_counts)
        ]
        cells = [
            rf"\texttt{{{latex_escape(display_name(name))}}}",
            str(qubits),
            str(initial),
            *(pair[0] for pair in count_pairs),
            *(pair[1] for pair in count_pairs),
        ]
        print(" & ".join(cells) + r" \\", file=stream)

    print(r"\hline", file=stream)
    total_count_pairs = [
        display_best_pair(
            totals["osr"][stage],
            totals["bqskit"][stage],
            lambda value: str(int(value)),
        )
        for stage in STAGES
    ]
    total_cells = [
        "Total",
        "--",
        str(totals["initial"]),
        *(pair[0] for pair in total_count_pairs),
        *(pair[1] for pair in total_count_pairs),
    ]
    print(" & ".join(total_cells) + r" \\", file=stream)
    print(r"\hline", file=stream)
    print(r"\end{tabular}", file=stream)
    end_table(
        stream,
        (
            f"{partition_description} partition CNOT counts for OSR and "
            "BQSKit. Bold values are best within each paired stage."
        ),
        f"tab:osr-bqskit-{label_suffix}-counts",
    )
    print(
        f"% Compared {len(rows)} circuits completed by both methods.",
        file=stream,
    )
    for stage in STAGES:
        print(
            f"% Count {stage} totals include "
            f"{paired_stage_counts[stage]} circuits with counts from both methods.",
            file=stream,
        )

    runtime_totals = {
        "osr": {field: 0.0 for field in TIME_FIELDS},
        "bqskit": {field: 0.0 for field in TIME_FIELDS},
    }
    paired_time_counts = {field: 0 for field in TIME_FIELDS}

    print(file=stream)
    begin_table(stream)
    print(r"\begin{tabular}{l|rrr|rrr}", file=stream)
    print(r"\hline", file=stream)
    print(
        r"Circuit"
        r" & \multicolumn{3}{c|}{OSR time (min)}"
        r" & \multicolumn{3}{c}{BQSKit time (min)} \\",
        file=stream,
    )
    print(
        r" & A2A & Routing & Final & A2A & Routing & Final \\",
        file=stream,
    )
    print(r"\hline", file=stream)

    for name, osr_entry, bqskit_entry in rows:
        osr_times = [
            timing_seconds(osr_entry, field) for field in TIME_FIELDS
        ]
        bqskit_times = [
            timing_seconds(bqskit_entry, field) for field in TIME_FIELDS
        ]
        for index, field in enumerate(TIME_FIELDS):
            if osr_times[index] is not None and bqskit_times[index] is not None:
                runtime_totals["osr"][field] += osr_times[index]
                runtime_totals["bqskit"][field] += bqskit_times[index]
                paired_time_counts[field] += 1

        time_pairs = [
            display_best_pair(osr_value, bqskit_value, display_minutes)
            for osr_value, bqskit_value in zip(osr_times, bqskit_times)
        ]
        cells = [
            rf"\texttt{{{latex_escape(display_name(name))}}}",
            *(pair[0] for pair in time_pairs),
            *(pair[1] for pair in time_pairs),
        ]
        print(" & ".join(cells) + r" \\", file=stream)

    print(r"\hline", file=stream)
    runtime_total_pairs = [
        display_best_pair(
            runtime_totals["osr"][field],
            runtime_totals["bqskit"][field],
            display_minutes,
        )
        for field in TIME_FIELDS
    ]
    runtime_total_cells = [
        "Total",
        *(pair[0] for pair in runtime_total_pairs),
        *(pair[1] for pair in runtime_total_pairs),
    ]
    print(" & ".join(runtime_total_cells) + r" \\", file=stream)
    print(r"\hline", file=stream)
    print(r"\end{tabular}", file=stream)
    end_table(
        stream,
        (
            f"{partition_description} partition stage runtimes for OSR and "
            "BQSKit, in minutes. Bold values are best within each paired stage."
        ),
        f"tab:osr-bqskit-{label_suffix}-times",
    )
    print(
        "% Final time is timing.optimization: the post-routing final "
        "optimization stage.",
        file=stream,
    )
    for field in TIME_FIELDS:
        print(
            f"% Runtime {field} totals include "
            f"{paired_time_counts[field]} circuits with times from both methods.",
            file=stream,
        )
    return len(rows)


def four_qubit_osr_only(
    osr: Dict[str, Dict[str, Any]],
    bqskit: Dict[str, Dict[str, Any]],
) -> Iterable[tuple[str, Dict[str, Any]]]:
    """Yield completed four-qubit OSR entries absent or unfinished in BQSKit."""
    for name, osr_entry in osr.items():
        if not is_complete(osr_entry) or is_complete(bqskit.get(name)):
            continue
        assert isinstance(osr_entry, dict)
        bqskit_entry = bqskit.get(name)
        if isinstance(bqskit_entry, dict):
            bqskit_metadata = metadata(bqskit_entry)
            if (
                bqskit_metadata is not None
                and bqskit_metadata != metadata(osr_entry)
            ):
                raise RuntimeError(
                    f"{name}: four-qubit OSR metadata {metadata(osr_entry)} "
                    f"does not match BQSKit metadata {bqskit_metadata}"
                )
        yield name, osr_entry


def write_four_qubit_osr_only_table(
    stream: TextIO,
    osr: Dict[str, Dict[str, Any]],
    bqskit: Dict[str, Dict[str, Any]],
) -> int:
    rows = list(four_qubit_osr_only(osr, bqskit))
    totals = {
        "initial": 0,
        "counts": {stage: 0 for stage in STAGES},
        "times": {field: 0.0 for field in TIME_FIELDS},
    }
    count_totals = {stage: 0 for stage in STAGES}
    time_totals = {field: 0 for field in TIME_FIELDS}

    print(file=stream)
    begin_table(stream)
    print(r"\begin{tabular}{lrr|rr|rr|rr}", file=stream)
    print(r"\hline", file=stream)
    print(
        r"Circuit & Qubits & Initial CNOTs"
        r" & \multicolumn{2}{c|}{A2A}"
        r" & \multicolumn{2}{c|}{Routed}"
        r" & \multicolumn{2}{c}{Final} \\",
        file=stream,
    )
    print(
        r" & & & CNOTs & Time (min) & CNOTs & Time (min)"
        r" & CNOTs & Time (min) \\",
        file=stream,
    )
    print(r"\hline", file=stream)

    for name, entry in rows:
        row_metadata = metadata(entry)
        assert row_metadata is not None
        qubits, initial = row_metadata
        totals["initial"] += initial
        stage_cells = []
        for stage, time_field in zip(STAGES, TIME_FIELDS):
            count = cnot_count(entry, stage)
            seconds = timing_seconds(entry, time_field)
            if count is not None:
                totals["counts"][stage] += count
                count_totals[stage] += 1
            if seconds is not None:
                totals["times"][time_field] += seconds
                time_totals[time_field] += 1
            stage_cells.extend((display_count(count), display_minutes(seconds)))
        cells = [
            rf"\texttt{{{latex_escape(display_name(name))}}}",
            str(qubits),
            str(initial),
            *stage_cells,
        ]
        print(" & ".join(cells) + r" \\", file=stream)

    print(r"\hline", file=stream)
    total_stage_cells = []
    for stage, time_field in zip(STAGES, TIME_FIELDS):
        total_stage_cells.extend(
            (
                str(totals["counts"][stage])
                if count_totals[stage]
                else "--",
                display_minutes(totals["times"][time_field])
                if time_totals[time_field]
                else "--",
            )
        )
    print(
        " & ".join(
            ["Total", "--", str(totals["initial"]), *total_stage_cells]
        )
        + r" \\",
        file=stream,
    )
    print(r"\hline", file=stream)
    print(r"\end{tabular}", file=stream)
    end_table(
        stream,
        (
            "Completed four-qubit partition OSR results for circuits without "
            "a completed four-qubit partition BQSKit result. Counts and stage "
            "runtimes are combined because no paired comparison is available."
        ),
        "tab:osr-four-qubit-unpaired",
    )
    print(
        f"% Listed {len(rows)} completed four-qubit OSR-only circuits.",
        file=stream,
    )
    return len(rows)


def main() -> int:
    args = parse_args()
    try:
        if not args.no_fetch:
            fetch_results(args.remote, args.bqskit)
            print(f"Fetched {args.remote} -> {args.bqskit}", file=sys.stderr)
            fetch_results(args.remote_4q, args.bqskit_4q)
            print(
                f"Fetched {args.remote_4q} -> {args.bqskit_4q}",
                file=sys.stderr,
            )

        osr = load_results(args.osr)
        bqskit = load_results(args.bqskit)
        osr_4q = load_results(args.osr_4q)
        bqskit_4q = load_results(args.bqskit_4q)
        if args.output:
            with args.output.open("w", encoding="utf-8") as stream:
                count_3q = write_comparison_tables(
                    stream, osr, bqskit, "Three-qubit", "three-qubit"
                )
                print(file=stream)
                count_4q = write_comparison_tables(
                    stream, osr_4q, bqskit_4q, "Four-qubit", "four-qubit"
                )
                osr_only_4q = write_four_qubit_osr_only_table(
                    stream, osr_4q, bqskit_4q
                )
            print(
                f"Wrote five tables to {args.output}: "
                f"{count_3q} paired three-qubit circuits, "
                f"{count_4q} paired four-qubit circuits, and "
                f"{osr_only_4q} unpaired four-qubit OSR circuits",
                file=sys.stderr,
            )
        else:
            write_comparison_tables(
                sys.stdout, osr, bqskit, "Three-qubit", "three-qubit"
            )
            print()
            write_comparison_tables(
                sys.stdout, osr_4q, bqskit_4q, "Four-qubit", "four-qubit"
            )
            write_four_qubit_osr_only_table(
                sys.stdout, osr_4q, bqskit_4q
            )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
