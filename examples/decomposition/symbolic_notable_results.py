#!/usr/bin/env python3
"""Produce a cited local-resource audit and a standalone LaTeX evidence ledger."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"squander/decomposition"))
import sympy as sp
from symbolic_gates import make_gate, decomposition_sequence
from symbolic_identity import prove_circuit_identity
from symbolic_sdk_lift import lift_sdk_candidate
from symbolic_circuit_atlas import escape, qiskit_latex, compressed_operations


PAPER = dict(title="Full quantum tomography study of Google's Sycamore gate on IBM's quantum computers",
    authors="Muhammad AbuGhanem and Hichem Eleuch", year=2024, figure=3,
    url="https://link.springer.com/article/10.1140/epjqt/s40507-024-00248-8",
    figure_url="https://media.springernature.com/full/springer-static/image/art%3A10.1140%2Fepjqt%2Fs40507-024-00248-8/MediaObjects/40507_2024_248_Fig3_HTML.jpg",
    figure_sha256="231c3a1a327190d3118514348e10013874229b7ce5465792236f5abfd1752807",
    local_gate_types_by_wire=[
        ["Rz", "Sx", "Rz", "Sx", "Rz", "Sx", "Rz", "Sx", "Sx", "Rz"],
        ["Rz", "Sx", "Rz", "Sx", "Rz", "Rz", "Sx", "Rz", "Sx", "Rz",
         "Sx", "Rz", "Sx", "Rz", "Rz", "Sx", "Rz"]],
    count_method="Manual transcription of gate TYPES in Figure 3; measurements excluded",
    angle_reconstruction=False, phase_policy="projective", novelty_claim=False)


def load(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def effective_projective(row):
    return row.get("projective") or (row.get("phase_exact") if row.get("projective_omitted_as_no_better") else None)


def local_audit(archive, baselines):
    sdk = {(r["gate"], r["basis"]):r for r in baselines["rows"] if r["status"] == "verified_numeric"}
    rows, counts = [], Counter()
    for row in archive["rows"]:
        if row["target_scope"] != "fixed_gate":
            continue
        current = effective_projective(row)
        baseline = sdk.get((row["gate"], row["basis"]))
        if not current or current["status"] != "verified_symbolic" or baseline is None:
            continue
        local = sum(len(op["wires"]) == 1 for op in baseline["operations"])
        cost = current["cost"]
        de, dl = cost["entanglers"]-baseline["entanglers"], cost["single_qubit_gates"]-local
        status = "fewer_entanglers" if de < 0 else "more_entanglers" if de > 0 else (
            "equal_E_fewer_locals" if dl < 0 else "equal_E_more_locals" if dl > 0 else "equal_E_equal_locals")
        counts[status] += 1
        rows.append(dict(gate=row["gate"], basis=row["basis"], phase_policy="projective",
            entanglers=cost["entanglers"], local_gates=cost["single_qubit_gates"],
            exact_parameter_score=cost["parameter_score"],
            sdk_entanglers=baseline["entanglers"], sdk_local_gates=local,
            delta_entanglers=de, delta_local_gates=dl, status=status,
            sdk_input=baseline.get("input"), sdk_parameter_score=None,
            score_comparison="not_certified_from_floating_SDK_angles", novelty_claim=False))
    return sorted(rows, key=lambda r:(r["gate"].casefold(), r["basis"])), dict(counts)


def phase_polynomial_certificate():
    """Three nonzero parity coefficients cannot be represented by two phases.

    CNOT+P circuits attach phases only to the three nonzero binary parities.
    Changing any truth-table angle by 2*pi changes solved coefficients by
    integer multiples of pi. All three residues here remain nonzero mod pi.
    This excludes two locals within this subalphabet, not arbitrary U(2).
    """
    matrix = sp.Matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    phases = sp.Matrix([-sp.pi/2, -sp.pi/2, -sp.pi/6])
    coefficients = matrix.inv()*phases
    lifts = matrix.inv()*2
    assert all(v.is_Integer for v in lifts)
    assert all((v/sp.pi).is_Rational and (v/sp.pi).is_Integer is False for v in coefficients)
    word = decomposition_sequence("SYC")
    assert prove_circuit_identity(make_gate("SYC"), word)
    assert sum(len(op[-1]) == 1 for op in word) == 3
    return dict(method="nonzero_binary_parity_phase_coefficients",
        local_gate_lower_bound=3, achieved_local_gates=3,
        coefficient_angles=[str(v) for v in coefficients],
        truth_table_modulo_lift="2*pi lifts change coefficients only by integer*pi",
        proof_scope="Ancilla-free CNOT+P circuits with fixed input/output wires; arbitrary CNOT count",
        arbitrary_U2_local_minimum_proved=False)


def independent_sycamore_sdk():
    from symbolic_local_baselines import normalize_baseline
    from symbolic_references import _numeric, _transpile
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    import qiskit
    import numpy as np
    target = _numeric(make_gate("SYC"))
    rows = []
    for basis in ("cx_u", "cx_rz_ry", "ibm_cx"):
        circuit = QuantumCircuit(2)
        circuit.append(UnitaryGate(target), [0, 1])
        result = _transpile(circuit, basis, 219)
        error = float(np.max(np.abs(Operator(result).data-target)))
        if not error < 2e-10:
            raise ValueError("Independent SDK Sycamore output failed phase-sensitive replay")
        operations = [dict(gate=i.operation.name, parameters=[float(p) for p in i.operation.params],
            wires=[result.find_bit(q).index for q in i.qubits]) for i in result.data]
        baseline = dict(status="verified_numeric", basis=basis, num_qubits=2, input="independent_matrix_synthesis",
            entanglers=sum(len(op["wires"]) == 2 for op in operations),
            single_qubit_gates=sum(len(op["wires"]) == 1 for op in operations),
            global_phase=float(result.global_phase), operations=operations, max_error=error)
        baseline = normalize_baseline(baseline)
        baseline["single_qubit_gates"] = sum(len(op["wires"]) == 1 for op in baseline["operations"])
        baseline["exact_lift"] = lift_sdk_candidate(make_gate("SYC"), baseline, basis)
        rows.append(baseline)
    return dict(qiskit_version=qiskit.__version__, seed=219, optimization_level=3,
                approximation_degree=1.0, ancillas=0, qubits_initially_zero=False,
                output_wire_order="fixed", supplied_our_circuit=False, rows=rows)


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tex(output, report, archive):
    syc = {r["basis"]:r for r in archive["rows"] if r["gate"] == "SYC"}
    drawings = []
    for basis, label in (("cx_h_phase", "Literal-phase CNOT/P construction"),
                         ("ibm_cx", "Same three-angle construction using Rz; scalar shown below")):
        result = syc[basis]["phase_exact"] if basis == "cx_h_phase" else effective_projective(syc[basis])
        drawings.append(r"\noindent\begin{minipage}{\linewidth}\paragraph{"+label+r"}\begin{center}\resizebox{0.75\linewidth}{!}{"+
            qiskit_latex(2, compressed_operations(result["operations"]))+r"}\end{center}\end{minipage}\par\medskip")
    sdk = report["sycamore"]["independent_sdk"]["rows"]
    table = [r"Published Figure 3 & Rz/SX/CX & 3 & 27 & projective \\",
             r"Our circuit & Rz/SX/CX & 3 & 3 & projective \\",
             r"Our circuit & P/H/CX & 3 & 3 & literal phase \\"]
    for row in sdk:
        table.append("Qiskit matrix input & "+escape(row["basis"])+" & "+str(row["entanglers"])+" & "+
                     str(row["single_qubit_gates"])+r" & scalar metadata \\")
    winners = [r for r in report["comparisons"] if r["delta_entanglers"] < 0]
    local = sorted((r for r in report["comparisons"] if r["status"] == "equal_E_fewer_locals"),
                   key=lambda r:(r["delta_local_gates"], r["gate"], r["basis"]))
    def resource_table(rows):
        return "\n".join(escape(r["gate"])+" & "+escape(r["basis"])+" & "+
            f"{r['entanglers']}/{r['sdk_entanglers']} & {r['local_gates']}/{r['sdk_local_gates']}"+r" \\" for r in rows)
    counts = report["comparison_summary"]
    preamble = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,qcircuit,hyperref}
\usepackage[T1]{fontenc}
\providecommand{\ket}[1]{\lvert #1\rangle}
\title{Symbolic decomposition: notable results and comparison evidence}
\author{SQUANDER symbolic decomposition project}
\date{6 November 2026}
\begin{document}\maketitle
\noindent Evidence ledger for a future appendix, not a completed optimality or novelty claim.
All counts preserve the recorded native basis, use no ancillas, and fix input/output wires.
The objective is entanglers first, the original parameter-sensitive score second,
and local gate count third. A smaller local count alone does not establish a lower score.
\section{Sycamore: a concrete local-gate reduction}
We use $\mathrm{SYC}=\mathrm{fSim}(\pi/2,\pi/6)$ with odd-parity swap amplitudes $-i$
and $\ket{11}$ phase $e^{-i\pi/6}$. Figure 3 of AbuGhanem and Eleuch~\cite{syc}
contains 15 Rz gates and 12 SX gates: 27 local operations and three CNOTs.
This is a manual count of the displayed gate boxes; the two measurements are excluded.
The figure has rounded angle labels, so its parameter-sensitive symbolic score and
literal scalar phase are not reconstructed or asserted here.
Our circuit has three locals and three CNOTs: an $8/9$ (88.9\%) local-count reduction
against that displayed implementation, in the same Rz/SX/CX alphabet projectively.
This does not assert that the displayed circuit is the best published local construction.
\begin{center}\begin{tabular}{lllll}\toprule
Construction & Alphabet & E & L & Phase contract \\\midrule
"""
    text = preamble+"\n".join(table)+r"\bottomrule\end{tabular}\end{center}"+"\n"+"\n".join(drawings)
    lifted = next(row["exact_lift"] for row in sdk if row["basis"] == "ibm_cx")
    ours = effective_projective(syc["ibm_cx"])
    if lifted["status"] == "verified_symbolic":
        ours_score = sp.sympify(ours["cost"]["parameter_score"])
        sdk_score = sp.sympify(lifted["cost"]["parameter_score"])
        text += (r"\paragraph{Matched score comparison.} In the same Rz/SX/CX basis, "
                 r"the independently synthesized SDK circuit has an exact rational-$\pi$ lift "
                 r"whose complete identity is proved. The original score is $"+sp.latex(ours_score)+
                 r"\approx "+f"{float(ours_score):.3f}"+r"$ for ours and $"+sp.latex(sdk_score)+
                 r"\approx "+f"{float(sdk_score):.3f}"+r"$ for the SDK circuit. Both have three CNOTs; "
                 r"the score includes the same entangler contribution and no free physical GP. "+"\n")
    text += r"""
The P circuit is exact without a GP operation. Replacing each P by Rz gives
$\mathrm{SYC}=e^{-7i\pi/24}C_{Rz}$; this scalar is projective metadata.
Literal equality in the Rz/SX/CX alphabet is obstructed by its determinant group.
The phase polynomial has coefficients $(-\pi/12,-\pi/12,-5\pi/12)$ on
$x_0,x_1,x_0\oplus x_1$. All are nonzero modulo $\pi$. Adding $2\pi$
to truth-table phases changes these coefficients only by integer multiples of $\pi$.
Therefore at least three phase gates are necessary within the CNOT/P subalphabet,
even if extra CNOTs are allowed. Our three-phase circuit attains that bound.
This is not a minimum-local proof over arbitrary U(2) gates or SX-containing circuits.

The independent SDK rows use matrix input, Qiskit 2.2.3, level 3, seed 219,
and approximation degree one, followed by native one-qubit block cleanup and
phase-sensitive replay. Their circuits and numerical replay errors are stored
in the evidence JSON. They are separate from the main baseline, which also lets
Qiskit optimize our supplied exact source. That source-fed baseline reproduces
the three-local circuit, so its tie is not independent discovery by Qiskit.
Where the SDK angles have an exact rational-$\pi$ lift, the full identity is proved
before assigning a symbolic score. Other floating angles are never rationalized
and assigned an invented exact score.
\section{The eleven entangler improvements}
The eleven earlier SDK improvements all concern controlled fermionic SWAP
(fFredkin) in different bases. They are completed constructions, separate from
the B/SQiSW universal-bound gaps. They are not eleven different gate discoveries.
Controlled fermionic SWAP is distinct from ordinary Fredkin; no comparison to
the latter's lower bounds is made. Existence of this operation in the literature
does not settle optimal decomposition cost~\cite{fermionic}.
\begin{center}\begin{tabular}{llll}\toprule
Target & Basis & E ours/SDK & L ours/SDK \\\midrule
"""+resource_table(winners)+r"\bottomrule\end{tabular}\end{center}"
    text += r"\section{Local-gate audit beyond Sycamore}"+"\n"
    text += (f"Across {len(report['comparisons'])} eligible fixed-target/basis comparisons, "
             f"there are {counts.get('equal_E_fewer_locals',0)} lower local counts, "
             f"{counts.get('equal_E_equal_locals',0)} ties and "
             f"{counts.get('equal_E_more_locals',0)} higher local counts at equal E. ")
    text += r"""Aliases and related basis presets are counted separately. Native one-qubit SDK
cleanup removes identity rotations introduced by the manual U-to-ZYZ adapter
before counting; savings from that adapter artifact are not credited.
These are SDK comparisons,
not novelty claims or an exhaustive literature review. The full CSV includes every
compared case, including regressions; a separate local-gap CSV is the work list.
The table shows the largest local-count savings. E and L are raw physical counts;
the exact score is recorded separately in the CSV/JSON.
\begin{center}\begin{longtable}{llll}\toprule
Target & Basis & E ours/SDK & L ours/SDK \\\midrule\endhead
"""+resource_table(local[:25])+r"\bottomrule\end{longtable}\end{center}"
    text += r"""
\section{Other useful reference results}
The separate Clifford atlas contains all 192 phase-resolved one-qubit Clifford
targets for each supported local alphabet. H/S has a global shortest-path
certificate; other alphabets retain their narrower recorded word-search scope.
The exact phase examples include a Cartan kernel with $a=b=c=\pi/8$: three
CNOTs projectively, four with literal phase and SU(2)-only Rz/Ry locals.
Likewise, Rxx$(\pi)$ requires zero entanglers projectively and one with literal
phase in the Rxx/Rz/Ry basis. These are certified resource-policy distinctions,
not claims of priority over the literature.

\section{Outstanding optimization and comparison work}
Classification covers all $327\times32$ cells in both phase modes. This means a
verified construction or a scoped obstruction, not global optimality.
The two-B~\cite{b} and three-SQiSW~\cite{sq} bounds still identify 158 over-cap
constructions per phase mode in the current archive. These are distinct from
the local-count gaps above. Further minimum proofs and score optimization are
needed even for many circuits below those caps. No competitor-wide dominance
or exhaustive novelty claim is justified.
UniversalQCompiler has zero executed comparisons pending a suitable Wolfram kernel.
Wolfram's free developer engine is not automatically a free paper-benchmark license:
its FAQ distinguishes development/testing from producing papers for organizational
use. An existing university/site license or written research permission is a
possible route~\cite{wolfram}; no new access has been obtained.

\section{Reproduction and evidence scope}
The companion JSON identifies the exact archive and SDK baseline SHA256 hashes,
stores the independent Sycamore SDK circuits, and records the published figure's
gate-type transcription and image hash. \texttt{local\_comparisons.csv} contains
all eligible comparisons; \texttt{local\_optimization\_gaps.csv} retains regressions.
Regenerate with \texttt{symbolic\_notable\_results.py}; compile this source with Tectonic.
No numerical SDK sample certifies a free-parameter identity or global minimum.
\begin{thebibliography}{9}
\bibitem{syc} M. AbuGhanem and H. Eleuch, EPJ Quantum Technology 11, 36 (2024),
\href{https://doi.org/10.1140/epjqt/s40507-024-00248-8}{doi:10.1140/epjqt/s40507-024-00248-8}, Figure 3.
\bibitem{fermionic} \href{https://doi.org/10.1038/s41534-023-00711-x}{npj Quantum Information (2023), doi:10.1038/s41534-023-00711-x}.
\bibitem{b} J. Zhang et al., Phys. Rev. Lett. 93, 020502 (2004),
\href{https://arxiv.org/abs/quant-ph/0312193}{arXiv:quant-ph/0312193}.
\bibitem{sq} C. Huang et al., Quantum Instruction Set Design for Performance,
\href{https://arxiv.org/abs/2105.06074}{arXiv:2105.06074}.
\bibitem{wolfram} Wolfram Engine FAQ, licensing and terms of use,
\url{https://www.wolfram.com/engine/faq/}, checked 6 November 2026.
\end{thebibliography}\end{document}
"""
    output.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive, baselines = load(args.archive), load(args.baselines)
    rows, counts = local_audit(archive, baselines)
    sdk = independent_sycamore_sdk()
    report = dict(schema="symbolic-notable-results-v1",
        archive=dict(path=str(args.archive), sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest()),
        sdk_baselines=dict(path=str(args.baselines), sha256=hashlib.sha256(args.baselines.read_bytes()).hexdigest()),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_summary=counts, comparisons=rows,
        sycamore=dict(published=PAPER, independent_sdk=sdk,
            archived_constructions=[dict(basis=row["basis"], phase_exact=row.get("phase_exact"),
                projective=effective_projective(row)) for row in archive["rows"]
                if row["gate"] == "SYC" and row["basis"] in {"cx_u", "cx_h_phase", "ibm_cx"}],
            phase_polynomial_certificate=phase_polynomial_certificate()),
        novelty_claim=False, all_local_scores_globally_minimal=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"notable_results.json").write_text(json.dumps(report, indent=2)+"\n")
    write_csv(args.output_dir/"local_comparisons.csv", rows)
    write_csv(args.output_dir/"local_savings.csv",
              [r for r in rows if r["status"] == "equal_E_fewer_locals"], fields=list(rows[0]))
    write_csv(args.output_dir/"local_optimization_gaps.csv",
              [r for r in rows if r["status"] == "equal_E_more_locals"], fields=list(rows[0]))
    write_tex(args.output_dir/"notable_results.tex", report, archive)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
