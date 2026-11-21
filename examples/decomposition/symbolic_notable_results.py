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
from symbolic_bases import get_basis
from symbolic_gates import gate_definition
from symbolic_circuit_atlas import escape, qiskit_latex, compressed_operations





def load(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def effective_projective(row):
    return row.get("projective") or (row.get("phase_exact") if row.get("projective_omitted_as_no_better") else None)


def historical_results(row):
    for mode in ("phase_exact","projective"):
        result=row.get(mode)
        if result and result.get("status")=="verified_symbolic":yield result
    for previous in row.get("previous_attempts",[]):yield from historical_results(previous)
    if row.get("raw_attempt"):yield from historical_results(row["raw_attempt"])


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
        shorter=[result for result in historical_results(row)
                 if result["cost"]["entanglers"]==cost["entanglers"]
                 and result["cost"]["single_qubit_gates"]<cost["single_qubit_gates"]]
        alternative=min(shorter,key=lambda r:float(sp.sympify(r["cost"]["parameter_score"])),default=None)
        alternative_score=alternative["cost"]["parameter_score"] if alternative else None
        priority_tradeoff=bool(alternative and float(sp.sympify(cost["parameter_score"]))<float(sp.sympify(alternative_score)))
        counts[status] += 1
        rows.append(dict(gate=row["gate"], basis=row["basis"], phase_policy="projective",
            entanglers=cost["entanglers"], local_gates=cost["single_qubit_gates"],
            exact_parameter_score=cost["parameter_score"],
            sdk_entanglers=baseline["entanglers"], sdk_local_gates=local,
            delta_entanglers=de, delta_local_gates=dl, status=status,
            sdk_input=baseline.get("input"), sdk_parameter_score=None,
            shorter_verified_alternative_score=alternative_score,
            smaller_local_count_has_higher_verified_score=priority_tradeoff,
            score_comparison="not_certified_from_floating_SDK_angles", novelty_claim=False))
    return sorted(rows, key=lambda r:(r["gate"].casefold(), r["basis"])), dict(counts)


def local_layer_audit(archive):
    """Local-count opportunities, without a claim about the secondary score.

    E entanglers split each wire into at most E+1 local blocks. U3/PhasedXZ
    represent each block projectively; one extra local absorbs the scalar
    for literal equality. Fewer locals need not mean a lower original score.
    """
    rows=[]
    for row in archive["rows"]:
        if gate_definition(row["gate"]).num_qubits!=2:continue
        if not ({"U3","U","PhasedXZ"}&set(get_basis(row["basis"]).single_qubit_gates)):continue
        for mode in ("phase_exact","projective"):
            result=row.get(mode) if mode=="phase_exact" else effective_projective(row)
            if not result or result.get("status")!="verified_symbolic":continue
            cost=result["cost"]
            cap=2*(cost["entanglers"]+1)+(mode=="phase_exact")
            if cost["single_qubit_gates"]>cap:
                rows.append(dict(gate=row["gate"],basis=row["basis"],phase_policy=mode,
                    entanglers=cost["entanglers"],local_gates=cost["single_qubit_gates"],
                    local_layer_count_cap=cap,parameter_score_improvement_proved=False,
                    scope="local-count opportunity; native basis and entangler count preserved"))
    return sorted(rows,key=lambda r:(r["gate"].casefold(),r["basis"],r["phase_policy"]))


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
    text=r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,qcircuit,hyperref}
\title{Symbolic circuit reference: selected identities and comparisons}
\author{Gregory Morse \and Peter Rakyta}\date{7 November 2026}
\begin{document}\maketitle
This appendix records exact identities and scoped comparisons. The revised
manuscript and its current independent compiler table are authoritative.
The atlas is not claimed to dominate every tool or publication.
\section{Controlled fermionic SWAP}
The literal CX/U3 circuit has four CNOTs and six locals. The interaction
count is optimal among arbitrary Ising-class interactions without ancillas;
the local count is an upper bound. BQSKit independently finds four-CNOT,
six-local circuits, so discovering the upper bound is not a new compiler
advantage. The lower theorem extends to all controlled balanced involutions.
The three-qubit hardware family of Warren et al.\ is related physical
context, not a proof of these CNOT counts~\cite{fermionic}.
\section{Sycamore phase contract}
For $\mathrm{SYC}=\mathrm{fSim}(\pi/2,\pi/6)$, the exact P/CX circuit and
the projective Rz/CX circuit below each use three entanglers and three
locals. In the Rz circuit, $\mathrm{SYC}=e^{-7i\pi/24}C$.
The phase-polynomial certificate proves the three-phase lower bound only
in the CNOT/P subalphabet; it does not establish an unrestricted local
minimum. The former tomography-figure comparison has been withdrawn.
'''
    for basis in ('cx_h_phase','ibm_cx'):
        row=next(r for r in archive['rows'] if r['gate']=='SYC' and r['basis']==basis)
        result=row['phase_exact'] if basis=='cx_h_phase' else effective_projective(row)
        text+=r'\paragraph{'+escape(basis)+r'}\begin{center}\resizebox{.9\linewidth}{!}{'+qiskit_latex(2,compressed_operations(result['operations']))+r'}\end{center}'+'\n'
    text+=r'''\section{Current independent fixed-target comparison}
The table gives $E/L$ in CX/U3 and the best verified recorded route for each
tool. Inputs are independent library words or target matrices, not the
atlas output. A dash excludes all recorded outputs on that route that fail
its replay threshold or budget. Quartz and QUESO optimize total NAM gates;
their locals are then fused to U3 without changing the entangling skeleton.
'''
    text+=(ROOT/'docs/symbolic_manuscript/generated/strong_sdks.tex').read_text()
    text+=r'''BQSKit matches fFredkin and beats the SSWAP local count. tket and
BQSKit beat the C3X local count. Quartz's 30-CX C4X circuit beats the current
36-CX atlas entry. These losses are retained. The earlier eleven SDK
entangler wins were all fFredkin in different alphabets.
\section{Historical broad local-count audit}
The CSV files retain comparisons against the older, explicitly identified
SDK routes, including source-fed routes. They are not comparisons against
all compiler strategies. Counts and parameter-description score are different
objectives; a smaller $L$ alone does not prove a better archived score.
'''
    losses=[r for r in report['comparisons'] if r['status']=='equal_E_more_locals']
    text+=r'\begin{center}\begin{tabular}{lr}\toprule Historical status & Cells\\\midrule'+'\n'
    for name,count in sorted(report['comparison_summary'].items()):text+=escape(name)+' & '+str(count)+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}\end{center}'+'\n'
    text+=r'''\section{Established native bounds and proof scope}
The universal two-B construction~\cite{b} and the SQiSW two-pulse Weyl
region and three-pulse universal construction~\cite{sq} are prior results.
The released exact charts implement them with explicit scalar handling and
branch definitions. The new invariant iff proof rewrites the known SQiSW
region; it is not a new universal pulse cap. The 192 one-qubit phase-resolved
Clifford circuits are a finite reference, not a large unexplored class.
\section{Reproduction}
The companion JSON and CSVs retain archive hashes, exact identities, older
comparison settings, and every local-count regression. The revised review
bundle additionally includes the stronger SDK/Quartz/QUESO outputs, failures,
optimality ledger, theorem checks, and both complete atlas PDFs and sources.
UniversalQCompiler has no executed comparison because its licensed kernel
was unavailable. No numerical sample proves a symbolic family identity or
an unrestricted global minimum.
\begin{thebibliography}{9}
\bibitem{fermionic} C. W. Warren, J. Fern\'andez-Pend\'as, S. Ahmed, et al.,
Extensive characterization and implementation of a family of three-qubit
gates at the coherence limit, npj Quantum Information 9, 44 (2023),
\href{https://doi.org/10.1038/s41534-023-00711-x}{doi:10.1038/s41534-023-00711-x}.
\bibitem{b} J. Zhang, J. Vala, S. Sastry, and K. B. Whaley, Minimum
construction of two-qubit quantum operations, Physical Review Letters 93,
020502 (2004), \href{https://arxiv.org/abs/quant-ph/0312193}{arXiv:quant-ph/0312193}.
\bibitem{sq} C. Huang et al., Quantum instruction set design for performance,
Physical Review Letters 130, 070601 (2023),
\href{https://arxiv.org/abs/2105.06074}{arXiv:2105.06074}.
\end{thebibliography}\end{document}
'''
    output.write_text('\n'.join(line.expandtabs().rstrip() for line in text.splitlines())+'\n')

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
        literature_uniform_bound_summary=archive.get("literature_uniform_bound_summary",{}),
        local_layer_opportunities=local_layer_audit(archive),
        native_examples=[dict(gate=row["gate"],basis=row["basis"],phase_exact=row.get("phase_exact"),
                              projective=effective_projective(row)) for row in archive["rows"]
            if (row["gate"],row["basis"]) in {("SYC","berkeley_b"),("SSWAP","berkeley_b"),("SSWAPdg","berkeley_b"),
                *((name,basis) for name in ("B","DCX","SSWAP","SSWAPdg","SYC","CZS")
                  for basis in ("sqrt_iswap_u","google_sqrt_iswap")),("CZS","berkeley_b"),
                ("CP","sqrt_iswap_u"),("CU","sqrt_iswap_u"),("IonQMS","sqrt_iswap_u")}],
        sycamore=dict(independent_sdk=sdk,
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
    write_csv(args.output_dir/"native_local_layer_opportunities.csv",report["local_layer_opportunities"],
        fields=["gate","basis","phase_policy","entanglers","local_gates","local_layer_count_cap",
                "parameter_score_improvement_proved","scope"])
    write_tex(args.output_dir/"notable_results.tex", report, archive)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
