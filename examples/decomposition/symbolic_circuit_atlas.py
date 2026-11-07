#!/usr/bin/env python3
"""Render archived symbolic circuits with Qiskit into indexed LaTeX atlases.

Gate pages contain 32 basis panels. A separate Clifford volume contains all
192 exact phases in each distinct local alphabet, 24 circuits per page.
Only verified constructions are drawn; open cases keep their actual status.
This tool consumes trusted archives produced by symbolic_synthesis_archive.py.
"""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "squander/decomposition"))
import sympy as sp
from symbolic_archive import symbolic_cases
from symbolic_bases import BASIS_SETS
from symbolic_clifford_search import group_data, clifford_path, verify_shortest_path_certificate


def escape(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}"}
    return "".join(replacements.get(char, char) for char in str(text))


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def ordered_targets(report, selection=()):
    names = sorted({row["gate"] for row in report["rows"]}, key=lambda name: (name.casefold(), name))
    if selection:
        unknown = set(selection) - set(names)
        if unknown:
            raise ValueError(f"Unknown target names: {sorted(unknown)}")
        names = [name for name in names if name in selection]
    return names


def gate_grid(report, name):
    rows = {row["basis"]: row for row in report["rows"] if row["gate"] == name}
    return [(basis, rows.get(basis)) for basis in sorted(BASIS_SETS)]


def phase_variants(row):
    """Retain exact first, and projective only when archived as strictly cheaper."""
    if row is None:
        return []
    return [(mode, row[mode]) for mode in ("phase_exact", "projective")
            if row.get(mode) and row[mode]["status"] == "verified_symbolic"]


def compressed_operations(operations):
    """Display consecutive identical pulses as an explicitly counted repeat."""
    groups = []
    for operation in operations:
        if groups and operation == groups[-1][0]:
            groups[-1] = operation, groups[-1][1] + 1
        else:
            groups.append((operation, 1))
    return groups


@lru_cache(maxsize=32768)
def parameter_latex(value):
    # Catalog parameters such as gamma and beta collide with SymPy function
    # names. Retain the real Symbols used by the actual target families.
    return sp.latex(sp.sympify(value, locals=parameter_symbols()))


@lru_cache(maxsize=1)
def parameter_symbols():
    symbols = {}
    def visit(value):
        if isinstance(value, sp.Basic):
            symbols.update((str(symbol), symbol) for symbol in value.free_symbols)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    for _, parameters, _ in symbolic_cases():
        visit(parameters)
    return symbols


def operation_label(operation, repeat=1, aliases=None):
    name = operation["gate"]
    labels = {"U3": r"U_3", "U": "U", "Rz": r"R_z", "Ry": r"R_y", "Rx": r"R_x",
              "Rxx": r"R_{XX}", "Rzz": r"R_{ZZ}", "Rzx": r"R_{ZX}",
              "Sx": r"\sqrt{X}", "Sxdg": r"\sqrt{X}^{\dagger}",
              "Sdg": r"S^{\dagger}", "Tdg": r"T^{\dagger}",
              "SiSWAP": r"\sqrt{\mathrm{iSWAP}}", "SSWAP": r"\sqrt{\mathrm{SWAP}}"}
    label = labels.get(name, r"\operatorname{" + escape(name) + "}")
    if operation["parameters"]:
        label += r"\left(" + ",".join((aliases or {}).get(p,parameter_latex(p)) for p in operation["parameters"]) + r"\right)"
    if repeat > 1:
        label = r"\left[" + label + r"\right]^{\times " + str(repeat) + "}"
    return "$" + label + "$"


def qiskit_latex(width, operations, aliases=None):
    """Use Qiskit's LaTeX drawer, with exact parameter text rather than floats."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    circuit = QuantumCircuit(width)
    displaytext = {}
    for operation, repeat in operations:
        name, wires = operation["gate"], operation["wires"]
        if name == "GP":
            raise ValueError("A physical GP gate cannot appear in the atlas")
        if repeat == 1 and not operation["parameters"] and name in {"CX", "CNOT", "CZ"}:
            getattr(circuit, "cz" if name == "CZ" else "cx")(*wires)
        elif repeat == 1 and not operation["parameters"] and name in {"H", "X", "Y", "Z", "S", "T", "Sdg", "Tdg"}:
            getattr(circuit, name.lower())(*wires)
        else:
            label = operation_label(operation, repeat, aliases)
            token = "atlas" + hashlib.sha256(label.encode()).hexdigest()[:16]
            displaytext[token] = label
            circuit.append(Gate(token, len(wires), []), wires)
    source = circuit.draw(output="latex_source", idle_wires=True, initial_state=False,
                          style={"displaytext": displaytext})
    body = source.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0].strip()
    return re.sub(r"@C=[0-9.]+em @R=[0-9.]+em", "@C=0.55em @R=0.5em", body)


class CircuitStore:
    def __init__(self, output, fold=18, long_fold=54):
        self.output = output
        self.fold = fold
        self.long_fold = long_fold
        self.directory = output / "circuits"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = {}
        self.aliases = {}
        self.programs = {}
        self.program_outputs = defaultdict(dict)

    def register_program(self, program):
        digest=hashlib.sha256(json.dumps(program,sort_keys=True).encode()).hexdigest()
        if digest not in self.programs:
            index=len(self.programs)+1
            self.programs[digest]=(index,program)
            directory=self.output/'programs'
            directory.mkdir(exist_ok=True)
            (directory/f'program-{index}.json').write_text(json.dumps(program,indent=2)+'\n')
        return self.programs[digest][0]

    def program_expression(self, value, program, compact=True):
        index=self.register_program(program)
        environment=dict(parameter_symbols())
        names={}
        for j,row in enumerate(program['bindings']):
            symbol=sp.Symbol(row['symbol'],**({'real':True} if row['real'] else {}))
            environment[row['symbol']]=symbol
            names[symbol]=r'\xi_{'+str(index)+','+str(j)+'}'
        expression=sp.sympify(value,locals=environment)
        formula=sp.latex(expression,symbol_names=names)
        if compact and len(formula)>100:
            # Native PhasedXZ lowering can repeat rational angle expressions.
            # Keep the exact output in a separate scoped glossary instead of
            # overflowing XeTeX's maximum box dimension inside Qcircuit.
            outputs=self.program_outputs[index]
            if value not in outputs:
                outputs[value]=r'\upsilon_{'+str(index)+','+str(len(outputs))+'}'
            return outputs[value]
        return formula

    def expression(self,value):
        formula=parameter_latex(value)
        if len(formula)<=60:
            return formula
        if value not in self.aliases:
            self.aliases[value]=r"\eta_{"+str(len(self.aliases)+1)+"}"
        return self.aliases[value]

    def render(self, width, operations, expression_program=None):
        displayed={value:(self.program_expression(value,expression_program) if expression_program else self.expression(value))
                   for op in operations for value in op["parameters"]}
        grouped = compressed_operations(operations)
        fold=self.long_fold if width>=4 and len(grouped)>4*self.fold else self.fold
        key = hashlib.sha256(json.dumps([3, width, operations, fold, displayed], sort_keys=True).encode()).hexdigest()[:24]
        if key in self.paths:
            return self.paths[key]
        chunks = [grouped[start:start+fold] for start in range(0, len(grouped), fold)] or [[]]
        paths = []
        for index, chunk in enumerate(chunks):
            relative = f"circuits/{key}-{index}.tex"
            path = self.output / relative
            if not path.exists():
                path.write_text(qiskit_latex(width, chunk, displayed))
            paths.append(relative)
        self.paths[key] = paths
        return paths


def preamble(title, paper="gates"):
    size = "paperwidth=841mm,paperheight=594mm" if paper == "gates" else "a3paper,landscape"
    return r"""\documentclass[10pt]{article}
\usepackage[""" + size + r""",margin=13mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,graphicx,xcolor,array,multicol,amsmath,amssymb}
\usepackage[braket,qm]{qcircuit}
\usepackage[unicode,colorlinks=true,linkcolor=ink,urlcolor=ink]{hyperref}
\definecolor{ink}{HTML}{203447}
\definecolor{muted}{HTML}{667784}
\definecolor{accent}{HTML}{B17B39}
\definecolor{exact}{HTML}{215F7A}
\definecolor{projective}{HTML}{916325}
\definecolor{rule}{HTML}{CFD7DD}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\pagestyle{plain}
\hypersetup{pdftitle={""" + escape(title) + r"""},pdfauthor={SQUANDER symbolic decomposition project}}
\newsavebox{\circuitbox}
\newsavebox{\panelbox}
\newsavebox{\contentbox}
\newcommand{\circuit}[1]{\sbox{\circuitbox}{#1}%
\ifdim\wd\circuitbox>\linewidth\resizebox{\linewidth}{!}{\usebox{\circuitbox}}\else\usebox{\circuitbox}\fi}
\newcommand{\fitpanel}[2]{\sbox{\panelbox}{\begin{minipage}{\linewidth}\centering #2\end{minipage}}%
\ifdim\dimexpr\ht\panelbox+\dp\panelbox\relax>#1\resizebox*{!}{#1}{\usebox{\panelbox}}\else\usebox{\panelbox}\fi}
\newcommand{\fitcontent}[2]{\sbox{\contentbox}{\begin{minipage}{\linewidth}#2\end{minipage}}%
\ifdim\dimexpr\ht\contentbox+\dp\contentbox\relax>#1\resizebox*{!}{#1}{\usebox{\contentbox}}\else\usebox{\contentbox}\fi}
\newcommand{\pagehead}[3]{\noindent{\color{accent}\small SQUANDER / SYMBOLIC CIRCUIT ATLAS}\hfill{\color{muted}\small #3}\par
{\color{ink}\fontsize{24}{28}\selectfont\bfseries #1}\par
{\color{muted}\small #2}\par\vspace{3mm}\hrule height .35pt\vspace{3mm}}
""" + (r"\AtBeginDocument{\renewcommand{\small}{\fontsize{14}{17}\selectfont}\renewcommand{\scriptsize}{\fontsize{11}{13}\selectfont}\renewcommand{\tiny}{\fontsize{8}{10}\selectfont}}" if paper == "gates" else "") + r"""
\AtBeginDocument{\color{ink}}
\begin{document}
"""


def panel(body, height):
    return (r"\begin{minipage}[t][" + height + r"][t]{0.243\textwidth}\vspace{0pt}\fitcontent{\dimexpr"+
            height+r"-2mm\relax}{"+body+r"}\end{minipage}")


def grid(panels, columns=4):
    lines = []
    for start in range(0, len(panels), columns):
        lines.append(r"\noindent " + r"\hfill ".join(panels[start:start+columns]) + r"\par\vspace{1mm}")
    return "\n".join(lines)


def drawing(store, width, result, max_height="33mm"):
    blocks = []
    program=result.get('expression_program')
    paths=store.render(width,result['operations'],expression_program=program) if program else store.render(width,result['operations'])
    for index, path in enumerate(paths):
        if index:
            blocks.append(r"\par\vspace{1pt}{\tiny\color{muted}$\longrightarrow$ continued in time order}\par")
        blocks.append(r"\circuit{\input{" + path + "}}")
    used=sorted({store.aliases[p] for op in result["operations"] for p in op["parameters"] if p in store.aliases})
    if used:
        blocks.append(r"\par{\scriptsize\color{muted}Exact definitions: $"+", ".join(used)+r"$ in the expression glossary.}")
    if program:
        index=store.register_program(program)
        blocks.append(r'\par{\scriptsize\color{muted}Derived angles: \hyperref[expression-program-'+str(index)+
                      r']{exact shared-expression program '+str(index)+r'}.}')
    return r"\fitpanel{" + max_height + "}{" + "\n".join(blocks) + "}"


def cost_line(result, independent_proof=None):
    cost = result["cost"]
    try:
        score = f"{float(sp.sympify(cost['parameter_score'])):.3f}"
    except (TypeError, ValueError):
        score = "symbolic"
    certificate = ("proved E minimum" if result.get("entangler_optimal") else
                   'minimum uniform E budget' if result.get('uniform_entangler_budget_certificate') else
                   "conditional E minimum" if result.get("conditional_entangler_optimality") else "E upper bound")
    if independent_proof:
        if independent_proof['scope'] == 'fixed_target_minimum':
            certificate = 'proved E minimum'
        elif not result.get('entangler_optimal'):
            certificate = 'minimum uniform E budget'
    score_text='score unresolved' if cost.get('parameter_score') is None else f'score $\\approx$ {score}'
    return (f"E={cost['entanglers']} / L={cost['single_qubit_gates']} / {score_text}"
            + r"\hfill " + escape(certificate))


def status_text(result):
    if not result:
        return "No archived result"
    status = result["status"]
    if status == "proved_impossible":
        obstruction = result.get("obstruction", "proved obstruction").replace("_", " ")
        return "Proved obstruction: " + obstruction
    return {"timeout": "Open: time limit reached", "unresolved": "Open: construction not found",
            "compiler_error": "Open: compiler error", "identity_not_proved": "Open: identity not proved"}.get(status, status)


def gate_panel(store, basis, row, width):
    header = r"{\small\bfseries\ttfamily " + escape(basis) + r"}\par{\color{rule}\rule{\linewidth}{.2pt}}\par"
    variants = phase_variants(row)
    if not variants:
        text = []
        for mode, label in (("phase_exact", "Exact"), ("projective", "Projective")):
            result = row.get(mode) if row else None
            if result is None and row and row.get("projective_omitted_as_no_better"):
                continue
            text.append(r"\textbf{" + label + "}: " + escape(status_text(result)))
        return panel(header + r"\vspace{4mm}{\small\color{muted}" + r"\par ".join(text) + "}", "62mm")
    bodies = []
    for mode, result in variants:
        color, label = ("exact", "EXACT PHASE") if mode == "phase_exact" else ("projective", "PROJECTIVE")
        independent = row.get('independent_entangler_certificates',{}).get(mode)
        bodies.append(r"{\scriptsize\color{" + color + r"}\bfseries " + label + r"}\quad{\scriptsize " + cost_line(result, independent) + r"}\par")
        bodies.append(drawing(store, width, result, "18mm" if len(variants) == 2 else "39mm") + r"\par\vspace{.8mm}")
        if mode == "projective":
            factor = result.get("virtual_gp", {}).get("factor", "1")
            program=result.get('expression_program')
            displayed=store.program_expression(factor,program) if program else store.expression(factor)
            bodies.append(r"\par{\scriptsize\color{muted}$U=" + displayed + r"\,C$. Scalar metadata; no physical GP.}\par")
    if row["phase_exact"]["status"] != "verified_symbolic":
        bodies.append(r"\par{\scriptsize\color{muted}Exact: " + escape(status_text(row["phase_exact"])) + "}")
    literature=row.get("literature_uniform_upper_bound_comparison",{})
    gaps=[result for result in literature.get("modes",{}).values()
          if result["status"]=="exceeds_published_uniform_bound"]
    if gaps:
        upper=min(result["uniform_upper_bound"] for result in gaps)
        bodies.append(r"\par{\scriptsize\color{projective}Published uniform cap: $E\leq "+str(upper)+
                      r"$. Current construction exceeds it; optimization remains.}")
    return panel(header + "\n".join(bodies), "62mm")


def gate_volume(report, output, store, names, archive_digest):
    source = [preamble("Symbolic gate atlas: 327 targets / 32 bases")]
    source.append(r"\pagehead{Symbolic gate atlas}{Exact identities. Explicit phase policies. Native entangler counts.}{Reference edition}")
    source.append(r"{\Large " + str(len(names)) + r" targets $\times$ 32 basis panels}\par\vspace{5mm}")
    source.append(r"\begin{minipage}{.8\linewidth}\large Each target has one page with 32 basis panels in alphabetical order. "
                  r"Blue labels denote literal matrix equality. Brown labels denote a projective circuit with its exact scalar factor. "
                  r"Both appear when the projective construction is cheaper or their secondary-score comparison remains unresolved. Physical GP operations are forbidden.\par "
                  r"E counts native two-qubit primitives, L counts one-qubit operations. Evaluated scores are displayed rounded; the archive stores exact expressions. Shared-expression scores that exceeded their evaluation budget are labelled unresolved. "
                  r"Objective version 2 clamps negative denominator penalties; an exact compatibility audit preserves the earlier retained scores. "
                  r"Only an explicit matching certificate establishes a minimum entangler count. A verified upper bound does not establish minimum local cost. "
                  r"A minimum uniform E budget concerns a circuit for the whole parameterized family: a certified hard member forces that budget, while special parameter values can require fewer pulses. "
                  r"Published-cap notes identify constructions above known universal two-qubit bounds; these remain optimization gaps. "
                  r"Proof conditions, source hashes and attempt histories remain in the archive.\par "
                  r"Open cases are shown as open. Proved obstructions preserve the supplied basis and may be conditional for parameterized families. "
                  r"Register constructors use the recorded integer-width or matrix fixtures. These pages do not claim a solution for every register size.\par "
                  r"Circuits are drawn by Qiskit's LaTeX drawer. Long circuits continue on successive rows in time order. "
                  r"An explicit $[G]^{\times n}$ box denotes n consecutive identical primitive operations; their full physical count remains in E.\par "
                  r"Long exact expressions use numbered $\eta_j$ labels defined in the expression glossary. No parameter is rounded or dropped.\par "
                  r"The separate Clifford volume lists every exact phase in each distinct one-qubit alphabet, with 24 circuits per page. "
                  r"Identical alphabets shared by multiple presets are printed once.\par\vspace{5mm}" +
                  r"\texttt{Archive SHA256: " + archive_digest + r"}\end{minipage}\newpage")
    source.append(r"\pagehead{Alphabetical target index}{Links lead to the 32-basis page for each target.}{" + str(len(names)) + " targets}")
    source.append(r"\begin{multicols}{4}\large")
    for name in names:
        source.append(r"\hyperref[gate-" + slug(name) + "]{" + escape(name) + r"}\dotfill\pageref{gate-" + slug(name) + r"}\par")
    source.append(r"\end{multicols}\newpage")
    source.append(r"\pagehead{Native basis key}{Every panel preserves its supplied alphabet. Native units and macro counts matter.}{32 presets}")
    source.append(r"\large\renewcommand{\arraystretch}{1.5}\begin{tabular}{@{}p{.19\textwidth}p{.13\textwidth}p{.23\textwidth}p{.4\textwidth}@{}}"
                  r"\textbf{Preset}&\textbf{Entangler}&\textbf{Local alphabet}&\textbf{Scope / units}\\\hline")
    for name, basis in sorted(BASIS_SETS.items()):
        source.append(escape(name) + " & " + escape(basis.entangler) + " & " +
                      escape(", ".join(basis.single_qubit_gates)) + " & " +
                      escape(basis.notes or basis.universality.replace("_", " ")) + r"\\")
    source.append(r"\end{tabular}\newpage")
    cases = {name: parameters for name, parameters, _ in symbolic_cases()}
    import symbolic_gates as gates
    manifest = []
    for index, name in enumerate(names):
        parameters = cases[name]
        definition = gates.gate_definition(name)
        width = definition.num_qubits or (gates.make_gate(name, *parameters).rows.bit_length() - 1)
        label = escape(name) + (r" $\left(" + ",".join(sp.latex(p) for p in parameters) + r"\right)$" if definition.num_qubits is not None and definition.parameters else "")
        source.append(r"\phantomsection\label{gate-" + slug(name) + r"}\pagehead{" + label + "}{" + str(width) + r" qubits / 32 preserved native bases / alphabetical panel order}{Target " + str(index+1) + " / " + str(len(names)) + "}")
        source.append(r"\pdfbookmark[0]{" + escape(name) + "}{bookmark-gate-" + slug(name) + "}")
        panels = []
        for basis, row in gate_grid(report, name):
            panels.append(gate_panel(store, basis, row, width))
            manifest.append(dict(gate=name, basis=basis, phase_statuses=row["phase_statuses"] if row else {},
                                 modes=[mode for mode, _ in phase_variants(row)],
                                 mode_source_run_indices=row.get("mode_source_run_indices",{}) if row else {},
                                 independent_entangler_certificates=row.get('independent_entangler_certificates',{}) if row else {},
                                 literature_uniform_upper_bound_comparison=row.get("literature_uniform_upper_bound_comparison") if row else None,
                                 synthesis_run_index=row.get("synthesis_run_index", 0) if row else None))
        source.append(grid(panels))
        source.append(r"\newpage")
        if (index + 1) % 25 == 0:
            print(f"Rendered {index+1}/{len(names)} target pages", flush=True)
    expressions=list(store.aliases.items())
    for start in range(0,len(expressions),18):
        source.append(r"\pagehead{Exact expression glossary}{Definitions shared by the numbered circuit parameters and scalar factors.}{"+str(start+1)+"--"+str(min(start+18,len(expressions)))+"}")
        source.append(r"\fitcontent{\dimexpr\textheight-36mm\relax}{")
        for value,symbol in expressions[start:start+18]:
            source.append(r"\par\vspace{3mm}\circuit{$"+symbol+"="+parameter_latex(value)+r"$}\par\vspace{3mm}")
        source.append("}")
        source.append(r"\newpage")
    for index,program in store.programs.values():
        bindings=program['bindings']
        for start in range(0,len(bindings),36):
            if start==0:
                source.append(r'\phantomsection\label{expression-program-'+str(index)+'}')
            source.append(r'\pagehead{Exact shared-expression program '+str(index)+
                          r'}{Original parameters: '+escape(', '.join(program['inputs']))+
                          r'. Bindings are definitions, not additional free angles.}{Bindings '+
                          str(start)+'--'+str(min(start+36,len(bindings))-1)+'}')
            source.append(r'\small Evaluate definitions in order. Piecewise expressions select the first satisfied case; unused singular branches are not evaluated. The source bundle also includes the machine-readable program.\par')
            source.append(r'\fitcontent{\dimexpr\textheight-53mm\relax}{\begin{multicols}{2}')
            for j,row in enumerate(bindings[start:start+36],start):
                formula=store.program_expression(row['expression'],program,compact=False)
                source.append(r'\par\vspace{2mm}\circuit{$\xi_{'+str(index)+','+str(j)+'}='+formula+r'$}\par')
            source.append(r'\end{multicols}}\newpage')
        outputs=list(store.program_outputs[index].items())
        for start in range(0,len(outputs),18):
            source.append(r'\pagehead{Derived parameters for program '+str(index)+
                          r'}{Exact output expressions used by the native gate labels.}{Output definitions}')
            source.append(r'\fitcontent{\dimexpr\textheight-36mm\relax}{')
            for value,symbol in outputs[start:start+18]:
                source.append(r'\par\vspace{3mm}\circuit{$'+symbol+'='+
                              store.program_expression(value,program,compact=False)+r'$}\par')
            source.append(r'}\newpage')
    source.append(r"\end{document}")
    (output / "gates.tex").write_text("\n".join(source))
    return manifest


def hydrate_certificate(data):
    def decode(operations):
        return tuple((op["gate"], *(sp.sympify(p) for p in op["parameters"]), tuple(op["wires"])) for op in operations)
    solution = dict(local_gates=tuple(data["local_gates"]), proof_scope=data["proof_scope"],
                    global_local_optimality=data["global_local_optimality"],
                    edges=[(edge["target"], edge["cost_ticks"], decode(edge["operations"])) for edge in data["edges"]],
                    distances={int(key): value for key, value in data["distances"].items()},
                    predecessors={int(key): tuple(value) for key, value in data["predecessors"].items()})
    if "exhaustive_word_certificate" in data:
        solution["exhaustive_word_certificate"]=data["exhaustive_word_certificate"]
    verify_shortest_path_certificate(solution)
    return solution


def clifford_phase_obstruction(state,basis):
    from symbolic_feasibility import determinant_group
    order=determinant_group(basis,1)
    if order is None:return None
    determinant=sp.simplify(group_data()[1][state].det())
    residual=sp.simplify(determinant**order-1)
    if residual.is_zero is False:
        return dict(method="one_wire_determinant_group",determinant=str(determinant),
                    determinant_root_order=order,nonzero_witness=str(residual))
    return None


def clifford_volume(report, output, store, selected=()):
    from symbolic_contract import encode_sequence
    data = group_data()
    aliases = defaultdict(list)
    for name, basis in BASIS_SETS.items():
        aliases["|".join(basis.single_qubit_gates)].append(name)
    alphabets = sorted(report["clifford_certificates"])
    if selected:
        if set(selected) - set(alphabets):
            raise ValueError("Unknown Clifford alphabet")
        alphabets = [key for key in alphabets if key in selected]
    source = [preamble("All-phase one-qubit Clifford atlas", "clifford")]
    source.append(r"\pagehead{All-phase Clifford atlas}{24 projective classes / eight exact scalar phases / each distinct local alphabet}{Reference edition}")
    source.append(r"\begin{minipage}{.9\linewidth}\large Each alphabet occupies eight pages. A page fixes $k$ and lists all 24 operators "
                  r"$e^{ik\pi/4}C_j$. The representative matrices $C_j$ are the exact named representatives in the catalog. "
                  r"Every displayed circuit equals its target literally; no scalar-phase repair or physical GP is included. "
                  r"The names identify actual catalog matrices, including the legacy distinction between SX and square-root X (Sx).\par "
                  r"Shortest-path costs use the audited parameter-sensitive metric with nonnegative denominator penalties. A fixed Clifford alphabet such as H/S exhausts its allowed words. "
                  r"H/S/T/Tdg also has an exhaustive exact-word proof, including non-Clifford intermediate paths. "
                  r"For continuous alphabets the certificate covers the listed specializations and macro words. Missing paths retain that scope.\par "
                  r"All presets with the same one-qubit alphabet share the same table. Entanglers and ancillary qubits are unavailable in this one-wire reference."
                  r"\end{minipage}\vspace{5mm}\par")
    source.append(r"\hyperref[clifford-representatives]{Exact representative matrices}\hfill p.~\pageref{clifford-representatives}\par")
    for key in alphabets:
        source.append(r"\hyperref[clifford-" + slug(key) + "]{" + escape(key) + r"}\hfill p.~\pageref{clifford-" + slug(key) + r"}\par")
    source.append(r"\newpage")
    source.append(r"\phantomsection\label{clifford-representatives}\pagehead{The 24 matrix representatives}{These exact matrices fix the global-phase convention for every table.}{Representative key}")
    matrix_panels=[]
    for index,name in enumerate(data[0].representatives):
        matrix=data[1][data[3][name]]
        formula=sp.latex(matrix,mat_str="pmatrix",mat_delim=None)
        matrix_panels.append(panel(r"{\small\bfseries $C_{"+str(index)+"}$ / "+escape(name)+r"}\par\vspace{3mm}\circuit{$"+formula+"$}","36mm"))
    source.append(grid(matrix_panels))
    source.append(r"\newpage")
    manifest = []
    for key in alphabets:
        certificate = report["clifford_certificates"][key]
        solution = hydrate_certificate(certificate)
        for phase in range(8):
            if phase == 0:
                source.append(r"\phantomsection\label{clifford-" + slug(key) + "}")
                source.append(r"\pdfbookmark[0]{" + escape(key) + "}{bookmark-clifford-" + slug(key) + "}")
            source.append(r"\pdfbookmark[1]{Phase " + str(phase) + " / 8}{bookmark-clifford-" + slug(key) + "-" + str(phase) + "}")
            presets = ", ".join(sorted(aliases[key]))
            source.append(r"\pagehead{" + escape(key) + r" / $k=" + str(phase) + r"$}{" + escape(presets) + r"}{$e^{ik\pi/4}C_j$ / 24 exact targets}")
            panels = []
            for representative, name in enumerate(data[0].representatives):
                actual = (data[3][name] % 8 + phase) % 8
                state = 8 * representative + actual
                title = r"{\small\bfseries $e^{i" + str(phase) + r"\pi/4}$ " + escape(name) + "}"
                obstruction=None
                if state in solution["distances"]:
                    sequence = clifford_path(solution, state)
                    result = dict(operations=encode_sequence(sequence))
                    body = title + r"\par{\scriptsize Score $" + sp.latex(sp.Rational(solution["distances"][state],500)) + r"$ / " + str(len(sequence)) + r" gates}\par"
                    body += drawing(store, 1, result, "21mm")
                else:
                    obstruction=clifford_phase_obstruction(state,BASIS_SETS[aliases[key][0]])
                    text="Impossible: phase outside the local determinant group." if obstruction else "No exact-phase path in the certified alphabet."
                    body = title + r"\par\vspace{4mm}{\small\color{muted}"+text+"}"
                panels.append(panel(body, "36mm"))
                manifest.append(dict(alphabet=key, phase=phase, representative=name, state=state,
                                     reachable=state in solution["distances"], proof_scope=solution["proof_scope"],
                                     obstruction=obstruction))
            source.append(grid(panels))
            source.append(r"{\scriptsize\color{muted}Certificate scope: " + escape(solution["proof_scope"]) + r". Phase is part of the target matrix.}")
            source.append(r"\newpage")
    source.append(r"\end{document}")
    (output / "cliffords.tex").write_text("\n".join(source))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", action="append", default=[])
    parser.add_argument("--clifford-alphabet", action="append", default=[])
    parser.add_argument("--fold", type=int, default=18)
    parser.add_argument("--long-fold", type=int, default=54,
                        help="Operations per row for long circuits on four or more wires")
    args = parser.parse_args()
    if args.fold < 1 or args.long_fold < 1:
        parser.error("Positive circuit fold length required")
    with gzip.open(args.archive, "rt") as stream:
        report = json.load(stream)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = CircuitStore(args.output_dir, args.fold, args.long_fold)
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    names = ordered_targets(report, args.gate)
    gates = gate_volume(report, args.output_dir, store, names, digest)
    cliffords = clifford_volume(report, args.output_dir, store, args.clifford_alphabet)
    manifest = dict(archive=str(args.archive), archive_sha256=digest, renderer="Qiskit latex_source",
                    gate_panels=gates, clifford_panels=cliffords, target_count=len(names),
                    basis_count=32, circuits_per_gate_page=32, circuits_per_clifford_page=24,
                    expression_aliases={symbol:value for value,symbol in store.aliases.items()},
                    expression_programs=[dict(index=index,bindings=len(program['bindings']),
                                              path=f'programs/program-{index}.json',
                                              derived_output_definitions=store.program_outputs[index])
                                         for index,program in store.programs.values()],
                    note="Exact symbolic proof statuses and certificate scopes are preserved; this atlas does not assert every target is solved.")
    with gzip.open(args.output_dir / "manifest.json.gz", "wt") as stream:
        json.dump(manifest, stream, separators=(",", ":"))
    print(json.dumps(dict(target_pages=len(names), gate_panels=len(gates),
                          clifford_pages=len(cliffords)//24, unique_circuits=len(store.paths)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
