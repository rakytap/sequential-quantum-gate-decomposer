import importlib.util
from pathlib import Path

import pytest
import sympy as sp

from symbolic_bases import BASIS_SETS

path = Path(__file__).resolve().parents[2] / "examples/decomposition/symbolic_circuit_atlas.py"
spec = importlib.util.spec_from_file_location("symbolic_circuit_atlas", path)
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)


def test_gate_page_keeps_all_32_bases_and_alphabetical_target_order():
    report = dict(rows=[dict(gate=name, basis=basis) for name, basis in
                        [("SWAP", "cx_u"), ("CP", "cx_u"), ("CP", "zz_u")]])
    assert atlas.ordered_targets(report) == ["CP", "SWAP"]
    grid = atlas.gate_grid(report, "CP")
    assert len(grid) == 32
    assert [basis for basis, _ in grid] == sorted(BASIS_SETS)
    assert sum(row is not None for _, row in grid) == 2
    with pytest.raises(ValueError):
        atlas.ordered_targets(report, ["invented"])


def test_open_and_impossible_cases_never_acquire_a_drawn_circuit():
    row = dict(phase_exact=dict(status="proved_impossible"),
               projective=dict(status="timeout"))
    assert atlas.phase_variants(row) == []
    row["projective"] = dict(status="verified_symbolic", virtual_gp=dict(factor="I"))
    assert [mode for mode, _ in atlas.phase_variants(row)] == ["projective"]


def test_repeated_pulse_display_preserves_every_physical_operation():
    operation = dict(gate="SYC", parameters=[], wires=[0, 1])
    local = dict(gate="Rz", parameters=["theta"], wires=[0])
    original = [operation] * 6 + [local] + [operation] * 2
    compressed = atlas.compressed_operations(original)
    assert [repeat for _, repeat in compressed] == [6, 1, 2]
    assert [op for op, repeat in compressed for _ in range(repeat)] == original
    assert r"\times 6" in atlas.operation_label(operation, 6)


def test_parameter_names_do_not_turn_into_sympy_special_functions():
    assert atlas.parameter_latex("gamma/2") == r"\frac{\gamma}{2}"
    assert atlas.parameter_latex("beta + pi/8") == r"\beta + \frac{\pi}{8}"


def test_qiskit_draws_exact_symbolic_math_and_operand_order():
    pytest.importorskip("qiskit")
    operations = [(dict(gate="Rzz", parameters=["gamma/2"], wires=[1, 0]), 1)]
    source = atlas.qiskit_latex(2, operations)
    assert r"R_{ZZ}" in source and r"\frac{\gamma}{2}" in source
    assert r"\string^" not in source and r"\mathrm{$" not in source
    assert "<<<{1}" in source and "<<<{0}" in source
    with pytest.raises(ValueError, match="GP"):
        atlas.qiskit_latex(1, [(dict(gate="GP", parameters=["theta"], wires=[0]), 1)])


def test_long_exact_parameters_keep_a_glossary_definition(tmp_path):
    store=atlas.CircuitStore(tmp_path)
    expression="atan((sqrt(2-sqrt(2))-sqrt(2+sqrt(2)))/(-sqrt(2+sqrt(2))-sqrt(2-sqrt(2))))"
    symbol=store.expression(expression)
    assert symbol==r"\eta_{1}"
    assert store.aliases[expression]==symbol
    operation=dict(gate="U3",parameters=["pi",expression,expression],wires=[0])
    paths=store.render(1,[operation])
    source=(tmp_path/paths[0]).read_text()
    assert r"\eta_{1}" in source
    assert r"\operatorname{atan}" not in source


def test_independent_pages_keep_fonts_and_bound_the_entire_panel():
    source=atlas.preamble("Gate atlas")
    preamble=source.split(r"\begin{document}")[0]
    assert r"\AtBeginDocument{\renewcommand{\small}{\fontsize{14}{17}" in preamble
    body=atlas.panel("circuit and status", "62mm")
    assert r"\fitcontent{\dimexpr62mm-2mm\relax}{circuit and status}" in body


def test_missing_clifford_phases_have_a_basis_preserving_obstruction():
    from symbolic_clifford_search import group_data
    state=group_data()[3]["I_q1"]
    proof=atlas.clifford_phase_obstruction(state,BASIS_SETS["cx_rz_ry"])
    assert proof["method"]=="one_wire_determinant_group"
    assert proof["determinant_root_order"]==1
    assert atlas.clifford_phase_obstruction(state,BASIS_SETS["clifford_only"]) is None
