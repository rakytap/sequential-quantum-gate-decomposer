import copy

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_archive import symbolic_cases
from symbolic_bases import get_basis
from symbolic_b_cartan import synthesize_b_cartan, verify_b_cartan_certificate, verify_b_cartan_lemma


def _evaluate(result, substitutions):
    actual = np.eye(4, dtype=complex)
    symbols = {str(symbol): symbol for symbol in substitutions}
    for operation in result['operations']:
        parameters = [sp.sympify(p, locals=symbols).subs(substitutions).evalf(22)
                      for p in operation['parameters']]
        primitive = gates.make_gate(operation['gate'], *parameters)
        embedded = gates.apply_to(sp.eye(4), 2, primitive, tuple(operation['wires']))
        actual = np.asarray(embedded.evalf(22), complex) @ actual
    if not result['phase_exact']:
        phase = sp.sympify(result['virtual_gp']['factor'], locals=symbols).subs(substitutions)
        actual *= complex(phase.evalf(22))
    return actual


@pytest.mark.parametrize('exact', [False, True])
def test_full_b_chart_exact_lemma_replay_and_fold_boundaries(exact):
    x, y, z = sp.symbols('x y z', real=True)
    target = gates.make_gate('Canonical', x, y, z)
    basis = get_basis('berkeley_b')
    result = synthesize_b_cartan(target, basis, phase_exact=exact)
    assert verify_b_cartan_lemma()
    assert result['cost']['entanglers'] == 2
    assert result['cost']['single_qubit_gates'] == (7 if exact else 6)
    assert verify_b_cartan_certificate(target, result, basis)
    for values in ((0, 0, 0), (sp.pi/4, sp.pi/4, sp.pi/4),
                   (sp.pi/3, -sp.pi/4, sp.pi/4), (sp.pi, sp.pi/2, -sp.pi),
                   (sp.Rational(-17, 5), sp.Rational(13, 7), sp.Rational(-19, 4))):
        substitutions = dict(zip((x, y, z), values))
        expected = np.asarray(target.subs(substitutions).evalf(22), complex)
        assert np.max(np.abs(_evaluate(result, substitutions)-expected)) < 5e-14
    for field in ('operations', 'cost', 'b_cartan_certificate', 'entangler_optimal', 'local_cost_optimal'):
        changed = copy.deepcopy(result)
        if field == 'operations': changed[field][0]['parameters'][0] = '0'
        elif field == 'cost': changed[field]['entanglers'] = 1
        elif field == 'b_cartan_certificate': changed[field]['folds'] = ['0', '0']
        else: changed[field] = not changed[field]
        assert not verify_b_cartan_certificate(target, changed, basis)


@pytest.mark.parametrize('name', ['Canonical', 'fSim', 'PhasedFSim', 'SWAPPow', 'TK2'])
def test_all_five_free_families_and_diagonal_gauges(name):
    parameters = next(parameters for gate, parameters, _ in symbolic_cases() if gate == name)
    target = gates.make_gate(name, *parameters)
    result = synthesize_b_cartan(target, get_basis('berkeley_b'), phase_exact=False)
    assert result['cost']['entanglers'] == 2
    substitutions = {p: sp.Rational(i+1, 7) for i, p in enumerate(parameters)}
    assert np.max(np.abs(_evaluate(result, substitutions)-np.asarray(target.subs(substitutions).evalf(22), complex))) < 5e-14


def test_b_chart_declines_other_native_entanglers():
    assert synthesize_b_cartan(gates.make_gate('CX'), get_basis('cx_u'), phase_exact=True) is None
