import copy
from functools import lru_cache

import numpy as np
import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_archive import symbolic_cases, wall_time_limit
from symbolic_bases import get_basis
from symbolic_expression_program import ExpressionProgram, evaluate_program
from symbolic_sqrt_iswap_cartan import (
    _canonicalize, _two_pulse, _qmatrix, instantiate_sqrt_iswap_cartan,
    synthesize_sqrt_iswap_cartan, verify_sqrt_iswap_cartan_certificate,
    verify_sqrt_iswap_cartan_lemmas,
)


@lru_cache(maxsize=None)
def construction(name='Canonical', basis_name='sqrt_iswap_u', exact=True):
    parameters = next(p for gate, p, _ in symbolic_cases() if gate == name)
    target = gates.make_gate(name, *parameters)
    with wall_time_limit(35):
        result = synthesize_sqrt_iswap_cartan(target, get_basis(basis_name), phase_exact=exact)
    return target, parameters, result


def matrix(result, parameters, values):
    with wall_time_limit(35):
        operations, phase = instantiate_sqrt_iswap_cartan(result, dict(zip(parameters, values)), digits=65)
    actual = sp.eye(4)
    for operation in operations:
        primitive = gates.make_gate(operation[0], *operation[1:-1])
        actual = gates.apply_to(actual, 2, primitive, operation[-1]).evalf(25)
    return np.asarray((phase*actual).evalf(25), complex)


def test_exact_component_proof_and_full_certificate_tamper_detection():
    target, _, result = construction()
    basis = get_basis('sqrt_iswap_u')
    assert verify_sqrt_iswap_cartan_lemmas()
    assert verify_sqrt_iswap_cartan_certificate(target, result, basis)
    assert result['cost'] == dict(entanglers=3, single_qubit_gates=9, parameter_score=None)
    assert result['local_cost_optimal'] is False
    assert result['expression_program']['inputs'] == sorted(map(str, target.free_symbols))
    for field in ('operations', 'cost', 'expression_program', 'sq_cartan_certificate', 'virtual_gp'):
        changed = copy.deepcopy(result)
        if field == 'operations': changed[field][0]['parameters'][0] = '0'
        elif field == 'cost': changed[field]['parameter_score'] = '0'
        elif field == 'expression_program': changed[field]['bindings'][0]['expression'] = '0'
        elif field == 'sq_cartan_certificate': changed[field]['coordinates'] = ['0', '0', '0']
        else: changed[field] = dict(factor='I', physical=False, cost=0)
        assert not verify_sqrt_iswap_cartan_certificate(target, changed, basis)


@pytest.mark.parametrize('values', [
    (0, 0, 0), (sp.pi/4, 0, 0), (sp.pi/4, sp.pi/4, sp.pi/4),
    (sp.pi/8, sp.pi/8, sp.pi/8), (sp.pi/8, sp.pi/8, 0),
    (sp.pi/4, sp.pi/8, -sp.pi/12), (sp.pi/4, -sp.pi/4, sp.pi/4),
    (sp.Rational(2, 7), sp.Rational(1, 7), sp.Rational(1, 13)),
    (sp.Rational(-17, 5), sp.Rational(13, 7), sp.Rational(-19, 4)),
])
def test_full_phase_exact_matrix_including_repeated_spectrum_and_all_fold_boundaries(values):
    target, parameters, result = construction()
    expected = np.asarray(target.subs(dict(zip(parameters, values))).evalf(25), complex)
    assert np.max(np.abs(matrix(result, parameters, values)-expected)) < 2e-14


@pytest.mark.parametrize('name', ['Canonical', 'fSim', 'PhasedFSim', 'SWAPPow', 'TK2'])
@pytest.mark.parametrize('basis_name', ['sqrt_iswap_u', 'google_sqrt_iswap'])
@pytest.mark.parametrize('exact', [False, True])
def test_twenty_previously_six_pulse_modes_use_three_with_full_matrix_and_phase(name, basis_name, exact):
    target, parameters, result = construction(name, basis_name, exact)
    assert result['cost']['entanglers'] == 3
    assert result['cost']['single_qubit_gates'] == (9 if exact else 8)
    assert result['cost']['parameter_score'] is None  # Never score temporary aliases.
    assert all(op['gate'] != 'GP' for op in result['operations'])
    values = [sp.Rational((i+1)*(-1)**i, 7) for i in range(len(parameters))]
    expected = np.asarray(target.subs(dict(zip(parameters, values))).evalf(25), complex)
    assert np.max(np.abs(matrix(result, parameters, values)-expected)) < 2e-14


def test_exact_constant_axis_frames_reconstruct_original_cartan_coordinates():
    values = (sp.pi/3, -sp.pi/4, -sp.pi/8)
    p = ExpressionProgram()
    canonical, left, right, phase = _canonicalize(p, values)
    expressions = [*canonical, *left[0], *left[1], *right[0], *right[1], phase]
    actual = evaluate_program(p.encode(), (), expressions, {})
    kernel = gates.make_gate('Canonical', *actual[:3])
    L = sp.kronecker_product(_qmatrix(actual[3:7]), _qmatrix(actual[7:11]))
    R = sp.kronecker_product(_qmatrix(actual[11:15]), _qmatrix(actual[15:19]))
    assert all(sp.simplify(value) == 0 for value in sp.exp(sp.I*actual[-1])*L*kernel*R
               -gates.make_gate('Canonical', *values))


def test_cnot_pole_two_pulse_outer_frames_as_exact_algebraic_matrices():
    p = ExpressionProgram()
    with wall_time_limit(35):
        layers = _two_pulse(p, (sp.pi/4, sp.S.Zero, sp.S.Zero))
        values = evaluate_program(p.encode(), (), [value for layer in layers for q in layer for value in q], {})
        actual = sp.eye(4)
        for layer in range(3):
            if layer:
                actual = gates.make_gate('SiSWAP')*actual
            start = 8*layer
            local = sp.kronecker_product(_qmatrix(values[start:start+4]), _qmatrix(values[start+4:start+8]))
            actual = (local*actual).applyfunc(sp.simplify)
        assert all(sp.simplify(value) == 0 for value in actual-gates.make_gate('Canonical', sp.pi/4, 0, 0))


def test_other_bases_and_non_cartan_matrices_are_declined():
    assert synthesize_sqrt_iswap_cartan(gates.make_gate('CX'), get_basis('cx_u'), phase_exact=True) is None
    assert synthesize_sqrt_iswap_cartan(gates.make_gate('H'), get_basis('sqrt_iswap_u'), phase_exact=True) is None
