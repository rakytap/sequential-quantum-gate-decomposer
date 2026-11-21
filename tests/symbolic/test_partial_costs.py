import sympy as sp
from symbolic_contract import compare_recorded_costs, retain_phase_results


def result(e, score, locals_, exact):
    return dict(status='verified_symbolic', phase_exact=exact,
                cost=dict(entanglers=e, parameter_score=score, single_qubit_gates=locals_))


def test_unknown_score_never_becomes_zero_tie_or_tertiary_comparison():
    a=result(3,None,9,True);b=result(3,'100',8,False)
    assert compare_recorded_costs(a['cost'],b['cost']) is None
    kept=retain_phase_results(a,b)
    assert kept['projective'] is b and kept['preferred']=='phase_exact'
    assert not kept['projective_omitted_as_no_better']
    assert kept['phase_cost_comparison']=='secondary_score_unresolved_both_modes_retained'
    assert compare_recorded_costs(a['cost'],result(6,'1',0,False)['cost'])==-1
    assert compare_recorded_costs(result(2,None,100,False)['cost'],a['cost'])==-1


def test_old_complete_cost_order_remains_entanglers_score_then_local_count():
    a=result(3,'20',9,True);b=result(3,'21',8,False)
    assert retain_phase_results(a,b)['projective'] is None
    b['cost']['parameter_score']='19'
    assert retain_phase_results(a,b)['preferred']=='projective'
    assert compare_recorded_costs(a['cost'],a['cost'])==0


def test_public_compiler_returns_the_three_pulse_shared_expression_incumbent():
    from symbolic_archive import wall_time_limit
    from symbolic_synthesis import synthesize
    from symbolic_sqrt_iswap_cartan import verify_sqrt_iswap_cartan_certificate
    from symbolic_gates import make_gate
    from symbolic_bases import get_basis
    a,b,c=sp.symbols('a b c',real=True)
    with wall_time_limit(35):
        value=synthesize('Canonical',(a,b,c),'sqrt_iswap_u',phase_exact=False,osr_seconds=.05)
        assert value['cost']['entanglers']==3
        assert value['cost']['parameter_score'] is None
        assert verify_sqrt_iswap_cartan_certificate(make_gate('Canonical',a,b,c),value,get_basis('sqrt_iswap_u'))
