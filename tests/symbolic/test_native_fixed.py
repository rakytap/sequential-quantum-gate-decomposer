import copy

import pytest
import sympy as sp

import symbolic_gates as gates
from symbolic_bases import get_basis
from symbolic_native_fixed import synthesize_fixed_native, verify_fixed_native_certificate


@pytest.mark.parametrize("name,count,locals",[("B",2,3),("DCX",2,4),("SSWAP",3,7),("SSWAPdg",3,7),("SYC",3,8)])
@pytest.mark.parametrize("basis_name",["sqrt_iswap_u","google_sqrt_iswap"])
@pytest.mark.parametrize("exact",[False,True])
def test_fixed_sqi_swap_motif_has_exact_components_and_replay(name,count,locals,basis_name,exact):
    basis=get_basis(basis_name)
    target=gates.make_gate(name)
    result=synthesize_fixed_native(target,basis,phase_exact=exact)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==count
    assert result["cost"]["single_qubit_gates"]<=locals+int(exact)
    assert verify_fixed_native_certificate(target,result,basis)
    assert not result["local_cost_optimal"]
    assert all(op["gate"]!="GP" for op in result["operations"])
    changed=copy.deepcopy(result)
    changed["operations"][0]["parameters"][0]="0"
    if changed["operations"]==result["operations"]:
        changed["operations"][0]["parameters"][0]="1"
    assert not verify_fixed_native_certificate(target,changed,basis)


def test_fixed_motif_does_not_generalize_a_fixed_point_to_a_free_family():
    theta=sp.Symbol("theta",real=True)
    assert synthesize_fixed_native(gates.make_gate("Canonical",theta,theta,theta),
                                  get_basis("sqrt_iswap_u"),phase_exact=True) is None


def test_matrix_api_keeps_the_proved_fixed_incumbent_when_later_search_expires(monkeypatch):
    import symbolic_synthesis as synthesis
    from symbolic_archive import SymbolicDeadline
    def timeout(*args,**kwargs):raise SymbolicDeadline()
    monkeypatch.setattr(synthesis,"_synthesize_matrix_existing",timeout)
    result=synthesis.synthesize_matrix(gates.make_gate("B"),"sqrt_iswap_u")
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==2
    assert result["cost"]["single_qubit_gates"]==3
