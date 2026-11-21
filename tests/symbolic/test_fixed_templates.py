import sympy as sp

from symbolic_bases import get_basis
from symbolic_fixed_templates import fixed_template_candidates
from symbolic_gates import make_gate
from symbolic_identity import prove_circuit_identity
from symbolic_synthesis import synthesize


def test_fixed_templates_are_basis_specific_and_do_not_specialize_free_families():
    basis = get_basis("zz_u")
    assert fixed_template_candidates("B", (), basis)
    assert fixed_template_candidates("B", (sp.Symbol("theta"),), basis) == []
    assert fixed_template_candidates("B", (), get_basis("clifford_only")) == []
    for sequence, factor in fixed_template_candidates("B", (), basis):
        assert prove_circuit_identity(make_gate("B"), sequence, factor)


def test_normal_symbolic_compiler_uses_the_proved_local_improvement():
    for exact in (False, True):
        result = synthesize("B", (), "zz_u", phase_exact=exact, osr_seconds=5)
        assert result["status"] == "verified_symbolic"
        assert result["cost"]["entanglers"] == 2
        assert result["cost"]["single_qubit_gates"] == (8 if exact else 6)
        assert not result["local_cost_optimal"]
