import itertools
import pickle

import pytest
import sympy as sp

from symbolic_archive import symbolic_cases
from symbolic_catalog_motifs import catalog_motif, _uniform_rotations
from symbolic_gates import make_gate
from symbolic_identity import prove_circuit_identity
from symbolic_synthesis import synthesize_matrix, synthesize


MOTIF_NAMES = {"XX90", "YY90", "ZZ90", "ZZMax", "IonQZZ", "ZZPhase", "IonQMS", "IonQMSMax", "TK2", "SYC",
               "CSXdg", "fFredkin", "Deutsch", "CCP", "CCZPow", "CCXPow", "FSWAPPow", "PhasedISWAPPow",
               "CZS", "CCZS", "Controlled", "MCMT", "UCRX", "UCRY", "UCRZ", "UC", "Diagonal", "Permutation", "LinearFunction"}


@pytest.mark.parametrize("name,parameters", [(name, parameters) for name, parameters, _ in symbolic_cases()
                                            if name in MOTIF_NAMES])
def test_catalog_motif_matches_its_independent_exact_definition(name, parameters):
    sequence, factor = catalog_motif(name, parameters)
    assert not any(op[0] == "GP" for op in sequence)
    assert not any(sp.sympify(p).atoms(sp.Float) for op in sequence for p in op[1:-1])
    assert prove_circuit_identity(make_gate(name, *parameters), sequence, factor)


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_gray_code_motif_keeps_four_independent_angles(axis):
    angles = sp.symbols("a:4", real=True)
    sequence = _uniform_rotations(axis, angles)
    assert prove_circuit_identity(make_gate("UCR"+axis, angles), sequence)


@pytest.mark.parametrize("pattern", list(itertools.permutations(range(3))))
def test_generated_wire_permutation_is_in_the_correct_direction(pattern):
    sequence, factor = catalog_motif("Permutation", (pattern,))
    assert prove_circuit_identity(make_gate("Permutation", pattern), sequence, factor)


def test_binary_gaussian_motif_uses_row_operations_in_reverse_order():
    for matrix in (((1,1),(0,1)), ((0,1),(1,1)), ((1,1,0),(0,1,1),(0,0,1))):
        sequence, factor = catalog_motif("LinearFunction", (matrix,))
        assert prove_circuit_identity(make_gate("LinearFunction", matrix), sequence, factor)


def test_nonclifford_native_basis_can_rebase_an_opaque_clifford_skeleton():
    result = synthesize_matrix(make_gate("Magic"), "zz_u", phase_exact=False, seconds=.1)
    assert result["status"] == "verified_symbolic"
    assert result["cost"]["entanglers"] == 1
    assert result["construction_method"] == "logical_CNOT_Clifford_skeleton_with_native_rebasing"


def test_exchange_echo_avoids_independent_fredkin_and_ccz_synthesis():
    result = synthesize("fFredkin", (), "cx_u", phase_exact=False)
    assert result["status"] == "verified_symbolic"
    assert result["cost"]["entanglers"] == 5
    assert result["source_motif"]["schema"] == "excitation_exchange_echo"


@pytest.mark.parametrize("basis,count",[("iswap_u",6),("sqrt_iswap_u",4)])
def test_nested_exchange_motifs_lower_directly_to_native_pulses(basis,count):
    result=synthesize("fFredkin",(),basis,phase_exact=False)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==count


def test_controlled_hadamards_use_one_interaction_per_target():
    result = synthesize("MCMT", (make_gate("H"), 1, 2), "zz_u", phase_exact=False)
    assert result["status"] == "verified_symbolic"
    assert result["cost"]["entanglers"] == 2
    assert result["entangler_optimal"]


def test_controlled_matrix_motif_survives_process_transport_normalization():
    parameters=pickle.loads(pickle.dumps((make_gate("Sx"),2)))
    result=synthesize("Controlled",parameters,"cx_u",phase_exact=False)
    assert result["status"]=="verified_symbolic"
    assert result["cost"]["entanglers"]==6


def test_symbolic_exchange_frontier_finishes_within_its_archive_budget():
    from symbolic_archive import bounded_synthesis
    parameters=sp.symbols("theta phi gamma",real=True)
    result=bounded_synthesis("CZS",parameters,"ibm_ecr",phase_exact=False,seconds=45)
    assert result["status"]=="verified_symbolic"
