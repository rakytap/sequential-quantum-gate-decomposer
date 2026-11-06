from collections import Counter
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"examples/decomposition"))
spec = importlib.util.spec_from_file_location("notable_results", ROOT/"examples/decomposition/symbolic_notable_results.py")
notable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notable)


def test_local_audit_keeps_regressions_and_distinguishes_entangler_priority():
    rows, sdk = [], []
    for gate, entanglers, locals_ in (("equal", 2, 3), ("saving", 2, 1), ("regression", 2, 5), ("more_E", 3, 0)):
        result = dict(status="verified_symbolic", cost=dict(entanglers=entanglers,
                       single_qubit_gates=locals_, parameter_score="23/3"))
        rows.append(dict(gate=gate, basis="cx_u", target_scope="fixed_gate",
                          phase_exact=result, projective=None, projective_omitted_as_no_better=True))
        sdk.append(dict(gate=gate, basis="cx_u", status="verified_numeric", entanglers=2,
                        input="stored_exact_reference", operations=[dict(wires=[0])]*3))
    compared, counts = notable.local_audit(dict(rows=rows), dict(rows=sdk))
    assert counts == dict(equal_E_equal_locals=1, equal_E_fewer_locals=1,
                         equal_E_more_locals=1, more_entanglers=1)
    assert all(row["sdk_parameter_score"] is None for row in compared)
    assert all(row["sdk_input"] == "stored_exact_reference" for row in compared)
    assert all(not row["novelty_claim"] for row in compared)


def test_sycamore_local_proof_has_the_restricted_phase_polynomial_scope():
    proof = notable.phase_polynomial_certificate()
    assert proof["local_gate_lower_bound"] == proof["achieved_local_gates"] == 3
    assert not proof["arbitrary_U2_local_minimum_proved"]
    assert proof["coefficient_angles"] == ["-pi/12", "-pi/12", "-5*pi/12"]
    counts = Counter(gate for row in notable.PAPER["local_gate_types_by_wire"] for gate in row)
    assert counts == dict(Rz=15, Sx=12)
