import pytest
import sympy as sp

pytest.importorskip("deap")
import sympy_param as search


def test_sample_alias_is_not_accepted_as_an_exact_family(capsys):
    # At the five-point proposal grid, Rz(10*theta) is identity at every
    # sample. It is not the identity as a symbolic function of theta.
    angle = search.make_angle(search.ParamIndex(0), search.AngleScale(10, 1))
    circuit = search.gen_gate("Rz", angle, 0)
    energy, residual = search.eval_circ(sp.eye(2), [(1, "theta")], circuit)
    assert energy[0] == 0 and abs(energy[2]) < 1e-12
    assert energy[1] > 0
    assert not search.exact_residual_identity(residual)
    assert "Found exact solution" not in capsys.readouterr().out


def test_symbolic_only_evaluation_does_not_accept_its_dummy_numeric_zero(capsys):
    energy, _ = search.eval_circ(sp.eye(2), [], search.gen_gate("H", 0), symonly=True)
    assert energy[2] == 0 and energy[1] > 0
    assert "Found exact solution" not in capsys.readouterr().out


def test_phase_tolerance_is_explicit_and_zero_or_floats_cannot_be_certified():
    theta = sp.Symbol("theta", real=True)
    phase = sp.exp(sp.I*theta)*sp.eye(2)
    assert search.exact_residual_identity(phase, allow_global_phase=True)
    assert not search.exact_residual_identity(phase)
    assert not search.exact_residual_identity(sp.zeros(2), allow_global_phase=True)
    assert not search.exact_residual_identity(sp.eye(2)*sp.Float(1))


def test_exact_trigonometric_identity_is_proved_before_goal_acceptance():
    theta = sp.Symbol("theta", real=True)
    residual = (sp.sin(theta)**2+sp.cos(theta)**2)*sp.eye(2)
    assert search.exact_residual_identity(residual)


def test_find_min_keeps_the_cheapest_goal_instead_of_the_last_goal():
    energies = {"cheap": (0., 0., 0., 2.), "expensive": (0., 0., 0., 10.)}
    result = search.best_first(
        ["expensive", "cheap"], lambda energy: energy[1] == 0,
        lambda *unused: [], lambda node: (energies[node], node),
        lambda node: (), find_min=True, workers=1)
    assert result == "cheap"


def test_heap_ties_do_not_require_search_nodes_to_be_orderable():
    first, second = object(), object()
    result = search.best_first(
        [first, second], lambda energy: True, lambda *unused: [],
        lambda node: ((0., 0., 0., 2.), id(node)), lambda node: (), workers=1)
    assert result is first


@pytest.mark.parametrize("gate,expected", [("H", "H"), ("X", "X")])
def test_real_single_qubit_search_uses_its_own_register_seed(gate, expected):
    result = search.decompose_unitary_search(
        gate, 1, layers=1, basis=("CNOT", "H", "X"),
        allow_global_phase=False, find_min=True, workers=1)
    assert result is not None
    assert result[1] == ((expected, (0,)),)


def test_deap_does_not_insert_a_sampled_false_positive_into_the_exact_hall(monkeypatch):
    class Hall:
        def clear(self):
            pytest.fail("A sampled false positive cleared the exact-solution hall")
        def insert(self, unused):
            pytest.fail("A sampled false positive entered the exact-solution hall")
    monkeypatch.setattr(search, "eval_circ", lambda *args, **kwargs: ((0., 2., 0., 0.), sp.eye(2)))
    energy = search.evaluate([], None, Hall(), sp.eye(2), [], 1, [], search.COST_METHOD_TRACE)
    assert energy == (0., 2., 0., 0.)
