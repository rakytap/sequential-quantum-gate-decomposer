"""PAM parser caching preserves native matrices, endian handling and lifetime."""
import numpy as np
import pytest

from squander.gates.qgd_Circuit import qgd_Circuit
from squander.partitioning import routing
from squander.partitioning import _pam_circuit_cache as cache_module


def qasm(angle):
    return (
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; '
        'u3(%s,0.2,-0.3) q[0];' % angle
    )


def matrix(circuit):
    return np.asarray(circuit.get_unitary(), dtype=np.complex128)


@pytest.mark.parametrize("compensate", [False, True])
def test_native_conversion_matches_uncached_and_isolates_endian_mutation(
        compensate, monkeypatch):
    # Exercise the parser-cache fallback explicitly. The independent direct
    # conversion tests cover U3/CNOT columns that now avoid QASM entirely.
    monkeypatch.setattr(
        cache_module, "native_pam_circuit", lambda *args, **kwargs: None)
    circuit = qgd_Circuit(2)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(0)
    parameters = np.array([0.17, -0.23, 0.41], dtype=np.float64)
    convert = routing._squander_to_bqskit
    reference = matrix(convert(circuit, parameters,
                               compensate_endianness=compensate))
    with cache_module.pam_circuit_conversion_cache() as cache:
        first = convert(circuit, parameters, compensate_endianness=compensate)
        np.testing.assert_allclose(matrix(first), reference, atol=1e-14, rtol=0)
        first.renumber_qudits((1, 0))
        second = convert(circuit, parameters, compensate_endianness=compensate)
        assert first is not second
        np.testing.assert_allclose(matrix(second), reference, atol=1e-14, rtol=0)
        assert cache.hits == 1 and cache.misses == 1
    assert not cache.templates


def test_exact_angle_text_is_part_of_the_key():
    with cache_module.pam_circuit_conversion_cache() as cache:
        first = matrix(cache_module.decode_pam_qasm(qasm("0.1")))
        second = matrix(cache_module.decode_pam_qasm(qasm("0.10000000001")))
        assert cache.misses == 2 and cache.hits == 0
        assert np.linalg.norm(first - second) > 1e-12


def test_cache_is_bounded_and_uses_least_recently_used_eviction():
    with cache_module.pam_circuit_conversion_cache(max_entries=2) as cache:
        cache_module.decode_pam_qasm(qasm("0.1"))
        cache_module.decode_pam_qasm(qasm("0.2"))
        cache_module.decode_pam_qasm(qasm("0.1"))
        cache_module.decode_pam_qasm(qasm("0.3"))
        assert len(cache.templates) == 2
        assert qasm("0.1") in cache.templates and qasm("0.2") not in cache.templates
        assert cache.hits == 1 and cache.misses == 3
    assert not cache.templates


def test_nested_scopes_restore_parent_and_clear_on_failure():
    with cache_module.pam_circuit_conversion_cache() as outer:
        cache_module.decode_pam_qasm(qasm("0.1"))
        with pytest.raises(RuntimeError, match="controlled"):
            with cache_module.pam_circuit_conversion_cache() as inner:
                cache_module.decode_pam_qasm(qasm("0.2"))
                raise RuntimeError("controlled")
        assert not inner.templates
        cache_module.decode_pam_qasm(qasm("0.1"))
        assert outer.hits == 1 and outer.misses == 1
    assert not outer.templates


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_cache_capacity_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        with cache_module.pam_circuit_conversion_cache(value):
            pass


def test_cache_is_not_used_in_a_different_process(monkeypatch):
    with cache_module.pam_circuit_conversion_cache() as cache:
        cache_module.decode_pam_qasm(qasm("0.1"))
        monkeypatch.setattr(cache_module.os, "getpid", lambda: cache.pid + 1)
        cache_module.decode_pam_qasm(qasm("0.1"))
        assert cache.hits == 0 and cache.misses == 1
