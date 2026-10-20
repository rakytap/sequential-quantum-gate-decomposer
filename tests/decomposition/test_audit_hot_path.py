"""Exact passthrough proofs and bounded audit replay caches."""
import copy
import importlib
import json
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from squander.decomposition._audit_replay_cache import AuditReplayCache, ExactJSONCache, current_cache
from squander.decomposition import _audit_composition as composition
from squander.gates.qgd_Circuit import qgd_Circuit
backend=importlib.import_module("squander.decomposition.qgd_Wide_Circuit_Optimization")


def rep(circuit):
    from squander.decomposition._audit_qasm import dump_native_audit_qasm
    return {"format":"squander_openqasm2","qubits":circuit.num_qubits,
            "qasm":dump_native_audit_qasm(circuit)}


def event(before,after,selections):
    return dict(kind="round",component="squander_wide_optimization",event_id="round",
                timestamp_ns=3,started_ns=1,stage="all_to_all",input=rep(before),
                output=rep(after),selections=selections,
                input_sha256=backend._exact_state_sha256(backend._qasm_exact_state(rep(before))),
                output_sha256=backend._exact_state_sha256(backend._qasm_exact_state(rep(after))))


def passthrough(indices,qubits):
    return dict(passthrough=True,source_gate_indices=indices,qubits=qubits)


def test_exact_cache_returns_fresh_values_and_obeys_budgets():
    cache=ExactJSONCache(max_bytes=4096,max_entries=2)
    calls=[]
    def compute():
        calls.append(True)
        return {"value":[1]}
    first=cache.get_or_compute(b"one",compute)
    first["value"][0]=99
    assert cache.get_or_compute(b"one",compute)=={"value":[1]}
    assert len(calls)==1
    cache.get_or_compute(b"two",compute)
    cache.get_or_compute(b"three",compute)
    assert len(cache.entries)==2 and cache.bytes<=4096
    cache.get_or_compute(b"one",compute)
    assert len(calls)==4
    tiny=ExactJSONCache(max_bytes=1)
    tiny.get_or_compute(b"x",compute)
    assert not tiny.entries


def test_cache_keys_preserve_parameter_bit_distinctions():
    a,b=QuantumCircuit(1),QuantumCircuit(1)
    a.rz(0.7,0)
    b.rz(np.nextafter(0.7,np.inf),0)
    with AuditReplayCache() as cache:
        sa=backend._qasm_exact_state(rep(a))
        sb=backend._qasm_exact_state(rep(b))
        assert sa!=sb and cache.states.misses==2
        sa["operations"].clear()
        assert backend._qasm_exact_state(rep(a))["operations"]
    assert current_cache() is None


def test_passthrough_replay_never_requests_a_numerical_certificate():
    circuit=QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0,1)
    item=event(circuit,circuit,[passthrough([0],[0]),passthrough([1],[0,1])])
    class NoLookup:
        def get(self,key):
            raise AssertionError("Passthroughs must not look up rewrites.")
    before,after=backend._verify_native_round_replay(item,NoLookup())
    assert before==after==backend._qasm_exact_state(rep(circuit))


@pytest.mark.parametrize("damage",[
    "parameter","support","duplicate_qubit","duplicate_gate","missing_gate",
    "causal_order","ambiguous_id","nonboolean",
])
def test_passthrough_tampering_fails_closed(damage):
    circuit=QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0,1)
    output=circuit.copy()
    selections=[passthrough([0],[0]),passthrough([1],[0,1])]
    if damage=="parameter":output.rz(0.001,0)
    elif damage=="support":selections[1]["qubits"]=[0]
    elif damage=="duplicate_qubit":selections[1]["qubits"]=[0,1,1]
    elif damage=="duplicate_gate":selections.append(passthrough([0],[0]))
    elif damage=="missing_gate":selections.pop()
    elif damage=="causal_order":selections.reverse()
    elif damage=="ambiguous_id":selections[0]["rewrite_event_id"]="some-rewrite"
    else:selections[0]["passthrough"]=1
    with pytest.raises(AssertionError):
        backend._verify_native_round_replay(event(circuit,output,selections),{})


def test_disjoint_passthrough_reordering_is_exact():
    a,b=QuantumCircuit(2),QuantumCircuit(2)
    a.h(0);a.x(1)
    b.x(1);b.h(0)
    backend._verify_native_round_replay(
        event(a,b,[passthrough([1],[1]),passthrough([0],[0])]),{})
    np.testing.assert_array_equal(Operator(a).data,Operator(b).data)


def test_passthrough_keeps_entangler_support_in_composition_layers():
    circuit=QuantumCircuit(21)
    circuit.h(0)
    circuit.cx(0,1)
    initial=rep(circuit)
    with composition.NativeAuditComposition(backend,initial,1e-8,1e-10) as proof:
        proof.consume(event(circuit,circuit,[passthrough([0],[0]),passthrough([1],[0,1])]))
        proof.require_current(initial)
        assert proof.report()["global_process_infidelity_bound"]==0
        assert proof.selected_rewrites==2


def test_export_and_verify_are_independent_and_replay_each_round_once(tmp_path,monkeypatch):
    circuit=QuantumCircuit(2)
    circuit.h(0);circuit.cx(0,1)
    representation=rep(circuit)
    events=[dict(kind="rewrite",component="test",event_id=f"rewrite-{i}",
                 timestamp_ns=i,before=representation,after=representation,
                 tolerance=1e-10,accepted=True) for i in range(30)]
    item=event(circuit,circuit,[dict(rewrite_event_id="rewrite-0",
               source_gate_indices=[0,1],qubits=[0,1])])
    item["timestamp_ns"]=31
    events.append(item)
    source=tmp_path/"events.jsonl"
    source.write_text("\n".join(json.dumps(x) for x in events))
    run=dict(input_circuit=representation,output_circuit=representation,
             input_circuit_sha256=item["input_sha256"],output_circuit_sha256=item["output_sha256"])
    metric_calls=[]
    real_metrics=backend._unitary_audit_metrics
    def metrics(before,after):
        metric_calls.append(True)
        return real_metrics(before,after)
    monkeypatch.setattr(backend,"_unitary_audit_metrics",metrics)
    audit,digest=backend.save_rewrite_audit(source,tmp_path/"audit.gz",run)
    assert len(metric_calls)==1
    round_calls=[]
    real_round=backend._verify_native_round_replay
    def replay(item,index):
        round_calls.append(True)
        return real_round(item,index)
    monkeypatch.setattr(backend,"_verify_native_round_replay",replay)
    report=backend.verify_rewrite_audit(audit,expected_sha256=digest,audit_path=tmp_path/"audit.gz")
    assert report["verified_rewrites"]==30
    assert len(metric_calls)==2 and len(round_calls)==1
    damaged=copy.deepcopy(audit)
    damaged["events"][5]["metrics"]["phase_aligned_spectral_bits"]="corrupted"
    with pytest.raises(AssertionError,match="not bit-identical"):
        backend.verify_rewrite_audit(damaged)


def test_native_generation_omits_unchanged_rewrite_events(tmp_path,monkeypatch):
    path=tmp_path/"native.jsonl"
    monkeypatch.setenv("SQUANDER_REWRITE_AUDIT_JSONL",str(path))
    circuit=qgd_Circuit(2)
    circuit.add_CNOT(1,0)
    owner=backend.qgd_Wide_Circuit_Optimization({
        "max_partition_size":3,"partition_workers":1,"parallel":0,
        "use_osr":True,"use_graph_search":True,"random_seed":20260926,
        "test_final_circuit":True,
    })
    fitted,parameters=owner.OptimizeWideCircuit(circuit,np.empty(0))
    np.testing.assert_allclose(circuit.get_Matrix(np.empty(0)),fitted.get_Matrix(parameters),
                               atol=1e-12,rtol=0)
    events=[json.loads(line) for line in path.read_text().splitlines()]
    rounds=[x for x in events if x.get("kind")=="round"]
    assert rounds
    assert any(s.get("passthrough") for x in rounds for s in x["selections"])
    assert not any(x.get("component")=="squander_selected_partition_rewrite"
                   and x["before"]==x["after"] for x in events)

