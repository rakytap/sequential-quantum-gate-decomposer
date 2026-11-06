"""Free-parameter catalog cases and honest, bounded synthesis records."""

from contextlib import contextmanager
import signal
import time
import sympy as sp

if __package__:
    from . import symbolic_gates as gates
    from .symbolic_bases import get_basis
    from .symbolic_contract import retain_phase_results
    from .symbolic_synthesis import synthesize
else:
    import symbolic_gates as gates
    from symbolic_bases import get_basis
    from symbolic_contract import retain_phase_results
    from symbolic_synthesis import synthesize


def symbolic_cases():
    """All fixed-width families; explicitly scoped register fixtures.

    Optional real parameters, including Cirq global_shift, remain free.
    Integer widths, control states and matrix-input domains are fixtures,
    rather than a claim to have covered every size or arbitrary unitary.
    """
    for definition in gates.GATE_DEFINITIONS.values():
        if definition.num_qubits is not None:
            parameters=tuple(sp.Symbol(name,real=True) for name in definition.parameters)
            yield definition.name,parameters,"free_real_family" if parameters else "fixed_gate"
    theta,alpha,beta=sp.symbols("theta alpha beta",real=True)
    fixtures=[
        ("GP",(theta,1)),("Identity",(3,)),("Controlled",(gates.make_gate("Sx"),2)),
        ("MCX",(2,)),("MCPhase",(theta,2)),("MCMT",(gates.make_gate("H"),1,2)),
        ("QFT",(3,)),("QFTdg",(3,)),("WalshHadamard",(3,)),
        ("Pauli",("XYZ",)),("PauliRotation",("XYZ",theta)),
        ("Diagonal",((1,sp.I,-1,-sp.I),)),("Unitary",(gates.make_gate("B"),)),
        ("Permutation",((2,0,1),)),("LinearFunction",(((1,1),(0,1)),)),
        ("UC",((gates.make_gate("H"),gates.make_gate("Sx")),)),
        ("UCRX",((alpha,beta),)),("UCRY",((alpha,beta),)),("UCRZ",((alpha,beta),)),
        ("MS",(3,theta)),
    ]
    for name,parameters in fixtures:
        yield name,parameters,"register_fixture"


class SymbolicDeadline(BaseException):
    """Keep algebra libraries' broad Exception handlers from swallowing alarms."""


@contextmanager
def wall_time_limit(seconds):
    """Unix worker deadline; interrupt algebra without accepting partial proofs."""
    if seconds is None:
        yield
        return
    if seconds<=0:raise ValueError("A positive time limit is required")
    def expired(unused_signal,unused_frame):
        raise SymbolicDeadline("Symbolic synthesis wall-clock budget expired")
    previous=signal.signal(signal.SIGALRM,expired)
    old_timer=signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        if old_timer[0]:signal.setitimer(signal.ITIMER_REAL,*old_timer)


def bounded_synthesis(name,parameters,basis,*,phase_exact,seconds):
    start=time.monotonic()
    try:
        with wall_time_limit(seconds):
            result=synthesize(name,parameters,basis,phase_exact=phase_exact,
                              osr_seconds=min(10,max(0.1,seconds/3)))
    except SymbolicDeadline:
        result=dict(status="timeout",phase_exact=phase_exact,reason="wall_clock_budget",budget_seconds=seconds)
    except Exception as error:
        # Failures are archived as failures, never mathematical impossibility.
        result=dict(status="compiler_error",phase_exact=phase_exact,
                    error_type=type(error).__name__,reason=str(error))
    return dict(result,elapsed_seconds=time.monotonic()-start)


def archive_cell(job):
    name,parameters,scope,basis,seconds=job
    exact=bounded_synthesis(name,parameters,basis,phase_exact=True,seconds=seconds)
    projective=bounded_synthesis(name,parameters,basis,phase_exact=False,seconds=seconds)
    return dict(gate=name,parameters=[str(p) for p in parameters],target_scope=scope,basis=basis,
                phase_statuses=dict(phase_exact=exact["status"],projective=projective["status"]),
                **retain_phase_results(exact,projective))


def comparison_contract():
    return dict(objective=["native_two_qubit_operations","original_parameter_sensitive_score","single_qubit_operations"],
                phase_exact="Literal equality; physical GP and free scalar corrections forbidden",
                projective="Exact equality after automatic virtual scalar metadata at zero physical cost",
                basis_policy="Preserve every supplied primitive; no silent augmentation",
                ancillas=0,connectivity="all_to_all",input_states="arbitrary",wire_order="fixed_input_and_output",
                proof_policy="Exact symbolic identities only; numerical samples cannot certify a family",
                optimality="A verified construction is an upper bound unless a matching certificate is attached",
                novelty="No novelty claim follows from beating an SDK output",
                hardware_domains="Mathematical families; consult basis notes for hardware angle ranges and TK2 macro accounting")
