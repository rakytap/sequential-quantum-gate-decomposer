"""Compare complete routes after certified native topology optimization.

This is an opt-in, incumbent-preserving routing policy, not OSR backtracking.
Every existing distinct route is evaluated after normal native cleanup.
The original post-optimized route is always evaluated.
Rejected branches never enter the caller's rewrite ledger or certificate.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time

from ._audit_composition import NativeAuditComposition, certificate_scope


@contextmanager
def _branch_audit(path, certificate):
    key = "SQUANDER_REWRITE_AUDIT_JSONL"
    previous = os.environ.get(key)
    os.environ[key] = str(path)
    try:
        with certificate_scope(certificate):
            yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def routing_outcome_key(outcome):
    """Minimize final CNOTs and preserve the original route on exact ties."""
    return (outcome["final_cnots"], not outcome["is_anchor"], outcome["index"])


def optimize_routing_portfolio(owner, circuit, parameters, backend):
    """Evaluate native routing alternatives without mutating the parent audit.

    Keep the legacy anchor and every distinct existing constructive route.
    No new synthesis catalog is built and no raw-cost shortlist is applied.
    """
    from squander.partitioning.routing import construct_exact_routed_circuit

    started = time.monotonic()
    input_representation = backend._squander_audit_representation(
        circuit, parameters, range(circuit.get_Qbit_Num()))
    shared_configuration = dict(owner.config)
    exact_route = None
    generation_seconds = 0.0
    post_seconds = 0.0
    anchor_signature = None
    anchor_bound = None
    seen = set()
    summaries = []

    def signature(raw_circuit, raw_parameters, solution):
        representation = backend._squander_audit_representation(
            raw_circuit, raw_parameters, range(raw_circuit.get_Qbit_Num()))
        return (backend._exact_state_sha256(
                    backend._qasm_exact_state(representation)),
                tuple(solution.initial_mapping), tuple(solution.final_mapping))

    with tempfile.TemporaryDirectory(prefix="squander-routing-portfolio-") as directory:
        def evaluate(index, raw_circuit=None, raw_parameters=None, solution=None):
            nonlocal exact_route, generation_seconds, post_seconds
            nonlocal anchor_signature, anchor_bound
            child = type(owner)(dict(shared_configuration))
            ledger = Path(directory) / ("candidate-%04d.jsonl" % index)
            child.config["_rewrite_audit_stage"] = "routing"
            with NativeAuditComposition(
                backend, input_representation,
                backend._circuit_validation_tolerance(child.config),
                backend._synthesis_acceptance_tolerance(child.config),
                topology=child.config["topology"],
            ) as certificate:
                certificate.enter_frame(input_representation)
                try:
                    with _branch_audit(ledger, certificate):
                        if solution is None:
                            route_started = time.monotonic()
                            raw_circuit, raw_parameters = child.route_circuit(
                                circuit, parameters)
                            generation_seconds = time.monotonic() - route_started
                            exact_route = child._routing_portfolio_result
                            solution = exact_route.solution
                            anchor = (exact_route.constructive_candidates[0]
                                      if exact_route.constructive_candidates
                                      else solution)
                            anchor_bound = anchor.cnot_count
                            anchor_circuit, anchor_parameters = construct_exact_routed_circuit(
                                anchor, circuit.get_Qbit_Num())
                            anchor_signature = signature(
                                anchor_circuit, anchor_parameters, anchor)
                        else:
                            candidate = replace(
                                exact_route, circuit=raw_circuit,
                                parameters=raw_parameters, solution=solution,
                                constructive_candidates=())
                            child._record_exact_routing_result(
                                circuit, parameters, candidate, time.time_ns())
                            child.check_valid_routing(raw_circuit, child.config["topology"])
                            child.check_compare_circuits(
                                circuit, parameters, raw_circuit, raw_parameters,
                                routing=True, label="routing_portfolio_candidate")
                        raw_signature = signature(raw_circuit, raw_parameters, solution)
                        seen.add(raw_signature)
                        child.config["routed_circuit"] = raw_circuit
                        child.config["routed_parameters"] = raw_parameters
                        child.config["_rewrite_audit_stage"] = "topology_optimization"
                        child.config.pop("topology_optimization_skipped", None)
                        print(
                            "Routing candidate %d (%s): %d raw CNOTs; explicit OSR cleanup."
                            % (index, solution.master_backend, solution.cnot_count),
                            flush=True,
                        )
                        post_started = time.monotonic()
                        fitted, fitted_parameters = child.OptimizeWideCircuit(
                            raw_circuit, raw_parameters)
                        elapsed = time.monotonic() - post_started
                        post_seconds += elapsed
                        child.check_valid_routing(fitted, child.config["topology"])
                        child.check_compare_circuits(
                            circuit, parameters, fitted, fitted_parameters,
                            routing=True, label="routing_portfolio_final")
                        output_representation = backend._squander_audit_representation(
                            fitted, fitted_parameters, range(fitted.get_Qbit_Num()))
                        certificate.require_current(output_representation)
                        proof = certificate.report()
                finally:
                    certificate.leave_frame()
            summary = {
                "index": index, "backend": solution.master_backend,
                "raw_cnots": solution.cnot_count,
                "final_cnots": int(backend.CNOTGateCount(fitted, 0)),
                "is_anchor": raw_signature == anchor_signature,
                "post_seconds": elapsed, "certified": True,
                "raw_circuit_sha256": raw_signature[0],
                "initial_mapping": list(solution.initial_mapping),
                "final_mapping": list(solution.final_mapping),
            }
            summaries.append(summary)
            return dict(summary, circuit=fitted, parameters=fitted_parameters,
                        configuration=dict(child.config), ledger=ledger, proof=proof)

        best = evaluate(0)
        duplicates = 0
        for solution in exact_route.constructive_candidates:
            raw_circuit, raw_parameters = construct_exact_routed_circuit(
                solution, circuit.get_Qbit_Num())
            if signature(raw_circuit, raw_parameters, solution) in seen:
                duplicates += 1
                continue
            candidate = evaluate(
                len(summaries), raw_circuit, raw_parameters, solution)
            if routing_outcome_key(candidate) < routing_outcome_key(best):
                best = candidate

        # Replay through the canonical writer, not by copying JSONL text:
        # an active large-circuit parent certificate must consume the winner.
        with best["ledger"].open(encoding="utf-8") as events:
            for line in events:
                if line.strip():
                    backend._append_rewrite_audit_event(json.loads(line))
        owner.config.update(best["configuration"])
        owner.config["routing_time"] = generation_seconds
        owner.config["optimization_time"] = post_seconds
        owner.config["routing_portfolio"] = {
            "policy": "certified-post-optimized-incumbent-portfolio",
            "raw_shortlist_is_heuristic": False,
            "raw_incumbent_bound": anchor_bound,
            "candidates": summaries, "selected_index": best["index"],
            "screened_higher_raw_cost": 0,
            "duplicate_routes": duplicates,
            "route_generation_seconds": generation_seconds,
            "total_post_seconds": post_seconds,
            "selected_post_seconds": best["post_seconds"],
            "total_policy_seconds": time.monotonic() - started,
            "winner_only_audit": True,
        }
        owner.config["circuit_validation_certificate"] = best["proof"]
        print(
            "Routing portfolio selected candidate %d: %d -> %d CNOTs; "
            "only its certified routing/cleanup chain is committed."
            % (best["index"], best["raw_cnots"], best["final_cnots"]),
            flush=True,
        )
        return best["circuit"], best["parameters"]
