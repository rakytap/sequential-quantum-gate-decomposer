"""Resume completed benchmarks at routing, retaining certified incumbents.

Run from a checkout containing the native extensions, e.g.::

    PYTHONPATH=. python benchmarks/partitioning/replay_routing.py \
        --source-root /path/to/original/checkout --jobs 8

The source tree is read only. Checkpoints precede final synthesis, so an
interruption never requires rebuilding A2A or the permutation catalogue.
Timings charge the original complete experiment plus newly measured work;
they are explicitly cumulative, not an invented fresh-run measurement.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = Path("benchmarks/partitioning")
WEIGHTS = (1.5, 3.0, 6.0, 12.0)
POLICY = "cached-pam-cover-and-packing-v1"


@contextmanager
def search_budget(seconds):
    """Bound optional heuristic work; the previously verified incumbent wins."""
    def expired(*_args):
        raise TimeoutError("Optional routing seed exceeded its wall-clock budget.")
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execution_identity(backend, representation, initial, final):
    """Counts alone never establish that final synthesis can be reused."""
    return (backend._exact_state_sha256(backend._qasm_exact_state(representation)),
            tuple(initial), tuple(final))


def cumulative_timing(archived, routing_seconds, final_seconds):
    result = dict(archived)
    for field, extra in (("routing", routing_seconds), ("optimization", final_seconds),
                         ("total", routing_seconds + final_seconds)):
        if result.get(field) is not None:
            result[field] = round(result[field] + extra, 2)
    return result


def recover_interrupted_final_attempt(case_dir, now=None):
    """Charge an abandoned attempt conservatively, without charging it twice."""
    pending = case_dir / "final-attempt.json"
    history = case_dir / "failed-attempts.json"
    attempts = json.loads(history.read_text()) if history.exists() else []
    if pending.exists():
        attempt = json.loads(pending.read_text())
        if attempt.get("status") == "running":
            now = time.monotonic() if now is None else now
            attempt_id = str(attempt["started_monotonic"])
            if not any(a.get("attempt_id") == attempt_id for a in attempts):
                attempts.append(dict(
                    attempt_id=attempt_id, stage="final-synthesis",
                    outcome="interrupted before verification completed",
                    timing_basis="conservative-wall-clock-upper-bound",
                    seconds=max(0.0, now - attempt["started_monotonic"]),
                    includes_preparation_and_recovery_delay=True))
                write_json(history, attempts)
            attempt["status"] = "charged"
            write_json(pending, attempt)
    return attempts


def verify_routing_checkpoint(saved, archived, backend):
    """Reuse a proved route across code updates, retaining its own provenance."""
    route = saved["route"]
    if route["input_sha256"] != archived["input_sha256"]:
        raise ValueError("Routing checkpoint has a different exact source.")
    if route["cnot_count"] > archived["cnot_count"]:
        raise ValueError("Routing checkpoint regresses the archived incumbent.")
    backend._verify_exact_osr_routing_replay(route)
    return route


def archived_routing_incumbent(old, audit, source_root, backend):
    """Find the best raw route even when an older complete final pipeline won."""
    from squander.decomposition._rewrite_audit_stream import close_audit, iter_audit_events
    routes = list(iter_audit_events(audit["events"], {"exact_osr_routing"}))
    if not routes:
        return None, old["rewrite_audit"]["sha256"]
    if len(routes) != 1:
        raise ValueError("Expected one archived exact route.")
    expected = old["routed"]["cnot_equiv"]
    prior = old.get("routing_replay", {})
    candidate_info = prior.get("candidate_audit")
    if candidate_info is None:
        if routes[0]["cnot_count"] != expected:
            raise ValueError("Archived routed count has no matching certificate.")
        return routes[0], old["rewrite_audit"]["sha256"]
    path = source_root / candidate_info["file"]
    if sha256(path) != candidate_info["sha256"]:
        raise ValueError("Archived routing candidate audit digest mismatch.")
    shared = candidate_info["sha256"] == old["rewrite_audit"]["sha256"]
    candidate = audit if shared else backend.load_rewrite_audit(path)
    try:
        if candidate["run"]["input_circuit_sha256"] != audit["run"]["input_circuit_sha256"]:
            raise ValueError("Archived raw and final pipelines have different inputs.")
        selected = list(iter_audit_events(candidate["events"], {"exact_osr_routing"}))
        if (len(selected) != 1 or selected[0]["cnot_count"] != expected
                or selected[0]["input_sha256"] != routes[0]["input_sha256"]):
            raise ValueError("Archived candidate does not certify the best routed result.")
        if sha256(source_root / prior["candidate_qasm"]) != candidate["run"]["output_file_sha256"]:
            raise ValueError("Archived candidate QASM digest mismatch.")
        backend._verify_exact_osr_routing_replay(selected[0])
        return selected[0], candidate_info["sha256"]
    finally:
        if not shared:
            close_audit(candidate)


def retain_prior_candidate(old, source_root, provenance):
    """Keep the raw-route certificate when its final pass is reused."""
    prior = old.get("routing_replay", {})
    if not prior.get("candidate_audit"):
        return
    for relative in (prior["candidate_audit"]["file"], prior["candidate_qasm"]):
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    for key in ("candidate_audit", "candidate_qasm", "candidate_final", "candidate_adopted",
                "raw_initial_mapping", "raw_final_mapping", "final_implementation_sha256", "final_resume"):
        if key in prior:
            provenance[key] = deepcopy(prior[key])


def final_prefix_boundary(path, route):
    """Find a completed whole-circuit round, excluding an unfinished tail."""
    boundary, last, rounds, matched_route = 0, None, 0, False
    with path.open("rb") as source:
        while True:
            line = source.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if event.get("kind") == "exact_osr_routing":
                matched_route = (event.get("event_id") == route.get("event_id")
                                 and event.get("output_sha256") == route.get("output_sha256"))
            if (matched_route and event.get("stage") == "topology_optimization"
                    and event.get("kind") in ("round", "squander_basis_conversion")):
                rounds += event["kind"] == "round"
                if rounds:
                    boundary, last = source.tell(), event
    return (boundary, last, rounds) if last is not None else None


def resume_final_prefix(case_dir, ledger, route, audit, backend):
    """Reverify completed rounds before resuming interrupted final synthesis."""
    candidates = []
    seen = set()
    for path in (case_dir / "candidate.interrupted.jsonl", ledger):
        if not path.exists():
            continue
        signature = (path.stat().st_size, path.stat().st_mtime_ns)
        if signature in seen:
            continue
        seen.add(signature)
        boundary = final_prefix_boundary(path, route)
        if boundary is not None:
            end, event, rounds = boundary
            counts = event["output"].get("gate_counts", {})
            cnots = sum(backend.CNOT_COUNT_DICT.get(name, 0) * count
                        for name, count in counts.items())
            candidates.append((cnots, -rounds, path, end, event))
    if not candidates:
        return None
    _, negative_rounds, source_path, end, event = min(candidates, key=lambda item: item[:2])
    prefix = case_dir / "final-prefix.jsonl"
    with source_path.open("rb") as source, prefix.open("wb") as destination:
        remaining = end
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Interrupted final ledger changed during recovery.")
            destination.write(chunk)
            remaining -= len(chunk)
    circuit, parameters = decode(backend, event["output"])
    representation = backend._squander_audit_representation(
        circuit, parameters, range(circuit.get_Qbit_Num()))
    if backend._exact_state_sha256(backend._qasm_exact_state(representation)) != event["output_sha256"]:
        raise ValueError("Interrupted final gate stream did not round-trip exactly.")
    if backend.CNOTGateCount(circuit) > route["cnot_count"]:
        raise ValueError("Interrupted final prefix regresses its routing input.")
    from benchmarks.partitioning.wide_circuit_optimization import save_qasm2
    qasm = case_dir / "final-prefix.qasm"
    save_qasm2(circuit, parameters, qasm)
    run = deepcopy(audit["run"])
    run.update(output_file=str(qasm.relative_to(ROOT)), output_circuit=representation,
               output_circuit_sha256=event["output_sha256"], output_file_sha256=sha256(qasm),
               initial_mapping=route["initial_mapping"], final_mapping=route["final_mapping"])
    run.pop("result_checkpoint", None)
    proof_path = case_dir / "final-prefix.audit.json.gz"
    proof, digest = backend.save_rewrite_audit(prefix, proof_path, run)
    from squander.decomposition._rewrite_audit_stream import close_audit
    try:
        verification = backend.verify_rewrite_audit(proof, expected_sha256=digest,
                                                    audit_path=proof_path)
    finally:
        close_audit(proof)
    shutil.copyfile(prefix, ledger)
    return circuit, parameters, dict(
        completed_rounds=-negative_rounds, output_sha256=event["output_sha256"],
        checkpoint_audit_sha256=digest, verified_rewrites=verification["verified_rewrites"])


@contextmanager
def deterministic_cover_solver():
    """Pin CBC's seed, threads and limits without changing synthesis settings."""
    from unittest.mock import patch
    import squander.partitioning.ilp as ilp

    original_ilp, original_cbc = ilp.ilp_global_optimal, ilp.cbc_solver

    def solve(*args, **kwargs):
        kwargs["gurobi_direct"] = False
        return original_ilp(*args, **kwargs)

    def cbc(pulp, **kwargs):
        kwargs.update(threads=1, timeLimit=90, maxNodes=5000,
                      options=["randomSeed 17", "randomCbcSeed 17"])
        return original_cbc(pulp, **kwargs)

    with patch.object(ilp, "ilp_global_optimal", solve), patch.object(ilp, "cbc_solver", cbc), \
            patch.object(ilp, "_GUROBI_AVAILABLE", False), \
            patch.object(ilp, "_GUROBI_AVAILABILITY_ERROR", RuntimeError("Replay pins deterministic CBC")), \
            patch.object(ilp, "_GUROBI_FALLBACK_WARNING_PRINTED", True):
        yield


def cached_portfolio(catalog, archived_route, progress):
    """Apply the same finite heuristic portfolio to every available circuit.

    Retain the archived route; compare existing cover policies under all four
    PAM weights, then add geometry-guided covers and packing-aware native
    beams. No comparator counts or circuit names enter this policy.
    """
    import numpy as np
    import squander.partitioning.routing as routing
    from squander.partitioning.ilp import get_all_partitions
    from squander.partitioning.partition import PartitionCircuit
    from squander.partitioning._routing_catalog_replay import expand_path_cover

    catalog = routing.augment_catalog_with_recombined_covers(catalog, strategies=("kahn",))
    n = catalog.circuit.get_Qbit_Num()
    path = routing._path_topology_order(catalog.topology, n)
    _, _, _, predecessors, *_ = get_all_partitions(catalog.circuit, 3)
    lookup = {gate_set: i for i, gate_set in enumerate(catalog.partitions)}
    covers = {}
    variants = []
    best = None
    best_count = archived_route["cnot_count"]

    def remember(candidate, label, elapsed, timed_out=False):
        nonlocal best, best_count
        variants.append(dict(label=label, seconds=round(elapsed, 6),
                             timed_out=timed_out,
                             cnots=None if candidate is None else candidate.cnot_count))
        if candidate is not None and candidate.cnot_count < best_count:
            best, best_count = candidate, candidate.cnot_count
        progress(label, best_count, variants)

    # The incumbent cover is available even when a solver chooses different
    # equal-cost single-qubit chains from those in the archived catalogue.
    archived_sets = [frozenset(s["source_gate_indices"]) for s in archived_route["selections"]]
    if all(gate_set in lookup for gate_set in archived_sets):
        covers["archived"] = tuple(lookup[gate_set] for gate_set in archived_sets)
    for strategy in ("kahn", "ilp", "ilp-routing"):
        begin = time.monotonic()
        try:
            with search_budget(120):
                _, _, parts = PartitionCircuit(catalog.circuit, catalog.parameters, 3, strategy=strategy)
        except TimeoutError:
            variants.append(dict(label="cover-" + strategy, timed_out=True))
            continue
        if all(frozenset(part) in lookup for part in parts):
            covers[strategy] = tuple(lookup[frozenset(part)] for part in parts)
        progress("cover-" + strategy, best_count, variants)
    for name, weights in (("minimum-partitions", None),
                          ("minimum-local-cnots", [min(a.cnot_count for a in values)
                                                   for values in catalog.alternatives.values()])):
        try:
            with search_budget(120):
                covers[name] = routing._minimum_partition_cover_selection(
                    gate_predecessors=predecessors, partitions=catalog.partitions, weights=weights)
        except TimeoutError:
            variants.append(dict(label=name, timed_out=True))
    initial = tuple(archived_route["initial_mapping"])
    final = tuple(archived_route["final_mapping"])
    unique_covers = {}
    for name, cover in covers.items():
        unique_covers.setdefault(tuple(cover), name)
    for cover, name in list(unique_covers.items()):
        # PAM relabels canonical programs itself; full device translations
        # are only required by native layout/beam search below.
        for layout_name, layout in (("free", None), ("incumbent", initial)):
            for weight in WEIGHTS:
                begin = time.monotonic()
                label = f"pam-{name}-{layout_name}-{weight:g}"
                expired = False
                try:
                    candidate = routing._precomputed_osr_pam_warm_start(
                        selected_partitions=cover, gate_predecessors=predecessors,
                        partitions=catalog.partitions, alternatives=catalog.alternatives,
                        logical_qubit_count=n, topology=catalog.topology, path=path,
                        layout_passes=3, swap_cnot_cost=weight, initial_mapping=layout,
                        master_backend=label, deadline=begin + 300)
                except routing.ExactRoutingLimitExceeded:
                    candidate, expired = None, True
                remember(candidate, label, time.monotonic() - begin, expired)
    # Reference trajectory -> additive gathering/permutation price -> cover
    # -> executable route; repeat once around the produced trajectory.
    begin = time.monotonic()
    observed_geometry = []
    def observe_geometry(candidate):
        observed_geometry.append(candidate)
        remember(candidate, f"geometry-{len(observed_geometry) - 1}", time.monotonic() - begin)
    try:
        with search_budget(max(300, 15 * n)):
            geometry = routing.mapping_aware_catalog_seeds(
                catalog, reference_seed_count=1, refinement_rounds=2, layout_passes=1,
                swap_cnot_cost=1.5, weight_swap_cnot_cost=1.5,
                fixed_cover_refine_seconds=0, fixed_cover_beam_width=4,
                use_packing_bound=True, recombine_covers=False, on_candidate=observe_geometry)
    except TimeoutError:
        geometry = tuple(observed_geometry)
        variants.append(dict(label="geometry", timed_out=True))
    for index, candidate in enumerate(geometry):
        remember(candidate, f"geometry-{index}", time.monotonic() - begin if index == 0 else 0)
        cover = tuple(s.partition for s in candidate.selections)
        unique_covers.setdefault(cover, f"geometry-{index}")
    # Widen the same search for every cover, keeping both priorities. Width
    # follows the existing device-size schedule, capped at 32 for this replay.
    width = min(32, max(8, 1024 // n))
    geometry_layouts = tuple(dict.fromkeys(c.initial_mapping for c in geometry))
    native_candidates = []
    for cover, name in unique_covers.items():
        expanded = expand_path_cover(catalog, cover)
        for packing in (False, True):
            begin = time.monotonic()
            label = f"native-{name}-packing-{packing}"
            candidate = routing._cover_selection_warm_start(
                selected_partitions=cover, gate_predecessors=predecessors,
                partitions=catalog.partitions, alternatives=expanded.alternatives,
                logical_qubit_count=n, path=path, initial_mapping=None,
                beam_initial_mappings=(None, *geometry_layouts, initial, final),
                master_backend=label, lookahead=True,
                exact_refine_timeout_seconds=None, beam_width=width,
                translation_limit=8, deadline=begin + max(180, 5 * n),
                layout_seed_count=0, layout_total_passes=3,
                layout_candidate_limit=2, use_packing_bound=packing)
            remember(candidate, label, time.monotonic() - begin)
            if candidate is not None:
                native_candidates.append((candidate.cnot_count, candidate.single_qubit_count,
                                          len(native_candidates), cover, name, packing, candidate))
    if width < 32:
        for _, _, _, cover, name, packing, previous in sorted(native_candidates)[:2]:
            expanded = expand_path_cover(catalog, cover)
            begin = time.monotonic()
            label = f"native-wide-{name}-packing-{packing}"
            candidate = routing._cover_selection_warm_start(
                selected_partitions=cover, gate_predecessors=predecessors,
                partitions=catalog.partitions, alternatives=expanded.alternatives,
                logical_qubit_count=n, path=path, initial_mapping=None,
                beam_initial_mappings=(None, *geometry_layouts, initial),
                master_backend=label, lookahead=True, preferred_solution=previous,
                exact_refine_timeout_seconds=None, beam_width=32,
                translation_limit=8, deadline=begin + 300,
                layout_seed_count=0, layout_total_passes=3,
                layout_candidate_limit=2, use_packing_bound=packing)
            remember(candidate, label, time.monotonic() - begin)
    if best is None:
        return catalog, None, variants
    circuit, parameters = routing.construct_exact_routed_circuit(best, n)
    exact = routing.ExactCircuitRoutingResult(
        circuit=circuit, parameters=np.asarray(parameters), solution=best,
        candidate_gate_sets=tuple(tuple(sorted(g)) for g in catalog.partitions),
        candidate_gate_orders=catalog.gate_orders)
    return catalog, exact, variants


def inventory(source_root):
    rows, incomplete = [], []
    for size in (3, 4):
        for dataset in ("IBMEagle", "QASMBenchmarks"):
            directory = BENCHMARKS / f"{dataset}_results_{size}qbit_TreeSearch"
            results_path = source_root / directory / "results.json"
            results = json.loads(results_path.read_text())
            for input_file in sorted((source_root / BENCHMARKS / dataset).glob("*.qasm")):
                entry = results.get(input_file.name, {})
                row = dict(size=size, dataset=dataset, file=input_file.name,
                           directory=str(directory), entry=entry)
                if entry.get("status") == "completed":
                    rows.append(row)
                else:
                    incomplete.append({**row, "reason": entry.get("status", "not_run")})
    return rows, incomplete


def decode(backend, representation):
    import numpy as np
    from qiskit import qasm2
    from squander import Qiskit_IO
    circuit, parameters = Qiskit_IO.convert_Qiskit_to_Squander(
        qasm2.loads(representation["qasm"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
    return circuit.get_Flat_Circuit(), np.asarray(parameters, dtype=np.float64)


def worker(args):
    import numpy as np
    import random
    import resource
    import squander.partitioning.routing as routing
    import squander.decomposition.qgd_Wide_Circuit_Optimization as backend
    from squander.decomposition._rewrite_audit_stream import close_audit, iter_audit_events

    resource.setrlimit(resource.RLIMIT_AS, (96 * 1024**3, 96 * 1024**3))
    random.seed(20261111)
    np.random.seed(20261111)
    implementation_digests = {
        str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "squander/partitioning/routing.py",
            ROOT / "squander/partitioning/_routing_catalog_replay.py")}
    source_root = args.source_root.resolve()
    row = json.loads((args.state_dir / "inventory.json").read_text())["completed"][args.case]
    old = row["entry"]
    case_dir = args.state_dir / f"case-{args.case:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    status_path = case_dir / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        return
    status = dict(case=args.case, dataset=row["dataset"], size=row["size"], file=row["file"],
                  status="loading-audit", policy=POLICY, variants=[])
    write_json(status_path, status)
    audit_info = old["rewrite_audit"]
    old_audit_path = source_root / audit_info["file"]
    if sha256(old_audit_path) != audit_info["sha256"]:
        raise ValueError("Archived rewrite audit digest mismatch.")
    audit = backend.load_rewrite_audit(old_audit_path)
    try:
        route_events = list(iter_audit_events(audit["events"], {"exact_osr_routing"}))
        if len(route_events) > 1:
            raise ValueError("Expected one archived exact route.")
        output = ROOT / old["output_file"]
        output.parent.mkdir(parents=True, exist_ok=True)
        original_output = source_root / old["output_file"]
        if sha256(original_output) != audit["run"]["output_file_sha256"]:
            raise ValueError("Archived final circuit digest mismatch.")
        shutil.copy2(original_output, output)
        canonical_audit = ROOT / audit_info["file"]
        if old.get("routing_osr_catalog"):
            catalog_file = old["routing_osr_catalog"]["file"]
            archived_catalog_copy = ROOT / catalog_file
            if not archived_catalog_copy.exists():
                shutil.copy2(source_root / catalog_file, archived_catalog_copy)
        checkpoint = case_dir / "route.json"
        routing_implementation_digests = implementation_digests
        failed_attempts = recover_interrupted_final_attempt(case_dir)
        route_seconds = 0.0
        exact = None
        catalog = None
        route, routing_audit_sha256 = archived_routing_incumbent(old, audit, source_root, backend)
        original_route = route
        if not route_events:
            # Already topology valid: there is no PAM stage to recompute.
            status.update(status="completed", shortcut="input-already-topology-valid",
                          entry=deepcopy(old))
            shutil.copy2(old_audit_path, canonical_audit)
            write_json(status_path, status)
            return
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["policy"] != POLICY or saved["old_audit_sha256"] != audit_info["sha256"]:
                raise ValueError("Routing checkpoint belongs to another experiment.")
            route = verify_routing_checkpoint(saved, original_route, backend)
            route_seconds = saved["seconds"]
            routing_implementation_digests = saved["implementation_sha256"]
            status["variants"] = saved["variants"]
        else:
            started = time.monotonic()
            status["status"] = "loading-catalogue"
            write_json(status_path, status)
            catalog_info = old["routing_osr_catalog"]
            catalog = routing.reconstruct_routing_osr_catalog(
                source_root / catalog_info["file"], catalog_info["sha256"], compact_path=True)
            from squander.partitioning._routing_catalog_replay import bind_catalog_source
            precise_circuit, precise_parameters = decode(backend, route["input"])
            precise_representation = backend._squander_audit_representation(
                precise_circuit, precise_parameters, range(precise_circuit.get_Qbit_Num()))
            if backend._exact_state_sha256(backend._qasm_exact_state(precise_representation)) != route["input_sha256"]:
                raise ValueError("Lossless archived A2A source did not round-trip exactly.")
            catalog = bind_catalog_source(catalog, precise_circuit, precise_parameters)
            original_synthesis = {}

            def forbidden(*_args, **_kwargs):
                raise AssertionError("Routing replay must never invoke local synthesis.")

            def progress(label, count, variants):
                status.update(status="routing", phase=label, best_routed_cnots=count,
                              variants=variants, routing_elapsed=time.monotonic() - started)
                write_json(status_path, status)

            try:
                for name in ("_iter_refined_routing_catalog", "_refine_boundary_synthesis",
                             "_call_shared_synthesis", "_call_shared_synthesis_batch"):
                    if hasattr(routing, name):
                        original_synthesis[name] = getattr(routing, name)
                        setattr(routing, name, forbidden)
                from squander.partitioning._pam_circuit_cache import pam_circuit_conversion_cache
                with deterministic_cover_solver(), pam_circuit_conversion_cache():
                    catalog, exact, variants = cached_portfolio(catalog, route, progress)
            finally:
                for name, function in original_synthesis.items():
                    setattr(routing, name, function)
            route_seconds = time.monotonic() - started
            if exact is not None:
                event_ledger = case_dir / "route-event.jsonl"
                event_ledger.unlink(missing_ok=True)
                os.environ["SQUANDER_REWRITE_AUDIT_JSONL"] = str(event_ledger)
                try:
                    backend._append_exact_osr_routing_event(
                        "routing", time.time_ns(), catalog.circuit, catalog.parameters,
                        exact.circuit, exact.parameters, exact, catalog.topology,
                        old["configuration"]["synthesis_acceptance_tolerance"])
                finally:
                    os.environ.pop("SQUANDER_REWRITE_AUDIT_JSONL", None)
                route = json.loads(event_ledger.read_text())
                backend._verify_exact_osr_routing_replay(route)
                if route["input_sha256"] != route_events[0]["input_sha256"]:
                    raise ValueError("Catalogue round-trip changed the exact archived A2A gate stream.")
            write_json(checkpoint, dict(policy=POLICY, old_audit_sha256=audit_info["sha256"],
                                        implementation_sha256=implementation_digests,
                                        route=route, seconds=route_seconds, variants=variants))
            status["variants"] = variants
        unchanged = execution_identity(backend, route["output"], route["initial_mapping"], route["final_mapping"]) == \
            execution_identity(backend, original_route["output"], original_route["initial_mapping"], original_route["final_mapping"])
        provenance = dict(policy=POLICY, timing_basis="archived-complete-run-plus-measured-extension",
                          original_timing=old["timing"], archived_audit_sha256=audit_info["sha256"],
                          archived_routing_audit_sha256=routing_audit_sha256,
                          archived_catalog_sha256=old["routing_osr_catalog"]["sha256"],
                          routing_seconds=route_seconds, a2a_reused=True, catalogue_synthesis_reused=True,
                          route_bit_identical=unchanged, new_routing_synthesis_calls=0)
        provenance["implementation_sha256"] = routing_implementation_digests
        provenance["runtime_limits"] = dict(address_space_gib=96)
        entry = deepcopy(old)
        entry["routing_replay"] = provenance
        if unchanged:
            shutil.copy2(old_audit_path, canonical_audit)
            retain_prior_candidate(old, source_root, provenance)
            entry["timing"] = cumulative_timing(old["timing"], route_seconds, 0)
            provenance.update(final_reused=True, final_seconds=0.0)
            status.update(status="completed", shortcut="identical-gates-parameters-and-mappings", entry=entry)
            write_json(status_path, status)
            return
        # Release the large catalogue before topology synthesis. Route events
        # contain the complete independent local-program proof.
        del catalog, exact
        import gc
        gc.collect()
        status.update(status="final-synthesis", routed_cnots=route["cnot_count"])
        write_json(status_path, status)
        ledger = case_dir / "candidate.jsonl"
        raw, raw_parameters = decode(backend, route["output"])
        resumed = resume_final_prefix(case_dir, ledger, route, audit, backend)
        if resumed is None:
            optimization_input, optimization_parameters = raw, raw_parameters
            with ledger.open("w") as destination:
                for event in iter_audit_events(audit["events"]):
                    if event.get("stage") == "all_to_all":
                        destination.write(json.dumps(event, separators=(",", ":")) + "\n")
                destination.write(json.dumps(route, separators=(",", ":")) + "\n")
        else:
            optimization_input, optimization_parameters, provenance["final_resume"] = resumed
        config = deepcopy(old["configuration"])
        config.update(strategy="TreeSearch", topology=route["topology"],
                      **{"pre-opt-strategy": "TreeSearch", "routing-strategy": "exact-osr"},
                      test_final_circuit=False, test_subcircuits=False,
                      _rewrite_audit_stage="topology_optimization")
        if row["size"] == 4:
            config.setdefault("osr_adaptive_partition_order", False)
        optimizer = backend.qgd_Wide_Circuit_Optimization(config)
        os.environ["SQUANDER_REWRITE_AUDIT_JSONL"] = str(ledger)
        begin = time.monotonic()
        pending_attempt = case_dir / "final-attempt.json"
        write_json(pending_attempt, dict(status="running", started_monotonic=begin))
        provenance["final_implementation_sha256"] = {
            name: sha256(ROOT / name) for name in (
                "benchmarks/partitioning/replay_routing.py",
                "squander/partitioning/ilp.py",
                "squander/decomposition/qgd_Wide_Circuit_Optimization.py")}
        try:
            fitted, fitted_parameters = optimizer.OptimizeWideCircuit(
                optimization_input, optimization_parameters)
        finally:
            os.environ.pop("SQUANDER_REWRITE_AUDIT_JSONL", None)
        final_seconds = time.monotonic() - begin
        backend.qgd_Wide_Circuit_Optimization.check_valid_routing(fitted, route["topology"])
        candidate_stats = backend.CircuitGateStats(fitted)
        candidate_qasm = output.with_suffix(".replay-candidate.qasm")
        from benchmarks.partitioning.wide_circuit_optimization import save_qasm2
        save_qasm2(fitted, fitted_parameters, candidate_qasm)
        adopted = candidate_stats["cnot_equiv"] < old["final"]["cnot_equiv"]
        if adopted:
            shutil.copy2(candidate_qasm, output)
        representation = backend._squander_audit_representation(
            fitted, fitted_parameters, range(fitted.get_Qbit_Num()))
        run = deepcopy(audit["run"])
        run.update(output_file=str((output if adopted else candidate_qasm).relative_to(ROOT)),
                   output_circuit=representation,
                   output_circuit_sha256=backend._exact_state_sha256(backend._qasm_exact_state(representation)),
                   output_file_sha256=sha256(candidate_qasm),
                   initial_mapping=route["initial_mapping"], final_mapping=route["final_mapping"],
                   routing_replay=provenance, audit_progress=True)
        run.pop("result_checkpoint", None)
        candidate_audit = output.with_suffix(".replay-candidate.audit.json.gz")
        candidate_proof, digest = backend.save_rewrite_audit(ledger, candidate_audit, run)
        try:
            verification = backend.verify_rewrite_audit(candidate_proof, expected_sha256=digest,
                                                        audit_path=candidate_audit)
        except BaseException:
            shutil.copy2(original_output, output)
            raise
        finally:
            close_audit(candidate_proof)
        # The routing-stage table reports the best raw candidate. Final output
        # is selected from complete pipelines, including the old verified one.
        # Keep the candidate certificate even if its final count loses.
        entry["routed"] = backend.CircuitGateStats(raw)
        failed_seconds = sum(attempt["seconds"] for attempt in failed_attempts)
        entry["timing"] = cumulative_timing(old["timing"], route_seconds, final_seconds + failed_seconds)
        if failed_attempts:
            provenance.update(failed_final_attempts=failed_attempts,
                              failed_final_attempts_seconds_upper_bound=failed_seconds)
        provenance.update(final_reused=not adopted, final_seconds=final_seconds,
                          candidate_final=candidate_stats, candidate_adopted=adopted,
                          candidate_qasm=str(candidate_qasm.relative_to(ROOT)),
                          candidate_audit=dict(file=str(candidate_audit.relative_to(ROOT)), sha256=digest,
                                               **verification),
                          raw_initial_mapping=route["initial_mapping"], raw_final_mapping=route["final_mapping"])
        if adopted:
            # Keep the proof's canonical file binding valid by retaining the
            # candidate output as well as the conventional final-QASM copy.
            shutil.copy2(candidate_qasm, output)
            shutil.copy2(candidate_audit, canonical_audit)
            entry["final"] = candidate_stats
            entry["rewrite_audit"] = dict(schema_version=2, file=str(canonical_audit.relative_to(ROOT)),
                                           sha256=digest, summary=candidate_proof["summary"], **verification)
        else:
            shutil.copy2(old_audit_path, canonical_audit)
        status.update(status="completed", entry=entry)
        write_json(status_path, status)
        write_json(pending_attempt, dict(status="completed", seconds=final_seconds))
    finally:
        close_audit(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".routing-refresh")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.source_root.resolve() == ROOT:
        parser.error("Use a separate immutable source checkout.")
    if args.case is not None:
        try:
            worker(args)
        except BaseException:
            path = args.state_dir / f"case-{args.case:03d}" / "status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(status="failed", traceback=traceback.format_exc())
            write_json(path, status)
            raise
        return 0
    rows, incomplete = inventory(args.source_root)
    manifest = args.state_dir / "inventory.json"
    current = dict(policy=POLICY, completed=rows, unavailable=incomplete)
    if manifest.exists() and json.loads(manifest.read_text()) != current:
        raise RuntimeError("Source results changed since this refresh began.")
    write_json(manifest, current)
    queue = list(range(len(rows)))
    # Start expensive final-stage inputs first so short cases fill around them.
    queue.sort(key=lambda i: -(rows[i]["entry"].get("timing", {}).get("optimization") or 0))
    active = {}
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", QISKIT_PARALLEL="FALSE", RAYON_NUM_THREADS="1")
    def stop(*_args):
        for process, log in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        while queue or active:
            while queue and len(active) < args.jobs:
                index = queue.pop(0)
                path = args.state_dir / f"case-{index:03d}" / "status.json"
                if path.exists() and json.loads(path.read_text()).get("status") == "completed":
                    continue
                log = (args.state_dir / f"case-{index:03d}.log").open("a")
                command = [sys.executable, str(Path(__file__).resolve()), "--source-root", str(args.source_root),
                           "--state-dir", str(args.state_dir), "--case", str(index)]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                active[index] = process, log
                print("START", index, rows[index]["size"], rows[index]["file"], flush=True)
            for index, (process, log) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del active[index]
                    print("FINISH", index, code, rows[index]["file"], flush=True)
            states = {}
            for index in range(len(rows)):
                path = args.state_dir / f"case-{index:03d}" / "status.json"
                if path.exists():
                    states[index] = json.loads(path.read_text())
            write_json(args.state_dir / "progress.json", dict(
                completed=sum(s.get("status") == "completed" for s in states.values()),
                failed=[i for i, s in states.items() if s.get("status") == "failed"],
                active=list(active), queued=len(queue), total=len(rows)))
            time.sleep(1)
    finally:
        for process, log in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            log.close()
    states = [json.loads((args.state_dir / f"case-{i:03d}" / "status.json").read_text()) for i in range(len(rows))]
    if any(s["status"] != "completed" for s in states):
        return 1
    for directory in sorted({row["directory"] for row in rows}):
        results = json.loads((args.source_root / directory / "results.json").read_text())
        for row, state in zip(rows, states):
            if row["directory"] == directory:
                results[row["file"]] = state["entry"]
        write_json(ROOT / directory / "results.json", results)
    print("ALL COMPLETED; result metadata replaced atomically.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
