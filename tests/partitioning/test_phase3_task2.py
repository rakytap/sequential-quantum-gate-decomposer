from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from squander import Circuit
from benchmarks.density_matrix.planner_surface.common import (
    build_phase3_story1_continuity_vqe,
)
from squander.partitioning.noisy_planner import (
    DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
    PARTITIONED_DENSITY_MODE,
    PHASE3_DESCRIPTOR_SCHEMA_VERSION,
    PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY,
    PHASE3_ENTRY_ROUTE_MICROCASE,
    PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY,
    PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY,
    PHASE3_WORKLOAD_FAMILY_MICROCASE,
    PHASE3_WORKLOAD_FAMILY_STRUCTURED,
    build_canonical_planner_surface_from_qgd_circuit,
    build_descriptor_audit_record,
    build_partition_descriptor_set,
    build_phase3_continuity_partition_descriptor_set,
    build_phase3_continuity_planner_surface,
    preflight_descriptor_request,
    validate_partition_descriptor_set,
    validate_partition_descriptor_set_against_surface,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    build_story2_microcase_descriptor_set,
    build_story2_microcase_surface,
    build_story2_structured_surface,
    iter_story2_microcase_descriptor_sets,
    iter_story2_structured_descriptor_sets,
)
from benchmarks.density_matrix.planner_surface.unsupported_descriptor_validation import (
    build_cases as build_story6_cases,
)


def _flatten_canonical_indices(payload: dict) -> list[int]:
    return [
        member["canonical_operation_index"]
        for partition in payload["partitions"]
        for member in partition["members"]
    ]


def _flatten_member_names(payload: dict) -> list[str]:
    return [
        member["name"] for partition in payload["partitions"] for member in partition["members"]
    ]


def _round_trip_partition_support(partition: dict) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    local_to_global = {
        local_qbit: global_qbit
        for local_qbit, global_qbit in enumerate(partition["local_to_global_qbits"])
    }
    return [
        (
            tuple(member["qubit_support"]),
            tuple(local_to_global[local_qbit] for local_qbit in member["local_qubit_support"]),
        )
        for member in partition["members"]
    ]


def _expected_partition_parameter_routing(partition: dict) -> list[dict]:
    return [
        {
            "global_param_start": member["param_start"],
            "local_param_start": member["local_param_start"],
            "param_count": member["param_count"],
        }
        for member in partition["members"]
        if member["param_count"] > 0
    ]


@pytest.mark.parametrize("qbit_num", [4, 6, 8, 10])
def test_phase3_task2_story1_continuity_descriptor_slice_matches_continuity_surface(qbit_num):
    vqe, _, _ = build_phase3_story1_continuity_vqe(qbit_num)
    surface = build_phase3_continuity_planner_surface(vqe)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    payload = descriptor_set.to_dict()

    assert descriptor_set.schema_version == PHASE3_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor_set.planner_schema_version == surface.schema_version
    assert descriptor_set.requested_mode == PARTITIONED_DENSITY_MODE
    assert descriptor_set.source_type == "generated_hea"
    assert descriptor_set.entry_route == PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    assert descriptor_set.workload_family == PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    assert descriptor_set.workload_id == f"phase2_xxz_hea_q{qbit_num}_continuity"
    assert descriptor_set.qbit_num == qbit_num
    assert descriptor_set.parameter_count == surface.parameter_count
    assert payload["partition_count"] > 0
    assert payload["descriptor_member_count"] == surface.operation_count
    assert payload["gate_count"] == surface.gate_count
    assert payload["noise_count"] == surface.noise_count
    assert payload["partition_count"] == len(payload["partitions"])
    assert payload["partition_member_counts"] == [
        len(partition["members"]) for partition in payload["partitions"]
    ]
    assert _flatten_canonical_indices(payload) == list(range(surface.operation_count))

    for expected_partition_index, partition in enumerate(payload["partitions"]):
        assert partition["partition_index"] == expected_partition_index
        assert partition["canonical_operation_indices"] == [
            member["canonical_operation_index"] for member in partition["members"]
        ]
        assert partition["partition_qubit_span"] == partition["local_to_global_qbits"]
        assert len(partition["partition_qubit_span"]) <= DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS


def test_phase3_task2_story1_continuity_descriptor_emission_is_deterministic():
    vqe, _, _ = build_phase3_story1_continuity_vqe(4)

    first_payload = build_phase3_continuity_partition_descriptor_set(vqe).to_dict()
    second_payload = build_phase3_continuity_partition_descriptor_set(vqe).to_dict()

    assert first_payload == second_payload


def test_phase3_task2_story2_microcase_descriptors_share_schema():
    reference_keys = None
    partition_keys = None
    member_keys = None

    for metadata, descriptor_set in iter_story2_microcase_descriptor_sets():
        payload = descriptor_set.to_dict()

        if reference_keys is None:
            reference_keys = set(payload.keys())
            partition_keys = set(payload["partitions"][0].keys())
            member_keys = set(payload["partitions"][0]["members"][0].keys())
        else:
            assert set(payload.keys()) == reference_keys
            assert set(payload["partitions"][0].keys()) == partition_keys
            assert set(payload["partitions"][0]["members"][0].keys()) == member_keys

        assert payload["schema_version"] == PHASE3_DESCRIPTOR_SCHEMA_VERSION
        assert payload["requested_mode"] == PARTITIONED_DENSITY_MODE
        assert payload["source_type"] == "microcase_builder"
        assert payload["entry_route"] == PHASE3_ENTRY_ROUTE_MICROCASE
        assert payload["workload_family"] == PHASE3_WORKLOAD_FAMILY_MICROCASE
        assert payload["workload_id"] == metadata["case_name"]
        assert payload["qbit_num"] == metadata["qbit_num"]
        assert payload["partition_count"] > 0
        assert payload["descriptor_member_count"] == payload["gate_count"] + payload["noise_count"]
        assert payload["max_partition_qubits"] == DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS


def test_phase3_task2_story2_structured_descriptors_share_schema():
    structured_cases = list(iter_story2_structured_descriptor_sets())
    assert len(structured_cases) == (
        len(STRUCTURED_FAMILY_NAMES) * len(STRUCTURED_QUBITS) * len(MANDATORY_NOISE_PATTERNS)
    )

    for metadata, descriptor_set in structured_cases:
        payload = descriptor_set.to_dict()

        assert payload["schema_version"] == PHASE3_DESCRIPTOR_SCHEMA_VERSION
        assert payload["requested_mode"] == PARTITIONED_DENSITY_MODE
        assert payload["source_type"] == "structured_family_builder"
        assert payload["entry_route"] == PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY
        assert payload["workload_family"] == PHASE3_WORKLOAD_FAMILY_STRUCTURED
        assert payload["workload_id"] == metadata["workload_id"]
        assert payload["qbit_num"] == metadata["qbit_num"]
        assert payload["partition_count"] > 0
        assert payload["descriptor_member_count"] == payload["gate_count"] + payload["noise_count"]
        assert payload["max_partition_qubits"] == DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS


def test_phase3_task2_story2_continuity_and_methods_share_descriptor_schema():
    continuity_vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    continuity_payload = build_phase3_continuity_partition_descriptor_set(
        continuity_vqe
    ).to_dict()
    _, microcase_descriptor_set = next(iter(iter_story2_microcase_descriptor_sets()))
    _, structured_descriptor_set = next(iter(iter_story2_structured_descriptor_sets()))

    payloads = (
        continuity_payload,
        microcase_descriptor_set.to_dict(),
        structured_descriptor_set.to_dict(),
    )

    top_level_keys = {frozenset(payload.keys()) for payload in payloads}
    partition_keys = {
        frozenset(payload["partitions"][0].keys()) for payload in payloads if payload["partitions"]
    }
    member_keys = {
        frozenset(payload["partitions"][0]["members"][0].keys())
        for payload in payloads
        if payload["partitions"] and payload["partitions"][0]["members"]
    }

    assert len(top_level_keys) == 1
    assert len(partition_keys) == 1
    assert len(member_keys) == 1


def test_phase3_task2_story2_workload_ids_are_unique():
    workload_ids = set()

    continuity_vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    continuity_payload = build_phase3_continuity_partition_descriptor_set(
        continuity_vqe
    ).to_dict()
    workload_ids.add(continuity_payload["workload_id"])

    for metadata, descriptor_set in iter_story2_microcase_descriptor_sets():
        assert descriptor_set.workload_id == metadata["case_name"]
        assert descriptor_set.workload_id not in workload_ids
        workload_ids.add(descriptor_set.workload_id)

    for metadata, descriptor_set in iter_story2_structured_descriptor_sets():
        assert descriptor_set.workload_id == metadata["workload_id"]
        assert descriptor_set.workload_id not in workload_ids
        workload_ids.add(descriptor_set.workload_id)


def test_phase3_task2_story3_continuity_descriptor_members_match_surface_order():
    vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    surface = build_phase3_continuity_planner_surface(vqe)
    descriptor_payload = build_phase3_continuity_partition_descriptor_set(vqe).to_dict()
    surface_payload = surface.to_dict()

    expected_names = [operation["name"] for operation in surface_payload["operations"]]
    actual_names = _flatten_member_names(descriptor_payload)

    assert _flatten_canonical_indices(descriptor_payload) == list(
        range(surface_payload["operation_count"])
    )
    assert actual_names == expected_names
    assert descriptor_payload["noise_count"] == sum(
        member["operation_class"] == "NoiseOperation"
        for partition in descriptor_payload["partitions"]
        for member in partition["members"]
    )


def test_phase3_task2_story3_boundary_microcase_keeps_noise_explicit_inside_partitions():
    surface = build_story2_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_payload = build_story2_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    ).to_dict()
    surface_payload = surface.to_dict()

    expected_names = [operation["name"] for operation in surface_payload["operations"]]
    actual_names = _flatten_member_names(descriptor_payload)

    assert _flatten_canonical_indices(descriptor_payload) == list(
        range(surface_payload["operation_count"])
    )
    assert actual_names == expected_names
    assert any(
        partition["gate_count"] > 0 and partition["noise_count"] > 0
        for partition in descriptor_payload["partitions"]
    )
    assert sum(partition["noise_count"] for partition in descriptor_payload["partitions"]) == surface_payload[
        "noise_count"
    ]


def test_phase3_task2_story3_structured_descriptor_members_match_surface_order():
    metadata, descriptor_set = next(iter(iter_story2_structured_descriptor_sets()))
    surface = build_story2_structured_surface(
        metadata["family_name"],
        qbit_num=metadata["qbit_num"],
        noise_pattern=metadata["noise_pattern"],
        seed=metadata["seed"],
    )
    descriptor_payload = descriptor_set.to_dict()
    surface_payload = surface.to_dict()

    assert _flatten_canonical_indices(descriptor_payload) == list(
        range(surface_payload["operation_count"])
    )
    assert _flatten_member_names(descriptor_payload) == [
        operation["name"] for operation in surface_payload["operations"]
    ]


def test_phase3_task2_story4_continuity_descriptors_validate_and_round_trip_support():
    vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    payload = validate_partition_descriptor_set(descriptor_set).to_dict()

    assert any(partition["requires_remap"] for partition in payload["partitions"])
    for partition in payload["partitions"]:
        for expected_global_support, reconstructed_global_support in _round_trip_partition_support(
            partition
        ):
            assert expected_global_support == reconstructed_global_support


def test_phase3_task2_story4_boundary_microcase_parameter_routing_round_trip():
    descriptor_set = build_story2_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )
    payload = validate_partition_descriptor_set(descriptor_set).to_dict()

    assert any(partition["parameter_routing"] for partition in payload["partitions"])
    for partition in payload["partitions"]:
        assert partition["parameter_routing"] == _expected_partition_parameter_routing(
            partition
        )


def test_phase3_task2_story4_structured_descriptor_exposes_nontrivial_remap():
    metadata, descriptor_set = next(iter(iter_story2_structured_descriptor_sets()))
    payload = validate_partition_descriptor_set(descriptor_set).to_dict()

    assert payload["workload_id"] == metadata["workload_id"]
    assert any(partition["requires_remap"] for partition in payload["partitions"])
    assert any(
        len(partition["partition_qubit_span"]) == DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS
        for partition in payload["partitions"]
    )


def test_phase3_task2_story5_audit_record_tracks_continuity_provenance():
    vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
    audit = build_descriptor_audit_record(
        descriptor_set, metadata={"case_kind": "continuity"}
    )

    assert audit["schema_version"] == PHASE3_DESCRIPTOR_SCHEMA_VERSION
    assert audit["provenance"]["source_type"] == "generated_hea"
    assert audit["provenance"]["entry_route"] == PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    assert audit["provenance"]["workload_family"] == PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    assert audit["summary"]["partition_count"] == descriptor_set.partition_count
    assert audit["summary"]["descriptor_member_count"] == descriptor_set.descriptor_member_count
    assert audit["metadata"]["case_kind"] == "continuity"


def test_phase3_task2_story5_audit_record_tracks_methods_workload_provenance():
    metadata, descriptor_set = next(iter(iter_story2_structured_descriptor_sets()))
    audit = build_descriptor_audit_record(descriptor_set, metadata=metadata)

    assert audit["provenance"]["source_type"] == "structured_family_builder"
    assert audit["provenance"]["entry_route"] == PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY
    assert audit["provenance"]["workload_family"] == PHASE3_WORKLOAD_FAMILY_STRUCTURED
    assert audit["provenance"]["workload_id"] == metadata["workload_id"]
    assert audit["summary"]["partition_count"] > 0
    assert audit["summary"]["parameter_routing_segment_count"] > 0


def test_phase3_task2_story5_audit_record_tracks_legacy_exact_provenance():
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)

    descriptor_set = build_partition_descriptor_set(
        build_canonical_planner_surface_from_qgd_circuit(
            circuit, workload_id="legacy_descriptor_audit"
        )
    )
    audit = build_descriptor_audit_record(
        descriptor_set, metadata={"case_kind": "legacy_exact"}
    )

    assert audit["provenance"]["source_type"] == "legacy_qgd_circuit_exact"
    assert audit["provenance"]["workload_id"] == "legacy_descriptor_audit"
    assert audit["summary"]["partition_count"] > 0
    assert audit["metadata"]["case_kind"] == "legacy_exact"


def test_phase3_task2_story5_supported_cases_share_audit_shape():
    continuity_vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    continuity_audit = build_descriptor_audit_record(
        build_phase3_continuity_partition_descriptor_set(continuity_vqe),
        metadata={"case_kind": "continuity"},
    )
    _, microcase_descriptor_set = next(iter(iter_story2_microcase_descriptor_sets()))
    microcase_audit = build_descriptor_audit_record(
        microcase_descriptor_set, metadata={"case_kind": "microcase"}
    )
    _, structured_descriptor_set = next(iter(iter_story2_structured_descriptor_sets()))
    structured_audit = build_descriptor_audit_record(
        structured_descriptor_set, metadata={"case_kind": "structured_family"}
    )

    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)
    legacy_audit = build_descriptor_audit_record(
        build_partition_descriptor_set(
            build_canonical_planner_surface_from_qgd_circuit(
                circuit, workload_id="legacy_descriptor_shape"
            )
        ),
        metadata={"case_kind": "legacy_exact"},
    )

    audits = (continuity_audit, microcase_audit, structured_audit, legacy_audit)
    top_level_keys = {frozenset(audit.keys()) for audit in audits}
    summary_keys = {frozenset(audit["summary"].keys()) for audit in audits}
    partition_keys = {
        frozenset(audit["partitions"][0].keys())
        for audit in audits
        if audit["partitions"]
    }

    assert len(top_level_keys) == 1
    assert len(summary_keys) == 1
    assert len(partition_keys) == 1


def test_phase3_task2_story6_preflight_accepts_supported_continuity_request():
    vqe, _, _ = build_phase3_story1_continuity_vqe(4)
    bridge = vqe.describe_density_bridge()

    descriptor_set = preflight_descriptor_request(
        source_type="generated_hea",
        workload_id="phase2_xxz_hea_q4_continuity",
        bridge_metadata=bridge,
        entry_route=PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY,
        workload_family=PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY,
    )

    assert descriptor_set.requested_mode == PARTITIONED_DENSITY_MODE
    assert descriptor_set.partition_count > 0
    assert descriptor_set.workload_id == "phase2_xxz_hea_q4_continuity"


def test_phase3_task2_story6_validate_against_surface_accepts_supported_descriptor():
    surface = build_story2_microcase_surface("microcase_4q_partition_boundary_triplet")
    descriptor_set = build_story2_microcase_descriptor_set(
        "microcase_4q_partition_boundary_triplet"
    )

    validated = validate_partition_descriptor_set_against_surface(surface, descriptor_set)

    assert validated.workload_id == surface.workload_id
    assert validated.partition_count == descriptor_set.partition_count


def test_phase3_task2_story6_negative_matrix_has_expected_categories_and_no_fallback():
    cases = build_story6_cases()
    expected_categories = {
        "partition_qubits_too_small": "partition_span",
        "dropped_operation": "dropped_operations",
        "hidden_noise_placement": "hidden_noise_placement",
        "incomplete_remapping": "incomplete_remapping",
        "ambiguous_parameter_routing": "ambiguous_parameter_routing",
        "reordering_across_noise_boundaries": "reordering_across_noise_boundaries",
    }

    assert {case["case_name"] for case in cases} == set(expected_categories)
    for case in cases:
        assert case["status"] == "unsupported"
        assert case["unsupported_category"] == expected_categories[case["case_name"]]
        assert case["fallback_used"] is False
        assert case["supported_descriptor_case_recorded"] is False
