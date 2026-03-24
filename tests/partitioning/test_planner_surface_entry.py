import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.planner_surface.common import (
    build_phase2_continuity_vqe,
)
from benchmarks.density_matrix.planner_surface.workloads import (
    MANDATORY_NOISE_PATTERNS,
    STRUCTURED_FAMILY_NAMES,
    STRUCTURED_QUBITS,
    iter_microcase_surfaces,
    iter_structured_surfaces,
    mandatory_microcase_definitions,
)
from squander import Circuit
from squander.partitioning.noisy_planner import (
    PARTITIONED_DENSITY_MODE,
    PLANNER_OP_KIND_GATE,
    PLANNER_OP_KIND_NOISE,
    PHASE3_ENTRY_ROUTE_MICROCASE,
    PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY,
    PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY,
    PHASE3_WORKLOAD_FAMILY_MICROCASE,
    PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY,
    PHASE3_WORKLOAD_FAMILY_STRUCTURED,
    SUPPORTED_GATE_NAMES,
    SUPPORTED_NOISE_NAMES,
    NoisyPlannerValidationError,
    build_bridge_overlap_report,
    build_canonical_planner_surface_from_bridge_metadata,
    build_canonical_planner_surface_from_qgd_circuit,
    build_phase3_continuity_planner_surface,
    build_planner_audit_record,
    phase3_entry_route_for_source_type,
    phase3_workload_family_for_source_type,
    preflight_planner_request,
)


@pytest.mark.parametrize("qbit_num", [4, 6, 8, 10])
def test_continuity_surface_matches_bridge_metadata(qbit_num):
    vqe, _, _ = build_phase2_continuity_vqe(qbit_num)

    bridge = vqe.describe_density_bridge()
    surface = build_phase3_continuity_planner_surface(vqe)
    payload = surface.to_dict()

    assert surface.schema_version == "phase3_canonical_noisy_planner_v1"
    assert surface.requested_mode == PARTITIONED_DENSITY_MODE
    assert surface.source_type == "generated_hea"
    assert phase3_entry_route_for_source_type(surface.source_type) == (
        PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    )
    assert phase3_workload_family_for_source_type(surface.source_type) == (
        PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    )
    assert surface.workload_id == f"phase2_xxz_hea_q{qbit_num}_continuity"
    assert surface.qbit_num == qbit_num
    assert surface.parameter_count == bridge["parameter_count"]
    assert surface.operation_count == bridge["operation_count"]
    assert surface.gate_count == bridge["gate_count"]
    assert surface.noise_count == bridge["noise_count"]
    for actual, expected in zip(payload["operations"], bridge["operations"]):
        for key in (
            "index",
            "kind",
            "name",
            "is_unitary",
            "source_gate_index",
            "target_qbit",
            "control_qbit",
            "param_count",
            "param_start",
            "fixed_value",
        ):
            assert actual[key] == expected[key]
        assert actual["qubit_support"] == [
            q
            for q in (expected["control_qbit"], expected["target_qbit"])
            if q is not None
        ]
    assert payload["gate_count"] + payload["noise_count"] == payload["operation_count"]
    assert payload["operations"][0]["kind"] == PLANNER_OP_KIND_GATE
    assert any(
        op["kind"] == PLANNER_OP_KIND_NOISE for op in payload["operations"]
    )


def test_rejects_non_partitioned_density_mode():
    vqe, _, _ = build_phase2_continuity_vqe(4)

    with pytest.raises(
        NoisyPlannerValidationError,
        match="supports only 'partitioned_density' requests",
    ) as err:
        build_phase3_continuity_planner_surface(vqe, requested_mode="state_vector")

    assert err.value.category == "mode"
    assert err.value.first_unsupported_condition == "unsupported_mode"
    assert err.value.failure_stage == "planner_entry_preflight"


def test_rejects_bridge_noise_with_invalid_after_gate_index():
    vqe, _, _ = build_phase2_continuity_vqe(4)
    bridge = vqe.describe_density_bridge()
    bridge["operations"] = [
        {
            "index": 0,
            "kind": PLANNER_OP_KIND_NOISE,
            "name": "local_depolarizing",
            "is_unitary": False,
            "source_gate_index": 99,
            "target_qbit": 0,
            "control_qbit": None,
            "param_count": 0,
            "param_start": 0,
            "fixed_value": 0.1,
        }
    ]

    with pytest.raises(
        NoisyPlannerValidationError,
        match="references unsupported after_gate_index 99",
    ) as err:
        build_canonical_planner_surface_from_bridge_metadata(
            bridge,
            workload_id="broken_bridge_case",
        )

    assert err.value.category == "noise_insertion"
    assert err.value.first_unsupported_condition == "after_gate_index"


def test_microcases_share_canonical_schema():
    reference_keys = None
    operation_keys = None

    for metadata, surface in iter_microcase_surfaces():
        payload = surface.to_dict()

        if reference_keys is None:
            reference_keys = set(payload.keys())
            operation_keys = set(payload["operations"][0].keys())
        else:
            assert set(payload.keys()) == reference_keys
            assert set(payload["operations"][0].keys()) == operation_keys

        assert payload["requested_mode"] == PARTITIONED_DENSITY_MODE
        assert payload["source_type"] == "microcase_builder"
        assert phase3_entry_route_for_source_type(payload["source_type"]) == (
            PHASE3_ENTRY_ROUTE_MICROCASE
        )
        assert phase3_workload_family_for_source_type(payload["source_type"]) == (
            PHASE3_WORKLOAD_FAMILY_MICROCASE
        )
        assert payload["workload_id"] == metadata["case_name"]
        assert payload["qbit_num"] == metadata["qbit_num"]
        assert payload["gate_count"] > 0
        assert payload["noise_count"] > 0
        assert set(surface.gate_names).issubset(SUPPORTED_GATE_NAMES)
        assert set(surface.noise_names).issubset(SUPPORTED_NOISE_NAMES)


def test_structured_families_share_canonical_schema():
    structured_cases = list(iter_structured_surfaces())
    assert len(structured_cases) == (
        len(STRUCTURED_FAMILY_NAMES)
        * len(STRUCTURED_QUBITS)
        * len(MANDATORY_NOISE_PATTERNS)
    )

    for metadata, surface in structured_cases:
        payload = surface.to_dict()

        assert payload["requested_mode"] == PARTITIONED_DENSITY_MODE
        assert payload["source_type"] == "structured_family_builder"
        assert phase3_entry_route_for_source_type(payload["source_type"]) == (
            PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY
        )
        assert phase3_workload_family_for_source_type(payload["source_type"]) == (
            PHASE3_WORKLOAD_FAMILY_STRUCTURED
        )
        assert payload["workload_id"] == metadata["workload_id"]
        assert payload["qbit_num"] == metadata["qbit_num"]
        assert payload["gate_count"] > 0
        assert payload["noise_count"] > 0
        assert set(surface.gate_names).issubset(SUPPORTED_GATE_NAMES)
        assert set(surface.noise_names).issubset(SUPPORTED_NOISE_NAMES)


def test_workload_ids_are_unique():
    workload_ids = set()

    for metadata, surface in iter_microcase_surfaces():
        assert surface.workload_id == metadata["case_name"]
        assert surface.workload_id not in workload_ids
        workload_ids.add(surface.workload_id)

    for metadata, surface in iter_structured_surfaces():
        assert surface.workload_id == metadata["workload_id"]
        assert surface.workload_id not in workload_ids
        workload_ids.add(surface.workload_id)


def test_audit_record_tracks_continuity_provenance():
    vqe, _, _ = build_phase2_continuity_vqe(4)
    surface = build_phase3_continuity_planner_surface(vqe)
    audit = build_planner_audit_record(surface, metadata={"case_kind": "continuity"})
    overlap = build_bridge_overlap_report(surface, vqe.describe_density_bridge())

    assert audit["provenance"]["source_type"] == "generated_hea"
    assert phase3_entry_route_for_source_type(audit["provenance"]["source_type"]) == (
        PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    )
    assert phase3_workload_family_for_source_type(
        audit["provenance"]["source_type"]
    ) == PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    assert audit["summary"]["operation_count"] == surface.operation_count
    assert audit["summary"]["max_qubit_span"] >= 1
    assert audit["operations"][0]["qubit_support"]
    assert audit["metadata"]["case_kind"] == "continuity"
    assert overlap["bridge_overlap_pass"]


def test_audit_record_tracks_structured_workload_provenance():
    metadata, surface = next(iter(iter_structured_surfaces()))
    audit = build_planner_audit_record(surface, metadata=metadata)

    assert audit["provenance"]["source_type"] == "structured_family_builder"
    assert phase3_entry_route_for_source_type(audit["provenance"]["source_type"]) == (
        PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY
    )
    assert phase3_workload_family_for_source_type(
        audit["provenance"]["source_type"]
    ) == PHASE3_WORKLOAD_FAMILY_STRUCTURED
    assert audit["provenance"]["workload_id"] == metadata["workload_id"]
    assert audit["summary"]["gate_sequence"]
    assert audit["summary"]["noise_sequence"]
    assert audit["summary"]["max_qubit_span"] >= 1


def test_exact_qgd_circuit_lowering_preserves_order_and_param_spans():
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_U3(1)
    circuit.add_CNOT(1, 0)
    circuit.add_U3(0)

    surface = build_canonical_planner_surface_from_qgd_circuit(
        circuit,
        workload_id="legacy_manual_u3_cnot",
    )
    gate_payloads = [
        op
        for op in surface.to_dict()["operations"]
        if op["kind"] == PLANNER_OP_KIND_GATE
    ]
    source_gates = circuit.get_Gates()

    assert surface.source_type == "legacy_qgd_circuit_exact"
    assert phase3_entry_route_for_source_type(surface.source_type) == (
        "phase3_legacy_exact_lowering"
    )
    assert phase3_workload_family_for_source_type(surface.source_type) == (
        "phase3_legacy_exact_lowering"
    )
    assert [op["name"] for op in gate_payloads] == ["U3", "U3", "CNOT", "U3"]
    assert [op["param_start"] for op in gate_payloads] == [
        gate.get_Parameter_Start_Index() for gate in source_gates
    ]
    assert [op["param_count"] for op in gate_payloads] == [
        gate.get_Parameter_Num() for gate in source_gates
    ]


def test_legacy_lowering_accepts_explicit_supported_noise():
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)

    surface = build_canonical_planner_surface_from_qgd_circuit(
        circuit,
        workload_id="legacy_manual_with_noise",
        density_noise=[
            {
                "channel": "local_depolarizing",
                "target": 1,
                "after_gate_index": 1,
                "error_rate": 0.1,
            }
        ],
    )
    payload = surface.to_dict()

    assert payload["noise_count"] == 1
    assert payload["operations"][2]["kind"] == PLANNER_OP_KIND_NOISE
    assert payload["operations"][2]["name"] == "local_depolarizing"
    assert payload["operations"][2]["source_gate_index"] == 1
    assert payload["operations"][2]["fixed_value"] == pytest.approx(0.1)


def test_rejects_unsupported_legacy_gate_family():
    circuit = Circuit(1)
    circuit.add_H(0)

    with pytest.raises(
        NoisyPlannerValidationError,
        match="Unsupported Phase 3 planner gate 'H' in legacy-source lowering",
    ) as err:
        build_canonical_planner_surface_from_qgd_circuit(
            circuit,
            workload_id="legacy_manual_with_h",
        )

    assert err.value.category == "gate_family"
    assert err.value.first_unsupported_condition == "H"


def test_rejects_invalid_legacy_noise_schedule():
    circuit = Circuit(2)
    circuit.add_U3(0)
    circuit.add_CNOT(1, 0)

    with pytest.raises(
        NoisyPlannerValidationError,
        match="unsupported after_gate_index 99",
    ) as err:
        build_canonical_planner_surface_from_qgd_circuit(
            circuit,
            workload_id="legacy_manual_bad_noise_index",
            density_noise=[
                {
                    "channel": "phase_damping",
                    "target": 0,
                    "after_gate_index": 99,
                    "lambda": 0.07,
                }
            ],
        )

    assert err.value.category == "noise_insertion"


def test_preflight_accepts_supported_bridge_request():
    vqe, _, _ = build_phase2_continuity_vqe(4)
    bridge = vqe.describe_density_bridge()
    surface = preflight_planner_request(
        source_type="generated_hea",
        workload_id="phase2_xxz_hea_q4_continuity",
        bridge_metadata=bridge,
    )

    assert surface.requested_mode == PARTITIONED_DENSITY_MODE
    assert phase3_entry_route_for_source_type(surface.source_type) == (
        PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY
    )
    assert phase3_workload_family_for_source_type(surface.source_type) == (
        PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY
    )


def test_preflight_rejects_unsupported_source_type():
    with pytest.raises(
        NoisyPlannerValidationError,
        match="Unsupported Phase 3 planner source type 'binary_import'",
    ) as err:
        preflight_planner_request(
            source_type="binary_import",
            workload_id="unsupported_source_case",
            operation_specs=[],
            qbit_num=2,
        )

    assert err.value.category == "source_type"
    assert err.value.first_unsupported_condition == "binary_import"


def test_preflight_rejects_missing_source_payload():
    with pytest.raises(
        NoisyPlannerValidationError,
        match="requires exactly one source payload",
    ) as err:
        preflight_planner_request(
            source_type="microcase_builder",
            workload_id="missing_payload_case",
            qbit_num=2,
        )

    assert err.value.category == "malformed_request"
    assert err.value.first_unsupported_condition == "missing_source_payload"


def test_preflight_rejects_unsupported_noise_model_in_operation_specs():
    microcase = mandatory_microcase_definitions()[0]
    bad_specs = list(microcase["operation_specs"])
    bad_specs.append(
        {
            "kind": "noise",
            "name": "readout_noise",
            "target_qbit": 0,
            "source_gate_index": 2,
            "fixed_value": 0.1,
            "param_count": 0,
        }
    )

    with pytest.raises(
        NoisyPlannerValidationError,
        match="Unsupported Phase 3 planner noise model 'readout_noise'",
    ) as err:
        preflight_planner_request(
            source_type="microcase_builder",
            workload_id="unsupported_noise_case",
            operation_specs=bad_specs,
            qbit_num=microcase["qbit_num"],
        )

    assert err.value.category == "noise_type"


def test_preflight_rejects_disallowed_mode_claim():
    microcase = mandatory_microcase_definitions()[0]
    with pytest.raises(
        NoisyPlannerValidationError,
        match="supports only 'partitioned_density' requests",
    ) as err:
        preflight_planner_request(
            source_type="microcase_builder",
            workload_id="wrong_mode_case",
            requested_mode="state_vector",
            operation_specs=microcase["operation_specs"],
            qbit_num=microcase["qbit_num"],
        )

    assert err.value.category == "mode"
