from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


CANONICAL_PLANNER_SCHEMA_VERSION = "phase3_canonical_noisy_planner_v1"
PARTITIONED_DENSITY_MODE = "partitioned_density"

PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY = "phase2_continuity_lowering"
PHASE3_ENTRY_ROUTE_MICROCASE = "phase3_microcase_generation"
PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY = "phase3_structured_family_generation"
PHASE3_ENTRY_ROUTE_LEGACY_EXACT = "phase3_legacy_exact_lowering"

PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY = "phase2_continuity_workflow"
PHASE3_WORKLOAD_FAMILY_MICROCASE = "phase3_micro_validation"
PHASE3_WORKLOAD_FAMILY_STRUCTURED = "phase3_structured_family"
PHASE3_WORKLOAD_FAMILY_LEGACY = "phase3_legacy_exact_lowering"

SUPPORTED_PHASE3_GATE_NAMES = frozenset({"U3", "CNOT"})
SUPPORTED_PHASE3_NOISE_NAMES = frozenset(
    {"local_depolarizing", "amplitude_damping", "phase_damping"}
)
SUPPORTED_PLANNER_SOURCE_TYPES = frozenset(
    {"generated_hea", "microcase_builder", "structured_family_builder", "legacy_qgd_circuit_exact"}
)


class NoisyPlannerValidationError(ValueError):
    """Structured planner-entry validation error for Phase 3 Task 1."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.first_unsupported_condition = first_unsupported_condition
        self.failure_stage = failure_stage
        self.source_type = source_type
        self.requested_mode = requested_mode
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "unsupported_category": self.category,
            "first_unsupported_condition": self.first_unsupported_condition,
            "failure_stage": self.failure_stage,
            "source_type": self.source_type,
            "requested_mode": self.requested_mode,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CanonicalNoisyPlannerOperation:
    index: int
    operation_class: str
    kind: str
    name: str
    is_unitary: bool
    source_gate_index: int
    target_qbit: int | None
    control_qbit: int | None
    param_count: int
    param_start: int
    fixed_value: float | None

    @property
    def qubit_support(self) -> tuple[int, ...]:
        support: list[int] = []
        if self.control_qbit is not None:
            support.append(self.control_qbit)
        if self.target_qbit is not None:
            support.append(self.target_qbit)
        return tuple(support)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "operation_class": self.operation_class,
            "kind": self.kind,
            "name": self.name,
            "is_unitary": self.is_unitary,
            "source_gate_index": self.source_gate_index,
            "target_qbit": self.target_qbit,
            "control_qbit": self.control_qbit,
            "qubit_support": list(self.qubit_support),
            "param_count": self.param_count,
            "param_start": self.param_start,
            "fixed_value": self.fixed_value,
        }


@dataclass(frozen=True)
class CanonicalNoisyPlannerSurface:
    schema_version: str
    requested_mode: str
    source_type: str
    entry_route: str
    workload_family: str
    workload_id: str
    qbit_num: int
    parameter_count: int
    operations: tuple[CanonicalNoisyPlannerOperation, ...]

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    @property
    def gate_count(self) -> int:
        return sum(op.operation_class == "GateOperation" for op in self.operations)

    @property
    def noise_count(self) -> int:
        return sum(op.operation_class == "NoiseOperation" for op in self.operations)

    @property
    def gate_names(self) -> tuple[str, ...]:
        return tuple(
            op.name for op in self.operations if op.operation_class == "GateOperation"
        )

    @property
    def noise_names(self) -> tuple[str, ...]:
        return tuple(
            op.name for op in self.operations if op.operation_class == "NoiseOperation"
        )

    @property
    def max_qubit_span(self) -> int:
        return max((len(op.qubit_support) for op in self.operations), default=0)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "entry_route": self.entry_route,
            "workload_family": self.workload_family,
            "workload_id": self.workload_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_mode": self.requested_mode,
            "source_type": self.source_type,
            "entry_route": self.entry_route,
            "workload_family": self.workload_family,
            "workload_id": self.workload_id,
            "provenance": self.provenance,
            "qbit_num": self.qbit_num,
            "parameter_count": self.parameter_count,
            "operation_count": self.operation_count,
            "gate_count": self.gate_count,
            "noise_count": self.noise_count,
            "gate_sequence": list(self.gate_names),
            "noise_sequence": list(self.noise_names),
            "max_qubit_span": self.max_qubit_span,
            "operations": [op.to_dict() for op in self.operations],
        }


def _normalize_gate_name(name: str) -> str:
    normalized = name.strip()
    aliases = {
        "u": "U3",
        "u3": "U3",
        "cx": "CNOT",
        "cnot": "CNOT",
    }
    return aliases.get(normalized.lower(), normalized.upper())


def _normalize_noise_name(name: str) -> str:
    normalized = name.strip()
    aliases = {
        "localdepolarizing": "local_depolarizing",
        "local_depolarizing": "local_depolarizing",
        "depolarizing": "depolarizing",
        "amplitudedamping": "amplitude_damping",
        "amplitude_damping": "amplitude_damping",
        "phasedamping": "phase_damping",
        "phase_damping": "phase_damping",
        "dephasing": "phase_damping",
    }
    return aliases.get(normalized.replace(" ", "").lower(), normalized.lower())


def _canonicalize_operation_name(name: str, operation_class: str) -> str:
    if operation_class == "GateOperation":
        return _normalize_gate_name(name)
    if operation_class == "NoiseOperation":
        return _normalize_noise_name(name)
    return name


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_operation_from_mapping(
    payload: Mapping[str, Any], *, fallback_index: int
) -> CanonicalNoisyPlannerOperation:
    operation_class = str(payload["operation_class"])
    kind = str(payload["kind"])
    is_unitary = bool(payload["is_unitary"])

    if operation_class == "GateOperation" and kind != "gate":
        raise ValueError(
            "GateOperation planner payload must declare kind='gate', got '{}'".format(
                kind
            )
        )
    if operation_class == "NoiseOperation" and kind != "noise":
        raise ValueError(
            "NoiseOperation planner payload must declare kind='noise', got '{}'".format(
                kind
            )
        )

    return CanonicalNoisyPlannerOperation(
        index=int(payload.get("index", fallback_index)),
        operation_class=operation_class,
        kind=kind,
        name=_canonicalize_operation_name(str(payload["name"]), operation_class),
        is_unitary=is_unitary,
        source_gate_index=int(payload["source_gate_index"]),
        target_qbit=_coerce_optional_int(payload.get("target_qbit")),
        control_qbit=_coerce_optional_int(payload.get("control_qbit")),
        param_count=int(payload["param_count"]),
        param_start=int(payload["param_start"]),
        fixed_value=_coerce_optional_float(payload.get("fixed_value")),
    )


def _build_operation_from_spec(
    payload: Mapping[str, Any],
    *,
    index: int,
    default_param_start: int,
    gate_index: int,
) -> CanonicalNoisyPlannerOperation:
    kind = str(payload["kind"])
    operation_class = "GateOperation" if kind == "gate" else "NoiseOperation"
    name = _canonicalize_operation_name(str(payload["name"]), operation_class)
    param_count = int(payload.get("param_count", 0))
    source_gate_index = int(payload.get("source_gate_index", gate_index))

    return CanonicalNoisyPlannerOperation(
        index=index,
        operation_class=operation_class,
        kind=kind,
        name=name,
        is_unitary=bool(payload.get("is_unitary", kind == "gate")),
        source_gate_index=source_gate_index,
        target_qbit=_coerce_optional_int(payload.get("target_qbit")),
        control_qbit=_coerce_optional_int(payload.get("control_qbit")),
        param_count=param_count,
        param_start=int(payload.get("param_start", default_param_start)),
        fixed_value=_coerce_optional_float(payload.get("fixed_value")),
    )


def _validate_mode(requested_mode: str, *, source_type: str) -> None:
    if requested_mode != PARTITIONED_DENSITY_MODE:
        raise NoisyPlannerValidationError(
            category="mode",
            first_unsupported_condition="unsupported_mode",
            failure_stage="planner_entry_preflight",
            source_type=source_type,
            requested_mode=requested_mode,
            reason=(
                "Phase 3 canonical planner surface supports only '{}' requests, got "
                "'{}'".format(PARTITIONED_DENSITY_MODE, requested_mode)
            ),
        )


def _validate_surface(
    surface: CanonicalNoisyPlannerSurface,
    *,
    strict_phase3_support: bool,
) -> CanonicalNoisyPlannerSurface:
    if surface.qbit_num <= 0:
        raise ValueError("Canonical planner surface requires qbit_num > 0")
    if surface.parameter_count < 0:
        raise ValueError("Canonical planner surface requires parameter_count >= 0")
    if not surface.operations:
        raise ValueError("Canonical planner surface requires at least one operation")

    expected_index = 0
    gate_count = 0
    seen_param_starts: list[int] = []
    for operation in surface.operations:
        if operation.index != expected_index:
            raise ValueError(
                "Canonical planner operations must use contiguous indices starting at 0"
            )
        if operation.param_start < 0 or operation.param_count < 0:
            raise ValueError("Planner operation parameter metadata must be non-negative")
        seen_param_starts.append(operation.param_start)

        if operation.operation_class == "GateOperation":
            gate_count += 1
            if operation.target_qbit is None:
                raise ValueError(
                    "GateOperation planner records must provide target_qbit"
                )
            if strict_phase3_support and operation.name not in SUPPORTED_PHASE3_GATE_NAMES:
                raise NoisyPlannerValidationError(
                    category="gate_family",
                    first_unsupported_condition=operation.name,
                    failure_stage="planner_entry_preflight",
                    source_type=surface.source_type,
                    requested_mode=surface.requested_mode,
                    reason=(
                        "Unsupported Phase 3 planner gate '{}' in canonical surface"
                    ).format(operation.name),
                )
        elif operation.operation_class == "NoiseOperation":
            if operation.target_qbit is None:
                raise ValueError(
                    "NoiseOperation planner records must provide target_qbit"
                )
            if operation.source_gate_index < 0 or operation.source_gate_index >= gate_count:
                raise NoisyPlannerValidationError(
                    category="noise_insertion",
                    first_unsupported_condition="after_gate_index",
                    failure_stage="planner_entry_preflight",
                    source_type=surface.source_type,
                    requested_mode=surface.requested_mode,
                    reason=(
                        "Canonical planner noise operation references unsupported "
                        "after_gate_index {}".format(operation.source_gate_index)
                    ),
                )
            if strict_phase3_support and operation.name not in SUPPORTED_PHASE3_NOISE_NAMES:
                raise NoisyPlannerValidationError(
                    category="noise_type",
                    first_unsupported_condition=operation.name,
                    failure_stage="planner_entry_preflight",
                    source_type=surface.source_type,
                    requested_mode=surface.requested_mode,
                    reason=(
                        "Unsupported Phase 3 planner noise model '{}' in canonical "
                        "surface"
                    ).format(operation.name),
                )
        else:
            raise ValueError(
                "Unsupported planner operation_class '{}'".format(
                    operation.operation_class
                )
            )

        expected_index += 1

    if seen_param_starts != sorted(seen_param_starts):
        raise ValueError(
            "Canonical planner operations must preserve monotonically increasing "
            "parameter starts"
        )

    return surface


def build_canonical_planner_surface_from_bridge_metadata(
    bridge_metadata: Mapping[str, Any],
    *,
    entry_route: str,
    workload_family: str,
    workload_id: str,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    source_type: str | None = None,
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    resolved_source_type = str(source_type or bridge_metadata["source_type"])
    _validate_mode(requested_mode, source_type=resolved_source_type)

    operations = tuple(
        _build_operation_from_mapping(payload, fallback_index=index)
        for index, payload in enumerate(bridge_metadata["operations"])
    )

    surface = CanonicalNoisyPlannerSurface(
        schema_version=CANONICAL_PLANNER_SCHEMA_VERSION,
        requested_mode=requested_mode,
        source_type=resolved_source_type,
        entry_route=entry_route,
        workload_family=workload_family,
        workload_id=workload_id,
        qbit_num=int(bridge_metadata["qbit_num"]),
        parameter_count=int(bridge_metadata["parameter_count"]),
        operations=operations,
    )
    return _validate_surface(surface, strict_phase3_support=strict_phase3_support)


def build_canonical_planner_surface_from_operation_specs(
    *,
    qbit_num: int,
    source_type: str,
    entry_route: str,
    workload_family: str,
    workload_id: str,
    operation_specs: Iterable[Mapping[str, Any]],
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    _validate_mode(requested_mode, source_type=source_type)

    canonical_operations: list[CanonicalNoisyPlannerOperation] = []
    param_start = 0
    gate_index = -1
    for index, payload in enumerate(operation_specs):
        kind = str(payload["kind"])
        if kind == "gate":
            gate_index += 1
        operation = _build_operation_from_spec(
            payload,
            index=index,
            default_param_start=param_start,
            gate_index=gate_index,
        )
        canonical_operations.append(operation)
        param_start = max(param_start, operation.param_start + operation.param_count)

    surface = CanonicalNoisyPlannerSurface(
        schema_version=CANONICAL_PLANNER_SCHEMA_VERSION,
        requested_mode=requested_mode,
        source_type=source_type,
        entry_route=entry_route,
        workload_family=workload_family,
        workload_id=workload_id,
        qbit_num=int(qbit_num),
        parameter_count=max(
            (operation.param_start + operation.param_count for operation in canonical_operations),
            default=0,
        ),
        operations=tuple(canonical_operations),
    )
    return _validate_surface(surface, strict_phase3_support=strict_phase3_support)


def _extract_legacy_noise_value(
    spec: Mapping[str, Any],
    *,
    channel: str,
    source_type: str,
    requested_mode: str,
) -> float:
    if "value" in spec:
        return float(spec["value"])
    key_by_channel = {
        "local_depolarizing": "error_rate",
        "amplitude_damping": "gamma",
        "phase_damping": "lambda",
    }
    value_key = key_by_channel.get(channel)
    if value_key is None or value_key not in spec:
        raise NoisyPlannerValidationError(
            category="noise_type",
            first_unsupported_condition=channel,
            failure_stage="planner_entry_preflight",
            source_type=source_type,
            requested_mode=requested_mode,
            reason=(
                "Legacy-source planner lowering requires an explicit fixed value for "
                "noise channel '{}'".format(channel)
            ),
        )
    return float(spec[value_key])


def _normalize_legacy_noise_specs(
    density_noise: Iterable[Mapping[str, Any]],
    *,
    gate_count: int,
    source_type: str,
    requested_mode: str,
) -> list[dict[str, Any]]:
    normalized_specs: list[dict[str, Any]] = []
    for spec in density_noise:
        channel = _normalize_noise_name(str(spec["channel"]))
        if channel not in SUPPORTED_PHASE3_NOISE_NAMES:
            raise NoisyPlannerValidationError(
                category="noise_type",
                first_unsupported_condition=channel,
                failure_stage="planner_entry_preflight",
                source_type=source_type,
                requested_mode=requested_mode,
                reason=(
                    "Unsupported Phase 3 planner noise model '{}' in legacy-source "
                    "lowering".format(channel)
                ),
            )
        after_gate_index = int(spec["after_gate_index"])
        if after_gate_index < 0 or after_gate_index >= gate_count:
            raise NoisyPlannerValidationError(
                category="noise_insertion",
                first_unsupported_condition="after_gate_index",
                failure_stage="planner_entry_preflight",
                source_type=source_type,
                requested_mode=requested_mode,
                reason=(
                    "Legacy-source planner lowering references unsupported "
                    "after_gate_index {}".format(after_gate_index)
                ),
            )
        normalized_specs.append(
            {
                "kind": "noise",
                "name": channel,
                "target_qbit": int(spec["target"]),
                "source_gate_index": after_gate_index,
                "fixed_value": _extract_legacy_noise_value(
                    spec,
                    channel=channel,
                    source_type=source_type,
                    requested_mode=requested_mode,
                ),
                "param_count": 0,
            }
        )
    return normalized_specs


def build_canonical_planner_surface_from_legacy_circuit(
    legacy_circuit: Any,
    *,
    workload_id: str,
    density_noise: Iterable[Mapping[str, Any]] | None = None,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    source_type: str = "legacy_qgd_circuit_exact",
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    _validate_mode(requested_mode, source_type=source_type)

    required_methods = ("get_Gates", "get_Qbit_Num")
    if any(not hasattr(legacy_circuit, method) for method in required_methods):
        raise NoisyPlannerValidationError(
            category="source_type",
            first_unsupported_condition="legacy_source_type",
            failure_stage="planner_entry_preflight",
            source_type=source_type,
            requested_mode=requested_mode,
            reason=(
                "Legacy-source planner lowering requires a qgd_Circuit- or "
                "Gates_block-like object exposing get_Gates() and get_Qbit_Num()"
            ),
        )

    gate_specs: list[dict[str, Any]] = []
    for gate in legacy_circuit.get_Gates():
        gate_name = _normalize_gate_name(str(gate.get_Name()))
        if strict_phase3_support and gate_name not in SUPPORTED_PHASE3_GATE_NAMES:
            raise NoisyPlannerValidationError(
                category="gate_family",
                first_unsupported_condition=gate_name,
                failure_stage="planner_entry_preflight",
                source_type=source_type,
                requested_mode=requested_mode,
                reason=(
                    "Unsupported Phase 3 planner gate '{}' in legacy-source lowering"
                ).format(gate_name),
            )
        spec = {
            "kind": "gate",
            "name": gate_name,
            "target_qbit": int(gate.get_Target_Qbit()),
            "param_count": int(gate.get_Parameter_Num()),
            "param_start": int(gate.get_Parameter_Start_Index()),
        }
        if hasattr(gate, "get_Control_Qbit"):
            control_qbit = int(gate.get_Control_Qbit())
            if control_qbit >= 0:
                spec["control_qbit"] = control_qbit
        gate_specs.append(spec)

    if not gate_specs:
        raise NoisyPlannerValidationError(
            category="source_type",
            first_unsupported_condition="empty_legacy_circuit",
            failure_stage="planner_entry_preflight",
            source_type=source_type,
            requested_mode=requested_mode,
            reason="Legacy-source planner lowering requires at least one supported gate",
        )

    noise_specs = _normalize_legacy_noise_specs(
        [] if density_noise is None else density_noise,
        gate_count=len(gate_specs),
        source_type=source_type,
        requested_mode=requested_mode,
    )

    noise_by_gate: dict[int, list[dict[str, Any]]] = {}
    for noise_spec in noise_specs:
        noise_by_gate.setdefault(noise_spec["source_gate_index"], []).append(noise_spec)

    operation_specs: list[dict[str, Any]] = []
    for gate_index, gate_spec in enumerate(gate_specs):
        operation_specs.append(gate_spec)
        operation_specs.extend(noise_by_gate.get(gate_index, []))

    return build_canonical_planner_surface_from_operation_specs(
        qbit_num=int(legacy_circuit.get_Qbit_Num()),
        source_type=source_type,
        entry_route=PHASE3_ENTRY_ROUTE_LEGACY_EXACT,
        workload_family=PHASE3_WORKLOAD_FAMILY_LEGACY,
        workload_id=workload_id,
        operation_specs=operation_specs,
        requested_mode=requested_mode,
        strict_phase3_support=strict_phase3_support,
    )


def build_canonical_planner_surface_from_qgd_circuit(
    circuit: Any,
    *,
    workload_id: str,
    density_noise: Iterable[Mapping[str, Any]] | None = None,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    return build_canonical_planner_surface_from_legacy_circuit(
        circuit,
        workload_id=workload_id,
        density_noise=density_noise,
        requested_mode=requested_mode,
        source_type="legacy_qgd_circuit_exact",
        strict_phase3_support=strict_phase3_support,
    )


def preflight_planner_request(
    *,
    source_type: str,
    workload_id: str,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    bridge_metadata: Mapping[str, Any] | None = None,
    operation_specs: Iterable[Mapping[str, Any]] | None = None,
    qbit_num: int | None = None,
    entry_route: str | None = None,
    workload_family: str | None = None,
    legacy_circuit: Any | None = None,
    density_noise: Iterable[Mapping[str, Any]] | None = None,
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    resolved_source_type = str(source_type)
    if resolved_source_type not in SUPPORTED_PLANNER_SOURCE_TYPES:
        raise NoisyPlannerValidationError(
            category="source_type",
            first_unsupported_condition=resolved_source_type,
            failure_stage="planner_entry_preflight",
            source_type=resolved_source_type,
            requested_mode=requested_mode,
            reason=(
                "Unsupported Phase 3 planner source type '{}'".format(
                    resolved_source_type
                )
            ),
        )

    provided_payload_count = sum(
        payload is not None
        for payload in (bridge_metadata, operation_specs, legacy_circuit)
    )
    if provided_payload_count != 1:
        raise NoisyPlannerValidationError(
            category="malformed_request",
            first_unsupported_condition=(
                "missing_source_payload"
                if provided_payload_count == 0
                else "ambiguous_source_payload"
            ),
            failure_stage="planner_entry_preflight",
            source_type=resolved_source_type,
            requested_mode=requested_mode,
            reason=(
                "Phase 3 planner preflight requires exactly one source payload "
                "(bridge_metadata, operation_specs, or legacy_circuit)"
            ),
        )

    if bridge_metadata is not None:
        if entry_route is None or workload_family is None:
            raise NoisyPlannerValidationError(
                category="malformed_request",
                first_unsupported_condition="missing_route_metadata",
                failure_stage="planner_entry_preflight",
                source_type=resolved_source_type,
                requested_mode=requested_mode,
                reason=(
                    "Bridge-based planner preflight requires entry_route and "
                    "workload_family metadata"
                ),
            )
        return build_canonical_planner_surface_from_bridge_metadata(
            bridge_metadata,
            entry_route=entry_route,
            workload_family=workload_family,
            workload_id=workload_id,
            requested_mode=requested_mode,
            source_type=resolved_source_type,
            strict_phase3_support=strict_phase3_support,
        )

    if operation_specs is not None:
        if qbit_num is None or entry_route is None or workload_family is None:
            raise NoisyPlannerValidationError(
                category="malformed_request",
                first_unsupported_condition="missing_route_metadata",
                failure_stage="planner_entry_preflight",
                source_type=resolved_source_type,
                requested_mode=requested_mode,
                reason=(
                    "Operation-spec planner preflight requires qbit_num, "
                    "entry_route, and workload_family metadata"
                ),
            )
        return build_canonical_planner_surface_from_operation_specs(
            qbit_num=qbit_num,
            source_type=resolved_source_type,
            entry_route=entry_route,
            workload_family=workload_family,
            workload_id=workload_id,
            operation_specs=operation_specs,
            requested_mode=requested_mode,
            strict_phase3_support=strict_phase3_support,
        )

    return build_canonical_planner_surface_from_legacy_circuit(
        legacy_circuit,
        workload_id=workload_id,
        density_noise=density_noise,
        requested_mode=requested_mode,
        source_type=resolved_source_type,
        strict_phase3_support=strict_phase3_support,
    )


def build_phase3_continuity_planner_surface(
    vqe,
    *,
    workload_id: str | None = None,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
) -> CanonicalNoisyPlannerSurface:
    bridge_metadata = vqe.describe_density_bridge()
    resolved_workload_id = workload_id or "phase2_xxz_hea_q{}_continuity".format(
        bridge_metadata["qbit_num"]
    )
    return build_canonical_planner_surface_from_bridge_metadata(
        bridge_metadata,
        entry_route=PHASE3_ENTRY_ROUTE_PHASE2_CONTINUITY,
        workload_family=PHASE3_WORKLOAD_FAMILY_PHASE2_CONTINUITY,
        workload_id=resolved_workload_id,
        requested_mode=requested_mode,
        strict_phase3_support=True,
    )


def build_planner_audit_record(
    surface: CanonicalNoisyPlannerSurface,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = surface.to_dict()
    return {
        "schema_version": payload["schema_version"],
        "requested_mode": payload["requested_mode"],
        "provenance": payload["provenance"],
        "summary": {
            "qbit_num": payload["qbit_num"],
            "parameter_count": payload["parameter_count"],
            "operation_count": payload["operation_count"],
            "gate_count": payload["gate_count"],
            "noise_count": payload["noise_count"],
            "gate_sequence": payload["gate_sequence"],
            "noise_sequence": payload["noise_sequence"],
            "max_qubit_span": payload["max_qubit_span"],
        },
        "operations": payload["operations"],
        "metadata": dict(metadata) if metadata is not None else {},
    }


def build_bridge_overlap_report(
    surface: CanonicalNoisyPlannerSurface, bridge_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    payload = surface.to_dict()
    mismatches: list[dict[str, Any]] = []

    for key in ("parameter_count", "operation_count", "gate_count", "noise_count"):
        if payload[key] != bridge_metadata[key]:
            mismatches.append(
                {"kind": "summary", "field": key, "actual": payload[key], "expected": bridge_metadata[key]}
            )

    for index, (actual, expected) in enumerate(
        zip(payload["operations"], bridge_metadata["operations"])
    ):
        for key in (
            "operation_class",
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
            if actual[key] != expected[key]:
                mismatches.append(
                    {
                        "kind": "operation",
                        "index": index,
                        "field": key,
                        "actual": actual[key],
                        "expected": expected[key],
                    }
                )
        expected_support = [
            q for q in (expected["control_qbit"], expected["target_qbit"]) if q is not None
        ]
        if actual["qubit_support"] != expected_support:
            mismatches.append(
                {
                    "kind": "operation",
                    "index": index,
                    "field": "qubit_support",
                    "actual": actual["qubit_support"],
                    "expected": expected_support,
                }
            )

    return {
        "bridge_overlap_pass": not mismatches,
        "mismatches": mismatches,
        "compared_operation_count": min(
            len(payload["operations"]), len(bridge_metadata["operations"])
        ),
    }
