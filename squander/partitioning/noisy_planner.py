from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, cast

from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base,
)

PARTITIONED_DENSITY_MODE = "partitioned_density"
DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS = 2

SUPPORTED_GATE_NAMES = frozenset({"U3", "CNOT"})
SUPPORTED_NOISE_NAMES = frozenset(
    {"local_depolarizing", "amplitude_damping", "phase_damping"}
)
SUPPORTED_PLANNER_SOURCE_TYPES = frozenset(
    {
        "generated_hea",
        "microcase_builder",
        "structured_family_builder",
        "legacy_qgd_circuit_exact",
    }
)

PLANNER_OP_KIND_GATE = "gate"
PLANNER_OP_KIND_NOISE = "noise"
PLANNER_OP_KINDS: frozenset[str] = frozenset(
    {PLANNER_OP_KIND_GATE, PLANNER_OP_KIND_NOISE}
)
PlannerOperationKindWire = Literal["gate", "noise"]


class NoisyPlannerValidationError(ValueError):
    """Structured planner-entry validation error for the Phase 3 noisy planner surface."""

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


class NoisyDescriptorValidationError(ValueError):
    """Structured descriptor-generation validation error for the Phase 3 partition descriptor surface."""

    def __init__(
        self,
        *,
        category: str,
        first_unsupported_condition: str,
        failure_stage: str,
        source_type: str,
        requested_mode: str,
        workload_id: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.category = category
        self.first_unsupported_condition = first_unsupported_condition
        self.failure_stage = failure_stage
        self.source_type = source_type
        self.requested_mode = requested_mode
        self.workload_id = workload_id
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "unsupported_category": self.category,
            "first_unsupported_condition": self.first_unsupported_condition,
            "failure_stage": self.failure_stage,
            "source_type": self.source_type,
            "requested_mode": self.requested_mode,
            "workload_id": self.workload_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CanonicalNoisyPlannerOperation:
    index: int
    kind: PlannerOperationKindWire
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
    requested_mode: str
    source_type: str
    workload_id: str
    qbit_num: int
    parameter_count: int
    operations: tuple[CanonicalNoisyPlannerOperation, ...]

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    @property
    def gate_count(self) -> int:
        return sum(op.kind == PLANNER_OP_KIND_GATE for op in self.operations)

    @property
    def noise_count(self) -> int:
        return sum(op.kind == PLANNER_OP_KIND_NOISE for op in self.operations)

    @property
    def gate_names(self) -> tuple[str, ...]:
        return tuple(
            op.name for op in self.operations if op.kind == PLANNER_OP_KIND_GATE
        )

    @property
    def noise_names(self) -> tuple[str, ...]:
        return tuple(
            op.name for op in self.operations if op.kind == PLANNER_OP_KIND_NOISE
        )

    @property
    def max_qubit_span(self) -> int:
        return max((len(op.qubit_support) for op in self.operations), default=0)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "workload_id": self.workload_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "source_type": self.source_type,
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


@dataclass(frozen=True)
class NoisyPartitionDescriptorMember:
    partition_member_index: int
    canonical_operation_index: int
    kind: PlannerOperationKindWire
    name: str
    is_unitary: bool
    source_gate_index: int
    target_qbit: int | None
    control_qbit: int | None
    qubit_support: tuple[int, ...]
    local_target_qbit: int | None
    local_control_qbit: int | None
    local_qubit_support: tuple[int, ...]
    param_count: int
    param_start: int
    local_param_start: int
    fixed_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_member_index": self.partition_member_index,
            "canonical_operation_index": self.canonical_operation_index,
            "kind": self.kind,
            "name": self.name,
            "is_unitary": self.is_unitary,
            "source_gate_index": self.source_gate_index,
            "target_qbit": self.target_qbit,
            "control_qbit": self.control_qbit,
            "qubit_support": list(self.qubit_support),
            "local_target_qbit": self.local_target_qbit,
            "local_control_qbit": self.local_control_qbit,
            "local_qubit_support": list(self.local_qubit_support),
            "param_count": self.param_count,
            "param_start": self.param_start,
            "local_param_start": self.local_param_start,
            "fixed_value": self.fixed_value,
        }


@dataclass(frozen=True)
class NoisyPartitionDescriptor:
    partition_index: int
    canonical_operation_indices: tuple[int, ...]
    local_to_global_qbits: tuple[int, ...]
    parameter_routing: tuple[tuple[int, int, int], ...]
    members: tuple[NoisyPartitionDescriptorMember, ...]

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def gate_count(self) -> int:
        return sum(member.kind == PLANNER_OP_KIND_GATE for member in self.members)

    @property
    def noise_count(self) -> int:
        return sum(member.kind == PLANNER_OP_KIND_NOISE for member in self.members)

    @property
    def partition_parameter_count(self) -> int:
        return sum(member.param_count for member in self.members)

    @property
    def partition_qubit_span(self) -> tuple[int, ...]:
        return self.local_to_global_qbits

    @property
    def requires_remap(self) -> bool:
        return self.local_to_global_qbits != tuple(
            range(len(self.local_to_global_qbits))
        )

    @property
    def global_to_local_qbits(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (global_qbit, local_qbit)
            for local_qbit, global_qbit in enumerate(self.local_to_global_qbits)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_index": self.partition_index,
            "member_count": self.member_count,
            "gate_count": self.gate_count,
            "noise_count": self.noise_count,
            "canonical_operation_indices": list(self.canonical_operation_indices),
            "partition_qubit_span": list(self.partition_qubit_span),
            "local_to_global_qbits": list(self.local_to_global_qbits),
            "global_to_local_qbits": [
                {"global_qbit": global_qbit, "local_qbit": local_qbit}
                for global_qbit, local_qbit in self.global_to_local_qbits
            ],
            "requires_remap": self.requires_remap,
            "partition_parameter_count": self.partition_parameter_count,
            "parameter_routing": [
                {
                    "global_param_start": global_param_start,
                    "local_param_start": local_param_start,
                    "param_count": param_count,
                }
                for global_param_start, local_param_start, param_count in self.parameter_routing
            ],
            "members": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class NoisyPartitionDescriptorSet:
    requested_mode: str
    source_type: str
    workload_id: str
    qbit_num: int
    parameter_count: int
    max_partition_qubits: int
    partitions: tuple[NoisyPartitionDescriptor, ...]

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "workload_id": self.workload_id,
        }

    @property
    def partition_count(self) -> int:
        return len(self.partitions)

    @property
    def descriptor_member_count(self) -> int:
        return sum(partition.member_count for partition in self.partitions)

    @property
    def gate_count(self) -> int:
        return sum(partition.gate_count for partition in self.partitions)

    @property
    def noise_count(self) -> int:
        return sum(partition.noise_count for partition in self.partitions)

    @property
    def max_partition_span(self) -> int:
        return max(
            (len(partition.partition_qubit_span) for partition in self.partitions),
            default=0,
        )

    @property
    def partition_member_counts(self) -> tuple[int, ...]:
        return tuple(partition.member_count for partition in self.partitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "source_type": self.source_type,
            "workload_id": self.workload_id,
            "provenance": self.provenance,
            "qbit_num": self.qbit_num,
            "parameter_count": self.parameter_count,
            "max_partition_qubits": self.max_partition_qubits,
            "partition_count": self.partition_count,
            "descriptor_member_count": self.descriptor_member_count,
            "gate_count": self.gate_count,
            "noise_count": self.noise_count,
            "max_partition_span": self.max_partition_span,
            "partition_member_counts": list(self.partition_member_counts),
            "partitions": [partition.to_dict() for partition in self.partitions],
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


def _coerce_operation_kind(value: Any) -> PlannerOperationKindWire:
    raw_kind = str(value)
    if raw_kind not in PLANNER_OP_KINDS:
        raise ValueError(
            "Unsupported planner kind '{}', expected one of {}".format(
                raw_kind, sorted(PLANNER_OP_KINDS)
            )
        )
    return cast(PlannerOperationKindWire, raw_kind)


def _canonicalize_operation_name(name: str, kind: PlannerOperationKindWire) -> str:
    if kind == PLANNER_OP_KIND_GATE:
        return _normalize_gate_name(name)
    if kind == PLANNER_OP_KIND_NOISE:
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
    kind = _coerce_operation_kind(payload["kind"])

    return CanonicalNoisyPlannerOperation(
        index=int(payload.get("index", fallback_index)),
        kind=kind,
        name=_canonicalize_operation_name(str(payload["name"]), kind),
        is_unitary=bool(payload["is_unitary"]),
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
    kind = _coerce_operation_kind(payload["kind"])

    return CanonicalNoisyPlannerOperation(
        index=index,
        kind=kind,
        name=_canonicalize_operation_name(str(payload["name"]), kind),
        is_unitary=bool(payload.get("is_unitary", kind == PLANNER_OP_KIND_GATE)),
        source_gate_index=int(payload.get("source_gate_index", gate_index)),
        target_qbit=_coerce_optional_int(payload.get("target_qbit")),
        control_qbit=_coerce_optional_int(payload.get("control_qbit")),
        param_count=int(payload.get("param_count", 0)),
        param_start=int(payload.get("param_start", default_param_start)),
        fixed_value=_coerce_optional_float(payload.get("fixed_value")),
    )


def _validate_mode(requested_mode: str, *, stage: str, source_type: str) -> None:
    if requested_mode != PARTITIONED_DENSITY_MODE:
        raise NoisyPlannerValidationError(
            category="mode",
            first_unsupported_condition="unsupported_mode",
            failure_stage=stage,
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
    stage: str,
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
            raise ValueError(
                "Planner operation parameter metadata must be non-negative"
            )
        seen_param_starts.append(operation.param_start)

        if operation.kind == PLANNER_OP_KIND_GATE:
            gate_count += 1
            if operation.target_qbit is None:
                raise ValueError(
                    "{} planner records must provide target_qbit".format(
                        PLANNER_OP_KIND_GATE
                    )
                )
            if strict_phase3_support and operation.name not in SUPPORTED_GATE_NAMES:
                raise NoisyPlannerValidationError(
                    category="gate_family",
                    first_unsupported_condition=operation.name,
                    failure_stage=stage,
                    source_type=surface.source_type,
                    requested_mode=surface.requested_mode,
                    reason=(
                        "Unsupported Phase 3 planner gate '{}' in canonical surface"
                    ).format(operation.name),
                )
        elif operation.kind == PLANNER_OP_KIND_NOISE:
            if operation.target_qbit is None:
                raise ValueError(
                    "{} planner records must provide target_qbit".format(
                        PLANNER_OP_KIND_NOISE
                    )
                )
            if (
                operation.source_gate_index < 0
                or operation.source_gate_index >= gate_count
            ):
                raise NoisyPlannerValidationError(
                    category="noise_insertion",
                    first_unsupported_condition="after_gate_index",
                    failure_stage=stage,
                    source_type=surface.source_type,
                    requested_mode=surface.requested_mode,
                    reason=(
                        "Canonical planner noise operation references unsupported "
                        "after_gate_index {}".format(operation.source_gate_index)
                    ),
                )
            if strict_phase3_support and operation.name not in SUPPORTED_NOISE_NAMES:
                raise NoisyPlannerValidationError(
                    category="noise_type",
                    first_unsupported_condition=operation.name,
                    failure_stage=stage,
                    source_type=surface.source_type,
                    requested_mode=surface.requested_mode,
                    reason=(
                        "Unsupported Phase 3 planner noise model '{}' in canonical "
                        "surface"
                    ).format(operation.name),
                )
        else:
            raise ValueError("Unsupported planner kind '{}'".format(operation.kind))

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
    workload_id: str,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    source_type: str | None = None,
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    resolved_source_type = str(source_type or bridge_metadata["source_type"])
    _validate_mode(
        requested_mode,
        stage="planner_entry_from_bridge_metadata_preflight",
        source_type=resolved_source_type,
    )

    operations = tuple(
        _build_operation_from_mapping(payload, fallback_index=index)
        for index, payload in enumerate(bridge_metadata["operations"])
    )

    surface = CanonicalNoisyPlannerSurface(
        requested_mode=requested_mode,
        source_type=resolved_source_type,
        workload_id=workload_id,
        qbit_num=int(bridge_metadata["qbit_num"]),
        parameter_count=int(bridge_metadata["parameter_count"]),
        operations=operations,
    )
    return _validate_surface(
        surface,
        stage="planner_entry_from_bridge_metadata_preflight",
        strict_phase3_support=strict_phase3_support,
    )


def build_canonical_planner_surface_from_operation_specs(
    *,
    qbit_num: int,
    source_type: str,
    workload_id: str,
    operation_specs: Iterable[Mapping[str, Any]],
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    strict_phase3_support: bool = True,
) -> CanonicalNoisyPlannerSurface:
    _validate_mode(
        requested_mode,
        stage="planner_entry_from_operation_specs_preflight",
        source_type=source_type,
    )

    canonical_operations: list[CanonicalNoisyPlannerOperation] = []
    param_start = 0
    gate_index = -1
    for index, payload in enumerate(operation_specs):
        kind = _coerce_operation_kind(payload["kind"])
        if kind == PLANNER_OP_KIND_GATE:
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
        requested_mode=requested_mode,
        source_type=source_type,
        workload_id=workload_id,
        qbit_num=int(qbit_num),
        parameter_count=max(
            (
                operation.param_start + operation.param_count
                for operation in canonical_operations
            ),
            default=0,
        ),
        operations=tuple(canonical_operations),
    )
    return _validate_surface(
        surface,
        stage="planner_entry_from_operation_specs_preflight",
        strict_phase3_support=strict_phase3_support,
    )


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
        if channel not in SUPPORTED_NOISE_NAMES:
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
    _validate_mode(
        requested_mode,
        stage="planner_entry_preflight",
        source_type=source_type,
    )

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
        if strict_phase3_support and gate_name not in SUPPORTED_GATE_NAMES:
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
        return build_canonical_planner_surface_from_bridge_metadata(
            bridge_metadata,
            workload_id=workload_id,
            requested_mode=requested_mode,
            source_type=resolved_source_type,
            strict_phase3_support=strict_phase3_support,
        )

    if operation_specs is not None:
        if qbit_num is None:
            raise NoisyPlannerValidationError(
                category="malformed_request",
                first_unsupported_condition="missing_qbit_num",
                failure_stage="planner_entry_preflight",
                source_type=resolved_source_type,
                requested_mode=requested_mode,
                reason="Operation-spec planner preflight requires qbit_num",
            )
        return build_canonical_planner_surface_from_operation_specs(
            qbit_num=qbit_num,
            source_type=resolved_source_type,
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


##
# @brief Build a canonical planner surface from a Phase 2 continuity VQE instance.
# @param vqe: A Phase 2 continuity VQE instance. The VQE instance should be configured for the exact noisy HEA anchor workflow.
# @param workload_id: The workload ID to use for the planner surface.
# @param requested_mode: The requested mode for the planner surface.
# @return A canonical planner surface.
def build_phase3_continuity_planner_surface(
    vqe: qgd_Variational_Quantum_Eigensolver_Base,
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


##
# @brief Build a bridge overlap report from a canonical planner surface and a bridge metadata.
# @param surface: A canonical planner surface.
# @param bridge_metadata: A bridge metadata.
# @return A bridge overlap report.
def build_bridge_overlap_report(
    surface: CanonicalNoisyPlannerSurface, bridge_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    payload = surface.to_dict()
    mismatches: list[dict[str, Any]] = []

    for key in ("parameter_count", "operation_count", "gate_count", "noise_count"):
        if payload[key] != bridge_metadata[key]:
            mismatches.append(
                {
                    "kind": "summary",
                    "field": key,
                    "actual": payload[key],
                    "expected": bridge_metadata[key],
                }
            )

    for index, (actual, expected) in enumerate(
        zip(payload["operations"], bridge_metadata["operations"])
    ):
        for key in (
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
            q
            for q in (expected["control_qbit"], expected["target_qbit"])
            if q is not None
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


def _validate_descriptor_request(
    surface: CanonicalNoisyPlannerSurface, *, max_partition_qubits: int
) -> None:
    if max_partition_qubits <= 0:
        raise NoisyDescriptorValidationError(
            category="descriptor_request",
            first_unsupported_condition="max_partition_qubits",
            failure_stage="descriptor_generation_preflight",
            source_type=surface.source_type,
            requested_mode=surface.requested_mode,
            workload_id=surface.workload_id,
            reason="Partition descriptor generation requires max_partition_qubits > 0",
        )
    if surface.max_qubit_span > max_partition_qubits:
        raise NoisyDescriptorValidationError(
            category="partition_span",
            first_unsupported_condition="max_partition_qubits",
            failure_stage="descriptor_generation_preflight",
            source_type=surface.source_type,
            requested_mode=surface.requested_mode,
            workload_id=surface.workload_id,
            reason=(
                "Partition descriptor generation requires max_partition_qubits >= "
                "the canonical max_qubit_span {} for workload '{}'".format(
                    surface.max_qubit_span, surface.workload_id
                )
            ),
        )


##
# @brief Partition the operations of a canonical planner surface into partitions.
# The partition is done by the following algorithm (greedy algorithm):
# 1. Start with an empty partition.
# 2. Add the first operation to the partition.
# 3. For each subsequent operation, add it to the partition if it can be added without exceeding the maximum number of qubits allowed in a partition.
# 4. If the next operation cannot be added to the partition without exceeding the maximum number of qubits allowed in a partition, start a new partition.
# 5. Repeat steps 2-4 until all operations have been added to a partition.
# @param surface: A canonical planner surface.
# @param max_partition_qubits: The maximum number of qubits allowed in a partition.
# @return A tuple of tuples of operations, where each tuple is a partition.
def _partition_surface_operations(
    surface: CanonicalNoisyPlannerSurface, *, max_partition_qubits: int
) -> tuple[tuple[CanonicalNoisyPlannerOperation, ...], ...]:
    partitions: list[tuple[CanonicalNoisyPlannerOperation, ...]] = []
    current_partition: list[CanonicalNoisyPlannerOperation] = []
    current_support: set[int] = set()

    for operation in surface.operations:
        operation_support = set(operation.qubit_support)
        expanded_support = current_support | operation_support
        if current_partition and len(expanded_support) > max_partition_qubits:
            partitions.append(tuple(current_partition))
            current_partition = []
            current_support = set()
        current_partition.append(operation)
        current_support |= operation_support

    if current_partition:
        partitions.append(tuple(current_partition))

    return tuple(partitions)


def _build_descriptor_partition(
    operations: tuple[CanonicalNoisyPlannerOperation, ...], *, partition_index: int
) -> NoisyPartitionDescriptor:
    local_to_global_qbits = tuple(
        sorted({qbit for operation in operations for qbit in operation.qubit_support})
    )
    global_to_local = {
        global_qbit: local_qbit
        for local_qbit, global_qbit in enumerate(local_to_global_qbits)
    }

    members: list[NoisyPartitionDescriptorMember] = []
    parameter_routing: list[tuple[int, int, int]] = []
    local_param_start = 0

    for partition_member_index, operation in enumerate(operations):
        if operation.param_count:
            parameter_routing.append(
                (operation.param_start, local_param_start, operation.param_count)
            )
        member = NoisyPartitionDescriptorMember(
            partition_member_index=partition_member_index,
            canonical_operation_index=operation.index,
            kind=operation.kind,
            name=operation.name,
            is_unitary=operation.is_unitary,
            source_gate_index=operation.source_gate_index,
            target_qbit=operation.target_qbit,
            control_qbit=operation.control_qbit,
            qubit_support=operation.qubit_support,
            local_target_qbit=(
                None
                if operation.target_qbit is None
                else global_to_local[operation.target_qbit]
            ),
            local_control_qbit=(
                None
                if operation.control_qbit is None
                else global_to_local[operation.control_qbit]
            ),
            local_qubit_support=tuple(
                global_to_local[qbit] for qbit in operation.qubit_support
            ),
            param_count=operation.param_count,
            param_start=operation.param_start,
            local_param_start=local_param_start,
            fixed_value=operation.fixed_value,
        )
        members.append(member)
        local_param_start += operation.param_count

    return NoisyPartitionDescriptor(
        partition_index=partition_index,
        canonical_operation_indices=tuple(operation.index for operation in operations),
        local_to_global_qbits=local_to_global_qbits,
        parameter_routing=tuple(parameter_routing),
        members=tuple(members),
    )


def build_partition_descriptor_set(
    surface: CanonicalNoisyPlannerSurface,
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
) -> NoisyPartitionDescriptorSet:
    _validate_descriptor_request(surface, max_partition_qubits=max_partition_qubits)
    partitions = tuple(
        _build_descriptor_partition(
            partition_operations, partition_index=partition_index
        )
        for partition_index, partition_operations in enumerate(
            _partition_surface_operations(
                surface, max_partition_qubits=max_partition_qubits
            )
        )
    )
    descriptor_set = NoisyPartitionDescriptorSet(
        requested_mode=surface.requested_mode,
        source_type=surface.source_type,
        workload_id=surface.workload_id,
        qbit_num=surface.qbit_num,
        parameter_count=surface.parameter_count,
        max_partition_qubits=max_partition_qubits,
        partitions=partitions,
    )
    return validate_partition_descriptor_set_against_surface(surface, descriptor_set)


def build_phase3_continuity_partition_descriptor_set(
    vqe,
    *,
    workload_id: str | None = None,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
) -> NoisyPartitionDescriptorSet:
    surface = build_phase3_continuity_planner_surface(
        vqe,
        workload_id=workload_id,
        requested_mode=requested_mode,
    )
    return build_partition_descriptor_set(
        surface, max_partition_qubits=max_partition_qubits
    )


def preflight_descriptor_request(
    *,
    source_type: str,
    workload_id: str,
    requested_mode: str = PARTITIONED_DENSITY_MODE,
    bridge_metadata: Mapping[str, Any] | None = None,
    operation_specs: Iterable[Mapping[str, Any]] | None = None,
    qbit_num: int | None = None,
    legacy_circuit: Any | None = None,
    density_noise: Iterable[Mapping[str, Any]] | None = None,
    strict_phase3_support: bool = True,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
) -> NoisyPartitionDescriptorSet:
    surface = preflight_planner_request(
        source_type=source_type,
        workload_id=workload_id,
        requested_mode=requested_mode,
        bridge_metadata=bridge_metadata,
        operation_specs=operation_specs,
        qbit_num=qbit_num,
        legacy_circuit=legacy_circuit,
        density_noise=density_noise,
        strict_phase3_support=strict_phase3_support,
    )
    return build_partition_descriptor_set(
        surface, max_partition_qubits=max_partition_qubits
    )


def _descriptor_error(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    category: str,
    first_unsupported_condition: str,
    failure_stage: str,
    reason: str,
) -> NoisyDescriptorValidationError:
    return NoisyDescriptorValidationError(
        category=category,
        first_unsupported_condition=first_unsupported_condition,
        failure_stage=failure_stage,
        source_type=descriptor_set.source_type,
        requested_mode=descriptor_set.requested_mode,
        workload_id=descriptor_set.workload_id,
        reason=reason,
    )


def validate_partition_descriptor_set(
    descriptor_set: NoisyPartitionDescriptorSet,
) -> NoisyPartitionDescriptorSet:
    expected_partition_index = 0
    expected_canonical_index = 0

    for partition in descriptor_set.partitions:
        if partition.partition_index != expected_partition_index:
            raise _descriptor_error(
                descriptor_set,
                category="reordering_across_noise_boundaries",
                first_unsupported_condition="partition_index",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptors require contiguous partition indices starting "
                    "at 0; got partition_index {} while expecting {}".format(
                        partition.partition_index, expected_partition_index
                    )
                ),
            )
        if not partition.members:
            raise _descriptor_error(
                descriptor_set,
                category="descriptor_request",
                first_unsupported_condition="empty_partition",
                failure_stage="descriptor_validation",
                reason="Partition descriptors do not support empty partitions",
            )

        if len(set(partition.local_to_global_qbits)) != len(
            partition.local_to_global_qbits
        ):
            raise _descriptor_error(
                descriptor_set,
                category="incomplete_remapping",
                first_unsupported_condition="duplicate_global_qbit",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor partition {} contains duplicate entries in "
                    "local_to_global_qbits".format(partition.partition_index)
                ),
            )

        canonical_member_indices = tuple(
            member.canonical_operation_index for member in partition.members
        )
        if partition.canonical_operation_indices != canonical_member_indices:
            raise _descriptor_error(
                descriptor_set,
                category="reordering_across_noise_boundaries",
                first_unsupported_condition="canonical_operation_indices",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor partition {} must keep canonical_operation_indices "
                    "aligned with ordered member canonical indices".format(
                        partition.partition_index
                    )
                ),
            )

        global_to_local = {
            global_qbit: local_qbit
            for local_qbit, global_qbit in enumerate(partition.local_to_global_qbits)
        }
        expected_parameter_routing: list[tuple[int, int, int]] = []
        expected_local_param_start = 0

        for member_index, member in enumerate(partition.members):
            if member.partition_member_index != member_index:
                raise _descriptor_error(
                    descriptor_set,
                    category="reordering_across_noise_boundaries",
                    first_unsupported_condition="partition_member_index",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} requires contiguous "
                        "partition_member_index values".format(
                            partition.partition_index
                        )
                    ),
                )
            if member.canonical_operation_index != expected_canonical_index:
                raise _descriptor_error(
                    descriptor_set,
                    category="reordering_across_noise_boundaries",
                    first_unsupported_condition="canonical_operation_index",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptors must preserve contiguous canonical "
                        "operation coverage; partition {} member {} expected canonical "
                        "index {} but got {}".format(
                            partition.partition_index,
                            member_index,
                            expected_canonical_index,
                            member.canonical_operation_index,
                        )
                    ),
                )
            expected_canonical_index += 1

            expected_local_qubit_support = tuple(
                global_to_local[qbit] for qbit in member.qubit_support
            )
            if member.local_qubit_support != expected_local_qubit_support:
                raise _descriptor_error(
                    descriptor_set,
                    category="incomplete_remapping",
                    first_unsupported_condition="local_qubit_support",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has inconsistent local_qubit_support "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            expected_local_target = (
                None
                if member.target_qbit is None
                else global_to_local[member.target_qbit]
            )
            if member.local_target_qbit != expected_local_target:
                raise _descriptor_error(
                    descriptor_set,
                    category="incomplete_remapping",
                    first_unsupported_condition="local_target_qbit",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has inconsistent local_target_qbit "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            expected_local_control = (
                None
                if member.control_qbit is None
                else global_to_local[member.control_qbit]
            )
            if member.local_control_qbit != expected_local_control:
                raise _descriptor_error(
                    descriptor_set,
                    category="incomplete_remapping",
                    first_unsupported_condition="local_control_qbit",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has inconsistent local_control_qbit "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            if member.local_param_start != expected_local_param_start:
                raise _descriptor_error(
                    descriptor_set,
                    category="ambiguous_parameter_routing",
                    first_unsupported_condition="local_param_start",
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor partition {} has non-contiguous local_param_start "
                        "for canonical operation {}".format(
                            partition.partition_index, member.canonical_operation_index
                        )
                    ),
                )
            if member.param_count:
                expected_parameter_routing.append(
                    (member.param_start, member.local_param_start, member.param_count)
                )
            expected_local_param_start += member.param_count

        if partition.parameter_routing != tuple(expected_parameter_routing):
            raise _descriptor_error(
                descriptor_set,
                category="ambiguous_parameter_routing",
                first_unsupported_condition="parameter_routing",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor partition {} must keep parameter_routing aligned "
                    "with ordered member parameter metadata".format(
                        partition.partition_index
                    )
                ),
            )
        expected_partition_index += 1

    if expected_canonical_index != descriptor_set.descriptor_member_count:
        raise _descriptor_error(
            descriptor_set,
            category="dropped_operations",
            first_unsupported_condition="descriptor_member_count",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor coverage ended at canonical index {} but "
                "descriptor_member_count reports {}".format(
                    expected_canonical_index, descriptor_set.descriptor_member_count
                )
            ),
        )

    return descriptor_set


def validate_partition_descriptor_set_against_surface(
    surface: CanonicalNoisyPlannerSurface,
    descriptor_set: NoisyPartitionDescriptorSet,
) -> NoisyPartitionDescriptorSet:
    validated = validate_partition_descriptor_set(descriptor_set)
    if (
        validated.requested_mode != surface.requested_mode
        or validated.source_type != surface.source_type
        or validated.workload_id != surface.workload_id
        or validated.qbit_num != surface.qbit_num
        or validated.parameter_count != surface.parameter_count
    ):
        raise _descriptor_error(
            validated,
            category="descriptor_request",
            first_unsupported_condition="provenance_mismatch",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor provenance and size metadata must match the "
                "canonical planner surface for workload '{}'".format(
                    surface.workload_id
                )
            ),
        )
    if validated.descriptor_member_count != surface.operation_count:
        raise _descriptor_error(
            validated,
            category="dropped_operations",
            first_unsupported_condition="operation_count",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor workload '{}' records {} descriptor members but "
                "the canonical surface contains {} operations".format(
                    surface.workload_id,
                    validated.descriptor_member_count,
                    surface.operation_count,
                )
            ),
        )
    if validated.noise_count != surface.noise_count:
        raise _descriptor_error(
            validated,
            category="hidden_noise_placement",
            first_unsupported_condition="noise_count",
            failure_stage="descriptor_validation",
            reason=(
                "Partition descriptor workload '{}' records {} noise operations but "
                "the canonical surface contains {}".format(
                    surface.workload_id,
                    validated.noise_count,
                    surface.noise_count,
                )
            ),
        )

    flattened_members = [
        member for partition in validated.partitions for member in partition.members
    ]
    for member in flattened_members:
        expected = surface.operations[member.canonical_operation_index]
        for field in (
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
            if getattr(member, field) != getattr(expected, field):
                category = (
                    "hidden_noise_placement"
                    if expected.kind == PLANNER_OP_KIND_NOISE
                    or member.kind == PLANNER_OP_KIND_NOISE
                    else "dropped_operations"
                )
                raise _descriptor_error(
                    validated,
                    category=category,
                    first_unsupported_condition=field,
                    failure_stage="descriptor_validation",
                    reason=(
                        "Partition descriptor canonical operation {} mismatches the "
                        "planner surface field '{}'".format(
                            member.canonical_operation_index, field
                        )
                    ),
                )
        if member.qubit_support != expected.qubit_support:
            category = (
                "hidden_noise_placement"
                if expected.kind == PLANNER_OP_KIND_NOISE
                or member.kind == PLANNER_OP_KIND_NOISE
                else "dropped_operations"
            )
            raise _descriptor_error(
                validated,
                category=category,
                first_unsupported_condition="qubit_support",
                failure_stage="descriptor_validation",
                reason=(
                    "Partition descriptor canonical operation {} mismatches the "
                    "planner surface qubit_support".format(
                        member.canonical_operation_index
                    )
                ),
            )

    return validated


def build_descriptor_audit_record(
    descriptor_set: NoisyPartitionDescriptorSet,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = validate_partition_descriptor_set(descriptor_set).to_dict()
    return {
        "requested_mode": payload["requested_mode"],
        "provenance": payload["provenance"],
        "summary": {
            "qbit_num": payload["qbit_num"],
            "parameter_count": payload["parameter_count"],
            "partition_count": payload["partition_count"],
            "descriptor_member_count": payload["descriptor_member_count"],
            "gate_count": payload["gate_count"],
            "noise_count": payload["noise_count"],
            "max_partition_qubits": payload["max_partition_qubits"],
            "max_partition_span": payload["max_partition_span"],
            "partition_member_counts": payload["partition_member_counts"],
            "remapped_partition_count": sum(
                partition["requires_remap"] for partition in payload["partitions"]
            ),
            "parameter_routing_segment_count": sum(
                len(partition["parameter_routing"])
                for partition in payload["partitions"]
            ),
        },
        "partitions": payload["partitions"],
        "metadata": dict(metadata) if metadata is not None else {},
    }
