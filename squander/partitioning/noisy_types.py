from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

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
    """Partition-local overlay; canonical gate/noise fields live in `NoisyPartitionDescriptorSet.operations`."""

    partition_member_index: int
    canonical_operation_index: int
    local_target_qbit: int | None
    local_control_qbit: int | None
    local_qubit_support: tuple[int, ...]
    local_param_start: int

    def to_dict(self, descriptor_set: NoisyPartitionDescriptorSet) -> dict[str, Any]:
        from squander.partitioning.noisy_descriptor import descriptor_member_to_dict

        return descriptor_member_to_dict(descriptor_set, self)


@dataclass(frozen=True)
class NoisyPartitionDescriptor:
    partition_index: int
    local_to_global_qbits: tuple[int, ...]
    members: tuple[NoisyPartitionDescriptorMember, ...]

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def partition_qubit_span(self) -> tuple[int, ...]:
        """Legacy compatibility name: same tuple as ``local_to_global_qbits`` (global indices)."""
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

    def to_dict(self, descriptor_set: NoisyPartitionDescriptorSet) -> dict[str, Any]:
        from squander.partitioning.noisy_descriptor import descriptor_partition_to_dict

        return descriptor_partition_to_dict(descriptor_set, self)


@dataclass(frozen=True)
class NoisyPartitionDescriptorSet:
    requested_mode: str
    source_type: str
    workload_id: str
    qbit_num: int
    parameter_count: int
    max_partition_qubits: int
    # Ordered canonical operations (same contract as CanonicalNoisyPlannerSurface.operations).
    operations: tuple[CanonicalNoisyPlannerOperation, ...]
    partitions: tuple[NoisyPartitionDescriptor, ...]

    def canonical_operation_for(
        self, member: NoisyPartitionDescriptorMember
    ) -> CanonicalNoisyPlannerOperation:
        return self.operations[member.canonical_operation_index]

    def partition_gate_count(self, partition: NoisyPartitionDescriptor) -> int:
        return sum(
            1
            for m in partition.members
            if self.canonical_operation_for(m).kind == PLANNER_OP_KIND_GATE
        )

    def partition_noise_count(self, partition: NoisyPartitionDescriptor) -> int:
        return sum(
            1
            for m in partition.members
            if self.canonical_operation_for(m).kind == PLANNER_OP_KIND_NOISE
        )

    def partition_parameter_count(self, partition: NoisyPartitionDescriptor) -> int:
        return sum(
            self.canonical_operation_for(m).param_count for m in partition.members
        )

    def partition_canonical_operation_indices(
        self, partition: NoisyPartitionDescriptor
    ) -> tuple[int, ...]:
        return tuple(m.canonical_operation_index for m in partition.members)

    def partition_parameter_routing(
        self, partition: NoisyPartitionDescriptor
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (
                self.canonical_operation_for(m).param_start,
                m.local_param_start,
                self.canonical_operation_for(m).param_count,
            )
            for m in partition.members
            if self.canonical_operation_for(m).param_count > 0
        )

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
        return sum(self.partition_gate_count(p) for p in self.partitions)

    @property
    def noise_count(self) -> int:
        return sum(self.partition_noise_count(p) for p in self.partitions)

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
            "partitions": [p.to_dict(self) for p in self.partitions],
        }
