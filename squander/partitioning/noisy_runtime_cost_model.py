"""Pure cost-model features and routing decisions for the Phase 3.1 hybrid path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from squander.partitioning.noisy_types import (
    PLANNER_OP_KIND_GATE,
    NoisyPartitionDescriptor,
    NoisyPartitionDescriptorSet,
)

CostDecision = Literal["fuse_channel_native", "skip_to_phase3"]

COST_MODEL_ROUTE_FUSE = "eligible_channel_native_motif"
COST_MODEL_ROUTE_SKIP = "cost_model_skip_kraus_expansion"
PHASE31_KRAUS_EXPANSION_MODEL_ID = "phase31_kraus_expansion_v0"
PHASE31_KRAUS_EXPANSION_MODEL_VERSION = "0.1.0"

# Rank upper bounds match the Kraus bundles built by noisy_runtime_channel_native.
_OPERATION_KRAUS_RANK_UPPER = {
    "U3": 1,
    "CNOT": 1,
    "local_depolarizing": 4,
    "dep": 4,
    "amplitude_damping": 2,
    "AD": 2,
    "phase_damping": 2,
    "PD": 2,
}


@dataclass(frozen=True)
class MotifCostFeatures:
    """Pure motif features used by the versioned Kraus-expansion model."""

    support_qubit_count: int
    motif_length: int
    gate_count: int
    noise_count: int
    predicted_kraus_count_upper: int
    predicted_sequential_kraus_count_sum: int
    qbit_num: int
    reuse_hint: int | None = None


@dataclass(frozen=True)
class CostModelDecision:
    """Auditable decision and predictions in abstract apply-cost units."""

    decision: CostDecision
    route_reason: str
    predicted_kraus_count_upper: int
    predicted_apply_cost: float
    predicted_baseline_cost: float
    model_id: str
    model_version: str


class ChannelNativeCostModel(Protocol):
    """Decision interface for eligible channel-native motifs."""

    model_id: str
    version: str

    def decide(self, features: MotifCostFeatures) -> CostModelDecision:
        ...


def extract_motif_cost_features(
    descriptor_set: NoisyPartitionDescriptorSet,
    partition: NoisyPartitionDescriptor,
    local_support: tuple[int, ...],
) -> MotifCostFeatures:
    """Extract rank bounds without constructing or composing Kraus bundles."""

    ranks: list[int] = []
    gate_count = 0
    for member in partition.members:
        operation = descriptor_set.canonical_operation_for(member)
        try:
            ranks.append(_OPERATION_KRAUS_RANK_UPPER[operation.name])
        except KeyError as exc:
            raise ValueError(
                "No channel-native Kraus-rank bound for operation '{}'".format(
                    operation.name
                )
            ) from exc
        if operation.kind == PLANNER_OP_KIND_GATE:
            gate_count += 1

    predicted_kraus_count_upper = 1
    for rank in ranks:
        predicted_kraus_count_upper *= rank

    return MotifCostFeatures(
        support_qubit_count=len(local_support),
        motif_length=len(partition.members),
        gate_count=gate_count,
        noise_count=len(partition.members) - gate_count,
        predicted_kraus_count_upper=predicted_kraus_count_upper,
        predicted_sequential_kraus_count_sum=sum(ranks),
        qbit_num=descriptor_set.qbit_num,
    )


@dataclass(frozen=True)
class Phase31KrausExpansionCostModelV0:
    """v0 gate: compare composed and sequential work in identical apply units."""

    model_id: str = PHASE31_KRAUS_EXPANSION_MODEL_ID
    version: str = PHASE31_KRAUS_EXPANSION_MODEL_VERSION

    def decide(self, features: MotifCostFeatures) -> CostModelDecision:
        # One abstract apply unit is 2^(2n) * d, where n is global width and
        # d=2^|S_M|. The same unit intentionally appears on both sides.
        apply_unit = float(
            (1 << (2 * features.qbit_num))
            * (1 << features.support_qubit_count)
        )
        predicted_apply_cost = (
            features.predicted_kraus_count_upper * apply_unit
        )
        predicted_baseline_cost = (
            features.predicted_sequential_kraus_count_sum * apply_unit
        )
        skip = predicted_apply_cost > predicted_baseline_cost
        return CostModelDecision(
            decision="skip_to_phase3" if skip else "fuse_channel_native",
            route_reason=COST_MODEL_ROUTE_SKIP if skip else COST_MODEL_ROUTE_FUSE,
            predicted_kraus_count_upper=features.predicted_kraus_count_upper,
            predicted_apply_cost=predicted_apply_cost,
            predicted_baseline_cost=predicted_baseline_cost,
            model_id=self.model_id,
            model_version=self.version,
        )


DEFAULT_CHANNEL_NATIVE_COST_MODEL = Phase31KrausExpansionCostModelV0()
