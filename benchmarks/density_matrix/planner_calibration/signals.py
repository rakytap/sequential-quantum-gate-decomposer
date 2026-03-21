from __future__ import annotations

from collections import defaultdict
from statistics import median

from squander.partitioning.noisy_runtime import PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED

PLANNER_CALIBRATION_DENSITY_SIGNAL_SCHEMA_VERSION = "phase3_planner_calibration_density_signal_v1"
PLANNER_CALIBRATION_STATE_VECTOR_PROXY_NAME = "state_vector_partition_proxy_v1"
PLANNER_CALIBRATION_DENSITY_AWARE_OBJECTIVE_NAME = "phase3_benchmark_cost_v1"


def build_state_vector_proxy_score(descriptor_set) -> float:
    descriptor_member_count = max(descriptor_set.descriptor_member_count, 1)
    return sum(
        (2 ** len(partition.partition_qubit_span)) * partition.member_count
        for partition in descriptor_set.partitions
    ) / descriptor_member_count


def build_density_partition_cost(descriptor_set) -> float:
    descriptor_member_count = max(descriptor_set.descriptor_member_count, 1)
    return sum(
        (4 ** len(partition.partition_qubit_span)) * partition.member_count
        for partition in descriptor_set.partitions
    ) / descriptor_member_count


def count_eligible_unitary_regions(runtime_result) -> int:
    return sum(
        region.classification == PHASE3_FUSION_CLASS_SUPPORTED_UNFUSED
        and region.reason == "fusion_disabled"
        for region in runtime_result.fused_regions
    )


def build_density_signal_record(
    metadata: dict,
    descriptor_set,
    runtime_result,
    density_metrics: dict,
) -> dict:
    peak_rss_mb = runtime_result.peak_rss_kb / 1024.0
    descriptor_member_count = max(descriptor_set.descriptor_member_count, 1)
    partition_count = max(descriptor_set.partition_count, 1)
    return {
        "signal_schema_version": PLANNER_CALIBRATION_DENSITY_SIGNAL_SCHEMA_VERSION,
        "candidate_id": metadata["candidate_id"],
        "planner_family": metadata["planner_family"],
        "planner_variant": metadata["planner_variant"],
        "max_partition_qubits": metadata["max_partition_qubits"],
        "case_kind": metadata["case_kind"],
        "workload_id": metadata["workload_id"],
        "workload_family": metadata["workload_family"],
        "qbit_num": metadata["qbit_num"],
        "family_name": metadata["family_name"],
        "noise_pattern": metadata["noise_pattern"],
        "planning_time_ms": metadata["planning_time_ms"],
        "runtime_path": runtime_result.runtime_path,
        "runtime_ms": runtime_result.runtime_ms,
        "peak_rss_kb": runtime_result.peak_rss_kb,
        "peak_rss_mb": peak_rss_mb,
        "partition_count": descriptor_set.partition_count,
        "max_partition_span": descriptor_set.max_partition_span,
        "descriptor_member_count": descriptor_set.descriptor_member_count,
        "noise_count": descriptor_set.noise_count,
        "noise_density": descriptor_set.noise_count / descriptor_member_count,
        "density_partition_cost": build_density_partition_cost(descriptor_set),
        "eligible_unitary_region_count": count_eligible_unitary_regions(runtime_result),
        "eligible_unitary_region_fraction": (
            count_eligible_unitary_regions(runtime_result) / partition_count
        ),
        "supported_unfused_region_count": runtime_result.supported_unfused_region_count,
        "deferred_region_count": runtime_result.deferred_region_count,
        "actual_fused_execution": runtime_result.actual_fused_execution,
        "state_vector_proxy_name": PLANNER_CALIBRATION_STATE_VECTOR_PROXY_NAME,
        "state_vector_proxy_score": build_state_vector_proxy_score(descriptor_set),
        "frobenius_norm_diff": density_metrics["frobenius_norm_diff"],
        "max_abs_diff": density_metrics["max_abs_diff"],
    }


def calibrate_memory_weight(records: list[dict]) -> float:
    peak_rss_mb_values = [record["peak_rss_mb"] for record in records if record["peak_rss_mb"] > 0.0]
    runtime_values = [record["runtime_ms"] for record in records if record["runtime_ms"] > 0.0]
    if not peak_rss_mb_values or not runtime_values:
        return 0.0
    return float(median(runtime_values) / median(peak_rss_mb_values))


def build_density_aware_score(record: dict, *, memory_weight: float) -> float:
    return float(
        record["runtime_ms"]
        + record["planning_time_ms"]
        + memory_weight * record["peak_rss_mb"]
    )


def apply_density_scores_and_rankings(
    records: list[dict], *, objective_name: str = PLANNER_CALIBRATION_DENSITY_AWARE_OBJECTIVE_NAME
) -> list[dict]:
    calibrated_memory_weight = calibrate_memory_weight(records)
    for record in records:
        record["density_aware_objective_name"] = objective_name
        record["calibrated_memory_weight"] = calibrated_memory_weight
        record["density_aware_score"] = build_density_aware_score(
            record, memory_weight=calibrated_memory_weight
        )

    grouped_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped_records[record["workload_id"]].append(record)

    for workload_records in grouped_records.values():
        density_sorted = sorted(
            workload_records,
            key=lambda record: (record["density_aware_score"], record["candidate_id"]),
        )
        for index, record in enumerate(density_sorted, start=1):
            record["density_aware_rank"] = index

        proxy_sorted = sorted(
            workload_records,
            key=lambda record: (record["state_vector_proxy_score"], record["candidate_id"]),
        )
        for index, record in enumerate(proxy_sorted, start=1):
            record["state_vector_proxy_rank"] = index

    return records
