from __future__ import annotations

import random
from typing import Iterable

from squander.partitioning.noisy_planner import (
    PHASE3_ENTRY_ROUTE_MICROCASE,
    PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY,
    PHASE3_WORKLOAD_FAMILY_MICROCASE,
    PHASE3_WORKLOAD_FAMILY_STRUCTURED,
    build_canonical_planner_surface_from_operation_specs,
)


MICROCASE_QUBITS = (2, 3, 4)
STRUCTURED_QUBITS = (8, 10)
MANDATORY_NOISE_PATTERNS = ("sparse", "periodic", "dense")
STRUCTURED_FAMILY_NAMES = (
    "layered_nearest_neighbor",
    "seeded_random_layered",
    "partition_stress_ladder",
)
DEFAULT_STRUCTURED_SEED = 20260318


def _u3(target_qbit: int) -> dict:
    return {
        "kind": "gate",
        "name": "U3",
        "target_qbit": target_qbit,
        "param_count": 3,
    }


def _cnot(target_qbit: int, control_qbit: int) -> dict:
    return {
        "kind": "gate",
        "name": "CNOT",
        "target_qbit": target_qbit,
        "control_qbit": control_qbit,
        "param_count": 0,
    }


def _noise(channel: str, target_qbit: int, after_gate_index: int, fixed_value: float) -> dict:
    return {
        "kind": "noise",
        "name": channel,
        "target_qbit": target_qbit,
        "source_gate_index": after_gate_index,
        "fixed_value": fixed_value,
        "param_count": 0,
    }


def _noise_value(channel: str) -> float:
    return {
        "local_depolarizing": 0.1,
        "amplitude_damping": 0.05,
        "phase_damping": 0.07,
    }[channel]


def mandatory_microcase_definitions() -> tuple[dict, ...]:
    return (
        {
            "case_name": "microcase_2q_entangler_local_depolarizing",
            "qbit_num": 2,
            "noise_pattern": "sparse",
            "operation_specs": [
                _u3(0),
                _u3(1),
                _cnot(1, 0),
                _noise("local_depolarizing", 1, 2, _noise_value("local_depolarizing")),
                _u3(0),
            ],
        },
        {
            "case_name": "microcase_3q_mixed_local_noise_sequence",
            "qbit_num": 3,
            "noise_pattern": "periodic",
            "operation_specs": [
                _u3(0),
                _u3(1),
                _cnot(1, 0),
                _noise("amplitude_damping", 1, 2, _noise_value("amplitude_damping")),
                _u3(2),
                _cnot(2, 1),
                _noise("phase_damping", 2, 4, _noise_value("phase_damping")),
                _u3(0),
            ],
        },
        {
            "case_name": "microcase_4q_partition_boundary_triplet",
            "qbit_num": 4,
            "noise_pattern": "dense",
            "operation_specs": [
                _u3(0),
                _u3(1),
                _cnot(1, 0),
                _noise("local_depolarizing", 0, 2, _noise_value("local_depolarizing")),
                _u3(2),
                _u3(3),
                _cnot(3, 2),
                _noise("phase_damping", 2, 5, _noise_value("phase_damping")),
                _cnot(2, 1),
                _noise("amplitude_damping", 1, 6, _noise_value("amplitude_damping")),
            ],
        },
    )


def build_story2_microcase_surface(case_name: str):
    case = next(
        candidate
        for candidate in mandatory_microcase_definitions()
        if candidate["case_name"] == case_name
    )
    return build_canonical_planner_surface_from_operation_specs(
        qbit_num=case["qbit_num"],
        source_type="microcase_builder",
        entry_route=PHASE3_ENTRY_ROUTE_MICROCASE,
        workload_family=PHASE3_WORKLOAD_FAMILY_MICROCASE,
        workload_id=case["case_name"],
        operation_specs=case["operation_specs"],
    )


def iter_story2_microcase_surfaces():
    for case in mandatory_microcase_definitions():
        yield case, build_story2_microcase_surface(case["case_name"])


def _layered_nearest_neighbor_layers(qbit_num: int, layer_count: int) -> list[list[dict]]:
    layers: list[list[dict]] = []
    for layer_index in range(layer_count):
        layer_specs = [_u3(target) for target in range(qbit_num)]
        start = layer_index % 2
        for control in range(start, qbit_num - 1, 2):
            layer_specs.append(_cnot(control + 1, control))
        layers.append(layer_specs)
    return layers


def _seeded_random_layered_layers(
    qbit_num: int, layer_count: int, seed: int
) -> list[list[dict]]:
    rng = random.Random(seed)
    layers: list[list[dict]] = []
    for layer_index in range(layer_count):
        layer_specs = [_u3(target) for target in range(qbit_num)]
        start = layer_index % 2
        for left in range(start, qbit_num - 1, 2):
            if rng.random() < 0.5:
                layer_specs.append(_cnot(left + 1, left))
            else:
                layer_specs.append(_cnot(left, left + 1))
        layers.append(layer_specs)
    return layers


def _partition_stress_layers(qbit_num: int, layer_count: int) -> list[list[dict]]:
    layers: list[list[dict]] = []
    for layer_index in range(layer_count):
        layer_specs = [_u3(target) for target in range(qbit_num)]
        if layer_index % 2 == 0:
            for control in range(qbit_num - 1):
                layer_specs.append(_cnot(control + 1, control))
        else:
            for target in range(qbit_num - 2, -1, -1):
                layer_specs.append(_cnot(target, target + 1))
        layers.append(layer_specs)
    return layers


def _apply_noise_pattern(
    layers: Iterable[list[dict]], qbit_num: int, noise_pattern: str
) -> list[dict]:
    operations: list[dict] = []
    gate_index = -1
    entangler_counter = 0
    channels = ("local_depolarizing", "amplitude_damping", "phase_damping")

    for layer_index, layer in enumerate(layers):
        layer_entanglers: list[tuple[int, dict]] = []
        for gate in layer:
            operations.append(gate)
            gate_index += 1
            if gate["name"] == "CNOT":
                layer_entanglers.append((gate_index, gate))

        if not layer_entanglers:
            continue

        if noise_pattern == "sparse":
            if layer_index % 2 == 0:
                gate_idx, gate = layer_entanglers[0]
                channel = channels[layer_index % len(channels)]
                operations.append(
                    _noise(channel, gate["target_qbit"], gate_idx, _noise_value(channel))
                )
        elif noise_pattern == "periodic":
            gate_idx, gate = layer_entanglers[-1]
            channel = channels[layer_index % len(channels)]
            operations.append(
                _noise(channel, gate["target_qbit"], gate_idx, _noise_value(channel))
            )
        elif noise_pattern == "dense":
            for gate_idx, gate in layer_entanglers:
                channel = channels[entangler_counter % len(channels)]
                operations.append(
                    _noise(channel, gate["target_qbit"], gate_idx, _noise_value(channel))
                )
                entangler_counter += 1
            continue
        else:
            raise ValueError("Unsupported Story 2 noise pattern '{}'".format(noise_pattern))

        entangler_counter += len(layer_entanglers)

    return operations


def _structured_family_layers(
    family_name: str,
    qbit_num: int,
    *,
    seed: int,
) -> list[list[dict]]:
    layer_count = 3 if qbit_num <= 8 else 4
    if family_name == "layered_nearest_neighbor":
        return _layered_nearest_neighbor_layers(qbit_num, layer_count)
    if family_name == "seeded_random_layered":
        return _seeded_random_layered_layers(qbit_num, layer_count, seed)
    if family_name == "partition_stress_ladder":
        return _partition_stress_layers(qbit_num, layer_count)
    raise ValueError("Unsupported Story 2 family '{}'".format(family_name))


def build_story2_structured_surface(
    family_name: str,
    *,
    qbit_num: int,
    noise_pattern: str,
    seed: int = DEFAULT_STRUCTURED_SEED,
):
    layers = _structured_family_layers(family_name, qbit_num, seed=seed)
    operation_specs = _apply_noise_pattern(layers, qbit_num, noise_pattern)
    workload_id = "{}_q{}_{}_seed{}".format(family_name, qbit_num, noise_pattern, seed)
    return build_canonical_planner_surface_from_operation_specs(
        qbit_num=qbit_num,
        source_type="structured_family_builder",
        entry_route=PHASE3_ENTRY_ROUTE_STRUCTURED_FAMILY,
        workload_family=PHASE3_WORKLOAD_FAMILY_STRUCTURED,
        workload_id=workload_id,
        operation_specs=operation_specs,
    )


def iter_story2_structured_surfaces(seed: int = DEFAULT_STRUCTURED_SEED):
    for family_name in STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in MANDATORY_NOISE_PATTERNS:
                metadata = {
                    "family_name": family_name,
                    "qbit_num": qbit_num,
                    "noise_pattern": noise_pattern,
                    "seed": seed,
                    "workload_id": "{}_q{}_{}_seed{}".format(
                        family_name, qbit_num, noise_pattern, seed
                    ),
                }
                yield metadata, build_story2_structured_surface(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=seed,
                )
