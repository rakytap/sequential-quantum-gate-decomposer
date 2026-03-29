from __future__ import annotations

import random
from typing import Iterable

from squander.partitioning.noisy_planner import (
    DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
    build_partition_descriptor_set,
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
PHASE31_PRIMARY_STRUCTURED_FAMILY_NAMES = (
    "phase31_pair_repeat",
    "phase31_alternating_ladder",
)
PHASE31_CONTROL_STRUCTURED_FAMILY_NAMES = ("layered_nearest_neighbor",)
PHASE31_PRIMARY_NOISE_PATTERNS = ("periodic", "dense")
PHASE31_CONTROL_NOISE_PATTERNS = ("sparse",)
PHASE31_PRIMARY_SEEDS = (
    DEFAULT_STRUCTURED_SEED,
    DEFAULT_STRUCTURED_SEED + 1,
    DEFAULT_STRUCTURED_SEED + 2,
)


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


def phase31_microcase_definitions() -> tuple[dict, ...]:
    return (
        {
            "case_name": "phase31_microcase_1q_u3_local_noise_chain",
            "qbit_num": 1,
            "noise_pattern": "dense",
            "support_qbits": (0,),
            "operation_specs": [
                _u3(0),
                _noise("local_depolarizing", 0, 0, _noise_value("local_depolarizing")),
                _u3(0),
                _noise("phase_damping", 0, 1, _noise_value("phase_damping")),
                _u3(0),
            ],
        },
        {
            "case_name": "phase31_microcase_2q_cnot_local_noise_pair",
            "qbit_num": 2,
            "noise_pattern": "periodic",
            "support_qbits": (0, 1),
            "operation_specs": [
                _u3(0),
                _u3(1),
                _cnot(1, 0),
                _noise("amplitude_damping", 1, 2, _noise_value("amplitude_damping")),
                _noise("phase_damping", 0, 2, _noise_value("phase_damping")),
                _u3(0),
            ],
        },
        {
            "case_name": "phase31_microcase_2q_multi_noise_entangler_chain",
            "qbit_num": 2,
            "noise_pattern": "periodic",
            "support_qbits": (0, 1),
            "operation_specs": [
                _u3(0),
                _u3(1),
                _cnot(1, 0),
                _noise("local_depolarizing", 1, 2, _noise_value("local_depolarizing")),
                _u3(1),
                _cnot(0, 1),
                _noise("amplitude_damping", 0, 4, _noise_value("amplitude_damping")),
                _u3(0),
                _noise("phase_damping", 1, 5, _noise_value("phase_damping")),
            ],
        },
        {
            "case_name": "phase31_microcase_2q_dense_same_support_motif",
            "qbit_num": 2,
            "noise_pattern": "dense",
            "support_qbits": (0, 1),
            "operation_specs": [
                _u3(0),
                _u3(1),
                _cnot(1, 0),
                _noise("local_depolarizing", 1, 2, _noise_value("local_depolarizing")),
                _cnot(0, 1),
                _noise("amplitude_damping", 0, 3, _noise_value("amplitude_damping")),
                _u3(0),
                _u3(1),
                _noise("phase_damping", 1, 5, _noise_value("phase_damping")),
            ],
        },
    )


def _lookup_microcase_definition(case_name: str) -> dict:
    return next(
        candidate
        for candidate in (*mandatory_microcase_definitions(), *phase31_microcase_definitions())
        if candidate["case_name"] == case_name
    )


def build_microcase_surface(case_name: str):
    case = _lookup_microcase_definition(case_name)
    return build_canonical_planner_surface_from_operation_specs(
        qbit_num=case["qbit_num"],
        source_type="microcase_builder",
        workload_id=case["case_name"],
        operation_specs=case["operation_specs"],
    )


def iter_microcase_surfaces():
    for case in mandatory_microcase_definitions():
        yield case, build_microcase_surface(case["case_name"])


def iter_phase31_microcase_surfaces():
    for case in phase31_microcase_definitions():
        yield case, build_microcase_surface(case["case_name"])


def build_microcase_descriptor_set(
    case_name: str,
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    return build_partition_descriptor_set(
        build_microcase_surface(case_name),
        max_partition_qubits=max_partition_qubits,
    )


def iter_microcase_descriptor_sets(
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    for case in mandatory_microcase_definitions():
        yield case, build_microcase_descriptor_set(
            case["case_name"], max_partition_qubits=max_partition_qubits
        )


def build_phase31_microcase_descriptor_set(
    case_name: str,
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    return build_microcase_descriptor_set(
        case_name,
        max_partition_qubits=max_partition_qubits,
    )


def iter_phase31_microcase_descriptor_sets(
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    for case in phase31_microcase_definitions():
        yield case, build_phase31_microcase_descriptor_set(
            case["case_name"], max_partition_qubits=max_partition_qubits
        )


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


def _phase31_pair_repeat_layers(qbit_num: int, layer_count: int) -> list[list[dict]]:
    layers: list[list[dict]] = []
    for layer_index in range(layer_count):
        layer_specs = [_u3(target) for target in range(qbit_num)]
        start = layer_index % 2
        for left in range(start, qbit_num - 1, 2):
            layer_specs.append(_cnot(left + 1, left))
            layer_specs.append(_u3(left))
            layer_specs.append(_u3(left + 1))
            layer_specs.append(_cnot(left, left + 1))
        layers.append(layer_specs)
    return layers


def _phase31_alternating_ladder_layers(qbit_num: int, layer_count: int) -> list[list[dict]]:
    layers: list[list[dict]] = []
    for layer_index in range(layer_count):
        layer_specs = [_u3(target) for target in range(qbit_num)]
        if layer_index % 2 == 0:
            for left in range(qbit_num - 1):
                layer_specs.append(_cnot(left + 1, left))
            layer_specs.extend(_u3(target) for target in range(qbit_num))
            for left in range(qbit_num - 2, -1, -1):
                layer_specs.append(_cnot(left, left + 1))
        else:
            for left in range(qbit_num - 2, -1, -1):
                layer_specs.append(_cnot(left, left + 1))
            layer_specs.extend(_u3(target) for target in range(qbit_num))
            for left in range(qbit_num - 1):
                layer_specs.append(_cnot(left + 1, left))
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
            raise ValueError(
                "Unsupported structured workload noise pattern '{}'".format(noise_pattern)
            )

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
    if family_name == "phase31_pair_repeat":
        return _phase31_pair_repeat_layers(qbit_num, layer_count)
    if family_name == "phase31_alternating_ladder":
        return _phase31_alternating_ladder_layers(qbit_num, layer_count)
    raise ValueError("Unsupported structured workload family '{}'".format(family_name))


def build_structured_surface(
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
        workload_id=workload_id,
        operation_specs=operation_specs,
    )


def iter_structured_surfaces(seed: int = DEFAULT_STRUCTURED_SEED):
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
                yield metadata, build_structured_surface(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=seed,
                )


def build_structured_descriptor_set(
    family_name: str,
    *,
    qbit_num: int,
    noise_pattern: str,
    seed: int = DEFAULT_STRUCTURED_SEED,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    return build_partition_descriptor_set(
        build_structured_surface(
            family_name,
            qbit_num=qbit_num,
            noise_pattern=noise_pattern,
            seed=seed,
        ),
        max_partition_qubits=max_partition_qubits,
    )


def iter_structured_descriptor_sets(
    *,
    seed: int = DEFAULT_STRUCTURED_SEED,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
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
                yield metadata, build_structured_descriptor_set(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=seed,
                    max_partition_qubits=max_partition_qubits,
                )


def iter_phase31_structured_surfaces():
    for family_name in PHASE31_PRIMARY_STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in PHASE31_PRIMARY_NOISE_PATTERNS:
                for seed in PHASE31_PRIMARY_SEEDS:
                    metadata = {
                        "family_name": family_name,
                        "qbit_num": qbit_num,
                        "noise_pattern": noise_pattern,
                        "seed": seed,
                        "workload_id": "{}_q{}_{}_seed{}".format(
                            family_name, qbit_num, noise_pattern, seed
                        ),
                    }
                    yield metadata, build_structured_surface(
                        family_name,
                        qbit_num=qbit_num,
                        noise_pattern=noise_pattern,
                        seed=seed,
                    )
    for family_name in PHASE31_CONTROL_STRUCTURED_FAMILY_NAMES:
        for qbit_num in STRUCTURED_QUBITS:
            for noise_pattern in PHASE31_CONTROL_NOISE_PATTERNS:
                metadata = {
                    "family_name": family_name,
                    "qbit_num": qbit_num,
                    "noise_pattern": noise_pattern,
                    "seed": DEFAULT_STRUCTURED_SEED,
                    "workload_id": "{}_q{}_{}_seed{}".format(
                        family_name, qbit_num, noise_pattern, DEFAULT_STRUCTURED_SEED
                    ),
                }
                yield metadata, build_structured_surface(
                    family_name,
                    qbit_num=qbit_num,
                    noise_pattern=noise_pattern,
                    seed=DEFAULT_STRUCTURED_SEED,
                )


def build_phase31_structured_descriptor_set(
    family_name: str,
    *,
    qbit_num: int,
    noise_pattern: str,
    seed: int = DEFAULT_STRUCTURED_SEED,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    return build_structured_descriptor_set(
        family_name,
        qbit_num=qbit_num,
        noise_pattern=noise_pattern,
        seed=seed,
        max_partition_qubits=max_partition_qubits,
    )


def iter_phase31_structured_descriptor_sets(
    *,
    max_partition_qubits: int = DEFAULT_PARTITION_DESCRIPTOR_MAX_QUBITS,
):
    for metadata, _surface in iter_phase31_structured_surfaces():
        yield metadata, build_phase31_structured_descriptor_set(
            metadata["family_name"],
            qbit_num=metadata["qbit_num"],
            noise_pattern=metadata["noise_pattern"],
            seed=metadata["seed"],
            max_partition_qubits=max_partition_qubits,
        )
