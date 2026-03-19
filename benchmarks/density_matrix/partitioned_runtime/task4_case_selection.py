#!/usr/bin/env python3
"""Shared Task 4 representative case iterators."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
from benchmarks.density_matrix.planner_surface.common import build_phase3_story1_continuity_vqe
from benchmarks.density_matrix.planner_surface.workloads import (
    iter_story2_microcase_descriptor_sets,
    iter_story2_structured_descriptor_sets,
)
from squander.partitioning.noisy_planner import build_phase3_continuity_partition_descriptor_set

CONTINUITY_QUBITS = (4, 6, 8, 10)


def iter_task4_continuity_cases():
    for qbit_num in CONTINUITY_QUBITS:
        vqe, hamiltonian, topology = build_phase3_story1_continuity_vqe(qbit_num)
        descriptor_set = build_phase3_continuity_partition_descriptor_set(vqe)
        parameters = build_initial_parameters(vqe.get_Parameter_Num())
        metadata = {
            "case_name": f"phase2_xxz_hea_q{qbit_num}_continuity",
            "case_kind": "continuity",
            "qbit_num": qbit_num,
            "topology": topology,
            "continuity_energy_real": float(vqe.Optimization_Problem(parameters)),
        }
        yield metadata, descriptor_set, parameters, hamiltonian


def iter_task4_microcase_cases():
    for metadata, descriptor_set in iter_story2_microcase_descriptor_sets():
        case_metadata = dict(metadata)
        case_metadata["case_kind"] = "microcase"
        yield case_metadata, descriptor_set, build_initial_parameters(
            descriptor_set.parameter_count
        )


def iter_task4_structured_cases():
    for metadata, descriptor_set in iter_story2_structured_descriptor_sets():
        case_metadata = dict(metadata)
        case_metadata["case_kind"] = "structured_family"
        yield case_metadata, descriptor_set, build_initial_parameters(
            descriptor_set.parameter_count
        )
