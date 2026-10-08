"""M-F5a task-1: density interop harness (module-level, no new energy API)."""

from __future__ import annotations

import numpy as np
import pytest

from squander.VQA.qgd_Variational_Quantum_Eigensolver_Base import (
    qgd_Variational_Quantum_Eigensolver_Base as VariationalQuantumEigensolver,
)
import squander.VQA.qgd_Variational_Quantum_Eigensolver_Base_Wrapper as vqe_wrapper_ext

import tests.VQE.test_VQE as vqe_test_module

# Host-pinned at cdcfe6b1 on rocky-local (ET-2); not retuned without Reviewer (N-17).
ET2_GOLDEN_ENERGY_CDCFE6B1 = -0.7583303034656004


@pytest.fixture
def density_vqe_4q():
    vqe, _ = vqe_test_module.Test_VQE()._build_density_backend_vqe(4)
    return vqe


def test_harness_symbols_live_on_extension_module_only(density_vqe_4q):
    for name in (
        "harness_density_lower_ns",
        "harness_density_set_timer_flag",
        "harness_density_subtimes_ns",
    ):
        assert hasattr(vqe_wrapper_ext, name)
        assert not hasattr(density_vqe_4q, name)


def test_harness_lower_ns_returns_positive_int64(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)
    elapsed_ns = vqe_wrapper_ext.harness_density_lower_ns(density_vqe_4q, parameters)
    assert isinstance(elapsed_ns, int)
    assert elapsed_ns > 0


def test_harness_subtimes_returns_six_int64s(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)
    vqe_wrapper_ext.harness_density_set_timer_flag(density_vqe_4q, True)
    vqe_wrapper_ext.harness_density_lower_ns(density_vqe_4q, parameters)
    subtimes = vqe_wrapper_ext.harness_density_subtimes_ns(density_vqe_4q)
    assert len(subtimes) == 6
    assert all(isinstance(value, int) for value in subtimes)
    assert sum(subtimes) > 0


def test_density_energy_bit_identical_with_timer_flag(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)

    vqe_wrapper_ext.harness_density_set_timer_flag(density_vqe_4q, False)
    energy_flag_off = float(density_vqe_4q.Optimization_Problem(parameters))

    vqe_wrapper_ext.harness_density_set_timer_flag(density_vqe_4q, True)
    energy_flag_on = float(density_vqe_4q.Optimization_Problem(parameters))

    assert energy_flag_off == energy_flag_on
    assert energy_flag_off == ET2_GOLDEN_ENERGY_CDCFE6B1


def test_harness_neither_path_returns_energy(density_vqe_4q):
    parameters = np.linspace(0.05, 0.05 * 18, 18, dtype=np.float64)
    lower_ns = vqe_wrapper_ext.harness_density_lower_ns(density_vqe_4q, parameters)
    subtimes = vqe_wrapper_ext.harness_density_subtimes_ns(density_vqe_4q)
    assert not isinstance(lower_ns, float)
    assert not any(isinstance(value, float) for value in subtimes)


def test_task2_evqe_6q_cell_pins_timer_identity_and_aer_oracle():
    pytest.importorskip("qiskit_aer")
    from benchmarks.density_matrix.interop_profile.interop_lane import build_task_evaluator

    vqe, hamiltonian = build_task_evaluator(6)
    parameters = np.linspace(0.05, 0.05 * 30, 30, dtype=np.float64)

    bridge = vqe.describe_density_bridge()
    assert vqe.get_Parameter_Num() == 30
    assert bridge["operation_count"] == 18
    assert bridge["gate_count"] == 15
    assert bridge["noise_count"] == 3
    assert int(hamiltonian.nnz) == 224

    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, False)
    energy_flag_off = float(vqe.Optimization_Problem(parameters))
    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, True)
    energy_flag_on = float(vqe.Optimization_Problem(parameters))
    assert energy_flag_off == energy_flag_on

    for name in (
        "harness_density_lower_ns",
        "harness_density_set_timer_flag",
        "harness_density_subtimes_ns",
    ):
        assert not hasattr(vqe, name)

    vqe.set_Optimized_Parameters(parameters)
    tester = vqe_test_module.Test_VQE()
    aer_real, aer_imag = tester._get_density_backend_aer_reference(vqe, hamiltonian)
    bound = 1e-12 + 1e-5 * abs(aer_real)
    assert abs(energy_flag_off - aer_real) <= bound
    assert abs(aer_imag) <= 1e-12


def test_task3_evqe_8q_cell_pins_timer_identity_and_aer_oracle():
    pytest.importorskip("qiskit_aer")
    from benchmarks.density_matrix.interop_profile.interop_lane import build_task_evaluator

    vqe, hamiltonian = build_task_evaluator(8)
    parameters = np.linspace(0.05, 0.05 * 42, 42, dtype=np.float64)

    bridge = vqe.describe_density_bridge()
    assert vqe.get_Parameter_Num() == 42
    assert bridge["operation_count"] == 24
    assert bridge["gate_count"] == 21
    assert bridge["noise_count"] == 3
    assert int(hamiltonian.nnz) == 1152

    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, False)
    energy_flag_off = float(vqe.Optimization_Problem(parameters))
    vqe_wrapper_ext.harness_density_set_timer_flag(vqe, True)
    energy_flag_on = float(vqe.Optimization_Problem(parameters))
    assert energy_flag_off == energy_flag_on

    for name in (
        "harness_density_lower_ns",
        "harness_density_set_timer_flag",
        "harness_density_subtimes_ns",
    ):
        assert not hasattr(vqe, name)

    vqe.set_Optimized_Parameters(parameters)
    tester = vqe_test_module.Test_VQE()
    aer_real, aer_imag = tester._get_density_backend_aer_reference(vqe, hamiltonian)
    bound = 1e-12 + 1e-5 * abs(aer_real)
    assert abs(energy_flag_off - aer_real) <= bound
    assert abs(aer_imag) <= 1e-12


def test_task4_width4_attribution_descriptor_pins():
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_width4_attribution_anchor,
    )

    vqe, descriptor_set, bridge = build_width4_attribution_anchor()
    assert vqe.get_Parameter_Num() == 18
    assert bridge["operation_count"] == 12
    assert bridge["gate_count"] == 9
    assert bridge["noise_count"] == 3
    assert descriptor_set.workload_id == "phase2_xxz_hea_q4_continuity"


@pytest.mark.parametrize(
    "route_id",
    ("R-base", "R-fused", "R-hybrid"),
)
def test_task4_attribution_route_sample_has_orchestration_and_apply(route_id: str):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_route_row,
        build_width4_attribution_anchor,
    )
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters

    expected_primitive_calls = {"R-base": 8, "R-fused": 8, "R-hybrid": 5}

    vqe, descriptor_set, _bridge = build_width4_attribution_anchor()
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    row = build_route_row(
        route_id, descriptor_set, parameters, sample_count=2, throughput_divisor=3072
    )
    assert row["orchestration"]["mean_ns"] >= 0
    assert row["apply_component"]["mean_ns"] > 0
    assert row["throughput"]["divisor"] == 3072
    assert row["throughput"]["mean_ns_per_op"] == pytest.approx(
        row["apply_component"]["mean_ns"] / 3072, rel=1e-12
    )
    assert row["throughput"]["upper_bound_95_ns_per_op"] == pytest.approx(
        row["apply_component"]["upper_bound_95_ns"] / 3072, rel=1e-12
    )
    assert "mean_O" not in row
    assert "overhead" not in row
    for sample in row["samples"]:
        wall_ns = int(sample["orchestration_ns"]) + int(sample["apply_component_ns"])
        assert int(sample["apply_component_ns"]) < wall_ns
        assert int(sample["apply_primitive_calls"]) > 0
    assert {sample["apply_primitive_calls"] for sample in row["samples"]} == {
        expected_primitive_calls[route_id]
    }
    executor = lane.ROUTE_TABLE[route_id]["executor"]
    for _ in range(2):
        sample, result = lane._time_route_sample(
            route_id, executor, descriptor_set, parameters
        )
        assert sample["apply_component_ns"] < int(result.runtime_ms * 1_000_000)
    if route_id == "R-hybrid":
        assert row["partition_runtime_classes"] == [
            "phase31_channel_native",
            "phase3_unitary_island_fused",
            "phase31_channel_native",
            "phase3_unitary_island_fused",
            "phase3_unitary_island_fused",
        ]
        assert "the executed class" not in row["apply_label"]


@pytest.mark.parametrize(
    "width,params,ops,gates,noise,workload",
    [
        (6, 30, 18, 15, 3, "phase2_xxz_hea_q6_continuity"),
        (8, 42, 24, 21, 3, "phase2_xxz_hea_q8_continuity"),
    ],
)
def test_task5_attribution_anchor_counts(width, params, ops, gates, noise, workload):
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_attribution_anchor,
    )

    vqe, descriptor_set, bridge = build_attribution_anchor(width)
    assert vqe.get_Parameter_Num() == params
    assert int(bridge["operation_count"]) == ops
    assert int(bridge["gate_count"]) == gates
    assert int(bridge["noise_count"]) == noise
    assert descriptor_set.workload_id == workload


def test_task4_r_strict_handback_on_continuity_anchor():
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_r_strict_refusal_row,
        build_route_row,
        build_width4_attribution_anchor,
    )
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters

    vqe, descriptor_set, _bridge = build_width4_attribution_anchor()
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    with pytest.raises(ValueError, match="refusal row"):
        build_route_row(
            "R-strict", descriptor_set, parameters, sample_count=1, throughput_divisor=3072
        )
    refusal = build_r_strict_refusal_row(descriptor_set, parameters)
    assert refusal["status"] == "handback_refused"
    assert "98eec857" in refusal["reason"]
    assert "channel_native_noise_presence" in refusal["reason"]
    assert "samples" not in refusal


def test_attribution_routes_flag_refuses_counted_evqe_output_names(tmp_path, monkeypatch):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.interop_profile import validation_pipeline as vp

    def must_not_run():
        raise AssertionError("counted attribution run started before output guard")

    monkeypatch.setattr(lane, "run_counted_attribution_bundle", must_not_run)
    for name in (
        "interop_profile_bundle.json",
        "interop_profile_bundle_w6.json",
        "interop_profile_bundle_w8.json",
    ):
        out = tmp_path / name
        with pytest.raises(ValueError, match="counted E-VQE"):
            vp.resolve_interop_output_path(4, out, attribution_routes=True)
        with pytest.raises(ValueError, match="counted E-VQE"):
            vp.main(["--attribution-routes", "--width", "4", "--output", str(out)])


def test_evqe_path_refuses_routes_output_name(tmp_path):
    from benchmarks.density_matrix.interop_profile.validation_pipeline import (
        resolve_interop_output_path,
    )

    out = tmp_path / "interop_profile_bundle_routes_w4.json"
    with pytest.raises(ValueError, match="attribution route bundle"):
        resolve_interop_output_path(4, out, attribution_routes=False)


@pytest.mark.parametrize("width", [5, 10])
def test_attribution_routes_refuses_unsupported_width(tmp_path, width):
    from benchmarks.density_matrix.interop_profile.validation_pipeline import (
        resolve_interop_output_path,
    )

    out = tmp_path / "interop_profile_bundle_routes_w6.json"
    with pytest.raises(ValueError, match="attribution routes allow widths 4, 6, and 8 only"):
        resolve_interop_output_path(width, out, attribution_routes=True)


@pytest.mark.parametrize("width,artifact", [(6, "interop_profile_bundle_routes_w6.json"), (8, "interop_profile_bundle_routes_w8.json")])
def test_attribution_routes_accepts_width_six_and_eight_outside_artifacts(tmp_path, width, artifact):
    from benchmarks.density_matrix.interop_profile.validation_pipeline import (
        resolve_interop_output_path,
    )

    out = tmp_path / artifact
    resolved = resolve_interop_output_path(width, out, attribution_routes=True)
    assert resolved == out.resolve()


@pytest.mark.parametrize(
    "width,name",
    [
        (4, "interop_profile_bundle_routes_w6.json"),
        (4, "interop_profile_bundle_routes_w8.json"),
        (6, "interop_profile_bundle_routes_w4.json"),
        (6, "interop_profile_bundle_routes_w8.json"),
        (8, "interop_profile_bundle_routes_w4.json"),
        (8, "interop_profile_bundle_routes_w6.json"),
    ],
)
def test_attribution_routes_refuses_cross_width_routes_names(tmp_path, width, name):
    from benchmarks.density_matrix.interop_profile.validation_pipeline import (
        resolve_interop_output_path,
    )

    with pytest.raises(ValueError, match="must not write"):
        resolve_interop_output_path(width, tmp_path / name, attribution_routes=True)


@pytest.mark.parametrize("width", [4, 6, 8])
def test_attribution_routes_default_output_follows_width(width):
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        DEFAULT_ATTRIBUTION_ARTIFACT_DIR,
    )
    from benchmarks.density_matrix.interop_profile.validation_pipeline import (
        resolve_interop_output_path,
    )

    resolved = resolve_interop_output_path(width, None, attribution_routes=True)
    assert resolved == (
        DEFAULT_ATTRIBUTION_ARTIFACT_DIR / f"interop_profile_bundle_routes_w{width}.json"
    ).resolve()


def test_attribution_output_width_bound_or_outside_artifacts(tmp_path, monkeypatch):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.interop_profile import validation_pipeline as vp
    from benchmarks.density_matrix.interop_profile.attribution_route_validation import (
        DEFAULT_ATTRIBUTION_ARTIFACT_DIR,
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("counted attribution run started before output guard")

    monkeypatch.setattr(lane, "run_counted_attribution_bundle", must_not_run)
    artifact_dir = DEFAULT_ATTRIBUTION_ARTIFACT_DIR
    listing_before = sorted(path.name for path in artifact_dir.iterdir())
    bad_inside = artifact_dir / "custom_routes.json"
    for width in (4, 6, 8):
        with pytest.raises(ValueError, match="must write only"):
            vp.resolve_interop_output_path(width, bad_inside, attribution_routes=True)
        with pytest.raises(ValueError, match="must write only"):
            vp.resolve_interop_output_path(
                width, artifact_dir / "sub" / "custom_routes.json", attribution_routes=True
            )
        with pytest.raises(ValueError, match="must write only"):
            vp.resolve_interop_output_path(
                width, artifact_dir / "sub" / ".." / "custom_routes.json", attribution_routes=True
            )
        dotdot_from_tmp = type(tmp_path)(
            str(tmp_path) + "/.." * len(tmp_path.parts) + str(bad_inside)
        )
        with pytest.raises(ValueError, match="must write only"):
            vp.resolve_interop_output_path(width, dotdot_from_tmp, attribution_routes=True)
        symlink = tmp_path / f"via_symlink_w{width}.json"
        symlink.symlink_to(bad_inside)
        with pytest.raises(ValueError, match="must write only"):
            vp.resolve_interop_output_path(width, symlink, attribution_routes=True)
        good_outside = tmp_path / f"interop_profile_bundle_routes_w{width}.json"
        assert (
            vp.resolve_interop_output_path(width, good_outside, attribution_routes=True)
            == good_outside.resolve()
        )
    assert sorted(path.name for path in artifact_dir.iterdir()) == listing_before


def test_attribution_apply_timer_wrap_set_and_clock_reads_inside_with(monkeypatch):
    import sys
    import types

    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters
    import squander.partitioning.noisy_runtime_channel_native as channel_native_mod
    from squander.density_matrix import DensityMatrix, NoisyCircuit

    _vqe, descriptor_set, _bridge = lane.build_width4_attribution_anchor()
    parameters = build_initial_parameters(_vqe.get_Parameter_Num())

    def active_wraps() -> set[str]:
        active: set[str] = set()
        for owner, name in (
            (NoisyCircuit, "apply_to"),
            (DensityMatrix, "apply_local_unitary"),
            (channel_native_mod, "_apply_kraus_bundle"),
        ):
            fn = owner.__dict__[name]
            if getattr(fn, "__name__", "") == "wrapped":
                active.add(name)
        return active

    reads: list[set[str]] = []
    real_time = lane.time

    def perf_counter_ns():
        frame = sys._getframe(1)
        if frame.f_code.co_name == "_time_route_sample":
            reads.append(active_wraps())
        return real_time.perf_counter_ns()

    monkeypatch.setattr(lane, "time", types.SimpleNamespace(perf_counter_ns=perf_counter_ns))
    expected = {
        "R-base": {"apply_to"},
        "R-fused": {"apply_to", "apply_local_unitary"},
        "R-hybrid": {"apply_local_unitary", "_apply_kraus_bundle"},
    }
    for route_id, wrap_set in expected.items():
        reads.clear()
        lane._time_route_sample(
            route_id, lane.ROUTE_TABLE[route_id]["executor"], descriptor_set, parameters
        )
        assert reads == [wrap_set, wrap_set]


def test_task4_timed_route_raise_hands_back_without_stub_timings(monkeypatch):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters

    _vqe, descriptor_set, _bridge = lane.build_width4_attribution_anchor()
    parameters = build_initial_parameters(_vqe.get_Parameter_Num())
    monkeypatch.setitem(
        lane.ROUTE_TABLE["R-hybrid"],
        "executor",
        lane.ROUTE_TABLE["R-strict"]["executor"],
    )
    with pytest.raises(ValueError, match="attribution route handback for R-hybrid"):
        lane.build_route_row(
            "R-hybrid", descriptor_set, parameters, sample_count=2, throughput_divisor=3072
        )


def test_attribution_routes_pipeline_clean_start_false_writes_nothing(tmp_path, monkeypatch):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.interop_profile import validation_pipeline as vp

    def fake_bundle(_qbit_num=4):
        return {"clean_start": False, "rows": [], "provenance": {}}

    monkeypatch.setattr(lane, "run_counted_attribution_bundle", fake_bundle)
    out = tmp_path / "routes_out.json"
    assert (
        vp.main(["--attribution-routes", "--width", "4", "--output", str(out)]) == 1
    )
    assert not out.exists()


@pytest.mark.parametrize("width", [6, 8])
def test_attribution_routes_pipeline_runs_the_requested_width(tmp_path, monkeypatch, width):
    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.interop_profile import validation_pipeline as vp

    seen = []

    def fake_bundle(qbit_num=4):
        seen.append(qbit_num)
        return {"clean_start": False, "rows": [], "provenance": {}}

    monkeypatch.setattr(lane, "run_counted_attribution_bundle", fake_bundle)
    out = tmp_path / f"interop_profile_bundle_routes_w{width}.json"
    assert vp.main(["--attribution-routes", "--width", str(width), "--output", str(out)]) == 1
    assert seen == [width]
    assert not out.exists()


@pytest.mark.parametrize("width,partitions", [(4, 5), (6, 7), (8, 9)])
def test_attribution_anchor_partition_members_total_operation_count(width, partitions):
    from benchmarks.density_matrix.interop_profile.attribution_route_lane import (
        build_attribution_anchor,
    )

    _vqe, descriptor_set, bridge = build_attribution_anchor(width)
    assert len(descriptor_set.partitions) == partitions
    assert sum(len(partition.members) for partition in descriptor_set.partitions) == int(
        bridge["operation_count"]
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("operation_count", 19, "operation_count mismatch"),
        ("parameter_count", 31, "parameter_count mismatch"),
        ("gate_count", 16, "gate_count mismatch"),
        ("noise_count", 4, "noise_count mismatch"),
        ("partition_count", 8, "partition_count mismatch"),
        ("workload_label", "phase2_xxz_hea_q6_other", "unexpected workload_id"),
    ],
)
def test_attribution_anchor_hands_back_when_counts_disagree(monkeypatch, field, value, match):
    import dataclasses

    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane

    real_profile = lane.attribution_width_profile(6)
    monkeypatch.setattr(
        lane,
        "attribution_width_profile",
        lambda _width: dataclasses.replace(real_profile, **{field: value}),
    )
    with pytest.raises(ValueError, match=match):
        lane.build_attribution_anchor(6)


def test_attribution_anchor_hands_back_when_members_miss_an_operation(monkeypatch):
    import types

    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane

    real_builder = lane.build_phase3_continuity_partition_descriptor_set

    def short_builder(vqe, **kwargs):
        real = real_builder(vqe, **kwargs)
        partitions = list(real.partitions)
        partitions[-1] = types.SimpleNamespace(members=tuple(partitions[-1].members)[:-1])
        return types.SimpleNamespace(workload_id=real.workload_id, partitions=partitions)

    monkeypatch.setattr(lane, "build_phase3_continuity_partition_descriptor_set", short_builder)
    with pytest.raises(ValueError, match="partition members do not total"):
        lane.build_attribution_anchor(6)


@pytest.mark.parametrize("route_id", ["R-base", "R-fused", "R-hybrid"])
def test_timed_route_hands_back_when_a_partition_is_not_executed(monkeypatch, route_id):
    import types

    from benchmarks.density_matrix.interop_profile import attribution_route_lane as lane
    from benchmarks.density_matrix.partitioned_runtime.common import build_initial_parameters

    vqe, descriptor_set, _bridge = lane.build_attribution_anchor(6)
    parameters = build_initial_parameters(vqe.get_Parameter_Num())
    real_executor = lane.ROUTE_TABLE[route_id]["executor"]

    def drop_last_partition(descriptors, params):
        result = real_executor(descriptors, params)
        return types.SimpleNamespace(partitions=result.partitions[:-1])

    monkeypatch.setitem(lane.ROUTE_TABLE[route_id], "executor", drop_last_partition)
    with pytest.raises(ValueError, match="not every partition was executed"):
        lane.build_route_row(
            route_id, descriptor_set, parameters, sample_count=2, throughput_divisor=73728
        )
