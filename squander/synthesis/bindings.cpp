/*
Copyright 2025 SQUANDER Contributors

pybind11 bindings for the SABRE routing engine.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <utility>

#include "sabre_router.hpp"

namespace py = pybind11;
using namespace squander::routing;

// ---------------------------------------------------------------------------
// Helper: extract fields from a Python PartitionCandidate object into CandidateData
// ---------------------------------------------------------------------------

static CandidateData extract_candidate(py::handle pc) {
    CandidateData cd;
    cd.partition_idx = pc.attr("partition_idx").cast<int>();
    cd.topology_idx = pc.attr("topology_idx").cast<int>();
    cd.permutation_idx = pc.attr("permutation_idx").cast<int>();
    cd.cnot_count = pc.attr("cnot_count").cast<int>();

    // P_i, P_o: tuples of ints
    cd.P_i = pc.attr("P_i").cast<std::vector<int>>();
    cd.P_o = pc.attr("P_o").cast<std::vector<int>>();

    // node_mapping: dict {Q* -> Q} -> flatten to dense array
    py::dict nm = pc.attr("node_mapping");
    int max_qstar = -1;
    for (auto [key, val] : nm) {
        int qs = key.cast<int>();
        if (qs > max_qstar) max_qstar = qs;
    }
    cd.node_mapping_flat.resize(max_qstar + 1, -1);
    for (auto [key, val] : nm) {
        cd.node_mapping_flat[key.cast<int>()] = val.cast<int>();
    }

    // qbit_map: dict {q -> q*}
    py::dict qm = pc.attr("qbit_map");
    cd.qbit_map_keys.reserve(py::len(qm));
    cd.qbit_map_vals.reserve(py::len(qm));
    for (auto [key, val] : qm) {
        cd.qbit_map_keys.push_back(key.cast<int>());
        cd.qbit_map_vals.push_back(val.cast<int>());
    }

    // involved_qbits: tuple of ints
    cd.involved_qbits = pc.attr("involved_qbits").cast<std::vector<int>>();

    return cd;
}

// ---------------------------------------------------------------------------
// Helper: extract canonical_data dict -> unordered_map
// ---------------------------------------------------------------------------

static std::vector<int> extract_int_array(py::handle obj) {
    std::vector<int> result;
    auto arr = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(obj);
    if (!arr) {
        return result;
    }
    auto acc = arr.unchecked<1>();
    result.resize(acc.shape(0));
    for (ssize_t i = 0; i < acc.shape(0); i++) {
        result[i] = acc(i);
    }
    return result;
}

static std::unordered_map<int, CanonicalEntry> extract_canonical_data(py::dict cd) {
    std::unordered_map<int, CanonicalEntry> result;
    for (auto [key, val] : cd) {
        int pidx = key.cast<int>();
        CanonicalEntry entry;
        py::dict d = py::reinterpret_borrow<py::dict>(val);
        if (d.contains("edges_u") && !d["edges_u"].is_none()) {
            entry.edges_u = extract_int_array(d["edges_u"]);
        }
        if (d.contains("edges_v") && !d["edges_v"].is_none()) {
            entry.edges_v = extract_int_array(d["edges_v"]);
        }
        entry.cnot = d["cnot"].cast<int>();
        result[pidx] = std::move(entry);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Helper: extract layout_partitions list -> vector<LayoutPartInfo>
// ---------------------------------------------------------------------------

static std::vector<LayoutPartInfo> extract_layout_partitions(py::list lp) {
    std::vector<LayoutPartInfo> result;
    result.reserve(py::len(lp));
    for (auto item : lp) {
        py::dict d = py::reinterpret_borrow<py::dict>(item);
        LayoutPartInfo info;
        info.is_single = d["is_single"].cast<bool>();
        info.involved_qbits = d["involved_qbits"].cast<std::vector<int>>();
        result.push_back(std::move(info));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_sabre_router, m) {
    m.doc() = "SQUANDER SABRE Routing Engine - C++ Backend";

    // Bind SabreConfig
    py::class_<SabreConfig>(m, "SabreConfig")
        .def(py::init<>())
        .def_readwrite("prefilter_top_k", &SabreConfig::prefilter_top_k)
        .def_readwrite("max_E_size", &SabreConfig::max_E_size)
        .def_readwrite("max_lookahead", &SabreConfig::max_lookahead)
        .def_readwrite("E_weight", &SabreConfig::E_weight)
        .def_readwrite("E_alpha", &SabreConfig::E_alpha)
        .def_readwrite("cnot_cost", &SabreConfig::cnot_cost)
        .def_readwrite("sabre_iterations", &SabreConfig::sabre_iterations)
        .def_readwrite("n_layout_trials", &SabreConfig::n_layout_trials)
        .def_readwrite("random_seed", &SabreConfig::random_seed)
        .def_readwrite("decay_delta", &SabreConfig::decay_delta)
        .def_readwrite("swap_burst_budget", &SabreConfig::swap_burst_budget)
        .def_readwrite("path_tiebreak_weight", &SabreConfig::path_tiebreak_weight)
        .def_readwrite("three_qubit_exit_weight", &SabreConfig::three_qubit_exit_weight);

    // Bind SabreRouter with data-converting constructor
    py::class_<SabreRouter>(m, "SabreRouter")
        .def(py::init(
            [](const SabreConfig& config,
               py::array_t<double, py::array::c_style> D_arr,
               std::vector<std::vector<int>> adj,
               std::vector<std::vector<int>> DAG,
               std::vector<std::vector<int>> IDAG,
               py::list candidate_cache_py,
               py::list layout_partitions_py,
               py::dict canonical_data_fwd_py,
               py::dict canonical_data_rev_py
            ) {
                // Extract D matrix
                auto buf = D_arr.request();
                if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
                    throw std::invalid_argument("D must be a square 2D array");
                }
                int N = static_cast<int>(buf.shape[0]);
                std::vector<double> D_flat(N * N);
                auto* ptr = static_cast<const double*>(buf.ptr);
                std::copy(ptr, ptr + N * N, D_flat.begin());

                // Convert candidate_cache: list of lists of PartitionCandidate
                std::vector<std::vector<CandidateData>> cc;
                cc.reserve(py::len(candidate_cache_py));
                for (auto part_cands : candidate_cache_py) {
                    std::vector<CandidateData> cands;
                    py::list cl = py::reinterpret_borrow<py::list>(part_cands);
                    cands.reserve(py::len(cl));
                    for (auto c : cl) {
                        auto cd = extract_candidate(c);
                        cd.candidate_idx = static_cast<int>(cands.size());
                        cands.push_back(std::move(cd));
                    }
                    cc.push_back(std::move(cands));
                }

                auto lp = extract_layout_partitions(layout_partitions_py);
                auto cd_fwd = extract_canonical_data(canonical_data_fwd_py);
                auto cd_rev = extract_canonical_data(canonical_data_rev_py);

                return new SabreRouter(
                    config, N, std::move(D_flat), std::move(adj), std::move(DAG), std::move(IDAG),
                    std::move(cc), std::move(lp), std::move(cd_fwd), std::move(cd_rev)
                );
            }),
            py::arg("config"),
            py::arg("D"),
            py::arg("adj"),
            py::arg("DAG"),
            py::arg("IDAG"),
            py::arg("candidate_cache"),
            py::arg("layout_partitions"),
            py::arg("canonical_data_fwd"),
            py::arg("canonical_data_rev")
        )
        .def("route_forward",
            [](const SabreRouter& self,
               const std::vector<int>& pi
            ) -> py::tuple {
                py::gil_scoped_release release;
                auto result = self.route_forward(pi);
                py::gil_scoped_acquire acquire;
                py::list steps;
                for (const auto& step : result.steps) {
                    if (step.type == 0) {
                        steps.append(py::make_tuple("swap", step.swaps));
                    } else if (step.type == 1) {
                        steps.append(py::make_tuple("partition", step.partition_idx, step.candidate_idx));
                    } else {
                        steps.append(py::make_tuple("single", step.partition_idx, step.physical_qubit));
                    }
                }
                return py::make_tuple(result.cnot_count, result.pi, result.pi_initial, steps);
            },
            py::arg("pi"),
            "Run actual forward routing and return CNOT count, final pi, initial pi, and route steps"
        )
        .def("run_trial",
            [](const SabreRouter& self,
               int trial_idx,
               const std::vector<int>& seeded_pi,
               int n_iterations,
               int n_trials
            ) -> py::tuple {
                py::gil_scoped_release release;
                auto result = self.run_trial(trial_idx, seeded_pi, n_iterations, n_trials);
                py::gil_scoped_acquire acquire;
                return py::make_tuple(result.total_cost, result.pi);
            },
            py::arg("trial_idx"),
            py::arg("seeded_pi"),
            py::arg("n_iterations"),
            py::arg("n_trials"),
            "Run a single layout trial (GIL-free, thread-safe)"
        );
}
