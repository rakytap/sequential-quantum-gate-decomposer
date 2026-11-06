#include "include/N_Qubit_Decomposition_Tree_Search.h"
#include "include/OSR_Logical_Factor.h"
#include "include/OSR_Logical_Factor_Inverse.h"
#include "include/OSR_Logical_Phase_Clifford.h"
#include "include/OSR_Phase_Polynomial.h"
#include "include/OSR_Cut_Capacity.h"
#include "include/N_Qubit_Decomposition_Cost_Function.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>

namespace {
using Complex = std::complex<double>;
using Local = std::array<Complex, 4>;

Local identity_local() { return {{1.0, 0.0, 0.0, 1.0}}; }

Local multiply_local(const Local& a, const Local& b) {
    Local result{};
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 2; ++col)
            for (int k = 0; k < 2; ++k) result[2 * row + col] += a[2 * row + k] * b[2 * k + col];
    return result;
}

Local native_u3(const double* angles) {
    const double c = std::cos(angles[0]), s = std::sin(angles[0]);
    const Complex imaginary(0.0, 1.0);
    return {{c, -std::exp(imaginary * angles[2]) * s,
             std::exp(imaginary * angles[1]) * s,
             std::exp(imaginary * (angles[1] + angles[2])) * c}};
}

void append_angles(const Local& matrix, std::vector<double>& angles) {
    const double theta = std::atan2(std::abs(matrix[2]), std::abs(matrix[0]));
    double phi, lambda;
    if (std::abs(matrix[2]) <= 32.0 * std::numeric_limits<double>::epsilon()) {
        phi = 0.0;
        lambda = std::arg(matrix[3]) - std::arg(matrix[0]);
    } else if (std::abs(matrix[0]) <= 32.0 * std::numeric_limits<double>::epsilon()) {
        phi = 0.0;
        lambda = std::arg(-matrix[1]) - std::arg(matrix[2]);
    } else {
        phi = std::arg(matrix[2]) - std::arg(matrix[0]);
        lambda = std::arg(-matrix[1]) - std::arg(matrix[0]);
    }
    angles.insert(angles.end(), {theta, phi, lambda});
}

bool flag(std::map<std::string, Config_Element>& config, const char* name, bool fallback) {
    auto found = config.find(name);
    if (found != config.end()) found->second.get_property(fallback);
    return fallback;
}

double number(std::map<std::string, Config_Element>& config, const char* name, double fallback) {
    auto found = config.find(name);
    if (found != config.end()) found->second.get_property(fallback);
    return fallback;
}

std::vector<std::vector<int>> cuts(int width) {
    std::vector<std::vector<int>> result;
    for (int mask = 1; mask < (1 << (width - 1)); ++mask) {
        std::vector<int> cut, other;
        for (int wire = 0; wire < width; ++wire)
            ((mask & (1 << wire)) ? cut : other).push_back(wire);
        result.push_back(cut.size() <= other.size() ? cut : other);
    }
    return result;
}

int strengthen_cut_bound(const Matrix& target, int width,
                         const std::vector<std::vector<int>>& cut_list,
                         std::vector<std::pair<int, double>> bounds,
                         const MinCnotBoundSolver& solver, double tolerance, int budget) {
    for (std::size_t index = 0; index < cut_list.size(); ++index) {
        int root = -1;
        if (cut_list[index].size() == 1) root = cut_list[index][0];
        else if (cut_list[index].size() == std::size_t(width - 1)) {
            for (int wire = 0; wire < width; ++wire)
                if (std::find(cut_list[index].begin(), cut_list[index].end(), wire) == cut_list[index].end()) root = wire;
        }
        if (root < 0) continue;
        for (bool adjoint : {false, true}) {
            const auto capacity = squander_osr::cut_capacity(
                target.get_data(), width, target.stride, root, tolerance, adjoint);
            if (capacity.valid && capacity.minimum_crossing_cnots > bounds[index].first)
                bounds[index] = {capacity.minimum_crossing_cnots, 0.0};
        }
    }
    return solver.solve_min_cnots(bounds, budget);
}

std::vector<std::pair<int, double>> spectral_bounds(
        const Matrix& target, int width, const std::vector<std::vector<int>>& cut_list,
        double tolerance) {
    std::vector<std::pair<int, double>> result;
    for (const auto& cut : cut_list)
        result.push_back(operator_schmidt_cnot_bound(
            target, width, cut, std::sqrt(double(1 << width)), tolerance));
    return result;
}

struct Candidate {
    squander_osr::LogicalFactor factor;
    Matrix child_target;
    std::vector<matrix_base<int>> child_topology;
    std::vector<int> active;
    int total_lower_bound;
    double surplus;
    bool adjoint_frame = false;
    bool complete_certificate = false;
};

bool same_factor(const squander_osr::LogicalFactor& a,
                 const squander_osr::LogicalFactor& b) {
    if (a.root != b.root || a.prefix_cnots != b.prefix_cnots ||
        a.process_infidelity != b.process_infidelity ||
        a.local_adjoint != b.local_adjoint || a.child_adjoint != b.child_adjoint ||
        a.prefix.size() != b.prefix.size()) return false;
    for (std::size_t index = 0; index < a.prefix.size(); ++index) {
        const auto& first = a.prefix[index];
        const auto& second = b.prefix[index];
        if (first.target != second.target || first.control != second.control ||
            first.matrix != second.matrix) return false;
    }
    return true;
}
} // namespace

bool N_Qubit_Decomposition_Tree_Search::complete_logical_factor(int cnot_budget, bool floor_only) {
    // The existing two-qubit route is already an exact local factor solver.
    // Recursion terminates there, rather than sending one qubit to graph search.
    if (qbit_num < 3 || qbit_num > 4 || cnot_budget < 0 || topology.empty() ||
        !flag(config, "use_osr", true) || !flag(config, "use_graph_search", true) ||
        !flag(config, "osr_certified_terminal_completion", true) ||
        !flag(config, "osr_geometric_completion_motifs", true) ||
        !flag(config, "osr_logical_factor_completion", true)) return false;
    const double tolerance = number(config, "synthesis_acceptance_tolerance", optimization_tolerance);
    if (!std::isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0) return false;
    const int dimension = 1 << qbit_num;
    const double guide_tolerance = std::max(4.0 * tolerance, number(config, "osr_optimization_tolerance", 1e-6));
    long seed = 0;
    auto seed_option = config.find("random_seed");
    if (seed_option != config.end()) seed_option->second.get_property(seed);
    auto exact_config = config;
    exact_config["use_float"].set_property("use_float", false);
    exact_config["float32"].set_property("float32", false);
    exact_config["parallel"].set_property("parallel", long(0));
    auto target_cuts = cuts(qbit_num);
    MinCnotBoundSolver target_solver(qbit_num, target_cuts, topology);
    const int physical_floor = strengthen_cut_bound(Umtx, qbit_num, target_cuts,
        spectral_bounds(Umtx, qbit_num, target_cuts, guide_tolerance), target_solver, tolerance, cnot_budget);
    if (physical_floor < 0) return false;
    const bool incumbent = best_first_osr_solution_found;
    int best_count = incumbent ? int(best_first_solution_path.size()) : cnot_budget + 1;
    if (best_count <= physical_floor) return false;
    int budget = floor_only ? physical_floor : std::min(cnot_budget, best_count - 1);
    unsigned couplers = 0;
    for (const auto& edge : topology) {
        const int a = std::min(edge[0], edge[1]), b = std::max(edge[0], edge[1]);
        couplers |= 1U << (a * qbit_num + b);
    }
    std::vector<Complex> target(dimension * dimension);
    const auto data = Umtx.get_data();
    for (int row = 0; row < dimension; ++row)
        for (int col = 0; col < dimension; ++col) {
            const auto& value = data[col * Umtx.stride + row];
            target[row * dimension + col] = {value.real, -value.imag};
        }
    std::vector<Candidate> candidates;
    std::size_t factored_models = 0, duplicate_factors = 0;
    bool improved = false;
    GrayCodeCNOT best_path;
    std::vector<double> best_parameters;
    for (bool clifford_only : {false, true}) {
        if (best_count <= physical_floor) break;
        if (clifford_only && !flag(config, "osr_logical_phase_clifford", false)) break;
        for (bool adjoint_frame : {false, true}) {
            if (best_count <= physical_floor) break;
            if (adjoint_frame && !flag(config, "osr_logical_factor_adjoint_frames", true)) break;
            for (int row = 0; row < dimension; ++row)
                for (int column = 0; column < dimension; ++column) {
                    const auto& value = adjoint_frame ? data[row * Umtx.stride + column]
                                                     : data[column * Umtx.stride + row];
                    target[row * dimension + column] =
                        Complex(value.real, adjoint_frame ? value.imag : -value.imag);
                }
            candidates.clear();
            for (int root = 0; root < qbit_num; ++root) {
                std::vector<int> active;
                for (int wire = 0; wire < qbit_num; ++wire) if (wire != root) active.push_back(wire);
                std::vector<matrix_base<int>> child_topology;
                for (const auto& edge : topology) {
                    auto a = std::find(active.begin(), active.end(), edge[0]);
                    auto b = std::find(active.begin(), active.end(), edge[1]);
                    if (a == active.end() || b == active.end()) continue;
                    matrix_base<int> pair(1, 2);
                    pair[0] = int(a - active.begin()); pair[1] = int(b - active.begin());
                    child_topology.push_back(pair);
                }
                std::vector<bool> reached(active.size(), false);
                if (!clifford_only && flag(config, "osr_phase_polynomial_completion", true)) {
                    auto factor = squander_osr::phase_polynomial_factor(
                        target.data(), qbit_num, dimension, root, couplers, tolerance, budget);
                    if (factor.valid && factor.prefix_cnots <= budget) {
                        const int half = dimension / 2;
                        Matrix identity_child(half, half);
                        for (int row = 0; row < half; ++row)
                            for (int col = 0; col < half; ++col)
                                identity_child.get_data()[row * identity_child.stride + col] =
                                    {row == col ? 1.0 : 0.0, 0.0};
                        const int count = factor.prefix_cnots;
                        candidates.push_back({std::move(factor), identity_child, child_topology,
                                              active, count, 0.0, adjoint_frame, true});
                        ++factored_models;
                    }
                }
                reached[0] = true;
                for (std::size_t step = 0; step < active.size(); ++step)
                    for (const auto& edge : child_topology)
                        if (reached[edge[0]] || reached[edge[1]]) reached[edge[0]] = reached[edge[1]] = true;
                if (std::find(reached.begin(), reached.end(), false) != reached.end()) continue;
                const int child_width = qbit_num - 1, half = dimension / 2;
                auto child_cuts = cuts(child_width);
                MinCnotBoundSolver child_solver(child_width, child_cuts, child_topology);
                auto register_factor = [&](squander_osr::LogicalFactor&& factor) {
                        if (!factor.valid || factor.prefix_cnots > budget) return;
                        ++factored_models;
                        if (std::any_of(candidates.begin(), candidates.end(), [&](const Candidate& existing) {
                            return existing.adjoint_frame == adjoint_frame &&
                                same_factor(factor, existing.factor);
                        })) {
                            ++duplicate_factors;
                            return;
                        }
                        Matrix child_target(half, half);
                        for (int row = 0; row < half; ++row)
                            for (int col = 0; col < half; ++col) {
                                const Complex value = std::conj(factor.child_adjoint[col * half + row]);
                                child_target.get_data()[row * child_target.stride + col] = {value.real(), value.imag()};
                            }
                        const double necessary = std::pow(std::sin(std::asin(std::sqrt(tolerance)) +
                            std::asin(std::sqrt(std::max(0.0, factor.process_infidelity)))), 2) *
                            (1.0 + 64.0 * std::numeric_limits<double>::epsilon() * dimension);
                        const auto bounds = spectral_bounds(child_target, child_width, child_cuts, necessary);
                        const int minimum = strengthen_cut_bound(child_target, child_width, child_cuts,
                            bounds, child_solver, necessary,
                            budget - factor.prefix_cnots);
                        if (minimum < 0) return;
                        double surplus;
                        std::vector<int> edge_counts;
                        child_solver.solve_min_cnots(bounds, surplus, edge_counts, budget - factor.prefix_cnots);
                        const int total_lower_bound = minimum + factor.prefix_cnots;
                        candidates.push_back({std::move(factor), child_target, child_topology, active,
                                 total_lower_bound, surplus, adjoint_frame});
                };
                if (!clifford_only) {
                    std::vector<int> order = active, neighbors(active.size());
                    do {
                        std::function<void(std::size_t)> enumerate = [&](std::size_t index) {
                            if (index != order.size()) {
                                neighbors[index] = root; enumerate(index + 1);
                                for (std::size_t next = index + 1; next < order.size(); ++next) {
                                    neighbors[index] = order[next]; enumerate(index + 1);
                                }
                                return;
                            }
                            auto factor = squander_osr::logical_factor(target.data(), qbit_num, dimension, root,
                                order.data(), neighbors.data(), couplers, tolerance, budget);
                            register_factor(std::move(factor));
                        };
                        enumerate(0);
                    } while (std::next_permutation(order.begin(), order.end()));
                    squander_osr::logical_phase_factors(
                        target.data(), qbit_num, dimension, root, couplers, tolerance,
                        budget, register_factor);
                } else {
                    squander_osr::logical_phase_clifford_factors(
                        target.data(), qbit_num, dimension, root, couplers, tolerance,
                        budget, register_factor);
                }
            }
            if (flag(config, "osr_native_path_trace", false)) {
                std::printf("OSR logical factor models: width=%d, floor=%d, budget=%d, factors=%zu, unique=%zu, duplicate_factors=%zu\n",
                            qbit_num, physical_floor, budget, factored_models, candidates.size(), duplicate_factors);
                std::fflush(stdout);
            }
            std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                if (a.total_lower_bound != b.total_lower_bound) return a.total_lower_bound < b.total_lower_bound;
                return a.surplus < b.surplus;
            });
            for (const auto& candidate : candidates) {
                if (candidate.total_lower_bound > budget || candidate.total_lower_bound >= best_count) continue;
                const auto& factor = candidate.factor;
                const double angle = std::max(0.0, std::asin(std::sqrt(tolerance)) -
                    std::asin(std::sqrt(std::max(0.0, factor.process_infidelity))));
                const double child_tolerance = std::pow(std::sin(angle), 2) *
                    (1.0 - 64.0 * std::numeric_limits<double>::epsilon() * dimension);
                if (!candidate.complete_certificate && child_tolerance <= 0.0) continue;
                std::unique_ptr<N_Qubit_Decomposition_Tree_Search> child;
                int child_cnots = 0;
                Matrix_real child_parameters(1, 3 * (qbit_num - 1));
                std::memset(child_parameters.get_data(), 0, child_parameters.size() * sizeof(double));
                if (!candidate.complete_certificate) {
                    auto child_config = exact_config;
                    child_config["synthesis_acceptance_tolerance"].set_property("synthesis_acceptance_tolerance", child_tolerance);
                    child_config["tree_level_max"].set_property("tree_level_max", long(budget - factor.prefix_cnots));
                    child.reset(new N_Qubit_Decomposition_Tree_Search(candidate.child_target, qbit_num - 1,
                        candidate.child_topology, child_config, 0));
                    child->set_verbose(0);
                    child->set_optimizer(alg);
                    child->set_random_seed(static_cast<int>(seed));
                    child->set_cost_function_variant(cost_fnc);
                    child->set_optimization_tolerance(optimization_tolerance);
                    child->start_decomposition();
                    if (!child->best_first_osr_solution_found) continue;
                    child_cnots = int(child->best_first_solution_path.size());
                    if (factor.prefix_cnots + child_cnots > budget) continue;
                    child_parameters = child->get_optimized_parameters();
                }
                if (child_parameters.size() != 6 * child_cnots + 3 * (qbit_num - 1)) continue;
                std::vector<Local> pending(qbit_num, identity_local());
                std::vector<int> word;
                std::vector<double> parameters;
                auto add_local = [&](int wire, const Local& local) { pending[wire] = multiply_local(local, pending[wire]); };
                auto add_cnot = [&](int control, int target_wire) {
                    int code = -1;
                    bool reversed = false;
                    for (int index = 0; index < possible_target_qbits.size(); ++index)
                        if (possible_target_qbits[index] == target_wire && possible_control_qbits[index] == control) code = int(index);
                    if (code < 0) {
                        for (int index = 0; index < possible_target_qbits.size(); ++index)
                            if (possible_target_qbits[index] == control && possible_control_qbits[index] == target_wire) code = int(index);
                        reversed = true;
                    }
                    if (code < 0) return false;
                    const double h = 1.0 / std::sqrt(2.0);
                    const Local hadamard{{h, h, h, -h}};
                    if (reversed) { add_local(control, hadamard); add_local(target_wire, hadamard); }
                    const int encoded_target = possible_target_qbits[code], encoded_control = possible_control_qbits[code];
                    append_angles(pending[encoded_target], parameters);
                    append_angles(pending[encoded_control], parameters);
                    pending[encoded_target] = pending[encoded_control] = identity_local();
                    word.push_back(code);
                    if (reversed) { add_local(control, hadamard); add_local(target_wire, hadamard); }
                    return true;
                };
                bool valid = true;
                for (const auto& operation : factor.prefix) {
                    if (operation.control < 0) add_local(operation.target, operation.matrix);
                    else if (!add_cnot(operation.control, operation.target)) { valid = false; break; }
                }
                const double* child_angles = child_parameters.get_data();
                for (int index = 0; valid && index < child_cnots; ++index) {
                    const int code = child->best_first_solution_path[index];
                    const int target_wire = candidate.active[child->possible_target_qbits[code]];
                    const int control = candidate.active[child->possible_control_qbits[code]];
                    add_local(target_wire, native_u3(child_angles + 6 * index));
                    add_local(control, native_u3(child_angles + 6 * index + 3));
                    valid = add_cnot(control, target_wire);
                }
                if (!valid) continue;
                for (int wire = 0; wire < qbit_num - 1; ++wire)
                    add_local(candidate.active[wire], native_u3(child_angles + 6 * child_cnots + 3 * wire));
                add_local(factor.root, factor.local_adjoint);
                for (int wire = 0; wire < qbit_num; ++wire) append_angles(pending[wire], parameters);
                if (candidate.adjoint_frame)
                    squander_osr::reverse_native_factor_word(
                        word, parameters, possible_target_qbits, possible_control_qbits,
                        qbit_num, append_angles, native_u3);
                matrix_base<int8_t> limits(1, word.size());
                for (std::size_t index = 0; index < word.size(); ++index) limits[index] = int8_t(topology.size());
                GrayCodeCNOT path(limits);
                for (std::size_t index = 0; index < word.size(); ++index) path[index] = int8_t(word[index]);
                std::unique_ptr<Gates_block> structure(construct_gate_structure_from_Gray_code(path, true));
                Matrix_real actual_parameters(1, parameters.size());
                std::memcpy(actual_parameters.get_data(), parameters.data(), parameters.size() * sizeof(double));
                Matrix verification(dimension, dimension);
                for (int row = 0; row < dimension; ++row)
                    for (int col = 0; col < dimension; ++col)
                        verification.get_data()[row * verification.stride + col] = data[row * Umtx.stride + col];
                structure->apply_to(actual_parameters, verification, 0);
                Complex trace(0.0, 0.0);
                for (int row = 0; row < dimension; ++row) {
                    const auto& value = verification.get_data()[row * verification.stride + row];
                    trace += Complex(value.real, value.imag);
                }
                if (std::abs(trace) == 0.0) continue;
                const Complex phase = trace / std::abs(trace);
                double chord = 0.0;
                for (int row = 0; row < dimension; ++row)
                    for (int col = 0; col < dimension; ++col) {
                        const auto& value = verification.get_data()[row * verification.stride + col];
                        chord += std::norm(Complex(value.real, value.imag) - (row == col ? phase : Complex(0.0, 0.0)));
                    }
                chord /= dimension;
                const double error = chord * (1.0 - 0.25 * chord);
                if (!std::isfinite(error) || error > tolerance || int(word.size()) >= best_count) continue;
                best_count = int(word.size());
                best_path = path;
                best_parameters = std::move(parameters);
                improved = true;
                budget = std::min(budget, best_count - 1);
                if (best_count <= physical_floor) break;
            }
        }
    }
    if (!improved) return false;
    best_first_solution_path = best_path;
    best_first_solution_parameters = std::move(best_parameters);
    best_first_osr_solution_found = true;
    return true;
}

extern "C" int qgd_osr_reverse_logical_factor_word(
        int width, int count, int* word, int coupler_count,
        const int* targets, const int* controls, double* parameters, int parameter_count) {
    if (width < 1 || count < 0 || coupler_count < 0 || !parameters ||
        parameter_count != 6LL * count + 3LL * width ||
        (count && (!word || !targets || !controls))) return -1;
    for (int index = 0; index < count; ++index) {
        const int digit = word[index];
        if (digit < 0 || digit >= coupler_count || targets[digit] < 0 || targets[digit] >= width ||
            controls[digit] < 0 || controls[digit] >= width || targets[digit] == controls[digit]) return -1;
    }
    for (int index = 0; index < parameter_count; ++index)
        if (!std::isfinite(parameters[index])) return -1;
    struct WordView {
        int* data; int count;
        std::size_t size() const { return count; }
        int& operator[](std::size_t index) { return data[index]; }
    } path{word, count};
    struct ParameterView {
        double* data; int count;
        std::size_t size() const { return count; }
        double* get_data() { return data; }
    } angles{parameters, parameter_count};
    struct CouplerView {
        const int* data;
        const int* get_data() const { return data; }
    } target_view{targets}, control_view{controls};
    try {
        squander_osr::reverse_native_factor_word(path, angles, target_view, control_view,
                                                width, append_angles, native_u3);
    } catch (...) {
        return -1;
    }
    return 0;
}
