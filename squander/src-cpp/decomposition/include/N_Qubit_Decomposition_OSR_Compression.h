/*
Created on Sat May 02 2026
Copyright 2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file N_Qubit_Decomposition_OSR_Compression.h
    \brief OSR-guided top-down compression for an existing gate structure.
*/

#ifndef N_Qubit_Decomposition_OSR_Compression_H
#define N_Qubit_Decomposition_OSR_Compression_H

#include "Gates_block.h"
#include "N_Qubit_Decomposition_custom.h"
#include "N_Qubit_Decomposition_Tree_Search.h"
#include "config_element.h"
#include "matrix.h"
#include "matrix_real.h"

#include <map>
#include <memory>
#include <vector>

/**
@brief Path to a gate inside a possibly nested Gates_block.

Each entry is an index inside the corresponding block. For example, path
{3, 2} means the third top-level gate is a block and its second child gate is
the selected gate.
*/
struct OSRGatePath {
    std::vector<int> indices;

    bool operator<(const OSRGatePath& other) const {
        return indices < other.indices;
    }
};

/**
@brief Tunable controls for OSR-guided compression.
*/
struct N_Qubit_Decomposition_OSR_Compression_Options {
    /// Number of candidates kept after each deletion depth.
    int beam_width = 8;
    /// Maximal number of entangling gates to remove. Negative means no limit.
    int max_removed_gates = -1;
    /// Keep OSR candidates whose estimated remaining CNOT count is at most this value.
    int osr_bound_limit = 0;
    /// Number of full Hilbert-Schmidt validation trials for final candidates.
    int validation_trials = 3;
    /// If true, run full optimization on final candidates before returning.
    bool validate_final = true;
    /// OSR numerical rank tolerance.
    double osr_tolerance = 1e-3;
    /// If true, augment deletion candidates with local circuit mutations before validation.
    bool enable_mutations = true;
    /// Number of local mutation rounds applied after the deletion beam.
    int mutation_rounds = 1;
    /// Maximal number of mutation candidates generated from each round.
    int mutation_candidates = 32;
    /// If true, rewiring mutations may use all qubit pairs instead of only observed edges.
    bool mutate_full_topology = false;
    /// If true, validate freshly synthesized U3+CNOT skeletons at compressed depths.
    bool enable_skeleton_search = true;
    /// Exact CNOT skeleton depth to test. Negative derives the target from max_removed_gates.
    int skeleton_target_cnots = -1;
    /// Maximal number of synthesized skeletons admitted to final validation.
    int skeleton_max_candidates = 4096;
};

/**
@brief OSR score of one compressed candidate.
*/
struct N_Qubit_Decomposition_OSR_Compression_Score {
    /// Lower-bound estimate of remaining CNOTs required by the residual OSR.
    int min_remaining_cnots = 0;
    /// Secondary bound-solver objective used as tie-breaker.
    double kappa = 0.0;
    /// Aggregate OSR residual tie-breaker.
    double residual = 0.0;
    /// Best per-topology-edge CNOT-count composition found by the bound solver.
    std::vector<int> edge_counts;
    /// Per-cut OSR rank/loss pairs used to derive the bound.
    std::vector<std::pair<int, double>> cut_bounds;
};

/**
@brief Result of OSR-guided compression.
*/
struct N_Qubit_Decomposition_OSR_Compression_Result {
    /// Newly allocated compressed gate structure. The input gate structure is not modified.
    std::unique_ptr<Gates_block> gate_structure;
    /// Parameters from the best validation run, if validation was enabled and run.
    Matrix_real optimized_parameters;
    /// Best final cost from validation, or infinity when validation was not run.
    double current_minimum;
    /// OSR score of the returned structure.
    N_Qubit_Decomposition_OSR_Compression_Score osr_score;
    /// Removed entangling-gate paths from the original gate structure.
    std::vector<OSRGatePath> removed_gate_paths;
    /// Number of entangling gates found in the input structure.
    int original_entangling_gate_num;
    /// Number of entangling gates left in the returned structure.
    int compressed_entangling_gate_num;
    /// Whether final Hilbert-Schmidt validation was run.
    bool validated;
    /// Whether the final validation reached optimization_tolerance.
    bool reached_tolerance;

    double decomposition_error;

    N_Qubit_Decomposition_OSR_Compression_Result();
    N_Qubit_Decomposition_OSR_Compression_Result(const N_Qubit_Decomposition_OSR_Compression_Result&) = delete;
    N_Qubit_Decomposition_OSR_Compression_Result& operator=(const N_Qubit_Decomposition_OSR_Compression_Result&) = delete;
    N_Qubit_Decomposition_OSR_Compression_Result(N_Qubit_Decomposition_OSR_Compression_Result&&) = default;
    N_Qubit_Decomposition_OSR_Compression_Result& operator=(N_Qubit_Decomposition_OSR_Compression_Result&&) = default;
};

/**
@brief Decomposition class that compresses an already supplied gate structure with OSR guidance.

The class assumes the starting circuit is already present in the inherited
Gates_block, typically through set_custom_gate_structure(...). It then searches
top-down by deleting entangling gates, uses OSR to keep promising compressed
candidates, and finally validates the chosen structure with the standard
optimization cost.
*/
class N_Qubit_Decomposition_OSR_Compression : public N_Qubit_Decomposition_custom {

public:
    N_Qubit_Decomposition_OSR_Compression();
    N_Qubit_Decomposition_OSR_Compression(Matrix Umtx_in, int qbit_num_in,
                                          std::map<std::string, Config_Element>& config,
                                          int accelerator_num = 0);
    N_Qubit_Decomposition_OSR_Compression(Matrix Umtx_in, int qbit_num_in,
                                          std::vector<matrix_base<int>> topology_in,
                                          std::map<std::string, Config_Element>& config,
                                          int accelerator_num = 0);
    virtual ~N_Qubit_Decomposition_OSR_Compression();

    /// Externally supplied hardware topology. Empty means infer from the gate structure.
    std::vector<matrix_base<int>> topology;

    virtual void start_decomposition();

    /**
    @brief Compress the supplied gate structure without modifying it.
    */
    N_Qubit_Decomposition_OSR_Compression_Result compress_gate_structure(
        Gates_block* gate_structure_in);

    N_Qubit_Decomposition_OSR_Compression_Options get_osr_compression_options();

protected:
    N_Qubit_Decomposition_OSR_Compression_Score evaluate_gate_structure_osr(
        Gates_block* gate_structure_in,
        const Matrix_real& initial_parameters,
        MinCnotBoundSolver& osr_bound_solver,
        std::vector<std::vector<int>>& all_cuts);

    N_Qubit_Decomposition_custom prepare_custom_optimizer(
        Gates_block* gate_structure_in,
        cost_function_type cost_function_variant);

    void validate_compressed_gate_structure(
        Gates_block* gate_structure_in,
        const Matrix_real& initial_parameters,
        N_Qubit_Decomposition_OSR_Compression_Result& result);
};

#endif
