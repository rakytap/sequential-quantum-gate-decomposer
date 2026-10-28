/*
 * Copyright 2026 Budapest Quantum Computing Group
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef OSR_MOTIF_LIBRARY_H
#define OSR_MOTIF_LIBRARY_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace osr_motifs {

struct EdgeBlock {
    int edge;
    unsigned support;
    int cnots;
};

// Enumerate block orders modulo exact commutations on disjoint supports.
// A newly appended block cannot pass a smaller commuting block to its left:
// this is the lexicographically least representative of each trace class.
inline std::vector<std::vector<int>> block_orders(
    std::vector<EdgeBlock> blocks)
{
    blocks.erase(std::remove_if(blocks.begin(), blocks.end(),
        [](const EdgeBlock& block) { return block.cnots <= 0; }), blocks.end());
    std::sort(blocks.begin(), blocks.end(),
        [](const EdgeBlock& left, const EdgeBlock& right) {
            return left.edge < right.edge;
        });
    for (size_t index = 1; index < blocks.size(); ++index) {
        if (blocks[index - 1].edge == blocks[index].edge) {
            throw std::invalid_argument("Motif blocks must use distinct edges");
        }
    }
    std::vector<std::vector<int>> words;
    std::vector<int> prefix;
    std::vector<bool> used(blocks.size(), false);
    std::function<void()> enumerate = [&]() {
        if (prefix.size() == blocks.size()) {
            std::vector<int> word;
            for (int block : prefix) {
                word.insert(word.end(), blocks[block].cnots, blocks[block].edge);
            }
            words.push_back(std::move(word));
            return;
        }
        for (size_t block = 0; block < blocks.size(); ++block) {
            if (used[block]) continue;
            bool canonical = true;
            for (auto previous = prefix.rbegin(); previous != prefix.rend();
                 ++previous) {
                if (blocks[*previous].support & blocks[block].support) break;
                if (*previous > static_cast<int>(block)) {
                    canonical = false;
                    break;
                }
            }
            if (!canonical) continue;
            used[block] = true;
            prefix.push_back(static_cast<int>(block));
            enumerate();
            prefix.pop_back();
            used[block] = false;
        }
    };
    enumerate();
    return words;
}

// Necessary full-unitary constraint, independent of fitted gate angles.
// If A=U^dag P U has squared Frobenius weight e outside the word's backward
// light cone, then e <= 4 ||U-exp(i phi)V||_F^2 for every realizable V.
// Consequently eta=e/(8N) gives process infidelity >= eta*(2-eta).
// This does NOT reject a partial OSR step, which need not reproduce U yet.
class CausalWitness {
    std::vector<std::vector<double>> losses_;

public:
    template<class Element>
    CausalWitness(int qubits, Element element)
    {
        if (qubits < 1 || qubits > 6) {
            throw std::invalid_argument("Causal motif witnesses require 1-6 qubits");
        }
        using Complex = std::complex<double>;
        const size_t dimension = size_t(1) << qubits;
        losses_.assign(qubits, std::vector<double>(dimension, 0.0));
        std::vector<Complex> image(dimension * dimension);
        const double roundoff = 256.0 * std::numeric_limits<double>::epsilon() *
                                dimension * dimension;
        for (int output = 0; output < qubits; ++output) {
            const size_t bit = size_t(1) << output;
            for (int pauli = 0; pauli < 3; ++pauli) {
                double image_norm = 0.0;
                for (size_t row = 0; row < dimension; ++row) {
                    for (size_t column = 0; column < dimension; ++column) {
                        Complex value(0.0, 0.0);
                        for (size_t inner = 0; inner < dimension; ++inner) {
                            Complex phase(1.0, 0.0);
                            if (pauli == 1) {
                                phase = (inner & bit) ? Complex(0.0, 1.0)
                                                     : Complex(0.0, -1.0);
                            } else if (pauli == 2 && (inner & bit)) {
                                phase = Complex(-1.0, 0.0);
                            }
                            const size_t other = pauli == 2 ? inner : inner ^ bit;
                            value += std::conj(element(inner, row)) * phase *
                                     element(other, column);
                        }
                        image[row * dimension + column] = value;
                        image_norm += std::norm(value);
                    }
                }
                for (size_t cone = 0; cone < dimension; ++cone) {
                    std::vector<size_t> kept;
                    std::vector<size_t> outside;
                    for (size_t basis = 0; basis < dimension; ++basis) {
                        if ((basis & ~cone) == 0) kept.push_back(basis);
                        if ((basis & cone) == 0) outside.push_back(basis);
                    }
                    double projected_norm = 0.0;
                    for (size_t row : kept) {
                        for (size_t column : kept) {
                            Complex trace(0.0, 0.0);
                            for (size_t environment : outside) {
                                trace += image[(row | environment) * dimension +
                                               (column | environment)];
                            }
                            projected_norm += std::norm(trace);
                        }
                    }
                    projected_norm /= outside.size();
                    const double forbidden = std::max(
                        image_norm - projected_norm - roundoff, 0.0);
                    const double eta = forbidden / (8.0 * dimension);
                    losses_[output][cone] = std::max(
                        losses_[output][cone], eta * (2.0 - eta));
                }
            }
        }
    }

    double lower_bound(const std::vector<int>& chronological_word,
                       const std::vector<unsigned>& edge_supports) const
    {
        double loss = 0.0;
        for (size_t output = 0; output < losses_.size(); ++output) {
            unsigned cone = unsigned(1) << output;
            for (auto edge = chronological_word.rbegin();
                 edge != chronological_word.rend(); ++edge) {
                if (*edge < 0 || static_cast<size_t>(*edge) >= edge_supports.size()) {
                    throw std::invalid_argument("Motif edge index is outside topology");
                }
                if (cone & edge_supports[*edge]) cone |= edge_supports[*edge];
            }
            loss = std::max(loss, losses_[output][cone]);
        }
        return loss;
    }
};

} // namespace osr_motifs
#endif
