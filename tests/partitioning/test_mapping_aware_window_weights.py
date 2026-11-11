"""The fast path-window score equals explicit complete-mapping Kendall detours."""
import itertools
import random

from squander.partitioning import routing


def test_window_score_matches_explicit_mapping_completions():
    rng = random.Random(20261111)
    checked = 0
    for n in range(2, 10):
        for width in range(1, min(n, 3) + 1):
            path = list(range(n))
            before = list(range(n))
            after = list(range(n))
            rng.shuffle(path)
            rng.shuffle(before)
            rng.shuffle(after)
            logicals = tuple(rng.sample(range(n), width))
            permutations = tuple(itertools.permutations(range(width)))
            alternatives = []
            for start in range(n - width + 1):
                for pin, pout in itertools.product(permutations, repeat=2):
                    alternatives.append(routing.RoutingAlternative(
                        0, logicals, tuple(path[start + q] for q in pin),
                        tuple(path[start + q] for q in pout), rng.randrange(12),
                    ))
            if width == 2 and n >= 4:
                # Existing callers may also supply a non-window realization.
                alternatives.append(routing.RoutingAlternative(
                    0, logicals, (path[0], path[3]), (path[3], path[0]), 2,
                ))
            detours = []
            for alt in alternatives:
                values = []
                for outside in (before, after):
                    incoming = routing._complete_partial_path_mapping(
                        outside, logicals, alt.input_physical, path,
                    )
                    outgoing = list(incoming)
                    for logical, physical in zip(logicals, alt.output_physical):
                        outgoing[logical] = physical
                    values.append(
                        routing._path_mapping_inversion_distance(before, incoming, path)
                        + routing._path_mapping_inversion_distance(outgoing, after, path)
                    )
                detours.append(min(values))
            for weight in (1.5, 3.0, 12.0):
                expected = min(alt.cnot_count + weight * cost
                               for alt, cost in zip(alternatives, detours))
                result = routing.mapping_aware_partition_weights(
                    partitions=(frozenset((0, 1)),), alternatives={0: alternatives},
                    gate_before_mapping={0: before, 1: before},
                    gate_after_mapping={0: after, 1: after}, path=path,
                    swap_cnot_cost=weight,
                )
                assert result == [expected]
                checked += len(alternatives)
    assert checked > 1000
