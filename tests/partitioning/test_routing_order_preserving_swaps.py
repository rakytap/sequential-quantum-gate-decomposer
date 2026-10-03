import itertools

from squander.partitioning.routing import _order_preserving_block_swap_count


def test_order_preserving_block_swap_count_matches_full_inversion_count():
    # Exhaust all source layouts, moved subsets, and block placements on four
    # vertices. The completion preserves the physical order of other tokens.
    qubit_count = 4
    logicals = range(qubit_count)
    for source in itertools.permutations(logicals):
        source_order = sorted(logicals, key=source.__getitem__)
        for width in range(1, qubit_count + 1):
            for moved in itertools.combinations(logicals, width):
                for desired in itertools.permutations(logicals, width):
                    target = [-1] * qubit_count
                    for logical, physical in zip(moved, desired):
                        target[logical] = physical
                    unmoved = [
                        logical for logical in source_order
                        if logical not in moved
                    ]
                    remaining = [
                        physical for physical in logicals
                        if physical not in desired
                    ]
                    for logical, physical in zip(unmoved, remaining):
                        target[logical] = physical
                    target_order = [target[logical] for logical in source_order]
                    expected = sum(
                        target_order[left] > target_order[right]
                        for left in logicals
                        for right in range(left + 1, qubit_count)
                    )
                    actual = _order_preserving_block_swap_count(
                        source, target, moved
                    )
                    assert actual == expected
