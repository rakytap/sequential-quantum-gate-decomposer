"""Exact path-distance reuse, input compatibility and bounded retention."""
import itertools
import inspect
from squander.partitioning import routing
from squander.partitioning._routing_distance_cache import (
    bounded_path_mapping_distance_cache,
)
import pytest

original = inspect.unwrap(routing._path_mapping_inversion_distance)
cached = bounded_path_mapping_distance_cache(original)

@pytest.mark.parametrize("width", (2, 3, 4, 5))
def test_all_small_permutation_distances_and_mutable_snapshots(width):
    path = tuple(3 * index + 1 for index in range(width))
    layouts = tuple(itertools.permutations(path))
    for source in layouts:
        mutable = list(source)
        for target in layouts:
            expected = original(source, target, path)
            assert cached(source, target, path) == expected
            assert cached(mutable, target, path) == expected
        mutable.reverse()
        assert cached(mutable, path, path) == original(mutable, path, path)

def test_fixed_cache_bound_and_repeated_distance_hits():
    cached.cache_clear()
    path = tuple(range(8))
    for source in itertools.islice(itertools.permutations(path), 5000):
        assert cached(source, path, path) == original(source, path, path)
    assert cached.cache_info().currsize == 4096
    before = cached.cache_info().hits
    cached(path[::-1], path, path)
    cached(path[::-1], path, path)
    assert cached.cache_info().hits > before

@pytest.mark.parametrize("source,target,path", [
    ((0, 0, 2), (0, 1, 2), (0, 1, 2)),
    ((0, 1, 2), (0, 1), (0, 1, 2)),
    ((0, 1, 9), (0, 1, 2), (0, 1, 2)),
    (([0], 1), (0, 1), (0, 1)),
])
def test_malformed_input_behavior_is_not_changed(source, target, path):
    try:
        expected = original(source, target, path)
    except Exception as error:
        with pytest.raises(type(error)):
            cached(source, target, path)
    else:
        assert cached(source, target, path) == expected

def test_mapping_objects_and_keyword_calls_preserve_original_semantics():
    for values in (({0: 2, 1: 1, 2: 0}, (0, 1, 2), (0, 1, 2)),
                   ((0, 1, 2), (0, 1, 2), (0, 1, 2))):
        try:
            expected = original(*values)
        except Exception as error:
            with pytest.raises(type(error)):
                cached(*values)
        else:
            assert cached(*values) == expected
            keywords = inspect.signature(original).bind(*values).arguments
            assert cached(**keywords) == expected


def test_nested_wrappers_report_and_clear_their_own_cache():
    inner = bounded_path_mapping_distance_cache(original)
    outer = bounded_path_mapping_distance_cache(inner)
    args = ((2, 1, 0), (0, 1, 2), (0, 1, 2))
    outer(*args)
    outer(*args)
    assert outer.cache_info().hits == 1
    assert outer.cache_info().currsize == 1
    outer.cache_clear()
    assert outer.cache_info().currsize == 0
    assert inner.cache_info().currsize == 1

