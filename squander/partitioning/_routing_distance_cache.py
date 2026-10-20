"""Bounded reuse of exact path-layout distances without retaining a catalog."""
from functools import lru_cache, wraps


def bounded_path_mapping_distance_cache(function):
    """Snapshot plain sequence arguments; preserve every other call unchanged.

    Distance results are immutable integers. The bounded cache holds no
    circuits, alternatives or synthesis data, and never changes their scores.
    """
    @lru_cache(maxsize=4096)
    def remember(arguments):
        return function(*arguments)

    @wraps(function)
    def distance(*args, **kwargs):
        if kwargs or not all(type(value) in (tuple, list) for value in args):
            return function(*args, **kwargs)
        frozen = tuple(tuple(value) for value in args)
        try:
            hash(frozen)
        except TypeError:
            return function(*args, **kwargs)
        return remember(frozen)

    distance.cache_info = remember.cache_info
    distance.cache_clear = remember.cache_clear
    return distance

