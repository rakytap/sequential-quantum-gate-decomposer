"""Pytest hooks for the SQUANDER test tree."""

from pathlib import Path

import pytest

# Paths relative to tests/ that belong to the density-matrix project regression lane.
_DENSITY_MATRIX_INCLUDE_PREFIXES = (
    "density_matrix/",
    "partitioning/",
    "VQE/test_VQE.py",
)

# Under tests/; excluded from the density_matrix marker (general upstream tests).
_DENSITY_MATRIX_EXCLUDE_RELATIVE = frozenset(
    {
        "partitioning/test_partition.py",
        "VQE/test_shot_noise_measurement.py",
    }
)


def _relative_test_path(item: pytest.Item) -> str | None:
    """Return path under tests/ using forward slashes, or None if not under tests/."""
    try:
        path = Path(str(item.fspath))
    except AttributeError:
        path = Path(item.path)
    parts = path.parts
    if "tests" not in parts:
        return None
    idx = parts.index("tests")
    return "/".join(parts[idx + 1 :])


def _is_density_matrix_regression_test(relative: str) -> bool:
    if relative in _DENSITY_MATRIX_EXCLUDE_RELATIVE:
        return False
    return any(
        relative == prefix or relative.startswith(prefix)
        for prefix in _DENSITY_MATRIX_INCLUDE_PREFIXES
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply density_matrix marker to tests in the density-matrix project track."""
    marker = pytest.mark.density_matrix
    for item in items:
        relative = _relative_test_path(item)
        if relative is None:
            continue
        if _is_density_matrix_regression_test(relative):
            item.add_marker(marker)
