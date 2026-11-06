"""Exercise the pure symbolic modules without a compiled SQUANDER backend."""

from pathlib import Path
import sys

import pytest

pytest.importorskip("sympy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "squander/decomposition"))
