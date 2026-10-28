"""Exercise debug filenames with allocators that expose stale Python buffers."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("allocator", ["debug", "malloc"])
def test_debugfile_owns_filename_and_propagates_conversion_errors(allocator):
    code = textwrap.dedent(
        """
        import numpy as np
        from squander.decomposition.qgd_N_Qubit_Decompositions_Wrapper import (
            qgd_N_Qubit_Decomposition_custom,
        )

        decomposition = qgd_N_Qubit_Decomposition_custom(
            np.eye(2, dtype=np.complex128), qbit_num=1
        )
        for size in (0, 1, 31, 512, 8192):
            decomposition.set_Debugfile("debug_" + "x" * size)
        decomposition.set_Debugfile("debug_" + chr(0x3BB))
        decomposition.set_Debugfile("debug_" + chr(0) + "tail")

        class Filename:
            def __str__(self):
                return "temporary_" + "x" * 512

        decomposition.set_Debugfile(Filename())

        class InvalidFilename:
            def __str__(self):
                raise ValueError("filename conversion failed")

        try:
            decomposition.set_Debugfile(InvalidFilename())
        except ValueError as error:
            assert str(error) == "filename conversion failed"
        else:
            raise AssertionError("filename conversion error was swallowed")

        try:
            decomposition.set_Debugfile(chr(0xD800))
        except UnicodeEncodeError:
            pass
        else:
            raise AssertionError("invalid UTF-8 filename was accepted")

        decomposition.set_Debugfile("usable_after_conversion_errors")
        """
    )
    env = os.environ.copy()
    env["PYTHONMALLOC"] = allocator
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

