"""Small CI diagnostics shared by the test suite."""

import os
import time


_ci_terminal = None
_ci_test_starts = {}


def pytest_configure(config):
    """Enable a live per-test clock only on CI runners.

    A job-level timeout kills pytest before its normal duration summary is
    produced.  Flushed start/finish records leave the active test visible in
    the log even when GitHub terminates the process.
    """
    global _ci_terminal
    if os.environ.get("CI"):
        _ci_terminal = config.pluginmanager.get_plugin("terminalreporter")


def pytest_sessionstart(session):
    """Initialize after the tests-level conftest has certainly been loaded."""
    global _ci_terminal
    if os.environ.get("CI"):
        _ci_terminal = session.config.pluginmanager.get_plugin("terminalreporter")


def pytest_runtest_logstart(nodeid, location):
    if _ci_terminal is None:
        return
    _ci_test_starts[nodeid] = time.perf_counter()
    _ci_terminal.write_line(f"CI TEST START  {nodeid}", flush=True)


def pytest_runtest_logfinish(nodeid, location):
    if _ci_terminal is None:
        return
    started = _ci_test_starts.pop(nodeid, None)
    elapsed = 0.0 if started is None else time.perf_counter() - started
    _ci_terminal.write_line(
        f"CI TEST FINISH {nodeid} ({elapsed:.3f} s)",
        flush=True,
    )
