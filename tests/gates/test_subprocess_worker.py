"""Check result isolation and recovery after a child process terminates."""

import pytest

from tests.gates._subprocess_worker import GateSubprocess


@pytest.fixture
def worker():
    process = GateSubprocess()
    try:
        yield process
    finally:
        process.close()


def test_worker_reuses_process_with_fresh_test_namespace(worker):
    first = worker.run("import os; marker = True; print(os.getpid())")
    second = worker.run("import os; print(os.getpid()); print('marker' in globals())")
    assert first.returncode == second.returncode == 0
    assert first.stdout.strip() == second.stdout.splitlines()[0]
    assert second.stdout.splitlines()[1] == "False"


def test_worker_reports_exception_and_remains_usable(worker):
    failed = worker.run("raise ValueError('gate failure')")
    assert failed.returncode == 1
    assert "ValueError: gate failure" in failed.stderr
    assert worker.run("print('recovered')").stdout.strip() == "recovered"


def test_worker_reports_process_exit_and_restarts(worker):
    failed = worker.run("import os; os._exit(23)")
    assert failed.returncode == 23
    recovered = worker.run("print('recovered')")
    assert recovered.returncode == 0
    assert recovered.stdout.strip() == "recovered"
