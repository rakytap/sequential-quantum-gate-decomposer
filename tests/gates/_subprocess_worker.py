"""Reuse imports while keeping native gate checks outside the pytest process."""

import contextlib
import faulthandler
import io
import json
import subprocess
import sys
import traceback


_RESULT_PREFIX = "__SQUANDER_GATE_RESULT__ "


class GateSubprocess:
    def __init__(self):
        self.process = None

    def run(self, script):
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "tests.gates._subprocess_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self.process.stdin.write(json.dumps(script) + "\n")
        self.process.stdin.flush()

        output = []
        for line in self.process.stdout:
            if line.startswith(_RESULT_PREFIX):
                result = json.loads(line[len(_RESULT_PREFIX):])
                return subprocess.CompletedProcess(
                    [sys.executable, "-c", script],
                    result["returncode"],
                    "".join(output) + result["stdout"],
                    result["stderr"],
                )
            output.append(line)
        return subprocess.CompletedProcess(
            [sys.executable, "-c", script],
            self.process.wait() or 1,
            "".join(output),
            "Gate worker exited before returning a result.",
        )

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.process = None


def _serve():
    faulthandler.enable()
    for request in sys.stdin:
        stdout = io.StringIO()
        stderr = io.StringIO()
        returncode = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                # A fresh namespace prevents one gate check's variables from
                # leaking into the next; imported modules remain cached.
                exec(compile(json.loads(request), "<gate-test>", "exec"),
                     {"__name__": "__main__"})
            except BaseException:
                returncode = 1
                traceback.print_exc()
        response = {
            "returncode": returncode,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
        }
        print(_RESULT_PREFIX + json.dumps(response), flush=True)


if __name__ == "__main__":
    _serve()
