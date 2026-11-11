"""Exercise post-link wheel mutations and the real wheel-pack repair command."""

import base64
import csv
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/check_wheel_records.py"
spec = importlib.util.spec_from_file_location("wheel_records", SCRIPT)
records = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records)
DIST = "sample-1.0.dist-info"
RECORD = DIST + "/RECORD"
NATIVE = "sample/gates/native.so"


def payloads():
    return {
        NATIVE: b"old binary",
        DIST + "/METADATA": b"Metadata-Version: 2.1\nName: sample\nVersion: 1.0\n",
        DIST + "/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n",
    }


def record_rows(files):
    return [[name, "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()).rstrip(b"=").decode(), str(len(data))]
        for name, data in files.items()] + [[RECORD, "", ""]]


def write_wheel(path, files, rows=None):
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(record_rows(files) if rows is None else rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(RECORD, stream.getvalue().encode())
    return path


def test_valid_record_with_utf8_and_comma_in_filename(tmp_path):
    files = {**payloads(), "sample/données,example.txt": b"example"}
    wheel = write_wheel(tmp_path / "sample-1.0-py3-none-any.whl", files)
    assert records.check_wheel(wheel) == len(files)


@pytest.mark.parametrize("mutation, message", [
    ("relink", "hash mismatch"),
    ("bundle", "file missing from RECORD"),
    ("remove", "RECORD lists absent file"),
])
def test_post_build_file_changes_fail(tmp_path, mutation, message):
    files = payloads()
    rows = record_rows(files)
    if mutation == "relink":
        files[NATIVE] = b"new binary"  # Same length: a size-only check misses it.
    elif mutation == "bundle":
        files["sample/.dylibs/libdependency.dylib"] = b"bundled library"
    else:
        del files[NATIVE]
    wheel = write_wheel(tmp_path / "sample-1.0-py3-none-any.whl", files, rows)
    with pytest.raises(records.WheelRecordError, match=message):
        records.check_wheel(wheel)


@pytest.mark.parametrize("mutation, message", [
    ("size", "size mismatch"),
    ("no_size", "missing or invalid size"),
    ("no_hash", "missing or invalid hash"),
    ("duplicate", "duplicate RECORD entry"),
    ("self_hash", "own hash and size must be empty"),
    ("self_missing", "file missing from RECORD"),
    ("columns", "invalid RECORD row"),
    ("weak_hash", "SHA-256 or stronger"),
])
def test_invalid_record_entries_fail(tmp_path, mutation, message):
    files = payloads()
    rows = record_rows(files)
    if mutation == "size":
        rows[0][2] = "999"
    elif mutation == "no_size":
        rows[0][2] = ""
    elif mutation == "no_hash":
        rows[0][1] = ""
    elif mutation == "duplicate":
        rows.append(rows[0][:])
    elif mutation == "self_hash":
        rows[-1][1] = "sha256=invalid"
    elif mutation == "self_missing":
        rows.pop()
    elif mutation == "columns":
        rows[0].append("unexpected")
    else:
        rows[0][1] = "sha1=" + base64.urlsafe_b64encode(
            hashlib.sha1(files[NATIVE]).digest()).rstrip(b"=").decode()
    wheel = write_wheel(tmp_path / "sample-1.0-py3-none-any.whl", files, rows)
    with pytest.raises(records.WheelRecordError, match=message):
        records.check_wheel(wheel)


def test_missing_record_fails(tmp_path):
    wheel = tmp_path / "sample-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(NATIVE, b"binary")
    with pytest.raises(records.WheelRecordError, match="exactly one"):
        records.check_wheel(wheel)


def test_duplicate_archive_members_fail(tmp_path):
    wheel = write_wheel(tmp_path / "sample-1.0-py3-none-any.whl", payloads())
    with zipfile.ZipFile(wheel, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(NATIVE, b"duplicate")
    with pytest.raises(records.WheelRecordError, match="duplicate archive member"):
        records.check_wheel(wheel)


def test_real_wheel_pack_refreshes_relinked_files_and_new_libraries(tmp_path):
    files = payloads()
    old_rows = record_rows(files)
    files[NATIVE] = b"new binary"
    files["sample/.dylibs/libdependency.dylib"] = b"bundled library"
    stale = write_wheel(tmp_path / "sample-1.0-py3-none-any.whl", files, old_rows)
    with pytest.raises(records.WheelRecordError):
        records.check_wheel(stale)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(stale) as archive:
        archive.extractall(unpacked)
    output = tmp_path / "repaired"
    output.mkdir()
    subprocess.run([sys.executable, "-m", "wheel", "pack", str(unpacked),
                    "--dest-dir", str(output)], check=True, capture_output=True)
    repaired = output / stale.name
    assert records.check_wheel(repaired) == len(files)
    with zipfile.ZipFile(repaired) as archive:
        for name, data in files.items():
            assert archive.read(name) == data


def test_cli_glob_checks_all_wheels(tmp_path, capsys):
    files = payloads()
    write_wheel(tmp_path / "good/sample-1.0-py3-none-any.whl", files)
    stale_rows = record_rows(files)
    files[NATIVE] = b"new binary"
    write_wheel(tmp_path / "bad/sample-1.0-py3-none-any.whl", files, stale_rows)
    assert records.main([str(tmp_path / "*/*.whl")]) == 1
    output = capsys.readouterr()
    assert "PASS" in output.out
    assert "hash mismatch" in output.err


def test_cli_rejects_a_missing_wheel_glob(tmp_path):
    with pytest.raises(SystemExit) as error:
        records.main([str(tmp_path / "*.whl")])
    assert error.value.code == 2


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("unzip") is None,
                    reason="The macOS repair step uses Bash and unzip")
def test_macos_workflow_repack_updates_record_after_binary_repair(tmp_path):
    # Execute the actual workflow body. Stand in only for macOS relinking and
    # library discovery; wheel pack and the final integrity check are real.
    files = payloads()
    files[DIST + "/WHEEL"] = files[DIST + "/WHEEL"].replace(
        b"py3-none-any", b"cp310-cp310-macosx_26_0_arm64")
    files["sample/libqgd.dylib"] = b"core library"
    wheel = write_wheel(tmp_path / "dist/sample-1.0-cp310-cp310-macosx_26_0_arm64.whl", files)
    script = tmp_path / ".github/scripts/check_wheel_records.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    tools = tmp_path / "tools"
    tools.mkdir()
    commands = {
        "otool": 'print("@rpath/libqgd.dylib")',
        "install_name_tool": (
            'from pathlib import Path; import sys; p=Path(sys.argv[-1]); '
            'p.write_bytes(p.read_bytes()+b"_relinked")'),
        "delocate-path": (
            'from pathlib import Path; import sys; '
            'p=Path(sys.argv[1])/"sample/.dylibs/dependency.dylib"; '
            'p.parent.mkdir(parents=True); p.write_bytes(b"bundled dependency")'),
    }
    for name, body in commands.items():
        command = tools / name
        command.write_text(f"#!{sys.executable}\n{body}\n")
        command.chmod(0o755)
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    step = workflow.split("      - name: Repair wheel (macOS)\n", 1)[1]
    step = step.split("\n      - name:", 1)[0]
    body = textwrap.dedent(step.split("        run: |\n", 1)[1])
    env = dict(os.environ, PATH=os.pathsep.join(
        [str(tools), str(Path(sys.executable).parent), os.environ["PATH"]]))
    result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", body],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert records.check_wheel(wheel) == len(files) + 1
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(NATIVE) == b"old binary_relinked"
        assert archive.read("sample/.dylibs/dependency.dylib") == b"bundled dependency"
