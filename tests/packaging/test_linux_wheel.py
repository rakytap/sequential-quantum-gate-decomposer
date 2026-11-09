"""Exercise installed native-library lookup and manylinux failure handling."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux ELF packaging")


def run(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def pack(stage, output):
    metadata = stage / "native_rpath_probe-1.0.dist-info"
    metadata.mkdir(exist_ok=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: native-rpath-probe\nVersion: 1.0\n")
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    (metadata / "WHEEL").write_text(
        f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}-{tag}-{platform}\n")
    output.mkdir()
    run([sys.executable, "-m", "wheel", "pack", str(stage), "-d", str(output)])
    return next(output.glob("*.whl"))


def probe_module(stage, env):
    module = next((stage / "squander/density_matrix").glob("_density_matrix_cpp*.so"))
    return subprocess.run([
        sys.executable, "-c",
        "import importlib.util, sys; "
        "s=importlib.util.spec_from_file_location('_density_matrix_cpp',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "assert m.probe()==42", str(module),
    ], cwd=stage, env=env, text=True, capture_output=True)


def test_density_matrix_install_and_auditwheel_find_bundled_core(tmp_path):
    for module in ("cmake", "pybind11", "auditwheel", "wheel"):
        pytest.importorskip(module)
    if not shutil.which("patchelf") or not shutil.which("c++"):
        pytest.skip("Native regression requires a C++ compiler and patchelf")
    import pybind11

    source = tmp_path / "source"
    density = source / "squander/src-cpp/density_matrix"
    (density / "include").mkdir(parents=True)
    # Use the production target/install rules with tiny sources: this tests
    # the wheel's library layout without recompiling the synthesis engine.
    shutil.copyfile(ROOT / "squander/src-cpp/density_matrix/CMakeLists.txt",
                    density / "CMakeLists.txt")
    for name in ("density_matrix", "noisy_circuit", "noise_channel",
                 "gate_operation", "noise_operation"):
        (density / f"{name}.cpp").write_text("// Packaging probe\n")
        (density / f"include/{name}.h").touch()
    (density / "include/density_operation.h").touch()
    bindings = source / "squander/density_matrix"
    bindings.mkdir(parents=True)
    (bindings / "bindings.cpp").write_text(
        '#include <pybind11/pybind11.h>\n'
        'extern "C" int qgd_packaging_probe();\n'
        'PYBIND11_MODULE(_density_matrix_cpp, m) {\n'
        '  m.def("probe", &qgd_packaging_probe);\n}\n')
    (source / "qgd.cpp").write_text('extern "C" int qgd_packaging_probe() { return 42; }\n')
    (source / "CMakeLists.txt").write_text(textwrap.dedent("""\
        cmake_minimum_required(VERSION 3.15)
        project(native_rpath_probe LANGUAGES CXX)
        add_library(qgd SHARED qgd.cpp)
        set_target_properties(qgd PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/squander)
        add_library(squander_common INTERFACE)
        add_subdirectory(squander/src-cpp/density_matrix)
        install(TARGETS qgd LIBRARY DESTINATION squander)
        """))
    build, stage = tmp_path / "build", tmp_path / "installed"
    run([sys.executable, "-m", "cmake", "-S", str(source), "-B", str(build),
         f"-DCMAKE_INSTALL_PREFIX={stage}", f"-DPYTHON_EXECUTABLE={sys.executable}",
         f"-Dpybind11_DIR={pybind11.get_cmake_dir()}", "-DCMAKE_BUILD_TYPE=Release"])
    run([sys.executable, "-m", "cmake", "--build", str(build), "--target", "install", "-j2"])
    env = {key: value for key, value in os.environ.items()
           if key not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
    loaded = probe_module(stage, env)
    assert loaded.returncode == 0, loaded.stderr

    # Removing only the installed RPATH reproduces the reported auditwheel
    # failure even though libqgd.so is present inside the same wheel.
    broken = tmp_path / "broken"
    shutil.copytree(stage, broken)
    module = next((broken / "squander/density_matrix").glob("*.so"))
    run(["patchelf", "--remove-rpath", str(module)])
    raw_broken = pack(broken, tmp_path / "broken-dist")
    failure = subprocess.run([
        sys.executable, "-m", "auditwheel", "repair", str(raw_broken),
        "-w", str(tmp_path / "broken-repaired"),
    ], env=env, text=True, capture_output=True)
    assert failure.returncode != 0
    assert 'required library "libqgd.so" could not be located' in failure.stderr

    wheel = pack(stage, tmp_path / "dist")
    output = tmp_path / "repaired"
    run([sys.executable, "-m", "auditwheel", "repair", str(wheel), "-w", str(output)], env=env)
    repaired = next(output.glob("*manylinux*.whl"))
    run([sys.executable, str(ROOT / ".github/scripts/check_wheel_records.py"), str(repaired)])
    relocated = tmp_path / "relocated"
    with zipfile.ZipFile(repaired) as archive:
        assert "squander/libqgd.so" in archive.namelist()
        assert not any(name.startswith("native_rpath_probe.libs/libqgd")
                       for name in archive.namelist())
        archive.extractall(relocated)
    loaded = probe_module(relocated, env)
    assert loaded.returncode == 0, loaded.stderr


@pytest.mark.parametrize("failure_step", ["build", "repair", "record"])
def test_manylinux_failure_preserves_original_wheel(tmp_path, failure_step):
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    job = workflow.split("  build-manylinux-wheels:\n", 1)[1]
    body = job.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
    body = textwrap.dedent(body).replace("${{ github.workspace }}", str(tmp_path))
    body = body.replace("${{ matrix.image }}", "manylinux_2_28_x86_64")
    body = body.replace("${{ matrix.python-version }}", "3.13")
    tools = tmp_path / "tools"
    tools.mkdir()
    commands = {
        "docker": (
            'import os, subprocess, sys; '
            'args=sys.argv[sys.argv.index("quay.io/pypa/manylinux_2_28_x86_64")+1:]; '
            'args[-1]=args[-1].replace("/opt/python/cp${PYTHON_TAG}-cp${PYTHON_TAG}/bin", '
            'os.environ["PROBE_TOOLS"]).replace("/io", os.getcwd()); '
            'sys.exit(subprocess.call(args))'),
        "yum": "pass",
        "pip": "pass",
        "find": 'import sys; print("/usr/lib64/libtbb.so" if "libtbb" in " ".join(sys.argv) else "/usr/include/tbb.h")',
        "python": (
            'import os, sys; '
            'step="build" if sys.argv[1:3]==["-m","build"] else "record"; '
            'failed=step==os.environ["PROBE_FAILURE"]; '
            'print(f"simulated {step} failure" if failed else step, file=sys.stderr); '
            'sys.exit(23 if failed else 0)'),
        "auditwheel": (
            'import os, sys; from pathlib import Path; '
            'failed=os.environ["PROBE_FAILURE"]=="repair"; '
            'print("simulated repair failure" if failed else "repair",file=sys.stderr); '
            'p=Path("wheelhouse"); p.mkdir(); '
            '(p/"repaired.whl").write_bytes(b"repaired artifact"); '
            'sys.exit(23 if failed else 0)'),
    }
    for name, code in commands.items():
        tool = tools / name
        tool.write_text(f"#!{sys.executable}\n{code}\n")
        tool.chmod(0o755)
    original = tmp_path / "dist/squander-1.9.8-cp313-cp313-linux_x86_64.whl"
    original.parent.mkdir()
    original.write_bytes(b"original artifact")
    env = dict(os.environ, PROBE_TOOLS=str(tools), PROBE_FAILURE=failure_step,
               PATH=str(tools) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", "-e", "-c", body], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 23, result.stdout + result.stderr
    assert f"simulated {failure_step} failure" in result.stderr
    assert original.read_bytes() == b"original artifact"
