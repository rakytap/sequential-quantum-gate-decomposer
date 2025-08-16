from qiskit import QuantumCircuit
from squander.partitioning.tools import qiskit_to_squander_name
import os
import tempfile
import requests
import zipfile
import shutil
from pathlib import Path

from squander.gates import gates_Wrapper as gate
SUPPORTED_GATES = {x for n in dir(gate) for x in (getattr(gate, n),) if not n.startswith("_") and issubclass(x, gate.Gate) and n != "Gate"}
SUPPORTED_GATES_NAMES = {n for n in dir(gate) if not n.startswith("_") and issubclass(getattr(gate, n), gate.Gate) and n != "Gate"}

def download_and_collect_qasm(repo_urls):
    """
    Downloads GitHub repositories as zip archives, extracts them into a temporary
    directory, and returns the list of all .qasm file paths found, along with
    the temp folder path (for later cleanup).

    Args:
        repo_urls (list[str]): List of GitHub repository URLs.

    Returns:
        (list[str], str): List of QASM file paths and the temp directory path.
    """
    temp_dir = tempfile.mkdtemp(prefix="repos_")
    qasm_files = []

    for url in repo_urls:
        # Normalize repo URL -> owner/repo
        parts = url.rstrip("/").replace(".git", "").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid repo URL: {url}")
        owner, repo = parts[-2], parts[-1]

        # GitHub zip URL (main branch assumed; you can change to 'master' if needed)
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"

        zip_path = os.path.join(temp_dir, f"{repo}.zip")

        # Download zip
        r = requests.get(zip_url, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download {zip_url}: HTTP {r.status_code}")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract zip
        extract_path = os.path.join(temp_dir, repo)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

        # Find QASM files
        for path in Path(extract_path).rglob("*.qasm"):
            qasm_files.append(str(path.resolve()))

    return qasm_files, temp_dir


def cleanup_repo(temp_dir):
    """
    Deletes the temporary repository folder created by download_and_collect_qasm.
    """
    shutil.rmtree(temp_dir, ignore_errors=True)

def squander_gate_support_check():
    #Veri-Q/Benchmark Missing gates: {'CCX': 172, 'CU1': 297, 'MEASURE': 232, 'BARRIER': 16, 'SWAP': 34, 'SDG': 19, 'MCX': 30, 'MCX_GRAY': 5, 'IF_ELSE': 203, 'SXDG': 19}
    #pnnl/QASMBench Missing gates: {'MEASURE': 238, 'BARRIER': 118, 'CCX': 26, 'RESET': 19, 'SDG': 6, 'IF_ELSE': 18, 'CRZ': 2, 'CSWAP': 18, 'RZZ': 2, 'RYY': 6, 'ADD4': 1, 'SWAP': 3, 'P': 1, 'CP': 1, 'CU1': 4, 'UNMAJ': 1, 'MAJORITY': 1, 'ID': 1, 'CTU': 2, 'SYNDROME': 1, 'RYY_<value>': 637}
    for repo in ["https://github.com/iic-jku/ibm_qx_mapping", "https://github.com/Veri-Q/Benchmark", "https://github.com/pnnl/QASMBench", "https://github.com/QML-Group/qbench"]:
        qasm, temp = download_and_collect_qasm([repo])
        #print(qasm, temp)
        bad_files, missing_gates = [], {}
        for filename in qasm:
            try:
                qc = QuantumCircuit.from_qasm_file(filename)
            except Exception as e:
                bad_files.append(filename)
                #print(e)
                continue
            qc_gates_names = {qiskit_to_squander_name(inst.operation.name) for inst in qc.data}
            if not qc_gates_names.issubset(SUPPORTED_GATES_NAMES):
                for x in qc_gates_names-SUPPORTED_GATES_NAMES:
                    if not x in missing_gates: missing_gates[x] = 0
                    missing_gates[x] += 1
                #print(f"Filename: {filename.replace(temp, '')} Unsupported gates: {qc_gates_names-SUPPORTED_GATES_NAMES}")
        print(repo, "Missing gates:", missing_gates)
        cleanup_repo(temp)

if __name__ == "__main__":
    squander_gate_support_check()
