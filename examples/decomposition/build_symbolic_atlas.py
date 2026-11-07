#!/usr/bin/env python3
"""Compile Qiskit/LaTeX atlas pages in parallel and assemble linked PDFs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def split_document(source):
    preamble, body = source.split(r"\begin{document}", 1)
    body = body.rsplit(r"\end{document}", 1)[0]
    pages = [page for page in body.split(r"\newpage") if page.strip()]
    destinations = {}
    for index, page in enumerate(pages):
        for label in re.findall(r"\\label\{((?:gate|clifford|expression-program)-[^}]+)\}", page):
            destinations[label] = index
    result = []
    for index, page in enumerate(pages):
        links = re.findall(r"\\hyperref\[([^]]+)\]", page)
        if any(link not in destinations for link in links):
            raise ValueError("An index target is missing from the atlas")
        page = re.sub(r"\\hyperref\[([^]]+)\]", lambda match: r"\hyperlink{"+match[1]+"}", page)
        page = re.sub(r"\\pageref\{([^}]+)\}", lambda match: str(destinations[match[1]]+1), page)
        # Named index links are resolved explicitly after assembly. Each page
        # is compiled once; no multi-pass full-book cross-reference run needed.
        document = preamble+r"\begin{document}\setcounter{page}{"+str(index+1)+"}"+page+r"\end{document}"
        result.append(dict(source=document, links=links,
                           bookmarks=re.findall(r"\\pdfbookmark\[([0-9]+)\]\{([^{}]+)\}\{[^}]+\}", page)))
    return result, destinations


def compile_page(job):
    compiler, root, volume, index, page, timeout = job
    directory = root / (volume+"-pages")
    directory.mkdir(exist_ok=True)
    source = directory / f"page-{index:04d}.tex"
    output = source.with_suffix(".pdf")
    digest = hashlib.sha256(page["source"].encode()).hexdigest()
    stamp = source.with_suffix(".sha256")
    if output.exists() and stamp.exists() and stamp.read_text() == digest:
        return index, output
    source.write_text(page["source"])
    command = [compiler, "--reruns", "0", "--keep-logs", "--outdir", str(directory), str(source)]
    process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)
    source.with_suffix(".console.log").write_text(process.stdout+process.stderr)
    if process.returncode and "TeX capacity exceeded" in process.stdout+process.stderr:
        # A dense 32-panel sheet can exceed XeTeX's fixed node pool. Compile
        # its Qcircuit fragments independently, retaining vector text/lines,
        # then compose those small PDFs using the same layout and font setup.
        source.write_text(vectorize_circuits(page["source"], compiler, root, directory, timeout))
        process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)
        with source.with_suffix(".console.log").open("a") as log:
            log.write("\nVector-fragment fallback:\n"+process.stdout+process.stderr)
    if process.returncode and "Running xdvipdfmx" in process.stdout and not process.stderr.strip():
        # A failed backend process can leave no diagnostic after a successful
        # TeX pass. Retry the same prepared sheet once; never cache a failure.
        process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)
        with source.with_suffix(".console.log").open("a") as log:
            log.write("\nPDF backend retry:\n"+process.stdout+process.stderr)
    if process.returncode:
        raise RuntimeError(f"Failed to compile {volume} page {index+1} (exit {process.returncode}): "+
                           (process.stdout+process.stderr)[-1200:])
    stamp.write_text(digest)
    return index, output


def vectorize_circuits(document, compiler, root, directory, timeout):
    fragments = sorted(set(re.findall(r"\\input\{(circuits/[^}]+\.tex)\}", document)))
    preamble = document.split(r"\begin{document}", 1)[0]
    preamble = re.sub(r"\\documentclass(?:\[[^]]*\])?\{article\}",
                      lambda _: r"\documentclass[border=1pt]{standalone}", preamble)
    preamble = re.sub(r"\\usepackage\[[^]]*\]\{geometry\}", "", preamble)
    tile_relative = "vector-circuits/"+hashlib.sha256(document.encode()).hexdigest()[:16]
    tile_directory = directory / tile_relative
    tile_directory.mkdir(parents=True, exist_ok=True)

    def render(fragment):
        body = (root / fragment).read_text()
        source = preamble+r"\begin{document}"+body+r"\end{document}"
        digest = hashlib.sha256(source.encode()).hexdigest()
        target = tile_directory / (digest+".pdf")
        if not target.exists():
            tex = target.with_suffix(".tex")
            tex.write_text(source)
            process = subprocess.run([compiler, "--reruns", "0", "--outdir", str(tile_directory), str(tex)],
                                     cwd=root, capture_output=True, text=True, timeout=timeout)
            tex.with_suffix(".console.log").write_text(process.stdout+process.stderr)
            if process.returncode:
                raise RuntimeError(f"Failed to compile vector circuit {fragment}: {process.stderr[-1200:]}")
        return fragment, tile_relative+"/"+target.name

    # Keep nested compilation bounded even when multiple dense sheets fail.
    with ThreadPoolExecutor(max_workers=4) as pool:
        replacements = dict(pool.map(render, fragments))
    return re.sub(r"\\input\{(circuits/[^}]+\.tex)\}",
                  lambda match: r"\includegraphics{"+replacements[match[1]]+"}", document)


def assemble(paths, pages, destinations, output):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ArrayObject, NameObject
    writer = PdfWriter()
    for index,path in enumerate(paths):
        reader = PdfReader(path)
        if len(reader.pages) != 1:
            raise ValueError(f"Expected exactly one sheet in {path}, got {len(reader.pages)}")
        if pages[index]["links"]:
            # pypdf omits links to destinations outside the input document.
            # Keep their rectangles through import with a valid local target,
            # then replace that target with the final merged page below.
            for item in reader.pages[0].get("/Annots",[]):
                annotation=item.get_object()
                if annotation.get("/Subtype")=="/Link":
                    annotation.pop(NameObject("/A"),None)
                    annotation[NameObject("/Dest")]=ArrayObject([reader.pages[0].indirect_reference,NameObject("/Fit")])
        writer.append(reader, import_outline=False)
    parents = {}
    link_count = 0
    for index, page in enumerate(pages):
        if page["links"]:
            annotations = writer.pages[index].get("/Annots", [])
            annotations = [item.get_object() for item in annotations
                           if item.get_object().get("/Subtype") == "/Link"]
            if len(annotations) != len(page["links"]):
                raise ValueError(f"Index link count differs on page {index+1}")
            for annotation, label in zip(annotations, page["links"]):
                annotation.pop(NameObject("/A"), None)
                target = writer.pages[destinations[label]].indirect_reference
                annotation[NameObject("/Dest")] = ArrayObject([target, NameObject("/Fit")])
                link_count += 1
        for level, title in page["bookmarks"]:
            level = int(level)
            title = title.replace(r"\_", "_")
            parents[level] = writer.add_outline_item(title, index, parent=parents.get(level-1))
    title="Symbolic gate atlas" if output.stem=="gates" else "All-phase Clifford atlas" if output.stem=="cliffords" else output.stem.replace("-"," ")
    writer.add_metadata({"/Title": title,
                         "/Author": "SQUANDER symbolic decomposition project"})
    writer.compress_identical_objects(remove_duplicates=True,remove_unreferenced=True)
    with output.open("wb") as stream:
        writer.write(stream)
    return dict(pages=len(paths), repaired_index_links=link_count,
                pdf_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    compiler=shutil.which("tectonic")
    local_compiler=Path.home()/".local/bin/tectonic"
    if compiler is None and local_compiler.is_file():compiler=str(local_compiler)
    parser.add_argument("--compiler", default=compiler)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--volume", choices=["gates", "cliffords"], action="append")
    parser.add_argument("--seconds-per-page", type=float, default=180)
    args = parser.parse_args()
    if not args.compiler or args.workers < 1 or args.seconds_per_page <= 0:
        parser.error("An available Tectonic compiler, positive worker count and timeout are required")
    root = args.source_dir.resolve()
    report = {}
    for volume in args.volume or ["gates", "cliffords"]:
        directory=root/(volume+"-pages")
        directory.mkdir(exist_ok=True)
        circuits=directory/"circuits"
        if not circuits.exists():circuits.symlink_to(root/"circuits",target_is_directory=True)
        pages, destinations = split_document((root/(volume+".tex")).read_text())
        paths = [None]*len(pages)
        jobs = [(args.compiler, root, volume, index, page, args.seconds_per_page)
                for index, page in enumerate(pages)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(compile_page, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                index, path = future.result()
                paths[index] = path
                if completed % 25 == 0:
                    print(f"Compiled {volume}: {completed}/{len(pages)} sheets", flush=True)
        report[volume] = assemble(paths, pages, destinations, root/(volume+".pdf"))
        print(json.dumps({volume: report[volume]}, sort_keys=True), flush=True)
    (root/"build-report.json").write_text(json.dumps(report, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
